"""Compare fixed H1 trend strata with saved 5-minute probabilities, without fitting."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_slope_response_audit_20261028_001'
START, END = 47602, 64890
TRAIN_CUTOFF, LATER_CUTOFF = 1735689600000000000, 1743465600000000000
DAY_NS, MINUTE_NS = 86_400_000_000_000, 60 * 10**9
SOURCE_PATHS = (
    'gx1/scripts/audit_entry_direction_slope_response_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/scripts/compare_entry_direction_matched_exposure_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_entry_direction_slope_response_v1.py',
)


def require(ok, reason):
    if not bool(ok): raise RuntimeError('DIRECTION_SLOPE_RESPONSE_' + reason)


def sigmoid(values):
    z = np.asarray(values,dtype=np.float64);p=np.empty_like(z);positive=z>=0
    p[positive]=1/(1+np.exp(-z[positive]));ez=np.exp(z[~positive]);p[~positive]=ez/(1+ez)
    return p


def bootstrap_contrasts(labels, probabilities, slope, block_ids, weights, *, alpha=.05/6):
    """Contrasts of two fixed groups under paired duplicate-calendar-block draws."""
    require(labels.dtype==np.int8 and np.isin(labels,(0,1)).all() and labels.ndim==1
        and probabilities.dtype==np.float64 and probabilities.shape==labels.shape and np.isfinite(probabilities).all()
        and slope.dtype==np.int8 and slope.shape==labels.shape and np.isin(slope,(-1,0,1)).all()
        and block_ids.dtype.kind=='i' and block_ids.shape==labels.shape and weights.ndim==2
        and weights.shape[1]>0 and np.all(block_ids>=0) and np.all(block_ids<weights.shape[1]),'BOOTSTRAP_INPUT')
    values={}
    for group,code in (('negative',-1),('positive',1)):
        mask=slope==code; counts=np.bincount(block_ids[mask],minlength=weights.shape[1])
        actual=np.bincount(block_ids[mask],weights=labels[mask],minlength=weights.shape[1])
        predicted=np.bincount(block_ids[mask],weights=probabilities[mask],minlength=weights.shape[1])
        den=weights@counts;require(np.all(den>0),'BOOTSTRAP_EMPTY_GROUP')
        values[group]=(weights@actual/den,weights@predicted/den)
    observed=values['positive'][0]-values['negative'][0]
    predicted=values['positive'][1]-values['negative'][1]
    result={}
    for key,draw in (('observed_positive_minus_negative',observed),('predicted_positive_minus_negative',predicted),
                     ('predicted_minus_observed_contrast',predicted-observed)):
        low,high=np.quantile(draw,[alpha/2,1-alpha/2],method='linear')
        result[key]={'low':float(low),'high':float(high),'alpha':alpha,'replicates':len(draw)}
    return result


def analyze(candidate, baseline, *, cutoff, replicates=10000, minimum_blocks=8):
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    from gx1.contracts.entry_native_direction_context_report_v1 import _panel
    from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights
    n=len(candidate['parent_rows']);anchors=np.array([0,4,14],np.int64)
    for name in ('parent_rows','bar_start_ns','decision_ns'):
        require(candidate[name].dtype==baseline[name].dtype==np.int64
            and candidate[name].shape==baseline[name].shape==(n,) and np.array_equal(candidate[name],baseline[name]),'AXIS:'+name)
    require(n>0 and np.all(np.diff(candidate['parent_rows'])>0) and np.all(np.diff(candidate['decision_ns'])>0)
        and np.array_equal(candidate['decision_ns'],candidate['bar_start_ns']+5*MINUTE_NS),'CLOCK')
    for name,dtype in (('terminal_time_ns',np.int64),('observed',np.bool_),('sign',np.int8)):
        require(candidate[name].dtype==baseline[name].dtype==dtype and candidate[name].shape==(n,60)
            and baseline[name].shape==(n,3) and np.array_equal(candidate[name][:,anchors],baseline[name]),'ANCHOR:'+name)
    require(np.array_equal(candidate['terminal_time_ns'],candidate['decision_ns'][:,None]
        + np.arange(1,61,dtype=np.int64)[None,:]*MINUTE_NS)
        and np.isin(candidate['sign'],(-1,0,1)).all() and np.all(candidate['sign'][~candidate['observed']]==0),'TARGET_CLOCK_DOMAIN')
    name='context__h1_ema20_slope_atr_sign';slope=candidate[name]
    require(slope.dtype==baseline[name].dtype==np.int8 and slope.shape==baseline[name].shape==(n,)
        and np.array_equal(slope,baseline[name]) and np.isin(slope,(-1,0,1)).all(),'SLOPE_TAG')
    require(candidate['logit_up'].shape==(n,60) and baseline['logit_up'].shape==(n,3)
        and np.isfinite(candidate['logit_up']).all() and np.isfinite(baseline['logit_up']).all(),'LOGIT_DOMAIN')
    mature=candidate['terminal_time_ns'][:,4]<cutoff
    eligible=candidate['observed'][:,4] & (candidate['sign'][:,4]!=0) & mature
    labels=(candidate['sign'][:,4]==1).astype(np.int8)
    candidate_prob=sigmoid(candidate['logit_up'][:,4]);baseline_prob=sigmoid(baseline['logit_up'][:,1])
    groups=[]
    for label,code in (('negative',-1),('zero',0),('positive',1)):
        take=eligible & (slope==code);y=np.ascontiguousarray(labels[take]);z=np.ascontiguousarray(candidate['logit_up'][take,4],dtype=np.float64)
        rank=direction_rank_diagnostics(y,z,score_kind='logit_up')['auc']
        groups.append({'slope':label,'parents':int((slope==code).sum()),'eligible':int(take.sum()),
            'actual_up_fraction':float(y.mean()) if len(y) else None,'candidate_mean_up_probability':float(candidate_prob[take].mean()) if len(y) else None,
            'baseline_mean_up_probability':float(baseline_prob[take].mean()) if len(y) else None,'candidate_auc':rank})
    negative,positive=groups[0],groups[2]
    require(negative['eligible']>0 and positive['eligible']>0,'FIXED_GROUPS')
    observed=positive['actual_up_fraction']-negative['actual_up_fraction']
    predicted=positive['candidate_mean_up_probability']-negative['candidate_mean_up_probability']
    days=candidate['decision_ns']//DAY_NS;intervals=[]
    for scheme,raw,seed in (('UTC_DAY',days,20261027),('UTC_WEEK_MONDAY',(days+3)//7,20261028)):
        blocks,ids=np.unique(raw,return_inverse=True);require(len(blocks)>=minimum_blocks,'MIN_BLOCKS')
        weights=block_weights(len(blocks),seed=seed,np=np,replicates=replicates)
        intervals.append({'scheme':scheme,'blocks':len(blocks),'contrasts':bootstrap_contrasts(labels[eligible],
            candidate_prob[eligible],slope[eligible],ids[eligible],weights)})
    full=_panel(candidate['logit_up'][:,4],candidate['sign'][:,4],candidate['observed'][:,4],mature)
    base=_panel(baseline['logit_up'][:,1],baseline['sign'][:,1],baseline['observed'][:,1],mature)
    return {'groups':groups,'observed_positive_minus_negative':float(observed),
        'candidate_positive_minus_negative':float(predicted),'candidate_minus_observed_contrast':float(predicted-observed),
        'intervals':intervals,'candidate_full_panel':full,'baseline_full_panel':base,
        'scope':'Previously noticed H1 slope strata, one reused 2025 Q1 development fold; 5m fixed logits only.',
        'limits':'Exploratory posthoc groups, multiple dependent observations and assumed calendar-block exchangeability; conditional associations do not prove a feature is disconnected, an intervention works, causality or economic edge.',
        'test_data_used':False,'economic_edge_qualified':False}


def main(argv=None):
    sys.dont_write_bytecode=True
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','plan-sha256','policy-sha256','source-head'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args(argv)
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC,_panel
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched
    from gx1.scripts.compare_entry_direction_saved_bps_v1 import block_weights
    require(not any(name.split('.')[0] in ('torch','sklearn','joblib') for name in sys.modules),'PRELOADED_MODEL')
    initial=candle.git_state();require(initial==(args.source_head,'work/gx1-current',''),'INITIAL_SOURCE')
    plan_ref={'path':str(candle.regular(args.plan)),'sha256':args.plan_sha256}
    policy_ref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan=binding.read_bound_json(Path(plan_ref['path']),plan_ref['sha256'])
        require(set(plan)=={'schema_version','stage','design','inputs','sources','output_dir','wall_seconds','test_data_used'}
            and plan['schema_version']=='gx1_direction_slope_response_plan_v1' and plan['stage']==STAGE
            and plan['wall_seconds']==600 and plan['test_data_used'] is False,'PLAN')
        require(plan['design']['path']==str(REPO/'docs/DIRECTION_SLOPE_RESPONSE_AUDIT_20261028.json'),'DESIGN_PATH')
        design=binding.read_bound_json(Path(plan['design']['path']),plan['design']['sha256'])
        require(design['schema_version']=='gx1_direction_slope_response_design_v1'
            and design['inputs']==plan['inputs'] and design['test_data_used'] is False,'DESIGN')
        policy=binding.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']);gate=policy[STAGE]
        require(gate['status']=='ADMITTED' and gate['plan']==plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations']==1 and gate['claimed_invocations']==gate['completed_invocations']==0
            and gate['relaunch_allowed'] is False,'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled','native_training_enabled',
            'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')),'GLOBAL_GATES')
        require(set(plan['sources'])==set(SOURCE_PATHS)
            and all(ref['path']==str(REPO/name) for name,ref in plan['sources'].items()),'SOURCE_SET')
        for ref in plan['sources'].values():candle.regular(ref['path']);binding.require_binding(ref,label='source')
        require(set(plan['inputs'])=={'candidate_report','baseline_report','baseline_rank_review',
            'candidate_predictions','baseline_predictions'},'INPUT_SET')
        meta={}
        for key in ('candidate_report','baseline_report','baseline_rank_review'):
            ref=plan['inputs'][key];meta[key]=binding.read_bound_json(Path(ref['path']),ref['sha256'])
        cr,br,rr=(meta[k] for k in ('candidate_report','baseline_report','baseline_rank_review'))
        fold={'fold_id':1,'prefix_rows':START,'oof_start_row':START,'oof_end_row':END,
            'train_cutoff_ns':TRAIN_CUTOFF,'oof_cutoff_ns':LATER_CUTOFF}
        require(cr['predictions']==plan['inputs']['candidate_predictions'] and br['predictions']==plan['inputs']['baseline_predictions']
            and cr['fold']==br['fold']==fold and all(x['test_data_used'] is False for x in (cr,br,rr)),'REPORT_BINDINGS')
        reference=next(p for p in rr['panels'] if p['horizon_minutes']==5)['arms']['mixed']
        require(rr['original_scores_reproduced'] is True,'BASELINE_RANK_REFERENCE')
        control_snapshot=binding.capture_verified_file_snapshot()
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'CAPPED_LOCK_SCOPE')
    out=Path(plan['output_dir']);require(ROOT.resolve(strict=True)==ROOT and out.parent==ROOT
        and out.name=='DIRECTION_SLOPE_RESPONSE_AUDIT_20261028_001' and not out.exists() and not out.is_symlink(),'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path,value):publication._publish_bytes(path,binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json',{'plan':plan_ref,'policy':policy_ref,'source_head':args.source_head,'execution':proof,'lock':lock})
    payloads=[plan['inputs']['candidate_predictions'],plan['inputs']['baseline_predictions']]
    require(len({x['path'] for x in payloads})==2,'DISTINCT_PAYLOADS')
    metadata=[plan_ref['path'],policy_ref['path'],plan['design']['path']]+[plan['inputs'][k]['path'] for k in ('candidate_report','baseline_report','baseline_rank_review')]
    sys.addaudithook(candle.make_guard(out,[x['path'] for x in payloads],metadata,source_paths=[x['path'] for x in plan['sources'].values()]))
    started=time.monotonic();previous_handler=signal.getsignal(signal.SIGALRM);previous_profile=sys.getprofile()
    def expired(*unused):raise RuntimeError('DIRECTION_SLOPE_RESPONSE_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);sys.setprofile(candle.no_queries_or_models)
    stage=out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:candle.regular(ref['path']);binding.require_binding(ref,label='prediction')
            baseline=matched._read_prediction(plan['inputs']['baseline_predictions'],context_spec=CONTEXT_SPEC)
            with Path(plan['inputs']['candidate_predictions']['path']).open('rb') as handle:
                with zipfile.ZipFile(handle) as archive:require(sum(x.file_size for x in archive.infolist())<=128*1024**2,'CANDIDATE_ARCHIVE_SIZE')
                handle.seek(0)
                with np.load(handle,allow_pickle=False) as archive:
                    names={'parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign','logit_up'}
                    require(set(archive.files)==names|{'context__'+name for name,_ in CONTEXT_SPEC},'CANDIDATE_MEMBERS')
                    names.add('context__h1_ema20_slope_atr_sign')
                    candidate={name:archive[name].copy() for name in names}
            require(np.array_equal(candidate['parent_rows'],np.arange(START,END,dtype=np.int64))
                and np.all(candidate['decision_ns']>=TRAIN_CUTOFF) and np.all(candidate['decision_ns']<LATER_CUTOFF),'EXACT_FOLD')
            result=analyze(candidate,baseline,cutoff=LATER_CUTOFF)
            for panel,report in ((result['candidate_full_panel'],cr),(result['baseline_full_panel'],br)):
                expected=report['model_metrics']['per_horizon'][4 if report is cr else 1]
                require(panel['counts']['eligible_cells']==expected['counts']['eligible_cells']
                    and all(np.isclose(panel[k],expected[k],rtol=0,atol=1e-12) for k in ('accuracy','logloss','brier')),'SCORE_REPLAY')
            mature=candidate['terminal_time_ns'][:,4]<LATER_CUTOFF
            eligible=candidate['observed'][:,4] & (candidate['sign'][:,4]!=0) & mature
            labels=np.ascontiguousarray((candidate['sign'][eligible,4]==1).astype(np.int8))
            for z,expected in ((candidate['logit_up'][eligible,4],cr['model_metrics']['per_horizon'][4]['rank']['auc']),
                               (baseline['logit_up'][eligible,1],reference['auc'])):
                got=direction_rank_diagnostics(labels,np.ascontiguousarray(z,dtype=np.float64),score_kind='logit_up')['auc']
                require(np.isclose(got,expected,rtol=0,atol=1e-12),'RANK_REPLAY')
            input_snapshot=binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged();control_snapshot.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock,'FINAL_SOURCE_LOCK')
        result.update(schema_version='gx1_direction_slope_response_result_v1',status='COMPLETE',plan=plan_ref,
            source_head=args.source_head,original_scores_reproduced=True,new_model_forwards=0,optimizer_steps=0)
        publish(stage/'RESULT.json',result)
        publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,
            'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT')
        publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False})
        raise
    finally:
        sys.setprofile(previous_profile);signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)


if __name__=='__main__':raise SystemExit(main())
