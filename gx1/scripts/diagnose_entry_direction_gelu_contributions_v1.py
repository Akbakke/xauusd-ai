"""Read-only diagnostic of GELU-region contributions in saved native276 representations.

The threshold has an analytical definition and is not an Entry decision rule.
No native model, fit, raw data, checkpoint or CONTROL/TEST access is authorized.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import types

REPO = Path('/home/andre2/src/GX1_CURRENT')
PROGRAM = Path('/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2')
HELPER_SHA = 'a648c47880b6b195d1213741a0826d34c7e475bd75193313a4c26f7723798c7a'
STAGE = 'direction_gelu_contribution_diagnostic_20261027'
DAY = 86_400_000_000_000
START = 1714521600000000000
PREFIX_END = 1735689600000000000
LATER_END = 1743465600000000000
PREFIX_ROWS, LATER_ROWS, WIDTH = 47602, 17288, 128
REPLICATES, ALPHA, SEED = 4000, .01, 20260911
BASE = {'parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign','logit_up'}
EXTRA = {'hidden','head_weight','head_bias','presentation_index'}
np = None


def require(condition, reason):
    if not condition:
        raise RuntimeError('GELU_CONTRIBUTION_' + reason)


def cutoff():
    # Ideal exact GELU(-0.5). For x >= -0.5 GELU is strictly increasing.
    return -.25 * (1. + math.erf(-.5 / math.sqrt(2.)))


def decompose(hidden, weight, logits):
    require(hidden.ndim == 2 and hidden.shape[1] == WIDTH and weight.shape == (3,WIDTH)
            and logits.shape == (len(hidden),3)
            and all(np.isfinite(x).all() for x in (hidden,weight,logits)), 'FINITE_SHAPES')
    h = hidden.astype(np.float64)
    w = weight[1].astype(np.float64)
    selected = h < cutoff()
    contribution = np.sum(np.where(selected, h, 0.) * w[None,:], axis=1)
    original = logits[:,1].astype(np.float64)
    remainder = original - contribution
    require(np.isfinite(contribution).all() and np.isfinite(remainder).all(), 'FINITE_CONTRIBUTIONS')
    return selected, contribution, remainder


def describe(hidden, weight, logits):
    selected, contribution, remainder = decompose(hidden,weight,logits)
    square = hidden.astype(np.float64)**2
    total_energy = float(square.sum())
    n = len(hidden)
    return {'parents':n, 'coordinates_per_parent':WIDTH,
        'guaranteed_low_slope_coordinates':int(selected.sum()),
        'guaranteed_low_slope_coordinate_fraction':float(selected.mean()),
        'guaranteed_low_slope_output_energy_fraction':float(square[selected].sum()/total_energy) if total_energy else None,
        'negative_output_fraction':float((hidden<0).mean()),
        'zero_output_fraction':float((hidden==0).mean()),
        'mean_low_slope_coordinates_per_parent':float(selected.sum(axis=1).mean()),
        'h5_low_slope_contribution_mean':float(contribution.mean()),
        'h5_low_slope_contribution_std':float(contribution.std()),
        'h5_remainder_std':float(remainder.std()),
        'h5_original_logit_std':float(logits[:,1].astype(np.float64).std())}


def score_later(later, probability, block_rank, paired, io):
    mask = later['observed'][:,1] & (later['sign'][:,1] != 0) & (later['terminal_time_ns'][:,1] < LATER_END)
    y = np.ascontiguousarray((later['sign'][mask,1] == 1).astype(np.int8))
    require(len(y)>0 and (y==0).any() and (y==1).any(), 'BOTH_CLASSES')
    _, contribution, remainder = decompose(later['hidden'],later['head_weight'],later['logit_up'])
    original = np.ascontiguousarray(later['logit_up'][mask,1],dtype=np.float64)
    low = np.ascontiguousarray(contribution[mask],dtype=np.float64)
    rest = np.ascontiguousarray(remainder[mask],dtype=np.float64)
    def point(z):
        d=probability.direction_rank_diagnostics(y,z,score_kind='logit_up')
        return {'auc':d['auc'],'up_down_pairs':d['up_down_pairs'],
                'distinct_scores':d['distinct_scores']}
    points={'original':point(original),'without_guaranteed_low_slope_contributions':point(rest),
            'guaranteed_low_slope_contribution_only':point(low)}
    intervals=[]
    days=later['decision_ns']//DAY
    for sid,scheme in enumerate(('UTC_WEEK_MONDAY','UTC_DAY')):
        labels=(days+3)//7 if sid==0 else days
        blocks,inverse=np.unique(labels,return_inverse=True)
        require(len(blocks)>=8,'MINIMUM_BLOCKS')
        weights=io.block_weights(int(len(blocks)),replicates=REPLICATES,
                                 seed=np.random.SeedSequence([SEED,sid,1]))
        original_draws,op=paired.auc_draws(y,original,inverse[mask],weights,block_rank)
        rest_draws,rp=paired.auc_draws(y,rest,inverse[mask],weights,block_rank)
        low_draws,lp=paired.auc_draws(y,low,inverse[mask],weights,block_rank)
        require(np.array_equal(op,rp) and np.array_equal(op,lp),'PAIRED_DENOMINATORS')
        intervals.append({'scheme':scheme,'blocks':len(blocks),
            'weights_sha256':io.digest(weights,np),
            'original_minus_remainder_auc_99_descriptive':paired.interval(original_draws-rest_draws,
                point=points['original']['auc']-points['without_guaranteed_low_slope_contributions']['auc'],
                alpha=ALPHA,replicates=REPLICATES),
            'contribution_only_auc_99_descriptive':paired.interval(low_draws,
                point=points['guaranteed_low_slope_contribution_only']['auc'],alpha=ALPHA,replicates=REPLICATES)})
    return {'horizon_minutes':5,'later_parents':len(later['parent_rows']),'eligible_nonzero_rows':int(mask.sum()),
            'actual_up_rows':int(y.sum()),'actual_down_rows':int((y==0).sum()),'points':points,'intervals':intervals}


def synthetic(rank, paired, io):
    cases=[]
    h=np.zeros((2,WIDTH),np.float32)
    h[0,:4]=[-.16,-.10,.20,0.]
    h[1,:4]=[-.17,-.01,.10,0.]
    w=np.zeros((3,WIDTH),np.float32)
    w[1,:4]=[2.,-1.,3.,7.]
    logits=np.zeros((2,3),np.float32)
    logits[:,1]=(h.astype(np.float64)@w[1].astype(np.float64)).astype(np.float32)
    mask,low,rest=decompose(h,w,logits)
    require(mask[:,0].all() and not mask[:,1:].any(),'SYNTHETIC_GROUP')
    require(np.allclose(low,h[:,0].astype(np.float64)*2)
            and np.allclose(low+rest,logits[:,1].astype(np.float64)),'SYNTHETIC_ADDITIVE')
    cases.extend(['analytical_post_gelu_cutoff','low_and_remainder_sum_to_original'])
    b=np.zeros((1,WIDTH),np.float64)
    b[0,0]=cutoff()
    b[0,1]=np.nextafter(cutoff(),-np.inf)
    m,_,_=decompose(b,w,np.zeros((1,3),np.float32))
    require(not m[0,0] and m[0,1],'SYNTHETIC_STRICT_BOUNDARY')
    cases.append('strict_threshold_boundary')
    summary=describe(h,w,logits)
    require(summary['guaranteed_low_slope_coordinates']==2 and summary['parents']==2,
            'SYNTHETIC_COUNTS')
    cases.append('coordinate_population_counts')
    corrupt=h.copy();corrupt[0,0]=np.nan
    try:
        decompose(corrupt,w,logits)
    except RuntimeError as exc:
        require(str(exc).endswith('FINITE_SHAPES'),'SYNTHETIC_EXPECTED_REJECTION')
    else:
        raise RuntimeError('GELU_CONTRIBUTION_SYNTHETIC_NAN_ACCEPTED')
    cases.append('nonfinite_activation_rejected')
    labels=np.array([0,1,0,1],np.int8)
    scores=np.array([0.,0.,-.1,.3],np.float64)
    inverse=np.array([0,0,1,1],np.int64)
    weights=np.array([[1.,1.],[2.,0.]],np.float64)
    got,pairs=paired.auc_draws(labels,scores,inverse,weights,rank)
    for i in range(2):
        ind=np.repeat(np.arange(4),weights[i,inverse].astype(int))
        pos=scores[ind][labels[ind]==1];neg=scores[ind][labels[ind]==0]
        expect=np.mean((pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:]))
        require(got[i]==expect and pairs[i]==len(pos)*len(neg),'SYNTHETIC_AUC_DUPLICATION')
    cases.append('paired_block_auc_matches_explicit_duplicate_rows_and_ties')
    return {'status':'PASS_SYNTHETIC_ONLY','passed':len(cases),'cases':cases,'saved_payload_reads':0}


def run_actual(meta, io, guard, frozen, probability, block_rank, paired, plan, paths, pins, fdpaths):
    inputs=plan['inputs']
    result={}
    for name in ('initial256','terminal_prefix_part1','terminal_later'):
        result[name]=meta.read_json(inputs[name]['result'])
        item=result[name]
        require(item['decision']=='COMPLETE_FROZEN_READOUT_EXTRACTION' and item['error'] is None
                and item['source_unchanged'] and not item['test_data_used'], 'ORIGINAL_RESULT:'+name)
        expected=inputs[name]['payload']
        require(item['representations']=={k:expected[k] for k in ('path','sha256')}, 'ORIGINAL_PAYLOAD:'+name)
    require(result['initial256']['state_role']=='own0U'
            and result['initial256']['representation_rows']==256
            and result['initial256']['original_prediction_parity']['bit_identical']
            and result['terminal_prefix_part1']['state_role']==result['terminal_later']['state_role']=='terminal_u2976'
            and result['terminal_prefix_part1']['representation_rows']==15872
            and result['terminal_later']['representation_rows']==LATER_ROWS
            and result['terminal_later']['original_prediction_parity']['bit_identical'], 'ORIGINAL_SCOPES')
    report=meta.read_json(inputs['original_oof_report'])
    require(report['role']=='OOF' and not report['test_data_used']
            and report['predictions']=={k:inputs['original_oof']['payload'][k] for k in ('path','sha256')},
            'ORIGINAL_OOF_REPORT')
    contexts=report['context_map']['contexts']
    names=BASE | {'context__'+c['name'] for c in contexts}
    def load(ref,keys):
        require(ref['path'] in paths,'DECLARED_PAYLOAD')
        pin=io.pin_artifact(ref);pins[ref['path']]=pin
        fdpaths.add('/proc/self/fd/'+str(pin['fd']))
        return io.read_archive(pin,set(keys),np)
    initial=load(inputs['initial256']['payload'],names | EXTRA)
    prefix=load(inputs['terminal_prefix_part1']['payload'],names | EXTRA)
    later=load(inputs['terminal_later']['payload'],names | EXTRA)
    original=load(inputs['original_oof']['payload'],names)
    frozen.validate_piece(initial,contexts,parent_start=0,parent_end=PREFIX_ROWS,
        presentation_start=0,presentation_end=256,lower_ns=START,cutoff_ns=PREFIX_END)
    frozen.validate_piece(prefix,contexts,parent_start=0,parent_end=PREFIX_ROWS,
        presentation_start=0,presentation_end=15872,lower_ns=START,cutoff_ns=PREFIX_END)
    frozen.validate_piece(later,contexts,parent_start=PREFIX_ROWS,parent_end=PREFIX_ROWS+LATER_ROWS,
        presentation_start=0,presentation_end=LATER_ROWS,lower_ns=PREFIX_END,cutoff_ns=LATER_END)
    require(np.array_equal(later['parent_rows'],np.arange(PREFIX_ROWS,PREFIX_ROWS+LATER_ROWS)),
            'EXACT_LATER_ROWS')
    indices=np.searchsorted(prefix['parent_rows'],initial['parent_rows'])
    require((indices<len(prefix['parent_rows'])).all(),'INITIAL_PREFIX_JOIN_RANGE')
    for key in ('parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign'):
        require(np.array_equal(initial[key],prefix[key][indices]),'INITIAL_PREFIX_JOIN:'+key)
    for key in names:
        require(later[key].dtype==original[key].dtype and later[key].shape==original[key].shape
                and later[key].tobytes()==original[key].tobytes(),'ORIGINAL_OOF_JOIN:'+key)
    require(np.array_equal(prefix['head_weight'],later['head_weight'])
            and np.array_equal(prefix['head_bias'],later['head_bias']), 'TERMINAL_HEAD_IDENTITY')
    recon={name:frozen.head_reconstruction(a) for name,a in (('initial256',initial),('terminal_prefix_part1',prefix),('terminal_later',later))}
    first={name:value[indices] for name,value in prefix.items() if name not in ('head_weight','head_bias')}
    first.update({key:prefix[key] for key in ('head_weight','head_bias')})
    panels={'initial256':describe(initial['hidden'],initial['head_weight'],initial['logit_up']),
            'terminal_same256':describe(first['hidden'],first['head_weight'],first['logit_up']),
            'terminal_later_full':describe(later['hidden'],later['head_weight'],later['logit_up'])}
    score=score_later(later,probability,block_rank,paired,io)
    require(abs(score['points']['original']['auc']-0.5030210874582608)<1e-12, 'NATIVE_AUC_REPRODUCED')
    return {'scope':'One reused TRAIN-development fold; no model fitting, forward, checkpoint or raw-data access.',
        'ideal_low_slope_definition':{'pre_gelu_x_less_than':-.5,'post_gelu_strictly_less_than':cutoff(),
            'maximum_absolute_ideal_derivative':0.13250487534383712,
            'interpretation':'Post-GELU criterion identifies a subset of low-slope preactivations. Deep negative preactivations can map near zero and are not counted; finite-precision error is not independently bounded.'},
        'matched_initial_terminal_parents':256,'head_reconstruction':recon,'panels':panels,'later_5m':score,
        'limitations':['Changing or omitting a head contribution is an exploratory diagnostic, not an Entry rule or retrained model.',
            'Input value before GELU cannot be reconstructed uniquely from negative GELU outputs.',
            'Intervals are descriptive on reused development data and do not prove economic edge or generalization.',
            'No attribution to a specific indicator, family, timeframe, or training mechanism.'],
        'model_forwards':0,'fits':0,'optimizer_steps':0,'checkpoint_reads':0,'test_data_used':False}


def main():
    global np
    sys.dont_write_bytecode=True
    helper=PROGRAM/'inspect_direction_gc_parquet_metadata.py'
    raw=helper.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==HELPER_SHA,'METADATA_HELPER')
    meta=types.ModuleType('gelu_contribution_metadata');meta.__file__=str(helper)
    exec(compile(raw,str(helper),'exec'),meta.__dict__)
    pref={'path':sys.argv[1],'sha256':sys.argv[2]}
    plan=meta.read_json(pref)
    design=meta.read_json(plan['design'])
    require(design['schema_version']=='gx1_direction_gelu_contribution_design_v1'
            and design['fixed_horizon_minutes']==5 and design['pre_gelu_cutoff']==-.5
            and design['bootstrap_replicates']==REPLICATES and design['interval_alpha']==ALPHA
            and design['bootstrap_seed']==SEED and design['test_data_used'] is False,
            'FIXED_ANALYTIC_DESIGN')
    polref={'path':str(REPO/'NEXT_RUN_POLICY.json'),'sha256':sys.argv[3]}
    policy=meta.read_json(polref);head=sys.argv[4]
    mode=plan['mode'];stage=policy[STAGE]
    require(plan['schema_version']=='gx1_direction_gelu_contribution_plan_v1' and mode in ('synthetic','actual'),'PLAN')
    require(stage['status']=='ADMITTED' and stage['plan']==pref and stage['mode']==mode
            and stage['execution_authorized'] is True and stage['max_invocations']==1
            and stage['claimed_invocations']==stage['completed_invocations']==0
            and stage['saved_payload_reads_authorized'] is (mode=='actual')
            and stage['relaunch_allowed'] is False,'ONE_SHOT')
    require(all(policy[k] is False for k in ('training_enabled','native_training_enabled','native_launch_authorized',
            'full_epoch_training_allowed','full_val_allowed')),'NATIVE_GATES_CLOSED')
    require(meta.git_head()==head and plan['wall_seconds']==120 and plan['max_rchar_bytes']==128*1024**2,'HEAD_AND_LIMITS')
    for ref in plan['sources'].values():meta.read_ref(ref)
    require(plan['sources']['worker']['path']==str(Path(__file__).resolve()),'WORKER_PATH')
    admission=meta.read_json(plan['source_admission'])
    require(str(Path(__file__).relative_to(REPO)) in admission['allowed_repo_files'],'SOURCE_ADMISSION')
    cap=meta.source_module('gelu_contribution_cap',plan['sources']['cap'])
    proof,lock=cap.require_capped_cpu_audit_execution(),cap.require_capped_lock_ancestry()
    require(lock['scope']==proof['cgroup_path'][len('/sys/fs/cgroup'):],'LOCK_SCOPE')
    io=meta.source_module('gelu_contribution_io',plan['sources']['io'])
    guard=meta.source_module('gelu_contribution_guard',plan['sources']['read_guard'])
    meta.PUBLISH=meta.source_module('gelu_contribution_publication',plan['sources']['publication'])
    out=Path(plan['out_dir'])
    require(out.parent==PROGRAM and out.parent.resolve(strict=True)==PROGRAM,'OUTPUT_ROOT')
    out.mkdir()
    claim=meta.publish(out,'INVOCATION_CLAIM.json',{'plan':pref,'policy':polref,'source_head':head,
        'mode':mode,'execution':proof,'lock':lock})
    paths=plan['payload_paths']
    require(len(paths)==len(set(paths))==(4 if mode=='actual' else 0),'PAYLOAD_COUNT')
    sys.addaudithook(guard.make_guard(out,paths))
    def expired(*unused):raise RuntimeError('GELU_CONTRIBUTION_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds'])
    started,before,previous=time.monotonic(),io.io_state(),sys.getprofile()
    pins,fdpaths,counts={},set(),{'numpy_load_calls':0,'numpy_load_returns':0}
    try:
        sys.path.insert(0,str(REPO))
        import numpy as numpy
        np=numpy;io.np=np
        rank=meta.source_module('gelu_contribution_rank',plan['sources']['probability'])
        paired=meta.source_module('gelu_contribution_paired',plan['sources']['paired'])
        paired.np=np
        frozen=meta.source_module('gelu_contribution_frozen',plan['sources']['frozen'])
        frozen.np=np
        block_rank=meta.source_module('gelu_contribution_block_rank',plan['sources']['rank'])
        def watch(frame,event,value):
            io.observe_work(frame,event,value)
            if frame.f_code is np.load.__code__ and event in ('call','return'):
                if event=='call':
                    counts['numpy_load_calls']+=1
                    require(counts['numpy_load_calls']<=(4 if mode=='actual' else 0)
                            and frame.f_locals.get('file') in fdpaths
                            and frame.f_locals.get('allow_pickle') is False,'LOAD_CAP')
                elif value is not None:counts['numpy_load_returns']+=1
        sys.setprofile(watch)
        if mode=='synthetic':
            result=synthetic(block_rank,paired,io)
        else:
            qualification=meta.read_json(plan['source_qualification'])
            require(qualification['status']=='PASS_SYNTHETIC_ONLY'
                    and qualification['worker']==plan['sources']['worker'],'QUALIFIED_WORKER')
            result=run_actual(meta,io,guard,frozen,rank,block_rank,paired,plan,paths,pins,fdpaths)
            require(counts['numpy_load_calls']==counts['numpy_load_returns']==4,'EXACT_LOAD_COUNT')
        require(not any(io.FORBIDDEN.values()),'NO_FORBIDDEN_WORK')
        require(counts['numpy_load_calls']==counts['numpy_load_returns'],'LOAD_RETURN_PARITY')
        io.pinned_bookends(pins)
        used=io.io_state()['rchar']-before['rchar'];require(used<=plan['max_rchar_bytes'],'IO_LIMIT')
        sys.setprofile(previous)
        for path,(identity,digest) in list(meta.PINNED.items()):
            require(meta.regular(path)==identity,'METADATA_BOOKEND')
            meta.read_ref({'path':path,'sha256':digest})
        require(meta.git_head()==head,'HEAD_BOOKEND')
        summary=meta.publish(out,'SUMMARY.json',{'execution_status':'COMPLETE','mode':mode,
            'plan':pref,'source_head':head,**result})
        terminal={'status':'COMPLETE','exit_code':0,'plan':pref,'policy':polref,'claim':claim,
            'summary':summary,'source_head':head,'source_unchanged':True,'inputs_unchanged':True,
            'counters':{**counts,**io.FORBIDDEN},'execution':proof,'lock':lock,
            'rchar_bytes':used,'elapsed_seconds':time.monotonic()-started,'test_data_used':False}
    except Exception as exc:
        sys.setprofile(previous)
        meta.publish(out,'TERMINAL.json',{'status':'FAILED','exit_code':1,'plan':pref,'policy':polref,
            'claim':claim,'source_head':head,'error':type(exc).__name__+': '+str(exc),
            'counters':{**counts,**io.FORBIDDEN},'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        raise
    finally:
        sys.setprofile(previous);signal.alarm(0)
        for pin in pins.values():os.close(pin['fd'])
    meta.publish(out,'TERMINAL.json',terminal)
    print(json.dumps({'status':'COMPLETE','mode':mode,'summary':summary,'elapsed_seconds':terminal['elapsed_seconds']}))


if __name__=='__main__':
    main()
