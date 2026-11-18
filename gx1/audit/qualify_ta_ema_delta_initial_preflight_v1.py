"""One capped 318/256 cold initial: FIT B16 loss and LATER B16 score; no fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from gx1.audit.market_state_ta_apr_jun_campaign_inputs_v1 import (
    REPO, COLD_FIT_ROWS, COLD_SCORE_START, COLD_SCORE_ROWS, axis_hash, read, ref,
)
from gx1.audit.run_market_state_ta_apr_jun_paired_fit_v1 import score
from gx1.audit.run_market_state_ta_m5_decision_fit_v1 import (
    COLD_TARGETS, load_targets, target_slice,
)
from gx1.contracts import entry_model_native_input_normalization_v1 as normal
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import contract as paired_contract
from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_endpoint_priors
from gx1.contracts.market_state_ta_m5_decision_v1 import (
    IntegratedTAM5DecisionModel, decision_loss, fit_statistics,
)
from gx1.contracts.market_state_time_query_binding_v1 import (
    load_bound_time_query_targets, sampled_query_k,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

STAGE = 'ta_m5_ema_delta_initial_preflight_20261118_001'
PLAN = REPO / 'configs/research/TA_M5_EMA_DELTA_INITIAL_PREFLIGHT_20261118_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_INITIAL_PREFLIGHT_20261118_001')
GATHER = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_FULL_GATHER_20261118_001')
PRIOR_ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_FOLD1_PRIORS_20261118_001')
SOURCE_BASE = '595ca0c86e15116f2229898f12a31f26ee64130e'
VIEW_SHA = '1bf7066a7595040ee541b83ca92765c500bc3619898553bca1714ed692fa3921'
LOSS_SHA = '2bcdacf11c849ff60da490af984ec065abf18948540f8494d6a8805517203f20'
STATS_SHA = '1e1278c2eced0a2344f2a7162fa8d54f5e5cb49d92640c59edf3edd38976fd30'
FIT_TARGET_SHA = '3f0a9c74a4a40da80fc62850b62dde3b67d7f896963440d58a5c810af9908024'
BATCH = 16
SEED = 20260911
WALL_SECONDS = 900
SURFACES = {'seq_x':'signal','snap_x':'signal','ctx_cont':'ctx_cont',
    **{'seq_' + tf:'mtf_' + tf for tf in ('m5','m15','h1','h4','d1')}}
EXPECTED_SHAPES = {'seq_x':(16,96,318),'snap_x':(16,318),'ctx_cont':(16,132),
    'ctx_cat':(16,1),'seq_m5':(16,16,256),'seq_m15':(16,64,256),
    'seq_h1':(16,96,256),'seq_h4':(16,96,256),'seq_d1':(16,252,256)}
GATHER_REFS = {
    'plan':dict(path=str(REPO / 'configs/research/TA_M5_EMA_DELTA_FULL_GATHER_20261118_001.json'),
        sha256='a8d1a39819402ca8fe80bb53e459f8431308fc7ea3153ac0a6e7ef65f2ff4864'),
    'result':dict(path=str(GATHER / 'RESULT.json'),
        sha256='f1589bff55919ac9a1bcb0c9ed99ae7848c51e5dd34bf453d9bd6d92d34b8f2a'),
    'coverage':dict(path=str(GATHER / 'COVERAGE.json'),
        sha256='d3c95a7a16a3a63c3276ba5f90dfa3b99a93957edea302cf990d521c3c20325c'),
    'terminal':dict(path=str(GATHER / 'TERMINAL.json'),
        sha256='57ff529479000918ac382268e857d518c8d89f7ebe8111bad16aedb4a6149ce3'),
}
SOURCES = (
    'gx1/audit/qualify_ta_ema_delta_initial_preflight_v1.py',
    'gx1/audit/market_state_ta_ema_delta_campaign_inputs_v1.py',
    'gx1/audit/qualify_ta_ema_delta_full_gather_v1.py',
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'gx1/audit/run_market_state_ta_apr_jun_paired_fit_v1.py',
    'gx1/audit/run_market_state_ta_m5_decision_fit_v1.py',
    'gx1/contracts/entry_ema_delta_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py',
    'gx1/contracts/entry_exit_feature_surface_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/market_state_ta_integrated_model_v1.py',
    'gx1/contracts/market_state_ta_integrated_paired_objective_v1.py',
    'gx1/contracts/market_state_ta_m5_decision_v1.py',
    'gx1/contracts/market_state_time_query_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/model_state_digest_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_ta_ema_delta_initial_preflight_v1.py',
)


def campaign():
    from gx1.audit import market_state_ta_ema_delta_campaign_inputs_v1 as owner
    return owner


def need(ok, reason):
    if not bool(ok):raise RuntimeError('TA_EMA_DELTA_INITIAL_' + reason)


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def checked(binding):
    need(ref(binding['path']) == binding, 'INPUT_REF')
    return read(binding)


def input_refs():
    return {'campaign':campaign().successor_input_refs(),
        'gather':GATHER_REFS,
        'fit_targets':dict(path=str(COLD_TARGETS),sha256=FIT_TARGET_SHA)}


def verify_gather(result, coverage, terminal, plan):
    refs = plan['inputs']
    gather = refs['gather']
    view_ref = refs['campaign']['view']
    normal_ref = refs['campaign']['normal_ta']
    material_ref = refs['campaign']['material_result']
    scope = result.get('scope',{})
    fit, later = result.get('fit',{}), result.get('later',{})
    need(result.get('status') == 'COMPLETE'
         and result.get('plan') == gather['plan']
         and result.get('view') == view_ref
         and result.get('normalization') == normal_ref
         and result.get('materialized_result') == material_ref
         and result.get('full_raw_and_normalized_selected_values') is True
         and result.get('source_unchanged') is True
         and result.get('target_arrays_decoded') == result.get('model_forwards')
             == result.get('normalization_fits') == result.get('optimizer_steps') == 0
         and result.get('test_data_used') is False
         and scope.get('fit_rows') == COLD_FIT_ROWS
         and scope.get('later_parent_start') == COLD_SCORE_START
         and scope.get('later_rows') == COLD_SCORE_ROWS
         and scope.get('selected_parent_rows') == COLD_FIT_ROWS + COLD_SCORE_ROWS
         and scope.get('feature_batches') == 2943
         and fit.get('rows') == COLD_FIT_ROWS and fit.get('batches') == 1849
         and fit.get('tail_rows') == 6
         and fit.get('parent_rows_sha256') == coverage.get('normalization_prefix',{}).get('parent_rows_sha256')
         and fit.get('decision_ns_sha256') == coverage.get('normalization_prefix',{}).get('decision_ns_sha256')
         and later.get('rows') == COLD_SCORE_ROWS and later.get('batches') == 1094
         and later.get('tail_rows') == 1
         and coverage.get('status') == 'COMPLETE'
         and coverage.get('view_sha256') == VIEW_SHA
         and coverage.get('qualification') == material_ref
         and coverage.get('score_rows') == COLD_SCORE_ROWS
         and coverage.get('selected_parent_rows_sha256') == later.get('parent_rows_sha256')
         and coverage.get('selected_decision_ns_sha256') == later.get('decision_ns_sha256')
         and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
         and terminal.get('plan') == gather['plan']
         and terminal.get('result') == gather['result']
         and terminal.get('coverage') == gather['coverage']
         and terminal.get('source_unchanged') is True
         and terminal.get('relaunch_allowed') is False, 'GATHER_CHAIN')
    for phase, rows in ((fit,COLD_FIT_ROWS),(later,COLD_SCORE_ROWS)):
        need(set(phase.get('surfaces',{})) == set(SURFACES)
             and all(phase['surfaces'][key]['shape_per_row'] == list(EXPECTED_SHAPES[key][1:])
                     and len(phase['surfaces'][key]['raw_sha256']) == 64
                     and len(phase['surfaces'][key]['normalized_sha256']) == 64
                     for key in SURFACES), 'GATHER_SURFACES')
    return fit,later


def build_plan():
    refs = input_refs()
    gather_plan = checked(refs['gather']['plan'])
    result, coverage, terminal = (checked(refs['gather'][k]) for k in
        ('result','coverage','terminal'))
    plan = dict(schema_version='gx1_ta_m5_ema_delta_initial_preflight_plan_v1',
        stage=STAGE, source_base_commit=SOURCE_BASE,
        source_owners={p:ref(REPO/p) for p in SOURCES},
        inputs=refs, output_root=str(ROOT),
        scope=dict(view_sha256=VIEW_SHA, fit_rows=COLD_FIT_ROWS,
            later_parent_start=COLD_SCORE_START,later_rows=COLD_SCORE_ROWS,
            fit_batch_rows=BATCH,later_batch_rows=BATCH,
            prior_loss_contract_sha256=LOSS_SHA,fit_statistics_sha256=STATS_SHA,
            fold1_paired_objective_sha256=paired_contract(fold_id=1)['contract_sha256'],
            model_constructions=1,model_forwards=2,backward_calls=0,
            optimizer_constructions=0,optimizer_steps=0,normalization_fits=0,
            test_data_used=False,economic_edge_proven=False),
        execution=dict(runner='scripts/gx1_capped_run.sh',job_class='audit',
            memory='4G',swap='512M',timeout_seconds=WALL_SECONDS,
            max_invocations=1,training_authorized=False,test_authorized=False))
    need(gather_plan['inputs']['view'] == refs['campaign']['view']
         and gather_plan['inputs']['normal_ta'] == refs['campaign']['normal_ta']
         and gather_plan['inputs']['material_result'] == refs['campaign']['material_result'],
         'GATHER_INPUT_REFS')
    verify_gather(result,coverage,terminal,plan)
    need(refs['campaign']['priors']['path'] == str(PRIOR_ROOT / 'ENDPOINT_PRIORS.json')
         and refs['campaign']['prior_result']['path'] == str(PRIOR_ROOT / 'RESULT.json')
         and refs['campaign']['prior_terminal']['path'] == str(PRIOR_ROOT / 'TERMINAL.json'),
         'NEW_PRIOR_ROOT')
    prior = checked(refs['campaign']['priors'])
    require_endpoint_priors(prior, fold_id=1, ta_only=True, dedicated_m5_side=True)
    need(prior['loss_contract_sha256'] == LOSS_SHA, 'CURRENT_PRIOR_LOSS')
    need(refs['campaign']['gather_result'] == refs['gather']['result']
         and refs['campaign']['gather_coverage'] == refs['gather']['coverage']
         and refs['campaign']['gather_terminal'] == refs['gather']['terminal'],
         'GATHER_CAMPAIGN_REFS')
    need(ref(COLD_TARGETS) == refs['fit_targets'], 'FIT_TARGET_REF')
    return plan


def inspect(plan_sha, *, execute):
    binding=ref(PLAN);plan=read(binding)
    need(binding['sha256']==plan_sha and plan==build_plan() and not ROOT.exists(),
         'PLAN_ONE_USE')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    clean=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)==''
    ancestor=subprocess.run(['git','merge-base','--is-ancestor',SOURCE_BASE,head],
        cwd=REPO,capture_output=True).returncode==0
    need(Path.cwd().resolve()==REPO and branch=='work/gx1-current' and ancestor,
         'SOURCE_BRANCH_CWD')
    if not execute:return plan,binding,None,head,clean
    need(clean,'CLEAN_SOURCE')
    policy_ref=ref(REPO/'NEXT_RUN_POLICY.json');policy=read(policy_ref)
    need(all(policy.get(k) is False for k in ('training_enabled',
        'native_training_enabled','native_launch_authorized',
        'full_epoch_training_allowed','full_val_allowed'))
        and policy.get('exit_training_and_evaluation_this_wave') is False,
        'TRAINING_TEST_CLOSED')
    gate=policy.get(STAGE,{})
    need(gate.get('status')=='ADMITTED' and gate.get('plan')==binding
         and gate.get('scope')==plan['scope']
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('model_constructions_authorized')==1
         and gate.get('model_forwards_authorized')==2
         and gate.get('optimizer_steps_authorized')==0
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('max_invocations')==gate.get('remaining_invocations')==1
         and gate.get('claimed_invocations')==gate.get('completed_invocations')==0
         and gate.get('relaunch_allowed') is False,'ONE_USE_ADMISSION')
    return plan,binding,policy_ref,head,clean


def normalize_probe(model,features,ta,phase):
    backbone=model.core.dual.state_model.backbone
    details={}
    need(set(features)==set(EXPECTED_SHAPES)
         and all(tuple(features[k].shape)==shape for k,shape in EXPECTED_SHAPES.items())
         and features['ctx_cat'].dtype==torch.int64,
         'B16_SHAPES')
    for key,surface in SURFACES.items():
        raw=features[key]
        need(raw.dtype==torch.float32 and bool(torch.isfinite(raw).all()),'RAW_'+key)
        with torch.no_grad():z=backbone._normalize_input_surface(raw,surface=surface)
        expected=normal.apply_surface_normalization(raw.numpy(),ta['surfaces'][surface])
        actual=z.detach().numpy()
        need(actual.shape==expected.shape and np.isfinite(actual).all()
             and np.allclose(actual,expected,rtol=2e-6,atol=2e-6),
             'MODEL_NORMALIZATION_'+key)
        details[key]=dict(shape=list(actual.shape),
            raw_sha256=hashlib.sha256(np.ascontiguousarray(raw.numpy()).tobytes()).hexdigest(),
            normalized_sha256=hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
            max_abs_reference_error=float(np.max(np.abs(actual-expected))))
    return details


def probes(model,context,fit,ta,*,deadline):
    need(context.get('fit_rows')==COLD_FIT_ROWS and context.get('score_rows')==COLD_SCORE_ROWS,
         'COLD_AXES')
    axes=context['axes'];selected=context['selected']
    need(context['fit_rows']==COLD_FIT_ROWS and context['score_rows']==COLD_SCORE_ROWS
         and context['fit_fold_id']==context['score_fold_id']==1
         and context['score_partition']=='score'
         and context['view']['contract_sha256']==VIEW_SHA
         and context['units_profile']==context['view']['unobserved_units_profile']
         and np.array_equal(axes['parent_rows'][:COLD_FIT_ROWS],fit['parent_rows'])
         and np.array_equal(axes['decision_ns'][:COLD_FIT_ROWS],fit['issue_ns'])
         and np.array_equal(selected,np.arange(COLD_SCORE_START,47338,dtype=np.int64)),
         'COLD_AXES')
    fit_rows,fit_issues=axes['parent_rows'][:BATCH],axes['decision_ns'][:BATCH]
    fit_features=context['features']('TRAIN',fit_rows,fit_issues)
    fit_norm=normalize_probe(model,fit_features,ta,'FIT')
    sampled=sampled_query_k(fit_issues,pass_index=0,seed=SEED)
    native=np.broadcast_to(np.asarray(NATIVE_K,dtype=np.int64),sampled.shape).copy()
    query=torch.from_numpy(np.stack((sampled,native),axis=1).copy())
    tape=context['fit_tape']
    sampled_truth=load_bound_time_query_targets().gather(fit_rows,fit_issues,sampled,
        fold_id=1,partition='fit',pass_index=0,seed=SEED)
    native_truth=load_bound_five_endpoint_targets().gather(fit_rows,fit_issues,
        fold_id=1,partition='fit')
    future_truth=load_bound_future_displacement().gather(fit_rows,fit_issues,
        fold_id=1,partition='fit')
    targets={key:fit[key][:BATCH] for key in
        ('terminal_observed','path_observed','net_bps','mfe_bps','pre_mfe_mae_upper_bps')}
    model.eval()
    with torch.inference_mode():
        output=model(fit_features,query_k=query,
            m1_path_bps=torch.from_numpy(tape['m1_mid_ohlc_bps'][:BATCH].copy()),
            m1_observed=torch.from_numpy(tape['m1_observed'][:BATCH].copy()))
        loss=decision_loss(model_output=output,targets=targets,model=model,
            sampled_truth=sampled_truth,native_truth=native_truth,
            future_truth=future_truth,query_grid=query,
            structure_hit=tape['structure_hit'][:BATCH],
            structure_observed=tape['structure_observed'][:BATCH])
    need(bool(torch.isfinite(loss['loss']))
         and loss['contract_sha256']==model.research_objective['contract_sha256'],
         'FIT_LOSS')
    later_rows,later_issues=(axes[k][selected[:BATCH]] for k in ('parent_rows','decision_ns'))
    later_features=context['features']('SCORE',later_rows,later_issues)
    later_norm=normalize_probe(model,later_features,ta,'LATER')
    def cached(label,rows,issues):
        need(label=='SCORE' and np.array_equal(rows,later_rows)
             and np.array_equal(issues,later_issues),'SCORE_B16_AXIS')
        return later_features
    shallow={**context,'selected':selected[:BATCH],
        'score_tape':{k:v[:BATCH] for k,v in context['score_tape'].items()},
        'features':cached}
    score_readout,capture=score(model,shallow,deadline=deadline,
        fold_id=1,partition='score',collect_decision=True)
    pred=capture['m5_cost_path_bps']
    need(score_readout['rows']==BATCH and score_readout['model_forwards']==1
         and pred.shape==(BATCH,6,2,3) and np.isfinite(pred).all(),'LATER_FORWARD')
    return dict(fit_normalization=fit_norm,later_normalization=later_norm,
        fit_loss=float(loss['loss']),fit_loss_contract_sha256=loss['contract_sha256'],
        later_decision_output_sha256=hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest())


def publish(name,value):
    data=encoded(value);need(len(data)<=2*1024**2,'JSON_BUDGET')
    _publish_bytes(ROOT/name,data);return ref(ROOT/name)


def bookend(plan,binding,policy_ref,head):
    need(ref(PLAN)==binding and ref(REPO/'NEXT_RUN_POLICY.json')==policy_ref
         and subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==head
         and subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)==''
         and all(ref(REPO/p)==r for p,r in plan['source_owners'].items())
         and all(ref(r['path'])==r for r in
             (*plan['inputs']['campaign'].values(),*plan['inputs']['gather'].values(),
              plan['inputs']['fit_targets'])),'SOURCE_INPUT_BOOKEND')


def run(plan_sha):
    started=time.monotonic()
    cap=require_capped_cpu_audit_execution();require_capped_lock_ancestry()
    need(cap['memory_max']==cap['memory_high']==4*1024**3
         and cap['swap']==512*1024**2,'CAP')
    plan,binding,policy_ref,head,_=inspect(plan_sha,execute=True)
    need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    ROOT.mkdir()
    claim=publish('CLAIM.json',dict(plan=binding,policy=policy_ref,
        source_head=head,cap=cap,scope=plan['scope']))
    status='FAILED_CONSUMED';result=failure=context=None
    old_alarm=signal.getsignal(signal.SIGALRM)
    def expired(*_):raise TimeoutError('TA_EMA_DELTA_INITIAL_TIMEOUT')
    try:
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,max(1,WALL_SECONDS-30-(time.monotonic()-started)))
        context=campaign().prepare(device=torch.device('cpu'),inputs=plan['inputs']['campaign'])
        need(context['input_refs']==plan['inputs']['campaign'],'CAMPAIGN_INPUT_REFS')
        full=load_targets(plan['inputs']['fit_targets'],47338)
        fit=target_slice(full,0,COLD_FIT_ROWS)
        stats=fit_statistics(fit)
        need(stats['sha256']==STATS_SHA and 'paired_quality_std_bps' in stats,
             'FIT_STATISTICS_V2')
        trainer._set_deterministic(SEED,torch.device('cpu'),trainer.DETERMINISTIC_FP32)
        state=views.build_native_direction_research_model(context['view'],
            prefix_normalization=context['packet'],constructor_metadata=context['constructor'],
            device=torch.device('cpu'),unobserved_units_profile=context['units_profile'])
        core=IntegratedTAMarketStateModel(NativeFutureDualModel(state))
        need(core.dual.state_model.research_objective['contract_sha256']
             ==context['view']['research_objective']['contract_sha256'],'CORE_VIEW')
        model=IntegratedTAM5DecisionModel(core,stats,aux_fold_id=1).eval()
        digest=canonical_model_state_sha256(model.state_dict())
        need(model.research_objective['cold_aux_contract_sha256']
             ==plan['scope']['fold1_paired_objective_sha256'],'FOLD1_OBJECTIVE')
        ta=checked(plan['inputs']['campaign']['normal_ta'])
        measured=probes(model,context,fit,ta,deadline=started+WALL_SECONDS-30)
        need(canonical_model_state_sha256(model.state_dict())==digest,
             'INITIAL_STATE_UNCHANGED')
        context['release']();context=None
        bookend(plan,binding,policy_ref,head)
        result=publish('RESULT.json',dict(
            status='COMPLETE_EMA_DELTA_INITIAL_FIT_SCORE_NO_FIT',
            plan=binding,claim=claim,policy=policy_ref,source_head=head,source_unchanged=True,
            view=plan['inputs']['campaign']['view'],
            normalization=plan['inputs']['campaign']['normal_ta'],
            raw_gather_result=plan['inputs']['gather']['result'],
            raw_gather_coverage=plan['inputs']['gather']['coverage'],
            raw_gather_terminal=plan['inputs']['gather']['terminal'],
            fit_targets=plan['inputs']['fit_targets'],
            prior=plan['inputs']['campaign']['priors'],
            prior_result=plan['inputs']['campaign']['prior_result'],
            prior_terminal=plan['inputs']['campaign']['prior_terminal'],
            scope=plan['scope'],
            initial_model_state_sha256=digest,
            core_contract_sha256=core.research_objective['contract_sha256'],
            decision_contract_sha256=model.research_objective['contract_sha256'],
            cold_paired_objective_sha256=plan['scope']['fold1_paired_objective_sha256'],
            fit_statistics_sha256=stats['sha256'],
            fit_loss_contract_sha256=measured['fit_loss_contract_sha256'],
            fit_loss=measured['fit_loss'],
            normalized_numeric_surfaces={'FIT':measured['fit_normalization'],
                'LATER':measured['later_normalization']},
            score_b16_decision_output_sha256=measured['later_decision_output_sha256'],
            model_constructions=1,model_forwards=2,backward_calls=0,
            optimizer_constructions=0,optimizer_steps=0,normalization_fits=0,
            test_data_used=False,economic_edge_proven=False))
        status='COMPLETE'
    except BaseException as exc:
        failure=publish('FAILED.json',dict(status='FAILED_CONSUMED',plan=binding,
            claim=claim,error_type=type(exc).__name__,error=str(exc)[:1800],result=result))
    finally:
        if context is not None:
            try:context['release']()
            except BaseException as exc:
                if failure is None:failure=publish('FAILED.json',dict(status='FAILED_CONSUMED',
                    plan=binding,claim=claim,error_type=type(exc).__name__,error=str(exc)[:1800],
                    result=result))
                status='FAILED_CONSUMED'
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_alarm)
        terminal=publish('TERMINAL.json',dict(status=status,
            exit_code=0 if status=='COMPLETE' else 1,plan=binding,claim=claim,
            result=result,failure=failure,no_relaunch=True,remaining_invocations=0,
            source_unchanged=subprocess.check_output(['git','rev-parse','HEAD'],
                cwd=REPO,text=True).strip()==head))
    return status,terminal


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--execute',action='store_true')
    parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args()
    if args.preflight:
        plan,binding,_,head,clean=inspect(args.plan_sha256,execute=False)
        print(json.dumps(dict(status='PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            plan=binding,source_head=head,source_clean=clean,claim_created=False,
            output_root_absent=True,scope=plan['scope']),sort_keys=True));return
    status,terminal=run(args.plan_sha256)
    print(json.dumps(dict(status=status,terminal=terminal),sort_keys=True))
    if status!='COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
