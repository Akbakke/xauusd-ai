"""Two admitted CUDA arms differing only in the order of 256 intact B16 batches.

The selected TRAIN population is scored in its original order. Cached INITIAL
scores are reused; evidence capsules do not grant canonical resume authority.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import io
import math
from pathlib import Path
import subprocess
import time

from gx1.audit import trace_market_state_ta_m5_original_cuda_prefix_v1 as original_trace
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

ARMS = ('CONTROL', 'INTERLEAVED')
TASKS = ('state', 'future')
CAPSULE_SCHEMA = 'gx1_ta_m5_paired_batch_order_evidence_v1'
FIT_MEASUREMENT_SCHEMA = 'gx1_ta_m5_paired_sampled_fit_measurement_v1'
SCORE_MARGIN = 1e-8


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_M5_PAIRED_BATCH_ORDER_' + reason)


def build_execution_orders(parent_rows, arm_batch_indices):
    """Keep each original B16 intact; unselected rows form an untouched suffix."""
    import numpy as np
    selected = [(i * 1847) // 255 for i in range(256)]
    expected = {'CONTROL': selected,
                'INTERLEAVED': [selected[r*16+c] for c in range(16) for r in range(16)]}
    _need(arm_batch_indices == expected, 'BATCH_SELECTION')
    _need(isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
          and parent_rows.shape == (29574,) and bool((np.diff(parent_rows) > 0).all()), 'ORDER_AXIS')
    positions = {arm: np.concatenate([np.arange(16*b,16*b+16,dtype=np.int64)
                                      for b in arm_batch_indices[arm]]) for arm in ARMS}
    chosen = np.sort(positions['CONTROL'])
    _need(chosen.size == np.unique(chosen).size == 4096
          and np.array_equal(chosen,np.sort(positions['INTERLEAVED'])), 'SELECTED_ROWS')
    keep = np.ones(len(parent_rows),dtype=bool); keep[chosen] = False
    orders = {arm: np.concatenate((parent_rows[positions[arm]],parent_rows[keep])) for arm in ARMS}
    _need(all(np.array_equal(np.sort(order),parent_rows) for order in orders.values())
          and not np.array_equal(orders['CONTROL'],orders['INTERLEAVED'])
          and np.array_equal(orders['CONTROL'][4096:],orders['INTERLEAVED'][4096:]), 'FULL_ORDER')
    return orders, positions


def sampled_fit_measurement(arrays, batch_losses, priors, *, model_contract_sha256,
                            loss_contract_sha256, feature_view_sha256):
    """Use the existing proper scorer on genuine sampled S / native F queries."""
    import numpy as np
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    n = len(arrays['parents'])
    _need(n > 0 and n % 16 == 0 and len(batch_losses) == n//16
          and all(math.isfinite(float(v)) for v in batch_losses), 'FIT_MEASUREMENT_BATCHES')
    digest = hashlib.sha256()
    for name in ('parents','issues','state_query','future_query'):
        value = arrays[name]
        shape = (n,) if name in ('parents','issues') else (n,5)
        _need(value.dtype == np.int64 and value.shape == shape, 'MEASUREMENT_AXIS')
        digest.update(np.ascontiguousarray(value).tobytes())
    scores, means, m5_prior = {}, {}, {}
    prior_raw = objective._endpoint_prior_arrays(priors,1,ta_only=True,dedicated_m5_side=True)
    for task in TASKS:
        raw = arrays[task+'_raw']
        _need(raw.dtype == np.float32 and raw.shape == (n,5,2) and np.isfinite(raw).all(),
              'MEASUREMENT_RAW')
        truth = {key:arrays[task+'_'+key] for key in ('allowed','eligible','mature')}
        for value in truth.values(): digest.update(np.ascontiguousarray(value).tobytes())
        scores[task] = objective._endpoint_task_sums(raw,truth,arrays[task+'_query'])
        means[task] = {}
        for tf, metrics in scores[task].items():
            means[task][tf] = {metric:{key:stats[key+'_sum']/stats['count']
                                     for key in ('brier','log') if key+'_sum' in stats}
                              for metric,stats in metrics.items() if stats['count']}
        reference = objective._endpoint_task_sums(
            np.broadcast_to(prior_raw[task],raw.shape),truth,arrays[task+'_query'])['M5']['conditional_side']
        m5_prior[task] = ({key:reference[key+'_sum']/reference['count'] for key in ('brier','log')}
                         if reference['count'] else {})
    return objective._seal(dict(schema_version=FIT_MEASUREMENT_SCHEMA,rows=n,batches=n//16,
        observation_chain_sha256=digest.hexdigest(),scores={'model':scores},
        means={'model':means,'prior':{t:{'M5':{'conditional_side':m5_prior[t]}} for t in TASKS}},
        mean_canonical_batch_loss=math.fsum(map(float,batch_losses))/len(batch_losses),
        canonical_batch_losses=list(map(float,batch_losses)),fold_id=1,pass_index=0,seed=20260911,
        query_domain={'state':'sampled_pass0','future':'native_endpoints'},
        loss_contract_sha256=loss_contract_sha256,model_contract_sha256=model_contract_sha256,
        feature_view_sha256=feature_view_sha256,frozen_prior_contract_sha256=priors['contract_sha256'],
        prior_comparisons='M5_only; no native_S_prior claim for other sampled_S_queries',
        test_data_used=False))


def compare_m5_measurements(measurements, *, baseline_score_atol):
    """Report every cell, respecting the reused INITIAL score error bound."""
    _need(type(baseline_score_atol) in (int,float) and math.isfinite(baseline_score_atol)
          and baseline_score_atol >= 0., 'BASELINE_SCORE_TOLERANCE')
    margins = {'INITIAL':max(SCORE_MARGIN,float(baseline_score_atol)),
               'CONTROL':SCORE_MARGIN,'PRIOR':SCORE_MARGIN}
    cells = {}
    for panel in ('TRAIN','LATER'):
        initial,control,candidate = (measurements[label+'_'+panel] for label in ('INITIAL',*ARMS))
        _need(all(m['observation_chain_sha256'] == initial['observation_chain_sha256']
                  and m['rows'] == initial['rows'] for m in (control,candidate)), 'MATCHED_OBSERVATIONS')
        for task in TASKS:
            n = initial['scores']['model'][task]['M5']['conditional_side']['count']
            _need(all(m['scores']['model'][task]['M5']['conditional_side']['count'] == n
                      for m in (control,candidate)), 'MATCHED_SUPPORT')
            refs = {label:m['means']['model'][task]['M5'].get('conditional_side',{})
                    for label,m in (('INITIAL',initial),('CONTROL',control))}
            refs['PRIOR'] = candidate['means']['prior'][task]['M5']['conditional_side']
            _need(refs['PRIOR'] == initial['means']['prior'][task]['M5']['conditional_side']
                  == control['means']['prior'][task]['M5']['conditional_side'], 'FROZEN_PRIOR')
            value = candidate['means']['model'][task]['M5'].get('conditional_side',{})
            differences = ({label:{metric:value[metric]-ref[metric] for metric in ('brier','log')}
                            for label,ref in refs.items()} if n else {})
            cells[panel+'_'+task] = dict(count=n,INTERLEAVED=value,references=refs,differences=differences,
                passed=bool(n and all(v < -margins[label] for label,d in differences.items() for v in d.values())))
    return dict(margin=SCORE_MARGIN,baseline_score_atol=float(baseline_score_atol),
        reference_margins=margins,cells=cells,
        decision='PASS_ALL_FOUR_CELLS' if all(c['passed'] for c in cells.values()) else 'FAIL_COMPARISON_NO_PROMOTION')


def _synthetic_checks(priors,recipe):
    """Small reporting/order/error checks only; no genuine payload or model."""
    import numpy as np
    orders,positions = build_execution_orders(np.arange(29574,dtype=np.int64),recipe['arm_batch_indices'])
    _need(np.array_equal(orders['CONTROL'][:4096],positions['CONTROL'])
          and all(int(positions[a][0]) == 0 for a in ARMS), 'SYNTHETIC_ORDER')
    try:
        build_execution_orders(np.arange(29574,dtype=np.int64),
                               {**recipe['arm_batch_indices'],'CONTROL':[0]*256})
    except RuntimeError: pass
    else: raise RuntimeError('SYNTHETIC_INVALID_ORDER_ACCEPTED')
    arrays = dict(parents=np.arange(16,dtype=np.int64),issues=np.arange(16,dtype=np.int64))
    for task in TASKS:
        arrays[task+'_query'] = np.tile(np.array([1,3,12,48,288],dtype=np.int64),(16,1))
        arrays[task+'_raw'] = np.zeros((16,5,2),np.float32)
        arrays[task+'_allowed'] = np.full((16,5),4,np.uint8)
        arrays[task+'_eligible'] = arrays[task+'_mature'] = np.ones((16,5),bool)
    args=dict(model_contract_sha256=recipe['expected_model_contract_sha256'],
              loss_contract_sha256=recipe['expected_loss_contract_sha256'],
              feature_view_sha256=recipe['expected_view_sha256'])
    measured=sampled_fit_measurement(arrays,[1.0],priors,**args)
    _need(measured['rows']==16 and measured['means']['model']['state']['M5']['conditional_side']['brier']==.25,
          'SYNTHETIC_SCORING')
    values={label+'_'+panel:copy.deepcopy(measured) for panel in ('TRAIN','LATER') for label in ('INITIAL',*ARMS)}
    tolerance=recipe['baseline_parity']['score_atol']
    _need(compare_m5_measurements(values,baseline_score_atol=tolerance)['decision']=='FAIL_COMPARISON_NO_PROMOTION','SYNTHETIC_NEGATIVE')
    for panel in ('TRAIN','LATER'):
        for task in TASKS:
            values['INTERLEAVED_'+panel]['means']['model'][task]['M5']['conditional_side']={'brier':0.,'log':0.}
    _need(compare_m5_measurements(values,baseline_score_atol=tolerance)['decision']=='PASS_ALL_FOUR_CELLS','SYNTHETIC_POSITIVE')
    for factor,expected in ((.5,'FAIL_COMPARISON_NO_PROMOTION'),(2.,'PASS_ALL_FOUR_CELLS')):
        close=copy.deepcopy(values)
        for panel in ('TRAIN','LATER'):
            for task in TASKS:
                close['INITIAL_'+panel]['means']['model'][task]['M5']['conditional_side']={
                    'brier':factor*max(SCORE_MARGIN,tolerance),'log':factor*max(SCORE_MARGIN,tolerance)}
        _need(compare_m5_measurements(close,baseline_score_atol=tolerance)['decision']==expected,
              'SYNTHETIC_CACHED_BASELINE_MARGIN')
    for task in TASKS: arrays[task+'_allowed'].fill(5)
    empty=sampled_fit_measurement(arrays,[1.0],priors,**args)
    _need(compare_m5_measurements({k:empty for k in values},baseline_score_atol=tolerance)['decision']=='FAIL_COMPARISON_NO_PROMOTION',
          'SYNTHETIC_ZERO_SUPPORT')
    values['CONTROL_TRAIN']['observation_chain_sha256']='0'*64
    try: compare_m5_measurements(values,baseline_score_atol=tolerance)
    except RuntimeError: pass
    else: raise RuntimeError('SYNTHETIC_MISMATCH_ACCEPTED')
    # Exercise the reused restoration comparison without constructing model/tensor state.
    sha=scope.campaign.canonical_sha256
    saved={key:{'toy':1} for key in ('model_state','buffer_state','optimizer_state','rng_state')}
    saved.update({key+'_sha256':sha(value) for key,value in list(saved.items())})
    saved.update(source_commit='a'*40,cursor=0,optimizer_updates=0,skipped_batches=0)
    saved['contract_sha256']=sha(saved)
    actual=copy.deepcopy(saved);actual['source_commit']='b'*40
    _need(original_trace._same_full_state(saved,actual,source='a'*40,tree_sha=sha,model_sha=sha)[
        'all_numerical_state_exact'],'SYNTHETIC_RESTORE')
    actual['optimizer_updates']=1
    try: original_trace._same_full_state(saved,actual,source='a'*40,tree_sha=sha,model_sha=sha)
    except RuntimeError: pass
    else: raise RuntimeError('SYNTHETIC_RESTORE_MISMATCH_ACCEPTED')
    return {'order_scoring_success_failure_zero_support_restore_checks':'PASS','model_constructions':0}


def preflight_read_only(recipe, *, source_commit=None):
    """Exact metadata/view qualification; no CLAIM, genuine arrays or model."""
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    if source_commit is None:
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=scope.CANONICAL_REPO,text=True).strip()
    scope.require_ta_m5_paired_batch_order_recipe(recipe,source_repo=scope.CANONICAL_REPO,
                                                source_commit=source_commit,verify_files=True)
    origin=scope.read_ta_m5_paired_batch_order_origin(recipe)
    receipts=zero.load_ta_input_receipts(current_source_owners={k:v['sha256'] for k,v in recipe['source_owners'].items()})
    view=zero._bind_ta_view(receipts,dedicated_m5_side=True)
    ref=origin['original_recipe']['input_refs']['ta_candidate_view']
    _need(view==scope.campaign.read_bound_json(Path(ref['path']),ref['sha256'])
          and view['contract_sha256']==recipe['expected_view_sha256'],'ORIGINAL_VIEW')
    ref=recipe['input_refs']['endpoint_priors']
    priors=scope.campaign.read_bound_json(Path(ref['path']),ref['sha256'])
    objective.require_endpoint_priors(priors,fold_id=1,ta_only=True,dedicated_m5_side=True)
    _need(recipe['numerical_recipe']==origin['original_recipe']['numerical_recipe']
          and recipe['numerical_recipe']['m5_equal_observation'] is True
          and recipe['numerical_recipe']['grad_accum_steps']==1
          and recipe['numerical_recipe']['grad_clip_norm']==recipe['numerical_recipe']['future_weight']==1.,
          'ORIGINAL_NUMERICAL')
    return dict(origin=origin,receipts=receipts,view=view,priors=priors,synthetic=_synthetic_checks(priors,recipe))


def run_paired_batch_order(*, policy_path:Path, policy_file_sha256:str, progress_path:Path)->dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    limits=require_guarded_cuda_trainer_execution()
    window,plan,invocation,marker,recipe=authority.require_native_direction_research_context(
        policy_path,policy_file_sha256,progress_path)
    _need(recipe['run_scope']==scope.TA_M5_PAIRED_BATCH_ORDER_SCOPE
          and recipe['max_private_updates']==recipe['max_backwards']==512
          and recipe['max_forwards']==1048 and recipe['max_checkpoint_loads']==1
          and recipe['max_full_state_restores']==2 and recipe['max_model_seconds']==2400
          and recipe['max_candidate_evidence_writes']==2 and recipe['max_checkpoint_writes']==0,
          'ADMITTED_BOUNDS')
    qualified=preflight_read_only(recipe,source_commit=plan['source_commit'])
    origin,receipts,view,priors=(qualified[k] for k in ('origin','receipts','view','priors'))
    original,refs,numerical=origin['original_recipe'],recipe['input_refs'],recipe['numerical_recipe']
    root=scope.TA_M5_PAIRED_BATCH_ORDER_ROOT;output=Path(window['output_directory'])
    output.mkdir(parents=True)
    _need(not any((root/name).exists() for name in
        ('RESULT.json','TERMINAL.json','CONTROL_EVIDENCE.pt','INTERLEAVED_EVIDENCE.pt')),'OUTPUT_EXISTS')
    counters={key:0 for key in ('model_constructions','model_construction_returns','optimizer_constructions',
        'forwards','forward_returns','training_forwards','evaluation_forwards','backwards','backward_returns',
        'optimizer_step_calls','optimizer_step_returns','optimizer_updates','checkpoint_loads',
        'resume_state_load_calls','resume_state_load_returns','strict_full_state_restore_calls',
        'strict_full_state_restore_returns','checkpoint_writes','candidate_evidence_writes','json_bytes',
        'cached_baseline_loss_batches','extra_truth_gathers')}
    arm_progress={arm:dict(optimizer_updates=0,parent_cursor=0,parent_order_sha256=None,
        batch_order_sha256=recipe['arm_batch_order_sha256'][arm]) for arm in ARMS}
    measurements={};measurement_refs={};capture_refs={};evidence={};restore_proofs={};first_steps={}
    training_rng={};step_records={arm:[] for arm in ARMS};timings={}
    started=time.monotonic();model_started=None;error=None
    provider=prepared=model=optimizer=initial=None
    same_selected=same_initial=same_queries=rng_parity=evaluation_unchanged=all_panels=source_unchanged=False
    score_gate=None

    def clock_guard():
        _need(time.monotonic()-started < window['max_invocation_seconds'],'INVOCATION_DEADLINE')
        if model_started is not None:
            _need(time.monotonic()-model_started < recipe['max_model_seconds'],'MODEL_DEADLINE')

    try:
        original_trace._bookend(recipe,plan,policy_path,policy_file_sha256)
        common._json(output/'CLAIM.json',dict(recipe=window['recipe'],plan_sha256=plan['plan_sha256'],
            source_commit=plan['source_commit'],limits=limits,test_data_used=False),counters,recipe['max_published_json_bytes'])
        import numpy as np
        import torch
        from gx1.contracts import entry_native_direction_feature_view_v1 as views
        from gx1.contracts import entry_native_direction_research_state_v1 as old
        from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
        from gx1.contracts import market_state_time_query_fold1_fit_v1 as full_scope
        from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
        from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets,sampled_query_k
        from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
        from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
        from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch,evaluate_endpoint_batch
        from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
        from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import _dynamic_m5_technical_tree_sha256 as tree_sha
        _need(torch.cuda.is_available(),'CUDA_REQUIRED');device=torch.device('cuda',0)
        bound,endpoint_bound,future_bound=load_bound_time_query_targets(),load_bound_five_endpoint_targets(),load_bound_future_displacement()
        provider,_,prepared,rows,issues=runner._prepared_fold1(receipts,view,np=np)
        objective.require_m5_equal_observation_axes(rows,issues)
        original_order=torch.from_numpy(rows.copy())
        _need(tree_sha(original_order)==origin['original_result']['parent_order_sha256'],'ORIGINAL_ORDER')
        orders,positions=build_execution_orders(rows,recipe['arm_batch_indices'])
        orders={arm:torch.from_numpy(order) for arm,order in orders.items()}
        for arm in ARMS:arm_progress[arm]['parent_order_sha256']=tree_sha(orders[arm])
        selected=positions['CONTROL'];same_selected=True
        axes=prepared['axes'];sl=slice(full_scope.FOLD['oof_start_row'],full_scope.FOLD['oof_end_row'])
        later_rows,later_issues=np.asarray(axes['parent_rows'][sl],np.int64),np.asarray(axes['decision_ns'][sl],np.int64)
        expected=endpoint_bound.mature_oof_axes(fold_id=1)
        _need(len(later_rows)==17753 and np.array_equal(later_rows,expected['score_parent_rows'])
              and np.array_equal(later_issues,expected['score_decision_ns']),'LATER_AXIS')
        later_positions=np.concatenate([np.arange(l,r,dtype=np.int64) for l,r in recipe['score_eval_ranges']])
        baseline=dict(parents=rows[selected].copy(),issues=issues[selected].copy())
        with np.load(refs['initial_fit_cache']['path'],allow_pickle=False) as cache, \
             np.load(refs['imported_predictions']['path'],allow_pickle=False) as imported:
            _need(np.array_equal(cache['parents'],rows) and np.array_equal(cache['issues'],issues),'BASELINE_FULL_AXIS')
            query=cache['query_train'];native=cache['query_native']
            _need(np.array_equal(query,sampled_query_k(issues,pass_index=0,seed=numerical['seed']))
                  and np.array_equal(native,np.tile(np.array([1,3,12,48,288],np.int64),(len(rows),1))),
                  'BASELINE_QUERIES')
            baseline['state_query'],baseline['future_query']=query[selected].copy(),native[selected].copy()
            for task,prefix in (('state','train_0'),('future','native_1')):
                raw=cache[prefix+'_original_raw'];side=imported['fit_'+task+'_original_imported']
                _need(raw.dtype==np.float32 and raw.shape==(29574,5,2) and side.shape==(29574,)
                      and np.isfinite(side).all()
                      and np.array_equal(side,side.astype(np.float32).astype(side.dtype)),'BASELINE_RAW')
                baseline[task+'_raw']=raw[selected].copy();baseline[task+'_raw'][:,0,1]=side[selected]
                for key in ('allowed','eligible','mature'):baseline[task+'_'+key]=cache[prefix+'_'+key][selected].copy()
                known=baseline[task+'_eligible'][:,0]&baseline[task+'_mature'][:,0]&np.isin(baseline[task+'_allowed'][:,0],(1,4))
                _need(np.array_equal(known,imported['fit_'+task+'_known'][selected])
                      and np.array_equal(baseline[task+'_allowed'][:,0]==4,imported['fit_'+task+'_y'][selected]),
                      'BASELINE_SIDE_SUPPORT')
        with np.load(refs['initial_later_capture']['path'],allow_pickle=False) as cache:
            later_capture={key:cache[key].copy() for key in ('parents','issues','query_native',
                'state_activity_side_logits','future_activity_side_logits','state_allowed','state_eligible',
                'state_mature','future_allowed','future_eligible','future_mature')}
        _need(np.array_equal(later_capture['parents'],later_rows[later_positions])
              and np.array_equal(later_capture['issues'],later_issues[later_positions]),'LATER_BASELINE_AXIS')
        measurements['INITIAL_LATER']=origin['initial_later_measurement']['measurement']
        measurement_refs['INITIAL_LATER']=refs['initial_later_measurement']
        capture_refs['INITIAL_LATER']=refs['initial_later_capture']
        with np.load(refs['original_trace']['path'],allow_pickle=False) as trace:
            first_reference={key:trace[key][0].copy() for key in ('state_raw','future_raw','state_query',
                'parents','issues','head_grad_before_clip','head_grad_before_clip_present','head_before')}
        timings['input_setup_seconds']=time.monotonic()-started
        clock_guard();model_started=time.monotonic()
        trainer._set_deterministic(numerical['seed'],device,numerical['precision_policy'])
        _need(trainer._GRAD_CLIP_NORM==numerical['grad_clip_norm']==1.,'CANONICAL_CLIP')
        counters['model_constructions']+=1
        backbone=views.build_native_direction_research_model(view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=device,unobserved_units_profile=receipts['fold1']['profile'])
        model=NativeFutureDualModel(backbone).to(device);counters['model_construction_returns']+=1
        view_identity=full_scope.require_checkpoint_view(model.state_model.feature_view_identity,
            recipe=original,verify_files=True,ta_sf=True)
        _need(model.research_objective['contract_sha256']==recipe['expected_model_contract_sha256'],'MODEL_CONTRACT')
        model.train();counters['optimizer_constructions']+=1
        optimizer,_=state.build_fold1_fit_optimizer(model,numerical_recipe=numerical)
        initial=common._load_state(refs['original_initial_checkpoint'],torch=torch,
            maximum=recipe['max_checkpoint_bytes'],counters=counters);counters['checkpoint_loads']+=1
        _need(initial['model_state_sha256']==recipe['expected_initial_model_state_sha256']
              and initial['cursor']==initial['optimizer_updates']==initial['skipped_batches']==0
              and not initial['optimizer_state']['state'],'EXACT_INITIAL')
        saved_seal=tree_sha(initial)
        pairs,active,inactive=state._parameters(model)
        head_params=[dict(pairs)[name] for name in original_trace.HEAD_NAMES]

        def signature():
            return dict(model=model_sha(model.state_dict()),buffers=tree_sha(dict(model.named_buffers())),
                optimizer=tree_sha(optimizer.state_dict()),rng=tree_sha(trainer._attended_session_rng_state(device=device)),
                modes={n:bool(m.training) for n,m in model.named_modules()})

        def publish_arrays(name,arrays):
            stream=io.BytesIO();np.savez_compressed(stream,**arrays)
            _need(stream.tell() <= recipe['max_trace_bytes'],'CAPTURE_BYTES')
            _publish_bytes(root/(name+'_READOUT.npz'),stream.getvalue())
            capture_refs[name]=common._hash_file(root/(name+'_READOUT.npz'))
            return capture_refs[name]

        def publish_measurement(name,measured,arrays,model_digest,updates):
            capture=publish_arrays(name,arrays)
            payload=dict(schema_version='gx1_ta_m5_paired_order_measurement_wrapper_v1',measurement=measured,
                capture=capture,evaluated_model_state_sha256=model_digest,optimizer_updates=updates,
                canonical_resume_authority=False,test_data_used=False)
            measurement_refs[name]=common._json(root/(name+'.json'),payload,counters,recipe['max_published_json_bytes'])
            measurements[name]=measured

        fit_score_args=dict(model_contract_sha256=recipe['expected_model_contract_sha256'],
            loss_contract_sha256=recipe['expected_loss_contract_sha256'],feature_view_sha256=recipe['expected_view_sha256'])
        baseline_losses=[]
        with torch.inference_mode():
            for left in range(0,4096,16):
                clock_guard();right=left+16
                truth={t:{k:baseline[t+'_'+k][left:right] for k in ('allowed','eligible','mature')} for t in TASKS}
                loss=objective.combined_loss(state_raw=torch.from_numpy(baseline['state_raw'][left:right]).to(device),
                    future_raw=torch.from_numpy(baseline['future_raw'][left:right]).to(device),
                    state_truth=truth['state'],future_truth=truth['future'],
                    state_query_k=torch.from_numpy(baseline['state_query'][left:right]).to(device),
                    future_weight=1.,ta_only=True,dedicated_m5_side=True,m5_equal_observation=True)
                baseline_losses.append(float(loss['loss']));counters['cached_baseline_loss_batches']+=1
        publish_measurement('INITIAL_TRAIN',sampled_fit_measurement(baseline,baseline_losses,priors,**fit_score_args),
                            baseline,initial['model_state_sha256'],0)

        def fit_report(original_start,training):
            raw,parents,clocks=runner._feature_batch(prepared,view,receipts,rows,issues,
                                                    original_start,original_start+16,np=np)
            features={k:torch.from_numpy(v).to(device) for k,v in raw.items()}
            _need(counters['forwards']<recipe['max_forwards'],'FORWARD_BUDGET')
            counters['forwards']+=1;counters['training_forwards' if training else 'evaluation_forwards']+=1
            if training:
                with trainer._training_mha_fastpath_context(model.backbone),trainer._training_autocast_context(), \
                     trainer._training_model_finite_check_context():
                    report=evaluate_fit_batch(model,features,bound,future_bound,parents,clocks,fold_id=1,
                        pass_index=0,seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,
                        fit_batch_start=original_start)
            else:
                with torch.inference_mode():
                    report=evaluate_fit_batch(model,features,bound,future_bound,parents,clocks,fold_id=1,
                        pass_index=0,seed=numerical['seed'],future_weight=1.,m5_equal_observation=True,
                        fit_batch_start=original_start)
            counters['forward_returns']+=1
            _need(report['model_contract_sha256']==recipe['expected_model_contract_sha256']
                  and report['loss_contract_sha256']==recipe['expected_loss_contract_sha256']
                  and report['skip_update'] is False and bool(torch.isfinite(report['loss'])),'CANONICAL_LOSS')
            slot=recipe['arm_batch_indices']['CONTROL'].index(original_start//16)*16
            _need(np.array_equal(parents,baseline['parents'][slot:slot+16])
                  and np.array_equal(clocks,baseline['issues'][slot:slot+16])
                  and np.array_equal(report['state_query_k'],baseline['state_query'][slot:slot+16])
                  and np.array_equal(report['future_native_k'],baseline['future_query'][slot]),'TRAIN_AXES_QUERY')
            for task in TASKS:
                for key in ('allowed','mature'):
                    _need(np.array_equal(report[task+'_'+key],baseline[task+'_'+key][slot:slot+16]),'TRAIN_TARGETS')
            # One extra gather per task only in evaluation verifies cached eligibility too.
            if not training:
                truths={'state':bound.gather_fit_sampled(parents,clocks,fold_id=1,pass_index=0,seed=numerical['seed']),
                        'future':future_bound.gather(parents,clocks,fold_id=1,partition='fit')}
                counters['extra_truth_gathers']+=2
                for task,truth in truths.items():
                    for key in ('allowed','eligible','mature'):
                        _need(np.array_equal(truth[key],baseline[task+'_'+key][slot:slot+16]),'CACHED_TRUTH_PARITY')
            return report,parents,clocks

        def measure_train(arm):
            before=signature();arrays={k:v.copy() for k,v in baseline.items()};losses=[]
            model.eval()
            try:
                for slot,b in enumerate(recipe['arm_batch_indices']['CONTROL']):
                    clock_guard();report,_,_=fit_report(16*b,False)
                    for task in TASKS:arrays[task+'_raw'][slot*16:(slot+1)*16]=report[task+'_activity_side_logits'].cpu().numpy()
                    losses.append(float(report['loss']));del report
            finally:
                for n,m in model.named_modules():m.training=before['modes'][n]
            _need(signature()==before and all(p.grad is None for _,p in pairs),'TRAIN_EVAL_CHANGED_STATE')
            measured=sampled_fit_measurement(arrays,losses,priors,**fit_score_args)
            _need(measured['observation_chain_sha256']==measurements['INITIAL_TRAIN']['observation_chain_sha256'],
                  'TRAIN_MATCHED_OBSERVATIONS')
            publish_measurement(arm+'_TRAIN',measured,arrays,before['model'],256)

        def measure_later(arm):
            before=signature();model.eval();total=None
            arrays={k:np.empty(shape,dtype) for k,(shape,dtype) in runner._ta_m5_capture_layout(192).items()}
            try:
                for left in range(0,192,16):
                    clock_guard()
                    raw,parents,clocks=runner._feature_batch(prepared,view,receipts,
                        later_rows[later_positions],later_issues[later_positions],left,left+16,np=np)
                    features={k:torch.from_numpy(v).to(device) for k,v in raw.items()}
                    _need(counters['forwards']<recipe['max_forwards'],'FORWARD_BUDGET')
                    counters['forwards']+=1;counters['evaluation_forwards']+=1
                    report=evaluate_endpoint_batch(model,features,endpoint_bound,future_bound,parents,clocks,
                        fold_id=1,partition='score',future_weight=1.,frozen_priors=priors,capture_m5_readout=True)
                    counters['forward_returns']+=1;total=objective.merge_endpoint_measurements(total,report['measurement'])
                    runner._ta_m5_capture_batch(arrays,report['m5_readout'],left=left,right=left+16,
                                               parents=parents,issues=clocks)
                    del raw,features,report
            finally:
                for n,m in model.named_modules():m.training=before['modes'][n]
            _need(signature()==before and all(p.grad is None for _,p in pairs),'LATER_EVAL_CHANGED_STATE')
            for key,value in later_capture.items():
                if not key.endswith('_logits'):_need(np.array_equal(arrays[key],value),'LATER_MATCHED_AXES_MASKS')
            measured=objective.finalize_endpoint_measurement(total)
            runner._ta_require_matched_observations(measurements['INITIAL_LATER'],measured)
            publish_measurement(arm+'_LATER',measured,arrays,before['model'],256)

        for arm in ARMS:
            clock_guard();counters['strict_full_state_restore_calls']+=1
            progress=state.restore_fold1_fit_state(initial,model,optimizer,numerical_recipe=numerical,
                feature_view_identity=view_identity,parent_order=original_order,
                source_commit=recipe['original_source_commit'],device=device)
            counters['strict_full_state_restore_returns']+=1
            restored=state.capture_fold1_fit_state(model,optimizer,numerical_recipe=numerical,
                feature_view_identity=view_identity,parent_order=original_order,source_commit=plan['source_commit'],
                device=device,**progress)
            restore_proofs[arm]=original_trace._same_full_state(initial,restored,
                source=recipe['original_source_commit'],tree_sha=tree_sha,model_sha=model_sha)
            _need(progress==dict(cursor=0,optimizer_updates=0,skipped_batches=0) and not optimizer.state
                  and all(p.grad is None for _,p in pairs),'EXACT_EMPTY_RESTORE')
            derived=state.capture_fold1_fit_state(model,optimizer,numerical_recipe=numerical,
                feature_view_identity=view_identity,parent_order=orders[arm],source_commit=plan['source_commit'],
                device=device,**progress)
            _need(all(tree_sha(restored[k])==tree_sha(derived[k])
                      for k in restored if k not in ('parent_order','contract_sha256')),'ORDER_ONLY_REBIND')
            restore_proofs[arm]['order_only_rebind']=True
            del restored,derived
            training_rng[arm]={'start':tree_sha(trainer._attended_session_rng_state(device=device))}
            fit_start=time.monotonic()
            for index,b in enumerate(recipe['arm_batch_indices'][arm]):
                clock_guard();start=16*b
                _need(np.array_equal(orders[arm][index*16:(index+1)*16].numpy(),rows[start:start+16]),'ARM_PREFIX')
                optimizer.zero_grad(set_to_none=True)
                report,parents,clocks=fit_report(start,True);loss=float(report['loss'].detach())
                counters['backwards']+=1;report['loss'].backward();counters['backward_returns']+=1
                if index==0:
                    gradient_sha=tree_sha({n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in pairs})
                    headgrad=np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in head_params])
                    _need(loss==original_trace.FIRST_LOSS
                          and np.array_equal(np.concatenate([p.detach().cpu().numpy().ravel() for p in head_params]),
                                             first_reference['head_before'])
                          and np.array_equal(headgrad,first_reference['head_grad_before_clip'])
                          and bool(first_reference['head_grad_before_clip_present'].all())
                          and np.array_equal(parents,first_reference['parents'])
                          and np.array_equal(clocks,first_reference['issues'])
                          and np.array_equal(report['state_query_k'],first_reference['state_query'])
                          and all(np.array_equal(report[t+'_activity_side_logits'].detach().cpu().numpy(),
                                                 first_reference[t+'_raw']) for t in TASKS),'ORIGINAL_FIRST_STEP')
                    first_steps[arm]=dict(loss=loss,full_gradient_sha256=gradient_sha,
                        head_gradient_sha256=hashlib.sha256(headgrad.tobytes()).hexdigest(),
                        rng_after_backward=tree_sha(trainer._attended_session_rng_state(device=device)),
                        original_trace_four_head_gradients_exact=True)
                counters['optimizer_step_calls']+=1
                norm=trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
                counters['optimizer_step_returns']+=1;counters['optimizer_updates']+=1
                arm_progress[arm]['optimizer_updates']=index+1;arm_progress[arm]['parent_cursor']=(index+1)*16
                optimizer.zero_grad(set_to_none=True)
                if index==0:
                    _need(float(norm)==original_trace.FIRST_NORM,'ORIGINAL_FIRST_NORM')
                    first_steps[arm]['gradient_norm']=float(norm)
                    if arm=='INTERLEAVED':_need(first_steps[arm]==first_steps['CONTROL'],'PAIRED_FIRST_STEP')
                step_records[arm].append(dict(original_batch_index=b,original_fit_batch_start=start,
                    optimizer_updates=index+1,loss=loss,gradient_norm=float(norm)))
                del report
                if (index+1)%64==0:print(arm+' updates='+str(index+1),flush=True)
            timings[arm+'_fit_seconds']=time.monotonic()-fit_start
            training_rng[arm]['end']=tree_sha(trainer._attended_session_rng_state(device=device))
            evaluate_start=time.monotonic();measure_train(arm);measure_later(arm)
            timings[arm+'_evaluation_seconds']=time.monotonic()-evaluate_start
            _need(all(p.grad is None and bool(torch.isfinite(p).all()) for _,p in pairs)
                  and state._binding(model,optimizer,numerical,view_identity)==initial['binding'],'FINAL_BINDING')
            model_state=copy.deepcopy(model.state_dict());buffers=copy.deepcopy(dict(model.named_buffers()))
            optimizer_state=copy.deepcopy(optimizer.state_dict());rng=copy.deepcopy(trainer._attended_session_rng_state(device=device))
            old._tensor_map(model_state,model.state_dict(),'EVIDENCE_MODEL')
            old._tensor_map(buffers,dict(model.named_buffers()),'EVIDENCE_BUFFERS')
            old._optimizer_state(optimizer_state,optimizer,model,256);old._rng(rng,device=device,seed=numerical['seed'])
            capsule=dict(schema_version=CAPSULE_SCHEMA,arm=arm,recipe=window['recipe'],source_commit=plan['source_commit'],
                original_checkpoint=refs['original_initial_checkpoint'],canonical_resume_authority=False,
                model_state=model_state,buffer_state=buffers,optimizer_state=optimizer_state,rng_state=rng,
                module_training={n:bool(m.training) for n,m in model.named_modules()},
                parent_order=orders[arm].clone(),parent_cursor=4096,physical_optimizer_updates=256,pending_gradients=0,
                parameter_binding=initial['binding'],original_numerical_recipe_reference=numerical,
                method_state=dict(arm_batch_indices=recipe['arm_batch_indices'][arm],
                    batch_order_sha256=arm_progress[arm]['batch_order_sha256'],
                    full_order_sha256=arm_progress[arm]['parent_order_sha256'],
                    original_fit_batch_starts=[16*b for b in recipe['arm_batch_indices'][arm]],
                    unselected_suffix_consumed=False,loss_and_optimizer_unchanged=True),
                model_state_sha256=model_sha(model_state),buffer_state_sha256=tree_sha(buffers),
                optimizer_state_sha256=tree_sha(optimizer_state),rng_state_sha256=tree_sha(rng),test_data_used=False)
            capsule['evidence_sha256']=tree_sha(capsule)
            stream=io.BytesIO();torch.save(capsule,stream)
            _need(0<stream.tell()<=recipe['max_candidate_evidence_bytes'],'EVIDENCE_BYTES');clock_guard()
            _publish_bytes(root/(arm+'_EVIDENCE.pt'),stream.getvalue());counters['candidate_evidence_writes']+=1
            evidence[arm]=common._hash_file(root/(arm+'_EVIDENCE.pt'))
            del capsule,model_state,buffers,optimizer_state,rng,stream
        _need(tree_sha(initial)==saved_seal,'ORIGINAL_PAYLOAD_MUTATED')
        same_initial=all(p['all_numerical_state_exact'] for p in restore_proofs.values())
        same_queries=evaluation_unchanged=all_panels=True
        rng_parity=training_rng['CONTROL']==training_rng['INTERLEAVED'];_need(rng_parity,'PAIRED_TRAIN_RNG')
        score_gate=compare_m5_measurements(measurements,baseline_score_atol=recipe['baseline_parity']['score_atol'])
        _need(all(counters[k]==v for k,v in scope.TA_M5_PAIRED_BATCH_ORDER_COMPLETE_COUNTS.items())
              and counters['forward_returns']==1048 and counters['backward_returns']==512
              and counters['optimizer_step_returns']==512 and counters['strict_full_state_restore_returns']==2,
              'EXACT_EXECUTION_COUNTS')
        original_trace._bookend(recipe,plan,policy_path,policy_file_sha256);clock_guard();source_unchanged=True
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        if prepared is not None:
            try:
                from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
                base._release_pages(provider,prepared)
            except BaseException as exc:error=(error+'; ' if error else '')+'cleanup: '+type(exc).__name__+': '+str(exc)
        prepared=provider=model=optimizer=initial=None;gc.collect()
    outcome='COMPLETE' if error is None and source_unchanged and all_panels else 'FAILED'
    payload=dict(schema_version=scope.TA_M5_PAIRED_BATCH_ORDER_RESULT_SCHEMA,outcome=outcome,
        decision='PASS_TA_M5_PAIRED_BATCH_ORDER_TECHNICAL' if outcome=='COMPLETE' else 'FAIL_TA_M5_PAIRED_BATCH_ORDER_TECHNICAL',
        recipe=window['recipe'],source_commit=plan['source_commit'],original_source_commit=recipe['original_source_commit'],
        optimizer_updates=counters['optimizer_updates'],physical_optimizer_updates=counters['optimizer_updates'],
        parent_cursor=None,parent_order_sha256=None,full_state=None,final_state=None,arm_progress=arm_progress,
        canonical_resume_authority=False,checkpoint_writes=0,skipped_batches=0,candidate_evidence=evidence,
        same_selected_rows=same_selected,same_initial_state=same_initial,same_training_queries=same_queries,
        rng_stream_parity=rng_parity,evaluation_state_unchanged=evaluation_unchanged,all_panels_complete=all_panels,
        first_steps=first_steps,restore_proofs=restore_proofs,training_rng=training_rng,step_records=step_records,
        measurement_refs=measurement_refs,captures=capture_refs,score_gate=score_gate,counts=counters,
        cached_INITIAL_TRAIN_proof=refs['objective_audit_result'],cached_INITIAL_LATER_proof=refs['centered_result'],
        synthetic=qualified['synthetic'],timings=dict(timings,elapsed_seconds=time.monotonic()-started),
        source_unchanged=source_unchanged,error=error,test_data_used=False,promotion_authorized=False,
        limits='Two 256-update technical arms, same4096TRAIN rows, intact B16 membership; batch presentation and dropout-to-row assignment differ. TRAIN uses sampledS/nativeF and later192 native endpoints. Cached INITIAL reconstruction has its qualified FP32 tolerance, not a new full-model baseline. All comparisons are development evidence; no edge, model selection, canonical resume or automatic continuation.')
    result=common._json(root/'RESULT.json',payload,counters,recipe['max_published_json_bytes'])
    common._json(root/'TERMINAL.json',dict(status=outcome,exit_code=0 if outcome=='COMPLETE' else 1,
        result=result,recipe=window['recipe'],source_unchanged=source_unchanged,test_data_used=False,
        no_relaunch_under_this_plan=True),counters,recipe['max_published_json_bytes'])
    cursor=scope.build_ta_m5_paired_batch_order_cursor(recipe_ref=window['recipe'],outcome=outcome,
        optimizer_updates=counters['optimizer_updates'],arm_progress=arm_progress,result=result,batch_size=16)
    cursor_ref=common._json(Path(window['campaign_cursor_path']),cursor,counters,recipe['max_published_json_bytes'])
    progress=authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=cursor_ref,optimizer_updates=counters['optimizer_updates'],
        parent_cursor=None,parent_order_sha256=None,outcome=outcome)
    common._json(progress_path,progress,counters,recipe['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    if error is not None:raise RuntimeError(error)
    return dict(result=result,checkpoint_pointer=cursor_ref,progress_path=str(progress_path))
