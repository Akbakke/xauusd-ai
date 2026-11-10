"""One admitted CUDA pilot using fixed TRAIN-mean M5 head coordinates."""
from __future__ import annotations

import copy
import gc
import io
from pathlib import Path
import time

from gx1.audit import trace_market_state_ta_m5_original_cuda_prefix_v1 as original_trace
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

TRAIN_RANGES = ((0, 64), (14272, 14336), (29504, 29568))
SCORE_RANGES = ((0, 64), (8352, 8416), (17680, 17744))
SCORE_MARGIN = 1e-8
CAPSULE_SCHEMA = 'gx1_ta_m5_centered_head_coordinates_evidence_v1'


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_M5_CENTERED_HEAD_COORDINATES_' + reason)


def centered_head_optimizer_step(*, model, optimizer, means, optimizer_step) -> dict:
    """Change only M5 head coordinates around one caller-owned clip/AdamW call.

    The finally block returns biases to ordinary forward coordinates even when
    the callback raises. This is not rollback: gradients, weights or Adam state
    may already have changed, so the caller must fail and stop after an error.
    """
    import torch
    heads = (model.state_model.m5_side_head, model.future_m5_side_head)
    _need(type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1
          and callable(optimizer_step) and isinstance(means, dict)
          and set(means) == {'state', 'future'}, 'STEP_OWNER')
    group = optimizer.param_groups[0]
    parameters = group['params']
    prepared = []
    for name, head in zip(('state', 'future'), heads):
        _need(type(head) is torch.nn.Linear and head.in_features == 128
              and head.out_features == 1 and head.bias is not None, 'HEAD_SHAPE')
        mu = means[name]
        _need(isinstance(mu, torch.Tensor) and mu.shape == (128,)
              and mu.dtype == torch.float32 and mu.device == head.weight.device
              and not mu.requires_grad and bool(torch.isfinite(mu).all()), 'FIXED_MEAN')
        for parameter in (head.weight, head.bias):
            _need(parameter.dtype == torch.float32 and parameter.device == mu.device
                  and parameter.requires_grad and parameter.grad is not None
                  and parameter.grad.dtype == parameter.dtype
                  and parameter.grad.shape == parameter.shape
                  and sum(parameter is item for item in parameters) == 1
                  and bool(torch.isfinite(parameter).all())
                  and bool(torch.isfinite(parameter.grad).all()), 'HEAD_PARAMETER_OR_GRADIENT')
        # One canonical group makes all W/b learning rates, decay and Adam
        # settings identical. Preserve those caller-owned nondefault settings.
        frozen = mu.detach().clone()
        with torch.no_grad():
            centered_gradient = head.weight.grad - head.bias.grad.reshape(1, 1) * frozen
            offset = (head.weight * frozen).sum().reshape_as(head.bias)
            _need(bool(torch.isfinite(centered_gradient).all())
                  and bool(torch.isfinite(head.bias + offset).all()), 'CENTERED_FINITE')
        prepared.append((head, frozen, centered_gradient, offset))
    converted = []
    try:
        with torch.no_grad():
            for head, mu, gradient, offset in prepared:
                head.weight.grad.copy_(gradient)
                head.bias.add_(offset)
                converted.append((head, mu))
        norm = optimizer_step()
        return {'gradient_norm': norm}
    finally:
        with torch.no_grad():
            for head, mu in reversed(converted):
                head.bias.sub_((head.weight * mu).sum().reshape_as(head.bias))


def run_centered_head_coordinates(*, policy_path: Path, policy_file_sha256: str,
                                 progress_path: Path) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    limits = require_guarded_cuda_trainer_execution()
    window, plan, invocation, marker, recipe = authority.require_native_direction_research_context(
        policy_path, policy_file_sha256, progress_path)
    _need(recipe['run_scope'] == scope.TA_M5_CENTERED_HEAD_COORDINATES_SCOPE
          and recipe['max_private_updates'] == recipe['max_backwards'] == 256
          and recipe['max_forwards'] == 328 and recipe['max_model_seconds'] == 900
          and recipe['max_checkpoint_loads'] == 2 and recipe['max_full_state_restores'] == 3
          and recipe['max_checkpoint_writes'] == 0
          and recipe['max_candidate_evidence_writes'] == 1
          and recipe['max_candidate_evidence_bytes'] == recipe['max_checkpoint_bytes']
          and all(recipe[key] is True for key in (
              'first_forward_raw_gradient_parity_required','control_p4096_restore_parity_required',
              'function_equivalent_initial_required','evaluation_state_unchanged_required')),
          'ADMITTED_BOUNDS')
    origin = scope.read_ta_m5_centered_head_coordinates_origin(recipe)
    original, refs = origin['original_recipe'], recipe['input_refs']
    numerical = recipe['numerical_recipe']
    _need(numerical == original['numerical_recipe'] and numerical['grad_accum_steps'] == 1
          and numerical['grad_clip_norm'] == numerical['future_weight'] == 1.0
          and numerical['m5_equal_observation'] is True, 'ORIGINAL_NUMERICAL')
    root = scope.TA_M5_CENTERED_HEAD_COORDINATES_ROOT
    output = Path(window['output_directory'])
    output.mkdir(parents=True)
    _need(not any((root/name).exists() for name in
          ('RESULT.json','TERMINAL.json','CANDIDATE_EVIDENCE.pt')), 'OUTPUT_EXISTS')
    counters = {key:0 for key in (
        'model_constructions','model_construction_returns','optimizer_constructions',
        'forwards','forward_returns','training_forwards','evaluation_forwards',
        'backwards','backward_returns','optimizer_step_calls','optimizer_step_returns',
        'optimizer_updates','checkpoint_loads','resume_state_load_calls','resume_state_load_returns',
        'strict_full_state_restore_calls','strict_full_state_restore_returns',
        'checkpoint_writes','candidate_evidence_writes','json_bytes')}
    started = time.monotonic()
    model_started = None
    provider = prepared = model = optimizer = initial = control = None
    error = None
    source_unchanged = all_panels_complete = evaluation_unchanged = initial_equivalent = False
    first_exact = False
    cursor, order_sha = 0, '0'*64
    measurements, measurement_refs, captures, restore_proofs, baseline_parity = {}, {}, {}, {}, {}
    step_records, timings, candidate_ref, score_gate = [], {}, None, None

    def clock_guard():
        now = time.monotonic()
        _need(now-started < window['max_invocation_seconds'], 'INVOCATION_DEADLINE')
        if model_started is not None:
            _need(now-model_started < recipe['max_model_seconds'], 'MODEL_DEADLINE')

    try:
        original_trace._bookend(recipe,plan,policy_path,policy_file_sha256)
        common._json(output/'CLAIM.json',dict(recipe=window['recipe'],
            plan_sha256=plan['plan_sha256'],source_commit=plan['source_commit'],
            original_source_commit=recipe['original_source_commit'],limits=limits,test_data_used=False),
            counters,recipe['max_published_json_bytes'])
        import hashlib
        import numpy as np
        import torch
        from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
        from gx1.contracts import entry_native_direction_feature_view_v1 as views
        from gx1.contracts import entry_native_direction_research_state_v1 as old
        from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
        from gx1.contracts import market_state_time_query_fold1_fit_v1 as full_scope
        from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
        from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
        from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
        from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
        from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch, evaluate_endpoint_batch
        from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
        from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
            _dynamic_m5_technical_tree_sha256 as tree_sha)
        _need(torch.cuda.is_available(), 'CUDA_REQUIRED')
        device = torch.device('cuda',0)
        receipts = zero.load_ta_input_receipts(
            current_source_owners={name:value['sha256'] for name,value in recipe['source_owners'].items()})
        view = zero._bind_ta_view(receipts,dedicated_m5_side=True)
        view_ref = original['input_refs']['ta_candidate_view']
        _need(view == authority.campaign.read_bound_json(Path(view_ref['path']),view_ref['sha256'])
              and view['contract_sha256'] == recipe['expected_view_sha256'], 'ORIGINAL_VIEW')
        bound = load_bound_time_query_targets()
        endpoint_bound, future_bound = load_bound_five_endpoint_targets(), load_bound_future_displacement()
        prior_ref = refs['endpoint_priors']
        priors = authority.campaign.read_bound_json(Path(prior_ref['path']),prior_ref['sha256'])
        objective.require_endpoint_priors(priors,fold_id=1,ta_only=True,dedicated_m5_side=True)
        provider,_,prepared,rows,issues = runner._prepared_fold1(receipts,view,np=np)
        objective.require_m5_equal_observation_axes(rows,issues)
        order = torch.from_numpy(rows.copy()); order_sha = tree_sha(order)
        _need(order_sha == origin['original_result']['parent_order_sha256'], 'ORIGINAL_ORDER')
        axes = prepared['axes']; sl = slice(full_scope.FOLD['oof_start_row'],full_scope.FOLD['oof_end_row'])
        later_rows = np.asarray(axes['parent_rows'][sl],np.int64)
        later_issues = np.asarray(axes['decision_ns'][sl],np.int64)
        expected = endpoint_bound.mature_oof_axes(fold_id=1)
        _need(len(later_rows) == len(later_issues) == 17753 and issues[-1] < later_issues[0]
              and np.array_equal(later_rows,expected['score_parent_rows'])
              and np.array_equal(later_issues,expected['score_decision_ns']), 'LATER_AXIS')
        panel_axes = {'TRAIN':(rows,issues,TRAIN_RANGES,'fit'),
                      'LATER':(later_rows,later_issues,SCORE_RANGES,'score')}
        means_np = {}
        with np.load(refs['frozen_means']['path'],allow_pickle=False) as saved:
            _need(set(saved.files) == {'state_mu0','future_mu0'}, 'MEAN_FIELDS')
            for task in ('state','future'):
                value = saved[task+'_mu0']
                _need(value.shape == (128,) and value.dtype == np.float64
                      and np.isfinite(value).all(), 'FROZEN_MEAN_SHAPE')
                means_np[task] = value.astype(np.float32)
                _need(np.isfinite(means_np[task]).all(), 'FP32_MEAN_FINITE')
        mean_shas = {name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in means_np.items()}
        with np.load(refs['original_trace']['path'],allow_pickle=False) as saved:
            first_reference = {key:saved[key][0].copy() for key in (
                'state_raw','future_raw','state_query','parents','issues','head_grad_before_clip',
                'head_grad_before_clip_present','head_before')}
        timings['input_setup_seconds'] = time.monotonic()-started
        clock_guard(); model_started = time.monotonic()
        trainer._set_deterministic(numerical['seed'],device,numerical['precision_policy'])
        _need(trainer._GRAD_CLIP_NORM == numerical['grad_clip_norm'], 'CANONICAL_CLIP')
        counters['model_constructions'] += 1
        backbone = views.build_native_direction_research_model(
            view,prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'],device=device,
            unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(backbone).to(device)
        counters['model_construction_returns'] += 1
        view_identity = full_scope.require_checkpoint_view(
            model.state_model.feature_view_identity,recipe=original,verify_files=True,ta_sf=True)
        _need(model.research_objective['contract_sha256'] == recipe['expected_model_contract_sha256'],
              'MODEL_CONTRACT')
        model.train(); counters['optimizer_constructions'] += 1
        optimizer,_ = state.build_fold1_fit_optimizer(model,numerical_recipe=numerical)
        means = {name:torch.from_numpy(value).to(device) for name,value in means_np.items()}
        initial = common._load_state(refs['original_initial_checkpoint'],torch=torch,
            maximum=recipe['max_checkpoint_bytes'],counters=counters)
        counters['checkpoint_loads'] += 1
        control = common._load_state(refs['original_p4096_checkpoint'],torch=torch,
            maximum=recipe['max_checkpoint_bytes'],counters=counters)
        counters['checkpoint_loads'] += 1

        def restore(saved,label):
            clock_guard(); counters['strict_full_state_restore_calls'] += 1
            progress = state.restore_fold1_fit_state(saved,model,optimizer,
                numerical_recipe=numerical,feature_view_identity=view_identity,parent_order=order,
                source_commit=recipe['original_source_commit'],device=device)
            counters['strict_full_state_restore_returns'] += 1
            current = state.capture_fold1_fit_state(model,optimizer,numerical_recipe=numerical,
                feature_view_identity=view_identity,parent_order=order,source_commit=plan['source_commit'],
                device=device,**progress)
            restore_proofs[label] = original_trace._same_full_state(saved,current,
                source=recipe['original_source_commit'],tree_sha=tree_sha,model_sha=model_sha)
            return progress

        def signature():
            return dict(model=model_sha(model.state_dict()),buffers=tree_sha(dict(model.named_buffers())),
                optimizer=tree_sha(optimizer.state_dict()),rng=tree_sha(trainer._attended_session_rng_state(device=device)),
                modes={name:bool(module.training) for name,module in model.named_modules()})

        def measure(label,partition_name,checkpoint,updates):
            clock_guard(); before = signature()
            axis,clocks,ranges,partition = panel_axes[partition_name]
            selected = np.concatenate([np.arange(left,right,dtype=np.int64) for left,right in ranges])
            panel_rows,panel_issues = axis[selected].copy(),clocks[selected].copy()
            arrays = {key:np.empty(shape,dtype) for key,(shape,dtype) in runner._ta_m5_capture_layout(192).items()}
            total = None; model.eval()
            try:
                for left in range(0,192,16):
                    clock_guard()
                    raw,parents,batch_issues = runner._feature_batch(
                        prepared,view,receipts,panel_rows,panel_issues,left,left+16,np=np)
                    features = {key:torch.from_numpy(value).to(device) for key,value in raw.items()}
                    _need(counters['forwards'] < 328, 'FORWARD_BUDGET')
                    counters['forwards'] += 1; counters['evaluation_forwards'] += 1
                    report = evaluate_endpoint_batch(model,features,endpoint_bound,future_bound,parents,batch_issues,
                        fold_id=1,partition=partition,future_weight=1.,frozen_priors=priors,capture_m5_readout=True)
                    counters['forward_returns'] += 1
                    total = objective.merge_endpoint_measurements(total,report['measurement'])
                    runner._ta_m5_capture_batch(arrays,report['m5_readout'],left=left,right=left+16,
                        parents=parents,issues=batch_issues)
                    del raw,features,report
            finally:
                for name,module in model.named_modules(): module.training=before['modes'][name]
            _need(signature() == before and all(p.grad is None for p in model.parameters()), 'EVAL_FULL_STATE_CHANGED')
            measured = objective.finalize_endpoint_measurement(total)
            _need(measured['rows'] == 192 and measured['batches'] == 12, 'PANEL_COMPLETE')
            key = label+'_'+partition_name
            stream = io.BytesIO(); np.savez_compressed(stream,**arrays)
            _need(stream.tell() <= 4*1024**2, 'PANEL_CAPTURE_BYTES')
            _publish_bytes(root/(key+'_READOUT.npz'),stream.getvalue())
            capture_ref = common._hash_file(root/(key+'_READOUT.npz'))
            payload = runner._ta_measurement_payload(measured,checkpoint=checkpoint,
                model_state_sha256=before['model'],optimizer_updates=updates)
            payload.update(fixed_subset_positions=selected.tolist(),full_population_evaluated=False,
                capture=capture_ref,evaluation_state_parity=before)
            measurement_refs[key] = common._json(root/(key+'.json'),payload,counters,recipe['max_published_json_bytes'])
            measurements[key],captures[key] = measured,capture_ref
            return arrays

        def historical_parity(arrays,ref,*,measurement_ref,measured):
            payload = authority.campaign.read_bound_json(
                Path(measurement_ref['path']),measurement_ref['sha256'])
            expected = payload['measurement']
            _need(payload['capture'] == ref
                  and expected['rows'] == measured['rows'] == 192
                  and expected['batches'] == measured['batches'] == 12,
                  'HISTORICAL_MEASUREMENT_BINDING')
            tolerance = recipe['historical_panel_parity']
            result = {}
            with np.load(ref['path'],allow_pickle=False) as saved:
                for key in ('parents','issues','query_native','state_allowed','state_eligible','state_mature',
                            'future_allowed','future_eligible','future_mature'):
                    _need(np.array_equal(arrays[key],saved[key]), 'HISTORICAL_PANEL_AXES_MASKS')
                for task in ('state','future'):
                    key=task+'_activity_side_logits'
                    raw=arrays[key].astype(np.float64); prior=saved[key].astype(np.float64)
                    raw_error=float(np.max(np.abs(raw-prior)))
                    prob_error=float(np.max(np.abs(np.exp(-np.logaddexp(0.,-raw[:,0,1]))-
                                                       np.exp(-np.logaddexp(0.,-prior[:,0,1])))))
                    actual_scores = measured['means']['model'][task]['M5']['conditional_side']
                    expected_scores = expected['means']['model'][task]['M5']['conditional_side']
                    score_errors = {key:abs(actual_scores[key]-expected_scores[key])
                                    for key in ('brier','log')}
                    _need(measured['scores']['model'][task]['M5']['conditional_side']['count']
                          == expected['scores']['model'][task]['M5']['conditional_side']['count'],
                          'HISTORICAL_SCORE_SUPPORT')
                    _need(raw_error <= tolerance['raw_atol']
                          and prob_error <= tolerance['probability_atol']
                          and all(value <= tolerance['score_atol'] for value in score_errors.values()),
                          'HISTORICAL_CPU_OUTPUT_PARITY')
                    result[task] = dict(raw_max_abs_error=raw_error,
                        M5_probability_max_abs_error=prob_error,M5_score_abs_errors=score_errors)
            return result

        _need(restore(initial,'INITIAL_BASELINE') == dict(cursor=0,optimizer_updates=0,skipped_batches=0)
              and initial['model_state_sha256'] == recipe['expected_initial_model_state_sha256']
              and not optimizer.state, 'INITIAL_BASELINE')
        baseline_parity['INITIAL'] = historical_parity(
            measure('INITIAL','TRAIN',refs['original_initial_checkpoint'],0),refs['panel_initial_capture'],
            measurement_ref=refs['panel_initial_measurement'],measured=measurements['INITIAL_TRAIN'])
        measure('INITIAL','LATER',refs['original_initial_checkpoint'],0)
        initial_equivalent = True
        _need(restore(control,'CONTROL_P4096') == dict(cursor=4096,optimizer_updates=256,skipped_batches=0),
              'CONTROL_PROGRESS')
        baseline_parity['CONTROL'] = historical_parity(
            measure('CONTROL','TRAIN',refs['original_p4096_checkpoint'],256),refs['panel_p4096_capture'],
            measurement_ref=refs['panel_p4096_measurement'],measured=measurements['CONTROL_TRAIN'])
        measure('CONTROL','LATER',refs['original_p4096_checkpoint'],256)
        _need(restore(initial,'CANDIDATE_INITIAL') == dict(cursor=0,optimizer_updates=0,skipped_batches=0)
              and not optimizer.state, 'CANDIDATE_EXACT_INITIAL')
        head_params = [dict(model.named_parameters())[name] for name in original_trace.HEAD_NAMES]
        _need(np.array_equal(np.concatenate([p.detach().cpu().numpy().ravel() for p in head_params]),
                            first_reference['head_before']), 'INITIAL_HEAD_BYTES')
        # All fresh mean tensors exist before this final strict RNG restore.
        fit_started = time.monotonic()
        for left in range(0,4096,16):
            clock_guard()
            raw,parents,batch_issues = runner._feature_batch(prepared,view,receipts,rows,issues,left,left+16,np=np)
            features = {name:torch.from_numpy(value).to(device) for name,value in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            _need(counters['forwards'] < 328, 'FORWARD_BUDGET')
            counters['forwards'] += 1; counters['training_forwards'] += 1
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                report = evaluate_fit_batch(model,features,bound,future_bound,parents,batch_issues,
                    fold_id=1,pass_index=0,seed=numerical['seed'],future_weight=1.,
                    m5_equal_observation=True,fit_batch_start=left)
            counters['forward_returns'] += 1
            _need(report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                  and report['loss_contract_sha256'] == recipe['expected_loss_contract_sha256']
                  and bool(torch.isfinite(report['loss'])) and report['skip_update'] is False,
                  'CANONICAL_TRAIN_LOSS')
            loss = float(report['loss'].detach())
            counters['backwards'] += 1; report['loss'].backward(); counters['backward_returns'] += 1
            if left == 0:
                grad = np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in head_params])
                first_exact = (loss == original_trace.FIRST_LOSS
                    and np.array_equal(grad,first_reference['head_grad_before_clip'])
                    and bool(first_reference['head_grad_before_clip_present'].all())
                    and np.array_equal(parents,first_reference['parents'])
                    and np.array_equal(batch_issues,first_reference['issues'])
                    and np.array_equal(report['state_query_k'],first_reference['state_query'])
                    and all(np.array_equal(report[task+'_activity_side_logits'].detach().cpu().numpy(),
                                           first_reference[task+'_raw']) for task in ('state','future')))
                _need(first_exact, 'ORIGINAL_FIRST_LOSS_RAW_GRADIENT')
            def step():
                nonlocal cursor
                counters['optimizer_step_calls'] += 1
                norm = trainer._optimizer_step_with_finite_gradients(model=model,optimizer=optimizer)
                counters['optimizer_step_returns'] += 1
                counters['optimizer_updates'] += 1
                cursor = left+16
                return norm
            stepped = centered_head_optimizer_step(model=model,optimizer=optimizer,means=means,optimizer_step=step)
            optimizer.zero_grad(set_to_none=True)
            step_records.append(dict(start_cursor=left,end_cursor=cursor,loss=loss,
                                     centered_preclip_gradient_norm=float(stepped['gradient_norm'])))
            state._progress(cursor,counters['optimizer_updates'],0,order,numerical)
            del raw,features,report,stepped
        timings['fit_seconds'] = time.monotonic()-fit_started
        _need(cursor == 4096 and counters['optimizer_updates'] == 256, 'CANDIDATE_PREFIX_COMPLETE')
        measure('CANDIDATE','TRAIN',None,256)
        measure('CANDIDATE','LATER',None,256)
        evaluation_unchanged = all_panels_complete = True
        cells = {}
        for panel in ('TRAIN','LATER'):
            candidate = measurements['CANDIDATE_'+panel]
            for label in ('INITIAL','CONTROL'):
                runner._ta_require_matched_observations(measurements[label+'_'+panel],candidate)
            for task in ('state','future'):
                proposed=candidate['means']['model'][task]['M5']['conditional_side']
                references={label:measurements[label+'_'+panel]['means']['model'][task]['M5']['conditional_side']
                            for label in ('INITIAL','CONTROL')}
                references['PRIOR']=candidate['means']['prior'][task]['M5']['conditional_side']
                differences={label:{metric:proposed[metric]-value[metric] for metric in ('brier','log')}
                             for label,value in references.items()}
                cells[panel+'_'+task]=dict(candidate=proposed,references=references,differences=differences,
                    passed=all(v < -SCORE_MARGIN for d in differences.values() for v in d.values()))
        score_gate=dict(margin=SCORE_MARGIN,cells=cells,
            decision='PASS_ALL_FOUR_CELLS' if all(v['passed'] for v in cells.values()) else 'FAIL_COMPARISON_NO_PROMOTION')
        _need(all(p.grad is None and bool(torch.isfinite(p).all()) for p in model.parameters()),
              'CANDIDATE_FINITE_NO_PARTIAL_GRADIENTS')
        _need(state._binding(model,optimizer,numerical,view_identity) == initial['binding'],
              'CANDIDATE_PARAMETER_ORDER_AND_HYPERS')
        model_state = copy.deepcopy(model.state_dict()); buffers = copy.deepcopy(dict(model.named_buffers()))
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        old._tensor_map(model_state,model.state_dict(),'CANDIDATE_MODEL')
        old._tensor_map(buffers,dict(model.named_buffers()),'CANDIDATE_BUFFERS')
        old._optimizer_state(optimizer_state,optimizer,model,256)
        rng = copy.deepcopy(trainer._attended_session_rng_state(device=device))
        old._rng(rng,device=device,seed=numerical['seed'])
        _need(all(torch.equal(means[name].cpu(),torch.from_numpy(means_np[name])) for name in means), 'MEANS_CHANGED')
        capsule=dict(schema_version=CAPSULE_SCHEMA,source_commit=plan['source_commit'],recipe=window['recipe'],
            canonical_resume_authority=False,model_state=model_state,buffer_state=buffers,
            optimizer_state=optimizer_state,rng_state=rng,module_training={n:bool(m.training) for n,m in model.named_modules()},
            parent_order=order.clone(),parent_cursor=4096,physical_optimizer_updates=256,pending_gradients=0,
            canonical_binding_reference=initial['binding'],original_numerical_recipe_reference=numerical,
            method_state=dict(name='fixed_TRAIN_mean_M5_head_coordinates',frozen_means=refs['frozen_means'],
                means_fp32={n:v.detach().cpu().clone() for n,v in means.items()},means_fp32_sha256=mean_shas,
                forward_weight_basis='ordinary_W_b',optimizer_moment_basis='W_and_centered_bias_b_plus_W_mu',
                gradient='gw_minus_mu_times_gb_before_unchanged_global_clip',hyperparameters_unchanged=True),
            model_state_sha256=model_sha(model_state),buffer_state_sha256=tree_sha(buffers),
            optimizer_state_sha256=tree_sha(optimizer_state),rng_state_sha256=tree_sha(rng),test_data_used=False)
        capsule['evidence_sha256']=tree_sha(capsule)
        stream=io.BytesIO();torch.save(capsule,stream)
        _need(0 < stream.tell() <= recipe['max_candidate_evidence_bytes'], 'CANDIDATE_EVIDENCE_BYTES')
        clock_guard()
        _publish_bytes(root/'CANDIDATE_EVIDENCE.pt',stream.getvalue())
        counters['candidate_evidence_writes'] += 1
        candidate_ref=common._hash_file(root/'CANDIDATE_EVIDENCE.pt')
        expected_counts=dict(model_constructions=1,optimizer_constructions=1,forwards=328,
            training_forwards=256,evaluation_forwards=72,backwards=256,optimizer_step_calls=256,
            optimizer_step_returns=256,optimizer_updates=256,checkpoint_loads=2,
            strict_full_state_restore_calls=3,candidate_evidence_writes=1,checkpoint_writes=0)
        _need(all(counters[k] == v for k,v in expected_counts.items()), 'EXACT_EXECUTION_COUNTS')
        original_trace._bookend(recipe,plan,policy_path,policy_file_sha256)
        clock_guard();source_unchanged=True
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        if prepared is not None:
            try:
                from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
                base._release_pages(provider,prepared)
            except BaseException as exc:
                error=(error+'; ' if error else '')+'cleanup: '+type(exc).__name__+': '+str(exc)
        prepared=provider=model=optimizer=initial=control=None
        gc.collect()
    outcome='COMPLETE' if error is None and source_unchanged and first_exact and all_panels_complete else 'FAILED'
    payload=dict(schema_version=scope.TA_M5_CENTERED_HEAD_COORDINATES_RESULT_SCHEMA,outcome=outcome,
        decision='PASS_TA_M5_CENTERED_HEAD_COORDINATES_TECHNICAL' if outcome=='COMPLETE' else 'FAIL_TA_M5_CENTERED_HEAD_COORDINATES_TECHNICAL',
        recipe=window['recipe'],source_commit=plan['source_commit'],original_source_commit=recipe['original_source_commit'],
        parent_cursor=cursor,parent_order_sha256=order_sha,optimizer_updates=counters['optimizer_updates'],
        physical_optimizer_updates=counters['optimizer_updates'],skipped_batches=0,full_state=None,
        candidate_evidence=candidate_ref,canonical_resume_authority=False,checkpoint_writes=0,
        function_equivalent_initial=initial_equivalent,first_forward_raw_gradient_exact=first_exact,
        first_forward_gradient_parameter_names=list(original_trace.HEAD_NAMES),
        full_model_gradient_parity_claimed=False,
        control_p4096_restore_exact=bool(restore_proofs.get('CONTROL_P4096',{}).get('all_numerical_state_exact')),
        evaluation_state_unchanged=evaluation_unchanged,all_panels_complete=all_panels_complete,
        restore_proofs=restore_proofs,historical_TRAIN_output_parity=baseline_parity,
        measurement_refs=measurement_refs,captures=captures,score_gate=score_gate,
        step_records=step_records,counts=counters,timings=dict(timings,elapsed_seconds=time.monotonic()-started),
        source_unchanged=source_unchanged,error=error,test_data_used=False,promotion_authorized=False,
        limits='One fixed coordinate-method pilot. Fresh same-device INITIAL/P4096 and candidate panels; reused later TRAIN is development data. Global clip factors and Adam geometry may change. Negative comparison remains a completed technical experiment; private method-state is not a canonical checkpoint or resume authority.')
    result=common._json(root/'RESULT.json',payload,counters,recipe['max_published_json_bytes'])
    common._json(root/'TERMINAL.json',dict(status=outcome,exit_code=0 if outcome=='COMPLETE' else 1,
        result=result,recipe=window['recipe'],source_unchanged=source_unchanged,test_data_used=False,
        no_relaunch_under_this_plan=True),counters,recipe['max_published_json_bytes'])
    cursor_ref=common._json(Path(window['campaign_cursor_path']),authority.build_research_cursor(
        recipe_ref=window['recipe'],outcome=outcome,optimizer_updates=counters['optimizer_updates'],
        parent_cursor=cursor,parent_order_sha256=order_sha,full_state=None,result=result,batch_size=16),
        counters,recipe['max_published_json_bytes'])
    progress=authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=cursor_ref,optimizer_updates=counters['optimizer_updates'],parent_cursor=cursor,
        parent_order_sha256=order_sha,outcome=outcome)
    common._json(progress_path,progress,counters,recipe['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=True)
    if error is not None:raise RuntimeError(error)
    return dict(result=result,checkpoint_pointer=cursor_ref,progress_path=str(progress_path))
