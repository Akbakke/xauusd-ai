"""Observer-only replay of the original CUDA fold-1 prefix through update 256.

Called only by the admitted native research dispatcher. Original checkpoints
are read-only; a successful result references P4096 and grants no resume right.
"""
from __future__ import annotations

import gc
import io
from pathlib import Path
import time

from gx1.contracts import market_state_native_F_private_campaign_v1 as scope

HEAD_NAMES = (
    'state_model.m5_side_head.weight', 'state_model.m5_side_head.bias',
    'future_m5_side_head.weight', 'future_m5_side_head.bias',
)
FIRST_LOSS = 2.7037248611450195
FIRST_NORM = 6.401226997375488
UPDATES = 256
ROWS = 4096


def _need(ok, reason):
    if not ok:
        raise RuntimeError('TA_M5_ORIGINAL_CUDA_PREFIX_' + reason)


def _state_identity(value):
    keys = ('source_commit', 'contract_sha256', 'model_state_sha256',
            'buffer_state_sha256', 'optimizer_state_sha256', 'rng_state_sha256',
            'cursor', 'optimizer_updates', 'skipped_batches')
    return {key: value[key] for key in keys}


def _same_full_state(expected, actual, *, source, tree_sha, model_sha):
    """Compare all sealed numerical state; new execution provenance stays new."""
    _need(expected['source_commit'] == source, 'ORIGINAL_STATE_SOURCE')
    _need(expected['contract_sha256'] == tree_sha({
        key: value for key, value in expected.items() if key != 'contract_sha256'}),
        'ORIGINAL_STATE_SEAL')
    _need(set(expected) == set(actual), 'STATE_KEYS')
    for key in ('model_state', 'buffer_state', 'optimizer_state', 'rng_state'):
        digest = model_sha if key == 'model_state' else tree_sha
        _need(digest(expected[key]) == expected[key + '_sha256']
              == digest(actual[key]) == actual[key + '_sha256'],
              'EXACT_' + key.upper())
    for key in set(expected) - {'source_commit', 'contract_sha256',
                               'model_state', 'buffer_state', 'optimizer_state', 'rng_state'}:
        _need(tree_sha(expected[key]) == tree_sha(actual[key]), 'EXACT_' + key.upper())
    return {'all_numerical_state_exact': True,
            'original': _state_identity(expected), 'replayed': _state_identity(actual),
            'compared_source_metadata': False, 'checkpoint_file_hash_parity_claimed': False}


def _bookend(recipe, plan, policy_path, policy_sha):
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _need(campaign.file_sha256(policy_path) == policy_sha, 'WINDOW_CHANGED')
    campaign.require_clean_source(plan)
    for value in (plan['recipe'], recipe['trace_worker'],
                  *recipe['source_owners'].values(), *recipe['input_refs'].values()):
        campaign.require_binding(value, label='original CUDA prefix bookend', verify_file=True)


def run_original_cuda_prefix(*, policy_path: Path, policy_file_sha256: str,
                             progress_path: Path) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common

    limits = require_guarded_cuda_trainer_execution()
    window, plan, invocation, marker, recipe = authority.require_native_direction_research_context(
        policy_path, policy_file_sha256, progress_path)
    _need(recipe['run_scope'] == scope.TA_M5_ORIGINAL_CUDA_PREFIX_SCOPE
          and recipe['max_private_updates'] == recipe['max_forwards']
          == recipe['max_backwards'] == UPDATES
          and recipe['max_model_seconds'] == 720
          and recipe['max_trace_bytes'] == 64 * 1024**2
          and recipe['max_checkpoint_loads'] == 2
          and recipe['max_full_state_restores'] == 1
          and recipe['max_checkpoint_writes'] == 0, 'ADMITTED_BOUNDS')
    origin = scope.read_ta_m5_original_cuda_prefix_origin(recipe)
    original = origin['original_recipe']
    _need(recipe['numerical_recipe'] == original['numerical_recipe']
          and recipe['numerical_recipe']['grad_accum_steps'] == 1
          and recipe['numerical_recipe']['grad_clip_norm'] == 1.0
          and recipe['numerical_recipe']['future_weight'] == 1.0
          and recipe['numerical_recipe']['m5_equal_observation'] is True,
          'UNCHANGED_NUMERICAL_RECIPE')
    refs = recipe['input_refs']
    initial_ref = refs['original_initial_checkpoint']
    target_ref = refs['original_p4096_checkpoint']
    output = Path(window['output_directory'])
    output.mkdir(parents=True)
    root = scope.TA_M5_ORIGINAL_CUDA_PREFIX_ROOT
    _need(not any((root / name).exists() for name in ('TRACE.npz', 'RESULT.json', 'TERMINAL.json')),
          'OUTPUT_EXISTS')
    counters = {key: 0 for key in (
        'model_constructions', 'model_construction_returns', 'optimizer_constructions',
        'forwards', 'forward_returns', 'backwards', 'backward_returns',
        'optimizer_step_calls', 'optimizer_updates', 'checkpoint_loads',
        'resume_state_load_calls', 'resume_state_load_returns',
        'strict_full_state_restore_calls', 'strict_full_state_restore_returns',
        'checkpoint_writes', 'extra_target_gathers', 'json_bytes')}
    started = time.monotonic()
    model_started = None
    provider = prepared = model = optimizer = initial_saved = target_saved = None
    handles, records, state_proofs = [], [], {}
    order_sha = '0' * 64
    first_step_exact = final_exact = source_unchanged = False
    trace_ref = error = None
    cursor = 0
    timings = {}
    current = {}

    def clock_guard():
        now = time.monotonic()
        _need(now - started < window['max_invocation_seconds'], 'INVOCATION_DEADLINE')
        if model_started is not None:
            _need(now - model_started < recipe['max_model_seconds'], 'MODEL_DEADLINE')

    try:
        _bookend(recipe, plan, policy_path, policy_file_sha256)
        common._json(output / 'CLAIM.json', {
            'recipe': window['recipe'], 'plan_sha256': plan['plan_sha256'],
            'source_commit': plan['source_commit'], 'original_source_commit': recipe['original_source_commit'],
            'checkpoint_inputs': [initial_ref, target_ref], 'limits': limits, 'test_data_used': False,
        }, counters, recipe['max_published_json_bytes'])
        import numpy as np
        import torch
        from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
        from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
        from gx1.contracts import entry_native_direction_feature_view_v1 as views
        from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
        from gx1.contracts import market_state_time_query_fold1_fit_v1 as full_scope
        from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
        from gx1.contracts.market_state_time_query_binding_v1 import load_bound_time_query_targets
        from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
        from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
        from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
        from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
        from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
        from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner
        from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
            _dynamic_m5_technical_tree_sha256 as tree_sha)

        _need(torch.cuda.is_available(), 'CUDA_REQUIRED')
        device, numerical = torch.device('cuda', 0), recipe['numerical_recipe']
        receipts = zero.load_ta_input_receipts(
            current_source_owners={name: value['sha256'] for name, value in recipe['source_owners'].items()})
        view = zero._bind_ta_view(receipts, dedicated_m5_side=True)
        view_ref = original['input_refs']['ta_candidate_view']
        _need(view == authority.campaign.read_bound_json(Path(view_ref['path']), view_ref['sha256'])
              and view['contract_sha256'] == recipe['expected_view_sha256'], 'EXACT_ORIGINAL_VIEW')
        bound = load_bound_time_query_targets()
        future_bound = load_bound_future_displacement()
        provider, _, prepared, rows, issues = runner._prepared_fold1(receipts, view, np=np)
        objective.require_m5_equal_observation_axes(rows, issues)
        order = torch.from_numpy(rows.copy())
        order_sha = tree_sha(order)
        _need(order_sha == origin['original_result']['parent_order_sha256'], 'ORIGINAL_PARENT_ORDER')
        timings['input_setup_seconds'] = time.monotonic() - started
        clock_guard()
        model_started = time.monotonic()
        trainer._set_deterministic(numerical['seed'], device, numerical['precision_policy'])
        _need(trainer._GRAD_CLIP_NORM == numerical['grad_clip_norm'], 'CANONICAL_CLIP')
        counters['model_constructions'] += 1
        backbone = views.build_native_direction_research_model(
            view, prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'], device=device,
            unobserved_units_profile=receipts['fold1']['profile'])
        model = NativeFutureDualModel(backbone).to(device)
        counters['model_construction_returns'] += 1
        view_identity = full_scope.require_checkpoint_view(
            model.state_model.feature_view_identity, recipe=original,
            verify_files=True, ta_sf=True)
        _need(model.research_objective['contract_sha256'] == recipe['expected_model_contract_sha256'],
              'EXACT_MODEL_CONTRACT')
        model.train()
        counters['optimizer_constructions'] += 1
        optimizer, binding = state.build_fold1_fit_optimizer(model, numerical_recipe=numerical)
        initial_saved = common._load_state(initial_ref, torch=torch,
            maximum=recipe['max_checkpoint_bytes'], counters=counters)
        counters['checkpoint_loads'] += 1
        counters['strict_full_state_restore_calls'] += 1
        restored = state.restore_fold1_fit_state(
            initial_saved, model, optimizer, numerical_recipe=numerical,
            feature_view_identity=view_identity, parent_order=order,
            source_commit=recipe['original_source_commit'], device=device)
        counters['strict_full_state_restore_returns'] += 1
        _need(restored == {'cursor': 0, 'optimizer_updates': 0, 'skipped_batches': 0}
              and initial_saved['model_state_sha256'] == recipe['expected_initial_model_state_sha256']
              and not optimizer.state, 'EXACT_ORIGINAL_INITIAL')
        fresh = state.capture_fold1_fit_state(
            model, optimizer, numerical_recipe=numerical, feature_view_identity=view_identity,
            parent_order=order, cursor=0, optimizer_updates=0, skipped_batches=0,
            source_commit=plan['source_commit'], device=device)
        state_proofs['initial'] = _same_full_state(initial_saved, fresh,
            source=recipe['original_source_commit'], tree_sha=tree_sha, model_sha=model_sha)
        del fresh
        head_params = [(name, dict(model.named_parameters())[name]) for name in HEAD_NAMES]
        _need([tuple(value.shape) for _, value in head_params] == [(1,128),(1,),(1,128),(1,)]
              and all(value.requires_grad and value.dtype == torch.float32 for _, value in head_params),
              'EXACT_HEAD_PARAMETERS')

        def vector(grad=False):
            present = np.asarray([value.grad is not None for _, value in head_params], dtype=np.bool_)
            values = [(value.grad if grad else value) for _, value in head_params]
            arrays = [(value.detach().cpu().numpy().reshape(-1).copy()
                       if value is not None else np.zeros(parameter.numel(), np.float32))
                      for value, (_, parameter) in zip(values, head_params)]
            return np.concatenate(arrays), present

        def moments():
            means, variances, steps, present = [], [], [], []
            for _, parameter in head_params:
                entry = optimizer.state.get(parameter)
                present.append(entry is not None)
                if entry is None:
                    means.append(np.zeros(parameter.numel(),np.float32))
                    variances.append(np.zeros(parameter.numel(),np.float32))
                    steps.append(0)
                else:
                    step = float(entry['step'])
                    _need(step.is_integer() and step > 0, 'OBSERVED_ADAM_STEP')
                    steps.append(int(step))
                    for key, destination in (('exp_avg',means),('exp_avg_sq',variances)):
                        value = entry[key].detach().cpu().numpy().reshape(-1).copy()
                        _need(value.dtype == np.float32 and value.size == parameter.numel()
                              and np.isfinite(value).all(), 'OBSERVED_ADAM_MOMENT')
                        destination.append(value)
            return (np.concatenate(means),np.concatenate(variances),
                    np.asarray(steps,np.int64),np.asarray(present,np.bool_))

        def head_hook(name):
            def observe(_module, inputs):
                _need(name not in current and len(inputs) == 1, 'HEAD_OBSERVER_CALLS')
                value = inputs[0].detach()
                expected = (16,1,128) if name == 'state_representation' else (16,128)
                _need(tuple(value.shape) == expected and value.dtype == torch.float32, 'HEAD_OBSERVER_SHAPE')
                current[name] = value.reshape(16,128).cpu().numpy().copy()
                return None
            return observe

        def observe_step(_optimizer, _args, _kwargs):
            _need('head_grad_after_clip' not in current, 'ONE_ADAM_OBSERVER')
            current['head_grad_after_clip'], current['head_grad_after_clip_present'] = vector(grad=True)
            return None

        handles.extend([
            model.state_model.m5_side_head.register_forward_pre_hook(head_hook('state_representation')),
            model.future_m5_side_head.register_forward_pre_hook(head_hook('future_representation')),
            optimizer.register_step_pre_hook(observe_step)])
        timings['model_and_restore_seconds'] = time.monotonic() - model_started
        fit_started = time.monotonic()
        for left in range(0, ROWS, 16):
            clock_guard()
            current.clear()
            raw, parents, clocks = runner._feature_batch(
                prepared, view, receipts, rows, issues, left, left + 16, np=np)
            features = {name: torch.from_numpy(value).to(device) for name, value in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            current['head_before'], _ = vector()
            counters['forwards'] += 1
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                report = evaluate_fit_batch(model, features, bound, future_bound, parents, clocks,
                    fold_id=1, pass_index=0, seed=numerical['seed'], future_weight=numerical['future_weight'],
                    m5_equal_observation=True, fit_batch_start=left)
            counters['forward_returns'] += 1
            _need(report['model_contract_sha256'] == recipe['expected_model_contract_sha256']
                  and report['loss_contract_sha256'] == recipe['expected_loss_contract_sha256']
                  and bool(torch.isfinite(report['loss'])) and report['skip_update'] is False,
                  'CANONICAL_LOSS')
            loss = float(report['loss'].detach())
            # Same bound objects, read again only because the report omits eligible.
            truths = {
                'state': bound.gather_fit_sampled(parents, clocks, fold_id=1, pass_index=0, seed=numerical['seed']),
                'future': future_bound.gather(parents, clocks, fold_id=1, partition='fit')}
            counters['extra_target_gathers'] += 2
            current.update(parents=parents.copy(), issues=clocks.copy(),
                state_query=report['state_query_k'].copy(),
                future_query=np.broadcast_to(report['future_native_k'],(16,5)).copy())
            _need(np.array_equal(truths['state']['query_k'],current['state_query'])
                  and np.array_equal(current['future_query'],
                      np.broadcast_to(model.state_model.query_native_k.detach().cpu().numpy(),(16,5))),
                  'OBSERVED_QUERY_PARITY')
            for task, truth in truths.items():
                _need(np.array_equal(truth['parent_rows'],parents)
                      and np.array_equal(truth['issue_ns'],clocks), 'OBSERVED_CLOCK_PARITY')
                for key in ('allowed','mature'):
                    _need(np.array_equal(truth[key],report[task+'_'+key]), 'OBSERVED_TARGET_PARITY')
                for key in ('allowed','eligible','mature'):
                    current[task+'_'+key] = truth[key].copy()
                info = truth['eligible'] & truth['mature'] & (truth['allowed'] != 7)
                known = info & np.isin(truth['allowed'],(1,4))
                _need(info.sum() == report[task+'_count']
                      and list(info.sum(0)) == [report[task+'_per_tf_counts'][tf] for tf in TIMEFRAMES],
                      'OBSERVED_SUPPORT_PARITY')
                current[task+'_known_count'] = known.sum(0)
                current[task+'_up_count'] = (known & (truth['allowed']==4)).sum(0)
                current[task+'_raw'] = report[task+'_activity_side_logits'].detach().cpu().numpy().copy()
            counters['backwards'] += 1
            report['loss'].backward()
            counters['backward_returns'] += 1
            current['head_grad_before_clip'], current['head_grad_before_clip_present'] = vector(grad=True)
            counters['optimizer_step_calls'] += 1
            norm = trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)
            counters['optimizer_updates'] += 1
            cursor = left + 16
            current['head_after'], _ = vector()
            (current['head_exp_avg_after'],current['head_exp_avg_sq_after'],
             current['head_step_after'],current['head_optimizer_state_present']) = moments()
            _need(set(current) >= {'state_representation','future_representation','head_grad_after_clip'},
                  'OBSERVER_COMPLETENESS')
            current['loss'] = np.asarray(loss,np.float64)
            current['gradient_norm'] = np.asarray(float(norm),np.float64)
            current['start_cursor'] = np.asarray(left,np.int64)
            current['end_cursor'] = np.asarray(left+16,np.int64)
            records.append(dict(current))
            optimizer.zero_grad(set_to_none=True)
            cursor = left + 16
            state._progress(cursor,counters['optimizer_updates'],0,order,numerical)
            if left == 0:
                first_step_exact = loss == FIRST_LOSS and float(norm) == FIRST_NORM
                _need(first_step_exact, 'ORIGINAL_FIRST_STEP')
            del report, features, raw, truths
        timings['fit_seconds'] = time.monotonic() - fit_started
        for handle in handles:
            handle.remove()
        handles.clear()
        clock_guard()
        _need(cursor == ROWS and counters['optimizer_updates'] == UPDATES, 'PREFIX_COMPLETE')
        target_saved = common._load_state(target_ref, torch=torch,
            maximum=recipe['max_checkpoint_bytes'], counters=counters)
        counters['checkpoint_loads'] += 1
        _need((target_saved['cursor'],target_saved['optimizer_updates'],target_saved['skipped_batches'])
              == (ROWS,UPDATES,0), 'ORIGINAL_P4096_PROGRESS')
        actual = state.capture_fold1_fit_state(
            model,optimizer,numerical_recipe=numerical,feature_view_identity=view_identity,parent_order=order,
            cursor=cursor,optimizer_updates=UPDATES,skipped_batches=0,
            source_commit=plan['source_commit'],device=device)
        state_proofs['P004096'] = _same_full_state(target_saved,actual,
            source=recipe['original_source_commit'],tree_sha=tree_sha,model_sha=model_sha)
        final_exact = True
        expected_counts = {'model_constructions':1,'optimizer_constructions':1,
            'forwards':UPDATES,'backwards':UPDATES,'optimizer_step_calls':UPDATES,
            'optimizer_updates':UPDATES,'checkpoint_loads':2,
            'strict_full_state_restore_calls':1,'checkpoint_writes':0}
        _need(all(counters[key] == value for key,value in expected_counts.items()),
              'EXACT_EXECUTION_COUNTS')
        _need(initial_saved['contract_sha256'] == tree_sha({
            k:v for k,v in initial_saved.items() if k!='contract_sha256'}), 'INITIAL_PAYLOAD_UNCHANGED')
        del actual
        _bookend(recipe,plan,policy_path,policy_file_sha256)
        clock_guard()
        source_unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        if prepared is not None:
            try:
                from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
                base._release_pages(provider,prepared)
            except BaseException as exc:
                error = (error + '; ' if error else '') + 'cleanup: ' + type(exc).__name__ + ': ' + str(exc)
        prepared = provider = model = optimizer = initial_saved = target_saved = None
        gc.collect()

    # Retain completed observed steps even on a parity failure; no weights are published.
    try:
        if records:
            import numpy as np
            from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
            arrays = {key:np.stack([record[key] for record in records]) for key in records[0]}
            _need(sum(value.nbytes for value in arrays.values()) <= recipe['max_trace_bytes'], 'TRACE_RAW_BYTES')
            stream = io.BytesIO()
            np.savez_compressed(stream,**arrays)
            _need(stream.tell() <= recipe['max_trace_bytes'], 'TRACE_FILE_BYTES')
            _publish_bytes(root/'TRACE.npz',stream.getvalue())
            trace_ref = common._hash_file(root/'TRACE.npz')
    except BaseException as exc:
        error = (error + '; ' if error else '') + 'trace publication: ' + type(exc).__name__ + ': ' + str(exc)
    outcome = 'COMPLETE' if first_step_exact and final_exact and source_unchanged and error is None else 'FAILED'
    final_ref = target_ref if outcome == 'COMPLETE' else None
    payload = {
        'schema_version':scope.TA_M5_ORIGINAL_CUDA_PREFIX_RESULT_SCHEMA,
        'outcome':outcome,
        'decision':'PASS_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE' if outcome=='COMPLETE' else 'FAIL_TA_M5_ORIGINAL_CUDA_PREFIX_TRACE',
        'recipe':window['recipe'], 'source_commit':plan['source_commit'],
        'original_source_commit':recipe['original_source_commit'], 'trace_worker':recipe['trace_worker'],
        'parent_cursor':cursor, 'optimizer_updates':counters['optimizer_updates'],
        'physical_optimizer_updates':counters['optimizer_updates'], 'skipped_batches':0,
        'parent_order_sha256':order_sha, 'initial_state':initial_ref, 'targetref':target_ref,
        'final_state':final_ref, 'first_step_exact':first_step_exact, 'final_full_state_exact':final_exact,
        'full_state_comparisons':state_proofs, 'trace':trace_ref, 'traced_steps':len(records),
        'head_parameter_names':list(HEAD_NAMES), 'head_parameter_shapes':[[1,128],[1],[1,128],[1]],
        'head_parameter_spans':[[0,128],[128,129],[129,257],[257,258]],
        'first_step_reference':{'loss':FIRST_LOSS,'gradient_norm':FIRST_NORM},
        'counts':counters, 'timings':dict(timings,elapsed_seconds=time.monotonic()-started),
        'source_unchanged':source_unchanged, 'error':error, 'limits':limits,
        'checkpoint_writes':0, 'canonical_resume_authority':False, 'test_data_used':False,
        'initial_evaluation_replayed':False, 'intermediate_evaluation_forwards':0,
        'qualification_weights_for_main_fit':False, 'predictive_value_claimed':False,
        'scope_limit':'Observer-only original CUDA prefix replay. COMPLETE requires full P4096 numerical-state parity; source metadata and serialization bytes are not claimed identical. No new checkpoint, selection, later scoring or promotion.',
    }
    result = common._json(root/'RESULT.json',payload,counters,recipe['max_published_json_bytes'])
    common._json(root/'TERMINAL.json', {
        'status':outcome,'exit_code':0 if outcome=='COMPLETE' else 1,'result':result,
        'recipe':window['recipe'],'source_unchanged':source_unchanged,'test_data_used':False,
        'no_relaunch_under_this_plan':True},counters,recipe['max_published_json_bytes'])
    cursor_ref = common._json(Path(window['campaign_cursor_path']),authority.build_research_cursor(
        recipe_ref=window['recipe'],outcome=outcome,optimizer_updates=counters['optimizer_updates'],
        parent_cursor=cursor,parent_order_sha256=order_sha,full_state=final_ref,result=result,batch_size=16),
        counters,recipe['max_published_json_bytes'])
    progress = authority.build_research_progress(window=window,plan=plan,invocation=invocation,
        checkpoint_pointer=cursor_ref,optimizer_updates=counters['optimizer_updates'],
        parent_cursor=cursor,parent_order_sha256=order_sha,outcome=outcome)
    common._json(progress_path,progress,counters,recipe['max_published_json_bytes'])
    authority.require_research_progress(progress,plan_sha256=plan['plan_sha256'],
        invocation=invocation,expected_selection_receipt_sha256=None,verify_file=True)
    if error is not None:
        raise RuntimeError(error)
    return {'result':result,'checkpoint_pointer':cursor_ref,'progress_path':str(progress_path)}
