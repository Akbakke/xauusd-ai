"""One guarded GPU TRAIN-only S/F AdamW and durable resume qualification.

Two prior-selected B16 batches: reference 2 updates, interrupted 1 update,
resumed 1 update. Four physical steps total. No predictor fit or OOF claim.
"""
from __future__ import annotations

import gc
from pathlib import Path
import time

from gx1.contracts import market_state_native_F_private_campaign_v1 as scope


def _need(ok, reason):
    if not ok:
        raise RuntimeError('NATIVE_F_PRIVATE_WINDOW_' + reason)


def _publish_terminal(*, output, progress_path, window, plan, invocation,
                      payload, counters, maximum, ta_sf=False):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    pass_decision = 'PASS_TA_SF_MECHANICAL' if ta_sf else 'PASS_NATIVE_F_PRIVATE_RESUME'
    outcome = 'COMPLETE' if payload['decision'] == pass_decision else 'FAILED'
    _need(payload['recipe'] == window['recipe']
          and payload['test_data_used'] is False
          and (outcome != 'COMPLETE' or payload['final_state'] is not None),
          'TERMINAL_PAYLOAD')
    result_path = (scope.TA_ROOT if ta_sf else output) / 'RESULT.json'
    result = common._json(result_path, payload, counters, maximum)
    if ta_sf:
        common._json(scope.TA_ROOT / 'TERMINAL.json', {
            'schema_version': scope.TA_RESULT_SCHEMA + '_terminal',
            'status': outcome, 'exit_code': 0 if outcome == 'COMPLETE' else 1,
            'result': result, 'recipe': window['recipe'], 'test_data_used': False,
            'no_relaunch_under_this_plan': True}, counters, maximum)
    cursor = common._json(Path(window['campaign_cursor_path']),
        scope.build_cursor(recipe_ref=window['recipe'], outcome=outcome,
            optimizer_updates=counters['optimizer_updates'],
            parent_cursor=payload['parent_cursor'],
            parent_order_sha256=payload['parent_order_sha256'],
            full_state=payload['final_state'], result=result, ta_sf=ta_sf),
        counters, maximum)
    progress = authority.build_research_progress(
        window=window, plan=plan, invocation=invocation,
        checkpoint_pointer=cursor, optimizer_updates=counters['optimizer_updates'],
        parent_cursor=payload['parent_cursor'],
        parent_order_sha256=payload['parent_order_sha256'], outcome=outcome)
    common._json(progress_path, progress, counters, maximum)
    authority.require_research_progress(
        progress, plan_sha256=plan['plan_sha256'], invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True)
    return {'result': result, 'checkpoint_pointer': cursor,
            'progress_path': str(progress_path)}


def run_private_window(*, policy_path: Path, policy_file_sha256: str,
                       progress_path: Path, ta_sf: bool = False) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits = require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    window, plan, invocation, marker, recipe = (
        authority.require_native_direction_research_context(
            policy_path, policy_file_sha256, progress_path))
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full_scope
    expected_scope = scope.TA_SCOPE if ta_sf else scope.SCOPE
    expected_numerical = full_scope.ta_sf_numerical_recipe() if ta_sf else scope.numerical_recipe()
    _need(type(ta_sf) is bool and recipe['run_scope'] == expected_scope
          and recipe['numerical_recipe'] == expected_numerical
          and (recipe.get('parent_selection') == scope.TA_PARENT_SELECTION if ta_sf
               else recipe['parent_rows'] == list(scope.PARENT_ROWS))
          and recipe['fold_id'] == 1 and recipe['test_data_used'] is False,
          'ADMITTED_SCOPE')

    import numpy as np
    import torch
    from gx1.audit import verify_market_state_time_query_v2_zero_step_v1 as zero
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_future_displacement_candidate_v1 as future
    from gx1.contracts import market_state_native_F_private_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as full_state_owner
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel, contract as model_contract
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as loss_contract
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_market_state_time_query_private_window_v1 as previous
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
        _dynamic_m5_technical_tree_sha256 as tree_sha,
    )

    _need(torch.cuda.is_available(), 'CUDA_REQUIRED')
    device = torch.device('cuda', 0)
    numerical = recipe['numerical_recipe']
    expected_model = model_contract(ta_only=ta_sf)['contract_sha256']
    expected_loss = loss_contract(ta_only=ta_sf)['contract_sha256']
    _need(numerical == (full_scope.ta_sf_numerical_recipe() if ta_sf else state.numerical_recipe())
          and recipe['expected_model_contract_sha256'] == expected_model
          and recipe['expected_loss_contract_sha256'] == expected_loss
          and scope.BATCH_SIZE == previous.scope.BATCH_SIZE,
          'SOURCE_RECIPE_BINDING')
    order = None if ta_sf else torch.tensor(scope.PARENT_ROWS, dtype=torch.int64)
    order_sha = '0' * 64 if ta_sf else tree_sha(order)
    selected_parent_rows = [] if ta_sf else list(scope.PARENT_ROWS)
    output = Path(window['output_directory'])
    output.mkdir(parents=True)
    counters = {name: 0 for name in (
        'model_constructions', 'model_construction_returns',
        'optimizer_constructions', 'forwards', 'forward_returns',
        'backwards', 'backward_returns', 'optimizer_step_calls',
        'optimizer_updates', 'staged_state_load_calls',
        'staged_state_load_returns', 'resume_state_load_calls',
        'resume_state_load_returns', 'json_bytes')}
    started = time.monotonic()
    io_before = common._io()
    initial = interrupted = final_state = None
    initial_sha = reference_sha = None
    parent_cursor = 0
    initial_baseline, steps, timings = [], [], {}
    source_unchanged = bit_exact = False
    error = None

    def clock_guard():
        _need(time.monotonic() - started < window['max_invocation_seconds'],
              'DEADLINE')

    def capture(model, optimizer, cursor, updates):
        if ta_sf:
            return full_state_owner.capture_fold1_fit_state(
                model, optimizer, numerical_recipe=numerical,
                feature_view_identity=model.state_model.feature_view_identity,
                parent_order=order, cursor=cursor, optimizer_updates=updates,
                skipped_batches=0, source_commit=plan['source_commit'], device=device)
        return state.capture(model, optimizer, recipe=numerical,
            feature_view_identity=model.state_model.feature_view_identity,
            parent_order=order, cursor=cursor, optimizer_updates=updates,
            device=device)

    try:
        previous._bookend(recipe, plan, policy_path, policy_file_sha256)
        previous._publish_claim(output=output, window=window, plan=plan,
            limits=limits, counters=counters,
            maximum=recipe['max_published_json_bytes'])
        if ta_sf:
            from gx1.audit import verify_market_state_time_query_zero_step_v1 as input_owner
            from gx1.contracts import market_state_time_query_binding_v1 as state_targets
            receipts = input_owner.load_ta_input_receipts(current_source_owners={
                path: ref['sha256'] for path, ref in recipe['source_owners'].items()})
            view = input_owner._bind_ta_view(receipts, max_batch_rows=scope.BATCH_SIZE)
            state_bound = state_targets.load_bound_time_query_targets()
            raw_batches, batch_reports, provider, provider_plan, full_order = (
                _load_ta_train_batches(receipts, view, np=np))
            order = torch.from_numpy(full_order)
            order_sha = tree_sha(order)
            selected_parent_rows = full_order[:scope.TA_SELECTED_FIT_ROWS].tolist()
        else:
            zero_plan = scope._read(scope._read(scope.ZERO_STEP_REFS['plan'],
                'native F zero-step plan')['inputs']['completed_v2_plan'],
                'completed v2 plan')
            receipts, view, state_bound = zero._inspect_inputs(zero_plan)
            raw_batches, batch_reports, provider, provider_plan = (
                previous._load_train_batches(receipts, view, np=np))
        _need(view['contract_sha256'] == recipe['expected_view_sha256'], 'FOLD1_VIEW')
        future_bound = future.load_bound_future_displacement()
        common._json(output / 'INPUT_QUALIFICATION.json',
            {'fold_id': 1, 'selection': (scope.TA_PARENT_SELECTION if ta_sf
                                        else 'fixed_chronological_train_rows_2_to_33'),
             'batch_size': 16, 'parent_order_sha256': order_sha,
             'batches': batch_reports, 'view_sha256': view['contract_sha256'],
             'label_blind_selection': True, 'test_data_used': False},
            counters, recipe['max_published_json_bytes'])
        batches = [({name: torch.from_numpy(value).to(device)
                     for name, value in raw.items()}, rows, issue)
                   for raw, rows, issue in raw_batches]
        del raw_batches
        torch.cuda.reset_peak_memory_stats(device)

        def fresh():
            clock_guard()
            trainer._set_deterministic(numerical['seed'], device,
                                       numerical['precision_policy'])
            counters['model_constructions'] += 1
            _need(counters['model_constructions'] <= scope.MAX_MODEL_CONSTRUCTIONS,
                  'MODEL_CONSTRUCTIONS')
            base = views.build_native_direction_research_model(
                view, prefix_normalization=receipts['fold1']['normal'],
                constructor_metadata=receipts['constructor'], device=device,
                unobserved_units_profile=receipts['fold1']['profile'])
            model = NativeFutureDualModel(base)
            counters['model_construction_returns'] += 1
            _need(base.feature_view_identity['contract_sha256']
                  == recipe['expected_view_sha256']
                  and model.research_objective['contract_sha256']
                  == recipe['expected_model_contract_sha256'],
                  'FRESH_MODEL_BINDING')
            model.train()
            optimizer, binding = (full_state_owner.build_fold1_fit_optimizer(
                model, numerical_recipe=numerical) if ta_sf
                else state.build_optimizer(model, recipe=numerical))
            counters['optimizer_constructions'] += 1
            _need({'state_model.query_head.weight', 'future_head.weight'}
                  <= set(binding['active_parameter_names']), 'BOTH_HEADS_ACTIVE')
            return model, optimizer, binding

        def evaluate(model, batch, *, baseline):
            clock_guard()
            features, rows, issue = batch
            counters['forwards'] += 1
            _need(counters['forwards'] <= scope.MAX_FORWARDS, 'FORWARD_BUDGET')
            torch.cuda.synchronize()
            began = time.monotonic()
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                if baseline:
                    with torch.no_grad():
                        outcome = evaluate_fit_batch(
                            model, features, state_bound, future_bound,
                            rows, issue, fold_id=1, pass_index=scope.PASS_INDEX,
                            seed=numerical['seed'], future_weight=numerical['future_weight'])
                else:
                    outcome = evaluate_fit_batch(
                        model, features, state_bound, future_bound,
                        rows, issue, fold_id=1, pass_index=scope.PASS_INDEX,
                        seed=numerical['seed'], future_weight=numerical['future_weight'])
            torch.cuda.synchronize()
            counters['forward_returns'] += 1
            _need(outcome['state_count'] > 0 and outcome['future_count'] > 0
                  and not outcome['skip_update']
                  and outcome['model_contract_sha256'] == expected_model
                  and outcome['loss_contract_sha256'] == expected_loss
                  and bool(torch.isfinite(outcome['loss'])),
                  'SUPERVISED_FINITE_LOSS')
            return outcome, time.monotonic() - began

        def update(model, optimizer, binding, batch, trajectory, index):
            optimizer.zero_grad(set_to_none=True)
            outcome, forward_seconds = evaluate(model, batch, baseline=False)
            counters['backwards'] += 1
            _need(counters['backwards'] <= scope.MAX_BACKWARDS, 'BACKWARD_BUDGET')
            torch.cuda.synchronize()
            began = time.monotonic()
            outcome['loss'].backward()
            torch.cuda.synchronize()
            backward_seconds = time.monotonic() - began
            counters['backward_returns'] += 1
            parameters = dict(model.named_parameters())
            _need(all(parameters[name].grad is None
                      for name in binding['inactive_parameter_names'])
                  and all(parameters[name].grad is not None
                          and bool(torch.isfinite(parameters[name].grad).all())
                          and bool(torch.count_nonzero(parameters[name].grad))
                          for name in ('state_model.query_head.weight',
                                       'future_head.weight',
                                       'state_model.backbone.entry_q_joint_in.weight')),
                  'ACTIVE_AND_FROZEN_GRADIENTS')
            counters['optimizer_step_calls'] += 1
            _need(counters['optimizer_step_calls'] <= scope.MAX_PRIVATE_UPDATES,
                  'OPTIMIZER_BUDGET')
            began = time.monotonic()
            norm = trainer._optimizer_step_with_finite_gradients(
                model=model, optimizer=optimizer)
            torch.cuda.synchronize()
            optimizer_seconds = time.monotonic() - began
            optimizer.zero_grad(set_to_none=True)
            counters['optimizer_updates'] += 1
            return {'trajectory': trajectory, 'logical_step': index + 1,
                    'parent_rows': batch[1].tolist(),
                    'state_count': outcome['state_count'],
                    'future_count': outcome['future_count'],
                    'state_per_tf_counts': outcome['state_per_tf_counts'],
                    'future_per_tf_counts': outcome['future_per_tf_counts'],
                    'loss': float(outcome['loss'].detach()),
                    'gradient_norm': float(norm),
                    'forward_seconds': forward_seconds,
                    'backward_seconds': backward_seconds,
                    'optimizer_seconds': optimizer_seconds}

        for trajectory in ('reference', 'interrupted', 'resumed'):
            model, optimizer, binding = fresh()
            fresh_state = capture(model, optimizer, 0, 0)
            fresh_sha = fresh_state['contract_sha256']
            if initial_sha is None:
                initial_sha = fresh_sha
                model.eval()
                for batch in batches:
                    outcome, elapsed = evaluate(model, batch, baseline=True)
                    initial_baseline.append({
                        'parent_rows': batch[1].tolist(),
                        'loss': float(outcome['loss'].detach()),
                        'state_count': outcome['state_count'],
                        'future_count': outcome['future_count'],
                        'forward_seconds': elapsed})
                model.train()
                _need(capture(model, optimizer, 0, 0)['contract_sha256']
                      == fresh_sha, 'BASELINE_CHANGED_INITIAL_STATE')
                initial = common._save_state(output / 'INITIAL_STATE.pt',
                    fresh_state, torch=torch, maximum=scope.MAX_CHECKPOINT_BYTES,
                    counters=counters)
            else:
                _need(fresh_sha == initial_sha, 'FRESH_INITIAL_STATE_MISMATCH')
            del fresh_state
            if trajectory == 'resumed':
                loaded = common._load_state(interrupted, torch=torch,
                    maximum=scope.MAX_CHECKPOINT_BYTES, counters=counters)
                if ta_sf:
                    restored = full_state_owner.restore_fold1_fit_state(
                        loaded, model, optimizer, numerical_recipe=numerical,
                        feature_view_identity=model.state_model.feature_view_identity,
                        parent_order=order, source_commit=plan['source_commit'], device=device)
                else:
                    restored = state.restore(loaded, model, optimizer,
                        recipe=numerical,
                        feature_view_identity=model.state_model.feature_view_identity,
                        parent_order=order, device=device)
                _need(restored['cursor'] == scope.BATCH_SIZE
                      and restored['optimizer_updates'] == 1
                      and capture(model, optimizer, scope.BATCH_SIZE, 1)['contract_sha256']
                          == loaded['contract_sha256'], 'SERIALIZED_RESUME')
                del loaded
            logical = (0, 1) if trajectory == 'reference' else (
                (0,) if trajectory == 'interrupted' else (1,))
            for index in logical:
                steps.append(update(model, optimizer, binding, batches[index],
                                    trajectory, index))
                parent_cursor = scope.BATCH_SIZE * (index + 1)
            saved = capture(model, optimizer, parent_cursor,
                            1 if trajectory == 'interrupted' else 2)
            if trajectory == 'reference':
                reference_sha = saved['contract_sha256']
            elif trajectory == 'interrupted':
                interrupted = common._save_state(output / 'INTERRUPTED_STEP1.pt',
                    saved, torch=torch, maximum=scope.MAX_CHECKPOINT_BYTES,
                    counters=counters)
            else:
                _need(saved['contract_sha256'] == reference_sha,
                      'BIT_EXACT_FULL_RESUME')
                final_state = common._save_state(output / 'FINAL_PRIVATE_STATE.pt',
                    saved, torch=torch, maximum=scope.MAX_CHECKPOINT_BYTES,
                    counters=counters)
                bit_exact = True
            del saved, model, optimizer, binding
            gc.collect()
            torch.cuda.empty_cache()
        _need(counters['model_constructions']
              == counters['model_construction_returns']
              == counters['optimizer_constructions'] == 3
              and counters['forwards'] == counters['forward_returns'] == 6
              and counters['backwards'] == counters['backward_returns'] == 4
              and counters['optimizer_step_calls'] == counters['optimizer_updates'] == 4
              and counters['staged_state_load_calls']
                  == counters['staged_state_load_returns'] == 3
              and counters['resume_state_load_calls']
                  == counters['resume_state_load_returns'] == 1
              and parent_cursor == len(scope.PARENT_ROWS) and bit_exact,
              'EXACT_PRIVATE_WORK')
        provider.validate_metadata(provider_plan)
        previous._bookend(recipe, plan, policy_path, policy_file_sha256)
        source_unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
        try:
            previous._bookend(recipe, plan, policy_path, policy_file_sha256)
            source_unchanged = True
        except Exception:
            source_unchanged = False
    io_after = common._io()
    gpu_peak = {'allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
                'reserved_bytes': int(torch.cuda.max_memory_reserved(device))}
    passed = bit_exact and source_unchanged and error is None
    payload = {
        'schema_version': scope.TA_RESULT_SCHEMA if ta_sf else scope.RESULT_SCHEMA,
        'decision': ('PASS_TA_SF_MECHANICAL' if ta_sf else 'PASS_NATIVE_F_PRIVATE_RESUME')
                    if passed else 'FAILED',
        'recipe': window['recipe'],
        'source_unchanged': source_unchanged,
        'initial_state': initial, 'interrupted_state': interrupted,
        'final_state': final_state,
        'initial_state_sha256': initial_sha,
        'reference_final_state_sha256': reference_sha,
        'bit_exact_resume': bit_exact,
        'initial_train_baseline': initial_baseline,
        'steps': steps,
        'counts': {key: value for key, value in counters.items()
                   if key != 'json_bytes'},
        'parent_order': selected_parent_rows,
        'parent_order_sha256': order_sha,
        'parent_cursor': parent_cursor,
        'physical_optimizer_updates': counters['optimizer_updates'],
        'elapsed_seconds': time.monotonic() - started,
        'io_before': io_before, 'io_after': io_after,
        'rchar_delta': io_after['rchar'] - io_before['rchar'],
        'peak_process_rss_bytes': io_after['peak_process_rss_bytes'],
        'gpu_peak': gpu_peak,
        'limits': limits, 'ema_created': False, 'scheduler_created': False,
        'target_model_created': False, 'private_weights_for_main_fit': False,
        'predictive_value_qualified': False, 'test_data_used': False,
        'error': error}
    if ta_sf:
        payload.update(candidate_view=recipe['input_refs']['ta_candidate_view'],
            input_audit=recipe['input_refs']['ta_input_audit'],
            fresh_initial_baseline=bool(initial is not None and len(initial_baseline) == 2),
            numerical_recipe=numerical, source_owners=recipe['source_owners'],
            parent_order_rows=full_scope.FIT_ROWS,
            parent_selection=scope.TA_PARENT_SELECTION,
            qualification_weights_for_main_fit=False)
    published = _publish_terminal(
        output=output, progress_path=progress_path,
        window=window, plan=plan, invocation=invocation,
        payload=payload, counters=counters,
        maximum=recipe['max_published_json_bytes'], ta_sf=ta_sf)
    previous._bookend(recipe, plan, policy_path, policy_file_sha256)
    if not passed:
        raise RuntimeError('NATIVE_F_PRIVATE_WINDOW_FAILED:' + str(error))
    return {'decision': 'PASS_TA_SF_MECHANICAL' if ta_sf else 'PASS_NATIVE_F_PRIVATE_RESUME',
            **published, 'counts': counters}


def _load_ta_train_batches(receipts, view, *, np):
    """Use the full-fit provider/order/gather for the first two actual batches."""
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as full
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    provider, provider_plan, prepared, parents, issues = full._prepared_fold1(
        receipts, view, np=np)
    try:
        _need(len(parents) == 29574 and len(issues) == 29574, 'TA_FULL_ORDER')
        batches, reports = [], []
        for left in (0, scope.BATCH_SIZE):
            raw, rows, issue = full._feature_batch(
                prepared, view, receipts, parents, issues, left, left + scope.BATCH_SIZE, np=np)
            batches.append((raw, rows.copy(), issue.copy()))
            reports.append({'parent_rows': rows.tolist(), 'issue_ns': issue.tolist()})
        return batches, reports, provider, provider_plan, parents.copy()
    finally:
        base._release_pages(provider, prepared)


def run_ta_sf_mechanical(*, policy_path: Path, policy_file_sha256: str,
                         progress_path: Path) -> dict:
    return run_private_window(policy_path=policy_path,
        policy_file_sha256=policy_file_sha256, progress_path=progress_path, ta_sf=True)
