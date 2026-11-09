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
                      payload, counters, maximum, ta_sf=False, dedicated_m5_side=False,
                      m5_equal_observation=False):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    pass_decision = ('PASS_TA_M5_EQUAL_LOSS_MECHANICAL' if m5_equal_observation else
                     'PASS_TA_M5_MECHANICAL' if dedicated_m5_side else
                     'PASS_TA_SF_MECHANICAL' if ta_sf else 'PASS_NATIVE_F_PRIVATE_RESUME')
    outcome = 'COMPLETE' if payload['decision'] == pass_decision else 'FAILED'
    _need(payload['recipe'] == window['recipe']
          and payload['test_data_used'] is False
          and (outcome != 'COMPLETE' or payload['final_state'] is not None),
          'TERMINAL_PAYLOAD')
    result_root = (scope.TA_M5_EQUAL_LOSS_ROOT if m5_equal_observation else
                   scope.TA_M5_ROOT if dedicated_m5_side else scope.TA_ROOT if ta_sf else output)
    result_path = result_root / 'RESULT.json'
    result = common._json(result_path, payload, counters, maximum)
    if ta_sf:
        common._json(result_root / 'TERMINAL.json', {
            'schema_version': (scope.TA_M5_EQUAL_LOSS_RESULT_SCHEMA if m5_equal_observation else
                               scope.TA_M5_RESULT_SCHEMA if dedicated_m5_side else scope.TA_RESULT_SCHEMA) + '_terminal',
            'status': outcome, 'exit_code': 0 if outcome == 'COMPLETE' else 1,
            'result': result, 'recipe': window['recipe'], 'test_data_used': False,
            'no_relaunch_under_this_plan': True}, counters, maximum)
    cursor = common._json(Path(window['campaign_cursor_path']),
        scope.build_cursor(recipe_ref=window['recipe'], outcome=outcome,
            optimizer_updates=counters['optimizer_updates'],
            parent_cursor=payload['parent_cursor'],
            parent_order_sha256=payload['parent_order_sha256'],
            full_state=payload['final_state'], result=result, ta_sf=ta_sf,
            dedicated_m5_side=dedicated_m5_side,
            **({'m5_equal_observation': True} if m5_equal_observation else {})),
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
                       progress_path: Path, ta_sf: bool = False,
                       dedicated_m5_side: bool = False,
                       m5_equal_observation: bool = False) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits = require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    window, plan, invocation, marker, recipe = (
        authority.require_native_direction_research_context(
            policy_path, policy_file_sha256, progress_path))
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full_scope
    expected_scope = (scope.TA_M5_EQUAL_LOSS_SCOPE if m5_equal_observation else
                      scope.TA_M5_SCOPE if dedicated_m5_side else scope.TA_SCOPE if ta_sf else scope.SCOPE)
    expected_numerical = (full_scope.ta_sf_numerical_recipe(dedicated_m5_side=dedicated_m5_side,
                          m5_equal_observation=m5_equal_observation)
                          if ta_sf else scope.numerical_recipe())
    _need(type(ta_sf) is bool and type(dedicated_m5_side) is bool
          and type(m5_equal_observation) is bool
          and (not m5_equal_observation or (ta_sf and dedicated_m5_side))
          and (not dedicated_m5_side or ta_sf) and recipe['run_scope'] == expected_scope
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
    expected_model = model_contract(ta_only=ta_sf, dedicated_m5_side=dedicated_m5_side)['contract_sha256']
    expected_loss = loss_contract(ta_only=ta_sf, dedicated_m5_side=dedicated_m5_side,
                                  m5_equal_observation=m5_equal_observation)['contract_sha256']
    _need(numerical == (full_scope.ta_sf_numerical_recipe(dedicated_m5_side=dedicated_m5_side,
                          m5_equal_observation=m5_equal_observation)
                          if ta_sf else state.numerical_recipe())
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
    migration_reports, baseline_parity, loss_rebindings = [], [], []
    imported_state = None
    legacy_model = legacy_rng = legacy_import = None
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
            view = input_owner._bind_ta_view(receipts, max_batch_rows=scope.BATCH_SIZE,
                                             dedicated_m5_side=dedicated_m5_side)
            state_bound = state_targets.load_bound_time_query_targets()
            raw_batches, batch_reports, provider, provider_plan, full_order = (
                _load_ta_train_batches(receipts, view, np=np,
                    **({'m5_equal_observation': True} if m5_equal_observation else {})))
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
        if m5_equal_observation:
            origin = recipe['imported_origin']
            _need(tree_sha(order) == origin['parent_order_sha256'], 'IMPORTED_PARENT_ORDER')
            imported_state = common._load_state(origin['imported_checkpoint'], torch=torch,
                maximum=recipe['max_checkpoint_bytes'], counters=counters)
        elif dedicated_m5_side:
            legacy_model, legacy_rng, legacy_import = _restore_m5_legacy_origin(
                origin=recipe['warmstart_origin'], receipts=receipts, order=order,
                device=device, counters=counters, maximum=recipe['max_checkpoint_bytes'])

        def fresh():
            clock_guard()
            trainer._set_deterministic(numerical['seed'], device,
                                       numerical['precision_policy'])
            counters['model_constructions'] += 1
            _need(counters['model_constructions'] <= recipe['max_model_constructions'],
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
            if dedicated_m5_side:
                _need({'state_model.m5_side_head.weight', 'future_m5_side_head.weight'}
                      <= set(binding['active_parameter_names']), 'DEDICATED_M5_HEADS_ACTIVE')
                if m5_equal_observation:
                    rebound, proof = full_state_owner.rebind_m5_equal_observation_initial_state(
                        imported_state, model, optimizer,
                        expected_model_state_sha256=origin['model_state_sha256'],
                        new_numerical_recipe=numerical,
                        feature_view_identity=model.state_model.feature_view_identity,
                        parent_order=order, source_commit=origin['source_commit'],
                        new_source_commit=plan['source_commit'], device=device)
                    _need(rebound['model_state_sha256'] == recipe['expected_initial_model_state_sha256']
                          and rebound['rng_state_sha256'] == origin['rng_state_sha256'],
                          'IMPORTED_MODEL_AND_RNG')
                    _need(not optimizer.state and all(module.training for module in model.modules()),
                          'IMPORTED_EMPTY_OPTIMIZER_AND_TRAIN_MODES')
                    loss_rebindings.append(proof)
                    del rebound
                else:
                    migration_reports.append(full_state_owner.copy_legacy_weights_to_dedicated_m5(
                        legacy_model, model, expected_legacy_model_sha256=
                        recipe['warmstart_origin']['model_state_sha256']))
                    _need(not optimizer.state and all(module.training for module in model.modules()),
                          'WARMSTART_FRESH_OPTIMIZER_AND_TRAIN_MODES')
                    trainer._restore_attended_session_rng_state(legacy_rng, device=device)
                    _need(tree_sha(trainer._attended_session_rng_state(device=device))
                          == legacy_import['rng_state_sha256'], 'WARMSTART_RNG_PARITY')
            return model, optimizer, binding

        def evaluate(model, batch, *, baseline, batch_start):
            clock_guard()
            features, rows, issue = batch
            counters['forwards'] += 1
            _need(counters['forwards'] <= scope.MAX_FORWARDS, 'FORWARD_BUDGET')
            torch.cuda.synchronize()
            began = time.monotonic()
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                def call():
                    return evaluate_fit_batch(
                        model, features, state_bound, future_bound,
                        rows, issue, fold_id=1, pass_index=scope.PASS_INDEX,
                        seed=numerical['seed'], future_weight=numerical['future_weight'],
                        **({'m5_equal_observation': True, 'fit_batch_start': batch_start}
                           if m5_equal_observation else {}))
                if baseline:
                    with torch.no_grad():
                        if dedicated_m5_side and not m5_equal_observation:
                            outcome, parity = _m5_baseline_parity(model, call,
                                atol=recipe['m5_side_parity_atol'],
                                rtol=recipe['m5_side_parity_rtol'])
                            baseline_parity.append(parity)
                        else:
                            outcome = call()
                else:
                    outcome = call()
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
            outcome, forward_seconds = evaluate(model, batch, baseline=False, batch_start=index * scope.BATCH_SIZE)
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
            m5_gradients = {}
            if dedicated_m5_side:
                for name in ('state_model.m5_side_head.weight', 'future_m5_side_head.weight'):
                    gradient = parameters[name].grad
                    _need(gradient is not None and bool(torch.isfinite(gradient).all()),
                          'DEDICATED_M5_FINITE_GRADIENT:' + name)
                    m5_gradients[name] = bool(torch.count_nonzero(gradient))
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
                    'optimizer_seconds': optimizer_seconds,
                    **({'dedicated_m5_gradient_nonzero': m5_gradients} if dedicated_m5_side else {})}

        for trajectory in ('reference', 'interrupted', 'resumed'):
            model, optimizer, binding = fresh()
            fresh_state = capture(model, optimizer, 0, 0)
            fresh_sha = fresh_state['contract_sha256']
            if initial_sha is None:
                initial_sha = fresh_sha
                model.eval()
                for batch_index, batch in enumerate(batches):
                    outcome, elapsed = evaluate(model, batch, baseline=True,
                                                batch_start=batch_index * scope.BATCH_SIZE)
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
              == counters['optimizer_constructions'] == (3 if m5_equal_observation else 4 if dedicated_m5_side else 3)
              and counters['forwards'] == counters['forward_returns'] == 6
              and counters['backwards'] == counters['backward_returns'] == 4
              and counters['optimizer_step_calls'] == counters['optimizer_updates'] == 4
              and counters['staged_state_load_calls']
                  == counters['staged_state_load_returns'] == 3
              and counters['resume_state_load_calls']
                  == counters['resume_state_load_returns'] == (2 if dedicated_m5_side else 1)
              and parent_cursor == len(scope.PARENT_ROWS) and bit_exact,
              'EXACT_PRIVATE_WORK')
        if m5_equal_observation:
            _need(len(loss_rebindings) == 3 and not migration_reports and not baseline_parity
                  and all(item == loss_rebindings[0] for item in loss_rebindings)
                  and all(any(step['dedicated_m5_gradient_nonzero'][name] for step in steps)
                          for name in ('state_model.m5_side_head.weight', 'future_m5_side_head.weight')),
                  'EQUAL_LOSS_REBINDING_AND_GRADIENTS')
        elif dedicated_m5_side:
            _need(len(migration_reports) == 3 and len(baseline_parity) == 2
                  and len({r['model_state_sha256'] for r in migration_reports}) == 1
                  and all(any(step['dedicated_m5_gradient_nonzero'][name] for step in steps)
                          for name in ('state_model.m5_side_head.weight', 'future_m5_side_head.weight')),
                  'DEDICATED_M5_MIGRATION_PARITY_AND_GRADIENTS')
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
        'schema_version': (scope.TA_M5_EQUAL_LOSS_RESULT_SCHEMA if m5_equal_observation else
                           scope.TA_M5_RESULT_SCHEMA if dedicated_m5_side else
                           scope.TA_RESULT_SCHEMA if ta_sf else scope.RESULT_SCHEMA),
        'decision': ('PASS_TA_M5_EQUAL_LOSS_MECHANICAL' if m5_equal_observation else
                     'PASS_TA_M5_MECHANICAL' if dedicated_m5_side else
                     'PASS_TA_SF_MECHANICAL' if ta_sf else 'PASS_NATIVE_F_PRIVATE_RESUME')
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
    if m5_equal_observation:
        payload.update(imported_origin=recipe['imported_origin'],
            loss_rebindings=loss_rebindings, m5_equal_observation=True,
            full_TRAIN_initialbaseline_qualified=False, native_fit_authorized=False)
    elif dedicated_m5_side:
        payload.update(warmstart_origin=recipe['warmstart_origin'],
            legacy_import=legacy_import, weight_migrations=migration_reports,
            baseline_output_parity=baseline_parity, full_TRAIN_initialbaseline_qualified=False,
            native_fit_authorized=False)
    published = _publish_terminal(
        output=output, progress_path=progress_path,
        window=window, plan=plan, invocation=invocation,
        payload=payload, counters=counters,
        maximum=recipe['max_published_json_bytes'], ta_sf=ta_sf,
        dedicated_m5_side=dedicated_m5_side,
        **({'m5_equal_observation': True} if m5_equal_observation else {}))
    previous._bookend(recipe, plan, policy_path, policy_file_sha256)
    if not passed:
        raise RuntimeError('NATIVE_F_PRIVATE_WINDOW_FAILED:' + str(error))
    return {'decision': ('PASS_TA_M5_EQUAL_LOSS_MECHANICAL' if m5_equal_observation else
                         'PASS_TA_M5_MECHANICAL' if dedicated_m5_side else
                         'PASS_TA_SF_MECHANICAL' if ta_sf else 'PASS_NATIVE_F_PRIVATE_RESUME'),
            **published, 'counts': counters}


def _load_ta_train_batches(receipts, view, *, np, m5_equal_observation=False):
    """Use the full-fit provider/order/gather for the first two actual batches."""
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as full
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    provider, provider_plan, prepared, parents, issues = full._prepared_fold1(
        receipts, view, np=np)
    try:
        _need(len(parents) == 29574 and len(issues) == 29574, 'TA_FULL_ORDER')
        if m5_equal_observation:
            from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_m5_equal_observation_axes
            require_m5_equal_observation_axes(parents, issues)
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


def run_ta_m5_mechanical(*, policy_path: Path, policy_file_sha256: str,
                         progress_path: Path) -> dict:
    return run_private_window(policy_path=policy_path,
        policy_file_sha256=policy_file_sha256, progress_path=progress_path,
        ta_sf=True, dedicated_m5_side=True)


def run_ta_m5_equal_loss_mechanical(*, policy_path: Path, policy_file_sha256: str,
                                   progress_path: Path) -> dict:
    return run_private_window(policy_path=policy_path,
        policy_file_sha256=policy_file_sha256, progress_path=progress_path,
        ta_sf=True, dedicated_m5_side=True, m5_equal_observation=True)


def _restore_m5_legacy_origin(*, origin, receipts, order, device, counters, maximum):
    """Strictly restore the original full checkpoint before any weight transfer."""
    import copy
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
        _dynamic_m5_technical_tree_sha256 as tree_sha,
    )
    old_recipe = scope._read(origin['original_recipe'], 'M5 warmstart original recipe')
    old_view = scope._read(origin['original_candidate_view'], 'M5 warmstart original view')
    numerical = full.ta_sf_numerical_recipe()
    _need(old_recipe['numerical_recipe'] == numerical
          and old_view.get('dedicated_m5_side', False) is False
          and old_view['contract_sha256'] == old_recipe['expected_view_sha256']
          and tree_sha(order) == origin['parent_order_sha256'], 'LEGACY_ORIGIN_BINDING')
    trainer._set_deterministic(numerical['seed'], device, numerical['precision_policy'])
    counters['model_constructions'] += 1
    _need(counters['model_constructions'] == 1, 'LEGACY_FIRST_CONSTRUCTION')
    base = views.build_native_direction_research_model(
        old_view, prefix_normalization=receipts['fold1']['normal'],
        constructor_metadata=receipts['constructor'], device=device,
        unobserved_units_profile=receipts['fold1']['profile'])
    model = NativeFutureDualModel(base)
    counters['model_construction_returns'] += 1
    model.train()
    optimizer, _ = state.build_fold1_fit_optimizer(model, numerical_recipe=numerical)
    counters['optimizer_constructions'] += 1
    import torch
    loaded = common._load_state(origin['original_checkpoint'], torch=torch,
                                maximum=maximum, counters=counters)
    restored = state.restore_fold1_fit_state(
        loaded, model, optimizer, numerical_recipe=numerical,
        feature_view_identity=model.state_model.feature_view_identity,
        parent_order=order, source_commit=origin['source_commit'], device=device)
    _need(restored == {key: origin[key] for key in
                      ('cursor', 'optimizer_updates', 'skipped_batches')}
          and loaded['model_state_sha256'] == origin['model_state_sha256'],
          'LEGACY_COMPLETE_FINAL_STATE')
    rng = copy.deepcopy(loaded['rng_state'])
    evidence = {
        'original_checkpoint': origin['original_checkpoint'],
        'source_commit': origin['source_commit'],
        'full_state_contract_sha256': loaded['contract_sha256'],
        'model_state_sha256': loaded['model_state_sha256'],
        'optimizer_state_sha256': loaded['optimizer_state_sha256'],
        'rng_state_sha256': loaded['rng_state_sha256'],
        'strict_full_restore': True, 'restored_progress': restored,
        'optimizer_history_transferred_to_candidate': False,
    }
    del loaded, optimizer
    return model, rng, evidence


def _m5_baseline_parity(model, forward, *, atol, rtol):
    """Compare the cloned M5 side with legacy shared outputs in one encoding."""
    import torch
    _need(model.state_model._dedicated_m5_side is True
          and type(atol) is float and atol == 2e-6
          and type(rtol) is float and rtol == 1e-6, 'M5_PARITY_BOUND')
    saved = {'state': [], 'future': []}
    def record(task):
        return lambda module, inputs, output: saved[task].append(output.detach().clone())
    handles = [
        model.state_model.query_head.register_forward_hook(record('state')),
        model.future_head.register_forward_hook(record('future')),
    ]
    try:
        outcome = forward()
    finally:
        for handle in handles:
            handle.remove()
    report = {'atol': atol, 'rtol': rtol, 'extra_native_forwards': 0, 'tasks': {}}
    for task in ('state', 'future'):
        _need(len(saved[task]) == 1, 'M5_PARITY_ONE_HEAD_CALL:' + task)
        shared = saved[task][0]
        if task == 'state':
            _need(shared.ndim == 4 and shared.shape[1] == 1, 'M5_PARITY_QUERY')
            shared = shared[:, 0]
        actual = outcome[task + '_activity_side_logits']
        _need(actual.shape == shared.shape and actual.shape[1:] == (5, 2)
              and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(shared).all()),
              'M5_PARITY_SHAPE')
        unchanged = torch.equal(actual[..., 0], shared[..., 0]) and torch.equal(
            actual[:, 1:, 1], shared[:, 1:, 1])
        side_equal = torch.allclose(actual[:, 0, 1], shared[:, 0, 1],
                                    atol=atol, rtol=rtol)
        _need(unchanged and side_equal, 'M5_CLONED_OUTPUT_PARITY:' + task)
        report['tasks'][task] = {
            'unchanged_coordinates_bit_equal': unchanged,
            'm5_side_within_fp32_tolerance': side_equal,
            'm5_side_max_abs_error': float((actual[:, 0, 1] - shared[:, 0, 1]).abs().max()),
        }
    return outcome, report


def _m5_import_score(logits, known, y):
    """Unweighted conditional-side metrics on exactly the certified rows."""
    import numpy as np
    _need(logits.ndim == known.ndim == y.ndim == 1 and logits.shape == known.shape == y.shape
          and known.dtype == np.bool_ and bool(known.any())
          and bool(np.isfinite(logits).all()), 'IMPORT_SCORE_INPUT')
    z, truth = logits[known].astype(np.float64), y[known].astype(np.float64)
    _need(bool(np.isin(truth, (0., 1.)).all()), 'IMPORT_SCORE_TRUTH')
    p = np.exp(-np.logaddexp(0., -z))
    return {'count': len(z), 'up_count': int(truth.sum()),
            'brier': float(np.mean((p - truth) ** 2)),
            'log': float(np.mean(np.logaddexp(0., z) - truth * z)),
            'mean_p': float(p.mean())}


def _publish_m5_import(*, output, progress_path, window, plan, invocation, payload, counters):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    maximum = scope.MAX_PUBLISHED_JSON_BYTES
    outcome = 'COMPLETE' if payload['decision'] == 'PASS_TA_M5_EQUAL_WEIGHT_IMPORT' else 'FAILED'
    result = common._json(scope.TA_M5_IMPORT_ROOT / 'RESULT.json', payload, counters, maximum)
    common._json(scope.TA_M5_IMPORT_ROOT / 'TERMINAL.json', {
        'status': outcome, 'exit_code': 0 if outcome == 'COMPLETE' else 1,
        'recipe': window['recipe'], 'result': result, 'test_data_used': False,
        'no_relaunch_under_this_plan': True}, counters, maximum)
    cursor = {'schema_version': scope.TA_M5_IMPORT_CURSOR_SCHEMA, 'recipe': window['recipe'],
              'outcome': outcome, 'optimizer_updates': 0, 'parent_cursor': 0,
              'parent_order_sha256': payload['parent_order_sha256'],
              'full_state': payload['final_state'], 'result': result, 'test_data_used': False}
    cursor['cursor_sha256'] = scope.campaign.canonical_sha256(cursor)
    scope.require_ta_m5_import_cursor(cursor, expected_recipe=window['recipe'])
    pointer = common._json(Path(window['campaign_cursor_path']), cursor, counters, maximum)
    progress = authority.build_research_progress(window=window, plan=plan, invocation=invocation,
        checkpoint_pointer=pointer, optimizer_updates=0, parent_cursor=0,
        parent_order_sha256=payload['parent_order_sha256'], outcome=outcome)
    common._json(progress_path, progress, counters, maximum)
    authority.require_research_progress(progress, plan_sha256=plan['plan_sha256'],
        invocation=invocation, expected_selection_receipt_sha256=None, verify_file=True)
    return {'result': result, 'checkpoint_pointer': pointer, 'progress_path': str(progress_path)}


def run_ta_m5_import(*, policy_path: Path, policy_file_sha256: str, progress_path: Path):
    """Import the fixed converged heads; six full forwards, zero updates."""
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    limits = require_guarded_cuda_trainer_execution()
    window, plan, invocation, marker, recipe = authority.require_native_direction_research_context(
        policy_path, policy_file_sha256, progress_path)
    _need(recipe['run_scope'] == scope.TA_M5_IMPORT_SCOPE
          and recipe['training_enabled'] is False and recipe['max_private_updates'] == 0,
          'IMPORT_SCOPE')
    import numpy as np
    import torch
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as input_owner
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_market_state_time_query_private_window_v1 as previous
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
        _dynamic_m5_technical_tree_sha256 as tree_sha)
    _need(torch.cuda.is_available(), 'IMPORT_CUDA')
    device = torch.device('cuda', 0)
    numerical, refs = recipe['numerical_recipe'], recipe['input_refs']
    output = Path(window['output_directory'])
    output.mkdir(parents=True)
    counters = {k: 0 for k in ('model_constructions', 'model_construction_returns',
        'optimizer_constructions', 'forwards', 'forward_returns', 'cached_head_calls',
        'backwards', 'optimizer_step_calls', 'optimizer_updates',
        'staged_state_load_calls', 'staged_state_load_returns',
        'resume_state_load_calls', 'resume_state_load_returns', 'json_bytes')}
    started = time.monotonic()
    error = None
    final_ref = None
    order_sha = '0' * 64
    legacy_evidence = migration = imported = None
    initial_model_sha = None
    full_forward_checks, cache_scores = [], {}
    bit_exact_resume = source_unchanged = False

    def clock_guard():
        _need(time.monotonic() - started < window['max_invocation_seconds'] - 120,
              'IMPORT_DEADLINE')

    def capture(model, optimizer):
        return state.capture_fold1_fit_state(model, optimizer, numerical_recipe=numerical,
            feature_view_identity=model.state_model.feature_view_identity, parent_order=order,
            cursor=0, optimizer_updates=0, skipped_batches=0,
            source_commit=plan['source_commit'], device=device)

    def fresh():
        clock_guard()
        trainer._set_deterministic(numerical['seed'], device, numerical['precision_policy'])
        counters['model_constructions'] += 1
        _need(counters['model_constructions'] <= recipe['max_model_constructions'], 'IMPORT_MODELS')
        model = NativeFutureDualModel(views.build_native_direction_research_model(
            view, prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'], device=device,
            unobserved_units_profile=receipts['fold1']['profile']))
        counters['model_construction_returns'] += 1
        model.train()
        optimizer, _ = state.build_fold1_fit_optimizer(model, numerical_recipe=numerical)
        counters['optimizer_constructions'] += 1
        _need(counters['optimizer_constructions'] <= recipe['max_optimizer_constructions']
              and not optimizer.state, 'IMPORT_EMPTY_OPTIMIZER')
        return model, optimizer

    def forward(model, batch, left, fit_cache):
        clock_guard()
        features, parents, issues = batch
        _need(np.array_equal(parents, fit_cache['parents'][left:left + len(parents)])
              and np.array_equal(issues, fit_cache['issues'][left:left + len(parents)]),
              'IMPORT_ACTUAL_CACHE_AXIS')
        query = torch.from_numpy(fit_cache['query_native'][left:left + len(parents)]).to(device)
        seen = {}
        def hook(task):
            def record(module, inputs, output):
                _need(task not in seen, 'IMPORT_SINGLE_HEAD_CALL')
                seen[task] = inputs[0].detach().clone()
            return record
        handles = [model.state_model.query_head.register_forward_hook(hook('state')),
                   model.future_head.register_forward_hook(hook('future'))]
        counters['forwards'] += 1
        _need(counters['forwards'] <= recipe['max_forwards'], 'IMPORT_FORWARD_BUDGET')
        try:
            with torch.inference_mode():
                result = model(**features, query_k=query)
        finally:
            for handle in handles:
                handle.remove()
        counters['forward_returns'] += 1
        for j, task in enumerate(('state', 'future')):
            actual = seen[task][:, 0] if task == 'state' else seen[task]
            expected = torch.from_numpy(fit_cache[f'native_{j}_representation'][left:left + len(parents)]).to(device)
            _need(torch.equal(actual, expected), 'IMPORT_REPRESENTATION_CACHE_PARITY:' + task)
        return {k: v.detach().cpu().clone() for k, v in result.items()}

    try:
        previous._bookend(recipe, plan, policy_path, policy_file_sha256)
        previous._publish_claim(output=output, window=window, plan=plan, limits=limits,
            counters=counters, maximum=recipe['max_published_json_bytes'])
        receipts = input_owner.load_ta_input_receipts(current_source_owners={
            name: ref['sha256'] for name, ref in recipe['source_owners'].items()})
        view = input_owner._bind_ta_view(receipts, max_batch_rows=16, dedicated_m5_side=True)
        _need(view['contract_sha256'] == recipe['expected_view_sha256'], 'IMPORT_VIEW')
        raw_batches, batch_reports, _, _, full_order = _load_ta_train_batches(receipts, view, np=np)
        order = torch.from_numpy(full_order)
        order_sha = tree_sha(order)
        batches = [({k: torch.from_numpy(v).to(device) for k, v in raw.items()}, rows, issues)
                   for raw, rows, issues in raw_batches]
        del raw_batches
        legacy, rng, legacy_evidence = _restore_m5_legacy_origin(
            origin=recipe['warmstart_origin'], receipts=receipts, order=order, device=device,
            counters=counters, maximum=recipe['max_checkpoint_bytes'])
        model, optimizer = fresh()
        migration = state.copy_legacy_weights_to_dedicated_m5(legacy, model,
            expected_legacy_model_sha256=recipe['warmstart_origin']['model_state_sha256'])
        del legacy
        trainer._restore_attended_session_rng_state(rng, device=device)
        initial = capture(model, optimizer)
        initial_model_sha = initial['model_state_sha256']
        _need(initial_model_sha == recipe['initial_model_state_sha256'], 'IMPORT_EXACT_INITIAL')
        with np.load(refs['fit_cache']['path'], allow_pickle=False) as fit_cache:
            model.eval()
            before = [forward(model, b, i * 16, fit_cache) for i, b in enumerate(batches)]
            model.train()
            _need(capture(model, optimizer)['contract_sha256'] == initial['contract_sha256'],
                  'IMPORT_BEFORE_MUTATION')
            with np.load(refs['equal_coefficients']['path'], allow_pickle=False) as coefficients:
                tensors = {}
                for task, prefix in (('state', 'state_model.m5_side_head'), ('future', 'future_m5_side_head')):
                    beta = coefficients[task + '_beta_fp32']
                    _need(beta.shape == (129,) and beta.dtype == np.float32, 'IMPORT_COEFFICIENT_LAYOUT')
                    tensors[prefix + '.weight'] = torch.from_numpy(beta[1:].copy().reshape(1, 128))
                    tensors[prefix + '.bias'] = torch.from_numpy(beta[:1].copy())
                imported = state.import_equal_weight_m5_side_heads(model, tensors,
                    expected_model_state_sha256=initial_model_sha,
                    coefficient_artifact=refs['equal_coefficients'], coefficient_recipe=refs['equal_plan'])
                candidate = capture(model, optimizer)
                _need(not optimizer.state and candidate['rng_state_sha256'] == initial['rng_state_sha256'],
                      'IMPORT_CANDIDATE_RNG_OPTIMIZER')
                model.eval()
                after = [forward(model, b, i * 16, fit_cache) for i, b in enumerate(batches)]
                for i, (old, new) in enumerate(zip(before, after)):
                    checks = {}
                    for task in ('state', 'future'):
                        name = task + '_activity_side_logits'
                        _need(torch.equal(old[name][..., 0], new[name][..., 0])
                              and torch.equal(old[name][:, 1:], new[name][:, 1:])
                              and torch.equal(old[task + '_log_probs'][:, 1:], new[task + '_log_probs'][:, 1:]),
                              'IMPORT_OTHER_OUTPUTS_CHANGED')
                        expected = torch.from_numpy(coefficients['fit_' + task + '_equal_observation_fp32'][i*16:(i+1)*16])
                        actual = new[name][:, 0, 1].double()
                        _need(torch.allclose(actual, expected, atol=recipe['output_parity_atol'],
                                             rtol=recipe['output_parity_rtol']), 'IMPORT_ACTUAL_READOUT_PARITY')
                        checks[task] = {'other_outputs_bit_equal': True,
                            'representation_cache_bit_equal': True,
                            'm5_max_abs_error': float((actual - expected).abs().max())}
                    full_forward_checks.append({'parent_rows': batches[i][1].tolist(), 'tasks': checks})
                # Fresh native head-module baseline on every cached row. This
                # does not claim a new full-population encoder execution.
                for partition in ('fit', 'score'):
                    cache_scores[partition] = {}
                    with np.load(refs[partition + '_cache']['path'], allow_pickle=False) as cache:
                        for j, task in enumerate(('state', 'future')):
                            clock_guard()
                            prefix = 'train_0' if partition == 'fit' and task == 'state' else f'native_{j}'
                            x = cache[prefix + '_representation'][:, 0].copy()
                            if partition == 'fit' and task == 'state':
                                _need(np.array_equal(x, cache['native_0_representation'][:, 0]),
                                      'IMPORT_S_M5_QUERY_PARITY')
                            _need(len(x) == recipe['cached_head_rows'][partition], 'IMPORT_CACHE_ROWS')
                            head = model.state_model.m5_side_head if task == 'state' else model.future_m5_side_head
                            pieces = []
                            with torch.no_grad():
                                for left in range(0, len(x), 16):
                                    pieces.append(head(torch.from_numpy(x[left:left+16]).to(device)).flatten().cpu().numpy())
                                    counters['cached_head_calls'] += 1
                            z = np.concatenate(pieces).astype(np.float64)
                            key = partition + '_' + task
                            expected = coefficients[key + '_equal_observation_fp32']
                            _need(np.allclose(z, expected, atol=recipe['output_parity_atol'],
                                               rtol=recipe['output_parity_rtol']), 'IMPORT_ALL_HEAD_PARITY')
                            allowed = cache[prefix + '_allowed'][:, 0]
                            known = cache[prefix + '_eligible'][:, 0] & cache[prefix + '_mature'][:, 0] & np.isin(allowed, (1, 4))
                            y = (allowed == 4).astype(np.float64)
                            _need(np.array_equal(known, coefficients[key + '_known'])
                                  and np.array_equal(y, coefficients[key + '_y']), 'IMPORT_SCORE_MASKS')
                            measured = _m5_import_score(z, known, y)
                            cache_scores[partition][task] = {
                                'measurement': measured,
                                'frozen_prior': _m5_import_score(coefficients[key + '_frozen_prior'], known, y),
                                'original': _m5_import_score(coefficients[key + '_original'], known, y),
                                'max_abs_cpu_reference_error': float(np.max(np.abs(z - expected))),
                                'rows': len(x), 'head_batches': len(pieces)}
                model.train()
                _need(capture(model, optimizer)['contract_sha256'] == candidate['contract_sha256'],
                      'IMPORT_AFTER_MUTATION')
                final_ref = common._save_state(output / 'IMPORTED_M5_STATE.pt', candidate,
                    torch=torch, maximum=recipe['max_checkpoint_bytes'], counters=counters)
            del model, optimizer
            gc.collect()
            resumed, resumed_optimizer = fresh()
            loaded = common._load_state(final_ref, torch=torch,
                maximum=recipe['max_checkpoint_bytes'], counters=counters)
            restored = state.restore_fold1_fit_state(loaded, resumed, resumed_optimizer,
                numerical_recipe=numerical, feature_view_identity=resumed.state_model.feature_view_identity,
                parent_order=order, source_commit=plan['source_commit'], device=device)
            _need(restored == {'cursor': 0, 'optimizer_updates': 0, 'skipped_batches': 0}
                  and capture(resumed, resumed_optimizer)['contract_sha256'] == candidate['contract_sha256'],
                  'IMPORT_CHECKPOINT_ROUNDTRIP')
            resumed.eval()
            repeated = [forward(resumed, b, i * 16, fit_cache) for i, b in enumerate(batches)]
            _need(all(torch.equal(a[k], b[k]) for a, b in zip(after, repeated) for k in a),
                  'IMPORT_RESUME_OUTPUT_PARITY')
            resumed.train()
            _need(capture(resumed, resumed_optimizer)['contract_sha256'] == candidate['contract_sha256'],
                  'IMPORT_RESUME_MUTATION')
            bit_exact_resume = True
        _need(counters['model_constructions'] == counters['model_construction_returns']
              == counters['optimizer_constructions'] == 3
              and counters['forwards'] == counters['forward_returns'] == 6
              and counters['cached_head_calls'] == 2 * (1849 + 1110)
              and counters['backwards'] == counters['optimizer_step_calls'] == counters['optimizer_updates'] == 0,
              'IMPORT_EXACT_WORK')
        previous._bookend(recipe, plan, policy_path, policy_file_sha256)
        source_unchanged = True
    except BaseException as exc:
        error = repr(exc)
    payload = {
        'schema_version': scope.TA_M5_IMPORT_RESULT_SCHEMA,
        'decision': 'PASS_TA_M5_EQUAL_WEIGHT_IMPORT' if error is None else 'FAILED',
        'recipe': window['recipe'], 'source_commit': plan['source_commit'],
        'source_unchanged': source_unchanged, 'parent_order_sha256': order_sha, 'parent_cursor': 0,
        'physical_optimizer_updates': 0, 'final_state': final_ref,
        'initial_model_state_sha256': initial_model_sha,
        'legacy_restore': legacy_evidence, 'migration': migration, 'head_import': imported,
        'fresh_full_forward_baseline': full_forward_checks, 'cached_head_baseline': cache_scores,
        'full_TRAIN_encoder_baseline_qualified': False,
        'bit_exact_resume': bit_exact_resume, 'counts': dict(counters),
        'elapsed_seconds': time.monotonic() - started, 'limits': limits,
        'native_fit_authorized': False, 'predictive_value_qualified': False,
        'test_data_used': False, 'error': error,
    }
    published = _publish_m5_import(output=output, progress_path=progress_path, window=window,
        plan=plan, invocation=invocation, payload=payload, counters=counters)
    if error is not None:
        raise RuntimeError('TA_M5_IMPORT_FAILED:' + error)
    return {'decision': payload['decision'], **published, 'counts': counters}
