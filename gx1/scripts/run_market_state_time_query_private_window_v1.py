"""One-use TRAIN-only time-query optimizer/resume probe in the native campaign.

Two fixed chronological B16 fit batches form a 2-update reference and a
1-update interrupted plus 1-update resumed trajectory. This only measures
mechanics and timing; its weights and loss are not predictor qualification.
"""
from __future__ import annotations

import gc
from pathlib import Path
import time

from gx1.contracts import market_state_time_query_private_campaign_v1 as scope


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('TIME_QUERY_PRIVATE_WINDOW_' + reason)


def _bookend(recipe: dict, plan: dict, policy_path: Path,
             policy_file_sha256: str) -> None:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _need(campaign.file_sha256(policy_path) == policy_file_sha256,
          'POLICY_CHANGED')
    campaign.require_clean_source(plan)
    for name, ref in recipe['source_owners'].items():
        _need(campaign.file_sha256(Path(ref['path'])) == ref['sha256'],
              'SOURCE_CHANGED:' + name)
    for name, ref in recipe['input_refs'].items():
        _need(campaign.file_sha256(Path(ref['path'])) == ref['sha256'],
              'INPUT_CHANGED:' + name)


def _load_train_batches(receipts: dict, view: dict, *, np):
    from gx1.audit import fit_market_state_mature_fold1_units_v1 as mature
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_batch_v1 import FEATURE_KEYS
    chain = receipts['provider_chain']
    provider = mature._provider_module()
    provider_plan = chain['provider_plan']
    info = provider.validate_metadata(provider_plan)
    prepared = mature._prepare_inputs(provider, provider_plan, info,
                                      chain['gc_layout'])
    try:
        mature._validate_axes(receipts['mature_plan']['cohorts'],
                              prepared['axes'])
        axes = prepared['axes']
        selected = np.asarray(scope.PARENT_ROWS, dtype=np.int64)
        at = np.searchsorted(axes['parent_rows'], selected)
        _need(int(at[-1]) < len(axes['parent_rows'])
              and np.array_equal(axes['parent_rows'][at], selected)
              and int(at[-1]) < receipts['fold1']['cohort']['joined_fit_rows'],
              'EXACT_FOLD1_TRAIN_ROWS')
        issue = np.asarray(axes['decision_ns'][at], dtype=np.int64)
        _need(len(issue) == 32 and bool(np.all(issue[1:] > issue[:-1])),
              'CHRONOLOGICAL_ISSUES')
        batches = []
        reports = []
        for left in (0, scope.BATCH_SIZE):
            rows = selected[left:left + scope.BATCH_SIZE]
            raw = views.gather_native_direction_feature_batch(
                view, parent_rows=axes['parent_rows'],
                bar_start_ns=axes['bar_start_ns'],
                decision_ns=axes['decision_ns'],
                entry_sequence_source=prepared['entry_sequence_source'],
                snap_x=prepared['current']['snap'],
                ctx_cont=prepared['current']['ctx_cont'],
                ctx_cat=prepared['current']['ctx_cat'],
                multi_tf_sources=prepared['cache'],
                gc_surfaces=prepared['gc']['surfaces'],
                gc_layout=prepared['gc']['layout'],
                batch_parent_rows=rows,
                unobserved_units_profile=receipts['fold1']['profile'])
            _need(set(raw['features']) == FEATURE_KEYS
                  and raw['witness']['parent_rows'] == rows.tolist()
                  and raw['witness']['decision_ns']
                      == issue[left:left + scope.BATCH_SIZE].tolist(),
                  'FEATURE_ONLY_BATCH')
            batches.append((raw['features'], rows.copy(),
                            issue[left:left + scope.BATCH_SIZE].copy()))
            reports.append({'parent_rows': rows.tolist(),
                            'issue_ns': issue[left:left + scope.BATCH_SIZE].tolist(),
                            'witness': raw['witness']})
        return batches, reports, provider, provider_plan
    finally:
        mature._release_pages(provider, prepared)



def _publish_claim(*, output: Path, window: dict, plan: dict,
                   limits: dict, counters: dict, maximum: int) -> dict:
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    return common._json(output / 'CLAIM.json',
                        {'recipe': window['recipe'],
                         'plan_sha256': plan['plan_sha256'],
                         'limits': limits, 'train_only': True},
                        counters, maximum)


def _publish_terminal(*, output: Path, progress_path: Path,
                      window: dict, plan: dict, invocation: dict,
                      payload: dict, counters: dict, maximum: int) -> dict:
    """Publish immutable result, cursor and progress for either terminal path."""
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    outcome = ('COMPLETE' if payload['decision']
               == 'PASS_PRIVATE_MARKET_STATE_TIME_QUERY_RESUME' else 'FAILED')
    _need(payload['recipe'] == window['recipe']
          and payload['test_data_used'] is False
          and (outcome != 'COMPLETE' or payload['final_state'] is not None),
          'TERMINAL_PAYLOAD')
    result = common._json(output / 'RESULT.json', payload, counters, maximum)
    cursor = common._json(
        Path(window['campaign_cursor_path']),
        scope.build_time_query_private_cursor(
            recipe_ref=window['recipe'], outcome=outcome,
            optimizer_updates=counters['optimizer_updates'],
            parent_cursor=payload['parent_cursor'],
            parent_order_sha256=payload['parent_order_sha256'],
            full_state=payload['final_state'], result=result),
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


def run_time_query_private_window(*, policy_path: Path,
                                  policy_file_sha256: str,
                                  progress_path: Path) -> dict:
    """Execute only inside the existing guarded one-use campaign invocation."""
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    limits = require_guarded_cuda_trainer_execution()
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    window, plan, invocation, marker, recipe = (
        authority.require_native_direction_research_context(
            policy_path, policy_file_sha256, progress_path))
    _need(recipe['run_scope'] == scope.SCOPE
          and recipe['numerical_recipe'] == scope.time_query_private_numerical_recipe()
          and recipe['parent_rows'] == list(scope.PARENT_ROWS)
          and recipe['fold_id'] == 1 and recipe['test_data_used'] is False,
          'ADMITTED_SCOPE')

    import numpy as np
    import torch
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_time_query_binding_v1 as targets
    from gx1.contracts import market_state_time_query_private_state_v1 as state
    from gx1.contracts.market_state_time_query_batch_v1 import evaluate_fit_batch
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
        _dynamic_m5_technical_tree_sha256 as tree_sha,
    )

    _need(torch.cuda.is_available(), 'CUDA_REQUIRED')
    device = torch.device('cuda', 0)
    numerical = recipe['numerical_recipe']
    order = torch.tensor(scope.PARENT_ROWS, dtype=torch.int64)
    order_sha = tree_sha(order)
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
    steps = []
    initial_baseline = []
    timings = {}
    source_unchanged = False
    bit_exact = False
    error = None

    def clock_guard() -> None:
        _need(time.monotonic() - started < window['max_invocation_seconds'],
              'DEADLINE')

    def capture(model, optimizer, cursor, updates):
        return state.capture_time_query_private_state(
            model, optimizer, numerical_recipe=numerical,
            feature_view_identity=model.feature_view_identity,
            parent_order=order, cursor=cursor, optimizer_updates=updates,
            device=device)

    try:
        tic = time.monotonic()
        _bookend(recipe, plan, policy_path, policy_file_sha256)
        _publish_claim(output=output, window=window, plan=plan,
                       limits=limits, counters=counters,
                       maximum=recipe['max_published_json_bytes'])
        receipts = zero._inspect_receipts()
        view = zero._bind_view(
            {'expected_view_sha256_by_fold':
             {'1': recipe['expected_view_sha256']}}, receipts, 1)
        _need(view['contract_sha256'] == recipe['expected_view_sha256'],
              'FOLD1_VIEW')
        timings['validation_seconds'] = time.monotonic() - tic
        tic = time.monotonic()
        raw_batches, batch_reports, provider, provider_plan = (
            _load_train_batches(receipts, view, np=np))
        timings['batch_gather_seconds'] = time.monotonic() - tic
        tic = time.monotonic()
        bound = targets.load_bound_time_query_targets()
        timings['target_binding_seconds'] = time.monotonic() - tic
        common._json(output / 'INPUT_QUALIFICATION.json',
                     {'fold_id': 1, 'selection': 'fixed_chronological_train_rows_2_to_33',
                      'batch_size': 16, 'parent_order_sha256': order_sha,
                      'batches': batch_reports, 'view_sha256': view['contract_sha256'],
                      'label_blind_selection': True, 'test_data_used': False},
                     counters, recipe['max_published_json_bytes'])
        batches = []
        for raw, rows, issue in raw_batches:
            batches.append(({
                name: torch.from_numpy(value).to(device)
                for name, value in raw.items()
            }, rows, issue))
        del raw_batches
        torch.cuda.reset_peak_memory_stats(device)

        def fresh():
            clock_guard()
            trainer._set_deterministic(numerical['seed'], device,
                                       numerical['precision_policy'])
            torch.cuda.synchronize()
            begin = time.monotonic()
            counters['model_constructions'] += 1
            _need(counters['model_constructions'] <= scope.MAX_MODEL_CONSTRUCTIONS,
                  'MODEL_CONSTRUCTION_BUDGET')
            model = views.build_native_direction_research_model(
                view, prefix_normalization=receipts['fold1']['normal'],
                constructor_metadata=receipts['constructor'], device=device,
                unobserved_units_profile=receipts['fold1']['profile'])
            counters['model_construction_returns'] += 1
            _need(model.feature_view_identity['contract_sha256']
                  == recipe['expected_view_sha256']
                  and model.research_objective['contract_sha256']
                  == recipe['expected_model_contract_sha256'],
                  'FRESH_MODEL_BINDING')
            model.train()
            optimizer, _ = state.build_time_query_private_optimizer(
                model, numerical_recipe=numerical)
            counters['optimizer_constructions'] += 1
            torch.cuda.synchronize()
            timings.setdefault('model_construction_seconds', []).append(
                time.monotonic() - begin)
            return model, optimizer

        def evaluate(model, batch, *, baseline: bool):
            clock_guard()
            features, rows, issue = batch
            counters['forwards'] += 1
            _need(counters['forwards'] <= scope.MAX_FORWARDS,
                  'FORWARD_BUDGET')
            torch.cuda.synchronize()
            begin = time.monotonic()
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                if baseline:
                    with torch.no_grad():
                        outcome = evaluate_fit_batch(
                            model, features, bound, rows, issue, fold_id=1,
                            pass_index=scope.PASS_INDEX, seed=numerical['seed'])
                else:
                    outcome = evaluate_fit_batch(
                        model, features, bound, rows, issue, fold_id=1,
                        pass_index=scope.PASS_INDEX, seed=numerical['seed'])
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            counters['forward_returns'] += 1
            _need(outcome['count'] > 0 and outcome['skip_update'] is False
                  and outcome['model_contract_sha256']
                      == recipe['expected_model_contract_sha256']
                  and outcome['loss_contract_sha256']
                      == recipe['expected_loss_contract_sha256']
                  and bool(torch.isfinite(outcome['loss'])),
                  'SUPERVISED_FINITE_LOSS')
            return outcome, elapsed

        def update(model, optimizer, batch, trajectory, index):
            optimizer.zero_grad(set_to_none=True)
            outcome, forward_seconds = evaluate(model, batch, baseline=False)
            counters['backwards'] += 1
            _need(counters['backwards'] <= scope.MAX_BACKWARDS,
                  'BACKWARD_BUDGET')
            torch.cuda.synchronize()
            begin = time.monotonic()
            outcome['loss'].backward()
            torch.cuda.synchronize()
            backward_seconds = time.monotonic() - begin
            counters['backward_returns'] += 1
            counters['optimizer_step_calls'] += 1
            _need(counters['optimizer_step_calls'] <= scope.MAX_PRIVATE_UPDATES,
                  'OPTIMIZER_BUDGET')
            torch.cuda.synchronize()
            begin = time.monotonic()
            norm = trainer._optimizer_step_with_finite_gradients(
                model=model, optimizer=optimizer)
            torch.cuda.synchronize()
            step_seconds = time.monotonic() - begin
            optimizer.zero_grad(set_to_none=True)
            counters['optimizer_updates'] += 1
            parent_rows = batch[1]
            return {'trajectory': trajectory, 'logical_step': index + 1,
                    'parent_rows': parent_rows.tolist(),
                    'count': outcome['count'],
                    'per_tf_counts': outcome['per_tf_counts'],
                    'loss': float(outcome['loss'].detach()),
                    'gradient_norm': float(norm),
                    'forward_seconds': forward_seconds,
                    'backward_seconds': backward_seconds,
                    'optimizer_seconds': step_seconds}

        for trajectory in ('reference', 'interrupted', 'resumed'):
            parent_cursor = 0
            model, optimizer = fresh()
            fresh_state = capture(model, optimizer, 0, 0)
            fresh_sha = fresh_state['contract_sha256']
            if initial_sha is None:
                initial_sha = fresh_sha
                model.eval()
                for batch in batches:
                    outcome, seconds = evaluate(model, batch, baseline=True)
                    initial_baseline.append({
                        'parent_rows': batch[1].tolist(),
                        'loss': float(outcome['loss'].detach()),
                        'count': outcome['count'],
                        'per_tf_counts': outcome['per_tf_counts'],
                        'forward_seconds': seconds})
                model.train()
                _need(capture(model, optimizer, 0, 0)['contract_sha256']
                      == fresh_sha, 'BASELINE_CHANGED_INITIAL_STATE')
                tic = time.monotonic()
                initial = common._save_state(
                    output / 'INITIAL_STATE.pt', fresh_state,
                    torch=torch, maximum=recipe['max_checkpoint_bytes'],
                    counters=counters)
                timings.setdefault('checkpoint_write_seconds', []).append(
                    time.monotonic() - tic)
            else:
                _need(fresh_sha == initial_sha, 'FRESH_INITIAL_STATE_MISMATCH')
            del fresh_state
            if trajectory == 'resumed':
                tic = time.monotonic()
                loaded = common._load_state(
                    interrupted, torch=torch,
                    maximum=recipe['max_checkpoint_bytes'], counters=counters)
                timings['checkpoint_read_seconds'] = time.monotonic() - tic
                restored = state.restore_time_query_private_state(
                    loaded, model, optimizer, numerical_recipe=numerical,
                    feature_view_identity=model.feature_view_identity,
                    parent_order=order, device=device)
                _need(restored['cursor'] == 16
                      and restored['optimizer_updates'] == 1
                      and capture(model, optimizer, 16, 1)['contract_sha256']
                          == loaded['contract_sha256'],
                      'SERIALIZED_RESUME')
                del loaded
                parent_cursor = 16
            logical = (0, 1) if trajectory == 'reference' else (
                (0,) if trajectory == 'interrupted' else (1,))
            for index in logical:
                steps.append(update(model, optimizer, batches[index],
                                    trajectory, index))
                parent_cursor = 16 * (index + 1)
            saved = capture(model, optimizer, parent_cursor,
                            1 if trajectory == 'interrupted' else 2)
            if trajectory == 'reference':
                reference_sha = saved['contract_sha256']
            elif trajectory == 'interrupted':
                tic = time.monotonic()
                interrupted = common._save_state(
                    output / 'INTERRUPTED_STEP1.pt', saved,
                    torch=torch, maximum=recipe['max_checkpoint_bytes'],
                    counters=counters)
                timings.setdefault('checkpoint_write_seconds', []).append(
                    time.monotonic() - tic)
            else:
                _need(saved['contract_sha256'] == reference_sha,
                      'BIT_EXACT_FULL_RESUME')
                tic = time.monotonic()
                final_state = common._save_state(
                    output / 'FINAL_PRIVATE_STATE.pt', saved,
                    torch=torch, maximum=recipe['max_checkpoint_bytes'],
                    counters=counters)
                timings.setdefault('checkpoint_write_seconds', []).append(
                    time.monotonic() - tic)
                bit_exact = True
            del saved, model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
        _need(counters['model_constructions']
              == counters['model_construction_returns']
              == counters['optimizer_constructions'] == 3
              and counters['forwards'] == counters['forward_returns'] == 6
              and counters['backwards'] == counters['backward_returns'] == 4
              and counters['optimizer_step_calls']
                  == counters['optimizer_updates'] == 4
              and counters['staged_state_load_calls']
                  == counters['staged_state_load_returns'] == 3
              and counters['resume_state_load_calls']
                  == counters['resume_state_load_returns'] == 1
              and parent_cursor == 32 and bit_exact,
              'EXACT_PRIVATE_WORK')
        provider.validate_metadata(provider_plan)
        _bookend(recipe, plan, policy_path, policy_file_sha256)
        source_unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
        try:
            _bookend(recipe, plan, policy_path, policy_file_sha256)
            source_unchanged = True
        except Exception:
            source_unchanged = False
    io_after = common._io()
    gpu_peak = {'allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
                'reserved_bytes': int(torch.cuda.max_memory_reserved(device))}
    passed = bit_exact and source_unchanged and error is None
    payload = {
        'schema_version': scope.RESULT_SCHEMA,
        'decision': ('PASS_PRIVATE_MARKET_STATE_TIME_QUERY_RESUME'
                     if passed else 'FAILED'),
        'recipe': window['recipe'],
        'source_unchanged': source_unchanged,
        'initial_state': initial, 'interrupted_state': interrupted,
        'final_state': final_state,
        'initial_state_sha256': initial_sha,
        'reference_final_state_sha256': reference_sha,
        'bit_exact_resume': bit_exact,
        'initial_train_baseline': initial_baseline,
        'steps': steps, 'counts': {key: value for key, value in counters.items()
                                  if key != 'json_bytes'},
        'parent_order': list(scope.PARENT_ROWS),
        'parent_order_sha256': order_sha,
        'parent_cursor': parent_cursor,
        'physical_optimizer_updates': counters['optimizer_updates'],
        'timings': timings, 'elapsed_seconds': time.monotonic() - started,
        'io_before': io_before, 'io_after': io_after,
        'rchar_delta': io_after['rchar'] - io_before['rchar'],
        'peak_process_rss_bytes': io_after['peak_process_rss_bytes'],
        'gpu_peak': gpu_peak,
        'limits': limits, 'ema_created': False, 'scheduler_created': False,
        'target_model_created': False, 'private_weights_for_main_fit': False,
        'predictive_value_qualified': False, 'test_data_used': False,
        'error': error}
    published = _publish_terminal(
        output=output, progress_path=progress_path,
        window=window, plan=plan, invocation=invocation,
        payload=payload, counters=counters,
        maximum=recipe['max_published_json_bytes'])
    _bookend(recipe, plan, policy_path, policy_file_sha256)
    if not passed:
        raise RuntimeError('TIME_QUERY_PRIVATE_WINDOW_FAILED:' + str(error))
    return {'decision': 'PASS_PRIVATE_MARKET_STATE_TIME_QUERY_RESUME',
            **published, 'counts': counters}
