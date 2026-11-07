"""One guarded fold-1 TRAIN fit followed by later TRAIN time-query OOF score.

The prefix-fit campaign validates policy, clean source, signed power guard and
physical launch before dispatching here. No invocation is started by this file.
"""
from __future__ import annotations

from collections import Counter
import gc
from pathlib import Path
import time

from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('TIME_QUERY_FOLD1_FIT_WINDOW_' + reason)


class _ScoreWindowExhausted(Exception):
    """Safe score interruption with no partial OOF publication."""


def _require_prior_cursor(prior: dict, *, order_sha: str,
                          view_identity: dict) -> bool:
    """Validate a durable fit cursor; return whether only score remains."""
    _need(prior['outcome'] == 'RESUMABLE'
          and prior['parent_order_sha256'] == order_sha
          and prior['feature_view_identity'] == view_identity
          and prior['oof_cursor'] == 0,
          'RESUME_CURSOR_IDENTITY')
    if prior['phase_state'] == 'SCORING':
        _need(prior['parent_cursor'] == scope.FIT_ROWS
              and prior['optimizer_updates'] <= scope.MAX_UPDATES
              and prior['final_train_measurement'] is not None
              and Path(prior['full_state']['path']).name
                  == 'FINAL_FOLD1_FIT_STATE.pt',
              'SCORE_ONLY_RESUME_FROZEN_FIT')
        return True
    _need(prior['phase_state'] == 'TRAINING'
          and prior['parent_cursor'] < scope.FIT_ROWS
          and prior['final_train_measurement'] is None,
          'TRAIN_RESUME_BOUNDARY')
    return False


def _window_binding(invocation: dict) -> tuple[Path, str]:
    """Use the campaign-validated launcher arguments for the window ref."""
    argv = invocation['launcher_argv']
    separator = argv.index('--')
    _need(argv[separator + 4] == '--window-policy'
          and argv[separator + 6] == '--window-policy-file-sha256',
          'WINDOW_LAUNCHER_ARGUMENTS')
    return Path(argv[separator + 5]), argv[separator + 7]


def _bookend(launch: dict, plan: dict, policy_path: Path,
             policy_sha256: str) -> None:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    _need(campaign.file_sha256(policy_path) == policy_sha256, 'WINDOW_CHANGED')
    campaign.require_clean_source(plan)
    _need(campaign.file_sha256(Path(plan['recipe']['path']))
          == plan['recipe']['sha256'], 'RECIPE_CHANGED')
    for name, ref in launch['source_owners'].items():
        _need(campaign.file_sha256(Path(ref['path'])) == ref['sha256'],
              'SOURCE_CHANGED:' + name)
    for name, ref in launch['input_refs'].items():
        _need(campaign.file_sha256(Path(ref['path'])) == ref['sha256'],
              'INPUT_CHANGED:' + name)
    _need(campaign.file_sha256(Path(launch['reference_result']['path']))
          == launch['reference_result']['sha256'], 'REFERENCE_CHANGED')


def _prepared_fold1(receipts: dict, view: dict, *, np):
    """One provider/cache setup for all fit batches; caller releases once."""
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    chain = receipts['provider_chain']
    provider = base._provider_module()
    provider_plan = chain['provider_plan']
    info = provider.validate_metadata(provider_plan)
    prepared = base._prepare_inputs(provider, provider_plan, info,
                                    chain['gc_layout'])
    try:
        base._validate_axes(receipts['mature_plan']['cohorts'],
                            prepared['axes'])
        axes = prepared['axes']
        fit_rows = np.asarray(axes['parent_rows'][:scope.FIT_ROWS],
                              dtype=np.int64).copy()
        issues = np.asarray(axes['decision_ns'][:scope.FIT_ROWS],
                            dtype=np.int64).copy()
        cohort = receipts['fold1']['cohort']
        _need(cohort['joined_fit_rows'] == scope.FIT_ROWS
              and int(issues[-1]) < scope.FOLD['train_cutoff_ns']
              and bool(np.all(issues[1:] > issues[:-1]))
              and bool(np.all(issues > 0)),
              'MATURE_CHRONOLOGICAL_FIT_AXIS')
        from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
        _need(mature._axis_sha(fit_rows) == cohort['fit_parent_rows_sha256']
              and mature._axis_sha(issues) == cohort['fit_decision_ns_sha256'],
              'FIT_AXIS_DIGESTS')
        for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            axes[name].setflags(write=False)
        for name in ('signal', 'times_ns', 'sample_positions'):
            prepared['entry_sequence_source'][name].setflags(write=False)
        for name in ('snap', 'ctx_cont', 'ctx_cat'):
            prepared['current'][name].setflags(write=False)
        return provider, provider_plan, prepared, fit_rows, issues
    except BaseException:
        base._release_pages(provider, prepared)
        raise


def _feature_batch(prepared: dict, view: dict, receipts: dict,
                   parent_rows, issue_ns, left: int, right: int, *, np):
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.market_state_time_query_batch_v1 import FEATURE_KEYS
    axes = prepared['axes']
    rows = parent_rows[left:right]
    issues = issue_ns[left:right]
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
          and raw['witness']['decision_ns'] == issues.tolist()
          and len(rows) == right - left,
          'EXACT_FEATURE_BATCH')
    return raw['features'], rows, issues


def _save_snapshot(path: Path, model, optimizer, *, numerical: dict,
                   view_identity: dict, order, cursor: int, updates: int,
                   skipped: int, source_commit: str, device, launch: dict,
                   counters: dict, state_dir: Path):
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    import torch
    captured = state.capture_fold1_fit_state(
        model, optimizer, numerical_recipe=numerical,
        feature_view_identity=view_identity, parent_order=order,
        cursor=cursor, optimizer_updates=updates,
        skipped_batches=skipped, source_commit=source_commit, device=device)
    seal = captured['contract_sha256']
    model_sha = captured['model_state_sha256']
    ref = common._save_state(
        path, captured, torch=torch, maximum=launch['max_checkpoint_bytes'],
        counters=counters)
    del captured
    total = sum(item.stat().st_size for item in state_dir.glob('*.pt'))
    _need(total <= launch['max_checkpoint_total_bytes'],
          'CHECKPOINT_TOTAL_BYTES')
    return ref, seal, model_sha


def _publish_terminal(*, work: Path, progress_path: Path,
                      window: dict, plan: dict, invocation: dict,
                      launch: dict, fold: dict, view_identity,
                      phase_state: str, outcome: str,
                      parent_cursor: int, updates: int, skipped: int,
                      order_sha: str, initial_state, initial_measurement,
                      final_train_measurement, full_state, oof_measurement,
                      oof_cursor: int, first_batch, last_batch, counts: dict,
                      timings: dict, limits: dict, error: str | None,
                      source_unchanged: bool, observed_before_failure,
                      counters: dict):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    payload = {
        'schema_version': scope.RESULT_SCHEMA,
        'outcome': outcome, 'phase_state': phase_state,
        'recipe': window['recipe'], 'fold': fold,
        'source_commit': plan['source_commit'],
        'feature_view_identity': view_identity,
        'parent_order_sha256': order_sha,
        'parent_cursor': parent_cursor,
        'consumed_batches': (parent_cursor + 15) // 16,
        'optimizer_updates': updates, 'skipped_batches': skipped,
        'oof_cursor': oof_cursor,
        'own_initial_state': initial_state,
        'initial_measurement': initial_measurement,
        'final_train_measurement': final_train_measurement,
        'full_state': full_state, 'oof_measurement': oof_measurement,
        'first_batch': first_batch, 'last_batch': last_batch,
        'counts': counts, 'timings': timings,
        'source_unchanged': source_unchanged,
        'observed_before_failure': observed_before_failure,
        'limits': limits, 'error': error,
        'test_data_used': False,
        'predictive_value_claimed': False,
        'private_weights_for_main_fit': False,
        'entry_or_exit_training': False,
    }
    result = common._json(
        work / 'RESULT.json', payload, counters,
        launch['max_published_json_bytes'])
    cursor = authority.build_prefix_fit_cursor(
        recipe_ref=window['recipe'], fold=fold,
        feature_view_identity=view_identity,
        phase_state=phase_state, outcome=outcome,
        optimizer_updates=updates, parent_cursor=parent_cursor,
        parent_order_sha256=order_sha, full_state=full_state,
        own_initial_state=initial_state,
        initial_measurement=initial_measurement,
        final_train_measurement=final_train_measurement,
        oof_cursor=oof_cursor, result=result)
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix_runner
    common._json(work / 'CURSOR.json', cursor, counters,
                 launch['max_published_json_bytes'])
    pointer = prefix_runner._latest_cursor(
        window['campaign_cursor_path'], cursor, common=common,
        counters=counters, limit=launch['max_published_json_bytes'])
    progress = authority.build_prefix_fit_progress(
        window=window, plan=plan, invocation=invocation,
        checkpoint_pointer=pointer, cursor=cursor)
    common._json(
        progress_path, progress, counters,
        launch['max_published_json_bytes'])
    authority.require_prefix_fit_progress(
        progress, plan_sha256=plan['plan_sha256'], invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True)
    if error is not None:
        raise RuntimeError('TIME_QUERY_FOLD1_FIT_FAILED:' + error)
    return {'decision': outcome, 'result': result,
            'checkpoint_pointer': pointer, 'counts': counters,
            'progress_path': str(progress_path)}


def run_market_state_time_query_fold1_fit_window(*, window: dict, plan: dict,
        invocation: dict, marker: dict, launch: dict, progress_path: Path,
        limits: dict) -> dict:
    """Execute one bounded prefix campaign window; no implicit second launch."""
    import numpy as np
    import torch
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_time_query_binding_v1 as targets
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.contracts.market_state_time_query_batch_v1 import evaluate_fit_batch
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
        _dynamic_m5_technical_tree_sha256 as tree_sha,
    )
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as authority

    _need(launch['run_scope'] == scope.SCOPE
          and launch['fold'] == scope.FOLD
          and launch['numerical_recipe'] == scope.fold1_numerical_recipe()
          and launch['reference_result'] == scope.REFERENCE_RESULT
          and launch['test_data_used'] is False
          and window['invocation_number'] == invocation['invocation_number'],
          'ADMITTED_SCOPE')
    _need(torch.cuda.is_available(), 'CUDA_REQUIRED')
    device = torch.device('cuda', 0)
    started = time.monotonic()
    deadline = started + window['max_invocation_seconds']
    numerical = launch['numerical_recipe']
    fold = launch['fold']
    number = window['invocation_number']
    output = Path(window['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    work = output / f'window-{number:04d}'
    work.mkdir()
    state_dir = output / 'states'
    state_dir.mkdir(exist_ok=True)
    counters = {key: 0 for key in (
        'model_constructions', 'optimizer_constructions', 'forward_calls',
        'backward_calls', 'optimizer_step_calls', 'optimizer_updates',
        'skipped_batches', 'staged_state_load_calls',
        'staged_state_load_returns', 'resume_state_load_calls',
        'resume_state_load_returns', 'json_bytes')}
    timings = {}
    provider = provider_plan = prepared = None
    model = optimizer = None
    view_identity = None
    initial_ref = initial_measurement = final_measurement = full_state = None
    oof_measurement = None
    cursor = updates = skipped = oof_cursor = 0
    order_sha = '0' * 64
    first_batch = last_batch = None
    outcome = 'FAILED'
    phase = 'TRAINING'
    error = None
    source_unchanged = False
    score_report = None
    score_started = False
    durable = None
    observed_before_failure = None

    try:
        tic = time.monotonic()
        _bookend(launch, plan, *_window_binding(invocation))
        common._json(
            work / 'CLAIM.json',
            {'recipe': window['recipe'], 'fold': fold,
             'source_commit': plan['source_commit'],
             'plan_sha256': plan['plan_sha256'],
             'invocation_number': number, 'limits': limits,
             'test_data_used': False},
            counters, launch['max_published_json_bytes'])
        zero_plan = zero._read_ref(scope.private.ZERO_STEP_REFS['plan'])
        receipts = zero._inspect_receipts()
        view = zero._bind_view(zero_plan, receipts, 1)
        _need(view['contract_sha256'] == scope.VIEW_SHA256,
              'FOLD1_MATURE_VIEW')
        scope._reference_report()
        bound = targets.load_bound_time_query_targets()
        provider, provider_plan, prepared, fit_rows, issues = _prepared_fold1(
            receipts, view, np=np)
        order = torch.from_numpy(fit_rows.copy())
        order_sha = tree_sha(order)
        timings['input_setup_seconds'] = time.monotonic() - tic
        tic = time.monotonic()

        trainer._set_deterministic(
            numerical['seed'], device, numerical['precision_policy'])
        model = views.build_native_direction_research_model(
            view, prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'], device=device,
            unobserved_units_profile=receipts['fold1']['profile'])
        counters['model_constructions'] += 1
        _need(model.research_objective['contract_sha256']
              == scope.MODEL_CONTRACT_SHA256
              and model.feature_view_identity['contract_sha256']
              == scope.VIEW_SHA256, 'FRESH_MODEL_BINDING')
        model.train()
        optimizer, _ = state.build_fold1_fit_optimizer(
            model, numerical_recipe=numerical)
        counters['optimizer_constructions'] += 1
        view_identity = scope.require_checkpoint_view(
            model.feature_view_identity, recipe=launch, verify_files=True)
        timings['fresh_model_seconds'] = time.monotonic() - tic

        fresh = state.capture_fold1_fit_state(
            model, optimizer, numerical_recipe=numerical,
            feature_view_identity=view_identity, parent_order=order,
            cursor=0, optimizer_updates=0, skipped_batches=0,
            source_commit=plan['source_commit'], device=device)
        if number == 1:
            initial_ref = common._save_state(
                state_dir / 'INITIAL_FOLD1_STATE.pt', fresh, torch=torch,
                maximum=launch['max_checkpoint_bytes'], counters=counters)
            initial_measurement = common._json(
                work / 'INITIAL_TRAIN.json',
                {'schema_version': 'gx1_market_state_time_query_fold1_initial_v1',
                 'fold_id': 1, 'fit_rows': scope.FIT_ROWS,
                 'view_sha256': scope.VIEW_SHA256,
                 'fresh_state_sha256': fresh['contract_sha256'],
                 'fresh_model_state_sha256': fresh['model_state_sha256'],
                 'forwards': 0, 'optimizer_updates': 0,
                 'test_data_used': False},
                counters, launch['max_published_json_bytes'])
            full_state = initial_ref
        else:
            prior_ref = {
                'path': window['campaign_cursor_path'],
                'sha256': marker['pointer_before_sha256'],
            }
            prior_raw = authority.campaign.read_bound_json(
                Path(prior_ref['path']), prior_ref['sha256'])
            prior = authority.require_prefix_fit_cursor(
                prior_raw, expected_recipe=window['recipe'],
                expected_fold=fold, verify_files=True)
            score_only_resume = _require_prior_cursor(
                prior, order_sha=order_sha, view_identity=view_identity)
            initial_ref = prior['own_initial_state']
            initial_measurement = prior['initial_measurement']
            initial_loaded = common._load_state(
                initial_ref, torch=torch,
                maximum=launch['max_checkpoint_bytes'], counters=counters)
            _need(initial_loaded['contract_sha256'] == fresh['contract_sha256'],
                  'FRESH_INITIAL_BASELINE')
            del initial_loaded
            loaded = common._load_state(
                prior['full_state'], torch=torch,
                maximum=launch['max_checkpoint_bytes'], counters=counters)
            restored = state.restore_fold1_fit_state(
                loaded, model, optimizer, numerical_recipe=numerical,
                feature_view_identity=view_identity, parent_order=order,
                source_commit=plan['source_commit'], device=device)
            cursor = restored['cursor']
            updates = restored['optimizer_updates']
            skipped = restored['skipped_batches']
            _need(cursor == prior['parent_cursor']
                  and updates == prior['optimizer_updates']
                  and (not score_only_resume
                       or (cursor == scope.FIT_ROWS
                           and Path(prior['full_state']['path']).name
                               == 'FINAL_FOLD1_FIT_STATE.pt')),
                  'RESUME_EXACT_PROGRESS')
            full_state = prior['full_state']
            final_measurement = prior['final_train_measurement']
            phase = prior['phase_state']
            del loaded
        del fresh
        durable = {'state': full_state, 'cursor': cursor,
                   'updates': updates, 'skipped': skipped,
                   'phase': phase, 'final_measurement': final_measurement}

        fit_loss_sum = 0.0
        fit_batches = 0
        per_tf_counts = Counter()
        while cursor < scope.FIT_ROWS and time.monotonic() < deadline - 120:
            left = cursor
            right = min(left + scope.BATCH_SIZE, scope.FIT_ROWS)
            raw, rows, batch_issues = _feature_batch(
                prepared, view, receipts, fit_rows, issues,
                left, right, np=np)
            features = {name: torch.from_numpy(value).to(device)
                        for name, value in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            counters['forward_calls'] += 1
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                report = evaluate_fit_batch(
                    model, features, bound, rows, batch_issues,
                    fold_id=1, pass_index=scope.PASS_INDEX,
                    seed=numerical['seed'])
            _need(report['model_contract_sha256'] == scope.MODEL_CONTRACT_SHA256
                  and report['loss_contract_sha256'] == scope.LOSS_CONTRACT_SHA256
                  and bool(torch.isfinite(report['loss']))
                  and report['skip_update'] is (report['count'] == 0),
                  'FIT_LOSS_AND_MASKS')
            record = {
                'start_cursor': left, 'end_cursor': right,
                'rows': len(rows), 'informative_cells': report['count'],
                'per_tf_counts': report['per_tf_counts'],
                'skip_update': report['skip_update'],
                'loss': float(report['loss'].detach()),
            }
            if report['skip_update']:
                skipped += 1
                counters['skipped_batches'] += 1
            else:
                counters['backward_calls'] += 1
                report['loss'].backward()
                counters['optimizer_step_calls'] += 1
                norm = trainer._optimizer_step_with_finite_gradients(
                    model=model, optimizer=optimizer)
                updates += 1
                counters['optimizer_updates'] += 1
                record['gradient_norm'] = float(norm)
            optimizer.zero_grad(set_to_none=True)
            fit_loss_sum += record['loss']
            fit_batches += 1
            per_tf_counts.update(report['per_tf_counts'])
            if first_batch is None:
                first_batch = record
            last_batch = record
            cursor = right
            state._progress(cursor, updates, skipped, order, numerical)
            _need(updates <= scope.MAX_UPDATES, 'UPDATE_CEILING')
            del report, features, raw
            if ((cursor + 15) // scope.BATCH_SIZE
                    % launch['checkpoint_every_updates'] == 0
                    and cursor < scope.FIT_ROWS):
                full_state, _, _ = _save_snapshot(
                    state_dir / f'w{number:04d}_p{cursor:06d}.pt',
                    model, optimizer, numerical=numerical,
                    view_identity=view_identity, order=order, cursor=cursor,
                    updates=updates, skipped=skipped,
                    source_commit=plan['source_commit'], device=device,
                    launch=launch, counters=counters, state_dir=state_dir)
                durable = {'state': full_state, 'cursor': cursor,
                           'updates': updates, 'skipped': skipped,
                           'phase': 'TRAINING', 'final_measurement': None}

        if cursor == scope.FIT_ROWS:
            _need(updates + skipped == scope.MAX_UPDATES
                  and ((last_batch is not None and last_batch['rows'] == 6)
                       or phase == 'SCORING'),
                  'FULL_PASS_TAIL_AND_ACCOUNTING')
            if full_state is None or Path(full_state['path']).name != 'FINAL_FOLD1_FIT_STATE.pt':
                final_state, final_seal, model_sha = _save_snapshot(
                    state_dir / 'FINAL_FOLD1_FIT_STATE.pt',
                    model, optimizer, numerical=numerical,
                    view_identity=view_identity, order=order, cursor=cursor,
                    updates=updates, skipped=skipped,
                    source_commit=plan['source_commit'], device=device,
                    launch=launch, counters=counters, state_dir=state_dir)
                full_state = final_state
                durable = {'state': full_state, 'cursor': cursor,
                           'updates': updates, 'skipped': skipped,
                           'phase': 'SCORING', 'final_measurement': None}
            else:
                final_loaded = common._load_state(
                    full_state, torch=torch,
                    maximum=launch['max_checkpoint_bytes'],
                    counters=counters)
                final_seal = final_loaded['contract_sha256']
                model_sha = final_loaded['model_state_sha256']
                del final_loaded
            if final_measurement is None:
                final_measurement = common._json(
                    work / 'FINAL_TRAIN.json',
                    {'schema_version': 'gx1_market_state_time_query_fold1_final_train_v1',
                     'fold_id': 1, 'fit_rows': scope.FIT_ROWS,
                     'consumed_batches': scope.MAX_UPDATES,
                     'optimizer_updates': updates,
                     'skipped_batches': skipped,
                     'view_sha256': scope.VIEW_SHA256,
                     'fit_checkpoint': full_state,
                     'fit_state_sha256': final_seal,
                     'fit_model_state_sha256': model_sha,
                     'observed_invocation_fit_batches': fit_batches,
                     'observed_invocation_loss_sum': fit_loss_sum,
                     'observed_invocation_per_tf_counts': dict(per_tf_counts),
                     'test_data_used': False,
                     'predictive_value_claimed': False},
                    counters, launch['max_published_json_bytes'])
            phase = 'SCORING'
            durable = {'state': full_state, 'cursor': cursor,
                       'updates': updates, 'skipped': skipped,
                       'phase': phase, 'final_measurement': final_measurement}
            base._release_pages(provider, prepared)
            provider.validate_metadata(provider_plan)
            provider = prepared = None
            del model, optimizer
            model = optimizer = None
            gc.collect()
            torch.cuda.empty_cache()
            if time.monotonic() < deadline - 3600:
                score_started = True
                from gx1.audit.score_market_state_time_query_fold1_oof_v1 import (
                    score_fold1_from_checkpoint,
                )
                def score_deadline() -> None:
                    if time.monotonic() >= deadline - 120:
                        raise _ScoreWindowExhausted()
                try:
                    score_report = score_fold1_from_checkpoint(
                        fit_checkpoint_ref=full_state,
                        expected_model_state_sha256=model_sha,
                        source_commit=plan['source_commit'],
                        device=device, check_deadline=score_deadline)
                except _ScoreWindowExhausted:
                    score_started = False
                    outcome = 'RESUMABLE'
                    phase = 'SCORING'
                _need(score_report is not None or outcome == 'RESUMABLE',
                      'SCORER_RETURNED_NO_REPORT')
                if score_report is not None:
                    _need(isinstance(score_report, dict)
                      and score_report.get('fold_id') == 1
                      and score_report.get('score_rows') == scope.SCORE_ROWS
                      and score_report.get('test_data_used') is False,
                          'COMPLETE_SCORE_REPORT')
                    oof_measurement = common._json(
                        work / 'OOF.json', score_report, counters,
                        launch['max_published_json_bytes'])
                    oof_cursor = scope.SCORE_ROWS
                    phase = 'COMPLETE'
                    outcome = 'COMPLETE'
            else:
                outcome = 'RESUMABLE'
        else:
            _need(cursor < scope.FIT_ROWS and time.monotonic() >= deadline - 120,
                  'FIT_INTERRUPTED_WITHOUT_DEADLINE')
            outcome = 'RESUMABLE'
            phase = 'TRAINING'
            full_state, _, _ = _save_snapshot(
                state_dir / f'w{number:04d}_TERMINAL_p{cursor:06d}.pt',
                model, optimizer, numerical=numerical,
                view_identity=view_identity, order=order, cursor=cursor,
                updates=updates, skipped=skipped,
                source_commit=plan['source_commit'], device=device,
                launch=launch, counters=counters, state_dir=state_dir)
            durable = {'state': full_state, 'cursor': cursor,
                       'updates': updates, 'skipped': skipped,
                       'phase': phase, 'final_measurement': None}
        _bookend(launch, plan, *_window_binding(invocation))
        source_unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ':' + str(exc)
        outcome = 'FAILED'
        observed_before_failure = {'parent_cursor': cursor,
                                   'optimizer_updates': updates,
                                   'skipped_batches': skipped}
        if durable is not None:
            full_state = durable['state']
            cursor = durable['cursor']
            updates = durable['updates']
            skipped = durable['skipped']
            phase = durable['phase']
            final_measurement = durable['final_measurement']
            oof_measurement = None
            oof_cursor = 0
        else:
            cursor = updates = skipped = oof_cursor = 0
            phase = 'TRAINING'
        if score_started and full_state is not None and cursor == scope.FIT_ROWS:
            phase = 'SCORING'
            oof_measurement = None
            oof_cursor = 0
        try:
            _bookend(launch, plan, *_window_binding(invocation))
            source_unchanged = True
        except Exception:
            source_unchanged = False
    finally:
        if prepared is not None:
            base._release_pages(provider, prepared)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    timings['elapsed_seconds'] = time.monotonic() - started
    _need(not (outcome == 'COMPLETE' and (not source_unchanged
          or oof_measurement is None or full_state is None
          or final_measurement is None)), 'FALSE_COMPLETE')
    return _publish_terminal(
        work=work, progress_path=progress_path,
        window=window, plan=plan, invocation=invocation,
        launch=launch, fold=fold, view_identity=view_identity,
        phase_state=phase, outcome=outcome,
        parent_cursor=cursor, updates=updates, skipped=skipped,
        order_sha=order_sha, initial_state=initial_ref,
        initial_measurement=initial_measurement,
        final_train_measurement=final_measurement,
        full_state=full_state, oof_measurement=oof_measurement,
        oof_cursor=oof_cursor, first_batch=first_batch,
        last_batch=last_batch, counts=counters.copy(),
        timings=timings, limits=limits, error=error,
        source_unchanged=source_unchanged,
        observed_before_failure=observed_before_failure,
        counters=counters)
