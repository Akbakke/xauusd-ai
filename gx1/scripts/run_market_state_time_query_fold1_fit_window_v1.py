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
                          view_identity: dict, ta_sf: bool = False) -> bool:
    """Validate a durable fit cursor; return whether only score remains."""
    _need(prior['outcome'] == 'RESUMABLE'
          and prior['parent_order_sha256'] == order_sha
          and prior['feature_view_identity'] == view_identity
          and prior['oof_cursor'] == 0,
          'RESUME_CURSOR_IDENTITY')
    if prior['phase_state'] == 'SCORING':
        _need(prior['parent_cursor'] == scope.FIT_ROWS
              and prior['optimizer_updates'] <= scope.MAX_UPDATES
              and (ta_sf or prior['final_train_measurement'] is not None)
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
    if launch['run_scope'] in (scope.TA_M5_EQUAL_LOSS_FIT_SCOPE,
                               scope.TA_M5_NEXT_NATIVE_PASS_SCOPE):
        campaign.require_binding(launch['imported_origin']['imported_checkpoint'],
            label='M5 equal-loss original import', verify_file=True)
    if launch['run_scope'] in (scope.TA_RECOVERY_SCOPE, scope.TA_M5_FIT_SCOPE):
        origin = launch['warmstart_origin'] if launch['run_scope'] == scope.TA_M5_FIT_SCOPE else launch['recovery_origin']
        for name, ref in origin.items():
            if isinstance(ref, dict):
                campaign.require_binding(ref, label='TA recovery ' + name, verify_file=True)


def _ta_m5_rebind_import(*, launch, model, optimizer, view_identity,
                        order, source_commit, device, counters):
    """Reuse the qualified original zero-step state with only its loss rebound."""
    import torch
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
        _dynamic_m5_technical_tree_sha256 as tree_sha,
    )
    origin = launch['imported_origin']
    _need(tree_sha(order) == origin['parent_order_sha256'], 'IMPORTED_PARENT_ORDER')
    saved = common._load_state(origin['imported_checkpoint'], torch=torch,
        maximum=launch['max_checkpoint_bytes'], counters=counters)
    rebound, proof = state.rebind_m5_equal_observation_initial_state(
        saved, model, optimizer,
        expected_model_state_sha256=origin['model_state_sha256'],
        new_numerical_recipe=launch['numerical_recipe'],
        feature_view_identity=view_identity, parent_order=order,
        source_commit=origin['source_commit'], new_source_commit=source_commit,
        device=device)
    _need(rebound['model_state_sha256'] == launch['expected_initial_model_state_sha256']
          and rebound['rng_state_sha256'] == origin['rng_state_sha256'],
          'IMPORTED_MODEL_AND_RNG')
    _need(not optimizer.state and all(module.training for module in model.modules()),
          'IMPORTED_EMPTY_OPTIMIZER_AND_TRAIN_MODES')
    return proof


def _ta_m5_restore_final_import(*, launch, model, optimizer, view_identity,
                              order, source_commit, device, counters):
    """Strictly restore the already equal-loss zero-step FINAL import."""
    import torch
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts.run_unified_exit_random_access_full_train_v1 import (
        _dynamic_m5_technical_tree_sha256 as tree_sha,
    )
    origin = launch['imported_origin']
    _need(tree_sha(order) == origin['parent_order_sha256'], 'FINAL_IMPORT_PARENT_ORDER')
    saved = common._load_state(origin['imported_checkpoint'], torch=torch,
        maximum=launch['max_checkpoint_bytes'], counters=counters)
    keys = ('cursor', 'optimizer_updates', 'skipped_batches')
    _need(all(type(saved.get(key)) is int and saved[key] == 0 for key in keys),
          'FINAL_IMPORT_ZERO_PROGRESS')
    _need(type(optimizer) is torch.optim.AdamW and not optimizer.state
          and isinstance(saved.get('optimizer_state'), dict)
          and saved['optimizer_state'].get('state') == {}, 'FINAL_IMPORT_EMPTY_OPTIMIZER')
    _need(saved['model_state_sha256'] == origin['model_state_sha256']
          == launch['expected_initial_model_state_sha256']
          and saved['rng_state_sha256'] == origin['rng_state_sha256'],
          'FINAL_IMPORT_MODEL_AND_RNG')
    identities = [(name, id(parameter)) for name, parameter in model.named_parameters()]
    restored = state.restore_fold1_fit_state(
        saved, model, optimizer, numerical_recipe=launch['numerical_recipe'],
        feature_view_identity=view_identity, parent_order=order,
        source_commit=origin['source_commit'], device=device)
    rebound = state.capture_fold1_fit_state(
        model, optimizer, numerical_recipe=launch['numerical_recipe'],
        feature_view_identity=view_identity, parent_order=order,
        cursor=0, optimizer_updates=0, skipped_batches=0,
        source_commit=source_commit, device=device)
    _need(restored == {key: 0 for key in keys}
          and set(rebound) == set(saved)
          and all(tree_sha(rebound[key]) == tree_sha(saved[key])
                  for key in saved if key not in ('source_commit', 'contract_sha256'))
          and identities == [(name, id(parameter)) for name, parameter in model.named_parameters()]
          and not optimizer.state and all(module.training for module in model.modules()),
          'FINAL_IMPORT_STRICT_PRESERVATION')
    return {'schema_version': 'gx1_ta_m5_final_import_strict_restore_v1',
            'source_model_state_sha256': saved['model_state_sha256'],
            'model_state_sha256': rebound['model_state_sha256'],
            'source_state_sha256': saved['contract_sha256'],
            'state_sha256': rebound['contract_sha256'],
            'source_commit': origin['source_commit'], 'new_source_commit': source_commit,
            'rng_state_sha256': rebound['rng_state_sha256'],
            'optimizer_steps_performed': 0, 'numerical_recipe_unchanged': True,
            'all_state_except_source_and_seal_unchanged': True}


def _prepared_fold1(receipts: dict, view: dict, *, np):
    """One provider/cache setup for all fit batches; caller releases once."""
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    chain = receipts['provider_chain']
    provider = base._provider_module()
    provider_plan = chain['provider_plan']
    info = provider.validate_metadata(provider_plan)
    prepared = base._prepare_inputs(provider, provider_plan, info,
                                    chain['gc_layout'],
                                    ta_only=view.get('full_ta_feature_profile') is True)
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
    ta_sf = view.get('full_ta_feature_profile') is True
    if ta_sf:
        from gx1.features.entry_specialist_feature_groups_v1 import FULL_TA_ENTRY_FEATURE_KEYS
        FEATURE_KEYS = FULL_TA_ENTRY_FEATURE_KEYS
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
        gc_surfaces=None if ta_sf else prepared['gc']['surfaces'],
        gc_layout=None if ta_sf else prepared['gc']['layout'],
        batch_parent_rows=rows,
        unobserved_units_profile=receipts['fold1']['profile'])
    _need(set(raw['features']) == FEATURE_KEYS
          and raw['witness']['parent_rows'] == rows.tolist()
          and raw['witness']['decision_ns'] == issues.tolist()
          and len(rows) == right - left,
          'EXACT_FEATURE_BATCH')
    return raw['features'], rows, issues



def _ta_m5_capture_layout(rows):
    """Only two 128D M5 inputs plus the already measured endpoint arrays."""
    import numpy as np
    layout = {'parents': ((rows,), np.int64), 'issues': ((rows,), np.int64),
              'query_native': ((rows, 5), np.int64)}
    for task in ('state', 'future'):
        layout[task + '_representation'] = ((rows, 128), np.float32)
        layout[task + '_activity_side_logits'] = ((rows, 5, 2), np.float32)
        for key, dtype in (('allowed', np.uint8), ('eligible', np.bool_), ('mature', np.bool_)):
            layout[task + '_' + key] = ((rows, 5), dtype)
    return layout


def _ta_m5_capture_batch(arrays, batch, *, left, right, parents, issues):
    import numpy as np
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    _need(set(batch) == set(arrays), 'M5_CAPTURE_FIELDS')
    _need(np.array_equal(batch['parents'], parents)
          and np.array_equal(batch['issues'], issues)
          and np.array_equal(batch['query_native'],
              np.broadcast_to(np.asarray(NATIVE_K, np.int64), (right-left, 5))),
          'M5_CAPTURE_AXES')
    for key, target in arrays.items():
        value = batch[key]
        _need(isinstance(value, np.ndarray)
              and value.shape == (right-left, *target.shape[1:])
              and value.dtype == target.dtype
              and bool(np.isfinite(value).all()), 'M5_CAPTURE_ARRAY:' + key)
        arrays[key][left:right] = value


def _ta_endpoint_measurement(*, model, prepared, view, receipts, rows, issues,
        partition, endpoint_bound, future_bound, frozen_priors, device, deadline,
        counters, capture_path: Path | None = None):
    """Evaluate each row once; optionally save M5 inputs from those same calls."""
    import numpy as np
    import torch
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_endpoint_batch
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    expected_rows = scope.FIT_ROWS if partition == 'fit' else scope.SCORE_ROWS
    _need(partition in ('fit', 'score') and len(rows) == len(issues) == expected_rows,
          'TA_MEASUREMENT_POPULATION')
    before_sha = canonical_model_state_sha256(model.state_dict())
    modes = {name: module.training for name, module in model.named_modules()}
    total = None
    captured = None
    if capture_path is not None:
        _need(isinstance(capture_path, Path) and capture_path.is_absolute()
              and capture_path.suffix == '.npz' and capture_path.parent.is_dir()
              and not capture_path.exists(), 'M5_CAPTURE_DESTINATION')
        captured = {key: np.empty(shape, dtype) for key, (shape, dtype)
                    in _ta_m5_capture_layout(expected_rows).items()}
        _need(sum(value.nbytes for value in captured.values()) <= 40 * 1024**2,
              'M5_CAPTURE_MEMORY_BOUND')
    model.eval()
    try:
        for left in range(0, len(rows), scope.BATCH_SIZE):
            if time.monotonic() >= deadline - 120:
                raise _ScoreWindowExhausted()
            right = min(left + scope.BATCH_SIZE, len(rows))
            raw, parents, clocks = _feature_batch(
                prepared, view, receipts, rows, issues, left, right, np=np)
            features = {key: torch.from_numpy(value).to(device) for key, value in raw.items()}
            counters['forward_calls'] += 1
            report = evaluate_endpoint_batch(model, features, endpoint_bound, future_bound,
                parents, clocks, fold_id=1, partition=partition,
                future_weight=1.0, frozen_priors=frozen_priors,
                **({'capture_m5_readout': True} if captured is not None else {}))
            total = objective.merge_endpoint_measurements(total, report['measurement'])
            if captured is not None:
                _ta_m5_capture_batch(captured, report['m5_readout'], left=left, right=right,
                                     parents=parents, issues=clocks)
            del raw, features, report
    finally:
        for name, module in model.named_modules():
            module.training = modes[name]
        _need(canonical_model_state_sha256(model.state_dict()) == before_sha,
              'TA_MEASUREMENT_MUTATED_MODEL')
    measured = objective.finalize_endpoint_measurement(total)
    _need(measured['rows'] == expected_rows
          and measured['batches'] == (expected_rows + scope.BATCH_SIZE - 1) // scope.BATCH_SIZE,
          'TA_MEASUREMENT_COMPLETE')
    if captured is not None:
        import io
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        from gx1.contracts.local_random_access_campaign_v2 import file_sha256
        stream = io.BytesIO()
        np.savez_compressed(stream, **captured)
        _publish_bytes(capture_path, stream.getvalue())
        measured['m5_readout_capture'] = {
            'schema_version': 'gx1_ta_m5_endpoint_readout_capture_v1',
            'artifact': {'path': str(capture_path), 'sha256': file_sha256(capture_path)},
            'partition': partition, 'rows': expected_rows,
            'model_state_sha256': before_sha,
            'observation_chain_sha256': measured['observation_chain_sha256'],
            'array_bytes': sum(value.nbytes for value in captured.values()),
            'layout': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                       for key, value in captured.items()},
            'additional_forward_calls': 0, 'test_data_used': False,
        }
        measured = objective._seal({
            key: value for key, value in measured.items() if key != 'contract_sha256'})
    return measured


def _ta_require_matched_observations(initial, final):
    """Paired learning comparison must use identical endpoints and masks."""
    from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
    for key in (*objective._MEASUREMENT_BINDINGS, 'rows', 'batches', 'observation_chain_sha256'):
        _need(initial[key] == final[key], 'TA_UNMATCHED_MEASUREMENT:' + key)
    _need(set(initial['scores']) == set(final['scores']), 'TA_UNMATCHED_SCORE_ROLES')
    for role, tasks in initial['scores'].items():
        for task, tfs in tasks.items():
            for tf, metrics in tfs.items():
                for metric, values in metrics.items():
                    _need(values['count'] == final['scores'][role][task][tf][metric]['count'],
                          'TA_UNMATCHED_SUPPORT')
    _need(initial['scores']['prior'] == final['scores']['prior'], 'TA_CHANGED_FROZEN_PRIOR')


def _ta_measurement_payload(measurement, *, checkpoint, model_state_sha256, optimizer_updates):
    _need(measurement.get('schema_version') == 'gx1_native_S_F_endpoint_measurement_v1',
          'TA_MEASUREMENT_PAYLOAD')
    return {'schema_version': 'gx1_market_state_ta_sf_fold1_evaluation_v1',
            'fold_id': 1, 'measurement': measurement, 'evaluated_checkpoint': checkpoint,
            'evaluated_model_state_sha256': model_state_sha256,
            'optimizer_updates': optimizer_updates, 'test_data_used': False,
            'predictive_value_claimed': False, 'realized_economics_claimed': False}

def _ta_restore_recovery(*, recovery, fresh, model, optimizer, numerical,
                         view_identity, order, device, load_state, restore_state):
    """Import exactly the preserved state through the unchanged full-state owner."""
    origin = recovery['origin']
    initial = load_state(origin['initial_state'])
    initial_meta = recovery['initial_metadata']
    _need(initial.get('contract_sha256') == initial_meta['contract_sha256']
          == fresh['contract_sha256']
          and initial.get('source_commit') == fresh.get('source_commit') == origin['source_commit']
          and initial.get('cursor') == initial.get('optimizer_updates') == 0
          and initial.get('model_state_sha256') == initial_meta['model_state_sha256']
          == fresh['model_state_sha256'], 'TA_RECOVERY_FRESH_INITIAL_BASELINE')
    del initial
    loaded = load_state(origin['checkpoint'])
    metadata = recovery['checkpoint_metadata']
    for key in ('schema_version', 'source_commit', 'cursor', 'optimizer_updates',
                'skipped_batches', 'contract_sha256', 'model_state_sha256',
                'feature_view_sha256'):
        _need(loaded.get(key) == metadata[key], 'TA_RECOVERY_CHECKPOINT:' + key)
    restored = restore_state(
        loaded, model, optimizer, numerical_recipe=numerical,
        feature_view_identity=view_identity, parent_order=order,
        source_commit=origin['source_commit'], device=device)
    _need(restored == {'cursor': scope.TA_RECOVERY_PARENT_CURSOR,
                      'optimizer_updates': scope.TA_RECOVERY_UPDATES, 'skipped_batches': 0},
          'TA_RECOVERY_EXACT_PROGRESS')
    return restored



def _ta_m5_restore_scoring_checkpoint(*, launch, model, optimizer, view_identity,
                                       order, device, counters):
    """Restore the completed original FINAL bit exactly, without fit or reinitialization."""
    import torch
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    origin = launch['scoring_recovery_origin']
    loaded = common._load_state(origin['final_checkpoint'], torch=torch,
        maximum=launch['max_checkpoint_bytes'], counters=counters)
    _need(loaded.get('cursor') == scope.FIT_ROWS
          and loaded.get('optimizer_updates') == scope.MAX_UPDATES
          and loaded.get('skipped_batches') == 0, 'M5_SCORE_RECOVERY_FULL_PROGRESS')
    identities = [id(parameter) for parameter in model.parameters()]
    restored = state.restore_fold1_fit_state(
        loaded, model, optimizer, numerical_recipe=launch['numerical_recipe'],
        feature_view_identity=view_identity, parent_order=order,
        source_commit=origin['source_commit'], device=device)
    captured = state.capture_fold1_fit_state(
        model, optimizer, numerical_recipe=launch['numerical_recipe'],
        feature_view_identity=view_identity, parent_order=order,
        source_commit=origin['source_commit'], device=device, **restored)
    _need(captured['contract_sha256'] == loaded['contract_sha256']
          and identities == [id(parameter) for parameter in model.parameters()],
          'M5_SCORE_RECOVERY_FULL_STATE_CHANGED')
    return {**restored, 'checkpoint': origin['final_checkpoint'],
            'source_commit': origin['source_commit'],
            'contract_sha256': captured['contract_sha256'],
            'model_state_sha256': captured['model_state_sha256'],
            'rng_state_sha256': captured['rng_state_sha256'],
            'optimizer_state_sha256': captured['optimizer_state_sha256'],
            'all_state_unchanged': True, 'optimizer_steps_performed': 0,
            'initial_forward_calls': 0, 'test_data_used': False}


def _ta_m5_apply_warmstart(*, model, optimizer, legacy_model, legacy_rng, origin, device):
    """Copy qualified original weights, keep new AdamW empty and restore original RNG."""
    from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    _need(not optimizer.state, 'M5_WARMSTART_REQUIRES_EMPTY_ADAMW')
    report = state.copy_legacy_weights_to_dedicated_m5(
        legacy_model, model,
        expected_legacy_model_sha256=origin['model_state_sha256'])
    trainer._restore_attended_session_rng_state(legacy_rng, device=device)
    _need(not optimizer.state, 'M5_WARMSTART_CHANGED_ADAMW')
    return report


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
        'schema_version': (scope.TA_RESULT_SCHEMA if launch['run_scope'] in (scope.TA_SF_SCOPE, scope.TA_RECOVERY_SCOPE, scope.TA_M5_FIT_SCOPE,
            scope.TA_M5_EQUAL_LOSS_FIT_SCOPE, scope.TA_M5_SCORE_RECOVERY_SCOPE)
                           else scope.RESULT_SCHEMA),
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
    if launch['run_scope'] == scope.TA_RECOVERY_SCOPE:
        payload['recovery_origin'] = launch['recovery_origin']
    if launch['run_scope'] == scope.TA_M5_FIT_SCOPE:
        payload['warmstart_origin'] = launch['warmstart_origin']
        payload['dedicated_m5_side'] = True
    if launch['run_scope'] == scope.TA_M5_EQUAL_LOSS_FIT_SCOPE:
        payload['imported_origin'] = launch['imported_origin']
        payload['dedicated_m5_side'] = True
        payload['m5_equal_observation'] = True
    if launch['run_scope'] == scope.TA_M5_SCORE_RECOVERY_SCOPE:
        payload['scoring_recovery_origin'] = launch['scoring_recovery_origin']
        payload['initial_measurement_reused'] = True
        payload['historical_optimizer_updates'] = updates
        payload['physical_optimizer_updates'] = counts['optimizer_updates']
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

    score_recovery_run = launch['run_scope'] == scope.TA_M5_SCORE_RECOVERY_SCOPE
    next_native_pass = launch['run_scope'] == scope.TA_M5_NEXT_NATIVE_PASS_SCOPE
    capture_m5 = next_native_pass or score_recovery_run
    recovery_run = launch['run_scope'] == scope.TA_RECOVERY_SCOPE
    ta_sf = launch['run_scope'] in (scope.TA_SF_SCOPE, scope.TA_RECOVERY_SCOPE, scope.TA_M5_FIT_SCOPE,
            scope.TA_M5_EQUAL_LOSS_FIT_SCOPE, scope.TA_M5_NEXT_NATIVE_PASS_SCOPE, scope.TA_M5_SCORE_RECOVERY_SCOPE)
    equal_loss_run = launch['run_scope'] in (
        scope.TA_M5_EQUAL_LOSS_FIT_SCOPE, scope.TA_M5_NEXT_NATIVE_PASS_SCOPE, scope.TA_M5_SCORE_RECOVERY_SCOPE)
    dedicated_m5_side = launch['run_scope'] in (
        scope.TA_M5_FIT_SCOPE, scope.TA_M5_EQUAL_LOSS_FIT_SCOPE, scope.TA_M5_NEXT_NATIVE_PASS_SCOPE, scope.TA_M5_SCORE_RECOVERY_SCOPE)
    m5_equal_observation = launch['numerical_recipe'].get('m5_equal_observation', False)
    expected_recipe = (scope.ta_sf_numerical_recipe(dedicated_m5_side=dedicated_m5_side,
                           m5_equal_observation=m5_equal_observation)
                       if ta_sf else scope.fold1_numerical_recipe())
    _need(launch['run_scope'] in {scope.SCOPE, scope.TA_SF_SCOPE, scope.TA_RECOVERY_SCOPE, scope.TA_M5_FIT_SCOPE,
            scope.TA_M5_EQUAL_LOSS_FIT_SCOPE, scope.TA_M5_NEXT_NATIVE_PASS_SCOPE, scope.TA_M5_SCORE_RECOVERY_SCOPE}
          and m5_equal_observation == equal_loss_run
          and ((launch.get('capture_m5_readout') is True
                and launch.get('m5_readout_capture_measurements') == ['FINAL_TRAIN', 'OOF']
                and launch.get('max_m5_readout_array_bytes') == 40 * 1024**2)
               if capture_m5 else 'capture_m5_readout' not in launch)
          and launch['fold'] == scope.FOLD
          and launch['numerical_recipe'] == expected_recipe
          and launch['reference_result'] == (
              launch['input_refs']['ta_endpoint_priors'] if ta_sf else scope.REFERENCE_RESULT)
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
    if dedicated_m5_side:
        counters['model_construction_returns'] = 0
    timings = {}
    recovery = None
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
        if score_recovery_run:
            recovery = scope.read_ta_m5_score_recovery_origin(launch)
        if recovery_run:
            recovery = scope.read_ta_recovery_origin(launch['recovery_origin'],
                expected_view_sha256=launch['expected_view_sha256'])
        if ta_sf:
            receipts = zero.load_ta_input_receipts()
            view = zero._bind_ta_view(receipts, dedicated_m5_side=dedicated_m5_side)
            candidate_ref = launch['input_refs']['ta_candidate_view']
            candidate = authority.campaign.read_bound_json(
                Path(candidate_ref['path']), candidate_ref['sha256'])
            _need(view == candidate, 'TA_EXACT_CANDIDATE_VIEW')
            from gx1.contracts.market_state_five_endpoint_target_binding_v1 import load_bound_five_endpoint_targets
            from gx1.contracts.market_state_future_displacement_candidate_v1 import load_bound_future_displacement
            from gx1.contracts import market_state_time_query_native_F_objective_v1 as sf_objective
            from gx1.contracts.market_state_time_query_native_F_batch_v1 import evaluate_fit_batch
            endpoint_bound = load_bound_five_endpoint_targets()
            future_bound = load_bound_future_displacement()
            prior_ref = launch['input_refs']['ta_endpoint_priors']
            frozen_priors = authority.campaign.read_bound_json(
                Path(prior_ref['path']), prior_ref['sha256'])
        else:
            zero_plan = zero._read_ref(scope.private.ZERO_STEP_REFS['plan'])
            receipts = zero._inspect_receipts()
            view = zero._bind_view(zero_plan, receipts, 1)
            scope._reference_report()
        _need(view['contract_sha256'] == launch['expected_view_sha256'],
              'FOLD1_MATURE_VIEW')
        bound = targets.load_bound_time_query_targets()
        provider, provider_plan, prepared, fit_rows, issues = _prepared_fold1(
            receipts, view, np=np)
        if ta_sf:
            if m5_equal_observation:
                sf_objective.require_m5_equal_observation_axes(fit_rows, issues)
            exact_priors = sf_objective.fit_endpoint_priors(
                endpoint_bound, future_bound, axes={
                    'fit_parent_rows': fit_rows, 'fit_decision_ns': issues},
                fold_id=1, ta_only=True, dedicated_m5_side=dedicated_m5_side)
            _need(exact_priors == frozen_priors, 'TA_FROZEN_FIT_PRIORS')
        order = torch.from_numpy(fit_rows.copy())
        order_sha = tree_sha(order)
        timings['input_setup_seconds'] = time.monotonic() - tic
        tic = time.monotonic()

        trainer._set_deterministic(
            numerical['seed'], device, numerical['precision_policy'])
        if dedicated_m5_side and not equal_loss_run:
            from gx1.scripts.run_market_state_native_F_private_window_v1 import _restore_m5_legacy_origin
            legacy_model, legacy_rng, legacy_import = _restore_m5_legacy_origin(
                origin=launch['warmstart_origin'], receipts=receipts, order=order,
                device=device, counters=counters, maximum=launch['max_checkpoint_bytes'])
        model = views.build_native_direction_research_model(
            view, prefix_normalization=receipts['fold1']['normal'],
            constructor_metadata=receipts['constructor'], device=device,
            unobserved_units_profile=receipts['fold1']['profile'])
        if ta_sf:
            from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
            model = NativeFutureDualModel(model).to(device)
        counters['model_constructions'] += 1
        if dedicated_m5_side:
            counters['model_construction_returns'] += 1
        actual_view = model.state_model.feature_view_identity if ta_sf else model.feature_view_identity
        _need(model.research_objective['contract_sha256']
              == launch['expected_model_contract_sha256']
              and actual_view['contract_sha256']
              == launch['expected_view_sha256'], 'FRESH_MODEL_BINDING')
        model.train()
        optimizer, _ = state.build_fold1_fit_optimizer(
            model, numerical_recipe=numerical)
        counters['optimizer_constructions'] += 1
        if equal_loss_run and not score_recovery_run:
            importer = _ta_m5_restore_final_import if next_native_pass else _ta_m5_rebind_import
            proof = importer(
                launch=launch, model=model, optimizer=optimizer,
                view_identity=actual_view, order=order,
                source_commit=plan['source_commit'], device=device, counters=counters)
            common._json(work / ('INITIAL_IMPORT.json' if next_native_pass else 'LOSS_REBINDING.json'), {
                'schema_version': ('gx1_market_state_ta_m5_final_import_restore_v1'
                                   if next_native_pass else 'gx1_market_state_ta_m5_equal_loss_fit_rebinding_v1'),
                'imported_origin': launch['imported_origin'],
                'loss_rebinding': proof, 'source_commit': plan['source_commit'],
                'fresh_full_initialbaseline_required': True,
                'qualification_weights_for_main_fit': False, 'test_data_used': False},
                counters, launch['max_published_json_bytes'])
        elif dedicated_m5_side and not score_recovery_run:
            migration = _ta_m5_apply_warmstart(
                model=model, optimizer=optimizer, legacy_model=legacy_model,
                legacy_rng=legacy_rng, origin=launch['warmstart_origin'], device=device)
            common._json(work / 'WARMSTART_IMPORT.json', {
                'schema_version': 'gx1_market_state_ta_m5_fold1_warmstart_v1',
                'legacy_import': legacy_import, 'weight_migration': migration,
                'source_commit': plan['source_commit'],
                'fresh_full_initialbaseline_required': True,
                'qualification_weights_for_main_fit': False, 'test_data_used': False},
                counters, launch['max_published_json_bytes'])
            del legacy_model, legacy_rng
        view_identity = scope.require_checkpoint_view(
            actual_view, recipe=launch, verify_files=True, ta_sf=ta_sf)
        timings['fresh_model_seconds'] = time.monotonic() - tic

        tic = time.monotonic()
        if score_recovery_run:
            _need(number == 1 and launch['max_optimizer_updates'] == 0
                  and launch['max_initial_forwards'] == 0, 'M5_SCORE_RECOVERY_ZERO_FIT')
            proof = _ta_m5_restore_scoring_checkpoint(
                launch=launch, model=model, optimizer=optimizer,
                view_identity=view_identity, order=order, device=device, counters=counters)
            initial_ref = recovery['origin']['initial_state']
            initial_measurement = recovery['origin']['initial_measurement']
            full_state = recovery['origin']['final_checkpoint']
            cursor, updates, skipped = (proof[key] for key in (
                'cursor', 'optimizer_updates', 'skipped_batches'))
            phase = 'SCORING'
            common._json(work / 'SCORING_IMPORT.json', {
                'schema_version': 'gx1_ta_m5_score_recovery_import_v1',
                'scoring_recovery_origin': recovery['origin'], **proof,
                'initial_measurement_reused': True,
                'execution_source_commit': plan['source_commit']},
                counters, launch['max_published_json_bytes'])
        else:
            initial_source_commit = (
                recovery['origin']['source_commit'] if recovery_run else plan['source_commit'])
            fresh = state.capture_fold1_fit_state(
                model, optimizer, numerical_recipe=numerical,
                feature_view_identity=view_identity, parent_order=order,
                cursor=0, optimizer_updates=0, skipped_batches=0,
                source_commit=initial_source_commit, device=device)
            if dedicated_m5_side:
                _need(fresh['model_state_sha256'] == launch['expected_initial_model_state_sha256']
                      and fresh['rng_state_sha256'] == (
                          launch['imported_origin']['rng_state_sha256'] if equal_loss_run
                          else legacy_import['rng_state_sha256']),
                      'M5_QUALIFIED_INITIAL_MODEL_AND_RNG')
            if number == 1 and recovery_run:
                initial_ref = recovery['origin']['initial_state']
                initial_measurement = recovery['origin']['initial_measurement']
                def load_original(ref):
                    return common._load_state(ref, torch=torch,
                        maximum=launch['max_checkpoint_bytes'], counters=counters)
                restored = _ta_restore_recovery(
                    recovery=recovery, fresh=fresh, model=model, optimizer=optimizer,
                    numerical=numerical, view_identity=view_identity, order=order, device=device,
                    load_state=load_original, restore_state=state.restore_fold1_fit_state)
                cursor, updates, skipped = (restored[key] for key in (
                    'cursor', 'optimizer_updates', 'skipped_batches'))
                full_state, imported_seal, imported_model_sha = _save_snapshot(
                    state_dir / f'RECOVERY_IMPORTED_p{cursor:06d}.pt',
                    model, optimizer, numerical=numerical, view_identity=view_identity,
                    order=order, cursor=cursor, updates=updates, skipped=skipped,
                    source_commit=plan['source_commit'], device=device, launch=launch,
                    counters=counters, state_dir=state_dir)
                _need(imported_model_sha == recovery['checkpoint_metadata']['model_state_sha256'],
                      'TA_RECOVERY_IMPORTED_MODEL')
                durable = {'state': full_state, 'cursor': cursor,
                           'updates': updates, 'skipped': skipped,
                           'phase': 'TRAINING', 'final_measurement': None}
                common._json(work / 'RECOVERY_IMPORT.json', {
                    'schema_version': 'gx1_market_state_ta_fold1_recovery_import_v1',
                    'recovery_origin': recovery['origin'],
                    'fresh_initial_state_sha256': fresh['contract_sha256'],
                    'source_commit': plan['source_commit'], 'imported_state': full_state,
                    'imported_state_sha256': imported_seal,
                    'imported_model_state_sha256': imported_model_sha,
                    'parent_order_sha256': order_sha, **restored,
                    'initial_measurement_reused': True, 'test_data_used': False},
                    counters, launch['max_published_json_bytes'])
            elif number == 1:
                initial_ref = common._save_state(
                    state_dir / 'INITIAL_FOLD1_STATE.pt', fresh, torch=torch,
                    maximum=launch['max_checkpoint_bytes'], counters=counters)
                if ta_sf:
                    measured = _ta_endpoint_measurement(
                        model=model, prepared=prepared, view=view, receipts=receipts,
                        rows=fit_rows, issues=issues, partition='fit',
                        endpoint_bound=endpoint_bound, future_bound=future_bound,
                        frozen_priors=frozen_priors, device=device, deadline=deadline,
                        counters=counters)
                    initial_measurement = common._json(work / 'INITIAL_TRAIN.json',
                        _ta_measurement_payload(measured, checkpoint=initial_ref,
                            model_state_sha256=fresh['model_state_sha256'],
                            optimizer_updates=0),
                        counters, launch['max_published_json_bytes'])
                    after_initial = state.capture_fold1_fit_state(
                        model, optimizer, numerical_recipe=numerical,
                        feature_view_identity=view_identity, parent_order=order,
                        cursor=0, optimizer_updates=0, skipped_batches=0,
                        source_commit=plan['source_commit'], device=device)
                    _need(after_initial['contract_sha256'] == fresh['contract_sha256'],
                          'TA_INITIAL_EVAL_MUTATED_FULL_STATE')
                    del after_initial
                else:
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
                    prior, order_sha=order_sha, view_identity=view_identity, ta_sf=ta_sf)
                initial_ref = prior['own_initial_state']
                initial_measurement = prior['initial_measurement']
                if recovery_run:
                    _need(initial_ref == recovery['origin']['initial_state']
                          and initial_measurement == recovery['origin']['initial_measurement'],
                          'TA_RECOVERY_RESUME_ORIGIN')
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
        timings['initial_or_resume_seconds'] = time.monotonic() - tic
        durable = {'state': full_state, 'cursor': cursor,
                   'updates': updates, 'skipped': skipped,
                   'phase': phase, 'final_measurement': final_measurement}

        fit_loss_sum = 0.0
        fit_batches = 0
        per_tf_counts = Counter()
        fit_started = block_started = time.monotonic()
        block_cursor, block_updates, block_skipped = cursor, updates, skipped
        timings['fit_blocks'] = []
        while not score_recovery_run and cursor < scope.FIT_ROWS and time.monotonic() < deadline - 120:
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
                if ta_sf:
                    report = evaluate_fit_batch(
                        model, features, bound, future_bound, rows, batch_issues,
                        fold_id=1, pass_index=scope.PASS_INDEX,
                        seed=numerical['seed'], future_weight=numerical['future_weight'],
                        **({'m5_equal_observation': True, 'fit_batch_start': left}
                           if m5_equal_observation else {}))
                    report['count'] = report['state_count'] + report['future_count']
                    report['per_tf_counts'] = {
                        task + ':' + tf: count
                        for task in ('state', 'future')
                        for tf, count in report[task + '_per_tf_counts'].items()}
                else:
                    report = evaluate_fit_batch(
                        model, features, bound, rows, batch_issues,
                        fold_id=1, pass_index=scope.PASS_INDEX,
                        seed=numerical['seed'])
            _need(report['model_contract_sha256'] == launch['expected_model_contract_sha256']
                  and report['loss_contract_sha256'] == launch['expected_loss_contract_sha256']
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
                block_finished = time.monotonic()
                timings['fit_blocks'].append({
                    'start_cursor': block_cursor, 'end_cursor': cursor,
                    'rows': cursor - block_cursor, 'optimizer_updates': updates - block_updates,
                    'skipped_batches': skipped - block_skipped,
                    'elapsed_seconds_including_checkpoint': block_finished - block_started})
                block_started = block_finished
                block_cursor, block_updates, block_skipped = cursor, updates, skipped
        timings['fit_seconds'] = time.monotonic() - fit_started
        if cursor > block_cursor:
            timings['fit_blocks'].append({
                'start_cursor': block_cursor, 'end_cursor': cursor,
                'rows': cursor - block_cursor, 'optimizer_updates': updates - block_updates,
                'skipped_batches': skipped - block_skipped,
                'elapsed_seconds_before_terminal_checkpoint': time.monotonic() - block_started})

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
            if ta_sf:
                phase = 'SCORING'
                durable = {'state': full_state, 'cursor': cursor,
                           'updates': updates, 'skipped': skipped,
                           'phase': phase, 'final_measurement': final_measurement}
                outcome = 'RESUMABLE'
                try:
                    if final_measurement is None:
                        tic = time.monotonic()
                        try:
                            measured = _ta_endpoint_measurement(
                                model=model, prepared=prepared, view=view, receipts=receipts,
                                rows=fit_rows, issues=issues, partition='fit',
                                endpoint_bound=endpoint_bound, future_bound=future_bound,
                                frozen_priors=frozen_priors, device=device, deadline=deadline,
                                counters=counters,
                                **({'capture_path': work / 'FINAL_M5_READOUT.npz'}
                                   if capture_m5 else {}))
                        finally:
                            timings['final_train_measurement_seconds'] = time.monotonic() - tic
                        initial_payload = authority.campaign.read_bound_json(
                            Path(initial_measurement['path']), initial_measurement['sha256'])
                        _ta_require_matched_observations(
                            initial_payload['measurement'], measured)
                        final_measurement = common._json(work / 'FINAL_TRAIN.json',
                            _ta_measurement_payload(measured, checkpoint=full_state,
                                model_state_sha256=model_sha, optimizer_updates=updates),
                            counters, launch['max_published_json_bytes'])
                        durable['final_measurement'] = final_measurement
                    axes = prepared['axes']
                    score_slice = slice(scope.FOLD['oof_start_row'], scope.FOLD['oof_end_row'])
                    score_rows = np.asarray(axes['parent_rows'][score_slice], dtype=np.int64)
                    score_issues = np.asarray(axes['decision_ns'][score_slice], dtype=np.int64)
                    score_axes = endpoint_bound.mature_oof_axes(fold_id=1)
                    _need(np.array_equal(score_rows, score_axes['score_parent_rows'])
                        and np.array_equal(score_issues, score_axes['score_decision_ns']),
                        'TA_EXACT_LATER_SCORE_AXIS')
                    score_started = True
                    tic = time.monotonic()
                    try:
                        measured = _ta_endpoint_measurement(
                            model=model, prepared=prepared, view=view, receipts=receipts,
                            rows=score_rows, issues=score_issues, partition='score',
                            endpoint_bound=endpoint_bound, future_bound=future_bound,
                            frozen_priors=frozen_priors, device=device, deadline=deadline,
                            counters=counters,
                            **({'capture_path': work / 'OOF_M5_READOUT.npz'}
                               if capture_m5 else {}))
                    finally:
                        timings['later_train_score_seconds'] = time.monotonic() - tic
                    oof_measurement = common._json(work / 'OOF.json',
                        _ta_measurement_payload(measured, checkpoint=full_state,
                            model_state_sha256=model_sha, optimizer_updates=updates),
                        counters, launch['max_published_json_bytes'])
                    oof_cursor = scope.SCORE_ROWS
                    phase = outcome = 'COMPLETE'
                except _ScoreWindowExhausted:
                    score_started = False
                    outcome, phase = 'RESUMABLE', 'SCORING'
            else:
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
        if score_recovery_run:
            after_score = state.capture_fold1_fit_state(
                model, optimizer, numerical_recipe=numerical,
                feature_view_identity=view_identity, parent_order=order,
                cursor=cursor, optimizer_updates=updates, skipped_batches=skipped,
                source_commit=recovery['origin']['source_commit'], device=device)
            _need(after_score['contract_sha256'] == proof['contract_sha256']
                  and counters['backward_calls'] == counters['optimizer_step_calls']
                      == counters['optimizer_updates'] == 0
                  and counters['forward_calls'] <= launch['max_forwards'],
                  'M5_SCORE_RECOVERY_MUTATED_STATE_OR_FIT')
            common._json(work / 'SCORING_STATE_PARITY.json', {
                'schema_version': 'gx1_ta_m5_score_recovery_parity_v1',
                'checkpoint': full_state, 'full_state_sha256': proof['contract_sha256'],
                'all_state_unchanged': True, 'physical_optimizer_updates': 0,
                'initial_forward_calls': 0, 'forward_calls': counters['forward_calls'],
                'historical_optimizer_updates': updates, 'test_data_used': False},
                counters, launch['max_published_json_bytes'])
            del after_score
        tic = time.monotonic()
        _bookend(launch, plan, *_window_binding(invocation))
        timings['terminal_bookend_seconds'] = time.monotonic() - tic
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
