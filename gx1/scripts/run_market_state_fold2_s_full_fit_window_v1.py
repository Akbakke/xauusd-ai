"""One guarded full-GC S fit and fixed later readout; no TEST or trading."""
from __future__ import annotations

import hashlib
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from gx1.contracts import market_state_fold2_s_full_fit_v1 as scope


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_S_FULL_FIT_' + reason)


def _tape(ref: dict):
    from gx1.audit import build_market_state_time_query_tape_v1 as producer
    from gx1.contracts.market_state_minute_query_candidate_v1 import MinuteQueryTargetTape
    from gx1.contracts.local_random_access_campaign_v2 import file_sha256
    path = Path(ref['path'])
    need(file_sha256(path) == ref['sha256'], 'TAPE_SHA')
    with np.load(path, allow_pickle=False) as saved:
        need(set(saved.files) == producer.TAPE_KEYS, 'TAPE_KEYS')
        return MinuteQueryTargetTape(
            np.array(saved['validity_ns'], copy=True),
            np.array(saved['m1_source_row'], copy=True),
            np.array(saved['allowed_set'], copy=True),
            np.array(saved['path_observed'], copy=True),
            np.array(saved['gap_kind'], copy=True))


def _as_tensors(features: dict, device: torch.device) -> dict:
    return {key: torch.from_numpy(value).to(device)
            for key, value in features.items()}


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    need(labels.ndim == scores.ndim == 1 and len(labels) == len(scores)
         and np.isfinite(scores).all()
         and np.isin(labels, (0, 1)).all(), 'AUC_INPUT')
    pos = int(labels.sum())
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = np.argsort(scores, kind='stable')
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    left = 0
    while left < len(scores):
        right = left + 1
        while right < len(scores) and sorted_scores[right] == sorted_scores[left]:
            right += 1
        ranks[order[left:right]] = (left + 1 + right) / 2.0
        left = right
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _metric_means(score: dict) -> list[dict]:
    from gx1.contracts.market_state_fold2_multiminute_path_v1 import QUERY_MINUTES
    from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES
    cells = []
    for q, entry in enumerate(score['per_checkpoint']):
        for j, tf in enumerate(TIMEFRAMES):
            metrics = {}
            for name, value in entry['per_tf'][tf].items():
                count = value['count']
                metrics[name] = {
                    'count': count,
                    'mean_brier': value['brier_sum'] / count if count else None,
                    **({'mean_log_loss': value['log_sum'] / count if count else None}
                       if 'log_sum' in value else {}),
                }
            cells.append({'checkpoint_index': q, 'timeframe': tf,
                          'query_minute': int(QUERY_MINUTES[q, j]),
                          'metrics': metrics})
    need(len(cells) == 25, 'CELL_COUNT')
    return cells


def _score(raw: np.ndarray, path: dict, issues: np.ndarray) -> dict:
    from gx1.contracts.market_state_fold2_multiminute_path_v1 import score_fixed_path
    need(raw.shape == (scope.LATER_ROWS, 5, 5, 2)
         and issues.shape == (scope.LATER_ROWS,), 'READOUT_SHAPE')
    report = score_fixed_path(raw, path)
    side = path['allowed'][:, 4, 0]
    mask = path['eligible'][:, 4, 0] & path['mature'][:, 4, 0] & np.isin(side, (1, 4))
    labels = (side[mask] == 4).astype(np.uint8)
    auc = _rank_auc(labels, raw[mask, 4, 0, 1].astype(np.float64))
    need(int(mask.sum()) == 1392 and auc is not None, 'FIXED_M5_SIDE_MASK')
    weeks = np.asarray([datetime.fromtimestamp(int(x) / 1e9, timezone.utc)
                         .strftime('%G-W%V') for x in issues])
    unique = sorted(set(weeks.tolist()))
    need(len(unique) == 14, 'FIXED_UTC_WEEKS')
    weekly = {}
    for week in unique:
        at = weeks == week
        part = {key: value[at] if isinstance(value, np.ndarray)
                and len(value) == len(issues) else value
                for key, value in path.items()}
        week_score = score_fixed_path(raw[at], part)
        weekly[week] = {'rows': int(at.sum()),
                        'per_cell': _metric_means(week_score)}
    return {'per_cell': _metric_means(report),
            'm5_minute5_known_side_auc': auc,
            'm5_minute5_known_side_rows': int(mask.sum()),
            'utc_weeks': weekly,
            'transition_observations': report['transition_observations'],
            'transition_prediction_diagnostic': report['transition_prediction_diagnostic'],
            'query_sha256': report['query_sha256'],
            'test_data_used': False}


def _capture_bytes(raw: np.ndarray, path: dict, issues: np.ndarray,
                   parents: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, raw=raw, issue_ns=issues, parent_rows=parents,
             allowed=path['allowed'], eligible=path['eligible'],
             mature=path['mature'], query_minutes=path['query_minutes'])
    blob = buffer.getvalue()
    need(len(blob) < 4 * 1024 * 1024, 'CAPTURE_BYTES')
    return blob


def _fresh_checkpoint(model, optimizer, *, cursor: int, updates: int,
                      recipe: dict, device: torch.device) -> dict:
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    need(all(parameter.grad is None for parameter in model.parameters()),
         'CHECKPOINT_GRADIENTS')
    model_state = {key: value.detach().cpu().clone()
                   for key, value in model.state_dict().items()}
    optimizer_state = optimizer.state_dict()

    def cpu_copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_copy(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(item) for item in value)
        return value

    return {'schema_version': 'gx1_fold2_s_full_state_v1',
            'recipe_sha256': recipe['recipe_sha256'],
            'cursor': cursor, 'optimizer_updates': updates,
            'model_state': model_state,
            'model_state_sha256': canonical_model_state_sha256(model_state),
            'optimizer_state': cpu_copy(optimizer_state),
            'rng_state': trainer._attended_session_rng_state(device=device),
            'source_resume_allowed': False,
            'test_data_used': False}


def _publish_capture(root: Path, name: str, raw: np.ndarray, path: dict,
                     issues: np.ndarray, parents: np.ndarray) -> dict:
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    blob = _capture_bytes(raw, path, issues, parents)
    target = root / name
    _publish_bytes(target, blob)
    return {'path': str(target), 'sha256': hashlib.sha256(blob).hexdigest()}


def _readout(model, context: dict, tape, device: torch.device,
             *, deadline: float) -> tuple[np.ndarray, dict]:
    from gx1.contracts.market_state_fold2_multiminute_path_v1 import gather_fixed_path
    from gx1.contracts.market_state_minute_query_candidate_v1 import decode_raw_minute_grid
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    issues = context['later_issue_ns']
    path = gather_fixed_path(tape, issues,
                             fold_upper_ns=context['later_score_upper_ns'])
    raw = np.empty((scope.LATER_ROWS, 5, 5, 2), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for left in range(0, scope.LATER_ROWS, 16):
            need(time.monotonic() < deadline, 'LATER_READOUT_TIME')
            right = left + 16
            features = _as_tensors(context['gather'](
                'LATER', context['later_parent_rows'][left:right],
                issues[left:right]), device)
            query = torch.from_numpy(path['query_minutes'][left:right]).to(device)
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                hidden = model.encode_features(**features)
                logits = decode_raw_minute_grid(model, hidden, query)
            need(logits.shape == (16, 5, 5, 2)
                 and bool(torch.isfinite(logits).all()), 'LATER_LOGITS')
            raw[left:right] = logits.detach().float().cpu().numpy()
    return raw, path


def _bookend(recipe: dict, plan: dict) -> None:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    campaign.require_clean_source(plan)
    need(scope.require_recipe(recipe, source_repo=scope.REPO,
                              source_commit=plan['source_commit']) == recipe,
         'SOURCE_BOOKEND')


def run_fold2_s_full_fit_window(*, policy_path, policy_file_sha256: str,
                               progress_path) -> dict:
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.market_state_fold2_selected_native_batches_v1 import load_two_batches
    from gx1.contracts.market_state_fold2_multiminute_path_v1 import gather_fixed_path
    from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import allowed_set_loss
    from gx1.contracts.market_state_minute_query_candidate_v1 import decode_raw_minute_grid
    from gx1.contracts.market_state_time_query_private_state_v1 import _parameters
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common

    started = time.monotonic()
    limits = require_guarded_cuda_trainer_execution()
    window, plan, invocation, marker, recipe = (
        authority.require_native_direction_research_context(
            policy_path, policy_file_sha256, progress_path))
    scope.require_recipe(recipe, source_repo=scope.REPO,
                         source_commit=plan['source_commit'])
    need(recipe['run_scope'] == scope.SCOPE
         and recipe['one_window'] == scope.ONE_WINDOW
         and invocation['optimizer_step_budget'] == scope.BATCH_VISITS
         and window['max_invocation_seconds'] == 12000
         and not scope.ROOT.joinpath('RESULT.json').exists(), 'ADMITTED_WINDOW')
    policy = json.loads((scope.REPO / 'NEXT_RUN_POLICY.json').read_text())
    gate = policy.get(scope.STAGE, {})
    need(policy.get('training_enabled') is True
         and policy.get('native_training_enabled') is True
         and policy.get('full_epoch_training_allowed') is True
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('genuine_forward_authorized') is True
         and gate.get('genuine_backward_authorized') is True
         and gate.get('genuine_updates_authorized') is True
         and gate.get('genuine_predictor_fit_authorized') is True
         and gate.get('test_data_used') is False, 'POLICY')
    device = torch.device('cuda', 0)
    need(torch.cuda.is_available(), 'CUDA')
    deadline = float(min(started + window['max_invocation_seconds'],
                         started + recipe['max_model_seconds']))
    root = scope.ROOT
    output = Path(window['output_directory'])
    need(not output.exists() and not Path(progress_path).exists()
         and not Path(window['campaign_cursor_path']).exists(), 'GENESIS')
    output.mkdir(parents=True)
    counters = {'json_bytes': 0, 'staged_state_load_calls': 0,
                'staged_state_load_returns': 0}
    claim = common._json(output / 'CLAIM.json', {
        'schema_version': 'gx1_fold2_s_full_fit_claim_v1',
        'recipe': window['recipe'], 'plan_sha256': plan['plan_sha256'],
        'source_commit': plan['source_commit'], 'limits': limits,
        'test_data_used': False}, counters, recipe['max_published_json_bytes'])
    state = {'batch_visits': 0, 'optimizer_updates': 0,
             'skipped_batches': 0, 'parent_cursor': 0,
             'batch_visit_in_flight': False,
             'optimizer_step_in_flight': False}
    captures, readouts, checkpoints, records = {}, {}, [], []
    error = None
    source_unchanged = False
    context = None
    initial_sha = final_sha = None
    tape = None
    try:
        _bookend(recipe, plan)
        context = load_two_batches(
            source_owners=recipe['source_owners'], stream=True,
            later_full_gc_ref=recipe['input_refs']['later_result'])
        need(context['view']['contract_sha256'] == recipe['expected_view_sha256']
             and context['view']['prefix']['cutoff_ns']
                 == recipe['feature_fit_cutoff_ns']
             and hashlib.sha256(context['train_parent_rows'].astype('<i8', copy=False).tobytes()).hexdigest()
                 == scope.TRAIN_PARENT_ORDER_SHA256
             and hashlib.sha256(context['train_issue_ns'].astype('<i8', copy=False).tobytes()).hexdigest()
                 == scope.TRAIN_ISSUES_SHA256
             and hashlib.sha256(context['later_parent_rows'].astype('<i8', copy=False).tobytes()).hexdigest()
                 == scope.LATER_PARENT_ORDER_SHA256
             and hashlib.sha256(context['later_issue_ns'].astype('<i8', copy=False).tobytes()).hexdigest()
                 == scope.LATER_ISSUES_SHA256, 'AXES_AND_VIEW')
        tape = _tape(recipe['input_refs']['candidate_tape'])
        trainer._set_deterministic(recipe['numerical_recipe']['seed'], device,
                                   recipe['numerical_recipe']['precision_policy'])
        model = views.build_native_direction_research_model(
            context['view'], prefix_normalization=context['packet'],
            constructor_metadata=context['constructor'], device=device,
            unobserved_units_profile=None).train()
        initial_sha = canonical_model_state_sha256(model.state_dict())
        need(initial_sha == scope.INITIAL_MODEL_SHA256
             and model.research_objective['contract_sha256']
                 == recipe['expected_model_contract_sha256'],
             'FRESH_MODEL')
        _, active, inactive = _parameters(model)
        for _, parameter in active:
            parameter.requires_grad_(True)
        for _, parameter in inactive:
            parameter.requires_grad_(False)
        kwargs = dict(recipe['numerical_recipe']['optimizer']['kwargs'])
        kwargs['betas'] = tuple(kwargs['betas'])
        optimizer = torch.optim.AdamW([p for _, p in active], **kwargs)
        need(not optimizer.state, 'EMPTY_ADAMW')
        checkpoint_state = _fresh_checkpoint(
            model, optimizer, cursor=0, updates=0, recipe=recipe, device=device)
        checkpoint_ref = common._save_state(
            output / 'STATE_0000.pt', checkpoint_state, torch=torch,
            maximum=recipe['max_checkpoint_bytes'], counters=counters)
        checkpoints.append({'batch_visits': 0, 'parent_cursor': 0,
                            'optimizer_updates': 0, 'state': checkpoint_ref})
        del checkpoint_state
        initial_raw, later_path = _readout(
            model, context, tape, device, deadline=deadline)
        need(canonical_model_state_sha256(model.state_dict()) == initial_sha,
             'PASSIVE_INITIAL_BASELINE')
        readouts['INITIAL'] = _score(
            initial_raw, later_path, context['later_issue_ns'])
        captures['INITIAL'] = _publish_capture(
            root, 'INITIAL_LATER_READOUT.npz', initial_raw, later_path,
            context['later_issue_ns'], context['later_parent_rows'])
        del initial_raw
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index in range(scope.BATCH_VISITS):
            need(time.monotonic() < deadline, 'FIT_TIME')
            left, right = scope.batch_bounds(index)
            state['batch_visit_in_flight'] = True
            rows = context['train_parent_rows'][left:right]
            issues = context['train_issue_ns'][left:right]
            features = _as_tensors(context['gather']('TRAIN', rows, issues), device)
            truth = gather_fixed_path(tape, issues,
                                      fold_upper_ns=recipe['s_target_fold_upper_ns'])
            target = {name: torch.from_numpy(truth[name]).to(device)
                      for name in ('allowed', 'eligible', 'mature', 'query_minutes')}
            with trainer._training_mha_fastpath_context(model.backbone), \
                 trainer._training_autocast_context(), \
                 trainer._training_model_finite_check_context():
                hidden = model.encode_features(**features)
                raw = decode_raw_minute_grid(model, hidden, target['query_minutes'])
                scored = allowed_set_loss(raw, target['allowed'],
                    target['eligible'], target['mature'], target['query_minutes'])
            need(scored['contract_sha256'] == recipe['expected_loss_contract_sha256']
                 and bool(torch.isfinite(scored['loss'])), 'LOSS')
            norm = None
            if scored['skip_update']:
                state['skipped_batches'] += 1
            else:
                scored['loss'].backward()
                state['optimizer_step_in_flight'] = True
                norm = float(trainer._optimizer_step_with_finite_gradients(
                    model=model, optimizer=optimizer))
                need(math.isfinite(norm), 'GRADIENT_NORM')
                state['optimizer_updates'] += 1
            optimizer.zero_grad(set_to_none=True)
            state.update(batch_visits=index + 1, parent_cursor=right,
                         batch_visit_in_flight=False, optimizer_step_in_flight=False)
            records.append({'batch_index': index, 'start': left, 'stop': right,
                            'loss': float(scored['loss'].detach()),
                            'gradient_norm': norm,
                            'informative': int(scored['informative_counts'].sum().item()),
                            'known_side': int(scored['known_side_counts'].sum().item())})
            del features, target, truth, hidden, raw, scored
            if state['batch_visits'] % 512 == 0 or state['batch_visits'] == scope.BATCH_VISITS:
                need(time.monotonic() < deadline, 'CHECKPOINT_TIME')
                checkpoint_state = _fresh_checkpoint(
                    model, optimizer, cursor=right,
                    updates=state['optimizer_updates'], recipe=recipe, device=device)
                checkpoint_ref = common._save_state(
                    output / f"STATE_{index + 1:04d}.pt", checkpoint_state,
                    torch=torch, maximum=recipe['max_checkpoint_bytes'],
                    counters=counters)
                checkpoints.append({'batch_visits': index + 1,
                                    'parent_cursor': right,
                                    'optimizer_updates': state['optimizer_updates'],
                                    'state': checkpoint_ref})
                del checkpoint_state
                common._json(output / f"PROGRESS_{index + 1:04d}.json",
                             {'batch_visits': index + 1,
                              'optimizer_updates': state['optimizer_updates'],
                              'parent_cursor': right, 'state': checkpoint_ref,
                              'test_data_used': False},
                             counters, recipe['max_published_json_bytes'])
        need(state['batch_visits'] == scope.BATCH_VISITS
             and state['parent_cursor'] == scope.FIT_ROWS
             and state['optimizer_updates'] + state['skipped_batches'] == scope.BATCH_VISITS,
             'ONE_PASS_COMPLETE')
        final_sha = canonical_model_state_sha256(model.state_dict())
        final_raw, final_path = _readout(
            model, context, tape, device, deadline=deadline)
        need(np.array_equal(later_path['allowed'], final_path['allowed'])
             and np.array_equal(later_path['eligible'], final_path['eligible'])
             and np.array_equal(later_path['mature'], final_path['mature'])
             and canonical_model_state_sha256(model.state_dict()) == final_sha,
             'PASSIVE_MATCHED_LATER')
        readouts['FINAL'] = _score(
            final_raw, final_path, context['later_issue_ns'])
        captures['FINAL'] = _publish_capture(
            root, 'FINAL_LATER_READOUT.npz', final_raw, final_path,
            context['later_issue_ns'], context['later_parent_rows'])
        del final_raw
        prior_doc = json.loads(Path(recipe['input_refs']['frozen_25_cell_prior']['path']).read_text())
        path_doc = json.loads(Path(recipe['input_refs']['frozen_25_cell_same_tf_path']['path']).read_text())
        frozen = next(cell for cell in prior_doc['per_cell_later_prior_mean_scores']
                      if cell['checkpoint_index'] == 4 and cell['timeframe'] == 'M5')
        prior_brier = frozen['metrics']['conditional_side']['mean_brier']
        current_auc = path_doc['measured']['later']['M5']['same_row_current_sign_auc_by_checkpoint'][4]
        final_cell = next(cell for cell in readouts['FINAL']['per_cell']
                          if cell['checkpoint_index'] == 4 and cell['timeframe'] == 'M5')
        final_brier = final_cell['metrics']['conditional_side']['mean_brier']
        need(frozen['metrics']['conditional_side']['count'] == 1392
             and readouts['FINAL']['m5_minute5_known_side_rows'] == 1392,
             'PRIOR_MATCHED_MASK')
        comparison = {'frozen_prior_brier': prior_brier,
                      'final_model_brier': final_brier,
                      'brier_improvement': prior_brier - final_brier,
                      'frozen_same_tf_current_sign_auc': current_auc,
                      'final_model_auc': readouts['FINAL']['m5_minute5_known_side_auc'],
                      'auc_improvement': (readouts['FINAL']['m5_minute5_known_side_auc']
                                          - current_auc),
                      'both_primary_metrics_improved':
                          final_brier < prior_brier
                          and readouts['FINAL']['m5_minute5_known_side_auc'] > current_auc,
                      'same_certified_rows': 1392, 'test_data_used': False}
        _bookend(recipe, plan)
        source_unchanged = True
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        try:
            campaign.require_clean_source(plan)
            source_unchanged = True
        except BaseException as source_exc:
            error += '; source: ' + type(source_exc).__name__ + ': ' + str(source_exc)
    finally:
        if context is not None:
            try:
                context['release']()
            except BaseException as release_exc:
                release_error = type(release_exc).__name__ + ': ' + str(release_exc)
                error = (error + '; release: ' + release_error
                         if error is not None else 'release: ' + release_error)
    outcome = 'COMPLETE' if error is None and source_unchanged else 'FAILED'
    visits, updates = state['batch_visits'], state['optimizer_updates']
    need(state['parent_cursor'] == scope.cursor_from_visits(visits)
         and 0 <= updates <= visits <= scope.BATCH_VISITS, 'TERMINAL_COUNTS')
    payload = {
        'schema_version': scope.RESULT_SCHEMA,
        'status': ('PASS_FOLD2_S_FULL_FIT_AND_LATER_READOUT'
                   if outcome == 'COMPLETE' else 'FAILED_CONSUMED_NO_RESUME'),
        'outcome': outcome, 'recipe': window['recipe'],
        'claim': claim, 'source_commit': plan['source_commit'],
        'source_unchanged': source_unchanged,
        'batch_visits': visits, 'optimizer_updates': updates,
        'skipped_batches': state['skipped_batches'],
        'parent_cursor': state['parent_cursor'],
        'parent_order_sha256': scope.TRAIN_PARENT_ORDER_SHA256,
        'batch_visit_in_flight_at_terminal': state['batch_visit_in_flight'],
        'optimizer_updates_is_lower_bound': state['optimizer_step_in_flight'],
        'initial_model_state_sha256': initial_sha,
        'final_model_state_sha256': final_sha,
        'checkpoint_writes': len(checkpoints),
        'checkpoints': checkpoints, 'captures': captures,
        'readouts': readouts if outcome == 'COMPLETE' else None,
        'primary_comparison': comparison if outcome == 'COMPLETE' else None,
        'training_records': records if outcome == 'COMPLETE' else None,
        'elapsed_seconds': time.monotonic() - started,
        'error': error, 'canonical_resume_authority': False,
        'predictive_edge_proven': False, 'economic_value_measured': False,
        'test_data_used': False}
    result = common._json(root / 'RESULT.json', payload, counters,
                          recipe['max_published_json_bytes'])
    terminal = common._json(root / 'TERMINAL.json', {
        'status': outcome, 'exit_code': 0 if outcome == 'COMPLETE' else 1,
        'result': result, 'recipe': window['recipe'],
        'source_unchanged': source_unchanged,
        'no_relaunch_under_this_plan': True, 'test_data_used': False},
        counters, recipe['max_published_json_bytes'])
    pointer = checkpoints[-1]['state'] if checkpoints else None
    cursor = authority.build_research_cursor(
        recipe_ref=window['recipe'], outcome=outcome,
        optimizer_updates=updates, parent_cursor=state['parent_cursor'],
        parent_order_sha256=scope.TRAIN_PARENT_ORDER_SHA256,
        full_state=pointer, result=result, batch_size=scope.BATCH_SIZE)
    cursor_ref = common._json(Path(window['campaign_cursor_path']), cursor,
                              counters, recipe['max_published_json_bytes'])
    progress = authority.build_research_progress(
        window=window, plan=plan, invocation=invocation,
        checkpoint_pointer=cursor_ref, optimizer_updates=updates,
        parent_cursor=state['parent_cursor'],
        parent_order_sha256=scope.TRAIN_PARENT_ORDER_SHA256, outcome=outcome)
    common._json(Path(progress_path), progress, counters,
                 recipe['max_published_json_bytes'])
    authority.require_research_progress(
        progress, plan_sha256=plan['plan_sha256'], invocation=invocation,
        expected_selection_receipt_sha256=None, verify_file=True)
    if error is not None:
        raise RuntimeError('FOLD2_S_FULL_FIT_CONSUMED_FAILURE: ' + error)
    return {'result': result, 'terminal': terminal,
            'checkpoint_pointer': cursor_ref, 'progress_path': str(progress_path),
            'predictive_edge_proven': False, 'test_data_used': False}
