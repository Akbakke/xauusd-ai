"""Score a frozen fold-1 time-query fit on later TRAIN rows only.

The guarded fit caller publishes its full checkpoint before invoking this
read-only scorer. Existing target, feature-view and matched-score owners retain
all label, clock, normalization and metric authority. No TEST or trade path.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable, Mapping

REPO = Path('/home/andre2/src/GX1_CURRENT')
FIT_CHECKPOINT_SCHEMA = 'gx1_market_state_time_query_fold1_fit_state_v1'
FOLD_ID = 1
BATCH_SIZE = 16
MAX_CHECKPOINT_BYTES = 1073741824
VIEW_SHA256 = '7e0d13f1b4f58cc4d3dd294bf869d3195b48a69c022fe570e6b4db02f1581f15'
ZERO_STEP_PLAN = {
    'path': str(REPO / 'configs/research/MARKET_STATE_TIME_QUERY_ZERO_STEP_20261107_002.json'),
    'sha256': '41b94d01e1cae7633946c565ab8975ce9d150d4809506df5dc19f59a77e54764',
}
REFERENCE_RESULT = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_REFERENCES_20261107_001/RESULT.json',
    'sha256': '1ef8e55ebf38a7571cd98e812491c37d2ab7498403734f7701d1f9ee442f4d4b',
}


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('TIME_QUERY_FOLD1_OOF_' + reason)


def _clean_source(source_commit: str) -> None:
    _need(isinstance(source_commit, str) and len(source_commit) == 40
          and all(c in '0123456789abcdef' for c in source_commit), 'SOURCE_COMMIT')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(REPO), 'branch', '--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain=v1',
                                     '--untracked-files=all'], text=True)
    _need(head == source_commit and branch == 'work/gx1-current' and dirty == '',
          'SOURCE_CHANGED')


def _frozen_fold_reference() -> dict:
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    ref = campaign.require_binding(REFERENCE_RESULT, label='frozen time-query reference',
                                   verify_file=True)
    result = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    _need(result.get('status') == 'COMPLETE_FROZEN_TRAIN_REFERENCES_NO_MODEL'
          and result.get('model_fits') == 0 and result.get('test_data_used') is False
          and result.get('read_only') is True, 'REFERENCE_SCOPE')
    folds = result.get('references', {}).get('folds', [])
    _need(isinstance(folds, list) and len(folds) == 5
          and [item.get('fold_id') for item in folds] == [1, 2, 3, 4, 5],
          'REFERENCE_FOLDS')
    return folds[0]


def _prior_arrays(fold: Mapping):
    """Broadcast frozen, separate target-population priors for all 288 ticks."""
    import numpy as np
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
    grid = fold['reference']['per_tf_k']
    _need(set(grid) == set(TIMEFRAMES), 'REFERENCE_TF')
    activity = np.empty((288, 5), dtype=np.float64)
    side = np.empty((288, 5), dtype=np.float64)
    three = np.empty((288, 5, 3), dtype=np.float64)
    for j, tf in enumerate(TIMEFRAMES):
        _need(set(grid[tf]) == {str(k) for k in range(1, int(NATIVE_K[j]) + 1)},
              'REFERENCE_K:' + tf)
        for step in range(1, 289):
            prior = grid[tf][str(min(step, int(NATIVE_K[j])))]['prior']
            _need(set(prior) == {'activity', 'conditional_up', 'three_class'},
                  'REFERENCE_PRIOR:' + tf)
            activity[step - 1, j] = prior['activity']
            side[step - 1, j] = prior['conditional_up']
            three[step - 1, j] = prior['three_class']
    _need(bool(np.isfinite(activity).all() and np.isfinite(side).all()
               and np.isfinite(three).all())
          and bool(np.all((activity > 0) & (activity < 1)))
          and bool(np.all((side > 0) & (side < 1)))
          and bool(np.all((three > 0) & (three < 1)))
          and bool(np.allclose(three.sum(axis=-1), 1, atol=1e-12, rtol=0)),
          'REFERENCE_PROBABILITIES')
    return activity, side, three


def _load_checkpoint(model, *, fit_checkpoint_ref: Mapping[str, str],
                     expected_model_state_sha256: str, source_commit: str,
                     device):
    import torch
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from gx1.contracts import entry_native_direction_research_state_v1 as full_state
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    _need(isinstance(expected_model_state_sha256, str)
          and len(expected_model_state_sha256) == 64
          and all(c in '0123456789abcdef' for c in expected_model_state_sha256),
          'EXPECTED_MODEL_DIGEST')
    ref = campaign.require_binding(fit_checkpoint_ref, label='frozen fold1 fit state',
                                   verify_file=True)
    path = Path(ref['path'])
    before = common._stat(path)
    _need(0 < before['st_size'] <= MAX_CHECKPOINT_BYTES,
          'CHECKPOINT_SIZE')
    state = torch.load(path, map_location='cpu', weights_only=True)
    _need(common._stat(path) == before and campaign.file_sha256(path) == ref['sha256'],
          'CHECKPOINT_BOOKEND')
    _need(isinstance(state, Mapping)
          and state.get('schema_version') == FIT_CHECKPOINT_SCHEMA
          and state.get('source_commit') == source_commit
          and state.get('fold_id') == FOLD_ID
          and state.get('feature_view_sha256') == VIEW_SHA256
          and state.get('model_state_sha256') == expected_model_state_sha256
          and isinstance(state.get('model_state'), Mapping)
          and isinstance(state.get('binding'), Mapping)
          and state['binding'].get('feature_view_identity') == model.feature_view_identity
          and type(state.get('cursor')) is int and state['cursor'] == 29574
          and type(state.get('optimizer_updates')) is int
          and type(state.get('skipped_batches')) is int
          and state['optimizer_updates'] > 0
          and state['optimizer_updates'] + state['skipped_batches'] == 1849,
          'CHECKPOINT_SCOPE')
    seal = state.get('contract_sha256')
    _need(isinstance(seal, str) and seal == full_state._tree_sha(
        {key: value for key, value in state.items() if key != 'contract_sha256'}),
        'CHECKPOINT_SEAL')
    _need(canonical_model_state_sha256(state['model_state'])
          == expected_model_state_sha256, 'CHECKPOINT_MODEL_DIGEST')
    model.load_state_dict(state['model_state'], strict=True)
    _need(canonical_model_state_sha256(model.state_dict())
          == expected_model_state_sha256
          and model.feature_view_identity['contract_sha256'] == VIEW_SHA256,
          'RESTORED_MODEL_DIGEST')
    model.eval()
    return dict(ref)


def _score_batch(model, features: Mapping, parent_rows, issue_ns, *,
                 bound, scorer, priors, device, check_deadline: Callable[[], None]):
    """Encode one issue batch once; bind all target queries in nine small blocks."""
    import numpy as np
    import torch
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
    _need(parent_rows.dtype == issue_ns.dtype == np.int64
          and parent_rows.ndim == issue_ns.ndim == 1
          and 0 < len(parent_rows) <= BATCH_SIZE
          and len(parent_rows) == len(issue_ns), 'BATCH_AXES')
    batch = len(parent_rows)
    scorer.begin_batch(issue_ns, fold_id=FOLD_ID)
    with torch.inference_mode():
        hidden = model.encode_features(**features)
        for step_start in range(1, 289, 32):
            check_deadline()
            steps = np.arange(step_start, min(step_start + 32, 289), dtype=np.int64)
            q = len(steps)
            k = np.minimum(steps[:, None], np.asarray(NATIVE_K, dtype=np.int64)[None, :])
            query = np.broadcast_to(k[None, :, :], (batch, q, len(TIMEFRAMES))).copy()
            logits = model.decode_query_grid(hidden, torch.from_numpy(query).to(device))
            model_p = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float64)
            # BoundTimeQueryTargets checks every repeated parent/issue pair before
            # the tape is read. One gather handles the whole bounded query block.
            flat_parent = np.repeat(parent_rows, q)
            flat_issue = np.repeat(issue_ns, q)
            truth = bound.gather(flat_parent, flat_issue,
                                 query.reshape(batch * q, len(TIMEFRAMES)),
                                 fold_id=FOLD_ID, partition='score')
            _need(np.array_equal(truth['issue_ns'], flat_issue)
                  and np.array_equal(truth['query_k'], query.reshape(batch * q, 5)),
                  'TARGET_AXIS')
            shape = (batch, q, 5)
            activity, side, three = priors
            scorer.add_block(step_start, query_k=query,
                validity_ns=truth['validity_ns'].reshape(shape), model_p=model_p,
                prior_activity=np.broadcast_to(activity[steps - 1][None, :, :], shape),
                prior_side_up=np.broadcast_to(side[steps - 1][None, :, :], shape),
                prior_three=np.broadcast_to(three[steps - 1][None, :, :, :],
                                            shape + (3,)),
                allowed=truth['allowed'].reshape(shape),
                eligible=truth['eligible'].reshape(shape),
                mature=truth['mature'].reshape(shape))
    scorer.end_batch()


def score_fold1_from_checkpoint(*, fit_checkpoint_ref: Mapping[str, str],
                                expected_model_state_sha256: str,
                                source_commit: str, device,
                                check_deadline: Callable[[], None] | None = None) -> dict:
    """Run one guarded, frozen fold-1 OOF score; return data for caller publication."""
    import numpy as np
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as old_target
    from gx1.contracts import market_state_time_query_binding_v1 as query_target
    from gx1.contracts.market_state_time_query_scorer_v1 import TimeQueryMatchedScorer
    from gx1.contracts.gx1_capped_execution_v1 import require_guarded_cuda_trainer_execution
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.contracts import local_random_access_campaign_v2 as campaign

    limits = require_guarded_cuda_trainer_execution()
    _need(isinstance(device, torch.device) and device.type == 'cuda'
          and torch.cuda.is_available(), 'GUARDED_CUDA')
    deadline = check_deadline or (lambda: None)
    _clean_source(source_commit)
    fold = _frozen_fold_reference()
    priors = _prior_arrays(fold)
    zero_plan = zero._read_ref(ZERO_STEP_PLAN)
    receipts = zero._inspect_receipts()
    view = zero._bind_view(zero_plan, receipts, FOLD_ID)
    _need(view['contract_sha256'] == VIEW_SHA256, 'VIEW_IDENTITY')
    bound = query_target.load_bound_time_query_targets()
    old = old_target.load_bound_five_endpoint_targets()
    _need(np.array_equal(bound.parent_rows, old.parent_rows)
          and np.array_equal(bound.decision_ns, old.decision_ns), 'TARGET_MODEL_AXIS')
    axes = old.mature_oof_axes(fold_id=FOLD_ID)
    _need(axes['score_parent_rows_sha256'] == fold['score_parent_rows_sha256']
          and axes['score_decision_ns_sha256'] == fold['score_issue_axis_sha256']
          and len(axes['score_parent_rows']) == fold['score_rows'] == 17753
          and axes['fit_decision_ns_sha256'] == fold['fit_issue_axis_sha256']
          and len(axes['fit_parent_rows']) == fold['fit_rows'] == 29574,
          'FROZEN_FOLD_AXES')
    model = views.build_native_direction_research_model(
        view, prefix_normalization=receipts['fold1']['normal'],
        constructor_metadata=receipts['constructor'], device=device,
        unobserved_units_profile=receipts['fold1']['profile'])
    checkpoint = _load_checkpoint(model, fit_checkpoint_ref=fit_checkpoint_ref,
        expected_model_state_sha256=expected_model_state_sha256,
        source_commit=source_commit, device=device)
    initial_digest = canonical_model_state_sha256(model.state_dict())
    chain = receipts['provider_chain']
    provider = base._provider_module()
    provider_plan = chain['provider_plan']
    info = provider.validate_metadata(provider_plan)
    prepared = base._prepare_inputs(provider, provider_plan, info,
                                    chain['gc_layout'])
    scorer = TimeQueryMatchedScorer()
    try:
        base._validate_axes(receipts['mature_plan']['cohorts'], prepared['axes'])
        all_axes = prepared['axes']
        _need(np.array_equal(all_axes['parent_rows'], bound.parent_rows)
              and np.array_equal(all_axes['decision_ns'], bound.decision_ns),
              'FEATURE_TARGET_AXIS')
        for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            all_axes[key].setflags(write=False)
        for key in ('signal', 'times_ns', 'sample_positions'):
            prepared['entry_sequence_source'][key].setflags(write=False)
        for key in ('snap', 'ctx_cont', 'ctx_cat'):
            prepared['current'][key].setflags(write=False)
        score_rows = axes['score_parent_rows']
        score_issues = axes['score_decision_ns']
        for left in range(0, len(score_rows), BATCH_SIZE):
            deadline()
            rows = score_rows[left:left + BATCH_SIZE]
            issues = score_issues[left:left + BATCH_SIZE]
            raw = views.gather_native_direction_feature_batch(
                view, parent_rows=all_axes['parent_rows'],
                bar_start_ns=all_axes['bar_start_ns'],
                decision_ns=all_axes['decision_ns'],
                entry_sequence_source=prepared['entry_sequence_source'],
                snap_x=prepared['current']['snap'],
                ctx_cont=prepared['current']['ctx_cont'],
                ctx_cat=prepared['current']['ctx_cat'],
                multi_tf_sources=prepared['cache'],
                gc_surfaces=prepared['gc']['surfaces'],
                gc_layout=prepared['gc']['layout'],
                batch_parent_rows=rows,
                unobserved_units_profile=receipts['fold1']['profile'])
            _need(raw['witness']['parent_rows'] == rows.tolist()
                  and raw['witness']['decision_ns'] == issues.tolist(),
                  'FEATURE_SCORE_ROWS')
            features = {name: torch.from_numpy(value).to(device)
                        for name, value in raw['features'].items()}
            _score_batch(model, features, rows, issues, bound=bound,
                         scorer=scorer, priors=priors, device=device,
                         check_deadline=deadline)
            del features, raw
        provider.validate_metadata(provider_plan)
    finally:
        base._release_pages(provider, prepared)
    score = scorer.result()
    _need(score['issue_axis']['1']['count'] == fold['score_rows']
          and score['issue_axis']['1']['sha256'] == fold['score_issue_axis_sha256'],
          'SCORED_ISSUE_AXIS')
    m5 = fold['m5_native_endpoint']['score']
    side = score['primary_M5_k1']['certified_side']
    activity = score['primary_M5_k1']['activity']
    _need(side['count'] == m5['side_known']
          and activity['count'] == m5['activity_known']
          and abs(side['prior_brier_sum'] - m5['side_prior_brier_sum']) <= 1e-6
          and abs(activity['prior_brier_sum'] - m5['activity_prior_brier_sum']) <= 1e-6,
          'MATCHED_FROZEN_REFERENCE')
    _need(canonical_model_state_sha256(model.state_dict()) == initial_digest
          and campaign.file_sha256(Path(checkpoint['path'])) == checkpoint['sha256']
          and campaign.file_sha256(Path(REFERENCE_RESULT['path'])) == REFERENCE_RESULT['sha256'],
          'SCORE_BOOKEND')
    _clean_source(source_commit)
    return {'schema_version': 'gx1_market_state_time_query_fold1_oof_result_v1',
            'scope': 'TRAIN_only_frozen_fold1_no_fit_no_TEST_no_trade',
            'checkpoint': checkpoint,
            'model_state_sha256': initial_digest,
            'feature_view_sha256': VIEW_SHA256,
            'frozen_reference': REFERENCE_RESULT,
            'fold_id': FOLD_ID,
            'fit_rows': fold['fit_rows'],
            'score_rows': fold['score_rows'],
            'primary_M5_k1': score['primary_M5_k1'],
            'matched_score': score,
            'limits': limits,
            'test_data_used': False,
            'entry_exit_or_trade_authority': False}


def _gc_d1_mask_columns(profile, gc_layout) -> dict[str, tuple[int, ...]]:
    """Use the sealed 51-field inventory on all four physical tensor aliases."""
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import entry_native_direction_prefix_units_v1 as units
    from gx1.contracts import entry_gc_mtf_history_v1 as history

    layout = history.require_gc_mtf_history_layout(gc_layout)
    fields, routes, _ = views._canonical()
    names = {'signal': fields['signal'], 'ctx_cont': fields['ctx_cont'],
             'gc_mtf_d1': layout['field_names']['D1']}
    inventory = {}
    for surface, ordered_names in names.items():
        columns = units.prefix_units_surface_columns(
            profile, surface=surface, field_names=ordered_names)
        prefix = 'ctx_cont.' if surface == 'signal' else ''
        inventory[surface] = {
            column['field'].removeprefix(prefix): (value_index, column['mask_index'])
            for value_index, column in columns.items()
        }
    expected = {entry['field'] for entry in profile['fields']}
    _need(len(expected) == 51 and all(set(value) == expected for value in inventory.values()),
          'GC_D1_INVENTORY')
    alias = dict(zip(routes['temporal_alias_signal_indices'],
                     routes['temporal_alias_ctx_cont_indices'], strict=True))
    _need(all(alias.get(inventory['signal'][name][kind])
              == inventory['ctx_cont'][name][kind]
              for name in expected for kind in (0, 1)), 'GC_D1_ALIAS_MAP')
    def indices(surface: str) -> tuple[int, ...]:
        result = tuple(index for name in sorted(expected)
                       for index in inventory[surface][name])
        _need(len(result) == 102 and len(set(result)) == 102,
              'GC_D1_DUPLICATE_COLUMN')
        return result
    signal = indices('signal')
    return {'seq_x': signal, 'snap_x': signal,
            'ctx_cont': indices('ctx_cont'),
            'gc_seq_d1': indices('gc_mtf_d1')}


def _without_gc_d1(features: Mapping, columns: Mapping[str, tuple[int, ...]]):
    """Copy only the four affected tensors; zero both values and observation masks."""
    import torch
    _need(set(columns) == {'seq_x', 'snap_x', 'ctx_cont', 'gc_seq_d1'}
          and set(columns) <= set(features), 'GC_D1_FEATURE_KEYS')
    masked = dict(features)
    for key, indices in columns.items():
        _need(len(indices) == 102 and len(set(indices)) == 102
              and features[key].ndim in (2, 3)
              and max(indices) < features[key].shape[-1], 'GC_D1_TENSOR_SHAPE:' + key)
        masked[key] = features[key].clone()
        masked[key][..., indices] = 0
        _need(bool(torch.count_nonzero(masked[key][..., indices]) == 0),
              'GC_D1_ZERO_FAILED:' + key)
    _need(torch.equal(masked['seq_x'][:, -1], masked['snap_x']),
          'GC_D1_SEQUENCE_SNAPSHOT_ALIAS')
    _need(torch.equal(masked['snap_x'][..., columns['snap_x']],
                      masked['ctx_cont'][..., columns['ctx_cont']]),
          'GC_D1_CONTEXT_ALIAS')
    return masked


def _tie_aware_auc(probabilities, labels) -> float | None:
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.uint8)
    _need(p.ndim == y.ndim == 1 and len(p) == len(y)
          and np.isfinite(p).all() and np.isin(y, (0, 1)).all(), 'AUC_INPUT')
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    wins = 0.0
    lower_negatives = 0
    left = 0
    while left < len(p):
        right = left + 1
        while right < len(p) and sorted_p[right] == sorted_p[left]:
            right += 1
        group_positives = int(sorted_y[left:right].sum())
        group_negatives = right - left - group_positives
        wins += group_positives * lower_negatives + 0.5 * group_positives * group_negatives
        lower_negatives += group_negatives
        left = right
    return wins / (positives * negatives)


def _paired_m5_summary(rows: list[tuple[int, float, float, int]],
                       prior_up: float) -> dict:
    """Return aggregate diagnostics only; issue-level probabilities stay in memory."""
    from datetime import datetime, timedelta, timezone
    import numpy as np
    _need(bool(rows) and 0 < prior_up < 1, 'DIAGNOSTIC_ROWS')
    issue_ns = np.asarray([row[0] for row in rows], dtype=np.int64)
    original = np.asarray([row[1] for row in rows], dtype=np.float64)
    absent = np.asarray([row[2] for row in rows], dtype=np.float64)
    labels = np.asarray([row[3] for row in rows], dtype=np.uint8)
    _need(np.all(issue_ns[1:] > issue_ns[:-1])
          and np.isfinite(original).all() and np.isfinite(absent).all()
          and np.all((original >= 0) & (original <= 1))
          and np.all((absent >= 0) & (absent <= 1)), 'DIAGNOSTIC_PAIRS')

    def score(p, y, *, bins: bool) -> dict:
        error = np.square(p - y.astype(np.float64))
        result = {'count': len(y), 'up_count': int(y.sum()),
                  'brier_sum': float(error.sum(dtype=np.float64)),
                  'brier_mean': float(error.mean(dtype=np.float64)),
                  'auc_tie_aware': _tie_aware_auc(p, y)}
        if bins:
            index = np.minimum(np.floor(p * 10).astype(np.int64), 9)
            result['reliability_bins'] = [
                {'lower_inclusive': i / 10, 'upper_exclusive_except_last': (i + 1) / 10,
                 'count': int((index == i).sum()),
                 'sum_predicted_up': float(p[index == i].sum(dtype=np.float64)),
                 'sum_observed_up': int(y[index == i].sum()),
                 'brier_sum': float(error[index == i].sum(dtype=np.float64))}
                for i in range(10)]
        return result

    original_score = score(original, labels, bins=True)
    absent_score = score(absent, labels, bins=True)
    prior_score = score(np.full(len(rows), prior_up), labels, bins=False)
    weeks = {}
    for i, ns in enumerate(issue_ns):
        day = datetime.fromtimestamp(int(ns // 1_000_000_000), tz=timezone.utc).date()
        monday = str(day - timedelta(days=day.weekday()))
        weeks.setdefault(monday, []).append(i)
    weekly = []
    for monday, indices in sorted(weeks.items()):
        take = np.asarray(indices, dtype=np.int64)
        left = score(original[take], labels[take], bins=False)
        right = score(absent[take], labels[take], bins=False)
        weekly.append({'week_start_utc': monday, 'count': len(take),
                       'up_count': left['up_count'], 'original': left,
                       'fit_absent_gc_d1': right,
                       'original_minus_fit_absent_brier_sum':
                           left['brier_sum'] - right['brier_sum']})
    return {'count': len(rows), 'original': original_score,
            'fit_absent_gc_d1': absent_score, 'frozen_prior': prior_score,
            'original_minus_fit_absent_brier_sum':
                original_score['brier_sum'] - absent_score['brier_sum'],
            'weekly': weekly}


def diagnose_fold1_from_checkpoint(*, fit_checkpoint_ref: Mapping[str, str],
                                    expected_model_state_sha256: str,
                                    source_commit: str, execution_commit: str,
                                    frozen_oof_ref: Mapping[str, str],
                                    reviewed_runner_ref: Mapping[str, str], device,
                                    check_deadline: Callable[[], None] | None = None) -> dict:
    """One guarded TRAIN-only M5 k=1 replay, original versus fit-absent GC D1."""
    import numpy as np
    import torch
    from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as base
    from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from gx1.contracts import market_state_five_endpoint_target_binding_v1 as old_target
    from gx1.contracts import market_state_time_query_binding_v1 as query_target
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_guarded_cuda_producer_execution, require_capped_lock_ancestry)
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
    from gx1.contracts import local_random_access_campaign_v2 as campaign

    limits = {**require_guarded_cuda_producer_execution(),
              'lock': require_capped_lock_ancestry()}
    _need(isinstance(device, torch.device) and device.type == 'cuda'
          and torch.cuda.is_available(), 'DIAGNOSTIC_GUARDED_CUDA')
    deadline = check_deadline or (lambda: None)
    _clean_source(execution_commit)
    frozen_ref = campaign.require_binding(frozen_oof_ref, label='frozen fold1 OOF',
                                          verify_file=True)
    frozen = campaign.read_bound_json(Path(frozen_ref['path']), frozen_ref['sha256'])
    _need(frozen.get('schema_version') == 'gx1_market_state_time_query_fold1_oof_result_v1'
          and frozen.get('scope') == 'TRAIN_only_frozen_fold1_no_fit_no_TEST_no_trade'
          and frozen.get('test_data_used') is False
          and frozen.get('checkpoint') == fit_checkpoint_ref
          and frozen.get('model_state_sha256') == expected_model_state_sha256
          and frozen.get('feature_view_sha256') == VIEW_SHA256
          and frozen.get('fold_id') == FOLD_ID
          and frozen.get('fit_rows') == 29574 and frozen.get('score_rows') == 17753,
          'FROZEN_OOF_SCOPE')
    fold = _frozen_fold_reference()
    prior_up = float(_prior_arrays(fold)[1][0, 0])
    receipts = zero._inspect_receipts(reviewed_runner_ref=reviewed_runner_ref)
    view = zero._bind_view(zero._read_ref(ZERO_STEP_PLAN), receipts, FOLD_ID)
    _need(view['contract_sha256'] == VIEW_SHA256, 'VIEW_IDENTITY')
    bound = query_target.load_bound_time_query_targets()
    old = old_target.load_bound_five_endpoint_targets()
    _need(np.array_equal(bound.parent_rows, old.parent_rows)
          and np.array_equal(bound.decision_ns, old.decision_ns), 'TARGET_MODEL_AXIS')
    axes = old.mature_oof_axes(fold_id=FOLD_ID)
    _need(axes['score_parent_rows_sha256'] == fold['score_parent_rows_sha256']
          and axes['score_decision_ns_sha256'] == fold['score_issue_axis_sha256']
          and len(axes['score_parent_rows']) == fold['score_rows'] == 17753
          and axes['fit_decision_ns_sha256'] == fold['fit_issue_axis_sha256']
          and len(axes['fit_parent_rows']) == fold['fit_rows'] == 29574
          and frozen['matched_score']['issue_axis']['1']
              == {'count': 17753, 'sha256': axes['score_decision_ns_sha256']},
          'DIAGNOSTIC_FOLD_AXES')
    model = views.build_native_direction_research_model(
        view, prefix_normalization=receipts['fold1']['normal'],
        constructor_metadata=receipts['constructor'], device=device,
        unobserved_units_profile=receipts['fold1']['profile'])
    checkpoint = _load_checkpoint(model, fit_checkpoint_ref=fit_checkpoint_ref,
        expected_model_state_sha256=expected_model_state_sha256,
        source_commit=source_commit, device=device)
    initial_digest = canonical_model_state_sha256(model.state_dict())
    chain = receipts['provider_chain']
    provider = base._provider_module()
    provider_plan = chain['provider_plan']
    info = provider.validate_metadata(provider_plan)
    prepared = base._prepare_inputs(provider, provider_plan, info, chain['gc_layout'])
    pairs: list[tuple[int, float, float, int]] = []
    support = {'current_any_rows': 0, 'history_any_rows': 0,
               'certified_side_current_any_rows': 0,
               'certified_side_history_any_rows': 0}
    batches = 0
    try:
        base._validate_axes(receipts['mature_plan']['cohorts'], prepared['axes'])
        all_axes = prepared['axes']
        _need(np.array_equal(all_axes['parent_rows'], bound.parent_rows)
              and np.array_equal(all_axes['decision_ns'], bound.decision_ns),
              'DIAGNOSTIC_FEATURE_TARGET_AXIS')
        for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            all_axes[key].setflags(write=False)
        for key in ('signal', 'times_ns', 'sample_positions'):
            prepared['entry_sequence_source'][key].setflags(write=False)
        for key in ('snap', 'ctx_cont', 'ctx_cat'):
            prepared['current'][key].setflags(write=False)
        columns = _gc_d1_mask_columns(receipts['fold1']['profile'],
                                      prepared['gc']['layout'])
        score_rows, score_issues = axes['score_parent_rows'], axes['score_decision_ns']
        for left in range(0, len(score_rows), BATCH_SIZE):
            deadline()
            rows = score_rows[left:left + BATCH_SIZE]
            issues = score_issues[left:left + BATCH_SIZE]
            raw = views.gather_native_direction_feature_batch(
                view, parent_rows=all_axes['parent_rows'],
                bar_start_ns=all_axes['bar_start_ns'],
                decision_ns=all_axes['decision_ns'],
                entry_sequence_source=prepared['entry_sequence_source'],
                snap_x=prepared['current']['snap'],
                ctx_cont=prepared['current']['ctx_cont'],
                ctx_cat=prepared['current']['ctx_cat'],
                multi_tf_sources=prepared['cache'],
                gc_surfaces=prepared['gc']['surfaces'],
                gc_layout=prepared['gc']['layout'], batch_parent_rows=rows,
                unobserved_units_profile=receipts['fold1']['profile'])
            _need(raw['witness']['parent_rows'] == rows.tolist()
                  and raw['witness']['decision_ns'] == issues.tolist(),
                  'DIAGNOSTIC_FEATURE_ROWS')
            current_mask = raw['features']['snap_x'][:, columns['snap_x'][1::2]]
            history_mask = raw['features']['gc_seq_d1'][..., columns['gc_seq_d1'][1::2]]
            _need(np.all((current_mask == 0) | (current_mask == 1))
                  and np.all((history_mask == 0) | (history_mask == 1)),
                  'DIAGNOSTIC_GC_D1_MASK_DOMAIN')
            current_any = np.any(current_mask == 1, axis=1)
            history_any = np.any(history_mask == 1, axis=(1, 2))
            support['current_any_rows'] += int(current_any.sum())
            support['history_any_rows'] += int(history_any.sum())
            features = {name: torch.from_numpy(value).to(device)
                        for name, value in raw['features'].items()}
            absent = _without_gc_d1(features, columns)
            query = np.ones((len(rows), 5), dtype=np.int64)
            truth = bound.gather(rows, issues, query, fold_id=FOLD_ID,
                                 partition='score')
            _need(np.array_equal(truth['issue_ns'], issues)
                  and np.array_equal(truth['query_k'], query),
                  'DIAGNOSTIC_TARGET_AXIS')
            # Match the saved OOF scorer's first 32-query decode geometry for
            # baseline parity; the counterfactual needs only its M5 k=1 query.
            steps = np.arange(1, 33, dtype=np.int64)
            grid = np.minimum(steps[:, None], np.asarray(NATIVE_K, dtype=np.int64)[None, :])
            original_query = np.broadcast_to(grid[None, :, :],
                                             (len(rows), 32, 5)).copy()
            with torch.inference_mode():
                oq = torch.from_numpy(original_query).to(device)
                q = torch.from_numpy(query[:, None, :]).to(device)
                original_p = torch.softmax(model.decode_query_grid(
                    model.encode_features(**features), oq), dim=-1)[:, 0, 0].cpu().numpy().astype(np.float64)
                absent_p = torch.softmax(model.decode_query_grid(
                    model.encode_features(**absent), q), dim=-1)[:, 0, 0].cpu().numpy().astype(np.float64)
            selected = (truth['eligible'][:, 0] & truth['mature'][:, 0]
                        & np.isin(truth['allowed'][:, 0], (1, 4)))
            support['certified_side_current_any_rows'] += int((selected & current_any).sum())
            support['certified_side_history_any_rows'] += int((selected & history_any).sum())
            for index in np.flatnonzero(selected):
                a, b = original_p[index], absent_p[index]
                _need(a[0] + a[2] > 0 and b[0] + b[2] > 0,
                      'DIAGNOSTIC_UNDEFINED_SIDE')
                pairs.append((int(issues[index]), float(a[2] / (a[0] + a[2])),
                              float(b[2] / (b[0] + b[2])),
                              int(truth['allowed'][index, 0] == 4)))
            batches += 1
            del features, absent, raw
        provider.validate_metadata(provider_plan)
    finally:
        base._release_pages(provider, prepared)
    _need(batches == (fold['score_rows'] + BATCH_SIZE - 1) // BATCH_SIZE,
          'DIAGNOSTIC_BATCH_COUNT')
    summary = _paired_m5_summary(pairs, prior_up)
    original = frozen['primary_M5_k1']['certified_side']
    _need(summary['count'] == original['count'] == fold['m5_native_endpoint']['score']['side_known']
          and abs(summary['original']['brier_sum'] - original['model_brier_sum']) <= 1e-6
          and abs(summary['frozen_prior']['brier_sum'] - original['prior_brier_sum']) <= 1e-6
          and abs(summary['frozen_prior']['brier_sum']
                  - fold['m5_native_endpoint']['score']['side_prior_brier_sum']) <= 1e-6,
          'DIAGNOSTIC_BASELINE_PARITY')
    _need(canonical_model_state_sha256(model.state_dict()) == initial_digest
          and campaign.file_sha256(Path(checkpoint['path'])) == checkpoint['sha256']
          and campaign.file_sha256(Path(frozen_ref['path'])) == frozen_ref['sha256']
          and campaign.file_sha256(Path(REFERENCE_RESULT['path'])) == REFERENCE_RESULT['sha256'],
          'DIAGNOSTIC_BOOKEND')
    _clean_source(execution_commit)
    return {'schema_version': 'gx1_market_state_fold1_gc_d1_paired_diagnostic_v1',
            'scope': 'TRAIN_only_M5_k1_frozen_checkpoint_no_fit_no_TEST_no_trade',
            'fit_source_commit': source_commit, 'execution_commit': execution_commit,
            'checkpoint': checkpoint, 'frozen_oof': dict(frozen_ref),
            'model_state_sha256': initial_digest,
            'feature_view_sha256': VIEW_SHA256,
            'frozen_reference': REFERENCE_RESULT, 'fold_id': FOLD_ID,
            'score_rows': fold['score_rows'],
            'certified_side_rows': summary['count'], 'model_forwards': 2 * batches,
            'certified_side': summary,
            'gc_d1_observed_support': support,
            'masked_gc_d1_distinct_fields': 51,
            'masked_gc_d1_value_and_mask_columns_per_surface': 102,
            'limits': limits, 'test_data_used': False,
            'entry_exit_or_trade_authority': False}
