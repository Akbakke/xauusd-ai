"""One-use TRAIN reference: closed EMA50/200 state conditions certified direction.

The fixed five-fold target, activity prior and score support are reused exactly.
This is a descriptive two-bin reference, not model fit or trading authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from gx1.audit import measure_market_state_five_endpoint_prior_v1 as base
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts.market_state_five_endpoint_contract_v1 import TIMEFRAMES, proper_score_sums
from gx1.contracts.market_state_five_endpoint_prior_v1 import fit_activity_side_prior
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory

REPO, RUNS, POLICY = base.REPO, base.RUNS, base.POLICY
STAGE = 'market_state_ema_side_reference_20261107_001'
SCHEMA = 'gx1_market_state_ema_side_reference_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
FEATURE_COLUMN = 98
FEATURE_NAME = 'ema50_200_bull_state'
DURATION_NS = {'M5': 300_000_000_000, 'M15': 900_000_000_000,
               'H1': 3_600_000_000_000, 'H4': 14_400_000_000_000,
               'D1': 86_400_000_000_000}
SHIFT = {'M5': '0 days 00:05:00', 'M15': '0 days 00:15:00',
         'H1': '0 days 01:00:00', 'H4': '0 days 04:00:00',
         'D1': '1 days 00:00:00'}
ORIGIN = {'M5': '0 days 00:00:00', 'M15': '0 days 00:00:00',
          'H1': '0 days 00:00:00', 'H4': '0 days 02:00:00',
          'D1': '0 days 22:00:00'}
PRIOR_REFS = {
    'plan': {'path': str(base.PLAN_PATH),
             'sha256': '7710bd98c7e14b1f2a6d4506f3f3499b02a0c04078f95c8b279b9cd033f00170'},
    'result': {'path': str(base.OUT_ROOT / 'RESULT.json'),
               'sha256': '5304433c42b43ea2bdf593066e627c8904604a751aa27c6fa5dab6349efefa48'},
    'terminal': {'path': str(base.OUT_ROOT / 'TERMINAL.json'),
                 'sha256': '8f9804b384baf0c1b3d5d321465f00424da17956c06ef39929f107767bc4e9a4'},
}
CACHE_SOURCE_PLAN = {
    'path': str(REPO / 'configs/research/TRENDLINE_CONDITIONAL_PREFIX_20261106_001.json'),
    'sha256': 'bc4469117ef476250e50ed308941b790da67ed30b134b5bf350e100240f17491',
}
CACHE_MANIFEST = {
    'path': '/home/andre2/GX1_DATA/data/data/prebuilt/XAU_GC_NATIVE_26M_20261018_003/MULTI_TF_V4_CACHE/manifest.json',
    'sha256': '3f4c3b4b54b08931850b716e0b7cf4b5de0b54c42b0fee813cde6f1324371da4',
}
INPUT_REFS = {
    'prior': PRIOR_REFS,
    'cache_source_plan': CACHE_SOURCE_PLAN,
    'cache_manifest': CACHE_MANIFEST,
    'target': {name: {'path': path, 'sha256': digest}
               for name, (path, digest) in target.REFS.items()},
}
SOURCE_PATHS = frozenset({
    'gx1/audit/measure_market_state_ema_side_reference_v1.py',
    'gx1/audit/measure_market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/features/htf_features.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
})
LIMITS = {'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
          'swap': 512 * 1024**2, 'max_wall_seconds': 1800}
EXECUTION = {'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '4G', 'swap': '512M', 'timeout_seconds': 1800,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.measure_market_state_ema_side_reference_v1',
             'model_forwards': 0, 'optimizer_steps': 0}
SCOPE = {'train_start_ns': target.TRAIN_START_NS,
         'train_end_exclusive_ns': target.FOLD_BOUNDARIES_NS[-1],
         'fold_ids': [1, 2, 3, 4, 5], 'timeframes': list(TIMEFRAMES),
         'cache_feature_column': FEATURE_COLUMN, 'cache_feature_name': FEATURE_NAME,
         'asof_rule': 'cache_label_plus_timeframe_duration_le_decision_ns'}


def _need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('EMA_SIDE_REFERENCE_' + reason)


_sha, _ref, _write, _clean_head = base._sha, base._ref, base._write, base._clean_head


def _file_sig(path: Path) -> tuple[int, int, int, int, int, int]:
    _need(path.is_file() and not path.is_symlink(), 'REGULAR:' + path.name)
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_mode, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _read_json(ref: dict, limit: int) -> dict:
    path = Path(ref['path'])
    _need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= limit,
          'REF_STAT:' + path.name)
    before = (path.stat().st_dev, path.stat().st_ino, path.stat().st_size,
              path.stat().st_mtime_ns, path.stat().st_ctime_ns)
    raw = path.read_bytes()
    after = (path.stat().st_dev, path.stat().st_ino, path.stat().st_size,
             path.stat().st_mtime_ns, path.stat().st_ctime_ns)
    _need(before == after and len(raw) == before[2]
          and hashlib.sha256(raw).hexdigest() == ref['sha256'],
          'REF_HASH:' + path.name)
    return json.loads(raw)


def _small_inputs() -> tuple[dict, dict]:
    """Inspect immutable metadata and stat cache arrays; no cache payload read."""
    prior_plan = _read_json(PRIOR_REFS['plan'], 4 * 1024 * 1024)
    prior = _read_json(PRIOR_REFS['result'], 4 * 1024 * 1024)
    terminal = _read_json(PRIOR_REFS['terminal'], 4 * 1024 * 1024)
    old = _read_json(CACHE_SOURCE_PLAN, 4 * 1024 * 1024)
    manifest = _read_json(CACHE_MANIFEST, 64 * 1024 * 1024)
    _need(prior_plan['input_refs'] == INPUT_REFS['target']
          and prior_plan['fold_ids'] == SCOPE['fold_ids']
          and prior_plan['test_data_used'] is False
          and prior['plan'] == PRIOR_REFS['plan']
          and prior['status'] == terminal['status'] == 'COMPLETE'
          and prior['decision'] == 'PASS_TRAIN_PRIOR_REFERENCE_ONLY'
          and terminal['result'] == PRIOR_REFS['result']
          and terminal['exit_code'] == 0
          and prior['model_forwards'] == prior['optimizer_steps'] == 0
          and prior['test_data_used'] is terminal['test_data_used'] is False
          and len(prior['folds']) == 5
          and old['inputs']['cache_manifest'] == CACHE_MANIFEST
          and old['test_data_used'] is False
          and manifest['feature_count'] == 253
          and manifest['feature_names'][FEATURE_COLUMN] == FEATURE_NAME
          and manifest['shift_contract'] == SHIFT
          and manifest['resample_origin_contract'] == ORIGIN,
          'RECEIPT_CHAIN')
    target_source = _read_json(INPUT_REFS['target']['target_source_plan'],
                               4 * 1024 * 1024)
    _need(manifest['m5_prebuilt_source'] == target_source['m5']['source']['path']
          and manifest['m5_prebuilt_source_sha256']
              == target_source['m5']['source']['sha256'],
          'CACHE_TARGET_SOURCE')
    for tf in TIMEFRAMES:
        meta = manifest['tfs'][tf]
        old_refs = old['inputs']['per_tf'][tf]
        root = Path(CACHE_MANIFEST['path']).parent
        for kind in ('feats', 'ts'):
            ref = old_refs[kind]
            _need(ref == {'path': str(root / meta[kind + '_npy']),
                          'sha256': meta[kind + '_npy_sha256']},
                  'CACHE_LINEAGE:' + tf + ':' + kind)
            path = Path(ref['path'])
            _need(path.is_file() and not path.is_symlink()
                  and path.stat().st_size == meta[kind + '_npy_size_bytes'],
                  'CACHE_STAT:' + tf + ':' + kind)
        _need(meta['feature_count'] == 253
              and meta['first_ts_ns'] < target.TRAIN_START_NS
              and meta['last_ts_ns'] < 1782864000000000000,
              'CACHE_SCOPE:' + tf)
    return prior, old['inputs']['per_tf']


def _inspect_plan(plan_path: Path, plan_sha256: str, out_dir: Path
                  ) -> tuple[dict, dict, str, dict, dict]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and plan_path.is_file()
          and not out_dir.exists() and len(plan_sha256) == 64
          and _sha(plan_path) == plan_sha256, 'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    _need(plan.get('schema_version') == SCHEMA and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('source_base_head') is not None
          and isinstance(plan['source_base_head'], str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0
          and plan.get('max_invocations') == 1
          and plan.get('scope') == SCOPE
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('limits') == LIMITS
          and plan.get('execution') == EXECUTION
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('test_data_used') is False,
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
    prior, cache_refs = _small_inputs()
    return plan, {'path': str(plan_path), 'sha256': plan_sha256}, head, prior, cache_refs


def preflight_read_only(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, ref, head, _, cache_refs = _inspect_plan(plan_path, plan_sha256, out_dir)
    return {'status': 'READY_NO_CLAIM_NO_CACHE_PAYLOAD', 'plan': ref,
            'source_head': head, 'source_owners_checked': len(plan['source_owners']),
            'cache_array_stats_checked': 2 * len(cache_refs),
            'claim_created': False, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False}


def _asof_binary(issue_ns: np.ndarray, cache_ts: np.ndarray, cache: np.ndarray,
                 *, tf: str, train_end_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Select the last observed cache bar closed no later than each M5 issue."""
    duration = DURATION_NS[tf]
    _need(issue_ns.dtype == cache_ts.dtype == np.int64
          and issue_ns.ndim == cache_ts.ndim == 1
          and len(issue_ns) > 0 and len(cache_ts) > 1
          and np.all(np.diff(issue_ns) > 0)
          and cache.dtype == np.float32 and cache.ndim == 2
          and cache.shape == (len(cache_ts), 253)
          and int(issue_ns[-1]) < train_end_ns,
          'CLOCK_AXIS:' + tf)
    # Only the TRAIN prefix participates in monotonicity and as-of validation.
    # Searchsorted touches the later clock solely to locate the prefix boundary;
    # no later feature value is selected or inspected.
    prefix = int(np.searchsorted(cache_ts, train_end_ns - duration, side='left'))
    _need(prefix > 1 and bool(np.all(np.diff(cache_ts[:prefix]) > 0))
          and int(cache_ts[prefix - 1]) + duration < train_end_ns
          and (prefix == len(cache_ts)
               or int(cache_ts[prefix]) + duration >= train_end_ns),
          'TRAIN_CACHE_PREFIX:' + tf)
    at = np.searchsorted(cache_ts[:prefix], issue_ns - duration, side='right') - 1
    _need(bool(np.all(at >= 0)), 'ASOF_HISTORY:' + tf)
    has_next = at + 1 < prefix
    _need(bool(np.all(cache_ts[at] + duration <= issue_ns))
          and bool(np.all(cache_ts[at[has_next] + 1] + duration
                          > issue_ns[has_next])),
          'ASOF_CLOSED_BAR:' + tf)
    values = np.asarray(cache[at, FEATURE_COLUMN], dtype=np.float64)
    _need(bool(np.isfinite(values).all())
          and bool(np.isin(values, (0.0, 1.0)).all()),
          'EMA_DOMAIN:' + tf)
    return values.astype(np.uint8), at.astype(np.int64)


def _fit_side_bins(truth: dict, feature: np.ndarray,
                   score_feature: np.ndarray) -> tuple[dict, np.ndarray, dict]:
    prior = fit_activity_side_prior(truth['allowed'], truth['eligible'],
                                    truth['mature'])
    _need(feature.dtype == score_feature.dtype == np.uint8
          and feature.shape == truth['allowed'].shape
          and score_feature.ndim == 2
          and score_feature.shape[1] == len(TIMEFRAMES)
          and len(score_feature) > 0
          and np.isin(feature, (0, 1)).all()
          and np.isin(score_feature, (0, 1)).all(), 'FIT_FEATURE_DOMAIN')
    # An unused bin has no estimate. A bin queried on any later score row
    # must have earlier mature certified side support; no neutral fallback.
    p_up = np.full((len(TIMEFRAMES), 2), np.nan, dtype=np.float64)
    counts = {}
    for j, tf in enumerate(TIMEFRAMES):
        certified = (truth['eligible'][:, j] & truth['mature'][:, j]
                     & np.isin(truth['allowed'][:, j], (1, 4)))
        counts[tf] = {}
        for state in (0, 1):
            selected = certified & (feature[:, j] == state)
            n = int(selected.sum())
            score_n = int(np.count_nonzero(score_feature[:, j] == state))
            _need(n > 0 or score_n == 0,
                  'EMPTY_CERTIFIED_FIT_BIN:' + tf + ':' + str(state))
            up = int(np.count_nonzero(truth['allowed'][selected, j] == 4))
            if n:
                p_up[j, state] = (up + 1) / (n + 2)
            counts[tf][str(state)] = {'fit_certified_side': n, 'fit_up': up,
                                      'score_rows': score_n}
        _need(sum(counts[tf][str(k)]['fit_certified_side'] for k in (0, 1))
              == prior['counts'][tf]['certified_side'], 'SIDE_FIT_SUPPORT:' + tf)
    return prior, p_up, counts


def _predictions(feature: np.ndarray, prior: dict,
                 p_up: np.ndarray) -> np.ndarray:
    _need(feature.dtype == np.uint8 and feature.ndim == 2
          and feature.shape[1] == len(TIMEFRAMES)
          and p_up.shape == (len(TIMEFRAMES), 2), 'PREDICTION_AXES')
    p = np.empty((len(feature), len(TIMEFRAMES), 3), dtype=np.float64)
    for j, tf in enumerate(TIMEFRAMES):
        _need(np.isfinite(p_up[j, feature[:, j]]).all(),
              'UNESTIMATED_SCORE_BIN:' + tf)
        c = prior['counts'][tf]
        active = (c['active'] + 1) / (c['activity'] + 2)
        up = p_up[j, feature[:, j]]
        p[:, j, 0] = active * (1 - up)
        p[:, j, 1] = 1 - active
        p[:, j, 2] = active * up
    _need(np.isfinite(p).all() and np.all(p > 0)
          and np.allclose(p.sum(axis=-1), 1, rtol=0, atol=1e-15),
          'SIMPLEX')
    return p


def _fold(bound: target.BoundFiveEndpointTargets, feature: np.ndarray,
          fold_id: int, saved: dict) -> dict:
    lower, upper = target.FOLD_BOUNDARIES_NS[fold_id - 1:fold_id + 1]
    fit = (bound.decision_ns >= target.TRAIN_START_NS) & (bound.decision_ns < lower)
    score = (bound.decision_ns >= lower) & (bound.decision_ns < upper)
    _need(bool(fit.any()) and bool(score.any()) and not bool((fit & score).any()),
          'FOLD_ROWS')
    fit_truth = bound.gather(bound.parent_rows[fit], bound.decision_ns[fit],
                             fold_id=fold_id, partition='fit')
    score_truth = bound.gather(bound.parent_rows[score], bound.decision_ns[score],
                               fold_id=fold_id, partition='score')
    activity_prior, p_up, bin_counts = _fit_side_bins(fit_truth, feature[fit], feature[score])
    predicted = _predictions(feature[score], activity_prior, p_up)
    scores = proper_score_sums(predicted, score_truth['allowed'],
                               score_truth['eligible'], score_truth['mature'])['per_tf']
    old_predictions = np.broadcast_to(activity_prior['probabilities'],
                                      predicted.shape)
    old_scores = proper_score_sums(old_predictions, score_truth['allowed'],
                                   score_truth['eligible'],
                                   score_truth['mature'])['per_tf']
    _need(saved['fold_id'] == fold_id
          and saved['fit_rows'] == int(fit.sum())
          and saved['score_rows'] == int(score.sum())
          and saved['fit_first_issue_ns'] == int(bound.decision_ns[fit][0])
          and saved['fit_last_issue_ns'] == int(bound.decision_ns[fit][-1])
          and saved['score_first_issue_ns'] == int(bound.decision_ns[score][0])
          and saved['score_last_issue_ns'] == int(bound.decision_ns[score][-1])
          and saved['fit_counts'] == activity_prior['counts']
          and np.array_equal(np.asarray(saved['prior_probabilities']),
                             activity_prior['probabilities']), 'FROZEN_PRIOR_FOLD')
    for tf in TIMEFRAMES:
        for name in ('activity', 'conditional_side', 'three_class'):
            current, frozen = old_scores[tf][name], saved['score_sums'][tf][name]
            _need(current['count'] == frozen['count']
                  and abs(current['brier_sum'] - frozen['brier_sum']) <= 1e-8,
                  'FROZEN_PRIOR_SCORE:' + tf + ':' + name)
            if name != 'activity':
                _need(abs(current['log_sum'] - frozen['log_sum']) <= 1e-8,
                      'FROZEN_PRIOR_LOG:' + tf + ':' + name)
            _need(scores[tf][name]['count'] == frozen['count'],
                  'IDENTICAL_SCORE_SUPPORT:' + tf + ':' + name)
        _need(abs(scores[tf]['activity']['brier_sum']
                  - old_scores[tf]['activity']['brier_sum']) <= 1e-8,
              'UNCHANGED_ACTIVITY:' + tf)
        # EMA changes only the conditional side. Preserve the fixed activity
        # score bit-for-bit from the frozen prior after checking roundoff above.
        scores[tf]['activity'] = dict(saved['score_sums'][tf]['activity'])
    paired = {}
    for tf in TIMEFRAMES:
        paired[tf] = {}
        for name in ('activity', 'conditional_side', 'three_class'):
            frozen = saved['score_sums'][tf][name]
            candidate = scores[tf][name]
            count = candidate['count']
            brier_gain = frozen['brier_sum'] - candidate['brier_sum']
            row = {'count': count, 'prior_brier_sum': frozen['brier_sum'],
                   'ema_brier_sum': candidate['brier_sum'],
                   'brier_gain_sum': brier_gain,
                   'brier_gain_mean': brier_gain / count if count else None}
            if name != 'activity':
                log_gain = frozen['log_sum'] - candidate['log_sum']
                row.update({'prior_log_sum': frozen['log_sum'],
                            'ema_log_sum': candidate['log_sum'],
                            'log_gain_sum': log_gain,
                            'log_gain_mean': log_gain / count if count else None})
            paired[tf][name] = row
    return {'fold_id': fold_id, 'fit_rows': int(fit.sum()),
            'score_rows': int(score.sum()), 'fit_bin_counts': bin_counts,
            'p_up_given_active_by_ema_state': {
                tf: {'0': float(p_up[j, 0]) if np.isfinite(p_up[j, 0]) else None,
                     '1': float(p_up[j, 1]) if np.isfinite(p_up[j, 1]) else None}
                for j, tf in enumerate(TIMEFRAMES)},
            'ema_score_sums': scores,
            'prior_score_sums': saved['score_sums'],
            'paired_score': paired}


def _measure(prior: dict, cache_refs: dict) -> dict:
    bound = target.load_bound_five_endpoint_targets()
    _need(len(bound.parent_rows) == prior['target_rows_joined']
          and bound.unjoined_target_rows == prior['unjoined_target_rows'],
          'TARGET_ROWS')
    feature = np.empty((len(bound.parent_rows), len(TIMEFRAMES)), dtype=np.uint8)
    cache_evidence = {}
    for j, tf in enumerate(TIMEFRAMES):
        refs = cache_refs[tf]
        before = {kind: _file_sig(Path(refs[kind]['path']))
                  for kind in ('ts', 'feats')}
        for kind in ('ts', 'feats'):
            path = Path(refs[kind]['path'])
            _need(_sha(path) == refs[kind]['sha256']
                  and _file_sig(path) == before[kind],
                  'CACHE_HASH_BOOKEND:' + tf + ':' + kind)
        ts = np.load(refs['ts']['path'], mmap_mode='r', allow_pickle=False)
        cache = np.load(refs['feats']['path'], mmap_mode='r', allow_pickle=False)
        selected, rows = _asof_binary(
            bound.decision_ns, ts, cache, tf=tf,
            train_end_ns=target.FOLD_BOUNDARIES_NS[-1])
        feature[:, j] = selected
        cache_evidence[tf] = {
            'feature_column': FEATURE_COLUMN,
            'selected_cache_row_min': int(rows.min()),
            'selected_cache_row_max': int(rows.max()),
            'selected_ema_state_0': int(np.count_nonzero(selected == 0)),
            'selected_ema_state_1': int(np.count_nonzero(selected == 1))}
        _need(all(_file_sig(Path(refs[kind]['path'])) == before[kind]
                  for kind in ('ts', 'feats')), 'CACHE_BOOKEND:' + tf)
        del ts, cache, selected, rows
    folds = [_fold(bound, feature, fid, prior['folds'][fid - 1])
             for fid in SCOPE['fold_ids']]
    pooled = {}
    for tf in TIMEFRAMES:
        pooled[tf] = {}
        for name in ('activity', 'conditional_side', 'three_class'):
            cells = [f['ema_score_sums'][tf][name] for f in folds]
            count = sum(v['count'] for v in cells)
            brier = sum(v['brier_sum'] for v in cells)
            _need(count == prior['pooled_score'][tf][name]['count'],
                  'POOLED_SUPPORT:' + tf + ':' + name)
            frozen = prior['pooled_score'][tf][name]
            pooled[tf][name] = {
                'count': count, 'prior_brier_sum': frozen['brier_sum'],
                'prior_brier_mean': frozen['brier_mean'],
                'ema_brier_sum': brier,
                'ema_brier_mean': brier / count if count else None,
                'brier_gain_vs_prior_mean':
                (frozen['brier_sum'] - brier) / count if count else None}
            if name != 'activity':
                log = sum(v['log_sum'] for v in cells)
                pooled[tf][name].update({
                    'prior_log_sum': frozen['log_sum'],
                    'prior_log_mean': frozen['log_mean'],
                    'ema_log_sum': log,
                    'ema_log_mean': log / count if count else None,
                    'log_gain_vs_prior_mean':
                    (frozen['log_sum'] - log) / count if count else None})
        _need(abs(pooled[tf]['activity']['ema_brier_sum']
                  - prior['pooled_score'][tf]['activity']['brier_sum']) <= 1e-8,
              'POOLED_ACTIVITY:' + tf)
    return {'scope_limit': 'Five chronological TRAIN folds; simple EMA-conditioned direction reference only, no model learning, sealed TEST or economic edge.',
            'target_rows_joined': len(bound.parent_rows),
            'unjoined_target_rows': bound.unjoined_target_rows,
            'cache_evidence': cache_evidence, 'folds': folds,
            'pooled_score': pooled, 'frozen_prior_result': PRIOR_REFS['result'],
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    plan, plan_ref, head, prior, cache_refs = _inspect_plan(
        plan_path, plan_sha256, out_dir)
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    _need(all(cap.get(key) == value for key, value in LIMITS.items()
              if key != 'max_wall_seconds')
          and cap.get('pids') == 64 and cap.get('numerical_threads') == 1
          and lock.get('scope') == cap.get('cgroup_path', '')[len('/sys/fs/cgroup'):],
          'CAP_LIMITS')
    _need(not any(name.split('.')[0] in ('sklearn', 'joblib')
                  for name in sys.modules), 'FIT_IMPORT')
    policy_ref = _ref(POLICY)
    policy = json.loads(POLICY.read_text())
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
    gate = policy.get(STAGE, {})
    _need(policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and gate.get('status') == gate.get('decision') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == 1
          and gate.get('relaunch_allowed') is False
          and gate.get('memory_max') == '4G'
          and gate.get('swap_max') == '512M'
          and gate.get('wall_seconds') == LIMITS['max_wall_seconds']
          and gate.get('test_data_used') is False
          and gate.get('checkpoint_reads_authorized') is False
          and gate.get('broker_access_allowed') is False
          and gate.get('model_forwards_authorized') == 0
          and gate.get('optimizer_steps_authorized') == 0
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('output_root') == str(OUT_ROOT),
          'POLICY_GATE')
    _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_BEFORE_CLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    claim = None
    old_alarm = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError('EMA_SIDE_REFERENCE_WALL_LIMIT')))
        signal.alarm(LIMITS['max_wall_seconds'])
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_ema_side_reference_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'fold_ids': SCOPE['fold_ids']})
        data = _measure(prior, cache_refs)
        _need(_sha(plan_path) == plan_sha256
              and _sha(POLICY) == policy_ref['sha256']
              and _clean_head() == head
              and _sha(out_dir / 'CLAIM.json') == claim['sha256']
              and {item.name for item in out_dir.iterdir()} == {'CLAIM.json'},
              'FINAL_SOURCE_IDENTITY')
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_ema_side_reference_result_v1',
            'status': 'COMPLETE', 'decision': 'PASS_TRAIN_EMA_REFERENCE_ONLY',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'elapsed_seconds': time.monotonic() - began, **data})
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_ema_side_reference_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json']})
        return {'result': result, 'terminal': terminal}
    except Exception as exc:
        failure = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_ema_side_reference_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_ema_side_reference_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failure, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True,
            'output_inventory': (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                 if claim else ['FAILED.json', 'TERMINAL.json'])})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight_read_only(args.plan, args.plan_sha256,
                                             args.out_dir), sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
