"""Pure TRAIN prefix priors and causal now-state references for M5-tick queries.

No source load, model fit, fold selection or publication. The caller must bind
one immutable candidate tape and exact chronological fold axes from the pinned
model decision/parent rows. This core cannot certify that external receipt.
"""
from __future__ import annotations

import hashlib
import numpy as np

from gx1.contracts.market_state_time_query_target_v1 import (
    NATIVE_K, TICK_NS, TIMEFRAMES, TimeQueryTargetTape,
)

_BITS = (1, 2, 4, 5, 7)
_K = np.asarray(NATIVE_K, dtype=np.int64)


def _require(ok: bool, name: str) -> None:
    if not ok:
        raise ValueError('MARKET_STATE_TIME_QUERY_REFERENCE_' + name)


def _axis(issues: np.ndarray, lower: int, upper: int) -> None:
    _require(isinstance(issues, np.ndarray) and issues.dtype == np.int64
             and issues.ndim == 1 and len(issues) > 0
             and bool(np.all(np.diff(issues) > 0))
             and bool(np.all(issues % TICK_NS == 0))
             and bool(np.all((issues >= lower) & (issues < upper))), 'ISSUE_PARTITION')


def _now_bits(tape: TimeQueryTargetTape, issues: np.ndarray) -> np.ndarray:
    """S_s(t) uses only the exact M1 minute at the closed M5 issue."""
    at = np.searchsorted(tape.validity_ns, issues)
    present = at < len(tape.validity_ns)
    present[present] &= tape.validity_ns[at[present]] == issues[present]
    out = np.full((len(issues), len(TIMEFRAMES)), 7, dtype=np.uint8)
    out[present] = tape.allowed_bits[at[present]]
    return out


def _week_start(issues: np.ndarray) -> np.ndarray:
    """UTC Monday date of each M5 issue; never use future target time."""
    days = issues // (24 * 60 * 60 * 1_000_000_000)
    monday = days - (days + 3) % 7  # 1970-01-01 was Thursday.
    return (np.datetime64('1970-01-01', 'D')
            + monday.astype('timedelta64[D]')).astype(str)


def _empty_counts() -> dict:
    return {name: 0 for name in (
        'issues', 'mature', 'exact_m1', 'activity_known', 'active',
        'side_known', 'up', 'singleton', 'down', 'flat', 'side_up',
    )}


def _fit_counts(tape: TimeQueryTargetTape, issues: np.ndarray, k: int,
                upper: int, chunk_rows: int) -> list[dict]:
    counts = [_empty_counts() for _ in TIMEFRAMES]
    query = np.minimum(k, _K)
    for start in range(0, len(issues), chunk_rows):
        batch = issues[start:start + chunk_rows]
        ks = np.broadcast_to(query, (len(batch), len(TIMEFRAMES))).copy()
        got = tape.gather(batch, ks, fold_upper_ns=upper)
        for j in range(len(TIMEFRAMES)):
            if k > _K[j]:
                continue
            bits = got['allowed'][:, j]
            valid = got['eligible'][:, j]
            c = counts[j]
            c['issues'] += len(batch)
            c['mature'] += int(got['mature'][:, j].sum())
            c['exact_m1'] += int((got['m1_source_row'][:, j] >= 0).sum())
            c['activity_known'] += int(valid.sum())
            c['active'] += int((valid & np.isin(bits, (1, 4, 5))).sum())
            c['side_known'] += int((valid & np.isin(bits, (1, 4))).sum())
            c['up'] += int((valid & (bits == 4)).sum())
            c['singleton'] += int((valid & np.isin(bits, (1, 2, 4))).sum())
            c['down'] += int((valid & (bits == 1)).sum())
            c['flat'] += int((valid & (bits == 2)).sum())
            c['side_up'] += int((valid & (bits == 4)).sum())
    return counts


def _priors(c: dict) -> dict:
    _require(c['activity_known'] == c['active'] + c['flat']
             and c['side_known'] == c['down'] + c['up']
             and c['singleton'] == c['down'] + c['flat'] + c['side_up'],
             'FIT_PARTITION')
    p_active = (c['active'] + 1) / (c['activity_known'] + 2)
    p_up = (c['up'] + 1) / (c['side_known'] + 2)
    denom = c['singleton'] + 3
    return {'activity': p_active, 'conditional_up': p_up,
            'three_class': [(c[name] + 1) / denom
                            for name in ('down', 'flat', 'side_up')]}


def _empty_score() -> dict:
    return {name: 0 for name in (
        'issues', 'mature', 'exact_m1', 'activity_known', 'side_known',
        'singleton', 'active_side_unknown', 'all_unknown',
        'activity_prior_brier_sum', 'side_prior_brier_sum',
        'three_class_prior_brier_sum',
        'activity_persistence_count', 'activity_persistence_brier_sum',
        'activity_prior_paired_brier_sum', 'side_persistence_count',
        'side_persistence_brier_sum', 'side_prior_paired_brier_sum',
    )}


def _score_chunk(score: dict, bits: np.ndarray, source: np.ndarray,
                 mature: np.ndarray, eligible: np.ndarray,
                 now: np.ndarray, prior: dict) -> None:
    score['issues'] += len(bits)
    score['mature'] += int(mature.sum())
    score['exact_m1'] += int((source >= 0).sum())
    score['all_unknown'] += int((bits == 7).sum())
    score['active_side_unknown'] += int((eligible & (bits == 5)).sum())
    activity = eligible
    side = eligible & np.isin(bits, (1, 4))
    singleton = eligible & np.isin(bits, (1, 2, 4))
    score['activity_known'] += int(activity.sum())
    score['side_known'] += int(side.sum())
    score['singleton'] += int(singleton.sum())
    y_active = bits[activity] != 2
    y_up = bits[side] == 4
    p_active, p_up = prior['activity'], prior['conditional_up']
    score['activity_prior_brier_sum'] += float(np.square(p_active - y_active).sum(dtype=np.float64))
    score['side_prior_brier_sum'] += float(np.square(p_up - y_up).sum(dtype=np.float64))
    sbits = bits[singleton]
    if len(sbits):
        labels = np.where(sbits == 1, 0, np.where(sbits == 2, 1, 2))
        p3 = np.asarray(prior['three_class'], dtype=np.float64)
        score['three_class_prior_brier_sum'] += float(
            (len(labels) * np.square(p3).sum()
             - 2 * p3[labels].sum() + len(labels)))
    paired_activity = activity & np.isin(now, (1, 2, 4, 5))
    if paired_activity.any():
        true = bits[paired_activity] != 2
        pred = now[paired_activity] != 2
        score['activity_persistence_count'] += int(paired_activity.sum())
        score['activity_persistence_brier_sum'] += int(np.count_nonzero(pred != true))
        score['activity_prior_paired_brier_sum'] += float(
            np.square(p_active - true).sum(dtype=np.float64))
    paired_side = side & np.isin(now, (1, 4))
    if paired_side.any():
        true = bits[paired_side] == 4
        pred = now[paired_side] == 4
        score['side_persistence_count'] += int(paired_side.sum())
        score['side_persistence_brier_sum'] += int(np.count_nonzero(pred != true))
        score['side_prior_paired_brier_sum'] += float(
            np.square(p_up - true).sum(dtype=np.float64))


def score_reference_fold(tape: TimeQueryTargetTape, fit_issues: np.ndarray,
                         score_issues: np.ndarray, *, fit_lower_ns: int,
                         fold_boundary_ns: int, score_upper_ns: int,
                         chunk_rows: int = 8192) -> dict:
    """Freeze prior on earlier mature TRAIN; score all declared future M5 ticks.

    Each k has its own fit prior. Consumers must macro-average TF and account
    for overlapping issues; raw cell counts are not independent examples.
    """
    _require(isinstance(tape, TimeQueryTargetTape)
             and all(type(x) is int for x in
                     (fit_lower_ns, fold_boundary_ns, score_upper_ns, chunk_rows))
             and 0 <= fit_lower_ns < fold_boundary_ns < score_upper_ns
             and 1 <= chunk_rows <= 65536, 'FOLD_SHAPE')
    _axis(fit_issues, fit_lower_ns, fold_boundary_ns)
    _axis(score_issues, fold_boundary_ns, score_upper_ns)
    now = _now_bits(tape, score_issues)
    weeks = _week_start(score_issues)
    week_names, week_n = np.unique(weeks, return_counts=True)
    week_issue_counts = dict(zip(week_names.tolist(), week_n.tolist()))
    report = {tf: {} for tf in TIMEFRAMES}
    weekly = {tf: {} for tf in TIMEFRAMES}
    exact_sources = {tf: set() for tf in TIMEFRAMES}
    informative_sources = {tf: set() for tf in TIMEFRAMES}
    for k in range(1, int(_K.max()) + 1):
        fit = _fit_counts(tape, fit_issues, k, fold_boundary_ns, chunk_rows)
        prior = [_priors(c) for c in fit]
        query = np.minimum(k, _K)
        scores = [_empty_score() for _ in TIMEFRAMES]
        for start in range(0, len(score_issues), chunk_rows):
            batch = score_issues[start:start + chunk_rows]
            ks = np.broadcast_to(query, (len(batch), len(TIMEFRAMES))).copy()
            got = tape.gather(batch, ks, fold_upper_ns=score_upper_ns)
            batch_weeks = weeks[start:start + len(batch)]
            for j, tf in enumerate(TIMEFRAMES):
                if k > _K[j]:
                    continue
                bits = got['allowed'][:, j]
                source = got['m1_source_row'][:, j]
                mature = got['mature'][:, j]
                eligible = got['eligible'][:, j]
                current = now[start:start + len(batch), j]
                _score_chunk(scores[j], bits, source, mature, eligible,
                             current, prior[j])
                exact_sources[tf].update(source[source >= 0].tolist())
                informative_sources[tf].update(source[eligible].tolist())
                for week in np.unique(batch_weeks):
                    choose = batch_weeks == week
                    bucket = weekly[tf].setdefault(str(week), _empty_score())
                    _score_chunk(bucket, bits[choose], source[choose],
                                 mature[choose], eligible[choose],
                                 current[choose], prior[j])
        for j, tf in enumerate(TIMEFRAMES):
            if k <= _K[j]:
                report[tf][str(k)] = {'fit': fit[j], 'prior': prior[j],
                                      'score': scores[j]}
    source_reuse = {}
    weekly_report = {}
    for tf in TIMEFRAMES:
        exact_cells = sum(v['score']['exact_m1'] for v in report[tf].values())
        informative_cells = sum(v['score']['activity_known'] for v in report[tf].values())
        unique_exact = len(exact_sources[tf])
        unique_informative = len(informative_sources[tf])
        _require(unique_exact <= exact_cells and unique_informative <= informative_cells
                 and unique_informative <= unique_exact, 'SOURCE_REUSE')
        source_reuse[tf] = {
            'exact_query_cells': exact_cells,
            'unique_exact_m1_source_rows': unique_exact,
            'exact_cells_per_unique_source': exact_cells / unique_exact if unique_exact else None,
            'informative_query_cells': informative_cells,
            'unique_informative_m1_source_rows': unique_informative,
            'informative_cells_per_unique_source': (
                informative_cells / unique_informative if unique_informative else None),
        }
        weekly_report[tf] = {}
        for week, raw in sorted(weekly[tf].items()):
            row = dict(raw)
            row['query_cells'] = row.pop('issues')
            row['unique_issue_rows'] = week_issue_counts[week]
            weekly_report[tf][week] = row
        for key in ('activity_known', 'side_known', 'singleton',
                    'activity_persistence_count', 'side_persistence_count'):
            _require(sum(v[key] for v in weekly[tf].values())
                     == sum(v['score'][key] for v in report[tf].values()),
                     'WEEK_PARTITION:' + tf + ':' + key)
    return {'schema_version': 'gx1_market_state_time_query_reference_core_v1',
            'scope': 'TRAIN_only_pure_fold_no_model_or_TEST',
            'fit_issues': len(fit_issues), 'score_issues': len(score_issues),
            'fit_issue_axis_sha256': hashlib.sha256(fit_issues.tobytes()).hexdigest(),
            'score_issue_axis_sha256': hashlib.sha256(score_issues.tobytes()).hexdigest(),
            'fit_lower_ns': fit_lower_ns, 'fold_boundary_ns': fold_boundary_ns,
            'score_upper_ns': score_upper_ns, 'per_tf_k': report,
            'weekly_by_issue_monday_utc': weekly_report,
            'score_source_reuse': source_reuse,
            'model_fits': 0, 'test_data_used': False}
