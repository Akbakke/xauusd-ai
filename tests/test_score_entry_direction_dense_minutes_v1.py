import numpy as np
import pytest

from gx1.scripts import score_entry_direction_dense_minutes_v1 as score


def test_week_pair_aggregation_matches_direct_ranking_with_ties():
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1], np.int8)
    z = np.array([0., 0., 2., 1., 1., 3., -1., 2.], np.float64)
    day = np.array([0, 0, 1, 1, 2, 2, 3, 3], np.int64)
    week_of_day = np.array([0, 0, 1, 1], np.int64)
    days = score.rank.block_rank_pairs(y, z, day)
    week_ids, weeks = score.coarsen_block_pairs(days, week_of_day)
    direct = score.rank.block_rank_pairs(y, z, week_of_day[day])
    assert week_ids.tolist() == [0, 1]
    for key in ('positive', 'negative', 'twice_wins'):
        np.testing.assert_array_equal(weeks[key], direct[key])


def test_simultaneous_band_uses_the_largest_horizon_deviation():
    points = np.array([.50, .60], np.float64)
    draws = np.array([[.51, .63], [.48, .59], [.55, .62]], np.float64)
    band = score.simultaneous_intervals(points, draws)
    expected = np.quantile([.03, .02, .05], .975, method='linear')
    assert band['half_width'] == pytest.approx(expected)
    np.testing.assert_allclose(band['low'], points - expected)
    assert band['per_scheme_alpha'] == .025 and band['horizons_simultaneous'] == 2


def synthetic_case(offsets=False):
    minute, day = score.dense.MINUTE_NS, score.DAY_NS
    base = np.int64(1704067200000000000)
    decisions = np.array([base + week * 7 * day + m * minute
        for week in range(40) for m in (5, 10)], np.int64)
    n = len(decisions)
    terminal = decisions[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * minute
    signs = np.broadcast_to(np.where(np.arange(n) % 2, 1, -1).astype(np.int8)[:, None], (n, 60)).copy()
    target = {'parent_rows': np.arange(n, dtype=np.int64), 'bar_start_ns': decisions - 5 * minute,
        'decision_ns': decisions, 'terminal_time_ns': terminal,
        'observed': np.ones((n, 60), np.bool_), 'sign': signs}
    folds = []; predictions = []; reports = []
    for number in range(1, 5):
        start, end = 16 * number, 16 * (number + 1)
        fold = {'fold_id': number, 'prefix_rows': start, 'oof_start_row': start, 'oof_end_row': end,
            'train_cutoff_ns': int(base + 8 * number * 7 * day),
            'oof_cutoff_ns': int(base + 8 * (number + 1) * 7 * day)}
        rows = np.arange(start, end, dtype=np.int64)
        logits = np.where(rows % 2, 1., -1.).astype(np.float32)
        if offsets: logits += np.float32(10 * number)
        pred = {'parent_rows': rows, 'bar_start_ns': target['bar_start_ns'][rows],
            'decision_ns': decisions[rows], 'terminal_time_ns': terminal[rows][:, [0, 4, 14]],
            'observed': target['observed'][rows][:, [0, 4, 14]], 'sign': signs[rows][:, [0, 4, 14]],
            'logit_up': np.repeat(logits[:, None], 3, axis=1)}
        labels = (signs[rows, 4] == 1).astype(np.int8)
        metrics = score.probability.direction_probability_metrics(labels, logits.astype(np.float64))
        report = {'fold': fold, 'test_data_used': False, 'model_metrics': {'per_horizon': [None,
            {'horizon_minutes': 5, 'counts': {'eligible_cells': len(rows)},
             'logloss': metrics['logloss'], 'brier': metrics['brier']}]}}
        folds.append(fold); predictions.append(pred); reports.append(report)
    return target, folds, predictions, reports


def test_all_60_results_use_within_fold_pairs_not_cross_fold_score_scale():
    a = score.score_fixed_signal(*synthetic_case(), replicates=20)
    b = score.score_fixed_signal(*synthetic_case(offsets=True), replicates=20)
    assert len(a['panels']) == 60
    assert [x['horizon_minutes'] for x in a['panels']] == list(range(1, 61))
    assert all(x['within_fold_pair_weighted_auc'] == 1 and x['both_band_lower_above_half'] for x in a['panels'])
    assert [x['within_fold_pair_weighted_auc'] for x in a['panels']] == [x['within_fold_pair_weighted_auc'] for x in b['panels']]
    assert not a['economic_edge_qualified'] and not a['test_data_used']


def test_changed_anchor_or_parent_clock_fails_before_scoring():
    target, folds, predictions, reports = synthetic_case()
    predictions[0]['sign'][0, 0] *= -1
    with pytest.raises(RuntimeError, match='ANCHOR_JOIN:sign'):
        score.score_fixed_signal(target, folds, predictions, reports, replicates=2)
