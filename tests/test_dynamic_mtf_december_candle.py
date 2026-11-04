import numpy as np
import pytest

from gx1.audit.dynamic_mtf_december_candle import _closed_window_ends, _native_labels, _rank_direction
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target


def test_rank_uses_continuous_amplitude_and_half_credit_for_ties():
    raw = np.array([1., 2., 2., 4., -20.])
    y = np.array([0, 1, 0, 1, -1], dtype=np.int8)
    ranked = _rank_direction(raw, y, y >= 0)
    assert ranked['up'] == ranked['down'] == 2
    assert ranked['pairs'] == 4 and ranked['ties'] == 1
    assert ranked['twice_wins'] == 7
    assert ranked['raw_auc'] == 7 / 8 and ranked['negative_auc'] == 1 / 8 and ranked['orientation'] == 1
    reverse = _rank_direction(-raw, y, y >= 0)
    assert reverse['orientation'] == -1 and reverse['oriented_auc'] == ranked['oriented_auc']


def test_constant_score_tie_selects_positive_but_absent_outcome_cannot_select():
    binary = np.array([1, 0], dtype=np.int8)
    result = _rank_direction(np.zeros(2), binary, binary >= 0)
    assert result['raw_auc'] == result['negative_auc'] == 0.5 and result['orientation'] == 1
    binary = np.array([-1, 0], dtype=np.int8)
    result = _rank_direction(np.array([1., 0.]), binary, binary >= 0)
    assert result['eligible_rows'] == 1 and result['down'] == 1 and result['pairs'] == 0
    assert result['raw_auc'] is result['negative_auc'] is result['orientation'] is None


def test_closed_route_uses_real_older_bar_across_a_gap_and_equality_at_close():
    opens = np.array([2, 4, 6, 18], dtype=np.int64)
    issues = np.array([7, 8, 19, 20], dtype=np.int64)
    assert _closed_window_ends(opens, issues, shift_ns=2, window=2, warmup=0).tolist() == [1, 2, 2, 3]
    with pytest.raises(ValueError, match='CANDLE_WARMUP'):
        _closed_window_ends(opens, issues, shift_ns=2, window=2, warmup=1)


def test_native_labels_distinguish_exact_maturity_cutoff_missing_and_unknown():
    minute = target.MINUTE_NS
    issue = np.array([10, 20, 30, 40], dtype=np.int64) * minute
    # +5 queries: 15 is UP, 25 has no source, 35 is observed FLAT, 45 is exactly cutoff.
    axis = np.array([10, 15, 20, 30, 35, 40], dtype=np.int64) * minute
    states = np.zeros((len(axis), 5), dtype=np.int8)
    states[1, 0], states[4, 0] = target.UP, target.FLAT
    obs = np.ones_like(states, dtype=bool)
    obs[0, 0] = False
    states[0, 0] = target.UNKNOWN
    arrays = {'parent_rows': np.arange(len(issue), dtype=np.int64), 'issue_ns': issue,
              'issue_source_index': np.searchsorted(axis, issue), 'validity_ns': axis, 'state': states,
              'observed': obs, 'reason': np.zeros_like(states, dtype=np.uint8), 'path_observed': obs,
              'maturity_ns': np.broadcast_to(axis[:, None], states.shape), 'atr_available_ns': np.zeros_like(states, dtype=np.int64)}
    cutoff = 45 * minute
    series = target.DynamicMtfStateSeries(arrays, {'cutoff_ns': cutoff})
    labels, reasons, observed, digest = _native_labels(series, cutoff, np.array([5, 15, 60, 240, 1440]))
    assert observed[:, 0].tolist() == [True, False, True, False]
    assert labels[:, 0].tolist() == [target.UP, target.UNKNOWN, target.FLAT, target.UNKNOWN]
    assert reasons[1, 0] == target.QUERY_UNAVAILABLE and reasons[-1, 0] == target.IMMATURE
    assert len(digest) == 64
