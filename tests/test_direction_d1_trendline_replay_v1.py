import numpy as np
import pytest

from gx1.scripts.audit_direction_d1_trendline_replay_v1 import compare_fields


def test_cache_comparison_detects_float_bits_and_missingness():
    a = np.array([[0., np.nan], [1., 2.]], dtype=np.float32)
    ts = np.array([10, 20], dtype=np.int64)
    assert compare_fields(a, a.copy(), ('x', 'y'), ts)['all_finite_float32_bits_and_missingness_equal']
    b = a.copy(); b[0, 0] = -0.; b[0, 1] = 0.; b[1, 1] = 3.
    got = compare_fields(a, b, ('x', 'y'), ts)
    assert got['mismatched_cells'] == 3
    assert got['per_field'][0]['first_mismatch_timestamp_ns'] == 10
    assert got['per_field'][1]['mismatched_rows'] == 2


def test_cache_comparison_rejects_invalid_geometry():
    with pytest.raises(RuntimeError, match='COMPARE_SHAPE'):
        compare_fields(np.ones((2, 1)), np.ones((3, 1)), ('x',), np.array([1, 2]))
    with pytest.raises(RuntimeError, match='COMPARE_SHAPE'):
        compare_fields(np.array([[np.inf]]), np.array([[np.inf]]), ('x',), np.array([1]))


def test_window_exposure_respects_close_and_real_lookback():
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import window_exposure
    day = 86400 * 10**9
    ts = np.arange(10, dtype=np.int64) * day
    bad = np.zeros(10, dtype=np.bool_); bad[[2, 7]] = True
    decisions = np.array([4, 7, 9], dtype=np.int64) * day
    result = window_exposure(ts, bad, decisions, 3)
    assert result['decisions_with_any_mismatched_d1_row'] == 2
    assert result['distinct_mismatched_d1_rows_in_windows'] == 2
    assert result['first_affected_decision_ns'] == 4 * day
    assert result['last_affected_decision_ns'] == 9 * day


def test_mismatch_classification_separates_missingness_and_finite_bits():
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import mismatch_matrix
    a = np.array([[np.nan, 0.], [1., 2.]], dtype=np.float32)
    b = np.array([[0., -0.], [1., 3.]], dtype=np.float32)
    missing, finite = mismatch_matrix(a, b)
    assert missing.sum() == 1 and finite.sum() == 2


def test_h4_windows_use_four_hour_close_and_distinct_names():
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import window_exposure
    hour = 3600 * 10**9
    ts = np.arange(10, dtype=np.int64) * (4 * hour)
    bad = np.zeros(10, dtype=np.bool_); bad[2] = True
    decisions = np.array([16, 28], dtype=np.int64) * hour
    got = window_exposure(ts, bad, decisions, 3, shift_ns=4 * hour, timeframe='H4')
    assert got['decisions_with_any_mismatched_h4_row'] == 1
    assert got['distinct_mismatched_h4_rows_in_windows'] == 1
