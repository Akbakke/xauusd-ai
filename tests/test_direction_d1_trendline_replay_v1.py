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


def test_m5_window_uses_five_minute_closed_clock():
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import window_exposure
    minute = 60 * 10**9
    ts = np.arange(10, dtype=np.int64) * (5 * minute)
    bad = np.zeros(10, dtype=np.bool_); bad[2] = True
    got = window_exposure(ts, bad, np.array([20, 35], dtype=np.int64) * minute,
        3, shift_ns=5 * minute, timeframe='M5')
    assert got['decisions_with_any_mismatched_m5_row'] == 1
    assert got['distinct_mismatched_m5_rows_in_windows'] == 1


def test_price_precision_summary_counts_collapsed_levels_and_signs():
    import pandas as pd
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import price_precision_summary
    frame = pd.DataFrame({'open':[2000., 2000.], 'high':[2001., 2001.], 'low':[1999., 1999.],
        'close':[2000., 2000.00001]})
    result = price_precision_summary(frame)
    assert result['m5_rows'] == 2 and result['distinct_original_closes'] == 2
    assert result['distinct_float32_closes'] == 1 and result['distinct_original_closes_collapsed'] == 1
    assert result['m5_close_open_direction_changed'] == 1 and result['max_absolute_error_usd'] > 0


def test_native_parts_prefix_enforces_chronological_unique_time(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix, CUTOFF_NS
    refs = []
    for year in (2023, 2024):
        times = pd.date_range(f'{year}-01-01', periods=2, freq='5min', tz='UTC')
        frame = pd.DataFrame({'time':times,'open':[1., 1.], 'high':[2., 2.], 'low':[.5, .5],
            'close':[1., 1.], 'volume':[1, 1]})
        path = tmp_path / f'{year}.parquet'; pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        refs.append({'path':str(path), 'sha256':'unused_in_pure_helper'})
    result = load_native_parts_prefix(refs, CUTOFF_NS)
    assert len(result) == 4 and result.index.is_monotonic_increasing
    with pytest.raises(RuntimeError, match='NATIVE_PART_ORDER'):
        load_native_parts_prefix(list(reversed(refs)), CUTOFF_NS)
