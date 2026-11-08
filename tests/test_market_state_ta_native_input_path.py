"""Critical audit defects: close boundaries, warmup, sparse windows and bit masks."""
import numpy as np
import pytest

from gx1.audit import verify_market_state_ta_native_input_path_v1 as audit


def test_closed_window_includes_boundary_and_never_open_bar():
    clock = np.array([0, 10, 20, 30, 40], dtype=np.int64)
    decisions = np.array([30, 39, 40], dtype=np.int64)
    got = audit.closed_windows(clock, decisions, duration_ns=10, length=2, warmup=0)
    assert got.tolist() == [[1, 2], [1, 2], [2, 3]]
    assert np.all(clock[got] + 10 <= decisions[:, None])


def test_sparse_observed_clock_does_not_invent_gap_rows():
    clock = np.array([0, 10, 50, 60], dtype=np.int64)
    got = audit.closed_windows(clock, np.array([60], np.int64),
                              duration_ns=10, length=3, warmup=0)
    assert clock[got].tolist() == [[0, 10, 50]]


def test_last_endpoint_after_warmup_does_not_make_early_history_valid():
    with pytest.raises(RuntimeError, match='HISTORY_WARMUP'):
        audit.closed_windows(np.array([0, 10, 20, 30], np.int64),
            np.array([40], np.int64), duration_ns=10, length=3, warmup=2)


@pytest.mark.parametrize('clock', ([0, 10, 10], [0, 20, 10]))
def test_nonunique_or_reversed_clock_is_rejected(clock):
    with pytest.raises(RuntimeError, match='CLOCK_INPUT'):
        audit.closed_windows(np.array(clock, np.int64), np.array([30], np.int64),
                             duration_ns=10, length=1, warmup=0)


def test_clock_addition_overflow_is_rejected():
    maximum = np.iinfo(np.int64).max
    with pytest.raises(RuntimeError, match='CLOCK_OVERFLOW'):
        audit.closed_windows(np.array([maximum - 1], np.int64),
            np.array([maximum], np.int64), duration_ns=10, length=1, warmup=0)


def test_history_union_retains_fit_score_gap_and_deduplicates_overlap():
    rows, intervals = audit.interval_union(np.array([4, 5, 5, 12], np.int64), 3)
    assert intervals == [[2, 6], [10, 13]]
    assert rows.tolist() == [2, 3, 4, 5, 10, 11, 12]
    assert [part.tolist() for part in audit.contiguous_parts(rows)] == [[2, 3, 4, 5], [10, 11, 12]]


def test_exact_rejects_zero_mask_changes_and_signed_zero():
    value = np.array([[1., 0., -2.]], np.float32)
    audit.exact(value.copy(), value, label='valid')
    for replacement in (-0., 1.):
        changed = value.copy(); changed[0, 1] = replacement
        with pytest.raises(RuntimeError, match='BIT_PARITY'):
            audit.exact(changed, value, label='mask')


def test_exact_never_treats_matching_missing_values_as_valid_model_input():
    values = np.array([[np.nan]], np.float32)
    with pytest.raises(RuntimeError, match='NONFINITE'):
        audit.exact(values, values.copy(), label='missing')
    with pytest.raises(RuntimeError, match='SHAPE_DTYPE'):
        audit.exact(np.ones((1, 1), np.float64), np.ones((1, 1), np.float32), label='precision')


def test_context_bridge_accepts_separate_rsi_rounding_and_checks_utc_boundaries():
    from types import SimpleNamespace
    from gx1.features import htf_features as htf
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    profile = compile_full_ta_entry_execution_profile()
    names = list(profile['physical_field_names']['signal'])
    times = np.array(['2024-11-03T06:55:00', '2024-11-03T23:55:00'], dtype='datetime64[ns]').astype(np.int64)
    local = np.zeros((2, len(names)), np.float32)
    cache = {}
    # Independently rounded raw and centered wires can differ after inversion,
    # most visibly near RSI zero. A generic bit-equality rule would be wrong.
    original_rsi = np.array([1.e-6, 99.999997], np.float64)
    raw = original_rsi.astype(np.float32)
    centered = ((original_rsi - 50.) / 50.).astype(np.float32)
    for tf in audit.TIMEFRAMES:
        columns = list(htf.MULTI_TF_PER_BAR_FEATURES_V4)
        values = np.zeros((2, len(columns)), np.float32)
        values[:, columns.index('rsi14_centered')] = centered
        clock = times + 300_000_000_000 - int(htf.MULTI_TF_SHIFT[tf].value)
        cache[tf] = SimpleNamespace(columns=columns, attrs={'ts_int64': clock,
            'feats_np': values, 'causal_warmup_rows': 0})
        local[:, names.index('ctx_cont.' + tf.lower() + '_rsi14_canon_v2')] = raw
    hour = np.array([7., 0.], np.float32)
    weekday = np.array([6., 0.], np.float32)
    for field, value in (
        ('hour_sin', np.sin(2 * np.pi * hour / 24)),
        ('hour_cos', np.cos(2 * np.pi * hour / 24)),
        ('dow_sin', np.sin(2 * np.pi * weekday / 7)),
        ('minutes_since_session_open', np.array([0., 120.], np.float32)),
    ):
        local[:, names.index('ctx_cont.' + field)] = value
    rows, report = audit._local_context_formula_bridges(local, times, cache, profile)
    assert report['fields'] == 37 and report['checked_values'] == 74
    assert all(values.tolist() == [0, 1] for values in rows.values())
    assert report['per_field']['m5_rsi14_canon_v2']['maximum_absolute_error'] > 0.
    assert all(item['available_rows'] == 2 for item in report['per_field'].values())
    changed = local.copy()
    changed[0, names.index('ctx_cont.m5_rsi14_canon_v2')] = 1.e-4
    with pytest.raises(RuntimeError, match='LOCAL_CONTEXT_RSI_ROUNDING'):
        audit._local_context_formula_bridges(changed, times, cache, profile)
    changed = local.copy()
    changed[1, names.index('ctx_cont.minutes_since_session_open')] = 121.
    with pytest.raises(RuntimeError, match='BIT_PARITY:LOCAL_CONTEXT:minutes_since_session_open'):
        audit._local_context_formula_bridges(changed, times, cache, profile)
