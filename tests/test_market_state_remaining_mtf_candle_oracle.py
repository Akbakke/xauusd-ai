"""Synthetic, source-independent candle geometry checks; no cache payloads."""
import numpy as np
import pandas as pd
import pytest

from gx1.audit.market_state_remaining_mtf_candle_oracle_v1 import (
    CACHE_COLUMNS, CANDLE_CACHE_NAMES, candle21_from_closed_ohlc,
)
from gx1.features.entry_candle_primitives_v1 import (
    CANDLE_PRIMITIVE_FEATURE_NAMES, build_entry_candle_primitive_layer,
)


def _bars(rows):
    values = np.asarray(rows, dtype=np.float64)
    return tuple(values[:, j].copy() for j in range(4))


def test_order_is_bound_to_current_candle_contract():
    assert CACHE_COLUMNS == tuple(range(29, 50))
    assert CANDLE_CACHE_NAMES == tuple(
        'mtf_' + name.replace('.', '_') for name in CANDLE_PRIMITIVE_FEATURE_NAMES)
    assert len(set(CANDLE_CACHE_NAMES)) == 21


def test_hand_calculated_warmup_units_events_and_zero_range():
    # O/H/L/C; bar 1 gaps above bar 0 and bearish body contains its bull body.
    o, h, l, c = _bars([(100, 102, 98, 101), (103, 104, 100, 100),
                        (100, 100, 100, 100), (100, 100, 100, 100)])
    a = candle21_from_closed_ohlc(o, h, l, c)
    assert a.shape == (4, 21) and a.dtype == np.float32
    np.testing.assert_allclose(a[0, :3], [.25, .25, .5], rtol=0, atol=0)
    assert a[0, 19] == 1  # body-direction duration exists on the first bar
    assert np.isnan(a[0, [*range(3, 19), 20]]).all()
    np.testing.assert_allclose(
        a[1, :11], [-.75, .25, 0, 1.25, -.25, .5, .5, .5, .25, 0, .25],
        rtol=0, atol=0)
    assert a[1, 11:19].tolist() == [1, 0, 0, 0, 0, 1, 1, 0]
    assert a[1, 19:].tolist() == [1, 1]
    assert a[2, :3].tolist() == [0, 0, 0]
    assert a[2, 12] == a[2, 14] == 1  # body and range strictly contained
    assert a[2, 19:].tolist() == [1, 1]
    assert a[3, 3] == 0  # storage zero when previous range is exactly zero
    assert a[3, 11:19].tolist() == [0, 0, 0, 0, 0, 0, 0, 0]
    assert a[3, 19:].tolist() == [2, 1]  # doji run, then equal-range state


def test_zero_previous_range_gap_and_unbounded_observed_age():
    o, h, l, c = _bars([(100, 100, 100, 100), (102, 103, 101, 102)])
    a = candle21_from_closed_ohlc(o, h, l, c)
    assert a[1, 3] == 0
    np.testing.assert_allclose(a[1, 8], np.float32(2 / 3), rtol=0, atol=0)
    assert a[1, 9] == 0
    long = np.tile(np.asarray([[100, 102, 99, 101]], dtype=np.float64), (1050, 1))
    age = candle21_from_closed_ohlc(*(long[:, j].copy() for j in range(4)))
    assert age[0, 19] == 1 and age[-1, 19] == 1050
    assert np.isnan(age[0, 20]) and age[1, 20] == 1 and age[-1, 20] == 1049


def test_selected_positions_keep_full_history_and_masks():
    o, h, l, c = _bars([(100, 102, 99, 101)] * 20)
    full = candle21_from_closed_ohlc(o, h, l, c)
    selected = candle21_from_closed_ohlc(
        o, h, l, c, positions=np.array([0, 1, 7, 19], np.int64))
    assert np.array_equal(selected.view(np.uint32), full[[0, 1, 7, 19]].view(np.uint32))
    assert selected[-1, 19] == 20 and selected[-1, 20] == 19


def test_rejects_invalid_source_and_selection():
    o, h, l, c = _bars([(100, 102, 99, 101), (101, 103, 100, 102)])
    cases = [
        (o.astype(np.float32), h, l, c, 'FLOAT64_OHLC'),
        (o, h, np.array([99, np.nan]), c, 'FINITE_POSITIVE_OHLC'),
        (o, np.array([102., 101.]), l, c, 'GEOMETRY'),
    ]
    for args in cases:
        with pytest.raises(ValueError, match=args[-1]):
            candle21_from_closed_ohlc(*args[:-1])
    with pytest.raises(ValueError, match='POSITIONS'):
        candle21_from_closed_ohlc(o, h, l, c,
                                  positions=np.array([1, 0], np.int64))


def test_independent_oracle_matches_producer_on_mixed_synthetic_bars():
    rng = np.random.default_rng(17)
    n = 1200
    close = 1900 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, .5, n)
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    for i in range(27, n, 89):
        open_[i] = high[i] = low[i] = close[i] = close[i - 1]
    frame = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='5min', tz='UTC'),
        'open': open_, 'high': high, 'low': low, 'close': close})
    independent = candle21_from_closed_ohlc(open_, high, low, close)
    producer, names = build_entry_candle_primitive_layer(frame)
    assert tuple('mtf_' + name.replace('.', '_') for name in names) == CANDLE_CACHE_NAMES
    assert np.array_equal(independent.view(np.uint32), producer.view(np.uint32))
