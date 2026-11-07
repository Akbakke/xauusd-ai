"""Small synthetic checks for the independent seven-field basic_v1 oracle."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gx1.audit.market_state_remaining_mtf_basic7_oracle_v1 import (
    CACHE_COLUMNS,
    FIELD_NAMES,
    UNITS,
    WARMUP_ROWS,
    independent_closed_tf_basic7_oracle,
)
from gx1.features.basic_v1 import build_basic_v1
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4


def _frame(n: int = 180) -> pd.DataFrame:
    row = np.arange(n, dtype=np.float64)
    close = 2000.0 + 0.05 * row + 2.0 * np.sin(0.37 * row) + 0.9 * np.sin(0.093 * row)
    half_range = 0.4 + 0.2 * (1.0 + np.sin(0.23 * row))
    index = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + half_range,
        "low": close - half_range,
        "close": close,
        "volume": 100.0 + row % 13,
        "spread_pct": np.full(n, 1e-4),
    }, index=index)


def _oracle(frame: pd.DataFrame, cutoff: int | None = None) -> dict:
    return independent_closed_tf_basic7_oracle(
        frame.high.to_numpy(), frame.low.to_numpy(), frame.close.to_numpy(),
        available_ns=frame.index.asi8 + 300_000_000_000,
        train_end_exclusive_ns=cutoff,
    )


def test_exact_cache_columns_names_and_units() -> None:
    assert CACHE_COLUMNS == (194, 195, 197, 198, 199, 200, 201)
    assert tuple(MULTI_TF_PER_BAR_FEATURES_V4[i] for i in CACHE_COLUMNS) == FIELD_NAMES
    assert WARMUP_ROWS == (19, 12, 47, 48, 60, 118, 35)
    assert UNITS == (
        "log_range_volatility", "price_relative_bandwidth_change",
        "range_zscore", "excess_kurtosis", "ATR_multiples",
        "bandwidth_relative_to_100_bar_mean", "ATR_multiples",
    )


def test_synthetic_closed_bar_owner_parity_and_warmup() -> None:
    frame = _frame()
    clock = frame.index.asi8 + 300_000_000_000
    oracle = _oracle(frame, int(clock[-1]) + 1)
    owner, _ = build_basic_v1(frame)
    assert oracle["values"].shape == (len(frame), 7)
    assert oracle["values"].dtype == np.float32
    for j, (name, first) in enumerate(zip(FIELD_NAMES, WARMUP_ROWS, strict=True)):
        got = oracle["values"][:, j]
        expected = owner[name].to_numpy(dtype=np.float32)
        assert np.isnan(got[:first]).all(), name
        assert np.isfinite(got[first:]).all(), name
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(oracle["available"][:, j], np.isfinite(got))
        assert not oracle["undefined_after_warmup"][:, j].any()


def test_parkinson_and_range_z_have_independent_analytic_values() -> None:
    n = 125
    close = np.full(n, 100.0)
    span = np.arange(n, dtype=np.float64) * 0.005 + 1.0
    high = close + span / 2.0
    low = close - span / 2.0
    clock = np.arange(1, n + 1, dtype=np.int64)
    out = independent_closed_tf_basic7_oracle(high, low, close, available_ns=clock)
    at = 47
    expected_pk = math.sqrt(sum(math.log(high[i] / low[i]) ** 2 for i in range(at - 19, at + 1)) / 20.0) / math.sqrt(4.0 * math.log(2.0))
    block = span[:48]
    expected_z = (span[at] - block.mean()) / block.std(ddof=0)
    assert np.isclose(out["values"][at, 0], expected_pk, rtol=1e-6)
    assert np.isclose(out["values"][at, 2], expected_z, rtol=1e-6)
    assert np.all(out["values"][12:, 1] == 0.0)  # constant close => bandwidth change zero


def test_undefined_denominators_remain_unavailable() -> None:
    n = 130
    close = np.full(n, 100.0)
    clock = np.arange(1, n + 1, dtype=np.int64)
    out = independent_closed_tf_basic7_oracle(close, close, close, available_ns=clock)
    assert np.all(out["values"][19:, 0] == 0.0)  # high/low log range is defined zero
    assert np.all(out["values"][12:, 1] == 0.0)
    for j in (2, 3, 4, 5, 6):
        assert np.isnan(out["values"][:, j]).all()
        assert out["undefined_after_warmup"][WARMUP_ROWS[j]:, j].all()
        assert not out["available"][:, j].any()


def test_future_changes_cannot_change_earlier_closed_bar_values() -> None:
    frame = _frame()
    initial = _oracle(frame)
    changed = frame.copy()
    changed.iloc[125:, changed.columns.get_loc("close")] += 40.0
    changed.iloc[125:, changed.columns.get_loc("high")] += 40.0
    changed.iloc[125:, changed.columns.get_loc("low")] += 40.0
    later = _oracle(changed)
    np.testing.assert_array_equal(initial["values"][:125], later["values"][:125])
    assert not np.array_equal(initial["values"][125:], later["values"][125:], equal_nan=True)


@pytest.mark.parametrize("bad", ["future_cutoff", "clock_order", "negative_price", "bad_geometry", "bad_dtype", "nonfinite"])
def test_invalid_input_fails_closed(bad: str) -> None:
    frame = _frame()
    h = frame.high.to_numpy().copy()
    l = frame.low.to_numpy().copy()
    c = frame.close.to_numpy().copy()
    clock = frame.index.asi8.copy() + 300_000_000_000
    cutoff = None
    if bad == "future_cutoff":
        cutoff = int(clock[-1])
    elif bad == "clock_order":
        clock[20] = clock[19]
    elif bad == "negative_price":
        l[0] = -1.0
    elif bad == "bad_geometry":
        h[0] = l[0] - 1.0
    elif bad == "bad_dtype":
        c = c.astype(np.int64)
    elif bad == "nonfinite":
        h[0] = np.nan
    with pytest.raises(ValueError, match="MTF_BASIC7_ORACLE_"):
        independent_closed_tf_basic7_oracle(h, l, c, available_ns=clock,
                                            train_end_exclusive_ns=cutoff)
