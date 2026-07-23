"""Tests for gx1/features/htf_features.py.

Verifies that on-the-fly HTF feature computation:
  - Returns None when warmup is insufficient (fail-soft, never returns made-up values).
  - Returns finite floats / ints when warmup is satisfied.
  - Matches the offline computation in add_ctx_cont_columns_to_prebuilt.py
    bit-for-bit on the same M5 OHLC input (modulo rounding floating point ops
    in the same order).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features import htf_features as htf
from gx1.scripts import add_ctx_cont_columns_to_prebuilt as offline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_m5(n_days: int = 300, seed: int = 7) -> pd.DataFrame:
    """Create synthetic M5 OHLC indexed by UTC DatetimeIndex.

    288 M5 bars per 24h, deterministic random walk OHLC.
    """
    rng = np.random.default_rng(seed)
    n_bars = n_days * 288
    start = pd.Timestamp("2024-06-01 00:00:00", tz="UTC")
    idx = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    rets = rng.normal(0.0, 0.001, size=n_bars)
    close = 1800.0 * np.exp(np.cumsum(rets))
    bar_range = np.abs(rng.normal(0.0, 0.5, size=n_bars))
    high = close + bar_range
    low = close - bar_range
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=idx
    )


# ---------------------------------------------------------------------------
# Input-validation tests
# ---------------------------------------------------------------------------


def test_validate_m5_input_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="HTF_INPUT_FAIL"):
        htf._validate_m5_input([1, 2, 3])  # type: ignore[arg-type]


def test_validate_m5_input_rejects_missing_columns() -> None:
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2025-01-01"], tz="UTC"))
    with pytest.raises(RuntimeError, match="missing required columns"):
        htf._validate_m5_input(df)


def test_validate_m5_input_rejects_non_datetime_index() -> None:
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.0]},
        index=[0],
    )
    with pytest.raises(RuntimeError, match="DatetimeIndex"):
        htf._validate_m5_input(df)


def test_vectorized_htf_alignment_uses_m5_decision_availability_time() -> None:
    target = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-06T12:05:00Z"),
            pd.Timestamp("2026-01-06T12:10:00Z"),
            pd.Timestamp("2026-01-06T12:15:00Z"),
        ]
    )
    htf_values = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-06T11:45:00Z"),
                pd.Timestamp("2026-01-06T12:00:00Z"),
            ]
        ),
    )

    aligned = htf._align_last_closed_tape(
        target,
        htf_values,
        pd.Timedelta(minutes=15),
    )

    assert aligned.tolist() == [1.0, 2.0, 2.0]


# ---------------------------------------------------------------------------
# Warmup tests (insufficient → None)
# ---------------------------------------------------------------------------


def test_d1_dist_from_ema200_atr_returns_none_below_220_d1_bars() -> None:
    m5 = _make_synthetic_m5(n_days=100)
    result = htf.compute_d1_dist_from_ema200_atr(m5, m5.index[-1])
    assert result is None


def test_h1_range_compression_returns_none_below_120_h1_bars() -> None:
    m5 = _make_synthetic_m5(n_days=2)
    result = htf.compute_h1_range_compression_ratio(m5, m5.index[-1])
    assert result is None


def test_m15_range_compression_returns_none_below_200_m15_bars() -> None:
    m5 = _make_synthetic_m5(n_days=1)
    short = m5.iloc[:30]
    result = htf.compute_m15_range_compression_ratio(short, short.index[-1])
    assert result is None


def test_h4_trend_sign_cat_returns_none_below_80_h4_bars() -> None:
    m5 = _make_synthetic_m5(n_days=10)
    result = htf.compute_h4_trend_sign_cat(m5, m5.index[-1])
    assert result is None


def test_d1_atr_percentile_252_returns_none_below_270_d1_bars() -> None:
    m5 = _make_synthetic_m5(n_days=260)
    result = htf.compute_d1_atr_percentile_252(m5, m5.index[-1])
    assert result is None


# ---------------------------------------------------------------------------
# Sufficient-warmup tests (computed → finite)
# ---------------------------------------------------------------------------


def test_d1_dist_from_ema200_atr_returns_finite_when_warmup_satisfied() -> None:
    m5 = _make_synthetic_m5(n_days=240)  # > 220 D1 bars
    result = htf.compute_d1_dist_from_ema200_atr(m5, m5.index[-1])
    assert result is not None
    assert np.isfinite(result)


def test_h1_range_compression_returns_finite_when_warmup_satisfied() -> None:
    m5 = _make_synthetic_m5(n_days=10)
    result = htf.compute_h1_range_compression_ratio(m5, m5.index[-1])
    assert result is not None
    assert np.isfinite(result)
    assert result > 0


def test_m15_range_compression_returns_finite_when_warmup_satisfied() -> None:
    m5 = _make_synthetic_m5(n_days=5)
    result = htf.compute_m15_range_compression_ratio(m5, m5.index[-1])
    assert result is not None
    assert np.isfinite(result)
    assert result > 0


def test_h4_trend_sign_cat_returns_int_in_0_1_2_when_warmup_satisfied() -> None:
    m5 = _make_synthetic_m5(n_days=30)
    result = htf.compute_h4_trend_sign_cat(m5, m5.index[-1])
    assert result is not None
    assert isinstance(result, int)
    assert result in {0, 1, 2}


def test_d1_atr_percentile_252_returns_value_in_0_1_when_warmup_satisfied() -> None:
    m5 = _make_synthetic_m5(n_days=280)
    result = htf.compute_d1_atr_percentile_252(m5, m5.index[-1])
    assert result is not None
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def test_compute_htf_features_returns_all_fields_when_warmup_satisfied() -> None:
    m5 = _make_synthetic_m5(n_days=280)
    result = htf.compute_htf_features(m5)
    assert result.d1_dist_from_ema200_atr is not None
    assert result.h1_range_compression_ratio is not None
    assert result.d1_atr_percentile_252 is not None
    assert result.m15_range_compression_ratio is not None
    assert result.h4_trend_sign_cat is not None
    assert result.insufficient_warmup_for_v1 == []


def test_compute_htf_features_lists_insufficient_when_short_buffer() -> None:
    m5 = _make_synthetic_m5(n_days=5)  # not enough for D1
    result = htf.compute_htf_features(m5)
    assert result.d1_dist_from_ema200_atr is None
    assert result.d1_atr_percentile_252 is None
    assert "D1_dist_from_ema200_atr" in result.insufficient_warmup_for_v1
    assert "D1_atr_percentile_252" in result.insufficient_warmup_for_v1


def test_compute_htf_features_uses_last_index_as_default_current_ts() -> None:
    m5 = _make_synthetic_m5(n_days=10)
    result_default = htf.compute_htf_features(m5)
    result_explicit = htf.compute_htf_features(m5, current_ts=m5.index[-1])
    assert result_default.h1_range_compression_ratio == result_explicit.h1_range_compression_ratio


def test_compute_htf_features_localizes_naive_timestamp_to_utc() -> None:
    m5 = _make_synthetic_m5(n_days=10)
    naive_ts = m5.index[-1].tz_localize(None)
    result = htf.compute_htf_features(m5, current_ts=naive_ts)
    assert result.h1_range_compression_ratio is not None


# ---------------------------------------------------------------------------
# Bit-for-bit consistency with offline builder
# ---------------------------------------------------------------------------


def test_h1_range_compression_matches_offline_helper() -> None:
    """Verify our H1 compression matches the offline _atr/ATR ratio
    computed via the same helpers in add_ctx_cont_columns_to_prebuilt."""
    m5 = _make_synthetic_m5(n_days=15)
    df_h1 = offline._resample_ohlc(m5, "1h")
    h1_atr14 = offline._atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).ffill()
    h1_atr100 = offline._atr(df_h1["high"], df_h1["low"], df_h1["close"], 100).ffill()
    expected_series = h1_atr14 / np.maximum(h1_atr100, offline.ATR_EPS)
    target_ts = m5.index[-1]
    expected = expected_series[expected_series.index <= target_ts - pd.Timedelta(hours=1)].dropna()
    expected_value = float(expected.iloc[-1])

    actual = htf.compute_h1_range_compression_ratio(m5, target_ts)
    assert actual is not None
    assert abs(actual - expected_value) < 1e-9


def test_d1_dist_matches_offline_helper() -> None:
    m5 = _make_synthetic_m5(n_days=240)
    df_d1 = offline._resample_ohlc(m5, "1D")
    d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
    d1_ema200 = offline._ema(d1_mid, 200)
    d1_atr14 = offline._atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()
    expected_series = (d1_mid - d1_ema200) / np.maximum(d1_atr14, offline.ATR_EPS)
    target_ts = m5.index[-1]
    expected = expected_series[expected_series.index <= target_ts - pd.Timedelta(days=1)].dropna()
    expected_value = float(expected.iloc[-1])

    actual = htf.compute_d1_dist_from_ema200_atr(m5, target_ts)
    assert actual is not None
    assert abs(actual - expected_value) < 1e-9


def test_h4_trend_sign_cat_in_valid_range_for_synthetic_data() -> None:
    m5 = _make_synthetic_m5(n_days=30, seed=42)
    result = htf.compute_h4_trend_sign_cat(m5, m5.index[-1])
    assert result is not None
    assert result in {0, 1, 2}


def test_compute_htf_features_does_not_modify_input() -> None:
    m5 = _make_synthetic_m5(n_days=15)
    snapshot = m5.copy()
    _ = htf.compute_htf_features(m5)
    pd.testing.assert_frame_equal(m5, snapshot)
