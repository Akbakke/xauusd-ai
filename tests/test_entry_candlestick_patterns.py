import numpy as np
import pandas as pd
import pytest

from gx1.features.entry_candlestick_patterns_v1 import (
    CANDLESTICK_PATTERN_FEATURE_NAMES,
    CANDLESTICK_PATTERN_FEATURE_PREFIX,
    build_entry_candlestick_pattern_layer,
    missing_candlestick_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature


def test_candlestick_pattern_layer_builds_closed_bar_numeric_patterns() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=9, freq="5min", tz="UTC"),
            "open": [10.0, 10.4, 10.7, 10.2, 10.2, 9.7, 10.4, 10.8, 11.2],
            "high": [10.5, 10.8, 10.9, 10.3, 10.4, 10.8, 10.9, 11.3, 11.7],
            "low": [9.9, 10.3, 10.0, 9.6, 9.5, 9.6, 10.2, 10.7, 11.1],
            "close": [10.4, 10.7, 10.1, 10.0, 9.9, 10.7, 10.85, 11.25, 11.6],
        }
    )

    out, names = build_entry_candlestick_pattern_layer(frame)
    idx = {name: i for i, name in enumerate(names)}

    assert tuple(names) == CANDLESTICK_PATTERN_FEATURE_NAMES
    assert len(names) == 53
    assert len(set(names)) == len(names)
    assert all(name.startswith(CANDLESTICK_PATTERN_FEATURE_PREFIX) for name in names)
    assert out.shape == (9, 53)
    assert np.isfinite(out).all()
    assert out[0, idx["candle.pattern_body_direction"]] == 1.0
    assert out[4, idx["candle.pattern_hammer_bull_reversal_score"]] > 0.0
    assert out[5, idx["candle.pattern_bullish_engulfing_score"]] > 0.0
    assert out[5, idx["candle.pattern_bullish_engulfing_quality"]] > 0.0
    assert out[5, idx["candle.pattern_tweezer_bottom_quality"]] > 0.0
    assert out[6, idx["candle.pattern_bull_rejection_continuation_score"]] > 0.0
    assert out[7, idx["candle.pattern_three_bar_bull_continuation_quality"]] > 0.0
    # V30 package 7 (2026-08-13): the six aggregate votes and the affine
    # duplicate close_pressure_signed are no longer emitted.
    for removed in (
        "candle.pattern_bull_reversal_pressure",
        "candle.pattern_bear_reversal_pressure",
        "candle.pattern_bull_continuation_pressure",
        "candle.pattern_bear_continuation_pressure",
        "candle.pattern_indecision_breakout_setup",
        "candle.pattern_tail_rejection_risk",
        "candle.pattern_close_pressure_signed",
    ):
        assert removed not in idx


def test_candlestick_pattern_layer_triple_quality_ends_on_current_closed_bar() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC"),
            "open": [10.0, 9.05, 9.1, 9.8, 10.0, 10.95, 10.9, 10.2],
            "high": [10.1, 9.15, 9.9, 10.0, 11.1, 11.1, 11.0, 10.3],
            "low": [8.9, 8.85, 9.0, 9.7, 9.9, 10.85, 10.1, 10.0],
            "close": [9.0, 9.0, 9.8, 9.95, 11.0, 10.95, 10.2, 10.05],
        }
    )

    out, names = build_entry_candlestick_pattern_layer(frame)
    idx = {name: i for i, name in enumerate(names)}

    morning = idx["candle.pattern_morning_star_bull_quality"]
    evening = idx["candle.pattern_evening_star_bear_quality"]
    assert out[2, morning] > 0.0
    assert out[3, morning] == 0.0
    assert out[6, evening] > 0.0
    assert out[7, evening] == 0.0


def test_candlestick_pattern_layer_uses_current_closed_bar_without_past_leakage() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=7, freq="5min", tz="UTC"),
            "open": [10.0, 10.4, 10.7, 10.2, 10.2, 9.7, 10.4],
            "high": [10.5, 10.8, 10.9, 10.3, 10.4, 10.8, 10.9],
            "low": [9.9, 10.3, 10.0, 9.6, 9.5, 9.6, 10.2],
            "close": [10.4, 10.7, 10.1, 10.0, 9.9, 10.7, 10.85],
        }
    )
    base, names = build_entry_candlestick_pattern_layer(frame)
    mutated = frame.copy()
    mutated.loc[5, ["open", "high", "low", "close"]] = [100.0, 140.0, 80.0, 135.0]
    changed, _ = build_entry_candlestick_pattern_layer(mutated)

    np.testing.assert_allclose(base[:5], changed[:5])
    assert not np.allclose(base[5], changed[5])
    assert tuple(names) == CANDLESTICK_PATTERN_FEATURE_NAMES


@pytest.mark.parametrize("frequency", ["5min", "15min", "1h", "4h", "1D"])
def test_candlestick_current_closed_bar_impulse_is_identical_across_timeframes(
    frequency: str,
) -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range(
                "2026-01-01",
                periods=12,
                freq=frequency,
                tz="UTC",
            ),
            "open": np.linspace(10.0, 11.1, 12),
            "high": np.linspace(10.5, 11.6, 12),
            "low": np.linspace(9.8, 10.9, 12),
            "close": np.linspace(10.3, 11.4, 12),
        }
    )
    base, names = build_entry_candlestick_pattern_layer(frame)
    mutated = frame.copy()
    mutated.loc[11, ["open", "high", "low", "close"]] = [
        11.4,
        11.8,
        10.8,
        10.9,
    ]
    changed, changed_names = build_entry_candlestick_pattern_layer(mutated)

    np.testing.assert_allclose(base[:-1], changed[:-1])
    assert not np.allclose(base[-1], changed[-1])
    assert changed[-1, names.index("candle.pattern_body_direction")] == -1.0
    assert names == changed_names


def test_candlestick_lag_preserves_float64_price_precision() -> None:
    # prev_close - open = 6e-5 USD is below float32 resolution at 2000
    # (ulp ~ 1.22e-4): a float32 lag collapses the strict open < prev_close
    # gate into a tie and zeroes the pattern. The lag must preserve float64.
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
            "open": [2001.0, 2000.0],
            "high": [2001.2, 2001.0],
            "low": [1999.9, 1999.95],
            "close": [2000.00006, 2000.9],
        }
    )
    assert np.float32(2000.00006) == np.float32(2000.0)  # documents the collapse
    out, names = build_entry_candlestick_pattern_layer(frame)
    idx = {name: i for i, name in enumerate(names)}
    assert out[1, idx["candle.pattern_piercing_line_bull_score"]] > 0.0


def test_candlestick_zero_range_bar_reports_neutral_close_location() -> None:
    # high == low: close location is undefined, so it must read neutral 0.5
    # (basic_v1 _v1_clv convention). Body/wick shares stay 0: a zero-range bar
    # genuinely has zero body and zero wicks.  The old
    # `close_pressure_signed == 0.0` leg of this regression is gone with the
    # column (V30 package 7); `close_location == 0.5` is the same proof, since
    # the removed field was exactly 2*close_location - 1.
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
            "open": [2000.0, 2001.0, 2000.5, 2000.5],
            "high": [2002.0, 2001.5, 2000.5, 2001.5],
            "low": [1999.0, 2000.0, 2000.5, 2000.0],
            "close": [2001.0, 2000.5, 2000.5, 2001.0],
        }
    )
    out, names = build_entry_candlestick_pattern_layer(frame)
    idx = {name: i for i, name in enumerate(names)}
    row = 2  # open == high == low == close
    assert out[row, idx["candle.pattern_close_location"]] == 0.5
    assert out[row, idx["candle.pattern_body_share"]] == 0.0
    assert out[row, idx["candle.pattern_upper_wick_share"]] == 0.0
    assert out[row, idx["candle.pattern_lower_wick_share"]] == 0.0


def test_candlestick_star_first_candle_body_uses_its_own_range() -> None:
    # First-candle body condition: lag2_body / lag2_range = 10/11 > 0.35
    # fires the pattern; dividing by the CURRENT bar's range (10/40 < 0.35)
    # would wrongly veto it whenever the third bar expands.
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [2010.0, 2000.2, 2000.5],
            "high": [2010.5, 2001.0, 2040.0],
            "low": [1999.5, 1999.8, 2000.0],
            "close": [2000.0, 2000.4, 2035.0],
        }
    )
    out, names = build_entry_candlestick_pattern_layer(frame)
    idx = {name: i for i, name in enumerate(names)}
    assert out[2, idx["candle.pattern_morning_star_bull_score"]] > 0.0
    assert out[2, idx["candle.pattern_evening_star_bear_score"]] == 0.0


def test_candlestick_pattern_layer_rejects_nan_and_inf() -> None:
    for field, value in (("open", np.nan), ("high", np.inf), ("low", -np.inf)):
        frame = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
                "open": [10.0, 10.1, 10.2, 10.3, 10.4],
                "high": [10.5, 10.6, 10.7, 10.8, 10.9],
                "low": [9.9, 10.0, 10.1, 10.2, 10.3],
                "close": [10.4, 10.5, 10.6, 10.7, 10.8],
            }
        )
        frame.loc[2, field] = value
        with pytest.raises(RuntimeError, match="candlestick source field is non-finite"):
            build_entry_candlestick_pattern_layer(frame)


def test_candlestick_pattern_layer_rejects_invalid_time_and_ohlc_geometry() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
            "open": [10.0, 10.1, 10.2],
            "high": [10.5, 10.6, 10.7],
            "low": [9.9, 10.0, 10.1],
            "close": [10.4, 10.5, 10.6],
        }
    )
    invalid_geometry = frame.copy()
    invalid_geometry.loc[1, "high"] = 9.5
    with pytest.raises(RuntimeError, match="candlestick OHLC geometry is invalid"):
        build_entry_candlestick_pattern_layer(invalid_geometry)

    naive_time = frame.copy()
    naive_time["time"] = naive_time["time"].dt.tz_localize(None)
    with pytest.raises(RuntimeError, match="timezone-aware UTC"):
        build_entry_candlestick_pattern_layer(naive_time)

    duplicate_time = frame.copy()
    duplicate_time.loc[2, "time"] = duplicate_time.loc[1, "time"]
    with pytest.raises(RuntimeError, match="strictly increasing and unique"):
        build_entry_candlestick_pattern_layer(duplicate_time)


def test_candlestick_pattern_layer_rejects_missing_canonical_field() -> None:
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
            "open": [10.0, 10.1],
            "high": [10.5, 10.6],
            "close": [10.4, 10.5],
        }
    )
    with pytest.raises(RuntimeError, match="CANDLESTICK_PATTERN_SOURCE_FIELDS_MISSING"):
        build_entry_candlestick_pattern_layer(frame)


def test_candlestick_source_contract_and_specialist_routing() -> None:
    assert missing_candlestick_source_fields(["time", "open", "high", "low", "close"]) == []
    assert missing_candlestick_source_fields(["time", "open", "high", "close"]) == ["low"]
    assert (
        classify_entry_specialist_feature("candle.pattern_bullish_engulfing_score")
        == "price_action_candle_encoder"
    )
    assert (
        classify_entry_specialist_feature("candle.pattern_bear_rejection_followthrough_score")
        == "price_action_candle_encoder"
    )
    assert (
        classify_entry_specialist_feature("candle.pattern_range_compression_vs_prev5")
        == "price_action_candle_encoder"
    )
