"""Mechanics tests for the tape-based pattern primitives (synthetic bars, rule 2c: no market claim)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import research_entry_pattern_primitives_v1 as pp


def _params(**overrides) -> pp.Params:
    base = dict(
        swing_lookback=3, zone_lookback_bars=50, fvg_min_gap_atr=0.0, ob_displacement_atr=1.5, ob_displacement_bars=3,
        ob_search_bars=5, eq_tolerance_atr=0.2, flag_impulse_bars=5, flag_impulse_atr=2.0, flag_consolidation_min_bars=3,
        flag_consolidation_max_bars=12, flag_consolidation_atr=1.0, range_breakout_bars=10, age_cap_bars=99,
    )
    base.update(overrides)
    return pp.Params(**base)


def test_fair_value_gap_detection_is_exact_three_candle_imbalance() -> None:
    high = np.array([10, 11, 10, 15, 16, 14, 13, 12], dtype=float)
    low = np.array([9, 10, 9, 13, 15, 12, 11, 10], dtype=float)
    atr = np.ones(8)
    bull, bb, bt, bear, sb, st = pp.fair_value_gaps(high, low, atr, min_gap_atr=0.0)
    # bar 4: low 15 > high[2] 10 -> bullish gap (10, 15); bar 3: low 13 > high[1] 11 -> bullish gap (11, 13)
    assert bull[3] and bb[3] == 11 and bt[3] == 13
    assert bull[4] and bb[4] == 10 and bt[4] == 15
    # bar 6: high 13 < low[4] 15 -> bearish gap (13, 15)
    assert bear[6] and sb[6] == 13 and st[6] == 15 and bear.sum() == 1
    # min gap filter removes the smaller gap
    bull2, *_ = pp.fair_value_gaps(high, low, atr, min_gap_atr=3.0)
    assert not bull2[3] and bull2[4]


def test_zone_tracker_one_shot_retest_and_failure() -> None:
    n = 8
    event = np.zeros(n, dtype=bool)
    event[1] = True
    bottom = np.full(n, np.nan)
    top = np.full(n, np.nan)
    bottom[1], top[1] = 100.0, 102.0
    close = np.array([110, 110, 110, 110, 101, 110, 110, 110], dtype=float)
    low = np.array([109, 109, 109, 109, 101.5, 109, 109, 109], dtype=float)
    high = close + 1
    atr = np.ones(n)
    out = pp.track_zones(bottom_new=bottom, top_new=top, event=event, low=low, high=high, close=close, atr=atr, side="bull", lookback=50, age_cap=99)
    assert out["active_count"][1] == 1 and out["active_count"][3] == 1
    assert out["retest_event"][4] == 1.0 and out["active_count"][4] == 0  # touched at bar 4 (low <= top), held (close >= bottom), removed
    assert out["nearest_dist_atr"][3] == pytest.approx(110 - 102)
    assert out["nearest_dist_atr"][5] == pp.DISTANCE_CAP_ATR
    # failure: close below the bottom at the touch
    close2 = close.copy()
    close2[4] = 99.0
    low2 = low.copy()
    low2[4] = 98.0
    out2 = pp.track_zones(bottom_new=bottom, top_new=top, event=event, low=low2, high=high, close=close2, atr=atr, side="bull", lookback=50, age_cap=99)
    assert out2["failure_event"][4] == 1.0 and out2["retest_event"][4] == 0.0


def test_order_block_is_last_opposite_candle_before_displacement() -> None:
    open_ = np.array([10, 11, 10.5, 10.2, 10.4, 11, 12, 13], dtype=float)
    close = np.array([11, 10.5, 10.0, 10.4, 11, 12, 13, 14], dtype=float)
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    atr = np.ones(8)
    bull, bb, bt, bear, *_ = pp.order_blocks(open_, high, low, close, atr, displacement_atr=2.0, displacement_bars=3, search_bars=5)
    # bar 5: close 12 - close[2] 10.0 = 2.0 >= 2 ATR -> first displacement bar; last bearish candle at/before bar 2 is bar 2 itself (open 10.5 > close 10.0)
    assert bull[5] and bb[5] == pytest.approx(low[2]) and bt[5] == pytest.approx(high[2])
    assert not bull[6] and not bull[7], "edge-triggered: the continuing displacement does not re-fire"
    assert not bear.any()


def test_equal_pools_form_and_sweep() -> None:
    n = 40
    high = np.full(n, 10.0)
    low = np.full(n, 9.0)
    close = np.full(n, 9.5)
    # two equal swing highs at bars 10 and 20 (pivot = max over +-3), then a sweep at bar 30
    high[10] = 12.0
    high[20] = 12.05
    high[30] = 12.5
    close[30] = 11.0
    atr = np.ones(n)
    out = pp.equal_pools(high, low, close, atr, swing_lookback=3, tolerance_atr=0.2, lookback=50, age_cap=99)
    assert out["eqh_form_event"][23] == 1.0  # second pivot confirmed at 20 + 3
    assert out["eqh_active_count"][25] == 1.0
    assert out["eqh_nearest_dist_atr"][25] == pytest.approx(12.05 - 9.5)
    assert out["eqh_sweep_event"][30] == 1.0 and out["eqh_active_count"][30] == 0.0


def test_flag_breakout_after_impulse_and_tight_consolidation() -> None:
    n = 30
    close = np.full(n, 100.0)
    close[5:11] = np.linspace(100, 106, 6)  # impulse over bars 5..10
    close[11:16] = 106.0 + np.array([0.1, -0.1, 0.2, 0.0, 0.1])  # consolidation
    close[16] = 107.5  # breakout
    high = close + 0.2
    low = close - 0.2
    atr = np.ones(n)
    out = pp.flags(high, low, close, atr, impulse_bars=5, impulse_atr=2.0, cons_min=3, cons_max=12, cons_atr=1.0, age_cap=99)
    assert out["bull_flag_active"][14] == 1.0
    assert out["bull_flag_breakout_event"][16] == 1.0
    assert out["bear_flag_breakout_event"].sum() == 0


def test_range_breakout_edge_triggered() -> None:
    close = np.array([1, 1, 1, 1, 1, 1, 2, 2.5, 2.6, 1, 1, 0.5], dtype=float)
    high = close + 0.1
    low = close - 0.1
    out = pp.range_breakouts(high, low, close, bars=5, age_cap=99)
    assert out["range_break_up_event"][6] == 1.0 and out["range_break_up_event"][7] == 0.0
    assert out["range_break_down_event"][11] == 1.0
    assert out["bars_since_range_break_up"][8] == 2.0


def test_session_levels_previous_day_and_asia_range() -> None:
    time = pd.date_range("2024-01-08T22:00:00Z", periods=2 * 288, freq="5min")  # two full trading days from the 22:00 UTC boundary
    n = len(time)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    high = close + 0.2
    low = close - 0.2
    open_ = close
    tape = pd.DataFrame({"time": time, "open": open_, "high": high, "low": low, "close": close})
    atr = np.ones(n)
    out = pp.session_anchored_levels(tape, atr, age_cap=99)
    day2 = np.arange(288, 2 * 288)
    assert (out["pdh_present"].to_numpy()[:288] == 0).all() and (out["pdh_present"].to_numpy()[day2] == 1).all()
    pdh = high[:288].max()
    expected = np.clip(close[day2] - pdh, -pp.DISTANCE_CAP_ATR, pp.DISTANCE_CAP_ATR)
    assert np.allclose(out["pdh_dist_atr"].to_numpy()[day2], expected)
    # Asia range of day 2 becomes current after 07:00 UTC (bar index 288 + 9h*12 = 396)
    asia_current = out["asia_range_current"].to_numpy()
    assert asia_current[288 + 100] == 0.0 and asia_current[288 + 110] == 1.0
    asia_hi = high[288 : 288 + 108].max()
    assert out["asia_hi_dist_atr"].to_numpy()[288 + 120] == pytest.approx(np.clip(close[288 + 120] - asia_hi, -20, 20))


def test_last_closed_sampling_uses_owner_cutoff_rule() -> None:
    labels = pd.DatetimeIndex(pd.date_range("2024-01-01T00:00:00Z", periods=10, freq="1h"))
    values = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=labels)
    decision = pd.DatetimeIndex(["2024-01-01T02:55:00Z", "2024-01-01T03:00:00Z", "2024-01-01T03:55:00Z"])
    sampled = pp.sample_last_closed(labels, values, decision, "H1")
    # 02:55 -> cutoff 02:00 -> bar 02:00 (index 2) is closed at 03:00 == 02:55+5min
    assert sampled["H1:x"].tolist() == [2.0, 2.0, 3.0]


def test_build_end_to_end_on_synthetic_tape() -> None:
    time = pd.date_range("2024-01-01T22:00:00Z", periods=5 * 288, freq="5min")
    n = len(time)
    rng = np.random.default_rng(1)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, n))
    tape = pd.DataFrame({"time": time, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": np.ones(n)})
    decision = pd.DatetimeIndex(time[300:1300])
    frame, stats = pp.build(tape, decision, _params())
    assert len(frame) == 1000 and "M5:fvg_bull_event" in frame.columns and "H4:ema_stack" in frame.columns and "M5:pdh_dist_atr" in frame.columns
    assert np.isfinite(frame.drop(columns=["time"]).to_numpy(np.float64)).all()
    assert set(stats) == set(pp.TIMEFRAMES)
