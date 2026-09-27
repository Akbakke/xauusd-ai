"""Mechanics of the preregistered model-free baselines (synthetic tapes prove only that code runs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.research_entry_direction_walkforward_v1 import Tape
from gx1.scripts import research_model_free_baselines_v1 as baselines

POLICY = {
    "slippage_bps_per_execution": 2.0,
    "commission_bps_per_execution": 0.0,
    "long_annual_cost_rate": 0.054,
    "short_annual_cost_rate": 0.0,
    "seconds_per_year": 31_557_600.0,
}


def _tape(times, mid, spread=0.2) -> Tape:
    mid = np.asarray(mid, dtype=np.float64)
    return Tape(
        time=pd.DatetimeIndex(times),
        mid=mid,
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        manifest_sha256="0" * 64,
        root="synthetic",
    )


def _grid(start: str, bars: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=bars, freq="5min", tz="UTC")


def test_trade_outcomes_fill_cost_and_financing_scenarios():
    tape = _tape(_grid("2012-01-02 10:00", 13), np.linspace(100.0, 101.2, 13))
    entry, exit_ = np.array([0, 0, 0]), np.array([12, 12, 12])
    gross, net_a = baselines.trade_outcomes(tape, entry, exit_, np.array([1, -1, 0]), POLICY, financing=True)
    _, net_b = baselines.trade_outcomes(tape, entry, exit_, np.array([1, -1, 0]), POLICY, financing=False)
    assert gross[0] == pytest.approx((tape.bid[12] / tape.ask[0] - 1) * 1e4)
    assert gross[1] == pytest.approx((1 - tape.ask[12] / tape.bid[0]) * 1e4)
    assert gross[2] == 0.0 and net_a[2] == 0.0
    financing = 0.054 * 3600 / 31_557_600.0 * 1e4
    assert net_b[0] == pytest.approx(gross[0] - 4.0)
    assert net_a[0] == pytest.approx(gross[0] - 4.0 - financing)
    assert net_a[1] == pytest.approx(net_b[1]) == pytest.approx(gross[1] - 4.0)


def test_momentum_trades_never_span_a_gap_and_never_overlap():
    times = _grid("2012-01-02 10:00", 20).append(_grid("2012-01-02 13:00", 20))
    mid = np.concatenate([np.linspace(100, 102, 20), np.linspace(102, 100, 20)])
    entries, exits, sides = baselines.momentum_trades(
        _tape(times, mid), lookback=2, hold=3, reverse=False, start=pd.Timestamp("2012-01-01", tz="UTC")
    )
    breaks = baselines.contiguity_breaks(pd.DatetimeIndex(times))
    assert len(entries) > 0
    assert np.all(breaks[entries - 2] == breaks[exits])
    assert np.all(entries[1:] >= exits[:-1])
    assert set(sides[entries < 20]) == {1} and set(sides[entries >= 22]) == {-1}
    _, _, reversed_sides = baselines.momentum_trades(
        _tape(times, mid), lookback=2, hold=3, reverse=True, start=pd.Timestamp("2012-01-01", tz="UTC")
    )
    assert np.array_equal(reversed_sides, -sides)


def test_d1_trading_day_starts_at_22_utc():
    times = pd.DatetimeIndex(["2012-01-02 21:50", "2012-01-02 21:55", "2012-01-02 22:00", "2012-01-03 21:55"], tz="UTC")
    assert baselines.d1_closes(_tape(times, [1, 2, 3, 4])).tolist() == [1, 3]


def test_orb_enters_on_first_close_outside_the_first_hour_and_exits_before_session_end():
    times = _grid("2012-01-02 07:00", 108)  # 07:00 .. 15:55
    mid = np.full(len(times), 100.0)
    mid[:12] = np.linspace(99.5, 100.5, 12)
    mid[14] = 100.8  # 08:10 closes above the range
    entries, exits, sides = baselines.orb_trades(
        _tape(times, mid), open_hour=7, exit_hour=16, start=pd.Timestamp("2012-01-01", tz="UTC")
    )
    assert entries.tolist() == [14] and sides.tolist() == [1]
    assert times[exits[0]] == pd.Timestamp("2012-01-02 15:55", tz="UTC")


def test_session_window_crossing_midnight():
    times = _grid("2012-01-02 22:00", 108)  # 22:00 .. 06:55 next day
    entries, exits, sides = baselines.session_trades(
        _tape(times, np.linspace(100, 101, 108)), start_hour=22, end_hour=7, side=-1,
        start=pd.Timestamp("2012-01-01", tz="UTC"),
    )
    assert entries.tolist() == [0] and exits.tolist() == [107] and sides.tolist() == [-1]


def test_swing_signals_tsmom_and_sma():
    close = np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 150, 60)])
    signals = baselines.swing_signals(close)
    assert np.isnan(signals["tsmom_252"][:252]).all()
    assert signals["tsmom_21"][250] == 1 and signals["tsmom_21"][-1] == -1
    assert signals["sma200"][299] == 1
    assert signals["combo"][299] == 1


@pytest.mark.parametrize(
    "stats,scenario_b,expected",
    [
        ({"t": 3.2, "positive_full_year_share": 0.7}, 1.0, "GO"),
        ({"t": 3.2, "positive_full_year_share": 0.7}, -1.0, "NO_GO"),
        ({"t": 3.2, "positive_full_year_share": 0.5}, None, "NO_GO"),
        ({"t": 2.4, "positive_full_year_share": 0.62}, None, "LOVENDE"),
        ({"t": 1.9, "positive_full_year_share": 0.9}, None, "NO_GO"),
        ({"n": 1}, None, "NO_GO"),
    ],
)
def test_verdict_rule_is_the_registration(stats, scenario_b, expected):
    assert baselines.verdict(stats, scenario_b_mean=scenario_b) == expected
