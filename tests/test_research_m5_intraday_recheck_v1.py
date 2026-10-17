"""M5 intraday recheck instrument: execution, costs, sequencing and fold hygiene on synthetic tapes (code checks only)."""

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import research_m5_intraday_recheck_v1 as m5

NS = 10**9


def _tape(start: str, minutes: int, *, gap_after: int | None = None, gap_hours: float = 0.0, bid: float = 100.0, spread: float = 0.2):
    t = pd.Timestamp(start, tz="UTC").value + np.arange(minutes, dtype=np.int64) * 60 * NS
    if gap_after is not None:
        t[gap_after:] += int(gap_hours * 3600 * NS)
    b = bid + np.arange(minutes) * 0.01
    return m5.Tape(t, b.copy(), b + spread, b.copy(), b + spread)


def test_entry_after_latency_exit_at_horizon_and_spread_counted_once():
    tape = _tape("2024-03-04 10:00", 400)
    decision = np.array([pd.Timestamp("2024-03-04 10:05", tz="UTC").value])
    ex = m5.execution_for_horizon(decision, tape, horizon_minutes=30, latency_seconds=60)
    assert ex.valid[0]
    assert ex.entry_ns[0] == decision[0] + 60 * NS
    assert ex.exit_ns[0] == ex.entry_ns[0] + 30 * 60 * NS
    ei, xi = 6, 36  # minutes after 10:00
    # long: buy the ask at entry, sell the bid at exit (spread inside the prices, no extra deduction)
    assert ex.long_gross_bps[0] == pytest.approx(1e4 * (tape.bid_open[xi] / tape.ask_open[ei] - 1))
    assert ex.short_gross_bps[0] == pytest.approx(1e4 * (tape.bid_open[ei] / tape.ask_open[xi] - 1))
    mid = (tape.bid_open + tape.ask_open) / 2
    assert ex.y_mid_bps[0] == pytest.approx(1e4 * np.log(mid[xi] / mid[ei]))


def test_weekly_gap_and_late_exit_are_censored():
    tape = _tape("2024-03-08 20:00", 200, gap_after=50, gap_hours=48)
    decision = np.array([pd.Timestamp("2024-03-08 20:30", tz="UTC").value])
    ex = m5.execution_for_horizon(decision, tape, horizon_minutes=120, latency_seconds=60)
    assert not ex.valid[0] and np.isnan(ex.long_gross_bps[0])
    late = _tape("2024-03-05 10:00", 200, gap_after=30, gap_hours=2)  # target 10:36 falls in a 2 h hole > 70 min tolerance
    d2 = np.array([pd.Timestamp("2024-03-05 10:05", tz="UTC").value])
    assert not m5.execution_for_horizon(d2, late, horizon_minutes=30, latency_seconds=60).valid[0]


def test_net_trade_charges_slippage_on_both_executions_and_financing_per_rollover():
    fin = {"long_bps_per_rollover": -1.5589, "short_bps_per_rollover": 0.8849}
    assert m5.net_trade_bps(1, 10.0, -12.0, 0, slippage_bps=2.0, financing=fin) == pytest.approx(6.0)
    assert m5.net_trade_bps(-1, 10.0, -12.0, 1, slippage_bps=2.0, financing=fin) == pytest.approx(-12.0 - 4.0 + 0.8849)
    assert m5.net_trade_bps(1, 10.0, -12.0, 1, slippage_bps=3.0, financing=fin) == pytest.approx(10.0 - 6.0 - 1.5589)


def test_rollover_count_at_17_new_york():
    e = pd.Timestamp("2024-03-05 16:50", tz="America/New_York").tz_convert("UTC").value
    x = pd.Timestamp("2024-03-05 17:10", tz="America/New_York").tz_convert("UTC").value
    x2 = pd.Timestamp("2024-03-05 16:59", tz="America/New_York").tz_convert("UTC").value
    assert m5.ny_rollovers_crossed(np.array([e, e]), np.array([x, x2])).tolist() == [1, 0]


def test_one_open_position_skips_decisions_until_exit():
    tape = _tape("2024-03-04 10:00", 600)
    decision = pd.Timestamp("2024-03-04 10:05", tz="UTC").value + np.arange(0, 60, 5, dtype=np.int64) * 60 * NS
    ex = m5.execution_for_horizon(decision, tape, horizon_minutes=30, latency_seconds=60)
    spec = {"execution": {"financing": {"long_bps_per_rollover": 0.0, "short_bps_per_rollover": 0.0},
                          "slippage_bps_per_execution": {"sensitivity": [2.0]}}}
    trades = m5.simulate(np.arange(len(decision)), np.ones(len(decision), int), ex, decision, spec)
    starts = [t["row"] for t in trades]
    assert starts == [0, 7]  # the first exits at 10:36; the next considered decision is 10:40 (row 7)
    for a, b in zip(trades, trades[1:]):
        assert decision[b["row"]] >= a["exit_ns"]


def test_fold_purge_embargo_and_inner_split_never_reach_the_evaluation_window():
    start = pd.Timestamp("2020-01-01", tz="UTC").value
    decision = start + np.arange(0, 5 * 365 * 288, 12, dtype=np.int64) * 5 * 60 * NS
    exit_ns = decision + 360 * 60 * NS
    valid = np.ones(len(decision), bool)
    fold = m5.build_fold("2023", "2023-01-01", "2024-01-01", decision, exit_ns, valid, train_years=3)
    s = pd.Timestamp("2023-01-01", tz="UTC").value
    assert exit_ns[fold.train].max() < s and decision[fold.train].max() < s - 86400 * NS
    assert decision[fold.train].min() >= pd.Timestamp("2020-01-01", tz="UTC").value
    assert exit_ns[fold.inner_train].max() < decision[fold.inner_val].min()
    assert np.isin(fold.inner_val, fold.train).all() and decision[fold.eval].min() >= s


def test_standardizer_uses_only_the_given_rows_and_spread_uses_the_closed_bar():
    x = np.array([[1.0], [3.0], [100.0]], dtype=np.float32)
    mean, std = m5.standardizer(x, np.array([0, 1]))
    assert mean[0] == pytest.approx(2.0) and std[0] == pytest.approx(1.0)
    tape = _tape("2024-03-04 10:00", 30, spread=0.5)
    d = np.array([pd.Timestamp("2024-03-04 10:10", tz="UTC").value])
    sp = m5.spread_at_decision(d, tape)
    i = 9  # bar 10:09 closes at 10:10
    assert sp[0] == pytest.approx(1e4 * 0.5 / ((tape.bid_close[i] + tape.ask_close[i]) / 2))


def test_period_names():
    assert m5.period_name("2016-01-01", "2017-01-01") == "2016"
    assert m5.period_name("2025-01-01", "2026-07-01") == "2025-01_2026-07"


def test_exact_intraday_research_mask_rejects_closure_and_delayed_quote():
    from gx1.scripts.research_m5_intraday_recheck_v1 import Tape,execution_for_horizon_v2,exact_intraday_mask_v3,MINUTE_NS
    t=np.arange(10,dtype=np.int64)*MINUTE_NS;bid=np.ones(10)*100.;ask=bid+.02
    tape=Tape(t,bid,ask,bid,ask);dec=np.array([0],np.int64)
    ex=execution_for_horizon_v2(dec,tape,horizon_minutes=5,latency_seconds=60)
    assert exact_intraday_mask_v3(dec,tape,ex,horizon_minutes=5)['valid'][0]
    keep=np.arange(10)!=3;broken=Tape(t[keep],bid[keep],ask[keep],bid[keep],ask[keep])
    ex2=execution_for_horizon_v2(dec,broken,horizon_minutes=5,latency_seconds=60)
    assert ex2.valid[0]
    assert not exact_intraday_mask_v3(dec,broken,ex2,horizon_minutes=5)['valid'][0]
    assert exact_intraday_mask_v3(dec,broken,ex2,horizon_minutes=5)['interior_gap_intervals'][0]==1
