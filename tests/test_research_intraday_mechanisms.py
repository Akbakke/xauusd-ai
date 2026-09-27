"""Mechanics of the preregistered intraday-mechanism cells (synthetic tapes prove only that code runs)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import research_intraday_mechanisms_v1 as im
from gx1.scripts.research_entry_direction_walkforward_v1 import Tape
from gx1.scripts.research_entry_pattern_setup_edge_v1 import declared_setups


def _tape(start: str, end: str, mid_at=None) -> Tape:
    time = pd.date_range(start, end, freq="5min", tz="UTC", inclusive="left")
    mid = np.full(len(time), 100.0) if mid_at is None else np.array([mid_at(t) for t in time], dtype=float)
    return Tape(time=time, mid=mid, bid=mid - 0.05, ask=mid + 0.05, manifest_sha256="0" * 64, root="synthetic")


def test_registered_cell_count_and_bonferroni_threshold():
    a = len(im.ROUND_GRIDS_USD) * 2 * len(im.HOLD_BARS)
    b = len(im.SETUP_PAIRS) * len(im.HOLD_BARS)
    c, d, e = 3, 3 * len(im.LBMA_AUCTIONS), 2 + 2 * len(im.SESSIONS)
    assert (a, b, c, d, e) == (8, 36, 3, 6, 8)
    assert a + b + c + d + e == im.CELL_COUNT == 61
    assert im.GO_T == pytest.approx(3.1488, abs=1e-4)


def test_local_clocks_follow_daylight_saving():
    winter, summer = pd.Timestamp("2015-01-15"), pd.Timestamp("2015-07-15")
    assert im.local_instant(winter, "08:20", im.NEW_YORK) == pd.Timestamp("2015-01-15 13:20", tz="UTC")
    assert im.local_instant(summer, "08:20", im.NEW_YORK) == pd.Timestamp("2015-07-15 12:20", tz="UTC")
    assert im.local_instant(winter, "15:00", im.LONDON) == pd.Timestamp("2015-01-15 15:00", tz="UTC")
    assert im.local_instant(summer, "15:00", im.LONDON) == pd.Timestamp("2015-07-15 14:00", tz="UTC")
    assert im.local_instant(summer, "09:00", im.TOKYO) == pd.Timestamp("2015-07-15 00:00", tz="UTC")
    assert im.previous_weekday(pd.Timestamp("2015-07-13")) == pd.Timestamp("2015-07-10")


def test_round_number_cross_and_reject_sides():
    close = np.array([1495, 1502, 1498, 1497, 1503, 1504, 1505, 1500, 1499.0])
    high = np.array([1496, 1503, 1502, 1500.5, 1504, 1505, 1510, 1501, 1500.0])
    low = np.array([1494, 1494, 1497, 1496, 1496, 1499.5, 1499, 1499, 1498.0])
    cross, reject = im.round_number_sides(close, high, low, 10.0)
    assert cross.tolist() == [0, 1, -1, 0, 1, 0, 0, 0, -1]
    assert reject.tolist() == [0, 0, 0, -1, 0, 1, 0, 1, 0]
    # a close exactly on the level from below is an up-cross; leaving it downwards is a down-cross
    cross2, _ = im.round_number_sides(np.array([1499.0, 1500.0, 1499.0]), np.array([1499.5, 1500.0, 1500.0]),
                                      np.array([1498.5, 1499.0, 1499.0]), 10.0)
    assert cross2.tolist() == [0, 1, -1]


def test_event_trades_are_non_overlapping_and_gap_free():
    time = pd.DatetimeIndex(list(pd.date_range("2015-01-05 10:00", periods=30, freq="5min", tz="UTC"))
                            + list(pd.date_range("2015-01-05 13:00", periods=30, freq="5min", tz="UTC")))
    side = np.zeros(len(time), dtype=np.int64)
    side[[2, 3, 6, 27, 40]] = [1, -1, -1, 1, -1]
    entries, exits, sides = im.event_side_trades(time, side, 4, first=0)
    # 2 taken (exit 6), 3 overlaps, 6 taken at the previous exit, 27 crosses the gap, 40 taken
    assert entries.tolist() == [2, 6, 40] and exits.tolist() == [6, 10, 44] and sides.tolist() == [1, -1, -1]


def test_intraday_momentum_uses_the_comex_clock():
    # July: New York is UTC-4, so 08:20 ET = 12:20 UTC and 13:30 ET = 17:30 UTC
    def mid(t):
        return 100.0 + (1.0 if t >= pd.Timestamp("2015-07-14 12:30", tz="UTC") else 0.0)
    tape = _tape("2015-07-13 00:00", "2015-07-15 00:00", mid)
    cells, skipped = im.intraday_momentum_cells(tape, pd.Timestamp("2015-07-14 00:00", tz="UTC"))
    t = tape.time
    first = cells["im_first_to_last"]
    assert t[first[0][0]] == pd.Timestamp("2015-07-14 16:25", tz="UTC")  # bar closing 12:30 ET
    assert t[first[1][0]] == pd.Timestamp("2015-07-14 17:25", tz="UTC")  # bar closing 13:30 ET
    assert first[2].tolist() == [1]
    hold = cells["im_first_hold_to_close"]
    assert t[hold[0][0]] == pd.Timestamp("2015-07-14 13:15", tz="UTC")  # bar closing 09:20 ET
    assert cells["im_overnight_to_last"][2].tolist() == [1]
    assert skipped == 1  # 13 July (New York date of the UTC start) opens before the start


def test_lbma_windows_on_the_london_clock():
    fix = pd.Timestamp("2015-01-14 15:00", tz="UTC")  # PM auction in winter
    tape = _tape("2015-01-14 00:00", "2015-01-15 00:00", lambda t: 100.0 - (1.0 if t >= fix else 0.0))
    cells, skipped = im.lbma_cells(tape, pd.Timestamp("2015-01-14 00:00", tz="UTC"))
    t = tape.time
    pre = cells["lbma_pm_pre_long"]
    assert t[pre[0][0]] == fix - pd.Timedelta(minutes=65) and t[pre[1][0]] == fix - pd.Timedelta(minutes=5)
    post = cells["lbma_pm_post_mom_h12"]
    assert t[post[0][0]] == fix + pd.Timedelta(minutes=10) and t[post[1][0]] == fix + pd.Timedelta(minutes=70)
    assert post[2].tolist() == [-1]
    assert "lbma_am_post_mom_h12" not in cells and skipped == {"am_post": 1}


def test_local_orb_and_sessions():
    open_ = pd.Timestamp("2015-07-14 07:00", tz="UTC")  # 08:00 London in summer
    tape = _tape("2015-07-14 00:00", "2015-07-15 00:00",
                 lambda t: 100.0 + (2.0 if t >= open_ + pd.Timedelta(minutes=90) else 0.0))
    start = pd.Timestamp("2015-07-14 00:00", tz="UTC")
    entries, exits, sides = im.orb_local_trades(tape, start, im.LONDON, im.LONDON_OPEN, im.LONDON_ORB_EXIT)
    assert tape.time[entries[0]] == open_ + pd.Timedelta(minutes=90) and sides.tolist() == [1]
    assert tape.time[exits[0]] == pd.Timestamp("2015-07-14 15:55", tz="UTC")  # last bar before 17:00 London
    e, x, s = im.session_local_trades(tape, start, *im.SESSIONS["asia"], side=1)
    assert tape.time[e[0]] == pd.Timestamp("2015-07-14 00:00", tz="UTC") and tape.time[x[0]] == pd.Timestamp("2015-07-14 06:55", tz="UTC")


def test_day_clustered_t_matches_the_formula():
    values = np.array([1.0, 3.0, -2.0, 4.0])
    times = pd.DatetimeIndex(["2015-01-05 10:00", "2015-01-05 12:00", "2015-01-06 10:00", "2015-01-07 10:00"], tz="UTC")
    mean = values.mean()
    sums = np.array([(1 - mean) + (3 - mean), -2 - mean, 4 - mean])
    se = math.sqrt(3 / 2 * np.sum(sums ** 2)) / 4
    assert im.day_clustered_t(values, times) == pytest.approx(mean / se)


def test_verdict_requires_both_sides_bear_fold_and_years():
    good = {"t_decision": 3.2, "positive_full_year_share": 0.62, "bear_fold_mean_bps": 0.1, "long_mean_bps": 1.0, "short_mean_bps": 0.5}
    assert im.verdict(good, two_sided=True) == "GO"
    assert im.verdict({**good, "t_decision": 2.1}, two_sided=True) == "LOVENDE"
    assert im.verdict({**good, "short_mean_bps": -0.1}, two_sided=True) == "NO_GO"
    assert im.verdict({**good, "short_mean_bps": None}, two_sided=False) == "GO"
    assert im.verdict({**good, "bear_fold_mean_bps": -0.1}, two_sided=True) == "NO_GO"
    assert im.verdict({**good, "positive_full_year_share": 0.5}, two_sided=True) == "NO_GO"


def test_setup_pairs_cover_every_declared_setup_with_the_kept_columns():
    frame = pd.DataFrame({c: np.zeros(3) for c in im.SETUP_COLUMNS})
    frame["M5:pdh_break_event"] = [1.0, 0.0, 0.0]
    frame["M5:pdl_break_event"] = [0.0, 1.0, 0.0]
    frame["H4:ema_stack"] = [1.0, -1.0, 1.0]
    sides = im.setup_pair_sides(frame)
    assert len(sides) == len(im.SETUP_PAIRS) == 18
    assert sorted(n for pair in im.SETUP_PAIRS for n in pair) == sorted(s.name for s in declared_setups())
    assert sides["pdh_break_trend_H4"].tolist() == [1, -1, 0]


def test_cell_stats_reports_every_cost_scenario():
    tape = _tape("2015-01-05 10:00", "2015-01-05 12:00", lambda t: 100.0 + (t.minute % 10) / 10.0)
    trades = (np.array([0, 6]), np.array([3, 9]), np.array([1, -1]))
    policy = {"slippage_bps_per_execution": 2.0, "commission_bps_per_execution": 0.0, "long_annual_cost_rate": 0.0,
              "short_annual_cost_rate": 0.0, "seconds_per_year": 31557600.0}
    stats = im.cell_stats(tape, trades, policy, {"low": 1.0, "central": 2.0, "high": 4.0}, two_sided=True)
    by = stats["mean_bps_by_scenario"]
    assert by["low"] - by["central"] == pytest.approx(2.0) and by["central"] - by["high"] == pytest.approx(4.0)
    assert stats["mean_bps"] == pytest.approx(by["low"]) and stats["long_n"] == 1 and stats["short_n"] == 1
