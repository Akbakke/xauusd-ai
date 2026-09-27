#!/usr/bin/env python3
"""Intraday XAUUSD mechanisms after cost (research only, wave 1 of 2026-09-27).

Preregistered in docs/INTRADAY_MECHANISMS_PREREG_20260927.md; every cell, clock, window and
decision threshold below is that registration. Five families: round-number order clusters (A),
the 2026-09-23 confluence setups as mirrored pairs over a sample with a bear market (B),
intraday momentum on the COMEX clock (C), LBMA auction windows (D) and sessions / opening-range
breakouts on local clocks (E). London, New York and Tokyo anchors come from the IANA time-zone
database, so they follow daylight saving. Fills, costs and statistics reuse the model-free
baseline owners; the setups reuse the pattern-primitive and setup-edge owners with the declared
parameters of the 2026-09-23 run. Never an Entry input (GX1_RULES.md rule 1); no VAL or TEST row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.research_entry_direction_walkforward_v1 import Tape, load_cost_policy, load_slippage_scenarios, load_tape
from gx1.scripts.research_entry_pattern_primitives_v1 import Params, build as build_primitives, load_tape_ohlcv
from gx1.scripts.research_entry_pattern_setup_edge_v1 import declared_setups
from gx1.scripts.research_macro_event_baselines_v1 import _bar_at
from gx1.scripts.research_model_free_baselines_v1 import (
    BAR,
    BEAR_FOLD,
    LOVENDE_T,
    ORB_RANGE_BARS,
    YEAR_SHARE,
    contiguity_breaks,
    summarize,
    trade_outcomes,
)

HOLD_BARS = (12, 48)  # 1 h and 4 h: the macro-event registration's post-event holds
ROUND_GRIDS_USD = (10.0, 50.0)
NEW_YORK, LONDON, TOKYO = "America/New_York", "Europe/London", "Asia/Tokyo"
COMEX_OPEN, COMEX_FIRST_HOUR_END, COMEX_LAST_HOUR_START, COMEX_CLOSE = "08:20", "09:20", "12:30", "13:30"
NY_DAILY_CLOSE = "17:00"
LONDON_OPEN, LONDON_ORB_EXIT, TOKYO_OPEN = "08:00", "17:00", "09:00"
LBMA_AUCTIONS = {"am": "10:30", "pm": "15:00"}
LBMA_PRE_WINDOW = pd.Timedelta(minutes=60)
LBMA_SIGNAL_END = pd.Timedelta(minutes=15)  # the macro registration's T+10 bar, which closes at T+15
LBMA_POST_HOLD_BARS = 12
SESSIONS = {
    "asia": ((TOKYO, TOKYO_OPEN), (LONDON, LONDON_OPEN)),
    "london": ((LONDON, LONDON_OPEN), (NEW_YORK, COMEX_OPEN)),
    "new_york": ((NEW_YORK, COMEX_OPEN), (NEW_YORK, COMEX_CLOSE)),
}
# The declared parameters of the 2026-09-23 primitives run (GX1_RUNS/V12_EPOCH1_REVIEW_20260923/
# ENTRY_DIRECTION_WALKFORWARD_20260923/patterns_v1/manifest.json), reused unchanged.
PRIMITIVE_PARAMS = Params(
    swing_lookback=3, zone_lookback_bars=96, fvg_min_gap_atr=0.0, ob_displacement_atr=2.0, ob_displacement_bars=3,
    ob_search_bars=5, eq_tolerance_atr=0.25, flag_impulse_bars=5, flag_impulse_atr=2.0, flag_consolidation_min_bars=3,
    flag_consolidation_max_bars=12, flag_consolidation_atr=1.0, range_breakout_bars=24, age_cap_bars=999,
)
SETUP_PAIRS = tuple(
    [(f"fvg_bull_retest_{tf}_trend_{ctx}", f"fvg_bear_retest_{tf}_trend_{ctx}") for tf, ctx in (("M5", "H1"), ("H1", "H4"), ("H4", "D1"))]
    + [(f"ob_bull_retest_{tf}_trend_{ctx}", f"ob_bear_retest_{tf}_trend_{ctx}") for tf, ctx in (("M5", "H1"), ("H1", "H4"), ("H4", "D1"))]
    + [(f"eql_sweep_fade_{tf}", f"eqh_sweep_fade_{tf}") for tf in ("M5", "H1")]
    + [(f"eql_sweep_fade_{tf}_trend_H4_bull", f"eqh_sweep_fade_{tf}_trend_H4_bear") for tf in ("M5", "H1")]
    + [(f"bull_flag_breakout_{tf}", f"bear_flag_breakout_{tf}") for tf in ("M5", "H1")]
    + [(f"range_break_up_{tf}_trend_H4", f"range_break_down_{tf}_trend_H4") for tf in ("M5", "H1")]
    + [
        ("pdh_break_trend_H4", "pdl_break_trend_H4"),
        ("asia_hi_break_trend_H1", "asia_lo_break_trend_H1"),
        ("momentum_confluence_long", "momentum_confluence_short"),
        ("trend_all_bull_any_bar", "trend_all_bear_any_bar"),
    ]
)
SETUP_COLUMNS = frozenset(
    [f"{tf}:{e}" for tf in ("M5", "H1", "H4") for e in ("fvg_bull_retest_event", "fvg_bear_retest_event", "ob_bull_retest_event", "ob_bear_retest_event")]
    + [f"{tf}:{e}" for tf in ("M5", "H1") for e in ("eqh_sweep_event", "eql_sweep_event", "bull_flag_breakout_event", "bear_flag_breakout_event", "range_break_up_event", "range_break_down_event")]
    + [f"M5:{e}" for e in ("pdh_break_event", "pdl_break_event", "asia_hi_break_event", "asia_lo_break_event")]
    + [f"{tf}:ema_stack" for tf in ("H1", "H4", "D1")]
)
DECISION_SCENARIO = "low"
CELL_COUNT = 61  # A 8 + B 36 + C 3 + D 6 + E 8
GO_T = NormalDist().inv_cdf(1.0 - 0.05 / CELL_COUNT)  # one-sided Bonferroni 0.05 / 61 = 3.149

Trades = tuple[np.ndarray, np.ndarray, np.ndarray]


def _arrays(entries: list[int], exits: list[int], sides: list[int]) -> Trades:
    return np.asarray(entries, dtype=np.int64), np.asarray(exits, dtype=np.int64), np.asarray(sides, dtype=np.int64)


def local_instant(day: pd.Timestamp, hhmm: str, tz: str) -> pd.Timestamp:
    """UTC instant of wall-clock ``hhmm`` on calendar ``day`` in ``tz`` (daylight saving included)."""
    return pd.Timestamp(f"{day.date()} {hhmm}", tz=tz).tz_convert("UTC")


def local_weekdays(tz: str, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = start.tz_convert(tz).tz_localize(None).normalize()
    last = end.tz_convert(tz).tz_localize(None).normalize()
    return [d for d in pd.date_range(first, last, freq="D") if d.weekday() < 5]


def previous_weekday(day: pd.Timestamp) -> pd.Timestamp:
    step = 3 if day.weekday() == 0 else 1
    return day - pd.Timedelta(days=step)


def bar_ending_at(tape: Tape, instant: pd.Timestamp) -> int | None:
    """The M5 bar whose close is at ``instant`` (it starts five minutes earlier), if on the tape."""
    return _bar_at(tape, instant - BAR)


def event_side_trades(time: pd.DatetimeIndex, side: np.ndarray, hold: int, first: int) -> Trades:
    """Greedy chronological non-overlapping trades at bars with a nonzero side, gap-free t-1 .. t+hold."""
    breaks = contiguity_breaks(time)
    last = len(time) - 1
    entries, exits, sides = [], [], []
    next_free = max(first, 1)
    for t in np.flatnonzero(side != 0):
        t = int(t)
        if t < next_free:
            continue
        if t + hold > last:
            break
        if breaks[t - 1] != breaks[t + hold]:
            continue
        entries.append(t)
        exits.append(t + hold)
        sides.append(int(side[t]))
        next_free = t + hold
    return _arrays(entries, exits, sides)


# --- A: round-number order clusters -------------------------------------------------------------

def round_number_sides(close: np.ndarray, high: np.ndarray, low: np.ndarray, grid: float) -> tuple[np.ndarray, np.ndarray]:
    """Per M5 bar (mid): cross-follow side and touch-reject fade side (0 = no event).

    A price at or above a level counts as above it. Cross: a level in (previous close, close] (up,
    +1) or in (close, previous close] (down, -1). Reject: no cross, and a level in
    (max(previous close, close), high] (resistance, fade -1) or in (low, min(previous close, close)]
    (support, fade +1); a bar touching both is ambiguous (0).
    """
    prev = np.concatenate([[np.nan], close[:-1]])
    at_or_below = np.floor(close / grid) * grid
    cross_up = prev < at_or_below
    cross_down = prev >= at_or_below + grid
    cross = np.where(cross_up, 1, np.where(cross_down, -1, 0))
    top = np.fmax(prev, close)
    bottom = np.fmin(prev, close)
    resistance = (np.floor(top / grid) + 1.0) * grid <= high
    support = np.floor(bottom / grid) * grid > low
    valid = np.isfinite(prev)
    no_cross = cross == 0
    reject = np.where(valid & no_cross & resistance & ~support, -1, np.where(valid & no_cross & support & ~resistance, 1, 0))
    return cross.astype(np.int64), reject.astype(np.int64)


def round_number_cells(tape: Tape, high: np.ndarray, low: np.ndarray, first: int) -> dict[str, Trades]:
    cells: dict[str, Trades] = {}
    for grid in ROUND_GRIDS_USD:
        cross, reject = round_number_sides(tape.mid, high, low, grid)
        for hold in HOLD_BARS:
            cells[f"rn{int(grid)}_reject_fade_h{hold}"] = event_side_trades(tape.time, reject, hold, first)
            cells[f"rn{int(grid)}_cross_follow_h{hold}"] = event_side_trades(tape.time, cross, hold, first)
    return cells


# --- B: mirrored confluence setups ----------------------------------------------------------------

def setup_pair_sides(primitives: pd.DataFrame) -> dict[str, np.ndarray]:
    """+1 where the LONG member fires alone, -1 where the SHORT member fires alone, else 0."""
    rules = {s.name: s for s in declared_setups()}
    members = [name for pair in SETUP_PAIRS for name in pair]
    if sorted(members) != sorted(rules) or len(set(members)) != len(members):
        raise RuntimeError("INTRADAY_SETUP_PAIRS_DO_NOT_COVER_THE_DECLARED_SETUPS")
    out: dict[str, np.ndarray] = {}
    for long_name, short_name in SETUP_PAIRS:
        if rules[long_name].side != "LONG" or rules[short_name].side != "SHORT":
            raise RuntimeError(f"INTRADAY_SETUP_PAIR_SIDES_INVALID: {long_name}")
        long_mask = np.asarray(rules[long_name].rule(primitives), dtype=bool)
        short_mask = np.asarray(rules[short_name].rule(primitives), dtype=bool)
        out[long_name] = np.where(long_mask & ~short_mask, 1, np.where(short_mask & ~long_mask, -1, 0)).astype(np.int64)
    return out


def setup_cells(tape: Tape, ohlcv: pd.DataFrame, first: int) -> tuple[dict[str, Trades], dict[str, Any]]:
    decision_time = tape.time[first:]
    primitives, _ = build_primitives(ohlcv, decision_time, PRIMITIVE_PARAMS, keep_columns=SETUP_COLUMNS)
    cells: dict[str, Trades] = {}
    fired: dict[str, Any] = {}
    for long_name, row_side in setup_pair_sides(primitives).items():
        side = np.zeros(len(tape.time), dtype=np.int64)
        side[first:] = row_side
        fired[long_name] = {"long_rows": int((row_side > 0).sum()), "short_rows": int((row_side < 0).sum())}
        for hold in HOLD_BARS:
            cells[f"setup_{long_name}_pair_h{hold}"] = event_side_trades(tape.time, side, hold, first)
    return cells, fired


# --- C: intraday momentum on the COMEX clock -------------------------------------------------------

def intraday_momentum_cells(tape: Tape, start: pd.Timestamp) -> tuple[dict[str, Trades], int]:
    lists = {name: ([], [], []) for name in ("im_first_to_last", "im_overnight_to_last", "im_first_hold_to_close")}
    skipped = 0
    for day in local_weekdays(NEW_YORK, start, tape.time[-1]):
        open_ = bar_ending_at(tape, local_instant(day, COMEX_OPEN, NEW_YORK))
        first_end = bar_ending_at(tape, local_instant(day, COMEX_FIRST_HOUR_END, NEW_YORK))
        last_start = bar_ending_at(tape, local_instant(day, COMEX_LAST_HOUR_START, NEW_YORK))
        close = bar_ending_at(tape, local_instant(day, COMEX_CLOSE, NEW_YORK))
        prior = bar_ending_at(tape, local_instant(previous_weekday(day), NY_DAILY_CLOSE, NEW_YORK))
        if None in (open_, first_end, last_start, close, prior) or tape.time[open_] < start:
            skipped += 1
            continue
        first_sign = int(np.sign(tape.mid[first_end] - tape.mid[open_]))
        overnight_sign = int(np.sign(tape.mid[first_end] - tape.mid[prior]))
        for name, side, entry in (
            ("im_first_to_last", first_sign, last_start),
            ("im_overnight_to_last", overnight_sign, last_start),
            ("im_first_hold_to_close", first_sign, first_end),
        ):
            if side != 0:
                lists[name][0].append(entry)
                lists[name][1].append(close)
                lists[name][2].append(side)
    return {name: _arrays(*v) for name, v in lists.items()}, skipped


# --- D: LBMA auction windows ------------------------------------------------------------------------

def lbma_cells(tape: Tape, start: pd.Timestamp) -> tuple[dict[str, Trades], dict[str, int]]:
    lists: dict[str, tuple[list, list, list]] = {}
    skipped: dict[str, int] = {}

    def add(name: str, entry: int, exit_: int, side: int) -> None:
        cell = lists.setdefault(name, ([], [], []))
        cell[0].append(entry)
        cell[1].append(exit_)
        cell[2].append(side)

    for day in local_weekdays(LONDON, start, tape.time[-1]):
        for auction, hhmm in LBMA_AUCTIONS.items():
            fix = local_instant(day, hhmm, LONDON)
            if fix - LBMA_PRE_WINDOW < start:
                continue
            pre_entry = bar_ending_at(tape, fix - LBMA_PRE_WINDOW)
            at_fix = bar_ending_at(tape, fix)
            if pre_entry is None or at_fix is None:
                skipped[f"{auction}_pre"] = skipped.get(f"{auction}_pre", 0) + 1
            else:
                add(f"lbma_{auction}_pre_long", pre_entry, at_fix, 1)
                add(f"lbma_{auction}_pre_short", pre_entry, at_fix, -1)
            signal_end = bar_ending_at(tape, fix + LBMA_SIGNAL_END)
            exit_ = None if signal_end is None else _bar_at(tape, tape.time[signal_end] + LBMA_POST_HOLD_BARS * BAR)
            side = 0 if at_fix is None or signal_end is None else int(np.sign(tape.mid[signal_end] - tape.mid[at_fix]))
            if side == 0 or exit_ is None:
                skipped[f"{auction}_post"] = skipped.get(f"{auction}_post", 0) + 1
                continue
            add(f"lbma_{auction}_post_mom_h{LBMA_POST_HOLD_BARS}", signal_end, exit_, side)
    return {name: _arrays(*v) for name, v in lists.items()}, skipped


# --- E: local-clock opening-range breakouts and sessions ---------------------------------------------

def orb_local_trades(tape: Tape, start: pd.Timestamp, tz: str, open_hhmm: str, exit_hhmm: str) -> Trades:
    """One opening-range breakout per local weekday (the model-free ORB rule on a local clock)."""
    entries, exits, sides = [], [], []
    time = tape.time
    for day in local_weekdays(tz, start, time[-1]):
        open_ = local_instant(day, open_hhmm, tz)
        if open_ < start:
            continue
        a = int(time.searchsorted(open_, side="left"))
        b = int(time.searchsorted(open_ + ORB_RANGE_BARS * BAR, side="left"))
        e = int(time.searchsorted(local_instant(day, exit_hhmm, tz), side="left"))
        if b - a != ORB_RANGE_BARS or e - b < 2:
            continue
        high = tape.mid[a:b].max()
        low = tape.mid[a:b].min()
        for j in range(b, e - 1):
            side = 1 if tape.mid[j] > high else -1 if tape.mid[j] < low else 0
            if side:
                entries.append(j)
                exits.append(e - 1)
                sides.append(side)
                break
    return _arrays(entries, exits, sides)


def session_local_trades(tape: Tape, start: pd.Timestamp, begin: tuple[str, str], end: tuple[str, str], side: int) -> Trades:
    """Hold from the first bar at/after the local start to the last bar before the local end."""
    entries, exits = [], []
    time = tape.time
    for day in local_weekdays(begin[0], start, time[-1]):
        t0 = local_instant(day, begin[1], begin[0])
        t1 = local_instant(day, end[1], end[0])
        if t0 < start or t1 <= t0:
            continue
        first = int(time.searchsorted(t0, side="left"))
        last = int(time.searchsorted(t1, side="left")) - 1
        if first >= len(time) or last <= first or time[first] >= t1:
            continue
        entries.append(first)
        exits.append(last)
    return _arrays(entries, exits, [side] * len(entries))


def local_clock_cells(tape: Tape, start: pd.Timestamp) -> dict[str, Trades]:
    cells = {
        "orb_london_local": orb_local_trades(tape, start, LONDON, LONDON_OPEN, LONDON_ORB_EXIT),
        "orb_new_york_local": orb_local_trades(tape, start, NEW_YORK, COMEX_OPEN, COMEX_CLOSE),
    }
    for name, (begin, end) in SESSIONS.items():
        for side, label in ((1, "long"), (-1, "short")):
            cells[f"session_{name}_local_{label}"] = session_local_trades(tape, start, begin, end, side)
    return cells


# --- statistics ------------------------------------------------------------------------------------

def day_clustered_t(values: np.ndarray, times: pd.DatetimeIndex) -> float | None:
    """Mean / standard error clustered by UTC entry date (several trades a day share one cluster)."""
    n = len(values)
    if n < 2:
        return None
    resid = values - values.mean()
    sums = pd.Series(resid).groupby(np.asarray(times.floor("D").asi8)).sum().to_numpy()
    groups = len(sums)
    if groups < 2:
        return None
    se = math.sqrt(groups / (groups - 1) * float(np.sum(sums ** 2))) / n
    return float(values.mean() / se) if se > 0 else None


def verdict(stats: dict[str, Any], *, two_sided: bool) -> str:
    t = stats.get("t_decision")
    if t is None or stats.get("positive_full_year_share", 0.0) < YEAR_SHARE:
        return "NO_GO"
    bear = stats.get("bear_fold_mean_bps")
    if bear is None or bear < 0:
        return "NO_GO"
    if two_sided and not (
        stats.get("long_mean_bps") is not None and stats["long_mean_bps"] > 0
        and stats.get("short_mean_bps") is not None and stats["short_mean_bps"] > 0
    ):
        return "NO_GO"
    if t >= GO_T:
        return "GO"
    if t >= LOVENDE_T:
        return "LOVENDE"
    return "NO_GO"


def cell_stats(tape: Tape, trades: Trades, policy: dict[str, Any], scenarios: dict[str, float], *, two_sided: bool) -> dict[str, Any]:
    entries, exits, sides = trades
    if len(entries) < 2:
        return {"n": int(len(entries)), "verdict": "NO_GO"}
    times = tape.time[entries]
    nets: dict[str, np.ndarray] = {}
    gross = np.zeros(0)
    for name, bps in scenarios.items():
        scenario = dict(policy)
        scenario["slippage_bps_per_execution"] = bps
        gross, nets[name] = trade_outcomes(tape, entries, exits, sides, scenario, financing=True)
    decision = nets[DECISION_SCENARIO]
    stats = summarize(decision, times)
    stats["decision_scenario"] = DECISION_SCENARIO
    stats["t_day_cluster"] = day_clustered_t(decision, times)
    stats["t_decision"] = None if stats.get("t") is None or stats["t_day_cluster"] is None else min(stats["t"], stats["t_day_cluster"])
    stats["gross_mean_bps"] = float(gross.mean())
    stats["gross_hit_rate"] = float((gross > 0).mean())
    stats["mean_bps_by_scenario"] = {name: float(v.mean()) for name, v in nets.items()}
    bear = (times >= BEAR_FOLD[0]) & (times < BEAR_FOLD[1])
    stats["bear_fold_n"] = int(bear.sum())
    stats["bear_fold_mean_bps"] = float(decision[bear].mean()) if bear.any() else None
    for label, mask in (("long", sides > 0), ("short", sides < 0)):
        stats[f"{label}_n"] = int(mask.sum())
        stats[f"{label}_mean_bps"] = float(decision[mask].mean()) if mask.any() else None
    stats["verdict"] = verdict(stats, two_sided=two_sided)
    return stats


FIXED_SIDE_CELL_PREFIXES = ("lbma_am_pre_", "lbma_pm_pre_", "session_")


def run(tape: Tape, ohlcv: pd.DataFrame, policy: dict[str, Any], scenarios: dict[str, float], start: pd.Timestamp) -> dict[str, Any]:
    if not np.array_equal(np.asarray(pd.DatetimeIndex(ohlcv["time"]).asi8), np.asarray(tape.time.asi8)):
        raise RuntimeError("INTRADAY_TAPE_READERS_DISAGREE")
    first = int(tape.time.searchsorted(start, side="left"))
    high = ohlcv["high"].to_numpy(np.float64)
    low = ohlcv["low"].to_numpy(np.float64)
    setups, fired = setup_cells(tape, ohlcv, first)
    momentum, momentum_skipped = intraday_momentum_cells(tape, start)
    lbma, lbma_skipped = lbma_cells(tape, start)
    families = {
        "A_round_numbers": round_number_cells(tape, high, low, first),
        "B_setup_pairs": setups,
        "C_intraday_momentum": momentum,
        "D_lbma": lbma,
        "E_local_clock": local_clock_cells(tape, start),
    }
    count = sum(len(cells) for cells in families.values())
    if count != CELL_COUNT:
        raise RuntimeError(f"INTRADAY_CELL_COUNT_NOT_REGISTERED: {count}")
    scored = {
        family: {
            name: cell_stats(tape, trades, policy, scenarios, two_sided=not name.startswith(FIXED_SIDE_CELL_PREFIXES))
            for name, trades in cells.items()
        }
        for family, cells in families.items()
    }
    return {
        "families": scored,
        "setup_fire_rows": fired,
        "skipped": {"intraday_momentum_days": momentum_skipped, "lbma": lbma_skipped},
    }


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2]).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-m5-root", type=Path, required=True)
    parser.add_argument("--cost-policy", type=Path, required=True)
    parser.add_argument("--cost-policy-sha256", required=True)
    parser.add_argument("--eval-start", required=True, help="first entry time (UTC)")
    parser.add_argument("--read-end-exclusive", required=True, help="tape is truncated before this time (UTC)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("INTRADAY_OUTPUT_EXISTS")
    start = pd.Timestamp(args.eval_start)
    end = pd.Timestamp(args.read_end_exclusive)
    if start.tzinfo is None or end.tzinfo is None:
        raise RuntimeError("INTRADAY_TIMES_MUST_BE_UTC_AWARE")
    tape = load_tape(args.native_m5_root, truncate_before=end)
    ohlcv = load_tape_ohlcv(args.native_m5_root, truncate_before=end)
    policy = load_cost_policy(args.cost_policy, args.cost_policy_sha256)
    scenarios = load_slippage_scenarios(args.cost_policy, args.cost_policy_sha256)
    body = run(tape, ohlcv, policy, scenarios, start)
    verdicts = {f"{family}:{name}": cell["verdict"] for family, cells in body["families"].items() for name, cell in cells.items()}
    result = {
        "schema_version": "gx1_intraday_mechanisms_v1",
        "preregistration": "docs/INTRADAY_MECHANISMS_PREREG_20260927.md",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_clean": _git("status", "--porcelain") == "",
        "tape": {"root": str(args.native_m5_root), "manifest_sha256": tape.manifest_sha256,
                 "first_bar": str(tape.time[0]), "last_bar": str(tape.time[-1]), "bars": len(tape.time)},
        "cost_policy": {k: v for k, v in policy.items() if k != "decision"},
        "slippage_scenarios_bps_per_execution": scenarios,
        "decision_scenario": DECISION_SCENARIO,
        "cell_count": CELL_COUNT,
        "go_t": GO_T,
        "eval_start": str(start),
        "read_end_exclusive": str(end),
        **body,
        "decision": {
            "go": sorted(n for n, v in verdicts.items() if v == "GO"),
            "lovende": sorted(n for n, v in verdicts.items() if v == "LOVENDE"),
        },
    }
    args.out_dir.mkdir(parents=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=float)
    (args.out_dir / "results.json").write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "results_sha256": hashlib.sha256(payload.encode()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
