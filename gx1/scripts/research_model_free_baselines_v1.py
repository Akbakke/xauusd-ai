#!/usr/bin/env python3
"""Model-free XAUUSD baselines: scalp rules on M5, trend rules on D1.

Preregistered in docs/MODEL_FREE_BASELINES_PREREG_20260927.md; every cell, window and decision
threshold below is that registration. Research evidence only: one pre-TEST native M5 tape is read
up to the declared end, fills and costs go through the walk-forward instrument's owners
(``load_tape``, ``load_cost_policy``, ``apply_cost_policy``), and per-cell statistics are written.
No model, no feature surface, no VAL or TEST row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.research_entry_direction_walkforward_v1 import (
    BPS,
    Tape,
    apply_cost_policy,
    load_cost_policy,
    load_tape,
)

MOM_LOOKBACK_BARS = (1, 6, 12)
MOM_HOLD_BARS = (6, 12, 19)
ORB_SESSIONS = {"london": (7, 16), "new_york": (13, 20)}  # UTC open hour -> exit hour
ORB_RANGE_BARS = 12
SESSION_WINDOWS = {"asia": (22, 7), "london": (7, 13), "new_york": (13, 20)}  # UTC start -> end hour
TSMOM_LOOKBACK_D1 = (21, 63, 126, 252)
SMA_D1 = 200
SWING_HORIZONS_D1 = (5, 10, 20)
TRADING_DAY_START_HOUR_UTC = 22
GO_T = 3.15  # one-sided Bonferroni 0.05 / 62 cells
LOVENDE_T = 2.0
YEAR_SHARE = 0.60
FULL_YEARS = tuple(range(2012, 2025))
BEAR_FOLD = (pd.Timestamp("2011-09-01", tz="UTC"), pd.Timestamp("2016-01-01", tz="UTC"))
BAR = pd.Timedelta(minutes=5)


def trade_outcomes(
    tape: Tape,
    entry: np.ndarray,
    exit_: np.ndarray,
    side: np.ndarray,
    policy: dict[str, Any],
    *,
    financing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Gross and net bps per trade; side +1 long, -1 short, 0 flat (flat costs nothing)."""
    entry = np.asarray(entry, dtype=np.int64)
    exit_ = np.asarray(exit_, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    if len(entry) == 0:
        return np.zeros(0), np.zeros(0)
    if np.any(exit_ <= entry):
        raise RuntimeError("BASELINE_EXIT_NOT_AFTER_ENTRY")
    long_gross = (tape.bid[exit_] / tape.ask[entry] - 1.0) * BPS
    short_gross = (1.0 - tape.ask[exit_] / tape.bid[entry]) * BPS
    elapsed = np.asarray((tape.time[exit_] - tape.time[entry]).total_seconds(), dtype=np.float64)
    scenario = dict(policy)
    if not financing:
        scenario["long_annual_cost_rate"] = 0.0
        scenario["short_annual_cost_rate"] = 0.0
    long_net, short_net = apply_cost_policy(long_gross, short_gross, elapsed, scenario)
    gross = np.where(side > 0, long_gross, np.where(side < 0, short_gross, 0.0))
    net = np.where(side > 0, long_net, np.where(side < 0, short_net, 0.0))
    return gross, net


def contiguity_breaks(time: pd.DatetimeIndex) -> np.ndarray:
    """breaks[i] = number of non-5-minute steps among bars 0..i; [a, b] is gap-free iff equal."""
    steps = np.diff(time.asi8) != BAR.value
    return np.concatenate([[0], np.cumsum(steps)])


def momentum_trades(
    tape: Tape, *, lookback: int, hold: int, reverse: bool, start: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy chronological non-overlapping MOM/REV trades on gap-free windows."""
    mid = tape.mid
    breaks = contiguity_breaks(tape.time)
    first = int(tape.time.searchsorted(start, side="left"))
    entries, exits, sides = [], [], []
    t = max(first, lookback)
    last = len(mid) - 1
    while t + hold <= last:
        if breaks[t - lookback] != breaks[t + hold]:
            t += 1
            continue
        signal = np.sign(mid[t] / mid[t - lookback] - 1.0)
        if signal == 0:
            t += 1
            continue
        entries.append(t)
        exits.append(t + hold)
        sides.append(int(-signal if reverse else signal))
        t += hold
    return np.asarray(entries, dtype=np.int64), np.asarray(exits, dtype=np.int64), np.asarray(sides, dtype=np.int64)


def _day_groups(time: pd.DatetimeIndex) -> dict[pd.Timestamp, np.ndarray]:
    """Contiguous index ranges per UTC calendar day of a sorted tape."""
    days = time.floor("D")
    cuts = np.flatnonzero(days[1:] != days[:-1]) + 1
    bounds = np.concatenate([[0], cuts, [len(time)]])
    return {days[a]: np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])}


def orb_trades(tape: Tape, *, open_hour: int, exit_hour: int, start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One opening-range breakout per UTC day and session."""
    entries, exits, sides = [], [], []
    minutes = tape.time.hour * 60 + tape.time.minute
    for day, idx in _day_groups(tape.time).items():
        if day < start.floor("D"):
            continue
        m = minutes[idx]
        range_idx = idx[(m >= open_hour * 60) & (m < open_hour * 60 + ORB_RANGE_BARS * 5)]
        if len(range_idx) != ORB_RANGE_BARS or tape.time[range_idx[0]] < start:
            continue
        window = idx[(m >= open_hour * 60 + ORB_RANGE_BARS * 5) & (m < exit_hour * 60)]
        if len(window) < 2:
            continue
        high = tape.mid[range_idx].max()
        low = tape.mid[range_idx].min()
        exit_index = int(window[-1])
        for j in window[:-1]:
            side = 1 if tape.mid[j] > high else -1 if tape.mid[j] < low else 0
            if side:
                entries.append(int(j))
                exits.append(exit_index)
                sides.append(side)
                break
    return np.asarray(entries, dtype=np.int64), np.asarray(exits, dtype=np.int64), np.asarray(sides, dtype=np.int64)


def session_trades(
    tape: Tape, *, start_hour: int, end_hour: int, side: int, start: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold from the first bar at/after the window start to the last bar before its end."""
    entries, exits = [], []
    time = tape.time
    days = pd.date_range(start.floor("D"), time[-1].floor("D"), freq="D", tz="UTC")
    for day in days:
        begin = day + pd.Timedelta(hours=start_hour)
        end = day + pd.Timedelta(hours=end_hour)
        if end <= begin:  # window crosses midnight: starts the previous day
            begin -= pd.Timedelta(days=1)
        if begin < start:
            continue
        first = int(time.searchsorted(begin, side="left"))
        last = int(time.searchsorted(end, side="left")) - 1
        if first >= len(time) or last <= first or time[first] >= end:
            continue
        entries.append(first)
        exits.append(last)
    n = len(entries)
    return np.asarray(entries, dtype=np.int64), np.asarray(exits, dtype=np.int64), np.full(n, side, dtype=np.int64)


def d1_closes(tape: Tape) -> np.ndarray:
    """Index of the last M5 bar of each trading day starting 22:00 UTC."""
    label = (tape.time + pd.Timedelta(hours=24 - TRADING_DAY_START_HOUR_UTC)).floor("D")
    change = np.flatnonzero(label[1:] != label[:-1])
    return np.concatenate([change, [len(tape.time) - 1]]).astype(np.int64)


def swing_signals(close: np.ndarray) -> dict[str, np.ndarray]:
    """Signal per D1 close; NaN where the lookback is not yet available."""
    n = len(close)
    out: dict[str, np.ndarray] = {}
    signs = []
    for lookback in TSMOM_LOOKBACK_D1:
        s = np.full(n, np.nan)
        s[lookback:] = np.sign(close[lookback:] / close[:-lookback] - 1.0)
        out[f"tsmom_{lookback}"] = s
        signs.append(s)
    out["combo"] = np.sign(np.mean(np.vstack(signs), axis=0))
    sma = np.full(n, np.nan)
    csum = np.cumsum(np.concatenate([[0.0], close]))
    sma[SMA_D1 - 1:] = (csum[SMA_D1:] - csum[:-SMA_D1]) / SMA_D1
    out["sma200"] = np.sign(close - sma)
    return out


def summarize(values: np.ndarray, times: pd.DatetimeIndex) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    n = int(len(values))
    out: dict[str, Any] = {"n": n}
    if n < 2:
        return out
    sd = float(values.std(ddof=1))
    mean = float(values.mean())
    out.update(mean_bps=mean, sd_bps=sd)
    if sd > 0:
        out["t"] = mean / (sd / math.sqrt(n))
    years = times.year
    per_year = {int(y): float(values[years == y].mean()) for y in FULL_YEARS if (years == y).any()}
    out["per_year_mean_bps"] = per_year
    out["positive_full_year_share"] = sum(1 for v in per_year.values() if v > 0) / len(FULL_YEARS)
    return out


def verdict(stats: dict[str, Any], *, scenario_b_mean: float | None = None) -> str:
    t = stats.get("t")
    if t is None or stats.get("positive_full_year_share", 0.0) < YEAR_SHARE:
        return "NO_GO"
    if scenario_b_mean is not None and not scenario_b_mean > 0:
        return "NO_GO"
    if t >= GO_T:
        return "GO"
    if t >= LOVENDE_T:
        return "LOVENDE"
    return "NO_GO"


def _trade_record(tape: Tape, entries, exits, sides, policy) -> dict[str, Any]:
    gross, net = trade_outcomes(tape, entries, exits, sides, policy, financing=True)
    times = tape.time[entries] if len(entries) else pd.DatetimeIndex([], tz="UTC")
    stats = summarize(net, times)
    if len(gross):
        stats["gross_mean_bps"] = float(gross.mean())
        stats["gross_hit_rate"] = float((gross > 0).mean())
    stats["verdict"] = verdict(stats)
    return stats


def scalp_arm(tape: Tape, policy: dict[str, Any], start: pd.Timestamp) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for lookback in MOM_LOOKBACK_BARS:
        for hold in MOM_HOLD_BARS:
            for reverse in (False, True):
                name = f"{'rev' if reverse else 'mom'}_k{lookback}_h{hold}"
                cells[name] = _trade_record(tape, *momentum_trades(tape, lookback=lookback, hold=hold, reverse=reverse, start=start), policy)
    for session, (open_hour, exit_hour) in ORB_SESSIONS.items():
        cells[f"orb_{session}"] = _trade_record(tape, *orb_trades(tape, open_hour=open_hour, exit_hour=exit_hour, start=start), policy)
    for window, (start_hour, end_hour) in SESSION_WINDOWS.items():
        for side, label in ((1, "long"), (-1, "short")):
            cells[f"session_{window}_{label}"] = _trade_record(
                tape, *session_trades(tape, start_hour=start_hour, end_hour=end_hour, side=side, start=start), policy
            )
    return cells


def swing_arm(tape: Tape, policy: dict[str, Any], start: pd.Timestamp) -> dict[str, Any]:
    closes = d1_closes(tape)
    mid_close = tape.mid[closes]
    signals = swing_signals(mid_close)
    first_valid = int(np.searchsorted(tape.time[closes].asi8, start.value, side="left"))
    first_valid = max(first_valid, max(TSMOM_LOOKBACK_D1), SMA_D1 - 1)
    cells: dict[str, Any] = {}
    for horizon in SWING_HORIZONS_D1:
        offsets: dict[str, list[float]] = {}
        for offset in range(horizon):
            days = np.arange(first_valid + offset, len(closes) - horizon, horizon)
            entries, exits = closes[days], closes[days + horizon]
            times = tape.time[entries]
            always = np.ones(len(days), dtype=np.int64)
            ref = {f: trade_outcomes(tape, entries, exits, always, policy, financing=f)[1] for f in (True, False)}
            for name, signal in signals.items():
                s = signal[days]
                if np.isnan(s).any():
                    raise RuntimeError("BASELINE_SIGNAL_UNAVAILABLE_AT_DECISION")
                for variant, sides in (("ls", s.astype(np.int64)), ("lf", (s > 0).astype(np.int64))):
                    cell = f"{name}_{variant}_h{horizon}"
                    net_a = trade_outcomes(tape, entries, exits, sides, policy, financing=True)[1]
                    net_b = trade_outcomes(tape, entries, exits, sides, policy, financing=False)[1]
                    delta_a, delta_b = net_a - ref[True], net_b - ref[False]
                    offsets.setdefault(cell, []).append(float(delta_a.mean()))
                    if offset != 0:
                        continue
                    stats = summarize(delta_a, times)
                    stats["strategy_mean_bps_A"] = float(net_a.mean())
                    stats["always_long_mean_bps_A"] = float(ref[True].mean())
                    stats["delta_mean_bps_B"] = float(delta_b.mean())
                    stats["always_long_mean_bps_B"] = float(ref[False].mean())
                    bear = (times >= BEAR_FOLD[0]) & (times < BEAR_FOLD[1])
                    stats["bear_fold_delta_mean_bps_A"] = float(delta_a[bear].mean()) if bear.any() else None
                    stats["bear_fold_n"] = int(bear.sum())
                    stats["verdict"] = verdict(stats, scenario_b_mean=stats["delta_mean_bps_B"])
                    cells[cell] = stats
        for cell, values in offsets.items():
            cells[cell]["offset_robustness_delta_A"] = {"mean": float(np.mean(values)), "min": float(np.min(values))}
    return cells


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
        raise RuntimeError("BASELINE_OUTPUT_EXISTS")
    start = pd.Timestamp(args.eval_start)
    end = pd.Timestamp(args.read_end_exclusive)
    if start.tzinfo is None or end.tzinfo is None:
        raise RuntimeError("BASELINE_TIMES_MUST_BE_UTC_AWARE")
    tape = load_tape(args.native_m5_root, truncate_before=end)
    policy = load_cost_policy(args.cost_policy, args.cost_policy_sha256)
    scalp = scalp_arm(tape, policy, start)
    swing = swing_arm(tape, policy, start)
    verdicts = {name: cell["verdict"] for name, cell in {**scalp, **swing}.items()}
    result = {
        "schema_version": "gx1_model_free_baselines_v1",
        "preregistration": "docs/MODEL_FREE_BASELINES_PREREG_20260927.md",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_clean": _git("status", "--porcelain") == "",
        "tape": {"root": str(args.native_m5_root), "manifest_sha256": tape.manifest_sha256,
                 "first_bar": str(tape.time[0]), "last_bar": str(tape.time[-1]), "bars": len(tape.time)},
        "cost_policy": {k: v for k, v in policy.items() if k != "decision"},
        "eval_start": str(start),
        "read_end_exclusive": str(end),
        "cells": {"scalp": scalp, "swing": swing},
        "decision": {
            "scalp_go": sorted(n for n, v in verdicts.items() if v == "GO" and n in scalp),
            "swing_go": sorted(n for n, v in verdicts.items() if v == "GO" and n in swing),
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
