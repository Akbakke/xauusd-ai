#!/usr/bin/env python3
"""Scheduled US macro events and XAUUSD direction after cost (research only).

Preregistered in docs/MACRO_EVENT_BASELINES_PREREG_20260927.md; the cells, windows and thresholds
below are that registration. The calendar is the hash-bound research calendar built by
gx1/scripts/build_macro_event_calendar_v1.py; fills, costs and statistics reuse the model-free
baseline owners. Never an Entry input (GX1_RULES.md rule 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.scripts.research_entry_direction_walkforward_v1 import Tape, load_cost_policy, load_tape
from gx1.scripts.research_model_free_baselines_v1 import summarize, trade_outcomes

EVENT_WINDOWS = {
    "fomc": (pd.Timestamp("2013-01-01", tz="UTC"), tuple(range(2013, 2025))),
    "nfp": (pd.Timestamp("2011-06-01", tz="UTC"), tuple(range(2012, 2025))),
    "cpi": (pd.Timestamp("2011-06-01", tz="UTC"), tuple(range(2012, 2025))),
}
PRE_ENTRY_OFFSET = pd.Timedelta(hours=24)
PRE_ENTRY_SEARCH = pd.Timedelta(hours=1)
SIGNAL_START_OFFSET = pd.Timedelta(minutes=-5)
SIGNAL_END_OFFSET = pd.Timedelta(minutes=10)
POST_HOLD_BARS = (12, 48)
BAR = pd.Timedelta(minutes=5)
GO_T = 2.77  # one-sided Bonferroni 0.05 / 18 cells
LOVENDE_T = 2.0
YEAR_SHARE = 0.60


def _bar_at(tape: Tape, start: pd.Timestamp) -> int | None:
    position = int(tape.time.searchsorted(start, side="left"))
    if position < len(tape.time) and tape.time[position] == start:
        return position
    return None


def _first_bar_in(tape: Tape, begin: pd.Timestamp, end: pd.Timestamp) -> int | None:
    position = int(tape.time.searchsorted(begin, side="left"))
    if position < len(tape.time) and tape.time[position] < end:
        return position
    return None


def event_trades(tape: Tape, releases: pd.DatetimeIndex) -> dict[str, dict[str, Any]]:
    """Entry/exit/side arrays per cell for one event type, plus skipped counts."""
    cells: dict[str, dict[str, list]] = {}
    skipped: dict[str, int] = {}

    def add(name: str, entry: int, exit_: int, side: int) -> None:
        cell = cells.setdefault(name, {"entry": [], "exit": [], "side": []})
        cell["entry"].append(entry)
        cell["exit"].append(exit_)
        cell["side"].append(side)

    for release in releases:
        pre_entry = _first_bar_in(tape, release - PRE_ENTRY_OFFSET, release - PRE_ENTRY_OFFSET + PRE_ENTRY_SEARCH)
        before = _bar_at(tape, release + SIGNAL_START_OFFSET)
        if pre_entry is None or before is None or pre_entry >= before:
            skipped["pre"] = skipped.get("pre", 0) + 1
        else:
            add("pre_long", pre_entry, before, 1)
            add("pre_short", pre_entry, before, -1)
        after = _bar_at(tape, release + SIGNAL_END_OFFSET)
        signal = 0 if before is None or after is None else int(np.sign(tape.mid[after] - tape.mid[before]))
        for hold in POST_HOLD_BARS:
            exit_ = None if after is None else _bar_at(tape, tape.time[after] + hold * BAR)
            if signal == 0 or exit_ is None:
                skipped[f"post_h{hold}"] = skipped.get(f"post_h{hold}", 0) + 1
                continue
            add(f"post_mom_h{hold}", after, exit_, signal)
            add(f"post_rev_h{hold}", after, exit_, -signal)
    out = {name: {k: np.asarray(v, dtype=np.int64) for k, v in cell.items()} for name, cell in cells.items()}
    return {"cells": out, "skipped": skipped}


def verdict(stats: dict[str, Any]) -> str:
    t = stats.get("t")
    if t is None or stats.get("positive_full_year_share", 0.0) < YEAR_SHARE:
        return "NO_GO"
    if t >= GO_T:
        return "GO"
    if t >= LOVENDE_T:
        return "LOVENDE"
    return "NO_GO"


def load_calendar(root: Path, manifest_sha256: str) -> pd.DataFrame:
    manifest_path = root / "MANIFEST.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
        raise RuntimeError("MACRO_CALENDAR_MANIFEST_HASH_MISMATCH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = root / manifest["events_csv"]["file"]
    if hashlib.sha256(events.read_bytes()).hexdigest() != manifest["events_csv"]["sha256"]:
        raise RuntimeError("MACRO_CALENDAR_EVENTS_HASH_MISMATCH")
    frame = pd.read_csv(events)
    frame["release_utc"] = pd.to_datetime(frame["release_utc"], utc=True)
    return frame


def run(tape: Tape, calendar: pd.DataFrame, policy: dict[str, Any], read_end: pd.Timestamp) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind, (start, full_years) in EVENT_WINDOWS.items():
        releases = pd.DatetimeIndex(calendar.loc[calendar["event"] == kind, "release_utc"])
        releases = releases[(releases >= start) & (releases + max(POST_HOLD_BARS) * BAR < read_end)]
        trades = event_trades(tape, releases)
        cells = {}
        for name, cell in trades["cells"].items():
            gross, net = trade_outcomes(tape, cell["entry"], cell["exit"], cell["side"], policy, financing=True)
            stats = summarize(net, tape.time[cell["entry"]], full_years)
            stats["gross_mean_bps"] = float(gross.mean())
            stats["gross_hit_rate"] = float((gross > 0).mean())
            stats["verdict"] = verdict(stats)
            cells[name] = stats
        result[kind] = {"events": int(len(releases)), "skipped": trades["skipped"], "cells": cells}
    return result


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2]).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-m5-root", type=Path, required=True)
    parser.add_argument("--calendar-root", type=Path, required=True)
    parser.add_argument("--calendar-manifest-sha256", required=True)
    parser.add_argument("--cost-policy", type=Path, required=True)
    parser.add_argument("--cost-policy-sha256", required=True)
    parser.add_argument("--read-end-exclusive", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("MACRO_EVENT_OUTPUT_EXISTS")
    read_end = pd.Timestamp(args.read_end_exclusive)
    if read_end.tzinfo is None:
        raise RuntimeError("MACRO_EVENT_READ_END_MUST_BE_UTC_AWARE")
    tape = load_tape(args.native_m5_root, truncate_before=read_end)
    calendar = load_calendar(args.calendar_root, args.calendar_manifest_sha256)
    policy = load_cost_policy(args.cost_policy, args.cost_policy_sha256)
    events = run(tape, calendar, policy, read_end)
    result = {
        "schema_version": "gx1_macro_event_baselines_v1",
        "preregistration": "docs/MACRO_EVENT_BASELINES_PREREG_20260927.md",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_clean": _git("status", "--porcelain") == "",
        "tape": {"root": str(args.native_m5_root), "manifest_sha256": tape.manifest_sha256,
                 "last_bar": str(tape.time[-1])},
        "calendar": {"root": str(args.calendar_root), "manifest_sha256": args.calendar_manifest_sha256},
        "cost_policy": {k: v for k, v in policy.items() if k != "decision"},
        "events": events,
        "decision": {
            "go": sorted(f"{k}:{n}" for k, e in events.items() for n, c in e["cells"].items() if c["verdict"] == "GO"),
            "lovende": sorted(f"{k}:{n}" for k, e in events.items() for n, c in e["cells"].items() if c["verdict"] == "LOVENDE"),
        },
    }
    args.out_dir.mkdir(parents=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=float)
    (args.out_dir / "results.json").write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "results_sha256": hashlib.sha256(payload.encode()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
