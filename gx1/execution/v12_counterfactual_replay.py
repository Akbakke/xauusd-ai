#!/usr/bin/env python3
"""V12 counterfactual replay — daily what-if analysis on paper-runner journal.

For each candidate logged by v12_paper_runner.py (TAKE + SKIP + BLOCKED), look
up what the actual forward-outcome would have been at K_HORIZONS using collected
M1 data (or backfill via OANDA). Reports:

  - "Missed opportunities": SKIP/BLOCKED candidates where forward MFE > 50 bps
  - "Correct skips": SKIP candidates where forward outcome was loss
  - "False takes": TAKE trades with negative outcome
  - PnL distribution per decision-class

Output: daily counterfactual report + JSONL with per-candidate what-if metrics
        for offline analysis and future ML retraining feedback.

Run (once per day, after K=1440 bars / 24h have passed):
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_counterfactual_replay.py \\
        --journal-date 20260511 [--journal-suffix shadow]

Reads: /home/andre2/GX1_DATA/reports/v12_paper_runs/v12_paper_journal_*.jsonl
M1 source: /home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_*.parquet
           (collector output) + canonical M1 tape as fallback.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("v12_cf_replay")
PAPER_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
LIVE_DATA_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
M1_TAPE_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")

K_HORIZONS_BPS = [12, 60, 240, 480, 1440]   # M1 minutes — match V12 trainer
HIGH_CONVICTION_BPS_THRESHOLD = 50.0


def load_m1_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Concatenate M1 bars from live collector + canonical tape covering [start, end].

    Live collector parquets cover from when collector started; canonical M1 tape
    covers historical. Use both, dedupe by time.
    """
    parts = []
    # Live collector
    for fp in sorted(LIVE_DATA_DIR.glob("xauusd_m1_*.parquet")):
        df = pd.read_parquet(fp)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    # Canonical M1 tape (year partitions)
    for year in range(start_ts.year, end_ts.year + 1):
        fp = M1_TAPE_DIR / f"year={year}" / "part-000.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["time"], keep="last")
            .sort_values("time")
            .reset_index(drop=True))


def compute_forward_outcome(m1: pd.DataFrame, entry_idx: int,
                             k_horizons: list[int]) -> dict[str, float]:
    """For a hypothetical trade at entry_idx, compute forward PnL stats per K-horizon.

    Returns max-MFE, terminal-PnL, and giveback per K for both LONG and SHORT.
    Spread-aware (long: ask_open entry, bid_close exit; short: reversed).
    """
    out: dict[str, float] = {}
    if entry_idx >= len(m1):
        return out
    entry_bar = m1.iloc[entry_idx]
    long_entry = float(entry_bar["ask_open"])
    short_entry = float(entry_bar["bid_open"])
    if long_entry <= 0 or short_entry <= 0:
        return out

    for K in k_horizons:
        end_idx = min(entry_idx + K, len(m1) - 1)
        if end_idx <= entry_idx:
            continue
        window = m1.iloc[entry_idx + 1: end_idx + 1]
        if len(window) == 0:
            continue
        # LONG: peak = max bid_high, terminal = bid_close at K
        long_peak = float(window["bid_high"].max())
        long_terminal = float(window["bid_close"].iloc[-1])
        long_max_mfe_bps = (long_peak - long_entry) / long_entry * 10000.0
        long_terminal_bps = (long_terminal - long_entry) / long_entry * 10000.0
        # SHORT: peak (favorable) = min ask_low (lower price), terminal = ask_close
        short_peak = float(window["ask_low"].min())
        short_terminal = float(window["ask_close"].iloc[-1])
        short_max_mfe_bps = (short_entry - short_peak) / short_entry * 10000.0
        short_terminal_bps = (short_entry - short_terminal) / short_entry * 10000.0

        out[f"long_max_mfe_K{K}"] = long_max_mfe_bps
        out[f"long_terminal_K{K}"] = long_terminal_bps
        out[f"short_max_mfe_K{K}"] = short_max_mfe_bps
        out[f"short_terminal_K{K}"] = short_terminal_bps

    return out


def replay_journal(journal_path: Path) -> tuple[list[dict], dict]:
    """Replay every event in journal; compute counterfactual + classify outcomes."""
    if not journal_path.exists():
        LOG.error(f"journal not found: {journal_path}")
        return [], {}

    events = []
    with journal_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    if not events:
        return [], {}

    # Determine M1 window we need
    ts_strs = [e.get("ts_utc") for e in events if e.get("ts_utc")]
    earliest = pd.to_datetime(min(ts_strs), utc=True)
    latest = pd.to_datetime(max(ts_strs), utc=True)
    fetch_end = latest + pd.Timedelta(minutes=max(K_HORIZONS_BPS) + 60)
    LOG.info(f"loading M1 window {earliest} → {fetch_end} for {len(events)} candidates")
    m1 = load_m1_window(earliest, fetch_end)
    if m1.empty:
        LOG.error("no M1 data available — backfill collector or wait 24h post-trades")
        return events, {"error": "no_m1_data"}
    m1_time_arr = m1["time"].values  # numpy datetime64 for fast searchsorted

    enriched = []
    for ev in events:
        ts_str = ev.get("ts_utc")
        if not ts_str:
            enriched.append(ev)
            continue
        ts = pd.to_datetime(ts_str, utc=True)
        idx = int(np.searchsorted(m1_time_arr, ts.value, side="left"))
        if idx >= len(m1):
            enriched.append(ev)
            continue
        cf = compute_forward_outcome(m1, idx, K_HORIZONS_BPS)
        ev["counterfactual"] = cf
        # Classify: best_what_if_pnl, missed_opportunity flag (direction-aware)
        if cf:
            best_long_mfe = max(cf.get(f"long_max_mfe_K{K}", -9999) for K in K_HORIZONS_BPS)
            best_short_mfe = max(cf.get(f"short_max_mfe_K{K}", -9999) for K in K_HORIZONS_BPS)
            best_what_if = max(best_long_mfe, best_short_mfe)
            ev["best_what_if_pnl_bps"] = best_what_if
            ev["best_what_if_side"] = "long" if best_long_mfe >= best_short_mfe else "short"
            decision = ev.get("v12_decision", {}).get("action", "UNKNOWN")
            order_status = ev.get("order_status", "UNKNOWN")
            # Direction-aware proposed-side MFE:
            #   TAKE_LONG_NOW gated by spread → use long_max_mfe (V12 wanted long)
            #   TAKE_SHORT_NOW gated → use short_max_mfe
            #   SKIP → use best of either side (V12 had no proposed direction)
            if decision == "TAKE_LONG_NOW":
                proposed_side_mfe = best_long_mfe
                proposed_side = "long"
            elif decision == "TAKE_SHORT_NOW":
                proposed_side_mfe = best_short_mfe
                proposed_side = "short"
            else:
                proposed_side_mfe = best_what_if
                proposed_side = ev["best_what_if_side"]
            ev["proposed_side_mfe_bps"] = proposed_side_mfe
            ev["proposed_side"] = proposed_side
            ev["missed_opportunity"] = (
                (decision == "SKIP" or order_status == "BLOCKED_BY_GATE")
                and proposed_side_mfe >= HIGH_CONVICTION_BPS_THRESHOLD
            )
        enriched.append(ev)

    # Aggregate stats
    stats = {
        "total_events": len(enriched),
        "decisions": {},
        "missed_opportunity_count": 0,
        "missed_opportunity_total_bps": 0.0,
        "high_value_missed_50plus": 0,
        "high_value_missed_100plus": 0,
    }
    for ev in enriched:
        d = ev.get("v12_decision", {}).get("action", "UNKNOWN")
        stats["decisions"].setdefault(d, 0)
        stats["decisions"][d] += 1
        if ev.get("missed_opportunity"):
            stats["missed_opportunity_count"] += 1
            bps = ev.get("best_what_if_pnl_bps", 0)
            stats["missed_opportunity_total_bps"] += bps
            if bps >= 50:
                stats["high_value_missed_50plus"] += 1
            if bps >= 100:
                stats["high_value_missed_100plus"] += 1
    if stats["missed_opportunity_count"] > 0:
        stats["missed_opportunity_mean_bps"] = (
            stats["missed_opportunity_total_bps"] / stats["missed_opportunity_count"]
        )
    return enriched, stats


def main() -> int:
    p = argparse.ArgumentParser(description="V12 counterfactual replay (daily what-if analysis)")
    p.add_argument("--journal-date", type=str, required=True,
                   help="Date in YYYYMMDD format (e.g., 20260511)")
    p.add_argument("--journal-suffix", type=str, default="",
                   help="Suffix matching paper-runner --journal-suffix")
    p.add_argument("--out-dir", type=str, default=str(PAPER_DIR / "counterfactual_reports"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    suf = f"_{args.journal_suffix}" if args.journal_suffix else ""
    journal_path = PAPER_DIR / f"v12_paper_journal_{args.journal_date}{suf}.jsonl"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_journal = out_dir / f"counterfactual_{args.journal_date}{suf}.jsonl"
    out_summary = out_dir / f"counterfactual_summary_{args.journal_date}{suf}.json"

    LOG.info(f"replaying {journal_path}")
    enriched, stats = replay_journal(journal_path)

    with out_journal.open("w") as f:
        for ev in enriched:
            f.write(json.dumps(ev, default=str) + "\n")
    out_summary.write_text(json.dumps(stats, indent=2, default=str))

    print("\n=== V12 Counterfactual Daily Report ===")
    print(json.dumps(stats, indent=2, default=str))
    print(f"\nFull enriched journal: {out_journal}")

    # Highlight high-value missed opportunities
    missed = [e for e in enriched if e.get("missed_opportunity")]
    if missed:
        print(f"\nTop 10 missed opportunities (>= 50 bps forward MFE):")
        missed_sorted = sorted(missed, key=lambda e: e.get("best_what_if_pnl_bps", 0), reverse=True)
        for ev in missed_sorted[:10]:
            ts = ev.get("ts_utc", "?")
            bps = ev.get("best_what_if_pnl_bps", 0)
            side = ev.get("best_what_if_side", "?")
            decision = ev.get("v12_decision", {}).get("action", "?")
            order = ev.get("order_status", "?")
            spread = ev.get("spread_bps", 0)
            print(f"  {ts}  {side:5s} would-have +{bps:6.1f} bps   "
                  f"V12 said {decision} (order_status={order})  spread={spread:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
