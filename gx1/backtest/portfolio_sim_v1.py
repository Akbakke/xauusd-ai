"""
Sequential MAE/drawdown-aware portfolio sim (companion to Phase 6 joint validation).

Why this exists
---------------
Phase 6 (v12_phase6_joint_validation.py) scores every candidate INDEPENDENTLY: its
"+X bps/take" is the mean of per-trade forward outcomes, blind to (a) how many trades
are open at once, (b) the live concurrency cap (max_trades) that DROPS entries when the
book is full, and (c) account-level drawdown when many correlated same-side trades pile
up and all move adverse together. The 2026-06-03 diagnosis found live opened up to ~36
concurrent positions in places and that pile-ups (e.g. the 16 stacked shorts on 06-02)
are the real live pain — none of which Phase 6 can represent.

This module replays the Phase 6 per-candidate takes as an event-driven sequential
portfolio with a configurable concurrency cap, and reports realized-equity drawdown,
concurrency distribution, and a worst-case concurrent-MAE proxy. cap=inf reproduces the
Phase 6 "all taken" view, so a cap sweep shows exactly what the cap costs/saves.

Inputs
------
A Phase 6 `per_candidate_*.csv` (or any frame) with columns:
  candidate_uid (entry M5 ts encoded as trailing ...YYYYMMDDTHHMMSS), side_v1,
  realized_pnl_bps, exit_bar (1-based M1 bars held), and (new 2026-06-03) mae_bps.

Risk model: equal notional per trade; equity is cumulative bps. This isolates the
STRUCTURAL effects (cap, concurrency, pile-up MAE) from position-sizing choices.

Contract: no exceptions in the core sim; pure functions over an in-memory frame.
"""

from __future__ import annotations

import heapq
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_UID_TS = re.compile(r"(\d{8}T\d{6})")


def _parse_entry_ts(uid: str) -> pd.Timestamp:
    """Entry M5 timestamp encoded as the LAST YYYYMMDDTHHMMSS token in the uid."""
    matches = _UID_TS.findall(str(uid))
    if not matches:
        return pd.NaT
    return pd.to_datetime(matches[-1], format="%Y%m%dT%H%M%S", utc=True)


@dataclass(frozen=True)
class _Trade:
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    pnl_bps: float
    mae_bps: float


def _build_trades(df: pd.DataFrame) -> list[_Trade]:
    take = df[df.get("exit_reason", pd.Series(["TAKE"] * len(df))) != "ENTRY_IQL_SKIP"].copy()
    take = take[take["candidate_uid"].notna()]
    trades: list[_Trade] = []
    for _, r in take.iterrows():
        ent = _parse_entry_ts(r["candidate_uid"])
        if pd.isna(ent):
            continue
        exit_bar = int(r.get("exit_bar", 1) or 1)
        hold_min = max(1, exit_bar)  # exit_bar is 1-based M1 bars
        trades.append(_Trade(
            entry_ts=ent,
            exit_ts=ent + pd.Timedelta(minutes=hold_min),
            side=str(r.get("side_v1", "?")),
            pnl_bps=float(r.get("realized_pnl_bps", 0.0)),
            mae_bps=float(r.get("mae_bps", 0.0)) if not pd.isna(r.get("mae_bps", np.nan)) else 0.0,
        ))
    trades.sort(key=lambda t: t.entry_ts)
    return trades


def _has_mae(df: pd.DataFrame) -> bool:
    """True if the frame carries a usable (non-all-zero/NaN) mae_bps column."""
    if "mae_bps" not in df.columns:
        return False
    m = pd.to_numeric(df["mae_bps"], errors="coerce")
    return bool(m.notna().any() and (m.abs() > 1e-9).any())


def simulate_portfolio(df: pd.DataFrame, max_concurrent: int = 5,
                       enforce_invariants: bool = True,
                       require_mae: bool = False) -> dict[str, Any]:
    """Event-driven sequential portfolio sim with a concurrency cap.

    Entries are processed in time order. Before each entry, all trades whose exit_ts has
    passed are closed (their pnl realized). A new entry is ADMITTED only if fewer than
    `max_concurrent` trades are open; otherwise it is DROPPED (the live cap rejects it).
    Returns a JSON-safe summary. `max_concurrent=10**9` => no cap (reproduces Phase 6).

    2026-06-03 (S6): also tracks the live execution invariants so Phase 6 can gate on
    them — signed net exposure, opposing-side overlaps, and max same-side concurrency.
    When `enforce_invariants` (default) the sim MODELS the always-on safety layer: it
    refuses opposing-side entries and same-side beyond the cap (so a healthy variant
    reports 0 opposing overlaps). `require_mae=True` fail-closes if mae_bps is absent
    (the concurrent-MAE proxy would otherwise silently read 0).
    """
    trades = _build_trades(df)
    if not trades:
        return {"n_trades_offered": 0, "note": "no takes with parseable entry ts"}
    if require_mae and not _has_mae(df):
        raise ValueError("portfolio_sim require_mae=True but per-candidate frame has no "
                         "non-zero mae_bps column — refusing to report a 0 concurrent-MAE proxy "
                         "(re-run Phase 6 with the 2026-06-03 MAE emitter)")

    open_heap: list[tuple[pd.Timestamp, int, _Trade]] = []   # (exit_ts, seq, trade)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    realized = []
    n_admitted = 0
    n_dropped = 0
    n_blocked_opposing = 0       # entries refused because an opposing-side trade was open
    concur_samples = []
    worst_concurrent_mae = 0.0   # most-negative sum of open-trade MAE seen at any entry
    n_opposing_overlaps = 0      # admitted entries that left long & short open at once
    max_same_side_concurrent = 0
    max_abs_net_exposure = 0     # max |open_long - open_short| (signed-exposure pile-up)
    seq = 0   # monotonic tiebreaker so ties on exit_ts never compare _Trade objects

    def _close_due(now: pd.Timestamp):
        nonlocal equity, peak, max_dd
        while open_heap and open_heap[0][0] <= now:
            _, _, tr = heapq.heappop(open_heap)
            equity += tr.pnl_bps
            realized.append(tr.pnl_bps)
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

    def _counts():
        nl = sum(1 for _, _, t in open_heap if t.side == "long")
        ns = sum(1 for _, _, t in open_heap if t.side == "short")
        return nl, ns

    for tr in trades:
        _close_due(tr.entry_ts)
        nl, ns = _counts()
        concur_samples.append(nl + ns)
        n_same = nl if tr.side == "long" else ns
        n_opp = ns if tr.side == "long" else nl
        open_mae = sum(t.mae_bps for _, _, t in open_heap) + tr.mae_bps
        worst_concurrent_mae = min(worst_concurrent_mae, open_mae)
        # Admission. enforce_invariants models the always-on live safety layer:
        # block opposing-side, and same-side beyond the cap. Otherwise the original
        # total-concurrency cap (Phase 6 'all taken' baseline shape).
        if enforce_invariants:
            if n_opp > 0:
                n_blocked_opposing += 1
                n_dropped += 1
                continue
            if n_same >= max_concurrent:
                n_dropped += 1
                continue
        elif len(open_heap) >= max_concurrent:
            n_dropped += 1
            continue
        heapq.heappush(open_heap, (tr.exit_ts, seq, tr))
        seq += 1
        n_admitted += 1
        nl2, ns2 = _counts()
        max_same_side_concurrent = max(max_same_side_concurrent, nl2, ns2)
        max_abs_net_exposure = max(max_abs_net_exposure, abs(nl2 - ns2))
        if nl2 > 0 and ns2 > 0:
            n_opposing_overlaps += 1
    # drain remaining open trades
    far = trades[-1].exit_ts + pd.Timedelta(days=3650)
    _close_due(far)

    realized = np.asarray(realized, dtype=np.float64)
    concur = np.asarray(concur_samples, dtype=np.float64)
    return {
        "max_concurrent_cap": (None if max_concurrent >= 10**9 else int(max_concurrent)),
        "enforce_invariants": bool(enforce_invariants),
        "n_trades_offered": len(trades),
        "n_admitted": int(n_admitted),
        "n_dropped_by_cap": int(n_dropped),
        "n_blocked_opposing": int(n_blocked_opposing),
        "admit_rate": round(n_admitted / len(trades), 4),
        "total_realized_pnl_bps": round(float(realized.sum()), 1),
        "mean_pnl_per_admitted_bps": round(float(realized.mean()), 2) if len(realized) else 0.0,
        "win_rate_admitted": round(float((realized > 0).mean()), 4) if len(realized) else 0.0,
        "max_account_drawdown_bps": round(float(max_dd), 1),          # realized-equity peak-to-trough (<=0)
        "max_concurrent_observed": int(concur.max()) if len(concur) else 0,
        "mean_concurrent_observed": round(float(concur.mean()), 2) if len(concur) else 0.0,
        "p95_concurrent_observed": int(np.quantile(concur, 0.95)) if len(concur) else 0,
        "worst_concurrent_mae_bps": round(float(worst_concurrent_mae), 1),  # proxy: sum of open-trade MAE (<=0)
        # --- side-aware execution-invariant metrics (S6) ---
        "n_opposing_side_overlaps": int(n_opposing_overlaps),
        "max_same_side_concurrent": int(max_same_side_concurrent),
        "max_abs_net_exposure": int(max_abs_net_exposure),
    }


def cap_sweep(df: pd.DataFrame, caps=(1, 2, 3, 5, 10, 10**9),
              enforce_invariants: bool = False) -> list[dict[str, Any]]:
    """Run the sim across concurrency caps. cap=10**9 == Phase 6 'all taken' baseline.

    Default enforce_invariants=False: total-concurrency cap (original semantics), and the
    side-aware metrics then expose the RAW model's natural opposing/pile-up tendency.
    The Phase 6 gate runs a separate enforce_invariants=True pass to model the live
    safety layer's realistic drawdown/admit_rate.
    """
    return [simulate_portfolio(df, max_concurrent=c, enforce_invariants=enforce_invariants)
            for c in caps]


def main() -> int:
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser(description="Sequential MAE/drawdown-aware portfolio sim")
    p.add_argument("--per-candidate-csv", required=True, help="Phase 6 per_candidate_*.csv")
    p.add_argument("--caps", default="1,2,3,5,10,inf",
                   help="Comma-separated concurrency caps; 'inf' = no cap (Phase 6 baseline)")
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()
    df = pd.read_csv(args.per_candidate_csv)
    caps = [10**9 if c.strip().lower() in ("inf", "none") else int(c) for c in args.caps.split(",")]
    sweep = cap_sweep(df, caps=caps)
    print(json.dumps(sweep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(sweep, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    # tiny self-test then (if args) CLI
    import sys
    if len(sys.argv) == 1:
        ts = pd.date_range("2026-01-01 09:00", periods=6, freq="2min")
        df = pd.DataFrame({
            "candidate_uid": [f"X:cand:v2_inf:{t.strftime('%Y%m%dT%H%M%S')}" for t in ts],
            "side_v1": ["long"] * 6,
            "realized_pnl_bps": [10, -5, 20, -8, 15, 30],
            "exit_bar": [30, 30, 30, 30, 30, 30],   # all hold 30 min => heavy overlap
            "mae_bps": [-12, -20, -3, -25, -5, -2],
            "exit_reason": ["EXIT_IQL_SIGNAL"] * 6,
        })
        uncapped = simulate_portfolio(df, max_concurrent=10**9)
        capped = simulate_portfolio(df, max_concurrent=2)
        assert uncapped["n_admitted"] == 6, uncapped
        assert capped["n_dropped_by_cap"] > 0, capped
        assert capped["max_concurrent_observed"] <= 2, capped
        assert uncapped["worst_concurrent_mae_bps"] <= capped["worst_concurrent_mae_bps"], "cap must reduce concurrent MAE"
        print("portfolio_sim_v1 self-test OK")
        print("  uncapped:", uncapped)
        print("  cap=2  :", capped)
        raise SystemExit(0)
    raise SystemExit(main())
