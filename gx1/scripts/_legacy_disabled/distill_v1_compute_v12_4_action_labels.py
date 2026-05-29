#!/usr/bin/env python3
"""Distillation Stage 2.5 — V12.4-policy action labels per bar.

For each bar in V12.2 per-bar dataset, compute what V12.4 (Strategy-F overlay)
WOULD HAVE DONE: HOLD (0) or EXIT_NOW (1).

This is the ground-truth label for V3-distillation (Stage 3b): V3 q_head learns
to predict V12.4's exit-decisions, NOT raw Exit-IQL Q-values. The training
target becomes the COMBINED policy (IQL + Strategy-F overlay), so distilled V3
can take the same exit-decisions without consulting IQL or overlay at runtime.

Reuses the simulate() logic from analyze_strategy_f_smart_exit.py:
  - Rule 1 (PROFIT-LOCK):    MFE ≥ 15 AND drawdown ≥ 30% × MFE
  - Rule 2 (BREAK-EVEN-CUT): MFE ≥ 10 AND pnl < 30% × MFE
  - Rule 3 (STRONG-HOLD):    IQL Q_adv < -200 → suppress 1+2
  - Otherwise: IQL bestemmer (R_V12 action)

Inputs (already exist):
  - V12.2 per-bar dataset:  BUILD_EXIT_IQL_PER_BAR_DATASET_V12_2_*_V3TRACKED_SOLO_*_LOCK/per_week/
  - R_V12 Q-targets cache:  IQL_EXIT_Q_TARGETS_V12_2_*_LOCK/per_week/  (333/333 ready)

Output:
  - V12_4_ACTION_LABELS_<ts>_LOCK/per_week/exit_action_labels_<week>.parquet
    cols: candidate_uid, bar_idx_v1, bar_ts_ns_v1, v12_4_action, decision_source

Schema:
  v12_4_action ∈ {0=HOLD, 1=EXIT_NOW}
  decision_source ∈ {
    "IQL_Q",                        # IQL fyrte EXIT_NOW (passive overlay)
    "MFE_GIVEBACK_OVERRIDE",        # Rule 1 fyrte
    "BREAKEVEN_CUT",                # Rule 2 fyrte
    "HOLD",                         # No exit at this bar
  }

Usage:
  PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -u \\
    gx1/scripts/distill_v1_compute_v12_4_action_labels.py
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# V12.4 policy params — MUST match v12_exit_iql_live.py constants.
# If those change, this script needs updating to stay consistent.
V12_4_PROFIT_LOCK_MIN_MFE   = 15.0
V12_4_PROFIT_LOCK_GIVEBACK  = 0.30
V12_4_BREAKEVEN_MIN_MFE     = 10.0
V12_4_BREAKEVEN_RATIO       = 0.30
V12_4_STRONG_HOLD_QADV      = -200.0


PER_BAR_ROOT = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_2_20260514T161504Z_R4_LOCK_"
    "V3TRACKED_SOLO_20260515T004836Z_LOCK"
)
Q_TARGETS_ROOT = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "IQL_EXIT_Q_TARGETS_V12_2_20260516T073004Z_LOCK"
)
OUT_PARENT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")


def compute_v12_4_actions(bars: pd.DataFrame, q_targets: pd.DataFrame) -> pd.DataFrame:
    """For each row, compute V12.4-policy action + decision_source.

    bars: per-bar dataframe with current_mfe_bps_v1, pnl_drawdown_from_peak_v1,
          current_unrealized_pnl_bps_v1, candidate_uid, bar_idx_v1, bar_ts_ns_v1
    q_targets: Stage 2a R_V12 Q-targets (candidate_uid, bar_idx_v1,
               iql_exit_q_advantage_v1, iql_exit_action_v1)

    Returns: dataframe with candidate_uid, bar_idx_v1, bar_ts_ns_v1,
             v12_4_action, decision_source
    """
    # Join bars + q_targets on (candidate_uid, bar_idx_v1)
    bars = bars[["candidate_uid", "bar_idx_v1", "bar_ts_ns_v1",
                 "current_mfe_bps_v1", "pnl_drawdown_from_peak_v1",
                 "current_unrealized_pnl_bps_v1",
                 "is_never_fire_v1", "forced_terminal_v1"]].copy()
    bars["uid_str"] = bars["candidate_uid"].astype(str)
    q_targets["uid_str"] = q_targets["candidate_uid"].astype(str)
    merged = bars.merge(
        q_targets[["uid_str", "bar_idx_v1",
                   "iql_exit_q_advantage_v1", "iql_exit_action_v1"]],
        on=["uid_str", "bar_idx_v1"], how="inner",
    )

    mfe   = merged["current_mfe_bps_v1"].to_numpy(dtype=np.float32)
    pnl   = merged["current_unrealized_pnl_bps_v1"].to_numpy(dtype=np.float32)
    drawdn= merged["pnl_drawdown_from_peak_v1"].to_numpy(dtype=np.float32)
    iql_q = merged["iql_exit_q_advantage_v1"].to_numpy(dtype=np.float32)
    iql_a = merged["iql_exit_action_v1"].to_numpy(dtype=np.int32)

    # Triggers
    profit_lock = (mfe >= V12_4_PROFIT_LOCK_MIN_MFE) \
                  & (drawdn >= V12_4_PROFIT_LOCK_GIVEBACK * mfe) \
                  & (mfe > 0)
    breakeven  = (mfe >= V12_4_BREAKEVEN_MIN_MFE) \
                  & (pnl < V12_4_BREAKEVEN_RATIO * mfe)
    strong_hold = (iql_q < V12_4_STRONG_HOLD_QADV)

    f_fires = (profit_lock | breakeven) & (~strong_hold)
    iql_exit = (iql_a == 1) & (~f_fires)
    v12_4_action = (f_fires | iql_exit).astype(np.int32)

    # Decision-source priority: profit-lock > breakeven > IQL_Q > HOLD
    decision_source = np.full(len(merged), "HOLD", dtype=object)
    decision_source[iql_exit] = "IQL_Q"
    decision_source[f_fires & profit_lock] = "MFE_GIVEBACK_OVERRIDE"
    decision_source[f_fires & breakeven & ~profit_lock] = "BREAKEVEN_CUT"

    return pd.DataFrame({
        "candidate_uid": merged["candidate_uid"],
        "bar_idx_v1":    merged["bar_idx_v1"],
        "bar_ts_ns_v1":  merged["bar_ts_ns_v1"],
        "v12_4_action":  v12_4_action,
        "decision_source": decision_source,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=None,
                    help="Default: V12_4_ACTION_LABELS_<ts>_LOCK under reports root")
    ap.add_argument("--start-week", type=int, default=0)
    ap.add_argument("--end-week",   type=int, default=None)
    args = ap.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root or
                    (OUT_PARENT / f"V12_4_ACTION_LABELS_{ts}_LOCK")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_week").mkdir(exist_ok=True)

    print(f"[V12_4_LABELS] V12.4 policy params:")
    print(f"  PROFIT_LOCK: min_mfe={V12_4_PROFIT_LOCK_MIN_MFE}  giveback={V12_4_PROFIT_LOCK_GIVEBACK}")
    print(f"  BREAKEVEN:   min_mfe={V12_4_BREAKEVEN_MIN_MFE}  ratio={V12_4_BREAKEVEN_RATIO}")
    print(f"  STRONG_HOLD: q_adv < {V12_4_STRONG_HOLD_QADV}")
    print(f"  Output: {out_root}")
    print()

    q_files = sorted((Q_TARGETS_ROOT / "per_week").glob("exit_iql_q_targets_*.parquet"))
    start, end = args.start_week, args.end_week or len(q_files)
    print(f"Processing weeks [{start}, {end}) of {len(q_files)}")

    t0 = time.time()
    decision_counts = {"IQL_Q": 0, "MFE_GIVEBACK_OVERRIDE": 0, "BREAKEVEN_CUT": 0, "HOLD": 0}
    total_rows = 0

    for w_idx, q_pq in enumerate(q_files):
        if w_idx < start or w_idx >= end:
            continue
        week_name = q_pq.stem.removeprefix("exit_iql_q_targets_")
        out_path = out_root / "per_week" / f"exit_action_labels_{week_name}.parquet"
        if out_path.exists():
            continue

        per_bar_pq = PER_BAR_ROOT / "per_week" / f"exit_per_bar_m1_{week_name}.parquet"
        if not per_bar_pq.exists():
            continue

        bars = pd.read_parquet(per_bar_pq)
        bars = bars[(bars["is_never_fire_v1"] == 0) & (bars["forced_terminal_v1"] == 0)]
        if len(bars) == 0:
            continue
        q_targets = pd.read_parquet(q_pq)
        labels = compute_v12_4_actions(bars, q_targets)

        labels.to_parquet(out_path, index=False)
        total_rows += len(labels)
        for src, n in labels["decision_source"].value_counts().items():
            decision_counts[src] = decision_counts.get(src, 0) + int(n)

        if (w_idx - start + 1) % 25 == 0 or w_idx == end - 1:
            elapsed = time.time() - t0
            rate = total_rows / max(1, elapsed)
            print(f"  [{w_idx+1}/{end}] {week_name}  total_rows={total_rows:,}  "
                  f"({rate:.0f}/s, {elapsed:.0f}s)", flush=True)
        gc.collect()

    print()
    print(f"[V12_4_LABELS] DONE — {total_rows:,} rows over {end - start} weeks")
    print(f"Decision-source distribution:")
    total = sum(decision_counts.values()) or 1
    for src, n in sorted(decision_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<28} {n:>10,}  ({100*n/total:>5.1f}%)")

    summary = {
        "action_v1": "V12_4_ACTION_LABELS_COMPUTE",
        "ran_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "v12_4_policy_params": {
            "profit_lock_min_mfe": V12_4_PROFIT_LOCK_MIN_MFE,
            "profit_lock_giveback": V12_4_PROFIT_LOCK_GIVEBACK,
            "breakeven_min_mfe":   V12_4_BREAKEVEN_MIN_MFE,
            "breakeven_ratio":     V12_4_BREAKEVEN_RATIO,
            "strong_hold_qadv":    V12_4_STRONG_HOLD_QADV,
        },
        "n_rows_total":   total_rows,
        "decision_counts": decision_counts,
        "wall_seconds":   time.time() - t0,
    }
    (out_root / "summary_v1.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Summary: {out_root}/summary_v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
