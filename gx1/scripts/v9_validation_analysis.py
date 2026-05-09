"""
V9 validation analysis (2026-05-07).

Re-runnable analysis on Phase 7 per-candidate output. Documents the key
findings that pivoted V9 priority from Entry-IQL SKIP (false problem) to
Exit-IQL activity (real goldmine).

Outputs both stdout summary and a JSON snapshot for permanent record.

Usage:
  python -m gx1.scripts.v9_validation_analysis [--phase7-dir <path>] [--out <json>]

If --phase7-dir omitted, defaults to wave 2 baseline:
  /home/andre2/GX1_DATA/reports/truth_e2e_sanity/JOINT_ENTRY_EXIT_IQL_VALIDATION_GATE_V2_20260506T133947Z_LOCK
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PHASE7 = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "JOINT_ENTRY_EXIT_IQL_VALIDATION_GATE_V2_20260506T133947Z_LOCK"
)


def analyse(phase7_dir: Path) -> dict:
    csv_path = phase7_dir / "per_candidate_joint_eval_v1.csv"
    df = pd.read_csv(csv_path)
    n_total = len(df)

    out: dict = {
        "phase7_source": str(csv_path),
        "n_candidates": int(n_total),
        "overall_mean_pnl_bps": float(df["joint_pnl_bps_v1"].mean()),
    }

    # Action breakdown
    action_breakdown = df.groupby("entry_action_label_v1")["joint_pnl_bps_v1"].agg(
        ["count", "mean", "sum"]
    )
    out["action_breakdown"] = {
        k: {"count": int(v["count"]), "mean_pnl_bps": float(v["mean"]),
            "total_pnl_bps": float(v["sum"])}
        for k, v in action_breakdown.iterrows()
    }

    # Loss rate
    losers = df[df["joint_pnl_bps_v1"] < 0]
    out["overall_loss_rate"] = float(len(losers) / n_total)
    out["overall_loser_total_bps"] = float(losers["joint_pnl_bps_v1"].sum())

    # === D: Survivorship bias check ===
    exit_iql = df[df["exit_reason_v1"] == "EXIT_IQL_SIGNAL"].copy()
    forced = df[df["exit_reason_v1"] == "FORCED_TERMINAL"].copy()
    skipped = df[df["exit_reason_v1"] == "ENTRY_IQL_SKIP"].copy()

    out["exit_reason"] = {
        "EXIT_IQL_SIGNAL": {
            "count": int(len(exit_iql)),
            "mean_pnl_bps": float(exit_iql["joint_pnl_bps_v1"].mean()),
            "loss_rate": float((exit_iql["joint_pnl_bps_v1"] < 0).mean()),
            "p25": float(exit_iql["joint_pnl_bps_v1"].quantile(.25)),
            "median": float(exit_iql["joint_pnl_bps_v1"].median()),
            "p75": float(exit_iql["joint_pnl_bps_v1"].quantile(.75)),
            "max": float(exit_iql["joint_pnl_bps_v1"].max()),
        },
        "FORCED_TERMINAL": {
            "count": int(len(forced)),
            "mean_pnl_bps": float(forced["joint_pnl_bps_v1"].mean()),
            "loss_rate": float((forced["joint_pnl_bps_v1"] < 0).mean()),
            "p25": float(forced["joint_pnl_bps_v1"].quantile(.25)),
            "median": float(forced["joint_pnl_bps_v1"].median()),
            "p75": float(forced["joint_pnl_bps_v1"].quantile(.75)),
            "max": float(forced["joint_pnl_bps_v1"].max()),
        },
        "ENTRY_IQL_SKIP": {
            "count": int(len(skipped)),
            "mean_pnl_bps": float(skipped["joint_pnl_bps_v1"].mean()),
        },
    }

    # Bar-bucket survivorship breakdown
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 96)]
    out["exit_iql_by_bar_bucket"] = {}
    for lo, hi in buckets:
        sub = exit_iql[(exit_iql["exit_bar_v1"] >= lo) & (exit_iql["exit_bar_v1"] < hi)]
        if len(sub) > 0:
            out["exit_iql_by_bar_bucket"][f"{lo}-{hi}"] = {
                "count": int(len(sub)),
                "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            }

    # === B: Deep dive ===
    # Side breakdown
    out["side_breakdown"] = {
        "exit_iql_signal": {
            side: {
                "count": int(len(sub)),
                "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            }
            for side, sub in exit_iql.groupby("side_used_v1")
        },
        "forced_terminal": {
            side: {
                "count": int(len(sub)),
                "mean_pnl_bps": float(sub["joint_pnl_bps_v1"].mean()),
            }
            for side, sub in forced.groupby("side_used_v1")
        },
    }

    # Time-of-day for EXIT_IQL_SIGNAL only
    exit_iql["hour"] = pd.to_datetime(exit_iql["decision_ts_utc"]).dt.hour
    out["exit_iql_by_hour"] = {
        int(h): {"count": int(v["count"]), "mean_pnl_bps": float(v["mean"])}
        for h, v in exit_iql.groupby("hour")["joint_pnl_bps_v1"].agg(["count", "mean"]).iterrows()
        if v["count"] > 5  # filter sparse hours
    }

    # Q-value correlation
    out["entry_advantage_pnl_correlation"] = float(
        df["entry_advantage_over_skip_bps_v1"].corr(df["joint_pnl_bps_v1"])
    )

    # === Counterfactual: would skipping low-Q-advantage trades help? ===
    df_sorted = df.sort_values("entry_advantage_over_skip_bps_v1")
    out["skip_bottom_pct_counterfactual"] = {}
    for pct in [1, 3, 5, 10, 20]:
        n = int(n_total * pct / 100)
        skipped_subset = df_sorted.head(n)
        kept = df_sorted.tail(n_total - n)
        out["skip_bottom_pct_counterfactual"][str(pct)] = {
            "n_skipped": int(n),
            "skipped_mean_pnl": float(skipped_subset["joint_pnl_bps_v1"].mean()),
            "new_overall_mean": float(kept["joint_pnl_bps_v1"].sum() / n_total),
            "delta_vs_baseline": float(
                kept["joint_pnl_bps_v1"].sum() / n_total
                - df["joint_pnl_bps_v1"].mean()
            ),
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7-dir", type=str, default=str(DEFAULT_PHASE7))
    parser.add_argument("--out", type=str,
                        default="/home/andre2/GX1_DATA/reports/v9_validation_snapshot_2026q2.json",
                        help="Where to write JSON snapshot of all numbers.")
    args = parser.parse_args()

    phase7_dir = Path(args.phase7_dir).expanduser().resolve()
    print(f"Analysing {phase7_dir}/per_candidate_joint_eval_v1.csv")
    out = analyse(phase7_dir)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"Snapshot written to {out_path}")
    print(f"\nKey numbers:")
    print(f"  Overall mean PnL: {out['overall_mean_pnl_bps']:.2f} bps")
    print(f"  EXIT_IQL_SIGNAL mean: {out['exit_reason']['EXIT_IQL_SIGNAL']['mean_pnl_bps']:.2f} bps "
          f"(n={out['exit_reason']['EXIT_IQL_SIGNAL']['count']})")
    print(f"  FORCED_TERMINAL mean: {out['exit_reason']['FORCED_TERMINAL']['mean_pnl_bps']:.2f} bps "
          f"(n={out['exit_reason']['FORCED_TERMINAL']['count']})")
    print(f"  EXIT_IQL loss rate: {out['exit_reason']['EXIT_IQL_SIGNAL']['loss_rate']*100:.1f}%")
    print(f"  FORCED loss rate: {out['exit_reason']['FORCED_TERMINAL']['loss_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
