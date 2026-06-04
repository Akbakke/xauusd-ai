#!/usr/bin/env python3
"""Augment Exit-IQL per-bar dataset with V3-tracking running-stat features.

Cement Exit-IQL bundle was trained on a dataset that ONLY had V3 prob outputs
(v3_v8_should_exit_prob etc.) per bar, but MISSING the 7 running-stat features
that runtime computes via trade_state.update_v3() + .build_v3_tracking_features().

This script:
  1. Reads the v2_m1_costfix_heads dataset (per-week parquets)
  2. Groups by trade_uid
  3. For each trade, walks bars in order + replays V3 updates through a
     TradeState instance (canonical runtime code) to derive the 7 tracking cols
  4. Writes a NEW per-bar dataset with these cols ADDED

Designed-not-forked:
  - Imports TradeState from gx1.execution.v12_trade_state and uses
    update_v3() + build_v3_tracking_features() as one-truth — no replication
    of the running-stat math.

Usage:
  python -m gx1.scripts.augment_exit_iql_dataset_with_v3_tracking \\
      --input-dir /home/andre2/GX1_DATA/reports/truth_e2e_sanity/EXIT_IQL_PER_BAR_DATASET_V2_M1_COSTFIX_HEADS_20260528T093243Z \\
      --output-dir /home/andre2/GX1_DATA/reports/truth_e2e_sanity/EXIT_IQL_PER_BAR_DATASET_V12_M1_COSTFIX_HEADS_V3TRACKING_20260602T100000Z
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
from gx1.execution.v12_trade_state import TradeState


V3_TRACKING_COLS = [
    "v3_should_exit_decision_v1",
    "v3_decision_confidence_v1",
    "v3_max_prob_in_trade_v1",
    "v3_consecutive_exits_v1",
    "v3_signal_acceleration_v1",
    "v3_total_exit_decisions_v1",
    "v3_max_consecutive_exits_v1",
]

# V10 entry-snapshot: V10's predictions at the time of trade entry, constant
# across all bars within a trade. Source = candidate fwd-outcome dataset
# (one row per candidate). Joined by candidate_uid.
# Mapping: output_col_name → source_col_name in candidate fwd-outcome dataset.
V10_SNAPSHOT_MAPPING = {
    "v10_p_long_at_entry_v1": "p_long",
    "v10_p_short_at_entry_v1": "p_short",
    "v10_path_quality_at_entry_v1": "path_quality_pred",
    "v10_mfe_pred_at_entry_v1": "mfe_first_n_pred",
    "v10_tf_agreement_at_entry_v1": "tf_agreement_pred",
    "v10_path_quality_std_at_entry_v1": "path_quality_std",
    "v10_position_size_at_entry_v1": "position_size_pred",
    "v10_hold_horizon_at_entry_v1": "hold_horizon_pred",
}


def load_v10_snapshot_lookup(candidate_fwd_dir: Path) -> pd.DataFrame:
    """Read all candidate fwd-outcome weekly parquets, return a single DataFrame
    keyed by candidate_uid with V10-snapshot cols renamed to their per-bar names.
    """
    dfs = []
    for p in sorted(candidate_fwd_dir.glob("*.parquet")):
        if "stats" in p.name:
            continue
        cols_needed = ["candidate_uid"] + list(V10_SNAPSHOT_MAPPING.values())
        df_p = pd.read_parquet(p, columns=[c for c in cols_needed if True])
        dfs.append(df_p)
    if not dfs:
        raise RuntimeError(f"no candidate fwd-outcome parquets in {candidate_fwd_dir}")
    big = pd.concat(dfs, ignore_index=True)
    # Rename to target col names
    rename = {src: dst for dst, src in V10_SNAPSHOT_MAPPING.items()}
    big = big.rename(columns=rename)
    big = big.drop_duplicates(subset=["candidate_uid"], keep="first")
    return big.set_index("candidate_uid")


def augment_per_trade(grp: pd.DataFrame, v3_prob_col: str) -> pd.DataFrame:
    """For a single trade's bars (sorted by bars_in_trade_v1), replay V3 updates
    through TradeState to derive the 7 V3-tracking features per bar.
    Note: dataset uses 'bars_in_trade_v1' (not 'bar_in_trade') and groups
    by 'candidate_uid' (no 'trade_uid' column).
    """
    sort_col = "bars_in_trade_v1" if "bars_in_trade_v1" in grp.columns else "bar_idx_v1"
    grp = grp.sort_values(sort_col).reset_index(drop=True)
    side = str(grp["side"].iloc[0]) if "side" in grp.columns else "long"
    # Minimal TradeState init — we only need v3 fields populated for update_v3()
    # to work. entry_bid/ask/ts placeholders since we're only computing v3 stats.
    ts = TradeState(
        side=side,
        entry_bid=1.0,
        entry_ask=1.0,
        entry_ts=pd.Timestamp("2020-01-01", tz="UTC"),
        entry_spread_bps=0.0,
        v10_snapshot={},
    )
    out_cols = {c: np.zeros(len(grp), dtype=np.float32) for c in V3_TRACKING_COLS}
    for i in range(len(grp)):
        prob = float(grp[v3_prob_col].iloc[i]) if v3_prob_col in grp.columns else 0.0
        if pd.isna(prob):
            prob = 0.0
        # Canonical update via TradeState
        ts.update_v3({"v3_v8_should_exit_prob": prob})
        feats = ts.build_v3_tracking_features()
        for col in V3_TRACKING_COLS:
            out_cols[col][i] = feats.get(col, 0.0)
    aug = grp.copy()
    for col in V3_TRACKING_COLS:
        aug[col] = out_cols[col]
    return aug


def process_file(in_path: Path, out_path: Path, v3_prob_col: str = "v3_v8_should_exit_prob",
                 v10_snap_lookup: pd.DataFrame | None = None) -> dict:
    """Process one weekly parquet — augment + write.
    Adds V3-tracking (7 cols) + optionally V10-snapshot (8 cols) via join.
    """
    df = pd.read_parquet(in_path)
    if v3_prob_col not in df.columns:
        return {"file": in_path.name, "status": "skipped", "reason": f"missing {v3_prob_col}"}
    group_col = "trade_uid" if "trade_uid" in df.columns else "candidate_uid"
    if group_col not in df.columns:
        return {"file": in_path.name, "status": "skipped", "reason": "no trade_uid/candidate_uid"}
    aug_groups = []
    for uid, grp in df.groupby(group_col, sort=False):
        aug_groups.append(augment_per_trade(grp, v3_prob_col))
    out = pd.concat(aug_groups, ignore_index=True)
    assert len(out) == len(df), f"row count drift: in={len(df)} out={len(out)}"
    # V10-snapshot join (constant within trade)
    n_v10_added = 0
    if v10_snap_lookup is not None and "candidate_uid" in out.columns:
        join_cols = list(V10_SNAPSHOT_MAPPING.keys())
        snap_subset = v10_snap_lookup[[c for c in join_cols if c in v10_snap_lookup.columns]]
        out = out.merge(snap_subset, left_on="candidate_uid", right_index=True, how="left")
        # NaN-fill for trades not in candidate lookup (defensive — shouldn't happen)
        for c in join_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype("float32")
                n_v10_added += 1
    out.to_parquet(out_path, index=False)
    return {
        "file": in_path.name,
        "status": "ok",
        "rows": len(out),
        "n_trades": df[group_col].nunique(),
        "n_v3_tracking_cols": len(V3_TRACKING_COLS),
        "n_v10_snapshot_cols": n_v10_added,
    }


def _process_one(args_tuple):
    """Worker: process a single (in_path, out_path, v3_prob_col, v10_snap_lookup)
    tuple. Top-level function so multiprocessing can pickle it.
    """
    in_path, out_path, v3_prob_col, v10_snap_lookup = args_tuple
    if out_path.exists():
        return {"file": in_path.name, "status": "skipped_existing"}
    return process_file(in_path, out_path, v3_prob_col, v10_snap_lookup)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--v3-prob-col", type=str, default="v3_v8_should_exit_prob")
    p.add_argument("--candidate-fwd-dir", type=Path, default=None,
                   help="Optional candidate fwd-outcome dir for V10-snapshot join")
    p.add_argument("--workers", type=int, default=12,
                   help="Parallel workers (multiprocessing). 0 = sequential.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_week").mkdir(parents=True, exist_ok=True)

    in_per_week = args.input_dir / "per_week"
    if not in_per_week.exists():
        print(f"FATAL: input per_week dir missing: {in_per_week}")
        return 1
    files = sorted(in_per_week.glob("*.parquet"))
    print(f"Found {len(files)} weekly parquets")

    # V10-snapshot lookup (loaded ONCE, shared via fork on Linux multiprocessing)
    v10_snap_lookup = None
    if args.candidate_fwd_dir is not None:
        cand_dir = args.candidate_fwd_dir / "per_week" if (args.candidate_fwd_dir / "per_week").exists() else args.candidate_fwd_dir
        print(f"Loading V10-snapshot lookup from {cand_dir}")
        v10_snap_lookup = load_v10_snapshot_lookup(cand_dir)
        print(f"  V10 snapshot lookup: {len(v10_snap_lookup):,} candidates × {len(v10_snap_lookup.columns)} cols")

    tasks = [(f, args.output_dir / "per_week" / f.name, args.v3_prob_col, v10_snap_lookup) for f in files]
    results = []
    if args.workers > 0:
        import multiprocessing as mp
        print(f"Running with {args.workers} parallel workers")
        with mp.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_process_one, tasks)):
                results.append(r)
                if (i + 1) % 25 == 0 or i == len(tasks) - 1:
                    ok = sum(1 for x in results if x.get("status") == "ok")
                    sk = sum(1 for x in results if x.get("status") == "skipped_existing")
                    print(f"  [{i+1}/{len(tasks)}]  ok={ok}  skipped_existing={sk}")
    else:
        for i, task in enumerate(tasks):
            r = _process_one(task)
            results.append(r)
            if (i + 1) % 25 == 0 or i == len(tasks) - 1:
                ok = sum(1 for x in results if x.get("status") == "ok")
                print(f"  [{i+1}/{len(tasks)}]  ok={ok}")

    # Manifest
    manifest = {
        "action_v1": "AUGMENT_EXIT_IQL_DATASET_WITH_V3_TRACKING",
        "input_dataset": str(args.input_dir),
        "v3_prob_col": args.v3_prob_col,
        "n_files_processed": sum(1 for r in results if r.get("status") == "ok"),
        "n_files_skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "new_cols": V3_TRACKING_COLS,
    }
    (args.output_dir / "manifest_v1.json").write_text(json.dumps(manifest, indent=2))
    # FAIL-CLOSED (2026-06-03 audit): a 'skipped' week (missing v3_prob / uid) writes NO
    # output parquet -> that week silently vanishes from the Exit-IQL training set (row-loss)
    # while main() still returned 0 = "success". Refuse to report success if any week dropped.
    n_skipped = manifest["n_files_skipped"]
    if n_skipped > 0:
        raise RuntimeError(
            f"[AUGMENT_V3_TRACKING] {n_skipped} week(s) SKIPPED (missing {args.v3_prob_col} "
            f"or uid) — they are silently absent from the output. Fix the inputs; refusing "
            f"to produce a silently-truncated dataset.")
    # Copy summary from input
    in_summary = args.input_dir / "summary_v1.json"
    if in_summary.exists():
        s = json.loads(in_summary.read_text())
        s["augmented_with_v3_tracking_v1"] = True
        s["v3_tracking_cols_v1"] = V3_TRACKING_COLS
        (args.output_dir / "summary_v1.json").write_text(json.dumps(s, indent=2, default=str))

    print(f"\n✓ Done. Output: {args.output_dir}")
    print(f"  Manifest: {args.output_dir / 'manifest_v1.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
