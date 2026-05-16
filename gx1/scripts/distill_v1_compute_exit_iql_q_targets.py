#!/usr/bin/env python3
"""Phase B Stage 2a — Exit-IQL Q-targets per-bar parquet (V12.2).

Runs Exit-IQL R_V12_FOLD_1.pt over the V12.2 V3TRACKED_SOLO per-bar dataset
and writes Q-values (q_hold, q_exit, q_advantage) per row to per-week parquets.
These are the supervision targets for V3-distillation in Phase B Stage 3b.

Input:
  V3TRACKED_SOLO_LOCK/per_week/exit_per_bar_m1_TRUTH_MONFRI_WEEK_*.parquet

Output:
  IQL_EXIT_Q_TARGETS_V12_2_<ts>_LOCK/
    per_week/exit_iql_q_targets_<week_name>.parquet  cols:
      candidate_uid, bar_idx_v1, bar_ts_ns_v1, side_v1,
      iql_exit_q_hold_v1, iql_exit_q_exit_v1, iql_exit_q_advantage_v1,
      iql_exit_action_v1  (0=HOLD, 1=EXIT_NOW per model argmax)
    summary_v1.json   (rows per week + Q-stats)

Notes:
  - chunk0-missing weeks: fill with 0.0 (NOT NaN) to match IQL training dist.
  - is_never_fire/forced_terminal rows are SKIPPED (model wouldn't decide there).
  - CUDA used if available; batch size 4096.

Throughput estimate: ~5K rows/sec on CUDA → ~4-5h for full 80M rows.

Usage:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -u \\
      gx1/scripts/distill_v1_compute_exit_iql_q_targets.py \\
      [--start-week 0] [--end-week 333] [--out-root <DIR>]
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/andre2/src/GX1_ENGINE")
sys.path.insert(0, str(REPO))

from gx1.runtime.exit_iql_v2_adapter import ExitIQLV2Adapter
from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe
from gx1.scripts import materialize_build_exit_iql_v2 as v2_train
from gx1.scripts import materialize_build_exit_iql_v3_m1 as v3_m1


ACTION = "DISTILL_V1_COMPUTE_EXIT_IQL_Q_TARGETS"

DEFAULT_V3TRACKED_LOCK = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_2_20260514T161504Z_R4_LOCK_"
    "V3TRACKED_SOLO_20260515T004836Z_LOCK"
)
DEFAULT_EXIT_IQL_LOCK = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_2_20260514T161504Z_R4_LOCK_"
    "V3TRACKED_SOLO_20260515T004836Z_LOCK_TRAINED_20260515T122939Z_LOCK"
)
REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")

OUTPUT_COLS = [
    "candidate_uid", "bar_idx_v1", "bar_ts_ns_v1", "side_v1",
    "iql_exit_q_hold_v1", "iql_exit_q_exit_v1", "iql_exit_q_advantage_v1",
    "iql_exit_action_v1",
]


def process_week(
    parquet_path: Path,
    adapter: ExitIQLV2Adapter,
    canonical_suf: pd.DataFrame,
    out_path: Path,
) -> dict:
    """Forward-pass IQL over one week's per-bar parquet, write Q-targets."""
    week_name = parquet_path.stem.removeprefix("exit_per_bar_m1_")

    df = pd.read_parquet(parquet_path)
    n_raw = len(df)
    # Skip rows where IQL would not actually decide (forced_terminal / never_fire)
    df = df[(df.get("is_never_fire_v1", 0) == 0) &
            (df.get("forced_terminal_v1", 0) == 0)]
    n_valid = len(df)
    if n_valid == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=OUTPUT_COLS).to_parquet(out_path, index=False)
        return {"week": week_name, "n_raw": n_raw, "n_valid": 0,
                "chunk0_present": None, "skipped": True}

    # Merge chunk0 (if present) + canonical to build 204-feature bar_state
    week_dir = REPORTS_ROOT / week_name
    chunk0 = fwd_pipe._load_chunk0_features(week_dir)
    chunk0_present = chunk0 is not None
    chunk0_suf = v3_m1._suffix_chunk0(chunk0) if chunk0_present else None
    if chunk0_suf is not None:
        df = v3_m1._merge_asof_features(df, chunk0_suf)
    else:
        # IQL was trained with 0.0-fill for missing chunk0 cols. Match that.
        for col in v2_train.NUMERIC_STATE_COLS_CHUNK0:
            if col not in df.columns:
                df[col] = 0.0
    df = v3_m1._merge_asof_features(df, canonical_suf)

    # side_v1 from existing 'side' col (long/short)
    if "side_v1" not in df.columns:
        df["side_v1"] = df["side"].astype(str)

    # Forward-pass IQL adapter in batches to keep RAM under 15GB host budget.
    # to_dict("records") on a 460K×246 frame allocates ~5GB in Python objects;
    # do it per-chunk and discard immediately.
    BATCH = 50_000
    q_hold_parts, q_exit_parts, q_adv_parts, action_parts = [], [], [], []
    for start in range(0, n_valid, BATCH):
        chunk = df.iloc[start:start + BATCH]
        bar_states = chunk.to_dict("records")
        recs = adapter.predict(bar_states)
        del bar_states
        qpa = np.array([r.q_per_action_v1 for r in recs], dtype=np.float32)
        q_hold_parts.append(qpa[:, 0])
        q_exit_parts.append(qpa[:, 1])
        q_adv_parts.append(qpa[:, 1] - qpa[:, 0])
        action_parts.append(np.array([r.action_id_v1 for r in recs], dtype=np.int32))
        del recs, qpa
    q_hold = np.concatenate(q_hold_parts); del q_hold_parts
    q_exit = np.concatenate(q_exit_parts); del q_exit_parts
    q_adv = np.concatenate(q_adv_parts);   del q_adv_parts
    actions = np.concatenate(action_parts); del action_parts

    # Write Q-targets (slim — just identifiers + Q-values)
    out_df = pd.DataFrame({
        "candidate_uid": df["candidate_uid"].astype(str).to_numpy(),
        "bar_idx_v1":    df["bar_idx_v1"].to_numpy(),
        "bar_ts_ns_v1":  df["bar_ts_ns_v1"].to_numpy(),
        "side_v1":       df["side_v1"].astype(str).to_numpy(),
        "iql_exit_q_hold_v1":      q_hold,
        "iql_exit_q_exit_v1":      q_exit,
        "iql_exit_q_advantage_v1": q_adv,
        "iql_exit_action_v1":      actions,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    return {
        "week": week_name, "n_raw": n_raw, "n_valid": n_valid,
        "chunk0_present": chunk0_present,
        "q_adv_mean": float(q_adv.mean()), "q_adv_std": float(q_adv.std()),
        "q_hold_mean": float(q_hold.mean()), "q_exit_mean": float(q_exit.mean()),
        "exit_action_frac": float((actions == 1).mean()),
        "skipped": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3tracked-lock", default=str(DEFAULT_V3TRACKED_LOCK))
    ap.add_argument("--exit-iql-lock",  default=str(DEFAULT_EXIT_IQL_LOCK))
    ap.add_argument("--variant",        default="R_V12")
    ap.add_argument("--fold-id",        default="FOLD_1")
    ap.add_argument("--out-root",       default=None,
                    help="Default: REPORTS_ROOT/IQL_EXIT_Q_TARGETS_V12_2_<ts>_LOCK")
    ap.add_argument("--start-week", type=int, default=0,
                    help="Index into sorted per_week file list")
    ap.add_argument("--end-week",   type=int, default=None,
                    help="Exclusive end index; default = all")
    args = ap.parse_args()

    v3tracked = Path(args.v3tracked_lock).resolve()
    iql_lock  = Path(args.exit_iql_lock).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root or
                    REPORTS_ROOT / f"IQL_EXIT_Q_TARGETS_V12_2_{ts}_LOCK").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_week").mkdir(exist_ok=True)

    print(f"[STAGE2a] v3tracked = {v3tracked}")
    print(f"[STAGE2a] exit-iql  = {iql_lock}")
    print(f"[STAGE2a] out_root  = {out_root}")

    # Load IQL adapter (one-time, ~5s)
    print(f"\n[1/3] Loading Exit-IQL adapter...")
    adapter = ExitIQLV2Adapter.load(
        artifact_root=iql_lock, variant=args.variant,
        fold_id=args.fold_id, prefer_cuda=True,
    )
    print(f"  features={len(adapter.feature_names)}  device={adapter.model.device}")

    # Load canonical features (resident throughout)
    print(f"\n[2/3] Loading canonical features (kept resident)...")
    canonical = fwd_pipe._load_canonical_features(v3_m1.DEFAULT_CANONICAL_FEATURES_PATH)
    canonical_suf = v3_m1._suffix_canonical(canonical)
    print(f"  canonical rows={len(canonical):,}")

    # Stream per-week
    print(f"\n[3/3] Streaming per-week Q-target compute...")
    week_files = sorted((v3tracked / "per_week").glob("exit_per_bar_m1_*.parquet"))
    start, end = args.start_week, args.end_week or len(week_files)
    print(f"  weeks total = {len(week_files)}  range = [{start}, {end})")

    summaries = []
    t0 = time.time()
    total_rows = 0
    for w_idx, parquet_path in enumerate(week_files):
        if w_idx < start or w_idx >= end:
            continue
        week_name = parquet_path.stem.removeprefix("exit_per_bar_m1_")
        out_path = out_root / "per_week" / f"exit_iql_q_targets_{week_name}.parquet"
        if out_path.exists():
            print(f"  [{w_idx+1}/{end}] {week_name}  skipped (exists)", flush=True)
            continue

        try:
            s = process_week(parquet_path, adapter, canonical_suf, out_path)
            summaries.append(s)
            total_rows += s["n_valid"]
            elapsed = time.time() - t0
            rate = total_rows / max(1, elapsed)
            eta_min = (end - w_idx - 1) * (elapsed / max(1, w_idx - start + 1)) / 60
            chunk0_flag = "✓" if s.get("chunk0_present") else "✗0fill"
            print(f"  [{w_idx+1}/{end}] {week_name}  rows={s['n_valid']:>6,}  "
                  f"chunk0={chunk0_flag}  q_adv={s.get('q_adv_mean', 0):+.2f}  "
                  f"({rate:>5.0f}/s, ETA {eta_min:.0f}min)", flush=True)
        except Exception as exc:
            print(f"  [{w_idx+1}/{end}] {week_name}  ERROR: {exc}", flush=True)
            summaries.append({"week": week_name, "error": str(exc)})

        # Cleanup: free pandas frames between weeks
        gc.collect()

    # Write summary
    summary = {
        "action_v1": ACTION,
        "ran_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "v3tracked_lock_v1": str(v3tracked),
        "exit_iql_lock_v1":  str(iql_lock),
        "variant_v1": args.variant, "fold_id_v1": args.fold_id,
        "n_weeks_processed": len(summaries),
        "n_rows_total":      total_rows,
        "wall_seconds":      time.time() - t0,
        "per_week":          summaries,
    }
    (out_root / "summary_v1.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[STAGE2a] DONE — {total_rows:,} rows over {len(summaries)} weeks")
    print(f"          summary  : {out_root}/summary_v1.json")
    print(f"          per_week : {out_root}/per_week/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
