#!/usr/bin/env python3
"""Phase B Stage 2b — Entry-IQL Q-targets per-candidate parquet (v3+ chain).

Twin to `distill_v1_compute_exit_iql_q_targets.py`. Runs Entry-IQL R_NET_REAL
FOLD_1 over the forward-outcome dataset and writes Q-values (q_skip, q_long,
q_short, advantage_over_skip) per CANDIDATE to per-week parquets. These are
the supervision targets for V10-distillation in Phase B Stage 3a.

Input:
  CANDIDATE_FORWARD_OUTCOME_*/per_week/forward_outcomes_*.parquet
    (output of materialize_build_candidate_forward_outcome_dataset_v1.py)

Output:
  IQL_ENTRY_Q_TARGETS_<chain_tag>_<ts>_LOCK/
    per_week/entry_iql_q_targets_<week_name>.parquet  cols:
      candidate_uid, decision_ts_utc,
      iql_entry_q_skip_v1, iql_entry_q_long_v1, iql_entry_q_short_v1,
      iql_entry_advantage_over_skip_v1, iql_entry_action_v1
    summary_v1.json

Throughput estimate: ~50K candidates / sec on CUDA → ~1-2 min for 60K
candidates (full 334-week dataset).

Usage:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -u \\
      gx1/scripts/distill_v1_compute_entry_iql_q_targets.py \\
      --forward-outcome-lock <DIR> \\
      --entry-iql-lock <DIR> \\
      [--variant R_NET_REAL] [--fold-id FOLD_1] [--out-root <DIR>]
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/andre2/src/GX1_ENGINE")
sys.path.insert(0, str(REPO))

from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter


ACTION = "DISTILL_V1_COMPUTE_ENTRY_IQL_Q_TARGETS"

DEFAULT_FORWARD_OUTCOME_LOCK = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "CANDIDATE_FORWARD_OUTCOME_V3PLUS_PORTFOLIO_PLUS5_20260521T110559Z_LOCK"
)
# Updated 2026-05-21: default now points at canonical PLUS5 5-seed ensemble
# (best seed 1337). For ensemble averaging, pass all 5 via --entry-iql-locks.
DEFAULT_ENTRY_IQL_LOCK = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_ENTRY_IQL_V3PLUS_PORTFOLIO_PLUS5_SEED1337_20260521T111046Z"
)
REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")

OUTPUT_COLS = [
    "candidate_uid", "decision_ts_utc",
    "iql_entry_q_skip_v1", "iql_entry_q_long_v1", "iql_entry_q_short_v1",
    "iql_entry_advantage_over_skip_v1", "iql_entry_action_v1",
]


def process_week(
    parquet_path: Path,
    adapters: list[EntryIQLV2Adapter],
    out_path: Path,
) -> dict:
    """Forward-pass Entry-IQL (single or ensemble-averaged) over one week."""
    week_name = parquet_path.stem.removeprefix("forward_outcomes_")
    df = pd.read_parquet(parquet_path)
    n_raw = len(df)
    if n_raw == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=OUTPUT_COLS).to_parquet(out_path, index=False)
        return {"week": week_name, "n_raw": 0, "skipped": True}

    BATCH = 8_192
    q_skip_parts, q_long_parts, q_short_parts, action_parts = [], [], [], []
    for start in range(0, n_raw, BATCH):
        chunk = df.iloc[start:start + BATCH]
        candidates = chunk.to_dict("records")
        # Ensemble: forward through each adapter and average q_per_action
        per_adapter_q = []
        for a in adapters:
            recs = a.predict(candidates)
            per_adapter_q.append(np.array([r.q_per_action_v1 for r in recs], dtype=np.float32))
        del candidates
        qpa = np.mean(np.stack(per_adapter_q, axis=0), axis=0)  # (batch, n_actions)
        q_skip_parts.append(qpa[:, 0])
        q_long_parts.append(qpa[:, 1])
        q_short_parts.append(qpa[:, 2])
        action_parts.append(np.argmax(qpa, axis=1).astype(np.int8))

    q_skip = np.concatenate(q_skip_parts)
    q_long = np.concatenate(q_long_parts)
    q_short = np.concatenate(q_short_parts)
    action = np.concatenate(action_parts)
    # advantage = max(q_long, q_short) - q_skip (matches live runtime semantics)
    advantage = np.maximum(q_long, q_short) - q_skip

    out_df = pd.DataFrame({
        "candidate_uid": df.get("candidate_uid", pd.Series([None] * n_raw)).values,
        "decision_ts_utc": df.get("decision_ts_utc", pd.Series([None] * n_raw)).values,
        "iql_entry_q_skip_v1": q_skip,
        "iql_entry_q_long_v1": q_long,
        "iql_entry_q_short_v1": q_short,
        "iql_entry_advantage_over_skip_v1": advantage,
        "iql_entry_action_v1": action,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    return {
        "week": week_name,
        "n_raw": n_raw,
        "q_skip_mean": float(q_skip.mean()),
        "q_long_mean": float(q_long.mean()),
        "q_short_mean": float(q_short.mean()),
        "advantage_mean": float(advantage.mean()),
        "action_dist": {int(a): int((action == a).sum()) for a in np.unique(action)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-outcome-lock", default=str(DEFAULT_FORWARD_OUTCOME_LOCK))
    ap.add_argument("--entry-iql-lock",       default=str(DEFAULT_ENTRY_IQL_LOCK),
                    help="Single bundle dir. Ignored when --entry-iql-locks is given.")
    ap.add_argument("--entry-iql-locks",      default=None,
                    help="Comma-separated bundle dirs for ensemble averaging "
                         "(e.g. PLUS5 5-seed). Q-values are averaged across all.")
    ap.add_argument("--variant",              default="R_NET_REAL")
    ap.add_argument("--fold-id",              default="FOLD_1")
    ap.add_argument("--out-root",             default=None,
                    help="Default: REPORTS_ROOT/IQL_ENTRY_Q_TARGETS_V3PLUS_<ts>_LOCK")
    ap.add_argument("--start-week", type=int, default=0)
    ap.add_argument("--end-week",   type=int, default=None)
    args = ap.parse_args()

    fwd_lock  = Path(args.forward_outcome_lock).resolve()
    if args.entry_iql_locks:
        iql_locks = [Path(p.strip()).resolve() for p in args.entry_iql_locks.split(",") if p.strip()]
    else:
        iql_locks = [Path(args.entry_iql_lock).resolve()]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root or
                    REPORTS_ROOT / f"IQL_ENTRY_Q_TARGETS_V3PLUS_{ts}_LOCK").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "per_week").mkdir(exist_ok=True)

    print(f"[STAGE2b] forward-outcome = {fwd_lock}")
    print(f"[STAGE2b] entry-iql       = {len(iql_locks)} bundle(s)")
    for d in iql_locks:
        print(f"                          - {d.name}")
    print(f"[STAGE2b] out_root        = {out_root}")

    print(f"\n[1/2] Loading {len(iql_locks)} Entry-IQL adapter(s)...")
    adapters = []
    for d in iql_locks:
        a = EntryIQLV2Adapter.load(
            artifact_root=d, variant=args.variant,
            fold_id=args.fold_id, prefer_cuda=True,
        )
        adapters.append(a)
    print(f"  features={len(adapters[0].feature_names)}  device={adapters[0].model.device}")
    # Sanity: all adapters must share the same feature_names
    for i, a in enumerate(adapters[1:], 1):
        if a.feature_names != adapters[0].feature_names:
            raise RuntimeError(
                f"ensemble feature_names mismatch at bundle {i}: "
                f"{len(adapters[0].feature_names)} vs {len(a.feature_names)}"
            )

    print(f"\n[2/2] Streaming per-week Q-target compute...")
    week_files = sorted((fwd_lock / "per_week").glob("forward_outcomes_*.parquet"))
    start, end = args.start_week, args.end_week or len(week_files)
    print(f"  weeks total = {len(week_files)}  range = [{start}, {end})")

    summaries = []
    t0 = time.time()
    total_rows = 0
    for w_idx, parquet_path in enumerate(week_files):
        if w_idx < start or w_idx >= end:
            continue
        week_name = parquet_path.stem.removeprefix("forward_outcomes_")
        out_path = out_root / "per_week" / f"entry_iql_q_targets_{week_name}.parquet"
        if out_path.exists():
            print(f"  [{w_idx+1}/{end}] {week_name}  skipped (exists)", flush=True)
            continue

        try:
            s = process_week(parquet_path, adapters, out_path)
            summaries.append(s)
            total_rows += s.get("n_raw", 0)
            elapsed = time.time() - t0
            rate = total_rows / max(1, elapsed)
            print(f"  [{w_idx+1}/{end}] {week_name}  n={s.get('n_raw',0):>4}  "
                  f"adv_mean={s.get('advantage_mean',0):+.2f}  "
                  f"({rate:>5.0f}/s)", flush=True)
        except Exception as exc:
            print(f"  [{w_idx+1}/{end}] {week_name}  ERROR: {exc}", flush=True)
            summaries.append({"week": week_name, "error": str(exc)})
        gc.collect()

    summary = {
        "action_v1": ACTION,
        "ran_at_utc_v1": datetime.now(timezone.utc).isoformat(),
        "forward_outcome_lock_v1": str(fwd_lock),
        "entry_iql_locks_v1":      [str(d) for d in iql_locks],
        "ensemble_size_v1":        len(iql_locks),
        "variant_v1": args.variant, "fold_id_v1": args.fold_id,
        "n_weeks_processed": len(summaries),
        "n_candidates_total": total_rows,
        "wall_seconds":       time.time() - t0,
        "per_week":           summaries,
    }
    (out_root / "summary_v1.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] {total_rows:,} candidates / {len(summaries)} weeks "
          f"in {summary['wall_seconds']:.0f}s")
    print(f"       → {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
