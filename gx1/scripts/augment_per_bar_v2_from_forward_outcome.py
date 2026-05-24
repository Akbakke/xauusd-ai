#!/usr/bin/env python3
"""Join V2 multi-TF + group-A cols (per-candidate) into per-bar dataset.

The Exit-IQL V2 builder expects per-bar dataset where each row has:
  - per-bar cols (from V3-tracked dataset)
  - V2 multi-TF + group-A cols (one value per candidate, broadcast to all bars
    of that candidate)

This script reads forward-outcome V2 augmented (which has the 153 V2 cols per
candidate) and broadcasts them to every per-bar row via candidate_uid join.

Usage:
    python -m gx1.scripts.augment_per_bar_v2_from_forward_outcome \\
        --per-bar-in /path/to/PER_BAR_V3V2_TRACKED \\
        --fwd-aug    /path/to/FORWARD_OUTCOME_V10V2_AUGMENTED \\
        --out-dir    /path/to/PER_BAR_V3V2_TRACKED_V2_FULL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow.parquet as pq


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-bar-in", type=Path, required=True,
                   help="Dir with per_week/exit_per_bar_m1_*.parquet (V3-tracked)")
    p.add_argument("--fwd-aug", type=Path, required=True,
                   help="Dir with per_week/forward_outcomes_*.parquet (V10 V2 + V2 augmented)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_per_week = args.out_dir / "per_week"
    out_per_week.mkdir(parents=True, exist_ok=True)

    # Build candidate_uid → V2 cols map from ALL forward-outcome weeks.
    fwd_files = sorted((args.fwd_aug / "per_week").glob("forward_outcomes_*.parquet"))
    print(f"[AUG_PB_V2] loading {len(fwd_files)} forward-outcome weeks for V2 cols", flush=True)

    # Discover V2 cols from first parquet schema.
    # 2026-05-24 FIX: also include _canon_v1 cols (canonical_v1 features from
    # canonical_v3 prebuilt) — these were never joined into per-bar before,
    # leaving Exit-IQL with ~80 dead candidate-context features.
    sample_cols = pq.ParquetFile(fwd_files[0]).schema_arrow.names
    # 2026-05-24 Bug 2 fix: added dist_to_m15_ and dist_to_d1_ prefixes
    # 2026-05-24 PM: added _v3 suffix for dip/struct features (Exit needs them too).
    v2_cols: List[str] = [
        c for c in sample_cols
        if c.endswith("_v2") or c.endswith("_canon_v1") or c.endswith("_v3") or c.startswith((
            "is_asia_eu_overlap", "is_eu_us_overlap",
            "is_eu_only", "is_us_only", "atr_ratio_",
            "vol_pct_", "dist_to_R", "dist_to_S",
            "dist_to_m5_", "dist_to_m15_", "dist_to_h1_", "dist_to_h4_", "dist_to_d1_",
            "long_win_rate_", "long_mean_pnl_",
            "long_n_consec_losses", "long_time_since_last_close_min",
            "short_win_rate_", "short_mean_pnl_",
            "short_n_consec_losses", "short_time_since_last_close_min",
        ))
    ]
    print(f"[AUG_PB_V2] {len(v2_cols)} V2 cols to broadcast: e.g. {v2_cols[:5]}...", flush=True)

    # Load candidate_uid + V2 cols across all weeks → big map.
    t0 = time.time()
    parts = []
    for fp in fwd_files:
        parts.append(pd.read_parquet(fp, columns=["candidate_uid"] + v2_cols))
    fwd_v2 = pd.concat(parts, ignore_index=True)
    # Dedupe by candidate_uid (should already be unique but be safe)
    fwd_v2 = fwd_v2.drop_duplicates(subset=["candidate_uid"], keep="first").set_index("candidate_uid")
    print(f"[AUG_PB_V2] V2 map built: {len(fwd_v2):,} unique candidates, "
          f"{len(v2_cols)} cols ({(time.time()-t0):.1f}s)", flush=True)

    # Per-week join
    pb_files = sorted((args.per_bar_in / "per_week").glob("exit_per_bar_m1_*.parquet"))
    print(f"[AUG_PB_V2] joining V2 into {len(pb_files)} per-bar weeks", flush=True)
    n_matched_total = 0
    n_total = 0
    t0 = time.time()
    for i, pb in enumerate(pb_files):
        df = pd.read_parquet(pb)
        n_total += len(df)
        # Join by candidate_uid
        df = df.join(fwd_v2, on="candidate_uid", how="left")
        n_matched = int(df[v2_cols[0]].notna().sum())
        n_matched_total += n_matched
        df.to_parquet(out_per_week / pb.name, index=False)
        if (i + 1) % 50 == 0 or i + 1 == len(pb_files):
            elapsed = time.time() - t0
            print(f"  [JOIN] {i+1}/{len(pb_files)} weeks, matched={n_matched_total:,}/{n_total:,} "
                  f"({100*n_matched_total/max(1,n_total):.1f}%), elapsed={elapsed:.0f}s", flush=True)

    # Write summary
    summary = {
        "per_bar_in": str(args.per_bar_in),
        "fwd_aug": str(args.fwd_aug),
        "out_dir": str(args.out_dir),
        "n_per_bar_rows": int(n_total),
        "n_matched_rows": int(n_matched_total),
        "match_rate": float(n_matched_total / max(1, n_total)),
        "n_v2_cols_joined": len(v2_cols),
        "v2_cols": v2_cols,
    }
    (args.out_dir / "augment_pb_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[DONE] match: {n_matched_total:,}/{n_total:,} "
          f"({100*n_matched_total/max(1,n_total):.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
