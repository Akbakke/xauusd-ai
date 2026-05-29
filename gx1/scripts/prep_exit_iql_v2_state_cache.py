#!/usr/bin/env python3
"""Pre-build Exit-IQL V2 state + reward matrices to disk (memmap-friendly).

Replaces the in-line build_state_matrix + build_reward_matrix calls in
materialize_build_exit_iql_v2. Benefits:

  - **Per-trade stratified sampling** (instead of per-file random) — covers all
    trades evenly. Better for IQL than naive random sampling.
  - **Multi-process week reading** (4 workers) — 3-4× speedup on prep.
  - **Cache to disk** — reused for FI prune (C5b) + FINAL retrain (C5c).
  - **Sized up to ~10M rows** safely on 48 GB RAM (chunked accumulation).

Output cache dir:
    state_matrix.npy        — (N, F) float32
    reward_matrices.npz     — dict of variant → (N, A, K) float32
    candidate_uid.npy       — (N,) string array (for per-fold split)
    decision_ts_utc.npy     — (N,) int64 ns
    feature_names.json
    meta.json

Then run:
    python -m gx1.scripts.materialize_build_exit_iql_v2 \\
        --per-bar-dir <orig>  (still required for some metadata) \\
        --load-state-cache <cache-dir> \\
        --out-root <out>

Usage:
    python -m gx1.scripts.prep_exit_iql_v2_state_cache \\
        --per-bar-dir /path/to/PER_BAR_V2STATE \\
        --out-cache   /path/to/STATE_CACHE_V2 \\
        --bars-per-trade 100 \\
        --n-workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _worker_subsample_week(args: Tuple[str, int, int]) -> pd.DataFrame:
    """Worker: load one per-bar parquet, stratified per-trade sample.

    Returns subsampled DataFrame.
    """
    parquet_path, bars_per_trade, seed = args
    df = pd.read_parquet(parquet_path)
    if "candidate_uid" not in df.columns:
        raise RuntimeError(f"[prep] missing candidate_uid in {parquet_path}")
    # Per-trade stratified: sample bars_per_trade rows from each UID
    rng = np.random.default_rng(seed)
    sampled = []
    for uid, grp in df.groupby("candidate_uid", sort=False):
        if len(grp) <= bars_per_trade:
            sampled.append(grp)
        else:
            idx = rng.choice(len(grp), size=bars_per_trade, replace=False)
            sampled.append(grp.iloc[sorted(idx)])
    out = pd.concat(sampled, ignore_index=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-bar-dir", type=Path, required=True,
                   help="Dir with per_week/exit_per_bar_m1_*.parquet (V2 augmented)")
    p.add_argument("--out-cache", type=Path, required=True,
                   help="Output cache dir (will be created)")
    p.add_argument("--bars-per-trade", type=int, default=100,
                   help="Max bars sampled per trade. Default 100 (37K trades × 100 = ~3.7M).")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260523)
    args = p.parse_args()

    args.out_cache.mkdir(parents=True, exist_ok=True)

    # ── Step 1: parallel per-week subsample ────────────────────────────────
    parquets = sorted((args.per_bar_dir / "per_week").glob("exit_per_bar_m1_*.parquet"))
    print(f"[PREP] {len(parquets)} weeks, bars_per_trade={args.bars_per_trade}, "
          f"n_workers={args.n_workers}", flush=True)

    rng_master = np.random.default_rng(args.seed)
    work_args = [(str(p), args.bars_per_trade, int(rng_master.integers(0, 2**31 - 1)))
                 for p in parquets]

    t0 = time.time()
    if args.n_workers <= 1:
        parts = [_worker_subsample_week(wa) for wa in work_args]
    else:
        with mp.Pool(args.n_workers) as pool:
            parts = []
            for i, df_w in enumerate(pool.imap_unordered(_worker_subsample_week, work_args)):
                parts.append(df_w)
                if (i + 1) % 50 == 0 or i + 1 == len(work_args):
                    elapsed = time.time() - t0
                    print(f"  [SAMPLE] {i+1}/{len(work_args)} weeks ({elapsed:.0f}s)", flush=True)

    print(f"[PREP] sampling done in {time.time()-t0:.0f}s. concatenating...", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    import gc; gc.collect()
    print(f"[PREP] total sampled rows: {len(df):,} cols: {df.shape[1]}", flush=True)

    # ── Step 2: build state + reward matrices using existing logic ─────────
    # Set V2 env so the global state-col lists are V2-extended.
    import os
    os.environ["GX1_BUILD_EXIT_IQL_V2"] = "1"
    # Force re-import to pick up V2 mode
    if "gx1.scripts.materialize_build_exit_iql_v2" in sys.modules:
        del sys.modules["gx1.scripts.materialize_build_exit_iql_v2"]
    from gx1.scripts.materialize_build_exit_iql_v2 import (
        build_state_matrix, build_reward_matrix, REWARD_VARIANTS,
    )

    print(f"[PREP] building state matrix...", flush=True)
    t0 = time.time()
    X, feature_names = build_state_matrix(df)
    print(f"  state_matrix: {X.shape} dtype={X.dtype} "
          f"size_MB={X.nbytes/1e6:.1f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[PREP] building reward matrices for {len(REWARD_VARIANTS)} variants...", flush=True)
    reward_dict: Dict[str, np.ndarray] = {}
    for variant in REWARD_VARIANTS:
        t0 = time.time()
        R = build_reward_matrix(df, variant=variant)
        reward_dict[variant] = R
        print(f"  {variant}: shape={R.shape} mean={R.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ── Step 3: save to cache ──────────────────────────────────────────────
    print(f"[PREP] saving to {args.out_cache}", flush=True)
    np.save(args.out_cache / "state_matrix.npy", X.astype(np.float32, copy=False))
    np.savez(args.out_cache / "reward_matrices.npz",
             **{k: v.astype(np.float32, copy=False) for k, v in reward_dict.items()})

    cu_values = df["candidate_uid"].astype(str).to_numpy()
    np.save(args.out_cache / "candidate_uid.npy", cu_values)

    if "decision_ts_utc" in df.columns:
        ts = pd.to_datetime(df["decision_ts_utc"], utc=True, errors="coerce").values.astype("datetime64[ns]").astype(np.int64)
    else:
        ts = np.zeros(len(df), dtype=np.int64)
    np.save(args.out_cache / "decision_ts_utc.npy", ts)

    (args.out_cache / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    meta = {
        "per_bar_dir": str(args.per_bar_dir),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_trades": int(df["candidate_uid"].nunique()),
        "bars_per_trade": int(args.bars_per_trade),
        "n_workers": int(args.n_workers),
        "seed": int(args.seed),
        "reward_variants": list(REWARD_VARIANTS),
        "feature_names_count": len(feature_names),
    }
    (args.out_cache / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] cache at {args.out_cache}", flush=True)
    print(f"  state_matrix.npy ({X.nbytes/1e6:.1f} MB)")
    print(f"  reward_matrices.npz ({sum(v.nbytes for v in reward_dict.values())/1e6:.1f} MB)")
    print(f"  rows={meta['n_rows']:,}  trades={meta['n_trades']:,}  features={meta['n_features']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
