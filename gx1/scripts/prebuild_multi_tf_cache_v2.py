#!/usr/bin/env python3
"""Pre-build V2 multi-TF feature cache to disk-backed numpy memmap.

Saves ~84s per trainer init (V2 build is single-threaded resampling of 456K M5
bars across 5 TFs). After running this once, subsequent V10/V3 trainer launches
load the cache directly (lazy mmap, ~instant).

Output layout:
    <cache-dir>/
        manifest.json          {tf -> {n_bars, feature_count, feats_npy, ts_npy}}
        M5_feats.npy           (N_m5, 25) float32 memmap
        M5_ts.npy              (N_m5,) int64 memmap (ns since epoch)
        M15_feats.npy / _ts.npy
        H1_feats.npy / _ts.npy
        H4_feats.npy / _ts.npy
        D1_feats.npy / _ts.npy

The trainer needs a small loader-shim that reconstructs the DataFrame .attrs
("ts_int64", "feats_np") from these memmaps. See `gx1.features.htf_features`.

Usage:
    python -m gx1.scripts.prebuild_multi_tf_cache_v2 \\
        --m5-prebuilt /home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_2020_2026.parquet \\
        --out-dir /home/andre2/GX1_DATA/data/data/prebuilt/MULTI_TF_V2_CACHE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m5-prebuilt", type=Path, required=True,
                   help="Path to canonical_v3 M5 OHLCV parquet")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for cache files")
    args = p.parse_args()

    sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
    from gx1.features.htf_features import (
        build_multi_tf_per_bar_features_v2,
        MULTI_TF_FEATURE_COUNT_V2,
    )
    import pyarrow.parquet as pq

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CACHE_V2] m5_prebuilt: {args.m5_prebuilt}")
    print(f"[CACHE_V2] out_dir: {args.out_dir}")

    cols = ["time", "open", "high", "low", "close"]
    if "volume" in pq.ParquetFile(args.m5_prebuilt).schema_arrow.names:
        cols.append("volume")
    print(f"[LOAD] {args.m5_prebuilt.name} cols={cols}")
    m5 = pd.read_parquet(args.m5_prebuilt, columns=cols)
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.set_index("time").sort_index()
    for c in ("open", "high", "low", "close"):
        m5[c] = m5[c].astype(np.float32)
    if "volume" in m5.columns:
        m5["volume"] = m5["volume"].astype(np.float32)
    print(f"[LOAD] {len(m5):,} M5 bars, range {m5.index[0]} → {m5.index[-1]}")

    print(f"[BUILD] V2 multi-TF features (5 TFs × {MULTI_TF_FEATURE_COUNT_V2} feats)...")
    feats = build_multi_tf_per_bar_features_v2(m5)

    manifest = {
        "feature_count": int(MULTI_TF_FEATURE_COUNT_V2),
        "m5_prebuilt_source": str(args.m5_prebuilt.resolve()),
        "tfs": {},
    }
    for tf, df in feats.items():
        feats_np = df.attrs["feats_np"]   # (N, 25) float32
        ts_int64 = df.attrs["ts_int64"]   # (N,) int64 ns
        feats_path = args.out_dir / f"{tf}_feats.npy"
        ts_path = args.out_dir / f"{tf}_ts.npy"
        np.save(feats_path, feats_np)
        np.save(ts_path, ts_int64)
        manifest["tfs"][tf] = {
            "n_bars": int(feats_np.shape[0]),
            "feature_count": int(feats_np.shape[1]),
            "feats_npy": feats_path.name,
            "ts_npy": ts_path.name,
            "first_ts_ns": int(ts_int64[0]),
            "last_ts_ns": int(ts_int64[-1]),
        }
        print(f"  [SAVED] {tf}: {feats_np.shape} → {feats_path.name} "
              f"+ ts.npy ({feats_path.stat().st_size/1e6:.2f} MB feats)")

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[DONE] manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
