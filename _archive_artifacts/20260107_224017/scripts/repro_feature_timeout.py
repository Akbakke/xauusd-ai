#!/usr/bin/env python3
"""
Repro script for feature building timeout issue.

Reads data_file parquet for 2025-01-06..2025-01-13, runs build_v9_runtime_features
on every 10th bar (limit 200 bars), prints perf breakdown and stops at first timeout.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from gx1.features.runtime_v9 import build_v9_runtime_features

# Set thread limits
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["GX1_REPLAY"] = "1"  # Enable replay mode for dtype asserts


def main():
    parser = argparse.ArgumentParser(description="Repro feature building timeout")
    parser.add_argument("--data_file", type=str, required=True, help="Parquet data file")
    parser.add_argument("--feature_meta_path", type=str, required=True, help="Feature metadata path")
    parser.add_argument("--start_date", type=str, default="2025-01-06", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-01-13", help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit_bars", type=int, default=200, help="Limit number of bars to test")
    parser.add_argument("--step", type=int, default=10, help="Test every Nth bar")
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    feature_meta_path = Path(args.feature_meta_path)
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return 1
    
    if not feature_meta_path.exists():
        print(f"❌ Feature metadata not found: {feature_meta_path}")
        return 1
    
    print(f"[REPRO] Loading data: {data_file}")
    df = pd.read_parquet(data_file)
    
    # Filter by date range
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    start_ts = pd.Timestamp(args.start_date, tz="UTC")
    end_ts = pd.Timestamp(args.end_date, tz="UTC")
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    
    print(f"[REPRO] Loaded {len(df):,} bars from {df.index.min()} to {df.index.max()}")
    
    # Check OHLCV dtypes
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    print("\n[REPRO] OHLCV dtypes:")
    for col in ohlcv_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        else:
            print(f"  {col}: MISSING")
    
    # Test on every Nth bar (up to limit_bars)
    test_indices = list(range(0, min(len(df), args.limit_bars), args.step))
    print(f"\n[REPRO] Testing {len(test_indices)} bars (every {args.step}th bar, max {args.limit_bars} bars)")
    
    timeout_count = 0
    error_count = 0
    success_count = 0
    
    for i, bar_idx in enumerate(test_indices):
        # Get window up to this bar
        df_window = df.iloc[:bar_idx+1].copy()
        
        print(f"\n[REPRO] Bar {i+1}/{len(test_indices)} (index {bar_idx}, shape={df_window.shape})")
        
        start_time = time.perf_counter()
        
        try:
            df_feats, seq_feat_names, snap_feat_names = build_v9_runtime_features(
                df_window,
                feature_meta_path,
            )
            
            elapsed = time.perf_counter() - start_time
            
            if elapsed > 30.0:
                print(f"⚠️  Slow: {elapsed:.2f}s (potential timeout risk)")
                timeout_count += 1
            else:
                print(f"✅ OK: {elapsed:.3f}s, features: {len(seq_feat_names)} seq + {len(snap_feat_names)} snap")
                success_count += 1
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            error_count += 1
            print(f"❌ ERROR after {elapsed:.3f}s: {e}")
            print(f"   Type: {type(e).__name__}")
            
            # Print stack trace
            stack_lines = traceback.format_exc().splitlines()[:20]
            print("   Stack trace (first 20 lines):")
            for line in stack_lines:
                print(f"   {line}")
            
            # Check if it's a timeout (pandas hanging)
            if "timeout" in str(e).lower() or elapsed > 30.0:
                print(f"\n❌ TIMEOUT detected at bar {bar_idx} (elapsed: {elapsed:.2f}s)")
                print(f"   DataFrame shape: {df_window.shape}")
                print(f"   DataFrame dtypes:\n{df_window[ohlcv_cols].dtypes}")
                print(f"   First 5 rows:\n{df_window[ohlcv_cols].head()}")
                
                return 1
    
    print(f"\n[REPRO] Summary:")
    print(f"  Success: {success_count}/{len(test_indices)}")
    print(f"  Slow (>30s): {timeout_count}")
    print(f"  Errors: {error_count}")
    
    if error_count == 0:
        print("\n✅ No timeouts detected!")
        return 0
    else:
        print(f"\n❌ {error_count} error(s) detected")
        return 1


if __name__ == "__main__":
    exit(main())

