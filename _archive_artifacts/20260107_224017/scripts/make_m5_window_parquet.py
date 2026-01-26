#!/usr/bin/env python3
"""
Extract M5 candles for a specific date window.

Input: Full M5 dataset (parquet)
Output: Filtered M5 candles parquet for date range

Required columns: time/ts + OHLC + bid/ask columns
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def verify_required_columns(df: pd.DataFrame) -> list[str]:
    """Verify required columns exist. Returns list of missing columns."""
    required_cols = [
        "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close",
    ]
    missing = []
    for col in required_cols:
        if col not in df.columns:
            missing.append(col)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Extract M5 candles for date window")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file (full dataset)")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    # Load data
    print(f"[DATA] Loading: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Set time index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    print(f"[DATA] Loaded {len(df):,} rows")
    print(f"[DATA] Date range: {df.index.min()} to {df.index.max()}")
    print(f"[DATA] Columns: {len(df.columns)} ({', '.join(df.columns[:10])}...)")
    
    # Verify required columns
    missing_cols = verify_required_columns(df)
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return 1
    
    # Filter by date range
    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    
    print(f"[DATA] Filtering: {start_ts} to {end_ts}")
    filtered = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    
    if len(filtered) == 0:
        print(f"❌ No data found in date range {args.start} to {args.end}")
        return 1
    
    print(f"[DATA] Filtered: {len(filtered):,} rows")
    print(f"[DATA] Date range: {filtered.index.min()} to {filtered.index.max()}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path)
    
    print(f"✅ Saved: {output_path}")
    print(f"   Rows: {len(filtered):,}")
    print(f"   Min TS: {filtered.index.min()}")
    print(f"   Max TS: {filtered.index.max()}")
    print(f"   Columns: {len(filtered.columns)}")
    
    return 0


if __name__ == "__main__":
    exit(main())



