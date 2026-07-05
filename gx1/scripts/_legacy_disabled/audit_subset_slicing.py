#!/usr/bin/env python3
"""
Audit script to verify subset slicing matches replay expectations.

This script loads candles data and slices it using the exact same logic
as the replay runner, then prints the subset length and timestamps.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from typing import Optional

def slice_candles_subset(
    data_path: Path,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Slice candles data using the same logic as replay runner.
    
    This matches the slicing logic in replay_eval_gated_parallel.py
    """
    df = pd.read_parquet(data_path)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(f"Data index must be DatetimeIndex, got: {type(df.index)}")
    
    if start_ts is not None or end_ts is not None:
        # Ensure timestamps are timezone-aware if df.index is timezone-aware
        if df.index.tz is not None:
            if start_ts and start_ts.tz is None:
                start_ts = start_ts.tz_localize(df.index.tz)
            elif start_ts:
                start_ts = start_ts.tz_convert(df.index.tz)
            
            if end_ts and end_ts.tz is None:
                end_ts = end_ts.tz_localize(df.index.tz)
            elif end_ts:
                end_ts = end_ts.tz_convert(df.index.tz)
        
        ts_start = start_ts if start_ts is not None else df.index[0]
        ts_end = end_ts if end_ts is not None else df.index[-1]
        df = df.loc[ts_start:ts_end]
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Audit subset slicing")
    parser.add_argument("--data", type=Path, required=True, help="Path to candles parquet file")
    parser.add_argument("--start-ts", type=str, help="Start timestamp (ISO format)")
    parser.add_argument("--end-ts", type=str, help="End timestamp (ISO format)")
    parser.add_argument("--date-range", type=str, help="Date range in format 'START..END' (e.g., '2025-01-01..2025-03-31')")
    
    args = parser.parse_args()
    
    # Parse date range if provided
    start_ts = None
    end_ts = None
    
    if args.date_range:
        if ".." not in args.date_range:
            print(f"ERROR: Date range must be in format 'START..END', got: {args.date_range}", file=sys.stderr)
            sys.exit(1)
        start_str, end_str = args.date_range.split("..", 1)
        start_ts = pd.to_datetime(start_str.strip())
        end_ts = pd.to_datetime(end_str.strip())
    else:
        if args.start_ts:
            start_ts = pd.to_datetime(args.start_ts)
        if args.end_ts:
            end_ts = pd.to_datetime(args.end_ts)
    
    # Load and slice
    try:
        df_subset = slice_candles_subset(args.data, start_ts=start_ts, end_ts=end_ts)
        
        subset_len = len(df_subset)
        subset_first_ts = df_subset.index.min()
        subset_last_ts = df_subset.index.max()
        
        print(f"subset_len: {subset_len}")
        print(f"subset_first_ts: {subset_first_ts.isoformat()}")
        print(f"subset_last_ts: {subset_last_ts.isoformat()}")
        
        if subset_len == 0:
            print("WARNING: Subset is empty!", file=sys.stderr)
            sys.exit(1)
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
