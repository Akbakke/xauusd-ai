#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay with Shadow Logging

Runs replay over historical data with SNIPER policy and shadow logging enabled.
Generates both shadow hits and trades for RL dataset building.

Usage:
    # Replay full 2025
    python gx1/scripts/replay_with_shadow.py \
        --data_path data/raw/xauusd_m5_2025.parquet \
        --output_dir runs/replay_shadow

    # Replay specific date range
    python gx1/scripts/replay_with_shadow.py \
        --data_path data/raw/xauusd_m5_2025.parquet \
        --start_date 2025-01-01 \
        --end_date 2025-03-31 \
        --output_dir runs/replay_shadow/Q1_2025
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from gx1.execution.oanda_demo_runner import GX1DemoRunner


def filter_data_by_date(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    if start_date:
        start_ts = pd.to_datetime(start_date, utc=True)
        df = df[df.index >= start_ts]
    
    if end_date:
        end_ts = pd.to_datetime(end_date, utc=True)
        df = df[df.index <= end_ts]
    
    return df


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Replay with Shadow Logging for RL Dataset Building"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to historical data file (CSV or Parquet)",
    )
    parser.add_argument(
        "--policy_path",
        type=Path,
        default=Path("gx1/configs/policies/sniper_snapshot/2025_SNIPER_V1/GX1_V11_REPLAY_SHADOW_2025.yaml"),
        help="Path to replay policy config (default: GX1_V11_REPLAY_SHADOW_2025.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/replay_shadow"),
        help="Output directory for replay run (default: runs/replay_shadow)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD, optional)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, optional)",
    )
    parser.add_argument(
        "--shadow_thresholds",
        type=str,
        default="0.55,0.58,0.60,0.62,0.65",
        help="Shadow thresholds (comma-separated, default: 0.55,0.58,0.60,0.62,0.65)",
    )
    parser.add_argument(
        "--max_bars",
        type=int,
        default=None,
        help="Maximum bars to process (for debugging, optional)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_path.exists():
        print(f"ERROR: Data file not found: {args.data_path}")
        return 1
    
    if not args.policy_path.exists():
        print(f"ERROR: Policy config not found: {args.policy_path}")
        return 1
    
    # Set shadow thresholds environment variable
    os.environ["SNIPER_SHADOW_THRESHOLDS"] = args.shadow_thresholds
    print(f"[SHADOW] Shadow thresholds: {args.shadow_thresholds}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run ID from timestamp
    run_id = f"REPLAY_SHADOW_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Replay with Shadow Logging")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Policy path: {args.policy_path}")
    print(f"Output dir: {run_dir}")
    print(f"Shadow thresholds: {args.shadow_thresholds}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")
    print("=" * 60)
    print("")
    
    # Load and filter data
    print(f"Loading data from {args.data_path}...")
    if args.data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.data_path)
    else:
        df = pd.read_csv(args.data_path)
    
    # Ensure time column is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Ensure index is timezone-aware UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    
    # Filter by date range if specified
    if args.start_date or args.end_date:
        df = filter_data_by_date(df, args.start_date, args.end_date)
        print(f"Filtered to {len(df)} bars")
    
    if len(df) == 0:
        print("ERROR: No data after filtering")
        return 1
    
    # Limit bars if specified
    if args.max_bars:
        df = df.iloc[:args.max_bars]
        print(f"Limited to {len(df)} bars (debug mode)")
    
    # Save filtered data to run directory
    filtered_data_path = run_dir / "price_data_filtered.parquet"
    df.to_parquet(filtered_data_path)
    print(f"Saved filtered data to {filtered_data_path}")
    
    # Set run_id environment variable to control output directory
    os.environ["GX1_RUN_ID"] = run_id
    
    # Initialize runner with replay mode
    print(f"\nInitializing runner with policy: {args.policy_path}")
    runner = GX1DemoRunner(
        args.policy_path,
        dry_run_override=True,  # Always dry_run in replay
        replay_mode=True,
        fast_replay=False,  # Full replay for complete data
        output_dir=run_dir,  # Explicit output directory
    )
    
    # Ensure run_dir is set correctly for shadow journal path resolution
    # In replay mode, run_dir is typically set to gx1/wf_runs/<run_id>
    # But we want it in runs/replay_shadow/<run_id> for shadow logging
    # So we override it after initialization
    runner.run_dir = str(run_dir)
    
    # Set max_bars if specified
    if args.max_bars:
        runner._max_bars = args.max_bars
    
    # Run replay
    print(f"\nStarting replay...")
    print(f"Run directory: {run_dir}")
    print("")
    
    try:
        runner.run_replay(filtered_data_path)
        print("\n" + "=" * 60)
        print("✅ Replay complete!")
        print("=" * 60)
        print(f"Run directory: {run_dir}")
        print("")
        print("Output files:")
        print(f"  - Shadow journal: {run_dir / 'shadow' / 'shadow_hits.jsonl'}")
        print(f"  - Trade journal: {run_dir / 'trade_journal' / 'trades' / '*.json'}")
        print(f"  - Run header: {run_dir / 'run_header.json'}")
        print("")
        print("Next steps:")
        print("  1. Build RL dataset:")
        print(f"     python gx1/scripts/build_historical_rl_dataset.py \\")
        print(f"       --wf_runs_dir {args.output_dir} \\")
        print(f"       --output_prefix FULLYEAR_2025")
        print("")
        print("  2. Train Entry Critic:")
        print("     python -m gx1.rl.train_entry_critic_v1 \\")
        print("       --dataset_path data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025.parquet")
        return 0
    except Exception as e:
        print(f"\n❌ Replay failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

