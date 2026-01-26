#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel Replay with Shadow Logging

Splits historical data into N chunks and runs replay_with_shadow.py in parallel.
Each worker processes its chunk independently.

Usage:
    python gx1/scripts/replay_with_shadow_parallel.py \
        --data_path data/raw/xauusd_m5_2025_bid_ask.parquet \
        --output_dir runs/replay_shadow \
        --n_workers 7
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))


def split_dataframe(df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
    """Split DataFrame into approximately equal chunks."""
    total_rows = len(df)
    chunk_size = total_rows // n_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(n_chunks):
        if i == n_chunks - 1:
            # Last chunk gets remainder
            end_idx = total_rows
        else:
            end_idx = start_idx + chunk_size
        
        chunks.append(df.iloc[start_idx:end_idx].copy())
        start_idx = end_idx
    
    return chunks


def run_worker_chunk(
    chunk_id: int,
    chunk_df: pd.DataFrame,
    policy_path: Path,
    output_dir: Path,
    shadow_thresholds: str,
    project_root: Path,
) -> bool:
    """Run replay_with_shadow.py for a single chunk."""
    try:
        # Create chunk-specific output directory
        chunk_output_dir = output_dir / f"worker_{chunk_id}"
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunk to temp parquet
        chunk_file = chunk_output_dir / f"chunk_{chunk_id}.parquet"
        chunk_df.to_parquet(chunk_file)
        
        # Set environment variables for this worker (CRITICAL: 1 thread per worker)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["GX1_XGB_THREADS"] = "1"
        env["SNIPER_SHADOW_THRESHOLDS"] = shadow_thresholds
        
        # Run replay_with_shadow.py for this chunk
        cmd = [
            sys.executable,
            str(script_dir / "replay_with_shadow.py"),
            "--data_path", str(chunk_file),
            "--policy_path", str(policy_path),
            "--output_dir", str(chunk_output_dir),
            "--shadow_thresholds", shadow_thresholds,
        ]
        
        log_file = chunk_output_dir / f"worker_{chunk_id}.log"
        print(f"[WORKER {chunk_id}] Starting... (chunk: {len(chunk_df):,} bars, period: {chunk_df.index.min()} to {chunk_df.index.max()})")
        
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = proc.wait()
        
        if returncode == 0:
            print(f"[WORKER {chunk_id}] ✅ Complete")
            return True
        else:
            print(f"[WORKER {chunk_id}] ❌ Failed (returncode={returncode}, see {log_file})")
            return False
    
    except Exception as e:
        print(f"[WORKER {chunk_id}] ❌ Exception: {e}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Parallel Replay with Shadow Logging"
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
        help="Path to replay policy config",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("runs/replay_shadow"),
        help="Output directory for replay runs",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=7,
        help="Number of parallel workers (default: 7)",
    )
    parser.add_argument(
        "--shadow_thresholds",
        type=str,
        default="0.55,0.58,0.60,0.62,0.65",
        help="Shadow thresholds (comma-separated)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    
    print("=" * 60)
    print("Parallel Replay with Shadow Logging")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Policy: {args.policy_path}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.n_workers}")
    print()
    
    # Load data
    print("[1/3] Loading data...")
    if args.data_path.suffix == ".parquet":
        df = pd.read_parquet(args.data_path)
    elif args.data_path.suffix == ".csv":
        df = pd.read_csv(args.data_path, index_col=0, parse_dates=True)
    else:
        print(f"ERROR: Unsupported file format: {args.data_path.suffix}")
        return 1
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    print(f"✅ Loaded {len(df):,} bars")
    print(f"   Period: {df.index.min()} to {df.index.max()}")
    print()
    
    # Split into chunks
    print(f"[2/3] Splitting into {args.n_workers} chunks...")
    chunks = split_dataframe(df, args.n_workers)
    chunk_sizes = [len(c) for c in chunks]
    print(f"   Chunk sizes: {chunk_sizes}")
    print(f"   Total: {sum(chunk_sizes):,} bars")
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunks first
    print("[3/3] Saving chunks and starting parallel workers...")
    for i, chunk in enumerate(chunks):
        chunk_file = args.output_dir / f"chunk_{i}.parquet"
        chunk.to_parquet(chunk_file)
        print(f"   Saved chunk {i}: {len(chunk):,} bars")
    print()
    
    start_time = time.time()
    
    # Run workers in parallel using subprocess
    processes = []
    for i in range(len(chunks)):
        chunk_file = args.output_dir / f"chunk_{i}.parquet"
        worker_output_dir = args.output_dir / f"worker_{i}"
        worker_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["GX1_XGB_THREADS"] = "1"
        env["SNIPER_SHADOW_THRESHOLDS"] = args.shadow_thresholds
        
        # Run replay_with_shadow.py for this chunk
        cmd = [
            sys.executable,
            str(script_dir / "replay_with_shadow.py"),
            "--data_path", str(chunk_file),
            "--policy_path", str(args.policy_path),
            "--output_dir", str(worker_output_dir),
            "--shadow_thresholds", args.shadow_thresholds,
        ]
        
        log_file = worker_output_dir / f"worker_{i}.log"
        print(f"[WORKER {i}] Starting... (PID will be shown)")
        
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((i, proc, log_file))
            print(f"[WORKER {i}] Started (PID: {proc.pid}, log: {log_file})")
    
    print()
    print("Waiting for all workers to complete...")
    print()
    
    # Wait for all workers
    results = []
    for chunk_id, proc, log_file in processes:
        returncode = proc.wait()
        success = returncode == 0
        results.append((chunk_id, success))
    
    elapsed = time.time() - start_time
    print()
    print(f"✅ All workers complete (took {elapsed/60:.1f} minutes)")
    print()
    
    # Summary
    successful = [cid for cid, success in results if success]
    failed = [cid for cid, success in results if not success]
    
    print(f"✅ Successful: {len(successful)}/{args.n_workers}")
    if failed:
        print(f"❌ Failed: {failed}")
    
    # List output directories
    print()
    print("Output directories:")
    for i in range(args.n_workers):
        worker_dir = args.output_dir / f"worker_{i}"
        if worker_dir.exists():
            shadow_file = worker_dir / "REPLAY_SHADOW_*/shadow/shadow_hits.jsonl"
            shadow_files = list(worker_dir.glob("REPLAY_SHADOW_*/shadow/shadow_hits.jsonl"))
            trade_count = len(list((worker_dir / "REPLAY_SHADOW_*/trade_journal/trades").glob("*.json"))) if (worker_dir / "REPLAY_SHADOW_*/trade_journal/trades").exists() else 0
            if shadow_files:
                shadow_count = sum(1 for _ in open(shadow_files[0])) if shadow_files[0].exists() else 0
                print(f"  Worker {i}: {shadow_count} shadow, {trade_count} trades")
            else:
                print(f"  Worker {i}: (no output yet)")
    
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    exit(main())

