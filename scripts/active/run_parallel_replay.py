#!/usr/bin/env python3
"""
Run parallel replay by splitting dataset into chunks and running multiple workers.
Each worker processes a chunk independently, then results are merged.
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def run_replay_chunk(chunk_file: Path, policy_file: Path, chunk_id: int, total_chunks: int):
    """Run replay for a single chunk with progress monitoring."""
    log_file = Path(f"/tmp/v9_replay_chunk_{chunk_id}.log")
    
    # Read config to determine if entry-only mode
    import yaml
    with open(policy_file, "r") as f:
        policy = yaml.safe_load(f)
    
    # For entry-only mode, use entry_only_log path
    mode = policy.get("mode", "NORMAL")
    if mode == "ENTRY_ONLY":
        # Entry-only logs use chunk_id in filename
        entry_log_path = Path(f"gx1/live/entry_only_log_v9_test_chunk_{chunk_id}.csv")
        output_log = entry_log_path
    else:
        # Normal mode: trade logs
        trade_log_template = policy.get("logging", {}).get("trade_log_csv", "gx1/live/trade_log_chunk_{chunk_id}.csv")
        trade_log_path = trade_log_template.format(chunk_id=chunk_id)
        output_log = Path(trade_log_path)
    
    cmd = [
        sys.executable, "-m", "gx1.execution.oanda_demo_runner",
        "--policy", str(policy_file),
        "--replay-csv", str(chunk_file),
        "--fast-replay",
    ]
    
    # Set GX1_CHUNK_ID environment variable for template formatting
    env = os.environ.copy()
    env["GX1_CHUNK_ID"] = str(chunk_id)
    
    print(f"[CHUNK {chunk_id}/{total_chunks}] Starting replay for {chunk_file.name}...")
    start_time = time.time()
    
    try:
        with open(log_file, "w") as f:
            # Start process
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            
            # Monitor progress
            last_size = 0
            while proc.poll() is None:
                time.sleep(5)  # Check every 5 seconds
                
                # Check log file for progress
                if log_file.exists():
                    current_size = log_file.stat().st_size
                    if current_size > last_size:
                        # Try to extract progress from log
                        try:
                            with open(log_file, "r") as log_f:
                                lines = log_f.readlines()
                                for line in reversed(lines[-50:]):  # Check last 50 lines
                                    if "REPLAY] Progress:" in line:
                                        # Extract progress percentage
                                        if "%" in line:
                                            # Progress extracted but not used (pickle issue)
                                            pass
                                        break
                        except:
                            pass
                        last_size = current_size
                
                # Check for errors in log
                try:
                    with open(log_file, "r") as log_f:
                        last_lines = log_f.readlines()[-10:]
                        for line in last_lines:
                            if any(err in line.upper() for err in ["TRACEBACK", "FATAL", "CRITICAL ERROR", "EXCEPTION"]):
                                print(f"[CHUNK {chunk_id}/{total_chunks}] ⚠️  Error detected in log, stopping...")
                                proc.terminate()
                                proc.wait(timeout=10)
                                return chunk_id, trade_log, False
                except:
                    pass
            
            # Wait for process to complete
            result = proc.wait()
        
        elapsed = time.time() - start_time
        if result == 0:
            print(f"[CHUNK {chunk_id}/{total_chunks}] ✅ Completed in {elapsed:.1f}s")
            return chunk_id, trade_log, True
        else:
            print(f"[CHUNK {chunk_id}/{total_chunks}] ❌ Failed with code {result}")
            return chunk_id, trade_log, False
    except subprocess.TimeoutExpired:
        print(f"[CHUNK {chunk_id}/{total_chunks}] ⏱️  Timeout after 2 hours")
        return chunk_id, trade_log, False
    except Exception as e:
        print(f"[CHUNK {chunk_id}/{total_chunks}] ❌ Error: {e}")
        return chunk_id, trade_log, False


def split_dataset(input_file: Path, n_chunks: int, output_dir: Path):
    """Split dataset into n_chunks."""
    print(f"Loading dataset: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Dataset shape: {df.shape}, date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate chunk size
    total_bars = len(df)
    chunk_size = total_bars // n_chunks
    remainder = total_bars % n_chunks
    
    print(f"Splitting into {n_chunks} chunks (~{chunk_size} bars each)")
    
    chunk_files = []
    start_idx = 0
    
    for i in range(n_chunks):
        # Distribute remainder across first chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_file = output_dir / f"chunk_{i}.parquet"
        chunk_df.to_parquet(chunk_file)
        
        print(f"  Chunk {i}: {len(chunk_df)} bars ({chunk_df.index.min()} to {chunk_df.index.max()})")
        chunk_files.append(chunk_file)
        
        start_idx = end_idx
    
    return chunk_files


def merge_trade_logs(trade_logs: list, output_file: Path):
    """Merge trade logs or entry-only logs from all chunks."""
    print(f"\nMerging {len(trade_logs)} log files...")
    
    all_data = []
    for log_file in trade_logs:
        if log_file.exists():
            try:
                df = pd.read_csv(log_file)
                if len(df) > 0:
                    all_data.append(df)
                    print(f"  ✅ {log_file.name}: {len(df)} rows")
            except Exception as e:
                print(f"  ⚠️  {log_file.name}: Error reading - {e}")
        else:
            print(f"  ⚠️  {log_file.name}: File not found")
    
    if not all_data:
        print("❌ No logs to merge")
        return
    
    merged = pd.concat(all_data, ignore_index=True)
    
    # Sort by timestamp (entry-only) or entry_time (trade logs)
    if "timestamp" in merged.columns:
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce", utc=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)
    elif "entry_time" in merged.columns:
        merged["entry_time"] = pd.to_datetime(merged["entry_time"], errors="coerce", utc=True)
        merged = merged.sort_values("entry_time").reset_index(drop=True)
    
    merged.to_csv(output_file, index=False)
    print(f"✅ Merged {len(merged)} rows -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run parallel replay with multiple workers")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file")
    parser.add_argument("--policy", type=Path, required=True, help="Policy YAML file")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers")
    parser.add_argument("--chunk-dir", type=Path, default=Path("data/replay_chunks"), help="Directory for chunks")
    args = parser.parse_args()
    
    # Create chunk directory
    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Split dataset
    print("="*80)
    print("STEP 1: Splitting dataset")
    print("="*80)
    chunk_files = split_dataset(args.input, args.workers, args.chunk_dir)
    
    # Run parallel replays with progress monitoring
    print("\n" + "="*80)
    print(f"STEP 2: Running {args.workers} parallel replays")
    print("="*80)
    
    start_time = time.time()
    results = []
    chunk_progress = {}  # Track progress per chunk
    chunk_errors = {}  # Track errors per chunk
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_replay_chunk, chunk_file, args.policy, i, args.workers): i
            for i, chunk_file in enumerate(chunk_files)
        }
        
        # Monitor progress and check for errors
        while futures:
            time.sleep(5)  # Check every 5 seconds
            
            # Check for completed futures
            completed_futures = []
            for future in list(futures.keys()):
                if future.done():
                    try:
                        chunk_id, trade_log, success = future.result(timeout=1)
                        results.append((chunk_id, trade_log, success))
                        if not success:
                            chunk_errors[chunk_id] = "Failed"
                        completed_futures.append(future)
                    except Exception as e:
                        chunk_id = futures[future]
                        chunk_errors[chunk_id] = str(e)
                        completed_futures.append(future)
            
            # Remove completed futures
            for future in completed_futures:
                del futures[future]
            
            # Check for errors in log files
            for chunk_id in range(args.workers):
                log_file = Path(f"/tmp/v9_replay_chunk_{chunk_id}.log")
                if log_file.exists():
                    try:
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            for line in lines[-20:]:  # Check last 20 lines
                                if any(err in line.upper() for err in ["TRACEBACK", "FATAL", "CRITICAL ERROR", "EXCEPTION"]):
                                    if chunk_id not in chunk_errors:
                                        chunk_errors[chunk_id] = "Error detected in log"
                                        print(f"\n[ERROR] Chunk {chunk_id} has errors - stopping all replays")
                                        # Cancel remaining futures
                                        for future in list(futures.keys()):
                                            future.cancel()
                                        futures.clear()
                                        break
                    except:
                        pass
            
            # Print progress summary by checking log file sizes
            completed = len([r for r in results if r[2]])  # Count successful
            running = len(futures)
            print(f"[PROGRESS] {completed}/{args.workers} chunks complete | {running} still running", end='\r')
    
    elapsed = time.time() - start_time
    print(f"\n✅ All chunks completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    # Report errors
    if chunk_errors:
        print("\n⚠️  ERRORS DETECTED:")
        for chunk_id, error in chunk_errors.items():
            print(f"  Chunk {chunk_id}: {error}")
    
    # Merge results
    print("\n" + "="*80)
    print("STEP 3: Merging results")
    print("="*80)
    
    successful_logs = [log for _, log, success in sorted(results) if success]
    if successful_logs:
        # Determine output path based on mode
        import yaml
        with open(args.policy, "r") as f:
            policy = yaml.safe_load(f)
        mode = policy.get("mode", "NORMAL")
        
        if mode == "ENTRY_ONLY":
            output_log = Path("gx1/live/entry_only_log_v9_test_merged.csv")
        else:
            output_log = Path("gx1/live/trade_log_gx1_v11_oanda_demo_v9_test_merged.csv")
        merge_trade_logs(successful_logs, output_log)
        print(f"\n✅ Parallel replay complete! Results in {output_log}")
    else:
        print("\n❌ No successful chunks to merge")


if __name__ == "__main__":
    main()

