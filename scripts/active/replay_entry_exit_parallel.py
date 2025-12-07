#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel offline replay script for testing entry + exit together.

Runs replay in parallel chunks using multiple workers.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import subprocess
import time
from joblib import Parallel, delayed

from gx1.execution.oanda_demo_runner import GX1DemoRunner, load_yaml_config

# Sandbox environments may block os.sysconf("SC_SEM_NSEMS_MAX"), causing loky backend to fail.
# Monkey patch os.sysconf to return a safe default for that key when permission is denied.
_original_sysconf = os.sysconf

def _safe_sysconf(name):
    try:
        return _original_sysconf(name)
    except PermissionError:
        target = name
        if isinstance(name, int):
            target = name
        if (
            (isinstance(name, str) and name == "SC_SEM_NSEMS_MAX")
            or (isinstance(name, int) and name == getattr(os, "SC_SEM_NSEMS_MAX", None))
        ):
            return 256  # Conservative default that satisfies joblib checks
        raise

os.sysconf = _safe_sysconf

log_level_name = os.getenv("GX1_PARALLEL_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


def split_dataframe(df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
    """Split DataFrame into n_chunks roughly equal chunks."""
    chunk_size = len(df) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        if i == n_chunks - 1:
            # Last chunk gets remainder
            end_idx = len(df)
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append(df.iloc[start_idx:end_idx].copy())
    return chunks


def run_replay_chunk(
    chunk_idx: int,
    chunk_df: pd.DataFrame,
    policy_path: Path,
    output_dir: Path,
    exit_config: Optional[str] = None,
) -> Tuple[int, Path, Path]:
    """
    Run replay on a single chunk.
    
    Returns:
        (chunk_idx, trade_log_path, results_path)
    """
    # Set thread environment variables for this worker (CRITICAL: must be set before any imports)
    # This ensures each worker uses only 1 thread, so n_jobs=7 means exactly 7 workers
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["GX1_XGB_THREADS"] = "1"
    
    # Save chunk to temp parquet
    chunk_path = output_dir / f"chunk_{chunk_idx}.parquet"
    chunk_df.to_parquet(chunk_path)
    
    # Create unique trade log for this chunk
    # Use FARM_V1-specific naming if FARM_V1 config is detected
    policy_dict_check = load_yaml_config(policy_path)
    exit_type = policy_dict_check.get("exit_config", "")
    if "FARM_V1" in str(exit_type) or policy_dict_check.get("exit", {}).get("type") == "FARM_V1":
        trade_log_path = Path("gx1/live") / f"trade_log_farm_v1_chunk_{chunk_idx}.csv"
    else:
        trade_log_path = output_dir / f"trade_log_chunk_{chunk_idx}.csv"
    
    # Create unique policy file for this chunk (with unique trade log path)
    import tempfile
    import yaml
    policy_dict = load_yaml_config(policy_path)
    # Ensure logging section exists
    if "logging" not in policy_dict:
        policy_dict["logging"] = {}
    policy_dict["logging"]["trade_log_csv"] = str(trade_log_path)
    
    # Override exit config if provided (must be done before saving)
    if exit_config:
        policy_dict["exit_config"] = exit_config
    
    chunk_policy_path = output_dir / f"policy_chunk_{chunk_idx}.yaml"
    with open(chunk_policy_path, "w") as f:
        yaml.dump(policy_dict, f)
    
    # Run replay
    try:
        runner = GX1DemoRunner(
            chunk_policy_path,
            dry_run_override=True,
            replay_mode=True,
            fast_replay=True,
        )
        
        runner.run_replay(chunk_path)
        
        # Flush feature log buffer if feature logging is enabled
        try:
            from gx1.policy.entry_v9_policy_farm_v2 import flush_feature_log_final
            flush_feature_log_final()
        except ImportError:
            pass  # Feature logging not available
        
        # Results path
        results_path = output_dir / f"results_chunk_{chunk_idx}.json"
        
        # Extract summary from trade log
        if trade_log_path.exists():
            trades_df = pd.read_csv(trade_log_path, on_bad_lines='skip', engine='python')
            closed_trades = trades_df[
                trades_df["exit_time"].notna() &
                trades_df["pnl_bps"].notna() &
                (trades_df["pnl_bps"] != "")
            ].copy()
            
            if len(closed_trades) > 0:
                closed_trades["pnl_bps"] = pd.to_numeric(closed_trades["pnl_bps"], errors='coerce')
                closed_trades = closed_trades[closed_trades["pnl_bps"].notna()]
                
                period_days = (chunk_df.index.max() - chunk_df.index.min()).total_seconds() / (24 * 3600)
                
                summary = {
                    "chunk_idx": chunk_idx,
                    "n_trades": len(closed_trades),
                    "mean_pnl_bps": float(closed_trades["pnl_bps"].mean()),
                    "median_pnl_bps": float(closed_trades["pnl_bps"].median()),
                    "period_days": period_days,
                    "trade_log_path": str(trade_log_path),
                }
                
                with open(results_path, "w") as f:
                    json.dump(summary, f, indent=2)
        
        return (chunk_idx, trade_log_path, results_path)
    except Exception as e:
        logger.error(f"Chunk {chunk_idx} failed: {e}", exc_info=True)
        return (chunk_idx, None, None)


def merge_chunk_results(chunk_results: List[Tuple[int, Path, Path]], output_path: Path, merged_trade_log_path: Optional[Path] = None) -> Dict[str, Any]:
    """Merge results from all chunks."""
    all_trades = []
    
    for chunk_idx, trade_log_path, results_path in chunk_results:
        if trade_log_path and trade_log_path.exists():
            try:
                trades_df = pd.read_csv(trade_log_path, on_bad_lines='skip', engine='python')
                if len(trades_df) > 0:
                    all_trades.append(trades_df)
            except Exception as e:
                logger.warning(f"Failed to read chunk {chunk_idx} trade log: {e}")
    
    if len(all_trades) == 0:
        return {"n_trades": 0, "mean_pnl_bps": 0.0}
    
    # Combine all trades - ensure all columns are preserved
    # Use join='outer' to keep all columns from all chunks
    combined_df = pd.concat(all_trades, ignore_index=True, sort=False)
    
    # Ensure FARM fields are extracted from extra if they exist in chunks but not as columns
    # This handles cases where some chunks have FARM columns and others don't
    if 'extra' in combined_df.columns and 'farm_entry_session' not in combined_df.columns:
        import json
        def extract_farm_from_extra(extra_str):
            if pd.isna(extra_str) or extra_str == "":
                return None, None, None
            try:
                extra_dict = json.loads(extra_str) if isinstance(extra_str, str) else extra_str
                if isinstance(extra_dict, dict):
                    return (
                        extra_dict.get("farm_entry_session"),
                        extra_dict.get("farm_entry_vol_regime"),
                        extra_dict.get("farm_guard_version")
                    )
            except:
                pass
            return None, None, None
        
        farm_data = combined_df["extra"].apply(extract_farm_from_extra)
        combined_df["farm_entry_session"] = [x[0] for x in farm_data]
        combined_df["farm_entry_vol_regime"] = [x[1] for x in farm_data]
        combined_df["farm_guard_version"] = [x[2] for x in farm_data]
        logger.info("✅ Extracted FARM fields from 'extra' column during merge")
    
    # Save merged trade log if path provided
    if merged_trade_log_path:
        merged_trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure schema order (use trade_log_schema if available)
        try:
            from gx1.execution.trade_log_schema import TRADE_LOG_FIELDS
            # Reorder columns to match schema, keeping any extra columns
            schema_cols = [c for c in TRADE_LOG_FIELDS if c in combined_df.columns]
            extra_cols = [c for c in combined_df.columns if c not in TRADE_LOG_FIELDS]
            combined_df = combined_df[schema_cols + extra_cols]
        except ImportError:
            pass  # If schema not available, use default order
        
        combined_df.to_csv(merged_trade_log_path, index=False)
        logger.info(f"✅ Merged trade log saved to {merged_trade_log_path} ({len(combined_df)} rows)")
    
    # Filter closed trades
    closed_trades = combined_df[
        combined_df["exit_time"].notna() &
        combined_df["pnl_bps"].notna() &
        (combined_df["pnl_bps"] != "")
    ].copy()
    
    if len(closed_trades) == 0:
        return {
            "n_trades": 0,
            "mean_pnl_bps": 0.0,
            "median_pnl_bps": 0.0,
            "trades_per_day": 0.0,
            "ev_per_day": 0.0,
            "tp2_rate": 0.0,
            "sl_rate": 0.0,
            "timeout_rate": 0.0,
            "sl_breakeven_rate": 0.0,
            "median_bars_held": 0.0,
        }
    
    closed_trades["pnl_bps"] = pd.to_numeric(closed_trades["pnl_bps"], errors='coerce')
    closed_trades = closed_trades[closed_trades["pnl_bps"].notna()]
    
    # Calculate period
    if "entry_time" in closed_trades.columns:
        entry_times = pd.to_datetime(closed_trades["entry_time"])
        period_days = (entry_times.max() - entry_times.min()).total_seconds() / (24 * 3600)
    else:
        period_days = 1.0
    
    # Exit reason breakdown
    exit_reason_col = "primary_exit_reason" if "primary_exit_reason" in closed_trades.columns else "exit_reason"
    if exit_reason_col in closed_trades.columns:
        tp2_rate = (closed_trades[exit_reason_col].str.contains("TP2", case=False, na=False)).mean()
        sl_rate = (closed_trades[exit_reason_col].str.contains("SL", case=False, na=False) & 
                  ~closed_trades[exit_reason_col].str.contains("BREAKEVEN", case=False, na=False)).mean()
        timeout_rate = (closed_trades[exit_reason_col].str.contains("TIMEOUT", case=False, na=False)).mean()
        sl_breakeven_rate = (closed_trades[exit_reason_col].str.contains("SL_BREAKEVEN", case=False, na=False) |
                            closed_trades[exit_reason_col].str.contains("BREAKEVEN", case=False, na=False)).mean()
    else:
        tp2_rate = sl_rate = timeout_rate = sl_breakeven_rate = 0.0
    
    # Bars held
    bars_held_col = "bars_held" if "bars_held" in closed_trades.columns else "bars_in_trade"
    median_bars_held = float(closed_trades[bars_held_col].median()) if bars_held_col in closed_trades.columns else 0.0
    
    trades_per_day = len(closed_trades) / period_days if period_days > 0 else 0.0
    mean_pnl = float(closed_trades["pnl_bps"].mean()) if len(closed_trades) > 0 else 0.0
    ev_per_day = mean_pnl * trades_per_day
    
    result = {
        "n_trades": len(closed_trades),
        "mean_pnl_bps": mean_pnl,
        "median_pnl_bps": float(closed_trades["pnl_bps"].median()) if len(closed_trades) > 0 else 0.0,
        "trades_per_day": trades_per_day,
        "ev_per_day": ev_per_day,
        "tp2_rate": float(tp2_rate),
        "sl_rate": float(sl_rate),
        "timeout_rate": float(timeout_rate),
        "sl_breakeven_rate": float(sl_breakeven_rate),
        "median_bars_held": median_bars_held,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel offline replay with entry + exit")
    parser.add_argument(
        "--price-data",
        type=str,
        required=True,
        help="Path to price data parquet/CSV",
    )
    parser.add_argument(
        "--base-policy",
        type=str,
        default="gx1/configs/policies/GX1_V11_OANDA_DEMO_V2_EXIT_ONLY.yaml",
        help="Base policy YAML path",
    )
    parser.add_argument(
        "--exit-config",
        type=str,
        default=None,
        help="Path to exit config",
    )
    parser.add_argument(
        "--limit-bars",
        type=int,
        default=None,
        help="Limit to last N bars",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/entry_exit_v2_drift_only.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=7,
        help="Number of parallel workers",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"PARALLEL OFFLINE REPLAY: {args.n_workers} WORKERS")
    print("=" * 80)
    print()
    
    # Load price data
    print(f"[1/4] Loading price data: {args.price_data}")
    price_path = Path(args.price_data)
    
    if price_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(price_path)
    else:
        df = pd.read_csv(price_path)
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Set up time index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Limit to last N bars if specified
    if args.limit_bars and len(df) > args.limit_bars:
        df = df.iloc[-args.limit_bars:].copy()
        print(f"   Limited to last {args.limit_bars:,} bars")
    
    print(f"✅ Loaded {len(df):,} bars")
    print(f"   Period: {df.index.min()} to {df.index.max()}")
    print()
    
    # Split into chunks
    print(f"[2/4] Splitting into {args.n_workers} chunks...")
    chunks = split_dataframe(df, args.n_workers)
    print(f"   Chunk sizes: {[len(c) for c in chunks]}")
    print()
    
    # Create output directory
    output_dir = Path(args.output).parent / "parallel_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run parallel replay
    print(f"[3/4] Running {args.n_workers} parallel replays...")
    print(f"   Using {args.n_workers} workers with MAX CPU")
    print()
    
    start_time = time.time()
    
    # Use threading backend to avoid oversubscription
    # Each worker uses 1 thread, so n_jobs=7 means exactly 7 workers
    parallel_backend = os.getenv("GX1_PARALLEL_BACKEND", "loky")
    chunk_results = Parallel(n_jobs=args.n_workers, backend=parallel_backend, verbose=10)(
        delayed(run_replay_chunk)(
            chunk_idx=i,
            chunk_df=chunk,
            policy_path=Path(args.base_policy),
            output_dir=output_dir,
            exit_config=args.exit_config,
        )
        for i, chunk in enumerate(chunks)
    )
    
    elapsed = time.time() - start_time
    print(f"✅ All chunks complete (took {elapsed/60:.1f} minutes)")
    print()
    
    # Merge results
    print("[4/4] Merging results...")
    # Determine merged trade log path from output path
    output_path = Path(args.output)
    # Check if FARM_V1 config is used
    policy_dict_check = load_yaml_config(Path(args.base_policy))
    exit_type = policy_dict_check.get("exit_config", "")
    if "FARM_V1" in str(exit_type) or policy_dict_check.get("exit", {}).get("type") == "FARM_V1":
        merged_trade_log_path = Path("gx1/live") / f"trade_log_farm_v1_test_merged.csv"
    else:
        merged_trade_log_path = output_path.parent / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
    merged_results = merge_chunk_results(chunk_results, output_path, merged_trade_log_path=merged_trade_log_path)
    
    # Save merged results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(merged_results, f, indent=2, default=str)
    
    print(f"✅ Results saved to {output_path}")
    print()
    
    # Print summary (robust to missing keys when 0 trades)
    print("=" * 80)
    print("PARALLEL REPLAY SUMMARY")
    print("=" * 80)
    print()
    
    n_trades = merged_results.get("n_trades", 0)
    print(f"n_trades: {n_trades:,}")
    
    if n_trades == 0:
        print()
        print("⚠️  No trades generated for this run.")
        print("   Check policy filters / regimes / period.")
        print()
        print("Metrics (all 0.0 due to no trades):")
        print(f"  EV/trade: {merged_results.get('mean_pnl_bps', 0.0):.2f} bps")
        print(f"  EV/day: {merged_results.get('ev_per_day', 0.0):.2f} bps")
        print(f"  Trades/day: {merged_results.get('trades_per_day', 0.0):.2f}")
        print()
    else:
        print(f"EV/trade (mean_pnl_bps): {merged_results.get('mean_pnl_bps', 0.0):.2f} bps")
        print(f"trades_per_day: {merged_results.get('trades_per_day', 0.0):.2f}")
        print(f"EV/day: {merged_results.get('ev_per_day', 0.0):.2f} bps")
        print()
        print("Exit Rates:")
        print(f"  TP2-rate: {merged_results.get('tp2_rate', 0.0):.4f} ({merged_results.get('tp2_rate', 0.0)*100:.2f}%)")
        print(f"  SL-rate: {merged_results.get('sl_rate', 0.0):.4f} ({merged_results.get('sl_rate', 0.0)*100:.2f}%)")
        print(f"  TIMEOUT-rate: {merged_results.get('timeout_rate', 0.0):.4f} ({merged_results.get('timeout_rate', 0.0)*100:.2f}%)")
        print(f"  SL_BREAKEVEN-rate: {merged_results.get('sl_breakeven_rate', 0.0):.4f} ({merged_results.get('sl_breakeven_rate', 0.0)*100:.2f}%)")
        print()
        print(f"median_bars_held: {merged_results.get('median_bars_held', 0.0):.1f}")
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())
