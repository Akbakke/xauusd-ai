#!/usr/bin/env python3
"""
Del B: Crash capsule script for reproducing pandas rolling segfault.

This script:
1. Reads last_good.json from segfault capsule directory
2. Loads a small data slice around the crash point
3. Calls the same rolling function that crashed
4. Reports diagnostics

Usage:
    python3 scripts/repro_segfault_rolling_capsule.py [--capsule-dir data/temp/segfault_capsule] [--window-size 2000] [--safe-threading-env]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from gx1.features.basic_v1 import _kama, _roll, _parkinson_sigma, _zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_data_slice(data_file: Path, crash_idx: int, window_size: int = 2000) -> pd.DataFrame:
    """Load data slice around crash point."""
    log.info(f"Loading data slice: idx={crash_idx}, window={window_size}")
    
    if data_file.suffix.lower() == ".parquet":
        df_full = pd.read_parquet(data_file)
    else:
        df_full = pd.read_csv(data_file)
    
    # Ensure index is DatetimeIndex
    if "time" in df_full.columns:
        df_full["time"] = pd.to_datetime(df_full["time"], utc=True)
        df_full = df_full.set_index("time").sort_index()
    elif not isinstance(df_full.index, pd.DatetimeIndex):
        if "ts" in df_full.columns:
            df_full["ts"] = pd.to_datetime(df_full["ts"], utc=True)
            df_full = df_full.set_index("ts").sort_index()
    
    # Calculate slice bounds
    start_idx = max(0, crash_idx - window_size)
    end_idx = min(len(df_full), crash_idx + window_size)
    
    log.info(f"Full data: {len(df_full)} rows, slice: [{start_idx}:{end_idx}]")
    df_slice = df_full.iloc[start_idx:end_idx].copy()
    
    return df_slice, crash_idx - start_idx  # Return adjusted crash_idx


def diagnose_series(series: pd.Series, name: str) -> dict:
    """Diagnose series for potential issues."""
    diag = {
        "name": name,
        "len": len(series),
        "dtype": str(series.dtype),
        "n_nan": int(series.isna().sum()),
        "n_inf": 0,
        "min": None,
        "max": None,
        "is_object": False,
        "is_mixed": False,
    }
    
    if pd.api.types.is_numeric_dtype(series):
        diag["n_inf"] = int(np.isinf(series).sum())
        finite_vals = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite_vals) > 0:
            diag["min"] = float(finite_vals.min())
            diag["max"] = float(finite_vals.max())
    else:
        diag["is_object"] = True
        if len(series) > 0:
            unique_types = set(type(v).__name__ for v in series.head(100) if pd.notna(v))
            diag["is_mixed"] = len(unique_types) > 1
    
    return diag


def repro_rolling_crash(context: dict, data_file: Path, window_size: int = 2000):
    """Reproduce rolling crash based on context."""
    log.info("=" * 80)
    log.info("REPRO: Rolling Segfault Capsule")
    log.info("=" * 80)
    log.info("")
    log.info(f"Context from last_good.json:")
    log.info(f"  Feature: {context.get('feature_name')}")
    log.info(f"  Function: {context.get('function_name')}")
    log.info(f"  Timestamp: {context.get('timestamp')}")
    log.info(f"  Index: {context.get('index')}")
    log.info(f"  Window: {context.get('window')}")
    log.info(f"  Min periods: {context.get('min_periods')}")
    log.info("")
    
    # Load data slice
    crash_idx = context.get("index", 0)
    if crash_idx is None:
        log.error("No crash index in context")
        return 1
    
    try:
        df_slice, adjusted_idx = load_data_slice(data_file, crash_idx, window_size)
        log.info(f"Loaded slice: {len(df_slice)} rows, adjusted crash_idx={adjusted_idx}")
    except Exception as e:
        log.error(f"Failed to load data slice: {e}", exc_info=True)
        return 1
    
    # Diagnose input series
    feature_name = context.get("feature_name", "")
    if "close" in df_slice.columns:
        close_series = df_slice["close"]
        diag = diagnose_series(close_series, "close")
        log.info(f"Series diagnostics: {json.dumps(diag, indent=2)}")
    
    # Try to reproduce the crash
    log.info("")
    log.info("Attempting to reproduce rolling operation...")
    log.info("")
    
    try:
        window = context.get("window", 20)
        min_periods = context.get("min_periods", 10)
        fn_name = context.get("function_name", "")
        
        if "_kama" in feature_name:
            log.info(f"Reproducing _kama(close, {window})...")
            if "close" not in df_slice.columns:
                log.error("'close' column not found in data slice")
                return 1
            result = _kama(df_slice["close"], window)
            log.info(f"✅ _kama completed: len={len(result)}")
            
        elif "_parkinson" in feature_name:
            log.info(f"Reproducing _parkinson_sigma(high, low)...")
            if "high" not in df_slice.columns or "low" not in df_slice.columns:
                log.error("'high' or 'low' column not found in data slice")
                return 1
            result = _parkinson_sigma(df_slice["high"], df_slice["low"])
            log.info(f"✅ _parkinson_sigma completed: len={len(result)}")
            
        elif "rsi" in feature_name or "bb" in feature_name or "comp3" in feature_name:
            log.info(f"Reproducing rolling operation: window={window}, min_periods={min_periods}...")
            if "close" not in df_slice.columns:
                log.error("'close' column not found in data slice")
                return 1
            # Try generic rolling
            if "mean" in fn_name:
                result = df_slice["close"].rolling(window, min_periods=min_periods).mean()
            elif "std" in fn_name:
                result = df_slice["close"].rolling(window, min_periods=min_periods).std(ddof=0)
            else:
                result = _roll(df_slice["close"], window, "mean", min_periods)
            log.info(f"✅ Rolling operation completed: len={len(result)}")
            
        else:
            log.warning(f"Unknown feature: {feature_name}, trying generic _roll...")
            if "close" not in df_slice.columns:
                log.error("'close' column not found in data slice")
                return 1
            result = _roll(df_slice["close"], window, "mean", min_periods)
            log.info(f"✅ Generic _roll completed: len={len(result)}")
        
        log.info("")
        log.info("=" * 80)
        log.info("✅ NO CRASH REPRODUCED - Operation completed successfully")
        log.info("=" * 80)
        log.info("")
        log.info("This suggests the segfault may be:")
        log.info("  1. Threading-related (try --safe-threading-env)")
        log.info("  2. Memory-related (large dataset vs small slice)")
        log.info("  3. State-dependent (previous operations affect crash)")
        log.info("")
        return 0
        
    except Exception as e:
        log.error(f"❌ Exception during reproduction: {e}", exc_info=True)
        return 1
    except SystemError as e:
        log.error(f"❌ SystemError (possible segfault): {e}", exc_info=True)
        return 1


def print_safe_threading_env():
    """Del C: Print recommended safe threading environment variables."""
    print("=" * 80)
    print("SAFE THREADING ENVIRONMENT VARIABLES")
    print("=" * 80)
    print("")
    print("To reduce native threading issues, set these before running:")
    print("")
    print("export OMP_NUM_THREADS=1")
    print("export MKL_NUM_THREADS=1")
    print("export OPENBLAS_NUM_THREADS=1")
    print("export VECLIB_MAXIMUM_THREADS=1")
    print("export NUMEXPR_MAX_THREADS=1")
    print("")
    print("Or in one line:")
    print("export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_MAX_THREADS=1")
    print("")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Reproduce pandas rolling segfault")
    parser.add_argument(
        "--capsule-dir",
        type=Path,
        default=Path("data/temp/segfault_capsule"),
        help="Directory containing last_good.json",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to data file (Parquet or CSV) used in original replay",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=2000,
        help="Number of rows before/after crash point to load",
    )
    parser.add_argument(
        "--safe-threading-env",
        action="store_true",
        help="Print recommended safe threading environment variables",
    )
    
    args = parser.parse_args()
    
    if args.safe_threading_env:
        print_safe_threading_env()
        return 0
    
    # Load last_good.json
    last_good_path = args.capsule_dir / "last_good.json"
    if not last_good_path.exists():
        log.error(f"last_good.json not found: {last_good_path}")
        log.error("Run a replay first to generate last_good.json on segfault")
        return 1
    
    with open(last_good_path, "r") as f:
        context = json.load(f)
    
    return repro_rolling_crash(context, args.data_file, args.window_size)


if __name__ == "__main__":
    sys.exit(main())


