#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay Evaluation for GATED_FUSION - Parallel Chunked Execution

Runs FULLYEAR replay with N workers, each processing a time chunk.
Each worker produces its own artifacts, which are merged at the end.
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add workspace root to path BEFORE importing gx1 modules
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# CRITICAL: Import dt_module for version stamping and fail-fast validation
from gx1.utils.dt_module import (
    get_dt_module_version,
    validate_dt_module_version,
    now_iso as dt_now_iso,
    strftime_now as dt_strftime_now,
)

import pandas as pd

# FASE 0.1: Import psutil for process detection
try:
    import psutil
except ImportError:
    psutil = None
    log = logging.getLogger(__name__)
    log.warning("[FASE_0] psutil not available - cannot detect parallel replays")

# DEL 2: Force spawn method for multiprocessing (avoid fork deadlocks)
if mp.get_start_method(allow_none=True) != "spawn":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# DEL 3: Thread library limits (OMP/MKL/OpenBLAS)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# FASE 1: DO NOT import GX1DemoRunner at top level - it imports live_features which imports basic_v1
# Import will happen in process_chunk() where workers have clean import state (spawn method)
# from gx1.execution.oanda_demo_runner import GX1DemoRunner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# DEL 2: Global flag for graceful shutdown (set by SIGTERM handler)
STOP_REQUESTED = False
MASTER_STOP_REQUESTED = False
POOL_REF = None  # Global reference to pool for SIGTERM handler
PERF_EXPORTED = False  # Global flag to prevent double export
PERF_EXPORT_LOCK = None  # Threading lock for perf export (initialized in main)


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=workspace_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def split_data_into_chunks(
    data_path: Path,
    n_chunks: int,
    slice_head: Optional[int] = None,
    days: Optional[int] = None,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Split data into N time-based chunks (no overlap).
    
    Args:
        data_path: Path to data file
        n_chunks: Number of chunks
        slice_head: If set, use only first N bars (deterministic)
        days: If set, use only first N days (deterministic, M5 = 288 bars/day)
        start_ts: If set, slice data to start from this timestamp (inclusive)
        end_ts: If set, slice data to end at this timestamp (inclusive)
    
    Returns list of (start_ts, end_ts, chunk_idx) tuples.
    """
    log.info(f"[PARALLEL] Loading data to determine chunks: {data_path}")
    df = pd.read_parquet(data_path)
    
    if len(df) == 0:
        raise ValueError(f"Data file is empty: {data_path}")
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df.index = pd.to_datetime(df["ts"])
        else:
            raise ValueError("Data must have datetime index or 'ts' column")
    
    df = df.sort_index()

    # Deterministic time-range slice (inclusive)
    if start_ts is not None or end_ts is not None:
        ts_start = start_ts if start_ts is not None else df.index[0]
        ts_end = end_ts if end_ts is not None else df.index[-1]
        
        # Ensure timezone-aware comparison (match index timezone)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            # Index is timezone-aware, make start_ts and end_ts timezone-aware too
            if ts_start is not None and ts_start.tz is None:
                ts_start = ts_start.tz_localize(df.index.tz)
            elif ts_start is not None and ts_start.tz is not None:
                ts_start = ts_start.tz_convert(df.index.tz)
            if ts_end is not None and ts_end.tz is None:
                ts_end = ts_end.tz_localize(df.index.tz)
            elif ts_end is not None and ts_end.tz is not None:
                ts_end = ts_end.tz_convert(df.index.tz)
        
        df = df.loc[ts_start:ts_end]
        log.info(f"[PARALLEL] Time-sliced to {ts_start}..{ts_end} (inclusive)")
    
    # Apply slicing if requested (deterministic: always use head)
    if days is not None:
        bars_per_day = 288  # M5
        slice_bars = days * bars_per_day
        df = df.head(slice_bars)
        log.info(f"[PARALLEL] Sliced to first {days} days ({slice_bars} bars)")
    elif slice_head is not None:
        df = df.head(slice_head)
        log.info(f"[PARALLEL] Sliced to first {slice_head} bars")
    
    if len(df) == 0:
        raise ValueError(f"Data is empty after slicing")
    
    start_ts = df.index[0]
    end_ts = df.index[-1]
    total_bars = len(df)
    
    log.info(
        f"[PARALLEL] Data range: {start_ts} to {end_ts} ({total_bars} bars, {n_chunks} chunks)"
    )
    
    # Calculate chunk boundaries (no overlap)
    chunk_size = total_bars // n_chunks
    chunks = []
    
    for i in range(n_chunks):
        chunk_start_idx = i * chunk_size
        chunk_end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else total_bars
        
        chunk_start_ts = df.index[chunk_start_idx]
        chunk_end_ts = df.index[chunk_end_idx - 1]  # Inclusive end
        
        chunks.append((chunk_start_ts, chunk_end_ts, i))
        
        log.info(
            f"[PARALLEL] Chunk {i}: {chunk_start_ts} to {chunk_end_ts} "
            f"({chunk_end_idx - chunk_start_idx} bars)"
        )
    
    return chunks


def _sigterm_handler(signum, frame):
    """DEL 2: SIGTERM handler for graceful shutdown (worker)."""
    global STOP_REQUESTED
    STOP_REQUESTED = True
    os.environ["GX1_STOP_REQUESTED"] = "1"
    log.warning(f"[TERM] Received SIGTERM (pid={os.getpid()}), will flush and exit gracefully")


def _master_sigterm_handler(signum, frame):
    """
    SIGTERM handler for master process (minimal work, watchdog thread does the rest).
    
    This handler:
    1. Sets MASTER_STOP_REQUESTED flag (watchdog thread polls this)
    2. Attempts to terminate pool (best effort, non-fatal)
    3. Does NOT export perf JSON (watchdog thread handles that)
    """
    global MASTER_STOP_REQUESTED, POOL_REF
    MASTER_STOP_REQUESTED = True
    log.warning(f"[MASTER] SIGTERM received -> STOP_REQUESTED=1 (watchdog will export perf JSON)")
    
    # Best effort: terminate pool if it exists
    if POOL_REF is not None:
        try:
            POOL_REF.terminate()
            log.info("[MASTER] Pool terminated (best effort)")
        except Exception:
            pass  # Non-fatal


def compute_bar_counters_snapshot(
    runner: Optional[Any],
    bars_processed: int,
    chunk_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """
    Compute standardized bar counter snapshot.
    
    Returns dict with:
        candles_iterated: Total candles iterated over (= bars_seen)
        warmup_skipped: Bars skipped due to warmup
        pregate_skipped: Bars skipped due to pregate/eligibility
        reached_entry_stage: Bars that reached entry stage (after warmup+pregate, before model)
        processed: Bars where model was called (evaluate_entry called)
    """
    # Get counters from runner if available
    if runner:
        candles_iterated = getattr(runner, "bars_seen", 0)
        warmup_skipped = getattr(runner, "bars_skipped_warmup", 0)
        pregate_skipped = getattr(runner, "bars_skipped_pregate", 0)
        reached_entry_stage = getattr(runner, "bars_reaching_entry_stage", 0)
    else:
        # Fallback: use chunk_df length as proxy
        candles_iterated = len(chunk_df) if chunk_df is not None else bars_processed
        warmup_skipped = 0
        pregate_skipped = 0
        reached_entry_stage = 0
    
    # processed = bars where model was called (evaluate_entry called)
    # This is bars_processed in the current code
    processed = bars_processed
    
    return {
        "candles_iterated": candles_iterated,
        "warmup_skipped": warmup_skipped,
        "pregate_skipped": pregate_skipped,
        "reached_entry_stage": reached_entry_stage,
        "processed": processed,
    }


def process_chunk(
    chunk_idx: int,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    data_path: Path,
    policy_path: Path,
    run_id: str,
    output_dir: Path,
    bundle_sha256: Optional[str] = None,
    prebuilt_parquet_path: Optional[str] = None,  # DEL 3: Force string, not Path
    bundle_dir: Optional[Path] = None,  # Bundle directory override
) -> Dict[str, Any]:
    """
    Process a single chunk in a worker process.
    
    DEL 1: Wrapped in try/finally to guarantee flush even on exceptions.
    DEL 2: SIGTERM handler for graceful shutdown.
    
    Returns dict with chunk results and paths to artifacts.
    """
    # Initialize variables that may be needed in exception handler
    chunk_output_dir = None
    runner = None
    chunk_df = None
    bars_processed = 0
    first_iter_ts = None
    last_iter_ts = None
    policy_id = None
    run_identity_data = None
    
    # DEL 1: Wrap entire function in try/except to catch ALL exceptions (including import errors)
    try:
        # DEL A: Minimal, korrekt fiks - WORKER_BOOT.json som første linje
        # Resolve chunk_dir first (must be absolute)
        chunk_output_dir = (output_dir / f"chunk_{chunk_idx}").resolve()
        
        # DEL A: Write WORKER_BOOT.json as FIRST line in process_chunk (hard fail if can't write)
        import os as os_module
        import sys
        from pathlib import Path
        import json as json_module
        # CRITICAL: Import dt_module for version stamping
        from gx1.utils.dt_module import (
            get_dt_module_version,
            validate_dt_module_version,
            now_iso as dt_now_iso,
        )
        
        # CRITICAL: Validate dt_module version immediately (fail-fast)
        validate_dt_module_version()
        dt_module_version = get_dt_module_version()
        
        worker_boot_payload = {
            "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
            "timestamp": dt_now_iso(),
            "pid": os_module.getpid(),
            "ppid": os_module.getppid() if hasattr(os_module, 'getppid') else None,
            "cwd": str(Path.cwd()),
            "sys_executable": sys.executable,
            "argv_snapshot": sys.argv.copy() if hasattr(sys, 'argv') else None,
            "prebuilt_parquet_path_raw": str(prebuilt_parquet_path) if prebuilt_parquet_path else None,
            "prebuilt_parquet_path_resolved": None,
            "prebuilt_exists": False,
            "prebuilt_size": 0,
            "chunk_output_dir": str(chunk_output_dir),
        }
        
        # Try to resolve prebuilt path (may fail, but log what we can)
        if prebuilt_parquet_path:
            try:
                prebuilt_resolved = str(Path(prebuilt_parquet_path).resolve())
                worker_boot_payload["prebuilt_parquet_path_resolved"] = prebuilt_resolved
                prebuilt_path_obj = Path(prebuilt_resolved)
                worker_boot_payload["prebuilt_exists"] = prebuilt_path_obj.exists()
                if worker_boot_payload["prebuilt_exists"]:
                    worker_boot_payload["prebuilt_size"] = prebuilt_path_obj.stat().st_size
            except Exception as resolve_error:
                worker_boot_payload["prebuilt_resolve_error"] = str(resolve_error)
        
        # DEL A: Hard fail if can't write WORKER_BOOT.json (no try/except, no fallback)
        # Ensure chunk dir exists (should already exist from master pre-create, but be safe)
        chunk_output_dir.mkdir(parents=True, exist_ok=True)
        
        worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
        with open(worker_boot_path, "w") as f:
            json_module.dump(worker_boot_payload, f, indent=2)
            f.flush()  # Force write to OS buffer
            os_module.fsync(f.fileno())  # Force write to disk
        
        global STOP_REQUESTED
        
        # DEL 2: Install SIGTERM handler for graceful shutdown
        signal.signal(signal.SIGTERM, _sigterm_handler)
        STOP_REQUESTED = False
        
        worker_start_time = time.time()
        
        # DEL 1: Write WORKER_START.json immediately (before any processing)
        worker_start_info = {
            "chunk_id": chunk_idx,
            "prebuilt_parquet_path_raw": str(prebuilt_parquet_path) if prebuilt_parquet_path else None,
            "prebuilt_parquet_path_raw_type": type(prebuilt_parquet_path).__name__ if prebuilt_parquet_path else "None",
            "prebuilt_parquet_path_resolved": None,
            "exists": False,
            "size": 0,
            "exception_full": None,
        }
        
        # DEL 3: Force string (not Path) - FATAL if wrong type
        if prebuilt_parquet_path is not None:
            if not isinstance(prebuilt_parquet_path, str):
                fatal_msg = (
                    f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] prebuilt_parquet_path must be str, got {type(prebuilt_parquet_path)}. "
                    f"repr: {repr(prebuilt_parquet_path)}"
                )
                worker_start_info["exception_full"] = fatal_msg
                worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                with open(worker_start_json_path, "w") as f:
                    import json
                    json.dump(worker_start_info, f, indent=2)
                fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
                with open(fatal_error_path, "w") as f:
                    f.write(fatal_msg + "\n")
                    f.write(f"prebuilt_parquet_path: {repr(prebuilt_parquet_path)}\n")
                    f.write(f"type: {type(prebuilt_parquet_path)}\n")
                raise RuntimeError(fatal_msg)
        
        # DEL A: SSoT logging - Root cause analysis
        import sys
        cwd = str(Path.cwd())
        python_exe = sys.executable
        output_dir_env = os_module.getenv("GX1_OUTPUT_DIR", "NOT_SET")
        prebuilt_env = os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "NOT_SET")
        prebuilt_path_env = os_module.getenv("GX1_REPLAY_PREBUILT_FEATURES_PATH", "NOT_SET")
        
        # DEL 4: Determine prebuilt path (explicit arg takes precedence, FATAL if missing)
        if prebuilt_parquet_path:
            # DEL 3: Already a string, resolve to absolute
            prebuilt_root = str(Path(prebuilt_parquet_path).resolve())
        else:
            # DEL 4: FATAL if arg missing (we expect explicit path in this code path)
            fatal_msg = (
                f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] prebuilt_parquet_path arg is None/empty. "
                f"env GX1_REPLAY_PREBUILT_FEATURES_PATH={prebuilt_path_env}"
            )
            worker_start_info["exception_full"] = fatal_msg
            worker_start_json_path = chunk_output_dir / "WORKER_START.json"
            with open(worker_start_json_path, "w") as f:
                import json
                json.dump(worker_start_info, f, indent=2)
            fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
            with open(fatal_error_path, "w") as f:
                f.write(fatal_msg + "\n")
                f.write(f"prebuilt_parquet_path (arg): {prebuilt_parquet_path}\n")
                f.write(f"GX1_REPLAY_PREBUILT_FEATURES_PATH (env): {prebuilt_path_env}\n")
            raise RuntimeError(fatal_msg)
        
        prebuilt_parquet_path_resolved = prebuilt_root
        
        # DEL 1: Update worker_start_info with resolved path
        prebuilt_path_obj = Path(prebuilt_parquet_path_resolved)
        worker_start_info["prebuilt_parquet_path_resolved"] = prebuilt_parquet_path_resolved
        worker_start_info["exists"] = prebuilt_path_obj.exists()
        if worker_start_info["exists"]:
            worker_start_info["size"] = prebuilt_path_obj.stat().st_size
        
        # DEL 5: Import logging NOW (before we use log)
        import logging as logging_module
        log = logging_module.getLogger(__name__)
        
        # Log SSoT info
        log.info(f"[CHUNK {chunk_idx}] [SSoT] Worker start diagnostics:")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   cwd = {cwd}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   sys.executable = {python_exe}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   GX1_OUTPUT_DIR = {output_dir_env}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   GX1_REPLAY_USE_PREBUILT_FEATURES = {prebuilt_env}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   GX1_REPLAY_PREBUILT_FEATURES_PATH (env) = {prebuilt_path_env}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   prebuilt_parquet_path (arg) = {prebuilt_parquet_path}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   prebuilt_parquet_path_resolved = {prebuilt_parquet_path_resolved}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   exists = {worker_start_info['exists']}, size = {worker_start_info['size']}")
        
        # DEL B: Validate prebuilt path (FATAL if missing when prebuilt enabled)
        prebuilt_enabled = os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
        if prebuilt_enabled:
            # DEL B: Invariant - path must be absolute
            if not prebuilt_path_obj.is_absolute():
                fatal_msg = (
                    f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Prebuilt path is not absolute: {prebuilt_parquet_path_resolved}"
                )
                worker_start_info["exception_full"] = fatal_msg
                worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                with open(worker_start_json_path, "w") as f:
                    import json
                    json.dump(worker_start_info, f, indent=2)
                fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
                with open(fatal_error_path, "w") as f:
                    f.write(fatal_msg + "\n")
                    f.write(f"prebuilt_parquet_path_resolved: {prebuilt_parquet_path_resolved}\n")
                    f.write(f"is_absolute(): {prebuilt_path_obj.is_absolute()}\n")
                raise RuntimeError(fatal_msg)
            
            # Check exists() and size
            if not worker_start_info["exists"]:
                fatal_msg = (
                    f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Prebuilt parquet file does not exist: {prebuilt_parquet_path_resolved}"
                )
                worker_start_info["exception_full"] = fatal_msg
                worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                with open(worker_start_json_path, "w") as f:
                    import json
                    json.dump(worker_start_info, f, indent=2)
                fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
                with open(fatal_error_path, "w") as f:
                    f.write(fatal_msg + "\n")
                    f.write(f"prebuilt_parquet_path_resolved: {prebuilt_parquet_path_resolved}\n")
                    f.write(f"exists(): False\n")
                    f.write(f"cwd: {cwd}\n")
                raise FileNotFoundError(fatal_msg)
            
            if worker_start_info["size"] == 0:
                fatal_msg = (
                    f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Prebuilt parquet file is empty (size=0): {prebuilt_parquet_path_resolved}"
                )
                worker_start_info["exception_full"] = fatal_msg
                worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                with open(worker_start_json_path, "w") as f:
                    import json
                    json.dump(worker_start_info, f, indent=2)
                fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
                with open(fatal_error_path, "w") as f:
                    f.write(fatal_msg + "\n")
                    f.write(f"prebuilt_parquet_path_resolved: {prebuilt_parquet_path_resolved}\n")
                    f.write(f"stat().st_size: {worker_start_info['size']}\n")
                raise RuntimeError(fatal_msg)
            
            log.info(f"[CHUNK {chunk_idx}] [SSoT]   prebuilt_parquet_path_resolved.exists() = True")
            log.info(f"[CHUNK {chunk_idx}] [SSoT]   prebuilt_parquet_path_resolved.stat().st_size = {worker_start_info['size']:,} bytes")
            
            # DEL 4: Set environment variable for runner (backward compatibility), but log that we use arg
            os_module.environ["GX1_REPLAY_PREBUILT_FEATURES_PATH"] = prebuilt_parquet_path_resolved
            log.info(f"[CHUNK {chunk_idx}] [SSoT] Using prebuilt_parquet_path from arg (not env): {prebuilt_parquet_path_resolved}")
        
        # DEL 5: Flytt tunge imports inni try etter at WORKER_BOOT.json er skrevet
        # (for å fange import-krasj)
        # Import logging first (before other imports that might use log)
        import logging as logging_module
        log = logging_module.getLogger(__name__)
        
        try:
            import pandas as pd
        except Exception as import_error:
            # If imports fail, write error to WORKER_BOOT.json if we wrote it
            if worker_boot_written:
                try:
                    worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
                    with open(worker_boot_path, "r") as f:
                        boot_data = json_module.load(f)
                    boot_data["import_error"] = str(import_error)
                    boot_data["import_error_type"] = type(import_error).__name__
                    with open(worker_boot_path, "w") as f:
                        json_module.dump(boot_data, f, indent=2)
                except Exception:
                    pass
            raise
        
        # Initialize status tracking
        status = "ok"
        error = None
        error_traceback = None
        runner = None
        chunk_df = None
        n_model_calls = 0
        n_trades_closed = 0
        bars_processed = 0
        last_checkpoint_bars = 0
        CHECKPOINT_EVERY_BARS = 1000  # DEL 3: Periodic checkpoint flush
        
        # Initialize entry stage telemetry (before runner is created)
        bars_seen = 0
        bars_skipped_warmup = 0
        bars_skipped_pregate = 0
        bars_reaching_entry_stage = 0
        pregate_enabled = False
        
        # FIX: Initialize all footer variables to prevent "referenced before assignment"
        total_bars = 0
        n_trades_created = 0
        feature_time_total = 0.0
        feature_time_mean_ms = 0.0
        feature_timeout_count = 0
        htf_align_warn_count = 0
        htf_align_time_total_sec = 0.0
        htf_align_call_count = 0
        htf_align_warning_time_sec = 0.0
        htf_align_fallback_count = 0
        htf_feature_compute_bars = 0
        pregate_skips = 0
        pregate_passes = 0
        pregate_missing_inputs = 0
        t_pregate_total_sec = 0.0
        t_feature_build_total_sec = 0.0
        t_model_total_sec = 0.0
        t_policy_total_sec = 0.0
        t_io_total_sec = 0.0
        wall_clock_sec = 0.0
        htf_h1_calls = 0
        htf_h4_calls = 0
        htf_h1_warns = 0
        htf_h4_warns = 0
        htf_last_m5_ts = None
        htf_last_j = None
        vol_regime_unknown_count = 0
        
        log.info(
            f"[CHUNK {chunk_idx}] Starting: {chunk_start} to {chunk_end}"
        )
        
        # Inner try for replay loop (nested in outer try)
        try:
            # Load chunk data
            df_full = pd.read_parquet(data_path)
            if not isinstance(df_full.index, pd.DatetimeIndex):
                if "ts" in df_full.columns:
                    df_full.index = pd.to_datetime(df_full["ts"])
                else:
                    raise ValueError("Data must have datetime index or 'ts' column")
            
            df_full = df_full.sort_index()
            
            # Filter to chunk time range
            chunk_df = df_full[(df_full.index >= chunk_start) & (df_full.index <= chunk_end)]
            
            if len(chunk_df) == 0:
                raise ValueError(f"Chunk {chunk_idx} is empty after filtering")
            
            bars_processed = len(chunk_df)
            log.info(f"[CHUNK {chunk_idx}] Loaded {bars_processed} bars")
            
            # DEL 4: Guard - verify chunk_df has OHLC before any processing
            required_cols_check = ["open", "high", "low", "close"]
            missing_before_save = [c for c in required_cols_check if c not in chunk_df.columns]
            if missing_before_save:
                raise RuntimeError(
                    f"DEL 4 GUARD: chunk_df missing OHLC columns before save: {missing_before_save}. "
                    f"Available: {list(chunk_df.columns)}"
                )
            
            # Save chunk data to temp file
            # CRITICAL: Reset index to column "time" to match _run_replay_impl expectations
            chunk_df_save = chunk_df.reset_index()
            if "time" not in chunk_df_save.columns and len(chunk_df_save.columns) > 0:
                # If index was DatetimeIndex, rename it to "time"
                chunk_df_save.rename(columns={chunk_df_save.columns[0]: "time"}, inplace=True)
            
            # DEL 4: Guard - verify chunk_df_save still has OHLC after reset_index
            missing_after_reset = [c for c in required_cols_check if c not in chunk_df_save.columns]
            if missing_after_reset:
                raise RuntimeError(
                    f"DEL 4 GUARD: chunk_df_save missing OHLC columns after reset_index: {missing_after_reset}. "
                    f"Available: {list(chunk_df_save.columns)}"
                )
            
            # DEL 6: Check for case-insensitive column collisions (close/CLOSE)
            # Apply compat-mode resolution if enabled (explicit env var only, not default)
            from gx1.runtime.column_collision_guard import assert_no_case_collisions, resolve_close_alias_collision
            resolution = assert_no_case_collisions(
                df=chunk_df_save,
                context=f"chunk_{chunk_idx}_data.parquet (before write)",
                allow_close_alias_compat=False,  # DEL 3: No default - must be explicit GX1_ALLOW_CLOSE_ALIAS_COMPAT=1
            )
            
            # Store resolution metadata for chunk_footer.json
            case_collision_resolution = None
            if resolution:
                # Apply resolution: drop CLOSE column
                log.warning(
                    f"[CHUNK {chunk_idx}] [COMPAT] Dropped CLOSE column due to collision with candles.close. "
                    "Alias expected: CLOSE -> candles.close"
                )
                chunk_df_save, resolution_meta = resolve_close_alias_collision(
                    df=chunk_df_save,
                    context=f"chunk_{chunk_idx}_data.parquet",
                    transformer_requires_close=False,
                )
                case_collision_resolution = resolution_meta
                log.info(
                    f"[CHUNK {chunk_idx}] [COMPAT] Resolution applied: {resolution_meta}"
                )
            
            # DEL 5: Atomic file write (temp -> final)
            chunk_data_path = chunk_output_dir / f"chunk_{chunk_idx}_data.parquet"
            chunk_data_path_tmp = chunk_output_dir / f"chunk_{chunk_idx}_data.parquet.tmp"
            
            # Write to temp file first
            chunk_df_save.to_parquet(chunk_data_path_tmp)
            
            # DEL 2: Calculate file hash/size for verification
            chunk_data_path_abs = chunk_data_path.resolve()
            chunk_data_path_tmp_abs = chunk_data_path_tmp.resolve()
            
            if chunk_data_path_tmp_abs.exists():
                file_size = chunk_data_path_tmp_abs.stat().st_size
                file_mtime = chunk_data_path_tmp_abs.stat().st_mtime
                # Calculate SHA256
                with open(chunk_data_path_tmp_abs, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
            else:
                file_size = 0
                file_mtime = 0
                file_hash = None
            
            # Atomic rename
            chunk_data_path_tmp_abs.rename(chunk_data_path_abs)
            
            log.info(
                f"[CHUNK {chunk_idx}] Saved chunk data to {chunk_data_path_abs} "
                f"(size={file_size}, mtime={file_mtime}, sha256={file_hash[:16] if file_hash else 'N/A'}...)"
            )
            
            # Set environment variables for this worker
            os_module.environ["GX1_GATED_FUSION_ENABLED"] = "1"
            # Respect existing GX1_REQUIRE_XGB_CALIBRATION if set, otherwise default to "1"
            if "GX1_REQUIRE_XGB_CALIBRATION" not in os_module.environ:
                os_module.environ["GX1_REQUIRE_XGB_CALIBRATION"] = "1"
            os_module.environ["GX1_REPLAY_INCREMENTAL_FEATURES"] = "1"
            os_module.environ["GX1_REPLAY_NO_CSV"] = "1"
            os_module.environ["GX1_FEATURE_USE_NP_ROLLING"] = "1"
            os_module.environ["GX1_RUN_ID"] = run_id
            os_module.environ["GX1_CHUNK_ID"] = str(chunk_idx)
            
            # Set thread limits (already set globally, but ensure in worker)
            os_module.environ["OMP_NUM_THREADS"] = "1"
            os_module.environ["MKL_NUM_THREADS"] = "1"
            os_module.environ["OPENBLAS_NUM_THREADS"] = "1"
            os_module.environ["VECLIB_MAXIMUM_THREADS"] = "1"
            os_module.environ["NUMEXPR_NUM_THREADS"] = "1"
            
            # DEL 5: Worker self-test before replay (minimal pyarrow-only)
            if prebuilt_enabled and prebuilt_parquet_path_resolved:
                log.info(f"[CHUNK {chunk_idx}] [SELF_TEST] Running prebuilt smoke test (pyarrow-only)...")
            try:
                import pyarrow.parquet as pq
                import traceback
                
                # DEL D: Retry logic for file locks (max 3 attempts with jitter)
                max_retries = 3
                retry_delay_base = 0.1  # 100ms base
                smoke_test_passed = False
                last_error = None
                error_type = None
                
                for attempt in range(max_retries):
                    try:
                        # Add jitter (0-200ms)
                        import random
                        jitter_ms = random.randint(0, 200)
                        if attempt > 0:
                            time.sleep(retry_delay_base + (jitter_ms / 1000.0))
                        
                        # DEL 5: Minimal pyarrow test (no pandas)
                        # DEL 6: Disable memory_map for concurrency safety
                        parquet_file = pq.ParquetFile(
                            prebuilt_parquet_path_resolved,
                            memory_map=False  # DEL 6: Disable mmap for parallel access
                        )
                        
                        # Read metadata
                        metadata = parquet_file.metadata
                        num_row_groups = metadata.num_row_groups
                        if num_row_groups == 0:
                            raise RuntimeError("Parquet file has 0 row groups")
                        
                        # Read first row from first row group
                        first_row_group = parquet_file.read_row_group(0)
                        num_rows = len(first_row_group)
                        if num_rows == 0:
                            raise RuntimeError("First row group is empty")
                        
                        # Get first row (just verify we can read it)
                        first_row = first_row_group.slice(0, 1)
                        
                        log.info(
                            f"[CHUNK {chunk_idx}] [SELF_TEST] ✅ Smoke test passed (pyarrow): "
                            f"row_groups={num_row_groups}, first_row_group_rows={num_rows}"
                        )
                        smoke_test_passed = True
                        break
                        
                    except FileNotFoundError as e:
                        error_type = "file_not_found"
                        last_error = e
                        log.warning(
                            f"[CHUNK {chunk_idx}] [SELF_TEST] Attempt {attempt + 1}/{max_retries} failed (file_not_found): {e}"
                        )
                        if attempt == max_retries - 1:
                            break
                    except PermissionError as e:
                        error_type = "permission_denied"
                        last_error = e
                        log.warning(
                            f"[CHUNK {chunk_idx}] [SELF_TEST] Attempt {attempt + 1}/{max_retries} failed (permission_denied): {e}"
                        )
                        if attempt == max_retries - 1:
                            break
                    except Exception as e:
                        # Check if it's a parquet decode / arrow error
                        error_str = str(e).lower()
                        if "parquet" in error_str or "arrow" in error_str or "decode" in error_str:
                            error_type = "parquet_decode"
                        else:
                            error_type = "unknown"
                        last_error = e
                        log.warning(
                            f"[CHUNK {chunk_idx}] [SELF_TEST] Attempt {attempt + 1}/{max_retries} failed ({error_type}): {e}"
                        )
                        if attempt == max_retries - 1:
                            break
                
                if not smoke_test_passed:
                    # Final attempt failed
                    fatal_msg = (
                        f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Smoke test failed after {max_retries} attempts. "
                        f"Error type: {error_type}, Last error: {last_error}"
                    )
                    worker_start_info["exception_full"] = f"{error_type}: {str(last_error)}\n{traceback.format_exc()}"
                    worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                    with open(worker_start_json_path, "w") as f:
                        import json
                        json.dump(worker_start_info, f, indent=2)
                    fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
                    with open(fatal_error_path, "w") as f:
                        f.write(fatal_msg + "\n")
                        f.write(f"Error type: {error_type}\n")
                        f.write(f"Last error: {last_error}\n")
                        f.write(f"Traceback:\n{traceback.format_exc()}\n")
                    raise RuntimeError(fatal_msg)
                
                # DEL 7: If --selftest-only, exit after smoke test
                selftest_only = os_module.getenv("GX1_SELFTEST_ONLY", "0") == "1"
                if selftest_only:
                    log.info(f"[CHUNK {chunk_idx}] [SELF_TEST] --selftest-only: Exiting after smoke test (PASS)")
                    # Write minimal footer to indicate success
                    from datetime import datetime as dt_module
                    stub_footer = {
                        "run_id": run_id,
                        "chunk_id": str(chunk_idx),
                        "status": "selftest_ok",
                        "smoke_test_passed": True,
                        "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
                        "timestamp": dt_now_iso(),
                    }
                    chunk_footer_path = chunk_output_dir / "chunk_footer.json"
                    with open(chunk_footer_path, "w") as f:
                        import json
                        json.dump(stub_footer, f, indent=2)
                    return {
                        "chunk_idx": chunk_idx,
                        "status": "selftest_ok",
                        "n_bars": 0,
                        "n_model_calls": 0,
                        "n_trades_closed": 0,
                    }
                    
            except Exception as e:
                import traceback
                fatal_msg = (
                    f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Worker self-test failed: {e}"
                )
                worker_start_info["exception_full"] = f"{str(e)}\n{traceback.format_exc()}"
                worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                with open(worker_start_json_path, "w") as f:
                    import json
                    json.dump(worker_start_info, f, indent=2)
                fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
                with open(fatal_error_path, "w") as f:
                    f.write(fatal_msg + "\n")
                    f.write(f"prebuilt_parquet_path_resolved: {prebuilt_parquet_path_resolved}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Full traceback:\n{traceback.format_exc()}\n")
                    f.write(f"cwd: {cwd}\n")
                raise
                    
            except Exception as e:
                import traceback
                fatal_msg = (
                    f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] Worker self-test failed: {e}"
                )
                worker_start_info["exception_full"] = f"{str(e)}\n{traceback.format_exc()}"
                worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                with open(worker_start_json_path, "w") as f:
                    import json
                    json.dump(worker_start_info, f, indent=2)
                fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
                with open(fatal_error_path, "w") as f:
                    f.write(fatal_msg + "\n")
                    f.write(f"prebuilt_parquet_path_resolved: {prebuilt_parquet_path_resolved}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Full traceback:\n{traceback.format_exc()}\n")
                    f.write(f"cwd: {cwd}\n")
                raise
            
            # FASE 1: Import GX1DemoRunner here (in worker process) to avoid importing live_features in master
            # With spawn method, workers have clean import state, so this won't affect master's sys.modules
            from gx1.execution.oanda_demo_runner import GX1DemoRunner
            
            # Set GX1_BUNDLE_DIR env var if bundle_dir is provided (overrides policy bundle_dir)
            # This must be set BEFORE creating GX1DemoRunner, as it reads GX1_BUNDLE_DIR during init
            if bundle_dir:
                import os
                bundle_dir_resolved = bundle_dir.resolve() if hasattr(bundle_dir, 'resolve') else Path(bundle_dir).resolve()
                os.environ["GX1_BUNDLE_DIR"] = str(bundle_dir_resolved)
                log.info(f"[CHUNK {chunk_idx}] Set GX1_BUNDLE_DIR={bundle_dir_resolved}")
            else:
                log.warning(f"[CHUNK {chunk_idx}] No bundle_dir provided - will use policy bundle_dir")
            
            # Create runner for this chunk
            runner = GX1DemoRunner(
                policy_path,
                replay_mode=True,
                fast_replay=False,
                output_dir=chunk_output_dir,
            )
            
            # Set run_id and chunk_id
            runner.run_id = run_id
            runner.chunk_id = str(chunk_idx)
            
            # Set bundle_sha256 from args (computed in master before workers start)
            if bundle_sha256:
                runner.bundle_sha256_from_master = bundle_sha256
            else:
                raise RuntimeError(
                    "[SSOT_FAIL] bundle_sha256 is missing in process_chunk. "
                    "This should never happen - bundle_sha256 must be computed before workers start."
                )
            
            # DEL 2: Verify worker reads correct file
            chunk_data_path_abs = chunk_data_path.resolve()
            
            # DEL 5: Wait for file to be stable (not .tmp, exists, size stable)
            max_wait = 10
            wait_interval = 0.5
            waited = 0
            while waited < max_wait:
                tmp_path = chunk_data_path_abs.parent / f"{chunk_data_path_abs.name}.tmp"
                if not tmp_path.exists() and chunk_data_path_abs.exists():
                    # Check size stability (2 checks)
                    size1 = chunk_data_path_abs.stat().st_size
                    time.sleep(0.1)
                    size2 = chunk_data_path_abs.stat().st_size
                    if size1 == size2 and size1 > 0:
                        break
                time.sleep(wait_interval)
                waited += wait_interval
            else:
                raise RuntimeError(
                    f"DEL 5: Chunk file not stable after {max_wait}s: "
                    f"tmp_exists={tmp_path.exists()}, final_exists={chunk_data_path_abs.exists()}"
                )
            
            # DEL 2: Verify path is absolute and matches what we wrote
            if not chunk_data_path_abs.is_absolute():
                raise RuntimeError(f"DEL 2: Chunk path is not absolute: {chunk_data_path_abs}")
            
            # DEL 2: Assert file exists and has size > 0
            if not chunk_data_path_abs.exists():
                raise RuntimeError(f"DEL 2: Chunk file does not exist: {chunk_data_path_abs}")
            
            file_size_check = chunk_data_path_abs.stat().st_size
            if file_size_check == 0:
                raise RuntimeError(f"DEL 2: Chunk file is empty: {chunk_data_path_abs}")
            
            log.info(
                f"[CHUNK {chunk_idx}] DEL 2: Verified chunk file: {chunk_data_path_abs} "
                f"(size={file_size_check}, matches expected)"
            )
            
            # DEL 2: Set STOP_REQUESTED flag in runner and env for bar loop checks
            runner._stop_requested = False
            os_module.environ["GX1_STOP_REQUESTED"] = "0"
            os_module.environ["GX1_CHECKPOINT_EVERY_BARS"] = str(CHECKPOINT_EVERY_BARS)
            
            # FASE 5: Quiet mode removed - all errors must be visible
            
            # DEL C: Filter sklearn warnings (UserWarning spam)
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
            warnings.filterwarnings("ignore", category=UserWarning, message="Loky-backed")
            
            # DEL 2: Update env var when SIGTERM is received (via handler)
            def update_stop_flag():
                """Update env var when STOP_REQUESTED changes."""
                if STOP_REQUESTED:
                    os_module.environ["GX1_STOP_REQUESTED"] = "1"
                    runner._stop_requested = True
            
            # Fast abort mode: set env var for abort-after-N-bars-per-chunk
            abort_after_n_bars = os_module.getenv("GX1_ABORT_AFTER_N_BARS_PER_CHUNK")
            if abort_after_n_bars:
                try:
                    abort_after_n_bars_int = int(abort_after_n_bars)
                    os_module.environ["GX1_ABORT_AFTER_N_BARS_PER_CHUNK"] = str(abort_after_n_bars_int)
                    log.info(f"[CHUNK {chunk_idx}] Fast abort mode: will stop after {abort_after_n_bars_int} bars")
                except ValueError:
                    log.warning(f"[CHUNK {chunk_idx}] Invalid GX1_ABORT_AFTER_N_BARS_PER_CHUNK: {abort_after_n_bars}")
            
            # Run replay for this chunk
            # DEL 3: Checkpoint flush is handled inside bar loop via env var
            # Entry stage telemetry: track bars directly from runner (fallback to local if runner telemetry missing)
            try:
                runner.run_replay(chunk_data_path_abs)
            except KeyboardInterrupt:
                log.warning(f"[CHUNK {chunk_idx}] Interrupted (KeyboardInterrupt)")
                status = "stopped"
                error = "Interrupted by KeyboardInterrupt"
            
            # DEL 2: Check if we were stopped
            if STOP_REQUESTED:
                log.warning(f"[CHUNK {chunk_idx}] STOP_REQUESTED flag is True, exiting early")
                status = "stopped"
                error = "Stopped early due to SIGTERM"
            
            # Extract metrics
            n_model_calls = getattr(runner, "perf_n_bars_processed", 0)
            # CRITICAL: bars_processed is for model calls (after warmup/pregate)
            # bars_iterated is total bars in subset (all bars loop iterated over)
            bars_processed = n_model_calls
            bars_iterated = n_model_calls  # Will be updated if loop completed all bars
            wall_clock_sec = time.time() - worker_start_time  # DEL 2: Always define wall_clock_sec (even on early failure)
            
            # Extract entry stage telemetry (where bars disappear)
            # Hard-check: verify telemetry attributes exist on runner (since perf_n_bars_processed works)
            if runner:
                # Required telemetry attributes (must exist if bar loop ran)
                required_attrs = ["bars_seen", "bars_skipped_warmup", "bars_skipped_pregate", "bars_reaching_entry_stage"]
                missing_attrs = [attr for attr in required_attrs if not hasattr(runner, attr)]
                
                if missing_attrs:
                    # Diagnostic: show what attributes are available
                    available_attrs = [x for x in dir(runner) if x.startswith(('bars_', 'perf_')) and not x.startswith('__')]
                    raise RuntimeError(
                        f"[CHUNK {chunk_idx}] TELEMETRY_MISSING: Required telemetry attributes missing on runner: {missing_attrs}. "
                        f"Runner type: {type(runner).__name__}. "
                        f"Available bars_/perf_ attributes: {available_attrs}. "
                        f"This indicates telemetry was not initialized in bar loop (but perf_n_bars_processed={bars_processed} suggests loop ran)."
                    )
                
                # Direct attribute access (hard-check passed, attributes exist)
                bars_seen = runner.bars_seen
                bars_skipped_warmup = runner.bars_skipped_warmup
                bars_skipped_pregate = runner.bars_skipped_pregate
                bars_reaching_entry_stage = runner.bars_reaching_entry_stage
                pregate_enabled = getattr(runner, "pregate_enabled", False)
            else:
                bars_seen = None
                bars_skipped_warmup = None
                bars_skipped_pregate = None
                bars_reaching_entry_stage = None
                pregate_enabled = False
            
            # If runner telemetry is None, use chunk_df length as proxy for bars_seen
            # This handles the case where runner telemetry wasn't initialized
            # bars_processed represents bars that reached the processing stage (after warmup)
            if bars_seen is None:
                # Use len(chunk_df) as proxy for total bars seen (all bars in chunk)
                if chunk_df is not None:
                    bars_seen = len(chunk_df)
                else:
                    # Fallback: use bars_processed (at least this many bars were seen)
                    bars_seen = bars_processed
            if bars_skipped_warmup is None:
                bars_skipped_warmup = 0  # Unknown, but at least 0
            if bars_skipped_pregate is None:
                bars_skipped_pregate = 0  # Unknown, but at least 0
            if bars_reaching_entry_stage is None:
                bars_reaching_entry_stage = 0  # Unknown, but at least 0
            
            # CRITICAL FIX: total_bars must be total bars in subset (bars_iterated), not bars_processed (after warmup)
            # bars_iterated = total bars loop iterated over (should equal len(chunk_df))
            # bars_processed = bars that reached model call stage (after warmup/pregate skips)
            # 
            # Check if loop completed all bars by comparing bars_seen to chunk_df length
            bars_seen_from_runner = getattr(runner, "bars_seen", None)
            if bars_seen_from_runner is not None and chunk_df is not None:
                # bars_seen is total bars loop iterated over (including warmup skips)
                bars_iterated = bars_seen_from_runner
                expected_bars = len(chunk_df)
                if bars_iterated < expected_bars:
                    # EARLY STOP DETECTED: Loop stopped before completing all bars
                    log.error(
                        f"[CHUNK {chunk_idx}] FATAL: Early stop before end of subset. "
                        f"bars_iterated={bars_iterated}, expected_bars={expected_bars}, diff={expected_bars - bars_iterated}"
                    )
                    # Hard-fail if not timeout/stop requested
                    if not STOP_REQUESTED and status == "ok":
                        error = f"Early stop: bars_iterated ({bars_iterated}) < expected_bars ({expected_bars})"
                        status = "failed"
                elif bars_iterated == expected_bars:
                    log.info(f"[CHUNK {chunk_idx}] ✅ Loop completed all bars: bars_iterated={bars_iterated} == expected_bars={expected_bars}")
                else:
                    log.warning(f"[CHUNK {chunk_idx}] ⚠️  bars_iterated ({bars_iterated}) > expected_bars ({expected_bars})")
                    bars_iterated = expected_bars  # Cap at expected
            else:
                # Fallback: use bars_processed (may be inaccurate if loop stopped early)
                bars_iterated = bars_processed
                if chunk_df is not None:
                    expected_bars = len(chunk_df)
                    if bars_iterated < expected_bars:
                        log.warning(
                            f"[CHUNK {chunk_idx}] ⚠️  Cannot verify completion (bars_seen not available). "
                            f"bars_processed={bars_processed}, expected_bars={expected_bars}"
                        )
            
            # total_bars = bars_iterated (total bars in subset, not bars_processed)
            total_bars = bars_iterated
            
            # Log when approaching end (100 bars before end)
            if chunk_df is not None:
                expected_bars = len(chunk_df)
                if bars_iterated >= expected_bars - 100 and bars_iterated < expected_bars:
                    log.info(
                        f"[CHUNK {chunk_idx}] Approaching end: bars_iterated={bars_iterated}, "
                        f"expected_bars={expected_bars}, remaining={expected_bars - bars_iterated}"
                    )
            
            # DEL 1: Extract performance summary metrics from runner
            n_trades_created = getattr(runner, "perf_n_trades_created", 0)
            feature_time_total = getattr(runner, "perf_feat_time", 0.0)
            feature_time_mean_ms = (feature_time_total / total_bars * 1000.0) if total_bars > 0 else 0.0
            feature_timeout_count = getattr(runner, "feature_timeout_count", 0)
            vol_regime_unknown_count = getattr(runner, "vol_regime_unknown_count", 0)
            htf_align_warn_count = getattr(runner, "htf_align_warn_count", 0)
            pregate_skips = getattr(runner, "pregate_skips", 0)
            pregate_passes = getattr(runner, "pregate_passes", 0)
            pregate_missing_inputs = getattr(runner, "pregate_missing_inputs", 0)
            
            # PATCH: Extract HTF alignment stats from runner (set in oanda_demo_runner.py)
            # FIX: Get stats directly from FeatureState/HTFAligner (not just perf collector)
            htf_align_time_total_sec = getattr(runner, "htf_align_time_total_sec", 0.0)
            htf_align_warning_time_sec = getattr(runner, "htf_align_warning_time_sec", 0.0)
            
            # FIX: Get HTFAligner stats directly from aligners via get_stats()
            if hasattr(runner, "feature_state") and runner.feature_state:
                h1_aligner = getattr(runner.feature_state, "h1_aligner", None)
                h4_aligner = getattr(runner.feature_state, "h4_aligner", None)
                
                if h1_aligner is not None:
                    h1_stats = h1_aligner.get_stats()
                    htf_h1_calls = h1_stats.get("call_count", 0)
                    htf_h1_warns = h1_stats.get("warn_count", 0)
                    htf_align_call_count += htf_h1_calls
                    htf_align_warn_count += htf_h1_warns
                    htf_last_m5_ts = h1_stats.get("last_m5_ts")
                    htf_last_j = h1_stats.get("last_j")
                
                if h4_aligner is not None:
                    h4_stats = h4_aligner.get_stats()
                    htf_h4_calls = h4_stats.get("call_count", 0)
                    htf_h4_warns = h4_stats.get("warn_count", 0)
                    htf_align_call_count += htf_h4_calls
                    htf_align_warn_count += htf_h4_warns
                
                # Get fallback count and feature compute bars from feature_state
                htf_align_fallback_count = getattr(runner.feature_state, "htf_align_fallback_count", 0)
                htf_feature_compute_bars = getattr(runner.feature_state, "htf_feature_compute_bars", 0)
            else:
                # Fallback to runner attributes if feature_state not available
                htf_align_call_count = getattr(runner, "htf_align_call_count", 0)
                htf_align_warn_count = getattr(runner, "htf_align_warn_count_total", htf_align_warn_count)
                htf_align_fallback_count = getattr(runner, "htf_align_fallback_count", 0)
                htf_feature_compute_bars = getattr(runner, "htf_feature_compute_bars", 0)
            
            # DEL 1: Extract phase timing (for "bars/sec" breakdown)
            t_pregate_total_sec = getattr(runner, "t_pregate_total_sec", 0.0)
            t_feature_build_total_sec = feature_time_total  # Pure feature build time (from perf_feat_time)
            t_model_total_sec = getattr(runner, "t_model_total_sec", 0.0)
            t_policy_total_sec = getattr(runner, "t_policy_total_sec", 0.0)
            t_io_total_sec = getattr(runner, "t_io_total_sec", 0.0)
            
            # D) Check prebuilt invariant: if prebuilt enabled, feature_time should be ~0
            prebuilt_used_env = os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1"
            if prebuilt_used_env and feature_time_mean_ms > 5.0:
                log.error(
                    f"[PREBUILT_FAIL] Prebuilt enabled but feature_time_mean_ms={feature_time_mean_ms:.2f}ms > 5ms. "
                    f"This indicates prebuilt features are not being used correctly."
                )
                # Mark as failed_invariant but continue to write footer
                if status == "ok":
                    status = "failed_invariant"
                    error = f"Prebuilt invariant violation: feature_time_mean_ms={feature_time_mean_ms:.2f}ms > 5ms"
            
            # Count trades from trade journal if available
            if hasattr(runner, "trade_journal") and runner.trade_journal:
                trade_journal_dir = chunk_output_dir / "trade_journal" / "trades"
                if trade_journal_dir.exists():
                    n_trades_closed = len(list(trade_journal_dir.glob("*.json")))
                else:
                    n_trades_closed = n_trades_created  # Fallback to created count
            else:
                n_trades_closed = n_trades_created  # Fallback to created count
            
            if status == "ok":
                worker_time = time.time() - worker_start_time
                log.info(
                    f"[CHUNK {chunk_idx}] Completed in {worker_time:.1f}s "
                    f"({bars_processed / worker_time:.1f} bars/sec)"
                )
        
        except Exception as e:
            import traceback as tb
            status = "failed"
            error = str(e)
            error_traceback = "".join(tb.format_exception(type(e), e, e.__traceback__))
            log.error(f"[CHUNK {chunk_idx}] FAILED: {error}", exc_info=True)
            
            # DEL 1: Write WORKER_START.json and FATAL_ERROR.txt on any exception
            worker_start_info["exception_full"] = error_traceback
            worker_start_json_path = chunk_output_dir / "WORKER_START.json"
            try:
                with open(worker_start_json_path, "w") as f:
                    import json
                    json.dump(worker_start_info, f, indent=2)
            except Exception as json_error:
                log.error(f"[CHUNK {chunk_idx}] Failed to write WORKER_START.json: {json_error}")
            
            fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
            try:
                with open(fatal_error_path, "w") as f:
                    f.write(f"[CHUNK {chunk_idx}] FAILED: {error}\n\n")
                    f.write(f"Full traceback:\n{error_traceback}\n\n")
                    f.write(f"Worker start info:\n")
                    f.write(f"  chunk_id: {chunk_idx}\n")
                    f.write(f"  prebuilt_parquet_path_raw: {worker_start_info.get('prebuilt_parquet_path_raw')}\n")
                    f.write(f"  prebuilt_parquet_path_raw_type: {worker_start_info.get('prebuilt_parquet_path_raw_type')}\n")
                    f.write(f"  prebuilt_parquet_path_resolved: {worker_start_info.get('prebuilt_parquet_path_resolved')}\n")
                    f.write(f"  exists: {worker_start_info.get('exists')}\n")
                    f.write(f"  size: {worker_start_info.get('size')}\n")
                    f.write(f"  cwd: {cwd}\n")
            except Exception as fatal_error:
                log.error(f"[CHUNK {chunk_idx}] Failed to write FATAL_ERROR.txt: {fatal_error}")
            
            # FIX: Ensure all footer variables are defined even on early failure
            wall_clock_sec = time.time() - worker_start_time
            # Set total_bars from bars_processed if available, otherwise keep 0
            if bars_processed > 0:
                total_bars = bars_processed
                # Note: os_module is already imported at function start
        
        finally:
            # DEL 1: Write WORKER_START.json if not already written (success case)
            if worker_start_info.get("exception_full") is None:
                # Success case - update with final state
                worker_start_json_path = chunk_output_dir / "WORKER_START.json"
                try:
                    with open(worker_start_json_path, "w") as f:
                        import json
                        json.dump(worker_start_info, f, indent=2)
                except Exception as json_error:
                    log.warning(f"[CHUNK {chunk_idx}] Failed to write WORKER_START.json (success case): {json_error}")
            
            # DEL 1: ALWAYS flush collectors, even on exceptions
            log.info(f"[FLUSH] chunk={chunk_idx} start")
            flush_count = 0
            
            # CRITICAL: Always flush entry feature telemetry in finally (even on exceptions)
            telemetry_required = os_module.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
            telemetry_flushed = False
            try:
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        # Check if no entry evaluations occurred
                        bars_reaching_entry_stage = getattr(runner, "bars_reaching_entry_stage", 0) if runner else 0
                        no_entry_evaluations = bars_reaching_entry_stage == 0 and bars_processed > 0
                        if no_entry_evaluations:
                            # Determine reason
                            bars_skipped_warmup = getattr(runner, "bars_skipped_warmup", 0) if runner else 0
                            bars_skipped_pregate = getattr(runner, "bars_skipped_pregate", 0) if runner else 0
                            if bars_skipped_warmup == bars_processed:
                                no_entry_reason = "warmup"
                            elif bars_skipped_pregate == bars_processed:
                                no_entry_reason = "pregate_blocked_all"
                            else:
                                no_entry_reason = "unknown"
                            
                            # Set env var for telemetry collector
                            os_module.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"] = "1"
                            os_module.environ["GX1_TELEMETRY_NO_ENTRY_REASON"] = no_entry_reason
                        
                        em.entry_feature_telemetry.write_all(chunk_output_dir)
                        telemetry_flushed = True
                        log.info(f"[FLUSH] [CHUNK {chunk_idx}] Entry feature telemetry flushed to {chunk_output_dir}")
                    elif telemetry_required:
                        log.error(
                            f"[FLUSH] [CHUNK {chunk_idx}] FATAL: entry_feature_telemetry not initialized "
                            f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Check EntryManager initialization."
                        )
                elif telemetry_required and bars_processed > 0:
                    log.error(
                        f"[FLUSH] [CHUNK {chunk_idx}] FATAL: runner or entry_manager not available "
                        f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Cannot flush telemetry."
                    )
            except Exception as telemetry_flush_error:
                log.error(f"[FLUSH] [CHUNK {chunk_idx}] Failed to flush entry feature telemetry: {telemetry_flush_error}", exc_info=True)
                if telemetry_required:
                    log.error(
                        f"[FLUSH] [CHUNK {chunk_idx}] FATAL: Failed to flush entry feature telemetry "
                        f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Error: {telemetry_flush_error}"
                    )
            finally:
                # Clean up env vars
                if "GX1_TELEMETRY_NO_ENTRY_EVALUATIONS" in os_module.environ:
                    del os_module.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"]
                if "GX1_TELEMETRY_NO_ENTRY_REASON" in os_module.environ:
                    del os_module.environ["GX1_TELEMETRY_NO_ENTRY_REASON"]
            
            try:
                if runner and hasattr(runner, "replay_eval_collectors") and runner.replay_eval_collectors:
                    # Import here to avoid circular imports
                    from gx1.scripts.replay_eval_gated import flush_replay_eval_collectors
                    
                    # Call flush with chunk_output_dir (writes directly to chunk dir)
                    flush_replay_eval_collectors(runner, runner.replay_eval_collectors, output_dir=chunk_output_dir)
                    
                    # Count written files
                    for pattern in [
                        f"raw_signals_{run_id}.parquet",
                        f"policy_decisions_{run_id}.parquet",
                        f"trade_outcomes_{run_id}.parquet",
                        f"attribution_{run_id}.json",
                        f"metrics_{run_id}.json",
                        f"summary_{run_id}.md",
                        f"raw_signals_{run_id}.metadata.json",
                    ]:
                        if (chunk_output_dir / pattern).exists():
                            flush_count += 1
                    
                    log.info(f"[FLUSH] chunk={chunk_idx} done: wrote {flush_count} files")
                else:
                    log.warning(f"[FLUSH] chunk={chunk_idx} no collectors to flush")
            except Exception as flush_error:
                log.error(f"[FLUSH] chunk={chunk_idx} flush failed: {flush_error}", exc_info=True)
            
            # DEL 2: Write chunk_footer.json with status
            # FIX: Import json and datetime at function level (needed in both try and except)
            # Note: os_module is already imported at function start
            import json
            from datetime import datetime
            import numpy as np
            
            try:
                import traceback as tb
                
                # Convert numpy types to native Python types for JSON serialization
                def convert_to_json_serializable(obj):
                    if isinstance(obj, (np.integer, np.int64, np.int32)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64, np.float32)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_json_serializable(item) for item in obj]
                    return obj
                
                # 5) Set metrics to null if status != ok (not 0.0 which masks errors)
                if status == "ok":
                    feature_time_mean_ms_val = convert_to_json_serializable(feature_time_mean_ms)
                    t_feature_build_total_sec_val = convert_to_json_serializable(t_feature_build_total_sec)
                    t_model_total_sec_val = convert_to_json_serializable(t_model_total_sec)
                    t_policy_total_sec_val = convert_to_json_serializable(t_policy_total_sec)
                    t_io_total_sec_val = convert_to_json_serializable(t_io_total_sec)
                    bars_per_sec_val = convert_to_json_serializable(bars_processed / wall_clock_sec if wall_clock_sec > 0 else 0.0)
                else:
                    feature_time_mean_ms_val = None
                    t_feature_build_total_sec_val = None
                    t_model_total_sec_val = None
                    t_policy_total_sec_val = None
                    t_io_total_sec_val = None
                    bars_per_sec_val = None
                
                # DEL 6: Get case collision resolution metadata (if available)
                case_collision_resolution_val = None
                if 'case_collision_resolution' in locals():
                    case_collision_resolution_val = convert_to_json_serializable(case_collision_resolution)
                
                chunk_footer = {
                    "run_id": run_id,
                    "chunk_id": str(chunk_idx),
                    "status": status,
                    "error": error,
                    "error_traceback": error_traceback[:5000] if error_traceback else None,  # Trim long tracebacks
                    "n_model_calls": convert_to_json_serializable(n_model_calls) if status == "ok" else None,
                    "n_trades_closed": convert_to_json_serializable(n_trades_closed) if status == "ok" else None,
                    # DEL 6: Case collision resolution metadata (if compat-mode was used)
                    "case_collision_resolution": case_collision_resolution_val,
                    # DEL 1: Add performance summary metrics for A/B comparison
                    "wall_clock_sec": convert_to_json_serializable(wall_clock_sec),
                    "total_bars": convert_to_json_serializable(total_bars) if status == "ok" else None,
                    "bars_per_sec": bars_per_sec_val,
                    "feature_time_mean_ms": feature_time_mean_ms_val,
                    "feature_timeout_count": convert_to_json_serializable(feature_timeout_count) if status == "ok" else None,
                    "htf_align_warn_count": convert_to_json_serializable(htf_align_warn_count) if status == "ok" else None,
                    "htf_align_time_total_sec": convert_to_json_serializable(htf_align_time_total_sec) if status == "ok" else None,
                    "htf_align_call_count": convert_to_json_serializable(htf_align_call_count) if status == "ok" else None,
                    "htf_align_warning_time_sec": convert_to_json_serializable(htf_align_warning_time_sec) if status == "ok" else None,
                    "htf_align_fallback_count": convert_to_json_serializable(htf_align_fallback_count) if status == "ok" else None,
                    "htf_feature_compute_bars": convert_to_json_serializable(htf_feature_compute_bars) if status == "ok" else None,
                    # FIX: Export HTFAligner stats (from get_stats())
                    "htf_h1_calls": convert_to_json_serializable(htf_h1_calls) if status == "ok" else None,
                    "htf_h4_calls": convert_to_json_serializable(htf_h4_calls) if status == "ok" else None,
                    "htf_h1_warns": convert_to_json_serializable(htf_h1_warns) if status == "ok" else None,
                    "htf_h4_warns": convert_to_json_serializable(htf_h4_warns) if status == "ok" else None,
                    "htf_last_m5_ts": convert_to_json_serializable(htf_last_m5_ts) if status == "ok" else None,
                    "htf_last_j": convert_to_json_serializable(htf_last_j) if status == "ok" else None,
                    "pregate_skips": convert_to_json_serializable(pregate_skips) if status == "ok" else None,
                    "pregate_passes": convert_to_json_serializable(pregate_passes) if status == "ok" else None,
                    "pregate_missing_inputs": convert_to_json_serializable(pregate_missing_inputs) if status == "ok" else None,
                    "vol_regime_unknown_count": convert_to_json_serializable(vol_regime_unknown_count) if status == "ok" else None,
                    # DEL 1: Phase timing breakdown
                    "t_pregate_total_sec": convert_to_json_serializable(t_pregate_total_sec) if status == "ok" else None,
                    "t_feature_build_total_sec": t_feature_build_total_sec_val,
                    "t_model_total_sec": t_model_total_sec_val,
                    "t_policy_total_sec": t_policy_total_sec_val,
                    "t_io_total_sec": t_io_total_sec_val,
                    "bars_processed": convert_to_json_serializable(bars_processed),
                    "start_ts": chunk_start.isoformat() if chunk_start else None,
                    "end_ts": chunk_end.isoformat() if chunk_end else None,
                    "worker_time_sec": float(time.time() - worker_start_time),
                    "pid": int(os_module.getpid()),
                    "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
                    "timestamp": dt_now_iso(),
                }
                
                # D) Add prebuilt features info to chunk footer
                # Get prebuilt_used from runner (not env var, as runner validates it)
                prebuilt_used = False
                if runner and hasattr(runner, "prebuilt_used"):
                    prebuilt_used = runner.prebuilt_used
                
                chunk_footer["prebuilt_used"] = prebuilt_used
                
                # Add prebuilt gate dump for diagnosis (from entry_manager)
                # If entry_manager hasn't been called yet, build dump from runner state directly
                prebuilt_gate_dump = None
                try:
                    if runner:
                        if hasattr(runner, "entry_manager") and hasattr(runner.entry_manager, "_prebuilt_gate_dump"):
                            prebuilt_gate_dump = runner.entry_manager._prebuilt_gate_dump
                        else:
                            # Fallback: build dump from runner state if entry_manager hasn't been called
                            prebuilt_features_df_exists = hasattr(runner, "prebuilt_features_df")
                            prebuilt_features_df_is_none = not prebuilt_features_df_exists or runner.prebuilt_features_df is None
                            prebuilt_features_df_len = len(runner.prebuilt_features_df) if not prebuilt_features_df_is_none else 0
                            prebuilt_features_df_index_type = type(runner.prebuilt_features_df.index).__name__ if not prebuilt_features_df_is_none else "N/A"
                            prebuilt_features_df_index_tz = str(getattr(runner.prebuilt_features_df.index, 'tz', None)) if not prebuilt_features_df_is_none else "N/A"
                            prebuilt_gate_dump = {
                                "is_replay": getattr(runner, "replay_mode", False),
                                "prebuilt_enabled": os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
                                "prebuilt_used_flag": getattr(runner, "prebuilt_used", False),
                                "prebuilt_features_df_exists": prebuilt_features_df_exists,
                                "prebuilt_features_df_is_none": prebuilt_features_df_is_none,
                                "prebuilt_features_df_len": prebuilt_features_df_len,
                                "prebuilt_features_df_index_type": prebuilt_features_df_index_type,
                                "prebuilt_features_df_index_tz": prebuilt_features_df_index_tz,
                            }
                    else:
                        # Runner not created yet - minimal dump
                        prebuilt_gate_dump = {
                            "is_replay": None,
                            "prebuilt_enabled": os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
                            "prebuilt_used_flag": None,
                            "prebuilt_features_df_exists": False,
                            "prebuilt_features_df_is_none": True,
                            "prebuilt_features_df_len": 0,
                            "prebuilt_features_df_index_type": "N/A",
                            "prebuilt_features_df_index_tz": "N/A",
                            "runner_is_none": True,
                        }
                except Exception as dump_error:
                    prebuilt_gate_dump = {
                        "error": str(dump_error),
                        "runner_exists": runner is not None,
                    }
                chunk_footer["prebuilt_gate_dump"] = prebuilt_gate_dump
                
                # Add lookup telemetry (always, for diagnosis)
                lookup_attempts = getattr(runner, "lookup_attempts", 0) if runner else 0
                lookup_hits = getattr(runner, "lookup_hits", 0) if runner else 0
                chunk_footer["lookup_attempts"] = lookup_attempts
                chunk_footer["lookup_hits"] = lookup_hits
                
                # Add evaluate_entry() call telemetry (from entry_manager)
                eval_calls_total = 0
                eval_calls_prebuilt_gate_true = 0
                eval_calls_prebuilt_gate_false = 0
                if runner and hasattr(runner, "entry_manager"):
                    eval_calls_total = getattr(runner.entry_manager, "eval_calls_total", 0)
                    eval_calls_prebuilt_gate_true = getattr(runner.entry_manager, "eval_calls_prebuilt_gate_true", 0)
                    eval_calls_prebuilt_gate_false = getattr(runner.entry_manager, "eval_calls_prebuilt_gate_false", 0)
                chunk_footer["eval_calls_total"] = eval_calls_total
                chunk_footer["eval_calls_prebuilt_gate_true"] = eval_calls_prebuilt_gate_true
                chunk_footer["eval_calls_prebuilt_gate_false"] = eval_calls_prebuilt_gate_false
                
                # Add model entry telemetry summary (for debugging why transformer may not be called)
                model_entry_summary = {
                    "attempts": 0,
                    "forward_calls": 0,
                    "block_reasons": {},
                }
                if runner and hasattr(runner, "entry_manager") and hasattr(runner.entry_manager, "entry_feature_telemetry") and runner.entry_manager.entry_feature_telemetry:
                    telemetry = runner.entry_manager.entry_feature_telemetry
                    model_entry_summary["attempts"] = sum(telemetry.model_attempt_calls.values())
                    model_entry_summary["forward_calls"] = sum(telemetry.model_forward_calls.values())
                    model_entry_summary["block_reasons"] = {
                        model_name: dict(reasons) 
                        for model_name, reasons in telemetry.model_block_counts.items()
                    }
                    model_entry_summary["transformer_forward_calls"] = telemetry.transformer_forward_calls
                    model_entry_summary["transformer_input_recorded"] = telemetry.transformer_input_recorded
                    
                    # Add entry routing telemetry
                    chunk_footer["entry_routing"] = {
                        "selected_model": telemetry.entry_routing_selected_model,
                        "reason": telemetry.entry_routing_reason,
                        "recorded": telemetry.entry_routing_recorded,
                    }
                    
                    # Add V10 callsite bridge telemetry
                    chunk_footer["v10_callsite"] = {
                        "entered": telemetry.v10_callsite_entered,
                        "returned": telemetry.v10_callsite_returned,
                        "exception": telemetry.v10_callsite_exception,
                        "last": telemetry.v10_callsite_last,
                        "last_exception": telemetry.v10_callsite_last_exception,
                    }
                    
                    # Add V10 enable-state telemetry
                    chunk_footer["entry_v10_enable_state"] = {
                        "enabled": telemetry.entry_v10_enabled,
                        "reason": telemetry.entry_v10_enabled_reason,
                        "enabled_true_count": telemetry.entry_v10_enabled_true_count,
                        "enabled_false_count": telemetry.entry_v10_enabled_false_count,
                        "reason_counts": dict(telemetry.entry_v10_enabled_reason_counts),
                    }
                    
                    # Add control-flow telemetry
                    chunk_footer["control_flow"] = {
                        "counts": dict(telemetry.control_flow_counts),
                        "last": telemetry.control_flow_last,
                    }
                    
                    # Add entry eval path telemetry
                    chunk_footer["entry_eval_path"] = {
                        "counts": dict(telemetry.entry_eval_path_counts),
                        "last": telemetry.entry_eval_path_last,
                    }
                    
                    # Add exception gap telemetry
                    chunk_footer["exception_gap"] = telemetry.exception_gap
                    
                    # SOFT ELIGIBILITY TRUTH: Compute execution-truth summary
                    control_flow_counts = dict(telemetry.control_flow_counts)
                    soft_eligibility_truth = {
                        "return_true_count": control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_TRUE", 0),
                        "return_false_count": control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_FALSE", 0),
                        "after_passed_count": control_flow_counts.get("AFTER_SOFT_ELIGIBILITY_PASSED", 0),
                        "blocked_branch_count": control_flow_counts.get("SOFT_ELIGIBILITY_BLOCKED_BRANCH", 0),
                    }
                    chunk_footer["soft_eligibility_truth"] = soft_eligibility_truth
                    
                    # EXCEPTION GAP INVARIANT (fail-fast, GX1_REQUIRE_ENTRY_TELEMETRY=1)
                    if telemetry_required:
                        after_soft_passed = control_flow_counts.get("AFTER_SOFT_ELIGIBILITY_PASSED", 0)
                        before_stage0_check = control_flow_counts.get("BEFORE_STAGE0_CHECK", 0)
                        exception_in_gap = control_flow_counts.get("EXCEPTION_IN_SOFT_TO_STAGE0_GAP", 0)
                        early_return_in_gap = control_flow_counts.get("EARLY_RETURN_IN_GAP", 0)
                        
                        # Invariant: If soft eligibility passed, we must either reach Stage-0 check OR have explicit exception/return
                        if after_soft_passed > 0 and before_stage0_check == 0:
                            # Check if we have explicit exception or early return
                            if exception_in_gap == 0 and early_return_in_gap == 0:
                                # UNKNOWN EXIT: No explicit exception or return recorded
                                raise RuntimeError(
                                    f"[CHUNK {chunk_idx}] FATAL: UNKNOWN_EXIT_SOFT_TO_STAGE0. "
                                    f"AFTER_SOFT_ELIGIBILITY_PASSED={after_soft_passed} > 0 but "
                                    f"BEFORE_STAGE0_CHECK=0 and EXCEPTION_IN_SOFT_TO_STAGE0_GAP=0 and "
                                    f"EARLY_RETURN_IN_GAP=0. "
                                    f"This indicates an uncaught exception or silent exit between soft eligibility and Stage-0 check. "
                                    f"Control flow: {control_flow_counts}, last: {telemetry.control_flow_last}"
                                )
                            
                            # Get last exception/return info for diagnostic
                            control_flow_last = telemetry.control_flow_last
                            last_event = control_flow_last.get("event", "UNKNOWN") if control_flow_last else "UNKNOWN"
                            last_reason = control_flow_last.get("reason", "UNKNOWN") if control_flow_last else "UNKNOWN"
                            last_line = control_flow_last.get("line", "UNKNOWN") if control_flow_last else "UNKNOWN"
                            exc_type = control_flow_last.get("exc_type", "UNKNOWN") if control_flow_last else "UNKNOWN"
                            exc_msg = control_flow_last.get("exc_msg", "UNKNOWN") if control_flow_last else "UNKNOWN"
                            
                            # Get entry state for diagnostic
                            entry_v10_enabled = telemetry.entry_v10_enabled
                            entry_v10_reason = telemetry.entry_v10_enabled_reason
                            
                            # Get lookup accounting if available
                            lookup_attempts = 0
                            if runner and hasattr(runner, "prebuilt_features_loader") and runner.prebuilt_features_loader:
                                lookup_accounting = runner.prebuilt_features_loader.get_lookup_accounting()
                                lookup_attempts = lookup_accounting.get("attempts", 0)
                            
                            if exception_in_gap > 0:
                                # Exception was caught and logged
                                raise RuntimeError(
                                    f"[EXCEPTION_IN_SOFT_TO_STAGE0_GAP] "
                                    f"AFTER_SOFT_ELIGIBILITY_PASSED={after_soft_passed} but BEFORE_STAGE0_CHECK=0. "
                                    f"Exception caught: {exc_type}: {exc_msg} at line {last_line}. "
                                    f"Entry state: entry_v10_enabled={entry_v10_enabled}, reason={entry_v10_reason}. "
                                    f"Lookup attempts: {lookup_attempts}. "
                                    f"Check control_flow_last for full exception details and traceback."
                                )
                            elif early_return_in_gap > 0:
                                # Early return was logged
                                raise RuntimeError(
                                    f"[EARLY_RETURN_BETWEEN_SOFT_AND_STAGE0] "
                                    f"AFTER_SOFT_ELIGIBILITY_PASSED={after_soft_passed} but BEFORE_STAGE0_CHECK=0. "
                                    f"Early return: reason={last_reason}, line={last_line}. "
                                    f"Entry state: entry_v10_enabled={entry_v10_enabled}, reason={entry_v10_reason}. "
                                    f"Lookup attempts: {lookup_attempts}. "
                                    f"Check control_flow_last for exact location and reason."
                                )
                            else:
                                # Unknown exit (should not happen with proper instrumentation)
                                raise RuntimeError(
                                    f"[UNKNOWN_EXIT_SOFT_TO_STAGE0] "
                                    f"AFTER_SOFT_ELIGIBILITY_PASSED={after_soft_passed} but BEFORE_STAGE0_CHECK=0. "
                                    f"No EXCEPTION_IN_SOFT_TO_STAGE0_GAP or EARLY_RETURN_IN_GAP recorded. "
                                    f"This indicates a missing instrumentation or unhandled exit. "
                                    f"Last control flow event: {last_event}. "
                                    f"Entry state: entry_v10_enabled={entry_v10_enabled}, reason={entry_v10_reason}. "
                                    f"Lookup attempts: {lookup_attempts}. "
                                    f"Control flow: {control_flow_counts}. "
                                    f"Check control_flow_last for details."
                                )
                    
                    # EARLY RETURN GAP INVARIANT (fail-fast, GX1_REQUIRE_ENTRY_TELEMETRY=1)
                    if telemetry_required:
                        after_hard_passed = control_flow_counts.get("AFTER_HARD_ELIGIBILITY_PASSED", 0)
                        before_soft_check = control_flow_counts.get("BEFORE_SOFT_ELIGIBILITY_CHECK", 0)
                        early_return_in_gap = control_flow_counts.get("EARLY_RETURN_IN_GAP", 0)
                        
                        # Invariant: If hard eligibility passed, we must either reach soft check OR have explicit early return
                        if after_hard_passed > 0 and before_soft_check == 0:
                            # Get last early return reason for diagnostic
                            control_flow_last = telemetry.control_flow_last
                            last_reason = control_flow_last.get("reason", "UNKNOWN") if control_flow_last else "UNKNOWN"
                            last_line = control_flow_last.get("line", "UNKNOWN") if control_flow_last else "UNKNOWN"
                            
                            # Get entry state for diagnostic
                            entry_v10_enabled = telemetry.entry_v10_enabled
                            entry_v10_reason = telemetry.entry_v10_enabled_reason
                            
                            # Get lookup accounting if available
                            lookup_attempts = 0
                            if runner and hasattr(runner, "prebuilt_features_loader") and runner.prebuilt_features_loader:
                                lookup_accounting = runner.prebuilt_features_loader.get_lookup_accounting()
                                lookup_attempts = lookup_accounting.get("attempts", 0)
                            
                            raise RuntimeError(
                                f"[EARLY_RETURN_BETWEEN_HARD_AND_SOFT] "
                                f"AFTER_HARD_ELIGIBILITY_PASSED={after_hard_passed} but BEFORE_SOFT_ELIGIBILITY_CHECK=0. "
                                f"This indicates an early return or exception between hard and soft eligibility. "
                                f"Last EARLY_RETURN_IN_GAP: reason={last_reason}, line={last_line}. "
                                f"Entry state: entry_v10_enabled={entry_v10_enabled}, reason={entry_v10_reason}. "
                                f"Lookup attempts: {lookup_attempts}. "
                                f"Control flow: {control_flow_counts}. "
                                f"Check control_flow_last for exact location and reason."
                            )
                    
                    # SOFT ELIGIBILITY INVARIANTS (fail-fast, GX1_REQUIRE_ENTRY_TELEMETRY=1)
                    if telemetry_required:
                        return_true = control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_TRUE", 0)
                        return_false = control_flow_counts.get("SOFT_ELIGIBILITY_RETURN_FALSE", 0)
                        after_passed = control_flow_counts.get("AFTER_SOFT_ELIGIBILITY_PASSED", 0)
                        blocked_branch = control_flow_counts.get("SOFT_ELIGIBILITY_BLOCKED_BRANCH", 0)
                        
                        # Get soft eligibility gate stats
                        gate_stats = dict(telemetry.gate_stats)
                        soft_gate = gate_stats.get("soft_eligibility", {})
                        soft_passed_count = soft_gate.get("passed", 0)
                        
                        # Invariant 1: If function returns True, control flow must continue
                        if return_true > 0 and after_passed == 0:
                            raise RuntimeError(
                                f"[SOFT_ELIGIBILITY_LIE_OR_EARLY_EXIT] "
                                f"SOFT_ELIGIBILITY_RETURN_TRUE={return_true} but AFTER_SOFT_ELIGIBILITY_PASSED=0. "
                                f"This indicates either: (1) gate telemetry lies, or (2) early exit/exception between return and sentinel. "
                                f"Control flow: {control_flow_counts}"
                            )
                        
                        # Invariant 2: If control flow continues, gate telemetry must match
                        if after_passed > 0 and soft_passed_count == 0:
                            raise RuntimeError(
                                f"[SOFT_ELIGIBILITY_TELEMETRY_MISMATCH] "
                                f"AFTER_SOFT_ELIGIBILITY_PASSED={after_passed} but soft_eligibility gate passed=0. "
                                f"This indicates gate telemetry is not recording execution-truth. "
                                f"Gate stats: {soft_gate}, Control flow: {control_flow_counts}"
                            )
                        
                        # Invariant 3: If gate telemetry says passed, function must have returned True
                        if soft_passed_count > 0 and return_true == 0:
                            raise RuntimeError(
                                f"[SOFT_ELIGIBILITY_FALSE_POSITIVE] "
                                f"soft_eligibility gate passed={soft_passed_count} but SOFT_ELIGIBILITY_RETURN_TRUE=0. "
                                f"This indicates gate telemetry recorded pass without function actually returning True. "
                                f"Gate stats: {soft_gate}, Control flow: {control_flow_counts}"
                            )
                        
                        # Invariant 4: Sanity check: return_true + return_false should match before_check
                        before_check = control_flow_counts.get("BEFORE_SOFT_ELIGIBILITY_CHECK", 0)
                        if before_check > 0 and (return_true + return_false) != before_check:
                            log.warning(
                                f"[SOFT_ELIGIBILITY_SANITY] "
                                f"BEFORE_SOFT_ELIGIBILITY_CHECK={before_check} but "
                                f"return_true={return_true} + return_false={return_false} = {return_true + return_false}. "
                                f"This may indicate exception in _check_soft_eligibility()"
                            )
                else:
                    # No telemetry available - set defaults
                    chunk_footer["entry_routing"] = {
                        "selected_model": None,
                        "reason": "TELEMETRY_NOT_AVAILABLE",
                        "recorded": False,
                    }
                    chunk_footer["v10_callsite"] = {
                        "entered": 0,
                        "returned": 0,
                        "exception": 0,
                        "last": None,
                        "last_exception": None,
                    }
                    chunk_footer["entry_v10_enable_state"] = {
                        "enabled": None,
                        "reason": "TELEMETRY_NOT_AVAILABLE",
                        "enabled_true_count": 0,
                        "enabled_false_count": 0,
                        "reason_counts": {},
                    }
                    chunk_footer["control_flow"] = {
                        "counts": {},
                        "last": None,
                    }
                    chunk_footer["entry_eval_path"] = {
                        "counts": {},
                        "last": None,
                    }
                chunk_footer["model_entry_summary"] = model_entry_summary

                # SNIPER_GUARD_V1 UNKNOWN policy counters (export from module-level telemetry)
                # These counters are per worker process (per chunk) and are deterministic for a given run.
                try:
                    from gx1.policy import farm_guards as farm_guards_module
                    chunk_footer["guard_unknown_pass_count"] = convert_to_json_serializable(
                        int(getattr(farm_guards_module, "SNIPER_GUARD_UNKNOWN_PASS_COUNT", 0))
                    )
                    chunk_footer["guard_unknown_block_count"] = convert_to_json_serializable(
                        int(getattr(farm_guards_module, "SNIPER_GUARD_UNKNOWN_BLOCK_COUNT", 0))
                    )
                except Exception:
                    chunk_footer["guard_unknown_pass_count"] = convert_to_json_serializable(0)
                    chunk_footer["guard_unknown_block_count"] = convert_to_json_serializable(0)

                # Kill-chain telemetry (READ-ONLY, where trades die)
                killchain_version = 1
                killchain_fields = {
                    "killchain_n_entry_pred_total": 0,
                    "killchain_n_above_threshold": 0,
                    "killchain_n_after_session_guard": 0,
                    "killchain_n_after_vol_guard": 0,
                    "killchain_n_after_regime_guard": 0,
                    "killchain_n_after_risk_sizing": 0,
                    "killchain_n_trade_create_attempts": 0,
                    "killchain_n_trade_created": 0,
                }
                killchain_block_reason_counts = {}
                if runner and hasattr(runner, "entry_manager"):
                    em = runner.entry_manager
                    killchain_version = int(getattr(em, "killchain_version", 1))
                    for k in list(killchain_fields.keys()):
                        killchain_fields[k] = int(getattr(em, k.replace("killchain_", "killchain_"), 0))
                    killchain_block_reason_counts = getattr(em, "killchain_block_reason_counts", {}) or {}

                # Deterministic JSON output (sorted reason keys)
                killchain_block_reason_counts_sorted = {
                    str(k): int(killchain_block_reason_counts.get(k, 0))
                    for k in sorted(killchain_block_reason_counts.keys())
                }

                # Compute top block reason (stable tie-break: lexicographic)
                top_reason = None
                if killchain_block_reason_counts_sorted:
                    top_reason = sorted(
                        killchain_block_reason_counts_sorted.items(),
                        key=lambda kv: (-kv[1], kv[0]),
                    )[0][0]

                chunk_footer["killchain_version"] = killchain_version
                for k, v in killchain_fields.items():
                    chunk_footer[k] = convert_to_json_serializable(v)
                chunk_footer["killchain_block_reason_counts"] = convert_to_json_serializable(killchain_block_reason_counts_sorted)
                chunk_footer["killchain_top_block_reason"] = top_reason

                # Kill-chain STAGE2 (post-vol) counters (SSoT)
                stage2 = {
                    "killchain_stage2_version": 1,
                    "killchain_n_pred_available": 0,
                    "killchain_n_pass_score_gate": 0,
                    "killchain_n_block_below_threshold": 0,
                    "killchain_n_block_spread_guard": 0,
                    "killchain_n_block_cost_guard": 0,
                    "killchain_n_block_session_time_guard": 0,
                    "killchain_n_block_position_limit": 0,
                    "killchain_n_block_cooldown": 0,
                    "killchain_n_block_risk_guard": 0,
                    "killchain_n_block_unknown_post_vol": 0,
                }
                stage2_unknown_examples = []
                if runner and hasattr(runner, "entry_manager"):
                    em = runner.entry_manager
                    stage2["killchain_stage2_version"] = int(getattr(em, "killchain_stage2_version", 1))
                    for k in list(stage2.keys()):
                        if k == "killchain_stage2_version":
                            continue
                        stage2[k] = int(getattr(em, k, 0))
                    stage2_unknown_examples = getattr(em, "killchain_stage2_unknown_examples", []) or []

                # Export stage2
                for k, v in stage2.items():
                    chunk_footer[k] = convert_to_json_serializable(v)
                chunk_footer["killchain_stage2_unknown_examples"] = convert_to_json_serializable(stage2_unknown_examples[:5])

                # Stage2 SSoT asserts (fail-fast if inconsistent)
                after_vol = int(killchain_fields.get("killchain_n_after_vol_guard", 0))
                pass_score = int(stage2.get("killchain_n_pass_score_gate", 0))
                block_below = int(stage2.get("killchain_n_block_below_threshold", 0))
                # All after-vol bars must partition into pass_score + block_below
                if after_vol != (pass_score + block_below):
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE2_FAIL] SSoT mismatch: "
                        f"after_vol_guard={after_vol} != pass_score_gate={pass_score} + block_below_threshold={block_below}"
                    )
                # pass_score must partition into trade_created + explicit post-score blocks
                trade_created = int(killchain_fields.get("killchain_n_trade_created", 0))
                post_blocks = (
                    int(stage2.get("killchain_n_block_spread_guard", 0))
                    + int(stage2.get("killchain_n_block_cost_guard", 0))
                    + int(stage2.get("killchain_n_block_session_time_guard", 0))
                    + int(stage2.get("killchain_n_block_position_limit", 0))
                    + int(stage2.get("killchain_n_block_cooldown", 0))
                    + int(stage2.get("killchain_n_block_risk_guard", 0))
                    + int(stage2.get("killchain_n_block_unknown_post_vol", 0))
                )
                if pass_score != (trade_created + post_blocks):
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE2_FAIL] SSoT mismatch: "
                        f"pass_score_gate={pass_score} != trade_created={trade_created} + post_blocks={post_blocks}"
                    )
                if pass_score < trade_created:
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE2_FAIL] SSoT mismatch: "
                        f"pass_score_gate={pass_score} < trade_created={trade_created}"
                    )

                # Stage2 top-3 block reasons (post-score only)
                post_reason_counts = {
                    "BLOCK_SPREAD_GUARD": int(stage2.get("killchain_n_block_spread_guard", 0)),
                    "BLOCK_COST_GUARD": int(stage2.get("killchain_n_block_cost_guard", 0)),
                    "BLOCK_SESSION_TIME_GUARD": int(stage2.get("killchain_n_block_session_time_guard", 0)),
                    "BLOCK_POSITION_LIMIT": int(stage2.get("killchain_n_block_position_limit", 0)),
                    "BLOCK_COOLDOWN": int(stage2.get("killchain_n_block_cooldown", 0)),
                    "BLOCK_RISK_GUARD": int(stage2.get("killchain_n_block_risk_guard", 0)),
                    "BLOCK_UNKNOWN_POST_VOL": int(stage2.get("killchain_n_block_unknown_post_vol", 0)),
                }
                top_post = sorted(post_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
                chunk_footer["killchain_stage2_top_post_blocks"] = convert_to_json_serializable(top_post)

                # Kill-chain STAGE3 (score histogram/CDF) - read-only
                # Population SSoT: hist_total == killchain_n_pred_available == killchain_n_after_vol_guard
                hist_bins = None
                hist_counts = None
                hist_total = 0
                if runner and hasattr(runner, "entry_manager"):
                    em = runner.entry_manager
                    hist_bins = getattr(em, "entry_score_hist_bins", None)
                    hist_counts = getattr(em, "entry_score_hist_counts", None)
                    hist_total = int(getattr(em, "entry_score_hist_total", 0) or 0)
                if hist_bins is None:
                    hist_bins = [i / 100.0 for i in range(0, 101)]
                if hist_counts is None:
                    hist_counts = [0 for _ in range(100)]

                # SSoT asserts for histogram
                if int(sum(hist_counts)) != int(hist_total):
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE3_FAIL] Histogram mismatch: "
                        f"sum(hist_counts)={int(sum(hist_counts))} != hist_total={int(hist_total)}"
                    )
                if int(hist_total) != int(after_vol):
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE3_FAIL] Population mismatch: "
                        f"entry_score_hist_total={int(hist_total)} != killchain_n_after_vol_guard={int(after_vol)}"
                    )
                if int(hist_total) != int(stage2.get("killchain_n_pred_available", 0)):
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE3_FAIL] Population mismatch: "
                        f"entry_score_hist_total={int(hist_total)} != killchain_n_pred_available={int(stage2.get('killchain_n_pred_available', 0))}"
                    )

                chunk_footer["entry_score_hist_bins"] = convert_to_json_serializable(hist_bins)
                chunk_footer["entry_score_hist_counts"] = convert_to_json_serializable(hist_counts)
                chunk_footer["entry_score_hist_total"] = convert_to_json_serializable(hist_total)

                # Export fixed CDF points (pct_ge_X) from histogram
                cdf_thresholds = [0.10, 0.15, 0.18, 0.20, 0.25, 0.30]
                for thr in cdf_thresholds:
                    # bin idx for >=thr: counts from idx..end (0.01 bins)
                    idx = int(thr * 100.0)
                    if idx < 0:
                        idx = 0
                    if idx > 100:
                        idx = 100
                    ge_count = int(sum(hist_counts[idx:])) if idx < 100 else 0
                    pct = (ge_count / float(hist_total)) if hist_total > 0 else 0.0
                    key = f"pct_ge_{str(thr).replace('.', '_')}"
                    chunk_footer[key] = convert_to_json_serializable(pct)
                
                # STEG 3: Export entry-score stats to footer (for entry-score distribution analysis)
                if runner:
                    entry_score_samples = getattr(runner, "entry_score_samples", [])
                    if entry_score_samples:
                        prob_longs = [s.get("prob_long") for s in entry_score_samples if s.get("prob_long") is not None and np.isfinite(s.get("prob_long"))]
                        if prob_longs:
                            import numpy as np
                            chunk_footer["entry_score_samples"] = len(prob_longs)
                            chunk_footer["entry_score_min"] = convert_to_json_serializable(np.min(prob_longs))
                            chunk_footer["entry_score_max"] = convert_to_json_serializable(np.max(prob_longs))
                            chunk_footer["entry_score_mean"] = convert_to_json_serializable(np.mean(prob_longs))
                            chunk_footer["entry_score_median"] = convert_to_json_serializable(np.median(prob_longs))
                            chunk_footer["entry_score_p5"] = convert_to_json_serializable(np.percentile(prob_longs, 5))
                            chunk_footer["entry_score_p25"] = convert_to_json_serializable(np.percentile(prob_longs, 25))
                            chunk_footer["entry_score_p50"] = convert_to_json_serializable(np.median(prob_longs))
                            chunk_footer["entry_score_p75"] = convert_to_json_serializable(np.percentile(prob_longs, 75))
                            chunk_footer["entry_score_p95"] = convert_to_json_serializable(np.percentile(prob_longs, 95))
                            chunk_footer["entry_score_p99"] = convert_to_json_serializable(np.percentile(prob_longs, 99))
                        else:
                            chunk_footer["entry_score_samples"] = 0
                    else:
                        chunk_footer["entry_score_samples"] = 0
                else:
                    chunk_footer["entry_score_samples"] = 0
                
                # Add entry stage telemetry (where bars disappear)
                chunk_footer["bars_seen"] = bars_seen
                chunk_footer["bars_skipped_warmup"] = bars_skipped_warmup
                chunk_footer["bars_skipped_pregate"] = bars_skipped_pregate
                chunk_footer["bars_reaching_entry_stage"] = bars_reaching_entry_stage
                chunk_footer["pregate_enabled"] = pregate_enabled
                
                # PRE-ENTRY FUNNEL COUNTERS: Track where bars die before entry evaluation (SSoT)
                if runner and hasattr(runner, "get_pre_entry_funnel_snapshot"):
                    try:
                        pre_entry_funnel = runner.get_pre_entry_funnel_snapshot()
                        chunk_footer["pre_entry_funnel"] = pre_entry_funnel
                    except Exception as e:
                        log.warning(f"[CHUNK {chunk_idx}] Failed to get pre_entry_funnel snapshot: {e}")
                        chunk_footer["pre_entry_funnel"] = {}
                else:
                    # Fallback: try to get individual attributes
                    candles_iterated = getattr(runner, "candles_iterated", 0) if runner else 0
                    warmup_skipped = getattr(runner, "warmup_skipped", 0) if runner else 0
                    pregate_checked_count = getattr(runner, "pregate_checked_count", 0) if runner else 0
                    pregate_skipped = getattr(runner, "pregate_skipped", 0) if runner else 0
                    prebuilt_available_checked = getattr(runner, "prebuilt_available_checked", 0) if runner else 0
                    prebuilt_missing_skipped = getattr(runner, "prebuilt_missing_skipped", 0) if runner else 0
                    bars_before_evaluate_entry = getattr(runner, "bars_before_evaluate_entry", 0) if runner else 0
                    evaluate_entry_called_count = getattr(runner, "evaluate_entry_called_count", 0) if runner else 0
                    bars_after_evaluate_entry = getattr(runner, "bars_after_evaluate_entry", 0) if runner else 0
                    last_stop_reason = getattr(runner, "last_stop_reason", None) if runner else None
                    
                    chunk_footer["pre_entry_funnel"] = {
                        "candles_iterated": candles_iterated,
                        "warmup_skipped": warmup_skipped,
                        "pregate_checked_count": pregate_checked_count,
                        "pregate_skipped": pregate_skipped,
                        "prebuilt_available_checked": prebuilt_available_checked,
                        "prebuilt_missing_skipped": prebuilt_missing_skipped,
                        "bars_before_evaluate_entry": bars_before_evaluate_entry,
                        "evaluate_entry_called_count": evaluate_entry_called_count,
                        "bars_after_evaluate_entry": bars_after_evaluate_entry,
                        "last_stop_reason": last_stop_reason,
                    }
                
                # DEL 4: Use standardized bar counter snapshot
                bar_counters = compute_bar_counters_snapshot(runner, bars_processed, chunk_df)
                candles_iterated = bar_counters["candles_iterated"]
                warmup_skipped = bar_counters["warmup_skipped"]
                pregate_skipped = bar_counters["pregate_skipped"]
                reached_entry_stage = bar_counters["reached_entry_stage"]
                processed = bar_counters["processed"]
                
                # Store in chunk_footer
                chunk_footer["bar_counters"] = bar_counters
                chunk_footer["bars_skipped"] = candles_iterated - processed
                chunk_footer["warmup_bars"] = warmup_skipped
                chunk_footer["eligibility_blocks"] = pregate_skipped
                
                # Hard invariant: skipped == warmup_skipped + pregate_skipped
                # skipped = candles_iterated - reached_entry_stage (bars that didn't reach entry stage)
                skipped = candles_iterated - reached_entry_stage
                expected_skipped = warmup_skipped + pregate_skipped
                
                if skipped != expected_skipped:
                    # Check panic mode (default: disabled for smokes)
                    panic_mode = os_module.getenv("GX1_PANIC_MODE", "0") == "1"
                    
                    # Get policy_id and run_identity_data if available
                    if policy_id is None:
                        try:
                            run_identity_path = chunk_output_dir / "run_header.json"
                            if run_identity_path.exists():
                                with open(run_identity_path, "r") as f:
                                    run_identity_data = json.load(f)
                                    policy_id = run_identity_data.get("policy_id")
                        except Exception:
                            pass
                    
                    # Get first/last iter timestamps from chunk_df if available
                    if first_iter_ts is None and chunk_df is not None and len(chunk_df) > 0:
                        first_iter_ts = chunk_df.index[0]
                        last_iter_ts = chunk_df.index[-1]
                    
                    # Write failure capsule BEFORE raising (if panic mode, write before kill)
                    try:
                        fail_capsule = {
                            "chunk_idx": chunk_idx,
                            "run_id": run_id,
                            "exception_type": "BARS_SKIP_INVARIANT_FAIL",
                            "exception_message": f"skipped={skipped} != warmup_skipped={warmup_skipped} + pregate_skipped={pregate_skipped} = {expected_skipped}",
                            "bar_counters": bar_counters,
                            "bars_seen": bars_seen,
                            "bars_processed": bars_processed,
                            "bars_reaching_entry_stage": bars_reaching_entry_stage,
                            "first_iter_ts": str(first_iter_ts) if first_iter_ts else None,
                            "last_iter_ts": str(last_iter_ts) if last_iter_ts else None,
                            "replay_mode": "PREBUILT" if os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1" else "UNKNOWN",
                            "policy_id": policy_id,
                            "bundle_sha256": bundle_sha256,
                            "run_identity_keys": list(run_identity_data.keys()) if run_identity_data else None,
                            "timestamp": dt_now_iso(),
                        }
                        
                        capsule_path = chunk_output_dir / "CHUNK_FAIL_CAPSULE.json"
                        with open(capsule_path, "w") as f:
                            json.dump(fail_capsule, f, indent=2)
                        log.error(f"[CHUNK {chunk_idx}] Wrote CHUNK_FAIL_CAPSULE.json before invariant failure")
                    except Exception as capsule_error:
                        log.error(f"[CHUNK {chunk_idx}] Failed to write CHUNK_FAIL_CAPSULE.json: {capsule_error}")
                        # Try fallback to /tmp
                        try:
                            import tempfile
                            fallback_path = Path(tempfile.gettempdir()) / f"chunk_{chunk_idx}_FAIL_CAPSULE.json"
                            with open(fallback_path, "w") as f:
                                json.dump({
                                    "chunk_idx": chunk_idx,
                                    "exception_type": "BARS_SKIP_INVARIANT_FAIL",
                                    "exception_message": f"skipped={skipped} != warmup_skipped={warmup_skipped} + pregate_skipped={pregate_skipped} = {expected_skipped}",
                                    "capsule_write_error": str(capsule_error),
                                }, f, indent=2)
                            log.error(f"[CHUNK {chunk_idx}] Wrote CHUNK_FAIL_CAPSULE.json to fallback: {fallback_path}")
                        except Exception:
                            pass
                    
                    if panic_mode:
                        log.error(f"[CHUNK {chunk_idx}] PANIC_MODE=1: Killing process after writing capsule")
                        import os as os_kill
                        os_kill._exit(1)
                    
                    raise RuntimeError(
                        f"[BARS_SKIP_INVARIANT_FAIL] skipped={skipped} != warmup_skipped={warmup_skipped} + pregate_skipped={pregate_skipped} = {expected_skipped}. "
                        f"This indicates bars are being skipped for reasons other than warmup/eligibility. "
                        f"candles_iterated={candles_iterated}, processed={processed}, reached_entry_stage={reached_entry_stage}"
                    )
                
                # Add effective gate values and override flags (Gate-SSoT)
                if runner:
                    warmup_required_effective = getattr(runner, "warmup_required_effective", None)
                    warmup_override_applied = getattr(runner, "warmup_override_applied", False)
                    pregate_enabled_effective = getattr(runner, "pregate_enabled_effective", pregate_enabled)
                    pregate_override_applied = getattr(runner, "pregate_override_applied", False)
                    
                    chunk_footer["warmup_required_effective"] = warmup_required_effective
                    chunk_footer["warmup_override_applied"] = warmup_override_applied
                    chunk_footer["pregate_enabled_effective"] = pregate_enabled_effective
                    chunk_footer["pregate_override_applied"] = pregate_override_applied
                else:
                    chunk_footer["warmup_required_effective"] = None
                    chunk_footer["warmup_override_applied"] = False
                    chunk_footer["pregate_enabled_effective"] = pregate_enabled
                    chunk_footer["pregate_override_applied"] = False
                
                # Add warmup info if available from runner (legacy fields, kept for compatibility)
                warmup_required = None
                warmup_seen = None
                if runner:
                    # Try to get warmup info from runner
                    if hasattr(runner, "warmup_floor") and runner.warmup_floor is not None:
                        warmup_required = True
                    elif hasattr(runner, "replay_eval_start_ts") and runner.replay_eval_start_ts is not None:
                        warmup_required = True
                    warmup_seen = getattr(runner, "n_bars_skipped_due_to_htf_warmup", None)
                chunk_footer["warmup_required"] = warmup_required
                chunk_footer["warmup_seen"] = warmup_seen
                
                if prebuilt_used:
                    prebuilt_path = os_module.getenv("GX1_REPLAY_PREBUILT_FEATURES_PATH", "")
                    if runner and hasattr(runner, "prebuilt_features_path_resolved"):
                        prebuilt_path = runner.prebuilt_features_path_resolved
                    chunk_footer["prebuilt_path"] = prebuilt_path
                    if runner and hasattr(runner, "prebuilt_features_sha256") and runner.prebuilt_features_sha256:
                        chunk_footer["features_file_sha256"] = runner.prebuilt_features_sha256
                    # Add bypass count (number of bars that used prebuilt features)
                    prebuilt_bypass_count = getattr(runner, "prebuilt_bypass_count", 0) if runner else 0
                    chunk_footer["prebuilt_bypass_count"] = prebuilt_bypass_count
                    
                    # Add lookup telemetry (SSoT counters)
                    # ATOMIC LOOKUP ACCOUNTING: Get from PrebuiltFeaturesLoader if available (SSoT)
                    prebuilt_loader = getattr(runner, "prebuilt_features_loader", None) if runner else None
                    if prebuilt_loader is not None:
                        # Use atomic accounting from loader (SSoT)
                        lookup_accounting = prebuilt_loader.get_lookup_accounting()
                        lookup_attempts = lookup_accounting["attempts"]
                        lookup_hits = lookup_accounting["hits"]
                        lookup_misses = lookup_accounting["misses"]
                        lookup_miss_details = lookup_accounting["miss_details"]
                        
                        # Fail-fast invariant check
                        if not lookup_accounting["invariant_holds"]:
                            raise RuntimeError(
                                f"[PREBUILT_LOOKUP_INVARIANT_FAIL] Atomic accounting invariant violated: "
                                f"hits={lookup_hits} + misses={lookup_misses} != attempts={lookup_attempts}. "
                                f"This should be impossible with atomic lookup accounting."
                            )
                    else:
                        # Fallback: use runner-level counters (backward compatibility)
                        lookup_attempts = getattr(runner, "lookup_attempts", 0) if runner else 0
                        lookup_hits = getattr(runner, "lookup_hits", 0) if runner else 0
                        lookup_misses = getattr(runner, "lookup_misses", 0) if runner else 0
                        lookup_miss_details = getattr(runner, "lookup_miss_details", []) if runner else []
                    lookup_phase = getattr(runner, "prebuilt_lookup_phase", "unknown") if runner else "unknown"
                    
                    # Get prebuilt loader metadata if available
                    prebuilt_index_aligned = False
                    subset_first_ts = None
                    subset_last_ts = None
                    subset_rows = 0
                    if runner and hasattr(runner, "prebuilt_features_loader") and runner.prebuilt_features_loader:
                        loader = runner.prebuilt_features_loader
                        prebuilt_index_aligned = getattr(loader, "prebuilt_index_aligned", False)
                        subset_first_ts = str(getattr(loader, "subset_first_ts", None)) if getattr(loader, "subset_first_ts", None) else None
                        subset_last_ts = str(getattr(loader, "subset_last_ts", None)) if getattr(loader, "subset_last_ts", None) else None
                        subset_rows = getattr(loader, "subset_rows", 0)
                    
                    chunk_footer["prebuilt_lookup_attempts"] = lookup_attempts
                    chunk_footer["prebuilt_lookup_hits"] = lookup_hits
                    chunk_footer["prebuilt_lookup_misses"] = lookup_misses
                    chunk_footer["prebuilt_lookup_phase"] = lookup_phase
                    chunk_footer["prebuilt_index_aligned"] = prebuilt_index_aligned
                    chunk_footer["prebuilt_subset_first_ts"] = subset_first_ts
                    chunk_footer["prebuilt_subset_last_ts"] = subset_last_ts
                    chunk_footer["prebuilt_subset_rows"] = subset_rows
                    chunk_footer["lookup_miss_details"] = lookup_miss_details[:3]  # First 3 misses only
                    
                    # Legacy fields for backward compatibility
                    chunk_footer["lookup_attempts"] = lookup_attempts
                    chunk_footer["lookup_hits"] = lookup_hits
                    chunk_footer["lookup_misses"] = lookup_misses
                    
                    # STEG 3: Assert prebuilt_bypass_count == lookup_hits (SSoT)
                    if prebuilt_bypass_count != lookup_hits:
                        log.error(
                            "[PREBUILT_FAIL] prebuilt_bypass_count=%d != lookup_hits=%d. "
                            "This indicates bypass_count is incremented incorrectly. "
                            "SSoT: bypass_count must equal lookup_hits.",
                            prebuilt_bypass_count, lookup_hits
                        )
                        # Don't raise here - let tripwire handle it, but log the mismatch
                    
                    # Add feature_build_call_count (should be 0 in prebuilt-run)
                    # FASE 1: Use PREBUILT-safe tripwire counter (does not import basic_v1)
                    from gx1.execution.feature_build_tripwires import (
                        get_feature_build_call_count,
                        get_feature_build_call_details,
                    )
                    feature_build_calls = get_feature_build_call_count()
                    feature_build_details = get_feature_build_call_details()
                    chunk_footer["feature_build_call_count"] = feature_build_calls
                    chunk_footer["feature_build_call_details"] = feature_build_details
                    # Legacy field for backward compatibility
                    chunk_footer["basic_v1_call_count"] = feature_build_details.get("basic_v1.build_basic_v1", 0)
                    
                    # HARD INVARIANT: If prebuilt_used=True, bypass_count should be > 0
                    if prebuilt_bypass_count == 0 and bars_processed > 0:
                        log.warning(
                            "[PREBUILT_FAIL] prebuilt_used=True but prebuilt_bypass_count=0. "
                            "This indicates prebuilt features were loaded but not used."
                        )
                    
                    # FASE 2: Tripwire - feature_build_call_count must be 0
                    if feature_build_calls > 0:
                        raise RuntimeError(
                            f"[PREBUILT_FAIL] FASE_2_TRIPWIRE: feature_build_call_count={feature_build_calls} > 0 in prebuilt-run (expected 0). "
                            f"Details: {feature_build_details}. "
                            f"This indicates feature-building functions were called despite prebuilt features being enabled. "
                            f"CRASH: Feature-building is forbidden in PREBUILT mode."
                        )
                    
                    # FASE 2: Tripwire - FEATURE_BUILD_TIMEOUT must be 0
                    if feature_timeout_count > 0:
                        raise RuntimeError(
                            f"[PREBUILT_FAIL] FASE_2_TRIPWIRE: FEATURE_BUILD_TIMEOUT={feature_timeout_count} > 0 in prebuilt-run (expected 0). "
                            f"This indicates feature-building timed out despite prebuilt features being enabled. "
                            f"CRASH: Feature-building is forbidden in PREBUILT mode."
                        )
                    
                    # FASE 2: Tripwire - feature_time_mean_ms must be <= 5ms
                    if feature_time_mean_ms is not None and feature_time_mean_ms > 5.0:
                        raise RuntimeError(
                            f"[PREBUILT_FAIL] FASE_2_TRIPWIRE: feature_time_mean_ms={feature_time_mean_ms:.2f}ms > 5ms in prebuilt-run. "
                            f"This indicates feature-building is still happening. "
                            f"CRASH: Feature-building is forbidden in PREBUILT mode."
                        )
                    
                    # FASE 2: Lookup invariant - depends on lookup_phase
                    # Always assert: hits + misses == attempts
                    if lookup_hits + lookup_misses != lookup_attempts:
                        raise RuntimeError(
                            f"[PREBUILT_LOOKUP_INVARIANT_FAIL] lookup_hits={lookup_hits} + lookup_misses={lookup_misses} != lookup_attempts={lookup_attempts}. "
                            f"This indicates counter mismatch. All lookup attempts must result in either hit or miss."
                        )
                    
                    # Phase-specific invariant
                    bar_counters = compute_bar_counters_snapshot(runner, bars_processed, chunk_df)
                    candles_iterated = bar_counters.get("candles_iterated", 0)
                    warmup_skipped = bar_counters.get("warmup_skipped", 0)
                    bars_reaching_entry_stage = bar_counters.get("reached_entry_stage", 0)
                    
                    # Get entry universe counters from telemetry
                    bars_passed_hard_eligibility = 0
                    bars_blocked_hard_eligibility = 0
                    bars_passed_soft_eligibility = 0
                    bars_blocked_soft_eligibility = 0
                    if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                        em = runner.entry_manager
                        if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                            telemetry = em.entry_feature_telemetry
                            bars_passed_hard_eligibility = telemetry.bars_passed_hard_eligibility
                            bars_blocked_hard_eligibility = telemetry.bars_blocked_hard_eligibility
                            bars_passed_soft_eligibility = telemetry.bars_passed_soft_eligibility
                            bars_blocked_soft_eligibility = telemetry.bars_blocked_soft_eligibility
                    
                    # Store in chunk_footer for debugging
                    chunk_footer["bars_passed_hard_eligibility"] = bars_passed_hard_eligibility
                    chunk_footer["bars_blocked_hard_eligibility"] = bars_blocked_hard_eligibility
                    chunk_footer["bars_passed_soft_eligibility"] = bars_passed_soft_eligibility
                    chunk_footer["bars_blocked_soft_eligibility"] = bars_blocked_soft_eligibility
                    
                    # Sanity invariants
                    if bars_passed_hard_eligibility > bars_reaching_entry_stage:
                        raise RuntimeError(
                            f"[PREBUILT_LOOKUP_INVARIANT_FAIL] bars_passed_hard_eligibility={bars_passed_hard_eligibility} > bars_reaching_entry_stage={bars_reaching_entry_stage}. "
                            f"This indicates counter mismatch. Hard eligibility cannot pass more bars than reached entry stage."
                        )
                    
                    if bars_passed_soft_eligibility > bars_passed_hard_eligibility:
                        raise RuntimeError(
                            f"[PREBUILT_LOOKUP_INVARIANT_FAIL] bars_passed_soft_eligibility={bars_passed_soft_eligibility} > bars_passed_hard_eligibility={bars_passed_hard_eligibility}. "
                            f"This indicates counter mismatch. Soft eligibility cannot pass more bars than hard eligibility."
                        )
                    
                    # Phase-specific invariant
                    if lookup_phase == "before_pregate":
                        # If lookup happens before pregate, attempts should equal all bars after warmup
                        expected_attempts = candles_iterated - warmup_skipped
                        if lookup_attempts != expected_attempts:
                            raise RuntimeError(
                                f"[PREBUILT_LOOKUP_INVARIANT_FAIL] lookup_phase='before_pregate': "
                                f"lookup_attempts={lookup_attempts} != candles_iterated={candles_iterated} - warmup_skipped={warmup_skipped} = {expected_attempts}. "
                                f"This indicates lookup was not attempted for all bars after warmup."
                            )
                    elif lookup_phase == "after_hard_eligibility":
                        # If lookup happens after hard eligibility, attempts should equal bars that passed hard eligibility
                        expected_attempts = bars_passed_hard_eligibility
                        if lookup_attempts != expected_attempts:
                            raise RuntimeError(
                                f"[PREBUILT_LOOKUP_INVARIANT_FAIL] lookup_phase='after_hard_eligibility': "
                                f"lookup_attempts={lookup_attempts} != bars_passed_hard_eligibility={bars_passed_hard_eligibility}. "
                                f"bars_reaching_entry_stage={bars_reaching_entry_stage}, "
                                f"bars_blocked_hard_eligibility={bars_blocked_hard_eligibility}, "
                                f"warmup_skipped={warmup_skipped}, pregate_skipped={bar_counters.get('pregate_skipped', 0)}. "
                                f"This indicates lookup was not attempted for all bars that passed hard eligibility."
                            )
                    elif lookup_phase == "after_pregate":
                        # Legacy phase name - treat as after_hard_eligibility
                        expected_attempts = bars_passed_hard_eligibility
                        if lookup_attempts != expected_attempts:
                            raise RuntimeError(
                                f"[PREBUILT_LOOKUP_INVARIANT_FAIL] lookup_phase='after_pregate' (legacy): "
                                f"lookup_attempts={lookup_attempts} != bars_passed_hard_eligibility={bars_passed_hard_eligibility}. "
                                f"This indicates lookup was not attempted for all bars that passed hard eligibility."
                            )
                    
                    # FASE 2: Tripwire - All eligible bars must use prebuilt features
                    # Semantikk: "Alle bars som ER eligible og når lookup, må bruke prebuilt"
                    # lookup_misses representerer hard eligibility blocks (faktiske KeyError hard-failer umiddelbart)
                    # Tripwire: lookup_hits == lookup_attempts - lookup_misses (eligibility blocks)
                    eligibility_blocks = lookup_misses  # Hard eligibility blocks (faktiske KeyError hard-failer allerede)
                    expected_prebuilt_hits = lookup_attempts - eligibility_blocks
                    
                    # Logg tripwire-detaljer til footer
                    chunk_footer["tripwire_eligibility_blocks"] = eligibility_blocks
                    chunk_footer["tripwire_expected_prebuilt_hits"] = expected_prebuilt_hits
                    chunk_footer["tripwire_passed"] = (lookup_hits == expected_prebuilt_hits)
                    
                    if lookup_hits != expected_prebuilt_hits:
                        # CRITICAL: Write PREBUILT_FAIL_CAPSULE.json BEFORE raising
                        try:
                            from gx1.utils.atomic_json import atomic_write_json
                            import sys
                            import traceback as tb_module
                            
                            # Find first forbidden module in import chain
                            forbidden_modules_present = []
                            forbidden_module_first = None
                            forbidden_stack = None
                            
                            try:
                                # Check sys.modules for forbidden modules
                                forbidden_patterns = ["gx1.features.basic_v1", "gx1.execution.live_features"]
                                for mod_name in sys.modules.keys():
                                    for pattern in forbidden_patterns:
                                        if pattern in mod_name:
                                            forbidden_modules_present.append(mod_name)
                                            if forbidden_module_first is None:
                                                forbidden_module_first = mod_name
                                
                                # Get current stack
                                forbidden_stack = "".join(tb_module.format_stack())
                            except Exception:
                                pass
                            
                            # Get lookup counters and phase for capsule
                            lookup_attempts_capsule = getattr(runner, "lookup_attempts", 0) if runner else 0
                            lookup_hits_capsule = getattr(runner, "lookup_hits", 0) if runner else 0
                            lookup_misses_capsule = getattr(runner, "lookup_misses", 0) if runner else 0
                            lookup_phase_capsule = getattr(runner, "prebuilt_lookup_phase", "unknown") if runner else "unknown"
                            
                            # Get bar counters for capsule
                            bar_counters_capsule = compute_bar_counters_snapshot(
                                runner if 'runner' in locals() else None,
                                bars_processed if 'bars_processed' in locals() else 0,
                                chunk_df if 'chunk_df' in locals() else None,
                            )
                            
                            # Build capsule payload
                            prebuilt_capsule = {
                                "chunk_idx": chunk_idx,
                                "run_id": run_id,
                                "failure_type": "PREBUILT_FAIL_FASE_2_TRIPWIRE",
                                "replay_mode": "PREBUILT",
                                "policy_id": policy_id if policy_id else None,
                                "bundle_sha256": bundle_sha256 if bundle_sha256 else None,
                                "prebuilt_lookup_attempts": lookup_attempts_capsule,
                                "prebuilt_lookup_hits": lookup_hits_capsule,
                                "prebuilt_lookup_misses": lookup_misses_capsule,
                                "prebuilt_lookup_phase": lookup_phase_capsule,
                                "lookup_hits": lookup_hits_capsule,  # Legacy
                                "expected_prebuilt_hits": expected_prebuilt_hits,
                                "lookup_attempts": lookup_attempts_capsule,  # Legacy
                                "eligibility_blocks": eligibility_blocks,
                                "bar_counters": bar_counters_capsule,
                                "forbidden_module_first": forbidden_module_first,
                                "forbidden_modules_present": forbidden_modules_present,
                                "stack": forbidden_stack[:5000] if forbidden_stack else None,
                                "argv": sys.argv.copy() if hasattr(sys, 'argv') else None,
                                "cwd": str(Path.cwd()),
                                "sys_executable": sys.executable if hasattr(sys, 'executable') else None,
                                "sys_path_head": sys.path[:10] if hasattr(sys, 'path') else None,
                                "timestamp": dt_now_iso(),
                            }
                            
                            # BONUS: Print user-friendly error message with forbidden module and top stack frames
                            if forbidden_module_first:
                                log.error("=" * 80)
                                log.error(f"[PREBUILT_FAIL] FASE_2_TRIPWIRE triggered!")
                                log.error(f"  Forbidden module found: {forbidden_module_first}")
                                log.error(f"  All forbidden modules present: {forbidden_modules_present[:5]}")
                                
                                # Extract top 5 stack frames (file:line:func)
                                if forbidden_stack:
                                    stack_lines = forbidden_stack.split('\n')
                                    # Filter out empty lines and extract meaningful frames
                                    frames = []
                                    for line in stack_lines:
                                        if 'File "' in line and ', line ' in line:
                                            # Extract file:line:func from stack line
                                            # Format: File "/path/to/file.py", line 123, in function_name
                                            try:
                                                parts = line.split('File "')
                                                if len(parts) > 1:
                                                    file_part = parts[1].split('", line ')[0]
                                                    line_part = parts[1].split('", line ')[1].split(',')[0]
                                                    func_part = parts[1].split('in ')[-1].strip() if 'in ' in parts[1] else '<module>'
                                                    file_name = Path(file_part).name
                                                    frames.append(f"{file_name}:{line_part}:{func_part}")
                                            except Exception:
                                                pass
                                    
                                    if frames:
                                        log.error(f"  Top {min(5, len(frames))} stack frames:")
                                        for i, frame in enumerate(frames[:5], 1):
                                            log.error(f"    {i}. {frame}")
                                log.error("=" * 80)
                            
                            # Write capsule (try chunk_dir, then run root, then /tmp)
                            capsule_written = False
                            if chunk_output_dir and chunk_output_dir.exists():
                                capsule_path = chunk_output_dir / "PREBUILT_FAIL_CAPSULE.json"
                                if atomic_write_json(capsule_path, prebuilt_capsule):
                                    capsule_written = True
                                    log.error(f"[CHUNK {chunk_idx}] Wrote PREBUILT_FAIL_CAPSULE.json to {capsule_path}")
                            
                            if not capsule_written:
                                # Try run root
                                run_root = chunk_output_dir.parent if chunk_output_dir else output_dir
                                if run_root and run_root.exists():
                                    capsule_path = run_root / f"PREBUILT_FAIL_CAPSULE_chunk_{chunk_idx}.json"
                                    if atomic_write_json(capsule_path, prebuilt_capsule):
                                        capsule_written = True
                                        log.error(f"[CHUNK {chunk_idx}] Wrote PREBUILT_FAIL_CAPSULE.json to {capsule_path}")
                            
                            if not capsule_written:
                                # Fallback to /tmp
                                import tempfile
                                fallback_path = Path(tempfile.gettempdir()) / f"PREBUILT_FAIL_CAPSULE_chunk_{chunk_idx}_{run_id}.json"
                                if atomic_write_json(fallback_path, prebuilt_capsule):
                                    log.error(f"[CHUNK {chunk_idx}] Wrote PREBUILT_FAIL_CAPSULE.json to fallback: {fallback_path}")
                        except Exception as capsule_error:
                            log.error(f"[CHUNK {chunk_idx}] Failed to write PREBUILT_FAIL_CAPSULE.json: {capsule_error}", exc_info=True)
                        
                        raise RuntimeError(
                            f"[PREBUILT_FAIL] FASE_2_TRIPWIRE: Expected all ELIGIBLE bars to use prebuilt features. "
                            f"lookup_hits={lookup_hits}, expected={expected_prebuilt_hits} "
                            f"(lookup_attempts={lookup_attempts}, eligibility_blocks={eligibility_blocks}). "
                            f"This indicates some eligible bars did not use prebuilt features. "
                            f"CRASH: All eligible bars must use prebuilt features in PREBUILT mode."
                        )
                else:
                    chunk_footer["prebuilt_path"] = None
                    chunk_footer["features_file_sha256"] = None
                    chunk_footer["prebuilt_bypass_count"] = 0
                    chunk_footer["basic_v1_call_count"] = None
                
                # Add SSoT bundle_sha256 to chunk footer
                if bundle_sha256:
                    chunk_footer["ssot"] = {
                        "bundle_sha256": bundle_sha256,
                    }
                else:
                    # HARD-FAIL: bundle_sha256 must be present
                    raise RuntimeError(
                        "[SSOT_FAIL] bundle_sha256 is missing in chunk footer. "
                        "This should never happen - bundle_sha256 must be computed before workers start."
                    )
                
                # Convert all values to JSON-serializable types
                chunk_footer = convert_to_json_serializable(chunk_footer)
                
                chunk_footer_path = chunk_output_dir / "chunk_footer.json"
                # ENTRY FEATURE TELEMETRY: Write telemetry files before chunk_footer (REQUIRED in replay mode)
                telemetry_written = False
                telemetry_required = os_module.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                
                # Check if no entry evaluations occurred (bars_reaching_entry_stage == 0)
                bars_reaching_entry_stage = getattr(runner, "bars_reaching_entry_stage", 0) if runner else 0
                no_entry_evaluations = bars_reaching_entry_stage == 0 and bars_processed > 0
                no_entry_reason = None
                if no_entry_evaluations:
                    # Determine reason
                    bars_skipped_warmup = getattr(runner, "bars_skipped_warmup", 0) if runner else 0
                    bars_skipped_pregate = getattr(runner, "bars_skipped_pregate", 0) if runner else 0
                    if bars_skipped_warmup == bars_processed:
                        no_entry_reason = "warmup"
                    elif bars_skipped_pregate == bars_processed:
                        no_entry_reason = "pregate_blocked_all"
                    else:
                        no_entry_reason = "unknown"
                    
                    # Set env var for telemetry collector
                    os_module.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"] = "1"
                    os_module.environ["GX1_TELEMETRY_NO_ENTRY_REASON"] = no_entry_reason
                    log.info(
                        f"[CHUNK {chunk_idx}] No entry evaluations occurred "
                        f"(bars_processed={bars_processed}, bars_reaching_entry_stage={bars_reaching_entry_stage}, reason={no_entry_reason})"
                    )
                
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        try:
                            em.entry_feature_telemetry.write_all(chunk_output_dir)
                            telemetry_written = True
                            log.info(f"[CHUNK {chunk_idx}] Entry feature telemetry written to {chunk_output_dir}")
                        except Exception as telemetry_error:
                            log.error(f"[CHUNK {chunk_idx}] Failed to write entry feature telemetry: {telemetry_error}", exc_info=True)
                            if telemetry_required:
                                raise RuntimeError(
                                    f"[CHUNK {chunk_idx}] FATAL: Failed to write entry feature telemetry "
                                    f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Error: {telemetry_error}"
                                )
                    elif telemetry_required:
                        raise RuntimeError(
                            f"[CHUNK {chunk_idx}] FATAL: entry_feature_telemetry not initialized "
                            f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Check EntryManager initialization."
                        )
                elif telemetry_required:
                    raise RuntimeError(
                        f"[CHUNK {chunk_idx}] FATAL: runner or entry_manager not available "
                        f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Cannot write telemetry."
                    )
                
                # Clean up env vars
                if "GX1_TELEMETRY_NO_ENTRY_EVALUATIONS" in os_module.environ:
                    del os_module.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"]
                if "GX1_TELEMETRY_NO_ENTRY_REASON" in os_module.environ:
                    del os_module.environ["GX1_TELEMETRY_NO_ENTRY_REASON"]
                
                # Fail-fast validation if telemetry is required
                if telemetry_required and bars_processed > 0:
                    entry_features_path = chunk_output_dir / "ENTRY_FEATURES_USED.json"
                    if not entry_features_path.exists():
                        raise RuntimeError(
                            f"[CHUNK {chunk_idx}] FATAL: ENTRY_FEATURES_USED.json not found after write_all "
                            f"(GX1_REQUIRE_ENTRY_TELEMETRY=1, bars_processed={bars_processed}). "
                            f"Telemetry must be written for A/B tests."
                        )
                    
                    # Verify telemetry has non-empty feature lists (unless no_entry_evaluations is set)
                    try:
                        with open(entry_features_path, "r") as f:
                            telemetry_data = json.load(f)
                        
                        # ENTRY ROUTING VALIDATION: Check aggregated routing telemetry (SSoT)
                        entry_routing_aggregate = telemetry_data.get("entry_routing_aggregate", {})
                        routing_total_recorded = entry_routing_aggregate.get("total_recorded", 0)
                        
                        # INVARIANT 1: If bars reached entry stage, routing must have been recorded at least once
                        if bars_reaching_entry_stage > 0 and routing_total_recorded == 0:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_routing_aggregate.total_recorded=0 but bars_reaching_entry_stage={bars_reaching_entry_stage} > 0. "
                                f"This indicates entry routing telemetry was never collected. "
                                f"entry_routing_aggregate={entry_routing_aggregate}"
                            )
                        
                        # INVARIANT 2: If v10_session_supported gate passed, routing should have been recorded
                        gate_stats = telemetry_data.get("gate_stats", {})
                        v10_session_gate = gate_stats.get("v10_session_supported", {})
                        v10_session_passed = v10_session_gate.get("passed", 0)
                        if v10_session_passed > 0 and routing_total_recorded < v10_session_passed:
                            log.warning(
                                f"[CHUNK {chunk_idx}] entry_routing_aggregate.total_recorded={routing_total_recorded} < "
                                f"v10_session_supported_passed={v10_session_passed}. "
                                f"This may indicate some bars passed the gate but routing was not recorded. "
                                f"entry_routing_aggregate={entry_routing_aggregate}"
                            )
                        
                        # V10 ENABLE-STATE INVARIANTS: Validate enable state telemetry
                        entry_v10_enable_state = telemetry_data.get("entry_v10_enable_state", {})
                        v10_enabled = entry_v10_enable_state.get("enabled")
                        v10_enabled_reason = entry_v10_enable_state.get("reason")
                        v10_enabled_true_count = entry_v10_enable_state.get("enabled_true_count", 0)
                        v10_enabled_false_count = entry_v10_enable_state.get("enabled_false_count", 0)
                        v10_enabled_reason_counts = entry_v10_enable_state.get("reason_counts", {})
                        
                        # INVARIANT 1: Enable state must be recorded (not None/unknown)
                        if v10_enabled is None and bars_reaching_entry_stage > 0:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state.enabled is None but bars_reaching_entry_stage={bars_reaching_entry_stage} > 0. "
                                f"This indicates enable state telemetry was not collected. "
                                f"entry_v10_enable_state={entry_v10_enable_state}"
                            )
                        
                        # INVARIANT 2: If V10 is disabled, reason must be non-empty
                        if v10_enabled is False and not v10_enabled_reason:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state.enabled=False but reason is empty. "
                                f"V10 disable reason must be recorded. "
                                f"entry_v10_enable_state={entry_v10_enable_state}"
                            )
                        
                        # CONTROL-FLOW INVARIANTS: Track execution path
                        control_flow = telemetry_data.get("control_flow", {})
                        control_flow_counts = control_flow.get("counts", {})
                        control_flow_last = control_flow.get("last")
                        
                        after_enable_check = control_flow_counts.get("AFTER_V10_ENABLE_CHECK", 0)
                        enter_routing_branch = control_flow_counts.get("ENTER_V10_ROUTING_BRANCH", 0)
                        
                        # INVARIANT 1: AFTER_V10_ENABLE_CHECK > 0 when entry_v10_enabled=True
                        if v10_enabled is True and after_enable_check == 0:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state.enabled=True but "
                                f"control_flow.AFTER_V10_ENABLE_CHECK=0. "
                                f"This indicates enable-state telemetry was recorded but control-flow sentinel was not. "
                                f"entry_v10_enable_state={entry_v10_enable_state}, control_flow={control_flow}"
                            )
                        
                        # INVARIANT 2: EXIT_BEFORE_V10_ROUTING fail-fast
                        # If V10 is enabled, we passed enable check, but never entered routing branch
                        if (v10_enabled is True and 
                            bars_reaching_entry_stage > 0 and 
                            after_enable_check > 0 and 
                            enter_routing_branch == 0):
                            
                            # Get last control-flow event for debugging
                            last_event_info = f"last_control_flow={control_flow_last}" if control_flow_last else "no_control_flow_last"
                            
                            # Get stack trace to pinpoint exact early return
                            import traceback
                            import sys
                            current_traceback = "".join(traceback.format_stack())
                            
                            error_msg = (
                                f"[CHUNK {chunk_idx}] FATAL: EXIT_BEFORE_V10_ROUTING\n"
                                f"entry_v10_enabled=True, bars_reaching_entry_stage={bars_reaching_entry_stage}, "
                                f"AFTER_V10_ENABLE_CHECK={after_enable_check}, ENTER_V10_ROUTING_BRANCH={enter_routing_branch}\n"
                                f"entry_v10_enabled_reason={v10_enabled_reason}\n"
                                f"{last_event_info}\n"
                                f"control_flow_counts={control_flow_counts}\n"
                                f"Stack trace:\n{current_traceback}"
                            )
                            
                            raise RuntimeError(error_msg)
                        
                        # Initialize selected_model_counts early (before use in invariants)
                        selected_model_counts = entry_routing_aggregate.get("selected_model_counts", {})
                        reason_counts = entry_routing_aggregate.get("reason_counts", {})
                        
                        # INVARIANT 3: If V10 is enabled and bars reached entry stage, routing should occur
                        if v10_enabled is True and bars_reaching_entry_stage > 0:
                            v10_routed_count = selected_model_counts.get("v10_hybrid", 0)
                            if v10_routed_count == 0:
                                # Check if there's an explicit fallback reason
                                fallback_reasons = ["V10_SESSION_NOT_SUPPORTED", "ROUTED_TO_V9_FALLBACK"]
                                routing_reasons = entry_routing_aggregate.get("reason_counts", {})
                                has_fallback = any(reason in routing_reasons for reason in fallback_reasons)
                                if not has_fallback:
                                    log.warning(
                                        f"[CHUNK {chunk_idx}] entry_v10_enable_state.enabled=True and bars_reaching_entry_stage={bars_reaching_entry_stage} > 0 "
                                        f"but ROUTED_TO_V10=0 and no explicit fallback reason. "
                                        f"This may indicate a routing bug. "
                                        f"entry_v10_enable_state={entry_v10_enable_state}, entry_routing_aggregate={entry_routing_aggregate}"
                                    )
                        
                        # INVARIANT 4: ROUTED_TO_V10 > 0 => ENTER_V10_ROUTING_BRANCH > 0
                        v10_routed_count = selected_model_counts.get("v10_hybrid", 0)
                        if v10_routed_count > 0 and enter_routing_branch == 0:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: ROUTED_TO_V10={v10_routed_count} > 0 but "
                                f"control_flow.ENTER_V10_ROUTING_BRANCH=0. "
                                f"This should be IMPOSSIBLE - routing can only happen in routing branch. "
                                f"entry_routing_aggregate={entry_routing_aggregate}, control_flow={control_flow}"
                            )
                        
                        # INVARIANT 5: ROUTED_TO_V10 > 0 => v10_callsite.entered > 0 (already covered by INVARIANT A)
                        
                        # PRE-ENTRY FUNNEL INVARIANTS: Validate that entry evaluation actually happens
                        # Get pre-entry funnel from chunk_footer (not telemetry_data, as it's runner-level)
                        chunk_footer_path = chunk_output_dir / "chunk_footer.json"
                        pre_entry_funnel = {}
                        if chunk_footer_path.exists():
                            try:
                                with open(chunk_footer_path) as f:
                                    chunk_footer_data = json.load(f)
                                    pre_entry_funnel = chunk_footer_data.get("pre_entry_funnel", {})
                            except Exception:
                                pass
                        
                        candles_iterated = pre_entry_funnel.get("candles_iterated", 0)
                        evaluate_entry_called_count = pre_entry_funnel.get("evaluate_entry_called_count", 0)
                        warmup_skipped = pre_entry_funnel.get("warmup_skipped", 0)
                        pregate_skipped = pre_entry_funnel.get("pregate_skipped", 0)
                        prebuilt_missing_skipped = pre_entry_funnel.get("prebuilt_missing_skipped", 0)
                        last_stop_reason = pre_entry_funnel.get("last_stop_reason")
                        
                        # INVARIANT: NO_ENTRY_EVALUATIONS fail-fast
                        # If candles were iterated but evaluate_entry was never called, fail with diagnostic info
                        if candles_iterated > 0 and evaluate_entry_called_count == 0:
                            # Determine dominant cause
                            if warmup_skipped == candles_iterated:
                                hint = "ALL_WARMUP"
                            elif pregate_skipped == candles_iterated - warmup_skipped:
                                hint = "ALL_PREGATE"
                            elif prebuilt_missing_skipped > 0:
                                hint = "PREBUILT_MISSING"
                            else:
                                hint = "UNKNOWN_EARLY_EXIT"
                            
                            # Get smoke args if available (from env or chunk metadata)
                            smoke_bars = os.getenv("GX1_SMOKE_BARS", "N/A")
                            smoke_date_range = os.getenv("GX1_SMOKE_DATE_RANGE", "N/A")
                            
                            error_msg = (
                                f"[CHUNK {chunk_idx}] FATAL: NO_ENTRY_EVALUATIONS\n"
                                f"candles_iterated={candles_iterated} > 0 but evaluate_entry_called_count=0\n"
                                f"Funnel breakdown:\n"
                                f"  - warmup_skipped: {warmup_skipped}\n"
                                f"  - pregate_skipped: {pregate_skipped}\n"
                                f"  - prebuilt_missing_skipped: {prebuilt_missing_skipped}\n"
                                f"  - last_stop_reason: {last_stop_reason}\n"
                                f"PRE_ENTRY_TRACE_HINT: {hint}\n"
                                f"Smoke args: smoke_bars={smoke_bars}, smoke_date_range={smoke_date_range}\n"
                                f"pre_entry_funnel={pre_entry_funnel}"
                            )
                            
                            if require_telemetry:
                                raise RuntimeError(error_msg)
                            else:
                                log.error(error_msg)
                        
                        # V10 CALLSITE BRIDGE INVARIANTS: Validate gap between routing and model call
                        v10_callsite = telemetry_data.get("v10_callsite", {})
                        callsite_entered = v10_callsite.get("entered", 0)
                        callsite_returned = v10_callsite.get("returned", 0)
                        callsite_exception = v10_callsite.get("exception", 0)
                        callsite_last_exception = v10_callsite.get("last_exception")
                        
                        # selected_model_counts and reason_counts already initialized above
                        
                        model_entry = telemetry_data.get("model_entry", {})
                        model_attempt_calls = model_entry.get("model_attempt_calls", {})
                        model_attempt_total = sum(model_attempt_calls.values())
                        
                        # INVARIANT A (HARD): If V10 was routed to, callsite MUST have been entered
                        # This is now guaranteed by execution order: callsite.entered happens BEFORE ROUTED_TO_V10
                        v10_routed_count = selected_model_counts.get("v10_hybrid", 0)
                        if v10_routed_count > 0 and callsite_entered == 0:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_routing_aggregate.selected_model_counts['v10_hybrid']={v10_routed_count} > 0 "
                                f"but v10_callsite.entered=0. "
                                f"This should be IMPOSSIBLE after refactor (callsite.entered happens before ROUTED_TO_V10). "
                                f"This indicates a semantic bug in routing telemetry placement. "
                                f"entry_routing_aggregate={entry_routing_aggregate}, v10_callsite={v10_callsite}"
                            )
                        
                        # INVARIANT B (CONSISTENCY): callsite.entered >= routed_count
                        # (entered can be > if we count per-callsite vs per-bar, but should never be <)
                        if callsite_entered < v10_routed_count:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: v10_callsite.entered={callsite_entered} < "
                                f"entry_routing_aggregate.selected_model_counts['v10_hybrid']={v10_routed_count}. "
                                f"This is impossible - we cannot route to V10 without entering callsite. "
                                f"This indicates a telemetry counting bug."
                            )
                        
                        # INVARIANT C: If callsite was entered, model_attempt_calls must be > 0
                        if callsite_entered > 0 and model_attempt_total == 0:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: v10_callsite.entered={callsite_entered} > 0 "
                                f"but model_attempt_calls=0. "
                                f"This indicates call site was entered but _predict_entry_v10_hybrid was never called, "
                                f"or record_model_attempt() was not called inside the function. "
                                f"v10_callsite={v10_callsite}, model_entry={model_entry}"
                            )
                        
                        # INVARIANT D: If exception occurred, it must be captured and run should fail clearly
                        if callsite_exception > 0:
                            if not callsite_last_exception:
                                raise RuntimeError(
                                    f"[CHUNK {chunk_idx}] FATAL: v10_callsite.exception={callsite_exception} > 0 "
                                    f"but v10_callsite.last_exception is missing. "
                                    f"This indicates exception telemetry was not properly recorded."
                                )
                            # Exception details should be in capsule, but we can log here
                            log.error(
                                f"[CHUNK {chunk_idx}] V10 callsite exception occurred: {callsite_last_exception}. "
                                f"This should have caused the run to fail. Check CHUNK_FAIL_CAPSULE.json for details."
                            )
                            # In require-telemetry mode: fail run (no silent fallback)
                            if require_telemetry:
                                raise RuntimeError(
                                    f"[CHUNK {chunk_idx}] FATAL: V10 callsite exception occurred and GX1_REQUIRE_ENTRY_TELEMETRY=1. "
                                    f"Run must fail - no silent fallback allowed. "
                                    f"Exception: {callsite_last_exception}"
                                )
                        
                        # INVARIANT E: Sanity check - returned <= entered
                        if callsite_returned > callsite_entered:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: v10_callsite.returned={callsite_returned} > "
                                f"v10_callsite.entered={callsite_entered}. "
                                f"This is impossible and indicates a telemetry bug."
                            )
                        
                        # Legacy invariant: If V10 was routed to, model_attempt_calls should be > 0
                        # (This is now redundant with invariant C, but kept for clarity)
                        if v10_routed_count > 0 and model_attempt_total == 0:
                            # This should have been caught by invariant C, but keep for backward compatibility
                            if callsite_entered == 0:
                                # Already caught by invariant A
                                pass
                            else:
                                # Should have been caught by invariant C
                                raise RuntimeError(
                                    f"[CHUNK {chunk_idx}] FATAL: entry_routing_aggregate.selected_model_counts['v10_hybrid']={v10_routed_count} > 0 "
                                    f"but model_attempt_calls=0. "
                                    f"This indicates routing to V10 was recorded but model was never attempted. "
                                    f"entry_routing_aggregate={entry_routing_aggregate}, model_entry={model_entry}, v10_callsite={v10_callsite}"
                                )
                        elif model_attempt_total > 0 and v10_routed_count == 0:
                            # If model was attempted but not routed to V10, this is inconsistent
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: model_attempt_calls={model_attempt_total} > 0 but "
                                f"entry_routing_aggregate.selected_model_counts['v10_hybrid']={v10_routed_count} == 0. "
                                f"This indicates routing telemetry is inconsistent with model entry telemetry. "
                                f"entry_routing_aggregate={entry_routing_aggregate}, model_entry={model_entry}"
                            )
                        
                        # Check for no_entry_evaluations flag
                        if telemetry_data.get("no_entry_evaluations", False):
                            log.info(
                                f"[CHUNK {chunk_idx}] No entry evaluations occurred (expected) "
                                f"(reason: {telemetry_data.get('no_entry_evaluations_reason', 'unknown')})"
                            )
                        else:
                            # Verify feature lists are non-empty
                            seq_count = telemetry_data.get("seq_features", {}).get("count", 0)
                            snap_count = telemetry_data.get("snap_features", {}).get("count", 0)
                            
                            if seq_count == 0 or snap_count == 0:
                                raise RuntimeError(
                                    f"[CHUNK {chunk_idx}] FATAL: Telemetry has empty feature lists "
                                    f"(seq_count={seq_count}, snap_count={snap_count}, bars_processed={bars_processed}). "
                                    f"This indicates telemetry was not collected during entry evaluation."
                                )
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"[CHUNK {chunk_idx}] FATAL: ENTRY_FEATURES_USED.json is invalid JSON "
                            f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Error: {e}"
                        )
                
                with open(chunk_footer_path, "w") as f:
                    json.dump(chunk_footer, f, indent=2)
                
                log.info(f"[CHUNK {chunk_idx}] chunk_footer.json written: status={status}")
            except Exception as footer_error:
                log.error(f"[CHUNK {chunk_idx}] Failed to write chunk_footer.json: {footer_error}", exc_info=True)
                
                # CRITICAL: Write CHUNK_FAIL_CAPSULE.json BEFORE stub_footer (atomic, always valid JSON)
                try:
                    from gx1.utils.atomic_json import atomic_write_json
                    import traceback as tb_module
                    import sys
                    
                    error_traceback = "".join(tb_module.format_exception(type(footer_error), footer_error, footer_error.__traceback__))
                    
                    # Compute bar counters snapshot (safe defaults, never NameError)
                    try:
                        bars_processed_safe = bars_processed if 'bars_processed' in locals() else 0
                    except Exception:
                        bars_processed_safe = 0
                    
                    bar_counters = compute_bar_counters_snapshot(
                        runner if 'runner' in locals() else None,
                        bars_processed_safe,
                        chunk_df if 'chunk_df' in locals() else None,
                    )
                    
                    # Get run identity data if available
                    run_identity_data = None
                    try:
                        if chunk_output_dir and chunk_output_dir.exists():
                            run_identity_path = chunk_output_dir / "run_header.json"
                            if run_identity_path.exists():
                                with open(run_identity_path, "r") as f:
                                    run_identity_data = json.load(f)
                    except Exception:
                        pass
                    
                    # Get first/last iter timestamps (safe)
                    first_iter_ts_safe = None
                    last_iter_ts_safe = None
                    try:
                        if chunk_df is not None and len(chunk_df) > 0:
                            first_iter_ts_safe = str(chunk_df.index[0])
                            last_iter_ts_safe = str(chunk_df.index[-1])
                        elif first_iter_ts:
                            first_iter_ts_safe = str(first_iter_ts)
                        if last_iter_ts:
                            last_iter_ts_safe = str(last_iter_ts)
                    except Exception:
                        pass
                    
                    # Get telemetry status (best-effort, never fail)
                    telemetry_status = {
                        "telemetry_required": os_module.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1",
                        "collector_initialized": False,
                        "telemetry_write_attempted": False,
                        "telemetry_files_written": [],
                        "no_entry_evaluations": False,
                        "no_entry_reason": None,
                    }
                    try:
                        if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                            em = runner.entry_manager
                            if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                                telemetry_status["collector_initialized"] = True
                                # Check if write was attempted (check for files)
                                if chunk_output_dir:
                                    entry_features_path = chunk_output_dir / "ENTRY_FEATURES_USED.json"
                                    if entry_features_path.exists():
                                        telemetry_status["telemetry_write_attempted"] = True
                                        telemetry_status["telemetry_files_written"].append("ENTRY_FEATURES_USED.json")
                                        
                                        # Check for no_entry_evaluations
                                        try:
                                            with open(entry_features_path, "r") as f:
                                                telemetry_data = json.load(f)
                                            if telemetry_data.get("no_entry_evaluations", False):
                                                telemetry_status["no_entry_evaluations"] = True
                                                telemetry_status["no_entry_reason"] = telemetry_data.get("no_entry_evaluations_reason")
                                        except Exception:
                                            pass
                    except Exception:
                        pass
                    
                    # Get bundle_dir_resolved if available
                    bundle_dir_resolved = None
                    try:
                        if runner and hasattr(runner, "bundle_dir_resolved"):
                            bundle_dir_resolved = str(runner.bundle_dir_resolved) if runner.bundle_dir_resolved else None
                    except Exception:
                        pass
                    
                    # Get lookup counters if available
                    lookup_attempts_capsule = 0
                    lookup_hits_capsule = 0
                    lookup_misses_capsule = 0
                    lookup_phase_capsule = "unknown"
                    try:
                        if runner:
                            lookup_attempts_capsule = getattr(runner, "lookup_attempts", 0)
                            lookup_hits_capsule = getattr(runner, "lookup_hits", 0)
                            lookup_misses_capsule = getattr(runner, "lookup_misses", 0)
                            lookup_phase_capsule = getattr(runner, "prebuilt_lookup_phase", "unknown")
                    except Exception:
                        pass
                    
                    # Get entry routing telemetry if available
                    entry_routing_capsule = {
                        "selected_model": None,
                        "reason": "TELEMETRY_NOT_AVAILABLE",
                        "recorded": False,
                    }
                    exception_gap_capsule = None
                    control_flow_last_3_capsule = []
                    entry_v10_enabled_capsule = None
                    try:
                        if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                            em = runner.entry_manager
                            if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                                telemetry = em.entry_feature_telemetry
                                entry_routing_capsule = {
                                    "selected_model": telemetry.entry_routing_selected_model,
                                    "reason": telemetry.entry_routing_reason,
                                    "recorded": telemetry.entry_routing_recorded,
                                }
                                exception_gap_capsule = telemetry.exception_gap
                                entry_v10_enabled_capsule = telemetry.entry_v10_enabled
                                # Get last 3 control flow events (for diagnostic)
                                if telemetry.control_flow_last:
                                    control_flow_last_3_capsule.append(telemetry.control_flow_last)
                    except Exception:
                        pass
                    
                    # Build capsule payload
                    fail_capsule = {
                        "chunk_idx": chunk_idx,
                        "run_id": run_id,
                        "exception_type": type(footer_error).__name__,
                        "exception_message": str(footer_error),
                        "traceback": error_traceback[:10000],  # Trim long tracebacks
                        "bar_counters": bar_counters,
                        "bars_seen": bar_counters.get("candles_iterated", 0),
                        "bars_processed": bars_processed_safe,
                        "bars_reaching_entry_stage": bar_counters.get("reached_entry_stage", 0),
                        "prebuilt_lookup_attempts": lookup_attempts_capsule,
                        "prebuilt_lookup_hits": lookup_hits_capsule,
                        "prebuilt_lookup_misses": lookup_misses_capsule,
                        "prebuilt_lookup_phase": lookup_phase_capsule,
                        "bars_passed_hard_eligibility": bars_passed_hard_eligibility_capsule,
                        "bars_blocked_hard_eligibility": bars_blocked_hard_eligibility_capsule,
                        "bars_passed_soft_eligibility": bars_passed_soft_eligibility_capsule,
                        "bars_blocked_soft_eligibility": bars_blocked_soft_eligibility_capsule,
                        "transformer_forward_calls": transformer_forward_calls_capsule,
                        "transformer_input_recorded": transformer_input_recorded_capsule,
                        "seq_feature_count": seq_feature_count_capsule,
                        "snap_feature_count": snap_feature_count_capsule,
                        "entry_routing": entry_routing_capsule,
                        "exception_gap": exception_gap_capsule,
                        "entry_v10_enabled": entry_v10_enabled_capsule,
                        "control_flow_last_3": control_flow_last_3_capsule,
                        "first_iter_ts": first_iter_ts_safe,
                        "last_iter_ts": last_iter_ts_safe,
                        "replay_mode": "PREBUILT" if os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1" else "UNKNOWN",
                        "policy_id": policy_id if 'policy_id' in locals() else None,
                        "bundle_sha256": bundle_sha256 if bundle_sha256 else None,
                        "bundle_dir_resolved": bundle_dir_resolved,
                        "run_identity_keys": list(run_identity_data.keys()) if run_identity_data else None,
                        "telemetry_status": telemetry_status,
                        "argv": sys.argv.copy() if hasattr(sys, 'argv') else None,
                        "cwd": str(Path.cwd()),
                        "sys_executable": sys.executable if hasattr(sys, 'executable') else None,
                        "hint": "failure happened before normal flush; see PREBUILT_FAIL_CAPSULE if present",
                        "timestamp": dt_now_iso(),
                    }
                    
                    # Write capsule atomically
                    capsule_written = False
                    if chunk_output_dir:
                        capsule_path = chunk_output_dir / "CHUNK_FAIL_CAPSULE.json"
                        if atomic_write_json(capsule_path, fail_capsule):
                            capsule_written = True
                            log.error(f"[CHUNK {chunk_idx}] Wrote CHUNK_FAIL_CAPSULE.json in footer_error handler")
                    
                    if not capsule_written:
                        # Fallback to /tmp
                        import tempfile
                        fallback_path = Path(tempfile.gettempdir()) / f"chunk_{chunk_idx}_FAIL_CAPSULE_{run_id}.json"
                        if atomic_write_json(fallback_path, fail_capsule):
                            log.error(f"[CHUNK {chunk_idx}] Wrote CHUNK_FAIL_CAPSULE.json to fallback: {fallback_path}")
                except Exception as capsule_error:
                    log.error(f"[CHUNK {chunk_idx}] Failed to write CHUNK_FAIL_CAPSULE.json in footer_error handler: {capsule_error}", exc_info=True)
                
                # FIX: Write minimal stub-footer on error (so aggregator can detect and hard-fail)
                # Include prebuilt_gate_dump in stub_footer for diagnosis
                try:
                    # Try to get gate dump from runner if available
                    stub_gate_dump = None
                    try:
                        if runner:
                            if hasattr(runner, "entry_manager") and hasattr(runner.entry_manager, "_prebuilt_gate_dump"):
                                stub_gate_dump = runner.entry_manager._prebuilt_gate_dump
                            else:
                                prebuilt_features_df_exists = hasattr(runner, "prebuilt_features_df")
                                prebuilt_features_df_is_none = not prebuilt_features_df_exists or runner.prebuilt_features_df is None
                                stub_gate_dump = {
                                    "is_replay": getattr(runner, "replay_mode", False),
                                    "prebuilt_enabled": os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
                                    "prebuilt_used_flag": getattr(runner, "prebuilt_used", False),
                                    "prebuilt_features_df_exists": prebuilt_features_df_exists,
                                    "prebuilt_features_df_is_none": prebuilt_features_df_is_none,
                                }
                        else:
                            # Runner not created yet - minimal dump
                            stub_gate_dump = {
                                "is_replay": None,
                                "prebuilt_enabled": os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
                                "prebuilt_used_flag": None,
                                "prebuilt_features_df_exists": False,
                                "prebuilt_features_df_is_none": True,
                            }
                    except Exception:
                        pass
                    
                    # Try to get prebuilt_used and other values from runner if available
                    stub_prebuilt_used = None
                    stub_prebuilt_bypass_count = None
                    stub_lookup_attempts = None
                    stub_lookup_hits = None
                    stub_eval_calls_total = None
                    stub_eval_calls_prebuilt_gate_true = None
                    stub_eval_calls_prebuilt_gate_false = None
                    try:
                        if runner:
                            stub_prebuilt_used = getattr(runner, "prebuilt_used", None)
                            stub_prebuilt_bypass_count = getattr(runner, "prebuilt_bypass_count", None)
                            stub_lookup_attempts = getattr(runner, "lookup_attempts", None)
                            stub_lookup_hits = getattr(runner, "lookup_hits", None)
                            # Get evaluate_entry() telemetry from entry_manager
                            if hasattr(runner, "entry_manager"):
                                stub_eval_calls_total = getattr(runner.entry_manager, "eval_calls_total", None)
                                stub_eval_calls_prebuilt_gate_true = getattr(runner.entry_manager, "eval_calls_prebuilt_gate_true", None)
                                stub_eval_calls_prebuilt_gate_false = getattr(runner.entry_manager, "eval_calls_prebuilt_gate_false", None)
                    except Exception:
                        pass
                    
                    # Try to get entry stage telemetry from runner if available
                    # Use same hard-check logic as normal footer (direct attribute access if exists)
                    stub_bars_seen = None
                    stub_bars_skipped_warmup = None
                    stub_bars_skipped_pregate = None
                    stub_bars_reaching_entry_stage = None
                    stub_pregate_enabled = None
                    try:
                        if runner:
                            # Check if attributes exist (same as normal footer)
                            required_attrs = ["bars_seen", "bars_skipped_warmup", "bars_skipped_pregate", "bars_reaching_entry_stage"]
                            missing_attrs = [attr for attr in required_attrs if not hasattr(runner, attr)]
                            if not missing_attrs:
                                # Attributes exist, use direct access (same as normal footer)
                                stub_bars_seen = runner.bars_seen
                                stub_bars_skipped_warmup = runner.bars_skipped_warmup
                                stub_bars_skipped_pregate = runner.bars_skipped_pregate
                                stub_bars_reaching_entry_stage = runner.bars_reaching_entry_stage
                                stub_pregate_enabled = getattr(runner, "pregate_enabled", False)
                            else:
                                # Attributes missing - this should not happen if bar loop ran
                                # But don't raise in stub_footer (already in error state)
                                stub_bars_seen = None
                                stub_bars_skipped_warmup = None
                                stub_bars_skipped_pregate = None
                                stub_bars_reaching_entry_stage = None
                                stub_pregate_enabled = None
                    except Exception:
                        pass
                    
                    # Fallback: use chunk_df length as proxy for bars_seen if runner telemetry is None
                    # But first check if runner has attributes (hard-check like in normal footer)
                    if runner:
                        required_attrs = ["bars_seen", "bars_skipped_warmup", "bars_skipped_pregate", "bars_reaching_entry_stage"]
                        missing_attrs = [attr for attr in required_attrs if not hasattr(runner, attr)]
                        if not missing_attrs:
                            # Attributes exist, use them directly
                            stub_bars_seen = runner.bars_seen
                            stub_bars_skipped_warmup = runner.bars_skipped_warmup
                            stub_bars_skipped_pregate = runner.bars_skipped_pregate
                            stub_bars_reaching_entry_stage = runner.bars_reaching_entry_stage
                        else:
                            # Attributes missing, use fallback
                            if chunk_df is not None:
                                stub_bars_seen = len(chunk_df)
                            else:
                                stub_bars_seen = bars_processed
                            stub_bars_skipped_warmup = 0
                            stub_bars_skipped_pregate = 0
                            stub_bars_reaching_entry_stage = 0
                    else:
                        # No runner, use fallback
                        if chunk_df is not None:
                            stub_bars_seen = len(chunk_df)
                        else:
                            stub_bars_seen = bars_processed
                        stub_bars_skipped_warmup = 0
                        stub_bars_skipped_pregate = 0
                        stub_bars_reaching_entry_stage = 0
                    
                    # Add warmup info to stub_footer if available
                    stub_warmup_required = None
                    stub_warmup_seen = None
                    try:
                        if runner:
                            if hasattr(runner, "warmup_floor") and runner.warmup_floor is not None:
                                stub_warmup_required = True
                            elif hasattr(runner, "replay_eval_start_ts") and runner.replay_eval_start_ts is not None:
                                stub_warmup_required = True
                            stub_warmup_seen = getattr(runner, "n_bars_skipped_due_to_htf_warmup", None)
                    except Exception:
                        pass
                    
                    # Convert all numeric values to JSON-serializable types
                    stub_footer = {
                        "run_id": run_id,
                        "chunk_id": str(chunk_idx),
                        "status": "footer_error",
                        "error": f"Failed to write chunk_footer.json: {str(footer_error)[:500]}",
                        "total_bars": convert_to_json_serializable(total_bars),
                        "bars_processed": convert_to_json_serializable(bars_processed),
                        "prebuilt_used": stub_prebuilt_used,
                        "prebuilt_bypass_count": convert_to_json_serializable(stub_prebuilt_bypass_count),
                        "lookup_attempts": convert_to_json_serializable(stub_lookup_attempts),
                        "lookup_hits": convert_to_json_serializable(stub_lookup_hits),
                        "lookup_misses": convert_to_json_serializable(getattr(runner, "lookup_misses", 0) if runner else 0),
                        "eval_calls_total": convert_to_json_serializable(stub_eval_calls_total),
                        "eval_calls_prebuilt_gate_true": convert_to_json_serializable(stub_eval_calls_prebuilt_gate_true),
                        "eval_calls_prebuilt_gate_false": convert_to_json_serializable(stub_eval_calls_prebuilt_gate_false),
                        "bars_seen": convert_to_json_serializable(stub_bars_seen),
                        "bars_skipped_warmup": convert_to_json_serializable(stub_bars_skipped_warmup),
                        "bars_skipped_pregate": convert_to_json_serializable(stub_bars_skipped_pregate),
                        "bars_reaching_entry_stage": convert_to_json_serializable(stub_bars_reaching_entry_stage),
                        "pregate_enabled": stub_pregate_enabled,
                        "warmup_required": stub_warmup_required,
                        "warmup_seen": convert_to_json_serializable(stub_warmup_seen),
                        "prebuilt_gate_dump": stub_gate_dump,
                        "lookup_miss_details": getattr(runner, "lookup_miss_details", [])[:3] if runner else [],
                        "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
                        "timestamp": dt_now_iso(),
                    }
                    chunk_footer_path = chunk_output_dir / "chunk_footer.json"
                    with open(chunk_footer_path, "w") as f:
                        json.dump(stub_footer, f, indent=2)
                    log.warning(f"[CHUNK {chunk_idx}] Wrote stub chunk_footer.json (footer_error)")
                except Exception as stub_error:
                    log.error(f"[CHUNK {chunk_idx}] Failed to write stub chunk_footer.json: {stub_error}", exc_info=True)
            
            # Collect artifact paths (even if failed, artifacts may exist from partial flush)
            chunk_artifacts = {
                "chunk_idx": chunk_idx,
                "status": status,
                "n_bars": bars_processed,
                "n_model_calls": n_model_calls,
                "n_trades_closed": n_trades_closed,
                # DEL 1: Add performance metrics to chunk_artifacts (for JSON export)
                "wall_clock_sec": wall_clock_sec,
                "total_bars": total_bars,
                "bars_per_sec": bars_processed / wall_clock_sec if wall_clock_sec > 0 else 0.0,
                "feature_time_mean_ms": feature_time_mean_ms,
                "feature_timeout_count": feature_timeout_count,
                "htf_align_warn_count": htf_align_warn_count,
                "htf_align_time_total_sec": htf_align_time_total_sec,
                "htf_align_call_count": htf_align_call_count,
                "htf_align_warning_time_sec": htf_align_warning_time_sec,
                "pregate_skips": pregate_skips,
                "pregate_passes": pregate_passes,
                "pregate_missing_inputs": pregate_missing_inputs,
                "artifacts": {
                    "raw_signals": chunk_output_dir / f"raw_signals_{run_id}.parquet",
                    "policy_decisions": chunk_output_dir / f"policy_decisions_{run_id}.parquet",
                    "attribution": chunk_output_dir / f"attribution_{run_id}.json",
                    "trade_outcomes": chunk_output_dir / f"trade_outcomes_{run_id}.parquet",
                    "metrics": chunk_output_dir / f"metrics_{run_id}.json",
                    "summary": chunk_output_dir / f"summary_{run_id}.md",
                    "chunk_footer": chunk_output_dir / "chunk_footer.json",
                },
            }
            
            # Raise if failed (but flush already happened in finally)
            if status == "failed":
                raise RuntimeError(f"CHUNK_{chunk_idx}_FAILED: {error}")
            
            return chunk_artifacts
    
    except Exception as outer_exception:
        # DEL 1: Catch ALL exceptions, including those before chunk_output_dir is created
        import traceback as tb_module
        import sys
        error_traceback = "".join(tb_module.format_exception(type(outer_exception), outer_exception, outer_exception.__traceback__))
        
        # CRITICAL: Write CHUNK_FAIL_CAPSULE.json BEFORE any other handling (atomic, always valid JSON)
        try:
            from gx1.utils.atomic_json import atomic_write_json
            
            # Ensure chunk_output_dir exists
            if chunk_output_dir is None:
                try:
                    chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
                    chunk_output_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    chunk_output_dir = None
            
            # Compute bar counters snapshot (safe defaults, never NameError)
            try:
                bars_processed_safe = bars_processed if 'bars_processed' in locals() else 0
            except Exception:
                bars_processed_safe = 0
            
            bar_counters = compute_bar_counters_snapshot(
                runner if 'runner' in locals() else None,
                bars_processed_safe,
                chunk_df if 'chunk_df' in locals() else None,
            )
            
            # Get run identity data if available
            run_identity_data = None
            try:
                if chunk_output_dir and chunk_output_dir.exists():
                    run_identity_path = chunk_output_dir / "run_header.json"
                    if run_identity_path.exists():
                        with open(run_identity_path, "r") as f:
                            run_identity_data = json.load(f)
            except Exception:
                pass
            
            # Get first/last iter timestamps (safe)
            first_iter_ts_safe = None
            last_iter_ts_safe = None
            try:
                if chunk_df is not None and len(chunk_df) > 0:
                    first_iter_ts_safe = str(chunk_df.index[0])
                    last_iter_ts_safe = str(chunk_df.index[-1])
                elif first_iter_ts:
                    first_iter_ts_safe = str(first_iter_ts)
                if last_iter_ts:
                    last_iter_ts_safe = str(last_iter_ts)
            except Exception:
                pass
            
            # Get telemetry status (best-effort, never fail)
            telemetry_status = {
                "telemetry_required": os_module.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1",
                "collector_initialized": False,
                "telemetry_write_attempted": False,
                "telemetry_files_written": [],
                "no_entry_evaluations": False,
                "no_entry_reason": None,
            }
            try:
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        telemetry_status["collector_initialized"] = True
                        # Check if write was attempted (check for files)
                        if chunk_output_dir:
                            entry_features_path = chunk_output_dir / "ENTRY_FEATURES_USED.json"
                            if entry_features_path.exists():
                                telemetry_status["telemetry_write_attempted"] = True
                                telemetry_status["telemetry_files_written"].append("ENTRY_FEATURES_USED.json")
                                
                                # Check for no_entry_evaluations
                                try:
                                    with open(entry_features_path, "r") as f:
                                        telemetry_data = json.load(f)
                                    if telemetry_data.get("no_entry_evaluations", False):
                                        telemetry_status["no_entry_evaluations"] = True
                                        telemetry_status["no_entry_reason"] = telemetry_data.get("no_entry_evaluations_reason")
                                except Exception:
                                    pass
            except Exception:
                pass
            
            # Get bundle_dir_resolved if available
            bundle_dir_resolved = None
            try:
                if runner and hasattr(runner, "bundle_dir_resolved"):
                    bundle_dir_resolved = str(runner.bundle_dir_resolved) if runner.bundle_dir_resolved else None
            except Exception:
                pass
            
            # Build capsule payload
            fail_capsule = {
                "chunk_idx": chunk_idx,
                "run_id": run_id,
                "exception_type": type(outer_exception).__name__,
                "exception_message": str(outer_exception),
                "traceback": error_traceback[:10000],  # Trim long tracebacks
                "bar_counters": bar_counters,
                "bars_seen": bar_counters.get("candles_iterated", 0),
                "bars_processed": bars_processed_safe,
                "bars_reaching_entry_stage": bar_counters.get("reached_entry_stage", 0),
                "first_iter_ts": first_iter_ts_safe,
                "last_iter_ts": last_iter_ts_safe,
                "replay_mode": "PREBUILT" if os_module.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1" else "UNKNOWN",
                "policy_id": policy_id if 'policy_id' in locals() else None,
                "bundle_sha256": bundle_sha256 if bundle_sha256 else None,
                "bundle_dir_resolved": bundle_dir_resolved,
                "run_identity_keys": list(run_identity_data.keys()) if run_identity_data else None,
                "telemetry_status": telemetry_status,
                "argv": sys.argv.copy() if hasattr(sys, 'argv') else None,
                "cwd": str(Path.cwd()),
                "sys_executable": sys.executable if hasattr(sys, 'executable') else None,
                "hint": "failure happened before normal flush; see PREBUILT_FAIL_CAPSULE if present",
                "timestamp": dt_now_iso(),
            }
            
            # Write capsule atomically
            capsule_written = False
            if chunk_output_dir:
                capsule_path = chunk_output_dir / "CHUNK_FAIL_CAPSULE.json"
                if atomic_write_json(capsule_path, fail_capsule):
                    capsule_written = True
                    log.error(f"[CHUNK {chunk_idx}] Wrote CHUNK_FAIL_CAPSULE.json")
            
            if not capsule_written:
                # Fallback to /tmp
                import tempfile
                fallback_path = Path(tempfile.gettempdir()) / f"chunk_{chunk_idx}_FAIL_CAPSULE_{run_id}.json"
                if atomic_write_json(fallback_path, fail_capsule):
                    log.error(f"[CHUNK {chunk_idx}] Wrote CHUNK_FAIL_CAPSULE.json to fallback: {fallback_path}")
        except Exception as capsule_error:
            log.error(f"[CHUNK {chunk_idx}] Failed to write CHUNK_FAIL_CAPSULE.json: {capsule_error}", exc_info=True)  # Give up
        
        # Try to create output dir and write logs (may fail if we can't create dir)
        try:
            if chunk_output_dir is None:
                chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
                chunk_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write WORKER_START.json with exception
            worker_start_info = {
                "chunk_id": chunk_idx,
                "prebuilt_parquet_path_raw": str(prebuilt_parquet_path) if prebuilt_parquet_path else None,
                "prebuilt_parquet_path_raw_type": type(prebuilt_parquet_path).__name__ if prebuilt_parquet_path else "None",
                "prebuilt_parquet_path_resolved": None,
                "exists": False,
                "size": 0,
                "exception_full": error_traceback,
            }
            
            worker_start_json_path = chunk_output_dir / "WORKER_START.json"
            with open(worker_start_json_path, "w") as f:
                import json
                json.dump(worker_start_info, f, indent=2)
            
            fatal_error_path = chunk_output_dir / "FATAL_ERROR.txt"
            with open(fatal_error_path, "w") as f:
                f.write(f"[CHUNK {chunk_idx}] FAILED (outer exception): {outer_exception}\n\n")
                f.write(f"Full traceback:\n{error_traceback}\n\n")
                f.write(f"prebuilt_parquet_path (arg): {repr(prebuilt_parquet_path) if prebuilt_parquet_path else 'NOT_SET'}\n")
                f.write(f"output_dir: {output_dir}\n")
                f.write(f"chunk_idx: {chunk_idx}\n")
        except Exception as log_error:
            # Can't even write logs - this is catastrophic
            # Try to write to a fallback location
            try:
                fallback_log = output_dir.parent / "_LOGS" / f"chunk_{chunk_idx}_FATAL.log"
                fallback_log.parent.mkdir(parents=True, exist_ok=True)
                with open(fallback_log, "w") as f:
                    f.write(f"[CHUNK {chunk_idx}] FAILED (outer exception): {outer_exception}\n")
                    f.write(f"Full traceback:\n{error_traceback}\n")
                    f.write(f"Failed to write to chunk_output_dir: {log_error}\n")
            except Exception:
                pass  # Give up - can't log anything
        
        # Re-raise to let multiprocessing handle it
        raise


def aggregate_entry_feature_telemetry(
    output_dir: Path,
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    run_id: str,
    policy_id: Optional[str] = None,
    bundle_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Aggregate entry feature telemetry from all chunks and write manifest + master files.
    
    Returns:
        Dict with paths to manifest and master files
    """
    import json
    from pathlib import Path
    
    log.info("[TELEMETRY] Aggregating entry feature telemetry from chunks...")
    
    # Scan chunks for telemetry files
    chunk_telemetry = []
    all_entry_features_used = []
    
    for chunk_start, chunk_end, chunk_idx in chunks:
        chunk_dir = output_dir / f"chunk_{chunk_idx}"
        if not chunk_dir.exists():
            log.warning(f"[TELEMETRY] Chunk {chunk_idx} directory not found: {chunk_dir}")
            continue
        
        # Find telemetry files
        entry_features_path = chunk_dir / "ENTRY_FEATURES_USED.json"
        mask_applied_path = chunk_dir / "FEATURE_MASK_APPLIED.json"
        telemetry_path = chunk_dir / "ENTRY_FEATURES_TELEMETRY.json"
        
        chunk_info = {
            "chunk_idx": chunk_idx,
            "chunk_dir": str(chunk_dir.relative_to(output_dir)),
            "entry_features_used": str(entry_features_path.relative_to(output_dir)) if entry_features_path.exists() else None,
            "feature_mask_applied": str(mask_applied_path.relative_to(output_dir)) if mask_applied_path.exists() else None,
            "entry_features_telemetry": str(telemetry_path.relative_to(output_dir)) if telemetry_path.exists() else None,
        }
        
        # Load ENTRY_FEATURES_USED.json if exists
        if entry_features_path.exists():
            try:
                with open(entry_features_path, "r") as f:
                    chunk_data = json.load(f)
                all_entry_features_used.append((chunk_idx, chunk_data))
                chunk_info["entry_features_used_loaded"] = True
            except Exception as e:
                log.warning(f"[TELEMETRY] Failed to load ENTRY_FEATURES_USED.json from chunk {chunk_idx}: {e}")
                chunk_info["entry_features_used_loaded"] = False
        else:
            chunk_info["entry_features_used_loaded"] = False
        
        chunk_telemetry.append(chunk_info)
    
    # Write manifest
    manifest = {
        "run_id": run_id,
        "policy_id": policy_id,
        "bundle_sha256": bundle_sha256,
        "chunks": chunk_telemetry,
        "n_chunks": len(chunk_telemetry),
        "n_chunks_with_telemetry": sum(1 for c in chunk_telemetry if c.get("entry_features_used_loaded", False)),
    }
    
    manifest_path = output_dir / "ENTRY_FEATURES_TELEMETRY_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        log.info(f"[TELEMETRY] Wrote manifest: {manifest_path}")
    
    if not all_entry_features_used:
        log.warning(
            f"[TELEMETRY] No ENTRY_FEATURES_USED.json files found in any chunks. "
            f"Telemetry may not have been collected or written. "
            f"Chunks scanned: {len(chunk_telemetry)}"
        )
        return {
            "manifest_path": str(manifest_path.relative_to(output_dir)),
            "master_json_path": None,
            "master_md_path": None,
        }
    
    # Aggregate master file if we have any telemetry
    master_data = None
    if all_entry_features_used:
        log.info(f"[TELEMETRY] Found telemetry in {len(all_entry_features_used)} chunks, aggregating...")
        # Assert identical architecture metadata across chunks
        first_chunk_idx, first_data = all_entry_features_used[0]
        first_seq_names = tuple(first_data.get("seq_features", {}).get("names", []))
        first_snap_names = tuple(first_data.get("snap_features", {}).get("names", []))
        first_n_xgb = first_data.get("xgb_flow", {}).get("n_xgb_channels_in_transformer_input", -1)
        first_xgb_channels = tuple(
            first_data.get("xgb_seq_channels", {}).get("names", []) +
            first_data.get("xgb_snap_channels", {}).get("names", [])
        )
        first_xgb_used_as = first_data.get("xgb_flow", {}).get("xgb_used_as", "unknown")
        
        # Verify consistency
        for chunk_idx, chunk_data in all_entry_features_used[1:]:
            chunk_seq_names = tuple(chunk_data.get("seq_features", {}).get("names", []))
            chunk_snap_names = tuple(chunk_data.get("snap_features", {}).get("names", []))
            chunk_n_xgb = chunk_data.get("xgb_flow", {}).get("n_xgb_channels_in_transformer_input", -1)
            chunk_xgb_channels = tuple(
                chunk_data.get("xgb_seq_channels", {}).get("names", []) +
                chunk_data.get("xgb_snap_channels", {}).get("names", [])
            )
            chunk_xgb_used_as = chunk_data.get("xgb_flow", {}).get("xgb_used_as", "unknown")
            
            if chunk_seq_names != first_seq_names:
                raise RuntimeError(
                    f"[TELEMETRY] FATAL: Chunk {chunk_idx} has different seq_feature_names than chunk {first_chunk_idx}. "
                    f"This indicates inconsistent architecture across chunks."
                )
            if chunk_snap_names != first_snap_names:
                raise RuntimeError(
                    f"[TELEMETRY] FATAL: Chunk {chunk_idx} has different snap_feature_names than chunk {first_chunk_idx}. "
                    f"This indicates inconsistent architecture across chunks."
                )
            if chunk_n_xgb != first_n_xgb:
                raise RuntimeError(
                    f"[TELEMETRY] FATAL: Chunk {chunk_idx} has n_xgb_channels_in_transformer_input={chunk_n_xgb}, "
                    f"but chunk {first_chunk_idx} has {first_n_xgb}. This indicates inconsistent XGB channel configuration."
                )
            if chunk_xgb_channels != first_xgb_channels:
                raise RuntimeError(
                    f"[TELEMETRY] FATAL: Chunk {chunk_idx} has different xgb_channel_names than chunk {first_chunk_idx}. "
                    f"This indicates inconsistent XGB channel configuration."
                )
            if chunk_xgb_used_as != first_xgb_used_as:
                raise RuntimeError(
                    f"[TELEMETRY] FATAL: Chunk {chunk_idx} has xgb_used_as={chunk_xgb_used_as}, "
                    f"but chunk {first_chunk_idx} has {first_xgb_used_as}. This indicates inconsistent XGB flow configuration."
                )
        
        # Aggregate counts (sum across chunks)
        total_gate_stats = {}
        total_xgb_stats = {}
        
        # Aggregate entry routing telemetry
        total_routing_recorded = 0
        aggregated_selected_model_counts = {}
        aggregated_reason_counts = {}
        
        for chunk_idx, chunk_data in all_entry_features_used:
            gate_stats = chunk_data.get("gate_stats", {})
            for gate_name, count in gate_stats.items():
                total_gate_stats[gate_name] = total_gate_stats.get(gate_name, 0) + count
            
            xgb_stats = chunk_data.get("xgb_stats", {})
            for stat_name, count in xgb_stats.items():
                total_xgb_stats[stat_name] = total_xgb_stats.get(stat_name, 0) + count
            
            # Aggregate entry routing
            entry_routing_aggregate = chunk_data.get("entry_routing_aggregate", {})
            total_routing_recorded += entry_routing_aggregate.get("total_recorded", 0)
            
            selected_model_counts = entry_routing_aggregate.get("selected_model_counts", {})
            for model_key, count in selected_model_counts.items():
                aggregated_selected_model_counts[model_key] = aggregated_selected_model_counts.get(model_key, 0) + count
            
            reason_counts = entry_routing_aggregate.get("reason_counts", {})
            for reason, count in reason_counts.items():
                aggregated_reason_counts[reason] = aggregated_reason_counts.get(reason, 0) + count
        
        # Sum XGB flow counts
        total_pre_predict = sum(
            chunk_data.get("xgb_flow", {}).get("xgb_pre_predict_count", 0)
            for _, chunk_data in all_entry_features_used
        )
        total_post_predict = sum(
            chunk_data.get("xgb_flow", {}).get("xgb_post_predict_count", 0)
            for _, chunk_data in all_entry_features_used
        )
        total_veto_applied = sum(
            chunk_data.get("xgb_flow", {}).get("veto_applied_count", 0)
            for _, chunk_data in all_entry_features_used
        )
        
        # Get toggle state from first chunk (should be identical)
        first_toggles = first_data.get("toggles", {})
        
        # Build master data
        master_data = {
            "run_id": run_id,
            "policy_id": policy_id,
            "bundle_sha256": bundle_sha256,
            "n_chunks": len(all_entry_features_used),
            "seq_features": first_data.get("seq_features", {}),
            "snap_features": first_data.get("snap_features", {}),
            "xgb_seq_channels": first_data.get("xgb_seq_channels", {}),
            "xgb_snap_channels": first_data.get("xgb_snap_channels", {}),
            "xgb_flow": {
                "xgb_used_as": first_xgb_used_as,
                "n_xgb_channels_in_transformer_input": first_n_xgb,
                "xgb_pre_predict_count": total_pre_predict,
                "xgb_post_predict_count": total_post_predict,
                "post_predict_called": total_post_predict > 0,
                "veto_applied_count": total_veto_applied,
            },
            "toggles": first_toggles,
            "gate_stats": total_gate_stats,
            "xgb_stats": total_xgb_stats,
            "entry_routing_aggregate": {
                "total_recorded": total_routing_recorded,
                "selected_model_counts": aggregated_selected_model_counts,
                "reason_counts": aggregated_reason_counts,
            },
        }
        
        # Aggregate pre_entry_funnel from chunk_footer files (runner-level, not in ENTRY_FEATURES_USED)
        pre_entry_funnel_aggregate = {
            "candles_iterated": 0,
            "warmup_skipped": 0,
            "pregate_checked_count": 0,
            "pregate_skipped": 0,
            "prebuilt_available_checked": 0,
            "prebuilt_missing_skipped": 0,
            "bars_before_evaluate_entry": 0,
            "evaluate_entry_called_count": 0,
            "bars_after_evaluate_entry": 0,
            "last_stop_reason_counts": {},
        }
        
        # Scan chunk directories for chunk_footer.json
        for chunk_idx, chunk_info in enumerate(chunk_telemetry):
            chunk_dir = output_dir / f"chunk_{chunk_idx}"
            chunk_footer_path = chunk_dir / "chunk_footer.json"
            if chunk_footer_path.exists():
                try:
                    with open(chunk_footer_path) as f:
                        chunk_footer_data = json.load(f)
                        funnel = chunk_footer_data.get("pre_entry_funnel", {})
                        if funnel:
                            pre_entry_funnel_aggregate["candles_iterated"] += funnel.get("candles_iterated", 0)
                            pre_entry_funnel_aggregate["warmup_skipped"] += funnel.get("warmup_skipped", 0)
                            pre_entry_funnel_aggregate["pregate_checked_count"] += funnel.get("pregate_checked_count", 0)
                            pre_entry_funnel_aggregate["pregate_skipped"] += funnel.get("pregate_skipped", 0)
                            pre_entry_funnel_aggregate["prebuilt_available_checked"] += funnel.get("prebuilt_available_checked", 0)
                            pre_entry_funnel_aggregate["prebuilt_missing_skipped"] += funnel.get("prebuilt_missing_skipped", 0)
                            pre_entry_funnel_aggregate["bars_before_evaluate_entry"] += funnel.get("bars_before_evaluate_entry", 0)
                            pre_entry_funnel_aggregate["evaluate_entry_called_count"] += funnel.get("evaluate_entry_called_count", 0)
                            pre_entry_funnel_aggregate["bars_after_evaluate_entry"] += funnel.get("bars_after_evaluate_entry", 0)
                            
                            # Aggregate last_stop_reason histogram
                            last_reason = funnel.get("last_stop_reason")
                            if last_reason:
                                pre_entry_funnel_aggregate["last_stop_reason_counts"][last_reason] = \
                                    pre_entry_funnel_aggregate["last_stop_reason_counts"].get(last_reason, 0) + 1
                except Exception as e:
                    log.warning(f"[TELEMETRY] Failed to read pre_entry_funnel from chunk {chunk_idx}: {e}")
        
        master_data["pre_entry_funnel_aggregate"] = pre_entry_funnel_aggregate
        
        # Write master JSON
        master_json_path = output_dir / "ENTRY_FEATURES_USED_MASTER.json"
        with open(master_json_path, "w") as f:
            json.dump(master_data, f, indent=2)
        log.info(f"[TELEMETRY] Wrote master JSON: {master_json_path}")
        
        # Write master MD
        master_md_path = output_dir / "ENTRY_FEATURES_USED_MASTER.md"
        md_lines = [
            "# Entry Features Used (Master Aggregation)",
            "",
            f"**Run ID:** {run_id}",
            f"**Policy ID:** {policy_id or 'N/A'}",
            f"**Bundle SHA256:** {bundle_sha256[:16] + '...' if bundle_sha256 else 'N/A'}",
            f"**Chunks Aggregated:** {len(all_entry_features_used)}",
            "",
            "## Architecture",
            "",
            f"**Sequence Features:** {master_data['seq_features'].get('count', 0)}",
            f"**Snapshot Features:** {master_data['snap_features'].get('count', 0)}",
            f"**XGB Channels in Transformer:** {first_n_xgb}",
            "",
            "## XGB Flow",
            "",
            f"**XGB Used As:** {first_xgb_used_as}",
            f"**Pre-Predict Calls:** {total_pre_predict}",
            f"**Post-Predict Calls:** {total_post_predict}",
            f"**Veto Applied Count:** {total_veto_applied}",
            "",
            "## Toggles",
            "",
            f"**Disable XGB Channels in Transformer:** {first_toggles.get('disable_xgb_channels_in_transformer_effective', False)}",
            f"**Disable XGB Post-Transformer:** {first_toggles.get('disable_xgb_post_transformer_effective', False)}",
            "",
            "## Entry Routing (Aggregated)",
            "",
            f"**Total Routing Recorded:** {total_routing_recorded}",
            "",
            "### Selected Model Counts",
            "",
        ]
        
        # Add selected model counts
        for model_key, count in sorted(aggregated_selected_model_counts.items()):
            md_lines.append(f"- **{model_key}:** {count}")
        
        md_lines.extend([
            "",
            "### Reason Counts",
            "",
        ])
        
        # Add reason counts
        for reason, count in sorted(aggregated_reason_counts.items(), key=lambda x: -x[1]):
            md_lines.append(f"- **{reason}:** {count}")
        
        md_lines.append("")
        
        with open(master_md_path, "w") as f:
            f.write("\n".join(md_lines))
        log.info(f"[TELEMETRY] Wrote master MD: {master_md_path}")
    
    return {
        "manifest_path": str(manifest_path.relative_to(output_dir)),
        "master_json_path": str((output_dir / "ENTRY_FEATURES_USED_MASTER.json").relative_to(output_dir)) if master_data else None,
        "master_md_path": str((output_dir / "ENTRY_FEATURES_USED_MASTER.md").relative_to(output_dir)) if master_data else None,
    }


def export_perf_json_from_footers(
    run_id: str,
    output_dir: Path,
    policy_path: Path,
    pregate_enabled: bool,
    pregate_enabled_source: str,  # "env" or "yaml"
    workers: int,
    actual_workers_started: int,
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    total_time: float,
    export_mode: str = "normal",  # "normal" or "watchdog_sigterm"
    export_partial: bool = False,  # True if export happened while workers still running
    dt_module_version: Optional[str] = None,  # CRITICAL: Version stamp (optional, will be fetched if None)
) -> Path:
    """
    Export perf JSON from chunk_footer.json files (robust, always works).
    
    This function reads chunk_footer.json files directly, so it works even if
    workers were stopped or pool.starmap() never returned.
    
    Side effects: Only writes to perf_json_path. No global state changes.
    
    Args:
        run_id: Unique run identifier
        output_dir: Directory containing chunk_* subdirectories
        policy_path: Path to policy YAML (for determining pregate_enabled if not provided)
        pregate_enabled: Whether PreGate is enabled
        workers: Number of workers (for validation)
        chunks: List of (start_ts, end_ts, chunk_idx) tuples
        total_time: Total wall clock time in seconds
    
    Returns:
        Path to written perf JSON file.
    
    Raises:
        Exception: If file write fails (caller should handle and write stub).
    """
    perf_json_path = output_dir / f"perf_{run_id}.json"
    
    log.info("[PERF] Exporting performance JSON from chunk footers...")
    
    # Collect chunk metrics from chunk_footer.json files
    chunks_metrics = []
    chunks_statuses = {}
    
    for chunk_start, chunk_end, chunk_idx in chunks:
        chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
        chunk_footer_path = chunk_output_dir / "chunk_footer.json"
        
        if chunk_footer_path.exists():
            try:
                with open(chunk_footer_path, "r") as f:
                    footer = json.load(f)
                
                status = footer.get("status", "unknown")
                chunks_statuses[chunk_idx] = status
                
                # FIX: Hard-fail if footer is stub/partial (footer_error status)
                if status == "footer_error":
                    error_msg = footer.get("error", "Unknown footer error")
                    raise RuntimeError(
                        f"Chunk {chunk_idx} has footer_error status: {error_msg}. "
                        f"Footer export failed - cannot aggregate metrics."
                    )
                
                # 5) Hard-fail if status=ok but required fields are null
                if status == "ok":
                    required_fields = ["total_bars", "n_model_calls", "feature_time_mean_ms", "t_feature_build_total_sec"]
                    for field in required_fields:
                        if footer.get(field) is None:
                            raise RuntimeError(
                                f"Chunk {chunk_idx} has status=ok but required field '{field}' is null. "
                                f"This indicates a bug in chunk footer generation."
                            )
                
                chunks_metrics.append({
                    "chunk_idx": chunk_idx,
                    "wall_clock_sec": footer.get("wall_clock_sec", footer.get("worker_time_sec", 0.0)),
                    "total_bars": footer.get("total_bars") if status == "ok" else None,
                    "bars_per_sec": footer.get("bars_per_sec"),
                    "n_model_calls": footer.get("n_model_calls") if status == "ok" else None,
                    "n_trades_closed": footer.get("n_trades_closed") if status == "ok" else None,
                    "feature_time_mean_ms": footer.get("feature_time_mean_ms"),
                    "feature_timeout_count": footer.get("feature_timeout_count") if status == "ok" else None,
                    "htf_align_warn_count": footer.get("htf_align_warn_count") if status == "ok" else None,
                    "htf_align_time_total_sec": footer.get("htf_align_time_total_sec") if status == "ok" else None,
                    "htf_align_call_count": footer.get("htf_align_call_count") if status == "ok" else None,
                    "htf_align_warning_time_sec": footer.get("htf_align_warning_time_sec") if status == "ok" else None,
                    "htf_align_fallback_count": footer.get("htf_align_fallback_count") if status == "ok" else None,
                    "htf_feature_compute_bars": footer.get("htf_feature_compute_bars") if status == "ok" else None,
                    # FIX: Export HTFAligner stats (from get_stats())
                    "htf_h1_calls": footer.get("htf_h1_calls") if status == "ok" else None,
                    "htf_h4_calls": footer.get("htf_h4_calls") if status == "ok" else None,
                    "htf_h1_warns": footer.get("htf_h1_warns") if status == "ok" else None,
                    "htf_h4_warns": footer.get("htf_h4_warns") if status == "ok" else None,
                    "htf_last_m5_ts": footer.get("htf_last_m5_ts") if status == "ok" else None,
                    "htf_last_j": footer.get("htf_last_j") if status == "ok" else None,
                    "pregate_skips": footer.get("pregate_skips") if status == "ok" else None,
                    "pregate_passes": footer.get("pregate_passes") if status == "ok" else None,
                    "pregate_missing_inputs": footer.get("pregate_missing_inputs") if status == "ok" else None,
                    "vol_regime_unknown_count": footer.get("vol_regime_unknown_count") if status == "ok" else None,
                    "status": status,
                    # Phase timing breakdown
                    "t_pregate_total_sec": footer.get("t_pregate_total_sec") if status == "ok" else None,
                    "t_feature_build_total_sec": footer.get("t_feature_build_total_sec"),
                    "t_model_total_sec": footer.get("t_model_total_sec"),
                    "t_policy_total_sec": footer.get("t_policy_total_sec"),
                    "t_io_total_sec": footer.get("t_io_total_sec"),
                })
            except json.JSONDecodeError as e:
                log.warning(f"[PERF] Failed to parse chunk_footer.json for chunk {chunk_idx} (truncated?): {e}")
                chunks_statuses[chunk_idx] = "parse_error"
            except Exception as e:
                log.warning(f"[PERF] Failed to read chunk_footer.json for chunk {chunk_idx}: {e}")
                chunks_statuses[chunk_idx] = "read_error"
        else:
            log.warning(f"[PERF] chunk_footer.json missing for chunk {chunk_idx}")
            chunks_statuses[chunk_idx] = "missing"
    
    # Compute aggregate stats (handle None values from failed chunks)
    total_model_calls = sum(m.get("n_model_calls") or 0 for m in chunks_metrics)
    total_bars = sum(m.get("total_bars") or 0 for m in chunks_metrics)
    total_vol_regime_unknown_count = sum(m.get("vol_regime_unknown_count") or 0 for m in chunks_metrics)
    chunks_completed = sum(1 for s in chunks_statuses.values() if s == "ok")
    
    # Collect environment info
    env_info = {
        "python_path": sys.executable,
        "python_version": sys.version.split()[0],
        "git_commit": get_git_commit_hash(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", "1"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS", "1"),
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS", "1"),
        "pregate_enabled": pregate_enabled,
        "quiet_mode": os.getenv("GX1_REPLAY_QUIET", "0") == "1",
    }
    
    # 1) Collect SSoT metadata from chunk footers
    prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1"
    prebuilt_used = False
    features_file_sha256 = None
    features_schema_version = None
    prebuilt_path_resolved = None
    
    # Get from first chunk footer that has prebuilt info
    for chunk_start, chunk_end, chunk_idx in chunks:
        chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
        chunk_footer_path = chunk_output_dir / "chunk_footer.json"
        if chunk_footer_path.exists():
            try:
                with open(chunk_footer_path, "r") as f:
                    footer = json.load(f)
                if footer.get("prebuilt_used"):
                    prebuilt_used = True
                    features_file_sha256 = footer.get("features_file_sha256")
                    prebuilt_path_resolved = footer.get("prebuilt_path")
                    # Try to get schema version from manifest
                    if prebuilt_path_resolved and Path(prebuilt_path_resolved).exists():
                        manifest_path = Path(prebuilt_path_resolved).with_suffix(".manifest.json")
                        if manifest_path.exists():
                            try:
                                with open(manifest_path, "r") as f_manifest:
                                    manifest = json.load(f_manifest)
                                    features_schema_version = manifest.get("schema_version")
                            except Exception:
                                pass
                    break
            except Exception:
                pass
    
    # 1) HARD INVARIANT: prebuilt_enabled==1 and prebuilt_used==0 => fail
    if prebuilt_enabled and not prebuilt_used:
        raise RuntimeError(
            "[PREBUILT_FAIL] Prebuilt enabled but not used. "
            "This indicates prebuilt features failed to load or validate. "
            "Check logs for [PREBUILT_FAIL] errors. "
            "Instructions: Run prebuilt_preflight.py to diagnose."
        )
    
    # Compute raw file metadata (fast: mtime+size, or SHA256 if available)
    raw_file_sha256 = None
    raw_file_mtime = None
    raw_file_size = None
    try:
        data_path_stat = data_path.stat()
        raw_file_mtime = data_path_stat.st_mtime
        raw_file_size = data_path_stat.st_size
        # Optionally compute SHA256 (slower, but more reliable)
        # For now, use mtime+size as it's fast
    except Exception:
        pass
    
    # Compute policy SHA256
    policy_sha256 = None
    try:
        if policy_path.exists():
            sha256_hash = hashlib.sha256()
            with open(policy_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            policy_sha256 = sha256_hash.hexdigest()
    except Exception:
        pass
    
    # 1) Collect bundle_sha256 from chunk footers (should be identical across all chunks)
    bundle_sha256_from_footers = None
    for chunk_start, chunk_end, chunk_idx in chunks:
        chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
        chunk_footer_path = chunk_output_dir / "chunk_footer.json"
        if chunk_footer_path.exists():
            try:
                with open(chunk_footer_path, "r") as f:
                    footer = json.load(f)
                footer_ssot = footer.get("ssot", {})
                footer_bundle_sha256 = footer_ssot.get("bundle_sha256")
                if footer_bundle_sha256:
                    if bundle_sha256_from_footers is None:
                        bundle_sha256_from_footers = footer_bundle_sha256
                    elif bundle_sha256_from_footers != footer_bundle_sha256:
                        log.warning(
                            f"[SSOT] bundle_sha256 mismatch in chunk {chunk_idx}: "
                            f"expected={bundle_sha256_from_footers[:16]}..., got={footer_bundle_sha256[:16]}..."
                        )
                    break  # Found one, use it
            except Exception:
                pass
    
    # HARD-FAIL if bundle_sha256 is missing
    if not bundle_sha256_from_footers:
        raise RuntimeError(
            "[SSOT_FAIL] bundle_sha256 is missing in all chunk footers. "
            "This should never happen - bundle_sha256 must be computed before workers start."
        )
    
    # 1) Build SSoT header
    ssot_header = {
        "prebuilt_enabled": prebuilt_enabled,
        "prebuilt_used": prebuilt_used,
        "prebuilt_path": prebuilt_path_resolved,
        "features_file_sha256": features_file_sha256,
        "features_schema_version": features_schema_version,
        "raw_file_mtime": raw_file_mtime,
        "raw_file_size": raw_file_size,
        "raw_file_sha256": raw_file_sha256,  # Optional, may be None
        "policy_path": str(policy_path.resolve()) if policy_path.exists() else str(policy_path),
        "policy_sha256": policy_sha256,
        "bundle_sha256": bundle_sha256_from_footers,  # CRITICAL: Must be present
        "git_commit": get_git_commit_hash(),
        "python_version": sys.version.split()[0],
    }
    
    perf_data = {
        "schema_version": "perf_v1",  # Schema version for future compatibility
        "run_id": run_id,
        "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
        "timestamp": _dt_now_iso(),
        "writer_pid": os.getpid(),  # PID of master process that wrote this file
        "export_seq": 1,  # Always 1 (hard-fail if someone tries to write twice)
        "pregate_enabled": pregate_enabled,
        "requested_workers": workers,
        "actual_workers_started": actual_workers_started,
        "workers": workers,  # Keep for backward compatibility
        "total_wall_clock_sec": total_time,
        "total_bars": total_bars,
        "total_model_calls": total_model_calls,
        "total_vol_regime_unknown_count": total_vol_regime_unknown_count,  # Aggregate vol_regime=UNKNOWN count
        "chunks_completed": chunks_completed,
        "chunks_total": len(chunks),
        "chunks_statuses": chunks_statuses,
        "chunks": chunks_metrics,
        "env_info": env_info,
        "export_mode": export_mode,  # "normal" or "watchdog_sigterm"
        "export_partial": export_partial,  # True if workers still running during export
        "ssot": ssot_header,  # 1) SSoT header with all metadata
    }
    
    # CRITICAL: Atomic write (tmp → rename) to prevent truncation if process dies mid-write
    perf_json_tmp = perf_json_path.with_suffix(".tmp")
    try:
        with open(perf_json_tmp, "w") as f:
            json.dump(perf_data, f, indent=2)
        # Atomic rename (POSIX guarantees this is atomic)
        perf_json_tmp.replace(perf_json_path)
    except Exception as e:
        # Clean up tmp file on error
        if perf_json_tmp.exists():
            perf_json_tmp.unlink()
        raise
    
    log.info(
        f"[PERF] Wrote perf_{run_id}.json "
        f"(chunks_completed={chunks_completed}/{len(chunks)}, "
        f"total_bars={total_bars}, total_model_calls={total_model_calls})"
    )
    
    # 4) Log run end
    run_status = "ok" if chunks_completed == len(chunks) and all(s == "ok" for s in chunks_statuses.values()) else "failed"
    log.info(
        "[RUN_END] run_id=%s status=%s wall_clock_sec=%.1f chunks_completed=%d/%d",
        run_id, run_status, total_time, chunks_completed, len(chunks)
    )
    
    return perf_json_path


def merge_artifacts(
    chunk_results: List[Dict[str, Any]],
    run_id: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Merge artifacts from all chunks into single files.
    
    SWEEP OPTIMIZATION: Respects GX1_OUTPUT_MODE (minimal vs full).
    In minimal mode, skips raw_signals, policy_decisions, trade_outcomes parquet files.
    
    Returns dict with paths to merged artifacts.
    """
    import os
    output_mode_env = os.getenv("GX1_OUTPUT_MODE", "").lower()
    output_mode = output_mode_env if output_mode_env in ("minimal", "full") else "full"
    
    log.info(f"[MERGE] Merging {len(chunk_results)} chunks (output_mode={output_mode})")
    
    merged_artifacts = {}
    
    # SWEEP OPTIMIZATION: In minimal mode, skip raw_signals, policy_decisions, trade_outcomes parquet
    # But ALWAYS merge metrics (required for decision-making)
    if output_mode != "minimal":
        # Merge parquet files (raw_signals, policy_decisions, trade_outcomes)
        for artifact_name in ["raw_signals", "policy_decisions", "trade_outcomes"]:
            dfs = []
            for chunk_result in chunk_results:
                artifact_path = chunk_result["artifacts"].get(artifact_name)
                if artifact_path and artifact_path.exists():
                    df = pd.read_parquet(artifact_path)
                    dfs.append(df)
            
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=False)
                merged_df = merged_df.sort_index() if isinstance(merged_df.index, pd.DatetimeIndex) else merged_df
                
                merged_path = output_dir / f"{artifact_name}_{run_id}_MERGED.parquet"
                merged_df.to_parquet(merged_path)
                merged_artifacts[artifact_name] = merged_path
                log.info(f"[MERGE] {artifact_name}: {len(merged_df)} rows")
    else:
        log.debug("[MERGE] Skipping raw_signals, policy_decisions, trade_outcomes parquet (output_mode=minimal)")
    
    # Merge JSON files (attribution, metrics)
    # SWEEP OPTIMIZATION: In minimal mode, skip attribution (not required for decision-making)
    if output_mode != "minimal":
        # Attribution: sum counts
        merged_attribution = {
            "total_decisions": 0,
            "by_decision": {},
            "by_reason": {},
        }
        
        for chunk_result in chunk_results:
            attr_path = chunk_result["artifacts"].get("attribution")
            if attr_path and attr_path.exists():
                with open(attr_path) as f:
                    attr = json.load(f)
                
                merged_attribution["total_decisions"] += attr.get("total_decisions", 0)
                
                # Sum by_decision counts (excluding _pct keys)
                for key, val in attr.get("by_decision", {}).items():
                    if not key.endswith("_pct"):
                        merged_attribution["by_decision"][key] = (
                            merged_attribution["by_decision"].get(key, 0) + val
                        )
                
                # Sum by_reason counts (excluding _pct keys)
                for key, val in attr.get("by_reason", {}).items():
                    if not key.endswith("_pct"):
                        merged_attribution["by_reason"][key] = (
                            merged_attribution["by_reason"].get(key, 0) + val
                        )
        
        # Recalculate percentages
        total = merged_attribution["total_decisions"]
        if total > 0:
            for key in list(merged_attribution["by_decision"].keys()):
                if not key.endswith("_pct"):
                    count = merged_attribution["by_decision"][key]
                    merged_attribution["by_decision"][f"{key}_pct"] = (count / total * 100)
            
            for key in list(merged_attribution["by_reason"].keys()):
                if not key.endswith("_pct"):
                    count = merged_attribution["by_reason"][key]
                    merged_attribution["by_reason"][f"{key}_pct"] = (count / total * 100)
        
        merged_attr_path = output_dir / f"attribution_{run_id}_MERGED.json"
        with open(merged_attr_path, "w") as f:
            json.dump(merged_attribution, f, indent=2)
        merged_artifacts["attribution"] = merged_attr_path
        log.info(f"[MERGE] attribution: {total} total decisions")
    else:
        log.debug("[MERGE] Skipping attribution (output_mode=minimal)")
    
    # Merge metrics: aggregate trade-level metrics
    merged_metrics = {
        "n_trades": 0,
        "total_pnl_bps": 0.0,
        "calibration_stats": {},
    }
    
    # SWEEP OPTIMIZATION: In minimal mode, skip trade_outcomes parquet (but still compute metrics from JSON)
    all_trade_outcomes = []
    if output_mode != "minimal":
        for chunk_result in chunk_results:
            outcomes_path = chunk_result["artifacts"].get("trade_outcomes")
            if outcomes_path and outcomes_path.exists():
                df = pd.read_parquet(outcomes_path)
                all_trade_outcomes.append(df)
        
        metrics_path = chunk_result["artifacts"].get("metrics")
        if metrics_path and metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            merged_metrics["n_trades"] += metrics.get("n_trades", 0)
            merged_metrics["total_pnl_bps"] += metrics.get("total_pnl_bps", 0.0)
            
            # Merge calibration stats
            calib_stats = metrics.get("calibration_stats", {})
            for key, val in calib_stats.items():
                merged_metrics["calibration_stats"][key] = (
                    merged_metrics["calibration_stats"].get(key, 0) + val
                )
    
    # Compute aggregate metrics from all trades
    if all_trade_outcomes:
        all_trades_df = pd.concat(all_trade_outcomes, ignore_index=True)
        merged_metrics["mean_pnl_bps"] = float(all_trades_df["pnl_bps"].mean())
        merged_metrics["median_pnl_bps"] = float(all_trades_df["pnl_bps"].median())
        merged_metrics["std_pnl_bps"] = float(all_trades_df["pnl_bps"].std())
        
        sorted_pnl = all_trades_df["pnl_bps"].sort_values()
        if len(sorted_pnl) > 0:
            merged_metrics["p1_loss"] = float(sorted_pnl.quantile(0.01))
            merged_metrics["p5_loss"] = float(sorted_pnl.quantile(0.05))
            merged_metrics["max_dd"] = float(sorted_pnl.min())
    
    merged_metrics_path = output_dir / f"metrics_{run_id}_MERGED.json"
    with open(merged_metrics_path, "w") as f:
        json.dump(merged_metrics, f, indent=2)
    merged_artifacts["metrics"] = merged_metrics_path
    log.info(f"[MERGE] metrics: {merged_metrics['n_trades']} trades")
    
    # Generate merged summary
    git_commit = get_git_commit_hash()
    summary_lines = [
        f"# Replay Evaluation Summary (MERGED)",
        "",
        f"**Run ID:** {run_id}",
        f"**Git Commit:** {git_commit or 'N/A'}",
        f"**Chunks:** {len(chunk_results)}",
        "",
        "## Metrics",
        "",
        f"- **Trades:** {merged_metrics['n_trades']}",
        f"- **Total PnL (bps):** {merged_metrics.get('total_pnl_bps', 0.0):.2f}",
        f"- **Mean PnL (bps):** {merged_metrics.get('mean_pnl_bps', 0.0):.2f}",
        f"- **Median PnL (bps):** {merged_metrics.get('median_pnl_bps', 0.0):.2f}",
        f"- **Max DD (bps):** {merged_metrics.get('max_dd', 0.0):.2f}",
        f"- **P1 Loss (bps):** {merged_metrics.get('p1_loss', 0.0):.2f}",
        f"- **P5 Loss (bps):** {merged_metrics.get('p5_loss', 0.0):.2f}",
    ]
    
    merged_summary_path = output_dir / f"summary_{run_id}_MERGED.md"
    with open(merged_summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    merged_artifacts["summary"] = merged_summary_path
    log.info(f"[MERGE] summary: {merged_summary_path}")
    
    return merged_artifacts


def main():
    # LEGACY_GUARD: Check for legacy modes before proceeding
    try:
        from gx1.runtime.legacy_guard import assert_no_legacy_mode_enabled
        assert_no_legacy_mode_enabled()
    except ImportError:
        # Guard not available - log warning but continue
        log.warning("[LEGACY_GUARD] legacy_guard not available - skipping check")
    except RuntimeError as e:
        # Legacy mode detected - hard fail
        log.error(f"[LEGACY_GUARD] {e}")
        raise
    
    parser = argparse.ArgumentParser(description="Parallel replay evaluation for GATED_FUSION")
    parser.add_argument("--policy", type=Path, required=True, help="Policy YAML path")
    parser.add_argument("--data", type=Path, required=True, help="Input data (parquet)")
    parser.add_argument("--workers", type=int, default=7, help="Number of parallel workers")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: reports/replay_eval/{run_id})")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    parser.add_argument("--slice-head", type=int, default=None, help="Use only first N bars (deterministic)")
    parser.add_argument("--days", type=int, default=None, help="Use only first N days (deterministic, M5=288 bars/day)")
    parser.add_argument("--start-ts", type=str, default=None, help="Slice start timestamp (inclusive, ISO8601)")
    parser.add_argument("--end-ts", type=str, default=None, help="Slice end timestamp (inclusive, ISO8601)")
    parser.add_argument("--abort-after-first-chunk", type=int, default=0, help="Stop after first chunk completes (default: 0)")
    parser.add_argument("--abort-after-n-bars-per-chunk", type=int, default=None, help="Stop each chunk after N bars processed (for fast verification, default: None)")
    parser.add_argument("--dry-run-prebuilt-check", type=int, default=0, help="Only load prebuilt + run checks, then exit (default: 0)")
    parser.add_argument("--prebuilt-parquet", type=Path, default=None, help="Absolute path to prebuilt features parquet file (required if prebuilt enabled)")
    parser.add_argument("--bundle-dir", type=Path, default=None, help="Override bundle_dir from policy (absolute path, highest priority)")
    parser.add_argument("--selftest-only", action="store_true", help="Run only worker self-test (smoke_open) and exit (for debugging)")
    
    args = parser.parse_args()
    
    # DEL 7: Set GX1_SELFTEST_ONLY env var if --selftest-only is set
    if args.selftest_only:
        os.environ["GX1_SELFTEST_ONLY"] = "1"
        log.info("[MASTER] --selftest-only: Workers will exit after smoke test")
    
    # FASE 0.1: Forby parallell replay - maks én aktiv replay_eval_gated_parallel per maskin
    # Exception: Allow parallel replay if GX1_ALLOW_PARALLEL_REPLAY=1 (for multi-year parallel execution)
    allow_parallel = os.getenv("GX1_ALLOW_PARALLEL_REPLAY", "0") == "1"
    if not allow_parallel and psutil is not None:
        current_pid = os.getpid()
        script_name = "replay_eval_gated_parallel.py"
        running_replays = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['pid'] == current_pid:
                    continue
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any(script_name in str(arg) for arg in cmdline):
                    running_replays.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if running_replays:
            raise RuntimeError(
                f"[PREBUILT_FAIL] Another replay_eval_gated_parallel.py is already running (PIDs: {running_replays}). "
                f"Only one replay can run at a time. Kill existing processes before starting a new one. "
                f"To allow parallel execution (multi-year), set GX1_ALLOW_PARALLEL_REPLAY=1."
            )
    elif allow_parallel:
        log.info("[FASE_0] Parallel replay allowed (GX1_ALLOW_PARALLEL_REPLAY=1)")
    else:
        log.warning("[FASE_0] psutil not available - skipping parallel replay detection")
    
    # DEL 1: Verify GX1_GATED_FUSION_ENABLED=1
    gated_fusion_enabled = os.getenv("GX1_GATED_FUSION_ENABLED", "0") == "1"
    if not gated_fusion_enabled:
        raise RuntimeError(
            "BASELINE_DISABLED: GX1_GATED_FUSION_ENABLED is not '1'. "
            "Set GX1_GATED_FUSION_ENABLED=1 to run replay eval."
        )
    
    # FASE 0.3: Global kill-switch - sett GX1_FEATURE_BUILD_DISABLED=1 når prebuilt enabled
    prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
    if prebuilt_enabled:
        os.environ["GX1_FEATURE_BUILD_DISABLED"] = "1"
        log.info("[FASE_0] GX1_FEATURE_BUILD_DISABLED=1 set (prebuilt enabled)")
    
    # Generate run_id (use local import to avoid scoping issues)
    from gx1.utils.dt_module import strftime_now as _dt_strftime_now_runid
    run_id = args.run_id or _dt_strftime_now_runid("%Y%m%d_%H%M%S")
    
    # FASE 0.2: Hard reset - output-dir må ikke eksistere
    if args.output_dir is None:
        # DEL 4A: Use GX1_DATA env vars for default paths
        default_reports_root = Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports"))
        args.output_dir = default_reports_root / "replay_eval" / run_id
    
    # Resolve all paths to absolute
    args.output_dir = args.output_dir.resolve()
    args.policy = args.policy.resolve()
    args.data = args.data.resolve()
    if args.prebuilt_parquet:
        args.prebuilt_parquet = args.prebuilt_parquet.resolve()
    
    # LEGACY_GUARD: Check policy file and output-dir after path resolution
    # Note: sys is imported at top level, so it's available here
    try:
        from gx1.runtime.legacy_guard import check_policy_for_legacy, assert_no_legacy_mode_enabled
        check_policy_for_legacy(args.policy)
        # Check output-dir and bundle-dir with resolved paths
        bundle_dir_resolved = args.bundle_dir.resolve() if args.bundle_dir else None
        # Use sys.argv from top-level import
        import sys  # Re-import to ensure it's in local scope (Python scoping quirk)
        assert_no_legacy_mode_enabled(
            argv=sys.argv,
            bundle_dir_resolved=bundle_dir_resolved,
            output_dir_resolved=args.output_dir,
        )
    except ImportError:
        log.warning("[LEGACY_GUARD] legacy_guard not available - skipping check")
    except RuntimeError as e:
        log.error(f"[LEGACY_GUARD] {e}")
        raise
    
    # DEL 1: Master-side "EARLY MASTER LOG" (ALLTID) - før pool opprettes
    
    # CRITICAL: Validate dt_module version in master (fail-fast)
    # Import explicitly to avoid scoping issues (even though imported at top level)
    from gx1.utils.dt_module import (
        get_dt_module_version as _get_dt_module_version,
        validate_dt_module_version as _validate_dt_module_version,
        now_iso as _dt_now_iso,
        strftime_now as _dt_strftime_now,
    )
    _validate_dt_module_version()
    dt_module_version = _get_dt_module_version()
    
    master_early_payload = {
        "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
        "timestamp": _dt_now_iso(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "sys_executable": sys.executable,  # sys is imported at top level
        "argv": sys.argv.copy(),  # sys is imported at top level
        "policy_path": str(args.policy),
        "policy_path_resolved": str(args.policy.resolve()),
        "data_path": str(args.data),
        "data_path_resolved": str(args.data.resolve()),
        "prebuilt_parquet_path": str(args.prebuilt_parquet) if args.prebuilt_parquet else None,
        "prebuilt_parquet_path_resolved": str(args.prebuilt_parquet.resolve()) if args.prebuilt_parquet else None,
        "output_dir": str(args.output_dir),
        "output_dir_resolved": str(args.output_dir.resolve()),
        "workers": args.workers,
        "start_method": mp.get_start_method(),
        "env_keys": {k: v for k, v in os.environ.items() if k.startswith("GX1_")},
        "run_id": run_id,
    }
    
    # Try to write to output_dir first, fallback to /tmp
    master_early_written = False
    try:
        # Try to create output_dir (will fail if not writable)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        master_early_path = args.output_dir / "master_early.json"
        with open(master_early_path, "w") as f:
            json.dump(master_early_payload, f, indent=2)
        master_early_written = True
        log.info(f"[MASTER_EARLY] Wrote master_early.json to {master_early_path}")
    except Exception as e:
        # Fallback to /tmp
        fallback_path = Path(f"/tmp/gx1_master_early_{run_id}.json")
        try:
            with open(fallback_path, "w") as f:
                json.dump(master_early_payload, f, indent=2)
            log.warning(f"[MASTER_EARLY] Failed to write to output_dir, wrote to fallback: {fallback_path}")
            log.warning(f"[MASTER_EARLY] Output dir error: {e}")
        except Exception as fallback_error:
            log.error(f"[MASTER_EARLY] Failed to write even to fallback: {fallback_error}")
            # Hard-fail if we can't write early log
            raise RuntimeError(
                f"[MASTER_EARLY] Cannot write master_early.json to output_dir or /tmp. "
                f"Output dir error: {e}, Fallback error: {fallback_error}"
            ) from fallback_error
    
    # FASE 0.2: Hard-fail hvis output-dir eksisterer (ingen gjenbruk) - etter at vi har prøvd å opprette det
    if args.output_dir.exists():
        # Check for any existing chunks or perf JSONs
        has_chunks = any(args.output_dir.glob("chunk_*"))
        has_perf = any(args.output_dir.glob("perf_*.json"))
        if has_chunks or has_perf:
            raise RuntimeError(
                f"[PREBUILT_FAIL] Output directory {args.output_dir} already exists and contains artifacts. "
                f"This violates FASE 0.2: Hard reset - no reuse of output-dir. "
                f"Instructions: Remove or rename the existing directory before starting a new run."
            )
    
    # 4) Standardize log path
    log_path = Path(f"/tmp/gx1_replay_{run_id}.log")
    
    # 4) Log run start
    prebuilt_enabled_log = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
    # Use CLI --prebuilt-parquet if provided, otherwise fallback to env var
    if args.prebuilt_parquet:
        prebuilt_path_log = str(args.prebuilt_parquet.resolve())
        # Also set env var for workers
        os.environ["GX1_REPLAY_PREBUILT_FEATURES_PATH"] = prebuilt_path_log
    else:
        prebuilt_path_log = os.getenv("GX1_REPLAY_PREBUILT_FEATURES_PATH", "N/A")
    
    # Reset basic_v1_call_count at run start (for prebuilt verification)
    # FASE 1: DO NOT import basic_v1 in PREBUILT mode - it violates FASE 1 separation
    # In PREBUILT mode, basic_v1_call_count should be 0 by default (never called)
    if prebuilt_enabled_log:
        # FASE 1: Skip reset in PREBUILT mode to avoid importing basic_v1
        # The counter will be checked in workers (where it's safe to import)
        log.info("[RUN_START] PREBUILT mode: basic_v1_call_count will be verified in workers (must be 0)")
    else:
        # BASELINE mode: Reset counter (safe to import here, but lazy to avoid top-level import)
        try:
            from gx1.features.basic_v1 import reset_basic_v1_call_count
            reset_basic_v1_call_count()
            log.info("[RUN_START] Reset basic_v1_call_count to 0 (baseline mode)")
        except ImportError:
            log.warning("[RUN_START] basic_v1 not available - skipping reset (may be PREBUILT mode)")
    
    log.info(
        "[RUN_START] run_id=%s workers=%d prebuilt_enabled=%d prebuilt_path=%s log=%s",
        run_id, args.workers, 1 if prebuilt_enabled_log else 0, prebuilt_path_log, log_path
    )
    
    # Set abort-after-N-bars-per-chunk if specified
    if args.abort_after_n_bars_per_chunk is not None:
        os.environ["GX1_ABORT_AFTER_N_BARS_PER_CHUNK"] = str(args.abort_after_n_bars_per_chunk)
        log.info(f"[RUN_START] Fast abort mode: will stop each chunk after {args.abort_after_n_bars_per_chunk} bars")
    
    log.info(f"[PARALLEL] Starting parallel replay evaluation")
    log.info(f"[PARALLEL] Policy: {args.policy}")
    log.info(f"[PARALLEL] Data: {args.data}")
    log.info(f"[PARALLEL] Workers: {args.workers}")
    log.info(f"[PARALLEL] Run ID: {run_id}")
    log.info(f"[PARALLEL] Output: {args.output_dir}")
    log.info(f"[PARALLEL] Log: {log_path}")
    log.info(f"[PARALLEL] Multiprocessing start method: {mp.get_start_method()}")
    log.info(f"[PARALLEL] Thread limits: OMP={os.getenv('OMP_NUM_THREADS')}, MKL={os.getenv('MKL_NUM_THREADS')}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL: Install SIGTERM handler EARLY (before mp.Pool is created)
    global MASTER_STOP_REQUESTED, POOL_REF
    MASTER_STOP_REQUESTED = False
    POOL_REF = None
    signal.signal(signal.SIGTERM, _master_sigterm_handler)
    log.info("[MASTER] SIGTERM handler installed")
    
    # CRITICAL: Start watchdog thread to guarantee perf JSON export even if main thread is blocked
    import threading
    watchdog_done = threading.Event()
    PERF_EXPORT_LOCK = threading.Lock()  # Initialize lock for perf export (must be before watchdog thread)
    
    def watchdog_thread():
        """Watchdog thread that polls MASTER_STOP_REQUESTED and exports perf JSON when set."""
        global MASTER_STOP_REQUESTED, PERF_EXPORTED
        poll_interval = 0.1  # Poll every 100ms
        while not watchdog_done.is_set():
            if MASTER_STOP_REQUESTED:
                log.warning("[PERF] Watchdog detected STOP_REQUESTED, exporting perf JSON...")
                
                # CRITICAL: Use lock to prevent double export (race condition)
                with PERF_EXPORT_LOCK:
                    if PERF_EXPORTED:
                        log.warning("[PERF] Perf already exported by another thread, exiting...")
                        os._exit(0)
                    
                    try:
                        # Determine pregate_enabled
                        env_pregate_enabled = os.getenv("GX1_REPLAY_PREGATE_ENABLED")
                        if env_pregate_enabled is not None:
                            pregate_enabled = env_pregate_enabled.lower() in ("1", "true")
                        else:
                            import yaml
                            with open(args.policy, "r") as f:
                                policy = yaml.safe_load(f)
                            replay_config = policy.get("replay_config", {})
                            pregate_cfg = replay_config.get("replay_pregate", {})
                            pregate_enabled = pregate_cfg.get("enabled", False) if isinstance(pregate_cfg, dict) else False
                        
                        # Get chunks (we'll read from footers, so we can use dummy timestamps)
                        chunks_for_export = []
                        for i in range(args.workers):
                            footer_path = args.output_dir / f"chunk_{i}" / "chunk_footer.json"
                            if footer_path.exists():
                                try:
                                    with open(footer_path) as f:
                                        footer = json.load(f)
                                    start_ts = pd.Timestamp(footer.get("start_ts", "2025-01-01"))
                                    end_ts = pd.Timestamp(footer.get("end_ts", "2025-01-02"))
                                    chunks_for_export.append((start_ts, end_ts, i))
                                except Exception:
                                    # Fallback to dummy timestamps
                                    chunks_for_export.append((pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), i))
                            else:
                                # Fallback to dummy timestamps
                                chunks_for_export.append((pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), i))
                        
                        # Determine pregate_enabled_source
                        env_pregate_enabled = os.getenv("GX1_REPLAY_PREGATE_ENABLED")
                        if env_pregate_enabled is not None:
                            pregate_enabled_source = "env"
                        else:
                            pregate_enabled_source = "yaml"
                        
                        # Export perf JSON (watchdog mode - workers may still be running)
                        total_time_estimate = 60.0  # Conservative estimate
                        perf_json_path = export_perf_json_from_footers(
                            run_id=run_id,
                            output_dir=args.output_dir,
                            policy_path=args.policy,
                            pregate_enabled=pregate_enabled,
                            pregate_enabled_source=pregate_enabled_source,
                            workers=args.workers,
                            actual_workers_started=args.workers,  # Best guess in watchdog mode
                            chunks=chunks_for_export,
                            total_time=total_time_estimate,
                            export_mode="watchdog_sigterm",
                            export_partial=True,  # Workers may still be running
                        )
                        
                        # Verify file was written
                        if perf_json_path.exists():
                            PERF_EXPORTED = True
                            log.warning(f"[WATCHDOG] ✅ Wrote perf_{run_id}.json from footers -> {perf_json_path}")
                            
                            # CRITICAL: Flush all logs before hard exit
                            import sys
                            sys.stdout.flush()
                            sys.stderr.flush()
                            logging.shutdown()  # Ensure all log handlers are flushed
                            
                            os._exit(0)  # Hard exit after perf JSON is written
                        else:
                            raise RuntimeError(f"Perf JSON path does not exist after export: {perf_json_path}")
                            
                    except Exception as e:
                        # NON-FATAL: Log warning instead of error, but track for summary
                        perf_export_error_count = 1
                        perf_export_last_error = str(e)
                        
                        # Log warning (tydelig, én gang)
                        log.warning(
                            f"[WATCHDOG] ⚠️  Perf JSON export failed (non-fatal): {e}\n"
                            f"    This will not stop the replay. Perf data may be incomplete."
                        )
                        
                        # Write to perf_export_warnings.log (append)
                        warnings_log_path = args.output_dir / "perf_export_warnings.log"
                        try:
                            import traceback
                            from gx1.utils.dt_module import now_iso as dt_now_iso
                            with open(warnings_log_path, "a") as f:
                                f.write(f"[{dt_now_iso()}] PERF EXPORT ERROR (WATCHDOG)\n")
                                f.write(f"Run ID: {run_id}\n")
                                f.write(f"Error: {e}\n")
                                f.write(f"Traceback:\n")
                                f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                                f.write("\n" + "="*60 + "\n\n")
                        except Exception as log_error:
                            log.warning(f"[WATCHDOG] Failed to write to perf_export_warnings.log: {log_error}")
                        
                    # CRITICAL: Write stub file with error info before exiting
                    import traceback
                    from gx1.utils.dt_module import get_dt_module_version, now_iso as dt_now_iso
                    error_stub = {
                        "schema_version": "perf_v1",  # Same schema as normal perf JSON
                        "run_id": run_id,
                        "dt_module_version": get_dt_module_version(),  # CRITICAL: Version stamp
                        "timestamp": dt_now_iso(),
                        "writer_pid": os.getpid(),
                        "export_seq": 1,
                        "status": "export_failed",
                        "export_error": str(e),
                        "export_traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                        "chunks_total": args.workers,
                        "chunks_statuses": {},
                        "note": "Perf export failed - this is a stub file written by watchdog thread",
                        # Include env_info for consistency with normal perf JSON
                        "env_info": {
                            "python_path": sys.executable,
                            "python_version": sys.version.split()[0],
                            "git_commit": get_git_commit_hash(),
                            "omp_num_threads": os.getenv("OMP_NUM_THREADS", "1"),
                            "mkl_num_threads": os.getenv("MKL_NUM_THREADS", "1"),
                            "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS", "1"),
                            "pregate_enabled": pregate_enabled,
                            "quiet_mode": False,  # FASE 5: Quiet mode removed
                        },
                    }
                    
                    # Try to read chunk statuses even if export failed
                    try:
                        for i in range(args.workers):
                            footer_path = args.output_dir / f"chunk_{i}" / "chunk_footer.json"
                            if footer_path.exists():
                                try:
                                    with open(footer_path) as f:
                                        footer = json.load(f)
                                    error_stub["chunks_statuses"][str(i)] = footer.get("status", "unknown")
                                except Exception:
                                    error_stub["chunks_statuses"][str(i)] = "read_error"
                            else:
                                error_stub["chunks_statuses"][str(i)] = "missing"
                    except Exception:
                        pass  # Non-fatal
                    
                    # Add env_info for consistency with normal perf JSON
                    error_stub["env_info"] = {
                        "python_path": sys.executable,
                        "python_version": sys.version.split()[0],
                        "git_commit": get_git_commit_hash(),
                        "omp_num_threads": os.getenv("OMP_NUM_THREADS", "1"),
                        "mkl_num_threads": os.getenv("MKL_NUM_THREADS", "1"),
                        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS", "1"),
                        "pregate_enabled": pregate_enabled,
                        "quiet_mode": os.getenv("GX1_REPLAY_QUIET", "0") == "1",
                    }
                    
                    # Write stub file
                    stub_path = args.output_dir / f"perf_{run_id}_FAILED_EXPORT.json"
                    try:
                        with open(stub_path, "w") as f:
                            json.dump(error_stub, f, indent=2)
                        log.error(f"[WATCHDOG] Export failed -> wrote stub {stub_path}")
                    except Exception as stub_error:
                        log.error(f"[WATCHDOG] Failed to write error stub: {stub_error}")
                        
                        # CRITICAL: Flush all logs before hard exit
                        import sys
                        sys.stdout.flush()
                        sys.stderr.flush()
                        logging.shutdown()  # Ensure all log handlers are flushed
                        
                        os._exit(2)  # Exit with error code 2 (export failed)
            
            watchdog_done.wait(timeout=poll_interval)
    
    watchdog = threading.Thread(target=watchdog_thread, daemon=True, name="perf-export-watchdog")
    watchdog.start()
    log.info("[MASTER] Watchdog thread started (guarantees perf JSON export on SIGTERM)")
    
    # CRITICAL: Compute bundle_sha256 BEFORE workers start (hard-fail if missing)
    log.info("[SSOT] Computing bundle_sha256 from policy + artifacts...")
    try:
        from gx1.utils.ssot_hash import compute_bundle_sha256, resolve_artifact_paths_from_policy
        
        # Resolve bundle_dir override (CLI > ENV > Policy)
        bundle_dir_override = None
        if args.bundle_dir:
            bundle_dir_override = Path(args.bundle_dir).resolve()
            log.info(f"[SSOT] Using CLI bundle_dir override: {bundle_dir_override}")
        elif os.getenv("GX1_BUNDLE_DIR"):
            bundle_dir_override = Path(os.getenv("GX1_BUNDLE_DIR")).resolve()
            log.info(f"[SSOT] Using ENV bundle_dir override: {bundle_dir_override}")
        
        # Resolve artifact paths
        resolved_artifact_paths = resolve_artifact_paths_from_policy(args.policy, bundle_dir_override)
        
        # Extract bundle_dir_source for logging
        bundle_dir_resolved = resolved_artifact_paths.get("bundle_dir")
        bundle_dir_source = resolved_artifact_paths.get("bundle_dir_source", "policy")
        bundle_dir_exists = bundle_dir_resolved.exists() if bundle_dir_resolved else False
        
        # Log bundle_dir resolution
        log.info(f"[SSOT] bundle_dir resolved from {bundle_dir_source}: {bundle_dir_resolved}")
        log.info(f"[SSOT] bundle_dir exists: {bundle_dir_exists}")
        
        # Update master_early_payload with bundle_dir info and rewrite if already written
        master_early_payload["bundle_dir_resolved"] = str(bundle_dir_resolved) if bundle_dir_resolved else None
        master_early_payload["bundle_dir_source"] = bundle_dir_source
        master_early_payload["bundle_dir_exists"] = bundle_dir_exists
        
        # Rewrite master_early.json with bundle_dir info if already written
        if master_early_written:
            try:
                master_early_path = args.output_dir / "master_early.json"
                with open(master_early_path, "w") as f:
                    json.dump(master_early_payload, f, indent=2)
                log.info(f"[MASTER_EARLY] Updated master_early.json with bundle_dir info")
            except Exception as e:
                log.warning(f"[MASTER_EARLY] Failed to update master_early.json with bundle_dir info: {e}")
        
        # Compute bundle_sha256
        bundle_sha256 = compute_bundle_sha256(args.policy, resolved_artifact_paths, bundle_dir_override)
        
        log.info(f"[SSOT] bundle_sha256={bundle_sha256[:16]}... (computed successfully)")
        log.info(f"[SSOT] bundle_sha256={bundle_sha256}")  # Full hash for grep
        
        # Log bundle metadata mismatch once in master (not in each worker)
        # FASE 1: Skip this in PREBUILT mode to avoid importing runtime_sniper_core
        prebuilt_enabled_check = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
        if not prebuilt_enabled_check:
            try:
                import yaml
                with open(args.policy, "r") as f:
                    policy = yaml.safe_load(f)
                entry_models = policy.get("entry_models", {})
                v10_ctx_cfg = entry_models.get("v10_ctx", {})
                bundle_dir = v10_ctx_cfg.get("bundle_dir")
                if bundle_dir:
                    bundle_dir_path = Path(bundle_dir)
                    if not bundle_dir_path.is_absolute():
                        bundle_dir_path = workspace_root / bundle_dir_path
                    bundle_metadata_path = bundle_dir_path / "bundle_metadata.json"
                    if bundle_metadata_path.exists():
                        with open(bundle_metadata_path, "r") as f:
                            bundle_meta = json.load(f)
                        expected_seq_dim = bundle_meta.get("seq_input_dim")
                        expected_snap_dim = bundle_meta.get("snap_input_dim")
                        
                        # Get runtime contract dims (from feature contract)
                        # FASE 1: Use feature_contract_v10_ctx instead of runtime_sniper_core (avoids importing basic_v1)
                        try:
                            from gx1.features.feature_contract_v10_ctx import get_contract_summary
                            contract_summary = get_contract_summary()
                            runtime_seq_dim = contract_summary["total_seq_features"]
                            runtime_snap_dim = contract_summary["total_snap_features"]
                            
                            if expected_seq_dim is not None and expected_seq_dim != runtime_seq_dim:
                                log.info(
                                    "[REPLAY] Using runtime contract dims seq=%d snap=%d (metadata seq=%d snap=%d; +3 XGB channels). "
                                    "Contract is source of truth.",
                                    runtime_seq_dim, runtime_snap_dim, expected_seq_dim, expected_snap_dim
                                )
                            elif expected_snap_dim is not None and expected_snap_dim != runtime_snap_dim:
                                log.info(
                                    "[REPLAY] Using runtime contract dims seq=%d snap=%d (metadata seq=%d snap=%d; +3 XGB channels). "
                                    "Contract is source of truth.",
                                    runtime_seq_dim, runtime_snap_dim, expected_seq_dim, expected_snap_dim
                                )
                        except ImportError:
                            pass  # Feature contract module not available
            except Exception:
                pass  # Non-fatal - just skip bundle metadata logging
    except Exception as ssot_error:
        log.error(f"[SSOT_FAIL] Failed to compute bundle_sha256: {ssot_error}", exc_info=True)
        raise RuntimeError(
            f"[SSOT_FAIL] bundle_sha256 computation failed. "
            f"This is a hard-fail - replay cannot proceed without bundle_sha256. "
            f"Error: {ssot_error}"
        ) from ssot_error
    
    # FASE 2: Tripwire - prebuilt_enabled=1 && prebuilt_used=0 skal ALDRI nå chunks
    # This check happens AFTER prebuilt validation in _run_replay_impl
    # We can't check prebuilt_used here because it's set in worker processes
    # But we can verify that prebuilt path exists and is valid BEFORE workers start
    if prebuilt_enabled:
        prebuilt_path_check = Path(prebuilt_path_log) if prebuilt_path_log != "N/A" else None
        if not prebuilt_path_check or not prebuilt_path_check.exists():
            raise RuntimeError(
                f"[PREBUILT_FAIL] FASE_2_TRIPWIRE: prebuilt_enabled=1 but prebuilt_path={prebuilt_path_log} does not exist. "
                f"CRASH: Prebuilt features file must exist before workers start."
            )
        # Verify manifest exists
        manifest_path = prebuilt_path_check.parent / f"{prebuilt_path_check.stem}.manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"[PREBUILT_FAIL] FASE_2_TRIPWIRE: prebuilt_enabled=1 but manifest={manifest_path} does not exist. "
                f"CRASH: Prebuilt features manifest must exist before workers start."
            )
        log.info(f"[FASE_2] Prebuilt path validated: {prebuilt_path_check} (exists)")
    
    # FASE 1: Hard guarantee - assert feature-building modules are NOT imported in PREBUILT mode
    # This check happens BEFORE workers start (in master process)
    # DEL 3: Deterministic, single-shot preflight selftest
    if prebuilt_enabled:
        log.info("[FASE_1] Running PREBUILT preflight selftest (deterministic, single-shot)...")
        
        import sys
        import traceback
        import inspect
        
        # Get policy_id from policy YAML
        policy_id = None
        try:
            import yaml
            with open(args.policy, "r") as f:
                policy = yaml.safe_load(f)
            policy_id = policy.get("policy_id") or policy.get("replay_config", {}).get("policy_id") or "unknown"
        except Exception:
            policy_id = "unknown"
        
        forbidden_modules = [
            "gx1.features.basic_v1",
            "gx1.execution.live_features",
            "gx1.features.runtime_v10_ctx",
            "gx1.features.runtime_sniper_core",
        ]
        imported_forbidden = [mod for mod in forbidden_modules if mod in sys.modules]
        
        if imported_forbidden:
            # Collect import violation details (first forbidden module + first importer)
            first_violation = None
            first_importer = None
            import_stack_full = []
            
            if imported_forbidden:
                first_forbidden = imported_forbidden[0]
                mod = sys.modules.get(first_forbidden)
                if mod:
                    mod_file = getattr(mod, "__file__", None)
                    
                    # Try to find first importer (best-effort via stack inspection)
                    try:
                        # Get call stack to find where import happened
                        stack = inspect.stack()
                        for frame_info in stack:
                            frame = frame_info.frame
                            # Check if this frame imported the forbidden module
                            if frame.f_globals.get(first_forbidden) is mod:
                                first_importer = {
                                    "file": frame_info.filename,
                                    "line": frame_info.lineno,
                                    "function": frame_info.function,
                                }
                                break
                    except Exception:
                        pass
                    
                    # Build full import stack (best-effort)
                    try:
                        import_stack_full = traceback.format_stack()
                    except Exception:
                        import_stack_full = ["Could not capture stack"]
                    
                    first_violation = {
                        "forbidden_module": first_forbidden,
                        "module_file": mod_file,
                        "first_importer": first_importer,
                        "all_forbidden_modules": imported_forbidden,
                    }
            
            # Write violation report to output directory (master-only, deterministic)
            violation_report = {
                "detected_at": datetime.now().isoformat(),
                "replay_mode": "PREBUILT",
                "policy_id": policy_id,
                "policy_path": str(args.policy.resolve()) if args.policy else None,
                "bundle_sha256": bundle_sha256[:16] + "..." if bundle_sha256 else None,
                "bundle_sha256_full": bundle_sha256 if bundle_sha256 else None,
                "sys.executable": sys.executable,
                "argv": sys.argv,
                "cwd": str(Path.cwd()),
                "first_violation": first_violation,
                "import_stack_full": import_stack_full,
                "message": "Forbidden feature-building modules imported in PREBUILT mode",
            }
            
            # Determine output directory (use args.output_dir if available, else default)
            # DEL 4A: Use GX1_DATA env vars for default paths
            default_reports_root = Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports"))
            output_dir_for_report = args.output_dir if args.output_dir else default_reports_root / "replay_eval"
            output_dir_for_report.mkdir(parents=True, exist_ok=True)
            violation_path = output_dir_for_report / "PREBUILT_IMPORT_VIOLATION.json"
            
            # Atomic write (master-only, no worker interference)
            with open(violation_path, "w") as f:
                json.dump(violation_report, f, indent=2)
            log.error(f"[PREBUILT_FAIL] Import violations written to: {violation_path}")
            
            raise RuntimeError(
                f"[PREBUILT_FAIL] FASE_1_SEPARATION: Forbidden feature-building modules imported in PREBUILT mode: {imported_forbidden}\n"
                f"This violates FASE 1: PREBUILT and BASELINE must be completely separate code paths.\n"
                f"CRASH: Feature-building code must not be imported in PREBUILT mode before workers start.\n"
                f"Violation details written to: {violation_path}\n"
                f"First forbidden module: {first_violation['forbidden_module'] if first_violation else 'unknown'}\n"
                f"First importer: {first_importer if first_importer else 'unknown'}"
            )
        
        log.info("[FASE_1] ✅ PREBUILT preflight selftest PASSED: No forbidden modules imported")
    
    # Split data into chunks (supports deterministic time-window slicing for screening)
    # Parse timestamps and ensure timezone-aware if data is timezone-aware
    parsed_start_ts = None
    parsed_end_ts = None
    if args.start_ts:
        parsed_start_ts = pd.to_datetime(args.start_ts)
    if args.end_ts:
        parsed_end_ts = pd.to_datetime(args.end_ts)
    
    # Check data timezone and adjust timestamps if needed
    # We'll do this inside split_data_into_chunks after loading data
    chunks = split_data_into_chunks(
        args.data,
        args.workers,
        slice_head=args.slice_head,
        days=args.days,
        start_ts=parsed_start_ts,
        end_ts=parsed_end_ts,
    )
    
    # Process chunks in parallel
    start_time = time.time()
    
    # Master timeout (default 6 hours for FULLYEAR, but configurable)
    master_timeout_sec = float(os.getenv("GX1_MASTER_WAIT_TIMEOUT_SEC", "21600"))  # 6 hours default
    
    # DEL 2: Master-side logging and validation before submit
    # DEL B: Get prebuilt path (absolute, explicit)
    prebuilt_parquet_path = None
    if args.prebuilt_parquet:
        prebuilt_parquet_path_obj = Path(args.prebuilt_parquet).resolve()
        if not prebuilt_parquet_path_obj.is_absolute():
            raise RuntimeError(
                f"[PREBUILT_FAIL] Prebuilt parquet path must be absolute: {prebuilt_parquet_path_obj}"
            )
        if not prebuilt_parquet_path_obj.exists():
            raise FileNotFoundError(
                f"[PREBUILT_FAIL] Prebuilt parquet file does not exist: {prebuilt_parquet_path_obj}"
            )
        file_size = prebuilt_parquet_path_obj.stat().st_size
        if file_size == 0:
            raise RuntimeError(
                f"[PREBUILT_FAIL] Prebuilt parquet file is empty (size=0): {prebuilt_parquet_path_obj}"
            )
        # DEL 3: Force string (not Path) for pickle safety
        prebuilt_parquet_path = str(prebuilt_parquet_path_obj)
        log.info(f"[MASTER] Prebuilt parquet path (explicit): {prebuilt_parquet_path}")
        log.info(f"[MASTER] Prebuilt parquet exists: True, size: {file_size:,} bytes")
    else:
        # Fallback to env (for backward compatibility)
        prebuilt_env_path = os.getenv("GX1_REPLAY_PREBUILT_FEATURES_PATH")
        if prebuilt_env_path:
            prebuilt_parquet_path_obj = Path(prebuilt_env_path).resolve()
            # DEL 3: Force string
            prebuilt_parquet_path = str(prebuilt_parquet_path_obj)
            log.info(f"[MASTER] Prebuilt parquet path (from env): {prebuilt_parquet_path}")
        else:
            # DEL 2: Hard-fail in master if prebuilt enabled but no path
            prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
            if prebuilt_enabled:
                raise RuntimeError(
                    "[PREBUILT_FAIL] Prebuilt enabled but no prebuilt_parquet_path provided (arg or env)"
                )
    
    # DEL 2: Master-side logging per chunk task (before submit)
    log.info(f"[MASTER] [PRE_SUBMIT] Validating prebuilt path for {len(chunks)} chunks...")
    for chunk_start, chunk_end, chunk_idx in chunks:
        log.info(f"[MASTER] [PRE_SUBMIT] Chunk {chunk_idx}:")
        log.info(f"[MASTER] [PRE_SUBMIT]   prebuilt_parquet_path = {repr(prebuilt_parquet_path)}")
        log.info(f"[MASTER] [PRE_SUBMIT]   type(prebuilt_parquet_path) = {type(prebuilt_parquet_path).__name__}")
        if prebuilt_parquet_path:
            prebuilt_path_obj = Path(prebuilt_parquet_path)
            log.info(f"[MASTER] [PRE_SUBMIT]   is_absolute = {prebuilt_path_obj.is_absolute()}")
            log.info(f"[MASTER] [PRE_SUBMIT]   exists = {prebuilt_path_obj.exists()}")
            if prebuilt_path_obj.exists():
                log.info(f"[MASTER] [PRE_SUBMIT]   size = {prebuilt_path_obj.stat().st_size:,} bytes")
            else:
                raise FileNotFoundError(
                    f"[PREBUILT_FAIL] [MASTER] Chunk {chunk_idx}: Prebuilt file does not exist: {prebuilt_parquet_path}"
                )
        else:
            # DEL 2: Hard-fail if None/empty when prebuilt enabled
            prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
            if prebuilt_enabled:
                raise RuntimeError(
                    f"[PREBUILT_FAIL] [MASTER] Chunk {chunk_idx}: prebuilt_parquet_path is None/empty but prebuilt enabled"
                )
    
    # DEL 1: Create master log file and write initial info
    master_log_path = args.output_dir.parent / "_LOGS" / "master.log"
    master_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_log_path, "w") as f:
        f.write(f"[MASTER] Starting parallel replay\n")
        f.write(f"run_id: {run_id}\n")
        f.write(f"workers: {args.workers}\n")
        f.write(f"chunks: {len(chunks)}\n")
        f.write(f"prebuilt_parquet_path: {prebuilt_parquet_path}\n")
        f.write(f"prebuilt_parquet_path_type: {type(prebuilt_parquet_path).__name__}\n")
        if prebuilt_parquet_path:
            prebuilt_path_obj = Path(prebuilt_parquet_path)
            f.write(f"prebuilt_exists: {prebuilt_path_obj.exists()}\n")
            if prebuilt_path_obj.exists():
                f.write(f"prebuilt_size: {prebuilt_path_obj.stat().st_size}\n")
        f.write(f"{'='*80}\n")
    
    chunk_tasks = [
        (
            chunk_idx,
            chunk_start,
            chunk_end,
            args.data,
            args.policy,
            run_id,
            args.output_dir,
            bundle_sha256,  # Pass bundle_sha256 to workers
            prebuilt_parquet_path,  # DEL B: Pass explicit prebuilt path to workers (as string)
            bundle_dir_override,  # Pass bundle_dir override to worker
        )
        for chunk_start, chunk_end, chunk_idx in chunks
    ]
    
    # DEL 2: Pre-create chunk dirs in MASTER før submit
    log.info(f"[MASTER] [PRE_CREATE] Pre-creating {len(chunks)} chunk directories...")
    for chunk_start, chunk_end, chunk_idx in chunks:
        chunk_dir = args.output_dir / f"chunk_{chunk_idx}"
        try:
            chunk_dir.mkdir(parents=True, exist_ok=False)
            # Write minimal chunk_master_created.json
            chunk_master_created = {
                "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
                "timestamp": _dt_now_iso(),
                "pid": os.getpid(),
                "chunk_id": chunk_idx,
                "chunk_start": str(chunk_start),
                "chunk_end": str(chunk_end),
            }
            chunk_master_created_path = chunk_dir / "chunk_master_created.json"
            with open(chunk_master_created_path, "w") as f:
                json.dump(chunk_master_created, f, indent=2)
            log.info(f"[MASTER] [PRE_CREATE] Created chunk_{chunk_idx} dir: {chunk_dir}")
        except FileExistsError:
            # Check if --force-clean-output was used (we don't have that flag, but check anyway)
            raise RuntimeError(
                f"[MASTER] [PRE_CREATE] Chunk directory already exists: {chunk_dir}. "
                f"This should not happen - output dir should be clean before starting."
            )
        except Exception as e:
            raise RuntimeError(
                f"[MASTER] [PRE_CREATE] Failed to create chunk directory {chunk_dir}: {e}"
            ) from e
    
    # DEL 3: Create master_failures.log file for error tracking
    master_failures_log_path = args.output_dir / "master_failures.log"
    master_failures_log_path.touch()  # Create empty file
    
    # Wrap worker execution in try/finally to ensure perf JSON is always written
    chunk_results = []
    total_time = 0.0
    merged_artifacts = {}
    pool = None
    
    # CRITICAL: Initialize merged_artifacts to empty dict (in case we exit early)
    merged_artifacts = {}
    
    # Track actual_workers_started in outer scope (accessible in finally)
    actual_workers_started = args.workers  # Default, will be set when pool is created
    
    try:
        # Use apply_async instead of starmap to allow polling/timeout
        log.info(f"[MASTER] Submitted {len(chunk_tasks)} chunks")
        
        # If workers=1, run directly without multiprocessing pool to avoid extra process
        if args.workers == 1:
            log.info("[MASTER] Workers=1, running directly without multiprocessing pool")
            actual_workers_started = 1
            # Initialize deadline for workers=1 case (needed for later check)
            deadline = time.time() + master_timeout_sec
            # Run directly in current process
            chunk_result = process_chunk(*chunk_tasks[0])
            chunk_results = [chunk_result]
            completed = set([0])
            pool = None
        else:
            pool = mp.Pool(processes=args.workers)
            actual_workers_started = args.workers  # Track actual workers started (for perf JSON)
            
            # DEL 4: Error callback for futures to capture full tracebacks
            def error_callback(chunk_idx, error):
                """Callback to log full traceback when a chunk fails."""
                import traceback
                error_msg = f"[CHUNK {chunk_idx}] Future failed with exception:\n"
                error_msg += f"Exception type: {type(error).__name__}\n"
                error_msg += f"Exception message: {str(error)}\n"
                error_msg += f"Full traceback:\n{''.join(traceback.format_exception(type(error), error, error.__traceback__))}\n"
                error_msg += f"Chunk task args: {chunk_tasks[chunk_idx] if chunk_idx < len(chunk_tasks) else 'N/A'}\n"
                error_msg += f"{'='*80}\n"
                
                # Write to master_failures.log
                try:
                    with open(master_failures_log_path, "a") as f:
                        f.write(error_msg)
                    log.error(f"[MASTER] Wrote failure to master_failures.log for chunk {chunk_idx}")
                except Exception as log_error:
                    log.error(f"[MASTER] Failed to write to master_failures.log: {log_error}")
            
            # Create async_results with error callbacks
            async_results = []
            for i, task in enumerate(chunk_tasks):
                # Wrap error_callback to capture chunk_idx
                def make_error_callback(idx):
                    def callback(error):
                        error_callback(idx, error)
                    return callback
                
                async_result = pool.apply_async(
                    process_chunk,
                    args=task,
                    error_callback=make_error_callback(i)
                )
                async_results.append(async_result)
            
            # Poll for completion with timeout
            poll_interval = 5.0  # Check every 5 seconds
            deadline = time.time() + master_timeout_sec
            main._deadline = deadline  # Store for finally block
            
            completed = set()
            while len(completed) < len(async_results):
                if MASTER_STOP_REQUESTED:
                    log.warning("[MASTER] Stop requested (SIGTERM), terminating pool...")
                    break
                
                if time.time() > deadline:
                    log.warning(f"[MASTER] Timeout waiting for workers (>{master_timeout_sec}s), terminating pool...")
                    break
                
                # Check which results are ready
                for i, result in enumerate(async_results):
                    if i in completed:
                        continue
                    
                    if result.ready():
                        # DEL 4: Always capture exception tracebacks from futures
                        if not result.successful():
                            try:
                                # Call result.get() to get the full exception and traceback
                                chunk_result = result.get(timeout=0.1)
                            except Exception as future_error:
                                import traceback
                                error_msg = f"[CHUNK {i}] Future failed with exception:\n"
                                error_msg += f"Exception type: {type(future_error).__name__}\n"
                                error_msg += f"Exception message: {str(future_error)}\n"
                                error_msg += f"Full traceback:\n{''.join(traceback.format_exception(type(future_error), future_error, future_error.__traceback__))}\n"
                                error_msg += f"Chunk task args: {chunk_tasks[i] if i < len(chunk_tasks) else 'N/A'}\n"
                                error_msg += f"{'='*80}\n"
                                
                                # Write to master_failures.log
                                try:
                                    with open(master_failures_log_path, "a") as f:
                                        f.write(error_msg)
                                    log.error(f"[MASTER] Wrote failure to master_failures.log for chunk {i}")
                                except Exception as log_error:
                                    log.error(f"[MASTER] Failed to write to master_failures.log: {log_error}")
                            
                            log.warning(f"[MASTER] Chunk {i} failed (result.successful()=False), will read from footer")
                            completed.add(i)  # Mark as done, will read from footer
                            continue
                        
                        try:
                            # Get result with short timeout (should be immediate if ready)
                            # This drains the result channel, but we don't rely on it for perf JSON
                            # CRITICAL: Always use timeout to avoid blocking
                            chunk_result = result.get(timeout=1.0)
                            chunk_results.append(chunk_result)
                            completed.add(i)
                            log.info(f"[MASTER] Progress: done={len(completed)}/{len(async_results)} pending={len(async_results) - len(completed)}")
                            
                            # E) Abort after first chunk
                            if args.abort_after_first_chunk == 1 and len(completed) >= 1:
                                log.info("[ABORT_MODE] First chunk completed - aborting remaining chunks")
                                MASTER_STOP_REQUESTED = True
                                break
                        except Exception as e:
                            log.warning(f"[MASTER] Error getting result for chunk {i} (will use footer): {e}")
                            completed.add(i)  # Mark as done to avoid infinite loop, will read from footer
                
                if len(completed) < len(async_results) and not (args.abort_after_first_chunk == 1 and len(completed) >= 1):
                    time.sleep(poll_interval)
                elif args.abort_after_first_chunk == 1 and len(completed) >= 1:
                    break  # Exit loop if abort mode and first chunk done
        
        total_time = time.time() - start_time
        
        # Deterministic pool cleanup: close() if all done, terminate() if stop/timeout
        if pool is not None:
            all_done = len(completed) == len(async_results) and not MASTER_STOP_REQUESTED
            if all_done:
                log.info(f"[PARALLEL] All chunks completed in {total_time:.1f}s")
                log.info("[MASTER] Closing pool (all chunks done)...")
                pool.close()  # Prevent new tasks, allow existing to finish
                # Note: pool.join() doesn't support timeout in Python 3.10, but workers should finish quickly after close()
                pool.join()  # Wait for workers to finish gracefully
            else:
                log.warning("[MASTER] Terminating pool due to stop/timeout...")
                pool.terminate()  # Force kill workers
                # After terminate(), workers should die quickly, but join() can still block
                # Use a short timeout - if it blocks, we'll continue anyway (perf JSON uses footers)
                # CRITICAL: Don't block perf JSON export on pool.join() - footers are already written
            import threading
            join_done = threading.Event()
            def join_pool():
                try:
                    # Note: pool.join() doesn't support timeout in Python 3.10
                    pool.join()  # Wait for workers to terminate
                    log.info("[MASTER] Pool joined successfully")
                except Exception as e:
                    log.warning(f"[MASTER] Pool join error (non-fatal): {e}")
                finally:
                    join_done.set()
            
            join_thread = threading.Thread(target=join_pool, daemon=True)
            join_thread.start()
            # Wait max 2s for join to complete (non-blocking for perf JSON export)
            # If it takes longer, we continue anyway - footers are already written
            if not join_done.wait(timeout=2):
                log.warning("[MASTER] Pool join still running, continuing to perf export (footers already written)...")
        
        # CRITICAL: Set pool to None BEFORE verification to ensure perf export can proceed
        pool = None  # Mark as closed (even if join() is still running in background thread)
        
        # If we were stopped, skip verification and go straight to perf export
        if MASTER_STOP_REQUESTED or time.time() > deadline:
            stop_reason = "SIGTERM" if MASTER_STOP_REQUESTED else "timeout"
            log.warning(f"[MASTER] STOP_REQUESTED=1 (reason={stop_reason}) -> exporting perf JSON from footers and exiting (merge skipped)")
            all_ok = False
            failed_chunks = []
        else:
            # Verify chunk completion via chunk_footer.json
            log.info("[PARALLEL] Verifying chunk completion...")
            failed_chunks = []
            all_ok = True
            
            for chunk_start, chunk_end, chunk_idx in chunks:
                chunk_output_dir = args.output_dir / f"chunk_{chunk_idx}"
                chunk_footer_path = chunk_output_dir / "chunk_footer.json"
                
                if not chunk_footer_path.exists():
                    failed_chunks.append((chunk_idx, "chunk_footer.json missing"))
                    log.error(f"[CHUNK {chunk_idx}] chunk_footer.json not found")
                    all_ok = False
                    continue
                
                try:
                    with open(chunk_footer_path) as f:
                        footer = json.load(f)
                    
                    chunk_status = footer.get("status", "unknown")
                    if chunk_status == "ok":
                        log.info(f"[CHUNK {chunk_idx}] Completed status=ok")
                    else:
                        error_msg = footer.get("error", "Unknown error")
                        failed_chunks.append((chunk_idx, error_msg))
                        log.error(f"[CHUNK {chunk_idx}] Failed: {error_msg}")
                        all_ok = False
                except Exception as e:
                    log.error(f"[CHUNK {chunk_idx}] Failed to read footer: {e}")
                    all_ok = False
        
        # Only merge if all chunks succeeded
        if all_ok and len(failed_chunks) == 0:
            log.info(f"[PARALLEL] All {len(chunks)} chunks completed successfully")
            log.info("[MERGE] Starting merge...")
            merged_artifacts = merge_artifacts(chunk_results, run_id, args.output_dir)
            log.info("[MERGE] Done.")
        else:
            log.warning(f"[PARALLEL] Skipping merge: {len(failed_chunks)} chunks failed/stopped")
            merged_artifacts = {}
        
    except (KeyboardInterrupt, SystemExit) as e:
        import traceback
        log.warning(f"[PARALLEL] Interrupted: {e}")
        # DEL 1: Write master.log with full traceback
        try:
            master_log_path = args.output_dir.parent / "_LOGS" / "master.log"
            master_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(master_log_path, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[MASTER] Interrupted: {e}\n")
                f.write(f"Full traceback:\n{traceback.format_exc()}\n")
                f.write(f"{'='*80}\n")
        except Exception:
            pass
        total_time = time.time() - start_time
        if pool:
            log.warning("[MASTER] Terminating pool due to interrupt...")
            pool.terminate()
            # Note: pool.join() doesn't support timeout in Python 3.10
            pool.join()  # Wait for workers to terminate
            pool = None
        # Continue to finally to write perf JSON
    except Exception as e:
        import traceback
        log.error(f"[PARALLEL] Error during execution: {e}", exc_info=True)
        # DEL 1: Write master.log with full traceback
        try:
            master_log_path = args.output_dir.parent / "_LOGS" / "master.log"
            master_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(master_log_path, "a") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"[MASTER] Error: {e}\n")
                f.write(f"Full traceback:\n{traceback.format_exc()}\n")
                f.write(f"{'='*80}\n")
        except Exception:
            pass
        total_time = time.time() - start_time
        if pool:
            log.warning("[MASTER] Terminating pool due to error...")
            pool.terminate()
            # Note: pool.join() doesn't support timeout in Python 3.10
            pool.join()  # Wait for workers to terminate
            pool = None
        # Continue to finally to write perf JSON
    finally:
        # Always write perf JSON (even on SIGTERM/KeyboardInterrupt/errors)
        log.info("[PERF] Exporting performance JSON (always, even on interrupt)...")
        
        perf_json_written = False
        perf_json_path = None
        perf_export_error_count = 0
        perf_export_last_error = None
        
        # CRITICAL: Use lock to prevent double export (race condition with watchdog)
        global PERF_EXPORTED
        with PERF_EXPORT_LOCK:
            if PERF_EXPORTED:
                log.info("[PERF] Perf already exported by watchdog thread, skipping...")
            else:
                try:
                    # Check if pregate is enabled (env override trumfer YAML)
                    env_pregate_enabled = os.getenv("GX1_REPLAY_PREGATE_ENABLED")
                    if env_pregate_enabled is not None:
                        pregate_enabled = env_pregate_enabled.lower() in ("1", "true")
                        log.info(f"[PERF] PreGate enabled from env: GX1_REPLAY_PREGATE_ENABLED={env_pregate_enabled} -> {pregate_enabled}")
                    else:
                        import yaml
                        with open(args.policy, "r") as f:
                            policy = yaml.safe_load(f)
                        replay_config = policy.get("replay_config", {})
                        pregate_cfg = replay_config.get("replay_pregate", {})
                        pregate_enabled = pregate_cfg.get("enabled", False) if isinstance(pregate_cfg, dict) else False
                        log.info(f"[PERF] PreGate enabled from YAML: {pregate_enabled}")
                    
                    # Determine pregate_enabled_source
                    env_pregate_check = os.getenv("GX1_REPLAY_PREGATE_ENABLED")
                    if env_pregate_check is not None:
                        pregate_enabled_source = "env"
                    else:
                        pregate_enabled_source = "yaml"
                    
                    # Use tracked actual_workers_started (set when pool was created)
                    # If not available (e.g., in finally after pool cleanup), use requested
                    try:
                        # actual_workers_started should be in scope from pool creation
                        if 'actual_workers_started' not in locals():
                            actual_workers_started = args.workers
                    except NameError:
                        # Fallback if variable not in scope (shouldn't happen, but be safe)
                        actual_workers_started = args.workers
                    
                    # Export perf JSON from footers (robust, always works)
                    perf_json_path = export_perf_json_from_footers(
                        run_id=run_id,
                        output_dir=args.output_dir,
                        policy_path=args.policy,
                        pregate_enabled=pregate_enabled,
                        pregate_enabled_source=pregate_enabled_source,
                        workers=args.workers,
                        actual_workers_started=actual_workers_started,
                        chunks=chunks,
                        total_time=total_time,
                        export_mode="normal",
                        export_partial=False,  # Normal completion
                    )
                    
                    # Verify file was written
                    if perf_json_path.exists():
                        PERF_EXPORTED = True
                        perf_json_written = True
                        log.info(f"[PERF] ✅ Verified: perf JSON written to {perf_json_path}")
                    else:
                        # NON-FATAL: Log warning instead of raising
                        perf_export_error_count += 1
                        perf_export_last_error = f"Perf JSON path does not exist after export: {perf_json_path}"
                        log.warning(f"[PERF] ⚠️  {perf_export_last_error} (non-fatal)")
                        
                        # Write to warnings log
                        warnings_log_path = args.output_dir / "perf_export_warnings.log"
                        try:
                            from gx1.utils.dt_module import now_iso as dt_now_iso
                            with open(warnings_log_path, "a") as f:
                                f.write(f"[{dt_now_iso()}] PERF EXPORT ERROR #{perf_export_error_count}\n")
                                f.write(f"Run ID: {run_id}\n")
                                f.write(f"Error: {perf_export_last_error}\n")
                                f.write("\n" + "="*60 + "\n\n")
                        except Exception:
                            pass  # Non-fatal
                        
                except Exception as perf_error:
                    # NON-FATAL: Log warning instead of error, but track for summary
                    perf_export_error_count += 1
                    perf_export_last_error = str(perf_error)
                    
                    # Log warning (tydelig, én gang)
                    log.warning(
                        f"[PERF] ⚠️  Perf JSON export failed (non-fatal): {perf_error}\n"
                        f"    This will not stop the replay. Perf data may be incomplete.\n"
                        f"    Error count: {perf_export_error_count}"
                    )
                    
                    # Write to perf_export_warnings.log (append)
                    warnings_log_path = args.output_dir / "perf_export_warnings.log"
                    try:
                        import traceback
                        from gx1.utils.dt_module import now_iso as dt_now_iso
                        with open(warnings_log_path, "a") as f:
                            f.write(f"[{dt_now_iso()}] PERF EXPORT ERROR #{perf_export_error_count}\n")
                            f.write(f"Run ID: {run_id}\n")
                            f.write(f"Error: {perf_error}\n")
                            f.write(f"Traceback:\n")
                            f.write("".join(traceback.format_exception(type(perf_error), perf_error, perf_error.__traceback__)))
                            f.write("\n" + "="*60 + "\n\n")
                    except Exception as log_error:
                        log.warning(f"[PERF] Failed to write to perf_export_warnings.log: {log_error}")
                    
                    # CRITICAL: Write stub file with error info (for debugging)
                    import traceback
                    from gx1.utils.dt_module import get_dt_module_version, now_iso as dt_now_iso
                    error_stub = {
                        "schema_version": "perf_v1",  # Same schema as normal perf JSON
                        "run_id": run_id,
                        "dt_module_version": get_dt_module_version(),  # CRITICAL: Version stamp
                        "timestamp": dt_now_iso(),
                        "writer_pid": os.getpid(),
                        "export_seq": 1,
                        "status": "export_failed",
                        "export_error": str(perf_error),
                        "export_traceback": "".join(traceback.format_exception(type(perf_error), perf_error, perf_error.__traceback__)),
                        "chunks_total": len(chunks),
                        "chunks_statuses": {},
                        "note": "Perf export failed - this is a stub file written by finally block (non-fatal)",
                    }
                    
                    # Try to read chunk statuses even if export failed
                    try:
                        for chunk_start, chunk_end, chunk_idx in chunks:
                            footer_path = args.output_dir / f"chunk_{chunk_idx}" / "chunk_footer.json"
                            if footer_path.exists():
                                try:
                                    with open(footer_path) as f:
                                        footer = json.load(f)
                                    error_stub["chunks_statuses"][str(chunk_idx)] = footer.get("status", "unknown")
                                except Exception:
                                    error_stub["chunks_statuses"][str(chunk_idx)] = "read_error"
                            else:
                                error_stub["chunks_statuses"][str(chunk_idx)] = "missing"
                    except Exception:
                        pass  # Non-fatal
                    
                    # Add env_info for consistency with normal perf JSON
                    error_stub["env_info"] = {
                        "python_path": sys.executable,
                        "python_version": sys.version.split()[0],
                        "git_commit": get_git_commit_hash(),
                        "omp_num_threads": os.getenv("OMP_NUM_THREADS", "1"),
                        "mkl_num_threads": os.getenv("MKL_NUM_THREADS", "1"),
                        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS", "1"),
                        "pregate_enabled": pregate_enabled,
                        "quiet_mode": os.getenv("GX1_REPLAY_QUIET", "0") == "1",
                    }
                    
                    # Write stub file
                    stub_path = args.output_dir / f"perf_{run_id}_FAILED_EXPORT.json"
                    try:
                        with open(stub_path, "w") as f:
                            json.dump(error_stub, f, indent=2)
                        log.warning(f"[PERF] Wrote error stub to {stub_path} (for debugging)")
                    except Exception as stub_error:
                        log.warning(f"[PERF] Failed to write error stub: {stub_error}")
        
        # Log final summary (include perf export status)
        perf_status = "✅" if perf_json_written else f"⚠️  ({perf_export_error_count} error(s))"
        log.info(f"[PARALLEL] Completed (total_time={total_time:.1f}s, perf_json={perf_status})")
        if merged_artifacts:
            log.info(f"[PARALLEL] Merged artifacts: {list(merged_artifacts.keys())}")
        
        # Aggregate entry feature telemetry from chunks
        telemetry_aggregation_result = None
        try:
            # Get policy_id and bundle_sha256 for manifest
            policy_id = None
            bundle_sha256_for_telemetry = None
            try:
                import yaml
                with open(args.policy, "r") as f:
                    policy = yaml.safe_load(f)
                policy_id = policy.get("policy_id") or args.policy.stem
            except Exception:
                policy_id = args.policy.stem
            
            # Get bundle_sha256 from SSOT (if available)
            if 'bundle_sha256' in locals():
                bundle_sha256_for_telemetry = bundle_sha256
            
            telemetry_aggregation_result = aggregate_entry_feature_telemetry(
                output_dir=args.output_dir,
                chunks=chunks,
                run_id=run_id,
                policy_id=policy_id,
                bundle_sha256=bundle_sha256_for_telemetry,
            )
            log.info("[TELEMETRY] ✅ Entry feature telemetry aggregation completed")
        except Exception as telemetry_error:
            log.warning(f"[TELEMETRY] Failed to aggregate entry feature telemetry: {telemetry_error}", exc_info=True)
            telemetry_aggregation_result = None
        
        # Write orchestrator summary with perf export status (always write, not just on errors)
        summary_path = args.output_dir / "orchestrator_summary.json"
        try:
            # Aggregate lookup counters from chunk footers
            aggregated_lookup_attempts = 0
            aggregated_lookup_hits = 0
            aggregated_lookup_misses = 0
            lookup_phase_aggregated = None
            try:
                for chunk_idx, (chunk_start, chunk_end, _) in enumerate(chunks):
                    chunk_dir = args.output_dir / f"chunk_{chunk_idx}"
                    chunk_footer_path = chunk_dir / "chunk_footer.json"
                    if chunk_footer_path.exists():
                        with open(chunk_footer_path, "r") as f:
                            chunk_footer = json.load(f)
                        aggregated_lookup_attempts += chunk_footer.get("prebuilt_lookup_attempts", 0)
                        aggregated_lookup_hits += chunk_footer.get("prebuilt_lookup_hits", 0)
                        aggregated_lookup_misses += chunk_footer.get("prebuilt_lookup_misses", 0)
                        if lookup_phase_aggregated is None:
                            lookup_phase_aggregated = chunk_footer.get("prebuilt_lookup_phase", "unknown")
            except Exception as lookup_agg_error:
                log.warning(f"[PARALLEL] Failed to aggregate lookup counters: {lookup_agg_error}")
            
            summary_data = {
                "run_id": run_id,
                "total_time_sec": total_time,
                "perf_json_written": perf_json_written,
                "perf_export_error_count": perf_export_error_count,
                "perf_export_last_error": perf_export_last_error,
                "merged_artifacts": list(merged_artifacts.keys()) if merged_artifacts else [],
                "entry_feature_telemetry": telemetry_aggregation_result,
                "prebuilt_lookup": {
                    "lookup_phase": lookup_phase_aggregated,
                    "aggregated_attempts": aggregated_lookup_attempts,
                    "aggregated_hits": aggregated_lookup_hits,
                    "aggregated_misses": aggregated_lookup_misses,
                },
                "entry_universe": {
                    "aggregated_bars_passed_hard_eligibility": aggregated_bars_passed_hard_eligibility,
                    "aggregated_bars_blocked_hard_eligibility": aggregated_bars_blocked_hard_eligibility,
                    "aggregated_bars_passed_soft_eligibility": aggregated_bars_passed_soft_eligibility,
                    "aggregated_bars_blocked_soft_eligibility": aggregated_bars_blocked_soft_eligibility,
                },
            }
            with open(summary_path, "w") as f:
                json.dump(summary_data, f, indent=2)
            if perf_export_error_count > 0:
                log.warning(f"[PARALLEL] Orchestrator summary written (with perf export errors): {summary_path}")
            else:
                log.info(f"[PARALLEL] Orchestrator summary written: {summary_path}")
        except Exception as summary_error:
            log.warning(f"[PARALLEL] Failed to write orchestrator summary: {summary_error}")
        
        # Stop watchdog thread (normal completion path)
        watchdog_done.set()
        
        # Hard exit-path after perf JSON export (prevents zombie master)
        # If we were stopped or timed out, exit immediately after perf export
        # Use os._exit(0) to bypass all cleanup handlers and ensure immediate termination
        if MASTER_STOP_REQUESTED or (hasattr(main, '_deadline') and time.time() > main._deadline):
            reason = "SIGTERM" if MASTER_STOP_REQUESTED else "timeout"
            log.info(f"[MASTER] Exiting immediately after perf JSON export (reason={reason})")
            os._exit(0)  # Hard exit after perf JSON is written (prevents zombie master)


if __name__ == "__main__":
    main()
