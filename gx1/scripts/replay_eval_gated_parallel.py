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

import pandas as pd

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

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

from gx1.execution.oanda_demo_runner import GX1DemoRunner

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
    data_path: Path, n_chunks: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Split data into N time-based chunks (no overlap).
    
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


def process_chunk(
    chunk_idx: int,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    data_path: Path,
    policy_path: Path,
    run_id: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Process a single chunk in a worker process.
    
    DEL 1: Wrapped in try/finally to guarantee flush even on exceptions.
    DEL 2: SIGTERM handler for graceful shutdown.
    
    Returns dict with chunk results and paths to artifacts.
    """
    global STOP_REQUESTED
    
    # DEL 2: Install SIGTERM handler for graceful shutdown
    signal.signal(signal.SIGTERM, _sigterm_handler)
    STOP_REQUESTED = False
    
    worker_start_time = time.time()
    chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    log.info(
        f"[CHUNK {chunk_idx}] Starting: {chunk_start} to {chunk_end}"
    )
    
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
        os.environ["GX1_GATED_FUSION_ENABLED"] = "1"
        os.environ["GX1_REQUIRE_XGB_CALIBRATION"] = "1"
        os.environ["GX1_REPLAY_INCREMENTAL_FEATURES"] = "1"
        os.environ["GX1_REPLAY_NO_CSV"] = "1"
        os.environ["GX1_FEATURE_USE_NP_ROLLING"] = "1"
        os.environ["GX1_RUN_ID"] = run_id
        os.environ["GX1_CHUNK_ID"] = str(chunk_idx)
        
        # Set thread limits (already set globally, but ensure in worker)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        
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
        os.environ["GX1_STOP_REQUESTED"] = "0"
        os.environ["GX1_CHECKPOINT_EVERY_BARS"] = str(CHECKPOINT_EVERY_BARS)
        
        # DEL C: Set GX1_REPLAY_QUIET=1 for workers (default ON for replay_eval_gated_parallel.py)
        os.environ["GX1_REPLAY_QUIET"] = "1"
        
        # DEL C: Filter sklearn warnings (UserWarning spam)
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
        warnings.filterwarnings("ignore", category=UserWarning, message="Loky-backed")
        
        # DEL 2: Update env var when SIGTERM is received (via handler)
        def update_stop_flag():
            """Update env var when STOP_REQUESTED changes."""
            if STOP_REQUESTED:
                os.environ["GX1_STOP_REQUESTED"] = "1"
                runner._stop_requested = True
        
        # Run replay for this chunk
        # DEL 3: Checkpoint flush is handled inside bar loop via env var
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
        bars_processed = n_model_calls
        wall_clock_sec = time.time() - worker_start_time  # DEL 2: Always define wall_clock_sec (even on early failure)
        
        # DEL 1: Extract performance summary metrics from runner
        total_bars = getattr(runner, "perf_n_bars_processed", 0)
        n_trades_created = getattr(runner, "perf_n_trades_created", 0)
        feature_time_total = getattr(runner, "perf_feat_time", 0.0)
        feature_time_mean_ms = (feature_time_total / total_bars * 1000.0) if total_bars > 0 else 0.0
        feature_timeout_count = getattr(runner, "feature_timeout_count", 0)
        htf_align_warn_count = getattr(runner, "htf_align_warn_count", 0)
        pregate_skips = getattr(runner, "pregate_skips", 0)
        pregate_passes = getattr(runner, "pregate_passes", 0)
        pregate_missing_inputs = getattr(runner, "pregate_missing_inputs", 0)
        
        # DEL 1: Extract phase timing (for "bars/sec" breakdown)
        t_pregate_total_sec = getattr(runner, "t_pregate_total_sec", 0.0)
        t_feature_build_total_sec = feature_time_total  # Pure feature build time (from perf_feat_time)
        t_model_total_sec = getattr(runner, "t_model_total_sec", 0.0)
        t_policy_total_sec = getattr(runner, "t_policy_total_sec", 0.0)
        t_io_total_sec = getattr(runner, "t_io_total_sec", 0.0)
        
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
        status = "failed"
        error = str(e)
        import traceback as tb
        error_traceback = "".join(tb.format_exception(type(e), e, e.__traceback__))
        log.error(f"[CHUNK {chunk_idx}] FAILED: {error}", exc_info=True)
        # DEL 2: Ensure wall_clock_sec is defined even on early failure
        if 'wall_clock_sec' not in locals():
            wall_clock_sec = time.time() - worker_start_time
    
    finally:
        # DEL 1: ALWAYS flush collectors, even on exceptions
        log.info(f"[FLUSH] chunk={chunk_idx} start")
        flush_count = 0
        
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
        try:
            import traceback as tb
            from datetime import datetime
            import numpy as np
            
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
            
            chunk_footer = {
                "run_id": run_id,
                "chunk_id": str(chunk_idx),
                "status": status,
                "error": error,
                "error_traceback": error_traceback[:5000] if error_traceback else None,  # Trim long tracebacks
                "n_model_calls": convert_to_json_serializable(n_model_calls),
                "n_trades_closed": convert_to_json_serializable(n_trades_closed),
                # DEL 1: Add performance summary metrics for A/B comparison
                "wall_clock_sec": convert_to_json_serializable(wall_clock_sec),
                "total_bars": convert_to_json_serializable(total_bars),
                "bars_per_sec": convert_to_json_serializable(bars_processed / wall_clock_sec if wall_clock_sec > 0 else 0.0),
                "feature_time_mean_ms": convert_to_json_serializable(feature_time_mean_ms),
                "feature_timeout_count": convert_to_json_serializable(feature_timeout_count),
                "htf_align_warn_count": convert_to_json_serializable(htf_align_warn_count),
                "pregate_skips": convert_to_json_serializable(pregate_skips),
                "pregate_passes": convert_to_json_serializable(pregate_passes),
                "pregate_missing_inputs": convert_to_json_serializable(pregate_missing_inputs),
                # DEL 1: Phase timing breakdown
                "t_pregate_total_sec": convert_to_json_serializable(t_pregate_total_sec),
                "t_feature_build_total_sec": convert_to_json_serializable(t_feature_build_total_sec),
                "t_model_total_sec": convert_to_json_serializable(t_model_total_sec),
                "t_policy_total_sec": convert_to_json_serializable(t_policy_total_sec),
                "t_io_total_sec": convert_to_json_serializable(t_io_total_sec),
                "bars_processed": convert_to_json_serializable(bars_processed),
                "start_ts": chunk_start.isoformat() if chunk_start else None,
                "end_ts": chunk_end.isoformat() if chunk_end else None,
                "worker_time_sec": float(time.time() - worker_start_time),
                "pid": int(os.getpid()),
                "timestamp": datetime.now().isoformat(),
            }
            
            # Convert all values to JSON-serializable types
            chunk_footer = convert_to_json_serializable(chunk_footer)
            
            chunk_footer_path = chunk_output_dir / "chunk_footer.json"
            with open(chunk_footer_path, "w") as f:
                json.dump(chunk_footer, f, indent=2)
            
            log.info(f"[CHUNK {chunk_idx}] chunk_footer.json written: status={status}")
        except Exception as footer_error:
            log.error(f"[CHUNK {chunk_idx}] Failed to write chunk_footer.json: {footer_error}", exc_info=True)
    
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
                
                chunks_metrics.append({
                    "chunk_idx": chunk_idx,
                    "wall_clock_sec": footer.get("wall_clock_sec", footer.get("worker_time_sec", 0.0)),
                    "total_bars": footer.get("total_bars", footer.get("bars_processed", 0)),
                    "bars_per_sec": footer.get("bars_per_sec", 0.0),
                    "n_model_calls": footer.get("n_model_calls", 0),
                    "n_trades_closed": footer.get("n_trades_closed", 0),
                    "feature_time_mean_ms": footer.get("feature_time_mean_ms", 0.0),
                    "feature_timeout_count": footer.get("feature_timeout_count", 0),
                    "htf_align_warn_count": footer.get("htf_align_warn_count", 0),
                    "pregate_skips": footer.get("pregate_skips", 0),
                    "pregate_passes": footer.get("pregate_passes", 0),
                    "pregate_missing_inputs": footer.get("pregate_missing_inputs", 0),
                    "status": status,
                    # Phase timing breakdown
                    "t_pregate_total_sec": footer.get("t_pregate_total_sec", 0.0),
                    "t_feature_build_total_sec": footer.get("t_feature_build_total_sec", 0.0),
                    "t_model_total_sec": footer.get("t_model_total_sec", 0.0),
                    "t_policy_total_sec": footer.get("t_policy_total_sec", 0.0),
                    "t_io_total_sec": footer.get("t_io_total_sec", 0.0),
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
    
    # Compute aggregate stats
    total_model_calls = sum(m.get("n_model_calls", 0) for m in chunks_metrics)
    total_bars = sum(m.get("total_bars", 0) for m in chunks_metrics)
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
    
    perf_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "pregate_enabled": pregate_enabled,
        "requested_workers": workers,
        "actual_workers_started": actual_workers_started,
        "workers": workers,  # Keep for backward compatibility
        "total_wall_clock_sec": total_time,
        "total_bars": total_bars,
        "total_model_calls": total_model_calls,
        "chunks_completed": chunks_completed,
        "chunks_total": len(chunks),
        "chunks_statuses": chunks_statuses,
        "chunks": chunks_metrics,
        "env_info": env_info,
        "export_mode": export_mode,  # "normal" or "watchdog_sigterm"
        "export_partial": export_partial,  # True if workers still running during export
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
    
    return perf_json_path


def merge_artifacts(
    chunk_results: List[Dict[str, Any]],
    run_id: str,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Merge artifacts from all chunks into single files.
    
    Returns dict with paths to merged artifacts.
    """
    log.info(f"[MERGE] Merging {len(chunk_results)} chunks")
    
    merged_artifacts = {}
    
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
    
    # Merge JSON files (attribution, metrics)
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
    
    # Merge metrics: aggregate trade-level metrics
    merged_metrics = {
        "n_trades": 0,
        "total_pnl_bps": 0.0,
        "calibration_stats": {},
    }
    
    all_trade_outcomes = []
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
    parser = argparse.ArgumentParser(description="Parallel replay evaluation for GATED_FUSION")
    parser.add_argument("--policy", type=Path, required=True, help="Policy YAML path")
    parser.add_argument("--data", type=Path, required=True, help="Input data (parquet)")
    parser.add_argument("--workers", type=int, default=7, help="Number of parallel workers")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/replay_eval/GATED"), help="Output directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    
    args = parser.parse_args()
    
    # DEL 1: Verify GX1_GATED_FUSION_ENABLED=1
    gated_fusion_enabled = os.getenv("GX1_GATED_FUSION_ENABLED", "0") == "1"
    if not gated_fusion_enabled:
        raise RuntimeError(
            "BASELINE_DISABLED: GX1_GATED_FUSION_ENABLED is not '1'. "
            "Set GX1_GATED_FUSION_ENABLED=1 to run replay eval."
        )
    
    # Generate run_id
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log.info(f"[PARALLEL] Starting parallel replay evaluation")
    log.info(f"[PARALLEL] Policy: {args.policy}")
    log.info(f"[PARALLEL] Data: {args.data}")
    log.info(f"[PARALLEL] Workers: {args.workers}")
    log.info(f"[PARALLEL] Run ID: {run_id}")
    log.info(f"[PARALLEL] Output: {args.output_dir}")
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
                        log.error(f"[WATCHDOG] Export failed: {e}", exc_info=True)
                        
                        # CRITICAL: Write stub file with error info before exiting
                        import traceback
                        error_stub = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(),
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
                                "quiet_mode": os.getenv("GX1_REPLAY_QUIET", "0") == "1",
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
    
    # Split data into chunks
    chunks = split_data_into_chunks(args.data, args.workers)
    
    # Process chunks in parallel
    start_time = time.time()
    
    # Master timeout (default 6 hours for FULLYEAR, but configurable)
    master_timeout_sec = float(os.getenv("GX1_MASTER_WAIT_TIMEOUT_SEC", "21600"))  # 6 hours default
    
    chunk_tasks = [
        (
            chunk_idx,
            chunk_start,
            chunk_end,
            args.data,
            args.policy,
            run_id,
            args.output_dir,
        )
        for chunk_start, chunk_end, chunk_idx in chunks
    ]
    
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
        
        pool = mp.Pool(processes=args.workers)
        actual_workers_started = args.workers  # Track actual workers started (for perf JSON)
        async_results = [
            pool.apply_async(process_chunk, args=task) for task in chunk_tasks
        ]
        
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
                    # Check if result was successful before trying to get()
                    if not result.successful():
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
                    except Exception as e:
                        log.warning(f"[MASTER] Error getting result for chunk {i} (will use footer): {e}")
                        completed.add(i)  # Mark as done to avoid infinite loop, will read from footer
            
            if len(completed) < len(async_results):
                time.sleep(poll_interval)
        
        total_time = time.time() - start_time
        
        # Deterministic pool cleanup: close() if all done, terminate() if stop/timeout
        all_done = len(completed) == len(async_results) and not MASTER_STOP_REQUESTED
        if all_done:
            log.info(f"[PARALLEL] All chunks completed in {total_time:.1f}s")
            log.info("[MASTER] Closing pool (all chunks done)...")
            pool.close()  # Prevent new tasks, allow existing to finish
            pool.join(timeout=30)  # Wait for workers to finish gracefully
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
                    pool.join(timeout=5)  # Short timeout after terminate
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
        log.warning(f"[PARALLEL] Interrupted: {e}")
        total_time = time.time() - start_time
        if pool:
            log.warning("[MASTER] Terminating pool due to interrupt...")
            pool.terminate()
            pool.join(timeout=30)
            pool = None
        # Continue to finally to write perf JSON
    except Exception as e:
        log.error(f"[PARALLEL] Error during execution: {e}", exc_info=True)
        total_time = time.time() - start_time
        if pool:
            log.warning("[MASTER] Terminating pool due to error...")
            pool.terminate()
            pool.join(timeout=30)
            pool = None
        # Continue to finally to write perf JSON
    finally:
        # Always write perf JSON (even on SIGTERM/KeyboardInterrupt/errors)
        log.info("[PERF] Exporting performance JSON (always, even on interrupt)...")
        
        perf_json_written = False
        perf_json_path = None
        
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
                        raise RuntimeError(f"Perf JSON path does not exist after export: {perf_json_path}")
                        
                except Exception as perf_error:
                    log.error(f"[PERF] Failed to write perf JSON: {perf_error}", exc_info=True)
                    
                    # CRITICAL: Write stub file with error info
                    import traceback
                    error_stub = {
                        "run_id": run_id,
                        "timestamp": datetime.now().isoformat(),
                        "status": "export_failed",
                        "export_error": str(perf_error),
                        "export_traceback": "".join(traceback.format_exception(type(perf_error), perf_error, perf_error.__traceback__)),
                        "chunks_total": len(chunks),
                        "chunks_statuses": {},
                        "note": "Perf export failed - this is a stub file written by finally block",
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
                        log.error(f"[PERF] Wrote error stub to {stub_path}")
                    except Exception as stub_error:
                        log.error(f"[PERF] Failed to write error stub: {stub_error}")
        
        # Log final summary
        log.info(f"[PARALLEL] Completed (total_time={total_time:.1f}s, perf_json_written={perf_json_written})")
        if merged_artifacts:
            log.info(f"[PARALLEL] Merged artifacts: {list(merged_artifacts.keys())}")
        
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
