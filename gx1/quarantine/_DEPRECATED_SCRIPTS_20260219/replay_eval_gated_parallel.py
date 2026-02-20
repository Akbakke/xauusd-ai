#!/home/andre2/venvs/gx1/bin/python
# -*- coding: utf-8 -*-
"""
Replay Evaluation for GATED_FUSION - Parallel Chunked Execution

Runs FULLYEAR replay with N workers, each processing a time chunk.
Each worker produces its own artifacts, which are merged at the end.

CRITICAL: This script has MASTER DEATH FORENSICS enabled.
- faulthandler is installed for native crash dumps
- All exits write completion contract (RUN_COMPLETED.json or RUN_FAILED.json)
"""

# ============================================================================
# ENV IDENTITY GATE: hard-fail on wrong interpreter BEFORE heavy imports
# ============================================================================
import sys
import os

REQUIRED_VENV = "/home/andre2/venvs/gx1/bin/python"


def _env_identity_gate() -> None:
    """
    Hard-fail on wrong python interpreter.

    NOTE: This file is imported as a module by other code (e.g. `replay_worker.py`),
    so the gate must NOT run at import-time.
    """
    if sys.executable != REQUIRED_VENV:
        raise RuntimeError(
            f"[ENV_IDENTITY_FAIL] Wrong python interpreter\n"
            f"Expected: {REQUIRED_VENV}\n"
            f"Actual:   {sys.executable}\n"
            f"Hint: source ~/venvs/gx1/bin/activate"
        )


if __name__ == "__main__":
    _env_identity_gate()

# ============================================================================
# MASTER DEATH FORENSICS: faulthandler MUST be first (before any imports that can crash)
# ============================================================================
import faulthandler
import signal

# Get output path for fault dump (from env or default)
_fault_dump_path = os.environ.get("GX1_FAULTHANDLER_OUTPUT", "/tmp/gx1_master_fault_dump.txt")
try:
    _fault_dump_file = open(_fault_dump_path, "w")
    faulthandler.enable(file=_fault_dump_file, all_threads=True)
    # Also enable on stderr as backup
    faulthandler.enable(file=sys.stderr, all_threads=True)
    # Register SIGUSR1 for manual dump trigger
    faulthandler.register(signal.SIGUSR1, file=_fault_dump_file, all_threads=True)
    print(f"[FAULTHANDLER] Enabled, output to {_fault_dump_path}", flush=True)
except Exception as _fh_err:
    print(f"[FAULTHANDLER] WARNING: Could not enable faulthandler: {_fh_err}", flush=True)
    _fault_dump_file = None

# ============================================================================
# STANDARD IMPORTS (after faulthandler)
# ============================================================================
import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import subprocess
import threading
import time
from datetime import datetime, timezone
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
    now_iso as _dt_now_iso,  # Alias for backward compatibility
    strftime_now as dt_strftime_now,
)

try:
    import pandas as pd
except ImportError:
    # Allow master to boot far enough to write ENV_IDENTITY_FATAL.json in TRUTH/SMOKE.
    pd = None

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

# CRITICAL: Log immediately when script starts (for wrapper verification)
print("[RUN] starting replay master", flush=True)

# DEL 2: Global flag for graceful shutdown (set by SIGTERM handler)
STOP_REQUESTED = False
MASTER_STOP_REQUESTED = False
POOL_REF = None  # Global reference to pool for SIGTERM handler
PERF_EXPORTED = False  # Global flag to prevent double export
PERF_EXPORT_LOCK = None  # Threading lock for perf export (initialized in main)

# HANG DETECTION: Global state for hang dump
MASTER_OUTPUT_DIR = None  # Set in main() for hang dump
MASTER_ASYNC_RESULTS = None  # Set in main() for hang dump
MASTER_COMPLETED = None  # Set in main() for hang dump


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


def write_master_os_error(op: str, paths: Dict[str, Any], exc: Exception, output_dir: Optional[Path]) -> Path:
    """
    Write MASTER_OS_ERROR capsule with full diagnostic information.
    
    Args:
        op: Operation that failed (e.g., "write_json_atomic", "output_dir_init", "makedirs")
        paths: Dict of path names to Path objects (e.g., {"target": path, "tmp": tmp_path})
        exc: Exception that occurred
        output_dir: Output directory (may be None if init failed)
    
    Returns:
        Path to written capsule file
    """
    import traceback
    import stat
    
    try:
        # Build capsule data
        capsule_data = {
            "op": op,
            "paths": {k: str(v) for k, v in paths.items()},
            "errno": getattr(exc, "errno", None),
            "strerror": getattr(exc, "strerror", None),
            "repr": repr(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "cwd": os.getcwd(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "gid": os.getgid() if hasattr(os, "getgid") else None,
            "umask": oct(os.umask(os.umask(0))) if hasattr(os, "umask") else "<best effort>",
            "gx1_data_env": {
                "GX1_DATA": os.getenv("GX1_DATA"),
                "GX1_DATA_DIR": os.getenv("GX1_DATA_DIR"),
                "GX1_DATA_ROOT": os.getenv("GX1_DATA_ROOT"),
            },
            "output_dir": str(output_dir) if output_dir else None,
            "output_dir_realpath": str(output_dir.resolve()) if output_dir and output_dir.exists() else None,
            "timestamp": dt_now_iso(),
            "pid": os.getpid(),
        }
        
        # Best effort: get disk free space if path exists
        try:
            if output_dir and output_dir.exists():
                statvfs = os.statvfs(output_dir)
                capsule_data["disk_free_gb"] = (statvfs.f_bavail * statvfs.f_frsize) / (1024.0 ** 3)
                capsule_data["disk_total_gb"] = (statvfs.f_blocks * statvfs.f_frsize) / (1024.0 ** 3)
            else:
                # Try /tmp
                statvfs = os.statvfs("/tmp")
                capsule_data["disk_free_gb"] = (statvfs.f_bavail * statvfs.f_frsize) / (1024.0 ** 3)
                capsule_data["disk_total_gb"] = (statvfs.f_blocks * statvfs.f_frsize) / (1024.0 ** 3)
        except Exception:
            capsule_data["disk_free_gb"] = None
            capsule_data["disk_total_gb"] = None
        
        # Check if output_dir is under reports (best effort)
        try:
            from gx1.utils.output_dir import resolve_gx1_data_root
            gx1_data_root = resolve_gx1_data_root()
            if output_dir:
                output_dir_resolved = output_dir.resolve()
                gx1_data_root_resolved = gx1_data_root.resolve()
                reports_dir = gx1_data_root_resolved / "reports"
                capsule_data["is_under_reports"] = str(output_dir_resolved).startswith(str(reports_dir))
                capsule_data["gx1_data_root"] = str(gx1_data_root_resolved)
            else:
                capsule_data["is_under_reports"] = None
                capsule_data["gx1_data_root"] = str(gx1_data_root.resolve())
        except Exception:
            capsule_data["is_under_reports"] = None
            capsule_data["gx1_data_root"] = None
        
        # Try to write to output_dir first
        if output_dir:
            try:
                # Ensure output_dir exists (best effort)
                output_dir.mkdir(parents=True, exist_ok=True)
                capsule_path = output_dir / "MASTER_OS_ERROR.json"
                with open(capsule_path, "w", encoding="utf-8") as f:
                    json.dump(capsule_data, f, indent=2, sort_keys=True)
                    f.flush()
                    os.fsync(f.fileno())
                log.error(f"[MASTER_OS_ERROR] Wrote capsule to {capsule_path}")
                return capsule_path
            except Exception as write_error:
                log.warning(f"[MASTER_OS_ERROR] Failed to write to output_dir: {write_error}, trying /tmp fallback")
        
        # Fallback to /tmp
        import tempfile
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = Path(tempfile.gettempdir()) / f"MASTER_OS_ERROR_{timestamp}_{os.getpid()}.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(capsule_data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        log.error(f"[MASTER_OS_ERROR] Wrote capsule to fallback: {fallback_path}")
        return fallback_path
        
    except Exception as capsule_error:
        # Last resort: try to write minimal error to stderr
        log.error(f"[MASTER_OS_ERROR] CRITICAL: Failed to write OS error capsule: {capsule_error}")
        log.error(f"[MASTER_OS_ERROR] Original error: {op} failed: {exc}")
        # Return None to indicate failure
        return None


def write_json_atomic(path: Path, obj: Dict[str, Any], output_dir: Optional[Path] = None) -> bool:
    """
    Write JSON file atomically (tmp -> rename) with OS error capsule on failure.
    
    Args:
        path: Target file path
        obj: Dictionary to write as JSON
        output_dir: Output directory for error capsule (optional, defaults to path.parent)
    
    Returns:
        True if successful, False otherwise (capsule written on failure)
    """
    if output_dir is None:
        output_dir = path.parent
    
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file first
        tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)
                f.flush()  # Force write to OS buffer
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic rename (POSIX guarantees this is atomic)
            os.replace(tmp_path, path)
            return True
            
        except OSError as e:
            # Write OS error capsule before re-raising
            write_master_os_error(
                op="write_json_atomic",
                paths={"target": path, "tmp": tmp_path},
                exc=e,
                output_dir=output_dir,
            )
            # Clean up temp file if it exists
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise  # Re-raise for TRUTH mode (hard fail)
            
    except OSError as e:
        # Write OS error capsule (directory creation failed)
        write_master_os_error(
            op="write_json_atomic",
            paths={"target": path, "parent": path.parent},
            exc=e,
            output_dir=output_dir,
        )
        raise  # Re-raise for TRUTH mode (hard fail)


def dump_master_hang_state(output_dir: Path):
    """
    Dump master hang state when heartbeat timeout is detected.
    Writes MASTER_HANG_DUMP.txt and MASTER_FUTURES_STATE.json.
    """
    try:
        import traceback
        import threading
        
        hang_dump_path = output_dir / "MASTER_HANG_DUMP.txt"
        futures_state_path = output_dir / "MASTER_FUTURES_STATE.json"
        
        with open(hang_dump_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MASTER HANG STATE DUMP\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {dt_now_iso()}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write("\n")
            
            # Dump stacktrace for all threads
            f.write("STACKTRACE (ALL THREADS):\n")
            f.write("-" * 80 + "\n")
            faulthandler.dump_traceback(file=f, all_threads=True)
            f.write("\n")
            
            # Dump thread states
            f.write("THREAD STATES:\n")
            f.write("-" * 80 + "\n")
            for thread_id, frame in sys._current_frames().items():
                thread_name = threading.current_thread().name if threading.current_thread().ident == thread_id else f"Thread-{thread_id}"
                f.write(f"\nThread {thread_id} ({thread_name}):\n")
                f.write("".join(traceback.format_stack(frame)))
            f.write("\n")
            
            # Dump executor/pool state if available
            f.write("EXECUTOR/POOL STATE:\n")
            f.write("-" * 80 + "\n")
            if MASTER_ASYNC_RESULTS is not None:
                f.write(f"async_results length: {len(MASTER_ASYNC_RESULTS)}\n")
                f.write(f"completed: {MASTER_COMPLETED}\n")
            if POOL_REF is not None:
                f.write(f"POOL_REF: {POOL_REF}\n")
                f.write(f"POOL_REF type: {type(POOL_REF)}\n")
            f.write("\n")
        
        # Dump futures state to JSON
        futures_state = {
            "timestamp": dt_now_iso(),
            "pid": os.getpid(),
            "async_results_count": len(MASTER_ASYNC_RESULTS) if MASTER_ASYNC_RESULTS is not None else 0,
            "completed": list(MASTER_COMPLETED) if MASTER_COMPLETED is not None else [],
            "pending": [],
        }
        
        if MASTER_ASYNC_RESULTS is not None:
            for i, result in enumerate(MASTER_ASYNC_RESULTS):
                try:
                    is_ready = result.ready() if hasattr(result, 'ready') else None
                    is_successful = None
                    if is_ready:
                        try:
                            is_successful = result.successful() if hasattr(result, 'successful') else None
                        except (ValueError, AttributeError):
                            is_successful = None
                except (ValueError, AttributeError):
                    is_ready = None
                    is_successful = None
                
                future_state = {
                    "chunk_idx": i,
                    "ready": is_ready,
                    "successful": is_successful,
                    "done": result.done() if hasattr(result, 'done') else None,
                    "cancelled": result.cancelled() if hasattr(result, 'cancelled') else None,
                }
                if MASTER_COMPLETED is None or i not in MASTER_COMPLETED:
                    futures_state["pending"].append(future_state)
        
        with open(futures_state_path, "w") as f:
            json.dump(futures_state, f, indent=2)
        
        log.error(f"[MASTER] Hang state dumped to {hang_dump_path} and {futures_state_path}")
    except Exception as e:
        log.error(f"[MASTER] Failed to dump hang state: {e}", exc_info=True)


def _master_hang_handler(signum, frame):
    """Signal handler for hang detection (SIGUSR2)."""
    log.warning(f"[MASTER] Received SIGUSR2 (hang detection trigger)")
    if MASTER_OUTPUT_DIR is not None:
        dump_master_hang_state(MASTER_OUTPUT_DIR)


def write_master_early_capsule(
    output_dir: Path,
    stage: str,
    argv: List[str],
    resolved_paths: Optional[Dict[str, Any]] = None,
    run_mode: Optional[str] = None,
    workers_requested: Optional[int] = None,
    workers_effective: Optional[int] = None,
) -> None:
    """
    Write MASTER_EARLY.json capsule at critical stages to track early crashes.
    
    Args:
        output_dir: Output directory (fallback to /tmp if not writable)
        stage: Stage identifier (e.g., "args_resolved", "gates_passed", "before_chunk_planning")
        argv: Command-line arguments
        resolved_paths: Dictionary with resolved paths (data, policy, output_dir, bundle_dir)
        run_mode: TRUTH/SMOKE/other
        workers_requested: Requested number of workers
        workers_effective: Effective number of workers (after RAM scheduler)
    """
    import tempfile
    
    # Try output_dir first, fallback to /tmp
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir = output_dir
    except Exception:
        report_dir = Path(tempfile.gettempdir()) / "gx1_master_early"
        report_dir.mkdir(parents=True, exist_ok=True)
    
    capsule_path = report_dir / "MASTER_EARLY.json"
    
    capsule = {
        "timestamp": dt_now_iso(),
        "pid": os.getpid(),
        "stage": stage,
        "sys.executable": sys.executable,
        "sys.version": sys.version.split()[0],  # Just version string, not full sys.version
        "cwd": str(Path.cwd()),
        "argv": argv,
        "resolved_paths": resolved_paths or {},
        "run_mode": run_mode,
        "workers_requested": workers_requested,
        "workers_effective": workers_effective,
    }
    
    # Atomic write with /tmp fallback
    try:
        write_json_atomic(capsule_path, capsule, output_dir=report_dir)
        log.info(f"[MASTER_EARLY] Stage '{stage}' capsule written to {capsule_path}")
    except Exception as e:
        # Last resort: write to /tmp directly
        tmp_path = Path(tempfile.gettempdir()) / f"gx1_master_early_{os.getpid()}_{stage}.json"
        try:
            with open(tmp_path, "w") as f:
                json.dump(capsule, f, indent=2)
            log.warning(f"[MASTER_EARLY] Fallback write to {tmp_path} (original error: {e})")
        except Exception:
            log.error(f"[MASTER_EARLY] Failed to write capsule even to /tmp: {e}")


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


def _write_killchain_ssoT_diff(
    chunk_output_dir: Path,
    chunk_idx: int,
    run_id: str,
    policy_hash: Optional[str],
    bundle_sha: Optional[str],
    replay_mode_enum: Optional[Any],
    stage2_total: Optional[int],
    stage2_pass: Optional[int],
    stage2_block: Optional[int],
    stage2_early_return: Optional[int],
    stage2_block_reasons: Optional[Dict[str, int]],
    stage2_early_return_reasons: Optional[Dict[str, int]],
    stage3_total: Optional[int],
    stage3_pass: Optional[int],
    stage3_block: Optional[int],
    stage3_early_return: Optional[int],
    stage3_block_reasons: Optional[Dict[str, int]],
    stage3_early_return_reasons: Optional[Dict[str, int]],
    error_type: str,
    error_msg: str,
    chunk_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Write KILLCHAIN_SSoT_DIFF.json capsule when KILLCHAIN stage invariant fails.
    
    This provides full diagnostic context for debugging KILLCHAIN SSoT mismatches.
    """
    try:
        from gx1.utils.atomic_json import atomic_write_json
        from datetime import datetime
        
        # Get first/last timestamps
        first_ts = None
        last_ts = None
        if chunk_df is not None and len(chunk_df) > 0:
            first_ts = str(chunk_df.index[0]) if hasattr(chunk_df.index, '__getitem__') else None
            last_ts = str(chunk_df.index[-1]) if hasattr(chunk_df.index, '__getitem__') else None
        
        # Build diff capsule
        diff_capsule = {
            "chunk_id": chunk_idx,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_msg,
            "policy_hash": policy_hash,
            "bundle_sha": bundle_sha,
            "replay_mode_enum": str(replay_mode_enum) if replay_mode_enum else None,
            "first_ts": first_ts,
            "last_ts": last_ts,
            # E) All stage totals/pass/block/early_return
            "stage2": {
                "total": stage2_total,
                "pass": stage2_pass,
                "block": stage2_block,
                "early_return": stage2_early_return,
                "block_reasons_top10": dict(sorted((stage2_block_reasons or {}).items(), key=lambda kv: -kv[1])[:10]),
                "early_return_reasons_top10": dict(sorted((stage2_early_return_reasons or {}).items(), key=lambda kv: -kv[1])[:10]),
            },
            "stage3": {
                "total": stage3_total,
                "pass": stage3_pass,
                "block": stage3_block,
                "early_return": stage3_early_return,
                "block_reasons_top10": dict(sorted((stage3_block_reasons or {}).items(), key=lambda kv: -kv[1])[:10]),
                "early_return_reasons_top10": dict(sorted((stage3_early_return_reasons or {}).items(), key=lambda kv: -kv[1])[:10]),
            },
        }
        
        diff_path = chunk_output_dir / "KILLCHAIN_SSoT_DIFF.json"
        atomic_write_json(diff_capsule, diff_path)
        log.error(f"[KILLCHAIN_SSoT_DIFF] Wrote diff capsule to: {diff_path}")
    except Exception as e:
        log.warning(f"[KILLCHAIN_SSoT_DIFF] Failed to write diff capsule: {e}")


def _write_ssoT_diff_capsule(
    chunk_output_dir: Path,
    chunk_idx: int,
    run_id: str,
    policy_hash: Optional[str],
    bundle_sha: Optional[str],
    replay_mode_enum: Optional[Any],
    counter_context: Dict[str, Any],
    expected_lookup_attempts: Optional[int],
    actual_lookup_attempts: Optional[int],
    stage2_total: Optional[int],
    stage2_pass: Optional[int],
    stage2_block: Optional[int],
    error_type: str,
    error_msg: str,
    chunk_df: Optional[pd.DataFrame],
    post_soft_prelookup_reached: Optional[int] = None,
    post_soft_early_return: Optional[int] = None,
    post_soft_early_return_reasons: Optional[Dict[str, int]] = None,
    bars_reaching_entry_stage_legacy: Optional[int] = None,
) -> None:
    """
    Write PREBUILT_SSoT_DIFF.json capsule when invariant fails.
    
    This provides full diagnostic context for debugging SSoT mismatches.
    """
    try:
        from gx1.utils.atomic_json import atomic_write_json
        from datetime import datetime
        
        # Get entry block reasons histogram
        entry_block_reasons = {}
        if chunk_output_dir.exists():
            # Try to read from telemetry if available
            telemetry_path = chunk_output_dir / "ENTRY_FEATURES_TELEMETRY.json"
            if telemetry_path.exists():
                try:
                    with open(telemetry_path, "r") as f:
                        telemetry = json.load(f)
                        entry_block_reasons = telemetry.get("entry_block_reasons", {})
                except Exception:
                    pass
        
        # Get first/last timestamps
        first_ts = None
        last_ts = None
        if chunk_df is not None and len(chunk_df) > 0:
            first_ts = str(chunk_df.index[0]) if hasattr(chunk_df.index, '__getitem__') else None
            last_ts = str(chunk_df.index[-1]) if hasattr(chunk_df.index, '__getitem__') else None
        
        # Build diff capsule
        diff_capsule = {
            "chunk_id": chunk_idx,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_msg,
            "policy_hash": policy_hash,
            "bundle_sha": bundle_sha,
            "replay_mode_enum": str(replay_mode_enum) if replay_mode_enum else None,
            "counter_context": counter_context,
            "expected_lookup_attempts": expected_lookup_attempts,
            "actual_lookup_attempts": actual_lookup_attempts,
            "stage2_total": stage2_total,
            "stage2_pass": stage2_pass,
            "stage2_block": stage2_block,
            "entry_block_reasons_top10": dict(sorted(entry_block_reasons.items(), key=lambda kv: -kv[1])[:10]) if entry_block_reasons else {},
            "first_ts": first_ts,
            "last_ts": last_ts,
            # D) Add prelookup counters to diff capsule
            "post_soft_prelookup_reached": post_soft_prelookup_reached,
            "post_soft_early_return": post_soft_early_return,
            "post_soft_early_return_reasons": post_soft_early_return_reasons if post_soft_early_return_reasons else {},
            "bars_reaching_entry_stage_legacy": bars_reaching_entry_stage_legacy,  # Legacy counter for comparison
        }
        
        diff_path = chunk_output_dir / "PREBUILT_SSoT_DIFF.json"
        atomic_write_json(diff_capsule, diff_path)
        log.error(f"[SSoT_DIFF] Wrote diff capsule to: {diff_path}")
    except Exception as e:
        log.warning(f"[SSoT_DIFF] Failed to write diff capsule: {e}")


def coalesce_chunks_by_min_post_warmup_bars(
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    data_path: Path,
    min_post_warmup_bars: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Coalesce chunks to ensure each chunk has at least min_post_warmup_bars after warmup.
    
    Args:
        chunks: List of (start_ts, end_ts, chunk_idx) tuples
        data_path: Path to data file (to compute bar counts)
        min_post_warmup_bars: Minimum bars after warmup per chunk
    
    Returns:
        Coalesced chunks list with fewer, larger chunks.
    """
    if min_post_warmup_bars is None or min_post_warmup_bars <= 0:
        return chunks
    
    log.info(f"[COALESCE] Coalescing chunks to ensure min {min_post_warmup_bars} bars after warmup")
    
    # Load data to compute bar counts
    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df.index = pd.to_datetime(df["ts"])
        else:
            raise ValueError("Data must have datetime index or 'ts' column")
    df = df.sort_index()
    
    # Estimate warmup (100 bars for M5, plus HTF warmup if enabled)
    # Conservative estimate: assume 100 + 200 = 300 bars warmup
    warmup_bars = 300
    
    coalesced = []
    current_chunk_start = None
    current_chunk_end = None
    current_chunk_bars = 0
    new_chunk_idx = 0
    
    for chunk_start, chunk_end, old_chunk_idx in chunks:
        if current_chunk_start is None:
            # Start new coalesced chunk
            current_chunk_start = chunk_start
            current_chunk_end = chunk_end
        else:
            # Extend current coalesced chunk
            current_chunk_end = chunk_end
        
        # Re-count bars in the actual coalesced chunk range (not sum of individual chunks)
        coalesced_df = df.loc[current_chunk_start:current_chunk_end]
        current_chunk_bars = len(coalesced_df)
        estimated_post_warmup_total = max(0, current_chunk_bars - warmup_bars)
        
        if estimated_post_warmup_total >= min_post_warmup_bars:
            # Current coalesced chunk meets requirement, finalize it
            coalesced.append((current_chunk_start, current_chunk_end, new_chunk_idx))
            log.info(
                f"[COALESCE] Coalesced chunk {new_chunk_idx}: {current_chunk_start} to {current_chunk_end} "
                f"({current_chunk_bars} bars, ~{estimated_post_warmup_total} after warmup)"
            )
            new_chunk_idx += 1
            current_chunk_start = None
            current_chunk_end = None
    
    # Finalize last chunk if any
    if current_chunk_start is not None:
        # Re-count bars in the final coalesced chunk
        coalesced_df = df.loc[current_chunk_start:current_chunk_end]
        final_chunk_bars = len(coalesced_df)
        estimated_post_warmup_total = max(0, final_chunk_bars - warmup_bars)
        coalesced.append((current_chunk_start, current_chunk_end, new_chunk_idx))
        log.info(
            f"[COALESCE] Coalesced chunk {new_chunk_idx}: {current_chunk_start} to {current_chunk_end} "
            f"({final_chunk_bars} bars, ~{estimated_post_warmup_total} after warmup)"
        )
    
    log.info(f"[COALESCE] Coalesced {len(chunks)} chunks into {len(coalesced)} chunks")
    return coalesced


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
    chunk_start: "pd.Timestamp",
    chunk_end: "pd.Timestamp",
    data_path: Path,
    policy_path: Path,
    run_id: str,
    output_dir: Path,
    bundle_sha256: Optional[str] = None,
    prebuilt_parquet_path: Optional[str] = None,  # DEL 3: Force string, not Path
    bundle_dir: Optional[Path] = None,  # Bundle directory override
    chunk_local_padding_days: int = 0,  # Chunk-local padding days (TRUTH/SMOKE only)
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
    
    # A) Initialize SKIP_LEDGER early (always-on, written in finally)
    skip_ledger = {
        "chunk_id": chunk_idx,
        "run_id": run_id,
        "stage": "init",
        "timestamp": None,  # Will be set in finally
        # Raw/prebuilt/join stats (updated as we progress)
        "raw_rows_loaded": None,
        "prebuilt_rows_loaded": None,
        "join_rows": None,
        "join_ratio": None,
        "ts_min_raw": None,
        "ts_max_raw": None,
        "ts_min_prebuilt": None,
        "ts_max_prebuilt": None,
        "ts_min_join": None,
        "ts_max_join": None,
        # Eval window
        "eval_start_ts": str(chunk_start),
        "eval_end_ts": str(chunk_end),
        "n_in_eval_window": None,
        # Warmup
        "warmup_bars_required": None,
        "warmup_bars_seen": None,
        # Skip breakdown
        "n_skipped_total": None,
        "skipped_breakdown": {
            "skipped_warmup": None,
            "skipped_pregate": None,
            "skipped_no_eval_window": None,
            "skipped_missing_features": None,
            "skipped_session_gate": None,
            "skipped_score_gate": None,
            "skipped_vol_gate": None,
            "skipped_other": None,
            "skipped_other_reason": None,
        },
        # Counters
        "candles_iterated": None,
        "reached_entry_stage": None,
        "processed": None,
        "bars_processed": None,
        # Exception info (if any)
        "exception_type": None,
        "exception_msg": None,
        "traceback": None,
        # Gating counters
        "gating_counters": {},
    }
    
    # DEL 1: Wrap entire function in try/except to catch ALL exceptions (including import errors)
    try:
        # ------------------------------------------------------------------------
        # WORKER PRE-FORK FREEZE VERIFICATION (TRUTH/SMOKE ONLY)
        # ------------------------------------------------------------------------
        # Verify that PRE_FORK_FREEZE.json exists in run root before processing.
        # This ensures workers only start after master has verified all guards.
        # ------------------------------------------------------------------------
        run_mode = os.getenv("GX1_RUN_MODE", "").upper()
        is_truth_or_smoke = run_mode in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"
        if is_truth_or_smoke:
            prefork_freeze_path = output_dir / "PRE_FORK_FREEZE.json"
            if not prefork_freeze_path.exists():
                # Import dt_module for timestamp
                from gx1.utils.dt_module import now_iso as dt_now_iso
                fatal_capsule = {
                    "timestamp": dt_now_iso(),
                    "chunk_id": chunk_idx,
                    "run_id": run_id,
                    "fatal_reason": "WORKER_PREFORK_MISSING",
                    "error_message": f"PRE_FORK_FREEZE.json not found in run root: {output_dir}",
                    "output_dir": str(output_dir),
                }
                fatal_path = output_dir / f"chunk_{chunk_idx}" / "WORKER_PREFORK_MISSING_FATAL.json"
                try:
                    fatal_path.parent.mkdir(parents=True, exist_ok=True)
                    import json as json_module
                    with open(fatal_path, "w") as f:
                        json_module.dump(fatal_capsule, f, indent=2)
                except Exception as e:
                    log.error(f"[WORKER_PREFORK] Failed to write FATAL capsule: {e}")
                raise RuntimeError(
                    f"[WORKER_PREFORK_MISSING] PRE_FORK_FREEZE.json not found. "
                    f"Worker cannot start without master freeze verification. See {fatal_path}"
                )
        
        # DEL A: Minimal, korrekt fiks - WORKER_BOOT.json som første linje
        # Resolve chunk_dir first (must be absolute)
        chunk_output_dir = (output_dir / f"chunk_{chunk_idx}").resolve()
        
        # DEL A: Write WORKER_BOOT.json as FIRST line in process_chunk (hard fail if can't write)
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
            "pid": os.getpid(),
            "ppid": os.getppid() if hasattr(os, 'getppid') else None,
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
        
        # STEG 4C: Add memory watermark at boot
        try:
            with open("/proc/self/status", "r") as status_file:
                status_lines = status_file.readlines()
                for line in status_lines:
                    if line.startswith("VmRSS:"):
                        worker_boot_payload["memory_vmrss_kb"] = int(line.split()[1])
                    elif line.startswith("VmHWM:"):
                        worker_boot_payload["memory_vmhwm_kb"] = int(line.split()[1])
        except Exception as mem_error:
            worker_boot_payload["memory_read_error"] = str(mem_error)
        
        worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
        with open(worker_boot_path, "w") as f:
            json_module.dump(worker_boot_payload, f, indent=2)
            f.flush()  # Force write to OS buffer
            os.fsync(f.fileno())  # Force write to disk
        
        global STOP_REQUESTED
        
        # DEL 2: Install SIGTERM handler for graceful shutdown
        signal.signal(signal.SIGTERM, _sigterm_handler)
        STOP_REQUESTED = False
        
        worker_start_time = time.time()
        
        # TRUTH-only timing breakdown
        is_truth_or_smoke_worker = os.getenv("GX1_RUN_MODE", "").upper() in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"
        t_init_s = 0.0
        t_load_raw_s = 0.0
        t_load_prebuilt_s = 0.0
        t_join_s = 0.0
        t_loop_s = 0.0
        t_write_s = 0.0
        t_total_s = 0.0
        
        if is_truth_or_smoke_worker:
            t_init_start = time.time()
        
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
        cwd = str(Path.cwd())
        python_exe = REQUIRED_VENV
        output_dir_env = os.getenv("GX1_OUTPUT_DIR", "NOT_SET")
        prebuilt_env = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "NOT_SET")
        
        # DEL 4: Determine prebuilt path (explicit arg takes precedence, FATAL if missing)
        if prebuilt_parquet_path:
            # DEL 3: Already a string, resolve to absolute
            prebuilt_root = str(Path(prebuilt_parquet_path).resolve())
        else:
            # FATAL if arg missing (no fallback)
            fatal_msg = (
                f"[PREBUILT_FAIL] [CHUNK {chunk_idx}] prebuilt_parquet_path arg is None/empty."
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
            raise RuntimeError(fatal_msg)
        
        prebuilt_parquet_path_resolved = prebuilt_root
        
        # DEL 1: Update worker_start_info with resolved path
        prebuilt_path_obj = Path(prebuilt_parquet_path_resolved)
        worker_start_info["prebuilt_parquet_path_resolved"] = prebuilt_parquet_path_resolved
        worker_start_info["exists"] = prebuilt_path_obj.exists()
        if worker_start_info["exists"]:
            worker_start_info["size"] = prebuilt_path_obj.stat().st_size
        
        # DEL 5: Import logging NOW (before we use log)
        
        # Log SSoT info
        log.info(f"[CHUNK {chunk_idx}] [SSoT] Worker start diagnostics:")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   cwd = {cwd}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   sys.executable = {python_exe}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   GX1_OUTPUT_DIR = {output_dir_env}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   GX1_REPLAY_USE_PREBUILT_FEATURES = {prebuilt_env}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   prebuilt_parquet_path (arg) = {prebuilt_parquet_path}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   prebuilt_parquet_path_resolved = {prebuilt_parquet_path_resolved}")
        log.info(f"[CHUNK {chunk_idx}] [SSoT]   exists = {worker_start_info['exists']}, size = {worker_start_info['size']}")
        
        # DEL B: Validate prebuilt path (FATAL if missing when prebuilt enabled)
        prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
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
            
            # Prebuilt mode is enabled via GX1_REPLAY_USE_PREBUILT_FEATURES; path is passed explicitly via args.
            os.environ["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"  # CRITICAL: Enable PREBUILT mode
            log.info(f"[CHUNK {chunk_idx}] [SSoT] Using prebuilt_parquet_path from arg (not env): {prebuilt_parquet_path_resolved}")
            log.info(f"[CHUNK {chunk_idx}] [SSoT] Set GX1_REPLAY_USE_PREBUILT_FEATURES=1 to enable PREBUILT mode")
        
        # DEL 5: Flytt tunge imports inni try etter at WORKER_BOOT.json er skrevet
        # (for å fange import-krasj)
        # Import logging first (before other imports that might use log)
        
        try:
            import pandas as pd
        except Exception as import_error:
            # Write import fail capsule
            try:
                from gx1.utils.import_capsule import write_import_fail_capsule
                capsule_path = write_import_fail_capsule("REPLAY_BOOT", import_error, output_dir=chunk_output_dir)
                log.error(f"[IMPORT_FAIL] Wrote capsule to {capsule_path}")
            except Exception:
                pass
            
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
            # CHUNK-LOCAL PADDING: Extend chunk start backwards for warmup, but eval only counts [chunk_start, chunk_end]
            actual_chunk_start = chunk_start
            if chunk_local_padding_days > 0:
                actual_chunk_start = chunk_start - pd.Timedelta(days=chunk_local_padding_days)
                log.info(
                    f"[CHUNK {chunk_idx}] [CHUNK_LOCAL_PADDING] chunk_start={chunk_start}, "
                    f"actual_chunk_start={actual_chunk_start} (padding={chunk_local_padding_days} days), "
                    f"eval_window=[{chunk_start}, {chunk_end}]"
                )

            # Load RAW candles for this chunk window (fast: filter by time and keep only needed columns)
            # NOTE: raw candles must contain OHLC; prebuilt features are validated separately.
            import pyarrow.parquet as pq
            from gx1.utils.ts_utils import ensure_ts_column

            raw_filters = [
                ("time", ">=", actual_chunk_start.to_pydatetime()),
                ("time", "<=", chunk_end.to_pydatetime()),
            ]

            raw_table = pq.read_table(
                data_path,
                filters=raw_filters,
                # raw candles parquet is small; keep all columns (includes bid/ask if present)
            )
            raw_df = raw_table.to_pandas()
            if len(raw_df) == 0:
                raise ValueError(
                    f"Chunk {chunk_idx} raw candles are empty after filtering "
                    f"(actual_chunk_start={actual_chunk_start}, chunk_end={chunk_end})"
                )

            raw_df = ensure_ts_column(raw_df, context=f"raw_candles_chunk_{chunk_idx}")
            raw_df["ts"] = pd.to_datetime(raw_df["ts"], utc=True)
            if raw_df["ts"].duplicated().any():
                dup_count = int(raw_df["ts"].duplicated().sum())
                raise RuntimeError(f"RAW_TS_DUPLICATES: raw candles have {dup_count} duplicate ts values (chunk={chunk_idx})")
            raw_df = raw_df.sort_values("ts")
            chunk_df = raw_df.set_index("ts", drop=False)

            # Update ledger after raw load
            skip_ledger["stage"] = "raw_loaded"
            skip_ledger["raw_rows_loaded"] = len(chunk_df)
            skip_ledger["ts_min_raw"] = str(chunk_df["ts"].min()) if "ts" in chunk_df.columns else None
            skip_ledger["ts_max_raw"] = str(chunk_df["ts"].max()) if "ts" in chunk_df.columns else None

            # Store eval window boundaries for warmup ledger
            eval_start_ts = chunk_start  # Eval window starts at original chunk_start (not actual_chunk_start)
            eval_end_ts = chunk_end

            # Must have at least some bars in the eval window (excluding padding)
            n_raw_eval = int(((chunk_df["ts"] >= chunk_start) & (chunk_df["ts"] <= chunk_end)).sum())
            skip_ledger["n_in_eval_window"] = n_raw_eval
            if n_raw_eval == 0:
                raise RuntimeError(
                    f"RAW_EVAL_WINDOW_EMPTY: raw candles have 0 rows in eval window [{chunk_start}, {chunk_end}] "
                    f"(chunk={chunk_idx}, padding_start={actual_chunk_start})"
                )

            bars_processed = len(chunk_df)
            log.info(f"[CHUNK {chunk_idx}] Loaded raw candles: {bars_processed} bars (eval_window_bars={n_raw_eval})")

            # Validate RAW + PREBUILT alignment deterministically (join on ts) and write join metrics
            prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
            if prebuilt_enabled:
                # TRUTH-only timing: t_load_prebuilt_s
                if is_truth_or_smoke_worker:
                    t_load_prebuilt_start = time.time()
                
                prebuilt_filters = [
                    ("time", ">=", actual_chunk_start.to_pydatetime()),
                    ("time", "<=", chunk_end.to_pydatetime()),
                ]
                # Read ONLY timestamp column from prebuilt (cheap), then compute join stats on ts.
                prebuilt_time_table = pq.read_table(prebuilt_parquet_path_resolved, columns=["time"], filters=prebuilt_filters)
                prebuilt_time_df = prebuilt_time_table.to_pandas()
                
                if is_truth_or_smoke_worker:
                    t_load_prebuilt_s = time.time() - t_load_prebuilt_start
                if len(prebuilt_time_df) == 0:
                    raise RuntimeError(
                        f"PREBUILT_WINDOW_EMPTY: prebuilt has 0 rows after filtering "
                        f"(actual_chunk_start={actual_chunk_start}, chunk_end={chunk_end}, chunk={chunk_idx})"
                    )
                prebuilt_time_df = ensure_ts_column(prebuilt_time_df, context=f"prebuilt_times_chunk_{chunk_idx}")
                prebuilt_time_df["ts"] = pd.to_datetime(prebuilt_time_df["ts"], utc=True)
                if prebuilt_time_df["ts"].duplicated().any():
                    dup_count = int(prebuilt_time_df["ts"].duplicated().sum())
                    raise RuntimeError(f"PREBUILT_TS_DUPLICATES: prebuilt has {dup_count} duplicate ts values (chunk={chunk_idx})")
                prebuilt_time_df = prebuilt_time_df.sort_values("ts")

                # Update ledger after prebuilt load
                skip_ledger["stage"] = "prebuilt_loaded"
                skip_ledger["prebuilt_rows_loaded"] = len(prebuilt_time_df)
                skip_ledger["ts_min_prebuilt"] = str(prebuilt_time_df["ts"].min()) if "ts" in prebuilt_time_df.columns else None
                skip_ledger["ts_max_prebuilt"] = str(prebuilt_time_df["ts"].max()) if "ts" in prebuilt_time_df.columns else None

                n_prebuilt_eval = int(((prebuilt_time_df["ts"] >= chunk_start) & (prebuilt_time_df["ts"] <= chunk_end)).sum())
                if n_prebuilt_eval == 0:
                    raise RuntimeError(
                        f"PREBUILT_EVAL_WINDOW_EMPTY: prebuilt has 0 rows in eval window [{chunk_start}, {chunk_end}] "
                        f"(chunk={chunk_idx}, padding_start={actual_chunk_start})"
                    )

                # IMPORTANT: avoid pandas ambiguity where "ts" is both an index level and a column label.
                # Always join on plain columns with a RangeIndex.
                # TRUTH-only timing: t_join_s
                if is_truth_or_smoke_worker:
                    t_join_start = time.time()
                
                raw_ts = chunk_df[["ts"]].reset_index(drop=True).copy()
                pre_ts = prebuilt_time_df[["ts"]].reset_index(drop=True).copy()
                joined = raw_ts.merge(pre_ts, on="ts", how="inner")
                n_raw = int(len(raw_ts))
                n_pre = int(len(pre_ts))
                n_join = int(len(joined))
                denom = min(n_raw, n_pre) if min(n_raw, n_pre) > 0 else 0
                join_ratio = (n_join / denom) if denom else 0.0
                
                if is_truth_or_smoke_worker:
                    t_join_s = time.time() - t_join_start
                
                # Update ledger after join
                skip_ledger["stage"] = "join_completed"
                skip_ledger["join_rows"] = n_join
                skip_ledger["join_ratio"] = join_ratio
                if len(joined) > 0 and "ts" in joined.columns:
                    skip_ledger["ts_min_join"] = str(joined["ts"].min())
                    skip_ledger["ts_max_join"] = str(joined["ts"].max())

                if n_join == 0:
                    raise RuntimeError(
                        f"RAW_PREBUILT_JOIN_EMPTY: join produced 0 rows (chunk={chunk_idx}). "
                        f"raw_n={n_raw} prebuilt_n={n_pre}"
                    )
                if is_truth_or_smoke_worker and join_ratio < 0.98:
                    raise RuntimeError(
                        f"RAW_PREBUILT_JOIN_RATIO_LOW: join_ratio={join_ratio:.4f} < 0.98 (chunk={chunk_idx}). "
                        f"raw_n={n_raw} prebuilt_n={n_pre} join_n={n_join}"
                    )

                join_metrics = {
                    "chunk_id": chunk_idx,
                    "actual_chunk_start": str(actual_chunk_start),
                    "eval_start_ts": str(chunk_start),
                    "eval_end_ts": str(chunk_end),
                    "raw_path": str(data_path),
                    "prebuilt_path": str(prebuilt_parquet_path_resolved),
                    "raw_rows_window": n_raw,
                    "prebuilt_rows_window": n_pre,
                    "joined_rows_window": n_join,
                    "join_ratio": join_ratio,
                    "raw_eval_rows": n_raw_eval,
                    "prebuilt_eval_rows": n_prebuilt_eval,
                    "raw_ts_min": str(chunk_df["ts"].min()),
                    "raw_ts_max": str(chunk_df["ts"].max()),
                    "prebuilt_ts_min": str(prebuilt_time_df["ts"].min()),
                    "prebuilt_ts_max": str(prebuilt_time_df["ts"].max()),
                }
                join_path = chunk_output_dir / "RAW_PREBUILT_JOIN.json"
                with open(join_path, "w") as f:
                    import json
                    json.dump(join_metrics, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                log.info(
                    f"[CHUNK {chunk_idx}] [RAW_PREBUILT_JOIN] raw={n_raw} prebuilt={n_pre} join={n_join} ratio={join_ratio:.4f}"
                )
            
            # A) Compute SSoT expected_bars early (before warmup calculation)
            # This will be updated after warmup is calculated, but we log it here for early visibility
            bars_total_coalesced = len(chunk_df)
            # Note: first_valid_eval_idx will be computed later by runner, but we log bars_total_coalesced now
            log.info(f"[SSOT_BARS] [CHUNK {chunk_idx}] bars_total_coalesced={bars_total_coalesced} (will compute first_valid_eval_idx after warmup)")
            
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
            # FIX: Handle case where "ts" column already exists (from prebuilt data)
            chunk_df_save = chunk_df.copy()
            if "ts" in chunk_df_save.columns:
                # "ts" column exists - rename it to "time" for _run_replay_impl
                chunk_df_save = chunk_df_save.rename(columns={"ts": "time"})
                # Drop index without converting to column (since we already have "time")
                if chunk_df_save.index.name == "ts":
                    chunk_df_save.index.name = None
                chunk_df_save = chunk_df_save.reset_index(drop=True)
            else:
                # No "ts" column - safe to reset index
                chunk_df_save = chunk_df_save.reset_index()
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
            os.environ["GX1_GATED_FUSION_ENABLED"] = "1"
            # Respect existing GX1_REQUIRE_XGB_CALIBRATION if set, otherwise default to "1"
            if "GX1_REQUIRE_XGB_CALIBRATION" not in os.environ:
                os.environ["GX1_REQUIRE_XGB_CALIBRATION"] = "1"
            os.environ["GX1_REPLAY_INCREMENTAL_FEATURES"] = "1"
            # TRUTH-only: Do NOT set GX1_REPLAY_NO_CSV=1 (trade journal requires CSV writes)
            # GX1_REPLAY_NO_CSV=1 is only for non-TRUTH runs where I/O is a bottleneck
            is_truth_or_smoke_worker = os.getenv("GX1_RUN_MODE", "").upper() in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"
            if not is_truth_or_smoke_worker:
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
                selftest_only = os.getenv("GX1_SELFTEST_ONLY", "0") == "1"
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
            
            # Set GX1_BUNDLE_DIR env var (priority: GX1_CANONICAL_BUNDLE_DIR > bundle_dir > policy)
            # This must be set BEFORE creating GX1DemoRunner, as it reads GX1_BUNDLE_DIR during init
            canonical_bundle_dir = os.getenv("GX1_CANONICAL_BUNDLE_DIR")
            if canonical_bundle_dir:
                # Use canonical bundle dir if set (TRUTH/SMOKE mode)
                os.environ["GX1_BUNDLE_DIR"] = canonical_bundle_dir
                log.info(f"[CHUNK {chunk_idx}] Set GX1_BUNDLE_DIR={canonical_bundle_dir} (from GX1_CANONICAL_BUNDLE_DIR)")
            elif bundle_dir:
                # Use bundle_dir from args/policy if canonical not set
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

            # ------------------------------------------------------------------------
            # TRUTH-only: Load OPEN_TRADES_SNAPSHOT.json from previous chunk (carry-forward)
            # ------------------------------------------------------------------------
            if is_truth_or_smoke_worker and chunk_idx > 0:
                try:
                    prev_chunk_dir = output_dir / f"chunk_{chunk_idx - 1}"
                    snapshot_path = prev_chunk_dir / "OPEN_TRADES_SNAPSHOT.json"
                    
                    if snapshot_path.exists():
                        with open(snapshot_path, "r") as f:
                            snapshot = json.load(f)
                        
                        open_trades_from_snapshot = snapshot.get("open_trades", [])
                        snapshot_count = len(open_trades_from_snapshot)
                        
                        if snapshot_count > 0:
                            log.info(f"[CHUNK {chunk_idx}] Loading {snapshot_count} open trades from chunk_{chunk_idx - 1}/OPEN_TRADES_SNAPSHOT.json")
                            
                            # Restore trades into runner.open_trades
                            # Note: LiveTrade is defined in oanda_demo_runner.py
                            from gx1.execution.oanda_demo_runner import LiveTrade
                            import pandas as pd
                            
                            restored_count = 0
                            for trade_snap in open_trades_from_snapshot:
                                try:
                                    entry_ts = None
                                    if trade_snap.get("entry_ts"):
                                        entry_ts = pd.Timestamp(trade_snap.get("entry_ts"), tz="UTC")
                                    
                                    # Create LiveTrade object from snapshot (minimal required fields)
                                    # trade_uid is required, fallback to trade_id if missing
                                    trade_uid = trade_snap.get("trade_uid") or trade_snap.get("trade_id", "")
                                    trade = LiveTrade(
                                        trade_id=trade_snap.get("trade_id", ""),
                                        trade_uid=trade_uid,
                                        entry_time=entry_ts,
                                        side=trade_snap.get("side", "long"),
                                        units=int(float(trade_snap.get("size", 0.0))),  # units must be int
                                        entry_price=float(trade_snap.get("entry_price", 0.0)),
                                        entry_bid=float(trade_snap.get("entry_bid", trade_snap.get("entry_price", 0.0))),
                                        entry_ask=float(trade_snap.get("entry_ask", trade_snap.get("entry_price", 0.0))),
                                        atr_bps=0.0,  # Default (not in snapshot)
                                        vol_bucket="unknown",  # Default (not in snapshot)
                                        entry_prob_long=0.5,  # Default (not in snapshot)
                                        entry_prob_short=0.5,  # Default (not in snapshot)
                                        dry_run=False,
                                    )
                                    if trade_snap.get("entry_score"):
                                        trade.entry_score = trade_snap.get("entry_score")
                                    if trade_snap.get("extra"):
                                        trade.extra = trade_snap.get("extra")
                                    
                                    # Ensure exit profile is set (required by runner)
                                    runner._ensure_exit_profile(trade, context="carry_forward")
                                    
                                    runner.open_trades.append(trade)
                                    restored_count += 1
                                except Exception as restore_error:
                                    log.warning(f"[CHUNK {chunk_idx}] Failed to restore trade {trade_snap.get('trade_id', 'unknown')}: {restore_error}", exc_info=True)
                            
                            log.info(f"[CHUNK {chunk_idx}] Restored {restored_count}/{snapshot_count} open trades from snapshot")
                        else:
                            log.debug(f"[CHUNK {chunk_idx}] Snapshot exists but has 0 open trades")
                    else:
                        # Check if previous chunk had open trades (invariant check)
                        prev_footer_path = prev_chunk_dir / "chunk_footer.json"
                        if prev_footer_path.exists():
                            try:
                                with open(prev_footer_path, "r") as f:
                                    prev_footer = json.load(f)
                                prev_open_end = prev_footer.get("open_trades_end_of_chunk", 0) or 0
                                if prev_open_end > 0:
                                    # Previous chunk had open trades but snapshot is missing - FATAL
                                    fatal_capsule = {
                                        "chunk_id": chunk_idx,
                                        "run_id": run_id,
                                        "fatal_reason": "OPEN_TRADES_SNAPSHOT_MISSING",
                                        "prev_chunk_id": chunk_idx - 1,
                                        "prev_open_trades_end": prev_open_end,
                                        "snapshot_path": str(snapshot_path),
                                        "message": f"Previous chunk (chunk_{chunk_idx - 1}) had {prev_open_end} open trades at end, but OPEN_TRADES_SNAPSHOT.json is missing. This violates carry-forward contract.",
                                        "timestamp": dt_now_iso(),
                                    }
                                    from gx1.utils.atomic_json import atomic_write_json
                                    fatal_path = chunk_output_dir / "OPEN_TRADES_SNAPSHOT_MISSING_FATAL.json"
                                    atomic_write_json(fatal_path, fatal_capsule)
                                    raise RuntimeError(
                                        f"[CHUNK {chunk_idx}] FATAL: OPEN_TRADES_SNAPSHOT_MISSING - "
                                        f"chunk_{chunk_idx - 1} had {prev_open_end} open trades but snapshot missing. See {fatal_path}"
                                    )
                            except Exception as check_error:
                                log.warning(f"[CHUNK {chunk_idx}] Failed to check previous chunk footer: {check_error}")
                except Exception as snapshot_load_error:
                    # Best-effort: do not break non-TRUTH runs
                    log.warning(f"[CHUNK {chunk_idx}] Failed to load OPEN_TRADES_SNAPSHOT.json: {snapshot_load_error}")
            
            # ------------------------------------------------------------------------
            # EXIT COVERAGE (TRUTH-only): initialize open_trades_start_of_chunk
            # ------------------------------------------------------------------------
            if is_truth_or_smoke_worker and runner and hasattr(runner, "exit_coverage") and isinstance(getattr(runner, "exit_coverage"), dict):
                try:
                    open_start = int(len(getattr(runner, "open_trades", []) or []))
                    runner.exit_coverage["open_trades_start_of_chunk"] = open_start
                    
                    # TRUTH-only invariant: if snapshot was loaded, open_start must match previous open_end
                    if chunk_idx > 0:
                        prev_chunk_dir = output_dir / f"chunk_{chunk_idx - 1}"
                        snapshot_path = prev_chunk_dir / "OPEN_TRADES_SNAPSHOT.json"
                        prev_footer_path = prev_chunk_dir / "chunk_footer.json"
                        
                        if snapshot_path.exists() and prev_footer_path.exists():
                            try:
                                with open(prev_footer_path, "r") as f:
                                    prev_footer = json.load(f)
                                prev_open_end = prev_footer.get("open_trades_end_of_chunk", 0) or 0
                                
                                # Allow for forced closes at boundary if explicitly enabled (accounting close)
                                # But if snapshot exists, we should have restored trades
                                if prev_open_end > 0 and open_start == 0:
                                    # This is suspicious - previous chunk had open trades, snapshot exists, but we restored 0
                                    log.warning(
                                        f"[CHUNK {chunk_idx}] WARNING: Previous chunk had {prev_open_end} open trades, "
                                        f"snapshot exists, but restored {open_start} trades. "
                                        f"This may indicate a restore failure."
                                    )
                                elif prev_open_end > 0 and open_start != prev_open_end:
                                    # Mismatch - log warning but don't fail (may be legitimate if trades were force-closed)
                                    log.info(
                                        f"[CHUNK {chunk_idx}] Previous chunk had {prev_open_end} open trades, "
                                        f"restored {open_start} trades (difference: {prev_open_end - open_start}). "
                                        f"This may be expected if trades were force-closed at boundary."
                                    )
                            except Exception as check_error:
                                log.warning(f"[CHUNK {chunk_idx}] Failed to verify snapshot state match: {check_error}")
                except Exception:
                    pass
            
            # CHUNK-LOCAL PADDING: Set eval window explicitly (only [chunk_start, chunk_end] counts for eval)
            if chunk_local_padding_days > 0:
                # Set eval window to original chunk boundaries (not actual_chunk_start)
                runner.replay_eval_start_ts = chunk_start
                runner.replay_eval_end_ts = chunk_end
                log.info(
                    f"[CHUNK {chunk_idx}] [CHUNK_LOCAL_PADDING] Set eval window: "
                    f"[{runner.replay_eval_start_ts}, {runner.replay_eval_end_ts}] "
                    f"(actual data range: [{actual_chunk_start}, {chunk_end}])"
                )
            
            # B) Prebuilt propagation (no env fallback): if runner does not have prebuilt_df, load from the explicit arg path.
            if prebuilt_parquet_path_resolved and (not hasattr(runner, "prebuilt_features_df") or runner.prebuilt_features_df is None):
                log.info(f"[CHUNK {chunk_idx}] [PREBUILT_PROPAGATE] Loading prebuilt_df into runner (explicit path)")
                try:
                    from gx1.execution.prebuilt_features_loader import PrebuiltFeaturesLoader
                    from gx1.utils.replay_mode import ReplayMode
                    prebuilt_loader_worker = PrebuiltFeaturesLoader(Path(prebuilt_parquet_path_resolved))
                    runner.prebuilt_features_loader = prebuilt_loader_worker
                    runner.prebuilt_features_df = prebuilt_loader_worker.df
                    runner.prebuilt_features_sha256 = prebuilt_loader_worker.sha256
                    runner.prebuilt_features_path_resolved = prebuilt_loader_worker.prebuilt_path_resolved
                    runner.prebuilt_schema_version = prebuilt_loader_worker.schema_version
                    runner.replay_mode_enum = ReplayMode.PREBUILT
                    runner.prebuilt_used = True
                    log.info(
                        f"[CHUNK {chunk_idx}] [PREBUILT_PROPAGATE] ✅ Prebuilt loaded: {len(runner.prebuilt_features_df):,} rows, {len(runner.prebuilt_features_df.columns)} columns"
                    )
                except Exception as prop_error:
                    fatal_msg = f"[PREBUILT_PROPAGATE_FAIL] Failed to load prebuilt in runner: {prop_error}"
                    log.error(fatal_msg, exc_info=True)
                    raise RuntimeError(fatal_msg) from prop_error
            
            # C) Update WORKER_BOOT.json with prebuilt status (after runner is created and prebuilt is loaded)
            has_prebuilt_df = hasattr(runner, "prebuilt_features_df") and runner.prebuilt_features_df is not None
            replay_mode_enum_value = None
            if hasattr(runner, "replay_mode_enum") and runner.replay_mode_enum:
                replay_mode_enum_value = runner.replay_mode_enum.value if hasattr(runner.replay_mode_enum, "value") else str(runner.replay_mode_enum)
            
            # Update worker_boot_payload with prebuilt status
            worker_boot_payload["has_prebuilt_df"] = has_prebuilt_df
            worker_boot_payload["replay_mode_enum"] = replay_mode_enum_value
            
            # B) entry_v10_enable_state: compute once per run/chunk from config/policy
            entry_v10_enabled = getattr(runner, "entry_v10_enabled", False) if runner else False
            entry_v10_ctx_enabled = getattr(runner, "entry_v10_ctx_enabled", False) if runner else False
            # V10 is enabled if either entry_v10_enabled or entry_v10_ctx_enabled is True
            entry_v10_enabled_final = entry_v10_enabled or entry_v10_ctx_enabled
            entry_v10_enabled_reason = None
            if not entry_v10_enabled_final:
                # Determine reason for disable
                if not entry_v10_enabled and not entry_v10_ctx_enabled:
                    entry_v10_enabled_reason = "BOTH_DISABLED"
                elif not entry_v10_enabled:
                    entry_v10_enabled_reason = "V10_DISABLED"
                elif not entry_v10_ctx_enabled:
                    entry_v10_enabled_reason = "V10_CTX_DISABLED"
            else:
                entry_v10_enabled_reason = "ENABLED"
            
            worker_boot_payload["entry_v10_enabled"] = entry_v10_enabled_final
            worker_boot_payload["entry_v10_enabled_reason"] = entry_v10_enabled_reason
            
            if has_prebuilt_df:
                worker_boot_payload["prebuilt_df_rows"] = len(runner.prebuilt_features_df)
                worker_boot_payload["prebuilt_df_cols"] = len(runner.prebuilt_features_df.columns)
                if len(runner.prebuilt_features_df) > 0:
                    worker_boot_payload["prebuilt_df_first_ts"] = str(runner.prebuilt_features_df.index[0])
                    worker_boot_payload["prebuilt_df_last_ts"] = str(runner.prebuilt_features_df.index[-1])
            else:
                worker_boot_payload["prebuilt_df_rows"] = 0
                worker_boot_payload["prebuilt_df_cols"] = 0
                worker_boot_payload["prebuilt_df_first_ts"] = None
                worker_boot_payload["prebuilt_df_last_ts"] = None
            
            # Hard fail if PREBUILT mode but no DF
            if replay_mode_enum_value == "PREBUILT" and not has_prebuilt_df:
                fatal_msg = "[PREBUILT_MODE_BUT_NO_DF] replay_mode_enum=PREBUILT but has_prebuilt_df=False"
                log.error(fatal_msg)
                worker_boot_payload["fatal_error"] = fatal_msg
                # Update WORKER_BOOT.json with error
                worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
                with open(worker_boot_path, "w") as f:
                    json_module.dump(worker_boot_payload, f, indent=2)
                fatal_capsule_path = chunk_output_dir / "PREBUILT_MODE_BUT_NO_DF.json"
                with open(fatal_capsule_path, "w") as f:
                    import json
                    json.dump({
                        "error_type": "PREBUILT_MODE_BUT_NO_DF",
                        "error_msg": fatal_msg,
                        "replay_mode_enum": replay_mode_enum_value,
                        "has_prebuilt_df": has_prebuilt_df,
                        "chunk_id": chunk_idx,
                        "timestamp": time.time(),
                    }, f, indent=2)
                raise RuntimeError(fatal_msg)
            
            # Write updated WORKER_BOOT.json
            worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
            with open(worker_boot_path, "w") as f:
                json_module.dump(worker_boot_payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            log.info(f"[CHUNK {chunk_idx}] [WORKER_BOOT] ✅ Updated WORKER_BOOT.json: has_prebuilt_df={has_prebuilt_df}, replay_mode_enum={replay_mode_enum_value}")
            
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
            os.environ["GX1_STOP_REQUESTED"] = "0"
            os.environ["GX1_CHECKPOINT_EVERY_BARS"] = str(CHECKPOINT_EVERY_BARS)
            
            # FASE 5: Quiet mode removed - all errors must be visible
            
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
            
            # Fast abort mode: set env var for abort-after-N-bars-per-chunk
            abort_after_n_bars = os.getenv("GX1_ABORT_AFTER_N_BARS_PER_CHUNK")
            if abort_after_n_bars:
                try:
                    abort_after_n_bars_int = int(abort_after_n_bars)
                    os.environ["GX1_ABORT_AFTER_N_BARS_PER_CHUNK"] = str(abort_after_n_bars_int)
                    log.info(f"[CHUNK {chunk_idx}] Fast abort mode: will stop after {abort_after_n_bars_int} bars")
                except ValueError:
                    log.warning(f"[CHUNK {chunk_idx}] Invalid GX1_ABORT_AFTER_N_BARS_PER_CHUNK: {abort_after_n_bars}")
            
            # STEG 4: Log replay mode and prebuilt status before starting replay (no env fallback)
            prebuilt_enabled_env = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
            has_prebuilt_df = hasattr(runner, "prebuilt_features_df") and runner.prebuilt_features_df is not None
            prebuilt_df_rows = len(runner.prebuilt_features_df) if has_prebuilt_df else 0
            prebuilt_df_cols = len(runner.prebuilt_features_df.columns) if has_prebuilt_df else 0
            prebuilt_path_logged = getattr(runner, "prebuilt_features_path_resolved", None)
            
            log.info(
                f"[CHUNK {chunk_idx}] [REPLAY_MODE] Before run_replay: "
                f"prebuilt_enabled_env={prebuilt_enabled_env}, "
                f"prebuilt_parquet_path={prebuilt_path_logged}, "
                f"has_prebuilt_df={has_prebuilt_df}, "
                f"prebuilt_df_rows={prebuilt_df_rows}, "
                f"prebuilt_df_cols={prebuilt_df_cols}"
            )
            
            # Run replay for this chunk
            # DEL 3: Checkpoint flush is handled inside bar loop via env var
            # Entry stage telemetry: track bars directly from runner (fallback to local if runner telemetry missing)
            # TRUTH-only timing: t_loop_s
            if is_truth_or_smoke_worker:
                t_loop_start = time.time()
            
            try:
                runner.run_replay(chunk_data_path_abs)
            except KeyboardInterrupt:
                log.warning(f"[CHUNK {chunk_idx}] Interrupted (KeyboardInterrupt)")
                status = "stopped"
                error = "Interrupted by KeyboardInterrupt"
            
            if is_truth_or_smoke_worker:
                t_loop_s = time.time() - t_loop_start
            
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
            
            # A) Define SSoT: expected_bars = len(coalesced_df) - first_valid_eval_idx
            # expected_bars_ssot represents the exact number of bars the loop will attempt to iterate
            # (ground truth from the actual coalesced dataframe, not an estimate)
            # NOTE: bars_iterated (from bars_seen) counts ALL loop iterations, including warmup skips
            # So expected_bars_ssot = bars_total_coalesced (loop iterates over ALL bars in chunk_df)
            bars_total_coalesced = len(chunk_df) if chunk_df is not None else 0
            first_valid_eval_idx = bars_skipped_warmup if bars_skipped_warmup is not None else 0
            
            # B) Log the three numbers early, once per chunk
            log.info(
                f"[SSOT_BARS] [CHUNK {chunk_idx}] "
                f"bars_total_coalesced={bars_total_coalesced} "
                f"first_valid_eval_idx={first_valid_eval_idx} "
                f"(expected_bars_ssot will be computed at check site)"
            )
            
            # Write to chunk_master_created.json (or update WORKER_BOOT.json if available)
            try:
                chunk_master_created_path = chunk_output_dir / "chunk_master_created.json"
                if chunk_master_created_path.exists():
                    import json
                    with open(chunk_master_created_path, "r") as f:
                        chunk_master_data = json.load(f)
                    chunk_master_data["ssot_bars"] = {
                        "bars_total_coalesced": bars_total_coalesced,
                        "first_valid_eval_idx": first_valid_eval_idx,
                        "expected_bars_ssot": expected_bars_ssot,
                    }
                    with open(chunk_master_created_path, "w") as f:
                        json.dump(chunk_master_data, f, indent=2)
            except Exception as e:
                log.warning(f"[SSOT_BARS] Failed to write to chunk_master_created.json: {e}")
            
            # CRITICAL FIX: Use canonical loop counters from runner (SSoT)
            # loop_iters_total = total loop iterations (must equal len(df))
            # loop_iters_post_warmup = iterations after first_valid_eval_idx
            # 
            # C) Check if loop completed all bars by comparing loop_iters_total to len(df)
            loop_iters_total = getattr(runner, "loop_iters_total", None)
            loop_iters_post_warmup = getattr(runner, "loop_iters_post_warmup", None)
            bars_total_coalesced_runner = getattr(runner, "bars_total_coalesced", None)
            first_valid_eval_idx_runner = getattr(runner, "first_valid_eval_idx_stored", None)
            
            # Fallback to legacy counters if new ones not available
            if loop_iters_total is None:
                loop_iters_total = getattr(runner, "bars_seen", None) or getattr(runner, "candles_iterated", 0)
            if bars_total_coalesced_runner is None:
                bars_total_coalesced_runner = len(chunk_df) if chunk_df is not None else 0
            if first_valid_eval_idx_runner is None:
                first_valid_eval_idx_runner = int(bars_skipped_warmup) if bars_skipped_warmup is not None else 0
            
            # Calculate expected values
            expected_loop_iters = bars_total_coalesced_runner
            expected_post_warmup = max(0, bars_total_coalesced_runner - first_valid_eval_idx_runner)
            
            # Get stop_reason from runner (single source of truth)
            stop_reason = getattr(runner, "last_stop_reason", "UNKNOWN")
            stop_exception = getattr(runner, "last_stop_exception", None)
            last_iterated_ts = getattr(runner, "last_iterated_ts", None)
            last_i = getattr(runner, "last_i", None)
            
            # D) Log canonical counters at check site
            log.info(
                f"[SSOT_BARS_CHECK] [CHUNK {chunk_idx}] "
                f"bars_total_coalesced={bars_total_coalesced_runner} "
                f"first_valid_eval_idx={first_valid_eval_idx_runner} "
                f"expected_loop_iters={expected_loop_iters} "
                f"expected_post_warmup={expected_post_warmup} "
                f"loop_iters_total={loop_iters_total} "
                f"loop_iters_post_warmup={loop_iters_post_warmup} "
                f"stop_reason={stop_reason} "
                f"last_i={last_i} "
                f"last_ts={last_iterated_ts}"
            )
            
            # C) Primary completion invariant: loop_iters_total == bars_total_coalesced
            if loop_iters_total is not None and loop_iters_total != expected_loop_iters:
                # D) Bulletproof capsule writer (no silent failure)
                # Get first/last timestamps
                first_ts = None
                last_ts = None
                if chunk_df is not None and len(chunk_df) > 0:
                    first_ts = str(chunk_df.index[0]) if hasattr(chunk_df.index, '__getitem__') else None
                    last_ts = str(chunk_df.index[-1]) if hasattr(chunk_df.index, '__getitem__') else None
                
                # Build payload with canonical counters
                import json
                from datetime import datetime
                early_stop_diff = {
                    "chunk_id": chunk_idx,
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    # Canonical counters (SSoT)
                    "loop_iters_total": loop_iters_total,
                    "expected_loop_iters": expected_loop_iters,
                    "loop_iters_post_warmup": loop_iters_post_warmup,
                    "expected_post_warmup": expected_post_warmup,
                    "bars_total_coalesced": bars_total_coalesced_runner,
                    "first_valid_eval_idx": first_valid_eval_idx_runner,
                    # Timestamps
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "last_iterated_ts": str(last_iterated_ts) if last_iterated_ts is not None else None,
                    "last_i": last_i,
                    # Stop reason
                    "stop_reason": stop_reason,
                    "stop_exception": stop_exception,
                    "diff": expected_loop_iters - loop_iters_total,
                    # Legacy counters for debugging
                    "legacy": {
                        "bars_seen": getattr(runner, "bars_seen", None),
                        "candles_iterated": getattr(runner, "candles_iterated", None),
                        "bars_processed": bars_processed,
                    },
                }
                
                # D) Layer 1: Try normal atomic JSON
                atomic_write_success = False
                atomic_write_error = None
                try:
                    from gx1.utils.atomic_json import atomic_write_json
                    early_stop_diff_path = chunk_output_dir / "EARLY_STOP_DIFF.json"
                    atomic_write_json(early_stop_diff, early_stop_diff_path)
                    atomic_write_success = True
                    log.error(f"[EARLY_STOP_DIFF] Wrote diff capsule to: {early_stop_diff_path}")
                except Exception as e:
                    atomic_write_error = str(e)
                    log.warning(f"[EARLY_STOP_DIFF] Failed to write atomic JSON: {e}")
                
                # D) Layer 2: Always write fallback txt (even if atomic succeeded, for redundancy)
                try:
                    fallback_path = chunk_output_dir / "EARLY_STOP_DIFF_FALLBACK.txt"
                    with open(fallback_path, "w") as f:
                        f.write("EARLY_STOP_DIFF (fallback format)\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(json.dumps(early_stop_diff, indent=2, default=str))
                        f.write("\n\n")
                        if atomic_write_error:
                            f.write(f"Atomic JSON write error: {atomic_write_error}\n")
                    log.error(f"[EARLY_STOP_DIFF] Wrote fallback txt to: {fallback_path}")
                except Exception as e:
                    log.error(f"[EARLY_STOP_DIFF] CRITICAL: Failed to write fallback txt: {e}")
                
                # D) Layer 3: Always write tiny marker (always succeeds)
                try:
                    marker_path = chunk_output_dir / "EARLY_STOP_MARKER.txt"
                    with open(marker_path, "w") as f:
                        f.write(f"reason={stop_reason} loop_iters_total={loop_iters_total} expected={expected_loop_iters}\n")
                    log.error(f"[EARLY_STOP_DIFF] Wrote marker to: {marker_path}")
                except Exception as e:
                    log.error(f"[EARLY_STOP_DIFF] CRITICAL: Failed to write marker: {e}")
                
                # D) Append stderr tail if available (from subprocess worker)
                try:
                    worker_stderr_path = chunk_output_dir / "WORKER_STDERR.txt"
                    if worker_stderr_path.exists():
                        with open(worker_stderr_path, "r") as f:
                            stderr_lines = f.readlines()
                            stderr_tail = "".join(stderr_lines[-200:])  # Last 200 lines
                            early_stop_diff["stderr_tail"] = stderr_tail
                            # Update fallback if it exists
                            if fallback_path.exists():
                                with open(fallback_path, "a") as f:
                                    f.write("\n\nWORKER_STDERR_TAIL:\n")
                                    f.write("=" * 80 + "\n")
                                    f.write(stderr_tail)
                except Exception as e:
                    log.warning(f"[EARLY_STOP_DIFF] Failed to append stderr tail: {e}")
                
                # Hard-fail if not timeout/stop requested
                if not STOP_REQUESTED and status == "ok":
                    error = f"Early stop: loop_iters_total ({loop_iters_total}) != expected ({expected_loop_iters}), diff={expected_loop_iters - loop_iters_total}"
                    status = "failed"
                    log.error(
                        f"[CHUNK {chunk_idx}] FATAL: Early stop before end of subset. "
                        f"loop_iters_total={loop_iters_total}, expected={expected_loop_iters}, diff={expected_loop_iters - loop_iters_total}"
                    )
            else:
                log.info(f"[CHUNK {chunk_idx}] ✅ Loop completed all bars: loop_iters_total={loop_iters_total} == expected={expected_loop_iters}")
            
            # total_bars = loop_iters_total (total bars in subset, not bars_processed)
            bars_iterated = loop_iters_total if loop_iters_total is not None else bars_processed
            total_bars = bars_iterated
            
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
            # B1: True timers (best-effort; semantics-neutral)
            t_xgb_predict_sec = getattr(runner, "t_xgb_predict_sec", 0.0)
            t_transformer_forward_sec = getattr(runner, "t_transformer_forward_sec", 0.0)
            t_gates_policy_sec = getattr(runner, "t_gates_policy_sec", 0.0)
            t_replay_tags_sec = getattr(runner, "t_replay_tags_sec", 0.0)
            t_telemetry_sec = getattr(runner, "t_telemetry_sec", 0.0)
            # Step 4A: replay tagger sub-timers
            t_replay_tags_build_inputs_sec = getattr(runner, "t_replay_tags_build_inputs_sec", 0.0)
            t_replay_tags_rolling_sec = getattr(runner, "t_replay_tags_rolling_sec", 0.0)
            t_replay_tags_ewm_sec = getattr(runner, "t_replay_tags_ewm_sec", 0.0)
            t_replay_tags_rank_sec = getattr(runner, "t_replay_tags_rank_sec", 0.0)
            t_replay_tags_assign_sec = getattr(runner, "t_replay_tags_assign_sec", 0.0)
            t_io_total_sec = getattr(runner, "t_io_total_sec", 0.0)
            
            # D) Check prebuilt invariant: if prebuilt enabled, feature_time should be ~0
            prebuilt_used_env = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1"
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
                # Note: os is already imported at function start
        
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
            telemetry_required = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
            truth_telemetry = os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
            # In TRUTH/PREBUILT mode, enable telemetry by default (fail-safe)
            if not telemetry_required and truth_telemetry:
                telemetry_required = True
                log.info(f"[TELEMETRY] GX1_TRUTH_TELEMETRY=1 enabled, requiring telemetry")
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
                            os.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"] = "1"
                            os.environ["GX1_TELEMETRY_NO_ENTRY_REASON"] = no_entry_reason
                        
                        em.entry_feature_telemetry.write_all(chunk_output_dir)
                        telemetry_flushed = True
                        log.info(f"[FLUSH] [CHUNK {chunk_idx}] Entry feature telemetry flushed to {chunk_output_dir}")
                        # NOTE: XGB fingerprint summary is written by master after all chunks complete
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
                if "GX1_TELEMETRY_NO_ENTRY_EVALUATIONS" in os.environ:
                    del os.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"]
                if "GX1_TELEMETRY_NO_ENTRY_REASON" in os.environ:
                    del os.environ["GX1_TELEMETRY_NO_ENTRY_REASON"]
            
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
            # Note: os is already imported at function start
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
                    t_xgb_predict_sec_val = convert_to_json_serializable(t_xgb_predict_sec)
                    t_transformer_forward_sec_val = convert_to_json_serializable(t_transformer_forward_sec)
                    t_gates_policy_sec_val = convert_to_json_serializable(t_gates_policy_sec)
                    t_replay_tags_sec_val = convert_to_json_serializable(t_replay_tags_sec)
                    t_telemetry_sec_val = convert_to_json_serializable(t_telemetry_sec)
                    t_replay_tags_build_inputs_sec_val = convert_to_json_serializable(t_replay_tags_build_inputs_sec)
                    t_replay_tags_rolling_sec_val = convert_to_json_serializable(t_replay_tags_rolling_sec)
                    t_replay_tags_ewm_sec_val = convert_to_json_serializable(t_replay_tags_ewm_sec)
                    t_replay_tags_rank_sec_val = convert_to_json_serializable(t_replay_tags_rank_sec)
                    t_replay_tags_assign_sec_val = convert_to_json_serializable(t_replay_tags_assign_sec)
                    t_io_total_sec_val = convert_to_json_serializable(t_io_total_sec)
                    bars_per_sec_val = convert_to_json_serializable(bars_processed / wall_clock_sec if wall_clock_sec > 0 else 0.0)
                else:
                    feature_time_mean_ms_val = None
                    t_feature_build_total_sec_val = None
                    t_xgb_predict_sec_val = None
                    t_transformer_forward_sec_val = None
                    t_gates_policy_sec_val = None
                    t_replay_tags_sec_val = None
                    t_telemetry_sec_val = None
                    t_replay_tags_build_inputs_sec_val = None
                    t_replay_tags_rolling_sec_val = None
                    t_replay_tags_ewm_sec_val = None
                    t_replay_tags_rank_sec_val = None
                    t_replay_tags_assign_sec_val = None
                    t_io_total_sec_val = None
                    bars_per_sec_val = None
                
                # DEL 6: Get case collision resolution metadata (if available)
                case_collision_resolution_val = None
                if 'case_collision_resolution' in locals():
                    case_collision_resolution_val = convert_to_json_serializable(case_collision_resolution)
                
                # TRUTH-only timing breakdown
                if is_truth_or_smoke_worker:
                    t_total_s = time.time() - worker_start_time
                    # t_write_s will be calculated after footer is written
                
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
                    # TRUTH-only timing breakdown
                    "t_init_s": convert_to_json_serializable(t_init_s) if is_truth_or_smoke_worker else None,
                    "t_load_raw_s": convert_to_json_serializable(t_load_raw_s) if is_truth_or_smoke_worker else None,
                    "t_load_prebuilt_s": convert_to_json_serializable(t_load_prebuilt_s) if is_truth_or_smoke_worker else None,
                    "t_join_s": convert_to_json_serializable(t_join_s) if is_truth_or_smoke_worker else None,
                    "t_loop_s": convert_to_json_serializable(t_loop_s) if is_truth_or_smoke_worker else None,
                    "t_write_s": convert_to_json_serializable(t_write_s) if is_truth_or_smoke_worker else None,
                    "t_total_s": convert_to_json_serializable(t_total_s) if is_truth_or_smoke_worker else None,
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
                    "t_xgb_predict_sec": t_xgb_predict_sec_val,
                    "t_transformer_forward_sec": t_transformer_forward_sec_val,
                    "t_gates_policy_sec": t_gates_policy_sec_val,
                    "t_replay_tags_sec": t_replay_tags_sec_val,
                    "t_replay_tags_build_inputs_sec": t_replay_tags_build_inputs_sec_val,
                    "t_replay_tags_rolling_sec": t_replay_tags_rolling_sec_val,
                    "t_replay_tags_ewm_sec": t_replay_tags_ewm_sec_val,
                    "t_replay_tags_rank_sec": t_replay_tags_rank_sec_val,
                    "t_replay_tags_assign_sec": t_replay_tags_assign_sec_val,
                    "t_telemetry_sec": t_telemetry_sec_val,
                    "t_io_total_sec": t_io_total_sec_val,
                    "bars_processed": convert_to_json_serializable(bars_processed),
                    "start_ts": chunk_start.isoformat() if chunk_start else None,
                    "end_ts": chunk_end.isoformat() if chunk_end else None,
                    "worker_time_sec": float(time.time() - worker_start_time),
                    "pid": int(os.getpid()),
                    "dt_module_version": dt_module_version,  # CRITICAL: Version stamp
                    "timestamp": dt_now_iso(),
                }
                
                # D) Add prebuilt features info to chunk footer
                # Get prebuilt_used from runner (not env var, as runner validates it)
                prebuilt_used = False
                if runner and hasattr(runner, "prebuilt_used"):
                    prebuilt_used = runner.prebuilt_used
                
                chunk_footer["prebuilt_used"] = prebuilt_used

                # Step 4A: replay tagger fast-skip counters (TRUTH/SMOKE only)
                try:
                    if runner:
                        chunk_footer["replay_tags_fastskip_hits"] = int(getattr(runner, "replay_tags_fastskip_hits", 0) or 0)
                        counts = getattr(runner, "replay_tags_fastskip_reason_counts", None)
                        top_reason = None
                        if isinstance(counts, dict) and counts:
                            top_reason = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
                        chunk_footer["replay_tags_fastskip_reason"] = top_reason
                except Exception:
                    pass
                
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
                                "prebuilt_enabled": os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
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
                            "prebuilt_enabled": os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
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
                    # Use entry_v10_enable_state dict directly (guaranteed to have enabled set, never None)
                    if hasattr(telemetry, "entry_v10_enable_state") and telemetry.entry_v10_enable_state:
                        chunk_footer["entry_v10_enable_state"] = dict(telemetry.entry_v10_enable_state)
                    else:
                        # Fallback to legacy fields if dict not available
                        chunk_footer["entry_v10_enable_state"] = {
                            "enabled": telemetry.entry_v10_enabled,
                            "reason": telemetry.entry_v10_enabled_reason,
                            "enabled_true_count": telemetry.entry_v10_enabled_true_count,
                            "enabled_false_count": telemetry.entry_v10_enabled_false_count,
                            "reason_counts": dict(telemetry.entry_v10_enabled_reason_counts),
                            "source": "legacy",
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
                threshold_value_used = None
                threshold_source = None
                if runner and hasattr(runner, "entry_manager"):
                    em = runner.entry_manager
                    killchain_version = int(getattr(em, "killchain_version", 1))
                    for k in list(killchain_fields.keys()):
                        killchain_fields[k] = int(getattr(em, k.replace("killchain_", "killchain_"), 0))
                    killchain_block_reason_counts = getattr(em, "killchain_block_reason_counts", {}) or {}
                    # Extract threshold_value_used (SSoT for score gate)
                    threshold_value_used = getattr(em, "threshold_used", None)
                    # Determine threshold_source from policy
                    if hasattr(runner, "replay_mode") and runner.replay_mode:
                        replay_policy_cfg = runner.policy.get("entry_policy_sniper_v10_ctx", {}) or runner.policy.get("sniper_policy", {})
                        if replay_policy_cfg:
                            threshold_source = "entry_policy_sniper_v10_ctx" if "entry_policy_sniper_v10_ctx" in runner.policy else "sniper_policy"
                        else:
                            threshold_source = "default (SniperPolicyParams)"
                    else:
                        policy_sniper_cfg = runner.policy.get("entry_v9_policy_sniper", {})
                        if policy_sniper_cfg:
                            threshold_source = "entry_v9_policy_sniper"
                        else:
                            threshold_source = "default (SniperPolicyParams)"

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
                # Score gate threshold (SSoT)
                chunk_footer["threshold_used"] = threshold_value_used
                chunk_footer["threshold_source"] = threshold_source

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

                # Get policy_hash, bundle_sha, replay_mode_enum from run_identity or runner
                policy_hash = None
                bundle_sha = None
                replay_mode_enum = None
                try:
                    run_identity_path = chunk_output_dir / "run_header.json"
                    if run_identity_path.exists():
                        with open(run_identity_path, "r") as f:
                            run_identity_data = json.load(f)
                            policy_hash = run_identity_data.get("policy_hash")
                            bundle_sha = run_identity_data.get("bundle_sha256") or bundle_sha256
                            replay_mode_enum_str = run_identity_data.get("replay_mode_enum")
                            if replay_mode_enum_str:
                                # Try to parse replay_mode_enum from string
                                try:
                                    from gx1.execution.replay_mode import ReplayModeEnum
                                    replay_mode_enum = ReplayModeEnum.from_string(replay_mode_enum_str)
                                except Exception:
                                    replay_mode_enum = replay_mode_enum_str
                except Exception:
                    pass
                
                # Fallback: get from runner if available
                if bundle_sha is None and bundle_sha256:
                    bundle_sha = bundle_sha256
                if replay_mode_enum is None and runner:
                    replay_mode_enum = getattr(runner, "replay_mode_enum", None)
                
                # Build CounterContext (SSoT for all counters) - must be defined before KILLCHAIN_STAGE2_FAIL checks
                # Get entry_eval_entered_total (canonical eval universe)
                entry_eval_entered_total = 0
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        entry_eval_entered_total = em.entry_feature_telemetry.entry_eval_entered_total
                
                # Get bars_passed_hard_eligibility and other counters (may not be defined yet if prebuilt_used is False)
                bars_passed_hard_eligibility_ctx = 0
                bars_blocked_hard_eligibility_ctx = 0
                bars_passed_soft_eligibility_ctx = 0
                bars_blocked_soft_eligibility_ctx = 0
                bars_reaching_entry_stage_ctx = bars_reaching_entry_stage if bars_reaching_entry_stage is not None else 0
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        telemetry = em.entry_feature_telemetry
                        bars_passed_hard_eligibility_ctx = telemetry.bars_passed_hard_eligibility
                        bars_blocked_hard_eligibility_ctx = telemetry.bars_blocked_hard_eligibility
                        bars_passed_soft_eligibility_ctx = telemetry.bars_passed_soft_eligibility
                        bars_blocked_soft_eligibility_ctx = telemetry.bars_blocked_soft_eligibility
                
                # CounterContext: canonical eval universe (post-coalesce, post-warmup, post-eligibility)
                pregate_blocks_total_ctx = bars_skipped_pregate if bars_skipped_pregate is not None else 0
                counter_context = {
                    "bars_total_coalesced": len(chunk_df) if chunk_df is not None else 0,
                    "first_valid_eval_idx": bars_skipped_warmup if bars_skipped_warmup is not None else 0,
                    "bars_after_warmup": (len(chunk_df) - bars_skipped_warmup) if chunk_df is not None and bars_skipped_warmup is not None else 0,
                    "bars_passed_hard_eligibility": bars_passed_hard_eligibility_ctx,
                    "bars_blocked_hard_eligibility": bars_blocked_hard_eligibility_ctx,
                    "bars_passed_soft_eligibility": bars_passed_soft_eligibility_ctx,
                    "bars_blocked_soft_eligibility": bars_blocked_soft_eligibility_ctx,
                    "bars_reached_entry_stage": bars_reaching_entry_stage_ctx,
                    "eligibility_blocks_total": bars_blocked_hard_eligibility_ctx,
                    "pregate_blocks_total": pregate_blocks_total_ctx,
                    "lookup_attempts": 0,  # Will be updated below if prebuilt_used
                    "lookup_hits": 0,  # Will be updated below if prebuilt_used
                    "lookup_misses": 0,  # Will be updated below if prebuilt_used
                    "entry_eval_entered_total": entry_eval_entered_total,  # Canonical eval universe
                    "stage2_total": entry_eval_entered_total,  # SSoT: stage2_total = entry_eval_entered_total (same universe)
                    "stage2_pass": int(stage2.get("killchain_n_pass_score_gate", 0)),
                    "stage2_block": int(stage2.get("killchain_n_block_below_threshold", 0)),
                    "model_attempt_calls_total": entry_eval_entered_total,  # Same as entry_eval_entered_total
                }
                
                # C) Rewrite Stage2 invariant (vol guard stage) - use canonical stage counters
                # Get canonical stage counters from telemetry
                stage2_total_canonical = 0
                stage2_pass_canonical = 0
                stage2_block_canonical = 0
                stage2_early_return_canonical = 0
                stage2_block_reasons_canonical = {}
                stage2_early_return_reasons_canonical = {}
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        telemetry = em.entry_feature_telemetry
                        stage2_total_canonical = getattr(telemetry, "post_vol_guard_reached", 0)
                        stage2_pass_canonical = getattr(telemetry, "stage2_pass", 0)
                        stage2_block_canonical = getattr(telemetry, "stage2_block", 0)
                        stage2_early_return_canonical = getattr(telemetry, "stage2_early_return_count", 0)
                        stage2_block_reasons_canonical = dict(getattr(telemetry, "stage2_block_reasons", {}))
                        stage2_early_return_reasons_canonical = dict(getattr(telemetry, "stage2_early_return_reasons", {}))
                
                # Legacy counters (for comparison)
                after_vol = int(killchain_fields.get("killchain_n_after_vol_guard", 0))
                pass_score = int(stage2.get("killchain_n_pass_score_gate", 0))
                block_below = int(stage2.get("killchain_n_block_below_threshold", 0))
                
                # C) Canonical Stage2 invariant: stage2_total == stage2_pass + stage2_block + stage2_early_return
                if stage2_total_canonical > 0 and stage2_total_canonical != (stage2_pass_canonical + stage2_block_canonical + stage2_early_return_canonical):
                    # Write KILLCHAIN_SSoT_DIFF.json
                    _write_killchain_ssoT_diff(
                        chunk_output_dir, chunk_idx, run_id, policy_hash, bundle_sha, replay_mode_enum,
                        stage2_total=stage2_total_canonical, stage2_pass=stage2_pass_canonical, stage2_block=stage2_block_canonical,
                        stage2_early_return=stage2_early_return_canonical, stage2_block_reasons=stage2_block_reasons_canonical,
                        stage2_early_return_reasons=stage2_early_return_reasons_canonical,
                        stage3_total=None, stage3_pass=None, stage3_block=None, stage3_early_return=None,
                        stage3_block_reasons=None, stage3_early_return_reasons=None,
                        error_type="KILLCHAIN_STAGE2_FAIL",
                        error_msg=f"stage2_total={stage2_total_canonical} != stage2_pass={stage2_pass_canonical} + stage2_block={stage2_block_canonical} + stage2_early_return={stage2_early_return_canonical}",
                        chunk_df=chunk_df
                    )
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE2_FAIL] SSoT mismatch: "
                        f"stage2_total={stage2_total_canonical} != stage2_pass={stage2_pass_canonical} + stage2_block={stage2_block_canonical} + stage2_early_return={stage2_early_return_canonical}"
                    )
                
                # pass_score is the number of bars that passed score gate.
                # NOTE: Not all score-passing bars necessarily create trades (policy may choose not to trade).
                # Therefore we only enforce a *lower bound* invariant here:
                #   trade_created + explicit_post_score_blocks <= pass_score
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
                if (trade_created + post_blocks) > pass_score:
                    _write_ssoT_diff_capsule(
                        chunk_output_dir, chunk_idx, run_id, policy_hash, bundle_sha, replay_mode_enum,
                        counter_context, expected_lookup_attempts=None, actual_lookup_attempts=None,
                        stage2_total=pass_score, stage2_pass=trade_created, stage2_block=post_blocks,
                        error_type="KILLCHAIN_STAGE2_FAIL",
                        error_msg=f"trade_created+post_blocks={trade_created + post_blocks} > pass_score_gate={pass_score}",
                        chunk_df=chunk_df
                    )
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE2_FAIL] SSoT mismatch: "
                        f"trade_created+post_blocks={trade_created + post_blocks} > pass_score_gate={pass_score}"
                    )
                if pass_score < trade_created:
                    # Write diff capsule before raising
                    _write_ssoT_diff_capsule(
                        chunk_output_dir, chunk_idx, run_id, policy_hash, bundle_sha, replay_mode_enum,
                        counter_context, expected_lookup_attempts=None, actual_lookup_attempts=None,
                        stage2_total=pass_score, stage2_pass=trade_created, stage2_block=0,
                        error_type="KILLCHAIN_STAGE2_FAIL", error_msg=f"pass_score_gate={pass_score} < trade_created={trade_created}",
                        chunk_df=chunk_df
                    )
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

                # D) Rewrite Stage3 invariant (score gate stage) - use canonical stage counters
                # Get canonical stage counters from telemetry
                stage3_total_canonical = 0
                stage3_pass_canonical = 0
                stage3_block_canonical = 0
                stage3_early_return_canonical = 0
                stage3_block_reasons_canonical = {}
                stage3_early_return_reasons_canonical = {}
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        telemetry = em.entry_feature_telemetry
                        stage3_total_canonical = getattr(telemetry, "post_score_gate_reached", 0)
                        stage3_pass_canonical = getattr(telemetry, "stage3_pass", 0)
                        stage3_block_canonical = getattr(telemetry, "stage3_block", 0)
                        stage3_early_return_canonical = getattr(telemetry, "stage3_early_return_count", 0)
                        stage3_block_reasons_canonical = dict(getattr(telemetry, "stage3_block_reasons", {}))
                        stage3_early_return_reasons_canonical = dict(getattr(telemetry, "stage3_early_return_reasons", {}))
                
                # D) Canonical Stage3 invariant: stage3_total == stage3_pass + stage3_block + stage3_early_return
                #
                # IMPORTANT: This is telemetry-only and must not fail TRUTH runs (no trading semantics).
                # If this invariant breaks, write a diff capsule + mark footer fields, but continue.
                if stage3_total_canonical > 0 and stage3_total_canonical != (stage3_pass_canonical + stage3_block_canonical + stage3_early_return_canonical):
                    _write_killchain_ssoT_diff(
                        chunk_output_dir, chunk_idx, run_id, policy_hash, bundle_sha, replay_mode_enum,
                        stage2_total=None, stage2_pass=None, stage2_block=None, stage2_early_return=None,
                        stage2_block_reasons=None, stage2_early_return_reasons=None,
                        stage3_total=stage3_total_canonical, stage3_pass=stage3_pass_canonical, stage3_block=stage3_block_canonical,
                        stage3_early_return=stage3_early_return_canonical, stage3_block_reasons=stage3_block_reasons_canonical,
                        stage3_early_return_reasons=stage3_early_return_reasons_canonical,
                        error_type="KILLCHAIN_STAGE3_MISMATCH",
                        error_msg=f"stage3_total={stage3_total_canonical} != stage3_pass={stage3_pass_canonical} + stage3_block={stage3_block_canonical} + stage3_early_return={stage3_early_return_canonical}",
                        chunk_df=chunk_df
                    )
                    chunk_footer["killchain_stage3_ssot_ok"] = False
                    chunk_footer["killchain_stage3_ssot_mismatch"] = {
                        "stage3_total": int(stage3_total_canonical),
                        "stage3_pass": int(stage3_pass_canonical),
                        "stage3_block": int(stage3_block_canonical),
                        "stage3_early_return": int(stage3_early_return_canonical),
                    }
                    log.error(
                        "[KILLCHAIN_STAGE3_MISMATCH] stage3_total=%s != stage3_pass=%s + stage3_block=%s + stage3_early_return=%s (telemetry-only; continuing)",
                        stage3_total_canonical, stage3_pass_canonical, stage3_block_canonical, stage3_early_return_canonical
                    )
                else:
                    chunk_footer["killchain_stage3_ssot_ok"] = True
                
                # Also enforce: post_score_gate_reached <= post_vol_guard_reached
                if stage3_total_canonical > 0 and stage2_total_canonical > 0 and stage3_total_canonical > stage2_total_canonical:
                    raise RuntimeError(
                        "[KILLCHAIN_STAGE3_FAIL] Stage ordering violation: "
                        f"post_score_gate_reached={stage3_total_canonical} > post_vol_guard_reached={stage2_total_canonical}"
                    )
                
                # Legacy histogram SSoT (for backward compatibility, but not used for invariant)
                if int(sum(hist_counts)) != int(hist_total):
                    log.warning(
                        "[KILLCHAIN_STAGE3_LEGACY] Histogram mismatch (legacy, not used for invariant): "
                        f"sum(hist_counts)={int(sum(hist_counts))} != hist_total={int(hist_total)}"
                    )
                if int(hist_total) != int(after_vol):
                    log.warning(
                        "[KILLCHAIN_STAGE3_LEGACY] Population mismatch (legacy, not used for invariant): "
                        f"entry_score_hist_total={int(hist_total)} != killchain_n_after_vol_guard={int(after_vol)}"
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
                # C) Use canonical prelookup counter if available, otherwise fall back to legacy counter
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        telemetry = em.entry_feature_telemetry
                        post_soft_prelookup_reached_canonical = getattr(telemetry, "post_soft_prelookup_reached", 0)
                        if post_soft_prelookup_reached_canonical > 0:
                            bars_reaching_entry_stage = post_soft_prelookup_reached_canonical  # Use canonical prelookup counter
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
                
                # Update ledger with bar counters
                skip_ledger["stage"] = "bar_counters_computed"
                skip_ledger["candles_iterated"] = candles_iterated
                skip_ledger["reached_entry_stage"] = reached_entry_stage
                skip_ledger["processed"] = processed
                skip_ledger["bars_processed"] = bars_processed
                skip_ledger["warmup_bars_required"] = getattr(runner, "warmup_bars_required", None) if runner else None
                skip_ledger["warmup_bars_seen"] = warmup_skipped
                skip_ledger["skipped_breakdown"]["skipped_warmup"] = warmup_skipped
                skip_ledger["skipped_breakdown"]["skipped_pregate"] = pregate_skipped
                if runner:
                    skip_ledger["gating_counters"]["pregate_enabled"] = getattr(runner, "pregate_enabled", False)
                    skip_ledger["gating_counters"]["pregate_skips"] = getattr(runner, "pregate_skips", None)
                    skip_ledger["gating_counters"]["pregate_passes"] = getattr(runner, "pregate_passes", None)
                
                # Store in chunk_footer
                chunk_footer["bar_counters"] = bar_counters
                chunk_footer["bars_skipped"] = candles_iterated - processed
                chunk_footer["warmup_bars"] = warmup_skipped
                chunk_footer["eligibility_blocks"] = pregate_skipped
                
                # Hard invariant: skipped == warmup_skipped + pregate_skipped
                # skipped = candles_iterated - reached_entry_stage (bars that didn't reach entry stage)
                skipped = candles_iterated - reached_entry_stage
                expected_skipped = warmup_skipped + pregate_skipped
                skip_ledger["n_skipped_total"] = skipped
                skip_ledger["skipped_breakdown"]["skipped_other"] = skipped - warmup_skipped - pregate_skipped
                skip_ledger["skipped_breakdown"]["skipped_other_reason"] = "unaccounted_skip" if (skipped - warmup_skipped - pregate_skipped) > 0 else None
                
                if skipped != expected_skipped:
                    # Check panic mode (default: disabled for smokes)
                    panic_mode = os.getenv("GX1_PANIC_MODE", "0") == "1"
                    
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
                            "replay_mode": "PREBUILT" if os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1" else "UNKNOWN",
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
                        os._exit(1)
                    
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
                    prebuilt_path = runner.prebuilt_features_path_resolved if runner and hasattr(runner, "prebuilt_features_path_resolved") else None
                    chunk_footer["prebuilt_path"] = prebuilt_path
                    if runner and hasattr(runner, "prebuilt_features_sha256") and runner.prebuilt_features_sha256:
                        chunk_footer["features_file_sha256"] = runner.prebuilt_features_sha256
                    # Add bypass count (number of bars that used prebuilt features)
                    prebuilt_bypass_count = getattr(runner, "prebuilt_bypass_count", 0) if runner else 0
                    chunk_footer["prebuilt_bypass_count"] = prebuilt_bypass_count
                    
                    # Add lookup telemetry (SSoT counters)
                    # NOTE: Lookup accounting is stored on runner, not on prebuilt_loader
                    # Use runner-level counters (lookup happens in entry_manager, increments runner.lookup_attempts)
                    lookup_attempts = getattr(runner, "lookup_attempts", 0) if runner else 0
                    lookup_hits = getattr(runner, "lookup_hits", 0) if runner else 0
                    lookup_misses = getattr(runner, "lookup_misses", 0) if runner else 0
                    lookup_miss_details = getattr(runner, "lookup_miss_details", []) if runner else []
                    
                    # Fail-fast invariant check
                    if lookup_attempts > 0 and lookup_hits + lookup_misses != lookup_attempts:
                        raise RuntimeError(
                            f"[PREBUILT_LOOKUP_INVARIANT_FAIL] Lookup accounting invariant violated: "
                            f"hits={lookup_hits} + misses={lookup_misses} != attempts={lookup_attempts}. "
                            f"This should be impossible with atomic lookup accounting."
                        )
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
                    
                    # Update CounterContext with lookup counters (if prebuilt_used)
                    if 'counter_context' in locals():
                        counter_context["lookup_attempts"] = lookup_attempts
                        counter_context["lookup_hits"] = lookup_hits
                        counter_context["lookup_misses"] = lookup_misses
                    
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
                            # C) Option 1: Redefine bars_reaching_entry_stage = post_soft_prelookup_reached (canonical)
                            # This is the actual point where lookup would have happened
                            post_soft_prelookup_reached_telemetry = getattr(telemetry, "post_soft_prelookup_reached", 0)
                            if post_soft_prelookup_reached_telemetry > 0:
                                bars_reaching_entry_stage = post_soft_prelookup_reached_telemetry  # Use canonical prelookup counter
                    
                    # Store in chunk_footer for debugging
                    chunk_footer["bars_passed_hard_eligibility"] = bars_passed_hard_eligibility
                    chunk_footer["bars_blocked_hard_eligibility"] = bars_blocked_hard_eligibility
                    chunk_footer["bars_passed_soft_eligibility"] = bars_passed_soft_eligibility
                    chunk_footer["bars_blocked_soft_eligibility"] = bars_blocked_soft_eligibility
                    # Store prelookup counters (canonical)
                    if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                        em = runner.entry_manager
                        if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                            telemetry = em.entry_feature_telemetry
                            chunk_footer["post_soft_prelookup_reached"] = getattr(telemetry, "post_soft_prelookup_reached", 0)
                            chunk_footer["post_soft_early_return"] = getattr(telemetry, "post_soft_early_return", 0)
                            chunk_footer["post_soft_early_return_reasons"] = dict(getattr(telemetry, "post_soft_early_return_reasons", {}))
                    
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
                    
                    # Legacy: keep old formula for backward compatibility, but use canonical prelookup counter if available
                    eligibility_blocks_total = bars_blocked_hard_eligibility
                    pregate_blocks_total = bars_skipped_pregate if bars_skipped_pregate is not None else 0
                    expected_lookup_attempts = bars_reaching_entry_stage - eligibility_blocks_total - pregate_blocks_total
                    
                    # Phase-specific invariant (for backward compatibility and detailed error messages)
                    if lookup_phase == "before_pregate":
                        # If lookup happens before pregate, attempts should equal all bars after warmup
                        expected_attempts_phase = candles_iterated - warmup_skipped
                        if lookup_attempts != expected_attempts_phase:
                            # Write diff capsule before raising
                            _write_ssoT_diff_capsule(
                                chunk_output_dir, chunk_idx, run_id, policy_hash, bundle_sha, replay_mode_enum,
                                counter_context, expected_lookup_attempts=expected_attempts_phase, actual_lookup_attempts=lookup_attempts,
                                stage2_total=None, stage2_pass=None, stage2_block=None,
                                error_type="PREBUILT_LOOKUP_INVARIANT_FAIL", error_msg=f"lookup_phase='before_pregate': lookup_attempts={lookup_attempts} != expected={expected_attempts_phase}",
                                chunk_df=chunk_df
                            )
                            raise RuntimeError(
                                f"[PREBUILT_LOOKUP_INVARIANT_FAIL] lookup_phase='before_pregate': "
                                f"lookup_attempts={lookup_attempts} != candles_iterated={candles_iterated} - warmup_skipped={warmup_skipped} = {expected_attempts_phase}. "
                                f"This indicates lookup was not attempted for all bars after warmup."
                            )
                    elif lookup_phase == "after_soft_eligibility" or lookup_phase == "after_hard_eligibility":
                        # B) Use canonical prelookup counter for invariant
                        # Get post_soft_prelookup_reached from telemetry (canonical prelookup point)
                        post_soft_prelookup_reached = 0
                        post_soft_early_return = 0
                        post_soft_early_return_reasons = {}
                        if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                            em = runner.entry_manager
                            if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                                telemetry = em.entry_feature_telemetry
                                post_soft_prelookup_reached = getattr(telemetry, "post_soft_prelookup_reached", 0)
                                post_soft_early_return = getattr(telemetry, "post_soft_early_return", 0)
                                post_soft_early_return_reasons = dict(getattr(telemetry, "post_soft_early_return_reasons", {}))
                        
                        # Canonical invariant: expected_lookup_attempts = post_soft_prelookup_reached - pregate_blocks_total
                        # This is the exact point where lookup would have happened
                        expected_lookup_attempts_canonical = post_soft_prelookup_reached - pregate_blocks_total
                        
                        # Sanity check: post_soft_prelookup_reached should equal lookup_attempts + post_soft_early_return + (any blocks after soft)
                        # This prevents hidden silent paths
                        sanity_sum = lookup_attempts + post_soft_early_return
                        if post_soft_prelookup_reached > 0 and abs(sanity_sum - post_soft_prelookup_reached) > 10:  # Allow small rounding differences
                            log.warning(
                                f"[PREBUILT_LOOKUP_SANITY] post_soft_prelookup_reached={post_soft_prelookup_reached} != "
                                f"lookup_attempts={lookup_attempts} + post_soft_early_return={post_soft_early_return} = {sanity_sum}. "
                                f"This may indicate hidden paths."
                            )
                        
                        if lookup_attempts != expected_lookup_attempts_canonical:
                            # Write diff capsule before raising (include all relevant counters)
                            _write_ssoT_diff_capsule(
                                chunk_output_dir, chunk_idx, run_id, policy_hash, bundle_sha, replay_mode_enum,
                                counter_context, expected_lookup_attempts=expected_lookup_attempts_canonical, actual_lookup_attempts=lookup_attempts,
                                stage2_total=None, stage2_pass=None, stage2_block=None,
                                error_type="PREBUILT_LOOKUP_INVARIANT_FAIL", 
                                error_msg=f"lookup_phase='{lookup_phase}': lookup_attempts={lookup_attempts} != expected={expected_lookup_attempts_canonical}",
                                chunk_df=chunk_df,
                                # Add prelookup counters to diff capsule
                                post_soft_prelookup_reached=post_soft_prelookup_reached,
                                post_soft_early_return=post_soft_early_return,
                                post_soft_early_return_reasons=post_soft_early_return_reasons,
                                bars_reaching_entry_stage_legacy=bars_reaching_entry_stage  # Legacy counter for comparison
                            )
                            raise RuntimeError(
                                f"[PREBUILT_LOOKUP_INVARIANT_FAIL] lookup_phase='{lookup_phase}': "
                                f"lookup_attempts={lookup_attempts} != expected_lookup_attempts={expected_lookup_attempts_canonical}. "
                                f"post_soft_prelookup_reached={post_soft_prelookup_reached}, "
                                f"post_soft_early_return={post_soft_early_return}, "
                                f"pregate_skipped={pregate_blocks_total}, "
                                f"warmup_skipped={warmup_skipped}. "
                                f"This indicates lookup was not attempted for all bars that reached the prelookup point."
                            )
                    elif lookup_phase == "after_pregate":
                        # Legacy phase name - treat as after_hard_eligibility
                        if lookup_attempts != expected_lookup_attempts:
                            # Write diff capsule before raising
                            _write_ssoT_diff_capsule(
                                chunk_output_dir, chunk_idx, run_id, policy_hash, bundle_sha, replay_mode_enum,
                                counter_context, expected_lookup_attempts=expected_lookup_attempts, actual_lookup_attempts=lookup_attempts,
                                stage2_total=None, stage2_pass=None, stage2_block=None,
                                error_type="PREBUILT_LOOKUP_INVARIANT_FAIL", error_msg=f"lookup_phase='after_pregate' (legacy): lookup_attempts={lookup_attempts} != expected={expected_lookup_attempts}",
                                chunk_df=chunk_df
                            )
                            raise RuntimeError(
                                f"[PREBUILT_LOOKUP_INVARIANT_FAIL] lookup_phase='after_pregate' (legacy): "
                                f"lookup_attempts={lookup_attempts} != expected_lookup_attempts={expected_lookup_attempts}. "
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
                
                # CHUNK-LOCAL PADDING: Add warmup ledger to chunk footer
                if chunk_local_padding_days > 0:
                    # Get warmup counters from runner
                    warmup_required_bars = getattr(runner, "warmup_bars", 288) if runner else 288
                    warmup_seen_bars = bars_seen if bars_seen is not None else 0
                    warmup_skipped_total = bars_skipped_warmup if bars_skipped_warmup is not None else 0
                    warmup_completed_ts = None
                    if runner and hasattr(runner, "first_valid_eval_idx_stored") and runner.first_valid_eval_idx_stored is not None:
                        if chunk_df is not None and len(chunk_df) > runner.first_valid_eval_idx_stored:
                            warmup_completed_ts = chunk_df.index[runner.first_valid_eval_idx_stored].isoformat()
                    
                    chunk_footer["warmup_ledger"] = {
                        "chunk_local_padding_days": chunk_local_padding_days,
                        "actual_replay_start_ts": actual_chunk_start.isoformat() if 'actual_chunk_start' in locals() else None,
                        "eval_start_ts": eval_start_ts.isoformat() if 'eval_start_ts' in locals() else chunk_start.isoformat(),
                        "eval_end_ts": eval_end_ts.isoformat() if 'eval_end_ts' in locals() else chunk_end.isoformat(),
                        "warmup_required_bars": warmup_required_bars,
                        "warmup_seen_bars": warmup_seen_bars,
                        "warmup_skipped_total": warmup_skipped_total,
                        "warmup_completed_ts": warmup_completed_ts,
                        "bars_processed_total": bars_processed,
                    }
                else:
                    chunk_footer["warmup_ledger"] = None

                # ------------------------------------------------------------------------
                # EXIT COVERAGE (TRUTH-only): export exit counters + write proof artifact
                # ------------------------------------------------------------------------
                if is_truth_or_smoke_worker and runner and hasattr(runner, "exit_coverage") and isinstance(getattr(runner, "exit_coverage"), dict):
                    try:
                        exit_cov = runner.exit_coverage

                        # Open trades end-of-chunk
                        open_end = int(len(getattr(runner, "open_trades", []) or []))
                        exit_cov["open_trades_end_of_chunk"] = open_end

                        # Export CSV exit row count if available
                        exit_rows_written = 0
                        tj = getattr(runner, "trade_journal", None)
                        if tj is not None and hasattr(tj, "event_row_counts"):
                            try:
                                exit_rows_written = int(getattr(tj, "event_row_counts", {}).get("EXIT", 0) or 0)
                            except Exception:
                                exit_rows_written = 0
                        exit_cov["exit_event_rows_written"] = exit_rows_written

                        # Derived count: open_to_closed_within_chunk (best-effort)
                        # Use exit_summary_logged as closed count signal (replay accounting)
                        try:
                            exit_cov["open_to_closed_within_chunk"] = int(exit_cov.get("exit_summary_logged", 0) or 0)
                        except Exception:
                            pass

                        # closed_trades_in_chunk: trades that closed within this chunk
                        closed_trades_in_chunk = int(exit_cov.get("exit_summary_logged", 0) or 0)
                        exit_cov["closed_trades_in_chunk"] = closed_trades_in_chunk

                        # Copy counters into footer (flat keys as requested)
                        for k in [
                            "exit_attempts_total",
                            "exit_request_close_called",
                            "exit_request_close_accepted",
                            "exit_summary_logged",
                            "exit_event_rows_written",
                            "force_close_attempts_replay_end",
                            "force_close_logged_replay_end",
                            "force_close_attempts_replay_eof",
                            "force_close_logged_replay_eof",
                            "open_trades_start_of_chunk",
                            "open_trades_end_of_chunk",
                            "open_to_closed_within_chunk",
                            "closed_trades_in_chunk",
                            "replay_end_or_eof_triggered",
                            "accounting_close_enabled",
                        ]:
                            chunk_footer[k] = exit_cov.get(k)

                        # TRUTH-only proof artifact per chunk
                        proof_payload = {
                            "chunk_id": chunk_idx,
                            "run_id": run_id,
                            "open_trades_start": exit_cov.get("open_trades_start_of_chunk"),
                            "open_trades_end": exit_cov.get("open_trades_end_of_chunk"),
                            "exit_attempts_total": exit_cov.get("exit_attempts_total", 0),
                            "exit_request_close_called": exit_cov.get("exit_request_close_called", 0),
                            "exit_request_close_accepted": exit_cov.get("exit_request_close_accepted", 0),
                            "exit_summary_logged": exit_cov.get("exit_summary_logged", 0),
                            "exit_event_rows_written": exit_cov.get("exit_event_rows_written", 0),
                            "force_close_attempts_replay_end": exit_cov.get("force_close_attempts_replay_end", 0),
                            "force_close_logged_replay_end": exit_cov.get("force_close_logged_replay_end", 0),
                            "force_close_attempts_replay_eof": exit_cov.get("force_close_attempts_replay_eof", 0),
                            "force_close_logged_replay_eof": exit_cov.get("force_close_logged_replay_eof", 0),
                            "open_to_closed_within_chunk": exit_cov.get("open_to_closed_within_chunk", 0),
                            "last_5_exit_reasons": list(exit_cov.get("last_5_exit_reasons") or []),
                            "last_5_trade_ids_closed": list(exit_cov.get("last_5_trade_ids_closed") or []),
                            "replay_end_or_eof_triggered": bool(exit_cov.get("replay_end_or_eof_triggered", False)),
                            "accounting_close_enabled": bool(exit_cov.get("accounting_close_enabled", False)),
                            "timestamp": dt_now_iso(),
                        }

                        from gx1.utils.atomic_json import atomic_write_json
                        proof_path = chunk_output_dir / "EXIT_COVERAGE_PROOF.json"
                        atomic_write_json(proof_path, proof_payload)

                        # ------------------------------------------------------------
                        # TRUTH invariant (C1): accepted close MUST be journaled
                        #
                        # If exit_request_close_accepted > 0 in this chunk, then we
                        # require evidence of journaling:
                        #   - exit_event_rows_written > 0  OR
                        #   - exit_summary_logged > 0
                        # Otherwise: hard-fail with capsule(s).
                        # ------------------------------------------------------------
                        accepted = int(proof_payload.get("exit_request_close_accepted", 0) or 0)
                        summary_logged = int(proof_payload.get("exit_summary_logged", 0) or 0)
                        event_rows = int(proof_payload.get("exit_event_rows_written", 0) or 0)
                        truth_ok = True
                        truth_fail_reason = None
                        if accepted > 0 and (summary_logged <= 0 and event_rows <= 0):
                            truth_ok = False
                            truth_fail_reason = "EXIT_ACCEPTED_BUT_NOT_JOURNALED"

                            # Best-effort: include an example trade id (first recent close)
                            example_trade_id = None
                            try:
                                tids = list(exit_cov.get("last_5_trade_ids_closed") or [])
                                if tids:
                                    example_trade_id = str(tids[0])
                            except Exception:
                                example_trade_id = None

                            capsule = {
                                "fatal_tag": "[TRUTH_FAIL] EXIT_ACCEPTED_BUT_NOT_JOURNALED",
                                "chunk_id": chunk_idx,
                                "run_id": run_id,
                                "counters": {
                                    "exit_request_close_accepted": accepted,
                                    "exit_summary_logged": summary_logged,
                                    "exit_event_rows_written": event_rows,
                                },
                                "paths": {
                                    "chunk_output_dir": str(chunk_output_dir),
                                    "exit_coverage_proof": str(proof_path),
                                    "trade_journal_dir": str((chunk_output_dir / "trade_journal").resolve()),
                                    "trade_journal_jsonl": str((chunk_output_dir / "trade_journal" / "trade_journal.jsonl").resolve()),
                                    "trade_journal_index_csv": str((chunk_output_dir / "trade_journal" / "trade_journal_index.csv").resolve()),
                                },
                                "example_trade_id": example_trade_id,
                                "timestamp": dt_now_iso(),
                            }
                            capsule_path_chunk = chunk_output_dir / "TRUTH_EXIT_JOURNAL_FAIL_CAPSULE.json"
                            capsule_path_root = args.output_dir / "TRUTH_EXIT_JOURNAL_FAIL_CAPSULE.json"
                            atomic_write_json(capsule_path_chunk, capsule)
                            atomic_write_json(capsule_path_root, capsule)

                        exit_cov["truth_exit_journal_ok"] = truth_ok
                        exit_cov["truth_exit_journal_fail_reason"] = truth_fail_reason
                        chunk_footer["truth_exit_journal_ok"] = truth_ok
                        chunk_footer["truth_exit_journal_fail_reason"] = truth_fail_reason

                        if not truth_ok:
                            raise RuntimeError(
                                f"[TRUTH_FAIL] EXIT_ACCEPTED_BUT_NOT_JOURNALED: "
                                f"accepted={accepted} exit_summary_logged={summary_logged} exit_event_rows_written={event_rows}. "
                                f"See {capsule_path_chunk}"
                            )

                        # TRUTH-only invariant: if replay end/eof triggered and open trades remain, we must have logged force-close
                        if proof_payload["replay_end_or_eof_triggered"] and open_end > 0:
                            force_logged = int(proof_payload.get("force_close_logged_replay_end", 0) or 0) + int(proof_payload.get("force_close_logged_replay_eof", 0) or 0)
                            if force_logged == 0:
                                fatal_capsule = {
                                    "chunk_id": chunk_idx,
                                    "run_id": run_id,
                                    "fatal_reason": "EXIT_FORCE_CLOSE_LOGGING",
                                    "open_trades_end_of_chunk": open_end,
                                    "proof_path": str(proof_path),
                                    "message": "REPLAY_END/EOF was triggered with open trades remaining, but force_close_logged_* == 0. This violates exit accounting contract.",
                                    "timestamp": dt_now_iso(),
                                }
                                fatal_path = chunk_output_dir / "EXIT_FORCE_CLOSE_LOGGING_FATAL.json"
                                atomic_write_json(fatal_path, fatal_capsule)
                                raise RuntimeError(
                                    f"[CHUNK {chunk_idx}] FATAL: EXIT_FORCE_CLOSE_LOGGING - open_trades_end={open_end} but force_close_logged==0. See {fatal_path}"
                                )
                    except Exception as exit_cov_error:
                        # Best-effort: do not break non-TRUTH runs.
                        # In TRUTH/SMOKE, re-raise if this is an explicit TRUTH_FAIL capsule-worthy invariant.
                        msg = str(exit_cov_error)
                        if "[TRUTH_FAIL]" in msg:
                            raise
                        log.warning(f"[CHUNK {chunk_idx}] EXIT_COVERAGE export failed: {exit_cov_error}")
                
                # Convert all values to JSON-serializable types
                chunk_footer = convert_to_json_serializable(chunk_footer)
                
                chunk_footer_path = chunk_output_dir / "chunk_footer.json"
                # ENTRY FEATURE TELEMETRY: Write telemetry files before chunk_footer (REQUIRED in replay mode)
                telemetry_written = False
                telemetry_required = os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1"
                
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
                    os.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"] = "1"
                    os.environ["GX1_TELEMETRY_NO_ENTRY_REASON"] = no_entry_reason
                    log.info(
                        f"[CHUNK {chunk_idx}] No entry evaluations occurred "
                        f"(bars_processed={bars_processed}, bars_reaching_entry_stage={bars_reaching_entry_stage}, reason={no_entry_reason})"
                    )
                
                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                        # Write WORKER_END.json FIRST (atomic, always written, even if write_all fails)
                        try:
                            log.info(f"[CHUNK {chunk_idx}] Writing WORKER_END.json...")
                            em.entry_feature_telemetry.write_worker_end(chunk_output_dir, chunk_idx, os.getpid())
                            log.info(f"[CHUNK {chunk_idx}] ✅ WORKER_END.json written successfully")
                        except Exception as worker_end_error:
                            log.error(f"[CHUNK {chunk_idx}] Failed to write WORKER_END.json: {worker_end_error}", exc_info=True)
                            # Continue anyway - try to write other telemetry
                        
                        try:
                            em.entry_feature_telemetry.write_all(chunk_output_dir)
                            telemetry_written = True
                            log.info(f"[CHUNK {chunk_idx}] Entry feature telemetry written to {chunk_output_dir}")
                            # NOTE: XGB fingerprint summary is written by master after all chunks complete
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
                    else:
                        log.warning(f"[CHUNK {chunk_idx}] entry_feature_telemetry not available (hasattr={hasattr(em, 'entry_feature_telemetry') if em else False}, is_none={em.entry_feature_telemetry is None if hasattr(em, 'entry_feature_telemetry') else 'N/A'})")
                elif telemetry_required:
                    raise RuntimeError(
                        f"[CHUNK {chunk_idx}] FATAL: runner or entry_manager not available "
                        f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Cannot write telemetry."
                    )
                else:
                    log.warning(f"[CHUNK {chunk_idx}] runner or entry_manager not available (runner={runner is not None}, has_entry_manager={hasattr(runner, 'entry_manager') if runner else False}, entry_manager={runner.entry_manager is not None if runner and hasattr(runner, 'entry_manager') else 'N/A'})")
                
                # Clean up env vars
                if "GX1_TELEMETRY_NO_ENTRY_EVALUATIONS" in os.environ:
                    del os.environ["GX1_TELEMETRY_NO_ENTRY_EVALUATIONS"]
                if "GX1_TELEMETRY_NO_ENTRY_REASON" in os.environ:
                    del os.environ["GX1_TELEMETRY_NO_ENTRY_REASON"]
                
                # Fail-fast validation if telemetry is required
                if telemetry_required and bars_processed > 0:
                    entry_features_path = chunk_output_dir / "ENTRY_FEATURES_USED.json"
                    worker_end_path = chunk_output_dir / "WORKER_END.json"
                    
                    # Check WORKER_END.json exists (atomic, always written)
                    if not worker_end_path.exists():
                        raise RuntimeError(
                            f"[CHUNK {chunk_idx}] FATAL: WORKER_END.json not found after chunk processing "
                            f"(GX1_REQUIRE_ENTRY_TELEMETRY=1, bars_processed={bars_processed}). "
                            f"This file must always be written. Check chunk_output_dir: {chunk_output_dir}"
                        )
                    
                    # Check ENTRY_FEATURES_USED.json exists
                    if not entry_features_path.exists():
                        raise RuntimeError(
                            f"[CHUNK {chunk_idx}] FATAL: ENTRY_FEATURES_USED.json not found after write_all "
                            f"(GX1_REQUIRE_ENTRY_TELEMETRY=1, bars_processed={bars_processed}). "
                            f"Telemetry must be written for A/B tests."
                        )
                    
                    # Load WORKER_END.json and verify telemetry_written flag
                    try:
                        import json
                        with open(worker_end_path) as f:
                            worker_end = json.load(f)
                        if not worker_end.get("telemetry_written", False):
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: WORKER_END.json indicates telemetry_written=False "
                                f"(GX1_REQUIRE_ENTRY_TELEMETRY=1). Error: {worker_end.get('error', 'unknown')}"
                            )
                        
                        # Hard rule: if bars_after_warmup > 500 but bars_reached_entry_stage==0 => FATAL
                        bars_after_warmup = worker_end.get("bars_after_warmup", 0)
                        bars_reached_entry_stage = worker_end.get("bars_reached_entry_stage", 0)
                        if bars_after_warmup > 500 and bars_reached_entry_stage == 0:
                            entry_eval_attempts = worker_end.get("entry_eval_attempts", 0)
                            entry_block_reasons = worker_end.get("entry_block_reasons", {})
                            top_reasons = sorted(entry_block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: bars_after_warmup={bars_after_warmup} > 500 but "
                                f"bars_reached_entry_stage={bars_reached_entry_stage} == 0. "
                                f"entry_eval_attempts={entry_eval_attempts}. "
                                f"Top 5 block reasons: {top_reasons}"
                            )
                    except json.JSONDecodeError as e:
                        raise RuntimeError(
                            f"[CHUNK {chunk_idx}] FATAL: Failed to parse WORKER_END.json: {e}"
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
                        
                        # B) entry_v10_enable_state: make it explicit and immutable per run
                        # Get expected enable state from WORKER_BOOT.json (computed once per run/chunk)
                        worker_boot_path = chunk_output_dir / "WORKER_BOOT.json"
                        expected_v10_enabled = None
                        expected_v10_enabled_reason = None
                        if worker_boot_path.exists():
                            try:
                                with open(worker_boot_path, "r") as f:
                                    worker_boot_data = json.load(f)
                                    expected_v10_enabled = worker_boot_data.get("entry_v10_enabled")
                                    expected_v10_enabled_reason = worker_boot_data.get("entry_v10_enabled_reason")
                            except Exception:
                                pass
                        
                        # C) Fix invariant to read the correct key
                        # Get actual enable state from telemetry
                        entry_v10_enable_state = telemetry_data.get("entry_v10_enable_state", {})
                        state = entry_v10_enable_state if isinstance(entry_v10_enable_state, dict) else {}
                        v10_enabled = state.get("enabled", None)
                        v10_enabled_reason = state.get("reason", None)
                        v10_enabled_source = state.get("source", None)
                        v10_enabled_true_count = state.get("enabled_true_count", 0)
                        v10_enabled_false_count = state.get("enabled_false_count", 0)
                        v10_enabled_reason_counts = state.get("reason_counts", {})
                        
                        # C) If enabled is None -> FATAL with ENABLE_STATE_MISSING.json
                        if v10_enabled is None and bars_reaching_entry_stage > 0:
                            # Write ENABLE_STATE_MISSING.json capsule
                            try:
                                from gx1.utils.atomic_json import atomic_write_json
                                from datetime import datetime
                                enable_state_missing_capsule = {
                                    "chunk_id": chunk_idx,
                                    "run_id": run_id,
                                    "timestamp": datetime.now().isoformat(),
                                    "error_type": "ENABLE_STATE_MISSING",
                                    "error_message": f"entry_v10_enable_state.enabled is None but bars_reaching_entry_stage={bars_reaching_entry_stage} > 0",
                                    "state_dict": state,
                                    "enabled_true_count": v10_enabled_true_count,
                                    "enabled_false_count": v10_enabled_false_count,
                                    "after_v10_enable_check_count": 0,  # Will be updated if available
                                    "bars_reaching_entry_stage": bars_reaching_entry_stage,
                                    "policy_hash": policy_hash,
                                    "bundle_sha": bundle_sha,
                                    "replay_mode_enum": str(replay_mode_enum) if replay_mode_enum else None,
                                }
                                # Get after_v10_enable_check_count if available
                                if runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                                    em = runner.entry_manager
                                    if hasattr(em, "entry_feature_telemetry") and em.entry_feature_telemetry:
                                        telemetry = em.entry_feature_telemetry
                                        enable_state_missing_capsule["after_v10_enable_check_count"] = getattr(telemetry, "after_v10_enable_check_count", 0)
                                
                                enable_state_missing_path = chunk_output_dir / "ENABLE_STATE_MISSING.json"
                                atomic_write_json(enable_state_missing_capsule, enable_state_missing_path)
                                log.error(f"[ENABLE_STATE_MISSING] Wrote capsule to: {enable_state_missing_path}")
                            except Exception as e:
                                log.warning(f"[ENABLE_STATE_MISSING] Failed to write capsule: {e}")
                            
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state.enabled is None but bars_reaching_entry_stage={bars_reaching_entry_stage} > 0. "
                                f"This indicates enable state telemetry was not collected. "
                                f"entry_v10_enable_state={entry_v10_enable_state}, state_dict={state}"
                            )
                        
                        # B) Hard rule: If replay_mode==PREBUILT and entry_v10 is expected enabled -> FATAL if enable_state false
                        replay_mode_enum_value = None
                        if runner and hasattr(runner, "replay_mode_enum") and runner.replay_mode_enum:
                            replay_mode_enum_value = runner.replay_mode_enum.value if hasattr(runner.replay_mode_enum, "value") else str(runner.replay_mode_enum)
                        
                        if (replay_mode_enum_value == "PREBUILT" and 
                            expected_v10_enabled is True and 
                            v10_enabled is False):
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state mismatch. "
                                f"Expected enabled=True (from WORKER_BOOT.json) but actual enabled=False. "
                                f"This indicates enable state changed during runtime (not allowed). "
                                f"expected_v10_enabled={expected_v10_enabled}, actual_v10_enabled={v10_enabled}, "
                                f"replay_mode_enum={replay_mode_enum_value}"
                            )
                        
                        # INVARIANT 1: Enable state must be recorded (not None/unknown) - now guaranteed by record_v10_enable_state
                        # This check is now redundant but kept for safety
                        if v10_enabled is None and bars_reaching_entry_stage > 0:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state.enabled is None but bars_reaching_entry_stage={bars_reaching_entry_stage} > 0. "
                                f"This should never happen after record_v10_enable_state fix. "
                                f"entry_v10_enable_state={entry_v10_enable_state}, state_dict={state}"
                            )
                        
                        # INVARIANT 2: If V10 is disabled, reason must be non-empty
                        if v10_enabled is False and not v10_enabled_reason:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state.enabled=False but reason is empty. "
                                f"V10 disable reason must be recorded. "
                                f"entry_v10_enable_state={entry_v10_enable_state}"
                            )
                        
                        # INVARIANT 3: Runtime state must match expected state (immutable per run)
                        if expected_v10_enabled is not None and v10_enabled != expected_v10_enabled:
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: entry_v10_enable_state changed during runtime. "
                                f"Expected enabled={expected_v10_enabled} (from WORKER_BOOT.json) but actual enabled={v10_enabled}. "
                                f"This indicates dynamic flip (not allowed). "
                                f"expected_v10_enabled={expected_v10_enabled}, actual_v10_enabled={v10_enabled}"
                            )
                        
                        # CONTROL-FLOW INVARIANTS: Track execution path
                        control_flow = telemetry_data.get("control_flow", {})
                        control_flow_counts = control_flow.get("counts", {})
                        control_flow_last = control_flow.get("last")
                        
                        after_enable_check = control_flow_counts.get("AFTER_V10_ENABLE_CHECK", 0)
                        enter_routing_branch = control_flow_counts.get("ENTER_V10_ROUTING_BRANCH", 0)
                        
                        # INVARIANT 4: AFTER_V10_ENABLE_CHECK > 0 when entry_v10_enabled=True
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
                
                # E) TRUTH-only invariant: Score gate impossible check
                # If score_max > threshold AND above_threshold == 0, this is impossible and must FATAL
                is_truth_or_smoke_local = os.getenv("GX1_RUN_MODE", "").upper() in ["TRUTH", "SMOKE"]
                if is_truth_or_smoke_local and runner and hasattr(runner, "entry_manager") and runner.entry_manager:
                    em = runner.entry_manager
                    killchain_n_above_threshold = getattr(em, "killchain_n_above_threshold", 0)
                    entry_score_max = chunk_footer.get("entry_score_max")
                    entry_score_samples = chunk_footer.get("entry_score_samples", 0)
                    
                    # Get threshold used (from killchain check - must match what killchain check actually uses)
                    # In replay mode, killchain check uses entry_policy_sniper_v10_ctx or sniper_policy config
                    # We must read from the same config to match killchain check logic
                    threshold_val = None
                    try:
                        if hasattr(runner, "replay_mode") and runner.replay_mode:
                            # Replay mode: read from entry_policy_sniper_v10_ctx or sniper_policy (same as killchain check)
                            replay_policy_cfg = runner.policy.get("entry_policy_sniper_v10_ctx", {}) or runner.policy.get("sniper_policy", {})
                            threshold_val = float(replay_policy_cfg.get("min_prob_long", 0.68))  # Match SniperPolicyParams default
                        else:
                            # Live mode: use entry_v9_policy_sniper config
                            policy_sniper_cfg = runner.policy.get("entry_v9_policy_sniper", {})
                            threshold_val = float(policy_sniper_cfg.get("min_prob_long", 0.67))
                    except Exception:
                        threshold_val = None
                    
                    # Check invariant: if score_max > threshold and above_threshold == 0, this is impossible
                    if entry_score_max is not None and threshold_val is not None and entry_score_samples > 0:
                        if entry_score_max > threshold_val and killchain_n_above_threshold == 0:
                            fatal_capsule = {
                                "chunk_idx": chunk_idx,
                                "run_id": run_id,
                                "fatal_reason": "SCORE_GATE_IMPOSSIBLE",
                                "entry_score_max": entry_score_max,
                                "threshold_used": threshold_val,
                                "killchain_n_above_threshold": killchain_n_above_threshold,
                                "entry_score_samples": entry_score_samples,
                                "killchain_n_entry_pred_total": getattr(em, "killchain_n_entry_pred_total", 0),
                                "message": f"Entry score max ({entry_score_max:.4f}) > threshold ({threshold_val:.4f}) but killchain_n_above_threshold=0. This is impossible and indicates a bug in score gate logic.",
                                "timestamp": dt_now_iso(),
                            }
                            fatal_path = chunk_output_dir / "SCORE_GATE_IMPOSSIBLE_FATAL.json"
                            from gx1.utils.atomic_json import atomic_write_json
                            atomic_write_json(fatal_path, fatal_capsule)
                            raise RuntimeError(
                                f"[CHUNK {chunk_idx}] FATAL: SCORE_GATE_IMPOSSIBLE - "
                                f"entry_score_max={entry_score_max:.4f} > threshold={threshold_val:.4f} "
                                f"but killchain_n_above_threshold=0. "
                                f"This indicates a bug in score gate logic. "
                                f"See {fatal_path} for details."
                            )
                
                with open(chunk_footer_path, "w") as f:
                    json.dump(chunk_footer, f, indent=2)
                
                log.info(f"[CHUNK {chunk_idx}] chunk_footer.json written: status={status}")
                
                # ------------------------------------------------------------------------
                # TRUTH-only: Write OPEN_TRADES_SNAPSHOT.json for carry-forward (status=ok only)
                # ------------------------------------------------------------------------
                if is_truth_or_smoke_worker and status == "ok" and runner:
                    try:
                        open_trades = getattr(runner, "open_trades", []) or []
                        open_end = len(open_trades)
                        
                        if open_end > 0:
                            # Build snapshot with minimal trade state needed to continue
                            snapshot_trades = []
                            for trade in open_trades:
                                try:
                                    trade_snapshot = {
                                        "trade_id": getattr(trade, "trade_id", None),
                                        "trade_uid": getattr(trade, "trade_uid", None),
                                        "entry_ts": getattr(trade, "entry_time", None).isoformat() if hasattr(trade, "entry_time") and trade.entry_time else None,
                                        "entry_price": float(getattr(trade, "entry_price", 0.0)),
                                        "entry_bid": float(getattr(trade, "entry_bid", getattr(trade, "entry_price", 0.0))),
                                        "entry_ask": float(getattr(trade, "entry_ask", getattr(trade, "entry_price", 0.0))),
                                        "side": getattr(trade, "side", "long"),
                                        "size": float(getattr(trade, "units", 0.0)),
                                        "entry_score": getattr(trade, "entry_score", None),
                                        "extra": getattr(trade, "extra", {}) or {},
                                    }
                                    snapshot_trades.append(trade_snapshot)
                                except Exception as trade_snap_error:
                                    log.warning(f"[CHUNK {chunk_idx}] Failed to snapshot trade {getattr(trade, 'trade_id', 'unknown')}: {trade_snap_error}")
                            
                            # Get last bar timestamp from chunk_df
                            last_bar_ts = None
                            if chunk_df is not None and len(chunk_df) > 0:
                                last_bar_ts = chunk_df.index[-1].isoformat()
                            
                            snapshot = {
                                "chunk_id": chunk_idx,
                                "run_id": run_id,
                                "last_bar_ts": last_bar_ts,
                                "open_trades_count": open_end,
                                "open_trades": snapshot_trades,
                                "timestamp": dt_now_iso(),
                            }
                            
                            from gx1.utils.atomic_json import atomic_write_json
                            snapshot_path = chunk_output_dir / "OPEN_TRADES_SNAPSHOT.json"
                            atomic_write_json(snapshot_path, snapshot)
                            log.info(f"[CHUNK {chunk_idx}] OPEN_TRADES_SNAPSHOT.json written: {open_end} open trades")
                        else:
                            # No open trades - snapshot not needed, but log for clarity
                            log.debug(f"[CHUNK {chunk_idx}] No open trades at end - skipping snapshot")
                    except Exception as snapshot_error:
                        # Best-effort: do not break non-TRUTH runs
                        log.warning(f"[CHUNK {chunk_idx}] Failed to write OPEN_TRADES_SNAPSHOT.json: {snapshot_error}")
                
                # CHUNK-LOCAL PADDING: Write WARMUP_LEDGER.json and .md per chunk
                if chunk_local_padding_days > 0 and chunk_footer.get("warmup_ledger"):
                    warmup_ledger = chunk_footer["warmup_ledger"]
                    warmup_ledger_json_path = chunk_output_dir / "WARMUP_LEDGER.json"
                    with open(warmup_ledger_json_path, "w") as f:
                        json.dump(warmup_ledger, f, indent=2)
                    
                    # Write Markdown version
                    warmup_ledger_md_path = chunk_output_dir / "WARMUP_LEDGER.md"
                    warmup_ledger_md_lines = [
                        f"# Warmup Ledger (Chunk {chunk_idx})",
                        "",
                        f"**Chunk ID:** {chunk_idx}",
                        f"**Chunk Local Padding Days:** {warmup_ledger.get('chunk_local_padding_days', 0)}",
                        "",
                        "## Timestamps",
                        "",
                        f"- **Actual Replay Start:** `{warmup_ledger.get('actual_replay_start_ts', 'N/A')}`",
                        f"- **Eval Start:** `{warmup_ledger.get('eval_start_ts', 'N/A')}`",
                        f"- **Eval End:** `{warmup_ledger.get('eval_end_ts', 'N/A')}`",
                        "",
                        "## Warmup Status",
                        "",
                        f"- **Warmup Required Bars:** {warmup_ledger.get('warmup_required_bars', 0)}",
                        f"- **Warmup Seen Bars:** {warmup_ledger.get('warmup_seen_bars', 0)}",
                        f"- **Warmup Skipped Total:** {warmup_ledger.get('warmup_skipped_total', 0)}",
                        f"- **Warmup Completed TS:** `{warmup_ledger.get('warmup_completed_ts', 'N/A')}`",
                        "",
                        "## Processing",
                        "",
                        f"- **Bars Processed Total:** {warmup_ledger.get('bars_processed_total', 0)}",
                        "",
                        "---",
                        f"*Generated: {dt_now_iso()}*",
                    ]
                    with open(warmup_ledger_md_path, "w") as f:
                        f.write("\n".join(warmup_ledger_md_lines))
                    
                    log.info(f"[CHUNK {chunk_idx}] WARMUP_LEDGER.json/.md written")
            except Exception as footer_error:
                log.error(f"[CHUNK {chunk_idx}] Failed to write chunk_footer.json: {footer_error}", exc_info=True)
                
                # CRITICAL: Write CHUNK_FAIL_CAPSULE.json BEFORE stub_footer (atomic, always valid JSON)
                try:
                    from gx1.utils.atomic_json import atomic_write_json
                    import traceback as tb_module
                    
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
                        "telemetry_required": os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1",
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
                        "bars_passed_hard_eligibility": bar_counters.get("bars_passed_hard_eligibility"),
                        "bars_blocked_hard_eligibility": bar_counters.get("bars_blocked_hard_eligibility"),
                        "bars_passed_soft_eligibility": bar_counters.get("bars_passed_soft_eligibility"),
                        "bars_blocked_soft_eligibility": bar_counters.get("bars_blocked_soft_eligibility"),
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
                        "replay_mode": "PREBUILT" if os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1" else "UNKNOWN",
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
                                    "prebuilt_enabled": os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
                                    "prebuilt_used_flag": getattr(runner, "prebuilt_used", False),
                                    "prebuilt_features_df_exists": prebuilt_features_df_exists,
                                    "prebuilt_features_df_is_none": prebuilt_features_df_is_none,
                                }
                        else:
                            # Runner not created yet - minimal dump
                            stub_gate_dump = {
                                "is_replay": None,
                                "prebuilt_enabled": os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1",
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
        error_traceback = "".join(tb_module.format_exception(type(outer_exception), outer_exception, outer_exception.__traceback__))
        
        # A) Update skip_ledger with exception info
        skip_ledger["stage"] = "exception"
        skip_ledger["exception_type"] = type(outer_exception).__name__
        skip_ledger["exception_msg"] = str(outer_exception)[:500]
        skip_ledger["traceback"] = error_traceback[:5000]  # Limit traceback size
        
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
                "telemetry_required": os.getenv("GX1_REQUIRE_ENTRY_TELEMETRY", "0") == "1",
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
            # B) Include skip_ledger_path in CHUNK_FAIL_CAPSULE
            skip_ledger_path_str = None
            if chunk_output_dir:
                skip_ledger_path_str = f"chunk_{chunk_idx}/SKIP_LEDGER_FINAL.json"
            
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
                "replay_mode": "PREBUILT" if os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES") == "1" else "UNKNOWN",
                "policy_id": policy_id if 'policy_id' in locals() else None,
                "bundle_sha256": bundle_sha256 if bundle_sha256 else None,
                "bundle_dir_resolved": bundle_dir_resolved,
                "run_identity_keys": list(run_identity_data.keys()) if run_identity_data else None,
                "telemetry_status": telemetry_status,
                "argv": sys.argv.copy() if hasattr(sys, 'argv') else None,
                "cwd": str(Path.cwd()),
                "sys_executable": sys.executable if hasattr(sys, 'executable') else None,
                "skip_ledger_path": skip_ledger_path_str,  # B) Link to SKIP_LEDGER_FINAL.json
                "hint": "failure happened before normal flush; see PREBUILT_FAIL_CAPSULE if present. See skip_ledger_path for detailed skip breakdown.",
                "timestamp": dt_now_iso(),
            }
            
            # Write capsule atomically
            capsule_written = False
            capsule_path = None
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
                    capsule_path = fallback_path
                    log.error(f"[CHUNK {chunk_idx}] Wrote CHUNK_FAIL_CAPSULE.json to fallback: {fallback_path}")
        except Exception as capsule_error:
            log.error(f"[CHUNK {chunk_idx}] Failed to write CHUNK_FAIL_CAPSULE.json: {capsule_error}", exc_info=True)  # Give up
            capsule_path = None
        
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
    
    finally:
        # A) ALWAYS write SKIP_LEDGER_FINAL.json (best-effort, never fail)
        # This ensures we have diagnostic info even if exception occurred before invariant check
        try:
            # Update timestamp
            skip_ledger["timestamp"] = dt_now_iso()
            
            # Ensure chunk_output_dir exists
            if chunk_output_dir is None:
                try:
                    chunk_output_dir = output_dir / f"chunk_{chunk_idx}"
                    chunk_output_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    chunk_output_dir = None
            
            if chunk_output_dir:
                skip_ledger_final_path = chunk_output_dir / "SKIP_LEDGER_FINAL.json"
                try:
                    from gx1.utils.atomic_json import atomic_write_json
                    atomic_write_json(skip_ledger_final_path, skip_ledger)
                    log.info(f"[CHUNK {chunk_idx}] [SKIP_LEDGER] Wrote SKIP_LEDGER_FINAL.json to {skip_ledger_final_path}")
                except Exception as ledger_error:
                    log.error(f"[CHUNK {chunk_idx}] [SKIP_LEDGER] Failed to write SKIP_LEDGER_FINAL.json (atomic): {ledger_error}")
                    # Fallback: write directly (non-atomic)
                    try:
                        with open(skip_ledger_final_path, "w") as f:
                            import json
                            json.dump(skip_ledger, f, indent=2, default=str)
                        log.warning(f"[CHUNK {chunk_idx}] [SKIP_LEDGER] Wrote SKIP_LEDGER_FINAL.json (non-atomic fallback)")
                    except Exception as fallback_error:
                        log.error(f"[CHUNK {chunk_idx}] [SKIP_LEDGER] CRITICAL: Failed to write SKIP_LEDGER_FINAL.json even with fallback: {fallback_error}")
                        # Last resort: write to /tmp
                        try:
                            import tempfile
                            tmp_path = Path(tempfile.gettempdir()) / f"chunk_{chunk_idx}_SKIP_LEDGER_FINAL_{run_id}.json"
                            with open(tmp_path, "w") as f:
                                json.dump(skip_ledger, f, indent=2, default=str)
                            log.error(f"[CHUNK {chunk_idx}] [SKIP_LEDGER] Wrote SKIP_LEDGER_FINAL.json to /tmp fallback: {tmp_path}")
                        except Exception:
                            log.error(f"[CHUNK {chunk_idx}] [SKIP_LEDGER] FATAL: Could not write SKIP_LEDGER_FINAL.json anywhere")
        except Exception as final_error:
            # Never fail in finally - just log
            log.error(f"[CHUNK {chunk_idx}] [SKIP_LEDGER] Exception in finally block: {final_error}", exc_info=True)



def _assert_no_local_os_imports():
    import inspect
    import re
    src = inspect.getsource(process_chunk)
    assert "import os" not in src, "Local os import forbidden in process_chunk"

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
        truth_telemetry = os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
        if truth_telemetry:
            # Hard fail if truth telemetry is expected but missing
            fatal_path = output_dir / "MASTER_FATAL.json"
            fatal_data = {
                "error": "TELEMETRY_MISSING",
                "reason": "No ENTRY_FEATURES_USED.json files found in any chunks",
                "chunks_scanned": len(chunk_telemetry),
                "gx1_truth_telemetry": True,
                "timestamp": datetime.utcnow().isoformat(),
            }
            write_json_atomic(fatal_path, fatal_data, output_dir=output_dir)
            log.error(
                f"[TELEMETRY] FATAL: GX1_TRUTH_TELEMETRY=1 but no telemetry found. "
                f"Wrote MASTER_FATAL.json: {fatal_path}"
            )
            raise RuntimeError(
                f"[TELEMETRY] FATAL: GX1_TRUTH_TELEMETRY=1 but no ENTRY_FEATURES_USED.json files found. "
                f"Telemetry must be written for truth verification. "
                f"Chunks scanned: {len(chunk_telemetry)}"
            )
        else:
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
        
        # Aggregate xgb_flows and transformer_outputs from chunks (for truth verification)
        aggregated_xgb_flows = []
        aggregated_transformer_outputs = []
        
        # Aggregate run-level funnel ledger (sum counters across chunks)
        aggregated_funnel_ledger = {
            "post_warmup_count": 0,
            "pregate_pass_count": 0,
            "pregate_block_count": 0,
            "post_pregate_enter_count": 0,
            "pre_eval_enter_count": 0,
            "eval_called_count": 0,
            "hard_eligibility_checked_count": 0,
            "soft_eligibility_checked_count": 0,
            "session_gate_checked_count": 0,
            "vol_guard_checked_count": 0,
            "score_gate_checked_count": 0,
            "score_gate_allow_count": 0,
            "score_gate_block_count": 0,
            "predict_entered_count": 0,
            "pre_call_count": 0,
            "transformer_forward_calls": 0,
            "exceptions_count": 0,
            "reason_counters": {
                "eligibility_blocks": 0,
                "session_blocks": 0,
                "vol_regime_blocks": 0,
                "score_blocks": 0,
            },
        }
        aggregated_per_session_funnel_ledger = {
            session: {
                "bars_seen": 0,
                "post_warmup_count": 0,
                "hard_eligibility_pass_count": 0,
                "hard_eligibility_block_count": 0,
                "hard_eligibility_checked_count": 0,
                "soft_eligibility_pass_count": 0,
                "soft_eligibility_block_count": 0,
                "soft_eligibility_checked_count": 0,
                "session_gate_pass_count": 0,
                "session_gate_block_count": 0,
                "session_gate_checked_count": 0,
                "vol_guard_pass_count": 0,
                "vol_guard_block_count": 0,
                "vol_guard_checked_count": 0,
                "score_gate_pass_count": 0,
                "score_gate_block_count": 0,
                "score_gate_checked_count": 0,
                "score_gate_allow_count": 0,
                "pregate_pass_count": 0,
                "pregate_block_count": 0,
                "post_pregate_enter_count": 0,
                "pre_eval_enter_count": 0,
                "eval_called_count": 0,
                "predict_entered_count": 0,
                "pre_call_count": 0,
                "transformer_forward_calls": 0,
                "exceptions_count": 0,
            }
            for session in ["ASIA", "EU", "OVERLAP", "US"]
        }
        aggregated_per_session_reason_counters = {
            session: {
                "eligibility_blocks": 0,
                "session_blocks": 0,
                "vol_regime_blocks": 0,
                "score_blocks": 0,
                "cost_blocks": 0,
                "hard_eligibility_block_reasons": {},
                "soft_eligibility_block_reasons": {},
                "session_gate_block_reasons": {},
                "vol_guard_block_reasons": {},
                "score_gate_block_reasons": {},
                "stage2_block_reasons": {},
                "stage3_block_reasons": {},
                "pre_model_return_reasons": {},
                "exception_types": {},
            }
            for session in ["ASIA", "EU", "OVERLAP", "US"]
        }
        aggregated_blocked_between_pregate_and_eval_by_session = {
            session: {
                "counts_by_reason": {},
                "first_ts_samples": [],
                "first_kill_reason": None,
            }
            for session in ["ASIA", "EU", "OVERLAP", "US"]
        }

        # Aggregate hard eligibility rule summary across chunks (deterministic)
        aggregated_hard_eligibility_rule_summary: Dict[str, Dict[str, Any]] = {}
        for _, chunk_data in all_entry_features_used:
            rule_summary = chunk_data.get("hard_eligibility_rule_summary", {}) or {}
            for rule_name, rule_data in rule_summary.items():
                agg = aggregated_hard_eligibility_rule_summary.setdefault(
                    rule_name,
                    {"fail_count": 0, "pass_count": 0, "sessions": set(), "sample_values": []},
                )
                agg["fail_count"] += int(rule_data.get("fail_count", 0))
                agg["pass_count"] += int(rule_data.get("pass_count", 0))
                agg["sessions"].update(rule_data.get("sessions", []))
                for sample in rule_data.get("sample_values", []):
                    if len(agg["sample_values"]) < 5:
                        agg["sample_values"].append(sample)
        aggregated_diagnostic_bypass_count_by_session = {}
        aggregated_diagnostic_bypass_gate_name_by_session = {}
        
        for chunk_idx, chunk_data in all_entry_features_used:
            gate_stats = chunk_data.get("gate_stats", {})
            for gate_name, stats in gate_stats.items():
                if gate_name not in total_gate_stats:
                    total_gate_stats[gate_name] = {"executed": 0, "blocked": 0, "passed": 0}
                if isinstance(stats, dict):
                    total_gate_stats[gate_name]["executed"] += stats.get("executed", 0)
                    total_gate_stats[gate_name]["blocked"] += stats.get("blocked", 0)
                    total_gate_stats[gate_name]["passed"] += stats.get("passed", 0)
                else:
                    total_gate_stats[gate_name]["executed"] += stats
            
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
            
            # Aggregate run-level funnel ledger from chunk
            chunk_funnel_ledger = chunk_data.get("run_level_funnel_ledger", {})
            if chunk_funnel_ledger:
                aggregated_funnel_ledger["post_warmup_count"] += chunk_funnel_ledger.get("post_warmup_count", 0)
                aggregated_funnel_ledger["pregate_pass_count"] += chunk_funnel_ledger.get("pregate_pass_count", 0)
                aggregated_funnel_ledger["pregate_block_count"] += chunk_funnel_ledger.get("pregate_block_count", 0)
                aggregated_funnel_ledger["post_pregate_enter_count"] += chunk_funnel_ledger.get("post_pregate_enter_count", 0)
                aggregated_funnel_ledger["pre_eval_enter_count"] += chunk_funnel_ledger.get("pre_eval_enter_count", 0)
                aggregated_funnel_ledger["eval_called_count"] += chunk_funnel_ledger.get("eval_called_count", 0)
                aggregated_funnel_ledger["hard_eligibility_checked_count"] += chunk_funnel_ledger.get("hard_eligibility_checked_count", 0)
                aggregated_funnel_ledger["soft_eligibility_checked_count"] += chunk_funnel_ledger.get("soft_eligibility_checked_count", 0)
                aggregated_funnel_ledger["session_gate_checked_count"] += chunk_funnel_ledger.get("session_gate_checked_count", 0)
                aggregated_funnel_ledger["vol_guard_checked_count"] += chunk_funnel_ledger.get("vol_guard_checked_count", 0)
                aggregated_funnel_ledger["score_gate_checked_count"] += chunk_funnel_ledger.get("score_gate_checked_count", 0)
                aggregated_funnel_ledger["score_gate_allow_count"] += chunk_funnel_ledger.get("score_gate_allow_count", 0)
                aggregated_funnel_ledger["score_gate_block_count"] += chunk_funnel_ledger.get("score_gate_block_count", 0)
                aggregated_funnel_ledger["predict_entered_count"] += chunk_funnel_ledger.get("predict_entered_count", 0)
                aggregated_funnel_ledger["pre_call_count"] += chunk_funnel_ledger.get("pre_call_count", 0)
                aggregated_funnel_ledger["transformer_forward_calls"] += chunk_funnel_ledger.get("transformer_forward_calls", 0)
                aggregated_funnel_ledger["exceptions_count"] += chunk_funnel_ledger.get("exceptions_count", 0)
                chunk_reason_counters = chunk_funnel_ledger.get("reason_counters", {})
                aggregated_funnel_ledger["reason_counters"]["eligibility_blocks"] += chunk_reason_counters.get("eligibility_blocks", 0)
                aggregated_funnel_ledger["reason_counters"]["session_blocks"] += chunk_reason_counters.get("session_blocks", 0)
                aggregated_funnel_ledger["reason_counters"]["vol_regime_blocks"] += chunk_reason_counters.get("vol_regime_blocks", 0)
                aggregated_funnel_ledger["reason_counters"]["score_blocks"] += chunk_reason_counters.get("score_blocks", 0)
            
            # Aggregate per-session funnel ledger from chunk
            chunk_per_session_funnel = chunk_data.get("per_session_funnel_ledger", {})
            for session, counters in chunk_per_session_funnel.items():
                if session not in aggregated_per_session_funnel_ledger:
                    aggregated_per_session_funnel_ledger[session] = {
                        "bars_seen": 0,
                        "post_warmup_count": 0,
                        "pregate_pass_count": 0,
                        "pregate_block_count": 0,
                        "post_pregate_enter_count": 0,
                        "pre_eval_enter_count": 0,
                        "eval_called_count": 0,
                        "predict_entered_count": 0,
                        "pre_call_count": 0,
                        "transformer_forward_calls": 0,
                        "exceptions_count": 0,
                        "score_gate_allow_count": 0,
                        "score_gate_block_count": 0,
                    }
                for key, value in counters.items():
                    if isinstance(value, (int, float)):
                        aggregated_per_session_funnel_ledger[session][key] += value
            
            # Aggregate per-session reason counters from chunk
            chunk_per_session_reasons = chunk_data.get("per_session_reason_counters", {})
            for session, reasons in chunk_per_session_reasons.items():
                if session not in aggregated_per_session_reason_counters:
                    aggregated_per_session_reason_counters[session] = {
                        "eligibility_blocks": 0,
                        "session_blocks": 0,
                        "vol_regime_blocks": 0,
                        "score_blocks": 0,
                        "cost_blocks": 0,
                        "stage2_block_reasons": {},
                        "stage3_block_reasons": {},
                        "pre_model_return_reasons": {},
                    }
                for key, value in reasons.items():
                    if (key.endswith("_reasons") or key == "exception_types") and isinstance(value, dict):
                        existing = aggregated_per_session_reason_counters[session].get(key, {})
                        for reason_key, reason_count in value.items():
                            existing[reason_key] = existing.get(reason_key, 0) + reason_count
                        aggregated_per_session_reason_counters[session][key] = existing
                    elif isinstance(value, (int, float)):
                        aggregated_per_session_reason_counters[session][key] = aggregated_per_session_reason_counters[session].get(key, 0) + value
            
            # Aggregate blocked_between_pregate_and_eval_by_session from chunk
            chunk_blocked_between = chunk_data.get("blocked_between_pregate_and_eval_by_session", {}) or {}
            for session, block_data in chunk_blocked_between.items():
                if session not in aggregated_blocked_between_pregate_and_eval_by_session:
                    aggregated_blocked_between_pregate_and_eval_by_session[session] = {
                        "counts_by_reason": {},
                        "first_ts_samples": [],
                        "first_kill_reason": None,
                    }
                existing = aggregated_blocked_between_pregate_and_eval_by_session[session]
                counts_by_reason = block_data.get("counts_by_reason", {}) if isinstance(block_data, dict) else {}
                for reason_key, reason_count in counts_by_reason.items():
                    if isinstance(reason_count, (int, float)):
                        existing_counts = existing.get("counts_by_reason", {})
                        existing_counts[reason_key] = existing_counts.get(reason_key, 0) + int(reason_count)
                        existing["counts_by_reason"] = existing_counts
                if not existing.get("first_kill_reason"):
                    first_kill_reason = block_data.get("first_kill_reason") if isinstance(block_data, dict) else None
                    if first_kill_reason:
                        existing["first_kill_reason"] = first_kill_reason
                samples = block_data.get("first_ts_samples", []) if isinstance(block_data, dict) else []
                if isinstance(samples, list):
                    for sample in samples:
                        if len(existing["first_ts_samples"]) >= 10:
                            break
                        existing["first_ts_samples"].append(sample)
            
            # Aggregate diagnostic bypass counters
            chunk_diag_counts = chunk_data.get("diagnostic_bypass_count_by_session", {})
            for session, count in chunk_diag_counts.items():
                if isinstance(count, (int, float)):
                    aggregated_diagnostic_bypass_count_by_session[session] = (
                        aggregated_diagnostic_bypass_count_by_session.get(session, 0) + count
                    )
                elif isinstance(count, dict):
                    # Defensive: sum nested counts if present
                    nested_total = sum(
                        v for v in count.values() if isinstance(v, (int, float))
                    )
                    aggregated_diagnostic_bypass_count_by_session[session] = (
                        aggregated_diagnostic_bypass_count_by_session.get(session, 0) + nested_total
                    )
                else:
                    log.warning(
                        f"[TELEMETRY] Unexpected diagnostic_bypass_count type for session={session}: {type(count)}"
                    )
            chunk_diag_gates = chunk_data.get("diagnostic_bypass_gate_name_by_session", {})
            for session, gate_counts in chunk_diag_gates.items():
                existing = aggregated_diagnostic_bypass_gate_name_by_session.get(session, {})
                for gate_name, count in gate_counts.items():
                    if isinstance(count, (int, float)):
                        existing[gate_name] = existing.get(gate_name, 0) + count
                    elif isinstance(count, dict):
                        nested_total = sum(
                            v for v in count.values() if isinstance(v, (int, float))
                        )
                        existing[gate_name] = existing.get(gate_name, 0) + nested_total
                    else:
                        log.warning(
                            f"[TELEMETRY] Unexpected diagnostic_bypass_gate count type for session={session}, "
                            f"gate={gate_name}: {type(count)}"
                        )
                aggregated_diagnostic_bypass_gate_name_by_session[session] = existing
            
            # Aggregate xgb_flows and transformer_outputs (if present in chunk telemetry)
            # Note: These may be in ENTRY_FEATURES_TELEMETRY.json, not ENTRY_FEATURES_USED.json
            # We'll load from ENTRY_FEATURES_TELEMETRY.json if available
            chunk_dir = output_dir / f"chunk_{chunk_idx}"
            telemetry_path = chunk_dir / "ENTRY_FEATURES_TELEMETRY.json"
            if telemetry_path.exists():
                try:
                    with open(telemetry_path, "r") as f:
                        chunk_telemetry_data = json.load(f)
                        # Aggregate xgb_flows
                        chunk_xgb_flows = chunk_telemetry_data.get("xgb_flows", [])
                        if chunk_xgb_flows:
                            aggregated_xgb_flows.extend(chunk_xgb_flows)
                        # Aggregate transformer_outputs
                        chunk_transformer_outputs = chunk_telemetry_data.get("transformer_outputs", [])
                        if chunk_transformer_outputs:
                            aggregated_transformer_outputs.extend(chunk_transformer_outputs)
                        # Aggregate per-session funnel ledger from telemetry (if not already in ENTRY_FEATURES_USED.json)
                        chunk_per_session_funnel_telemetry = chunk_telemetry_data.get("per_session_funnel_ledger", {})
                        if chunk_per_session_funnel_telemetry:
                            for session, counters in chunk_per_session_funnel_telemetry.items():
                                if session not in aggregated_per_session_funnel_ledger:
                                    aggregated_per_session_funnel_ledger[session] = {
                                        "bars_seen": 0,
                                        "post_warmup_count": 0,
                                        "pregate_pass_count": 0,
                                        "pregate_block_count": 0,
                                        "eval_called_count": 0,
                                        "predict_entered_count": 0,
                                        "pre_call_count": 0,
                                        "transformer_forward_calls": 0,
                                        "exceptions_count": 0,
                                    }
                                for key, value in counters.items():
                                    if isinstance(value, (int, float)):
                                        aggregated_per_session_funnel_ledger[session][key] += value
                        # Aggregate per-session reason counters from telemetry
                        chunk_per_session_reasons_telemetry = chunk_telemetry_data.get("per_session_reason_counters", {})
                        if chunk_per_session_reasons_telemetry:
                            for session, reasons in chunk_per_session_reasons_telemetry.items():
                                if session not in aggregated_per_session_reason_counters:
                                    aggregated_per_session_reason_counters[session] = {
                                        "eligibility_blocks": 0,
                                        "session_blocks": 0,
                                        "vol_regime_blocks": 0,
                                        "score_blocks": 0,
                                        "cost_blocks": 0,
                                        "stage2_block_reasons": {},
                                        "stage3_block_reasons": {},
                                        "pre_model_return_reasons": {},
                                    }
                                for key, value in reasons.items():
                                    if key.endswith("_reasons") and isinstance(value, dict):
                                        existing = aggregated_per_session_reason_counters[session].get(key, {})
                                        for reason_key, reason_count in value.items():
                                            existing[reason_key] = existing.get(reason_key, 0) + reason_count
                                        aggregated_per_session_reason_counters[session][key] = existing
                                    elif isinstance(value, (int, float)):
                                        aggregated_per_session_reason_counters[session][key] = aggregated_per_session_reason_counters[session].get(key, 0) + value
                except Exception as e:
                    log.warning(f"[TELEMETRY] Failed to load xgb_flows/transformer_outputs from chunk {chunk_idx}: {e}")
        
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
        aggregated_score_gate_decision_samples: List[Dict[str, Any]] = []
        for _, chunk_data in all_entry_features_used:
            samples = chunk_data.get("score_gate_decision_samples", [])
            if isinstance(samples, list):
                for sample in samples:
                    if isinstance(sample, dict):
                        aggregated_score_gate_decision_samples.append(sample)
        aggregated_score_gate_decision_samples.sort(
            key=lambda s: (s.get("ts") or "", s.get("session") or "")
        )
        if len(aggregated_score_gate_decision_samples) > 10:
            aggregated_score_gate_decision_samples = aggregated_score_gate_decision_samples[:10]
        
        # Get toggle state from first chunk (should be identical)
        first_toggles = first_data.get("toggles", {})
        
        # Check for no_entry_evaluations flag (if any chunk has it, set it in master)
        master_no_entry_evaluations = False
        master_no_entry_reason = None
        for _, chunk_data in all_entry_features_used:
            if chunk_data.get("no_entry_evaluations", False):
                master_no_entry_evaluations = True
                master_no_entry_reason = chunk_data.get("no_entry_evaluations_reason", "unknown")
                break  # Use first chunk's reason
        
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
            "score_gate_decision_samples": aggregated_score_gate_decision_samples,
            "run_level_funnel_ledger": aggregated_funnel_ledger,  # Run-level funnel ledger (SSoT for gated verdict)
            "hard_eligibility_rule_summary": {
                rule_name: {
                    "fail_count": aggregated_hard_eligibility_rule_summary[rule_name]["fail_count"],
                    "pass_count": aggregated_hard_eligibility_rule_summary[rule_name]["pass_count"],
                    "sessions": sorted(aggregated_hard_eligibility_rule_summary[rule_name]["sessions"]),
                    "sample_values": aggregated_hard_eligibility_rule_summary[rule_name]["sample_values"],
                }
                for rule_name in sorted(aggregated_hard_eligibility_rule_summary.keys())
            },
        }
        
        # Add no_entry_evaluations flag if any chunk has it
        if master_no_entry_evaluations:
            master_data["no_entry_evaluations"] = True
            if master_no_entry_reason:
                master_data["no_entry_evaluations_reason"] = master_no_entry_reason
        
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
        
        # Add aggregated per-session funnel ledger and reason counters (for root-cause analysis)
        master_data["per_session_funnel_ledger"] = aggregated_per_session_funnel_ledger
        master_data["blocked_between_pregate_and_eval_by_session"] = aggregated_blocked_between_pregate_and_eval_by_session
        master_data["per_session_reason_counters"] = aggregated_per_session_reason_counters
        
        # Add aggregated diagnostic bypass counters (if present)
        if aggregated_diagnostic_bypass_count_by_session:
            master_data["diagnostic_bypass_count_by_session"] = aggregated_diagnostic_bypass_count_by_session
        if aggregated_diagnostic_bypass_gate_name_by_session:
            master_data["diagnostic_bypass_gate_name_by_session"] = aggregated_diagnostic_bypass_gate_name_by_session
        
        # Add aggregated xgb_flows and transformer_outputs (for truth verification)
        master_data["xgb_flows"] = aggregated_xgb_flows
        master_data["transformer_outputs"] = aggregated_transformer_outputs
        log.info(f"[TELEMETRY] Aggregated {len(aggregated_xgb_flows)} xgb_flows and {len(aggregated_transformer_outputs)} transformer_outputs from chunks")
        
        # Write master JSON (both _MASTER.json and ENTRY_FEATURES_USED.json for compatibility)
        master_json_path = output_dir / "ENTRY_FEATURES_USED_MASTER.json"
        with open(master_json_path, "w") as f:
            json.dump(master_data, f, indent=2, sort_keys=True)
        log.info(f"[TELEMETRY] Wrote master JSON: {master_json_path}")
        
        # Also write as ENTRY_FEATURES_USED.json for truth verification scripts
        entry_features_path = output_dir / "ENTRY_FEATURES_USED.json"
        with open(entry_features_path, "w") as f:
            json.dump(master_data, f, indent=2, sort_keys=True)
        log.info(f"[TELEMETRY] Wrote ENTRY_FEATURES_USED.json: {entry_features_path}")
        
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
    
    OUTPUT POLICY: Respects GX1_OUTPUT_MODE (MINIMAL/DEBUG/TRUTH).
    In MINIMAL mode, skips raw_signals, policy_decisions, trade_outcomes parquet files.
    
    Returns dict with paths to merged artifacts.
    """
    # Use global os import (line 18) instead of local import
    # OUTPUT POLICY: Get output mode (MINIMAL/DEBUG/TRUTH)
    # Default: MINIMAL (prevents reports explosion)
    output_mode_env = os.getenv("GX1_OUTPUT_MODE", "").upper()
    if output_mode_env in ("MINIMAL", "DEBUG", "TRUTH"):
        output_mode = output_mode_env
    else:
        output_mode = "MINIMAL"  # Default: MINIMAL
    
    log.info(f"[MERGE] Merging {len(chunk_results)} chunks (output_mode={output_mode})")
    
    merged_artifacts = {}
    
    # OUTPUT POLICY: In MINIMAL mode, skip raw_signals, policy_decisions, trade_outcomes parquet
    # But ALWAYS merge metrics (required for decision-making)
    if output_mode.upper() != "MINIMAL":
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
    # OUTPUT POLICY: In MINIMAL mode, skip attribution (not required for decision-making)
    if output_mode.upper() != "MINIMAL":
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
    
    # OUTPUT POLICY: In MINIMAL mode, skip trade_outcomes parquet (but still compute metrics from JSON)
    all_trade_outcomes = []
    if output_mode.upper() != "MINIMAL":
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


def write_exit_coverage_summary(output_dir: Path) -> None:
    """
    TRUTH-only: Aggregate per-chunk exit coverage counters and write:
      - EXIT_COVERAGE_SUMMARY.json
      - EXIT_COVERAGE_SUMMARY.md
    """
    try:
        output_dir = Path(output_dir).resolve()
        chunk_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
        per_chunk = []

        totals = {
            "exit_attempts_total": 0,
            "exit_request_close_called": 0,
            "exit_request_close_accepted": 0,
            "exit_summary_logged": 0,
            "exit_event_rows_written": 0,
            "force_close_attempts_replay_end": 0,
            "force_close_logged_replay_end": 0,
            "force_close_attempts_replay_eof": 0,
            "force_close_logged_replay_eof": 0,
            "open_trades_start_of_chunk": 0,
            "open_trades_end_of_chunk": 0,
            "open_to_closed_within_chunk": 0,
            "replay_end_or_eof_triggered_chunks": 0,
            "accounting_close_enabled_chunks": 0,
        }

        for chunk_dir in chunk_dirs:
            footer_path = chunk_dir / "chunk_footer.json"
            if not footer_path.exists():
                continue
            try:
                with open(footer_path, "r") as f:
                    footer = json.load(f)
            except Exception:
                continue

            row = {
                "chunk": chunk_dir.name,
                "status": footer.get("status"),
                "exit_attempts_total": footer.get("exit_attempts_total", 0) or 0,
                "exit_request_close_called": footer.get("exit_request_close_called", 0) or 0,
                "exit_request_close_accepted": footer.get("exit_request_close_accepted", 0) or 0,
                "exit_summary_logged": footer.get("exit_summary_logged", 0) or 0,
                "exit_event_rows_written": footer.get("exit_event_rows_written", 0) or 0,
                "force_close_attempts_replay_end": footer.get("force_close_attempts_replay_end", 0) or 0,
                "force_close_logged_replay_end": footer.get("force_close_logged_replay_end", 0) or 0,
                "force_close_attempts_replay_eof": footer.get("force_close_attempts_replay_eof", 0) or 0,
                "force_close_logged_replay_eof": footer.get("force_close_logged_replay_eof", 0) or 0,
                "open_trades_start_of_chunk": footer.get("open_trades_start_of_chunk"),
                "open_trades_end_of_chunk": footer.get("open_trades_end_of_chunk"),
                "open_to_closed_within_chunk": footer.get("open_to_closed_within_chunk", 0) or 0,
                "replay_end_or_eof_triggered": bool(footer.get("replay_end_or_eof_triggered", False)),
                "accounting_close_enabled": bool(footer.get("accounting_close_enabled", False)),
            }
            per_chunk.append(row)

            for k in [
                "exit_attempts_total",
                "exit_request_close_called",
                "exit_request_close_accepted",
                "exit_summary_logged",
                "exit_event_rows_written",
                "force_close_attempts_replay_end",
                "force_close_logged_replay_end",
                "force_close_attempts_replay_eof",
                "force_close_logged_replay_eof",
                "open_to_closed_within_chunk",
            ]:
                totals[k] += int(row.get(k, 0) or 0)

            if row.get("open_trades_start_of_chunk") is not None:
                totals["open_trades_start_of_chunk"] += int(row.get("open_trades_start_of_chunk") or 0)
            if row.get("open_trades_end_of_chunk") is not None:
                totals["open_trades_end_of_chunk"] += int(row.get("open_trades_end_of_chunk") or 0)
            if row.get("replay_end_or_eof_triggered"):
                totals["replay_end_or_eof_triggered_chunks"] += 1
            if row.get("accounting_close_enabled"):
                totals["accounting_close_enabled_chunks"] += 1

        summary = {
            "output_dir": str(output_dir),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunks_seen": len(per_chunk),
            "totals": totals,
            "per_chunk": per_chunk[-50:],  # cap for size
        }
        # TRUTH contract (C1): accepted close implies journaling evidence
        try:
            accepted_total = int((totals.get("exit_request_close_accepted") or 0))
            journal_evidence_total = int((totals.get("exit_event_rows_written") or 0)) + int((totals.get("exit_summary_logged") or 0))
            truth_ok = True
            truth_reason = None
            if accepted_total > 0 and journal_evidence_total <= 0:
                truth_ok = False
                truth_reason = "EXIT_ACCEPTED_BUT_NOT_JOURNALED"
            summary["truth_exit_journal_ok"] = truth_ok
            summary["truth_exit_journal_fail_reason"] = truth_reason
        except Exception:
            summary["truth_exit_journal_ok"] = None
            summary["truth_exit_journal_fail_reason"] = "ERROR_COMPUTING_TRUTH_EXIT_JOURNAL_FIELDS"

        from gx1.utils.atomic_json import atomic_write_json
        json_path = output_dir / "EXIT_COVERAGE_SUMMARY.json"
        atomic_write_json(json_path, summary)

        # Markdown summary
        md_lines = [
            "# EXIT COVERAGE SUMMARY",
            "",
            f"**Output dir:** `{output_dir}`",
            f"**Generated:** {summary['timestamp']}",
            "",
            "## Totals",
            "",
        ]
        for k, v in totals.items():
            md_lines.append(f"- **{k}:** {v}")
        md_lines.extend([
            "",
            "## Per-chunk (last 50)",
            "",
            "| chunk | status | open_start | open_end | exit_summary_logged | exit_event_rows_written | force_end_logged | force_eof_logged | accounting_close |",
            "|------|--------|------------|----------|---------------------|------------------------|------------------|------------------|------------------|",
        ])
        for r in per_chunk[-50:]:
            md_lines.append(
                f"| {r.get('chunk')} | {r.get('status')} | {r.get('open_trades_start_of_chunk')} | {r.get('open_trades_end_of_chunk')} | "
                f"{r.get('exit_summary_logged')} | {r.get('exit_event_rows_written')} | {r.get('force_close_logged_replay_end')} | "
                f"{r.get('force_close_logged_replay_eof')} | {r.get('accounting_close_enabled')} |"
            )
        md_path = output_dir / "EXIT_COVERAGE_SUMMARY.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))
    except Exception as e:
        log.warning(f"[EXIT_COVERAGE] Failed to write master summary: {e}")


def main():
    # CRITICAL: Log immediately when main() starts (for wrapper verification)
    print("[MASTER_START] replay master started", flush=True)
    
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

    parser.add_argument("--chunks", type=int, default=None, help="Number of chunks to split data into (default: same as workers)")
    parser.add_argument("--max-procs", type=int, default=None, help="Hard cap on concurrent subprocesses (default: workers)")
    parser.add_argument("--min-free-mem-gb", type=float, default=6.0, help="Minimum free memory to keep available (GB)")
    parser.add_argument("--mem-per-proc-gb", type=float, default=3.0, help="Estimated memory per subprocess (GB)")
    parser.add_argument("--mem-check-interval-sec", type=int, default=2, help="Interval between memory checks (seconds)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: reports/replay_eval/{run_id})")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (default: timestamp)")
    parser.add_argument("--slice-head", type=int, default=None, help="Use only first N bars (deterministic)")
    parser.add_argument("--days", type=int, default=None, help="Use only first N days (deterministic, M5=288 bars/day)")
    parser.add_argument("--start-ts", type=str, default=None, help="Slice start timestamp (inclusive, ISO8601)")
    parser.add_argument("--end-ts", type=str, default=None, help="Slice end timestamp (inclusive, ISO8601)")
    parser.add_argument("--min-post-warmup-bars-per-chunk", type=int, default=None, help="Minimum bars after warmup per chunk (will coalesce chunks to meet requirement)")
    parser.add_argument("--abort-after-first-chunk", type=int, default=0, help="Stop after first chunk completes (default: 0)")
    parser.add_argument("--abort-after-n-bars-per-chunk", type=int, default=None, help="Stop each chunk after N bars processed (for fast verification, default: None)")
    parser.add_argument("--dry-run-prebuilt-check", type=int, default=0, help="Only load prebuilt + run checks, then exit (default: 0)")
    parser.add_argument("--prebuilt-parquet", type=Path, default=None, help="Absolute path to prebuilt features parquet file (required if prebuilt enabled)")
    parser.add_argument("--bundle-dir", type=Path, default=None, help="Override bundle_dir from policy (absolute path, highest priority)")
    parser.add_argument("--selftest-only", action="store_true", help="Run only worker self-test (smoke_open) and exit (for debugging)")
    parser.add_argument("--chunk-local-padding-days", type=int, default=None, help="Chunk-local padding days (TRUTH/SMOKE only, default: 7 if TRUTH/SMOKE, else 0)")
    
    args = parser.parse_args()
    
    # Log workers and PID after parsing args (for wrapper verification)
    print(f"[MASTER_START] workers={args.workers} pid={os.getpid()} cpu_count={os.cpu_count()}", flush=True)
    
    # DEL 7: Set GX1_SELFTEST_ONLY env var if --selftest-only is set
    if args.selftest_only:
        os.environ["GX1_SELFTEST_ONLY"] = "1"
        log.info("[MASTER] --selftest-only: Workers will exit after smoke test")

    # Determine if TRUTH/SMOKE mode (must be available BEFORE preflight gates)
    run_mode = os.getenv("GX1_RUN_MODE", "").upper()
    is_truth_or_smoke = run_mode in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"

    # Resolve output directory using helper (with TRUTH/SMOKE validation) EARLY
    from gx1.utils.output_dir import resolve_output_dir

    output_dir_str = str(args.output_dir) if args.output_dir else None
    args.output_dir = resolve_output_dir(
        kind="replay_eval",
        prefix="REPLAY_EVAL",
        output_dir=output_dir_str,
        truth_or_smoke=is_truth_or_smoke,
    )

    # TRUTH BANLIST: fail-fast on forbidden fallback envs / legacy module imports (best-effort).
    # NOTE: This does not replace structural removal of legacy code; it is an extra guardrail.
    from gx1.utils.truth_banlist import assert_truth_banlist_clean
    assert_truth_banlist_clean(output_dir=args.output_dir, stage="replay_eval_gated_parallel:master_start")

    # B0.2: MASTER_EARLY.json Stage 1 - After args + output_dir resolve
    try:
        resolved_paths = {
            "data": str(args.data) if args.data else None,
            "policy": str(args.policy) if args.policy else None,
            "output_dir": str(args.output_dir),
            "bundle_dir": str(args.bundle_dir) if args.bundle_dir else None,
        }
        write_master_early_capsule(
            output_dir=args.output_dir,
            stage="args_resolved",
            argv=sys.argv,
            resolved_paths=resolved_paths,
            run_mode=run_mode,
            workers_requested=args.workers,
        )
    except Exception as e:
        log.warning(f"[MASTER_EARLY] Failed to write stage 1 capsule: {e}")

    # ------------------------------------------------------------------------
    # SYNTAX COMPILE GATE (TRUTH/SMOKE ONLY)
    #
    # Must run before expensive work / worker start.
    # Always writes SYNTAX_AUDIT_REPORT.md (output_dir preferred, /tmp fallback).
    # On FAIL: writes SYNTAX_FATAL.json atomically + exits with code 2.
    # ------------------------------------------------------------------------
    try:
        from gx1.utils.syntax_gate import run_syntax_gate_or_fatal

        run_syntax_gate_or_fatal(output_dir=args.output_dir, truth_or_smoke=is_truth_or_smoke)
    except SystemExit:
        raise
    except Exception as e:
        # If the gate itself fails unexpectedly, treat as fatal in TRUTH/SMOKE.
        if is_truth_or_smoke:
            print(f"[SYNTAX_GATE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            sys.exit(2)
        log.warning(f"[SYNTAX_GATE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # ------------------------------------------------------------------------
    # ENV IDENTITY GATE (TRUTH/SMOKE ONLY)
    #
    # Proves which Python/venv is active and hard-fails (exit 2) if pandas is missing.
    # Does NOT install deps; it only reports identity and aborts with a clear capsule.
    # ------------------------------------------------------------------------
    try:
        from gx1.utils.env_identity_gate import run_env_identity_gate_or_fatal

        run_env_identity_gate_or_fatal(output_dir=args.output_dir, truth_or_smoke=is_truth_or_smoke)
    except SystemExit:
        raise
    except Exception as e:
        if is_truth_or_smoke:
            print(f"[ENV_IDENTITY_GATE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            sys.exit(2)
        log.warning(f"[ENV_IDENTITY_GATE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # ------------------------------------------------------------------------
    # BUNDLE IDENTITY GATE (TRUTH/SMOKE ONLY)
    # ------------------------------------------------------------------------
    # Enforces canonical bundle directory and verifies bundle SHA256 matches expected.
    # Hard-fails (exit code 2) if GX1_CANONICAL_BUNDLE_DIR is not set, bundle dir mismatch, or SHA mismatch.
    # ------------------------------------------------------------------------
    try:
        from gx1.utils.bundle_identity_gate import run_bundle_identity_gate_or_fatal

        run_bundle_identity_gate_or_fatal(
            output_dir=args.output_dir,
            truth_or_smoke=is_truth_or_smoke,
            policy_path=args.policy,
            bundle_dir_override=args.bundle_dir,
        )
    except SystemExit:
        raise
    except Exception as e:
        if is_truth_or_smoke:
            print(f"[BUNDLE_IDENTITY_GATE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            sys.exit(2)
        log.warning(f"[BUNDLE_IDENTITY_GATE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # ------------------------------------------------------------------------
    # POLICY IDENTITY GATE (TRUTH/SMOKE ONLY)
    # ------------------------------------------------------------------------
    # Enforces canonical policy file and verifies policy SHA256 matches expected.
    # Hard-fails (exit code 2) if policy file does not exist or SHA256 mismatch.
    # ------------------------------------------------------------------------
    try:
        from gx1.utils.policy_identity_gate import run_policy_identity_gate_or_fatal

        run_policy_identity_gate_or_fatal(
            output_dir=args.output_dir,
            truth_or_smoke=is_truth_or_smoke,
            policy_path=args.policy,
        )
    except SystemExit:
        raise
    except Exception as e:
        if is_truth_or_smoke:
            print(f"[POLICY_IDENTITY_GATE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            sys.exit(2)
        log.warning(f"[POLICY_IDENTITY_GATE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # ------------------------------------------------------------------------
    # RAW CANDLES SCHEMA GATE (TRUTH/SMOKE ONLY)
    # ------------------------------------------------------------------------
    # Verifies that --data (raw candles parquet) contains OHLC columns.
    # NOTE: Prebuilt features (--prebuilt-parquet) do NOT need OHLC.
    # Hard-fails (exit code 2) if raw candles are missing OHLC.
    # ------------------------------------------------------------------------
    if args.data:
        try:
            from gx1.utils.raw_candles_schema_gate import run_raw_candles_schema_gate_or_fatal

            raw_data_path = Path(args.data)
            run_raw_candles_schema_gate_or_fatal(
                output_dir=args.output_dir,
                truth_or_smoke=is_truth_or_smoke,
                raw_data_path=raw_data_path,
            )
        except SystemExit:
            raise
        except Exception as e:
            if is_truth_or_smoke:
                print(f"[RAW_CANDLES_SCHEMA_GATE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                sys.exit(2)
            log.warning(f"[RAW_CANDLES_SCHEMA_GATE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # ------------------------------------------------------------------------
    # PREBUILT FEATURE SCHEMA GATE (TRUTH/SMOKE ONLY)
    # ------------------------------------------------------------------------
    # Validates --prebuilt-parquet contains:
    # - timestamp column (ts/time/timestamp)
    # - required minimal feature set (SSoT guardrail)
    # - no banned prob_* columns
    # ------------------------------------------------------------------------
    # Prebuilt path comes from --prebuilt-parquet only (no fallback / no auto-discovery).
    prebuilt_path_for_gate = Path(args.prebuilt_parquet) if args.prebuilt_parquet else None
    
    if prebuilt_path_for_gate and prebuilt_path_for_gate.exists():
        try:
            from gx1.utils.prebuilt_feature_schema_gate import run_prebuilt_feature_schema_gate_or_fatal

            run_prebuilt_feature_schema_gate_or_fatal(
                output_dir=args.output_dir,
                truth_or_smoke=is_truth_or_smoke,
                prebuilt_path=prebuilt_path_for_gate,
            )
        except SystemExit:
            raise
        except Exception as e:
            if is_truth_or_smoke:
                print(f"[PREBUILT_FEATURE_SCHEMA_GATE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                sys.exit(2)
            log.warning(f"[PREBUILT_FEATURE_SCHEMA_GATE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # ------------------------------------------------------------------------
    # PREBUILT IDENTITY GATE (TRUTH/SMOKE ONLY)
    # ------------------------------------------------------------------------
    # Enforces canonical prebuilt symlink and verifies schema is clean (no prob_* columns).
    # Hard-fails (exit code 2) if prebuilt is not canonical or contains prob_* columns.
    # NOTE: This checks the PREBUILT file (from --prebuilt-parquet or env), not --data.
    # ------------------------------------------------------------------------
    if prebuilt_path_for_gate and prebuilt_path_for_gate.exists():
        try:
            from gx1.utils.prebuilt_identity_gate import run_prebuilt_identity_gate_or_fatal

            run_prebuilt_identity_gate_or_fatal(
                prebuilt_path=prebuilt_path_for_gate,
                output_dir=args.output_dir,
                truth_or_smoke=is_truth_or_smoke,
            )
        except SystemExit:
            raise
        except Exception as e:
            if is_truth_or_smoke:
                print(f"[PREBUILT_IDENTITY_GATE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                sys.exit(2)
            log.warning(f"[PREBUILT_IDENTITY_GATE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # B0.2: MASTER_EARLY.json Stage 2 - After all gates (syntax, env, bundle, prebuilt)
    try:
        resolved_paths = {
            "data": str(args.data) if args.data else None,
            "policy": str(args.policy) if args.policy else None,
            "output_dir": str(args.output_dir),
            "bundle_dir": str(args.bundle_dir) if args.bundle_dir else None,
        }
        write_master_early_capsule(
            output_dir=args.output_dir,
            stage="gates_passed",
            argv=sys.argv,
            resolved_paths=resolved_paths,
            run_mode=run_mode,
            workers_requested=args.workers,
        )
    except Exception as e:
        log.warning(f"[MASTER_EARLY] Failed to write stage 2 capsule: {e}")

    # In TRUTH/SMOKE: forbid local OS imports (after gates so we always get capsules first)
    if is_truth_or_smoke:
        _assert_no_local_os_imports()
    
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
    
    # C) Robust output_dir init: resolve to canonical path and validate
    try:
        output_dir_resolved = args.output_dir.resolve()
        args.output_dir = output_dir_resolved
        
        # Hard assert: output_dir must be under GX1_DATA_ROOT/reports
        from gx1.utils.output_dir import resolve_gx1_data_root
        gx1_data_root = resolve_gx1_data_root()
        gx1_data_root_resolved = gx1_data_root.resolve()
        reports_dir = gx1_data_root_resolved / "reports"
        
        if not str(output_dir_resolved).startswith(str(reports_dir)):
            raise RuntimeError(
                f"[OUTPUT_DIR_INIT] Output directory must be under {reports_dir}, "
                f"got: {output_dir_resolved}"
            )
        
        # Pre-create output_dir and chunks subdirectory
        # In TRUTH/SMOKE mode: fail if exists with artifacts (no reuse)
        # In other modes: allow exist_ok=True
        # BUGFIX: Check if dir exists before mkdir to avoid race condition
        # NOTE: MASTER_EARLY.json may have been written already (by write_master_early_capsule),
        # so we exclude it from artifact check
        if output_dir_resolved.exists():
            # Directory already exists - check if it has artifacts (excluding MASTER_EARLY.json)
            json_files = list(output_dir_resolved.glob("*.json"))
            # Exclude MASTER_EARLY.json from artifact check (it's written before OUTPUT_DIR_INIT)
            json_files = [f for f in json_files if f.name != "MASTER_EARLY.json"]
            has_artifacts = any(output_dir_resolved.glob("chunk_*")) or len(json_files) > 0
            if is_truth_or_smoke and has_artifacts:
                raise RuntimeError(
                    f"[OUTPUT_DIR_INIT] Output directory already exists with artifacts: {output_dir_resolved}. "
                    f"This violates TRUTH/SMOKE mode: no reuse of output-dir. "
                    f"Remove or rename the existing directory before starting a new run."
                )
            # Empty directory or non-TRUTH/SMOKE: allow reuse, ensure it exists
            output_dir_resolved.mkdir(parents=True, exist_ok=True)
        else:
            # Directory doesn't exist - create it
            output_dir_resolved.mkdir(parents=True, exist_ok=True)
        
        # Pre-create chunks subdirectory
        chunks_dir = output_dir_resolved / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
    except OSError as e:
        # Write OS error capsule before failing
        write_master_os_error(
            op="output_dir_init",
            paths={"output_dir": args.output_dir, "gx1_data_root": gx1_data_root_resolved if 'gx1_data_root_resolved' in locals() else None},
            exc=e,
            output_dir=None,  # Can't use output_dir since init failed
        )
        log.error(f"[OUTPUT_DIR_INIT] Failed to create output directory: {e}")
        sys.exit(2)
    except Exception as e:
        # Other errors (validation, etc.)
        log.error(f"[OUTPUT_DIR_INIT] Failed: {e}")
        raise
    
    # Generate run_id (use local import to avoid scoping issues)
    from gx1.utils.dt_module import strftime_now as _dt_strftime_now_runid
    run_id = args.run_id or args.output_dir.name or _dt_strftime_now_runid("%Y%m%d_%H%M%S")
    
    # Resolve all other paths to absolute
    args.policy = args.policy.resolve()
    args.data = args.data.resolve()
    if args.prebuilt_parquet:
        args.prebuilt_parquet = args.prebuilt_parquet.resolve()
    
    # C) Write MASTER_START.json early (atomic) as boot marker
    try:
        from gx1.utils.output_dir import resolve_gx1_data_root
        gx1_data_root = resolve_gx1_data_root()
        master_start_data = {
            "timestamp": dt_now_iso(),
            "run_id": run_id,
            "output_dir": str(args.output_dir),
            "output_dir_realpath": str(args.output_dir.resolve()),
            "gx1_data_root_realpath": str(gx1_data_root.resolve()),
            "argv": sys.argv.copy(),
            "sys_executable": sys.executable,
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
        }
        write_json_atomic(args.output_dir / "MASTER_START.json", master_start_data, output_dir=args.output_dir)
        log.info(f"[MASTER_START] Wrote MASTER_START.json to {args.output_dir / 'MASTER_START.json'}")
    except Exception as e:
        log.warning(f"[MASTER_START] Failed to write MASTER_START.json (non-fatal): {e}")
        # Continue anyway - this is just a boot marker
    
    # DOCTOR PREFLIGHT: Run gx1 doctor before expensive operations (TRUTH/SMOKE only)
    # Note: is_truth_or_smoke already determined above
    if is_truth_or_smoke:
        try:
            from gx1.utils.preflight_doctor import run_gx1_doctor_or_fatal
            run_gx1_doctor_or_fatal(
                strict=True,  # Always strict in TRUTH/SMOKE
                truth_or_smoke=True,
                output_dir=args.output_dir,
            )
        except RuntimeError as doctor_error:
            # Hard-fail: doctor found blocking issues
            log.error(f"[DOCTOR_FATAL] {doctor_error}")
            raise
        except Exception as doctor_import_error:
            # Non-fatal: if doctor can't be imported, log warning but continue
            log.warning(f"[DOCTOR] Failed to import/run gx1 doctor (non-fatal): {doctor_import_error}")
    
    # LEGACY_GUARD: Check policy file and output-dir after path resolution
    # Note: sys is imported at top level, so it's available here
    try:
        from gx1.runtime.legacy_guard import check_policy_for_legacy, assert_no_legacy_mode_enabled
        check_policy_for_legacy(args.policy)
        # Check output-dir and bundle-dir with resolved paths
        bundle_dir_resolved = args.bundle_dir.resolve() if args.bundle_dir else None
        # Use sys.argv from top-level import
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

    # Ritual warmup padding (TRUTH/SMOKE replay only, explicit opt-in)
    # Compute requested/effective windows BEFORE RUN_IDENTITY is written.
    parsed_start_ts = None
    parsed_end_ts = None
    if args.start_ts:
        parsed_start_ts = pd.to_datetime(args.start_ts)
    if args.end_ts:
        parsed_end_ts = pd.to_datetime(args.end_ts)
    
    ritual_padding_days_raw = os.getenv("GX1_RITUAL_WARMUP_PADDING_DAYS", "0").strip()
    ritual_padding_days = 0
    if ritual_padding_days_raw:
        try:
            ritual_padding_days = int(ritual_padding_days_raw)
        except ValueError as e:
            raise ValueError(
                f"GX1_RITUAL_WARMUP_PADDING_DAYS must be int >= 0 (got: {ritual_padding_days_raw})"
            ) from e
    if ritual_padding_days < 0:
        raise ValueError(
            f"GX1_RITUAL_WARMUP_PADDING_DAYS must be int >= 0 (got: {ritual_padding_days})"
        )
    if ritual_padding_days > 60:
        raise ValueError(
            f"GX1_RITUAL_WARMUP_PADDING_DAYS too large (max 60, got: {ritual_padding_days})"
        )
    
    # Chunk-local padding (TRUTH/SMOKE only, default: 7 if TRUTH/SMOKE, else 0)
    chunk_local_padding_days = args.chunk_local_padding_days
    if chunk_local_padding_days is None:
        chunk_local_padding_days = 7 if is_truth_or_smoke else 0
    elif chunk_local_padding_days > 0 and not is_truth_or_smoke:
        log.warning(f"[CHUNK_LOCAL_PADDING] chunk_local_padding_days={chunk_local_padding_days} but not TRUTH/SMOKE mode, ignoring")
        chunk_local_padding_days = 0
    elif chunk_local_padding_days < 0:
        raise ValueError(f"--chunk-local-padding-days must be >= 0 (got: {chunk_local_padding_days})")
    elif chunk_local_padding_days > 30:
        raise ValueError(f"--chunk-local-padding-days too large (max 30, got: {chunk_local_padding_days})")
    
    if chunk_local_padding_days > 0:
        log.info(f"[CHUNK_LOCAL_PADDING] Enabled: {chunk_local_padding_days} days per chunk (TRUTH/SMOKE mode)")
    
    truth_or_smoke_or_telemetry = is_truth_or_smoke or os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
    requested_eval_start_ts = parsed_start_ts
    requested_eval_end_ts = parsed_end_ts
    effective_replay_start_ts = parsed_start_ts
    effective_replay_end_ts = parsed_end_ts
    if ritual_padding_days > 0 and truth_or_smoke_or_telemetry and parsed_start_ts is not None:
        effective_replay_start_ts = parsed_start_ts - pd.Timedelta(days=ritual_padding_days)
        log.info(
            "[RITUAL_WARMUP_PADDING] requested_start=%s requested_end=%s padding_days=%d effective_replay_start=%s",
            parsed_start_ts,
            parsed_end_ts,
            ritual_padding_days,
            effective_replay_start_ts,
        )
    elif ritual_padding_days > 0 and not truth_or_smoke_or_telemetry:
        log.info(
            "[RITUAL_WARMUP_PADDING] padding ignored (not TRUTH/SMOKE/telemetry). "
            "requested_start=%s requested_end=%s padding_days=%d",
            parsed_start_ts,
            parsed_end_ts,
            ritual_padding_days,
        )

    # RUN_IDENTITY: Write once in master to output root (hard-fail in TRUTH/smoke)
    # Note: is_truth_or_smoke already determined above
    hard_fail_identity = is_truth_or_smoke
    policy_id = None
    policy_sha256 = None
    bundle_dir_for_identity = None
    bundle_dir_source = None
    
    try:
        import yaml
        with open(args.policy, "r") as f:
            policy = yaml.safe_load(f) or {}
        policy_id = (
            policy.get("policy_id")
            or policy.get("replay_config", {}).get("policy_id")
            or args.policy.stem
        )
        bundle_dir_str = (
            policy.get("entry_models", {}).get("v10_ctx", {}).get("bundle_dir")
            or policy.get("entry_models", {}).get("v10", {}).get("bundle_dir")
        )
        if bundle_dir_str:
            bundle_dir_for_identity = Path(bundle_dir_str)
            if not bundle_dir_for_identity.is_absolute():
                bundle_dir_for_identity = args.policy.parent / bundle_dir_for_identity
            bundle_dir_for_identity = bundle_dir_for_identity.resolve()
            bundle_dir_source = "policy"
    except Exception:
        policy_id = policy_id or args.policy.stem
    
    if is_truth_or_smoke:
        # ONE truth only: require GX1_CANONICAL_BUNDLE_DIR and forbid alternate selection paths.
        if args.bundle_dir is not None:
            raise RuntimeError("[TRUTH_FAIL] BUNDLE_DIR_CLI_FORBIDDEN: use GX1_CANONICAL_BUNDLE_DIR only in TRUTH/SMOKE")
        canonical = os.getenv("GX1_CANONICAL_BUNDLE_DIR") or ""
        if not canonical:
            # Ambiguity guard: report candidates if multiple exist.
            candidates = []
            try:
                gx1_data_root_env = os.getenv("GX1_DATA_DIR") or os.getenv("GX1_DATA_ROOT") or os.getenv("GX1_DATA")
                if gx1_data_root_env:
                    gx1_data_root = Path(gx1_data_root_env).expanduser().resolve()
                    cand_dir = gx1_data_root / "models" / "models" / "entry_v10_ctx"
                    if cand_dir.exists():
                        candidates = sorted([str(p.resolve()) for p in cand_dir.glob("XGB_V13_REFINED3_PRUNE14_*/MASTER_MODEL_LOCK.json")])
            except Exception:
                candidates = []
            fatal = {
                "status": "FAIL",
                "error": "AMBIGUOUS_TRUTH",
                "message": "TRUTH requires explicit GX1_CANONICAL_BUNDLE_DIR (absolute).",
                "candidates_found_n": len(candidates),
                "candidates_found": candidates[:50],
            }
            try:
                from gx1.utils.json_atomic import write_json_atomic
                write_json_atomic(args.output_dir / "BUNDLE_DIR_FATAL.json", fatal, output_dir=args.output_dir)
            except Exception:
                pass
            raise RuntimeError("[TRUTH_FAIL] TRUTH_NO_FALLBACK: missing GX1_CANONICAL_BUNDLE_DIR")
        bundle_dir_for_identity = Path(canonical).expanduser().resolve()
        bundle_dir_source = "env:GX1_CANONICAL_BUNDLE_DIR"
        if not bundle_dir_for_identity.is_absolute():
            raise RuntimeError(f"[TRUTH_FAIL] GX1_CANONICAL_BUNDLE_DIR must be absolute: {bundle_dir_for_identity}")
        if not bundle_dir_for_identity.exists():
            raise RuntimeError(f"[TRUTH_FAIL] GX1_CANONICAL_BUNDLE_DIR does not exist: {bundle_dir_for_identity}")
    else:
        if args.bundle_dir:
            bundle_dir_for_identity = args.bundle_dir.resolve()
            bundle_dir_source = "cli"
        elif os.getenv("GX1_BUNDLE_DIR"):
            bundle_dir_for_identity = Path(os.getenv("GX1_BUNDLE_DIR")).resolve()
            bundle_dir_source = "env"
    
    try:
        if args.policy.exists():
            sha256_hash = hashlib.sha256()
            with open(args.policy, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            policy_sha256 = sha256_hash.hexdigest()
    except Exception:
        policy_sha256 = None
    
    if hard_fail_identity:
        if not policy_id:
            raise RuntimeError("RUN_IDENTITY_FAIL: policy_id is missing")
        if not policy_sha256:
            raise RuntimeError("RUN_IDENTITY_FAIL: policy_sha256 is missing")
        if bundle_dir_for_identity is None or not bundle_dir_for_identity.exists():
            raise RuntimeError("RUN_IDENTITY_FAIL: bundle_dir is missing or does not exist")
    
    # RUN_IDENTITY: Create and write (hard-fail in TRUTH/SMOKE if policy loading fails)
    # Policy loading happens inside create_run_identity via ensure_xgb_policy_fields_loaded()
    # Do NOT catch exceptions here - let them propagate in TRUTH/SMOKE mode
    from gx1.runtime.run_identity import create_run_identity
    # Get entry_model_id from replay_config (required for replay)
    entry_model_id = None
    try:
        if policy:
            entry_model_id = policy.get("replay_config", {}).get("entry_model_id")
    except Exception:
        pass
    # Signal-only truth: session tokens are forbidden (no legacy snap extensions).
    if os.getenv("GX1_TRANSFORMER_SESSION_TOKEN", "0") == "1" and is_truth_or_smoke:
        raise RuntimeError("[TRUTH_FAIL] SESSION_TOKENS_FORBIDDEN: GX1_TRANSFORMER_SESSION_TOKEN=1 under signal-only truth")
    session_tokens_enabled = False
    diagnostic_force_eval_sessions = os.getenv("GX1_DIAGNOSTIC_FORCE_EVAL_SESSIONS", "")
    diagnostic_enabled = bool(diagnostic_force_eval_sessions) and os.getenv("GX1_TRUTH_TELEMETRY", "0") == "1"
    diagnostic_bypass_gate_env = os.getenv("GX1_DIAGNOSTIC_BYPASS_GATE", "").strip().lower()
    diagnostic_bypassed_gate = None
    diagnostic_mode = None
    def _resolve_auto_bypass_gate(baseline_output_dir: str, force_sessions: str) -> str:
        baseline_dir = Path(baseline_output_dir)
        if not baseline_dir.exists():
            raise RuntimeError(f"[DIAGNOSTIC_AUTO] Baseline output dir not found: {baseline_dir}")
        candidates = sorted(
            baseline_dir.glob("SESSION_FUNNEL_ROOT_CAUSE_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"[DIAGNOSTIC_AUTO] No SESSION_FUNNEL_ROOT_CAUSE_*.json in {baseline_dir}")
        with open(candidates[0], "r") as f:
            report = json.load(f)
        first_kill_summary = report.get("first_kill_summary", {})
        sessions = [s.strip().upper() for s in force_sessions.split(",") if s.strip()]
        stage_to_gate = {
            "hard_eligibility_pass": "hard_eligibility",
            "soft_eligibility_pass": "soft_eligibility",
            "session_gate_pass": "session_gate",
            "vol_guard_pass": "vol_guard",
            "score_gate_pass": "score_gate",
            "pregate_pass": "pregate",
            "eval_called": "score_gate",
            "predict_entered": "score_gate",
            "pre_call": "score_gate",
            "forward_calls": "score_gate",
        }
        gate_order = ["hard_eligibility", "soft_eligibility", "session_gate", "vol_guard", "score_gate", "pregate"]
        resolved_gates = []
        for session in sessions:
            stage = first_kill_summary.get(session, {}).get("first_kill_stage")
            if not stage:
                raise RuntimeError(f"[DIAGNOSTIC_AUTO] Missing first_kill_stage for session={session}")
            gate = stage_to_gate.get(stage)
            if not gate:
                raise RuntimeError(f"[DIAGNOSTIC_AUTO] Unmapped first_kill_stage={stage} for session={session}")
            resolved_gates.append(gate)
        if not resolved_gates:
            raise RuntimeError("[DIAGNOSTIC_AUTO] No resolved gates for AUTO bypass")
        resolved_gate = min(resolved_gates, key=lambda g: gate_order.index(g))
        log.info("[DIAGNOSTIC_AUTO] Resolved bypass gate=%s (sessions=%s)", resolved_gate, sessions)
        return resolved_gate
    if diagnostic_enabled:
        diagnostic_mode = "replay_only"
        if diagnostic_bypass_gate_env in ("", "auto"):
            baseline_output_dir = os.getenv("GX1_DIAGNOSTIC_BASELINE_OUTPUT_DIR", "").strip()
            if not baseline_output_dir:
                raise RuntimeError(
                    "[DIAGNOSTIC_AUTO] GX1_DIAGNOSTIC_BASELINE_OUTPUT_DIR must be set when GX1_DIAGNOSTIC_BYPASS_GATE=AUTO"
                )
            diagnostic_bypassed_gate = _resolve_auto_bypass_gate(baseline_output_dir, diagnostic_force_eval_sessions)
            os.environ["GX1_DIAGNOSTIC_BYPASS_GATE"] = diagnostic_bypassed_gate
        else:
            diagnostic_bypassed_gate = diagnostic_bypass_gate_env
    # Signal-only truth: dims are from signal bridge contract.
    from gx1.contracts.signal_bridge_v1 import CONTRACT_SHA256 as SIGNAL_BRIDGE_SHA256, SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
    snap_dim_expected = int(SNAP_SIGNAL_DIM)
    snap_dim_effective = int(SNAP_SIGNAL_DIM)

    # XGB lock + schema manifest identity fields (SSoT)
    xgb_lock_path = None
    xgb_lock_file_sha256 = None
    xgb_lock_feature_contract_id = None
    xgb_lock_feature_list_sha256 = None
    xgb_lock_model_sha256 = None
    prebuilt_schema_manifest_path = None
    prebuilt_schema_manifest_sha256 = None
    try:
        if bundle_dir_for_identity is None:
            raise RuntimeError("bundle_dir_for_identity is None")
        xgb_lock_path = (bundle_dir_for_identity / "MASTER_MODEL_LOCK.json")
        if is_truth_or_smoke and not xgb_lock_path.exists():
            raise RuntimeError(f"MASTER_MODEL_LOCK.json missing: {xgb_lock_path}")
        if xgb_lock_path.exists():
            lock_obj = json.loads(xgb_lock_path.read_text(encoding="utf-8"))
            import hashlib as _hashlib
            xgb_lock_file_sha256 = _hashlib.sha256(xgb_lock_path.read_bytes()).hexdigest()
            xgb_lock_feature_contract_id = str(lock_obj.get("feature_contract_id") or "")
            xgb_lock_feature_list_sha256 = str(lock_obj.get("feature_list_sha256") or "")
            xgb_lock_model_sha256 = str(lock_obj.get("model_sha256") or "")
        if args.prebuilt_parquet:
            prebuilt_schema_manifest_path = str(Path(str(args.prebuilt_parquet)).with_suffix(".schema_manifest.json").resolve())
            psm = Path(prebuilt_schema_manifest_path)
            if is_truth_or_smoke and not psm.exists():
                raise RuntimeError(f"Prebuilt schema manifest missing: {psm}")
            if psm.exists():
                import hashlib as _hashlib
                prebuilt_schema_manifest_sha256 = _hashlib.sha256(psm.read_bytes()).hexdigest()
    except Exception as e:
        if is_truth_or_smoke:
            raise RuntimeError(f"[RUN_IDENTITY_FAIL] signal-only SSoT extraction failed: {e}") from e
    try:
        def _ts_to_iso(ts_value: Optional[pd.Timestamp]) -> Optional[str]:
            if ts_value is None:
                return None
            try:
                return ts_value.isoformat()
            except Exception:
                return str(ts_value)
        create_run_identity(
            output_dir=args.output_dir,
            policy_id=policy_id or "unknown",
            policy_sha256=policy_sha256 or "unknown",
            bundle_dir=bundle_dir_for_identity,
            bundle_dir_source=bundle_dir_source,
            prebuilt_path=args.prebuilt_parquet,
            allow_dirty=True,
            prebuilt_used=prebuilt_enabled and args.prebuilt_parquet is not None,
            entry_model_id=entry_model_id,  # REPLAY SSoT: Entry model ID from replay_config
            session_tokens_enabled=session_tokens_enabled,
            snap_dim_expected=snap_dim_expected,
            snap_dim_effective=snap_dim_effective,
            signal_bridge_contract_sha256=str(SIGNAL_BRIDGE_SHA256),
            seq_signal_dim=int(SEQ_SIGNAL_DIM),
            snap_signal_dim=int(SNAP_SIGNAL_DIM),
            xgb_lock_path=str(xgb_lock_path) if xgb_lock_path else None,
            xgb_lock_file_sha256=str(xgb_lock_file_sha256) if xgb_lock_file_sha256 else None,
            xgb_lock_feature_contract_id=str(xgb_lock_feature_contract_id) if xgb_lock_feature_contract_id else None,
            xgb_lock_feature_list_sha256=str(xgb_lock_feature_list_sha256) if xgb_lock_feature_list_sha256 else None,
            xgb_lock_model_sha256=str(xgb_lock_model_sha256) if xgb_lock_model_sha256 else None,
            prebuilt_schema_manifest_path=str(prebuilt_schema_manifest_path) if prebuilt_schema_manifest_path else None,
            prebuilt_schema_manifest_sha256=str(prebuilt_schema_manifest_sha256) if prebuilt_schema_manifest_sha256 else None,
            diagnostic_enabled=diagnostic_enabled,
            diagnostic_force_eval_sessions=diagnostic_force_eval_sessions or None,
            diagnostic_bypassed_gate=diagnostic_bypassed_gate,
            diagnostic_mode=diagnostic_mode,
            ritual_warmup_padding_days=ritual_padding_days,
            requested_eval_start_ts=_ts_to_iso(requested_eval_start_ts),
            requested_eval_end_ts=_ts_to_iso(requested_eval_end_ts),
            effective_replay_start_ts=_ts_to_iso(effective_replay_start_ts),
            effective_replay_end_ts=_ts_to_iso(effective_replay_end_ts),
        )
    except Exception as e:
        # In TRUTH/SMOKE mode, never swallow exceptions from RUN_IDENTITY creation
        if hard_fail_identity:
            raise
        # In non-TRUTH mode, log but continue (dev mode)
        log.warning(f"[RUN_IDENTITY] Failed to create RUN_IDENTITY (non-TRUTH mode): {e}")
    
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
    # No fallback / no env propagation: prebuilt path must be explicit via --prebuilt-parquet when enabled.
    prebuilt_path_log = str(args.prebuilt_parquet.resolve()) if args.prebuilt_parquet else "N/A"
    
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
    
    # Install SIGUSR2 handler for hang detection
    global MASTER_OUTPUT_DIR
    MASTER_OUTPUT_DIR = args.output_dir
    signal.signal(signal.SIGUSR2, _master_hang_handler)
    log.info("[MASTER] SIGUSR2 handler installed (hang detection)")
    
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
                        
                        # CRITICAL: On SIGTERM/STOP_REQUESTED, NEVER write RUN_COMPLETED.json.
                        # This is an abort path; write RUN_FAILED.json with a clear reason so downstream
                        # tooling does not treat partial exports as a successful run.
                        try:
                            chunks_completed_count = 0
                            try:
                                for chunk_idx in range(int(args.chunks)):
                                    chunk_footer_path = args.output_dir / f"chunk_{chunk_idx}" / "chunk_footer.json"
                                    if chunk_footer_path.exists():
                                        try:
                                            with open(chunk_footer_path, "r") as f:
                                                footer = json.load(f)
                                            if footer.get("status") == "ok":
                                                chunks_completed_count += 1
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            
                            completion_contract_path = args.output_dir / "RUN_FAILED.json"
                            completion_data = {
                                "status": "FAILED",
                                "reason": "SIGTERM_ABORTED",
                                "run_id": run_id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "chunks_submitted": int(args.chunks),
                                "chunks_completed": int(chunks_completed_count),
                                "chunks_incomplete": int(max(0, int(args.chunks) - int(chunks_completed_count))),
                                "written_by": "watchdog_thread",
                            }
                            
                            write_json_atomic(completion_contract_path, completion_data, output_dir=args.output_dir)
                            log.warning("[COMPLETION_CONTRACT] Writing RUN_FAILED.json (watchdog thread, SIGTERM_ABORTED)")
                            
                            # B) TRUTH invariant: trade journal contract
                            # If TRUTH and trades_created_total > 0, require trade journal
                            is_truth_or_smoke = os.getenv("GX1_RUN_MODE", "").upper() in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"
                            if is_truth_or_smoke:
                                # Aggregate trades_created from all chunk footers
                                trades_created_total = 0
                                chunk_dirs_for_check = sorted([d for d in args.output_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
                                for chunk_dir in chunk_dirs_for_check:
                                    footer_path = chunk_dir / "chunk_footer.json"
                                    if footer_path.exists():
                                        try:
                                            with open(footer_path, "r") as f:
                                                footer = json.load(f)
                                            if footer.get("status") == "ok":
                                                trades_created_total += footer.get("trades_created", 0) or footer.get("n_trades_closed", 0)
                                        except Exception:
                                            pass
                                
                                if trades_created_total > 0:
                                    # Check for trade journal
                                    trade_journal_index_path = args.output_dir / "trade_journal_index.csv"
                                    trade_journal_dir = args.output_dir / "trade_journal" / "trades"
                                    journal_exists = trade_journal_index_path.exists() or (trade_journal_dir.exists() and len(list(trade_journal_dir.glob("*.json"))) > 0)
                                    
                                    # Also check chunk journals (aggregated)
                                    chunk_journals_exist = False
                                    chunk_journals_with_data = 0
                                    for chunk_dir in chunk_dirs_for_check:
                                        chunk_journal = chunk_dir / "trade_journal" / "trade_journal_index.csv"
                                        if chunk_journal.exists():
                                            try:
                                                with open(chunk_journal, "r") as f:
                                                    lines = f.readlines()
                                                    if len(lines) > 1:  # Has data beyond header
                                                        chunk_journals_with_data += 1
                                                        chunk_journals_exist = True
                                            except Exception:
                                                pass
                                    
                                    if not journal_exists and not chunk_journals_exist:
                                        # FATAL: trades created but no journal
                                        fatal_capsule = {
                                            "timestamp": dt_now_iso(),
                                            "run_id": run_id,
                                            "fatal_reason": "TRADE_JOURNAL_MISSING",
                                            "trades_created_total": trades_created_total,
                                            "expected_paths": {
                                                "run_root_index": str(trade_journal_index_path),
                                                "run_root_trades_dir": str(trade_journal_dir),
                                                "chunk_pattern": "chunk_*/trade_journal/trade_journal_index.csv",
                                            },
                                            "journal_config_path": str(args.output_dir / "JOURNAL_CONFIG.json"),
                                            "message": f"TRUTH run created {trades_created_total} trades but trade journal is missing. This violates the trade journal contract.",
                                        }
                                        fatal_path = args.output_dir / "TRADE_JOURNAL_MISSING_FATAL.json"
                                        from gx1.utils.atomic_json import atomic_write_json
                                        atomic_write_json(fatal_path, fatal_capsule)
                                        log.error(f"[TRUTH_INVARIANT] ❌ FATAL: {fatal_capsule['message']}")
                                        log.error(f"[TRUTH_INVARIANT] See {fatal_path} for details")
                                        # Don't exit here - let watchdog complete, but mark as failed
                                        completion_data["status"] = "FAILED"
                                        completion_data["fatal_reason"] = "TRADE_JOURNAL_MISSING"
                                        write_json_atomic(completion_contract_path, completion_data, output_dir=args.output_dir)
                                    elif chunk_journals_exist:
                                        log.info(f"[TRUTH_INVARIANT] ✅ Trade journals found in {chunk_journals_with_data} chunks")
                            
                            # Append to run index ledger
                            try:
                                from gx1.utils.run_index import build_run_index_entry, append_run_index
                                from gx1.utils.output_dir import resolve_gx1_data_root
                                entry = build_run_index_entry(args.output_dir)
                                gx1_data_root = resolve_gx1_data_root()
                                append_run_index(gx1_data_root, entry)
                                log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir} (watchdog thread)")
                            except Exception as index_error:
                                log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
                        except Exception as completion_contract_error:
                            log.error(f"[COMPLETION_CONTRACT] ❌ Failed to write completion contract in watchdog: {completion_contract_error}")
                        
                        # Verify file was written
                        if perf_json_path.exists():
                            PERF_EXPORTED = True
                            log.warning(f"[WATCHDOG] ✅ Wrote perf_{run_id}.json from footers -> {perf_json_path}")
                            
                            # CRITICAL: Flush all logs before hard exit
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
                        
                        # Write stub file
                        stub_path = args.output_dir / f"perf_{run_id}_FAILED_EXPORT.json"
                        try:
                            with open(stub_path, "w") as f:
                                json.dump(error_stub, f, indent=2)
                            log.error(f"[WATCHDOG] Export failed -> wrote stub {stub_path}")
                        except Exception as stub_error:
                            log.error(f"[WATCHDOG] Failed to write error stub: {stub_error}")
                            
                    # CRITICAL: Flush all logs before hard exit (outside try/except to avoid import issues)
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
            # DEL 4A: Use GX1_DATA env vars for default paths (never relative to cwd)
            gx1_data_root_env = os.getenv("GX1_DATA_DIR") or os.getenv("GX1_DATA_ROOT")
            gx1_data_root = Path(gx1_data_root_env) if gx1_data_root_env else Path.home() / "GX1_DATA"
            gx1_data_root = gx1_data_root.expanduser().resolve()
            if gx1_data_root.name != "GX1_DATA":
                raise RuntimeError(f"GX1_DATA root must end with 'GX1_DATA': {gx1_data_root}")
            default_reports_root = Path(os.getenv("GX1_REPORTS_ROOT", str(gx1_data_root / "reports"))).resolve()
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
    
    # B0.2: MASTER_EARLY.json Stage 3 - Before chunk planning starts
    try:
        resolved_paths = {
            "data": str(args.data) if args.data else None,
            "policy": str(args.policy) if args.policy else None,
            "output_dir": str(args.output_dir),
            "bundle_dir": str(args.bundle_dir) if args.bundle_dir else None,
        }
        write_master_early_capsule(
            output_dir=args.output_dir,
            stage="before_chunk_planning",
            argv=sys.argv,
            resolved_paths=resolved_paths,
            run_mode=run_mode,
            workers_requested=args.workers,
        )
    except Exception as e:
        log.warning(f"[MASTER_EARLY] Failed to write stage 3 capsule: {e}")

    # --- Watchdog Setup (A) ---
    # Start run watchdog thread to monitor progress and detect stalls
    run_watchdog_done = threading.Event()
    # Store running_procs in a mutable container so watchdog can access it
    running_procs_container = {"procs": None}  # Will be set when running_procs is created
    
    try:
        from gx1.utils.watchdog import run_watchdog
        
        # Get running_procs function (will access running_procs_container)
        def get_running_procs_fn():
            """Helper to get running process PIDs if available."""
            if running_procs_container["procs"] is not None:
                try:
                    return [proc.pid for proc, _, _, _, _, _, _ in running_procs_container["procs"].values() if proc.poll() is None]
                except Exception:
                    return []
            return []
        
        # Start watchdog thread
        run_watchdog_thread = threading.Thread(
            target=run_watchdog,
            args=(
                args.output_dir,
                run_id,
                run_watchdog_done,
                get_running_procs_fn,
            ),
            kwargs={
                "stall_timeout_seconds": int(os.getenv("GX1_WATCHDOG_STALL_TIMEOUT_SEC", "120")),
                "progress_window_seconds": int(os.getenv("GX1_WATCHDOG_PROGRESS_WINDOW_SEC", "30")),
                "heartbeat_interval_seconds": float(os.getenv("GX1_WATCHDOG_HEARTBEAT_INTERVAL_SEC", "5.0")),
            },
            daemon=True,
            name="run-watchdog",
        )
        run_watchdog_thread.start()
        log.info("[WATCHDOG] Run watchdog thread started (monitors progress and detects stalls)")
    except Exception as watchdog_error:
        log.warning(f"[WATCHDOG] Failed to start run watchdog thread (non-fatal): {watchdog_error}")

    # ------------------------------------------------------------------------
    # PRE-FORK FREEZE GATE (TRUTH/SMOKE ONLY)
    # ------------------------------------------------------------------------
    # Verifies that all required code fixes/guards are present BEFORE any
    # workers are spawned. This prevents mixed-logic runs where some chunks
    # start before fixes are applied.
    # ------------------------------------------------------------------------
    if is_truth_or_smoke:
        log.info("[PRE_FORK_FREEZE] Verifying required code fixes before spawning workers...")
        try:
            from gx1.utils.prefork_freeze_gate import run_prefork_freeze_gate_or_fatal
            
            # Get git head SHA, policy SHA, bundle SHA for provenance
            git_head_sha = None
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(Path(__file__).parent.parent.parent),
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    git_head_sha = result.stdout.strip()
            except Exception:
                pass
            
            policy_sha = None
            if args.policy and Path(args.policy).exists():
                try:
                    with open(args.policy, "rb") as f:
                        policy_sha = hashlib.sha256(f.read()).hexdigest()
                except Exception:
                    pass
            
            bundle_sha = bundle_sha256 if 'bundle_sha256' in locals() else None
            
            run_prefork_freeze_gate_or_fatal(
                output_dir=args.output_dir,
                truth_or_smoke=is_truth_or_smoke,
                git_head_sha=git_head_sha,
                policy_sha=policy_sha,
                bundle_sha=bundle_sha,
            )
            log.info("[PRE_FORK_FREEZE] ✅ All required guards verified - workers can be spawned")
        except SystemExit:
            raise
        except Exception as e:
            if is_truth_or_smoke:
                print(f"[PRE_FORK_FREEZE] FATAL: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                sys.exit(2)
            log.warning(f"[PRE_FORK_FREEZE] Unexpected error (non-fatal, non-TRUTH/SMOKE): {e}")

    # Split data into chunks (supports deterministic time-window slicing for screening)
    # Check data timezone and adjust timestamps if needed
    # We'll do this inside split_data_into_chunks after loading data
    chunks = split_data_into_chunks(
        args.data,
        args.chunks if args.chunks is not None else args.workers,
        slice_head=args.slice_head,
        days=args.days,
        start_ts=effective_replay_start_ts,
        end_ts=effective_replay_end_ts,
    )
    
    # Coalesce chunks if min-post-warmup-bars-per-chunk is specified
    if args.min_post_warmup_bars_per_chunk is not None:
        chunks = coalesce_chunks_by_min_post_warmup_bars(
            chunks,
            args.data,
            args.min_post_warmup_bars_per_chunk,
        )
        log.info(f"[PARALLEL] After coalescing: {len(chunks)} chunks")
    
    # PREFLIGHT: RunIdentity Provenance Gate (TRUTH/SMOKE mode only)
    # Must run AFTER RUN_IDENTITY.json is created but BEFORE workers start
    if hard_fail_identity:
        log.info("[PREFLIGHT] Running RunIdentity Provenance Gate check...")
        try:
            from gx1.scripts.preflight_run_identity_provenance_check import (
                load_run_identity_json,
                validate_run_identity_provenance,
                write_fail_capsule,
            )
            
            # Load and validate RUN_IDENTITY.json
            identity_path = args.output_dir / "RUN_IDENTITY.json"
            if not identity_path.exists():
                raise RuntimeError(f"RUN_IDENTITY.json not found: {identity_path}")
            
            identity_data = load_run_identity_json(identity_path)
            is_valid, error_message, fail_capsule = validate_run_identity_provenance(identity_data, args.output_dir)
            
            if not is_valid:
                # Write fail capsule
                capsule_path = write_fail_capsule(args.output_dir, fail_capsule, error_message)
                
                # Write MASTER_FATAL capsule
                fatal_capsule = {
                    "timestamp": dt_now_iso(),
                    "run_id": run_id,
                    "fatal_reason": "PREFLIGHT_RUN_IDENTITY_PROVENANCE_FAIL",
                    "error_message": error_message,
                    "output_dir": str(args.output_dir),
                    "preflight_fail_capsule": str(capsule_path),
                }
                fatal_path = args.output_dir / "MASTER_FATAL.json"
                write_json_atomic(fatal_path, fatal_capsule, output_dir=args.output_dir)
                
                log.error(f"[MASTER_FATAL] Preflight RunIdentity Provenance Gate failed: {error_message}")
                log.error(f"[MASTER_FATAL] Fail capsule: {capsule_path}")
                log.error(f"[MASTER_FATAL] Fatal capsule: {fatal_path}")
                
                # Append to run index ledger
                try:
                    from gx1.utils.run_index import build_run_index_entry, append_run_index
                    from gx1.utils.output_dir import resolve_gx1_data_root
                    entry = build_run_index_entry(args.output_dir)
                    gx1_data_root = resolve_gx1_data_root()
                    append_run_index(gx1_data_root, entry)
                    log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir} (MASTER_FATAL)")
                except Exception as index_error:
                    log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
                
                raise RuntimeError(
                    f"[MASTER_FATAL] PREFLIGHT_RUN_IDENTITY_PROVENANCE_FAIL: "
                    f"RunIdentity provenance validation failed: {error_message}. "
                    f"See {capsule_path} for details."
                )
            
            # Success
            log.info("[PREFLIGHT] ✅ RunIdentity Provenance Gate PASSED")
        except RuntimeError:
            # Re-raise RuntimeError (already logged)
            raise
        except Exception as e:
            # Unexpected error in preflight check itself
            log.error(f"[PREFLIGHT] Unexpected error running provenance check: {e}")
            import traceback
            traceback.print_exc()
            
            # Write MASTER_FATAL capsule
            fatal_capsule = {
                "timestamp": dt_now_iso(),
                "run_id": run_id,
                "fatal_reason": "PREFLIGHT_RUN_IDENTITY_PROVENANCE_ERROR",
                "error_message": str(e),
                "exception_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "output_dir": str(args.output_dir),
            }
            fatal_path = args.output_dir / "MASTER_FATAL.json"
            write_json_atomic(fatal_path, fatal_capsule, output_dir=args.output_dir)
            
            # Append to run index ledger
            try:
                from gx1.utils.run_index import build_run_index_entry, append_run_index_dedup
                from gx1.utils.output_dir import resolve_gx1_data_root
                entry = build_run_index_entry(args.output_dir)
                gx1_data_root = resolve_gx1_data_root()
                append_run_index_dedup(gx1_data_root, entry)
                log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir} (MASTER_FATAL)")
            except Exception as index_error:
                log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
            
            raise RuntimeError(
                f"[MASTER_FATAL] PREFLIGHT_RUN_IDENTITY_PROVENANCE_ERROR: "
                f"Failed to run preflight check: {e}. See {fatal_path} for details."
            ) from e
    
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
        # No fallback / no auto-discovery (TRUTH invariant).
        prebuilt_enabled = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
        if prebuilt_enabled:
            # Provide ambiguity context if candidates exist on disk.
            candidates = []
            try:
                gx1_data_root_env = os.getenv("GX1_DATA_DIR") or os.getenv("GX1_DATA_ROOT") or os.getenv("GX1_DATA")
                if gx1_data_root_env:
                    gx1_data_root = Path(gx1_data_root_env).expanduser().resolve()
                    cand_dir = gx1_data_root / "data" / "data" / "prebuilt" / "V13_REFINED3_PRUNE14" / "2025"
                    if cand_dir.exists():
                        candidates = sorted([str(p.resolve()) for p in cand_dir.glob("*.parquet")])
            except Exception:
                candidates = []
            fatal = {
                "status": "FAIL",
                "error": "AMBIGUOUS_TRUTH",
                "message": "Prebuilt enabled but --prebuilt-parquet was not provided. TRUTH forbids fallback.",
                "candidates_found_n": len(candidates),
                "candidates_found": candidates[:50],
            }
            try:
                from gx1.utils.json_atomic import write_json_atomic
                write_json_atomic(args.output_dir / "PREBUILT_PATH_FATAL.json", fatal, output_dir=args.output_dir)
            except Exception:
                pass
            raise RuntimeError("[PREBUILT_FAIL] TRUTH_NO_FALLBACK: missing --prebuilt-parquet (absolute)")
    
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
    
    # DEL 4: Preflight import check before launching workers
    # Verify that critical imports work with current sys.executable
    log.info("[MASTER] [PREFLIGHT] Verifying imports before launching workers...")
    try:
        import subprocess
        preflight_cmd = [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); import gx1; from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle; from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer; print('SUCCESS')"
        ]
        result = subprocess.run(
            preflight_cmd,
            cwd=str(Path(__file__).parent.parent.parent),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            log.error(f"[MASTER] [PREFLIGHT] Import check failed:")
            log.error(f"  stdout: {result.stdout}")
            log.error(f"  stderr: {result.stderr}")
            raise RuntimeError(f"[PREFLIGHT_FAIL] Critical imports failed: {result.stderr}")
        log.info(f"[MASTER] [PREFLIGHT] ✅ Import check passed: {result.stdout.strip()}")
        log.info(f"[MASTER] [PREFLIGHT] sys.executable: {sys.executable}")
        log.info(f"[MASTER] [PREFLIGHT] gx1.__file__: {result.stdout.strip() if 'SUCCESS' in result.stdout else 'N/A'}")
    except Exception as e:
        log.error(f"[MASTER] [PREFLIGHT] Failed to verify imports: {e}")
        raise RuntimeError(f"[PREFLIGHT_FAIL] Cannot proceed without import verification: {e}")
    
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
            chunk_local_padding_days,  # Pass chunk-local padding days to worker
        )
        for chunk_start, chunk_end, chunk_idx in chunks
    ]
    
    # DEL 2: Pre-create chunk dirs in MASTER før submit
    # B0.3: Pre-create chunk dirs and write CHUNK_PLAN.json
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
            
            # B0.3: Write CHUNK_PLAN.json with padding info
            chunk_plan = {
                "chunk_id": chunk_idx,
                "chunk_start": str(chunk_start),
                "chunk_end": str(chunk_end),
                "chunk_local_padding_days": chunk_local_padding_days if 'chunk_local_padding_days' in locals() else 0,
                "effective_chunk_start": str(chunk_start - pd.Timedelta(days=chunk_local_padding_days)) if 'chunk_local_padding_days' in locals() and chunk_local_padding_days > 0 else str(chunk_start),
                "data_path": str(args.data) if args.data else None,
                "policy_path": str(args.policy) if args.policy else None,
                "timestamp": _dt_now_iso(),
            }
            chunk_plan_path = chunk_dir / "CHUNK_PLAN.json"
            write_json_atomic(chunk_plan_path, chunk_plan, output_dir=chunk_dir)
            
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
        
        # If workers=1, run directly without multiprocessing pool to avoid extra process.
        # IMPORTANT: Still honor args.chunks > 1 by running all chunk_tasks sequentially.
        if args.workers == 1:
            log.info("[MASTER] Workers=1, running directly without multiprocessing pool")
            actual_workers_started = 1
            # Initialize deadline for workers=1 case (needed for later check)
            deadline = time.time() + master_timeout_sec
            completed = set()
            # Run directly in current process (sequential per chunk).
            perf_profile_enabled = os.getenv("GX1_PERF_PROFILE", "0") in ("1", "true", "TRUE", "yes", "YES")
            if perf_profile_enabled:
                import cProfile
                import pstats
                from io import StringIO

                prof = cProfile.Profile()
                prof.enable()
                try:
                    for i, task in enumerate(chunk_tasks):
                        chunk_results.append(process_chunk(*task))
                        completed.add(task[0])
                        if int(getattr(args, "abort_after_first_chunk", 0) or 0) == 1:
                            break
                finally:
                    prof.disable()
                    try:
                        pstats_path = args.output_dir / "profile_master.pstats"
                        prof.dump_stats(str(pstats_path))
                        s = StringIO()
                        ps = pstats.Stats(prof, stream=s).sort_stats("cumulative")
                        ps.print_stats(40)
                        top_txt = args.output_dir / "profile_master_top40_cumulative.txt"
                        top_txt.write_text(s.getvalue(), encoding="utf-8")
                        log.info(f"[PERF_PROFILE] Wrote {pstats_path} and {top_txt}")
                    except Exception as e:
                        log.warning(f"[PERF_PROFILE] Failed to write profile artifacts: {e}")
            else:
                for i, task in enumerate(chunk_tasks):
                    chunk_results.append(process_chunk(*task))
                    completed.add(task[0])
                    if int(getattr(args, "abort_after_first_chunk", 0) or 0) == 1:
                        break
            pool = None
        else:
            # SUBPROCESS-PER-CHUNK ARCHITECTURE (eliminates multiprocessing/IPC problems)
            # CRITICAL: Use subprocess.Popen instead of multiprocessing pool
            # Each chunk runs in completely isolated subprocess
            import subprocess
            
            CHUNK_TIMEOUT = 600  # 10 minutes per chunk (hard limit)
            MAX_CONCURRENT = args.max_procs if args.max_procs is not None else args.workers  # Max concurrent subprocesses
            MIN_FREE_MEM_GB = args.min_free_mem_gb
            MEM_PER_PROC_GB = args.mem_per_proc_gb
            MEM_CHECK_INTERVAL = args.mem_check_interval_sec
            
            log.info(f"[MASTER] Using subprocess-per-chunk architecture with RAM-aware scheduler")
            log.info(f"[MASTER]   max_procs={MAX_CONCURRENT}, min_free_mem_gb={MIN_FREE_MEM_GB}, mem_per_proc_gb={MEM_PER_PROC_GB}, chunk_timeout={CHUNK_TIMEOUT}s")
            
            # RAM-aware scheduler helper function
            def get_mem_available_gb():
                """Read MemAvailable from /proc/meminfo and return in GB."""
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                mem_avail_kb = int(line.split()[1])
                                return mem_avail_kb / (1024.0 * 1024.0)  # Convert to GB
                except Exception as e:
                    log.warning(f"[SCHED] Failed to read /proc/meminfo: {e}, assuming 8GB available")
                    return 8.0  # Conservative fallback
                return 8.0  # Fallback if MemAvailable not found
            
            def compute_allowed_procs():
                """Compute allowed concurrent processes based on available memory."""
                mem_avail_gb = get_mem_available_gb()
                allowed_by_mem = int((mem_avail_gb - MIN_FREE_MEM_GB) / MEM_PER_PROC_GB)
                allowed = max(1, min(allowed_by_mem, MAX_CONCURRENT))  # Clamp between 1 and MAX_CONCURRENT
                return allowed, mem_avail_gb
            
            actual_workers_started = 0  # Will track actual subprocesses started
            
            # Build worker command template
            worker_script = Path(__file__).parent / "replay_worker.py"
            if not worker_script.exists():
                raise RuntimeError(f"Worker script not found: {worker_script}")
            
            python_exe = REQUIRED_VENV
            base_cmd = [
                python_exe,
                str(worker_script),
                "--data-path", str(args.data),
                "--policy-yaml", str(args.policy),
                "--bundle-dir", str(bundle_dir_override) if bundle_dir_override else str(Path(args.bundle_dir).resolve()) if args.bundle_dir else "",
                "--prebuilt-parquet", prebuilt_parquet_path,
                "--output-dir", str(args.output_dir),
                "--run-id", run_id,
                "--bundle-sha256", bundle_sha256,
            ]
            
            # Track running subprocesses: {chunk_idx: (proc, start_time, chunk_start, chunk_end, mem_avail_at_start)}
            running_procs = {}
            # Update container for watchdog access
            if 'running_procs_container' in locals():
                running_procs_container["procs"] = running_procs
            completed = set()
            chunk_results = []
            chunk_exit_codes = {}  # {chunk_idx: exit_code}
            chunk_exit_statuses = []  # List of {chunk_id, pid, rc, signal, start_ts, end_ts, duration_s}
            chunk_retry_count = {}  # {chunk_idx: retry_count} for SIGKILL retries
            last_mem_check = 0.0
            
            # CHUNK_EXIT_STATUS.jsonl file for observability
            exit_status_log_path = args.output_dir / "CHUNK_EXIT_STATUS.jsonl"
            exit_status_log_path.touch()  # Create file
            
            # Process chunks with max concurrent limit
            next_chunk_idx = 0  # Index into chunk_tasks list
            total_chunks = len(chunk_tasks)
            
            log.info(f"[MASTER] Processing {total_chunks} chunks with RAM-aware scheduler (max_procs={MAX_CONCURRENT})")
            
            poll_interval = 5.0
            last_progress_log = time.time()
            last_heartbeat = time.time()
            heartbeat_interval = 10.0  # Write heartbeat every 10 seconds
            
            while len(completed) < total_chunks:
                if MASTER_STOP_REQUESTED:
                    log.warning("[MASTER] Stop requested (SIGTERM), terminating all subprocesses...")
                    # Kill all running subprocesses
                    for cid, (proc, _, _, _, _, stdout_file, stderr_file) in running_procs.items():
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        except Exception as e:
                            log.warning(f"[MASTER] Error terminating chunk {cid}: {e}")
                        # Close files
                        try:
                            stdout_file.close()
                        except Exception:
                            pass
                        try:
                            stderr_file.close()
                        except Exception:
                            pass
                    break
                
                # RAM-aware scheduler: Check memory and compute allowed processes
                current_time = time.time()
                if current_time - last_mem_check >= MEM_CHECK_INTERVAL:
                    allowed_procs, mem_avail_gb = compute_allowed_procs()
                    last_mem_check = current_time
                else:
                    # Use cached value (recompute if needed)
                    allowed_procs, mem_avail_gb = compute_allowed_procs()
                
                # Start new subprocesses if we have slots available (both by count and memory)
                while len(running_procs) < allowed_procs and next_chunk_idx < total_chunks:
                    task = chunk_tasks[next_chunk_idx]
                    # chunk_tasks is: (chunk_idx, chunk_start, chunk_end, data, policy, run_id, output_dir, bundle_sha256, prebuilt_parquet_path, bundle_dir_override)
                    # We only need: chunk_idx, chunk_start, chunk_end
                    chunk_idx = task[0]
                    chunk_start = task[1]
                    chunk_end = task[2]
                    
                    # Build command for this chunk
                    cmd = base_cmd + [
                        "--chunk-id", str(chunk_idx),
                        "--chunk-start", str(chunk_start),
                        "--chunk-end", str(chunk_end),
                        "--chunk-local-padding-days", str(chunk_local_padding_days),
                    ]
                    
                    # Check if this chunk should be retried (SIGKILL retry policy)
                    if chunk_idx in chunk_retry_count and chunk_retry_count[chunk_idx] > 0:
                        # Reduce allowed_procs temporarily for retry
                        allowed_procs_retry = max(1, allowed_procs - 2)
                        if len(running_procs) >= allowed_procs_retry:
                            log.info(f"[RETRY] Chunk {chunk_idx} retry delayed (reduced concurrency: {allowed_procs_retry})")
                            break  # Wait for slot
                    
                    mem_avail_before = get_mem_available_gb()
                    log.info(f"[SCHED] mem_avail_gb={mem_avail_before:.2f}, allowed={allowed_procs}, running={len(running_procs)}, starting_chunk={chunk_idx}")
                    
                    # Start subprocess
                    try:
                        # CRITICAL: Redirect stdout/stderr to files to avoid pipe blocking
                        chunk_log_dir = args.output_dir / f"chunk_{chunk_idx}" / "logs"
                        chunk_log_dir.mkdir(parents=True, exist_ok=True)
                        stdout_file = open(chunk_log_dir / "worker_stdout.log", "w")
                        stderr_file = open(chunk_log_dir / "worker_stderr.log", "w")
                        
                        proc = subprocess.Popen(
                            cmd,
                            stdout=stdout_file,
                            stderr=stderr_file,
                            cwd=str(Path(__file__).parent.parent.parent),
                            env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent.parent.parent)),
                        )
                        start_time = time.time()
                        running_procs[chunk_idx] = (proc, start_time, chunk_start, chunk_end, mem_avail_before, stdout_file, stderr_file)
                        actual_workers_started = max(actual_workers_started, len(running_procs))
                        log.info(f"[MASTER] Started subprocess for chunk {chunk_idx} (PID={proc.pid}, mem_avail={mem_avail_before:.2f}GB)")
                    except Exception as e:
                        log.error(f"[MASTER] Failed to start subprocess for chunk {chunk_idx}: {e}")
                        completed.add(chunk_idx)
                        chunk_exit_codes[chunk_idx] = -1
                        # Log exit status
                        exit_status = {
                            "chunk_id": chunk_idx,
                            "pid": None,
                            "rc": -1,
                            "signal": None,
                            "start_ts": _dt_now_iso(),
                            "end_ts": _dt_now_iso(),
                            "duration_s": 0.0,
                            "error": str(e),
                        }
                        chunk_exit_statuses.append(exit_status)
                        with open(exit_status_log_path, "a") as f:
                            f.write(json.dumps(exit_status) + "\n")
                        continue
                    
                    next_chunk_idx += 1
                
                # TRUTH-only master heartbeat with worker metrics
                current_time = time.time()
                is_truth_or_smoke_master = os.getenv("GX1_RUN_MODE", "").upper() in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"
                if is_truth_or_smoke_master and (current_time - last_heartbeat) >= heartbeat_interval:
                    n_running_workers = len(running_procs)
                    n_submitted = next_chunk_idx
                    n_completed = len(completed)
                    n_failed = sum(1 for cid in completed if chunk_exit_codes.get(cid, 0) != 0)
                    n_pending = total_chunks - n_submitted
                    
                    heartbeat = {
                        "timestamp": _dt_now_iso(),
                        "n_running_workers": n_running_workers,
                        "n_submitted": n_submitted,
                        "n_completed": n_completed,
                        "n_failed": n_failed,
                        "n_pending": n_pending,
                        "total_chunks": total_chunks,
                        "mem_avail_gb": get_mem_available_gb(),
                        "allowed_procs": allowed_procs,
                    }
                    heartbeat_path = args.output_dir / "MASTER_HEARTBEAT.json"
                    write_json_atomic(heartbeat_path, heartbeat, output_dir=args.output_dir)
                    last_heartbeat = current_time
                    log.info(f"[MASTER_HEARTBEAT] running={n_running_workers}, submitted={n_submitted}, completed={n_completed}, failed={n_failed}, pending={n_pending}")
                
                # Check for completed subprocesses
                for chunk_idx in list(running_procs.keys()):
                    proc, start_time, chunk_start, chunk_end, mem_avail_at_start, stdout_file, stderr_file = running_procs[chunk_idx]
                    
                    # Check if process has finished
                    exit_code = proc.poll()
                    if exit_code is not None:
                        # Process finished
                        elapsed = current_time - start_time
                        
                        # Decode exit code: if < 0, it's a signal
                        signal_num = None
                        exit_code_actual = None
                        if exit_code < 0:
                            signal_num = -exit_code
                        else:
                            exit_code_actual = exit_code
                        
                        # Log exit status
                        exit_status = {
                            "chunk_id": chunk_idx,
                            "pid": proc.pid,
                            "rc": exit_code_actual if exit_code_actual is not None else exit_code,
                            "signal": signal_num,
                            "signal_name": f"SIG{signal_num}" if signal_num else None,
                            "start_ts": _dt_now_iso() if start_time else None,
                            "end_ts": _dt_now_iso(),
                            "duration_s": elapsed,
                            "mem_avail_at_start_gb": mem_avail_at_start,
                        }
                        chunk_exit_statuses.append(exit_status)
                        with open(exit_status_log_path, "a") as f:
                            f.write(json.dumps(exit_status) + "\n")
                        
                        chunk_exit_codes[chunk_idx] = exit_code
                        
                        # SIGKILL retry policy (C)
                        if signal_num == 9:  # SIGKILL
                            retry_count = chunk_retry_count.get(chunk_idx, 0)
                            if retry_count == 0:
                                log.warning(f"[MASTER] ❌ Chunk {chunk_idx} killed with SIGKILL (OOM?), will retry once (mem_avail_at_start={mem_avail_at_start:.2f}GB)")
                                chunk_retry_count[chunk_idx] = 1
                                # Don't mark as completed, will retry
                                del running_procs[chunk_idx]
                                # Reset next_chunk_idx to retry this chunk (find its position in chunk_tasks)
                                for i, task in enumerate(chunk_tasks):
                                    if task[0] == chunk_idx:
                                        next_chunk_idx = min(next_chunk_idx, i)
                                        break
                                continue
                            else:
                                log.error(f"[MASTER] ❌ Chunk {chunk_idx} killed with SIGKILL again (retry failed), marking as failed")
                        
                        if exit_code == 0:
                            log.info(f"[MASTER] ✅ Chunk {chunk_idx} completed successfully (exit_code=0, elapsed={elapsed:.1f}s)")
                            # Read result from chunk_footer.json (subprocess doesn't return result object)
                            chunk_results.append({"chunk_idx": chunk_idx, "status": "ok"})
                        else:
                            signal_str = f" (signal={signal_num})" if signal_num else ""
                            log.warning(f"[MASTER] ❌ Chunk {chunk_idx} failed (exit_code={exit_code}{signal_str}, elapsed={elapsed:.1f}s)")
                            # Read stderr from file (already redirected)
                            try:
                                stderr_file.flush()
                                stderr_file.seek(0)
                                stderr_output = stderr_file.read()
                                if stderr_output:
                                    error_log_path = args.output_dir / f"chunk_{chunk_idx}" / "WORKER_STDERR.txt"
                                    error_log_path.parent.mkdir(parents=True, exist_ok=True)
                                    with open(error_log_path, "w") as f:
                                        f.write(stderr_output)
                                    log.warning(f"[MASTER] Chunk {chunk_idx} stderr written to: {error_log_path}")
                            except Exception as e:
                                # Try reading from file directly if file handle is not readable
                                try:
                                    stderr_log_path = args.output_dir / f"chunk_{chunk_idx}" / "logs" / "worker_stderr.log"
                                    if stderr_log_path.exists():
                                        with open(stderr_log_path, "r") as f:
                                            stderr_output = f.read()
                                        if stderr_output:
                                            error_log_path = args.output_dir / f"chunk_{chunk_idx}" / "WORKER_STDERR.txt"
                                            with open(error_log_path, "w") as f:
                                                f.write(stderr_output)
                                            log.warning(f"[MASTER] Chunk {chunk_idx} stderr written to: {error_log_path}")
                                except Exception as e2:
                                    log.warning(f"[MASTER] Failed to read stderr for chunk {chunk_idx}: {e}, {e2}")
                        
                        completed.add(chunk_idx)
                        # Close stdout/stderr files
                        try:
                            stdout_file.close()
                        except Exception:
                            pass
                        try:
                            stderr_file.close()
                        except Exception:
                            pass
                        del running_procs[chunk_idx]
                        log.info(f"[MASTER] Progress: done={len(completed)}/{total_chunks} pending={total_chunks - len(completed)} running={len(running_procs)}")
                    
                    # Check for timeout
                    elif current_time - start_time > CHUNK_TIMEOUT:
                        log.error(f"[MASTER] ❌ Chunk {chunk_idx} timeout ({CHUNK_TIMEOUT}s), terminating...")
                        
                        # Send SIGUSR2 for dump before killing
                        try:
                            proc.send_signal(signal.SIGUSR2)
                            time.sleep(2)  # Give time for dump
                        except Exception as e:
                            log.warning(f"[MASTER] Failed to send SIGUSR2 to chunk {chunk_idx}: {e}")
                        
                        # Terminate process
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        except Exception as e:
                            log.warning(f"[MASTER] Error terminating chunk {chunk_idx}: {e}")
                        
                        # Close stdout/stderr files
                        try:
                            stdout_file.close()
                        except Exception:
                            pass
                        try:
                            stderr_file.close()
                        except Exception:
                            pass
                        
                        # Log exit status for timeout
                        exit_status = {
                            "chunk_id": chunk_idx,
                            "pid": proc.pid,
                            "rc": -2,
                            "signal": None,
                            "signal_name": "TIMEOUT",
                            "start_ts": _dt_now_iso() if start_time else None,
                            "end_ts": _dt_now_iso(),
                            "duration_s": current_time - start_time,
                            "mem_avail_at_start_gb": mem_avail_at_start,
                        }
                        chunk_exit_statuses.append(exit_status)
                        with open(exit_status_log_path, "a") as f:
                            f.write(json.dumps(exit_status) + "\n")
                        
                        chunk_exit_codes[chunk_idx] = -2  # Timeout
                        completed.add(chunk_idx)
                        del running_procs[chunk_idx]
                        log.warning(f"[MASTER] Chunk {chunk_idx} terminated due to timeout")
                
                # Log progress every 30 seconds
                if time.time() - last_progress_log > 30:
                    log.info(f"[MASTER] Progress: done={len(completed)}/{total_chunks} pending={total_chunks - len(completed)} running={len(running_procs)}")
                    last_progress_log = time.time()
                
                # Sleep if we have running processes
                if len(running_procs) > 0:
                    time.sleep(poll_interval)
                elif next_chunk_idx >= total_chunks:
                    # All chunks started, wait for remaining to finish
                    if len(completed) < total_chunks:
                        time.sleep(poll_interval)
                    else:
                        break
                
                # Log progress every 30 seconds
                if time.time() - last_progress_log > 30:
                    log.info(f"[MASTER] Progress: done={len(completed)}/{total_chunks} pending={total_chunks - len(completed)} running={len(running_procs)}")
                    last_progress_log = time.time()
                
                # Sleep if we have running processes
                if len(running_procs) > 0:
                    time.sleep(poll_interval)
                elif chunk_idx >= total_chunks:
                    # All chunks started, wait for remaining to finish
                    if len(completed) < total_chunks:
                        time.sleep(poll_interval)
                    else:
                        break
            
            # All chunks processed
            log.info(f"[MASTER] All chunks processed: {len(completed)}/{total_chunks}")
            
            # Set deadline for compatibility
            deadline = time.time() + master_timeout_sec
            main._deadline = deadline  # Store for finally block
            
            # Store exit codes for completion contract
            pool = None  # No pool in subprocess mode
            
            # Completion contract based on exit codes
            all_exit_zero = all(ec == 0 for ec in chunk_exit_codes.values())
            if all_exit_zero and len(chunk_exit_codes) == total_chunks:
                log.info(f"[MASTER] ✅ All {total_chunks} chunks completed successfully (all exit_code=0)")
            else:
                failed_chunk_ids = [cid for cid, ec in chunk_exit_codes.items() if ec != 0]
                log.warning(f"[MASTER] ❌ {len(failed_chunk_ids)} chunks failed: {failed_chunk_ids}")
                for cid in failed_chunk_ids:
                    log.warning(f"[MASTER]   Chunk {cid}: exit_code={chunk_exit_codes[cid]}")
        
        total_time = time.time() - start_time
        
        # Pool cleanup already done per batch, but check if any pool remains
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
        # Cleanup: Kill all running subprocesses if any
        if 'running_procs' in locals():
            log.warning("[MASTER] Killing all running subprocesses due to interrupt...")
            for chunk_idx, (proc, _, _, _) in running_procs.items():
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except Exception as e:
                    log.warning(f"[MASTER] Error killing chunk {chunk_idx}: {e}")
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
        
        # STEG 2: Write MASTER_FATAL.json capsule (atomic)
        try:
            master_fatal_path = args.output_dir / "MASTER_FATAL.json"
            master_fatal_data = {
                "error": "MASTER_FATAL",
                "exception_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunks_submitted": len(chunk_tasks),
                "chunks_completed": len(completed) if 'completed' in locals() else 0,
                "futures_status": {
                    "total": len(async_results) if 'async_results' in locals() else 0,
                    "completed": len(completed) if 'completed' in locals() else 0,
                    "pending": len(async_results) - len(completed) if 'async_results' in locals() and 'completed' in locals() else 0,
                },
                "pool_status": "terminated" if pool else "none",
            }
            write_json_atomic(master_fatal_path, master_fatal_data, output_dir=args.output_dir)
            log.error(f"[MASTER_FATAL] Wrote MASTER_FATAL.json to {master_fatal_path}")
            
            # Append to run index ledger
            try:
                from gx1.utils.run_index import build_run_index_entry, append_run_index_dedup
                from gx1.utils.output_dir import resolve_gx1_data_root
                entry = build_run_index_entry(args.output_dir)
                gx1_data_root = resolve_gx1_data_root()
                append_run_index_dedup(gx1_data_root, entry)
                log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir} (MASTER_FATAL)")
            except Exception as index_error:
                log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
        except Exception as fatal_capsule_error:
            log.error(f"[MASTER_FATAL] Failed to write MASTER_FATAL.json: {fatal_capsule_error}")
        
        # STEG 4: Write RUN_FAILED.json (completion contract)
        try:
            run_failed_path = args.output_dir / "RUN_FAILED.json"
            run_failed_data = {
                "status": "FAILED",
                "reason": "MASTER_EXCEPTION",
                "exception_type": type(e).__name__,
                "message": str(e),
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chunks_submitted": len(chunk_tasks),
                "chunks_completed": len(completed) if 'completed' in locals() else 0,
            }
            write_json_atomic(run_failed_path, run_failed_data, output_dir=args.output_dir)
            log.error(f"[COMPLETION_CONTRACT] Wrote RUN_FAILED.json to {run_failed_path}")
            
            # Append to run index ledger
            try:
                from gx1.utils.run_index import build_run_index_entry, append_run_index_dedup
                from gx1.utils.output_dir import resolve_gx1_data_root
                entry = build_run_index_entry(args.output_dir)
                gx1_data_root = resolve_gx1_data_root()
                append_run_index_dedup(gx1_data_root, entry)
                log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir}")
            except Exception as index_error:
                log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
        except Exception as run_failed_error:
            log.error(f"[COMPLETION_CONTRACT] Failed to write RUN_FAILED.json: {run_failed_error}")
        
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
        # CRITICAL: Write completion contract FIRST in finally block (before anything else)
        # This ensures we always know if run completed or failed, even if master dies
        try:
            # Count completed chunks from footers (more reliable than in-memory state)
            chunks_completed_count = 0
            try:
                for chunk_idx, (chunk_start, chunk_end, _) in enumerate(chunks):
                    chunk_dir = args.output_dir / f"chunk_{chunk_idx}"
                    chunk_footer_path = chunk_dir / "chunk_footer.json"
                    if chunk_footer_path.exists():
                        try:
                            with open(chunk_footer_path, "r") as f:
                                footer = json.load(f)
                            if footer.get("status") == "ok":
                                chunks_completed_count += 1
                        except Exception:
                            pass
            except Exception:
                pass  # Non-fatal, use 0 as fallback
            
            # Write completion contract based on actual chunk completion
            all_chunks_completed = chunks_completed_count == len(chunks)
            completion_contract_path = args.output_dir / "RUN_COMPLETED.json" if all_chunks_completed else args.output_dir / "RUN_FAILED.json"
            
            if all_chunks_completed:
                # ------------------------------------------------------------------
                # TRUTH/SMOKE observability: aggregate stage timings + perf counters
                # from chunk footers. Read-only; does not affect semantics.
                # ------------------------------------------------------------------
                stage_timings_s = {}
                perf_counters = {}
                try:
                    from collections import defaultdict

                    stage_acc = defaultdict(float)
                    bars_processed_total = 0
                    model_calls_total = 0
                    chunk_wall_clock_total = 0.0

                    for chunk_dir in sorted(args.output_dir.glob("chunk_*")):
                        footer_path = chunk_dir / "chunk_footer.json"
                        if not footer_path.exists():
                            continue
                        try:
                            footer = json.loads(footer_path.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        if (footer.get("status") or "").lower() != "ok":
                            continue

                        bars_processed_total += int(footer.get("bars_processed", 0) or 0)
                        model_calls_total += int(footer.get("eval_calls_total", 0) or 0)
                        chunk_wall_clock_total += float(footer.get("wall_clock_sec", 0.0) or 0.0)

                        for k, v in (footer or {}).items():
                            if not isinstance(k, str):
                                continue
                            if not k.startswith("t_") or not k.endswith("_sec"):
                                continue
                            try:
                                stage_acc[k] += float(v or 0.0)
                            except Exception:
                                pass
                        for k in ("htf_align_time_total_sec", "htf_align_warning_time_sec"):
                            if k in footer:
                                try:
                                    stage_acc[k] += float(footer.get(k) or 0.0)
                                except Exception:
                                    pass

                    stage_timings_s = dict(stage_acc)
                    perf_counters = {
                        "bars_processed_total": int(bars_processed_total),
                        "model_calls_total": int(model_calls_total),
                        "chunk_wall_clock_sec_total": float(chunk_wall_clock_total),
                        "wall_clock_sec_total": float(total_time if 'total_time' in locals() else 0.0),
                    }
                    wt = float(total_time) if 'total_time' in locals() and float(total_time) > 0.0 else 0.0
                    if wt > 0.0:
                        perf_counters["bars_per_sec"] = float(bars_processed_total) / wt
                except Exception as e:
                    stage_timings_s = {"error": str(e)}
                    perf_counters = {"error": str(e)}

                completion_data = {
                    "status": "COMPLETED",
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "chunks_submitted": len(chunks),
                    "chunks_completed": chunks_completed_count,
                    "total_time_sec": total_time if 'total_time' in locals() else 0.0,
                    "stage_timings_s": stage_timings_s,
                    "perf_counters": perf_counters,
                }
                log.info(f"[COMPLETION_CONTRACT] Writing RUN_COMPLETED.json (chunks_completed={chunks_completed_count}/{len(chunks)})")
            else:
                completion_data = {
                    "status": "FAILED",
                    "reason": "INCOMPLETE_CHUNKS",
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "chunks_submitted": len(chunks),
                    "chunks_completed": chunks_completed_count,
                    "chunks_incomplete": len(chunks) - chunks_completed_count,
                }
                log.warning(f"[COMPLETION_CONTRACT] Writing RUN_FAILED.json (chunks_completed={chunks_completed_count}/{len(chunks)})")
            
            write_json_atomic(completion_contract_path, completion_data, output_dir=args.output_dir)
            log.info(f"[COMPLETION_CONTRACT] ✅ Wrote {completion_contract_path.name} to {completion_contract_path}")

            # TRUTH-only: write master exit coverage summary (works for workers=1 direct-run path too)
            try:
                is_truth_or_smoke = os.getenv("GX1_RUN_MODE", "").upper() in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"
                if is_truth_or_smoke and all_chunks_completed:
                    write_exit_coverage_summary(args.output_dir)
                    log.info("[EXIT_COVERAGE] Wrote EXIT_COVERAGE_SUMMARY.{json,md}")
            except Exception as e:
                log.warning(f"[EXIT_COVERAGE] Failed to write summary (non-fatal): {e}")
            
            # Append to run index ledger
            try:
                from gx1.utils.run_index import build_run_index_entry, append_run_index_dedup
                from gx1.utils.output_dir import resolve_gx1_data_root
                entry = build_run_index_entry(args.output_dir)
                gx1_data_root = resolve_gx1_data_root()
                append_run_index_dedup(gx1_data_root, entry)
                log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir}")
            except Exception as index_error:
                log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
        except Exception as completion_contract_error:
            log.error(f"[COMPLETION_CONTRACT] ❌ Failed to write completion contract: {completion_contract_error}", exc_info=True)
        
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
            
            # Write XGB fingerprint summary (master-only, after all chunks complete)
            if os.getenv("GX1_XGB_INPUT_FINGERPRINT", "0") == "1":
                try:
                    from gx1.execution.oanda_demo_runner import GX1DemoRunner
                    min_logged = int(os.getenv("GX1_XGB_INPUT_FINGERPRINT_MIN_LOGGED", "50"))
                    GX1DemoRunner.write_xgb_fingerprint_summary_static(
                        output_dir=args.output_dir,
                        run_id=run_id,
                        chunk_id="master",
                        min_logged=min_logged,
                    )
                    log.info("[XGB_FINGERPRINT] ✅ Summary written by master")
                except Exception as fingerprint_error:
                    log.warning(f"[XGB_FINGERPRINT] Failed to write summary (non-fatal): {fingerprint_error}")
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
        
        # CHUNK-LOCAL PADDING: Write master WARMUP_LEDGER summary (aggregate from all chunks)
        if chunk_local_padding_days > 0:
            try:
                master_warmup_ledger = {
                    "chunk_local_padding_days": chunk_local_padding_days,
                    "total_chunks": len(chunks),
                    "chunks": [],
                }
                
                for chunk_start, chunk_end, chunk_idx in chunks:
                    chunk_dir = args.output_dir / f"chunk_{chunk_idx}"
                    chunk_footer_path = chunk_dir / "chunk_footer.json"
                    if chunk_footer_path.exists():
                        try:
                            with open(chunk_footer_path, "r") as f:
                                chunk_footer = json.load(f)
                            warmup_ledger = chunk_footer.get("warmup_ledger")
                            if warmup_ledger:
                                master_warmup_ledger["chunks"].append({
                                    "chunk_id": chunk_idx,
                                    "chunk_start": str(chunk_start),
                                    "chunk_end": str(chunk_end),
                                    **warmup_ledger,
                                })
                        except Exception as e:
                            log.warning(f"[WARMUP_LEDGER] Failed to read warmup ledger from chunk {chunk_idx}: {e}")
                
                # Write master WARMUP_LEDGER.json
                master_warmup_ledger_json_path = args.output_dir / "WARMUP_LEDGER.json"
                with open(master_warmup_ledger_json_path, "w") as f:
                    json.dump(master_warmup_ledger, f, indent=2)
                
                # Write master WARMUP_LEDGER.md
                master_warmup_ledger_md_path = args.output_dir / "WARMUP_LEDGER.md"
                md_lines = [
                    "# Master Warmup Ledger",
                    "",
                    f"**Chunk Local Padding Days:** {chunk_local_padding_days}",
                    f"**Total Chunks:** {len(chunks)}",
                    "",
                    "## Summary",
                    "",
                ]
                
                # Aggregate totals
                total_warmup_skipped = sum(c.get("warmup_skipped_total", 0) for c in master_warmup_ledger["chunks"])
                total_bars_processed = sum(c.get("bars_processed_total", 0) for c in master_warmup_ledger["chunks"])
                total_warmup_seen = sum(c.get("warmup_seen_bars", 0) for c in master_warmup_ledger["chunks"])
                
                md_lines.extend([
                    f"- **Total Warmup Skipped:** {total_warmup_skipped}",
                    f"- **Total Bars Processed:** {total_bars_processed}",
                    f"- **Total Warmup Seen:** {total_warmup_seen}",
                    "",
                    "## Per-Chunk Details",
                    "",
                ])
                
                for chunk_data in master_warmup_ledger["chunks"]:
                    md_lines.extend([
                        f"### Chunk {chunk_data['chunk_id']}",
                        "",
                        f"- **Actual Replay Start:** `{chunk_data.get('actual_replay_start_ts', 'N/A')}`",
                        f"- **Eval Start:** `{chunk_data.get('eval_start_ts', 'N/A')}`",
                        f"- **Eval End:** `{chunk_data.get('eval_end_ts', 'N/A')}`",
                        f"- **Warmup Required Bars:** {chunk_data.get('warmup_required_bars', 0)}",
                        f"- **Warmup Seen Bars:** {chunk_data.get('warmup_seen_bars', 0)}",
                        f"- **Warmup Skipped Total:** {chunk_data.get('warmup_skipped_total', 0)}",
                        f"- **Warmup Completed TS:** `{chunk_data.get('warmup_completed_ts', 'N/A')}`",
                        f"- **Bars Processed Total:** {chunk_data.get('bars_processed_total', 0)}",
                        "",
                    ])
                
                md_lines.extend([
                    "---",
                    f"*Generated: {dt_now_iso()}*",
                ])
                
                with open(master_warmup_ledger_md_path, "w") as f:
                    f.write("\n".join(md_lines))
                
                log.info(f"[WARMUP_LEDGER] Master summary written: {master_warmup_ledger_json_path}")
            except Exception as warmup_ledger_error:
                log.warning(f"[WARMUP_LEDGER] Failed to write master warmup ledger: {warmup_ledger_error}")
        
        # SSoT-AUDIT: Duplicate eval bars audit (TRUTH/SMOKE only, master-only)
        if is_truth_or_smoke:
            try:
                from collections import Counter, defaultdict
                
                # Collect all (session, ts) pairs where eval_called occurred from all chunks
                eval_bars_by_session_ts: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
                
                for chunk_start, chunk_end, chunk_idx in chunks:
                    chunk_dir = args.output_dir / f"chunk_{chunk_idx}"
                    entry_features_path = chunk_dir / "ENTRY_FEATURES_USED.json"
                    
                    if not entry_features_path.exists():
                        continue
                    
                    try:
                        with open(entry_features_path, "r") as f:
                            chunk_telemetry = json.load(f)
                        
                        # Extract entry_eval_trace_events (contains (session, ts) pairs)
                        entry_eval_trace = chunk_telemetry.get("entry_eval_trace", {})
                        # Use entry_eval_trace_summary counts_by_session to get total counts
                        # But we need actual (session, ts) pairs, so we use samples
                        # Note: samples might be capped, but for duplicate detection, we need to check
                        # if the same (session, ts) appears in multiple chunks
                        events = entry_eval_trace.get("entry_eval_trace_samples", [])
                        
                        # For duplicate detection, we only care about (session, ts) pairs that appear
                        # in multiple chunks. If a (session, ts) appears multiple times in the same chunk,
                        # that's also a duplicate, but we focus on cross-chunk duplicates first.
                        for event in events:
                            session = event.get("session")
                            ts = event.get("ts")
                            if session and ts:
                                eval_bars_by_session_ts[session][ts].append(chunk_idx)
                    except Exception as e:
                        log.warning(f"[EVAL_DEDUP_AUDIT] Failed to read telemetry from chunk {chunk_idx}: {e}")
                
                # Compute duplicates per session
                audit_results = {}
                total_duplicates = 0
                top_duplicates = []
                
                for session in sorted(eval_bars_by_session_ts.keys()):
                    session_ts_map = eval_bars_by_session_ts[session]
                    n_total_eval_calls = sum(len(chunk_ids) for chunk_ids in session_ts_map.values())
                    n_unique_eval_ts = len(session_ts_map)
                    duplicate_eval_calls = n_total_eval_calls - n_unique_eval_ts
                    
                    audit_results[session] = {
                        "n_total_eval_calls": n_total_eval_calls,
                        "n_unique_eval_ts": n_unique_eval_ts,
                        "duplicate_eval_calls": duplicate_eval_calls,
                    }
                    
                    total_duplicates += duplicate_eval_calls
                    
                    # Collect duplicates for top 20
                    for ts, chunk_ids in session_ts_map.items():
                        if len(chunk_ids) > 1:
                            top_duplicates.append({
                                "session": session,
                                "ts": ts,
                                "count": len(chunk_ids),
                                "chunk_ids": sorted(chunk_ids),
                            })
                
                # Sort by count (descending) and take top 20
                top_duplicates.sort(key=lambda x: x["count"], reverse=True)
                top_duplicates = top_duplicates[:20]
                
                # Write EVAL_DEDUP_AUDIT.md
                audit_md_path = args.output_dir / "EVAL_DEDUP_AUDIT.md"
                md_lines = [
                    "# Eval Deduplication Audit",
                    "",
                    f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
                    f"**Total Duplicates:** {total_duplicates}",
                    "",
                    "## Per-Session Summary",
                    "",
                    "| Session | Total Eval Calls | Unique Eval TS | Duplicate Calls |",
                    "|---------|------------------|----------------|-----------------|",
                ]
                
                for session in sorted(audit_results.keys()):
                    result = audit_results[session]
                    md_lines.append(
                        f"| {session} | {result['n_total_eval_calls']} | {result['n_unique_eval_ts']} | {result['duplicate_eval_calls']} |"
                    )
                
                md_lines.extend([
                    "",
                    "## Top 20 Duplicated (session, ts) Pairs",
                    "",
                ])
                
                if top_duplicates:
                    md_lines.extend([
                        "| Session | TS | Count | Chunk IDs |",
                        "|---------|----|----|-----------|",
                    ])
                    for dup in top_duplicates:
                        md_lines.append(
                            f"| {dup['session']} | `{dup['ts']}` | {dup['count']} | {', '.join(map(str, dup['chunk_ids']))} |"
                        )
                else:
                    md_lines.append("*No duplicates found.*")
                
                md_lines.extend([
                    "",
                    "## Conclusion",
                    "",
                ])
                
                status = "PASS" if total_duplicates == 0 else "FAIL"
                md_lines.append(f"**Status:** {status}")
                if total_duplicates > 0:
                    md_lines.append(f"**Reason:** Found {total_duplicates} duplicate eval calls across all sessions.")
                else:
                    md_lines.append("**Reason:** No duplicate eval calls detected.")
                
                md_lines.extend([
                    "",
                    "---",
                    f"*Generated: {datetime.now(timezone.utc).isoformat()}*",
                ])
                
                with open(audit_md_path, "w") as f:
                    f.write("\n".join(md_lines))
                
                log.info(f"[EVAL_DEDUP_AUDIT] Audit report written: {audit_md_path} (status={status})")
                
                # Hard-fail if duplicates > 0
                if total_duplicates > 0:
                    fatal_path = args.output_dir / "DUPLICATE_EVAL_BARS_FATAL.json"
                    capsule = {
                        "status": "FAIL",
                        "message": f"Found {total_duplicates} duplicate eval calls across all sessions",
                        "total_duplicates": total_duplicates,
                        "per_session": audit_results,
                        "top_20_duplicates": top_duplicates,
                        "sys.executable": sys.executable,
                        "sys.version": sys.version,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    with open(fatal_path, "w") as f:
                        json.dump(capsule, f, indent=2)
                    log.error(f"[EVAL_DEDUP_AUDIT] FATAL: Duplicate eval bars detected. Capsule: {fatal_path}")
                    raise SystemExit(2)
            except SystemExit:
                raise
            except Exception as audit_error:
                log.warning(f"[EVAL_DEDUP_AUDIT] Failed to run duplicate eval bars audit: {audit_error}", exc_info=True)
        
        # Stop watchdog thread (normal completion path)
        watchdog_done.set()
        
        # Stop run watchdog thread
        if 'run_watchdog_done' in locals():
            run_watchdog_done.set()
        
        # STEG 4: Write RUN_COMPLETED.json (completion contract) - only if all chunks completed
        try:
            all_chunks_completed = len(completed) == len(async_results) if 'completed' in locals() and 'async_results' in locals() else False
            if all_chunks_completed and not MASTER_STOP_REQUESTED:
                run_completed_path = args.output_dir / "RUN_COMPLETED.json"

                # ------------------------------------------------------------------
                # TRUTH/SMOKE observability: aggregate stage timings + perf counters
                # from chunk footers. This is read-only and does not affect semantics.
                # ------------------------------------------------------------------
                stage_timings_s = {}
                perf_counters = {}
                try:
                    from collections import defaultdict

                    stage_acc = defaultdict(float)
                    bars_processed_total = 0
                    model_calls_total = 0
                    chunk_wall_clock_total = 0.0

                    for chunk_dir in sorted(args.output_dir.glob("chunk_*")):
                        footer_path = chunk_dir / "chunk_footer.json"
                        if not footer_path.exists():
                            continue
                        try:
                            footer = json.loads(footer_path.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        if (footer.get("status") or "").lower() != "ok":
                            continue

                        bars_processed_total += int(footer.get("bars_processed", 0) or 0)
                        model_calls_total += int(footer.get("eval_calls_total", 0) or 0)
                        chunk_wall_clock_total += float(footer.get("wall_clock_sec", 0.0) or 0.0)

                        # Stage timings: sum all keys like t_*_sec (seconds)
                        for k, v in (footer or {}).items():
                            if not isinstance(k, str):
                                continue
                            if not k.startswith("t_") or not k.endswith("_sec"):
                                continue
                            try:
                                stage_acc[k] += float(v or 0.0)
                            except Exception:
                                pass
                        # Also include HTF align timings if present
                        for k in ("htf_align_time_total_sec", "htf_align_warning_time_sec"):
                            if k in footer:
                                try:
                                    stage_acc[k] += float(footer.get(k) or 0.0)
                                except Exception:
                                    pass

                    stage_timings_s = dict(stage_acc)
                    perf_counters = {
                        "bars_processed_total": int(bars_processed_total),
                        "model_calls_total": int(model_calls_total),
                        "chunk_wall_clock_sec_total": float(chunk_wall_clock_total),
                        "wall_clock_sec_total": float(total_time),
                    }
                    # Derived throughput (avoid divide-by-zero)
                    if float(total_time) > 0.0:
                        perf_counters["bars_per_sec"] = float(bars_processed_total) / float(total_time)
                except Exception as e:
                    # Best-effort only; never fail completion contract due to observability export
                    stage_timings_s = {"error": str(e)}
                    perf_counters = {"error": str(e)}

                run_completed_data = {
                    "status": "COMPLETED",
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "chunks_submitted": len(chunk_tasks),
                    "chunks_completed": len(completed) if 'completed' in locals() else 0,
                    "total_time_sec": total_time,
                    "perf_json_written": perf_json_written,
                    "stage_timings_s": stage_timings_s,
                    "perf_counters": perf_counters,
                }
                with open(run_completed_path, "w") as f:
                    json.dump(run_completed_data, f, indent=2)
                log.info(f"[COMPLETION_CONTRACT] Wrote RUN_COMPLETED.json to {run_completed_path}")

                # TRUTH-only: write master exit coverage summary
                try:
                    is_truth_or_smoke = os.getenv("GX1_RUN_MODE", "").upper() in ("TRUTH", "SMOKE") or os.getenv("GX1_SMOKE", "0") == "1"
                    if is_truth_or_smoke:
                        write_exit_coverage_summary(args.output_dir)
                        log.info("[EXIT_COVERAGE] Wrote EXIT_COVERAGE_SUMMARY.{json,md}")
                except Exception as e:
                    log.warning(f"[EXIT_COVERAGE] Failed to write summary (non-fatal): {e}")
                
                # Append to run index ledger
                try:
                    from gx1.utils.run_index import build_run_index_entry, append_run_index
                    from gx1.utils.output_dir import resolve_gx1_data_root
                    entry = build_run_index_entry(args.output_dir)
                    gx1_data_root = resolve_gx1_data_root()
                    append_run_index(gx1_data_root, entry)
                    log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir}")
                except Exception as index_error:
                    log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
                
                # HARD-FAIL TRIPWIRE: Check file count limits before cleanup
                # This prevents reports explosion (like the 1.8M files issue)
                try:
                    from gx1.scripts.reports_cleanup import check_file_count_limit, get_output_mode_from_run_identity
                    from gx1.runtime.run_identity import load_run_identity
                    
                    # Get output_mode from RUN_IDENTITY
                    identity_path = args.output_dir / "RUN_IDENTITY.json"
                    if identity_path.exists():
                        identity = load_run_identity(identity_path)
                        output_mode = identity.output_mode
                    else:
                        output_mode = get_output_mode_from_run_identity(args.output_dir)
                    
                    check_file_count_limit(args.output_dir, output_mode)
                    log.info(f"[TRIPWIRE] File count check passed for {output_mode} mode")
                except RuntimeError as tripwire_error:
                    # Hard-fail: this is a critical safety check
                    log.error(f"[TRIPWIRE] REPORTS_EXPLOSION_TRIPWIRE triggered: {tripwire_error}")
                    raise
                except Exception as tripwire_check_error:
                    # Non-fatal: log warning but continue (tripwire check failed, not the limit itself)
                    log.warning(f"[TRIPWIRE] File count check failed (non-fatal): {tripwire_check_error}")
                
                # POST-RUN CLEANUP: Apply OUTPUT_MODE policy (automatic cleanup)
                # This includes a hard-fail tripwire for file count limits
                try:
                    from gx1.scripts.reports_cleanup import cleanup_output_directory
                    cleanup_result = cleanup_output_directory(args.output_dir)
                    log.info(
                        f"[CLEANUP] Post-run cleanup completed: {cleanup_result['files_deleted']} files deleted, "
                        f"{cleanup_result['bytes_freed_mb']} MB freed (output_mode={cleanup_result['output_mode']})"
                    )
                    
                    # Update RUN_IDENTITY with cleanup stats (if it exists)
                    identity_path = args.output_dir / "RUN_IDENTITY.json"
                    if identity_path.exists():
                        try:
                            from gx1.runtime.run_identity import load_run_identity
                            identity = load_run_identity(identity_path)
                            # Note: cleanup stats could be added to RUN_IDENTITY if needed
                            # For now, we just log them
                        except Exception as e:
                            log.warning(f"[CLEANUP] Failed to update RUN_IDENTITY with cleanup stats: {e}")
                except Exception as cleanup_error:
                    # Non-fatal: log warning but don't fail the run
                    log.warning(f"[CLEANUP] Post-run cleanup failed (non-fatal): {cleanup_error}")
                    import traceback
                    log.debug(f"[CLEANUP] Cleanup traceback: {traceback.format_exc()}")
            elif MASTER_STOP_REQUESTED or (hasattr(main, '_deadline') and time.time() > main._deadline):
                # Write RUN_FAILED.json for stop/timeout cases
                run_failed_path = args.output_dir / "RUN_FAILED.json"
                reason = "SIGTERM" if MASTER_STOP_REQUESTED else "timeout"
                run_failed_data = {
                    "status": "FAILED",
                    "reason": reason,
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "chunks_submitted": len(chunk_tasks),
                    "chunks_completed": len(completed) if 'completed' in locals() else 0,
                }
                with open(run_failed_path, "w") as f:
                    json.dump(run_failed_data, f, indent=2)
                log.warning(f"[COMPLETION_CONTRACT] Wrote RUN_FAILED.json to {run_failed_path} (reason={reason})")
                
                # Append to run index ledger
                try:
                    from gx1.utils.run_index import build_run_index_entry, append_run_index
                    from gx1.utils.output_dir import resolve_gx1_data_root
                    entry = build_run_index_entry(args.output_dir)
                    gx1_data_root = resolve_gx1_data_root()
                    append_run_index(gx1_data_root, entry)
                    log.info(f"[RUN_INDEX] Appended run index entry for {args.output_dir}")
                except Exception as index_error:
                    log.warning(f"[RUN_INDEX] Failed to append run index entry (non-fatal): {index_error}")
        except Exception as completion_contract_error:
            log.error(f"[COMPLETION_CONTRACT] Failed to write completion contract: {completion_contract_error}")
        
        # Hard exit-path after perf JSON export (prevents zombie master)
        # If we were stopped or timed out, exit immediately after perf export
        # Use os._exit(0) to bypass all cleanup handlers and ensure immediate termination
        if MASTER_STOP_REQUESTED or (hasattr(main, '_deadline') and time.time() > main._deadline):
            reason = "SIGTERM" if MASTER_STOP_REQUESTED else "timeout"
            log.info(f"[MASTER] Exiting immediately after perf JSON export (reason={reason})")
            os._exit(0)  # Hard exit after perf JSON is written (prevents zombie master)


if __name__ == "__main__":
    # B0.2: Wrap main() in try/except to catch uncaught exceptions
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # Re-raise these (they're expected)
        raise
    except Exception as e:
        import traceback
        import tempfile
        
        # Try to get output_dir from args if available
        output_dir = None
        try:
            # Try to parse args to get output_dir
            parser = argparse.ArgumentParser()
            parser.add_argument("--output-dir", type=Path)
            args, _ = parser.parse_known_args()
            if args.output_dir:
                output_dir = args.output_dir
        except Exception:
            pass
        
        # Fallback to /tmp if output_dir not available
        if output_dir is None:
            output_dir = Path(tempfile.gettempdir()) / "gx1_master_uncaught"
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write MASTER_UNCAUGHT_FATAL.json
        fatal_capsule = {
            "status": "FATAL",
            "exception_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": dt_now_iso(),
            "pid": os.getpid(),
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "argv": sys.argv,
        }
        
        fatal_path = output_dir / "MASTER_UNCAUGHT_FATAL.json"
        try:
            write_json_atomic(fatal_path, fatal_capsule, output_dir=output_dir)
            log.error(f"[MASTER_UNCAUGHT_FATAL] Uncaught exception written to {fatal_path}")
        except Exception as write_error:
            # Last resort: write to /tmp directly
            tmp_path = Path(tempfile.gettempdir()) / f"gx1_master_uncaught_{os.getpid()}.json"
            try:
                with open(tmp_path, "w") as f:
                    json.dump(fatal_capsule, f, indent=2)
                log.error(f"[MASTER_UNCAUGHT_FATAL] Fallback write to {tmp_path} (original error: {write_error})")
            except Exception:
                log.error(f"[MASTER_UNCAUGHT_FATAL] Failed to write capsule even to /tmp: {write_error}")
        
        # Re-raise to ensure proper exit code
        raise