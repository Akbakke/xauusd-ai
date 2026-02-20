#!/usr/bin/env python3
"""
Run Watchdog for GX1 Replay Master.

Purpose:
- Monitor run progress via filesystem-based signals
- Write HEARTBEAT.json every 5 seconds
- Detect stalls and write RUN_STALL_FATAL.json
- Hard-fail (exit 2) if run stalls beyond timeout

Contract:
- Progress is defined as filesystem events (chunk_footer.json, WORKER_BOOT.json, etc.)
- Watchdog runs in a separate thread and monitors deterministically
- Uses atomic file writes for all capsules
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# --- Utility functions for atomic writes ---
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json_atomic(path: Path, obj: Dict[str, Any], output_dir: Path) -> None:
    """Write JSON file atomically."""
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        # Fallback to /tmp if output_dir is not writable
        fallback_dir = Path(tempfile.gettempdir()) / "gx1_watchdog_capsules"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / f"{path.name}_{int(time.time())}.json"
        try:
            with open(fallback_path, "w") as f:
                json.dump(obj, f, indent=2)
            sys.stderr.write(f"[WATCHDOG] Failed to write to {path}, wrote to fallback {fallback_path}: {e}\n")
        except Exception as fallback_e:
            sys.stderr.write(f"[WATCHDOG] FATAL: Failed to write capsule to both {path} and fallback {fallback_path}: {fallback_e}\n")

def _get_ps_snapshot() -> Dict[str, Any]:
    """Get process snapshot (master pid + child pids if available)."""
    snapshot = {
        "master_pid": os.getpid(),
        "child_pids": [],
    }
    try:
        # Try to get child pids from /proc (Linux-specific)
        proc_dir = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
        if proc_dir.exists():
            try:
                with open(proc_dir, "r") as f:
                    child_pids_str = f.read().strip()
                    if child_pids_str:
                        snapshot["child_pids"] = [int(pid) for pid in child_pids_str.split() if pid.isdigit()]
            except Exception:
                pass
    except Exception:
        pass
    return snapshot

def _detect_progress(
    output_dir: Path,
    last_known_chunk_footers: Set[int],
    last_known_worker_boots: Set[int],
    last_known_joins: Set[int],
    last_exit_status_lines: int,
) -> tuple[bool, Set[int], Set[int], Set[int], int]:
    """
    Detect progress by checking filesystem for new artifacts.
    
    Returns:
        (progress_detected, new_chunk_footers, new_worker_boots, new_joins, new_exit_status_lines)
    """
    progress_detected = False
    new_chunk_footers = set()
    new_worker_boots = set()
    new_joins = set()
    new_exit_status_lines = last_exit_status_lines
    
    # Check for chunk directories
    for chunk_dir in output_dir.glob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        
        try:
            chunk_idx = int(chunk_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        
        # 1. Check for chunk_footer.json (new or updated)
        chunk_footer_path = chunk_dir / "chunk_footer.json"
        if chunk_footer_path.exists():
            if chunk_idx not in last_known_chunk_footers:
                new_chunk_footers.add(chunk_idx)
                progress_detected = True
            else:
                # Check if file was updated (mtime changed)
                try:
                    mtime = chunk_footer_path.stat().st_mtime
                    # If mtime is very recent (within last 10 seconds), consider it progress
                    if time.time() - mtime < 10.0:
                        progress_detected = True
                except Exception:
                    pass
        
        # 2. Check for WORKER_BOOT.json (new)
        worker_boot_path = chunk_dir / "WORKER_BOOT.json"
        if worker_boot_path.exists():
            if chunk_idx not in last_known_worker_boots:
                new_worker_boots.add(chunk_idx)
                progress_detected = True
        
        # 3. Check for RAW_PREBUILT_JOIN.json (new or updated)
        join_path = chunk_dir / "RAW_PREBUILT_JOIN.json"
        if join_path.exists():
            if chunk_idx not in last_known_joins:
                new_joins.add(chunk_idx)
                progress_detected = True
            else:
                # Check if file was updated
                try:
                    mtime = join_path.stat().st_mtime
                    if time.time() - mtime < 10.0:
                        progress_detected = True
                except Exception:
                    pass

        # 3b. BAR_PROGRESS_HEARTBEAT.json (1W1C in-process: bar loop writes every 1000 bars)
        heartbeat_path = chunk_dir / "BAR_PROGRESS_HEARTBEAT.json"
        if heartbeat_path.exists():
            try:
                mtime = heartbeat_path.stat().st_mtime
                if time.time() - mtime < 30.0:
                    progress_detected = True
            except Exception:
                pass

    # 4. Check for CHUNK_EXIT_STATUS.jsonl (new lines)
    exit_status_path = output_dir / "CHUNK_EXIT_STATUS.jsonl"
    if exit_status_path.exists():
        try:
            with open(exit_status_path, "r") as f:
                lines = f.readlines()
            current_lines = len(lines)
            if current_lines > last_exit_status_lines:
                new_exit_status_lines = current_lines
                progress_detected = True
            else:
                new_exit_status_lines = last_exit_status_lines
        except Exception:
            new_exit_status_lines = last_exit_status_lines
    else:
        new_exit_status_lines = last_exit_status_lines
    
    return progress_detected, new_chunk_footers, new_worker_boots, new_joins, new_exit_status_lines

def _get_chunk_artifact_snapshot(output_dir: Path) -> Dict[str, Any]:
    """Get snapshot of chunk directories and their artifacts."""
    snapshot = {}
    for chunk_dir in sorted(output_dir.glob("chunk_*")):
        if not chunk_dir.is_dir():
            continue
        
        try:
            chunk_idx = int(chunk_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        
        artifacts = []
        if (chunk_dir / "WORKER_BOOT.json").exists():
            artifacts.append("boot")
        if (chunk_dir / "RAW_PREBUILT_JOIN.json").exists():
            artifacts.append("join")
        if (chunk_dir / "chunk_footer.json").exists():
            artifacts.append("footer")
        
        snapshot[f"chunk_{chunk_idx}"] = {
            "artifacts": artifacts,
            "path": str(chunk_dir),
        }
    
    return snapshot

def run_watchdog(
    output_dir: Path,
    run_id: str,
    watchdog_done: threading.Event,
    get_running_procs: Optional[Callable[[], List[int]]] = None,
    stall_timeout_seconds: int = 120,
    progress_window_seconds: int = 30,
    heartbeat_interval_seconds: float = 5.0,
) -> None:
    """
    Run watchdog thread to monitor replay progress.
    
    Args:
        output_dir: Output directory for the replay run
        run_id: Run ID for logging
        watchdog_done: Event to signal watchdog to stop
        get_running_procs: Optional function to get list of running process PIDs
        stall_timeout_seconds: Seconds without progress before triggering stall fatal
        progress_window_seconds: Window for detecting progress (not used directly, but logged)
        heartbeat_interval_seconds: Interval between heartbeat writes
    """
    # Write WATCHDOG_STATE.json
    watchdog_state = {
        "stall_timeout_seconds": stall_timeout_seconds,
        "progress_window_seconds": progress_window_seconds,
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
        "progress_definition": [
            "New chunk_footer.json written (status ok/failed)",
            "New line appended to CHUNK_EXIT_STATUS.jsonl",
            "New/updated RAW_PREBUILT_JOIN.json in chunk dir",
            "New WORKER_BOOT.json in chunk dir",
        ],
        "timestamp": _now_utc_iso(),
    }
    watchdog_state_path = output_dir / "WATCHDOG_STATE.json"
    _write_json_atomic(watchdog_state_path, watchdog_state, output_dir)
    
    # Track state
    last_progress_ts_utc = time.time()
    last_known_chunk_footers: Set[int] = set()
    last_known_worker_boots: Set[int] = set()
    last_known_joins: Set[int] = set()
    last_exit_status_lines = 0
    stage = "planning"  # Will be updated by heartbeat
    
    # Initial heartbeat
    heartbeat = {
        "timestamp_utc": _now_utc_iso(),
        "stage": stage,
        "chunks_planned": 0,
        "chunks_submitted": 0,
        "chunks_completed": 0,
        "chunks_failed": 0,
        "last_progress_ts_utc": _now_utc_iso(),
        "last_completed_chunk_id": None,
        "active_children_pids": [],
    }
    heartbeat_path = output_dir / "HEARTBEAT.json"
    _write_json_atomic(heartbeat_path, heartbeat, output_dir)
    
    # Main watchdog loop
    while not watchdog_done.is_set():
        try:
            # Update stage based on filesystem state
            # Check if RUN_COMPLETED.json exists -> "completed"
            if (output_dir / "RUN_COMPLETED.json").exists():
                stage = "completed"
            # Check if any chunk_footer.json exists -> "running"
            elif any((output_dir / f"chunk_{i}").exists() and (output_dir / f"chunk_{i}" / "chunk_footer.json").exists() for i in range(100)):
                stage = "running"
            # Check if any WORKER_BOOT.json exists -> "submitting"
            elif any((output_dir / f"chunk_{i}").exists() and (output_dir / f"chunk_{i}" / "WORKER_BOOT.json").exists() for i in range(100)):
                stage = "submitting"
            else:
                stage = "planning"
            
            # Count chunks
            chunks_planned = len(list(output_dir.glob("chunk_*")))
            chunks_completed = 0
            chunks_failed = 0
            last_completed_chunk_id = None
            
            for chunk_dir in output_dir.glob("chunk_*"):
                if not chunk_dir.is_dir():
                    continue
                
                chunk_footer_path = chunk_dir / "chunk_footer.json"
                if chunk_footer_path.exists():
                    try:
                        with open(chunk_footer_path, "r") as f:
                            footer = json.load(f)
                        status = footer.get("status", "unknown")
                        if status == "ok":
                            chunks_completed += 1
                            try:
                                chunk_idx = int(chunk_dir.name.split("_")[1])
                                if last_completed_chunk_id is None or chunk_idx > last_completed_chunk_id:
                                    last_completed_chunk_id = chunk_idx
                            except (ValueError, IndexError):
                                pass
                        elif status in ("failed", "error"):
                            chunks_failed += 1
                    except Exception:
                        pass
            
            chunks_submitted = chunks_completed + chunks_failed + len([d for d in output_dir.glob("chunk_*") if d.is_dir() and (d / "WORKER_BOOT.json").exists()])
            
            # Detect progress
            progress_detected, new_footers, new_boots, new_joins, new_exit_lines = _detect_progress(
                output_dir,
                last_known_chunk_footers,
                last_known_worker_boots,
                last_known_joins,
                last_exit_status_lines,
            )
            
            if progress_detected:
                last_progress_ts_utc = time.time()
                last_known_chunk_footers.update(new_footers)
                last_known_worker_boots.update(new_boots)
                last_known_joins.update(new_joins)
                last_exit_status_lines = new_exit_lines
            
            # Get active children PIDs (if function provided)
            active_children_pids = []
            if get_running_procs is not None:
                try:
                    active_children_pids = get_running_procs()
                except Exception:
                    pass
            
            # Write heartbeat
            heartbeat = {
                "timestamp_utc": _now_utc_iso(),
                "stage": stage,
                "chunks_planned": chunks_planned,
                "chunks_submitted": chunks_submitted,
                "chunks_completed": chunks_completed,
                "chunks_failed": chunks_failed,
                "last_progress_ts_utc": datetime.fromtimestamp(last_progress_ts_utc, tz=timezone.utc).isoformat(),
                "last_completed_chunk_id": last_completed_chunk_id,
                "active_children_pids": active_children_pids,
            }
            _write_json_atomic(heartbeat_path, heartbeat, output_dir)
            
            # Check for stall
            seconds_since_progress = time.time() - last_progress_ts_utc
            if seconds_since_progress > stall_timeout_seconds and stage not in ("completed", "planning"):
                # STALL DETECTED: Write RUN_STALL_FATAL.json
                chunk_artifact_snapshot = _get_chunk_artifact_snapshot(output_dir)
                ps_snapshot = _get_ps_snapshot()
                
                stall_fatal = {
                    "status": "STALL_FATAL",
                    "now_utc": _now_utc_iso(),
                    "last_progress_utc": datetime.fromtimestamp(last_progress_ts_utc, tz=timezone.utc).isoformat(),
                    "seconds_since_progress": seconds_since_progress,
                    "stall_timeout_seconds": stall_timeout_seconds,
                    "stage": stage,
                    "counts": {
                        "chunks_planned": chunks_planned,
                        "chunks_submitted": chunks_submitted,
                        "chunks_completed": chunks_completed,
                        "chunks_failed": chunks_failed,
                    },
                    "chunk_artifact_snapshot": chunk_artifact_snapshot,
                    "ps_snapshot": ps_snapshot,
                    "run_id": run_id,
                }
                
                stall_fatal_path = output_dir / "RUN_STALL_FATAL.json"
                _write_json_atomic(stall_fatal_path, stall_fatal, output_dir)
                
                # Write RUN_FAILED marker
                run_failed = {
                    "status": "FAILED",
                    "reason": "STALL_FATAL",
                    "run_id": run_id,
                    "timestamp": _now_utc_iso(),
                    "stall_fatal_path": str(stall_fatal_path),
                }
                run_failed_path = output_dir / "RUN_FAILED.json"
                _write_json_atomic(run_failed_path, run_failed, output_dir)
                
                # Log to stderr (watchdog runs in separate thread, so stdout might be buffered)
                sys.stderr.write(
                    f"[WATCHDOG] FATAL: Run stalled for {seconds_since_progress:.1f}s "
                    f"(timeout={stall_timeout_seconds}s). "
                    f"Stage={stage}, chunks_completed={chunks_completed}/{chunks_planned}. "
                    f"Writing RUN_STALL_FATAL.json and exiting.\n"
                )
                sys.stderr.flush()
                
                # Exit with code 2 (hard-fail)
                # Note: This will terminate the entire process, including the main thread
                # Use os._exit to bypass Python cleanup handlers
                os._exit(2)
            
            # Sleep until next heartbeat
            watchdog_done.wait(timeout=heartbeat_interval_seconds)
            
        except Exception as e:
            # Log error but continue (watchdog should be resilient)
            sys.stderr.write(f"[WATCHDOG] Error in watchdog loop: {e}\n")
            sys.stderr.flush()
            time.sleep(heartbeat_interval_seconds)
    
    # Final heartbeat on shutdown
    try:
        heartbeat = {
            "timestamp_utc": _now_utc_iso(),
            "stage": "shutdown",
            "chunks_planned": chunks_planned if 'chunks_planned' in locals() else 0,
            "chunks_submitted": chunks_submitted if 'chunks_submitted' in locals() else 0,
            "chunks_completed": chunks_completed if 'chunks_completed' in locals() else 0,
            "chunks_failed": chunks_failed if 'chunks_failed' in locals() else 0,
            "last_progress_ts_utc": datetime.fromtimestamp(last_progress_ts_utc, tz=timezone.utc).isoformat() if 'last_progress_ts_utc' in locals() else _now_utc_iso(),
            "last_completed_chunk_id": last_completed_chunk_id if 'last_completed_chunk_id' in locals() else None,
            "active_children_pids": active_children_pids if 'active_children_pids' in locals() else [],
        }
        _write_json_atomic(heartbeat_path, heartbeat, output_dir)
    except Exception:
        pass
