#!/usr/bin/env python3
"""
Run Watchdog for GX1 (TRUTH/SMOKE).

Purpose:
- Continuously monitor run progress via filesystem-based signals.
- Write HEARTBEAT.json every 5 seconds.
- Detect stalled runs and trigger hard-fail with RUN_STALL_FATAL.json.

Contract:
- Progress is defined by deterministic file-based signals (chunk_footer.json, WORKER_BOOT.json, etc.).
- If no progress for stall_timeout_seconds, write RUN_STALL_FATAL.json and exit code 2.
- Always writes HEARTBEAT.json and WATCHDOG_STATE.json atomically.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# --- Utility functions ---
def _dt_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON file atomically."""
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

def _get_active_child_pids() -> List[int]:
    """Get list of active child process PIDs (if available)."""
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        return [c.pid for c in children]
    except ImportError:
        # psutil not available, return empty list
        return []
    except Exception:
        # Any other error, return empty list
        return []

def _detect_progress(
    output_dir: Path,
    last_known_chunk_footers: Set[Path],
    last_known_worker_boots: Set[Path],
    last_known_join_files: Set[Path],
    last_known_exit_status_lines: int,
) -> tuple[bool, Set[Path], Set[Path], Set[Path], int]:
    """
    Detect if progress has occurred based on filesystem signals.
    
    Returns:
        (progress_detected, new_footers, new_boots, new_joins, new_exit_status_lines)
    """
    progress_detected = False
    new_footers = set()
    new_boots = set()
    new_joins = set()
    new_exit_status_lines = 0
    
    # Check for new chunk_footer.json files
    for chunk_dir in output_dir.glob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        footer_path = chunk_dir / "chunk_footer.json"
        if footer_path.exists() and footer_path not in last_known_chunk_footers:
            new_footers.add(footer_path)
            progress_detected = True
    
    # Check for new WORKER_BOOT.json files
    for chunk_dir in output_dir.glob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        boot_path = chunk_dir / "WORKER_BOOT.json"
        if boot_path.exists() and boot_path not in last_known_worker_boots:
            new_boots.add(boot_path)
            progress_detected = True
    
    # Check for new/updated RAW_PREBUILT_JOIN.json files
    for chunk_dir in output_dir.glob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        join_path = chunk_dir / "RAW_PREBUILT_JOIN.json"
        if join_path.exists() and join_path not in last_known_join_files:
            new_joins.add(join_path)
            progress_detected = True
    
    # Check for new lines in CHUNK_EXIT_STATUS.jsonl
    exit_status_path = output_dir / "CHUNK_EXIT_STATUS.jsonl"
    if exit_status_path.exists():
        try:
            with open(exit_status_path, "r") as f:
                lines = f.readlines()
            new_exit_status_lines = len(lines)
            if new_exit_status_lines > last_known_exit_status_lines:
                progress_detected = True
        except Exception:
            pass
    
    return progress_detected, new_footers, new_boots, new_joins, new_exit_status_lines

def _get_chunk_artifacts_snapshot(output_dir: Path) -> Dict[str, Dict[str, bool]]:
    """Get snapshot of chunk directories and their artifacts."""
    snapshot = {}
    for chunk_dir in sorted(output_dir.glob("chunk_*")):
        if not chunk_dir.is_dir():
            continue
        chunk_id = chunk_dir.name
        snapshot[chunk_id] = {
            "has_boot": (chunk_dir / "WORKER_BOOT.json").exists(),
            "has_join": (chunk_dir / "RAW_PREBUILT_JOIN.json").exists(),
            "has_footer": (chunk_dir / "chunk_footer.json").exists(),
        }
    return snapshot

def _get_ps_snapshot() -> Dict[str, Any]:
    """Get process snapshot (master PID + child PIDs + status)."""
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        return {
            "master_pid": os.getpid(),
            "master_status": current_process.status(),
            "child_pids": [c.pid for c in children],
            "child_statuses": {c.pid: c.status() for c in children},
        }
    except ImportError:
        return {
            "master_pid": os.getpid(),
            "master_status": "unknown",
            "child_pids": [],
            "child_statuses": {},
            "note": "psutil not available",
        }
    except Exception as e:
        return {
            "master_pid": os.getpid(),
            "master_status": "unknown",
            "child_pids": [],
            "child_statuses": {},
            "error": str(e),
        }

class RunWatchdog:
    """Watchdog thread for monitoring run progress."""
    
    def __init__(
        self,
        output_dir: Path,
        stall_timeout_seconds: int = 120,
        progress_window_seconds: int = 30,
        heartbeat_interval_seconds: float = 5.0,
    ):
        self.output_dir = output_dir
        self.stall_timeout_seconds = stall_timeout_seconds
        self.progress_window_seconds = progress_window_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # State tracking
        self._stage = "planning"
        self._chunks_planned = 0
        self._chunks_submitted = 0
        self._chunks_completed = 0
        self._chunks_failed = 0
        self._last_progress_ts_utc: Optional[datetime] = None
        self._last_completed_chunk_id: Optional[int] = None
        
        # File-based progress tracking
        self._last_known_chunk_footers: Set[Path] = set()
        self._last_known_worker_boots: Set[Path] = set()
        self._last_known_join_files: Set[Path] = set()
        self._last_known_exit_status_lines = 0
        
        # Lock for thread-safe updates
        self._lock = threading.Lock()
    
    def update_stage(self, stage: str) -> None:
        """Update current stage (planning, submitting, running, aggregating)."""
        with self._lock:
            self._stage = stage
    
    def update_chunks_planned(self, count: int) -> None:
        """Update number of chunks planned."""
        with self._lock:
            self._chunks_planned = count
    
    def update_chunks_submitted(self, count: int) -> None:
        """Update number of chunks submitted."""
        with self._lock:
            self._chunks_submitted = count
    
    def update_chunks_completed(self, count: int) -> None:
        """Update number of chunks completed."""
        with self._lock:
            self._chunks_completed = count
    
    def update_chunks_failed(self, count: int) -> None:
        """Update number of chunks failed."""
        with self._lock:
            self._chunks_failed = count
    
    def update_last_completed_chunk_id(self, chunk_id: int) -> None:
        """Update last completed chunk ID."""
        with self._lock:
            self._last_completed_chunk_id = chunk_id
    
    def start(self) -> None:
        """Start the watchdog thread."""
        if self._thread is not None:
            raise RuntimeError("Watchdog thread already started")
        
        # Write initial WATCHDOG_STATE.json
        self._write_watchdog_state()
        
        # Initialize last_progress_ts to now
        self._last_progress_ts_utc = datetime.now(timezone.utc)
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the watchdog thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
    
    def _write_heartbeat(self) -> None:
        """Write HEARTBEAT.json atomically."""
        with self._lock:
            heartbeat_data = {
                "timestamp_utc": _dt_now_iso(),
                "stage": self._stage,
                "chunks_planned": self._chunks_planned,
                "chunks_submitted": self._chunks_submitted,
                "chunks_completed": self._chunks_completed,
                "chunks_failed": self._chunks_failed,
                "last_progress_ts_utc": self._last_progress_ts_utc.isoformat() if self._last_progress_ts_utc else None,
                "last_completed_chunk_id": self._last_completed_chunk_id,
                "active_children_pids": _get_active_child_pids(),
            }
        
        heartbeat_path = self.output_dir / "HEARTBEAT.json"
        _write_json_atomic(heartbeat_path, heartbeat_data)
    
    def _write_watchdog_state(self) -> None:
        """Write WATCHDOG_STATE.json atomically."""
        state_data = {
            "stall_timeout_seconds": self.stall_timeout_seconds,
            "progress_window_seconds": self.progress_window_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "progress_definition": {
                "signals": [
                    "New chunk_footer.json written (status ok/failed)",
                    "New line appended to CHUNK_EXIT_STATUS.jsonl",
                    "New/updated RAW_PREBUILT_JOIN.json in chunk dir",
                    "New WORKER_BOOT.json in chunk dir",
                ],
                "note": "Progress is detected if at least one of these signals occurs within progress_window_seconds",
            },
        }
        
        state_path = self.output_dir / "WATCHDOG_STATE.json"
        _write_json_atomic(state_path, state_data)
    
    def _write_stall_fatal(self, seconds_since_progress: float) -> None:
        """Write RUN_STALL_FATAL.json and trigger exit."""
        now_utc = datetime.now(timezone.utc)
        
        with self._lock:
            stage = self._stage
            counts = {
                "planned": self._chunks_planned,
                "submitted": self._chunks_submitted,
                "completed": self._chunks_completed,
                "failed": self._chunks_failed,
            }
        
        chunk_artifacts_snapshot = _get_chunk_artifacts_snapshot(self.output_dir)
        ps_snapshot = _get_ps_snapshot()
        
        fatal_data = {
            "status": "STALLED",
            "now_utc": now_utc.isoformat(),
            "last_progress_utc": self._last_progress_ts_utc.isoformat() if self._last_progress_ts_utc else None,
            "seconds_since_progress": seconds_since_progress,
            "stage": stage,
            "counts": counts,
            "chunk_artifacts_snapshot": chunk_artifacts_snapshot,
            "ps_snapshot": ps_snapshot,
            "stall_timeout_seconds": self.stall_timeout_seconds,
        }
        
        fatal_path = self.output_dir / "RUN_STALL_FATAL.json"
        _write_json_atomic(fatal_path, fatal_data)
        
        # Also write RUN_FAILED.json marker
        run_failed_path = self.output_dir / "RUN_FAILED.json"
        run_failed_data = {
            "status": "FAILED",
            "reason": "STALLED",
            "stall_seconds": seconds_since_progress,
            "timestamp": now_utc.isoformat(),
        }
        _write_json_atomic(run_failed_path, run_failed_data)
        
        # Exit with code 2
        sys.exit(2)
    
    def _run(self) -> None:
        """Main watchdog loop."""
        while not self._stop_event.is_set():
            try:
                # Detect progress
                progress_detected, new_footers, new_boots, new_joins, new_exit_status_lines = _detect_progress(
                    self.output_dir,
                    self._last_known_chunk_footers,
                    self._last_known_worker_boots,
                    self._last_known_join_files,
                    self._last_known_exit_status_lines,
                )
                
                if progress_detected:
                    # Update last progress timestamp
                    self._last_progress_ts_utc = datetime.now(timezone.utc)
                    
                    # Update known files
                    self._last_known_chunk_footers.update(new_footers)
                    self._last_known_worker_boots.update(new_boots)
                    self._last_known_join_files.update(new_joins)
                    self._last_known_exit_status_lines = new_exit_status_lines
                
                # Check for stall
                if self._last_progress_ts_utc is not None:
                    now_utc = datetime.now(timezone.utc)
                    seconds_since_progress = (now_utc - self._last_progress_ts_utc).total_seconds()
                    
                    if seconds_since_progress > self.stall_timeout_seconds:
                        # STALL DETECTED: write fatal and exit
                        self._write_stall_fatal(seconds_since_progress)
                        return  # Exit thread (sys.exit(2) was called)
                
                # Write heartbeat
                self._write_heartbeat()
                
                # Sleep until next heartbeat
                self._stop_event.wait(timeout=self.heartbeat_interval_seconds)
                
            except Exception as e:
                # Log error but continue (watchdog should be resilient)
                try:
                    error_log_path = self.output_dir / "WATCHDOG_ERROR.log"
                    with open(error_log_path, "a") as f:
                        f.write(f"[{_dt_now_iso()}] Watchdog error: {e}\n")
                except Exception:
                    pass  # If we can't even write error log, just continue
        
        # Thread stopped normally
        # Write final heartbeat
        try:
            self._write_heartbeat()
        except Exception:
            pass
