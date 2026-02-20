#!/usr/bin/env python3
"""
Run index ledger - append-only log of all runs.

Maintains a JSON Lines file at $GX1_DATA/reports/_index.jsonl with one entry per run.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

INDEX_FILENAME = "_index.jsonl"


def build_run_index_entry(output_dir: Path) -> dict:
    """
    Build a minimal run index entry from output directory.
    
    Reads available files from output_dir and constructs a minimal entry.
    
    Args:
        output_dir: Path to run output directory
    
    Returns:
        Dictionary with run metadata
    """
    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "kind": "unknown",
        "run_id": output_dir.name,
        "status": "UNKNOWN",
    }
    
    # Determine kind from path (e.g., "replay_eval", "us_disabled_proof")
    # Path structure: $GX1_DATA/reports/<kind>/<run_id>
    try:
        resolved = output_dir.resolve()
        parts = resolved.parts
        # Find "reports" in path and take next segment as kind
        if "reports" in parts:
            reports_idx = parts.index("reports")
            if reports_idx + 1 < len(parts):
                entry["kind"] = parts[reports_idx + 1]
    except Exception:
        pass  # Keep default "unknown"
    
    # Determine status and event from completion files
    run_completed = output_dir / "RUN_COMPLETED.json"
    run_failed = output_dir / "RUN_FAILED.json"
    master_fatal = output_dir / "MASTER_FATAL.json"
    doctor_fatal = output_dir / "DOCTOR_FATAL.json"
    
    event = "UNKNOWN"
    if run_completed.exists():
        entry["status"] = "COMPLETED"
        event = "RUN_COMPLETED"
    elif master_fatal.exists():
        entry["status"] = "FAILED"
        event = "MASTER_FATAL"
    elif doctor_fatal.exists():
        entry["status"] = "FAILED"
        event = "DOCTOR_FATAL"
    elif run_failed.exists():
        entry["status"] = "FAILED"
        event = "RUN_FAILED"
    
    entry["event"] = event
    
    # Read RUN_IDENTITY.json if available
    run_identity = output_dir / "RUN_IDENTITY.json"
    if run_identity.exists():
        try:
            with open(run_identity, "r", encoding="utf-8") as f:
                identity = json.load(f)
                entry["output_mode"] = identity.get("output_mode", "MINIMAL")
                entry["git_head_sha"] = identity.get("git_head_sha")
                entry["replay_mode"] = identity.get("replay_mode")
                entry["prebuilt_used"] = identity.get("prebuilt_used")
        except Exception as e:
            log.warning(f"Failed to read RUN_IDENTITY.json from {output_dir}: {e}")
    
    # Try to find and read metrics file
    metrics_patterns = [
        "*_METRICS.json",
        "metrics_*_MERGED.json",
        "metrics_*.json",
    ]
    
    metrics_data = None
    for pattern in metrics_patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            # Use first match (most specific pattern first)
            try:
                with open(matches[0], "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                break
            except Exception as e:
                log.warning(f"Failed to read metrics file {matches[0]}: {e}")
                continue
    
    # Extract key metrics if available
    if metrics_data:
        # Try to find metrics in various possible structures
        metrics_fields = [
            "total_pnl_bps",
            "max_dd_bps",
            "trades",
            "winrate",
            "bars_seen",
            "bars_processed",
        ]
        
        # Check if metrics_data is a dict with nested structure
        if isinstance(metrics_data, dict):
            # Try top-level first
            for field in metrics_fields:
                if field in metrics_data:
                    entry[field] = metrics_data[field]
            
            # Try nested under "summary" or "aggregated"
            for key in ["summary", "aggregated", "metrics"]:
                if key in metrics_data and isinstance(metrics_data[key], dict):
                    for field in metrics_fields:
                        if field in metrics_data[key] and field not in entry:
                            entry[field] = metrics_data[key][field]
    
    # Read failure capsules if status is FAILED
    if entry["status"] == "FAILED":
        if master_fatal.exists():
            try:
                with open(master_fatal, "r", encoding="utf-8") as f:
                    fatal_data = json.load(f)
                    # Try multiple possible keys for fatal_reason
                    entry["fatal_reason"] = (
                        fatal_data.get("fatal_reason") or
                        fatal_data.get("reason") or
                        fatal_data.get("error_type") or
                        fatal_data.get("error")
                    )
                    # Try multiple possible keys for error message
                    error_msg = (
                        fatal_data.get("error_message") or
                        fatal_data.get("message") or
                        fatal_data.get("traceback", "").split("\n")[0] if fatal_data.get("traceback") else ""
                    )
                    if error_msg:
                        # Take first line as hint (max 200 chars)
                        entry["error_hint"] = error_msg.split("\n")[0][:200]
            except Exception as e:
                log.warning(f"Failed to read MASTER_FATAL.json from {output_dir}: {e}")
        elif run_failed.exists():
            try:
                with open(run_failed, "r", encoding="utf-8") as f:
                    failed_data = json.load(f)
                    # Try multiple possible keys for fatal_reason
                    entry["fatal_reason"] = (
                        failed_data.get("fatal_reason") or
                        failed_data.get("reason") or
                        failed_data.get("error_type") or
                        failed_data.get("error")
                    )
                    # Try multiple possible keys for error message
                    error_msg = (
                        failed_data.get("error_message") or
                        failed_data.get("message") or
                        failed_data.get("traceback", "").split("\n")[0] if failed_data.get("traceback") else ""
                    )
                    if error_msg:
                        entry["error_hint"] = error_msg.split("\n")[0][:200]
            except Exception as e:
                log.warning(f"Failed to read RUN_FAILED.json from {output_dir}: {e}")
    
    # Generate entry_id: SHA256 of run_id|event|git_head_sha|canonical_output_dir
    # Use canonical output_dir (realpath + normalized path separator)
    # Use event (not status) as it's the stable identifier
    canonical_output_dir = str(Path(entry["output_dir"]).resolve()).replace("\\", "/")
    entry_id_parts = [
        entry["run_id"],
        entry.get("event", "UNKNOWN"),
        entry.get("git_head_sha") or "",
        canonical_output_dir,
    ]
    entry_id_str = "|".join(entry_id_parts)
    entry_id = hashlib.sha256(entry_id_str.encode("utf-8")).hexdigest()
    entry["entry_id"] = entry_id
    
    return entry


def append_run_index(gx1_data_root: Path, entry: dict) -> None:
    """
    Append a run index entry to the ledger file.
    
    Uses file locking (fcntl.flock) to ensure atomic writes even with parallel runs.
    
    Args:
        gx1_data_root: Path to GX1_DATA root
        entry: Run index entry dictionary
    
    Raises:
        RuntimeError: If output_dir is not under GX1_DATA/reports/
    """
    reports_root = gx1_data_root / "reports"
    index_path = reports_root / INDEX_FILENAME
    
    # Validate that entry's output_dir is under reports/
    output_dir_str = entry.get("output_dir", "")
    if output_dir_str:
        try:
            output_dir_path = Path(output_dir_str).resolve()
            reports_root_resolved = reports_root.resolve()
            if not str(output_dir_path).startswith(str(reports_root_resolved)):
                raise RuntimeError(
                    f"Output directory {output_dir_path} is not under GX1_DATA/reports/ "
                    f"({reports_root_resolved}). Refusing to index."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to validate output_dir: {e}") from e
    
    # Ensure reports directory exists
    reports_root.mkdir(parents=True, exist_ok=True)
    
    # Open file in append mode with exclusive lock
    try:
        with open(index_path, "a", encoding="utf-8") as f:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Write entry as JSON line
                json_line = json.dumps(entry, sort_keys=True) + "\n"
                f.write(json_line)
                # Flush and sync to ensure data is written
                f.flush()
                import os
                os.fsync(f.fileno())
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        raise RuntimeError(f"Failed to append to run index {index_path}: {e}") from e


def append_run_index_dedup(
    gx1_data_root: Path,
    entry: dict,
    last_n: int = 2000,
    fullscan_max_bytes: int = 20_000_000,
    debug_stats: Optional[dict] = None,
) -> bool:
    """
    Append a run index entry to the ledger file with deduplication.
    
    Implements "size-threshold fullscan" + fallback:
    - If file size <= fullscan_max_bytes: scan entire file line-by-line, build set() of entry_id
    - Otherwise: fallback to last_n scan (check last N lines)
    
    This prevents duplicate entry_id even if more than last_n entries were added after
    the original entry.
    
    Args:
        gx1_data_root: Path to GX1_DATA root
        entry: Run index entry dictionary (must have entry_id field)
        last_n: Number of recent lines to check for duplicates if file is too large (default: 2000)
        fullscan_max_bytes: Maximum file size for full scan (default: 20MB)
        debug_stats: Optional dict to populate with debug statistics (invalid_lines_ignored count)
    
    Returns:
        True if entry was appended, False if duplicate was found
    
    Raises:
        RuntimeError: If output_dir is not under GX1_DATA/reports/ or entry_id is missing
    """
    if "entry_id" not in entry:
        raise RuntimeError("Entry must have 'entry_id' field for deduplication")
    
    reports_root = gx1_data_root / "reports"
    index_path = reports_root / INDEX_FILENAME
    
    # Validate that entry's output_dir is under reports/
    output_dir_str = entry.get("output_dir", "")
    if output_dir_str:
        try:
            output_dir_path = Path(output_dir_str).resolve()
            reports_root_resolved = reports_root.resolve()
            if not str(output_dir_path).startswith(str(reports_root_resolved)):
                raise RuntimeError(
                    f"Output directory {output_dir_path} is not under GX1_DATA/reports/ "
                    f"({reports_root_resolved}). Refusing to index."
                )
        except Exception as e:
            raise RuntimeError(f"Failed to validate output_dir: {e}") from e
    
    # Ensure reports directory exists
    reports_root.mkdir(parents=True, exist_ok=True)
    
    entry_id = entry["entry_id"]
    invalid_lines_ignored = 0
    dedup_hit = False
    dedup_mode = None
    scanned_lines = 0
    
    # Open file in append mode with exclusive lock
    try:
        with open(index_path, "a+", encoding="utf-8") as f:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Get file size
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                
                if file_size > 0:
                    # Decide: fullscan or last_n fallback
                    if file_size <= fullscan_max_bytes:
                        # Fullscan: read entire file line-by-line
                        dedup_mode = "fullscan"
                        f.seek(0)
                        seen_entry_ids = set()
                        for line in f:
                            scanned_lines += 1
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                existing_entry = json.loads(line)
                                existing_id = existing_entry.get("entry_id")
                                if existing_id:
                                    seen_entry_ids.add(existing_id)
                            except (json.JSONDecodeError, KeyError, TypeError):
                                # Ignore malformed/partial lines, but count them
                                invalid_lines_ignored += 1
                                continue
                        
                        # Check if entry_id already exists
                        if entry_id in seen_entry_ids:
                            dedup_hit = True
                            log.debug(f"Duplicate entry_id {entry_id[:16]}... found in fullscan, skipping append")
                            if debug_stats is not None:
                                debug_stats.update({
                                    "dedup_hit": True,
                                    "dedup_mode": "fullscan",
                                    "scanned_lines": scanned_lines,
                                    "invalid_lines_ignored": invalid_lines_ignored,
                                })
                            return False
                    else:
                        # Fallback: last_n scan (original behavior)
                        dedup_mode = "last_n"
                        
                        # Read last N lines (approximate by reading last chunk)
                        # Each line is typically 200-500 bytes, so read last ~1MB
                        read_size = min(file_size, last_n * 1000)
                        seek_pos = max(0, file_size - read_size)
                        f.seek(seek_pos)
                        
                        # If we didn't seek to start of file, skip first line (may be partial)
                        if seek_pos > 0:
                            f.readline()  # Skip potentially partial first line
                        
                        recent_lines = f.readlines()
                        scanned_lines = len(recent_lines)
                        
                        # Check for duplicate entry_id in recent lines
                        for line in recent_lines:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                existing_entry = json.loads(line)
                                if existing_entry.get("entry_id") == entry_id:
                                    dedup_hit = True
                                    log.debug(f"Duplicate entry_id {entry_id[:16]}... found in last_n scan, skipping append")
                                    if debug_stats is not None:
                                        debug_stats.update({
                                            "dedup_hit": True,
                                            "dedup_mode": "last_n",
                                            "scanned_lines": scanned_lines,
                                            "invalid_lines_ignored": invalid_lines_ignored,
                                        })
                                    return False
                            except (json.JSONDecodeError, KeyError, TypeError):
                                # Skip malformed lines, but count them
                                invalid_lines_ignored += 1
                                continue
                
                # No duplicate found - append entry
                json_line = json.dumps(entry, sort_keys=True) + "\n"
                f.write(json_line)
                # Flush and sync to ensure data is written
                f.flush()
                import os
                os.fsync(f.fileno())
                log.debug(f"Appended entry_id {entry_id[:16]}... to run index")
                if debug_stats is not None:
                    debug_stats.update({
                        "dedup_hit": False,
                        "dedup_mode": dedup_mode or "none",
                        "scanned_lines": scanned_lines,
                        "invalid_lines_ignored": invalid_lines_ignored,
                    })
                return True
            finally:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        raise RuntimeError(f"Failed to append to run index {index_path}: {e}") from e


def read_run_index(gx1_data_root: Path, ignore_partial: bool = True) -> list[dict]:
    """
    Read all entries from the run index ledger.
    
    Args:
        gx1_data_root: Path to GX1_DATA root
        ignore_partial: If True, ignore the last line if it's partial JSON (incomplete write)
    
    Returns:
        List of run index entries (dictionaries)
    """
    reports_root = gx1_data_root / "reports"
    index_path = reports_root / INDEX_FILENAME
    
    if not index_path.exists():
        return []
    
    entries = []
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            # Process all lines except possibly the last one
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is the last line and we should ignore partial
                is_last_line = (i == len(lines) - 1)
                if is_last_line and ignore_partial:
                    # Check if line looks incomplete (doesn't end with } or ])
                    if not (line.endswith("}") or line.endswith("]")):
                        log.debug(f"Ignoring partial last line in {index_path}")
                        continue
                
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse JSON line {i+1} in {index_path}: {e}")
                    if not is_last_line:
                        # Non-last line with parse error is suspicious
                        log.warning(f"Skipping malformed line: {line[:100]}")
                    continue
    except Exception as e:
        log.error(f"Failed to read run index {index_path}: {e}")
        raise RuntimeError(f"Failed to read run index: {e}") from e
    
    return entries
