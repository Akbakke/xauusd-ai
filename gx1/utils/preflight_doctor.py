#!/usr/bin/env python3
"""
Preflight Doctor - Run gx1 doctor before expensive operations.

This module provides a helper to run gx1 doctor programmatically
and fail fast with a capsule if checks fail.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add workspace root to path
_workspace_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_workspace_root))


def run_gx1_doctor_or_fatal(
    strict: bool,
    truth_or_smoke: bool,
    output_dir: Optional[Path] = None
) -> None:
    """
    Run gx1 doctor and fail fast if checks fail.
    
    Args:
        strict: Use --strict flag (make warnings into failures)
        truth_or_smoke: If True, always use --strict
        output_dir: Optional output directory for DOCTOR_FATAL.json capsule
    
    Raises:
        RuntimeError: If doctor checks fail (with [DOCTOR_FATAL] prefix)
    """
    # Import doctor module
    try:
        from gx1.tools.gx1_doctor import main as doctor_main, run_checks, resolve_gx1_data_root, find_engine_root
    except ImportError as e:
        raise RuntimeError(
            f"[DOCTOR_FATAL] Cannot import gx1_doctor: {e}. "
            f"This indicates a critical code issue."
        ) from e
    
    # Build argv for doctor
    argv = []
    if strict or truth_or_smoke:
        argv.append("--strict")
    argv.append("--json")
    
    # Capture doctor output
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exit_code = doctor_main(argv)
    except Exception as e:
        # Doctor itself failed (not just checks)
        raise RuntimeError(
            f"[DOCTOR_FATAL] gx1_doctor raised exception: {e}"
        ) from e
    
    # Parse JSON output
    json_output_str = stdout_capture.getvalue()
    try:
        doctor_report = json.loads(json_output_str)
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to get results directly
        try:
            checks, has_failures = run_checks(strict=strict or truth_or_smoke)
            doctor_report = {
                "checks": [c.to_dict() for c in checks],
                "has_failures": has_failures,
                "exit_code": exit_code,
            }
        except Exception as inner_e:
            raise RuntimeError(
                f"[DOCTOR_FATAL] Failed to parse doctor output: {e}. "
                f"Also failed to run checks directly: {inner_e}"
            ) from inner_e
    
    # Check if doctor found failures
    if exit_code == 2 or doctor_report.get("has_failures", False):
        # Write DOCTOR_FATAL.json capsule
        capsule_path = _write_doctor_fatal_capsule(
            doctor_report=doctor_report,
            output_dir=output_dir,
            strict=strict or truth_or_smoke,
        )
        
        # Build error message
        failed_checks = [c for c in doctor_report.get("checks", []) if c.get("status") == "FAIL"]
        failed_names = [c.get("name", "unknown") for c in failed_checks]
        
        error_msg = (
            f"[DOCTOR_FATAL] gx1 doctor found {len(failed_checks)} blocking failures: {', '.join(failed_names)}. "
            f"Details written to {capsule_path}"
        )
        
        raise RuntimeError(error_msg)
    
    # Success - log one line
    print("[DOCTOR] doctor_ok")


def _write_doctor_fatal_capsule(
    doctor_report: dict,
    output_dir: Optional[Path],
    strict: bool,
) -> Path:
    """
    Write DOCTOR_FATAL.json capsule with full diagnostic information.
    
    Returns:
        Path to written capsule file
    """
    # Resolve paths
    try:
        gx1_data_root, _ = resolve_gx1_data_root()
        gx1_data_root_str = str(gx1_data_root) if gx1_data_root else None
    except Exception:
        gx1_data_root_str = None
    
    try:
        engine_root, _ = find_engine_root()
        engine_root_str = str(engine_root) if engine_root else None
    except Exception:
        engine_root_str = None
    
    # Build capsule payload
    timestamp = datetime.now().isoformat()
    capsule_payload = {
        "reason": "DOCTOR_FATAL",
        "timestamp": timestamp,
        "strict_mode": strict,
        "resolved_gx1_data_root": gx1_data_root_str,
        "resolved_engine_root": engine_root_str,
        "failed_checks": [
            c for c in doctor_report.get("checks", [])
            if c.get("status") == "FAIL" or (c.get("blocking") and c.get("status") == "WARN")
        ],
        "full_doctor_report": doctor_report,
        "argv_hints": {
            "sys.argv": sys.argv[:5],  # First 5 args only
            "script_name": sys.argv[0] if sys.argv else None,
        },
        "env_hints": {
            "GX1_DATA_DIR": os.environ.get("GX1_DATA_DIR"),
            "GX1_DATA_ROOT": os.environ.get("GX1_DATA_ROOT"),
            "GX1_REPLAY_MODE": os.environ.get("GX1_REPLAY_MODE"),
            "GX1_OUTPUT_MODE": os.environ.get("GX1_OUTPUT_MODE"),
            "HOME": os.environ.get("HOME"),
        },
    }
    
    # Determine capsule path
    if output_dir:
        output_dir = Path(output_dir)
        # Try to resolve, but if it fails (e.g., parent doesn't exist), use as-is
        try:
            output_dir = output_dir.resolve()
        except (OSError, RuntimeError):
            pass  # Use unresolved path
        # Try to create directory (with parents), but if it fails, still try to write capsule
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError, FileNotFoundError) as mkdir_error:
            # If we can't create the directory (e.g., parent /nonexistent/path doesn't exist),
            # try to create just the capsule file's parent directory
            try:
                # Try to create parent of output_dir if it's a file path issue
                if not output_dir.parent.exists():
                    output_dir.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                # If that also fails, we'll try to write the capsule anyway
                # (might fail, but at least we tried)
                pass
        capsule_path = output_dir / "DOCTOR_FATAL.json"
    else:
        # Write to GX1_DATA/reports/_capsules/
        if gx1_data_root:
            capsules_dir = gx1_data_root / "reports" / "_capsules"
            capsules_dir.mkdir(parents=True, exist_ok=True)
            capsule_path = capsules_dir / f"DOCTOR_FATAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            # Fallback to /tmp
            capsule_path = Path(f"/tmp/DOCTOR_FATAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Write capsule atomically
    temp_path = capsule_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(capsule_payload, f, indent=2, sort_keys=True)
    temp_path.replace(capsule_path)
    
    return capsule_path
