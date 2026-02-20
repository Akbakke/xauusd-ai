"""
Import failure capsule utilities.

Provides fail-fast observability for import errors in replay workers.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def write_import_fail_capsule(
    context: str,
    exc: Exception,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Write import failure capsule with full diagnostic information.
    
    Args:
        context: Context string (e.g., "REPLAY_BOOT", "WORKER_IMPORT")
        exc: Exception that occurred
        output_dir: Optional output directory (defaults to reports/import_fail_capsules/)
    
    Returns:
        Path to written capsule JSON file
    """
    # Determine output directory
    if output_dir is None:
        workspace_root = Path(__file__).resolve().parents[2]
        output_dir = workspace_root / "gx1" / "reports" / "import_fail_capsules"
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fallback to /tmp if reports path doesn't exist
    if not output_dir.exists():
        output_dir = Path("/tmp")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    capsule_path = output_dir / f"IMPORT_FAIL_{timestamp}.json"
    
    # Collect diagnostic information
    capsule_data = {
        "context": context,
        "timestamp": datetime.now().isoformat(),
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
        },
        "system": {
            "sys_executable": sys.executable,
            "sys_version": sys.version,
            "cwd": os.getcwd(),
            "argv": sys.argv.copy() if hasattr(sys, "argv") else None,
        },
        "environment": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "GX1_DATA_ROOT": os.environ.get("GX1_DATA_ROOT"),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
            "GX1_RUN_MODE": os.environ.get("GX1_RUN_MODE"),
            "GX1_REPLAY_USE_PREBUILT_FEATURES": os.environ.get("GX1_REPLAY_USE_PREBUILT_FEATURES"),
        },
        "python_path": {
            "sys_path_count": len(sys.path),
            "sys_path_first_20": sys.path[:20],
            "sys_path_all": sys.path if len(sys.path) <= 50 else sys.path[:50] + [f"... ({len(sys.path) - 50} more)"],
        },
    }
    
    # Try to get gx1 module info if possible
    try:
        import gx1
        capsule_data["gx1_module"] = {
            "__file__": getattr(gx1, "__file__", None),
            "__path__": list(getattr(gx1, "__path__", [])),
        }
    except Exception:
        capsule_data["gx1_module"] = {"error": "Could not import gx1"}
    
    # Write capsule
    with open(capsule_path, "w") as f:
        json.dump(capsule_data, f, indent=2)
    
    return capsule_path
