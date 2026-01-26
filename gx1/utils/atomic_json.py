"""
Atomic JSON write utility for crash capsules and telemetry.

Ensures all JSON files are written atomically and are always valid JSON,
even if the process crashes mid-write.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback


def _to_json_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Handles:
    - Path -> str
    - numpy scalars -> int/float
    - sets/tuples -> list
    - exceptions -> repr
    - None -> None
    - pandas Timestamp/DatetimeIndex -> str
    """
    import numpy as np
    
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [_to_json_serializable(item) for item in sorted(obj)]
    elif isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, Exception):
        return {
            "type": type(obj).__name__,
            "message": str(obj),
            "repr": repr(obj),
        }
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int8, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        # Try pandas types (lazy import)
        try:
            import pandas as pd
            if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
                return str(obj)
        except ImportError:
            pass
        
        # Fallback: try to convert to string
        try:
            return str(obj)
        except Exception:
            return repr(obj)


def atomic_write_json(path: Path, payload: Dict[str, Any], fallback_on_error: bool = True) -> bool:
    """
    Write JSON file atomically (tmp -> rename).
    
    Args:
        path: Target file path
        payload: Dictionary to write as JSON
        fallback_on_error: If True, write fallback txt file on error
    
    Returns:
        True if successful, False otherwise
    """
    import pandas as pd
    
    # Convert payload to JSON-serializable format
    try:
        serializable_payload = _to_json_serializable(payload)
    except Exception as convert_error:
        if fallback_on_error:
            _write_fallback_error(path, convert_error, payload, "json_serialization")
        return False
    
    # Ensure parent directory exists
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as dir_error:
        if fallback_on_error:
            _write_fallback_error(path, dir_error, payload, "directory_creation")
        return False
    
    # Write to temp file first
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable_payload, f, indent=2, ensure_ascii=False)
            f.flush()  # Force write to OS buffer
            os.fsync(f.fileno())  # Force write to disk
        
        # Atomic rename (POSIX guarantees this is atomic)
        tmp_path.replace(path)
        return True
    except Exception as write_error:
        if fallback_on_error:
            _write_fallback_error(path, write_error, payload, "file_write")
        # Clean up temp file if it exists
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False


def _write_fallback_error(path: Path, error: Exception, payload: Any, error_stage: str) -> None:
    """Write fallback error file when atomic_write_json fails."""
    try:
        fallback_path = path.parent / f"{path.stem}_WRITE_FAIL.txt"
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(f"Failed to write {path.name} at stage: {error_stage}\n\n")
            f.write(f"Error: {type(error).__name__}: {error}\n\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
            f.write(f"Payload keys (if dict): {list(payload.keys()) if isinstance(payload, dict) else 'N/A'}\n")
            f.write(f"Payload type: {type(payload).__name__}\n")
            f.write(f"Payload repr (first 1000 chars): {repr(payload)[:1000]}\n")
    except Exception:
        # Give up - can't even write fallback
        pass
