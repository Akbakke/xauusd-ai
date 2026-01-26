"""
Rolling operation logging for identifying pandas rolling hotspots.

This module provides logging functionality to track pandas rolling operations
before they execute, to identify which rolling call is causing timeouts/hangs.
"""

import json
import logging
import os
import inspect
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_SEGFAULT_CAPSULE_DIR: Optional[Path] = None
_LAST_GOOD_CONTEXT: Optional[dict] = None


def _init_segfault_capsule_dir() -> Path:
    """Initialize and return the segfault capsule directory."""
    global _SEGFAULT_CAPSULE_DIR
    if _SEGFAULT_CAPSULE_DIR is None:
        capsule_dir = os.getenv("GX1_SEGFAULT_CAPSULE_DIR", "data/temp/segfault_capsule")
        _SEGFAULT_CAPSULE_DIR = Path(capsule_dir)
        _SEGFAULT_CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
    return _SEGFAULT_CAPSULE_DIR


def log_rolling_context(
    feature_name: str,
    series,
    window: int,
    min_periods: Optional[int] = None,
    fn_name: Optional[str] = None,
    rolling_expr: Optional[str] = None,
    caller_line: Optional[str] = None,
):
    """
    Log context before a pandas rolling operation (Del 3).
    
    Args:
        feature_name: Name of the feature being built
        series: The pandas Series being rolled
        window: Rolling window size
        min_periods: Minimum periods (if specified)
        fn_name: Function name calling rolling (if known)
        rolling_expr: Expression string (e.g., "close.rolling(3).std(ddof=0)")
        caller_line: Caller line info (function name + line number)
    """
    global _LAST_GOOD_CONTEXT
    try:
        capsule_dir = _init_segfault_capsule_dir()
        
        # Extract context from series
        series_len = len(series) if hasattr(series, '__len__') else 0
        last_idx = series_len - 1 if series_len > 0 else -1
        dtype = str(series.dtype) if hasattr(series, 'dtype') else 'unknown'
        
        # Get last timestamp if available
        last_timestamp = None
        if hasattr(series, 'index') and len(series.index) > 0:
            last_timestamp = str(series.index[-1])
        
        # Count NaN/Inf
        nan_count = 0
        inf_count = 0
        if hasattr(series, 'values'):
            values = series.values
            nan_count = int(sum(1 for v in values if hasattr(v, '__bool__') and (v != v or (hasattr(values, 'dtype') and 'float' in str(values.dtype) and (v == float('inf') or v == float('-inf'))))))
            # More robust NaN/Inf counting
            try:
                import numpy as np
                if hasattr(np, 'isnan') and hasattr(np, 'isinf'):
                    nan_count = int(np.sum(np.isnan(values)))
                    inf_count = int(np.sum(np.isinf(values)))
            except Exception:
                pass
        
        # Build context dictionary
        context = {
            "feature_name": feature_name,
            "fn_name": fn_name or "unknown",
            "rolling_expr": rolling_expr or f"series.rolling({window})",
            "caller_line": caller_line or "unknown",
            "window": window,
            "min_periods": min_periods,
            "series_len": series_len,
            "last_idx": last_idx,
            "last_timestamp": last_timestamp,
            "dtype": dtype,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
        
        # Add min/max if numeric
        if hasattr(series, 'min') and hasattr(series, 'max'):
            try:
                context["min_value"] = float(series.min())
                context["max_value"] = float(series.max())
            except Exception:
                pass
        
        _LAST_GOOD_CONTEXT = context
        
        # Write to last_good.json
        last_good_path = capsule_dir / "last_good.json"
        with open(last_good_path, "w") as f:
            json.dump(context, f, indent=2)
        
        log.debug(
            "[ROLLING_HOTSPOT] feature=%s expr=%s caller=%s idx=%d len=%d window=%d dtype=%s",
            feature_name, rolling_expr or f"rolling({window})", caller_line or "unknown",
            last_idx, series_len, window, dtype
        )
    except Exception as e:
        log.warning("[ROLLING_HOTSPOT] Failed to log rolling context: %s", e)


def get_caller_info() -> str:
    """Get caller function name and line number for logging."""
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_frame = frame.f_back.f_back
            func_name = caller_frame.f_code.co_name
            line_no = caller_frame.f_lineno
            return f"{func_name}:{line_no}"
    except Exception:
        pass
    return "unknown"

