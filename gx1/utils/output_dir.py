#!/usr/bin/env python3
"""
Output directory resolution helper.

Provides consistent logic for resolving GX1_DATA root and output directories
across all replay/smoke/truth scripts.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def resolve_gx1_data_root() -> Path:
    """
    Resolve GX1_DATA root using standard logic.
    
    Logic: GX1_DATA_DIR → GX1_DATA_ROOT → ~/GX1_DATA
    
    Returns:
        Path to GX1_DATA root
    
    Raises:
        RuntimeError: If basename != "GX1_DATA"
    """
    gx1_data_env = os.environ.get("GX1_DATA_DIR") or os.environ.get("GX1_DATA_ROOT")
    gx1_data = Path(gx1_data_env) if gx1_data_env else Path.home() / "GX1_DATA"
    gx1_data = gx1_data.expanduser().resolve()
    
    if gx1_data.name != "GX1_DATA":
        raise RuntimeError(
            f"GX1_DATA root must end with 'GX1_DATA', got: {gx1_data.name}. "
            f"Resolved path: {gx1_data}. "
            f"Set GX1_DATA_DIR or GX1_DATA_ROOT to a path ending with 'GX1_DATA'."
        )
    
    return gx1_data


def make_run_id(prefix: str) -> str:
    """
    Generate a run ID with prefix and timestamp.
    
    Args:
        prefix: Prefix for run ID (e.g., "SMOKE_US_DISABLED", "REPLAY_EVAL")
    
    Returns:
        Run ID string: PREFIX_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def resolve_output_dir(
    kind: str,
    prefix: str,
    output_dir: Optional[str],
    truth_or_smoke: bool = False,
) -> Path:
    """
    Resolve output directory with validation.
    
    Args:
        kind: Directory kind (e.g., "replay_eval", "us_disabled_proof")
        prefix: Prefix for run_id if auto-generating (e.g., "REPLAY_EVAL", "SMOKE_US_DISABLED")
        output_dir: Optional explicit output directory path
        truth_or_smoke: If True, require explicit output_dir (no auto-generation)
    
    Returns:
        Resolved Path to output directory
    
    Raises:
        RuntimeError: If validation fails or TRUTH/SMOKE requires explicit output_dir
    """
    gx1_data_root = resolve_gx1_data_root()
    reports_root = gx1_data_root / "reports"
    
    if output_dir:
        # Explicit output_dir provided - validate it's under GX1_DATA/reports/
        output_path = Path(output_dir).expanduser().resolve()
        
        # Check that it's under reports/
        try:
            reports_root_resolved = reports_root.resolve()
            if not str(output_path).startswith(str(reports_root_resolved)):
                raise RuntimeError(
                    f"Output directory must be under GX1_DATA/reports/, got: {output_path}. "
                    f"GX1_DATA/reports/ is: {reports_root_resolved}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to validate output directory: {e}"
            ) from e
        
        return output_path
    
    # No explicit output_dir - check if auto-generation is allowed
    if truth_or_smoke:
        raise RuntimeError(
            f"TRUTH/SMOKE mode requires explicit --output-dir. "
            f"Auto-generation is not allowed for deterministic runs. "
            f"Example: --output-dir {reports_root}/{kind}/MY_RUN_ID"
        )
    
    # Auto-generate output_dir
    run_id = make_run_id(prefix)
    output_path = reports_root / kind / run_id
    
    return output_path
