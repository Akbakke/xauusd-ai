"""
Run identity utilities for generating stable run_id and chunk_id.

Used for generating globally unique trade_uid across parallel replay chunks.

SSoT: run_id must come from CLI (--run-id) or GX1_RUN_ID. NEVER derive from
Path(run_root).name / basename - causes merge/lookup bugs in sweep/aggregator.
"""
import os
import re
from pathlib import Path
from typing import Optional


def get_run_id(output_dir: Optional[Path] = None, env_run_id: Optional[str] = None) -> str:
    """
    Get stable run_id for this run.

    Priority:
    1. Environment variable GX1_RUN_ID (or env_run_id param)
    2. Auto-generated timestamp-based ID

    NEVER uses output_dir.name/basename - run_id must be explicit SSoT from CLI/env.
    """
    # Priority 1: Environment variable / explicit pass
    if env_run_id:
        return env_run_id

    # Priority 2: Auto-generate (never derive from output_dir basename)
    from datetime import datetime, timezone
    ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts_str}"


def get_chunk_id(output_dir: Optional[Path] = None, env_chunk_id: Optional[str] = None) -> str:
    """
    Get chunk_id for parallel replay chunks.
    
    Priority:
    1. Environment variable GX1_CHUNK_ID
    2. Parse from output_dir path (e.g., .../chunk_0/... -> "chunk_0")
    3. Return "single" for non-parallel runs
    
    Parameters
    ----------
    output_dir : Path, optional
        Output directory (may contain "chunk_XXX" in path)
    env_chunk_id : str, optional
        chunk_id from environment variable (GX1_CHUNK_ID)
    
    Returns
    -------
    str
        chunk_id (e.g., "chunk_0", "chunk_1", or "single")
    """
    # Priority 1: Environment variable
    if env_chunk_id:
        return env_chunk_id
    
    # Priority 2: Parse from output_dir path
    if output_dir:
        output_dir = Path(output_dir)
        output_dir_str = str(output_dir)
        # Match chunk_(\d+) anywhere in path
        chunk_match = re.search(r'chunk_(\d+)', output_dir_str)
        if chunk_match:
            chunk_idx = int(chunk_match.group(1))
            return f"chunk_{chunk_idx:03d}"
    
    # Priority 3: Fallback to "single" for non-parallel runs
    return "single"

