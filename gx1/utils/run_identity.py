"""
Run identity utilities for generating stable run_id and chunk_id.

Used for generating globally unique trade_uid across parallel replay chunks.
"""
import os
import re
from pathlib import Path
from typing import Optional


def get_run_id(output_dir: Optional[Path] = None, env_run_id: Optional[str] = None) -> str:
    """
    Get stable run_id for this run.
    
    Priority:
    1. Environment variable GX1_RUN_ID
    2. Policy config run_id
    3. For replay: basename of output_dir (if output_dir contains "FULLYEAR" or similar pattern)
    4. Auto-generated timestamp-based ID
    
    Parameters
    ----------
    output_dir : Path, optional
        Output directory for this run (used to infer run_id from path)
    env_run_id : str, optional
        run_id from environment variable (GX1_RUN_ID)
    
    Returns
    -------
    str
        Stable run_id for this run
    """
    # Priority 1: Environment variable
    if env_run_id:
        return env_run_id
    
    # Priority 2: Output dir basename (for replay runs with structured output dirs)
    if output_dir:
        output_dir = Path(output_dir)
        # Use basename if it looks like a run identifier (contains timestamp pattern)
        basename = output_dir.name
        # Check if basename looks like a run tag (e.g., FULLYEAR_2025_20260105_190429)
        if re.match(r'^[A-Z_0-9]+_\d{8}_\d{6}', basename) or 'FULLYEAR' in basename or 'SNIPER' in basename:
            return basename
        # If output_dir is inside a run directory, use parent basename
        if output_dir.parent.name and (output_dir.parent.name.startswith('FULLYEAR') or output_dir.parent.name.startswith('SNIPER')):
            return output_dir.parent.name
    
    # Priority 3: Auto-generate (fallback, should rarely happen)
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

