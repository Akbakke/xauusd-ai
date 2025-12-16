#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path resolver for PROD_BASELINE mode.

Resolves all paths relative to gx1/prod/current/ when meta.role == PROD_BASELINE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


PROD_CURRENT_DIR = Path(__file__).parent / "current"


def resolve_prod_path(
    path_str: str,
    prod_baseline: bool,
    base_path: Optional[Path] = None,
) -> Path:
    """
    Resolve path for PROD_BASELINE mode.
    
    Args:
        path_str: Path string (may be relative or absolute)
        prod_baseline: Whether we're in PROD_BASELINE mode
        base_path: Base path for resolution (default: PROD_CURRENT_DIR)
        
    Returns:
        Resolved Path
    """
    if not prod_baseline:
        # Not PROD_BASELINE: use path as-is
        return Path(path_str)
    
    # PROD_BASELINE: resolve relative to gx1/prod/current/
    base = base_path or PROD_CURRENT_DIR
    
    path = Path(path_str)
    
    # If absolute path, check if it's within repo
    if path.is_absolute():
        # Try to resolve relative to prod/current
        try:
            relative = path.relative_to(Path.cwd())
            # If it starts with gx1/, resolve from prod/current
            if str(relative).startswith("gx1/"):
                # Extract path after gx1/
                suffix = str(relative)[5:]  # Remove "gx1/"
                return base / suffix
        except ValueError:
            pass
        # If can't resolve, return as-is (external path)
        return path
    
    # Relative path: resolve from prod/current
    return (base / path).resolve()


def resolve_model_path(
    model_path: Optional[str],
    prod_baseline: bool,
) -> Optional[str]:
    """
    Resolve model path for router.
    
    Args:
        model_path: Model path from config (may be None)
        prod_baseline: Whether we're in PROD_BASELINE mode
        
    Returns:
        Resolved model path string (or None)
    """
    if model_path is None:
        return None
    
    resolved = resolve_prod_path(model_path, prod_baseline)
    return str(resolved)

