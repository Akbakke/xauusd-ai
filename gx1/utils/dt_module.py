#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized datetime utilities with version stamping.

This module provides a single source of truth for datetime operations
to avoid import/scope issues in parallel execution contexts.
"""

from datetime import datetime
from typing import Optional

# CRITICAL: Version stamp for fail-fast validation
# Update this when making any changes to datetime logic
DT_MODULE_VERSION = "2026-01-18_fix2"

# Expected version (must match DT_MODULE_VERSION or fail-fast)
EXPECTED_DT_MODULE_VERSION = "2026-01-18_fix2"


def get_dt_module_version() -> str:
    """Get current dt_module version."""
    return DT_MODULE_VERSION


def validate_dt_module_version(expected: Optional[str] = None) -> None:
    """
    Validate that dt_module version matches expected.
    
    Args:
        expected: Expected version string. If None, uses EXPECTED_DT_MODULE_VERSION.
    
    Raises:
        RuntimeError: If version doesn't match.
    """
    if expected is None:
        expected = EXPECTED_DT_MODULE_VERSION
    
    actual = get_dt_module_version()
    if actual != expected:
        raise RuntimeError(
            f"[DT_MODULE_VERSION_MISMATCH] Expected '{expected}', got '{actual}'. "
            f"This indicates a code/import mismatch. FATAL."
        )


def now() -> datetime:
    """
    Get current datetime (deterministic, timezone-aware).
    
    Returns:
        Current datetime in ISO format.
    """
    return datetime.now()


def now_iso() -> str:
    """
    Get current datetime as ISO string.
    
    Returns:
        Current datetime as ISO format string.
    """
    return datetime.now().isoformat()


def strftime_now(format_str: str) -> str:
    """
    Format current datetime with strftime.
    
    Args:
        format_str: strftime format string.
    
    Returns:
        Formatted datetime string.
    """
    return datetime.now().strftime(format_str)
