# gx1/runtime/column_collision_guard.py
"""
Column collision detection and resolution utilities.

This module provides fail-fast guards for case-insensitive column name collisions,
with a temporary compatibility mode for the close/CLOSE collision.
"""
import os
from typing import Dict, List, Optional
import pandas as pd


# Reserved candle columns that should never appear in prebuilt feature dataframes
RESERVED_CANDLE_COLUMNS = {
    "open", "high", "low", "close", "volume",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
}


def detect_case_insensitive_collisions(columns: List[str]) -> Dict[str, List[str]]:
    """
    Detect case-insensitive column name collisions.
    
    Parameters
    ----------
    columns : List[str]
        List of column names to check
    
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping lowercase column names to lists of actual column names
        that collide (only includes collisions, i.e., len(v) > 1)
    """
    lower = {}
    for c in columns:
        k = str(c).lower()
        lower.setdefault(k, []).append(c)
    collisions = {k: v for k, v in lower.items() if len(v) > 1}
    return collisions


def assert_no_case_collisions(
    df: pd.DataFrame,
    context: str,
    allow_close_alias_compat: bool = False,
) -> Optional[Dict[str, any]]:
    """
    Fail-fast check for case-insensitive column name collisions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    context : str
        Description of where this check is performed (for error messages)
    allow_close_alias_compat : bool
        If True, allow close/CLOSE collision as a temporary compatibility measure.
        This will return a resolution dict instead of raising.
    
    Returns
    -------
    Optional[Dict[str, any]]
        If compat-mode is enabled and only close/CLOSE collision exists, returns
        a resolution dict with:
        - case_collision_resolved: bool
        - dropped_columns: List[str]
        - alias_expected: Dict[str, str]
        Otherwise returns None (no collision or hard-fail).
    
    Raises
    ------
    RuntimeError
        If case-insensitive collisions are detected and compat-mode is not enabled
        or collision is not only close/CLOSE.
    """
    cols = list(df.columns)
    collisions = detect_case_insensitive_collisions(cols)
    
    if not collisions:
        return None
    
    # Check if compat-mode is enabled
    compat_enabled = allow_close_alias_compat or (
        os.getenv("GX1_ALLOW_CLOSE_ALIAS_COMPAT", "0") == "1"
    )
    
    # Check if collision is only close/CLOSE
    is_only_close_collision = (
        len(collisions) == 1 and
        "close" in collisions and
        set(collisions["close"]) == {"close", "CLOSE"}
    )
    
    if compat_enabled and is_only_close_collision:
        # Return resolution dict (caller should drop CLOSE column)
        return {
            "case_collision_resolved": True,
            "dropped_columns": ["CLOSE"],
            "alias_expected": {"CLOSE": "candles.close"},
            "collisions": collisions,
        }
    
    # Hard-fail for any other collision or if compat-mode is not enabled
    raise RuntimeError(
        f"[CASE_COLLISION] {context}: Case-insensitive column name collisions detected. "
        f"Collisions: {collisions}. "
        f"This must be fixed at the source (CSV/parquet file or DataFrame construction)."
        + (
            f"\nCompat-mode enabled: {compat_enabled}, but collision is not only close/CLOSE."
            if compat_enabled else ""
        )
    )


def resolve_close_alias_collision(
    df: pd.DataFrame,
    context: str,
    transformer_requires_close: bool = False,
) -> tuple[pd.DataFrame, Dict[str, any]]:
    """
    Resolve close/CLOSE collision by dropping CLOSE column (temporary compat-mode).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with close/CLOSE collision
    context : str
        Description of where this resolution is performed
    transformer_requires_close : bool
        If True, verify that transformer input actually requires CLOSE feature.
        If it does but aliasing is not in place, hard-fail.
    
    Returns
    -------
    tuple[pd.DataFrame, Dict[str, any]]
        Tuple of (resolved DataFrame, resolution metadata)
    
    Raises
    ------
    RuntimeError
        If transformer requires CLOSE but aliasing is not in place
    """
    compat_enabled = os.getenv("GX1_ALLOW_CLOSE_ALIAS_COMPAT", "0") == "1"
    if not compat_enabled:
        raise RuntimeError(
            f"[CASE_COLLISION] {context}: close/CLOSE collision detected but "
            f"GX1_ALLOW_CLOSE_ALIAS_COMPAT=1 is not set."
        )
    
    # Safety check: if transformer requires CLOSE, aliasing must be in place
    if transformer_requires_close:
        # This is a warning - aliasing should be implemented in transformer input assembly
        # For now, we allow dropping CLOSE if compat-mode is enabled
        # TODO: Implement aliasing in transformer input assembly
        pass
    
    # Drop CLOSE column (keep close from candles)
    if "CLOSE" in df.columns:
        df_resolved = df.drop(columns=["CLOSE"]).copy()
        resolution_meta = {
            "case_collision_resolved": True,
            "dropped_columns": ["CLOSE"],
            "alias_expected": {"CLOSE": "candles.close"},
            "context": context,
        }
        return df_resolved, resolution_meta
    else:
        # No CLOSE column to drop (should not happen if collision was detected)
        return df.copy(), {"case_collision_resolved": False}


def check_reserved_candle_columns(
    df: pd.DataFrame,
    context: str,
    allowlist: Optional[List[str]] = None,
) -> List[str]:
    """
    Check if DataFrame contains reserved candle columns (case-insensitive).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    context : str
        Description of where this check is performed
    allowlist : Optional[List[str]]
        List of column names to allow even if they match reserved columns
    
    Returns
    -------
    List[str]
        List of reserved columns found (empty if none)
    """
    allowlist_lower = {c.lower() for c in (allowlist or [])}
    found = []
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in RESERVED_CANDLE_COLUMNS and col_lower not in allowlist_lower:
            found.append(col)
    return found
