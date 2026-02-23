"""
Timestamp column utilities.

Canonical function for ensuring "ts" column exists in DataFrames.
"""

import pandas as pd
from typing import Optional


def ensure_ts_column(df: pd.DataFrame, context: str = "unknown") -> pd.DataFrame:
    """
    Ensure DataFrame has "ts" column.
    
    Rules:
    - If "ts" already exists: return df as-is (and optionally validate type)
    - If df.index is DatetimeIndex: extract to "ts" column
    - If "timestamp" exists: rename to "ts"
    - Otherwise: FATAL
    
    Args:
        df: DataFrame to ensure "ts" column for
        context: Context string for error messages
    
    Returns:
        DataFrame with "ts" column guaranteed
    
    Raises:
        RuntimeError: If "ts" column cannot be created
    """
    # Check if "ts" already exists
    if "ts" in df.columns:
        # Validate: ensure it's not duplicated
        ts_count = df.columns.tolist().count("ts")
        if ts_count > 1:
            raise RuntimeError(
                f"TS_COLLISION: DataFrame has {ts_count} 'ts' columns (duplicate). "
                f"Context: {context}. Columns: {list(df.columns)[:30]}..."
            )
        # "ts" exists and is unique - return as-is
        return df
    
    # "ts" doesn't exist - create it
    df = df.copy()
    
    # Try 1: Extract from DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df.insert(0, "ts", df.index)
        return df
    
    # Try 2: Rename common timestamp column names to "ts"
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "ts"})
        return df

    # Try 2b: Some GX1 parquet files use "time" as the timestamp column (raw candles, prebuilt).
    # Promote it to the canonical "ts" column.
    if "time" in df.columns:
        df = df.rename(columns={"time": "ts"})
        return df
    
    # Try 3: Check if index has a name that suggests timestamp
    if df.index.name in ["ts", "timestamp", "time", "datetime"]:
        df.insert(0, "ts", df.index)
        return df
    
    # FATAL: Cannot create "ts" column
    raise RuntimeError(
        f"TS_COLUMN_MISSING: Cannot create 'ts' column. "
        f"Context: {context}. "
        f"Index type: {type(df.index)}, index name: {df.index.name}. "
        f"Columns: {list(df.columns)[:30]}..."
    )


def validate_ts_column(df: pd.DataFrame, context: str = "unknown") -> None:
    """
    Validate "ts" column exists and is well-formed.
    
    Args:
        df: DataFrame to validate
        context: Context string for error messages
    
    Raises:
        RuntimeError: If validation fails
    """
    if "ts" not in df.columns:
        raise RuntimeError(
            f"TS_VALIDATION_FAIL: 'ts' column missing. Context: {context}"
        )
    
    # Check for duplicates
    ts_count = df.columns.tolist().count("ts")
    if ts_count != 1:
        raise RuntimeError(
            f"TS_VALIDATION_FAIL: 'ts' column appears {ts_count} times (expected 1). "
            f"Context: {context}"
        )
    
    # Check for duplicates in values (optional - may be valid in some cases)
    # This is just a warning, not a hard fail
    if df["ts"].duplicated().any():
        dup_count = df["ts"].duplicated().sum()
        import warnings
        warnings.warn(
            f"TS_VALIDATION_WARNING: 'ts' column has {dup_count} duplicate values. "
            f"Context: {context}"
        )
