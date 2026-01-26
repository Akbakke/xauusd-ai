"""
Trading Session Detector - SSoT for session classification.

Defines trading sessions based on UTC time:
- ASIA: 00:00-07:00 UTC
- EU: 07:00-13:00 UTC  
- OVERLAP: 13:00-17:00 UTC (EU + US overlap)
- US: 17:00-21:00 UTC
- ASIA (late): 21:00-00:00 UTC

Usage:
    from gx1.time.session_detector import get_session, get_session_vectorized
    
    session = get_session(pd.Timestamp("2025-01-15 10:30:00", tz="UTC"))
    # Returns: "EU"
    
    sessions = get_session_vectorized(df["timestamp"])
    # Returns: pd.Series of session labels
"""

import numpy as np
import pandas as pd
from typing import Union


# Session boundaries in UTC hours
SESSION_BOUNDARIES = {
    "ASIA_EARLY": (0, 7),   # 00:00-07:00 UTC
    "EU": (7, 13),          # 07:00-13:00 UTC
    "OVERLAP": (13, 17),    # 13:00-17:00 UTC
    "US": (17, 21),         # 17:00-21:00 UTC
    "ASIA_LATE": (21, 24),  # 21:00-00:00 UTC
}


def get_session(ts: pd.Timestamp) -> str:
    """
    Get trading session for a single timestamp.
    
    Args:
        ts: Timestamp (must be UTC or timezone-aware)
    
    Returns:
        Session label: "EU", "US", "OVERLAP", or "ASIA"
    
    Raises:
        ValueError: If timestamp is not UTC/timezone-aware
    """
    if ts.tzinfo is None:
        # Assume UTC if no timezone
        ts = ts.tz_localize("UTC")
    
    # Convert to UTC if needed
    ts_utc = ts.tz_convert("UTC")
    hour = ts_utc.hour
    
    if 0 <= hour < 7:
        return "ASIA"
    elif 7 <= hour < 13:
        return "EU"
    elif 13 <= hour < 17:
        return "OVERLAP"
    elif 17 <= hour < 21:
        return "US"
    else:  # 21-24
        return "ASIA"


def get_session_vectorized(timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]) -> pd.Series:
    """
    Get trading sessions for a series of timestamps (vectorized).
    
    Args:
        timestamps: Series/array of timestamps
    
    Returns:
        Series of session labels
    """
    if isinstance(timestamps, np.ndarray):
        timestamps = pd.Series(timestamps)
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)
    
    # Extract hour (assumes UTC or uses local hour if naive)
    if hasattr(timestamps, 'dt'):
        hours = timestamps.dt.hour
    else:
        hours = pd.Series(timestamps).dt.hour
    
    # Vectorized session assignment
    sessions = pd.Series(index=hours.index, dtype=str)
    sessions[(hours >= 0) & (hours < 7)] = "ASIA"
    sessions[(hours >= 7) & (hours < 13)] = "EU"
    sessions[(hours >= 13) & (hours < 17)] = "OVERLAP"
    sessions[(hours >= 17) & (hours < 21)] = "US"
    sessions[(hours >= 21) & (hours < 24)] = "ASIA"
    
    return sessions


def validate_timestamps_monotonic(timestamps: pd.Series) -> bool:
    """Check that timestamps are monotonically increasing."""
    if len(timestamps) < 2:
        return True
    return (timestamps.diff().dropna() >= pd.Timedelta(0)).all()


def get_session_stats(sessions: pd.Series) -> dict:
    """Get statistics about session distribution."""
    counts = sessions.value_counts()
    total = len(sessions)
    
    return {
        "counts": counts.to_dict(),
        "percentages": {k: v / total * 100 for k, v in counts.items()},
        "total": total,
    }
