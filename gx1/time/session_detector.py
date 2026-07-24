"""
Trading Session Detector - SSoT for session classification.

Defines trading sessions based on UTC time:
- ASIA: 22:00-07:00 UTC
- EU: 07:00-12:00 UTC
- OVERLAP: 12:00-16:00 UTC
- US: 16:00-22:00 UTC

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


# Session boundaries in UTC hours (SSoT)
# ASIA: 22:00-07:00 UTC (spans midnight)
# EU: 07:00-12:00 UTC
# OVERLAP: 12:00-16:00 UTC
# US: 16:00-22:00 UTC
SESSION_BOUNDARIES = {
    "ASIA": (22, 7),      # 22:00-07:00 UTC (wrap)
    "EU": (7, 12),        # 07:00-12:00 UTC
    "OVERLAP": (12, 16),  # 12:00-16:00 UTC
    "US": (16, 22),       # 16:00-22:00 UTC
}

# Canonical session_id mapping (observerable context)
SESSION_ID_MAP = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
SESSION_ID_INV = {v: k for k, v in SESSION_ID_MAP.items()}
M5_BAR_DURATION = pd.Timedelta(minutes=5)


def m5_decision_availability(
    bar_start_timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
) -> pd.DatetimeIndex:
    """Return exact UTC availability times for M5 bar-start labels.

    This is the shared clock contract for canonical M5 evidence.  A row
    labelled ``T`` contains the completed bar ``[T,T+5min)`` and may first be
    used at ``T+5min``.  Market gaps are allowed; malformed, unordered, or
    off-grid labels are not.
    """

    try:
        labels = pd.DatetimeIndex(
            pd.to_datetime(bar_start_timestamps, utc=True, errors="raise")
        )
    except Exception as exc:
        raise RuntimeError("M5_DECISION_BAR_LABEL_INVALID") from exc
    if (
        labels.hasnans
        or labels.has_duplicates
        or not labels.is_monotonic_increasing
    ):
        raise RuntimeError("M5_DECISION_BAR_LABEL_ORDER_INVALID")
    grid_ns = int(M5_BAR_DURATION.value)
    if len(labels) and np.any(labels.asi8 % grid_ns != 0):
        raise RuntimeError("M5_DECISION_BAR_LABEL_OFF_GRID")
    return labels.tz_convert("UTC") + M5_BAR_DURATION


def _as_datetime_series(timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]) -> pd.Series:
    """Normalize timestamp-like input while preserving DatetimeIndex alignment."""
    if isinstance(timestamps, pd.Series):
        out = timestamps
    elif isinstance(timestamps, pd.DatetimeIndex):
        out = pd.Series(timestamps, index=timestamps)
    elif isinstance(timestamps, np.ndarray):
        out = pd.Series(timestamps)
    else:
        out = pd.Series(timestamps)
    if not pd.api.types.is_datetime64_any_dtype(out):
        converted = pd.to_datetime(out)
        out = converted if isinstance(converted, pd.Series) else pd.Series(converted)
    return out


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
    
    if 7 <= hour < 12:
        return "EU"
    elif 12 <= hour < 16:
        return "OVERLAP"
    elif 16 <= hour < 22:
        return "US"
    else:  # 22-07
        return "ASIA"


def get_session_vectorized(timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]) -> pd.Series:
    """
    Get trading sessions for a series of timestamps (vectorized).
    
    Args:
        timestamps: Series/array of timestamps
    
    Returns:
        Series of session labels
    """
    ts_series = _as_datetime_series(timestamps)
    hours = ts_series.dt.hour
    
    # Vectorized session assignment
    sessions = pd.Series(index=hours.index, dtype=str)
    sessions[(hours >= 7) & (hours < 12)] = "EU"
    sessions[(hours >= 12) & (hours < 16)] = "OVERLAP"
    sessions[(hours >= 16) & (hours < 22)] = "US"
    sessions[(hours >= 22) | (hours < 7)] = "ASIA"
    
    return sessions


def get_session_id(ts: pd.Timestamp) -> int:
    """
    Get canonical session_id for a single timestamp.
    Mapping: ASIA=0, EU=1, OVERLAP=2, US=3.
    """
    session = get_session(ts)
    return SESSION_ID_MAP.get(session, 0)


def get_session_id_vectorized(timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]) -> pd.Series:
    """
    Vectorized session_id for timestamps.
    Mapping: ASIA=0, EU=1, OVERLAP=2, US=3.
    """
    sessions = get_session_vectorized(timestamps)
    return sessions.map(SESSION_ID_MAP).fillna(0).astype("int32")


def get_session_minutes_since_open_vectorized(
    timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]
) -> pd.Series:
    """
    Minutes since session open (UTC).
    ASIA open = 22:00, EU open = 07:00, OVERLAP open = 12:00, US open = 16:00.
    """
    ts_series = _as_datetime_series(timestamps)
    hours = ts_series.dt.hour
    minutes = ts_series.dt.minute
    minute_of_day = hours * 60 + minutes

    # ASIA: 22:00-07:00 (wrap)
    asia = (minute_of_day >= 22 * 60) | (minute_of_day < 7 * 60)
    # If after 22:00 -> since open = minute_of_day - 1320
    # If before 07:00 -> since open = minute_of_day + 120
    since_open = pd.Series(0.0, index=ts_series.index, dtype=float)
    since_open[asia & (minute_of_day >= 22 * 60)] = (minute_of_day[asia & (minute_of_day >= 22 * 60)] - 22 * 60).astype(float)
    since_open[asia & (minute_of_day < 7 * 60)] = (minute_of_day[asia & (minute_of_day < 7 * 60)] + 120).astype(float)

    # EU: 07:00-12:00
    eu = (minute_of_day >= 7 * 60) & (minute_of_day < 12 * 60)
    since_open[eu] = (minute_of_day[eu] - 7 * 60).astype(float)

    # OVERLAP: 12:00-16:00
    overlap = (minute_of_day >= 12 * 60) & (minute_of_day < 16 * 60)
    since_open[overlap] = (minute_of_day[overlap] - 12 * 60).astype(float)

    # US: 16:00-22:00
    us = (minute_of_day >= 16 * 60) & (minute_of_day < 22 * 60)
    since_open[us] = (minute_of_day[us] - 16 * 60).astype(float)

    return since_open


def get_session_minutes_to_next_boundary_vectorized(
    timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]
) -> pd.Series:
    """
    Minutes to next session boundary (UTC).
    Boundaries at 07:00, 12:00, 16:00, 22:00.
    """
    ts_series = _as_datetime_series(timestamps)
    hours = ts_series.dt.hour
    minutes = ts_series.dt.minute
    minute_of_day = hours * 60 + minutes

    to_next = pd.Series(0.0, index=ts_series.index, dtype=float)

    # ASIA: 22:00-07:00
    asia = (minute_of_day >= 22 * 60) | (minute_of_day < 7 * 60)
    # If after 22:00 -> next boundary at 07:00 next day
    to_next[asia & (minute_of_day >= 22 * 60)] = (
        (24 * 60 - minute_of_day[asia & (minute_of_day >= 22 * 60)]) + 7 * 60
    ).astype(float)
    # If before 07:00 -> next boundary at 07:00 same day
    to_next[asia & (minute_of_day < 7 * 60)] = (7 * 60 - minute_of_day[asia & (minute_of_day < 7 * 60)]).astype(float)

    # EU: 07:00-12:00
    eu = (minute_of_day >= 7 * 60) & (minute_of_day < 12 * 60)
    to_next[eu] = (12 * 60 - minute_of_day[eu]).astype(float)

    # OVERLAP: 12:00-16:00
    overlap = (minute_of_day >= 12 * 60) & (minute_of_day < 16 * 60)
    to_next[overlap] = (16 * 60 - minute_of_day[overlap]).astype(float)

    # US: 16:00-22:00
    us = (minute_of_day >= 16 * 60) & (minute_of_day < 22 * 60)
    to_next[us] = (22 * 60 - minute_of_day[us]).astype(float)

    return to_next


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
