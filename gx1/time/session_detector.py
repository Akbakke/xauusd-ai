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
SESSION_ORDER = tuple(SESSION_BOUNDARIES)
SESSION_ID_MAP = {
    name: index for index, name in enumerate(SESSION_ORDER)
}
SESSION_ID_INV = {v: k for k, v in SESSION_ID_MAP.items()}
SESSION_NAME_BY_ID = SESSION_ID_INV
ASIA_SESSION_ID = SESSION_ID_MAP["ASIA"]
M5_BAR_DURATION = pd.Timedelta(minutes=5)


def _session_mask(
    minute_of_day: pd.Series,
    *,
    start_hour: int,
    end_hour: int,
) -> pd.Series:
    start = start_hour * 60
    end = end_hour * 60
    if start < end:
        return (minute_of_day >= start) & (minute_of_day < end)
    return (minute_of_day >= start) | (minute_of_day < end)


def _session_for_minute(minute_of_day: int) -> str:
    matches = [
        name
        for name, (start, end) in SESSION_BOUNDARIES.items()
        if (
            (start < end and start * 60 <= minute_of_day < end * 60)
            or (
                start >= end
                and (
                    minute_of_day >= start * 60
                    or minute_of_day < end * 60
                )
            )
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "SESSION_BOUNDARY_CONTRACT_INVALID: "
            f"minute={minute_of_day} matches={matches}"
        )
    return matches[0]


for _minute in range(24 * 60):
    _session_for_minute(_minute)


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
    converted = pd.to_datetime(out, utc=True, errors="raise")
    return (
        converted
        if isinstance(converted, pd.Series)
        else pd.Series(converted, index=out.index)
    )


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
    return _session_for_minute(ts_utc.hour * 60 + ts_utc.minute)


def get_session_vectorized(timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]) -> pd.Series:
    """
    Get trading sessions for a series of timestamps (vectorized).
    
    Args:
        timestamps: Series/array of timestamps
    
    Returns:
        Series of session labels
    """
    ts_series = _as_datetime_series(timestamps)
    minute_of_day = ts_series.dt.hour * 60 + ts_series.dt.minute
    sessions = pd.Series(index=minute_of_day.index, dtype="object")
    for name, (start, end) in SESSION_BOUNDARIES.items():
        sessions[
            _session_mask(
                minute_of_day,
                start_hour=start,
                end_hour=end,
            )
        ] = name
    if sessions.isna().any():
        raise RuntimeError("SESSION_BOUNDARY_CONTRACT_UNCOVERED")
    return sessions


def get_session_id(ts: pd.Timestamp) -> int:
    """
    Get canonical session_id for a single timestamp.
    Mapping: ASIA=0, EU=1, OVERLAP=2, US=3.

    An unknown session fails closed. Defaulting to 0 would silently label it
    ASIA, which the Entry context contract forbids outright: a fabricated ASIA
    flag must yield no direction rather than a plausible session.
    """
    session = get_session(ts)
    if session not in SESSION_ID_MAP:
        raise RuntimeError(f"SESSION_ID_UNKNOWN: session={session!r} ts={ts!r}")
    return SESSION_ID_MAP[session]


def get_session_id_vectorized(timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray]) -> pd.Series:
    """
    Vectorized session_id for timestamps.
    Mapping: ASIA=0, EU=1, OVERLAP=2, US=3.

    Unmapped sessions fail closed for the same reason as the scalar helper; the
    former ``fillna(0)`` turned every unknown session into ASIA for the three
    consumers that read this directly (canonical build, the Entry context adder
    and the live context augmenter).
    """
    sessions = get_session_vectorized(timestamps)
    ids = sessions.map(SESSION_ID_MAP)
    if bool(ids.isna().any()):
        unknown = sorted({str(v) for v in sessions[ids.isna()].unique()})
        raise RuntimeError(
            f"SESSION_ID_UNKNOWN: rows={int(ids.isna().sum())} sessions={unknown}"
        )
    return ids.astype("int32")


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

    since_open = pd.Series(0.0, index=ts_series.index, dtype=float)
    for _name, (start, end) in SESSION_BOUNDARIES.items():
        mask = _session_mask(
            minute_of_day,
            start_hour=start,
            end_hour=end,
        )
        since_open[mask] = (
            (minute_of_day[mask] - start * 60) % (24 * 60)
        ).astype(float)

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

    for _name, (start, end) in SESSION_BOUNDARIES.items():
        mask = _session_mask(
            minute_of_day,
            start_hour=start,
            end_hour=end,
        )
        to_next[mask] = (
            (end * 60 - minute_of_day[mask]) % (24 * 60)
        ).astype(float)

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
