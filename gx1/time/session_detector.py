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
M1_BAR_DURATION = pd.Timedelta(minutes=1)
M5_BAR_DURATION = pd.Timedelta(minutes=5)

# ── The four session-overlap flags (V30 package 3, 2026-08-13) ──────────────
#
# ONE session clock.  These four flags used to be produced by a SECOND,
# overlapping hour-set definition inside gx1/scripts/augment_forward_outcome_v2
# (ASIA_HOURS {22..8}, EU_HOURS {7..16}, US_HOURS {13..21}) that disagreed with
# the SESSION_BOUNDARIES partition above at h=8, 16 and 22-23.  That definition
# is retired; the flags are derived here, from the partition, by the module
# that owns the boundaries.
#
# The partition already carries an EU/US overlap state, so three of the four
# map onto it exactly:
#     is_eu_us_overlap := session == OVERLAP   (12:00-16:00 UTC)
#     is_eu_only       := session == EU, after the handover window below
#     is_us_only       := session == US        (16:00-22:00 UTC)
#
# The partition has NO asia/eu overlap state — ASIA ends at 07:00 exactly where
# EU begins — so `is_asia_eu_overlap := session == <asia/eu state>` would be
# provably constant 0, which is a designed-in liveness failure and forbidden.
# The retired flag's market content is the Tokyo/London handover, and the
# retired definition located it precisely: ASIA_HOURS n EU_HOURS = {7, 8} =
# 07:00-09:00 UTC = the first 120 minutes of the surviving EU session.  The
# flag therefore becomes a boundary-adjacent window on the one partition,
# measured with `minutes_since_session_open`, which is already a produced
# field:
#     is_asia_eu_overlap := session == EU and minutes_since_open < 120
#
# 120 is not a new magnitude (rule 2b): it is the width of the retired
# ASIA_HOURS n EU_HOURS intersection, re-expressed as minutes from the
# surviving EU open.  The four flags stay mutually exclusive and all-zero on
# ASIA, exactly as the retired hour-set construction was.
#
# Liveness, proven from the partition (no data needed): the flags fire on
# 120/1440 = 8.33%, 180/1440 = 12.50%, 240/1440 = 16.67% and 360/1440 = 25.00%
# of minutes-of-day respectively — all far above the 1% activity floor, and
# within a percentage point or two of the retired definition's own rates
# (8.33% / 16.67% / 16.67% / 20.83%).
ASIA_EU_HANDOVER_MINUTES = 120

SESSION_OVERLAP_FLAG_NAMES = (
    "is_asia_eu_overlap",
    "is_eu_us_overlap",
    "is_eu_only",
    "is_us_only",
)


def session_overlap_flags(ts: pd.Timestamp) -> dict:
    """Return the four session-overlap flags for one UTC timestamp.

    ``ts`` must already be the timestamp at which the classification is taken
    (for a bar, its close/availability time — the caller owns that shift, the
    same convention ``decision_availability`` states).
    """
    if not isinstance(ts, pd.Timestamp) or ts.tzinfo is None:
        raise RuntimeError(f"SESSION_OVERLAP_TIMESTAMP_INVALID: {ts!r}")
    ts_utc = ts.tz_convert("UTC")
    minute_of_day = ts_utc.hour * 60 + ts_utc.minute
    session = _session_for_minute(minute_of_day)
    start_hour = SESSION_BOUNDARIES[session][0]
    minutes_since_open = (minute_of_day - start_hour * 60) % (24 * 60)
    is_eu = session == "EU"
    handover = is_eu and minutes_since_open < ASIA_EU_HANDOVER_MINUTES
    return {
        "is_asia_eu_overlap": float(handover),
        "is_eu_us_overlap": float(session == "OVERLAP"),
        "is_eu_only": float(is_eu and not handover),
        "is_us_only": float(session == "US"),
    }


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


def decision_availability(
    bar_start_timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
    *,
    bar_duration: pd.Timedelta,
    context: str,
) -> pd.DatetimeIndex:
    """Return exact UTC availability times for one closed-bar clock."""
    if not isinstance(bar_duration, pd.Timedelta) or bar_duration <= pd.Timedelta(0):
        raise RuntimeError(f"{context}_BAR_DURATION_INVALID")
    try:
        labels = pd.DatetimeIndex(
            pd.to_datetime(bar_start_timestamps, utc=True, errors="raise")
        )
    except Exception as exc:
        raise RuntimeError(f"{context}_BAR_LABEL_INVALID") from exc
    if (
        labels.hasnans
        or labels.has_duplicates
        or not labels.is_monotonic_increasing
    ):
        raise RuntimeError(f"{context}_BAR_LABEL_ORDER_INVALID")
    grid_ns = int(bar_duration.value)
    if len(labels) and np.any(labels.asi8 % grid_ns != 0):
        raise RuntimeError(f"{context}_BAR_LABEL_OFF_GRID")
    return labels.tz_convert("UTC") + bar_duration


def m5_decision_availability(
    bar_start_timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
) -> pd.DatetimeIndex:
    """Return exact UTC availability times for M5 bar-start labels."""

    return decision_availability(
        bar_start_timestamps,
        bar_duration=M5_BAR_DURATION,
        context="M5_DECISION",
    )


def m1_decision_availability(
    bar_start_timestamps: Union[pd.Series, pd.DatetimeIndex, np.ndarray],
) -> pd.DatetimeIndex:
    """Return exact UTC availability times for M1 bar-start labels."""

    return decision_availability(
        bar_start_timestamps,
        bar_duration=M1_BAR_DURATION,
        context="M1_DECISION",
    )


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
