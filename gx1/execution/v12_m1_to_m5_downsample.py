#!/usr/bin/env python3
"""V12 M1 → M5 aggregation primitive.

``m1_to_m5`` remains the shared pure aggregation primitive for immutable
derivative snapshots. The former CLI writer into the canonical M5 root is
retired: canonical M5 is source-native and cannot have an M1-aggregated second
owner.
"""
from __future__ import annotations

import pandas as pd


M1_BAR_DURATION = pd.Timedelta(minutes=1)
M5_BAR_DURATION = pd.Timedelta(minutes=5)


def latest_closed_m5_start_at(
    availability_time: pd.Timestamp,
) -> pd.Timestamp:
    """Return the latest M5 bar start closed at ``availability_time``.

    Bar timestamps are start labels.  At exactly 12:05, the M5 bar labelled
    12:00 is closed and available, so the answer is 12:00.
    """

    timestamp = pd.Timestamp(availability_time)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    return timestamp.floor("5min") - M5_BAR_DURATION


def closed_m5_start_for_m1_bar_labels(
    m1_bar_labels: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Map closed M1 start labels to their newest available closed M5 bar.

    An M1 bar labelled ``T`` is not available until ``T + 1 minute``.  The
    phase-4 bar (for example 12:04) therefore sees the M5 bar labelled 12:00,
    while phases 0..3 still see 11:55.  This is the shared train/serve mapping;
    callers must not replace it with ``floor(T, 5m) - 5m``.
    """

    labels = pd.to_datetime(m1_bar_labels, utc=True, errors="coerce")
    labels = pd.DatetimeIndex(labels)
    if labels.hasnans:
        raise ValueError("M1_BAR_LABELS_INVALID")
    return pd.DatetimeIndex(
        (labels + M1_BAR_DURATION).floor("5min") - M5_BAR_DURATION
    )


def m1_to_m5(m1: pd.DataFrame, tape_end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate M1 OHLC into 5-min buckets; emit every PROVABLY COMPLETE bucket.

    The tape holds only COMPLETE candles (oanda_client.py:254 exclude_incomplete
    default), so an M1 bar stamped T finalizes [T, T+1min) and the tape is final
    through tape_end+1min. Bucket [B, B+5min) is provably complete iff
    B+5min <= tape_end+1min; buckets failing the test sit at the live edge (bars
    may still arrive) and are SUPPRESSED — the next run's overlap re-read +
    keep='last' merge finalizes them. Buckets that pass with <5 bars are
    tick-empty minutes (e.g. the 22:00Z reopen after the 21-22Z break) and MUST
    be emitted: OANDA-native M5 history keeps any bucket with >=1 trade, and the
    old counts>=5 filter forked the tape's bar-density convention by permanently
    dropping them.

    tape_end=None infers the edge from the input slice's last bar — correct only
    when the slice ends at the GLOBAL tape tail (v12_canonical_incremental's
    collector window; the final-year slice here). Pass the global tape end for
    earlier-year slices, else their year-end terminal bucket is false-suppressed
    and never healed.
    """
    m1 = m1.copy()
    m1["time"] = pd.to_datetime(m1["time"], utc=True)
    m1["m5_bucket"] = m1["time"].dt.floor("5min")
    agg_funcs = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last",
        "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last",
        "volume": "sum",
    }
    agg_cols = {c: agg_funcs[c] for c in agg_funcs if c in m1.columns}
    m5 = m1.groupby("m5_bucket").agg(agg_cols).reset_index()
    m5 = m5.rename(columns={"m5_bucket": "time"})
    if tape_end is None:
        tape_end = m1["time"].max()
    complete = m5["time"] + pd.Timedelta(minutes=5) <= tape_end + pd.Timedelta(minutes=1)
    m5 = m5[complete].reset_index(drop=True)
    return m5


def main() -> int:
    raise RuntimeError(
        "CANONICAL_M5_SINGLE_OWNER_VIOLATION: the M1->M5 canonical-root writer "
        "is retired; use m1_to_m5 only inside a hash-bound derivative snapshot"
    )


if __name__ == "__main__":
    raise SystemExit(main())
