#!/usr/bin/env python3
"""V12 M1 → M5 downsampler — extend M5 tape to match M1 tape's coverage.

Reads canonical M1 tape, aggregates to M5 OHLC (open=first, high=max, low=min,
close=last, volume=sum) per 5-minute bucket aligned to UTC. Appends to
xauusd_m5_bid_ask__CANONICAL/year={Y}/part-000.parquet.

Required after v12_backfill_to_present.py extends M1 tape — canonical_v2/v3
builders read M5 tape, so M5 must be brought up to date before re-running
feature builders.

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_m1_to_m5_downsample.py

Env:
    GX1_M5_REBUILD_FROM=YYYYMMDD — force the M1 re-read to start at that UTC date
    instead of latest-M5 minus 1h, so interior buckets previously dropped (old
    counts>=5 filter) backfill through the normal keep='last' merge path.
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG = logging.getLogger("v12_m5_downsample")
M1_ROOT = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
M5_ROOT = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL")

OHLC_COLS = ["open", "high", "low", "close",
             "bid_open", "bid_high", "bid_low", "bid_close",
             "ask_open", "ask_high", "ask_low", "ask_close"]


def _atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomic write: tmp + os.replace (same FS, no torn read for concurrent readers).

    Same pattern as v12_canonical_incremental._atomic_write_parquet — NOT imported
    from there because that module imports m1_to_m5 from THIS one (circular).
    A partial part-000.parquet.tmp from a crashed run is harmless: no consumer
    matches it (readers glob year=*/part-000.parquet) and the next run's
    to_parquet to the same deterministic tmp path overwrites it.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")
    M5_ROOT.mkdir(parents=True, exist_ok=True)

    # Find each year-partition in M1, downsample, write/merge to M5
    m1_files = sorted(M1_ROOT.glob("year=*/part-000.parquet"))
    if not m1_files:
        LOG.error(f"no M1 partitions under {M1_ROOT}")
        return 1

    # Get latest existing M5 timestamp to skip already-built data
    m5_existing_files = sorted(M5_ROOT.glob("year=*/part-000.parquet"))
    latest_m5 = pd.Timestamp.min.tz_localize("UTC")
    if m5_existing_files:
        for f in m5_existing_files:
            df = pd.read_parquet(f, columns=["time"])
            t = pd.to_datetime(df["time"].max(), utc=True)
            if t > latest_m5:
                latest_m5 = t
        LOG.info(f"M5 tape latest: {latest_m5.isoformat()}")
    else:
        LOG.info("no M5 tape exists — full build")

    # GLOBAL M1 tape end for m1_to_m5's provably-complete test (files sorted by
    # year => last file holds the tail). Per-year-slice inference would
    # false-suppress an earlier year's terminal bucket.
    tape_end = pd.to_datetime(
        pd.read_parquet(m1_files[-1], columns=["time"])["time"].max(), utc=True)

    # Re-read cutoff: normally latest-M5 minus 1h (re-aggregates + finalizes the
    # previously suppressed live edge via the keep='last' merge).
    # GX1_M5_REBUILD_FROM=YYYYMMDD forces a deeper re-read so interior buckets
    # dropped by the old counts>=5 filter backfill through the same merge path.
    rebuild_from = os.environ.get("GX1_M5_REBUILD_FROM", "").strip()
    if rebuild_from:
        # strict parse — a malformed override must fail loud, never silently no-op
        cutoff = pd.Timestamp(datetime.strptime(rebuild_from, "%Y%m%d"), tz="UTC")
        LOG.info(f"GX1_M5_REBUILD_FROM={rebuild_from} → re-reading M1 from {cutoff.isoformat()}")
    elif m5_existing_files:
        cutoff = (latest_m5 - pd.Timedelta(hours=1)).floor("5min")
    else:
        cutoff = None  # virgin build — aggregate everything

    # Process each year (cheap idempotent merge)
    for m1_file in m1_files:
        year = m1_file.parent.name.split("=")[1]
        m1 = pd.read_parquet(m1_file)
        m1["time"] = pd.to_datetime(m1["time"], utc=True)
        # Slice start MUST sit ON a 5-min bucket boundary and INCLUDE the boundary
        # bar (>=): a mid-bucket slice would re-aggregate a partial bucket whose
        # 'open' comes from a later minute, and keep='last' would overwrite a
        # correct row. (cutoff is 5-min aligned: floored bucket-start − 1h, or a
        # YYYYMMDD midnight.) Overlap is OK — dedupe handles it.
        m1_recent = m1 if cutoff is None else m1[m1["time"] >= cutoff]
        if len(m1_recent) == 0:
            continue
        m5 = m1_to_m5(m1_recent, tape_end=tape_end)
        if len(m5) == 0:
            continue
        out_dir = M5_ROOT / f"year={year}"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "part-000.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            existing["time"] = pd.to_datetime(existing["time"], utc=True)
            combined = (pd.concat([existing, m5], ignore_index=True)
                        .drop_duplicates(subset=["time"], keep="last")
                        .sort_values("time")
                        .reset_index(drop=True))
        else:
            combined = m5.sort_values("time").reset_index(drop=True)
        # mid-write crash must not corrupt the one-truth year tape (hourly
        # cf-daemon reads the same file — torn-read race)
        _atomic_to_parquet(combined, out_path)
        LOG.info(f"year={year}: {len(combined):,} M5 rows  range={combined['time'].min()} → {combined['time'].max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
