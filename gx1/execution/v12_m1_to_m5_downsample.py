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
"""
from __future__ import annotations

import logging
import sys
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


def m1_to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M1 OHLC into 5-min buckets."""
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
    # Drop incomplete final bucket (less than 5 M1 bars)
    counts = m1.groupby("m5_bucket").size()
    full_buckets = counts[counts >= 5].index
    m5 = m5[m5["time"].isin(full_buckets)].reset_index(drop=True)
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

    # Process each year (cheap idempotent merge)
    for m1_file in m1_files:
        year = m1_file.parent.name.split("=")[1]
        m1 = pd.read_parquet(m1_file)
        m1["time"] = pd.to_datetime(m1["time"], utc=True)
        # Filter to bars after latest_m5 (slight overlap is OK — dedupe handles it)
        m1_recent = m1[m1["time"] > latest_m5 - pd.Timedelta(hours=1)]
        if len(m1_recent) == 0:
            continue
        m5 = m1_to_m5(m1_recent)
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
        combined.to_parquet(out_path, index=False)
        LOG.info(f"year={year}: {len(combined):,} M5 rows  range={combined['time'].min()} → {combined['time'].max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
