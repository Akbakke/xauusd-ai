#!/usr/bin/env python3
"""V12 backfill — extend M1 tape from last available timestamp to present (UTC).

Reads canonical M1 tape under xauusd_m1_bid_ask__CANONICAL/year=*, identifies
the latest timestamp, fetches OANDA M1 candles forward to now, and appends/
deduplicates back into the year partition.

Required for V12 live trading: V10/V3 transformer windows (96/512 M1 bars)
plus D1 features (200+ days EMA, 252-day ATR percentile) all need contiguous
M1 history up to the candidate moment.

Companion: after backfill completes, re-run canonical_v3 builder to extend
features through to present. (Done in a separate step.)

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_backfill_to_present.py
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Auto-load .env
ENV_FILE = REPO_ROOT / ".env"
if ENV_FILE.is_file():
    with ENV_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials

LOG = logging.getLogger("v12_backfill")
INSTRUMENT = "XAU_USD"
M1_TAPE_ROOT = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")


def find_latest_m1_timestamp() -> tuple[pd.Timestamp, Path]:
    """Scan year=* partitions for newest timestamp."""
    files = sorted(M1_TAPE_ROOT.glob("year=*/part-000.parquet"))
    if not files:
        raise FileNotFoundError(f"No M1 partitions found under {M1_TAPE_ROOT}")
    last_file = files[-1]
    df = pd.read_parquet(last_file, columns=["time"])
    latest = pd.to_datetime(df["time"].max(), utc=True)
    return latest, last_file


def backfill_forward(client: OandaClient, from_ts: pd.Timestamp, to_ts: pd.Timestamp) -> pd.DataFrame:
    """Fetch M1 candles in 1-day windows via from_ts/to_ts API."""
    chunks = []
    cursor = from_ts
    LOG.info(f"backfill {from_ts.isoformat()} → {to_ts.isoformat()}")
    iteration = 0
    while cursor < to_ts:
        iteration += 1
        window_end = min(cursor + pd.Timedelta(days=1), to_ts)
        try:
            df = client.get_candles(
                instrument=INSTRUMENT, granularity="M1",
                from_ts=cursor, to_ts=window_end,
            )
        except Exception as exc:
            LOG.error(f"chunk {iteration} failed [{cursor} → {window_end}]: {exc}")
            cursor = window_end  # skip and continue
            continue
        if df.index.name == "time":
            df = df.reset_index()
        if len(df) > 0:
            chunks.append(df)
            LOG.info(f"  chunk {iteration}: {len(df):,} bars  range {df['time'].min()} → {df['time'].max()}")
        else:
            LOG.info(f"  chunk {iteration}: 0 bars [{cursor} → {window_end}] (likely weekend/closed)")
        cursor = window_end
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")


def append_to_year_partition(df: pd.DataFrame) -> dict:
    """Append df to year=*/part-000.parquet. Splits across years if span crosses year-boundary."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["_year"] = df["time"].dt.year
    stats = {}
    for year, sub in df.groupby("_year"):
        sub = sub.drop(columns=["_year"])
        year_dir = M1_TAPE_ROOT / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        out_path = year_dir / "part-000.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            existing["time"] = pd.to_datetime(existing["time"], utc=True)
            combined = (pd.concat([existing, sub], ignore_index=True)
                        .drop_duplicates(subset=["time"], keep="last")
                        .sort_values("time"))
        else:
            combined = sub.sort_values("time")
        combined.to_parquet(out_path, index=False)
        n_new = len(combined) - (len(existing) if out_path.exists() and 'existing' in locals() else 0)
        stats[year] = {"total_rows": len(combined), "new_added_approx": int(len(sub))}
        LOG.info(f"  year={year}: total {len(combined):,} bars, +{len(sub):,} new")
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    creds = load_oanda_credentials()
    client = OandaClient(OandaClientConfig(api_key=creds.api_token,
                                            account_id=creds.account_id,
                                            env=creds.env))

    latest, _ = find_latest_m1_timestamp()
    now_utc = pd.Timestamp.now(tz="UTC").floor("min")
    LOG.info(f"M1 tape latest:  {latest.isoformat()}")
    LOG.info(f"now (UTC):       {now_utc.isoformat()}")
    gap_minutes = int((now_utc - latest).total_seconds() // 60)
    LOG.info(f"gap:             {gap_minutes:,} minutes ({gap_minutes/1440:.1f} days)")

    if gap_minutes < 5:
        LOG.info("gap < 5 min — already up-to-date, nothing to do")
        return 0

    # Backfill from latest+1min to now
    from_ts = latest + pd.Timedelta(minutes=1)
    df_new = backfill_forward(client, from_ts, now_utc)
    if df_new.empty:
        LOG.error("backfill returned 0 rows — nothing appended")
        return 1
    LOG.info(f"fetched {len(df_new):,} new bars  range {df_new['time'].min()} → {df_new['time'].max()}")

    stats = append_to_year_partition(df_new)
    LOG.info(f"backfill complete. year stats: {stats}")
    LOG.info("NEXT: run canonical_v3 rebuild to extend features through new data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
