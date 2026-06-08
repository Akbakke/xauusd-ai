#!/usr/bin/env python3
"""V12 OANDA practice data collector.

Polls XAU_USD M1 candles every 60s via OANDA practice API and appends to
daily parquet files. Designed to run continuously for 1-2+ weeks to build
a real live-data corpus for V12 shadow-replay.

Output:
    /home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_YYYYMMDD.parquet

Each file: time, open/high/low/close (mid), bid_open/.../close, ask_open/.../close, volume.
Companion script v12_shadow_replay.py runs V12 stack on collected data.

Run:
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_oanda_data_collector.py

Reads OANDA credentials from .env in repo root or env vars.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Auto-load .env if present
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

LOG = logging.getLogger("v12_collector")
INSTRUMENT = "XAU_USD"
# Poll cadence is env-overridable (no magic constant) — live SLA tuning. Default 60s
# (OANDA-friendly); set GX1_COLLECTOR_POLL_SECONDS=15 in the systemd unit to pick up a
# newly-closed M1 bar within ~15s instead of ~60s (tightens the M1 exit price reaction).
# Note: a *closed* M1 bar is inherently up to 60s old; sub-1-min reaction needs tick
# streaming (OANDA pricing/stream), not a faster poll — this only removes poll latency.
POLL_SECONDS = int(os.environ.get("GX1_COLLECTOR_POLL_SECONDS", "60"))
FETCH_LOOKBACK = 30                    # poll the last 30 M1 candles each iteration
OUT_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")


def _today_path() -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return OUT_DIR / f"xauusd_m1_{today}.parquet"


def _merge_and_dedupe(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Concat + dedupe by `time`, sorted ascending."""
    if existing is None or len(existing) == 0:
        return new_df.sort_values("time").reset_index(drop=True)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"], keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


def collect_loop(client: OandaClient) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info(f"V12 collector starting → {OUT_DIR}")
    LOG.info(f"  instrument={INSTRUMENT}  poll={POLL_SECONDS}s  fetch_lookback={FETCH_LOOKBACK}")

    consecutive_errors = 0
    while True:
        loop_start = time.time()
        try:
            df = client.get_candles(INSTRUMENT, granularity="M1", count=FETCH_LOOKBACK)
            # OANDA client returns time as index; normalize to column
            if df.index.name == "time" or "time" not in df.columns:
                df = df.reset_index()
            if len(df) == 0:
                LOG.warning("OANDA returned 0 candles — sleeping")
            else:
                out_path = _today_path()
                if out_path.exists():
                    existing = pd.read_parquet(out_path)
                    df = _merge_and_dedupe(existing, df)
                df.to_parquet(out_path, index=False)
                latest_time = pd.to_datetime(df["time"].max())
                LOG.info(f"wrote {len(df):,} bars  latest={latest_time.isoformat()}  → {out_path.name}")
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            LOG.error(f"poll failed (consecutive={consecutive_errors}): {exc}")
            # Exponential backoff up to 5 min
            backoff = min(POLL_SECONDS * (2 ** min(consecutive_errors, 6)), 300)
            time.sleep(backoff)
            continue

        elapsed = time.time() - loop_start
        sleep_for = max(0, POLL_SECONDS - elapsed)
        time.sleep(sleep_for)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    creds = load_oanda_credentials()
    LOG.info(f"OANDA env={creds.env}  account={creds.account_id}  api_url={creds.api_url}")

    cfg = OandaClientConfig(
        api_key=creds.api_token,
        account_id=creds.account_id,
        env=creds.env,
    )
    client = OandaClient(cfg)

    # Sanity-test: pull 5 candles to verify connectivity
    try:
        df_test = client.get_candles(INSTRUMENT, granularity="M1", count=5)
        LOG.info(f"connectivity test OK — fetched {len(df_test)} candles, latest={df_test['time'].max() if 'time' in df_test.columns else 'unknown'}")
    except Exception as exc:
        LOG.error(f"connectivity test FAILED: {exc}")
        return 1

    try:
        collect_loop(client)
    except KeyboardInterrupt:
        LOG.info("interrupted — exiting")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
