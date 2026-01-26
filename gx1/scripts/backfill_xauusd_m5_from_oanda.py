#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Idempotent XAU_USD M5 Candle Backfill from OANDA

Fetches missing M5 candles from OANDA and merges with existing data.
Safe to run multiple times - only fetches new candles.

Usage:
    python gx1/scripts/backfill_xauusd_m5_from_oanda.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from gx1.scripts.backfill_oanda_candles_m5 import (
    fetch_candles_bid_ask,
    merge_candles,
)
from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Default paths
DEFAULT_CANDLE_FILE = Path("data/raw/xauusd_m5_2025_bid_ask.parquet")
INSTRUMENT = "XAU_USD"
GRANULARITY = "M5"


def main() -> dict:
    """
    Main entry point for idempotent candle backfill.
    
    Returns:
        dict with keys: new_candles, from_time, to_time, success
    """
    log.info("=" * 60)
    log.info("XAU_USD M5 Candle Backfill (Idempotent)")
    log.info("=" * 60)
    
    # Load .env
    load_dotenv_if_present()
    
    # Load OANDA credentials
    try:
        credentials = load_oanda_credentials(prod_baseline=False)
        log.info(f"Loaded OANDA credentials: env={credentials.env}")
    except Exception as e:
        log.error(f"Failed to load OANDA credentials: {e}")
        return {"success": False, "error": str(e)}
    
    # Initialize OANDA client
    try:
        config = OandaClientConfig(
            api_key=credentials.api_token,
            account_id=credentials.account_id,
            env=credentials.env,
        )
        client = OandaClient(config)
        log.info("Initialized OANDA client")
    except Exception as e:
        log.error(f"Failed to initialize OANDA client: {e}")
        return {"success": False, "error": str(e)}
    
    # Load existing candle file
    candle_file = DEFAULT_CANDLE_FILE
    existing_df = pd.DataFrame()
    
    if candle_file.exists():
        try:
            existing_df = pd.read_parquet(candle_file)
            log.info(f"Loaded {len(existing_df):,} existing candles")
            if len(existing_df) > 0:
                log.info(f"  Existing range: {existing_df.index.min()} to {existing_df.index.max()}")
        except Exception as e:
            log.warning(f"Failed to load existing parquet: {e}")
            existing_df = pd.DataFrame()
    else:
        log.info("Candle file does not exist - will create new file")
    
    # Determine time range
    now_utc = pd.Timestamp.now(tz="UTC")
    now_utc_floor = now_utc.floor("5min")  # Round down to nearest M5
    
    if existing_df.empty or len(existing_df) == 0:
        # No existing data - fetch from a reasonable start date
        from_time = pd.Timestamp("2025-01-01", tz="UTC")
        to_time = now_utc_floor
        log.info(f"No existing candles - fetching from {from_time.date()} to {to_time.date()}")
    else:
        # Get last timestamp
        last_ts = existing_df.index.max()
        log.info(f"Last existing candle: {last_ts}")
        
        # Check if we need to fetch new candles
        # We want at least 1 bar gap before fetching (to avoid fetching incomplete bars)
        next_bar = last_ts + pd.Timedelta(minutes=5)
        
        if last_ts >= now_utc_floor - pd.Timedelta(minutes=5):
            log.info(f"[BACKFILL_CANDLES] No new candles to backfill (last_ts={last_ts}, now={now_utc_floor})")
            return {
                "success": True,
                "new_candles": 0,
                "from_time": last_ts,
                "to_time": now_utc_floor,
                "total_candles": len(existing_df),
            }
        
        from_time = next_bar
        to_time = now_utc_floor
        log.info(f"Fetching candles from {from_time} to {to_time}")
    
    # Fetch new candles
    try:
        new_df = fetch_candles_bid_ask(
            client=client,
            instrument=INSTRUMENT,
            start=from_time,
            end=to_time,
            granularity=GRANULARITY,
        )
    except Exception as e:
        log.error(f"Failed to fetch candles: {e}")
        return {"success": False, "error": str(e)}
    
    if new_df.empty:
        log.info(f"[BACKFILL_CANDLES] No new candles fetched (API returned empty for {from_time} to {to_time})")
        return {
            "success": True,
            "new_candles": 0,
            "from_time": from_time,
            "to_time": to_time,
            "total_candles": len(existing_df),
        }
    
    log.info(f"[BACKFILL_CANDLES] Fetched {len(new_df):,} new candles")
    log.info(f"[BACKFILL_CANDLES] New range: {new_df.index.min()} to {new_df.index.max()}")
    
    # Merge with existing data
    if existing_df.empty:
        final_df = new_df
    else:
        log.info("Merging with existing candles...")
        final_df = merge_candles(existing_df, new_df)
        log.info(f"Merged: {len(existing_df):,} existing + {len(new_df):,} new = {len(final_df):,} total")
    
    # Save result
    candle_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(candle_file)
    
    log.info("=" * 60)
    log.info("✅ Candle Backfill Complete!")
    log.info("=" * 60)
    log.info(f"[BACKFILL_CANDLES] Added {len(new_df):,} candles from {from_time} to {to_time}. Total now: {len(final_df):,} bars.")
    
    return {
        "success": True,
        "new_candles": len(new_df),
        "from_time": from_time,
        "to_time": to_time,
        "total_candles": len(final_df),
    }


if __name__ == "__main__":
    result = main()
    if not result.get("success", False):
        sys.exit(1)
    sys.exit(0)

