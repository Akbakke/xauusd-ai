#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill OANDA M5 Candles

Fetches M5 candles from OANDA and merges with existing candle data.
Extends candle history up to today (or specified date range).

Usage:
    python gx1/scripts/backfill_oanda_candles_m5.py \
        --instrument XAU_USD \
        --output_path data/raw/xauusd_m5_2025_bid_ask.parquet \
        --merge true
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def fetch_candles_bid_ask(
    client: OandaClient,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: str = "M5",
) -> pd.DataFrame:
    """
    Fetch candles with bid/ask prices from OANDA.
    
    Returns DataFrame with columns matching existing parquet:
    - time (index)
    - open, high, low, close (mid prices)
    - volume
    - bid_open, bid_high, bid_low, bid_close
    - ask_open, ask_high, ask_low, ask_close
    """
    log.info(f"Fetching {instrument} {granularity} from {start.date()} to {end.date()}")
    
    all_candles = []
    chunk_days = 15  # ~17 days max per request (M5 = 288/day, limit ~5000)
    current_start = start
    
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days), end)
        
        log.info(f"  Fetching chunk: {current_start.date()} to {current_end.date()}")
        
        # Use _request directly to get raw response with bid/ask
        params = {
            "from": current_start.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "to": current_end.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
            "granularity": granularity,
            "price": "MBA",  # M=mid, B=bid, A=ask
        }
        
        try:
            data = client._request(
                "GET",
                f"/instruments/{instrument}/candles",
                params=params,
            )
        except Exception as e:
            log.error(f"  Failed to fetch chunk {current_start.date()} to {current_end.date()}: {e}")
            current_start = current_end
            continue
        
        candles = data.get("candles", [])
        if not candles:
            log.warning(f"  No candles for {current_start.date()} to {current_end.date()}")
            current_start = current_end
            continue
        
        # Parse candles with bid/ask
        rows = []
        for candle in candles:
            # Only include complete bars
            if not candle.get("complete", False):
                continue
            
            time_str = candle["time"]
            mid = candle.get("mid", {})
            bid = candle.get("bid", {})
            ask = candle.get("ask", {})
            volume = candle.get("volume", 0)
            
            # Parse timestamp
            raw_time = pd.to_datetime(time_str)
            if raw_time.tzinfo is None:
                raw_time = raw_time.tz_localize("UTC")
            else:
                raw_time = raw_time.tz_convert("UTC")
            normalized_time = raw_time.floor("5min")
            
            rows.append({
                "time": normalized_time,
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": float(volume),
                "bid_open": float(bid.get("o", 0)) if bid else float(mid.get("o", 0)),
                "bid_high": float(bid.get("h", 0)) if bid else float(mid.get("h", 0)),
                "bid_low": float(bid.get("l", 0)) if bid else float(mid.get("l", 0)),
                "bid_close": float(bid.get("c", 0)) if bid else float(mid.get("c", 0)),
                "ask_open": float(ask.get("o", 0)) if ask else float(mid.get("o", 0)),
                "ask_high": float(ask.get("h", 0)) if ask else float(mid.get("h", 0)),
                "ask_low": float(ask.get("l", 0)) if ask else float(mid.get("l", 0)),
                "ask_close": float(ask.get("c", 0)) if ask else float(mid.get("c", 0)),
            })
        
        if rows:
            df_chunk = pd.DataFrame(rows)
            df_chunk = df_chunk.set_index("time")
            all_candles.append(df_chunk)
            log.info(f"  ✅ Fetched {len(df_chunk):,} candles")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
        current_start = current_end
    
    if not all_candles:
        log.warning("No candles fetched")
        return pd.DataFrame()
    
    # Combine all chunks
    df = pd.concat(all_candles, axis=0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
    
    return df


def determine_date_range(
    output_path: Path,
    from_date: Optional[str],
    to_date: Optional[str],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Determine date range for backfill.
    
    If from_date/to_date provided, use them.
    Else, auto-detect from existing parquet file.
    """
    if from_date and to_date:
        from_ts = pd.to_datetime(from_date).tz_localize("UTC")
        to_ts = pd.to_datetime(to_date).tz_localize("UTC")
        log.info(f"Using provided date range: {from_ts.date()} to {to_ts.date()}")
        return from_ts, to_ts
    
    # Auto-detect from existing file
    if output_path.exists():
        log.info(f"Reading existing parquet: {output_path}")
        existing_df = pd.read_parquet(output_path)
        
        if len(existing_df) == 0:
            log.warning("Existing parquet is empty, using default range")
            from_ts = pd.Timestamp("2025-01-01", tz="UTC")
            to_ts = pd.Timestamp.now(tz="UTC") + timedelta(days=1)
            return from_ts, to_ts
        
        # Get last timestamp
        last_ts = existing_df.index.max()
        if isinstance(last_ts, pd.Timestamp):
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")
        else:
            # If index is not timestamp, try to find a time column
            if "time" in existing_df.columns:
                last_ts = pd.to_datetime(existing_df["time"].max())
                if last_ts.tzinfo is None:
                    last_ts = last_ts.tz_localize("UTC")
            else:
                log.warning("Cannot determine last timestamp, using default range")
                from_ts = pd.Timestamp("2025-01-01", tz="UTC")
                to_ts = pd.Timestamp.now(tz="UTC") + timedelta(days=1)
                return from_ts, to_ts
        
        # Start from day after last timestamp
        from_ts = last_ts + timedelta(minutes=5)  # Start from next M5 bar
        # End at current time (floor to 5min to exclude incomplete bar)
        now_utc = pd.Timestamp.now(tz="UTC")
        to_ts = now_utc.floor("5min")  # Exclude incomplete current bar
        
        log.info(f"Auto-detected range: {from_ts.date()} to {to_ts.date()}")
        log.info(f"  Last existing candle: {last_ts}")
        return from_ts, to_ts
    else:
        # File doesn't exist, use default range
        log.info("Output file doesn't exist, using default range")
        from_ts = pd.Timestamp("2025-01-01", tz="UTC")
        to_ts = pd.Timestamp.now(tz="UTC") + timedelta(days=1)
        return from_ts, to_ts


def merge_candles(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge existing and new candles, removing duplicates.
    
    Keeps last occurrence if duplicate timestamps exist.
    """
    if existing_df.empty:
        return new_df
    
    if new_df.empty:
        return existing_df
    
    # Combine
    combined = pd.concat([existing_df, new_df], axis=0)
    
    # Remove duplicates (keep last)
    combined = combined[~combined.index.duplicated(keep='last')]
    
    # Sort by timestamp
    combined = combined.sort_index()
    
    return combined


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill OANDA M5 Candles"
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="XAU_USD",
        help="Instrument symbol (default: XAU_USD)",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="M5",
        help="Granularity (default: M5)",
    )
    parser.add_argument(
        "--from_date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD, optional - auto-detected if not provided)",
    )
    parser.add_argument(
        "--to_date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, optional - defaults to today+1 day)",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/raw/xauusd_m5_2025_bid_ask.parquet"),
        help="Output parquet file path",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge with existing parquet if it exists (default: True)",
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="Overwrite existing parquet (don't merge)",
    )
    
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("OANDA M5 Candle Backfill")
    log.info("=" * 60)
    log.info(f"Instrument: {args.instrument}")
    log.info(f"Granularity: {args.granularity}")
    log.info(f"Output: {args.output_path}")
    log.info(f"Merge: {args.merge}")
    
    # Load .env file
    load_dotenv_if_present()
    
    # Load OANDA credentials
    try:
        credentials = load_oanda_credentials(prod_baseline=False)
        log.info(f"Loaded OANDA credentials: env={credentials.env}")
    except Exception as e:
        log.error(f"Failed to load OANDA credentials: {e}")
        return 1
    
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
        return 1
    
    # Determine date range
    from_ts, to_ts = determine_date_range(args.output_path, args.from_date, args.to_date)
    
    if from_ts >= to_ts:
        log.warning("Date range is empty or invalid (from_ts >= to_ts)")
        log.info("No new candles to fetch")
        return 0
    
    # Load existing candles if merging
    existing_df = pd.DataFrame()
    if args.merge and args.output_path.exists():
        try:
            existing_df = pd.read_parquet(args.output_path)
            log.info(f"Loaded {len(existing_df):,} existing candles")
            if len(existing_df) > 0:
                log.info(f"  Existing range: {existing_df.index.min()} to {existing_df.index.max()}")
        except Exception as e:
            log.warning(f"Failed to load existing parquet: {e}")
            existing_df = pd.DataFrame()
    
    # Fetch new candles
    log.info(f"Fetching candles from {from_ts.date()} to {to_ts.date()}...")
    new_df = fetch_candles_bid_ask(
        client=client,
        instrument=args.instrument,
        start=from_ts,
        end=to_ts,
        granularity=args.granularity,
    )
    
    if new_df.empty:
        log.warning("No new candles fetched")
        return 0
    
    log.info(f"Fetched {len(new_df):,} new candles")
    log.info(f"  New range: {new_df.index.min()} to {new_df.index.max()}")
    
    # Merge if requested
    if args.merge and not existing_df.empty:
        log.info("Merging with existing candles...")
        final_df = merge_candles(existing_df, new_df)
        log.info(f"Merged: {len(existing_df):,} existing + {len(new_df):,} new = {len(final_df):,} total")
    else:
        final_df = new_df
        log.info(f"Writing {len(final_df):,} candles (no merge)")
    
    # Save result
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(args.output_path)
    
    log.info("=" * 60)
    log.info("✅ Candle Backfill Complete!")
    log.info("=" * 60)
    log.info(f"[CANDLE_BACKFILL] Final candles: {len(final_df):,}")
    log.info(f"[CANDLE_BACKFILL] Date range: {final_df.index.min()} to {final_df.index.max()}")
    if not existing_df.empty:
        log.info(f"[CANDLE_BACKFILL] Existing: {len(existing_df):,} candles")
        log.info(f"[CANDLE_BACKFILL] New: {len(new_df):,} candles")
        log.info(f"[CANDLE_BACKFILL] Old max_ts: {existing_df.index.max()}")
        log.info(f"[CANDLE_BACKFILL] New max_ts: {final_df.index.max()}")
    
    return 0


if __name__ == "__main__":
    exit(main())

