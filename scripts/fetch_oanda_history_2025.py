#!/usr/bin/env python3
"""
Download M5 historical data from OANDA for 2025 (with bid/ask).

Downloads XAUUSD M5 candles from 2025-01-01 to 2025-12-05 (Friday market close).
Saves as parquet with bid/ask columns.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional

from gx1.execution.oanda_client import OandaClient, OandaClientConfig, OandaAPIError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def fetch_candles_bid_ask(
    client: OandaClient,
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    granularity: str = "M5",
) -> pd.DataFrame:
    """
    Fetch candles with bid/ask prices from OANDA using get_candles_chunked.
    
    Returns DataFrame with columns:
    - time (index)
    - open, high, low, close (mid prices)
    - volume
    - bid_open, bid_high, bid_low, bid_close
    - ask_open, ask_high, ask_low, ask_close
    """
    log.info(f"Fetching {instrument} {granularity} from {start.date()} to {end.date()}")
    
    # Use existing get_candles_chunked method which handles chunking and retries
    df = client.get_candles_chunked(
        instrument=instrument,
        granularity=granularity,
        from_ts=start,
        to_ts=end,
        include_mid=True,  # Include mid prices
    )
    
    # The get_candles_chunked returns mid prices in open/high/low/close
    # We need to extract bid/ask from the raw API response
    # Let's fetch again with price="BA" to get bid/ask, then merge
    
    # Actually, get_candles_chunked uses price="MBA" which should include bid/ask
    # But it only returns mid prices. We need to parse the raw response.
    # For now, let's use a simpler approach: fetch in chunks and parse bid/ask manually
    
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
        
        data = client._request(
            "GET",
            f"/instruments/{instrument}/candles",
            params=params,
        )
        
        candles = data.get("candles", [])
        if not candles:
            log.warning(f"  No candles for {current_start.date()} to {current_end.date()}")
            current_start = current_end
            continue
        
        # Parse candles with bid/ask
        rows = []
        for candle in candles:
            if candle.get("complete", False):
                time_str = candle["time"]
                mid = candle.get("mid", {})
                bid = candle.get("bid", {})
                ask = candle.get("ask", {})
                volume = candle.get("volume", 0)
                
                rows.append({
                    "time": pd.to_datetime(time_str),
                    "open": float(mid.get("o", 0)),
                    "high": float(mid.get("h", 0)),
                    "low": float(mid.get("l", 0)),
                    "close": float(mid.get("c", 0)),
                    "volume": int(volume),
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
        raise ValueError("No candles fetched")
    
    # Combine all chunks
    df = pd.concat(all_candles, axis=0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Download OANDA M5 historical data with bid/ask")
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OANDA API key",
    )
    parser.add_argument(
        "--account-id",
        type=str,
        required=True,
        help="OANDA account ID",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="XAU_USD",
        help="Instrument (default: XAU_USD)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date (YYYY-MM-DD, default: 2025-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-05",
        help="End date (YYYY-MM-DD, default: 2025-12-05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/xauusd_m5_2025_bid_ask.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="live",
        choices=["practice", "live"],
        help="OANDA environment (default: live)",
    )
    args = parser.parse_args()
    
    # Parse dates
    start_date = pd.to_datetime(args.start_date).tz_localize("UTC")
    end_date = pd.to_datetime(args.end_date).tz_localize("UTC")
    # Add end of day for end_date (Friday market close ~22:00 UTC)
    end_date = end_date.replace(hour=22, minute=0, second=0)
    
    log.info(f"Downloading {args.instrument} M5 data from {start_date.date()} to {end_date.date()}")
    log.info(f"Output: {args.output}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize OANDA client
    config = OandaClientConfig(
        api_key=args.api_key,
        account_id=args.account_id,
        env=args.env,
    )
    client = OandaClient(config)
    
    # Fetch candles
    log.info("Starting download...")
    df = fetch_candles_bid_ask(
        client=client,
        instrument=args.instrument,
        start=start_date,
        end=end_date,
        granularity="M5",
    )
    
    log.info(f"✅ Downloaded {len(df):,} candles")
    log.info(f"   Period: {df.index.min()} to {df.index.max()}")
    log.info(f"   Columns: {list(df.columns)}")
    
    # Save to parquet
    df.to_parquet(output_path, compression="snappy")
    log.info(f"✅ Saved to {output_path} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
    
    return 0


if __name__ == "__main__":
    exit(main())

