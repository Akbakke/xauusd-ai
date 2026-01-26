#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill Trade Journal from OANDA Transaction History

Fetches all OANDA transactions for a time window, reconstructs trades,
and writes them to SNIPER's trade_journal format.

Usage:
    python gx1/scripts/backfill_trade_journal_from_oanda.py \
        --run_dir runs/live_demo/SNIPER_20251226_113527 \
        --instrument XAU_USD
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present
from gx1.utils.pnl import compute_pnl_bps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_iso_time(ts_str: str) -> datetime:
    """Parse ISO timestamp string to datetime."""
    # Handle RFC3339 format from OANDA
    ts_str = ts_str.replace("Z", "+00:00")
    if "+" not in ts_str and ts_str.count("-") >= 3:
        # Assume UTC if no timezone
        ts_str = ts_str + "+00:00"
    
    # OANDA timestamps can have nanoseconds (9 digits), but Python only supports up to 6
    # Try to parse, and if it fails, truncate nanoseconds
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        # Handle nanosecond precision by truncating to microseconds
        # Format: 2025-09-23T10:13:39.223390986+00:00
        if "." in ts_str and "+" in ts_str:
            parts = ts_str.split(".")
            if len(parts) == 2:
                decimal_part = parts[1].split("+")[0].split("-")[0]
                if len(decimal_part) > 6:
                    # Truncate to 6 digits (microseconds)
                    truncated_decimal = decimal_part[:6]
                    timezone_part = ts_str.split("+")[1] if "+" in ts_str else ts_str.split("-")[-1]
                    if "+" in ts_str:
                        ts_str = f"{parts[0]}.{truncated_decimal}+{timezone_part}"
                    else:
                        # Handle negative timezone
                        ts_str = f"{parts[0]}.{truncated_decimal}-{timezone_part}"
        return datetime.fromisoformat(ts_str)


def infer_time_window_from_shadow(run_dir: Path) -> Tuple[datetime, datetime]:
    """
    Infer time window from shadow journal timestamps.
    
    Returns:
        (from_time, to_time) with 1 day buffer on each side
    """
    # Try multiple shadow journal locations
    shadow_paths = [
        run_dir / "shadow" / "shadow_hits.jsonl",
        run_dir / "shadow" / "shadow" / "shadow_hits.jsonl",
        # Also try parent directory (runs/live_demo/shadow/shadow/shadow_hits.jsonl)
        run_dir.parent / "shadow" / "shadow" / "shadow_hits.jsonl",
        run_dir.parent.parent / "shadow" / "shadow" / "shadow_hits.jsonl",
    ]
    
    timestamps = []
    for shadow_path in shadow_paths:
        if shadow_path.exists():
            log.info(f"Reading shadow journal: {shadow_path}")
            with open(shadow_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        ts_str = record.get("ts")
                        if ts_str:
                            ts = parse_iso_time(ts_str)
                            timestamps.append(ts)
                    except Exception as e:
                        log.warning(f"Failed to parse shadow line: {e}")
                        continue
            break  # Use first found shadow journal
    
    if not timestamps:
        raise ValueError(f"No shadow timestamps found. Checked paths: {shadow_paths}")
    
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    
    # Add 1 day buffer on each side
    from_time = min_ts - timedelta(days=1)
    to_time = max_ts + timedelta(days=1)
    
    log.info(f"Inferred time window from shadow: {from_time.isoformat()} to {to_time.isoformat()}")
    return from_time, to_time


def fetch_trades_from_oanda(
    client: OandaClient,
    instrument: str = "XAU_USD",
) -> List[Dict[str, Any]]:
    """
    Fetch all trades (open + closed) from OANDA using get_trades() API.
    
    This is the preferred method since transactions API doesn't return
    data for open trades in Practice accounts.
    
    Returns:
        List of all trades (both open and closed)
    """
    all_trades = []
    
    # Fetch open trades
    try:
        open_response = client.get_trades(state="OPEN", instrument=instrument)
        open_trades = open_response.get("trades", [])
        log.info(f"Fetched {len(open_trades)} open trades")
        all_trades.extend(open_trades)
    except Exception as e:
        log.warning(f"Failed to fetch open trades: {e}")
    
    # Fetch closed trades
    try:
        closed_response = client.get_trades(state="CLOSED", instrument=instrument, count=500)
        closed_trades = closed_response.get("trades", [])
        log.info(f"Fetched {len(closed_trades)} closed trades")
        all_trades.extend(closed_trades)
    except Exception as e:
        log.warning(f"Failed to fetch closed trades: {e}")
    
    # Also try ALL state to catch any we might have missed
    try:
        all_response = client.get_trades(state="ALL", instrument=instrument, count=500)
        all_state_trades = all_response.get("trades", [])
        # Deduplicate by trade ID
        existing_ids = {str(t.get("id", "")) for t in all_trades}
        new_trades = [t for t in all_state_trades if str(t.get("id", "")) not in existing_ids]
        if new_trades:
            log.info(f"Found {len(new_trades)} additional trades from ALL state")
            all_trades.extend(new_trades)
    except Exception as e:
        log.warning(f"Failed to fetch ALL trades: {e}")
    
    log.info(f"Total trades fetched: {len(all_trades)}")
    return all_trades


def reconstruct_trades_from_oanda_trades(
    trades: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Reconstruct trades from OANDA trade objects (from get_trades() API).
    
    This is simpler than reconstructing from transactions since trade objects
    already contain all the information we need.
    
    Returns:
        List of reconstructed trade dicts
    """
    reconstructed_trades = []
    
    for trade in trades:
        trade_id = str(trade.get("id", ""))
        if not trade_id:
            log.warning("Skipping trade with no ID")
            continue
        
        # Extract entry information
        entry_time_str = trade.get("openTime", "")
        entry_price = trade.get("price", 0.0)
        
        # Try to get units - can be currentUnits or initialUnits, and might be string or number
        # For closed trades, currentUnits might be "0" but initialUnits has the original size
        state = trade.get("state", "")
        
        # For closed trades, always use initialUnits
        # For open trades, prefer currentUnits but fall back to initialUnits
        if state == "CLOSED":
            units_raw = trade.get("initialUnits", 0)
        else:
            units_raw = trade.get("currentUnits")
            if not units_raw or (isinstance(units_raw, str) and units_raw in ("0", "0.0")):
                units_raw = trade.get("initialUnits", 0)
        
        try:
            if isinstance(units_raw, str):
                units = float(units_raw)
            else:
                units = float(units_raw) if units_raw else 0.0
        except (ValueError, TypeError):
            units = 0.0
        
        # Determine side from units
        if units > 0:
            side = "LONG"
        elif units < 0:
            side = "SHORT"
        else:
            log.warning(f"Trade {trade_id} has zero units (currentUnits={trade.get('currentUnits')}, initialUnits={trade.get('initialUnits')}), skipping")
            continue
        
        # Extract exit information (for closed trades)
        # (state already extracted above)
        exit_time_str = None
        exit_price = None
        realized_pnl = None
        
        if state == "CLOSED":
            exit_time_str = trade.get("closeTime", "")
            # For closed trades, we need to get exit price from averageClosePrice
            exit_price = trade.get("averageClosePrice", 0.0)
            realized_pnl = trade.get("realizedPL", 0.0)
        elif state == "OPEN":
            # Open trade - no exit yet
            pass
        else:
            log.warning(f"Trade {trade_id} has unknown state: {state}")
        
        # Parse timestamps
        try:
            entry_time = parse_iso_time(entry_time_str) if entry_time_str else None
            exit_time = parse_iso_time(exit_time_str) if exit_time_str else None
        except Exception as e:
            log.warning(f"Failed to parse timestamps for trade {trade_id}: {e}")
            continue
        
        if not entry_time:
            log.warning(f"Trade {trade_id} has no entry time, skipping")
            continue
        
        # Build trade record
        trade_record = {
            "oanda_trade_id": trade_id,
            "entry_time": entry_time.isoformat(),
            "entry_price": float(entry_price),
            "entry_units": int(abs(units)),  # Store absolute value
            "side": side,
            "exit_time": exit_time.isoformat() if exit_time else None,
            "exit_price": float(exit_price) if exit_price else None,
            "realized_pnl": float(realized_pnl) if realized_pnl is not None else None,
            "state": state,
        }
        
        reconstructed_trades.append(trade_record)
    
    log.info(f"Reconstructed {len(reconstructed_trades)} trades from OANDA trade objects")
    return reconstructed_trades


def calculate_pnl_bps_from_trade(
    trade: Dict[str, Any],
) -> Optional[float]:
    """
    Calculate PnL in basis points from trade record.
    
    Uses realized_pnl if available, otherwise calculates from entry/exit prices.
    """
    # If we have realized_pnl, we need to convert it to bps
    # For XAU_USD, realized_pnl is in account currency (USD typically)
    # We need entry_price and units to convert to bps
    
    entry_price = trade.get("entry_price")
    exit_price = trade.get("exit_price")
    units = trade.get("entry_units", 0)
    side = trade.get("side", "").upper()
    
    if not entry_price or not exit_price:
        return None
    
    # For basis points calculation, we need bid/ask
    # Since OANDA transactions don't always provide bid/ask separately,
    # we'll use the price as both bid and ask (mid price approximation)
    # This is a limitation but acceptable for backfill
    
    if side == "LONG":
        # LONG: entry at ask, exit at bid
        # Approximation: use price as mid, assume small spread
        entry_bid = entry_price * 0.99995  # ~0.5 bps spread
        entry_ask = entry_price * 1.00005
        exit_bid = exit_price * 0.99995
        exit_ask = exit_price * 1.00005
    else:  # SHORT
        # SHORT: entry at bid, exit at ask
        entry_bid = entry_price * 0.99995
        entry_ask = entry_price * 1.00005
        exit_bid = exit_price * 0.99995
        exit_ask = exit_price * 1.00005
    
    try:
        pnl_bps = compute_pnl_bps(
            entry_bid=entry_bid,
            entry_ask=entry_ask,
            exit_bid=exit_bid,
            exit_ask=exit_ask,
            side=side.lower(),
        )
        return pnl_bps
    except Exception as e:
        log.warning(f"Failed to calculate PnL bps: {e}")
        return None


def write_trade_journal_entry(
    run_dir: Path,
    trade: Dict[str, Any],
    instrument: str,
    force: bool = False,
) -> None:
    """
    Write a single trade to trade journal format.
    
    Creates minimal but valid trade journal JSON and updates index CSV.
    """
    trade_journal_dir = run_dir / "trade_journal" / "trades"
    trade_journal_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate trade_id from OANDA trade ID
    oanda_trade_id = trade.get("oanda_trade_id", "")
    trade_id = f"OANDA-{oanda_trade_id}"
    
    # Calculate PnL bps
    pnl_bps = calculate_pnl_bps_from_trade(trade)
    
    # Build minimal trade journal JSON
    trade_journal = {
        "trade_id": trade_id,
        "instrument": instrument,
        "side": trade.get("side", "").lower(),
        "units": trade.get("entry_units", 0),
        "entry_time": trade.get("entry_time"),
        "entry_price": trade.get("entry_price"),
        "exit_time": trade.get("exit_time"),
        "exit_price": trade.get("exit_price"),
        "pnl_bps": pnl_bps,
        "entry_snapshot": {
            "trade_id": trade_id,
            "entry_time": trade.get("entry_time"),
            "instrument": instrument,
            "side": trade.get("side", "").lower(),
            "entry_price": trade.get("entry_price"),
            "units": trade.get("entry_units", 0),
        },
        "exit_summary": {
            "exit_time": trade.get("exit_time"),
            "exit_price": trade.get("exit_price"),
            "exit_reason": "OANDA_BACKFILL",  # Unknown from transactions
            "realized_pnl_bps": pnl_bps,
        } if trade.get("exit_time") else None,
        "extra": {
            "oanda_trade_id": oanda_trade_id,
            "source": "OANDA_BACKFILL",
        },
    }
    
    # Write JSON file
    trade_json_path = trade_journal_dir / f"{trade_id}.json"
    with open(trade_json_path, "w", encoding="utf-8") as f:
        json.dump(trade_journal, f, indent=2, ensure_ascii=False, default=str)
    
    log.debug(f"Wrote trade journal: {trade_json_path}")


def update_trade_journal_index(
    run_dir: Path,
    trades: List[Dict[str, Any]],
    instrument: str,
) -> None:
    """
    Write/overwrite trade_journal_index.csv with backfilled trades.
    """
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    
    fieldnames = [
        "trade_id",
        "instrument",
        "side",
        "units",
        "entry_time",
        "exit_time",
        "pnl_bps",
        "oanda_trade_id",
    ]
    
    rows = []
    for trade in trades:
        oanda_trade_id = trade.get("oanda_trade_id", "")
        trade_id = f"OANDA-{oanda_trade_id}"
        pnl_bps = calculate_pnl_bps_from_trade(trade)
        
        row = {
            "trade_id": trade_id,
            "instrument": instrument,
            "side": trade.get("side", "").lower(),
            "units": trade.get("entry_units", 0),
            "entry_time": trade.get("entry_time", ""),
            "exit_time": trade.get("exit_time", ""),
            "pnl_bps": pnl_bps if pnl_bps is not None else "",
            "oanda_trade_id": oanda_trade_id,
        }
        rows.append(row)
    
    # Write CSV
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    log.info(f"Wrote trade journal index: {index_path} ({len(rows)} trades)")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backfill Trade Journal from OANDA Transaction History"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("runs/live_demo/SNIPER_20251226_113527"),
        help="Path to SNIPER live run directory",
    )
    parser.add_argument(
        "--from_time",
        type=str,
        default=None,
        help="Start time (ISO format, optional - inferred from shadow if not provided)",
    )
    parser.add_argument(
        "--to_time",
        type=str,
        default=None,
        help="End time (ISO format, optional - inferred from shadow if not provided)",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="XAU_USD",
        help="Instrument symbol (default: XAU_USD)",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of trade journal (overwrite existing entries)",
    )
    
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("OANDA → Trade Journal Backfill")
    log.info("=" * 60)
    log.info(f"Run directory: {args.run_dir}")
    log.info(f"Instrument: {args.instrument}")
    
    # Validate run directory
    if not args.run_dir.exists():
        log.error(f"Run directory not found: {args.run_dir}")
        return 1
    
    # Determine time window
    if args.from_time and args.to_time:
        from_time = parse_iso_time(args.from_time)
        to_time = parse_iso_time(args.to_time)
        log.info(f"Using provided time window: {from_time.isoformat()} to {to_time.isoformat()}")
    else:
        log.info("Inferring time window from shadow journal...")
        from_time, to_time = infer_time_window_from_shadow(args.run_dir)
    
    # Load .env file first
    load_dotenv_if_present()
    
    # Load OANDA credentials
    try:
        credentials = load_oanda_credentials(prod_baseline=False)
        log.info(f"Loaded OANDA credentials: env={credentials.env}, account_id={credentials.account_id[:10]}...")
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
    
    # Fetch trades from OANDA (using get_trades() instead of transactions API)
    log.info(f"Fetching trades from OANDA for {args.instrument}...")
    trades = fetch_trades_from_oanda(client, instrument=args.instrument)
    
    if not trades:
        log.warning(f"No trades found for {args.instrument} - nothing to backfill")
        return 0
    
    # Filter trades by time window (if provided)
    if args.from_time and args.to_time:
        from_time = parse_iso_time(args.from_time)
        to_time = parse_iso_time(args.to_time)
        
        filtered_trades = []
        for trade in trades:
            entry_time_str = trade.get("openTime", "")
            if entry_time_str:
                try:
                    entry_time = parse_iso_time(entry_time_str)
                    if from_time <= entry_time <= to_time:
                        filtered_trades.append(trade)
                except Exception:
                    pass
        
        log.info(f"Filtered {len(filtered_trades)} trades within time window (from {len(trades)} total)")
        trades = filtered_trades
    
    if not trades:
        log.warning("No trades found within specified time window")
        return 0
    
    # Reconstruct trades
    log.info("Reconstructing trades from OANDA trade objects...")
    reconstructed_trades = reconstruct_trades_from_oanda_trades(trades)
    
    if not reconstructed_trades:
        log.warning("No trades could be reconstructed")
        return 0
    
    # Write trade journal entries
    log.info(f"Writing {len(reconstructed_trades)} trades to trade journal...")
    for trade in reconstructed_trades:
        write_trade_journal_entry(args.run_dir, trade, args.instrument)
    
    # Update index CSV
    update_trade_journal_index(args.run_dir, reconstructed_trades, args.instrument)
    
    # Summary
    closed_trades = sum(1 for t in reconstructed_trades if t.get("exit_time"))
    open_trades = len(reconstructed_trades) - closed_trades
    
    log.info("=" * 60)
    log.info("✅ Backfill Complete!")
    log.info("=" * 60)
    log.info(f"Total trades: {len(reconstructed_trades)}")
    log.info(f"  - Closed: {closed_trades}")
    log.info(f"  - Open: {open_trades}")
    log.info(f"Trade journal directory: {args.run_dir / 'trade_journal' / 'trades'}")
    log.info(f"Trade journal index: {args.run_dir / 'trade_journal' / 'trade_journal_index.csv'}")
    
    return 0


if __name__ == "__main__":
    exit(main())

