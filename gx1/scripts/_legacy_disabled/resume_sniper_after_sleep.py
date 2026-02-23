#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNIPER Resume After Sleep

Orchestrates:
1. Candle backfill from OANDA
2. (Optional) Trade backfill for today
3. Start SNIPER live

Usage:
    python gx1/scripts/resume_sniper_after_sleep.py [--skip-trade-backfill]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from gx1.scripts.backfill_xauusd_m5_from_oanda import main as backfill_candles
from gx1.scripts.backfill_trade_journal_from_oanda import (
    parse_iso_time,
    fetch_trades_from_oanda,
    reconstruct_trades_from_oanda_trades,
    update_trade_journal_index,
    write_trade_journal_entry,
)
from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def find_latest_sniper_run(runs_dir: Path) -> Optional[Path]:
    """
    Find the latest SNIPER run directory.
    
    Looks for directories matching SNIPER_* pattern and returns the newest one.
    """
    if not runs_dir.exists():
        return None
    
    sniper_dirs = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("SNIPER_")
    ]
    
    if not sniper_dirs:
        return None
    
    # Sort by name (which includes timestamp) and return newest
    sniper_dirs.sort(key=lambda x: x.name, reverse=True)
    return sniper_dirs[0]


def backfill_trades_for_today(
    run_dir: Path,
    instrument: str = "XAU_USD",
) -> dict:
    """
    Backfill trades for today into trade journal.
    
    Returns:
        dict with keys: success, trades_backfilled
    """
    log.info("=" * 60)
    log.info("Trade Backfill for Today")
    log.info("=" * 60)
    
    # Determine today's date (UTC)
    today_utc = datetime.now(timezone.utc).date()
    start_of_day = datetime.combine(today_utc, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_of_day = datetime.combine(today_utc, datetime.max.time()).replace(tzinfo=timezone.utc)
    
    log.info(f"Backfilling trades for {today_utc}")
    log.info(f"  Time range: {start_of_day} to {end_of_day}")
    
    # Load OANDA credentials
    try:
        credentials = load_oanda_credentials(prod_baseline=False)
    except Exception as e:
        log.warning(f"Failed to load OANDA credentials: {e}")
        return {"success": False, "trades_backfilled": 0, "error": str(e)}
    
    # Initialize OANDA client
    try:
        config = OandaClientConfig(
            api_key=credentials.api_token,
            account_id=credentials.account_id,
            env=credentials.env,
        )
        client = OandaClient(config)
    except Exception as e:
        log.warning(f"Failed to initialize OANDA client: {e}")
        return {"success": False, "trades_backfilled": 0, "error": str(e)}
    
    # Fetch trades
    try:
        all_trades = fetch_trades_from_oanda(client, instrument=instrument)
    except Exception as e:
        log.warning(f"Failed to fetch trades from OANDA: {e}")
        return {"success": False, "trades_backfilled": 0, "error": str(e)}
    
    if not all_trades:
        log.info("No trades found in OANDA")
        return {"success": True, "trades_backfilled": 0}
    
    # Filter to today
    filtered_trades = []
    for trade in all_trades:
        entry_time_str = trade.get("openTime", "")
        if entry_time_str:
            try:
                entry_time = parse_iso_time(entry_time_str)
                if start_of_day <= entry_time <= end_of_day:
                    filtered_trades.append(trade)
            except Exception:
                pass
    
    log.info(f"Found {len(filtered_trades)} trades for today (from {len(all_trades)} total)")
    
    if not filtered_trades:
        log.info("No trades to backfill for today")
        return {"success": True, "trades_backfilled": 0}
    
    # Reconstruct trades
    try:
        reconstructed_trades = reconstruct_trades_from_oanda_trades(filtered_trades)
    except Exception as e:
        log.warning(f"Failed to reconstruct trades: {e}")
        return {"success": False, "trades_backfilled": 0, "error": str(e)}
    
    if not reconstructed_trades:
        log.info("No trades could be reconstructed")
        return {"success": True, "trades_backfilled": 0}
    
    # Write trade journal entries
    log.info(f"Writing {len(reconstructed_trades)} trades to trade journal...")
    for trade in reconstructed_trades:
        try:
            write_trade_journal_entry(run_dir, trade, instrument, force=False)
        except Exception as e:
            log.warning(f"Failed to write trade {trade.get('oanda_trade_id')}: {e}")
    
    # Update index CSV
    try:
        update_trade_journal_index(run_dir, reconstructed_trades, instrument)
    except Exception as e:
        log.warning(f"Failed to update trade journal index: {e}")
    
    closed_count = sum(1 for t in reconstructed_trades if t.get("exit_time"))
    open_count = len(reconstructed_trades) - closed_count
    
    log.info(f"✅ Trade backfill complete: {len(reconstructed_trades)} trades ({closed_count} closed, {open_count} open)")
    
    return {
        "success": True,
        "trades_backfilled": len(reconstructed_trades),
        "closed": closed_count,
        "open": open_count,
    }


def start_sniper_live() -> int:
    """
    Start SNIPER live via shell script.
    
    Returns:
        Exit code from SNIPER process
    """
    script_path = Path("scripts/run_live_demo_sniper.sh")
    
    if not script_path.exists():
        log.error(f"SNIPER script not found: {script_path}")
        return 1
    
    log.info("=" * 60)
    log.info("Launching SNIPER via scripts/run_live_demo_sniper.sh")
    log.info("=" * 60)
    
    # Run SNIPER and stream output
    try:
        process = subprocess.Popen(
            ["bash", str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end="")
        
        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        log.info("\n[RESUME] SNIPER interrupted by user")
        if process:
            process.terminate()
        return 130
    except Exception as e:
        log.error(f"Failed to start SNIPER: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SNIPER Resume After Sleep"
    )
    parser.add_argument(
        "--skip-trade-backfill",
        action="store_true",
        default=False,
        help="Skip trade backfill step",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs/live_demo"),
        help="Directory containing SNIPER run directories",
    )
    
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("[RESUME] Starting resume_sniper_after_sleep for XAU_USD M5")
    log.info("=" * 60)
    
    # Load .env
    load_dotenv_if_present()
    
    # Step A: Candle backfill
    log.info("\n[RESUME] Step A: Candle backfill")
    log.info("-" * 60)
    
    try:
        candle_result = backfill_candles()
    except Exception as e:
        log.error(f"[RESUME] Candle backfill failed: {e}")
        log.error("[RESUME] Exiting - cannot start SNIPER without candle data")
        return 1
    
    if not candle_result.get("success", False):
        log.error(f"[RESUME] Candle backfill failed: {candle_result.get('error', 'Unknown error')}")
        log.error("[RESUME] Exiting - cannot start SNIPER without candle data")
        return 1
    
    new_candles = candle_result.get("new_candles", 0)
    from_time = candle_result.get("from_time")
    to_time = candle_result.get("to_time")
    
    if from_time and to_time:
        log.info(f"[RESUME] Candle backfill: {new_candles} new bars from {from_time} to {to_time}")
    else:
        log.info(f"[RESUME] Candle backfill: {new_candles} new bars")
    
    # Step B: Trade backfill (optional)
    trade_result = {"success": True, "trades_backfilled": 0}
    
    if not args.skip_trade_backfill:
        log.info("\n[RESUME] Step B: Trade backfill for today")
        log.info("-" * 60)
        
        # Find latest SNIPER run
        latest_run = find_latest_sniper_run(args.runs_dir)
        if not latest_run:
            log.warning(f"[RESUME] No SNIPER run directories found in {args.runs_dir}")
            log.warning("[RESUME] Skipping trade backfill")
        else:
            log.info(f"[RESUME] Using run directory: {latest_run}")
            try:
                trade_result = backfill_trades_for_today(latest_run, instrument="XAU_USD")
            except Exception as e:
                log.warning(f"[RESUME] Trade backfill failed: {e}")
                log.warning("[RESUME] Continuing anyway...")
                trade_result = {"success": False, "trades_backfilled": 0}
        
        if trade_result.get("success", False):
            trades_count = trade_result.get("trades_backfilled", 0)
            today_utc = datetime.now(timezone.utc).date()
            log.info(f"[RESUME] Trade backfill: {trades_count} trades updated for {today_utc}")
        else:
            log.warning("[RESUME] Trade backfill: Failed (continuing anyway)")
    else:
        log.info("\n[RESUME] Step B: Trade backfill skipped (--skip-trade-backfill)")
    
    # Step C: Start SNIPER
    log.info("\n[RESUME] Step C: Starting SNIPER live")
    log.info("-" * 60)
    log.info("[RESUME] Launching SNIPER via scripts/run_live_demo_sniper.sh")
    log.info("")
    
    exit_code = start_sniper_live()
    
    if exit_code == 0:
        log.info("\n[RESUME] SNIPER exited cleanly")
    else:
        log.warning(f"\n[RESUME] SNIPER exited with code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    exit(main())

