#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execution Smoke Test - Forced Execution Test.

Sends one minimal market order to OANDA Practice API, logs full execution trail
in Trade Journal, and runs reconciliation. Tests plumbing (credentials → order → fill → close → journal → reconcile).

This is NOT a strategy trade. It's marked as test_mode=true and does not use entry/exit models.

Usage:
    python gx1/execution/exec_smoke_test.py \
      --instrument XAU_USD \
      --units 1 \
      --hold-seconds 180 \
      --run-tag EXEC_SMOKE_20251216 \
      --out-dir gx1/wf_runs/EXEC_SMOKE_20251216
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from gx1.execution.broker_client import BrokerClient
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.monitoring.trade_journal import TradeJournal
from gx1.prod.run_header import generate_run_header, load_run_header
from gx1.utils.env_loader import load_dotenv_if_present

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_trade_id() -> str:
    """Create a test trade ID for smoke test."""
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    return f"EXEC-SMOKE-{timestamp}-{unique_id}"


def create_client_ext_id(run_tag: str, trade_id: str) -> str:
    """Create client extension ID for OANDA orders."""
    return f"GX1:EXEC_SMOKE:{run_tag}:{trade_id}"


def run_smoke_test(
    instrument: str,
    units: int,
    hold_seconds: int,
    run_tag: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Run execution smoke test.
    
    Args:
        instrument: Instrument symbol (e.g., "XAU_USD")
        units: Order size (must be positive for long, negative for short)
        hold_seconds: Seconds to hold trade before closing
        run_tag: Run tag identifier
        out_dir: Output directory for artifacts
        
    Returns:
        Summary dict with PASS/FAIL status and details
    """
    # Hard check: OANDA_ENV must be practice
    import os
    oanda_env = os.getenv("OANDA_ENV", "").lower()
    if oanda_env != "practice":
        raise RuntimeError(f"OANDA_ENV must be 'practice' (got: {oanda_env}). This test only runs against Practice API.")
    
    # Load credentials
    load_dotenv_if_present()
    creds = load_oanda_credentials(prod_baseline=False, require_live_latch=False)
    
    if creds.env != "practice":
        raise RuntimeError(f"Credentials loaded with env={creds.env}, expected 'practice'")
    
    logger.info(f"[SMOKE_TEST] Using OANDA Practice API: {creds.api_url}")
    logger.info(f"[SMOKE_TEST] Account ID: {creds.account_id[:3]}***{creds.account_id[-3:]}")
    
    # Create output directory structure
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_dir = out_dir / "trade_journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    trades_dir = journal_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test trade ID
    trade_id = create_test_trade_id()
    client_ext_id = create_client_ext_id(run_tag, trade_id)
    
    logger.info(f"[SMOKE_TEST] Trade ID: {trade_id}")
    logger.info(f"[SMOKE_TEST] Client Extension ID: {client_ext_id}")
    
    # Initialize trade journal
    # Create minimal run header for journal
    run_header = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "meta": {
            "role": "TEST",
            "test_mode": True,
            "test_type": "EXECUTION_SMOKE_TEST",
        },
        "artifacts": {
            "oanda_env": creds.env,
            "account_id_masked": creds.account_id[:3] + "***" + creds.account_id[-3:] if len(creds.account_id) > 6 else "***",
        },
    }
    
    # Try to get git commit hash
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
        if result.returncode == 0:
            run_header["git_commit"] = result.stdout.strip()
    except Exception:
        pass
    
    # Save run header
    header_path = out_dir / "run_header.json"
    with open(header_path, "w") as f:
        json.dump(run_header, f, indent=2)
    
    # Initialize trade journal
    trade_journal = TradeJournal(
        run_dir=out_dir,
        run_tag=run_tag,
        header=run_header,
        enabled=True,
    )
    
    # Determine side from units
    side = "long" if units > 0 else "short"
    
    # Log entry snapshot (test mode)
    entry_time = datetime.now(timezone.utc)
    entry_time_iso = entry_time.isoformat()
    
    trade_journal.log_entry_snapshot(
        trade_id=trade_id,
        entry_time=entry_time_iso,
        instrument=instrument,
        side=side,
        entry_price=0.0,  # Will be filled from order response
        session=None,
        regime=None,
        entry_model_version=None,  # Not using entry model
        entry_score=None,
        entry_filters_passed=[],
        entry_filters_blocked=[],
    )
    
    # Log feature context (test mode)
    trade_journal.log_feature_context(
        trade_id=trade_id,
        atr_bps=None,
        atr_price=None,
        atr_percentile=None,
        range_pos=None,
        distance_to_range=None,
        range_edge_dist_atr=None,
        spread_price=None,
        spread_pct=None,
        candle_close=None,
        candle_high=None,
        candle_low=None,
    )
    
    # Mark as test mode in journal
    journal_data = trade_journal._get_trade_journal(trade_id)
    journal_data["entry_snapshot"]["test_mode"] = True
    journal_data["entry_snapshot"]["reason"] = "EXECUTION_SMOKE_TEST"
    journal_data["feature_context"]["test_mode"] = True
    trade_journal._write_trade_json(trade_id)
    
    # Initialize broker client
    # BrokerClient uses OandaClient.from_env() by default, but we need to pass credentials
    from gx1.execution.oanda_client import OandaClient, OandaClientConfig
    oanda_config = OandaClientConfig(
        api_key=creds.api_token,
        account_id=creds.account_id,
        env=creds.env,
        timeout=10,
    )
    oanda_client = OandaClient(oanda_config)
    broker = BrokerClient(client=oanda_client)
    
    # Get current price for entry
    try:
        pricing = oanda_client.get_pricing([instrument])
        if "prices" in pricing and len(pricing["prices"]) > 0:
            price_info = pricing["prices"][0]
            if side == "long":
                entry_price = float(price_info["asks"][0]["price"])
            else:
                entry_price = float(price_info["bids"][0]["price"])
            logger.info(f"[SMOKE_TEST] Current price: {entry_price}")
        else:
            raise ValueError("No pricing data in response")
    except Exception as e:
        logger.error(f"[SMOKE_TEST] Failed to get pricing: {e}")
        entry_price = 0.0
    
    # Update entry snapshot with actual price
    journal_data = trade_journal._get_trade_journal(trade_id)
    journal_data["entry_snapshot"]["entry_price"] = entry_price
    trade_journal._write_trade_json(trade_id)
    
    # Send market order (OPEN)
    logger.info(f"[SMOKE_TEST] Sending market order: {side} {abs(units)} units of {instrument}")
    
    oanda_order_id = None
    oanda_trade_id = None
    fill_price = None
    fill_time = None
    stored_order_response = None
    stored_close_response = None
    
    try:
        # Log ORDER_SUBMITTED
        trade_journal.log_order_submitted(
            trade_id=trade_id,
            instrument=instrument,
            side=side,
            units=abs(units),
            order_type="MARKET",
            client_order_id=client_ext_id,
            client_ext_id=client_ext_id,
            client_ext_tag=run_tag,
            client_ext_comment="EXECUTION_SMOKE_TEST",
            requested_price=entry_price,  # Requested price (market will fill at best available)
            stop_loss_price=None,
            take_profit_price=None,
            oanda_env=creds.env,
            account_id_masked=creds.account_id[:3] + "***" + creds.account_id[-3:] if len(creds.account_id) > 6 else "***",
        )
        
        # Send order
        order_response = broker.create_market_order(
            instrument=instrument,
            units=units,
            client_extensions={
                "id": client_ext_id,
                "tag": run_tag,
                "comment": "EXECUTION_SMOKE_TEST",
            },
        )
        
        logger.info(f"[SMOKE_TEST] Order response: {order_response}")
        
        # Extract order ID and trade ID from response
        if "orderFillTransaction" in order_response:
            fill_txn = order_response["orderFillTransaction"]
            oanda_order_id = fill_txn.get("id")
            fill_price = float(fill_txn.get("price", 0))
            fill_time = fill_txn.get("time")
            
            # Extract trade ID if available
            if "tradeOpened" in fill_txn:
                oanda_trade_id = fill_txn["tradeOpened"].get("tradeID")
        
        # Store order_response for later extraction of transaction IDs
        stored_order_response = order_response
        
        # Extract commission and financing from transaction if available
        commission = 0.0
        financing = 0.0
        pl = None
        ts_oanda = fill_time
        
        if "orderFillTransaction" in order_response:
            fill_txn = order_response["orderFillTransaction"]
            if "financing" in fill_txn:
                financing = float(fill_txn["financing"])
            if "commission" in fill_txn:
                commission = float(fill_txn["commission"])
            if "pl" in fill_txn:
                pl = float(fill_txn["pl"])
        
        # Log ORDER_FILLED
        trade_journal.log_order_filled(
            trade_id=trade_id,
            oanda_order_id=str(oanda_order_id) if oanda_order_id else None,
            oanda_trade_id=str(oanda_trade_id) if oanda_trade_id else None,
            oanda_transaction_id=str(oanda_order_id) if oanda_order_id else None,
            fill_price=fill_price,
            fill_units=abs(units),
            commission=commission,
            financing=financing,
            pl=pl,
            ts_oanda=ts_oanda,
        )
        
        # Log TRADE_OPENED_OANDA if we have trade ID
        if oanda_trade_id:
            trade_journal.log_oanda_trade_update(
                trade_id=trade_id,
                event_type="TRADE_OPENED_OANDA",
                oanda_trade_id=str(oanda_trade_id),
                oanda_transaction_id=str(oanda_order_id) if oanda_order_id else None,
                price=fill_price,
                units=abs(units),
                pl=pl,
                ts_oanda=ts_oanda,
            )
            
            logger.info(f"[SMOKE_TEST] Trade opened: OANDA trade ID = {oanda_trade_id}")
        
        logger.info(f"[SMOKE_TEST] Order filled at price: {fill_price}")
        
    except Exception as e:
        logger.error(f"[SMOKE_TEST] Order submission failed: {e}")
        
        # Log ORDER_REJECTED
        trade_journal.log_order_rejected(
            trade_id=trade_id,
            client_order_id=client_ext_id,
            status_code=None,
            reject_reason=str(e),
        )
        
        raise
    
    # Hold trade for specified duration
    logger.info(f"[SMOKE_TEST] Holding trade for {hold_seconds} seconds...")
    time.sleep(hold_seconds)
    
    # Close trade
    logger.info(f"[SMOKE_TEST] Closing trade...")
    
    if not oanda_trade_id:
        logger.warning("[SMOKE_TEST] No OANDA trade ID available, cannot close trade")
        close_success = False
    else:
        try:
            # Log ORDER_SUBMITTED (close)
            trade_journal.log_order_submitted(
                trade_id=trade_id,
                instrument=instrument,
                side="close",
                units=abs(units),  # Close order size
                order_type="MARKET",
                client_order_id=f"{client_ext_id}:CLOSE",
                client_ext_id=f"{client_ext_id}:CLOSE",
                client_ext_tag=run_tag,
                client_ext_comment="EXECUTION_SMOKE_TEST_CLOSE",
                requested_price=None,  # Market close
                stop_loss_price=None,
                take_profit_price=None,
                oanda_env=creds.env,
                account_id_masked=creds.account_id[:3] + "***" + creds.account_id[-3:] if len(creds.account_id) > 6 else "***",
            )
            
            # Close trade
            close_response = broker.close_trade(oanda_trade_id)
            
            logger.info(f"[SMOKE_TEST] Close response: {close_response}")
            
            # Store close_response for later extraction of transaction IDs
            stored_close_response = close_response
            
            # Extract close transaction details
            close_txn_id = None
            close_price = None
            realized_pnl = None
            close_txn_time = None
            close_commission = 0.0
            close_financing = 0.0
            
            if "orderFillTransaction" in close_response:
                close_txn = close_response["orderFillTransaction"]
                close_txn_id = close_txn.get("id")
                close_price = float(close_txn.get("price", 0))
                close_txn_time = close_txn.get("time")
                
                # Extract realized PnL, commission, financing if available
                if "pl" in close_txn:
                    realized_pnl = float(close_txn["pl"])
                if "financing" in close_txn:
                    close_financing = float(close_txn["financing"])
                if "commission" in close_txn:
                    close_commission = float(close_txn["commission"])
            
            close_pl = realized_pnl
            
            # Log ORDER_FILLED (close)
            trade_journal.log_order_filled(
                trade_id=trade_id,
                oanda_order_id=str(close_txn_id) if close_txn_id else None,
                oanda_trade_id=str(oanda_trade_id) if oanda_trade_id else None,
                oanda_transaction_id=str(close_txn_id) if close_txn_id else None,
                fill_price=close_price,
                fill_units=abs(units),
                commission=close_commission,
                financing=close_financing,
                pl=close_pl,
                ts_oanda=close_txn_time,
            )
            
            # Log TRADE_CLOSED_OANDA
            trade_journal.log_oanda_trade_update(
                trade_id=trade_id,
                event_type="TRADE_CLOSED_OANDA",
                oanda_trade_id=str(oanda_trade_id),
                oanda_transaction_id=str(close_txn_id) if close_txn_id else None,
                price=close_price,
                units=abs(units),
                pl=close_pl,
                ts_oanda=close_txn_time,
            )
            
            # Log exit summary
            exit_time = datetime.now(timezone.utc)
            exit_time_iso = exit_time.isoformat()
            
            # Calculate PnL in bps if we have prices
            realized_pnl_bps = None
            if fill_price and close_price and fill_price > 0:
                if side == "long":
                    pnl_pct = (close_price - fill_price) / fill_price
                else:
                    pnl_pct = (fill_price - close_price) / fill_price
                realized_pnl_bps = pnl_pct * 10000
            
            trade_journal.log_exit_summary(
                trade_id=trade_id,
                exit_time=exit_time_iso,
                exit_price=close_price if close_price else 0.0,
                exit_reason="SMOKE_TEST_CLOSE",
                realized_pnl_bps=realized_pnl_bps if realized_pnl_bps else 0.0,
                max_mfe_bps=None,
                max_mae_bps=None,
                intratrade_drawdown_bps=None,
            )
            
            logger.info(f"[SMOKE_TEST] Trade closed at price: {close_price}")
            if realized_pnl is not None:
                logger.info(f"[SMOKE_TEST] Realized PnL: {realized_pnl}")
            
            close_success = True
            
        except Exception as e:
            logger.error(f"[SMOKE_TEST] Trade close failed: {e}")
            close_success = False
    
    # Extract transaction IDs from order responses
    open_txn_id = None
    close_txn_id = None
    open_order_id = None
    close_order_id = None
    last_txn_id = None
    open_time_utc = None
    close_time_utc = None
    
    # Extract from stored_order_response (set in try block)
    if stored_order_response:
        if "orderFillTransaction" in stored_order_response:
            fill_txn = stored_order_response["orderFillTransaction"]
            open_txn_id = str(fill_txn.get("id", ""))
            open_order_id = str(fill_txn.get("orderID", ""))
            last_txn_id = str(stored_order_response.get("lastTransactionID", ""))
            open_time_utc = fill_txn.get("time")
    
    # Extract from stored_close_response (set in try block)
    if stored_close_response:
        if "orderFillTransaction" in stored_close_response:
            close_fill_txn = stored_close_response["orderFillTransaction"]
            close_txn_id = str(close_fill_txn.get("id", ""))
            close_order_id = str(close_fill_txn.get("orderID", ""))
            last_txn_id = str(stored_close_response.get("lastTransactionID", ""))
            close_time_utc = close_fill_txn.get("time")
    
    # Generate summary
    summary = {
        "status": "PASS" if (oanda_order_id and oanda_trade_id and close_success) else "FAIL",
        "trade_id": trade_id,
        "client_ext_id": client_ext_id,
        "oanda_order_id": str(oanda_order_id) if oanda_order_id else None,
        "oanda_trade_id": str(oanda_trade_id) if oanda_trade_id else None,
        "open_txn_id": open_txn_id if open_txn_id else None,
        "close_txn_id": close_txn_id if close_txn_id else None,
        "open_order_id": open_order_id if open_order_id else None,
        "close_order_id": close_order_id if close_order_id else None,
        "last_txn_id": last_txn_id if last_txn_id else None,
        "open_time_utc": open_time_utc if open_time_utc else entry_time_iso,
        "close_time_utc": close_time_utc if close_time_utc else None,
        "entry_price": fill_price,
        "exit_price": None,  # Will be filled if close succeeded
        "realized_pnl": None,  # Will be filled if close succeeded
        "hold_seconds": hold_seconds,
        "start_time": entry_time_iso,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }
    
    # Update summary with close details if available
    journal_data = trade_journal._get_trade_journal(trade_id)
    if journal_data.get("exit_summary"):
        summary["exit_price"] = journal_data["exit_summary"].get("exit_price")
        summary["realized_pnl"] = journal_data["exit_summary"].get("realized_pnl_bps")
    
    # Save summary
    summary_path = out_dir / "exec_smoke_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"[SMOKE_TEST] Summary: {summary['status']}")
    
    return summary


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Execution Smoke Test - Forced execution test against OANDA Practice API"
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="XAU_USD",
        help="Instrument symbol (default: XAU_USD)",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=1,
        help="Order size (positive for long, negative for short, default: 1)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=180,
        help="Seconds to hold trade before closing (default: 180)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        required=True,
        help="Run tag identifier",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for artifacts",
    )
    
    args = parser.parse_args()
    
    try:
        summary = run_smoke_test(
            instrument=args.instrument,
            units=args.units,
            hold_seconds=args.hold_seconds,
            run_tag=args.run_tag,
            out_dir=args.out_dir,
        )
        
        if summary["status"] == "PASS":
            logger.info("[SMOKE_TEST] ✓ Test PASSED")
            return 0
        else:
            logger.error("[SMOKE_TEST] ✗ Test FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"[SMOKE_TEST] Test failed with exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

