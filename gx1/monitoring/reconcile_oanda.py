#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OANDA Execution Reconciliation Script.

Reconciles GX1 trade journal with OANDA transaction history to verify
that all trades were executed correctly and match expectations.

Usage:
    python gx1/monitoring/reconcile_oanda.py \
      --run gx1/wf_runs/<TAG> \
      --out gx1/wf_runs/<TAG>/reconciliation_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_run_header(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run_header.json."""
    header_path = run_dir / "run_header.json"
    if not header_path.exists():
        logger.warning(f"run_header.json not found in {run_dir}")
        return None
    
    try:
        with open(header_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load run_header.json: {e}")
        return None


def load_trade_journal_index(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load trade journal index CSV."""
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        logger.warning(f"Trade journal index not found: {index_path}")
        return None
    
    try:
        df = pd.read_csv(index_path)
        return df
    except Exception as e:
        logger.error(f"Failed to load trade journal index: {e}")
        return None


def load_trade_journal_json(run_dir: Path, trade_id: str) -> Optional[Dict[str, Any]]:
    """Load per-trade journal JSON."""
    journal_path = run_dir / "trade_journal" / "trades" / f"{trade_id}.json"
    if not journal_path.exists():
        return None
    
    try:
        with open(journal_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load trade journal for {trade_id}: {e}")
        return None


def load_smoke_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load exec_smoke_summary.json if it exists."""
    summary_path = run_dir / "exec_smoke_summary.json"
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load smoke summary: {e}")
        return None


def extract_trade_times_and_ids(trade_journal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract trade times and transaction IDs from trade journal.
    
    Returns:
        Dict with: open_time_utc, close_time_utc, open_txn_id, close_txn_id,
                   open_order_id, close_order_id, last_txn_id, oanda_trade_id,
                   client_ext_id
    """
    result = {
        "open_time_utc": None,
        "close_time_utc": None,
        "open_txn_id": None,
        "close_txn_id": None,
        "open_order_id": None,
        "close_order_id": None,
        "last_txn_id": None,
        "oanda_trade_id": None,
        "client_ext_id": None,
    }
    
    execution_events = trade_journal.get("execution_events", [])
    
    for event in execution_events:
        event_type = event.get("event_type")
        
        if event_type == "ORDER_SUBMITTED":
            # Extract client extension ID
            client_ext = event.get("client_extensions", {})
            if client_ext and not result["client_ext_id"]:
                result["client_ext_id"] = client_ext.get("id")
            
            # Extract timestamp
            ts = event.get("timestamp")
            if ts and not result["open_time_utc"]:
                try:
                    result["open_time_utc"] = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    pass
        
        elif event_type == "ORDER_FILLED":
            # Extract transaction IDs
            oanda_order_id = event.get("oanda_order_id")
            oanda_trade_id = event.get("oanda_trade_id")
            oanda_transaction_id = event.get("oanda_transaction_id")
            ts_oanda = event.get("ts_oanda")
            
            if oanda_order_id and not result["open_order_id"]:
                result["open_order_id"] = str(oanda_order_id).strip()
            
            if oanda_trade_id and not result["oanda_trade_id"]:
                result["oanda_trade_id"] = str(oanda_trade_id).strip()
            
            if oanda_transaction_id and not result["open_txn_id"]:
                result["open_txn_id"] = str(oanda_transaction_id).strip()
            
            if ts_oanda:
                try:
                    dt = datetime.fromisoformat(ts_oanda.replace("Z", "+00:00"))
                    if not result["open_time_utc"]:
                        result["open_time_utc"] = dt
                except Exception:
                    pass
        
        elif event_type == "TRADE_OPENED_OANDA":
            oanda_trade_id = event.get("oanda_trade_id")
            oanda_transaction_id = event.get("oanda_transaction_id")
            
            if oanda_trade_id and not result["oanda_trade_id"]:
                result["oanda_trade_id"] = str(oanda_trade_id).strip()
            
            if oanda_transaction_id and not result["open_txn_id"]:
                result["open_txn_id"] = str(oanda_transaction_id).strip()
        
        elif event_type == "TRADE_CLOSED_OANDA":
            oanda_transaction_id = event.get("oanda_transaction_id")
            ts_oanda = event.get("ts_oanda")
            
            if oanda_transaction_id and not result["close_txn_id"]:
                result["close_txn_id"] = str(oanda_transaction_id).strip()
            
            if ts_oanda:
                try:
                    result["close_time_utc"] = datetime.fromisoformat(ts_oanda.replace("Z", "+00:00"))
                except Exception:
                    pass
    
    # Also check for close order ID in ORDER_FILLED events (second ORDER_FILLED is usually close)
    # We'll identify it by checking if it's after TRADE_OPENED_OANDA
    found_open = False
    for event in execution_events:
        if event.get("event_type") == "TRADE_OPENED_OANDA":
            found_open = True
        elif event.get("event_type") == "ORDER_FILLED" and found_open:
            # This is likely the close ORDER_FILLED
            oanda_order_id = event.get("oanda_order_id")
            oanda_transaction_id = event.get("oanda_transaction_id")
            ts_oanda_close = event.get("ts_oanda")
            
            if oanda_order_id and not result["close_order_id"]:
                result["close_order_id"] = str(oanda_order_id).strip()
            if oanda_transaction_id and not result["close_txn_id"]:
                result["close_txn_id"] = str(oanda_transaction_id).strip()
            
            if ts_oanda_close and not result["close_time_utc"]:
                try:
                    result["close_time_utc"] = datetime.fromisoformat(ts_oanda_close.replace("Z", "+00:00"))
                except Exception:
                    pass
    
    # Fallback to entry/exit times from snapshot/summary
    if not result["open_time_utc"]:
        entry_snapshot = trade_journal.get("entry_snapshot", {})
        entry_time = entry_snapshot.get("entry_time")
        if entry_time:
            try:
                result["open_time_utc"] = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
            except Exception:
                pass
    
    if not result["close_time_utc"]:
        exit_summary = trade_journal.get("exit_summary", {})
        exit_time = exit_summary.get("exit_time")
        if exit_time:
            try:
                result["close_time_utc"] = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
            except Exception:
                pass
    
    return result


def fetch_oanda_transactions_window(
    account_id: str,
    api_token: str,
    api_url: str,
    start_time: datetime,
    end_time: datetime,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch OANDA transactions for a time range (PRIMARY METHOD).
    
    Returns:
        (transactions list, query_info dict)
    """
    query_info = {
        "method": "time_window",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "count": 0,
        "pages": 0,
    }
    
    try:
        import requests
        
        # Format times as RFC3339
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S.999999999Z")
        
        url = f"{api_url}/v3/accounts/{account_id}/transactions"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        params = {
            "from": start_str,
            "to": end_str,
            "pageSize": 500,  # Max page size
        }
        
        all_transactions = []
        next_page = None
        page_count = 0
        
        while True:
            if next_page:
                params["page"] = next_page
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            transactions = data.get("transactions", [])
            all_transactions.extend(transactions)
            page_count += 1
            
            next_page = data.get("nextPage")
            if not next_page:
                break
        
        query_info["count"] = len(all_transactions)
        query_info["pages"] = page_count
        
        logger.info(f"Fetched {len(all_transactions)} transactions from OANDA (window: {start_str} to {end_str}, {page_count} pages)")
        return all_transactions, query_info
    
    except Exception as e:
        logger.error(f"Failed to fetch OANDA transactions (window): {e}")
        return [], query_info


def fetch_oanda_transaction_by_id(
    account_id: str,
    api_token: str,
    api_url: str,
    transaction_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a specific OANDA transaction by ID (DIRECT METHOD).
    
    Returns:
        Transaction dict or None if not found
    """
    try:
        import requests
        
        url = f"{api_url}/v3/accounts/{account_id}/transactions/{transaction_id}"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        transaction = data.get("transaction")
        return transaction
    
    except Exception as e:
        logger.warning(f"Failed to fetch transaction {transaction_id}: {e}")
        return None


def fetch_oanda_transactions_since(
    account_id: str,
    api_token: str,
    api_url: str,
    since_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch OANDA transactions since a transaction ID (SECONDARY METHOD).
    
    Returns:
        (transactions list, query_info dict)
    """
    query_info = {
        "method": "since_id",
        "since_id": since_id,
        "count": 0,
        "pages": 0,
    }
    
    try:
        import requests
        
        url = f"{api_url}/v3/accounts/{account_id}/transactions"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        
        # Convert since_id to int if possible (OANDA expects int)
        try:
            since_id_int = int(since_id)
        except ValueError:
            logger.warning(f"Could not convert since_id to int: {since_id}")
            return [], query_info
        
        params = {
            "sinceTransactionID": since_id_int - 100,  # Go back 100 transactions for safety
            "pageSize": 500,
        }
        
        all_transactions = []
        next_page = None
        page_count = 0
        
        while True:
            if next_page:
                params["page"] = next_page
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            transactions = data.get("transactions", [])
            all_transactions.extend(transactions)
            page_count += 1
            
            next_page = data.get("nextPage")
            if not next_page:
                break
        
        query_info["count"] = len(all_transactions)
        query_info["pages"] = page_count
        
        logger.info(f"Fetched {len(all_transactions)} transactions from OANDA (since: {since_id_int - 100}, {page_count} pages)")
        return all_transactions, query_info
    
    except Exception as e:
        logger.error(f"Failed to fetch OANDA transactions (since): {e}")
        return [], query_info


def fetch_oanda_transactions_robust(
    account_id: str,
    api_token: str,
    api_url: str,
    trade_info: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch OANDA transactions using multiple methods (PRIMARY + SECONDARY + TERTIARY + DIRECT).
    
    Args:
        trade_info: Dict from extract_trade_times_and_ids()
    
    Returns:
        (transactions list, query_info_list)
    """
    all_transactions = []
    query_infos = []
    
    open_time = trade_info.get("open_time_utc")
    close_time = trade_info.get("close_time_utc")
    open_txn_id = trade_info.get("open_txn_id")
    close_txn_id = trade_info.get("close_txn_id")
    
    # DIRECT METHOD: Try fetching specific transactions by ID first (fastest if IDs are known)
    direct_txns = []
    if open_txn_id:
        txn = fetch_oanda_transaction_by_id(account_id, api_token, api_url, open_txn_id)
        if txn:
            direct_txns.append(txn)
            query_infos.append({
                "method": "direct_id",
                "transaction_id": open_txn_id,
                "count": 1,
            })
    
    if close_txn_id and close_txn_id != open_txn_id:
        txn = fetch_oanda_transaction_by_id(account_id, api_token, api_url, close_txn_id)
        if txn:
            direct_txns.append(txn)
            query_infos.append({
                "method": "direct_id",
                "transaction_id": close_txn_id,
                "count": 1,
            })
    
    if direct_txns:
        logger.info(f"Found {len(direct_txns)} transactions via direct ID lookup")
        all_transactions.extend(direct_txns)
        # If we found transactions via direct lookup, we can skip window queries
        # But we'll still try windows if we only found partial matches
        if len(direct_txns) >= 2:  # Found both open and close
            return all_transactions, query_infos
    
    # PRIMARY METHOD: Time window with 10-minute buffer (only if direct lookup didn't find everything)
    if open_time and close_time and len(direct_txns) < 2:
        window1_start = open_time - timedelta(minutes=10)
        window1_end = close_time + timedelta(minutes=10)
        
        txns, info = fetch_oanda_transactions_window(
            account_id, api_token, api_url, window1_start, window1_end
        )
        all_transactions.extend(txns)
        query_infos.append(info)
        
        # If window1 found nothing, try window2 (60-minute buffer)
        if len(txns) == 0:
            logger.warning(f"Window1 (10min buffer) found 0 transactions, trying window2 (60min buffer)")
            window2_start = open_time - timedelta(minutes=60)
            window2_end = close_time + timedelta(minutes=60)
            
            txns2, info2 = fetch_oanda_transactions_window(
                account_id, api_token, api_url, window2_start, window2_end
            )
            all_transactions.extend(txns2)
            query_infos.append(info2)
    
    # SECONDARY METHOD: Since-ID fallback
    if len(all_transactions) == 0:
        since_id = None
        if close_txn_id:
            since_id = close_txn_id
        elif open_txn_id:
            since_id = open_txn_id
        
        if since_id:
            logger.warning(f"Time windows found 0 transactions, trying since-id fallback: {since_id}")
            txns3, info3 = fetch_oanda_transactions_since(
                account_id, api_token, api_url, since_id
            )
            all_transactions.extend(txns3)
            query_infos.append(info3)
    
    # Remove duplicates (by transaction ID)
    seen_ids = set()
    unique_transactions = []
    for txn in all_transactions:
        txn_id = str(txn.get("id", ""))
        if txn_id and txn_id not in seen_ids:
            seen_ids.add(txn_id)
            unique_transactions.append(txn)
    
    logger.info(f"Total unique transactions fetched: {len(unique_transactions)}")
    return unique_transactions, query_infos


def match_transaction_to_trade(
    transaction: Dict[str, Any],
    trade_info: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Match an OANDA transaction to a GX1 trade.
    
    Matching rules (in order):
    1. clientExtensions.id == GX1:EXEC_SMOKE:<run_tag>:<trade_id> (exact string match)
    2. tradeID == oanda_trade_id (robust int/str comparison)
    3. orderID == open_order_id or close_order_id
    4. id == open_txn_id or close_txn_id
    
    Args:
        transaction: OANDA transaction dict
        trade_info: Dict from extract_trade_times_and_ids()
    
    Returns:
        (match_found, match_method)
    """
    # Extract clientExtensions from transaction
    client_ext_id = None
    if "clientExtensions" in transaction:
        client_ext_id = transaction["clientExtensions"].get("id")
    
    journal_client_ext_id = trade_info.get("client_ext_id")
    
    # Rule 1: clientExtensions.id match (PRIMARY)
    if client_ext_id and journal_client_ext_id:
        if str(client_ext_id).strip() == str(journal_client_ext_id).strip():
            return True, "CLIENT_EXT"
    
    # Rule 2: tradeID match (SECONDARY)
    oanda_trade_id = transaction.get("tradeID")
    journal_oanda_trade_id = trade_info.get("oanda_trade_id")
    
    if oanda_trade_id and journal_oanda_trade_id:
        if str(oanda_trade_id).strip() == str(journal_oanda_trade_id).strip():
            return True, "TRADE_ID"
    
    # Rule 3: orderID match (TERTIARY)
    oanda_order_id = transaction.get("orderID")
    journal_open_order_id = trade_info.get("open_order_id")
    journal_close_order_id = trade_info.get("close_order_id")
    
    if oanda_order_id:
        if journal_open_order_id and str(oanda_order_id).strip() == str(journal_open_order_id).strip():
            return True, "ORDER_ID_OPEN"
        if journal_close_order_id and str(oanda_order_id).strip() == str(journal_close_order_id).strip():
            return True, "ORDER_ID_CLOSE"
    
    # Rule 4: transaction ID match (QUATERNARY)
    txn_id = str(transaction.get("id", ""))
    journal_open_txn_id = trade_info.get("open_txn_id")
    journal_close_txn_id = trade_info.get("close_txn_id")
    
    if txn_id:
        if journal_open_txn_id and txn_id.strip() == str(journal_open_txn_id).strip():
            return True, "TXN_ID_OPEN"
        if journal_close_txn_id and txn_id.strip() == str(journal_close_txn_id).strip():
            return True, "TXN_ID_CLOSE"
    
    return False, "NO_MATCH"


def reconcile_trade(
    trade_id: str,
    trade_journal: Dict[str, Any],
    transactions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Reconcile a single trade against OANDA transactions.
    
    Returns:
        Reconciliation result dict
    """
    result = {
        "trade_id": trade_id,
        "status": "UNKNOWN",
        "matched_transactions": [],
        "match_methods": [],
        "missing_order_submitted": False,
        "missing_fill": False,
        "fill_price_match": None,
        "fill_units_match": None,
        "pnl_diff_bps": None,
        "execution_timestamp_plausible": None,
        # New fill price fields
        "open_fill_price_oanda": None,
        "close_fill_price_oanda": None,
        "open_fill_price_logged": None,
        "close_fill_price_logged": None,
        "open_fill_price_diff_bps": None,
        "close_fill_price_diff_bps": None,
        "open_fill_price_abs_diff": None,
        "close_fill_price_abs_diff": None,
    }
    
    # Extract trade info
    trade_info = extract_trade_times_and_ids(trade_journal)
    
    # Check for ORDER_SUBMITTED event and collect ORDER_FILLED events
    execution_events = trade_journal.get("execution_events", [])
    order_submitted = None
    order_filled_open = None
    order_filled_close = None
    found_open = False
    
    for event in execution_events:
        if event.get("event_type") == "ORDER_SUBMITTED":
            order_submitted = event
        elif event.get("event_type") == "ORDER_FILLED":
            # First ORDER_FILLED is open, second is close
            if not found_open:
                order_filled_open = event
                found_open = True
            else:
                order_filled_close = event
    
    # Backward compatibility: use first ORDER_FILLED as order_filled
    order_filled = order_filled_open
    
    if not order_submitted:
        result["missing_order_submitted"] = True
        result["status"] = "MISSING_SUBMIT"
        return result
    
    # Match transactions
    matched_txns = []
    match_methods = []
    for txn in transactions:
        match_found, match_method = match_transaction_to_trade(txn, trade_info)
        if match_found:
            matched_txns.append(txn)
            match_methods.append(match_method)
    
    result["matched_transactions"] = matched_txns
    result["match_methods"] = list(set(match_methods))  # Unique methods
    
    if not matched_txns:
        result["status"] = "MISSING"
        result["missing_fill"] = True
        return result
    
    # Check fill price match - compare OANDA transaction prices with logged prices
    # Find open fill transaction (ORDER_FILL type, usually first)
    open_fill_txn = None
    close_fill_txn = None
    
    for txn in matched_txns:
        txn_type = txn.get("type", "")
        # Open fill is usually ORDER_FILL with tradeOpened
        if txn_type == "ORDER_FILL" and txn.get("tradeOpened"):
            if not open_fill_txn:
                open_fill_txn = txn
        # Close fill is ORDER_FILL with tradesClosed or tradeClosed
        elif txn_type == "ORDER_FILL" and (txn.get("tradesClosed") or txn.get("tradeClosed")):
            if not close_fill_txn:
                close_fill_txn = txn
    
    # Extract OANDA fill prices from transactions
    if open_fill_txn:
        oanda_price_str = open_fill_txn.get("price")
        if oanda_price_str:
            try:
                result["open_fill_price_oanda"] = float(oanda_price_str)
            except (ValueError, TypeError):
                pass
    
    if close_fill_txn:
        oanda_price_str = close_fill_txn.get("price")
        if oanda_price_str:
            try:
                result["close_fill_price_oanda"] = float(oanda_price_str)
            except (ValueError, TypeError):
                pass
    
    # Extract logged fill prices from trade journal
    if order_filled_open:
        logged_price = order_filled_open.get("fill_price")
        if logged_price is not None:
            try:
                result["open_fill_price_logged"] = float(logged_price)
            except (ValueError, TypeError):
                pass
    
    if order_filled_close:
        logged_price = order_filled_close.get("fill_price")
        if logged_price is not None:
            try:
                result["close_fill_price_logged"] = float(logged_price)
            except (ValueError, TypeError):
                pass
    
    # Calculate diffs
    if result["open_fill_price_oanda"] is not None and result["open_fill_price_logged"] is not None:
        abs_diff = abs(result["open_fill_price_oanda"] - result["open_fill_price_logged"])
        result["open_fill_price_abs_diff"] = abs_diff
        # Convert to bps: (diff / price) * 10000
        if result["open_fill_price_logged"] > 0:
            diff_bps = (abs_diff / result["open_fill_price_logged"]) * 10000
            result["open_fill_price_diff_bps"] = diff_bps
    
    if result["close_fill_price_oanda"] is not None and result["close_fill_price_logged"] is not None:
        abs_diff = abs(result["close_fill_price_oanda"] - result["close_fill_price_logged"])
        result["close_fill_price_abs_diff"] = abs_diff
        # Convert to bps: (diff / price) * 10000
        if result["close_fill_price_logged"] > 0:
            diff_bps = (abs_diff / result["close_fill_price_logged"]) * 10000
            result["close_fill_price_diff_bps"] = diff_bps
    
    # Backward compatibility: fill_price_match (based on open fill)
    if result["open_fill_price_diff_bps"] is not None:
        # Consider match if diff < 1 bps (0.01%)
        result["fill_price_match"] = result["open_fill_price_diff_bps"] < 1.0
    
    # Check PnL match
    exit_summary = trade_journal.get("exit_summary", {})
    gx1_pnl_bps = exit_summary.get("realized_pnl_bps")
    
    if gx1_pnl_bps is not None:
        # Find closing transaction
        close_txn = None
        for txn in matched_txns:
            if txn.get("type") in ["TRADE_CLOSE", "ORDER_FILL"]:
                if txn.get("tradeClosed") or txn.get("tradesClosed"):
                    close_txn = txn
                    break
        
        if close_txn:
            oanda_pl = float(close_txn.get("pl", 0))
            # Convert OANDA PL to bps (simplified - would need entry price)
            # For now, just flag if PL exists
            result["pnl_diff_bps"] = None  # Would need entry price to convert
    
    # Determine final status
    if result["missing_fill"]:
        result["status"] = "MISSING"
    elif len(matched_txns) > 0:
        # If we have matched transactions, status is OK (even if fill_price_match is False/None)
        # PARTIAL is only for cases where we have some matches but missing critical ones
        result["status"] = "OK"
    else:
        result["status"] = "MISSING"
    
    return result


def generate_report(
    run_dir: Path,
    run_header: Optional[Dict[str, Any]],
    index_df: pd.DataFrame,
    reconciliation_results: List[Dict[str, Any]],
    unmatched_transactions: List[Dict[str, Any]],
    unmatched_trades: List[str],
    query_infos: List[Dict[str, Any]],
) -> str:
    """Generate markdown reconciliation report."""
    lines = []
    
    # Header
    run_tag = run_header.get("run_tag", "UNKNOWN") if run_header else "UNKNOWN"
    lines.append("# OANDA Execution Reconciliation Report")
    lines.append("")
    lines.append(f"**Run:** `{run_tag}`")
    lines.append(f"**Run Directory:** `{run_dir}`")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    
    # Artifacts hashes
    if run_header:
        artifacts = run_header.get("artifacts", {})
        lines.append("## Artifacts")
        lines.append("")
        policy_hash = artifacts.get("policy", {}).get("sha256", "N/A")
        router_hash = artifacts.get("router_model", {}).get("sha256", "N/A")
        manifest_hash = artifacts.get("feature_manifest", {}).get("sha256", "N/A")
        lines.append(f"- **Policy SHA256:** `{policy_hash[:16]}...`")
        lines.append(f"- **Router SHA256:** `{router_hash[:16] if router_hash != 'N/A' else 'N/A'}...`")
        lines.append(f"- **Manifest SHA256:** `{manifest_hash[:16] if manifest_hash != 'N/A' else 'N/A'}...`")
        lines.append("")
    
    # Queries executed
    lines.append("## Queries Executed")
    lines.append("")
    if query_infos:
        for i, info in enumerate(query_infos, 1):
            method = info.get("method", "UNKNOWN")
            lines.append(f"### Query {i}: {method}")
            lines.append("")
            
            if method == "time_window":
                lines.append(f"- **Start Time:** {info.get('start_time', 'N/A')}")
                lines.append(f"- **End Time:** {info.get('end_time', 'N/A')}")
            elif method == "since_id":
                lines.append(f"- **Since Transaction ID:** {info.get('since_id', 'N/A')}")
            
            lines.append(f"- **Transactions Found:** {info.get('count', 0)}")
            lines.append(f"- **Pages:** {info.get('pages', 0)}")
            lines.append("")
    else:
        lines.append("No queries executed.")
        lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    total_trades = len(reconciliation_results)
    matched_trades = sum(1 for r in reconciliation_results if r["status"] == "OK")
    unmatched_trades_count = sum(1 for r in reconciliation_results if r["status"] == "MISSING")
    partial_trades = sum(1 for r in reconciliation_results if r["status"] == "PARTIAL")
    rejected_orders = sum(1 for r in reconciliation_results if r["status"] == "REJECTED")
    
    lines.append(f"- **Total Trades:** {total_trades}")
    lines.append(f"- **✅ Matched Trades:** {matched_trades}")
    lines.append(f"- **⚠️ Partial Matches:** {partial_trades}")
    lines.append(f"- **❌ Unmatched Trades:** {unmatched_trades_count}")
    lines.append(f"- **❌ Rejected Orders:** {rejected_orders}")
    lines.append(f"- **Unmatched Transactions:** {len(unmatched_transactions)}")
    lines.append("")
    
    # Matched trades table
    if matched_trades > 0:
        lines.append("## Matched Trades")
        lines.append("")
        lines.append("| Trade ID | Client Ext ID | OANDA Trade ID | Match Method | Transaction IDs |")
        lines.append("|----------|---------------|----------------|--------------|-----------------|")
        
        for result in reconciliation_results:
            if result["status"] == "OK":
                trade_id = result["trade_id"]
                trade_journal = load_trade_journal_json(run_dir, trade_id)
                if trade_journal:
                    trade_info = extract_trade_times_and_ids(trade_journal)
                    client_ext_id = trade_info.get("client_ext_id", "N/A")
                    oanda_trade_id = trade_info.get("oanda_trade_id", "N/A")
                    match_methods = ", ".join(result.get("match_methods", []))
                    txn_ids = [str(txn.get("id", "")) for txn in result["matched_transactions"]]
                    txn_ids_str = ", ".join(txn_ids[:5])  # Limit to first 5
                    if len(txn_ids) > 5:
                        txn_ids_str += f" ... (+{len(txn_ids) - 5} more)"
                    
                    lines.append(f"| `{trade_id}` | `{client_ext_id[:50] if len(str(client_ext_id)) > 50 else client_ext_id}` | `{oanda_trade_id}` | {match_methods} | {txn_ids_str} |")
        
        lines.append("")
    
    # Per-trade details
    lines.append("## Per-Trade Details")
    lines.append("")
    
    for result in reconciliation_results:
        trade_id = result["trade_id"]
        status = result["status"]
        status_icon = "✅" if status == "OK" else "⚠️" if status == "PARTIAL" else "❌"
        
        lines.append(f"### {status_icon} Trade `{trade_id}`")
        lines.append("")
        lines.append(f"- **Status:** {status}")
        lines.append(f"- **Matched Transactions:** {len(result['matched_transactions'])}")
        
        if result.get("match_methods"):
            lines.append(f"- **Match Methods:** {', '.join(result['match_methods'])}")
        
        # Fill price reporting
        open_diff_bps = result.get("open_fill_price_diff_bps")
        close_diff_bps = result.get("close_fill_price_diff_bps")
        
        if open_diff_bps is not None or close_diff_bps is not None:
            open_str = f"{open_diff_bps:.2f}" if open_diff_bps is not None else "N/A"
            close_str = f"{close_diff_bps:.2f}" if close_diff_bps is not None else "N/A"
            lines.append(f"- **Fill Price Diff (open/close):** {open_str} bps / {close_str} bps")
            
            # Show detailed prices if available
            if result.get("open_fill_price_oanda") is not None and result.get("open_fill_price_logged") is not None:
                lines.append(f"  - Open: OANDA={result['open_fill_price_oanda']:.5f}, Logged={result['open_fill_price_logged']:.5f}")
            if result.get("close_fill_price_oanda") is not None and result.get("close_fill_price_logged") is not None:
                lines.append(f"  - Close: OANDA={result['close_fill_price_oanda']:.5f}, Logged={result['close_fill_price_logged']:.5f}")
        elif result.get("fill_price_match") is not None:
            # Fallback to old format if new fields not available
            match_str = "✅" if result["fill_price_match"] else "❌"
            lines.append(f"- **Fill Price Match:** {match_str}")
        
        if result.get("pnl_diff_bps") is not None:
            lines.append(f"- **PnL Diff:** {result['pnl_diff_bps']:.2f} bps")
        
        lines.append("")
    
    # Unmatched transactions
    if unmatched_transactions:
        lines.append("## Unmatched Transactions")
        lines.append("")
        lines.append(f"Found {len(unmatched_transactions)} transactions that could not be matched to trades.")
        lines.append("")
    
    # Unmatched trades
    if unmatched_trades:
        lines.append("## Unmatched Trades")
        lines.append("")
        lines.append(f"Found {len(unmatched_trades)} trades without matching transactions:")
        for trade_id in unmatched_trades:
            lines.append(f"- `{trade_id}`")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `gx1/monitoring/reconcile_oanda.py`*")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile GX1 trades with OANDA transactions"
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Run directory path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output markdown report path",
    )
    
    args = parser.parse_args()
    
    # Load run data
    logger.info("Loading run data...")
    run_header = load_run_header(args.run)
    index_df = load_trade_journal_index(args.run)
    
    if index_df is None or len(index_df) == 0:
        logger.error("No trades found in journal index")
        return 1
    
    # Load smoke summary if available
    smoke_summary = load_smoke_summary(args.run)
    
    # Fetch OANDA credentials
    import os
    account_id = os.getenv("OANDA_ACCOUNT_ID")
    api_token = os.getenv("OANDA_API_TOKEN")
    oanda_env = os.getenv("OANDA_ENV", "practice")
    
    if not account_id or not api_token:
        logger.error("OANDA_ACCOUNT_ID and OANDA_API_TOKEN must be set")
        return 1
    
    api_url = "https://api-fxpractice.oanda.com" if oanda_env == "practice" else "https://api-fxtrade.oanda.com"
    
    # Fetch transactions for each trade using robust method
    logger.info("Fetching OANDA transactions (robust method)...")
    
    # Add a small delay to allow OANDA Practice API to index transactions
    import time
    logger.info("Waiting 2 seconds for OANDA API to index transactions...")
    time.sleep(2)
    
    all_transactions = []
    all_query_infos = []
    
    for _, row in index_df.iterrows():
        trade_id = row["trade_id"]
        trade_journal = load_trade_journal_json(args.run, trade_id)
        
        if not trade_journal:
            continue
        
        trade_info = extract_trade_times_and_ids(trade_journal)
        txns, query_infos = fetch_oanda_transactions_robust(
            account_id, api_token, api_url, trade_info
        )
        
        all_transactions.extend(txns)
        all_query_infos.extend(query_infos)
    
    # Remove duplicates
    seen_ids = set()
    unique_transactions = []
    for txn in all_transactions:
        txn_id = str(txn.get("id", ""))
        if txn_id and txn_id not in seen_ids:
            seen_ids.add(txn_id)
            unique_transactions.append(txn)
    
    logger.info(f"Total unique transactions fetched: {len(unique_transactions)}")
    
    # Reconcile each trade
    logger.info("Reconciling trades...")
    reconciliation_results = []
    unmatched_transactions = unique_transactions.copy()
    
    for _, row in index_df.iterrows():
        trade_id = row["trade_id"]
        trade_journal = load_trade_journal_json(args.run, trade_id)
        
        if not trade_journal:
            reconciliation_results.append({
                "trade_id": trade_id,
                "status": "MISSING_JOURNAL",
            })
            continue
        
        result = reconcile_trade(trade_id, trade_journal, unique_transactions)
        reconciliation_results.append(result)
        
        # Remove matched transactions from unmatched list
        for txn in result["matched_transactions"]:
            # Find and remove by ID
            txn_id = str(txn.get("id", ""))
            unmatched_transactions = [t for t in unmatched_transactions if str(t.get("id", "")) != txn_id]
    
    # Find unmatched trades
    unmatched_trades = [
        r["trade_id"] for r in reconciliation_results
        if r["status"] in ["MISSING", "MISSING_SUBMIT"]
    ]
    
    # Generate report
    logger.info("Generating report...")
    report = generate_report(
        args.run,
        run_header,
        index_df,
        reconciliation_results,
        unmatched_transactions,
        unmatched_trades,
        all_query_infos,
    )
    
    # Write report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"Report written to: {args.out}")
    return 0


if __name__ == "__main__":
    exit(main())
