#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug OANDA Transactions - Verify API Access

This script verifies that we can access OANDA transactions and compares
with what the backfill script sees. Used to debug why backfill sees 0 transactions.

Usage:
    python gx1/scripts/debug_oanda_transactions_today.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

from gx1.execution.oanda_client import OandaClient
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"[{title}]")
    print('=' * 60)


def print_account_summary(client: OandaClient, credentials: Any) -> None:
    """Print account summary information."""
    print_section("ACCOUNT_SUMMARY")
    
    print(f"Account ID: {credentials.account_id}")
    print(f"Environment: {credentials.env}")
    print(f"API URL: {credentials.api_url}")
    
    try:
        summary = client.get_account_summary()
        account = summary.get("account", {})
        
        print(f"\nAccount Details:")
        print(f"  Balance: {account.get('balance', 'N/A')}")
        print(f"  NAV: {account.get('NAV', 'N/A')}")
        print(f"  Open Trade Count: {account.get('openTradeCount', 'N/A')}")
        print(f"  Open Position Count: {account.get('openPositionCount', 'N/A')}")
        print(f"  Unrealized P/L: {account.get('unrealizedPL', 'N/A')}")
        print(f"  Realized P/L: {account.get('realizedPL', 'N/A')}")
        print(f"  P/L: {account.get('pl', 'N/A')}")
        
    except Exception as e:
        print(f"ERROR: Failed to get account summary: {e}")


def print_open_trades(client: OandaClient) -> None:
    """Print current open trades."""
    print_section("OPEN_TRADES")
    
    try:
        # Get open trades
        response = client.get_trades(state="OPEN")
        trades = response.get("trades", [])
        
        print(f"Number of open trades: {len(trades)}")
        
        if trades:
            print("\nOpen Trades:")
            for i, trade in enumerate(trades, 1):
                print(f"\n  Trade {i}:")
                print(f"    Trade ID: {trade.get('id', 'N/A')}")
                print(f"    Instrument: {trade.get('instrument', 'N/A')}")
                print(f"    Units: {trade.get('currentUnits', 'N/A')}")
                print(f"    Price: {trade.get('price', 'N/A')}")
                print(f"    Open Time: {trade.get('openTime', 'N/A')}")
                print(f"    Unrealized P/L: {trade.get('unrealizedPL', 'N/A')}")
        else:
            print("  (No open trades)")
            
    except Exception as e:
        print(f"ERROR: Failed to get open trades: {e}")


def fetch_last_transactions_direct(
    credentials: Any,
    count: int = 50,
) -> List[Dict[str, Any]]:
    """
    Fetch last N transactions directly from OANDA API without time filters.
    
    This bypasses fetch_oanda_transactions_window to test if the API
    itself returns transactions.
    """
    url = f"{credentials.api_url}/v3/accounts/{credentials.account_id}/transactions"
    headers = {
        "Authorization": f"Bearer {credentials.api_token}",
        "Content-Type": "application/json",
    }
    
    # Try multiple approaches:
    # 1. Without any params (should return recent transactions)
    # 2. With sinceTransactionID (if we can get a recent ID)
    # 3. With pageSize only
    
    all_transactions = []
    
    # Approach 1: Just pageSize
    params = {
        "pageSize": min(count, 500),  # Max page size is 500
    }
    
    next_page = None
    page_count = 0
    
    while len(all_transactions) < count and page_count < 10:  # Limit to 10 pages
        if next_page:
            params["page"] = next_page
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            transactions = data.get("transactions", [])
            all_transactions.extend(transactions)
            page_count += 1
            
            next_page = data.get("nextPage")
            if not next_page:
                break
            
            # Remove page param for next iteration
            if "page" in params:
                del params["page"]
                
        except Exception as e:
            print(f"ERROR: Failed to fetch transactions (pageSize method): {e}")
            print(f"Response: {response.text[:500] if 'response' in locals() else 'N/A'}")
            break
    
    # If still no transactions, try with sinceTransactionID (go back from a high number)
    if len(all_transactions) == 0:
        print("\n  Trying alternative: sinceTransactionID approach...")
        # Try with a high transaction ID (assuming recent transactions have high IDs)
        # Start from a very high number and go backwards
        params2 = {
            "sinceTransactionID": 999999999,  # Very high number
            "pageSize": min(count, 500),
        }
        
        try:
            response = requests.get(url, headers=headers, params=params2, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            transactions = data.get("transactions", [])
            if transactions:
                print(f"  Found {len(transactions)} transactions with sinceTransactionID method")
                all_transactions.extend(transactions)
        except Exception as e:
            print(f"  ERROR: sinceTransactionID method also failed: {e}")
    
    return all_transactions[:count]  # Return only requested count


def print_last_transactions(
    credentials: Any,
    count: int = 50,
) -> None:
    """Print last N transactions without time filters."""
    print_section("LAST_50_TRANSACTIONS")
    
    print("Attempting to fetch transactions...")
    transactions = fetch_last_transactions_direct(credentials, count=count)
    
    print(f"\nNumber of transactions returned: {len(transactions)}")
    
    if transactions:
        print(f"\nFirst {min(10, len(transactions))} transactions:")
        for i, txn in enumerate(transactions[:10], 1):
            print(f"\n  Transaction {i}:")
            print(f"    ID: {txn.get('id', 'N/A')}")
            print(f"    Time: {txn.get('time', 'N/A')}")
            print(f"    Type: {txn.get('type', 'N/A')}")
            print(f"    Instrument: {txn.get('instrument', 'N/A')}")
            print(f"    Units: {txn.get('units', 'N/A')}")
            print(f"    Trade ID: {txn.get('tradeID', 'N/A')}")
            print(f"    Order ID: {txn.get('orderID', 'N/A')}")
            
            # Check for nested trade info
            trade_opened = txn.get("tradeOpened")
            if trade_opened:
                print(f"    Trade Opened ID: {trade_opened.get('id', 'N/A')}")
            
            trade_closed = txn.get("tradeClosed")
            if trade_closed:
                print(f"    Trade Closed ID: {trade_closed.get('tradeID', 'N/A')}")
            
            trades_closed = txn.get("tradesClosed", [])
            if trades_closed:
                print(f"    Trades Closed: {len(trades_closed)}")
    else:
        print("  (No transactions found)")
        print("\n  NOTE: This is unusual given that we see 9 open trades.")
        print("  Possible reasons:")
        print("    - Transactions API may have a delay in indexing")
        print("    - Transactions may only be available for closed trades")
        print("    - API endpoint or parameters may need adjustment")


def print_transaction_type_counts(
    credentials: Any,
    count: int = 50,
) -> None:
    """Print counts by transaction type."""
    print_section("TX_TYPE_COUNTS")
    
    transactions = fetch_last_transactions_direct(credentials, count=count)
    
    if not transactions:
        print("No transactions to analyze")
        return
    
    # Count by type
    type_counts = Counter(txn.get("type", "UNKNOWN") for txn in transactions)
    
    print("Transaction type counts:")
    for txn_type, count in type_counts.most_common():
        print(f"  {txn_type}: {count}")
    
    # Count XAU_USD transactions
    xau_transactions = [
        txn for txn in transactions
        if txn.get("instrument") == "XAU_USD"
    ]
    print(f"\nXAU_USD transactions: {len(xau_transactions)}")
    
    # Count today's transactions (2025-12-26)
    today_str = "2025-12-26"
    today_transactions = [
        txn for txn in transactions
        if txn.get("time", "").startswith(today_str)
    ]
    print(f"Today's transactions (2025-12-26): {len(today_transactions)}")
    
    # Show transaction types for today
    if today_transactions:
        today_types = Counter(txn.get("type", "UNKNOWN") for txn in today_transactions)
        print("\nToday's transaction types:")
        for txn_type, count in today_types.most_common():
            print(f"  {txn_type}: {count}")


def compare_with_backfill(credentials: Any) -> None:
    """Compare what we see here vs what backfill would see."""
    print_section("BACKFILL_COMPARISON")
    
    # Simulate what backfill does: fetch with time window
    from gx1.monitoring.reconcile_oanda import fetch_oanda_transactions_window
    from datetime import datetime, timedelta
    
    # Use today's date range
    today = datetime(2025, 12, 26, tzinfo=timezone.utc)
    from_time = today - timedelta(days=1)
    to_time = today + timedelta(days=1)
    
    print(f"Backfill time window: {from_time.isoformat()} to {to_time.isoformat()}")
    
    transactions, query_info = fetch_oanda_transactions_window(
        account_id=credentials.account_id,
        api_token=credentials.api_token,
        api_url=credentials.api_url,
        start_time=from_time,
        end_time=to_time,
    )
    
    print(f"\nBackfill method returned: {len(transactions)} transactions")
    print(f"Query info: {json.dumps(query_info, indent=2)}")
    
    # Compare
    direct_transactions = fetch_last_transactions_direct(credentials, count=100)
    print(f"\nDirect API call (no time filter): {len(direct_transactions)} transactions")
    
    if len(direct_transactions) > 0 and len(transactions) == 0:
        print("\n⚠️  ISSUE DETECTED:")
        print("   Direct API call returns transactions, but backfill method returns 0.")
        print("   This suggests the issue is in fetch_oanda_transactions_window:")
        print("   - Time window parameters may be incorrect")
        print("   - Timezone mismatch")
        print("   - API endpoint/parameter differences")
    elif len(direct_transactions) == 0:
        print("\n⚠️  NO TRANSACTIONS FOUND:")
        print("   Even direct API call returns 0 transactions.")
        print("   However, we DO see 9 open trades via get_trades() API.")
        print("\n   HYPOTHESIS:")
        print("   - OANDA Practice API may only return transactions for CLOSED trades")
        print("   - Open trades may not have transactions yet (they're still active)")
        print("   - Transactions might only appear when trades are closed")
        print("\n   RECOMMENDATION:")
        print("   - Try fetching transactions for a wider time range (e.g., last 30 days)")
        print("   - Or reconstruct trades from open trades + trade history instead")
        print("   - Check if there's a separate endpoint for trade history vs transactions")


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("OANDA Transactions Debug Script")
    print("=" * 60)
    
    # Load .env file
    load_dotenv_if_present()
    
    # Load credentials
    try:
        credentials = load_oanda_credentials(prod_baseline=False)
        print(f"\n✅ Loaded credentials: env={credentials.env}")
    except Exception as e:
        print(f"\n❌ Failed to load credentials: {e}")
        return 1
    
    # Initialize OANDA client
    try:
        from gx1.execution.oanda_client import OandaClientConfig, OandaClient
        
        config = OandaClientConfig(
            api_key=credentials.api_token,
            account_id=credentials.account_id,
            env=credentials.env,
        )
        client = OandaClient(config)
        print(f"✅ Initialized OANDA client")
    except Exception as e:
        print(f"\n❌ Failed to initialize OANDA client: {e}")
        return 1
    
    # Print account summary
    print_account_summary(client, credentials)
    
    # Print open trades
    print_open_trades(client)
    
    # Print last transactions (no time filter)
    print_last_transactions(credentials, count=50)
    
    # Print transaction type counts
    print_transaction_type_counts(credentials, count=50)
    
    # Compare with backfill method
    compare_with_backfill(credentials)
    
    print("\n" + "=" * 60)
    print("✅ Debug script complete")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

