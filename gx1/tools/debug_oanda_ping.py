"""
Quick connectivity check against the OANDA REST API.

Reads credentials from the environment, instantiates the shared OandaClient,
and prints responses for a few basic endpoints.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict

from gx1.execution.oanda_client import OandaClient, OandaAPIError
from gx1.utils.env_loader import load_dotenv_if_present

log = logging.getLogger(__name__)


def _print_json(title: str, payload: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    load_dotenv_if_present()
    log.info("debug_oanda_ping starting up; ensured .env is loaded.")

    try:
        client = OandaClient.from_env()
    except Exception as exc:
        print(f"Failed to initialise OandaClient: {exc}", file=sys.stderr)
        sys.exit(1)

    # Account summary
    try:
        summary = client.get_account_summary()
        _print_json("Account Summary", summary)
    except OandaAPIError as exc:
        print(f"Account summary request failed: {exc}", file=sys.stderr)

    # Instrument candles (XAU_USD M5)
    try:
        candles = client.get_candles("XAU_USD", granularity="M5", count=5, include_mid=True)
        print("\n=== XAU_USD M5 Candles (last 5) ===")
        print(candles.tail(5))
    except OandaAPIError as exc:
        print(f"Candles request failed: {exc}", file=sys.stderr)
    
    # Open trades
    try:
        open_trades = client.get_open_trades()
        trades = open_trades.get("trades", [])
        print(f"\n=== Open Trades ({len(trades)}) ===")
        if trades:
            for trade in trades:
                print(f"  Trade ID: {trade.get('id')}")
                print(f"  Instrument: {trade.get('instrument')}")
                print(f"  Units: {trade.get('currentUnits')}")
                print(f"  Price: {trade.get('price')}")
                print(f"  Open Time: {trade.get('openTime')}")
                print()
        else:
            print("  No open trades")
    except OandaAPIError as exc:
        print(f"Open trades request failed: {exc}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()

