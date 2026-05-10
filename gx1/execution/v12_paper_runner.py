#!/usr/bin/env python3
"""V12 paper-trade runner — production template for live deployment.

Status: SKELETON / DESIGN TEMPLATE. The V12 decision-stack call is stubbed
(`make_v12_decision`) and must be wired in once shadow-replay validates the
feature pipeline on live data (Mon-Tue). Pre-trade spread check, reject
handling, and reject-logging are production-ready.

Modus operandi:
    1. Wait for next M1 candle close (poll OANDA every 5-10s).
    2. Pre-trade spread check: skip if (ask-bid)/bid > spread_threshold_bps.
    3. Make V12 decision (stub for now).
    4. If TAKE: place market order via OANDA (no SL/TP per V12 mandate).
    5. Catch MARKET_ORDER_REJECT_TRANSACTION; log reason + spread + time.
    6. If trade open: per-bar V3 v8 + ExitDeciderV12 → close order on EXIT_NOW.
    7. Log everything to daily journal for replay/comparison vs Phase 6 baseline.

Run (after wiring V12 decision):
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_paper_runner.py [--asia-skip] [--max-spread-bps N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

LOG = logging.getLogger("v12_paper")
INSTRUMENT = "XAU_USD"
JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")

# Pre-trade gates
DEFAULT_MAX_SPREAD_BPS = 10.0          # skip if spread > this
DEFAULT_DEFAULT_UNITS = 1              # smallest position size for paper-trade
DEFAULT_POLL_SECONDS = 10              # how often to check for new M1 close
ASIA_HOURS_UTC = range(0, 7)           # 00:00-07:00 UTC = Asia session


# ── Pre-trade spread + session checks ─────────────────────────────────────


def get_current_spread_bps(client: OandaClient) -> tuple[float, float, float]:
    """Returns (spread_bps, bid, ask). Raises on API error."""
    pricing = client.get_pricing([INSTRUMENT])
    quote = pricing["prices"][0]
    bid = float(quote["bids"][0]["price"])
    ask = float(quote["asks"][0]["price"])
    if bid <= 0:
        raise ValueError(f"Invalid bid: {bid}")
    spread_bps = (ask - bid) / bid * 10000.0
    return spread_bps, bid, ask


def can_trade_now(spread_bps: float, *, max_spread_bps: float, asia_skip: bool,
                   now_utc: datetime) -> tuple[bool, str]:
    """Pre-trade gating. Returns (allowed, reason)."""
    if asia_skip and now_utc.hour in ASIA_HOURS_UTC:
        return False, f"asia_session_skip (hour={now_utc.hour})"
    if spread_bps > max_spread_bps:
        return False, f"spread_too_wide ({spread_bps:.1f} > {max_spread_bps})"
    return True, "ok"


# ── V12 decision stub (TO BE WIRED IN) ────────────────────────────────────


def make_v12_decision(features_snapshot: dict[str, Any]) -> dict[str, Any]:
    """STUB — replace with real V12 stack call.

    Real implementation (Mon-Tue task):
      - Build candidate features from rolling M1+M5 window
      - Run XGB → V10 → Entry-IQL via existing adapters
      - Returns full state-vector for counterfactual replay:
        {
          "action": "SKIP"|"TAKE_LONG_NOW"|"TAKE_SHORT_NOW",
          "q_skip": float,
          "q_take_long": float,
          "q_take_short": float,
          "advantage_over_skip_long": float,
          "advantage_over_skip_short": float,
          "v10_path_quality_pred": float,
          "v10_mfe_first_n_pred": float,
          "v10_p_long": float, "v10_p_short": float,
          "xgb_signal7": list[float],
        }

    For wiring instructions see project_gx1_v12_deploy_runtime.md.
    """
    return {"action": "SKIP", "stub": True,
             "q_skip": 0.0, "q_take_long": 0.0, "q_take_short": 0.0,
             "advantage_over_skip_long": 0.0, "advantage_over_skip_short": 0.0}


def make_v12_exit_decision(bar_state: dict[str, Any]) -> dict[str, Any]:
    """STUB — replace with ExitDeciderV12Adapter call.

    Real implementation:
        from gx1.runtime.exit_decider_v12_adapter import ExitDeciderV12Adapter
        decider = ExitDeciderV12Adapter.load(<V12_1_1_LOCK>, variant="R_V12_1", fold_id="FOLD_1",
                                              v3_override_threshold=None)  # drop V3 fail-safe
        rec = decider.decide(bar_state)
        return {"action_id": rec.action_id_v1, "decision_source": rec.decision_source_v1}
    """
    return {"action_id": 0, "stub": True}


# ── Order execution + reject handling ────────────────────────────────────


def attempt_market_entry(client: OandaClient, side: str,
                         units: int = DEFAULT_DEFAULT_UNITS) -> dict[str, Any]:
    """Submit market order. Returns dict with status + reason if rejected.

    OANDA returns MARKET_ORDER_REJECT_TRANSACTION on rejection. Common reasons:
      MARKET_HALTED, INSUFFICIENT_LIQUIDITY, INSTRUMENT_HALTED,
      ACCOUNT_NOT_TRADEABLE, MARGIN_RATE_INVALID, PRICE_PRECISION_EXCEEDED.
    """
    if side not in ("long", "short"):
        return {"status": "skipped", "reason": f"invalid_side {side}"}
    signed_units = abs(units) if side == "long" else -abs(units)
    client_order_id = f"v12_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{side}"

    try:
        response = client.create_market_order(
            INSTRUMENT, units=signed_units, client_order_id=client_order_id,
        )
    except Exception as exc:
        LOG.error(f"OANDA order call failed: {exc}")
        return {"status": "api_error", "reason": str(exc)}

    # Parse response: OANDA returns {orderCreateTransaction, orderFillTransaction OR orderRejectTransaction, ...}
    if "orderRejectTransaction" in response or "orderFillTransaction" not in response:
        reject = response.get("orderRejectTransaction", {})
        reason = reject.get("rejectReason", "UNKNOWN")
        LOG.warning(f"REJECTED side={side} reason={reason}  cid={client_order_id}")
        return {"status": "rejected", "reason": reason, "client_order_id": client_order_id,
                 "raw": response}

    fill = response["orderFillTransaction"]
    LOG.info(f"FILLED side={side} units={signed_units}  price={fill.get('price')}  trade_id={fill.get('tradeOpened', {}).get('tradeID')}")
    return {"status": "filled",
             "fill_price": float(fill.get("price", 0)),
             "trade_id": fill.get("tradeOpened", {}).get("tradeID"),
             "client_order_id": client_order_id,
             "raw": response}


# ── Journal — all decisions + outcomes for daily replay ──────────────────


def log_journal_event(journal_path: Path, event: dict[str, Any]) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    event["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
    with journal_path.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def daily_journal_path(suffix: str = "") -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suf = f"_{suffix}" if suffix else ""
    return JOURNAL_DIR / f"v12_paper_journal_{today}{suf}.jsonl"


# ── Main loop ────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="V12 paper-trade runner (skeleton)")
    p.add_argument("--max-spread-bps", type=float, default=DEFAULT_MAX_SPREAD_BPS)
    p.add_argument("--asia-skip", action="store_true",
                   help="Skip trades in Asia session (00:00-07:00 UTC)")
    p.add_argument("--units", type=int, default=DEFAULT_DEFAULT_UNITS)
    p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't actually send orders — just log what would happen (shadow mode)")
    p.add_argument("--journal-suffix", type=str, default="",
                   help="Suffix for journal filename (e.g. 'live' or 'shadow') to allow parallel runners")
    p.add_argument("--allow-stub", action="store_true",
                   help="Permit running with stubbed V12 decision/exit logic. Required for shadow-mode "
                        "smoke-tests; refuses live orders unless --dry-run.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    creds = load_oanda_credentials()
    client = OandaClient(OandaClientConfig(api_key=creds.api_token,
                                            account_id=creds.account_id,
                                            env=creds.env))
    LOG.info(f"V12 paper runner starting  env={creds.env}  account={creds.account_id}")
    LOG.info(f"  max_spread_bps={args.max_spread_bps}  asia_skip={args.asia_skip}  "
             f"units={args.units}  dry_run={args.dry_run}")

    entry_stubbed = bool(make_v12_decision({}).get("stub"))
    exit_stubbed = bool(make_v12_exit_decision({}).get("stub"))
    if entry_stubbed or exit_stubbed:
        LOG.warning(f"⚠️  V12 stubbed (entry={entry_stubbed} exit={exit_stubbed}) — "
                    f"shadow-mode only. Wire real stack before live trading.")
        if not args.allow_stub:
            LOG.error("Refusing to start: stubbed V12 logic detected but --allow-stub not set.")
            return 2
        if not args.dry_run:
            LOG.error("Refusing to start: --dry-run is required when V12 is stubbed "
                      "(no exit loop wired → trades would never close).")
            return 2

    last_decision_minute = None
    consecutive_errors = 0

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            current_minute = now_utc.replace(second=0, microsecond=0)

            # Decide once per M1
            if last_decision_minute == current_minute:
                time.sleep(args.poll_seconds)
                continue

            spread_bps, bid, ask = get_current_spread_bps(client)
            allowed, reason = can_trade_now(
                spread_bps, max_spread_bps=args.max_spread_bps,
                asia_skip=args.asia_skip, now_utc=now_utc,
            )
            event = {
                "ts_utc": current_minute.isoformat(),
                "bid": bid, "ask": ask, "spread_bps": spread_bps,
                "allowed": allowed, "gate_reason": reason,
            }

            # === V12 decision is ALWAYS made (even when gate blocks) for counterfactual logging.
            # This lets us replay "would V12 have taken?" + 24h forward-outcome offline.
            decision = make_v12_decision({"ts": current_minute, "bid": bid, "ask": ask})
            event["v12_decision"] = decision

            if not allowed:
                # Gate blocked the order (spread/asia) — but log V12's intent for counterfactual analysis
                event["order_status"] = "BLOCKED_BY_GATE"
                # Flag high-conviction opportunities even when blocked
                adv_long = float(decision.get("advantage_over_skip_long", 0.0))
                adv_short = float(decision.get("advantage_over_skip_short", 0.0))
                event["high_conviction_blocked"] = (adv_long >= 50.0 or adv_short >= 50.0)
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            if decision["action"] in ("TAKE_LONG_NOW", "TAKE_SHORT_NOW"):
                side = "long" if decision["action"] == "TAKE_LONG_NOW" else "short"
                if args.dry_run:
                    event["order_status"] = "DRY_RUN"
                else:
                    order_result = attempt_market_entry(client, side, units=args.units)
                    event["order_status"] = order_result["status"]
                    event["order_details"] = order_result
            else:
                # SKIP — flag if V12 would have had >50 bps Q-advantage on either side
                # (these are "missed opportunities" we want to replay/learn from)
                adv_long = float(decision.get("advantage_over_skip_long", 0.0))
                adv_short = float(decision.get("advantage_over_skip_short", 0.0))
                event["order_status"] = "SKIP"
                event["high_conviction_skip"] = (adv_long >= 50.0 or adv_short >= 50.0)

            log_journal_event(daily_journal_path(args.journal_suffix), event)
            last_decision_minute = current_minute
            consecutive_errors = 0

        except Exception as exc:
            consecutive_errors += 1
            LOG.error(f"loop error (consec={consecutive_errors}): {exc}")
            backoff = min(args.poll_seconds * (2 ** min(consecutive_errors, 5)), 300)
            time.sleep(backoff)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
