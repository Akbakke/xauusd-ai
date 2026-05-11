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
from gx1.execution.v12_live_features import LiveFeatureBuilder
from gx1.execution.v12_pipeline import V12Pipeline
from gx1.execution.v12_trade_state import TradeState

LOG = logging.getLogger("v12_paper")
INSTRUMENT = "XAU_USD"
JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
COLLECTOR_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
CANONICAL_M1_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
TRADE_STATE_FILE = JOURNAL_DIR / "open_trade_state.json"  # persistent open-trade marker
TRADE_ALERTS_FILE = Path("/home/andre2/TRADES_ALERTS.txt")  # easy-to-tail alerts file


def write_trade_alert(line: str) -> None:
    """Append a one-line alert to TRADES_ALERTS.txt (for `tail -f` monitoring)."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with TRADE_ALERTS_FILE.open("a") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        pass   # alerts file is best-effort; never crash the runner over it

# Pre-trade gates
DEFAULT_MAX_SPREAD_BPS = 10.0          # skip if spread > this
DEFAULT_DEFAULT_UNITS = 1              # smallest position size for paper-trade
DEFAULT_POLL_SECONDS = 10              # how often to check for new M1 close
DEFAULT_QUOTE_MAX_AGE_SEC = 90.0       # treat quote as stale (market closed/halted) if older
ASIA_HOURS_UTC = range(0, 7)           # 00:00-07:00 UTC = Asia session


class StaleQuoteError(RuntimeError):
    """Raised when OANDA returns a quote older than max_age_sec.

    Happens when market is closed (e.g. weekend) — OANDA keeps serving the
    last close-of-week quote until Sunday Sydney open. Without this guard the
    paper-runner would log thousands of fake events with stale spreads.
    """
    def __init__(self, age_sec: float, quote_time: str):
        super().__init__(f"Quote is {age_sec:.0f}s old (quote_time={quote_time}) — market likely closed")
        self.age_sec = age_sec
        self.quote_time = quote_time


# ── Pre-trade spread + session checks ─────────────────────────────────────


def get_current_spread_bps(client: OandaClient,
                            *, max_age_sec: float = DEFAULT_QUOTE_MAX_AGE_SEC,
                            now_utc: datetime | None = None,
                            ) -> tuple[float, float, float]:
    """Returns (spread_bps, bid, ask). Raises StaleQuoteError if quote is older
    than max_age_sec (market closed). Raises ValueError on invalid bid."""
    pricing = client.get_pricing([INSTRUMENT])
    quote = pricing["prices"][0]
    quote_time_str = quote.get("time", "")
    if quote_time_str:
        quote_time = pd.to_datetime(quote_time_str, utc=True)
        now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
        age_sec = (now - quote_time).total_seconds()
        if age_sec > max_age_sec:
            raise StaleQuoteError(age_sec, quote_time_str)
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


# ── V12 decision (wired in sesjon 1-5) ────────────────────────────────────


def make_v12_decision(pipeline: V12Pipeline, now_minute: datetime,
                      bid: float, ask: float) -> dict[str, Any]:
    """Run the full V12 entry stack: XGB v5 → V10 v3 → Entry-IQL v2.

    Returns dict with action + Q-trio + V10 outputs + XGB outputs +
    decision timestamp. Pipeline maintains caches so per-M5 cold-build
    cost is amortized across all M1 ticks in that bucket.
    """
    return pipeline.make_entry_decision(pd.Timestamp(now_minute), bid, ask)


def make_v12_exit_decision(pipeline: V12Pipeline, trade: TradeState,
                            now_minute: datetime, bid: float, ask: float,
                            m1_close: float) -> dict[str, Any]:
    """Run Exit-IQL V12.1 for one M1 bar on an open trade.

    Advances trade-state (PnL/MFE/MAE) and queries the Exit-IQL adapter.
    """
    return pipeline.make_exit_decision(trade, pd.Timestamp(now_minute), bid, ask, m1_close)


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
    feature_builder = LiveFeatureBuilder(
        collector_dir=COLLECTOR_DIR,
        canonical_m1_dir=CANONICAL_M1_DIR,
    )
    LOG.info(f"V12 paper runner starting  env={creds.env}  account={creds.account_id}")
    LOG.info(f"  max_spread_bps={args.max_spread_bps}  asia_skip={args.asia_skip}  "
             f"units={args.units}  dry_run={args.dry_run}")
    LOG.info(f"  feature_builder: 26-feature live snapshot (Phase A)")

    # Load full V12 pipeline (XGB v5 + V10 v3 + Entry-IQL v2 + Exit-IQL V12.1)
    LOG.info("loading V12Pipeline (XGB + V10 + Entry-IQL + Exit-IQL)...")
    pipeline = V12Pipeline.load_default()
    LOG.info("✓ V12 entry+exit stacks loaded — runner is live-wired")

    last_decision_minute = None
    consecutive_errors = 0
    last_stale_log_minute = None

    # Resume any open trade from disk (survives runner crash/restart)
    open_trade: TradeState | None = TradeState.load(TRADE_STATE_FILE)
    if open_trade is not None:
        LOG.info(f"resumed open trade from {TRADE_STATE_FILE}: "
                  f"side={open_trade.side} bars={open_trade.bars_in_trade} "
                  f"entry_ts={open_trade.entry_ts}  pnl={open_trade.current_pnl_bps:+.1f} bps")
    else:
        LOG.info(f"no open trade state at {TRADE_STATE_FILE} — starting fresh")

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            current_minute = now_utc.replace(second=0, microsecond=0)

            # Decide once per M1
            if last_decision_minute == current_minute:
                time.sleep(args.poll_seconds)
                continue

            try:
                spread_bps, bid, ask = get_current_spread_bps(client, now_utc=now_utc)
            except StaleQuoteError as exc:
                # Market closed (weekend/holiday) — OANDA serves last close-of-week quote.
                # Skip silently; log once per hour to confirm daemon alive without polluting journal.
                if last_stale_log_minute is None or (current_minute - last_stale_log_minute).total_seconds() >= 3600:
                    LOG.info(f"stale quote ({exc.age_sec:.0f}s old, market closed) — pausing journal writes")
                    last_stale_log_minute = current_minute
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue
            allowed, reason = can_trade_now(
                spread_bps, max_spread_bps=args.max_spread_bps,
                asia_skip=args.asia_skip, now_utc=now_utc,
            )

            # Build live feature snapshot (Phase A: counterfactual context).
            # Failure is non-fatal — log warning, journal still has core fields.
            try:
                live_feats = feature_builder.compute(
                    pd.Timestamp(current_minute),
                    bid=bid, ask=ask, spread_bps=spread_bps,
                )
            except Exception as exc:
                LOG.warning(f"feature_builder failed at {current_minute}: {exc}")
                live_feats = {}

            event = {
                "ts_utc": current_minute.isoformat(),
                "bid": bid, "ask": ask, "spread_bps": spread_bps,
                "allowed": allowed, "gate_reason": reason,
                "features": live_feats,
                "has_open_trade": open_trade is not None,
            }

            # ── EXIT branch: open trade → run Exit-IQL V12.1 per M1 ──
            if open_trade is not None:
                m1_close = (bid + ask) / 2.0   # mid as proxy for M1 close until next collector tick
                exit_decision = make_v12_exit_decision(
                    pipeline, open_trade, current_minute, bid, ask, m1_close,
                )
                event["v12_exit_decision"] = exit_decision
                event["trade_open_ts"] = open_trade.entry_ts.isoformat()
                event["trade_side"] = open_trade.side
                event["trade_bars"] = open_trade.bars_in_trade
                event["trade_pnl_bps"] = open_trade.current_pnl_bps
                event["trade_peak_bps"] = open_trade.cum_mfe_bps
                event["trade_mae_bps"] = open_trade.cum_mae_bps

                if exit_decision.get("action_id") == 1:   # EXIT_NOW
                    event["order_status"] = "EXIT_NOW"
                    write_trade_alert(
                        f"EXIT  side={open_trade.side}  bars={open_trade.bars_in_trade}  "
                        f"pnl={open_trade.current_pnl_bps:+.1f} bps  peak={open_trade.cum_mfe_bps:+.1f}  "
                        f"mae={open_trade.cum_mae_bps:+.1f}  source={exit_decision.get('decision_source','IQL_Q')}"
                    )
                    if args.dry_run:
                        LOG.info(f"[DRY] EXIT_NOW after {open_trade.bars_in_trade} bars  "
                                  f"pnl={open_trade.current_pnl_bps:+.1f} bps  side={open_trade.side}")
                    else:
                        close_side = "short" if open_trade.side == "long" else "long"
                        close_result = attempt_market_entry(client, close_side, units=args.units)
                        event["close_order_details"] = close_result
                    open_trade = None
                    TRADE_STATE_FILE.unlink(missing_ok=True)   # delete persisted state
                else:
                    event["order_status"] = "HOLDING_TRADE"
                    open_trade.save(TRADE_STATE_FILE)          # persist running state
                # 24h hard cap fail-safe
                if open_trade is not None and open_trade.bars_in_trade >= 1440:
                    LOG.warning(f"24h cap reached — forced close")
                    event["order_status"] = "FORCED_CLOSE_24H"
                    if not args.dry_run:
                        close_side = "short" if open_trade.side == "long" else "long"
                        attempt_market_entry(client, close_side, units=args.units)
                    open_trade = None
                    TRADE_STATE_FILE.unlink(missing_ok=True)

                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            # ── ENTRY branch: no open trade → run XGB→V10→Entry-IQL ──
            decision = make_v12_decision(pipeline, current_minute, bid, ask)
            event["v12_decision"] = decision

            if not allowed:
                # Gate blocked the order (spread/asia) — but log V12's intent for counterfactual analysis
                event["order_status"] = "BLOCKED_BY_GATE"
                # Flag high-conviction opportunities even when blocked (direction-aware)
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
                    if order_result.get("status") == "filled":
                        # Open virtual trade with the entry quote (use current bid/ask
                        # snapshot since OANDA fill_price collapses bid/ask).
                        open_trade = TradeState.open(
                            entry_ts=pd.Timestamp(current_minute),
                            side=side, entry_bid=bid, entry_ask=ask,
                            v10_snapshot=decision.get("_v10_snapshot", {}),
                        )
                        open_trade.save(TRADE_STATE_FILE)   # persist immediately
                        write_trade_alert(
                            f"OPEN  side={side}  entry={ask if side=='long' else bid:.2f}  "
                            f"spread={spread_bps:.1f}bps  "
                            f"v10_p_long={decision.get('v10_p_long', 0):.3f}  "
                            f"q_take={decision.get('q_take_long' if side=='long' else 'q_take_short', 0):+.1f}  "
                            f"adv={decision.get('advantage_over_skip', 0):+.1f}bps  "
                            f"trade_id={order_result.get('trade_id','?')}"
                        )
                        LOG.info(f"opened trade  side={side}  entry={ask if side=='long' else bid}  "
                                  f"v10_p_long={decision.get('v10_p_long', 0):.3f}  "
                                  f"q_take={decision.get('q_take_long' if side=='long' else 'q_take_short', 0):+.1f}")
                # In dry-run mode, also open a virtual trade for shadow exit-loop testing.
                if args.dry_run:
                    open_trade = TradeState.open(
                        entry_ts=pd.Timestamp(current_minute),
                        side=side, entry_bid=bid, entry_ask=ask,
                        v10_snapshot=decision.get("_v10_snapshot", {}),
                    )
                    open_trade.save(TRADE_STATE_FILE)   # persist immediately
                    LOG.info(f"[DRY] virtual trade opened  side={side}")
            else:
                # SKIP — flag if V12 would have had >50 bps Q-advantage on either side
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
