#!/usr/bin/env python3
"""V12 paper-trade runner — production live deployment.

Status: V12.4 CEMENTED 2026-05-16. Full V12Pipeline (XGB v5 + V10 multi-TF +
Entry-IQL v2 + V3 v9 multi-TF + Exit-IQL R_V12 + V12.4 Strategy-F overlay)
is loaded and wired via `pipeline.make_entry_decision()` / `.make_exit_decision()`.

Modus operandi:
    1. Wait for next M1 candle close (poll OANDA every 5-10s).
    2. Pre-trade spread check: skip if (ask-bid)/bid > spread_threshold_bps.
    3. Make V12.4 decision via V12Pipeline.
    4. If TAKE: place market order via OANDA (no SL/TP per V12 mandate).
    5. Catch MARKET_ORDER_REJECT_TRANSACTION; log reason + spread + time.
    6. If trade open: per-bar V3 v9 + V12.4 overlay → close order on EXIT_NOW.
    7. Log everything to daily journal for replay/comparison vs Phase 6 baseline.

Run (live demo on OANDA practice):
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_paper_runner.py --asia-skip --max-trades 5 --units 1

V12.4 lockdown: Strategy-F params are hardcoded in v12_exit_iql_live.py.
GX1_MFE_GIVEBACK_* env vars are rejected (RuntimeError) — no env-side ablation.
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
from gx1.monitoring.trade_journal import TradeJournal

LOG = logging.getLogger("v12_paper")
INSTRUMENT = "XAU_USD"
JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
COLLECTOR_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
CANONICAL_M1_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
TRADE_STATE_FILE = JOURNAL_DIR / "open_trade_state.json"  # LEGACY single-trade marker (migrated on startup)
TRADE_STATE_DIR = JOURNAL_DIR / "open_trades"             # one JSON file per open virtual trade
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
DEFAULT_MAX_TRADES = 1                 # max concurrent virtual trades held simultaneously
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

    # Parse response: OANDA returns {orderCreateTransaction, orderFillTransaction} on success.
    # On failure it returns one of: orderRejectTransaction (instrument/price errors) or
    # orderCancelTransaction (e.g. INSUFFICIENT_MARGIN cancels post-creation).
    if "orderRejectTransaction" in response or "orderCancelTransaction" in response \
            or "orderFillTransaction" not in response:
        reject = (response.get("orderRejectTransaction")
                  or response.get("orderCancelTransaction") or {})
        reason = (reject.get("rejectReason")
                  or reject.get("reason") or "UNKNOWN")
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


def attempt_close_trade(client: OandaClient, trade: TradeState) -> dict[str, Any]:
    """Close a specific virtual trade by its OANDA tradeID.

    Works in both NETTING and HEDGING account modes: OANDA's
    PUT /accounts/{id}/trades/{tradeID}/close endpoint closes only the units
    associated with that tradeID, not the full netted position.

    Falls back to a counter-direction market order if `trade.trade_id` is
    missing (e.g. virtual dry-run trade) — best-effort, may net incorrectly
    if multiple same-direction trades are open.
    """
    if trade.trade_id is None:
        LOG.warning(f"close_trade: trade has no OANDA tradeID — using counter-market fallback")
        close_side = "short" if trade.side == "long" else "long"
        return attempt_market_entry(client, close_side, units=trade.units)
    try:
        response = client.close_trade(trade.trade_id)
        fill = response.get("orderFillTransaction", {})
        LOG.info(f"CLOSED trade_id={trade.trade_id} units={fill.get('units')}  "
                  f"price={fill.get('price')}  pl={fill.get('pl')}")
        return {"status": "closed",
                 "trade_id": trade.trade_id,
                 "fill_price": float(fill.get("price", 0) or 0),
                 "realized_pl": float(fill.get("pl", 0) or 0),
                 "raw": response}
    except Exception as exc:
        LOG.error(f"OANDA close_trade({trade.trade_id}) failed: {exc}")
        return {"status": "api_error", "trade_id": trade.trade_id, "reason": str(exc)}


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
    p.add_argument("--max-trades", type=int, default=DEFAULT_MAX_TRADES,
                   help="Max concurrent virtual trades held simultaneously (default: 1)")
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
    # Suppress noisy "Using legacy trade_id" warnings (we intentionally use trade_id mode).
    logging.getLogger("gx1.monitoring.trade_journal").setLevel(logging.ERROR)

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
             f"units={args.units}  max_trades={args.max_trades}  dry_run={args.dry_run}")
    LOG.info(f"  feature_builder: 26-feature live snapshot (Phase A)")

    # Load full V12 pipeline (XGB v5 + V10 v3 + Entry-IQL v2 + Exit-IQL V12.1)
    LOG.info("loading V12Pipeline (XGB + V10 + Entry-IQL + Exit-IQL)...")
    pipeline = V12Pipeline.load_default()
    LOG.info("✓ V12 entry+exit stacks loaded — runner is live-wired")

    # Structured trade journal — per-trade JSON + aggregate index CSV.
    # Captures: entry_snapshot (V10/XGB context), feature_context,
    # per-bar v12_bar_decisions (IQL+V3), execution_events (OANDA fills),
    # exit_summary (realized PnL, MFE/MAE peaks).
    journal = TradeJournal(
        run_dir=JOURNAL_DIR,
        run_tag=f"v12_paper_{args.journal_suffix or 'live'}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        header={"runner": "v12_paper_runner",
                "env": creds.env,
                "max_trades": args.max_trades,
                "units": args.units,
                "max_spread_bps": args.max_spread_bps,
                "asia_skip": args.asia_skip,
                "dry_run": args.dry_run},
    )
    LOG.info(f"  trade journal: {journal.journal_dir}")

    last_decision_minute = None
    consecutive_errors = 0
    last_stale_log_minute = None

    # Resume any open trades from disk (survives runner crash/restart).
    # Auto-migrates legacy single-file state into the per-trade directory.
    open_trades: list[TradeState] = TradeState.load_all(
        TRADE_STATE_DIR, legacy_single_file=TRADE_STATE_FILE,
    )
    if open_trades:
        for t in open_trades:
            LOG.info(f"resumed open trade {t.trade_id or '(no-id)'}: "
                      f"side={t.side} bars={t.bars_in_trade} "
                      f"entry_ts={t.entry_ts}  pnl={t.current_pnl_bps:+.1f} bps")
    else:
        LOG.info(f"no open trades in {TRADE_STATE_DIR} — starting fresh")

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
                "n_open_trades": len(open_trades),
                "max_trades": args.max_trades,
                "has_open_trade": len(open_trades) > 0,   # back-compat for analytics
            }

            # ── EXIT branch: iterate all open trades, evaluate each per M1 ──
            m1_close = (bid + ask) / 2.0   # mid as proxy for M1 close
            per_trade_records: list[dict[str, Any]] = []
            survivors: list[TradeState] = []
            for trade in open_trades:
                exit_decision = make_v12_exit_decision(
                    pipeline, trade, current_minute, bid, ask, m1_close,
                )
                record = {
                    "trade_id": trade.trade_id,
                    "side": trade.side,
                    "entry_ts": trade.entry_ts.isoformat(),
                    "bars_in_trade": trade.bars_in_trade,
                    "units": trade.units,
                    "pnl_bps": trade.current_pnl_bps,
                    "peak_bps": trade.cum_mfe_bps,
                    "mae_bps": trade.cum_mae_bps,
                    "exit_decision": exit_decision,
                }

                if exit_decision.get("action_id") == 1:   # EXIT_NOW
                    record["order_status"] = "EXIT_NOW"
                    write_trade_alert(
                        f"EXIT  trade_id={trade.trade_id}  side={trade.side}  "
                        f"bars={trade.bars_in_trade}  "
                        f"pnl={trade.current_pnl_bps:+.1f} bps  peak={trade.cum_mfe_bps:+.1f}  "
                        f"mae={trade.cum_mae_bps:+.1f}  source={exit_decision.get('decision_source','IQL_Q')}"
                    )
                    if args.dry_run:
                        LOG.info(f"[DRY] EXIT_NOW trade_id={trade.trade_id} after {trade.bars_in_trade} bars  "
                                  f"pnl={trade.current_pnl_bps:+.1f} bps  side={trade.side}")
                    else:
                        close_result = attempt_close_trade(client, trade)
                        record["close_order_details"] = close_result
                    trade.delete_state_file(TRADE_STATE_DIR)
                elif trade.bars_in_trade >= 1440:   # 24h hard cap
                    LOG.warning(f"24h cap reached for trade {trade.trade_id} — forced close")
                    record["order_status"] = "FORCED_CLOSE_24H"
                    if not args.dry_run:
                        attempt_close_trade(client, trade)
                    trade.delete_state_file(TRADE_STATE_DIR)
                else:
                    record["order_status"] = "HOLDING_TRADE"
                    trade.save(TRADE_STATE_DIR)
                    survivors.append(trade)

                # ── TradeJournal: per-bar V12 decision capture ──
                if trade.trade_id:
                    journal.log_v12_bar_decision(
                        trade_id=trade.trade_id,
                        timestamp=current_minute.isoformat(),
                        bars_in_trade=trade.bars_in_trade,
                        bid=bid, ask=ask,
                        current_pnl_bps=trade.current_pnl_bps,
                        cum_mfe_bps=trade.cum_mfe_bps,
                        cum_mae_bps=trade.cum_mae_bps,
                        bars_since_mfe_peak=trade.bars_since_mfe_peak,
                        atr_bps=trade.last_atr_bps,
                        iql_action=exit_decision.get("action", "?"),
                        iql_action_id=int(exit_decision.get("action_id", 0)),
                        iql_decision_source=exit_decision.get("decision_source", "?"),
                        iql_q_hold=exit_decision.get("q_hold"),
                        iql_q_exit=exit_decision.get("q_exit"),
                        iql_q_advantage=exit_decision.get("q_advantage"),
                        v3_should_exit_prob=exit_decision.get("v3_should_exit_prob"),
                        v3_max_prob_in_trade=trade.v3_max_prob_in_trade,
                        v3_consecutive_exits=trade.v3_consecutive_exits,
                        v3_total_exit_decisions=trade.v3_total_exit_decisions,
                        v3_signal_acceleration=trade.v3_signal_acceleration,
                    )
                    # On EXIT_NOW or 24h cap: also log exit_summary
                    if record.get("order_status") in ("EXIT_NOW", "FORCED_CLOSE_24H"):
                        close_result = record.get("close_order_details") or {}
                        exit_price = float(close_result.get("fill_price", 0.0) or
                                            (bid if trade.side == "long" else ask))
                        realized_pnl = float(close_result.get("realized_pl", 0.0))
                        journal.log_oanda_trade_update(
                            trade_id=trade.trade_id,
                            event_type="TRADE_CLOSED_OANDA",
                            oanda_trade_id=trade.trade_id,
                            price=exit_price, units=trade.units, pl=realized_pnl,
                        )
                        journal.log_exit_summary(
                            trade_id=trade.trade_id,
                            exit_time=current_minute.isoformat(),
                            exit_price=exit_price,
                            exit_bid=bid, exit_ask=ask,
                            exit_spread_bps=spread_bps,
                            exit_reason=record["order_status"],
                            realized_pnl_bps=float(trade.current_pnl_bps),
                            max_mfe_bps=float(trade.cum_mfe_bps),
                            max_mae_bps=float(trade.cum_mae_bps),
                            intratrade_drawdown_bps=float(trade.cum_mfe_bps - trade.current_pnl_bps),
                        )
                per_trade_records.append(record)
            open_trades = survivors
            event["open_trade_records"] = per_trade_records
            # Back-compat top-level fields when exactly one trade was active
            if len(per_trade_records) == 1:
                r0 = per_trade_records[0]
                event["v12_exit_decision"] = r0["exit_decision"]
                event["trade_open_ts"] = r0["entry_ts"]
                event["trade_side"] = r0["side"]
                event["trade_bars"] = r0["bars_in_trade"]
                event["trade_pnl_bps"] = r0["pnl_bps"]
                event["trade_peak_bps"] = r0["peak_bps"]
                event["trade_mae_bps"] = r0["mae_bps"]
                event["order_status"] = r0["order_status"]

            # ── ENTRY branch: if room for more trades, evaluate XGB→V10→Entry-IQL ──
            if len(open_trades) >= args.max_trades:
                event.setdefault("order_status", "AT_MAX_TRADES")
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            decision = make_v12_decision(pipeline, current_minute, bid, ask)
            event["v12_decision"] = decision

            if not allowed:
                # Gate blocked the order (spread/asia) — but log V12's intent for counterfactual analysis
                event["order_status"] = "BLOCKED_BY_GATE"
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
                    virtual_id = f"virtual_{current_minute.strftime('%Y%m%dT%H%M%S')}"
                    new_trade = TradeState.open(
                        entry_ts=pd.Timestamp(current_minute),
                        side=side, entry_bid=bid, entry_ask=ask,
                        v10_snapshot=decision.get("_v10_snapshot", {}),
                        trade_id=virtual_id, units=args.units,
                    )
                    new_trade.save(TRADE_STATE_DIR)
                    open_trades.append(new_trade)
                    LOG.info(f"[DRY] virtual trade opened  side={side}  id={virtual_id}")
                else:
                    order_result = attempt_market_entry(client, side, units=args.units)
                    event["order_status"] = order_result["status"]
                    event["order_details"] = order_result
                    if order_result.get("status") == "filled":
                        new_trade = TradeState.open(
                            entry_ts=pd.Timestamp(current_minute),
                            side=side, entry_bid=bid, entry_ask=ask,
                            v10_snapshot=decision.get("_v10_snapshot", {}),
                            trade_id=str(order_result.get("trade_id") or ""),
                            units=args.units,
                        )
                        new_trade.save(TRADE_STATE_DIR)
                        open_trades.append(new_trade)

                        # ── TradeJournal: log entry lifecycle ──
                        if new_trade.trade_id:
                            v10 = decision.get("_v10_snapshot") or {}
                            journal.log_order_submitted(
                                trade_id=new_trade.trade_id,
                                instrument=INSTRUMENT, side=side,
                                units=args.units, order_type="MARKET",
                                client_order_id=order_result.get("client_order_id"),
                                oanda_env=creds.env,
                            )
                            journal.log_order_filled(
                                trade_id=new_trade.trade_id,
                                oanda_trade_id=new_trade.trade_id,
                                fill_price=order_result.get("fill_price"),
                                fill_units=args.units,
                            )
                            journal.log_entry_snapshot(
                                trade_id=new_trade.trade_id,
                                entry_time=current_minute.isoformat(),
                                instrument=INSTRUMENT, side=side,
                                entry_price=float(order_result.get("fill_price") or
                                                  (ask if side == "long" else bid)),
                                entry_bid=bid, entry_ask=ask,
                                entry_spread_bps=spread_bps,
                                entry_model_version="V12.4",
                                entry_score={
                                    "q_take_long": float(decision.get("q_take_long", 0.0)),
                                    "q_take_short": float(decision.get("q_take_short", 0.0)),
                                    "q_skip": float(decision.get("q_skip", 0.0)),
                                    "advantage_over_skip": float(decision.get("advantage_over_skip", 0.0)),
                                    "advantage_over_skip_long": float(decision.get("advantage_over_skip_long", 0.0)),
                                    "advantage_over_skip_short": float(decision.get("advantage_over_skip_short", 0.0)),
                                    "v10_p_long": float(decision.get("v10_p_long", 0.0)),
                                    "v10_p_short": float(decision.get("v10_p_short", 0.0)),
                                    "v10_path_quality_pred": float(decision.get("v10_path_quality_pred", 0.0)),
                                    "v10_mfe_pred_at_entry": float(decision.get("v10_mfe_pred_at_entry", 0.0)),
                                    "v10_tradable_prob": float(decision.get("v10_tradable_prob", 0.0)),
                                    "v10_bad_path_prob": float(decision.get("v10_bad_path_prob", 0.0)),
                                    "decision_ts": str(decision.get("decision_ts", "")),
                                },
                                entry_filters_passed=["spread_ok", "v12_take"],
                                base_units=args.units, units=args.units,
                                atr_bps=v10.get("atr_bps"),
                                spread_bps=spread_bps,
                            )
                            journal.log_feature_context(
                                trade_id=new_trade.trade_id,
                                spread_pct=spread_bps / 10000.0,
                                candle_close=(bid + ask) / 2.0,
                            )
                            journal.log_oanda_trade_update(
                                trade_id=new_trade.trade_id,
                                event_type="TRADE_OPENED_OANDA",
                                oanda_trade_id=new_trade.trade_id,
                                price=float(order_result.get("fill_price") or ask),
                                units=args.units,
                            )

                        write_trade_alert(
                            f"OPEN  trade_id={new_trade.trade_id}  side={side}  "
                            f"entry={ask if side=='long' else bid:.2f}  "
                            f"spread={spread_bps:.1f}bps  units={args.units}  "
                            f"open_count={len(open_trades)}/{args.max_trades}  "
                            f"v10_p_long={decision.get('v10_p_long', 0):.3f}  "
                            f"q_take={decision.get('q_take_long' if side=='long' else 'q_take_short', 0):+.1f}  "
                            f"adv={decision.get('advantage_over_skip', 0):+.1f}bps"
                        )
                        LOG.info(f"opened trade  id={new_trade.trade_id}  side={side}  "
                                  f"entry={ask if side=='long' else bid}  "
                                  f"open_count={len(open_trades)}/{args.max_trades}  "
                                  f"v10_p_long={decision.get('v10_p_long', 0):.3f}")
                    elif order_result.get("status") in ("rejected", "api_error"):
                        # Reject without a trade_id — log via run-level JSONL so reject stream
                        # is preserved for triage even if no per-trade journal exists.
                        journal.log(
                            event_type="ORDER_REJECTED",
                            payload={
                                "ts_utc": current_minute.isoformat(),
                                "side": side,
                                "units": args.units,
                                "reason": order_result.get("reason"),
                                "client_order_id": order_result.get("client_order_id"),
                                "v12_action": decision.get("action"),
                                "advantage_over_skip": float(decision.get("advantage_over_skip", 0.0)),
                                "v10_p_long": float(decision.get("v10_p_long", 0.0)),
                                "bid": bid, "ask": ask,
                                "n_open_trades": len(open_trades),
                            },
                        )
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
