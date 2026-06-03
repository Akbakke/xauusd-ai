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
        gx1/execution/v12_paper_runner.py --max-trades 5 --units 1

We trade year-round, all sessions (Asia included) — session is a learned
feature, not a hardcoded skip. Strategy-F overlay is ABLATABLE via
GX1_STRATEGY_F_ENABLED / GX1_MFE_GIVEBACK_* env vars (default = cemented values);
set GX1_STRATEGY_F_ENABLED=0 for the pure Exit-IQL policy.
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

# 2026-05-29: when env var GX1_PURE_PHASE6=1, every runtime wrapper filter
# (adaptive_min_adv, regime block, portfolio caps, low-confidence,
# TIME_OF_DAY_EXIT, spread cap) is bypassed so live runtime matches the
# Phase 6 OOT simulation 1:1. Used after the COSTFIX cement to verify live
# PnL tracks the +136 bps/take Phase 6 number.
PURE_PHASE6 = bool(int(os.environ.get("GX1_PURE_PHASE6", "0")))
if PURE_PHASE6:
    LOG.warning("[GX1_PURE_PHASE6] all runtime wrappers DISABLED — runner = Phase 6 OOT 1:1")

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
DEFAULT_MAX_SPREAD_BPS = 7.0           # 2026-05-20 tightening: was 10.0, but
                                        # news-spike spreads at 8-10 bps ate
                                        # entry edge. Backtest data had clean
                                        # OANDA M5 spreads typically 1-3 bps.
DEFAULT_DEFAULT_UNITS = 1              # smallest position size for paper-trade
DEFAULT_POLL_SECONDS = 10              # how often to check for new M1 close
DEFAULT_QUOTE_MAX_AGE_SEC = 90.0       # treat quote as stale (market closed/halted) if older
DEFAULT_MAX_TRADES = 1                 # max concurrent virtual trades held simultaneously


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


def can_trade_now(spread_bps: float, *, max_spread_bps: float,
                   now_utc: datetime) -> tuple[bool, str]:
    """Pre-trade gating — SPREAD SAFETY ONLY. We trade year-round, all sessions
    (incl. Asia): session is a learned feature the Entry-IQL policy decides on,
    not a hardcoded skip. `now_utc` kept for future safety gates.

    PURE_PHASE6: spread cap bypassed (Phase 6 OOT did not model spread filtering).
    """
    if PURE_PHASE6:
        return True, "ok_pure_phase6"
    if spread_bps > max_spread_bps:
        return False, f"spread_too_wide ({spread_bps:.1f} > {max_spread_bps})"
    return True, "ok"


# ── Execution safety invariant (2026-06-03) ───────────────────────────────


def evaluate_entry_safety(side: str, n_same_side: int, n_opposing: int,
                          hard_max_same_side: int) -> tuple[bool, str, int]:
    """Pure no-short-in-long + no-pile-up entry-safety decision.

    Always-on execution invariant, independent of PURE_PHASE6. The counts passed in
    MUST already be reconciled against broker truth by the caller. Returns
    (allowed, reason, detail). On violation the caller SKIPs the entry — it never
    flattens existing trades (that would deadlock with the per-trade M1 exit
    lifecycle). Opposing-side is checked first: never open a short while a long is
    open (or vice-versa); then the same-side hard cap (the -2000 06-02 pile-up fix).
    """
    if n_opposing > 0:
        return False, "blocked_opposing_side_open", n_opposing
    if n_same_side >= hard_max_same_side:
        return False, "blocked_hard_same_side_cap", n_same_side
    return True, "ok", 0


# ── V12 decision (wired in sesjon 1-5) ────────────────────────────────────


def make_v12_decision(pipeline: V12Pipeline, now_minute: datetime,
                      bid: float, ask: float,
                      open_trades: list | None = None) -> dict[str, Any]:
    """Run the full V12 entry stack: XGB v5 → V10 v3 → Entry-IQL v2.

    Returns dict with action + Q-trio + V10 outputs + XGB outputs +
    decision timestamp. Pipeline maintains caches so per-M5 cold-build
    cost is amortized across all M1 ticks in that bucket.

    `open_trades` is the runner's list of currently-open TradeState objects.
    Aggregated into portfolio_state and passed to Entry-IQL so model sees
    current portfolio context (#1 improvement 2026-05-21).
    """
    portfolio_state = None
    if open_trades:
        n_long = sum(1 for t in open_trades if t.side == "long")
        n_short = sum(1 for t in open_trades if t.side == "short")
        combined_pnl = sum(t.current_pnl_bps for t in open_trades)
        latest_entry = max(t.entry_ts for t in open_trades) if open_trades else None
        if latest_entry is not None:
            # now_minute may already be tz-aware; pandas refuses tz= kwarg in that case.
            ts_now = pd.Timestamp(now_minute)
            if ts_now.tz is None:
                ts_now = ts_now.tz_localize("UTC")
            time_since_last = (ts_now - latest_entry).total_seconds() / 60.0
            time_since_last = max(0.0, min(240.0, time_since_last))
        else:
            time_since_last = 240.0
        portfolio_state = {
            "n_open_long": float(n_long),
            "n_open_short": float(n_short),
            "combined_pnl_bps": float(combined_pnl),
            "time_since_last_entry_min": float(time_since_last),
        }
    return pipeline.make_entry_decision(
        pd.Timestamp(now_minute), bid, ask, portfolio_state=portfolio_state,
    )


def make_v12_exit_decision(pipeline: V12Pipeline, trade: TradeState,
                            now_minute: datetime, bid: float, ask: float,
                            m1_close: float) -> dict[str, Any]:
    """Run Exit-IQL V12.1 for one M1 bar on an open trade.

    Advances trade-state (PnL/MFE/MAE) and queries the Exit-IQL adapter.
    """
    return pipeline.make_exit_decision(trade, pd.Timestamp(now_minute), bid, ask, m1_close)


# ── Order execution + reject handling ────────────────────────────────────


def units_from_position_size_pred(base_units: int, position_size_pred: float,
                                    max_multiplier: float = 1.0) -> tuple[int, float]:
    """Map V10 v3+ position_size_pred ∈ [0,1] to actual units.

    Returns (units, applied_multiplier). Multiplier mapping:
      pred < 0.3 → 0.25×
      0.3-0.5    → 0.5×
      0.5-0.7    → 1.0×
      > 0.7      → 2.0×

    Falls back to plain base_units when:
      - pred is invalid (-1.0 sentinel from legacy bundles)
      - max_multiplier == 1.0 (feature disabled)
    Multiplier is clamped at max_multiplier so the runner can disable
    high-conviction sizing while still allowing reductions.
    """
    if position_size_pred < 0.0 or max_multiplier == 1.0:
        return base_units, 1.0
    if position_size_pred < 0.3:
        mult = 0.25
    elif position_size_pred < 0.5:
        mult = 0.5
    elif position_size_pred < 0.7:
        mult = 1.0
    else:
        mult = 2.0
    mult = min(mult, float(max_multiplier))
    units = max(1, int(round(base_units * mult)))   # never below 1 unit
    return units, mult


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
    # Extract trade_id: OANDA returns different paths depending on account mode +
    # netting outcome. HEDGING + new position: tradeOpened.tradeID. NETTING + offset:
    # tradesClosed[0].tradeID (the reduced trade). Fallback to orderFillTransaction.id
    # so we always have an identifier even if the position structure is odd.
    trade_id = (
        fill.get("tradeOpened", {}).get("tradeID")
        or fill.get("tradeReduced", {}).get("tradeID")
        or (fill.get("tradesClosed") or [{}])[0].get("tradeID")
        or fill.get("id")
    )
    if not fill.get("tradeOpened", {}).get("tradeID"):
        # Audit log when we fall back to non-tradeOpened parsing (was the
        # 2026-05-20 phantom-LONG bug — OANDA netted the longs against existing
        # shorts so tradeOpened was missing).
        LOG.warning(
            f"FILL parse fallback: tradeOpened missing, resolved trade_id={trade_id} via "
            f"{'tradeReduced' if fill.get('tradeReduced') else 'tradesClosed' if fill.get('tradesClosed') else 'fill.id'}. "
            f"Raw orderFillTransaction keys: {sorted(fill.keys())}"
        )
    LOG.info(f"FILLED side={side} units={signed_units}  price={fill.get('price')}  trade_id={trade_id}")
    return {"status": "filled",
             "fill_price": float(fill.get("price", 0)),
             "trade_id": trade_id,
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
    p.add_argument("--units", type=int, default=DEFAULT_DEFAULT_UNITS,
                   help="Base unit size per trade. If V10 v3+ position_size_pred is "
                        "available, actual order is units × multiplier (clamped via --max-units-multiplier).")
    p.add_argument("--max-units-multiplier", type=float, default=1.0,
                   help="Cap on position_size multiplier from V10 v3+ pred. "
                        "1.0 = ignore prediction (use plain --units). 2.0 = allow up to 2× units.")
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
    LOG.info(f"  max_spread_bps={args.max_spread_bps}  (year-round, all sessions)  "
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
                "dry_run": args.dry_run,
                # B4 fix (2026-06-04): record the env flags that change live decisions, so a
                # journal can be reconciled against the Phase-6 config post-hoc ("which config ran?").
                "pure_phase6": PURE_PHASE6,
                "cluster1_disable": bool(int(os.environ.get("GX1_CLUSTER1_DISABLE", "0"))),
                "regime_v4": bool(int(os.environ.get("GX1_REGIME_V4", "0")))},
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
                spread_bps, max_spread_bps=args.max_spread_bps, now_utc=now_utc,
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

            # MULTI-TF TREND ENRICHMENT (2026-05-22): LiveFeatureBuilder is M5-only,
            # but canonical_v3 prebuilt has per-TF EMA trend signals baked in via the
            # offline augmenter. Pull the latest row's per-TF trend features so the
            # entry gates and shadow filters can see H1/H4/D1 trend state, not just M5.
            try:
                cv3 = pipeline.prebuilt_loader._cv3
                if cv3 is not None and len(cv3) > 0:
                    last_row = cv3.iloc[-1]
                    for col in (
                        "_v1_ema_diff",            # M5 EMA distance (price units)
                        "_v1h1_ema_diff",          # H1 EMA distance
                        "_v1h4_ema_diff",          # H4 EMA distance
                        "pos_vs_ema200",           # M5 position vs EMA200 (bps)
                        "d1_ema_slope_20_canon_v2",  # D1 EMA20 slope
                        "ema20_slope",             # M5 EMA20 slope
                        "ema100_slope",            # M5 EMA100 slope
                    ):
                        if col in cv3.columns:
                            try: live_feats[col] = float(last_row[col])
                            except (TypeError, ValueError) as exc:
                                # Silent-skip historically hid stale cv3 columns;
                                # 2026-06-01 audit flagged that → log it so we
                                # catch feature gaps early instead of decisions
                                # silently running with 0.0 fallback.
                                LOG.warning(f"[multi-TF trend] col={col} cast failed: {exc} — "
                                            f"shadow filter for this col will see default 0.0")
            except Exception as exc:
                LOG.warning(f"multi-TF trend enrichment failed: {exc}")

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
            # H4 fix 2026-05-19: use the LAST CLOSED M1 bar's (bid_close+ask_close)/2
            # from the data collector parquet. Tick-mid (bid+ask)/2 at decision time
            # is biased — the bar hasn't closed yet, so we'd be using a mid-bar
            # snapshot when training used the M1 bar's actual close. Falls back to
            # tick-mid if the parquet read fails.
            m1_close = (bid + ask) / 2.0   # tick-mid fallback
            try:
                from pathlib import Path
                _day = current_minute.strftime("%Y%m%d")
                _p = Path(f"/home/andre2/GX1_DATA/reports/v12_live_data/xauusd_m1_{_day}.parquet")
                if _p.exists():
                    _df = pd.read_parquet(_p, columns=["time", "ask_close", "bid_close"]).tail(1)
                    if len(_df) > 0:
                        _row = _df.iloc[-1]
                        m1_close = float((_row["ask_close"] + _row["bid_close"]) / 2.0)
            except Exception:
                pass
            # TIME-OF-DAY EXIT (2026-05-21 audit improvement #19): force-close
            # all open trades by 21:00 UTC daily (US close) to avoid holding
            # through low-liquidity overnight chop. Soft session limit — trades
            # opened in EU/US must be resolved within session.
            DAILY_FORCE_EXIT_HOUR_UTC = 21
            # PURE_PHASE6: never force-exit; let Exit-IQL alone decide.
            force_exit_active = (not PURE_PHASE6) and now_utc.hour >= DAILY_FORCE_EXIT_HOUR_UTC and open_trades
            if force_exit_active:
                LOG.info(
                    f"[TIME_OF_DAY_EXIT] hour={now_utc.hour} >= {DAILY_FORCE_EXIT_HOUR_UTC} UTC, "
                    f"closing {len(open_trades)} open trade(s)"
                )
            per_trade_records: list[dict[str, Any]] = []
            survivors: list[TradeState] = []
            for trade in open_trades:
                if force_exit_active:
                    # Force EXIT_NOW; bypass Exit-IQL decision
                    close_result = attempt_close_trade(client, trade)
                    LOG.info(
                        f"[TIME_OF_DAY_EXIT] closed {trade.side} trade_id={trade.trade_id} "
                        f"pnl={trade.current_pnl_bps:+.2f}bps  status={close_result.get('status')}"
                    )
                    trade.delete_state_file(TRADE_STATE_DIR)
                    continue
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

            decision = make_v12_decision(pipeline, current_minute, bid, ask, open_trades=open_trades)
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
                # NETTING-guard removed 2026-05-20: account migrated to HEDGING
                # (101-004-31061417-002). Each market order now opens a separate
                # trade regardless of existing positions on the same instrument,
                # so opposite-side entries no longer net against existing trades.
                # If you reinstate a NETTING account, restore the guard from git.

                # ─── ALWAYS-ON EXECUTION SAFETY INVARIANTS (2026-06-03, vedtak
                # _execution_safety_invariants) — NOT gated on PURE_PHASE6. The
                # -2000 incident (06-02) was a SAME-SIDE pile-up: 16 shorts stacked
                # because the same-side cap was disabled by PURE_PHASE6=1. These are
                # hard, always-on, and reconciled against BROKER TRUTH (the runner
                # otherwise trusts only local TradeState, which can desync and blind
                # an otherwise-correct guard). Fail-CLOSED on broker fetch error.
                HARD_MAX_SAME_SIDE = int(os.environ.get("GX1_HARD_MAX_SAME_SIDE", "2"))
                opp_side = "short" if side == "long" else "long"
                n_same = sum(1 for t in open_trades if t.side == side)
                n_opp = sum(1 for t in open_trades if t.side == opp_side)
                if not args.dry_run:
                    try:
                        _bt = client.get_open_trades().get("trades", [])
                        _b_long = sum(1 for t in _bt if t.get("instrument") == "XAU_USD"
                                      and float(t.get("currentUnits", 0) or 0) > 0)
                        _b_short = sum(1 for t in _bt if t.get("instrument") == "XAU_USD"
                                       and float(t.get("currentUnits", 0) or 0) < 0)
                        # Trust the LARGER of local/broker per side (defensive vs desync).
                        n_same = max(n_same, _b_long if side == "long" else _b_short)
                        n_opp = max(n_opp, _b_short if side == "long" else _b_long)
                    except Exception as exc:  # noqa: BLE001 — fail-closed on broker error
                        LOG.error(f"[SAFETY_RECONCILE] broker get_open_trades failed: {exc} "
                                  f"— fail-closed SKIP of {side} entry")
                        event["order_status"] = "blocked_broker_reconcile_fail"
                        log_journal_event(daily_journal_path(args.journal_suffix), event)
                        last_decision_minute = current_minute
                        consecutive_errors = 0
                        time.sleep(args.poll_seconds)
                        continue
                # No-short-in-long + no-pile-up via the pure, unit-tested helper.
                # SKIP on violation — never flatten-first (no deadlock).
                _safe_ok, _safe_reason, _safe_detail = evaluate_entry_safety(
                    side, n_same, n_opp, HARD_MAX_SAME_SIDE)
                if not _safe_ok:
                    LOG.warning(f"[SAFETY] blocking {side} entry — {_safe_reason} "
                                f"(same={n_same}, opp={n_opp}, hard_cap={HARD_MAX_SAME_SIDE})")
                    event["order_status"] = _safe_reason
                    event["safety_n_same_side"] = n_same
                    event["safety_n_opposing"] = n_opp
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue

                # ADAPTIVE MIN-ADVANTAGE (2026-05-21, loosened 2026-05-22): scale
                # min_adv with current M5 ATR. Reduced from 0.5 → 0.35 to increase
                # trade-flow so we accumulate post-PLUS5 trades for N≥30 analysis
                # faster. Shadow filter still logs would-block flags for retrospective
                # A/B. Floor raised slightly (2 → 1.5) to keep accepting low-vol signals.
                # Env-var GX1_ADAPTIVE_MIN_ADV_ATR_MULT overrides for quick tuning.
                ADAPTIVE_MIN_ADV_ATR_MULT = float(
                    os.environ.get("GX1_ADAPTIVE_MIN_ADV_ATR_MULT", "0.35")
                )
                ADAPTIVE_MIN_ADV_FLOOR_BPS = float(
                    os.environ.get("GX1_ADAPTIVE_MIN_ADV_FLOOR_BPS", "1.5")
                )
                atr_bps_now = float(live_feats.get("atr14_m5_bps", 0.0) or 0.0)
                adaptive_min_adv = max(ADAPTIVE_MIN_ADV_FLOOR_BPS, ADAPTIVE_MIN_ADV_ATR_MULT * atr_bps_now)
                adv_taken = float(decision.get(
                    "advantage_over_skip_long" if side == "long" else "advantage_over_skip_short", 0.0
                ))

                # SHADOW FILTERS (2026-05-21): compute candidate gates that the
                # root-cause analysis on 137 pre-PLUS5 trades suggests would have
                # blocked deep-MAE entries. These are LOGGED ONLY — they do NOT
                # alter the actual decision. After N=30 post-PLUS5 trades we can
                # A/B-compare: would-block flags vs realized PnL/MAE per trade.
                dist_ema200_sh = float(live_feats.get("dist_to_ema200_bps", 0.0) or 0.0)
                ema_stack_aligned_sh = int(live_feats.get("ema_stack_aligned", 0) or 0)
                spread_z_sh = float(live_feats.get("spread_z_24h", 0.0) or 0.0)
                mins_to_sess_sh = float(live_feats.get("minutes_to_session_change", 0.0) or 0.0)
                q_long_sh = float(decision.get("q_take_long", 0.0) or 0.0)
                q_short_sh = float(decision.get("q_take_short", 0.0) or 0.0)
                q_chosen_sh = q_long_sh if side == "long" else q_short_sh
                # Multi-TF trend signals (2026-05-22: user-requested per-TF gates).
                # All values are price-units except pos_vs_ema200 (bps) and slopes.
                # Sign convention: NEGATIVE = downtrend on that TF.
                h1_diff = float(live_feats.get("_v1h1_ema_diff", 0.0) or 0.0)
                h4_diff = float(live_feats.get("_v1h4_ema_diff", 0.0) or 0.0)
                d1_slope = float(live_feats.get("d1_ema_slope_20_canon_v2", 0.0) or 0.0)
                m5_pos_ema200 = float(live_feats.get("pos_vs_ema200", 0.0) or 0.0)
                ema100_slope_v = float(live_feats.get("ema100_slope", 0.0) or 0.0)
                # Per-TF counter-trend: LONG when TF is in downtrend; SHORT when uptrend
                # Thresholds picked from observed ranges (small ε keeps "neutral" trades).
                h1_counter = (side == "long" and h1_diff < -0.3) or (side == "short" and h1_diff > 0.3)
                h4_counter = (side == "long" and h4_diff < -5.0) or (side == "short" and h4_diff > 5.0)
                d1_counter = (side == "long" and d1_slope < -10.0) or (side == "short" and d1_slope > 10.0)
                m5_ema200_counter = (side == "long" and m5_pos_ema200 < -5.0) or (side == "short" and m5_pos_ema200 > 5.0)
                ema100_counter = (side == "long" and ema100_slope_v < -0.5) or (side == "short" and ema100_slope_v > 0.5)
                event["shadow_filters"] = {
                    # ORIGINAL 5 shadow gates (kept for retrospective continuity)
                    "would_block_strict_trend": bool(
                        (side == "long" and dist_ema200_sh < -10.0) or
                        (side == "short" and dist_ema200_sh > +10.0)
                    ),
                    "would_block_ema_stack_misaligned": bool(ema_stack_aligned_sh != 1),
                    "would_block_spread_wide": bool(spread_z_sh > 0.0),
                    "would_block_midsession_dead": bool(mins_to_sess_sh > 280.0),
                    "would_block_q_overconfident": bool(abs(q_chosen_sh) > 200.0),
                    # NEW (2026-05-22): per-TF counter-trend gates from M5-cadence
                    # multi-TF features baked into canonical_v3 prebuilt.
                    "would_block_M5_ema200_counter": bool(m5_ema200_counter),
                    "would_block_M5_ema100_slope_counter": bool(ema100_counter),
                    "would_block_H1_counter": bool(h1_counter),
                    "would_block_H4_counter": bool(h4_counter),
                    "would_block_D1_counter": bool(d1_counter),
                    # Combined: ANY higher-TF (H1/H4/D1) disagrees with side
                    "would_block_any_higher_tf_counter": bool(h1_counter or h4_counter or d1_counter),
                    # Strict: ALL TFs must agree (TYPICAL TRADER RULE — every TF in trend direction)
                    "would_block_unless_all_tfs_agree": bool(
                        h1_counter or h4_counter or d1_counter or m5_ema200_counter or ema100_counter
                    ),
                    # raw diagnostic values
                    "shadow_dist_ema200_bps": dist_ema200_sh,
                    "shadow_ema_stack_aligned": ema_stack_aligned_sh,
                    "shadow_spread_z_24h": spread_z_sh,
                    "shadow_minutes_to_session_change": mins_to_sess_sh,
                    "shadow_q_chosen": q_chosen_sh,
                    "shadow_h1_ema_diff": h1_diff,
                    "shadow_h4_ema_diff": h4_diff,
                    "shadow_d1_ema_slope": d1_slope,
                    "shadow_m5_pos_ema200": m5_pos_ema200,
                    "shadow_ema100_slope": ema100_slope_v,
                }
                event["shadow_filters"]["would_block_any"] = any(
                    v for k, v in event["shadow_filters"].items() if k.startswith("would_block_")
                )
                if (not PURE_PHASE6) and adv_taken < adaptive_min_adv:
                    LOG.info(
                        f"[ADAPTIVE_MIN_ADV] blocking {side} entry — adv={adv_taken:+.2f} "
                        f"< adaptive_min={adaptive_min_adv:.2f} (ATR={atr_bps_now:.1f}, mult={ADAPTIVE_MIN_ADV_ATR_MULT})"
                    )
                    event["order_status"] = "blocked_adaptive_min_adv"
                    event["adaptive_min_adv_threshold"] = adaptive_min_adv
                    event["adaptive_min_adv_atr_bps"] = atr_bps_now
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue

                # REGIME-AWARE FILTER (2026-05-21): counterfactual replay on
                # 2026-05-20 journal showed Entry-IQL SHORTs lost 100% of the
                # time (mean -52 bps) in a strong uptrend (dist_to_ema200 > 50),
                # while LONGs won 88.7% (+43 bps 4h terminal). Block counter-trend
                # entries when price is far from EMA200. Asymmetric threshold so
                # we don't kill all entries during chop.
                REGIME_DIST_THRESHOLD_BPS = 50.0
                dist_ema200 = float(live_feats.get("dist_to_ema200_bps", 0.0) or 0.0)
                if (not PURE_PHASE6) and dist_ema200 > REGIME_DIST_THRESHOLD_BPS and side == "short":
                    LOG.info(
                        f"[REGIME_FILTER] blocking SHORT — dist_to_ema200={dist_ema200:+.1f} bps "
                        f"> +{REGIME_DIST_THRESHOLD_BPS} (strong uptrend, counter-trend shorts lose)"
                    )
                    event["order_status"] = "blocked_regime_uptrend"
                    event["regime_dist_ema200_bps"] = dist_ema200
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                if (not PURE_PHASE6) and dist_ema200 < -REGIME_DIST_THRESHOLD_BPS and side == "long":
                    LOG.info(
                        f"[REGIME_FILTER] blocking LONG — dist_to_ema200={dist_ema200:+.1f} bps "
                        f"< -{REGIME_DIST_THRESHOLD_BPS} (strong downtrend, counter-trend longs lose)"
                    )
                    event["order_status"] = "blocked_regime_downtrend"
                    event["regime_dist_ema200_bps"] = dist_ema200
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue

                # PORTFOLIO-AWARE OVERLAY (2026-05-20): Entry-IQL was trained on
                # independent single-step candidates with NO portfolio state. In
                # live, repeated same-side entries during prolonged drawdowns
                # stack losses. These hardcoded rules approximate a portfolio-
                # aware policy until proper retrain with portfolio features.
                # Same-side pile-up cap moved to the ALWAYS-ON safety invariant above
                # (HARD_MAX_SAME_SIDE). The old PURE_PHASE6-gated MAX_SAME_SIDE_OPEN=10
                # guard was removed 2026-06-03 — it was disabled live (PURE_PHASE6=1)
                # and let 16 shorts stack on 06-02 (-2000). Drawdown-block below is
                # now also always-on (un-gated from PURE_PHASE6).
                DRAWDOWN_BLOCK_BPS = float(os.environ.get("GX1_DRAWDOWN_BLOCK_BPS", "-100.0"))
                combined_pnl = sum(t.current_pnl_bps for t in open_trades)
                if combined_pnl < DRAWDOWN_BLOCK_BPS:
                    LOG.info(
                        f"[PORTFOLIO_GUARD] blocking {side} entry — combined unrealized "
                        f"PnL {combined_pnl:+.1f} bps < {DRAWDOWN_BLOCK_BPS} threshold"
                    )
                    event["order_status"] = "blocked_portfolio_drawdown"
                    event["portfolio_combined_pnl_bps"] = combined_pnl
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                # V10 v3+ Target 3: position_size_pred — model's confidence the
                # trade is worth taking at full size. Default behaviour: drive
                # ADAPTIVE sizing (units × multiplier, below) and let the learned
                # Entry-IQL policy decide whether to enter — no hardcoded skip.
                # The old hard-cutoff is ablatable via GX1_POSITION_CONFIDENCE_MIN
                # (default -1.0 = disabled). Set e.g. 0.6 to re-enable the cutoff.
                POSITION_CONFIDENCE_MIN = float(os.environ.get("GX1_POSITION_CONFIDENCE_MIN", "-1.0"))
                pos_pred = float(decision.get("v10_position_size_pred", -1.0))
                if (not PURE_PHASE6) and pos_pred >= 0.0 and pos_pred < POSITION_CONFIDENCE_MIN:
                    LOG.info(
                        f"[CONFIDENCE_FILTER] blocking {side} entry — "
                        f"position_size_pred={pos_pred:.3f} < {POSITION_CONFIDENCE_MIN}"
                    )
                    event["order_status"] = "blocked_low_confidence"
                    event["position_size_pred"] = pos_pred
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                trade_units, units_mult = units_from_position_size_pred(
                    args.units, pos_pred, max_multiplier=args.max_units_multiplier,
                )
                event["units_multiplier_applied"] = units_mult
                event["position_size_pred"] = pos_pred
                if units_mult != 1.0:
                    LOG.info(f"adaptive units: pos_size_pred={pos_pred:.3f} → mult={units_mult}× → {trade_units}u (base={args.units})")
                if args.dry_run:
                    event["order_status"] = "DRY_RUN"
                    virtual_id = f"virtual_{current_minute.strftime('%Y%m%dT%H%M%S')}"
                    new_trade = TradeState.open(
                        entry_ts=pd.Timestamp(current_minute),
                        side=side, entry_bid=bid, entry_ask=ask,
                        v10_snapshot=decision.get("_v10_snapshot", {}),
                        trade_id=virtual_id, units=trade_units,
                    )
                    new_trade.save(TRADE_STATE_DIR)
                    open_trades.append(new_trade)
                    try:
                        decision_ts_str = decision.get("decision_ts")
                        if decision_ts_str:
                            pipeline.record_entry_for_cluster(
                                side, pd.Timestamp(decision_ts_str),
                            )
                    except Exception:
                        pass
                    LOG.info(f"[DRY] virtual trade opened  side={side}  id={virtual_id}")
                else:
                    order_result = attempt_market_entry(client, side, units=trade_units)
                    event["order_status"] = order_result["status"]
                    event["order_details"] = order_result
                    if order_result.get("status") == "filled":
                        # NETTING-mode detection: if OANDA closed/reduced existing
                        # opposite-side trades (instead of opening a new one), we
                        # must NOT register a new TradeState. Drop the closed
                        # TradeState(s) from our internal list to match broker
                        # reality (2026-05-20 phantom-LONG bug).
                        raw_fill = order_result.get("raw", {}).get("orderFillTransaction", {})
                        had_open = bool(raw_fill.get("tradeOpened", {}).get("tradeID"))
                        closed_ids = [t.get("tradeID") for t in (raw_fill.get("tradesClosed") or [])]
                        reduced_id = raw_fill.get("tradeReduced", {}).get("tradeID")
                        netting_ids = [tid for tid in (closed_ids + [reduced_id]) if tid]
                        if not had_open and netting_ids:
                            # OANDA netted: remove matched open trades, do NOT open new.
                            LOG.warning(
                                f"[NETTING] {side} fill {trade_units}u netted against existing "
                                f"opposite-side trades {netting_ids} — no new TradeState created"
                            )
                            survivors_after_net: list[TradeState] = []
                            removed = 0
                            for t in open_trades:
                                if t.trade_id and t.trade_id in netting_ids:
                                    t.delete_state_file(TRADE_STATE_DIR)
                                    removed += 1
                                else:
                                    survivors_after_net.append(t)
                            open_trades[:] = survivors_after_net
                            event["netting_removed_n"] = removed
                            event["netting_trade_ids"] = netting_ids
                            event["order_status"] = "netted"
                            log_journal_event(daily_journal_path(args.journal_suffix), event)
                            last_decision_minute = current_minute
                            consecutive_errors = 0
                            time.sleep(args.poll_seconds)
                            continue
                        new_trade = TradeState.open(
                            entry_ts=pd.Timestamp(current_minute),
                            side=side, entry_bid=bid, entry_ask=ask,
                            v10_snapshot=decision.get("_v10_snapshot", {}),
                            trade_id=str(order_result.get("trade_id") or ""),
                            units=trade_units,
                        )
                        new_trade.save(TRADE_STATE_DIR)
                        open_trades.append(new_trade)
                        # Audit 3 C-3 fix 2026-05-20: only advance cluster-state
                        # AFTER trade is actually opened. Previously the pipeline
                        # advanced state on TAKE_*_NOW recommendation regardless
                        # of downstream gate/portfolio rejection.
                        try:
                            decision_ts_str = decision.get("decision_ts")
                            if decision_ts_str:
                                pipeline.record_entry_for_cluster(
                                    side, pd.Timestamp(decision_ts_str),
                                )
                        except Exception as exc:
                            LOG.warning(f"cluster-state record failed: {exc}")

                        # ── TradeJournal: log entry lifecycle ──
                        if new_trade.trade_id:
                            v10 = decision.get("_v10_snapshot") or {}
                            journal.log_order_submitted(
                                trade_id=new_trade.trade_id,
                                instrument=INSTRUMENT, side=side,
                                units=trade_units, order_type="MARKET",
                                client_order_id=order_result.get("client_order_id"),
                                oanda_env=creds.env,
                            )
                            journal.log_order_filled(
                                trade_id=new_trade.trade_id,
                                oanda_trade_id=new_trade.trade_id,
                                fill_price=order_result.get("fill_price"),
                                fill_units=trade_units,
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
                                base_units=args.units, units=trade_units,
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
                                units=trade_units,
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
            import traceback as _tb
            LOG.error(f"loop error (consec={consecutive_errors}): {exc}\n{_tb.format_exc()}")
            backoff = min(args.poll_seconds * (2 ** min(consecutive_errors, 5)), 300)
            time.sleep(backoff)

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
