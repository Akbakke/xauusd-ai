#!/usr/bin/env python3
"""GX1 XAUUSD model-native paper/live execution runner.

Entry direction is the calibrated model-native XAU model's exact
LONG/SHORT/FLAT argmax.
The runner may fail closed for execution safety, but it has no session, trend,
utility, confidence, sizing, or threshold rule that can rewrite model direction.

Modus operandi:
    1. Wait for next M1 candle close (poll OANDA every 5-10s).
    2. Pre-trade spread check: skip if (ask-bid)/bid > spread_threshold_bps.
    3. Make the contract-bound model-direction decision via V12Pipeline.
    4. If model direction is LONG/SHORT and execution safety admits it, place a
       learned, proof-bound integer-unit market order via OANDA.
    5. Catch MARKET_ORDER_REJECT_TRANSACTION; log reason + spread + time.
    6. If trade open: per-bar V3 v9 + V12.4 overlay → close order on EXIT_NOW.
    7. Log everything to daily journal for replay/comparison vs Phase 6 baseline.

Run (live demo on OANDA practice):
    PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 \\
        gx1/execution/v12_paper_runner.py --max-trades 5

We trade year-round, all sessions (Asia included): session, structure, trend,
liquidity, volatility, momentum, price action, path quality, and utility are
model inputs/evidence, never post-model direction overrides.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (  # noqa: E402
    MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC,
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    MODEL_NATIVE_RUNTIME_POLICY,
    RETIRED_RUNTIME_EVIDENCE_FRAGMENTS,
    require_model_native_entry_time,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (  # noqa: E402
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    ModelNativeSizingUnavailable,
    ValidatedLearnedSizingAuthority,
    apply_model_native_sizing,
    prepare_model_native_sizing_authority,
    require_model_native_sizing_authority_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (  # noqa: E402
    MODEL_DIRECTION_SELECTION_MODE,
)

ENV_FILE = REPO_ROOT / ".env"
if ENV_FILE.is_file():
    with ENV_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

LOG = logging.getLogger("v12_paper")
INSTRUMENT = "XAU_USD"
_RUNTIME_DEPS_LOADED = False


def _load_runtime_dependencies() -> None:
    """Import artifact-touching runtime dependencies only after the Entry guard."""
    global _RUNTIME_DEPS_LOADED
    global OandaClient, OandaClientConfig, load_oanda_credentials
    global EntryDecisionUnavailable, ExitDecisionUnavailable, V12Pipeline, TradeState, TradeJournal
    if _RUNTIME_DEPS_LOADED:
        return

    from gx1.execution.oanda_client import OandaClient as _OandaClient
    from gx1.execution.oanda_client import OandaClientConfig as _OandaClientConfig
    from gx1.execution.oanda_credentials import load_oanda_credentials as _load_oanda_credentials
    from gx1.execution.v12_pipeline import (
        EntryDecisionUnavailable as _EntryDecisionUnavailable,
        ExitDecisionUnavailable as _ExitDecisionUnavailable,
        V12Pipeline as _V12Pipeline,
    )
    from gx1.execution.v12_trade_state import TradeState as _TradeState
    from gx1.monitoring.trade_journal import TradeJournal as _TradeJournal

    OandaClient = _OandaClient
    OandaClientConfig = _OandaClientConfig
    load_oanda_credentials = _load_oanda_credentials
    EntryDecisionUnavailable = _EntryDecisionUnavailable
    ExitDecisionUnavailable = _ExitDecisionUnavailable
    V12Pipeline = _V12Pipeline
    TradeState = _TradeState
    TradeJournal = _TradeJournal
    _RUNTIME_DEPS_LOADED = True

# These legacy variables used to arm live-only direction gates or post-model
# sizing. Their code paths are deleted. Presence is rejected at startup so a
# stale shell/.env cannot silently imply an operating point that no longer exists.
RETIRED_ENTRY_OVERRIDE_ENV = (
    "GX1_PURE_PHASE6",
    "GX1_SKIP_ASIA",
    "GX1_ADAPTIVE_MIN_ADV_ATR_MULT",
    "GX1_ADAPTIVE_MIN_ADV_FLOOR_BPS",
    "GX1_POSITION_CONFIDENCE_MIN",
    "GX1_SIZING_MODE",
    "GX1_SIZING_MAX_MULT",
    "GX1_SIZING_MIN_MULT",
    "GX1_SIZING_CONV_LO",
    "GX1_SIZING_CONV_HI",
    "GX1_SIZING_CONV_SRC",
    "GX1_SIZING_MARGIN_POW",
    "GX1_SIZING_MARGIN_REF",
    "GX1_SIZING_ATR_REF_BPS",
    "GX1_SIZING_ATR_FLOOR_BPS",
    "GX1_DYNAMIC_SIZING",
    "GX1_POSITION_SIZE_MULTIPLIER",
    "GX1_POSITION_SIZE_FROM_MODEL",
    "GX1_REGIME_SIZE_MULTIPLIER",
    "GX1_SESSION_SIZE_MULTIPLIER",
    "GX1_TREND_SIZE_MULTIPLIER",
    "GX1_UTILITY_SIZE_MULTIPLIER",
    "GX1_CONVICTION_SIZE_MULTIPLIER",
    "GX1_REGIME_RECAL",
    "GX1_MAX_ENTRY_DECISION_LATENCY_SEC",
    "GX1_SMART_CTX_MTF_INCREMENTAL",
    "GX1_SMART_CTX_MTF_WARMUP_M5",
    "GX1_MODEL_NATIVE_CTX_MTF_WARMUP_M5",
    "GX1_SMART_PARITY_GATE_MAX_AGE_HOURS",
    "GX1_SMART_PARITY_GATE_MAX_CUTOFF_LAG_HOURS",
    "GX1_SMART_DIRECTION_AUDIT_MAX_AGE_HOURS",
    "GX1_SMART_CTX_MAX_STALENESS_M5",
)
RETIRED_ENTRY_SIZING_ENV_PREFIXES = (
    "GX1_SIZING_",
    "GX1_DYNAMIC_SIZING",
    "GX1_POSITION_SIZE_",
    "GX1_REGIME_SIZE_",
    "GX1_SESSION_SIZE_",
    "GX1_TREND_SIZE_",
    "GX1_UTILITY_SIZE_",
    "GX1_CONVICTION_SIZE_",
)

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt_optional_float(value: Any, spec: str) -> str:
    parsed = _float_or_none(value)
    return "NA" if parsed is None else format(parsed, spec)

# Pre-trade gates
DEFAULT_MAX_SPREAD_BPS = 7.0           # 2026-05-20 tightening: was 10.0, but
                                        # news-spike spreads at 8-10 bps ate
                                        # entry edge. Backtest data had clean
                                        # OANDA M5 spreads typically 1-3 bps.
DEFAULT_POLL_SECONDS = 10              # how often to check for new M1 close
DEFAULT_QUOTE_MAX_AGE_SEC = 90.0       # treat quote as stale (market closed/halted) if older
DEFAULT_MAX_TRADES = 1                 # max concurrent virtual trades held simultaneously
_EXECUTABLE_SNAPSHOT_ONLY_FIELDS = frozenset(
    {
        "runtime_evidence_schema_version",
        "model_policy",
        "atr_bps",
    }
)
MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS = frozenset(
    (
        MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
        - _EXECUTABLE_SNAPSHOT_ONLY_FIELDS
    )
    | MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    | {
        "action",
        "action_id",
        "edge_score",
        "selection_score_mode",
        "selection_score",
        "session",
        "p_long",
        "p_short",
        "p_flat",
        "v10_path_quality_pred",
        "v10_mfe_pred_at_entry",
        "v10_tradable_prob",
        "v10_bad_path_prob",
        "_v10_snapshot",
        "policy",
        "stub",
        "entry_signal_latency_min",
        "entry_signal_latency_cap_sec",
        "entry_signal_stale",
        "context_refresh_in_flight",
        "context_mtf_incremental",
    }
)


def _same_runtime_value(left: Any, right: Any) -> bool:
    """Exact envelope parity without allowing array-style truth ambiguity."""

    try:
        result = left == right
    except Exception:
        return False
    return result if isinstance(result, bool) else False


def require_executable_model_native_entry_decision(
    decision: dict[str, Any],
    entry_time: Any,
) -> dict[str, Any]:
    """Validate the complete live decision envelope before any order is sent.

    SmartEntry validates its pure model snapshot first; the pipeline then adds
    the complete timing evidence.  This runner boundary validates both again,
    requires an exact outer schema, and proves that the action/policy/timing
    envelope is identical to the frozen snapshot consumed by TradeState and the
    journal.
    """

    if not isinstance(decision, dict) or not decision:
        raise RuntimeError("[RUNNER_MODEL_NATIVE_DECISION_INVALID] missing decision")
    retired = sorted(
        key
        for key in decision
        if any(fragment in str(key).lower() for fragment in RETIRED_RUNTIME_EVIDENCE_FRAGMENTS)
    )
    if retired:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] retired fields=" + repr(retired)
        )
    observed_fields = set(decision)
    if observed_fields != set(MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] exact schema mismatch "
            f"missing={sorted(MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS - observed_fields)} "
            f"unexpected={sorted(observed_fields - MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS)}"
        )
    snapshot_raw = decision.get("_v10_snapshot")
    if not isinstance(snapshot_raw, dict):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] _v10_snapshot missing"
        )
    snapshot = require_model_native_runtime_evidence(
        snapshot_raw,
        context="V12_PAPER_RUNNER_PRE_ORDER",
    )
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(snapshot):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] complete timing evidence missing"
        )
    require_model_native_entry_time(
        snapshot,
        entry_time,
        context="V12_PAPER_RUNNER_PRE_ORDER",
    )
    if (
        decision.get("policy") != MODEL_NATIVE_RUNTIME_POLICY
        or snapshot["model_policy"] != decision["policy"]
    ):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] model policy mismatch"
        )

    shared_fields = (
        MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS
        - _EXECUTABLE_SNAPSHOT_ONLY_FIELDS
    ) | MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS
    mismatched = sorted(
        field
        for field in shared_fields
        if not _same_runtime_value(decision[field], snapshot[field])
    )
    if mismatched:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] snapshot parity mismatch "
            f"fields={mismatched}"
        )

    direction_index = snapshot["model_direction_index"]
    expected_action = ("TAKE_LONG_NOW", "TAKE_SHORT_NOW", "SKIP")[direction_index]
    expected_action_id = (1, 2, 0)[direction_index]
    if decision["action"] != expected_action or decision["action_id"] != expected_action_id:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] action/direction parity mismatch"
        )
    if decision["selection_score_mode"] != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] selection mode mismatch"
        )
    probabilities = snapshot["direction_probs"]
    expected_edge = max(probabilities[0], probabilities[1]) - probabilities[2]
    expected_selection = probabilities[direction_index]
    for field, expected in (
        ("edge_score", expected_edge),
        ("selection_score", expected_selection),
        ("p_long", probabilities[0]),
        ("p_short", probabilities[1]),
        ("p_flat", probabilities[2]),
    ):
        observed = _float_or_none(decision[field])
        if observed is None or not math.isclose(
            observed,
            float(expected),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise RuntimeError(
                f"[RUNNER_MODEL_NATIVE_DECISION_INVALID] {field} parity mismatch"
            )
    latency_sec = float(snapshot["entry_signal_latency_sec"])
    latency_min = _float_or_none(decision["entry_signal_latency_min"])
    latency_cap = _float_or_none(decision["entry_signal_latency_cap_sec"])
    if (
        latency_min is None
        or not math.isclose(latency_min, latency_sec / 60.0, rel_tol=1e-6, abs_tol=1e-7)
        or latency_cap != MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC
        or latency_sec > MODEL_NATIVE_MAX_ENTRY_SIGNAL_LATENCY_SEC
        or decision["entry_signal_stale"] is not False
    ):
        raise RuntimeError(
            "[RUNNER_MODEL_NATIVE_DECISION_INVALID] latency contract mismatch"
        )
    if decision["stub"] is not False:
        raise RuntimeError("[RUNNER_MODEL_NATIVE_DECISION_INVALID] stub decision forbidden")
    if decision["session"] not in {"ASIA", "EU", "OVERLAP", "US"}:
        raise RuntimeError("[RUNNER_MODEL_NATIVE_DECISION_INVALID] session missing")
    return snapshot


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
    quote_time_str = str(quote.get("time") or "").strip()
    if not quote_time_str:
        raise ValueError("quote_time_missing")
    quote_time = pd.to_datetime(quote_time_str, utc=True, errors="raise")
    now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    age_sec = (now - quote_time).total_seconds()
    if age_sec > max_age_sec:
        raise StaleQuoteError(age_sec, quote_time_str)
    bid = float(quote["bids"][0]["price"])
    ask = float(quote["asks"][0]["price"])
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= bid:
        raise ValueError(f"invalid_quote_prices bid={bid} ask={ask}")
    spread_bps = (ask - bid) / bid * 10000.0
    return spread_bps, bid, ask


def can_trade_now(spread_bps: float, *, max_spread_bps: float,
                   now_utc: datetime) -> tuple[bool, str]:
    """Operational spread safety only; never a session/direction policy.

    Session evidence is model input. `now_utc` stays in the stable call contract,
    but no hand-written time/session rule may turn a model trade into FLAT. A
    wide spread blocks execution without changing the reported model direction.
    """
    del now_utc
    if spread_bps > max_spread_bps:
        return False, f"spread_too_wide ({spread_bps:.1f} > {max_spread_bps})"
    return True, "ok"


# ── V12 decision (wired in sesjon 1-5) ────────────────────────────────────


def make_v12_decision(pipeline: V12Pipeline, now_minute: datetime,
                      bid: float, ask: float) -> dict[str, Any]:
    """Run the model-native SMART LONG/SHORT/FLAT entry stack.

    Returns the model argmax action plus diagnostics and timestamp. Pipeline
    caches state so each newly closed M5 row is decided exactly once.

    Portfolio/exposure state is execution admission, not a hidden parallel
    direction input. The model receives only its explicit hashed input contract.
    """
    return pipeline.make_entry_decision(pd.Timestamp(now_minute), bid, ask)


def learned_sizing_runtime_constraints(
    client: OandaClient,
    *,
    bid: float,
    ask: float,
    validated_authority: ValidatedLearnedSizingAuthority,
) -> dict[str, Any]:
    """Read fresh broker-truth account, instrument, and XAU exposure facts.

    These are facts, not knobs: the learned calibration rejects any value that
    differs from its immutable instrument contract.  Missing, fractional, or
    ambiguous broker fields raise and therefore authorize no order.
    """

    parsed_bid = _float_or_none(bid)
    parsed_ask = _float_or_none(ask)
    if (
        parsed_bid is None
        or parsed_ask is None
        or parsed_bid <= 0.0
        or parsed_ask < parsed_bid
    ):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] invalid live XAU quote"
        )
    if not isinstance(validated_authority, ValidatedLearnedSizingAuthority):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] startup-validated authority missing"
        )
    instrument_contract = validated_authority.calibration["instrument_constraints"]
    account_payload = client.get_account_summary()
    account_observed_utc = datetime.now(timezone.utc).isoformat()
    account = account_payload.get("account") if isinstance(account_payload, dict) else None
    if not isinstance(account, dict):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] account summary missing account"
        )
    if account.get("hedgingEnabled") is not True:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] account must prove hedgingEnabled=true"
        )
    instrument_payload = client.get_account_instruments([INSTRUMENT])
    instrument_observed_utc = datetime.now(timezone.utc).isoformat()
    rows = (
        instrument_payload.get("instruments")
        if isinstance(instrument_payload, dict)
        else None
    )
    matches = [
        row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("name") == INSTRUMENT
    ]
    if len(matches) != 1:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] exact XAU instrument row missing"
        )
    instrument = matches[0]
    exposure_payload = client.get_open_trades()
    exposure_observed_utc = datetime.now(timezone.utc).isoformat()
    broker_trades = (
        exposure_payload.get("trades")
        if isinstance(exposure_payload, dict)
        else None
    )
    if not isinstance(broker_trades, list):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] open-trades exposure missing"
        )
    try:
        precision = int(instrument["tradeUnitsPrecision"])
        minimum_units_float = float(instrument["minimumTradeSize"])
        broker_maximum_units_float = float(instrument["maximumOrderUnits"])
        margin_rate = float(instrument["marginRate"])
        equity = float(account["NAV"])
        balance = float(account["balance"])
        margin_available = float(account["marginAvailable"])
        margin_used = float(account["marginUsed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker sizing field invalid"
        ) from exc
    if precision != 0:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] XAU units must be integer precision"
        )
    minimum_units = int(minimum_units_float)
    broker_maximum_units = int(broker_maximum_units_float)
    if (
        minimum_units_float != minimum_units
        or broker_maximum_units_float != broker_maximum_units
        or not all(
            math.isfinite(value)
            for value in (
                margin_rate,
                equity,
                balance,
                margin_available,
                margin_used,
                minimum_units_float,
                broker_maximum_units_float,
            )
        )
    ):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] non-exact broker unit constraints"
        )
    current_xau_abs_units = 0
    for row in broker_trades:
        if not isinstance(row, dict) or row.get("instrument") != INSTRUMENT:
            continue
        try:
            units_float = float(row["currentUnits"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelNativeSizingUnavailable(
                "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] XAU exposure units invalid"
            ) from exc
        units = int(units_float)
        if not math.isfinite(units_float) or units_float != units:
            raise ModelNativeSizingUnavailable(
                "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] XAU exposure units not exact integer"
            )
        current_xau_abs_units += abs(units)
    maximum_gross_units = int(instrument_contract["maximum_gross_xau_units"])
    if broker_maximum_units < maximum_gross_units:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker maximum is below immutable gross cap"
        )
    if (
        account.get("currency") != instrument_contract["account_currency"]
        or margin_rate != instrument_contract["margin_rate"]
        or minimum_units != instrument_contract["minimum_order_units"]
    ):
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker facts differ from adopted instrument contract"
        )
    transaction_ids: dict[str, str] = {}
    for name, payload in (
        ("account", account_payload),
        ("instrument", instrument_payload),
        ("exposure", exposure_payload),
    ):
        raw = payload.get("lastTransactionID") if isinstance(payload, dict) else None
        if not isinstance(raw, str) or not raw.strip():
            raise ModelNativeSizingUnavailable(
                f"[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] {name} lastTransactionID missing"
            )
        transaction_ids[name] = raw.strip()
    if len(set(transaction_ids.values())) != 1:
        raise ModelNativeSizingUnavailable(
            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] broker facts span different lastTransactionID snapshots"
        )
    decision_utc = datetime.now(timezone.utc).isoformat()
    account_floating_drawdown_bps = max(
        0.0, (balance - equity) / balance * 10_000.0
    ) if balance > 0.0 else float("nan")
    return {
        "instrument": INSTRUMENT,
        "account_currency": account.get("currency"),
        "account_equity": equity,
        "account_balance": balance,
        "account_floating_drawdown_bps": account_floating_drawdown_bps,
        "margin_available": margin_available,
        "margin_used": margin_used,
        "mark_price": (parsed_bid + parsed_ask) / 2.0,
        "margin_rate": margin_rate,
        "unit_step": 1,
        "minimum_order_units": minimum_units,
        "maximum_gross_xau_units": maximum_gross_units,
        "current_xau_abs_units": current_xau_abs_units,
        "sizing_decision_utc": decision_utc,
        "account_observed_utc": account_observed_utc,
        "instrument_observed_utc": instrument_observed_utc,
        "exposure_observed_utc": exposure_observed_utc,
        "account_last_transaction_id": transaction_ids["account"],
        "instrument_last_transaction_id": transaction_ids["instrument"],
        "exposure_last_transaction_id": transaction_ids["exposure"],
        "fact_provenance_mode": "broker_live",
    }


def make_v12_exit_decision(pipeline: V12Pipeline, trade: TradeState,
                            now_minute: datetime, bid: float, ask: float,
                            m1_close: float | None = None) -> dict[str, Any]:
    """Run Exit-IQL V12.1 for one M1 bar on an open trade.

    Advances trade-state (PnL/MFE/MAE) and queries the Exit-IQL adapter.
    """
    return pipeline.make_exit_decision(trade, pd.Timestamp(now_minute), bid, ask, m1_close)


# ── Order execution + reject handling ────────────────────────────────────


def attempt_market_entry(client: OandaClient, side: str,
                         units: int) -> dict[str, Any]:
    """Submit market order. Returns dict with status + reason if rejected.

    OANDA returns MARKET_ORDER_REJECT_TRANSACTION on rejection. Common reasons:
      MARKET_HALTED, INSUFFICIENT_LIQUIDITY, INSTRUMENT_HALTED,
      ACCOUNT_NOT_TRADEABLE, MARGIN_RATE_INVALID, PRICE_PRECISION_EXCEEDED.
    """
    if side not in ("long", "short"):
        return {"status": "skipped", "reason": f"invalid_side {side}"}
    if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
        return {
            "status": "skipped",
            "reason": "units must be an exact positive learned-sizing integer",
        }
    signed_units = units if side == "long" else -units
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
    raw_fill_units = fill.get("units")
    try:
        fill_units_float = float(raw_fill_units)
        signed_fill_units = int(fill_units_float)
    except (TypeError, ValueError, OverflowError):
        signed_fill_units = None
    fill_units_exact = bool(
        signed_fill_units is not None
        and math.isfinite(fill_units_float)
        and fill_units_float == signed_fill_units
        and signed_fill_units == signed_units
    )
    trade_opened = fill.get("tradeOpened")
    raw_opened_units = (
        trade_opened.get("units") if isinstance(trade_opened, dict) else None
    )
    try:
        opened_units_float = float(raw_opened_units)
        signed_opened_units = int(opened_units_float)
    except (TypeError, ValueError, OverflowError):
        signed_opened_units = None
    pure_trade_open = bool(
        isinstance(trade_opened, dict)
        and isinstance(trade_opened.get("tradeID"), str)
        and trade_opened["tradeID"].strip()
        and signed_opened_units is not None
        and math.isfinite(opened_units_float)
        and opened_units_float == signed_opened_units
        and signed_opened_units == signed_units
        and not fill.get("tradesClosed")
        and not fill.get("tradeReduced")
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
    status = (
        "filled"
        if fill_units_exact and pure_trade_open
        else "filled_structure_mismatch"
        if fill_units_exact
        else "filled_units_mismatch"
    )
    if status != "filled":
        LOG.error(
            "FILL CONTRACT MISMATCH status=%s side=%s requested=%s fill=%r "
            "tradeOpened=%r closed=%r reduced=%r trade_id=%s",
            status,
            side,
            signed_units,
            raw_fill_units,
            raw_opened_units,
            fill.get("tradesClosed"),
            fill.get("tradeReduced"),
            trade_id,
        )
    else:
        LOG.info(
            "FILLED side=%s units=%s price=%s trade_id=%s",
            side,
            signed_fill_units,
            fill.get("price"),
            trade_id,
        )
    return {"status": status,
             "fill_price": _float_or_none(fill.get("price")),
             "requested_signed_units": signed_units,
             "filled_signed_units": signed_fill_units,
             "trade_opened_signed_units": signed_opened_units,
             "fill_units_exact": fill_units_exact,
             "pure_trade_open": pure_trade_open,
             "fill_time": fill.get("time"),
             "oanda_transaction_id": fill.get("id"),
             "oanda_order_id": fill.get("orderID"),
             "trade_id": trade_id,
             "client_order_id": client_order_id,
             "raw": response}


def attempt_close_trade(client: OandaClient, trade: TradeState) -> dict[str, Any]:
    """Close a specific virtual trade by its OANDA tradeID.

    Works in both NETTING and HEDGING account modes: OANDA's
    PUT /accounts/{id}/trades/{tradeID}/close endpoint closes only the units
    associated with that tradeID, not the full netted position.

    Missing ``trade_id`` is an unresolved broker-identity failure. Sending a
    counter-direction market order is forbidden because a hedging account may
    open a second trade instead of reducing the original exposure.
    """
    if not trade.trade_id:
        LOG.error("close_trade: trade has no OANDA tradeID; refusing counter-order")
        return {
            "status": "missing_trade_id",
            "trade_id": None,
            "reason": "broker trade identity is required for fail-closed close",
        }
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
    record = dict(event)
    record["logged_at_utc"] = datetime.now(timezone.utc).isoformat()
    encoded = (json.dumps(record, default=str) + "\n").encode("utf-8")
    fd = os.open(
        journal_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short journal write: {journal_path}")
            view = view[written:]
        os.fsync(fd)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def daily_journal_path(suffix: str = "") -> Path:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suf = f"_{suffix}" if suffix else ""
    return JOURNAL_DIR / f"v12_paper_journal_{today}{suf}.jsonl"


def assert_no_retired_entry_overrides() -> None:
    """Reject every legacy post-model entry knob, including disabled values."""
    present = sorted(
        name
        for name in os.environ
        if name in RETIRED_ENTRY_OVERRIDE_ENV
        or any(
            name.startswith(prefix)
            for prefix in RETIRED_ENTRY_SIZING_ENV_PREFIXES
        )
    )
    if present:
        raise SystemExit(
            "[SMART_GATE] retired entry override environment variables are forbidden; "
            "remove them instead of relying on disabled/no-op values: " + ", ".join(present)
        )


def _sha256_regular_file(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"[SMART_GATE] {label} is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_runtime_entry_launch_lease(
    *,
    expected_lease: dict[str, str] | None = None,
) -> dict[str, str]:
    """Revalidate launch authority and reject any mid-process identity change."""

    from gx1.execution.v12_smart_entry_live import assert_smart_serving_gate
    from gx1_guards import artifacts as artifact_guard

    state_path = artifact_guard.XAU_DIRECTION_LAUNCH_CONTRACT
    registry_path = artifact_guard.SELECTION_CONTRACT
    before = {
        "launch_state_sha256": _sha256_regular_file(
            state_path, label="launch state"
        ),
        "artifact_registry_sha256": _sha256_regular_file(
            registry_path, label="artifact registry"
        ),
    }
    assert_smart_serving_gate()
    entry = artifact_guard.load_decision_entry("v10_entry")
    after = {
        "launch_state_sha256": _sha256_regular_file(
            state_path, label="launch state"
        ),
        "artifact_registry_sha256": _sha256_regular_file(
            registry_path, label="artifact registry"
        ),
    }
    if before != after:
        raise RuntimeError(
            "[SMART_GATE] launch authority changed during lease validation"
        )
    launch_state = entry.get("xau_direction_launch_state")
    approval = (
        launch_state.get("accepted_via_vedtak")
        if isinstance(launch_state, dict)
        else None
    )
    if not isinstance(approval, dict):
        raise RuntimeError("[SMART_GATE] validated launch approval is missing")
    lease = {
        **after,
        "accepted_bundle_dir": str(Path(entry["path"]).resolve()),
        "approval_event_sha256": str(approval.get("event_sha256") or ""),
        "approval_vedtak_id": str(approval.get("vedtak_id") or ""),
    }
    if expected_lease is not None and lease != expected_lease:
        raise RuntimeError(
            "[SMART_GATE] launch authority was replaced or revoked; restart required"
        )
    return lease


def enforce_entry_next_edge_runner_guard(args: argparse.Namespace) -> dict[str, str]:
    """Require the exact immutable launch approval and serving evidence.

    Direct runner startup and the shell launcher use the same artifact-bound
    authority. Ambient environment text is never launch authorization.
    """
    assert_no_retired_entry_overrides()
    from gx1_guards.artifacts import load_decision_entry
    from gx1.models.entry_v10.direction_decision_contract import (
        require_model_direction_operating_point,
    )

    lease = require_runtime_entry_launch_lease()
    entry = load_decision_entry("v10_entry")
    op = require_model_direction_operating_point(
        entry.get("operating_point"), context="paper runner v10_entry"
    )
    expected_max_trades = int(op["max_trades"])
    if int(args.max_trades) != expected_max_trades:
        raise SystemExit(
            f"[SMART_GATE] --max-trades {args.max_trades} != "
            "contract v10_entry.operating_point.max_trades "
            f"{expected_max_trades}"
        )
    approval = entry["xau_direction_launch_state"]["accepted_via_vedtak"]
    LOG.warning(
        "[SMART_GATE] runner start authorized: vedtak=%s parity=%s "
        "launch_state_sha256=%s",
        approval["vedtak_id"],
        "PASS",
        lease["launch_state_sha256"],
    )
    return lease


# ── Main loop ────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="GX1 XAUUSD model-direction runner")
    p.add_argument("--max-spread-bps", type=float, default=DEFAULT_MAX_SPREAD_BPS)
    p.add_argument("--max-trades", type=int, default=DEFAULT_MAX_TRADES,
                   help="Max concurrent virtual trades held simultaneously (default: 1)")
    p.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    p.add_argument("--dry-run", action="store_true",
                   help="Don't actually send orders — just log what would happen (shadow mode)")
    p.add_argument("--shadow-only", action="store_true",
                   help="Observation-only mode: requires --dry-run, logs decisions, never opens virtual trades or writes trade state.")
    p.add_argument("--journal-suffix", type=str, default="",
                   help="Suffix for journal filename (e.g. 'live' or 'shadow') to allow parallel runners")
    args = p.parse_args()
    if args.shadow_only and not args.dry_run:
        raise SystemExit("--shadow-only requires --dry-run")
    startup_launch_lease = enforce_entry_next_edge_runner_guard(args)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")
    _load_runtime_dependencies()

    creds = load_oanda_credentials()
    client = OandaClient(OandaClientConfig(api_key=creds.api_token,
                                            account_id=creds.account_id,
                                            env=creds.env))
    LOG.info(f"V12 paper runner starting  env={creds.env}  account={creds.account_id}")
    LOG.info(f"  max_spread_bps={args.max_spread_bps}  (year-round, all sessions)  "
             f"max_trades={args.max_trades}  dry_run={args.dry_run}")
    LOG.info("  entry: exact model direction; immutable learned calibrated sizing")

    # Load model-native Entry plus the separately admitted Exit stack.
    LOG.info("loading V12Pipeline (model-native Entry + Exit stack)...")
    pipeline = V12Pipeline.load_default()
    if pipeline.smart_entry is None:
        raise SystemExit("model-native learned sizing requires loaded smart_entry")
    runtime_sizing_authority = require_model_native_sizing_authority_contract(
        getattr(pipeline.smart_entry, "_sizing_authority", None),
        context="V12_PAPER_RUNNER_STARTUP",
        required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
    )
    validated_sizing_authority = prepare_model_native_sizing_authority(
        runtime_sizing_authority,
        context="V12_PAPER_RUNNER_STARTUP",
    )
    LOG.info("✓ V12 entry+exit stacks loaded — runner is live-wired")

    # Structured trade journal — per-trade JSON + aggregate index CSV.
    # Captures: exact model-native entry evidence, applied learned sizing,
    # per-bar exit decisions, execution events (OANDA fills),
    # exit_summary (realized PnL, MFE/MAE peaks).
    journal = TradeJournal(
        run_dir=JOURNAL_DIR,
        run_tag=f"v12_paper_{args.journal_suffix or 'live'}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        header={"runner": "v12_paper_runner",
                "env": creds.env,
                "max_trades": args.max_trades,
                "max_spread_bps": args.max_spread_bps,
                "dry_run": args.dry_run,
                "shadow_only": args.shadow_only,
                "entry_direction_mode": "model_direction_argmax",
                "entry_units_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
                "sizing_authority_contract": runtime_sizing_authority},
    )
    LOG.info(f"  trade journal: {journal.journal_dir}")

    last_decision_minute = None
    consecutive_errors = 0
    last_stale_log_minute = None

    # Resume any open trades from disk (survives runner crash/restart).
    # Auto-migrates legacy single-file state into the per-trade directory.
    open_trades: list[TradeState]
    if args.shadow_only:
        open_trades = []
        LOG.warning("[SHADOW_ONLY] open trade state loading disabled; no orders or virtual trades will be opened")
    else:
        open_trades = TradeState.load_all(
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

            event = {
                "ts_utc": current_minute.isoformat(),
                "bid": bid, "ask": ask, "spread_bps": spread_bps,
                "allowed": allowed, "gate_reason": reason,
                "n_open_trades": len(open_trades),
                "max_trades": args.max_trades,
                "has_open_trade": len(open_trades) > 0,   # back-compat for analytics
            }

            # ── EXIT branch: iterate all open trades, evaluate each per M1 ──
            # The pipeline owns the unique exact-closed-M1 contract.  The runner
            # must not manufacture that state from a tick midpoint/latest-row read.
            per_trade_records: list[dict[str, Any]] = []
            survivors: list[TradeState] = []
            exit_decision_unavailable = False
            exit_execution_unresolved = False
            for trade in open_trades:
                try:
                    exit_decision = make_v12_exit_decision(
                        pipeline, trade, current_minute, bid, ask,
                    )
                except ExitDecisionUnavailable as exc:
                    exit_decision_unavailable = True
                    record = {
                        "trade_id": trade.trade_id,
                        "side": trade.side,
                        "entry_ts": trade.entry_ts.isoformat(),
                        "bars_in_trade": trade.bars_in_trade,
                        "units": trade.units,
                        "pnl_bps": trade.current_pnl_bps,
                        "peak_bps": trade.cum_mfe_bps,
                        "mae_bps": trade.cum_mae_bps,
                        "exit_decision": None,
                        "exit_decision_unavailable_reason": exc.reason,
                        "exit_decision_unavailable_evidence": exc.evidence,
                        "order_status": "EXIT_MODEL_DECISION_UNAVAILABLE",
                    }
                    LOG.error(
                        "[EXIT_MODEL_DECISION_UNAVAILABLE] trade_id=%s reason=%s evidence=%s",
                        trade.trade_id,
                        exc.reason,
                        exc.evidence,
                    )
                    if trade.trade_id:
                        journal.log(
                            event_type="EXIT_MODEL_DECISION_UNAVAILABLE",
                            trade_id=trade.trade_id,
                            payload={
                                "timestamp": current_minute.isoformat(),
                                "reason": exc.reason,
                                "evidence": exc.evidence,
                                "bars_in_trade": int(trade.bars_in_trade),
                                "current_pnl_bps": float(trade.current_pnl_bps),
                            },
                        )

                    # Preserve the independent 24h execution-safety close.  It
                    # reduces an existing exposure; it does not invent HOLD/EXIT
                    # model authority or permit a new Entry decision this minute.
                    if trade.bars_in_trade >= 1440:
                        LOG.warning(
                            "24h cap reached while Exit model unavailable for trade %s — forced close",
                            trade.trade_id,
                        )
                        record["order_status"] = "FORCED_CLOSE_24H"
                        if args.dry_run:
                            trade.delete_state_file(TRADE_STATE_DIR)
                        else:
                            close_result = attempt_close_trade(client, trade)
                            record["close_order_details"] = close_result
                            if close_result.get("status") in ("closed", "filled"):
                                trade.delete_state_file(TRADE_STATE_DIR)
                            else:
                                exit_execution_unresolved = True
                                record["order_status"] = "FORCED_CLOSE_24H_FAILED"
                                trade.save(TRADE_STATE_DIR)
                                survivors.append(trade)
                    else:
                        trade.save(TRADE_STATE_DIR)
                        survivors.append(trade)
                    per_trade_records.append(record)
                    continue
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
                        trade.delete_state_file(TRADE_STATE_DIR)
                    else:
                        close_result = attempt_close_trade(client, trade)
                        record["close_order_details"] = close_result
                        if close_result.get("status") in ("closed", "filled"):
                            trade.delete_state_file(TRADE_STATE_DIR)
                        else:
                            exit_execution_unresolved = True
                            record["order_status"] = "EXIT_CLOSE_FAILED"
                            trade.save(TRADE_STATE_DIR)
                            survivors.append(trade)
                            write_trade_alert(
                                f"EXIT CLOSE FAILED trade_id={trade.trade_id} side={trade.side} "
                                f"status={close_result.get('status')} reason={close_result.get('reason')}"
                            )
                elif trade.bars_in_trade >= 1440:   # 24h hard cap
                    LOG.warning(f"24h cap reached for trade {trade.trade_id} — forced close")
                    record["order_status"] = "FORCED_CLOSE_24H"
                    if args.dry_run:
                        trade.delete_state_file(TRADE_STATE_DIR)
                    else:
                        close_result = attempt_close_trade(client, trade)
                        record["close_order_details"] = close_result
                        if close_result.get("status") in ("closed", "filled"):
                            trade.delete_state_file(TRADE_STATE_DIR)
                        else:
                            exit_execution_unresolved = True
                            record["order_status"] = "FORCED_CLOSE_24H_FAILED"
                            trade.save(TRADE_STATE_DIR)
                            survivors.append(trade)
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
                        if not args.dry_run:
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

            # An unavailable Exit contract is neither HOLD nor permission to
            # evaluate/open a new trade. Persist evidence and end this M1 step.
            if exit_decision_unavailable:
                event["order_status"] = "EXIT_MODEL_DECISION_UNAVAILABLE"
                event["exit_model_decision_unavailable"] = True
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            if exit_execution_unresolved:
                event["order_status"] = "EXIT_EXECUTION_UNRESOLVED"
                event["exit_execution_unresolved"] = True
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            # ── ENTRY branch: evaluate the admitted model-native XAU model ──
            if len(open_trades) >= args.max_trades:
                event.setdefault("order_status", "AT_MAX_TRADES")
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            try:
                require_runtime_entry_launch_lease(
                    expected_lease=startup_launch_lease
                )
            except RuntimeError as exc:
                event["order_status"] = "LAUNCH_AUTHORITY_UNAVAILABLE_NO_ORDER"
                event["launch_authority_evidence"] = str(exc)
                log_journal_event(
                    daily_journal_path(args.journal_suffix),
                    event,
                )
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            try:
                decision = make_v12_decision(
                    pipeline,
                    datetime.now(timezone.utc),
                    bid,
                    ask,
                )
            except EntryDecisionUnavailable as exc:
                event["order_status"] = "MODEL_DECISION_UNAVAILABLE"
                event["model_decision_unavailable_reason"] = exc.reason
                event["model_decision_unavailable_evidence"] = exc.evidence
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue
            decision["_v10_snapshot"] = require_executable_model_native_entry_decision(
                decision,
                current_minute,
            )
            event["v12_decision"] = decision
            direction_to_action = {
                "LONG": (0, "TAKE_LONG_NOW"),
                "SHORT": (1, "TAKE_SHORT_NOW"),
                "FLAT": (2, "SKIP"),
            }
            model_direction = str(decision.get("model_direction") or "")
            if model_direction not in direction_to_action:
                raise RuntimeError(
                    f"model decision lacks exact LONG/SHORT/FLAT direction: {model_direction!r}"
                )
            expected_index, expected_action = direction_to_action[model_direction]
            if decision.get("model_direction_index") != expected_index:
                raise RuntimeError(
                    "model direction index disagrees with model direction: "
                    f"{decision.get('model_direction_index')!r} != {expected_index}"
                )
            if decision.get("action") != expected_action:
                raise RuntimeError(
                    "runner action disagrees with model direction argmax: "
                    f"{decision.get('action')!r} != {expected_action!r}"
                )
            if args.shadow_only:
                event["order_status"] = "SHADOW_ONLY_NO_ORDER"
                event["shadow_only"] = True
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            if not allowed:
                # Execution safety blocks the order without rewriting the model direction.
                event["order_status"] = "BLOCKED_BY_EXECUTION_SPREAD"
                log_journal_event(daily_journal_path(args.journal_suffix), event)
                last_decision_minute = current_minute
                consecutive_errors = 0
                time.sleep(args.poll_seconds)
                continue

            if expected_action in ("TAKE_LONG_NOW", "TAKE_SHORT_NOW"):
                side = "long" if model_direction == "LONG" else "short"
                # NETTING-guard removed 2026-05-20: account migrated to HEDGING
                # (101-004-31061417-002). Each market order now opens a separate
                # trade regardless of existing positions on the same instrument,
                # so opposite-side entries no longer net against existing trades.
                # If you reinstate a NETTING account, restore the guard from git.

                try:
                    sizing_constraints = learned_sizing_runtime_constraints(
                        client,
                        bid=bid,
                        ask=ask,
                        validated_authority=validated_sizing_authority,
                    )
                    decision_sizing_authority = require_model_native_sizing_authority_contract(
                        decision["_v10_snapshot"]["sizing_authority_contract"],
                        context="V12_PAPER_RUNNER_DECISION_SIZING_AUTHORITY",
                        required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
                    )
                    if decision_sizing_authority != runtime_sizing_authority:
                        raise ModelNativeSizingUnavailable(
                            "[V12_PAPER_RUNNER_SIZING_UNAVAILABLE] decision/startup adoption mismatch"
                        )
                    sizing_application = apply_model_native_sizing(
                        validated_authority=validated_sizing_authority,
                        position_size_logit=decision["_v10_snapshot"][
                            "position_size_logit"
                        ],
                        model_direction=model_direction,
                        runtime_constraints=sizing_constraints,
                        context="V12_PAPER_RUNNER_ENTRY",
                    )
                except (ModelNativeSizingUnavailable, RuntimeError) as exc:
                    LOG.error("[SIZING_UNAVAILABLE] no order: %s", exc)
                    event["order_status"] = "SIZING_UNAVAILABLE_NO_ORDER"
                    event["sizing_unavailable_evidence"] = str(exc)
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                event["sizing_application"] = sizing_application
                if not sizing_application["authorized_order"]:
                    event["order_status"] = "MODEL_NATIVE_SIZING_NO_ORDER"
                    event["sizing_no_order_reason"] = sizing_application[
                        "no_order_reason"
                    ]
                    log_journal_event(daily_journal_path(args.journal_suffix), event)
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                try:
                    require_runtime_entry_launch_lease(
                        expected_lease=startup_launch_lease
                    )
                except RuntimeError as exc:
                    event["order_status"] = (
                        "LAUNCH_AUTHORITY_UNAVAILABLE_NO_ORDER"
                    )
                    event["launch_authority_evidence"] = str(exc)
                    log_journal_event(
                        daily_journal_path(args.journal_suffix),
                        event,
                    )
                    last_decision_minute = current_minute
                    consecutive_errors = 0
                    time.sleep(args.poll_seconds)
                    continue
                trade_units = int(sizing_application["units"])
                event["units_mode"] = MODEL_NATIVE_SIZING_MODE_LEARNED
                event["units"] = trade_units
                event["sizing_authority_contract"] = sizing_application[
                    "sizing_authority_contract"
                ]
                if args.dry_run:
                    event["order_status"] = "DRY_RUN"
                    virtual_id = f"virtual_{current_minute.strftime('%Y%m%dT%H%M%S')}"
                    new_trade = TradeState.open(
                        entry_ts=pd.Timestamp(current_minute),
                        side=side, entry_bid=bid, entry_ask=ask,
                        v10_snapshot=decision["_v10_snapshot"],
                        trade_id=virtual_id,
                        units=trade_units,
                        sizing_application=sizing_application,
                        fill_transaction_id=f"virtual:{virtual_id}",
                        execution_mode="learned_virtual_dry_run",
                    )
                    journal.log_entry_snapshot(
                        trade_id=virtual_id,
                        entry_time=current_minute.isoformat(),
                        instrument=INSTRUMENT,
                        side=side,
                        entry_price=ask if side == "long" else bid,
                        model_evidence=dict(decision["_v10_snapshot"]),
                        entry_bid=bid,
                        entry_ask=ask,
                        entry_spread_bps=new_trade.entry_spread_bps,
                        session=str(decision["session"]),
                        model_policy=str(decision["policy"]),
                        execution_checks=[
                            "fresh_quote",
                            "spread_within_execution_cap",
                            "exposure_safety_admitted",
                            "virtual_dry_run",
                            "learned_sizing_proof_bound",
                        ],
                        capacity_units=sizing_application["capacity_units"],
                        reference_pre_round_units=sizing_application[
                            "reference_pre_round_units"
                        ],
                        pre_round_units=sizing_application["pre_round_units"],
                        units=trade_units,
                        applied_size_multiplier=sizing_application[
                            "applied_size_multiplier"
                        ],
                        sizing_application=sizing_application,
                        atr_bps=decision["_v10_snapshot"]["atr_bps"],
                    )
                    new_trade.save(TRADE_STATE_DIR)
                    open_trades.append(new_trade)
                    LOG.info(f"[DRY] virtual trade opened  side={side}  id={virtual_id}")
                else:
                    order_result = attempt_market_entry(client, side, units=trade_units)
                    event["order_status"] = order_result["status"]
                    event["order_details"] = order_result
                    if order_result.get("status") in {
                        "filled_units_mismatch",
                        "filled_structure_mismatch",
                    }:
                        structure_mismatch = (
                            order_result.get("status")
                            == "filled_structure_mismatch"
                        )
                        mismatch_trade_id = str(
                            order_result.get("trade_id") or ""
                        ).strip()
                        observed_signed_units = order_result.get(
                            "filled_signed_units"
                        )
                        recovery_trade = SimpleNamespace(
                            trade_id=mismatch_trade_id or None,
                            side=side,
                            units=(
                                abs(int(observed_signed_units))
                                if isinstance(observed_signed_units, int)
                                and observed_signed_units != 0
                                else trade_units
                            ),
                        )
                        if not mismatch_trade_id:
                            event["order_status"] = "FILL_CONTRACT_UNTRACKED_FATAL"
                            log_journal_event(
                                daily_journal_path(args.journal_suffix), event
                            )
                            raise SystemExit(
                                "[BROKER_EXPOSURE_UNTRACKED] anomalous fill lacks "
                                "an exact tradeID; counter-order recovery is forbidden"
                            )
                        recovery_result = attempt_close_trade(client, recovery_trade)
                        event["order_status"] = "FILL_CONTRACT_MISMATCH_RECOVERY"
                        event["fill_units_mismatch_recovery"] = recovery_result
                        write_trade_alert(
                            "FILL UNITS MISMATCH "
                            f"requested={order_result.get('requested_signed_units')} "
                            f"observed={observed_signed_units!r} "
                            f"trade_id={mismatch_trade_id or '(missing)'} "
                            f"recovery={recovery_result.get('status')}"
                        )
                        log_journal_event(
                            daily_journal_path(args.journal_suffix), event
                        )
                        if structure_mismatch:
                            journal.log(
                                event_type="BROKER_RECONCILIATION_REQUIRED",
                                trade_id=mismatch_trade_id,
                                payload={
                                    "ts_utc": current_minute.isoformat(),
                                    "reason": "unexpected netting/fill structure on hedging account",
                                    "order_result": order_result,
                                    "best_effort_recovery": recovery_result,
                                },
                            )
                            raise SystemExit(
                                "[BROKER_RECONCILIATION_REQUIRED] unexpected fill "
                                "structure invalidated local trade state; operator "
                                "reconciliation is mandatory"
                            )
                        if recovery_result.get("status") not in ("closed", "filled"):
                            raise SystemExit(
                                "[BROKER_EXPOSURE_UNTRACKED] exact fill units differed "
                                "from learned sizing and recovery failed"
                            )
                        last_decision_minute = current_minute
                        consecutive_errors = 0
                        time.sleep(args.poll_seconds)
                        continue
                    if order_result.get("status") == "filled":
                        filled_trade_units = abs(
                            int(order_result["filled_signed_units"])
                        )
                        if filled_trade_units != trade_units:
                            raise SystemExit(
                                "[BROKER_FILL_UNITS_CONTRACT_BROKEN] filled status without exact units"
                            )
                        event["filled_units"] = filled_trade_units
                        raw_fill = order_result.get("raw", {}).get("orderFillTransaction", {})
                        if (
                            not raw_fill.get("tradeOpened", {}).get("tradeID")
                            or raw_fill.get("tradesClosed")
                            or raw_fill.get("tradeReduced")
                        ):
                            raise SystemExit(
                                "[BROKER_FILL_STRUCTURE_CONTRACT_BROKEN] filled status "
                                "contained netting/reduction legs"
                            )
                        fill_price = _float_or_none(order_result.get("fill_price"))
                        fill_trade_id = str(order_result.get("trade_id") or "").strip()
                        fill_transaction_id = str(
                            order_result.get("oanda_transaction_id") or ""
                        ).strip()
                        observed_bid = _float_or_none(bid)
                        observed_ask = _float_or_none(ask)
                        state_entry_bid = observed_bid
                        state_entry_ask = observed_ask
                        if fill_price is not None and fill_price > 0.0:
                            if side == "long":
                                state_entry_ask = fill_price
                            else:
                                state_entry_bid = fill_price
                        paired_entry_state_valid = bool(
                            state_entry_bid is not None
                            and state_entry_ask is not None
                            and state_entry_bid > 0.0
                            and state_entry_ask > state_entry_bid
                        )
                        if (
                            fill_price is None
                            or fill_price <= 0.0
                            or not fill_trade_id
                            or not fill_transaction_id
                            or not paired_entry_state_valid
                        ):
                            # The broker has accepted/filled an order, but the
                            # authoritative entry state needed by TradeState is
                            # incomplete. Never substitute the polling quote and
                            # then let Exit decisioning run on a fabricated entry.
                            # Immediately reduce/close the already-created exposure.
                            recovery_trade = SimpleNamespace(
                                trade_id=fill_trade_id or None,
                                side=side,
                                units=filled_trade_units,
                            )
                            recovery_result = attempt_close_trade(client, recovery_trade)
                            event["order_status"] = "FILLED_STATE_UNAVAILABLE_RECOVERY"
                            event["filled_state_unavailable"] = {
                                "fill_price": fill_price,
                                "trade_id": fill_trade_id or None,
                                "fill_transaction_id": fill_transaction_id or None,
                                "observed_bid": observed_bid,
                                "observed_ask": observed_ask,
                                "paired_entry_state_valid": paired_entry_state_valid,
                            }
                            event["filled_state_recovery"] = recovery_result
                            write_trade_alert(
                                f"FILL STATE UNAVAILABLE side={side} units={trade_units} "
                                f"trade_id={fill_trade_id or '(missing)'} — recovery={recovery_result.get('status')}"
                            )
                            journal.log(
                                event_type="FILLED_STATE_UNAVAILABLE_RECOVERY",
                                trade_id=fill_trade_id or None,
                                payload={
                                    "ts_utc": current_minute.isoformat(),
                                    "side": side,
                                    "units": trade_units,
                                    "fill_price": fill_price,
                                    "observed_bid": observed_bid,
                                    "observed_ask": observed_ask,
                                    "paired_entry_state_valid": paired_entry_state_valid,
                                    "recovery": recovery_result,
                                },
                            )
                            log_journal_event(daily_journal_path(args.journal_suffix), event)
                            if recovery_result.get("status") not in ("closed", "filled"):
                                raise SystemExit(
                                    "[BROKER_EXPOSURE_UNTRACKED] filled order lacked authoritative "
                                    "TradeState inputs and immediate close/reduce recovery failed"
                                )
                            last_decision_minute = current_minute
                            consecutive_errors = 0
                            time.sleep(args.poll_seconds)
                            continue
                        new_trade = TradeState.open(
                            entry_ts=pd.Timestamp(current_minute),
                            side=side, entry_bid=state_entry_bid, entry_ask=state_entry_ask,
                            v10_snapshot=decision["_v10_snapshot"],
                            trade_id=fill_trade_id,
                            units=filled_trade_units,
                            sizing_application=sizing_application,
                            fill_transaction_id=fill_transaction_id,
                            execution_mode="learned_broker_fill",
                        )
                        new_trade.save(TRADE_STATE_DIR)
                        open_trades.append(new_trade)

                        # ── TradeJournal: log entry lifecycle ──
                        if new_trade.trade_id:
                            v10 = dict(decision["_v10_snapshot"])
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
                                oanda_order_id=order_result.get("oanda_order_id"),
                                oanda_transaction_id=order_result.get("oanda_transaction_id"),
                                fill_price=fill_price,
                                fill_units=filled_trade_units,
                                ts_oanda=order_result.get("fill_time"),
                            )
                            journal.log_entry_snapshot(
                                trade_id=new_trade.trade_id,
                                entry_time=current_minute.isoformat(),
                                instrument=INSTRUMENT,
                                side=side,
                                entry_price=fill_price,
                                model_evidence=v10,
                                entry_bid=state_entry_bid,
                                entry_ask=state_entry_ask,
                                entry_spread_bps=new_trade.entry_spread_bps,
                                session=str(decision["session"]),
                                model_policy=str(decision["policy"]),
                                execution_checks=[
                                    "fresh_quote",
                                    "spread_within_execution_cap",
                                    "broker_state_reconciled",
                                    "exposure_safety_admitted",
                                    "learned_sizing_proof_bound",
                                ],
                                capacity_units=sizing_application["capacity_units"],
                                reference_pre_round_units=sizing_application[
                                    "reference_pre_round_units"
                                ],
                                pre_round_units=sizing_application["pre_round_units"],
                                units=filled_trade_units,
                                applied_size_multiplier=sizing_application[
                                    "applied_size_multiplier"
                                ],
                                sizing_application=sizing_application,
                                atr_bps=v10["atr_bps"],
                            )
                            journal.log_oanda_trade_update(
                                trade_id=new_trade.trade_id,
                                event_type="TRADE_OPENED_OANDA",
                                oanda_trade_id=new_trade.trade_id,
                                price=float(order_result.get("fill_price") or ask),
                                units=filled_trade_units,
                            )

                        write_trade_alert(
                            f"OPEN  trade_id={new_trade.trade_id}  side={side}  "
                            f"entry={ask if side=='long' else bid:.2f}  "
                            f"spread={spread_bps:.1f}bps  units={trade_units}  "
                            f"open_count={len(open_trades)}/{args.max_trades}  "
                            f"direction={model_direction}  "
                            f"p_long={_as_float(decision.get('p_long')):.3f}  "
                            f"p_short={_as_float(decision.get('p_short')):.3f}  "
                            f"p_flat={_as_float(decision.get('p_flat')):.3f}  "
                            f"edge={_as_float(decision.get('edge_score')):+.3f}  "
                            f"mode={decision.get('selection_score_mode','')}  "
                            f"score={_fmt_optional_float(decision.get('selection_score'), '+.2f')}  "
                            f"p_trade={_as_float(decision.get('p_trade')):.3f}  "
                            f"lat={_fmt_optional_float(decision.get('entry_signal_latency_sec'), '.0f')}s"
                        )
                        LOG.info(f"opened trade  id={new_trade.trade_id}  side={side}  "
                                  f"entry={ask if side=='long' else bid}  "
                                  f"open_count={len(open_trades)}/{args.max_trades}  "
                                  f"direction={model_direction}  "
                                  f"p_long={_as_float(decision.get('p_long')):.3f}  "
                                  f"p_short={_as_float(decision.get('p_short')):.3f}  "
                                  f"p_flat={_as_float(decision.get('p_flat')):.3f}  "
                                  f"edge={_as_float(decision.get('edge_score')):+.3f}  "
                                  f"mode={decision.get('selection_score_mode','')}  "
                                  f"score={_fmt_optional_float(decision.get('selection_score'), '+.2f')}  "
                                  f"p_trade={_as_float(decision.get('p_trade')):.3f}  "
                                  f"entry_latency={_fmt_optional_float(decision.get('entry_signal_latency_sec'), '.0f')}s")
                    elif order_result.get("status") in ("rejected", "api_error"):
                        # Reject without a trade_id — log via run-level JSONL so reject stream
                        # is preserved for triage even if no per-trade journal exists.
                        journal.log(
                            event_type="ORDER_REJECTED",
                            payload={
                                "ts_utc": current_minute.isoformat(),
                                "side": side,
                                "units": trade_units,
                                "reason": order_result.get("reason"),
                                "client_order_id": order_result.get("client_order_id"),
                                "model_action": expected_action,
                                "model_direction": model_direction,
                                "model_direction_index": expected_index,
                                "direction_probs": decision.get("direction_probs"),
                                "bid": bid, "ask": ask,
                                "n_open_trades": len(open_trades),
                            },
                        )
            else:
                event["order_status"] = "MODEL_DIRECTION_FLAT"
                event["units_mode"] = MODEL_NATIVE_SIZING_MODE_LEARNED
                event["units"] = 0

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
