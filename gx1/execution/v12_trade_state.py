#!/usr/bin/env python3
"""V12 trade-state tracker — maintains per-trade running state across M1 bars
for the Exit-IQL V12.1 per-bar exit-decision loop.

Each open trade has a TradeState that records:
  - entry timestamp + side + entry prices (bid+ask snapshot)
  - V10 snapshot at entry (frozen — used as 'v10_*_at_entry_v1' features)
  - Per-bar PnL trajectory (used for MFE/MAE/cum_peak/drawdown features)
  - Recent M1 return window (used for vol/return-since-entry features)

The Exit-IQL state vector at any M1 bar combines:
  - TradeState-derived running stats (~15 trade-state features)
  - V3 v8 outputs at this bar (4 required features)
  - V10 entry-snapshot (10 features)
  - canonical_v3 + augmented features at current bar (~170 features)

Usage:
    trade = TradeState.open_unit_normalized_research(
        entry_ts=current_minute,
        side="long",
        entry_bid=4685.0, entry_ask=4685.5,
        v10_snapshot=v10_out,
        normalization_contract="unit_normalized_direction_exit_research_v1",
    )
    # Each M1 bar after entry:
    trade.update_bar(
        bid=4686.0,
        ask=4686.5,
        m1_close=4686.25,
        bid_high=4686.3,
        bid_low=4685.8,
        ask_high=4686.8,
        ask_low=4686.3,
    )
    bar_state = exit_iql.build_bar_state(
        trade,
        canonical_v3_row=augmented_cv3.loc[now_ts],
        v3_v8_out=v3_v8_out,
        now_minute=now_ts,
    )
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    require_model_native_entry_time,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    require_model_native_sizing_application_record,
)
from gx1.features.trade_overlay import OVERLAY_COL_NAMES, compute_trade_overlay

LOG = logging.getLogger("v12_trade_state")

SIDE_LONG = "long"
SIDE_SHORT = "short"
SIDES = (SIDE_LONG, SIDE_SHORT)

PERSISTED_TRADE_STATE_SCHEMA_VERSION = "gx1_persisted_trade_state_v2"
TRADE_STATE_SIZING_EXECUTION_EVIDENCE_SCHEMA_VERSION = (
    "trade_state_sizing_execution_evidence_v1"
)
M1_RETURNS_WINDOW_MAXLEN = 120
TRAJECTORY_HISTORY_MAXLEN = 2000
_PERSISTED_TRADE_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "entry_ts",
        "side",
        "entry_bid",
        "entry_ask",
        "entry_spread_bps",
        "v10_snapshot",
        "trade_id",
        "units",
        "sizing_execution_evidence",
        "bars_in_trade",
        "current_bid",
        "current_ask",
        "current_pnl_bps",
        "cum_mfe_bps",
        "cum_mae_bps",
        "bars_since_mfe_peak",
        "m1_returns_window",
        "v3_last_prob",
        "v3_max_prob_in_trade",
        "v3_consecutive_exits",
        "v3_max_consecutive_exits",
        "v3_total_exit_decisions",
        "v3_signal_acceleration",
        "pnl_history",
        "mfe_at_bar",
        "time_since_mfe_peak_bars",
        "last_atr_bps",
        "peak_history",
        "trough_history",
        "atr_bps_history",
    }
)
_INTRABAR_HISTORY_FIELDS = frozenset(
    {"peak_history", "trough_history", "atr_bps_history"}
)
_SIZING_EXECUTION_EVIDENCE_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "executable_order_authority",
        "requested_units",
        "filled_units",
        "fill_transaction_id",
        "sizing_application",
        "research_normalization_contract",
    }
)


def _require_sizing_execution_evidence(
    value: Any,
    *,
    snapshot: dict[str, Any],
    side: str,
    units: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SIZING_EXECUTION_EVIDENCE_FIELDS:
        raise ValueError("trade sizing execution evidence exact schema mismatch")
    evidence = dict(value)
    if evidence["schema_version"] != TRADE_STATE_SIZING_EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("trade sizing execution evidence schema_version mismatch")
    if evidence["requested_units"] != units or evidence["filled_units"] != units:
        raise ValueError("trade sizing execution evidence units mismatch")
    mode = evidence["mode"]
    if mode == "unit_normalized_research_only":
        if evidence != {
            "schema_version": TRADE_STATE_SIZING_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "mode": "unit_normalized_research_only",
            "executable_order_authority": False,
            "requested_units": 1,
            "filled_units": 1,
            "fill_transaction_id": None,
            "sizing_application": None,
            "research_normalization_contract": (
                "unit_normalized_direction_exit_research_v1"
            ),
        }:
            raise ValueError("research sizing declaration is not exact/non-executable")
        return evidence
    if mode not in ("learned_virtual_dry_run", "learned_broker_fill"):
        raise ValueError("trade sizing execution mode is not executable learned sizing")
    application = require_model_native_sizing_application_record(
        evidence["sizing_application"], context="TRADE_STATE_SIZING_APPLICATION"
    )
    if application["authorized_order"] is not True or application["units"] != units:
        raise ValueError("trade units differ from authorized learned sizing application")
    expected_direction = "LONG" if side == SIDE_LONG else "SHORT"
    if application["model_direction"] != expected_direction:
        raise ValueError("trade side differs from learned sizing application direction")
    if application["sizing_authority_contract"] != snapshot["sizing_authority_contract"]:
        raise ValueError("trade sizing application authority differs from Entry snapshot")
    transaction_id = evidence["fill_transaction_id"]
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise ValueError("learned trade sizing evidence lacks fill transaction identity")
    if mode == "learned_broker_fill":
        if evidence["executable_order_authority"] is not True:
            raise ValueError("broker fill sizing evidence must be executable")
        if transaction_id.startswith("virtual:"):
            raise ValueError("broker fill cannot use virtual transaction identity")
    else:
        if evidence["executable_order_authority"] is not False:
            raise ValueError("dry-run sizing evidence must be non-executable")
        if not transaction_id.startswith("virtual:"):
            raise ValueError("dry-run sizing evidence requires virtual identity")
    if evidence["research_normalization_contract"] is not None:
        raise ValueError("learned sizing evidence cannot claim research normalization")
    return evidence


def require_model_native_entry_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete frozen Entry evidence consumed by Exit/state.

    This compatibility name delegates to the one shared runtime contract.  A
    recovered trade therefore cannot carry a smaller or older evidence schema
    than the decision, journal, and daily-review paths.
    """

    validated = require_model_native_runtime_evidence(snapshot, context="TRADE_STATE")
    if not MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS.issubset(validated):
        raise RuntimeError(
            "[TRADE_STATE_MODEL_NATIVE_TIMING_EVIDENCE_MISSING] complete executable "
            "timing evidence is required"
        )
    return validated


def _finite_persisted_number(
    payload: dict[str, Any],
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"persisted trade state {field_name} must be a JSON number"
        )
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"persisted trade state {field_name} must be finite")
    if positive and parsed <= 0.0:
        raise ValueError(f"persisted trade state {field_name} must be positive")
    if nonnegative and parsed < 0.0:
        raise ValueError(
            f"persisted trade state {field_name} must be nonnegative"
        )
    return parsed


def _nonnegative_persisted_integer(
    payload: dict[str, Any], field_name: str, *, positive: bool = False
) -> int:
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"persisted trade state {field_name} must be an exact integer"
        )
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(
            f"persisted trade state {field_name} must be {qualifier}"
        )
    return value


def _finite_persisted_history(
    payload: dict[str, Any], field_name: str, *, positive: bool = False
) -> np.ndarray:
    raw = payload[field_name]
    if not isinstance(raw, list):
        raise ValueError(f"persisted trade state {field_name} must be a list")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in raw
    ):
        raise ValueError(
            f"persisted trade state {field_name} contains non-numeric values"
        )
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all() or (positive and np.any(values <= 0.0)):
        if field_name in _INTRABAR_HISTORY_FIELDS:
            raise ValueError("persisted trade-state intrabar histories are invalid")
        raise ValueError(f"persisted trade state {field_name} is invalid")
    return values


def _validate_persisted_trade_state_payload(
    payload: dict[str, Any],
) -> tuple[pd.Timestamp, dict[str, Any]]:
    """Validate the exact JSON persistence contract without filling any field."""

    if not isinstance(payload, dict):
        raise ValueError("persisted trade state must be a JSON object")
    missing = _PERSISTED_TRADE_STATE_FIELDS - set(payload)
    unexpected = set(payload) - _PERSISTED_TRADE_STATE_FIELDS
    if missing & _INTRABAR_HISTORY_FIELDS:
        raise ValueError(
            "persisted trade state missing exact intrabar histories: "
            f"{sorted(missing & _INTRABAR_HISTORY_FIELDS)}"
        )
    if missing or unexpected:
        raise ValueError(
            "persisted trade state exact schema mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    if payload["schema_version"] != PERSISTED_TRADE_STATE_SCHEMA_VERSION:
        raise ValueError(
            "persisted trade state schema_version mismatch: "
            f"{payload['schema_version']!r} != "
            f"{PERSISTED_TRADE_STATE_SCHEMA_VERSION!r}"
        )

    raw_entry_ts = payload["entry_ts"]
    if not isinstance(raw_entry_ts, str) or not raw_entry_ts.strip():
        raise ValueError("persisted trade state entry_ts must be an ISO UTC string")
    try:
        entry_ts = pd.Timestamp(raw_entry_ts)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "persisted trade state entry_ts must be an ISO UTC string"
        ) from exc
    if (
        pd.isna(entry_ts)
        or entry_ts.tzinfo is None
        or entry_ts.utcoffset() is None
        or entry_ts.utcoffset().total_seconds() != 0.0
    ):
        raise ValueError("persisted trade state entry_ts must be timezone-aware UTC")
    entry_ts = entry_ts.tz_convert("UTC")

    side = payload["side"]
    if not isinstance(side, str) or side not in SIDES:
        raise ValueError(f"persisted trade state side must be one of {SIDES}")

    entry_bid = _finite_persisted_number(payload, "entry_bid", positive=True)
    entry_ask = _finite_persisted_number(payload, "entry_ask", positive=True)
    if entry_ask <= entry_bid:
        raise ValueError("persisted trade state entry ask must exceed entry bid")
    entry_spread_bps = _finite_persisted_number(
        payload, "entry_spread_bps", positive=True
    )
    expected_spread_bps = (entry_ask - entry_bid) / entry_bid * 10_000.0
    if not np.isclose(
        entry_spread_bps, expected_spread_bps, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(
            "persisted trade state entry_spread_bps does not match entry prices"
        )

    snapshot_raw = payload["v10_snapshot"]
    if not isinstance(snapshot_raw, dict) or not snapshot_raw:
        raise ValueError("persisted trade state v10_snapshot must be a JSON object")
    snapshot = require_model_native_entry_snapshot(dict(snapshot_raw))
    require_model_native_entry_time(
        snapshot,
        entry_ts,
        context="TRADE_STATE_PERSISTED",
    )
    expected_side = {"LONG": SIDE_LONG, "SHORT": SIDE_SHORT}.get(
        snapshot["model_direction"]
    )
    if expected_side != side:
        raise ValueError(
            "persisted trade state side does not match frozen model direction"
        )

    trade_id = payload["trade_id"]
    if trade_id is not None and (
        not isinstance(trade_id, str)
        or not trade_id
        or trade_id.strip() != trade_id
        or any(separator in trade_id for separator in ("/", "\\", "\x00"))
    ):
        raise ValueError("persisted trade state trade_id is invalid")

    persisted_units = _nonnegative_persisted_integer(payload, "units", positive=True)
    _require_sizing_execution_evidence(
        payload["sizing_execution_evidence"],
        snapshot=snapshot,
        side=side,
        units=persisted_units,
    )
    bars_in_trade = _nonnegative_persisted_integer(payload, "bars_in_trade")
    bars_since_mfe_peak = _nonnegative_persisted_integer(
        payload, "bars_since_mfe_peak"
    )
    time_since_mfe_peak_bars = _nonnegative_persisted_integer(
        payload, "time_since_mfe_peak_bars"
    )
    if bars_since_mfe_peak > bars_in_trade:
        raise ValueError(
            "persisted trade state bars_since_mfe_peak exceeds bars_in_trade"
        )
    if time_since_mfe_peak_bars != bars_since_mfe_peak:
        raise ValueError(
            "persisted trade state MFE age counters are not aligned"
        )

    current_bid = _finite_persisted_number(payload, "current_bid", positive=True)
    current_ask = _finite_persisted_number(payload, "current_ask", positive=True)
    if current_ask < current_bid:
        raise ValueError("persisted trade state current ask is below current bid")
    current_pnl_bps = _finite_persisted_number(payload, "current_pnl_bps")
    cum_mfe_bps = _finite_persisted_number(payload, "cum_mfe_bps")
    cum_mae_bps = _finite_persisted_number(payload, "cum_mae_bps")
    mfe_at_bar = _finite_persisted_number(payload, "mfe_at_bar")
    _finite_persisted_number(payload, "last_atr_bps", nonnegative=True)
    expected_current_pnl_bps = (
        (current_bid - entry_ask) / entry_ask * 10_000.0
        if side == SIDE_LONG
        else (entry_bid - current_ask) / entry_bid * 10_000.0
    )
    fresh_zero_bar_state = (
        bars_in_trade == 0
        and np.isclose(current_bid, entry_bid, rtol=0.0, atol=0.0)
        and np.isclose(current_ask, entry_ask, rtol=0.0, atol=0.0)
        and current_pnl_bps == 0.0
    )
    if not fresh_zero_bar_state and not np.isclose(
        current_pnl_bps, expected_current_pnl_bps, rtol=1e-12, atol=1e-9
    ):
        raise ValueError(
            "persisted trade state current_pnl_bps does not match current prices"
        )
    if cum_mfe_bps < -1e-9 or cum_mfe_bps + 1e-9 < current_pnl_bps:
        raise ValueError("persisted trade state cumulative MFE is invalid")
    if cum_mae_bps > 1e-9 or cum_mae_bps - 1e-9 > current_pnl_bps:
        raise ValueError("persisted trade state cumulative MAE is invalid")
    if not np.isclose(mfe_at_bar, cum_mfe_bps, rtol=1e-12, atol=1e-9):
        raise ValueError("persisted trade state mfe_at_bar does not match cumulative MFE")

    m1_values = _finite_persisted_history(
        payload, "m1_returns_window", positive=True
    )
    pnl_values = _finite_persisted_history(payload, "pnl_history")
    peak_values = _finite_persisted_history(payload, "peak_history")
    trough_values = _finite_persisted_history(payload, "trough_history")
    atr_values = _finite_persisted_history(
        payload, "atr_bps_history", positive=True
    )
    expected_m1_length = min(bars_in_trade, M1_RETURNS_WINDOW_MAXLEN)
    expected_trajectory_length = min(
        bars_in_trade, TRAJECTORY_HISTORY_MAXLEN
    )
    if len(m1_values) != expected_m1_length:
        raise ValueError(
            "persisted trade state m1_returns_window length does not match "
            "bars_in_trade/deque maxlen"
        )
    if not (
        len(pnl_values)
        == len(peak_values)
        == len(trough_values)
        == len(atr_values)
        == expected_trajectory_length
    ):
        raise ValueError(
            "persisted trade-state intrabar histories are not aligned with "
            "pnl_history/bars_in_trade/deque maxlen"
        )
    if np.any(peak_values + 1e-9 < trough_values) or np.any(
        pnl_values > peak_values + 1e-9
    ) or np.any(pnl_values < trough_values - 1e-9):
        raise ValueError("persisted trade-state intrabar histories are invalid")
    if len(pnl_values):
        if cum_mfe_bps + 1e-9 < max(0.0, float(np.max(pnl_values))):
            raise ValueError("persisted trade state cumulative MFE/history mismatch")
        if cum_mae_bps - 1e-9 > min(0.0, float(np.min(pnl_values))):
            raise ValueError("persisted trade state cumulative MAE/history mismatch")

    v3_last_prob = _finite_persisted_number(payload, "v3_last_prob")
    v3_max_prob = _finite_persisted_number(payload, "v3_max_prob_in_trade")
    if not 0.0 <= v3_last_prob <= 1.0 or not 0.0 <= v3_max_prob <= 1.0:
        raise ValueError("persisted trade state V3 probabilities are outside [0,1]")
    if v3_max_prob + 1e-12 < v3_last_prob:
        raise ValueError("persisted trade state V3 max probability is invalid")
    v3_signal_acceleration = _finite_persisted_number(
        payload, "v3_signal_acceleration"
    )
    if not -1.0 <= v3_signal_acceleration <= 1.0:
        raise ValueError("persisted trade state V3 acceleration is outside [-1,1]")
    v3_consecutive = _nonnegative_persisted_integer(
        payload, "v3_consecutive_exits"
    )
    v3_max_consecutive = _nonnegative_persisted_integer(
        payload, "v3_max_consecutive_exits"
    )
    v3_total_exits = _nonnegative_persisted_integer(
        payload, "v3_total_exit_decisions"
    )
    if not (
        v3_consecutive <= v3_max_consecutive <= v3_total_exits <= bars_in_trade
    ):
        raise ValueError("persisted trade state V3 counters are inconsistent")
    if (v3_last_prob > 0.5) != (v3_consecutive > 0):
        raise ValueError(
            "persisted trade state V3 probability/consecutive counter mismatch"
        )

    return entry_ts, snapshot


@dataclass
class TradeState:
    """Per-trade running state.

    Long entry: ask_open at entry → exit at bid_close (spread cost on both sides).
    Short entry: bid_open at entry → exit at ask_close.
    """
    entry_ts: pd.Timestamp
    side: str                            # "long" or "short"
    entry_bid: float                     # bid at entry minute
    entry_ask: float                     # ask at entry minute
    entry_spread_bps: float
    v10_snapshot: dict[str, Any]         # frozen V10 outputs at entry
    units: int
    sizing_execution_evidence: dict[str, Any]

    # Identity (set after OANDA fill — used as state-file name + close-trade id)
    trade_id: str | None = None

    # Running state (updated per M1 bar)
    bars_in_trade: int = 0
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_pnl_bps: float = 0.0         # bid/ask asymmetric
    cum_mfe_bps: float = 0.0             # running max favorable PnL
    cum_mae_bps: float = 0.0             # running min (most-adverse) PnL
    bars_since_mfe_peak: int = 0

    # Recent M1 return window (for vol/momentum features)
    m1_returns_window: deque = field(
        default_factory=lambda: deque(maxlen=M1_RETURNS_WINDOW_MAXLEN)
    )

    # V3 v8 running stats (updated per bar via update_v3)
    v3_last_prob: float = 0.0
    v3_max_prob_in_trade: float = 0.0
    v3_consecutive_exits: int = 0
    v3_max_consecutive_exits: int = 0
    v3_total_exit_decisions: int = 0
    v3_signal_acceleration: float = 0.0      # latest delta-prob bar-to-bar

    # Per-bar trajectory (used for V3 overlay + trade-state metrics)
    pnl_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )  # bps per bar
    mfe_at_bar: float = 0.0                  # mfe_bps at last bar
    time_since_mfe_peak_bars: int = 0
    last_atr_bps: float = 0.0

    # V4 (R13) intrabar excursion history — feeds the ONE-TRUTH V3 overlay helper
    # (gx1.features.trade_overlay.compute_trade_overlay), the SAME function the
    # train builder calls, so build_v3_overlay is bit-identical to training.
    # peak/trough = INTRABAR favorable/adverse excursion bps (spread-side: long
    # bid_high/bid_low, short ask_low/ask_high); atr = per-bar (ask_high-bid_low)/
    # mid*1e4. One value appended per CLOSED M1 bar, in lock-step with pnl_history.
    peak_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )
    trough_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )
    atr_bps_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )

    @classmethod
    def open(
        cls,
        entry_ts: pd.Timestamp,
        side: str,
        entry_bid: float,
        entry_ask: float,
        v10_snapshot: dict[str, Any] | None = None,
        trade_id: str | None = None,
        *,
        units: int,
        sizing_application: dict[str, Any],
        fill_transaction_id: str,
        execution_mode: str,
    ) -> "TradeState":
        """Open executable/dry-run state only from an exact learned application."""

        if isinstance(units, bool) or not isinstance(units, int) or units <= 0:
            raise ValueError("units must be an exact positive learned-sizing integer")
        if execution_mode not in ("learned_virtual_dry_run", "learned_broker_fill"):
            raise ValueError("execution_mode must be learned_virtual_dry_run/broker_fill")
        evidence = {
            "schema_version": TRADE_STATE_SIZING_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "mode": execution_mode,
            "executable_order_authority": execution_mode == "learned_broker_fill",
            "requested_units": units,
            "filled_units": units,
            "fill_transaction_id": fill_transaction_id,
            "sizing_application": sizing_application,
            "research_normalization_contract": None,
        }
        return cls._open_with_sizing_evidence(
            entry_ts=entry_ts,
            side=side,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
            v10_snapshot=v10_snapshot,
            trade_id=trade_id,
            units=units,
            sizing_execution_evidence=evidence,
        )

    @classmethod
    def open_unit_normalized_research(
        cls,
        entry_ts: pd.Timestamp,
        side: str,
        entry_bid: float,
        entry_ask: float,
        v10_snapshot: dict[str, Any] | None = None,
        trade_id: str | None = None,
        *,
        normalization_contract: str,
    ) -> "TradeState":
        """Explicit non-executable 1-unit constructor for direction/Exit research."""

        if normalization_contract != "unit_normalized_direction_exit_research_v1":
            raise ValueError("exact unit-normalized research contract is required")
        evidence = {
            "schema_version": TRADE_STATE_SIZING_EXECUTION_EVIDENCE_SCHEMA_VERSION,
            "mode": "unit_normalized_research_only",
            "executable_order_authority": False,
            "requested_units": 1,
            "filled_units": 1,
            "fill_transaction_id": None,
            "sizing_application": None,
            "research_normalization_contract": normalization_contract,
        }
        return cls._open_with_sizing_evidence(
            entry_ts=entry_ts,
            side=side,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
            v10_snapshot=v10_snapshot,
            trade_id=trade_id,
            units=1,
            sizing_execution_evidence=evidence,
        )

    @classmethod
    def _open_with_sizing_evidence(
        cls,
        *,
        entry_ts: pd.Timestamp,
        side: str,
        entry_bid: float,
        entry_ask: float,
        v10_snapshot: dict[str, Any] | None,
        trade_id: str | None,
        units: int,
        sizing_execution_evidence: dict[str, Any],
    ) -> "TradeState":
        if side not in SIDES:
            raise ValueError(f"side must be {SIDES}, got {side!r}")
        if entry_bid <= 0 or entry_ask <= 0 or entry_ask <= entry_bid:
            raise ValueError(f"invalid prices: bid={entry_bid} ask={entry_ask}")
        snapshot = require_model_native_entry_snapshot(dict(v10_snapshot or {}))
        parsed_entry_ts = pd.Timestamp(entry_ts)
        require_model_native_entry_time(
            snapshot,
            parsed_entry_ts,
            context="TRADE_STATE_OPEN",
        )
        validated_sizing_evidence = _require_sizing_execution_evidence(
            sizing_execution_evidence,
            snapshot=snapshot,
            side=side,
            units=units,
        )
        spread_bps = (entry_ask - entry_bid) / entry_bid * 10000.0
        return cls(
            entry_ts=parsed_entry_ts,
            side=str(side),
            entry_bid=float(entry_bid),
            entry_ask=float(entry_ask),
            entry_spread_bps=float(spread_bps),
            v10_snapshot=dict(snapshot),
            units=units,
            sizing_execution_evidence=validated_sizing_evidence,
            trade_id=trade_id,
            current_bid=float(entry_bid),
            current_ask=float(entry_ask),
        )

    def _pnl_bps(self, bid: float, ask: float) -> float:
        """Bid/ask asymmetric unrealized PnL in bps."""
        if self.side == SIDE_LONG:
            # entry @ ask, mark @ bid
            return (bid - self.entry_ask) / self.entry_ask * 10000.0
        else:
            # entry @ bid, mark @ ask
            return (self.entry_bid - ask) / self.entry_bid * 10000.0

    def _intrabar_excursion(
        self, bid_high: float, bid_low: float, ask_high: float, ask_low: float,
        bid_close: float, ask_close: float,
    ) -> tuple[float, float, float]:
        """Per-bar INTRABAR favorable/adverse excursion + atr in bps, IDENTICAL to
        the train builder (materialize_build_exit_iql_per_bar_dataset_v1.compute_per_bar_signals).

          long  (entry@ask): peak=(bid_high-entry_ask)/entry_ask, trough=(bid_low-entry_ask)/entry_ask
          short (entry@bid): peak=(entry_bid-ask_low)/entry_bid,  trough=(entry_bid-ask_high)/entry_bid
          atr = (ask_high-bid_low)/mid*1e4,  mid=(ask_close+bid_close)/2
        """
        if self.side == SIDE_LONG:
            peak = (bid_high - self.entry_ask) / self.entry_ask * 10000.0
            trough = (bid_low - self.entry_ask) / self.entry_ask * 10000.0
        else:
            peak = (self.entry_bid - ask_low) / self.entry_bid * 10000.0
            trough = (self.entry_bid - ask_high) / self.entry_bid * 10000.0
        mid = (ask_close + bid_close) / 2.0
        atr = (ask_high - bid_low) / mid * 10000.0 if mid > 0 else 0.0
        return float(peak), float(trough), float(atr)

    def update_bar(
        self, bid: float, ask: float, m1_close: float,
        bid_high: float, bid_low: float,
        ask_high: float, ask_low: float,
    ) -> None:
        """Advance trade state by one CLOSED M1 bar.

        bid/ask = this bar's CLOSE bid/ask (the mark prices). m1_close = mid close
        (drives the return window). bid_high/bid_low/ask_high/ask_low = this bar's
        exact intrabar range. All seven values are required, finite, positive,
        and geometrically consistent before any trade state is mutated. This is
        the V4 one-truth V3-overlay basis used by the train builder; a close-only
        substitute is not a valid Exit-IQL state.
        """
        raw_values = {
            "bid": bid,
            "ask": ask,
            "m1_close": m1_close,
            "bid_high": bid_high,
            "bid_low": bid_low,
            "ask_high": ask_high,
            "ask_low": ask_low,
        }
        values: dict[str, float] = {}
        invalid_fields: list[str] = []
        for name, raw_value in raw_values.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid_fields.append(name)
                continue
            if not np.isfinite(value) or value <= 0.0:
                invalid_fields.append(name)
            else:
                values[name] = value
        if invalid_fields:
            raise ValueError(
                f"closed M1 bar has non-finite/non-positive fields: {invalid_fields}"
            )

        bid = values["bid"]
        ask = values["ask"]
        m1_close = values["m1_close"]
        bid_high = values["bid_high"]
        bid_low = values["bid_low"]
        ask_high = values["ask_high"]
        ask_low = values["ask_low"]
        expected_mid_close = (bid + ask) / 2.0
        invalid_geometry = (
            ask <= bid
            or bid_low > bid
            or bid > bid_high
            or ask_low > ask
            or ask > ask_high
            or ask_low <= bid_low
            or ask_high <= bid_high
            or not np.isclose(m1_close, expected_mid_close, rtol=1e-9, atol=1e-9)
        )
        if invalid_geometry:
            raise ValueError(
                "closed M1 bid/ask OHLC geometry invalid: "
                f"bid_low={bid_low} bid={bid} bid_high={bid_high} "
                f"ask_low={ask_low} ask={ask} ask_high={ask_high} "
                f"m1_close={m1_close} expected_mid_close={expected_mid_close}"
            )

        self.bars_in_trade += 1
        self.m1_returns_window.append(m1_close)

        self.current_bid = bid
        self.current_ask = ask
        self.current_pnl_bps = float(self._pnl_bps(bid, ask))
        prev_peak = self.cum_mfe_bps
        self.cum_mfe_bps = max(self.cum_mfe_bps, self.current_pnl_bps)
        self.cum_mae_bps = min(self.cum_mae_bps, self.current_pnl_bps)
        if self.cum_mfe_bps > prev_peak:
            self.bars_since_mfe_peak = 0
            self.time_since_mfe_peak_bars = 0
        else:
            self.bars_since_mfe_peak += 1
            self.time_since_mfe_peak_bars += 1
        self.mfe_at_bar = self.cum_mfe_bps
        self.pnl_history.append(self.current_pnl_bps)

        # V4 (R13): exact INTRABAR excursion + per-bar atr, never close-only.
        peak, trough, atr = self._intrabar_excursion(
            bid_high, bid_low, ask_high, ask_low, bid, ask,
        )
        self.peak_history.append(float(peak))
        self.trough_history.append(float(trough))
        self.atr_bps_history.append(float(atr))

    def update_v3(self, v3_v8_out: dict | None) -> None:
        """Update V3 v8 running stats from latest V3 inference output."""
        if not v3_v8_out:
            return
        prob = float(v3_v8_out.get("v3_v8_should_exit_prob", 0.0))
        self.v3_signal_acceleration = prob - self.v3_last_prob
        self.v3_last_prob = prob
        if prob > self.v3_max_prob_in_trade:
            self.v3_max_prob_in_trade = prob
        if prob > 0.5:
            self.v3_consecutive_exits += 1
            self.v3_total_exit_decisions += 1
            if self.v3_consecutive_exits > self.v3_max_consecutive_exits:
                self.v3_max_consecutive_exits = self.v3_consecutive_exits
        else:
            self.v3_consecutive_exits = 0

    # ── feature construction ────────────────────────────────────────

    def _rolling_return_bps(self, n: int) -> float:
        """M1 close return over EXACTLY n bars (bps).

        Returns 0.0 if fewer than (n+1) bars are available — matches training
        convention where partial-window features were 0-filled at trade start.
        (Audit H2 2026-05-19: previously returned a SHORTER-lookback value
        silently, e.g. ``m1_last_60bar_return_bps_v1`` was actually a 2-bar
        return early in trade, diverging from training data.)
        """
        if len(self.m1_returns_window) < n + 1:
            return 0.0
        prev = self.m1_returns_window[-(n + 1)]
        cur = self.m1_returns_window[-1]
        if prev <= 0:
            return 0.0
        return (cur - prev) / prev * 10000.0

    def _rolling_vol_bps(self, n: int) -> float:
        """Std of M1 close-to-close returns (bps) over last n bars."""
        if len(self.m1_returns_window) < 3:
            return 0.0
        arr = np.array(list(self.m1_returns_window)[-(n + 1):], dtype=np.float64)
        if len(arr) < 3:
            return 0.0
        rets = np.diff(arr) / arr[:-1] * 10000.0
        return float(rets.std())

    def build_trade_state_features(self) -> dict[str, float]:
        """The ~13 per-bar trade-state features Exit-IQL V12.1 expects."""
        drawdown_from_peak = self.cum_mfe_bps - self.current_pnl_bps
        # bar_return_bps_v1: return of THIS bar (last - prev close)
        if len(self.m1_returns_window) >= 2:
            bar_return = (self.m1_returns_window[-1] - self.m1_returns_window[-2]) / self.m1_returns_window[-2] * 10000.0
        else:
            bar_return = 0.0

        # Trade-state derivatives — formulas MUST match training EXACTLY
        # (`materialize_build_exit_iql_per_bar_dataset_v1.py:283-322`). Live had
        # subtly wrong formulas in earlier fix (audit 3 C-1/C-2 2026-05-20):
        #   - mfe_decay_rate: training is cum_peak[t] - cum_peak[t-4]
        #     (forward growth ≥ 0), live had used (max(h[-4:]) - h[-1])/4
        #   - giveback_ratio: training clips to [-10, 10] with 1e-6 epsilon,
        #     live had clipped to [0, 2] with 1.0 epsilon (different scale).
        #   - giveback_acceleration: training is SECOND diff of giveback,
        #     live had used FIRST diff (velocity, not acceleration).
        #   - rolling_slope: training is closed-form OLS slope of pnl vs bar_idx
        #     over [0..t], live had used rolling-return divided by bars.
        h = np.asarray(list(self.pnl_history), dtype=np.float64)
        n = len(h)
        if n >= 1:
            cum_peak = np.maximum.accumulate(h)
        # pnl_velocity / pnl_acceleration: first / second discrete differences
        pnl_velocity = float(h[-1] - h[-2]) if n >= 2 else 0.0
        pnl_acceleration = float(h[-1] - 2.0 * h[-2] + h[-3]) if n >= 3 else 0.0
        # mfe_decay_rate: cum_peak[t] - cum_peak[t-4]  (monotone ≥ 0 increase)
        mfe_decay_rate = float(cum_peak[-1] - cum_peak[-5]) if n >= 5 else 0.0
        # giveback_ratio: 1 - cur_pnl/max(cum_peak, 1e-6), clipped to [-10, 10]
        if n >= 1:
            pos_peak = max(float(cum_peak[-1]), 1e-6)
            giveback_ratio = 1.0 - h[-1] / pos_peak
            giveback_ratio = float(np.clip(giveback_ratio, -10.0, 10.0))
        else:
            giveback_ratio = 0.0
        # giveback_acceleration: second diff of giveback timeseries
        if n >= 3:
            gv_t = 1.0 - h[-1] / max(float(cum_peak[-1]), 1e-6)
            gv_t1 = 1.0 - h[-2] / max(float(cum_peak[-2]), 1e-6)
            gv_t2 = 1.0 - h[-3] / max(float(cum_peak[-3]), 1e-6)
            giveback_acceleration = float(np.clip(gv_t - 2.0 * gv_t1 + gv_t2, -10.0, 10.0))
        else:
            giveback_acceleration = 0.0

        # rolling_slope_since_entry_v1: closed-form OLS slope of pnl vs bar_idx
        # over [0..n-1]. slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²).
        if n >= 3:
            idx = np.arange(n, dtype=np.float64)
            sum_x = idx.sum()
            sum_x2 = (idx * idx).sum()
            sum_y = h.sum()
            sum_xy = (idx * h).sum()
            denom = n * sum_x2 - sum_x * sum_x
            rolling_slope = float((n * sum_xy - sum_x * sum_y) / denom) if abs(denom) > 1e-9 else 0.0
        else:
            rolling_slope = 0.0

        return {
            "bars_in_trade_v1": float(self.bars_in_trade),
            "current_unrealized_pnl_bps_v1": float(self.current_pnl_bps),
            "current_mfe_bps_v1": float(self.cum_mfe_bps),
            "current_mae_bps_v1": float(self.cum_mae_bps),
            "bars_since_mfe_peak_v1": float(self.bars_since_mfe_peak),
            "pnl_drawdown_from_peak_v1": float(drawdown_from_peak),
            "bar_return_bps_v1": float(bar_return),
            "m1_last_5bar_return_bps_v1": float(self._rolling_return_bps(5)),
            "m1_last_15bar_return_bps_v1": float(self._rolling_return_bps(15)),
            "m1_last_60bar_return_bps_v1": float(self._rolling_return_bps(60)),
            "m1_realized_vol_15bar_bps_v1": float(self._rolling_vol_bps(15)),
            "m1_realized_vol_60bar_bps_v1": float(self._rolling_vol_bps(60)),
            # 6 trade-state derivatives — match training builder formulas exactly.
            "pnl_velocity_v1": float(pnl_velocity),
            "pnl_acceleration_v1": float(pnl_acceleration),
            "mfe_decay_rate_v1": float(mfe_decay_rate),
            "giveback_ratio_v1": float(giveback_ratio),
            "giveback_acceleration_v1": float(giveback_acceleration),
            "rolling_slope_since_entry_v1": float(rolling_slope),
        }

    def build_v10_entry_snapshot_features(self) -> dict[str, float]:
        """V10 outputs frozen at trade entry, exposed as exit-IQL features."""
        s = require_model_native_entry_snapshot(self.v10_snapshot)
        dp = s["direction_probs"]
        p_long_e = float(dp[0])
        p_short_e = float(dp[1])
        return {
            "v10_p_long_at_entry_v1": p_long_e,
            "v10_p_short_at_entry_v1": p_short_e,
            "v10_path_quality_at_entry_v1": float(s["path_quality"]),
            "v10_mfe_pred_at_entry_v1": float(s["mfe_first_n"]),
            "v10_tradable_at_entry_v1": float(s["tradable_prob"]),
            "v10_bad_path_at_entry_v1": float(s["bad_path_prob"]),
            # V10 v3+ aux head outputs frozen at entry — Exit-IQL was retrained
            # 2026-05-19 to consume these 4 features (208-dim state vector).
            # Without them they're silently 0-filled and Exit-IQL Q-values
            # drift from what training saw.
            "v10_tf_agreement_at_entry_v1": float(s["tf_agreement_pred"]),
            "v10_path_quality_std_at_entry_v1": float(s["path_quality_std"]),
            # Model-native sizing is evidence-only for Entry execution today,
            # but Exit state must still receive the exact learned value.  A
            # missing head is a contract failure, never a silent zero-fill.
            "v10_position_size_at_entry_v1": float(s["position_size_pred"]),
            # The active model-native contract blocks the stale hold head. Keep
            # its explicit inactive sentinel; never synthesize a neutral value.
            "v10_hold_horizon_at_entry_v1": -1.0,
            # V3 v8 frozen at entry (would be from V3 inference at entry bar)
            "p_long_entry_v1": p_long_e,
            "p_hat_entry_v1": float(max(p_long_e, p_short_e, 1.0 - p_long_e - p_short_e)),
            "uncertainty_entry_v1": float(1.0 - max(p_long_e, p_short_e, 1.0 - p_long_e - p_short_e)),
            # margin = top1-top2 gap of the 3-class probs — matches the candidate-gen
            # (sorted[-1]-sorted[-2]) and the Exit-IQL state builder
            # (materialize_build_exit_iql_per_bar_dataset_v2_m1.py:247 candidate_row["margin"]).
            # NOT abs(p_long-p_short) (wrong when p_flat is top1).
            "margin_entry_v1": float(
                sorted((p_long_e, p_short_e, max(0.0, 1.0 - p_long_e - p_short_e)))[-1]
                - sorted((p_long_e, p_short_e, max(0.0, 1.0 - p_long_e - p_short_e)))[-2]),
            # entropy
            "entropy_entry_v1": float(
                _shannon_entropy([p_long_e, p_short_e, float(dp[2])])
            ),
            # rolling_slope_since_entry_v1 is now produced by build_trade_state_features()
            # using closed-form OLS slope over pnl_history (matches training).
        }

    def build_side_one_hot(self) -> dict[str, float]:
        return {
            "side_v1_long": 1.0 if self.side == SIDE_LONG else 0.0,
            "side_v1_short": 1.0 if self.side == SIDE_SHORT else 0.0,
        }

    def build_v3_tracking_features(self) -> dict[str, float]:
        """The 7 V3-tracking features Exit-IQL V12.1.1 expects in its state vector.

        Computed as running stats over v3_v8_should_exit_prob across the trade's
        bars (per V12 Phase 4 spec):

          v3_should_exit_decision_v1  = (latest prob > 0.5)
          v3_decision_confidence_v1   = |latest prob - 0.5|
          v3_max_prob_in_trade_v1     = running max of prob since entry
          v3_consecutive_exits_v1     = current consecutive bars with prob > 0.5
          v3_signal_acceleration_v1   = Δ prob bar-to-bar
          v3_total_exit_decisions_v1  = count of bars with prob > 0.5
          v3_max_consecutive_exits_v1 = max run anywhere in trade
        """
        prob = self.v3_last_prob
        return {
            "v3_should_exit_decision_v1": 1.0 if prob > 0.5 else 0.0,
            "v3_decision_confidence_v1": float(abs(prob - 0.5)),
            "v3_max_prob_in_trade_v1": float(self.v3_max_prob_in_trade),
            "v3_consecutive_exits_v1": float(self.v3_consecutive_exits),
            "v3_signal_acceleration_v1": float(self.v3_signal_acceleration),
            "v3_total_exit_decisions_v1": float(self.v3_total_exit_decisions),
            "v3_max_consecutive_exits_v1": float(self.v3_max_consecutive_exits),
        }

    def build_v3_overlay(self) -> dict[str, np.ndarray]:
        """Build the 19-feature trade-state overlay for V3's in-trade portion of
        the 512-bar window, via the ONE-TRUTH helper
        (`gx1.features.trade_overlay.compute_trade_overlay`). Any future V3
        training builder must call that owner and prove build==serve parity.
        Replaces the pre-V4 "MVP" overlay
        that approximated ~10 of the 19 slots (close-mark MFE instead of intrabar,
        1-based bars_held, wrong mfe_decay lag/giveback formula, slope=pnl/bars).

        Arrays have length = number of recorded in-trade bars (one per CLOSED M1
        bar, peak/trough/atr in lock-step with pnl_history). The consumer
        (v12_v3_live) RIGHT-ALIGNS them onto the END of the 512-bar window and
        uses the last min(len, 512) values.
        """
        n = len(self.pnl_history)
        if n == 0:
            # No closed bar yet — emit a single zero row (consumer right-aligns).
            return {name: np.zeros(1, dtype=np.float32) for name in OVERLAY_COL_NAMES}

        # peak/trough/atr are appended and persisted in lock-step with pnl_history.
        # Reload rejects older or partial state instead of reconstructing evidence.
        peak = np.fromiter(self.peak_history, dtype=np.float64, count=n)
        trough = np.fromiter(self.trough_history, dtype=np.float64, count=n)
        cur_pnl = np.fromiter(self.pnl_history, dtype=np.float64, count=n)
        atr = np.fromiter(self.atr_bps_history, dtype=np.float64, count=n)

        # Entry-snapshot (V10 direction softmax @entry, frozen). margin = top1-top2
        # gap of the model-native 3-class probabilities. This is deliberately
        # NOT abs(p_long-p_short).
        s = require_model_native_entry_snapshot(self.v10_snapshot)
        dp = s["direction_probs"]
        p_long_e = float(dp[0])
        p_short_e = float(dp[1])
        p_flat_e = float(dp[2])
        p_hat_e = max(p_long_e, p_short_e, p_flat_e)
        _sorted = sorted((p_long_e, p_short_e, p_flat_e))
        margin_e = _sorted[-1] - _sorted[-2]
        uncertainty_e = 1.0 - p_hat_e
        entropy_e = _shannon_entropy([p_long_e, p_short_e, p_flat_e])

        overlay = compute_trade_overlay(peak, trough, cur_pnl, atr, {
            "p_long_entry": p_long_e,
            "p_hat_entry": p_hat_e,
            "uncertainty_entry": uncertainty_e,
            "entropy_entry": entropy_e,
            "margin_entry": margin_e,
        })
        # Consumer maps by name → return one (n,) array per overlay column.
        return {name: overlay[:, i] for i, name in enumerate(OVERLAY_COL_NAMES)}

    # ── persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize only the exact, versioned, reload-valid state schema."""
        payload = _jsonable({
            "schema_version": PERSISTED_TRADE_STATE_SCHEMA_VERSION,
            "entry_ts": self.entry_ts.isoformat(),
            "side": self.side,
            "entry_bid": self.entry_bid,
            "entry_ask": self.entry_ask,
            "entry_spread_bps": self.entry_spread_bps,
            "v10_snapshot": _jsonable(self.v10_snapshot),
            "trade_id": self.trade_id,
            "units": self.units,
            "sizing_execution_evidence": _jsonable(
                self.sizing_execution_evidence
            ),
            "bars_in_trade": self.bars_in_trade,
            "current_bid": self.current_bid,
            "current_ask": self.current_ask,
            "current_pnl_bps": self.current_pnl_bps,
            "cum_mfe_bps": self.cum_mfe_bps,
            "cum_mae_bps": self.cum_mae_bps,
            "bars_since_mfe_peak": self.bars_since_mfe_peak,
            "m1_returns_window": list(self.m1_returns_window),
            # V3 tracking stats
            "v3_last_prob": self.v3_last_prob,
            "v3_max_prob_in_trade": self.v3_max_prob_in_trade,
            "v3_consecutive_exits": self.v3_consecutive_exits,
            "v3_max_consecutive_exits": self.v3_max_consecutive_exits,
            "v3_total_exit_decisions": self.v3_total_exit_decisions,
            "v3_signal_acceleration": self.v3_signal_acceleration,
            # Per-bar trajectory
            "pnl_history": list(self.pnl_history),
            "mfe_at_bar": self.mfe_at_bar,
            "time_since_mfe_peak_bars": self.time_since_mfe_peak_bars,
            "last_atr_bps": self.last_atr_bps,
            # V4 (R13) intrabar excursion history — in lock-step with pnl_history.
            "peak_history": list(self.peak_history),
            "trough_history": list(self.trough_history),
            "atr_bps_history": list(self.atr_bps_history),
        })
        if not isinstance(payload, dict):  # pragma: no cover - fixed literal shape
            raise AssertionError("trade-state serialization did not produce an object")
        _validate_persisted_trade_state_payload(payload)
        return payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeState":
        """Rehydrate only an exact versioned state; never fill missing evidence."""
        entry_ts, snapshot = _validate_persisted_trade_state_payload(d)
        t = cls(
            entry_ts=entry_ts,
            side=d["side"],
            entry_bid=float(d["entry_bid"]),
            entry_ask=float(d["entry_ask"]),
            entry_spread_bps=float(d["entry_spread_bps"]),
            v10_snapshot=dict(snapshot),
            trade_id=d["trade_id"],
            units=d["units"],
            sizing_execution_evidence=dict(d["sizing_execution_evidence"]),
            bars_in_trade=d["bars_in_trade"],
            current_bid=float(d["current_bid"]),
            current_ask=float(d["current_ask"]),
            current_pnl_bps=float(d["current_pnl_bps"]),
            cum_mfe_bps=float(d["cum_mfe_bps"]),
            cum_mae_bps=float(d["cum_mae_bps"]),
            bars_since_mfe_peak=d["bars_since_mfe_peak"],
            v3_last_prob=float(d["v3_last_prob"]),
            v3_max_prob_in_trade=float(d["v3_max_prob_in_trade"]),
            v3_consecutive_exits=d["v3_consecutive_exits"],
            v3_max_consecutive_exits=d["v3_max_consecutive_exits"],
            v3_total_exit_decisions=d["v3_total_exit_decisions"],
            v3_signal_acceleration=float(d["v3_signal_acceleration"]),
            mfe_at_bar=float(d["mfe_at_bar"]),
            time_since_mfe_peak_bars=d["time_since_mfe_peak_bars"],
            last_atr_bps=float(d["last_atr_bps"]),
        )
        t.m1_returns_window.extend(float(v) for v in d["m1_returns_window"])
        t.pnl_history.extend(float(v) for v in d["pnl_history"])
        t.peak_history.extend(float(v) for v in d["peak_history"])
        t.trough_history.extend(float(v) for v in d["trough_history"])
        t.atr_bps_history.extend(float(v) for v in d["atr_bps_history"])
        return t

    def save(self, path: Path) -> None:
        """Atomically write trade state to disk (so an interrupted write can't corrupt).

        `path` may be either a file path or a directory (when using multi-trade
        persistence). If a directory is given, the filename is derived from
        `trade_id` via `state_filename()`.
        """
        if path.is_dir():
            path = path / self.state_filename()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), default=str, indent=2))
        os.replace(tmp, path)

    def state_filename(self) -> str:
        """Filename for this trade's state file (one file per trade)."""
        tid = self.trade_id or f"virtual_{self.entry_ts.strftime('%Y%m%dT%H%M%S')}"
        return f"open_trade_{tid}.json"

    def delete_state_file(self, directory: Path) -> None:
        """Remove this trade's persisted state file from the directory."""
        (directory / self.state_filename()).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "TradeState | None":
        """Load a saved trade state, or fail closed on corrupt/stale evidence."""
        if not path.is_file():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            raise RuntimeError(f"failed to load trade state from {path}: {exc}") from exc

    @classmethod
    def load_all(cls, directory: Path,
                 legacy_single_file: Path | None = None) -> list["TradeState"]:
        """Load all exact persisted open trades from `directory`.

        ``legacy_single_file`` denotes only the retired *location*. Its content
        must already satisfy the current versioned schema before it can move;
        unversioned state is not upgraded or backfilled.

        Returns trades sorted by entry_ts ascending.
        """
        directory.mkdir(parents=True, exist_ok=True)
        trades: list[TradeState] = []

        if legacy_single_file is not None and legacy_single_file.is_file():
            t = cls.load(legacy_single_file)
            if t is not None:
                target = directory / t.state_filename()
                if target.resolve() != legacy_single_file.resolve() and target.exists():
                    raise RuntimeError(
                        "refusing to overwrite existing exact trade state while "
                        f"moving retired location: {target}"
                    )
                if target.resolve() != legacy_single_file.resolve():
                    t.save(target)
                    legacy_single_file.unlink()
                    LOG.info(
                        "migrated exact trade state location %s → %s",
                        legacy_single_file,
                        target,
                    )

        seen_identities: set[tuple[str, str]] = set()
        for p in sorted(directory.glob("open_trade_*.json")):
            t = cls.load(p)
            if t is not None:
                expected_filename = t.state_filename()
                if p.name != expected_filename:
                    raise RuntimeError(
                        "persisted trade-state filename/identity mismatch: "
                        f"{p.name!r} != {expected_filename!r}"
                    )
                identity = (
                    ("trade_id", t.trade_id)
                    if t.trade_id is not None
                    else ("entry_ts", t.entry_ts.isoformat())
                )
                if identity in seen_identities:
                    raise RuntimeError(
                        f"duplicate persisted open-trade identity: {identity!r}"
                    )
                seen_identities.add(identity)
                trades.append(t)
        trades.sort(key=lambda x: x.entry_ts)
        return trades


def _jsonable(o):
    """Recursively convert numpy/pandas/etc. to JSON-safe types."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


def _shannon_entropy(probs):
    s = 0.0
    for p in probs:
        if p > 1e-12:
            s -= p * float(np.log(p))
    return s
