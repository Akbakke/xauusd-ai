#!/usr/bin/env python3
"""Exact per-trade state for the unified model's closed-M1 Exit path.

Each open trade has a TradeState that records:
  - entry timestamp + side + entry prices (bid+ask snapshot)
  - the exact model-native Entry snapshot, frozen at Entry
  - ordered literal mid/bid/ask OHLCV path with immutable source identity
  - exact executable intrabar excursion and spread-inclusive one-bar range
  - content-hashable path evidence with no synthetic early-history features
"""
from __future__ import annotations

import json
import logging
import os
import hashlib
from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_OPTIONAL_TIMING_FIELDS,
    require_model_native_entry_time,
    require_model_native_exit_replay_entry_time,
    require_model_native_fill_time,
    require_model_native_runtime_head_evidence,
    require_model_native_runtime_evidence,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    require_model_native_sizing_application_record,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
)
from gx1.contracts.entry_decision_token_v1 import (
    ENTRY_DECISION_TOKEN_KEY,
    build_entry_decision_token_snapshot,
    require_entry_decision_token_bindings,
    require_entry_decision_token_snapshot,
)
from gx1.models.entry_v10.direction_decision_contract import (
    CLOSED_M1_PATH_FIELDS,
    CLOSED_M1_PATH_SCHEMA_VERSION,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_CHAIN_GENESIS_SHA256,
    UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
    canonical_closed_m1_bar,
    canonical_closed_m1_path_sha256,
    canonical_closed_m1_full_path_chain_sha256,
    canonical_unified_evidence_sha256,
    extend_closed_m1_path_chain_sha256,
    require_model_direction_operating_point,
    require_unified_exit_output,
    require_unified_exit_path_envelope,
)
from gx1.contracts.unified_exit_input_v1 import (
    require_unified_exit_input_envelope,
)
from gx1.contracts.unified_exit_incremental_carry_v1 import (
    UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256,
    require_unified_exit_incremental_carry_envelope,
)
LOG = logging.getLogger("v12_trade_state")

SIDE_LONG = "long"
SIDE_SHORT = "short"
SIDES = (SIDE_LONG, SIDE_SHORT)

PERSISTED_TRADE_STATE_SCHEMA_VERSION = "gx1_persisted_trade_state_v13"
TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION = (
    "gx1_trade_state_model_bundle_binding_v2"
)
TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION = (
    "gx1_trade_state_source_pair_binding_v1"
)
TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION = (
    "gx1_trade_state_broker_account_binding_v1"
)
TRADE_STATE_SIZING_EXECUTION_EVIDENCE_SCHEMA_VERSION = (
    "trade_state_sizing_execution_evidence_v1"
)
M1_RETURNS_WINDOW_MAXLEN = 120
TRAJECTORY_HISTORY_MAXLEN = UNIFIED_EXIT_MAX_PATH_BARS


def first_full_closed_m1_bar_ts(entry_fill_ts: pd.Timestamp) -> pd.Timestamp:
    """Return the first M1 bar whose full interval starts at/after the fill."""

    parsed = pd.Timestamp(entry_fill_ts)
    if (
        pd.isna(parsed)
        or parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0.0
    ):
        raise ValueError("entry fill timestamp must be timezone-aware UTC")
    return parsed.tz_convert("UTC").ceil("min")


_PERSISTED_TRADE_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "entry_ts",
        "side",
        "entry_bid",
        "entry_ask",
        "entry_spread_bps",
        "v10_snapshot",
        "entry_decision_token_snapshot",
        "trade_id",
        "units",
        "sizing_execution_evidence",
        "model_bundle_binding",
        "entry_source_pair_binding",
        "broker_account_binding",
        "bars_in_trade",
        "last_processed_m1_ts",
        "current_bid",
        "current_ask",
        "current_pnl_bps",
        "cum_mfe_bps",
        "cum_mae_bps",
        "bars_since_mfe_peak",
        "m1_returns_window",
        "pnl_history",
        "mfe_at_bar",
        "time_since_mfe_peak_bars",
        "last_executable_range_bps",
        "peak_history",
        "trough_history",
        "executable_range_bps_history",
        "closed_m1_path",
        "full_path_chain_sha256",
        "last_exit_input_envelope",
        "last_exit_decision",
        "exit_incremental_carry_envelope",
    }
)
_MODEL_BUNDLE_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "bundle_dir",
        "bundle_sha256",
        "input_normalization_sha256",
        "contract_mode",
        "operating_point",
    }
)
_SOURCE_PAIR_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "pair_generation_id",
        "pair_manifest_sha256",
    }
)
_BROKER_ACCOUNT_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "account_id_sha256",
    }
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_INTRABAR_HISTORY_FIELDS = frozenset(
    {
        "peak_history",
        "trough_history",
        "executable_range_bps_history",
        "closed_m1_path",
    }
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


def _require_sha256(value: object, *, label: str) -> str:
    parsed = str(value or "")
    if (
        len(parsed) != 64
        or parsed.lower() != parsed
        or any(character not in _SHA256_CHARACTERS for character in parsed)
    ):
        raise ValueError(f"{label} must be a lowercase sha256")
    return parsed


def require_trade_model_bundle_binding(
    value: Any,
    *,
    executable: bool,
) -> dict[str, Any] | None:
    """Validate the immutable same-bundle recovery authority."""

    if not executable:
        if value is not None:
            raise ValueError(
                "non-executable trade state cannot claim a model bundle"
            )
        return None
    if not isinstance(value, dict) or set(value) != _MODEL_BUNDLE_BINDING_FIELDS:
        raise ValueError("trade model bundle binding exact schema mismatch")
    if (
        value["schema_version"]
        != TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION
    ):
        raise ValueError("trade model bundle binding schema_version mismatch")
    raw_path = value["bundle_dir"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("trade model bundle path is invalid")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("trade model bundle path must be absolute/normalized")
    operating_point = require_model_direction_operating_point(
        value["operating_point"],
        context="TRADE_STATE_MODEL_BUNDLE_BINDING",
    )
    if value["contract_mode"] != MODEL_NATIVE_CONTRACT_MODE:
        raise ValueError("trade model bundle contract mode mismatch")
    return {
        "schema_version": TRADE_STATE_MODEL_BUNDLE_BINDING_SCHEMA_VERSION,
        "bundle_dir": str(path),
        "bundle_sha256": _require_sha256(
            value["bundle_sha256"],
            label="trade model bundle",
        ),
        "input_normalization_sha256": _require_sha256(
            value["input_normalization_sha256"],
            label="trade input normalization",
        ),
        "contract_mode": value["contract_mode"],
        "operating_point": operating_point,
    }


def require_trade_source_pair_binding(
    value: Any,
    *,
    executable: bool,
) -> dict[str, Any] | None:
    """Validate the exact source pair used for the Entry inference."""

    if not executable:
        if value is not None:
            raise ValueError(
                "non-executable trade state cannot claim a source pair"
            )
        return None
    if not isinstance(value, dict) or set(value) != _SOURCE_PAIR_BINDING_FIELDS:
        raise ValueError("trade source pair binding exact schema mismatch")
    if (
        value["schema_version"]
        != TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION
    ):
        raise ValueError("trade source pair binding schema_version mismatch")
    return {
        "schema_version": TRADE_STATE_SOURCE_PAIR_BINDING_SCHEMA_VERSION,
        "pair_generation_id": _require_sha256(
            value["pair_generation_id"],
            label="trade source pair generation",
        ),
        "pair_manifest_sha256": _require_sha256(
            value["pair_manifest_sha256"],
            label="trade source pair manifest",
        ),
    }


def build_trade_broker_account_binding(
    *,
    environment: str,
    account_id: str,
) -> dict[str, str]:
    """Build a non-secret exact identity for one OANDA account authority."""

    if environment not in {"practice", "live"}:
        raise ValueError("broker account environment must be practice/live")
    if (
        not isinstance(account_id, str)
        or not account_id
        or account_id.strip() != account_id
    ):
        raise ValueError("broker account id must be a non-empty exact string")
    return {
        "schema_version": TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION,
        "environment": environment,
        "account_id_sha256": hashlib.sha256(
            account_id.encode("utf-8")
        ).hexdigest(),
    }


def require_trade_broker_account_binding(
    value: Any,
    *,
    execution_mode: str,
) -> dict[str, str] | None:
    """Require exact environment/account ownership for broker exposure."""

    if execution_mode != "learned_broker_fill":
        if value is not None:
            raise ValueError(
                "non-broker trade state cannot claim a broker account"
            )
        return None
    if (
        not isinstance(value, dict)
        or set(value) != _BROKER_ACCOUNT_BINDING_FIELDS
        or value.get("schema_version")
        != TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION
        or value.get("environment") not in {"practice", "live"}
    ):
        raise ValueError("trade broker account binding exact schema mismatch")
    return {
        "schema_version": TRADE_STATE_BROKER_ACCOUNT_BINDING_SCHEMA_VERSION,
        "environment": value["environment"],
        "account_id_sha256": _require_sha256(
            value.get("account_id_sha256"),
            label="trade broker account id",
        ),
    }


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


def _require_trade_entry_snapshot(
    snapshot: dict[str, Any],
    *,
    sizing_execution_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Admit a pre-sizing envelope only for non-executable Exit research."""

    if (
        sizing_execution_evidence.get("mode") == "unit_normalized_research_only"
        and "runtime_head_evidence_schema_version" in snapshot
    ):
        return require_model_native_runtime_head_evidence(
            snapshot,
            context="TRADE_STATE_EXIT_RESEARCH",
        )
    return require_model_native_entry_snapshot(snapshot)


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
) -> tuple[pd.Timestamp, dict[str, Any], pd.Timestamp | None]:
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
    sizing_evidence_raw = payload["sizing_execution_evidence"]
    if not isinstance(sizing_evidence_raw, dict):
        raise ValueError("persisted trade sizing execution evidence must be an object")
    snapshot = _require_trade_entry_snapshot(
        dict(snapshot_raw),
        sizing_execution_evidence=sizing_evidence_raw,
    )
    if "runtime_head_evidence_schema_version" in snapshot:
        require_model_native_exit_replay_entry_time(
            snapshot,
            entry_ts,
            context="TRADE_STATE_PERSISTED_EXIT_RESEARCH",
        )
    elif sizing_evidence_raw.get("mode") == "unit_normalized_research_only":
        require_model_native_entry_time(
            snapshot,
            entry_ts,
            context="TRADE_STATE_PERSISTED_RESEARCH",
        )
    else:
        require_model_native_fill_time(
            snapshot,
            entry_ts,
            context="TRADE_STATE_PERSISTED_FILL",
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
    validated_sizing_evidence = _require_sizing_execution_evidence(
        payload["sizing_execution_evidence"],
        snapshot=snapshot,
        side=side,
        units=persisted_units,
    )
    executable = validated_sizing_evidence["mode"] in {
        "learned_virtual_dry_run",
        "learned_broker_fill",
    }
    bundle_binding = require_trade_model_bundle_binding(
        payload["model_bundle_binding"],
        executable=executable,
    )
    source_pair_binding = require_trade_source_pair_binding(
        payload["entry_source_pair_binding"],
        executable=executable,
    )
    require_trade_broker_account_binding(
        payload["broker_account_binding"],
        execution_mode=validated_sizing_evidence["mode"],
    )
    if (
        payload["model_bundle_binding"] != bundle_binding
        or payload["entry_source_pair_binding"] != source_pair_binding
    ):
        raise ValueError("persisted trade immutable binding is not canonical")
    if not isinstance(trade_id, str):
        raise ValueError(
            "persisted trade identity is required for the Entry-decision token"
        )
    if bundle_binding is not None:
        model_identity_kind = "bundle_sha256"
        model_identity_sha256 = bundle_binding["bundle_sha256"]
        input_normalization_sha256 = bundle_binding[
            "input_normalization_sha256"
        ]
        contract_mode = bundle_binding["contract_mode"]
    else:
        research_token = require_entry_decision_token_snapshot(
            payload["entry_decision_token_snapshot"]
        )
        model_identity_kind = research_token["model_identity_kind"]
        model_identity_sha256 = research_token["model_identity_sha256"]
        input_normalization_sha256 = research_token[
            "input_normalization_sha256"
        ]
        contract_mode = research_token["contract_mode"]
        if (
            model_identity_kind == "training_state_sha256"
            and model_identity_sha256
            != canonical_unified_evidence_sha256(snapshot)
        ):
            raise ValueError(
                "persisted research token differs from Entry state"
            )
    try:
        token_snapshot = require_entry_decision_token_bindings(
            payload["entry_decision_token_snapshot"],
            raw_token_alias=snapshot[ENTRY_DECISION_TOKEN_KEY],
            decision_time=snapshot["decision_ts"],
            fill_time=entry_ts,
            model_identity_kind=model_identity_kind,
            model_identity_sha256=model_identity_sha256,
            input_normalization_sha256=input_normalization_sha256,
            contract_mode=contract_mode,
            model_direction_index=int(snapshot["model_direction_index"]),
            model_direction=str(snapshot["model_direction"]),
            side=side,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
            trade_identity=trade_id,
            context="TRADE_STATE_PERSISTED",
        )
    except RuntimeError as exc:
        raise ValueError("persisted Entry-decision token is invalid") from exc
    if token_snapshot != payload["entry_decision_token_snapshot"]:
        raise ValueError("persisted Entry-decision token is not canonical")
    bars_in_trade = _nonnegative_persisted_integer(payload, "bars_in_trade")
    raw_last_processed = payload["last_processed_m1_ts"]
    if raw_last_processed is None:
        last_processed_m1_ts = None
    elif isinstance(raw_last_processed, str) and raw_last_processed.strip():
        try:
            last_processed_m1_ts = pd.Timestamp(raw_last_processed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "persisted trade state last_processed_m1_ts must be ISO UTC"
            ) from exc
        if (
            pd.isna(last_processed_m1_ts)
            or last_processed_m1_ts.tzinfo is None
            or last_processed_m1_ts.utcoffset() is None
            or last_processed_m1_ts.utcoffset().total_seconds() != 0.0
            or last_processed_m1_ts != last_processed_m1_ts.floor("min")
        ):
            raise ValueError(
                "persisted trade state last_processed_m1_ts must be an exact UTC minute"
            )
        last_processed_m1_ts = last_processed_m1_ts.tz_convert("UTC")
    else:
        raise ValueError(
            "persisted trade state last_processed_m1_ts must be ISO UTC or null"
        )
    if (bars_in_trade == 0) != (last_processed_m1_ts is None):
        raise ValueError(
            "persisted trade state last_processed_m1_ts/bar count mismatch"
        )
    if (
        last_processed_m1_ts is not None
        and last_processed_m1_ts < first_full_closed_m1_bar_ts(entry_ts)
    ):
        raise ValueError(
            "persisted trade state M1 row clock precedes entry"
        )
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
    _finite_persisted_number(
        payload, "last_executable_range_bps", nonnegative=True
    )
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
    executable_range_values = _finite_persisted_history(
        payload, "executable_range_bps_history", positive=True
    )
    raw_path = payload["closed_m1_path"]
    if not isinstance(raw_path, list):
        raise ValueError("persisted trade state closed_m1_path must be a list")
    canonical_path: list[dict[str, Any]] = []
    for row in raw_path:
        if not isinstance(row, dict) or tuple(row) != CLOSED_M1_PATH_FIELDS:
            raise ValueError(
                "persisted trade state closed_m1_path row exact schema mismatch"
            )
        canonical = canonical_closed_m1_bar(
            m1_bar_ts=pd.Timestamp(row["time"]),
            complete=row["complete"],
            source_path=row["source_path"],
            source_sha256=row["source_sha256"],
            bid_open=row["bid_open"],
            bid_high=row["bid_high"],
            bid_low=row["bid_low"],
            bid_close=row["bid_close"],
            ask_open=row["ask_open"],
            ask_high=row["ask_high"],
            ask_low=row["ask_low"],
            ask_close=row["ask_close"],
            mid_open=row["mid_open"],
            mid_high=row["mid_high"],
            mid_low=row["mid_low"],
            mid_close=row["mid_close"],
            volume=row["volume"],
        )
        if row != canonical:
            raise ValueError(
                "persisted trade state closed_m1_path row is not canonical"
            )
        canonical_path.append(canonical)
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
        == len(executable_range_values)
        == expected_trajectory_length
    ):
        raise ValueError(
            "persisted trade-state intrabar histories are not aligned with "
            "pnl_history/bars_in_trade/deque maxlen"
        )
    if len(canonical_path) != expected_trajectory_length:
        raise ValueError(
            "persisted trade state retained literal path length is invalid"
        )
    full_path_chain_sha256 = payload["full_path_chain_sha256"]
    if (
        not isinstance(full_path_chain_sha256, str)
        or len(full_path_chain_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in full_path_chain_sha256
        )
        or (
            bars_in_trade == 0
            and full_path_chain_sha256
            != UNIFIED_EXIT_PATH_CHAIN_GENESIS_SHA256
        )
        or (
            0 < bars_in_trade <= UNIFIED_EXIT_MAX_PATH_BARS
            and full_path_chain_sha256
            != canonical_closed_m1_full_path_chain_sha256(canonical_path)
        )
    ):
        raise ValueError("persisted trade state full path chain is invalid")
    if canonical_path:
        minimum_first = first_full_closed_m1_bar_ts(entry_ts)
        previous_ts: pd.Timestamp | None = None
        for row in canonical_path:
            observed_ts = pd.Timestamp(row["time"])
            if (
                (
                    previous_ts is None
                    and (
                        observed_ts < minimum_first
                        or (
                            bars_in_trade <= UNIFIED_EXIT_MAX_PATH_BARS
                            and observed_ts != minimum_first
                        )
                    )
                )
                or (previous_ts is not None and observed_ts <= previous_ts)
            ):
                raise ValueError(
                    "persisted trade state closed_m1_path row clock mismatch"
                )
            previous_ts = observed_ts
        if pd.Timestamp(canonical_path[-1]["time"]) != last_processed_m1_ts:
            raise ValueError(
                "persisted trade state closed_m1_path terminal timestamp mismatch"
            )
        derived_path = canonical_path
        path_bid_close = np.asarray(
            [row["bid_close"] for row in derived_path], dtype=np.float64
        )
        path_ask_close = np.asarray(
            [row["ask_close"] for row in derived_path], dtype=np.float64
        )
        expected_pnl = (
            (path_bid_close - entry_ask) / entry_ask * 10_000.0
            if side == SIDE_LONG
            else (entry_bid - path_ask_close) / entry_bid * 10_000.0
        )
        expected_peak = (
            (
                np.asarray(
                    [row["bid_high"] for row in derived_path],
                    dtype=np.float64,
                )
                - entry_ask
            )
            / entry_ask
            * 10_000.0
            if side == SIDE_LONG
            else (
                entry_bid
                - np.asarray(
                    [row["ask_low"] for row in derived_path],
                    dtype=np.float64,
                )
            )
            / entry_bid
            * 10_000.0
        )
        expected_trough = (
            (
                np.asarray(
                    [row["bid_low"] for row in derived_path],
                    dtype=np.float64,
                )
                - entry_ask
            )
            / entry_ask
            * 10_000.0
            if side == SIDE_LONG
            else (
                entry_bid
                - np.asarray(
                    [row["ask_high"] for row in derived_path],
                    dtype=np.float64,
                )
            )
            / entry_bid
            * 10_000.0
        )
        expected_range = np.asarray(
            [
                (row["ask_high"] - row["bid_low"])
                / row["mid_close"]
                * 10_000.0
                for row in derived_path
            ],
            dtype=np.float64,
        )
        if not (
            np.allclose(pnl_values, expected_pnl, rtol=1e-12, atol=1e-9)
            and np.allclose(
                peak_values, expected_peak, rtol=1e-12, atol=1e-9
            )
            and np.allclose(
                trough_values, expected_trough, rtol=1e-12, atol=1e-9
            )
            and np.allclose(
                executable_range_values,
                expected_range,
                rtol=1e-12,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "persisted trade-state derived histories do not match literal path"
            )
        if not np.isclose(
            current_bid, canonical_path[-1]["bid_close"], rtol=0.0, atol=0.0
        ) or not np.isclose(
            current_ask, canonical_path[-1]["ask_close"], rtol=0.0, atol=0.0
        ):
            raise ValueError(
                "persisted trade-state current quote does not match literal path"
            )
        expected_recent_mid = np.asarray(
            [row["mid_close"] for row in canonical_path[-len(m1_values) :]],
            dtype=np.float64,
        )
        if not np.allclose(
            m1_values, expected_recent_mid, rtol=0.0, atol=0.0
        ):
            raise ValueError(
                "persisted trade-state M1 close window does not match literal path"
            )
        if not np.isclose(
            payload["last_executable_range_bps"],
            executable_range_values[-1],
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ValueError(
                "persisted last executable range does not match literal path"
            )
    elif payload["last_executable_range_bps"] != 0.0:
        raise ValueError(
            "zero-bar persisted state must have zero executable range"
        )
    if np.any(peak_values + 1e-9 < trough_values) or np.any(
        pnl_values > peak_values + 1e-9
    ) or np.any(pnl_values < trough_values - 1e-9):
        raise ValueError("persisted trade-state intrabar histories are invalid")
    if len(pnl_values):
        full_peak = (
            np.asarray(
                [row["bid_high"] for row in canonical_path],
                dtype=np.float64,
            )
            - entry_ask
        ) / entry_ask * 10_000.0 if side == SIDE_LONG else (
            entry_bid
            - np.asarray(
                [row["ask_low"] for row in canonical_path],
                dtype=np.float64,
            )
        ) / entry_bid * 10_000.0
        full_trough = (
            np.asarray(
                [row["bid_low"] for row in canonical_path],
                dtype=np.float64,
            )
            - entry_ask
        ) / entry_ask * 10_000.0 if side == SIDE_LONG else (
            entry_bid
            - np.asarray(
                [row["ask_high"] for row in canonical_path],
                dtype=np.float64,
            )
        ) / entry_bid * 10_000.0
        tail_mfe = max(0.0, float(np.max(full_peak)))
        tail_mae = min(0.0, float(np.min(full_trough)))
        if (
            (
                bars_in_trade <= UNIFIED_EXIT_MAX_PATH_BARS
                and (
                    not np.isclose(cum_mfe_bps, tail_mfe, rtol=1e-12, atol=1e-9)
                    or not np.isclose(cum_mae_bps, tail_mae, rtol=1e-12, atol=1e-9)
                )
            )
            or cum_mfe_bps + 1e-9 < tail_mfe
            or cum_mae_bps - 1e-9 > tail_mae
        ):
            raise ValueError(
                "persisted trade state cumulative excursion is not exact"
            )
    raw_last_exit_decision = payload["last_exit_decision"]
    raw_last_exit_input = payload["last_exit_input_envelope"]
    raw_exit_carry = payload["exit_incremental_carry_envelope"]
    if bars_in_trade == 0:
        if (
            raw_last_exit_decision is not None
            or raw_last_exit_input is not None
            or raw_exit_carry is not None
        ):
            raise ValueError(
                "zero-bar persisted state cannot contain Exit input/decision"
            )
    else:
        if not isinstance(raw_last_exit_decision, dict):
            raise ValueError(
                "processed persisted state requires its last Exit decision"
            )
        if not isinstance(raw_last_exit_input, dict):
            raise ValueError(
                "processed persisted state requires its last Exit input"
            )
        try:
            exit_path_envelope = require_unified_exit_path_envelope(
                {
                    "schema_version": (
                        UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION
                    ),
                    "entry_fill_ts": entry_ts.isoformat(),
                    "first_full_m1_bar_ts": (
                        first_full_closed_m1_bar_ts(entry_ts).isoformat()
                    ),
                    "last_closed_m1_bar_ts": (
                        last_processed_m1_ts.isoformat()
                    ),
                    "bars_in_trade": bars_in_trade,
                    "retained_path_length": len(canonical_path),
                    "path_rows": canonical_path,
                    "path_rows_sha256": canonical_closed_m1_path_sha256(
                        canonical_path
                    ),
                    "full_path_chain_sha256": full_path_chain_sha256,
                },
                context="TRADE_STATE_PERSISTED",
            )
            validated_exit_input = require_unified_exit_input_envelope(
                raw_last_exit_input
            )
            expected_exit_bundle_sha256 = (
                bundle_binding["bundle_sha256"]
                if bundle_binding is not None
                else validated_exit_input["bundle_sha256"]
            )
            if (
                validated_exit_input["decision_time"]
                != last_processed_m1_ts.isoformat()
                or validated_exit_input["side"] != side
                or float(validated_exit_input["entry_bid"]) != entry_bid
                or float(validated_exit_input["entry_ask"]) != entry_ask
                or trade_id is None
                or validated_exit_input["decision_identity"] != trade_id
                or validated_exit_input["entry_decision_token_snapshot"]
                != token_snapshot
                or (
                    source_pair_binding is not None
                    and validated_exit_input["m1_feature_window"][
                        "pair_generation_id"
                    ]
                    != source_pair_binding["pair_generation_id"]
                )
            ):
                raise RuntimeError(
                    "persisted Exit input differs from trade identity"
                )
            validated_exit_decision = require_unified_exit_output(
                raw_last_exit_decision,
                context="TRADE_STATE_PERSISTED",
                expected_bundle_sha256=expected_exit_bundle_sha256,
                entry_snapshot=snapshot,
                exit_path_envelope=exit_path_envelope,
                exit_input_envelope=validated_exit_input,
            )
            validated_carry = require_unified_exit_incremental_carry_envelope(
                raw_exit_carry,
                expected_trade_identity=trade_id,
                expected_side=side,
                expected_bundle_sha256=expected_exit_bundle_sha256,
                expected_input_normalization_sha256=(
                    token_snapshot["input_normalization_sha256"]
                ),
                expected_entry_token_snapshot_sha256=(
                    canonical_unified_evidence_sha256(token_snapshot)
                ),
                expected_full_path_chain_sha256=full_path_chain_sha256,
                expected_last_closed_m1_bar_ts=last_processed_m1_ts,
                expected_step_count=bars_in_trade,
                expected_input_envelope_sha256=validated_exit_input[
                    "input_envelope_sha256"
                ],
                expected_mtf_last_row_sha256=validated_exit_input[
                    "mtf_last_row_sha256"
                ],
                expected_previous_carry_envelope_sha256=(
                    UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
                    if bars_in_trade == 1
                    else None
                ),
            )
            if (
                validated_carry
                != validated_exit_decision[
                    "exit_incremental_carry_envelope"
                ]
                or validated_carry["input_envelope_sha256"]
                != validated_exit_input["input_envelope_sha256"]
            ):
                raise RuntimeError(
                    "persisted Exit carry differs from decision/input"
                )
        except RuntimeError as exc:
            raise ValueError(
                "persisted last Exit decision is invalid"
            ) from exc
        if validated_exit_decision != raw_last_exit_decision:
            raise ValueError(
                "persisted last Exit decision is not canonical"
            )

    return entry_ts, snapshot, last_processed_m1_ts


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
    entry_decision_token_snapshot: dict[str, Any]
    units: int
    sizing_execution_evidence: dict[str, Any]
    model_bundle_binding: dict[str, Any] | None
    entry_source_pair_binding: dict[str, Any] | None
    broker_account_binding: dict[str, str] | None

    # Identity (set after OANDA fill — used as state-file name + close-trade id)
    trade_id: str | None = None

    # Running state (updated per M1 bar)
    bars_in_trade: int = 0
    last_processed_m1_ts: pd.Timestamp | None = None
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

    # Per-bar trajectory used by the unified model path envelope.
    pnl_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )  # bps per bar
    mfe_at_bar: float = 0.0                  # mfe_bps at last bar
    time_since_mfe_peak_bars: int = 0
    last_executable_range_bps: float = 0.0

    # Intrabar excursion history. Peak/trough are favorable/adverse excursion
    # bps on the executable spread side: long
    # bid_high/bid_low, short ask_low/ask_high). Executable range is the
    # spread-inclusive one-bar range (ask_high-bid_low)/literal_mid_close*1e4;
    # it is deliberately not called ATR. One value is appended per literal,
    # source-bound CLOSED M1 bar in lock-step with pnl_history.
    peak_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )
    trough_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )
    executable_range_bps_history: deque = field(
        default_factory=lambda: deque(maxlen=TRAJECTORY_HISTORY_MAXLEN)
    )
    closed_m1_path: deque = field(
        default_factory=lambda: deque(maxlen=UNIFIED_EXIT_MAX_PATH_BARS)
    )
    full_path_chain_sha256: str = UNIFIED_EXIT_PATH_CHAIN_GENESIS_SHA256
    last_exit_decision: dict[str, Any] | None = None
    last_exit_input_envelope: dict[str, Any] | None = None
    exit_incremental_carry_envelope: dict[str, Any] | None = None

    def require_entry_snapshot(self) -> dict[str, Any]:
        """Validate this trade's snapshot under its exact execution mode."""

        return _require_trade_entry_snapshot(
            self.v10_snapshot,
            sizing_execution_evidence=self.sizing_execution_evidence,
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
        model_bundle_binding: dict[str, Any],
        entry_source_pair_binding: dict[str, Any],
        broker_account_binding: dict[str, str] | None = None,
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
            model_bundle_binding=model_bundle_binding,
            entry_source_pair_binding=entry_source_pair_binding,
            broker_account_binding=broker_account_binding,
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
            model_bundle_binding=None,
            entry_source_pair_binding=None,
            broker_account_binding=None,
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
        model_bundle_binding: dict[str, Any] | None,
        entry_source_pair_binding: dict[str, Any] | None,
        broker_account_binding: dict[str, str] | None,
    ) -> "TradeState":
        if side not in SIDES:
            raise ValueError(f"side must be {SIDES}, got {side!r}")
        if entry_bid <= 0 or entry_ask <= 0 or entry_ask <= entry_bid:
            raise ValueError(f"invalid prices: bid={entry_bid} ask={entry_ask}")
        raw_snapshot = dict(v10_snapshot or {})
        snapshot = _require_trade_entry_snapshot(
            raw_snapshot,
            sizing_execution_evidence=sizing_execution_evidence,
        )
        parsed_entry_ts = pd.Timestamp(entry_ts)
        if (
            pd.isna(parsed_entry_ts)
            or parsed_entry_ts.tzinfo is None
            or parsed_entry_ts.utcoffset() is None
            or parsed_entry_ts.utcoffset().total_seconds() != 0.0
        ):
            raise ValueError("entry fill timestamp must be timezone-aware UTC")
        parsed_entry_ts = parsed_entry_ts.tz_convert("UTC")
        if "runtime_head_evidence_schema_version" in snapshot:
            require_model_native_exit_replay_entry_time(
                snapshot,
                parsed_entry_ts,
                context="TRADE_STATE_OPEN_EXIT_RESEARCH",
            )
        elif sizing_execution_evidence.get("mode") == "unit_normalized_research_only":
            require_model_native_entry_time(
                snapshot,
                parsed_entry_ts,
                context="TRADE_STATE_OPEN_RESEARCH",
            )
        else:
            require_model_native_fill_time(
                snapshot,
                parsed_entry_ts,
                context="TRADE_STATE_OPEN_FILL",
            )
        validated_sizing_evidence = _require_sizing_execution_evidence(
            sizing_execution_evidence,
            snapshot=snapshot,
            side=side,
            units=units,
        )
        executable = validated_sizing_evidence["mode"] in {
            "learned_virtual_dry_run",
            "learned_broker_fill",
        }
        validated_bundle_binding = require_trade_model_bundle_binding(
            model_bundle_binding,
            executable=executable,
        )
        validated_source_pair_binding = require_trade_source_pair_binding(
            entry_source_pair_binding,
            executable=executable,
        )
        validated_broker_account_binding = (
            require_trade_broker_account_binding(
                broker_account_binding,
                execution_mode=validated_sizing_evidence["mode"],
            )
        )
        if not isinstance(trade_id, str) or not trade_id:
            raise ValueError(
                "trade identity is required to freeze the Entry-decision token"
            )
        if validated_bundle_binding is not None:
            model_identity_kind = "bundle_sha256"
            model_identity_sha256 = validated_bundle_binding["bundle_sha256"]
            input_normalization_sha256 = validated_bundle_binding[
                "input_normalization_sha256"
            ]
            contract_mode = validated_bundle_binding["contract_mode"]
        else:
            model_identity_kind = "training_state_sha256"
            model_identity_sha256 = canonical_unified_evidence_sha256(snapshot)
            normalization_contract = validated_sizing_evidence[
                "research_normalization_contract"
            ]
            input_normalization_sha256 = hashlib.sha256(
                str(normalization_contract).encode("utf-8")
            ).hexdigest()
            contract_mode = MODEL_NATIVE_CONTRACT_MODE
        entry_decision_token_snapshot = build_entry_decision_token_snapshot(
            token=snapshot[ENTRY_DECISION_TOKEN_KEY],
            decision_time=snapshot["decision_ts"],
            fill_time=parsed_entry_ts,
            model_identity_kind=model_identity_kind,
            model_identity_sha256=model_identity_sha256,
            input_normalization_sha256=input_normalization_sha256,
            contract_mode=contract_mode,
            model_direction_index=int(snapshot["model_direction_index"]),
            model_direction=str(snapshot["model_direction"]),
            side=str(side),
            entry_bid=float(entry_bid),
            entry_ask=float(entry_ask),
            trade_identity=trade_id,
        )
        spread_bps = (entry_ask - entry_bid) / entry_bid * 10000.0
        return cls(
            entry_ts=parsed_entry_ts,
            side=str(side),
            entry_bid=float(entry_bid),
            entry_ask=float(entry_ask),
            entry_spread_bps=float(spread_bps),
            v10_snapshot=dict(snapshot),
            entry_decision_token_snapshot=entry_decision_token_snapshot,
            units=units,
            sizing_execution_evidence=validated_sizing_evidence,
            model_bundle_binding=validated_bundle_binding,
            entry_source_pair_binding=validated_source_pair_binding,
            broker_account_binding=validated_broker_account_binding,
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
        self,
        bid_high: float,
        bid_low: float,
        ask_high: float,
        ask_low: float,
    ) -> tuple[float, float]:
        """Return exact intrabar favorable/adverse excursion in bps.

          long  (entry@ask): peak=(bid_high-entry_ask)/entry_ask, trough=(bid_low-entry_ask)/entry_ask
          short (entry@bid): peak=(entry_bid-ask_low)/entry_bid,  trough=(entry_bid-ask_high)/entry_bid
        """
        if self.side == SIDE_LONG:
            peak = (bid_high - self.entry_ask) / self.entry_ask * 10000.0
            trough = (bid_low - self.entry_ask) / self.entry_ask * 10000.0
        else:
            peak = (self.entry_bid - ask_low) / self.entry_bid * 10000.0
            trough = (self.entry_bid - ask_high) / self.entry_bid * 10000.0
        return float(peak), float(trough)

    def update_bar(
        self,
        *,
        schema_version: str,
        time: str,
        complete: bool,
        source_path: str,
        source_sha256: str,
        bid_open: float,
        bid_high: float,
        bid_low: float,
        bid_close: float,
        ask_open: float,
        ask_high: float,
        ask_low: float,
        ask_close: float,
        mid_open: float,
        mid_high: float,
        mid_low: float,
        mid_close: float,
        volume: int,
    ) -> None:
        """Advance state by one literal, complete, source-bound M/B/A M1 row.

        The first admitted bar starts at ``ceil(actual_fill_time, 1m)``. A
        mid-minute broker fill can therefore never inherit pre-fill high/low.
        Literal mid prices are retained rather than reconstructed from bid/ask.
        """
        if self.bars_in_trade >= UNIFIED_EXIT_MAX_PATH_BARS:
            raise ValueError(
                "unified Exit current capacity requires terminal EXIT_NOW"
            )
        if schema_version != CLOSED_M1_PATH_SCHEMA_VERSION:
            raise ValueError("closed M1 path schema_version mismatch")
        canonical_bar = canonical_closed_m1_bar(
            m1_bar_ts=pd.Timestamp(time),
            complete=complete,
            source_path=source_path,
            source_sha256=source_sha256,
            bid_open=bid_open,
            bid_high=bid_high,
            bid_low=bid_low,
            bid_close=bid_close,
            ask_open=ask_open,
            ask_high=ask_high,
            ask_low=ask_low,
            ask_close=ask_close,
            mid_open=mid_open,
            mid_high=mid_high,
            mid_low=mid_low,
            mid_close=mid_close,
            volume=volume,
        )
        parsed_m1_bar_ts = pd.Timestamp(canonical_bar["time"])
        minimum_m1_bar_ts = (
            first_full_closed_m1_bar_ts(self.entry_ts)
            if self.last_processed_m1_ts is None
            else self.last_processed_m1_ts
        )
        if (
            parsed_m1_bar_ts < minimum_m1_bar_ts
            or (
                self.last_processed_m1_ts is not None
                and parsed_m1_bar_ts == minimum_m1_bar_ts
            )
        ):
            raise ValueError(
                "closed M1 row clock duplicate/reversal: "
                f"minimum={minimum_m1_bar_ts.isoformat()} "
                f"observed={parsed_m1_bar_ts.isoformat()}"
            )

        bid = canonical_bar["bid_close"]
        ask = canonical_bar["ask_close"]
        m1_close = canonical_bar["mid_close"]
        bid_high = canonical_bar["bid_high"]
        bid_low = canonical_bar["bid_low"]
        ask_high = canonical_bar["ask_high"]
        ask_low = canonical_bar["ask_low"]

        self.bars_in_trade += 1
        self.m1_returns_window.append(m1_close)

        self.current_bid = bid
        self.current_ask = ask
        self.current_pnl_bps = float(self._pnl_bps(bid, ask))
        peak, trough = self._intrabar_excursion(
            bid_high, bid_low, ask_high, ask_low,
        )
        executable_range_bps = (
            (ask_high - bid_low) / m1_close * 10_000.0
        )
        prev_peak = self.cum_mfe_bps
        self.cum_mfe_bps = max(self.cum_mfe_bps, peak)
        self.cum_mae_bps = min(self.cum_mae_bps, trough)
        if self.cum_mfe_bps > prev_peak:
            self.bars_since_mfe_peak = 0
            self.time_since_mfe_peak_bars = 0
        else:
            self.bars_since_mfe_peak += 1
            self.time_since_mfe_peak_bars += 1
        self.mfe_at_bar = self.cum_mfe_bps
        self.pnl_history.append(self.current_pnl_bps)

        self.peak_history.append(float(peak))
        self.trough_history.append(float(trough))
        self.executable_range_bps_history.append(float(executable_range_bps))
        self.last_executable_range_bps = float(executable_range_bps)
        self.full_path_chain_sha256 = extend_closed_m1_path_chain_sha256(
            self.full_path_chain_sha256,
            canonical_bar,
        )
        self.closed_m1_path.append(canonical_bar)
        self.last_processed_m1_ts = parsed_m1_bar_ts

    def build_closed_m1_path_evidence(self) -> dict[str, Any]:
        """Return the exact persisted path prefix and its content digest."""

        rows = [dict(row) for row in self.closed_m1_path]
        if len(rows) != min(self.bars_in_trade, UNIFIED_EXIT_MAX_PATH_BARS):
            raise ValueError("closed M1 path/bar count mismatch")
        if not rows:
            raise ValueError("closed M1 path evidence requires at least one bar")
        envelope = {
            "schema_version": UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION,
            "entry_fill_ts": self.entry_ts.isoformat(),
            "first_full_m1_bar_ts": first_full_closed_m1_bar_ts(
                self.entry_ts
            ).isoformat(),
            "last_closed_m1_bar_ts": self.last_processed_m1_ts.isoformat(),
            "bars_in_trade": self.bars_in_trade,
            "retained_path_length": len(rows),
            "path_rows": rows,
            "path_rows_sha256": canonical_closed_m1_path_sha256(rows),
            "full_path_chain_sha256": self.full_path_chain_sha256,
        }
        return require_unified_exit_path_envelope(
            envelope,
            context="TRADE_STATE",
        )

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
            "entry_decision_token_snapshot": _jsonable(
                self.entry_decision_token_snapshot
            ),
            "trade_id": self.trade_id,
            "units": self.units,
            "sizing_execution_evidence": _jsonable(
                self.sizing_execution_evidence
            ),
            "model_bundle_binding": _jsonable(
                self.model_bundle_binding
            ),
            "entry_source_pair_binding": _jsonable(
                self.entry_source_pair_binding
            ),
            "broker_account_binding": _jsonable(
                self.broker_account_binding
            ),
            "bars_in_trade": self.bars_in_trade,
            "last_processed_m1_ts": (
                self.last_processed_m1_ts.isoformat()
                if self.last_processed_m1_ts is not None
                else None
            ),
            "current_bid": self.current_bid,
            "current_ask": self.current_ask,
            "current_pnl_bps": self.current_pnl_bps,
            "cum_mfe_bps": self.cum_mfe_bps,
            "cum_mae_bps": self.cum_mae_bps,
            "bars_since_mfe_peak": self.bars_since_mfe_peak,
            "m1_returns_window": list(self.m1_returns_window),
            # Per-bar trajectory
            "pnl_history": list(self.pnl_history),
            "mfe_at_bar": self.mfe_at_bar,
            "time_since_mfe_peak_bars": self.time_since_mfe_peak_bars,
            "last_executable_range_bps": self.last_executable_range_bps,
            # V5 literal M/B/A path and exact intrabar state.
            "peak_history": list(self.peak_history),
            "trough_history": list(self.trough_history),
            "executable_range_bps_history": list(
                self.executable_range_bps_history
            ),
            "closed_m1_path": list(self.closed_m1_path),
            "full_path_chain_sha256": self.full_path_chain_sha256,
            "last_exit_input_envelope": _jsonable(
                self.last_exit_input_envelope
            ),
            "last_exit_decision": _jsonable(self.last_exit_decision),
            "exit_incremental_carry_envelope": _jsonable(
                self.exit_incremental_carry_envelope
            ),
        })
        if not isinstance(payload, dict):  # pragma: no cover - fixed literal shape
            raise AssertionError("trade-state serialization did not produce an object")
        _validate_persisted_trade_state_payload(payload)
        return payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeState":
        """Rehydrate only an exact versioned state; never fill missing evidence."""
        entry_ts, snapshot, last_processed_m1_ts = (
            _validate_persisted_trade_state_payload(d)
        )
        t = cls(
            entry_ts=entry_ts,
            side=d["side"],
            entry_bid=float(d["entry_bid"]),
            entry_ask=float(d["entry_ask"]),
            entry_spread_bps=float(d["entry_spread_bps"]),
            v10_snapshot=dict(snapshot),
            entry_decision_token_snapshot=dict(
                require_entry_decision_token_snapshot(
                    d["entry_decision_token_snapshot"]
                )
            ),
            trade_id=d["trade_id"],
            units=d["units"],
            sizing_execution_evidence=dict(d["sizing_execution_evidence"]),
            model_bundle_binding=(
                dict(d["model_bundle_binding"])
                if d["model_bundle_binding"] is not None
                else None
            ),
            entry_source_pair_binding=(
                dict(d["entry_source_pair_binding"])
                if d["entry_source_pair_binding"] is not None
                else None
            ),
            broker_account_binding=(
                dict(d["broker_account_binding"])
                if d["broker_account_binding"] is not None
                else None
            ),
            bars_in_trade=d["bars_in_trade"],
            last_processed_m1_ts=last_processed_m1_ts,
            current_bid=float(d["current_bid"]),
            current_ask=float(d["current_ask"]),
            current_pnl_bps=float(d["current_pnl_bps"]),
            cum_mfe_bps=float(d["cum_mfe_bps"]),
            cum_mae_bps=float(d["cum_mae_bps"]),
            bars_since_mfe_peak=d["bars_since_mfe_peak"],
            mfe_at_bar=float(d["mfe_at_bar"]),
            time_since_mfe_peak_bars=d["time_since_mfe_peak_bars"],
            last_executable_range_bps=float(
                d["last_executable_range_bps"]
            ),
            last_exit_decision=(
                dict(d["last_exit_decision"])
                if d["last_exit_decision"] is not None
                else None
            ),
            last_exit_input_envelope=(
                dict(d["last_exit_input_envelope"])
                if d["last_exit_input_envelope"] is not None
                else None
            ),
            exit_incremental_carry_envelope=(
                dict(d["exit_incremental_carry_envelope"])
                if d["exit_incremental_carry_envelope"] is not None
                else None
            ),
            full_path_chain_sha256=d["full_path_chain_sha256"],
        )
        t.m1_returns_window.extend(float(v) for v in d["m1_returns_window"])
        t.pnl_history.extend(float(v) for v in d["pnl_history"])
        t.peak_history.extend(float(v) for v in d["peak_history"])
        t.trough_history.extend(float(v) for v in d["trough_history"])
        t.executable_range_bps_history.extend(
            float(v) for v in d["executable_range_bps_history"]
        )
        t.closed_m1_path.extend(dict(row) for row in d["closed_m1_path"])
        return t

    def clone_for_exit_decision(self) -> "TradeState":
        """Return an exact detached copy used for transactional inference."""

        return self.from_dict(self.to_dict())

    def bind_unified_exit_decision(
        self,
        decision: dict[str, Any],
        *,
        expected_bundle_sha256: str,
        exit_input_envelope: dict[str, Any],
    ) -> None:
        """Bind one exact same-bundle decision to the current path prefix."""

        if self.bars_in_trade <= 0:
            raise ValueError("Exit decision requires one complete M1 bar")
        snapshot = self.require_entry_snapshot()
        path_envelope = self.build_closed_m1_path_evidence()
        input_envelope = require_unified_exit_input_envelope(
            exit_input_envelope
        )
        executable = self.sizing_execution_evidence.get("mode") in {
            "learned_virtual_dry_run",
            "learned_broker_fill",
        }
        bundle_binding = require_trade_model_bundle_binding(
            self.model_bundle_binding,
            executable=executable,
        )
        source_pair_binding = require_trade_source_pair_binding(
            self.entry_source_pair_binding,
            executable=executable,
        )
        if (
            bundle_binding is not None
            and expected_bundle_sha256 != bundle_binding["bundle_sha256"]
        ):
            raise ValueError(
                "Exit decision bundle differs from immutable trade model bundle"
            )
        if (
            input_envelope["decision_time"]
            != self.last_processed_m1_ts.isoformat()
            or input_envelope["side"] != self.side
            or float(input_envelope["entry_bid"]) != self.entry_bid
            or float(input_envelope["entry_ask"]) != self.entry_ask
            or self.trade_id is None
            or input_envelope["decision_identity"] != self.trade_id
            or input_envelope["bundle_sha256"] != expected_bundle_sha256
            or input_envelope["entry_decision_token_snapshot"]
            != self.entry_decision_token_snapshot
            or (
                source_pair_binding is not None
                and input_envelope["m1_feature_window"][
                    "pair_generation_id"
                ]
                != source_pair_binding["pair_generation_id"]
            )
        ):
            raise ValueError("Exit input envelope differs from trade state")
        validated = require_unified_exit_output(
            decision,
            context="TRADE_STATE_BIND_EXIT",
            expected_bundle_sha256=expected_bundle_sha256,
            entry_snapshot=snapshot,
            exit_path_envelope=path_envelope,
            exit_input_envelope=input_envelope,
        )
        self.last_exit_decision = deepcopy(validated)
        self.last_exit_input_envelope = deepcopy(input_envelope)
        self.exit_incremental_carry_envelope = deepcopy(
            validated["exit_incremental_carry_envelope"]
        )

    def commit_complete_exit_bar(self, staged: "TradeState") -> None:
        """Atomically adopt one fully validated staged M1/Exit transition."""

        if not isinstance(staged, TradeState):
            raise TypeError("staged exit state must be TradeState")
        staged_payload = staged.to_dict()
        immutable_fields = (
            "entry_ts",
            "side",
            "entry_bid",
            "entry_ask",
            "entry_spread_bps",
            "v10_snapshot",
            "trade_id",
            "units",
            "sizing_execution_evidence",
            "model_bundle_binding",
            "entry_source_pair_binding",
            "broker_account_binding",
            "entry_decision_token_snapshot",
        )
        current_payload = self.to_dict()
        if any(
            staged_payload[field_name] != current_payload[field_name]
            for field_name in immutable_fields
        ):
            raise ValueError("staged exit state changed immutable trade identity")
        if staged.bars_in_trade != self.bars_in_trade + 1:
            raise ValueError("staged exit state must contain exactly one new M1 bar")
        staged_carry = require_unified_exit_incremental_carry_envelope(
            staged.exit_incremental_carry_envelope,
            expected_previous_carry_envelope_sha256=(
                UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
                if self.exit_incremental_carry_envelope is None
                else self.exit_incremental_carry_envelope[
                    "carry_envelope_sha256"
                ]
            ),
        )
        if int(staged_carry["step_count"]) != staged.bars_in_trade:
            raise ValueError("staged Exit carry step differs from trade state")
        minimum_ts = (
            first_full_closed_m1_bar_ts(self.entry_ts)
            if self.last_processed_m1_ts is None
            else self.last_processed_m1_ts
        )
        if (
            staged.last_processed_m1_ts is None
            or staged.last_processed_m1_ts < minimum_ts
            or (
                self.last_processed_m1_ts is not None
                and staged.last_processed_m1_ts == minimum_ts
            )
        ):
            raise ValueError("staged exit state M1 row clock is not forward")
        if (
            staged.last_exit_decision is None
            or staged.last_exit_input_envelope is None
        ):
            raise ValueError(
                "staged exit state requires its exact model input/decision"
            )
        self.__dict__.clear()
        self.__dict__.update(deepcopy(staged.__dict__))

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
        encoded = json.dumps(
            self.to_dict(),
            default=str,
            indent=2,
        ).encode("utf-8")
        descriptor = os.open(
            tmp,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(f"short trade-state write: {tmp}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def state_filename(self) -> str:
        """Filename for this trade's state file (one file per trade)."""
        tid = self.trade_id or f"virtual_{self.entry_ts.strftime('%Y%m%dT%H%M%S')}"
        return f"open_trade_{tid}.json"

    def delete_state_file(self, directory: Path) -> None:
        """Remove this trade's persisted state file from the directory."""
        path = directory / self.state_filename()
        if path.exists():
            path.unlink()
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

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
