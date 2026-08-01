"""Canonical model-native LONG/SHORT/FLAT decision contract.

This module describes the public decision surface written into every new
Entry-V10 bundle.  Consumers must validate the bundle metadata before using
``direction_logits``; auxiliary heads never have authority to rewrite the
model's final argmax.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    CLASS_ORDER,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    entry_exit_shared_feature_base_contract,
)

DIRECTION_DECISION_CONTRACT_SCHEMA_VERSION = "gx1_model_direction_decision_v3"
MODEL_DIRECTION_SELECTION_MODE = "model_direction_argmax"
DIRECTION_LOGITS_KEY = "direction_logits"
PUBLIC_TRADE_FLAT_LOGITS_KEY = "public_trade_flat_decision_logits"
MODEL_DIRECTION_OPERATING_POINT_KEYS = frozenset({"selection_score", "max_trades"})
MODEL_DIRECTION_CLASS_ORDER = CLASS_ORDER
MODEL_DIRECTION_ACTION_ORDER = (
    "TAKE_LONG_NOW",
    "TAKE_SHORT_NOW",
    "SKIP",
)
if len(MODEL_DIRECTION_CLASS_ORDER) != len(MODEL_DIRECTION_ACTION_ORDER):
    raise RuntimeError("MODEL_DIRECTION_CLASS_ACTION_LAYOUT_INVALID")
MODEL_DIRECTION_NAME_BY_INDEX = dict(enumerate(MODEL_DIRECTION_CLASS_ORDER))
MODEL_DIRECTION_ACTION_BY_INDEX = dict(enumerate(MODEL_DIRECTION_ACTION_ORDER))
MODEL_DIRECTION_INDEX_BY_NAME = {
    name: index for index, name in MODEL_DIRECTION_NAME_BY_INDEX.items()
}
MODEL_DIRECTION_LONG_INDEX = MODEL_DIRECTION_INDEX_BY_NAME["LONG"]
MODEL_DIRECTION_SHORT_INDEX = MODEL_DIRECTION_INDEX_BY_NAME["SHORT"]
MODEL_DIRECTION_FLAT_INDEX = MODEL_DIRECTION_INDEX_BY_NAME["FLAT"]
MODEL_DIRECTION_ACTION_ID_BY_INDEX = dict(enumerate((1, 2, 0)))
MODEL_DIRECTION_EXECUTION_SIDE_BY_INDEX = {
    MODEL_DIRECTION_LONG_INDEX: "long",
    MODEL_DIRECTION_SHORT_INDEX: "short",
    MODEL_DIRECTION_FLAT_INDEX: None,
}
MODEL_DIRECTION_TRADE_INDICES = (
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SHORT_INDEX,
)
PUBLIC_TRADE_INDEX = 0
PUBLIC_FLAT_INDEX = 1
UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION = "gx1_unified_entry_exit_v1"
UNIFIED_EXIT_LOGITS_KEY = "exit_action_logits"
UNIFIED_EXIT_PROBS_KEY = "exit_action_probs"
UNIFIED_EXIT_ACTION_ORDER = ("HOLD", "EXIT_NOW")
UNIFIED_EXIT_SIDE_ORDER = ("long", "short")
UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY = "entry_shared_representation"
UNIFIED_EXIT_MODEL_REPRESENTATION_KEY = "shared_feature_representation"
UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM = 128
UNIFIED_EXIT_PATH_TENSOR_SCHEMA_VERSION = "gx1_unified_exit_path_tensor_v1"
UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION = (
    "gx1_unified_exit_closed_m1_path_envelope_v1"
)
UNIFIED_EXIT_MAX_PATH_BARS = 512
UNIFIED_EXIT_PATH_ENCODER_LAYERS = 2
UNIFIED_EXIT_PATH_PRICE_FIELDS = (
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "mid_open",
    "mid_high",
    "mid_low",
    "mid_close",
)
UNIFIED_EXIT_PATH_FEATURE_ORDER = (
    *tuple(f"{name}_bps_from_entry_mid" for name in UNIFIED_EXIT_PATH_PRICE_FIELDS),
    "log1p_volume",
    "entry_spread_bps",
)
UNIFIED_EXIT_PATH_FEATURE_DIM = len(UNIFIED_EXIT_PATH_FEATURE_ORDER)
UNIFIED_EXIT_REQUIRED_HASH_KEYS = (
    "bundle_sha256",
    "entry_snapshot_sha256",
    "exit_path_envelope_sha256",
    "output_evidence_sha256",
)
UNIFIED_EXIT_OUTPUT_FIELDS = frozenset(
    {
        UNIFIED_EXIT_LOGITS_KEY,
        UNIFIED_EXIT_PROBS_KEY,
        "exit_action_index",
        "action",
        "decision_source",
        *UNIFIED_EXIT_REQUIRED_HASH_KEYS,
    }
)
CLOSED_M1_PATH_SCHEMA_VERSION = "gx1_closed_m1_literal_mba_path_v1"
CLOSED_M1_PATH_FIELDS = (
    "schema_version",
    "time",
    "complete",
    "source_path",
    "source_sha256",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
    "mid_open",
    "mid_high",
    "mid_low",
    "mid_close",
    "volume",
)
UNIFIED_EXIT_PATH_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "entry_fill_ts",
        "first_full_m1_bar_ts",
        "last_closed_m1_bar_ts",
        "bars_in_trade",
        "retained_path_length",
        "path_rows",
        "path_rows_sha256",
    }
)


def canonical_closed_m1_bar(
    *,
    m1_bar_ts: pd.Timestamp,
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
) -> dict[str, Any]:
    """Validate and canonicalize one literal OANDA M/B/A closed M1 row."""

    parsed_ts = pd.Timestamp(m1_bar_ts)
    if (
        pd.isna(parsed_ts)
        or parsed_ts.tzinfo is None
        or parsed_ts.utcoffset() is None
        or parsed_ts.utcoffset().total_seconds() != 0.0
        or parsed_ts != parsed_ts.floor("min")
    ):
        raise ValueError("closed M1 timestamp must be an exact UTC minute")
    parsed_ts = parsed_ts.tz_convert("UTC")
    if complete is not True:
        raise ValueError("closed M1 bar must be explicitly complete")
    if (
        not isinstance(source_path, str)
        or not source_path
        or not Path(source_path).is_absolute()
    ):
        raise ValueError("closed M1 source_path must be an absolute path")
    if (
        not isinstance(source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
    ):
        raise ValueError("closed M1 source_sha256 must be 64 lowercase hex")
    if isinstance(volume, bool) or not isinstance(volume, (int, np.integer)):
        raise ValueError("closed M1 volume must be an exact non-negative integer")
    volume_int = int(volume)
    if volume_int < 0:
        raise ValueError("closed M1 volume must be an exact non-negative integer")

    raw_prices = {
        "bid_open": bid_open,
        "bid_high": bid_high,
        "bid_low": bid_low,
        "bid_close": bid_close,
        "ask_open": ask_open,
        "ask_high": ask_high,
        "ask_low": ask_low,
        "ask_close": ask_close,
        "mid_open": mid_open,
        "mid_high": mid_high,
        "mid_low": mid_low,
        "mid_close": mid_close,
    }
    prices: dict[str, float] = {}
    invalid: list[str] = []
    for name, raw_value in raw_prices.items():
        try:
            parsed_value = float(raw_value)
        except (TypeError, ValueError):
            invalid.append(name)
            continue
        if not np.isfinite(parsed_value) or parsed_value <= 0.0:
            invalid.append(name)
        else:
            prices[name] = parsed_value
    if invalid:
        raise ValueError(
            f"closed M1 bar has non-finite/non-positive fields: {invalid}"
        )

    own_ohlc_invalid = any(
        prices[f"{prefix}_low"]
        > min(prices[f"{prefix}_open"], prices[f"{prefix}_close"])
        or prices[f"{prefix}_high"]
        < max(prices[f"{prefix}_open"], prices[f"{prefix}_close"])
        or prices[f"{prefix}_low"] > prices[f"{prefix}_high"]
        for prefix in ("bid", "ask", "mid")
    )
    cross_price_invalid = any(
        prices[f"ask_{component}"] < prices[f"bid_{component}"]
        for component in ("open", "high", "low", "close")
    )
    if own_ohlc_invalid or cross_price_invalid:
        raise ValueError("closed M1 literal M/B/A OHLC geometry invalid")

    return {
        "schema_version": CLOSED_M1_PATH_SCHEMA_VERSION,
        "time": parsed_ts.isoformat(),
        "complete": True,
        "source_path": source_path,
        "source_sha256": source_sha256,
        **prices,
        "volume": volume_int,
    }


def canonical_closed_m1_path_sha256(path_rows: list[dict[str, Any]]) -> str:
    """Hash the exact ordered path bytes used by replay and serving."""

    payload = json.dumps(
        path_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_unified_exit_path_envelope(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate the exact untruncated closed-M1 prefix consumed by Exit."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} Exit path envelope must be a mapping")
    envelope = dict(value)
    missing = sorted(UNIFIED_EXIT_PATH_ENVELOPE_FIELDS - set(envelope))
    unexpected = sorted(set(envelope) - UNIFIED_EXIT_PATH_ENVELOPE_FIELDS)
    if missing or unexpected:
        raise RuntimeError(
            f"{context} Exit path envelope exact schema mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    if (
        envelope.get("schema_version")
        != UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION
    ):
        raise RuntimeError(f"{context} Exit path envelope version mismatch")
    try:
        entry_fill = pd.Timestamp(envelope["entry_fill_ts"])
        first_full = pd.Timestamp(envelope["first_full_m1_bar_ts"])
        last_closed = pd.Timestamp(envelope["last_closed_m1_bar_ts"])
    except Exception as exc:
        raise RuntimeError(
            f"{context} Exit path envelope timestamp invalid"
        ) from exc
    if any(
        pd.isna(timestamp)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != pd.Timedelta(0)
        for timestamp in (entry_fill, first_full, last_closed)
    ):
        raise RuntimeError(
            f"{context} Exit path envelope timestamps must be UTC"
        )
    entry_fill = entry_fill.tz_convert("UTC")
    first_full = first_full.tz_convert("UTC")
    last_closed = last_closed.tz_convert("UTC")
    if (
        first_full != entry_fill.ceil("min")
        or first_full != first_full.floor("min")
        or last_closed != last_closed.floor("min")
        or envelope["entry_fill_ts"] != entry_fill.isoformat()
        or envelope["first_full_m1_bar_ts"] != first_full.isoformat()
        or envelope["last_closed_m1_bar_ts"] != last_closed.isoformat()
    ):
        raise RuntimeError(
            f"{context} Exit path envelope fill/closed-bar geometry invalid"
        )
    bars_in_trade = envelope.get("bars_in_trade")
    retained_length = envelope.get("retained_path_length")
    if (
        isinstance(bars_in_trade, bool)
        or not isinstance(bars_in_trade, int)
        or isinstance(retained_length, bool)
        or not isinstance(retained_length, int)
        or not 1 <= bars_in_trade <= UNIFIED_EXIT_MAX_PATH_BARS
        or retained_length != bars_in_trade
    ):
        raise RuntimeError(
            f"{context} Exit path envelope length contract invalid"
        )
    rows = envelope.get("path_rows")
    if not isinstance(rows, list) or len(rows) != bars_in_trade:
        raise RuntimeError(f"{context} Exit path rows/length mismatch")
    expected_row_fields = frozenset(CLOSED_M1_PATH_FIELDS)
    canonical_rows: list[dict[str, Any]] = []
    source_identity: tuple[str, str] | None = None
    previous_time: pd.Timestamp | None = None
    for raw in rows:
        if not isinstance(raw, Mapping) or frozenset(raw) != expected_row_fields:
            raise RuntimeError(f"{context} closed M1 path row schema mismatch")
        try:
            canonical = canonical_closed_m1_bar(
                m1_bar_ts=pd.Timestamp(raw["time"]),
                complete=raw["complete"],
                source_path=raw["source_path"],
                source_sha256=raw["source_sha256"],
                bid_open=raw["bid_open"],
                bid_high=raw["bid_high"],
                bid_low=raw["bid_low"],
                bid_close=raw["bid_close"],
                ask_open=raw["ask_open"],
                ask_high=raw["ask_high"],
                ask_low=raw["ask_low"],
                ask_close=raw["ask_close"],
                mid_open=raw["mid_open"],
                mid_high=raw["mid_high"],
                mid_low=raw["mid_low"],
                mid_close=raw["mid_close"],
                volume=raw["volume"],
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{context} closed M1 path row invalid"
            ) from exc
        if dict(raw) != canonical:
            raise RuntimeError(f"{context} closed M1 path row is not canonical")
        identity = (canonical["source_path"], canonical["source_sha256"])
        if source_identity is None:
            source_identity = identity
        elif identity != source_identity:
            raise RuntimeError(
                f"{context} closed M1 source identity changed within path"
            )
        current_time = pd.Timestamp(canonical["time"])
        if previous_time is not None and current_time != previous_time + pd.Timedelta(
            minutes=1
        ):
            raise RuntimeError(f"{context} closed M1 path cadence mismatch")
        previous_time = current_time
        canonical_rows.append(canonical)
    if (
        pd.Timestamp(canonical_rows[0]["time"]) != first_full
        or pd.Timestamp(canonical_rows[-1]["time"]) != last_closed
    ):
        raise RuntimeError(
            f"{context} Exit path envelope terminal timestamps mismatch"
        )
    expected_rows_sha256 = canonical_closed_m1_path_sha256(canonical_rows)
    if envelope.get("path_rows_sha256") != expected_rows_sha256:
        raise RuntimeError(f"{context} Exit path row hash mismatch")
    return json.loads(
        json.dumps(envelope, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )


def unified_exit_path_tensor(
    *,
    path_rows: list[dict[str, Any]],
    entry_bid: float,
    entry_ask: float,
) -> np.ndarray:
    """Convert one exact literal M1 prefix to the sole Exit tensor surface.

    Prices are expressed in bps from the executable fill midpoint.  This is a
    deterministic unit transform, not a direction rule: long/short remains a
    separate learned embedding and all literal bid/ask/mid OHLC values survive
    independently.  Padding and prefix truncation are deliberately absent.
    """

    if not isinstance(path_rows, list) or not path_rows:
        raise ValueError("unified Exit path requires at least one closed M1 row")
    if len(path_rows) > UNIFIED_EXIT_MAX_PATH_BARS:
        raise ValueError(
            "unified Exit path exceeds exact model capacity; truncation is forbidden"
        )
    canonical_rows: list[dict[str, Any]] = []
    expected_keys = frozenset(CLOSED_M1_PATH_FIELDS)
    source_identity: tuple[str, str] | None = None
    previous_time: pd.Timestamp | None = None
    for raw in path_rows:
        if not isinstance(raw, Mapping) or frozenset(raw) != expected_keys:
            raise ValueError("closed M1 path row exact schema mismatch")
        canonical = canonical_closed_m1_bar(
            m1_bar_ts=pd.Timestamp(raw["time"]),
            complete=raw["complete"],
            source_path=raw["source_path"],
            source_sha256=raw["source_sha256"],
            bid_open=raw["bid_open"],
            bid_high=raw["bid_high"],
            bid_low=raw["bid_low"],
            bid_close=raw["bid_close"],
            ask_open=raw["ask_open"],
            ask_high=raw["ask_high"],
            ask_low=raw["ask_low"],
            ask_close=raw["ask_close"],
            mid_open=raw["mid_open"],
            mid_high=raw["mid_high"],
            mid_low=raw["mid_low"],
            mid_close=raw["mid_close"],
            volume=raw["volume"],
        )
        if dict(raw) != canonical:
            raise ValueError("closed M1 path row is not canonical")
        current_identity = (
            canonical["source_path"],
            canonical["source_sha256"],
        )
        if source_identity is None:
            source_identity = current_identity
        elif current_identity != source_identity:
            raise ValueError("closed M1 path source identity changed within prefix")
        current_time = pd.Timestamp(canonical["time"])
        if previous_time is not None and current_time != previous_time + pd.Timedelta(
            minutes=1
        ):
            raise ValueError("closed M1 path cadence gap/duplicate")
        previous_time = current_time
        canonical_rows.append(canonical)

    return unified_exit_path_tensor_from_values(
        price_values=np.asarray(
            [
                [float(row[name]) for name in UNIFIED_EXIT_PATH_PRICE_FIELDS]
                for row in canonical_rows
            ],
            dtype=np.float64,
        ),
        volumes=np.asarray(
            [int(row["volume"]) for row in canonical_rows],
            dtype=np.int64,
        ),
        entry_bid=entry_bid,
        entry_ask=entry_ask,
    )


def unified_exit_path_tensor_from_values(
    *,
    price_values: np.ndarray,
    volumes: np.ndarray,
    entry_bid: float,
    entry_ask: float,
) -> np.ndarray:
    """Apply the one train/replay/live Exit path value transform."""

    prices = np.asarray(price_values, dtype=np.float64)
    raw_volumes = np.asarray(volumes)
    if (
        prices.ndim != 2
        or prices.shape[1] != len(UNIFIED_EXIT_PATH_PRICE_FIELDS)
        or not 1 <= prices.shape[0] <= UNIFIED_EXIT_MAX_PATH_BARS
        or raw_volumes.ndim != 1
        or raw_volumes.shape[0] != prices.shape[0]
    ):
        raise ValueError("unified Exit path value shape is invalid")
    try:
        parsed_volumes = raw_volumes.astype(np.float64)
        parsed_entry_bid = float(entry_bid)
        parsed_entry_ask = float(entry_ask)
    except (TypeError, ValueError) as exc:
        raise ValueError("unified Exit path values are not numeric") from exc
    if (
        not np.isfinite(prices).all()
        or bool((prices <= 0.0).any())
        or not np.isfinite(parsed_volumes).all()
        or bool((parsed_volumes < 0.0).any())
        or not np.equal(parsed_volumes, np.floor(parsed_volumes)).all()
        or not math.isfinite(parsed_entry_bid)
        or not math.isfinite(parsed_entry_ask)
        or parsed_entry_bid <= 0.0
        or parsed_entry_ask <= parsed_entry_bid
    ):
        raise ValueError("unified Exit path values are invalid")
    entry_mid = 0.5 * (parsed_entry_bid + parsed_entry_ask)
    entry_spread_bps = (
        (parsed_entry_ask - parsed_entry_bid) / entry_mid * 10_000.0
    )
    tensor = np.empty(
        (prices.shape[0], UNIFIED_EXIT_PATH_FEATURE_DIM),
        dtype=np.float32,
    )
    tensor[:, : len(UNIFIED_EXIT_PATH_PRICE_FIELDS)] = (
        (prices / entry_mid - 1.0) * 10_000.0
    ).astype(np.float32)
    tensor[:, -2] = np.log1p(parsed_volumes).astype(np.float32)
    tensor[:, -1] = np.float32(entry_spread_bps)
    if not np.isfinite(tensor).all():
        raise ValueError("unified Exit path tensor contains non-finite values")
    return tensor


def canonical_unified_evidence_sha256(value: Mapping[str, Any]) -> str:
    """Hash one exact JSON-safe unified Entry/Exit evidence mapping."""

    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_direction_decision_contract_metadata() -> dict[str, Any]:
    """Return a fresh, JSON-serializable copy of the exact public contract."""

    return {
        "schema_version": DIRECTION_DECISION_CONTRACT_SCHEMA_VERSION,
        "selection_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_logits_key": DIRECTION_LOGITS_KEY,
        "direction_class_order": list(MODEL_DIRECTION_CLASS_ORDER),
        "direction_decision": "argmax(direction_logits)",
        "public_trade_flat_logits_key": PUBLIC_TRADE_FLAT_LOGITS_KEY,
        "public_trade_flat_class_order": ["TRADE", "FLAT"],
        "public_trade_flat_formula": (
            "[max(direction_logits[LONG],direction_logits[SHORT]),"
            "direction_logits[FLAT]]"
        ),
        "output_stage": (
            "final_model_forward_after_learned_evidence_fusion_and_calibration"
        ),
        "auxiliary_heads_direction_authority": "none",
        "runtime_direction_overrides_allowed": False,
        "runtime_direction_thresholds_allowed": False,
        "sizing_authority": "separate_top_level_bundle_contract",
    }


def require_model_direction_decision_contract(
    metadata: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate and return the canonical contract, or fail closed.

    Exact equality is intentional: a partially populated or softly compatible
    declaration cannot prove that training, audit, replay, and live all use the
    same public decision path.
    """

    raw = metadata.get("direction_decision_contract")
    expected = model_direction_decision_contract_metadata()
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"{context} missing required direction_decision_contract metadata"
        )
    observed = dict(raw)
    mismatches = [
        key
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    ]
    unexpected = sorted(set(observed) - set(expected))
    missing = sorted(set(expected) - set(observed))
    if mismatches or unexpected or missing:
        raise RuntimeError(
            f"{context} direction_decision_contract mismatch: "
            f"mismatched={sorted(set(mismatches))} missing={missing} "
            f"unexpected={unexpected}"
        )
    return observed


def require_model_direction_operating_point(
    operating_point: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Require the exact rule-free live operating-point surface.

    ``max_trades`` is an execution exposure limit, not a direction selector.
    Everything else that historically rewrote model direction (edge/utility
    thresholds, session allowlists, side overlays, and similar pass-through
    keys) is rejected by exact-key equality rather than silently ignored.
    """

    if not isinstance(operating_point, Mapping):
        raise RuntimeError(f"{context} missing required operating_point mapping")
    observed = dict(operating_point)
    missing = sorted(MODEL_DIRECTION_OPERATING_POINT_KEYS - set(observed))
    unexpected = sorted(set(observed) - MODEL_DIRECTION_OPERATING_POINT_KEYS)
    if missing or unexpected:
        raise RuntimeError(
            f"{context} operating_point contract mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    if observed.get("selection_score") != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError(
            f"{context} operating_point.selection_score must be exactly "
            f"{MODEL_DIRECTION_SELECTION_MODE!r}"
        )
    max_trades = observed.get("max_trades")
    if isinstance(max_trades, bool) or not isinstance(max_trades, int) or max_trades <= 0:
        raise RuntimeError(
            f"{context} operating_point.max_trades must be a positive integer; "
            f"got {max_trades!r}"
        )
    return observed


def unified_entry_exit_contract_metadata() -> dict[str, Any]:
    """Return the exact one-model Entry/Exit ownership contract."""

    return {
        "schema_version": UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION,
        "single_model_bundle": True,
        "shared_feature_encoder": True,
        "shared_feature_contract": True,
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
        "exit_requires_shared_m1_feature_surface": True,
        "exit_feature_encoder": "same_entry_specialist_modules_no_duplicate_stack",
        "exit_path_is_additive_not_replacement": True,
        "exit_bound_to_entry_snapshot": True,
        "entry_logits_key": DIRECTION_LOGITS_KEY,
        "entry_action_order": list(MODEL_DIRECTION_CLASS_ORDER),
        "exit_logits_key": UNIFIED_EXIT_LOGITS_KEY,
        "exit_probs_key": UNIFIED_EXIT_PROBS_KEY,
        "exit_action_order": list(UNIFIED_EXIT_ACTION_ORDER),
        "exit_decision": "argmax(exit_action_logits)",
        "exit_requires_exact_closed_m1_path": True,
        "exit_path_schema_version": CLOSED_M1_PATH_SCHEMA_VERSION,
        "exit_path_ordered_fields": list(CLOSED_M1_PATH_FIELDS),
        "exit_path_price_source": "literal_oanda_mid_bid_ask_ohlcv",
        "exit_path_tensor_schema_version": (
            UNIFIED_EXIT_PATH_TENSOR_SCHEMA_VERSION
        ),
        "exit_path_feature_order": list(UNIFIED_EXIT_PATH_FEATURE_ORDER),
        "exit_path_feature_dim": UNIFIED_EXIT_PATH_FEATURE_DIM,
        "exit_entry_representation_key": (
            UNIFIED_EXIT_ENTRY_REPRESENTATION_KEY
        ),
        "exit_entry_representation_dim": (
            UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM
        ),
        "exit_entry_representation_dtype": "float32",
        "exit_max_path_bars": UNIFIED_EXIT_MAX_PATH_BARS,
        "exit_path_envelope_schema_version": (
            UNIFIED_EXIT_PATH_ENVELOPE_SCHEMA_VERSION
        ),
        "exit_path_envelope_fields": sorted(
            UNIFIED_EXIT_PATH_ENVELOPE_FIELDS
        ),
        "exit_path_encoder_layers": UNIFIED_EXIT_PATH_ENCODER_LAYERS,
        "exit_path_transform": (
            "12 literal prices in bps from executable entry midpoint;"
            "log1p(volume);entry spread bps"
        ),
        "exit_side_order": list(UNIFIED_EXIT_SIDE_ORDER),
        "exit_frozen_entry_surface": UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
        "exit_padding_policy": "right_zero_padding_with_exact_lengths",
        "entry_fill_time_source": "exact_broker_or_replay_fill_timestamp",
        "first_full_m1_bar": "ceil(entry_fill_timestamp,1min)",
        "literal_mid_reconstruction_allowed": False,
        "closed_m1_source_sha256_required": True,
        "path_prefix_truncation_allowed": False,
        "exit_required_hash_keys": list(UNIFIED_EXIT_REQUIRED_HASH_KEYS),
        "external_decision_models_allowed": False,
        "runtime_entry_overrides_allowed": False,
        "runtime_exit_overrides_allowed": False,
    }


def require_unified_entry_exit_contract(
    metadata: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Require one bundle to own both decisions, with no sidecar authority."""

    raw = metadata.get("unified_entry_exit_contract")
    expected = unified_entry_exit_contract_metadata()
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"{context} missing required unified_entry_exit_contract metadata"
        )
    observed = dict(raw)
    mismatches = sorted(
        key
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    )
    unexpected = sorted(set(observed) - set(expected))
    missing = sorted(set(expected) - set(observed))
    if mismatches or unexpected or missing:
        raise RuntimeError(
            f"{context} unified_entry_exit_contract mismatch: "
            f"mismatched={mismatches} missing={missing} unexpected={unexpected}"
        )
    return observed


def require_unified_exit_output(
    output: Mapping[str, Any] | Any,
    *,
    context: str,
    expected_bundle_sha256: str,
    entry_snapshot: Mapping[str, Any],
    exit_path_envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and content-bind the exact same-bundle Exit output."""

    if not isinstance(output, Mapping):
        raise RuntimeError(f"{context} output must be a mapping")
    observed = dict(output)
    missing = sorted(UNIFIED_EXIT_OUTPUT_FIELDS - set(observed))
    unexpected = sorted(set(observed) - UNIFIED_EXIT_OUTPUT_FIELDS)
    if missing or unexpected:
        raise RuntimeError(
            f"{context} Exit output exact schema mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    logits_raw = observed.get(UNIFIED_EXIT_LOGITS_KEY)
    probs_raw = observed.get(UNIFIED_EXIT_PROBS_KEY)
    if not isinstance(logits_raw, (list, tuple)) or len(logits_raw) != 2:
        raise RuntimeError(f"{context} exit logits must have exact length 2")
    if not isinstance(probs_raw, (list, tuple)) or len(probs_raw) != 2:
        raise RuntimeError(f"{context} exit probabilities must have exact length 2")
    try:
        logits = tuple(float(value) for value in logits_raw)
        probs = tuple(float(value) for value in probs_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} Exit outputs must be numeric") from exc
    if not all(math.isfinite(value) for value in (*logits, *probs)):
        raise RuntimeError(f"{context} Exit outputs must be finite")
    if any(value < 0.0 or value > 1.0 for value in probs) or not math.isclose(
        sum(probs),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(f"{context} Exit probabilities are not a simplex")
    if math.isclose(logits[0], logits[1], rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError(f"{context} tied Exit logits have no exact argmax")
    max_logit = max(logits)
    exponentials = tuple(math.exp(value - max_logit) for value in logits)
    denominator = sum(exponentials)
    expected_probs = tuple(value / denominator for value in exponentials)
    if any(
        not math.isclose(
            observed_prob,
            expected_prob,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for observed_prob, expected_prob in zip(probs, expected_probs)
    ):
        raise RuntimeError(
            f"{context} Exit probabilities are not softmax(exit_action_logits)"
        )
    expected_index = 0 if logits[0] > logits[1] else 1
    raw_index = observed.get("exit_action_index")
    if (
        isinstance(raw_index, bool)
        or not isinstance(raw_index, int)
        or raw_index != expected_index
    ):
        raise RuntimeError(
            f"{context} exit_action_index does not match logit argmax"
        )
    if probs[expected_index] <= probs[1 - expected_index]:
        raise RuntimeError(
            f"{context} probability argmax does not match logit argmax"
        )
    if observed.get("action") != UNIFIED_EXIT_ACTION_ORDER[expected_index]:
        raise RuntimeError(f"{context} action does not match logit argmax")
    if observed.get("decision_source") != "unified_model":
        raise RuntimeError(f"{context} decision_source must be unified_model")
    for key in UNIFIED_EXIT_REQUIRED_HASH_KEYS:
        value = observed.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"{context} {key} must be lowercase sha256")
    if (
        not isinstance(expected_bundle_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256) is None
    ):
        raise RuntimeError(f"{context} expected bundle identity is invalid")
    if not isinstance(entry_snapshot, Mapping) or not entry_snapshot:
        raise RuntimeError(f"{context} entry snapshot is missing")
    if not isinstance(exit_path_envelope, Mapping) or not exit_path_envelope:
        raise RuntimeError(f"{context} Exit path envelope is missing")
    expected_hashes = {
        "bundle_sha256": expected_bundle_sha256,
        "entry_snapshot_sha256": canonical_unified_evidence_sha256(
            entry_snapshot
        ),
        "exit_path_envelope_sha256": canonical_unified_evidence_sha256(
            exit_path_envelope
        ),
    }
    output_payload = {
        key: value
        for key, value in observed.items()
        if key != "output_evidence_sha256"
    }
    expected_hashes["output_evidence_sha256"] = (
        canonical_unified_evidence_sha256(output_payload)
    )
    mismatched_hashes = sorted(
        key
        for key, expected in expected_hashes.items()
        if observed[key] != expected
    )
    if mismatched_hashes:
        raise RuntimeError(
            f"{context} Exit content hash mismatch: {mismatched_hashes}"
        )
    return observed
