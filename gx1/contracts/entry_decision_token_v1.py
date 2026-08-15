"""Exact learned Entry-decision token and immutable fill snapshot.

The model token is a learned projection of every declared tensor on the final
Entry decision path.  Persistence stores the exact little-endian float32 bytes
and binds them to the model/normalization identity and the trade fill.  Exit
consumers decode this snapshot; they never reconstruct a token from later bars.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


ENTRY_DECISION_TOKEN_PROJECTION_SCHEMA_VERSION = (
    "gx1_entry_decision_token_projection_v2"
)
ENTRY_DECISION_TOKEN_SNAPSHOT_SCHEMA_VERSION = (
    "gx1_entry_decision_token_fill_snapshot_v2"
)
ENTRY_DECISION_TOKEN_KEY = "entry_decision_representation"
ENTRY_DECISION_TOKEN_SNAPSHOT_KEY = "entry_decision_token_snapshot"
ENTRY_DECISION_TOKEN_DIM = 128
ENTRY_DECISION_TOKEN_DTYPE = "float32_le"
ENTRY_DECISION_TOKEN_COMPONENTS = (
    ("local_model_native_representation", 128),
    ("final_model_native_representation", 128),
    ("multi_timeframe_representation", 128),
    ("family_context_representation", 128),
    ("entry_q_joint_hidden", 128),
    ("entry_action_q_bps", 3),
)
ENTRY_DECISION_TOKEN_SOURCE_DIM = sum(
    width for _name, width in ENTRY_DECISION_TOKEN_COMPONENTS
)
ENTRY_DECISION_TOKEN_MODEL_IDENTITY_KINDS = (
    "bundle_sha256",
    "training_state_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "projection_schema_version",
        "source_components_sha256",
        "decision_time",
        "fill_time",
        "model_identity_kind",
        "model_identity_sha256",
        "input_normalization_sha256",
        "contract_mode",
        "model_direction_index",
        "model_direction",
        "side",
        "entry_bid",
        "entry_ask",
        "trade_identity",
        "dtype",
        "shape",
        "tensor_bytes_b64",
        "tensor_sha256",
        "snapshot_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"ENTRY_DECISION_TOKEN_{label}_SHA256_INVALID")
    return value


def _canonical_utc_timestamp(value: Any, *, label: str) -> str:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(
            f"ENTRY_DECISION_TOKEN_{label}_TIMESTAMP_INVALID"
        ) from exc
    if (
        pd.isna(parsed)
        or parsed.tzinfo is None
        or parsed.utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError(f"ENTRY_DECISION_TOKEN_{label}_TIMESTAMP_INVALID")
    return parsed.tz_convert("UTC").isoformat()


def entry_decision_token_projection_metadata() -> dict[str, Any]:
    """Return the exact ordered learned projection contract."""

    components: list[dict[str, Any]] = []
    start = 0
    for name, width in ENTRY_DECISION_TOKEN_COMPONENTS:
        stop = start + width
        components.append(
            {"name": name, "width": width, "start": start, "stop": stop}
        )
        start = stop
    if start != ENTRY_DECISION_TOKEN_SOURCE_DIM:  # pragma: no cover
        raise AssertionError("Entry-decision token source width drift")
    payload = {
        "schema_version": ENTRY_DECISION_TOKEN_PROJECTION_SCHEMA_VERSION,
        "components": components,
        "source_dim": ENTRY_DECISION_TOKEN_SOURCE_DIM,
        "token_dim": ENTRY_DECISION_TOKEN_DIM,
        "projection": "LayerNorm+Linear+GELU",
        "decision_source": (
            "joint_local_multi_timeframe_family_context_pre_q_evidence_"
            "plus_raw_entry_action_q_bps"
        ),
        "handwritten_component_weights": False,
        "recomputed_after_fill": False,
        "exit_consumption": "exact_frozen_fill_snapshot_tensor_bytes",
    }
    payload["components_sha256"] = _canonical_sha256(components)
    return payload


def require_entry_decision_token_projection_metadata(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected = entry_decision_token_projection_metadata()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_ENTRY_DECISION_TOKEN_PROJECTION_INVALID]")
    return expected


def _canonical_tensor_bytes(value: Any) -> bytes:
    array = np.asarray(value, dtype=np.dtype("<f4"))
    array = np.ascontiguousarray(array.reshape(-1))
    if array.shape != (ENTRY_DECISION_TOKEN_DIM,) or not np.isfinite(array).all():
        raise RuntimeError("ENTRY_DECISION_TOKEN_TENSOR_INVALID")
    return array.tobytes(order="C")


def _tensor_sha256(raw: bytes) -> str:
    header = {
        "dtype": ENTRY_DECISION_TOKEN_DTYPE,
        "shape": [ENTRY_DECISION_TOKEN_DIM],
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    digest.update(b"\x00")
    digest.update(raw)
    return digest.hexdigest()


def build_entry_decision_token_snapshot(
    *,
    token: Any,
    decision_time: Any,
    fill_time: Any,
    model_identity_kind: str,
    model_identity_sha256: str,
    input_normalization_sha256: str,
    contract_mode: str,
    model_direction_index: int,
    model_direction: str,
    side: str,
    entry_bid: float,
    entry_ask: float,
    trade_identity: str,
) -> dict[str, Any]:
    """Freeze exact token bytes against the Entry fill identity."""

    raw = _canonical_tensor_bytes(token)
    decision_ts = _canonical_utc_timestamp(decision_time, label="DECISION")
    fill_ts = _canonical_utc_timestamp(fill_time, label="FILL")
    if pd.Timestamp(fill_ts) < pd.Timestamp(decision_ts):
        raise RuntimeError("ENTRY_DECISION_TOKEN_FILL_PRECEDES_DECISION")
    if model_identity_kind not in ENTRY_DECISION_TOKEN_MODEL_IDENTITY_KINDS:
        raise RuntimeError("ENTRY_DECISION_TOKEN_MODEL_IDENTITY_KIND_INVALID")
    model_sha = _require_sha256(
        model_identity_sha256,
        label="MODEL_IDENTITY",
    )
    normalization_sha = _require_sha256(
        input_normalization_sha256,
        label="INPUT_NORMALIZATION",
    )
    if (
        not isinstance(contract_mode, str)
        or not contract_mode
        or contract_mode.strip() != contract_mode
        or "\x00" in contract_mode
    ):
        raise RuntimeError("ENTRY_DECISION_TOKEN_CONTRACT_MODE_INVALID")
    if (
        isinstance(model_direction_index, bool)
        or not isinstance(model_direction_index, (int, np.integer))
        or int(model_direction_index) not in (0, 1)
    ):
        raise RuntimeError("ENTRY_DECISION_TOKEN_DIRECTION_INVALID")
    direction_index = int(model_direction_index)
    expected_direction = ("LONG", "SHORT")[direction_index]
    expected_side = ("long", "short")[direction_index]
    if model_direction != expected_direction or side != expected_side:
        raise RuntimeError("ENTRY_DECISION_TOKEN_DIRECTION_SIDE_MISMATCH")
    try:
        bid = float(entry_bid)
        ask = float(entry_ask)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ENTRY_DECISION_TOKEN_ENTRY_QUOTES_INVALID") from exc
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= bid:
        raise RuntimeError("ENTRY_DECISION_TOKEN_ENTRY_QUOTES_INVALID")
    if (
        not isinstance(trade_identity, str)
        or not trade_identity
        or trade_identity.strip() != trade_identity
        or any(separator in trade_identity for separator in ("/", "\\", "\x00"))
    ):
        raise RuntimeError("ENTRY_DECISION_TOKEN_TRADE_IDENTITY_INVALID")

    projection = entry_decision_token_projection_metadata()
    snapshot: dict[str, Any] = {
        "schema_version": ENTRY_DECISION_TOKEN_SNAPSHOT_SCHEMA_VERSION,
        "projection_schema_version": projection["schema_version"],
        "source_components_sha256": projection["components_sha256"],
        "decision_time": decision_ts,
        "fill_time": fill_ts,
        "model_identity_kind": model_identity_kind,
        "model_identity_sha256": model_sha,
        "input_normalization_sha256": normalization_sha,
        "contract_mode": contract_mode,
        "model_direction_index": direction_index,
        "model_direction": expected_direction,
        "side": expected_side,
        "entry_bid": bid,
        "entry_ask": ask,
        "trade_identity": trade_identity,
        "dtype": ENTRY_DECISION_TOKEN_DTYPE,
        "shape": [ENTRY_DECISION_TOKEN_DIM],
        "tensor_bytes_b64": base64.b64encode(raw).decode("ascii"),
        "tensor_sha256": _tensor_sha256(raw),
    }
    snapshot["snapshot_sha256"] = _canonical_sha256(snapshot)
    return require_entry_decision_token_snapshot(snapshot)


def require_entry_decision_token_snapshot(value: Any) -> dict[str, Any]:
    """Validate the complete snapshot and its byte/hash chain."""

    if not isinstance(value, Mapping) or set(value) != _SNAPSHOT_FIELDS:
        raise RuntimeError("ENTRY_DECISION_TOKEN_SNAPSHOT_SCHEMA_INVALID")
    snapshot = dict(value)
    projection = entry_decision_token_projection_metadata()
    if (
        snapshot["schema_version"] != ENTRY_DECISION_TOKEN_SNAPSHOT_SCHEMA_VERSION
        or snapshot["projection_schema_version"] != projection["schema_version"]
        or snapshot["source_components_sha256"] != projection["components_sha256"]
        or snapshot["dtype"] != ENTRY_DECISION_TOKEN_DTYPE
        or snapshot["shape"] != [ENTRY_DECISION_TOKEN_DIM]
    ):
        raise RuntimeError("ENTRY_DECISION_TOKEN_SNAPSHOT_CONTRACT_INVALID")
    try:
        raw = base64.b64decode(
            snapshot["tensor_bytes_b64"],
            validate=True,
        )
    except Exception as exc:
        raise RuntimeError("ENTRY_DECISION_TOKEN_BYTES_INVALID") from exc
    if len(raw) != ENTRY_DECISION_TOKEN_DIM * np.dtype("<f4").itemsize:
        raise RuntimeError("ENTRY_DECISION_TOKEN_BYTES_INVALID")
    tensor = np.frombuffer(raw, dtype=np.dtype("<f4"))
    if tensor.shape != (ENTRY_DECISION_TOKEN_DIM,) or not np.isfinite(tensor).all():
        raise RuntimeError("ENTRY_DECISION_TOKEN_BYTES_INVALID")
    if snapshot["tensor_sha256"] != _tensor_sha256(raw):
        raise RuntimeError("ENTRY_DECISION_TOKEN_TENSOR_HASH_MISMATCH")
    payload = {
        key: item for key, item in snapshot.items() if key != "snapshot_sha256"
    }
    _require_sha256(snapshot["snapshot_sha256"], label="SNAPSHOT")
    if snapshot["snapshot_sha256"] != _canonical_sha256(payload):
        raise RuntimeError("ENTRY_DECISION_TOKEN_SNAPSHOT_HASH_MISMATCH")

    # Re-validate non-byte fields without recursively rebuilding the snapshot.
    decision_ts = _canonical_utc_timestamp(snapshot["decision_time"], label="DECISION")
    fill_ts = _canonical_utc_timestamp(snapshot["fill_time"], label="FILL")
    if decision_ts != snapshot["decision_time"] or fill_ts != snapshot["fill_time"]:
        raise RuntimeError("ENTRY_DECISION_TOKEN_TIMESTAMP_NOT_CANONICAL")
    if pd.Timestamp(fill_ts) < pd.Timestamp(decision_ts):
        raise RuntimeError("ENTRY_DECISION_TOKEN_FILL_PRECEDES_DECISION")
    _require_sha256(snapshot["model_identity_sha256"], label="MODEL_IDENTITY")
    _require_sha256(
        snapshot["input_normalization_sha256"],
        label="INPUT_NORMALIZATION",
    )
    if snapshot["model_identity_kind"] not in ENTRY_DECISION_TOKEN_MODEL_IDENTITY_KINDS:
        raise RuntimeError("ENTRY_DECISION_TOKEN_MODEL_IDENTITY_KIND_INVALID")
    direction_index = snapshot["model_direction_index"]
    if (
        isinstance(direction_index, bool)
        or not isinstance(direction_index, int)
        or direction_index not in (0, 1)
        or snapshot["model_direction"] != ("LONG", "SHORT")[direction_index]
        or snapshot["side"] != ("long", "short")[direction_index]
    ):
        raise RuntimeError("ENTRY_DECISION_TOKEN_DIRECTION_SIDE_MISMATCH")
    try:
        bid = float(snapshot["entry_bid"])
        ask = float(snapshot["entry_ask"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ENTRY_DECISION_TOKEN_ENTRY_QUOTES_INVALID") from exc
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= bid:
        raise RuntimeError("ENTRY_DECISION_TOKEN_ENTRY_QUOTES_INVALID")
    if (
        not isinstance(snapshot["contract_mode"], str)
        or not snapshot["contract_mode"].strip()
        or snapshot["contract_mode"].strip() != snapshot["contract_mode"]
        or not isinstance(snapshot["trade_identity"], str)
        or not snapshot["trade_identity"]
        or snapshot["trade_identity"].strip() != snapshot["trade_identity"]
        or any(
            separator in snapshot["trade_identity"]
            for separator in ("/", "\\", "\x00")
        )
    ):
        raise RuntimeError("ENTRY_DECISION_TOKEN_IDENTITY_INVALID")
    return snapshot


def entry_decision_token_tensor(value: Any) -> np.ndarray:
    """Return an owned float32 tensor decoded only from the frozen bytes."""

    snapshot = require_entry_decision_token_snapshot(value)
    raw = base64.b64decode(snapshot["tensor_bytes_b64"], validate=True)
    return np.frombuffer(raw, dtype=np.dtype("<f4")).copy()


def require_entry_decision_token_bindings(
    value: Any,
    *,
    raw_token_alias: Any,
    decision_time: Any,
    fill_time: Any,
    model_identity_kind: str,
    model_identity_sha256: str,
    input_normalization_sha256: str,
    contract_mode: str,
    model_direction_index: int,
    model_direction: str,
    side: str,
    entry_bid: float,
    entry_ask: float,
    trade_identity: str,
    context: str,
) -> dict[str, Any]:
    """Require exact fill bindings and float32 byte parity with the raw alias."""

    snapshot = require_entry_decision_token_snapshot(value)
    expected = build_entry_decision_token_snapshot(
        token=raw_token_alias,
        decision_time=decision_time,
        fill_time=fill_time,
        model_identity_kind=model_identity_kind,
        model_identity_sha256=model_identity_sha256,
        input_normalization_sha256=input_normalization_sha256,
        contract_mode=contract_mode,
        model_direction_index=model_direction_index,
        model_direction=model_direction,
        side=side,
        entry_bid=entry_bid,
        entry_ask=entry_ask,
        trade_identity=trade_identity,
    )
    if snapshot != expected:
        mismatches = sorted(
            key for key in snapshot if snapshot.get(key) != expected.get(key)
        )
        raise RuntimeError(
            f"[{context}_ENTRY_DECISION_TOKEN_BINDING_MISMATCH] "
            f"fields={mismatches}"
        )
    return snapshot


__all__ = (
    "ENTRY_DECISION_TOKEN_COMPONENTS",
    "ENTRY_DECISION_TOKEN_DIM",
    "ENTRY_DECISION_TOKEN_KEY",
    "ENTRY_DECISION_TOKEN_PROJECTION_SCHEMA_VERSION",
    "ENTRY_DECISION_TOKEN_SNAPSHOT_KEY",
    "ENTRY_DECISION_TOKEN_SNAPSHOT_SCHEMA_VERSION",
    "ENTRY_DECISION_TOKEN_SOURCE_DIM",
    "build_entry_decision_token_snapshot",
    "entry_decision_token_projection_metadata",
    "entry_decision_token_tensor",
    "require_entry_decision_token_bindings",
    "require_entry_decision_token_projection_metadata",
    "require_entry_decision_token_snapshot",
)
