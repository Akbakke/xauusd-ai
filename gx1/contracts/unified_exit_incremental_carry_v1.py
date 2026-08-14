"""Immutable persisted state for the episode-native incremental Exit owner."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)


UNIFIED_EXIT_INCREMENTAL_CARRY_SCHEMA_VERSION = (
    "gx1_unified_exit_incremental_carry_v1"
)
UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256 = "0" * 64
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "step_count",
        "last_closed_m1_bar_ts",
        "trade_identity",
        "side",
        "bundle_sha256",
        "input_normalization_sha256",
        "entry_token_snapshot_sha256",
        "full_path_chain_sha256",
        "input_envelope_sha256",
        "previous_carry_envelope_sha256",
        "mtf_last_row_sha256",
        "tensor_state",
        "tensor_state_sha256",
        "carry_envelope_sha256",
    }
)
_TENSOR_FIELDS = frozenset({"dtype", "shape", "bytes_b64", "sha256"})


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


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeError(f"UNIFIED_EXIT_CARRY_{label}_SHA256_INVALID")
    return value


def _canonical_m1_timestamp(value: Any) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError("UNIFIED_EXIT_CARRY_M1_CLOCK_INVALID") from exc
    if (
        pd.isna(timestamp)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != pd.Timedelta(0)
        or timestamp != timestamp.floor("min")
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_M1_CLOCK_INVALID")
    return timestamp.tz_convert("UTC").isoformat()


def _tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("UNIFIED_EXIT_CARRY_TENSOR_INVALID")
    array = np.ascontiguousarray(
        value.detach().cpu().to(torch.float32).numpy(), dtype=np.dtype("<f4")
    )
    if array.ndim < 1 or not np.isfinite(array).all():
        raise RuntimeError("UNIFIED_EXIT_CARRY_TENSOR_INVALID")
    raw = array.tobytes(order="C")
    return {
        "dtype": "float32_le",
        "shape": list(array.shape),
        "bytes_b64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build_unified_exit_incremental_carry_envelope(
    *,
    tensor_state: Mapping[str, torch.Tensor],
    step_count: int,
    last_closed_m1_bar_ts: Any,
    trade_identity: str,
    side: str,
    bundle_sha256: str,
    input_normalization_sha256: str,
    entry_token_snapshot_sha256: str,
    full_path_chain_sha256: str,
    input_envelope_sha256: str,
    previous_carry_envelope_sha256: str,
    mtf_last_row_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if (
        isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count < 1
        or not isinstance(trade_identity, str)
        or not trade_identity
        or trade_identity.strip() != trade_identity
        or side not in {"long", "short"}
        or not isinstance(tensor_state, Mapping)
        or not tensor_state
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_IDENTITY_INVALID")
    hashes = {
        "bundle_sha256": _require_sha(bundle_sha256, "BUNDLE"),
        "input_normalization_sha256": _require_sha(
            input_normalization_sha256, "NORMALIZATION"
        ),
        "entry_token_snapshot_sha256": _require_sha(
            entry_token_snapshot_sha256, "TOKEN"
        ),
        "full_path_chain_sha256": _require_sha(
            full_path_chain_sha256, "PATH_CHAIN"
        ),
        "input_envelope_sha256": _require_sha(
            input_envelope_sha256, "INPUT_ENVELOPE"
        ),
        "previous_carry_envelope_sha256": _require_sha(
            previous_carry_envelope_sha256, "PREVIOUS"
        ),
    }
    expected_mtf_names = {
        timeframe.lower() for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES
    }
    if (
        not isinstance(mtf_last_row_sha256, Mapping)
        or set(mtf_last_row_sha256) != expected_mtf_names
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_MTF_BINDING_INVALID")
    mtf_hashes = {
        str(name): _require_sha(value, f"MTF_{str(name).upper()}")
        for name, value in mtf_last_row_sha256.items()
    }
    descriptors = {
        str(name): _tensor_descriptor(value)
        for name, value in sorted(tensor_state.items())
    }
    envelope = {
        "schema_version": UNIFIED_EXIT_INCREMENTAL_CARRY_SCHEMA_VERSION,
        "step_count": step_count,
        "last_closed_m1_bar_ts": _canonical_m1_timestamp(
            last_closed_m1_bar_ts
        ),
        "trade_identity": trade_identity,
        "side": side,
        **hashes,
        "mtf_last_row_sha256": mtf_hashes,
        "tensor_state": descriptors,
        "tensor_state_sha256": _canonical_sha256(descriptors),
    }
    envelope["carry_envelope_sha256"] = _canonical_sha256(envelope)
    return require_unified_exit_incremental_carry_envelope(envelope)


def require_unified_exit_incremental_carry_envelope(
    value: Mapping[str, Any],
    *,
    expected_trade_identity: str | None = None,
    expected_side: str | None = None,
    expected_bundle_sha256: str | None = None,
    expected_input_normalization_sha256: str | None = None,
    expected_entry_token_snapshot_sha256: str | None = None,
    expected_full_path_chain_sha256: str | None = None,
    expected_input_envelope_sha256: str | None = None,
    expected_mtf_last_row_sha256: Mapping[str, str] | None = None,
    expected_last_closed_m1_bar_ts: Any | None = None,
    expected_step_count: int | None = None,
    expected_previous_carry_envelope_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TOP_FIELDS:
        raise RuntimeError("UNIFIED_EXIT_CARRY_SCHEMA_INVALID")
    observed = dict(value)
    if observed["schema_version"] != UNIFIED_EXIT_INCREMENTAL_CARRY_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_CARRY_SCHEMA_INVALID")
    canonical_clock = _canonical_m1_timestamp(observed["last_closed_m1_bar_ts"])
    if (
        isinstance(observed["step_count"], bool)
        or not isinstance(observed["step_count"], int)
        or observed["step_count"] < 1
        or observed["side"] not in {"long", "short"}
        or not isinstance(observed["trade_identity"], str)
        or not observed["trade_identity"]
        or observed["trade_identity"].strip() != observed["trade_identity"]
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_IDENTITY_INVALID")
    if (
        observed["step_count"] == 1
        and observed["previous_carry_envelope_sha256"]
        != UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
    ) or (
        observed["step_count"] > 1
        and observed["previous_carry_envelope_sha256"]
        == UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_CHAIN_POSITION_INVALID")
    for field in (
        "bundle_sha256",
        "input_normalization_sha256",
        "entry_token_snapshot_sha256",
        "full_path_chain_sha256",
        "input_envelope_sha256",
        "previous_carry_envelope_sha256",
        "tensor_state_sha256",
        "carry_envelope_sha256",
    ):
        _require_sha(observed[field], field.upper())
    descriptors = observed["tensor_state"]
    if not isinstance(descriptors, Mapping) or not descriptors:
        raise RuntimeError("UNIFIED_EXIT_CARRY_TENSOR_STATE_INVALID")
    for name, descriptor in descriptors.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != _TENSOR_FIELDS
            or descriptor["dtype"] != "float32_le"
            or not isinstance(descriptor["shape"], list)
            or not descriptor["shape"]
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                for dim in descriptor["shape"]
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_CARRY_TENSOR_STATE_INVALID")
        try:
            raw = base64.b64decode(descriptor["bytes_b64"], validate=True)
        except Exception as exc:
            raise RuntimeError("UNIFIED_EXIT_CARRY_TENSOR_BYTES_INVALID") from exc
        expected_bytes = int(np.prod(descriptor["shape"], dtype=np.int64)) * 4
        if (
            len(raw) != expected_bytes
            or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]
            or not np.isfinite(
                np.frombuffer(raw, dtype=np.dtype("<f4"))
            ).all()
        ):
            raise RuntimeError("UNIFIED_EXIT_CARRY_TENSOR_BYTES_INVALID")
    mtf_hashes = observed["mtf_last_row_sha256"]
    if (
        not isinstance(mtf_hashes, Mapping)
        or set(mtf_hashes)
        != {timeframe.lower() for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES}
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_MTF_BINDING_INVALID")
    for name, digest in mtf_hashes.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("UNIFIED_EXIT_CARRY_MTF_BINDING_INVALID")
        _require_sha(digest, f"MTF_{name.upper()}")
    if (
        observed["tensor_state_sha256"] != _canonical_sha256(descriptors)
        or observed["carry_envelope_sha256"]
        != _canonical_sha256(
            {
                key: item
                for key, item in observed.items()
                if key != "carry_envelope_sha256"
            }
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_CONTENT_HASH_INVALID")
    exact = {
        "trade_identity": expected_trade_identity,
        "side": expected_side,
        "bundle_sha256": expected_bundle_sha256,
        "input_normalization_sha256": expected_input_normalization_sha256,
        "entry_token_snapshot_sha256": expected_entry_token_snapshot_sha256,
        "full_path_chain_sha256": expected_full_path_chain_sha256,
        "input_envelope_sha256": expected_input_envelope_sha256,
        "step_count": expected_step_count,
        "previous_carry_envelope_sha256": (
            expected_previous_carry_envelope_sha256
        ),
    }
    if any(expected is not None and observed[field] != expected for field, expected in exact.items()):
        raise RuntimeError("UNIFIED_EXIT_CARRY_EXPECTED_BINDING_MISMATCH")
    if expected_mtf_last_row_sha256 is not None:
        expected_mtf = {
            str(name): _require_sha(value, f"EXPECTED_MTF_{str(name).upper()}")
            for name, value in expected_mtf_last_row_sha256.items()
        }
        if dict(mtf_hashes) != expected_mtf:
            raise RuntimeError("UNIFIED_EXIT_CARRY_EXPECTED_MTF_BINDING_MISMATCH")
    if (
        expected_last_closed_m1_bar_ts is not None
        and canonical_clock
        != _canonical_m1_timestamp(expected_last_closed_m1_bar_ts)
    ):
        raise RuntimeError("UNIFIED_EXIT_CARRY_EXPECTED_CLOCK_MISMATCH")
    return observed


def decode_unified_exit_incremental_carry_tensors(
    value: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observed = require_unified_exit_incremental_carry_envelope(value)
    result: dict[str, torch.Tensor] = {}
    for name, descriptor in observed["tensor_state"].items():
        raw = base64.b64decode(descriptor["bytes_b64"], validate=True)
        array = np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(
            descriptor["shape"]
        ).copy()
        result[name] = torch.from_numpy(array).to(device=device)
    return result


__all__ = (
    "UNIFIED_EXIT_INCREMENTAL_CARRY_GENESIS_SHA256",
    "UNIFIED_EXIT_INCREMENTAL_CARRY_SCHEMA_VERSION",
    "build_unified_exit_incremental_carry_envelope",
    "decode_unified_exit_incremental_carry_tensors",
    "require_unified_exit_incremental_carry_envelope",
)
