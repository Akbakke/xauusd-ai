"""Exact content envelope for every tensor consumed by unified Exit.

The envelope is deliberately small: immutable artifact identities plus
shape/dtype/content hashes of the actual M1 and MTF arrays.  It is built before
the model call and persisted beside the output, so a HOLD/EXIT_NOW result can
never be detached from the bytes, clocks, side, quotes, Entry evidence or path
that produced it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_decision_token_v1 import (
    ENTRY_DECISION_TOKEN_KEY,
    entry_decision_token_tensor,
    require_entry_decision_token_snapshot,
)


UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION = (
    "gx1_unified_exit_full_input_envelope_v4"
)
UNIFIED_EXIT_INPUT_ARRAY_HASH_SCHEMA_VERSION = (
    "gx1_unified_exit_array_bytes_v1"
)
UNIFIED_EXIT_MTF_CACHE_BINDING_FIELDS = frozenset(
    {"cache_identity_sha256", "manifest_sha256"}
)
UNIFIED_EXIT_INPUT_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "decision_time",
        "decision_identity",
        "side",
        "entry_bid",
        "entry_ask",
        "bundle_sha256",
        "entry_snapshot_sha256",
        "entry_decision_token_snapshot",
        "exit_path_envelope_sha256",
        "m1_feature_window",
        "mtf_cache_binding",
        "mtf_feature_order_sha256",
        "per_tf_seq_lens",
        "mtf_windows",
        "mtf_last_row_sha256",
        "input_envelope_sha256",
    }
)
_M1_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "decision_time",
        "sequence_first_time",
        "sequence_last_time",
        "dataset_run_id",
        "pair_generation_id",
        "feature_base_sha256",
        "feature_manifest_sha256",
        "feature_field_order_sha256",
        "sequence_bars",
        "signal",
        "snap",
        "ctx_cont",
        "ctx_cat",
        "content_sha256",
    }
)
_ARRAY_DESCRIPTOR_FIELDS = frozenset(
    {"schema_version", "dtype", "shape", "sha256"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"UNIFIED_EXIT_INPUT_{label}_SHA256_INVALID")
    return value


def _raw_float32_row_sha256(value: Any, *, label: str) -> str:
    try:
        array = np.ascontiguousarray(value, dtype=np.dtype("<f4"))
    except Exception as exc:
        raise RuntimeError(
            f"UNIFIED_EXIT_INPUT_{label}_LAST_ROW_INVALID"
        ) from exc
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise RuntimeError(f"UNIFIED_EXIT_INPUT_{label}_LAST_ROW_INVALID")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _canonical_timestamp(value: Any, *, label: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(
            f"UNIFIED_EXIT_INPUT_{label}_TIMESTAMP_INVALID"
        ) from exc
    if (
        pd.isna(timestamp)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != pd.Timedelta(0)
        or timestamp != timestamp.floor("min")
    ):
        raise RuntimeError(f"UNIFIED_EXIT_INPUT_{label}_TIMESTAMP_INVALID")
    return timestamp.tz_convert("UTC").isoformat()


def canonical_exit_array_descriptor(
    value: Any,
    *,
    dtype: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    """Return a platform-stable descriptor of the exact consumed bytes."""

    if dtype == "float32":
        array = np.asarray(value, dtype=np.dtype("<f4"))
        dtype_label = "float32_le"
    elif dtype == "int64":
        array = np.asarray(value, dtype=np.dtype("<i8"))
        dtype_label = "int64_le"
    else:  # pragma: no cover - fixed internal callers
        raise AssertionError(f"unsupported canonical Exit dtype {dtype!r}")
    array = np.ascontiguousarray(array)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise RuntimeError(f"UNIFIED_EXIT_INPUT_{label}_ARRAY_INVALID")
    header = {
        "schema_version": UNIFIED_EXIT_INPUT_ARRAY_HASH_SCHEMA_VERSION,
        "dtype": dtype_label,
        "shape": list(expected_shape),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return {**header, "sha256": digest.hexdigest()}


def _require_array_descriptor(
    value: Any,
    *,
    dtype: str,
    expected_shape: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ARRAY_DESCRIPTOR_FIELDS:
        raise RuntimeError(f"UNIFIED_EXIT_INPUT_{label}_DESCRIPTOR_INVALID")
    observed = dict(value)
    expected_dtype = "float32_le" if dtype == "float32" else "int64_le"
    if (
        observed["schema_version"]
        != UNIFIED_EXIT_INPUT_ARRAY_HASH_SCHEMA_VERSION
        or observed["dtype"] != expected_dtype
        or observed["shape"] != list(expected_shape)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_INPUT_{label}_DESCRIPTOR_INVALID")
    _require_sha256(observed["sha256"], label=f"{label}_ARRAY")
    return observed


def build_unified_exit_input_envelope(
    *,
    decision_time: Any,
    decision_identity: str,
    side: str,
    entry_bid: float,
    entry_ask: float,
    bundle_sha256: str,
    entry_snapshot: Mapping[str, Any],
    entry_decision_token_snapshot: Mapping[str, Any],
    exit_path_envelope: Mapping[str, Any],
    m1_feature_window: Mapping[str, Any],
    mtf_windows: Mapping[str, Any],
    mtf_cache_binding: Mapping[str, Any],
    per_tf_seq_lens: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only admitted identity for one unified-Exit model call."""

    from gx1.contracts.entry_exit_feature_surface_v1 import (
        require_m1_feature_window,
    )
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
    from gx1.models.entry_v10.direction_decision_contract import (
        canonical_unified_evidence_sha256,
        require_unified_exit_path_envelope,
    )

    decision_ts = _canonical_timestamp(decision_time, label="DECISION")
    if (
        not isinstance(decision_identity, str)
        or not decision_identity
        or decision_identity.strip() != decision_identity
        or "\x00" in decision_identity
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_DECISION_IDENTITY_INVALID")
    if side not in ("long", "short"):
        raise RuntimeError("UNIFIED_EXIT_INPUT_SIDE_INVALID")
    try:
        bid = float(entry_bid)
        ask = float(entry_ask)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_QUOTES_INVALID") from exc
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= bid:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_QUOTES_INVALID")
    bundle_sha = _require_sha256(bundle_sha256, label="BUNDLE")
    if not isinstance(entry_snapshot, Mapping) or not entry_snapshot:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_SNAPSHOT_INVALID")
    token_snapshot = require_entry_decision_token_snapshot(
        entry_decision_token_snapshot
    )
    path = require_unified_exit_path_envelope(
        exit_path_envelope,
        context="UNIFIED_EXIT_INPUT",
    )
    if path["last_closed_m1_bar_ts"] != decision_ts:
        raise RuntimeError("UNIFIED_EXIT_INPUT_PATH_CLOCK_MISMATCH")
    try:
        alias = np.asarray(
            entry_snapshot[ENTRY_DECISION_TOKEN_KEY],
            dtype=np.dtype("<f4"),
        ).reshape(-1)
    except Exception as exc:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_TOKEN_ALIAS_INVALID") from exc
    token = entry_decision_token_tensor(token_snapshot)
    if alias.shape != token.shape or not np.array_equal(alias, token):
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_TOKEN_ALIAS_MISMATCH")
    if (
        token_snapshot["contract_mode"] != MODEL_NATIVE_CONTRACT_MODE
        or token_snapshot["fill_time"] != path["entry_fill_ts"]
        or token_snapshot["trade_identity"] != decision_identity
        or token_snapshot["side"] != side
        or float(token_snapshot["entry_bid"]) != bid
        or float(token_snapshot["entry_ask"]) != ask
        or token_snapshot["model_direction"]
        != ("LONG" if side == "long" else "SHORT")
        or entry_snapshot.get("model_direction")
        != token_snapshot["model_direction"]
        or entry_snapshot.get("model_direction_index")
        != token_snapshot["model_direction_index"]
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_TOKEN_BINDING_MISMATCH")
    if (
        token_snapshot["model_identity_kind"] == "bundle_sha256"
        and token_snapshot["model_identity_sha256"] != bundle_sha
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_TOKEN_BUNDLE_MISMATCH")
    window = require_m1_feature_window(
        dict(m1_feature_window),
        context="UNIFIED_EXIT_INPUT",
    )
    if window["decision_time"] != decision_ts:
        raise RuntimeError("UNIFIED_EXIT_INPUT_M1_CLOCK_MISMATCH")

    m1_descriptor: dict[str, Any] = {
        key: window[key]
        for key in (
            "schema_version",
            "decision_time",
            "sequence_first_time",
            "sequence_last_time",
            "dataset_run_id",
            "pair_generation_id",
            "feature_base_sha256",
            "feature_manifest_sha256",
            "feature_field_order_sha256",
            "sequence_bars",
        )
    }
    m1_descriptor.update(
        {
            "signal": canonical_exit_array_descriptor(
                window["signal"],
                dtype="float32",
                expected_shape=(EXIT_FEATURE_SEQUENCE_BARS, MODEL_NATIVE_SIGNAL_DIM),
                label="M1_SIGNAL",
            ),
            "snap": canonical_exit_array_descriptor(
                window["snap"],
                dtype="float32",
                expected_shape=(MODEL_NATIVE_SIGNAL_DIM,),
                label="M1_SNAP",
            ),
            "ctx_cont": canonical_exit_array_descriptor(
                window["ctx_cont"],
                dtype="float32",
                expected_shape=(MODEL_NATIVE_CTX_CONT_DIM,),
                label="M1_CTX_CONT",
            ),
            "ctx_cat": canonical_exit_array_descriptor(
                window["ctx_cat"],
                dtype="int64",
                expected_shape=(MODEL_NATIVE_CTX_CAT_DIM,),
                label="M1_CTX_CAT",
            ),
        }
    )
    m1_descriptor["content_sha256"] = _canonical_json_sha256(m1_descriptor)

    route = tuple(EXIT_MTF_CONTEXT_TIMEFRAMES)
    if not isinstance(per_tf_seq_lens, Mapping) or tuple(per_tf_seq_lens) != route:
        raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_SEQUENCE_LENGTHS_INVALID")
    lengths: dict[str, int] = {}
    for timeframe in route:
        value = per_tf_seq_lens[timeframe]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_SEQUENCE_LENGTHS_INVALID")
        lengths[timeframe] = int(value)
    if not isinstance(mtf_windows, Mapping) or tuple(mtf_windows) != route:
        raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_WINDOWS_INVALID")
    mtf_width = len(MULTI_TF_PER_BAR_FEATURES_V4)
    mtf_descriptors = {
        timeframe: canonical_exit_array_descriptor(
            mtf_windows[timeframe],
            dtype="float32",
            expected_shape=(lengths[timeframe], mtf_width),
            label=f"MTF_{timeframe}",
        )
        for timeframe in route
    }
    mtf_last_row_sha256 = {
        timeframe.lower(): _raw_float32_row_sha256(
            np.asarray(mtf_windows[timeframe])[-1],
            label=f"MTF_{timeframe}",
        )
        for timeframe in route
    }
    if (
        not isinstance(mtf_cache_binding, Mapping)
        or set(mtf_cache_binding) != UNIFIED_EXIT_MTF_CACHE_BINDING_FIELDS
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_CACHE_BINDING_INVALID")
    cache_binding = dict(mtf_cache_binding)
    for key in sorted(UNIFIED_EXIT_MTF_CACHE_BINDING_FIELDS):
        _require_sha256(cache_binding[key], label=f"MTF_CACHE_{key.upper()}")

    envelope: dict[str, Any] = {
        "schema_version": UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION,
        "decision_time": decision_ts,
        "decision_identity": decision_identity,
        "side": side,
        "entry_bid": bid,
        "entry_ask": ask,
        "bundle_sha256": bundle_sha,
        "entry_snapshot_sha256": canonical_unified_evidence_sha256(
            entry_snapshot
        ),
        "entry_decision_token_snapshot": token_snapshot,
        "exit_path_envelope_sha256": canonical_unified_evidence_sha256(path),
        "m1_feature_window": m1_descriptor,
        "mtf_cache_binding": cache_binding,
        "mtf_feature_order_sha256": _canonical_json_sha256(
            list(MULTI_TF_PER_BAR_FEATURES_V4)
        ),
        "per_tf_seq_lens": lengths,
        "mtf_windows": mtf_descriptors,
        "mtf_last_row_sha256": mtf_last_row_sha256,
    }
    envelope["input_envelope_sha256"] = _canonical_json_sha256(envelope)
    return require_unified_exit_input_envelope(envelope)


def require_unified_exit_input_envelope(value: Any) -> dict[str, Any]:
    """Validate a persisted full-input identity without inventing source bytes."""

    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4

    if not isinstance(value, Mapping) or set(value) != UNIFIED_EXIT_INPUT_ENVELOPE_FIELDS:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_INVALID")
    envelope = dict(value)
    if envelope["schema_version"] != UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENVELOPE_VERSION_INVALID")
    decision_ts = _canonical_timestamp(envelope["decision_time"], label="DECISION")
    if decision_ts != envelope["decision_time"]:
        raise RuntimeError("UNIFIED_EXIT_INPUT_DECISION_TIMESTAMP_INVALID")
    if (
        not isinstance(envelope["decision_identity"], str)
        or not envelope["decision_identity"]
        or envelope["decision_identity"].strip() != envelope["decision_identity"]
        or "\x00" in envelope["decision_identity"]
        or envelope["side"] not in ("long", "short")
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENVELOPE_IDENTITY_INVALID")
    try:
        bid = float(envelope["entry_bid"])
        ask = float(envelope["entry_ask"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_QUOTES_INVALID") from exc
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= bid:
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_QUOTES_INVALID")
    for key in (
        "bundle_sha256",
        "entry_snapshot_sha256",
        "exit_path_envelope_sha256",
        "mtf_feature_order_sha256",
        "input_envelope_sha256",
    ):
        _require_sha256(envelope[key], label=key.upper())
    token_snapshot = require_entry_decision_token_snapshot(
        envelope["entry_decision_token_snapshot"]
    )
    if (
        token_snapshot["contract_mode"] != MODEL_NATIVE_CONTRACT_MODE
        or token_snapshot["trade_identity"] != envelope["decision_identity"]
        or token_snapshot["side"] != envelope["side"]
        or float(token_snapshot["entry_bid"]) != bid
        or float(token_snapshot["entry_ask"]) != ask
        or (
            token_snapshot["model_identity_kind"] == "bundle_sha256"
            and token_snapshot["model_identity_sha256"]
            != envelope["bundle_sha256"]
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENTRY_TOKEN_BINDING_MISMATCH")

    m1 = envelope["m1_feature_window"]
    if not isinstance(m1, Mapping) or set(m1) != _M1_DESCRIPTOR_FIELDS:
        raise RuntimeError("UNIFIED_EXIT_INPUT_M1_DESCRIPTOR_INVALID")
    m1 = dict(m1)
    if (
        m1["decision_time"] != decision_ts
        or _canonical_timestamp(
            m1["sequence_first_time"], label="M1_SEQUENCE_FIRST"
        )
        != m1["sequence_first_time"]
        or _canonical_timestamp(
            m1["sequence_last_time"], label="M1_SEQUENCE_LAST"
        )
        != m1["sequence_last_time"]
        or not pd.Timestamp(m1["sequence_first_time"])
        < pd.Timestamp(m1["sequence_last_time"])
        or m1["sequence_last_time"] != decision_ts
        or m1["sequence_bars"] != EXIT_FEATURE_SEQUENCE_BARS
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_M1_DESCRIPTOR_INVALID")
    for key in ("dataset_run_id", "pair_generation_id"):
        if not isinstance(m1[key], str) or not m1[key]:
            raise RuntimeError("UNIFIED_EXIT_INPUT_M1_DESCRIPTOR_INVALID")
    for key in (
        "feature_base_sha256",
        "feature_manifest_sha256",
        "feature_field_order_sha256",
        "content_sha256",
    ):
        _require_sha256(m1[key], label=f"M1_{key.upper()}")
    _require_array_descriptor(
        m1["signal"],
        dtype="float32",
        expected_shape=(EXIT_FEATURE_SEQUENCE_BARS, MODEL_NATIVE_SIGNAL_DIM),
        label="M1_SIGNAL",
    )
    _require_array_descriptor(
        m1["snap"],
        dtype="float32",
        expected_shape=(MODEL_NATIVE_SIGNAL_DIM,),
        label="M1_SNAP",
    )
    _require_array_descriptor(
        m1["ctx_cont"],
        dtype="float32",
        expected_shape=(MODEL_NATIVE_CTX_CONT_DIM,),
        label="M1_CTX_CONT",
    )
    _require_array_descriptor(
        m1["ctx_cat"],
        dtype="int64",
        expected_shape=(MODEL_NATIVE_CTX_CAT_DIM,),
        label="M1_CTX_CAT",
    )
    m1_payload = {key: item for key, item in m1.items() if key != "content_sha256"}
    if m1["content_sha256"] != _canonical_json_sha256(m1_payload):
        raise RuntimeError("UNIFIED_EXIT_INPUT_M1_CONTENT_HASH_MISMATCH")

    route = tuple(EXIT_MTF_CONTEXT_TIMEFRAMES)
    lengths = envelope["per_tf_seq_lens"]
    windows = envelope["mtf_windows"]
    mtf_last_row_sha256 = envelope["mtf_last_row_sha256"]
    if (
        not isinstance(lengths, Mapping)
        or tuple(lengths) != route
        or not isinstance(windows, Mapping)
        or tuple(windows) != route
        or not isinstance(mtf_last_row_sha256, Mapping)
        or set(mtf_last_row_sha256)
        != {timeframe.lower() for timeframe in route}
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_DESCRIPTOR_INVALID")
    mtf_width = len(MULTI_TF_PER_BAR_FEATURES_V4)
    for timeframe in route:
        length = lengths[timeframe]
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_DESCRIPTOR_INVALID")
        _require_array_descriptor(
            windows[timeframe],
            dtype="float32",
            expected_shape=(length, mtf_width),
            label=f"MTF_{timeframe}",
        )
        _require_sha256(
            mtf_last_row_sha256[timeframe.lower()],
            label=f"MTF_{timeframe}_LAST_ROW",
        )
    cache = envelope["mtf_cache_binding"]
    if not isinstance(cache, Mapping) or set(cache) != UNIFIED_EXIT_MTF_CACHE_BINDING_FIELDS:
        raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_CACHE_BINDING_INVALID")
    for key in sorted(UNIFIED_EXIT_MTF_CACHE_BINDING_FIELDS):
        _require_sha256(cache[key], label=f"MTF_CACHE_{key.upper()}")
    expected_order_sha = _canonical_json_sha256(list(MULTI_TF_PER_BAR_FEATURES_V4))
    if envelope["mtf_feature_order_sha256"] != expected_order_sha:
        raise RuntimeError("UNIFIED_EXIT_INPUT_MTF_FEATURE_ORDER_MISMATCH")
    payload = {
        key: item
        for key, item in envelope.items()
        if key != "input_envelope_sha256"
    }
    if envelope["input_envelope_sha256"] != _canonical_json_sha256(payload):
        raise RuntimeError("UNIFIED_EXIT_INPUT_ENVELOPE_HASH_MISMATCH")
    return envelope


__all__ = (
    "UNIFIED_EXIT_INPUT_ARRAY_HASH_SCHEMA_VERSION",
    "UNIFIED_EXIT_INPUT_ENVELOPE_FIELDS",
    "UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION",
    "build_unified_exit_input_envelope",
    "canonical_exit_array_descriptor",
    "require_unified_exit_input_envelope",
)
