"""Positive, state-bound input-scale contract for the five MTF branches."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

import numpy as np

from gx1.features.htf_features import MULTI_TF_TIMEFRAMES_LOWER

SCHEMA_VERSION = "entry_model_native_tf_input_scale_v1"
PARAMETERIZATION = "min_effective_plus_softplus_raw_v1"
STATE_KEY_SEMANTICS = "tf_input_scale_<tf>_stores_unconstrained_raw_scalar"
MIN_EFFECTIVE_SCALE = 1.0e-4
NEUTRAL_EFFECTIVE_INIT = 1.0
TF_NAMES = MULTI_TF_TIMEFRAMES_LOWER


def effective_tf_input_scale_from_raw(raw: float) -> float:
    value = float(raw)
    if not math.isfinite(value):
        raise RuntimeError("[ENTRY_TF_INPUT_SCALE_RAW_NONFINITE]")
    return float(MIN_EFFECTIVE_SCALE + np.logaddexp(0.0, value))


def raw_tf_input_scale_from_effective(effective: float) -> float:
    value = float(effective)
    delta = value - float(MIN_EFFECTIVE_SCALE)
    if not math.isfinite(delta) or delta <= 0.0:
        raise RuntimeError(
            "[ENTRY_TF_INPUT_SCALE_EFFECTIVE_INVALID] "
            f"observed={value!r} required>{MIN_EFFECTIVE_SCALE}"
        )
    # Stable inverse softplus.
    raw = delta + math.log1p(-math.exp(-delta))
    if not math.isfinite(raw):
        raise RuntimeError("[ENTRY_TF_INPUT_SCALE_RAW_INIT_NONFINITE]")
    return float(raw)


def tf_input_scale_raw_sha256(raw: float) -> str:
    value = np.asarray([float(raw)], dtype="<f4")
    if not np.isfinite(value).all():
        raise RuntimeError("[ENTRY_TF_INPUT_SCALE_RAW_NONFINITE]")
    digest = hashlib.sha256()
    digest.update(b"entry_model_native_tf_input_scale_raw_v1\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def build_tf_input_scale_contract(
    *,
    init_effective: Mapping[str, float],
    learned_raw: Mapping[str, float],
) -> dict[str, Any]:
    if tuple(init_effective) != TF_NAMES or tuple(learned_raw) != TF_NAMES:
        raise RuntimeError("[ENTRY_TF_INPUT_SCALE_TF_ORDER_INVALID]")
    initial = {name: float(init_effective[name]) for name in TF_NAMES}
    # State is persisted as float32; quantize before deriving either the hash
    # or effective value so metadata and loaded tensor bytes have one owner.
    raw = {
        name: float(np.float32(learned_raw[name])) for name in TF_NAMES
    }
    for value in initial.values():
        raw_tf_input_scale_from_effective(value)
    learned = {
        name: effective_tf_input_scale_from_raw(raw[name]) for name in TF_NAMES
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "parameterization": PARAMETERIZATION,
        "state_key_semantics": STATE_KEY_SEMANTICS,
        "min_effective_scale": float(MIN_EFFECTIVE_SCALE),
        "init": initial,
        "learned": learned,
        "raw_state_sha256": {
            name: tf_input_scale_raw_sha256(raw[name]) for name in TF_NAMES
        },
    }


def require_tf_input_scale_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("[ENTRY_TF_INPUT_SCALE_CONTRACT_MISSING]")
    data = dict(value)
    expected_keys = {
        "schema_version",
        "enabled",
        "parameterization",
        "state_key_semantics",
        "min_effective_scale",
        "init",
        "learned",
        "raw_state_sha256",
    }
    if set(data) != expected_keys:
        raise RuntimeError("[ENTRY_TF_INPUT_SCALE_CONTRACT_SCHEMA_INVALID]")
    if (
        data["schema_version"] != SCHEMA_VERSION
        or data["enabled"] is not True
        or data["parameterization"] != PARAMETERIZATION
        or data["state_key_semantics"] != STATE_KEY_SEMANTICS
        or float(data["min_effective_scale"]) != float(MIN_EFFECTIVE_SCALE)
    ):
        raise RuntimeError("[ENTRY_TF_INPUT_SCALE_CONTRACT_IDENTITY_INVALID]")
    for field in ("init", "learned", "raw_state_sha256"):
        mapping = data[field]
        if not isinstance(mapping, Mapping) or tuple(mapping) != TF_NAMES:
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_CONTRACT_TF_ORDER_INVALID] field={field}"
            )
    for name in TF_NAMES:
        raw_tf_input_scale_from_effective(float(data["init"][name]))
        learned = float(data["learned"][name])
        if (
            not math.isfinite(learned)
            or learned <= float(MIN_EFFECTIVE_SCALE)
        ):
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_LEARNED_INVALID] tf={name}"
            )
        observed_hash = str(data["raw_state_sha256"][name])
        if (
            len(observed_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in observed_hash)
        ):
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_RAW_HASH_INVALID] tf={name}"
            )
    return data


def require_tf_input_scale_state(
    contract: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> dict[str, float]:
    data = require_tf_input_scale_contract(contract)
    learned: dict[str, float] = {}
    for name in TF_NAMES:
        key = f"tf_input_scale_{name}"
        value = state_dict.get(key)
        if value is None:
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_STATE_MISSING] key={key}"
            )
        try:
            array = value.detach().cpu().numpy()
        except Exception as exc:
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_STATE_INVALID] key={key}"
            ) from exc
        if array.shape != ():
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_STATE_SHAPE_INVALID] "
                f"key={key} shape={array.shape}"
            )
        raw = float(array.item())
        observed_hash = tf_input_scale_raw_sha256(raw)
        if observed_hash != data["raw_state_sha256"][name]:
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_STATE_HASH_MISMATCH] tf={name}"
            )
        effective = effective_tf_input_scale_from_raw(raw)
        if effective != float(data["learned"][name]):
            raise RuntimeError(
                f"[ENTRY_TF_INPUT_SCALE_EFFECTIVE_MISMATCH] tf={name}"
            )
        learned[name] = effective
    return learned
