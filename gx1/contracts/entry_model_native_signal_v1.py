"""Fail-closed signal contract for the model-native XAU entry candidate.

The retired Smart520 surface prepended seven XGBoost-derived values to 34
genuine per-bar price-state fields.  Fresh XAU builds filled those seven values
with constants, while the Transformer still interpreted three of them as
direction-anchor probabilities.  This contract removes that dead bridge from
the input surface and makes the Transformer direction logits model-native.

Of the selected 479 specialist fields, all 305 registered causal full-stack
layer outputs are code-owned and mandatory.  Only the remaining 174 positions
are ranking-owned.  The emitted manifest still owns the audited exact order,
while this module owns the immutable base order, mandatory registry identity,
dimensions, forbidden legacy fields, and validation of the combined surface.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gx1.contracts.signal_bridge_v3 import (
    ORDERED_BRIDGE_FIELDS_V3,
    PER_BAR_PRICE_STATE_FIELDS_V3,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
)


MODEL_NATIVE_SIGNAL_SCHEMA_VERSION = "entry_model_native_signal_v2"
MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION = (
    "entry_model_native_seq513_split_manifest_v2"
)
MODEL_NATIVE_CONTRACT_MODE = "xau_seq513_model_native_direction_v1"
RETIRED_NEUTRAL_BRIDGE_CONTRACT_MODE = "smart_seq520_candidate"
MODEL_NATIVE_DIRECTION_LOGIT_MODE = "model_native"
LEGACY_ANCHOR_DIRECTION_LOGIT_MODE = "xgb_anchor_residual"

MODEL_NATIVE_BASE_FIELDS = tuple(PER_BAR_PRICE_STATE_FIELDS_V3)
FORBIDDEN_LEGACY_BRIDGE_FIELDS = tuple(ORDERED_BRIDGE_FIELDS_V3)
MODEL_NATIVE_BASE_SIGNAL_DIM = 34
MODEL_NATIVE_SELECTED_FEATURE_COUNT = 479
MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT = (
    MODEL_NATIVE_SELECTED_FEATURE_COUNT - MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
)
MODEL_NATIVE_SIGNAL_DIM = 513
MODEL_NATIVE_SEQ_LEN = 96
MODEL_NATIVE_CTX_CONT_DIM = 142
MODEL_NATIVE_CTX_CAT_DIM = 5

if len(MODEL_NATIVE_BASE_FIELDS) != MODEL_NATIVE_BASE_SIGNAL_DIM:
    raise RuntimeError(
        "MODEL_NATIVE_BASE_SIGNAL_DIM_MISMATCH: "
        f"fields={len(MODEL_NATIVE_BASE_FIELDS)} expected={MODEL_NATIVE_BASE_SIGNAL_DIM}"
    )
if set(MODEL_NATIVE_BASE_FIELDS) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_BASE_FIELDS_CONTAIN_FORBIDDEN_BRIDGE_FIELDS")
if MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT != 174:
    raise RuntimeError(
        "MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT_MISMATCH: "
        f"observed={MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT} expected=174"
    )
if set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS) & set(MODEL_NATIVE_BASE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FIELDS_OVERLAP_BASE_FIELDS")
if set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FIELDS_CONTAIN_FORBIDDEN_BRIDGE_FIELDS")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION = (
    "entry_model_native_mandatory_full_stack_v1"
)
MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256 = _sha256_json(
    MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
)


def model_native_mandatory_full_stack_metadata() -> dict[str, Any]:
    """Return the immutable exact family/name registry embedded in artifacts."""

    return {
        "schema_version": MODEL_NATIVE_MANDATORY_FULL_STACK_SCHEMA_VERSION,
        "ordered_family_fields_sha256": MODEL_NATIVE_MANDATORY_FULL_STACK_SHA256,
        "family_count": len(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES),
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "family_order": [
            family for family, _features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        ],
        "family_feature_counts": {
            family: len(features)
            for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        },
        "family_features": {
            family: list(features)
            for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
        },
        "ordered_mandatory_fields": list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS),
    }


MODEL_NATIVE_STATIC_CONTRACT_SHA256 = _sha256_json(
    {
        "schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "base_fields": MODEL_NATIVE_BASE_FIELDS,
        "forbidden_legacy_bridge_fields": FORBIDDEN_LEGACY_BRIDGE_FIELDS,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
    }
)


def ordered_model_native_signal_fields(selected_fields: Sequence[str]) -> tuple[str, ...]:
    """Return the exact 513-field surface or fail on any soft compatibility."""

    selected = tuple(str(name).strip() for name in selected_fields)
    failures: list[str] = []
    if len(selected) != MODEL_NATIVE_SELECTED_FEATURE_COUNT:
        failures.append(
            "selected_feature_count="
            f"{len(selected)} expected={MODEL_NATIVE_SELECTED_FEATURE_COUNT}"
        )
    blank = [index for index, name in enumerate(selected) if not name]
    if blank:
        failures.append(f"blank_selected_field_indices={blank[:10]}")
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        failures.append(f"duplicate_selected_fields={duplicates[:20]}")
    forbidden = sorted(set(selected) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    if forbidden:
        failures.append(f"forbidden_legacy_bridge_fields={forbidden}")
    base_overlap = sorted(set(selected) & set(MODEL_NATIVE_BASE_FIELDS))
    if base_overlap:
        failures.append(f"selected_fields_duplicate_base_fields={base_overlap[:20]}")
    selected_set = set(selected)
    missing_mandatory = [
        name for name in MODEL_NATIVE_MANDATORY_SELECTED_FIELDS if name not in selected_set
    ]
    if missing_mandatory:
        failures.append(
            "missing_mandatory_full_stack_fields="
            f"{missing_mandatory[:20]} total={len(missing_mandatory)}"
        )
    else:
        # The first 305 positions must exactly equal the immutable causal-layer
        # registry order; membership alone is not the documented contract.
        mandatory_prefix = tuple(
            selected[:MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT]
        )
        if mandatory_prefix != tuple(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS):
            wrong_positions = [
                index
                for index, (got, want) in enumerate(
                    zip(mandatory_prefix, MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
                )
                if got != want
            ]
            failures.append(
                "mandatory_registry_prefix_order_violation="
                f"positions{wrong_positions[:10]} total={len(wrong_positions)}"
            )
    fields = MODEL_NATIVE_BASE_FIELDS + selected
    if len(fields) != MODEL_NATIVE_SIGNAL_DIM:
        failures.append(f"signal_dim={len(fields)} expected={MODEL_NATIVE_SIGNAL_DIM}")
    if failures:
        raise RuntimeError("MODEL_NATIVE_SIGNAL_FIELDS_INVALID: " + " | ".join(failures))
    return fields


def model_native_signal_contract_metadata(selected_fields: Sequence[str]) -> dict[str, Any]:
    selected = tuple(str(name).strip() for name in selected_fields)
    fields = ordered_model_native_signal_fields(selected)
    return {
        "schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "static_contract_sha256": MODEL_NATIVE_STATIC_CONTRACT_SHA256,
        "ordered_fields_sha256": _sha256_json(fields),
        "base_fields": list(MODEL_NATIVE_BASE_FIELDS),
        "selected_fields": list(selected),
        "fields": list(fields),
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "mandatory_full_stack": model_native_mandatory_full_stack_metadata(),
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "forbidden_legacy_bridge_fields": list(FORBIDDEN_LEGACY_BRIDGE_FIELDS),
        "bridge_dim": 0,
        "bridge_source": None,
        "anchor_source": None,
    }


def model_native_signal_contract_failures(contract: Mapping[str, Any]) -> list[str]:
    """Validate a serialized dataset/bundle contract without filling defaults."""

    failures: list[str] = []
    if not isinstance(contract, Mapping) or not contract:
        return ["model_native_signal_contract missing"]

    exact_scalars = {
        "schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "static_contract_sha256": MODEL_NATIVE_STATIC_CONTRACT_SHA256,
        "base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "mandatory_selected_feature_count": MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
        "ranked_remainder_feature_count": MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "bridge_dim": 0,
    }
    for key, expected in exact_scalars.items():
        if contract.get(key) != expected:
            failures.append(f"{key}={contract.get(key)!r} expected={expected!r}")

    base_fields = tuple(str(value) for value in (contract.get("base_fields") or ()))
    selected_fields = tuple(str(value) for value in (contract.get("selected_fields") or ()))
    fields = tuple(str(value) for value in (contract.get("fields") or ()))
    forbidden_declared = tuple(
        str(value) for value in (contract.get("forbidden_legacy_bridge_fields") or ())
    )
    if base_fields != MODEL_NATIVE_BASE_FIELDS:
        failures.append("base_fields order mismatch")
    if forbidden_declared != FORBIDDEN_LEGACY_BRIDGE_FIELDS:
        failures.append("forbidden_legacy_bridge_fields order mismatch")
    mandatory_declared = contract.get("mandatory_full_stack")
    mandatory_expected = model_native_mandatory_full_stack_metadata()
    if not isinstance(mandatory_declared, Mapping):
        failures.append("mandatory_full_stack missing")
    elif dict(mandatory_declared) != mandatory_expected:
        failures.append("mandatory_full_stack metadata mismatch")
    try:
        expected_fields = ordered_model_native_signal_fields(selected_fields)
    except RuntimeError as exc:
        failures.append(str(exc))
        expected_fields = ()
    if fields != expected_fields:
        failures.append("fields order mismatch")
    forbidden_present = sorted(set(fields) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
    if forbidden_present:
        failures.append(f"fields contain forbidden legacy bridge inputs: {forbidden_present}")
    expected_hash = _sha256_json(fields)
    if str(contract.get("ordered_fields_sha256") or "") != expected_hash:
        failures.append(
            "ordered_fields_sha256 mismatch: "
            f"declared={contract.get('ordered_fields_sha256')!r} actual={expected_hash}"
        )
    if contract.get("bridge_source") is not None:
        failures.append(f"bridge_source must be null, got {contract.get('bridge_source')!r}")
    if contract.get("anchor_source") is not None:
        failures.append(f"anchor_source must be null, got {contract.get('anchor_source')!r}")
    return failures


def require_model_native_signal_contract(contract: Mapping[str, Any], *, context: str) -> None:
    failures = model_native_signal_contract_failures(contract)
    if failures:
        raise RuntimeError(f"[{context}_MODEL_NATIVE_SIGNAL_CONTRACT_INVALID] " + " | ".join(failures))


def require_model_native_manifest(manifest: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    """Validate the feature-selection manifest and return its exact contract."""

    mode = str(manifest.get("manifest_variant") or "").strip()
    if mode == RETIRED_NEUTRAL_BRIDGE_CONTRACT_MODE:
        raise RuntimeError(
            f"[{context}_RETIRED_SMART520_CONTRACT] {mode} contains the retired "
            "seven-field XGB/neutral bridge; materialize a fresh "
            f"{MODEL_NATIVE_CONTRACT_MODE} manifest"
        )
    if mode != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            f"[{context}_CONTRACT_MODE_INVALID] manifest_variant={mode!r} "
            f"expected={MODEL_NATIVE_CONTRACT_MODE!r}"
        )
    if int(manifest.get("base_signal_feature_count") or -1) != MODEL_NATIVE_BASE_SIGNAL_DIM:
        raise RuntimeError(
            f"[{context}_BASE_SIGNAL_DIM_INVALID] got={manifest.get('base_signal_feature_count')!r} "
            f"expected={MODEL_NATIVE_BASE_SIGNAL_DIM}"
        )
    if int(manifest.get("expected_seq_snap_width") or -1) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            f"[{context}_SIGNAL_DIM_INVALID] got={manifest.get('expected_seq_snap_width')!r} "
            f"expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    selected = manifest.get("selected_features")
    if not isinstance(selected, list):
        raise RuntimeError(f"[{context}_SELECTED_FIELDS_MISSING]")
    declared_mandatory = manifest.get("mandatory_full_stack")
    expected_mandatory = model_native_mandatory_full_stack_metadata()
    if not isinstance(declared_mandatory, Mapping):
        raise RuntimeError(f"[{context}_MANDATORY_FULL_STACK_METADATA_MISSING]")
    if dict(declared_mandatory) != expected_mandatory:
        raise RuntimeError(f"[{context}_MANDATORY_FULL_STACK_METADATA_STALE]")
    contract = model_native_signal_contract_metadata(selected)
    declared = manifest.get("model_native_signal_contract")
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"[{context}_MODEL_NATIVE_SIGNAL_CONTRACT_MISSING]")
    require_model_native_signal_contract(declared, context=context)
    if dict(declared) != contract:
        raise RuntimeError(f"[{context}_MODEL_NATIVE_SIGNAL_CONTRACT_NOT_EXACT]")
    return contract
