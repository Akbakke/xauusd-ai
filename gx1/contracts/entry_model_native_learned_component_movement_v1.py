"""Exact post-training movement proof for the fitted-Q Entry authority."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)


SCHEMA_VERSION = "gx1_entry_fitted_q_parameter_movement_v2"
REFERENCE = "direct_joint_representation_raw_bps_q_head"
_HIDDEN_DIM = UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM
_JOINT_DIM = 4 * _HIDDEN_DIM
PARAMETER_SHAPES = {
    "entry_q_joint_norm.weight": [_JOINT_DIM],
    "entry_q_joint_norm.bias": [_JOINT_DIM],
    "entry_q_joint_in.weight": [_HIDDEN_DIM, _JOINT_DIM],
    "entry_q_joint_in.bias": [_HIDDEN_DIM],
    "head_entry_action_q.weight": [3, _HIDDEN_DIM],
    "head_entry_action_q.bias": [3],
}
COMPONENT_PARAMETERS = {
    "joint_projection": (
        "entry_q_joint_norm.weight",
        "entry_q_joint_norm.bias",
        "entry_q_joint_in.weight",
        "entry_q_joint_in.bias",
    ),
    "raw_q_head": (
        "head_entry_action_q.weight",
        "head_entry_action_q.bias",
    ),
}
ENCODER_COMPONENT_PREFIXES = {
    **{
        f"local_specialist_encoder:{specialist}": (
            f"specialist_encoder.{specialist}."
        )
        for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS
    },
    **{
        f"mtf_family_encoder:{specialist}": (
            f"mtf_family_encoder.{specialist}."
        )
        for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS
    },
}


def require_learned_component_movement_metadata(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_MISSING]")
    expected_keys = {
        "schema_version",
        "reference",
        "selected_checkpoint_epoch",
        "parameter_deltas",
        "component_changed",
        "encoder_component_movement",
        "output_rows_distinct",
        "decision",
    }
    if set(value) != expected_keys:
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_KEYS_INVALID]")
    epoch = value.get("selected_checkpoint_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_EPOCH_INVALID]")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("reference") != REFERENCE:
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_SCHEMA_INVALID]")
    if value.get("decision") != "PASS" or value.get("output_rows_distinct") is not True:
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_DECISION_INVALID]")

    deltas = value.get("parameter_deltas")
    if not isinstance(deltas, Mapping) or set(deltas) != set(PARAMETER_SHAPES):
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_PARAMETERS_INVALID]")
    normalized_deltas: dict[str, dict[str, Any]] = {}
    for key, shape in PARAMETER_SHAPES.items():
        row = deltas[key]
        if not isinstance(row, Mapping) or set(row) != {
            "shape",
            "max_abs_delta",
            "l2_delta",
            "changed",
        }:
            raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_ROW_INVALID] {key}")
        if row.get("shape") != shape:
            raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_SHAPE_INVALID] {key}")
        try:
            max_abs_delta = float(row["max_abs_delta"])
            l2_delta = float(row["l2_delta"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"[{context}_LEARNED_COMPONENT_MOVEMENT_DELTA_INVALID] {key}"
            ) from exc
        if (
            not math.isfinite(max_abs_delta)
            or not math.isfinite(l2_delta)
            or max_abs_delta < 0.0
            or l2_delta < 0.0
        ):
            raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_DELTA_INVALID] {key}")
        expected_changed = max_abs_delta > 0.0 and l2_delta > 0.0
        if row.get("changed") is not expected_changed:
            raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_CHANGED_INVALID] {key}")
        normalized_deltas[key] = {
            "shape": list(shape),
            "max_abs_delta": max_abs_delta,
            "l2_delta": l2_delta,
            "changed": expected_changed,
        }

    components = value.get("component_changed")
    if not isinstance(components, Mapping) or set(components) != set(COMPONENT_PARAMETERS):
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_COMPONENTS_INVALID]")
    for component, keys in COMPONENT_PARAMETERS.items():
        expected_changed = any(normalized_deltas[key]["changed"] for key in keys)
        if components.get(component) is not expected_changed or not expected_changed:
            raise RuntimeError(
                f"[{context}_LEARNED_COMPONENT_MOVEMENT_COMPONENT_INACTIVE] {component}"
            )
    encoder_movement = value.get("encoder_component_movement")
    if (
        not isinstance(encoder_movement, Mapping)
        or set(encoder_movement) != set(ENCODER_COMPONENT_PREFIXES)
    ):
        raise RuntimeError(f"[{context}_LEARNED_COMPONENT_MOVEMENT_ENCODERS_INVALID]")
    normalized_encoders: dict[str, dict[str, Any]] = {}
    for component in ENCODER_COMPONENT_PREFIXES:
        row = encoder_movement[component]
        if not isinstance(row, Mapping) or set(row) != {
            "parameter_count",
            "changed_parameter_count",
            "max_abs_delta",
            "l2_delta",
            "changed",
        }:
            raise RuntimeError(
                f"[{context}_LEARNED_COMPONENT_MOVEMENT_ENCODER_ROW_INVALID] {component}"
            )
        parameter_count = row.get("parameter_count")
        changed_parameter_count = row.get("changed_parameter_count")
        if (
            isinstance(parameter_count, bool)
            or not isinstance(parameter_count, int)
            or parameter_count <= 0
            or isinstance(changed_parameter_count, bool)
            or not isinstance(changed_parameter_count, int)
            or not 1 <= changed_parameter_count <= parameter_count
        ):
            raise RuntimeError(
                f"[{context}_LEARNED_COMPONENT_MOVEMENT_ENCODER_COUNT_INVALID] {component}"
            )
        try:
            max_abs_delta = float(row["max_abs_delta"])
            l2_delta = float(row["l2_delta"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"[{context}_LEARNED_COMPONENT_MOVEMENT_ENCODER_DELTA_INVALID] {component}"
            ) from exc
        if (
            not math.isfinite(max_abs_delta)
            or not math.isfinite(l2_delta)
            or max_abs_delta <= 0.0
            or l2_delta <= 0.0
            or row.get("changed") is not True
        ):
            raise RuntimeError(
                f"[{context}_LEARNED_COMPONENT_MOVEMENT_ENCODER_INACTIVE] {component}"
            )
        normalized_encoders[component] = {
            "parameter_count": parameter_count,
            "changed_parameter_count": changed_parameter_count,
            "max_abs_delta": max_abs_delta,
            "l2_delta": l2_delta,
            "changed": True,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "reference": REFERENCE,
        "selected_checkpoint_epoch": epoch,
        "parameter_deltas": normalized_deltas,
        "component_changed": {component: True for component in COMPONENT_PARAMETERS},
        "encoder_component_movement": normalized_encoders,
        "output_rows_distinct": True,
        "decision": "PASS",
    }
