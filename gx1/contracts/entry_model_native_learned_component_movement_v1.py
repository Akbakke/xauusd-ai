"""Exact post-training movement proof for the fitted-Q Entry authority."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
)


SCHEMA_VERSION = "gx1_entry_fitted_q_parameter_movement_v1"
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
    return {
        "schema_version": SCHEMA_VERSION,
        "reference": REFERENCE,
        "selected_checkpoint_epoch": epoch,
        "parameter_deltas": normalized_deltas,
        "component_changed": {component: True for component in COMPONENT_PARAMETERS},
        "output_rows_distinct": True,
        "decision": "PASS",
    }
