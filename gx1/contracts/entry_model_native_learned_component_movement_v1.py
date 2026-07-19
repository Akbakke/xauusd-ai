"""Exact post-training movement proof for the sole direction fusion."""

from __future__ import annotations

import math
from typing import Any, Mapping

from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    HIDDEN_DIM,
    INPUT_DIM,
    OUTPUT_DIM,
)


SCHEMA_VERSION = "entry_model_native_learned_component_movement_v1"
REFERENCE = "post_initialization_pre_optimizer_step"
PARAMETER_SHAPES = {
    "evidence_fusion_norm.weight": [INPUT_DIM],
    "evidence_fusion_norm.bias": [INPUT_DIM],
    "evidence_fusion_in.weight": [HIDDEN_DIM, INPUT_DIM],
    "evidence_fusion_in.bias": [HIDDEN_DIM],
    "evidence_fusion_out.weight": [OUTPUT_DIM, HIDDEN_DIM],
    "evidence_fusion_out.bias": [OUTPUT_DIM],
}
COMPONENT_PARAMETERS = {
    "evidence_fusion_norm": (
        "evidence_fusion_norm.weight",
        "evidence_fusion_norm.bias",
    ),
    "evidence_fusion_in": (
        "evidence_fusion_in.weight",
        "evidence_fusion_in.bias",
    ),
    "evidence_fusion_out": (
        "evidence_fusion_out.weight",
        "evidence_fusion_out.bias",
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
