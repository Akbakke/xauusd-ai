"""VAL-only selected-checkpoint input-response evidence for Entry Q.

This is deliberately trainer-owned evidence.  It proves that the chosen
checkpoint responds to every retained physical Entry input and every
local/MTF family route without opening the sealed TEST split or invoking a
serve/runtime adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM
from gx1.contracts.model_native_serve_gate_v1 import (
    individual_input_influence_layout,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)


SCHEMA_VERSION = "gx1_entry_val_input_influence_v1"
SPLIT = "val"
SAMPLE_COUNT = 8
NUMERIC_GRADIENT_EPSILON = 1e-12
COUNTERFACTUAL_DELTA_EPSILON = 1e-7
FAMILY_ABLATION_EPSILON = 1e-7
SAMPLING_CONTRACT = (
    "eight_evenly_spaced_direct_causal_val_dataset_positions_v1"
)
COMPARISON_SURFACE = "pairwise_class_centered_entry_action_q_bps"


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_above(value: Any, epsilon: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > epsilon
    )


def _require_metric_row(
    value: Any,
    *,
    metric_name: str,
    epsilon: float,
    sample_count: int | None = None,
) -> bool:
    expected = {"decision", "failures", metric_name}
    if sample_count is not None:
        expected |= {"changed_rows", "total_rows", "counterfactual"}
    if not isinstance(value, Mapping) or set(value) != expected:
        return False
    if value.get("decision") != "PASS" or value.get("failures") != []:
        return False
    if not _finite_above(value.get(metric_name), epsilon):
        return False
    if sample_count is not None:
        changed = value.get("changed_rows")
        if (
            isinstance(changed, bool)
            or not isinstance(changed, int)
            or not 1 <= changed <= sample_count
            or value.get("total_rows") != sample_count
            or value.get("counterfactual")
            != "valid_owner_manifold_counterfactual"
        ):
            return False
    return True


def require_entry_val_input_influence(
    value: Mapping[str, Any] | Any,
    *,
    ordered_signal_names: list[str] | tuple[str, ...],
    val_data_sha256: str,
    multi_tf_cache_identity_sha256: str,
    selected_model_state_dict_sha256: str,
    local_context_routing_sha256: str,
    multi_tf_routing_sha256: str,
    context: str,
) -> dict[str, Any]:
    """Require all physical Entry inputs and all 8×(local+MTF) routes alive."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_MISSING]")
    signal_names = [str(item) for item in ordered_signal_names]
    if len(signal_names) != MODEL_NATIVE_SIGNAL_DIM or len(set(signal_names)) != len(
        signal_names
    ):
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_SIGNAL_ORDER_INVALID]")
    ownership = individual_input_influence_layout(
        signal_names,
        mtf_timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
    )
    expected_keys = {
        "schema_version",
        "decision",
        "failures",
        "required_for_candidate",
        "split",
        "sample_count",
        "sampling_contract",
        "comparison_surface",
        "numeric_gradient_epsilon",
        "counterfactual_delta_epsilon",
        "family_ablation_epsilon",
        "sample_entry_row_indices",
        "sample_decision_times_ns",
        "val_data_sha256",
        "multi_tf_cache_identity_sha256",
        "selected_model_state_dict_sha256",
        "ordered_signal_names",
        "signal_names_sha256",
        "input_ownership",
        "input_ownership_sha256",
        "numeric_input_count",
        "continuous_manifold_input_count",
        "categorical_input_count",
        "individual",
        "family_ablation",
    }
    if set(value) != expected_keys:
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_KEYS_INVALID]")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("decision") != "PASS"
        or value.get("failures") != []
        or value.get("required_for_candidate") is not True
        or value.get("split") != SPLIT
        or value.get("sample_count") != SAMPLE_COUNT
        or value.get("sampling_contract") != SAMPLING_CONTRACT
        or value.get("comparison_surface") != COMPARISON_SURFACE
        or value.get("numeric_gradient_epsilon") != NUMERIC_GRADIENT_EPSILON
        or value.get("counterfactual_delta_epsilon")
        != COUNTERFACTUAL_DELTA_EPSILON
        or value.get("family_ablation_epsilon") != FAMILY_ABLATION_EPSILON
        or value.get("ordered_signal_names") != signal_names
        or value.get("signal_names_sha256") != canonical_json_sha256(signal_names)
        or value.get("input_ownership") != ownership
        or value.get("input_ownership_sha256") != canonical_json_sha256(ownership)
    ):
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_METADATA_INVALID]")
    for field, expected in (
        ("val_data_sha256", val_data_sha256),
        ("multi_tf_cache_identity_sha256", multi_tf_cache_identity_sha256),
        ("selected_model_state_dict_sha256", selected_model_state_dict_sha256),
    ):
        if not _is_sha256(expected) or value.get(field) != expected:
            raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_BINDING_INVALID] {field}")
    for field in ("sample_entry_row_indices", "sample_decision_times_ns"):
        rows = value.get(field)
        if (
            not isinstance(rows, list)
            or len(rows) != SAMPLE_COUNT
            or any(isinstance(item, bool) or not isinstance(item, int) for item in rows)
        ):
            raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_SAMPLE_INVALID] {field}")
    if (
        value.get("numeric_input_count")
        != sum(len(row["tokens"]) for row in ownership["numeric"].values())
        or value.get("continuous_manifold_input_count")
        != len(ownership["continuous_manifold"])
        or value.get("categorical_input_count") != len(ownership["categorical"])
    ):
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_COUNT_INVALID]")

    individual = value.get("individual")
    if not isinstance(individual, Mapping) or set(individual) != {
        "numeric",
        "continuous_manifold",
        "categorical",
    }:
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_INDIVIDUAL_INVALID]")
    numeric = individual["numeric"]
    if not isinstance(numeric, Mapping) or set(numeric) != set(ownership["numeric"]):
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_NUMERIC_SET_INVALID]")
    for surface, owner in ownership["numeric"].items():
        row = numeric[surface]
        tokens = owner["tokens"]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"tokens", "source_indices", "metrics"}
            or row.get("tokens") != tokens
            or row.get("source_indices") != owner["source_indices"]
            or not isinstance(row.get("metrics"), Mapping)
            or set(row["metrics"]) != set(tokens)
            or not all(
                _require_metric_row(
                    row["metrics"][token],
                    metric_name="max_abs_entry_action_q_class_margin_gradient",
                    epsilon=NUMERIC_GRADIENT_EPSILON,
                )
                for token in tokens
            )
        ):
            raise RuntimeError(
                f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_NUMERIC_INVALID] {surface}"
            )
    for key in ("continuous_manifold", "categorical"):
        expected = ownership[key]
        observed = individual[key]
        expected_tokens = [str(row["token"]) for row in expected]
        if not isinstance(observed, Mapping) or set(observed) != set(expected_tokens):
            raise RuntimeError(
                f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_{key.upper()}_SET_INVALID]"
            )
        for token in expected_tokens:
            if not _require_metric_row(
                observed[token],
                metric_name="max_abs_entry_action_q_delta_bps",
                epsilon=COUNTERFACTUAL_DELTA_EPSILON,
                sample_count=SAMPLE_COUNT,
            ):
                raise RuntimeError(
                    f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_{key.upper()}_INVALID] {token}"
                )

    family = value.get("family_ablation")
    expected_mtf = {
        f"{timeframe.lower()}:{specialist}"
        for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
        for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS
    }
    if not isinstance(family, Mapping) or set(family) != {
        "epsilon",
        "sample_count",
        "local_context_routing_sha256",
        "multi_tf_routing_sha256",
        "local_context",
        "multi_tf",
    }:
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_FAMILY_KEYS_INVALID]")
    if (
        family.get("epsilon") != FAMILY_ABLATION_EPSILON
        or family.get("sample_count") != SAMPLE_COUNT
        or family.get("local_context_routing_sha256") != local_context_routing_sha256
        or family.get("multi_tf_routing_sha256") != multi_tf_routing_sha256
        or not _is_sha256(local_context_routing_sha256)
        or not _is_sha256(multi_tf_routing_sha256)
    ):
        raise RuntimeError(f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_FAMILY_BINDING_INVALID]")
    for label, expected_tokens in (
        ("local_context", set(MODEL_NATIVE_TRAINING_SPECIALISTS)),
        ("multi_tf", expected_mtf),
    ):
        observed = family.get(label)
        if not isinstance(observed, Mapping) or set(observed) != expected_tokens:
            raise RuntimeError(
                f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_{label.upper()}_SET_INVALID]"
            )
        for token, row in observed.items():
            if (
                not isinstance(row, Mapping)
                or set(row) != {
                    "decision",
                    "failures",
                    "source_binding_sha256",
                    "max_abs_entry_action_q_delta_bps",
                    "changed_rows",
                    "total_rows",
                }
                or row.get("decision") != "PASS"
                or row.get("failures") != []
                or not _is_sha256(row.get("source_binding_sha256"))
                or not _finite_above(
                    row.get("max_abs_entry_action_q_delta_bps"),
                    FAMILY_ABLATION_EPSILON,
                )
                or isinstance(row.get("changed_rows"), bool)
                or not isinstance(row.get("changed_rows"), int)
                or not 1 <= int(row["changed_rows"]) <= SAMPLE_COUNT
                or row.get("total_rows") != SAMPLE_COUNT
            ):
                raise RuntimeError(
                    f"[{context}_ENTRY_VAL_INPUT_INFLUENCE_{label.upper()}_INVALID] {token}"
                )
    return dict(value)


__all__ = [
    "COMPARISON_SURFACE",
    "COUNTERFACTUAL_DELTA_EPSILON",
    "FAMILY_ABLATION_EPSILON",
    "NUMERIC_GRADIENT_EPSILON",
    "SAMPLE_COUNT",
    "SAMPLING_CONTRACT",
    "SCHEMA_VERSION",
    "SPLIT",
    "canonical_json_sha256",
    "require_entry_val_input_influence",
]
