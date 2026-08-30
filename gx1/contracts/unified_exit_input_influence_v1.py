"""Field-level learned-input reachability contract for unified Exit."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    individual_input_influence_layout,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_PATH_FEATURE_ORDER,
)


SCHEMA_VERSION = "gx1_unified_exit_input_influence_v4"
SPLIT = "val"
SAMPLE_COUNT = 8
SIDE_ROWS = 4
NUMERIC_GRADIENT_EPSILON = 1e-12
CATEGORICAL_DELTA_EPSILON = 1e-7
SAMPLING_CONTRACT = (
    "deterministic_first_valid_label_independent_probe_per_side_at_or_after_"
    "four_even_val_entry_positions_v1"
)
COMPARISON_SURFACE = "all_causal_exit_now_minus_hold_q_bps"


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


def unified_exit_input_influence_layout(
    ordered_signal_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    layout = individual_input_influence_layout(
        ordered_signal_names,
        mtf_timeframes=EXIT_MTF_CONTEXT_TIMEFRAMES,
    )
    numeric = dict(layout["numeric"])
    # Episode-native Exit owns one causal local timeline.  A separately fed
    # snapshot would duplicate its current row and is therefore forbidden.
    numeric.pop("snap_signal", None)
    numeric["entry_decision_representation"] = {
        "tokens": [
            f"entry_decision_representation[{index:03d}]"
            for index in range(UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM)
        ],
        "source_indices": list(range(UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM)),
    }
    numeric["exit_path"] = {
        "tokens": list(UNIFIED_EXIT_PATH_FEATURE_ORDER),
        "source_indices": list(range(len(UNIFIED_EXIT_PATH_FEATURE_ORDER))),
    }
    categorical = [dict(row) for row in layout["categorical"]]
    return {
        "mtf_timeframes": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
        "numeric": numeric,
        "categorical": categorical,
        "structural": [
            {
                "token": "exit_side_axis",
                "owner": "learned_long_short_side_embedding",
                "manifold": "same_market_token_and_path_compare_both_side_axes",
            }
        ],
    }


def _finite_above(value: Any, epsilon: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > float(epsilon)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_unified_exit_input_influence(
    value: Mapping[str, Any],
    *,
    ordered_signal_names: list[str] | tuple[str, ...],
    selected_online_model_state_sha256: str,
    val_data_sha256: str,
    multi_tf_cache_identity_sha256: str,
    unified_exit_lifecycle_root_manifest_sha256: str,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_EXIT_INPUT_INFLUENCE_MISSING]")
    layout = unified_exit_input_influence_layout(ordered_signal_names)
    failures: list[str] = []
    exact_top_keys = {
        "schema_version",
        "decision",
        "failures",
        "split",
        "sample_count",
        "side_rows",
        "sampling_contract",
        "comparison_surface",
        "numeric_gradient_epsilon",
        "categorical_delta_epsilon",
        "sample_entry_row_indices",
        "sample_decision_times_ns",
        "selected_online_model_state_sha256",
        "val_data_sha256",
        "multi_tf_cache_identity_sha256",
        "unified_exit_lifecycle_root_manifest_sha256",
        "ordered_signal_names",
        "signal_names_sha256",
        "input_ownership",
        "input_ownership_sha256",
        "numeric_input_count",
        "categorical_input_count",
        "numeric",
        "categorical",
        "structural",
    }
    if set(value) != exact_top_keys:
        failures.append("top-level keys")
    if value.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version")
    if value.get("decision") != "PASS" or value.get("failures") != []:
        failures.append("decision")
    for field, expected in (
        ("split", SPLIT),
        ("sample_count", SAMPLE_COUNT),
        ("side_rows", {"long": SIDE_ROWS, "short": SIDE_ROWS}),
        ("sampling_contract", SAMPLING_CONTRACT),
        ("comparison_surface", COMPARISON_SURFACE),
        ("numeric_gradient_epsilon", NUMERIC_GRADIENT_EPSILON),
        ("categorical_delta_epsilon", CATEGORICAL_DELTA_EPSILON),
    ):
        if value.get(field) != expected:
            failures.append(field)
    for field in ("sample_entry_row_indices", "sample_decision_times_ns"):
        rows = value.get(field)
        if (
            not isinstance(rows, list)
            or len(rows) != SAMPLE_COUNT
            or any(isinstance(item, bool) or not isinstance(item, int) for item in rows)
        ):
            failures.append(field)
    signal_names = [str(name) for name in ordered_signal_names]
    if value.get("ordered_signal_names") != signal_names:
        failures.append("ordered_signal_names")
    if value.get("signal_names_sha256") != canonical_json_sha256(signal_names):
        failures.append("signal_names_sha256")
    if value.get("input_ownership") != layout:
        failures.append("input_ownership")
    if value.get("input_ownership_sha256") != canonical_json_sha256(layout):
        failures.append("input_ownership_sha256")
    for field, expected in (
        ("selected_online_model_state_sha256", selected_online_model_state_sha256),
        ("val_data_sha256", val_data_sha256),
        ("multi_tf_cache_identity_sha256", multi_tf_cache_identity_sha256),
        (
            "unified_exit_lifecycle_root_manifest_sha256",
            unified_exit_lifecycle_root_manifest_sha256,
        ),
    ):
        if not _is_sha256(expected) or value.get(field) != expected:
            failures.append(field)

    expected_numeric = layout["numeric"]
    expected_categorical = layout["categorical"]
    if value.get("numeric_input_count") != sum(
        len(row["tokens"]) for row in expected_numeric.values()
    ):
        failures.append("numeric_input_count")
    if value.get("categorical_input_count") != len(expected_categorical):
        failures.append("categorical_input_count")
    numeric = value.get("numeric")
    if not isinstance(numeric, Mapping) or set(numeric) != set(expected_numeric):
        failures.append("numeric surface set")
    else:
        for surface, owner in expected_numeric.items():
            row = numeric.get(surface)
            tokens = owner["tokens"]
            if (
                not isinstance(row, Mapping)
                or set(row) != {"tokens", "source_indices", "metrics"}
                or row.get("tokens") != tokens
                or row.get("source_indices") != owner["source_indices"]
                or not isinstance(row.get("metrics"), Mapping)
                or set(row["metrics"]) != set(tokens)
            ):
                failures.append(f"numeric.{surface}")
                continue
            for token in tokens:
                metric = row["metrics"][token]
                if (
                    not isinstance(metric, Mapping)
                    or set(metric)
                    != {
                        "decision",
                        "failures",
                        "max_abs_exit_margin_gradient",
                    }
                    or metric.get("decision") != "PASS"
                    or metric.get("failures") != []
                    or not _finite_above(
                        metric.get("max_abs_exit_margin_gradient"),
                        NUMERIC_GRADIENT_EPSILON,
                    )
                ):
                    failures.append(f"numeric.{surface}.{token}")

    categorical = value.get("categorical")
    categorical_tokens = [str(row["token"]) for row in expected_categorical]
    if (
        not isinstance(categorical, Mapping)
        or set(categorical) != set(categorical_tokens)
    ):
        failures.append("categorical set")
    else:
        for token in categorical_tokens:
            metric = categorical[token]
            if (
                not isinstance(metric, Mapping)
                or set(metric)
                != {
                    "decision",
                    "failures",
                    "counterfactual",
                    "max_abs_exit_margin_delta",
                    "changed_rows",
                    "total_rows",
                }
                or metric.get("decision") != "PASS"
                or metric.get("failures") != []
                or metric.get("counterfactual")
                != "next_valid_category_on_exact_owner_manifold"
                or not _finite_above(
                    metric.get("max_abs_exit_margin_delta"),
                    CATEGORICAL_DELTA_EPSILON,
                )
                or isinstance(metric.get("changed_rows"), bool)
                or not isinstance(metric.get("changed_rows"), int)
                or not 1 <= int(metric["changed_rows"]) <= SAMPLE_COUNT
                or metric.get("total_rows") != SAMPLE_COUNT
            ):
                failures.append(f"categorical.{token}")
    structural = value.get("structural")
    metric = structural.get("exit_side_axis") if isinstance(structural, Mapping) else None
    if (
        not isinstance(metric, Mapping)
        or set(metric)
        != {
            "decision",
            "failures",
            "counterfactual",
            "max_abs_exit_margin_delta",
            "changed_rows",
            "total_rows",
        }
        or metric.get("decision") != "PASS"
        or metric.get("failures") != []
        or metric.get("counterfactual")
        != "same_market_token_and_path_compare_both_side_axes"
        or not _finite_above(
            metric.get("max_abs_exit_margin_delta"),
            CATEGORICAL_DELTA_EPSILON,
        )
        or isinstance(metric.get("changed_rows"), bool)
        or not isinstance(metric.get("changed_rows"), int)
        or not 1 <= int(metric["changed_rows"]) <= SAMPLE_COUNT
        or metric.get("total_rows") != SAMPLE_COUNT
    ):
        failures.append("structural.exit_side_axis")
    if failures:
        raise RuntimeError(
            f"[{context}_EXIT_INPUT_INFLUENCE_INVALID] " + "; ".join(failures)
        )


__all__ = [
    "CATEGORICAL_DELTA_EPSILON",
    "COMPARISON_SURFACE",
    "NUMERIC_GRADIENT_EPSILON",
    "SAMPLE_COUNT",
    "SAMPLING_CONTRACT",
    "SCHEMA_VERSION",
    "SIDE_ROWS",
    "SPLIT",
    "canonical_json_sha256",
    "require_unified_exit_input_influence",
    "unified_exit_input_influence_layout",
]
