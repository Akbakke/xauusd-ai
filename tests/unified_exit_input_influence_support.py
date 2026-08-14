from __future__ import annotations

from gx1.contracts.unified_exit_input_influence_v1 import (
    CATEGORICAL_DELTA_EPSILON,
    COMPARISON_SURFACE,
    NUMERIC_GRADIENT_EPSILON,
    SAMPLE_COUNT,
    SAMPLING_CONTRACT,
    SCHEMA_VERSION,
    SIDE_ROWS,
    SPLIT,
    canonical_json_sha256,
    unified_exit_input_influence_layout,
)


def passing_unified_exit_input_influence(
    signal_names: list[str],
) -> dict[str, object]:
    ownership = unified_exit_input_influence_layout(signal_names)
    numeric = {}
    for surface, owner in ownership["numeric"].items():
        tokens = list(owner["tokens"])
        numeric[surface] = {
            "tokens": tokens,
            "source_indices": list(owner["source_indices"]),
            "metrics": {
                token: {
                    "decision": "PASS",
                    "failures": [],
                    "max_abs_exit_margin_gradient": 0.1,
                }
                for token in tokens
            },
        }
    categorical = {
        str(owner["token"]): {
            "decision": "PASS",
            "failures": [],
            "counterfactual": "next_valid_category_on_exact_owner_manifold",
            "max_abs_exit_margin_delta": 0.1,
            "changed_rows": SAMPLE_COUNT,
            "total_rows": SAMPLE_COUNT,
        }
        for owner in ownership["categorical"]
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "failures": [],
        "split": SPLIT,
        "sample_count": SAMPLE_COUNT,
        "side_rows": {"long": SIDE_ROWS, "short": SIDE_ROWS},
        "sampling_contract": SAMPLING_CONTRACT,
        "comparison_surface": COMPARISON_SURFACE,
        "numeric_gradient_epsilon": NUMERIC_GRADIENT_EPSILON,
        "categorical_delta_epsilon": CATEGORICAL_DELTA_EPSILON,
        "sample_entry_row_indices": list(range(SAMPLE_COUNT)),
        "sample_decision_times_ns": list(range(1, SAMPLE_COUNT + 1)),
        "ordered_signal_names": list(signal_names),
        "signal_names_sha256": canonical_json_sha256(signal_names),
        "input_ownership": ownership,
        "input_ownership_sha256": canonical_json_sha256(ownership),
        "numeric_input_count": sum(
            len(owner["tokens"]) for owner in ownership["numeric"].values()
        ),
        "categorical_input_count": len(ownership["categorical"]),
        "numeric": numeric,
        "categorical": categorical,
        "structural": {
            "exit_path_length": {
                "decision": "PASS",
                "failures": [],
                "counterfactual": "truncate_one_bar_and_zero_removed_suffix",
                "max_abs_exit_margin_delta": 0.1,
                "changed_rows": SAMPLE_COUNT,
                "total_rows": SAMPLE_COUNT,
            }
        },
    }


__all__ = ["passing_unified_exit_input_influence"]
