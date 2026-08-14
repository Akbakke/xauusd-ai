from __future__ import annotations

import hashlib
import json

from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS,
    SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS,
    SERVE_PARITY_FORWARD_HEADS,
    SERVE_PARITY_FORWARD_TOL,
    SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS,
    SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS,
    SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE,
    SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE,
    SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE,
    SERVE_PARITY_HEAD_VARIATION_EPSILON,
    SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON,
    SERVE_PARITY_INDIVIDUAL_INPUT_COMPARISON_SURFACE,
    SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON,
    SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
    SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS,
    SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY,
    SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT,
    SERVE_PARITY_SPECIALIST_GATE_MIN_STD,
    SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT,
    SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR,
    SERVE_PARITY_SPECIALIST_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON,
    SERVE_PARITY_SPECIALIST_INFLUENCE_METHODS,
    SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT,
    SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
    SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
    SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES,
    SERVE_PARITY_UPSTREAM_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON,
    SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS,
    SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLING_CONTRACT,
    SERVE_SOURCE_IDENTITY_CONTRACT,
    SERVE_SOURCE_IDENTITY_EXCLUDED_TRACKED_PATHS,
    SERVE_SOURCE_IDENTITY_SCHEMA_VERSION,
    UTC_TIME_COVERAGE_SCHEMA_VERSION,
    direction_pocket_wilson_upper_95,
    individual_input_influence_layout,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coverage(rows: int) -> dict[str, object]:
    return {
        "schema_version": UTC_TIME_COVERAGE_SCHEMA_VERSION,
        "rows": rows,
        "first_utc": "2026-01-01T00:00:00+00:00",
        "last_utc": "2026-04-10T00:00:00+00:00",
        "utc_ns_sha256": "c" * 64,
    }


def passing_serve_source_identity() -> dict[str, object]:
    return {
        "schema_version": SERVE_SOURCE_IDENTITY_SCHEMA_VERSION,
        "contract": SERVE_SOURCE_IDENTITY_CONTRACT,
        "excluded_transaction_bound_paths": list(
            SERVE_SOURCE_IDENTITY_EXCLUDED_TRACKED_PATHS
        ),
        "tracked_file_count": 100,
        "tracked_total_bytes": 10_000,
        "tracked_paths_sha256": "8" * 64,
        "tracked_bytes_sha256": "9" * 64,
        "untracked_source_paths": [],
    }


def exact_specialist_indices() -> dict[str, list[int]]:
    counts = {
        "structure_swing_encoder": 70,
        "smc_liquidity_encoder": 87,
        "trend_ema_encoder": 32,
        "vol_compression_encoder": 49,
        "momentum_flow_encoder": 31,
        "session_regime_encoder": 123,
        "chart_geometry_encoder": 58,
        "price_action_candle_encoder": 63,
    }
    assert tuple(counts) == tuple(MODEL_NATIVE_REQUIRED_SPECIALISTS)
    start = 0
    result: dict[str, list[int]] = {}
    for specialist, count in counts.items():
        result[specialist] = list(range(start, start + count))
        start += count
    assert start == MODEL_NATIVE_SIGNAL_DIM
    return result


def passing_test_prediction_liveness(rows: int) -> dict[str, object]:
    active_head_evidence: dict[str, object] = {}
    for head, field_contract in SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS.items():
        active_head_evidence[head] = {
            "decision": "PASS",
            "failures": [],
            "fields": {
                field: {
                    "width": width,
                    "rows": rows,
                    "finite": True,
                    "component_std": [0.01] * width,
                    "min_component_std": 0.01,
                }
                for field, width in field_contract.items()
            },
        }
    def cooperation_gate(tokens: tuple[str, ...]) -> dict[str, object]:
        weight = 1.0 / len(tokens)
        return {
            "decision": "PASS",
            "failures": [],
            "rows": rows,
            "finite": True,
            "tokens": list(tokens),
            "row_sum_max_abs_error": 0.0,
            "entropy_mean": 1.5,
            "mean_weight": {token: weight for token in tokens},
            "std_weight": {token: 0.01 for token in tokens},
            "top_rank_count": {token: 1 for token in tokens},
            "thresholds": {
                "row_sum_max_abs_error": (
                    SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
                ),
                "min_mean_weight_exclusive": (
                    SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT
                ),
                "min_entropy_inclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY,
                "min_std_exclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_STD,
                "min_top_rank_count_inclusive": (
                    SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT
                ),
            },
        }

    weight = 1.0 / len(MODEL_NATIVE_REQUIRED_SPECIALISTS)
    timeframes = ("M5", "M15", "H1", "H4", "D1")
    return {
        "decision": "PASS",
        "failures": [],
        "rows": rows,
        "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
        "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
        "head_variation_epsilon": SERVE_PARITY_HEAD_VARIATION_EPSILON,
        "active_head_evidence": active_head_evidence,
        "specialist_gate": {
            "decision": "PASS",
            "failures": [],
            "rows": rows,
            "finite": True,
            "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "row_sum_max_abs_error": 0.0,
            "entropy_mean": 1.5,
            "mean_weight": {
                specialist: weight
                for specialist in MODEL_NATIVE_REQUIRED_SPECIALISTS
            },
            "std_weight": {
                specialist: 0.01
                for specialist in MODEL_NATIVE_REQUIRED_SPECIALISTS
            },
            "top_rank_count": {
                specialist: 1
                for specialist in MODEL_NATIVE_REQUIRED_SPECIALISTS
            },
            "thresholds": {
                "row_sum_max_abs_error": (
                    SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
                ),
                "min_mean_weight_exclusive": (
                    SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT
                ),
                "min_entropy_inclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY,
                "min_std_exclusive": SERVE_PARITY_SPECIALIST_GATE_MIN_STD,
                "min_top_rank_count_inclusive": (
                    SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT
                ),
            },
        },
        "tf_gate": cooperation_gate(timeframes),
        "family_tf_cooperation_gate": cooperation_gate(
            SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS
        ),
        "family_tf_feature_gate": {
            "decision": "PASS",
            "failures": [],
            "rows": rows,
            "finite": True,
            "tokens": list(SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS),
            "mean_weight": {
                token: 1.0 for token in SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS
            },
            "std_weight": {
                token: 0.01 for token in SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS
            },
            "min_observed": {
                token: 0.9 for token in SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS
            },
            "max_observed": {
                token: 1.1 for token in SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS
            },
            "thresholds": {
                "min_weight_exclusive": SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE,
                "max_weight_exclusive": SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE,
                "min_std_exclusive": SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE,
            },
        },
    }


def passing_specialist_decision_influence() -> dict[str, object]:
    indices = exact_specialist_indices()
    methods: dict[str, object] = {}
    for method in SERVE_PARITY_SPECIALIST_INFLUENCE_METHODS:
        methods[method] = {
            "decision": "PASS",
            "failures": [],
            "ablation_surface": (
                "seq_and_snap_exact_specialist_input_indices"
                if method == "input_family_mask"
                else "specialist_encoder_output_zero_hook"
            ),
            "specialists": {
                specialist: {
                    "decision": "PASS",
                    "failures": [],
                    "target": (
                        f"signal_indices:{specialist}"
                        if method == "input_family_mask"
                        else f"model.specialist_encoder.{specialist}"
                    ),
                    "input_indices_sha256": _canonical_sha256(
                        indices[specialist]
                    ),
                    "max_abs_class_centered_raw_logit_delta": 0.01,
                    "raw_changed_rows": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
                    "max_abs_class_centered_logit_delta": 0.01,
                    "changed_rows": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
                    "total_rows": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
                }
                for specialist in MODEL_NATIVE_REQUIRED_SPECIALISTS
            },
        }
    return {
        "decision": "PASS",
        "failures": [],
        "sample_count": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": coverage(
            SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT
        ),
        "comparison_surface": SERVE_PARITY_SPECIALIST_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS,
        "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "specialist_input_indices": indices,
        "specialist_input_indices_sha256": _canonical_sha256(indices),
        "model_metadata_indices_exact_match": True,
        "model_buffer_indices_exact_match": True,
        "methods": methods,
    }


def passing_upstream_context_decision_influence() -> dict[str, object]:
    metrics = {
        method: {
            "decision": "PASS",
            "failures": [],
            "target": (
                "model.input.ctx_cont"
                if method == "ctx_cont_zero_mask"
                else "model.input.ctx_cat"
            ),
            "ablation_surface": "full_tensor_zero_mask",
            "max_abs_class_centered_raw_logit_delta": 0.01,
            "raw_changed_rows": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
            "max_abs_class_centered_logit_delta": 0.01,
            "changed_rows": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
            "total_rows": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
        }
        for method in SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS
    }
    return {
        "decision": "PASS",
        "failures": [],
        "sample_count": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": coverage(
            SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT
        ),
        "comparison_surface": SERVE_PARITY_UPSTREAM_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS,
        "methods": list(SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS),
        "metrics": metrics,
    }


def passing_individual_input_decision_influence() -> dict[str, object]:
    signal_names = [
        f"model_native_signal_{index:03d}"
        for index in range(MODEL_NATIVE_SIGNAL_DIM)
    ]
    ownership = individual_input_influence_layout(signal_names)
    numeric_owners = ownership["numeric"]
    numeric = {
        surface: {
            "tokens": owner["tokens"],
            "source_indices": owner["source_indices"],
            "metrics": {
                token: {
                    "decision": "PASS",
                    "failures": [],
                    "max_abs_raw_class_margin_gradient": 0.01,
                    "max_abs_final_class_margin_gradient": 0.01,
                }
                for token in owner["tokens"]
            },
        }
        for surface, owner in numeric_owners.items()
    }
    categorical = {
        owner["token"]: {
            "decision": "PASS",
            "failures": [],
            "counterfactual": "next_valid_category_on_exact_owner_manifold",
            "max_abs_class_centered_raw_logit_delta": 0.01,
            "raw_changed_rows": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
            "max_abs_class_centered_logit_delta": 0.01,
            "changed_rows": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
            "total_rows": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
        }
        for owner in ownership["categorical"]
    }
    continuous_manifold = {
        owner["token"]: {
            "decision": "PASS",
            "failures": [],
            "counterfactual": (
                "joint_seq_last_snap_ctx_cont_binary_flip_or_train_center"
            ),
            "max_abs_class_centered_raw_logit_delta": 0.01,
            "raw_changed_rows": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
            "max_abs_class_centered_logit_delta": 0.01,
            "changed_rows": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
            "total_rows": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
        }
        for owner in ownership["continuous_manifold"]
    }
    return {
        "decision": "PASS",
        "failures": [],
        "sample_count": SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
        "sample_positions": list(
            SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS
        ),
        "sampled_test_coverage": coverage(
            SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT
        ),
        "comparison_surface": (
            SERVE_PARITY_INDIVIDUAL_INPUT_COMPARISON_SURFACE
        ),
        "gradient_epsilon": (
            SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON
        ),
        "categorical_delta_epsilon": (
            SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON
        ),
        "numeric_input_count": sum(
            len(owner["tokens"]) for owner in numeric_owners.values()
        ),
        "continuous_manifold_input_count": len(
            ownership["continuous_manifold"]
        ),
        "categorical_input_count": len(ownership["categorical"]),
        "ordered_signal_names": signal_names,
        "signal_names_sha256": _canonical_sha256(signal_names),
        "ctx_cont_names_sha256": _canonical_sha256(
            list(MODEL_NATIVE_CTX_CONT_FIELDS)
        ),
        "ctx_cat_names_sha256": _canonical_sha256(
            list(MODEL_NATIVE_CTX_CAT_FIELDS)
        ),
        "input_ownership": ownership,
        "input_ownership_sha256": _canonical_sha256(ownership),
        "numeric": numeric,
        "continuous_manifold": continuous_manifold,
        "categorical": categorical,
    }


def passing_multi_tf_decision_influence() -> dict[str, object]:
    metrics = {
        timeframe: {
            "decision": "PASS",
            "failures": [],
            "target": f"model.input.seq_{timeframe.lower()}",
            "ablation_surface": "full_tensor_zero_mask",
            "max_abs_class_centered_raw_logit_delta": 0.01,
            "raw_changed_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
            "max_abs_class_centered_logit_delta": 0.01,
            "changed_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
            "total_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        }
        for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
    }
    return {
        "decision": "PASS",
        "failures": [],
        "sample_count": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": coverage(
            SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT
        ),
        "comparison_surface": SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
        "timeframes": list(SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES),
        "ablation": "candidate_specific_full_tensor_zero_ablation_v1",
        "metrics": metrics,
    }


def passing_family_tf_decision_influence() -> dict[str, object]:
    routing = require_multi_tf_specialist_routing_v4(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    tokens = list(SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS)
    metrics = {}
    for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES:
        for specialist, indices in routing.items():
            token = f"{timeframe.lower()}:{specialist}"
            metrics[token] = {
                "decision": "PASS",
                "failures": [],
                "target": (
                    f"model.input.seq_{timeframe.lower()}["
                    f"{','.join(str(index) for index in indices)}]"
                ),
                "ablation_surface": "exact_family_feature_indices_zero_mask",
                "max_abs_class_centered_raw_logit_delta": 0.01,
                "raw_changed_rows": (
                    SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT
                ),
                "max_abs_class_centered_logit_delta": 0.01,
                "changed_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
                "total_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
            }
    return {
        "decision": "PASS",
        "failures": [],
        "sample_count": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(
            SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS
        ),
        "sampled_test_coverage": coverage(
            SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT
        ),
        "comparison_surface": SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
        "family_timeframe_tokens": tokens,
        "ablation": "candidate_specific_family_tensor_index_zero_ablation_v1",
        "metrics": metrics,
    }


def passing_serve_parity_liveness_sections(
    test_rows: int,
    *,
    bundle_dir: str = "/home/andre2/unit_bundle",
    bundle_metadata_sha256: str = "e" * 64,
    master_transformer_lock_sha256: str = "f" * 64,
) -> dict[str, object]:
    return {
        "test_prediction_liveness": passing_test_prediction_liveness(test_rows),
        "specialist_decision_influence": passing_specialist_decision_influence(),
        "individual_input_decision_influence": (
            passing_individual_input_decision_influence()
        ),
        "upstream_context_decision_influence": (
            passing_upstream_context_decision_influence()
        ),
        "multi_tf_decision_influence": passing_multi_tf_decision_influence(),
        "family_tf_decision_influence": (
            passing_family_tf_decision_influence()
        ),
        "forward_parity_per_head_tolerance": {
            head: SERVE_PARITY_FORWARD_TOL for head in SERVE_PARITY_FORWARD_HEADS
        },
    }


def passing_direction_repair_pockets() -> dict[str, dict[str, object]]:
    selected_rows = 120
    error_count = 2
    correct_count = selected_rows - error_count
    error_rate = error_count / selected_rows
    error_wilson = direction_pocket_wilson_upper_95(
        failures=error_count,
        total=selected_rows,
    )
    pockets: dict[str, dict[str, object]] = {}
    for name in DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS:
        pockets[name] = {
            "rows": 140,
            "selected_rows": selected_rows,
            "selected_side_long_count": selected_rows // 2,
            "selected_side_short_count": selected_rows // 2,
            "selected_side_long_rate": 0.5,
            "selected_side_short_rate": 0.5,
            "selected_label_correct_count": correct_count,
            "selected_label_error_count": error_count,
            "selected_label_correct_rate": correct_count / selected_rows,
            "selected_label_error_rate": error_rate,
            "selected_label_error_wilson_upper_95": error_wilson,
            "selected_mean_proxy_pnl_bps": 12.0,
        }
    return pockets
