from __future__ import annotations

import hashlib
import json

from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    DIRECTION_POCKET_LONG_WRONG_SIDE_REPAIR_POCKETS,
    DIRECTION_POCKET_SHORT_WRONG_SIDE_REPAIR_POCKETS,
    SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS,
    SERVE_PARITY_CALIBRATION_EQUATION,
    SERVE_PARITY_CALIBRATION_TOL,
    SERVE_PARITY_FORWARD_HEADS,
    SERVE_PARITY_FORWARD_TOL,
    SERVE_PARITY_FUSION_INFLUENCE_COMPARISON_SURFACE,
    SERVE_PARITY_FUSION_INFLUENCE_EPSILON,
    SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS,
    SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
    SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_POSITIONS,
    SERVE_PARITY_FUSION_INFLUENCE_SAMPLING_CONTRACT,
    SERVE_PARITY_FUSION_REFERENCE_AGGREGATION,
    SERVE_PARITY_FUSION_REFERENCE_SPLIT,
    SERVE_PARITY_HEAD_VARIATION_EPSILON,
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
    UTC_TIME_COVERAGE_SCHEMA_VERSION,
    direction_pocket_wilson_upper_95,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    INPUTS as DIRECTION_EVIDENCE_FUSION_INPUTS,
    INPUTS_SHA256 as DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256,
    INPUT_DIM as DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
    ORDERED_INPUT_LAYOUT as DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT,
    direction_evidence_fusion_metadata,
)


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
    assert start == 513
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
    weight = 1.0 / len(MODEL_NATIVE_REQUIRED_SPECIALISTS)
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


def passing_multi_tf_decision_influence() -> dict[str, object]:
    metrics = {
        timeframe: {
            "decision": "PASS",
            "failures": [],
            "target": f"model.input.seq_{timeframe.lower()}",
            "ablation_surface": "full_tensor_zero_mask",
            "max_abs_class_centered_raw_logit_delta": 0.01,
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


def passing_direction_evidence_fusion_influence(
    *,
    bundle_dir: str,
    bundle_metadata_sha256: str,
    master_transformer_lock_sha256: str,
) -> dict[str, object]:
    means = {
        name: [0.01 * (index + 1) for index in range(width)]
        for name, width in DIRECTION_EVIDENCE_FUSION_INPUTS
    }
    ordered = [
        item
        for name, _width in DIRECTION_EVIDENCE_FUSION_INPUTS
        for item in means[name]
    ]
    reference = {
        "split": SERVE_PARITY_FUSION_REFERENCE_SPLIT,
        "aggregation": SERVE_PARITY_FUSION_REFERENCE_AGGREGATION,
        "coverage": coverage(500),
        "input_dim": DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
        "inputs_sha256": DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256,
        "mean_by_input": means,
        "ordered_mean_sha256": _canonical_sha256(ordered),
    }
    groups: dict[str, object] = {}
    for layout in DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT:
        name = str(layout["name"])
        groups[name] = {
            "decision": "PASS",
            "failures": [],
            "target": (
                f"model.evidence_fusion_norm.input["
                f"{layout['start']}:{layout['stop']}]"
            ),
            "start": layout["start"],
            "stop": layout["stop"],
            "width": layout["width"],
            "reference_values_sha256": _canonical_sha256(means[name]),
            "max_abs_class_centered_logit_delta": 0.01,
            "changed_rows": SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
            "total_rows": SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
        }
    normalized_bundle = bundle_dir.rstrip("/")
    return {
        "decision": "PASS",
        "failures": [],
        "sample_count": SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
        "sampling_contract": SERVE_PARITY_FUSION_INFLUENCE_SAMPLING_CONTRACT,
        "sample_positions": list(SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_POSITIONS),
        "sampled_test_coverage": coverage(
            SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT
        ),
        "comparison_surface": SERVE_PARITY_FUSION_INFLUENCE_COMPARISON_SURFACE,
        "epsilon": SERVE_PARITY_FUSION_INFLUENCE_EPSILON,
        "min_changed_rows": SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS,
        "ablation": "replace_exact_fusion_slice_with_immutable_candidate_val_mean_v1",
        "fusion_metadata": direction_evidence_fusion_metadata(),
        "ordered_input_layout": DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT,
        "inputs_sha256": DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256,
        "input_dim": DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
        "bundle_metadata_path": f"{normalized_bundle}/bundle_metadata.json",
        "bundle_metadata_sha256": bundle_metadata_sha256,
        "bundle_metadata_exact_match": True,
        "master_transformer_lock_path": (
            f"{normalized_bundle}/MASTER_TRANSFORMER_LOCK.json"
        ),
        "master_transformer_lock_sha256": master_transformer_lock_sha256,
        "master_transformer_lock_exact_match": True,
        "reference": reference,
        "groups": groups,
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
        "upstream_context_decision_influence": (
            passing_upstream_context_decision_influence()
        ),
        "multi_tf_decision_influence": passing_multi_tf_decision_influence(),
        "direction_evidence_fusion_influence": (
            passing_direction_evidence_fusion_influence(
                bundle_dir=bundle_dir,
                bundle_metadata_sha256=bundle_metadata_sha256,
                master_transformer_lock_sha256=master_transformer_lock_sha256,
            )
        ),
        "forward_parity_per_head_tolerance": {
            head: SERVE_PARITY_FORWARD_TOL for head in SERVE_PARITY_FORWARD_HEADS
        },
        "direction_calibration_parity": {
            "decision": "PASS",
            "failures": [],
            "n_compared": 256,
            "equation": SERVE_PARITY_CALIBRATION_EQUATION,
            "enabled": True,
            "temperature": 1.0,
            "bias": [0.0, 0.0, 0.0],
            "tolerance": SERVE_PARITY_CALIBRATION_TOL,
            "max_abs_diff": 0.0,
            "worst_ts": None,
        },
    }


def passing_direction_repair_pockets() -> dict[str, dict[str, object]]:
    selected_rows = 120
    wrong_count = 2
    right_count = selected_rows - wrong_count
    wrong_rate = wrong_count / selected_rows
    wrong_wilson = direction_pocket_wilson_upper_95(
        failures=wrong_count,
        total=selected_rows,
    )
    pockets: dict[str, dict[str, object]] = {}
    for name in DIRECTION_POCKET_SHORT_WRONG_SIDE_REPAIR_POCKETS:
        pockets[name] = {
            "rows": 140,
            "selected_rows": selected_rows,
            "selected_side_long_count": right_count,
            "selected_side_short_count": wrong_count,
            "selected_side_long_rate": right_count / selected_rows,
            "selected_side_short_rate": wrong_rate,
            "selected_side_long_wilson_upper_95": direction_pocket_wilson_upper_95(
                failures=right_count,
                total=selected_rows,
            ),
            "selected_side_short_wilson_upper_95": wrong_wilson,
            "selected_mean_proxy_pnl_bps": 12.0,
        }
    for name in DIRECTION_POCKET_LONG_WRONG_SIDE_REPAIR_POCKETS:
        pockets[name] = {
            "rows": 140,
            "selected_rows": selected_rows,
            "selected_side_long_count": wrong_count,
            "selected_side_short_count": right_count,
            "selected_side_long_rate": wrong_rate,
            "selected_side_short_rate": right_count / selected_rows,
            "selected_side_long_wilson_upper_95": wrong_wilson,
            "selected_side_short_wilson_upper_95": direction_pocket_wilson_upper_95(
                failures=right_count,
                total=selected_rows,
            ),
            "selected_mean_proxy_pnl_bps": 12.0,
        }
    return pockets
