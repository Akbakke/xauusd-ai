"""Exact launch-admission contract for model-native XAU Entry evidence.

This module contains evidence-admission constants only.  None of the values
are live direction rules: LONG/SHORT/FLAT remains the model's final calibrated
``direction_logits`` argmax.  The constants make the offline TEST proof
surface immutable and prevent callers from weakening its scope.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.features.htf_features import (
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_TIMEFRAMES,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    require_multi_tf_specialist_routing_v4,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    INPUTS as DIRECTION_EVIDENCE_FUSION_INPUTS,
    INPUTS_SHA256 as DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256,
    INPUT_DIM as DIRECTION_EVIDENCE_FUSION_INPUT_DIM,
    ORDERED_INPUT_LAYOUT as DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT,
    direction_evidence_fusion_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION,
)

MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION = (
    "xau_model_native_exact_test_full_stack_serve_gate_v11"
)
MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION = "model_native_serve_parity_v11"
MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION = (
    "model_native_direction_pocket_audit_v2"
)
MODEL_NATIVE_REQUIRED_TEST_SPLIT = "test"
MODEL_NATIVE_REQUIRED_MODEL_NAME = "candidate"

SERVE_PARITY_SAMPLE_COUNT = 256
SERVE_PARITY_STATE_TOL = 1e-5
SERVE_PARITY_FORWARD_TOL = 1e-3
SERVE_PARITY_SAMPLING_CONTRACT = (
    "deterministic_even_positions_over_exact_full_test_coverage_v1"
)
SERVE_PARITY_ENV_PINS = {
    "CUDA_VISIBLE_DEVICES": "",
}
SERVE_SOURCE_IDENTITY_SCHEMA_VERSION = "gx1_serve_source_identity_v1"
SERVE_SOURCE_IDENTITY_CONTRACT = (
    "git_tracked_bytes_excluding_transaction_bound_mutable_authority_v1"
)
SERVE_SOURCE_IDENTITY_EXCLUDED_TRACKED_PATHS = (
    "PROJECT_STATE_artifacts.json",
    "PROJECT_STATE_xau_direction_launch.json",
)
SERVE_SOURCE_IDENTITY_UNTRACKED_SOURCE_SUFFIXES = (
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
)
SERVE_SOURCE_IDENTITY_UNTRACKED_SOURCE_ROOTS = (
    "gx1/",
    "gx1_guards/",
    "scripts/",
)
SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES = MULTI_TF_TIMEFRAMES
SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS = (
    0,
    36,
    73,
    109,
    146,
    182,
    219,
    255,
)
SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT = len(
    SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS
)
SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON = 1e-12
SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON = 1e-7
SERVE_PARITY_INDIVIDUAL_INPUT_COMPARISON_SURFACE = (
    "all_class_margin_input_gradients_plus_valid_categorical_counterfactual_v1"
)
SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS = tuple(
    f"{timeframe.lower()}:{specialist}"
    for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
    for specialist in MODEL_NATIVE_REQUIRED_SPECIALISTS
)
SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS = tuple(
    f"{timeframe.lower()}:{feature}"
    for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
    for feature in MULTI_TF_PER_BAR_FEATURES_V4
)
if len(SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS) != 40:
    raise RuntimeError("SERVE_PARITY_FAMILY_TF_COOPERATION_TOKEN_COUNT_INVALID")
# Token width is DERIVED from the one ordered per-TF surface owner (rule 13;
# V29 stage 2 grew the surface, so a restated literal would drift).
if len(SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS) != (
    len(SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES)
    * len(MULTI_TF_PER_BAR_FEATURES_V4)
):
    raise RuntimeError("SERVE_PARITY_FAMILY_TF_FEATURE_TOKEN_COUNT_INVALID")

# Exact retained model tensors plus every numeric diagnostic consumed by the
# live head schema.  Width one is scalar; wider values are dense vectors.  This
# is TRAIN==SERVE comparison evidence only and never modifies a direction.
SERVE_PARITY_FORWARD_FIELD_WIDTHS = {
    **dict(DIRECTION_EVIDENCE_FUSION_INPUTS),
    "path_quality": 1,
    "bad_path_logit": 1,
    "specialist_gate": len(MODEL_NATIVE_REQUIRED_SPECIALISTS),
    "tf_gate": 5,
    "family_tf_cooperation_gate": len(
        SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS
    ),
    "family_tf_feature_gate": len(SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS),
    "raw_direction_logits": 3,
    "direction_logits": 3,
    "public_trade_flat_decision_logits": 2,
    "direction_probs": 3,
    "public_trade_flat_decision_probs": 2,
    "p_long": 1,
    "p_short": 1,
    "p_flat": 1,
    "edge_score": 1,
    "public_trade_probability": 1,
    "public_flat_probability": 1,
    "path_quality_pred": 1,
    "tradable_prob": 1,
    "mfe_first_n_pred": 1,
    "bad_path_prob": 1,
    "clean_edge_prob": 1,
    "survival_prob": 1,
    "tf_agreement_prob": 1,
    "path_quality_std": 1,
    "position_size_pred": 1,
    "p_long_given_trade": 1,
    "p_short_given_trade": 1,
    "side_probs": 2,
    "long_bad_path_prob": 1,
    "short_bad_path_prob": 1,
    "long_validity_prob": 1,
    "short_validity_prob": 1,
    "mtf_dir_probs": 3,
    "trendline_rail_probs": 6,
}
SERVE_PARITY_FORWARD_HEADS = tuple(SERVE_PARITY_FORWARD_FIELD_WIDTHS)

# Each active model head must have finite, non-constant evidence over the
# complete, hash-bound candidate TEST prediction artifact.  Width one denotes
# a scalar column; wider values are dense vector columns with that exact width.
SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS = {
    "direction": {
        "model_native_logits": 3,
        "raw_direction_logits": 3,
        "direction_logits": 3,
    },
    "tradable": {"tradable_logit": 1, "tradable_prob": 1},
    "path_quality": {
        "path_quality_raw": 1,
        "path_quality": 1,
        "path_quality_pred": 1,
    },
    "mfe_first_n": {"mfe_first_n": 1, "mfe_first_n_pred": 1},
    "bad_path": {
        "bad_path_logit_raw": 1,
        "bad_path_logit": 1,
        "bad_path_prob": 1,
    },
    "clean_edge": {"clean_edge_logit": 1, "clean_edge_prob": 1},
    "survival": {"survival_logit": 1, "survival_prob": 1},
    "tf_agreement": {"tf_agreement_logit": 1, "tf_agreement_prob": 1},
    "path_quality_log_var": {"path_quality_log_var": 1},
    "position_size": {"position_size_logit": 1, "position_size_pred": 1},
    "dip": {"dip_pred": 18},
    "forecast": {"forecast_pred": 4},
    "timing": {"timing_pred": 12},
    "tail_risk": {"tail_risk_pred": 6},
    "vol_forecast": {"vol_forecast_pred": 3},
    "mtf_direction": {"mtf_dir_logits": 3},
    "trade_side_hierarchy": {
        "trade_logit": 1,
        "side_logits": 2,
        "side_utility": 2,
        "side_bad_path_logit": 2,
        "side_mae": 2,
    },
    "trendline_rail": {"trendline_rail_logits": 6},
    "side_validity": {"side_validity_logit": 2},
    "offline_rl_action_value": {
        "action_value": 9,
        "action_advantage": 9,
    },
    "offline_rl_expectile_value": {"expectile_value": 3},
    "model_native_evidence_fusion": dict(DIRECTION_EVIDENCE_FUSION_INPUTS),
}
SERVE_PARITY_HEAD_VARIATION_EPSILON = 1e-8
SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR = 1e-5
SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT = 0.01
SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY = 0.5
SERVE_PARITY_SPECIALIST_GATE_MIN_STD = 1e-6
SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT = 1
SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE = 0.0
SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE = 2.0
SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE = 1e-6

# Direct decision influence is audited on a deterministic subset of the same
# 256 parity states.  Both evidence-family masking and an isolated specialist
# encoder-output hook ablation must move both class-centred raw and final
# calibrated logits.
SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT = 16
SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS = tuple(range(0, 256, 17))
SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT = (
    "deterministic_even_positions_over_exact_256_parity_states_v1"
)
SERVE_PARITY_SPECIALIST_INFLUENCE_COMPARISON_SURFACE = (
    "class_centered_raw_and_final_calibrated_direction_logits_v2"
)
SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON = 1e-6
SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS = 8
SERVE_PARITY_SPECIALIST_INFLUENCE_METHODS = (
    "input_family_mask",
    "encoder_output_hook_ablation",
)
SERVE_PARITY_CALIBRATION_EQUATION = (
    "direction_logits=raw_direction_logits/temperature+bias"
)
SERVE_PARITY_CALIBRATION_TOL = 1e-6
SERVE_PARITY_FUSION_REFERENCE_SPLIT = "val"
SERVE_PARITY_FUSION_REFERENCE_AGGREGATION = "finite_column_mean_v1"
SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT = 16
SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_POSITIONS = (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS
)
SERVE_PARITY_FUSION_INFLUENCE_SAMPLING_CONTRACT = (
    "deterministic_even_positions_over_exact_256_parity_states_v1"
)
SERVE_PARITY_FUSION_INFLUENCE_COMPARISON_SURFACE = (
    "class_centered_raw_and_final_calibrated_direction_logits_v2"
)
SERVE_PARITY_FUSION_INFLUENCE_EPSILON = 1e-6
SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS = 8
SERVE_PARITY_FUSION_INPUT_GRADIENT_EPSILON = 1e-12
SERVE_PARITY_FUSION_DERIVED_MANIFOLD_INPUTS = (
    "action_value",
    "expectile_value",
    "action_advantage",
)
SERVE_PARITY_FUSION_DERIVED_RELATION_ATOL = 1e-6
SERVE_PARITY_FUSION_INFLUENCE_ABLATION = (
    "exact_slice_or_q_v_a_manifold_preserving_candidate_val_mean_v2"
)
SERVE_PARITY_FUSION_DERIVED_ABLATION_SURFACES = {
    "action_value": (
        "q_val_mean_plus_recomputed_advantage_preserving_q_minus_v"
    ),
    "expectile_value": (
        "v_val_mean_plus_recomputed_advantage_preserving_q_minus_v"
    ),
    "action_advantage": (
        "joint_q_v_val_means_plus_recomputed_advantage_preserving_q_minus_v"
    ),
}
SERVE_PARITY_FUSION_DERIVED_REFERENCE_INPUTS = {
    "action_value": ("action_value",),
    "expectile_value": ("expectile_value",),
    "action_advantage": ("action_value", "expectile_value"),
}
SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT = (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT
)
SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_POSITIONS = (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS
)
SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLING_CONTRACT = (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT
)
SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS = (
    "ctx_cont_zero_mask",
    "ctx_cat_zero_mask",
)
SERVE_PARITY_UPSTREAM_INFLUENCE_COMPARISON_SURFACE = (
    "class_centered_raw_and_final_calibrated_direction_logits_v2"
)
SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON = 1e-6
SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS = 8
SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT = (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT
)
SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS = (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS
)
SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT = (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT
)
SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE = (
    "class_centered_raw_and_final_calibrated_direction_logits_v2"
)
SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON = 1e-6
SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS = 8

if tuple(SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS) != tuple(
    MODEL_NATIVE_ACTIVE_HEADS
):
    raise RuntimeError("SERVE_PARITY_ACTIVE_HEAD_SET_ORDER_MISMATCH")
if len(SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS) != (
    SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT
):
    raise RuntimeError("SERVE_PARITY_SPECIALIST_SAMPLE_POSITION_COUNT_MISMATCH")

DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE = 0.10
DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95 = 0.15
DIRECTION_POCKET_WILSON_CONFIDENCE_LEVEL = 0.95
DIRECTION_POCKET_WILSON_Z = 1.959963984540054
DIRECTION_POCKET_MIN_SELECTED_ROWS = 100
DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE = 0.0
DIRECTION_POCKET_SPREAD_AWARE_PROXY_CONTRACT = (
    "canonical_side_path_utility_bps_after_spread_v1"
)
DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS = (
    "intraday_bull",
    "intraday_bull__htf_bull",
    "intraday_bull__htf_bear",
    "intraday_bear",
    "intraday_bear__htf_bear",
    "intraday_bear__htf_bull",
    "rising_channel_support_touch",
    "support_retest_continuation",
    "rising_channel_support_continuation",
    "countertrend_short_trap",
    "short_high_mae_low_mfe_early_failure",
    "falling_channel_resistance_touch",
    "resistance_retest_continuation",
    "falling_channel_resistance_continuation",
    "countertrend_long_trap",
    "long_high_mae_low_mfe_early_failure",
)

UTC_TIME_COVERAGE_SCHEMA_VERSION = "sorted_unique_utc_ns_sha256_v1"
_TIME_COVERAGE_KEYS = {
    "schema_version",
    "rows",
    "first_utc",
    "last_utc",
    "utc_ns_sha256",
}


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _is_exact_number(value: object, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) == float(expected)
    )


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
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


def _git_null_paths(repo_root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args, "-z"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"serve source identity git {' '.join(args)} failed: {stderr}"
        )
    return sorted(
        item.decode("utf-8", errors="strict")
        for item in result.stdout.split(b"\0")
        if item
    )


def build_serve_source_identity(repo_root: Path) -> dict[str, Any]:
    """Hash exact tracked repository bytes used by parity and live serving.

    The two mutable authority JSON files are excluded because the transactional
    finalizer necessarily replaces them after parity.  Their exact bytes remain
    separately hash-bound by the launch transaction contract.  All other
    tracked bytes, plus the absence of untracked source/config files in runtime
    roots, remain mandatory and are independent of whether authority updates
    have subsequently been committed.
    """

    root = Path(repo_root).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"serve source identity root is not a git repo: {root}")
    tracked = _git_null_paths(root, "ls-files", "--cached")
    missing_exclusions = sorted(
        set(SERVE_SOURCE_IDENTITY_EXCLUDED_TRACKED_PATHS) - set(tracked)
    )
    if missing_exclusions:
        raise RuntimeError(
            "serve source identity mutable authority files are not tracked: "
            + ",".join(missing_exclusions)
        )
    included = [
        relative
        for relative in tracked
        if relative not in SERVE_SOURCE_IDENTITY_EXCLUDED_TRACKED_PATHS
    ]
    path_digest = hashlib.sha256()
    byte_digest = hashlib.sha256()
    total_bytes = 0
    for relative in included:
        encoded_path = relative.encode("utf-8")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                "serve source identity tracked path is missing/non-regular/symlink: "
                f"{relative}"
            )
        payload = path.read_bytes()
        total_bytes += len(payload)
        path_digest.update(len(encoded_path).to_bytes(8, "big"))
        path_digest.update(encoded_path)
        byte_digest.update(len(encoded_path).to_bytes(8, "big"))
        byte_digest.update(encoded_path)
        byte_digest.update(len(payload).to_bytes(8, "big"))
        byte_digest.update(payload)

    untracked = _git_null_paths(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
    )
    untracked_source = [
        relative
        for relative in untracked
        if (
            relative.startswith(SERVE_SOURCE_IDENTITY_UNTRACKED_SOURCE_ROOTS)
            or (
                "/" not in relative
                and relative.endswith(
                    SERVE_SOURCE_IDENTITY_UNTRACKED_SOURCE_SUFFIXES
                )
            )
        )
        and relative.endswith(SERVE_SOURCE_IDENTITY_UNTRACKED_SOURCE_SUFFIXES)
    ]
    return {
        "schema_version": SERVE_SOURCE_IDENTITY_SCHEMA_VERSION,
        "contract": SERVE_SOURCE_IDENTITY_CONTRACT,
        "excluded_transaction_bound_paths": list(
            SERVE_SOURCE_IDENTITY_EXCLUDED_TRACKED_PATHS
        ),
        "tracked_file_count": len(included),
        "tracked_total_bytes": total_bytes,
        "tracked_paths_sha256": path_digest.hexdigest(),
        "tracked_bytes_sha256": byte_digest.hexdigest(),
        "untracked_source_paths": untracked_source,
    }


def serve_source_identity_contract_failures(value: object) -> list[str]:
    """Validate the immutable source-identity shape without reading the repo."""

    label = "serve parity source identity"
    if not isinstance(value, dict):
        return [f"{label} must be an exact object"]
    failures: list[str] = []
    expected_keys = {
        "schema_version",
        "contract",
        "excluded_transaction_bound_paths",
        "tracked_file_count",
        "tracked_total_bytes",
        "tracked_paths_sha256",
        "tracked_bytes_sha256",
        "untracked_source_paths",
    }
    if set(value) != expected_keys:
        failures.append(f"{label} keys mismatch")
    if value.get("schema_version") != SERVE_SOURCE_IDENTITY_SCHEMA_VERSION:
        failures.append(f"{label} schema_version mismatch")
    if value.get("contract") != SERVE_SOURCE_IDENTITY_CONTRACT:
        failures.append(f"{label} contract mismatch")
    if value.get("excluded_transaction_bound_paths") != list(
        SERVE_SOURCE_IDENTITY_EXCLUDED_TRACKED_PATHS
    ):
        failures.append(f"{label} authority exclusion mismatch")
    for field in ("tracked_file_count", "tracked_total_bytes"):
        observed = value.get(field)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed <= 0
        ):
            failures.append(f"{label} {field} must be a positive integer")
    for field in ("tracked_paths_sha256", "tracked_bytes_sha256"):
        if not _is_sha256(value.get(field)):
            failures.append(f"{label} {field} is not an exact SHA-256")
    if value.get("untracked_source_paths") != []:
        failures.append(f"{label} contains untracked runtime source/config paths")
    return failures


def direction_pocket_wilson_upper_95(*, failures: int, total: int) -> float:
    """Exact two-sided 95% Wilson upper bound used by immutable pocket proof."""

    if (
        isinstance(failures, bool)
        or isinstance(total, bool)
        or not isinstance(failures, int)
        or not isinstance(total, int)
        or total <= 0
        or failures < 0
        or failures > total
    ):
        raise ValueError("Wilson failures/total counts are invalid")
    rate = failures / total
    z = DIRECTION_POCKET_WILSON_Z
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = rate + z2 / (2.0 * total)
    radius = z * math.sqrt(
        (rate * (1.0 - rate) / total) + (z2 / (4.0 * total * total))
    )
    return (center + radius) / denominator


def _zero_failure_pass(value: object, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an exact object"]
    if value.get("decision") != "PASS" or value.get("failures") != []:
        return [f"{label} must be a zero-failure PASS"]
    return []


def _test_prediction_liveness_contract_failures(
    value: object,
    *,
    expected_rows: int,
) -> list[str]:
    label = "serve parity test_prediction_liveness"
    failures = _zero_failure_pass(value, label=label)
    if not isinstance(value, dict):
        return failures
    exact_keys = {
        "decision",
        "failures",
        "rows",
        "active_heads",
        "blocked_heads",
        "head_variation_epsilon",
        "active_head_evidence",
        "specialist_gate",
        "tf_gate",
        "family_tf_cooperation_gate",
        "family_tf_feature_gate",
    }
    if set(value) != exact_keys:
        failures.append(
            f"{label} keys={sorted(value)} expected={sorted(exact_keys)}"
        )
    if value.get("rows") != expected_rows:
        failures.append(f"{label}.rows mismatch")
    if value.get("active_heads") != list(MODEL_NATIVE_ACTIVE_HEADS):
        failures.append(f"{label}.active_heads mismatch")
    if value.get("blocked_heads") != list(MODEL_NATIVE_BLOCKED_HEADS):
        failures.append(f"{label}.blocked_heads mismatch")
    if not _is_exact_number(
        value.get("head_variation_epsilon"),
        SERVE_PARITY_HEAD_VARIATION_EPSILON,
    ):
        failures.append(f"{label}.head_variation_epsilon mismatch")

    evidence = value.get("active_head_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(
        MODEL_NATIVE_ACTIVE_HEADS
    ):
        failures.append(f"{label}.active_head_evidence set mismatch")
    else:
        for head, expected_fields in SERVE_PARITY_ACTIVE_HEAD_EVIDENCE_FIELDS.items():
            head_label = f"{label}.active_head_evidence.{head}"
            row = evidence.get(head)
            failures.extend(_zero_failure_pass(row, label=head_label))
            if not isinstance(row, dict):
                continue
            if set(row) != {"decision", "failures", "fields"}:
                failures.append(f"{head_label} keys mismatch")
            fields = row.get("fields")
            if not isinstance(fields, dict) or set(fields) != set(expected_fields):
                failures.append(f"{head_label}.fields set mismatch")
                continue
            for field, width in expected_fields.items():
                metric_label = f"{head_label}.fields.{field}"
                metric = fields.get(field)
                if not isinstance(metric, dict) or set(metric) != {
                    "width",
                    "rows",
                    "finite",
                    "component_std",
                    "min_component_std",
                }:
                    failures.append(f"{metric_label} metric keys mismatch")
                    continue
                if metric.get("width") != width:
                    failures.append(f"{metric_label}.width mismatch")
                if metric.get("rows") != expected_rows:
                    failures.append(f"{metric_label}.rows mismatch")
                if metric.get("finite") is not True:
                    failures.append(f"{metric_label} is not finite")
                component_std = metric.get("component_std")
                if (
                    not isinstance(component_std, list)
                    or len(component_std) != width
                    or not all(_is_finite_number(item) for item in component_std)
                ):
                    failures.append(f"{metric_label}.component_std is invalid")
                    continue
                std = metric.get("min_component_std")
                if not _is_finite_number(std) or float(std) <= (
                    SERVE_PARITY_HEAD_VARIATION_EPSILON
                ):
                    failures.append(
                        f"{metric_label}.min_component_std lacks required variation"
                    )
                elif abs(float(std) - min(float(item) for item in component_std)) > 1e-15:
                    failures.append(f"{metric_label}.min_component_std mismatch")

    gate = value.get("specialist_gate")
    gate_label = f"{label}.specialist_gate"
    failures.extend(_zero_failure_pass(gate, label=gate_label))
    if isinstance(gate, dict):
        exact_gate_keys = {
            "decision",
            "failures",
            "rows",
            "finite",
            "specialists",
            "row_sum_max_abs_error",
            "entropy_mean",
            "mean_weight",
            "std_weight",
            "top_rank_count",
            "thresholds",
        }
        if set(gate) != exact_gate_keys:
            failures.append(f"{gate_label} keys mismatch")
        if gate.get("rows") != expected_rows:
            failures.append(f"{gate_label}.rows mismatch")
        if gate.get("finite") is not True:
            failures.append(f"{gate_label} is not finite")
        if gate.get("specialists") != list(MODEL_NATIVE_REQUIRED_SPECIALISTS):
            failures.append(f"{gate_label}.specialists mismatch")
        expected_thresholds = {
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
        }
        if gate.get("thresholds") != expected_thresholds:
            failures.append(f"{gate_label}.thresholds mismatch")
        row_error = gate.get("row_sum_max_abs_error")
        if (
            not _is_finite_number(row_error)
            or float(row_error) < 0.0
            or float(row_error)
            > SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
        ):
            failures.append(f"{gate_label}.row_sum_max_abs_error exceeds contract")
        entropy = gate.get("entropy_mean")
        if not _is_finite_number(entropy) or float(entropy) < (
            SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY
        ):
            failures.append(f"{gate_label}.entropy_mean below contract")
        for metric_name, threshold, inclusive in (
            (
                "mean_weight",
                SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT,
                False,
            ),
            ("std_weight", SERVE_PARITY_SPECIALIST_GATE_MIN_STD, False),
            (
                "top_rank_count",
                SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT,
                True,
            ),
        ):
            metrics = gate.get(metric_name)
            if not isinstance(metrics, dict) or set(metrics) != set(
                MODEL_NATIVE_REQUIRED_SPECIALISTS
            ):
                failures.append(f"{gate_label}.{metric_name} set mismatch")
                continue
            for specialist, metric in metrics.items():
                if metric_name == "top_rank_count":
                    valid = (
                        not isinstance(metric, bool)
                        and isinstance(metric, int)
                        and metric >= int(threshold)
                    )
                else:
                    valid = _is_finite_number(metric) and (
                        float(metric) >= float(threshold)
                        if inclusive
                        else float(metric) > float(threshold)
                    )
                if not valid:
                    failures.append(
                        f"{gate_label}.{metric_name}.{specialist} violates contract"
                    )
        mean_weight = gate.get("mean_weight")
        if isinstance(mean_weight, dict) and all(
            _is_finite_number(item) for item in mean_weight.values()
        ):
            if abs(sum(float(item) for item in mean_weight.values()) - 1.0) > (
                SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
            ):
                failures.append(f"{gate_label}.mean_weight does not sum to one")
    for gate_name, tokens in (
        ("tf_gate", SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES),
        (
            "family_tf_cooperation_gate",
            SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS,
        ),
    ):
        report = value.get(gate_name)
        gate_label = f"{label}.{gate_name}"
        failures.extend(_zero_failure_pass(report, label=gate_label))
        if not isinstance(report, dict):
            continue
        expected_gate_keys = {
            "decision",
            "failures",
            "rows",
            "finite",
            "tokens",
            "row_sum_max_abs_error",
            "entropy_mean",
            "mean_weight",
            "std_weight",
            "top_rank_count",
            "thresholds",
        }
        if set(report) != expected_gate_keys:
            failures.append(f"{gate_label} keys mismatch")
        if report.get("rows") != expected_rows:
            failures.append(f"{gate_label}.rows mismatch")
        if report.get("finite") is not True:
            failures.append(f"{gate_label} is not finite")
        if report.get("tokens") != list(tokens):
            failures.append(f"{gate_label}.tokens mismatch")
        expected_thresholds = {
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
        }
        if report.get("thresholds") != expected_thresholds:
            failures.append(f"{gate_label}.thresholds mismatch")
        row_error = report.get("row_sum_max_abs_error")
        if (
            not _is_finite_number(row_error)
            or float(row_error) < 0.0
            or float(row_error)
            > SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
        ):
            failures.append(f"{gate_label}.row_sum_max_abs_error exceeds contract")
        entropy = report.get("entropy_mean")
        if not _is_finite_number(entropy) or float(entropy) < (
            SERVE_PARITY_SPECIALIST_GATE_MIN_ENTROPY
        ):
            failures.append(f"{gate_label}.entropy_mean below contract")
        for metric_name, threshold in (
            ("mean_weight", SERVE_PARITY_SPECIALIST_GATE_MIN_MEAN_WEIGHT),
            ("std_weight", SERVE_PARITY_SPECIALIST_GATE_MIN_STD),
            (
                "top_rank_count",
                SERVE_PARITY_SPECIALIST_GATE_MIN_TOP_RANK_COUNT,
            ),
        ):
            metrics = report.get(metric_name)
            if not isinstance(metrics, dict) or set(metrics) != set(tokens):
                failures.append(f"{gate_label}.{metric_name} set mismatch")
                continue
            for token, metric in metrics.items():
                if metric_name == "top_rank_count":
                    valid = (
                        not isinstance(metric, bool)
                        and isinstance(metric, int)
                        and metric >= int(threshold)
                    )
                else:
                    valid = _is_finite_number(metric) and float(metric) > float(
                        threshold
                    )
                if not valid:
                    failures.append(
                        f"{gate_label}.{metric_name}.{token} violates contract"
                    )
        mean_weight = report.get("mean_weight")
        if isinstance(mean_weight, dict) and all(
            _is_finite_number(item) for item in mean_weight.values()
        ):
            if abs(sum(float(item) for item in mean_weight.values()) - 1.0) > (
                SERVE_PARITY_SPECIALIST_GATE_ROW_SUM_MAX_ABS_ERROR
            ):
                failures.append(f"{gate_label}.mean_weight does not sum to one")
    feature_gate = value.get("family_tf_feature_gate")
    feature_label = f"{label}.family_tf_feature_gate"
    failures.extend(_zero_failure_pass(feature_gate, label=feature_label))
    if isinstance(feature_gate, dict):
        expected_feature_keys = {
            "decision",
            "failures",
            "rows",
            "finite",
            "tokens",
            "mean_weight",
            "std_weight",
            "min_observed",
            "max_observed",
            "thresholds",
        }
        if set(feature_gate) != expected_feature_keys:
            failures.append(f"{feature_label} keys mismatch")
        if feature_gate.get("rows") != expected_rows:
            failures.append(f"{feature_label}.rows mismatch")
        if feature_gate.get("finite") is not True:
            failures.append(f"{feature_label} is not finite")
        if feature_gate.get("tokens") != list(
            SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS
        ):
            failures.append(f"{feature_label}.tokens mismatch")
        expected_thresholds = {
            "min_weight_exclusive": SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE,
            "max_weight_exclusive": SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE,
            "min_std_exclusive": SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE,
        }
        if feature_gate.get("thresholds") != expected_thresholds:
            failures.append(f"{feature_label}.thresholds mismatch")
        for metric_name in (
            "mean_weight",
            "std_weight",
            "min_observed",
            "max_observed",
        ):
            metrics = feature_gate.get(metric_name)
            if not isinstance(metrics, dict) or set(metrics) != set(
                SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS
            ):
                failures.append(f"{feature_label}.{metric_name} set mismatch")
                continue
            for token, metric in metrics.items():
                if not _is_finite_number(metric):
                    failures.append(
                        f"{feature_label}.{metric_name}.{token} is non-finite"
                    )
                    continue
                numeric = float(metric)
                if (
                    metric_name == "std_weight"
                    and numeric <= SERVE_PARITY_FEATURE_GATE_MIN_STD_EXCLUSIVE
                ):
                    failures.append(
                        f"{feature_label}.{metric_name}.{token} violates contract"
                    )
                elif (
                    metric_name == "min_observed"
                    and numeric <= SERVE_PARITY_FEATURE_GATE_MIN_EXCLUSIVE
                ):
                    failures.append(
                        f"{feature_label}.{metric_name}.{token} violates contract"
                    )
                elif (
                    metric_name == "max_observed"
                    and numeric >= SERVE_PARITY_FEATURE_GATE_MAX_EXCLUSIVE
                ):
                    failures.append(
                        f"{feature_label}.{metric_name}.{token} violates contract"
                    )
    return failures


def _specialist_decision_influence_contract_failures(
    value: object,
) -> list[str]:
    label = "serve parity specialist_decision_influence"
    failures = _zero_failure_pass(value, label=label)
    if not isinstance(value, dict):
        return failures
    exact_keys = {
        "decision",
        "failures",
        "sample_count",
        "sampling_contract",
        "sample_positions",
        "sampled_test_coverage",
        "comparison_surface",
        "epsilon",
        "min_changed_rows",
        "specialists",
        "specialist_input_indices",
        "specialist_input_indices_sha256",
        "model_metadata_indices_exact_match",
        "model_buffer_indices_exact_match",
        "methods",
    }
    if set(value) != exact_keys:
        failures.append(f"{label} keys={sorted(value)} expected={sorted(exact_keys)}")
    if value.get("sample_count") != SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT:
        failures.append(f"{label}.sample_count mismatch")
    if (
        value.get("sampling_contract")
        != SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLING_CONTRACT
    ):
        failures.append(f"{label}.sampling_contract mismatch")
    if value.get("sample_positions") != list(
        SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_POSITIONS
    ):
        failures.append(f"{label}.sample_positions mismatch")
    failures.extend(
        time_coverage_contract_failures(
            value.get("sampled_test_coverage"),
            label=f"{label}.sampled_test_coverage",
            expected_rows=SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT,
        )
    )
    if (
        value.get("comparison_surface")
        != SERVE_PARITY_SPECIALIST_INFLUENCE_COMPARISON_SURFACE
    ):
        failures.append(f"{label}.comparison_surface mismatch")
    if not _is_exact_number(
        value.get("epsilon"), SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON
    ):
        failures.append(f"{label}.epsilon mismatch")
    if (
        value.get("min_changed_rows")
        != SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS
    ):
        failures.append(f"{label}.min_changed_rows mismatch")
    if value.get("specialists") != list(MODEL_NATIVE_REQUIRED_SPECIALISTS):
        failures.append(f"{label}.specialists mismatch")
    if value.get("model_metadata_indices_exact_match") is not True:
        failures.append(f"{label}.model_metadata_indices_exact_match is not true")
    if value.get("model_buffer_indices_exact_match") is not True:
        failures.append(f"{label}.model_buffer_indices_exact_match is not true")

    indices = value.get("specialist_input_indices")
    index_hashes: dict[str, str] = {}
    seen: set[int] = set()
    total_indices = 0
    if not isinstance(indices, dict) or set(indices) != set(
        MODEL_NATIVE_REQUIRED_SPECIALISTS
    ):
        failures.append(f"{label}.specialist_input_indices set mismatch")
    else:
        for specialist, raw in indices.items():
            valid = (
                isinstance(raw, list)
                and bool(raw)
                and all(
                    not isinstance(item, bool)
                    and isinstance(item, int)
                    and 0 <= item < MODEL_NATIVE_SIGNAL_DIM
                    for item in raw
                )
                and raw == sorted(raw)
                and len(raw) == len(set(raw))
                and not seen.intersection(raw)
            )
            if not valid:
                failures.append(f"{label}.specialist_input_indices.{specialist} invalid")
                continue
            seen.update(raw)
            total_indices += len(raw)
            index_hashes[specialist] = _canonical_sha256(raw)
        if total_indices != MODEL_NATIVE_SIGNAL_DIM:
            failures.append(
                f"{label}.specialist_input_indices total={total_indices} "
                f"expected={MODEL_NATIVE_SIGNAL_DIM}"
            )
        try:
            observed_indices_sha = _canonical_sha256(indices)
        except (TypeError, ValueError):
            observed_indices_sha = ""
        if value.get("specialist_input_indices_sha256") != observed_indices_sha:
            failures.append(f"{label}.specialist_input_indices_sha256 mismatch")

    methods = value.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(
        SERVE_PARITY_SPECIALIST_INFLUENCE_METHODS
    ):
        failures.append(f"{label}.methods set mismatch")
        return failures
    expected_surfaces = {
        "input_family_mask": "seq_and_snap_exact_specialist_input_indices",
        "encoder_output_hook_ablation": "specialist_encoder_output_zero_hook",
    }
    for method in SERVE_PARITY_SPECIALIST_INFLUENCE_METHODS:
        method_label = f"{label}.methods.{method}"
        row = methods.get(method)
        failures.extend(_zero_failure_pass(row, label=method_label))
        if not isinstance(row, dict):
            continue
        if set(row) != {
            "decision",
            "failures",
            "ablation_surface",
            "specialists",
        }:
            failures.append(f"{method_label} keys mismatch")
        if row.get("ablation_surface") != expected_surfaces[method]:
            failures.append(f"{method_label}.ablation_surface mismatch")
        specialist_rows = row.get("specialists")
        if not isinstance(specialist_rows, dict) or set(specialist_rows) != set(
            MODEL_NATIVE_REQUIRED_SPECIALISTS
        ):
            failures.append(f"{method_label}.specialists set mismatch")
            continue
        for specialist, metric in specialist_rows.items():
            metric_label = f"{method_label}.specialists.{specialist}"
            failures.extend(_zero_failure_pass(metric, label=metric_label))
            if not isinstance(metric, dict):
                continue
            if set(metric) != {
                "decision",
                "failures",
                "target",
                "input_indices_sha256",
                "max_abs_class_centered_raw_logit_delta",
                "raw_changed_rows",
                "max_abs_class_centered_logit_delta",
                "changed_rows",
                "total_rows",
            }:
                failures.append(f"{metric_label} keys mismatch")
            expected_target = (
                f"signal_indices:{specialist}"
                if method == "input_family_mask"
                else f"model.specialist_encoder.{specialist}"
            )
            if metric.get("target") != expected_target:
                failures.append(f"{metric_label}.target mismatch")
            expected_hash = index_hashes.get(specialist)
            if not expected_hash or metric.get("input_indices_sha256") != expected_hash:
                failures.append(f"{metric_label}.input_indices_sha256 mismatch")
            delta = metric.get("max_abs_class_centered_logit_delta")
            if not _is_finite_number(delta) or float(delta) <= (
                SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON
            ):
                failures.append(f"{metric_label} lacks >epsilon influence")
            raw_delta = metric.get("max_abs_class_centered_raw_logit_delta")
            if not _is_finite_number(raw_delta) or float(raw_delta) <= (
                SERVE_PARITY_SPECIALIST_INFLUENCE_EPSILON
            ):
                failures.append(f"{metric_label} lacks >epsilon raw influence")
            if metric.get("total_rows") != (
                SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT
            ):
                failures.append(f"{metric_label}.total_rows mismatch")
            changed = metric.get("changed_rows")
            if (
                isinstance(changed, bool)
                or not isinstance(changed, int)
                or changed < SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS
                or changed > SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT
            ):
                failures.append(f"{metric_label}.changed_rows violates contract")
            raw_changed = metric.get("raw_changed_rows")
            if (
                isinstance(raw_changed, bool)
                or not isinstance(raw_changed, int)
                or raw_changed < SERVE_PARITY_SPECIALIST_INFLUENCE_MIN_CHANGED_ROWS
                or raw_changed > SERVE_PARITY_SPECIALIST_INFLUENCE_SAMPLE_COUNT
            ):
                failures.append(f"{metric_label}.raw_changed_rows violates contract")
    return failures


def _masked_input_influence_contract_failures(
    value: object,
    *,
    label: str,
    sample_count: int,
    sample_positions: tuple[int, ...],
    sampling_contract: str,
    comparison_surface: str,
    epsilon: float,
    min_changed_rows: int,
    names: tuple[str, ...],
    names_field: str,
    expected_targets: Mapping[str, str],
    report_ablation: str | None,
    metric_ablation_surface: str,
) -> list[str]:
    failures = _zero_failure_pass(value, label=label)
    if not isinstance(value, dict):
        return failures
    exact_keys = {
        "decision",
        "failures",
        "sample_count",
        "sampling_contract",
        "sample_positions",
        "sampled_test_coverage",
        "comparison_surface",
        "epsilon",
        "min_changed_rows",
        names_field,
        "metrics",
    }
    if report_ablation is not None:
        exact_keys.add("ablation")
    if set(value) != exact_keys:
        failures.append(f"{label} keys mismatch")
    if value.get("sample_count") != sample_count:
        failures.append(f"{label}.sample_count mismatch")
    if value.get("sampling_contract") != sampling_contract:
        failures.append(f"{label}.sampling_contract mismatch")
    if value.get("sample_positions") != list(sample_positions):
        failures.append(f"{label}.sample_positions mismatch")
    failures.extend(
        time_coverage_contract_failures(
            value.get("sampled_test_coverage"),
            label=f"{label}.sampled_test_coverage",
            expected_rows=sample_count,
        )
    )
    if value.get("comparison_surface") != comparison_surface:
        failures.append(f"{label}.comparison_surface mismatch")
    if not _is_exact_number(value.get("epsilon"), epsilon):
        failures.append(f"{label}.epsilon mismatch")
    if value.get("min_changed_rows") != min_changed_rows:
        failures.append(f"{label}.min_changed_rows mismatch")
    if value.get(names_field) != list(names):
        failures.append(f"{label}.{names_field} mismatch")
    if report_ablation is not None and value.get("ablation") != report_ablation:
        failures.append(f"{label}.ablation mismatch")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(names):
        failures.append(f"{label}.metrics set mismatch")
        return failures
    metric_keys = {
        "decision",
        "failures",
        "target",
        "ablation_surface",
        "max_abs_class_centered_raw_logit_delta",
        "raw_changed_rows",
        "max_abs_class_centered_logit_delta",
        "changed_rows",
        "total_rows",
    }
    for name in names:
        metric_label = f"{label}.metrics.{name}"
        metric = metrics.get(name)
        failures.extend(_zero_failure_pass(metric, label=metric_label))
        if not isinstance(metric, dict):
            continue
        if set(metric) != metric_keys:
            failures.append(f"{metric_label} keys mismatch")
        if metric.get("target") != expected_targets[name]:
            failures.append(f"{metric_label}.target mismatch")
        if metric.get("ablation_surface") != metric_ablation_surface:
            failures.append(f"{metric_label}.ablation_surface mismatch")
        delta = metric.get("max_abs_class_centered_raw_logit_delta")
        if not _is_finite_number(delta) or float(delta) <= epsilon:
            failures.append(f"{metric_label} lacks >epsilon raw influence")
        raw_changed = metric.get("raw_changed_rows")
        if (
            isinstance(raw_changed, bool)
            or not isinstance(raw_changed, int)
            or raw_changed < min_changed_rows
            or raw_changed > sample_count
        ):
            failures.append(f"{metric_label}.raw_changed_rows violates contract")
        final_delta = metric.get("max_abs_class_centered_logit_delta")
        if not _is_finite_number(final_delta) or float(final_delta) <= epsilon:
            failures.append(f"{metric_label} lacks >epsilon final influence")
        changed = metric.get("changed_rows")
        if (
            isinstance(changed, bool)
            or not isinstance(changed, int)
            or changed < min_changed_rows
            or changed > sample_count
        ):
            failures.append(f"{metric_label}.changed_rows violates contract")
        if metric.get("total_rows") != sample_count:
            failures.append(f"{metric_label}.total_rows mismatch")
    return failures


def _multi_tf_decision_influence_contract_failures(value: object) -> list[str]:
    return _masked_input_influence_contract_failures(
        value,
        label="serve parity multi_tf_decision_influence",
        sample_count=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        sample_positions=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS,
        sampling_contract=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
        comparison_surface=SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
        epsilon=SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
        min_changed_rows=SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
        names=SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES,
        names_field="timeframes",
        expected_targets={
            timeframe: f"model.input.seq_{timeframe.lower()}"
            for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
        },
        report_ablation="candidate_specific_full_tensor_zero_ablation_v1",
        metric_ablation_surface="full_tensor_zero_mask",
    )


def _family_tf_decision_influence_contract_failures(
    value: object,
) -> list[str]:
    routing = require_multi_tf_specialist_routing_v4(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    tokens = tuple(SERVE_PARITY_FAMILY_TF_COOPERATION_TOKENS)
    return _masked_input_influence_contract_failures(
        value,
        label="serve parity family_tf_decision_influence",
        sample_count=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_COUNT,
        sample_positions=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLE_POSITIONS,
        sampling_contract=SERVE_PARITY_MULTI_TF_INFLUENCE_SAMPLING_CONTRACT,
        comparison_surface=SERVE_PARITY_MULTI_TF_INFLUENCE_COMPARISON_SURFACE,
        epsilon=SERVE_PARITY_MULTI_TF_INFLUENCE_EPSILON,
        min_changed_rows=SERVE_PARITY_MULTI_TF_INFLUENCE_MIN_CHANGED_ROWS,
        names=tokens,
        names_field="family_timeframe_tokens",
        expected_targets={
            f"{timeframe.lower()}:{specialist}": (
                f"model.input.seq_{timeframe.lower()}["
                f"{','.join(str(index) for index in indices)}]"
            )
            for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
            for specialist, indices in routing.items()
        },
        report_ablation=(
            "candidate_specific_family_tensor_index_zero_ablation_v1"
        ),
        metric_ablation_surface="exact_family_feature_indices_zero_mask",
    )


def _upstream_context_decision_influence_contract_failures(
    value: object,
) -> list[str]:
    return _masked_input_influence_contract_failures(
        value,
        label="serve parity upstream_context_decision_influence",
        sample_count=SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_COUNT,
        sample_positions=SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLE_POSITIONS,
        sampling_contract=SERVE_PARITY_UPSTREAM_INFLUENCE_SAMPLING_CONTRACT,
        comparison_surface=SERVE_PARITY_UPSTREAM_INFLUENCE_COMPARISON_SURFACE,
        epsilon=SERVE_PARITY_UPSTREAM_INFLUENCE_EPSILON,
        min_changed_rows=SERVE_PARITY_UPSTREAM_INFLUENCE_MIN_CHANGED_ROWS,
        names=SERVE_PARITY_UPSTREAM_INFLUENCE_METHODS,
        names_field="methods",
        expected_targets={
            "ctx_cont_zero_mask": "model.input.ctx_cont",
            "ctx_cat_zero_mask": "model.input.ctx_cat",
        },
        report_ablation=None,
        metric_ablation_surface="full_tensor_zero_mask",
    )


def _individual_input_decision_influence_contract_failures(
    value: object,
) -> list[str]:
    label = "serve parity individual_input_decision_influence"
    failures = _zero_failure_pass(value, label=label)
    if not isinstance(value, dict):
        return failures
    expected_keys = {
        "decision",
        "failures",
        "sample_count",
        "sample_positions",
        "sampled_test_coverage",
        "comparison_surface",
        "gradient_epsilon",
        "categorical_delta_epsilon",
        "numeric_input_count",
        "categorical_input_count",
        "signal_names_sha256",
        "ctx_cont_names_sha256",
        "ctx_cat_names_sha256",
        "numeric",
        "categorical",
    }
    if set(value) != expected_keys:
        failures.append(f"{label} keys mismatch")
    if value.get("sample_count") != SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT:
        failures.append(f"{label}.sample_count mismatch")
    if value.get("sample_positions") != list(
        SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_POSITIONS
    ):
        failures.append(f"{label}.sample_positions mismatch")
    failures.extend(
        time_coverage_contract_failures(
            value.get("sampled_test_coverage"),
            label=f"{label}.sampled_test_coverage",
            expected_rows=SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT,
        )
    )
    if (
        value.get("comparison_surface")
        != SERVE_PARITY_INDIVIDUAL_INPUT_COMPARISON_SURFACE
    ):
        failures.append(f"{label}.comparison_surface mismatch")
    if not _is_exact_number(
        value.get("gradient_epsilon"),
        SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON,
    ):
        failures.append(f"{label}.gradient_epsilon mismatch")
    if not _is_exact_number(
        value.get("categorical_delta_epsilon"),
        SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON,
    ):
        failures.append(f"{label}.categorical_delta_epsilon mismatch")

    numeric = value.get("numeric")
    signal_row = numeric.get("seq_signal") if isinstance(numeric, dict) else None
    raw_signal_names = (
        signal_row.get("tokens") if isinstance(signal_row, dict) else None
    )
    signal_names = (
        list(raw_signal_names) if isinstance(raw_signal_names, list) else []
    )
    if (
        len(signal_names) != MODEL_NATIVE_SIGNAL_DIM
        or len(signal_names) != len(set(signal_names))
        or not all(isinstance(item, str) and item for item in signal_names)
    ):
        failures.append(f"{label} signal tokens are not exact unique seq513")
    ctx_cont_names = list(MODEL_NATIVE_CTX_CONT_FIELDS)
    ctx_cat_names = list(MODEL_NATIVE_CTX_CAT_FIELDS)
    expected_numeric_tokens = {
        "seq_signal": signal_names,
        "snap_signal": signal_names,
        "ctx_cont": ctx_cont_names,
        **{
            f"seq_{timeframe.lower()}": [
                f"{timeframe.lower()}:{feature}"
                for feature in MULTI_TF_PER_BAR_FEATURES_V4
            ]
            for timeframe in SERVE_PARITY_MULTI_TF_INFLUENCE_TIMEFRAMES
        },
    }
    expected_numeric_count = (
        (2 * MODEL_NATIVE_SIGNAL_DIM)
        + len(MODEL_NATIVE_CTX_CONT_FIELDS)
        + len(SERVE_PARITY_FAMILY_TF_FEATURE_TOKENS)
    )
    if value.get("numeric_input_count") != expected_numeric_count:
        failures.append(f"{label}.numeric_input_count mismatch")
    if value.get("categorical_input_count") != len(ctx_cat_names):
        failures.append(f"{label}.categorical_input_count mismatch")
    for field, names in (
        ("signal_names_sha256", signal_names),
        ("ctx_cont_names_sha256", ctx_cont_names),
        ("ctx_cat_names_sha256", ctx_cat_names),
    ):
        if value.get(field) != _canonical_sha256(names):
            failures.append(f"{label}.{field} mismatch")

    if not isinstance(numeric, dict) or set(numeric) != set(
        expected_numeric_tokens
    ):
        failures.append(f"{label}.numeric surface set mismatch")
    else:
        for surface, tokens in expected_numeric_tokens.items():
            surface_label = f"{label}.numeric.{surface}"
            row = numeric.get(surface)
            if not isinstance(row, dict) or set(row) != {"tokens", "metrics"}:
                failures.append(f"{surface_label} keys mismatch")
                continue
            if row.get("tokens") != tokens:
                failures.append(f"{surface_label}.tokens mismatch")
            metrics = row.get("metrics")
            if not isinstance(metrics, dict) or set(metrics) != set(tokens):
                failures.append(f"{surface_label}.metrics set mismatch")
                continue
            for token in tokens:
                metric_label = f"{surface_label}.metrics.{token}"
                metric = metrics[token]
                failures.extend(_zero_failure_pass(metric, label=metric_label))
                if not isinstance(metric, dict) or set(metric) != {
                    "decision",
                    "failures",
                    "max_abs_raw_class_margin_gradient",
                    "max_abs_final_class_margin_gradient",
                }:
                    failures.append(f"{metric_label} keys mismatch")
                    continue
                for field in (
                    "max_abs_raw_class_margin_gradient",
                    "max_abs_final_class_margin_gradient",
                ):
                    observed = metric.get(field)
                    if (
                        not _is_finite_number(observed)
                        or float(observed)
                        <= SERVE_PARITY_INDIVIDUAL_INPUT_GRADIENT_EPSILON
                    ):
                        failures.append(f"{metric_label}.{field} is dead")

    categorical = value.get("categorical")
    if not isinstance(categorical, dict) or set(categorical) != set(ctx_cat_names):
        failures.append(f"{label}.categorical set mismatch")
    else:
        exact_metric_keys = {
            "decision",
            "failures",
            "counterfactual",
            "max_abs_class_centered_raw_logit_delta",
            "raw_changed_rows",
            "max_abs_class_centered_logit_delta",
            "changed_rows",
            "total_rows",
        }
        for name in ctx_cat_names:
            metric_label = f"{label}.categorical.{name}"
            metric = categorical[name]
            failures.extend(_zero_failure_pass(metric, label=metric_label))
            if not isinstance(metric, dict) or set(metric) != exact_metric_keys:
                failures.append(f"{metric_label} keys mismatch")
                continue
            if (
                metric.get("counterfactual")
                != "next_valid_embedding_category_modulo_domain"
            ):
                failures.append(f"{metric_label}.counterfactual mismatch")
            for field in (
                "max_abs_class_centered_raw_logit_delta",
                "max_abs_class_centered_logit_delta",
            ):
                observed = metric.get(field)
                if (
                    not _is_finite_number(observed)
                    or float(observed)
                    <= SERVE_PARITY_INDIVIDUAL_INPUT_CAT_DELTA_EPSILON
                ):
                    failures.append(f"{metric_label}.{field} is dead")
            for field in ("raw_changed_rows", "changed_rows"):
                observed = metric.get(field)
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, int)
                    or observed < 1
                    or observed > SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT
                ):
                    failures.append(f"{metric_label}.{field} invalid")
            if metric.get("total_rows") != (
                SERVE_PARITY_INDIVIDUAL_INPUT_SAMPLE_COUNT
            ):
                failures.append(f"{metric_label}.total_rows mismatch")
    return failures


def _direction_evidence_fusion_influence_contract_failures(
    value: object,
    *,
    bundle_dir: object,
    prediction_evidence: object,
) -> list[str]:
    label = "serve parity direction_evidence_fusion_influence"
    failures = _zero_failure_pass(value, label=label)
    if not isinstance(value, dict):
        return failures
    exact_keys = {
        "decision",
        "failures",
        "sample_count",
        "sampling_contract",
        "sample_positions",
        "sampled_test_coverage",
        "comparison_surface",
        "epsilon",
        "fusion_input_gradient_epsilon",
        "min_changed_rows",
        "ablation",
        "fusion_metadata",
        "ordered_input_layout",
        "inputs_sha256",
        "input_dim",
        "bundle_metadata_path",
        "bundle_metadata_sha256",
        "bundle_metadata_exact_match",
        "master_transformer_lock_path",
        "master_transformer_lock_sha256",
        "master_transformer_lock_exact_match",
        "reference",
        "groups",
    }
    if set(value) != exact_keys:
        failures.append(f"{label} keys mismatch")
    if value.get("sample_count") != SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT:
        failures.append(f"{label}.sample_count mismatch")
    if (
        value.get("sampling_contract")
        != SERVE_PARITY_FUSION_INFLUENCE_SAMPLING_CONTRACT
    ):
        failures.append(f"{label}.sampling_contract mismatch")
    if value.get("sample_positions") != list(
        SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_POSITIONS
    ):
        failures.append(f"{label}.sample_positions mismatch")
    failures.extend(
        time_coverage_contract_failures(
            value.get("sampled_test_coverage"),
            label=f"{label}.sampled_test_coverage",
            expected_rows=SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT,
        )
    )
    if (
        value.get("comparison_surface")
        != SERVE_PARITY_FUSION_INFLUENCE_COMPARISON_SURFACE
    ):
        failures.append(f"{label}.comparison_surface mismatch")
    if not _is_exact_number(
        value.get("epsilon"), SERVE_PARITY_FUSION_INFLUENCE_EPSILON
    ):
        failures.append(f"{label}.epsilon mismatch")
    if not _is_exact_number(
        value.get("fusion_input_gradient_epsilon"),
        SERVE_PARITY_FUSION_INPUT_GRADIENT_EPSILON,
    ):
        failures.append(f"{label}.fusion_input_gradient_epsilon mismatch")
    if (
        value.get("min_changed_rows")
        != SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS
    ):
        failures.append(f"{label}.min_changed_rows mismatch")
    if (
        value.get("ablation")
        != SERVE_PARITY_FUSION_INFLUENCE_ABLATION
    ):
        failures.append(f"{label}.ablation mismatch")
    expected_metadata = direction_evidence_fusion_metadata()
    if value.get("fusion_metadata") != expected_metadata:
        failures.append(f"{label}.fusion_metadata mismatch")
    if value.get("ordered_input_layout") != DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT:
        failures.append(f"{label}.ordered_input_layout mismatch")
    if value.get("inputs_sha256") != DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256:
        failures.append(f"{label}.inputs_sha256 mismatch")
    if value.get("input_dim") != DIRECTION_EVIDENCE_FUSION_INPUT_DIM:
        failures.append(f"{label}.input_dim mismatch")
    if value.get("bundle_metadata_exact_match") is not True:
        failures.append(f"{label}.bundle_metadata_exact_match is not true")
    if value.get("master_transformer_lock_exact_match") is not True:
        failures.append(f"{label}.master_transformer_lock_exact_match is not true")

    resolved_bundle = Path(str(bundle_dir or "")).expanduser()
    expected_metadata_path = resolved_bundle / "bundle_metadata.json"
    expected_lock_path = resolved_bundle / "MASTER_TRANSFORMER_LOCK.json"
    for field, expected_path in (
        ("bundle_metadata_path", expected_metadata_path),
        ("master_transformer_lock_path", expected_lock_path),
    ):
        raw_path = Path(str(value.get(field) or "")).expanduser()
        if (
            not resolved_bundle.is_absolute()
            or not raw_path.is_absolute()
            or raw_path != expected_path
        ):
            failures.append(f"{label}.{field} bundle binding mismatch")
    metadata_sha = value.get("bundle_metadata_sha256")
    lock_sha = value.get("master_transformer_lock_sha256")
    if not _is_sha256(metadata_sha):
        failures.append(f"{label}.bundle_metadata_sha256 invalid")
    if not _is_sha256(lock_sha):
        failures.append(f"{label}.master_transformer_lock_sha256 invalid")
    if (
        not isinstance(prediction_evidence, dict)
        or prediction_evidence.get("bundle_metadata_sha256") != metadata_sha
        or prediction_evidence.get("bundle_metadata_path")
        != value.get("bundle_metadata_path")
    ):
        failures.append(f"{label} prediction/bundle metadata binding mismatch")

    reference = value.get("reference")
    reference_label = f"{label}.reference"
    expected_reference_keys = {
        "split",
        "aggregation",
        "coverage",
        "input_dim",
        "inputs_sha256",
        "derived_relation",
        "mean_by_input",
        "ordered_mean_sha256",
    }
    means: dict[str, list[float]] = {}
    if not isinstance(reference, dict) or set(reference) != expected_reference_keys:
        failures.append(f"{reference_label} keys mismatch")
    else:
        if reference.get("split") != SERVE_PARITY_FUSION_REFERENCE_SPLIT:
            failures.append(f"{reference_label}.split mismatch")
        if (
            reference.get("aggregation")
            != SERVE_PARITY_FUSION_REFERENCE_AGGREGATION
        ):
            failures.append(f"{reference_label}.aggregation mismatch")
        failures.extend(
            time_coverage_contract_failures(
                reference.get("coverage"), label=f"{reference_label}.coverage"
            )
        )
        if reference.get("input_dim") != DIRECTION_EVIDENCE_FUSION_INPUT_DIM:
            failures.append(f"{reference_label}.input_dim mismatch")
        if reference.get("inputs_sha256") != DIRECTION_EVIDENCE_FUSION_INPUTS_SHA256:
            failures.append(f"{reference_label}.inputs_sha256 mismatch")
        relation = reference.get("derived_relation")
        if not isinstance(relation, dict) or set(relation) != {
            "equation",
            "max_abs_error",
            "atol",
        }:
            failures.append(f"{reference_label}.derived_relation keys mismatch")
        else:
            if relation.get("equation") != (
                "action_advantage=action_value-expectile_value_by_horizon"
            ):
                failures.append(
                    f"{reference_label}.derived_relation.equation mismatch"
                )
            if not _is_exact_number(
                relation.get("atol"),
                SERVE_PARITY_FUSION_DERIVED_RELATION_ATOL,
            ):
                failures.append(
                    f"{reference_label}.derived_relation.atol mismatch"
                )
            relation_error = relation.get("max_abs_error")
            if (
                not _is_finite_number(relation_error)
                or float(relation_error)
                > SERVE_PARITY_FUSION_DERIVED_RELATION_ATOL
            ):
                failures.append(
                    f"{reference_label}.derived_relation exceeds tolerance"
                )
        raw_means = reference.get("mean_by_input")
        if not isinstance(raw_means, dict) or set(raw_means) != {
            name for name, _width in DIRECTION_EVIDENCE_FUSION_INPUTS
        }:
            failures.append(f"{reference_label}.mean_by_input set mismatch")
        else:
            ordered_means: list[float] = []
            for name, width in DIRECTION_EVIDENCE_FUSION_INPUTS:
                mean = raw_means.get(name)
                if (
                    not isinstance(mean, list)
                    or len(mean) != width
                    or not all(_is_finite_number(item) for item in mean)
                ):
                    failures.append(f"{reference_label}.mean_by_input.{name} invalid")
                    continue
                means[name] = [float(item) for item in mean]
                ordered_means.extend(means[name])
            if len(ordered_means) == DIRECTION_EVIDENCE_FUSION_INPUT_DIM:
                if reference.get("ordered_mean_sha256") != _canonical_sha256(
                    ordered_means
                ):
                    failures.append(f"{reference_label}.ordered_mean_sha256 mismatch")

    groups = value.get("groups")
    if not isinstance(groups, dict) or set(groups) != {
        name for name, _width in DIRECTION_EVIDENCE_FUSION_INPUTS
    }:
        failures.append(f"{label}.groups set mismatch")
        return failures
    metric_keys = {
        "decision",
        "failures",
        "target",
        "ablation_surface",
        "start",
        "stop",
        "width",
        "reference_inputs",
        "reference_values_sha256",
        "max_abs_raw_class_margin_input_gradient",
        "max_abs_final_class_margin_input_gradient",
        "max_abs_class_centered_raw_logit_delta",
        "raw_changed_rows",
        "max_abs_class_centered_logit_delta",
        "changed_rows",
        "total_rows",
    }
    for layout in DIRECTION_EVIDENCE_FUSION_ORDERED_INPUT_LAYOUT:
        name = str(layout["name"])
        metric_label = f"{label}.groups.{name}"
        metric = groups.get(name)
        failures.extend(_zero_failure_pass(metric, label=metric_label))
        if not isinstance(metric, dict):
            continue
        if set(metric) != metric_keys:
            failures.append(f"{metric_label} keys mismatch")
        for field in ("start", "stop", "width"):
            if metric.get(field) != layout[field]:
                failures.append(f"{metric_label}.{field} mismatch")
        if name == "action_value":
            expected_target = (
                "model.evidence_fusion_norm.input["
                "action_value+action_advantage]"
            )
        elif name == "expectile_value":
            expected_target = (
                "model.evidence_fusion_norm.input["
                "expectile_value+action_advantage]"
            )
        elif name == "action_advantage":
            expected_target = (
                "model.evidence_fusion_norm.input["
                "action_value+expectile_value+action_advantage]"
            )
        else:
            expected_target = (
                f"model.evidence_fusion_norm.input["
                f"{layout['start']}:{layout['stop']}]"
            )
        if metric.get("target") != expected_target:
            failures.append(f"{metric_label}.target mismatch")
        expected_surface = SERVE_PARITY_FUSION_DERIVED_ABLATION_SURFACES.get(
            name, "exact_fusion_slice_val_mean_replacement"
        )
        if metric.get("ablation_surface") != expected_surface:
            failures.append(f"{metric_label}.ablation_surface mismatch")
        expected_reference_inputs = list(
            SERVE_PARITY_FUSION_DERIVED_REFERENCE_INPUTS.get(name, (name,))
        )
        if metric.get("reference_inputs") != expected_reference_inputs:
            failures.append(f"{metric_label}.reference_inputs mismatch")
        reference_values = [
            item
            for reference_name in expected_reference_inputs
            for item in means.get(reference_name, ())
        ]
        if (
            len(reference_values)
            == sum(
                dict(DIRECTION_EVIDENCE_FUSION_INPUTS)[reference_name]
                for reference_name in expected_reference_inputs
            )
            and metric.get("reference_values_sha256")
            != _canonical_sha256(reference_values)
        ):
            failures.append(f"{metric_label}.reference_values_sha256 mismatch")
        for field in (
            "max_abs_raw_class_margin_input_gradient",
            "max_abs_final_class_margin_input_gradient",
        ):
            gradient = metric.get(field)
            if (
                not _is_finite_number(gradient)
                or float(gradient)
                <= SERVE_PARITY_FUSION_INPUT_GRADIENT_EPSILON
            ):
                failures.append(f"{metric_label}.{field} is dead")
        raw_delta = metric.get("max_abs_class_centered_raw_logit_delta")
        if (
            not _is_finite_number(raw_delta)
            or float(raw_delta) <= SERVE_PARITY_FUSION_INFLUENCE_EPSILON
        ):
            failures.append(f"{metric_label} lacks >epsilon raw influence")
        raw_changed = metric.get("raw_changed_rows")
        if (
            isinstance(raw_changed, bool)
            or not isinstance(raw_changed, int)
            or raw_changed < SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS
            or raw_changed > SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT
        ):
            failures.append(f"{metric_label}.raw_changed_rows violates contract")
        delta = metric.get("max_abs_class_centered_logit_delta")
        if (
            not _is_finite_number(delta)
            or float(delta) <= SERVE_PARITY_FUSION_INFLUENCE_EPSILON
        ):
            failures.append(f"{metric_label} lacks >epsilon final influence")
        changed = metric.get("changed_rows")
        if (
            isinstance(changed, bool)
            or not isinstance(changed, int)
            or changed < SERVE_PARITY_FUSION_INFLUENCE_MIN_CHANGED_ROWS
            or changed > SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT
        ):
            failures.append(f"{metric_label}.changed_rows violates contract")
        if metric.get("total_rows") != SERVE_PARITY_FUSION_INFLUENCE_SAMPLE_COUNT:
            failures.append(f"{metric_label}.total_rows mismatch")
    return failures


def time_coverage_contract_failures(
    value: object,
    *,
    label: str,
    expected_rows: int | None = None,
) -> list[str]:
    failures: list[str] = []
    if not isinstance(value, dict):
        return [f"{label} must be an exact time-coverage object"]
    if set(value) != _TIME_COVERAGE_KEYS:
        failures.append(
            f"{label} keys={sorted(value)} expected={sorted(_TIME_COVERAGE_KEYS)}"
        )
    if value.get("schema_version") != UTC_TIME_COVERAGE_SCHEMA_VERSION:
        failures.append(f"{label} schema_version mismatch")
    rows = value.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        failures.append(f"{label} rows must be a positive exact integer")
    elif expected_rows is not None and rows != expected_rows:
        failures.append(f"{label} rows={rows} expected={expected_rows}")
    for field in ("first_utc", "last_utc"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            failures.append(f"{label} {field} must be a non-empty UTC timestamp")
    if not _is_sha256(value.get("utc_ns_sha256")):
        failures.append(f"{label} utc_ns_sha256 is not an exact SHA-256")
    return failures


def serve_gate_event_contract_failures(
    payload: Mapping[str, Any],
    *,
    evidence_name: str,
) -> list[str]:
    """Return every semantic admission mismatch for one immutable gate event."""

    failures: list[str] = []
    if payload.get("contract_version") != MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION:
        failures.append(f"{evidence_name} contract_version mismatch")
    if payload.get("decision") != "PASS" or payload.get("failures") != []:
        failures.append(f"{evidence_name} must be a zero-failure PASS")
    if payload.get("split") != MODEL_NATIVE_REQUIRED_TEST_SPLIT:
        failures.append(
            f"{evidence_name} split={payload.get('split')!r} expected='test'"
        )
    if payload.get("model_name") != MODEL_NATIVE_REQUIRED_MODEL_NAME:
        failures.append(
            f"{evidence_name} model_name={payload.get('model_name')!r} "
            "expected='candidate'"
        )
    dataset_parquet = payload.get("dataset_parquet")
    if not isinstance(dataset_parquet, str) or not dataset_parquet.startswith("/"):
        failures.append(f"{evidence_name} dataset_parquet must be absolute")
    if not _is_sha256(payload.get("dataset_parquet_sha256")):
        failures.append(
            f"{evidence_name} dataset_parquet_sha256 is not an exact SHA-256"
        )
    prediction = payload.get("prediction_evidence")
    if not isinstance(prediction, dict):
        failures.append(f"{evidence_name} prediction_evidence is missing")
    else:
        if (
            prediction.get("schema_version")
            != RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION
        ):
            failures.append(f"{evidence_name} prediction evidence schema mismatch")
        if prediction.get("runtime_head_evidence_authoritative") is not True:
            failures.append(
                f"{evidence_name} runtime-head prediction evidence is not authoritative"
            )
        if prediction.get("authoritative") is not True:
            failures.append(f"{evidence_name} prediction evidence is not authoritative")
        prediction_path = prediction.get("path")
        if not isinstance(prediction_path, str) or not prediction_path.startswith("/"):
            failures.append(f"{evidence_name} prediction evidence path must be absolute")
        if not _is_sha256(prediction.get("sha256")):
            failures.append(f"{evidence_name} prediction evidence SHA-256 is invalid")
    report_evidence = payload.get("prediction_report_evidence")
    if not isinstance(report_evidence, dict) or set(report_evidence) != {
        "json_path",
        "sha256",
    }:
        failures.append(f"{evidence_name} prediction_report_evidence is incomplete")
    else:
        report_path = report_evidence.get("json_path")
        if not isinstance(report_path, str) or not report_path.startswith("/"):
            failures.append(f"{evidence_name} prediction report path must be absolute")
        if not _is_sha256(report_evidence.get("sha256")):
            failures.append(f"{evidence_name} prediction report SHA-256 is invalid")

    coverage = payload.get("test_coverage")
    if not isinstance(coverage, dict) or set(coverage) != {
        "dataset",
        "predictions",
        "exact_match",
    }:
        failures.append(f"{evidence_name} test_coverage contract is incomplete")
    else:
        failures.extend(
            time_coverage_contract_failures(
                coverage.get("dataset"), label=f"{evidence_name}.test_coverage.dataset"
            )
        )
        failures.extend(
            time_coverage_contract_failures(
                coverage.get("predictions"),
                label=f"{evidence_name}.test_coverage.predictions",
            )
        )
        if coverage.get("exact_match") is not True:
            failures.append(f"{evidence_name} test coverage is not declared exact")
        if coverage.get("dataset") != coverage.get("predictions"):
            failures.append(
                f"{evidence_name} prediction/test time coverage is not exactly equal"
            )

    if evidence_name == "model_native_serve_parity":
        if payload.get("schema_version") != MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION:
            failures.append("serve parity schema_version mismatch")
        if payload.get("n_bars") != SERVE_PARITY_SAMPLE_COUNT:
            failures.append(
                f"serve parity n_bars={payload.get('n_bars')!r} "
                f"expected={SERVE_PARITY_SAMPLE_COUNT}"
            )
        if payload.get("sampling_contract") != SERVE_PARITY_SAMPLING_CONTRACT:
            failures.append("serve parity sampling_contract mismatch")
        if not _is_exact_number(payload.get("state_tol"), SERVE_PARITY_STATE_TOL):
            failures.append("serve parity state_tol mismatch")
        if not _is_exact_number(payload.get("forward_tol"), SERVE_PARITY_FORWARD_TOL):
            failures.append("serve parity forward_tol mismatch")
        if payload.get("env_pins") != SERVE_PARITY_ENV_PINS:
            failures.append("serve parity env_pins mismatch")
        failures.extend(
            serve_source_identity_contract_failures(
                payload.get("serve_source_identity")
            )
        )
        operating_point = payload.get("operating_point")
        if (
            not isinstance(operating_point, dict)
            or set(operating_point) != {"selection_score", "max_trades"}
            or operating_point.get("selection_score")
            != "model_direction_argmax"
            or isinstance(operating_point.get("max_trades"), bool)
            or not isinstance(operating_point.get("max_trades"), int)
            or int(operating_point["max_trades"]) <= 0
        ):
            failures.append("serve parity operating_point contract mismatch")
        if payload.get("runtime_device") != "cpu":
            failures.append("serve parity runtime_device must be exactly 'cpu'")
        failures.extend(
            time_coverage_contract_failures(
                payload.get("sampled_test_coverage"),
                label="serve parity sampled_test_coverage",
                expected_rows=SERVE_PARITY_SAMPLE_COUNT,
            )
        )
        state = payload.get("state_parity")
        if not isinstance(state, dict):
            failures.append("serve parity state_parity is missing")
        else:
            if state.get("n_compared") != SERVE_PARITY_SAMPLE_COUNT:
                failures.append("serve parity state_parity.n_compared mismatch")
            if not _is_exact_number(state.get("tolerance"), SERVE_PARITY_STATE_TOL):
                failures.append("serve parity state_parity.tolerance mismatch")
        forward = payload.get("forward_parity")
        if not isinstance(forward, dict):
            failures.append("serve parity forward_parity is missing")
        else:
            if forward.get("n_compared") != SERVE_PARITY_SAMPLE_COUNT:
                failures.append("serve parity forward_parity.n_compared mismatch")
            if not _is_exact_number(
                forward.get("tolerance"), SERVE_PARITY_FORWARD_TOL
            ):
                failures.append("serve parity forward_parity.tolerance mismatch")
            per_head = forward.get("per_head_tolerance")
            if not isinstance(per_head, dict) or set(per_head) != set(
                SERVE_PARITY_FORWARD_HEADS
            ):
                failures.append(
                    "serve parity per_head_tolerance set is not exact"
                )
            elif any(
                not _is_exact_number(value, SERVE_PARITY_FORWARD_TOL)
                for value in per_head.values()
            ):
                failures.append("serve parity per-head tolerances are not exact")
        calibration = payload.get("direction_calibration_parity")
        calibration_label = "serve parity direction_calibration_parity"
        failures.extend(_zero_failure_pass(calibration, label=calibration_label))
        if isinstance(calibration, dict):
            exact_calibration_keys = {
                "decision",
                "failures",
                "n_compared",
                "equation",
                "enabled",
                "temperature",
                "bias",
                "tolerance",
                "max_abs_diff",
                "worst_ts",
            }
            if set(calibration) != exact_calibration_keys:
                failures.append(f"{calibration_label} keys mismatch")
            if calibration.get("n_compared") != SERVE_PARITY_SAMPLE_COUNT:
                failures.append(f"{calibration_label}.n_compared mismatch")
            if calibration.get("equation") != SERVE_PARITY_CALIBRATION_EQUATION:
                failures.append(f"{calibration_label}.equation mismatch")
            if calibration.get("enabled") is not True:
                failures.append(f"{calibration_label}.enabled is not true")
            temperature = calibration.get("temperature")
            if not _is_finite_number(temperature) or float(temperature) <= 0.0:
                failures.append(f"{calibration_label}.temperature invalid")
            bias = calibration.get("bias")
            if (
                not isinstance(bias, list)
                or len(bias) != 3
                or not all(_is_finite_number(item) for item in bias)
            ):
                failures.append(f"{calibration_label}.bias invalid")
            if not _is_exact_number(
                calibration.get("tolerance"), SERVE_PARITY_CALIBRATION_TOL
            ):
                failures.append(f"{calibration_label}.tolerance mismatch")
            max_diff = calibration.get("max_abs_diff")
            if (
                not _is_finite_number(max_diff)
                or float(max_diff) < 0.0
                or float(max_diff) > SERVE_PARITY_CALIBRATION_TOL
            ):
                failures.append(f"{calibration_label}.max_abs_diff exceeds contract")
        expected_rows = None
        if isinstance(coverage, dict):
            dataset_coverage = coverage.get("dataset")
            if isinstance(dataset_coverage, dict):
                rows = dataset_coverage.get("rows")
                if not isinstance(rows, bool) and isinstance(rows, int) and rows > 0:
                    expected_rows = rows
        if expected_rows is None:
            failures.append("serve parity TEST row count is unavailable for liveness")
        else:
            failures.extend(
                _test_prediction_liveness_contract_failures(
                    payload.get("test_prediction_liveness"),
                    expected_rows=expected_rows,
                )
            )
        failures.extend(
            _specialist_decision_influence_contract_failures(
                payload.get("specialist_decision_influence")
            )
        )
        failures.extend(
            _individual_input_decision_influence_contract_failures(
                payload.get("individual_input_decision_influence")
            )
        )
        failures.extend(
            _upstream_context_decision_influence_contract_failures(
                payload.get("upstream_context_decision_influence")
            )
        )
        failures.extend(
            _multi_tf_decision_influence_contract_failures(
                payload.get("multi_tf_decision_influence")
            )
        )
        failures.extend(
            _family_tf_decision_influence_contract_failures(
                payload.get("family_tf_decision_influence")
            )
        )
        failures.extend(
            _direction_evidence_fusion_influence_contract_failures(
                payload.get("direction_evidence_fusion_influence"),
                bundle_dir=payload.get("bundle_dir"),
                prediction_evidence=payload.get("prediction_evidence"),
            )
        )
    elif evidence_name == "model_native_direction_pocket_audit":
        if (
            payload.get("schema_version")
            != MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION
        ):
            failures.append("direction pocket schema_version mismatch")
        if not _is_exact_number(
            payload.get("max_selected_label_error_rate"),
            DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE,
        ):
            failures.append(
                "direction pocket max_selected_label_error_rate mismatch"
            )
        if payload.get("min_selected_rows") != DIRECTION_POCKET_MIN_SELECTED_ROWS:
            failures.append("direction pocket min_selected_rows mismatch")
        if not _is_exact_number(
            payload.get("min_mean_proxy_pnl_bps_exclusive"),
            DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE,
        ):
            failures.append("direction pocket min_mean_proxy_pnl_bps mismatch")
        if not _is_exact_number(
            payload.get("max_selected_label_error_wilson_upper_95"),
            DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95,
        ):
            failures.append(
                "direction pocket selected-label Wilson threshold mismatch"
            )
        if not _is_exact_number(
            payload.get("wilson_confidence_level"),
            DIRECTION_POCKET_WILSON_CONFIDENCE_LEVEL,
        ):
            failures.append("direction pocket Wilson confidence mismatch")
        if (
            payload.get("spread_aware_proxy_pnl_contract")
            != DIRECTION_POCKET_SPREAD_AWARE_PROXY_CONTRACT
        ):
            failures.append("direction pocket spread-aware proxy contract mismatch")
        pockets = payload.get("pockets")
        required_pockets = set(DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS)
        if not isinstance(pockets, dict):
            failures.append("direction pocket metrics object is missing")
        else:
            missing_pockets = sorted(required_pockets - set(pockets))
            if missing_pockets:
                failures.append(
                    "direction pocket required repair metrics missing: "
                    + ",".join(missing_pockets)
                )
            for pocket_name in sorted(required_pockets.intersection(pockets)):
                row = pockets.get(pocket_name)
                if not isinstance(row, dict):
                    failures.append(f"direction pocket {pocket_name} is not an object")
                    continue
                selected_rows = row.get("selected_rows")
                pocket_rows = row.get("rows")
                correct_count = row.get("selected_label_correct_count")
                error_count = row.get("selected_label_error_count")
                if (
                    isinstance(pocket_rows, bool)
                    or not isinstance(pocket_rows, int)
                    or pocket_rows < DIRECTION_POCKET_MIN_SELECTED_ROWS
                ):
                    failures.append(
                        f"direction pocket {pocket_name}.rows below contract"
                    )
                if (
                    isinstance(selected_rows, bool)
                    or not isinstance(selected_rows, int)
                    or selected_rows < DIRECTION_POCKET_MIN_SELECTED_ROWS
                ):
                    failures.append(
                        f"direction pocket {pocket_name}.selected_rows below contract"
                    )
                    continue
                if (
                    isinstance(correct_count, bool)
                    or not isinstance(correct_count, int)
                    or isinstance(error_count, bool)
                    or not isinstance(error_count, int)
                    or correct_count < 0
                    or error_count < 0
                    or correct_count + error_count != selected_rows
                ):
                    failures.append(
                        f"direction pocket {pocket_name} selected-label counts invalid"
                    )
                    continue
                expected_error_rate = error_count / selected_rows
                observed_error_rate = row.get("selected_label_error_rate")
                if (
                    not _is_finite_number(observed_error_rate)
                    or abs(
                        float(observed_error_rate) - expected_error_rate
                    )
                    > 1e-12
                    or float(observed_error_rate)
                    > DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_RATE
                ):
                    failures.append(
                        f"direction pocket {pocket_name}.selected_label_error_rate "
                        "violates contract"
                    )
                observed_correct_rate = row.get("selected_label_correct_rate")
                if (
                    not _is_finite_number(observed_correct_rate)
                    or abs(
                        float(observed_correct_rate)
                        - (correct_count / selected_rows)
                    )
                    > 1e-12
                ):
                    failures.append(
                        f"direction pocket {pocket_name}.selected_label_correct_rate "
                        "is inconsistent"
                    )
                expected_wilson = direction_pocket_wilson_upper_95(
                    failures=error_count,
                    total=selected_rows,
                )
                observed_wilson = row.get(
                    "selected_label_error_wilson_upper_95"
                )
                if (
                    not _is_finite_number(observed_wilson)
                    or abs(float(observed_wilson) - expected_wilson) > 1e-12
                    or float(observed_wilson)
                    > DIRECTION_POCKET_MAX_SELECTED_LABEL_ERROR_WILSON_UPPER_95
                ):
                    failures.append(
                        f"direction pocket {pocket_name} selected-label Wilson "
                        "upper bound violates contract"
                    )
                mean_proxy = row.get("selected_mean_proxy_pnl_bps")
                if (
                    not _is_finite_number(mean_proxy)
                    or float(mean_proxy)
                    <= DIRECTION_POCKET_MIN_MEAN_PROXY_PNL_BPS_EXCLUSIVE
                ):
                    failures.append(
                        f"direction pocket {pocket_name} spread-aware proxy edge is unproven"
                    )
    else:
        failures.append(f"unknown serve-gate evidence name {evidence_name!r}")
    return failures


def cross_gate_contract_failures(
    parity: Mapping[str, Any],
    direction: Mapping[str, Any],
) -> list[str]:
    """Require both launch proofs to describe the same immutable TEST event."""

    failures: list[str] = []
    for field in (
        "bundle_dir",
        "dataset_dir",
        "dataset_parquet",
        "dataset_parquet_sha256",
        "prediction_evidence",
        "prediction_report_evidence",
        "test_coverage",
    ):
        if parity.get(field) != direction.get(field):
            failures.append(f"serve-gate cross-event {field} mismatch")
    return failures
