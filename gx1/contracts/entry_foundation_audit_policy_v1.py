"""Immutable policy shared by every Entry foundation audit and consumer."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from gx1.contracts.entry_full_input_liveness_v1 import RARE_EVENT_MINIMUMS
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    CLASS_ORDER,
)
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CONTRACT_MODE
from gx1.time.session_detector import SESSION_ORDER


FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION = "entry_foundation_audit_policy_v7"
FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION = "entry_target_foundation_audit_v2"
FOUNDATION_AUDIT_DATA_SPLITS = ("train", "val", "test")
FOUNDATION_AUDIT_SMOKE_SPLITS = ("val", "test")
FOUNDATION_AUDIT_POLICY_SECTIONS = {
    "feature": "feature_liveness",
    "target": "target_quality",
    "specialist": "specialist_liveness",
}

_MIN_SPECIALIST_SIGNAL_COUNTS = {
    "structure_swing_encoder": 10,
    "smc_liquidity_encoder": 8,
    "trend_ema_encoder": 6,
    "vol_compression_encoder": 6,
    "momentum_flow_encoder": 3,
    "session_regime_encoder": 6,
    "chart_geometry_encoder": 8,
    "price_action_candle_encoder": 6,
}

_SPECIALIST_SIGNAL_RARE_EVENT_MINIMUMS = {
    field: int(minimums["train"])
    for (surface, field), minimums in sorted(RARE_EVENT_MINIMUMS.items())
    if surface == "signal" and "train" in minimums
}

_FOUNDATION_AUDIT_POLICY: dict[str, Any] = {
    "schema_version": FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION,
    "audit_data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
    "feature_liveness": {
        "liveness_epsilon": 1e-7,
        "near_constant_std": 1e-9,
        "min_required_family_active_rate": 0.01,
        "min_required_objective_active_rate": 0.01,
        "min_required_source_active_rate": 0.0001,
        "min_required_source_active_count": 1,
        "parquet_batch_size": 4096,
        "selected_feature_learnability_splits": ["train"],
        "selected_features_finite_all_splits": True,
        "foundation_outputs_live_all_splits": True,
        "foundation_sources_live_all_splits": True,
        "oos_single_state_is_observation_not_train_learnability_failure": True,
    },
    "target_quality": {
        "max_majority_rate": 0.90,
        "min_tradable_rate": 0.01,
        "max_tradable_rate": 0.99,
        "direction_horizon_bars": 24,
        "path_quality_horizon_bars": 10,
        "xau_direction_repair_targets_required": True,
        "all_active_head_targets_live_each_split": True,
        "scalar_bad_path_path_quality_max_spearman_exclusive": 0.0,
        "side_quality": {
            "max_bad_path_vs_utility_spearman": -0.10,
            "min_bad_path_vs_expected_mae_spearman": 0.10,
            "require_bad_utility_below_clean": True,
            "require_bad_mae_above_clean": True,
        },
        "position_size_target": {
            "formula": (
                "sigmoid((mfe_first_n_bps-mae_first_n_bps)/(2*atr_bps))"
            ),
            "mae_semantics": "non_negative_adverse_magnitude",
            "atr_denominator_multiplier": 2.0,
            "logit_clip_abs": 80.0,
            "flat_direction_id": 2,
            "flat_value": 0.5,
            "atr_bps_min_exclusive": 0.0,
            "max_abs_error": 1e-6,
            "live_size_application_authority": False,
        },
        "offline_rl_target": {
            "action_order": list(CLASS_ORDER),
            "horizon_bars": [12, 48, 96],
            "flat_reward_bps": 0.0,
            "min_best_action_rate_per_horizon": 0.01,
            "max_best_action_rate_per_horizon": 0.98,
            "all_counterfactual_actions_required": True,
        },
    },
    "specialist_liveness": {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "liveness_epsilon": 1e-7,
        "near_constant_std": 1e-9,
        "min_feature_active_rate": 0.01,
        "train_live_statuses": ["LIVE", "ALLOWED_RARE_EVENT"],
        "oos_observed_statuses": [
            "OBSERVED_VARIABLE",
            "OBSERVED_RARE_EVENT",
            "OBSERVED_SINGLE_STATE",
        ],
        "rare_event_minimum_active_count": dict(
            _SPECIALIST_SIGNAL_RARE_EVENT_MINIMUMS
        ),
        "rare_event_rule": (
            "finite_and_variable_and_exact_train_support_floor; "
            "never a constant allowlist or direction pass-through"
        ),
        "min_specialist_mean_active_rate": 0.01,
        "min_signal_counts": dict(_MIN_SPECIALIST_SIGNAL_COUNTS),
        "min_live_feature_counts": dict(_MIN_SPECIALIST_SIGNAL_COUNTS),
    },
    "smoke_edge_pockets": {
        "evaluation_splits": list(FOUNDATION_AUDIT_SMOKE_SPLITS),
        "min_direction_accuracy": 0.90,
        "min_balanced_accuracy": 0.90,
        "min_trade_direction_precision": 0.98,
        "min_class_precision": 0.95,
        # Point estimates alone can report perfect precision on negligible
        # support.  Promotion therefore also requires fixed sample floors and
        # one-sided evidence through the lower 95% Wilson bound.  These values
        # are immutable policy, never caller- or environment-tunable.
        "wilson_confidence_level": 0.95,
        "wilson_z_score": 1.959963984540054,
        "min_trade_rows": 200,
        "min_prediction_rows_per_class": 100,
        "min_trade_precision_wilson_lower": 0.95,
        "min_class_precision_wilson_lower": 0.90,
        "context_fields": ["session", "vol_regime"],
        "expected_sessions": list(SESSION_ORDER),
        "min_rows_per_context_slice": 32,
        "min_context_trade_direction_precision": 0.95,
        "min_context_trade_rows": 32,
        "min_context_trade_precision_wilson_lower": 0.85,
        "min_specialist_mean_weight": 0.01,
        "min_specialist_gate_entropy": 0.50,
        "min_specialist_gate_std": 1e-6,
        "turning_point_evidence": {
            "evaluation_horizon_bars": 12,
            "near_turn_max_fraction": 0.25,
            "min_prediction_target_spearman": 0.10,
            "max_prediction_target_mae": 0.20,
            "min_near_turn_trade_rows_per_side": 50,
            "min_near_turn_direction_precision": 0.98,
            "min_near_turn_precision_wilson_lower": 0.90,
            "min_near_turn_timing_precision": 0.80,
            "min_near_turn_timing_precision_wilson_lower": 0.70,
            "prediction_bounds_inclusive": [0.0, 1.0],
            "direction_authority": "final_model_direction_argmax_only",
        },
        "offline_rl_evidence": {
            "min_q_target_spearman": 0.10,
            "max_q_target_mae_scaled": 1.00,
            "max_flat_q_abs_mean_scaled": 0.25,
            "min_reward_argmax_accuracy_per_horizon": 0.70,
            "min_unique_reward_rows_per_horizon": 100,
            "min_value_vs_max_q_spearman": 0.10,
            "max_advantage_parity_abs": 1e-6,
            "separate_direction_authority": False,
        },
    },
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FOUNDATION_AUDIT_POLICY_SHA256 = _canonical_sha256(_FOUNDATION_AUDIT_POLICY)


def foundation_audit_policy_metadata() -> dict[str, Any]:
    """Return a defensive JSON-safe copy of the one accepted policy."""

    return copy.deepcopy(_FOUNDATION_AUDIT_POLICY)


def foundation_audit_policy_binding() -> dict[str, Any]:
    return {
        "foundation_audit_policy_schema_version": (
            FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION
        ),
        "foundation_audit_policy_sha256": FOUNDATION_AUDIT_POLICY_SHA256,
        "foundation_audit_policy": foundation_audit_policy_metadata(),
    }


def foundation_audit_policy_enforcement(audit_kind: str) -> dict[str, Any]:
    kind = str(audit_kind)
    section_name = FOUNDATION_AUDIT_POLICY_SECTIONS.get(kind)
    if section_name is None:
        raise ValueError(f"unknown foundation audit kind: {audit_kind!r}")
    return {
        "audit_kind": kind,
        "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
        "policy_section_name": section_name,
        "policy_section": copy.deepcopy(_FOUNDATION_AUDIT_POLICY[section_name]),
    }


def require_foundation_audit_policy_binding(
    value: Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Require the embedded full policy and its canonical identity exactly."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_FOUNDATION_AUDIT_POLICY_BINDING_MISSING]")
    schema = value.get("foundation_audit_policy_schema_version")
    digest = str(value.get("foundation_audit_policy_sha256") or "").lower()
    policy = value.get("foundation_audit_policy")
    if schema != FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION:
        raise RuntimeError(f"[{context}_FOUNDATION_AUDIT_POLICY_SCHEMA_INVALID]")
    if not isinstance(policy, Mapping):
        raise RuntimeError(f"[{context}_FOUNDATION_AUDIT_POLICY_MISSING]")
    normalized_policy = dict(policy)
    if normalized_policy != _FOUNDATION_AUDIT_POLICY:
        raise RuntimeError(f"[{context}_FOUNDATION_AUDIT_POLICY_PAYLOAD_INVALID]")
    if _canonical_sha256(normalized_policy) != FOUNDATION_AUDIT_POLICY_SHA256:
        raise RuntimeError(f"[{context}_FOUNDATION_AUDIT_POLICY_CONTENT_HASH_INVALID]")
    if digest != FOUNDATION_AUDIT_POLICY_SHA256:
        raise RuntimeError(f"[{context}_FOUNDATION_AUDIT_POLICY_SHA256_INVALID]")
    return foundation_audit_policy_binding()


def require_foundation_audit_report_policy(
    value: Any,
    *,
    audit_kind: str,
    context: str,
) -> dict[str, Any]:
    """Require exact policy identity plus the section enforced by one audit."""

    binding = require_foundation_audit_policy_binding(value, context=context)
    expected_enforcement = foundation_audit_policy_enforcement(audit_kind)
    if value.get("foundation_audit_policy_enforcement") != expected_enforcement:
        raise RuntimeError(
            f"[{context}_FOUNDATION_AUDIT_POLICY_ENFORCEMENT_INVALID]"
        )
    if tuple(value.get("data_splits") or ()) != FOUNDATION_AUDIT_DATA_SPLITS:
        raise RuntimeError(f"[{context}_FOUNDATION_AUDIT_SPLITS_INVALID]")
    return {
        **binding,
        "foundation_audit_policy_enforcement": expected_enforcement,
        "data_splits": list(FOUNDATION_AUDIT_DATA_SPLITS),
    }


__all__ = [
    "FOUNDATION_AUDIT_DATA_SPLITS",
    "FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION",
    "FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION",
    "FOUNDATION_AUDIT_POLICY_SECTIONS",
    "FOUNDATION_AUDIT_POLICY_SHA256",
    "FOUNDATION_AUDIT_SMOKE_SPLITS",
    "foundation_audit_policy_binding",
    "foundation_audit_policy_enforcement",
    "foundation_audit_policy_metadata",
    "require_foundation_audit_policy_binding",
    "require_foundation_audit_report_policy",
]
