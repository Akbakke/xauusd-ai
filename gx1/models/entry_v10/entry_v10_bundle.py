"""
ENTRY_V10 / ENTRY_V10_CTX bundle loader (STRICT, manifest-only friendly).

- No dummy models.
- No fallback paths.
- No hardcoded directories.
- No network.
- CPU/GPU determined by caller.
- Hard-fails if bundle is incomplete.

This loader assumes:
bundle_dir contains:
    - MASTER_TRANSFORMER_LOCK.json
    - bundle_metadata.json
    - model_state_dict.pt
"""

from __future__ import annotations

import io
import json
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Set

import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_fitted_q_v1 import (
    require_entry_fitted_q_contract,
    require_entry_fitted_q_iteration_state,
    require_entry_fitted_q_production_economics_readiness,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_MTF_CONTEXT_COUNT,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_MTF_CONTEXT_COUNT,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    current_entry_exit_architecture_observation,
    require_entry_exit_production_architecture,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    require_training_objective_contract,
)
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    JOINT_TASK_STATE_KEYS,
    require_joint_task_weighting_metadata,
)
from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
    require_learned_component_movement_metadata,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    require_model_native_aux_target_contract,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    NEUTRAL_EFFECTIVE_INIT as TF_INPUT_SCALE_NEUTRAL_INIT,
    TF_NAMES as TF_INPUT_SCALE_NAMES,
    require_tf_input_scale_contract,
    require_tf_input_scale_state,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    EXPECTED_TFS as INPUT_NORMALIZATION_TFS,
    require_input_normalization_contract,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PrefreezeTestSealLineageError,
    require_prefreeze_test_seal_lineage,
    require_prefreeze_test_seal_lineage_metadata,
)
from gx1.contracts.entry_model_native_training_run_lineage_v1 import (
    EntryModelNativeTrainingRunLineageError,
    require_training_run_lineage,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    require_unified_exit_lifecycle_authority_evidence,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_fitted_q_contract,
    require_unified_exit_fitted_q_iteration_state,
)
from gx1.contracts.unified_exit_gate_evidence_v1 import (
    require_unified_exit_gate_evidence,
)
from gx1.contracts.unified_exit_input_influence_v1 import (
    SCHEMA_VERSION as UNIFIED_EXIT_INPUT_INFLUENCE_SCHEMA_VERSION,
    require_unified_exit_input_influence,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
    require_model_direction_decision_contract,
    require_unified_entry_exit_contract,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION,
    require_multi_tf_specialist_routing_v4,
    require_model_native_context_specialist_routing,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_FEATURE_NAMES_SHA256_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    require_multi_tf_decision_window_coverage_metadata,
    require_multi_tf_resolution_pyramid,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    MODEL_ARCHITECTURE_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
)


@dataclass
class EntryV10Bundle:
    bundle_dir: str
    bundle_sha256: str
    device: torch.device
    transformer_model: Any
    metadata: Dict[str, Any]
    transformer_config: Dict[str, Any]
    capabilities: Dict[str, Any]


def _guard_required(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"[ENTRY_V10_BUNDLE_MISSING] {label} not found: {path}")


def _resolve_device(device: str) -> torch.device:
    if not isinstance(device, str) or not device.strip():
        raise RuntimeError("[ENTRY_V10_DEVICE_NOT_EXPLICIT]")
    return torch.device(device)


_ENTRY_HEAD_STATE_KEYS: Dict[str, Set[str]] = {
    "entry_action_q": {
        "entry_q_joint_norm.weight",
        "entry_q_joint_norm.bias",
        "entry_q_joint_in.weight",
        "entry_q_joint_in.bias",
        "head_entry_action_q.weight",
        "head_entry_action_q.bias",
    },
    "position_size": {"head_position_size.weight", "head_position_size.bias"},
    "trendline_event": {
        "head_trendline_event.weight",
        "head_trendline_event.bias",
    },
    "side_mae": {
        "head_side_mae.weight",
        "head_side_mae.bias",
    },
    "dip": {"head_dip.weight", "head_dip.bias"},
    "forecast": {"head_forecast.weight", "head_forecast.bias"},
    "timing": {"head_timing.weight", "head_timing.bias"},
    "tail_risk": {"head_tail_risk.weight", "head_tail_risk.bias"},
    "vol_forecast": {"head_vol_forecast.weight", "head_vol_forecast.bias"},
    "unified_exit": {
        "head_exit_action.weight",
        "head_exit_action.bias",
    },
}

_MODEL_NATIVE_METADATA_ONLY_COMPONENTS: frozenset[str] = frozenset()

_MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS = frozenset(
    {
        "entry_action_q",
        "position_size",
        "dip",
        "forecast",
        "timing",
        "tail_risk",
        "vol_forecast",
        "side_mae",
        "trendline_event",
        "unified_exit",
    }
)

_MODEL_NATIVE_REQUIRED_SPECIALISTS = MODEL_NATIVE_TRAINING_SPECIALISTS

_MODEL_NATIVE_FORBIDDEN_COMPATIBILITY_ARTIFACTS = (
    "feature_meta_path",
    "seq_scaler_path",
    "snap_scaler_path",
)


def _require_mapping_field(parent: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_METADATA_MISSING] {context}.{key}")
    return value


def _bundle_architecture_observation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Project existing bundle metadata without reading model-state bytes."""

    observed = current_entry_exit_architecture_observation()
    signal_contract = payload.get("model_native_signal_contract")
    normalization = payload.get("input_normalization")
    mtf = payload.get("multi_tf")
    specialist = payload.get("specialist_fusion")
    unified = payload.get("unified_entry_exit_contract")
    context_routing = payload.get("context_specialist_routing")
    signal_contract = signal_contract if isinstance(signal_contract, Mapping) else {}
    normalization = normalization if isinstance(normalization, Mapping) else {}
    mtf = mtf if isinstance(mtf, Mapping) else {}
    specialist = specialist if isinstance(specialist, Mapping) else {}
    unified = unified if isinstance(unified, Mapping) else {}
    context_routing = (
        context_routing if isinstance(context_routing, Mapping) else {}
    )
    shared_base = unified.get("shared_feature_base_contract")
    shared_base = shared_base if isinstance(shared_base, Mapping) else {}
    local_indices = specialist.get("input_indices")
    if not isinstance(local_indices, Mapping):
        local_indices = context_routing.get("ctx_cont_indices")
    mtf_indices = mtf.get("specialist_input_indices")

    observed["schemas"]["signal"] = signal_contract.get("schema_version")
    observed["schemas"]["input_normalization"] = normalization.get(
        "schema_version"
    )
    observed["schemas"]["mtf_matrix"] = mtf.get("matrix_contract")
    observed["shared_surface"] = {
        "signal_dim": payload.get("seq_input_dim"),
        "snap_dim": payload.get("snap_input_dim"),
        "ctx_cont_dim": payload.get("ctx_cont_dim"),
        "ctx_cat_dim": payload.get("ctx_cat_dim"),
    }
    observed["shared_encoder"] = shared_base.get("shared_encoder")
    observed["local_specialists"] = (
        list(local_indices) if isinstance(local_indices, Mapping) else local_indices
    )
    observed["mtf_specialists"] = (
        list(mtf_indices) if isinstance(mtf_indices, Mapping) else mtf_indices
    )
    observed["mtf"] = {
        "generation": "V4" if mtf.get("v4_mode") is True else mtf.get("v4_mode"),
        "native_source_timeframe": shared_base.get("entry", {}).get(
            "decision_timeframe"
        )
        if isinstance(shared_base.get("entry"), Mapping)
        else None,
        "cache_timeframes": ["M5", "M15", "H1", "H4", "D1"],
        "per_tf_widths": {
            "M5": mtf.get("m5_seq_dim"),
            "M15": mtf.get("m15_seq_dim"),
            "H1": mtf.get("h1_seq_dim"),
            "H4": mtf.get("h4_seq_dim"),
            "D1": mtf.get("d1_seq_dim"),
        },
        "per_tf_window_bars": {
            "M5": mtf.get("m5_seq_len"),
            "M15": mtf.get("m15_seq_len"),
            "H1": mtf.get("h1_seq_len"),
            "H4": mtf.get("h4_seq_len"),
            "D1": mtf.get("d1_seq_len"),
        },
        "m5_route_consumption": {
            "entry": (
                "M5" in mtf.get("entry_route_timeframes", ())
                if isinstance(mtf.get("entry_route_timeframes"), list)
                else None
            ),
            "exit": (
                "M5" in mtf.get("exit_route_timeframes", ())
                if isinstance(mtf.get("exit_route_timeframes"), list)
                else None
            ),
        },
    }
    observed["entry"] = {
        "local_timeframe": unified.get("entry_local_timeframe"),
        "sequence_bars": payload.get("seq_len"),
        "mtf_route": mtf.get("entry_route_timeframes"),
    }
    observed["exit"] = {
        "local_timeframe": unified.get("exit_local_timeframe"),
        "sequence_bars": unified.get("exit_local_sequence_bars"),
        "mtf_route": mtf.get("exit_route_timeframes"),
        "max_path_bars": unified.get("exit_max_path_bars"),
    }
    return observed


def _require_exact_model_native_bundle_metadata(
    meta: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    """Reject every metadata default that could alter a model-native forward.

    Lock and metadata must each carry the complete reconstruction contract;
    the loader never infers a missing value from the other file or a default.
    """

    stale_artifacts = [
        f"{owner}.{key}"
        for owner, payload in (("meta", meta), ("lock", lock))
        for key in _MODEL_NATIVE_FORBIDDEN_COMPATIBILITY_ARTIFACTS
        if key in payload
    ]
    if stale_artifacts:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_STALE_COMPATIBILITY_ARTIFACT_FORBIDDEN] "
            f"fields={stale_artifacts}"
        )

    shared_exact = (
        "contract_mode",
        "seq_input_dim",
        "snap_input_dim",
        "seq_len",
        "dropout",
        "ctx_cont_dim",
        "ctx_cat_dim",
        "ordered_signal_names",
        "ordered_ctx_cont_names",
        "ordered_ctx_cat_names",
        "model_native_signal_contract",
        "aux_head_target_contract",
        "model_native_training_objective",
        "model_native_joint_task_weighting",
        "model_native_entry_fitted_q",
        "entry_fitted_q_production_economics",
        "selected_entry_fitted_q_iteration_state",
        "model_native_learned_component_movement",
        "context_specialist_routing",
        "input_normalization",
        "input_normalization_fit_population_proof",
        "multi_tf",
        "tf_input_scale",
        "run_lineage",
        "prefreeze_test_seal_lineage",
        "m1_feature_surface_binding",
        # TRAIN and VAL windows are reconstructed from their immutable M5
        # feature surfaces. This evidence is not an inference-time model
        # parameter, but both reconstruction proofs are part of the exact
        # trained-data lineage and must never differ between metadata and the
        # transformer lock.
        "sequence_source_reconstruction",
        "model_architecture_schema_version",
        "model_output_schema_version",
    )
    missing_meta = [key for key in shared_exact if key not in meta]
    missing_lock = [key for key in shared_exact if key not in lock]
    if missing_meta or missing_lock:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_EXACT_METADATA_MISSING] "
            f"meta={missing_meta} lock={missing_lock}"
        )
    split_brain = [key for key in shared_exact if meta[key] != lock[key]]
    if split_brain:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_META_LOCK_SPLIT_BRAIN] "
            f"fields={split_brain}"
        )
    if (
        meta.get("schema_version") != "entry_v10_ctx_bundle_metadata_v3"
        or lock.get("version") != "entry_v10_ctx_lock_v3"
        or meta["model_architecture_schema_version"]
        != MODEL_ARCHITECTURE_SCHEMA_VERSION
        or lock["model_architecture_schema_version"]
        != MODEL_ARCHITECTURE_SCHEMA_VERSION
        or meta["model_output_schema_version"] != MODEL_OUTPUT_SCHEMA_VERSION
        or lock["model_output_schema_version"] != MODEL_OUTPUT_SCHEMA_VERSION
        or meta.get("arch_id") != MODEL_ARCHITECTURE_SCHEMA_VERSION
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_OUTPUT_SCHEMA_INVALID]")
    meta_training_objective = require_training_objective_contract(
        meta["model_native_training_objective"],
        context="ENTRY_BUNDLE_META",
    )
    lock_training_objective = require_training_objective_contract(
        lock["model_native_training_objective"],
        context="ENTRY_BUNDLE_LOCK",
    )
    if meta_training_objective != lock_training_objective:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_TRAINING_OBJECTIVE_SPLIT_BRAIN]"
        )
    meta_joint_task_weighting = require_joint_task_weighting_metadata(
        meta["model_native_joint_task_weighting"],
        context="ENTRY_BUNDLE_META",
    )
    lock_joint_task_weighting = require_joint_task_weighting_metadata(
        lock["model_native_joint_task_weighting"],
        context="ENTRY_BUNDLE_LOCK",
    )
    if meta_joint_task_weighting != lock_joint_task_weighting:
        raise RuntimeError(
            "[ENTRY_BUNDLE_JOINT_TASK_WEIGHTING_SPLIT_BRAIN]"
        )
    require_entry_fitted_q_contract(
        meta["model_native_entry_fitted_q"],
        context="ENTRY_BUNDLE_META",
    )
    require_entry_fitted_q_production_economics_readiness(
        meta["entry_fitted_q_production_economics"],
        context="ENTRY_BUNDLE_META",
        require_ready=False,
    )
    require_learned_component_movement_metadata(
        meta["model_native_learned_component_movement"],
        context="ENTRY_BUNDLE_META",
    )
    require_learned_component_movement_metadata(
        lock["model_native_learned_component_movement"],
        context="ENTRY_BUNDLE_LOCK",
    )
    require_model_native_aux_target_contract(
        meta["aux_head_target_contract"],
        context="ENTRY_BUNDLE_MODEL_NATIVE",
    )
    if meta["contract_mode"] != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MODE_MISSING]")
    if int(meta["seq_input_dim"]) != MODEL_NATIVE_SIGNAL_DIM or int(meta["snap_input_dim"]) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_SIGNAL_DIM_INVALID] "
            f"seq={meta['seq_input_dim']} snap={meta['snap_input_dim']} expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    if int(meta["seq_len"]) != MODEL_NATIVE_SEQ_LEN:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_SEQ_LEN_INVALID] "
            f"got={meta['seq_len']!r} expected={MODEL_NATIVE_SEQ_LEN}"
        )
    if (
        isinstance(meta["dropout"], bool)
        or not math.isfinite(float(meta["dropout"]))
        or not 0.0 <= float(meta["dropout"]) < 1.0
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_DROPOUT_INVALID] "
            f"got={meta['dropout']!r}"
        )
    if (
        int(meta["ctx_cont_dim"]) != MODEL_NATIVE_CTX_CONT_DIM
        or int(meta["ctx_cat_dim"]) != MODEL_NATIVE_CTX_CAT_DIM
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_CTX_DIM_INVALID] "
            f"continuous={meta['ctx_cont_dim']!r} categorical={meta['ctx_cat_dim']!r} "
            f"expected={MODEL_NATIVE_CTX_CONT_DIM}+{MODEL_NATIVE_CTX_CAT_DIM}"
        )
    if len(meta["ordered_ctx_cont_names"]) != int(meta["ctx_cont_dim"]):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CONT_ORDER_INVALID]")
    if len(meta["ordered_ctx_cat_names"]) != int(meta["ctx_cat_dim"]):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CAT_ORDER_INVALID]")
    if list(meta["ordered_ctx_cont_names"]) != list(MODEL_NATIVE_CTX_CONT_FIELDS):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CONT_ORDER_INVALID]")
    if list(meta["ordered_ctx_cat_names"]) != list(MODEL_NATIVE_CTX_CAT_FIELDS):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CTX_CAT_ORDER_INVALID]")
    meta_signal_contract = meta["model_native_signal_contract"]
    lock_signal_contract = lock["model_native_signal_contract"]
    require_model_native_signal_contract(meta_signal_contract, context="ENTRY_BUNDLE_META")
    require_model_native_signal_contract(lock_signal_contract, context="ENTRY_BUNDLE_LOCK")
    if list(meta["ordered_signal_names"]) != list(meta_signal_contract["fields"]):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SIGNAL_ORDER_INVALID]")
    if meta.get("supports_context_features") is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_CONTEXT_FEATURES_REQUIRED]")
    if meta.get("anchored_entry_enabled") is not False or meta.get("anchor_source") is not None:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ANCHOR_METADATA_FORBIDDEN]")
    anchor_gate = _require_mapping_field(meta, "anchor_gate", context="meta")
    if anchor_gate.get("enabled") is not False:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ANCHOR_GATE_METADATA_FORBIDDEN]")

    require_model_direction_decision_contract(meta, context="ENTRY_BUNDLE_META")
    require_model_direction_decision_contract(lock, context="ENTRY_BUNDLE_LOCK")
    if meta["direction_decision_contract"] != lock["direction_decision_contract"]:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_DIRECTION_CONTRACT_SPLIT_BRAIN]")
    require_unified_entry_exit_contract(
        meta,
        context="ENTRY_BUNDLE_META",
    )
    require_unified_entry_exit_contract(
        lock,
        context="ENTRY_BUNDLE_LOCK",
    )
    if (
        meta["unified_entry_exit_contract"]
        != lock["unified_entry_exit_contract"]
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_ENTRY_EXIT_CONTRACT_SPLIT_BRAIN]"
        )
    exit_evidence = _require_mapping_field(
        meta,
        "unified_exit_training_evidence",
        context="meta",
    )
    lock_exit_evidence = _require_mapping_field(
        lock,
        "unified_exit_training_evidence",
        context="lock",
    )
    if exit_evidence != lock_exit_evidence:
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_EXIT_TRAINING_EVIDENCE_SPLIT_BRAIN]"
        )
    exit_validation = exit_evidence.get("selected_checkpoint_validation")
    exit_movement = exit_evidence.get(
        "selected_checkpoint_parameter_movement"
    )
    exit_lifecycle = exit_evidence.get("lifecycle")
    full_trajectory_validation = exit_evidence.get(
        "full_trajectory_validation"
    )
    exit_input_influence = exit_evidence.get("individual_input_influence")
    training_profile = meta.get("run_lineage", {}).get("training_profile")
    try:
        fitted_q_contract = require_unified_exit_fitted_q_contract(
            exit_evidence.get("fitted_q_contract"),
            context="ENTRY_BUNDLE",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_EXIT_FITTED_Q_CONTRACT_INVALID]"
        ) from exc
    fitted_q_state = exit_evidence.get("selected_fitted_q_iteration_state")
    try:
        fitted_q_state = require_unified_exit_fitted_q_iteration_state(
            fitted_q_state,
            context="ENTRY_BUNDLE",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_EXIT_FITTED_Q_STATE_INVALID]"
        ) from exc
    try:
        require_entry_fitted_q_iteration_state(
            meta["selected_entry_fitted_q_iteration_state"],
            exit_fitted_q_iteration_state=fitted_q_state,
            context="ENTRY_BUNDLE_ENTRY_FITTED_Q",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "[ENTRY_BUNDLE_ENTRY_FITTED_Q_STATE_INVALID]"
        ) from exc
    try:
        require_unified_exit_lifecycle_authority_evidence(exit_lifecycle)
    except RuntimeError as exc:
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_EXIT_LIFECYCLE_AUTHORITY_INVALID]"
        ) from exc
    if (
        exit_evidence.get("schema_version")
        != "gx1_unified_exit_training_evidence_v10"
        or exit_evidence.get("decision") != "PASS"
        or exit_evidence.get("shared_model_state_dict") is not True
        or exit_evidence.get("entry_representation_surface")
        != UNIFIED_EXIT_MODEL_REPRESENTATION_KEY
        or exit_evidence.get("future_outcomes_used_as_model_inputs") is not False
        or exit_evidence.get("exit_action_task_name") != "unified_exit_action"
        or exit_evidence.get("exit_action_target")
        != "train_fitted_raw_bps_q_iteration"
        or exit_evidence.get("exit_action_loss")
        != "mean_squared_error_over_valid_q_cells"
        or exit_evidence.get("gamma") != 1.0
        or exit_evidence.get("intermediate_hold_reward_bps") != 0.0
        or exit_evidence.get("baseline_cross_entropy_authority") is not False
        or fitted_q_state["fitted_q_contract"] != fitted_q_contract
        or exit_evidence.get("loss_scalarization")
        != "model_native_joint_task_weighting"
        or not isinstance(exit_validation, Mapping)
        or int(exit_validation.get("unified_exit_population_rows", 0)) <= 0
        or int(exit_validation.get("unified_exit_q_valid_cells", 0)) <= 0
        or int(exit_validation.get("unified_exit_target_equivalent_action_rows", -1)) < 0
        or int(exit_validation.get("unified_exit_hold_target_greedy_rows", 0)) <= 0
        or int(exit_validation.get("unified_exit_exit_now_target_greedy_rows", 0)) <= 0
        or not math.isfinite(
            float(exit_validation.get("unified_exit_raw_bps_q_mse_mean", float("nan")))
        )
        or float(exit_validation.get("unified_exit_raw_bps_q_mse_mean", -1.0)) < 0.0
        or not isinstance(exit_movement, Mapping)
        or exit_movement.get("all_exit_components_moved") is not True
        or not isinstance(exit_lifecycle, Mapping)
        or exit_lifecycle.get("future_outcomes_used_as_model_inputs") is not False
        or not isinstance(full_trajectory_validation, Mapping)
        or not isinstance(exit_input_influence, Mapping)
        or (
            training_profile == "candidate"
            and (
                full_trajectory_validation.get("schema_version")
                != "gx1_unified_exit_full_trajectory_validation_v7"
                or full_trajectory_validation.get("decision") != "PASS"
                or full_trajectory_validation.get("population")
                != "all_causal_states_both_sides_batched_episode_forward"
                or int(full_trajectory_validation.get("population_rows", 0)) <= 0
                or int(full_trajectory_validation.get("q_valid_cells", 0)) <= 0
                or int(full_trajectory_validation.get("target_equivalent_action_rows", -1)) < 0
                or int(full_trajectory_validation.get("predicted_tied_rows", -1)) != 0
                or not math.isfinite(
                    float(full_trajectory_validation.get("fitted_q_bellman_mse_mean", float("nan")))
                )
                or float(full_trajectory_validation.get("fitted_q_bellman_mse_mean", -1.0)) < 0.0
                or not 0.0 <= float(
                    full_trajectory_validation.get(
                        "unique_target_action_agreement", float("nan")
                    )
                ) <= 1.0
                or any(
                    not math.isfinite(float(full_trajectory_validation.get(key, float("nan"))))
                    for key in (
                        "learned_policy_mean_realized_executable_pnl_bps",
                        "immediate_exit_mean_realized_executable_pnl_bps",
                        "terminal_exit_mean_realized_executable_pnl_bps",
                    )
                )
                or int(full_trajectory_validation.get("long_population_rows", 0)) <= 0
                or int(full_trajectory_validation.get("short_population_rows", 0)) <= 0
                or int(full_trajectory_validation["long_population_rows"])
                + int(full_trajectory_validation["short_population_rows"])
                != int(full_trajectory_validation["population_rows"])
                or full_trajectory_validation.get(
                    "predicted_exact_q_tie_runtime_policy"
                ) != "fail_closed"
                or full_trajectory_validation.get("gamma") != 1.0
                or full_trajectory_validation.get(
                    "intermediate_hold_reward_bps"
                ) != 0.0
                or full_trajectory_validation.get(
                    "future_outcomes_used_as_model_inputs"
                )
                is not False
            )
        )
        or (
            training_profile == "smoke"
            and (
                full_trajectory_validation.get("decision")
                != "NOT_RUN_SMOKE_CANNOT_AUTHORIZE_CANDIDATE"
                or full_trajectory_validation.get("required_for_candidate")
                is not True
                or exit_input_influence.get("schema_version")
                != UNIFIED_EXIT_INPUT_INFLUENCE_SCHEMA_VERSION
                or exit_input_influence.get("decision")
                != "NOT_RUN_SMOKE_CANNOT_AUTHORIZE_CANDIDATE"
                or exit_input_influence.get("required_for_candidate") is not True
            )
        )
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_EXIT_TRAINING_EVIDENCE_INVALID]"
        )
    try:
        _require_profiled_unified_exit_gate_evidence(
            training_profile=str(training_profile),
            exit_validation=exit_validation,
            full_trajectory_validation=full_trajectory_validation,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_EXIT_GATE_EVIDENCE_INVALID]"
        ) from exc
    if training_profile == "candidate":
        try:
            require_unified_exit_input_influence(
                exit_input_influence,
                ordered_signal_names=meta.get("ordered_signal_names", ()),
                context="ENTRY_BUNDLE_SELECTED_CHECKPOINT",
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "[ENTRY_BUNDLE_UNIFIED_EXIT_INPUT_INFLUENCE_INVALID]"
            ) from exc

    train_recipe = _require_mapping_field(meta, "train_recipe", context="meta")
    active_raw = train_recipe.get("active_heads")
    if not isinstance(active_raw, list) or any(not isinstance(value, str) for value in active_raw):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ACTIVE_COMPONENTS_MISSING]")
    active = frozenset(active_raw)
    if len(active_raw) != len(active):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_ACTIVE_COMPONENTS_DUPLICATE]")
    if active != _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_ACTIVE_COMPONENTS_MISMATCH] "
            f"missing={sorted(_MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS - active)} "
            f"unexpected={sorted(active - _MODEL_NATIVE_REQUIRED_ACTIVE_COMPONENTS)}"
        )

    mtf = _require_mapping_field(meta, "multi_tf", context="meta")
    required_mtf = (
        "enabled",
        "v4_mode",
        "route_schema_version",
        "entry_route_timeframes",
        "exit_route_timeframes",
        "entry_target_availability_shift_minutes",
        "exit_target_availability_shift_minutes",
        "entry_tf_gate_width",
        "exit_tf_gate_width",
        "entry_family_tf_gate_width",
        "exit_family_tf_gate_width",
        "shared_cache_identity_sha256",
        "shared_cache_manifest_sha256",
        "shared_cache_dir",
        "shared_cache_manifest_path",
        "shared_cache_m5_source",
        "shared_cache_m5_source_sha256",
        "m5_seq_dim",
        "m5_seq_len",
        "m15_seq_dim",
        "m15_seq_len",
        "h1_seq_dim",
        "h1_seq_len",
        "h4_seq_dim",
        "h4_seq_len",
        "d1_seq_dim",
        "d1_seq_len",
        "multi_tf_num_layers",
        "multi_tf_scale",
        "feature_contract",
        "matrix_contract",
        "feature_names",
        "feature_names_sha256",
        "closed_bar_target_availability",
        "resolution_pyramid",
        "decision_window_coverage",
        "specialist_routing_schema_version",
        "specialist_input_indices",
        "parameter_family_tf_token_order",
        "entry_family_tf_token_order",
        "exit_family_tf_token_order",
    )
    missing_mtf = [key for key in required_mtf if key not in mtf]
    if missing_mtf:
        raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_MTF_METADATA_MISSING] {missing_mtf}")
    if mtf["enabled"] is not True or mtf["v4_mode"] is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_V4_REQUIRED]")
    # The frozen production architecture owns the shared surface, routes and
    # per-timeframe window bars. It is presented once the metadata is known to
    # be structurally present, and before any window/pyramid interpretation.
    for _owner, _payload in (("META", meta), ("LOCK", lock)):
        require_entry_exit_production_architecture(
            _bundle_architecture_observation(_payload),
            context=f"ENTRY_V10_BUNDLE_{_owner}_EXACT_METADATA",
        )
    # The bundle states which per-bar contract it was trained against and is
    # verified against exactly that one. Accepting a declared contract is not
    # the same as accepting any width: an unknown declaration fails closed, and
    # names, order and hash must all match the one named.
    _declared_contract = mtf.get("matrix_contract")
    if _declared_contract != HTF_V4_MATRIX_CONTRACT:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_EIGHT_FAMILY_REQUIRED] "
            f"observed={_declared_contract!r} expected={HTF_V4_MATRIX_CONTRACT!r}"
        )
    _mtf_names = MULTI_TF_PER_BAR_FEATURES_V4
    _mtf_names_sha = MULTI_TF_FEATURE_NAMES_SHA256_V4
    _mtf_count = MULTI_TF_FEATURE_COUNT_V4
    if (
        mtf["feature_contract"] != HTF_V4_MATRIX_CONTRACT
        or mtf["feature_names"] != list(_mtf_names)
        or mtf["feature_names_sha256"] != _mtf_names_sha
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_FEATURE_CONTRACT_INVALID]"
        )
    if not math.isfinite(float(mtf["multi_tf_scale"])) or float(mtf["multi_tf_scale"]) <= 0.0:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_SCALE_INVALID]")
    if (
        isinstance(mtf["multi_tf_num_layers"], bool)
        or not isinstance(mtf["multi_tf_num_layers"], int)
        or int(mtf["multi_tf_num_layers"]) <= 0
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_NUM_LAYERS_INVALID]"
        )
    if mtf["closed_bar_target_availability"] is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_CLOSED_BAR_REQUIRED]")
    expected_family_count = len(MODEL_NATIVE_TRAINING_SPECIALISTS)
    if (
        mtf["route_schema_version"] != "entry_exit_shared_mtf_routes_v1"
        or mtf["entry_route_timeframes"] != list(ENTRY_MTF_CONTEXT_TIMEFRAMES)
        or mtf["exit_route_timeframes"] != list(EXIT_MTF_CONTEXT_TIMEFRAMES)
        or float(mtf["entry_target_availability_shift_minutes"])
        != ENTRY_DECISION_BAR_SECONDS / 60.0
        or float(mtf["exit_target_availability_shift_minutes"])
        != EXIT_DECISION_BAR_SECONDS / 60.0
        or int(mtf["entry_tf_gate_width"]) != ENTRY_MTF_CONTEXT_COUNT
        or int(mtf["exit_tf_gate_width"]) != EXIT_MTF_CONTEXT_COUNT
        or int(mtf["entry_family_tf_gate_width"])
        != ENTRY_MTF_CONTEXT_COUNT * expected_family_count
        or int(mtf["exit_family_tf_gate_width"])
        != EXIT_MTF_CONTEXT_COUNT * expected_family_count
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_ROUTE_CONTRACT_INVALID]")
    for field in (
        "shared_cache_identity_sha256",
        "shared_cache_manifest_sha256",
        "shared_cache_m5_source_sha256",
    ):
        value = str(mtf[field])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_MTF_CACHE_BINDING_INVALID] {field}"
            )
    cache_dir = Path(str(mtf["shared_cache_dir"]))
    cache_manifest_path = Path(str(mtf["shared_cache_manifest_path"]))
    cache_m5_source = Path(str(mtf["shared_cache_m5_source"]))
    if (
        not cache_dir.is_absolute()
        or cache_dir.expanduser() != cache_dir
        or not cache_manifest_path.is_absolute()
        or cache_manifest_path.expanduser() != cache_manifest_path
        or cache_manifest_path != cache_dir / "manifest.json"
        or not cache_m5_source.is_absolute()
        or cache_m5_source.expanduser() != cache_m5_source
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_CACHE_PATH_BINDING_INVALID]"
        )
    per_tf_seq_lens = {
        "M5": int(mtf["m5_seq_len"]),
        "M15": int(mtf["m15_seq_len"]),
        "H1": int(mtf["h1_seq_len"]),
        "H4": int(mtf["h4_seq_len"]),
        "D1": int(mtf["d1_seq_len"]),
    }
    observed_pyramid = mtf["resolution_pyramid"]
    expected_pyramid = require_multi_tf_resolution_pyramid(per_tf_seq_lens)
    if observed_pyramid != expected_pyramid:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_RESOLUTION_PYRAMID_INVALID]"
        )
    try:
        require_multi_tf_decision_window_coverage_metadata(
            mtf["decision_window_coverage"],
            per_tf_seq_lens=per_tf_seq_lens,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_DECISION_COVERAGE_INVALID]"
        ) from exc
    expected_mtf_specialist_indices = {
        str(name): list(indices)
        for name, indices in require_multi_tf_specialist_routing_v4(
            mtf["feature_names"]
        ).items()
    }
    if (
        mtf["specialist_routing_schema_version"]
        != MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION
        or mtf["specialist_input_indices"]
        != expected_mtf_specialist_indices
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_SPECIALIST_ROUTING_INVALID]"
        )
    expected_parameter_family_tf_token_order = [
        f"{tf}:{specialist}"
        for tf in TF_INPUT_SCALE_NAMES
        for specialist in expected_mtf_specialist_indices
    ]
    expected_entry_family_tf_token_order = [
        f"{tf.lower()}:{specialist}"
        for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES
        for specialist in expected_mtf_specialist_indices
    ]
    expected_exit_family_tf_token_order = [
        f"{tf.lower()}:{specialist}"
        for tf in EXIT_MTF_CONTEXT_TIMEFRAMES
        for specialist in expected_mtf_specialist_indices
    ]
    if (
        mtf["parameter_family_tf_token_order"]
        != expected_parameter_family_tf_token_order
        or mtf["entry_family_tf_token_order"]
        != expected_entry_family_tf_token_order
        or mtf["exit_family_tf_token_order"]
        != expected_exit_family_tf_token_order
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_MTF_TOKEN_ORDER_INVALID]"
        )
    for key in (
        "m5_seq_dim",
        "m5_seq_len",
        "m15_seq_dim",
        "m15_seq_len",
        "h1_seq_dim",
        "h1_seq_len",
        "h4_seq_dim",
        "h4_seq_len",
        "d1_seq_dim",
        "d1_seq_len",
    ):
        if isinstance(mtf[key], bool) or int(mtf[key]) <= 0:
            raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_MTF_VALUE_INVALID] {key}={mtf[key]!r}")
    if any(
        int(mtf[key]) != int(_mtf_count)
        for key in (
            "m5_seq_dim",
            "m15_seq_dim",
            "h1_seq_dim",
            "h4_seq_dim",
            "d1_seq_dim",
        )
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_MTF_WIDTH_INVALID]")

    input_normalization = require_input_normalization_contract(
        meta["input_normalization"],
        expected_field_names={
            "signal": list(meta["ordered_signal_names"]),
            "ctx_cont": list(meta["ordered_ctx_cont_names"]),
            **{
                f"mtf_{tf.lower()}": list(_mtf_names)
                for tf in INPUT_NORMALIZATION_TFS
            },
        },
        expected_ctx_cat_names=list(meta["ordered_ctx_cat_names"]),
    )
    fit_proof = meta["input_normalization_fit_population_proof"]
    if not isinstance(fit_proof, Mapping):
        raise RuntimeError(
            "[ENTRY_BUNDLE_INPUT_NORMALIZATION_FIT_PROOF_MISSING]"
        )
    fit_proof_without_hash = dict(fit_proof)
    observed_fit_proof_hash = str(
        fit_proof_without_hash.pop("proof_sha256", "")
    )
    expected_fit_proof_hash = hashlib.sha256(
        json.dumps(
            fit_proof_without_hash,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    lineage = input_normalization["lineage"]
    expected_normalization_seq_lens = {
        "M5": int(mtf["m5_seq_len"]),
        "M15": int(mtf["m15_seq_len"]),
        "H1": int(mtf["h1_seq_len"]),
        "H4": int(mtf["h4_seq_len"]),
        "D1": int(mtf["d1_seq_len"]),
    }
    if (
        lineage["dataset_run_id"]
        != meta["run_lineage"]["dataset_run_id"]
        or lineage["per_tf_seq_lens"] != expected_normalization_seq_lens
        or lineage["mtf_feature_names_sha256"] != _mtf_names_sha
        or lineage["mtf_cache_manifest_path"]
        != mtf["shared_cache_manifest_path"]
        or lineage["mtf_cache_manifest_sha256"]
        != mtf["shared_cache_manifest_sha256"]
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_INPUT_NORMALIZATION_LINEAGE_MISMATCH]"
        )
    if (
        fit_proof.get("schema_version")
        != "entry_v10_train_input_normalization_population_proof_v1"
        or fit_proof.get("fit_scope") != "train_only"
        or int(fit_proof.get("train_decision_row_count", -1))
        != int(lineage["train_row_count"])
        or fit_proof.get("val_fit_row_count") != 0
        or fit_proof.get("test_fit_row_count") != 0
        or fit_proof.get("temporal_aliases_sha256")
        != input_normalization["temporal_aliases_sha256"]
        or observed_fit_proof_hash != expected_fit_proof_hash
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_INPUT_NORMALIZATION_FIT_PROOF_INVALID]"
        )
    mtf_populations = fit_proof.get("mtf_populations")
    if (
        not isinstance(mtf_populations, Mapping)
        or tuple(mtf_populations) != INPUT_NORMALIZATION_TFS
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_INPUT_NORMALIZATION_MTF_PROOF_INVALID]"
        )
    for tf in INPUT_NORMALIZATION_TFS:
        raw_population = mtf_populations[tf]
        window = lineage["per_tf_fit_windows"][tf]
        population_without_hash = (
            dict(raw_population) if isinstance(raw_population, Mapping) else {}
        )
        population_hash = str(
            population_without_hash.pop("selection_proof_sha256", "")
        )
        expected_population_hash = hashlib.sha256(
            json.dumps(
                population_without_hash,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if (
            population_hash != expected_population_hash
            or not isinstance(raw_population, Mapping)
            or raw_population.get("tf") != tf
            or int(raw_population.get("seq_len", -1))
            != int(lineage["per_tf_seq_lens"][tf])
            or any(
                raw_population.get(field) != window[field]
                for field in (
                    "left_index_inclusive",
                    "right_index_exclusive",
                    "selected_unique_row_count",
                    "selected_row_indices_sha256",
                    "selected_row_values_sha256",
                    "time_min_utc",
                    "time_max_utc",
                )
            )
        ):
            raise RuntimeError(
                "[ENTRY_BUNDLE_INPUT_NORMALIZATION_MTF_PROOF_INVALID] "
                f"tf={tf}"
            )

    _require_model_native_architecture_markers(meta)
    normalized_tf_scale = require_tf_input_scale_contract(
        meta["tf_input_scale"]
    )
    if any(
        float(normalized_tf_scale["init"][tf])
        != float(TF_INPUT_SCALE_NEUTRAL_INIT)
        for tf in TF_INPUT_SCALE_NAMES
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_TF_INPUT_SCALE_PRIOR_FORBIDDEN]"
        )

    forbidden_direction_artifacts = (
        "hierarchical_direction_composition",
        "residual_scale",
    )
    stale = [
        f"{owner}.{key}"
        for owner, payload in (("meta", meta), ("lock", lock))
        for key in forbidden_direction_artifacts
        if key in payload
    ]
    if stale:
        raise RuntimeError(
            "[ENTRY_BUNDLE_STALE_DIRECTION_ARTIFACT_FORBIDDEN] " + ",".join(stale)
        )

    _require_model_native_retired_head_contract(meta)
    state_contract = _require_mapping_field(meta, "model_native_state_contract", context="meta")
    if not state_contract:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_STATE_CONTRACT_MISSING]")
    run_lineage = _require_mapping_field(meta, "run_lineage", context="meta")
    try:
        run_lineage = require_training_run_lineage(run_lineage)
    except EntryModelNativeTrainingRunLineageError as exc:
        raise RuntimeError(
            f"[ENTRY_BUNDLE_MODEL_NATIVE_RUN_LINEAGE_INVALID] {exc}"
        ) from exc
    dataset_run_id = run_lineage["dataset_run_id"]
    if state_contract.get("entry_run_id") != dataset_run_id:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_DATASET_RUN_LINEAGE_MISMATCH]")
    try:
        require_prefreeze_test_seal_lineage_metadata(
            meta["prefreeze_test_seal_lineage"],
            expected_dataset_run_id=dataset_run_id,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[ENTRY_BUNDLE_PREFREEZE_TEST_SEAL_LINEAGE_INVALID] {exc}"
        ) from exc
    m1_binding = _require_mapping_field(
        meta,
        "m1_feature_surface_binding",
        context="meta",
    )
    if set(m1_binding) != {
        "parquet_path",
        "manifest_path",
        "dataset_run_id",
        "pair_generation_id",
        "parquet_sha256",
        "manifest_sha256",
        "feature_field_order_sha256",
    }:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_M1_FEATURE_BINDING_FIELDS_INVALID]"
        )
    ordered_signal_sha256 = hashlib.sha256(
        json.dumps(
            meta["ordered_signal_names"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    m1_hashes = (
        m1_binding.get("parquet_sha256"),
        m1_binding.get("manifest_sha256"),
        m1_binding.get("feature_field_order_sha256"),
    )
    if (
        m1_binding.get("dataset_run_id") != dataset_run_id
        or not isinstance(m1_binding.get("pair_generation_id"), str)
        or not m1_binding.get("pair_generation_id")
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in m1_hashes
        )
        or m1_binding.get("feature_field_order_sha256")
        != ordered_signal_sha256
        or any(
            not Path(str(m1_binding.get(key) or "")).is_absolute()
            for key in ("parquet_path", "manifest_path")
        )
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_M1_FEATURE_BINDING_INVALID]"
        )
    specialist = _require_mapping_field(meta, "specialist_fusion", context="meta")
    if specialist.get("enabled") is not True:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_FUSION_REQUIRED]")
    if specialist.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_MODE_INVALID]")
    context_routing = require_model_native_context_specialist_routing(
        meta["context_specialist_routing"],
        ordered_signal_names=meta["ordered_signal_names"],
        context="ENTRY_BUNDLE",
    )
    if specialist.get("context_routing") != context_routing:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_CONTEXT_ROUTING_SPLIT_BRAIN]"
        )
    input_indices = specialist.get("input_indices")
    if not isinstance(input_indices, Mapping) or not input_indices:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDICES_MISSING]")
    if list(input_indices) != list(_MODEL_NATIVE_REQUIRED_SPECIALISTS):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_ORDER_MISMATCH] "
            f"observed={list(input_indices)} expected={list(_MODEL_NATIVE_REQUIRED_SPECIALISTS)}"
        )
    seen_specialist_indices: set[int] = set()
    for specialist_name in _MODEL_NATIVE_REQUIRED_SPECIALISTS:
        indices = input_indices[specialist_name]
        if not isinstance(indices, list) or not indices:
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDICES_INVALID] {specialist_name}"
            )
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_TYPE_INVALID] {specialist_name}"
            )
        if indices != sorted(set(indices)):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_ORDER_INVALID] {specialist_name}"
            )
        if any(index < 0 or index >= MODEL_NATIVE_SIGNAL_DIM for index in indices):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_RANGE_INVALID] {specialist_name}"
            )
        overlap = seen_specialist_indices.intersection(indices)
        if overlap:
            raise RuntimeError(
                "[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_OVERLAP] "
                f"{specialist_name} overlap={sorted(overlap)}"
            )
        seen_specialist_indices.update(indices)
    expected_specialist_indices = set(range(MODEL_NATIVE_SIGNAL_DIM))
    if seen_specialist_indices != expected_specialist_indices:
        missing = sorted(expected_specialist_indices - seen_specialist_indices)
        unexpected = sorted(seen_specialist_indices - expected_specialist_indices)
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_INDEX_COVERAGE_INVALID] "
            f"missing={missing[:20]} total_missing={len(missing)} "
            f"unexpected={unexpected[:20]} total_unexpected={len(unexpected)}"
        )
    for alias in context_routing["temporal_alias_policy"]["aliases"]:
        owner = str(alias["specialist"])
        signal_index = int(alias["signal_index"])
        if signal_index not in input_indices.get(owner, []):
            raise RuntimeError(
                "[ENTRY_BUNDLE_MODEL_NATIVE_CONTEXT_ALIAS_OWNER_MISMATCH] "
                f"field={alias['signal_field']} index={signal_index} "
                f"owner={owner}"
            )
    if list(specialist.get("trainable_specialists") or []) != list(
        _MODEL_NATIVE_REQUIRED_SPECIALISTS
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRAINABLE_SPECIALISTS_MISMATCH]")
    for key in ("num_layers", "fusion_scale", "cross_family_fusion_scale"):
        if key not in specialist:
            raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_METADATA_MISSING] {key}")
    if isinstance(specialist["num_layers"], bool) or int(specialist["num_layers"]) <= 0:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_NUM_LAYERS_INVALID]")
    if (
        not math.isfinite(float(specialist["fusion_scale"]))
        or float(specialist["fusion_scale"]) <= 0.0
    ):
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_SPECIALIST_FUSION_SCALE_INVALID]")
    if (
        not math.isfinite(float(specialist["cross_family_fusion_scale"]))
        or float(specialist["cross_family_fusion_scale"]) <= 0.0
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_CROSS_FAMILY_FUSION_SCALE_INVALID]"
        )

    if "direction_calibration" in meta or "direction_calibration" in lock:
        raise RuntimeError(
            "[ENTRY_BUNDLE_ENTRY_Q_DIRECTION_CALIBRATION_FORBIDDEN]"
        )
    if "path_calibration" in meta or "path_calibration" in lock:
        raise RuntimeError(
            "[ENTRY_BUNDLE_RETIRED_PATH_CALIBRATION_FORBIDDEN]"
        )


def _require_profiled_unified_exit_gate_evidence(
    *,
    training_profile: str,
    exit_validation: Mapping[str, Any],
    full_trajectory_validation: Mapping[str, Any],
) -> None:
    """Apply Exit-gate admission at the profile that owns its authority.

    A bounded smoke records the raw gate diagnostics but deliberately admits a
    checkpoint on active-head liveness only. Its run lineage marks it as a
    non-candidate and every serving/candidate route requires a full-population
    candidate lineage. Requiring candidate-grade Exit gate variation while
    loading that same smoke bundle contradicts the smoke admission contract and
    makes the diagnostic bundle impossible to export. Candidate bundles keep
    both strict selected-checkpoint and full-trajectory gate proofs.
    """

    if training_profile == "smoke":
        return
    if training_profile != "candidate":
        raise RuntimeError("[ENTRY_BUNDLE_TRAINING_PROFILE_INVALID]")
    require_unified_exit_gate_evidence(
        exit_validation,
        expected_rows=int(exit_validation["unified_exit_population_rows"]),
        context="ENTRY_BUNDLE_SELECTED_CHECKPOINT",
    )
    require_unified_exit_gate_evidence(
        full_trajectory_validation,
        expected_rows=int(full_trajectory_validation["population_rows"]),
        context="ENTRY_BUNDLE_FULL_TRAJECTORY",
    )


def _require_model_native_architecture_markers(meta: Mapping[str, Any]) -> None:
    """Reject a train/serve architecture split before state loading.

    The active model owns positional encodings, but Regime FiLM is retired: it
    has no constructor path or state keys.  The trainer records both facts in
    bundle metadata so a loader cannot silently construct a different model.
    """

    if meta.get("enable_pos_enc") is not True:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_FULL_STACK_COMPONENT_REQUIRED] "
            "meta.enable_pos_enc"
        )
    if meta.get("enable_regime_film") is not False:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_RETIRED_REGIME_FILM_FORBIDDEN] "
            "meta.enable_regime_film"
        )


def _require_model_native_retired_head_contract(meta: Mapping[str, Any]) -> None:
    """Require the active event head and reject retired metadata-only heads."""

    if "hierarchical_entry_heads" in meta or "trendline_rail_head" in meta:
        raise RuntimeError("[ENTRY_BUNDLE_STALE_ENTRY_HEAD_METADATA_FORBIDDEN]")
    trendline = _require_mapping_field(meta, "trendline_event_head", context="meta")
    if trendline.get("enabled") is not True or int(trendline.get("output_dim", 0)) != 4:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRENDLINE_EVENT_CONTRACT_INVALID]")
    if trendline.get("hand_written_direction_pressure") is not False:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRENDLINE_DIRECTION_PRESSURE_FORBIDDEN]")
    if trendline.get("direction_mapping") != "representation_only_no_entry_authority":
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_TRENDLINE_ROLE_INVALID]")


def _require_model_native_state_head_contract(
    meta: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> None:
    active = frozenset(meta["train_recipe"]["active_heads"])
    expected_state_heads = active - _MODEL_NATIVE_METADATA_ONLY_COMPONENTS
    state_heads = {
        name
        for name, keys in _ENTRY_HEAD_STATE_KEYS.items()
        if keys.issubset(set(state_dict))
    }
    if state_heads != expected_state_heads:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_STATE_HEAD_CONTRACT_MISMATCH] "
            f"missing={sorted(expected_state_heads - state_heads)} "
            f"unexpected={sorted(state_heads - expected_state_heads)}"
        )


def _require_joint_task_weighting_state(
    meta: Mapping[str, Any],
    state_dict: Mapping[str, Any],
) -> None:
    contract = require_joint_task_weighting_metadata(
        meta["model_native_joint_task_weighting"],
        context="ENTRY_BUNDLE_STATE",
    )
    observed_keys = {
        key for key in state_dict if key.startswith("task_log_variances.")
    }
    if observed_keys != set(JOINT_TASK_STATE_KEYS):
        raise RuntimeError(
            "[ENTRY_BUNDLE_JOINT_TASK_WEIGHTING_STATE_KEYS_INVALID] "
            f"missing={sorted(set(JOINT_TASK_STATE_KEYS) - observed_keys)} "
            f"unexpected={sorted(observed_keys - set(JOINT_TASK_STATE_KEYS))}"
        )
    failures: list[str] = []
    selected = contract["selected_log_variances"]
    for task_name in JOINT_TASK_NAMES:
        key = f"task_log_variances.{task_name}"
        tensor = state_dict.get(key)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.numel() != 1
            or not bool(torch.isfinite(tensor).all().item())
        ):
            failures.append(f"{task_name}:state_invalid")
            continue
        if float(tensor.item()) != float(selected[task_name]):
            failures.append(f"{task_name}:metadata_state_split_brain")
    if failures:
        raise RuntimeError(
            "[ENTRY_BUNDLE_JOINT_TASK_WEIGHTING_STATE_INVALID] "
            + "; ".join(failures)
        )


_MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS: Dict[str, tuple[str, ...]] = {
    # These blocks are deliberately zero-initialized.  Merely finding their
    # keys in a state_dict therefore does not prove they ever joined the
    # learned decision path.  Require the data-dependent weight itself to move;
    # a nonzero bias alone is only a constant offset and cannot prove that
    # specialist, regime, or timeframe evidence was consumed.
    "specialist_fusion_output": (
        "specialist_out.weight",
    ),
    "cross_tf_output": (
        "cross_tf_out.weight",
    ),
    "cross_tf_gate": ("tf_gate_logits",),
    "family_tf_cooperation_output": (
        "family_tf_cooperation_out.weight",
    ),
}

_RETIRED_DIRECTION_STATE_PREFIXES = (
    "head_direction.",
    "head_mtf_direction.",
    "head_action_value.",
    "head_expectile_value.",
    "evidence_fusion_",
    "regime_film.",
    "head_public_trade.",
    "head_public_flat.",
    "head_public_side.",
    "hierarchical_ctx_prior_adapter.",
    "hierarchical_ctx_direction_calibration.",
)
_RETIRED_DIRECTION_STATE_KEYS = frozenset({"mtf_dir_scale"})


def _require_entry_q_state_contract(state_dict: Mapping[str, Any]) -> None:
    d_model = UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM
    expected_shapes = {
        "entry_q_joint_norm.weight": (4 * d_model,),
        "entry_q_joint_norm.bias": (4 * d_model,),
        "entry_q_joint_in.weight": (d_model, 4 * d_model),
        "entry_q_joint_in.bias": (d_model,),
        "head_entry_action_q.weight": (3, d_model),
        "head_entry_action_q.bias": (3,),
    }
    failures: list[str] = []
    for key, expected_shape in expected_shapes.items():
        value = state_dict.get(key)
        if not isinstance(value, torch.Tensor):
            failures.append(f"{key}:missing_or_non_tensor")
            continue
        if tuple(value.shape) != expected_shape:
            failures.append(
                f"{key}:shape={tuple(value.shape)} expected={expected_shape}"
            )
            continue
        if not bool(torch.isfinite(value).all().item()):
            failures.append(f"{key}:non_finite")
    for key in (
        "entry_q_joint_norm.weight",
        "entry_q_joint_in.weight",
        "head_entry_action_q.weight",
    ):
        value = state_dict.get(key)
        if isinstance(value, torch.Tensor) and not bool(torch.count_nonzero(value).item()):
            failures.append(f"{key}:all_zero")
    out_weight = state_dict.get("head_entry_action_q.weight")
    if isinstance(out_weight, torch.Tensor) and tuple(out_weight.shape) == (
        3,
        d_model,
    ):
        if any(
            bool(torch.equal(out_weight[i], out_weight[j]))
            for i in range(3)
            for j in range(i + 1, 3)
        ):
            failures.append("head_entry_action_q.weight:identical_action_rows")
    stale = sorted(
        key
        for key in state_dict
        if key in _RETIRED_DIRECTION_STATE_KEYS
        or key.startswith(_RETIRED_DIRECTION_STATE_PREFIXES)
    )
    if stale:
        failures.append(f"retired_direction_state={stale}")
    if failures:
        raise RuntimeError(
            "[ENTRY_BUNDLE_ENTRY_FITTED_Q_STATE_INVALID] "
            + " | ".join(failures)
        )


def _require_model_native_learned_component_liveness(
    state_dict: Mapping[str, Any],
    *,
    training_profile: str,
) -> None:
    """Reject structurally present full-stack blocks that remained pass-throughs.

    The bounded smoke is deliberately a tiny, deterministic compute sample.  It
    can establish that every routing matrix is present, finite and shaped for
    the full production surface, but it cannot require a rare event feature to
    occur in that sample.  Candidate training owns the population-wide
    requirement that every zero-initialized feature gate row must move.
    """

    if training_profile not in {"smoke", "candidate"}:
        raise RuntimeError("[ENTRY_BUNDLE_TRAINING_PROFILE_INVALID]")

    failures: list[str] = []
    for component, keys in _MODEL_NATIVE_ZERO_INIT_COMPONENT_GROUPS.items():
        missing = [key for key in keys if key not in state_dict]
        if missing:
            failures.append(f"{component}:missing={missing}")
            continue
        tensors = [state_dict[key] for key in keys]
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            failures.append(f"{component}:non_tensor_state")
            continue
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            failures.append(f"{component}:non_finite_state")
            continue
        if not any(bool(torch.count_nonzero(value).item()) for value in tensors):
            failures.append(f"{component}:zero_init_pass_through")
    mtf_routing = require_multi_tf_specialist_routing_v4(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    for timeframe in TF_INPUT_SCALE_NAMES:
        for specialist, indices in mtf_routing.items():
            key = (
                "mtf_feature_context_gate."
                f"{timeframe}__{specialist}.weight"
            )
            value = state_dict.get(key)
            expected_rows = len(indices)
            if not isinstance(value, torch.Tensor):
                failures.append(f"feature_tf_context_gate:{key}:missing_or_non_tensor")
                continue
            if (
                value.ndim != 2
                or int(value.shape[0]) != expected_rows
                or int(value.shape[1]) <= 0
            ):
                failures.append(
                    "feature_tf_context_gate:"
                    f"{key}:shape={tuple(value.shape)} expected_rows={expected_rows}"
                )
                continue
            if not bool(torch.isfinite(value).all().item()):
                failures.append(f"feature_tf_context_gate:{key}:non_finite")
                continue
            if training_profile == "candidate":
                dead_rows = torch.nonzero(
                    torch.count_nonzero(value, dim=1) == 0,
                    as_tuple=False,
                ).flatten()
                if int(dead_rows.numel()) > 0:
                    failures.append(
                        "feature_tf_context_gate:"
                        f"{key}:zero_init_pass_through_rows={dead_rows.tolist()}"
                    )
    _require_entry_q_state_contract(state_dict)
    if failures:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_LEARNED_COMPONENT_LIVENESS_INVALID] "
            + " | ".join(failures)
        )


def _infer_entry_bundle_capabilities(meta: Dict[str, Any], state_dict: Dict[str, Any]) -> Dict[str, Any]:
    declared_heads = {str(h) for h in meta["train_recipe"]["active_heads"]}
    state_heads = {
        head_name
        for head_name, required_keys in _ENTRY_HEAD_STATE_KEYS.items()
        if required_keys.issubset(set(state_dict.keys()))
    }
    missing_declared = declared_heads - state_heads - _MODEL_NATIVE_METADATA_ONLY_COMPONENTS
    if missing_declared:
        raise RuntimeError(
            f"[ENTRY_BUNDLE_CAPABILITY_MISMATCH] metadata declares unsupported heads: {sorted(missing_declared)}"
        )
    supported_heads = {"entry_action_q"} | declared_heads
    unsupported_heads = sorted(set(_ENTRY_HEAD_STATE_KEYS) - supported_heads)
    return {
        "supported_heads": sorted(supported_heads),
        "unsupported_heads": unsupported_heads,
        "declared_active_heads": sorted(declared_heads),
        "state_dict_heads": sorted(state_heads),
        "supports_context_features": True,
    }


def _require_current_prefreeze_test_seal_lineage(
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate only the immutable seal event, never sealed TEST artifacts."""

    declared = meta.get("prefreeze_test_seal_lineage")
    run_lineage = meta.get("run_lineage")
    if not isinstance(declared, Mapping) or not isinstance(run_lineage, Mapping):
        raise RuntimeError("[ENTRY_BUNDLE_PREFREEZE_TEST_SEAL_LINEAGE_MISSING]")
    try:
        normalized = require_prefreeze_test_seal_lineage_metadata(
            declared,
            expected_dataset_run_id=str(run_lineage.get("dataset_run_id") or ""),
        )
        seal_event = normalized["seal_event"]
        observed = require_prefreeze_test_seal_lineage(
            seal_event["path"],
            seal_event["sha256"],
            expected_dataset_run_id=normalized["dataset_run_id"],
            expected_dataset_dir=normalized["dataset_dir"],
        )
    except (PrefreezeTestSealLineageError, OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"[ENTRY_BUNDLE_PREFREEZE_TEST_SEAL_REVALIDATION_FAILED] {exc}"
        ) from exc
    if observed != normalized:
        raise RuntimeError(
            "[ENTRY_BUNDLE_PREFREEZE_TEST_SEAL_EVENT_LINEAGE_MISMATCH]"
        )
    return observed


def load_entry_v10_ctx_bundle(
    *,
    bundle_dir: str | Path,
    device: str,
) -> EntryV10Bundle:

    supplied_bundle_dir = Path(bundle_dir).expanduser()
    if (
        not supplied_bundle_dir.is_absolute()
        or any(
            component.is_symlink()
            for component in (
                supplied_bundle_dir,
                *supplied_bundle_dir.parents,
            )
        )
    ):
        raise RuntimeError("[ENTRY_BUNDLE_PATH_NOT_CANONICAL]")
    try:
        bd = supplied_bundle_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("[ENTRY_BUNDLE_PATH_NOT_CANONICAL]") from exc
    if bd != supplied_bundle_dir:
        raise RuntimeError("[ENTRY_BUNDLE_PATH_NOT_CANONICAL]")
    _guard_required(bd, "bundle_dir")

    lock_path = bd / "MASTER_TRANSFORMER_LOCK.json"
    meta_path = bd / "bundle_metadata.json"
    state_path = bd / "model_state_dict.pt"

    _guard_required(lock_path, "MASTER_TRANSFORMER_LOCK.json")
    _guard_required(meta_path, "bundle_metadata.json")
    _guard_required(state_path, "model_state_dict.pt")

    lock_bytes = lock_path.read_bytes()
    meta_bytes = meta_path.read_bytes()
    try:
        lock = json.loads(lock_bytes.decode("utf-8"))
        meta = json.loads(meta_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_JSON_INVALID]") from exc
    for owner, payload in (("META", meta), ("LOCK", lock)):
        require_entry_exit_production_architecture(
            _bundle_architecture_observation(payload),
            context=f"ENTRY_V10_BUNDLE_{owner}_PREWEIGHT",
        )

    committed_bundle = require_bundle_commit_manifest(bd)
    state_bytes = state_path.read_bytes()
    for name, payload in (
        ("MASTER_TRANSFORMER_LOCK.json", lock_bytes),
        ("bundle_metadata.json", meta_bytes),
        ("model_state_dict.pt", state_bytes),
    ):
        binding = committed_bundle["artifacts"][name]
        if (
            int(binding["size_bytes"]) != len(payload)
            or str(binding["sha256"])
            != hashlib.sha256(payload).hexdigest()
        ):
            raise RuntimeError(
                "[ENTRY_BUNDLE_COMMITTED_BYTES_MISMATCH] "
                f"name={name}"
            )
    _require_exact_model_native_bundle_metadata(meta, lock)
    _require_current_prefreeze_test_seal_lineage(meta)
    seq_input_dim = int(meta["seq_input_dim"])
    snap_input_dim = int(meta["snap_input_dim"])
    seq_len = int(meta["seq_len"])
    ctx_cont_dim = int(meta["ctx_cont_dim"])
    ctx_cat_dim = int(meta["ctx_cat_dim"])
    if lock.get("model_path_relative") != state_path.name:
        raise RuntimeError("[ENTRY_BUNDLE_MODEL_NATIVE_STATE_PATH_BINDING_INVALID]")
    observed_state_sha256 = hashlib.sha256(state_bytes).hexdigest()
    lock_state_sha256 = lock.get("model_sha256")
    meta_state_sha256 = meta.get("state_dict_sha256")
    if lock_state_sha256 != observed_state_sha256 or meta_state_sha256 != observed_state_sha256:
        raise RuntimeError(
            "[ENTRY_BUNDLE_MODEL_NATIVE_STATE_SHA256_MISMATCH] "
            f"observed={observed_state_sha256} lock={lock_state_sha256!r} "
            f"meta={meta_state_sha256!r}"
        )

    dev = _resolve_device(device)
    logging.getLogger(__name__).info(
        "[ENTRY_BUNDLE_LOAD_PROOF] ctx_cont_dim=%d ctx_cat_dim=%d seq_input_dim=%d snap_input_dim=%d",
        ctx_cont_dim,
        ctx_cat_dim,
        seq_input_dim,
        snap_input_dim,
    )

    # Import real model
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
        EntryV10CtxHybridTransformer,
    )

    # Reconstruct the exact multi-TF contract declared by the validated bundle.
    mtf_meta = meta["multi_tf"]
    _m15_seq_dim = int(mtf_meta["m15_seq_dim"])
    _h1_seq_dim = int(mtf_meta["h1_seq_dim"])
    _h4_seq_dim = int(mtf_meta["h4_seq_dim"])
    _d1_seq_dim = int(mtf_meta["d1_seq_dim"])
    _m15_seq_len = int(mtf_meta["m15_seq_len"])
    _h1_seq_len = int(mtf_meta["h1_seq_len"])
    _h4_seq_len = int(mtf_meta["h4_seq_len"])
    _d1_seq_len = int(mtf_meta["d1_seq_len"])
    mtf_scale = float(mtf_meta["multi_tf_scale"])
    mtf_num_layers = int(mtf_meta["multi_tf_num_layers"])
    _m5_seq_dim = int(mtf_meta["m5_seq_dim"])
    _m5_seq_len = int(mtf_meta["m5_seq_len"])
    state_dict_preview = torch.load(
        io.BytesIO(state_bytes),
        map_location="cpu",
        weights_only=True,
    )
    _require_model_native_state_head_contract(meta, state_dict_preview)
    _require_joint_task_weighting_state(meta, state_dict_preview)
    _require_model_native_learned_component_liveness(
        state_dict_preview,
        training_profile=str(meta["run_lineage"]["training_profile"]),
    )
    required_tf_scale_keys = {
        f"tf_input_scale_{tf}" for tf in TF_INPUT_SCALE_NAMES
    }
    missing_tf_scale_keys = sorted(required_tf_scale_keys - set(state_dict_preview))
    if missing_tf_scale_keys:
        raise RuntimeError(f"[ENTRY_BUNDLE_MODEL_NATIVE_TF_INPUT_SCALE_STATE_MISSING] {missing_tf_scale_keys}")
    require_tf_input_scale_state(meta["tf_input_scale"], state_dict_preview)

    _specialist_cfg = meta["specialist_fusion"]
    _indices = _specialist_cfg["input_indices"]
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        dropout=float(meta["dropout"]),
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        m15_seq_dim=_m15_seq_dim, h1_seq_dim=_h1_seq_dim, h4_seq_dim=_h4_seq_dim, d1_seq_dim=_d1_seq_dim,
        m15_seq_len=_m15_seq_len, h1_seq_len=_h1_seq_len,
        h4_seq_len=_h4_seq_len, d1_seq_len=_d1_seq_len,
        m5_seq_dim=_m5_seq_dim, m5_seq_len=_m5_seq_len,
        multi_tf_num_layers=mtf_num_layers,
        multi_tf_scale=mtf_scale,
        specialist_input_indices={
            str(k): list(v) for k, v in _indices.items()
        },
        specialist_ctx_cont_indices={
            str(k): list(v)
            for k, v in _specialist_cfg["context_routing"][
                "ctx_cont_indices"
            ].items()
        },
        specialist_ctx_cont_nominal_indices={
            str(k): list(v)
            for k, v in _specialist_cfg["context_routing"][
                "ctx_cont_nominal_indices"
            ].items()
        },
        specialist_ctx_cat_indices={
            str(k): list(v)
            for k, v in _specialist_cfg["context_routing"][
                "ctx_cat_indices"
            ].items()
        },
        multi_tf_specialist_input_indices={
            str(name): list(indices)
            for name, indices in require_multi_tf_specialist_routing_v4(
                mtf_meta["feature_names"]
            ).items()
        },
        temporal_alias_signal_indices=list(
            _specialist_cfg["context_routing"]["temporal_alias_policy"][
                "signal_indices"
            ]
        ),
        temporal_alias_ctx_cont_indices=list(
            _specialist_cfg["context_routing"]["temporal_alias_policy"][
                "ctx_cont_indices"
            ]
        ),
        specialist_num_layers=int(_specialist_cfg["num_layers"]),
        specialist_fusion_scale=float(_specialist_cfg["fusion_scale"]),
        cross_family_fusion_scale=float(
            _specialist_cfg["cross_family_fusion_scale"]
        ),
        input_normalization=meta["input_normalization"],
    ).to(dev)
    if int(model.cfg.d_model) != UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM:
        raise RuntimeError(
            "[ENTRY_BUNDLE_UNIFIED_EXIT_REPRESENTATION_DIM_INVALID] "
            f"observed={model.cfg.d_model!r} "
            f"expected={UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM}"
        )

    state_dict = state_dict_preview
    model.load_state_dict(state_dict, strict=True)
    model.require_input_normalization_state()
    model.eval()
    capabilities = _infer_entry_bundle_capabilities(meta, state_dict)
    logging.getLogger(__name__).info(
        "[ENTRY_BUNDLE_CAPABILITIES] supported_heads=%s declared_active_heads=%s unsupported_heads=%s",
        capabilities["supported_heads"],
        capabilities["declared_active_heads"],
        capabilities["unsupported_heads"],
    )

    bundle = EntryV10Bundle(
        bundle_dir=str(bd),
        bundle_sha256=str(committed_bundle["commit_sha256"]),
        device=dev,
        transformer_model=model,
        metadata={
            **meta,
            "model_variant": "v10_ctx",
            "capabilities": capabilities,
        },
        transformer_config={
            "seq_input_dim": seq_input_dim,
            "snap_input_dim": snap_input_dim,
            "seq_len": seq_len,
        },
        capabilities=capabilities,
    )

    if require_bundle_commit_manifest(bd) != committed_bundle:
        raise RuntimeError("[ENTRY_BUNDLE_CHANGED_DURING_LOAD]")
    return bundle
