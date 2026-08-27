#!/usr/bin/env python3
"""
Canonical ENTRY_V10_CTX trainer.

ONE UNIVERSE (STRICT):
- Signal bridge: versioned Entry signal bridge.
- Context: exact ctx_cont/ctx_cat order from entry_model_native_signal_v1.
- Joint fitted-Q Entry/Exit training from frozen TRAIN target snapshots.
- No legacy
- No fallback
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
import logging
import math
import mmap
import os
import random
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler

# Canonical context ordering; exact model-native dimensions are verified below.
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_state_v2 import (
    validate_state_contract_metadata_v2,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS as BUNDLE_COMMIT_CORE_ARTIFACTS,
    publish_bundle_directory_noreplace,
    write_bundle_commit_manifest,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    canonical_json_sha256,
    validate_xau_tape_provenance_v1,
)
from gx1.contracts.entry_causal_m1_position_size_target_policy_v1 import (
    require_causal_m1_position_size_target_manifest_binding,
)
from gx1.contracts.entry_position_size_target_policy_v1 import (
    require_entry_position_size_target_manifest_binding,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    training_objective_contract_metadata,
)
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    joint_task_weighting_metadata,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES,
    require_model_native_recipe_env,
    resolve_weight_ema_decay,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PrefreezeTestSealLineageError,
    require_prefreeze_test_seal_lineage,
    require_prefreeze_test_seal_lineage_metadata,
)
from gx1.contracts.entry_model_native_readiness_v1 import MODEL_NATIVE_ACTIVE_HEADS
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_TARGET_COLUMNS,
    MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
    MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS,
    MODEL_NATIVE_TIMING_TARGET_COLUMNS,
    MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS,
    require_model_native_aux_target_emission_contract as _require_model_native_aux_target_emission_contract,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    NEUTRAL_EFFECTIVE_INIT as TF_INPUT_SCALE_NEUTRAL_INIT,
    TF_NAMES as TF_INPUT_SCALE_NAMES,
    build_tf_input_scale_contract,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UnifiedExitLifecycleCorpus,
    UnifiedExitLifecycleSplit,
    require_unified_exit_lifecycle_authority_evidence,
)
from gx1.contracts.unified_exit_episode_pack_v1 import (
    UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION,
    require_unified_exit_episode_pack,
    seal_unified_exit_episode_pack,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    build_unified_exit_fitted_q_targets,
    replay_unified_exit_fitted_q_policy,
    unified_exit_first_state_side_values,
    unified_exit_fitted_q_contract,
)
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    build_entry_fitted_q_targets,
    entry_fill_binding_sha256,
    entry_fitted_q_contract,
    entry_fitted_q_production_economics_readiness,
    require_entry_fitted_q_iteration_state,
)
from gx1.contracts.unified_exit_gate_evidence_v1 import (
    COOPERATION_GATE_WIDTHS as UNIFIED_EXIT_GATE_WIDTHS,
    FEATURE_GATE_MIN_STD as UNIFIED_EXIT_FEATURE_GATE_MIN_STD,
    FEATURE_TF_GATE_SHAPE as UNIFIED_EXIT_FEATURE_GATE_SHAPE,
    SCHEMA_VERSION as UNIFIED_EXIT_GATE_EVIDENCE_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_input_influence_v1 import (
    CATEGORICAL_DELTA_EPSILON as UNIFIED_EXIT_INFLUENCE_CAT_EPSILON,
    COMPARISON_SURFACE as UNIFIED_EXIT_INFLUENCE_SURFACE,
    NUMERIC_GRADIENT_EPSILON as UNIFIED_EXIT_INFLUENCE_GRAD_EPSILON,
    SAMPLE_COUNT as UNIFIED_EXIT_INFLUENCE_SAMPLE_COUNT,
    SAMPLING_CONTRACT as UNIFIED_EXIT_INFLUENCE_SAMPLING_CONTRACT,
    SCHEMA_VERSION as UNIFIED_EXIT_INPUT_INFLUENCE_SCHEMA_VERSION,
    SIDE_ROWS as UNIFIED_EXIT_INFLUENCE_SIDE_ROWS,
    SPLIT as UNIFIED_EXIT_INFLUENCE_SPLIT,
    canonical_json_sha256 as unified_exit_influence_sha256,
    require_unified_exit_input_influence,
    unified_exit_input_influence_layout,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_EXIT_RESOLUTION_RATIO,
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
from gx1.contracts.entry_sequence_source_reconstruction_v1 import (
    feature_surface_binding_from_split_manifest,
    require_sequence_source_reconstruction_audit,
)
from gx1.models.entry_v10.entry_v10_input_normalization import (
    TrainNormalizationArtifacts,
    fit_entry_v10_train_input_normalization,
    require_dataset_manifest_multi_tf_cache_binding,
    require_multi_tf_v4_cache_binding_files,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
    MODEL_ARCHITECTURE_SCHEMA_VERSION,
    MODEL_OUTPUT_SCHEMA_VERSION,
    TRAIN_ACTIVATION_CHECKPOINT_POLICY,
    DIP_HEAD_DIM,
    FORECAST_HEAD_DIM,
    TIMING_HEAD_DIM,
    TAIL_RISK_HEAD_DIM,
    VOL_FORECAST_HEAD_DIM,
    DIP_DIRECTIONS, DIP_HORIZONS, DIP_TARGETS, FORECAST_HORIZONS,
    TIMING_DIRECTIONS, TIMING_HORIZONS, TIMING_TARGETS,
    TAIL_RISK_DIRECTIONS, TAIL_RISK_HORIZONS, TAIL_RISK_QUANTILE,
    VOL_FORECAST_HORIZONS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_MODEL_REPRESENTATION_KEY,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_FEATURE_DIM,
    model_direction_decision_contract_metadata,
    unified_entry_exit_contract_metadata,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION,
    require_multi_tf_specialist_routing_v4,
    require_model_native_context_specialist_routing,
    required_training_specialists_for_mode,
    require_model_native_specialist_contract_mode,
    specialist_model_contract_for_mode,
)
from gx1.features.htf_features import (
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_SHIFT,
    MULTI_TF_TIMEFRAMES,
)


_DIP_TARGET_COLS = MODEL_NATIVE_DIP_TARGET_COLUMNS
_FORECAST_TARGET_COLS = MODEL_NATIVE_FORECAST_TARGET_COLUMNS
_TIMING_TARGET_COLS = MODEL_NATIVE_TIMING_TARGET_COLUMNS
_TAIL_RISK_TARGET_COLS = MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS
_VOL_FORECAST_TARGET_COLS = MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS

# Exact emitted target surface for the five active forward-path auxiliary heads.
_DIP_FORECAST_TARGET_COLS = (
    _DIP_TARGET_COLS
    + _FORECAST_TARGET_COLS
    + _TIMING_TARGET_COLS
    + _TAIL_RISK_TARGET_COLS
    + _VOL_FORECAST_TARGET_COLS
)
def _require_active_aux_head_prediction(
    out: dict,
    batch: dict,
    *,
    output_name: str,
    target_names: tuple[str, ...],
) -> torch.Tensor:
    prediction = out.get(output_name)
    missing_targets = [name for name in target_names if name not in batch]
    if not isinstance(prediction, torch.Tensor):
        raise RuntimeError(
            f"[ENTRY_MODEL_NATIVE_ACTIVE_HEAD_MISSING] output={output_name}"
        )
    if missing_targets:
        raise RuntimeError(
            "[ENTRY_MODEL_NATIVE_ACTIVE_HEAD_TARGET_MISSING] "
            f"output={output_name} missing={missing_targets}"
        )
    return prediction


def _masked_position_size_mse(
    position_size_logit: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE only on immutable tradable selected-side sizing rows."""

    prediction = torch.sigmoid(position_size_logit).reshape(-1)
    expected = target.reshape(-1).to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    observed_mask = mask.reshape(-1).to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if prediction.shape != expected.shape or observed_mask.shape != expected.shape:
        raise RuntimeError("[ENTRY_POSITION_SIZE_MASK_SHAPE_INVALID]")
    if not bool(torch.isfinite(expected).all()) or bool(
        ((expected < 0.0) | (expected > 1.0)).any()
    ):
        raise RuntimeError("[ENTRY_POSITION_SIZE_TARGET_INVALID]")
    if not bool(torch.isfinite(observed_mask).all()) or bool(
        ((observed_mask != 0.0) & (observed_mask != 1.0)).any()
    ):
        raise RuntimeError("[ENTRY_POSITION_SIZE_MASK_INVALID]")
    active = observed_mask == 1.0
    if bool(active.any()):
        return torch.nn.functional.mse_loss(prediction[active], expected[active])
    # Preserve the graph while contributing exactly zero on an all-FLAT batch.
    return position_size_logit.sum() * 0.0


def _joint_task_loss(
    model: nn.Module,
    task_losses: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the sole learned multi-task scalarization to nonempty tasks.

    A task omitted from ``task_losses`` contributes neither its raw loss nor
    ``+s_i``. This prevents an empty supervision mask from updating a task
    weight without outcome evidence.
    """

    log_variances = getattr(model, "task_log_variances", None)
    if not isinstance(log_variances, nn.ParameterDict):
        raise RuntimeError("[ENTRY_JOINT_TASK_WEIGHTING_PARAMETERS_MISSING]")
    if set(log_variances) != set(JOINT_TASK_NAMES):
        raise RuntimeError("[ENTRY_JOINT_TASK_WEIGHTING_PARAMETER_SURFACE_INVALID]")
    unexpected = sorted(set(task_losses) - set(JOINT_TASK_NAMES))
    if unexpected:
        raise RuntimeError(
            f"[ENTRY_JOINT_TASK_WEIGHTING_TASK_INVALID] unexpected={unexpected}"
        )
    if not task_losses:
        raise RuntimeError("[ENTRY_JOINT_TASK_WEIGHTING_EMPTY_BATCH]")
    total: Optional[torch.Tensor] = None
    stats: dict[str, float] = {}
    for task_name in JOINT_TASK_NAMES:
        if task_name not in task_losses:
            continue
        raw_loss = task_losses[task_name]
        if not isinstance(raw_loss, torch.Tensor) or raw_loss.numel() != 1:
            raise RuntimeError(
                f"[ENTRY_JOINT_TASK_LOSS_SHAPE_INVALID] task={task_name}"
            )
        if not bool(torch.isfinite(raw_loss).all().item()):
            raise RuntimeError(
                f"[ENTRY_JOINT_TASK_LOSS_NONFINITE] task={task_name}"
            )
        log_variance = log_variances[task_name]
        weighted = torch.exp(-log_variance) * raw_loss + log_variance
        total = weighted if total is None else total + weighted
        stats[f"joint_task_raw_loss_{task_name}"] = float(
            raw_loss.detach().cpu().item()
        )
        stats[f"joint_task_log_variance_{task_name}"] = float(
            log_variance.detach().cpu().item()
        )
        stats[f"joint_task_effective_precision_{task_name}"] = float(
            torch.exp(-log_variance.detach()).cpu().item()
        )
    if total is None:
        raise RuntimeError("[ENTRY_JOINT_TASK_WEIGHTING_NO_ACTIVE_TASK]")
    return total, stats


def _observe_joint_task_weight_gradients(
    model: nn.Module,
    observed: dict[str, bool],
) -> None:
    log_variances = getattr(model, "task_log_variances", None)
    if not isinstance(log_variances, nn.ParameterDict):
        raise RuntimeError("[ENTRY_JOINT_TASK_WEIGHTING_PARAMETERS_MISSING]")
    for task_name in JOINT_TASK_NAMES:
        gradient = log_variances[task_name].grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError(
                f"[ENTRY_JOINT_TASK_WEIGHT_GRADIENT_NONFINITE] task={task_name}"
            )
        if bool((gradient.detach().abs() > 0.0).any().item()):
            observed[task_name] = True


def dip_forecast_task_losses(
    out: dict,
    batch: dict,
    device,
) -> dict[str, torch.Tensor]:
    """Return separate genuine losses in their labels' native units."""
    dip_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="dip_pred",
        target_names=_DIP_TARGET_COLS,
    )
    tgts, qs = [], []
    for d in DIP_DIRECTIONS:
        for K in DIP_HORIZONS:
            for tgt in DIP_TARGETS:
                if tgt.startswith("recovery"):
                    tgts.append(batch[f"y_dip_mfe_{d}_K{K}"])
                    qs.append(0.5)
                else:  # dip_p50 / dip_p90
                    tgts.append(batch[f"y_dip_mae_{d}_K{K}"])
                    qs.append(0.9 if "p90" in tgt else 0.5)
    tgt = torch.stack(tgts, dim=1).to(device).float()
    q = torch.tensor(qs, device=device, dtype=tgt.dtype).view(1, -1)
    err = tgt - dip_pred.float()
    dip_loss = torch.maximum(q * err, (q - 1.0) * err).mean()
    fc_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="forecast_pred",
        target_names=_FORECAST_TARGET_COLS,
    )
    fc_tgt = torch.stack(
        [batch[f"y_forecast_ret_K{K}"] for K in FORECAST_HORIZONS],
        dim=1,
    ).to(device).float()
    forecast_loss = torch.nn.functional.l1_loss(
        fc_pred.float(),
        fc_tgt,
    )
    # ── dip-timing head (12, exact L1) — WHEN the dip bottoms / favorable peak ─
    timing_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="timing_pred",
        target_names=_TIMING_TARGET_COLS,
    )
    t_tgts = []
    for d in TIMING_DIRECTIONS:
        for K in TIMING_HORIZONS:
            for tgt in TIMING_TARGETS:
                t_tgts.append(batch[f"y_{tgt}_{d}_K{K}"])
    t_tgt = torch.stack(t_tgts, dim=1).to(device).float()          # (B, 12)
    timing_loss = torch.nn.functional.l1_loss(timing_pred.float(), t_tgt)
    # ── tail-risk head (6, pinball q=0.9) — worst adverse over full horizon ─────
    tail_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="tail_risk_pred",
        target_names=_TAIL_RISK_TARGET_COLS,
    )
    tail_tgts = [batch[f"y_tail_mae_{d}_K{K}"]
                 for d in TAIL_RISK_DIRECTIONS for K in TAIL_RISK_HORIZONS]
    tail_tgt = torch.stack(tail_tgts, dim=1).to(device).float()
    q = float(TAIL_RISK_QUANTILE)
    err = tail_tgt - tail_pred.float()
    tail_loss = torch.maximum(q * err, (q - 1.0) * err).mean()
    # ── vol-forecast head (3, exact L1) — forward realized vol (bps) ────────────
    vol_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="vol_forecast_pred",
        target_names=_VOL_FORECAST_TARGET_COLS,
    )
    vol_tgt = torch.stack(
        [batch[f"y_vol_fwd_K{K}"] for K in VOL_FORECAST_HORIZONS],
        dim=1,
    ).to(device).float()
    vol_loss = torch.nn.functional.l1_loss(
        vol_pred.float(),
        vol_tgt,
    )
    return {
        "dip_bps": dip_loss,
        "forecast_return_bps": forecast_loss,
        "dip_timing_fraction": timing_loss,
        "tail_risk_bps": tail_loss,
        "forward_volatility_bps": vol_loss,
    }


_MODEL_NATIVE_ACTIVE_CORE_TARGET_COLS = (
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
)
_MODEL_NATIVE_ACTIVE_EVENT_TARGET_COLS = (
    # V29 stage 2: forward-realized registry line-hold labels replace the
    # retired same-bar tautologies; the two *_mask columns carry the
    # touch-event loss mask (the y_side_mask pattern).
    "y_line_support_touch_held",
    "y_line_support_touch_mask",
    "y_line_resistance_touch_held",
    "y_line_resistance_touch_mask",
    "y_countertrend_short_trap",
    "y_countertrend_long_trap",
)
_MODEL_NATIVE_ACTIVE_TARGET_COLS = (
    _MODEL_NATIVE_ACTIVE_CORE_TARGET_COLS
    + (
        "y_position_size_target",
        "y_position_size_mask",
    )
    + _MODEL_NATIVE_ACTIVE_EVENT_TARGET_COLS
    + _DIP_FORECAST_TARGET_COLS
)
_MODEL_NATIVE_BINARY_TARGET_COLS = (
    "y_position_size_mask",
) + _MODEL_NATIVE_ACTIVE_EVENT_TARGET_COLS
_MODEL_NATIVE_UNIT_INTERVAL_TARGET_COLS = (
    "y_position_size_target",
) + _TIMING_TARGET_COLS
# Spread-aware dip-MFE is a signed forward outcome and must stay signed
# through validation and both losses (rule 16); only the dip-MAE half of the
# dip surface is a non-negative adverse magnitude.
_MODEL_NATIVE_NONNEGATIVE_TARGET_COLS = (
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
) + tuple(MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS) + _TAIL_RISK_TARGET_COLS + _VOL_FORECAST_TARGET_COLS


def _model_native_active_target_failures(
    split_name: str,
    df: pd.DataFrame,
) -> list[str]:
    failures: list[str] = []
    if df.empty:
        return [f"{split_name} model-native active target frame is empty"]
    duplicate_columns = sorted(
        {str(name) for name in df.columns[df.columns.duplicated()].tolist()}
    )
    if duplicate_columns:
        return [
            f"{split_name} model-native target frame has duplicate columns: {duplicate_columns}"
        ]
    missing = [name for name in _MODEL_NATIVE_ACTIVE_TARGET_COLS if name not in df.columns]
    if missing:
        return [
            f"{split_name} missing model-native active target columns: {missing}; "
            "rebuild the dataset because target defaults and aliases are forbidden"
        ]

    numeric: Dict[str, np.ndarray] = {}
    for name in _MODEL_NATIVE_ACTIVE_TARGET_COLS:
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
        numeric[name] = values
        if not np.isfinite(values).all():
            failures.append(f"{split_name} model-native target {name} contains non-finite values")
    if failures:
        return failures

    for name in _MODEL_NATIVE_BINARY_TARGET_COLS:
        values = numeric[name]
        if bool((~np.isin(values, [0.0, 1.0])).any()):
            failures.append(f"{split_name} model-native binary target {name} is outside {{0,1}}")
    for name in _MODEL_NATIVE_UNIT_INTERVAL_TARGET_COLS:
        values = numeric[name]
        if bool(((values < 0.0) | (values > 1.0)).any()):
            failures.append(f"{split_name} model-native target {name} is outside [0,1]")
    for name in _MODEL_NATIVE_NONNEGATIVE_TARGET_COLS:
        if bool((numeric[name] < 0.0).any()):
            failures.append(f"{split_name} model-native target {name} contains negative values")
    return failures


_MODEL_NATIVE_COOPERATION_GATE_WIDTHS = {
    "specialist_gate": 8,
    "tf_gate": ENTRY_MTF_CONTEXT_COUNT,
    "family_tf_cooperation_gate": (
        ENTRY_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS)
    ),
}
_MODEL_NATIVE_FEATURE_TF_GATE_SHAPE = (
    ENTRY_MTF_CONTEXT_COUNT,
    MULTI_TF_FEATURE_COUNT_V4,
)
_UNIFIED_EXIT_COOPERATION_GATE_WIDTHS = UNIFIED_EXIT_GATE_WIDTHS
_UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE = UNIFIED_EXIT_FEATURE_GATE_SHAPE
_UNIFIED_EXIT_GATE_OUTPUT_NAMES = {
    "specialist_gate": "exit_specialist_gate",
    "tf_gate": "exit_tf_gate",
    "family_tf_cooperation_gate": "exit_family_tf_cooperation_gate",
    "family_tf_feature_gate": "exit_family_tf_feature_gate",
}
_MODEL_NATIVE_FEATURE_TF_GATE_MIN_STD = UNIFIED_EXIT_FEATURE_GATE_MIN_STD


_MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS = {
    "entry_action_q_bps": 3,
    "entry_q_joint_hidden": UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    "position_size_logit": 1,
    "dip_pred": DIP_HEAD_DIM,
    "forecast_pred": FORECAST_HEAD_DIM,
    "timing_pred": TIMING_HEAD_DIM,
    "tail_risk_pred": TAIL_RISK_HEAD_DIM,
    "vol_forecast_pred": VOL_FORECAST_HEAD_DIM,
    "side_mae_bps": 2,
    "trendline_event_logits": 4,
    **_MODEL_NATIVE_COOPERATION_GATE_WIDTHS,
}

_ACTIVE_HEAD_OUTPUT_COMPONENTS = {
    "entry_action_q": ("entry_action_q_bps",),
    "position_size": ("position_size_logit",),
    "dip": ("dip_pred",),
    "forecast": ("forecast_pred",),
    "timing": ("timing_pred",),
    "tail_risk": ("tail_risk_pred",),
    "vol_forecast": ("vol_forecast_pred",),
    "side_mae": ("side_mae_bps",),
    "trendline_event": ("trendline_event_logits",),
}
_ACTIVE_HEAD_TARGET_COMPONENTS = dict(_ACTIVE_HEAD_OUTPUT_COMPONENTS)
_ACTIVE_HEAD_COMPONENT_WIDTHS = {
    component: _MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS[component]
    for components in _ACTIVE_HEAD_OUTPUT_COMPONENTS.values()
    for component in components
}
_ACTIVE_HEAD_DIAGNOSTIC_MIN_ROWS = 16
_ACTIVE_HEAD_DIAGNOSTIC_LIVENESS_EPS = 1e-8
_ACTIVE_HEAD_STRUCTURAL_CONSTANT_COLUMNS = {
    "entry_action_q_bps": frozenset({2}),
}
_ACTIVE_HEAD_DERIVED_TARGET_COMPONENTS = frozenset()
_ACTIVE_HEAD_ACTION_AUTHORITY_NONE = frozenset(
    set(_ACTIVE_HEAD_OUTPUT_COMPONENTS) - {"entry_action_q"}
)


def _model_native_active_output_head_failures(out: Dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for name, expected_width in _MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS.items():
        value = out.get(name)
        if not isinstance(value, torch.Tensor):
            failures.append(f"missing tensor output {name}")
            continue
        if value.ndim != 2 or int(value.shape[1]) != int(expected_width):
            failures.append(
                f"output {name} shape={tuple(value.shape)} expected=(batch,{expected_width})"
            )
        elif not bool(torch.isfinite(value).all().item()):
            failures.append(f"output {name} contains non-finite values")
    feature_gate = out.get("family_tf_feature_gate")
    if not isinstance(feature_gate, torch.Tensor):
        failures.append("missing tensor output family_tf_feature_gate")
    elif (
        feature_gate.ndim != 3
        or tuple(feature_gate.shape[1:]) != _MODEL_NATIVE_FEATURE_TF_GATE_SHAPE
    ):
        failures.append(
            "output family_tf_feature_gate "
            f"shape={tuple(feature_gate.shape)} expected="
            f"(batch,{_MODEL_NATIVE_FEATURE_TF_GATE_SHAPE[0]},"
            f"{_MODEL_NATIVE_FEATURE_TF_GATE_SHAPE[1]})"
        )
    elif not bool(torch.isfinite(feature_gate).all().item()):
        failures.append("output family_tf_feature_gate contains non-finite values")
    return failures


def _direction_decision_contract_export_failures(
    lock: Dict[str, Any],
    meta: Dict[str, Any],
) -> list[str]:
    canonical = model_direction_decision_contract_metadata()
    lock_contract = lock.get("direction_decision_contract")
    meta_contract = meta.get("direction_decision_contract")
    failures: list[str] = []
    if lock_contract != canonical:
        failures.append("MASTER_TRANSFORMER_LOCK direction_decision_contract is not canonical")
    if meta_contract != canonical:
        failures.append("bundle_metadata direction_decision_contract is not canonical")
    if lock_contract != meta_contract:
        failures.append("direction_decision_contract split-brain between lock and metadata")
    return failures


def _unified_exit_export_failures(
    lock: Dict[str, Any],
    meta: Dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    canonical = unified_entry_exit_contract_metadata()
    if lock.get("unified_entry_exit_contract") != canonical:
        failures.append(
            "MASTER_TRANSFORMER_LOCK unified_entry_exit_contract is not canonical"
        )
    if meta.get("unified_entry_exit_contract") != canonical:
        failures.append(
            "bundle_metadata unified_entry_exit_contract is not canonical"
        )
    lock_evidence = lock.get("unified_exit_training_evidence")
    meta_evidence = meta.get("unified_exit_training_evidence")
    if (
        not isinstance(lock_evidence, dict)
        or lock_evidence.get("decision") != "PASS"
    ):
        failures.append(
            "MASTER_TRANSFORMER_LOCK unified_exit_training_evidence missing"
        )
    if lock_evidence != meta_evidence:
        failures.append(
            "unified_exit_training_evidence split-brain between lock and metadata"
        )
    if isinstance(lock_evidence, dict):
        try:
            require_unified_exit_lifecycle_authority_evidence(
                lock_evidence.get("lifecycle")
            )
        except RuntimeError as exc:
            failures.append(
                "unified_exit lifecycle M1 authority evidence invalid: "
                f"{exc}"
            )
    return failures


def _m1_feature_surface_binding_from_lifecycle(
    lifecycle_evidence: Mapping[str, Any],
    *,
    dataset_run_id: str,
) -> dict[str, str]:
    """Export the exact M1 runtime binding proved by lifecycle admission."""

    feature_path = Path(str(lifecycle_evidence.get("m1_feature_base_path", "")))
    manifest_path = Path(
        str(lifecycle_evidence.get("m1_feature_base_manifest_path", ""))
    )
    if (
        not feature_path.is_absolute()
        or feature_path.is_symlink()
        or not feature_path.is_file()
        or not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise RuntimeError("ENTRY_EXPORT_M1_FEATURE_SURFACE_BINDING_PATH_INVALID")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("ENTRY_EXPORT_M1_FEATURE_SURFACE_BINDING_MANIFEST_INVALID") from exc
    pair_generation_id = manifest.get("pair_generation_id")
    feature_sha256 = _sha256_file(feature_path)
    manifest_sha256 = _sha256_file(manifest_path)
    feature_field_order = manifest.get("feature_field_order")
    feature_field_order_sha256 = hashlib.sha256(
        json.dumps(
            feature_field_order,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        not isinstance(pair_generation_id, str)
        or not pair_generation_id
        or not isinstance(feature_field_order, list)
        or len(feature_field_order) != MODEL_NATIVE_SIGNAL_DIM
        or len(set(feature_field_order)) != MODEL_NATIVE_SIGNAL_DIM
        or manifest.get("dataset_run_id") != dataset_run_id
        or manifest.get("output_parquet") != str(feature_path)
        or manifest.get("output_parquet_sha256") != feature_sha256
        or manifest.get("feature_field_order_sha256")
        != feature_field_order_sha256
        or lifecycle_evidence.get("m1_feature_base_sha256")
        != feature_sha256
        or lifecycle_evidence.get("m1_feature_base_manifest_sha256")
        != manifest_sha256
    ):
        raise RuntimeError("ENTRY_EXPORT_M1_FEATURE_SURFACE_BINDING_LINEAGE_INVALID")
    return {
        "parquet_path": str(feature_path),
        "manifest_path": str(manifest_path),
        "dataset_run_id": str(dataset_run_id),
        "pair_generation_id": pair_generation_id,
        "parquet_sha256": feature_sha256,
        "manifest_sha256": manifest_sha256,
        "feature_field_order_sha256": feature_field_order_sha256,
    }


# -----------------------------------------------------------------------------
# Separate legacy-RL import guard (fail-fast). Internal Q/V heads live here.
# -----------------------------------------------------------------------------
def _guard_no_rl() -> None:
    """Hard-fail if gx1.rl or legacy was imported."""
    for mod in list(sys.modules.keys()):
        if mod == "gx1.rl" or mod.startswith("gx1.rl."):
            raise RuntimeError(
                "[ENTRY_V10_CTX_RL_FORBIDDEN] gx1.rl must not be imported. "
                f"Found: {mod}"
            )
        if "legacy" in mod and mod.startswith("gx1."):
            raise RuntimeError(
                "[ENTRY_V10_CTX_LEGACY_FORBIDDEN] gx1 legacy must not be imported. "
                f"Found: {mod}"
            )


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_TRAINER_MEMORY_LIMIT_BYTES = 20 * 1024**3
_TRAINER_SWAP_LIMIT_BYTES = 512 * 1024**2
_TRAINER_PIDS_LIMIT = 64
_TRAINER_CGROUP_ENV = {
    "memory": "GX1_CAPPED_MEMORY_BYTES",
    "swap": "GX1_CAPPED_SWAP_BYTES",
    "pids": "GX1_CAPPED_TASKS_MAX",
}


def _require_trainer_cgroup_preflight(
    *,
    environ: Mapping[str, str] | None = None,
    read_text: Any = None,
) -> dict[str, Any]:
    """Prove the current process is the capped trainer before data reads."""

    env = os.environ if environ is None else environ
    if str(env.get("GX1_CAPPED_CLASS") or "") != "trainer":
        raise RuntimeError("[ENTRY_TRAIN_CGROUP_CLASS_INVALID]")

    expected: dict[str, int] = {}
    for label, name in _TRAINER_CGROUP_ENV.items():
        raw = str(env.get(name) or "")
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(
                f"[ENTRY_TRAIN_CGROUP_ENV_PROOF_INVALID] field={name}"
            )
        expected[label] = int(raw)
    if (
        expected["memory"] > _TRAINER_MEMORY_LIMIT_BYTES
        or expected["swap"] > _TRAINER_SWAP_LIMIT_BYTES
        or expected["pids"] > _TRAINER_PIDS_LIMIT
    ):
        raise RuntimeError("[ENTRY_TRAIN_CGROUP_ENV_LIMIT_EXCEEDED]")

    reader = read_text or (lambda path: path.read_text(encoding="utf-8"))
    try:
        cgroup_lines = str(reader(Path("/proc/self/cgroup"))).splitlines()
    except Exception as exc:
        raise RuntimeError("[ENTRY_TRAIN_CGROUP_PATH_UNAVAILABLE]") from exc
    unified = [
        line.split(":", 2)[2]
        for line in cgroup_lines
        if len(line.split(":", 2)) == 3
        and line.split(":", 2)[0] == "0"
        and line.split(":", 2)[1] == ""
    ]
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise RuntimeError("[ENTRY_TRAIN_CGROUP_PATH_INVALID]")
    relative_parts = Path(unified[0]).parts[1:]
    if (
        not relative_parts
        or not relative_parts[-1].endswith(".scope")
        or any(part in {"", ".", ".."} for part in relative_parts)
    ):
        raise RuntimeError("[ENTRY_TRAIN_CGROUP_PATH_INVALID]")
    cgroup_dir = Path("/sys/fs/cgroup").joinpath(*relative_parts)

    def _read_limit(name: str) -> int:
        try:
            raw = str(reader(cgroup_dir / name)).strip()
        except Exception as exc:
            raise RuntimeError(
                f"[ENTRY_TRAIN_CGROUP_LIMIT_UNAVAILABLE] field={name}"
            ) from exc
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(
                f"[ENTRY_TRAIN_CGROUP_LIMIT_INVALID] field={name}"
            )
        return int(raw)

    actual = {
        "memory_max": _read_limit("memory.max"),
        "memory_high": _read_limit("memory.high"),
        "swap": _read_limit("memory.swap.max"),
        "pids": _read_limit("pids.max"),
    }
    if (
        actual["memory_max"] > _TRAINER_MEMORY_LIMIT_BYTES
        or actual["memory_high"] > _TRAINER_MEMORY_LIMIT_BYTES
        or actual["swap"] > _TRAINER_SWAP_LIMIT_BYTES
        or actual["pids"] > _TRAINER_PIDS_LIMIT
    ):
        raise RuntimeError("[ENTRY_TRAIN_CGROUP_ACTUAL_LIMIT_EXCEEDED]")
    if (
        actual["memory_max"] != expected["memory"]
        or actual["memory_high"] != expected["memory"]
        or actual["swap"] != expected["swap"]
        or actual["pids"] != expected["pids"]
    ):
        raise RuntimeError("[ENTRY_TRAIN_CGROUP_ENV_ACTUAL_MISMATCH]")
    return {
        "class": "trainer",
        "cgroup_path": str(cgroup_dir),
        **actual,
    }


def _flush_memmap_pages(*arrays: np.ndarray) -> None:
    """Flush disk-backed arrays and release clean mapped pages from RSS when supported."""
    for arr in arrays:
        if not isinstance(arr, np.memmap):
            continue
        arr.flush()
        mm = getattr(arr, "_mmap", None)
        if mm is None or not hasattr(mm, "madvise"):
            continue
        try:
            mm.madvise(mmap.MADV_DONTNEED)
        except (OSError, ValueError, AttributeError):
            pass


# The immutable TRAIN parquet stores nested values as float64.  An 8192-row
# Arrow batch transiently held the Arrow values, a float32 cast and dirty
# memmap pages together and could exhaust the 10 GiB cgroup before training.
# These are I/O scheduling bounds only; row order and tensor bytes are exact.
_NESTED_ARROW_BATCH_ROWS = 512
_MEMMAP_WRITEBACK_ROWS = 2048
_MEMMAP_MIN_BYTES = 512 * 1024**2
_MEMMAP_ROOT = Path("/home/andre2/GX1_DATA/tmp/entry_v10_memmap")


def _env_str(name: str) -> str:
    """Read one recipe-owned training value from the single canonical origin.

    The trainer used to carry literal defaults at individual call sites.  Those
    values formed a shadow contract that could drift from the recipe owner.
    The recipe owner is now the only origin, so drift is impossible by
    construction and no value here is invented.  An unknown key is an error,
    never a substitution.
    """

    if name not in MODEL_NATIVE_RECIPE_ENV:
        raise RuntimeError(
            "[ENTRY_TRAIN_RECIPE_KEY_UNKNOWN] "
            f"{name} is not owned by the canonical recipe contract"
        )
    return str(os.getenv(name, MODEL_NATIVE_RECIPE_ENV[name])).strip()


# V30 package 5 stability dampers (measured limit cycle across three seeds,
# 2026-08-12/13; see the recipe owner's origin comment).  Both are OFF-able to
# exactly today's behaviour: the cosine switch at 0 constructs no scheduler and
# never writes a param_group, and the EMA decay at 0.0 allocates no shadow
# weights, so validation and checkpoint selection see the raw training weights.
ENTRY_TRAIN_LR_COSINE_DECAY = int(_env_str("ENTRY_TRAIN_LR_COSINE_DECAY"))
if ENTRY_TRAIN_LR_COSINE_DECAY not in (0, 1):
    raise RuntimeError(
        "[ENTRY_TRAIN_LR_COSINE_DECAY_INVALID] "
        f"got={ENTRY_TRAIN_LR_COSINE_DECAY!r} expected 0 or 1"
    )
# The recipe key declares a HORIZON, not a magnitude (V30 package 6, operator
# decision 2026-08-13): "0.0" is the exact-compatibility OFF sentinel and
# "epoch" selects the recipe owner's derivation, decay = 1 - 1/steps_per_epoch
# over the run's declared budget. The declared string is validated here; the
# float itself cannot exist yet because the budget is a CLI/dataset quantity,
# so it is resolved once inside run_train through
# `resolve_weight_ema_decay` — the recipe owner is the only origin (rule 14).
ENTRY_TRAIN_WEIGHT_EMA_DECAY_DECLARED = _env_str("ENTRY_TRAIN_WEIGHT_EMA_DECAY")
if ENTRY_TRAIN_WEIGHT_EMA_DECAY_DECLARED not in (
    MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES
):
    raise RuntimeError(
        "[ENTRY_TRAIN_WEIGHT_EMA_DECAY_INVALID] "
        f"got={ENTRY_TRAIN_WEIGHT_EMA_DECAY_DECLARED!r} expected one of "
        f"{MODEL_NATIVE_WEIGHT_EMA_DECAY_DECLARED_VALUES}"
    )
# The only checkpoint authority is realized executable Entry-policy PnL on
# complete VAL episodes. Aggregate auxiliary loss and historical direction
# class accuracy are diagnostics and cannot select the shipped policy.
ENTRY_CKPT_MONITOR = _env_str("GX1_V10_CKPT_MONITOR").strip().lower()
if ENTRY_CKPT_MONITOR != "entry_policy_pnl":
    raise RuntimeError(
        "[ENTRY_CKPT_MONITOR_INVALID] "
        f"got={ENTRY_CKPT_MONITOR!r} expected=entry_policy_pnl"
    )

# -----------------------------------------------------------------------------
# Auxiliary labels use their native units. Relative task influence is learned.
# -----------------------------------------------------------------------------
# Negative outcome labels remain inputs to the learned clean-edge auxiliary
# objective.  They do not hand-weight direction CE or directly suppress a
# LONG/SHORT probability.

# The unified-exit action loss runs every valid exit sequence through the shared
# encoder, and each sequence carries a 480-bar M1 surface attended over full
# O(480^2) by one main plus eight specialist encoders. Processing all valid rows
# of a batch at once peaks ~3.9 GB and overflows the 10G trainer cap even at
# batch 8. The loss reads only raw-bps Exit Q values and reduces masked MSE
# over valid action cells through LayerNorm-only sublayers, so the row
# dimension is chunked and the per-chunk loss sums accumulated: exact for the loss
# (dropout RNG ordering aside), with the attention peak bounded by the chunk size
# independent of batch. 8 rows keeps the 480-bar attention transient near 1.2 GB.
UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS = 8
# CUDA measurement 2026-08-08 verified VRAM headroom (0.06-0.17 GB) only up to
# 128 rows and was then extrapolated to "unbounded" (chunk_rows=valid_rows) on
# the assumption that cost is linear in rows past that point. A real batch=640
# run on 2026-08-09 disproved the extrapolation: the same 8,000-row subsample
# took ~9.5x longer wall-clock at batch=640 (up to 2,560 valid rows in one
# unchunked call) than at batch=64 (up to 256 rows), and validation's first
# large unchunked call crashed with a CUDA "illegal memory access" after 13
# training steps had already pressured the allocator. The cost past 128 rows is
# not proven linear, so CUDA is bounded at the one size actually measured safe.
UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS_CUDA = 128

# The attended route is a bounded trainability diagnostic, never a candidate
# producer.  Its Exit branch is intentionally narrower than the canonical
# trainer: this bounds its transient attention allocation below the group size
# that reached the VRAM ceiling in the first attended smoke.  The current
# 60-step budget is one checkpointed, operator-present research session under
# the guard-owned five-minute CUDA window.  It replaces the former two-step
# probe only after that probe completed the complete V46 data preflight and
# CUDA model stage without a guard breach.  Both values are source-owned; no
# CLI argument or ambient variable may expand either one.
_ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION = "gx1_attended_research_session_v1"
_ATTENDED_RESEARCH_MAX_OPTIMIZER_STEPS = 60
# An attended session may allocate at most half of the device through the
# PyTorch caching allocator.  The independent guard observes total NVML usage
# every second and stops at the same 12 GiB threshold; the allocator fence is
# needed because a WSL residency failure can occur before the next poll.
_ATTENDED_RESEARCH_CUDA_MEMORY_FRACTION = 0.50
_ATTENDED_RESEARCH_BATCH_SIZE = 8
# The former 32-row attended group kept almost all of the WSL GPU allocation
# resident in the historical smoke.  Use the only group size with a documented
# bounded 480-bar attention measurement instead.  This is a diagnostic-only,
# checkpointed lane; it trades speed for a releasable graph after each group.
_ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS = 8
_ATTENDED_RESEARCH_SESSION_DIR_PREFIX = ".gx1-attended-research-session."
_ATTENDED_RESEARCH_CONTRACT_FILENAME = "ATTENDED_RESEARCH_SESSION_CONTRACT.json"
_ATTENDED_RESEARCH_ACTIVE_FILENAME = "ATTENDED_RESEARCH_SESSION_ACTIVE.json"
_ATTENDED_RESEARCH_STATE_FILENAMES = (
    "attended_research_state_slot_0.pt",
    "attended_research_state_slot_1.pt",
)


class _ExactIndexSampler(Sampler[int]):
    """Yield one persisted order without consuming any additional RNG state."""

    def __init__(self, order: torch.Tensor, *, batch_offset: int, batch_size: int):
        if (
            not isinstance(order, torch.Tensor)
            or order.dtype != torch.int64
            or order.ndim != 1
            or int(batch_offset) < 0
            or int(batch_size) < 1
        ):
            raise RuntimeError("[ATTENDED_RESEARCH_SAMPLER_ARGUMENT_INVALID]")
        start = int(batch_offset) * int(batch_size)
        if start > int(order.numel()):
            raise RuntimeError("[ATTENDED_RESEARCH_SAMPLER_OFFSET_INVALID]")
        self._order = order.detach().cpu().contiguous()
        self._start = start

    def __iter__(self):
        return iter(self._order[self._start :].tolist())

    def __len__(self) -> int:
        return int(self._order.numel()) - self._start


def _attended_session_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _attended_session_sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _attended_session_atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError("[ATTENDED_RESEARCH_SESSION_PATH_INVALID]")
    payload = _attended_session_json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_regular_file(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _attended_session_read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"[ATTENDED_RESEARCH_SESSION_{label}_PATH_INVALID]")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            f"[ATTENDED_RESEARCH_SESSION_{label}_JSON_INVALID]"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"[ATTENDED_RESEARCH_SESSION_{label}_JSON_INVALID]")
    return value


def _attended_session_rng_state(*, device: torch.device) -> dict[str, Any]:
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    if numpy_name != "MT19937":
        raise RuntimeError("[ATTENDED_RESEARCH_NUMPY_RNG_UNSUPPORTED]")
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_name": numpy_name,
        "numpy_keys": torch.as_tensor(numpy_keys, dtype=torch.uint32).cpu(),
        "numpy_position": int(numpy_position),
        "numpy_has_gauss": int(numpy_has_gauss),
        "numpy_cached": float(numpy_cached),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = [
            item.detach().cpu() for item in torch.cuda.get_rng_state_all()
        ]
    return state


def _restore_attended_session_rng_state(
    state: Mapping[str, Any], *, device: torch.device
) -> None:
    expected = {
        "python",
        "numpy_name",
        "numpy_keys",
        "numpy_position",
        "numpy_has_gauss",
        "numpy_cached",
        "torch_cpu",
    }
    if device.type == "cuda":
        expected.add("torch_cuda")
    if set(state) != expected:
        raise RuntimeError("[ATTENDED_RESEARCH_RNG_STATE_SCHEMA_INVALID]")
    python_state = state["python"]
    numpy_keys = state["numpy_keys"]
    torch_cpu = state["torch_cpu"]
    if (
        state["numpy_name"] != "MT19937"
        or not isinstance(numpy_keys, torch.Tensor)
        or numpy_keys.dtype != torch.uint32
        or numpy_keys.ndim != 1
        or not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
    ):
        raise RuntimeError("[ATTENDED_RESEARCH_RNG_STATE_INVALID]")
    try:
        random.setstate(python_state)
        np.random.set_state(
            (
                "MT19937",
                numpy_keys.detach().cpu().numpy().astype(np.uint32, copy=False),
                int(state["numpy_position"]),
                int(state["numpy_has_gauss"]),
                float(state["numpy_cached"]),
            )
        )
        torch.set_rng_state(torch_cpu.detach().cpu())
        if device.type == "cuda":
            cuda_state = state["torch_cuda"]
            if (
                not isinstance(cuda_state, list)
                or not cuda_state
                or any(
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.uint8
                    or value.ndim != 1
                    for value in cuda_state
                )
            ):
                raise RuntimeError("[ATTENDED_RESEARCH_CUDA_RNG_STATE_INVALID]")
            torch.cuda.set_rng_state_all(
                [value.detach().cpu() for value in cuda_state]
            )
    except (TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("[ATTENDED_"):
            raise
        raise RuntimeError("[ATTENDED_RESEARCH_RNG_STATE_INVALID]") from exc


class _AttendedResearchSession:
    """Two-slot, hash-bound progress state for one attended smoke only.

    The session is intentionally a sibling of the still-nonexistent bundle
    directory.  Therefore it cannot be mistaken for a partial bundle or be
    consumed by any candidate/promotion path.  A completed optimizer step is
    the only checkpoint boundary; the active JSON must hash-bind the selected
    slot before any resume is accepted.
    """

    def __init__(
        self,
        *,
        out_bundle_dir: Path,
        contract: Mapping[str, Any],
    ) -> None:
        output = Path(out_bundle_dir).expanduser().resolve()
        if output.exists() or output.is_symlink() or output.parent.is_symlink():
            raise RuntimeError("[ATTENDED_RESEARCH_OUTPUT_PATH_INVALID]")
        if not output.parent.is_dir():
            raise RuntimeError("[ATTENDED_RESEARCH_OUTPUT_PARENT_INVALID]")
        self._directory = output.parent / (
            _ATTENDED_RESEARCH_SESSION_DIR_PREFIX + output.name
        )
        self._contract = dict(contract)
        self._contract_bytes = _attended_session_json_bytes(self._contract)
        self._contract_sha256 = _attended_session_sha256_bytes(self._contract_bytes)
        self._active_path = self._directory / _ATTENDED_RESEARCH_ACTIVE_FILENAME
        self._contract_path = self._directory / _ATTENDED_RESEARCH_CONTRACT_FILENAME
        if self._directory.exists():
            directory_stat = os.stat(self._directory, follow_symlinks=False)
            if (
                self._directory.is_symlink()
                or not self._directory.is_dir()
                or directory_stat.st_uid != os.getuid()
                or directory_stat.st_mode & 0o077
            ):
                raise RuntimeError("[ATTENDED_RESEARCH_SESSION_DIRECTORY_INVALID]")
            on_disk = _attended_session_read_json(
                self._contract_path, label="CONTRACT"
            )
            if on_disk != self._contract:
                raise RuntimeError("[ATTENDED_RESEARCH_SESSION_CONTRACT_MISMATCH]")
        else:
            self._directory.mkdir(mode=0o700)
            try:
                descriptor = os.open(
                    self._contract_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(self._contract_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_regular_file(self._contract_path)
            except Exception:
                # An incomplete, first-use session has no checkpoint and must
                # fail closed rather than be treated as resumable evidence.
                raise

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def contract_sha256(self) -> str:
        return self._contract_sha256

    def _slot_path(self, slot: int) -> Path:
        if slot not in (0, 1):
            raise RuntimeError("[ATTENDED_RESEARCH_SLOT_INVALID]")
        return self._directory / _ATTENDED_RESEARCH_STATE_FILENAMES[slot]

    def load_checkpoint(self) -> Optional[dict[str, Any]]:
        if self._active_path.is_symlink():
            raise RuntimeError("[ATTENDED_RESEARCH_SESSION_ACTIVE_PATH_INVALID]")
        if not self._active_path.exists():
            return None
        active = _attended_session_read_json(self._active_path, label="ACTIVE")
        expected_keys = {
            "schema_version",
            "session_contract_sha256",
            "slot",
            "checkpoint_index",
            "state_sha256",
            "complete_optimizer_steps",
            "epoch_index",
            "next_batch_offset",
            "epoch_order_sha256",
            "complete",
        }
        if (
            set(active) != expected_keys
            or active.get("schema_version") != _ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION
            or active.get("session_contract_sha256") != self._contract_sha256
            or active.get("slot") not in (0, 1)
            or any(
                not isinstance(active.get(key), int) or isinstance(active.get(key), bool)
                or int(active[key]) < 0
                for key in (
                    "checkpoint_index",
                    "complete_optimizer_steps",
                    "epoch_index",
                    "next_batch_offset",
                )
            )
            or not isinstance(active.get("state_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(active.get("state_sha256")))
            or not isinstance(active.get("epoch_order_sha256"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(active.get("epoch_order_sha256"))
            )
            or not isinstance(active.get("complete"), bool)
        ):
            raise RuntimeError("[ATTENDED_RESEARCH_ACTIVE_POINTER_INVALID]")
        state_path = self._slot_path(int(active["slot"]))
        if state_path.is_symlink() or not state_path.is_file():
            raise RuntimeError("[ATTENDED_RESEARCH_STATE_PATH_INVALID]")
        if _sha256_file(state_path) != active["state_sha256"]:
            raise RuntimeError("[ATTENDED_RESEARCH_STATE_SHA256_MISMATCH]")
        try:
            # A checkpoint contains CUDA-originating tensors.  Always stage it
            # in host memory first: loading straight to CUDA can briefly retain
            # both the serialized state and the restored optimizer state on an
            # already constrained device.
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("[ATTENDED_RESEARCH_STATE_LOAD_INVALID]") from exc
        if not isinstance(state, dict):
            raise RuntimeError("[ATTENDED_RESEARCH_STATE_SCHEMA_INVALID]")
        if (
            state.get("schema_version") != _ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION
            or state.get("session_contract_sha256") != self._contract_sha256
            or int(state.get("checkpoint_index", -1))
            != int(active["checkpoint_index"])
            or int(state.get("complete_optimizer_steps", -1))
            != int(active["complete_optimizer_steps"])
            or int(state.get("epoch_index", -1)) != int(active["epoch_index"])
            or int(state.get("next_batch_offset", -1))
            != int(active["next_batch_offset"])
            or bool(state.get("complete", False)) != bool(active["complete"])
        ):
            raise RuntimeError("[ATTENDED_RESEARCH_STATE_POINTER_MISMATCH]")
        order = state.get("epoch_order")
        if (
            not isinstance(order, torch.Tensor)
            or order.dtype != torch.int64
            or order.ndim != 1
            or hashlib.sha256(order.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            != active["epoch_order_sha256"]
        ):
            raise RuntimeError("[ATTENDED_RESEARCH_ORDER_INVALID]")
        return state

    def save_checkpoint(
        self,
        *,
        model: nn.Module,
        target_model: nn.Module,
        optimizer: optim.Optimizer,
        weight_ema: Optional["_WeightEma"],
        lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
        device: torch.device,
        checkpoint_index: int,
        complete_optimizer_steps: int,
        epoch_index: int,
        next_batch_offset: int,
        epoch_order: torch.Tensor,
        complete: bool,
    ) -> None:
        if (
            int(checkpoint_index) < 1
            or int(complete_optimizer_steps) < 1
            or int(epoch_index) < 0
            or int(next_batch_offset) < 0
            or not isinstance(epoch_order, torch.Tensor)
            or epoch_order.dtype != torch.int64
            or epoch_order.ndim != 1
        ):
            raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_ARGUMENT_INVALID]")
        previous = self.load_checkpoint()
        previous_slot = -1 if previous is None else int(
            _attended_session_read_json(self._active_path, label="ACTIVE")["slot"]
        )
        slot = 0 if previous_slot != 0 else 1
        state_path = self._slot_path(slot)
        if state_path.is_symlink():
            raise RuntimeError("[ATTENDED_RESEARCH_STATE_PATH_INVALID]")
        order_cpu = epoch_order.detach().cpu().contiguous()
        state = {
            "schema_version": _ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION,
            "session_contract_sha256": self._contract_sha256,
            "checkpoint_index": int(checkpoint_index),
            "complete_optimizer_steps": int(complete_optimizer_steps),
            "epoch_index": int(epoch_index),
            "next_batch_offset": int(next_batch_offset),
            "epoch_order": order_cpu,
            "model_state": model.state_dict(),
            "target_model_state": target_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "weight_ema_state": (
                weight_ema.checkpoint_state() if weight_ema is not None else None
            ),
            "lr_scheduler_state": (
                lr_scheduler.state_dict() if lr_scheduler is not None else None
            ),
            "rng_state": _attended_session_rng_state(device=device),
            "complete": bool(complete),
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{state_path.name}.", dir=str(self._directory))
        try:
            os.close(fd)
            torch.save(state, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, state_path)
            _fsync_regular_file(state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        active = {
            "schema_version": _ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION,
            "session_contract_sha256": self._contract_sha256,
            "slot": slot,
            "checkpoint_index": int(checkpoint_index),
            "state_sha256": _sha256_file(state_path),
            "complete_optimizer_steps": int(complete_optimizer_steps),
            "epoch_index": int(epoch_index),
            "next_batch_offset": int(next_batch_offset),
            "epoch_order_sha256": hashlib.sha256(
                order_cpu.numpy().tobytes()
            ).hexdigest(),
            "complete": bool(complete),
        }
        _attended_session_atomic_write_json(self._active_path, active)


def _enforce_canonical_train_env_contract() -> None:
    """Require the exact recipe at the trainer boundary with no ambient controls."""
    forbidden = [
        name
        for name in (
            "GX1_NON_CANONICAL_DIAGNOSTIC",
            "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES",
        )
        if name in os.environ
    ]
    if forbidden:
        raise RuntimeError(
            "[ENTRY_CANONICAL_TRAIN_ENV_BYPASS_FORBIDDEN] "
            + ", ".join(sorted(forbidden))
        )
    recipe_env = {
        key: os.environ[key]
        for key in MODEL_NATIVE_RECIPE_ENV_KEYS
        if key in os.environ
    }
    try:
        require_model_native_recipe_env(recipe_env)
    except RuntimeError as exc:
        raise RuntimeError(f"[ENTRY_TRAIN_RECIPE_ENV_INVALID] {exc}") from exc
    allowed_runtime_env = {
        *_TRAIN_ARTIFACT_HASH_ENV.values(),
        _TRAIN_DATASET_RUN_ID_ENV,
        _TRAIN_MULTI_TF_CACHE_ENV,
        *_TRAIN_CAPPED_SCOPE_ENV,
    }
    extra_controls = sorted(
        key
        for key in os.environ
        if (
            key.startswith("ENTRY_")
            or key.startswith("GX1_")
        )
        and key not in MODEL_NATIVE_RECIPE_ENV_KEYS
        and key not in allowed_runtime_env
    )
    if extra_controls:
        raise RuntimeError(
            "[ENTRY_TRAIN_AMBIENT_CONTROL_FORBIDDEN] "
            + ", ".join(extra_controls)
        )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_SEQUENCE_ROLL_AUDIT_SCHEMA_VERSION = "entry_model_native_sequence_roll_audit_v1"
_SEQUENCE_ROLL_AUDIT_CHECKS = {
    "all_values_finite_float32": True,
    "every_seq_last_equals_snap_bit_identical": True,
    "every_adjacent_sequence_rolls_one_snapshot_bit_identical": True,
    "batch_boundary_rolls_bit_identical": True,
}
_SEQUENCE_ROLL_AUDIT_AUTHORITY = {
    "data_reconstruction_only": True,
    "candidate": False,
    "test": False,
    "promotion": False,
    "paper": False,
    "live": False,
}
_SEQUENCE_SOURCE_SURFACE_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _sequence_source_exact_regular_file(raw: Path, *, label: str) -> Path:
    """Resolve one immutable source-reconstruction input without indirection."""

    supplied = Path(raw).expanduser()
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or any(parent.is_symlink() for parent in supplied.parents)
    ):
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_PATH_INVALID]"
        )
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_PATH_INVALID]"
        ) from exc
    if resolved != supplied or not resolved.is_file():
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_PATH_INVALID]"
        )
    return resolved


def _load_sequence_source_surface(
    surface_path: Path,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the exact M5 signal surface in bounded Arrow batches.

    Unlike the legacy emitted-row roll shortcut this backing is the real M5
    event timeline.  It stays below one gigabyte and is shared logically by
    every filtered supervised row, avoiding a 20+ GiB scratch ``seq`` mirror.
    """

    cache_key = str(surface_path)
    cached = _SEQUENCE_SOURCE_SURFACE_CACHE.get(cache_key)
    if cached is not None:
        cached_times, cached_signal = cached
        if (
            cached_times.shape == (int(expected_rows),)
            and cached_signal.shape
            == (int(expected_rows), MODEL_NATIVE_SIGNAL_DIM)
        ):
            return cached
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_CACHE_INVALID]"
        )

    import pyarrow.parquet as pq

    try:
        feature_surface = pq.ParquetFile(surface_path)
    except Exception as exc:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_OPEN_INVALID]"
        ) from exc
    if tuple(feature_surface.schema_arrow.names) != (
        "time",
        "signal",
        "ctx_cont",
        "ctx_cat",
    ):
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_SCHEMA_INVALID]"
        )
    rows = int(feature_surface.metadata.num_rows)
    if rows != int(expected_rows) or rows < MODEL_NATIVE_SEQ_LEN:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_ROWS_INVALID]"
        )
    try:
        table = feature_surface.read(columns=["time"]).combine_chunks()
        times = table.column("time").to_numpy(zero_copy_only=False)
        times = times.astype("datetime64[ns]").astype(np.int64, copy=False)
    except Exception as exc:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_TIME_INVALID]"
        ) from exc
    if (
        times.shape != (rows,)
        or np.any(times == np.iinfo(np.int64).min)
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(times) % (ENTRY_DECISION_BAR_SECONDS * 1_000_000_000))
    ):
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_TIME_INVALID]"
        )
    signal_surface = np.empty((rows, MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32)
    offset = 0
    try:
        batches = feature_surface.iter_batches(
            batch_size=_NESTED_ARROW_BATCH_ROWS,
            columns=["signal"],
            use_threads=False,
        )
        for batch in batches:
            count = int(batch.num_rows)
            values = batch.column("signal")
            if not hasattr(values, "values"):
                raise RuntimeError("signal values missing")
            flat = values.values.to_numpy(zero_copy_only=False)
            if flat.shape != (count * MODEL_NATIVE_SIGNAL_DIM,):
                raise RuntimeError("signal width invalid")
            decoded = np.asarray(flat, dtype=np.float32).reshape(
                count, MODEL_NATIVE_SIGNAL_DIM
            )
            if not np.isfinite(decoded).all():
                raise RuntimeError("signal nonfinite")
            signal_surface[offset : offset + count] = decoded
            offset += count
    except RuntimeError as exc:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_SIGNAL_INVALID]"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_SIGNAL_INVALID]"
        ) from exc
    if offset != rows:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_ROW_COUNT_MISMATCH]"
        )
    _SEQUENCE_SOURCE_SURFACE_CACHE[cache_key] = (times, signal_surface)
    return times, signal_surface


def _require_sequence_source_reconstruction(
    audit_path: Path,
    *,
    parquet_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Authorize one source-backed sequence view and load its exact surface."""

    audit_path = _sequence_source_exact_regular_file(audit_path, label="AUDIT")
    parquet_path = _sequence_source_exact_regular_file(parquet_path, label="PARQUET")
    manifest_path = _sequence_source_exact_regular_file(manifest_path, label="MANIFEST")
    audit = _sequence_roll_read_json_object(audit_path)
    manifest = _sequence_roll_read_json_object(manifest_path)
    feature_surface = feature_surface_binding_from_split_manifest(manifest)
    source_path = _sequence_source_exact_regular_file(
        Path(feature_surface["path"]), label="FEATURE_SURFACE"
    )
    source_manifest_path = _sequence_source_exact_regular_file(
        Path(feature_surface["manifest_path"]), label="FEATURE_SURFACE_MANIFEST"
    )
    import pyarrow.parquet as pq

    rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
    try:
        require_sequence_source_reconstruction_audit(
            audit,
            expected_parquet_path=parquet_path,
            expected_manifest_path=manifest_path,
            expected_parquet_sha256=_sha256_file(parquet_path),
            expected_manifest_sha256=_sha256_file(manifest_path),
            expected_feature_surface=manifest,
            expected_rows=rows,
            expected_seq_len=MODEL_NATIVE_SEQ_LEN,
            expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_AUDIT_INVALID] {exc}"
        ) from exc
    if (
        _sha256_file(source_path) != feature_surface["sha256"]
        or _sha256_file(source_manifest_path) != feature_surface["manifest_sha256"]
    ):
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SOURCE_BINDING_INVALID]"
        )
    source_times, source_signal = _load_sequence_source_surface(
        source_path, expected_rows=int(feature_surface["rows"])
    )
    return dict(audit), source_times, source_signal


def _sequence_roll_exact_regular_file(raw: Path, *, label: str) -> Path:
    """Resolve one immutable rolling-proof input without mutable indirection."""

    supplied = Path(raw).expanduser()
    if (
        not supplied.is_absolute()
        or supplied.is_symlink()
        or any(parent.is_symlink() for parent in supplied.parents)
    ):
        raise RuntimeError(f"[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_{label}_PATH_INVALID]")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_{label}_PATH_INVALID]"
        ) from exc
    if resolved != supplied or not resolved.is_file():
        raise RuntimeError(f"[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_{label}_PATH_INVALID]")
    return resolved


def _sequence_roll_read_json_object(path: Path) -> dict[str, Any]:
    """Parse a proof strictly: duplicate JSON keys are a fail-closed error."""

    def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_AUDIT_JSON_INVALID]"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_AUDIT_JSON_INVALID]")
    return value


def _sequence_roll_exact_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            "[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_AUDIT_SHA256_INVALID] "
            f"field={field}"
        )
    return value


def _require_sequence_roll_audit(
    audit_path: Path,
    *,
    parquet_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Authorize a zero-copy ``seq`` view only from a full exact proof.

    This is deliberately a data-storage optimization, not a second sequence
    producer.  The audit binds *both* source files by path and current bytes;
    candidates never supply it and retain the materialized code path.
    """

    audit_path = _sequence_roll_exact_regular_file(audit_path, label="AUDIT")
    parquet_path = _sequence_roll_exact_regular_file(parquet_path, label="PARQUET")
    manifest_path = _sequence_roll_exact_regular_file(
        manifest_path,
        label="MANIFEST",
    )
    audit = _sequence_roll_read_json_object(audit_path)
    expected_keys = {
        "schema_version",
        "decision",
        "created_utc",
        "parquet_path",
        "parquet_sha256",
        "manifest_path",
        "manifest_sha256",
        "rows",
        "sequence_shape",
        "snapshot_shape",
        "checks",
        "sequence_snapshot_chain_sha256",
        "authority",
    }
    if set(audit) != expected_keys:
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_AUDIT_SCHEMA_INVALID]")
    if (
        audit.get("schema_version") != _SEQUENCE_ROLL_AUDIT_SCHEMA_VERSION
        or audit.get("decision") != "PASS"
        or not isinstance(audit.get("created_utc"), str)
        or not audit["created_utc"]
    ):
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_AUDIT_DECISION_INVALID]")
    if (
        audit.get("parquet_path") != str(parquet_path)
        or audit.get("manifest_path") != str(manifest_path)
        or _sequence_roll_exact_sha256(
            audit.get("parquet_sha256"), field="parquet_sha256"
        )
        != _sha256_file(parquet_path)
        or _sequence_roll_exact_sha256(
            audit.get("manifest_sha256"), field="manifest_sha256"
        )
        != _sha256_file(manifest_path)
    ):
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_SOURCE_BINDING_INVALID]")
    import pyarrow.parquet as pq

    rows = int(pq.ParquetFile(parquet_path).metadata.num_rows)
    if (
        isinstance(audit.get("rows"), bool)
        or audit.get("rows") != rows
        or audit.get("sequence_shape")
        != [rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM]
        or audit.get("snapshot_shape") != [rows, MODEL_NATIVE_SIGNAL_DIM]
        or audit.get("checks") != _SEQUENCE_ROLL_AUDIT_CHECKS
        or audit.get("authority") != _SEQUENCE_ROLL_AUDIT_AUTHORITY
    ):
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_PROOF_INVALID]")
    _sequence_roll_exact_sha256(
        audit.get("sequence_snapshot_chain_sha256"),
        field="sequence_snapshot_chain_sha256",
    )
    return audit


def _model_state_sha256(model: nn.Module) -> str:
    """Hash one exact immutable target-network state without serialization."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def _git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)

def _require_nonneg(name: str, v: float) -> None:
    if float(v) < 0.0:
        raise RuntimeError(
            f"[ENTRY_TRAIN_RECIPE_VALUE_INVALID] {name} must be >= 0.0, got {v}"
        )


def _build_active_head_names() -> List[str]:
    """Return the one exact model-native learned-component surface."""
    return [*MODEL_NATIVE_ACTIVE_HEADS, "unified_exit"]


_ENTRY_Q_MOVEMENT_KEYS: Tuple[str, ...] = (
    "entry_q_joint_norm.weight",
    "entry_q_joint_norm.bias",
    "entry_q_joint_in.weight",
    "entry_q_joint_in.bias",
    "head_entry_action_q.weight",
    "head_entry_action_q.bias",
)


def _capture_entry_q_initial_state(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    missing = [key for key in _ENTRY_Q_MOVEMENT_KEYS if key not in state]
    if missing:
        raise RuntimeError(
            "[ENTRY_FITTED_Q_INITIAL_STATE_MISSING] "
            f"keys={missing}"
        )
    return {
        key: state[key].detach().cpu().clone()
        for key in _ENTRY_Q_MOVEMENT_KEYS
    }


def _entry_fitted_q_movement_proof(
    initial_state: Dict[str, torch.Tensor],
    selected_state: Dict[str, torch.Tensor],
    *,
    selected_checkpoint_epoch: int,
) -> Dict[str, Any]:
    if int(selected_checkpoint_epoch) <= 0:
        raise RuntimeError(
            "[ENTRY_FITTED_Q_MOVEMENT_EPOCH_INVALID] "
            f"selected_checkpoint_epoch={selected_checkpoint_epoch}"
        )

    parameter_deltas: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    for key in _ENTRY_Q_MOVEMENT_KEYS:
        initial = initial_state.get(key)
        selected = selected_state.get(key)
        if not isinstance(initial, torch.Tensor) or not isinstance(selected, torch.Tensor):
            failures.append(f"{key}:missing_or_non_tensor")
            continue
        if tuple(initial.shape) != tuple(selected.shape):
            failures.append(
                f"{key}:shape={tuple(selected.shape)} expected={tuple(initial.shape)}"
            )
            continue
        initial_f64 = initial.detach().cpu().to(dtype=torch.float64)
        selected_f64 = selected.detach().cpu().to(dtype=torch.float64)
        if not bool(torch.isfinite(initial_f64).all().item()):
            failures.append(f"{key}:initial_non_finite")
            continue
        if not bool(torch.isfinite(selected_f64).all().item()):
            failures.append(f"{key}:selected_non_finite")
            continue
        delta = selected_f64 - initial_f64
        max_abs_delta = float(delta.abs().max().item()) if delta.numel() else 0.0
        l2_delta = float(torch.linalg.vector_norm(delta).item()) if delta.numel() else 0.0
        changed = bool(max_abs_delta > 0.0 and l2_delta > 0.0)
        if not np.isfinite(max_abs_delta) or not np.isfinite(l2_delta):
            failures.append(f"{key}:non_finite_delta")
        parameter_deltas[key] = {
            "shape": [int(value) for value in selected.shape],
            "max_abs_delta": max_abs_delta,
            "l2_delta": l2_delta,
            "changed": changed,
        }

    component_keys = {
        "joint_projection": tuple(
            key for key in _ENTRY_Q_MOVEMENT_KEYS if key.startswith("entry_q_joint")
        ),
        "raw_q_head": tuple(
            key for key in _ENTRY_Q_MOVEMENT_KEYS if key.startswith("head_entry_action_q")
        ),
    }
    component_changed = {
        component: any(
            bool(parameter_deltas.get(key, {}).get("changed", False)) for key in keys
        )
        for component, keys in component_keys.items()
    }
    for component, changed in component_changed.items():
        if not changed:
            failures.append(f"{component}:no_learned_parameter_movement")

    out_weight = selected_state.get("head_entry_action_q.weight")
    output_rows_distinct = bool(
        isinstance(out_weight, torch.Tensor)
        and out_weight.ndim == 2
        and int(out_weight.shape[0]) == 3
        and all(
            not bool(torch.equal(out_weight[i], out_weight[j]))
            for i in range(3)
            for j in range(i + 1, 3)
        )
    )
    if not output_rows_distinct:
        failures.append("head_entry_action_q.weight:action_rows_not_distinct")

    proof = {
        "schema_version": "gx1_entry_fitted_q_parameter_movement_v1",
        "reference": "direct_joint_representation_raw_bps_q_head",
        "selected_checkpoint_epoch": int(selected_checkpoint_epoch),
        "parameter_deltas": parameter_deltas,
        "component_changed": component_changed,
        "output_rows_distinct": output_rows_distinct,
        "decision": "PASS",
    }
    if failures:
        raise RuntimeError(
            "[ENTRY_FITTED_Q_LEARNED_MOVEMENT_REQUIRED] "
            f"failures={failures} proof={json.dumps(proof, sort_keys=True)}"
        )
    return proof


# V12.2: grad-clip norm + weight-decay set at runtime via CLI flag. Module-level
# so we don't have to thread through 6 layers of function args.
_GRAD_CLIP_NORM: float = 1.0
_WEIGHT_DECAY: float = 1e-5


def _model_forward_fp32(
    model: nn.Module,
    *args,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Use the one deterministic fp32 model path owned by the recipe."""

    out = model(*args, **kwargs)
    if isinstance(out, dict):
        out = {k: (v.float() if hasattr(v, "float") and torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}
    elif torch.is_tensor(out) and out.is_floating_point():
        out = out.float()
    return out


def _multi_tf_kwargs_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Extract Entry's exact M15/H1/H4/D1 route, or fail closed."""
    out: Dict[str, torch.Tensor] = {}
    for key in ("seq_m15", "seq_h1", "seq_h4", "seq_d1"):
        if key not in batch or not isinstance(batch[key], torch.Tensor):
            raise RuntimeError(f"[ENTRY_EXACT_MULTI_TF_BATCH_MISSING] {key}")
        out[key] = batch[key].to(device)
    return out


def _load_specialist_fusion_contract(
    audit_json: Optional[Path],
    *,
    expected_signal_dim: int,
    ordered_signal_names: list[str],
    contract_mode: str,
) -> tuple[Dict[str, list[int]], Dict[str, Any]]:
    try:
        normalized_contract_mode = require_model_native_specialist_contract_mode(contract_mode)
    except ValueError as exc:
        raise RuntimeError(
            "[SPECIALIST_MODEL_NATIVE_CONTRACT_REQUIRED] "
            f"contract_mode={contract_mode!r} expected={MODEL_NATIVE_CONTRACT_MODE!r}"
        ) from exc
    required_training_specialists = required_training_specialists_for_mode(normalized_contract_mode)
    expected_model_contract = specialist_model_contract_for_mode(normalized_contract_mode)
    if audit_json is None:
        raise RuntimeError("[SPECIALIST_AUDIT_EXPLICIT_PATH_REQUIRED]")
    path = Path(audit_json).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"[SPECIALIST_AUDIT_MISSING] {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if str(report.get("decision")) != "PASS":
        raise RuntimeError(f"[SPECIALIST_AUDIT_NOT_PASS] {path} decision={report.get('decision')} failures={report.get('failures')}")
    signal_dim = int(report.get("signal_field_count") or 0)
    if signal_dim != int(expected_signal_dim):
        raise RuntimeError(f"[SPECIALIST_SIGNAL_DIM_MISMATCH] audit={signal_dim} expected={expected_signal_dim}")
    specialist_model_contract = (
        report.get("specialist_model_contract")
        if isinstance(report.get("specialist_model_contract"), dict)
        else {}
    )
    specialist_model_failures = list(report.get("specialist_model_contract_failures") or [])
    if not bool(report.get("specialist_model_contract_valid")):
        specialist_model_failures.append("specialist audit did not declare specialist_model_contract_valid=true")
    observed_contract_mode = report.get("contract_mode")
    if observed_contract_mode != normalized_contract_mode:
        specialist_model_failures.append(
            f"specialist audit contract mode mismatch: observed={observed_contract_mode} expected={normalized_contract_mode}"
        )
    expected_model_contract_keys = {str(name) for name in expected_model_contract}
    observed_model_contract_keys = {str(name) for name in specialist_model_contract}
    if observed_model_contract_keys != expected_model_contract_keys:
        specialist_model_failures.append(
            "specialist model contract set mismatch: "
            f"observed={sorted(observed_model_contract_keys)} expected={sorted(expected_model_contract_keys)}"
        )
    for name, expected_spec in expected_model_contract.items():
        observed_spec = specialist_model_contract.get(name)
        if not isinstance(observed_spec, dict):
            specialist_model_failures.append(f"specialist model contract missing spec for {name}")
            continue
        if str(observed_spec.get("model_role") or "") != str(expected_spec.get("model_role") or ""):
            specialist_model_failures.append(f"specialist model contract model_role mismatch: {name}")
        for field in ("owned_objectives", "primary_signal_families", "supports_heads"):
            observed_values = tuple(str(x) for x in observed_spec.get(field) or ())
            expected_values = tuple(str(x) for x in expected_spec.get(field) or ())
            if observed_values != expected_values:
                specialist_model_failures.append(
                    f"specialist model contract {field} mismatch: {name}"
                )
    if specialist_model_failures:
        raise RuntimeError(
            "[SPECIALIST_MODEL_CONTRACT_INVALID] "
            f"{path} failures={specialist_model_failures[:5]}"
        )
    arch = report.get("architecture_contract") if isinstance(report.get("architecture_contract"), dict) else {}
    raw = arch.get("specialist_input_indices") if isinstance(arch.get("specialist_input_indices"), dict) else {}
    context_routing = (
        arch.get("context_specialist_routing")
        if isinstance(arch.get("context_specialist_routing"), dict)
        else {}
    )
    if report.get("context_specialist_routing_all_mapped") is not True:
        raise RuntimeError(
            "[SPECIALIST_CONTEXT_ROUTING_NOT_PROVEN] "
            f"failures={report.get('context_specialist_routing_failures')}"
        )
    if int(report.get("context_specialist_routing_failure_count") or 0) != 0:
        raise RuntimeError("[SPECIALIST_CONTEXT_ROUTING_FAILURES_PRESENT]")
    context_routing = require_model_native_context_specialist_routing(
        context_routing,
        ordered_signal_names=ordered_signal_names,
        context="SPECIALIST_AUDIT",
    )
    recommended = arch.get("recommended_fusion") if isinstance(arch.get("recommended_fusion"), dict) else {}
    active_heads = [str(head) for head in recommended.get("active_heads") or recommended.get("heads") or [] if str(head)]
    blocked_heads = [str(head) for head in recommended.get("blocked_heads") or [] if str(head)]
    if set(active_heads) != set(SPECIALIST_FUSION_ACTIVE_HEADS):
        raise RuntimeError(
            "[SPECIALIST_ACTIVE_HEADS_MISMATCH] "
            f"audit={sorted(active_heads)} expected={list(SPECIALIST_FUSION_ACTIVE_HEADS)}"
        )
    if set(blocked_heads) != set(SPECIALIST_FUSION_BLOCKED_HEADS):
        raise RuntimeError(
            "[SPECIALIST_BLOCKED_HEADS_MISMATCH] "
            f"audit={sorted(blocked_heads)} expected={list(SPECIALIST_FUSION_BLOCKED_HEADS)}"
        )
    if (
        recommended.get("independent_timeframe_only_head") is not None
        or recommended.get("independent_timeframe_only_head_allowed") is not False
    ):
        raise RuntimeError("[SPECIALIST_RETIRED_MTF_DIRECTION_HEAD_PRESENT]")
    overlap = sorted(set(active_heads) & set(blocked_heads))
    if overlap:
        raise RuntimeError(f"[SPECIALIST_HEADS_ACTIVE_AND_BLOCKED] {overlap}")
    trainable = set(required_training_specialists)
    blocked = {"forbidden_legacy_bridge", "unmapped"}
    indices: Dict[str, list[int]] = {}
    excluded_groups: Dict[str, int] = {}
    for name, values in raw.items():
        key = str(name)
        if key in blocked or key not in trainable:
            if isinstance(values, list) and values:
                excluded_groups[key] = int(len(values))
            continue
    seen_indices: set[int] = set()
    for key in required_training_specialists:
        values = raw.get(key)
        if not isinstance(values, list) or not values:
            raise RuntimeError(f"[SPECIALIST_REQUIRED_GROUP_INVALID] {key}")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise RuntimeError(f"[SPECIALIST_INDEX_TYPE_INVALID] {key}")
        idx = list(values)
        if idx != sorted(set(idx)):
            raise RuntimeError(f"[SPECIALIST_INDEX_ORDER_INVALID] {key}")
        if min(idx) < 0 or max(idx) >= int(expected_signal_dim):
            raise RuntimeError(f"[SPECIALIST_INDEX_OOB] {key}: min={min(idx)} max={max(idx)} dim={expected_signal_dim}")
        duplicate = seen_indices.intersection(idx)
        if duplicate:
            raise RuntimeError(
                f"[SPECIALIST_INDEX_OVERLAP] {key} overlap={sorted(duplicate)}"
            )
        seen_indices.update(idx)
        indices[key] = idx
    expected_indices = set(range(int(expected_signal_dim)))
    if seen_indices != expected_indices:
        missing_indices = sorted(expected_indices - seen_indices)
        unexpected_indices = sorted(seen_indices - expected_indices)
        raise RuntimeError(
            "[SPECIALIST_INDEX_COVERAGE_INVALID] "
            f"missing={missing_indices[:20]} total_missing={len(missing_indices)} "
            f"unexpected={unexpected_indices[:20]} total_unexpected={len(unexpected_indices)}"
        )
    for alias in context_routing["temporal_alias_policy"]["aliases"]:
        owner = str(alias["specialist"])
        signal_index = int(alias["signal_index"])
        if signal_index not in indices.get(owner, []):
            raise RuntimeError(
                "[SPECIALIST_CONTEXT_TEMPORAL_ALIAS_OWNER_MISMATCH] "
                f"field={alias['signal_field']} index={signal_index} "
                f"owner={owner}"
            )
    required = list(required_training_specialists)
    missing = [name for name in required if name not in indices]
    if missing:
        raise RuntimeError(f"[SPECIALIST_REQUIRED_GROUPS_MISSING] {missing}")
    meta = {
        "enabled": True,
        "audit_json": str(path),
        "audit_created_utc": str(report.get("created_utc") or ""),
        "signal_field_count": signal_dim,
        "selected_feature_count": int(report.get("selected_feature_count") or 0),
        "input_indices": indices,
        "group_feature_counts": {name: len(vals) for name, vals in indices.items()},
        "context_routing": context_routing,
        "contract_mode": normalized_contract_mode,
        "audit_contract_mode": observed_contract_mode,
        "trainable_specialists": list(required_training_specialists),
        "excluded_specialist_groups": excluded_groups,
        "active_heads": list(SPECIALIST_FUSION_ACTIVE_HEADS),
        "blocked_heads": list(SPECIALIST_FUSION_BLOCKED_HEADS),
        "specialist_model_contract": specialist_model_contract,
        "specialist_model_contract_valid": True,
        "specialist_model_contract_failures": [],
        "specialist_model_contract_set_exact": True,
        "specialist_model_contract_owned_objectives_match": True,
        "specialist_model_contract_signal_families_match": True,
        "specialist_model_contract_support_heads_match": True,
        "specialist_model_contract_model_roles_match": True,
        "specialist_model_contract_source": "entry_specialist_feature_group_audit",
    }
    return indices, meta

def _resolve_gx1_data(override: str = "") -> Path:
    base = Path(override or os.environ.get("GX1_DATA", "")).expanduser().resolve()
    if not base.is_dir():
        raise RuntimeError(f"GX1_DATA invalid or missing: {base}")
    return base

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("[CUDA_NOT_AVAILABLE] requested cuda but torch.cuda.is_available() is False")
    return torch.device(device_str)

def _set_deterministic(seed: int, device: torch.device) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
# -----------------------------------------------------------------------------
# Exact immutable dataset identity
# -----------------------------------------------------------------------------
_TRAIN_ARTIFACT_HASH_ENV = {
    "train_manifest": "GX1_ENTRY_TRAIN_MANIFEST_SHA256",
    "val_manifest": "GX1_ENTRY_VAL_MANIFEST_SHA256",
    "train_parquet": "GX1_ENTRY_TRAIN_PARQUET_SHA256",
    "val_parquet": "GX1_ENTRY_VAL_PARQUET_SHA256",
    "m5_prebuilt_path": "GX1_ENTRY_M5_PREBUILT_SHA256",
    "unified_exit_lifecycle_manifest": (
        "GX1_ENTRY_UNIFIED_EXIT_LIFECYCLE_MANIFEST_SHA256"
    ),
    "train_sequence_source_audit": (
        "GX1_ENTRY_TRAIN_SEQUENCE_SOURCE_AUDIT_SHA256"
    ),
    "val_sequence_source_audit": (
        "GX1_ENTRY_VAL_SEQUENCE_SOURCE_AUDIT_SHA256"
    ),
}
_TRAIN_DATASET_RUN_ID_ENV = "GX1_ENTRY_DATASET_RUN_ID"
# Recipe-validated absolute path of the mandatory verified multi-TF V2 disk
# cache. The launch contract emits this row; it is exact runtime identity, not
# an ambient control.
_TRAIN_MULTI_TF_CACHE_ENV = "GX1_V10_MULTI_TF_V4_CACHE_DIR"
# Scope identity published by scripts/gx1_capped_run.sh, which the trainer is
# required to run under. The runner re-reads these to verify that a nested
# capped job matches its parent scope. They name the cgroup the process already
# lives in - class, memory, swap, task ceiling and the already-enforced guard
# settings. The attended FIFO/token is also guard-created transport only; the
# trainer validates it immediately before model construction. None reaches a
# model input, target, threshold or checkpoint decision, so these are runtime
# identity rather than ambient control.
_TRAIN_CAPPED_SCOPE_ENV = (
    "GX1_CAPPED_CLASS",
    "GX1_CAPPED_MEMORY_BYTES",
    "GX1_CAPPED_SWAP_BYTES",
    "GX1_CAPPED_TASKS_MAX",
    "GX1_GPU_GUARD_PATH",
    # Guard-owned, exclusive sidecar created by gx1_capped_run.sh.  It records
    # safety telemetry only and has no bearing on model inputs, targets or
    # checkpoint decisions.
    "GX1_TRAINER_GUARD_LOG_PATH",
    "GX1_TRAINER_DEVICE",
    "GX1_TRAINER_EXECUTION_MODE",
    "GX1_TRAINER_MAX_WALL_SECONDS",
    "GX1_TRAINER_MODEL_MAX_WALL_SECONDS",
    "GX1_TRAINER_ATTENDED_STAGE_REQUIRED",
    "GX1_TRAINER_ATTENDED_STAGE_FIFO",
    "GX1_TRAINER_ATTENDED_STAGE_TOKEN",
    "GX1_TRAINER_GPU_INDEX",
    "GX1_TRAINER_GPU_MAX_CORE_TEMP_C",
    "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C",
    "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W",
    "GX1_TRAINER_GPU_MAX_POWER_DRAW_W",
    "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB",
    "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS",
    "GX1_TRAINER_NVIDIA_SMI_PATH",
    # The capped guard, not the model, owns canonical host telemetry.  These
    # values must traverse the trainer process so its parent guard can keep
    # its signed canonical path distinct from the attended native-SMI path.
    # They never affect model inputs, targets, loss or checkpoint selection.
    "GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH",
    "GX1_TRAINER_HOST_TELEMETRY_URL",
    "GX1_TRAINER_HOST_TELEMETRY_CERT_PATH",
    "GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256",
    "GX1_TRAINER_HOST_TELEMETRY_GPU_UUID",
    "GX1_TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS",
)
def _explicit_regular_artifact(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise RuntimeError(f"[ENTRY_TRAIN_ARTIFACT_PATH_NOT_ABSOLUTE] {label}={raw}")
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"[ENTRY_TRAIN_ARTIFACT_NOT_REGULAR] {label}={raw}")
    resolved = raw.resolve()
    if resolved != raw or any("latest" in part.lower() for part in raw.parts):
        raise RuntimeError(f"[ENTRY_TRAIN_ARTIFACT_PATH_MUTABLE] {label}={raw}")
    return resolved


def _expected_train_artifact_sha256(label: str) -> str:
    env_name = _TRAIN_ARTIFACT_HASH_ENV[label]
    value = str(os.environ.get(env_name) or "").strip().lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"[ENTRY_TRAIN_ARTIFACT_HASH_ENV_INVALID] {env_name}")
    return value


def _require_bound_sequence_source_audit(
    path: Path,
    *,
    split: str,
) -> Path:
    """Return one recipe-bound source-window proof without a TOCTOU gap."""

    label = f"{split}_sequence_source_audit"
    audit_path = _explicit_regular_artifact(path, label=label)
    expected_sha = _expected_train_artifact_sha256(label)
    observed_sha = _sha256_file(audit_path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            "[ENTRY_TRAIN_ARTIFACT_SHA256_MISMATCH] "
            f"{label} expected={expected_sha} observed={observed_sha}"
        )
    return audit_path


def _resolve_explicit_train_split_artifacts(
    *,
    train_manifest: Path,
    val_manifest: Path,
    train_parquet: Path,
    val_parquet: Path,
    unified_exit_lifecycle_manifest_path: Path,
    m5_prebuilt_path: Path,
    dataset_run_id: str,
    profile: str,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """Verify every launch-bound dataset artifact without discovery/inference."""

    if profile not in ("smoke", "candidate"):
        raise RuntimeError(f"[ENTRY_TRAIN_PROFILE_INVALID] {profile!r}")
    expected_dataset_run_id = str(
        os.environ.get(_TRAIN_DATASET_RUN_ID_ENV) or ""
    ).strip()
    if not expected_dataset_run_id or expected_dataset_run_id != dataset_run_id:
        raise RuntimeError(
            "[ENTRY_TRAIN_DATASET_RUN_ID_ENV_MISMATCH] "
            f"cli={dataset_run_id!r} env={expected_dataset_run_id!r}"
        )

    manifests = {
        "train": _explicit_regular_artifact(train_manifest, label="train_manifest"),
        "val": _explicit_regular_artifact(val_manifest, label="val_manifest"),
    }
    parquets = {
        "train": _explicit_regular_artifact(train_parquet, label="train_parquet"),
        "val": _explicit_regular_artifact(val_parquet, label="val_parquet"),
    }
    m5_prebuilt = _explicit_regular_artifact(
        m5_prebuilt_path,
        label="m5_prebuilt_path",
    )
    lifecycle_manifest = _explicit_regular_artifact(
        unified_exit_lifecycle_manifest_path,
        label="unified_exit_lifecycle_manifest",
    )
    expected_lifecycle_sha256 = _expected_train_artifact_sha256(
        "unified_exit_lifecycle_manifest"
    )
    observed_lifecycle_sha256 = _sha256_file(lifecycle_manifest)
    if observed_lifecycle_sha256 != expected_lifecycle_sha256:
        raise RuntimeError(
            "[ENTRY_TRAIN_ARTIFACT_SHA256_MISMATCH] "
            "unified_exit_lifecycle_manifest "
            f"expected={expected_lifecycle_sha256} "
            f"observed={observed_lifecycle_sha256}"
        )
    expected_m5_sha256 = _expected_train_artifact_sha256("m5_prebuilt_path")
    observed_m5_sha256 = _sha256_file(m5_prebuilt)
    if observed_m5_sha256 != expected_m5_sha256:
        raise RuntimeError(
            "[ENTRY_TRAIN_ARTIFACT_SHA256_MISMATCH] m5_prebuilt_path "
            f"expected={expected_m5_sha256} observed={observed_m5_sha256}"
        )
    all_paths = tuple(manifests.values()) + tuple(parquets.values())
    if len(set(all_paths)) != len(all_paths):
        raise RuntimeError("[ENTRY_TRAIN_SPLIT_ARTIFACT_PATHS_NOT_DISTINCT]")
    parents = {path.parent for path in all_paths}
    if len(parents) != 1:
        raise RuntimeError(
            f"[ENTRY_TRAIN_SPLIT_ARTIFACT_PARENT_MISMATCH] parents={sorted(map(str, parents))}"
        )

    reference_contract: Dict[str, Any] | None = None
    reference_state_contract: Dict[str, Any] | None = None
    reference_mtf_cache_binding: Dict[str, str] | None = None
    for split in ("train", "val"):
        manifest_path = manifests[split]
        parquet_path = parquets[split]
        for label, path in (
            (f"{split}_manifest", manifest_path),
            (f"{split}_parquet", parquet_path),
        ):
            expected_sha = _expected_train_artifact_sha256(label)
            observed_sha = _sha256_file(path)
            if observed_sha != expected_sha:
                raise RuntimeError(
                    f"[ENTRY_TRAIN_ARTIFACT_SHA256_MISMATCH] {label} "
                    f"expected={expected_sha} observed={observed_sha}"
                )

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"[ENTRY_TRAIN_SPLIT_MANIFEST_INVALID_JSON] {split}={manifest_path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"[ENTRY_TRAIN_SPLIT_MANIFEST_NOT_OBJECT] {split}")
        if payload.get("schema_version") != MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(f"[ENTRY_TRAIN_SPLIT_MANIFEST_SCHEMA_MISMATCH] {split}")
        if payload.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE:
            raise RuntimeError(f"[ENTRY_TRAIN_SPLIT_MANIFEST_MODE_MISMATCH] {split}")
        declared_raw = str(payload.get("output_data_path") or "").strip()
        declared = Path(declared_raw).expanduser()
        if not declared.is_absolute() or declared != parquet_path:
            raise RuntimeError(
                f"[ENTRY_TRAIN_SPLIT_MANIFEST_SELF_PATH_MISMATCH] {split}: "
                f"declared={declared_raw!r} expected={parquet_path}"
            )
        if not manifest_path.name.endswith(
            f"_{split}.manifest.json"
        ) or not parquet_path.name.endswith(f"_{split}.parquet"):
            raise RuntimeError(f"[ENTRY_TRAIN_SPLIT_FILENAME_MISMATCH] {split}")
        contract = _signal_contract_from_manifest_obj(payload)
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise RuntimeError("[ENTRY_TRAIN_SPLIT_SIGNAL_CONTRACT_MISMATCH]")
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        mtf_cache_binding = require_dataset_manifest_multi_tf_cache_binding(
            payload,
            dataset_run_id=dataset_run_id,
            context=f"ENTRY_TRAIN_{split.upper()}_MTF_BINDING",
        )
        if reference_mtf_cache_binding is None:
            reference_mtf_cache_binding = mtf_cache_binding
        elif mtf_cache_binding != reference_mtf_cache_binding:
            raise RuntimeError(
                "[ENTRY_TRAIN_SPLIT_MTF_CACHE_BINDING_MISMATCH]"
            )
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
        declared_m5_raw = str(inputs.get("source_parquet") or "").strip()
        declared_m5 = Path(declared_m5_raw).expanduser()
        if not declared_m5.is_absolute() or declared_m5 != m5_prebuilt:
            raise RuntimeError(
                f"[ENTRY_TRAIN_SPLIT_M5_SOURCE_PATH_MISMATCH] {split}: "
                f"declared={declared_m5_raw!r} expected={m5_prebuilt}"
            )
        state_contract = (
            extra.get("model_native_state_contract")
            if isinstance(extra.get("model_native_state_contract"), dict)
            else None
        )
        try:
            validate_state_contract_metadata_v2(state_contract)
        except RuntimeError as exc:
            raise RuntimeError(
                f"[ENTRY_TRAIN_SPLIT_STATE_BINDING_MISMATCH] {split}: {exc}"
            ) from exc
        manifest_dataset_run_id = extra.get("entry_run_id")
        state_dataset_run_id = (
            state_contract.get("entry_run_id")
            if state_contract is not None
            else None
        )
        if (
            manifest_dataset_run_id != dataset_run_id
            or state_dataset_run_id != dataset_run_id
        ):
            raise RuntimeError(f"[ENTRY_TRAIN_SPLIT_RUN_ID_LINEAGE_MISMATCH] {split}")
        if reference_state_contract is None:
            reference_state_contract = state_contract
        elif state_contract != reference_state_contract:
            raise RuntimeError("[ENTRY_TRAIN_SPLIT_STATE_CONTRACT_MISMATCH]")
        if _sha256_file(manifest_path) != _expected_train_artifact_sha256(
            f"{split}_manifest"
        ):
            raise RuntimeError(f"[ENTRY_TRAIN_SPLIT_MANIFEST_CHANGED] {split}")
        log.info(
            "[ENTRY_DATASET_MANIFEST_PROOF] split=%s manifest=%s parquet=%s sha256=%s",
            split,
            manifest_path,
            parquet_path,
            _expected_train_artifact_sha256(f"{split}_parquet"),
        )
    if reference_mtf_cache_binding is None:
        raise RuntimeError("[ENTRY_TRAIN_MTF_CACHE_BINDING_MISSING]")
    cache_dir_raw = str(os.environ.get(_TRAIN_MULTI_TF_CACHE_ENV) or "").strip()
    if not cache_dir_raw:
        raise RuntimeError("[ENTRY_TRAIN_MTF_CACHE_ENV_MISSING]")
    require_multi_tf_v4_cache_binding_files(
        reference_mtf_cache_binding,
        expected_cache_dir=Path(cache_dir_raw),
        context="ENTRY_TRAIN_LAUNCH_MTF_BINDING",
    )
    return manifests, parquets


def _signal_contract_from_manifest_obj(data: Dict[str, Any]) -> Dict[str, Any]:
    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
    sb = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    contract_mode = str(extra.get("contract_mode") or "").strip()
    if contract_mode != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            "[ENTRY_DATASET_MODEL_NATIVE_MODE_REQUIRED] "
            f"got={contract_mode!r} expected={MODEL_NATIVE_CONTRACT_MODE!r}"
        )
    direction_logit_mode = str(extra.get("direction_logit_mode") or "").strip()
    if direction_logit_mode != MODEL_NATIVE_DIRECTION_LOGIT_MODE:
        raise RuntimeError(
            "[ENTRY_DATASET_MODEL_NATIVE_DIRECTION_MODE_REQUIRED] "
            f"got={direction_logit_mode!r} expected={MODEL_NATIVE_DIRECTION_LOGIT_MODE!r}"
        )
    model_native_signal_contract = extra.get("model_native_signal_contract")
    if not isinstance(model_native_signal_contract, dict):
        raise RuntimeError("[ENTRY_DATASET_MODEL_NATIVE_SIGNAL_CONTRACT_MISSING]")
    require_model_native_signal_contract(
        model_native_signal_contract,
        context="ENTRY_DATASET",
    )
    fields_raw = sb.get("fields")
    if not isinstance(fields_raw, list):
        raise RuntimeError("[ENTRY_DATASET_MODEL_NATIVE_SIGNAL_FIELDS_MISSING]")
    fields = [str(x) for x in fields_raw]
    seq_dim = int(sb.get("seq_input_dim") or 0)
    snap_dim = int(sb.get("snap_input_dim") or 0)
    if seq_dim != MODEL_NATIVE_SIGNAL_DIM or snap_dim != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "[ENTRY_DATASET_MODEL_NATIVE_SIGNAL_DIM_INVALID] "
            f"seq={seq_dim} snap={snap_dim} expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    if fields != list(model_native_signal_contract["fields"]):
        raise RuntimeError("[ENTRY_DATASET_MODEL_NATIVE_SIGNAL_ORDER_MISMATCH]")
    aux_head_target_contract = _require_model_native_aux_target_emission_contract(
        extra.get("aux_head_target_contract"),
        context="ENTRY_DATASET",
    )
    return {
        "seq_input_dim": seq_dim,
        "snap_input_dim": snap_dim,
        "fields": fields,
        "contract_mode": contract_mode,
        "direction_logit_mode": direction_logit_mode,
        "model_native_signal_contract": model_native_signal_contract,
        "aux_head_target_contract": aux_head_target_contract,
    }


def _signal_contract_from_manifest_path(dataset_manifest: Optional[Path]) -> Dict[str, Any]:
    if dataset_manifest is None:
        raise RuntimeError("[ENTRY_MODEL_NATIVE_DATASET_MANIFEST_REQUIRED]")
    p = Path(dataset_manifest).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"[ENTRY_MODEL_NATIVE_DATASET_MANIFEST_MISSING] {p}")
    return _signal_contract_from_manifest_obj(json.loads(p.read_text(encoding="utf-8")))


def _model_native_state_contract_from_manifest_obj(data: Dict[str, Any]) -> Dict[str, Any]:
    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
    contract = extra.get("model_native_state_contract")
    return dict(contract) if isinstance(contract, dict) else {}


def _model_native_state_contract_from_manifest_path(dataset_manifest: Optional[Path]) -> Dict[str, Any]:
    if dataset_manifest is None:
        return {}
    p = Path(dataset_manifest).expanduser().resolve()
    if not p.exists():
        return {}
    return _model_native_state_contract_from_manifest_obj(json.loads(p.read_text(encoding="utf-8")))


def _model_native_state_contract_for_parquet(parquet_path: Path) -> Dict[str, Any]:
    return _model_native_state_contract_from_manifest_path(
        Path(parquet_path).expanduser().resolve().with_suffix(".manifest.json")
    )


def _entry_position_size_target_policy_from_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("[ENTRY_POSITION_SIZE_POLICY_MANIFEST_NOT_OBJECT]")
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    splits = payload.get("splits")
    train_window = (
        splits.get("train")
        if isinstance(splits, dict) and isinstance(splits.get("train"), dict)
        else {}
    )
    if "entry_causal_m1_position_size_target_policy" not in extra:
        return require_entry_position_size_target_manifest_binding(
            extra,
            expected_train_start=train_window.get("start"),
            expected_train_end=train_window.get("end"),
        )
    lifecycle = extra.get("unified_exit_lifecycle")
    expected_m1_source = (
        str(lifecycle.get("m1_source_sha256") or "").strip().lower()
        if isinstance(lifecycle, dict)
        else ""
    )
    if len(expected_m1_source) != 64:
        raise RuntimeError("[ENTRY_POSITION_SIZE_POLICY_M1_SOURCE_MISSING]")
    return require_causal_m1_position_size_target_manifest_binding(
        extra,
        expected_m1_source_sha256=expected_m1_source,
        expected_train_start=train_window.get("start"),
        expected_train_end=train_window.get("end"),
    )


def _model_native_state_contract_failures(contract: Dict[str, Any], *, split: str) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return [f"{split} manifest missing model_native_state_contract for XAU direction repair"]
    try:
        validate_state_contract_metadata_v2(contract)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        return [f"{split} model_native_state_contract v2 invalid: {exc}"]
    return []


def _signal_contract_for_parquet(parquet_path: Path, seq_dim: int, snap_dim: int) -> Dict[str, Any]:
    manifest_path = Path(parquet_path).expanduser().resolve().with_suffix(".manifest.json")
    contract = _signal_contract_from_manifest_path(manifest_path)
    if int(contract["seq_input_dim"]) != int(seq_dim) or int(contract["snap_input_dim"]) != int(snap_dim):
        raise RuntimeError(
            "[ENTRY_V10_CTX_MANIFEST_SIGNAL_DIM_MISMATCH] "
            f"{manifest_path} declares seq/snap={contract['seq_input_dim']}/{contract['snap_input_dim']} "
            f"but parquet has {seq_dim}/{snap_dim}"
        )
    return contract


def _xau_direction_repair_source_failures(paths: Dict[str, Any]) -> list[str]:
    failures: list[str] = []
    stale_markers = (
        "utilityrepair",
        "20260710",
        "smart_candidate_20260630",
        "julyext",
    )
    for label, raw_path in paths.items():
        text = str(raw_path or "").strip()
        low = text.lower()
        if not text:
            failures.append(f"{label} missing for XAU direction repair")
            continue
        for marker in stale_markers:
            if marker in low:
                failures.append(f"{label} references stale pre-repair dataset marker {marker!r}: {text}")
    return failures


def _xau_direction_repair_manifest_failures(parquet_paths: Dict[str, Any]) -> list[str]:
    failures: list[str] = []
    state_contracts: Dict[str, Dict[str, Any]] = {}
    tape_provenance_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
    tape_provenance_by_split: Dict[str, Dict[str, Any]] = {}
    for split, raw_path in parquet_paths.items():
        parquet_path = Path(raw_path).expanduser()
        manifest_path = parquet_path.with_suffix(".manifest.json")
        if manifest_path.is_symlink() or not manifest_path.is_file():
            failures.append(f"{split} manifest missing for XAU direction repair: {manifest_path}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{split} manifest unreadable for XAU direction repair: {manifest_path}: {exc}")
            continue
        extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
        inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
        try:
            _signal_contract_from_manifest_obj(manifest)
        except RuntimeError as exc:
            failures.append(f"{split} {exc}")
        tape_root = str(
            manifest.get("tape_root")
            or extra.get("tape_root")
            or inputs.get("tape_root")
            or ""
        ).strip()
        state_contract = _model_native_state_contract_from_manifest_obj(manifest)
        failures.extend(_model_native_state_contract_failures(state_contract, split=split))
        expected_run_id = str(state_contract.get("entry_run_id") or "").strip()
        cache_key = (tape_root, expected_run_id)
        try:
            if cache_key not in tape_provenance_cache:
                tape_provenance_cache[cache_key] = validate_xau_tape_provenance_v1(
                    tape_root,
                    expected_run_id=expected_run_id,
                    require_current=True,
                )
            tape_provenance_by_split[split] = tape_provenance_cache[cache_key]
            if extra.get("xau_tape_provenance") != tape_provenance_by_split[split]:
                failures.append(
                    f"{split} dataset manifest XAU_USD tape binding differs from "
                    "the revalidated immutable tape lineage"
                )
        except (RuntimeError, OSError, ValueError) as exc:
            failures.append(f"{split} immutable XAU_USD tape provenance invalid: {exc}")
        split_windows = manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {}
        train_window = (
            split_windows.get("train")
            if isinstance(split_windows.get("train"), dict)
            else {}
        )
        if not train_window:
            failures.append(f"{split} manifest missing exact TRAIN split window")
        if state_contract:
            state_contracts[split] = state_contract
    if len(state_contracts) > 1:
        baseline_split = next(iter(state_contracts))
        baseline = state_contracts[baseline_split]
        for split, contract in state_contracts.items():
            if contract != baseline:
                failures.append(
                    f"{split} model_native_state_contract differs from {baseline_split}; "
                    "TRAIN/VAL/TEST must share one immutable history contract"
                )
    if len(tape_provenance_by_split) > 1:
        baseline_split = next(iter(tape_provenance_by_split))
        baseline = tape_provenance_by_split[baseline_split]
        for split, proof in tape_provenance_by_split.items():
            if proof != baseline:
                failures.append(
                    f"{split} immutable XAU_USD tape provenance differs from "
                    f"{baseline_split}; TRAIN/VAL must share one exact tape lineage"
                )
    return failures


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# Module-level cache for the one verified causal V4 multi-TF artifact. The
# contract-qualified key is shared by the pre-train load and every dataset
# instance so one source/cache identity produces one in-process object.
_MULTI_TF_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}
_MULTI_TF_ACTIVE_CACHE_KEYS: Dict[str, str] = {}
_MULTI_TF_CACHE_CONTRACT = "V4_EIGHT_FAMILY_CAUSAL"


def _multi_tf_cache_key(
    m5_prebuilt_path: Path,
    *,
    source_sha256: Optional[str] = None,
    backend_identity: str = "verified_v4_cache",
    contract_mode: str = _MULTI_TF_CACHE_CONTRACT,
) -> str:
    if contract_mode != _MULTI_TF_CACHE_CONTRACT:
        raise RuntimeError(
            "[MULTI_TF_CACHE_CONTRACT_MODE_INVALID] "
            f"observed={contract_mode!r} expected={_MULTI_TF_CACHE_CONTRACT!r}"
        )
    source_path = Path(m5_prebuilt_path).expanduser().resolve()
    observed_source_sha256 = (
        str(source_sha256).strip().lower()
        if source_sha256 is not None
        else _sha256_file(source_path)
    )
    if (
        len(observed_source_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in observed_source_sha256)
    ):
        raise RuntimeError("[MULTI_TF_CACHE_SOURCE_SHA256_INVALID]")
    normalized_backend = str(backend_identity).strip()
    if not normalized_backend:
        raise RuntimeError("[MULTI_TF_CACHE_BACKEND_IDENTITY_INVALID]")
    return (
        f"{source_path}"
        f"|source_sha256={observed_source_sha256}"
        f"|backend={normalized_backend}"
        f"|contract={_MULTI_TF_CACHE_CONTRACT}"
    )


def _prebuild_multi_tf_features_once(
    m5_prebuilt_path: Path,
) -> Dict[str, pd.DataFrame]:
    """Load the exact verified V4 cache once under a byte-bound identity."""

    require_entry_exit_production_architecture(
        current_entry_exit_architecture_observation(),
        context="ENTRY_V10_MTF_CACHE_LOAD",
    )

    supplied = Path(m5_prebuilt_path).expanduser()
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    if any(component.is_symlink() for component in (absolute, *absolute.parents)):
        raise RuntimeError(
            f"[MULTI_TF_SOURCE_CONTRACT] source path traverses symlink: {absolute}"
        )
    try:
        m5_path = absolute.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"[MULTI_TF_INIT_FAIL] M5 prebuilt missing: {absolute}"
        ) from exc
    if not m5_path.is_file() or m5_path.is_symlink():
        raise FileNotFoundError(f"[MULTI_TF_INIT_FAIL] M5 prebuilt missing: {m5_path}")
    source_sha256 = _sha256_file(m5_path)
    disk_cache_raw = os.environ.get(_TRAIN_MULTI_TF_CACHE_ENV, "").strip()
    if not disk_cache_raw:
        raise RuntimeError(
            "[MULTI_TF_V4_CACHE_DIR_REQUIRED] source-build fallback is forbidden"
        )
    backend_locator = (
        f"disk_path:{Path(disk_cache_raw).expanduser().resolve()}"
    )
    active_identity = (
        f"{m5_path}|source_sha256={source_sha256}"
        f"|backend_locator={backend_locator}"
        f"|contract={_MULTI_TF_CACHE_CONTRACT}"
    )
    active_cache_key = _MULTI_TF_ACTIVE_CACHE_KEYS.get(active_identity)
    if active_cache_key is not None:
        active_cached = _MULTI_TF_CACHE.get(active_cache_key)
        if active_cached is None:
            raise RuntimeError("[MULTI_TF_CACHE_ACTIVE_IDENTITY_DANGLING]")
        return active_cached

    from gx1.features.htf_features import load_multi_tf_v4_cache

    loaded = load_multi_tf_v4_cache(disk_cache_raw)
    # The verified V4 cache binds its own full-history canonical M5 source
    # (the cascade-audited canonical-v3 parquet). The trainer's
    # --m5-prebuilt-path is the model-range seq/snapshot source — a distinct
    # identity bound through the split manifests — so the cache source is
    # proven against its own declared bytes, not against m5_path.
    cache_source = Path(
        str(getattr(loaded, "m5_prebuilt_source", "") or "")
    ).expanduser()
    cache_source_sha256 = str(
        getattr(loaded, "m5_prebuilt_source_sha256", "") or ""
    )
    if (
        not cache_source.is_absolute()
        or not cache_source.is_file()
        or cache_source.is_symlink()
    ):
        raise RuntimeError(
            "[MULTI_TF_CACHE_SOURCE_MISSING] "
            f"cache_source={str(cache_source)!r}"
        )
    observed_cache_source_sha256 = _sha256_file(cache_source)
    if observed_cache_source_sha256 != cache_source_sha256:
        raise RuntimeError(
            "[MULTI_TF_CACHE_SOURCE_BINDING_MISMATCH] "
            f"cache_source={str(cache_source)!r} "
            f"declared_sha256={cache_source_sha256} "
            f"observed_sha256={observed_cache_source_sha256}"
        )
    cache_identity_sha256 = str(
        getattr(loaded, "cache_identity_sha256", "")
    )
    cache_key = _multi_tf_cache_key(
        m5_path,
        source_sha256=source_sha256,
        backend_identity=(
            f"{backend_locator}:cache_identity={cache_identity_sha256}"
        ),
    )
    cached = _MULTI_TF_CACHE.get(cache_key)
    if cached is not None:
        _MULTI_TF_ACTIVE_CACHE_KEYS[active_identity] = cache_key
        return cached

    import pyarrow.parquet as pq

    prebuilt_max = pd.to_datetime(
        pq.read_table(m5_path, columns=["time"])
        .column("time")
        .to_numpy(zero_copy_only=False)
        .max(),
        utc=True,
    )
    cache_max = loaded["M5"].index.max()
    if cache_max != prebuilt_max:
        raise RuntimeError(
            "[MULTI_TF_CACHE_STALE] "
            f"disk cache M5 max {cache_max} does not exactly match "
            f"prebuilt max {prebuilt_max}"
        )
    _MULTI_TF_CACHE[cache_key] = loaded
    _MULTI_TF_ACTIVE_CACHE_KEYS[active_identity] = cache_key
    return loaded


def _deterministic_liveness_storage_indices(
    dataset: object,
    *,
    sample_rows: int,
) -> np.ndarray:
    """Map deterministic dataset samples to the currently materialized arrays."""

    selected_rows = np.asarray(getattr(dataset, "indices", None), dtype=np.int64)
    if selected_rows.ndim != 1 or selected_rows.size < 1:
        raise RuntimeError("[FEATURE_LIVENESS_INDEX_MAP_INVALID] empty selection")
    rows = min(int(sample_rows), int(selected_rows.size))
    if rows <= 0:
        raise RuntimeError("[FEATURE_LIVENESS_SAMPLE_ROWS_INVALID]")
    dataset_offsets = np.linspace(
        0,
        selected_rows.size - 1,
        num=rows,
        dtype=np.int64,
    )
    if bool(getattr(dataset, "_sequence_source_reconstructed", False)):
        # Source-backed windows have no N×96 materialised array.  Return the
        # immutable split row ids; the caller resolves each via the source
        # surface and separately maps snap/context storage if compacted.
        return selected_rows[dataset_offsets]
    compact_rows = getattr(dataset, "_compact_row_indices", None)
    if compact_rows is None:
        storage_indices = selected_rows[dataset_offsets]
    else:
        compact_rows = np.asarray(compact_rows, dtype=np.int64)
        if not np.array_equal(compact_rows, selected_rows):
            raise RuntimeError(
                "[FEATURE_LIVENESS_COMPACT_INDEX_MAP_INVALID] compact rows do not "
                "match the dataset selection"
            )
        storage_indices = dataset_offsets
    for array_name in ("_np_seq", "_np_snap", "_np_ctx_cont"):
        array = getattr(dataset, array_name, None)
        if array is None or int(array.shape[0]) <= int(storage_indices.max()):
            raise RuntimeError(
                "[FEATURE_LIVENESS_STORAGE_INDEX_OOB] "
                f"array={array_name} rows={getattr(array, 'shape', None)} "
                f"max_index={int(storage_indices.max())}"
            )
    return storage_indices


class EntryV10CtxDataset(Dataset):
    """
    Builds rolling-window samples from canonical ENTRY_V10_CTX parquet.
    ctx_cont / ctx_cat are per-sample (B, N), not per-timestep.

    The dataset owns one shared M5/M15/H1/H4/D1 V4 cache. Entry slices only
    M15/H1/H4/D1; Exit slices M5/M15/H1/H4/D1 at its native M1 clock. Source is
    the M5 canonical_v3 prebuilt, resampled and
    feature-engineered once at __init__. Resampled tables are cached at module
    level so train_ds + val_ds share them. Adds ~25s init time first call,
    instant on subsequent dataset instantiations with same prebuilt path.
    """

    def __init__(
        self,
        parquet_path: Path,
        seq_len: int,
        m5_prebuilt_path: Path,
        per_tf_seq_lens: Dict[str, int],
        multi_tf_closed_bar: bool,
        sequence_roll_audit_json: Optional[Path] = None,
        sequence_source_audit_json: Optional[Path] = None,
    ):
        architecture = current_entry_exit_architecture_observation()
        architecture["entry"]["sequence_bars"] = seq_len
        architecture["exit"]["sequence_bars"] = (
            seq_len * ENTRY_EXIT_RESOLUTION_RATIO
            if isinstance(seq_len, int) and not isinstance(seq_len, bool)
            else seq_len
        )
        architecture["mtf"]["cache_timeframes"] = (
            list(per_tf_seq_lens)
            if isinstance(per_tf_seq_lens, Mapping)
            else per_tf_seq_lens
        )
        architecture["mtf"]["per_tf_window_bars"] = (
            dict(per_tf_seq_lens)
            if isinstance(per_tf_seq_lens, Mapping)
            else per_tf_seq_lens
        )
        require_entry_exit_production_architecture(
            architecture,
            context="ENTRY_V10_DATASET_CONSTRUCTION",
        )
        self.parquet_path = Path(parquet_path)
        self.seq_len = int(seq_len)
        if multi_tf_closed_bar is not True:
            raise RuntimeError(
                "ENTRY_MULTI_TF_CAUSALITY: explicit closed-bar=True is required"
            )
        self._multi_tf_closed_bar = True
        expected_timeframes = MULTI_TF_TIMEFRAMES
        if (
            not isinstance(per_tf_seq_lens, dict)
            or tuple(per_tf_seq_lens) != expected_timeframes
            or any(
                isinstance(per_tf_seq_lens[tf], bool)
                or int(per_tf_seq_lens[tf]) <= 0
                for tf in expected_timeframes
            )
        ):
            raise RuntimeError(
                "ENTRY_PER_TF_SEQ_LEN_CONTRACT_INVALID: exact ordered positive "
                "M5/M15/H1/H4/D1 mapping required; fallback is forbidden"
            )
        self.per_tf_seq_lens = {
            tf: int(per_tf_seq_lens[tf]) for tf in expected_timeframes
        }
        self._multi_tf_feats: Optional[Dict[str, pd.DataFrame]] = None
        self._multi_tf_feature_count: int = 0
        self._memmap_tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self._sequence_reconstruction_chain: Optional[np.ndarray] = None
        self._sequence_roll_reconstructed = False
        self._sequence_roll_audit: Optional[dict[str, Any]] = None
        self._sequence_source_reconstructed = False
        self._sequence_source_audit: Optional[dict[str, Any]] = None
        self._sequence_source_times_ns: Optional[np.ndarray] = None
        self._sequence_source_signal: Optional[np.ndarray] = None
        self._sequence_source_positions: Optional[np.ndarray] = None
        # When a bounded smoke uses a uniform subset, this maps the compact
        # in-memory rows back to their original immutable parquet row ids.  The
        # full dataset/lifecycle row space remains authoritative; this is only
        # a storage optimization after TRAIN normalization has been fitted.
        self._compact_row_indices: Optional[np.ndarray] = None
        self._unified_exit_lifecycle: Optional[
            UnifiedExitLifecycleSplit
        ] = None

        if not self.parquet_path.exists():
            raise FileNotFoundError(self.parquet_path)
        if sequence_roll_audit_json is not None and sequence_source_audit_json is not None:
            raise RuntimeError(
                "[ENTRY_SEQUENCE_RECONSTRUCTION_PROOF_MODE_AMBIGUOUS]"
            )
        sequence_roll_audit: Optional[dict[str, Any]] = None
        if sequence_roll_audit_json is not None:
            sequence_roll_audit = _require_sequence_roll_audit(
                Path(sequence_roll_audit_json),
                parquet_path=self.parquet_path,
                manifest_path=self.parquet_path.with_suffix(".manifest.json"),
            )
            self._sequence_roll_audit = dict(sequence_roll_audit)
        sequence_source_audit: Optional[dict[str, Any]] = None
        if sequence_source_audit_json is not None:
            (
                sequence_source_audit,
                self._sequence_source_times_ns,
                self._sequence_source_signal,
            ) = _require_sequence_source_reconstruction(
                Path(sequence_source_audit_json),
                parquet_path=self.parquet_path,
                manifest_path=self.parquet_path.with_suffix(".manifest.json"),
            )
            self._sequence_source_audit = dict(sequence_source_audit)
            self._sequence_source_reconstructed = True
        signal_contract = _signal_contract_from_manifest_path(
            self.parquet_path.with_suffix(".manifest.json")
        )
        self.seq_input_dim = int(signal_contract["seq_input_dim"])
        self.snap_input_dim = int(signal_contract["snap_input_dim"])
        self.signal_names = list(signal_contract["fields"])
        self.contract_mode = MODEL_NATIVE_CONTRACT_MODE
        self.direction_logit_mode = MODEL_NATIVE_DIRECTION_LOGIT_MODE
        self.model_native_signal_contract = signal_contract["model_native_signal_contract"]
        self.aux_head_target_contract = signal_contract["aux_head_target_contract"]

        manifest_architecture = current_entry_exit_architecture_observation()
        manifest_architecture["entry"]["sequence_bars"] = self.seq_len
        manifest_architecture["exit"]["sequence_bars"] = (
            self.seq_len * ENTRY_EXIT_RESOLUTION_RATIO
        )
        manifest_architecture["shared_surface"]["signal_dim"] = (
            self.seq_input_dim
        )
        manifest_architecture["shared_surface"]["snap_dim"] = (
            self.snap_input_dim
        )
        manifest_architecture["schemas"]["signal"] = (
            self.model_native_signal_contract.get("schema_version")
            if isinstance(self.model_native_signal_contract, Mapping)
            else None
        )
        require_entry_exit_production_architecture(
            manifest_architecture,
            context="ENTRY_V10_DATASET_MANIFEST",
        )

        # ── Memory fix (V12.2): EXCLUDE nested-list columns from pandas load.
        # The nested-list columns (seq/snap/ctx_cont/ctx_cat) stored as
        # list<list<double>> blow up to ~30GB when materialized in pandas.
        # We re-read them via chunked pyarrow into pre-allocated numpy arrays
        # below for the only admitted nested model-native schema.
        import pyarrow.parquet as pq
        _all_cols = pq.ParquetFile(self.parquet_path).schema_arrow.names
        _nested_cols = {"seq", "snap", "ctx_cont", "ctx_cat"}
        missing_nested = sorted(_nested_cols - set(_all_cols))
        if missing_nested:
            raise RuntimeError(
                "[ENTRY_MODEL_NATIVE_NESTED_SCHEMA_REQUIRED] "
                f"missing={missing_nested} parquet={self.parquet_path}"
            )
        _load_cols = [c for c in _all_cols if c not in _nested_cols]
        df = pd.read_parquet(self.parquet_path, columns=_load_cols)
        # Re-add empty stubs for downstream presence checks. They are replaced
        # by the chunked Arrow arrays below.
        for _c in ("seq", "snap", "ctx_cont", "ctx_cat"):
            df[_c] = None

        if "seq" in df.columns:
            # ---- advanced schema: builder has prebuilt samples
            required_advanced = [
                "time",
                "seq",
                "snap",
                "ctx_cont",
                "ctx_cat",
                *_MODEL_NATIVE_ACTIVE_TARGET_COLS,
            ]
            target_failures = _model_native_active_target_failures(
                self.parquet_path.stem,
                df,
            )
            if target_failures:
                raise RuntimeError(
                    "[ENTRY_V10_CTX_MODEL_NATIVE_ACTIVE_TARGET_CONTRACT_INVALID] "
                    + "; ".join(target_failures)
                )
            missing = [c for c in required_advanced if c not in df.columns]
            _require(not missing, f"[ENTRY_V10_CTX_SCHEMA_MISSING] advanced {missing}")

            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            _require(not df["time"].isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
            _require(
                bool(df["time"].is_monotonic_increasing),
                "[ENTRY_V10_CTX_ADVANCED_TIME_ORDER_FAIL] advanced parquet rows must be time-monotonic; "
                "nested seq/snap tensors are loaded in parquet row order and cannot be sorted independently",
            )
            df = df.sort_values("time").reset_index(drop=True)

            # ── Memory fix (V12.2 OOM): bypass pandas entirely for nested-list
            # columns. pandas converts list<list<double>> to Python objects (~28
            # bytes per float = 30+GB for 307k samples). Even pyarrow Table held
            # whole-column = 8.8GB. Solution: chunked pyarrow read + pre-allocated
            # float32 numpy fill = 5GB peak. Drop nested cols from df before this
            # path was taken — we now re-read those 4 cols via chunked pyarrow.
            import pyarrow.parquet as pq
            log.info("[MEM_FIX] chunked pyarrow load of nested cols (bypass pandas)...")
            pf = pq.ParquetFile(self.parquet_path)
            n_rows = int(pf.metadata.num_rows)
            # Probe one batch to learn dims
            source_reconstruction = sequence_source_audit is not None
            first_batch = next(
                pf.iter_batches(
                    batch_size=64,
                    columns=(
                        ["snap", "ctx_cont", "ctx_cat"]
                        if source_reconstruction
                        else ["seq", "snap", "ctx_cont", "ctx_cat"]
                    ),
                )
            )
            if source_reconstruction:
                seq_dim = int(first_batch.column("snap")[0].values.__len__())
                seq_len = self.seq_len
            else:
                seq_dim = int(first_batch.column("seq")[0].values[0].values.__len__())
                seq_len = int(first_batch.column("seq")[0].values.__len__())
            snap_dim = int(first_batch.column("snap")[0].values.__len__())
            ctx_cont_dim = int(first_batch.column("ctx_cont")[0].values.__len__())
            ctx_cat_dim = int(first_batch.column("ctx_cat")[0].values.__len__())
            _signal_contract_for_parquet(
                self.parquet_path,
                seq_dim=seq_dim,
                snap_dim=snap_dim,
            )
            parquet_architecture = current_entry_exit_architecture_observation()
            parquet_architecture["entry"]["sequence_bars"] = seq_len
            parquet_architecture["exit"]["sequence_bars"] = (
                seq_len * ENTRY_EXIT_RESOLUTION_RATIO
            )
            parquet_architecture["shared_surface"] = {
                "signal_dim": seq_dim,
                "snap_dim": snap_dim,
                "ctx_cont_dim": ctx_cont_dim,
                "ctx_cat_dim": ctx_cat_dim,
            }
            require_entry_exit_production_architecture(
                parquet_architecture,
                context="ENTRY_V10_DATASET_PARQUET_PREALLOCATION",
            )
            log.info(
                f"[MEM_FIX] schema probe: seq=(N,{seq_len},{seq_dim})  snap=(N,{snap_dim})  "
                f"ctx_cont=(N,{ctx_cont_dim})  ctx_cat=(N,{ctx_cat_dim})"
            )
            seq_shape = (n_rows, seq_len, seq_dim)
            snap_shape = (n_rows, snap_dim)
            ctx_cont_shape = (n_rows, ctx_cont_dim)
            ctx_cat_shape = (n_rows, ctx_cat_dim)
            nested_bytes = (
                np.prod(seq_shape, dtype=np.int64) * np.dtype(np.float32).itemsize
                + np.prod(snap_shape, dtype=np.int64) * np.dtype(np.float32).itemsize
                + np.prod(ctx_cont_shape, dtype=np.int64) * np.dtype(np.float32).itemsize
                + np.prod(ctx_cat_shape, dtype=np.int64) * np.dtype(np.int64).itemsize
            )
            # Keep validation and other non-compacted split arrays off RSS as
            # well. The train smoke is compacted later, but the untouched VAL
            # surface is still ~1.2GB under the canonical recipe; retaining it
            # in anonymous RAM is unnecessary pressure inside the WSL cap.
            # A full, source-bound rolling-identity audit can prove that the
            # entire nested sequence is exactly a sliding view over the first
            # 95 history bars plus snap.  Only then may smoke avoid writing a
            # 26GB regenerable memmap mirror. Candidate/full paths provide no
            # audit and stay on the existing materialised branch.
            reconstruct_sequence_from_snapshots = sequence_roll_audit is not None
            # The source-reconstruction proof is stronger for causally
            # filtered rows: it binds every stored window to the immutable M5
            # feature surface rather than pretending emitted labels are an
            # uninterrupted bar stream.
            use_memmap = (
                nested_bytes >= _MEMMAP_MIN_BYTES
                and not reconstruct_sequence_from_snapshots
                and not source_reconstruction
            )
            first_sequence: Optional[np.ndarray] = None
            if not source_reconstruction:
                first_sequence = (
                    first_batch.column("seq")[0]
                    .values
                    .flatten()
                    .to_numpy(zero_copy_only=False)
                    .reshape(seq_len, seq_dim)
                    .astype(np.float32, copy=False)
                )
            if use_memmap:
                memmap_root = _MEMMAP_ROOT
                memmap_root.mkdir(parents=True, exist_ok=True)
                self._memmap_tmpdir = tempfile.TemporaryDirectory(
                    prefix=f"{self.parquet_path.stem}_{os.getpid()}_",
                    dir=str(memmap_root),
                )
                memmap_dir = Path(self._memmap_tmpdir.name)
                self._np_seq = np.memmap(memmap_dir / "seq.float32.mmap", dtype=np.float32, mode="w+", shape=seq_shape)
                self._np_snap = np.memmap(memmap_dir / "snap.float32.mmap", dtype=np.float32, mode="w+", shape=snap_shape)
                self._np_ctx_cont = np.memmap(
                    memmap_dir / "ctx_cont.float32.mmap", dtype=np.float32, mode="w+", shape=ctx_cont_shape
                )
                self._np_ctx_cat = np.memmap(
                    memmap_dir / "ctx_cat.int64.mmap", dtype=np.int64, mode="w+", shape=ctx_cat_shape
                )
                log.info(
                    "[MEMMAP] advanced nested arrays disk-backed: total=%.2f GB threshold=%.2f GB dir=%s",
                    nested_bytes / 1e9,
                    _MEMMAP_MIN_BYTES / 1024**3,
                    memmap_dir,
                )
            else:
                if source_reconstruction:
                    self._np_seq = None
                elif reconstruct_sequence_from_snapshots:
                    if first_sequence is None:
                        raise RuntimeError(
                            "[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_FIRST_WINDOW_MISSING]"
                        )
                    chain = np.empty(
                        (n_rows + seq_len - 1, seq_dim), dtype=np.float32
                    )
                    chain[: seq_len - 1] = first_sequence[:-1]
                    self._sequence_reconstruction_chain = chain
                    self._np_seq = np.lib.stride_tricks.sliding_window_view(
                        chain,
                        window_shape=seq_len,
                        axis=0,
                    ).swapaxes(1, 2)
                    self._sequence_roll_reconstructed = True
                else:
                    self._np_seq = np.zeros(seq_shape, dtype=np.float32)
                self._np_snap = np.zeros(snap_shape, dtype=np.float32)
                self._np_ctx_cont = np.zeros(ctx_cont_shape, dtype=np.float32)
                self._np_ctx_cat = np.zeros(ctx_cat_shape, dtype=np.int64)
            # Re-iterate (first batch was consumed) — read the whole file in chunks
            idx = 0
            for batch in pq.ParquetFile(self.parquet_path).iter_batches(
                batch_size=_NESTED_ARROW_BATCH_ROWS,
                columns=(
                    ["snap", "ctx_cont", "ctx_cat"]
                    if (reconstruct_sequence_from_snapshots or source_reconstruction)
                    else ["seq", "snap", "ctx_cont", "ctx_cat"]
                ),
            ):
                nb = batch.num_rows
                if not (reconstruct_sequence_from_snapshots or source_reconstruction):
                    self._np_seq[idx:idx+nb] = batch.column("seq").flatten().flatten().to_numpy(
                        zero_copy_only=False).reshape(nb, seq_len, seq_dim).astype(np.float32, copy=False)
                self._np_snap[idx:idx+nb] = batch.column("snap").flatten().to_numpy(
                    zero_copy_only=False).reshape(nb, snap_dim).astype(np.float32, copy=False)
                self._np_ctx_cont[idx:idx+nb] = batch.column("ctx_cont").flatten().to_numpy(
                    zero_copy_only=False).reshape(nb, ctx_cont_dim).astype(np.float32, copy=False)
                self._np_ctx_cat[idx:idx+nb] = batch.column("ctx_cat").flatten().to_numpy(
                    zero_copy_only=False).reshape(nb, ctx_cat_dim).astype(np.int64, copy=False)
                if reconstruct_sequence_from_snapshots:
                    self._sequence_reconstruction_chain[
                        seq_len - 1 + idx : seq_len - 1 + idx + nb
                    ] = self._np_snap[idx:idx+nb]
                idx += nb
                if use_memmap and idx % _MEMMAP_WRITEBACK_ROWS == 0:
                    _flush_memmap_pages(self._np_seq, self._np_snap, self._np_ctx_cont, self._np_ctx_cat)
            for arr in (self._np_seq, self._np_snap, self._np_ctx_cont, self._np_ctx_cat):
                if isinstance(arr, np.memmap):
                    arr.flush()
            if use_memmap:
                _flush_memmap_pages(self._np_seq, self._np_snap, self._np_ctx_cont, self._np_ctx_cat)
            if reconstruct_sequence_from_snapshots:
                if first_sequence is None:
                    raise RuntimeError(
                        "[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_FIRST_WINDOW_MISSING]"
                    )
                if not np.array_equal(self._np_seq[0], first_sequence):
                    raise RuntimeError(
                        "[ENTRY_SEQUENCE_ROLL_RECONSTRUCTION_FIRST_WINDOW_MISMATCH]"
                    )
                log.info(
                    "[SEQUENCE_ROLL_RECONSTRUCTION] audit=%s rows=%d base_bytes=%.2f MB "
                    "avoided_sequence_memmap_bytes=%.2f GB",
                    sequence_roll_audit_json,
                    n_rows,
                    self._sequence_reconstruction_chain.nbytes / 1e6,
                    np.prod(seq_shape, dtype=np.int64)
                    * np.dtype(np.float32).itemsize
                    / 1e9,
                )
            if source_reconstruction:
                if (
                    self._sequence_source_times_ns is None
                    or self._sequence_source_signal is None
                ):
                    raise RuntimeError(
                        "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_BACKING_MISSING]"
                    )
                # Arrow/Pandas can retain the physical Parquet resolution
                # (for example ``datetime64[ms, UTC]``).  Casting that Series
                # directly to int64 would then produce milliseconds, while the
                # immutable feature surface is explicitly nanoseconds.
                sample_times_ns = (
                    df["time"]
                    .to_numpy(dtype="datetime64[ns]")
                    .astype(np.int64, copy=False)
                )
                positions = np.searchsorted(
                    self._sequence_source_times_ns, sample_times_ns
                ).astype(np.int64, copy=False)
                if (
                    np.any(positions < self.seq_len - 1)
                    or np.any(positions >= len(self._sequence_source_times_ns))
                    or not np.array_equal(
                        self._sequence_source_times_ns[positions], sample_times_ns
                    )
                    or not np.array_equal(
                        self._np_snap,
                        self._sequence_source_signal[positions],
                    )
                ):
                    raise RuntimeError(
                        "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_RUNTIME_BINDING_INVALID]"
                    )
                self._sequence_source_positions = positions
                log.info(
                    "[SEQUENCE_SOURCE_RECONSTRUCTION] audit=%s rows=%d "
                    "source_rows=%d source_bytes=%.2f MB avoided_sequence_memmap_bytes=%.2f GB",
                    sequence_source_audit_json,
                    n_rows,
                    int(len(self._sequence_source_times_ns)),
                    self._sequence_source_signal.nbytes / 1e6,
                    np.prod(seq_shape, dtype=np.int64)
                    * np.dtype(np.float32).itemsize
                    / 1e9,
                )
            if reconstruct_sequence_from_snapshots:
                log.info(
                    "[MEM_FIX] rolling seq view built: logical_shape=%s "
                    "logical_bytes=%.2f GB physical_chain_bytes=%.2f MB",
                    self._np_seq.shape,
                    self._np_seq.nbytes / 1e9,
                    self._sequence_reconstruction_chain.nbytes / 1e6,
                )
            elif not source_reconstruction:
                log.info(
                    "[MEM_FIX] arrays built: seq=%s (%.2f GB)",
                    self._np_seq.shape,
                    self._np_seq.nbytes / 1e9,
                )
            # Drop nested cols from df (they were in df from earlier pd.read_parquet)
            # to free the pandas object memory.
            for c in ("seq", "snap", "ctx_cont", "ctx_cat"):
                if c in df.columns:
                    df = df.drop(columns=[c])
            import gc

            gc.collect()

            self.df = df
            self._advanced = True
            self.signal_cols = None
            self.ctx_cont_cols = None
            self.ctx_cat_cols = None
            # Infer ctx dims from pre-converted arrays
            self.ctx_cont_dim = int(self._np_ctx_cont.shape[1])
            self.ctx_cat_dim = int(self._np_ctx_cat.shape[1])
            self._ctx_vnext_extra = None
            if (
                self.ctx_cont_dim != MODEL_NATIVE_CTX_CONT_DIM
                or self.ctx_cat_dim != MODEL_NATIVE_CTX_CAT_DIM
            ):
                raise RuntimeError(
                    "[ENTRY_MODEL_NATIVE_CTX_DIM_INVALID] "
                    f"ctx_cont_dim={self.ctx_cont_dim} expected={MODEL_NATIVE_CTX_CONT_DIM} "
                    f"ctx_cat_dim={self.ctx_cat_dim} expected={MODEL_NATIVE_CTX_CAT_DIM}"
                )

            self.indices = np.arange(len(df))
            _require(len(self.indices) > 0, "[ENTRY_V10_CTX_NO_SAMPLES]")

            log.info(
                f"[DATASET_SCHEMA] advanced | rows={len(df)} samples={len(self.indices)} "
                f"time=[{df['time'].min()} .. {df['time'].max()}]"
            )

        # ── Multi-TF prebuild (V12.2) ──
        # Build M15/H1/H4/D1 per-bar feature tables once. At __getitem__ time we
        # just slice the last N bars at-or-before the sample's timestamp.
        # IMPORTANT: cache at module level so train_ds + val_ds share the
        # ~3GB resampled feature tables. Without this, peak memory hits OOM
        # on 15GB hosts (1.5GB parquet × 2 + multi-TF × 2 + train arrays).
        # Mandatory V4: all eight families on M5/M15/H1/H4/D1. Older generic
        # V2/V3 matrices are historical cache formats, not trainable inputs.
        from gx1.features.htf_features import (
            HTF_V4_MATRIX_CONTRACT,
            MULTI_TF_FEATURE_COUNT_V4,
            MULTI_TF_PER_BAR_FEATURES_V4,
        )
        if m5_prebuilt_path is None:
            raise RuntimeError(
                "[MULTI_TF_INIT_FAIL] exact architecture requires m5_prebuilt_path "
                "(path to canonical_v3 M5 OHLC parquet)."
            )
        m5_path = Path(m5_prebuilt_path)
        # One loader owns the verified V4 disk-cache path. Its in-process
        # identity includes the exact M5 SHA and full cache identity.
        self._multi_tf_feats = _prebuild_multi_tf_features_once(m5_path)
        self._multi_tf_cache_identity_sha256 = str(
            getattr(self._multi_tf_feats, "cache_identity_sha256", "")
        )
        self._multi_tf_cache_manifest_sha256 = str(
            getattr(self._multi_tf_feats, "manifest_sha256", "")
        )
        self._multi_tf_cache_dir = str(
            Path(os.environ[_TRAIN_MULTI_TF_CACHE_ENV]).expanduser().resolve()
        )
        self._multi_tf_cache_manifest_path = str(
            Path(self._multi_tf_cache_dir) / "manifest.json"
        )
        self._multi_tf_cache_m5_source = str(
            getattr(self._multi_tf_feats, "m5_prebuilt_source", "")
        )
        self._multi_tf_cache_m5_source_sha256 = str(
            getattr(self._multi_tf_feats, "m5_prebuilt_source_sha256", "")
        )
        for _name, _value in (
            ("cache_identity", self._multi_tf_cache_identity_sha256),
            ("manifest", self._multi_tf_cache_manifest_sha256),
            ("m5_source", self._multi_tf_cache_m5_source_sha256),
        ):
            if len(_value) != 64 or any(
                char not in "0123456789abcdef" for char in _value
            ):
                raise RuntimeError(
                    f"[MULTI_TF_V4_CACHE_{_name.upper()}_SHA256_INVALID]"
                )
        # The loaded tables declare their own exact V4 contract; no historical
        # width is inferred or accepted here.
        _known_contracts = {
            HTF_V4_MATRIX_CONTRACT: (
                MULTI_TF_FEATURE_COUNT_V4,
                MULTI_TF_PER_BAR_FEATURES_V4,
            ),
        }
        _declared = {
            str(feats.attrs.get("htf_feature_contract"))
            for feats in self._multi_tf_feats.values()
        }
        if len(_declared) != 1:
            raise RuntimeError(
                f"[MULTI_TF_CONTRACT_SPLIT_BRAIN] timeframes declare {sorted(_declared)}"
            )
        _contract = _declared.pop()
        if _contract not in _known_contracts:
            raise RuntimeError(
                f"[MULTI_TF_CONTRACT_UNKNOWN] {_contract!r} is not a declared "
                f"per-bar contract: {sorted(_known_contracts)}"
            )
        if _contract != HTF_V4_MATRIX_CONTRACT:
            raise RuntimeError(
                "[MULTI_TF_EIGHT_FAMILY_CONTRACT_REQUIRED] "
                f"observed={_contract!r} required={HTF_V4_MATRIX_CONTRACT!r}"
            )
        _expected_count, _expected_names = _known_contracts[_contract]
        for _tf_name, _feats in self._multi_tf_feats.items():
            if int(_feats.shape[1]) != int(_expected_count):
                raise RuntimeError(
                    f"[MULTI_TF_CONTRACT_WIDTH_MISMATCH] {_tf_name} "
                    f"width={int(_feats.shape[1])} contract={_contract} "
                    f"expected={int(_expected_count)}"
                )
            if tuple(_feats.columns) != tuple(_expected_names):
                raise RuntimeError(
                    f"[MULTI_TF_CONTRACT_ORDER_MISMATCH] {_tf_name} under {_contract}"
                )
        self._multi_tf_contract = _contract
        self._multi_tf_feature_names = tuple(_expected_names)
        self._multi_tf_feature_count = int(_expected_count)
        self._multi_tf_v4 = True
        log.info(
            "[MULTI_TF_CONTRACT] %s width=%d across %d timeframes",
            _contract,
            int(_expected_count),
            len(self._multi_tf_feats),
        )
        for tf_name, feats in self._multi_tf_feats.items():
            log.info(
                f"[MULTI_TF] {tf_name}: {len(feats):,} bars × {feats.shape[1]} feats  "
                f"range {feats.index[0]} → {feats.index[-1]}"
            )

    def _get_multi_tf_window(
        self,
        target_ts: pd.Timestamp,
        *,
        route_timeframes: tuple[str, ...],
        base_bar_seconds: int,
        key_prefix: str,
    ) -> Dict[str, np.ndarray]:
        """Slice one exact route from the shared V4 cache, or fail closed."""

        from gx1.features.htf_features import (
            get_model_native_multi_tf_route_windows,
        )

        windows = get_model_native_multi_tf_route_windows(
            self._multi_tf_feats,
            decision_bar_start=pd.Timestamp(target_ts),
            per_tf_seq_lens=self.per_tf_seq_lens,
            route_timeframes=route_timeframes,
            base_bar_duration=pd.Timedelta(seconds=base_bar_seconds),
        )
        return {
            f"{key_prefix}{tf.lower()}": value
            for tf, value in windows.items()
        }

    def _get_exit_multi_tf_episode_histories(
        self,
        decision_time_ns: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return each unique native MTF history and exact as-of gathers once."""

        decision_ns = np.asarray(decision_time_ns, dtype=np.int64)
        if (
            decision_ns.ndim != 1
            or decision_ns.shape != (UNIFIED_EXIT_MAX_PATH_BARS,)
            or np.any(np.diff(decision_ns) <= 0)
        ):
            raise RuntimeError("UNIFIED_EXIT_EPISODE_MTF_CLOCK_INVALID")
        out: dict[str, np.ndarray] = {}
        availability_ns = decision_ns + int(
            pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value
        )
        for tf in EXIT_MTF_CONTEXT_TIMEFRAMES:
            feats = self._multi_tf_feats[tf]
            ts_int64 = np.asarray(feats.attrs.get("ts_int64"))
            feats_np = np.asarray(feats.attrs.get("feats_np"))
            warmup_rows = feats.attrs.get("causal_warmup_rows")
            if (
                ts_int64.dtype != np.dtype(np.int64)
                or ts_int64.shape != (len(feats),)
                or feats_np.dtype != np.dtype(np.float32)
                or feats_np.shape
                != (len(feats), self._multi_tf_feature_count)
                or isinstance(warmup_rows, bool)
                or not isinstance(warmup_rows, (int, np.integer))
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_EPISODE_MTF_SOURCE_INVALID:{tf}"
                )
            cutoff_ns = availability_ns - int(MULTI_TF_SHIFT[tf].value)
            right = np.searchsorted(ts_int64, cutoff_ns, side="right").astype(
                np.int64
            )
            tail = right - 1
            required = int(self.per_tf_seq_lens[tf])
            left = int(tail[0]) - required + 1
            last = int(tail[-1])
            if (
                np.any(np.diff(tail) < 0)
                or left < int(warmup_rows)
                or left < 0
                or last >= len(feats)
            ):
                raise RuntimeError(
                    f"UNIFIED_EXIT_EPISODE_MTF_HISTORY_INVALID:{tf}"
                )
            history = np.ascontiguousarray(
                feats_np[left : last + 1], dtype=np.float32
            )
            gather = np.ascontiguousarray(tail - left, dtype=np.int64)
            out[f"exit_mtf_history_{tf.lower()}"] = history
            out[f"exit_mtf_history_time_ns_{tf.lower()}"] = (
                np.ascontiguousarray(ts_int64[left : last + 1], dtype=np.int64)
            )
            out[f"exit_mtf_gather_{tf.lower()}"] = gather
        return out

    def materialize_full_exit_episode(
        self,
        entry_row_index: int,
    ) -> dict[str, Any] | None:
        """Materialize one hash-bound one-pass Exit episode pack."""

        if self._unified_exit_lifecycle is None:
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_NOT_BOUND")
        core = self._unified_exit_lifecycle.materialize_causal_episode_core(
            int(entry_row_index)
        )
        if core is None:
            return None
        pack = seal_unified_exit_episode_pack(
            {
                "schema_version": UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION,
                **core,
                "multi_tf_cache_identity_sha256": (
                    self._multi_tf_cache_identity_sha256
                ),
                **self._get_exit_multi_tf_episode_histories(
                    core["exit_decision_time_ns"]
                ),
            }
        )
        return require_unified_exit_episode_pack(
            pack,
            per_tf_seq_lens=self.per_tf_seq_lens,
            expected_mtf_cache_identity_sha256=(
                self._multi_tf_cache_identity_sha256
            ),
            context="ENTRY_DATASET_EXIT_EPISODE",
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _storage_position_for_full_row(self, row_index: int) -> int:
        """Map one immutable split row to its current snap/context backing."""

        row = int(row_index)
        if row < 0 or row >= len(self.df):
            raise RuntimeError("[ENTRY_V10_CTX_STORAGE_ROW_OOB]")
        if self._compact_row_indices is None:
            return row
        storage_position = int(np.searchsorted(self._compact_row_indices, row))
        if (
            storage_position >= len(self._compact_row_indices)
            or int(self._compact_row_indices[storage_position]) != row
        ):
            raise RuntimeError("[ENTRY_V10_CTX_COMPACT_ROW_LOOKUP_MISMATCH]")
        return storage_position

    def sequence_for_full_row(self, row_index: int) -> np.ndarray:
        """Return one exact Entry history without materialising all windows."""

        row = int(row_index)
        if self._sequence_source_reconstructed:
            if (
                self._sequence_source_signal is None
                or self._sequence_source_positions is None
            ):
                raise RuntimeError(
                    "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_BACKING_MISSING]"
                )
            position = int(self._sequence_source_positions[row])
            sequence = self._sequence_source_signal[
                position - self.seq_len + 1 : position + 1
            ]
        else:
            storage_position = self._storage_position_for_full_row(row)
            sequence = self._np_seq[storage_position]
        if sequence.shape != (self.seq_len, self.seq_input_dim):
            raise RuntimeError("[ENTRY_V10_CTX_SEQUENCE_SOURCE_SHAPE_INVALID]")
        return sequence

    def source_reconstruction_normalization_input(self) -> Optional[dict[str, np.ndarray]]:
        """Expose the exact source representation to TRAIN-only normalization."""

        if not self._sequence_source_reconstructed:
            return None
        if (
            self._sequence_source_signal is None
            or self._sequence_source_times_ns is None
            or self._sequence_source_positions is None
            or self._compact_row_indices is not None
        ):
            raise RuntimeError(
                "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_NORMALIZATION_BACKING_INVALID]"
            )
        return {
            "signal": self._sequence_source_signal,
            "times_ns": self._sequence_source_times_ns,
            "sample_positions": self._sequence_source_positions,
        }

    def compact_materialized_rows(self, row_indices: Sequence[int]) -> None:
        """Release the full nested-array backing after a bounded smoke sample.

        The parquet dataframe and original row ids stay intact so lifecycle
        binding, target lookup and evidence hashes continue to use the exact
        canonical split coordinates.  Only the already selected nested input
        rows are retained for the smoke DataLoader.  Candidate/full training
        never calls this method.
        """

        if not bool(getattr(self, "_advanced", False)):
            raise RuntimeError(
                "[ENTRY_V10_CTX_COMPACT_REQUIRES_ADVANCED_DATASET]"
            )
        observed = np.asarray(row_indices, dtype=np.int64)
        if observed.ndim != 1 or observed.size < 1:
            raise RuntimeError(
                "[ENTRY_V10_CTX_COMPACT_ROW_INDICES_INVALID] expected non-empty 1-D"
            )
        if np.any(observed < 0) or np.any(observed >= len(self.df)):
            raise RuntimeError(
                "[ENTRY_V10_CTX_COMPACT_ROW_INDICES_OOB]"
            )
        if np.any(np.diff(observed) <= 0):
            raise RuntimeError(
                "[ENTRY_V10_CTX_COMPACT_ROW_INDICES_NOT_SORTED_UNIQUE]"
            )
        if not np.array_equal(observed, np.asarray(self.indices, dtype=np.int64)):
            raise RuntimeError(
                "[ENTRY_V10_CTX_COMPACT_ROW_INDICES_MUST_MATCH_DATASET_SELECTION]"
            )
        if self._compact_row_indices is not None:
            raise RuntimeError(
                "[ENTRY_V10_CTX_COMPACT_ALREADY_APPLIED]"
            )

        old_arrays = (
            self._np_seq,
            self._np_snap,
            self._np_ctx_cont,
            self._np_ctx_cat,
        )
        if self._sequence_source_reconstructed:
            # Keep the compact smoke's sequence view source-backed. Copying
            # 10k windows here would recreate a near-gigabyte duplicate just
            # after the full byte-identical audit allowed us to avoid it.
            compact_arrays = (
                None,
                np.ascontiguousarray(self._np_snap[observed], dtype=np.float32),
                np.ascontiguousarray(self._np_ctx_cont[observed], dtype=np.float32),
                np.ascontiguousarray(self._np_ctx_cat[observed], dtype=np.int64),
            )
        else:
            compact_arrays = (
                np.ascontiguousarray(self._np_seq[observed], dtype=np.float32),
                np.ascontiguousarray(self._np_snap[observed], dtype=np.float32),
                np.ascontiguousarray(self._np_ctx_cont[observed], dtype=np.float32),
                np.ascontiguousarray(self._np_ctx_cat[observed], dtype=np.int64),
            )
        for array in old_arrays:
            if isinstance(array, np.memmap):
                array.flush()
        memmap_tmpdir = self._memmap_tmpdir
        self._np_seq, self._np_snap, self._np_ctx_cont, self._np_ctx_cat = (
            compact_arrays
        )
        self._compact_row_indices = observed.copy()
        self._memmap_tmpdir = None
        # A reconstructed sequence view retains its compact base chain through
        # ndarray.base.  Once the smoke's selected rows are copied, release
        # that full-TRAIN chain exactly as we release the ordinary memmap
        # backing.  The boolean is retained for non-authoritative diagnostics;
        # it does not grant any lifecycle authority.
        self._sequence_reconstruction_chain = None
        for array in old_arrays:
            if isinstance(array, np.memmap):
                mmap_handle = getattr(array, "_mmap", None)
                if mmap_handle is not None:
                    mmap_handle.close()
        if memmap_tmpdir is not None:
            memmap_tmpdir.cleanup()
        import gc

        gc.collect()
        log.info(
            "[MEM_COMPACT] smoke_rows=%d original_rows=%d retained_nested_bytes=%.2f MB",
            int(observed.size),
            int(len(self.df)),
            sum(int(array.nbytes) for array in compact_arrays if array is not None) / 1e6,
        )

    def bind_unified_exit_lifecycle(
        self,
        lifecycle: UnifiedExitLifecycleSplit,
    ) -> None:
        """Bind the exact split-local Exit episodes to immutable Entry rows."""

        if not isinstance(lifecycle, UnifiedExitLifecycleSplit):
            raise RuntimeError(
                "UNIFIED_EXIT_LIFECYCLE_SPLIT_OBJECT_REQUIRED"
            )
        if int(lifecycle.entry_row_count) != len(self.df):
            raise RuntimeError(
                "UNIFIED_EXIT_LIFECYCLE_ENTRY_ROW_COUNT_MISMATCH: "
                f"split={lifecycle.split} lifecycle={lifecycle.entry_row_count} "
                f"dataset={len(self.df)}"
            )
        if self._unified_exit_lifecycle is not None:
            raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_ALREADY_BOUND")
        self._unified_exit_lifecycle = lifecycle

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if self._advanced:
            t = int(self.indices[i])
            row = self.df.iloc[t]
            # V12.2: nested cols were pre-converted to np arrays in __init__;
            # __getitem__ now just slices for speed + memory efficiency.
            storage_position = self._storage_position_for_full_row(t)
            seq = self.sequence_for_full_row(t)
            snap = self._np_snap[storage_position]
            ctx_cont = self._np_ctx_cont[storage_position]
            ctx_cat = self._np_ctx_cat[storage_position]
            if seq.shape != (self.seq_len, self.seq_input_dim):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] seq shape {seq.shape} expected ({self.seq_len}, {self.seq_input_dim})"
                )
            if snap.shape != (self.snap_input_dim,):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] snap shape {snap.shape} expected ({self.snap_input_dim},)"
                )
            if ctx_cont.shape != (self.ctx_cont_dim,):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] ctx_cont shape {ctx_cont.shape} expected ({self.ctx_cont_dim},)"
                )
            if ctx_cat.shape != (self.ctx_cat_dim,):
                raise RuntimeError(
                    f"[ENTRY_V10_CTX_SHAPE_MISMATCH] ctx_cat shape {ctx_cat.shape} expected ({self.ctx_cat_dim},)"
                )

            out_batch = {
                "entry_row_index": torch.tensor(t, dtype=torch.long),
                "seq_x": torch.tensor(seq),
                "snap_x": torch.tensor(snap),
                "ctx_cont": torch.tensor(ctx_cont),
                "ctx_cat": torch.tensor(ctx_cat),
            }
            # Every active target was validated in __init__ and is read
            # directly. There are no aliases, defaults, or compatibility rows.
            for target_name in _MODEL_NATIVE_ACTIVE_TARGET_COLS:
                target_dtype = torch.long if target_name == "y_side" else torch.float32
                target_value = (
                    int(row[target_name])
                    if target_dtype == torch.long
                    else float(row[target_name])
                )
                out_batch[target_name] = torch.tensor(target_value, dtype=target_dtype)
            mtf = self._get_multi_tf_window(
                pd.Timestamp(row["time"]),
                route_timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
                base_bar_seconds=ENTRY_DECISION_BAR_SECONDS,
                key_prefix="seq_",
            )
            for k, v in mtf.items():
                # Cache-backed MTF windows can be read-only Arrow/NumPy views.
                # The model does not mutate inputs, but PyTorch correctly
                # rejects a non-writable backing as undefined if any later
                # operation did. Copy only this per-sample MTF window into a
                # writable float32 buffer; sequence/source storage stays
                # shared and untouched.
                out_batch[k] = torch.from_numpy(
                    np.array(v, dtype=np.float32, copy=True, order="C")
                )
            return out_batch

# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------
def _new_cooperation_gate_epoch_accumulator(
    gate_widths: Mapping[str, int] = _MODEL_NATIVE_COOPERATION_GATE_WIDTHS,
) -> dict[str, dict[str, Any]]:
    return {
        output_name: {
            "rows": 0,
            "sum": np.zeros(expected_width, dtype=np.float64),
            "entropy_sum": 0.0,
        }
        for output_name, expected_width in gate_widths.items()
    }


def _new_feature_tf_gate_epoch_accumulator(
    gate_shape: tuple[int, int] = _MODEL_NATIVE_FEATURE_TF_GATE_SHAPE,
) -> dict[str, Any]:
    shape = tuple(int(value) for value in gate_shape)
    return {
        "rows": 0,
        "sum": np.zeros(shape, dtype=np.float64),
        "sum_sq": np.zeros(shape, dtype=np.float64),
        "min": np.full(shape, np.inf, dtype=np.float64),
        "max": np.full(shape, -np.inf, dtype=np.float64),
    }


def _accumulate_feature_tf_gate_epoch(
    accumulator: dict[str, Any],
    out: dict[str, Any],
    *,
    gate_shape: tuple[int, int] = _MODEL_NATIVE_FEATURE_TF_GATE_SHAPE,
) -> None:
    gate = out.get("family_tf_feature_gate")
    if (
        not isinstance(gate, torch.Tensor)
        or gate.ndim != 3
        or tuple(gate.shape[1:]) != tuple(gate_shape)
        or gate.numel() == 0
    ):
        raise RuntimeError(
            "[ENTRY_MODEL_NATIVE_FEATURE_TF_GATE_SHAPE_INVALID] "
            f"shape={getattr(gate, 'shape', None)}"
        )
    values = gate.detach().double()
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("[ENTRY_MODEL_NATIVE_FEATURE_TF_GATE_NONFINITE]")
    array = values.cpu().numpy()
    accumulator["rows"] = int(accumulator["rows"]) + int(array.shape[0])
    accumulator["sum"] += array.sum(axis=0)
    accumulator["sum_sq"] += np.square(array).sum(axis=0)
    accumulator["min"] = np.minimum(accumulator["min"], array.min(axis=0))
    accumulator["max"] = np.maximum(accumulator["max"], array.max(axis=0))


def _finalize_feature_tf_gate_epoch(
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    rows = int(accumulator["rows"])
    if rows <= 0:
        raise RuntimeError(
            "[ENTRY_MODEL_NATIVE_FEATURE_TF_GATE_EPOCH_EVIDENCE_MISSING]"
        )
    mean = np.asarray(accumulator["sum"], dtype=np.float64) / float(rows)
    variance = (
        np.asarray(accumulator["sum_sq"], dtype=np.float64) / float(rows)
        - np.square(mean)
    )
    std = np.sqrt(np.maximum(variance, 0.0))
    minimum = np.asarray(accumulator["min"], dtype=np.float64)
    maximum = np.asarray(accumulator["max"], dtype=np.float64)
    if not all(
        bool(np.isfinite(value).all())
        for value in (mean, std, minimum, maximum)
    ):
        raise RuntimeError(
            "[ENTRY_MODEL_NATIVE_FEATURE_TF_GATE_EPOCH_EVIDENCE_INVALID]"
        )
    return {
        "family_tf_feature_gate_rows": rows,
        "family_tf_feature_gate_mean_weight": mean.reshape(-1).tolist(),
        "family_tf_feature_gate_std_weight": std.reshape(-1).tolist(),
        "family_tf_feature_gate_min_observed": minimum.reshape(-1).tolist(),
        "family_tf_feature_gate_max_observed": maximum.reshape(-1).tolist(),
        "family_tf_feature_gate_min_std": float(std.min()),
    }


def _accumulate_cooperation_gate_epoch(
    accumulator: dict[str, dict[str, Any]],
    out: dict[str, Any],
    *,
    gate_widths: Mapping[str, int] = _MODEL_NATIVE_COOPERATION_GATE_WIDTHS,
) -> None:
    """Accumulate exact epoch-wide cooperation use, not batch-minimum proxies."""
    for output_name, expected_width in gate_widths.items():
        gate = out.get(output_name)
        if not isinstance(gate, torch.Tensor):
            raise RuntimeError(
                f"[ENTRY_MODEL_NATIVE_ACTIVE_HEAD_MISSING] output={output_name}"
            )
        if gate.ndim != 2 or int(gate.shape[1]) != expected_width or gate.numel() == 0:
            raise RuntimeError(
                "[ENTRY_MODEL_NATIVE_GATE_SHAPE_INVALID] "
                f"output={output_name} shape={tuple(gate.shape)} "
                f"expected=(batch,{expected_width})"
            )
        detached = gate.detach().double()
        if not bool(torch.isfinite(detached).all().item()):
            raise RuntimeError(
                f"[ENTRY_MODEL_NATIVE_GATE_NONFINITE] output={output_name}"
            )
        clipped = detached.clamp(min=1e-12)
        state = accumulator[output_name]
        state["rows"] = int(state["rows"]) + int(detached.shape[0])
        state["sum"] += detached.sum(dim=0).cpu().numpy()
        state["entropy_sum"] = float(state["entropy_sum"]) + float(
            (-(clipped * clipped.log()).sum(dim=1).sum()).cpu().item()
        )


def _finalize_cooperation_gate_epoch(
    accumulator: dict[str, dict[str, Any]],
    *,
    gate_widths: Mapping[str, int] = _MODEL_NATIVE_COOPERATION_GATE_WIDTHS,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for output_name, expected_width in gate_widths.items():
        state = accumulator[output_name]
        rows = int(state["rows"])
        if rows <= 0:
            raise RuntimeError(
                f"[ENTRY_MODEL_NATIVE_GATE_EPOCH_EVIDENCE_MISSING] output={output_name}"
            )
        mean_weight = np.asarray(state["sum"], dtype=np.float64) / float(rows)
        if (
            mean_weight.shape != (expected_width,)
            or not np.isfinite(mean_weight).all()
            or not np.isclose(float(mean_weight.sum()), 1.0, rtol=1e-6, atol=1e-7)
        ):
            raise RuntimeError(
                "[ENTRY_MODEL_NATIVE_GATE_EPOCH_EVIDENCE_INVALID] "
                f"output={output_name} mean_shape={mean_weight.shape} "
                f"mean_sum={float(mean_weight.sum()):.12g}"
            )
        stats[f"{output_name}_rows"] = rows
        stats[f"{output_name}_mean_weight"] = mean_weight.tolist()
        stats[f"{output_name}_min_mean"] = float(mean_weight.min())
        stats[f"{output_name}_entropy_mean"] = float(state["entropy_sum"]) / float(rows)
    return stats


def _cooperation_gate_health_failures(
    stats: dict[str, Any],
    *,
    gate_widths: Mapping[str, int] = _MODEL_NATIVE_COOPERATION_GATE_WIDTHS,
    feature_gate_shape: tuple[int, int] = _MODEL_NATIVE_FEATURE_TF_GATE_SHAPE,
) -> list[str]:
    """Validate empirical liveness without a hand-written distribution target."""
    failures: list[str] = []
    for output_name, expected_width in gate_widths.items():
        mean_weight = np.asarray(
            stats.get(f"{output_name}_mean_weight", ()),
            dtype=np.float64,
        )
        entropy = stats.get(f"{output_name}_entropy_mean")
        if (
            mean_weight.shape != (expected_width,)
            or not np.isfinite(mean_weight).all()
            or not np.isclose(float(mean_weight.sum()), 1.0, rtol=1e-6, atol=1e-7)
        ):
            failures.append(
                f"{output_name} epoch-wide mean-weight evidence is missing or invalid"
            )
            continue
        observed_min = float(mean_weight.min())
        if observed_min <= 0.0:
            failures.append(
                f"{output_name} min mean={observed_min:.6f} "
                "(every observed routing path must be live)"
            )
        if entropy is None or not np.isfinite(float(entropy)):
            failures.append(f"{output_name} epoch-wide entropy evidence is missing")
        elif float(entropy) <= 0.0:
            failures.append(
                f"{output_name} entropy={float(entropy):.6f} "
                "(routing must not be exactly deterministic for the whole epoch)"
            )
    feature_count = int(np.prod(feature_gate_shape))
    feature_std = np.asarray(
        stats.get("family_tf_feature_gate_std_weight", ()),
        dtype=np.float64,
    )
    feature_min = np.asarray(
        stats.get("family_tf_feature_gate_min_observed", ()),
        dtype=np.float64,
    )
    feature_max = np.asarray(
        stats.get("family_tf_feature_gate_max_observed", ()),
        dtype=np.float64,
    )
    if any(
        value.shape != (feature_count,) or not np.isfinite(value).all()
        for value in (feature_std, feature_min, feature_max)
    ):
        failures.append(
            "family_tf_feature_gate epoch-wide evidence is missing or invalid"
        )
    else:
        dead = np.flatnonzero(
            feature_std <= _MODEL_NATIVE_FEATURE_TF_GATE_MIN_STD
        )
        saturated = np.flatnonzero(
            (feature_min <= 0.0) | (feature_max >= 2.0)
        )
        if dead.size:
            failures.append(
                "family_tf_feature_gate constant/dead indices="
                f"{dead.tolist()}"
            )
        if saturated.size:
            failures.append(
                "family_tf_feature_gate saturated indices="
                f"{saturated.tolist()}"
            )
    return failures


def _unified_exit_gate_view(out: Mapping[str, Any]) -> dict[str, Any]:
    """Map Exit-prefixed model outputs onto the shared gate evidence contract."""

    observed = {
        canonical_name: out.get(exit_name)
        for canonical_name, exit_name in _UNIFIED_EXIT_GATE_OUTPUT_NAMES.items()
    }
    specialist = observed.get("specialist_gate")
    tf_gate = observed.get("tf_gate")
    cooperation = observed.get("family_tf_cooperation_gate")
    feature = observed.get("family_tf_feature_gate")
    if isinstance(specialist, torch.Tensor) and specialist.ndim == 3:
        observed["specialist_gate"] = specialist.reshape(-1, specialist.shape[-1])
    if isinstance(tf_gate, torch.Tensor) and tf_gate.ndim == 3:
        observed["tf_gate"] = tf_gate.reshape(-1, tf_gate.shape[-1])
    if isinstance(cooperation, torch.Tensor) and cooperation.ndim == 4:
        observed["family_tf_cooperation_gate"] = cooperation.reshape(
            cooperation.shape[0] * cooperation.shape[1], -1
        )
    if isinstance(feature, torch.Tensor) and feature.ndim == 4:
        observed["family_tf_feature_gate"] = feature.reshape(
            feature.shape[0] * feature.shape[1],
            feature.shape[2],
            feature.shape[3],
        )
    return observed


def _finalize_unified_exit_gate_epoch(
    cooperation_accumulator: dict[str, dict[str, Any]],
    feature_accumulator: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return exact five-TF Exit gate evidence and checkpoint failures."""

    raw_stats = _finalize_cooperation_gate_epoch(
        cooperation_accumulator,
        gate_widths=_UNIFIED_EXIT_COOPERATION_GATE_WIDTHS,
    )
    raw_stats.update(_finalize_feature_tf_gate_epoch(feature_accumulator))
    failures = _cooperation_gate_health_failures(
        raw_stats,
        gate_widths=_UNIFIED_EXIT_COOPERATION_GATE_WIDTHS,
        feature_gate_shape=_UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE,
    )
    prefixed = {f"exit_{key}": value for key, value in raw_stats.items()}
    prefixed["exit_gate_evidence_schema_version"] = (
        UNIFIED_EXIT_GATE_EVIDENCE_SCHEMA_VERSION
    )
    prefixed["exit_cooperation_gate_health_ok"] = not failures
    prefixed["exit_cooperation_gate_health_failures"] = list(failures)
    return prefixed, failures


def _side_mae_auxiliary_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Regress the two raw side-specific adverse excursions in bps."""

    side_mae = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="side_mae_bps",
        target_names=("y_long_expected_mae_bps", "y_short_expected_mae_bps"),
    )
    non_blocking = device.type == "cuda"
    y_long_mae = _require_nonnegative_target(
        batch["y_long_expected_mae_bps"].to(device, non_blocking=non_blocking).float(),
        name="y_long_expected_mae_bps",
    )
    y_short_mae = _require_nonnegative_target(
        batch["y_short_expected_mae_bps"].to(device, non_blocking=non_blocking).float(),
        name="y_short_expected_mae_bps",
    )

    mae_target = torch.stack([y_long_mae, y_short_mae], dim=1).to(
        dtype=side_mae.dtype
    )
    loss = nn.functional.l1_loss(side_mae, mae_target)
    return loss, {
        "side_mae_loss": float(loss.detach().cpu().item()),
        "side_mae_target_mean_bps": float(
            mae_target.detach().mean().cpu().item()
        ),
    }



def _trendline_event_aux_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Train genuine forward registry events without a direction mapping."""
    logits = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="trendline_event_logits",
        target_names=(
            "y_line_support_touch_held",
            "y_line_resistance_touch_held",
            "y_countertrend_short_trap",
            "y_countertrend_long_trap",
        ),
    )
    stats = {
        "trendline_event_loss": 0.0,
        "trendline_event_rows": 0.0,
        "trendline_support_rows": 0.0,
        "trendline_resistance_rows": 0.0,
    }
    non_blocking = device.type == "cuda"
    rising = batch["y_line_support_touch_held"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    rising_mask = batch["y_line_support_touch_mask"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    falling = batch["y_line_resistance_touch_held"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    falling_mask = batch["y_line_resistance_touch_mask"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    short_trap = batch["y_countertrend_short_trap"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    long_trap = batch["y_countertrend_long_trap"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    targets = torch.stack(
        [rising, falling, short_trap, long_trap],
        dim=1,
    ).to(dtype=logits.dtype)
    if logits.ndim != 2 or logits.shape[1] != targets.shape[1]:
        raise RuntimeError(
            "[ENTRY_TRENDLINE_EVENT_OUTPUT_DIM_MISMATCH] "
            f"logits_shape={tuple(logits.shape)} targets_shape={tuple(targets.shape)}"
        )
    # V29 stage 2 masked objective: the two line-hold dims are supervised
    # ONLY on registry touch-event rows (their forward outcome is defined
    # there and nowhere else — the y_side_mask masking pattern); the four
    # trap/early-failure dims stay dense.
    element_mask = torch.ones_like(targets)
    element_mask[:, 0] = rising_mask
    element_mask[:, 1] = falling_mask
    mask_total = element_mask.sum()
    if float(mask_total.detach().cpu().item()) <= 0.0:
        raise RuntimeError("[ENTRY_TRENDLINE_EVENT_LOSS_MASK_EMPTY]")
    per_element = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    loss = (per_element * element_mask).sum() / mask_total
    stats["trendline_event_loss"] = float(loss.detach().cpu().item())
    stats["trendline_event_rows"] = float(int((targets.max(dim=1).values > 0.5).sum().detach().cpu().item()))
    stats["trendline_support_rows"] = float(int((rising_mask > 0.5).sum().detach().cpu().item()))
    stats["trendline_resistance_rows"] = float(int((falling_mask > 0.5).sum().detach().cpu().item()))
    return loss, stats


class _WeightEma:
    """Exponential moving average of the model weights (V30 package 5).

    ``shadow <- decay*shadow + (1 - decay)*current`` after every optimizer
    step, over the model's complete ``state_dict`` (floating tensors are
    averaged; integer/bool buffers are carried by exact copy because an average
    of a counter is not a counter).  The raw weights keep training untouched;
    only VALIDATION and checkpoint selection read the averaged weights, through
    ``evaluating``.

    This object is constructed ONLY when the recipe decay is > 0.0.  At the 0.0
    OFF sentinel no instance exists, nothing is allocated and no state_dict is
    ever swapped, so the training path is byte-identical to the pre-package-5
    trainer.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        decay = float(decay)
        if (not math.isfinite(decay)) or decay <= 0.0 or decay >= 1.0:
            raise RuntimeError(
                f"[ENTRY_TRAIN_WEIGHT_EMA_DECAY_INVALID] got={decay!r} "
                "expected finite in (0.0, 1.0) for an enabled EMA"
            )
        self.decay = decay
        self._shadow: Dict[str, torch.Tensor] = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self._steps = 0

    @property
    def steps(self) -> int:
        return int(self._steps)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        if set(current) != set(self._shadow):
            raise RuntimeError("[ENTRY_TRAIN_WEIGHT_EMA_STATE_KEYS_CHANGED]")
        for name, tensor in current.items():
            shadow = self._shadow[name]
            if tensor.is_floating_point():
                shadow.mul_(self.decay).add_(
                    tensor.detach(), alpha=1.0 - self.decay
                )
            else:
                shadow.copy_(tensor.detach())
        self._steps += 1

    @contextlib.contextmanager
    def evaluating(self, model: nn.Module):
        """Temporarily install the averaged weights on ``model``."""

        if self._steps <= 0:
            raise RuntimeError(
                "[ENTRY_TRAIN_WEIGHT_EMA_NOT_UPDATED] refusing to evaluate an "
                "EMA that has taken no optimizer step"
            )
        saved = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        model.load_state_dict(self._shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(saved, strict=True)

    def state_dict_clone(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone() for name, tensor in self._shadow.items()}

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return the in-place shadow without a host-side clone.

        The attended cgroup has only a small RSS margin after an exact Exit
        step.  ``torch.save`` serializes each storage synchronously, so keeping
        the device tensors by reference avoids an avoidable full CPU duplicate
        at the checkpoint boundary.  The call is made only after an optimizer
        step and before the next forward can mutate any parameter.
        """

        return {
            "decay": float(self.decay),
            "steps": int(self._steps),
            "shadow": dict(self._shadow),
        }

    def restore_checkpoint_state(
        self,
        state: Mapping[str, Any],
        *,
        model: nn.Module,
    ) -> None:
        if set(state) != {"decay", "steps", "shadow"}:
            raise RuntimeError("[ATTENDED_RESEARCH_WEIGHT_EMA_SCHEMA_INVALID]")
        if float(state["decay"]) != float(self.decay) or int(state["steps"]) < 0:
            raise RuntimeError("[ATTENDED_RESEARCH_WEIGHT_EMA_STATE_INVALID]")
        shadow = state["shadow"]
        expected = model.state_dict()
        if not isinstance(shadow, Mapping) or set(shadow) != set(expected):
            raise RuntimeError("[ATTENDED_RESEARCH_WEIGHT_EMA_KEYS_INVALID]")
        restored: Dict[str, torch.Tensor] = {}
        for name, expected_tensor in expected.items():
            value = shadow.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected_tensor.shape
                or value.dtype != expected_tensor.dtype
                or not bool(torch.isfinite(value).all().item())
            ):
                raise RuntimeError(
                    "[ATTENDED_RESEARCH_WEIGHT_EMA_TENSOR_INVALID] "
                    f"name={name}"
                )
            restored[name] = value.detach().to(expected_tensor.device).clone()
        self._shadow = restored
        self._steps = int(state["steps"])


def _step_partial_gradient_accumulation(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    configured_steps: int,
    observed_steps: int,
    weight_ema: Optional["_WeightEma"] = None,
) -> bool:
    """Apply a final partial accumulation with the same mean-gradient scale."""

    configured = int(configured_steps)
    observed = int(observed_steps)
    if observed == 0:
        return False
    if configured < 1 or observed < 1 or observed >= configured:
        raise RuntimeError(
            "[ENTRY_GRAD_ACCUM_REMAINDER_INVALID] "
            f"configured={configured} observed={observed}"
        )
    remainder_rescale = float(configured) / float(observed)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(remainder_rescale)
    torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    # The remainder step is a real optimizer step, so the weight EMA must see
    # it too; skipping it here would make the average depend on the batch count
    # modulo the accumulation width.
    if weight_ema is not None:
        weight_ema.update(model)
    return True


_UNIFIED_EXIT_MOVEMENT_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "m5_mtf_route": (
        "tf_input_scale_m5",
        "mtf_feature_context_gate.m5__",
        "mtf_nominal_embeddings.m5_",
    ),
    "path_projection": ("exit_path_proj.",),
    "path_encoder": ("exit_path_encoder.",),
    "side_embedding": ("exit_side_embedding.",),
    "entry_path_attention": ("exit_entry_path_attention.",),
    "entry_path_fusion": (
        "exit_entry_query_norm.",
        "exit_fuse.",
    ),
    "action_head": ("head_exit_action.",),
}


def _capture_unified_exit_initial_state(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    selected = {
        key: value.detach().cpu().clone()
        for key, value in state.items()
        if any(
            key.startswith(prefix)
            for prefixes in _UNIFIED_EXIT_MOVEMENT_PREFIXES.values()
            for prefix in prefixes
        )
    }
    missing_components = [
        component
        for component, prefixes in _UNIFIED_EXIT_MOVEMENT_PREFIXES.items()
        if not any(
            any(key.startswith(prefix) for prefix in prefixes)
            for key in selected
        )
    ]
    if missing_components:
        raise RuntimeError(
            "[UNIFIED_EXIT_INITIAL_STATE_MISSING] "
            f"components={missing_components}"
        )
    return selected


def _unified_exit_movement_proof(
    initial_state: Mapping[str, torch.Tensor],
    selected_state: Mapping[str, torch.Tensor],
    *,
    selected_checkpoint_epoch: int,
) -> Dict[str, Any]:
    parameter_max_abs_delta: Dict[str, float] = {}
    component_max_abs_delta: Dict[str, float] = {}
    for key, initial in initial_state.items():
        selected = selected_state.get(key)
        if not isinstance(selected, torch.Tensor) or selected.shape != initial.shape:
            raise RuntimeError(
                f"[UNIFIED_EXIT_SELECTED_STATE_INVALID] parameter={key}"
            )
        delta = float(
            torch.max(
                torch.abs(
                    selected.detach().cpu().to(torch.float64)
                    - initial.to(torch.float64)
                )
            ).item()
        )
        if not math.isfinite(delta):
            raise RuntimeError(
                f"[UNIFIED_EXIT_MOVEMENT_NONFINITE] parameter={key}"
            )
        parameter_max_abs_delta[key] = delta
    for component, prefixes in _UNIFIED_EXIT_MOVEMENT_PREFIXES.items():
        deltas = [
            delta
            for key, delta in parameter_max_abs_delta.items()
            if any(key.startswith(prefix) for prefix in prefixes)
        ]
        if not deltas:
            raise RuntimeError(
                f"[UNIFIED_EXIT_MOVEMENT_COMPONENT_MISSING] {component}"
            )
        component_max_abs_delta[component] = max(deltas)
    dead = [
        component
        for component, delta in component_max_abs_delta.items()
        if delta <= 0.0
    ]
    if dead:
        raise RuntimeError(
            "[UNIFIED_EXIT_SELECTED_CHECKPOINT_UNTRAINED] "
            f"components={dead}"
        )
    return {
        "schema_version": "gx1_unified_exit_parameter_movement_v1",
        "selected_checkpoint_epoch": int(selected_checkpoint_epoch),
        "component_max_abs_delta": component_max_abs_delta,
        "parameter_max_abs_delta": parameter_max_abs_delta,
        "all_exit_components_moved": True,
    }


def _unified_exit_full_population_eval_loss(
    *,
    model: nn.Module,
    target_model: nn.Module,
    entry_decision_representations: torch.Tensor,
    target_entry_decision_representations: torch.Tensor,
    entry_row_indices: torch.Tensor,
    dataset: "EntryV10CtxDataset",
    device: torch.device,
    exit_cooperation_gate_epoch: Optional[dict[str, dict[str, Any]]] = None,
    exit_feature_tf_gate_epoch: Optional[dict[str, Any]] = None,
    full_trajectory_accumulator: Optional[dict[str, Any]] = None,
) -> tuple[
    torch.Tensor,
    Dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Evaluate raw-bps Bellman Q regression over every valid action cell."""

    return _episode_native_exit_eval_loss(
        model=model,
        target_model=target_model,
        entry_decision_representations=entry_decision_representations,
        target_entry_decision_representations=(
            target_entry_decision_representations
        ),
        entry_row_indices=entry_row_indices,
        dataset=dataset,
        device=device,
        exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
        exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
        full_trajectory_accumulator=full_trajectory_accumulator,
    )

def _train_unified_exit_full_population(
    *,
    model: nn.Module,
    target_model: nn.Module,
    entry_decision_representations: torch.Tensor,
    target_entry_decision_representations: torch.Tensor,
    entry_row_indices: torch.Tensor,
    dataset: "EntryV10CtxDataset",
    device: torch.device,
    grad_accum_steps: int,
    exit_cooperation_gate_epoch: dict[str, dict[str, Any]],
    exit_feature_tf_gate_epoch: dict[str, Any],
    profile_timing: bool = False,
    exit_action_forward_chunk_rows: Optional[int] = None,
) -> tuple[
    torch.Tensor,
    Dict[str, Any],
    torch.Tensor,
    torch.Tensor,
]:
    """Train one complete causal scan per eligible two-side episode."""

    return _episode_native_exit_train(
        model=model,
        target_model=target_model,
        entry_decision_representations=entry_decision_representations,
        target_entry_decision_representations=(
            target_entry_decision_representations
        ),
        entry_row_indices=entry_row_indices,
        dataset=dataset,
        device=device,
        grad_accum_steps=grad_accum_steps,
        exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
        exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
        profile_timing=profile_timing,
        exit_action_forward_chunk_rows=exit_action_forward_chunk_rows,
    )

def _forward_unified_exit_episode_pack(
    *,
    model: nn.Module,
    entry_decision_representation: torch.Tensor,
    episode: Mapping[str, Any],
    device: torch.device,
    exit_cooperation_gate_epoch: Optional[dict[str, dict[str, Any]]] = None,
    exit_feature_tf_gate_epoch: Optional[dict[str, Any]] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Forward one complete hash-validated episode exactly once."""

    tf_names = tuple(tf.lower() for tf in EXIT_MTF_CONTEXT_TIMEFRAMES)
    inputs = {
        "entry_decision_representation": entry_decision_representation,
        "exit_local_history_x": torch.from_numpy(
            np.asarray(episode["exit_local_history_x"], dtype=np.float32)
        ).unsqueeze(0).to(device),
        "exit_state_ctx_cat": torch.from_numpy(
            np.asarray(episode["exit_state_ctx_cat"], dtype=np.int64)
        ).unsqueeze(0).to(device),
        "exit_state_ctx_cont": torch.from_numpy(
            np.asarray(episode["exit_state_ctx_cont"], dtype=np.float32)
        ).unsqueeze(0).to(device),
        "exit_path_x": torch.from_numpy(
            np.asarray(episode["exit_path_x"], dtype=np.float32)
        ).unsqueeze(0).to(device),
        "exit_mtf_histories": {
            tf: torch.from_numpy(
                np.asarray(episode[f"exit_mtf_history_{tf}"], dtype=np.float32)
            ).unsqueeze(0).to(device)
            for tf in tf_names
        },
        "exit_mtf_gathers": {
            tf: torch.from_numpy(
                np.asarray(episode[f"exit_mtf_gather_{tf}"], dtype=np.int64)
            ).unsqueeze(0).to(device)
            for tf in tf_names
        },
        "exit_mtf_history_lengths": {
            tf: torch.tensor(
                [len(np.asarray(episode[f"exit_mtf_history_{tf}"]))],
                dtype=torch.long,
                device=device,
            )
            for tf in tf_names
        },
    }
    output = model.forward_exit_episode(**inputs)
    q_values = output.get("exit_action_q_bps")
    model_valid = output.get("exit_action_valid_mask")
    terminal = output.get("exit_terminal_mask")
    terminal_reason = output.get("exit_terminal_reason_index")
    lengths = output.get("exit_episode_lengths")
    valid = torch.from_numpy(
        np.asarray(episode["exit_action_valid_mask"], dtype=np.bool_)
    ).unsqueeze(0).to(device)
    state_valid = torch.from_numpy(
        np.asarray(episode["exit_state_valid_mask"], dtype=np.bool_)
    ).unsqueeze(0).to(device)
    target_terminal = torch.from_numpy(
        np.asarray(episode["exit_terminal_mask"], dtype=np.bool_)
    ).unsqueeze(0).to(device)
    target_terminal_reason = torch.from_numpy(
        np.asarray(episode["exit_terminal_reason_index"], dtype=np.int64)
    ).unsqueeze(0).to(device)
    target_lengths = torch.from_numpy(
        np.asarray(episode["exit_episode_lengths"], dtype=np.int64)
    ).unsqueeze(0).to(device)
    expected_shape = (1, 2, UNIFIED_EXIT_MAX_PATH_BARS, 2)
    if (
        not isinstance(q_values, torch.Tensor)
        or tuple(q_values.shape) != expected_shape
        or not bool(torch.isfinite(q_values).all().item())
        or not isinstance(model_valid, torch.Tensor)
        or not torch.equal(model_valid, valid)
        or not isinstance(terminal, torch.Tensor)
        or not torch.equal(terminal, target_terminal)
        or not isinstance(terminal_reason, torch.Tensor)
        or tuple(terminal_reason.shape) != (1, 2, UNIFIED_EXIT_MAX_PATH_BARS)
        or not torch.equal(terminal_reason, target_terminal_reason)
        or not isinstance(lengths, torch.Tensor)
        or not torch.equal(lengths, target_lengths)
        or not torch.equal(valid[..., 1], state_valid)
        or not torch.equal(valid[..., 0], state_valid & ~target_terminal)
    ):
        raise RuntimeError("[UNIFIED_EXIT_EPISODE_FORWARD_INVALID]")
    if exit_cooperation_gate_epoch is not None:
        if exit_feature_tf_gate_epoch is None:
            raise RuntimeError("[UNIFIED_EXIT_GATE_ACCUMULATOR_PAIR_REQUIRED]")
        gate_view = _unified_exit_gate_view(output)
        _accumulate_cooperation_gate_epoch(
            exit_cooperation_gate_epoch,
            gate_view,
            gate_widths=_UNIFIED_EXIT_COOPERATION_GATE_WIDTHS,
        )
        _accumulate_feature_tf_gate_epoch(
            exit_feature_tf_gate_epoch,
            gate_view,
            gate_shape=_UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE,
        )
    return q_values, valid, state_valid, target_terminal, target_lengths


def _forward_unified_exit_episode_batch(
    *,
    model: nn.Module,
    entry_decision_representations: torch.Tensor,
    episodes: Sequence[Mapping[str, Any]],
    device: torch.device,
    exit_cooperation_gate_epoch: Optional[dict[str, dict[str, Any]]] = None,
    exit_feature_tf_gate_epoch: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad only batch transport and scan all episodes in one call."""

    batch_size = len(episodes)
    if (
        batch_size < 1
        or entry_decision_representations.ndim != 2
        or int(entry_decision_representations.shape[0]) != batch_size
    ):
        raise RuntimeError("[UNIFIED_EXIT_EPISODE_BATCH_INVALID]")
    tf_names = tuple(tf.lower() for tf in EXIT_MTF_CONTEXT_TIMEFRAMES)
    inputs: dict[str, Any] = {
        "entry_decision_representation": entry_decision_representations,
        "exit_local_history_x": torch.from_numpy(
            np.stack(
                [episode["exit_local_history_x"] for episode in episodes],
                axis=0,
            ).astype(np.float32, copy=False)
        ).to(device),
        "exit_state_ctx_cat": torch.from_numpy(
            np.stack(
                [episode["exit_state_ctx_cat"] for episode in episodes],
                axis=0,
            ).astype(np.int64, copy=False)
        ).to(device),
        "exit_state_ctx_cont": torch.from_numpy(
            np.stack(
                [episode["exit_state_ctx_cont"] for episode in episodes],
                axis=0,
            ).astype(np.float32, copy=False)
        ).to(device),
        "exit_path_x": torch.from_numpy(
            np.stack(
                [episode["exit_path_x"] for episode in episodes], axis=0
            ).astype(np.float32, copy=False)
        ).to(device),
        "exit_mtf_histories": {},
        "exit_mtf_gathers": {},
        "exit_mtf_history_lengths": {},
    }
    for tf in tf_names:
        histories = [
            np.asarray(episode[f"exit_mtf_history_{tf}"], dtype=np.float32)
            for episode in episodes
        ]
        lengths = np.asarray([len(history) for history in histories], dtype=np.int64)
        max_rows = int(lengths.max())
        padded = np.zeros(
            (batch_size, max_rows, histories[0].shape[1]), dtype=np.float32
        )
        for row, history in enumerate(histories):
            padded[row, : len(history)] = history
        inputs["exit_mtf_histories"][tf] = torch.from_numpy(padded).to(device)
        inputs["exit_mtf_gathers"][tf] = torch.from_numpy(
            np.stack(
                [episode[f"exit_mtf_gather_{tf}"] for episode in episodes],
                axis=0,
            ).astype(np.int64, copy=False)
        ).to(device)
        inputs["exit_mtf_history_lengths"][tf] = torch.from_numpy(lengths).to(
            device
        )
    output = model.forward_exit_episode(**inputs)
    q_values = output.get("exit_action_q_bps")
    model_valid = output.get("exit_action_valid_mask")
    terminal = output.get("exit_terminal_mask")
    terminal_reason = output.get("exit_terminal_reason_index")
    lengths = output.get("exit_episode_lengths")
    valid = torch.from_numpy(
        np.stack(
            [episode["exit_action_valid_mask"] for episode in episodes], axis=0
        ).astype(np.bool_, copy=False)
    ).to(device)
    state_valid = torch.from_numpy(
        np.stack(
            [episode["exit_state_valid_mask"] for episode in episodes], axis=0
        ).astype(np.bool_, copy=False)
    ).to(device)
    target_terminal = torch.from_numpy(
        np.stack(
            [episode["exit_terminal_mask"] for episode in episodes], axis=0
        ).astype(np.bool_, copy=False)
    ).to(device)
    target_reason = torch.from_numpy(
        np.stack(
            [episode["exit_terminal_reason_index"] for episode in episodes],
            axis=0,
        ).astype(np.int64, copy=False)
    ).to(device)
    target_lengths = torch.from_numpy(
        np.stack(
            [episode["exit_episode_lengths"] for episode in episodes], axis=0
        ).astype(np.int64, copy=False)
    ).to(device)
    expected_shape = (batch_size, 2, UNIFIED_EXIT_MAX_PATH_BARS, 2)
    if (
        not isinstance(q_values, torch.Tensor)
        or tuple(q_values.shape) != expected_shape
        or not bool(torch.isfinite(q_values).all().item())
        or not isinstance(model_valid, torch.Tensor)
        or not torch.equal(model_valid, valid)
        or not isinstance(terminal, torch.Tensor)
        or not torch.equal(terminal, target_terminal)
        or not isinstance(terminal_reason, torch.Tensor)
        or not torch.equal(terminal_reason, target_reason)
        or not isinstance(lengths, torch.Tensor)
        or not torch.equal(lengths, target_lengths)
        or not torch.equal(valid[..., 1], state_valid)
        or not torch.equal(valid[..., 0], state_valid & ~target_terminal)
    ):
        raise RuntimeError("[UNIFIED_EXIT_EPISODE_BATCH_FORWARD_INVALID]")
    if exit_cooperation_gate_epoch is not None:
        if exit_feature_tf_gate_epoch is None:
            raise RuntimeError("[UNIFIED_EXIT_GATE_ACCUMULATOR_PAIR_REQUIRED]")
        gate_view = _unified_exit_gate_view(output)
        _accumulate_cooperation_gate_epoch(
            exit_cooperation_gate_epoch,
            gate_view,
            gate_widths=_UNIFIED_EXIT_COOPERATION_GATE_WIDTHS,
        )
        _accumulate_feature_tf_gate_epoch(
            exit_feature_tf_gate_epoch,
            gate_view,
            gate_shape=_UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE,
        )
    return q_values, valid, state_valid, target_terminal, target_lengths


def _episode_stats_update(
    stats: dict[str, int],
    *,
    q_values: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    flat_q = q_values.reshape(-1, 2)
    flat_targets = targets.reshape(-1, 2)
    flat_valid = valid.reshape(-1, 2)
    target_value = flat_targets.masked_fill(~flat_valid, -torch.inf).amax(
        dim=1, keepdim=True
    )
    flat_equivalence = (flat_targets == target_value) & flat_valid
    target_equivalent = flat_equivalence[:, 0] & flat_equivalence[:, 1]
    masked_q = flat_q.masked_fill(~flat_valid, -torch.inf)
    predicted_tie = flat_valid.all(dim=1) & (flat_q[:, 0] == flat_q[:, 1])
    predictions = torch.argmax(masked_q, dim=1)
    stats["population_rows"] += int(flat_q.shape[0])
    stats["q_valid_cells"] += int(flat_valid.sum().item())
    stats["target_equivalent_action_rows"] += int(target_equivalent.sum().item())
    stats["predicted_tied_rows"] += int(predicted_tie.sum().item())
    stats["target_tied_prediction_unique_rows"] += int(
        (target_equivalent & ~predicted_tie).sum().item()
    )
    stats["unique_target_action_agreement_rows"] += int(
        (
            flat_equivalence.gather(1, predictions[:, None]).squeeze(1)
            & ~predicted_tie
            & ~target_equivalent
        ).sum().item()
    )
    stats["hold_target_greedy_rows"] += int(flat_equivalence[:, 0].sum().item())
    stats["exit_now_target_greedy_rows"] += int(flat_equivalence[:, 1].sum().item())


def _empty_exit_stats() -> dict[str, int]:
    return {
        "population_rows": 0,
        "q_valid_cells": 0,
        "target_equivalent_action_rows": 0,
        "hold_target_greedy_rows": 0,
        "exit_now_target_greedy_rows": 0,
        "unique_target_action_agreement_rows": 0,
        "predicted_tied_rows": 0,
        "target_tied_prediction_unique_rows": 0,
        "eligible_entry_rows": 0,
    }


def _fitted_q_targets_for_episode(
    *,
    target_model: nn.Module,
    target_entry_decision_representation: torch.Tensor,
    episode: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one immutable target snapshot and stop-gradient Bellman owner."""

    if target_model.training:
        raise RuntimeError("[UNIFIED_EXIT_TARGET_MODEL_MUST_BE_FROZEN_EVAL]")
    with torch.no_grad():
        target_q, valid, state_valid, terminal, _lengths = (
            _forward_unified_exit_episode_pack(
                model=target_model,
                entry_decision_representation=(
                    target_entry_decision_representation.detach()
                ),
                episode=episode,
                device=device,
            )
        )
        rewards = torch.from_numpy(
            np.asarray(episode["exit_now_reward_bps"], dtype=np.float32)
        ).unsqueeze(0).to(device)
        targets, target_mask = build_unified_exit_fitted_q_targets(
            frozen_target_q_bps=target_q,
            exit_now_reward_bps=rewards,
            action_valid_mask=valid,
            state_valid_mask=state_valid,
            terminal_mask=terminal,
        )
    if targets.requires_grad or target_mask.requires_grad:
        raise RuntimeError("[UNIFIED_EXIT_FITTED_Q_TARGET_NOT_FROZEN]")
    return targets, target_mask, terminal


def _fitted_q_targets_for_episode_batch(
    *,
    target_model: nn.Module,
    target_entry_decision_representations: torch.Tensor,
    episodes: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if target_model.training:
        raise RuntimeError("[UNIFIED_EXIT_TARGET_MODEL_MUST_BE_FROZEN_EVAL]")
    with torch.no_grad():
        target_q, valid, state_valid, terminal, _lengths = (
            _forward_unified_exit_episode_batch(
                model=target_model,
                entry_decision_representations=(
                    target_entry_decision_representations.detach()
                ),
                episodes=episodes,
                device=device,
            )
        )
        rewards = torch.from_numpy(
            np.stack(
                [episode["exit_now_reward_bps"] for episode in episodes],
                axis=0,
            ).astype(np.float32, copy=False)
        ).to(device)
        targets, target_mask = build_unified_exit_fitted_q_targets(
            frozen_target_q_bps=target_q,
            exit_now_reward_bps=rewards,
            action_valid_mask=valid,
            state_valid_mask=state_valid,
            terminal_mask=terminal,
        )
        first_side_values = unified_exit_first_state_side_values(
            frozen_target_q_bps=target_q,
            action_valid_mask=valid,
            state_valid_mask=state_valid,
        )
        first_side_valid = state_valid[..., 0]
    return (
        targets,
        target_mask,
        terminal,
        first_side_values,
        first_side_valid,
    )


def _episode_native_exit_eval_loss(
    *,
    model: nn.Module,
    target_model: nn.Module,
    entry_decision_representations: torch.Tensor,
    target_entry_decision_representations: torch.Tensor,
    entry_row_indices: torch.Tensor,
    dataset: "EntryV10CtxDataset",
    device: torch.device,
    exit_cooperation_gate_epoch: Optional[dict[str, dict[str, Any]]],
    exit_feature_tf_gate_epoch: Optional[dict[str, Any]],
    full_trajectory_accumulator: Optional[dict[str, Any]] = None,
) -> tuple[
    torch.Tensor,
    Dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if (
        entry_decision_representations.ndim != 2
        or entry_row_indices.ndim != 1
        or int(entry_decision_representations.shape[0])
        != int(entry_row_indices.shape[0])
        or target_entry_decision_representations.shape
        != entry_decision_representations.shape
    ):
        raise RuntimeError("[UNIFIED_EXIT_ENTRY_BATCH_SHAPE_INVALID]")
    rows = entry_row_indices.detach().cpu().tolist()
    if full_trajectory_accumulator is not None:
        if "entry_rows_scanned" not in full_trajectory_accumulator:
            raise RuntimeError("UNIFIED_EXIT_FULL_VAL_ACCUMULATOR_SCHEMA_INVALID")
        full_trajectory_accumulator["entry_rows_scanned"] += len(rows)
    materialized = [
        dataset.materialize_full_exit_episode(int(index)) for index in rows
    ]
    selected = [index for index, episode in enumerate(materialized) if episode is not None]
    loss_sum = entry_decision_representations.sum() * 0.0
    stats = _empty_exit_stats()
    if not selected:
        entry_targets, entry_valid, _ = build_entry_fitted_q_targets(
            frozen_exit_first_state_values_bps=torch.zeros(
                (len(rows), 2),
                device=device,
                dtype=entry_decision_representations.dtype,
            ),
            exit_side_valid_mask=torch.zeros(
                (len(rows), 2), device=device, dtype=torch.bool
            ),
            episode_pack_sha256=[None] * len(rows),
            fill_binding_sha256=[None] * len(rows),
        )
        return (
            loss_sum,
            {**stats, "raw_loss": 0.0},
            entry_targets,
            entry_valid,
            torch.zeros_like(entry_targets),
        )
    episodes = [materialized[index] for index in selected]
    selected_index = torch.tensor(selected, device=device, dtype=torch.long)
    q_values, valid, _state_valid, _terminal, _lengths = (
        _forward_unified_exit_episode_batch(
            model=model,
            entry_decision_representations=entry_decision_representations.index_select(
                0, selected_index
            ),
            episodes=episodes,
            device=device,
            exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
            exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
        )
    )
    (
        targets,
        target_mask,
        _target_terminal,
        first_side_values,
        first_side_valid,
    ) = (
        _fitted_q_targets_for_episode_batch(
            target_model=target_model,
            target_entry_decision_representations=(
                target_entry_decision_representations.index_select(
                    0, selected_index
                )
            ),
            episodes=episodes,
            device=device,
        )
    )
    if not torch.equal(valid, target_mask):
        raise RuntimeError("[UNIFIED_EXIT_FITTED_Q_MASK_SPLIT_BRAIN]")
    loss_sum = loss_sum + nn.functional.mse_loss(
        q_values[valid], targets[valid], reduction="sum"
    )
    stats["eligible_entry_rows"] = len(episodes)
    _episode_stats_update(
        stats, q_values=q_values, targets=targets, valid=valid
    )
    if full_trajectory_accumulator is not None:
        _accumulate_unified_exit_full_trajectory(
            full_trajectory_accumulator,
            raw_entry_indices=rows,
            selected_positions=selected,
            episodes=episodes,
            q_values=q_values,
            targets=targets,
            valid=valid,
        )
    valid_cells = int(stats["q_valid_cells"])
    all_first_values = torch.zeros(
        (len(rows), 2), device=device, dtype=first_side_values.dtype
    )
    all_first_valid = torch.zeros(
        (len(rows), 2), device=device, dtype=torch.bool
    )
    all_first_values.index_copy_(0, selected_index, first_side_values)
    all_first_valid.index_copy_(0, selected_index, first_side_valid)
    episode_hashes: list[str | None] = [None] * len(rows)
    fill_hashes: list[str | None] = [None] * len(rows)
    for batch_position, episode in zip(selected, episodes):
        if int(episode["entry_row_index"]) != int(rows[batch_position]):
            raise RuntimeError("[ENTRY_FITTED_Q_EPISODE_ROW_BINDING_INVALID]")
        episode_hash = str(episode["episode_pack_sha256"])
        episode_hashes[batch_position] = episode_hash
        fill_hashes[batch_position] = entry_fill_binding_sha256(
            entry_row_index=int(rows[batch_position]),
            episode_pack_sha256=episode_hash,
            first_exit_state_time_ns=int(
                np.asarray(episode["exit_state_row_time_ns"], dtype=np.int64)[0]
            ),
            exit_entry_bid_ask=episode["exit_entry_bid_ask"],
        )
    entry_targets, entry_valid, _ = build_entry_fitted_q_targets(
        frozen_exit_first_state_values_bps=all_first_values,
        exit_side_valid_mask=all_first_valid,
        episode_pack_sha256=episode_hashes,
        fill_binding_sha256=fill_hashes,
    )
    entry_realized_pnl = torch.zeros_like(entry_targets)
    for selected_row, (episode, episode_q, episode_valid) in enumerate(
        zip(episodes, q_values, valid)
    ):
        rewards = np.asarray(episode["exit_now_reward_bps"], dtype=np.float64)
        for side_index in range(2):
            replay = replay_unified_exit_fitted_q_policy(
                predicted_q_bps=episode_q[side_index].detach().cpu().numpy(),
                action_valid_mask=episode_valid[side_index].detach().cpu().numpy(),
                exit_now_reward_bps=rewards[side_index],
            )
            entry_realized_pnl[selected_index[selected_row], side_index] = float(
                replay["realized_executable_pnl_bps"]
            )
    if valid_cells <= 0:
        return (
            loss_sum,
            {**stats, "raw_loss": 0.0},
            entry_targets,
            entry_valid,
            entry_realized_pnl,
        )
    raw_loss = loss_sum / float(valid_cells)
    return (
        raw_loss,
        {**stats, "raw_loss": float(raw_loss.detach().cpu().item())},
        entry_targets,
        entry_valid,
        entry_realized_pnl,
    )


def _synchronized_exit_profile_clock(device: torch.device) -> float:
    """Return a wall clock that includes queued CUDA work for one probe only."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _episode_native_exit_train(
    *,
    model: nn.Module,
    target_model: nn.Module,
    entry_decision_representations: torch.Tensor,
    target_entry_decision_representations: torch.Tensor,
    entry_row_indices: torch.Tensor,
    dataset: "EntryV10CtxDataset",
    device: torch.device,
    grad_accum_steps: int,
    exit_cooperation_gate_epoch: dict[str, dict[str, Any]],
    exit_feature_tf_gate_epoch: dict[str, Any],
    profile_timing: bool = False,
    exit_action_forward_chunk_rows: Optional[int] = None,
) -> tuple[
    torch.Tensor,
    Dict[str, Any],
    torch.Tensor,
    torch.Tensor,
]:
    if grad_accum_steps < 1:
        raise RuntimeError("[UNIFIED_EXIT_ENTRY_BATCH_SHAPE_INVALID]")
    if target_entry_decision_representations.shape != entry_decision_representations.shape:
        raise RuntimeError("[UNIFIED_EXIT_TARGET_ENTRY_BATCH_SHAPE_INVALID]")
    if exit_action_forward_chunk_rows is not None:
        if int(exit_action_forward_chunk_rows) < 1:
            raise RuntimeError("[UNIFIED_EXIT_ACTION_CHUNK_ROWS_INVALID]")
        # This attended-only path streams one contiguous group of complete
        # episodes at a time.  The canonical route retains the established
        # unchunked implementation below exactly.
        return _episode_native_exit_train_chunked(
            model=model,
            target_model=target_model,
            entry_decision_representations=entry_decision_representations,
            target_entry_decision_representations=(
                target_entry_decision_representations
            ),
            entry_row_indices=entry_row_indices,
            dataset=dataset,
            device=device,
            grad_accum_steps=grad_accum_steps,
            exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
            exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
            profile_timing=profile_timing,
            exit_action_forward_chunk_rows=int(exit_action_forward_chunk_rows),
        )
    profile_start = (
        _synchronized_exit_profile_clock(device) if profile_timing else None
    )
    rows = entry_row_indices.detach().cpu().tolist()
    episodes = [
        dataset.materialize_full_exit_episode(int(index)) for index in rows
    ]
    profile_materialized = (
        _synchronized_exit_profile_clock(device) if profile_timing else None
    )
    total_valid = sum(
        int(np.asarray(episode["exit_action_valid_mask"], dtype=np.bool_).sum())
        for episode in episodes
        if episode is not None
    )
    if total_valid == 0:
        entry_targets, entry_valid, _ = build_entry_fitted_q_targets(
            frozen_exit_first_state_values_bps=torch.zeros(
                (len(rows), 2),
                device=device,
                dtype=entry_decision_representations.dtype,
            ),
            exit_side_valid_mask=torch.zeros(
                (len(rows), 2), device=device, dtype=torch.bool
            ),
            episode_pack_sha256=[None] * len(rows),
            fill_binding_sha256=[None] * len(rows),
        )
        return (
            torch.zeros_like(entry_decision_representations),
            {**_empty_exit_stats(), "raw_loss": 0.0},
            entry_targets,
            entry_valid,
        )
    entry_gradients = torch.zeros_like(entry_decision_representations)
    stats = _empty_exit_stats()
    selected = [index for index, episode in enumerate(episodes) if episode is not None]
    selected_episodes = [episodes[index] for index in selected]
    selected_index = torch.tensor(selected, device=device, dtype=torch.long)
    token = (
        entry_decision_representations.index_select(0, selected_index)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    q_values, valid, _state_valid, _terminal, _lengths = (
        _forward_unified_exit_episode_batch(
            model=model,
            entry_decision_representations=token,
            episodes=selected_episodes,
            device=device,
            exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
            exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
        )
    )
    profile_online_forward = (
        _synchronized_exit_profile_clock(device) if profile_timing else None
    )
    (
        targets,
        target_mask,
        _target_terminal,
        first_side_values,
        first_side_valid,
    ) = (
        _fitted_q_targets_for_episode_batch(
            target_model=target_model,
            target_entry_decision_representations=(
                target_entry_decision_representations.index_select(
                    0, selected_index
                )
            ),
            episodes=selected_episodes,
            device=device,
        )
    )
    profile_target_forward = (
        _synchronized_exit_profile_clock(device) if profile_timing else None
    )
    if not torch.equal(valid, target_mask):
        raise RuntimeError("[UNIFIED_EXIT_FITTED_Q_MASK_SPLIT_BRAIN]")
    q_loss_sum = nn.functional.mse_loss(
        q_values[valid], targets[valid], reduction="sum"
    )
    _episode_stats_update(
        stats, q_values=q_values, targets=targets, valid=valid
    )
    stats["eligible_entry_rows"] = len(selected_episodes)
    (
        torch.exp(-model.task_log_variances["unified_exit_action"])
        * q_loss_sum
        / float(total_valid)
        / float(grad_accum_steps)
    ).backward()
    if token.grad is None or not bool(torch.isfinite(token.grad).all().item()):
        raise RuntimeError("[UNIFIED_EXIT_EPISODE_TOKEN_GRADIENT_INVALID]")
    profile_backward = (
        _synchronized_exit_profile_clock(device) if profile_timing else None
    )
    entry_gradients.index_copy_(0, selected_index, token.grad.detach())
    if (
        int(stats["q_valid_cells"]) != total_valid
        or int(stats["population_rows"])
        != sum(
            int(np.asarray(episode["exit_state_valid_mask"], dtype=np.bool_).sum())
            for episode in episodes
            if episode is not None
        )
    ):
        raise RuntimeError("[UNIFIED_EXIT_EPISODE_POPULATION_COUNT_MISMATCH]")
    all_first_values = torch.zeros(
        (len(rows), 2),
        device=device,
        dtype=first_side_values.dtype,
    )
    all_first_valid = torch.zeros(
        (len(rows), 2), device=device, dtype=torch.bool
    )
    all_first_values.index_copy_(0, selected_index, first_side_values)
    all_first_valid.index_copy_(0, selected_index, first_side_valid)
    episode_hashes: list[str | None] = [None] * len(rows)
    fill_hashes: list[str | None] = [None] * len(rows)
    for batch_position, episode in zip(selected, selected_episodes):
        if int(episode["entry_row_index"]) != int(rows[batch_position]):
            raise RuntimeError("[ENTRY_FITTED_Q_EPISODE_ROW_BINDING_INVALID]")
        episode_hash = str(episode["episode_pack_sha256"])
        episode_hashes[batch_position] = episode_hash
        fill_hashes[batch_position] = entry_fill_binding_sha256(
            entry_row_index=int(rows[batch_position]),
            episode_pack_sha256=episode_hash,
            first_exit_state_time_ns=int(
                np.asarray(episode["exit_state_row_time_ns"], dtype=np.int64)[0]
            ),
            exit_entry_bid_ask=episode["exit_entry_bid_ask"],
        )
    entry_targets, entry_valid, _entry_binding = build_entry_fitted_q_targets(
        frozen_exit_first_state_values_bps=all_first_values,
        exit_side_valid_mask=all_first_valid,
        episode_pack_sha256=episode_hashes,
        fill_binding_sha256=fill_hashes,
    )
    if profile_timing:
        assert (
            profile_start is not None
            and profile_materialized is not None
            and profile_online_forward is not None
            and profile_target_forward is not None
            and profile_backward is not None
        )
        profile_end = _synchronized_exit_profile_clock(device)
        log.info(
            "[UNIFIED_EXIT_PROFILE] eligible_entries=%d materialize_s=%.6f "
            "online_forward_s=%.6f target_forward_s=%.6f "
            "bellman_backward_s=%.6f post_backward_s=%.6f total_s=%.6f",
            len(selected_episodes),
            profile_materialized - profile_start,
            profile_online_forward - profile_materialized,
            profile_target_forward - profile_online_forward,
            profile_backward - profile_target_forward,
            profile_end - profile_backward,
            profile_end - profile_start,
        )
    return (
        entry_gradients,
        {
            **stats,
            "raw_loss": float(q_loss_sum.detach().cpu().item())
            / float(total_valid),
        },
        entry_targets,
        entry_valid,
    )


def _episode_native_exit_train_chunked(
    *,
    model: nn.Module,
    target_model: nn.Module,
    entry_decision_representations: torch.Tensor,
    target_entry_decision_representations: torch.Tensor,
    entry_row_indices: torch.Tensor,
    dataset: "EntryV10CtxDataset",
    device: torch.device,
    grad_accum_steps: int,
    exit_cooperation_gate_epoch: dict[str, dict[str, Any]],
    exit_feature_tf_gate_epoch: dict[str, Any],
    profile_timing: bool,
    exit_action_forward_chunk_rows: int,
) -> tuple[
    torch.Tensor,
    Dict[str, Any],
    torch.Tensor,
    torch.Tensor,
]:
    """Stream attended Exit groups without retaining a full-batch graph.

    This is deliberately separate from the canonical unchunked owner above.
    It is used only by bounded attended research, whose session contract binds
    the group size.  Each group backpropagates its sum-normalized contribution
    immediately, so CUDA can release its attention graph before the next group.
    """

    profile_start = (
        _synchronized_exit_profile_clock(device) if profile_timing else None
    )
    rows = entry_row_indices.detach().cpu().tolist()
    episodes = [
        dataset.materialize_full_exit_episode(int(index)) for index in rows
    ]
    profile_materialized = (
        _synchronized_exit_profile_clock(device) if profile_timing else None
    )
    total_valid = sum(
        int(np.asarray(episode["exit_action_valid_mask"], dtype=np.bool_).sum())
        for episode in episodes
        if episode is not None
    )
    if total_valid == 0:
        entry_targets, entry_valid, _ = build_entry_fitted_q_targets(
            frozen_exit_first_state_values_bps=torch.zeros(
                (len(rows), 2),
                device=device,
                dtype=entry_decision_representations.dtype,
            ),
            exit_side_valid_mask=torch.zeros(
                (len(rows), 2), device=device, dtype=torch.bool
            ),
            episode_pack_sha256=[None] * len(rows),
            fill_binding_sha256=[None] * len(rows),
        )
        return (
            torch.zeros_like(entry_decision_representations),
            {**_empty_exit_stats(), "raw_loss": 0.0},
            entry_targets,
            entry_valid,
        )
    selected = [index for index, episode in enumerate(episodes) if episode is not None]
    selected_episodes = [episodes[index] for index in selected]
    selected_index = torch.tensor(selected, device=device, dtype=torch.long)
    entry_gradients = torch.zeros_like(entry_decision_representations)
    token = (
        entry_decision_representations.index_select(0, selected_index)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    stats = _empty_exit_stats()
    raw_loss_sum = 0.0
    selected_first_values = torch.zeros(
        (len(selected), 2),
        device=device,
        dtype=target_entry_decision_representations.dtype,
    )
    selected_first_valid = torch.zeros(
        (len(selected), 2), device=device, dtype=torch.bool
    )
    profile_online_forward_s = 0.0
    profile_target_forward_s = 0.0
    profile_backward_s = 0.0
    chunk_count = 0
    for start in range(0, len(selected_episodes), exit_action_forward_chunk_rows):
        stop = min(start + exit_action_forward_chunk_rows, len(selected_episodes))
        positions = torch.arange(start, stop, device=device, dtype=torch.long)
        raw_indices = selected_index.index_select(0, positions)
        chunk_episodes = selected_episodes[start:stop]
        chunk_online_start = (
            _synchronized_exit_profile_clock(device) if profile_timing else None
        )
        q_values, valid, _state_valid, _terminal, _lengths = (
            _forward_unified_exit_episode_batch(
                model=model,
                entry_decision_representations=token.index_select(0, positions),
                episodes=chunk_episodes,
                device=device,
                exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
                exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
            )
        )
        chunk_online_end = (
            _synchronized_exit_profile_clock(device) if profile_timing else None
        )
        (
            targets,
            target_mask,
            _target_terminal,
            first_side_values,
            first_side_valid,
        ) = _fitted_q_targets_for_episode_batch(
            target_model=target_model,
            target_entry_decision_representations=(
                target_entry_decision_representations.index_select(0, raw_indices)
            ),
            episodes=chunk_episodes,
            device=device,
        )
        chunk_target_end = (
            _synchronized_exit_profile_clock(device) if profile_timing else None
        )
        if not torch.equal(valid, target_mask):
            raise RuntimeError("[UNIFIED_EXIT_FITTED_Q_MASK_SPLIT_BRAIN]")
        q_loss_sum = nn.functional.mse_loss(
            q_values[valid], targets[valid], reduction="sum"
        )
        _episode_stats_update(
            stats, q_values=q_values, targets=targets, valid=valid
        )
        selected_first_values.index_copy_(0, positions, first_side_values)
        selected_first_valid.index_copy_(0, positions, first_side_valid)
        (
            torch.exp(-model.task_log_variances["unified_exit_action"])
            * q_loss_sum
            / float(total_valid)
            / float(grad_accum_steps)
        ).backward()
        chunk_backward_end = (
            _synchronized_exit_profile_clock(device) if profile_timing else None
        )
        raw_loss_sum += float(q_loss_sum.detach().cpu().item())
        chunk_count += 1
        if profile_timing:
            assert (
                chunk_online_start is not None
                and chunk_online_end is not None
                and chunk_target_end is not None
                and chunk_backward_end is not None
            )
            profile_online_forward_s += chunk_online_end - chunk_online_start
            profile_target_forward_s += chunk_target_end - chunk_online_end
            profile_backward_s += chunk_backward_end - chunk_target_end
        # Keep no large Exit graph alive across groups.  This is important on
        # WSL/DXG where a cached near-capacity allocation can fail residency.
        del q_values, valid, targets, target_mask, first_side_values, first_side_valid

    if token.grad is None or not bool(torch.isfinite(token.grad).all().item()):
        raise RuntimeError("[UNIFIED_EXIT_EPISODE_TOKEN_GRADIENT_INVALID]")
    entry_gradients.index_copy_(0, selected_index, token.grad.detach())
    if (
        int(stats["q_valid_cells"]) != total_valid
        or int(stats["population_rows"])
        != sum(
            int(np.asarray(episode["exit_state_valid_mask"], dtype=np.bool_).sum())
            for episode in episodes
            if episode is not None
        )
    ):
        raise RuntimeError("[UNIFIED_EXIT_EPISODE_POPULATION_COUNT_MISMATCH]")
    all_first_values = torch.zeros(
        (len(rows), 2), device=device, dtype=selected_first_values.dtype
    )
    all_first_valid = torch.zeros(
        (len(rows), 2), device=device, dtype=torch.bool
    )
    all_first_values.index_copy_(0, selected_index, selected_first_values)
    all_first_valid.index_copy_(0, selected_index, selected_first_valid)
    episode_hashes: list[str | None] = [None] * len(rows)
    fill_hashes: list[str | None] = [None] * len(rows)
    for batch_position, episode in zip(selected, selected_episodes):
        if int(episode["entry_row_index"]) != int(rows[batch_position]):
            raise RuntimeError("[ENTRY_FITTED_Q_EPISODE_ROW_BINDING_INVALID]")
        episode_hash = str(episode["episode_pack_sha256"])
        episode_hashes[batch_position] = episode_hash
        fill_hashes[batch_position] = entry_fill_binding_sha256(
            entry_row_index=int(rows[batch_position]),
            episode_pack_sha256=episode_hash,
            first_exit_state_time_ns=int(
                np.asarray(episode["exit_state_row_time_ns"], dtype=np.int64)[0]
            ),
            exit_entry_bid_ask=episode["exit_entry_bid_ask"],
        )
    entry_targets, entry_valid, _entry_binding = build_entry_fitted_q_targets(
        frozen_exit_first_state_values_bps=all_first_values,
        exit_side_valid_mask=all_first_valid,
        episode_pack_sha256=episode_hashes,
        fill_binding_sha256=fill_hashes,
    )
    if profile_timing:
        assert profile_start is not None and profile_materialized is not None
        profile_end = _synchronized_exit_profile_clock(device)
        log.info(
            "[UNIFIED_EXIT_PROFILE] eligible_entries=%d chunk_rows=%d chunks=%d "
            "materialize_s=%.6f online_forward_s=%.6f target_forward_s=%.6f "
            "bellman_backward_s=%.6f post_backward_s=%.6f total_s=%.6f",
            len(selected_episodes),
            exit_action_forward_chunk_rows,
            chunk_count,
            profile_materialized - profile_start,
            profile_online_forward_s,
            profile_target_forward_s,
            profile_backward_s,
            profile_end - profile_materialized
            - profile_online_forward_s
            - profile_target_forward_s
            - profile_backward_s,
            profile_end - profile_start,
        )
    return (
        entry_gradients,
        {**stats, "raw_loss": raw_loss_sum / float(total_valid)},
        entry_targets,
        entry_valid,
    )


def _collect_unified_exit_influence_sample(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], list[int], list[int]]:
    """Collect four deterministic complete VAL episodes (both side axes)."""

    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, EntryV10CtxDataset):
        raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_DATASET_INVALID")
    dataset_indices = np.asarray(dataset.indices, dtype=np.int64)
    if dataset_indices.ndim != 1 or dataset_indices.size < UNIFIED_EXIT_INFLUENCE_SIDE_ROWS:
        raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_INDEX_POPULATION_INVALID")
    offsets = (
        np.arange(UNIFIED_EXIT_INFLUENCE_SIDE_ROWS, dtype=np.int64)
        * (dataset_indices.size - 1)
        // UNIFIED_EXIT_INFLUENCE_SIDE_ROWS
    )
    target_rows = set(int(value) for value in dataset_indices[offsets].tolist())
    episodes: list[Mapping[str, Any]] = []
    tokens: list[torch.Tensor] = []
    entry_rows: list[int] = []
    decision_times_ns: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = _model_forward_fp32(
                model,
                batch["seq_x"].to(device),
                batch["snap_x"].to(device),
                ctx_cat=batch["ctx_cat"].to(device),
                ctx_cont=batch["ctx_cont"].to(device),
                **_multi_tf_kwargs_from_batch(batch, device),
            )
            batch_tokens = out.get(UNIFIED_EXIT_MODEL_REPRESENTATION_KEY)
            indices = batch.get("entry_row_index")
            if (
                not isinstance(batch_tokens, torch.Tensor)
                or not isinstance(indices, torch.Tensor)
                or batch_tokens.ndim != 2
                or indices.ndim != 1
                or int(batch_tokens.shape[0]) != int(indices.shape[0])
            ):
                raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_ENTRY_TOKEN_INVALID")
            for batch_index, raw_entry_row in enumerate(indices.detach().cpu().tolist()):
                entry_row = int(raw_entry_row)
                if entry_row not in target_rows:
                    continue
                episode = dataset.materialize_full_exit_episode(entry_row)
                if episode is None:
                    continue
                episodes.append(episode)
                tokens.append(batch_tokens[batch_index].detach())
                entry_rows.extend((entry_row, entry_row))
                first_time = int(np.asarray(episode["exit_decision_time_ns"])[0])
                decision_times_ns.extend((first_time, first_time))
                target_rows.remove(entry_row)
            if not target_rows:
                break
    if (
        target_rows
        or len(episodes) != UNIFIED_EXIT_INFLUENCE_SIDE_ROWS
        or len(entry_rows) != UNIFIED_EXIT_INFLUENCE_SAMPLE_COUNT
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_SAMPLE_INCOMPLETE")
    sampled: dict[str, Any] = {
        "entry_decision_representation": torch.stack(tokens, dim=0).to(device),
        "exit_local_history_x": torch.from_numpy(np.stack([
            episode["exit_local_history_x"] for episode in episodes
        ]).astype(np.float32, copy=False)).to(device),
        "exit_state_ctx_cat": torch.from_numpy(np.stack([
            episode["exit_state_ctx_cat"] for episode in episodes
        ]).astype(np.int64, copy=False)).to(device),
        "exit_state_ctx_cont": torch.from_numpy(np.stack([
            episode["exit_state_ctx_cont"] for episode in episodes
        ]).astype(np.float32, copy=False)).to(device),
        "exit_path_x": torch.from_numpy(np.stack([
            episode["exit_path_x"] for episode in episodes
        ]).astype(np.float32, copy=False)).to(device),
        "exit_mtf_histories": {},
        "exit_mtf_gathers": {},
        "exit_mtf_history_lengths": {},
    }
    for tf in (name.lower() for name in EXIT_MTF_CONTEXT_TIMEFRAMES):
        histories = [
            np.asarray(episode[f"exit_mtf_history_{tf}"], dtype=np.float32)
            for episode in episodes
        ]
        lengths = np.asarray([len(history) for history in histories], dtype=np.int64)
        padded = np.zeros(
            (len(episodes), int(lengths.max()), histories[0].shape[1]),
            dtype=np.float32,
        )
        for row, history in enumerate(histories):
            padded[row, : len(history)] = history
        sampled["exit_mtf_histories"][tf] = torch.from_numpy(padded).to(device)
        sampled["exit_mtf_gathers"][tf] = torch.from_numpy(np.stack([
            episode[f"exit_mtf_gather_{tf}"] for episode in episodes
        ]).astype(np.int64, copy=False)).to(device)
        sampled["exit_mtf_history_lengths"][tf] = torch.from_numpy(lengths).to(device)
    return sampled, entry_rows, decision_times_ns


def _unified_exit_influence_forward(
    model: nn.Module,
    inputs: Mapping[str, Any],
) -> torch.Tensor:
    output = model.forward_exit_episode(**dict(inputs))
    q_values = output.get("exit_action_q_bps")
    if (
        not isinstance(q_values, torch.Tensor)
        or q_values.ndim != 4
        or tuple(q_values.shape[:2]) != (UNIFIED_EXIT_INFLUENCE_SIDE_ROWS, 2)
        or int(q_values.shape[-1]) != 2
        or not bool(torch.isfinite(q_values).all().item())
    ):
        raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_Q_INVALID")
    return q_values


def _next_valid_category(
    values: torch.Tensor,
    domain_values: object,
) -> torch.Tensor:
    domain = tuple(int(value) for value in domain_values)
    if len(domain) < 2 or len(domain) != len(set(domain)):
        raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_CATEGORY_DOMAIN_INVALID")
    result = values.clone()
    covered = torch.zeros_like(values, dtype=torch.bool)
    for position, value in enumerate(domain):
        mask = values == int(value)
        result[mask] = int(domain[(position + 1) % len(domain)])
        covered |= mask
    if not bool(covered.all().item()):
        raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_CATEGORY_VALUE_INVALID")
    return result


def _unified_exit_input_influence_contract(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Prove every physical Exit input can move the HOLD/EXIT_NOW margin."""

    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, EntryV10CtxDataset):
        raise RuntimeError("UNIFIED_EXIT_INPUT_INFLUENCE_DATASET_INVALID")
    signal_names = [str(name) for name in dataset.signal_names]
    ownership = unified_exit_input_influence_layout(signal_names)
    sampled, entry_rows, decision_times_ns = _collect_unified_exit_influence_sample(
        model=model,
        loader=loader,
        device=device,
    )
    numeric_input_names = {
        "seq_signal": "exit_local_history_x",
        "ctx_cont": "exit_state_ctx_cont",
        **{
            f"seq_{tf.lower()}": tf.lower()
            for tf in EXIT_MTF_CONTEXT_TIMEFRAMES
        },
        "entry_decision_representation": "entry_decision_representation",
        "exit_path": "exit_path_x",
    }
    numeric_tensors = {
        surface: (
            sampled["exit_mtf_histories"][input_name]
            if surface.startswith("seq_") and surface != "seq_signal"
            else sampled[input_name]
        ).detach().clone().requires_grad_(True)
        for surface, input_name in numeric_input_names.items()
    }
    gradient_inputs = {
        name: (
            {key: value.clone() for key, value in source.items()}
            if isinstance(source, Mapping)
            else source.clone()
        )
        for name, source in sampled.items()
    }
    for surface, input_name in numeric_input_names.items():
        if surface.startswith("seq_") and surface != "seq_signal":
            gradient_inputs["exit_mtf_histories"][input_name] = numeric_tensors[
                surface
            ]
        else:
            gradient_inputs[input_name] = numeric_tensors[surface]
    logits = _unified_exit_influence_forward(model, gradient_inputs)
    margin = (logits[..., 1] - logits[..., 0]).sum()
    gradients = torch.autograd.grad(
        margin,
        tuple(numeric_tensors.values()),
        allow_unused=True,
    )
    numeric_metrics: dict[str, Any] = {}
    failures: list[str] = []
    for (surface, tensor), gradient in zip(numeric_tensors.items(), gradients):
        width = int(tensor.shape[-1])
        if gradient is None:
            values = np.zeros(width, dtype=np.float64)
        else:
            absolute = gradient.detach().abs()
            reduce_dims = tuple(range(absolute.ndim - 1))
            reduced = absolute.amax(dim=reduce_dims) if reduce_dims else absolute
            values = reduced.cpu().double().numpy().reshape(-1)
        owner = ownership["numeric"][surface]
        source_indices = np.asarray(owner["source_indices"], dtype=np.int64)
        tokens = [str(token) for token in owner["tokens"]]
        if values.shape != (width,) or source_indices.shape != (len(tokens),):
            raise RuntimeError(
                f"UNIFIED_EXIT_INPUT_INFLUENCE_GRADIENT_SHAPE_INVALID: {surface}"
            )
        selected = values[source_indices]
        metrics: dict[str, Any] = {}
        for token, value in zip(tokens, selected.tolist()):
            row_failures = (
                []
                if math.isfinite(float(value))
                and float(value) > UNIFIED_EXIT_INFLUENCE_GRAD_EPSILON
                else ["exit margin gradient is dead"]
            )
            metrics[token] = {
                "decision": "PASS" if not row_failures else "FAIL",
                "failures": row_failures,
                "max_abs_exit_margin_gradient": float(value),
            }
            failures.extend(f"numeric/{surface}/{token}: {item}" for item in row_failures)
        numeric_metrics[surface] = {
            "tokens": tokens,
            "source_indices": source_indices.tolist(),
            "metrics": metrics,
        }

    with torch.no_grad():
        baseline_margin = (
            lambda value: value[..., 1] - value[..., 0]
        )(_unified_exit_influence_forward(model, sampled))
    categorical_metrics: dict[str, Any] = {}
    for owner in ownership["categorical"]:
        token = str(owner["token"])
        perturbed = {
            name: (
                {key: value.clone() for key, value in source.items()}
                if isinstance(source, Mapping)
                else source.clone()
            )
            for name, source in sampled.items()
        }
        owner_name = str(owner["owner"])
        if owner_name == "ctx_cat_embedding":
            index = int(owner["source_index"])
            perturbed["exit_state_ctx_cat"][..., index] = _next_valid_category(
                perturbed["exit_state_ctx_cat"][..., index], owner["domain"]
            )
        elif owner_name == "ctx_cont_nominal_embedding":
            index = int(owner["source_index"])
            next_value = _next_valid_category(
                perturbed["exit_state_ctx_cont"][..., index], owner["domain"]
            )
            perturbed["exit_state_ctx_cont"][..., index] = next_value.to(
                perturbed["exit_state_ctx_cont"].dtype
            )
        elif owner_name == "signal_ctx_temporal_alias_nominal_embedding":
            signal_index = int(owner["signal_index"])
            ctx_index = int(owner["ctx_cont_index"])
            next_value = _next_valid_category(
                perturbed["exit_state_ctx_cont"][..., ctx_index], owner["domain"]
            )
            perturbed["exit_local_history_x"][..., signal_index] = (
                next_value[:, :1]
                .to(perturbed["exit_local_history_x"].dtype)
                .expand(-1, perturbed["exit_local_history_x"].shape[1])
            )
            perturbed["exit_state_ctx_cont"][..., ctx_index] = next_value.to(
                perturbed["exit_state_ctx_cont"].dtype
            )
        elif owner_name == "mtf_nominal_embedding":
            name = str(owner["timeframe"]).lower()
            index = int(owner["source_index"])
            history = perturbed["exit_mtf_histories"][name]
            history[..., index] = _next_valid_category(
                history[..., index], owner["domain"]
            ).to(history.dtype)
        else:
            raise RuntimeError(
                f"UNIFIED_EXIT_INPUT_INFLUENCE_OWNER_INVALID: {owner_name}"
            )
        with torch.no_grad():
            changed_margin = (
                lambda value: value[..., 1] - value[..., 0]
            )(_unified_exit_influence_forward(model, perturbed))
        deltas = (changed_margin - baseline_margin).abs().detach().cpu().double().numpy()
        max_delta = float(deltas.max())
        changed_rows = int(np.count_nonzero(
            np.any(
                deltas.reshape(UNIFIED_EXIT_INFLUENCE_SIDE_ROWS, 2, -1)
                > UNIFIED_EXIT_INFLUENCE_CAT_EPSILON,
                axis=2,
            )
        ))
        row_failures = (
            []
            if math.isfinite(max_delta)
            and max_delta > UNIFIED_EXIT_INFLUENCE_CAT_EPSILON
            and changed_rows >= 1
            else ["categorical counterfactual is dead"]
        )
        categorical_metrics[token] = {
            "decision": "PASS" if not row_failures else "FAIL",
            "failures": row_failures,
            "counterfactual": "next_valid_category_on_exact_owner_manifold",
            "max_abs_exit_margin_delta": max_delta,
            "changed_rows": changed_rows,
            "total_rows": UNIFIED_EXIT_INFLUENCE_SAMPLE_COUNT,
        }
        failures.extend(f"categorical/{token}: {item}" for item in row_failures)

    side_deltas = (
        (baseline_margin[:, 0] - baseline_margin[:, 1])
        .abs().detach().cpu().double().numpy()
    )
    side_max_delta = float(side_deltas.max())
    side_changed_rows = 2 * int(np.count_nonzero(
        np.any(side_deltas > UNIFIED_EXIT_INFLUENCE_CAT_EPSILON, axis=1)
    ))
    side_failures = (
        []
        if math.isfinite(side_max_delta)
        and side_max_delta > UNIFIED_EXIT_INFLUENCE_CAT_EPSILON
        and side_changed_rows >= 1
        else ["learned side-axis interaction is dead"]
    )
    failures.extend(f"structural/exit_side_axis: {item}" for item in side_failures)
    report = {
        "schema_version": UNIFIED_EXIT_INPUT_INFLUENCE_SCHEMA_VERSION,
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "split": UNIFIED_EXIT_INFLUENCE_SPLIT,
        "sample_count": UNIFIED_EXIT_INFLUENCE_SAMPLE_COUNT,
        "side_rows": {
            "long": UNIFIED_EXIT_INFLUENCE_SIDE_ROWS,
            "short": UNIFIED_EXIT_INFLUENCE_SIDE_ROWS,
        },
        "sampling_contract": UNIFIED_EXIT_INFLUENCE_SAMPLING_CONTRACT,
        "comparison_surface": UNIFIED_EXIT_INFLUENCE_SURFACE,
        "numeric_gradient_epsilon": UNIFIED_EXIT_INFLUENCE_GRAD_EPSILON,
        "categorical_delta_epsilon": UNIFIED_EXIT_INFLUENCE_CAT_EPSILON,
        "sample_entry_row_indices": entry_rows,
        "sample_decision_times_ns": decision_times_ns,
        "ordered_signal_names": signal_names,
        "signal_names_sha256": unified_exit_influence_sha256(signal_names),
        "input_ownership": ownership,
        "input_ownership_sha256": unified_exit_influence_sha256(ownership),
        "numeric_input_count": sum(
            len(row["tokens"]) for row in ownership["numeric"].values()
        ),
        "categorical_input_count": len(ownership["categorical"]),
        "numeric": numeric_metrics,
        "categorical": categorical_metrics,
        "structural": {
            "exit_side_axis": {
                "decision": "PASS" if not side_failures else "FAIL",
                "failures": side_failures,
                "counterfactual": (
                    "same_market_token_and_path_compare_both_side_axes"
                ),
                "max_abs_exit_margin_delta": side_max_delta,
                "changed_rows": side_changed_rows,
                "total_rows": UNIFIED_EXIT_INFLUENCE_SAMPLE_COUNT,
            }
        },
    }
    require_unified_exit_input_influence(
        report,
        ordered_signal_names=signal_names,
        context="UNIFIED_EXIT_SELECTED_CHECKPOINT",
    )
    return report


_UNIFIED_EXIT_FULL_TRAJECTORY_VALIDATION_SCHEMA_VERSION = (
    "gx1_unified_exit_full_trajectory_validation_v6"
)


def _new_unified_exit_full_trajectory_accumulator(
    *,
    model: nn.Module,
    target_model: nn.Module,
) -> dict[str, Any]:
    """Allocate report-only state for the already-required VAL Exit pass.

    A candidate checkpoint used to perform the complete VAL scan in ``validate``
    and then repeat the same online/target Exit forwards solely to create this
    report.  The accumulator records that report from the first scan and binds
    it to both immutable parameter states.  It never supplies model inputs,
    targets, gradients, or checkpoint scores.
    """

    return {
        "online_model_state_sha256": _model_state_sha256(model),
        "target_model_state_sha256": _model_state_sha256(target_model),
        "entry_rows_scanned": 0,
        "eligible_entry_rows": 0,
        "population_rows": 0,
        "q_valid_cells": 0,
        "target_equivalent_action_rows": 0,
        "predicted_tied_rows": 0,
        "target_tied_prediction_unique_rows": 0,
        "unique_target_action_agreement_rows": 0,
        "long_population_rows": 0,
        "short_population_rows": 0,
        "loss_sum": 0.0,
        "learned_realized": [],
        "immediate_realized": [],
        "terminal_realized": [],
        "learned_exit_states": [],
        "state_stream": hashlib.sha256(),
    }


def _accumulate_unified_exit_full_trajectory(
    accumulator: dict[str, Any],
    *,
    raw_entry_indices: Sequence[int],
    selected_positions: Sequence[int],
    episodes: Sequence[Mapping[str, Any]],
    q_values: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Record one validated batch without another model/target forward."""

    if (
        len(selected_positions) != len(episodes)
        or len(set(int(value) for value in selected_positions))
        != len(selected_positions)
        or any(
            int(value) < 0 or int(value) >= len(raw_entry_indices)
            for value in selected_positions
        )
        or q_values.shape != targets.shape
        or valid.shape != q_values.shape
        or tuple(q_values.shape[1:])
        != (2, UNIFIED_EXIT_MAX_PATH_BARS, 2)
    ):
        raise RuntimeError("UNIFIED_EXIT_FULL_VAL_ACCUMULATOR_INPUT_INVALID")
    if int(q_values.shape[0]) != len(episodes):
        raise RuntimeError("UNIFIED_EXIT_FULL_VAL_ACCUMULATOR_BATCH_INVALID")

    flat_q = q_values.reshape(-1, 2)
    flat_target = targets.reshape(-1, 2)
    flat_valid = valid.reshape(-1, 2)
    target_value = flat_target.masked_fill(~flat_valid, -torch.inf).amax(
        dim=1, keepdim=True
    )
    equivalence = (flat_target == target_value) & flat_valid
    target_tie = equivalence.all(dim=1)
    prediction_tie = flat_valid.all(dim=1) & (flat_q[:, 0] == flat_q[:, 1])
    prediction = torch.argmax(
        flat_q.masked_fill(~flat_valid, -torch.inf), dim=1
    )
    accumulator["eligible_entry_rows"] += len(episodes)
    accumulator["population_rows"] += int(flat_q.shape[0])
    accumulator["q_valid_cells"] += int(flat_valid.sum().item())
    accumulator["target_equivalent_action_rows"] += int(target_tie.sum().item())
    accumulator["predicted_tied_rows"] += int(prediction_tie.sum().item())
    accumulator["target_tied_prediction_unique_rows"] += int(
        (target_tie & ~prediction_tie).sum().item()
    )
    accumulator["unique_target_action_agreement_rows"] += int(
        (
            equivalence.gather(1, prediction[:, None]).squeeze(1)
            & ~target_tie
            & ~prediction_tie
        ).sum().item()
    )
    accumulator["loss_sum"] += float(
        nn.functional.mse_loss(
            q_values[valid], targets[valid], reduction="sum"
        )
        .detach()
        .cpu()
        .item()
    )
    q_np = q_values.detach().cpu().double().numpy()
    valid_np = valid.detach().cpu().numpy()
    for episode_position, (entry_position, episode) in enumerate(
        zip(selected_positions, episodes, strict=True)
    ):
        entry_index = int(raw_entry_indices[int(entry_position)])
        rewards = np.asarray(episode["exit_now_reward_bps"], dtype=np.float64)
        for side_index, side_name in enumerate(("long", "short")):
            accumulator[f"{side_name}_population_rows"] += (
                UNIFIED_EXIT_MAX_PATH_BARS
            )
            replay = replay_unified_exit_fitted_q_policy(
                predicted_q_bps=q_np[episode_position, side_index],
                action_valid_mask=valid_np[episode_position, side_index],
                exit_now_reward_bps=rewards[side_index],
            )
            accumulator["learned_realized"].append(
                float(replay["realized_executable_pnl_bps"])
            )
            accumulator["learned_exit_states"].append(
                int(replay["exit_state_index"])
            )
            accumulator["immediate_realized"].append(
                float(rewards[side_index, 0])
            )
            accumulator["terminal_realized"].append(
                float(rewards[side_index, -1])
            )
        stream_rows = np.column_stack((
            np.repeat(entry_index, 2 * UNIFIED_EXIT_MAX_PATH_BARS),
            np.repeat(np.arange(2), UNIFIED_EXIT_MAX_PATH_BARS),
            np.tile(np.arange(UNIFIED_EXIT_MAX_PATH_BARS), 2),
            q_np[episode_position, ..., 0].reshape(-1),
            q_np[episode_position, ..., 1].reshape(-1),
        ))
        accumulator["state_stream"].update(
            np.ascontiguousarray(stream_rows).tobytes()
        )


def _finalize_unified_exit_full_trajectory_validation(
    accumulator: Mapping[str, Any],
    *,
    dataset: "EntryV10CtxDataset",
    exit_gate_stats: Mapping[str, Any],
    exit_gate_failures: Sequence[str],
) -> dict[str, Any]:
    """Fail closed while converting the first-pass accumulator to evidence."""

    if exit_gate_failures:
        raise RuntimeError(
            "[UNIFIED_EXIT_FULL_VAL_GATE_HEALTH_INVALID] "
            + "; ".join(exit_gate_failures)
        )
    required = {
        "online_model_state_sha256",
        "target_model_state_sha256",
        "entry_rows_scanned",
        "eligible_entry_rows",
        "population_rows",
        "q_valid_cells",
        "target_equivalent_action_rows",
        "predicted_tied_rows",
        "target_tied_prediction_unique_rows",
        "unique_target_action_agreement_rows",
        "long_population_rows",
        "short_population_rows",
        "loss_sum",
        "learned_realized",
        "immediate_realized",
        "terminal_realized",
        "learned_exit_states",
        "state_stream",
    }
    if set(accumulator) != required:
        raise RuntimeError("UNIFIED_EXIT_FULL_VAL_ACCUMULATOR_SCHEMA_INVALID")
    eligible_entry_rows = int(accumulator["eligible_entry_rows"])
    population_rows = int(accumulator["population_rows"])
    valid_cells = int(accumulator["q_valid_cells"])
    equivalent_rows = int(accumulator["target_equivalent_action_rows"])
    predicted_tied_rows = int(accumulator["predicted_tied_rows"])
    expected_population = eligible_entry_rows * 2 * UNIFIED_EXIT_MAX_PATH_BARS
    expected_valid = expected_population * 2 - eligible_entry_rows * 2
    learned_realized = accumulator["learned_realized"]
    immediate_realized = accumulator["immediate_realized"]
    terminal_realized = accumulator["terminal_realized"]
    learned_exit_states = accumulator["learned_exit_states"]
    loss_sum = float(accumulator["loss_sum"])
    if (
        population_rows != expected_population
        or valid_cells != expected_valid
        or valid_cells <= 0
        or int(accumulator["long_population_rows"]) <= 0
        or int(accumulator["short_population_rows"]) <= 0
        or int(accumulator["entry_rows_scanned"]) != len(dataset)
        or not math.isfinite(loss_sum)
        or loss_sum < 0.0
        or predicted_tied_rows != 0
        or not all(
            isinstance(values, list) and len(values) == eligible_entry_rows * 2
            for values in (
                learned_realized,
                immediate_realized,
                terminal_realized,
                learned_exit_states,
            )
        )
    ):
        raise RuntimeError(
            "UNIFIED_EXIT_FULL_VAL_POPULATION_INVALID: "
            f"population={population_rows}/{expected_population} "
            f"q_valid={valid_cells}/{expected_valid} "
            f"sides={accumulator['long_population_rows']}/"
            f"{accumulator['short_population_rows']} "
            f"predicted_ties={predicted_tied_rows} "
            f"entries={accumulator['entry_rows_scanned']}/{len(dataset)}"
        )
    state_stream = accumulator["state_stream"]
    if not hasattr(state_stream, "hexdigest"):
        raise RuntimeError("UNIFIED_EXIT_FULL_VAL_STREAM_INVALID")
    return {
        "schema_version": _UNIFIED_EXIT_FULL_TRAJECTORY_VALIDATION_SCHEMA_VERSION,
        "decision": "PASS",
        "population": "all_causal_states_both_sides_batched_episode_forward",
        "entry_rows_scanned": int(accumulator["entry_rows_scanned"]),
        "eligible_entry_rows": eligible_entry_rows,
        "population_rows": population_rows,
        "q_valid_cells": valid_cells,
        "target_equivalent_action_rows": equivalent_rows,
        "target_unique_action_rows": population_rows - equivalent_rows,
        "predicted_tied_rows": predicted_tied_rows,
        "target_tied_prediction_unique_rows": int(
            accumulator["target_tied_prediction_unique_rows"]
        ),
        "target_tied_prediction_unique_fraction": (
            int(accumulator["target_tied_prediction_unique_rows"])
            / max(1, equivalent_rows)
        ),
        "long_population_rows": int(accumulator["long_population_rows"]),
        "short_population_rows": int(accumulator["short_population_rows"]),
        "fitted_q_bellman_mse_mean": loss_sum / valid_cells,
        "unique_target_action_agreement": int(
            accumulator["unique_target_action_agreement_rows"]
        )
        / max(1, population_rows - equivalent_rows),
        "learned_policy_mean_realized_executable_pnl_bps": float(
            np.mean(learned_realized)
        ),
        "immediate_exit_mean_realized_executable_pnl_bps": float(
            np.mean(immediate_realized)
        ),
        "terminal_exit_mean_realized_executable_pnl_bps": float(
            np.mean(terminal_realized)
        ),
        "learned_mean_exit_state_index": float(np.mean(learned_exit_states)),
        "state_prediction_stream_sha256": state_stream.hexdigest(),
        "online_model_state_sha256": accumulator["online_model_state_sha256"],
        "target_model_state_sha256": accumulator["target_model_state_sha256"],
        "future_outcomes_used_as_model_inputs": False,
        "predicted_exact_q_tie_runtime_policy": "fail_closed",
        "gamma": 1.0,
        "intermediate_hold_reward_bps": 0.0,
        **dict(exit_gate_stats),
    }


def _train_rss_gib() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return float(ln.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return -1.0
    return -1.0


def train_epoch(
    model,
    target_model,
    loader,
    optimizer,
    device,
    grad_accum_steps: int,
    task_supervision_observed: dict[str, bool],
    task_gradient_observed: dict[str, bool],
    weight_ema: Optional["_WeightEma"] = None,
    attended_batch_offset: int = 0,
    attended_max_optimizer_steps: Optional[int] = None,
    attended_checkpoint_hook: Optional[Any] = None,
    attended_exit_action_forward_chunk_rows: Optional[int] = None,
) -> tuple[float, dict[str, Any], bool]:
    model.train()
    target_model.eval()
    if any(parameter.requires_grad for parameter in target_model.parameters()):
        raise RuntimeError("[UNIFIED_EXIT_TARGET_MODEL_NOT_FROZEN]")
    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, EntryV10CtxDataset):
        raise RuntimeError("[UNIFIED_EXIT_TRAIN_DATASET_INVALID]")
    _accum_steps = int(grad_accum_steps)
    if _accum_steps < 1:
        raise RuntimeError(
            f"[ENTRY_GRAD_ACCUM_STEPS_INVALID] observed={_accum_steps} expected>=1"
        )
    if _accum_steps > 1:
        log.info("[GRAD_ACCUM] accumulating gradients over %d batches per optimizer step", _accum_steps)
    if attended_max_optimizer_steps is not None:
        if (
            int(attended_max_optimizer_steps) < 1
            or attended_checkpoint_hook is None
            or _accum_steps != 1
            or int(attended_batch_offset) < 0
            or attended_exit_action_forward_chunk_rows is None
            or int(attended_exit_action_forward_chunk_rows) < 1
        ):
            raise RuntimeError("[ATTENDED_RESEARCH_TRAIN_EPOCH_ARGUMENT_INVALID]")
    _accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    entry_q_loss_sum = 0.0
    cooperation_gate_epoch = _new_cooperation_gate_epoch_accumulator()
    feature_tf_gate_epoch = _new_feature_tf_gate_epoch_accumulator()
    exit_cooperation_gate_epoch = _new_cooperation_gate_epoch_accumulator(
        _UNIFIED_EXIT_COOPERATION_GATE_WIDTHS
    )
    exit_feature_tf_gate_epoch = _new_feature_tf_gate_epoch_accumulator(
        _UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE
    )
    n = 0
    side_mae_loss_sum = 0.0
    trendline_event_loss_sum = 0.0
    trendline_event_rows_sum = 0
    trendline_support_rows_sum = 0
    trendline_resistance_rows_sum = 0
    unified_exit_loss_sum = 0.0
    unified_exit_population_rows = 0
    unified_exit_rows = 0
    unified_exit_tied_rows = 0
    unified_exit_eligible_entry_rows = 0
    unified_exit_hold_rows = 0
    unified_exit_now_rows = 0
    unified_exit_correct = 0

    log.info("[TRAIN_RSS] epoch_start rss_gib=%.2f", _train_rss_gib())
    _first_batch_logged = False
    _batch_i = 0
    _optimizer_steps_this_call = 0
    for batch in loader:
        _batch_i += 1
        _absolute_batch_i = int(attended_batch_offset) + _batch_i
        log.info("[TRAIN_STEP] batch=%d begin rss_gib=%.2f", _absolute_batch_i, _train_rss_gib())
        if not _first_batch_logged:
            log.info("[TRAIN_RSS] first_batch_fetched rss_gib=%.2f", _train_rss_gib())
        non_blocking = device.type == "cuda"
        seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
        snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
        ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
        ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
        batch_rows = int(seq_x.shape[0])
        # Grad accum: zero_grad happens AFTER step (or at start of epoch).
        # See loss.backward() / optimizer.step() block below for the gated step.
        _profile_timing = not _first_batch_logged
        _profile_batch_start = (
            _synchronized_exit_profile_clock(device) if _profile_timing else None
        )
        if _profile_timing and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if not _first_batch_logged:
            log.info("[TRAIN_RSS] before_forward rss_gib=%.2f", _train_rss_gib())
        out = _model_forward_fp32(
            model,
            seq_x,
            snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
            **_multi_tf_kwargs_from_batch(batch, seq_x.device),
        )
        _profile_entry_online_forward = (
            _synchronized_exit_profile_clock(device) if _profile_timing else None
        )
        with torch.no_grad():
            target_out = _model_forward_fp32(
                target_model,
                seq_x,
                snap_x,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
                **_multi_tf_kwargs_from_batch(batch, seq_x.device),
            )
        _profile_entry_target_forward = (
            _synchronized_exit_profile_clock(device) if _profile_timing else None
        )
        if not _first_batch_logged:
            log.info("[TRAIN_RSS] before_exit_loss rss_gib=%.2f", _train_rss_gib())
        entry_representations = out.get(UNIFIED_EXIT_MODEL_REPRESENTATION_KEY)
        target_entry_representations = target_out.get(
            UNIFIED_EXIT_MODEL_REPRESENTATION_KEY
        )
        entry_row_indices = batch.get("entry_row_index")
        if (
            not isinstance(entry_representations, torch.Tensor)
            or not isinstance(target_entry_representations, torch.Tensor)
            or not isinstance(entry_row_indices, torch.Tensor)
        ):
            raise RuntimeError("[UNIFIED_EXIT_TRAIN_ENTRY_EVIDENCE_MISSING]")
        (
            exit_entry_gradients,
            unified_exit_stats,
            entry_action_q_targets,
            entry_action_q_valid,
        ) = (
            _train_unified_exit_full_population(
                model=model,
                target_model=target_model,
                entry_decision_representations=entry_representations,
                target_entry_decision_representations=(
                    target_entry_representations
                ),
                entry_row_indices=entry_row_indices,
                dataset=dataset,
                device=device,
                grad_accum_steps=_accum_steps,
                exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
                exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
                profile_timing=_profile_timing,
                exit_action_forward_chunk_rows=(
                    attended_exit_action_forward_chunk_rows
                    if attended_max_optimizer_steps is not None
                    else None
                ),
            )
        )
        _profile_exit_train = (
            _synchronized_exit_profile_clock(device) if _profile_timing else None
        )
        unified_exit_loss = torch.tensor(
            float(unified_exit_stats["raw_loss"]),
            device=device,
            dtype=entry_representations.dtype,
        )
        entry_action_q_bps = out["entry_action_q_bps"]
        if (
            not isinstance(entry_action_q_bps, torch.Tensor)
            or tuple(entry_action_q_bps.shape)
            != tuple(entry_action_q_targets.shape)
            or not bool(entry_action_q_valid[:, 2].all().item())
        ):
            raise RuntimeError("[ENTRY_FITTED_Q_PREDICTION_SHAPE_INVALID]")
        _accumulate_cooperation_gate_epoch(cooperation_gate_epoch, out)
        _accumulate_feature_tf_gate_epoch(feature_tf_gate_epoch, out)
        entry_action_q_loss = nn.functional.mse_loss(
            entry_action_q_bps[entry_action_q_valid],
            entry_action_q_targets[entry_action_q_valid],
        )
        if not _first_batch_logged:
            log.info("[TRAIN_RSS] after_forward_losses rss_gib=%.2f", _train_rss_gib())
            _first_batch_logged = True
        task_losses: dict[str, torch.Tensor] = {
            "entry_action_q": entry_action_q_loss
        }
        side_mae_loss, side_mae_stats = _side_mae_auxiliary_loss(
            out,
            batch,
            device,
        )
        task_losses["side_mae_bps"] = side_mae_loss
        trendline_event_loss, trendline_stats = _trendline_event_aux_loss(
            out, batch, device
        )
        task_losses["trendline_event"] = trendline_event_loss
        position_size_logit = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="position_size_logit",
            target_names=("y_position_size_target", "y_position_size_mask"),
        )
        y_pos_size = batch["y_position_size_target"].to(device, non_blocking=non_blocking)
        y_pos_size_mask = batch["y_position_size_mask"].to(
            device, non_blocking=non_blocking
        )
        if bool((y_pos_size_mask.reshape(-1) == 1.0).any()):
            task_losses["position_size"] = _masked_position_size_mse(
                position_size_logit,
                y_pos_size,
                y_pos_size_mask,
            )

        task_losses.update(dip_forecast_task_losses(out, batch, device))
        for task_name in task_losses:
            task_supervision_observed[task_name] = True
        loss, _joint_task_stats = _joint_task_loss(model, task_losses)
        exit_supervised = int(unified_exit_stats["q_valid_cells"]) > 0
        if exit_supervised:
            task_supervision_observed["unified_exit_action"] = True
            exit_log_variance = model.task_log_variances[
                "unified_exit_action"
            ]
            # The precision*Q-MSE gradient was already streamed per chunk. Add
            # its detached value for exact reporting and +s once for this
            # genuinely supervised batch.
            loss = (
                loss
                + torch.exp(-exit_log_variance.detach())
                * unified_exit_loss.detach()
                + exit_log_variance
            )

        masked_entry_q = entry_action_q_bps.masked_fill(
            ~entry_action_q_valid, -torch.inf
        )
        winner_count = masked_entry_q.eq(
            masked_entry_q.amax(dim=1, keepdim=True)
        ).sum(dim=1)
        if bool((winner_count != 1).any().item()):
            raise RuntimeError("[ENTRY_FITTED_Q_TRAIN_PREDICTED_TIE]")
        # Grad accumulation: scale loss down by accum_steps so .backward() sums to
        # the same magnitude as a single big-batch step. Only step + zero every Nth batch.
        log.info("[TRAIN_STEP] batch=%d loss_ready rss_gib=%.2f", _absolute_batch_i, _train_rss_gib())
        scaled_main_loss = loss / float(_accum_steps)
        if exit_supervised:
            scaled_main_loss = scaled_main_loss + (
                entry_representations * exit_entry_gradients
            ).sum()
        scaled_main_loss.backward()
        _observe_joint_task_weight_gradients(model, task_gradient_observed)
        log.info("[TRAIN_STEP] batch=%d backward_done rss_gib=%.2f", _absolute_batch_i, _train_rss_gib())
        _accum_count += 1
        if _accum_count >= _accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # V30 package 5: the weight EMA advances once per OPTIMIZER step
            # (not per micro-batch), so its decay means the same thing at any
            # accumulation width. None when the recipe decay is the 0.0 OFF
            # sentinel — then nothing here executes at all.
            if weight_ema is not None:
                weight_ema.update(model)
            _accum_count = 0
            _optimizer_steps_this_call += 1
            log.info("[TRAIN_STEP] batch=%d step_done", _absolute_batch_i)
            if attended_checkpoint_hook is not None:
                _is_final_batch = _batch_i == len(loader)
                attended_checkpoint_hook(
                    next_batch_offset=_absolute_batch_i,
                    complete_epoch=_is_final_batch,
                )
            if (
                attended_max_optimizer_steps is not None
                and _optimizer_steps_this_call >= int(attended_max_optimizer_steps)
                and _batch_i < len(loader)
            ):
                log.info(
                    "[ATTENDED_RESEARCH_SESSION_PAUSE] batches_completed=%d "
                    "optimizer_steps_this_session=%d max_optimizer_steps=%d",
                    _absolute_batch_i,
                    _optimizer_steps_this_call,
                    int(attended_max_optimizer_steps),
                )
                return (
                    total / max(1, n),
                    {
                        "partial": True,
                        "batches_completed": _absolute_batch_i,
                        "optimizer_steps_this_session": _optimizer_steps_this_call,
                    },
                    False,
                )
        if _profile_timing:
            _profile_end = _synchronized_exit_profile_clock(device)
            assert (
                _profile_batch_start is not None
                and _profile_entry_online_forward is not None
                and _profile_entry_target_forward is not None
                and _profile_exit_train is not None
            )
            _peak_cuda_mib = (
                int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
                if device.type == "cuda"
                else 0
            )
            log.info(
                "[TRAIN_PROFILE] batch=%d entry_online_forward_s=%.6f "
                "entry_target_forward_s=%.6f exit_train_s=%.6f "
                "post_exit_backward_s=%.6f total_s=%.6f peak_cuda_mib=%d",
                _absolute_batch_i,
                _profile_entry_online_forward - _profile_batch_start,
                _profile_entry_target_forward - _profile_entry_online_forward,
                _profile_exit_train - _profile_entry_target_forward,
                _profile_end - _profile_exit_train,
                _profile_end - _profile_batch_start,
                _peak_cuda_mib,
            )

        bs = batch_rows
        total += float(loss) * bs
        entry_q_loss_sum += float(entry_action_q_loss) * bs
        side_mae_loss_sum += float(side_mae_stats["side_mae_loss"]) * bs
        trendline_event_loss_sum += float(
            trendline_stats["trendline_event_loss"]
        ) * bs
        trendline_event_rows_sum += int(trendline_stats["trendline_event_rows"])
        trendline_support_rows_sum += int(
            trendline_stats["trendline_support_rows"]
        )
        trendline_resistance_rows_sum += int(
            trendline_stats["trendline_resistance_rows"]
        )
        _exit_rows = int(unified_exit_stats["q_valid_cells"])
        unified_exit_loss_sum += (
            float(unified_exit_loss.detach().cpu().item()) * _exit_rows
        )
        unified_exit_population_rows += int(
            unified_exit_stats["population_rows"]
        )
        unified_exit_rows += _exit_rows
        unified_exit_tied_rows += int(
            unified_exit_stats["target_equivalent_action_rows"]
        )
        unified_exit_eligible_entry_rows += int(
            unified_exit_stats["eligible_entry_rows"]
        )
        unified_exit_hold_rows += int(unified_exit_stats["hold_target_greedy_rows"])
        unified_exit_now_rows += int(
            unified_exit_stats["exit_now_target_greedy_rows"]
        )
        unified_exit_correct += int(
            unified_exit_stats["unique_target_action_agreement_rows"]
        )
        n += bs

    if _accum_count:
        _step_partial_gradient_accumulation(
            model=model,
            optimizer=optimizer,
            configured_steps=_accum_steps,
            observed_steps=_accum_count,
            weight_ema=weight_ema,
        )
    if (
        unified_exit_rows <= 0
        or unified_exit_hold_rows <= 0
        or unified_exit_now_rows <= 0
        or not math.isfinite(unified_exit_loss_sum)
        or unified_exit_loss_sum < 0.0
    ):
        raise RuntimeError(
            "[UNIFIED_EXIT_TRAIN_EPOCH_EVIDENCE_INVALID] "
            f"rows={unified_exit_rows} hold={unified_exit_hold_rows} "
            f"exit_now={unified_exit_now_rows} loss_sum={unified_exit_loss_sum}"
        )

    stats = {
        "entry_action_q_raw_bps_mse_mean": (entry_q_loss_sum / max(1, n)),
        "side_mae_loss_mean": (side_mae_loss_sum / max(1, n)),
        "trendline_event_loss_mean": (trendline_event_loss_sum / max(1, n)),
        "trendline_event_rows": int(trendline_event_rows_sum),
        "trendline_support_rows": int(trendline_support_rows_sum),
        "trendline_resistance_rows": int(trendline_resistance_rows_sum),
        "unified_exit_raw_bps_q_mse_mean": (
            unified_exit_loss_sum / unified_exit_rows
        ),
        "unified_exit_population_rows": int(unified_exit_population_rows),
        "unified_exit_q_valid_cells": int(unified_exit_rows),
        "unified_exit_target_equivalent_action_rows": int(unified_exit_tied_rows),
        "unified_exit_unique_target_rows": int(
            unified_exit_population_rows - unified_exit_tied_rows
        ),
        "unified_exit_eligible_entry_rows": int(
            unified_exit_eligible_entry_rows
        ),
        "unified_exit_hold_target_greedy_rows": int(unified_exit_hold_rows),
        "unified_exit_exit_now_target_greedy_rows": int(unified_exit_now_rows),
        "unified_exit_unique_target_action_agreement": (
            unified_exit_correct
            / max(1, unified_exit_population_rows - unified_exit_tied_rows)
        ),
    }
    stats.update(
        {
            f"joint_task_log_variance_{name}": float(
                model.task_log_variances[name].detach().cpu().item()
            )
            for name in JOINT_TASK_NAMES
        }
    )
    stats.update(_finalize_cooperation_gate_epoch(cooperation_gate_epoch))
    stats.update(_finalize_feature_tf_gate_epoch(feature_tf_gate_epoch))
    exit_gate_stats, _exit_gate_failures = _finalize_unified_exit_gate_epoch(
        exit_cooperation_gate_epoch,
        exit_feature_tf_gate_epoch,
    )
    stats.update(exit_gate_stats)
    return total / max(1, n), stats, True


def _active_head_contract_failures() -> List[str]:
    expected = tuple(MODEL_NATIVE_ACTIVE_HEADS)
    observed = tuple(_ACTIVE_HEAD_OUTPUT_COMPONENTS)
    failures: List[str] = []
    if observed != expected:
        failures.append(
            "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_CONTRACT_MISMATCH] "
            f"expected={expected} observed={observed}"
        )
    target_observed = tuple(_ACTIVE_HEAD_TARGET_COMPONENTS)
    if target_observed != expected:
        failures.append(
            "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_TARGET_CONTRACT_MISMATCH] "
            f"expected={expected} observed={target_observed}"
        )
    return failures


def _active_head_batch_target(
    batch: Dict[str, torch.Tensor],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_TARGET_MISSING] target={name}"
        )
    return value.to(device=device, non_blocking=device.type == "cuda").float()


def _active_head_target_surfaces(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Bind every active head to the exact target semantics used by its loss."""

    contract_failures = _active_head_contract_failures()
    if contract_failures:
        raise RuntimeError("; ".join(contract_failures))

    def _prediction(name: str) -> torch.Tensor:
        value = out.get(name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_OUTPUT_MISSING] output={name}"
            )
        return value.float()

    entry_q_target = out.get("_entry_action_q_target")
    entry_q_valid = out.get("_entry_action_q_valid")
    if (
        not isinstance(entry_q_target, torch.Tensor)
        or not isinstance(entry_q_valid, torch.Tensor)
        or entry_q_valid.dtype != torch.bool
        or tuple(entry_q_target.shape) != tuple(entry_q_valid.shape)
        or entry_q_target.ndim != 2
        or int(entry_q_target.shape[1]) != 3
    ):
        raise RuntimeError("[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_ENTRY_Q_TARGET_INVALID]")
    batch_size = int(entry_q_target.shape[0])
    all_rows = torch.ones(batch_size, dtype=torch.bool, device=device)

    dip_targets: List[torch.Tensor] = []
    for direction in DIP_DIRECTIONS:
        for horizon in DIP_HORIZONS:
            for target_name in DIP_TARGETS:
                column = (
                    f"y_dip_mfe_{direction}_K{horizon}"
                    if target_name.startswith("recovery")
                    else f"y_dip_mae_{direction}_K{horizon}"
                )
                dip_targets.append(_active_head_batch_target(batch, column, device))
    dip_target = torch.stack(dip_targets, dim=1)
    forecast_target = torch.stack(
        [
            _active_head_batch_target(batch, f"y_forecast_ret_K{horizon}", device)
            for horizon in FORECAST_HORIZONS
        ],
        dim=1,
    )
    timing_target = torch.stack(
        [
            _active_head_batch_target(
                batch,
                f"y_{target_name}_{direction}_K{horizon}",
                device,
            )
            for direction in TIMING_DIRECTIONS
            for horizon in TIMING_HORIZONS
            for target_name in TIMING_TARGETS
        ],
        dim=1,
    )
    tail_target = torch.stack(
        [
            _active_head_batch_target(
                batch,
                f"y_tail_mae_{direction}_K{horizon}",
                device,
            )
            for direction in TAIL_RISK_DIRECTIONS
            for horizon in TAIL_RISK_HORIZONS
        ],
        dim=1,
    )
    vol_target = torch.stack(
        [
            _active_head_batch_target(batch, f"y_vol_fwd_K{horizon}", device)
            for horizon in VOL_FORECAST_HORIZONS
        ],
        dim=1,
    )

    side_mae_target = torch.stack(
        [
            _active_head_batch_target(
                batch, "y_long_expected_mae_bps", device
            ).reshape(-1).clamp_min(0.0),
            _active_head_batch_target(
                batch, "y_short_expected_mae_bps", device
            ).reshape(-1).clamp_min(0.0),
        ],
        dim=1,
    )
    trendline_target = torch.stack(
        [
            _active_head_batch_target(
                batch, "y_line_support_touch_held", device
            ),
            _active_head_batch_target(
                batch, "y_line_resistance_touch_held", device
            ),
            _active_head_batch_target(batch, "y_countertrend_short_trap", device),
            _active_head_batch_target(batch, "y_countertrend_long_trap", device),
        ],
        dim=1,
    ).clamp(0.0, 1.0)

    surfaces: Dict[
        str,
        Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ] = {
        "entry_action_q": {
            "entry_action_q_bps": (
                _prediction("entry_action_q_bps"),
                entry_q_target,
                entry_q_valid.any(dim=1),
            )
        },
        "position_size": {
            "position_size_logit": (
                _prediction("position_size_logit"),
                _active_head_batch_target(
                    batch, "y_position_size_target", device
                ).reshape(-1, 1),
                (
                    _active_head_batch_target(
                        batch, "y_position_size_mask", device
                    ).reshape(-1)
                    > 0.5
                ),
            )
        },
        "dip": {"dip_pred": (_prediction("dip_pred"), dip_target, all_rows)},
        "forecast": {
            "forecast_pred": (
                _prediction("forecast_pred"),
                forecast_target,
                all_rows,
            )
        },
        "timing": {
            "timing_pred": (_prediction("timing_pred"), timing_target, all_rows)
        },
        "tail_risk": {
            "tail_risk_pred": (
                _prediction("tail_risk_pred"),
                tail_target,
                all_rows,
            )
        },
        "vol_forecast": {
            "vol_forecast_pred": (
                _prediction("vol_forecast_pred"),
                vol_target,
                all_rows,
            )
        },
        "side_mae": {
            "side_mae_bps": (
                _prediction("side_mae_bps"),
                side_mae_target,
                all_rows,
            ),
        },
        "trendline_event": {
            "trendline_event_logits": (
                _prediction("trendline_event_logits"),
                trendline_target,
                all_rows,
            )
        },
    }
    if tuple(surfaces) != tuple(MODEL_NATIVE_ACTIVE_HEADS):
        raise RuntimeError(
            "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_TARGET_SURFACE_MISMATCH] "
            f"expected={tuple(MODEL_NATIVE_ACTIVE_HEADS)} observed={tuple(surfaces)}"
        )
    return surfaces


def _new_active_head_epoch_accumulator() -> Dict[str, Any]:
    return {
        "heads": {
            head_name: {"components": {}}
            for head_name in MODEL_NATIVE_ACTIVE_HEADS
        }
    }


def _accumulate_active_head_epoch(
    accumulator: Dict[str, Any],
    model: nn.Module,
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    surfaces = _active_head_target_surfaces(
        out,
        batch,
        device,
    )
    head_store = accumulator.get("heads")
    if not isinstance(head_store, dict) or tuple(head_store) != tuple(
        MODEL_NATIVE_ACTIVE_HEADS
    ):
        raise RuntimeError("[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_ACCUMULATOR_INVALID]")

    for head_name, components in surfaces.items():
        for component_name, (prediction, target, mask) in components.items():
            if prediction.ndim != 2 or target.ndim != 2:
                raise RuntimeError(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENT_SHAPE_INVALID] "
                    f"head={head_name} component={component_name} "
                    f"prediction={tuple(prediction.shape)} target={tuple(target.shape)}"
                )
            if prediction.shape != target.shape or mask.ndim != 1:
                raise RuntimeError(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENT_SHAPE_MISMATCH] "
                    f"head={head_name} component={component_name} "
                    f"prediction={tuple(prediction.shape)} target={tuple(target.shape)} "
                    f"mask={tuple(mask.shape)}"
                )
            if int(mask.shape[0]) != int(prediction.shape[0]):
                raise RuntimeError(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_MASK_ROW_MISMATCH] "
                    f"head={head_name} component={component_name}"
                )
            component_store = head_store[head_name]["components"].setdefault(
                component_name,
                {"prediction": [], "target": []},
            )
            component_store["prediction"].append(
                prediction[mask].detach().float().cpu().numpy()
            )
            component_store["target"].append(
                target[mask].detach().float().cpu().numpy()
            )



def _active_head_epoch_diagnostics(
    accumulator: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Finalize epoch-wide target, output-liveness and fusion-influence proof."""
    failures = _active_head_contract_failures()
    head_store = accumulator.get("heads") if isinstance(accumulator, dict) else None
    if not isinstance(head_store, dict):
        return {}, failures + ["[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_EVIDENCE_MISSING]"]

    missing_heads = sorted(set(MODEL_NATIVE_ACTIVE_HEADS) - set(head_store))
    unexpected_heads = sorted(set(head_store) - set(MODEL_NATIVE_ACTIVE_HEADS))
    if missing_heads or unexpected_heads:
        failures.append(
            "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_HEAD_SET_INVALID] "
            f"missing={missing_heads} unexpected={unexpected_heads}"
        )

    head_metrics: Dict[str, Any] = {}
    for head_name in MODEL_NATIVE_ACTIVE_HEADS:
        failure_count_before = len(failures)
        evidence = head_store.get(head_name)
        if not isinstance(evidence, dict):
            failures.append(
                f"[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_HEAD_MISSING] head={head_name}"
            )
            continue
        components = evidence.get("components")
        component_metrics: Dict[str, Any] = {}
        if not isinstance(components, dict) or not components:
            failures.append(
                "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENTS_MISSING] "
                f"head={head_name}"
            )
            components = {}
        expected_components = set(_ACTIVE_HEAD_TARGET_COMPONENTS[head_name])
        missing_components = sorted(expected_components - set(components))
        unexpected_components = sorted(set(components) - expected_components)
        if missing_components or unexpected_components:
            failures.append(
                "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENT_SET_INVALID] "
                f"head={head_name} missing={missing_components} "
                f"unexpected={unexpected_components}"
            )
        for component_name, component in components.items():
            prediction_chunks = (
                component.get("prediction") if isinstance(component, dict) else None
            )
            target_chunks = (
                component.get("target") if isinstance(component, dict) else None
            )
            if not prediction_chunks or not target_chunks:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENT_EVIDENCE_MISSING] "
                    f"head={head_name} component={component_name}"
                )
                continue
            try:
                prediction = np.concatenate(
                    [np.asarray(value, dtype=np.float64) for value in prediction_chunks],
                    axis=0,
                )
                target = np.concatenate(
                    [np.asarray(value, dtype=np.float64) for value in target_chunks],
                    axis=0,
                )
            except Exception as exc:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENT_EVIDENCE_INVALID] "
                    f"head={head_name} component={component_name} error={exc}"
                )
                continue
            if prediction.ndim != 2 or target.shape != prediction.shape:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENT_SHAPE_INVALID] "
                    f"head={head_name} component={component_name} "
                    f"prediction={prediction.shape} target={target.shape}"
                )
                continue
            expected_width = int(_ACTIVE_HEAD_COMPONENT_WIDTHS[component_name])
            if int(prediction.shape[1]) != expected_width:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_COMPONENT_WIDTH_INVALID] "
                    f"head={head_name} component={component_name} "
                    f"observed={int(prediction.shape[1])} expected={expected_width}"
                )
                continue
            rows = int(prediction.shape[0])
            if rows < _ACTIVE_HEAD_DIAGNOSTIC_MIN_ROWS:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_ROWS_INSUFFICIENT] "
                    f"head={head_name} component={component_name} rows={rows}"
                )
                continue
            if not np.isfinite(prediction).all() or not np.isfinite(target).all():
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_NONFINITE] "
                    f"head={head_name} component={component_name}"
                )
                continue
            prediction_range = np.ptp(prediction, axis=0)
            target_range = np.ptp(target, axis=0)
            prediction_std = np.std(prediction, axis=0)
            target_std = np.std(target, axis=0)
            dead_prediction_columns = np.flatnonzero(
                prediction_range <= _ACTIVE_HEAD_DIAGNOSTIC_LIVENESS_EPS
            ).astype(int).tolist()
            dead_target_columns = np.flatnonzero(
                target_range <= _ACTIVE_HEAD_DIAGNOSTIC_LIVENESS_EPS
            ).astype(int).tolist()
            structural_constant_columns = set(
                _ACTIVE_HEAD_STRUCTURAL_CONSTANT_COLUMNS.get(
                    component_name,
                    frozenset(),
                )
            )
            blocking_dead_prediction_columns = sorted(
                set(dead_prediction_columns) - structural_constant_columns
            )
            blocking_dead_target_columns = sorted(
                set(dead_target_columns) - structural_constant_columns
            )
            if blocking_dead_prediction_columns:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_OUTPUT_DEAD] "
                    f"head={head_name} component={component_name} "
                    f"columns={blocking_dead_prediction_columns}"
                )
            target_dead = bool(blocking_dead_target_columns)
            if component_name in _ACTIVE_HEAD_DERIVED_TARGET_COMPONENTS:
                target_dead = len(blocking_dead_target_columns) == int(
                    target.shape[1]
                )
            if target_dead:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_TARGET_DEAD] "
                    f"head={head_name} component={component_name} "
                    f"columns={blocking_dead_target_columns}"
                )
            component_metrics[component_name] = {
                "rows": rows,
                "width": int(prediction.shape[1]),
                "prediction_min_range": float(np.min(prediction_range)),
                "prediction_min_std": float(np.min(prediction_std)),
                "target_min_range": float(np.min(target_range)),
                "target_min_std": float(np.min(target_std)),
                "prediction_dead_columns": dead_prediction_columns,
                "target_dead_columns": dead_target_columns,
                "structural_constant_columns": sorted(
                    structural_constant_columns
                ),
            }

        head_metrics[head_name] = {
            "ok": len(failures) == failure_count_before,
            "components": component_metrics,
            "entry_action_authority": (
                "representation_only"
                if head_name in _ACTIVE_HEAD_ACTION_AUTHORITY_NONE
                else "sole_raw_bps_entry_q"
            ),
        }

    metrics = {
        "active_head_diagnostic_schema": (
            "entry_model_native_active_head_epoch_diagnostics_v4"
        ),
        "active_head_contract": list(MODEL_NATIVE_ACTIVE_HEADS),
        "active_head_diagnostics": head_metrics,
        "active_head_health_ok": not failures,
        "active_head_health_failures": list(failures),
    }
    return metrics, failures


def _require_nonnegative_target(values: torch.Tensor, *, name: str) -> torch.Tensor:
    """Fail closed on a target the contract declares non-negative.

    Rule 16 makes MAE a non-negative adverse magnitude. Silently clamping a
    negative value rewrites the target and hides a producer defect: that exact
    absorber is why V24's signed-target corruption survived until a post-hoc
    audit. A violation is a corrupt dataset, not something to repair here.
    """

    if bool(torch.isnan(values).any()):
        raise RuntimeError(f"[ENTRY_TARGET_NONFINITE] {name}")
    minimum = float(values.min())
    if minimum < 0.0:
        raise RuntimeError(
            f"[ENTRY_TARGET_NEGATIVE_MAGNITUDE] {name} min={minimum:.9g}; "
            "rebuild the dataset — the target contract forbids rewriting it"
        )
    return values


def _checkpoint_admission_ok(
    *,
    profile: str,
    active_head_health_ok: bool,
    cooperation_gate_health_ok: bool,
    exit_cooperation_gate_health_ok: bool,
) -> bool:
    """Decide checkpoint admission for the exact training profile.

    Profile-separated admission (user vedtak 2026-07-25).

    ``candidate`` requires exact active-head, Entry cooperation and independent
    five-TF Exit cooperation health. Retired threshold/composite heads have no
    parallel health gate.

    ``smoke`` answers the trainability question it is named for — does this
    recipe train at all and does the raw-Q authority remain live — so it admits
    on active-head liveness.
    Cooperation health is still computed and journaled as diagnostics; it does
    not veto a smoke checkpoint. A smoke
    bundle carries zero edge, promotion or launch authority: the smoke bundle
    audit, candidate readiness, serve parity, sizing, Exit and launch
    finalizer contracts are unchanged and still require the full evidence set.
    """

    if profile == "candidate":
        return bool(
            active_head_health_ok
            and cooperation_gate_health_ok
            and exit_cooperation_gate_health_ok
        )
    if profile == "smoke":
        return bool(active_head_health_ok)
    raise RuntimeError(f"[ENTRY_TRAIN_PROFILE_INVALID] {profile!r}")


def _train_json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def _resolve_train_out_bundle_dir(out_bundle_dir: Path, gx1_data_override: str) -> Path:
    path = Path(out_bundle_dir).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_resolve_gx1_data(gx1_data_override) / path).resolve()


def _attended_research_session_contract(
    *,
    out_bundle_dir: Path,
    run_id: str,
    dataset_run_id: str,
    train_parquet: Path,
    val_parquet: Path,
    m5_prebuilt_path: Path,
    lifecycle_manifest_path: Path,
    input_normalization: Mapping[str, Any],
    seed: int,
    batch_size: int,
    epochs: int,
    grad_accum_steps: int,
    subsample_rows: int,
    lr: float,
    dropout: float,
    execution_tier: str,
    device_type: str,
    max_optimizer_steps: int,
) -> dict[str, Any]:
    """Bind an attended session to one exact, non-promotable train surface."""

    if int(epochs) != 1 or int(grad_accum_steps) != 1:
        raise RuntimeError(
            "[ATTENDED_RESEARCH_TRAIN_BUDGET_INVALID] attended research "
            "requires exactly one epoch and grad_accum_steps=1"
        )
    if not _is_attended_execution_tier(execution_tier):
        raise RuntimeError("[ATTENDED_RESEARCH_EXECUTION_TIER_INVALID]")
    if device_type not in ("cpu", "cuda"):
        raise RuntimeError("[ATTENDED_RESEARCH_DEVICE_INVALID]")
    if int(max_optimizer_steps) < 1:
        raise RuntimeError("[ATTENDED_RESEARCH_MAX_STEPS_INVALID]")
    normalized = dict(input_normalization)
    normalization_sha256 = normalized.get("contract_sha256")
    if not isinstance(normalization_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", normalization_sha256
    ):
        raise RuntimeError("[ATTENDED_RESEARCH_NORMALIZATION_BINDING_INVALID]")
    artifact_paths = {
        "train_parquet": Path(train_parquet).resolve(strict=True),
        "val_parquet": Path(val_parquet).resolve(strict=True),
        "m5_prebuilt_path": Path(m5_prebuilt_path).resolve(strict=True),
        "unified_exit_lifecycle_manifest": Path(lifecycle_manifest_path).resolve(
            strict=True
        ),
    }
    if any(path.is_symlink() or not path.is_file() for path in artifact_paths.values()):
        raise RuntimeError("[ATTENDED_RESEARCH_ARTIFACT_PATH_INVALID]")
    return {
        "schema_version": _ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION,
        "authority": {
            "research_trainability_only": True,
            "candidate": False,
            "validation": False,
            "test": False,
            "bundle": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "source_commit": _git_commit(),
        "out_bundle_dir": str(Path(out_bundle_dir).expanduser().resolve()),
        "run_id": str(run_id),
        "dataset_run_id": str(dataset_run_id),
        "profile": "smoke",
        "execution_tier": str(execution_tier),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "input_normalization_sha256": normalization_sha256,
        "training": {
            "seed": int(seed),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "grad_accum_steps": int(grad_accum_steps),
            "subsample_rows": int(subsample_rows),
            "learning_rate": float(lr),
            "dropout": float(dropout),
            "device": str(device_type),
            "max_optimizer_steps_per_session": int(max_optimizer_steps),
            "unified_exit_action_forward_chunk_rows": (
                _ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS
            ),
            "precision": "deterministic_fp32",
            "compile": False,
            "tf32": False,
            "autocast": False,
        },
    }


def _new_attended_research_epoch_order(dataset_rows: int) -> torch.Tensor:
    if int(dataset_rows) < 1:
        raise RuntimeError("[ATTENDED_RESEARCH_DATASET_ROWS_INVALID]")
    # Match the trainer's ordinary shuffle source (the current torch RNG), but
    # persist its exact result before a resume can be attempted.  The resumed
    # process therefore consumes no fresh sampling RNG and cannot silently
    # replace a remaining slice with a newly shuffled one.
    order = torch.randperm(int(dataset_rows), dtype=torch.int64)
    if (
        order.ndim != 1
        or int(order.numel()) != int(dataset_rows)
        or not torch.equal(torch.sort(order).values, torch.arange(int(dataset_rows)))
    ):
        raise RuntimeError("[ATTENDED_RESEARCH_ORDER_CONSTRUCTION_INVALID]")
    return order.cpu()


def _restore_attended_research_checkpoint(
    state: Mapping[str, Any],
    *,
    session: _AttendedResearchSession,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: optim.Optimizer,
    weight_ema: Optional[_WeightEma],
    lr_scheduler: Optional[optim.lr_scheduler.LRScheduler],
    device: torch.device,
    dataset_rows: int,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "session_contract_sha256",
        "checkpoint_index",
        "complete_optimizer_steps",
        "epoch_index",
        "next_batch_offset",
        "epoch_order",
        "model_state",
        "target_model_state",
        "optimizer_state",
        "weight_ema_state",
        "lr_scheduler_state",
        "rng_state",
        "complete",
    }
    if (
        set(state) != expected_keys
        or state.get("schema_version") != _ATTENDED_RESEARCH_SESSION_SCHEMA_VERSION
        or state.get("session_contract_sha256") != session.contract_sha256
        or int(state.get("epoch_index", -1)) != 0
        or int(state.get("checkpoint_index", 0)) < 1
        or int(state.get("complete_optimizer_steps", 0)) < 1
        or int(state.get("next_batch_offset", -1)) < 0
        or not isinstance(state.get("complete"), bool)
    ):
        raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_SCHEMA_INVALID]")
    order = state["epoch_order"]
    if (
        not isinstance(order, torch.Tensor)
        or order.dtype != torch.int64
        or order.ndim != 1
        or int(order.numel()) != int(dataset_rows)
        or not torch.equal(
            torch.sort(order.detach().cpu()).values,
            torch.arange(int(dataset_rows)),
        )
    ):
        raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_ORDER_INVALID]")
    model_state = state["model_state"]
    target_state = state["target_model_state"]
    optimizer_state = state["optimizer_state"]
    if not isinstance(model_state, Mapping) or not isinstance(target_state, Mapping):
        raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_MODEL_STATE_INVALID]")
    if not isinstance(optimizer_state, Mapping):
        raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_OPTIMIZER_INVALID]")
    try:
        model.load_state_dict(model_state, strict=True)
        target_model.load_state_dict(target_state, strict=True)
        target_model.requires_grad_(False)
        target_model.eval()
        optimizer.load_state_dict(optimizer_state)
        if weight_ema is None:
            if state["weight_ema_state"] is not None:
                raise RuntimeError("[ATTENDED_RESEARCH_WEIGHT_EMA_UNEXPECTED]")
        else:
            if not isinstance(state["weight_ema_state"], Mapping):
                raise RuntimeError("[ATTENDED_RESEARCH_WEIGHT_EMA_MISSING]")
            weight_ema.restore_checkpoint_state(
                state["weight_ema_state"], model=model
            )
        if lr_scheduler is None:
            if state["lr_scheduler_state"] is not None:
                raise RuntimeError("[ATTENDED_RESEARCH_LR_SCHEDULER_UNEXPECTED]")
        else:
            if not isinstance(state["lr_scheduler_state"], Mapping):
                raise RuntimeError("[ATTENDED_RESEARCH_LR_SCHEDULER_MISSING]")
            lr_scheduler.load_state_dict(state["lr_scheduler_state"])
        if not isinstance(state["rng_state"], Mapping):
            raise RuntimeError("[ATTENDED_RESEARCH_RNG_STATE_INVALID]")
        _restore_attended_session_rng_state(state["rng_state"], device=device)
    except (RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("[ATTENDED_"):
            raise
        raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_RESTORE_INVALID]") from exc
    return {
        "checkpoint_index": int(state["checkpoint_index"]),
        "complete_optimizer_steps": int(state["complete_optimizer_steps"]),
        "epoch_index": int(state["epoch_index"]),
        "next_batch_offset": int(state["next_batch_offset"]),
        "epoch_order": order.detach().cpu().contiguous(),
        "complete": bool(state["complete"]),
    }


def _fsync_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"[ENTRY_BUNDLE_STAGE_ARTIFACT_INVALID] {path}"
        )
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_checkpoint_failure_evidence(
    out_bundle_dir: Path, payload: Dict[str, Any]
) -> Path:
    resolved = Path(out_bundle_dir).expanduser().resolve()
    path = resolved.parent / f"{resolved.name}__checkpoint_failure_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "evidence_json": str(path),
        "bundle_written": False,
        "promotion_shadow_live_allowed": False,
    }
    path.write_text(
        json.dumps(
            enriched,
            indent=2,
            sort_keys=True,
            default=_train_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def validate(
    model,
    target_model,
    loader,
    device,
    *,
    collect_full_exit_trajectory: bool = False,
):
    model.eval()
    target_model.eval()
    if any(parameter.requires_grad for parameter in target_model.parameters()):
        raise RuntimeError("[UNIFIED_EXIT_TARGET_MODEL_NOT_FROZEN]")
    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, EntryV10CtxDataset):
        raise RuntimeError("[UNIFIED_EXIT_VALIDATION_DATASET_INVALID]")
    if dataset._unified_exit_lifecycle is None:
        raise RuntimeError("UNIFIED_EXIT_FULL_VAL_LIFECYCLE_MISSING")
    full_trajectory_accumulator = (
        _new_unified_exit_full_trajectory_accumulator(
            model=model,
            target_model=target_model,
        )
        if collect_full_exit_trajectory
        else None
    )
    total = 0.0
    entry_q_loss_sum = 0.0
    cooperation_gate_epoch = _new_cooperation_gate_epoch_accumulator()
    feature_tf_gate_epoch = _new_feature_tf_gate_epoch_accumulator()
    exit_cooperation_gate_epoch = _new_cooperation_gate_epoch_accumulator(
        _UNIFIED_EXIT_COOPERATION_GATE_WIDTHS
    )
    exit_feature_tf_gate_epoch = _new_feature_tf_gate_epoch_accumulator(
        _UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE
    )
    n = 0
    side_mae_loss_sum = 0.0
    trendline_event_loss_sum = 0.0
    trendline_event_rows_sum = 0
    trendline_support_rows_sum = 0
    trendline_resistance_rows_sum = 0
    unified_exit_loss_sum = 0.0
    unified_exit_population_rows = 0
    unified_exit_rows = 0
    unified_exit_tied_rows = 0
    unified_exit_eligible_entry_rows = 0
    unified_exit_hold_rows = 0
    unified_exit_now_rows = 0
    unified_exit_correct = 0
    active_head_epoch = _new_active_head_epoch_accumulator()
    entry_policy_realized_pnl_chunks: List[np.ndarray] = []
    entry_unique_target_rows = 0
    entry_target_equivalent_rows = 0
    entry_unique_target_agreement_rows = 0

    with torch.no_grad():
        for batch in loader:
            non_blocking = device.type == "cuda"
            seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
            snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
            ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
            ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
            batch_rows = int(seq_x.shape[0])
            out = _model_forward_fp32(
                model,
                seq_x,
                snap_x,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
                **_multi_tf_kwargs_from_batch(batch, seq_x.device),
            )
            target_out = _model_forward_fp32(
                target_model,
                seq_x,
                snap_x,
                ctx_cat=ctx_cat,
                ctx_cont=ctx_cont,
                **_multi_tf_kwargs_from_batch(batch, seq_x.device),
            )
            entry_representations = out.get(
                UNIFIED_EXIT_MODEL_REPRESENTATION_KEY
            )
            target_entry_representations = target_out.get(
                UNIFIED_EXIT_MODEL_REPRESENTATION_KEY
            )
            entry_row_indices = batch.get("entry_row_index")
            if (
                not isinstance(entry_representations, torch.Tensor)
                or not isinstance(target_entry_representations, torch.Tensor)
                or not isinstance(entry_row_indices, torch.Tensor)
            ):
                raise RuntimeError(
                    "[UNIFIED_EXIT_VALIDATION_ENTRY_EVIDENCE_MISSING]"
                )
            (
                unified_exit_loss,
                unified_exit_stats,
                entry_action_q_targets,
                entry_action_q_valid,
                entry_policy_realized_pnl_bps,
            ) = (
                _unified_exit_full_population_eval_loss(
                    model=model,
                    target_model=target_model,
                    entry_decision_representations=entry_representations,
                    target_entry_decision_representations=(
                        target_entry_representations
                    ),
                    entry_row_indices=entry_row_indices,
                    dataset=dataset,
                    device=device,
                    exit_cooperation_gate_epoch=exit_cooperation_gate_epoch,
                    exit_feature_tf_gate_epoch=exit_feature_tf_gate_epoch,
                    full_trajectory_accumulator=full_trajectory_accumulator,
                )
            )
            active_head_out = dict(out)
            active_head_out["_entry_action_q_target"] = (
                entry_action_q_targets
            )
            active_head_out["_entry_action_q_valid"] = (
                entry_action_q_valid
            )
            _accumulate_active_head_epoch(
                active_head_epoch,
                model,
                active_head_out,
                batch,
                device,
            )
            entry_action_q_bps = out["entry_action_q_bps"]
            if (
                not isinstance(entry_action_q_bps, torch.Tensor)
                or tuple(entry_action_q_bps.shape)
                != tuple(entry_action_q_targets.shape)
            ):
                raise RuntimeError(
                    "[ENTRY_FITTED_Q_VALIDATION_PREDICTION_SHAPE_INVALID]"
                )
            _accumulate_cooperation_gate_epoch(cooperation_gate_epoch, out)
            _accumulate_feature_tf_gate_epoch(feature_tf_gate_epoch, out)

            entry_action_q_loss = nn.functional.mse_loss(
                entry_action_q_bps[entry_action_q_valid],
                entry_action_q_targets[entry_action_q_valid],
            )
            task_losses: dict[str, torch.Tensor] = {
                "entry_action_q": entry_action_q_loss
            }
            if int(unified_exit_stats["q_valid_cells"]) > 0:
                task_losses["unified_exit_action"] = unified_exit_loss
            side_mae_loss, side_mae_stats = _side_mae_auxiliary_loss(
                out,
                batch,
                device,
            )
            task_losses["side_mae_bps"] = side_mae_loss
            trendline_event_loss, trendline_stats = _trendline_event_aux_loss(
                out, batch, device
            )
            task_losses["trendline_event"] = trendline_event_loss
            position_size_logit = _require_active_aux_head_prediction(
                out,
                batch,
                output_name="position_size_logit",
                target_names=("y_position_size_target", "y_position_size_mask"),
            )
            y_pos_size = batch["y_position_size_target"].to(device, non_blocking=non_blocking)
            y_pos_size_mask = batch["y_position_size_mask"].to(
                device, non_blocking=non_blocking
            )
            if bool((y_pos_size_mask.reshape(-1) == 1.0).any()):
                task_losses["position_size"] = _masked_position_size_mse(
                    position_size_logit,
                    y_pos_size,
                    y_pos_size_mask,
                )
            task_losses.update(dip_forecast_task_losses(out, batch, device))
            loss, _joint_task_stats = _joint_task_loss(model, task_losses)
            bs = batch_rows
            total += float(loss) * bs
            entry_q_loss_sum += float(entry_action_q_loss) * bs
            side_mae_loss_sum += float(side_mae_stats["side_mae_loss"]) * bs
            trendline_event_loss_sum += float(
                trendline_stats["trendline_event_loss"]
            ) * bs
            trendline_event_rows_sum += int(
                trendline_stats["trendline_event_rows"]
            )
            trendline_support_rows_sum += int(
                trendline_stats["trendline_support_rows"]
            )
            trendline_resistance_rows_sum += int(
                trendline_stats["trendline_resistance_rows"]
            )
            _exit_rows = int(unified_exit_stats["q_valid_cells"])
            unified_exit_loss_sum += (
                float(unified_exit_loss.detach().cpu().item()) * _exit_rows
            )
            unified_exit_population_rows += int(
                unified_exit_stats["population_rows"]
            )
            unified_exit_rows += _exit_rows
            unified_exit_tied_rows += int(
                unified_exit_stats["target_equivalent_action_rows"]
            )
            unified_exit_eligible_entry_rows += int(
                unified_exit_stats["eligible_entry_rows"]
            )
            unified_exit_hold_rows += int(
                unified_exit_stats["hold_target_greedy_rows"]
            )
            unified_exit_now_rows += int(
                unified_exit_stats["exit_now_target_greedy_rows"]
            )
            unified_exit_correct += int(
                unified_exit_stats["unique_target_action_agreement_rows"]
            )
            n += bs

            masked_entry_q = entry_action_q_bps.masked_fill(
                ~entry_action_q_valid, -torch.inf
            )
            winner_count = masked_entry_q.eq(
                masked_entry_q.amax(dim=1, keepdim=True)
            ).sum(dim=1)
            if bool((winner_count != 1).any().item()):
                raise RuntimeError("[ENTRY_FITTED_Q_VALIDATION_PREDICTED_TIE]")
            entry_predictions = torch.argmax(masked_entry_q, dim=1)
            realized_policy_pnl = entry_policy_realized_pnl_bps.gather(
                1, entry_predictions[:, None]
            ).squeeze(1)
            target_best = entry_action_q_targets.masked_fill(
                ~entry_action_q_valid, -torch.inf
            ).amax(dim=1, keepdim=True)
            target_equivalence = (
                entry_action_q_targets == target_best
            ) & entry_action_q_valid
            target_tied = target_equivalence.sum(dim=1) > 1
            target_agreement = target_equivalence.gather(
                1, entry_predictions[:, None]
            ).squeeze(1)
            entry_target_equivalent_rows += int(target_tied.sum().item())
            entry_unique_target_rows += int((~target_tied).sum().item())
            entry_unique_target_agreement_rows += int(
                (target_agreement & ~target_tied).sum().item()
            )
            entry_policy_realized_pnl_chunks.append(
                realized_policy_pnl.detach().cpu().numpy()
            )

    if (
        unified_exit_rows <= 0
        or unified_exit_hold_rows <= 0
        or unified_exit_now_rows <= 0
        or not math.isfinite(unified_exit_loss_sum)
        or unified_exit_loss_sum < 0.0
    ):
        raise RuntimeError(
            "[UNIFIED_EXIT_VALIDATION_EVIDENCE_INVALID] "
            f"rows={unified_exit_rows} hold={unified_exit_hold_rows} "
            f"exit_now={unified_exit_now_rows} loss_sum={unified_exit_loss_sum}"
        )

    acc = entry_unique_target_agreement_rows / max(1, entry_unique_target_rows)
    entry_policy_realized_pnl_bps = np.concatenate(
        entry_policy_realized_pnl_chunks, axis=0
    )
    if (
        entry_policy_realized_pnl_bps.shape != (n,)
        or not np.isfinite(entry_policy_realized_pnl_bps).all()
    ):
        raise RuntimeError("[ENTRY_FITTED_Q_REALIZED_POLICY_PNL_INVALID]")
    stats: Dict[str, Any] = {
        "entry_action_q_raw_bps_mse_mean": (entry_q_loss_sum / max(1, n)),
        "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean": float(
            np.mean(entry_policy_realized_pnl_bps)
        ),
        "entry_unique_target_rows": int(entry_unique_target_rows),
        "entry_target_equivalent_rows": int(entry_target_equivalent_rows),
        "entry_unique_target_action_agreement": (
            entry_unique_target_agreement_rows
            / max(1, entry_unique_target_rows)
        ),
        "side_mae_loss_mean": (side_mae_loss_sum / max(1, n)),
        "trendline_event_loss_mean": (trendline_event_loss_sum / max(1, n)),
        "trendline_event_rows": int(trendline_event_rows_sum),
        "trendline_support_rows": int(trendline_support_rows_sum),
        "trendline_resistance_rows": int(trendline_resistance_rows_sum),
        "unified_exit_raw_bps_q_mse_mean": (
            unified_exit_loss_sum / unified_exit_rows
        ),
        "unified_exit_population_rows": int(unified_exit_population_rows),
        "unified_exit_q_valid_cells": int(unified_exit_rows),
        "unified_exit_target_equivalent_action_rows": int(unified_exit_tied_rows),
        "unified_exit_unique_target_rows": int(
            unified_exit_population_rows - unified_exit_tied_rows
        ),
        "unified_exit_eligible_entry_rows": int(
            unified_exit_eligible_entry_rows
        ),
        "unified_exit_hold_target_greedy_rows": int(unified_exit_hold_rows),
        "unified_exit_exit_now_target_greedy_rows": int(unified_exit_now_rows),
        "unified_exit_unique_target_action_agreement": (
            unified_exit_correct
            / max(1, unified_exit_population_rows - unified_exit_tied_rows)
        ),
    }
    stats.update(
        {
            f"joint_task_log_variance_{name}": float(
                model.task_log_variances[name].detach().cpu().item()
            )
            for name in JOINT_TASK_NAMES
        }
    )
    stats.update(_finalize_cooperation_gate_epoch(cooperation_gate_epoch))
    stats.update(_finalize_feature_tf_gate_epoch(feature_tf_gate_epoch))
    exit_gate_stats, exit_gate_failures = _finalize_unified_exit_gate_epoch(
        exit_cooperation_gate_epoch,
        exit_feature_tf_gate_epoch,
    )
    stats.update(exit_gate_stats)
    if full_trajectory_accumulator is not None:
        stats["unified_exit_full_trajectory_validation"] = (
            _finalize_unified_exit_full_trajectory_validation(
                full_trajectory_accumulator,
                dataset=dataset,
                exit_gate_stats=exit_gate_stats,
                exit_gate_failures=exit_gate_failures,
            )
        )
    active_head_metrics, active_head_failures = _active_head_epoch_diagnostics(
        active_head_epoch
    )
    stats.update(active_head_metrics)
    if active_head_failures:
        log.error(
            "[ENTRY_ACTIVE_HEAD_HEALTH_CHECKPOINT_BLOCKED] %s",
            "; ".join(active_head_failures),
        )
    gate_failures = _cooperation_gate_health_failures(stats)
    stats["cooperation_gate_health_ok"] = not gate_failures
    stats["cooperation_gate_health_failures"] = list(gate_failures)
    if gate_failures:
        log.error(
            "[ENTRY_COOPERATION_GATE_HEALTH_CHECKPOINT_BLOCKED] %s",
            "; ".join(gate_failures),
        )
    if exit_gate_failures:
        log.error(
            "[UNIFIED_EXIT_COOPERATION_GATE_HEALTH_CHECKPOINT_BLOCKED] %s",
            "; ".join(exit_gate_failures),
        )
    # AUC is intentionally disabled for this 3-class path (previously hardcoded 0.0)
    return total / max(1, n), float("nan"), acc, float("nan"), stats


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
_ATTENDED_PREFLIGHT_NOTIFICATION_PREFIX = "gx1_attended_preflight_ready_v1:"
_ATTENDED_PREFLIGHT_TOKEN_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTENDED_EXECUTION_TIERS = frozenset(("attended_only", "attended_cpu_only"))
_ATTENDED_CPU_MAX_OPTIMIZER_STEPS = 1


def _is_attended_execution_tier(execution_tier: str) -> bool:
    """Return whether a tier is a non-promotable, operator-present smoke."""

    return execution_tier in _ATTENDED_EXECUTION_TIERS


def _announce_attended_preflight_ready(*, execution_tier: str) -> None:
    """Release the outer guard into its separately bounded model phase.

    The guard creates this private FIFO and unpredictable token immediately
    before it starts the exact canonical trainer. This function deliberately
    has no CLI arguments and is called at one location only: immediately after
    every full data/contract proof and immediately before model construction.
    """

    if not _is_attended_execution_tier(execution_tier):
        raise RuntimeError(
            "[ENTRY_ATTENDED_STAGE_NOTIFICATION_TIER_INVALID] "
            "only attended-only tiers may announce preflight completion"
        )
    fifo_raw = str(os.environ.get("GX1_TRAINER_ATTENDED_STAGE_FIFO") or "")
    token = str(os.environ.get("GX1_TRAINER_ATTENDED_STAGE_TOKEN") or "")
    if not fifo_raw or not _ATTENDED_PREFLIGHT_TOKEN_RE.fullmatch(token):
        raise RuntimeError(
            "[ENTRY_ATTENDED_STAGE_NOTIFICATION_MISSING] "
            "guard-created FIFO/token are required for attended-only tiers"
        )
    fifo_path = Path(fifo_raw)
    if not fifo_path.is_absolute():
        raise RuntimeError(
            "[ENTRY_ATTENDED_STAGE_NOTIFICATION_FIFO_INVALID] "
            "FIFO path must be absolute"
        )
    try:
        fifo_stat = os.stat(fifo_path, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            "[ENTRY_ATTENDED_STAGE_NOTIFICATION_FIFO_UNAVAILABLE] "
            f"{exc}"
        ) from exc
    if (
        not stat.S_ISFIFO(fifo_stat.st_mode)
        or fifo_stat.st_uid != os.getuid()
        or fifo_stat.st_mode & 0o077
    ):
        raise RuntimeError(
            "[ENTRY_ATTENDED_STAGE_NOTIFICATION_FIFO_UNSAFE] "
            "expected a private FIFO owned by this trainer user"
        )
    flags = os.O_WRONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    payload = f"{_ATTENDED_PREFLIGHT_NOTIFICATION_PREFIX}{token}\n".encode(
        "ascii"
    )
    fd: Optional[int] = None
    try:
        fd = os.open(fifo_path, flags)
        written = os.write(fd, payload)
    except OSError as exc:
        raise RuntimeError(
            "[ENTRY_ATTENDED_STAGE_NOTIFICATION_WRITE_FAILED] "
            f"{exc}"
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)
    if written != len(payload):
        raise RuntimeError(
            "[ENTRY_ATTENDED_STAGE_NOTIFICATION_PARTIAL_WRITE] "
            f"written={written} expected={len(payload)}"
        )
    log.info(
        "[ATTENDED_STAGE_NOTIFICATION] stage=data_preflight status=sent"
    )


def run_train(
    train_parquet: Path,
    train_manifest_path: Path,
    val_parquet: Path,
    unified_exit_lifecycle_manifest_path: Path,
    seq_len: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    out_bundle_dir: Path,
    gx1_data_override: str,
    num_workers: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    m5_prebuilt_path: Path,
    specialist_audit_json: Path,
    specialist_contract_mode: str,
    dropout: float,
    multi_tf_num_layers: int,
    per_tf_seq_len_m5: int,
    per_tf_seq_len_m15: int,
    per_tf_seq_len_h1: int,
    per_tf_seq_len_h4: int,
    per_tf_seq_len_d1: int,
    multi_tf_scale: float,
    subsample_rows: int,
    specialist_num_layers: int,
    specialist_fusion_scale: float,
    cross_family_fusion_scale: float,
    grad_accum_steps: int,
    prefreeze_test_seal_lineage: Mapping[str, Any],
    run_id: str = "",
    dataset_run_id: str = "",
    profile: str = "",
    execution_tier: str = "canonical",
    train_sequence_roll_audit_json: Optional[Path] = None,
    val_sequence_roll_audit_json: Optional[Path] = None,
    train_sequence_source_audit_json: Optional[Path] = None,
    val_sequence_source_audit_json: Optional[Path] = None,
) -> None:
    architecture = current_entry_exit_architecture_observation()
    architecture["entry"]["sequence_bars"] = seq_len
    architecture["exit"]["sequence_bars"] = (
        seq_len * ENTRY_EXIT_RESOLUTION_RATIO
        if isinstance(seq_len, int) and not isinstance(seq_len, bool)
        else seq_len
    )
    architecture["mtf"]["per_tf_window_bars"] = {
        "M5": per_tf_seq_len_m5,
        "M15": per_tf_seq_len_m15,
        "H1": per_tf_seq_len_h1,
        "H4": per_tf_seq_len_h4,
        "D1": per_tf_seq_len_d1,
    }
    require_entry_exit_production_architecture(
        architecture,
        context="ENTRY_V10_TRAINER_CONSTRUCTION",
    )
    _guard_no_rl()
    if profile not in ("smoke", "candidate"):
        raise RuntimeError(f"[ENTRY_TRAIN_PROFILE_INVALID] {profile!r}")
    if execution_tier not in ("canonical", *_ATTENDED_EXECUTION_TIERS):
        raise RuntimeError(
            f"[ENTRY_TRAIN_EXECUTION_TIER_INVALID] {execution_tier!r}"
        )
    if _is_attended_execution_tier(execution_tier) and profile != "smoke":
        raise RuntimeError(
            "[ENTRY_TRAIN_ATTENDED_TIER_PROFILE_INVALID] attended-only tiers require smoke"
        )
    if _is_attended_execution_tier(execution_tier) and (
        int(batch_size) != _ATTENDED_RESEARCH_BATCH_SIZE
        or int(epochs) != 1
        or int(grad_accum_steps) != 1
    ):
        raise RuntimeError(
            "[ATTENDED_RESEARCH_LOW_VRAM_GEOMETRY_INVALID] "
            "requires batch_size=8, epochs=1 and grad_accum_steps=1"
        )
    if execution_tier == "attended_only" and device.type != "cuda":
        raise RuntimeError("[ATTENDED_RESEARCH_CUDA_TIER_DEVICE_INVALID]")
    if execution_tier == "attended_cpu_only" and device.type != "cpu":
        raise RuntimeError("[ATTENDED_RESEARCH_CPU_TIER_DEVICE_INVALID]")
    reconstruction_audits = (
        train_sequence_source_audit_json,
        val_sequence_source_audit_json,
    )
    if any(value is None for value in reconstruction_audits):
        raise RuntimeError(
            "[ENTRY_TRAIN_SEQUENCE_SOURCE_PROOFS_REQUIRED] "
            "both TRAIN and VAL proofs are required for the mandatory "
            "source-backed sequence representation"
        )
    train_sequence_source_audit_json = _require_bound_sequence_source_audit(
        Path(train_sequence_source_audit_json), split="train"
    )
    val_sequence_source_audit_json = _require_bound_sequence_source_audit(
        Path(val_sequence_source_audit_json), split="val"
    )
    if any(value is not None for value in (
        train_sequence_roll_audit_json,
        val_sequence_roll_audit_json,
    )):
        raise RuntimeError(
            "[ENTRY_TRAIN_SEQUENCE_ROLL_RECONSTRUCTION_RETIRED]"
        )
    if profile == "candidate" and int(subsample_rows) != 0:
        raise RuntimeError(
            "[ENTRY_CANDIDATE_SUBSAMPLE_FORBIDDEN] candidate training must "
            "use the full TRAIN population"
        )
    try:
        prefreeze_test_seal_lineage = (
            require_prefreeze_test_seal_lineage_metadata(
                prefreeze_test_seal_lineage,
                expected_dataset_run_id=dataset_run_id,
                expected_dataset_dir=Path(train_parquet).parent,
            )
        )
    except (PrefreezeTestSealLineageError, ValueError) as exc:
        raise RuntimeError(
            f"[ENTRY_TRAIN_PREFREEZE_TEST_SEAL_LINEAGE_INVALID] {exc}"
        ) from exc
    if (
        isinstance(dropout, bool)
        or not math.isfinite(float(dropout))
        or not 0.0 <= float(dropout) < 1.0
    ):
        raise RuntimeError(
            "[ENTRY_TRAIN_DROPOUT_INVALID] dropout must be explicit, finite "
            f"and in [0,1); got {dropout!r}"
        )

    try:
        normalized_specialist_contract_mode = require_model_native_specialist_contract_mode(
            specialist_contract_mode
        )
    except ValueError as exc:
        raise RuntimeError(
            "[ENTRY_TRAIN_MODEL_NATIVE_SPECIALIST_CONTRACT_REQUIRED] "
            f"observed={specialist_contract_mode!r} expected={MODEL_NATIVE_CONTRACT_MODE!r}"
        ) from exc

    if m5_prebuilt_path is None:
        raise RuntimeError("[MULTI_TF_MANDATORY] m5_prebuilt_path is required")
    mtf_cache_raw = str(
        os.environ.get(_TRAIN_MULTI_TF_CACHE_ENV) or ""
    ).strip()
    mtf_cache_dir = Path(mtf_cache_raw).expanduser()
    if not mtf_cache_raw or not mtf_cache_dir.is_absolute():
        raise RuntimeError(
            "[MULTI_TF_DISK_CACHE_MANDATORY] "
            "GX1_V10_MULTI_TF_V4_CACHE_DIR must name the exact absolute "
            "verified V4 cache used by training and normalization"
        )
    if int(grad_accum_steps) < 1:
        raise RuntimeError(
            "[ENTRY_GRAD_ACCUM_STEPS_INVALID] "
            f"observed={int(grad_accum_steps)} expected>=1"
        )
    if (
        isinstance(multi_tf_num_layers, bool)
        or int(multi_tf_num_layers) <= 0
    ):
        raise RuntimeError(
            "[ENTRY_MULTI_TF_NUM_LAYERS_INVALID] expected a positive "
            f"explicit integer; got {multi_tf_num_layers!r}"
        )
    if (
        isinstance(specialist_num_layers, bool)
        or int(specialist_num_layers) <= 0
    ):
        raise RuntimeError(
            "[ENTRY_SPECIALIST_NUM_LAYERS_INVALID] expected a positive "
            f"explicit integer; got {specialist_num_layers!r}"
        )

    log.info(
        f"[TRAIN] seed={seed} device={device} batch_size={batch_size} epochs={epochs} lr={lr} "
        f"signal_dim={MODEL_NATIVE_SIGNAL_DIM} ctx_cont={MODEL_NATIVE_CTX_CONT_DIM} "
        f"ctx_cat={MODEL_NATIVE_CTX_CAT_DIM} early_stop_patience={early_stopping_patience} "
        f"early_stop_min_delta={early_stopping_min_delta}"
    )

    # Build exact per-TF sequence lengths.
    # How far back each timeframe reaches is decision-affecting, so it is an
    # explicit caller input for every timeframe - never an ambient environment
    # value, wrapper default or zero-to-global fallback (rule 14).
    _requested_tf_lens = {
        "M5": int(per_tf_seq_len_m5),
        "M15": int(per_tf_seq_len_m15),
        "H1": int(per_tf_seq_len_h1),
        "H4": int(per_tf_seq_len_h4),
        "D1": int(per_tf_seq_len_d1),
    }
    for _tf_name, _tf_len in _requested_tf_lens.items():
        if _tf_len <= 0:
            raise RuntimeError(
                f"[ENTRY_PER_TF_SEQ_LEN_INVALID] {_tf_name}={_tf_len} "
                "expected > 0; fallback is forbidden"
            )
    _per_tf_lens: Dict[str, int] = dict(_requested_tf_lens)
    _effective_tf_lens = dict(_per_tf_lens)
    from gx1.features.htf_features import require_multi_tf_resolution_pyramid

    multi_tf_resolution_pyramid = require_multi_tf_resolution_pyramid(
        _effective_tf_lens
    )
    log.info(
        "[PER_TF_SEQ_LEN_DECLARED] %s coverage_seconds=%s",
        " ".join(f"{tf}={n}" for tf, n in _effective_tf_lens.items()),
        multi_tf_resolution_pyramid["coverage_seconds"],
    )

    _set_deterministic(seed, device)

    # Pre-build the one V4 cache, bind TRAIN/VAL lifecycle clocks, then prove
    # both exact routes before normalization or optimization begins.
    multi_tf_features = _prebuild_multi_tf_features_once(m5_prebuilt_path)
    from gx1.features.htf_features import (
        require_multi_tf_decision_window_coverage,
    )

    unified_exit_lifecycle = UnifiedExitLifecycleCorpus(
        root_manifest_path=unified_exit_lifecycle_manifest_path,
        entry_parquets={
            "train": Path(train_parquet),
            "val": Path(val_parquet),
        },
        dataset_run_id=str(dataset_run_id),
        splits=("train", "val"),
    )
    decision_times_by_route_split: dict[str, dict[str, object]] = {
        "entry": {},
        "exit": {},
    }
    for _split_name, _split_path in (
        ("train", train_parquet),
        ("val", val_parquet),
    ):
        try:
            _split_times = pd.read_parquet(
                _split_path, columns=["time"]
            )["time"]
        except Exception as exc:
            raise RuntimeError(
                "[MULTI_TF_DECISION_COVERAGE_SPLIT_READ_FAIL] "
                f"{_split_name}={_split_path}: {exc}"
            ) from exc
        decision_times_by_route_split["entry"][_split_name] = _split_times
        decision_times_by_route_split["exit"][_split_name] = pd.to_datetime(
            unified_exit_lifecycle.splits[
                _split_name
            ].selected_current_decision_times_ns(),
            unit="ns",
            utc=True,
        )
    multi_tf_decision_window_coverage = (
        require_multi_tf_decision_window_coverage(
            multi_tf_features,
            per_tf_seq_lens=_effective_tf_lens,
            decision_times_by_route_split=decision_times_by_route_split,
        )
    )
    log.info(
        "[MULTI_TF_DECISION_WINDOW_COVERAGE] contract_sha256=%s",
        multi_tf_decision_window_coverage["contract_sha256"],
    )


    train_ds = EntryV10CtxDataset(
        train_parquet,
        seq_len=seq_len,
        m5_prebuilt_path=m5_prebuilt_path,
        per_tf_seq_lens=_per_tf_lens,
        multi_tf_closed_bar=True,
        sequence_source_audit_json=train_sequence_source_audit_json,
    )
    physical_train_rows = int(len(train_ds))
    # Bind the immutable TRAIN lifecycle before normalization so the shared
    # The owner-declared signal/context fit sees the exact M1 Exit rows selected by the same lifecycle
    # used by optimization. This occurs before any smoke subsampling.
    train_exit_lifecycle = unified_exit_lifecycle.splits["train"]
    train_ds.bind_unified_exit_lifecycle(train_exit_lifecycle)
    # Normalization must be fitted over the same windows the model reads, so it
    # takes the one resolution above rather than re-deriving it. A second
    # derivation is a second truth waiting to drift.
    normalization_per_tf_seq_lens = dict(_effective_tf_lens)
    normalization_fit = fit_entry_v10_train_input_normalization(
        train_seq=train_ds._np_seq,
        train_snap=train_ds._np_snap,
        train_ctx_cont=train_ds._np_ctx_cont,
        train_ctx_cat=train_ds._np_ctx_cat,
        train_times=train_ds.df["time"],
        train_exit_lifecycle=train_exit_lifecycle,
        ordered_signal_names=list(train_ds.signal_names),
        per_tf_seq_lens=normalization_per_tf_seq_lens,
        artifacts=TrainNormalizationArtifacts(
            dataset_run_id=str(dataset_run_id),
            train_parquet_path=Path(train_parquet),
            train_manifest_path=Path(train_manifest_path),
            m5_prebuilt_path=Path(m5_prebuilt_path),
            mtf_cache_dir=mtf_cache_dir,
        ),
        prevalidated_multi_tf_cache=multi_tf_features,
        entry_sequence_source=(
            train_ds.source_reconstruction_normalization_input()
        ),
    )
    input_normalization = normalization_fit["normalization_contract"]
    input_normalization_fit_population_proof = normalization_fit[
        "fit_population_proof"
    ]
    log.info(
        "[ENTRY_INPUT_NORMALIZATION_FIT] contract_sha256=%s "
        "shared_context_train_rows=%d val_rows=0 test_rows=0",
        input_normalization["contract_sha256"],
        int(
            input_normalization_fit_population_proof[
                "train_decision_row_count"
            ]
        ),
    )
    # V12.2 sweep mode: uniform TRAIN-only subsample (VAL untouched). Sampling
    # may not depend on a retired direction label.
    if subsample_rows > 0 and subsample_rows < len(train_ds):
        rng = np.random.default_rng(seed=seed)
        sampled_idx = sorted(
            rng.choice(
                np.arange(len(train_ds), dtype=np.int64),
                size=int(subsample_rows),
                replace=False,
            ).tolist()
        )
        train_ds.indices = np.array(sampled_idx, dtype=np.int64)
        train_ds.compact_materialized_rows(train_ds.indices)
        log.info(
            "[SUBSAMPLE] uniform TRAIN-only subsample: %d/%d rows",
            len(sampled_idx),
            len(train_ds.df),
        )
    effective_train_rows = int(len(train_ds))
    val_ds = EntryV10CtxDataset(
        val_parquet,
        seq_len=seq_len,
        m5_prebuilt_path=m5_prebuilt_path,
        per_tf_seq_lens=_per_tf_lens,
        multi_tf_closed_bar=True,
        sequence_source_audit_json=val_sequence_source_audit_json,
    )
    val_ds.bind_unified_exit_lifecycle(
        unified_exit_lifecycle.splits["val"]
    )
    if (
        not train_ds._sequence_source_reconstructed
        or not val_ds._sequence_source_reconstructed
        or train_ds._sequence_source_audit is None
        or val_ds._sequence_source_audit is None
    ):
        raise RuntimeError(
            "[ENTRY_TRAIN_SEQUENCE_SOURCE_RECONSTRUCTION_MISSING]"
        )
    sequence_source_reconstruction_evidence: dict[str, Any] = {
        "schema_version": "entry_model_native_sequence_source_reconstruction_v1",
        "authority": "data_reconstruction_only",
        "candidate": False,
        "test": False,
        "promotion": False,
        "paper": False,
        "live": False,
        "splits": {
            "train": dict(train_ds._sequence_source_audit),
            "val": dict(val_ds._sequence_source_audit),
        },
    }
    log.info(
        "[SEQUENCE_SOURCE_RECONSTRUCTION] TRAIN+VAL proof-bound storage "
        "representation active; authority=data_reconstruction_only"
    )
    unified_exit_lifecycle_evidence = dict(
        unified_exit_lifecycle.evidence
    )
    m1_feature_surface_binding = _m1_feature_surface_binding_from_lifecycle(
        unified_exit_lifecycle_evidence,
        dataset_run_id=str(dataset_run_id),
    )
    log.info(
        "[UNIFIED_EXIT_LIFECYCLE_BOUND] manifest_sha256=%s "
        "train_state_rows=%s val_state_rows=%s",
        unified_exit_lifecycle_evidence["root_manifest_sha256"],
        unified_exit_lifecycle_evidence["splits"]["train"][
            "state_population_rows"
        ],
        unified_exit_lifecycle_evidence["splits"]["val"][
            "state_population_rows"
        ],
    )
    train_contract_mode = str(train_ds.contract_mode)
    val_contract_mode = str(val_ds.contract_mode)
    if (
        train_contract_mode != MODEL_NATIVE_CONTRACT_MODE
        or val_contract_mode != MODEL_NATIVE_CONTRACT_MODE
    ):
        raise RuntimeError(
            "[ENTRY_TRAIN_MODEL_NATIVE_SIGNAL_CONTRACT_REQUIRED] "
            f"train={train_contract_mode!r} val={val_contract_mode!r}"
        )
    direction_logit_mode = str(train_ds.direction_logit_mode)
    if (
        direction_logit_mode != MODEL_NATIVE_DIRECTION_LOGIT_MODE
        or str(val_ds.direction_logit_mode) != MODEL_NATIVE_DIRECTION_LOGIT_MODE
    ):
        raise RuntimeError("[ENTRY_TRAIN_MODEL_NATIVE_DIRECTION_MODE_REQUIRED]")
    if normalized_specialist_contract_mode != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError("[ENTRY_TRAIN_SPECIALIST_CONTRACT_MODE_INVALID]")
    model_native_training_objective = training_objective_contract_metadata()
    for split_name, ds_obj in (("train", train_ds), ("val", val_ds)):
        contract = ds_obj.model_native_signal_contract
        require_model_native_signal_contract(
            contract,
            context=f"ENTRY_TRAIN_{split_name.upper()}",
        )
        if list(ds_obj.signal_names) != list(contract["fields"]):
            raise RuntimeError(
                f"[ENTRY_TRAIN_{split_name.upper()}_MODEL_NATIVE_SIGNAL_ORDER_MISMATCH]"
            )
    if train_ds.aux_head_target_contract != val_ds.aux_head_target_contract:
        raise RuntimeError(
            "[ENTRY_TRAIN_AUX_TARGET_CONTRACT_SPLIT_MISMATCH] "
            "TRAIN and VAL must share one immutable 46-column future-target contract"
        )
    entry_position_size_target_policy = (
        _entry_position_size_target_policy_from_manifest(
            Path(train_manifest_path),
        )
    )
    val_entry_position_size_target_policy = (
        _entry_position_size_target_policy_from_manifest(
            Path(val_parquet).expanduser().resolve().with_suffix(
                ".manifest.json"
            ),
        )
    )
    if (
        val_entry_position_size_target_policy
        != entry_position_size_target_policy
    ):
        raise RuntimeError("[ENTRY_TRAIN_POSITION_SIZE_POLICY_SPLIT_MISMATCH]")
    contract_failures: list[str] = []
    contract_failures.extend(
        _xau_direction_repair_source_failures(
            {
                "train_parquet": train_parquet,
                "val_parquet": val_parquet,
                "m5_prebuilt_path": m5_prebuilt_path,
            }
        )
    )
    contract_failures.extend(
        _xau_direction_repair_manifest_failures(
            {
                "train": train_parquet,
                "val": val_parquet,
            }
        )
    )
    for split_name, ds_obj in (("train", train_ds), ("val", val_ds)):
        forbidden_present = sorted(
            set(ds_obj.signal_names)
            & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
        )
        if forbidden_present:
            contract_failures.append(
                f"{split_name} dataset contains forbidden bridge fields: {forbidden_present}"
            )
    if contract_failures:
        raise RuntimeError(
            "[ENTRY_XAU_DIRECTION_REPAIR_CONTRACT_INVALID] "
            + "; ".join(contract_failures)
        )
    required_raw_aux_cols = [
            "y_long_expected_mae_bps",
            "y_short_expected_mae_bps",
            "y_line_support_touch_held",
            "y_line_support_touch_mask",
            "y_line_resistance_touch_held",
            "y_line_resistance_touch_mask",
            "y_countertrend_short_trap",
            "y_countertrend_long_trap",
    ]
    for split_name, ds_obj in (("train", train_ds), ("val", val_ds)):
        missing = [c for c in required_raw_aux_cols if c not in ds_obj.df.columns]
        if missing:
            raise RuntimeError(
                f"[ENTRY_RAW_AUX_LABEL_CONTRACT_MISSING] split={split_name} "
                f"missing={missing}. Rebuild the hash-bound dataset; raw "
                "outcome/event auxiliaries must not use fallback labels."
            )

    if int(num_workers) != 0:
        raise RuntimeError(
            "[ENTRY_DATALOADER_WORKERS_INVALID] num_workers must equal 0 "
            "under the fixed low-memory recipe"
        )
    pin_memory = False
    persistent_workers = False
    prefetch_factor = None
    log.info(
        "[DATALOADER_CONFIG] num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        num_workers, pin_memory, persistent_workers, str(prefetch_factor),
    )

    attended_session: Optional[_AttendedResearchSession] = None
    attended_checkpoint_state: Optional[dict[str, Any]] = None
    attended_epoch_order: Optional[torch.Tensor] = None
    attended_batch_offset = 0
    attended_max_optimizer_steps: Optional[int] = None
    if _is_attended_execution_tier(execution_tier):
        attended_max_optimizer_steps = (
            _ATTENDED_CPU_MAX_OPTIMIZER_STEPS
            if device.type == "cpu"
            else _ATTENDED_RESEARCH_MAX_OPTIMIZER_STEPS
        )
        attended_session = _AttendedResearchSession(
            out_bundle_dir=_resolve_train_out_bundle_dir(
                out_bundle_dir, gx1_data_override
            ),
            contract=_attended_research_session_contract(
                out_bundle_dir=_resolve_train_out_bundle_dir(
                    out_bundle_dir, gx1_data_override
                ),
                run_id=run_id,
                dataset_run_id=dataset_run_id,
                train_parquet=Path(train_parquet),
                val_parquet=Path(val_parquet),
                m5_prebuilt_path=Path(m5_prebuilt_path),
                lifecycle_manifest_path=Path(
                    unified_exit_lifecycle_manifest_path
                ),
                input_normalization=input_normalization,
                seed=seed,
                batch_size=batch_size,
                epochs=epochs,
                grad_accum_steps=grad_accum_steps,
                subsample_rows=subsample_rows,
                lr=lr,
                dropout=dropout,
                execution_tier=execution_tier,
                device_type=device.type,
                max_optimizer_steps=attended_max_optimizer_steps,
            ),
        )
        attended_checkpoint_state = attended_session.load_checkpoint()
        if attended_checkpoint_state is None:
            attended_epoch_order = _new_attended_research_epoch_order(
                effective_train_rows
            )
        else:
            pending_order = attended_checkpoint_state.get("epoch_order")
            if not isinstance(pending_order, torch.Tensor):
                raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_ORDER_INVALID]")
            attended_epoch_order = pending_order.detach().cpu().contiguous()
            attended_batch_offset = int(
                attended_checkpoint_state.get("next_batch_offset", -1)
            )
            if bool(attended_checkpoint_state.get("complete", False)):
                log.info(
                    "[ATTENDED_RESEARCH_SESSION_ALREADY_COMPLETE] directory=%s "
                    "authority=none",
                    attended_session.directory,
                )
        if attended_batch_offset > -(-int(effective_train_rows) // int(batch_size)):
            raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_OFFSET_INVALID]")
        log.info(
            "[ATTENDED_RESEARCH_SESSION] directory=%s resumed=%d batch_offset=%d "
            "max_optimizer_steps=%d exit_chunk_rows=%d authority=none",
            attended_session.directory,
            int(attended_checkpoint_state is not None),
            attended_batch_offset,
            int(attended_max_optimizer_steps),
            _ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS,
        )

    if attended_epoch_order is None:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=_ExactIndexSampler(
                attended_epoch_order,
                batch_offset=(
                    0
                    if attended_checkpoint_state is not None
                    and bool(attended_checkpoint_state.get("complete", False))
                    else attended_batch_offset
                ),
                batch_size=batch_size,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    # V30 package 6: resolve the recipe-declared weight-EMA horizon into the
    # decay this run applies. The recipe owner owns both the declared vocabulary
    # and the formula (rule 14); the trainer only supplies the run's declared
    # budget. `effective_train_rows` is the rows one epoch iterates: the
    # declared --subsample-rows budget as realized by uniform sampling on
    # the smoke profile, and the complete declared TRAIN population on the
    # candidate profile (where --subsample-rows is 0 by contract). Rule 2g: the
    # horizon is only "one epoch" if that row count is the one the optimizer
    # actually steps through, so the loader's own micro-batch count is required
    # to agree before the derivation is trusted.
    _expected_train_batches = -(-int(effective_train_rows) // int(batch_size))
    _require(
        (
            len(train_loader) == _expected_train_batches
            if not _is_attended_execution_tier(execution_tier)
            else len(train_loader)
            == _expected_train_batches
            - (
                0
                if attended_checkpoint_state is not None
                and bool(attended_checkpoint_state.get("complete", False))
                else attended_batch_offset
            )
        ),
        "[WEIGHT_EMA_EPOCH_ROWS_MISMATCH] "
        f"train_loader batches={len(train_loader)} != "
        f"expected attended-aware batches for ceil({effective_train_rows}/{batch_size}) — the EMA horizon cannot "
        "be derived from a row budget the epoch does not iterate",
    )
    weight_ema_derivation = resolve_weight_ema_decay(
        ENTRY_TRAIN_WEIGHT_EMA_DECAY_DECLARED,
        train_rows=int(effective_train_rows),
        batch_size=int(batch_size),
        grad_accum_steps=int(grad_accum_steps),
    )
    train_weight_ema_decay = float(weight_ema_derivation["weight_ema_decay"])
    log.info(
        "[TRAIN_WEIGHT_EMA_DERIVATION] %s",
        json.dumps(weight_ema_derivation, sort_keys=True),
    )

    # Before first epoch: prove the exact model-native input contract.
    sample = next(iter(train_loader))
    seq_input_dim = int(sample["seq_x"].shape[2])
    snap_input_dim = int(sample["snap_x"].shape[1])
    _require(seq_input_dim == snap_input_dim and seq_input_dim > 0, f"[SIGNAL_DIM_INVALID] seq={seq_input_dim} snap={snap_input_dim}")
    ctx_cont_dim = int(sample["ctx_cont"].shape[1])
    ctx_cat_dim = int(sample["ctx_cat"].shape[1])
    ordered_ctx_cont_names = list(MODEL_NATIVE_CTX_CONT_FIELDS)
    ordered_ctx_cat_names = list(MODEL_NATIVE_CTX_CAT_FIELDS)
    if len(ordered_ctx_cat_names) != ctx_cat_dim:
        raise RuntimeError(
            f"[CTX_CAT_NAME_DIM_MISMATCH] ordered_ctx_cat_names={len(ordered_ctx_cat_names)} "
            f"!= trained ctx_cat_dim={ctx_cat_dim}"
        )
    log.info(
        f"[TRAIN_CONTRACT] seq_x={sample['seq_x'].shape} snap_x={sample['snap_x'].shape} "
        f"ctx_cont={sample['ctx_cont'].shape} ctx_cat={sample['ctx_cat'].shape}"
    )
    log.info(
        "[ENTRY_INPUT_SCHEMA_PROOF] signal_dim=%d ctx_cont_dim=%d ctx_cat_dim=%d contract_mode=%s",
        seq_input_dim,
        ctx_cont_dim,
        ctx_cat_dim,
        MODEL_NATIVE_CONTRACT_MODE,
    )
    _require(
        ctx_cont_dim == MODEL_NATIVE_CTX_CONT_DIM,
        f"[ENTRY_CTX_CONT_DIM_MISMATCH] expected ctx_cont_dim={MODEL_NATIVE_CTX_CONT_DIM} got={ctx_cont_dim}",
    )
    _require(
        ctx_cat_dim == MODEL_NATIVE_CTX_CAT_DIM,
        f"[ENTRY_CTX_CAT_DIM_MISMATCH] expected ctx_cat_dim={MODEL_NATIVE_CTX_CAT_DIM} got={ctx_cat_dim}",
    )
    _require(
        sample["seq_x"].shape[2] == seq_input_dim
        and sample["snap_x"].shape[1] == snap_input_dim
        and sample["ctx_cont"].shape[1] == ctx_cont_dim
        and sample["ctx_cat"].shape[1] == ctx_cat_dim,
        f"[TRAIN_CONTRACT_MISMATCH] expected signal={seq_input_dim} ctx_cont={ctx_cont_dim} ctx_cat={ctx_cat_dim}",
    )

    # Width and contract both come from the Dataset, which verified them
    # against the cache's own declaration.
    _mtf_feat_count = int(train_ds._multi_tf_feature_count)
    # Exact mode always includes the causal M5 branch and all four higher TFs.
    _mtf_v4 = bool(getattr(train_ds, "_multi_tf_v4", False))
    if not _mtf_v4 or _mtf_feat_count <= 0:
        raise RuntimeError(
            "[MULTI_TF_EXACT_ARCHITECTURE_REQUIRED] expected causal "
            "M5/M15/H1/H4/D1 V4 eight-family input"
        )
    # One exact positive per-TF resolution, mirrored into the model so metadata records the
    # exact windows and live reads the same ones (train==serve).
    _m5_len = int(_effective_tf_lens["M5"])
    _m15_len = int(_effective_tf_lens["M15"])
    _h1_len = int(_effective_tf_lens["H1"])
    _h4_len = int(_effective_tf_lens["H4"])
    _d1_len = int(_effective_tf_lens["D1"])
    specialist_indices, specialist_meta = _load_specialist_fusion_contract(
        specialist_audit_json,
        expected_signal_dim=seq_input_dim,
        ordered_signal_names=list(train_ds.signal_names),
        contract_mode=specialist_contract_mode,
    )
    multi_tf_specialist_indices = {
        str(name): list(indices)
        for name, indices in require_multi_tf_specialist_routing_v4(
            train_ds._multi_tf_feature_names
        ).items()
    }
    log.info("[SPECIALIST_FUSION] exact groups=%s", sorted(specialist_indices))
    log.info(
        "[MULTI_TF_SPECIALIST_FUSION] exact groups=%s",
        {
            name: len(indices)
            for name, indices in multi_tf_specialist_indices.items()
        },
    )
    # This is the last point after complete immutable-data preflight and
    # before CUDA/model allocation.  Keep the measurement source-local so an
    # attended cgroup result can distinguish data-residency pressure from a
    # model/episode allocation without changing either training surface.
    log.info(
        "[TRAIN_RSS] pre_model_construct rss_gib=%.2f",
        _train_rss_gib(),
    )
    if _is_attended_execution_tier(execution_tier):
        _announce_attended_preflight_ready(execution_tier=execution_tier)
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
                cuda_index = torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(
                    _ATTENDED_RESEARCH_CUDA_MEMORY_FRACTION,
                    cuda_index,
                )
                total_mib = int(
                    torch.cuda.get_device_properties(cuda_index).total_memory // (1024 * 1024)
                )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "[ATTENDED_RESEARCH_CUDA_MEMORY_FENCE_FAILED]"
                ) from exc
            log.info(
                "[ATTENDED_RESEARCH_CUDA_MEMORY_FENCE] fraction=%.2f budget_mib=%d "
                "batch_size=%d exit_chunk_rows=%d",
                _ATTENDED_RESEARCH_CUDA_MEMORY_FRACTION,
                int(total_mib * _ATTENDED_RESEARCH_CUDA_MEMORY_FRACTION),
                _ATTENDED_RESEARCH_BATCH_SIZE,
                _ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS,
            )
        else:
            log.info(
                "[ATTENDED_RESEARCH_CPU_FENCE] batch_size=%d max_optimizer_steps=%d "
                "exit_chunk_rows=%d authority=none",
                _ATTENDED_RESEARCH_BATCH_SIZE,
                _ATTENDED_CPU_MAX_OPTIMIZER_STEPS,
                _ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS,
            )
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        dropout=float(dropout),
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        m15_seq_dim=_mtf_feat_count,
        h1_seq_dim=_mtf_feat_count,
        h4_seq_dim=_mtf_feat_count,
        d1_seq_dim=_mtf_feat_count,
        m15_seq_len=_m15_len,
        h1_seq_len=_h1_len,
        h4_seq_len=_h4_len,
        d1_seq_len=_d1_len,
        m5_seq_dim=_mtf_feat_count,
        m5_seq_len=_m5_len,
        multi_tf_num_layers=int(multi_tf_num_layers),
        multi_tf_scale=multi_tf_scale,
        specialist_input_indices=specialist_indices,
        specialist_ctx_cont_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"][
                "ctx_cont_indices"
            ].items()
        },
        specialist_ctx_cont_nominal_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"][
                "ctx_cont_nominal_indices"
            ].items()
        },
        specialist_ctx_cat_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"][
                "ctx_cat_indices"
            ].items()
        },
        multi_tf_specialist_input_indices=multi_tf_specialist_indices,
        temporal_alias_signal_indices=list(
            specialist_meta["context_routing"]["temporal_alias_policy"][
                "signal_indices"
            ]
        ),
        temporal_alias_ctx_cont_indices=list(
            specialist_meta["context_routing"]["temporal_alias_policy"][
                "ctx_cont_indices"
            ]
        ),
        specialist_num_layers=int(specialist_num_layers),
        specialist_fusion_scale=float(specialist_fusion_scale),
        cross_family_fusion_scale=float(cross_family_fusion_scale),
        input_normalization=input_normalization,
    ).to(device)
    # This is deliberately a preflight invariant, not a best-effort log.  It
    # proves the normalization buffers and their host-only routing caches are
    # exactly the immutable metadata contract before any optimizer state or
    # GPU work is allocated to a training candidate.
    model.require_input_normalization_state()
    entry_q_initial_state = _capture_entry_q_initial_state(model)
    unified_exit_initial_state = _capture_unified_exit_initial_state(model)
    log.info(
        "[TF_INPUT_SCALE] all five learnable scales start from the same "
        "contract-owned neutral identity; no timeframe prior"
    )
    log.info(
        "[ENTRY_EXACT_HEADS] entry_q=true side_mae_raw_bps=true "
        "trendline_events=true position_size=true all_aux_direction_authority=none",
    )
    log.info(
        "[TRAIN_MEMORY_POLICY] activation_checkpoint=%s "
        "features=unchanged samples=unchanged batch_semantics=unchanged",
        TRAIN_ACTIVATION_CHECKPOINT_POLICY,
    )
    log.info(
        "[MULTI_TF_PROOF] enabled=True TFs=M5+M15+H1+H4+D1 (V4) "
        "per_tf_dim=%d per_tf_lens=%s total_extra_params≈%dK",
        _mtf_feat_count, _effective_tf_lens,
        (sum(p.numel() for p in model.parameters()) - 691977) // 1000,
    )
    head_out = int(getattr(model.head_entry_action_q, "out_features", -1))
    if head_out != 3:
        raise RuntimeError("[ENTRY_ACTION_Q_HEAD_WIDTH_INVALID]")
    log.info("[ENTRY_FITTED_Q_PROOF] head_entry_action_q_out=%s", head_out)
    log.info(
        "[ENTRY_CTX_SCALE] ctx_cat_scale=%.3f ctx_cont_scale=%.3f",
        float(model.cfg.ctx_cat_scale),
        float(model.cfg.ctx_cont_scale),
    )
    log.info(
        "[ENTRY_MODEL_NATIVE_DIRECTION_PROOF] mode=%s signal_dim=%d",
        direction_logit_mode,
        int(seq_input_dim),
    )
    preflight_batch = {
        key: (value[:1] if isinstance(value, torch.Tensor) and value.ndim > 0 else value)
        for key, value in sample.items()
    }
    preflight_seq = preflight_batch["seq_x"].to(device)
    preflight_snap = preflight_batch["snap_x"].to(device)
    preflight_ctx_cont = preflight_batch["ctx_cont"].to(device)
    preflight_ctx_cat = preflight_batch["ctx_cat"].to(device)
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        preflight_out = _model_forward_fp32(
            model,
            preflight_seq,
            preflight_snap,
            ctx_cat=preflight_ctx_cat,
            ctx_cont=preflight_ctx_cont,
            **_multi_tf_kwargs_from_batch(preflight_batch, device),
        )
    if was_training:
        model.train()
    output_head_failures = _model_native_active_output_head_failures(preflight_out)
    if output_head_failures:
        raise RuntimeError(
            "[ENTRY_TRAIN_MODEL_NATIVE_ACTIVE_OUTPUT_CONTRACT_INVALID] "
            + "; ".join(output_head_failures)
        )
    preflight_task_names = set(
        dip_forecast_task_losses(preflight_out, preflight_batch, device)
    )
    expected_preflight_tasks = {
        "dip_bps",
        "forecast_return_bps",
        "dip_timing_fraction",
        "tail_risk_bps",
        "forward_volatility_bps",
    }
    if preflight_task_names != expected_preflight_tasks:
        raise RuntimeError(
            "[ENTRY_JOINT_TASK_PREFLIGHT_INVALID] "
            f"observed={sorted(preflight_task_names)} "
            f"expected={sorted(expected_preflight_tasks)}"
        )
    del preflight_out, preflight_batch

    log.info(
        "[ENTRY_TRAIN_RECIPE] lr_cosine_decay=%d weight_ema_decay=%.6f "
        "entry_action_q_loss=masked_raw_bps_mse "
        "main_direction_loss=retired mtf_direction_loss=retired "
        "handwritten_direction_distribution_forcing=0 "
        "handwritten_hierarchical_distribution_forcing=0 "
        "classification_weighting=unweighted "
        "task_weighting=trainable_homoscedastic_uncertainty "
        "task_initialization=neutral_equal raw_bps_targets=1 "
        "rank_losses=retired composite_losses=retired gate_gradient_forcing=retired",
        int(ENTRY_TRAIN_LR_COSINE_DECAY),
        float(train_weight_ema_decay),
    )

    joint_task_parameters = list(model.task_log_variances.parameters())
    joint_task_parameter_ids = {id(parameter) for parameter in joint_task_parameters}
    representation_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in joint_task_parameter_ids
    ]
    optimizer = optim.AdamW(
        [
            {
                "params": representation_parameters,
                "weight_decay": _WEIGHT_DECAY,
            },
            {
                "params": joint_task_parameters,
                # Decoupled decay would be an unadvertised fixed pull toward
                # equal task weights and therefore is forbidden here.
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
    )

    # ── V30 package 5 stability dampers (recipe-owned, both exactly OFF-able) ─
    # (i) Cosine LR decay over the DECLARED epoch budget. The scheduler is the
    #     library's own CosineAnnealingLR with T_max = epochs and the default
    #     eta_min = 0.0; epoch t therefore trains at
    #     lr * 0.5 * (1 + cos(pi * t / epochs)) and the schedule reaches exactly
    #     0 at the end of the declared budget. No warmup, no restarts, and no
    #     magnitude is introduced here — the only inputs are `lr` (CLI) and
    #     `epochs` (CLI). At the OFF switch `lr_scheduler` stays None: no object
    #     is built and no param_group is ever written, so every step runs at the
    #     same constant `lr` the pre-package-5 trainer used.
    lr_scheduler = None
    if int(ENTRY_TRAIN_LR_COSINE_DECAY) == 1:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(epochs),
            eta_min=0.0,
        )
    # (ii) Weight EMA, read ONLY by validation and checkpoint selection. The raw
    #      weights keep training. At the 0.0 OFF sentinel no instance exists.
    weight_ema = (
        _WeightEma(model, float(train_weight_ema_decay))
        if float(train_weight_ema_decay) > 0.0
        else None
    )
    log.info(
        "[TRAIN_STABILITY_DAMPERS] lr_cosine_decay=%d epochs=%d lr=%.6g "
        "weight_ema_decay=%.6f weight_ema_active=%d",
        int(ENTRY_TRAIN_LR_COSINE_DECAY),
        int(epochs),
        float(lr),
        float(train_weight_ema_decay),
        int(weight_ema is not None),
    )

    if attended_session is not None:
        # This branch is intentionally terminal.  It does not invoke VAL,
        # checkpoint selection, bundle writing or any promotion-capable code;
        # its only product is the hash-bound research-session state owned above.
        if (
            attended_checkpoint_state is not None
            and bool(attended_checkpoint_state.get("complete", False))
        ):
            log.info(
                "[ATTENDED_RESEARCH_SESSION_TERMINAL] directory=%s "
                "status=already_complete authority=none",
                attended_session.directory,
            )
            return
        if attended_epoch_order is None:
            raise RuntimeError("[ATTENDED_RESEARCH_ORDER_MISSING]")
        target_model = copy.deepcopy(model).to(device)
        target_model.requires_grad_(False)
        target_model.eval()
        if attended_checkpoint_state is None:
            attended_progress = {
                "checkpoint_index": 0,
                "complete_optimizer_steps": 0,
                "epoch_index": 0,
                "next_batch_offset": 0,
                "epoch_order": attended_epoch_order,
                "complete": False,
            }
        else:
            attended_progress = _restore_attended_research_checkpoint(
                attended_checkpoint_state,
                session=attended_session,
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                weight_ema=weight_ema,
                lr_scheduler=lr_scheduler,
                device=device,
                dataset_rows=effective_train_rows,
            )
            del attended_checkpoint_state
            attended_checkpoint_state = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
        attended_order = attended_progress["epoch_order"]
        attended_start_offset = int(attended_progress["next_batch_offset"])
        if attended_start_offset >= _expected_train_batches:
            raise RuntimeError("[ATTENDED_RESEARCH_CHECKPOINT_OFFSET_INVALID]")
        attended_train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=_ExactIndexSampler(
                attended_order,
                batch_offset=attended_start_offset,
                batch_size=batch_size,
            ),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        if len(attended_train_loader) != _expected_train_batches - attended_start_offset:
            raise RuntimeError("[ATTENDED_RESEARCH_REMAINING_LOADER_INVALID]")
        attended_checkpoint_index = int(attended_progress["checkpoint_index"])
        attended_complete_steps = int(
            attended_progress["complete_optimizer_steps"]
        )
        target_model_state_sha256 = _model_state_sha256(target_model)
        log.info(
            "[ATTENDED_RESEARCH_SESSION_START] directory=%s checkpoint_index=%d "
            "complete_optimizer_steps=%d batch_offset=%d target_sha256=%s "
            "exit_chunk_rows=%d authority=none",
            attended_session.directory,
            attended_checkpoint_index,
            attended_complete_steps,
            attended_start_offset,
            target_model_state_sha256,
            _ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS,
        )

        def _checkpoint_attended_step(
            *, next_batch_offset: int, complete_epoch: bool
        ) -> None:
            nonlocal attended_checkpoint_index, attended_complete_steps
            attended_checkpoint_index += 1
            attended_complete_steps += 1
            attended_session.save_checkpoint(
                model=model,
                target_model=target_model,
                optimizer=optimizer,
                weight_ema=weight_ema,
                lr_scheduler=lr_scheduler,
                device=device,
                checkpoint_index=attended_checkpoint_index,
                complete_optimizer_steps=attended_complete_steps,
                epoch_index=0,
                next_batch_offset=int(next_batch_offset),
                epoch_order=attended_order,
                complete=bool(complete_epoch),
            )
            log.info(
                "[ATTENDED_RESEARCH_CHECKPOINT] directory=%s checkpoint_index=%d "
                "complete_optimizer_steps=%d next_batch_offset=%d complete=%d",
                attended_session.directory,
                attended_checkpoint_index,
                attended_complete_steps,
                int(next_batch_offset),
                int(bool(complete_epoch)),
            )

        attended_supervision = {name: False for name in JOINT_TASK_NAMES}
        attended_gradients = {name: False for name in JOINT_TASK_NAMES}
        _attended_loss, attended_stats, attended_epoch_complete = train_epoch(
            model,
            target_model,
            attended_train_loader,
            optimizer,
            device,
            grad_accum_steps=1,
            task_supervision_observed=attended_supervision,
            task_gradient_observed=attended_gradients,
            weight_ema=weight_ema,
            attended_batch_offset=attended_start_offset,
            attended_max_optimizer_steps=attended_max_optimizer_steps,
            attended_checkpoint_hook=_checkpoint_attended_step,
            attended_exit_action_forward_chunk_rows=(
                _ATTENDED_RESEARCH_UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS
            ),
        )
        log.info(
            "[ATTENDED_RESEARCH_SESSION_DONE] directory=%s epoch_complete=%d "
            "stats=%s authority=none bundle_written=0 validation_run=0",
            attended_session.directory,
            int(attended_epoch_complete),
            json.dumps(attended_stats, sort_keys=True, default=_train_json_default),
        )
        return

    best_state = None
    best_val = float("inf")
    best_policy_pnl = float("-inf")
    best_unique_target_action_agreement = float("-inf")
    best_unified_exit_validation: Dict[str, Any] = {}
    best_unified_exit_full_trajectory_validation: Dict[str, Any] = {}
    best_unified_exit_fitted_q_state: Dict[str, Any] = {}
    best_entry_fitted_q_state: Dict[str, Any] = {}
    best_fitted_q_target_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    epochs_since_improve = 0
    last_epoch = 0
    last_val_stats: Dict[str, Any] = {}
    early_stopped = False
    joint_task_supervision_observed = {name: False for name in JOINT_TASK_NAMES}
    joint_task_gradient_observed = {name: False for name in JOINT_TASK_NAMES}
    _ckpt_monitor = ENTRY_CKPT_MONITOR
    log.info(
        "[CKPT_MONITOR] selecting best checkpoint on %s "
        "entry_action_q_is_sole_authority=1 "
        "target_action_agreement_affects_checkpoint_score=0",
        _ckpt_monitor,
    )

    for epoch in range(epochs):
        last_epoch = epoch + 1
        # One immutable fitted-Q target snapshot per declared iteration.  It is
        # copied before any optimizer step in this epoch, never updated from
        # VAL/TEST, and all Bellman targets are stop-gradient outputs from it.
        target_model = copy.deepcopy(model).to(device)
        target_model.requires_grad_(False)
        target_model.eval()
        target_model_state_sha256 = _model_state_sha256(target_model)
        fitted_q_iteration_state = {
            "schema_version": "gx1_unified_exit_fitted_q_iteration_state_v1",
            "iteration_index": int(epoch),
            "target_model_state_sha256": target_model_state_sha256,
            "train_split_sha256": _sha256_file(Path(train_parquet)),
            "train_fold_sha256": unified_exit_lifecycle_evidence["splits"][
                "train"
            ]["lifecycle_manifest_sha256"],
            "source_lineage_sha256": unified_exit_lifecycle_evidence[
                "root_manifest_sha256"
            ],
            "normalization_sha256": input_normalization["contract_sha256"],
            "fitted_q_contract": unified_exit_fitted_q_contract(),
            "target_updated_from_val_or_test": False,
        }
        entry_fitted_q_iteration_state = {
            "schema_version": ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
            "iteration_index": int(epoch),
            "entry_target_model_state_sha256": target_model_state_sha256,
            "exit_target_model_state_sha256": target_model_state_sha256,
            "exit_fitted_q_iteration_state_sha256": canonical_json_sha256(
                fitted_q_iteration_state
            ),
            "train_split_sha256": fitted_q_iteration_state[
                "train_split_sha256"
            ],
            "train_fold_sha256": fitted_q_iteration_state["train_fold_sha256"],
            "source_lineage_sha256": fitted_q_iteration_state[
                "source_lineage_sha256"
            ],
            "normalization_sha256": fitted_q_iteration_state[
                "normalization_sha256"
            ],
            "entry_fitted_q_contract": entry_fitted_q_contract(),
            "exit_fitted_q_contract": unified_exit_fitted_q_contract(),
            "target_updated_from_val_or_test": False,
        }
        require_entry_fitted_q_iteration_state(
            entry_fitted_q_iteration_state,
            exit_fitted_q_iteration_state=fitted_q_iteration_state,
            context="ENTRY_TRAIN",
        )
        log.info(
            "[UNIFIED_EXIT_FITTED_Q_ITERATION] iteration=%d target_sha256=%s",
            int(epoch),
            target_model_state_sha256,
        )
        # Specialist-branch opening observability. `specialist_out` is
        # zero-initialized, so the eight-specialist evidence path contributes
        # nothing and passes no gradient upstream until this weight grows away
        # from zero. Checkpoint admission and serve-parity both require the
        # specialists to influence class margins, so a branch that stays closed
        # is a fail-closed condition that must be visible per epoch rather than
        # inferred from red slice metrics.
        # Gradient reaching the upstream specialist blocks is implied exactly by
        # a non-zero weight here: the backward path is W-transpose times the
        # incoming gradient, so a zero weight passes zero and any non-zero
        # weight passes signal. Reporting live gradient norms at this point
        # would be misleading because the previous epoch already cleared them,
        # so the weight state is the honest and sufficient measurement.
        _spec_w = model.specialist_out.weight.detach()
        _spec_upstream_tensors = sum(
            1
            for name, _param in model.named_parameters()
            if name.split(".")[0]
            in (
                "specialist_encoder",
                "specialist_proj",
                "specialist_cross_attn",
                "specialist_gate",
                "specialist_token_gate",
                "specialist_token_identity",
            )
        )
        log.info(
            "[ENTRY_SPECIALIST_BRANCH_HEALTH] epoch=%d specialist_out_weight_norm=%.6e "
            "specialist_out_nonzero=%d/%d upstream_tensors_gated=%d branch_open=%d",
            epoch + 1,
            float(_spec_w.norm()),
            int((_spec_w != 0).sum()),
            int(_spec_w.numel()),
            _spec_upstream_tensors,
            int(bool(float(_spec_w.norm()) > 0.0)),
        )
        tr_loss, tr_stats, tr_epoch_complete = train_epoch(
            model,
            target_model,
            train_loader,
            optimizer,
            device,
            grad_accum_steps=int(grad_accum_steps),
            task_supervision_observed=joint_task_supervision_observed,
            task_gradient_observed=joint_task_gradient_observed,
            weight_ema=weight_ema,
        )
        if not tr_epoch_complete:
            raise RuntimeError("[ENTRY_CANONICAL_TRAIN_EPOCH_PARTIAL_FORBIDDEN]")
        # V30 package 5: the LR schedule advances once per epoch, AFTER that
        # epoch's training, so epoch 0 trains at the declared `lr` exactly as
        # before. At the OFF switch this is a no-op branch.
        if lr_scheduler is not None:
            lr_scheduler.step()
            log.info(
                "[TRAIN_LR_SCHEDULE] epoch=%d next_lr=%.8g",
                epoch + 1,
                float(optimizer.param_groups[0]["lr"]),
            )

        def _validate_current_weights():
            return validate(
                model,
                target_model,
                val_loader,
                device,
                collect_full_exit_trajectory=(profile == "candidate"),
            )

        # V30 package 5: when the EMA is active the checkpoint gate must judge
        # the weights it will actually ship, so validation runs ON the averaged
        # weights and the captured `best_state` below is the EMA state. When it
        # is off the raw model is validated, unchanged.
        if weight_ema is not None:
            with weight_ema.evaluating(model):
                va_loss, auc, acc, val_short_to_long, val_stats = (
                    _validate_current_weights()
                )
        else:
            va_loss, auc, acc, val_short_to_long, val_stats = (
                _validate_current_weights()
            )
        last_val_stats = dict(val_stats or {})
        auc_display = "DISABLED" if not np.isfinite(auc) else f"{auc:.4f}"
        log.info(
            f"[EPOCH {epoch+1}/{epochs}] "
            f"train={tr_loss:.6f} val={va_loss:.6f} auc={auc_display} acc={acc:.4f} "
            f"short_to_long_val={val_short_to_long:.6f}"
        )
        active_head_details = (
            val_stats.get("active_head_diagnostics", {})
            if isinstance(val_stats, dict)
            else {}
        )
        for head_name in MODEL_NATIVE_ACTIVE_HEADS:
            details = (
                active_head_details.get(head_name, {})
                if isinstance(active_head_details, dict)
                else {}
            )
            components = details.get("components", {}) if isinstance(details, dict) else {}
            prediction_ranges = [
                float(component.get("prediction_min_range", 0.0))
                for component in components.values()
                if isinstance(component, dict)
            ]
            target_ranges = [
                float(component.get("target_min_range", 0.0))
                for component in components.values()
                if isinstance(component, dict)
            ]
            log.info(
                "[ENTRY_ACTIVE_HEAD_HEALTH] split=val epoch=%d head=%s ok=%d "
                "components=%d prediction_min_range=%.12g target_min_range=%.12g "
                "class_centered_influence_max_abs=%.12g",
                epoch + 1,
                head_name,
                int(bool(details.get("ok", False))) if isinstance(details, dict) else 0,
                len(components),
                min(prediction_ranges) if prediction_ranges else 0.0,
                min(target_ranges) if target_ranges else 0.0,
                float(details.get("influence_class_centered_max_abs", 0.0))
                if isinstance(details, dict)
                else 0.0,
            )
        if val_stats:
            log.info(
                "[ENTRY_LOSS_SUMMARY] split=val epoch=%d entry_q_mse=%.6f "
                "side_mae_l1=%.6f trendline_event_bce=%.6f total=%.6f",
                epoch + 1,
                float(val_stats.get("entry_action_q_raw_bps_mse_mean", 0.0)),
                float(val_stats.get("side_mae_loss_mean", 0.0)),
                float(val_stats.get("trendline_event_loss_mean", 0.0)),
                float(va_loss),
            )
            log.info(
                "[ENTRY_COOPERATION_GATE_HEALTH] split=val epoch=%d ok=%d "
                "specialist_entropy=%.6f specialist_min_mean=%.6f "
                "tf_entropy=%.6f tf_min_mean=%.6f "
                "family_tf_entropy=%.6f family_tf_min_mean=%.6f",
                epoch + 1,
                int(bool(val_stats.get("cooperation_gate_health_ok", False))),
                float(val_stats.get("specialist_gate_entropy_mean", 0.0)),
                float(val_stats.get("specialist_gate_min_mean", 0.0)),
                float(val_stats.get("tf_gate_entropy_mean", 0.0)),
                float(val_stats.get("tf_gate_min_mean", 0.0)),
                float(
                    val_stats.get(
                        "family_tf_cooperation_gate_entropy_mean",
                        0.0,
                    )
                ),
                float(
                    val_stats.get(
                        "family_tf_cooperation_gate_min_mean",
                        0.0,
                    )
                ),
            )
        if tr_stats:
            log.info(
                "[ENTRY_LOSS_SUMMARY] split=train epoch=%d entry_q_mse=%.6f "
                "side_mae_l1=%.6f trendline_event_bce=%.6f total=%.6f",
                epoch + 1,
                float(tr_stats.get("entry_action_q_raw_bps_mse_mean", 0.0)),
                float(tr_stats.get("side_mae_loss_mean", 0.0)),
                float(tr_stats.get("trendline_event_loss_mean", 0.0)),
                float(tr_loss),
            )
            log.info(
                "[ENTRY_COOPERATION_GATE_EVIDENCE] split=train epoch=%d "
                "specialist_entropy=%.6f specialist_min_mean=%.6f "
                "tf_entropy=%.6f tf_min_mean=%.6f "
                "family_tf_entropy=%.6f family_tf_min_mean=%.6f",
                epoch + 1,
                float(tr_stats.get("specialist_gate_entropy_mean", 0.0)),
                float(tr_stats.get("specialist_gate_min_mean", 0.0)),
                float(tr_stats.get("tf_gate_entropy_mean", 0.0)),
                float(tr_stats.get("tf_gate_min_mean", 0.0)),
                float(
                    tr_stats.get(
                        "family_tf_cooperation_gate_entropy_mean",
                        0.0,
                    )
                ),
                float(
                    tr_stats.get(
                        "family_tf_cooperation_gate_min_mean",
                        0.0,
                    )
                ),
            )
        _policy_pnl = float(
            val_stats.get(
                "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean",
                float("nan"),
            )
        )
        _improved = np.isfinite(_policy_pnl) and (
            _policy_pnl - best_policy_pnl
        ) > float(early_stopping_min_delta)
        _active_head_health_ok = bool(
            val_stats.get("active_head_health_ok", False)
        ) if val_stats else False
        _cooperation_gate_health_ok = bool(
            val_stats.get("cooperation_gate_health_ok", False)
        ) if val_stats else False
        _exit_cooperation_gate_health_ok = bool(
            val_stats.get("exit_cooperation_gate_health_ok", False)
        ) if val_stats else False
        _admission_ok = _checkpoint_admission_ok(
            profile=profile,
            active_head_health_ok=_active_head_health_ok,
            cooperation_gate_health_ok=_cooperation_gate_health_ok,
            exit_cooperation_gate_health_ok=(
                _exit_cooperation_gate_health_ok
            ),
        )
        if _improved and not _admission_ok:
            log.info(
                "[ENTRY_CHECKPOINT_ADMISSION_BLOCKED] epoch=%d profile=%s "
                "active_head_health_ok=%d "
                "cooperation_gate_health_ok=%d "
                "exit_cooperation_gate_health_ok=%d",
                epoch + 1,
                profile,
                int(_active_head_health_ok),
                int(_cooperation_gate_health_ok),
                int(_exit_cooperation_gate_health_ok),
            )
        _improved = bool(_improved and _admission_ok)
        if _improved:
            best_val = va_loss
            best_policy_pnl = _policy_pnl
            if np.isfinite(acc):
                best_unique_target_action_agreement = acc
            best_unified_exit_validation = {
                key: val_stats[key]
                for key in val_stats
                if key.startswith("unified_exit_")
                or key.startswith("exit_")
            }
            if profile == "candidate":
                full_trajectory = val_stats.get(
                    "unified_exit_full_trajectory_validation"
                )
                if not isinstance(full_trajectory, Mapping):
                    raise RuntimeError(
                        "[UNIFIED_EXIT_SELECTED_CHECKPOINT_FULL_VAL_MISSING]"
                    )
                best_unified_exit_full_trajectory_validation = dict(
                    full_trajectory
                )
            best_unified_exit_fitted_q_state = dict(fitted_q_iteration_state)
            best_entry_fitted_q_state = dict(entry_fitted_q_iteration_state)
            best_fitted_q_target_state = {
                key: value.detach().cpu().clone()
                for key, value in target_model.state_dict().items()
            }
            # V30 package 5: capture exactly the weights the gate just judged.
            # With the EMA active that is the averaged state (validation above
            # ran on it); with the EMA off it is the raw training state, an
            # unchanged expression.
            best_state = (
                weight_ema.state_dict_clone()
                if weight_ema is not None
                else {k: v.cpu().clone() for k, v in model.state_dict().items()}
            )
            best_epoch = epoch + 1
            epochs_since_improve = 0
            log.info(
                "[BEST_CHECKPOINT] epoch=%d val=%.6f "
                "entry_policy_pnl_bps=%.6f unique_target_action_agreement=%.6f "
                "target_action_agreement_affects_checkpoint_score=0 "
                "active_head_health_ok=%d "
                "monitor=%s",
                best_epoch,
                best_val,
                best_policy_pnl,
                acc,
                int(bool(_active_head_health_ok)),
                _ckpt_monitor,
            )
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= int(early_stopping_patience):
                early_stopped = True
                log.info(
                    "[EARLY_STOP] epoch=%d best_epoch=%d "
                    "best_entry_policy_pnl_bps=%.6f patience=%d min_delta=%.6f",
                    epoch + 1,
                    best_epoch,
                    best_policy_pnl,
                    int(early_stopping_patience),
                    float(early_stopping_min_delta),
                )
                break

    if best_state is None or best_fitted_q_target_state is None:
        intended_out_bundle_dir = _resolve_train_out_bundle_dir(
            out_bundle_dir,
            gx1_data_override,
        )
        evidence_path = _write_checkpoint_failure_evidence(
            intended_out_bundle_dir,
            {
                "schema_version": "entry_checkpoint_admission_failure_evidence_v1",
                "created_at_utc": _utc_now(),
                "decision": "FAIL_NO_ADMISSIBLE_CHECKPOINT",
                "failure_code": "TRAIN_FAIL_NO_BEST_STATE",
                "reason": (
                    "No validation checkpoint satisfied the profile admission "
                    "contract; no model bundle was written."
                ),
                "profile": str(profile),
                "run_id": str(run_id or ""),
                "git_commit": _git_commit(),
                "intended_out_bundle_dir": str(intended_out_bundle_dir),
                "train_data": str(train_parquet),
                "val_data": str(val_parquet),
                "train_data_sha256": _sha256_file(Path(train_parquet)),
                "val_data_sha256": _sha256_file(Path(val_parquet)),
                "best_epoch": int(best_epoch),
                "last_epoch": int(last_epoch),
                "epochs": int(epochs),
                "early_stopped": bool(early_stopped),
                "best_unique_target_action_agreement": (
                    float(best_unique_target_action_agreement)
                    if np.isfinite(best_unique_target_action_agreement)
                    else None
                ),
                "last_validation": last_val_stats,
                "trainer_cli": {
                    "seed": int(seed),
                    "device": str(device),
                    "batch_size": int(batch_size),
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "seq_len": int(seq_len),
                    "early_stopping_patience": int(early_stopping_patience),
                    "early_stopping_min_delta": float(early_stopping_min_delta),
                    "subsample_rows": int(subsample_rows),
                    "multi_tf_num_layers": int(multi_tf_num_layers),
                    "specialist_num_layers": int(specialist_num_layers),
                    "multi_tf_scale": float(multi_tf_scale),
                    "specialist_fusion_scale": float(specialist_fusion_scale),
                    "cross_family_fusion_scale": float(cross_family_fusion_scale),
                    "grad_accum_steps": int(grad_accum_steps),
                },
                "trainer_env": dict(MODEL_NATIVE_RECIPE_ENV),
            },
        )
        log.error("[ENTRY_CHECKPOINT_ADMISSION_FAILURE_EVIDENCE] path=%s", evidence_path)
        raise RuntimeError(
            "[TRAIN_FAIL_NO_BEST_STATE] no validation checkpoint satisfied "
            "profile admission; failure evidence was written and bundle creation "
            "is refused"
        )
    selected_joint_task_log_variances: dict[str, float] = {}
    for task_name in JOINT_TASK_NAMES:
        state_key = f"task_log_variances.{task_name}"
        state_value = best_state.get(state_key)
        if not isinstance(state_value, torch.Tensor) or state_value.numel() != 1:
            raise RuntimeError(
                "[ENTRY_SELECTED_JOINT_TASK_WEIGHT_STATE_INVALID] "
                f"task={task_name} key={state_key}"
            )
        selected_joint_task_log_variances[task_name] = float(state_value.item())
    model_native_joint_task_weighting = joint_task_weighting_metadata(
        selected_joint_task_log_variances,
        supervision_observed=joint_task_supervision_observed,
        gradient_observed=joint_task_gradient_observed,
    )
    log.info(
        "[ENTRY_JOINT_TASK_WEIGHTING_PASS] epoch=%d tasks=%d "
        "all_tasks_supervised=1 all_tasks_received_gradient=1 "
        "all_tasks_moved_from_neutral=1",
        int(best_epoch),
        len(JOINT_TASK_NAMES),
    )
    model_native_learned_component_movement = (
        _entry_fitted_q_movement_proof(
            entry_q_initial_state,
            best_state,
            selected_checkpoint_epoch=best_epoch,
        )
    )
    log.info(
        "[ENTRY_FITTED_Q_MOVEMENT_PASS] epoch=%d components=%s",
        int(best_epoch),
        model_native_learned_component_movement["component_changed"],
    )
    unified_exit_parameter_movement = _unified_exit_movement_proof(
        unified_exit_initial_state,
        best_state,
        selected_checkpoint_epoch=best_epoch,
    )
    log.info(
        "[UNIFIED_EXIT_MOVEMENT_PASS] epoch=%d components=%s",
        int(best_epoch),
        unified_exit_parameter_movement["component_max_abs_delta"],
    )
    if (
        int(
            best_unified_exit_validation.get(
                "unified_exit_q_valid_cells", 0
            )
        )
        <= 0
        or int(
            best_unified_exit_validation.get(
                "unified_exit_hold_target_greedy_rows", 0
            )
        )
        <= 0
        or int(
            best_unified_exit_validation.get(
                    "unified_exit_exit_now_target_greedy_rows",
                0,
            )
        )
        <= 0
        or not math.isfinite(
            float(
                best_unified_exit_validation.get(
                    "unified_exit_raw_bps_q_mse_mean",
                    float("nan"),
                )
            )
        )
        or float(
            best_unified_exit_validation.get(
                "unified_exit_raw_bps_q_mse_mean",
                0.0,
            )
        )
        < 0.0
    ):
        raise RuntimeError(
            "[UNIFIED_EXIT_SELECTED_CHECKPOINT_VALIDATION_INVALID] "
            f"{best_unified_exit_validation}"
        )
    if profile == "candidate":
        model.load_state_dict(best_state, strict=True)
        selected_target_model = copy.deepcopy(model).to(device)
        selected_target_model.load_state_dict(
            best_fitted_q_target_state, strict=True
        )
        selected_target_model.requires_grad_(False)
        selected_target_model.eval()
        full_trajectory_validation = dict(
            best_unified_exit_full_trajectory_validation
        )
        selected_model_state_sha256 = _model_state_sha256(model)
        selected_target_model_state_sha256 = _model_state_sha256(
            selected_target_model
        )
        if (
            full_trajectory_validation.get("schema_version")
            != _UNIFIED_EXIT_FULL_TRAJECTORY_VALIDATION_SCHEMA_VERSION
            or full_trajectory_validation.get("decision") != "PASS"
            or full_trajectory_validation.get("online_model_state_sha256")
            != selected_model_state_sha256
            or full_trajectory_validation.get("target_model_state_sha256")
            != selected_target_model_state_sha256
        ):
            raise RuntimeError(
                "[UNIFIED_EXIT_SELECTED_CHECKPOINT_FULL_VAL_STATE_MISMATCH] "
                f"report={full_trajectory_validation.get('online_model_state_sha256')}/"
                f"{full_trajectory_validation.get('target_model_state_sha256')} "
                f"selected={selected_model_state_sha256}/"
                f"{selected_target_model_state_sha256}"
            )
        # The selected target snapshot was needed only to prove the report's
        # provenance.  The input-influence audit is online-model only, so
        # release this full CUDA/CPU replica before that audit begins.
        del selected_target_model
        unified_exit_input_influence = (
            _unified_exit_input_influence_contract(
                model=model,
                loader=val_loader,
                device=device,
            )
        )
        log.info(
            "[UNIFIED_EXIT_INPUT_INFLUENCE_PASS] numeric=%d categorical=%d "
            "sample_rows=%d",
            int(unified_exit_input_influence["numeric_input_count"]),
            int(unified_exit_input_influence["categorical_input_count"]),
            int(unified_exit_input_influence["sample_count"]),
        )
        log.info(
            "[UNIFIED_EXIT_FULL_TRAJECTORY_VAL_PASS] population=%d "
            "q_valid=%d target_equivalent=%d policy_pnl_bps=%.6f "
            "fitted_q_mse=%.6f stream_sha256=%s",
            int(full_trajectory_validation["population_rows"]),
            int(full_trajectory_validation["q_valid_cells"]),
            int(full_trajectory_validation["target_equivalent_action_rows"]),
            float(full_trajectory_validation[
                "learned_policy_mean_realized_executable_pnl_bps"
            ]),
            float(full_trajectory_validation["fitted_q_bellman_mse_mean"]),
            full_trajectory_validation["state_prediction_stream_sha256"],
        )
    else:
        unified_exit_input_influence = {
            "schema_version": UNIFIED_EXIT_INPUT_INFLUENCE_SCHEMA_VERSION,
            "decision": "NOT_RUN_SMOKE_CANNOT_AUTHORIZE_CANDIDATE",
            "required_for_candidate": True,
        }
        full_trajectory_validation = {
            "schema_version": "gx1_unified_exit_full_trajectory_validation_v6",
            "decision": "NOT_RUN_SMOKE_CANNOT_AUTHORIZE_CANDIDATE",
            "required_for_candidate": True,
        }
    unified_entry_exit_contract = unified_entry_exit_contract_metadata()
    unified_exit_training_evidence = {
        "schema_version": "gx1_unified_exit_training_evidence_v10",
        "decision": "PASS",
        "shared_model_state_dict": True,
        "entry_representation_surface": (
            UNIFIED_EXIT_MODEL_REPRESENTATION_KEY
        ),
        "future_outcomes_used_as_model_inputs": False,
        "exit_action_task_name": "unified_exit_action",
        "exit_action_target": "train_fitted_raw_bps_q_iteration",
        "exit_action_loss": "mean_squared_error_over_valid_q_cells",
        "gamma": 1.0,
        "intermediate_hold_reward_bps": 0.0,
        "baseline_cross_entropy_authority": False,
        "fitted_q_contract": unified_exit_fitted_q_contract(),
        "selected_fitted_q_iteration_state": (
            best_unified_exit_fitted_q_state
        ),
        "loss_scalarization": "model_native_joint_task_weighting",
        "lifecycle": unified_exit_lifecycle_evidence,
        "selected_checkpoint_validation": best_unified_exit_validation,
        "individual_input_influence": unified_exit_input_influence,
        "full_trajectory_validation": full_trajectory_validation,
        "selected_checkpoint_parameter_movement": (
            unified_exit_parameter_movement
        ),
    }
    # Historical class-distribution and context-slice checkpoint gates are not
    # part of fitted-Q admission.  This gross research metric cannot authorize
    # serving until the explicit production-economics contract is READY.


    # Build the complete bundle in a hidden sibling directory.  The requested
    # immutable destination does not exist until every strict verification has
    # passed and Linux RENAME_NOREPLACE publishes the directory in one step.
    final_out_bundle_dir = _resolve_train_out_bundle_dir(
        out_bundle_dir,
        gx1_data_override,
    )
    final_out_bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    if (
        final_out_bundle_dir.exists()
        or final_out_bundle_dir.parent.is_symlink()
        or not final_out_bundle_dir.parent.is_dir()
    ):
        raise RuntimeError(
            "[ENTRY_BUNDLE_IMMUTABLE_DESTINATION_INVALID] "
            f"{final_out_bundle_dir}"
        )
    staging_directory = tempfile.TemporaryDirectory(
        prefix=f".{final_out_bundle_dir.name}.staging.",
        dir=final_out_bundle_dir.parent,
    )
    out_bundle_dir = Path(staging_directory.name).resolve(strict=True)

    model_path = out_bundle_dir / "model_state_dict.pt"
    torch.save(best_state, model_path)
    state_dict_sha256 = _sha256_file(model_path)
    trained_signal_names = list(train_ds.signal_names)
    trained_model_native_signal_contract = train_ds.model_native_signal_contract
    require_model_native_signal_contract(
        trained_model_native_signal_contract,
        context="ENTRY_EXPORT",
    )
    trained_model_native_state_contract = _model_native_state_contract_for_parquet(Path(train_parquet))
    state_contract_failures = _model_native_state_contract_failures(
        trained_model_native_state_contract,
        split="train",
    )
    if state_contract_failures:
        raise RuntimeError(
            "[XAU_DIRECTION_REPAIR_STATE_CONTRACT_FAIL] "
            + " | ".join(state_contract_failures)
        )
    if trained_model_native_state_contract.get("entry_run_id") != dataset_run_id:
        raise RuntimeError(
            "[ENTRY_TRAIN_EXPORT_DATASET_RUN_ID_LINEAGE_MISMATCH] "
            f"cli={dataset_run_id!r} "
            f"state={trained_model_native_state_contract.get('entry_run_id')!r}"
        )

    direction_decision_contract = model_direction_decision_contract_metadata()
    model_native_entry_fitted_q = entry_fitted_q_contract()
    entry_fitted_q_production_economics = (
        entry_fitted_q_production_economics_readiness()
    )
    run_lineage = {
        "schema_version": "entry_model_native_training_run_lineage_v2",
        "training_run_id": str(run_id),
        "dataset_run_id": str(dataset_run_id),
        "training_profile": str(profile),
        "execution_tier": str(execution_tier),
        "requested_subsample_rows": int(subsample_rows),
        "physical_train_rows": physical_train_rows,
        "effective_train_rows": effective_train_rows,
    }
    lock = {
        "version": "entry_v10_ctx_lock_v3",
        "model_architecture_schema_version": MODEL_ARCHITECTURE_SCHEMA_VERSION,
        "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "signal_bridge_id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "signal_bridge_contract_sha256": trained_model_native_signal_contract["static_contract_sha256"],
        "contract_mode": train_contract_mode,
        "direction_decision_contract": direction_decision_contract,
        "unified_entry_exit_contract": unified_entry_exit_contract,
        "unified_exit_training_evidence": unified_exit_training_evidence,
        "m1_feature_surface_binding": m1_feature_surface_binding,
        "model_native_entry_fitted_q": model_native_entry_fitted_q,
        "entry_fitted_q_production_economics": (
            entry_fitted_q_production_economics
        ),
        "selected_entry_fitted_q_iteration_state": best_entry_fitted_q_state,
        "model_native_learned_component_movement": model_native_learned_component_movement,
        "model_native_signal_contract": trained_model_native_signal_contract,
        "entry_position_size_target_policy": entry_position_size_target_policy,
        "entry_position_size_target_policy_sha256": (
            entry_position_size_target_policy["policy_sha256"]
        ),
        "context_specialist_routing": specialist_meta["context_routing"],
        "input_normalization": input_normalization,
        "input_normalization_fit_population_proof": (
            input_normalization_fit_population_proof
        ),
        "run_lineage": run_lineage,
        "execution_tier": str(execution_tier),
        "prefreeze_test_seal_lineage": prefreeze_test_seal_lineage,
        "aux_head_target_contract": train_ds.aux_head_target_contract,
        "model_native_training_objective": model_native_training_objective,
        "model_native_joint_task_weighting": model_native_joint_task_weighting,
        "ctx_tag": f"CTX{ctx_cont_dim}CAT{ctx_cat_dim}",
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ordered_ctx_cont_names": list(ordered_ctx_cont_names),
        "ordered_ctx_cat_names": list(ordered_ctx_cat_names),
        "ordered_signal_names": trained_signal_names,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "seq_len": seq_len,
        "dropout": float(dropout),
        "num_classes": 3,
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": state_dict_sha256,
    }
    # State stores unconstrained raw scalars; the immutable contract records
    # their hashes and the corresponding strictly-positive effective scales.
    learned_tf_input_scale_raw: Dict[str, float] = {}
    for _tf in TF_INPUT_SCALE_NAMES:
        _key = f"tf_input_scale_{_tf}"
        if _key not in best_state:
            raise RuntimeError(f"[TF_INPUT_SCALE_STATE_MISSING] {_key}")
        _value = float(best_state[_key].item())
        if not np.isfinite(_value):
            raise RuntimeError(f"[TF_INPUT_SCALE_STATE_NONFINITE] {_key}={_value}")
        learned_tf_input_scale_raw[_tf] = _value
    tf_input_scale_contract = build_tf_input_scale_contract(
        init_effective={
            tf: float(TF_INPUT_SCALE_NEUTRAL_INIT)
            for tf in TF_INPUT_SCALE_NAMES
        },
        learned_raw=learned_tf_input_scale_raw,
    )
    log.info(
        "[TF_INPUT_SCALE_LEARNED] %s",
        {
            k: round(float(v), 4)
            for k, v in tf_input_scale_contract["learned"].items()
        },
    )

    active_heads = _build_active_head_names()

    meta = {
        "schema_version": "entry_v10_ctx_bundle_metadata_v3",
        "model_architecture_schema_version": MODEL_ARCHITECTURE_SCHEMA_VERSION,
        "model_output_schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "execution_tier": str(execution_tier),
        "model_native_training_objective": model_native_training_objective,
        "model_native_joint_task_weighting": model_native_joint_task_weighting,
        "unified_entry_exit_contract": unified_entry_exit_contract,
        "unified_exit_training_evidence": unified_exit_training_evidence,
        "m1_feature_surface_binding": m1_feature_surface_binding,
        "model_native_entry_fitted_q": model_native_entry_fitted_q,
        "entry_fitted_q_production_economics": (
            entry_fitted_q_production_economics
        ),
        "selected_entry_fitted_q_iteration_state": best_entry_fitted_q_state,
        "model_native_learned_component_movement": model_native_learned_component_movement,
        "context_specialist_routing": specialist_meta["context_routing"],
        "input_normalization": input_normalization,
        "input_normalization_fit_population_proof": (
            input_normalization_fit_population_proof
        ),
        "train_data": str(train_parquet),
        "val_data": str(val_parquet),
        "train_data_sha256": _sha256_file(Path(train_parquet)),
        "val_data_sha256": _sha256_file(Path(val_parquet)),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "best_entry_policy_realized_gross_spread_inclusive_pnl_bps": (
            best_policy_pnl
        ),
        "ckpt_monitor": _ckpt_monitor,
        "best_unique_target_action_agreement": (
            float(best_unique_target_action_agreement)
            if np.isfinite(best_unique_target_action_agreement)
            else None
        ),
        "entry_action_q_loss": "masked_raw_bps_mean_squared_error",
        "main_direction_loss": "retired",
        "mtf_direction_loss": "retired",
        "target_action_agreement_affects_checkpoint_score": False,
        "last_epoch": last_epoch,
        "early_stopped": bool(early_stopped),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "epochs": epochs,
        "lr": lr,
        # Exact offline-only shared V4 MTF contract. Entry and Exit use one
        # cache and one encoder, with route-specific clocks and TF sets.
        "multi_tf": {
            "enabled": True,
            "v4_mode": True,
            "route_schema_version": "entry_exit_shared_mtf_routes_v1",
            "entry_route_timeframes": list(ENTRY_MTF_CONTEXT_TIMEFRAMES),
            "exit_route_timeframes": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
            "entry_target_availability_shift_minutes": (
                ENTRY_DECISION_BAR_SECONDS / 60.0
            ),
            "exit_target_availability_shift_minutes": (
                EXIT_DECISION_BAR_SECONDS / 60.0
            ),
            "entry_tf_gate_width": ENTRY_MTF_CONTEXT_COUNT,
            "exit_tf_gate_width": EXIT_MTF_CONTEXT_COUNT,
            "entry_family_tf_gate_width": (
                ENTRY_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS)
            ),
            "exit_family_tf_gate_width": (
                EXIT_MTF_CONTEXT_COUNT * len(MODEL_NATIVE_TRAINING_SPECIALISTS)
            ),
            "shared_cache_identity_sha256": (
                train_ds._multi_tf_cache_identity_sha256
            ),
            "shared_cache_manifest_sha256": (
                train_ds._multi_tf_cache_manifest_sha256
            ),
            "shared_cache_dir": train_ds._multi_tf_cache_dir,
            "shared_cache_manifest_path": (
                train_ds._multi_tf_cache_manifest_path
            ),
            "shared_cache_m5_source": train_ds._multi_tf_cache_m5_source,
            "shared_cache_m5_source_sha256": (
                train_ds._multi_tf_cache_m5_source_sha256
            ),
            "m5_seq_dim": int(_mtf_feat_count),
            "m5_seq_len": int(_m5_len),
            "m15_seq_dim": int(_mtf_feat_count),
            "h1_seq_dim": int(_mtf_feat_count),
            "h4_seq_dim": int(_mtf_feat_count),
            "d1_seq_dim": int(_mtf_feat_count),
            "m15_seq_len": int(_m15_len),
            "h1_seq_len": int(_h1_len),
            "h4_seq_len": int(_h4_len),
            "d1_seq_len": int(_d1_len),
            "multi_tf_num_layers": int(multi_tf_num_layers),
            "multi_tf_scale": float(multi_tf_scale),
            "feature_contract": str(train_ds._multi_tf_contract),
            # What live reads back must be the surface this run actually trained
            # on, not whichever contract this module imports (rule 6).
            "matrix_contract": str(train_ds._multi_tf_contract),
            "feature_names": list(train_ds._multi_tf_feature_names),
            "feature_names_sha256": hashlib.sha256(
                json.dumps(
                    list(train_ds._multi_tf_feature_names),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "closed_bar_target_availability": True,
            "resolution_pyramid": multi_tf_resolution_pyramid,
            "decision_window_coverage": (
                multi_tf_decision_window_coverage
            ),
            "specialist_routing_schema_version": (
                MULTI_TF_SPECIALIST_ROUTING_SCHEMA_VERSION
            ),
            "specialist_input_indices": multi_tf_specialist_indices,
            "parameter_family_tf_token_order": list(model.family_tf_token_order),
            "entry_family_tf_token_order": list(
                model.entry_family_tf_token_order
            ),
            "exit_family_tf_token_order": list(model.exit_family_tf_token_order),
        },
        # All five scale parameters begin at the same contract-owned neutral
        # identity. Learned state is immutable evidence; there is no per-TF
        # wrapper prior.
        "tf_input_scale": tf_input_scale_contract,
        # Positional encoding marker — buffer is persistent=False (not in
        # state_dict), so the live bundle loader MUST read this to rebuild the
        # model with matching forward behaviour.
        "enable_pos_enc": True,
        "enable_regime_film": False,
        "enable_mtf_direction_head": False,
        "batch_size": batch_size,
        "seed": seed,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "ordered_signal_names": trained_signal_names,
        "contract_mode": train_contract_mode,
        "direction_decision_contract": direction_decision_contract,
        "model_native_signal_contract": trained_model_native_signal_contract,
        "entry_position_size_target_policy": entry_position_size_target_policy,
        "entry_position_size_target_policy_sha256": (
            entry_position_size_target_policy["policy_sha256"]
        ),
        "run_lineage": run_lineage,
        "prefreeze_test_seal_lineage": prefreeze_test_seal_lineage,
        "aux_head_target_contract": train_ds.aux_head_target_contract,
        "model_native_state_contract": trained_model_native_state_contract,
        "seq_len": seq_len,
        "dropout": float(dropout),
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ordered_ctx_cont_names": list(ordered_ctx_cont_names),
        "ordered_ctx_cat_names": list(ordered_ctx_cat_names),
        "num_classes": 3,
        "expected_ctx_cont_dim": ctx_cont_dim,
        "expected_ctx_cat_dim": ctx_cat_dim,
        "supports_context_features": True,
        "signal_bridge_id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "ctx_tag": f"CTX{ctx_cont_dim}CAT{ctx_cat_dim}",
        "model_class": "EntryV10CtxHybridTransformer",
        "arch_id": MODEL_ARCHITECTURE_SCHEMA_VERSION,
        "specialist_fusion": {
            **specialist_meta,
            "num_layers": int(specialist_num_layers),
            "fusion_scale": float(specialist_fusion_scale),
            "cross_family_fusion_scale": float(cross_family_fusion_scale),
        },
        "state_dict_sha256": state_dict_sha256,
        "anchored_entry_enabled": False,
        "anchor_source": None,
        "anchor_gate": {"enabled": False},
        "side_mae_head": {
            "enabled": True,
            "output_dim": 2,
            "unit": "raw_bps",
            "labels": [
                "y_long_expected_mae_bps",
                "y_short_expected_mae_bps",
            ],
            "joint_task_name": "side_mae_bps",
            "entry_action_authority": False,
        },
        "trendline_event_head": {
            "enabled": True,
            "output_dim": 4,
            "labels": [
                "y_line_support_touch_held",
                "y_line_resistance_touch_held",
                "y_countertrend_short_trap",
                "y_countertrend_long_trap",
            ],
            "loss_masks": {
                "y_line_support_touch_held": "y_line_support_touch_mask",
                "y_line_resistance_touch_held": "y_line_resistance_touch_mask",
            },
            "joint_task_name": "trendline_event",
            "direction_mapping": "representation_only_no_entry_authority",
            "hand_written_direction_pressure": False,
        },
        # V30 package 5 stability dampers. `train_lr_cosine_decay` selects the
        # library cosine anneal over the declared epoch budget (0 = the fixed
        # LR); `train_weight_ema_decay` = 0.0 means no weight averaging took
        # place and the shipped weights are the raw training weights. V30
        # package 6: the decay is DERIVED from this run's declared budget, so
        # the complete derivation (declared token, row budget, batch geometry,
        # steps per epoch) is recorded next to it — the number alone would not
        # let a reader reproduce it.
        "train_lr_cosine_decay": int(ENTRY_TRAIN_LR_COSINE_DECAY),
        "train_weight_ema_decay": float(train_weight_ema_decay),
        "train_weight_ema_derivation": dict(weight_ema_derivation),
        "gate_gradient_forcing": False,
        "rank_losses": False,
        "composite_losses": False,
        "target_regression_units": "raw_native_units",
        "grad_clip_norm": float(_GRAD_CLIP_NORM),
        "weight_decay": float(_WEIGHT_DECAY),
        "train_recipe": {
            "entry_action_q_loss": "masked_raw_bps_mean_squared_error",
            "main_direction_loss": "retired",
            "mtf_direction_loss": "retired",
            "joint_task_weighting": model_native_joint_task_weighting,
            "ckpt_monitor": str(_ckpt_monitor),
            "side_mae_head_enabled": True,
            "side_mae_output_dim": 2,
            "side_mae_unit": "raw_bps",
            "trendline_event_head_enabled": True,
            "trendline_event_output_dim": 4,
            "trendline_event_joint_task_name": "trendline_event",
            "threshold_derived_auxiliary_heads_enabled": False,
            "fixed_relative_task_weights": False,
            "gate_gradient_forcing": False,
            "rank_losses": False,
            "validation_objective_matches_train": True,
            "validation_objective_scope_note": (
                "training validation uses the same joint exact-label task family; routing "
                "gates are empirical liveness evidence only and rank/composite losses are absent."
            ),
            "active_heads": active_heads,
        },
    }
    if sequence_source_reconstruction_evidence is not None:
        # This records the exact storage-reconstruction proof for every
        # profile. It is an explicit denial of candidate and deployment
        # authority by itself, never a substitute for an OOS result.
        lock["sequence_source_reconstruction"] = (
            sequence_source_reconstruction_evidence
        )
        meta["sequence_source_reconstruction"] = (
            sequence_source_reconstruction_evidence
        )
    # Architecture reconstruction fields are duplicated exactly in the lock;
    # neither side may infer MTF layout or positive-scale semantics from the
    # other.
    lock["multi_tf"] = meta["multi_tf"]
    lock["tf_input_scale"] = meta["tf_input_scale"]
    export_contract_failures = _direction_decision_contract_export_failures(lock, meta)
    export_contract_failures.extend(
        _unified_exit_export_failures(lock, meta)
    )
    if export_contract_failures:
        raise RuntimeError(
            "[ENTRY_EXPORT_DIRECTION_DECISION_CONTRACT_INVALID] "
            + "; ".join(export_contract_failures)
        )
    (out_bundle_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(
        json.dumps(lock, indent=2)
    )
    (out_bundle_dir / "bundle_metadata.json").write_text(json.dumps(meta, indent=2))
    for artifact_name in BUNDLE_COMMIT_CORE_ARTIFACTS:
        _fsync_regular_file(out_bundle_dir / artifact_name)
    write_bundle_commit_manifest(
        bundle_dir=out_bundle_dir,
        artifact_names=BUNDLE_COMMIT_CORE_ARTIFACTS,
        bundle_kind="trained",
        created_at_utc=str(meta["created_at_utc"]),
    )
    stage_fd = os.open(
        out_bundle_dir,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        os.fsync(stage_fd)
    finally:
        os.close(stage_fd)

    # Post-export verify: reconstruct the one exact architecture and strict-load.
    model2 = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        dropout=float(dropout),
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        m15_seq_dim=_mtf_feat_count,
        h1_seq_dim=_mtf_feat_count,
        h4_seq_dim=_mtf_feat_count,
        d1_seq_dim=_mtf_feat_count,
        m15_seq_len=_m15_len,
        h1_seq_len=_h1_len,
        h4_seq_len=_h4_len,
        d1_seq_len=_d1_len,
        m5_seq_dim=_mtf_feat_count,
        m5_seq_len=_m5_len,
        multi_tf_num_layers=int(multi_tf_num_layers),
        multi_tf_scale=float(multi_tf_scale),
        specialist_input_indices=specialist_indices,
        specialist_ctx_cont_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"][
                "ctx_cont_indices"
            ].items()
        },
        specialist_ctx_cont_nominal_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"][
                "ctx_cont_nominal_indices"
            ].items()
        },
        specialist_ctx_cat_indices={
            str(name): list(values)
            for name, values in specialist_meta["context_routing"][
                "ctx_cat_indices"
            ].items()
        },
        multi_tf_specialist_input_indices=multi_tf_specialist_indices,
        temporal_alias_signal_indices=list(
            specialist_meta["context_routing"]["temporal_alias_policy"][
                "signal_indices"
            ]
        ),
        temporal_alias_ctx_cont_indices=list(
            specialist_meta["context_routing"]["temporal_alias_policy"][
                "ctx_cont_indices"
            ]
        ),
        specialist_num_layers=int(specialist_num_layers),
        specialist_fusion_scale=float(specialist_fusion_scale),
        cross_family_fusion_scale=float(cross_family_fusion_scale),
        input_normalization=input_normalization,
    )
    model2.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model2.require_input_normalization_state()
    model2.eval()
    with torch.no_grad():
        B = 2
        signal_center = torch.tensor(
            input_normalization["surfaces"]["signal"]["center"],
            dtype=torch.float32,
        )
        ctx_cont_center = torch.tensor(
            input_normalization["surfaces"]["ctx_cont"]["center"],
            dtype=torch.float32,
        )
        dummy_seq = signal_center.view(1, 1, -1).repeat(B, seq_len, 1)
        dummy_snap = signal_center.view(1, -1).repeat(B, 1)
        dummy_cat = torch.zeros(B, ctx_cat_dim, dtype=torch.long)
        dummy_cont = ctx_cont_center.view(1, -1).repeat(B, 1)
        entry_mtf_kwargs = {
            f"seq_{tf.lower()}": torch.tensor(
                input_normalization["surfaces"][f"mtf_{tf.lower()}"][
                    "center"
                ],
                dtype=torch.float32,
            ).view(1, 1, -1).repeat(
                B,
                normalization_per_tf_seq_lens[tf],
                1,
            )
            for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES
        }
        entry_out = model2(
            dummy_seq,
            dummy_snap,
            ctx_cat=dummy_cat,
            ctx_cont=dummy_cont,
            **entry_mtf_kwargs,
        )
        shared_entry = entry_out.get(UNIFIED_EXIT_MODEL_REPRESENTATION_KEY)
        if not isinstance(shared_entry, torch.Tensor):
            raise RuntimeError(
                "[ENTRY_BUNDLE_STAGE_EXIT_SHARED_REPRESENTATION_MISSING]"
            )
        exit_mtf_rows = {
            tf.lower(): torch.tensor(
                input_normalization["surfaces"][f"mtf_{tf.lower()}"][
                    "center"
                ],
                dtype=torch.float32,
            ).view(1, 1, -1).repeat(
                B,
                normalization_per_tf_seq_lens[tf],
                1,
            )
            for tf in EXIT_MTF_CONTEXT_TIMEFRAMES
        }
        _, restored_carry = model2.forward_exit_incremental_step(
            entry_decision_representation=shared_entry,
            exit_local_rows_x=signal_center.view(1, 1, -1).repeat(
                B,
                seq_len * ENTRY_EXIT_RESOLUTION_RATIO,
                1,
            ),
            exit_state_ctx_cat=dummy_cat,
            exit_state_ctx_cont=dummy_cont,
            exit_mtf_new_rows=exit_mtf_rows,
            exit_path_row_x=torch.zeros(
                B,
                2,
                UNIFIED_EXIT_PATH_FEATURE_DIM,
                dtype=torch.float32,
            ),
            carry=None,
        )
        _ = model2.export_exit_incremental_carry_tensor_state(restored_carry)
    log.info(
        "[ENTRY_BUNDLE_STAGE_VERIFIED] strict state load OK: %s",
        out_bundle_dir,
    )

    # Bundle load proof via runtime loader (strict)
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
    _ = load_entry_v10_ctx_bundle(
        bundle_dir=out_bundle_dir,
        device="cpu",
    )

    # ── ALWAYS-RUN exact model-native feature-liveness audit ───────────────────────
    # No input slicing, constant allowlist, or transient-error pass-through is
    # permitted. Both seq_x and snap_x must prove the exact owner-declared
    # signal surface; ctx_cont must prove the owner-declared field count; every MTF surface
    # must be present/live.
    try:
        from gx1.audit.feature_liveness import assert_v10_batch_liveness, FeatureLivenessError
        _live_cc = list(ordered_ctx_cont_names)
        if len(_live_cc) != MODEL_NATIVE_CTX_CONT_DIM:
            raise FeatureLivenessError(
                "[FEATURE_LIVENESS_CTX_CONTRACT] "
                f"names={len(_live_cc)} expected={MODEL_NATIVE_CTX_CONT_DIM}"
            )
        _live_ds = train_ds
        if len(_live_ds) <= 0:
            raise FeatureLivenessError("[FEATURE_LIVENESS_EMPTY_TRAIN_SPLIT]")
        _snap_names = list(getattr(_live_ds, "signal_names", ()))
        if len(_snap_names) != MODEL_NATIVE_SIGNAL_DIM:
            raise FeatureLivenessError(
                f"[FEATURE_LIVENESS_SIGNAL_CONTRACT] names={len(_snap_names)} "
                f"expected={MODEL_NATIVE_SIGNAL_DIM}"
            )
        if (
            bool(getattr(_live_ds, "_advanced", False))
            and hasattr(_live_ds, "_np_seq")
            and hasattr(_live_ds, "_np_ctx_cont")
            and hasattr(_live_ds, "_np_snap")
        ):
            _sample_rows = min(1024, len(_live_ds))
            # Sample the rows the model actually trains on. Full datasets use
            # immutable parquet row ids as storage positions; bounded smoke
            # datasets compact those selected rows to 0..N-1. The mapper below
            # owns that distinction so neither coordinate space can leak into
            # the other or turn a variable TRAIN field into a prefix-only
            # liveness verdict.
            _live_positions = np.asarray(
                getattr(_live_ds, "indices", None)
                if getattr(_live_ds, "indices", None) is not None
                else np.arange(len(_live_ds)),
                dtype=np.int64,
            )
            if _live_positions.size != len(_live_ds):
                raise RuntimeError(
                    "[FEATURE_LIVENESS_INDEX_MAP_INVALID] "
                    f"positions={_live_positions.size} dataset_len={len(_live_ds)}"
                )
            _sample_idx = _deterministic_liveness_storage_indices(
                _live_ds,
                sample_rows=_sample_rows,
            )
            if bool(getattr(_live_ds, "_sequence_source_reconstructed", False)):
                _storage_idx = np.asarray(
                    [
                        _live_ds._storage_position_for_full_row(int(row))
                        for row in _sample_idx
                    ],
                    dtype=np.int64,
                )
                _ab = {
                    "seq_x": np.stack(
                        [
                            _live_ds.sequence_for_full_row(int(row))
                            for row in _sample_idx
                        ]
                    ).astype(np.float32, copy=False),
                    "ctx_cont": np.asarray(
                        _live_ds._np_ctx_cont[_storage_idx], dtype=np.float32
                    ),
                    "snap_x": np.asarray(
                        _live_ds._np_snap[_storage_idx], dtype=np.float32
                    ),
                }
            else:
                _ab = {
                    "seq_x": np.asarray(_live_ds._np_seq[_sample_idx], dtype=np.float32),
                    "ctx_cont": np.asarray(_live_ds._np_ctx_cont[_sample_idx], dtype=np.float32),
                    "snap_x": np.asarray(_live_ds._np_snap[_sample_idx], dtype=np.float32),
                }
            if getattr(_live_ds, "_multi_tf_feats", None):
                for _tf, _feats in _live_ds._multi_tf_feats.items():
                    _arr = np.asarray(_feats.attrs.get("feats_np"), dtype=np.float32)
                    if _arr.size:
                        # Start after the causal warmup prefix the feature owner
                        # itself declares. Sampling from row 0 reaches indicator
                        # warmup, where NaN is the correct causal value and no
                        # model row ever looks; the liveness gate then reports
                        # every timeframe field as silently dead. The count is
                        # the owner's own `causal_warmup_rows`, never a guess,
                        # and non-finite values after it stay a real failure.
                        _warmup = _feats.attrs.get("causal_warmup_rows")
                        if _warmup is None:
                            raise RuntimeError(
                                "[FEATURE_LIVENESS_MTF_WARMUP_UNDECLARED] "
                                f"tf={_tf} cache does not declare causal_warmup_rows"
                            )
                        _first = int(_warmup)
                        if not 0 <= _first < int(_arr.shape[0]):
                            raise RuntimeError(
                                "[FEATURE_LIVENESS_MTF_WARMUP_INVALID] "
                                f"tf={_tf} warmup={_first} rows={int(_arr.shape[0])}"
                            )
                        _tf_rows = min(8192, int(_arr.shape[0]) - _first)
                        _tf_idx = np.linspace(
                            _first,
                            int(_arr.shape[0]) - 1,
                            num=_tf_rows,
                            dtype=np.int64,
                        )
                        _sampled_tf = _arr[_tf_idx]
                        _ab[f"seq_{str(_tf).lower()}"] = _sampled_tf.reshape(
                            1,
                            _sampled_tf.shape[0],
                            _sampled_tf.shape[1],
                        )
            log.info(
                "[FEATURE_LIVENESS] using deterministic broad sample rows=%d seq_dim=%d snap_dim=%d ctx_cont_dim=%d",
                int(_ab["ctx_cont"].shape[0]),
                int(_ab["seq_x"].shape[2]),
                int(_ab["snap_x"].shape[1]),
                int(_ab["ctx_cont"].shape[1]),
            )
        else:
            _ab = next(
                iter(
                    DataLoader(
                        _live_ds,
                        batch_size=min(1024, len(_live_ds)),
                        shuffle=True,
                        num_workers=0,
                    )
                )
            )
        # Escalation resolver. A 1024-row sample cannot decide deadness: an
        # impulse flag firing on 0.024% of rows is absent from almost every
        # sample, and a score whose natural range is [0, 0.0044] falls under
        # DEAD_STD on scale alone. So when the sample flags a field, measure
        # that one field over the complete currently materialized population
        # and let the gate rule on that. This is the full physical population
        # for candidates and the complete uniform TRAIN selection for smoke.
        # The sequence surface is ruled on the snap population by a proof from
        # source, not by assumption. The builder cuts both surfaces from ONE
        # matrix in the same column order
        # (build_entry_v10_ctx_training_dataset_v3.py:3468-3469):
        #     seq  = sig_mat[i - (seq_len - 1) : i + 1]
        #     snap = sig_mat[i]
        # so for any column j the seq population is the union of trailing
        # windows and therefore a superset of the snap population. Non-constant
        # in snap thus implies non-constant in seq, which is the only direction
        # needed to clear a sequence flag. Measuring _np_seq directly would
        # stride-read all 72.75 GB of the memmap per field for no added proof.
        _pop_arrays = {
            "signal": ("signal", _live_ds._np_snap, _snap_names),
            "signal_sequence": ("signal", _live_ds._np_snap, _snap_names),
            "ctx_cont": ("ctx_cont", _live_ds._np_ctx_cont, _live_cc),
        }
        _live_mtf_names = list(
            getattr(_live_ds, "_multi_tf_feature_names", ())
        )
        if len(_live_mtf_names) != int(
            getattr(_live_ds, "_multi_tf_feature_count", -1)
        ):
            raise FeatureLivenessError(
                "[FEATURE_LIVENESS_MTF_CONTRACT] ordered names do not match "
                "the dataset-declared width"
            )
        for _tf, _feats in getattr(_live_ds, "_multi_tf_feats", {}).items():
            _pop_arrays[f"multi_tf.{_tf}"] = (
                f"multi_tf.{_tf}",
                np.asarray(_feats.attrs.get("feats_np"), dtype=np.float32),
                _live_mtf_names,
            )
        _pop_cache: dict = {}

        def _population_stats(surface: str, name: str):
            entry = _pop_arrays.get(str(surface))
            if entry is None:
                return None
            role, arr, names = entry
            if str(name) not in names:
                return None
            key = (role, str(name))
            if key not in _pop_cache:
                col = np.asarray(
                    arr[..., names.index(str(name))], dtype=np.float64
                ).reshape(-1)
                finite = col[np.isfinite(col)]
                if finite.size == 0:
                    return None
                _pop_cache[key] = (
                    float(finite.std()),
                    int(np.unique(finite).size),
                )
            return _pop_cache[key]

        assert_v10_batch_liveness(_ab, ctx_cont_names=_live_cc,
                                  snap_names=_snap_names,
                                  multi_tf_names=_live_mtf_names,
                                  raise_on_fail=True,
                                  population_stats=_population_stats)
        if _pop_cache:
            log.info(
                "[FEATURE_LIVENESS_POPULATION_ESCALATION] %d field(s) below "
                "DEAD_STD on the sample were ruled on the full population: %s",
                len(_pop_cache),
                "; ".join(
                    f"{s}:{n} std={v[0]:.1e} nunique={v[1]}"
                    for (s, n), v in sorted(_pop_cache.items())
                ),
            )
        log.info(
            "[FEATURE_LIVENESS] post-export audit OK — exact "
            "seq513/ctx%d+%d/5x%d MTF inputs are live",
            MODEL_NATIVE_CTX_CONT_DIM,
            MODEL_NATIVE_CTX_CAT_DIM,
            len(_live_mtf_names),
        )
    except FeatureLivenessError:
        raise
    except Exception as _e:
        raise RuntimeError(
            f"[FEATURE_LIVENESS_AUDIT_UNAVAILABLE] {_e!r}"
        ) from _e
    publish_bundle_directory_noreplace(
        out_bundle_dir,
        final_out_bundle_dir,
    )
    staging_directory.cleanup()
    log.info(
        "[DONE] immutable bundle atomically published: %s",
        final_out_bundle_dir,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _install_attended_smoke_termination_handler() -> None:
    """Turn the guard's SIGTERM into normal Python unwinding for attended smoke.

    ``TemporaryDirectory`` releases the large memmap mirrors when its Dataset
    object is unwound.  Python's default SIGTERM action exits immediately and
    skips that finalizer; the bounded attended run would therefore leave an
    orphaned, regenerable scratch mirror after its wall-clock stop.  This hook
    is deliberately unavailable to canonical/candidate training and changes no
    safety threshold: the parent guard still owns termination and KILL fallback.
    """

    def _terminate(signum: int, _frame: object) -> None:
        log.warning(
            "[ATTENDED_SMOKE_GRACEFUL_TERMINATION] signal=%s; unwinding temporary scratch",
            signum,
        )
        raise KeyboardInterrupt("attended smoke stopped by the safety guard")

    signal.signal(signal.SIGTERM, _terminate)


def main() -> None:
    _require_trainer_cgroup_preflight()
    _enforce_canonical_train_env_contract()

    parser = argparse.ArgumentParser("ENTRY_V10_CTX exact model-native trainer")
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--profile", choices=("smoke", "candidate"), required=True)
    parser.add_argument(
        "--execution-tier",
        choices=("canonical", "attended_only", "attended_cpu_only"),
        default="canonical",
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--dataset-run-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, required=True, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--train-manifest-json", type=Path, required=True)
    parser.add_argument("--val-manifest-json", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--train-sequence-roll-audit-json", type=Path)
    parser.add_argument("--val-sequence-roll-audit-json", type=Path)
    parser.add_argument("--train-sequence-source-audit-json", type=Path, required=True)
    parser.add_argument("--val-sequence-source-audit-json", type=Path, required=True)
    parser.add_argument("--prefreeze-test-seal-json", type=Path, required=True)
    parser.add_argument("--prefreeze-test-seal-sha256", type=str, required=True)
    parser.add_argument(
        "--unified-exit-lifecycle-manifest-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--out_bundle_dir", type=Path, required=True)
    parser.add_argument("--gx1-data", type=str, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--early-stopping-patience", type=int, required=True)
    parser.add_argument("--early-stopping-min-delta", type=float, required=True)
    parser.add_argument("--m5-prebuilt-path", type=Path, required=True)
    parser.add_argument("--multi-tf-num-layers", type=int, required=True)
    parser.add_argument("--per-tf-seq-len-m5", type=int, required=True)
    parser.add_argument("--per-tf-seq-len-m15", type=int, required=True)
    parser.add_argument("--per-tf-seq-len-h1", type=int, required=True)
    parser.add_argument("--per-tf-seq-len-h4", type=int, required=True)
    parser.add_argument("--per-tf-seq-len-d1", type=int, required=True)
    parser.add_argument("--multi-tf-scale", type=float, required=True)
    parser.add_argument("--specialist-audit-json", type=Path, required=True)
    parser.add_argument(
        "--specialist-contract-mode",
        choices=(MODEL_NATIVE_CONTRACT_MODE,),
        required=True,
    )
    parser.add_argument("--specialist-num-layers", type=int, required=True)
    parser.add_argument("--specialist-fusion-scale", type=float, required=True)
    parser.add_argument("--cross-family-fusion-scale", type=float, required=True)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--subsample-rows", type=int, required=True)
    parser.add_argument("--grad-clip-norm", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    args = parser.parse_args()

    from gx1.contracts.entry_run_lineage_v1 import EntryRunLineageError, require_entry_run_id

    try:
        require_entry_run_id(args.run_id)
        require_entry_run_id(args.dataset_run_id)
    except EntryRunLineageError as exc:
        parser.error(str(exc))
    if args.run_id == args.dataset_run_id:
        parser.error(
            "training --run-id must differ from immutable --dataset-run-id"
        )
    if args.profile == "candidate" and int(args.subsample_rows) != 0:
        parser.error("candidate training requires --subsample-rows 0")
    if _is_attended_execution_tier(args.execution_tier) and args.profile != "smoke":
        parser.error("attended execution tiers require --profile smoke")
    if _is_attended_execution_tier(args.execution_tier) and (
        int(args.batch_size) != _ATTENDED_RESEARCH_BATCH_SIZE
        or int(args.epochs) != 1
        or int(args.grad_accum_steps) != 1
    ):
        parser.error(
            "attended execution tiers require --batch_size 8, --epochs 1 "
            "and --grad-accum-steps 1"
        )
    if args.execution_tier == "attended_only" and args.device != "cuda":
        parser.error("--execution-tier attended_only requires --device cuda")
    if args.execution_tier == "attended_cpu_only" and args.device != "cpu":
        parser.error("--execution-tier attended_cpu_only requires --device cpu")
    if any(value is not None for value in (
        args.train_sequence_roll_audit_json,
        args.val_sequence_roll_audit_json,
    )):
        parser.error("sequence-roll reconstruction proofs are retired")
    if _is_attended_execution_tier(args.execution_tier):
        _install_attended_smoke_termination_handler()

    global _GRAD_CLIP_NORM, _WEIGHT_DECAY
    _GRAD_CLIP_NORM = float(args.grad_clip_norm)
    _WEIGHT_DECAY = float(args.weight_decay)
    _guard_no_rl()
    device = _resolve_device(args.device)
    log.info(
        "[CONFIG] seed=%d device=%s deterministic=true grad_clip_norm=%.6f "
        "weight_decay=%.6f dropout=%.6f",
        args.seed,
        device,
        _GRAD_CLIP_NORM,
        _WEIGHT_DECAY,
        float(args.dropout),
    )

    _resolve_gx1_data(args.gx1_data)
    _manifests, parquets = _resolve_explicit_train_split_artifacts(
        train_manifest=args.train_manifest_json,
        val_manifest=args.val_manifest_json,
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        unified_exit_lifecycle_manifest_path=(
            args.unified_exit_lifecycle_manifest_json
        ),
        m5_prebuilt_path=args.m5_prebuilt_path,
        dataset_run_id=args.dataset_run_id,
        profile=args.profile,
    )
    train_parquet = parquets["train"]
    val_parquet = parquets["val"]
    try:
        prefreeze_test_seal_lineage = require_prefreeze_test_seal_lineage(
            args.prefreeze_test_seal_json,
            args.prefreeze_test_seal_sha256,
            expected_dataset_run_id=str(args.dataset_run_id),
            expected_dataset_dir=train_parquet.parent,
        )
    except (PrefreezeTestSealLineageError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"[ENTRY_TRAIN_PREFREEZE_TEST_SEAL_REJECTED] {exc}"
        ) from exc

    run_train(
        train_parquet=train_parquet,
        train_manifest_path=_manifests["train"],
        val_parquet=val_parquet,
        unified_exit_lifecycle_manifest_path=(
            args.unified_exit_lifecycle_manifest_json
        ),
        seq_len=args.seq_len,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        out_bundle_dir=args.out_bundle_dir,
        gx1_data_override=args.gx1_data,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        m5_prebuilt_path=args.m5_prebuilt_path,
        specialist_audit_json=args.specialist_audit_json,
        specialist_contract_mode=str(args.specialist_contract_mode),
        dropout=float(args.dropout),
        multi_tf_num_layers=int(args.multi_tf_num_layers),
        multi_tf_scale=args.multi_tf_scale,
        subsample_rows=args.subsample_rows,
        specialist_num_layers=int(args.specialist_num_layers),
        specialist_fusion_scale=float(args.specialist_fusion_scale),
        cross_family_fusion_scale=float(args.cross_family_fusion_scale),
        per_tf_seq_len_m5=int(args.per_tf_seq_len_m5),
        per_tf_seq_len_m15=int(args.per_tf_seq_len_m15),
        per_tf_seq_len_h1=int(args.per_tf_seq_len_h1),
        per_tf_seq_len_h4=int(args.per_tf_seq_len_h4),
        per_tf_seq_len_d1=int(args.per_tf_seq_len_d1),
        grad_accum_steps=int(args.grad_accum_steps),
        prefreeze_test_seal_lineage=prefreeze_test_seal_lineage,
        run_id=str(args.run_id),
        dataset_run_id=str(args.dataset_run_id),
        profile=str(args.profile),
        execution_tier=str(args.execution_tier),
        train_sequence_roll_audit_json=args.train_sequence_roll_audit_json,
        val_sequence_roll_audit_json=args.val_sequence_roll_audit_json,
        train_sequence_source_audit_json=args.train_sequence_source_audit_json,
        val_sequence_source_audit_json=args.val_sequence_source_audit_json,
    )


if __name__ == "__main__":
    main()
