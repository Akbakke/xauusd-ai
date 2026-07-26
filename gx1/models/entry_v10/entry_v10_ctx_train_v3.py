#!/usr/bin/env python3
"""
Canonical ENTRY_V10_CTX trainer.

ONE UNIVERSE (STRICT):
- Signal bridge: versioned Entry signal bridge.
- Context: exact ctx_cont/ctx_cat order from entry_model_native_signal_v1.
- Internal full-counterfactual offline-RL evidence; no separate policy/runtime
- No legacy
- No fallback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import mmap
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.metrics import accuracy_score

# Canonical context ordering; exact model-native dimensions are verified below.
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
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
from gx1.contracts.xau_tape_provenance_v1 import validate_xau_tape_provenance_v1
from gx1.contracts.entry_model_native_training_objective_v1 import (
    FIXED_POSITIVE_LOSS_WEIGHTS as _MODEL_NATIVE_FIXED_POSITIVE_LOSS_WEIGHTS,
    active_loss_weight_failures as _model_native_active_loss_weight_failures,
    training_objective_contract_metadata,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    require_model_native_recipe_env,
)
from gx1.contracts.entry_model_native_readiness_v1 import MODEL_NATIVE_ACTIVE_HEADS
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    direction_evidence_fusion_metadata,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_COUNT as OFFLINE_RL_ACTION_COUNT,
    ACTION_VALUE_DIM,
    ACTION_VALUE_TARGET_COLUMNS,
    EXPECTILE_VALUE_DIM,
    HORIZON_COUNT as OFFLINE_RL_HORIZON_COUNT,
    REWARD_SCALE_BPS as OFFLINE_RL_REWARD_SCALE_BPS,
    expectile_loss,
    q_ranking_margin_loss,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_TARGET_COLUMNS,
    MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
    MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS,
    MODEL_NATIVE_TIMING_TARGET_COLUMNS,
    MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS,
    require_model_native_aux_target_emission_contract as _require_model_native_aux_target_emission_contract,
)
from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
    COMPONENT_PARAMETERS as _EVIDENCE_FUSION_MOVEMENT_COMPONENTS,
    PARAMETER_SHAPES as _EVIDENCE_FUSION_PARAMETER_SHAPES,
    REFERENCE as _EVIDENCE_FUSION_MOVEMENT_REFERENCE,
    SCHEMA_VERSION as _EVIDENCE_FUSION_MOVEMENT_SCHEMA_VERSION,
    require_learned_component_movement_metadata,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    TF_NAMES as TF_INPUT_SCALE_NAMES,
    build_tf_input_scale_contract,
)
from gx1.models.entry_v10.entry_v10_input_normalization import (
    TrainNormalizationArtifacts,
    fit_entry_v10_train_input_normalization,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
    EXACT_EVIDENCE_FUSION_OUTPUTS,
    DIP_DIRECTIONS, DIP_HORIZONS, DIP_TARGETS, FORECAST_HORIZONS,
    TIMING_DIRECTIONS, TIMING_HORIZONS, TIMING_TARGETS,
    TAIL_RISK_DIRECTIONS, TAIL_RISK_HORIZONS, TAIL_RISK_QUANTILE,
    VOL_FORECAST_HORIZONS,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    require_model_native_context_specialist_routing,
    required_training_specialists_for_mode,
    require_model_native_specialist_contract_mode,
    specialist_model_contract_for_mode,
)


_DIP_TARGET_COLS = MODEL_NATIVE_DIP_TARGET_COLUMNS
_FORECAST_TARGET_COLS = MODEL_NATIVE_FORECAST_TARGET_COLUMNS
_TIMING_TARGET_COLS = MODEL_NATIVE_TIMING_TARGET_COLUMNS
_TAIL_RISK_TARGET_COLS = MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS
_VOL_FORECAST_TARGET_COLS = MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS
_OFFLINE_RL_TARGET_COLS = ACTION_VALUE_TARGET_COLUMNS

# Exact emitted target surface for the five active forward-path auxiliary heads.
_DIP_FORECAST_TARGET_COLS = (
    _DIP_TARGET_COLS
    + _FORECAST_TARGET_COLS
    + _TIMING_TARGET_COLS
    + _TAIL_RISK_TARGET_COLS
    + _VOL_FORECAST_TARGET_COLS
)
# The immutable parquet column is `y_direction`; the Dataset converts it once
# to the canonical class-index batch tensor `y`.  Active heads must consume
# that exact tensor rather than requiring a second alias in every batch.
_DIRECTION_BATCH_TARGET_NAMES = ("y",)

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


def dip_forecast_loss(
    out: dict,
    batch: dict,
    device,
    *,
    bps_scale: float,
) -> "torch.Tensor":
    """Loss for the exact dip/forecast/timing/tail/vol target surface."""
    normalized_bps_scale = float(bps_scale)
    if (
        not np.isfinite(normalized_bps_scale)
        or normalized_bps_scale <= 0.0
    ):
        raise RuntimeError(
            "[ENTRY_FORWARD_AUX_BPS_SCALE_INVALID] "
            f"observed={normalized_bps_scale!r}"
        )
    total = torch.zeros((), device=device)
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
    tgt = (
        torch.stack(tgts, dim=1).to(device).float()
        / normalized_bps_scale
    )  # (B, 18), model-unit scale
    q = torch.tensor(qs, device=device, dtype=tgt.dtype).view(1, -1)
    err = tgt - dip_pred.float()
    total = total + torch.maximum(q * err, (q - 1.0) * err).mean()
    fc_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="forecast_pred",
        target_names=_FORECAST_TARGET_COLS,
    )
    fc_tgt = (
        torch.stack(
            [batch[f"y_forecast_ret_K{K}"] for K in FORECAST_HORIZONS],
            dim=1,
        ).to(device).float()
        / normalized_bps_scale
    )
    total = total + torch.nn.functional.smooth_l1_loss(fc_pred.float(), fc_tgt)
    # ── dip-timing head (12, smooth_l1) — WHEN the dip bottoms / favorable peak ─
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
    total = total + torch.nn.functional.smooth_l1_loss(timing_pred.float(), t_tgt)
    # ── tail-risk head (6, pinball q=0.9) — worst adverse over full horizon ─────
    tail_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="tail_risk_pred",
        target_names=_TAIL_RISK_TARGET_COLS,
    )
    tail_tgts = [batch[f"y_tail_mae_{d}_K{K}"]
                 for d in TAIL_RISK_DIRECTIONS for K in TAIL_RISK_HORIZONS]
    tail_tgt = (
        torch.stack(tail_tgts, dim=1).to(device).float()
        / normalized_bps_scale
    )  # (B, 6), model-unit scale
    q = float(TAIL_RISK_QUANTILE)
    err = tail_tgt - tail_pred.float()
    total = total + torch.maximum(q * err, (q - 1.0) * err).mean()
    # ── vol-forecast head (3, smooth_l1) — forward realized vol (bps) ───────────
    vol_pred = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="vol_forecast_pred",
        target_names=_VOL_FORECAST_TARGET_COLS,
    )
    vol_tgt = (
        torch.stack(
            [batch[f"y_vol_fwd_K{K}"] for K in VOL_FORECAST_HORIZONS],
            dim=1,
        ).to(device).float()
        / normalized_bps_scale
    )
    total = total + torch.nn.functional.smooth_l1_loss(vol_pred.float(), vol_tgt)
    return total


def offline_rl_aux_loss(
    out: dict,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """Mandatory Q/V/ranking supervision for internal contextual-bandit evidence."""

    q_flat = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="action_value",
        target_names=_OFFLINE_RL_TARGET_COLS,
    )
    value = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="expectile_value",
        target_names=_OFFLINE_RL_TARGET_COLS,
    )
    advantage = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="action_advantage",
        target_names=_OFFLINE_RL_TARGET_COLS,
    )
    batch_size = int(q_flat.shape[0])
    if tuple(q_flat.shape) != (batch_size, ACTION_VALUE_DIM):
        raise RuntimeError("[ENTRY_OFFLINE_RL_Q_SHAPE_INVALID]")
    if tuple(value.shape) != (batch_size, EXPECTILE_VALUE_DIM):
        raise RuntimeError("[ENTRY_OFFLINE_RL_V_SHAPE_INVALID]")
    if tuple(advantage.shape) != (batch_size, ACTION_VALUE_DIM):
        raise RuntimeError("[ENTRY_OFFLINE_RL_ADVANTAGE_SHAPE_INVALID]")

    q_values = q_flat.float().reshape(
        batch_size,
        OFFLINE_RL_ACTION_COUNT,
        OFFLINE_RL_HORIZON_COUNT,
    )
    value = value.float()
    expected_advantage = (q_values - value.unsqueeze(1)).reshape(
        batch_size,
        ACTION_VALUE_DIM,
    )
    if not torch.allclose(
        advantage.float(),
        expected_advantage,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("[ENTRY_OFFLINE_RL_ADVANTAGE_CONTRACT_INVALID]")

    reward_targets = (
        torch.stack(
            [batch[name] for name in _OFFLINE_RL_TARGET_COLS],
            dim=1,
        )
        .to(device)
        .float()
        .reshape(
            batch_size,
            OFFLINE_RL_ACTION_COUNT,
            OFFLINE_RL_HORIZON_COUNT,
        )
        / float(OFFLINE_RL_REWARD_SCALE_BPS)
    )
    q_loss = nn.functional.mse_loss(q_values, reward_targets)
    detached_max_q = q_values.detach().max(dim=1).values
    value_loss = expectile_loss(detached_max_q - value)
    rank_loss = q_ranking_margin_loss(q_values, reward_targets)
    return (
        float(ENTRY_OFFLINE_RL_Q_WEIGHT) * q_loss
        + float(ENTRY_OFFLINE_RL_V_WEIGHT) * value_loss
        + float(ENTRY_OFFLINE_RL_RANK_WEIGHT) * rank_loss
    )


_MODEL_NATIVE_ACTIVE_CORE_TARGET_COLS = (
    "y_direction",
    "y_tradable",
    "mfe_first_n_bps",
    "path_quality_bps",
    "y_bad_path",
    "y_dead_negative_long",
    "y_teaser_negative_long",
    "y_hard_negative_long",
    "y_dead_negative_short",
    "y_teaser_negative_short",
    "y_hard_negative_short",
    "y_clean_edge_long",
    "y_survival_long",
    "y_selector_long_mask",
    "y_selector_short_mask",
    "y_clean_edge_bidir",
    "y_survival_bidir",
    "y_trade",
    "y_side",
    "y_side_mask",
    "y_long_path_utility_bps",
    "y_short_path_utility_bps",
    "y_long_bad_path",
    "y_short_bad_path",
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
)
_MODEL_NATIVE_ACTIVE_RAIL_TARGET_COLS = (
    "y_rising_channel_support_touch",
    "y_falling_channel_resistance_touch",
    "y_countertrend_short_trap",
    "y_countertrend_long_trap",
    "y_short_high_mae_low_mfe_early_failure",
    "y_long_high_mae_low_mfe_early_failure",
)
_MODEL_NATIVE_ACTIVE_TARGET_COLS = (
    _MODEL_NATIVE_ACTIVE_CORE_TARGET_COLS
    + ("y_tf_agreement_score", "y_position_size_target")
    + _MODEL_NATIVE_ACTIVE_RAIL_TARGET_COLS
    + _DIP_FORECAST_TARGET_COLS
    + _OFFLINE_RL_TARGET_COLS
)
_MODEL_NATIVE_BINARY_TARGET_COLS = (
    "y_tradable",
    "y_bad_path",
    "y_dead_negative_long",
    "y_teaser_negative_long",
    "y_hard_negative_long",
    "y_dead_negative_short",
    "y_teaser_negative_short",
    "y_hard_negative_short",
    "y_clean_edge_long",
    "y_survival_long",
    "y_selector_long_mask",
    "y_selector_short_mask",
    "y_clean_edge_bidir",
    "y_survival_bidir",
    "y_trade",
    "y_side",
    "y_side_mask",
    "y_long_bad_path",
    "y_short_bad_path",
) + _MODEL_NATIVE_ACTIVE_RAIL_TARGET_COLS
_MODEL_NATIVE_UNIT_INTERVAL_TARGET_COLS = (
    "y_tf_agreement_score",
    "y_position_size_target",
) + _TIMING_TARGET_COLS
# `mfe_first_n_bps` is intentionally absent: it is the selected-side,
# spread-aware favorable excursion and remains signed when price never earns
# back the entry spread.  MAE targets below are adverse magnitudes.
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

    direction = numeric["y_direction"]
    if bool((~np.isin(direction, [0.0, 1.0, 2.0])).any()):
        failures.append(f"{split_name} model-native y_direction must be exact LONG/SHORT/FLAT ids")
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
    "tf_gate": 5,
    "family_tf_cooperation_gate": 13,
}


_MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS = {
    **dict(EXACT_EVIDENCE_FUSION_OUTPUTS),
    "raw_direction_logits": 3,
    "direction_logits": 3,
    "path_quality": 1,
    "bad_path_logit": 1,
    "public_trade_flat_decision_logits": 2,
    **_MODEL_NATIVE_COOPERATION_GATE_WIDTHS,
}


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


def _env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


def _parse_three_float_value(name: str, raw: str) -> Tuple[float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"[{name}_INVALID] expected three comma-separated floats, got {raw!r}")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise RuntimeError(f"[{name}_INVALID] expected three comma-separated floats, got {raw!r}") from exc
    if any((not np.isfinite(value)) or value <= 0.0 for value in values):
        raise RuntimeError(f"[{name}_INVALID] all weights must be finite and >0, got {raw!r}")
    return values  # type: ignore[return-value]


def _parse_three_float_env(name: str, default: str) -> Tuple[float, float, float]:
    return _parse_three_float_value(name, _env_str(name, default))


def _running_in_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except Exception:
        return False


# -----------------------------------------------------------------------------
# SHORT collapse countermeasures (training-only)
# Secondary training controls. The model-native launch audit owns the exact
# subset that may affect its decision path.
# -----------------------------------------------------------------------------
ENTRY_DIRECTION_CLASS_WEIGHT_CAP = float(
    _env_str("ENTRY_DIRECTION_CLASS_WEIGHT_CAP", "8.0")
)
ENTRY_FLAT_CLASS_WEIGHT_FLOOR = float(_env_str("ENTRY_FLAT_CLASS_WEIGHT_FLOOR", "1.0"))
# -----------------------------------------------------------------------------
# Cost-sensitive loss (ENTRY 3-class)
# -----------------------------------------------------------------------------
# Defaults are deliberately moderate: wrong-side (LONG<->SHORT) costs clearly more
# than LONG/SHORT->FLAT, while FLAT->LONG/SHORT remains moderate.
ENTRY_COST_SENSITIVE_ENABLED = int(_env_str("ENTRY_COST_SENSITIVE_LOSS", "1"))
ENTRY_COST_SENSITIVE_SCALE = float(_env_str("ENTRY_COST_SENSITIVE_SCALE", "0.25"))
# 2026-05-26: directional costs SYMMETRIZED. The old asymmetric defaults
# (short_to_long=3.00 > long_to_short=2.00; flat_to_long=2.75 > flat_to_short=1.60)
# penalized LONG predictions harder → model over-called SHORT (65% short candidates
# vs ~47% true H=24 direction over a gold bull market). That anti-long tilt was
# tuned in an earlier wave when the model was long-biased; with balanced labels +
# symmetric class weights it over-corrected. Now long↔short and flat→dir costs are
# equal (no directional preference). See project_gx1_v10_short_bias_costmatrix.
ENTRY_COST_LONG_TO_SHORT = float(_env_str("ENTRY_COST_LONG_TO_SHORT", "2.00"))
ENTRY_COST_LONG_TO_FLAT = float(_env_str("ENTRY_COST_LONG_TO_FLAT", "0.45"))
ENTRY_COST_SHORT_TO_LONG = float(_env_str("ENTRY_COST_SHORT_TO_LONG", "2.00"))
ENTRY_COST_SHORT_TO_FLAT = float(_env_str("ENTRY_COST_SHORT_TO_FLAT", "0.45"))
ENTRY_COST_FLAT_TO_LONG = float(_env_str("ENTRY_COST_FLAT_TO_LONG", "1.60"))
ENTRY_COST_FLAT_TO_SHORT = float(_env_str("ENTRY_COST_FLAT_TO_SHORT", "1.60"))

# -----------------------------------------------------------------------------
# Prediction balance regularizer (anti-collapse; training/eval loss only)
# -----------------------------------------------------------------------------
# Default is mild and label-aligned: nudges mean predicted distribution toward
# the batch label distribution (not uniform), to reduce single-side collapse.
ENTRY_PRED_BALANCE_ALPHA = float(_env_str("ENTRY_PRED_BALANCE_ALPHA", "0.0"))
ENTRY_PRED_BALANCE_TARGET = _env_str("ENTRY_PRED_BALANCE_TARGET", "label").lower()
ENTRY_PRED_BALANCE_CLASS_WEIGHTS = _parse_three_float_env("ENTRY_PRED_BALANCE_CLASS_WEIGHTS", "1.0,1.0,1.0")
ENTRY_DIRECTION_CE_SCALE = float(_env_str("ENTRY_DIRECTION_CE_SCALE", "4.00"))
ENTRY_TAIL_DIRECTION_CE_WEIGHT = float(_env_str("ENTRY_TAIL_DIRECTION_CE_WEIGHT", "0.35"))
ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE = float(_env_str("ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE", "0.70"))
ENTRY_TAIL_DIRECTION_MIN_BATCH = int(_env_str("ENTRY_TAIL_DIRECTION_MIN_BATCH", "8"))
ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT = float(_env_str("ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT", "0.05"))
ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT = float(_env_str("ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT", "0.50"))
ENTRY_SPECIALIST_GATE_MIN_MEAN = float(_env_str("ENTRY_SPECIALIST_GATE_MIN_MEAN", "0.01"))
# Forceful MTF→direction (2026-06-06): aux CE on the multi-TF direction logits vs
# the direction label, forcing the 5 multi-TF streams to predict direction.
# Env-overridable; default 0.3 (secondary to the main direction CE).
ENTRY_MTF_DIR_AUX_WEIGHT = float(_env_str("ENTRY_MTF_DIR_AUX_WEIGHT", "0.30"))
ENTRY_OFFLINE_RL_Q_WEIGHT = float(_env_str("ENTRY_OFFLINE_RL_Q_WEIGHT", "0.50"))
ENTRY_OFFLINE_RL_V_WEIGHT = float(_env_str("ENTRY_OFFLINE_RL_V_WEIGHT", "0.20"))
ENTRY_OFFLINE_RL_RANK_WEIGHT = float(_env_str("ENTRY_OFFLINE_RL_RANK_WEIGHT", "0.05"))
# Checkpoint-selection monitor (diagnosis fix #3, 2026-06-06): the early-stop / best-checkpoint
# was selected on TOTAL multi-head val loss, which saves the aux-overfit epoch (the cement froze
# at epoch-2 = the aux optimum, NOT the best-direction epoch). "dir_acc" instead keeps the epoch
# with the highest direction validation accuracy (the metric the chain actually acts on).
# Direction is the production objective; aggregate auxiliary validation loss
# can select an epoch with worse LONG/SHORT/FLAT decisions.  Invalid monitor
# values fail closed instead of silently reverting to the historical behavior.
ENTRY_CKPT_MONITOR = _env_str("GX1_V10_CKPT_MONITOR", "dir_acc").strip().lower()
if ENTRY_CKPT_MONITOR not in {"val_loss", "dir_acc"}:
    raise RuntimeError(
        "[ENTRY_CKPT_MONITOR_INVALID] "
        f"got={ENTRY_CKPT_MONITOR!r} expected=val_loss|dir_acc"
    )
ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT = float(_env_str("ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT", "0.0"))
ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL = float(_env_str("ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL", "0.0"))
ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE = float(_env_str("ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE", "0.0"))
ENTRY_CKPT_DIRECTION_SLICE_GUARD = _env_str("ENTRY_CKPT_DIRECTION_SLICE_GUARD", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT = float(_env_str("ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT", "0.0"))
ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION = float(_env_str("ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION", "0.0"))
ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR = float(_env_str("ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR", "0.0"))
ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE = float(
    _env_str("ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", "1.0")
)
ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT", "0.0")
)
ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE = float(
    _env_str("ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE", "0.02")
)
ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE = float(
    _env_str("ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", "0.10")
)
ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT", "0.0")
)
ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION = float(
    _env_str("ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION", str(ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION))
)
ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR = float(
    _env_str("ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR", str(ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR))
)
ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE = float(_env_str("ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE", "0.10"))
ENTRY_DIRECTION_SLICE_MIN_ROWS = int(float(_env_str("ENTRY_DIRECTION_SLICE_MIN_ROWS", "8")))
ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES = _env_str("ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0,1,2,3,4")
ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT = float(_env_str("ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT", "0.0"))
ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR = float(_env_str("ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR", "0.30"))
ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE = float(
    _env_str("ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT = float(_env_str("ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT", "0.0"))
ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE = float(
    _env_str("ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT = float(_env_str("ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT", "0.0"))
ENTRY_DIRECTION_SLICE_TRUE_MARGIN = float(_env_str("ENTRY_DIRECTION_SLICE_TRUE_MARGIN", "0.10"))
ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE = float(
    _env_str("ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT", "0.0")
)
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN = float(
    _env_str("ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN", "0.02")
)
ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT", "0.0")
)
ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN = float(
    _env_str("ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN", "0.02")
)
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE = float(
    _env_str("ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT", "0.0")
)
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE = float(
    _env_str("ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE", "0.02")
)
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE = float(
    _env_str("ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION = _env_str("ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean").strip().lower()
ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER = _env_str("ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE", "0"))
)
ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS = int(
    float(_env_str("ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS", "6"))
)
ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT = float(_env_str("ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT", "0.0"))
ENTRY_DIRECTION_VS_FLAT_MARGIN = float(_env_str("ENTRY_DIRECTION_VS_FLAT_MARGIN", "0.0"))
ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT = float(_env_str("ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT", "4.00"))
ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS = float(_env_str("ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS", "15.0"))
ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN = float(_env_str("ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN", "0.10"))
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT", "6.00")
)
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS = float(
    _env_str("ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS", "15.0")
)
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN = float(
    _env_str("ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN", "0.10")
)
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT", "8.00")
)
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS", "15.0")
)
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS", "0.0")
)
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH", "0.50")
)
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN", "0.10")
)
ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT = float(_env_str("ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT", "8.00"))
ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS", "15.0")
)
ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS", "0.0")
)
ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH", "0.50")
)
ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP", "4.0")
)
ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT = float(_env_str("ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT", "0.0"))
ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE = float(
    _env_str("ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE", "0.10")
)
ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS = int(float(_env_str("ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS", "8")))
ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION = float(
    _env_str("ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION", "0.50")
)
ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR = float(
    _env_str("ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR", "0.10")
)
ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN = float(
    _env_str("ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN", "0.10")
)

_DIRECTION_AUDIT_MIN_LABEL_RATE = 0.10
_DIRECTION_AUDIT_MIN_PRED_RATE = 0.05
_DIRECTION_AUDIT_MIN_PRED_TO_LABEL = 0.35
_DIRECTION_AUDIT_MIN_SLICE_ROWS = 64
_DIRECTION_SLICE_CKPT_FAILURE_PENALTY = 0.02
_DIRECTION_SLICE_CKPT_DEFICIT_PENALTY = 0.25
_DIRECTION_SLICE_LOSS_AGGREGATIONS = {"mean", "max", "mean_max", "sum", "sqrt"}

# -----------------------------------------------------------------------------
# Auxiliary losses (use existing dataset targets)
# -----------------------------------------------------------------------------
ENTRY_AUX_PATH_WEIGHT = float(_env_str("ENTRY_AUX_PATH_WEIGHT", "0.90"))
ENTRY_AUX_MFE_WEIGHT = float(_env_str("ENTRY_AUX_MFE_WEIGHT", "0.25"))
ENTRY_AUX_TRADABLE_WEIGHT = float(_env_str("ENTRY_AUX_TRADABLE_WEIGHT", "1.15"))
ENTRY_AUX_BAD_PATH_WEIGHT = float(_env_str("ENTRY_AUX_BAD_PATH_WEIGHT", "1.25"))
ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP = float(_env_str("ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP", "20.0"))
ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT = float(_env_str("ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT", "2.00"))
ENTRY_BAD_PATH_QUALITY_RANK_MARGIN = float(_env_str("ENTRY_BAD_PATH_QUALITY_RANK_MARGIN", "0.20"))
ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE = float(_env_str("ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE", "0.25"))
ENTRY_PATH_QUALITY_RANK_WEIGHT = float(_env_str("ENTRY_PATH_QUALITY_RANK_WEIGHT", "2.00"))
ENTRY_PATH_QUALITY_RANK_MARGIN = float(_env_str("ENTRY_PATH_QUALITY_RANK_MARGIN", "0.20"))
ENTRY_PATH_QUALITY_RANK_QUANTILE = float(_env_str("ENTRY_PATH_QUALITY_RANK_QUANTILE", "0.25"))
# Scale bps targets to keep regression losses in a stable range
ENTRY_AUX_PATH_SCALE_BPS = float(_env_str("ENTRY_AUX_PATH_SCALE_BPS", "50.0"))
ENTRY_AUX_MFE_SCALE_BPS = float(_env_str("ENTRY_AUX_MFE_SCALE_BPS", "20.0"))
ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP = float(_env_str("ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP", "12.0"))
ENTRY_HARD_NEG_LONG_CE_MULTIPLIER = float(_env_str("ENTRY_HARD_NEG_LONG_CE_MULTIPLIER", "1.35"))
ENTRY_HARD_NEG_LONG_PROB_PENALTY = float(_env_str("ENTRY_HARD_NEG_LONG_PROB_PENALTY", "0.20"))
ENTRY_DEAD_LONG_CE_MULTIPLIER = float(_env_str("ENTRY_DEAD_LONG_CE_MULTIPLIER", "1.80"))
ENTRY_DEAD_LONG_PROB_PENALTY = float(_env_str("ENTRY_DEAD_LONG_PROB_PENALTY", "0.40"))
ENTRY_TEASER_LONG_CE_MULTIPLIER = float(_env_str("ENTRY_TEASER_LONG_CE_MULTIPLIER", "1.35"))
ENTRY_TEASER_LONG_PROB_PENALTY = float(_env_str("ENTRY_TEASER_LONG_PROB_PENALTY", "0.16"))
# Mirror the LONG hard-negative objective onto SHORT with the same magnitudes.
# The exact launch recipe must set this to one; there is no diagnostic bypass.
ENTRY_SYMMETRIC_NEGATIVES = _env_str("ENTRY_SYMMETRIC_NEGATIVES", "1") in {"1", "true", "yes", "on"}
ENTRY_BAD_PATH_CE_MULTIPLIER = float(_env_str("ENTRY_BAD_PATH_CE_MULTIPLIER", "1.50"))
ENTRY_BAD_PATH_PROB_PENALTY = float(_env_str("ENTRY_BAD_PATH_PROB_PENALTY", "0.24"))
ENTRY_AUX_CLEAN_EDGE_WEIGHT = float(_env_str("ENTRY_AUX_CLEAN_EDGE_WEIGHT", "0.45"))
ENTRY_AUX_SURVIVAL_WEIGHT = float(_env_str("ENTRY_AUX_SURVIVAL_WEIGHT", "0.10"))
ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP = float(_env_str("ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP", "16.0"))
ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP = float(_env_str("ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP", "10.0"))
ENTRY_CLEAN_EDGE_RANKING_WEIGHT = float(_env_str("ENTRY_CLEAN_EDGE_RANKING_WEIGHT", "0.25"))
ENTRY_CLEAN_EDGE_RANKING_MARGIN = float(_env_str("ENTRY_CLEAN_EDGE_RANKING_MARGIN", "0.12"))
# XAU direction repair (2026-07-10): losses for opt-in hierarchical
# trade/no-trade, conditional side and per-side path heads.
ENTRY_HIER_TRADE_WEIGHT = float(_env_str("ENTRY_HIER_TRADE_WEIGHT", "2.00"))
ENTRY_HIER_SIDE_WEIGHT = float(_env_str("ENTRY_HIER_SIDE_WEIGHT", "1.75"))
ENTRY_HIER_UTILITY_WEIGHT = float(_env_str("ENTRY_HIER_UTILITY_WEIGHT", "1.00"))
ENTRY_HIER_BAD_PATH_WEIGHT = float(_env_str("ENTRY_HIER_BAD_PATH_WEIGHT", "1.25"))
ENTRY_HIER_MAE_WEIGHT = float(_env_str("ENTRY_HIER_MAE_WEIGHT", "0.35"))
ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP = float(_env_str("ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP", "20.0"))
ENTRY_HIER_SIDE_VALIDITY_WEIGHT = float(_env_str("ENTRY_HIER_SIDE_VALIDITY_WEIGHT", "1.50"))
ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS = float(_env_str("ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS", "15.0"))
ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP = float(_env_str("ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP", "8.0"))
ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT = float(
    _env_str("ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT", "0.0")
)
ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE = float(
    _env_str("ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE", "0.02")
)
ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT = float(
    _env_str("ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT", "0.0")
)
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE = float(
    _env_str("ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE", "0.02")
)
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS = int(
    float(_env_str("ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT = float(
    _env_str("ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT", "0.0")
)
ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN = float(
    _env_str("ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN", "0.02")
)
ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT = float(_env_str("ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT", "0.0"))
ENTRY_HIER_FLAT_LOGIT_MARGIN = float(_env_str("ENTRY_HIER_FLAT_LOGIT_MARGIN", "0.10"))
ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT = float(
    _env_str("ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT", "0.0")
)
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN = float(_env_str("ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN", "0.10"))
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS = int(
    float(_env_str("ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_HIER_SLICE_SIDE_CE_WEIGHT = float(_env_str("ENTRY_HIER_SLICE_SIDE_CE_WEIGHT", "0.0"))
ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT = float(
    _env_str("ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT", "0.0")
)
ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN = float(_env_str("ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN", "0.10"))
ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT = float(
    _env_str("ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT", "0.0")
)
ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN = float(
    _env_str("ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN", "0.02")
)
ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_SIDE_MIN_ROWS = int(
    float(_env_str("ENTRY_HIER_SLICE_SIDE_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
)
ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT = float(
    _env_str("ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT", "0.0")
)
ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE = float(
    _env_str("ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE", "0.02")
)
ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", str(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT = float(
    _env_str("ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT", "0.0")
)
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE = float(
    _env_str("ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE", "0.02")
)
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE", str(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS = int(
    float(_env_str("ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS", str(ENTRY_HIER_SLICE_SIDE_MIN_ROWS)))
)
ENTRY_TRENDLINE_RAIL_AUX_WEIGHT = float(_env_str("ENTRY_TRENDLINE_RAIL_AUX_WEIGHT", "1.00"))


def _current_model_native_active_loss_weights() -> Dict[str, float]:
    """Return the exact configurable objective surface recorded in each bundle."""

    return {
        "ENTRY_DIRECTION_CE_SCALE": float(ENTRY_DIRECTION_CE_SCALE),
        "ENTRY_TAIL_DIRECTION_CE_WEIGHT": float(ENTRY_TAIL_DIRECTION_CE_WEIGHT),
        "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT": float(ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT),
        "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT": float(ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT),
        "ENTRY_MTF_DIR_AUX_WEIGHT": float(ENTRY_MTF_DIR_AUX_WEIGHT),
        "ENTRY_OFFLINE_RL_Q_WEIGHT": float(ENTRY_OFFLINE_RL_Q_WEIGHT),
        "ENTRY_OFFLINE_RL_V_WEIGHT": float(ENTRY_OFFLINE_RL_V_WEIGHT),
        "ENTRY_OFFLINE_RL_RANK_WEIGHT": float(ENTRY_OFFLINE_RL_RANK_WEIGHT),
        "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT": float(ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT),
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT": float(
            ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT
        ),
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT": float(
            ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT
        ),
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT),
        "ENTRY_AUX_PATH_WEIGHT": float(ENTRY_AUX_PATH_WEIGHT),
        "ENTRY_AUX_MFE_WEIGHT": float(ENTRY_AUX_MFE_WEIGHT),
        "ENTRY_AUX_TRADABLE_WEIGHT": float(ENTRY_AUX_TRADABLE_WEIGHT),
        "ENTRY_AUX_BAD_PATH_WEIGHT": float(ENTRY_AUX_BAD_PATH_WEIGHT),
        "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT": float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT),
        "ENTRY_PATH_QUALITY_RANK_WEIGHT": float(ENTRY_PATH_QUALITY_RANK_WEIGHT),
        "ENTRY_AUX_CLEAN_EDGE_WEIGHT": float(ENTRY_AUX_CLEAN_EDGE_WEIGHT),
        "ENTRY_AUX_SURVIVAL_WEIGHT": float(ENTRY_AUX_SURVIVAL_WEIGHT),
        "ENTRY_CLEAN_EDGE_RANKING_WEIGHT": float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT),
        "ENTRY_HIER_TRADE_WEIGHT": float(ENTRY_HIER_TRADE_WEIGHT),
        "ENTRY_HIER_SIDE_WEIGHT": float(ENTRY_HIER_SIDE_WEIGHT),
        "ENTRY_HIER_UTILITY_WEIGHT": float(ENTRY_HIER_UTILITY_WEIGHT),
        "ENTRY_HIER_BAD_PATH_WEIGHT": float(ENTRY_HIER_BAD_PATH_WEIGHT),
        "ENTRY_HIER_MAE_WEIGHT": float(ENTRY_HIER_MAE_WEIGHT),
        "ENTRY_HIER_SIDE_VALIDITY_WEIGHT": float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT),
        "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT": float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT),
    }

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

def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

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
        raise RuntimeError(f"[ENTRY_COST_INVALID] {name} must be >= 0.0, got {v}")


def _build_active_head_names() -> List[str]:
    """Return the one exact model-native learned-component surface."""
    return list(MODEL_NATIVE_ACTIVE_HEADS)


_EVIDENCE_FUSION_MOVEMENT_KEYS: Tuple[str, ...] = tuple(
    _EVIDENCE_FUSION_PARAMETER_SHAPES
)


def _capture_evidence_fusion_initial_state(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    state = model.state_dict()
    missing = [key for key in _EVIDENCE_FUSION_MOVEMENT_KEYS if key not in state]
    if missing:
        raise RuntimeError(
            "[ENTRY_EVIDENCE_FUSION_INITIAL_STATE_MISSING] "
            f"keys={missing}"
        )
    return {
        key: state[key].detach().cpu().clone()
        for key in _EVIDENCE_FUSION_MOVEMENT_KEYS
    }


def _model_native_evidence_fusion_movement_proof(
    initial_state: Dict[str, torch.Tensor],
    selected_state: Dict[str, torch.Tensor],
    *,
    selected_checkpoint_epoch: int,
) -> Dict[str, Any]:
    if int(selected_checkpoint_epoch) <= 0:
        raise RuntimeError(
            "[ENTRY_EVIDENCE_FUSION_MOVEMENT_EPOCH_INVALID] "
            f"selected_checkpoint_epoch={selected_checkpoint_epoch}"
        )

    parameter_deltas: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    for key in _EVIDENCE_FUSION_MOVEMENT_KEYS:
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
        if [int(value) for value in selected.shape] != list(
            _EVIDENCE_FUSION_PARAMETER_SHAPES[key]
        ):
            failures.append(
                f"{key}:contract_shape={tuple(selected.shape)} "
                f"expected={tuple(_EVIDENCE_FUSION_PARAMETER_SHAPES[key])}"
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

    component_changed = {
        component: any(
            bool(parameter_deltas.get(key, {}).get("changed", False))
            for key in keys
        )
        for component, keys in _EVIDENCE_FUSION_MOVEMENT_COMPONENTS.items()
    }
    for component, changed in component_changed.items():
        if not changed:
            failures.append(f"{component}:no_learned_parameter_movement")

    out_weight = selected_state.get("evidence_fusion_out.weight")
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
        failures.append("evidence_fusion_out.weight:class_rows_not_distinct")

    proof = {
        "schema_version": _EVIDENCE_FUSION_MOVEMENT_SCHEMA_VERSION,
        "reference": _EVIDENCE_FUSION_MOVEMENT_REFERENCE,
        "selected_checkpoint_epoch": int(selected_checkpoint_epoch),
        "parameter_deltas": parameter_deltas,
        "component_changed": component_changed,
        "output_rows_distinct": output_rows_distinct,
        "decision": "PASS",
    }
    if failures:
        raise RuntimeError(
            "[ENTRY_EVIDENCE_FUSION_LEARNED_MOVEMENT_REQUIRED] "
            f"failures={failures} proof={json.dumps(proof, sort_keys=True)}"
        )
    return require_learned_component_movement_metadata(
        proof,
        context="ENTRY_EXPORT",
    )


# V12.2: grad-clip norm + weight-decay set at runtime via CLI flag. Module-level
# so we don't have to thread through 6 layers of function args.
_GRAD_CLIP_NORM: float = 1.0
_WEIGHT_DECAY: float = 1e-5


def _autocast_forward(model: nn.Module, device: torch.device, *args, **kwargs) -> Dict[str, torch.Tensor]:
    """Forward through model in bf16 autocast (if GX1_FAST_TRAIN=1), then cast
    outputs back to fp32 for stable loss computation. No-op (just calls model)
    when fast-train is off.
    """
    from gx1.utils.fast_train import autocast_context
    with autocast_context(device):
        out = model(*args, **kwargs)
    if isinstance(out, dict):
        out = {k: (v.float() if hasattr(v, "float") and torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in out.items()}
    elif torch.is_tensor(out) and out.is_floating_point():
        out = out.float()
    return out


def _multi_tf_kwargs_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Extract the five mandatory causal TF tensors, or fail closed."""
    out: Dict[str, torch.Tensor] = {}
    for key in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1"):
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

def _set_deterministic(seed: int, device: torch.device, deterministic: bool) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    # GX1_FAST_TRAIN: TF32 matmul + cuDNN tf32 enabled when master flag set.
    # Overrides cudnn.deterministic above when deterministic=False — by design.
    try:
        from gx1.utils.fast_train import apply_global_speedups
        _fast_report = apply_global_speedups()
        log.info("[FAST_TRAIN] %s", _fast_report)
    except Exception as _e:
        log.warning("[FAST_TRAIN] init failed: %r", _e)

# -----------------------------------------------------------------------------
# Exact immutable dataset identity
# -----------------------------------------------------------------------------
_TRAIN_ARTIFACT_HASH_ENV = {
    "train_manifest": "GX1_ENTRY_TRAIN_MANIFEST_SHA256",
    "val_manifest": "GX1_ENTRY_VAL_MANIFEST_SHA256",
    "test_manifest": "GX1_ENTRY_TEST_MANIFEST_SHA256",
    "train_parquet": "GX1_ENTRY_TRAIN_PARQUET_SHA256",
    "val_parquet": "GX1_ENTRY_VAL_PARQUET_SHA256",
    "test_parquet": "GX1_ENTRY_TEST_PARQUET_SHA256",
    "m5_prebuilt_path": "GX1_ENTRY_M5_PREBUILT_SHA256",
}
_TRAIN_DATASET_RUN_ID_ENV = "GX1_ENTRY_DATASET_RUN_ID"
# Recipe-validated absolute path of the mandatory verified multi-TF V2 disk
# cache. The launch contract emits this row; it is exact runtime identity, not
# an ambient control.
_TRAIN_MULTI_TF_CACHE_ENV = "GX1_V10_MULTI_TF_V2_CACHE_DIR"
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


def _resolve_explicit_train_split_artifacts(
    *,
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    train_parquet: Path,
    val_parquet: Path,
    test_parquet: Path,
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
        "test": _explicit_regular_artifact(test_manifest, label="test_manifest"),
    }
    parquets = {
        "train": _explicit_regular_artifact(train_parquet, label="train_parquet"),
        "val": _explicit_regular_artifact(val_parquet, label="val_parquet"),
        "test": _explicit_regular_artifact(test_parquet, label="test_parquet"),
    }
    m5_prebuilt = _explicit_regular_artifact(
        m5_prebuilt_path,
        label="m5_prebuilt_path",
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
    for split in ("train", "val", "test"):
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
        state_m5_raw = str(
            state_contract.get("rank_reference_source_parquet")
            if state_contract is not None
            else ""
        ).strip()
        state_m5 = Path(state_m5_raw).expanduser()
        state_m5_sha256 = str(
            state_contract.get("rank_reference_source_parquet_sha256")
            if state_contract is not None
            else ""
        ).strip().lower()
        if (
            not state_m5.is_absolute()
            or state_m5 != m5_prebuilt
            or state_m5_sha256 != expected_m5_sha256
        ):
            raise RuntimeError(
                f"[ENTRY_TRAIN_SPLIT_M5_STATE_BINDING_MISMATCH] {split}: "
                f"path={state_m5_raw!r} sha256={state_m5_sha256!r}"
            )
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


def _model_native_state_contract_failures(contract: Dict[str, Any], *, split: str) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return [f"{split} manifest missing model_native_state_contract for XAU direction repair"]
    try:
        validate_state_contract_metadata_v2(contract, require_artifact=True)
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
        elif state_contract:
            try:
                train_start = pd.Timestamp(pd.to_datetime(train_window.get("start"), utc=True))
                train_end = pd.Timestamp(pd.to_datetime(train_window.get("end"), utc=True))
                rank_fit_start = pd.Timestamp(
                    pd.to_datetime(state_contract.get("rank_fit_start_utc"), utc=True)
                )
                rank_fit_end = pd.Timestamp(
                    pd.to_datetime(state_contract.get("rank_fit_end_utc"), utc=True)
                )
            except Exception:
                failures.append(f"{split} TRAIN/state rank-fit timestamps are invalid")
            else:
                if rank_fit_start != train_start or rank_fit_end != train_end:
                    failures.append(
                        f"{split} TRAIN-only rank fit {rank_fit_start}..{rank_fit_end} "
                        f"does not equal TRAIN window {train_start}..{train_end}"
                    )
        if state_contract:
            state_contracts[split] = state_contract
    if len(state_contracts) > 1:
        baseline_split = next(iter(state_contracts))
        baseline = state_contracts[baseline_split]
        for split, contract in state_contracts.items():
            if contract != baseline:
                failures.append(
                    f"{split} model_native_state_contract differs from {baseline_split}; "
                    "TRAIN/VAL/TEST must share one immutable rank/history contract"
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


def _xau_direction_repair_target_failures(split_name: str, df: pd.DataFrame) -> list[str]:
    required = [
        "y_direction",
        "y_bad_path",
        "y_trade",
        "y_tradable",
        "y_side",
        "y_side_mask",
        "mae_first_n_bps",
        "mfe_first_n_bps",
        "path_quality_bps",
        "y_position_size_target",
        "mfe_long_first_n_bps",
        "mae_long_first_n_bps",
        "mfe_short_first_n_bps",
        "mae_short_first_n_bps",
        "bad_path_long_first_n",
        "bad_path_short_first_n",
        "y_long_final_pnl_at_direction_horizon_bps",
        "y_short_final_pnl_at_direction_horizon_bps",
        "y_direction_target_mode_id",
        "y_direction_long_score_bps",
        "y_direction_short_score_bps",
        "y_long_path_utility_bps",
        "y_short_path_utility_bps",
        "y_long_bad_path",
        "y_short_bad_path",
        "y_long_expected_mae_bps",
        "y_short_expected_mae_bps",
    ]
    failures: list[str] = []
    missing = [name for name in required if name not in df.columns]
    if missing:
        failures.append(
            f"{split_name} missing XAU outcome target columns: {missing}. "
            "Rebuild the fresh model-native dataset; targets must not be inferred or repaired."
        )
        return failures

    y_direction = pd.to_numeric(df["y_direction"], errors="coerce").to_numpy(dtype=np.float64)
    y_trade = pd.to_numeric(df["y_trade"], errors="coerce").to_numpy(dtype=np.float64)
    y_tradable = pd.to_numeric(df["y_tradable"], errors="coerce").to_numpy(dtype=np.float64)
    y_side = pd.to_numeric(df["y_side"], errors="coerce").to_numpy(dtype=np.float64)
    y_side_mask = pd.to_numeric(df["y_side_mask"], errors="coerce").to_numpy(dtype=np.float64)
    y_bad_path = pd.to_numeric(df["y_bad_path"], errors="coerce").to_numpy(dtype=np.float64)
    mae_first = pd.to_numeric(df["mae_first_n_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mfe_first = pd.to_numeric(df["mfe_first_n_bps"], errors="coerce").to_numpy(dtype=np.float64)
    path_quality = pd.to_numeric(df["path_quality_bps"], errors="coerce").to_numpy(dtype=np.float64)
    y_long_utility = pd.to_numeric(df["y_long_path_utility_bps"], errors="coerce").to_numpy(dtype=np.float64)
    y_short_utility = pd.to_numeric(df["y_short_path_utility_bps"], errors="coerce").to_numpy(dtype=np.float64)
    y_long_bad = pd.to_numeric(df["y_long_bad_path"], errors="coerce").to_numpy(dtype=np.float64)
    y_short_bad = pd.to_numeric(df["y_short_bad_path"], errors="coerce").to_numpy(dtype=np.float64)
    y_long_mae = pd.to_numeric(df["y_long_expected_mae_bps"], errors="coerce").to_numpy(dtype=np.float64)
    y_short_mae = pd.to_numeric(df["y_short_expected_mae_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mfe_long = pd.to_numeric(df["mfe_long_first_n_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mae_long = pd.to_numeric(df["mae_long_first_n_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mfe_short = pd.to_numeric(df["mfe_short_first_n_bps"], errors="coerce").to_numpy(dtype=np.float64)
    mae_short = pd.to_numeric(df["mae_short_first_n_bps"], errors="coerce").to_numpy(dtype=np.float64)
    raw_long_bad = pd.to_numeric(df["bad_path_long_first_n"], errors="coerce").to_numpy(dtype=np.float64)
    raw_short_bad = pd.to_numeric(df["bad_path_short_first_n"], errors="coerce").to_numpy(dtype=np.float64)
    pnl_long = pd.to_numeric(
        df["y_long_final_pnl_at_direction_horizon_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    pnl_short = pd.to_numeric(
        df["y_short_final_pnl_at_direction_horizon_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    target_mode_id = pd.to_numeric(
        df["y_direction_target_mode_id"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    y_direction_long_score = pd.to_numeric(
        df["y_direction_long_score_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    y_direction_short_score = pd.to_numeric(
        df["y_direction_short_score_bps"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    y_position_size = pd.to_numeric(df["y_position_size_target"], errors="coerce").to_numpy(dtype=np.float64)
    arrays = [
        y_direction,
        y_trade,
        y_tradable,
        y_side,
        y_side_mask,
        y_bad_path,
        mae_first,
        mfe_first,
        path_quality,
        y_long_utility,
        y_short_utility,
        y_long_bad,
        y_short_bad,
        y_long_mae,
        y_short_mae,
        mfe_long,
        mae_long,
        mfe_short,
        mae_short,
        raw_long_bad,
        raw_short_bad,
        pnl_long,
        pnl_short,
        target_mode_id,
        y_direction_long_score,
        y_direction_short_score,
        y_position_size,
    ]
    if any(not np.isfinite(arr).all() for arr in arrays):
        failures.append(f"{split_name} XAU outcome targets contain non-finite values")
        return failures

    if not bool(np.equal(target_mode_id, 1.0).all()):
        failures.append(
            f"{split_name}: y_direction_target_mode_id must be exact split-wide value 1 (path_utility_v2)"
        )
        return failures

    from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
        V12_DIRECTION_UTILITY_MAE_WEIGHT,
        V12_DIRECTION_UTILITY_MFE_WEIGHT,
        V12_DIRECTION_UTILITY_MIN_BPS,
        V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS,
        V12_DIRECTION_UTILITY_PATH_WEIGHT,
    )

    mfe_long_f = mfe_long.astype(np.float32)
    mae_long_f = mae_long.astype(np.float32)
    mfe_short_f = mfe_short.astype(np.float32)
    mae_short_f = mae_short.astype(np.float32)
    expected_long_utility = (
        pnl_long.astype(np.float32)
        + float(V12_DIRECTION_UTILITY_MFE_WEIGHT) * mfe_long_f
        - float(V12_DIRECTION_UTILITY_MAE_WEIGHT) * mae_long_f
        + float(V12_DIRECTION_UTILITY_PATH_WEIGHT) * (mfe_long_f - mae_long_f)
    ).astype(np.float32).astype(np.float64)
    expected_short_utility = (
        pnl_short.astype(np.float32)
        + float(V12_DIRECTION_UTILITY_MFE_WEIGHT) * mfe_short_f
        - float(V12_DIRECTION_UTILITY_MAE_WEIGHT) * mae_short_f
        + float(V12_DIRECTION_UTILITY_PATH_WEIGHT) * (mfe_short_f - mae_short_f)
    ).astype(np.float32).astype(np.float64)
    utility_margin = expected_long_utility - expected_short_utility
    tradable_long = (
        (expected_long_utility >= float(V12_DIRECTION_UTILITY_MIN_BPS))
        & (utility_margin >= float(V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS))
    )
    tradable_short = (
        (expected_short_utility >= float(V12_DIRECTION_UTILITY_MIN_BPS))
        & ((-utility_margin) >= float(V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS))
    )

    expected_direction = np.full(len(df), 2.0, dtype=np.float64)
    only_long = tradable_long & (~tradable_short)
    only_short = tradable_short & (~tradable_long)
    both = tradable_long & tradable_short
    expected_direction[only_long] = 0.0
    expected_direction[only_short] = 1.0
    expected_direction[both & (expected_long_utility >= expected_short_utility)] = 0.0
    expected_direction[both & (expected_short_utility > expected_long_utility)] = 1.0

    expected_scalar_bad = np.zeros(len(df), dtype=np.float64)
    long_rows = (y_trade > 0.5) & (y_side == 0)
    short_rows = (y_trade > 0.5) & (y_side == 1)
    flat_rows = y_trade <= 0.5
    expected_scalar_bad[long_rows] = y_long_bad[long_rows]
    expected_scalar_bad[short_rows] = y_short_bad[short_rows]
    expected_mfe = np.zeros(len(df), dtype=np.float64)
    expected_mae = np.zeros(len(df), dtype=np.float64)
    expected_mfe[long_rows] = mfe_long[long_rows]
    expected_mae[long_rows] = mae_long[long_rows]
    expected_mfe[short_rows] = mfe_short[short_rows]
    expected_mae[short_rows] = mae_short[short_rows]
    expected_path = (
        expected_mfe.astype(np.float32) - expected_mae.astype(np.float32)
    ).astype(np.float32).astype(np.float64)

    checks = {
        "y_direction contains values outside LONG/SHORT/FLAT": ~np.isin(y_direction, [0.0, 1.0, 2.0]),
        "trade rows have y_direction FLAT": (y_trade > 0.5) & (y_direction == 2),
        "LONG direction rows are not marked y_trade": (y_direction == 0) & (y_trade <= 0.5),
        "SHORT direction rows are not marked y_trade": (y_direction == 1) & (y_trade <= 0.5),
        "FLAT direction rows are marked y_trade": (y_direction == 2) & (y_trade > 0.5),
        "y_tradable mismatches y_trade": np.abs(y_tradable - y_trade) > 1e-5,
        "trade rows have y_side_mask off": (y_trade > 0.5) & (y_side_mask <= 0.5),
        "flat rows have y_side_mask on": flat_rows & (y_side_mask > 0.5),
        "LONG trade rows have non-LONG y_side": long_rows & (y_side != 0),
        "SHORT trade rows have non-SHORT y_side": short_rows & (y_side != 1),
        "LONG direction rows have non-LONG y_side": (y_direction == 0) & (y_side_mask > 0.5) & (y_side != 0),
        "SHORT direction rows have non-SHORT y_side": (y_direction == 1) & (y_side_mask > 0.5) & (y_side != 1),
        "y_direction mismatches future outcome side selection": np.abs(y_direction - expected_direction) > 1e-5,
        "scalar y_bad_path mismatches selected side-specific bad-path outcome": np.abs(y_bad_path - expected_scalar_bad) > 1e-5,
        "mfe_first_n_bps mismatches selected side-specific MFE": np.abs(mfe_first - expected_mfe) > 1e-5,
        "mae_first_n_bps mismatches selected side-specific MAE": np.abs(mae_first - expected_mae) > 1e-5,
        "path_quality_bps mismatches selected side-specific path": np.abs(path_quality - expected_path) > 1e-5,
        "FLAT/no-trade rows have non-neutral y_position_size_target": flat_rows & (np.abs(y_position_size - 0.5) > 1e-5),
        "long utility is not the declared future-outcome formula": np.abs(y_long_utility - expected_long_utility) > 1e-4,
        "short utility is not the declared future-outcome formula": np.abs(y_short_utility - expected_short_utility) > 1e-4,
        "long bad-path target differs from raw future path outcome": np.abs(y_long_bad - raw_long_bad) > 1e-5,
        "short bad-path target differs from raw future path outcome": np.abs(y_short_bad - raw_short_bad) > 1e-5,
        "long expected MAE differs from raw future MAE": np.abs(y_long_mae - mae_long) > 1e-5,
        "short expected MAE differs from raw future MAE": np.abs(y_short_mae - mae_short) > 1e-5,
        "y_direction_long_score_bps mismatches outcome long utility": np.abs(y_direction_long_score - y_long_utility) > 1e-5,
        "y_direction_short_score_bps mismatches outcome short utility": np.abs(y_direction_short_score - y_short_utility) > 1e-5,
    }
    for reason, mask in checks.items():
        count = int(np.asarray(mask, dtype=bool).sum())
        if count:
            failures.append(f"{split_name}: {reason}: mismatches={count}")
    return failures


def _log_label_distribution(parquet_path: Path, split: str) -> None:
    p = Path(parquet_path).expanduser().resolve()
    if not p.exists():
        log.warning("[ENTRY_LABEL_DISTRIBUTION] split=%s status=missing path=%s", split, p)
        return
    try:
        df = pd.read_parquet(p, columns=["y_direction", "ctx_cat"])
    except Exception:
        df = pd.read_parquet(p, columns=["y_direction"])
    if "y_direction" not in df.columns:
        log.warning("[ENTRY_LABEL_DISTRIBUTION] split=%s status=no_y_direction path=%s", split, p)
        return
    y = df["y_direction"].astype(int)
    n = int(len(y))
    if n == 0:
        log.warning("[ENTRY_LABEL_DISTRIBUTION] split=%s status=empty path=%s", split, p)
        return
    long_c = int((y == 0).sum())
    short_c = int((y == 1).sum())
    flat_c = int((y == 2).sum())
    long_rate = long_c / n
    short_rate = short_c / n
    flat_rate = flat_c / n
    log.info(
        "[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s n=%d long=%d (%.6f) short=%d (%.6f) flat=%d (%.6f) path=%s",
        split,
        n,
        long_c,
        long_rate,
        short_c,
        short_rate,
        flat_c,
        flat_rate,
        p,
    )
    log.info(
        "[ENTRY_FLAT_LABEL_PROOF] split=%s flat=%d flat_rate=%.6f status=%s path=%s",
        split,
        flat_c,
        flat_rate,
        "OK" if flat_c > 0 else "EMPTY",
        p,
    )

    if "ctx_cat" in df.columns:
        try:
            sess_ids = df["ctx_cat"].apply(lambda v: int(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else None)
            df_s = pd.DataFrame({"y": y, "session_id": sess_ids}).dropna(subset=["session_id"])
            if not df_s.empty:
                for sid, grp in df_s.groupby("session_id"):
                    n_s = int(len(grp))
                    long_rate_s = float((grp["y"] == 0).mean())
                    short_rate_s = float((grp["y"] == 1).mean())
                    flat_rate_s = float((grp["y"] == 2).mean())
                    session_name = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}.get(int(sid), "UNKNOWN")
                    log.info(
                        "[ENTRY_LABEL_BY_SESSION_PROOF] split=%s session=%s session_id=%s n=%d long_rate=%.6f short_rate=%.6f flat_rate=%.6f",
                        split,
                        session_name,
                        int(sid),
                        n_s,
                        long_rate_s,
                        short_rate_s,
                        flat_rate_s,
                    )
        except Exception:
            pass

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# Module-level cache for the exact causal V2 multi-TF feature tables.  The
# contract-qualified key is shared by the pre-train peak-memory build and every
# dataset instance so one source path can produce only one in-process object.
_MULTI_TF_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}
_MULTI_TF_ACTIVE_CACHE_KEYS: Dict[str, str] = {}
_MULTI_TF_CACHE_CONTRACT = "V2_CAUSAL"


def _multi_tf_cache_key(
    m5_prebuilt_path: Path,
    *,
    source_sha256: Optional[str] = None,
    backend_identity: str = "source_build",
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


def _prebuild_multi_tf_v2_features_once(
    m5_prebuilt_path: Path,
) -> Dict[str, pd.DataFrame]:
    """Load/build exact V2 tables once under a byte-bound source identity."""

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
    disk_cache_raw = os.environ.get("GX1_V10_MULTI_TF_V2_CACHE_DIR", "").strip()
    backend_locator = (
        f"disk_path:{Path(disk_cache_raw).expanduser().resolve()}"
        if disk_cache_raw
        else "source_build"
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

    if disk_cache_raw:
        from gx1.features.htf_features import load_multi_tf_v2_cache

        loaded = load_multi_tf_v2_cache(disk_cache_raw)
        # The verified V2 cache binds its own full-history canonical M5 source
        # (the cascade-audited canonical-v3 parquet). The trainer's
        # --m5-prebuilt-path is the model-range seq/snapshot source — a
        # distinct identity bound through the split manifests — so the cache
        # source is proven against its own declared bytes, not against
        # m5_path. Requiring equality here would force the five-timeframe
        # surfaces onto the model-range-trimmed file and reintroduce the
        # truncated HTF warmup defect.
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

    cache_key = _multi_tf_cache_key(
        m5_path,
        source_sha256=source_sha256,
        backend_identity=backend_locator,
    )
    cached = _MULTI_TF_CACHE.get(cache_key)
    if cached is not None:
        _MULTI_TF_ACTIVE_CACHE_KEYS[active_identity] = cache_key
        return cached

    from gx1.features.htf_features import build_multi_tf_per_bar_features_v2

    load_cols = ["time", "open", "high", "low", "close", "volume"]
    import pyarrow.parquet as pq

    source_columns = set(pq.ParquetFile(m5_path).schema_arrow.names)
    missing = [name for name in load_cols if name not in source_columns]
    if missing:
        raise RuntimeError(
            "[MULTI_TF_SOURCE_CONTRACT] exact canonical M5 OHLCV source missing: "
            f"{missing}"
        )
    log.info(
        "[MULTI_TF] pre-building exact V2 causal features once: %s",
        m5_path.name,
    )
    m5 = pd.read_parquet(m5_path, columns=load_cols)
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.set_index("time").sort_index()
    for column in ("open", "high", "low", "close", "volume"):
        m5[column] = m5[column].astype(np.float32)
    feats = build_multi_tf_per_bar_features_v2(m5)
    del m5
    import gc

    gc.collect()
    if _sha256_file(m5_path) != source_sha256:
        raise RuntimeError("[MULTI_TF_SOURCE_CHANGED_DURING_BUILD]")
    _MULTI_TF_CACHE[cache_key] = feats
    _MULTI_TF_ACTIVE_CACHE_KEYS[active_identity] = cache_key
    for tf_name, frame in feats.items():
        log.info(
            "[MULTI_TF] %s: %s bars × %s feats",
            tf_name,
            f"{len(frame):,}",
            frame.shape[1],
        )
    return feats


class EntryV10CtxDataset(Dataset):
    """
    Builds rolling-window samples from canonical ENTRY_V10_CTX parquet.
    ctx_cont / ctx_cat are per-sample (B, N), not per-timestep.

    The dataset always serves M5/M15/H1/H4/D1 per-bar feature windows for the
    exact multi-TF model. Source is the M5 canonical_v3 prebuilt, resampled and
    feature-engineered once at __init__. Resampled tables are cached at module
    level so train_ds + val_ds share them. Adds ~25s init time first call,
    instant on subsequent dataset instantiations with same prebuilt path.
    """

    def __init__(
        self,
        parquet_path: Path,
        seq_len: int,
        m5_prebuilt_path: Path,
        multi_tf_seq_len: int = 96,
        # Per-TF sequence-length overrides are operational memory/receptive-field knobs.
        per_tf_seq_lens: Optional[Dict[str, int]] = None,
        multi_tf_closed_bar: Optional[bool] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.seq_len = int(seq_len)
        self.multi_tf_seq_len = int(multi_tf_seq_len)
        if multi_tf_closed_bar is False:
            raise RuntimeError(
                "ENTRY_MULTI_TF_CAUSALITY: forming higher-timeframe bars are forbidden"
            )
        self._multi_tf_closed_bar = True
        # per_tf_seq_lens: dict like {"M5": 96, "M15": 96, "H1": 96, "H4": 48, "D1": 30}.
        # Unset TFs fall back to multi_tf_seq_len.
        self.per_tf_seq_lens: Dict[str, int] = dict(per_tf_seq_lens) if per_tf_seq_lens else {}
        self._multi_tf_feats: Optional[Dict[str, pd.DataFrame]] = None
        self._multi_tf_shift: Optional[Dict[str, pd.Timedelta]] = None
        self._multi_tf_feature_count: int = 0
        self._memmap_tmpdir: Optional[tempfile.TemporaryDirectory] = None

        if not self.parquet_path.exists():
            raise FileNotFoundError(self.parquet_path)
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
                "y_direction",
                "mae_first_n_bps",
                "y_early_move",
                "y_quality_score",
                "y_tradable",
                "mfe_first_n_bps",
                "path_quality_bps",
                "y_bad_path",
                "y_dead_negative_long",
                "y_teaser_negative_long",
                "y_hard_negative_long",
                "y_clean_edge_long",
                "y_survival_long",
                "y_selector_long_mask",
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
            first_batch = next(pf.iter_batches(batch_size=64, columns=["seq", "snap", "ctx_cont", "ctx_cat"]))
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
            memmap_min_gb = float(os.environ.get("ENTRY_V10_CTX_MEMMAP_MIN_GB", "8.0"))
            memmap_disabled = os.environ.get("ENTRY_V10_CTX_DISABLE_MEMMAP", "0") == "1"
            use_memmap = (not memmap_disabled) and nested_bytes >= int(memmap_min_gb * (1024 ** 3))
            if use_memmap:
                memmap_root = Path(
                    os.environ.get("ENTRY_V10_CTX_MEMMAP_ROOT", "/home/andre2/GX1_DATA/tmp/entry_v10_memmap")
                )
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
                    memmap_min_gb,
                    memmap_dir,
                )
            else:
                self._np_seq = np.zeros(seq_shape, dtype=np.float32)
                self._np_snap = np.zeros(snap_shape, dtype=np.float32)
                self._np_ctx_cont = np.zeros(ctx_cont_shape, dtype=np.float32)
                self._np_ctx_cat = np.zeros(ctx_cat_shape, dtype=np.int64)
            memmap_flush_rows = max(0, int(os.environ.get("ENTRY_V10_CTX_MEMMAP_FLUSH_ROWS", "8192")))
            # Re-iterate (first batch was consumed) — read the whole file in chunks
            idx = 0
            for batch in pq.ParquetFile(self.parquet_path).iter_batches(
                batch_size=8192, columns=["seq", "snap", "ctx_cont", "ctx_cat"]
            ):
                nb = batch.num_rows
                self._np_seq[idx:idx+nb] = batch.column("seq").flatten().flatten().to_numpy(
                    zero_copy_only=False).reshape(nb, seq_len, seq_dim).astype(np.float32, copy=False)
                self._np_snap[idx:idx+nb] = batch.column("snap").flatten().to_numpy(
                    zero_copy_only=False).reshape(nb, snap_dim).astype(np.float32, copy=False)
                self._np_ctx_cont[idx:idx+nb] = batch.column("ctx_cont").flatten().to_numpy(
                    zero_copy_only=False).reshape(nb, ctx_cont_dim).astype(np.float32, copy=False)
                self._np_ctx_cat[idx:idx+nb] = batch.column("ctx_cat").flatten().to_numpy(
                    zero_copy_only=False).reshape(nb, ctx_cat_dim).astype(np.int64, copy=False)
                idx += nb
                if use_memmap and memmap_flush_rows and idx % memmap_flush_rows == 0:
                    _flush_memmap_pages(self._np_seq, self._np_snap, self._np_ctx_cont, self._np_ctx_cat)
            for arr in (self._np_seq, self._np_snap, self._np_ctx_cont, self._np_ctx_cat):
                if isinstance(arr, np.memmap):
                    arr.flush()
            if use_memmap:
                _flush_memmap_pages(self._np_seq, self._np_snap, self._np_ctx_cont, self._np_ctx_cat)
            log.info(f"[MEM_FIX] arrays built: seq={self._np_seq.shape} ({self._np_seq.nbytes/1e9:.2f} GB)")
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

            y = df["y_direction"].astype(int).values
            if len(np.unique(y)) < 2:
                raise RuntimeError(
                    "[ENTRY_V10_CTX_LABELS_CONSTANT] exact training requires at least two classes"
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
        # MANDATORY V2: 5 TFs (M5/M15/H1/H4/D1) × 25 feats. The V1 (4-TF/17-feat)
        # path + GX1_V10_MULTI_TF_V2 env-gate were removed 2026-05-26 — multi-TF×5
        # is the only supported mode (rule: multi_tf_always_mandatory).
        from gx1.features.htf_features import (
            MULTI_TF_FEATURE_COUNT_V2,
            MULTI_TF_SHIFT,
        )
        if m5_prebuilt_path is None:
            raise RuntimeError(
                "[MULTI_TF_INIT_FAIL] exact architecture requires m5_prebuilt_path "
                "(path to canonical_v3 M5 OHLC parquet)."
            )
        m5_path = Path(m5_prebuilt_path)
        # One loader owns both source-build and verified disk-cache paths. Its
        # in-process identity includes the exact M5 SHA and, when used, the
        # full byte-bound disk-cache identity. No path-only cache shortcut is
        # permitted.
        self._multi_tf_feats = _prebuild_multi_tf_v2_features_once(m5_path)
        self._multi_tf_shift = MULTI_TF_SHIFT
        self._multi_tf_target_availability_shift = pd.Timedelta(minutes=5)
        self._multi_tf_feature_count = int(MULTI_TF_FEATURE_COUNT_V2)
        self._multi_tf_v2 = True
        for tf_name, feats in self._multi_tf_feats.items():
            log.info(
                f"[MULTI_TF] {tf_name}: {len(feats):,} bars × {feats.shape[1]} feats  "
                f"range {feats.index[0]} → {feats.index[-1]}"
            )

    def _get_multi_tf_window(self, target_ts: pd.Timestamp) -> Dict[str, np.ndarray]:
        """Slice the multi-TF window at-or-before target_ts, using per-TF seq_len.

        Returns dict with one 'seq_<tf>' key per TF in self._multi_tf_feats. Each
        array shape = (per_tf_lens[TF], feature_count) float32. Insufficient
        history or indicator warmup fails closed.
        """
        from gx1.features.htf_features import get_last_n_at_or_before
        default_n = self.multi_tf_seq_len
        out: Dict[str, np.ndarray] = {}
        availability_ts = pd.Timestamp(target_ts) + getattr(
            self,
            "_multi_tf_target_availability_shift",
            pd.Timedelta(0),
        )
        for tf, feats in self._multi_tf_feats.items():
            n = int(self.per_tf_seq_lens.get(tf, default_n))
            out[f"seq_{tf.lower()}"] = get_last_n_at_or_before(
                feats, availability_ts, n=n, tf_shift=self._multi_tf_shift[tf],
            )
        return out

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if self._advanced:
            t = int(self.indices[i])
            row = self.df.iloc[t]
            # V12.2: nested cols were pre-converted to np arrays in __init__;
            # __getitem__ now just slices for speed + memory efficiency.
            seq = self._np_seq[t]
            snap = self._np_snap[t]
            ctx_cont = self._np_ctx_cont[t]
            ctx_cat = self._np_ctx_cat[t]
            y = int(np.asarray(row["y_direction"]).ravel()[0])
            if y not in (0, 1, 2):
                raise RuntimeError(f"[ENTRY_V10_CTX_LABEL_INVALID] y_direction={y} expected 0/1/2")

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
                "seq_x": torch.tensor(seq),
                "snap_x": torch.tensor(snap),
                "ctx_cont": torch.tensor(ctx_cont),
                "ctx_cat": torch.tensor(ctx_cat),
                "y": torch.tensor(y, dtype=torch.long),
            }
            # Every active target was validated in __init__ and is read
            # directly. There are no aliases, defaults, or compatibility rows.
            for target_name in _MODEL_NATIVE_ACTIVE_TARGET_COLS:
                if target_name == "y_direction":
                    continue
                target_dtype = torch.long if target_name == "y_side" else torch.float32
                target_value = (
                    int(row[target_name])
                    if target_dtype == torch.long
                    else float(row[target_name])
                )
                out_batch[target_name] = torch.tensor(target_value, dtype=target_dtype)
            mtf = self._get_multi_tf_window(pd.Timestamp(row["time"]))
            for k, v in mtf.items():
                out_batch[k] = torch.from_numpy(v)
            return out_batch

# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------
class CostSensitiveCrossEntropyLoss(nn.Module):
    """
    Cost-sensitive cross-entropy for ENTRY 3-class (0=LONG, 1=SHORT, 2=FLAT).
    Base CE uses optional class weights; expected misclassification cost is added
    using a fixed cost matrix indexed by true class.
    """

    def __init__(
        self,
        *,
        class_weights: Optional[torch.Tensor],
        cost_matrix: torch.Tensor,
        cost_scale: float = 1.0,
        enabled: bool = True,
        balance_alpha: float = 0.0,
        balance_target: str = "label",
        balance_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.cost_scale = float(cost_scale)
        self.balance_alpha = float(balance_alpha)
        self.balance_target = str(balance_target).strip().lower()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.register_buffer("cost_matrix", cost_matrix.float())
        if balance_class_weights is None:
            balance_class_weights = torch.ones(3, dtype=torch.float32)
        self.register_buffer("balance_class_weights", balance_class_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)  # (B,)
        probs = torch.softmax(logits, dim=1)
        loss = ce
        if self.enabled:
            cost = self.cost_matrix.to(dtype=logits.dtype)[targets]  # (B,3)
            expected_cost = (cost * probs).sum(dim=1)
            loss = loss + (self.cost_scale * expected_cost)
        balance_loss = _direction_balance_term(probs, targets, self)
        if balance_loss.numel() == 1:
            loss = loss + balance_loss
        min_pred_rate_loss = _direction_min_pred_rate_term(probs, targets)
        if min_pred_rate_loss.numel() == 1:
            loss = loss + min_pred_rate_loss
        flat_margin_loss = _direction_vs_flat_margin_term(logits, targets)
        if flat_margin_loss.numel() == 1:
            loss = loss + flat_margin_loss
        return loss.mean()


def _direction_balance_term(
    probs: torch.Tensor,
    targets: torch.Tensor,
    criterion: Any,
) -> torch.Tensor:
    zero = torch.zeros((), device=probs.device, dtype=probs.dtype)
    if float(getattr(criterion, "balance_alpha", 0.0)) <= 0.0:
        return zero
    mean_probs = probs.mean(dim=0)
    if str(getattr(criterion, "balance_target", "label")).strip().lower() == "uniform":
        target = torch.full_like(mean_probs, 1.0 / mean_probs.numel())
    else:
        counts = torch.bincount(targets, minlength=mean_probs.numel()).float()
        denom = counts.sum().clamp(min=1.0)
        target = counts / denom
    weights = getattr(criterion, "balance_class_weights", torch.ones_like(mean_probs))
    weights = weights.to(device=mean_probs.device, dtype=mean_probs.dtype)
    balance_loss = torch.mean(((mean_probs - target) ** 2) * weights)
    return float(getattr(criterion, "balance_alpha", 0.0)) * balance_loss


def _direction_min_pred_rate_term(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=probs.device, dtype=probs.dtype)
    weight = float(ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT)
    if weight <= 0.0:
        return zero
    if probs.ndim != 2 or probs.shape[1] < 2:
        return zero
    pred_rate_probs = _direction_pred_rate_probs(probs)
    mean_probs = pred_rate_probs.mean(dim=0)
    counts = torch.bincount(targets, minlength=probs.shape[1]).to(device=probs.device, dtype=probs.dtype)
    counts = counts[: probs.shape[1]]
    label_rates = counts / counts.sum().clamp(min=1.0)
    active = label_rates > 0.0
    if not bool(active.any().detach().cpu().item()):
        return zero
    pred_rates = mean_probs[: label_rates.numel()]
    fraction_req = torch.clamp(label_rates * float(ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION), min=0.0)
    floor_req = torch.full_like(fraction_req, max(0.0, float(ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR)))
    required = torch.maximum(fraction_req, floor_req)
    deficit = torch.relu(required[active] - pred_rates[active])
    return weight * deficit.sum()


def _direction_pred_rate_probs(probs: torch.Tensor) -> torch.Tensor:
    temperature = float(ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE)
    if abs(temperature - 1.0) <= 1e-12:
        return probs
    temperature = max(temperature, 1e-6)
    return torch.softmax(
        torch.log(torch.clamp(probs, min=torch.finfo(probs.dtype).tiny)) / temperature,
        dim=1,
    )


def _direction_global_prior_match_term(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=probs.device, dtype=probs.dtype)
    weight = float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT)
    if weight <= 0.0:
        return zero
    if probs.ndim != 2 or probs.shape[1] < 3 or len(targets) != probs.shape[0]:
        return zero

    tolerance = float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE)
    min_label_rate = float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE)
    if tolerance < 0.0 or tolerance > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE_INVALID] "
            f"ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE={tolerance:.6f} expected [0.0, 1.0]"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )

    pred_rate_probs = _direction_pred_rate_probs(probs)
    counts = torch.bincount(targets.long(), minlength=probs.shape[1]).to(
        device=probs.device,
        dtype=probs.dtype,
    )
    counts = counts[: probs.shape[1]]
    label_rates = counts / counts.sum().clamp(min=1.0)
    active = label_rates >= min_label_rate
    if int(active.sum().detach().cpu().item()) < 2:
        return zero
    pred_rates = pred_rate_probs.mean(dim=0)[: label_rates.numel()]
    drift = torch.abs(pred_rates[active] - label_rates[active])
    tol = torch.as_tensor(tolerance, device=probs.device, dtype=probs.dtype)
    return weight * torch.relu(drift - tol).sum()


def _direction_slice_ctx_cat_indices(ctx_cat_dim: int) -> list[int]:
    out: list[int] = []
    for raw in str(ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            idx = int(raw)
        except ValueError:
            continue
        if 0 <= idx < int(ctx_cat_dim):
            out.append(idx)
    return sorted(set(out))


class _DirectionSliceBalancedSampler(Sampler[int]):
    """Orders every selected row once while concentrating audited slices in batches."""

    def __init__(
        self,
        *,
        labels: np.ndarray,
        ctx_cat: np.ndarray,
        ctx_cat_indices: list[int],
        batch_size: int,
        min_rows: int,
        min_label_rate: float,
        seed: int,
    ) -> None:
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        ctx_arr = np.asarray(ctx_cat, dtype=np.int64)
        if ctx_arr.ndim != 2:
            raise RuntimeError(
                f"[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_CTX_INVALID] ctx_cat_shape={ctx_arr.shape}"
            )
        if labels_arr.shape[0] != ctx_arr.shape[0]:
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_SHAPE_MISMATCH] "
                f"labels={labels_arr.shape[0]} ctx_cat={ctx_arr.shape[0]}"
            )
        if labels_arr.shape[0] <= 0:
            raise RuntimeError("[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_EMPTY_DATASET]")
        if int(batch_size) < 2:
            raise RuntimeError(
                f"[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_BATCH_INVALID] batch_size={batch_size} expected >=2"
            )
        if int(min_rows) < 2:
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS_INVALID] "
                f"ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS={min_rows} expected >=2"
            )
        if int(batch_size) < int(min_rows):
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_BATCH_TOO_SMALL] "
                f"batch_size={batch_size} min_rows={min_rows}"
            )
        if float(min_label_rate) < 0.0 or float(min_label_rate) > 1.0:
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_LABEL_RATE_INVALID] "
                f"min_label_rate={float(min_label_rate):.6f} expected [0.0, 1.0]"
            )
        indices = [int(idx) for idx in ctx_cat_indices if 0 <= int(idx) < int(ctx_arr.shape[1])]
        if not indices:
            raise RuntimeError("[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_NO_CTX_CAT_INDICES]")

        self.labels = labels_arr
        self.ctx_cat = ctx_arr
        self.batch_size = int(batch_size)
        self.min_rows = int(min_rows)
        self.seed = int(seed)
        self._iteration = 0
        self.num_samples = int(labels_arr.shape[0])
        self._slice_rows: dict[tuple[int, int], np.ndarray] = {}
        self._slice_class_rows: dict[tuple[tuple[int, int], int], np.ndarray] = {}
        self._active_classes: dict[tuple[int, int], list[int]] = {}

        for idx in indices:
            for raw_value in np.unique(ctx_arr[:, idx]):
                value = int(raw_value)
                mask = ctx_arr[:, idx] == value
                rows = np.flatnonzero(mask).astype(np.int64)
                if int(rows.size) < self.min_rows:
                    continue
                slice_labels = labels_arr[rows]
                counts = np.bincount(slice_labels, minlength=3).astype(np.float64)
                label_rates = counts / max(1.0, float(counts.sum()))
                active = [
                    int(cls)
                    for cls in range(min(3, len(label_rates)))
                    if label_rates[cls] >= float(min_label_rate) and int(counts[cls]) > 0
                ]
                if not active:
                    continue
                key = (int(idx), int(value))
                self._slice_rows[key] = rows
                self._active_classes[key] = active
                for cls in active:
                    cls_rows = rows[slice_labels == int(cls)].astype(np.int64)
                    if cls_rows.size <= 0:
                        raise RuntimeError(
                            "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_CLASS_EMPTY] "
                            f"slice={key} class={cls}"
                        )
                    self._slice_class_rows[(key, int(cls))] = cls_rows

        self._slice_keys = sorted(self._slice_rows)
        if not self._slice_keys:
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_NO_ACTIVE_SLICES] "
                f"indices={indices} min_rows={self.min_rows} min_label_rate={float(min_label_rate):.3f}"
            )

    @property
    def audited_slice_count(self) -> int:
        return len(self._slice_keys)

    def __len__(self) -> int:
        return self.num_samples

    def _sample_slice_rows_without_replacement(
        self,
        rng: np.random.Generator,
        key: tuple[int, int],
        available: np.ndarray,
    ) -> list[int]:
        active_classes = list(self._active_classes[key])
        out: list[int] = []
        shuffled_classes = list(rng.permutation(active_classes).tolist())
        for cls in shuffled_classes:
            cls_rows = self._slice_class_rows[(key, int(cls))]
            eligible = cls_rows[available[cls_rows]]
            if eligible.size <= 0:
                return []
            out.append(int(rng.choice(eligible)))
        all_rows = self._slice_rows[key]
        remaining = all_rows[available[all_rows]]
        if out:
            remaining = remaining[~np.isin(remaining, np.asarray(out, dtype=np.int64))]
        needed = self.min_rows - len(out)
        if needed < 0 or int(remaining.size) < int(needed):
            return []
        if needed:
            out.extend(
                int(value)
                for value in np.asarray(
                    rng.choice(remaining, size=int(needed), replace=False),
                    dtype=np.int64,
                ).tolist()
            )
        rng.shuffle(out)
        if len(out) != self.min_rows or len(set(out)) != len(out):
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_SLICE_SELECTION_INVALID] "
                f"slice={key} selected={len(out)} unique={len(set(out))} expected={self.min_rows}"
            )
        return out

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        batches = max(1, int(np.ceil(self.num_samples / self.batch_size)))
        slice_keys = [self._slice_keys[int(i)] for i in rng.permutation(len(self._slice_keys))]
        slice_pos = 0
        available = np.ones(self.num_samples, dtype=bool)
        emitted = 0
        for _ in range(batches):
            batch: list[int] = []
            remaining_count = int(available.sum())
            target_batch_size = min(self.batch_size, remaining_count)
            if target_batch_size <= 0:
                break
            attempts = 0
            while (
                len(batch) + self.min_rows <= target_batch_size
                and attempts < len(self._slice_keys)
            ):
                key = slice_keys[slice_pos]
                slice_pos += 1
                if slice_pos >= len(slice_keys):
                    slice_keys = [self._slice_keys[int(i)] for i in rng.permutation(len(self._slice_keys))]
                    slice_pos = 0
                attempts += 1
                selected = self._sample_slice_rows_without_replacement(
                    rng,
                    key,
                    available,
                )
                if not selected:
                    continue
                selected_arr = np.asarray(selected, dtype=np.int64)
                if not bool(available[selected_arr].all()):
                    raise RuntimeError(
                        "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_DUPLICATE_SELECTION]"
                    )
                available[selected_arr] = False
                batch.extend(selected)
            needed = target_batch_size - len(batch)
            if needed > 0:
                remaining = np.flatnonzero(available).astype(np.int64)
                if int(remaining.size) < int(needed):
                    raise RuntimeError(
                        "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_REMAINDER_INVALID] "
                        f"remaining={remaining.size} needed={needed}"
                    )
                selected_arr = np.asarray(
                    rng.choice(remaining, size=int(needed), replace=False),
                    dtype=np.int64,
                )
                available[selected_arr] = False
                batch.extend(int(value) for value in selected_arr.tolist())
            rng.shuffle(batch)
            for sample_idx in batch:
                emitted += 1
                yield int(sample_idx)
        if emitted != self.num_samples or bool(available.any()):
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_EMIT_MISMATCH] "
                f"emitted={emitted} expected={self.num_samples} remaining={int(available.sum())}"
            )


def _direction_slice_sampler_arrays(dataset: EntryV10CtxDataset) -> tuple[np.ndarray, np.ndarray]:
    row_indices = np.asarray(getattr(dataset, "indices", []), dtype=np.int64)
    if row_indices.size <= 0:
        raise RuntimeError("[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_DATASET_EMPTY]")
    if "y_direction" not in dataset.df.columns:
        raise RuntimeError("[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_LABEL_MISSING] y_direction")
    labels_all = dataset.df["y_direction"].to_numpy(dtype=np.int64)
    labels = labels_all[row_indices]
    if hasattr(dataset, "_np_ctx_cat"):
        ctx_all = np.asarray(getattr(dataset, "_np_ctx_cat"), dtype=np.int64)
        ctx_cat = ctx_all[row_indices]
    elif hasattr(dataset, "ctx_cat_cols"):
        ctx_cols = list(getattr(dataset, "ctx_cat_cols"))
        missing = [col for col in ctx_cols if col not in dataset.df.columns]
        if missing:
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_CTX_MISSING] "
                + ",".join(str(col) for col in missing)
            )
        ctx_cat = dataset.df[ctx_cols].to_numpy(dtype=np.int64)[row_indices]
    else:
        raise RuntimeError("[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_CTX_UNAVAILABLE]")
    return labels, ctx_cat


def _direction_slice_loss_aggregate(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise RuntimeError("[ENTRY_DIRECTION_SLICE_LOSS_EMPTY]")
    stacked = torch.stack(values)
    mode = str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION or "mean").strip().lower()
    if mode == "mean":
        return stacked.mean()
    if mode == "max":
        return stacked.max()
    if mode == "mean_max":
        return stacked.mean() + stacked.max()
    if mode == "sum":
        return stacked.sum()
    if mode == "sqrt":
        denom = torch.sqrt(
            torch.as_tensor(float(stacked.numel()), device=stacked.device, dtype=stacked.dtype)
        )
        return stacked.sum() / denom
    raise RuntimeError(
        "[ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION_INVALID] "
        f"ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION={ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION!r}"
    )


def _direction_slice_min_pred_rate_term(
    probs: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=probs.device, dtype=probs.dtype)
    weight = float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT)
    if weight <= 0.0:
        return zero
    if probs.ndim != 2 or probs.shape[1] < 3 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(targets) != probs.shape[0] or ctx_cat.shape[0] != probs.shape[0]:
        return zero

    min_rows = max(2, int(ENTRY_DIRECTION_SLICE_MIN_ROWS))
    fraction = max(0.0, float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION))
    floor = max(0.0, float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR))
    min_label_rate = max(0.0, float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    pred_rate_probs = _direction_pred_rate_probs(probs)
    values: list[torch.Tensor] = []
    target_i = targets.long()
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=probs.shape[1]).to(
                device=probs.device,
                dtype=probs.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active = label_rates >= min_label_rate
            if not bool(active.any().detach().cpu().item()):
                continue
            pred_rates = pred_rate_probs[mask].mean(dim=0)
            required = torch.maximum(
                label_rates * fraction,
                torch.full_like(label_rates, floor),
            )
            values.append(torch.relu(required[active] - pred_rates[active]).sum())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _direction_slice_recall_prob_term(
    probs: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=probs.device, dtype=probs.dtype)
    weight = float(ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT)
    if weight <= 0.0:
        return zero
    if probs.ndim != 2 or probs.shape[1] < 3 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(targets) != probs.shape[0] or ctx_cat.shape[0] != probs.shape[0]:
        return zero

    min_rows = max(2, int(ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS))
    min_label_rate = max(0.0, float(ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE))
    prob_floor = max(0.0, min(1.0, float(ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR)))
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    values: list[torch.Tensor] = []
    target_i = targets.long()
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=probs.shape[1]).to(
                device=probs.device,
                dtype=probs.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active_classes = torch.nonzero(label_rates >= min_label_rate, as_tuple=False).flatten()
            if active_classes.numel() <= 0:
                continue
            slice_probs = probs[mask]
            for cls in active_classes.tolist():
                class_mask = slice_targets == int(cls)
                class_rows = int(class_mask.sum().detach().cpu().item())
                if class_rows <= 0:
                    continue
                class_prob = slice_probs[class_mask, int(cls)].mean()
                values.append(torch.relu(torch.as_tensor(prob_floor, device=probs.device, dtype=probs.dtype) - class_prob))
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _direction_slice_balanced_ce_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(targets) != logits.shape[0] or ctx_cat.shape[0] != logits.shape[0]:
        return zero

    min_rows = int(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS)
    min_label_rate = float(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE)
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    log_probs = torch.log_softmax(logits, dim=1)
    target_i = targets.long()
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=logits.shape[1]).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active_classes = torch.nonzero(label_rates >= min_label_rate, as_tuple=False).flatten()
            if active_classes.numel() <= 0:
                continue
            slice_log_probs = log_probs[mask]
            for cls in active_classes.tolist():
                class_mask = slice_targets == int(cls)
                if not bool(class_mask.any().detach().cpu().item()):
                    continue
                values.append(-slice_log_probs[class_mask, int(cls)].mean())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _direction_slice_true_margin_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(targets) != logits.shape[0] or ctx_cat.shape[0] != logits.shape[0]:
        return zero

    margin = float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN)
    min_rows = int(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS)
    min_label_rate = float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE)
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_TRUE_MARGIN_INVALID] "
            f"ENTRY_DIRECTION_SLICE_TRUE_MARGIN={margin:.6f} expected >=0.0"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    target_i = targets.long()
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=logits.shape[1]).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active_classes = torch.nonzero(label_rates >= min_label_rate, as_tuple=False).flatten()
            if active_classes.numel() <= 0:
                continue
            slice_logits = logits[mask]
            for cls in active_classes.tolist():
                cls_i = int(cls)
                class_mask = slice_targets == cls_i
                if not bool(class_mask.any().detach().cpu().item()):
                    continue
                class_logits = slice_logits[class_mask]
                true_logit = class_logits[:, cls_i]
                wrong_logits = class_logits.clone()
                wrong_logits[:, cls_i] = torch.finfo(wrong_logits.dtype).min
                wrong_max = wrong_logits.max(dim=1).values
                margin_t = torch.as_tensor(margin, device=logits.device, dtype=logits.dtype)
                values.append(torch.relu(margin_t - (true_logit - wrong_max)).mean())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _hier_slice_side_balanced_ce_term(
    side_logits: torch.Tensor,
    side_targets: torch.Tensor,
    side_mask: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=side_logits.device, dtype=side_logits.dtype)
    weight = float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT)
    if weight <= 0.0:
        return zero
    if side_logits.ndim != 2 or side_logits.shape[1] < 2 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(side_targets) != side_logits.shape[0] or ctx_cat.shape[0] != side_logits.shape[0]:
        return zero

    min_rows = int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE)
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    target_i = side_targets.long().clamp(0, side_logits.shape[1] - 1)
    valid_side = side_mask.bool()
    log_probs = torch.log_softmax(side_logits, dim=1)
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = valid_side & (ctx_cat[:, idx].long() == value)
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=side_logits.shape[1]).to(
                device=side_logits.device,
                dtype=side_logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active_classes = torch.nonzero(label_rates >= min_label_rate, as_tuple=False).flatten()
            if active_classes.numel() < 2:
                continue
            slice_log_probs = log_probs[mask]
            for cls in active_classes.tolist():
                cls_i = int(cls)
                class_mask = slice_targets == cls_i
                if not bool(class_mask.any().detach().cpu().item()):
                    continue
                values.append(-slice_log_probs[class_mask, cls_i].mean())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _hier_slice_side_true_margin_term(
    side_logits: torch.Tensor,
    side_targets: torch.Tensor,
    side_mask: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=side_logits.device, dtype=side_logits.dtype)
    weight = float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT)
    if weight <= 0.0:
        return zero
    if side_logits.ndim != 2 or side_logits.shape[1] < 2 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(side_targets) != side_logits.shape[0] or ctx_cat.shape[0] != side_logits.shape[0]:
        return zero

    margin = float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN)
    min_rows = int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE)
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN={margin:.6f} expected >=0.0"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    target_i = side_targets.long().clamp(0, side_logits.shape[1] - 1)
    valid_side = side_mask.bool()
    margin_t = torch.as_tensor(margin, device=side_logits.device, dtype=side_logits.dtype)
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = valid_side & (ctx_cat[:, idx].long() == value)
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=side_logits.shape[1]).to(
                device=side_logits.device,
                dtype=side_logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active_classes = torch.nonzero(label_rates >= min_label_rate, as_tuple=False).flatten()
            if active_classes.numel() < 2:
                continue
            slice_logits = side_logits[mask]
            for cls in active_classes.tolist():
                cls_i = int(cls)
                class_mask = slice_targets == cls_i
                if not bool(class_mask.any().detach().cpu().item()):
                    continue
                class_logits = slice_logits[class_mask]
                true_logit = class_logits[:, cls_i]
                wrong_logits = class_logits.clone()
                wrong_logits[:, cls_i] = torch.finfo(wrong_logits.dtype).min
                wrong_max = wrong_logits.max(dim=1).values
                values.append(torch.relu(margin_t - (true_logit - wrong_max)).mean())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _hier_slice_side_accuracy_edge_term(
    side_logits: torch.Tensor,
    side_targets: torch.Tensor,
    side_mask: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=side_logits.device, dtype=side_logits.dtype)
    weight = float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT)
    if weight <= 0.0:
        return zero
    if side_logits.ndim != 2 or side_logits.shape[1] < 2 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(side_targets) != side_logits.shape[0] or ctx_cat.shape[0] != side_logits.shape[0]:
        return zero

    margin = float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN)
    min_rows = int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE)
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN={margin:.6f} expected >=0.0"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    target_i = side_targets.long().clamp(0, side_logits.shape[1] - 1)
    valid_side = side_mask.bool()
    probs = torch.softmax(side_logits, dim=1)
    hardish_probs = _direction_pred_rate_probs(probs)
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = valid_side & (ctx_cat[:, idx].long() == value)
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=side_logits.shape[1]).to(
                device=side_logits.device,
                dtype=side_logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active = label_rates >= min_label_rate
            if int(active.sum().detach().cpu().item()) < 2:
                continue
            majority = label_rates[:2].max()
            true_prob = hardish_probs[mask][
                torch.arange(rows, device=side_logits.device),
                slice_targets.clamp(min=0, max=hardish_probs.shape[1] - 1),
            ].mean()
            required = torch.clamp(
                majority + torch.as_tensor(margin, device=side_logits.device, dtype=side_logits.dtype),
                max=1.0,
            )
            values.append(torch.relu(required - true_prob))
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _hier_trade_prior_probs(trade_logit: torch.Tensor) -> torch.Tensor:
    trade_prob = torch.sigmoid(trade_logit.reshape(-1))
    flat_prob = 1.0 - trade_prob
    return _direction_pred_rate_probs(torch.stack([trade_prob, flat_prob], dim=1))


def _hier_trade_prior_targets(y_trade: torch.Tensor) -> torch.Tensor:
    trade = y_trade.reshape(-1).float() > 0.5
    return torch.where(
        trade,
        torch.zeros_like(trade, dtype=torch.long),
        torch.ones_like(trade, dtype=torch.long),
    )


def _hier_trade_global_prior_match_term(
    trade_logit: torch.Tensor,
    y_trade: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
    weight = float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT)
    if weight <= 0.0:
        return zero
    if trade_logit.numel() <= 1 or len(y_trade) != trade_logit.reshape(-1).shape[0]:
        return zero

    tolerance = float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE)
    min_label_rate = float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE)
    if tolerance < 0.0 or tolerance > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE_INVALID] "
            f"ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE={tolerance:.6f} expected [0.0, 1.0]"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )

    targets = _hier_trade_prior_targets(y_trade).to(device=trade_logit.device)
    counts = torch.bincount(targets, minlength=2).to(device=trade_logit.device, dtype=trade_logit.dtype)
    label_rates = counts / counts.sum().clamp(min=1.0)
    active = label_rates >= min_label_rate
    if int(active.sum().detach().cpu().item()) < 2:
        return zero
    pred_rates = _hier_trade_prior_probs(trade_logit).mean(dim=0)
    drift = torch.abs(pred_rates[active] - label_rates[active])
    tol = torch.as_tensor(tolerance, device=trade_logit.device, dtype=trade_logit.dtype)
    return weight * torch.relu(drift - tol).sum()


def _hier_slice_trade_prior_match_term(
    trade_logit: torch.Tensor,
    y_trade: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
    weight = float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT)
    if weight <= 0.0:
        return zero
    if ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(y_trade) != trade_logit.reshape(-1).shape[0] or ctx_cat.shape[0] != trade_logit.reshape(-1).shape[0]:
        return zero

    tolerance = float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE)
    min_rows = int(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE)
    if tolerance < 0.0 or tolerance > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE_INVALID] "
            f"ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE={tolerance:.6f} expected [0.0, 1.0]"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    pred_rate_probs = _hier_trade_prior_probs(trade_logit)
    targets = _hier_trade_prior_targets(y_trade).to(device=trade_logit.device)
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = targets[mask]
            counts = torch.bincount(slice_targets, minlength=2).to(
                device=trade_logit.device,
                dtype=trade_logit.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active = label_rates >= min_label_rate
            if int(active.sum().detach().cpu().item()) < 2:
                continue
            pred_rates = pred_rate_probs[mask].mean(dim=0)
            drift = torch.abs(pred_rates[active] - label_rates[active])
            tol = torch.as_tensor(tolerance, device=trade_logit.device, dtype=trade_logit.dtype)
            values.append(torch.relu(drift - tol).sum())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _hier_slice_trade_accuracy_edge_term(
    trade_logit: torch.Tensor,
    y_trade: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
    weight = float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT)
    if weight <= 0.0:
        return zero
    if ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    logits = trade_logit.reshape(-1)
    if len(y_trade) != logits.shape[0] or ctx_cat.shape[0] != logits.shape[0]:
        return zero

    margin = float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN)
    min_rows = int(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE)
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN_INVALID] "
            f"ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN={margin:.6f} expected >=0.0"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    targets = _hier_trade_prior_targets(y_trade).to(device=trade_logit.device)
    hardish_probs = _hier_trade_prior_probs(trade_logit)
    margin_t = torch.as_tensor(margin, device=trade_logit.device, dtype=trade_logit.dtype)
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = targets[mask]
            counts = torch.bincount(slice_targets, minlength=2).to(
                device=trade_logit.device,
                dtype=trade_logit.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active = label_rates >= min_label_rate
            if int(active.sum().detach().cpu().item()) < 2:
                continue
            true_prob = hardish_probs[mask][
                torch.arange(rows, device=trade_logit.device),
                slice_targets.clamp(min=0, max=1),
            ].mean()
            required = torch.clamp(label_rates[:2].max() + margin_t, max=1.0)
            values.append(torch.relu(required - true_prob))
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _hier_flat_logit_margin_term(
    trade_logit: torch.Tensor,
    y_trade: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
    weight = float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT)
    if weight <= 0.0:
        return zero
    flat_mask = y_trade.reshape(-1).float() <= 0.5
    logits = trade_logit.reshape(-1)
    if len(flat_mask) != logits.shape[0]:
        return zero
    flat_rows = int(flat_mask.sum().detach().cpu().item())
    if flat_rows < 1:
        return zero
    flat_rate = float(flat_rows) / float(max(1, int(flat_mask.numel())))
    min_label_rate = float(ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE)
    margin = float(ENTRY_HIER_FLAT_LOGIT_MARGIN)
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_HIER_FLAT_LOGIT_MARGIN_INVALID] "
            f"ENTRY_HIER_FLAT_LOGIT_MARGIN={margin:.6f} expected >=0.0"
        )
    if flat_rate < min_label_rate:
        return zero
    margin_t = torch.as_tensor(margin, device=trade_logit.device, dtype=trade_logit.dtype)
    return weight * torch.relu(logits[flat_mask] + margin_t).mean()


def _hier_slice_flat_logit_margin_term(
    trade_logit: torch.Tensor,
    y_trade: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
    weight = float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT)
    if weight <= 0.0:
        return zero
    if ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    logits = trade_logit.reshape(-1)
    flat_mask = y_trade.reshape(-1).float() <= 0.5
    if len(flat_mask) != logits.shape[0] or ctx_cat.shape[0] != logits.shape[0]:
        return zero

    margin = float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN)
    min_rows = int(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE)
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_INVALID] "
            f"ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN={margin:.6f} expected >=0.0"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    margin_t = torch.as_tensor(margin, device=trade_logit.device, dtype=trade_logit.dtype)
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_flat = flat_mask & mask
            flat_rows = int(slice_flat.sum().detach().cpu().item())
            if flat_rows < 1:
                continue
            if (float(flat_rows) / float(max(1, rows))) < min_label_rate:
                continue
            values.append(torch.relu(logits[slice_flat] + margin_t).mean())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _hier_side_global_prior_match_term(
    side_logits: torch.Tensor,
    side_targets: torch.Tensor,
    side_mask: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=side_logits.device, dtype=side_logits.dtype)
    weight = float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT)
    if weight <= 0.0:
        return zero
    if side_logits.ndim != 2 or side_logits.shape[1] < 2 or len(side_targets) != side_logits.shape[0]:
        return zero

    tolerance = float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE)
    min_label_rate = float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE)
    if tolerance < 0.0 or tolerance > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE_INVALID] "
            f"ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE={tolerance:.6f} expected [0.0, 1.0]"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )

    valid_side = side_mask.bool()
    rows = int(valid_side.sum().detach().cpu().item())
    if rows < 2:
        return zero
    side_probs = torch.softmax(side_logits[valid_side], dim=1)
    pred_rate_probs = _direction_pred_rate_probs(side_probs)
    target_i = side_targets[valid_side].long().clamp(0, side_logits.shape[1] - 1)
    counts = torch.bincount(target_i, minlength=side_logits.shape[1]).to(
        device=side_logits.device,
        dtype=side_logits.dtype,
    )
    label_rates = counts / counts.sum().clamp(min=1.0)
    active = label_rates >= min_label_rate
    if int(active.sum().detach().cpu().item()) < 2:
        return zero
    pred_rates = pred_rate_probs.mean(dim=0)[: label_rates.numel()]
    drift = torch.abs(pred_rates[active] - label_rates[active])
    tol = torch.as_tensor(tolerance, device=side_logits.device, dtype=side_logits.dtype)
    return weight * torch.relu(drift - tol).sum()

def _hier_slice_side_prior_match_term(
    side_logits: torch.Tensor,
    side_targets: torch.Tensor,
    side_mask: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=side_logits.device, dtype=side_logits.dtype)
    weight = float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT)
    if weight <= 0.0:
        return zero
    if side_logits.ndim != 2 or side_logits.shape[1] < 2 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(side_targets) != side_logits.shape[0] or ctx_cat.shape[0] != side_logits.shape[0]:
        return zero

    tolerance = float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE)
    min_rows = int(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE)
    if tolerance < 0.0 or tolerance > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE={tolerance:.6f} expected [0.0, 1.0]"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    side_probs = torch.softmax(side_logits, dim=1)
    pred_rate_probs = _direction_pred_rate_probs(side_probs)
    target_i = side_targets.long().clamp(0, side_logits.shape[1] - 1)
    valid_side = side_mask.bool()
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = valid_side & (ctx_cat[:, idx].long() == value)
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=side_logits.shape[1]).to(
                device=side_logits.device,
                dtype=side_logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active = label_rates >= min_label_rate
            if int(active.sum().detach().cpu().item()) < 2:
                continue
            pred_rates = pred_rate_probs[mask].mean(dim=0)
            drift = torch.abs(pred_rates[active] - label_rates[active])
            tol = torch.as_tensor(tolerance, device=side_logits.device, dtype=side_logits.dtype)
            values.append(torch.relu(drift - tol).sum())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)

def _direction_slice_accuracy_edge_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(targets) != logits.shape[0] or ctx_cat.shape[0] != logits.shape[0]:
        return zero

    margin = float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN)
    min_rows = int(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS)
    min_label_rate = float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE)
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN_INVALID] "
            f"ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN={margin:.6f} expected >=0.0"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    probs = torch.softmax(logits, dim=1)
    hardish_probs = _direction_pred_rate_probs(probs)
    target_i = targets.long()
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=logits.shape[1]).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active = label_rates >= min_label_rate
            if int(active.sum().detach().cpu().item()) < 2:
                continue
            majority = label_rates[:3].max()
            true_prob = hardish_probs[mask][
                torch.arange(rows, device=logits.device),
                target_i[mask].clamp(min=0, max=hardish_probs.shape[1] - 1),
            ].mean()
            required = torch.clamp(
                majority + torch.as_tensor(margin, device=logits.device, dtype=logits.dtype),
                max=1.0,
            )
            values.append(torch.relu(required - true_prob))
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)

def _direction_slice_confusion_pair_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(targets) != logits.shape[0] or ctx_cat.shape[0] != logits.shape[0]:
        return zero

    margin = float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN)
    min_rows = int(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS)
    min_label_rate = float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE)
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN_INVALID] "
            f"ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN={margin:.6f} expected >=0.0"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    probs = torch.softmax(logits, dim=1)
    hardish_probs = _direction_pred_rate_probs(probs)
    target_i = targets.long().clamp(min=0, max=hardish_probs.shape[1] - 1)
    margin_t = torch.as_tensor(margin, device=logits.device, dtype=logits.dtype)
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=logits.shape[1]).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active_classes = torch.nonzero(label_rates[:3] >= min_label_rate, as_tuple=False).flatten()
            if int(active_classes.numel()) < 2:
                continue
            slice_probs = hardish_probs[mask, :3]
            for cls_t in active_classes:
                cls = int(cls_t.detach().cpu().item())
                class_mask = slice_targets == cls
                if int(class_mask.sum().detach().cpu().item()) < 1:
                    continue
                true_mean = slice_probs[class_mask, cls].mean()
                wrong_means = [
                    slice_probs[class_mask, int(wrong_t.detach().cpu().item())].mean()
                    for wrong_t in active_classes
                    if int(wrong_t.detach().cpu().item()) != cls
                ]
                if not wrong_means:
                    continue
                worst_wrong = torch.stack(wrong_means).max()
                values.append(torch.relu(worst_wrong + margin_t - true_mean))
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)

def _direction_slice_prior_match_term(
    probs: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=probs.device, dtype=probs.dtype)
    weight = float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT)
    if weight <= 0.0:
        return zero
    if probs.ndim != 2 or probs.shape[1] < 3 or ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    if len(targets) != probs.shape[0] or ctx_cat.shape[0] != probs.shape[0]:
        return zero

    tolerance = float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE)
    min_rows = int(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS)
    min_label_rate = float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE)
    if tolerance < 0.0 or tolerance > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE={tolerance:.6f} expected [0.0, 1.0]"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    pred_rate_probs = _direction_pred_rate_probs(probs)
    target_i = targets.long()
    values: list[torch.Tensor] = []
    for idx in indices:
        slice_values = torch.unique(ctx_cat[:, idx].long())
        for value in slice_values:
            mask = ctx_cat[:, idx].long() == value
            rows = int(mask.sum().detach().cpu().item())
            if rows < min_rows:
                continue
            slice_targets = target_i[mask]
            counts = torch.bincount(slice_targets, minlength=probs.shape[1]).to(
                device=probs.device,
                dtype=probs.dtype,
            )
            label_rates = counts / counts.sum().clamp(min=1.0)
            active = label_rates >= min_label_rate
            if int(active.sum().detach().cpu().item()) < 2:
                continue
            pred_rates = pred_rate_probs[mask].mean(dim=0)
            drift = torch.abs(pred_rates[active] - label_rates[active])
            tol = torch.as_tensor(tolerance, device=probs.device, dtype=probs.dtype)
            values.append(torch.relu(drift - tol).sum())
    if not values:
        return zero
    return weight * _direction_slice_loss_aggregate(values)


def _direction_vs_flat_margin_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3:
        return zero
    target = targets.long()
    valid = (target >= 0) & (target < logits.shape[1])
    if not bool(valid.any().detach().cpu().item()):
        return zero
    margin = max(0.0, float(ENTRY_DIRECTION_VS_FLAT_MARGIN))
    logits_v = logits[valid]
    target_v = target[valid]
    target_logits = logits_v.gather(1, target_v.view(-1, 1)).squeeze(1)
    competitor_logits = logits_v.clone()
    competitor_logits.scatter_(1, target_v.view(-1, 1), torch.finfo(logits_v.dtype).min)
    competitor = competitor_logits.max(dim=1).values
    return weight * nn.functional.softplus(competitor - target_logits + margin).mean()


def _direction_utility_margin_term(
    logits: torch.Tensor,
    y_long_utility_bps: torch.Tensor,
    y_short_utility_bps: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3:
        return zero
    long_u = y_long_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    short_u = y_short_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if long_u.shape[0] != logits.shape[0] or short_u.shape[0] != logits.shape[0]:
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_MARGIN_SHAPE_MISMATCH] "
            f"logits={tuple(logits.shape)} long={tuple(long_u.shape)} short={tuple(short_u.shape)}"
        )
    gap = long_u - short_u
    min_gap = max(0.0, float(ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS))
    margin = max(0.0, float(ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN))
    long_clear = gap >= min_gap
    short_clear = gap <= -min_gap
    terms: list[torch.Tensor] = []
    if bool(long_clear.any().detach().cpu().item()):
        allowed = torch.maximum(logits[:, 0], logits[:, 2])
        terms.append(nn.functional.softplus(logits[long_clear, 1] - allowed[long_clear] + margin).mean())
    if bool(short_clear.any().detach().cpu().item()):
        allowed = torch.maximum(logits[:, 1], logits[:, 2])
        terms.append(nn.functional.softplus(logits[short_clear, 0] - allowed[short_clear] + margin).mean())
    if not terms:
        return zero
    return weight * torch.stack(terms).mean()


def _direction_side_utility_conviction_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    y_long_utility_bps: torch.Tensor,
    y_short_utility_bps: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3:
        return zero
    target = targets.long().to(device=logits.device).reshape(-1)
    long_u = y_long_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    short_u = y_short_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if target.shape[0] != logits.shape[0] or long_u.shape[0] != logits.shape[0] or short_u.shape[0] != logits.shape[0]:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_SHAPE_MISMATCH] "
            f"logits={tuple(logits.shape)} target={tuple(target.shape)} "
            f"long={tuple(long_u.shape)} short={tuple(short_u.shape)}"
        )
    min_gap = float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS)
    margin = float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN)
    if min_gap < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_INVALID] "
            f"ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS={min_gap:.6f} expected >=0.0"
        )
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN_INVALID] "
            f"ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN={margin:.6f} expected >=0.0"
        )

    gap = long_u - short_u
    long_mask = (target == 0) & (gap >= min_gap)
    short_mask = (target == 1) & (gap <= -min_gap)
    terms: list[torch.Tensor] = []
    if bool(long_mask.any().detach().cpu().item()):
        wrong = torch.maximum(logits[:, 1], logits[:, 2])
        terms.append(nn.functional.softplus(wrong[long_mask] - logits[long_mask, 0] + margin).mean())
    if bool(short_mask.any().detach().cpu().item()):
        wrong = torch.maximum(logits[:, 0], logits[:, 2])
        terms.append(nn.functional.softplus(wrong[short_mask] - logits[short_mask, 1] + margin).mean())
    if not terms:
        return zero
    return weight * torch.stack(terms).mean()


def _direction_utility_trade_conviction_term(
    logits: torch.Tensor,
    y_long_utility_bps: torch.Tensor,
    y_short_utility_bps: torch.Tensor,
    y_long_bad_path: torch.Tensor,
    y_short_bad_path: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3:
        return zero
    long_u = y_long_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    short_u = y_short_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    long_bad = y_long_bad_path.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    short_bad = y_short_bad_path.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if (
        long_u.shape[0] != logits.shape[0]
        or short_u.shape[0] != logits.shape[0]
        or long_bad.shape[0] != logits.shape[0]
        or short_bad.shape[0] != logits.shape[0]
    ):
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_SHAPE_MISMATCH] "
            f"logits={tuple(logits.shape)} long={tuple(long_u.shape)} short={tuple(short_u.shape)} "
            f"long_bad={tuple(long_bad.shape)} short_bad={tuple(short_bad.shape)}"
        )
    min_gap = float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS)
    min_utility = float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS)
    max_bad_path = float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH)
    margin = float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN)
    if min_gap < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_INVALID] "
            f"ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS={min_gap:.6f} expected >=0.0"
        )
    if max_bad_path < 0.0 or max_bad_path > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH_INVALID] "
            f"ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH={max_bad_path:.6f} expected [0.0, 1.0]"
        )
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN_INVALID] "
            f"ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN={margin:.6f} expected >=0.0"
        )

    gap = long_u - short_u
    long_mask = (gap >= min_gap) & (long_u >= min_utility) & (long_bad <= max_bad_path)
    short_mask = (gap <= -min_gap) & (short_u >= min_utility) & (short_bad <= max_bad_path)
    terms: list[torch.Tensor] = []
    if bool(long_mask.any().detach().cpu().item()):
        wrong = torch.maximum(logits[:, 1], logits[:, 2])
        terms.append(nn.functional.softplus(wrong[long_mask] - logits[long_mask, 0] + margin).mean())
    if bool(short_mask.any().detach().cpu().item()):
        wrong = torch.maximum(logits[:, 0], logits[:, 2])
        terms.append(nn.functional.softplus(wrong[short_mask] - logits[short_mask, 1] + margin).mean())
    if not terms:
        return zero
    return weight * torch.stack(terms).mean()


def _direction_utility_triad_ce_term(
    logits: torch.Tensor,
    y_long_utility_bps: torch.Tensor,
    y_short_utility_bps: torch.Tensor,
    y_long_bad_path: torch.Tensor,
    y_short_bad_path: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3:
        return zero
    long_u = y_long_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    short_u = y_short_utility_bps.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    long_bad = y_long_bad_path.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    short_bad = y_short_bad_path.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    if (
        long_u.shape[0] != logits.shape[0]
        or short_u.shape[0] != logits.shape[0]
        or long_bad.shape[0] != logits.shape[0]
        or short_bad.shape[0] != logits.shape[0]
    ):
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRIAD_CE_SHAPE_MISMATCH] "
            f"logits={tuple(logits.shape)} long={tuple(long_u.shape)} short={tuple(short_u.shape)} "
            f"long_bad={tuple(long_bad.shape)} short_bad={tuple(short_bad.shape)}"
        )
    min_gap = float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS)
    min_utility = float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS)
    max_bad_path = float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH)
    class_weight_cap = float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP)
    if min_gap < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_INVALID] "
            f"ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS={min_gap:.6f} expected >=0.0"
        )
    if max_bad_path < 0.0 or max_bad_path > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH_INVALID] "
            f"ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH={max_bad_path:.6f} expected [0.0, 1.0]"
        )
    if class_weight_cap < 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP_INVALID] "
            f"ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP={class_weight_cap:.6f} expected >=1.0"
        )

    gap = long_u - short_u
    long_mask = (gap >= min_gap) & (long_u >= min_utility) & (long_bad <= max_bad_path)
    short_mask = (gap <= -min_gap) & (short_u >= min_utility) & (short_bad <= max_bad_path)
    triad_target = torch.full((logits.shape[0],), 2, device=logits.device, dtype=torch.long)
    triad_target[long_mask] = 0
    triad_target[short_mask] = 1

    losses = nn.functional.cross_entropy(logits, triad_target, reduction="none")
    class_counts = torch.bincount(triad_target, minlength=3).to(device=logits.device, dtype=logits.dtype)
    present = class_counts > 0
    class_weights = torch.ones(3, device=logits.device, dtype=logits.dtype)
    if bool(present.any().detach().cpu().item()):
        present_count = present.to(dtype=logits.dtype).sum().clamp_min(1.0)
        total_present = class_counts[present].sum().clamp_min(1.0)
        balanced = total_present / (present_count * class_counts[present].clamp_min(1.0))
        class_weights[present] = torch.clamp(balanced, min=1.0 / class_weight_cap, max=class_weight_cap)
    return weight * (losses * class_weights[triad_target]).mean()


def _direction_flat_starvation_term(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
    weight = float(ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT)
    if weight <= 0.0:
        return zero
    if logits.ndim != 2 or logits.shape[1] < 3 or len(targets) != logits.shape[0]:
        return zero

    min_label_rate = float(ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE)
    min_rows = int(ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS)
    pred_fraction = float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION)
    pred_floor = float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR)
    margin = float(ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN)
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS={min_rows} expected >=2"
        )
    if pred_fraction < 0.0 or pred_fraction > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION_INVALID] "
            f"ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION={pred_fraction:.6f} expected [0.0, 1.0]"
        )
    if pred_floor < 0.0 or pred_floor > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR_INVALID] "
            f"ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR={pred_floor:.6f} expected [0.0, 1.0]"
        )
    if margin < 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN_INVALID] "
            f"ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN={margin:.6f} expected >=0.0"
        )

    target_i = targets.long().to(device=logits.device)
    flat_mask = target_i == 2
    flat_rows = int(flat_mask.sum().detach().cpu().item())
    if flat_rows <= 0:
        return zero
    flat_label_rate = torch.as_tensor(flat_rows / max(1, logits.shape[0]), device=logits.device, dtype=logits.dtype)
    if float(flat_label_rate.detach().cpu().item()) < min_label_rate:
        return zero

    probs = torch.softmax(logits, dim=1)
    hardish_probs = _direction_pred_rate_probs(probs)
    required_global = torch.maximum(
        flat_label_rate * torch.as_tensor(pred_fraction, device=logits.device, dtype=logits.dtype),
        torch.as_tensor(pred_floor, device=logits.device, dtype=logits.dtype),
    )
    values: list[torch.Tensor] = [torch.relu(required_global - hardish_probs[:, 2].mean())]

    flat_logits = logits[flat_mask]
    wrong_side = torch.maximum(flat_logits[:, 0], flat_logits[:, 1])
    values.append(nn.functional.softplus(wrong_side - flat_logits[:, 2] + margin).mean())

    if ctx_cat is not None and ctx_cat.ndim == 2 and ctx_cat.shape[0] == logits.shape[0]:
        indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
        for idx in indices:
            slice_values = torch.unique(ctx_cat[:, idx].long())
            for value in slice_values:
                mask = ctx_cat[:, idx].long().to(device=logits.device) == value.to(device=logits.device)
                rows = int(mask.sum().detach().cpu().item())
                if rows < min_rows:
                    continue
                slice_targets = target_i[mask]
                flat_count = int((slice_targets == 2).sum().detach().cpu().item())
                if flat_count <= 0:
                    continue
                flat_rate = torch.as_tensor(flat_count / rows, device=logits.device, dtype=logits.dtype)
                if float(flat_rate.detach().cpu().item()) < min_label_rate:
                    continue
                required = torch.maximum(
                    flat_rate * torch.as_tensor(pred_fraction, device=logits.device, dtype=logits.dtype),
                    torch.as_tensor(pred_floor, device=logits.device, dtype=logits.dtype),
                )
                values.append(torch.relu(required - hardish_probs[mask, 2].mean()))

    return weight * _direction_slice_loss_aggregate(values)


def _build_cost_sensitive_criterion(
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    cost_long_to_short: float,
    cost_long_to_flat: float,
    cost_short_to_long: float,
    cost_short_to_flat: float,
    cost_flat_to_long: float,
    cost_flat_to_short: float,
    cost_scale: float,
    enabled: bool,
    balance_alpha: float,
    balance_target: str,
    balance_class_weights: torch.Tensor,
) -> Tuple[CostSensitiveCrossEntropyLoss, torch.Tensor]:
    cost_matrix = torch.tensor(
        [
            [0.0, float(cost_long_to_short), float(cost_long_to_flat)],
            [float(cost_short_to_long), 0.0, float(cost_short_to_flat)],
            [float(cost_flat_to_long), float(cost_flat_to_short), 0.0],
        ],
        device=device,
    )
    criterion = CostSensitiveCrossEntropyLoss(
        class_weights=class_weights,
        cost_matrix=cost_matrix,
        cost_scale=float(cost_scale),
        enabled=bool(enabled),
        balance_alpha=float(balance_alpha),
        balance_target=str(balance_target),
        balance_class_weights=balance_class_weights,
    )
    return criterion, cost_matrix


def _probability_gate_regularization(
    out: dict[str, Any],
    device: torch.device,
    *,
    output_name: str,
    expected_width: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    gate = out.get(output_name)
    if not isinstance(gate, torch.Tensor):
        raise RuntimeError(
            f"[ENTRY_MODEL_NATIVE_ACTIVE_HEAD_MISSING] output={output_name}"
        )
    if gate.ndim != 2 or gate.shape[1] != expected_width or gate.numel() == 0:
        raise RuntimeError(
            "[ENTRY_MODEL_NATIVE_GATE_SHAPE_INVALID] "
            f"output={output_name} shape={tuple(gate.shape)} "
            f"expected=(batch,{expected_width})"
        )
    if not bool(torch.isfinite(gate).all().item()):
        raise RuntimeError(
            f"[ENTRY_MODEL_NATIVE_GATE_NONFINITE] output={output_name}"
        )
    gate = gate.float().clamp(min=1e-8)
    mean_gate = gate.mean(dim=0)
    entropy = -(gate * gate.log()).sum(dim=1).mean()
    max_entropy = torch.log(torch.tensor(float(gate.shape[1]), device=device, dtype=gate.dtype))
    # Prevent one-token collapse without forcing every market state to use an
    # almost-uniform gate. Dynamic specialization is the point of these gates.
    entropy_floor = 0.5 * max_entropy
    entropy_loss = (entropy_floor - entropy).clamp_min(0.0)
    uniform = torch.full_like(mean_gate, 1.0 / float(mean_gate.numel()))
    kl_uniform = (uniform * (uniform.clamp(min=1e-8).log() - mean_gate.clamp(min=1e-8).log())).sum()
    floor = torch.tensor(float(ENTRY_SPECIALIST_GATE_MIN_MEAN), device=device, dtype=gate.dtype)
    floor_hinge = torch.relu(floor - mean_gate).sum()
    loss = (
        float(ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT) * entropy_loss
        + float(ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT) * (kl_uniform + floor_hinge)
    )
    return loss, {
        "entropy": float(entropy.detach().cpu().item()),
        "min_mean": float(mean_gate.min().detach().cpu().item()),
        "kl_uniform": float(kl_uniform.detach().cpu().item()),
        "floor_hinge": float(floor_hinge.detach().cpu().item()),
    }


def _specialist_gate_regularization(
    out: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep every learned cooperation gate live without prescribing direction.

    The loss only prevents token starvation/collapse.  It does not encode any
    LONG/SHORT/FLAT preference or hand-written confluence rule.
    """
    losses: list[torch.Tensor] = []
    all_stats: dict[str, float] = {}
    for output_name, expected_width in _MODEL_NATIVE_COOPERATION_GATE_WIDTHS.items():
        gate_loss, gate_stats = _probability_gate_regularization(
            out,
            device,
            output_name=output_name,
            expected_width=expected_width,
        )
        losses.append(gate_loss)
        prefix = "specialist_gate" if output_name == "specialist_gate" else output_name
        for key, value in gate_stats.items():
            all_stats[f"{prefix}_{key}"] = value
    # Preserve the historical specialist statistic names consumed by logs.
    all_stats.update(
        {
            "entropy": all_stats["specialist_gate_entropy"],
            "min_mean": all_stats["specialist_gate_min_mean"],
            "kl_uniform": all_stats["specialist_gate_kl_uniform"],
            "floor_hinge": all_stats["specialist_gate_floor_hinge"],
        }
    )
    return torch.stack(losses).mean(), all_stats


def _new_cooperation_gate_epoch_accumulator() -> dict[str, dict[str, Any]]:
    return {
        output_name: {
            "rows": 0,
            "sum": np.zeros(expected_width, dtype=np.float64),
            "entropy_sum": 0.0,
        }
        for output_name, expected_width in _MODEL_NATIVE_COOPERATION_GATE_WIDTHS.items()
    }


def _accumulate_cooperation_gate_epoch(
    accumulator: dict[str, dict[str, Any]],
    out: dict[str, Any],
) -> None:
    """Accumulate exact epoch-wide cooperation use, not batch-minimum proxies."""
    for output_name, expected_width in _MODEL_NATIVE_COOPERATION_GATE_WIDTHS.items():
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
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for output_name, expected_width in _MODEL_NATIVE_COOPERATION_GATE_WIDTHS.items():
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


def _cooperation_gate_health_failures(stats: dict[str, Any]) -> list[str]:
    """Fail checkpoint admission when any learned cooperation path is starved."""
    failures: list[str] = []
    min_mean = float(ENTRY_SPECIALIST_GATE_MIN_MEAN)
    for output_name, expected_width in _MODEL_NATIVE_COOPERATION_GATE_WIDTHS.items():
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
        if observed_min <= min_mean:
            failures.append(
                f"{output_name} min mean={observed_min:.6f} "
                f"(must be > {min_mean:.6f})"
            )
        entropy_floor = 0.5 * float(np.log(float(expected_width)))
        if entropy is None or not np.isfinite(float(entropy)):
            failures.append(f"{output_name} epoch-wide entropy evidence is missing")
        elif float(entropy) < entropy_floor:
            failures.append(
                f"{output_name} entropy={float(entropy):.6f} "
                f"(must be >= {entropy_floor:.6f})"
            )
    return failures


def _bad_path_quality_rank_loss(
    bad_path_logit: torch.Tensor,
    path_quality_bps: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    zero = torch.zeros((), device=device)
    if float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT) <= 0.0:
        return zero
    if not isinstance(bad_path_logit, torch.Tensor) or bad_path_logit.numel() == 0:
        raise RuntimeError(
            "[ENTRY_MODEL_NATIVE_ACTIVE_HEAD_MISSING] output=bad_path_logit"
        )
    logits = bad_path_logit.reshape(-1).float()
    quality = path_quality_bps.reshape(-1).float()
    finite = torch.isfinite(logits) & torch.isfinite(quality)
    if int(finite.sum().detach().cpu().item()) < 8:
        return zero
    logits = logits[finite]
    quality = quality[finite]
    if not torch.isfinite(quality).all() or (quality.max() - quality.min()).abs() <= 1e-6:
        return zero
    q = min(0.45, max(0.05, float(ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE)))
    low_cut = torch.quantile(quality.detach(), q)
    high_cut = torch.quantile(quality.detach(), 1.0 - q)
    low_mask = quality <= low_cut
    high_mask = quality >= high_cut
    if int(low_mask.sum().detach().cpu().item()) < 2 or int(high_mask.sum().detach().cpu().item()) < 2:
        return zero
    low_bad_logit = logits[low_mask].mean()
    high_bad_logit = logits[high_mask].mean()
    rank_gap = low_bad_logit - high_bad_logit
    margin = torch.tensor(float(ENTRY_BAD_PATH_QUALITY_RANK_MARGIN), device=device, dtype=logits.dtype)
    return float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT) * torch.relu(margin - rank_gap)


def _path_quality_rank_loss(
    path_quality_pred: torch.Tensor,
    path_quality_bps: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    zero = torch.zeros((), device=device)
    if float(ENTRY_PATH_QUALITY_RANK_WEIGHT) <= 0.0:
        return zero
    if not isinstance(path_quality_pred, torch.Tensor) or path_quality_pred.numel() == 0:
        raise RuntimeError(
            "[ENTRY_MODEL_NATIVE_ACTIVE_HEAD_MISSING] output=path_quality"
        )
    pred = path_quality_pred.reshape(-1).float()
    quality = path_quality_bps.reshape(-1).float()
    finite = torch.isfinite(pred) & torch.isfinite(quality)
    if int(finite.sum().detach().cpu().item()) < 8:
        return zero
    pred = pred[finite]
    quality = quality[finite]
    if not torch.isfinite(quality).all() or (quality.max() - quality.min()).abs() <= 1e-6:
        return zero
    q = min(0.45, max(0.05, float(ENTRY_PATH_QUALITY_RANK_QUANTILE)))
    low_cut = torch.quantile(quality.detach(), q)
    high_cut = torch.quantile(quality.detach(), 1.0 - q)
    low_mask = quality <= low_cut
    high_mask = quality >= high_cut
    if int(low_mask.sum().detach().cpu().item()) < 2 or int(high_mask.sum().detach().cpu().item()) < 2:
        return zero
    low_pred = pred[low_mask].mean()
    high_pred = pred[high_mask].mean()
    rank_gap = high_pred - low_pred
    margin = torch.tensor(float(ENTRY_PATH_QUALITY_RANK_MARGIN), device=device, dtype=pred.dtype)
    return float(ENTRY_PATH_QUALITY_RANK_WEIGHT) * torch.relu(margin - rank_gap)


def _tail_direction_mask(
    y: torch.Tensor,
    y_tradable: torch.Tensor,
    y_bad_path: torch.Tensor,
    path_quality_bps: torch.Tensor,
) -> torch.Tensor:
    finite_quality = torch.isfinite(path_quality_bps.float())
    directional = y.long() != 2
    tradable = y_tradable.float() > 0.5
    clean_path = y_bad_path.float() <= 0.5
    base = finite_quality & directional & tradable & clean_path
    if int(base.sum().detach().cpu().item()) < int(ENTRY_TAIL_DIRECTION_MIN_BATCH):
        return torch.zeros_like(base, dtype=torch.bool)
    quality = path_quality_bps.float()[base]
    if quality.numel() == 0 or (quality.max() - quality.min()).abs() <= 1e-6:
        return torch.zeros_like(base, dtype=torch.bool)
    q = min(0.95, max(0.50, float(ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE)))
    cutoff = torch.quantile(quality.detach(), q)
    mask = base & (path_quality_bps.float() >= cutoff)
    if int(mask.sum().detach().cpu().item()) < int(ENTRY_TAIL_DIRECTION_MIN_BATCH):
        return torch.zeros_like(base, dtype=torch.bool)
    return mask


def _hard_negative_residual(
    y_hard_negative: torch.Tensor,
    y_dead_negative: torch.Tensor,
    y_teaser_negative: torch.Tensor,
) -> torch.Tensor:
    return torch.clamp(
        y_hard_negative.float() - y_dead_negative.float() - y_teaser_negative.float(),
        min=0.0,
        max=1.0,
    )


def _direction_ce_sample_weight(
    y_bad_path: torch.Tensor,
    y_dead_negative_long: torch.Tensor,
    y_teaser_negative_long: torch.Tensor,
    residual_hard_neg_long: torch.Tensor,
    y_dead_negative_short: torch.Tensor,
    y_teaser_negative_short: torch.Tensor,
    residual_hard_neg_short: torch.Tensor,
) -> torch.Tensor:
    ce_sample_weight = torch.ones_like(y_bad_path.float())
    if float(ENTRY_DEAD_LONG_CE_MULTIPLIER) > 1.0:
        ce_sample_weight = ce_sample_weight + (
            (float(ENTRY_DEAD_LONG_CE_MULTIPLIER) - 1.0) * y_dead_negative_long.float()
        )
    if float(ENTRY_TEASER_LONG_CE_MULTIPLIER) > 1.0:
        ce_sample_weight = ce_sample_weight + (
            (float(ENTRY_TEASER_LONG_CE_MULTIPLIER) - 1.0) * y_teaser_negative_long.float()
        )
    if float(ENTRY_BAD_PATH_CE_MULTIPLIER) > 1.0:
        ce_sample_weight = ce_sample_weight + (
            (float(ENTRY_BAD_PATH_CE_MULTIPLIER) - 1.0) * y_bad_path.float()
        )
    if float(ENTRY_HARD_NEG_LONG_CE_MULTIPLIER) > 1.0:
        ce_sample_weight = ce_sample_weight + (
            (float(ENTRY_HARD_NEG_LONG_CE_MULTIPLIER) - 1.0) * residual_hard_neg_long
        )
    if ENTRY_SYMMETRIC_NEGATIVES:
        if float(ENTRY_DEAD_LONG_CE_MULTIPLIER) > 1.0:
            ce_sample_weight = ce_sample_weight + (
                (float(ENTRY_DEAD_LONG_CE_MULTIPLIER) - 1.0) * y_dead_negative_short.float()
            )
        if float(ENTRY_TEASER_LONG_CE_MULTIPLIER) > 1.0:
            ce_sample_weight = ce_sample_weight + (
                (float(ENTRY_TEASER_LONG_CE_MULTIPLIER) - 1.0) * y_teaser_negative_short.float()
            )
        if float(ENTRY_HARD_NEG_LONG_CE_MULTIPLIER) > 1.0:
            ce_sample_weight = ce_sample_weight + (
                (float(ENTRY_HARD_NEG_LONG_CE_MULTIPLIER) - 1.0) * residual_hard_neg_short
            )
    return ce_sample_weight


def _direction_aux_ce_loss(
    aux_logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: CostSensitiveCrossEntropyLoss,
    ce_sample_weight: torch.Tensor,
) -> torch.Tensor:
    aux_ce_per = criterion.ce(aux_logits, targets)
    aux_weight = ce_sample_weight.to(device=aux_ce_per.device, dtype=aux_ce_per.dtype)
    aux_ce = (aux_ce_per * aux_weight).mean()
    aux_probs = torch.softmax(aux_logits, dim=1)
    aux_balance = _direction_balance_term(aux_probs, targets, criterion)
    return aux_ce + aux_balance


def _hierarchical_entry_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    trade_pos_weight: float,
    side_bad_path_pos_weight: Any,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Supervise the mandatory evidence heads consumed by direction fusion."""
    total = torch.tensor(0.0, device=device)
    stats: Dict[str, float] = {
        "hier_trade_loss": 0.0,
        "hier_trade_global_prior_loss": 0.0,
        "hier_slice_trade_prior_loss": 0.0,
        "hier_slice_trade_accuracy_edge_loss": 0.0,
        "hier_flat_logit_margin_loss": 0.0,
        "hier_slice_flat_logit_margin_loss": 0.0,
        "hier_side_loss": 0.0,
        "hier_slice_side_ce_loss": 0.0,
        "hier_slice_side_margin_loss": 0.0,
        "hier_slice_side_accuracy_edge_loss": 0.0,
        "hier_side_global_prior_loss": 0.0,
        "hier_slice_side_prior_loss": 0.0,
        "hier_utility_loss": 0.0,
        "hier_bad_path_loss": 0.0,
        "hier_mae_loss": 0.0,
        "hier_side_validity_loss": 0.0,
        "hier_long_valid_target_rate": 0.0,
        "hier_short_valid_target_rate": 0.0,
        "hier_long_valid_prob_mean": 0.0,
        "hier_short_valid_prob_mean": 0.0,
        "hier_side_rows": 0.0,
        "hier_side_acc": 0.0,
        "hier_long_bad_target_rate": 0.0,
        "hier_short_bad_target_rate": 0.0,
    }
    trade_logit = out["trade_logit"]
    side_logits = out["side_logits"]
    side_utility = out["side_utility"]
    side_bad_path_logit = out["side_bad_path_logit"]
    side_mae = out["side_mae"]
    side_validity_logit = out["side_validity_logit"]

    non_blocking = device.type == "cuda"
    y_trade = batch["y_trade"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    y_side = batch["y_side"].to(device, non_blocking=non_blocking).long().clamp(0, 1)
    y_side_mask = batch["y_side_mask"].to(device, non_blocking=non_blocking).float() > 0.5
    ctx_cat_value = batch["ctx_cat"]
    if not isinstance(ctx_cat_value, torch.Tensor):
        raise RuntimeError("[ENTRY_MODEL_NATIVE_ACTIVE_CONTEXT_MISSING] ctx_cat")
    ctx_cat = ctx_cat_value.to(device, non_blocking=non_blocking)
    y_long_util = batch["y_long_path_utility_bps"].to(device, non_blocking=non_blocking).float()
    y_short_util = batch["y_short_path_utility_bps"].to(device, non_blocking=non_blocking).float()
    y_long_bad = batch["y_long_bad_path"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    y_short_bad = batch["y_short_bad_path"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    y_long_mae = batch["y_long_expected_mae_bps"].to(device, non_blocking=non_blocking).float().clamp_min(0.0)
    y_short_mae = batch["y_short_expected_mae_bps"].to(device, non_blocking=non_blocking).float().clamp_min(0.0)

    # Side utility, bad-path, and MAE targets are forward-outcome contracts.  Do
    # not rewrite them from structural/trend labels here: those labels remain
    # available to the model through the full signal stack and the dedicated
    # rail auxiliary head, while direction is learned only from outcome truth.
    stats["hier_long_bad_target_rate"] = float(y_long_bad.detach().mean().cpu().item())
    stats["hier_short_bad_target_rate"] = float(y_short_bad.detach().mean().cpu().item())
    util_scale = max(1.0, float(ENTRY_AUX_PATH_SCALE_BPS))
    mae_scale = max(1.0, float(ENTRY_AUX_MFE_SCALE_BPS))
    valid_long_trade_target = (
        (y_long_util >= float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS))
        & (y_long_bad < 0.5)
    )
    valid_short_trade_target = (
        (y_short_util >= float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS))
        & (y_short_bad < 0.5)
    )

    if float(ENTRY_HIER_TRADE_WEIGHT) > 0.0:
        raw = nn.functional.binary_cross_entropy_with_logits(
            trade_logit.squeeze(1),
            y_trade,
            pos_weight=torch.tensor(float(trade_pos_weight), device=device, dtype=trade_logit.dtype),
        )
        weighted = float(ENTRY_HIER_TRADE_WEIGHT) * raw
        total = total + weighted
        stats["hier_trade_loss"] = float(weighted.detach().cpu().item())

    hier_trade_global_prior = _hier_trade_global_prior_match_term(trade_logit, y_trade)
    total = total + hier_trade_global_prior
    stats["hier_trade_global_prior_loss"] = float(hier_trade_global_prior.detach().cpu().item())
    hier_slice_trade_prior = _hier_slice_trade_prior_match_term(trade_logit, y_trade, ctx_cat)
    total = total + hier_slice_trade_prior
    stats["hier_slice_trade_prior_loss"] = float(hier_slice_trade_prior.detach().cpu().item())
    hier_slice_trade_accuracy_edge = _hier_slice_trade_accuracy_edge_term(
        trade_logit,
        y_trade,
        ctx_cat,
    )
    total = total + hier_slice_trade_accuracy_edge
    stats["hier_slice_trade_accuracy_edge_loss"] = float(
        hier_slice_trade_accuracy_edge.detach().cpu().item()
    )
    hier_flat_logit_margin = _hier_flat_logit_margin_term(trade_logit, y_trade)
    total = total + hier_flat_logit_margin
    stats["hier_flat_logit_margin_loss"] = float(hier_flat_logit_margin.detach().cpu().item())
    hier_slice_flat_logit_margin = _hier_slice_flat_logit_margin_term(trade_logit, y_trade, ctx_cat)
    total = total + hier_slice_flat_logit_margin
    stats["hier_slice_flat_logit_margin_loss"] = float(
        hier_slice_flat_logit_margin.detach().cpu().item()
    )
    if float(ENTRY_HIER_SIDE_WEIGHT) > 0.0 and y_side_mask.any():
        raw = nn.functional.cross_entropy(side_logits[y_side_mask], y_side[y_side_mask])
        weighted = float(ENTRY_HIER_SIDE_WEIGHT) * raw
        total = total + weighted
        pred_side = torch.argmax(side_logits[y_side_mask], dim=1)
        stats["hier_side_rows"] = float(int(y_side_mask.sum().detach().cpu().item()))
        stats["hier_side_acc"] = float((pred_side == y_side[y_side_mask]).float().mean().detach().cpu().item())
        stats["hier_side_loss"] = float(weighted.detach().cpu().item())
    hier_slice_side_ce = _hier_slice_side_balanced_ce_term(side_logits, y_side, y_side_mask, ctx_cat)
    total = total + hier_slice_side_ce
    stats["hier_slice_side_ce_loss"] = float(hier_slice_side_ce.detach().cpu().item())
    hier_slice_side_margin = _hier_slice_side_true_margin_term(side_logits, y_side, y_side_mask, ctx_cat)
    total = total + hier_slice_side_margin
    stats["hier_slice_side_margin_loss"] = float(hier_slice_side_margin.detach().cpu().item())
    hier_slice_side_accuracy_edge = _hier_slice_side_accuracy_edge_term(
        side_logits,
        y_side,
        y_side_mask,
        ctx_cat,
    )
    total = total + hier_slice_side_accuracy_edge
    stats["hier_slice_side_accuracy_edge_loss"] = float(
        hier_slice_side_accuracy_edge.detach().cpu().item()
    )
    hier_side_global_prior = _hier_side_global_prior_match_term(side_logits, y_side, y_side_mask)
    total = total + hier_side_global_prior
    stats["hier_side_global_prior_loss"] = float(hier_side_global_prior.detach().cpu().item())
    hier_slice_side_prior = _hier_slice_side_prior_match_term(side_logits, y_side, y_side_mask, ctx_cat)
    total = total + hier_slice_side_prior
    stats["hier_slice_side_prior_loss"] = float(hier_slice_side_prior.detach().cpu().item())

    if float(ENTRY_HIER_UTILITY_WEIGHT) > 0.0:
        util_target = (torch.stack([y_long_util, y_short_util], dim=1) / util_scale).to(dtype=side_utility.dtype)
        raw = nn.functional.smooth_l1_loss(side_utility, util_target)
        weighted = float(ENTRY_HIER_UTILITY_WEIGHT) * raw
        total = total + weighted
        stats["hier_utility_loss"] = float(weighted.detach().cpu().item())

    if float(ENTRY_HIER_BAD_PATH_WEIGHT) > 0.0:
        bad_target = torch.stack([y_long_bad, y_short_bad], dim=1).to(dtype=side_bad_path_logit.dtype)
        if isinstance(side_bad_path_pos_weight, (list, tuple, np.ndarray)):
            weights = [float(x) for x in list(side_bad_path_pos_weight)[:2]]
            if len(weights) != 2:
                raise RuntimeError(
                    "[ENTRY_HIER_BAD_PATH_POS_WEIGHT_INVALID] expected exactly two weights"
                )
        else:
            weights = [float(side_bad_path_pos_weight), float(side_bad_path_pos_weight)]
        raw = nn.functional.binary_cross_entropy_with_logits(
            side_bad_path_logit,
            bad_target,
            pos_weight=torch.tensor(
                weights,
                device=device,
                dtype=side_bad_path_logit.dtype,
            ),
        )
        weighted = float(ENTRY_HIER_BAD_PATH_WEIGHT) * raw
        total = total + weighted
        stats["hier_bad_path_loss"] = float(weighted.detach().cpu().item())

    if float(ENTRY_HIER_MAE_WEIGHT) > 0.0:
        mae_target = (torch.stack([y_long_mae, y_short_mae], dim=1) / mae_scale).to(dtype=side_mae.dtype)
        raw = nn.functional.smooth_l1_loss(side_mae, mae_target)
        weighted = float(ENTRY_HIER_MAE_WEIGHT) * raw
        total = total + weighted
        stats["hier_mae_loss"] = float(weighted.detach().cpu().item())

    if float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT) > 0.0:
        validity_target = torch.stack(
            [valid_long_trade_target.float(), valid_short_trade_target.float()],
            dim=1,
        ).to(dtype=side_validity_logit.dtype)
        pos = validity_target.sum(dim=0)
        neg = torch.clamp(torch.tensor(float(validity_target.shape[0]), device=device, dtype=validity_target.dtype) - pos, min=0.0)
        cap = max(1.0, float(ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP))
        pos_weight = torch.where(
            pos > 0.5,
            torch.clamp(neg / torch.clamp(pos, min=1.0), min=1.0, max=cap),
            torch.ones_like(pos),
        )
        raw = nn.functional.binary_cross_entropy_with_logits(
            side_validity_logit,
            validity_target,
            pos_weight=pos_weight.to(device=device, dtype=side_validity_logit.dtype),
        )
        weighted = float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT) * raw
        total = total + weighted
        valid_prob = torch.sigmoid(side_validity_logit.detach().float())
        stats["hier_side_validity_loss"] = float(weighted.detach().cpu().item())
        stats["hier_long_valid_target_rate"] = float(validity_target[:, 0].detach().mean().cpu().item())
        stats["hier_short_valid_target_rate"] = float(validity_target[:, 1].detach().mean().cpu().item())
        stats["hier_long_valid_prob_mean"] = float(valid_prob[:, 0].mean().cpu().item())
        stats["hier_short_valid_prob_mean"] = float(valid_prob[:, 1].mean().cpu().item())

    return total, stats


def _trendline_rail_aux_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Train rail evidence without imposing a hand-written direction mapping."""
    logits = _require_active_aux_head_prediction(
        out,
        batch,
        output_name="trendline_rail_logits",
        target_names=(
            "y_rising_channel_support_touch",
            "y_falling_channel_resistance_touch",
            "y_countertrend_short_trap",
            "y_countertrend_long_trap",
            "y_short_high_mae_low_mfe_early_failure",
            "y_long_high_mae_low_mfe_early_failure",
        ),
    )
    stats = {
        "trendline_rail_loss": 0.0,
        "trendline_rail_rows": 0.0,
        "trendline_rising_rows": 0.0,
        "trendline_falling_rows": 0.0,
    }
    if float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT) <= 0.0:
        raise RuntimeError(
            "[ENTRY_TRENDLINE_RAIL_HEAD_UNTRAINED] loss weight must be positive"
        )

    non_blocking = device.type == "cuda"
    rising = batch["y_rising_channel_support_touch"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    falling = batch["y_falling_channel_resistance_touch"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    short_trap = batch["y_countertrend_short_trap"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    long_trap = batch["y_countertrend_long_trap"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    short_early_fail = batch["y_short_high_mae_low_mfe_early_failure"].to(
        device, non_blocking=non_blocking
    ).float().clamp(0.0, 1.0)
    long_early_fail = batch["y_long_high_mae_low_mfe_early_failure"].to(
        device, non_blocking=non_blocking
    ).float().clamp(0.0, 1.0)
    targets = torch.stack(
        [rising, falling, short_trap, long_trap, short_early_fail, long_early_fail],
        dim=1,
    ).to(dtype=logits.dtype)
    if logits.ndim != 2 or logits.shape[1] != targets.shape[1]:
        raise RuntimeError(
            "[ENTRY_TRENDLINE_RAIL_OUTPUT_DIM_MISMATCH] "
            f"logits_shape={tuple(logits.shape)} targets_shape={tuple(targets.shape)}"
        )
    loss = float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT) * nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    )
    stats["trendline_rail_loss"] = float(loss.detach().cpu().item())
    stats["trendline_rail_rows"] = float(int((targets.max(dim=1).values > 0.5).sum().detach().cpu().item()))
    stats["trendline_rising_rows"] = float(int((rising > 0.5).sum().detach().cpu().item()))
    stats["trendline_falling_rows"] = float(int((falling > 0.5).sum().detach().cpu().item()))
    return loss, stats


def _aux_selector_mask(
    y_selector_long_mask: torch.Tensor,
    y_selector_short_mask: torch.Tensor,
) -> torch.Tensor:
    if ENTRY_SYMMETRIC_NEGATIVES:
        return (y_selector_long_mask.float() + y_selector_short_mask.float()) > 0.5
    return y_selector_long_mask.float() > 0.5


def _active_aux_target(
    y_target_long: torch.Tensor,
    y_target_bidir: torch.Tensor,
) -> torch.Tensor:
    """Select the exact target semantics used by the active BCE objective."""
    return (y_target_bidir if ENTRY_SYMMETRIC_NEGATIVES else y_target_long).float()


def _aux_clean_edge_target(
    y_clean_edge_long: torch.Tensor,
    y_clean_edge_bidir: torch.Tensor,
) -> torch.Tensor:
    return _active_aux_target(y_clean_edge_long, y_clean_edge_bidir)


def _aux_survival_target(
    y_survival_long: torch.Tensor,
    y_survival_bidir: torch.Tensor,
) -> torch.Tensor:
    return _active_aux_target(y_survival_long, y_survival_bidir)


def _active_aux_target_rate_from_frame(
    frame: pd.DataFrame,
    *,
    split_name: str,
    target_name: str,
    long_column: str,
    bidir_column: str,
) -> float:
    """Measure the same target, on the same selector mask, as the active BCE."""
    required = (
        long_column,
        bidir_column,
        "y_selector_long_mask",
        "y_selector_short_mask",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(
            "[ENTRY_AUX_TARGET_RATE_COLUMNS_MISSING] "
            f"split={split_name} target={target_name} missing={missing}"
        )
    target = _active_aux_target(
        torch.as_tensor(frame[long_column].to_numpy(dtype=np.float32, copy=False)),
        torch.as_tensor(frame[bidir_column].to_numpy(dtype=np.float32, copy=False)),
    )
    selector_mask = _aux_selector_mask(
        torch.as_tensor(
            frame["y_selector_long_mask"].to_numpy(dtype=np.float32, copy=False)
        ),
        torch.as_tensor(
            frame["y_selector_short_mask"].to_numpy(dtype=np.float32, copy=False)
        ),
    )
    if target.ndim != 1 or selector_mask.ndim != 1 or target.shape != selector_mask.shape:
        raise RuntimeError(
            "[ENTRY_AUX_TARGET_RATE_SHAPE_INVALID] "
            f"split={split_name} target={target_name} "
            f"target_shape={tuple(target.shape)} selector_shape={tuple(selector_mask.shape)}"
        )
    if not bool(selector_mask.any()):
        raise RuntimeError(
            "[ENTRY_AUX_TARGET_RATE_EMPTY_SELECTOR] "
            f"split={split_name} target={target_name}"
        )
    active_target = target[selector_mask]
    if not bool(torch.isfinite(active_target).all()):
        raise RuntimeError(
            "[ENTRY_AUX_TARGET_RATE_NONFINITE] "
            f"split={split_name} target={target_name}"
        )
    if bool(((active_target != 0.0) & (active_target != 1.0)).any()):
        raise RuntimeError(
            "[ENTRY_AUX_TARGET_RATE_NOT_BINARY] "
            f"split={split_name} target={target_name}"
        )
    return float(active_target.mean().item())


def _positive_class_weight_from_rate(rate: float, cap: float) -> tuple[float, float]:
    """Return the raw and capped BCE positive weight for an exact active rate."""
    rate = float(rate)
    cap = float(cap)
    if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise RuntimeError(f"[ENTRY_AUX_TARGET_RATE_INVALID] rate={rate}")
    if not np.isfinite(cap) or cap < 1.0:
        raise RuntimeError(f"[ENTRY_AUX_POS_WEIGHT_CAP_INVALID] cap={cap}")
    raw = ((1.0 - rate) / max(rate, 1e-9)) if rate > 0.0 else 1.0
    return float(raw), float(min(cap, max(1.0, raw)))


def _selected_side_bad_path_probability_penalty(
    probs: torch.Tensor,
    y_direction: torch.Tensor,
    y_bad_path: torch.Tensor,
    penalty_weight: float,
) -> torch.Tensor:
    """Penalize only the selected LONG/SHORT probability on bad-path rows."""
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise RuntimeError(
            "[ENTRY_SELECTED_SIDE_BAD_PATH_PROB_SHAPE_INVALID] "
            f"probs_shape={tuple(probs.shape)}"
        )
    direction = y_direction.reshape(-1).long()
    bad_path = y_bad_path.reshape(-1).float()
    if direction.shape[0] != probs.shape[0] or bad_path.shape[0] != probs.shape[0]:
        raise RuntimeError(
            "[ENTRY_SELECTED_SIDE_BAD_PATH_TARGET_SHAPE_INVALID] "
            f"probs_rows={probs.shape[0]} direction_rows={direction.shape[0]} "
            f"bad_path_rows={bad_path.shape[0]}"
        )
    if not bool(torch.isfinite(bad_path).all()):
        raise RuntimeError("[ENTRY_SELECTED_SIDE_BAD_PATH_TARGET_NONFINITE]")
    weight = float(penalty_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise RuntimeError(
            f"[ENTRY_SELECTED_SIDE_BAD_PATH_WEIGHT_INVALID] weight={weight}"
        )

    bad_mask = bad_path > 0.5
    flat_bad_mask = bad_mask & (direction == 2)
    if bool(flat_bad_mask.any()):
        raise RuntimeError(
            "[ENTRY_SELECTED_SIDE_BAD_PATH_FLAT_TARGET_INVALID] "
            f"rows={int(flat_bad_mask.sum().item())}"
        )
    invalid_direction_mask = bad_mask & ((direction < 0) | (direction > 1))
    if bool(invalid_direction_mask.any()):
        invalid = sorted(
            {
                int(value)
                for value in direction[invalid_direction_mask].detach().cpu().tolist()
            }
        )
        raise RuntimeError(
            "[ENTRY_SELECTED_SIDE_BAD_PATH_DIRECTION_INVALID] "
            f"classes={invalid}"
        )
    if weight == 0.0 or not bool(bad_mask.any()):
        return probs.sum() * 0.0

    selected_side = direction[bad_mask]
    selected_side_prob = probs[bad_mask].gather(
        1,
        selected_side.unsqueeze(1),
    ).squeeze(1)
    return weight * selected_side_prob.mean()


def _signed_scaled_aux_regression_target(
    values: torch.Tensor,
    positive_mask: torch.Tensor,
    scale_bps: float,
) -> torch.Tensor:
    """Preserve the exact signed forward-outcome target used by the dataset."""
    return values[positive_mask].float() / max(1.0, float(scale_bps))


def _clean_edge_rank_masks(
    y_clean_edge_long: torch.Tensor,
    y_clean_edge_bidir: torch.Tensor,
    y_dead_negative_long: torch.Tensor,
    y_teaser_negative_long: torch.Tensor,
    residual_hard_neg_long: torch.Tensor,
    y_dead_negative_short: torch.Tensor,
    y_teaser_negative_short: torch.Tensor,
    residual_hard_neg_short: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ENTRY_SYMMETRIC_NEGATIVES:
        clean_pos = y_clean_edge_bidir.float() > 0.5
        ranked_neg = (
            (y_dead_negative_long.float() > 0.5)
            | (y_teaser_negative_long.float() > 0.5)
            | (residual_hard_neg_long.float() > 0.5)
            | (y_dead_negative_short.float() > 0.5)
            | (y_teaser_negative_short.float() > 0.5)
            | (residual_hard_neg_short.float() > 0.5)
        )
        return clean_pos, ranked_neg

    clean_pos = y_clean_edge_long.float() > 0.5
    ranked_neg = (
        (y_dead_negative_long.float() > 0.5)
        | (y_teaser_negative_long.float() > 0.5)
        | (residual_hard_neg_long.float() > 0.5)
    )
    return clean_pos, ranked_neg


def _step_partial_gradient_accumulation(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    configured_steps: int,
    observed_steps: int,
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
    if scheduler is not None:
        scheduler.step()
    return True


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    aux_path_weight: float,
    aux_mfe_weight: float,
    aux_tradable_weight: float,
    aux_path_scale_bps: float,
    aux_mfe_scale_bps: float,
    tradable_pos_weight: float,
    clean_edge_pos_weight: float,
    survival_pos_weight: float,
    bad_path_pos_weight: float,
    hier_trade_pos_weight: float,
    hier_bad_path_pos_weight: Any,
    scheduler=None,  # GX1_FAST_TRAIN: cosine+warmup scheduler, stepped per opt.step()
    grad_accum_steps: int = 1,
):
    model.train()
    _accum_steps = int(grad_accum_steps)
    if _accum_steps < 1:
        raise RuntimeError(
            f"[ENTRY_GRAD_ACCUM_STEPS_INVALID] observed={_accum_steps} expected>=1"
        )
    if _accum_steps > 1:
        log.info("[GRAD_ACCUM] accumulating gradients over %d batches per optimizer step", _accum_steps)
    _accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    total_ce = 0.0
    total_cost = 0.0
    total_balance = 0.0
    total_direction_min_pred = 0.0
    total_direction_global_prior_match = 0.0
    total_direction_slice_min_pred = 0.0
    total_direction_slice_recall = 0.0
    total_direction_slice_balanced_ce = 0.0
    total_direction_slice_true_margin = 0.0
    total_direction_slice_accuracy_edge = 0.0
    total_direction_slice_confusion_pair = 0.0
    total_direction_slice_prior_match = 0.0
    total_direction_flat_margin = 0.0
    total_direction_utility_margin = 0.0
    total_direction_side_utility_conviction = 0.0
    total_direction_utility_trade_conviction = 0.0
    total_direction_utility_triad_ce = 0.0
    total_direction_flat_starvation = 0.0
    total_tail_direction = 0.0
    specialist_gate_loss_sum = 0.0
    cooperation_gate_epoch = _new_cooperation_gate_epoch_accumulator()
    bad_path_quality_rank_loss_sum = 0.0
    path_quality_rank_loss_sum = 0.0
    n = 0
    short_total = 0
    short_pred_long = 0
    path_loss_sum = 0.0
    mfe_loss_sum = 0.0
    tradable_loss_sum = 0.0
    clean_edge_loss_sum = 0.0
    survival_loss_sum = 0.0
    clean_edge_rank_loss_sum = 0.0
    aux_bad_path_bce_loss_sum = 0.0
    bad_path_prob_penalty_loss_sum = 0.0
    hard_neg_prob_loss_sum = 0.0
    tail_direction_rows = 0
    hier_trade_loss_sum = 0.0
    hier_trade_global_prior_loss_sum = 0.0
    hier_slice_trade_prior_loss_sum = 0.0
    hier_slice_trade_accuracy_edge_loss_sum = 0.0
    hier_flat_logit_margin_loss_sum = 0.0
    hier_slice_flat_logit_margin_loss_sum = 0.0
    hier_side_loss_sum = 0.0
    hier_slice_side_ce_loss_sum = 0.0
    hier_slice_side_margin_loss_sum = 0.0
    hier_slice_side_accuracy_edge_loss_sum = 0.0
    hier_side_global_prior_loss_sum = 0.0
    hier_slice_side_prior_loss_sum = 0.0
    hier_utility_loss_sum = 0.0
    hier_side_bad_path_loss_sum = 0.0
    hier_side_mae_loss_sum = 0.0
    hier_side_validity_loss_sum = 0.0
    hier_long_valid_target_rate_sum = 0.0
    hier_short_valid_target_rate_sum = 0.0
    hier_long_valid_prob_sum = 0.0
    hier_short_valid_prob_sum = 0.0
    hier_long_bad_target_rate_sum = 0.0
    hier_short_bad_target_rate_sum = 0.0
    hier_countertrend_long_trap_rate_sum = 0.0
    hier_countertrend_short_trap_rate_sum = 0.0
    hier_side_rows_sum = 0
    hier_side_correct_sum = 0.0
    trendline_rail_loss_sum = 0.0
    trendline_rail_rows_sum = 0
    trendline_rising_rows_sum = 0
    trendline_falling_rows_sum = 0

    for batch in loader:
        non_blocking = device.type == "cuda"
        seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
        snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
        ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
        ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
        y = batch["y"].to(device, non_blocking=non_blocking)
        y_mfe_first = batch["mfe_first_n_bps"].to(device, non_blocking=non_blocking)
        y_path_quality = batch["path_quality_bps"].to(device, non_blocking=non_blocking)
        y_tradable = batch["y_tradable"].to(device, non_blocking=non_blocking)
        y_bad_path = batch["y_bad_path"].to(device, non_blocking=non_blocking)
        y_dead_negative_long = batch["y_dead_negative_long"].to(device, non_blocking=non_blocking)
        y_teaser_negative_long = batch["y_teaser_negative_long"].to(device, non_blocking=non_blocking)
        y_hard_negative_long = batch["y_hard_negative_long"].to(device, non_blocking=non_blocking)
        y_dead_negative_short = batch["y_dead_negative_short"].to(device, non_blocking=non_blocking)
        y_teaser_negative_short = batch["y_teaser_negative_short"].to(device, non_blocking=non_blocking)
        y_hard_negative_short = batch["y_hard_negative_short"].to(device, non_blocking=non_blocking)
        y_clean_edge_long = batch["y_clean_edge_long"].to(device, non_blocking=non_blocking)
        y_survival_long = batch["y_survival_long"].to(device, non_blocking=non_blocking)
        y_selector_long_mask = batch["y_selector_long_mask"].to(device, non_blocking=non_blocking)
        # SYM (run_id v10_symmetric_negatives_20260603): short-side selector + bidir quality
        # labels (already built in the dataset, never read by cement). Used only when symmetric.
        y_selector_short_mask = batch["y_selector_short_mask"].to(device, non_blocking=non_blocking)
        y_clean_edge_bidir = batch["y_clean_edge_bidir"].to(device, non_blocking=non_blocking)
        y_survival_bidir = batch["y_survival_bidir"].to(device, non_blocking=non_blocking)

        # Grad accum: zero_grad happens AFTER step (or at start of epoch).
        # See loss.backward() / optimizer.step() block below for the gated step.
        out = _autocast_forward(model, seq_x.device, seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **_multi_tf_kwargs_from_batch(batch, seq_x.device))
        logits = out["direction_logits"]
        path_pred = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="path_quality",
            target_names=("path_quality_bps",),
        )
        path_log_var = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="path_quality_log_var",
            target_names=("path_quality_bps",),
        )
        mfe_pred = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="mfe_first_n",
            target_names=("mfe_first_n_bps",),
        )
        tradable_logit = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="tradable_logit",
            target_names=("y_tradable",),
        )
        bad_path_logit = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="bad_path_logit",
            target_names=("y_bad_path",),
        )
        clean_edge_logit = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="clean_edge_logit",
            target_names=("y_clean_edge_long", "y_clean_edge_bidir"),
        )
        survival_logit = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="survival_logit",
            target_names=("y_survival_long", "y_survival_bidir"),
        )
        specialist_gate_loss, specialist_gate_stats = _specialist_gate_regularization(out, device)
        _accumulate_cooperation_gate_epoch(cooperation_gate_epoch, out)
        bad_path_quality_rank_loss = _bad_path_quality_rank_loss(bad_path_logit, y_path_quality, device)
        path_quality_rank_loss = _path_quality_rank_loss(path_pred, y_path_quality, device)

        residual_hard_neg_long = _hard_negative_residual(
            y_hard_negative_long, y_dead_negative_long, y_teaser_negative_long
        )
        residual_hard_neg_short = _hard_negative_residual(
            y_hard_negative_short, y_dead_negative_short, y_teaser_negative_short
        )
        ce_per = criterion.ce(logits, y)
        ce_sample_weight = _direction_ce_sample_weight(
            y_bad_path,
            y_dead_negative_long,
            y_teaser_negative_long,
            residual_hard_neg_long,
            y_dead_negative_short,
            y_teaser_negative_short,
            residual_hard_neg_short,
        ).to(device=ce_per.device, dtype=ce_per.dtype)
        ce_loss_raw = (ce_per * ce_sample_weight).mean()
        ce_loss = float(ENTRY_DIRECTION_CE_SCALE) * ce_loss_raw
        probs = torch.softmax(logits, dim=1)
        tail_direction_loss = torch.tensor(0.0, device=device)
        tail_direction_mask = torch.zeros_like(y, dtype=torch.bool)
        if float(ENTRY_TAIL_DIRECTION_CE_WEIGHT) > 0.0:
            tail_direction_mask = _tail_direction_mask(y, y_tradable, y_bad_path, y_path_quality)
            if tail_direction_mask.any():
                tail_direction_loss = float(ENTRY_TAIL_DIRECTION_CE_WEIGHT) * ce_per[tail_direction_mask].mean()

        cost_term = torch.tensor(0.0, device=device)
        balance_term = _direction_balance_term(probs, y, criterion)
        min_pred_rate_term = _direction_min_pred_rate_term(probs, y)
        global_prior_match_term = _direction_global_prior_match_term(probs, y)
        slice_min_pred_rate_term = _direction_slice_min_pred_rate_term(probs, y, ctx_cat)
        slice_recall_term = _direction_slice_recall_prob_term(probs, y, ctx_cat)
        slice_balanced_ce_term = _direction_slice_balanced_ce_term(logits, y, ctx_cat)
        slice_true_margin_term = _direction_slice_true_margin_term(logits, y, ctx_cat)
        slice_accuracy_edge_term = _direction_slice_accuracy_edge_term(logits, y, ctx_cat)
        slice_confusion_pair_term = _direction_slice_confusion_pair_term(logits, y, ctx_cat)
        slice_prior_match_term = _direction_slice_prior_match_term(probs, y, ctx_cat)
        direction_flat_margin_term = _direction_vs_flat_margin_term(logits, y)
        direction_utility_margin_term = _direction_utility_margin_term(
            logits,
            batch["y_long_path_utility_bps"],
            batch["y_short_path_utility_bps"],
        )
        direction_side_utility_conviction_term = _direction_side_utility_conviction_term(
            logits,
            y,
            batch["y_long_path_utility_bps"],
            batch["y_short_path_utility_bps"],
        )
        direction_utility_trade_conviction_term = _direction_utility_trade_conviction_term(
            logits,
            batch["y_long_path_utility_bps"],
            batch["y_short_path_utility_bps"],
            batch["y_long_bad_path"],
            batch["y_short_bad_path"],
        )
        direction_utility_triad_ce_term = _direction_utility_triad_ce_term(
            logits,
            batch["y_long_path_utility_bps"],
            batch["y_short_path_utility_bps"],
            batch["y_long_bad_path"],
            batch["y_short_bad_path"],
        )
        direction_flat_starvation_term = _direction_flat_starvation_term(logits, y, ctx_cat)
        if bool(getattr(criterion, "enabled", False)):
            cost = criterion.cost_matrix.to(dtype=logits.dtype)[y]
            expected_cost = (cost * probs).sum(dim=1)
            cost_term = float(getattr(criterion, "cost_scale", 1.0)) * expected_cost.mean()

        loss = (
            ce_loss
            + cost_term
            + balance_term
            + min_pred_rate_term
            + global_prior_match_term
            + slice_min_pred_rate_term
            + slice_recall_term
            + slice_balanced_ce_term
            + slice_true_margin_term
            + slice_accuracy_edge_term
            + slice_confusion_pair_term
            + slice_prior_match_term
            + direction_flat_margin_term
            + direction_utility_margin_term
            + direction_side_utility_conviction_term
            + direction_utility_trade_conviction_term
            + direction_utility_triad_ce_term
            + direction_flat_starvation_term
        )
        if float(ENTRY_TAIL_DIRECTION_CE_WEIGHT) > 0.0:
            loss = loss + tail_direction_loss
        if float(ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT) > 0.0 or float(ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT) > 0.0:
            loss = loss + specialist_gate_loss
        if float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT) > 0.0:
            loss = loss + bad_path_quality_rank_loss
        if float(ENTRY_PATH_QUALITY_RANK_WEIGHT) > 0.0:
            loss = loss + path_quality_rank_loss
        hier_loss, hier_stats = _hierarchical_entry_loss(
            out,
            batch,
            device,
            trade_pos_weight=hier_trade_pos_weight,
            side_bad_path_pos_weight=hier_bad_path_pos_weight,
        )
        if hier_loss.numel() == 1:
            loss = loss + hier_loss
        trendline_rail_loss, trendline_stats = _trendline_rail_aux_loss(out, batch, device)
        if trendline_rail_loss.numel() == 1:
            loss = loss + trendline_rail_loss
        hard_neg_prob_loss = torch.tensor(0.0, device=device)
        dead_neg_prob_loss = torch.tensor(0.0, device=device)
        teaser_neg_prob_loss = torch.tensor(0.0, device=device)
        dead_neg_mask = y_dead_negative_long.float() > 0.5
        teaser_neg_mask = y_teaser_negative_long.float() > 0.5
        hard_neg_mask = residual_hard_neg_long > 0.5
        if float(ENTRY_DEAD_LONG_PROB_PENALTY) > 0.0 and dead_neg_mask.any():
            dead_neg_prob_loss = float(ENTRY_DEAD_LONG_PROB_PENALTY) * probs[dead_neg_mask, 0].mean()
            loss = loss + dead_neg_prob_loss
        if float(ENTRY_TEASER_LONG_PROB_PENALTY) > 0.0 and teaser_neg_mask.any():
            teaser_neg_prob_loss = float(ENTRY_TEASER_LONG_PROB_PENALTY) * probs[teaser_neg_mask, 0].mean()
            loss = loss + teaser_neg_prob_loss
        bad_path_prob_penalty_loss = _selected_side_bad_path_probability_penalty(
            probs,
            y,
            y_bad_path,
            ENTRY_BAD_PATH_PROB_PENALTY,
        )
        loss = loss + bad_path_prob_penalty_loss
        if float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) > 0.0 and hard_neg_mask.any():
            hard_neg_prob_loss = float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) * probs[hard_neg_mask, 0].mean()
            loss = loss + hard_neg_prob_loss
        # SYMMETRIC SHORT prob-penalties (run_id v10_symmetric_negatives_20260603) — push down
        # probs[:,1] (SHORT) on short-negative samples, mirroring the long penalties on probs[:,0].
        # This is the direct counterweight to the LONG-suppression. OFF by default (cement).
        if ENTRY_SYMMETRIC_NEGATIVES:
            dead_neg_short_mask = y_dead_negative_short.float() > 0.5
            teaser_neg_short_mask = y_teaser_negative_short.float() > 0.5
            hard_neg_short_mask = residual_hard_neg_short > 0.5
            if float(ENTRY_DEAD_LONG_PROB_PENALTY) > 0.0 and dead_neg_short_mask.any():
                loss = loss + float(ENTRY_DEAD_LONG_PROB_PENALTY) * probs[dead_neg_short_mask, 1].mean()
            if float(ENTRY_TEASER_LONG_PROB_PENALTY) > 0.0 and teaser_neg_short_mask.any():
                loss = loss + float(ENTRY_TEASER_LONG_PROB_PENALTY) * probs[teaser_neg_short_mask, 1].mean()
            if float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) > 0.0 and hard_neg_short_mask.any():
                loss = loss + float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) * probs[hard_neg_short_mask, 1].mean()

        tradable_loss = torch.tensor(0.0, device=device)
        clean_edge_loss = torch.tensor(0.0, device=device)
        survival_loss = torch.tensor(0.0, device=device)
        clean_edge_rank_loss = torch.tensor(0.0, device=device)
        path_loss = torch.tensor(0.0, device=device)
        mfe_loss = torch.tensor(0.0, device=device)
        positive_mask = y_tradable.float() > 0.5
        selector_mask = _aux_selector_mask(y_selector_long_mask, y_selector_short_mask)
        clean_edge_target = _aux_clean_edge_target(y_clean_edge_long, y_clean_edge_bidir)
        survival_target = _aux_survival_target(y_survival_long, y_survival_bidir)
        if aux_path_weight > 0.0:
            if positive_mask.any():
                # Learn the exact signed path outcome on tradable rows; negative
                # paths must remain distinguishable from parked zeros.
                path_target = _signed_scaled_aux_regression_target(
                    y_path_quality,
                    positive_mask,
                    aux_path_scale_bps,
                )
                # V10 v3+ Target 2: if heteroscedastic head is active, use Gaussian NLL
                # so model learns uncertainty (high var on regime-conflict samples).
                # NLL = 0.5 * (log_var + (y - mu)^2 / exp(log_var))
                mu = path_pred.squeeze(1)[positive_mask]
                lv = path_log_var.squeeze(1)[positive_mask].clamp(min=-5.0, max=5.0)
                sq_err = (path_target.float() - mu) ** 2
                path_loss = 0.5 * (lv + sq_err / torch.exp(lv)).mean()
                path_loss = float(aux_path_weight) * path_loss
                loss = loss + path_loss
                path_loss_sum += float(path_loss.item()) * y.shape[0]
        if aux_mfe_weight > 0.0:
            if positive_mask.any():
                mfe_target = _signed_scaled_aux_regression_target(
                    y_mfe_first,
                    positive_mask,
                    aux_mfe_scale_bps,
                )
                mfe_loss = nn.functional.smooth_l1_loss(
                    mfe_pred.squeeze(1)[positive_mask], mfe_target.float()
                )
                mfe_loss = float(aux_mfe_weight) * mfe_loss
                loss = loss + mfe_loss
                mfe_loss_sum += float(mfe_loss.item()) * y.shape[0]
        if aux_tradable_weight > 0.0:
            if selector_mask.any():
                tradable_loss = nn.functional.binary_cross_entropy_with_logits(
                    tradable_logit.squeeze(1)[selector_mask],
                    y_tradable.float()[selector_mask],
                    pos_weight=torch.tensor(float(tradable_pos_weight), device=device, dtype=tradable_logit.dtype),
                )
                tradable_loss = float(aux_tradable_weight) * tradable_loss
                loss = loss + tradable_loss
                tradable_loss_sum += float(tradable_loss.item()) * y.shape[0]
        if float(ENTRY_AUX_BAD_PATH_WEIGHT) > 0.0:
            if selector_mask.any():
                bad_path_bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    bad_path_logit.squeeze(1)[selector_mask],
                    y_bad_path.float()[selector_mask],
                    pos_weight=torch.tensor(float(bad_path_pos_weight), device=device, dtype=bad_path_logit.dtype),
                )
                bad_path_bce_loss = float(ENTRY_AUX_BAD_PATH_WEIGHT) * bad_path_bce_loss
                loss = loss + bad_path_bce_loss
                aux_bad_path_bce_loss_sum += float(bad_path_bce_loss.item()) * y.shape[0]
        if float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) > 0.0:
            if selector_mask.any():
                clean_edge_loss = nn.functional.binary_cross_entropy_with_logits(
                    clean_edge_logit.squeeze(1)[selector_mask],
                    clean_edge_target[selector_mask],
                    pos_weight=torch.tensor(float(clean_edge_pos_weight), device=device, dtype=clean_edge_logit.dtype),
                )
                clean_edge_loss = float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) * clean_edge_loss
                loss = loss + clean_edge_loss
                clean_edge_loss_sum += float(clean_edge_loss.item()) * y.shape[0]
        if float(ENTRY_AUX_SURVIVAL_WEIGHT) > 0.0:
            if selector_mask.any():
                survival_loss = nn.functional.binary_cross_entropy_with_logits(
                    survival_logit.squeeze(1)[selector_mask],
                    survival_target[selector_mask],
                    pos_weight=torch.tensor(float(survival_pos_weight), device=device, dtype=survival_logit.dtype),
                )
                survival_loss = float(ENTRY_AUX_SURVIVAL_WEIGHT) * survival_loss
                loss = loss + survival_loss
                survival_loss_sum += float(survival_loss.item()) * y.shape[0]
        if float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT) > 0.0:
            clean_edge_prob = torch.sigmoid(clean_edge_logit.squeeze(1))
            clean_pos, ranked_neg = _clean_edge_rank_masks(
                y_clean_edge_long,
                y_clean_edge_bidir,
                y_dead_negative_long,
                y_teaser_negative_long,
                residual_hard_neg_long,
                y_dead_negative_short,
                y_teaser_negative_short,
                residual_hard_neg_short,
            )
            if clean_pos.any() and ranked_neg.any():
                pos_long = clean_edge_prob[clean_pos].mean()
                neg_long = clean_edge_prob[ranked_neg].mean()
                clean_edge_rank_loss = float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT) * torch.relu(
                    torch.tensor(float(ENTRY_CLEAN_EDGE_RANKING_MARGIN), device=device, dtype=probs.dtype)
                    - (pos_long - neg_long)
                )
                loss = loss + clean_edge_rank_loss
                clean_edge_rank_loss_sum += float(clean_edge_rank_loss.item()) * y.shape[0]
        tf_agreement_logit = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="tf_agreement_logit",
            target_names=("y_tf_agreement_score",),
        )
        y_tf_agreement = batch["y_tf_agreement_score"].to(device, non_blocking=non_blocking)
        tf_pred = torch.sigmoid(tf_agreement_logit).squeeze(-1)
        tf_agreement_loss = torch.nn.functional.mse_loss(tf_pred, y_tf_agreement)
        loss = loss + (
            _MODEL_NATIVE_FIXED_POSITIVE_LOSS_WEIGHTS["tf_agreement"]
            * tf_agreement_loss
        )

        position_size_logit = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="position_size_logit",
            target_names=("y_position_size_target",),
        )
        y_pos_size = batch["y_position_size_target"].to(device, non_blocking=non_blocking)
        pos_pred = torch.sigmoid(position_size_logit).squeeze(-1)
        position_size_loss = torch.nn.functional.mse_loss(pos_pred, y_pos_size)
        loss = loss + (
            _MODEL_NATIVE_FIXED_POSITIVE_LOSS_WEIGHTS["position_size"]
            * position_size_loss
        )

        # Forceful MTF→direction aux CE (2026-06-06): force the multi-TF repr to
        # predict direction (LONG/SHORT/FLAT). Mirrors the active direction
        # repair recipe: class weights, bad-path/side sample weights and
        # prediction-balance, NOT selector-masked.
        mtf_dir_logits = _require_active_aux_head_prediction(
            out,
            batch,
            output_name="mtf_dir_logits",
            target_names=_DIRECTION_BATCH_TARGET_NAMES,
        )
        mtf_dir_loss = _direction_aux_ce_loss(
            mtf_dir_logits,
            y,
            criterion,
            ce_sample_weight,
        )
        loss = loss + float(ENTRY_MTF_DIR_AUX_WEIGHT) * mtf_dir_loss

        loss = loss + _MODEL_NATIVE_FIXED_POSITIVE_LOSS_WEIGHTS[
            "dip_forecast_timing_tail_vol_composite"
        ] * dip_forecast_loss(
            out,
            batch,
            device,
            bps_scale=aux_mfe_scale_bps,
        )
        loss = loss + offline_rl_aux_loss(out, batch, device)

        preds = torch.argmax(probs, dim=1)
        short_mask = y == 1
        if short_mask.any():
            short_total += int(short_mask.sum().item())
            short_pred_long += int(((preds == 0) & short_mask).sum().item())
        # Grad accumulation: scale loss down by accum_steps so .backward() sums to
        # the same magnitude as a single big-batch step. Only step + zero every Nth batch.
        if _accum_steps > 1:
            (loss / float(_accum_steps)).backward()
        else:
            loss.backward()
        _accum_count += 1
        if _accum_count >= _accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            _accum_count = 0

        bs = y.shape[0]
        total += float(loss) * bs
        total_ce += float(ce_loss) * bs
        total_cost += float(cost_term.detach().cpu().item()) * bs
        total_balance += float(balance_term.detach().cpu().item()) * bs
        total_direction_min_pred += float(min_pred_rate_term.detach().cpu().item()) * bs
        total_direction_global_prior_match += float(global_prior_match_term.detach().cpu().item()) * bs
        total_direction_slice_min_pred += float(slice_min_pred_rate_term.detach().cpu().item()) * bs
        total_direction_slice_recall += float(slice_recall_term.detach().cpu().item()) * bs
        total_direction_slice_balanced_ce += float(slice_balanced_ce_term.detach().cpu().item()) * bs
        total_direction_slice_true_margin += float(slice_true_margin_term.detach().cpu().item()) * bs
        total_direction_slice_accuracy_edge += float(slice_accuracy_edge_term.detach().cpu().item()) * bs
        total_direction_slice_confusion_pair += float(slice_confusion_pair_term.detach().cpu().item()) * bs
        total_direction_slice_prior_match += float(slice_prior_match_term.detach().cpu().item()) * bs
        total_direction_flat_margin += float(direction_flat_margin_term.detach().cpu().item()) * bs
        total_direction_utility_margin += float(direction_utility_margin_term.detach().cpu().item()) * bs
        total_direction_side_utility_conviction += float(
            direction_side_utility_conviction_term.detach().cpu().item()
        ) * bs
        total_direction_utility_trade_conviction += float(
            direction_utility_trade_conviction_term.detach().cpu().item()
        ) * bs
        total_direction_utility_triad_ce += float(direction_utility_triad_ce_term.detach().cpu().item()) * bs
        total_direction_flat_starvation += float(direction_flat_starvation_term.detach().cpu().item()) * bs
        total_tail_direction += float(tail_direction_loss.detach().cpu().item()) * bs
        tail_direction_rows += int(tail_direction_mask.sum().detach().cpu().item())
        specialist_gate_loss_sum += float(specialist_gate_loss.detach().cpu().item()) * bs
        bad_path_quality_rank_loss_sum += float(bad_path_quality_rank_loss.detach().cpu().item()) * bs
        path_quality_rank_loss_sum += float(path_quality_rank_loss.detach().cpu().item()) * bs
        hard_neg_prob_loss_sum += float(hard_neg_prob_loss) * bs
        bad_path_prob_penalty_loss_sum += float(
            bad_path_prob_penalty_loss.detach().cpu().item()
        ) * bs
        hier_trade_loss_sum += float(hier_stats.get("hier_trade_loss", 0.0)) * bs
        hier_trade_global_prior_loss_sum += float(hier_stats.get("hier_trade_global_prior_loss", 0.0)) * bs
        hier_slice_trade_prior_loss_sum += float(hier_stats.get("hier_slice_trade_prior_loss", 0.0)) * bs
        hier_slice_trade_accuracy_edge_loss_sum += float(
            hier_stats.get("hier_slice_trade_accuracy_edge_loss", 0.0)
        ) * bs
        hier_flat_logit_margin_loss_sum += float(hier_stats.get("hier_flat_logit_margin_loss", 0.0)) * bs
        hier_slice_flat_logit_margin_loss_sum += (
            float(hier_stats.get("hier_slice_flat_logit_margin_loss", 0.0)) * bs
        )
        hier_side_loss_sum += float(hier_stats.get("hier_side_loss", 0.0)) * bs
        hier_slice_side_ce_loss_sum += float(hier_stats.get("hier_slice_side_ce_loss", 0.0)) * bs
        hier_slice_side_margin_loss_sum += float(hier_stats.get("hier_slice_side_margin_loss", 0.0)) * bs
        hier_slice_side_accuracy_edge_loss_sum += float(
            hier_stats.get("hier_slice_side_accuracy_edge_loss", 0.0)
        ) * bs
        hier_side_global_prior_loss_sum += float(hier_stats.get("hier_side_global_prior_loss", 0.0)) * bs
        hier_slice_side_prior_loss_sum += float(hier_stats.get("hier_slice_side_prior_loss", 0.0)) * bs
        hier_utility_loss_sum += float(hier_stats.get("hier_utility_loss", 0.0)) * bs
        hier_side_bad_path_loss_sum += float(hier_stats.get("hier_bad_path_loss", 0.0)) * bs
        hier_side_mae_loss_sum += float(hier_stats.get("hier_mae_loss", 0.0)) * bs
        hier_side_validity_loss_sum += float(hier_stats.get("hier_side_validity_loss", 0.0)) * bs
        hier_long_valid_target_rate_sum += float(hier_stats.get("hier_long_valid_target_rate", 0.0)) * bs
        hier_short_valid_target_rate_sum += float(hier_stats.get("hier_short_valid_target_rate", 0.0)) * bs
        hier_long_valid_prob_sum += float(hier_stats.get("hier_long_valid_prob_mean", 0.0)) * bs
        hier_short_valid_prob_sum += float(hier_stats.get("hier_short_valid_prob_mean", 0.0)) * bs
        hier_long_bad_target_rate_sum += float(hier_stats.get("hier_long_bad_target_rate", 0.0)) * bs
        hier_short_bad_target_rate_sum += float(hier_stats.get("hier_short_bad_target_rate", 0.0)) * bs
        hier_countertrend_long_trap_rate_sum += float(hier_stats.get("hier_countertrend_long_trap_rate", 0.0)) * bs
        hier_countertrend_short_trap_rate_sum += float(hier_stats.get("hier_countertrend_short_trap_rate", 0.0)) * bs
        _side_rows = int(hier_stats.get("hier_side_rows", 0.0))
        if _side_rows > 0:
            hier_side_rows_sum += _side_rows
            hier_side_correct_sum += float(hier_stats.get("hier_side_acc", 0.0)) * _side_rows
        trendline_rail_loss_sum += float(trendline_stats.get("trendline_rail_loss", 0.0)) * bs
        trendline_rail_rows_sum += int(trendline_stats.get("trendline_rail_rows", 0.0))
        trendline_rising_rows_sum += int(trendline_stats.get("trendline_rising_rows", 0.0))
        trendline_falling_rows_sum += int(trendline_stats.get("trendline_falling_rows", 0.0))
        n += bs

    if _accum_count:
        _step_partial_gradient_accumulation(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            configured_steps=_accum_steps,
            observed_steps=_accum_count,
        )

    stats = {
        "ce_loss_mean": (total_ce / max(1, n)),
        "cost_loss_mean": (total_cost / max(1, n)),
        "balance_loss_mean": (total_balance / max(1, n)),
        "direction_min_pred_rate_loss_mean": (total_direction_min_pred / max(1, n)),
        "direction_global_prior_match_loss_mean": (total_direction_global_prior_match / max(1, n)),
        "direction_slice_min_pred_rate_loss_mean": (total_direction_slice_min_pred / max(1, n)),
        "direction_slice_recall_loss_mean": (total_direction_slice_recall / max(1, n)),
        "direction_slice_balanced_ce_loss_mean": (total_direction_slice_balanced_ce / max(1, n)),
        "direction_slice_true_margin_loss_mean": (total_direction_slice_true_margin / max(1, n)),
        "direction_slice_accuracy_edge_loss_mean": (total_direction_slice_accuracy_edge / max(1, n)),
        "direction_slice_confusion_pair_loss_mean": (total_direction_slice_confusion_pair / max(1, n)),
        "direction_slice_prior_match_loss_mean": (total_direction_slice_prior_match / max(1, n)),
        "direction_flat_margin_loss_mean": (total_direction_flat_margin / max(1, n)),
        "direction_utility_margin_loss_mean": (total_direction_utility_margin / max(1, n)),
        "direction_side_utility_conviction_loss_mean": (
            total_direction_side_utility_conviction / max(1, n)
        ),
        "direction_utility_trade_conviction_loss_mean": (
            total_direction_utility_trade_conviction / max(1, n)
        ),
        "direction_utility_triad_ce_loss_mean": (total_direction_utility_triad_ce / max(1, n)),
        "direction_flat_starvation_loss_mean": (total_direction_flat_starvation / max(1, n)),
        "tail_direction_loss_mean": (total_tail_direction / max(1, n)),
        "tail_direction_rows": int(tail_direction_rows),
        "specialist_gate_loss_mean": (specialist_gate_loss_sum / max(1, n)),
        "bad_path_quality_rank_loss_mean": (bad_path_quality_rank_loss_sum / max(1, n)),
        "path_quality_rank_loss_mean": (path_quality_rank_loss_sum / max(1, n)),
        "hard_neg_prob_loss_mean": (hard_neg_prob_loss_sum / max(1, n)),
        "aux_bad_path_bce_loss_mean": (aux_bad_path_bce_loss_sum / max(1, n)),
        "bad_path_prob_penalty_loss_mean": (
            bad_path_prob_penalty_loss_sum / max(1, n)
        ),
        "short_pred_long_rate": (short_pred_long / short_total if short_total > 0 else 0.0),
        "aux_path_loss_mean": (path_loss_sum / max(1, n)),
        "aux_mfe_loss_mean": (mfe_loss_sum / max(1, n)),
        "aux_tradable_loss_mean": (tradable_loss_sum / max(1, n)),
        "aux_clean_edge_loss_mean": (clean_edge_loss_sum / max(1, n)),
        "aux_survival_loss_mean": (survival_loss_sum / max(1, n)),
        "clean_edge_rank_loss_mean": (clean_edge_rank_loss_sum / max(1, n)),
        "hier_trade_loss_mean": (hier_trade_loss_sum / max(1, n)),
        "hier_trade_global_prior_loss_mean": (hier_trade_global_prior_loss_sum / max(1, n)),
        "hier_slice_trade_prior_loss_mean": (hier_slice_trade_prior_loss_sum / max(1, n)),
        "hier_slice_trade_accuracy_edge_loss_mean": (
            hier_slice_trade_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_flat_logit_margin_loss_mean": (hier_flat_logit_margin_loss_sum / max(1, n)),
        "hier_slice_flat_logit_margin_loss_mean": (
            hier_slice_flat_logit_margin_loss_sum / max(1, n)
        ),
        "hier_side_loss_mean": (hier_side_loss_sum / max(1, n)),
        "hier_slice_side_ce_loss_mean": (hier_slice_side_ce_loss_sum / max(1, n)),
        "hier_slice_side_margin_loss_mean": (hier_slice_side_margin_loss_sum / max(1, n)),
        "hier_slice_side_accuracy_edge_loss_mean": (
            hier_slice_side_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_side_global_prior_loss_mean": (hier_side_global_prior_loss_sum / max(1, n)),
        "hier_slice_side_prior_loss_mean": (hier_slice_side_prior_loss_sum / max(1, n)),
        "hier_utility_loss_mean": (hier_utility_loss_sum / max(1, n)),
        "hier_bad_path_loss_mean": (hier_side_bad_path_loss_sum / max(1, n)),
        "hier_mae_loss_mean": (hier_side_mae_loss_sum / max(1, n)),
        "hier_side_validity_loss_mean": (hier_side_validity_loss_sum / max(1, n)),
        "hier_long_valid_target_rate": (hier_long_valid_target_rate_sum / max(1, n)),
        "hier_short_valid_target_rate": (hier_short_valid_target_rate_sum / max(1, n)),
        "hier_long_valid_prob_mean": (hier_long_valid_prob_sum / max(1, n)),
        "hier_short_valid_prob_mean": (hier_short_valid_prob_sum / max(1, n)),
        "hier_long_bad_target_rate": (hier_long_bad_target_rate_sum / max(1, n)),
        "hier_short_bad_target_rate": (hier_short_bad_target_rate_sum / max(1, n)),
        "hier_countertrend_long_trap_rate": (hier_countertrend_long_trap_rate_sum / max(1, n)),
        "hier_countertrend_short_trap_rate": (hier_countertrend_short_trap_rate_sum / max(1, n)),
        "hier_side_rows": int(hier_side_rows_sum),
        "hier_side_acc": (hier_side_correct_sum / hier_side_rows_sum if hier_side_rows_sum > 0 else 0.0),
        "trendline_rail_loss_mean": (trendline_rail_loss_sum / max(1, n)),
        "trendline_rail_rows": int(trendline_rail_rows_sum),
        "trendline_rising_rows": int(trendline_rising_rows_sum),
        "trendline_falling_rows": int(trendline_falling_rows_sum),
    }
    stats.update(_finalize_cooperation_gate_epoch(cooperation_gate_epoch))
    return total / max(1, n), stats


def _aux_head_diagnostics(
    head_preds: "dict[str, list]",
    binary_labels: "dict[str, list]",
    realized: "dict[str, list]",
    row_masks: "dict[str, list]",
) -> "tuple[dict, list]":
    """Compute mandatory auxiliary-head health evidence.

    Computes, on the accumulated validation predictions:
      (a) cross-head Spearman between aux-head predictions -> catches the rho~0.99
          head-collapse (clean_edge/survival/tradable becoming redundant);
      (b) tradable AUC on all rows, but path-head AUC on TRADE and separately
          LONG/SHORT rows -> prevents FLAT recognition from masquerading as
          selected-side path discrimination;
      (c) path-head AUC lift over the generic tradable score on each conditional
          scope -> proves that a dedicated path head contributes incremental
          evidence instead of duplicating trade/no-trade;
      (c) Spearman(head_pred, realized outcome) -> catches MIS-TARGETING such as the
          documented bad_path head predicting volatility instead of loss (bad_path prob
          should correlate NEGATIVELY with realized path_quality_bps).

    Missing masks, insufficient per-side support, non-finite metrics,
    non-discriminative/incrementally redundant heads, anti-targeting, and
    collapsed heads are checkpoint-blocking failures.
    """
    metrics: "dict[str, float]" = {}
    warns: "list[str]" = []
    try:
        from scipy.stats import spearmanr  # local import: fail-soft if scipy absent
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            f"[ENTRY_AUX_HEAD_HEALTH_DEPENDENCY_MISSING] {exc}"
        ) from exc

    def _cat(d, k):
        vs = d.get(k) or []
        if not vs:
            raise RuntimeError(f"[ENTRY_AUX_HEAD_HEALTH_EVIDENCE_MISSING] {k}")
        try:
            arr = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in vs])
        except Exception as exc:
            raise RuntimeError(f"[ENTRY_AUX_HEAD_HEALTH_EVIDENCE_INVALID] {k}: {exc}") from exc
        if arr.size < 16 or not np.isfinite(arr).all():
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_EVIDENCE_INVALID] {k} rows={arr.size} finite={np.isfinite(arr).all()}"
            )
        return arr

    def _cat_mask(k: str) -> np.ndarray:
        arr = _cat(row_masks, k)
        if not np.logical_or(arr == 0.0, arr == 1.0).all():
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_MASK_NOT_BINARY] {k}"
            )
        return arr > 0.5

    required_preds = (
        "tradable",
        "bad_path",
        "clean_edge",
        "survival",
        "path_quality",
        "mfe_first_n",
    )
    pred_arrays = {k: _cat(head_preds, k) for k in required_preds}
    masks = {name: _cat_mask(name) for name in ("trade", "long", "short")}
    row_count = int(next(iter(pred_arrays.values())).size)
    if any(int(value.size) != row_count for value in pred_arrays.values()):
        raise RuntimeError("[ENTRY_AUX_HEAD_HEALTH_PREDICTION_ROW_MISMATCH]")
    if any(int(value.size) != row_count for value in masks.values()):
        raise RuntimeError("[ENTRY_AUX_HEAD_HEALTH_MASK_ROW_MISMATCH]")
    if np.logical_and(masks["long"], masks["short"]).any() or not np.array_equal(
        masks["trade"],
        np.logical_or(masks["long"], masks["short"]),
    ):
        raise RuntimeError("[ENTRY_AUX_HEAD_HEALTH_SIDE_MASK_PARTITION_INVALID]")
    metrics["conditional_rows__all"] = row_count
    for name, mask in masks.items():
        rows = int(mask.sum())
        metrics[f"conditional_rows__{name}"] = rows
        if rows < 16:
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_ROWS_INSUFFICIENT] scope={name} rows={rows}"
            )

    # (a) Cross-head collapse must be measured inside the actual edge rows.
    # Flat-vs-trade separation is not evidence that two path heads differ.
    names = sorted(pred_arrays.keys())
    max_abs_rho = 0.0
    max_pair = ""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = pred_arrays[names[i]][masks["trade"]]
            b = pred_arrays[names[j]][masks["trade"]]
            if a.size != b.size:
                raise RuntimeError(
                    f"[ENTRY_AUX_HEAD_HEALTH_ROW_MISMATCH] {names[i]}={a.size} {names[j]}={b.size}"
                )
            try:
                rho = spearmanr(a, b).correlation
            except Exception as exc:
                raise RuntimeError(
                    f"[ENTRY_AUX_HEAD_HEALTH_SPEARMAN_FAILED] {names[i]}~{names[j]}: {exc}"
                ) from exc
            if rho is None or not np.isfinite(rho):
                raise RuntimeError(
                    f"[ENTRY_AUX_HEAD_HEALTH_SPEARMAN_NONFINITE] {names[i]}~{names[j]}"
                )
            metrics[f"xhead_rho__{names[i]}__{names[j]}"] = float(rho)
            if abs(rho) > abs(max_abs_rho):
                max_abs_rho = float(rho)
                max_pair = f"{names[i]}~{names[j]}"
            if abs(rho) >= 0.95:
                warns.append(
                    f"[V10-AUX-02] cross-head COLLAPSE: {names[i]}~{names[j]} "
                    f"Spearman={rho:+.3f} (>=0.95 => heads are redundant)"
                )
    if max_pair:
        metrics["xhead_max_abs_rho"] = float(max_abs_rho)
        if abs(max_abs_rho) >= 0.95:
            warns.append(f"[V10-AUX-02] highest cross-head |rho|={max_abs_rho:+.3f} @ {max_pair}")

    def _auc(
        *,
        head: str,
        prediction: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray,
        scope: str,
    ) -> float:
        pred = prediction[mask]
        lbl = label[mask]
        if pred.size < 16:
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_ROWS_INSUFFICIENT] "
                f"head={head} scope={scope} rows={pred.size}"
            )
        unique = np.unique(lbl)
        if unique.size < 2:
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_LABEL_CONSTANT] "
                f"head={head} scope={scope}"
            )
        if not np.logical_or(lbl == 0.0, lbl == 1.0).all():
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_LABEL_NOT_BINARY] "
                f"head={head} scope={scope}"
            )
        try:
            value = float(roc_auc_score(lbl.astype(int), pred))
        except Exception as exc:
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_AUC_FAILED] "
                f"head={head} scope={scope}: {exc}"
            ) from exc
        if not np.isfinite(value):
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_AUC_NONFINITE] "
                f"head={head} scope={scope}"
            )
        return value

    # (b) The generic tradable head owns the all-row task.
    tradable_label = _cat(binary_labels, "tradable")
    if tradable_label.size != row_count:
        raise RuntimeError(
            "[ENTRY_AUX_HEAD_HEALTH_ROW_MISMATCH] "
            f"pred_tradable={row_count} label_tradable={tradable_label.size}"
        )
    tradable_auc = _auc(
        head="tradable",
        prediction=pred_arrays["tradable"],
        label=tradable_label,
        mask=np.ones(row_count, dtype=bool),
        scope="all",
    )
    metrics["auc__tradable__all"] = tradable_auc
    metrics["auc__tradable"] = tradable_auc
    if tradable_auc < 0.52:
        warns.append(
            f"[V10-AUX-02] tradable AUC={tradable_auc:.3f} "
            "(~chance => head not discriminating)"
        )

    # Dedicated path heads must discriminate within TRADE, LONG and SHORT.
    for k in ("bad_path", "clean_edge", "survival"):
        lbl = _cat(binary_labels, k)
        if lbl.size != row_count:
            raise RuntimeError(
                f"[ENTRY_AUX_HEAD_HEALTH_ROW_MISMATCH] "
                f"pred_{k}={row_count} label_{k}={lbl.size}"
            )
        for scope in ("trade", "long", "short"):
            mask = masks[scope]
            auc = _auc(
                head=k,
                prediction=pred_arrays[k],
                label=lbl,
                mask=mask,
                scope=scope,
            )
            metrics[f"auc__{k}__{scope}"] = auc
            if scope == "trade":
                metrics[f"auc__{k}"] = auc
            if auc < 0.52:
                warns.append(
                    f"[V10-AUX-02] {k} conditional AUC={auc:.3f} "
                    f"scope={scope} (~chance => path head not discriminating)"
                )

            baseline_prediction = (
                1.0 - pred_arrays["tradable"]
                if k == "bad_path"
                else pred_arrays["tradable"]
            )
            baseline_auc = _auc(
                head=f"{k}_tradable_baseline",
                prediction=baseline_prediction,
                label=lbl,
                mask=mask,
                scope=scope,
            )
            lift = float(auc - baseline_auc)
            metrics[f"baseline_auc__{k}__{scope}"] = baseline_auc
            metrics[f"incremental_auc_lift__{k}__{scope}"] = lift
            if lift < 0.01:
                warns.append(
                    f"[V10-AUX-02] {k} no incremental conditional edge: "
                    f"scope={scope} head_auc={auc:.3f} "
                    f"tradable_baseline_auc={baseline_auc:.3f} "
                    f"lift={lift:+.3f} required>=+0.010"
                )

    # (c) Realized-target alignment is also conditional on edge/side.
    for rk in realized:
        rv = _cat(realized, rk)
        for hk, pred in pred_arrays.items():
            if pred.size != rv.size:
                raise RuntimeError(
                    f"[ENTRY_AUX_HEAD_HEALTH_ROW_MISMATCH] pred_{hk}={pred.size} realized_{rk}={rv.size}"
                )
            for scope in ("trade", "long", "short"):
                mask = masks[scope]
                if int(mask.sum()) < 16:
                    raise RuntimeError(
                        f"[ENTRY_AUX_HEAD_HEALTH_ROWS_INSUFFICIENT] "
                        f"{hk}~{rk} scope={scope}"
                    )
                try:
                    rho = spearmanr(pred[mask], rv[mask]).correlation
                except Exception as exc:
                    raise RuntimeError(
                        f"[ENTRY_AUX_HEAD_HEALTH_SPEARMAN_FAILED] "
                        f"{hk}~{rk} scope={scope}: {exc}"
                    ) from exc
                if rho is None or not np.isfinite(rho):
                    raise RuntimeError(
                        f"[ENTRY_AUX_HEAD_HEALTH_SPEARMAN_NONFINITE] "
                        f"{hk}~{rk} scope={scope}"
                    )
                metrics[f"realized_rho__{hk}__{rk}__{scope}"] = float(rho)
                if scope == "trade":
                    metrics[f"realized_rho__{hk}__{rk}"] = float(rho)
                if (
                    hk == "bad_path"
                    and rk == "path_quality_bps"
                    and rho > -0.02
                ):
                    warns.append(
                        "[V10-AUX-02] bad_path ANTI-TARGETED: "
                        f"scope={scope} Spearman(bad_path,path_quality_bps)="
                        f"{rho:+.3f} expected<=-0.020"
                    )
                if (
                    (hk, rk)
                    in {
                        ("path_quality", "path_quality_bps"),
                        ("mfe_first_n", "mfe_first_n_bps"),
                    }
                    and rho < 0.02
                ):
                    warns.append(
                        f"[V10-AUX-02] {hk} MIS-TARGETED: scope={scope} "
                        f"Spearman({hk},{rk})={rho:+.3f} expected>=+0.020"
                    )
    return metrics, warns


_ACTIVE_HEAD_FUSION_INPUTS: Dict[str, Tuple[str, ...]] = {
    "direction": ("model_native_logits",),
    "tradable": ("tradable_logit",),
    "path_quality": ("path_quality_raw",),
    "mfe_first_n": ("mfe_first_n",),
    "bad_path": ("bad_path_logit_raw",),
    "clean_edge": ("clean_edge_logit",),
    "survival": ("survival_logit",),
    "tf_agreement": ("tf_agreement_logit",),
    "path_quality_log_var": ("path_quality_log_var",),
    "position_size": ("position_size_logit",),
    "dip": ("dip_pred",),
    "forecast": ("forecast_pred",),
    "timing": ("timing_pred",),
    "tail_risk": ("tail_risk_pred",),
    "vol_forecast": ("vol_forecast_pred",),
    "mtf_direction": ("mtf_dir_logits",),
    "trade_side_hierarchy": (
        "trade_logit",
        "side_logits",
        "side_utility",
        "side_bad_path_logit",
        "side_mae",
    ),
    "trendline_rail": ("trendline_rail_logits",),
    "side_validity": ("side_validity_logit",),
    "offline_rl_action_value": ("action_value", "action_advantage"),
    "offline_rl_expectile_value": ("expectile_value", "action_advantage"),
    "model_native_evidence_fusion": tuple(
        name for name, _width in EXACT_EVIDENCE_FUSION_OUTPUTS
    ),
}
_ACTIVE_HEAD_TARGET_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "direction": ("model_native_logits",),
    "tradable": ("tradable_logit",),
    "path_quality": ("path_quality_raw",),
    "mfe_first_n": ("mfe_first_n",),
    "bad_path": ("bad_path_logit_raw",),
    "clean_edge": ("clean_edge_logit",),
    "survival": ("survival_logit",),
    "tf_agreement": ("tf_agreement_logit",),
    "path_quality_log_var": ("path_quality_log_var",),
    "position_size": ("position_size_logit",),
    "dip": ("dip_pred",),
    "forecast": ("forecast_pred",),
    "timing": ("timing_pred",),
    "tail_risk": ("tail_risk_pred",),
    "vol_forecast": ("vol_forecast_pred",),
    "mtf_direction": ("mtf_dir_logits",),
    "trade_side_hierarchy": (
        "trade_logit",
        "side_logits",
        "side_utility",
        "side_bad_path_logit",
        "side_mae",
    ),
    "trendline_rail": ("trendline_rail_logits",),
    "side_validity": ("side_validity_logit",),
    "offline_rl_action_value": ("action_value", "action_advantage"),
    "offline_rl_expectile_value": ("expectile_value",),
    "model_native_evidence_fusion": ("raw_direction_logits",),
}
_ACTIVE_HEAD_COMPONENT_WIDTHS = {
    **{name: int(width) for name, width in EXACT_EVIDENCE_FUSION_OUTPUTS},
    "raw_direction_logits": 3,
}
# Advantage is a derived Q-V evidence surface rather than a separately
# supervised target. One action can truthfully remain the best for an entire
# horizon, making that action's target advantage identically zero. Require
# liveness somewhere in the derived target vector while still requiring every
# emitted advantage column and its fusion influence to be alive.
_ACTIVE_HEAD_DERIVED_TARGET_COMPONENTS = frozenset({"action_advantage"})
# FLAT's counterfactual reward is exactly zero by contract at every horizon.
# Those three Q columns are supervised structural constants, not dead evidence;
# LONG/SHORT Q and the head-level class-centred fusion intervention must remain
# alive.
_ACTIVE_HEAD_STRUCTURAL_CONSTANT_COLUMNS = {
    "action_value": frozenset(
        range(
            (OFFLINE_RL_ACTION_COUNT - 1) * OFFLINE_RL_HORIZON_COUNT,
            OFFLINE_RL_ACTION_COUNT * OFFLINE_RL_HORIZON_COUNT,
        )
    )
}
_ACTIVE_HEAD_DIAGNOSTIC_MIN_ROWS = 16
_ACTIVE_HEAD_DIAGNOSTIC_LIVENESS_EPS = 1e-8
_ACTIVE_HEAD_DIAGNOSTIC_INFLUENCE_EPS = 1e-8


def _active_head_contract_failures() -> List[str]:
    expected = tuple(MODEL_NATIVE_ACTIVE_HEADS)
    observed = tuple(_ACTIVE_HEAD_FUSION_INPUTS)
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
    fusion_inputs = {name for name, _width in EXACT_EVIDENCE_FUSION_OUTPUTS}
    for head_name, output_names in _ACTIVE_HEAD_FUSION_INPUTS.items():
        missing = sorted(set(output_names) - fusion_inputs)
        if missing:
            failures.append(
                "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_FUSION_INPUT_MISSING] "
                f"head={head_name} outputs={missing}"
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
    *,
    path_scale_bps: float,
    mfe_scale_bps: float,
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

    y = batch.get("y")
    if not isinstance(y, torch.Tensor):
        raise RuntimeError("[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_TARGET_MISSING] target=y")
    y = y.to(device=device, non_blocking=device.type == "cuda").long().reshape(-1)
    if bool(((y < 0) | (y > 2)).any()):
        raise RuntimeError("[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_DIRECTION_TARGET_INVALID]")
    batch_size = int(y.shape[0])
    all_rows = torch.ones(batch_size, dtype=torch.bool, device=device)
    direction_target = nn.functional.one_hot(y, num_classes=3).float()

    y_tradable = _active_head_batch_target(batch, "y_tradable", device).reshape(-1)
    positive_rows = y_tradable > 0.5
    selector_rows = _aux_selector_mask(
        _active_head_batch_target(batch, "y_selector_long_mask", device).reshape(-1),
        _active_head_batch_target(batch, "y_selector_short_mask", device).reshape(-1),
    )
    clean_target = _aux_clean_edge_target(
        _active_head_batch_target(batch, "y_clean_edge_long", device).reshape(-1),
        _active_head_batch_target(batch, "y_clean_edge_bidir", device).reshape(-1),
    ).unsqueeze(1)
    survival_target = _aux_survival_target(
        _active_head_batch_target(batch, "y_survival_long", device).reshape(-1),
        _active_head_batch_target(batch, "y_survival_bidir", device).reshape(-1),
    ).unsqueeze(1)

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
    forward_bps_scale = float(mfe_scale_bps)
    if not np.isfinite(forward_bps_scale) or forward_bps_scale <= 0.0:
        raise RuntimeError(
            "[ENTRY_FORWARD_AUX_BPS_SCALE_INVALID] "
            f"observed={forward_bps_scale!r}"
        )
    dip_target = torch.stack(dip_targets, dim=1) / forward_bps_scale
    forecast_target = torch.stack(
        [
            _active_head_batch_target(batch, f"y_forecast_ret_K{horizon}", device)
            for horizon in FORECAST_HORIZONS
        ],
        dim=1,
    ) / forward_bps_scale
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
    ) / forward_bps_scale
    vol_target = torch.stack(
        [
            _active_head_batch_target(batch, f"y_vol_fwd_K{horizon}", device)
            for horizon in VOL_FORECAST_HORIZONS
        ],
        dim=1,
    ) / forward_bps_scale

    y_trade = _active_head_batch_target(batch, "y_trade", device).reshape(-1, 1)
    y_side = _active_head_batch_target(batch, "y_side", device).long().reshape(-1)
    y_side_mask = (
        _active_head_batch_target(batch, "y_side_mask", device).reshape(-1) > 0.5
    )
    side_target = nn.functional.one_hot(y_side.clamp(0, 1), num_classes=2).float()
    y_long_utility = _active_head_batch_target(
        batch, "y_long_path_utility_bps", device
    ).reshape(-1)
    y_short_utility = _active_head_batch_target(
        batch, "y_short_path_utility_bps", device
    ).reshape(-1)
    side_utility_target = torch.stack(
        [y_long_utility, y_short_utility],
        dim=1,
    ) / max(1.0, float(path_scale_bps))
    y_long_bad = _active_head_batch_target(
        batch, "y_long_bad_path", device
    ).reshape(-1).clamp(0.0, 1.0)
    y_short_bad = _active_head_batch_target(
        batch, "y_short_bad_path", device
    ).reshape(-1).clamp(0.0, 1.0)
    side_bad_target = torch.stack([y_long_bad, y_short_bad], dim=1)
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
    ) / max(1.0, float(mfe_scale_bps))
    side_validity_target = torch.stack(
        [
            (
                (y_long_utility >= float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS))
                & (y_long_bad < 0.5)
            ).float(),
            (
                (y_short_utility >= float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS))
                & (y_short_bad < 0.5)
            ).float(),
        ],
        dim=1,
    )
    trendline_target = torch.stack(
        [
            _active_head_batch_target(
                batch, "y_rising_channel_support_touch", device
            ),
            _active_head_batch_target(
                batch, "y_falling_channel_resistance_touch", device
            ),
            _active_head_batch_target(batch, "y_countertrend_short_trap", device),
            _active_head_batch_target(batch, "y_countertrend_long_trap", device),
            _active_head_batch_target(
                batch, "y_short_high_mae_low_mfe_early_failure", device
            ),
            _active_head_batch_target(
                batch, "y_long_high_mae_low_mfe_early_failure", device
            ),
        ],
        dim=1,
    ).clamp(0.0, 1.0)

    reward_target = (
        torch.stack(
            [
                _active_head_batch_target(batch, name, device)
                for name in _OFFLINE_RL_TARGET_COLS
            ],
            dim=1,
        )
        / float(OFFLINE_RL_REWARD_SCALE_BPS)
    )
    reward_cube = reward_target.reshape(
        batch_size,
        OFFLINE_RL_ACTION_COUNT,
        OFFLINE_RL_HORIZON_COUNT,
    )
    value_target = reward_cube.max(dim=1).values
    advantage_target = (reward_cube - value_target.unsqueeze(1)).reshape(
        batch_size,
        ACTION_VALUE_DIM,
    )

    surfaces: Dict[
        str,
        Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ] = {
        "direction": {
            "model_native_logits": (
                _prediction("model_native_logits"),
                direction_target,
                all_rows,
            )
        },
        "tradable": {
            "tradable_logit": (
                _prediction("tradable_logit"),
                y_tradable.unsqueeze(1),
                selector_rows,
            )
        },
        "path_quality": {
            "path_quality_raw": (
                _prediction("path_quality_raw"),
                (
                    _active_head_batch_target(
                        batch, "path_quality_bps", device
                    ).reshape(-1, 1)
                    / max(1.0, float(path_scale_bps))
                ),
                positive_rows,
            )
        },
        "mfe_first_n": {
            "mfe_first_n": (
                _prediction("mfe_first_n"),
                (
                    _active_head_batch_target(
                        batch, "mfe_first_n_bps", device
                    ).reshape(-1, 1)
                    / max(1.0, float(mfe_scale_bps))
                ),
                positive_rows,
            )
        },
        "bad_path": {
            "bad_path_logit_raw": (
                _prediction("bad_path_logit_raw"),
                _active_head_batch_target(
                    batch, "y_bad_path", device
                ).reshape(-1, 1),
                selector_rows,
            )
        },
        "clean_edge": {
            "clean_edge_logit": (
                _prediction("clean_edge_logit"),
                clean_target,
                selector_rows,
            )
        },
        "survival": {
            "survival_logit": (
                _prediction("survival_logit"),
                survival_target,
                selector_rows,
            )
        },
        "tf_agreement": {
            "tf_agreement_logit": (
                _prediction("tf_agreement_logit"),
                _active_head_batch_target(
                    batch, "y_tf_agreement_score", device
                ).reshape(-1, 1),
                all_rows,
            )
        },
        "path_quality_log_var": {
            "path_quality_log_var": (
                _prediction("path_quality_log_var"),
                (
                    _active_head_batch_target(
                        batch, "path_quality_bps", device
                    ).reshape(-1, 1)
                    / max(1.0, float(path_scale_bps))
                ),
                positive_rows,
            )
        },
        "position_size": {
            "position_size_logit": (
                _prediction("position_size_logit"),
                _active_head_batch_target(
                    batch, "y_position_size_target", device
                ).reshape(-1, 1),
                all_rows,
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
        "mtf_direction": {
            "mtf_dir_logits": (
                _prediction("mtf_dir_logits"),
                direction_target,
                all_rows,
            )
        },
        "trade_side_hierarchy": {
            "trade_logit": (_prediction("trade_logit"), y_trade, all_rows),
            "side_logits": (
                _prediction("side_logits"),
                side_target,
                y_side_mask,
            ),
            "side_utility": (
                _prediction("side_utility"),
                side_utility_target,
                all_rows,
            ),
            "side_bad_path_logit": (
                _prediction("side_bad_path_logit"),
                side_bad_target,
                all_rows,
            ),
            "side_mae": (
                _prediction("side_mae"),
                side_mae_target,
                all_rows,
            ),
        },
        "trendline_rail": {
            "trendline_rail_logits": (
                _prediction("trendline_rail_logits"),
                trendline_target,
                all_rows,
            )
        },
        "side_validity": {
            "side_validity_logit": (
                _prediction("side_validity_logit"),
                side_validity_target,
                all_rows,
            )
        },
        "offline_rl_action_value": {
            "action_value": (
                _prediction("action_value"),
                reward_target,
                all_rows,
            ),
            "action_advantage": (
                _prediction("action_advantage"),
                advantage_target,
                all_rows,
            ),
        },
        "offline_rl_expectile_value": {
            "expectile_value": (
                _prediction("expectile_value"),
                value_target,
                all_rows,
            )
        },
        "model_native_evidence_fusion": {
            "raw_direction_logits": (
                _prediction("raw_direction_logits"),
                direction_target,
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


def _active_head_ablated_fusion_inputs(
    head_name: str,
    pre_fusion_outputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Apply one symmetric, model-native zero intervention to a head's evidence."""
    if head_name not in _ACTIVE_HEAD_FUSION_INPUTS:
        raise RuntimeError(
            f"[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_UNKNOWN_HEAD] head={head_name}"
        )
    ablated = dict(pre_fusion_outputs)
    if head_name == "offline_rl_action_value":
        action_value = torch.zeros_like(pre_fusion_outputs["action_value"])
        value = pre_fusion_outputs["expectile_value"]
        ablated["action_value"] = action_value
        ablated["action_advantage"] = (
            action_value.reshape(
                action_value.shape[0],
                OFFLINE_RL_ACTION_COUNT,
                OFFLINE_RL_HORIZON_COUNT,
            )
            - value.unsqueeze(1)
        ).reshape(action_value.shape[0], ACTION_VALUE_DIM)
        return ablated
    if head_name == "offline_rl_expectile_value":
        action_value = pre_fusion_outputs["action_value"]
        value = torch.zeros_like(pre_fusion_outputs["expectile_value"])
        ablated["expectile_value"] = value
        ablated["action_advantage"] = (
            action_value.reshape(
                action_value.shape[0],
                OFFLINE_RL_ACTION_COUNT,
                OFFLINE_RL_HORIZON_COUNT,
            )
            - value.unsqueeze(1)
        ).reshape(action_value.shape[0], ACTION_VALUE_DIM)
        return ablated
    for output_name in _ACTIVE_HEAD_FUSION_INPUTS[head_name]:
        ablated[output_name] = torch.zeros_like(pre_fusion_outputs[output_name])
    return ablated


def _new_active_head_epoch_accumulator() -> Dict[str, Any]:
    return {
        "heads": {
            head_name: {"components": {}, "influence": []}
            for head_name in MODEL_NATIVE_ACTIVE_HEADS
        }
    }


def _accumulate_active_head_epoch(
    accumulator: Dict[str, Any],
    model: nn.Module,
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    path_scale_bps: float,
    mfe_scale_bps: float,
) -> None:
    surfaces = _active_head_target_surfaces(
        out,
        batch,
        device,
        path_scale_bps=path_scale_bps,
        mfe_scale_bps=mfe_scale_bps,
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

    pre_fusion_outputs = {
        output_name: out[output_name].float()
        for output_name, _width in EXACT_EVIDENCE_FUSION_OUTPUTS
    }
    raw_logits = out.get("raw_direction_logits")
    if not isinstance(raw_logits, torch.Tensor):
        raise RuntimeError(
            "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_OUTPUT_MISSING] "
            "output=raw_direction_logits"
        )
    raw_centered = raw_logits.float() - raw_logits.float().mean(
        dim=1,
        keepdim=True,
    )
    fusion_owner = model._orig_mod if hasattr(model, "_orig_mod") else model
    fuse = getattr(fusion_owner, "_fuse_direction_evidence", None)
    if not callable(fuse):
        raise RuntimeError("[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_FUSION_CALL_MISSING]")
    for head_name in MODEL_NATIVE_ACTIVE_HEADS:
        ablated_inputs = _active_head_ablated_fusion_inputs(
            head_name,
            pre_fusion_outputs,
        )
        ablated_logits = fuse(ablated_inputs).float()
        ablated_centered = ablated_logits - ablated_logits.mean(
            dim=1,
            keepdim=True,
        )
        head_store[head_name]["influence"].append(
            (raw_centered - ablated_centered).detach().float().cpu().numpy()
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

        influence_chunks = evidence.get("influence")
        influence_max_abs = 0.0
        influence_rms = 0.0
        influence_rows = 0
        if not influence_chunks:
            failures.append(
                "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_INFLUENCE_MISSING] "
                f"head={head_name}"
            )
        else:
            try:
                influence = np.concatenate(
                    [np.asarray(value, dtype=np.float64) for value in influence_chunks],
                    axis=0,
                )
            except Exception as exc:
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_INFLUENCE_INVALID] "
                    f"head={head_name} error={exc}"
                )
                influence = np.empty((0, 3), dtype=np.float64)
            influence_rows = int(influence.shape[0]) if influence.ndim == 2 else 0
            if (
                influence.ndim != 2
                or influence.shape[1] != 3
                or influence_rows < _ACTIVE_HEAD_DIAGNOSTIC_MIN_ROWS
                or not np.isfinite(influence).all()
            ):
                failures.append(
                    "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_INFLUENCE_EVIDENCE_INVALID] "
                    f"head={head_name} shape={influence.shape}"
                )
            else:
                influence_max_abs = float(np.max(np.abs(influence)))
                influence_rms = float(np.sqrt(np.mean(np.square(influence))))
                if (
                    influence_max_abs
                    <= _ACTIVE_HEAD_DIAGNOSTIC_INFLUENCE_EPS
                ):
                    failures.append(
                        "[ENTRY_ACTIVE_HEAD_DIAGNOSTIC_INFLUENCE_DEAD] "
                        f"head={head_name} max_abs={influence_max_abs:.12g}"
                    )
        head_metrics[head_name] = {
            "ok": len(failures) == failure_count_before,
            "components": component_metrics,
            "influence_rows": influence_rows,
            "influence_class_centered_max_abs": influence_max_abs,
            "influence_class_centered_rms": influence_rms,
        }

    metrics = {
        "active_head_diagnostic_schema": (
            "entry_model_native_active_head_epoch_diagnostics_v1"
        ),
        "active_head_contract": list(MODEL_NATIVE_ACTIVE_HEADS),
        "active_head_diagnostics": head_metrics,
        "active_head_health_ok": not failures,
        "active_head_health_failures": list(failures),
    }
    return metrics, failures


def _direction_ckpt_balance_stats(
    targets_np: np.ndarray,
    preds_np: np.ndarray,
    acc: float,
) -> Dict[str, Any]:
    targets_i = np.asarray(targets_np, dtype=np.int64)
    preds_i = np.asarray(preds_np, dtype=np.int64)
    label_counts = np.bincount(targets_i, minlength=3)[:3].astype(np.float64)
    pred_counts = np.bincount(preds_i, minlength=3)[:3].astype(np.float64)
    label_total = float(label_counts.sum())
    pred_total = float(pred_counts.sum())
    label_rates = label_counts / max(label_total, 1.0)
    pred_rates = pred_counts / max(pred_total, 1.0)
    active = label_rates > 0.0
    pred_to_label = np.divide(
        pred_rates,
        np.maximum(label_rates, 1e-12),
        out=np.zeros_like(pred_rates),
        where=active,
    )
    min_pred_to_label = float(np.min(pred_to_label[active])) if np.any(active) else 0.0
    l1_drift = float(np.abs(pred_rates - label_rates).sum())
    min_pred_to_label_req = float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL)
    min_pred_rate_req = float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE)
    required_pred_rates = np.maximum(label_rates * min_pred_to_label_req, min_pred_rate_req)
    guard_ok = bool(np.all(pred_rates[active] + 1e-12 >= required_pred_rates[active]))
    guard_weight = float(ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT)
    penalty = guard_weight * l1_drift
    if guard_weight > 0.0 and not guard_ok:
        penalty += guard_weight
    score = float(acc) - float(penalty)
    return {
        "direction_label_rate_long": float(label_rates[0]),
        "direction_label_rate_short": float(label_rates[1]),
        "direction_label_rate_flat": float(label_rates[2]),
        "direction_pred_rate_long": float(pred_rates[0]),
        "direction_pred_rate_short": float(pred_rates[1]),
        "direction_pred_rate_flat": float(pred_rates[2]),
        "direction_pred_label_l1": l1_drift,
        "direction_min_pred_to_label": min_pred_to_label,
        "direction_class_balance_guard_ok": guard_ok,
        "direction_ckpt_balance_penalty": float(penalty),
        "direction_ckpt_score": score,
        "direction_ckpt_guard_weight": guard_weight,
        "direction_ckpt_min_pred_to_label": min_pred_to_label_req,
        "direction_ckpt_min_pred_rate": min_pred_rate_req,
    }


def _bounded_pos_weight(raw_pos_weight: float, cap: float, *, allow_below_one: bool = False) -> float:
    cap_f = max(1.0, float(cap))
    floor = (1.0 / cap_f) if bool(allow_below_one) else 1.0
    raw = float(raw_pos_weight)
    if not np.isfinite(raw) or raw <= 0.0:
        raw = floor
    return float(min(cap_f, max(floor, raw)))


def _optional_float_1d(values: Optional[np.ndarray], expected_size: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != int(expected_size):
        return None
    return arr


def _optional_int_1d(values: Optional[np.ndarray], expected_size: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size != int(expected_size):
        return None
    return arr


def _direction_hierarchy_output_stats(
    targets_np: np.ndarray,
    trade_prob_np: Optional[np.ndarray] = None,
    side_pred_np: Optional[np.ndarray] = None,
    side_long_prob_np: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    targets_i = np.asarray(targets_np, dtype=np.int64).reshape(-1)
    n = int(targets_i.size)
    if n <= 0:
        return {}

    stats: Dict[str, Any] = {}
    edge_mask = targets_i != 2
    flat_mask = targets_i == 2
    if bool(np.any(edge_mask)):
        stats["hier_side_target_long_rate_on_edge"] = float(np.mean(targets_i[edge_mask] == 0))
    if bool(np.any(flat_mask)):
        stats["hier_flat_label_rate"] = float(np.mean(flat_mask))
    stats["hier_trade_target_rate"] = float(np.mean(edge_mask))

    trade_prob = _optional_float_1d(trade_prob_np, n)
    if trade_prob is not None:
        finite = np.isfinite(trade_prob)
        if bool(np.any(finite)):
            trade_pred = trade_prob >= 0.5
            stats["hier_trade_prob_mean"] = float(np.mean(trade_prob[finite]))
            stats["hier_flat_prob_mean"] = float(np.mean(1.0 - trade_prob[finite]))
            stats["hier_trade_pred_rate"] = float(np.mean(trade_pred[finite]))
            stats["hier_flat_pred_rate"] = float(np.mean(~trade_pred[finite]))
            if bool(np.any(edge_mask & finite)):
                stats["hier_trade_prob_label_edge_mean"] = float(np.mean(trade_prob[edge_mask & finite]))
            if bool(np.any(flat_mask & finite)):
                stats["hier_trade_prob_label_flat_mean"] = float(np.mean(trade_prob[flat_mask & finite]))
                stats["hier_flat_prob_label_flat_mean"] = float(np.mean(1.0 - trade_prob[flat_mask & finite]))

    side_pred = _optional_int_1d(side_pred_np, n)
    if side_pred is not None:
        finite_side = np.isfinite(side_pred.astype(np.float64, copy=False))
        if bool(np.any(edge_mask & finite_side)):
            side_targets = targets_i[edge_mask & finite_side]
            side_preds = side_pred[edge_mask & finite_side]
            stats["hier_side_pred_long_rate_on_edge"] = float(np.mean(side_preds == 0))
            stats["hier_side_pred_short_rate_on_edge"] = float(np.mean(side_preds == 1))
            stats["hier_side_acc_on_edge"] = float(np.mean(side_preds == side_targets))

    side_long_prob = _optional_float_1d(side_long_prob_np, n)
    if side_long_prob is not None:
        finite_long = np.isfinite(side_long_prob)
        if bool(np.any(edge_mask & finite_long)):
            stats["hier_side_long_prob_mean_on_edge"] = float(np.mean(side_long_prob[edge_mask & finite_long]))
        long_edge = (targets_i == 0) & finite_long
        short_edge = (targets_i == 1) & finite_long
        if bool(np.any(long_edge)):
            stats["hier_side_long_prob_label_long_mean"] = float(np.mean(side_long_prob[long_edge]))
        if bool(np.any(short_edge)):
            stats["hier_side_long_prob_label_short_mean"] = float(np.mean(side_long_prob[short_edge]))
    return stats


def _direction_slice_balance_stats(
    targets_np: np.ndarray,
    preds_np: np.ndarray,
    ctx_cat_np: Optional[np.ndarray],
    trade_prob_np: Optional[np.ndarray] = None,
    side_pred_np: Optional[np.ndarray] = None,
    side_long_prob_np: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Audit final three-class direction and learned hierarchy by context slice."""

    targets_i = np.asarray(targets_np, dtype=np.int64).reshape(-1)
    preds_i = np.asarray(preds_np, dtype=np.int64).reshape(-1)

    def _empty() -> Dict[str, Any]:
        return {
            "direction_slice_audited_count": 0,
            "direction_slice_failure_count": 0,
            "direction_slice_accuracy_failure_count": 0,
            "direction_slice_pred_rate_failure_count": 0,
            "direction_slice_accuracy_deficit": 0.0,
            "direction_slice_pred_rate_shortfall": 0.0,
            "direction_slice_failure_details": [],
        }

    if ctx_cat_np is None or targets_i.size <= 0 or preds_i.size != targets_i.size:
        return _empty()
    cat = np.asarray(ctx_cat_np)
    if cat.ndim == 1:
        cat = cat.reshape(-1, 1)
    if cat.ndim != 2 or cat.shape[0] != targets_i.size:
        return _empty()

    min_rows = max(_DIRECTION_AUDIT_MIN_SLICE_ROWS, int(ENTRY_DIRECTION_SLICE_MIN_ROWS))
    min_label_rate = max(
        _DIRECTION_AUDIT_MIN_LABEL_RATE,
        float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE),
    )
    min_pred_rate = max(
        _DIRECTION_AUDIT_MIN_PRED_RATE,
        float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE),
    )
    pred_to_label = max(
        _DIRECTION_AUDIT_MIN_PRED_TO_LABEL,
        float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL),
    )
    indices = _direction_slice_ctx_cat_indices(int(cat.shape[1]))
    if not indices:
        indices = list(range(int(cat.shape[1])))

    trade_prob = _optional_float_1d(trade_prob_np, targets_i.size)
    side_pred = _optional_int_1d(side_pred_np, targets_i.size)
    side_long_prob = _optional_float_1d(side_long_prob_np, targets_i.size)

    audited = 0
    accuracy_failures = 0
    pred_rate_failures = 0
    accuracy_deficit = 0.0
    pred_rate_shortfall = 0.0
    failure_details: list[dict[str, Any]] = []
    for idx in indices:
        values = cat[:, idx].astype(np.float64, copy=False)
        finite = np.isfinite(values)
        if not bool(np.any(finite)):
            continue
        int_values = cat[:, idx].astype(np.int64, copy=False)
        for raw_value in sorted(set(int(v) for v in values[finite].ravel())):
            mask = finite & (int_values == int(raw_value))
            rows = int(mask.sum())
            if rows < min_rows:
                continue
            labels_s = targets_i[mask]
            preds_s = preds_i[mask]
            label_counts = np.bincount(labels_s, minlength=3)[:3].astype(np.float64)
            label_rates = label_counts / max(float(label_counts.sum()), 1.0)
            active = label_rates >= min_label_rate
            if int(active.sum()) < 2:
                continue

            audited += 1
            pred_counts = np.bincount(preds_s, minlength=3)[:3].astype(np.float64)
            pred_rates = pred_counts / max(float(pred_counts.sum()), 1.0)
            accuracy = float(np.mean(preds_s == labels_s))
            majority = float(label_rates.max())
            slice_accuracy_deficit = max(0.0, majority - accuracy)
            accuracy_failed = bool(accuracy <= majority + 1e-12)
            if accuracy_failed:
                accuracy_failures += 1
                accuracy_deficit += slice_accuracy_deficit

            required = np.maximum(label_rates * pred_to_label, min_pred_rate)
            shortfalls = np.maximum(required[active] - pred_rates[active], 0.0)
            failed_active = shortfalls > 1e-12
            pred_rate_failures += int(np.sum(failed_active))
            pred_rate_shortfall += float(shortfalls.sum())
            if not accuracy_failed and not bool(np.any(failed_active)):
                continue

            active_classes = np.flatnonzero(active)
            failed_classes = [
                int(class_id)
                for class_id, failed in zip(
                    active_classes.tolist(),
                    failed_active.tolist(),
                    strict=True,
                )
                if bool(failed)
            ]
            hierarchy_detail = _direction_hierarchy_output_stats(
                labels_s,
                trade_prob[mask] if trade_prob is not None else None,
                side_pred[mask] if side_pred is not None else None,
                side_long_prob[mask] if side_long_prob is not None else None,
            )
            failure_details.append(
                {
                    "ctx_cat_index": int(idx),
                    "ctx_cat_value": int(raw_value),
                    "rows": rows,
                    "accuracy": accuracy,
                    "majority": majority,
                    "accuracy_failed": accuracy_failed,
                    "accuracy_deficit": slice_accuracy_deficit,
                    "label_rates": [float(v) for v in label_rates.tolist()],
                    "pred_rates": [float(v) for v in pred_rates.tolist()],
                    "required_pred_rates": [float(v) for v in required.tolist()],
                    "pred_rate_failed_classes": failed_classes,
                    "pred_rate_shortfalls": [float(v) for v in shortfalls.tolist()],
                    "pred_rate_shortfall": float(shortfalls.sum()),
                    **hierarchy_detail,
                }
            )

    failure_count = int(accuracy_failures + pred_rate_failures)
    failure_details = sorted(
        failure_details,
        key=lambda item: (
            float(item.get("accuracy_deficit", 0.0) or 0.0)
            + float(item.get("pred_rate_shortfall", 0.0) or 0.0),
            int(item.get("rows", 0) or 0),
        ),
        reverse=True,
    )[:32]
    return {
        "direction_slice_audited_count": int(audited),
        "direction_slice_failure_count": failure_count,
        "direction_slice_accuracy_failure_count": int(accuracy_failures),
        "direction_slice_pred_rate_failure_count": int(pred_rate_failures),
        "direction_slice_accuracy_deficit": float(accuracy_deficit),
        "direction_slice_pred_rate_shortfall": float(pred_rate_shortfall),
        "direction_slice_contract_ok": bool(audited > 0 and failure_count == 0),
        "direction_slice_min_rows": int(min_rows),
        "direction_slice_min_label_rate": float(min_label_rate),
        "direction_slice_min_pred_rate": float(min_pred_rate),
        "direction_slice_min_pred_to_label": float(pred_to_label),
        "direction_slice_failure_details": failure_details,
    }



def _direction_slice_ckpt_score(base_score: float, slice_stats: Dict[str, Any]) -> float:
    failures = float(slice_stats.get("direction_slice_failure_count", 0.0) or 0.0)
    rate_shortfall = float(slice_stats.get("direction_slice_pred_rate_shortfall", 0.0) or 0.0)
    acc_deficit = float(slice_stats.get("direction_slice_accuracy_deficit", 0.0) or 0.0)
    return float(base_score) - (
        _DIRECTION_SLICE_CKPT_FAILURE_PENALTY * failures
        + _DIRECTION_SLICE_CKPT_DEFICIT_PENALTY
        * (rate_shortfall + acc_deficit)
    )


def _checkpoint_admission_ok(
    *,
    profile: str,
    aux_head_health_ok: bool,
    active_head_health_ok: bool,
    cooperation_gate_health_ok: bool,
    class_support_ok: bool,
) -> bool:
    """Decide checkpoint admission for the exact training profile.

    Profile-separated admission (user vedtak 2026-07-25).

    ``candidate`` is unchanged: auxiliary head health, active head health and
    cooperation gate health all block admission, and only a candidate bundle
    may enter the acceptance chain.

    ``smoke`` answers the trainability question it is named for — does this
    recipe train at all, and does it emit a non-degenerate three-class
    decision — so it admits on active-head liveness plus class support.
    Auxiliary and cooperation health are still computed, logged and journaled
    identically as diagnostics; they do not veto a smoke checkpoint. A smoke
    bundle carries zero edge, promotion or launch authority: the smoke bundle
    audit, candidate readiness, serve parity, sizing, Exit and launch
    finalizer contracts are unchanged and still require the full evidence set.
    """

    if profile == "candidate":
        return bool(
            aux_head_health_ok
            and active_head_health_ok
            and cooperation_gate_health_ok
        )
    if profile == "smoke":
        return bool(active_head_health_ok and class_support_ok)
    raise RuntimeError(f"[ENTRY_TRAIN_PROFILE_INVALID] {profile!r}")


def _direction_slice_hard_red_stop_ready(
    *,
    epoch: int,
    epochs_since_improve: int,
    best_slice_contract_ok: Optional[bool],
    val_stats: Optional[Dict[str, Any]],
) -> bool:
    patience = int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE)
    if patience <= 0:
        return False
    if int(epoch) < int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS):
        return False
    if int(epochs_since_improve) < patience:
        return False
    if bool(best_slice_contract_ok):
        return False
    if not val_stats:
        return False
    if bool(val_stats.get("direction_slice_contract_ok", False)):
        return False
    return int(val_stats.get("direction_slice_failure_count", 0) or 0) > 0


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


def _direction_slice_stats_snapshot(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    keys = (
        "direction_slice_audited_count",
        "direction_slice_failure_count",
        "direction_slice_accuracy_failure_count",
        "direction_slice_pred_rate_failure_count",
        "direction_slice_accuracy_deficit",
        "direction_slice_pred_rate_shortfall",
        "direction_slice_contract_ok",
        "direction_slice_min_rows",
        "direction_slice_min_label_rate",
        "direction_slice_min_pred_rate",
        "direction_slice_min_pred_to_label",
        "direction_slice_failure_details",
        "direction_ckpt_score",
        "direction_slice_ckpt_score",
        "direction_class_balance_guard_ok",
        "direction_pred_rate_long",
        "direction_pred_rate_short",
        "direction_pred_rate_flat",
        "direction_label_rate_long",
        "direction_label_rate_short",
        "direction_label_rate_flat",
        "hier_trade_target_rate",
        "hier_trade_prob_mean",
        "hier_flat_prob_mean",
        "hier_trade_pred_rate",
        "hier_flat_pred_rate",
        "hier_trade_prob_label_edge_mean",
        "hier_trade_prob_label_flat_mean",
        "hier_flat_prob_label_flat_mean",
        "hier_side_target_long_rate_on_edge",
        "hier_side_pred_long_rate_on_edge",
        "hier_side_pred_short_rate_on_edge",
        "hier_side_acc_on_edge",
        "hier_side_long_prob_mean_on_edge",
        "hier_side_long_prob_label_long_mean",
        "hier_side_long_prob_label_short_mean",
    )
    return {key: stats.get(key) for key in keys if key in stats}


def _direction_slice_failure_evidence_path(out_bundle_dir: Path) -> Path:
    resolved = Path(out_bundle_dir).expanduser().resolve()
    return resolved.parent / f"{resolved.name}__direction_slice_failure_evidence.json"


def _write_direction_slice_failure_evidence(out_bundle_dir: Path, payload: Dict[str, Any]) -> Path:
    path = _direction_slice_failure_evidence_path(out_bundle_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "evidence_json": str(path),
        "bundle_written": False,
        "promotion_shadow_live_allowed": False,
    }
    path.write_text(
        json.dumps(enriched, indent=2, sort_keys=True, default=_train_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _direction_ckpt_balance_guard_required() -> bool:
    return (
        float(ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT) > 0.0
        and (
            float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL) > 0.0
            or float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE) > 0.0
        )
    )


def _direction_ckpt_slice_guard_required() -> bool:
    return bool(ENTRY_CKPT_DIRECTION_SLICE_GUARD)


def validate(
    model,
    loader,
    criterion,
    device,
    aux_path_weight: float,
    aux_mfe_weight: float,
    aux_tradable_weight: float,
    aux_path_scale_bps: float,
    aux_mfe_scale_bps: float,
    tradable_pos_weight: float,
    clean_edge_pos_weight: float,
    survival_pos_weight: float,
    bad_path_pos_weight: float,
    hier_trade_pos_weight: float,
    hier_bad_path_pos_weight: Any,
):
    model.eval()
    total = 0.0
    total_ce = 0.0
    total_cost = 0.0
    total_balance = 0.0
    total_direction_min_pred = 0.0
    total_direction_global_prior_match = 0.0
    total_direction_slice_min_pred = 0.0
    total_direction_slice_recall = 0.0
    total_direction_slice_balanced_ce = 0.0
    total_direction_slice_true_margin = 0.0
    total_direction_slice_accuracy_edge = 0.0
    total_direction_slice_confusion_pair = 0.0
    total_direction_slice_prior_match = 0.0
    total_direction_flat_margin = 0.0
    total_direction_utility_margin = 0.0
    total_direction_side_utility_conviction = 0.0
    total_direction_utility_trade_conviction = 0.0
    total_direction_utility_triad_ce = 0.0
    total_direction_flat_starvation = 0.0
    total_tail_direction = 0.0
    specialist_gate_loss_sum = 0.0
    cooperation_gate_epoch = _new_cooperation_gate_epoch_accumulator()
    bad_path_quality_rank_loss_sum = 0.0
    path_quality_rank_loss_sum = 0.0
    n = 0
    preds, targets = [], []
    ctx_cats: List[np.ndarray] = []
    short_total = 0
    short_pred_long = 0
    path_loss_sum = 0.0
    mfe_loss_sum = 0.0
    tradable_loss_sum = 0.0
    clean_edge_loss_sum = 0.0
    survival_loss_sum = 0.0
    clean_edge_rank_loss_sum = 0.0
    aux_bad_path_bce_loss_sum = 0.0
    bad_path_prob_penalty_loss_sum = 0.0
    hard_neg_prob_loss_sum = 0.0
    tail_direction_rows = 0
    hier_trade_loss_sum = 0.0
    hier_trade_global_prior_loss_sum = 0.0
    hier_slice_trade_prior_loss_sum = 0.0
    hier_slice_trade_accuracy_edge_loss_sum = 0.0
    hier_flat_logit_margin_loss_sum = 0.0
    hier_slice_flat_logit_margin_loss_sum = 0.0
    hier_side_loss_sum = 0.0
    hier_slice_side_ce_loss_sum = 0.0
    hier_slice_side_margin_loss_sum = 0.0
    hier_slice_side_accuracy_edge_loss_sum = 0.0
    hier_side_global_prior_loss_sum = 0.0
    hier_slice_side_prior_loss_sum = 0.0
    hier_utility_loss_sum = 0.0
    hier_side_bad_path_loss_sum = 0.0
    hier_side_mae_loss_sum = 0.0
    hier_side_validity_loss_sum = 0.0
    hier_long_valid_target_rate_sum = 0.0
    hier_short_valid_target_rate_sum = 0.0
    hier_long_valid_prob_sum = 0.0
    hier_short_valid_prob_sum = 0.0
    hier_long_bad_target_rate_sum = 0.0
    hier_short_bad_target_rate_sum = 0.0
    hier_countertrend_long_trap_rate_sum = 0.0
    hier_countertrend_short_trap_rate_sum = 0.0
    hier_side_rows_sum = 0
    hier_side_correct_sum = 0.0
    trendline_rail_loss_sum = 0.0
    trendline_rail_rows_sum = 0
    trendline_rising_rows_sum = 0
    trendline_falling_rows_sum = 0
    # V10-AUX-02: read-only accumulators for the cross-head / AUC / realized-target panel.
    _diag_pred: "dict[str, list]" = {k: [] for k in (
        "tradable", "bad_path", "clean_edge", "survival", "path_quality", "mfe_first_n")}
    _diag_lbl: "dict[str, list]" = {k: [] for k in ("tradable", "bad_path", "clean_edge", "survival")}
    _diag_real: "dict[str, list]" = {k: [] for k in ("mfe_first_n_bps", "path_quality_bps")}
    _diag_mask: "dict[str, list]" = {
        k: [] for k in ("trade", "long", "short")
    }
    active_head_epoch = _new_active_head_epoch_accumulator()
    hierarchy_trade_prob_chunks: List[np.ndarray] = []
    hierarchy_side_pred_chunks: List[np.ndarray] = []
    hierarchy_side_long_prob_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            non_blocking = device.type == "cuda"
            seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
            snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
            ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
            ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
            y = batch["y"].to(device, non_blocking=non_blocking)
            y_mfe_first = batch["mfe_first_n_bps"].to(device, non_blocking=non_blocking)
            y_path_quality = batch["path_quality_bps"].to(device, non_blocking=non_blocking)
            y_tradable = batch["y_tradable"].to(device, non_blocking=non_blocking)
            y_bad_path = batch["y_bad_path"].to(device, non_blocking=non_blocking)
            y_dead_negative_long = batch["y_dead_negative_long"].to(device, non_blocking=non_blocking)
            y_teaser_negative_long = batch["y_teaser_negative_long"].to(device, non_blocking=non_blocking)
            y_hard_negative_long = batch["y_hard_negative_long"].to(device, non_blocking=non_blocking)
            y_dead_negative_short = batch["y_dead_negative_short"].to(device, non_blocking=non_blocking)
            y_teaser_negative_short = batch["y_teaser_negative_short"].to(device, non_blocking=non_blocking)
            y_hard_negative_short = batch["y_hard_negative_short"].to(device, non_blocking=non_blocking)
            y_clean_edge_long = batch["y_clean_edge_long"].to(device, non_blocking=non_blocking)
            y_survival_long = batch["y_survival_long"].to(device, non_blocking=non_blocking)
            y_selector_long_mask = batch["y_selector_long_mask"].to(device, non_blocking=non_blocking)
            y_selector_short_mask = batch["y_selector_short_mask"].to(device, non_blocking=non_blocking)
            y_clean_edge_bidir = batch["y_clean_edge_bidir"].to(device, non_blocking=non_blocking)
            y_survival_bidir = batch["y_survival_bidir"].to(device, non_blocking=non_blocking)

            out = _autocast_forward(model, seq_x.device, seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **_multi_tf_kwargs_from_batch(batch, seq_x.device))
            _accumulate_active_head_epoch(
                active_head_epoch,
                model,
                out,
                batch,
                device,
                path_scale_bps=aux_path_scale_bps,
                mfe_scale_bps=aux_mfe_scale_bps,
            )
            logits = out["direction_logits"]
            path_pred = out["path_quality"]
            path_log_var = out["path_quality_log_var"]
            mfe_pred = out["mfe_first_n"]
            tradable_logit = out["tradable_logit"]
            bad_path_logit = out["bad_path_logit"]
            clean_edge_logit = out["clean_edge_logit"]
            survival_logit = out["survival_logit"]
            trade_logit = out["trade_logit"]
            side_logits = out["side_logits"]
            specialist_gate_loss, _specialist_gate_stats = _specialist_gate_regularization(out, device)
            _accumulate_cooperation_gate_epoch(cooperation_gate_epoch, out)
            bad_path_quality_rank_loss = _bad_path_quality_rank_loss(bad_path_logit, y_path_quality, device)
            path_quality_rank_loss = _path_quality_rank_loss(path_pred, y_path_quality, device)

            # V10-AUX-02: accumulate per-head probs/preds + labels + realized targets for
            # the read-only diagnostic panel (computed after the loop). Detached, no grad.
            def _np1d(t):
                return t.detach().float().cpu().numpy().reshape(-1)
            _diag_pred["tradable"].append(_np1d(torch.sigmoid(tradable_logit)))
            _diag_pred["bad_path"].append(_np1d(torch.sigmoid(bad_path_logit)))
            _diag_pred["clean_edge"].append(_np1d(torch.sigmoid(clean_edge_logit)))
            _diag_pred["survival"].append(_np1d(torch.sigmoid(survival_logit)))
            _diag_pred["path_quality"].append(_np1d(path_pred))
            _diag_pred["mfe_first_n"].append(_np1d(mfe_pred))
            hierarchy_trade_prob_chunks.append(
                torch.sigmoid(trade_logit.float()).detach().cpu().numpy().reshape(-1)
            )
            side_probs = torch.softmax(side_logits.float(), dim=1)
            hierarchy_side_pred_chunks.append(
                torch.argmax(side_probs, dim=1).detach().cpu().numpy().astype(np.int64).reshape(-1)
            )
            hierarchy_side_long_prob_chunks.append(side_probs[:, 0].detach().cpu().numpy().reshape(-1))
            _diag_lbl["tradable"].append(_np1d(y_tradable))
            _diag_lbl["bad_path"].append(_np1d(y_bad_path))
            _diag_lbl["clean_edge"].append(_np1d(_aux_clean_edge_target(y_clean_edge_long, y_clean_edge_bidir)))
            _diag_lbl["survival"].append(_np1d(_aux_survival_target(y_survival_long, y_survival_bidir)))
            _diag_real["mfe_first_n_bps"].append(_np1d(y_mfe_first))
            _diag_real["path_quality_bps"].append(_np1d(y_path_quality))
            _diag_mask["trade"].append(_np1d(y != 2))
            _diag_mask["long"].append(_np1d(y == 0))
            _diag_mask["short"].append(_np1d(y == 1))

            residual_hard_neg_long = _hard_negative_residual(
                y_hard_negative_long, y_dead_negative_long, y_teaser_negative_long
            )
            residual_hard_neg_short = _hard_negative_residual(
                y_hard_negative_short, y_dead_negative_short, y_teaser_negative_short
            )
            ce_per = criterion.ce(logits, y)
            ce_sample_weight = _direction_ce_sample_weight(
                y_bad_path,
                y_dead_negative_long,
                y_teaser_negative_long,
                residual_hard_neg_long,
                y_dead_negative_short,
                y_teaser_negative_short,
                residual_hard_neg_short,
            ).to(device=ce_per.device, dtype=ce_per.dtype)
            ce_loss_raw = (ce_per * ce_sample_weight).mean()
            ce_loss = float(ENTRY_DIRECTION_CE_SCALE) * ce_loss_raw
            probs = torch.softmax(logits, dim=1)
            tail_direction_loss = torch.tensor(0.0, device=device)
            tail_direction_mask = torch.zeros_like(y, dtype=torch.bool)
            if float(ENTRY_TAIL_DIRECTION_CE_WEIGHT) > 0.0:
                tail_direction_mask = _tail_direction_mask(y, y_tradable, y_bad_path, y_path_quality)
                if tail_direction_mask.any():
                    tail_direction_loss = float(ENTRY_TAIL_DIRECTION_CE_WEIGHT) * ce_per[tail_direction_mask].mean()

            cost_term = torch.tensor(0.0, device=device)
            balance_term = _direction_balance_term(probs, y, criterion)
            min_pred_rate_term = _direction_min_pred_rate_term(probs, y)
            global_prior_match_term = _direction_global_prior_match_term(probs, y)
            slice_min_pred_rate_term = _direction_slice_min_pred_rate_term(probs, y, ctx_cat)
            slice_recall_term = _direction_slice_recall_prob_term(probs, y, ctx_cat)
            slice_balanced_ce_term = _direction_slice_balanced_ce_term(logits, y, ctx_cat)
            slice_true_margin_term = _direction_slice_true_margin_term(logits, y, ctx_cat)
            slice_accuracy_edge_term = _direction_slice_accuracy_edge_term(logits, y, ctx_cat)
            slice_confusion_pair_term = _direction_slice_confusion_pair_term(logits, y, ctx_cat)
            slice_prior_match_term = _direction_slice_prior_match_term(probs, y, ctx_cat)
            direction_flat_margin_term = _direction_vs_flat_margin_term(logits, y)
            direction_utility_margin_term = _direction_utility_margin_term(
                logits,
                batch["y_long_path_utility_bps"],
                batch["y_short_path_utility_bps"],
            )
            direction_side_utility_conviction_term = _direction_side_utility_conviction_term(
                logits,
                y,
                batch["y_long_path_utility_bps"],
                batch["y_short_path_utility_bps"],
            )
            direction_utility_trade_conviction_term = _direction_utility_trade_conviction_term(
                logits,
                batch["y_long_path_utility_bps"],
                batch["y_short_path_utility_bps"],
                batch["y_long_bad_path"],
                batch["y_short_bad_path"],
            )
            direction_utility_triad_ce_term = _direction_utility_triad_ce_term(
                logits,
                batch["y_long_path_utility_bps"],
                batch["y_short_path_utility_bps"],
                batch["y_long_bad_path"],
                batch["y_short_bad_path"],
            )
            direction_flat_starvation_term = _direction_flat_starvation_term(logits, y, ctx_cat)
            if bool(getattr(criterion, "enabled", False)):
                cost = criterion.cost_matrix.to(dtype=logits.dtype)[y]
                expected_cost = (cost * probs).sum(dim=1)
                cost_term = float(getattr(criterion, "cost_scale", 1.0)) * expected_cost.mean()

            loss = (
                ce_loss
                + cost_term
                + balance_term
                + min_pred_rate_term
                + global_prior_match_term
                + slice_min_pred_rate_term
                + slice_recall_term
                + slice_balanced_ce_term
                + slice_true_margin_term
                + slice_accuracy_edge_term
                + slice_confusion_pair_term
                + slice_prior_match_term
                + direction_flat_margin_term
                + direction_utility_margin_term
                + direction_side_utility_conviction_term
                + direction_utility_trade_conviction_term
                + direction_utility_triad_ce_term
                + direction_flat_starvation_term
            )
            if float(ENTRY_TAIL_DIRECTION_CE_WEIGHT) > 0.0:
                loss = loss + tail_direction_loss
            if float(ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT) > 0.0 or float(ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT) > 0.0:
                loss = loss + specialist_gate_loss
            if float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT) > 0.0:
                loss = loss + bad_path_quality_rank_loss
            if float(ENTRY_PATH_QUALITY_RANK_WEIGHT) > 0.0:
                loss = loss + path_quality_rank_loss
            hier_loss, hier_stats = _hierarchical_entry_loss(
                out,
                batch,
                device,
                trade_pos_weight=hier_trade_pos_weight,
                side_bad_path_pos_weight=hier_bad_path_pos_weight,
            )
            if hier_loss.numel() == 1:
                loss = loss + hier_loss
            trendline_rail_loss, trendline_stats = _trendline_rail_aux_loss(out, batch, device)
            if trendline_rail_loss.numel() == 1:
                loss = loss + trendline_rail_loss
            hard_neg_prob_loss = torch.tensor(0.0, device=device)
            dead_neg_prob_loss = torch.tensor(0.0, device=device)
            teaser_neg_prob_loss = torch.tensor(0.0, device=device)
            dead_neg_mask = y_dead_negative_long.float() > 0.5
            teaser_neg_mask = y_teaser_negative_long.float() > 0.5
            hard_neg_mask = residual_hard_neg_long > 0.5
            if float(ENTRY_DEAD_LONG_PROB_PENALTY) > 0.0 and dead_neg_mask.any():
                dead_neg_prob_loss = float(ENTRY_DEAD_LONG_PROB_PENALTY) * probs[dead_neg_mask, 0].mean()
                loss = loss + dead_neg_prob_loss
            if float(ENTRY_TEASER_LONG_PROB_PENALTY) > 0.0 and teaser_neg_mask.any():
                teaser_neg_prob_loss = float(ENTRY_TEASER_LONG_PROB_PENALTY) * probs[teaser_neg_mask, 0].mean()
                loss = loss + teaser_neg_prob_loss
            bad_path_prob_penalty_loss = _selected_side_bad_path_probability_penalty(
                probs,
                y,
                y_bad_path,
                ENTRY_BAD_PATH_PROB_PENALTY,
            )
            loss = loss + bad_path_prob_penalty_loss
            if float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) > 0.0 and hard_neg_mask.any():
                hard_neg_prob_loss = float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) * probs[hard_neg_mask, 0].mean()
                loss = loss + hard_neg_prob_loss
            if ENTRY_SYMMETRIC_NEGATIVES:
                dead_neg_short_mask = y_dead_negative_short.float() > 0.5
                teaser_neg_short_mask = y_teaser_negative_short.float() > 0.5
                hard_neg_short_mask = residual_hard_neg_short > 0.5
                if float(ENTRY_DEAD_LONG_PROB_PENALTY) > 0.0 and dead_neg_short_mask.any():
                    loss = loss + float(ENTRY_DEAD_LONG_PROB_PENALTY) * probs[dead_neg_short_mask, 1].mean()
                if float(ENTRY_TEASER_LONG_PROB_PENALTY) > 0.0 and teaser_neg_short_mask.any():
                    loss = loss + float(ENTRY_TEASER_LONG_PROB_PENALTY) * probs[teaser_neg_short_mask, 1].mean()
                if float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) > 0.0 and hard_neg_short_mask.any():
                    loss = loss + float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) * probs[hard_neg_short_mask, 1].mean()

            tradable_loss = torch.tensor(0.0, device=device)
            clean_edge_loss = torch.tensor(0.0, device=device)
            survival_loss = torch.tensor(0.0, device=device)
            clean_edge_rank_loss = torch.tensor(0.0, device=device)
            path_loss = torch.tensor(0.0, device=device)
            mfe_loss = torch.tensor(0.0, device=device)
            positive_mask = y_tradable.float() > 0.5
            selector_mask = _aux_selector_mask(y_selector_long_mask, y_selector_short_mask)
            clean_edge_target = _aux_clean_edge_target(y_clean_edge_long, y_clean_edge_bidir)
            survival_target = _aux_survival_target(y_survival_long, y_survival_bidir)
            if aux_path_weight > 0.0:
                if positive_mask.any():
                    path_target = _signed_scaled_aux_regression_target(
                        y_path_quality,
                        positive_mask,
                        aux_path_scale_bps,
                    )
                    # V10 v3+ Target 2: heteroscedastic path_quality NLL (val)
                    mu = path_pred.squeeze(1)[positive_mask]
                    lv = path_log_var.squeeze(1)[positive_mask].clamp(min=-5.0, max=5.0)
                    sq_err = (path_target.float() - mu) ** 2
                    path_loss = 0.5 * (lv + sq_err / torch.exp(lv)).mean()
                    loss = loss + (float(aux_path_weight) * path_loss)
                    path_loss_sum += float(path_loss.item()) * y.shape[0]
            if aux_mfe_weight > 0.0:
                if positive_mask.any():
                    mfe_target = _signed_scaled_aux_regression_target(
                        y_mfe_first,
                        positive_mask,
                        aux_mfe_scale_bps,
                    )
                    mfe_loss = nn.functional.smooth_l1_loss(
                        mfe_pred.squeeze(1)[positive_mask], mfe_target.float()
                    )
                    loss = loss + (float(aux_mfe_weight) * mfe_loss)
                    mfe_loss_sum += float(mfe_loss.item()) * y.shape[0]
            if aux_tradable_weight > 0.0:
                if selector_mask.any():
                    tradable_loss = nn.functional.binary_cross_entropy_with_logits(
                        tradable_logit.squeeze(1)[selector_mask],
                        y_tradable.float()[selector_mask],
                        pos_weight=torch.tensor(float(tradable_pos_weight), device=device, dtype=tradable_logit.dtype),
                    )
                    loss = loss + (float(aux_tradable_weight) * tradable_loss)
                    tradable_loss_sum += float(tradable_loss.item()) * y.shape[0]
            if float(ENTRY_AUX_BAD_PATH_WEIGHT) > 0.0:
                if selector_mask.any():
                    bad_path_bce_loss = nn.functional.binary_cross_entropy_with_logits(
                        bad_path_logit.squeeze(1)[selector_mask],
                        y_bad_path.float()[selector_mask],
                        pos_weight=torch.tensor(float(bad_path_pos_weight), device=device, dtype=bad_path_logit.dtype),
                    )
                    bad_path_bce_loss = float(ENTRY_AUX_BAD_PATH_WEIGHT) * bad_path_bce_loss
                    loss = loss + bad_path_bce_loss
                    aux_bad_path_bce_loss_sum += float(bad_path_bce_loss.item()) * y.shape[0]
            if float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) > 0.0:
                if selector_mask.any():
                    clean_edge_loss = nn.functional.binary_cross_entropy_with_logits(
                        clean_edge_logit.squeeze(1)[selector_mask],
                        clean_edge_target[selector_mask],
                        pos_weight=torch.tensor(float(clean_edge_pos_weight), device=device, dtype=clean_edge_logit.dtype),
                    )
                    loss = loss + (float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) * clean_edge_loss)
                    clean_edge_loss_sum += float(clean_edge_loss.item()) * y.shape[0]
            if float(ENTRY_AUX_SURVIVAL_WEIGHT) > 0.0:
                if selector_mask.any():
                    survival_loss = nn.functional.binary_cross_entropy_with_logits(
                        survival_logit.squeeze(1)[selector_mask],
                        survival_target[selector_mask],
                        pos_weight=torch.tensor(float(survival_pos_weight), device=device, dtype=survival_logit.dtype),
                    )
                    loss = loss + (float(ENTRY_AUX_SURVIVAL_WEIGHT) * survival_loss)
                    survival_loss_sum += float(survival_loss.item()) * y.shape[0]
            if float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT) > 0.0:
                clean_edge_prob = torch.sigmoid(clean_edge_logit.squeeze(1))
                clean_pos, ranked_neg = _clean_edge_rank_masks(
                    y_clean_edge_long,
                    y_clean_edge_bidir,
                    y_dead_negative_long,
                    y_teaser_negative_long,
                    residual_hard_neg_long,
                    y_dead_negative_short,
                    y_teaser_negative_short,
                    residual_hard_neg_short,
                )
                if clean_pos.any() and ranked_neg.any():
                    pos_long = clean_edge_prob[clean_pos].mean()
                    neg_long = clean_edge_prob[ranked_neg].mean()
                    clean_edge_rank_loss = float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT) * torch.relu(
                        torch.tensor(float(ENTRY_CLEAN_EDGE_RANKING_MARGIN), device=device, dtype=probs.dtype)
                    - (pos_long - neg_long)
                )
                loss = loss + clean_edge_rank_loss
                clean_edge_rank_loss_sum += float(clean_edge_rank_loss.item()) * y.shape[0]
            tf_agreement_logit = _require_active_aux_head_prediction(
                out,
                batch,
                output_name="tf_agreement_logit",
                target_names=("y_tf_agreement_score",),
            )
            y_tf_agreement = batch["y_tf_agreement_score"].to(device, non_blocking=non_blocking)
            tf_pred = torch.sigmoid(tf_agreement_logit).squeeze(-1)
            loss = loss + _MODEL_NATIVE_FIXED_POSITIVE_LOSS_WEIGHTS[
                "tf_agreement"
            ] * torch.nn.functional.mse_loss(tf_pred, y_tf_agreement)
            position_size_logit = _require_active_aux_head_prediction(
                out,
                batch,
                output_name="position_size_logit",
                target_names=("y_position_size_target",),
            )
            y_pos_size = batch["y_position_size_target"].to(device, non_blocking=non_blocking)
            pos_pred = torch.sigmoid(position_size_logit).squeeze(-1)
            loss = loss + _MODEL_NATIVE_FIXED_POSITIVE_LOSS_WEIGHTS[
                "position_size"
            ] * torch.nn.functional.mse_loss(pos_pred, y_pos_size)
            mtf_dir_logits = _require_active_aux_head_prediction(
                out,
                batch,
                output_name="mtf_dir_logits",
                target_names=_DIRECTION_BATCH_TARGET_NAMES,
            )
            mtf_dir_loss = _direction_aux_ce_loss(
                mtf_dir_logits,
                y,
                criterion,
                ce_sample_weight,
            )
            loss = loss + float(ENTRY_MTF_DIR_AUX_WEIGHT) * mtf_dir_loss
            loss = loss + _MODEL_NATIVE_FIXED_POSITIVE_LOSS_WEIGHTS[
                "dip_forecast_timing_tail_vol_composite"
            ] * dip_forecast_loss(
                out,
                batch,
                device,
                bps_scale=aux_mfe_scale_bps,
            )
            loss = loss + offline_rl_aux_loss(out, batch, device)
            bs = y.shape[0]
            total += float(loss) * bs
            total_ce += float(ce_loss) * bs
            total_cost += float(cost_term.detach().cpu().item()) * bs
            total_balance += float(balance_term.detach().cpu().item()) * bs
            total_direction_min_pred += float(min_pred_rate_term.detach().cpu().item()) * bs
            total_direction_global_prior_match += float(global_prior_match_term.detach().cpu().item()) * bs
            total_direction_slice_min_pred += float(slice_min_pred_rate_term.detach().cpu().item()) * bs
            total_direction_slice_recall += float(slice_recall_term.detach().cpu().item()) * bs
            total_direction_slice_balanced_ce += float(slice_balanced_ce_term.detach().cpu().item()) * bs
            total_direction_slice_true_margin += float(slice_true_margin_term.detach().cpu().item()) * bs
            total_direction_slice_accuracy_edge += float(slice_accuracy_edge_term.detach().cpu().item()) * bs
            total_direction_slice_confusion_pair += float(slice_confusion_pair_term.detach().cpu().item()) * bs
            total_direction_slice_prior_match += float(slice_prior_match_term.detach().cpu().item()) * bs
            total_direction_flat_margin += float(direction_flat_margin_term.detach().cpu().item()) * bs
            total_direction_utility_margin += float(direction_utility_margin_term.detach().cpu().item()) * bs
            total_direction_side_utility_conviction += float(
                direction_side_utility_conviction_term.detach().cpu().item()
            ) * bs
            total_direction_utility_trade_conviction += float(
                direction_utility_trade_conviction_term.detach().cpu().item()
            ) * bs
            total_direction_utility_triad_ce += float(direction_utility_triad_ce_term.detach().cpu().item()) * bs
            total_direction_flat_starvation += float(direction_flat_starvation_term.detach().cpu().item()) * bs
            total_tail_direction += float(tail_direction_loss.detach().cpu().item()) * bs
            tail_direction_rows += int(tail_direction_mask.sum().detach().cpu().item())
            specialist_gate_loss_sum += float(specialist_gate_loss.detach().cpu().item()) * bs
            bad_path_quality_rank_loss_sum += float(bad_path_quality_rank_loss.detach().cpu().item()) * bs
            path_quality_rank_loss_sum += float(path_quality_rank_loss.detach().cpu().item()) * bs
            hard_neg_prob_loss_sum += float(hard_neg_prob_loss) * bs
            bad_path_prob_penalty_loss_sum += float(
                bad_path_prob_penalty_loss.detach().cpu().item()
            ) * bs
            hier_trade_loss_sum += float(hier_stats.get("hier_trade_loss", 0.0)) * bs
            hier_trade_global_prior_loss_sum += float(hier_stats.get("hier_trade_global_prior_loss", 0.0)) * bs
            hier_slice_trade_prior_loss_sum += float(hier_stats.get("hier_slice_trade_prior_loss", 0.0)) * bs
            hier_slice_trade_accuracy_edge_loss_sum += float(
                hier_stats.get("hier_slice_trade_accuracy_edge_loss", 0.0)
            ) * bs
            hier_flat_logit_margin_loss_sum += float(hier_stats.get("hier_flat_logit_margin_loss", 0.0)) * bs
            hier_slice_flat_logit_margin_loss_sum += (
                float(hier_stats.get("hier_slice_flat_logit_margin_loss", 0.0)) * bs
            )
            hier_side_loss_sum += float(hier_stats.get("hier_side_loss", 0.0)) * bs
            hier_slice_side_ce_loss_sum += float(hier_stats.get("hier_slice_side_ce_loss", 0.0)) * bs
            hier_slice_side_margin_loss_sum += float(hier_stats.get("hier_slice_side_margin_loss", 0.0)) * bs
            hier_slice_side_accuracy_edge_loss_sum += float(
                hier_stats.get("hier_slice_side_accuracy_edge_loss", 0.0)
            ) * bs
            hier_side_global_prior_loss_sum += float(hier_stats.get("hier_side_global_prior_loss", 0.0)) * bs
            hier_slice_side_prior_loss_sum += float(hier_stats.get("hier_slice_side_prior_loss", 0.0)) * bs
            hier_utility_loss_sum += float(hier_stats.get("hier_utility_loss", 0.0)) * bs
            hier_side_bad_path_loss_sum += float(hier_stats.get("hier_bad_path_loss", 0.0)) * bs
            hier_side_mae_loss_sum += float(hier_stats.get("hier_mae_loss", 0.0)) * bs
            hier_side_validity_loss_sum += float(hier_stats.get("hier_side_validity_loss", 0.0)) * bs
            hier_long_valid_target_rate_sum += float(hier_stats.get("hier_long_valid_target_rate", 0.0)) * bs
            hier_short_valid_target_rate_sum += float(hier_stats.get("hier_short_valid_target_rate", 0.0)) * bs
            hier_long_valid_prob_sum += float(hier_stats.get("hier_long_valid_prob_mean", 0.0)) * bs
            hier_short_valid_prob_sum += float(hier_stats.get("hier_short_valid_prob_mean", 0.0)) * bs
            hier_long_bad_target_rate_sum += float(hier_stats.get("hier_long_bad_target_rate", 0.0)) * bs
            hier_short_bad_target_rate_sum += float(hier_stats.get("hier_short_bad_target_rate", 0.0)) * bs
            hier_countertrend_long_trap_rate_sum += float(hier_stats.get("hier_countertrend_long_trap_rate", 0.0)) * bs
            hier_countertrend_short_trap_rate_sum += float(hier_stats.get("hier_countertrend_short_trap_rate", 0.0)) * bs
            _side_rows = int(hier_stats.get("hier_side_rows", 0.0))
            if _side_rows > 0:
                hier_side_rows_sum += _side_rows
                hier_side_correct_sum += float(hier_stats.get("hier_side_acc", 0.0)) * _side_rows
            trendline_rail_loss_sum += float(trendline_stats.get("trendline_rail_loss", 0.0)) * bs
            trendline_rail_rows_sum += int(trendline_stats.get("trendline_rail_rows", 0.0))
            trendline_rising_rows_sum += int(trendline_stats.get("trendline_rising_rows", 0.0))
            trendline_falling_rows_sum += int(trendline_stats.get("trendline_falling_rows", 0.0))
            n += bs

            p = probs.cpu().numpy()
            preds.extend(np.argmax(p, axis=1).tolist())
            targets.extend(y.cpu().numpy().tolist())
            ctx_cats.append(batch["ctx_cat"].detach().cpu().numpy().astype(np.int64))
            y_np = y.cpu().numpy()
            pred_np = np.argmax(p, axis=1)
            short_total += int((y_np == 1).sum())
            if short_total > 0:
                short_pred_long += int(((pred_np == 0) & (y_np == 1)).sum())

    preds_np = np.asarray(preds)
    targets_np = np.asarray(targets)
    ctx_cat_np = np.concatenate(ctx_cats, axis=0) if ctx_cats else None
    hierarchy_trade_prob_np = (
        np.concatenate(hierarchy_trade_prob_chunks, axis=0) if hierarchy_trade_prob_chunks else None
    )
    hierarchy_side_pred_np = (
        np.concatenate(hierarchy_side_pred_chunks, axis=0) if hierarchy_side_pred_chunks else None
    )
    hierarchy_side_long_prob_np = (
        np.concatenate(hierarchy_side_long_prob_chunks, axis=0)
        if hierarchy_side_long_prob_chunks
        else None
    )

    acc = float(accuracy_score(targets_np.astype(int), preds_np.astype(int)))
    short_pred_long_rate = (short_pred_long / short_total if short_total > 0 else 0.0)
    stats: Dict[str, Any] = {
        "specialist_gate_loss_mean": (specialist_gate_loss_sum / max(1, n)),
        "aux_path_loss_mean": (path_loss_sum / max(1, n)),
        "aux_mfe_loss_mean": (mfe_loss_sum / max(1, n)),
        "aux_tradable_loss_mean": (tradable_loss_sum / max(1, n)),
        "aux_bad_path_bce_loss_mean": (aux_bad_path_bce_loss_sum / max(1, n)),
        "bad_path_prob_penalty_loss_mean": (
            bad_path_prob_penalty_loss_sum / max(1, n)
        ),
        "aux_clean_edge_loss_mean": (clean_edge_loss_sum / max(1, n)),
        "aux_survival_loss_mean": (survival_loss_sum / max(1, n)),
        "clean_edge_rank_loss_mean": (clean_edge_rank_loss_sum / max(1, n)),
        "ce_loss_mean": (total_ce / max(1, n)),
        "cost_loss_mean": (total_cost / max(1, n)),
        "balance_loss_mean": (total_balance / max(1, n)),
        "direction_min_pred_rate_loss_mean": (total_direction_min_pred / max(1, n)),
        "direction_global_prior_match_loss_mean": (
            total_direction_global_prior_match / max(1, n)
        ),
        "direction_slice_min_pred_rate_loss_mean": (
            total_direction_slice_min_pred / max(1, n)
        ),
        "direction_slice_recall_loss_mean": (total_direction_slice_recall / max(1, n)),
        "direction_slice_balanced_ce_loss_mean": (
            total_direction_slice_balanced_ce / max(1, n)
        ),
        "direction_slice_true_margin_loss_mean": (
            total_direction_slice_true_margin / max(1, n)
        ),
        "direction_slice_accuracy_edge_loss_mean": (
            total_direction_slice_accuracy_edge / max(1, n)
        ),
        "direction_slice_confusion_pair_loss_mean": (
            total_direction_slice_confusion_pair / max(1, n)
        ),
        "direction_slice_prior_match_loss_mean": (
            total_direction_slice_prior_match / max(1, n)
        ),
        "direction_flat_margin_loss_mean": (total_direction_flat_margin / max(1, n)),
        "direction_utility_margin_loss_mean": (
            total_direction_utility_margin / max(1, n)
        ),
        "direction_side_utility_conviction_loss_mean": (
            total_direction_side_utility_conviction / max(1, n)
        ),
        "direction_utility_trade_conviction_loss_mean": (
            total_direction_utility_trade_conviction / max(1, n)
        ),
        "direction_utility_triad_ce_loss_mean": (
            total_direction_utility_triad_ce / max(1, n)
        ),
        "direction_flat_starvation_loss_mean": (
            total_direction_flat_starvation / max(1, n)
        ),
        "tail_direction_loss_mean": (total_tail_direction / max(1, n)),
        "tail_direction_rows": int(tail_direction_rows),
        "bad_path_quality_rank_loss_mean": (
            bad_path_quality_rank_loss_sum / max(1, n)
        ),
        "path_quality_rank_loss_mean": (path_quality_rank_loss_sum / max(1, n)),
        "hard_neg_prob_loss_mean": (hard_neg_prob_loss_sum / max(1, n)),
        "hier_trade_loss_mean": (hier_trade_loss_sum / max(1, n)),
        "hier_trade_global_prior_loss_mean": (
            hier_trade_global_prior_loss_sum / max(1, n)
        ),
        "hier_slice_trade_prior_loss_mean": (
            hier_slice_trade_prior_loss_sum / max(1, n)
        ),
        "hier_slice_trade_accuracy_edge_loss_mean": (
            hier_slice_trade_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_flat_logit_margin_loss_mean": (
            hier_flat_logit_margin_loss_sum / max(1, n)
        ),
        "hier_slice_flat_logit_margin_loss_mean": (
            hier_slice_flat_logit_margin_loss_sum / max(1, n)
        ),
        "hier_side_loss_mean": (hier_side_loss_sum / max(1, n)),
        "hier_slice_side_ce_loss_mean": (hier_slice_side_ce_loss_sum / max(1, n)),
        "hier_slice_side_margin_loss_mean": (
            hier_slice_side_margin_loss_sum / max(1, n)
        ),
        "hier_slice_side_accuracy_edge_loss_mean": (
            hier_slice_side_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_side_global_prior_loss_mean": (
            hier_side_global_prior_loss_sum / max(1, n)
        ),
        "hier_slice_side_prior_loss_mean": (
            hier_slice_side_prior_loss_sum / max(1, n)
        ),
        "hier_utility_loss_mean": (hier_utility_loss_sum / max(1, n)),
        "hier_bad_path_loss_mean": (hier_side_bad_path_loss_sum / max(1, n)),
        "hier_mae_loss_mean": (hier_side_mae_loss_sum / max(1, n)),
        "hier_side_validity_loss_mean": (
            hier_side_validity_loss_sum / max(1, n)
        ),
        "hier_long_valid_target_rate": (
            hier_long_valid_target_rate_sum / max(1, n)
        ),
        "hier_short_valid_target_rate": (
            hier_short_valid_target_rate_sum / max(1, n)
        ),
        "hier_long_valid_prob_mean": (hier_long_valid_prob_sum / max(1, n)),
        "hier_short_valid_prob_mean": (hier_short_valid_prob_sum / max(1, n)),
        "hier_long_bad_target_rate": (hier_long_bad_target_rate_sum / max(1, n)),
        "hier_short_bad_target_rate": (
            hier_short_bad_target_rate_sum / max(1, n)
        ),
        "hier_countertrend_long_trap_rate": (
            hier_countertrend_long_trap_rate_sum / max(1, n)
        ),
        "hier_countertrend_short_trap_rate": (
            hier_countertrend_short_trap_rate_sum / max(1, n)
        ),
        "hier_side_rows": int(hier_side_rows_sum),
        "hier_side_acc": (
            hier_side_correct_sum / hier_side_rows_sum
            if hier_side_rows_sum > 0
            else 0.0
        ),
        "trendline_rail_loss_mean": (trendline_rail_loss_sum / max(1, n)),
        "trendline_rail_rows": int(trendline_rail_rows_sum),
        "trendline_rising_rows": int(trendline_rising_rows_sum),
        "trendline_falling_rows": int(trendline_falling_rows_sum),
    }
    stats.update(_finalize_cooperation_gate_epoch(cooperation_gate_epoch))
    stats.update(_direction_ckpt_balance_stats(targets_np, preds_np, acc))
    stats.update(
        _direction_hierarchy_output_stats(
            targets_np,
            hierarchy_trade_prob_np,
            hierarchy_side_pred_np,
            hierarchy_side_long_prob_np,
        )
    )
    slice_stats = _direction_slice_balance_stats(
        targets_np,
        preds_np,
        ctx_cat_np,
        trade_prob_np=hierarchy_trade_prob_np,
        side_pred_np=hierarchy_side_pred_np,
        side_long_prob_np=hierarchy_side_long_prob_np,
    )
    stats.update(slice_stats)
    stats["direction_slice_ckpt_score"] = _direction_slice_ckpt_score(
        float(stats.get("direction_ckpt_score", acc)),
        slice_stats,
    )
    _diag_metrics, _diag_failures = _aux_head_diagnostics(
        _diag_pred,
        _diag_lbl,
        _diag_real,
        _diag_mask,
    )
    stats.update(_diag_metrics)
    stats["aux_head_health_ok"] = not _diag_failures
    stats["aux_head_health_failures"] = list(_diag_failures)
    if _diag_failures:
        log.error(
            "[ENTRY_AUX_HEAD_HEALTH_CHECKPOINT_BLOCKED] %s",
            "; ".join(_diag_failures),
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
    # AUC is intentionally disabled for this 3-class path (previously hardcoded 0.0)
    return total / max(1, n), float("nan"), acc, short_pred_long_rate, stats


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def run_train(
    train_parquet: Path,
    train_manifest_path: Path,
    val_parquet: Path,
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
    deterministic: bool = True,
    multi_tf_seq_len: int = 96,
    multi_tf_scale: float = 0.5,
    subsample_rows: int = 0,
    specialist_num_layers: int = 1,
    specialist_fusion_scale: float = 0.25,
    # V2 fast-train extras
    per_tf_seq_len_h4: int = 0,
    per_tf_seq_len_d1: int = 0,
    grad_accum_steps: int = 0,
    tf_input_scale_init_m5: float = 1.0,
    tf_input_scale_init_m15: float = 1.0,
    tf_input_scale_init_h1: float = 0.7,
    tf_input_scale_init_h4: float = 0.5,
    tf_input_scale_init_d1: float = 0.3,
    run_id: str = "",
    dataset_run_id: str = "",
    profile: str = "",
) -> None:
    _guard_no_rl()
    if profile not in ("smoke", "candidate"):
        raise RuntimeError(f"[ENTRY_TRAIN_PROFILE_INVALID] {profile!r}")

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
            "GX1_V10_MULTI_TF_V2_CACHE_DIR must name the exact absolute "
            "verified V2 cache used by training and normalization"
        )
    if int(grad_accum_steps) < 1:
        raise RuntimeError(
            "[ENTRY_GRAD_ACCUM_STEPS_INVALID] "
            f"observed={int(grad_accum_steps)} expected>=1"
        )

    log.info(
        f"[TRAIN] seed={seed} device={device} batch_size={batch_size} epochs={epochs} lr={lr} "
        f"signal_dim={MODEL_NATIVE_SIGNAL_DIM} ctx_cont={MODEL_NATIVE_CTX_CONT_DIM} "
        f"ctx_cat={MODEL_NATIVE_CTX_CAT_DIM} early_stop_patience={early_stopping_patience} "
        f"early_stop_min_delta={early_stopping_min_delta}"
    )

    _set_deterministic(seed, device, deterministic=deterministic)

    _log_label_distribution(train_parquet, split="train")
    _log_label_distribution(val_parquet, split="val")

    # V12.2: pre-build multi-TF features BEFORE loading train_parquet so peak
    # memory = max(train_parquet, M5_prebuilt) instead of their sum. Without
    # this, OOM on 15GB hosts during Dataset construction (1.5GB parquet ×
    # pandas overhead + 1.5GB M5 prebuilt × pandas overhead > 15GB).
    _prebuild_multi_tf_v2_features_once(m5_prebuilt_path)

    # Build exact per-TF sequence lengths.
    _per_tf_lens: Dict[str, int] = {}
    # B10 tapered-MTF (GX1_MTF_TAPERED=1, default OFF): coarser TFs reach further with FEWER
    # bars — m15=64 (drops the ~8h M5 overlap), d1=252 (~1yr regime memory, matches the
    # D1_atr_percentile_252 lookback). Explicit --per-tf-seq-len-* args still win. The model's
    # m15/d1_seq_len below mirror these so the bundle metadata records them → live reads the
    # same lens (train==serve). m5/h1/h4 stay at the global default.
    _tapered = os.environ.get("GX1_MTF_TAPERED", "0") == "1"
    if int(per_tf_seq_len_h4) > 0:
        _per_tf_lens["H4"] = int(per_tf_seq_len_h4)
    _d1_eff = int(per_tf_seq_len_d1) if int(per_tf_seq_len_d1) > 0 else (252 if _tapered else 0)
    if _d1_eff > 0:
        _per_tf_lens["D1"] = _d1_eff
    if _tapered:
        _per_tf_lens["M15"] = 64
        log.info("[GX1_MTF_TAPERED] per-TF seq-lens: %s (m5/h1=default %d)", _per_tf_lens, int(multi_tf_seq_len))
    train_ds = EntryV10CtxDataset(
        train_parquet,
        seq_len=seq_len,
        m5_prebuilt_path=m5_prebuilt_path,
        multi_tf_seq_len=multi_tf_seq_len,
        per_tf_seq_lens=_per_tf_lens,
        multi_tf_closed_bar=True,
    )
    normalization_per_tf_seq_lens = {
        "M5": int(multi_tf_seq_len),
        "M15": 64 if _tapered else int(multi_tf_seq_len),
        "H1": int(multi_tf_seq_len),
        "H4": (
            int(per_tf_seq_len_h4)
            if int(per_tf_seq_len_h4) > 0
            else int(multi_tf_seq_len)
        ),
        "D1": (
            int(per_tf_seq_len_d1)
            if int(per_tf_seq_len_d1) > 0
            else (252 if _tapered else int(multi_tf_seq_len))
        ),
    }
    normalization_fit = fit_entry_v10_train_input_normalization(
        train_seq=train_ds._np_seq,
        train_snap=train_ds._np_snap,
        train_ctx_cont=train_ds._np_ctx_cont,
        train_ctx_cat=train_ds._np_ctx_cat,
        train_times=train_ds.df["time"],
        ordered_signal_names=list(train_ds.signal_names),
        per_tf_seq_lens=normalization_per_tf_seq_lens,
        artifacts=TrainNormalizationArtifacts(
            dataset_run_id=str(dataset_run_id),
            train_parquet_path=Path(train_parquet),
            train_manifest_path=Path(train_manifest_path),
            m5_prebuilt_path=Path(m5_prebuilt_path),
            mtf_cache_dir=mtf_cache_dir,
        ),
    )
    input_normalization = normalization_fit["normalization_contract"]
    input_normalization_fit_population_proof = normalization_fit[
        "fit_population_proof"
    ]
    log.info(
        "[ENTRY_INPUT_NORMALIZATION_FIT] contract_sha256=%s "
        "full_train_rows=%d val_rows=0 test_rows=0",
        input_normalization["contract_sha256"],
        int(
            input_normalization_fit_population_proof[
                "train_decision_row_count"
            ]
        ),
    )
    # V12.2 sweep mode: stratified subsample on training set ONLY (val untouched).
    if subsample_rows > 0 and subsample_rows < len(train_ds):
        rng = np.random.default_rng(seed=seed)
        ys = train_ds.df["y_direction"].to_numpy()
        sampled_idx: List[int] = []
        for cls in (0, 1, 2):
            mask = (ys == cls)
            class_n = int(mask.sum())
            target = int(round(subsample_rows * class_n / len(ys)))
            class_idx = np.where(mask)[0]
            keep = rng.choice(class_idx, size=min(target, class_n), replace=False)
            sampled_idx.extend(keep.tolist())
        sampled_idx = sorted(sampled_idx)
        train_ds.indices = np.array(sampled_idx, dtype=np.int64)
        log.info(
            f"[SUBSAMPLE] stratified subsample: {len(sampled_idx):,}/{len(ys):,} rows  "
            f"(class counts: {[int((ys[sampled_idx]==c).sum()) for c in (0,1,2)]})"
        )
    val_ds = EntryV10CtxDataset(
        val_parquet,
        seq_len=seq_len,
        m5_prebuilt_path=m5_prebuilt_path,
        multi_tf_seq_len=multi_tf_seq_len,
        per_tf_seq_lens=_per_tf_lens,
        multi_tf_closed_bar=True,
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
    if not bool(ENTRY_SYMMETRIC_NEGATIVES):
        raise RuntimeError("[ENTRY_TRAIN_SYMMETRIC_NEGATIVES_REQUIRED]")
    active_model_native_loss_weights = _current_model_native_active_loss_weights()
    active_loss_weight_failures = _model_native_active_loss_weight_failures(
        active_model_native_loss_weights
    )
    if active_loss_weight_failures:
        raise RuntimeError(
            "[ENTRY_TRAIN_MODEL_NATIVE_ACTIVE_LOSS_WEIGHT_INVALID] "
            + "; ".join(active_loss_weight_failures)
        )
    model_native_training_objective = training_objective_contract_metadata(
        active_model_native_loss_weights
    )
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
        contract_failures.extend(_xau_direction_repair_target_failures(split_name, ds_obj.df))
    if contract_failures:
        raise RuntimeError(
            "[ENTRY_XAU_DIRECTION_REPAIR_CONTRACT_INVALID] "
            + "; ".join(contract_failures)
        )
    required_hier_cols = [
            "y_trade",
            "y_side",
            "y_side_mask",
            "y_long_path_utility_bps",
            "y_short_path_utility_bps",
            "y_long_bad_path",
            "y_short_bad_path",
            "y_long_expected_mae_bps",
            "y_short_expected_mae_bps",
            "y_rising_channel_support_touch",
            "y_falling_channel_resistance_touch",
            "y_support_retest_continuation",
            "y_resistance_retest_continuation",
            "y_countertrend_short_trap",
            "y_countertrend_long_trap",
            "y_mtf_conflict_m5_vs_higher_side",
            "y_long_high_mae_low_mfe_early_failure",
            "y_short_high_mae_low_mfe_early_failure",
    ]
    for split_name, ds_obj in (("train", train_ds), ("val", val_ds)):
        missing = [c for c in required_hier_cols if c not in ds_obj.df.columns]
        if missing:
            raise RuntimeError(
                f"[ENTRY_HIER_LABEL_CONTRACT_MISSING] split={split_name} "
                f"missing={missing}. Rebuild the XAU path-utility dataset; "
                "hierarchical repair heads must not train on fallback labels."
            )
    train_bad_path_rate = float(train_ds.df["y_bad_path"].astype(float).mean())
    val_bad_path_rate = float(val_ds.df["y_bad_path"].astype(float).mean())
    train_tradable_rate = float(train_ds.df["y_tradable"].astype(float).mean())
    val_tradable_rate = float(val_ds.df["y_tradable"].astype(float).mean())
    train_trade_rate = float(train_ds.df["y_trade"].astype(float).mean())
    val_trade_rate = float(val_ds.df["y_trade"].astype(float).mean())
    train_long_bad_path_rate = float(train_ds.df["y_long_bad_path"].astype(float).mean())
    train_short_bad_path_rate = float(train_ds.df["y_short_bad_path"].astype(float).mean())
    _train_side_bad_arr = pd.concat(
        [train_ds.df["y_long_bad_path"].astype(float), train_ds.df["y_short_bad_path"].astype(float)],
        ignore_index=True,
    )
    train_side_bad_path_rate = float(_train_side_bad_arr.mean())
    _val_side_bad_arr = pd.concat(
        [val_ds.df["y_long_bad_path"].astype(float), val_ds.df["y_short_bad_path"].astype(float)],
        ignore_index=True,
    )
    val_side_bad_path_rate = float(_val_side_bad_arr.mean())
    train_hard_neg_long_rate = float(train_ds.df["y_hard_negative_long"].astype(float).mean())
    val_hard_neg_long_rate = float(val_ds.df["y_hard_negative_long"].astype(float).mean())
    train_dead_neg_long_rate = float(train_ds.df["y_dead_negative_long"].astype(float).mean())
    val_dead_neg_long_rate = float(val_ds.df["y_dead_negative_long"].astype(float).mean())
    train_teaser_neg_long_rate = float(train_ds.df["y_teaser_negative_long"].astype(float).mean())
    val_teaser_neg_long_rate = float(val_ds.df["y_teaser_negative_long"].astype(float).mean())
    train_clean_edge_rate = _active_aux_target_rate_from_frame(
        train_ds.df,
        split_name="train",
        target_name="clean_edge",
        long_column="y_clean_edge_long",
        bidir_column="y_clean_edge_bidir",
    )
    val_clean_edge_rate = _active_aux_target_rate_from_frame(
        val_ds.df,
        split_name="val",
        target_name="clean_edge",
        long_column="y_clean_edge_long",
        bidir_column="y_clean_edge_bidir",
    )
    train_survival_rate = _active_aux_target_rate_from_frame(
        train_ds.df,
        split_name="train",
        target_name="survival",
        long_column="y_survival_long",
        bidir_column="y_survival_bidir",
    )
    val_survival_rate = _active_aux_target_rate_from_frame(
        val_ds.df,
        split_name="val",
        target_name="survival",
        long_column="y_survival_long",
        bidir_column="y_survival_bidir",
    )
    train_selector_long_mask_rate = float(train_ds.df["y_selector_long_mask"].astype(float).mean())
    val_selector_long_mask_rate = float(val_ds.df["y_selector_long_mask"].astype(float).mean())
    train_label_counts = train_ds.df["y_direction"].value_counts().to_dict()
    train_long_rate = float(train_label_counts.get(0, 0) / max(len(train_ds.df), 1))
    train_short_rate = float(train_label_counts.get(1, 0) / max(len(train_ds.df), 1))
    train_flat_rate = float(train_label_counts.get(2, 0) / max(len(train_ds.df), 1))
    if train_bad_path_rate > 0.0:
        raw_bad_path_pos_weight = (1.0 - train_bad_path_rate) / max(train_bad_path_rate, 1e-9)
    else:
        raw_bad_path_pos_weight = 1.0
    if train_tradable_rate > 0.0:
        raw_tradable_pos_weight = (1.0 - train_tradable_rate) / max(train_tradable_rate, 1e-9)
    else:
        raw_tradable_pos_weight = 1.0
    if train_trade_rate > 0.0:
        raw_hier_trade_pos_weight = (1.0 - train_trade_rate) / max(train_trade_rate, 1e-9)
    else:
        raw_hier_trade_pos_weight = 1.0
    if train_long_bad_path_rate > 0.0:
        raw_hier_long_bad_path_pos_weight = (1.0 - train_long_bad_path_rate) / max(train_long_bad_path_rate, 1e-9)
    else:
        raw_hier_long_bad_path_pos_weight = 1.0
    if train_short_bad_path_rate > 0.0:
        raw_hier_short_bad_path_pos_weight = (1.0 - train_short_bad_path_rate) / max(train_short_bad_path_rate, 1e-9)
    else:
        raw_hier_short_bad_path_pos_weight = 1.0
    bad_path_pos_weight = _bounded_pos_weight(
        raw_bad_path_pos_weight,
        ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP,
    )
    tradable_pos_weight = _bounded_pos_weight(
        raw_tradable_pos_weight,
        ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP,
    )
    hier_trade_pos_weight = _bounded_pos_weight(
        raw_hier_trade_pos_weight,
        ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP,
        allow_below_one=True,
    )
    hier_bad_path_pos_weight = [
        float(min(float(ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP), max(1.0, raw_hier_long_bad_path_pos_weight))),
        float(min(float(ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP), max(1.0, raw_hier_short_bad_path_pos_weight))),
    ]
    raw_clean_edge_pos_weight, clean_edge_pos_weight = _positive_class_weight_from_rate(
        train_clean_edge_rate,
        ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP,
    )
    raw_survival_pos_weight, survival_pos_weight = _positive_class_weight_from_rate(
        train_survival_rate,
        ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP,
    )
    # 2026-05-26: SYMMETRIC + sqrt-softened directional class weights. The old
    # per-side inverse-frequency weights made the model over-predict SHORT (short
    # rarer → higher weight: 4.67 vs long 4.22) AND over-predict direction vs flat
    # (4.5 vs 1.0). With the re-balanced labels (~18/18/63) that aggressive scheme
    # over-corrects. Fix: ONE shared directional weight from the COMBINED directional
    # rate (removes long/short asymmetry), sqrt-softened (shrinks the directional-vs-
    # flat gap). One shared cap clamps both directions. flat stays 1.0.
    _dir_rate = 0.5 * (float(train_long_rate) + float(train_short_rate))
    _raw_dir = ((1.0 - _dir_rate) / max(_dir_rate, 1e-9)) if _dir_rate > 0.0 else 1.0
    _dir_w = float(np.sqrt(max(_raw_dir, 1.0)))  # sqrt-soften; >=1
    raw_long_class_weight = _raw_dir   # kept for the proof log
    raw_short_class_weight = _raw_dir
    long_class_weight = float(
        min(float(ENTRY_DIRECTION_CLASS_WEIGHT_CAP), max(1.0, _dir_w))
    )
    short_class_weight = long_class_weight
    flat_class_weight = float(max(float(ENTRY_FLAT_CLASS_WEIGHT_FLOOR), 1.0))
    log.info(
        "[ENTRY_BAD_PATH_BALANCE_PROOF] train_rate=%.6f val_rate=%.6f raw_pos_weight=%.6f capped_pos_weight=%.6f cap=%.3f",
        train_bad_path_rate,
        val_bad_path_rate,
        raw_bad_path_pos_weight,
        bad_path_pos_weight,
        float(ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP),
    )
    log.info(
        "[ENTRY_TRADABLE_BALANCE_PROOF] train_rate=%.6f val_rate=%.6f raw_pos_weight=%.6f capped_pos_weight=%.6f cap=%.3f",
        train_tradable_rate,
        val_tradable_rate,
        raw_tradable_pos_weight,
        tradable_pos_weight,
        float(ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP),
    )
    log.info(
        "[ENTRY_HIER_BALANCE_PROOF] train_trade_rate=%.6f val_trade_rate=%.6f "
        "raw_trade_pos_weight=%.6f trade_pos_weight=%.6f "
        "train_side_bad_path_rate=%.6f val_side_bad_path_rate=%.6f "
        "train_long_bad_path_rate=%.6f train_short_bad_path_rate=%.6f "
        "side_bad_path_pos_weight_long=%.6f side_bad_path_pos_weight_short=%.6f",
        train_trade_rate,
        val_trade_rate,
        raw_hier_trade_pos_weight,
        hier_trade_pos_weight,
        train_side_bad_path_rate,
        val_side_bad_path_rate,
        train_long_bad_path_rate,
        train_short_bad_path_rate,
        hier_bad_path_pos_weight[0],
        hier_bad_path_pos_weight[1],
    )
    log.info(
        "[ENTRY_DEAD_LONG_RATE_PROOF] train_rate=%.6f val_rate=%.6f ce_multiplier=%.3f prob_penalty=%.3f",
        train_dead_neg_long_rate,
        val_dead_neg_long_rate,
        float(ENTRY_DEAD_LONG_CE_MULTIPLIER),
        float(ENTRY_DEAD_LONG_PROB_PENALTY),
    )
    log.info(
        "[ENTRY_TEASER_LONG_RATE_PROOF] train_rate=%.6f val_rate=%.6f ce_multiplier=%.3f prob_penalty=%.3f",
        train_teaser_neg_long_rate,
        val_teaser_neg_long_rate,
        float(ENTRY_TEASER_LONG_CE_MULTIPLIER),
        float(ENTRY_TEASER_LONG_PROB_PENALTY),
    )
    log.info(
        "[ENTRY_HARD_NEG_LONG_RATE_PROOF] train_rate=%.6f val_rate=%.6f ce_multiplier=%.3f prob_penalty=%.3f",
        train_hard_neg_long_rate,
        val_hard_neg_long_rate,
        float(ENTRY_HARD_NEG_LONG_CE_MULTIPLIER),
        float(ENTRY_HARD_NEG_LONG_PROB_PENALTY),
    )
    log.info(
        "[ENTRY_CLEAN_EDGE_RATE_PROOF] target_mode=%s selector_mode=%s "
        "train_rate=%.6f val_rate=%.6f raw_pos_weight=%.6f "
        "capped_pos_weight=%.6f cap=%.3f",
        "bidir" if ENTRY_SYMMETRIC_NEGATIVES else "long",
        "long_short_union" if ENTRY_SYMMETRIC_NEGATIVES else "long_only",
        train_clean_edge_rate,
        val_clean_edge_rate,
        raw_clean_edge_pos_weight,
        clean_edge_pos_weight,
        float(ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP),
    )
    log.info(
        "[ENTRY_SURVIVAL_RATE_PROOF] target_mode=%s selector_mode=%s "
        "train_rate=%.6f val_rate=%.6f raw_pos_weight=%.6f "
        "capped_pos_weight=%.6f cap=%.3f",
        "bidir" if ENTRY_SYMMETRIC_NEGATIVES else "long",
        "long_short_union" if ENTRY_SYMMETRIC_NEGATIVES else "long_only",
        train_survival_rate,
        val_survival_rate,
        raw_survival_pos_weight,
        survival_pos_weight,
        float(ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP),
    )
    log.info(
        "[ENTRY_SELECTOR_MASK_RATE_PROOF] train=%.6f val=%.6f",
        train_selector_long_mask_rate,
        val_selector_long_mask_rate,
    )
    log.info(
        "[ENTRY_DIRECTION_BALANCE_PROOF] train_long_rate=%.6f train_short_rate=%.6f train_flat_rate=%.6f "
        "raw_long_class_weight=%.6f raw_short_class_weight=%.6f long_class_weight=%.6f short_class_weight=%.6f flat_class_weight=%.6f",
        train_long_rate,
        train_short_rate,
        train_flat_rate,
        raw_long_class_weight,
        raw_short_class_weight,
        long_class_weight,
        short_class_weight,
        flat_class_weight,
    )

    use_cuda = device.type == "cuda"
    if use_cuda and num_workers < 0:
        num_workers = max(2, min(8, (os.cpu_count() or 4)))
    # Centralized loader tuning — bumps prefetch_factor 2→4 when GX1_FAST_TRAIN=1.
    from gx1.utils.fast_train import loader_kwargs as _loader_kwargs
    _dl_kwargs = _loader_kwargs(num_workers, use_cuda=use_cuda)
    if _running_in_wsl():
        _dl_kwargs["pin_memory"] = False
    pin_memory = _dl_kwargs["pin_memory"]
    persistent_workers = _dl_kwargs["persistent_workers"]
    prefetch_factor = _dl_kwargs["prefetch_factor"]
    log.info(
        "[DATALOADER_CONFIG] num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%s",
        num_workers, pin_memory, persistent_workers, str(prefetch_factor),
    )

    train_sampler: Optional[Sampler[int]] = None
    if bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER):
        sampler_labels, sampler_ctx_cat = _direction_slice_sampler_arrays(train_ds)
        train_sampler = _DirectionSliceBalancedSampler(
            labels=sampler_labels,
            ctx_cat=sampler_ctx_cat,
            ctx_cat_indices=_direction_slice_ctx_cat_indices(int(sampler_ctx_cat.shape[1])),
            batch_size=int(batch_size),
            min_rows=int(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS),
            min_label_rate=float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE),
            seed=int(seed),
        )
        log.info(
            "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER] enabled=1 audited_train_slices=%d "
            "batch_size=%d min_rows=%d min_label_rate=%.3f num_samples=%d "
            "coverage_preserving=1 replacement=0 padding=0",
            int(getattr(train_sampler, "audited_slice_count", 0)),
            int(batch_size),
            int(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS),
            float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE),
            int(len(train_sampler)),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    # V12.2: detect multi-TF feature count from dataset (avoid hardcoding 19)
    from gx1.features.htf_features import (
        HTF_V2_MATRIX_CONTRACT,
        MULTI_TF_FEATURE_NAMES_SHA256_V2,
        MULTI_TF_PER_BAR_FEATURES_V2,
    )

    _mtf_feat_count = int(train_ds._multi_tf_feature_count)
    # Exact mode always includes the causal M5 branch and all four higher TFs.
    _mtf_v2 = bool(getattr(train_ds, "_multi_tf_v2", False))
    if not _mtf_v2 or _mtf_feat_count <= 0:
        raise RuntimeError(
            "[MULTI_TF_EXACT_ARCHITECTURE_REQUIRED] expected causal M5/M15/H1/H4/D1 V2"
        )
    # Per-TF seq_len overrides (default 0 → fall back to global multi_tf_seq_len).
    _h4_len = int(per_tf_seq_len_h4) if int(per_tf_seq_len_h4) > 0 else int(multi_tf_seq_len)
    _d1_len = int(per_tf_seq_len_d1) if int(per_tf_seq_len_d1) > 0 else (252 if _tapered else int(multi_tf_seq_len))
    _m15_len = 64 if _tapered else int(multi_tf_seq_len)  # B10 tapered-MTF — mirrors _per_tf_lens above (train==serve via bundle meta)
    if _h4_len != multi_tf_seq_len or _d1_len != multi_tf_seq_len or _m15_len != multi_tf_seq_len:
        log.info("[PER_TF_SEQ_LEN] M15=%d H4=%d D1=%d (global=%d)", _m15_len, _h4_len, _d1_len, int(multi_tf_seq_len))
    specialist_indices, specialist_meta = _load_specialist_fusion_contract(
        specialist_audit_json,
        expected_signal_dim=seq_input_dim,
        ordered_signal_names=list(train_ds.signal_names),
        contract_mode=specialist_contract_mode,
    )
    log.info("[SPECIALIST_FUSION] exact groups=%s", sorted(specialist_indices))
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        m15_seq_dim=_mtf_feat_count,
        h1_seq_dim=_mtf_feat_count,
        h4_seq_dim=_mtf_feat_count,
        d1_seq_dim=_mtf_feat_count,
        m15_seq_len=_m15_len,
        h1_seq_len=multi_tf_seq_len,
        h4_seq_len=_h4_len,
        d1_seq_len=_d1_len,
        m5_seq_dim=_mtf_feat_count,
        m5_seq_len=multi_tf_seq_len,
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
        cross_family_fusion_scale=float(specialist_fusion_scale),
        tf_input_scale_init_m5=tf_input_scale_init_m5,
        tf_input_scale_init_m15=tf_input_scale_init_m15,
        tf_input_scale_init_h1=tf_input_scale_init_h1,
        tf_input_scale_init_h4=tf_input_scale_init_h4,
        tf_input_scale_init_d1=tf_input_scale_init_d1,
        input_normalization=input_normalization,
    ).to(device)
    evidence_fusion_initial_state = _capture_evidence_fusion_initial_state(model)
    log.info(
        "[TF_INPUT_SCALE] mandatory learnable per-TF inits: M5=%.2f M15=%.2f H1=%.2f H4=%.2f D1=%.2f",
        tf_input_scale_init_m5, tf_input_scale_init_m15,
        tf_input_scale_init_h1, tf_input_scale_init_h4, tf_input_scale_init_d1,
    )
    log.info(
        "[ENTRY_EXACT_HEADS] hierarchy=true side_validity=true trendline_rail=true "
        "tf_agreement=true path_variance=true position_size=true",
    )
    log.info(
        "[MULTI_TF_PROOF] enabled=True TFs=M5+M15+H1+H4+D1 (V2) "
        "per_tf_dim=%d per_tf_len=%d total_extra_params≈%dK",
        _mtf_feat_count, multi_tf_seq_len,
        (sum(p.numel() for p in model.parameters()) - 691977) // 1000,
    )
    # GX1_FAST_TRAIN=1 wraps model with torch.compile (best-effort).
    try:
        from gx1.utils.fast_train import maybe_compile, compile_enabled
        if compile_enabled():
            log.info("[FAST_TRAIN] torch.compile=on (mode=reduce-overhead)")
        model = maybe_compile(model)
    except Exception as _e:
        log.warning("[FAST_TRAIN] compile wrap failed: %r", _e)
    try:
        head_out = int(getattr(model.head_direction, "out_features", -1))
    except Exception:
        head_out = -1
    log.info("[ENTRY_3CLASS_PROOF] head_direction_out=%s", head_out)
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
        preflight_out = _autocast_forward(
            model,
            device,
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
    dip_forecast_loss(
        preflight_out,
        preflight_batch,
        device,
        bps_scale=ENTRY_AUX_MFE_SCALE_BPS,
    )
    offline_rl_aux_loss(
        preflight_out,
        preflight_batch,
        device,
    )
    del preflight_out, preflight_batch

    _require_nonneg("ENTRY_COST_SENSITIVE_SCALE", ENTRY_COST_SENSITIVE_SCALE)
    _require_nonneg("ENTRY_COST_LONG_TO_SHORT", ENTRY_COST_LONG_TO_SHORT)
    _require_nonneg("ENTRY_COST_LONG_TO_FLAT", ENTRY_COST_LONG_TO_FLAT)
    _require_nonneg("ENTRY_COST_SHORT_TO_LONG", ENTRY_COST_SHORT_TO_LONG)
    _require_nonneg("ENTRY_COST_SHORT_TO_FLAT", ENTRY_COST_SHORT_TO_FLAT)
    _require_nonneg("ENTRY_COST_FLAT_TO_LONG", ENTRY_COST_FLAT_TO_LONG)
    _require_nonneg("ENTRY_COST_FLAT_TO_SHORT", ENTRY_COST_FLAT_TO_SHORT)
    _require_nonneg("ENTRY_PRED_BALANCE_ALPHA", ENTRY_PRED_BALANCE_ALPHA)
    _require_nonneg("ENTRY_TAIL_DIRECTION_CE_WEIGHT", ENTRY_TAIL_DIRECTION_CE_WEIGHT)
    _require_nonneg("ENTRY_HIER_TRADE_WEIGHT", ENTRY_HIER_TRADE_WEIGHT)
    _require_nonneg("ENTRY_HIER_SIDE_WEIGHT", ENTRY_HIER_SIDE_WEIGHT)
    _require_nonneg("ENTRY_HIER_UTILITY_WEIGHT", ENTRY_HIER_UTILITY_WEIGHT)
    _require_nonneg("ENTRY_HIER_BAD_PATH_WEIGHT", ENTRY_HIER_BAD_PATH_WEIGHT)
    _require_nonneg("ENTRY_HIER_MAE_WEIGHT", ENTRY_HIER_MAE_WEIGHT)
    _require_nonneg("ENTRY_HIER_SIDE_VALIDITY_WEIGHT", ENTRY_HIER_SIDE_VALIDITY_WEIGHT)
    _require_nonneg("ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS", ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS)
    _require_nonneg("ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP", ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_CE_WEIGHT", ENTRY_HIER_SLICE_SIDE_CE_WEIGHT)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT", ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN", ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT", ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN", ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE", ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_MIN_ROWS", ENTRY_HIER_SLICE_SIDE_MIN_ROWS)
    _require_nonneg("ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT", ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT)
    _require_nonneg("ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE", ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE)
    _require_nonneg(
        "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE",
        ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT", ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT)
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE", ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE)
    _require_nonneg(
        "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE",
        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS", ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_AUX_WEIGHT", ENTRY_TRENDLINE_RAIL_AUX_WEIGHT)
    _require_nonneg("ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT", ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT)
    _require_nonneg("ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL", ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL)
    _require_nonneg("ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE", ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE)
    _require_nonneg("ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT", ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION", ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION)
    _require_nonneg("ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR", ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR)
    _require_nonneg(
        "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE",
        ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE,
    )
    _require_nonneg("ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT", ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE", ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE)
    _require_nonneg(
        "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE",
        ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE,
    )
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT",
        ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT,
    )
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION",
        ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION,
    )
    _require_nonneg("ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR", ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR)
    _require_nonneg("ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE", ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE)
    _require_nonneg("ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT", ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR", ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR)
    _require_nonneg("ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE", ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE)
    _require_nonneg("ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT", ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT)
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE",
        ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS", ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS)
    _require_nonneg("ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT", ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_SLICE_TRUE_MARGIN", ENTRY_DIRECTION_SLICE_TRUE_MARGIN)
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE",
        ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS", ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS)
    _require_nonneg("ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT", ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN", ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN)
    _require_nonneg("ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT", ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN", ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN)
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE",
        ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS", ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS)
    _require_nonneg("ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT", ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE", ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE)
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE",
        ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS", ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS)
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS",
        ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS,
    )
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE",
        ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE,
    )
    _require_nonneg(
        "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS",
        ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS,
    )
    _require_nonneg("ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT", ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_VS_FLAT_MARGIN", ENTRY_DIRECTION_VS_FLAT_MARGIN)
    _require_nonneg("ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT", ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS", ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS)
    _require_nonneg("ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN", ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN)
    _require_nonneg("ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT", ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT)
    _require_nonneg(
        "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE",
        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE,
    )
    _require_nonneg(
        "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE",
        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT", ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT)
    _require_nonneg(
        "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE",
        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE,
    )
    _require_nonneg(
        "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE",
        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS", ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS)
    _require_nonneg(
        "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT",
        ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT,
    )
    _require_nonneg(
        "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN",
        ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN,
    )
    _require_nonneg("ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT", ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_HIER_FLAT_LOGIT_MARGIN", ENTRY_HIER_FLAT_LOGIT_MARGIN)
    _require_nonneg(
        "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE",
        ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT", ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN", ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN)
    _require_nonneg(
        "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE",
        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE,
    )
    _require_nonneg(
        "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS",
        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS,
    )
    _require_nonneg(
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT",
        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT,
    )
    _require_nonneg(
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS",
        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS,
    )
    _require_nonneg(
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN",
        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN,
    )
    _require_nonneg("ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT", ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT)
    _require_nonneg(
        "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE",
        ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE,
    )
    _require_nonneg("ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS", ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS)
    _require_nonneg(
        "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION",
        ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION,
    )
    _require_nonneg("ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR", ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR)
    _require_nonneg("ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN", ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN)
    if ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL > 1.0:
        raise RuntimeError(
            "[ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL_INVALID] "
            f"ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL={ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL:.6f} expected <=1.0"
        )
    if ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE_INVALID] "
            f"ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE={ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION_INVALID] "
            f"ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION={ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR_INVALID] "
            f"ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR={ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE <= 0.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE_INVALID] "
            "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE="
            f"{ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE:.6f} expected >0.0"
        )
    if ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE_INVALID] "
            "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE="
            f"{ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE="
            f"{ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE_INVALID] "
            "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE="
            f"{ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE="
            f"{ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE_INVALID] "
            "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE="
            f"{ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE="
            f"{ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS={ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS} expected >=2"
        )
    if ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN_INVALID] "
            "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN="
            f"{ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN:.6f} expected <=1.0"
        )
    if ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE="
            f"{ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE="
            f"{ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS={ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS} expected >=2"
        )
    if ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE_INVALID] "
            "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE="
            f"{ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS={ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS} expected >=2"
        )
    if ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION_INVALID] "
            "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION="
            f"{ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR_INVALID] "
            "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR="
            f"{ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION_INVALID] "
            f"ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION={ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION:.6f} "
            "expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR_INVALID] "
            f"ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR={ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR:.6f} "
            "expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE={ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR_INVALID] "
            f"ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR={ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE={ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE:.6f} "
            "expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE_INVALID] "
            "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE="
            f"{ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS={ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS} expected >=2"
        )
    if ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE_INVALID] "
            "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE="
            f"{ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS={ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS} expected >=2"
        )
    if ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE_INVALID] "
            "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE="
            f"{ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN_INVALID] "
            "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN="
            f"{ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS={ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS} expected >=2"
        )
    if ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE_INVALID] "
            "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE="
            f"{ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE="
            f"{ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS={ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS} expected >=2"
        )
    if ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE={ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_SIDE_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_MIN_ROWS={ENTRY_HIER_SLICE_SIDE_MIN_ROWS} expected >=2"
        )
    if ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN_INVALID] "
            "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN="
            f"{ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE_INVALID] "
            "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE="
            f"{ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE="
            f"{ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE_INVALID] "
            "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE="
            f"{ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE="
            f"{ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS={ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS} expected >=2"
        )
    if ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION not in _DIRECTION_SLICE_LOSS_AGGREGATIONS:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION_INVALID] "
            f"ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION={ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION!r} "
            f"expected one of {sorted(_DIRECTION_SLICE_LOSS_AGGREGATIONS)}"
        )
    if ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS={ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS} expected >=2"
        )
    if ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS < 1:
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS_INVALID] "
            f"ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS={ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS} expected >=1"
        )
    if bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER) and int(batch_size) < int(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS):
        raise RuntimeError(
            "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_BATCH_TOO_SMALL] "
            f"batch_size={batch_size} min_rows={ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS}"
        )
    if ENTRY_HIER_SIDE_VALIDITY_WEIGHT <= 0.0:
        raise RuntimeError(
            "[ENTRY_SIDE_VALIDITY_HEAD_UNTRAINED] exact architecture requires "
            f"ENTRY_HIER_SIDE_VALIDITY_WEIGHT>0, got {ENTRY_HIER_SIDE_VALIDITY_WEIGHT:.6f}"
        )
    if ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE < 0.50 or ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE > 0.95:
        raise RuntimeError(
            "[ENTRY_TAIL_DIRECTION_QUANTILE_INVALID] "
            f"ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE={ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE:.6f} expected [0.50, 0.95]"
        )
    if ENTRY_TAIL_DIRECTION_MIN_BATCH < 2:
        raise RuntimeError(
            "[ENTRY_TAIL_DIRECTION_MIN_BATCH_INVALID] "
            f"ENTRY_TAIL_DIRECTION_MIN_BATCH={ENTRY_TAIL_DIRECTION_MIN_BATCH} expected >=2"
        )
    if ENTRY_PRED_BALANCE_TARGET not in ("label", "uniform"):
        raise RuntimeError(
            f"[ENTRY_BALANCE_TARGET_INVALID] ENTRY_PRED_BALANCE_TARGET={ENTRY_PRED_BALANCE_TARGET!r} "
            "expected 'label' or 'uniform'"
        )
    pred_balance_class_weights = torch.tensor(
        [float(value) for value in ENTRY_PRED_BALANCE_CLASS_WEIGHTS],
        device=device,
    )

    class_weights = torch.tensor(
        [float(long_class_weight), float(short_class_weight), float(flat_class_weight)],
        device=device,
    )
    criterion, cost_matrix = _build_cost_sensitive_criterion(
        device=device,
        class_weights=class_weights,
        cost_long_to_short=float(ENTRY_COST_LONG_TO_SHORT),
        cost_long_to_flat=float(ENTRY_COST_LONG_TO_FLAT),
        cost_short_to_long=float(ENTRY_COST_SHORT_TO_LONG),
        cost_short_to_flat=float(ENTRY_COST_SHORT_TO_FLAT),
        cost_flat_to_long=float(ENTRY_COST_FLAT_TO_LONG),
        cost_flat_to_short=float(ENTRY_COST_FLAT_TO_SHORT),
        cost_scale=float(ENTRY_COST_SENSITIVE_SCALE),
        enabled=bool(ENTRY_COST_SENSITIVE_ENABLED),
        balance_alpha=float(ENTRY_PRED_BALANCE_ALPHA),
        balance_target=str(ENTRY_PRED_BALANCE_TARGET),
        balance_class_weights=pred_balance_class_weights,
    )
    log.info(
        "[ENTRY_TRAIN_RECIPE] direction_ce_scale=%.3f tail_direction_w=%.3f tail_direction_q=%.3f tradable_w=%.3f path_w=%.3f mfe_w=%.3f tradable_pos_weight=%.3f bad_path_w=%.3f bad_path_pos_weight=%.3f clean_edge_w=%.3f clean_edge_pos_weight=%.3f survival_w=%.3f survival_pos_weight=%.3f rank_w=%.3f rank_margin=%.3f",
        float(ENTRY_DIRECTION_CE_SCALE),
        float(ENTRY_TAIL_DIRECTION_CE_WEIGHT),
        float(ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE),
        float(ENTRY_AUX_TRADABLE_WEIGHT),
        float(ENTRY_AUX_PATH_WEIGHT),
        float(ENTRY_AUX_MFE_WEIGHT),
        float(tradable_pos_weight),
        float(ENTRY_AUX_BAD_PATH_WEIGHT),
        float(bad_path_pos_weight),
        float(ENTRY_AUX_CLEAN_EDGE_WEIGHT),
        float(clean_edge_pos_weight),
        float(ENTRY_AUX_SURVIVAL_WEIGHT),
        float(survival_pos_weight),
        float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT),
        float(ENTRY_CLEAN_EDGE_RANKING_MARGIN),
    )
    log.info(
        "[ENTRY_TRAIN_SECONDARY_CONTROLS] cost_sensitive=%d cost_scale=%.3f pred_balance_alpha=%.3f pred_balance_class_weights=%s "
        "directional_class_weights=(long=%.3f,short=%.3f)",
        int(bool(ENTRY_COST_SENSITIVE_ENABLED)),
        float(ENTRY_COST_SENSITIVE_SCALE),
        float(ENTRY_PRED_BALANCE_ALPHA),
        ",".join(f"{float(value):.3f}" for value in ENTRY_PRED_BALANCE_CLASS_WEIGHTS),
        float(long_class_weight),
        float(short_class_weight),
    )
    log.info(
        "[ENTRY_SPECIALIST_GATE_RECIPE] entropy_w=%.3f balance_w=%.3f min_mean=%.3f",
        float(ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT),
        float(ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT),
        float(ENTRY_SPECIALIST_GATE_MIN_MEAN),
    )
    log.info(
        "[ENTRY_DIRECTION_MIN_PRED_RATE_RECIPE] weight=%.3f fraction=%.3f floor=%.3f temp=%.3f "
        "global_prior_w=%.3f global_prior_tol=%.3f global_prior_min_label_rate=%.3f "
        "slice_w=%.3f slice_fraction=%.3f slice_floor=%.3f slice_min_rows=%d slice_min_label_rate=%.3f "
        "slice_ctx_cat=%s slice_recall_w=%.3f slice_recall_floor=%.3f slice_recall_min_rows=%d "
        "slice_recall_min_label_rate=%.3f slice_balanced_ce_w=%.3f slice_balanced_ce_min_rows=%d "
        "slice_balanced_ce_min_label_rate=%.3f slice_true_margin_w=%.3f slice_true_margin=%.3f "
        "slice_true_margin_min_rows=%d slice_true_margin_min_label_rate=%.3f slice_acc_edge_w=%.3f "
        "slice_acc_edge_margin=%.3f slice_confusion_pair_w=%.3f slice_confusion_pair_margin=%.3f "
        "slice_acc_edge_min_rows=%d slice_acc_edge_min_label_rate=%.3f "
        "slice_prior_w=%.3f slice_prior_tol=%.3f slice_prior_min_rows=%d "
        "slice_prior_min_label_rate=%.3f slice_agg=%s "
        "slice_balanced_sampler=%d slice_balanced_sampler_min_rows=%d hard_red_stop_patience=%d "
        "hard_red_stop_min_epochs=%d flat_margin_w=%.3f flat_margin=%.3f "
        "utility_margin_w=%.3f utility_min_gap_bps=%.3f utility_logit_margin=%.3f "
        "side_utility_conviction_w=%.3f side_utility_conviction_min_gap_bps=%.3f "
        "side_utility_conviction_margin=%.3f "
        "utility_trade_conviction_w=%.3f utility_trade_conviction_min_gap_bps=%.3f "
        "utility_trade_conviction_min_utility_bps=%.3f utility_trade_conviction_max_bad_path=%.3f "
        "utility_trade_conviction_margin=%.3f "
        "utility_triad_ce_w=%.3f utility_triad_ce_min_gap_bps=%.3f "
        "utility_triad_ce_min_utility_bps=%.3f utility_triad_ce_max_bad_path=%.3f "
        "utility_triad_ce_class_weight_cap=%.3f "
        "flat_starvation_w=%.3f flat_starvation_min_label_rate=%.3f flat_starvation_min_rows=%d "
        "flat_starvation_fraction=%.3f flat_starvation_floor=%.3f flat_starvation_margin=%.3f",
        float(ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT),
        float(ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION),
        float(ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR),
        float(ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE),
        float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT),
        float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE),
        float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE),
        float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT),
        float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION),
        float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR),
        int(ENTRY_DIRECTION_SLICE_MIN_ROWS),
        float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE),
        str(ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES),
        float(ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT),
        float(ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR),
        int(ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS),
        float(ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE),
        float(ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT),
        int(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS),
        float(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE),
        float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT),
        float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN),
        int(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS),
        float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE),
        float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT),
        float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN),
        float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT),
        float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN),
        int(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS),
        float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE),
        float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT),
        float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE),
        int(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS),
        float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE),
        str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION),
        int(bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER)),
        int(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS),
        int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE),
        int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS),
        float(ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT),
        float(ENTRY_DIRECTION_VS_FLAT_MARGIN),
        float(ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT),
        float(ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS),
        float(ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN),
        float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT),
        float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS),
        float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN),
        float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT),
        float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS),
        float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS),
        float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH),
        float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN),
        float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT),
        float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS),
        float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS),
        float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH),
        float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP),
        float(ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT),
        float(ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE),
        int(ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS),
        float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION),
        float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR),
        float(ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN),
    )
    log.info(
        "[ENTRY_BAD_PATH_RANK_RECIPE] weight=%.3f margin=%.3f quantile=%.3f",
        float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT),
        float(ENTRY_BAD_PATH_QUALITY_RANK_MARGIN),
        float(ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE),
    )
    log.info(
        "[ENTRY_PATH_QUALITY_RANK_RECIPE] weight=%.3f margin=%.3f quantile=%.3f",
        float(ENTRY_PATH_QUALITY_RANK_WEIGHT),
        float(ENTRY_PATH_QUALITY_RANK_MARGIN),
        float(ENTRY_PATH_QUALITY_RANK_QUANTILE),
    )
    log.info(
        "[ENTRY_HIER_RECIPE] enabled=%d trade_w=%.3f side_w=%.3f utility_w=%.3f bad_path_w=%.3f mae_w=%.3f "
        "trade_global_prior_w=%.3f trade_global_prior_tol=%.3f trade_global_prior_min_label_rate=%.3f "
        "slice_trade_prior_w=%.3f slice_trade_prior_tol=%.3f slice_trade_prior_min_rows=%d slice_trade_prior_min_label_rate=%.3f "
        "slice_trade_acc_edge_w=%.3f slice_trade_acc_edge_margin=%.3f "
        "flat_logit_margin_w=%.3f flat_logit_margin=%.3f flat_logit_margin_min_label_rate=%.3f "
        "slice_flat_logit_margin_w=%.3f slice_flat_logit_margin=%.3f slice_flat_logit_margin_min_rows=%d slice_flat_logit_margin_min_label_rate=%.3f "
        "slice_side_ce_w=%.3f slice_side_margin_w=%.3f slice_side_margin=%.3f "
        "slice_side_acc_edge_w=%.3f slice_side_acc_edge_margin=%.3f "
        "slice_side_min_rows=%d slice_side_min_label_rate=%.3f "
        "side_global_prior_w=%.3f side_global_prior_tol=%.3f side_global_prior_min_label_rate=%.3f "
        "slice_side_prior_w=%.3f slice_side_prior_tol=%.3f slice_side_prior_min_rows=%d slice_side_prior_min_label_rate=%.3f "
        "trade_pos_weight=%.3f bad_path_pos_weight_long=%.3f bad_path_pos_weight_short=%.3f "
        "utility_scale_bps=%.3f mae_scale_bps=%.3f",
        1,
        float(ENTRY_HIER_TRADE_WEIGHT),
        float(ENTRY_HIER_SIDE_WEIGHT),
        float(ENTRY_HIER_UTILITY_WEIGHT),
        float(ENTRY_HIER_BAD_PATH_WEIGHT),
        float(ENTRY_HIER_MAE_WEIGHT),
        float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT),
        float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE),
        float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE),
        float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT),
        float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE),
        int(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS),
        float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE),
        float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT),
        float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN),
        float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT),
        float(ENTRY_HIER_FLAT_LOGIT_MARGIN),
        float(ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE),
        float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT),
        float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN),
        int(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS),
        float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE),
        float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT),
        float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT),
        float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN),
        float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT),
        float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN),
        int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS),
        float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE),
        float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT),
        float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE),
        float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE),
        float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT),
        float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE),
        int(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS),
        float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE),
        float(hier_trade_pos_weight),
        float(hier_bad_path_pos_weight[0]),
        float(hier_bad_path_pos_weight[1]),
        max(1.0, float(ENTRY_AUX_PATH_SCALE_BPS)),
        max(1.0, float(ENTRY_AUX_MFE_SCALE_BPS)),
    )
    log.info(
        "[ENTRY_HARD_NEG_RECIPE] dead_long_ce_multiplier=%.3f dead_long_prob_penalty=%.3f teaser_long_ce_multiplier=%.3f teaser_long_prob_penalty=%.3f hard_neg_long_ce_multiplier=%.3f hard_neg_long_prob_penalty=%.3f",
        float(ENTRY_DEAD_LONG_CE_MULTIPLIER),
        float(ENTRY_DEAD_LONG_PROB_PENALTY),
        float(ENTRY_TEASER_LONG_CE_MULTIPLIER),
        float(ENTRY_TEASER_LONG_PROB_PENALTY),
        float(ENTRY_HARD_NEG_LONG_CE_MULTIPLIER),
        float(ENTRY_HARD_NEG_LONG_PROB_PENALTY),
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=_WEIGHT_DECAY)

    # GX1_FAST_TRAIN: cosine LR + 10% warmup scheduler (steps-based).
    # Step every batch; works correctly with grad-accum since we still call
    # scheduler.step() in train_epoch for each forward pass.
    _scheduler = None
    try:
        from gx1.utils.fast_train import build_cosine_warmup_scheduler, _fast_train_master
        if _fast_train_master():
            _steps_per_epoch = max(
                1,
                math.ceil(len(train_loader) / int(grad_accum_steps)),
            )
            _total_steps = _steps_per_epoch * int(epochs)
            _warmup_steps = max(1, int(_total_steps * 0.10))
            _scheduler = build_cosine_warmup_scheduler(
                optimizer, total_steps=_total_steps, warmup_steps=_warmup_steps,
            )
            log.info(
                "[FAST_TRAIN] cosine+warmup scheduler: total_steps=%d warmup_steps=%d (10%%)",
                _total_steps, _warmup_steps,
            )
    except Exception as _e:
        log.warning("[FAST_TRAIN] scheduler init failed: %r", _e)

    best_state = None
    best_val = float("inf")
    best_acc = float("-inf")  # direction-acc monitor (GX1_V10_CKPT_MONITOR=dir_acc)
    best_dir_ckpt_score = float("-inf")
    best_direction_balance_guard_ok: Optional[bool] = None
    best_direction_slice_contract_ok: Optional[bool] = None
    raw_best_direction_balance_guard_ok: Optional[bool] = None
    raw_best_direction_slice_contract_ok: Optional[bool] = None
    best_direction_slice_stats: Dict[str, Any] = {}
    last_direction_slice_stats: Dict[str, Any] = {}
    best_epoch = -1
    epochs_since_improve = 0
    last_epoch = 0
    early_stopped = False
    hard_red_stopped = False
    _ckpt_monitor = ENTRY_CKPT_MONITOR
    log.info(
        "[CKPT_MONITOR] selecting best checkpoint on %s class_balance_guard_weight=%.3f min_pred_to_label=%.3f min_pred_rate=%.3f",
        _ckpt_monitor,
        float(ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT),
        float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL),
        float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE),
    )

    for epoch in range(epochs):
        last_epoch = epoch + 1
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
        _spec_ckpt_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        _spec_w = _spec_ckpt_model.specialist_out.weight.detach()
        _spec_upstream_tensors = sum(
            1
            for name, _param in _spec_ckpt_model.named_parameters()
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
        tr_loss, tr_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            aux_path_weight=ENTRY_AUX_PATH_WEIGHT,
            aux_mfe_weight=ENTRY_AUX_MFE_WEIGHT,
            aux_tradable_weight=ENTRY_AUX_TRADABLE_WEIGHT,
            aux_path_scale_bps=ENTRY_AUX_PATH_SCALE_BPS,
            aux_mfe_scale_bps=ENTRY_AUX_MFE_SCALE_BPS,
            tradable_pos_weight=tradable_pos_weight,
            clean_edge_pos_weight=clean_edge_pos_weight,
            survival_pos_weight=survival_pos_weight,
            bad_path_pos_weight=bad_path_pos_weight,
            hier_trade_pos_weight=hier_trade_pos_weight,
            hier_bad_path_pos_weight=hier_bad_path_pos_weight,
            scheduler=_scheduler,
            grad_accum_steps=int(grad_accum_steps),
        )
        va_loss, auc, acc, val_short_to_long, val_stats = validate(
            model,
            val_loader,
            criterion,
            device,
            aux_path_weight=ENTRY_AUX_PATH_WEIGHT,
            aux_mfe_weight=ENTRY_AUX_MFE_WEIGHT,
            aux_tradable_weight=ENTRY_AUX_TRADABLE_WEIGHT,
            aux_path_scale_bps=ENTRY_AUX_PATH_SCALE_BPS,
            aux_mfe_scale_bps=ENTRY_AUX_MFE_SCALE_BPS,
            tradable_pos_weight=tradable_pos_weight,
            clean_edge_pos_weight=clean_edge_pos_weight,
            survival_pos_weight=survival_pos_weight,
            bad_path_pos_weight=bad_path_pos_weight,
            hier_trade_pos_weight=hier_trade_pos_weight,
            hier_bad_path_pos_weight=hier_bad_path_pos_weight,
        )
        last_direction_slice_stats = _direction_slice_stats_snapshot(val_stats)
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
                "[ENTRY_LOSS_SUMMARY] split=val epoch=%d ce=%.6f min_pred=%.6f global_prior=%.6f slice_min_pred=%.6f flat_margin=%.6f utility_margin=%.6f side_utility_conviction=%.6f utility_trade_conviction=%.6f utility_triad_ce=%.6f flat_starvation=%.6f slice_recall=%.6f slice_bal_ce=%.6f slice_true_margin=%.6f slice_acc_edge=%.6f slice_confusion_pair=%.6f slice_prior=%.6f tail_direction=%.6f tail_rows=%d path=%.6f mfe=%.6f tradable=%.6f hier_trade=%.6f hier_trade_global_prior=%.6f hier_slice_trade_prior=%.6f hier_flat_logit_margin=%.6f hier_slice_flat_logit_margin=%.6f hier_side=%.6f hier_slice_side_ce=%.6f hier_slice_side_margin=%.6f hier_slice_side_acc_edge=%.6f hier_side_global_prior=%.6f hier_slice_side_prior=%.6f hier_side_acc=%.4f total=%.6f",
                epoch + 1,
                float(val_stats.get("ce_loss_mean", 0.0)),
                float(val_stats.get("direction_min_pred_rate_loss_mean", 0.0)),
                float(val_stats.get("direction_global_prior_match_loss_mean", 0.0)),
                float(val_stats.get("direction_slice_min_pred_rate_loss_mean", 0.0)),
                float(val_stats.get("direction_flat_margin_loss_mean", 0.0)),
                float(val_stats.get("direction_utility_margin_loss_mean", 0.0)),
                float(val_stats.get("direction_side_utility_conviction_loss_mean", 0.0)),
                float(val_stats.get("direction_utility_trade_conviction_loss_mean", 0.0)),
                float(val_stats.get("direction_utility_triad_ce_loss_mean", 0.0)),
                float(val_stats.get("direction_flat_starvation_loss_mean", 0.0)),
                float(val_stats.get("direction_slice_recall_loss_mean", 0.0)),
                float(val_stats.get("direction_slice_balanced_ce_loss_mean", 0.0)),
                float(val_stats.get("direction_slice_true_margin_loss_mean", 0.0)),
                float(val_stats.get("direction_slice_accuracy_edge_loss_mean", 0.0)),
                float(val_stats.get("direction_slice_confusion_pair_loss_mean", 0.0)),
                float(val_stats.get("direction_slice_prior_match_loss_mean", 0.0)),
                float(val_stats.get("tail_direction_loss_mean", 0.0)),
                int(val_stats.get("tail_direction_rows", 0)),
                float(val_stats.get("aux_path_loss_mean", 0.0)),
                float(val_stats.get("aux_mfe_loss_mean", 0.0)),
                float(val_stats.get("aux_tradable_loss_mean", 0.0)),
                float(val_stats.get("hier_trade_loss_mean", 0.0)),
                float(val_stats.get("hier_trade_global_prior_loss_mean", 0.0)),
                float(val_stats.get("hier_slice_trade_prior_loss_mean", 0.0)),
                float(val_stats.get("hier_flat_logit_margin_loss_mean", 0.0)),
                float(val_stats.get("hier_slice_flat_logit_margin_loss_mean", 0.0)),
                float(val_stats.get("hier_side_loss_mean", 0.0)),
                float(val_stats.get("hier_slice_side_ce_loss_mean", 0.0)),
                float(val_stats.get("hier_slice_side_margin_loss_mean", 0.0)),
                float(val_stats.get("hier_slice_side_accuracy_edge_loss_mean", 0.0)),
                float(val_stats.get("hier_side_global_prior_loss_mean", 0.0)),
                float(val_stats.get("hier_slice_side_prior_loss_mean", 0.0)),
                float(val_stats.get("hier_side_acc", 0.0)),
                float(va_loss),
            )
            log.info(
                "[ENTRY_PATH_RANK_LOSS] split=val epoch=%d bad_path_quality=%.6f path_quality=%.6f",
                epoch + 1,
                float(val_stats.get("bad_path_quality_rank_loss_mean", 0.0)),
                float(val_stats.get("path_quality_rank_loss_mean", 0.0)),
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
            log.info(
                "[ENTRY_DIR_CKPT_BALANCE] split=val epoch=%d score=%.6f raw_acc=%.6f l1=%.6f "
                "penalty=%.6f guard_ok=%d pred_long=%.6f pred_short=%.6f pred_flat=%.6f "
                "label_long=%.6f label_short=%.6f label_flat=%.6f min_pred_to_label=%.6f",
                epoch + 1,
                float(val_stats.get("direction_ckpt_score", acc)),
                float(acc),
                float(val_stats.get("direction_pred_label_l1", 0.0)),
                float(val_stats.get("direction_ckpt_balance_penalty", 0.0)),
                int(bool(val_stats.get("direction_class_balance_guard_ok", True))),
                float(val_stats.get("direction_pred_rate_long", 0.0)),
                float(val_stats.get("direction_pred_rate_short", 0.0)),
                float(val_stats.get("direction_pred_rate_flat", 0.0)),
                float(val_stats.get("direction_label_rate_long", 0.0)),
                float(val_stats.get("direction_label_rate_short", 0.0)),
                float(val_stats.get("direction_label_rate_flat", 0.0)),
                float(val_stats.get("direction_min_pred_to_label", 0.0)),
            )
            if "hier_trade_prob_mean" in val_stats:
                log.info(
                    "[ENTRY_HIER_OUTPUT] split=val epoch=%d trade_target=%.6f trade_pred=%.6f "
                    "trade_prob=%.6f flat_prob=%.6f trade_prob_label_flat=%.6f "
                    "side_pred_long_edge=%.6f side_acc_edge=%.6f",
                    epoch + 1,
                    float(val_stats.get("hier_trade_target_rate", 0.0)),
                    float(val_stats.get("hier_trade_pred_rate", 0.0)),
                    float(val_stats.get("hier_trade_prob_mean", 0.0)),
                    float(val_stats.get("hier_flat_prob_mean", 0.0)),
                    float(val_stats.get("hier_trade_prob_label_flat_mean", 0.0)),
                    float(val_stats.get("hier_side_pred_long_rate_on_edge", 0.0)),
                    float(val_stats.get("hier_side_acc_on_edge", 0.0)),
                )
            log.info(
                "[ENTRY_DIR_SLICE_CKPT] split=val epoch=%d score=%.6f failures=%d "
                "acc_failures=%d pred_rate_failures=%d audited=%d acc_deficit=%.6f pred_shortfall=%.6f",
                epoch + 1,
                float(val_stats.get("direction_slice_ckpt_score", val_stats.get("direction_ckpt_score", acc))),
                int(val_stats.get("direction_slice_failure_count", 0)),
                int(val_stats.get("direction_slice_accuracy_failure_count", 0)),
                int(val_stats.get("direction_slice_pred_rate_failure_count", 0)),
                int(val_stats.get("direction_slice_audited_count", 0)),
                float(val_stats.get("direction_slice_accuracy_deficit", 0.0)),
                float(val_stats.get("direction_slice_pred_rate_shortfall", 0.0)),
            )
            for detail in list(val_stats.get("direction_slice_failure_details") or [])[:8]:
                label_rates = detail.get("label_rates") or [0.0, 0.0, 0.0]
                pred_rates = detail.get("pred_rates") or [0.0, 0.0, 0.0]
                required_rates = detail.get("required_pred_rates") or [0.0, 0.0, 0.0]
                log.info(
                    "[ENTRY_DIR_SLICE_FAILURE] split=val epoch=%d ctx_idx=%d ctx_value=%d rows=%d "
                    "acc=%.6f majority=%.6f acc_failed=%d acc_deficit=%.6f "
                    "label_rates=%.6f,%.6f,%.6f pred_rates=%.6f,%.6f,%.6f "
                    "required=%.6f,%.6f,%.6f pred_rate_failed_classes=%s pred_shortfall=%.6f "
                    "hier_trade_target=%.6f hier_trade_pred=%.6f hier_trade_prob=%.6f "
                    "hier_trade_prob_label_flat=%.6f hier_side_pred_long_edge=%.6f "
                    "hier_side_acc_edge=%.6f",
                    epoch + 1,
                    int(detail.get("ctx_cat_index", -1)),
                    int(detail.get("ctx_cat_value", -1)),
                    int(detail.get("rows", 0)),
                    float(detail.get("accuracy", 0.0)),
                    float(detail.get("majority", 0.0)),
                    int(bool(detail.get("accuracy_failed", False))),
                    float(detail.get("accuracy_deficit", 0.0)),
                    float(label_rates[0] if len(label_rates) > 0 else 0.0),
                    float(label_rates[1] if len(label_rates) > 1 else 0.0),
                    float(label_rates[2] if len(label_rates) > 2 else 0.0),
                    float(pred_rates[0] if len(pred_rates) > 0 else 0.0),
                    float(pred_rates[1] if len(pred_rates) > 1 else 0.0),
                    float(pred_rates[2] if len(pred_rates) > 2 else 0.0),
                    float(required_rates[0] if len(required_rates) > 0 else 0.0),
                    float(required_rates[1] if len(required_rates) > 1 else 0.0),
                    float(required_rates[2] if len(required_rates) > 2 else 0.0),
                    ",".join(str(int(cls)) for cls in detail.get("pred_rate_failed_classes", [])),
                    float(detail.get("pred_rate_shortfall", 0.0)),
                    float(detail.get("hier_trade_target_rate", 0.0)),
                    float(detail.get("hier_trade_pred_rate", 0.0)),
                    float(detail.get("hier_trade_prob_mean", 0.0)),
                    float(detail.get("hier_trade_prob_label_flat_mean", 0.0)),
                    float(detail.get("hier_side_pred_long_rate_on_edge", 0.0)),
                    float(detail.get("hier_side_acc_on_edge", 0.0)),
                )
        log.info(
            "[SHORT_TO_LONG_TRAIN] rate=%.6f",
            float(tr_stats.get("short_pred_long_rate", 0.0)),
        )
        if tr_stats:
            log.info(
                "[ENTRY_LOSS_SUMMARY] split=train epoch=%d ce=%.6f min_pred=%.6f global_prior=%.6f slice_min_pred=%.6f flat_margin=%.6f utility_margin=%.6f side_utility_conviction=%.6f utility_trade_conviction=%.6f utility_triad_ce=%.6f flat_starvation=%.6f slice_recall=%.6f slice_bal_ce=%.6f slice_true_margin=%.6f slice_acc_edge=%.6f slice_confusion_pair=%.6f slice_prior=%.6f tail_direction=%.6f tail_rows=%d path=%.6f mfe=%.6f tradable=%.6f hier_trade=%.6f hier_trade_global_prior=%.6f hier_slice_trade_prior=%.6f hier_flat_logit_margin=%.6f hier_slice_flat_logit_margin=%.6f hier_side=%.6f hier_slice_side_ce=%.6f hier_slice_side_margin=%.6f hier_slice_side_acc_edge=%.6f hier_side_global_prior=%.6f hier_slice_side_prior=%.6f hier_side_acc=%.4f total=%.6f",
                epoch + 1,
                float(tr_stats.get("ce_loss_mean", 0.0)),
                float(tr_stats.get("direction_min_pred_rate_loss_mean", 0.0)),
                float(tr_stats.get("direction_global_prior_match_loss_mean", 0.0)),
                float(tr_stats.get("direction_slice_min_pred_rate_loss_mean", 0.0)),
                float(tr_stats.get("direction_flat_margin_loss_mean", 0.0)),
                float(tr_stats.get("direction_utility_margin_loss_mean", 0.0)),
                float(tr_stats.get("direction_side_utility_conviction_loss_mean", 0.0)),
                float(tr_stats.get("direction_utility_trade_conviction_loss_mean", 0.0)),
                float(tr_stats.get("direction_utility_triad_ce_loss_mean", 0.0)),
                float(tr_stats.get("direction_flat_starvation_loss_mean", 0.0)),
                float(tr_stats.get("direction_slice_recall_loss_mean", 0.0)),
                float(tr_stats.get("direction_slice_balanced_ce_loss_mean", 0.0)),
                float(tr_stats.get("direction_slice_true_margin_loss_mean", 0.0)),
                float(tr_stats.get("direction_slice_accuracy_edge_loss_mean", 0.0)),
                float(tr_stats.get("direction_slice_confusion_pair_loss_mean", 0.0)),
                float(tr_stats.get("direction_slice_prior_match_loss_mean", 0.0)),
                float(tr_stats.get("tail_direction_loss_mean", 0.0)),
                int(tr_stats.get("tail_direction_rows", 0)),
                float(tr_stats.get("aux_path_loss_mean", 0.0)),
                float(tr_stats.get("aux_mfe_loss_mean", 0.0)),
                float(tr_stats.get("aux_tradable_loss_mean", 0.0)),
                float(tr_stats.get("hier_trade_loss_mean", 0.0)),
                float(tr_stats.get("hier_trade_global_prior_loss_mean", 0.0)),
                float(tr_stats.get("hier_slice_trade_prior_loss_mean", 0.0)),
                float(tr_stats.get("hier_flat_logit_margin_loss_mean", 0.0)),
                float(tr_stats.get("hier_slice_flat_logit_margin_loss_mean", 0.0)),
                float(tr_stats.get("hier_side_loss_mean", 0.0)),
                float(tr_stats.get("hier_slice_side_ce_loss_mean", 0.0)),
                float(tr_stats.get("hier_slice_side_margin_loss_mean", 0.0)),
                float(tr_stats.get("hier_slice_side_accuracy_edge_loss_mean", 0.0)),
                float(tr_stats.get("hier_side_global_prior_loss_mean", 0.0)),
                float(tr_stats.get("hier_slice_side_prior_loss_mean", 0.0)),
                float(tr_stats.get("hier_side_acc", 0.0)),
                float(tr_loss),
            )
            log.info(
                "[ENTRY_COOPERATION_GATE_LOSS] split=train epoch=%d loss=%.6f "
                "specialist_entropy=%.6f specialist_min_mean=%.6f "
                "tf_entropy=%.6f tf_min_mean=%.6f "
                "family_tf_entropy=%.6f family_tf_min_mean=%.6f",
                epoch + 1,
                float(tr_stats.get("specialist_gate_loss_mean", 0.0)),
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
            log.info(
                "[ENTRY_BAD_PATH_RANK_LOSS] split=train epoch=%d loss=%.6f",
                epoch + 1,
                float(tr_stats.get("bad_path_quality_rank_loss_mean", 0.0)),
            )
            log.info(
                "[ENTRY_PATH_QUALITY_RANK_LOSS] split=train epoch=%d loss=%.6f",
                epoch + 1,
                float(tr_stats.get("path_quality_rank_loss_mean", 0.0)),
            )
        if _ckpt_monitor == "dir_acc":
            _dir_ckpt_score = (
                float(
                    val_stats.get(
                        "direction_slice_ckpt_score",
                        acc,
                    )
                )
                if val_stats
                else float(acc)
            )
            _improved = np.isfinite(_dir_ckpt_score) and (
                _dir_ckpt_score - best_dir_ckpt_score
            ) > float(early_stopping_min_delta)
        else:
            _dir_ckpt_score = float(val_stats.get("direction_ckpt_score", acc)) if val_stats else float(acc)
            _improved = (best_val - va_loss) > float(early_stopping_min_delta)
        _aux_head_health_ok = bool(
            val_stats.get("aux_head_health_ok", False)
        ) if val_stats else False
        _active_head_health_ok = bool(
            val_stats.get("active_head_health_ok", False)
        ) if val_stats else False
        _cooperation_gate_health_ok = bool(
            val_stats.get("cooperation_gate_health_ok", False)
        ) if val_stats else False
        _class_support_ok = bool(
            val_stats.get("direction_class_balance_guard_ok", False)
        ) if val_stats else False
        _admission_ok = _checkpoint_admission_ok(
            profile=profile,
            aux_head_health_ok=_aux_head_health_ok,
            active_head_health_ok=_active_head_health_ok,
            cooperation_gate_health_ok=_cooperation_gate_health_ok,
            class_support_ok=_class_support_ok,
        )
        if _improved and not _admission_ok:
            log.info(
                "[ENTRY_CHECKPOINT_ADMISSION_BLOCKED] epoch=%d profile=%s "
                "aux_head_health_ok=%d active_head_health_ok=%d "
                "cooperation_gate_health_ok=%d class_support_ok=%d",
                epoch + 1,
                profile,
                int(_aux_head_health_ok),
                int(_active_head_health_ok),
                int(_cooperation_gate_health_ok),
                int(_class_support_ok),
            )
        _improved = bool(_improved and _admission_ok)
        if _improved:
            best_val = va_loss
            if np.isfinite(acc):
                best_acc = acc
            if np.isfinite(_dir_ckpt_score):
                best_dir_ckpt_score = _dir_ckpt_score
            best_direction_balance_guard_ok = (
                bool(val_stats.get("direction_class_balance_guard_ok", True)) if val_stats else True
            )
            best_direction_slice_contract_ok = (
                bool(val_stats.get("direction_slice_contract_ok", False)) if val_stats else False
            )
            best_direction_slice_stats = _direction_slice_stats_snapshot(val_stats)
            _ckpt_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.cpu().clone() for k, v in _ckpt_model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_since_improve = 0
            log.info(
                "[BEST_CHECKPOINT] epoch=%d val=%.6f dir_acc=%.6f dir_ckpt_score=%.6f "
                "balance_guard_ok=%d slice_contract_ok=%d active_head_health_ok=%d "
                "monitor=%s",
                best_epoch,
                best_val,
                acc,
                best_dir_ckpt_score,
                int(bool(best_direction_balance_guard_ok)),
                int(bool(best_direction_slice_contract_ok)),
                int(bool(_active_head_health_ok)),
                _ckpt_monitor,
            )
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= int(early_stopping_patience):
                early_stopped = True
                log.info(
                    "[EARLY_STOP] epoch=%d best_epoch=%d best_val=%.6f patience=%d min_delta=%.6f",
                    epoch + 1,
                    best_epoch,
                    best_val,
                    int(early_stopping_patience),
                    float(early_stopping_min_delta),
                )
                break
        if _direction_ckpt_slice_guard_required() and _direction_slice_hard_red_stop_ready(
            epoch=epoch + 1,
            epochs_since_improve=epochs_since_improve,
            best_slice_contract_ok=best_direction_slice_contract_ok,
            val_stats=val_stats,
        ):
            hard_red_stopped = True
            log.info(
                "[ENTRY_DIR_HARD_RED_STOP] epoch=%d best_epoch=%d best_dir_ckpt_score=%.6f "
                "epochs_since_improve=%d patience=%d min_epochs=%d current_failures=%d "
                "current_acc_failures=%d current_pred_rate_failures=%d; refusing to burn more "
                "compute on a no-progress hard-red slice run",
                epoch + 1,
                best_epoch,
                float(best_dir_ckpt_score),
                int(epochs_since_improve),
                int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE),
                int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS),
                int(val_stats.get("direction_slice_failure_count", 0) if val_stats else 0),
                int(val_stats.get("direction_slice_accuracy_failure_count", 0) if val_stats else 0),
                int(val_stats.get("direction_slice_pred_rate_failure_count", 0) if val_stats else 0),
            )
            break

    _require(best_state is not None, "[TRAIN_FAIL_NO_BEST_STATE]")
    model_native_learned_component_movement = (
        _model_native_evidence_fusion_movement_proof(
            evidence_fusion_initial_state,
            best_state,
            selected_checkpoint_epoch=best_epoch,
        )
    )
    log.info(
        "[ENTRY_EVIDENCE_FUSION_MOVEMENT_PASS] epoch=%d components=%s",
        int(best_epoch),
        model_native_learned_component_movement["component_changed"],
    )
    raw_best_direction_balance_guard_ok = best_direction_balance_guard_ok
    raw_best_direction_slice_contract_ok = best_direction_slice_contract_ok
    if _direction_ckpt_balance_guard_required() and not bool(best_direction_balance_guard_ok):
        intended_out_bundle_dir = _resolve_train_out_bundle_dir(out_bundle_dir, gx1_data_override)
        evidence_path = _write_direction_slice_failure_evidence(
            intended_out_bundle_dir,
            {
                "schema_version": "entry_direction_slice_failure_evidence_v1",
                "created_at_utc": _utc_now(),
                "decision": "FAIL_DIRECTION_CLASS_BALANCE_GUARD",
                "failure_code": "TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD",
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
                "hard_red_stopped": bool(hard_red_stopped),
                "best_dir_acc": (float(best_acc) if np.isfinite(best_acc) else None),
                "best_dir_ckpt_score": (
                    float(best_dir_ckpt_score) if np.isfinite(best_dir_ckpt_score) else None
                ),
                "best_direction_balance_guard_ok": best_direction_balance_guard_ok,
                "best_direction_slice_contract_ok": best_direction_slice_contract_ok,
                "raw_best_direction_balance_guard_ok": raw_best_direction_balance_guard_ok,
                "raw_best_direction_slice_contract_ok": raw_best_direction_slice_contract_ok,
                "best_direction_slice_stats": best_direction_slice_stats,
                "last_direction_slice_stats": last_direction_slice_stats,
                "train_recipe": {
                    "ckpt_monitor": str(_ckpt_monitor),
                    "ckpt_class_balance_guard_weight": float(ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT),
                    "ckpt_class_balance_min_pred_to_label": float(
                        ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL
                    ),
                    "ckpt_class_balance_min_pred_rate": float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE),
                    "ckpt_direction_slice_guard": bool(ENTRY_CKPT_DIRECTION_SLICE_GUARD),
                    "direction_global_prior_match_weight": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT),
                    "direction_global_prior_match_tolerance": float(
                        ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE
                    ),
                    "direction_global_prior_match_min_label_rate": float(
                        ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "direction_slice_prior_match_weight": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT),
                    "direction_slice_prior_match_tolerance": float(
                        ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE
                    ),
                    "direction_slice_loss_aggregation": str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION),
                    "direction_slice_balanced_sampler": bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER),
                    "direction_slice_hard_red_stop_patience": int(
                        ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE
                    ),
                    "direction_slice_hard_red_stop_min_epochs": int(
                        ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS
                    ),
                    "direction_vs_flat_margin_weight": float(ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT),
                    "direction_vs_flat_margin": float(ENTRY_DIRECTION_VS_FLAT_MARGIN),
                    "direction_utility_margin_weight": float(ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT),
                    "direction_utility_min_gap_bps": float(ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS),
                    "direction_utility_logit_margin": float(ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN),
                    "direction_side_utility_conviction_weight": float(
                        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT
                    ),
                    "direction_side_utility_conviction_min_gap_bps": float(
                        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS
                    ),
                    "direction_side_utility_conviction_logit_margin": float(
                        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN
                    ),
                    "direction_utility_trade_conviction_weight": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT
                    ),
                    "direction_utility_trade_conviction_min_gap_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS
                    ),
                    "direction_utility_trade_conviction_min_utility_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS
                    ),
                    "direction_utility_trade_conviction_max_bad_path": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH
                    ),
                    "direction_utility_trade_conviction_logit_margin": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN
                    ),
                    "direction_utility_triad_ce_weight": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT),
                    "direction_utility_triad_ce_min_gap_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS
                    ),
                    "direction_utility_triad_ce_min_utility_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS
                    ),
                    "direction_utility_triad_ce_max_bad_path": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH
                    ),
                    "direction_utility_triad_ce_class_weight_cap": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP
                    ),
                    "hier_trade_global_prior_match_weight": float(
                        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_trade_global_prior_match_tolerance": float(
                        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_trade_global_prior_match_min_label_rate": float(
                        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_trade_prior_match_weight": float(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_slice_trade_prior_match_tolerance": float(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_slice_trade_prior_match_min_label_rate": float(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_trade_prior_match_min_rows": int(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS
                    ),
                    "hier_slice_trade_accuracy_edge_weight": float(
                        ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT
                    ),
                    "hier_slice_trade_accuracy_edge_margin": float(
                        ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN
                    ),
                    "hier_flat_logit_margin_weight": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT),
                    "hier_flat_logit_margin": float(ENTRY_HIER_FLAT_LOGIT_MARGIN),
                    "hier_flat_logit_margin_min_label_rate": float(
                        ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE
                    ),
                    "hier_slice_flat_logit_margin_weight": float(
                        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT
                    ),
                    "hier_slice_flat_logit_margin": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN),
                    "hier_slice_flat_logit_margin_min_label_rate": float(
                        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE
                    ),
                    "hier_slice_flat_logit_margin_min_rows": int(
                        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS
                    ),
                    "hier_slice_side_ce_weight": float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT),
                    "hier_slice_side_true_margin_weight": float(
                        ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT
                    ),
                    "hier_slice_side_true_margin": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN),
                    "hier_slice_side_accuracy_edge_weight": float(
                        ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT
                    ),
                    "hier_slice_side_accuracy_edge_margin": float(
                        ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN
                    ),
                    "hier_slice_side_min_label_rate": float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE),
                    "hier_slice_side_min_rows": int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS),
                    "hier_side_global_prior_match_weight": float(
                        ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_side_global_prior_match_tolerance": float(
                        ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_side_global_prior_match_min_label_rate": float(
                        ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_side_prior_match_weight": float(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_slice_side_prior_match_tolerance": float(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_slice_side_prior_match_min_label_rate": float(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_side_prior_match_min_rows": int(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS
                    ),
                    "direction_flat_starvation_weight": float(ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT),
                    "direction_flat_starvation_min_label_rate": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE
                    ),
                    "direction_flat_starvation_min_rows": int(ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS),
                    "direction_flat_starvation_pred_fraction": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION
                    ),
                    "direction_flat_starvation_pred_floor": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR
                    ),
                    "direction_flat_starvation_logit_margin": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN
                    ),
                },
            },
        )
        log.error("[ENTRY_DIR_CLASS_BALANCE_FAILURE_EVIDENCE] path=%s", evidence_path)
        raise RuntimeError(
            "[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD] "
            "best checkpoint failed active LONG/SHORT/FLAT class-balance guard; "
            "refusing to write a collapsed direction bundle"
        )
    if _direction_ckpt_slice_guard_required() and not bool(best_direction_slice_contract_ok):
        intended_out_bundle_dir = _resolve_train_out_bundle_dir(out_bundle_dir, gx1_data_override)
        evidence_path = _write_direction_slice_failure_evidence(
            intended_out_bundle_dir,
            {
                "schema_version": "entry_direction_slice_failure_evidence_v1",
                "created_at_utc": _utc_now(),
                "decision": "FAIL_DIRECTION_SLICE_GUARD",
                "failure_code": "TRAIN_FAIL_DIRECTION_SLICE_GUARD",
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
                "hard_red_stopped": bool(hard_red_stopped),
                "best_dir_acc": (float(best_acc) if np.isfinite(best_acc) else None),
                "best_dir_ckpt_score": (
                    float(best_dir_ckpt_score) if np.isfinite(best_dir_ckpt_score) else None
                ),
                "best_direction_balance_guard_ok": best_direction_balance_guard_ok,
                "best_direction_slice_contract_ok": best_direction_slice_contract_ok,
                "raw_best_direction_balance_guard_ok": raw_best_direction_balance_guard_ok,
                "raw_best_direction_slice_contract_ok": raw_best_direction_slice_contract_ok,
                "best_direction_slice_stats": best_direction_slice_stats,
                "last_direction_slice_stats": last_direction_slice_stats,
                "train_recipe": {
                    "ckpt_monitor": str(_ckpt_monitor),
                    "ckpt_direction_slice_guard": bool(ENTRY_CKPT_DIRECTION_SLICE_GUARD),
                    "direction_global_prior_match_weight": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT),
                    "direction_global_prior_match_tolerance": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE),
                    "direction_global_prior_match_min_label_rate": float(
                        ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "direction_slice_min_pred_rate_loss_weight": float(
                        ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT
                    ),
                    "direction_slice_min_pred_rate_fraction": float(
                        ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION
                    ),
                    "direction_slice_min_pred_rate_floor": float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR),
                    "direction_slice_min_label_rate": float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE),
                    "direction_slice_min_rows": int(ENTRY_DIRECTION_SLICE_MIN_ROWS),
                    "direction_slice_ctx_cat_indices": str(ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES),
                    "direction_slice_recall_loss_weight": float(ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT),
                    "direction_slice_recall_prob_floor": float(ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR),
                    "direction_slice_recall_min_label_rate": float(ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE),
                    "direction_slice_recall_min_rows": int(ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS),
                    "direction_slice_balanced_ce_weight": float(ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT),
                    "direction_slice_balanced_ce_min_label_rate": float(
                        ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE
                    ),
                    "direction_slice_balanced_ce_min_rows": int(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS),
                    "direction_slice_true_margin_weight": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT),
                    "direction_slice_true_margin": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN),
                    "direction_slice_true_margin_min_label_rate": float(
                        ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE
                    ),
                    "direction_slice_true_margin_min_rows": int(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS),
                    "direction_slice_accuracy_edge_weight": float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT),
                    "direction_slice_accuracy_edge_margin": float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN),
                    "direction_slice_confusion_pair_weight": float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT),
                    "direction_slice_confusion_pair_margin": float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN),
                    "direction_slice_accuracy_edge_min_label_rate": float(
                        ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE
                    ),
                    "direction_slice_accuracy_edge_min_rows": int(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS),
                    "direction_slice_prior_match_weight": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT),
                    "direction_slice_prior_match_tolerance": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE),
                    "direction_slice_prior_match_min_label_rate": float(
                        ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "direction_slice_prior_match_min_rows": int(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS),
                    "direction_slice_loss_aggregation": str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION),
                    "direction_slice_balanced_sampler": bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER),
                    "direction_slice_balanced_sampler_min_rows": int(
                        ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS
                    ),
                    "direction_slice_hard_red_stop_patience": int(
                        ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE
                    ),
                    "direction_slice_hard_red_stop_min_epochs": int(
                        ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS
                    ),
                    "direction_vs_flat_margin_weight": float(ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT),
                    "direction_vs_flat_margin": float(ENTRY_DIRECTION_VS_FLAT_MARGIN),
                    "direction_utility_margin_weight": float(ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT),
                    "direction_utility_min_gap_bps": float(ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS),
                    "direction_utility_logit_margin": float(ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN),
                    "direction_side_utility_conviction_weight": float(
                        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT
                    ),
                    "direction_side_utility_conviction_min_gap_bps": float(
                        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS
                    ),
                    "direction_side_utility_conviction_logit_margin": float(
                        ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN
                    ),
                    "direction_utility_trade_conviction_weight": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT
                    ),
                    "direction_utility_trade_conviction_min_gap_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS
                    ),
                    "direction_utility_trade_conviction_min_utility_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS
                    ),
                    "direction_utility_trade_conviction_max_bad_path": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH
                    ),
                    "direction_utility_trade_conviction_logit_margin": float(
                        ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN
                    ),
                    "direction_utility_triad_ce_weight": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT),
                    "direction_utility_triad_ce_min_gap_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS
                    ),
                    "direction_utility_triad_ce_min_utility_bps": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS
                    ),
                    "direction_utility_triad_ce_max_bad_path": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH
                    ),
                    "direction_utility_triad_ce_class_weight_cap": float(
                        ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP
                    ),
                    "hier_trade_global_prior_match_weight": float(
                        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_trade_global_prior_match_tolerance": float(
                        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_trade_global_prior_match_min_label_rate": float(
                        ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_trade_prior_match_weight": float(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_slice_trade_prior_match_tolerance": float(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_slice_trade_prior_match_min_label_rate": float(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_trade_prior_match_min_rows": int(
                        ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS
                    ),
                    "hier_slice_trade_accuracy_edge_weight": float(
                        ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT
                    ),
                    "hier_slice_trade_accuracy_edge_margin": float(
                        ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN
                    ),
                    "hier_flat_logit_margin_weight": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT),
                    "hier_flat_logit_margin": float(ENTRY_HIER_FLAT_LOGIT_MARGIN),
                    "hier_flat_logit_margin_min_label_rate": float(
                        ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE
                    ),
                    "hier_slice_flat_logit_margin_weight": float(
                        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT
                    ),
                    "hier_slice_flat_logit_margin": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN),
                    "hier_slice_flat_logit_margin_min_label_rate": float(
                        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE
                    ),
                    "hier_slice_flat_logit_margin_min_rows": int(
                        ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS
                    ),
                    "hier_slice_side_ce_weight": float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT),
                    "hier_slice_side_true_margin_weight": float(
                        ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT
                    ),
                    "hier_slice_side_true_margin": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN),
                    "hier_slice_side_accuracy_edge_weight": float(
                        ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT
                    ),
                    "hier_slice_side_accuracy_edge_margin": float(
                        ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN
                    ),
                    "hier_slice_side_min_label_rate": float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE),
                    "hier_slice_side_min_rows": int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS),
                    "hier_side_global_prior_match_weight": float(
                        ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_side_global_prior_match_tolerance": float(
                        ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_side_global_prior_match_min_label_rate": float(
                        ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_side_prior_match_weight": float(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT
                    ),
                    "hier_slice_side_prior_match_tolerance": float(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE
                    ),
                    "hier_slice_side_prior_match_min_label_rate": float(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE
                    ),
                    "hier_slice_side_prior_match_min_rows": int(
                        ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS
                    ),
                    "direction_flat_starvation_weight": float(ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT),
                    "direction_flat_starvation_min_label_rate": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE
                    ),
                    "direction_flat_starvation_min_rows": int(ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS),
                    "direction_flat_starvation_pred_fraction": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION
                    ),
                    "direction_flat_starvation_pred_floor": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR
                    ),
                    "direction_flat_starvation_logit_margin": float(
                        ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN
                    ),
                },
            },
        )
        log.error("[ENTRY_DIR_SLICE_FAILURE_EVIDENCE] path=%s", evidence_path)
        raise RuntimeError(
            "[TRAIN_FAIL_DIRECTION_SLICE_GUARD] "
            "best checkpoint failed active direction slice contract; "
            "refusing to write a slice-failed direction bundle"
        )

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
    model_native_direction_evidence_fusion = direction_evidence_fusion_metadata()
    run_lineage = {
        "schema_version": "entry_model_native_training_run_lineage_v1",
        "training_run_id": str(run_id),
        "dataset_run_id": str(dataset_run_id),
    }
    lock = {
        "version": "entry_v10_ctx_lock_v1",
        "created_at_utc": _utc_now(),
        "signal_bridge_id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "signal_bridge_contract_sha256": trained_model_native_signal_contract["static_contract_sha256"],
        "contract_mode": train_contract_mode,
        "direction_logit_mode": direction_logit_mode,
        "direction_decision_contract": direction_decision_contract,
        "model_native_direction_evidence_fusion": model_native_direction_evidence_fusion,
        "model_native_learned_component_movement": model_native_learned_component_movement,
        "model_native_signal_contract": trained_model_native_signal_contract,
        "context_specialist_routing": specialist_meta["context_routing"],
        "input_normalization": input_normalization,
        "input_normalization_fit_population_proof": (
            input_normalization_fit_population_proof
        ),
        "run_lineage": run_lineage,
        "aux_head_target_contract": train_ds.aux_head_target_contract,
        "model_native_training_objective": model_native_training_objective,
        "ctx_tag": f"CTX{ctx_cont_dim}CAT{ctx_cat_dim}",
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ordered_ctx_cont_names": list(ordered_ctx_cont_names),
        "ordered_ctx_cat_names": list(ordered_ctx_cat_names),
        "ordered_signal_names": trained_signal_names,
        "neutral_xgb_bridge": False,
        "xgb_bridge_source": None,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "seq_len": seq_len,
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": state_dict_sha256,
    }
    # State stores unconstrained raw scalars; the immutable contract records
    # their hashes and the corresponding strictly-positive effective scales.
    learned_tf_input_scale_raw: Dict[str, float] = {}
    for _tf in TF_INPUT_SCALE_NAMES:
        _key = f"tf_input_scale_{_tf}"
        _state_key = _key if _key in best_state else f"_orig_mod.{_key}"
        if _state_key not in best_state:
            raise RuntimeError(f"[TF_INPUT_SCALE_STATE_MISSING] {_key}")
        _value = float(best_state[_state_key].item())
        if not np.isfinite(_value):
            raise RuntimeError(f"[TF_INPUT_SCALE_STATE_NONFINITE] {_key}={_value}")
        learned_tf_input_scale_raw[_tf] = _value
    tf_input_scale_contract = build_tf_input_scale_contract(
        init_effective={
            "m5": float(tf_input_scale_init_m5),
            "m15": float(tf_input_scale_init_m15),
            "h1": float(tf_input_scale_init_h1),
            "h4": float(tf_input_scale_init_h4),
            "d1": float(tf_input_scale_init_d1),
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
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(),
        "model_native_training_objective": model_native_training_objective,
        "model_native_direction_evidence_fusion": model_native_direction_evidence_fusion,
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
        "ckpt_monitor": _ckpt_monitor,
        "best_dir_acc": (float(best_acc) if np.isfinite(best_acc) else None),
        "best_dir_ckpt_score": (float(best_dir_ckpt_score) if np.isfinite(best_dir_ckpt_score) else None),
        "best_direction_balance_guard_ok": best_direction_balance_guard_ok,
        "raw_best_direction_balance_guard_ok": raw_best_direction_balance_guard_ok,
        "best_direction_slice_contract_ok": best_direction_slice_contract_ok,
        "raw_best_direction_slice_contract_ok": raw_best_direction_slice_contract_ok,
        "ckpt_class_balance_guard_weight": float(ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT),
        "ckpt_class_balance_min_pred_to_label": float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL),
        "ckpt_class_balance_min_pred_rate": float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE),
        "ckpt_direction_slice_guard": bool(ENTRY_CKPT_DIRECTION_SLICE_GUARD),
        "last_epoch": last_epoch,
        "early_stopped": bool(early_stopped),
        "hard_red_stopped": bool(hard_red_stopped),
        "early_stopping_patience": int(early_stopping_patience),
        "early_stopping_min_delta": float(early_stopping_min_delta),
        "epochs": epochs,
        "lr": lr,
        # V12.2 multi-TF marker — live inference inspects this to decide
        # whether to feed seq_m15/seq_h1/seq_h4/seq_d1 into the model.
        "multi_tf": {
            "enabled": True,
            "v2_mode": True,
            "m5_seq_dim": int(_mtf_feat_count),
            "m5_seq_len": int(multi_tf_seq_len),
            "m15_seq_dim": int(_mtf_feat_count),
            "h1_seq_dim": int(_mtf_feat_count),
            "h4_seq_dim": int(_mtf_feat_count),
            "d1_seq_dim": int(_mtf_feat_count),
            "m15_seq_len": int(_m15_len),
            "h1_seq_len": int(multi_tf_seq_len),
            "h4_seq_len": int(_h4_len),
            "d1_seq_len": int(_d1_len),
            "multi_tf_scale": float(multi_tf_scale),
            "feature_contract": "MULTI_TF_PER_BAR_V2",
            "matrix_contract": HTF_V2_MATRIX_CONTRACT,
            "feature_names": list(MULTI_TF_PER_BAR_FEATURES_V2),
            "feature_names_sha256": MULTI_TF_FEATURE_NAMES_SHA256_V2,
            "closed_bar_target_availability": bool(
                getattr(train_ds, "_multi_tf_target_availability_shift", pd.Timedelta(0)) > pd.Timedelta(0)
            ),
            "target_availability_shift_minutes": float(
                getattr(train_ds, "_multi_tf_target_availability_shift", pd.Timedelta(0)).total_seconds()
                / 60.0
            ),
        },
        # 2026-06-02: per-TF learnable input scaling marker. Inference must
        # init the model with the same exact learnable scale priors
        # values used at train time so state_dict load is shape-compatible.
        # Learned values overwrite the inits via state_dict; we surface them
        # here for inspection/debugging.
        "tf_input_scale": tf_input_scale_contract,
        # Positional encoding marker — buffer is persistent=False (not in
        # state_dict), so the live bundle loader MUST read this to rebuild the
        # model with matching forward behaviour.
        "enable_pos_enc": True,
        "enable_regime_film": True,
        "enable_mtf_direction_head": True,
        "mtf_dir_aux_weight": float(ENTRY_MTF_DIR_AUX_WEIGHT),
        "mtf_dir_aux_uses_direction_balance_repair": bool(
            float(ENTRY_MTF_DIR_AUX_WEIGHT) > 0.0
        ),
        "batch_size": batch_size,
        "seed": seed,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "ordered_signal_names": trained_signal_names,
        "contract_mode": train_contract_mode,
        "direction_logit_mode": direction_logit_mode,
        "direction_decision_contract": direction_decision_contract,
        "model_native_signal_contract": trained_model_native_signal_contract,
        "run_lineage": run_lineage,
        "aux_head_target_contract": train_ds.aux_head_target_contract,
        "neutral_xgb_bridge": False,
        "xgb_bridge_source": None,
        "model_native_state_contract": trained_model_native_state_contract,
        "seq_len": seq_len,
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ordered_ctx_cont_names": list(ordered_ctx_cont_names),
        "ordered_ctx_cat_names": list(ordered_ctx_cat_names),
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "expected_ctx_cont_dim": ctx_cont_dim,
        "expected_ctx_cat_dim": ctx_cat_dim,
        "supports_context_features": True,
        "signal_bridge_id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "ctx_tag": f"CTX{ctx_cont_dim}CAT{ctx_cat_dim}",
        "model_class": "EntryV10CtxHybridTransformer",
        "arch_id": "entry_v10_ctx_hybrid_transformer",
        "specialist_fusion": {
            **specialist_meta,
            "num_layers": int(specialist_num_layers),
            "fusion_scale": float(specialist_fusion_scale),
            "cross_family_fusion_scale": float(specialist_fusion_scale),
        },
        "state_dict_sha256": state_dict_sha256,
        "anchored_entry_enabled": False,
        "anchor_source": None,
        "anchor_gate": {"enabled": False},
        "hierarchical_entry_heads": {
            "enabled": True,
            "selection_score": MODEL_DIRECTION_SELECTION_MODE,
            "auxiliary_head_role": "training_and_diagnostics_only",
            "side_utility_scale_bps": max(1.0, float(ENTRY_AUX_PATH_SCALE_BPS)),
            "side_mae_scale_bps": max(1.0, float(ENTRY_AUX_MFE_SCALE_BPS)),
            "heads": [
                "trade_vs_flat",
                "long_vs_short_given_trade",
                "side_path_utility_bps",
                "side_bad_path_probability",
                "side_expected_mae_bps",
                "side_valid_trade_probability",
            ],
            "side_validity": {
                "enabled": True,
                "loss_weight": float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT),
                "min_utility_bps": float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS),
                "pos_weight_cap": float(ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP),
                "targets": {
                    "long_valid_trade": [
                        "y_long_path_utility_bps >= min_utility_bps",
                        "y_long_bad_path == 0",
                    ],
                    "short_valid_trade": [
                        "y_short_path_utility_bps >= min_utility_bps",
                        "y_short_bad_path == 0",
                    ],
                },
                "runtime_rule_free": True,
            },
            "trade_prior_supervision": {
                "enabled": (
                    float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT) > 0.0
                    or float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT) > 0.0
                ),
                "global_prior_match_weight": float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT),
                "global_prior_match_tolerance": float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE),
                "global_prior_match_min_label_rate": float(
                    ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                ),
                "prior_match_weight": float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT),
                "prior_match_tolerance": float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE),
                "prior_match_min_label_rate": float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE),
                "prior_match_min_rows": int(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS),
                "accuracy_edge_weight": float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT),
                "accuracy_edge_margin": float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN),
                "flat_logit_margin_weight": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT),
                "flat_logit_margin": float(ENTRY_HIER_FLAT_LOGIT_MARGIN),
                "flat_logit_margin_min_label_rate": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE),
                "slice_flat_logit_margin_weight": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT),
                "slice_flat_logit_margin": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN),
                "slice_flat_logit_margin_min_label_rate": float(
                    ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE
                ),
                "slice_flat_logit_margin_min_rows": int(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS),
                "ctx_cat_indices": str(ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES),
                "loss_aggregation": str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION),
                "target_classes": ["trade", "flat"],
                "runtime_rule_free": True,
            },
            "slice_side_supervision": {
                "enabled": (
                    float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT) > 0.0
                ),
                "applies_to": ["side_logits"],
                "balanced_ce_weight": float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT),
                "true_margin_weight": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT),
                "true_margin": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN),
                "accuracy_edge_weight": float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT),
                "accuracy_edge_margin": float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN),
                "min_label_rate": float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE),
                "min_rows": int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS),
                "global_prior_match_weight": float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT),
                "global_prior_match_tolerance": float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE),
                "global_prior_match_min_label_rate": float(
                    ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
                ),
                "prior_match_weight": float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT),
                "prior_match_tolerance": float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE),
                "prior_match_min_label_rate": float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE),
                "prior_match_min_rows": int(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS),
                "ctx_cat_indices": str(ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES),
                "loss_aggregation": str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION),
                "runtime_rule_free": True,
            },
            "side_outcome_targets": {
                "utility": ["y_long_path_utility_bps", "y_short_path_utility_bps"],
                "bad_path": ["y_long_bad_path", "y_short_bad_path"],
                "expected_mae": ["y_long_expected_mae_bps", "y_short_expected_mae_bps"],
                "structural_direction_rewrite": False,
            },
        },
        "trendline_rail_head": {
            "enabled": True,
            "output_dim": 6,
            "labels": [
                "y_rising_channel_support_touch",
                "y_falling_channel_resistance_touch",
                "y_countertrend_short_trap",
                "y_countertrend_long_trap",
                "y_short_high_mae_low_mfe_early_failure",
                "y_long_high_mae_low_mfe_early_failure",
            ],
            "aux_weight": float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT),
            "direction_mapping": "direct_learned_evidence_fusion",
            "hand_written_direction_pressure": False,
        },
        "class_weights": {
            "long": float(long_class_weight),
            "short": float(short_class_weight),
            "flat": float(flat_class_weight),
        },
        "cost_matrix": {
            "long_to_short": float(ENTRY_COST_LONG_TO_SHORT),
            "long_to_flat": float(ENTRY_COST_LONG_TO_FLAT),
            "short_to_long": float(ENTRY_COST_SHORT_TO_LONG),
            "short_to_flat": float(ENTRY_COST_SHORT_TO_FLAT),
            "flat_to_long": float(ENTRY_COST_FLAT_TO_LONG),
            "flat_to_short": float(ENTRY_COST_FLAT_TO_SHORT),
        },
        "cost_sensitive_loss_enabled": bool(ENTRY_COST_SENSITIVE_ENABLED),
        "cost_sensitive_loss_scale": float(ENTRY_COST_SENSITIVE_SCALE),
        "pred_balance_alpha": float(ENTRY_PRED_BALANCE_ALPHA),
        "pred_balance_target": str(ENTRY_PRED_BALANCE_TARGET),
        "pred_balance_class_weights": [float(value) for value in ENTRY_PRED_BALANCE_CLASS_WEIGHTS],
        "direction_ce_scale": float(ENTRY_DIRECTION_CE_SCALE),
        "direction_min_pred_rate_loss_weight": float(ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT),
        "direction_min_pred_rate_fraction": float(ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION),
        "direction_min_pred_rate_floor": float(ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR),
        "direction_min_pred_rate_softmax_temperature": float(
            ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE
        ),
        "direction_global_prior_match_weight": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT),
        "direction_global_prior_match_tolerance": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE),
        "direction_global_prior_match_min_label_rate": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE),
        "direction_slice_min_pred_rate_loss_weight": float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT),
        "direction_slice_min_pred_rate_fraction": float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION),
        "direction_slice_min_pred_rate_floor": float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR),
        "direction_slice_min_label_rate": float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE),
        "direction_slice_min_rows": int(ENTRY_DIRECTION_SLICE_MIN_ROWS),
        "direction_slice_ctx_cat_indices": str(ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES),
        "direction_slice_recall_loss_weight": float(ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT),
        "direction_slice_recall_prob_floor": float(ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR),
        "direction_slice_recall_min_label_rate": float(ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE),
        "direction_slice_recall_min_rows": int(ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS),
        "direction_slice_balanced_ce_weight": float(ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT),
        "direction_slice_balanced_ce_min_label_rate": float(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE),
        "direction_slice_balanced_ce_min_rows": int(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS),
        "direction_slice_true_margin_weight": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT),
        "direction_slice_true_margin": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN),
        "direction_slice_true_margin_min_label_rate": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE),
        "direction_slice_true_margin_min_rows": int(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS),
        "direction_slice_accuracy_edge_weight": float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT),
        "direction_slice_accuracy_edge_margin": float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN),
        "direction_slice_confusion_pair_weight": float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT),
        "direction_slice_confusion_pair_margin": float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN),
        "direction_slice_accuracy_edge_min_label_rate": float(
            ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE
        ),
        "direction_slice_accuracy_edge_min_rows": int(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS),
        "direction_slice_prior_match_weight": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT),
        "direction_slice_prior_match_tolerance": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE),
        "direction_slice_prior_match_min_label_rate": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE),
        "direction_slice_prior_match_min_rows": int(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS),
        "direction_slice_loss_aggregation": str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION),
        "direction_slice_balanced_sampler": bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER),
        "direction_slice_balanced_sampler_min_rows": int(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS),
        "direction_vs_flat_margin_weight": float(ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT),
        "direction_vs_flat_margin": float(ENTRY_DIRECTION_VS_FLAT_MARGIN),
        "direction_utility_margin_weight": float(ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT),
        "direction_utility_min_gap_bps": float(ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS),
        "direction_utility_logit_margin": float(ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN),
        "direction_side_utility_conviction_weight": float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT),
        "direction_side_utility_conviction_min_gap_bps": float(
            ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS
        ),
        "direction_side_utility_conviction_logit_margin": float(
            ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN
        ),
        "direction_utility_trade_conviction_weight": float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT),
        "direction_utility_trade_conviction_min_gap_bps": float(
            ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS
        ),
        "direction_utility_trade_conviction_min_utility_bps": float(
            ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS
        ),
        "direction_utility_trade_conviction_max_bad_path": float(
            ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH
        ),
        "direction_utility_trade_conviction_logit_margin": float(
            ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN
        ),
        "direction_utility_triad_ce_weight": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT),
        "direction_utility_triad_ce_min_gap_bps": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS),
        "direction_utility_triad_ce_min_utility_bps": float(
            ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS
        ),
        "direction_utility_triad_ce_max_bad_path": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH),
        "direction_utility_triad_ce_class_weight_cap": float(
            ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP
        ),
        "hier_trade_global_prior_match_weight": float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT),
        "hier_trade_global_prior_match_tolerance": float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE),
        "hier_trade_global_prior_match_min_label_rate": float(
            ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
        ),
        "hier_slice_trade_prior_match_weight": float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT),
        "hier_slice_trade_prior_match_tolerance": float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE),
        "hier_slice_trade_prior_match_min_label_rate": float(
            ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE
        ),
        "hier_slice_trade_prior_match_min_rows": int(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS),
        "hier_slice_trade_accuracy_edge_weight": float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT),
        "hier_slice_trade_accuracy_edge_margin": float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN),
        "hier_flat_logit_margin_weight": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT),
        "hier_flat_logit_margin": float(ENTRY_HIER_FLAT_LOGIT_MARGIN),
        "hier_flat_logit_margin_min_label_rate": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE),
        "hier_slice_flat_logit_margin_weight": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT),
        "hier_slice_flat_logit_margin": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN),
        "hier_slice_flat_logit_margin_min_label_rate": float(
            ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE
        ),
        "hier_slice_flat_logit_margin_min_rows": int(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS),
        "hier_slice_side_ce_weight": float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT),
        "hier_slice_side_true_margin_weight": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT),
        "hier_slice_side_true_margin": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN),
        "hier_slice_side_accuracy_edge_weight": float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT),
        "hier_slice_side_accuracy_edge_margin": float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN),
        "hier_slice_side_min_label_rate": float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE),
        "hier_slice_side_min_rows": int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS),
        "hier_side_global_prior_match_weight": float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT),
        "hier_side_global_prior_match_tolerance": float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE),
        "hier_side_global_prior_match_min_label_rate": float(
            ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
        ),
        "hier_slice_side_prior_match_weight": float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT),
        "hier_slice_side_prior_match_tolerance": float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE),
        "hier_slice_side_prior_match_min_label_rate": float(
            ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE
        ),
        "hier_slice_side_prior_match_min_rows": int(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS),
        "direction_flat_starvation_weight": float(ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT),
        "direction_flat_starvation_min_label_rate": float(ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE),
        "direction_flat_starvation_min_rows": int(ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS),
        "direction_flat_starvation_pred_fraction": float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION),
        "direction_flat_starvation_pred_floor": float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR),
        "direction_flat_starvation_logit_margin": float(ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN),
        "tail_direction_ce_weight": float(ENTRY_TAIL_DIRECTION_CE_WEIGHT),
        "tail_direction_quality_quantile": float(ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE),
        "tail_direction_min_batch": int(ENTRY_TAIL_DIRECTION_MIN_BATCH),
        "specialist_gate_entropy_weight": float(ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT),
        "specialist_gate_balance_weight": float(ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT),
        "specialist_gate_min_mean": float(ENTRY_SPECIALIST_GATE_MIN_MEAN),
        "bad_path_quality_rank_weight": float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT),
        "bad_path_quality_rank_margin": float(ENTRY_BAD_PATH_QUALITY_RANK_MARGIN),
        "bad_path_quality_rank_quantile": float(ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE),
        "path_quality_rank_weight": float(ENTRY_PATH_QUALITY_RANK_WEIGHT),
        "path_quality_rank_margin": float(ENTRY_PATH_QUALITY_RANK_MARGIN),
        "path_quality_rank_quantile": float(ENTRY_PATH_QUALITY_RANK_QUANTILE),
        "grad_clip_norm": float(_GRAD_CLIP_NORM),
        "weight_decay": float(_WEIGHT_DECAY),
        "train_recipe": {
            "direction_ce_scale": float(ENTRY_DIRECTION_CE_SCALE),
            "tail_direction_ce_weight": float(ENTRY_TAIL_DIRECTION_CE_WEIGHT),
            "tail_direction_quality_quantile": float(ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE),
            "tail_direction_min_batch": int(ENTRY_TAIL_DIRECTION_MIN_BATCH),
            "pred_balance_alpha": float(ENTRY_PRED_BALANCE_ALPHA),
            "pred_balance_target": str(ENTRY_PRED_BALANCE_TARGET),
            "pred_balance_class_weights": [float(value) for value in ENTRY_PRED_BALANCE_CLASS_WEIGHTS],
            "mtf_dir_aux_weight": float(ENTRY_MTF_DIR_AUX_WEIGHT),
            "mtf_dir_aux_uses_direction_balance_repair": bool(
                float(ENTRY_MTF_DIR_AUX_WEIGHT) > 0.0
            ),
            "ckpt_monitor": str(_ckpt_monitor),
            "ckpt_class_balance_guard_weight": float(ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT),
            "ckpt_class_balance_min_pred_to_label": float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL),
            "ckpt_class_balance_min_pred_rate": float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE),
            "ckpt_direction_slice_guard": bool(ENTRY_CKPT_DIRECTION_SLICE_GUARD),
            "direction_min_pred_rate_loss_weight": float(ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT),
            "direction_min_pred_rate_fraction": float(ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION),
            "direction_min_pred_rate_floor": float(ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR),
            "direction_min_pred_rate_softmax_temperature": float(
                ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE
            ),
            "direction_global_prior_match_weight": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT),
            "direction_global_prior_match_tolerance": float(ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE),
            "direction_global_prior_match_min_label_rate": float(
                ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
            ),
            "direction_slice_min_pred_rate_loss_weight": float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT),
            "direction_slice_min_pred_rate_fraction": float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION),
            "direction_slice_min_pred_rate_floor": float(ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR),
            "direction_slice_min_label_rate": float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE),
            "direction_slice_min_rows": int(ENTRY_DIRECTION_SLICE_MIN_ROWS),
            "direction_slice_ctx_cat_indices": str(ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES),
            "direction_slice_recall_loss_weight": float(ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT),
            "direction_slice_recall_prob_floor": float(ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR),
            "direction_slice_recall_min_label_rate": float(ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE),
            "direction_slice_recall_min_rows": int(ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS),
            "direction_slice_balanced_ce_weight": float(ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT),
            "direction_slice_balanced_ce_min_label_rate": float(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE),
            "direction_slice_balanced_ce_min_rows": int(ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS),
            "direction_slice_true_margin_weight": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT),
            "direction_slice_true_margin": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN),
            "direction_slice_true_margin_min_label_rate": float(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE),
            "direction_slice_true_margin_min_rows": int(ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS),
            "direction_slice_accuracy_edge_weight": float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT),
            "direction_slice_accuracy_edge_margin": float(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN),
            "direction_slice_confusion_pair_weight": float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT),
            "direction_slice_confusion_pair_margin": float(ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN),
            "direction_slice_accuracy_edge_min_label_rate": float(
                ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE
            ),
            "direction_slice_accuracy_edge_min_rows": int(ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS),
            "direction_slice_prior_match_weight": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT),
            "direction_slice_prior_match_tolerance": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE),
            "direction_slice_prior_match_min_label_rate": float(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE),
            "direction_slice_prior_match_min_rows": int(ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS),
            "direction_slice_loss_aggregation": str(ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION),
            "direction_slice_balanced_sampler": bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER),
            "direction_slice_balanced_sampler_min_rows": int(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS),
            "direction_slice_hard_red_stop_patience": int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE),
            "direction_slice_hard_red_stop_min_epochs": int(ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS),
            "direction_vs_flat_margin_weight": float(ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT),
            "direction_vs_flat_margin": float(ENTRY_DIRECTION_VS_FLAT_MARGIN),
            "direction_utility_margin_weight": float(ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT),
            "direction_utility_min_gap_bps": float(ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS),
            "direction_utility_logit_margin": float(ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN),
            "direction_side_utility_conviction_weight": float(ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT),
            "direction_side_utility_conviction_min_gap_bps": float(
                ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS
            ),
            "direction_side_utility_conviction_logit_margin": float(
                ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN
            ),
            "direction_utility_trade_conviction_weight": float(ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT),
            "direction_utility_trade_conviction_min_gap_bps": float(
                ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS
            ),
            "direction_utility_trade_conviction_min_utility_bps": float(
                ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS
            ),
            "direction_utility_trade_conviction_max_bad_path": float(
                ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH
            ),
            "direction_utility_trade_conviction_logit_margin": float(
                ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN
            ),
            "direction_utility_triad_ce_weight": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT),
            "direction_utility_triad_ce_min_gap_bps": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS),
            "direction_utility_triad_ce_min_utility_bps": float(
                ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS
            ),
            "direction_utility_triad_ce_max_bad_path": float(ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH),
            "direction_utility_triad_ce_class_weight_cap": float(
                ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP
            ),
            "hier_trade_global_prior_match_weight": float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT),
            "hier_trade_global_prior_match_tolerance": float(ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE),
            "hier_trade_global_prior_match_min_label_rate": float(
                ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
            ),
            "hier_slice_trade_prior_match_weight": float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT),
            "hier_slice_trade_prior_match_tolerance": float(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE),
            "hier_slice_trade_prior_match_min_label_rate": float(
                ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE
            ),
            "hier_slice_trade_prior_match_min_rows": int(ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS),
            "hier_slice_trade_accuracy_edge_weight": float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT),
            "hier_slice_trade_accuracy_edge_margin": float(ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN),
            "hier_flat_logit_margin_weight": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT),
            "hier_flat_logit_margin": float(ENTRY_HIER_FLAT_LOGIT_MARGIN),
            "hier_flat_logit_margin_min_label_rate": float(ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE),
            "hier_slice_flat_logit_margin_weight": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT),
            "hier_slice_flat_logit_margin": float(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN),
            "hier_slice_flat_logit_margin_min_label_rate": float(
                ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE
            ),
            "hier_slice_flat_logit_margin_min_rows": int(ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS),
            "hier_slice_side_ce_weight": float(ENTRY_HIER_SLICE_SIDE_CE_WEIGHT),
            "hier_slice_side_true_margin_weight": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT),
            "hier_slice_side_true_margin": float(ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN),
            "hier_slice_side_accuracy_edge_weight": float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT),
            "hier_slice_side_accuracy_edge_margin": float(ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN),
            "hier_slice_side_min_label_rate": float(ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE),
            "hier_slice_side_min_rows": int(ENTRY_HIER_SLICE_SIDE_MIN_ROWS),
            "hier_side_global_prior_match_weight": float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT),
            "hier_side_global_prior_match_tolerance": float(ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE),
            "hier_side_global_prior_match_min_label_rate": float(
                ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE
            ),
            "hier_slice_side_prior_match_weight": float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT),
            "hier_slice_side_prior_match_tolerance": float(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE),
            "hier_slice_side_prior_match_min_label_rate": float(
                ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE
            ),
            "hier_slice_side_prior_match_min_rows": int(ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS),
            "direction_flat_starvation_weight": float(ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT),
            "direction_flat_starvation_min_label_rate": float(ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE),
            "direction_flat_starvation_min_rows": int(ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS),
            "direction_flat_starvation_pred_fraction": float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION),
            "direction_flat_starvation_pred_floor": float(ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR),
            "direction_flat_starvation_logit_margin": float(ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN),
            "hierarchical_entry_heads_enabled": True,
            "side_validity_head_enabled": True,
            "trendline_rail_head_enabled": True,
            "trendline_rail_output_dim": 6,
            "trendline_rail_aux_weight": float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT),
            "trendline_rail_hand_written_direction_pressure": False,
            "hier_trade_weight": float(ENTRY_HIER_TRADE_WEIGHT),
            "hier_side_weight": float(ENTRY_HIER_SIDE_WEIGHT),
            "hier_utility_weight": float(ENTRY_HIER_UTILITY_WEIGHT),
            "hier_bad_path_weight": float(ENTRY_HIER_BAD_PATH_WEIGHT),
            "hier_mae_weight": float(ENTRY_HIER_MAE_WEIGHT),
            "hier_side_validity_weight": float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT),
            "hier_side_validity_min_utility_bps": float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS),
            "hier_side_validity_pos_weight_cap": float(ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP),
            "hier_structural_direction_target_rewrite": False,
            "hier_trade_pos_weight": float(hier_trade_pos_weight),
            "hier_bad_path_pos_weight": [float(hier_bad_path_pos_weight[0]), float(hier_bad_path_pos_weight[1])],
            "hier_side_utility_scale_bps": max(1.0, float(ENTRY_AUX_PATH_SCALE_BPS)),
            "hier_side_mae_scale_bps": max(1.0, float(ENTRY_AUX_MFE_SCALE_BPS)),
            "tradable_weight": float(ENTRY_AUX_TRADABLE_WEIGHT),
            "tradable_pos_weight": float(tradable_pos_weight),
            "bad_path_weight": float(ENTRY_AUX_BAD_PATH_WEIGHT),
            "bad_path_pos_weight": float(bad_path_pos_weight),
            "bad_path_quality_rank_weight": float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT),
            "bad_path_quality_rank_margin": float(ENTRY_BAD_PATH_QUALITY_RANK_MARGIN),
            "bad_path_quality_rank_quantile": float(ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE),
            "path_quality_rank_weight": float(ENTRY_PATH_QUALITY_RANK_WEIGHT),
            "path_quality_rank_margin": float(ENTRY_PATH_QUALITY_RANK_MARGIN),
            "path_quality_rank_quantile": float(ENTRY_PATH_QUALITY_RANK_QUANTILE),
            "path_weight": float(ENTRY_AUX_PATH_WEIGHT),
            "mfe_weight": float(ENTRY_AUX_MFE_WEIGHT),
            "specialist_gate_entropy_weight": float(ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT),
            "specialist_gate_balance_weight": float(ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT),
            "specialist_gate_min_mean": float(ENTRY_SPECIALIST_GATE_MIN_MEAN),
            "dead_long_ce_multiplier": float(ENTRY_DEAD_LONG_CE_MULTIPLIER),
            "dead_long_prob_penalty": float(ENTRY_DEAD_LONG_PROB_PENALTY),
            "teaser_long_ce_multiplier": float(ENTRY_TEASER_LONG_CE_MULTIPLIER),
            "teaser_long_prob_penalty": float(ENTRY_TEASER_LONG_PROB_PENALTY),
            "hard_neg_long_ce_multiplier": float(ENTRY_HARD_NEG_LONG_CE_MULTIPLIER),
            "hard_neg_long_prob_penalty": float(ENTRY_HARD_NEG_LONG_PROB_PENALTY),
            "clean_edge_weight": float(ENTRY_AUX_CLEAN_EDGE_WEIGHT),
            "clean_edge_pos_weight": float(clean_edge_pos_weight),
            "survival_weight": float(ENTRY_AUX_SURVIVAL_WEIGHT),
            "survival_pos_weight": float(survival_pos_weight),
            "clean_edge_ranking_weight": float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT),
            "clean_edge_ranking_margin": float(ENTRY_CLEAN_EDGE_RANKING_MARGIN),
            "selector_masked_aux": True,
            "symmetric_negatives": bool(ENTRY_SYMMETRIC_NEGATIVES),  # A7 2026-06-06: long==short
            "validation_objective_matches_train": True,
            "validation_objective_scope_note": (
                "training validation includes the active train loss family: direction, hierarchy, "
                "path/rank, specialist gate, all aux heads, MTF-direction, and future-path terms."
            ),
            "aux_selector_mode": "long_short_union" if ENTRY_SYMMETRIC_NEGATIVES else "long_only",
            "clean_edge_target_mode": "bidir" if ENTRY_SYMMETRIC_NEGATIVES else "long",
            "survival_target_mode": "bidir" if ENTRY_SYMMETRIC_NEGATIVES else "long",
            "bad_path_ce_in_direction_loss": bool(ENTRY_BAD_PATH_CE_MULTIPLIER > 1.0),
            "bad_path_prob_penalty_in_validation": bool(ENTRY_BAD_PATH_PROB_PENALTY > 0.0),
            "symmetric_short_prob_penalties": bool(ENTRY_SYMMETRIC_NEGATIVES),
            "symmetric_clean_edge_rank": bool(ENTRY_SYMMETRIC_NEGATIVES),
            "aux_regression_positive_only": True,
            "path_quality_rank_full_batch": bool(ENTRY_PATH_QUALITY_RANK_WEIGHT > 0.0),
            "active_heads": active_heads,
        },
        "secondary_training_controls": {
            "cost_sensitive_loss_enabled": bool(ENTRY_COST_SENSITIVE_ENABLED),
            "pred_balance_alpha": float(ENTRY_PRED_BALANCE_ALPHA),
            "tradable_pos_weight_cap": float(ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP),
        },
    }
    # Architecture reconstruction fields are duplicated exactly in the lock;
    # neither side may infer MTF layout or positive-scale semantics from the
    # other.
    lock["multi_tf"] = meta["multi_tf"]
    lock["tf_input_scale"] = meta["tf_input_scale"]
    export_contract_failures = _direction_decision_contract_export_failures(lock, meta)
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
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        m15_seq_dim=_mtf_feat_count,
        h1_seq_dim=_mtf_feat_count,
        h4_seq_dim=_mtf_feat_count,
        d1_seq_dim=_mtf_feat_count,
        m15_seq_len=_m15_len,
        h1_seq_len=multi_tf_seq_len,
        h4_seq_len=_h4_len,
        d1_seq_len=_d1_len,
        m5_seq_dim=_mtf_feat_count,
        m5_seq_len=multi_tf_seq_len,
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
        tf_input_scale_init_m5=tf_input_scale_init_m5,
        tf_input_scale_init_m15=tf_input_scale_init_m15,
        tf_input_scale_init_h1=tf_input_scale_init_h1,
        tf_input_scale_init_h4=tf_input_scale_init_h4,
        tf_input_scale_init_d1=tf_input_scale_init_d1,
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
        mtf_kwargs = {
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
            for tf in ("M5", "M15", "H1", "H4", "D1")
        }
        _ = model2(dummy_seq, dummy_snap, ctx_cat=dummy_cat, ctx_cont=dummy_cont, **mtf_kwargs)
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
    # permitted. Both seq_x and snap_x must prove the exact 513-field surface;
    # ctx_cont must prove all 142 fields; every MTF surface must be present/live.
    try:
        from gx1.audit.feature_liveness import assert_v10_batch_liveness, FeatureLivenessError
        _live_cc = list(ordered_ctx_cont_names)
        if len(_live_cc) != 142:
            raise FeatureLivenessError(
                f"[FEATURE_LIVENESS_CTX_CONTRACT] names={len(_live_cc)} expected=142"
            )
        _live_ds = train_ds
        if len(_live_ds) <= 0:
            raise FeatureLivenessError("[FEATURE_LIVENESS_EMPTY_TRAIN_SPLIT]")
        _snap_names = list(getattr(_live_ds, "signal_names", ()))
        if len(_snap_names) != 513:
            raise FeatureLivenessError(
                f"[FEATURE_LIVENESS_SIGNAL_CONTRACT] names={len(_snap_names)} expected=513"
            )
        if (
            bool(getattr(_live_ds, "_advanced", False))
            and hasattr(_live_ds, "_np_seq")
            and hasattr(_live_ds, "_np_ctx_cont")
            and hasattr(_live_ds, "_np_snap")
        ):
            _sample_rows = min(1024, len(_live_ds))
            _sample_idx = np.linspace(
                0,
                len(_live_ds) - 1,
                num=_sample_rows,
                dtype=np.int64,
            )
            _ab = {
                "seq_x": np.asarray(_live_ds._np_seq[_sample_idx], dtype=np.float32),
                "ctx_cont": np.asarray(_live_ds._np_ctx_cont[_sample_idx], dtype=np.float32),
                "snap_x": np.asarray(_live_ds._np_snap[_sample_idx], dtype=np.float32),
            }
            if getattr(_live_ds, "_multi_tf_feats", None):
                for _tf, _feats in _live_ds._multi_tf_feats.items():
                    _arr = np.asarray(_feats.attrs.get("feats_np"), dtype=np.float32)
                    if _arr.size:
                        _tf_rows = min(8192, int(_arr.shape[0]))
                        _tf_idx = np.linspace(
                            0,
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
        assert_v10_batch_liveness(_ab, ctx_cont_names=_live_cc,
                                  snap_names=_snap_names, raise_on_fail=True)
        log.info("[FEATURE_LIVENESS] post-export audit OK — exact seq513/ctx142 inputs are live")
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
def main() -> None:
    _enforce_canonical_train_env_contract()

    parser = argparse.ArgumentParser("ENTRY_V10_CTX exact model-native trainer")
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--profile", choices=("smoke", "candidate"), required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--dataset-run-id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=MODEL_NATIVE_SEQ_LEN)
    parser.add_argument("--train-manifest-json", type=Path, required=True)
    parser.add_argument("--val-manifest-json", type=Path, required=True)
    parser.add_argument("--test-manifest-json", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--out_bundle_dir", type=Path, required=True)
    parser.add_argument("--gx1-data", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the audited non-deterministic CUDA execution mode.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--m5-prebuilt-path", type=Path, required=True)
    parser.add_argument("--multi-tf-seq-len", type=int, default=96)
    parser.add_argument("--per-tf-seq-len-h4", type=int, default=0)
    parser.add_argument("--per-tf-seq-len-d1", type=int, default=0)
    parser.add_argument("--multi-tf-scale", type=float, default=0.5)
    parser.add_argument("--specialist-audit-json", type=Path, required=True)
    parser.add_argument(
        "--specialist-contract-mode",
        choices=(MODEL_NATIVE_CONTRACT_MODE,),
        required=True,
    )
    parser.add_argument("--specialist-num-layers", type=int, default=1)
    parser.add_argument("--specialist-fusion-scale", type=float, default=0.25)
    parser.add_argument("--grad-accum-steps", type=int, required=True)
    parser.add_argument("--tf-input-scale-init-m5", type=float, default=1.0)
    parser.add_argument("--tf-input-scale-init-m15", type=float, default=1.0)
    parser.add_argument("--tf-input-scale-init-h1", type=float, default=0.7)
    parser.add_argument("--tf-input-scale-init-h4", type=float, default=0.5)
    parser.add_argument("--tf-input-scale-init-d1", type=float, default=0.3)
    parser.add_argument("--subsample-rows", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    args = parser.parse_args()

    from gx1.contracts.entry_run_lineage_v1 import EntryRunLineageError, require_entry_run_id

    try:
        require_entry_run_id(args.run_id)
        require_entry_run_id(args.dataset_run_id)
    except EntryRunLineageError as exc:
        parser.error(str(exc))

    global _GRAD_CLIP_NORM, _WEIGHT_DECAY
    _GRAD_CLIP_NORM = float(args.grad_clip_norm)
    _WEIGHT_DECAY = float(args.weight_decay)
    _guard_no_rl()
    device = _resolve_device(args.device)
    log.info(
        "[CONFIG] seed=%d device=%s deterministic=%s grad_clip_norm=%.6f weight_decay=%.6f",
        args.seed,
        device,
        not args.fast,
        _GRAD_CLIP_NORM,
        _WEIGHT_DECAY,
    )

    _resolve_gx1_data(args.gx1_data)
    _manifests, parquets = _resolve_explicit_train_split_artifacts(
        train_manifest=args.train_manifest_json,
        val_manifest=args.val_manifest_json,
        test_manifest=args.test_manifest_json,
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        test_parquet=args.test_parquet,
        m5_prebuilt_path=args.m5_prebuilt_path,
        dataset_run_id=args.dataset_run_id,
        profile=args.profile,
    )
    train_parquet = parquets["train"]
    val_parquet = parquets["val"]
    test_parquet = parquets["test"]
    _log_label_distribution(test_parquet, split="test")

    run_train(
        train_parquet=train_parquet,
        train_manifest_path=_manifests["train"],
        val_parquet=val_parquet,
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
        deterministic=not args.fast,
        multi_tf_seq_len=args.multi_tf_seq_len,
        multi_tf_scale=args.multi_tf_scale,
        subsample_rows=args.subsample_rows,
        specialist_num_layers=int(args.specialist_num_layers),
        specialist_fusion_scale=float(args.specialist_fusion_scale),
        per_tf_seq_len_h4=int(args.per_tf_seq_len_h4),
        per_tf_seq_len_d1=int(args.per_tf_seq_len_d1),
        grad_accum_steps=int(args.grad_accum_steps),
        tf_input_scale_init_m5=float(args.tf_input_scale_init_m5),
        tf_input_scale_init_m15=float(args.tf_input_scale_init_m15),
        tf_input_scale_init_h1=float(args.tf_input_scale_init_h1),
        tf_input_scale_init_h4=float(args.tf_input_scale_init_h4),
        tf_input_scale_init_d1=float(args.tf_input_scale_init_d1),
        run_id=str(args.run_id),
        dataset_run_id=str(args.dataset_run_id),
        profile=str(args.profile),
    )


if __name__ == "__main__":
    main()
