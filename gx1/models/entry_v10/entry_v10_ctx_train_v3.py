#!/usr/bin/env python3
"""
Canonical ENTRY_V10_CTX trainer.

ONE UNIVERSE (STRICT):
- Signal bridge: versioned Entry signal bridge.
- Context: contract-driven ctx_cont/ctx_cat dimensions from signal_bridge_v3.
- No RL
- No legacy
- No fallback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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

# V2: switched from signal_bridge_v1 (7-dim seq) to signal_bridge_v3 (37-dim seq with SMC).
# The architecture itself is dimension-flexible — only the import names change.
from gx1.contracts.signal_bridge_v3 import (
    ORDERED_SEQ_FIELDS_V3 as SIGNAL_FIELDS,
    CONTRACT_SHA256_V3 as SIGNAL_BRIDGE_CONTRACT_SHA256,
    SEQ_SIGNAL_DIM_V3 as SEQ_SIGNAL_DIM,
    SNAP_SIGNAL_DIM_V3 as SNAP_SIGNAL_DIM,
    DEFAULT_SEQ_LEN_V3,
    CTX_CONT_DIM_V3,
    ORDERED_CTX_CAT_NAMES_V3,
)
from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
from gx1.time.session_detector import (
    get_session_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
    DIP_DIRECTIONS, DIP_HORIZONS, DIP_TARGETS, FORECAST_HORIZONS,
    TIMING_DIRECTIONS, TIMING_HORIZONS, TIMING_TARGETS,
    TAIL_RISK_DIRECTIONS, TAIL_RISK_HORIZONS, TAIL_RISK_QUANTILE,
    VOL_FORECAST_HORIZONS,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_CONTRACT_MODES,
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)


def dip_forecast_loss(out: dict, batch: dict, device) -> "torch.Tensor":
    """Shared loss for the dip-head (18, pinball quantile) + forecast-head (4,
    smooth_l1). Returns 0 if a head/its targets are absent (gated). Output layout
    MUST match the model: dip = dir×K×{dip_p50,dip_p90,recovery_p50}; recovery
    uses the mfe label, dip_p50/p90 use the mae_before_mfe label."""
    import torch
    total = torch.zeros((), device=device)
    dip_pred = out.get("dip_pred")
    if dip_pred is not None and "y_dip_mae_long_K12" in batch:
        tgts, qs = [], []
        for d in DIP_DIRECTIONS:
            for K in DIP_HORIZONS:
                for tgt in DIP_TARGETS:
                    if tgt.startswith("recovery"):
                        tgts.append(batch[f"y_dip_mfe_{d}_K{K}"]); qs.append(0.5)
                    else:  # dip_p50 / dip_p90
                        tgts.append(batch[f"y_dip_mae_{d}_K{K}"]); qs.append(0.9 if "p90" in tgt else 0.5)
        tgt = torch.stack(tgts, dim=1).to(device).float()          # (B, 18)
        q = torch.tensor(qs, device=device, dtype=tgt.dtype).view(1, -1)
        err = tgt - dip_pred.float()
        total = total + torch.maximum(q * err, (q - 1.0) * err).mean()
    fc_pred = out.get("forecast_pred")
    if fc_pred is not None and "y_forecast_ret_K1" in batch:
        fc_tgt = torch.stack([batch[f"y_forecast_ret_K{K}"] for K in FORECAST_HORIZONS], dim=1).to(device).float()
        total = total + torch.nn.functional.smooth_l1_loss(fc_pred.float(), fc_tgt)
    # ── dip-timing head (12, smooth_l1) — WHEN the dip bottoms / favorable peak ─
    timing_pred = out.get("timing_pred")
    if timing_pred is not None and "y_dip_bottom_frac_long_K12" in batch:
        t_tgts = []
        for d in TIMING_DIRECTIONS:
            for K in TIMING_HORIZONS:
                for tgt in TIMING_TARGETS:
                    t_tgts.append(batch[f"y_{tgt}_{d}_K{K}"])
        t_tgt = torch.stack(t_tgts, dim=1).to(device).float()          # (B, 12)
        total = total + torch.nn.functional.smooth_l1_loss(timing_pred.float(), t_tgt)
    # ── tail-risk head (6, pinball q=0.9) — worst adverse over full horizon ─────
    tail_pred = out.get("tail_risk_pred")
    if tail_pred is not None and "y_tail_mae_long_K12" in batch:
        tail_tgts = [batch[f"y_tail_mae_{d}_K{K}"]
                     for d in TAIL_RISK_DIRECTIONS for K in TAIL_RISK_HORIZONS]
        tail_tgt = torch.stack(tail_tgts, dim=1).to(device).float()    # (B, 6)
        q = float(TAIL_RISK_QUANTILE)
        err = tail_tgt - tail_pred.float()
        total = total + torch.maximum(q * err, (q - 1.0) * err).mean()
    # ── vol-forecast head (3, smooth_l1) — forward realized vol (bps) ───────────
    vol_pred = out.get("vol_forecast_pred")
    if vol_pred is not None and "y_vol_fwd_K12" in batch:
        vol_tgt = torch.stack([batch[f"y_vol_fwd_K{K}"] for K in VOL_FORECAST_HORIZONS], dim=1).to(device).float()
        total = total + torch.nn.functional.smooth_l1_loss(vol_pred.float(), vol_tgt)
    return total


# Aux-head regression target columns batched per sample (fallback 0.0). Must
# stay in sync with the builder's _HEAD_TARGET_COLS and the model's heads.
_DIP_FORECAST_TARGET_COLS = (
    [f"y_dip_mae_{d}_K{K}" for d in DIP_DIRECTIONS for K in DIP_HORIZONS]
    + [f"y_dip_mfe_{d}_K{K}" for d in DIP_DIRECTIONS for K in DIP_HORIZONS]
    + [f"y_forecast_ret_K{K}" for K in FORECAST_HORIZONS]
    + [f"y_{tgt}_{d}_K{K}" for d in TIMING_DIRECTIONS for K in TIMING_HORIZONS for tgt in TIMING_TARGETS]
    + [f"y_tail_mae_{d}_K{K}" for d in TAIL_RISK_DIRECTIONS for K in TAIL_RISK_HORIZONS]
    + [f"y_vol_fwd_K{K}" for K in VOL_FORECAST_HORIZONS]
)

# -----------------------------------------------------------------------------
# RL / legacy guard (fail-fast)
# -----------------------------------------------------------------------------
def _guard_no_rl() -> None:
    """Hard-fail if gx1.rl or legacy was imported."""
    allowed_legacy_modules = {"gx1.runtime.entry_next_edge_legacy_guard"}
    for mod in list(sys.modules.keys()):
        if mod == "gx1.rl" or mod.startswith("gx1.rl."):
            raise RuntimeError(
                "[ENTRY_V10_CTX_RL_FORBIDDEN] gx1.rl must not be imported. "
                f"Found: {mod}"
            )
        if "legacy" in mod and mod.startswith("gx1.") and mod not in allowed_legacy_modules:
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


_ENTRY_ALLOWED_COMPAT_STATE_KEYS = {
    "head_early_move.weight",
    "head_early_move.bias",
    "head_quality.weight",
    "head_quality.bias",
    "head_bad_path.weight",
    "head_bad_path.bias",
    "head_clean_edge.weight",
    "head_clean_edge.bias",
    "head_survival.weight",
    "head_survival.bias",
    # V10 v3+ aux heads (Targets 1-4) — optional, present only when
    # bundle was trained with --enable-*-head flags.
    "head_tf_agreement.weight",
    "head_tf_agreement.bias",
    "head_path_quality_log_var.weight",
    "head_path_quality_log_var.bias",
    "head_position_size.weight",
    "head_position_size.bias",
    "head_hold_horizon.weight",
    "head_hold_horizon.bias",
    # XAU direction repair challengers (opt-in).
    "head_anchor_gate.weight",
    "head_anchor_gate.bias",
    "head_trade.weight",
    "head_trade.bias",
    "head_side.weight",
    "head_side.bias",
    "head_side_utility.weight",
    "head_side_utility.bias",
    "head_side_bad_path.weight",
    "head_side_bad_path.bias",
    "head_side_mae.weight",
    "head_side_mae.bias",
    "head_trendline_rail.weight",
    "head_trendline_rail.bias",
}


def _load_entry_model_state_compat(model: nn.Module, state: Dict[str, Any], *, label: str) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(getattr(incompatible, "missing_keys", []) or [])
    unexpected = set(getattr(incompatible, "unexpected_keys", []) or [])
    unexpected_active = unexpected - _ENTRY_ALLOWED_COMPAT_STATE_KEYS
    if unexpected_active:
        raise RuntimeError(f"[ENTRY_V10_CTX_UNEXPECTED_KEYS] {label}: {sorted(unexpected_active)}")
    missing_active = missing - _ENTRY_ALLOWED_COMPAT_STATE_KEYS
    if missing_active:
        raise RuntimeError(f"[ENTRY_V10_CTX_MISSING_KEYS] {label}: {sorted(missing_active)}")

# -----------------------------------------------------------------------------
# SHORT collapse countermeasures (training-only)
# Canonical lane keeps these parked unless we have clear evidence they help more
# than they complicate the recipe.
# -----------------------------------------------------------------------------
SHORT_CLASS_WEIGHT = float(_env_str("ENTRY_SHORT_CLASS_WEIGHT", "0.90"))
ENTRY_LONG_CLASS_WEIGHT_CAP = float(_env_str("ENTRY_LONG_CLASS_WEIGHT_CAP", "20.0"))
ENTRY_SHORT_CLASS_WEIGHT_CAP = float(_env_str("ENTRY_SHORT_CLASS_WEIGHT_CAP", "8.0"))
ENTRY_FLAT_CLASS_WEIGHT_FLOOR = float(_env_str("ENTRY_FLAT_CLASS_WEIGHT_FLOOR", "1.0"))
XGB_SHORT_LEAD_MARGIN = float(_env_str("ENTRY_XGB_SHORT_LEAD_MARGIN", "0.0"))
XGB_SHORT_LONG_PENALTY = float(_env_str("ENTRY_XGB_SHORT_LONG_PENALTY", "0.0"))

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
ENTRY_RESIDUAL_SIDE_BIAS_ALPHA = float(_env_str("ENTRY_RESIDUAL_SIDE_BIAS_ALPHA", "0.0"))
ENTRY_DIRECTION_CE_SCALE = float(_env_str("ENTRY_DIRECTION_CE_SCALE", "1.30"))
ENTRY_TAIL_DIRECTION_CE_WEIGHT = float(_env_str("ENTRY_TAIL_DIRECTION_CE_WEIGHT", "0.0"))
ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE = float(_env_str("ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE", "0.70"))
ENTRY_TAIL_DIRECTION_MIN_BATCH = int(_env_str("ENTRY_TAIL_DIRECTION_MIN_BATCH", "8"))
ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT = float(_env_str("ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT", "0.0"))
ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT = float(_env_str("ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT", "0.0"))
ENTRY_SPECIALIST_GATE_MIN_MEAN = float(_env_str("ENTRY_SPECIALIST_GATE_MIN_MEAN", "0.01"))
# Forceful MTF→direction (2026-06-06): aux CE on the multi-TF direction logits vs
# the direction label, forcing the 5 multi-TF streams to predict direction.
# Env-overridable; default 0.3 (secondary to the main direction CE).
ENTRY_MTF_DIR_AUX_WEIGHT = float(_env_str("ENTRY_MTF_DIR_AUX_WEIGHT", "0.30"))
# Checkpoint-selection monitor (diagnosis fix #3, 2026-06-06): the early-stop / best-checkpoint
# was selected on TOTAL multi-head val loss, which saves the aux-overfit epoch (the cement froze
# at epoch-2 = the aux optimum, NOT the best-direction epoch). "dir_acc" instead keeps the epoch
# with the highest direction validation accuracy (the metric the chain actually acts on).
# Default "val_loss" = bit-identical to the historical behavior.
ENTRY_CKPT_MONITOR = _env_str("GX1_V10_CKPT_MONITOR", "val_loss").strip().lower()
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
ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT = float(_env_str("ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT", "0.0"))
ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS = float(_env_str("ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS", "15.0"))
ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN = float(_env_str("ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN", "0.10"))
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT", "0.0")
)
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS = float(
    _env_str("ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS", "15.0")
)
ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN = float(
    _env_str("ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN", "0.10")
)
ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT = float(
    _env_str("ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT", "0.0")
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
ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT = float(_env_str("ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT", "0.0"))
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
ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION = int(
    float(_env_str("ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION", "0"))
)
ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP = float(_env_str("ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP", "0.0"))
ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL = int(
    float(_env_str("ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL", "0"))
)
ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE = int(
    float(_env_str("ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE", "0"))
)
ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION = (
    _env_str("ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION", "logprob").strip().lower()
)
ENTRY_HIER_PUBLIC_TRADE_HEAD = int(float(_env_str("ENTRY_HIER_PUBLIC_TRADE_HEAD", "0")))
ENTRY_HIER_PUBLIC_SIDE_HEAD = int(float(_env_str("ENTRY_HIER_PUBLIC_SIDE_HEAD", "0")))
ENTRY_HIER_CTX_PRIOR_ADAPTER = int(float(_env_str("ENTRY_HIER_CTX_PRIOR_ADAPTER", "0")))
ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE = float(_env_str("ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE", "0.0"))
ENTRY_HIER_CTX_DIRECTION_CALIBRATION = int(
    float(_env_str("ENTRY_HIER_CTX_DIRECTION_CALIBRATION", "0"))
)
ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE = float(
    _env_str("ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE", "0.0")
)
ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP = float(
    _env_str("ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP", "0.0")
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
# Timing loss (early adverse move penalty)
# -----------------------------------------------------------------------------
ENTRY_TIMING_TARGET_BPS = float(_env_str("ENTRY_TIMING_TARGET_BPS", "3.0"))
ENTRY_TIMING_LOSS_SCALE = float(_env_str("ENTRY_TIMING_LOSS_SCALE", "0.0"))

# -----------------------------------------------------------------------------
# Auxiliary losses (use existing dataset targets)
# -----------------------------------------------------------------------------
# Canonical lane keeps only the auxiliary heads that directly support runtime
# gates (tradable, mfe_first_n, path_quality). Early/quality-score/bad-path stay parked.
ENTRY_AUX_EARLY_WEIGHT = float(_env_str("ENTRY_AUX_EARLY_WEIGHT", "0.0"))
ENTRY_AUX_QUALITY_WEIGHT = float(_env_str("ENTRY_AUX_QUALITY_WEIGHT", "0.0"))
ENTRY_AUX_PATH_WEIGHT = float(_env_str("ENTRY_AUX_PATH_WEIGHT", "0.90"))
ENTRY_AUX_MFE_WEIGHT = float(_env_str("ENTRY_AUX_MFE_WEIGHT", "0.25"))
ENTRY_AUX_TRADABLE_WEIGHT = float(_env_str("ENTRY_AUX_TRADABLE_WEIGHT", "1.15"))
# Canonical lane keeps bad-path parked until it shows clean incremental value.
ENTRY_AUX_BAD_PATH_WEIGHT = float(_env_str("ENTRY_AUX_BAD_PATH_WEIGHT", "0.0"))
ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP = float(_env_str("ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP", "20.0"))
ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT = float(_env_str("ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT", "0.0"))
ENTRY_BAD_PATH_QUALITY_RANK_MARGIN = float(_env_str("ENTRY_BAD_PATH_QUALITY_RANK_MARGIN", "0.20"))
ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE = float(_env_str("ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE", "0.25"))
ENTRY_PATH_QUALITY_RANK_WEIGHT = float(_env_str("ENTRY_PATH_QUALITY_RANK_WEIGHT", "0.0"))
ENTRY_PATH_QUALITY_RANK_MARGIN = float(_env_str("ENTRY_PATH_QUALITY_RANK_MARGIN", "0.20"))
ENTRY_PATH_QUALITY_RANK_QUANTILE = float(_env_str("ENTRY_PATH_QUALITY_RANK_QUANTILE", "0.25"))
# Scale bps targets to keep regression losses in a stable range
ENTRY_AUX_QUALITY_SCALE_BPS = float(_env_str("ENTRY_AUX_QUALITY_SCALE_BPS", "50.0"))
ENTRY_AUX_PATH_SCALE_BPS = float(_env_str("ENTRY_AUX_PATH_SCALE_BPS", "50.0"))
ENTRY_AUX_MFE_SCALE_BPS = float(_env_str("ENTRY_AUX_MFE_SCALE_BPS", "20.0"))
ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP = float(_env_str("ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP", "12.0"))
ENTRY_HARD_NEG_LONG_CE_MULTIPLIER = float(_env_str("ENTRY_HARD_NEG_LONG_CE_MULTIPLIER", "1.35"))
ENTRY_HARD_NEG_LONG_PROB_PENALTY = float(_env_str("ENTRY_HARD_NEG_LONG_PROB_PENALTY", "0.20"))
ENTRY_DEAD_LONG_CE_MULTIPLIER = float(_env_str("ENTRY_DEAD_LONG_CE_MULTIPLIER", "1.80"))
ENTRY_DEAD_LONG_PROB_PENALTY = float(_env_str("ENTRY_DEAD_LONG_PROB_PENALTY", "0.40"))
ENTRY_TEASER_LONG_CE_MULTIPLIER = float(_env_str("ENTRY_TEASER_LONG_CE_MULTIPLIER", "1.35"))
ENTRY_TEASER_LONG_PROB_PENALTY = float(_env_str("ENTRY_TEASER_LONG_PROB_PENALTY", "0.16"))
# 2026-06-03 (vedtak v10_symmetric_negatives_20260603): mirror the LONG hard-negative
# penalty stack onto the SHORT side (probs[:,1] + y_*_negative_short), reusing the SAME
# magnitudes (equal weight). Default OFF = bit-identical to cement (the LONG-only asymmetry
# that suppressed the LONG side). Enable for the retrain with ENTRY_SYMMETRIC_NEGATIVES=1
# (+ GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES=1, since it's a deliberate non-cement recipe).
ENTRY_SYMMETRIC_NEGATIVES = _env_str("ENTRY_SYMMETRIC_NEGATIVES", "0") in {"1", "true", "yes", "on"}
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
ENTRY_HIER_TRADE_WEIGHT = float(_env_str("ENTRY_HIER_TRADE_WEIGHT", "0.85"))
ENTRY_HIER_SIDE_WEIGHT = float(_env_str("ENTRY_HIER_SIDE_WEIGHT", "0.85"))
ENTRY_HIER_UTILITY_WEIGHT = float(_env_str("ENTRY_HIER_UTILITY_WEIGHT", "0.20"))
ENTRY_HIER_BAD_PATH_WEIGHT = float(_env_str("ENTRY_HIER_BAD_PATH_WEIGHT", "0.35"))
ENTRY_HIER_MAE_WEIGHT = float(_env_str("ENTRY_HIER_MAE_WEIGHT", "0.15"))
ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP = float(_env_str("ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP", "20.0"))
ENTRY_HIER_LEGACY_CE_MULT = float(_env_str("ENTRY_HIER_LEGACY_CE_MULT", "0.35"))
ENTRY_HIER_SIDE_VALIDITY_WEIGHT = float(_env_str("ENTRY_HIER_SIDE_VALIDITY_WEIGHT", "0.0"))
ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS = float(_env_str("ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS", "10.0"))
ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP = float(_env_str("ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP", "20.0"))
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
ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT = float(
    _env_str("ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT", "0.0")
)
ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT = float(
    _env_str("ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT", "0.0")
)
ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE = float(
    _env_str("ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE", str(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
)
ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS = int(
    float(_env_str("ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS", str(ENTRY_DIRECTION_SLICE_MIN_ROWS)))
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
ENTRY_HIER_POCKET_ABSTAIN_WEIGHT = float(_env_str("ENTRY_HIER_POCKET_ABSTAIN_WEIGHT", "0.0"))
ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT = float(_env_str("ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT", "0.0"))
ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS = float(_env_str("ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS", "10.0"))
ENTRY_TRENDLINE_RAIL_AUX_WEIGHT = float(_env_str("ENTRY_TRENDLINE_RAIL_AUX_WEIGHT", "0.25"))
ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT = float(_env_str("ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT", "0.20"))
ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT = float(
    _env_str(
        "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT",
        str(ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT),
    )
)
ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT = float(
    _env_str(
        "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT",
        str(ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT),
    )
)
ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT = float(_env_str("ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT", "0.0"))
ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT = float(_env_str("ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT", "0.0"))
ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT = float(_env_str("ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT", "0.0"))
ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT = float(_env_str("ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT", "0.0"))
ENTRY_TRENDLINE_RAIL_MARGIN = float(_env_str("ENTRY_TRENDLINE_RAIL_MARGIN", "0.50"))
ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS = float(_env_str("ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS", "5.0"))

# -----------------------------------------------------------------------------
# Micro features (ctx_cont extension)
# -----------------------------------------------------------------------------
MICRO_FEATURE_NAMES = [
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
]
SWING_FEATURE_NAMES = [
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
]
EXT_CTX_FEATURE_NAMES = list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES)

# -----------------------------------------------------------------------------
# V_NEXT session-context extension (ctx_cont +5)
# Canonical default is V_NEXT (CTX21). Explicitly set GX1_CTX_CONTRACT=V_CURRENT
# only for legacy/debug use. Training without this env set now defaults to CTX21.
# -----------------------------------------------------------------------------
_CTX_CONTRACT_MODE = _env_str("GX1_CTX_CONTRACT", "V_NEXT").upper()
V_NEXT_EXTRA_CTX_CONT = [
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
]

def _is_vnext() -> bool:
    return _CTX_CONTRACT_MODE == "V_NEXT"

def _expected_ctx_cont_dim() -> int:
    # V2+: signal_bridge_v3 exposes the active ctx_cont dim through CTX_CONT_DIM_V3.
    # V_NEXT (legacy) was 21. V_BASE (legacy) was 16.
    from gx1.contracts.signal_bridge_v3 import CTX_CONT_DIM_V3
    return int(CTX_CONT_DIM_V3)

def _expected_ctx_cat_dim() -> int:
    # R4 (2026-06-04): ctx_cat is contract-driven, mirroring _expected_ctx_cont_dim — no
    # hardcoded 6. signal_bridge_v3: CTX_CAT_DIM_V3 = 5 (GX1_REGIME_V4=1, trend_regime_id
    # dropped — continuous D1_dist + 16 multi-TF REGIME_V4 carry trend) or 6 (cement,
    # GX1_REGIME_V4=0). The stale signal_bridge_v1 anchor (frozen 6) is NOT consulted here.
    from gx1.contracts.signal_bridge_v3 import CTX_CAT_DIM_V3
    return int(CTX_CAT_DIM_V3)

def _build_ordered_ctx_cont_names(ctx_cont_dim: int, base_names: List[str]) -> List[str]:
    try:
        from gx1.contracts.signal_bridge_v3 import ORDERED_CTX_CONT_NAMES_V3
        if int(ctx_cont_dim) == len(ORDERED_CTX_CONT_NAMES_V3):
            return list(ORDERED_CTX_CONT_NAMES_V3)
    except Exception:
        pass
    ordered = list(base_names)
    if ctx_cont_dim > len(ordered):
        ordered = ordered + list(EXT_CTX_FEATURE_NAMES)
    if _is_vnext() and ctx_cont_dim > len(ordered):
        ordered = ordered + list(V_NEXT_EXTRA_CTX_CONT)
    return ordered

# -----------------------------------------------------------------------------
# Anchored ENTRY (residual over XGB signal7 probs)
# -----------------------------------------------------------------------------
# Keep the canonical anchor mix unchanged for this bad-path candidate so replay
# deltas reflect the new adverse-first supervision rather than a different anchor.
ENTRY_RESIDUAL_SCALE = float(_env_str("ENTRY_RESIDUAL_SCALE", "0.35"))
ENTRY_ANCHOR_EPS = float(_env_str("ENTRY_ANCHOR_EPS", "1e-6"))

_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS: Dict[str, str] = {
    "ENTRY_SHORT_CLASS_WEIGHT": "0.90",
    "ENTRY_LONG_CLASS_WEIGHT_CAP": "20.0",
    "ENTRY_SHORT_CLASS_WEIGHT_CAP": "8.0",
    "ENTRY_FLAT_CLASS_WEIGHT_FLOOR": "1.0",
    "ENTRY_XGB_SHORT_LEAD_MARGIN": "0.0",
    "ENTRY_XGB_SHORT_LONG_PENALTY": "0.0",
    "ENTRY_COST_SENSITIVE_LOSS": "1",
    "ENTRY_COST_SENSITIVE_SCALE": "0.25",
    "ENTRY_COST_LONG_TO_SHORT": "2.00",
    "ENTRY_COST_LONG_TO_FLAT": "0.45",
    "ENTRY_COST_SHORT_TO_LONG": "2.00",  # 2026-05-26: symmetrized (was 3.00, anti-long)
    "ENTRY_COST_SHORT_TO_FLAT": "0.45",
    "ENTRY_COST_FLAT_TO_LONG": "1.60",   # 2026-05-26: symmetrized (was 2.75, anti-long)
    "ENTRY_COST_FLAT_TO_SHORT": "1.60",
    "ENTRY_PRED_BALANCE_ALPHA": "0.0",
    "ENTRY_PRED_BALANCE_TARGET": "label",
    "ENTRY_PRED_BALANCE_CLASS_WEIGHTS": "1.0,1.0,1.0",
    "ENTRY_RESIDUAL_SIDE_BIAS_ALPHA": "0.0",
    "ENTRY_DIRECTION_CE_SCALE": "1.30",
    "ENTRY_TAIL_DIRECTION_CE_WEIGHT": "0.0",
    "ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE": "0.70",
    "ENTRY_TAIL_DIRECTION_MIN_BATCH": "8",
    "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT": "0.0",
    "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL": "0.0",
    "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE": "0.0",
    "ENTRY_CKPT_DIRECTION_SLICE_GUARD": "0",
    "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT": "0.0",
    "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION": "0.0",
    "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR": "0.0",
    "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE": "1.0",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT": "0.0",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION": "0.0",
    "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR": "0.0",
    "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES": "0,1,2,3,4",
    "ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR": "0.30",
    "ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN": "0.10",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN": "0.02",
    "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN": "0.02",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION": "mean",
    "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER": "0",
    "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE": "0",
    "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS": "6",
    "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT": "0.0",
    "ENTRY_DIRECTION_VS_FLAT_MARGIN": "0.0",
    "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT": "0.0",
    "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN": "0.10",
    "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT": "0.0",
    "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN": "0.10",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT": "0.0",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS": "0.0",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH": "0.50",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN": "0.10",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT": "0.0",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS": "0.0",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH": "0.50",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP": "4.0",
    "ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION": "0",
    "ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP": "0.0",
    "ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL": "0",
    "ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE": "0",
    "ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION": "logprob",
    "ENTRY_HIER_PUBLIC_TRADE_HEAD": "0",
    "ENTRY_HIER_PUBLIC_SIDE_HEAD": "0",
    "ENTRY_HIER_CTX_PRIOR_ADAPTER": "0",
    "ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE": "0.0",
    "ENTRY_HIER_CTX_DIRECTION_CALIBRATION": "0",
    "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE": "0.0",
    "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP": "0.0",
    "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT": "0.0",
    "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS": "8",
    "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION": "0.50",
    "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR": "0.10",
    "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN": "0.10",
    "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT": "0.0",
    "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT": "0.0",
    "ENTRY_SPECIALIST_GATE_MIN_MEAN": "0.01",
    "ENTRY_TIMING_TARGET_BPS": "3.0",
    "ENTRY_TIMING_LOSS_SCALE": "0.0",
    "ENTRY_AUX_EARLY_WEIGHT": "0.0",
    "ENTRY_AUX_QUALITY_WEIGHT": "0.0",
    "ENTRY_AUX_PATH_WEIGHT": "0.90",
    "ENTRY_AUX_MFE_WEIGHT": "0.25",
    "ENTRY_AUX_TRADABLE_WEIGHT": "1.15",
    "ENTRY_AUX_BAD_PATH_WEIGHT": "0.0",
    "ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP": "20.0",
    "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT": "0.0",
    "ENTRY_BAD_PATH_QUALITY_RANK_MARGIN": "0.20",
    "ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE": "0.25",
    "ENTRY_PATH_QUALITY_RANK_WEIGHT": "0.0",
    "ENTRY_PATH_QUALITY_RANK_MARGIN": "0.20",
    "ENTRY_PATH_QUALITY_RANK_QUANTILE": "0.25",
    "ENTRY_AUX_QUALITY_SCALE_BPS": "50.0",
    "ENTRY_AUX_PATH_SCALE_BPS": "50.0",
    "ENTRY_AUX_MFE_SCALE_BPS": "20.0",
    "ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP": "12.0",
    "ENTRY_HARD_NEG_LONG_CE_MULTIPLIER": "1.35",
    "ENTRY_HARD_NEG_LONG_PROB_PENALTY": "0.20",
    "ENTRY_DEAD_LONG_CE_MULTIPLIER": "1.80",
    "ENTRY_DEAD_LONG_PROB_PENALTY": "0.40",
    "ENTRY_TEASER_LONG_CE_MULTIPLIER": "1.35",
    "ENTRY_TEASER_LONG_PROB_PENALTY": "0.16",
    "ENTRY_SYMMETRIC_NEGATIVES": "0",
    "ENTRY_BAD_PATH_CE_MULTIPLIER": "1.50",
    "ENTRY_BAD_PATH_PROB_PENALTY": "0.24",
    "ENTRY_AUX_CLEAN_EDGE_WEIGHT": "0.45",
    "ENTRY_AUX_SURVIVAL_WEIGHT": "0.10",
    "ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP": "16.0",
    "ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP": "10.0",
    "ENTRY_CLEAN_EDGE_RANKING_WEIGHT": "0.25",
    "ENTRY_CLEAN_EDGE_RANKING_MARGIN": "0.12",
    "ENTRY_HIER_TRADE_WEIGHT": "0.85",
    "ENTRY_HIER_SIDE_WEIGHT": "0.85",
    "ENTRY_HIER_UTILITY_WEIGHT": "0.20",
    "ENTRY_HIER_BAD_PATH_WEIGHT": "0.35",
    "ENTRY_HIER_MAE_WEIGHT": "0.15",
    "ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP": "20.0",
    "ENTRY_HIER_LEGACY_CE_MULT": "0.35",
    "ENTRY_HIER_SIDE_VALIDITY_WEIGHT": "0.0",
    "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS": "10.0",
    "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP": "20.0",
    "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT": "0.0",
    "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN": "0.02",
    "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT": "0.0",
    "ENTRY_HIER_FLAT_LOGIT_MARGIN": "0.10",
    "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN": "0.10",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS": "8",
    "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT": "0.0",
    "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS": "8",
    "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN": "0.10",
    "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN": "0.02",
    "ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_SIDE_MIN_ROWS": "8",
    "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT": "0.0",
    "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT": "0.0",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_HIER_POCKET_ABSTAIN_WEIGHT": "0.0",
    "ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT": "0.0",
    "ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS": "10.0",
    "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT": "0.25",
    "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT": "0.20",
    "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT": "0.20",
    "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT": "0.20",
    "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT": "0.0",
    "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT": "0.0",
    "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT": "0.0",
    "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT": "0.0",
    "ENTRY_TRENDLINE_RAIL_MARGIN": "0.50",
    "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS": "5.0",
    "GX1_CTX_CONTRACT": "V_NEXT",
    "ENTRY_RESIDUAL_SCALE": "0.35",
    "ENTRY_ANCHOR_EPS": "1e-6",
}


def _enforce_canonical_train_env_contract() -> None:
    """
    Canonical ENTRY training must not be silently steered by ad-hoc env knobs.
    Non-canonical experimentation may opt in explicitly.
    """
    allow_noncanonical = _env_str("GX1_NON_CANONICAL_DIAGNOSTIC", "0") in {"1", "true", "yes", "on"} or _env_str(
        "GX1_ENTRY_ALLOW_TRAIN_ENV_OVERRIDES", "0"
    ) in {"1", "true", "yes", "on"}
    if allow_noncanonical:
        log.warning("[ENTRY_CANONICAL_TRAIN_ENV] non-canonical env overrides explicitly enabled")
        return

    tripped: List[str] = []
    for name, default in _CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS.items():
        if name not in os.environ:
            continue
        actual = str(os.environ.get(name, "")).strip()
        if actual != str(default):
            tripped.append(f"{name}={actual!r} (default={default!r})")
    if tripped:
        raise RuntimeError(
            "[ENTRY_CANONICAL_TRAIN_ENV_FORBIDDEN] canonical entry training forbids non-default env overrides: "
            + ", ".join(sorted(tripped))
        )

# --- Determinism utilities (SSoT training API) ---------------------------------
# NOTE: These functions are part of the stable training-module API.
# The wrapper (gx1/scripts/train_entry_v10_ctx_depth_ladder.py) expects them.
# No fallbacks, no silent behavior changes—just deterministic seeding + thread limits.
import os as _os_det
import random as _random_det
from typing import Optional as _OptionalDet

import numpy as _np_det
import torch as _torch_det


def set_seed(seed: int) -> None:
    """
    Deterministic seeding for ENTRY_V10_CTX training.

    This is intentionally minimal and stable. Do not add "smart defaults" here.
    The caller decides the seed value.
    """
    if seed is None:
        raise ValueError("seed must be an int (not None)")

    # Python / NumPy
    _os_det.environ["PYTHONHASHSEED"] = str(int(seed))
    _random_det.seed(int(seed))
    _np_det.random.seed(int(seed))

    # Torch
    _torch_det.manual_seed(int(seed))
    if _torch_det.cuda.is_available():
        _torch_det.cuda.manual_seed_all(int(seed))

    # Determinism knobs (best-effort; no silent fallback logic)
    try:
        _torch_det.backends.cudnn.deterministic = True
        _torch_det.backends.cudnn.benchmark = False
    except Exception:
        # Some builds may not expose these flags; do not hard-fail.
        pass

    # PyTorch deterministic algorithms (may raise if op not supported; keep best-effort)
    try:
        _torch_det.use_deterministic_algorithms(True)
    except Exception:
        pass


def set_thread_limits(threads: int = 1) -> None:
    """
    Limit CPU thread usage for deterministic / reproducible runs (TRUTH doctrine).

    Best-effort: do not hard-fail on environments that do not support all settings.
    """
    if threads is None:
        raise ValueError("threads must be an int (not None)")
    t = int(threads)
    if t <= 0:
        raise ValueError(f"threads must be >= 1, got {t}")

    # Common BLAS/OpenMP knobs
    _os_det.environ["OMP_NUM_THREADS"] = str(t)
    _os_det.environ["MKL_NUM_THREADS"] = str(t)
    _os_det.environ["NUMEXPR_NUM_THREADS"] = str(t)
    _os_det.environ["OPENBLAS_NUM_THREADS"] = str(t)
    _os_det.environ["VECLIB_MAXIMUM_THREADS"] = str(t)

    # Torch thread limits
    try:
        _torch_det.set_num_threads(t)
    except Exception:
        pass

    try:
        _torch_det.set_num_interop_threads(t)
    except Exception:
        pass
# --- end determinism utilities -------------------------------------------------

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


def _build_active_head_names(
    *,
    enable_tf_agreement_head: bool,
    enable_path_quality_variance_head: bool,
    enable_position_size_head: bool,
    enable_hold_horizon_head: bool,
    enable_mtf_direction_head: bool,
    enable_dip_head: bool,
    enable_forecast_head: bool,
    enable_timing_head: bool,
    enable_tail_risk_head: bool,
    enable_vol_forecast_head: bool,
    enable_anchor_gate: bool = False,
    enable_hierarchical_entry_heads: bool = False,
    enable_hierarchical_public_trade_head: bool = False,
    enable_hierarchical_public_side_head: bool = False,
    enable_hierarchical_ctx_prior_adapter: bool = False,
    enable_hierarchical_ctx_direction_calibration: bool = False,
    enable_side_validity_head: bool = False,
    enable_trendline_rail_head: bool = False,
) -> List[str]:
    heads = [
        "direction",
        "tradable",
        "path_quality",
        "mfe_first_n",
        "bad_path",
        "clean_edge",
        "survival",
    ]
    optional_heads = [
        ("tf_agreement", enable_tf_agreement_head),
        ("path_quality_log_var", enable_path_quality_variance_head),
        ("position_size", enable_position_size_head),
        ("hold_horizon", enable_hold_horizon_head),
        ("mtf_direction", enable_mtf_direction_head),
        ("dip", enable_dip_head),
        ("forecast", enable_forecast_head),
        ("timing", enable_timing_head),
        ("tail_risk", enable_tail_risk_head),
        ("vol_forecast", enable_vol_forecast_head),
        ("anchor_gate", enable_anchor_gate),
        ("trade_side_hierarchy", enable_hierarchical_entry_heads),
        ("hierarchical_public_trade_head", enable_hierarchical_public_trade_head),
        ("hierarchical_public_side_head", enable_hierarchical_public_side_head),
        ("hierarchical_ctx_prior_adapter", enable_hierarchical_ctx_prior_adapter),
        ("hierarchical_ctx_direction_calibration", enable_hierarchical_ctx_direction_calibration),
        ("side_validity", enable_side_validity_head),
        ("trendline_rail", enable_trendline_rail_head),
    ]
    heads.extend(name for name, enabled in optional_heads if bool(enabled))
    return heads


# V12.2: grad-clip norm + weight-decay set at runtime via CLI flag. Module-level
# so we don't have to thread through 6 layers of function args.
_GRAD_CLIP_NORM: float = 1.0
_WEIGHT_DECAY: float = 1e-5
DEFAULT_SPECIALIST_AUDIT_JSON = Path(
    "/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/"
    "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
)


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
    """Extract multi-TF tensors from batch if present (V12.2). Returns {} for v3 batches.
    Centralized helper so each model() call site doesn't need its own conditional logic.

    Includes seq_m5 even though V10's base sequence IS M5 — V10's forward
    declares seq_m5 as ignored kwarg for symmetry with V3 (which DOES use M5).
    Same helper is used by V3 trainer (where M5 is needed).
    """
    out: Dict[str, torch.Tensor] = {}
    for key in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1"):
        if key in batch:
            t = batch[key]
            if hasattr(t, "to"):
                t = t.to(device)
            out[key] = t
    return out


def _load_specialist_fusion_contract(
    audit_json: Optional[Path],
    *,
    expected_signal_dim: int,
    contract_mode: str = "foundation_seq146",
) -> tuple[Dict[str, list[int]], Dict[str, Any]]:
    normalized_contract_mode = str(contract_mode or "foundation_seq146").strip()
    required_training_specialists = required_training_specialists_for_mode(normalized_contract_mode)
    expected_model_contract = specialist_model_contract_for_mode(normalized_contract_mode)
    path = Path(audit_json or DEFAULT_SPECIALIST_AUDIT_JSON).expanduser().resolve()
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
    observed_contract_mode = str(report.get("contract_mode") or "foundation_seq146").strip()
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
    blocked = {"neutral_bridge_anchor", "unmapped"}
    indices: Dict[str, list[int]] = {}
    excluded_groups: Dict[str, int] = {}
    for name, values in raw.items():
        key = str(name)
        idx = sorted({int(v) for v in list(values or [])})
        if key in blocked or key not in trainable:
            if idx:
                excluded_groups[key] = int(len(idx))
            continue
        if not idx:
            continue
        if min(idx) < 0 or max(idx) >= int(expected_signal_dim):
            raise RuntimeError(f"[SPECIALIST_INDEX_OOB] {key}: min={min(idx)} max={max(idx)} dim={expected_signal_dim}")
        indices[key] = idx
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
# Dataset resolution (manifest or dir)
# -----------------------------------------------------------------------------
def _resolve_train_val_parquets(
    dataset_manifest: Optional[Path],
    dataset_dir: Optional[Path],
    gx1_data: Path,
    train_parquet_hint: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Resolve (train_parquet, val_parquet). Exactly one of dataset_manifest or dataset_dir must be set.
    When dataset_dir is set, train/val are matched by strict suffix *_train.parquet / *_val.parquet.
    If train_parquet_hint is provided, that path is used as train and val is inferred (same stem, _val.parquet).
    """
    if dataset_manifest is not None and dataset_dir is not None:
        raise RuntimeError(
            "[ENTRY_V10_CTX_DATASET_ARGS] Use only one of --dataset_manifest or --dataset_dir"
        )
    if dataset_manifest is None and dataset_dir is None:
        raise RuntimeError(
            "[ENTRY_V10_CTX_DATASET_ARGS] Provide --dataset_manifest or --dataset_dir"
        )

    if dataset_manifest is not None:
        p = Path(dataset_manifest).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_MISSING] {p}")
        if p.suffix.lower() != ".json":
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_NOT_JSON] {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        train_path = Path(data.get("output_data_path", "")).expanduser().resolve()
        if not train_path.is_absolute():
            train_path = (p.parent / train_path).resolve()
        if not train_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_TRAIN_PARQUET_MISSING] {train_path}")
        # Val: same dir, stem with _train -> _val
        stem = train_path.stem
        if stem.endswith("_train"):
            val_stem = stem[: -len("_train")] + "_val"
        else:
            val_stem = stem.replace("train", "val", 1) if "train" in stem else stem + "_val"
        val_path = train_path.parent / f"{val_stem}.parquet"
        if not val_path.exists():
            raise RuntimeError(
                f"[ENTRY_V10_CTX_VAL_PARQUET_MISSING] {val_path} (inferred from train)"
            )
        return train_path, val_path

    # dataset_dir: strict suffix match _train.parquet / _val.parquet only
    d = Path(dataset_dir).expanduser().resolve()
    if not d.is_dir():
        raise RuntimeError(f"[ENTRY_V10_CTX_DATASET_DIR_MISSING] {d}")
    parquets = list(d.glob("*.parquet"))
    train_candidates = [f for f in parquets if f.stem.endswith("_train")]
    val_candidates = [f for f in parquets if f.stem.endswith("_val")]

    if train_parquet_hint is not None:
        train_path = Path(train_parquet_hint).expanduser().resolve()
        if not train_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_TRAIN_PARQUET_MISSING] {train_path}")
        if not train_path.stem.endswith("_train"):
            raise RuntimeError(
                f"[ENTRY_V10_CTX_TRAIN_STEM] train_parquet_hint stem must end with _train, got {train_path.stem}"
            )
        val_stem = train_path.stem[: -len("_train")] + "_val"
        val_path = train_path.parent / f"{val_stem}.parquet"
        if not val_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_VAL_PARQUET_MISSING] {val_path} (inferred from train)")
        log.info("[DATASET_RESOLVE] train=%s val=%s", train_path, val_path)
        return train_path, val_path

    if len(train_candidates) != 1:
        raise RuntimeError(
            f"[ENTRY_V10_CTX_NO_TRAIN_PARQUET] expected exactly one *_train.parquet in {d}, got {len(train_candidates)}"
        )
    if len(val_candidates) != 1:
        raise RuntimeError(
            f"[ENTRY_V10_CTX_NO_VAL_PARQUET] expected exactly one *_val.parquet in {d}, got {len(val_candidates)}"
        )
    train_path = train_candidates[0].resolve()
    val_path = val_candidates[0].resolve()
    log.info("[DATASET_RESOLVE] train=%s val=%s", train_path, val_path)
    return train_path, val_path


def _log_manifest_proof(dataset_manifest: Optional[Path]) -> None:
    if dataset_manifest is None:
        return
    p = Path(dataset_manifest).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_MISSING] {p}")
    if p.suffix.lower() != ".json":
        raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_NOT_JSON] {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    inputs = data.get("inputs") or {}
    fc = data.get("feature_contract") or {}
    xgb_bundle = str(inputs.get("xgb_bundle") or "")
    xgb_model_sha256 = str(inputs.get("xgb_model_sha256") or "")
    bridge_id = str(fc.get("signal_bridge_id") or "")
    bridge_sha = str(fc.get("signal_bridge_contract_sha256") or "")

    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        override_path = str(Path(xgb_override).expanduser().resolve())
        if xgb_bundle and Path(xgb_bundle).expanduser().resolve() != Path(override_path).expanduser().resolve():
            raise RuntimeError(
                "[ENTRY_V10_CTX_XGB_OVERRIDE_MISMATCH] "
                f"GX1_XGB_BUNDLE_DIR={override_path} dataset_manifest.xgb_bundle={xgb_bundle}"
            )

    log.info(
        "[ENTRY_DATASET_MANIFEST_PROOF] manifest=%s xgb_bundle=%s xgb_model_sha256=%s signal_bridge_id=%s signal_bridge_sha256=%s",
        p,
        xgb_bundle,
        xgb_model_sha256,
        bridge_id,
        bridge_sha,
    )


def _normalize_signal_names(names: List[str], dim: int) -> List[str]:
    out = [str(x) for x in names if str(x).strip()]
    if len(out) < int(dim):
        out.extend(f"seq_extra_{i}" for i in range(len(out), int(dim)))
    return out[: int(dim)]


def _default_signal_names(dim: int) -> List[str]:
    return _normalize_signal_names(list(SIGNAL_FIELDS), int(dim))


def _signal_contract_from_manifest_obj(data: Dict[str, Any]) -> Dict[str, Any]:
    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
    inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
    fc = data.get("feature_contract") if isinstance(data.get("feature_contract"), dict) else {}
    sb = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    fields_raw = sb.get("fields") or fc.get("signal_bridge_fields") or list(SIGNAL_FIELDS)
    fields = [str(x) for x in fields_raw]
    seq_dim = int(sb.get("seq_input_dim") or len(fields) or SEQ_SIGNAL_DIM)
    snap_dim = int(sb.get("snap_input_dim") or seq_dim)
    return {
        "seq_input_dim": seq_dim,
        "snap_input_dim": snap_dim,
        "fields": _normalize_signal_names(fields, seq_dim),
        "neutral_xgb_bridge": bool(
            extra.get("neutral_xgb_bridge", False)
            or inputs.get("neutral_xgb_bridge", False)
            or sb.get("neutral_xgb_bridge", False)
        ),
        "bridge_source": str(extra.get("xgb_bridge_source") or inputs.get("xgb_bridge_source") or sb.get("bridge_source") or ""),
    }


def _signal_contract_from_manifest_path(dataset_manifest: Optional[Path]) -> Dict[str, Any]:
    if dataset_manifest is None:
        return {
            "seq_input_dim": int(SEQ_SIGNAL_DIM),
            "snap_input_dim": int(SNAP_SIGNAL_DIM),
            "fields": list(SIGNAL_FIELDS),
            "neutral_xgb_bridge": False,
            "bridge_source": "xgb_bundle_inference",
        }
    p = Path(dataset_manifest).expanduser().resolve()
    if not p.exists():
        return {
            "seq_input_dim": int(SEQ_SIGNAL_DIM),
            "snap_input_dim": int(SNAP_SIGNAL_DIM),
            "fields": list(SIGNAL_FIELDS),
            "neutral_xgb_bridge": False,
            "bridge_source": "xgb_bundle_inference",
        }
    return _signal_contract_from_manifest_obj(json.loads(p.read_text(encoding="utf-8")))


def _smart520_state_contract_from_manifest_obj(data: Dict[str, Any]) -> Dict[str, Any]:
    extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
    contract = extra.get("smart520_state_contract")
    return dict(contract) if isinstance(contract, dict) else {}


def _smart520_state_contract_from_manifest_path(dataset_manifest: Optional[Path]) -> Dict[str, Any]:
    if dataset_manifest is None:
        return {}
    p = Path(dataset_manifest).expanduser().resolve()
    if not p.exists():
        return {}
    return _smart520_state_contract_from_manifest_obj(json.loads(p.read_text(encoding="utf-8")))


def _smart520_state_contract_for_parquet(parquet_path: Path) -> Dict[str, Any]:
    return _smart520_state_contract_from_manifest_path(
        Path(parquet_path).expanduser().resolve().with_suffix(".manifest.json")
    )


def _smart520_state_contract_failures(contract: Dict[str, Any], *, split: str) -> list[str]:
    failures: list[str] = []
    required = {
        "schema_version",
        "frame_anchor_utc",
        "model_range_start_utc",
        "rank_reference_end_utc",
        "rank_reference_npz",
        "rank_reference_npz_sha256",
    }
    if not isinstance(contract, dict) or not contract:
        return [f"{split} manifest missing smart520_state_contract for XAU direction repair"]
    missing = sorted(required - set(contract))
    if missing:
        failures.append(
            f"{split} smart520_state_contract missing fields for XAU direction repair: {','.join(missing)}"
        )
    if str(contract.get("schema_version") or "") != "smart520_state_contract_v1":
        failures.append(
            f"{split} smart520_state_contract schema_version must be smart520_state_contract_v1, "
            f"got {contract.get('schema_version')!r}"
        )
    rank_ref = str(contract.get("rank_reference_npz") or "").strip()
    rank_ref_low = rank_ref.lower()
    if not rank_ref:
        failures.append(f"{split} smart520_state_contract rank_reference_npz missing")
    elif not Path(rank_ref).expanduser().is_file():
        failures.append(f"{split} smart520_state_contract rank_reference_npz does not exist: {rank_ref}")
    else:
        rank_ref_path = Path(rank_ref).expanduser()
        expected_sha = str(contract.get("rank_reference_npz_sha256") or "").strip().lower()
        actual_sha = _sha256_file(rank_ref_path)
        if expected_sha != actual_sha:
            failures.append(
                f"{split} smart520_state_contract rank_reference_npz_sha256 mismatch: "
                f"metadata={expected_sha!r} actual={actual_sha} path={rank_ref}"
            )
        sidecar_path = rank_ref_path.with_suffix(rank_ref_path.suffix + ".json")
        if not sidecar_path.is_file():
            failures.append(f"{split} smart520_state_contract rank reference sidecar missing: {sidecar_path}")
        else:
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                sidecar_sha = str(sidecar.get("out_npz_sha256") or "").strip().lower()
                if sidecar_sha != expected_sha:
                    failures.append(
                        f"{split} smart520_state_contract sidecar out_npz_sha256 mismatch: "
                        f"sidecar={sidecar_sha!r} metadata={expected_sha!r}"
                    )
            except Exception as exc:
                failures.append(f"{split} smart520_state_contract rank reference sidecar unreadable: {sidecar_path}: {exc}")
    for marker in ("julyext", "smart_candidate_20260630", "utilityrepair", "20260710"):
        if marker in rank_ref_low:
            failures.append(
                f"{split} smart520_state_contract rank_reference_npz references stale marker "
                f"{marker!r}: {rank_ref}"
            )
    parsed_ts: Dict[str, pd.Timestamp] = {}
    for key in ("frame_anchor_utc", "model_range_start_utc", "rank_reference_end_utc"):
        try:
            ts = pd.to_datetime(str(contract.get(key) or ""), utc=True, errors="coerce")
            if pd.isna(ts):
                raise ValueError("NaT")
            parsed_ts[key] = ts
        except Exception:
            failures.append(f"{split} smart520_state_contract {key} is not a valid timestamp")
    if not failures and parsed_ts["frame_anchor_utc"] < parsed_ts["model_range_start_utc"]:
        failures.append(f"{split} smart520_state_contract frame_anchor_utc precedes model_range_start_utc")
    if not failures and parsed_ts["rank_reference_end_utc"] < parsed_ts["model_range_start_utc"]:
        failures.append(f"{split} smart520_state_contract rank_reference_end_utc precedes model_range_start_utc")
    if not failures and parsed_ts["frame_anchor_utc"] > parsed_ts["rank_reference_end_utc"]:
        failures.append(f"{split} smart520_state_contract frame_anchor_utc exceeds rank_reference_end_utc")
    return failures


def _signal_contract_for_parquet(parquet_path: Path, seq_dim: int, snap_dim: int) -> Dict[str, Any]:
    manifest_path = Path(parquet_path).expanduser().resolve().with_suffix(".manifest.json")
    if manifest_path.exists():
        contract = _signal_contract_from_manifest_path(manifest_path)
        if int(contract["seq_input_dim"]) != int(seq_dim) or int(contract["snap_input_dim"]) != int(snap_dim):
            raise RuntimeError(
                "[ENTRY_V10_CTX_MANIFEST_SIGNAL_DIM_MISMATCH] "
                f"{manifest_path} declares seq/snap={contract['seq_input_dim']}/{contract['snap_input_dim']} "
                f"but parquet has {seq_dim}/{snap_dim}"
            )
        return contract
    return {
        "seq_input_dim": int(seq_dim),
        "snap_input_dim": int(snap_dim),
        "fields": _default_signal_names(seq_dim),
        "neutral_xgb_bridge": False,
        "bridge_source": "unknown_no_manifest",
    }


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
        if "/eur" in low or "\\eur" in low or "_eur" in low or "eur_" in low:
            failures.append(f"{label} must not reference non-XAU project data for XAU direction repair: {text}")
        if "xau" not in low:
            failures.append(f"{label} must be XAU-specific for XAU direction repair: {text}")
        for marker in stale_markers:
            if marker in low:
                failures.append(f"{label} references stale pre-repair dataset marker {marker!r}: {text}")
    return failures


def _xau_direction_repair_manifest_failures(parquet_paths: Dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for split, raw_path in parquet_paths.items():
        parquet_path = Path(raw_path).expanduser()
        manifest_path = parquet_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            failures.append(f"{split} manifest missing for XAU direction repair: {manifest_path}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{split} manifest unreadable for XAU direction repair: {manifest_path}: {exc}")
            continue
        extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
        bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
        neutral_xgb_bridge = bool(
            manifest.get("neutral_xgb_bridge", False)
            or bridge.get("neutral_xgb_bridge", False)
        )
        xgb_bridge_source = str(
            manifest.get("xgb_bridge_source")
            or bridge.get("bridge_source")
            or extra.get("xgb_bridge_source")
            or ""
        )
        tape_root = str(manifest.get("tape_root") or extra.get("tape_root") or "").lower()
        if neutral_xgb_bridge is not True:
            failures.append(f"{split} manifest must declare neutral_xgb_bridge=true for XAU direction repair")
        if xgb_bridge_source != "neutral_uniform_proba":
            failures.append(
                f"{split} manifest must declare xgb_bridge_source=neutral_uniform_proba "
                f"for XAU direction repair, got {xgb_bridge_source!r}"
            )
        if "xauusd" not in tape_root:
            failures.append(f"{split} manifest must prove XAUUSD tape_root for XAU direction repair, got {tape_root!r}")
        failures.extend(
            _smart520_state_contract_failures(
                _smart520_state_contract_from_manifest_obj(manifest),
                split=split,
            )
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
        "y_long_high_mae_low_mfe_early_failure",
        "y_short_high_mae_low_mfe_early_failure",
    ]
    failures: list[str] = []
    missing = [name for name in required if name not in df.columns]
    if missing:
        failures.append(
            f"{split_name} missing XAU direction-repair target columns: {missing}. "
            "Rebuild the fresh XAU path-utility dataset; repair heads must not train on fallback labels."
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
        y_position_size,
    ]
    if any(not np.isfinite(arr).all() for arr in arrays):
        failures.append(f"{split_name} XAU direction-repair targets contain non-finite values")
        return failures

    anti_short = np.zeros(len(df), dtype=bool)
    for name in (
        "y_rising_channel_support_touch",
        "y_support_retest_continuation",
        "y_countertrend_short_trap",
        "y_short_high_mae_low_mfe_early_failure",
    ):
        anti_short |= pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    anti_long = np.zeros(len(df), dtype=bool)
    for name in (
        "y_falling_channel_resistance_touch",
        "y_resistance_retest_continuation",
        "y_countertrend_long_trap",
        "y_long_high_mae_low_mfe_early_failure",
    ):
        anti_long |= pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) > 0.5
    anti_short_only = anti_short & (~anti_long)
    anti_long_only = anti_long & (~anti_short)
    conflict_rows = anti_short & anti_long

    repaired_scalar_bad = np.zeros(len(df), dtype=np.float64)
    long_rows = (y_trade > 0.5) & (y_side == 0)
    short_rows = (y_trade > 0.5) & (y_side == 1)
    flat_rows = y_trade <= 0.5
    repaired_scalar_bad[long_rows] = y_long_bad[long_rows]
    repaired_scalar_bad[short_rows] = y_short_bad[short_rows]
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
        "scalar y_bad_path mismatches repaired side-specific bad-path targets": np.abs(y_bad_path - repaired_scalar_bad) > 1e-5,
        "mfe_first_n_bps mismatches selected side-specific MFE": np.abs(mfe_first - expected_mfe) > 1e-5,
        "mae_first_n_bps mismatches selected side-specific MAE": np.abs(mae_first - expected_mae) > 1e-5,
        "path_quality_bps mismatches selected side-specific path": np.abs(path_quality - expected_path) > 1e-5,
        "FLAT/no-trade rows have non-neutral y_position_size_target": flat_rows & (np.abs(y_position_size - 0.5) > 1e-5),
        "anti-short structural rows still labeled SHORT": anti_short_only & (y_direction == 1),
        "anti-long structural rows still labeled LONG": anti_long_only & (y_direction == 0),
        "anti-short structural rows still teach SHORT through side mask": anti_short_only & (y_side_mask > 0.5) & (y_side == 1),
        "anti-long structural rows still teach LONG through side mask": anti_long_only & (y_side_mask > 0.5) & (y_side == 0),
        "conflict structural rows are not FLAT/no-trade": conflict_rows
        & ((y_direction != 2) | (y_trade > 0.5) | (y_side_mask > 0.5)),
        "anti-short structural rows still have SHORT utility >= LONG utility": anti_short_only
        & (y_short_utility >= y_long_utility),
        "anti-long structural rows still have LONG utility >= SHORT utility": anti_long_only
        & (y_long_utility >= y_short_utility),
        "anti-short structural rows do not force SHORT bad-path target": anti_short_only & (y_short_bad < 0.999),
        "anti-long structural rows do not force LONG bad-path target": anti_long_only & (y_long_bad < 0.999),
        "anti-short structural rows do not make SHORT expected MAE worse": anti_short_only & (y_short_mae <= y_long_mae),
        "anti-long structural rows do not make LONG expected MAE worse": anti_long_only & (y_long_mae <= y_short_mae),
    }
    if "y_direction_long_score_bps" in df.columns:
        alias = pd.to_numeric(df["y_direction_long_score_bps"], errors="coerce").to_numpy(dtype=np.float64)
        checks["y_direction_long_score_bps mismatches repaired long utility"] = np.abs(alias - y_long_utility) > 1e-5
    if "y_direction_short_score_bps" in df.columns:
        alias = pd.to_numeric(df["y_direction_short_score_bps"], errors="coerce").to_numpy(dtype=np.float64)
        checks["y_direction_short_score_bps mismatches repaired short utility"] = np.abs(alias - y_short_utility) > 1e-5
    for reason, mask in checks.items():
        count = int(np.asarray(mask, dtype=bool).sum())
        if count:
            failures.append(f"{split_name}: {reason}: mismatches={count}")
    return failures


def _resolve_test_parquet(
    dataset_manifest: Optional[Path],
    dataset_dir: Optional[Path],
    test_parquet: Optional[Path],
    gx1_data: Path,
    bundle_dir: Optional[Path] = None,
) -> Path:
    """
    Resolve test parquet. Priority:
    1) Explicit --test_parquet
    2) From --dataset_dir: single *test*.parquet
    3) From --dataset_manifest: infer _test.parquet from train stem
    """
    if test_parquet is not None:
        p = Path(test_parquet).expanduser().resolve()
        _require(p.exists(), f"[ENTRY_V10_CTX_TEST_PARQUET_MISSING] {p}")
        return p

    if dataset_dir is not None:
        d = Path(dataset_dir).expanduser().resolve()
        _require(d.is_dir(), f"[ENTRY_V10_CTX_DATASET_DIR_MISSING] {d}")
        parquets = list(d.glob("*.parquet"))
        test_candidates = [f for f in parquets if "test" in f.stem.lower()]
        if len(test_candidates) == 1:
            return test_candidates[0]
        if len(test_candidates) > 1 and bundle_dir is not None:
            meta_path = Path(bundle_dir).expanduser() / "bundle_metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    train_path = Path(meta.get("train_data", "")).expanduser()
                    if not train_path.is_absolute():
                        train_path = (d / train_path).resolve()
                    stem = train_path.stem
                    if stem.endswith("_train"):
                        test_stem = stem[: -len("_train")] + "_test"
                    else:
                        test_stem = stem.replace("train", "test", 1) if "train" in stem else stem + "_test"
                    inferred = train_path.parent / f"{test_stem}.parquet"
                    if inferred.exists():
                        return inferred
                except Exception:
                    pass
        raise RuntimeError(
            f"[ENTRY_V10_CTX_TEST_AMBIGUOUS] expected exactly one *test*.parquet in {d}, got {len(test_candidates)}"
        )

    if dataset_manifest is not None:
        p = Path(dataset_manifest).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_MISSING] {p}")
        if p.suffix.lower() != ".json":
            raise RuntimeError(f"[ENTRY_V10_CTX_MANIFEST_NOT_JSON] {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        train_path = Path(data.get("output_data_path", "")).expanduser()
        if not train_path.is_absolute():
            train_path = (p.parent / train_path).resolve()
        if not train_path.exists():
            raise RuntimeError(f"[ENTRY_V10_CTX_TRAIN_PARQUET_MISSING] {train_path}")
        stem = train_path.stem
        if stem.endswith("_train"):
            test_stem = stem[: -len("_train")] + "_test"
        else:
            test_stem = stem.replace("train", "test", 1) if "train" in stem else stem + "_test"
        test_path = train_path.parent / f"{test_stem}.parquet"
        if not test_path.exists():
            raise RuntimeError(
                f"[ENTRY_V10_CTX_TEST_PARQUET_MISSING] {test_path} (inferred from train)"
            )
        return test_path

    raise RuntimeError(
        "[ENTRY_V10_CTX_TEST_RESOLVE_FAIL] provide --test_parquet or dataset manifest/dir"
    )


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
# Module-level cache for multi-TF feature tables. Keyed by absolute m5
# prebuilt path. Lets train_ds + val_ds share the ~3GB resampled DataFrames
# instead of each building their own (memory-blow-up that caused OOM on
# 15GB hosts during V12.2 development).
_MULTI_TF_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}


class EntryV10CtxDataset(Dataset):
    """
    Builds rolling-window samples from canonical ENTRY_V10_CTX parquet.
    ctx_cont / ctx_cat are per-sample (B, N), not per-timestep.

    Multi-TF mode (V12.2): when enable_multi_tf=True, the dataset also serves
    M15/H1/H4/D1 per-bar feature windows (96 bars × 19 features each) for the
    multi-TF V10 mode. Source is the M5 canonical_v3 prebuilt, resampled and
    feature-engineered once at __init__. Resampled tables are cached at module
    level so train_ds + val_ds share them. Adds ~25s init time first call,
    instant on subsequent dataset instantiations with same prebuilt path.
    """

    def __init__(
        self,
        parquet_path: Path,
        seq_len: int,
        allow_constant_labels: bool,
        enable_multi_tf: bool = False,
        m5_prebuilt_path: Optional[Path] = None,
        multi_tf_seq_len: int = 96,
        # V2 fast-train: per-TF seq_len overrides + smoke date subset
        per_tf_seq_lens: Optional[Dict[str, int]] = None,
        multi_tf_closed_bar: Optional[bool] = None,
        smoke_date_from: str = "",
        smoke_date_to: str = "",
    ):
        self.parquet_path = Path(parquet_path)
        self.seq_len = int(seq_len)
        self.seq_input_dim = int(SEQ_SIGNAL_DIM)
        self.snap_input_dim = int(SNAP_SIGNAL_DIM)
        self.signal_names = list(SIGNAL_FIELDS)
        self.neutral_xgb_bridge = False
        self.xgb_bridge_source = "xgb_bundle_inference"
        self.enable_multi_tf = bool(enable_multi_tf)
        self.multi_tf_seq_len = int(multi_tf_seq_len)
        self._multi_tf_closed_bar = (
            os.environ.get("GX1_PERTF_CLOSED_BAR", "0") == "1"
            if multi_tf_closed_bar is None
            else bool(multi_tf_closed_bar)
        )
        # per_tf_seq_lens: dict like {"M5": 96, "M15": 96, "H1": 96, "H4": 48, "D1": 30}.
        # Unset TFs fall back to multi_tf_seq_len.
        self.per_tf_seq_lens: Dict[str, int] = dict(per_tf_seq_lens) if per_tf_seq_lens else {}
        self.smoke_date_from = str(smoke_date_from or "")
        self.smoke_date_to = str(smoke_date_to or "")
        self._multi_tf_feats: Optional[Dict[str, pd.DataFrame]] = None
        self._multi_tf_shift: Optional[Dict[str, pd.Timedelta]] = None
        self._multi_tf_feature_count: int = 0
        self._memmap_tmpdir: Optional[tempfile.TemporaryDirectory] = None

        if not self.parquet_path.exists():
            raise FileNotFoundError(self.parquet_path)

        # ── Memory fix (V12.2): EXCLUDE nested-list columns from pandas load.
        # The nested-list columns (seq/snap/ctx_cont/ctx_cat) stored as
        # list<list<double>> blow up to ~30GB when materialized in pandas.
        # We re-read them via chunked pyarrow into pre-allocated numpy arrays
        # below (only for the advanced schema). For flat schema (smoke-test)
        # the columns are scalar and small — load normally.
        import pyarrow.parquet as pq
        _all_cols = pq.ParquetFile(self.parquet_path).schema_arrow.names
        _nested_cols = {"seq", "snap", "ctx_cont", "ctx_cat"}
        _has_nested = bool(_nested_cols & set(_all_cols))
        _load_cols = [c for c in _all_cols if c not in _nested_cols] if _has_nested else None
        df = pd.read_parquet(self.parquet_path, columns=_load_cols)
        if _has_nested:
            # Re-add empty stubs for downstream "in df.columns" presence checks.
            # They get filled (or dropped) by the chunked pyarrow load below.
            for _c in ("seq", "snap", "ctx_cont", "ctx_cat"):
                df[_c] = None

        ctx = get_canonical_ctx_contract()
        ctx_cont = list(ctx["ctx_cont_names"])
        ctx_cat = list(ctx["ctx_cat_names"])

        if "seq" in df.columns:
            # ---- advanced schema: builder has prebuilt samples
            if "y_teacher_bad_long" not in df.columns and "y_v6_teacher_bad_long" in df.columns:
                df["y_teacher_bad_long"] = df["y_v6_teacher_bad_long"]
            if "y_teacher_winner_long" not in df.columns and "y_v6_teacher_winner_long" in df.columns:
                df["y_teacher_winner_long"] = df["y_v6_teacher_winner_long"]
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
                "y_teacher_bad_long",
                "y_teacher_winner_long",
                "y_selector_long_mask",
            ]
            optional_defaults = {
                "y_dead_negative_long": 0.0,
                "y_teaser_negative_long": 0.0,
                "y_hard_negative_long": 0.0,
                "y_clean_edge_long": 0.0,
                "y_survival_long": 0.0,
                "y_teacher_bad_long": 0.0,
                "y_teacher_winner_long": 0.0,
                "y_selector_long_mask": 0.0,
            }
            for col, default in optional_defaults.items():
                if col not in df.columns:
                    df[col] = default
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
            seq_inner_dim = first_batch.column("seq").type.value_type.value_type
            seq_dim = int(first_batch.column("seq")[0].values[0].values.__len__())
            seq_len = int(first_batch.column("seq")[0].values.__len__())
            snap_dim = int(first_batch.column("snap")[0].values.__len__())
            ctx_cont_dim = int(first_batch.column("ctx_cont")[0].values.__len__())
            ctx_cat_dim = int(first_batch.column("ctx_cat")[0].values.__len__())
            signal_contract = _signal_contract_for_parquet(self.parquet_path, seq_dim=seq_dim, snap_dim=snap_dim)
            self.seq_input_dim = int(signal_contract["seq_input_dim"])
            self.snap_input_dim = int(signal_contract["snap_input_dim"])
            self.signal_names = list(signal_contract["fields"])
            self.neutral_xgb_bridge = bool(signal_contract.get("neutral_xgb_bridge", False))
            self.xgb_bridge_source = str(signal_contract.get("bridge_source") or "")
            log.info(
                f"[MEM_FIX] schema probe: seq=(N,{seq_len},{seq_dim})  snap=(N,{snap_dim})  "
                f"ctx_cont=(N,{ctx_cont_dim})  ctx_cat=(N,{ctx_cat_dim}) "
                f"neutral_xgb_bridge={self.neutral_xgb_bridge}"
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
            import gc; gc.collect()

            self.df = df
            self._advanced = True
            self.signal_cols = None
            self.ctx_cont_cols = None
            self.ctx_cat_cols = None
            # Infer ctx dims from pre-converted arrays
            self.ctx_cont_dim = int(self._np_ctx_cont.shape[1])
            self.ctx_cat_dim = int(self._np_ctx_cat.shape[1])
            self._ctx_vnext_extra = None
            if _is_vnext():
                if self.ctx_cont_dim == 21:
                    self._ctx_vnext_extra = None
                    log.info(
                        "[ENTRY_CTX_VNEXT_EXTRA_PROOF] source=prebuilt ctx_cont_dim=%d status=present",
                        self.ctx_cont_dim,
                    )
                elif self.ctx_cont_dim == 16:
                    ts = pd.to_datetime(self.df["time"], utc=True, errors="coerce")
                    _require(not ts.isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
                    sessions = get_session_vectorized(ts)
                    sess_id_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
                    sess_id = sessions.map(sess_id_map).fillna(0).astype("int32")
                    mins_since = get_session_minutes_since_open_vectorized(ts).astype("float32")
                    mins_to = get_session_minutes_to_next_boundary_vectorized(ts).astype("float32")
                    sess_change = sess_id.diff().fillna(0).ne(0).astype("int8")
                    sess_tradable = (sess_id != 0).astype("int8")
                    is_asia = (sess_id == 0).astype("int8")
                    self._ctx_vnext_extra = np.column_stack(
                        [is_asia.values, mins_since.values, mins_to.values, sess_change.values, sess_tradable.values]
                    ).astype(np.float32)
                    self.ctx_cont_dim = self.ctx_cont_dim + 5
                    uniq, cnt = np.unique(sess_id.values, return_counts=True)
                    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
                    log.info(
                        "[ENTRY_CTX_VNEXT_EXTRA_PROOF] source=session_detector ctx_cont_dim_base=16 added=5 "
                        "session_id_dist=%s mins_since_range=[%.1f, %.1f] mins_to_range=[%.1f, %.1f]",
                        dist,
                        float(np.nanmin(mins_since.values)),
                        float(np.nanmax(mins_since.values)),
                        float(np.nanmin(mins_to.values)),
                        float(np.nanmax(mins_to.values)),
                    )
                elif self.ctx_cont_dim in (43, 45, CTX_CONT_DIM_V3):
                    # V2 (43) / legacy V3 (45) / current V3 variants are contract-sized.
                    self._ctx_vnext_extra = None
                    log.info(
                        "[ENTRY_CTX_VNEXT_EXTRA_PROOF] source=prebuilt_v3 ctx_cont_dim=%d status=present (signal_bridge_v3)",
                        self.ctx_cont_dim,
                    )
                else:
                    raise RuntimeError(
                        f"[ENTRY_V10_CTX_VNEXT_DIM] expected ctx_cont_dim 16, 21, 43, 45, or {CTX_CONT_DIM_V3}, got {self.ctx_cont_dim}"
                    )

            y = df["y_direction"].astype(int).values
            if not allow_constant_labels:
                if len(np.unique(y)) < 2:
                    raise RuntimeError(
                        "[ENTRY_V10_CTX_LABELS_CONSTANT] "
                        "All y_direction identical. Use --allow-constant-labels only for smoke/plumbing."
                    )

            self.indices = np.arange(len(df))
            _require(len(self.indices) > 0, "[ENTRY_V10_CTX_NO_SAMPLES]")

            log.info(
                f"[DATASET_SCHEMA] advanced | rows={len(df)} samples={len(self.indices)} "
                f"time=[{df['time'].min()} .. {df['time'].max()}]"
            )
        else:
            # ---- flat columns (rolling-window); canary/smoke
            required_signal = list(SIGNAL_FIELDS)
            required = ["time"] + required_signal + ctx_cont + ctx_cat + ["y_direction"]
            missing = [c for c in required if c not in df.columns]
            _require(not missing, f"[ENTRY_V10_CTX_SCHEMA_MISSING] {missing}")

            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            _require(not df["time"].isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
            df = df.sort_values("time").reset_index(drop=True)

            self.df = df
            self._advanced = False
            self.signal_cols = required_signal
            self.ctx_cont_cols = ctx_cont
            self.ctx_cat_cols = ctx_cat
            self.ctx_cont_dim = int(len(ctx_cont))
            self.ctx_cat_dim = int(len(ctx_cat))
            self._ctx_vnext_extra = None
            if _is_vnext():
                ts = pd.to_datetime(self.df["time"], utc=True, errors="coerce")
                _require(not ts.isna().any(), "[ENTRY_V10_CTX_TIME_PARSE_FAIL]")
                sessions = get_session_vectorized(ts)
                sess_id_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
                sess_id = sessions.map(sess_id_map).fillna(0).astype("int32")
                mins_since = get_session_minutes_since_open_vectorized(ts).astype("float32")
                mins_to = get_session_minutes_to_next_boundary_vectorized(ts).astype("float32")
                sess_change = sess_id.diff().fillna(0).ne(0).astype("int8")
                sess_tradable = (sess_id != 0).astype("int8")
                is_asia = (sess_id == 0).astype("int8")
                self._ctx_vnext_extra = np.column_stack(
                    [is_asia.values, mins_since.values, mins_to.values, sess_change.values, sess_tradable.values]
                ).astype(np.float32)
                self.ctx_cont_dim = self.ctx_cont_dim + 5
                uniq, cnt = np.unique(sess_id.values, return_counts=True)
                dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
                log.info(
                    "[ENTRY_CTX_VNEXT_EXTRA_PROOF] source=session_detector ctx_cont_dim_base=%d added=5 "
                    "session_id_dist=%s mins_since_range=[%.1f, %.1f] mins_to_range=[%.1f, %.1f]",
                    int(len(ctx_cont)),
                    dist,
                    float(np.nanmin(mins_since.values)),
                    float(np.nanmax(mins_since.values)),
                    float(np.nanmin(mins_to.values)),
                    float(np.nanmax(mins_to.values)),
                )

            y = df["y_direction"].astype(int).values
            if not allow_constant_labels:
                if len(np.unique(y)) < 2:
                    raise RuntimeError(
                        "[ENTRY_V10_CTX_LABELS_CONSTANT] "
                        "All y_direction identical. Use --allow-constant-labels only for smoke/plumbing."
                    )

            self.indices = np.arange(self.seq_len - 1, len(df))
            _require(len(self.indices) > 0, "[ENTRY_V10_CTX_NO_SAMPLES] after seq_len windowing")

            log.info(
                f"[DATASET_SCHEMA] flat | rows={len(df)} samples={len(self.indices)} "
                f"time=[{df['time'].min()} .. {df['time'].max()}]"
            )

        # ── Multi-TF prebuild (V12.2) ──
        # Build M15/H1/H4/D1 per-bar feature tables once. At __getitem__ time we
        # just slice the last N bars at-or-before the sample's timestamp.
        # IMPORTANT: cache at module level so train_ds + val_ds share the
        # ~3GB resampled feature tables. Without this, peak memory hits OOM
        # on 15GB hosts (1.5GB parquet × 2 + multi-TF × 2 + train arrays).
        if self.enable_multi_tf:
            # MANDATORY V2: 5 TFs (M5/M15/H1/H4/D1) × 25 feats. The V1 (4-TF/17-feat)
            # path + GX1_V10_MULTI_TF_V2 env-gate were removed 2026-05-26 — multi-TF×5
            # is the only supported mode (rule: multi_tf_always_mandatory).
            v2_mode = True
            from gx1.features.htf_features import (
                build_multi_tf_per_bar_features,
                build_multi_tf_per_bar_features_v2,
                MULTI_TF_FEATURE_COUNT,
                MULTI_TF_FEATURE_COUNT_V2,
                MULTI_TF_SHIFT,
            )
            if m5_prebuilt_path is None:
                raise RuntimeError(
                    "[MULTI_TF_INIT_FAIL] enable_multi_tf=True requires m5_prebuilt_path "
                    "(path to canonical_v3 M5 OHLC parquet)."
                )
            m5_path = Path(m5_prebuilt_path)
            if not m5_path.is_file():
                raise FileNotFoundError(f"[MULTI_TF_INIT_FAIL] M5 prebuilt missing: {m5_path}")
            # Cache key includes v2 flag so V1 and V2 don't alias each other.
            cache_key = f"{m5_path.resolve()}|v2={v2_mode}"
            cached = _MULTI_TF_CACHE.get(cache_key)
            # Disk cache (V2 only): GX1_V10_MULTI_TF_V2_CACHE_DIR points at a pre-built
            # cache from scripts/prebuild_multi_tf_cache_v2.py. Saves ~84s per init.
            _disk_cache_dir = os.environ.get("GX1_V10_MULTI_TF_V2_CACHE_DIR", "").strip()
            if cached is not None:
                log.info(f"[MULTI_TF] reusing cached feature tables (key={m5_path.name} v2={v2_mode})")
                self._multi_tf_feats = cached
            elif v2_mode and _disk_cache_dir:
                from gx1.features.htf_features import (
                    MULTI_TF_PER_BAR_FEATURES_V2,
                    MULTI_TF_SHIFT,
                    load_multi_tf_v2_cache,
                )
                log.info(f"[MULTI_TF] loading V2 disk cache: {_disk_cache_dir}")
                _disk_manifest_path = Path(_disk_cache_dir) / "manifest.json"
                if not _disk_manifest_path.is_file():
                    raise RuntimeError(f"[MULTI_TF_CACHE_MANIFEST_MISSING] {_disk_manifest_path}")
                _disk_manifest = json.loads(_disk_manifest_path.read_text(encoding="utf-8"))
                _observed_features = [str(x) for x in (_disk_manifest.get("feature_names") or [])]
                _expected_features = list(MULTI_TF_PER_BAR_FEATURES_V2)
                if _observed_features != _expected_features:
                    raise RuntimeError(
                        "[MULTI_TF_CACHE_FEATURE_CONTRACT_MISMATCH] "
                        f"cache={_disk_cache_dir} observed={_observed_features or '<missing>'} "
                        f"expected={_expected_features}"
                    )
                _observed_shift = (
                    _disk_manifest.get("shift_contract")
                    if isinstance(_disk_manifest.get("shift_contract"), dict)
                    else {}
                )
                _expected_shift = {tf: str(shift) for tf, shift in MULTI_TF_SHIFT.items()}
                if _observed_shift != _expected_shift:
                    raise RuntimeError(
                        "[MULTI_TF_CACHE_SHIFT_CONTRACT_MISMATCH] "
                        f"cache={_disk_cache_dir} observed={_observed_shift or '<missing>'} "
                        f"expected={_expected_shift}"
                    )
                _expected_source_sha = _sha256_file(m5_path)
                _observed_source_sha = str(_disk_manifest.get("m5_prebuilt_source_sha256") or "").strip()
                if _observed_source_sha != _expected_source_sha:
                    raise RuntimeError(
                        "[MULTI_TF_CACHE_SOURCE_SHA_MISMATCH] "
                        f"cache={_disk_cache_dir} observed={_observed_source_sha or '<missing>'} "
                        f"expected={_expected_source_sha} m5_prebuilt={m5_path}"
                    )
                _loaded_cache = load_multi_tf_v2_cache(_disk_cache_dir)
                # FRESHNESS GUARD (2026-07-04 stale-cache incident): the disk cache is
                # keyed by filename, so an in-place-extended prebuilt silently serves a
                # stale snapshot — June eval bars got MTF context frozen at 2026-05-25
                # (same freeze class as the 2026-05-25 BASE34 incident, rule 9). The
                # dataset-build side already fail-closes on this; this is the eval/train
                # twin. Cache lagging the prebuilt by >2 days = hard error, never a
                # silent frozen-context eval.
                import pyarrow.parquet as _pq
                _prebuilt_max = pd.to_datetime(
                    _pq.read_table(m5_path, columns=["time"]).column("time").to_numpy(zero_copy_only=False).max(),
                    utc=True,
                )
                _cache_max = _loaded_cache["M5"].index.max()
                if _cache_max + pd.Timedelta(days=2) < _prebuilt_max:
                    raise RuntimeError(
                        f"[MULTI_TF_CACHE_STALE] disk cache M5 max {_cache_max} lags prebuilt "
                        f"{_prebuilt_max} by >2d — regenerate via "
                        f"python -m gx1.scripts.prebuild_multi_tf_cache_v2 --m5-prebuilt {m5_path} "
                        f"--out-dir {_disk_cache_dir}"
                    )
                self._multi_tf_feats = _loaded_cache
                _MULTI_TF_CACHE[cache_key] = self._multi_tf_feats
            else:
                log.info(f"[MULTI_TF] loading M5 prebuilt for resample: {m5_path.name} (v2={v2_mode})")
                load_cols = ["time", "open", "high", "low", "close"]
                if v2_mode:
                    # V2 needs volume for VWAP family — fall back to tick-equal if missing.
                    import pyarrow.parquet as pq
                    if "volume" in pq.ParquetFile(m5_path).schema_arrow.names:
                        load_cols.append("volume")
                m5 = pd.read_parquet(m5_path, columns=load_cols)
                m5["time"] = pd.to_datetime(m5["time"], utc=True)
                m5 = m5.set_index("time").sort_index()
                for c in ("open", "high", "low", "close"):
                    m5[c] = m5[c].astype(np.float32)
                if "volume" in m5.columns:
                    m5["volume"] = m5["volume"].astype(np.float32)
                tf_label = "M5+M15+H1+H4+D1 (V2 25-feat)" if v2_mode else "M15+H1+H4+D1 (V1 17-feat)"
                log.info(f"[MULTI_TF] M5 prebuilt: {len(m5):,} rows, building {tf_label}...")
                if v2_mode:
                    self._multi_tf_feats = build_multi_tf_per_bar_features_v2(m5)
                else:
                    self._multi_tf_feats = build_multi_tf_per_bar_features(m5)
                    # V1 returned all 5 keys (M5/M15/H1/H4/D1) historically; V1
                    # V10 trainer expects only 4 (M5 is the base seq). Filter.
                    self._multi_tf_feats = {
                        k: v for k, v in self._multi_tf_feats.items() if k != "M5"
                    }
                del m5
                import gc; gc.collect()
                _MULTI_TF_CACHE[cache_key] = self._multi_tf_feats
            self._multi_tf_shift = MULTI_TF_SHIFT
            self._multi_tf_target_availability_shift = (
                pd.Timedelta(minutes=5)
                if self._multi_tf_closed_bar
                else pd.Timedelta(0)
            )
            self._multi_tf_feature_count = (
                int(MULTI_TF_FEATURE_COUNT_V2) if v2_mode else int(MULTI_TF_FEATURE_COUNT)
            )
            self._multi_tf_v2 = bool(v2_mode)
            for tf_name, feats in self._multi_tf_feats.items():
                log.info(
                    f"[MULTI_TF] {tf_name}: {len(feats):,} bars × {feats.shape[1]} feats  "
                    f"range {feats.index[0]} → {feats.index[-1]}"
                )

        # V2 fast-train: smoke-date subset (applies to BOTH advanced and flat schemas).
        # Subsets self.indices only — np_seq + df rows are kept intact (idx-addressed).
        if self.smoke_date_from or self.smoke_date_to:
            times = pd.to_datetime(self.df["time"], utc=True, errors="coerce")
            mask = np.ones(len(times), dtype=bool)
            if self.smoke_date_from:
                from_ts = pd.Timestamp(self.smoke_date_from, tz="UTC")
                mask &= (times >= from_ts).values
            if self.smoke_date_to:
                to_ts = pd.Timestamp(self.smoke_date_to, tz="UTC")
                mask &= (times <= to_ts).values
            kept = np.where(mask)[0]
            # Intersect with existing self.indices (preserves seq_len warmup constraint of flat schema).
            self.indices = np.array(sorted(set(self.indices.tolist()) & set(kept.tolist())), dtype=np.int64)
            log.info(
                f"[SMOKE_DATE] range=[{self.smoke_date_from or '*'}..{self.smoke_date_to or '*'}] "
                f"samples kept: {len(self.indices):,}/{len(times):,}"
            )

    def _get_multi_tf_window(self, target_ts: pd.Timestamp) -> Dict[str, np.ndarray]:
        """Slice the multi-TF window at-or-before target_ts, using per-TF seq_len.

        Returns dict with one 'seq_<tf>' key per TF in self._multi_tf_feats. Each
        array shape = (per_tf_lens[TF], feature_count) float32, left-zero-padded
        on warmup.
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
            ctx_cont = self._np_ctx_cont[t].copy()   # copy so concatenate below is safe
            if _is_vnext() and self._ctx_vnext_extra is not None:
                extra = self._ctx_vnext_extra[t]
                ctx_cont = np.concatenate([ctx_cont, extra.astype(np.float32)], axis=0)
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
                "y_tradable": torch.tensor(float(row["y_tradable"]), dtype=torch.float32),
                "mfe_first_n_bps": torch.tensor(float(row["mfe_first_n_bps"]), dtype=torch.float32),
                "path_quality_bps": torch.tensor(float(row["path_quality_bps"]), dtype=torch.float32),
                "y_bad_path": torch.tensor(float(row.get("y_bad_path", 0.0)), dtype=torch.float32),
                "y_dead_negative_long": torch.tensor(float(row.get("y_dead_negative_long", 0.0)), dtype=torch.float32),
                "y_teaser_negative_long": torch.tensor(float(row.get("y_teaser_negative_long", 0.0)), dtype=torch.float32),
                "y_hard_negative_long": torch.tensor(float(row.get("y_hard_negative_long", 0.0)), dtype=torch.float32),
                # SHORT-side negatives (vedtak v10_symmetric_negatives_20260603) — fed only when
                # ENTRY_SYMMETRIC_NEGATIVES=1; default 0.0 fallback keeps cement bit-parity.
                "y_dead_negative_short": torch.tensor(float(row.get("y_dead_negative_short", 0.0)), dtype=torch.float32),
                "y_teaser_negative_short": torch.tensor(float(row.get("y_teaser_negative_short", 0.0)), dtype=torch.float32),
                "y_hard_negative_short": torch.tensor(float(row.get("y_hard_negative_short", 0.0)), dtype=torch.float32),
                "y_clean_edge_long": torch.tensor(float(row.get("y_clean_edge_long", 0.0)), dtype=torch.float32),
                "y_survival_long": torch.tensor(float(row.get("y_survival_long", 0.0)), dtype=torch.float32),
                # bidir quality labels + short selector (vedtak v10_symmetric_negatives_20260603)
                "y_selector_short_mask": torch.tensor(float(row.get("y_selector_short_mask", 0.0)), dtype=torch.float32),
                "y_clean_edge_bidir": torch.tensor(float(row.get("y_clean_edge_bidir", 0.0)), dtype=torch.float32),
                "y_survival_bidir": torch.tensor(float(row.get("y_survival_bidir", 0.0)), dtype=torch.float32),
                "y_trade": torch.tensor(float(row.get("y_trade", row.get("y_tradable", 1.0 if y in (0, 1) else 0.0))), dtype=torch.float32),
                "y_side": torch.tensor(int(row.get("y_side", 0 if y == 0 else 1 if y == 1 else 0)), dtype=torch.long),
                "y_side_mask": torch.tensor(float(row.get("y_side_mask", 1.0 if y in (0, 1) else 0.0)), dtype=torch.float32),
                "y_long_path_utility_bps": torch.tensor(float(row.get("y_long_path_utility_bps", row.get("y_direction_long_score_bps", 0.0))), dtype=torch.float32),
                "y_short_path_utility_bps": torch.tensor(float(row.get("y_short_path_utility_bps", row.get("y_direction_short_score_bps", 0.0))), dtype=torch.float32),
                "y_long_bad_path": torch.tensor(float(row.get("y_long_bad_path", 0.0)), dtype=torch.float32),
                "y_short_bad_path": torch.tensor(float(row.get("y_short_bad_path", 0.0)), dtype=torch.float32),
                "y_long_expected_mae_bps": torch.tensor(float(row.get("y_long_expected_mae_bps", 0.0)), dtype=torch.float32),
                "y_short_expected_mae_bps": torch.tensor(float(row.get("y_short_expected_mae_bps", 0.0)), dtype=torch.float32),
                "y_rising_channel_support_touch": torch.tensor(float(row.get("y_rising_channel_support_touch", 0.0)), dtype=torch.float32),
                "y_falling_channel_resistance_touch": torch.tensor(float(row.get("y_falling_channel_resistance_touch", 0.0)), dtype=torch.float32),
                "y_support_retest_continuation": torch.tensor(float(row.get("y_support_retest_continuation", 0.0)), dtype=torch.float32),
                "y_resistance_retest_continuation": torch.tensor(float(row.get("y_resistance_retest_continuation", 0.0)), dtype=torch.float32),
                "y_countertrend_short_trap": torch.tensor(float(row.get("y_countertrend_short_trap", 0.0)), dtype=torch.float32),
                "y_countertrend_long_trap": torch.tensor(float(row.get("y_countertrend_long_trap", 0.0)), dtype=torch.float32),
                "y_mtf_conflict_m5_vs_higher_side": torch.tensor(float(row.get("y_mtf_conflict_m5_vs_higher_side", 0.0)), dtype=torch.float32),
                "y_long_high_mae_low_mfe_early_failure": torch.tensor(float(row.get("y_long_high_mae_low_mfe_early_failure", 0.0)), dtype=torch.float32),
                "y_short_high_mae_low_mfe_early_failure": torch.tensor(float(row.get("y_short_high_mae_low_mfe_early_failure", 0.0)), dtype=torch.float32),
                "y_teacher_bad_long": torch.tensor(float(row.get("y_teacher_bad_long", row.get("y_v6_teacher_bad_long", 0.0))), dtype=torch.float32),
                "y_teacher_winner_long": torch.tensor(float(row.get("y_teacher_winner_long", row.get("y_v6_teacher_winner_long", 0.0))), dtype=torch.float32),
                "y_selector_long_mask": torch.tensor(float(row.get("y_selector_long_mask", 0.0)), dtype=torch.float32),
                # V10 v3+ Target 1: multi-TF trend-agreement score (fallback 0.5 = neutral)
                "y_tf_agreement_score": torch.tensor(float(row.get("y_tf_agreement_score", 0.5)), dtype=torch.float32),
                # V10 v3+ Target 3: position-size target (fallback 0.5 = neutral)
                "y_position_size_target": torch.tensor(float(row.get("y_position_size_target", 0.5)), dtype=torch.float32),
                # V10 v3+ Target 4: hold-horizon target (fallback 0.5 = ~720 bars)
                "y_hold_horizon_target": torch.tensor(float(row.get("y_hold_horizon_target", 0.5)), dtype=torch.float32),
            }
            for _tcol in _DIP_FORECAST_TARGET_COLS:  # dip(12) + forecast(4) head targets
                out_batch[_tcol] = torch.tensor(float(row.get(_tcol, 0.0)), dtype=torch.float32)
            if self.enable_multi_tf:
                mtf = self._get_multi_tf_window(pd.Timestamp(row["time"]))
                for k, v in mtf.items():
                    out_batch[k] = torch.from_numpy(v)
            return out_batch
        else:
            t = self.indices[i]
            start = t - self.seq_len + 1

            seq = self.df.iloc[start : t + 1][self.signal_cols].values.astype(np.float32)
            snap = self.df.iloc[t][self.signal_cols].values.astype(np.float32)
            ctx_cont = self.df.iloc[t][self.ctx_cont_cols].values.astype(np.float32)
            if _is_vnext() and self._ctx_vnext_extra is not None:
                extra = self._ctx_vnext_extra[t]
                ctx_cont = np.concatenate([ctx_cont, extra.astype(np.float32)], axis=0)
            ctx_cat = self.df.iloc[t][self.ctx_cat_cols].values.astype(np.int64)
            y = int(self.df.iloc[t]["y_direction"])
            if y not in (0, 1, 2):
                raise RuntimeError(f"[ENTRY_V10_CTX_LABEL_INVALID] y_direction={y} expected 0/1/2")

            out_batch = {
                "seq_x": torch.tensor(seq),
                "snap_x": torch.tensor(snap),
                "ctx_cont": torch.tensor(ctx_cont),
                "ctx_cat": torch.tensor(ctx_cat),
                "y": torch.tensor(y, dtype=torch.long),
                "y_tradable": torch.tensor(0.0, dtype=torch.float32),
                "mfe_first_n_bps": torch.tensor(0.0, dtype=torch.float32),
                "path_quality_bps": torch.tensor(0.0, dtype=torch.float32),
                "y_bad_path": torch.tensor(0.0, dtype=torch.float32),
                "y_dead_negative_long": torch.tensor(0.0, dtype=torch.float32),
                "y_teaser_negative_long": torch.tensor(0.0, dtype=torch.float32),
                "y_hard_negative_long": torch.tensor(0.0, dtype=torch.float32),
                "y_dead_negative_short": torch.tensor(0.0, dtype=torch.float32),
                "y_teaser_negative_short": torch.tensor(0.0, dtype=torch.float32),
                "y_hard_negative_short": torch.tensor(0.0, dtype=torch.float32),
                "y_clean_edge_long": torch.tensor(0.0, dtype=torch.float32),
                "y_survival_long": torch.tensor(0.0, dtype=torch.float32),
                "y_selector_short_mask": torch.tensor(0.0, dtype=torch.float32),
                "y_clean_edge_bidir": torch.tensor(0.0, dtype=torch.float32),
                "y_survival_bidir": torch.tensor(0.0, dtype=torch.float32),
                "y_trade": torch.tensor(float(1.0 if y in (0, 1) else 0.0), dtype=torch.float32),
                "y_side": torch.tensor(0 if y == 0 else 1 if y == 1 else 0, dtype=torch.long),
                "y_side_mask": torch.tensor(float(1.0 if y in (0, 1) else 0.0), dtype=torch.float32),
                "y_long_path_utility_bps": torch.tensor(0.0, dtype=torch.float32),
                "y_short_path_utility_bps": torch.tensor(0.0, dtype=torch.float32),
                "y_long_bad_path": torch.tensor(0.0, dtype=torch.float32),
                "y_short_bad_path": torch.tensor(0.0, dtype=torch.float32),
                "y_long_expected_mae_bps": torch.tensor(0.0, dtype=torch.float32),
                "y_short_expected_mae_bps": torch.tensor(0.0, dtype=torch.float32),
                "y_rising_channel_support_touch": torch.tensor(0.0, dtype=torch.float32),
                "y_falling_channel_resistance_touch": torch.tensor(0.0, dtype=torch.float32),
                "y_support_retest_continuation": torch.tensor(0.0, dtype=torch.float32),
                "y_resistance_retest_continuation": torch.tensor(0.0, dtype=torch.float32),
                "y_countertrend_short_trap": torch.tensor(0.0, dtype=torch.float32),
                "y_countertrend_long_trap": torch.tensor(0.0, dtype=torch.float32),
                "y_mtf_conflict_m5_vs_higher_side": torch.tensor(0.0, dtype=torch.float32),
                "y_long_high_mae_low_mfe_early_failure": torch.tensor(0.0, dtype=torch.float32),
                "y_short_high_mae_low_mfe_early_failure": torch.tensor(0.0, dtype=torch.float32),
                "y_teacher_bad_long": torch.tensor(0.0, dtype=torch.float32),
                "y_teacher_winner_long": torch.tensor(0.0, dtype=torch.float32),
                "y_selector_long_mask": torch.tensor(0.0, dtype=torch.float32),
                # V10 v3+ Target 1: tf_agreement neutral fallback for flat schema
                "y_tf_agreement_score": torch.tensor(0.5, dtype=torch.float32),
                # V10 v3+ Target 3: position-size neutral fallback for flat schema
                "y_position_size_target": torch.tensor(0.5, dtype=torch.float32),
                # V10 v3+ Target 4: hold-horizon neutral fallback for flat schema
                "y_hold_horizon_target": torch.tensor(0.5, dtype=torch.float32),
            }
            _row_t = self.df.iloc[t]
            for _tcol in _DIP_FORECAST_TARGET_COLS:  # dip(12) + forecast(4) head targets
                _v = _row_t[_tcol] if _tcol in self.df.columns else 0.0
                out_batch[_tcol] = torch.tensor(float(_v), dtype=torch.float32)
            if self.enable_multi_tf:
                target_ts = pd.Timestamp(self.df.iloc[t]["time"])
                mtf = self._get_multi_tf_window(target_ts)
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
    """Orders training samples so slice-loss batches contain audited ctx_cat slices."""

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
        self.num_samples = int(np.ceil(labels_arr.shape[0] / self.batch_size) * self.batch_size)
        self._all_positions = np.arange(labels_arr.shape[0], dtype=np.int64)
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

    def _sample_slice_rows(
        self,
        rng: np.random.Generator,
        key: tuple[int, int],
    ) -> list[int]:
        active_classes = list(self._active_classes[key])
        out: list[int] = []
        shuffled_classes = list(rng.permutation(active_classes).tolist())
        for cls in shuffled_classes:
            cls_rows = self._slice_class_rows[(key, int(cls))]
            out.append(int(rng.choice(cls_rows)))
        all_rows = self._slice_rows[key]
        while len(out) < self.min_rows:
            out.append(int(rng.choice(all_rows)))
        rng.shuffle(out)
        return out[: self.min_rows]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._iteration)
        self._iteration += 1
        batches = max(1, self.num_samples // self.batch_size)
        slice_keys = [self._slice_keys[int(i)] for i in rng.permutation(len(self._slice_keys))]
        slice_pos = 0
        emitted = 0
        for _ in range(batches):
            batch: list[int] = []
            while len(batch) + self.min_rows <= self.batch_size:
                key = slice_keys[slice_pos]
                slice_pos += 1
                if slice_pos >= len(slice_keys):
                    slice_keys = [self._slice_keys[int(i)] for i in rng.permutation(len(self._slice_keys))]
                    slice_pos = 0
                batch.extend(self._sample_slice_rows(rng, key))
            while len(batch) < self.batch_size:
                batch.append(int(rng.choice(self._all_positions)))
            rng.shuffle(batch)
            for sample_idx in batch:
                emitted += 1
                yield int(sample_idx)
        if emitted != self.num_samples:
            raise RuntimeError(
                "[ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_EMIT_MISMATCH] "
                f"emitted={emitted} expected={self.num_samples}"
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


def _hier_public_flat_consistency_term(
    direction_logits: Optional[torch.Tensor],
    trade_logit: torch.Tensor,
    y_trade: torch.Tensor,
) -> torch.Tensor:
    zero = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
    weight = float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT)
    if weight <= 0.0:
        return zero
    if direction_logits is None or direction_logits.ndim != 2 or direction_logits.shape[1] < 3:
        return zero
    logits = trade_logit.reshape(-1)
    if direction_logits.shape[0] != logits.shape[0]:
        return zero
    flat_mask = y_trade.reshape(-1).float() <= 0.5
    if flat_mask.numel() != logits.shape[0]:
        return zero
    min_label_rate = float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE)
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    flat_rate = float(flat_mask.float().mean().detach().cpu().item())
    if flat_rate < min_label_rate:
        return zero
    public_flat_prob = torch.softmax(direction_logits.float(), dim=1)[:, 2]
    hier_flat_prob = 1.0 - torch.sigmoid(logits.float())
    return weight * nn.functional.mse_loss(public_flat_prob, hier_flat_prob)


def _hier_slice_public_flat_consistency_term(
    direction_logits: Optional[torch.Tensor],
    trade_logit: torch.Tensor,
    y_trade: torch.Tensor,
    ctx_cat: Optional[torch.Tensor],
) -> torch.Tensor:
    zero = torch.zeros((), device=trade_logit.device, dtype=trade_logit.dtype)
    weight = float(ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT)
    if weight <= 0.0:
        return zero
    if direction_logits is None or direction_logits.ndim != 2 or direction_logits.shape[1] < 3:
        return zero
    if ctx_cat is None or ctx_cat.ndim != 2:
        return zero
    logits = trade_logit.reshape(-1)
    if direction_logits.shape[0] != logits.shape[0] or ctx_cat.shape[0] != logits.shape[0]:
        return zero
    flat_mask = y_trade.reshape(-1).float() <= 0.5
    if flat_mask.numel() != logits.shape[0]:
        return zero

    min_rows = int(ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS)
    min_label_rate = float(ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE)
    if min_rows < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS_INVALID] "
            f"ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS={min_rows} expected >=2"
        )
    if min_label_rate < 0.0 or min_label_rate > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE_INVALID] "
            f"ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE={min_label_rate:.6f} expected [0.0, 1.0]"
        )
    indices = _direction_slice_ctx_cat_indices(int(ctx_cat.shape[1]))
    if not indices:
        return zero

    public_flat_prob = torch.softmax(direction_logits.float(), dim=1)[:, 2]
    hier_flat_prob = 1.0 - torch.sigmoid(logits.float())
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
            values.append(nn.functional.mse_loss(public_flat_prob[mask], hier_flat_prob[mask]))
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


def _specialist_gate_regularization(out: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    gate = out.get("specialist_gate")
    zero = torch.zeros((), device=device)
    if gate is None or not isinstance(gate, torch.Tensor) or gate.numel() == 0:
        return zero, {"entropy": 0.0, "min_mean": 0.0, "kl_uniform": 0.0, "floor_hinge": 0.0}
    gate = gate.float().clamp(min=1e-8)
    mean_gate = gate.mean(dim=0)
    entropy = -(gate * gate.log()).sum(dim=1).mean()
    max_entropy = torch.log(torch.tensor(float(gate.shape[1]), device=device, dtype=gate.dtype))
    entropy_loss = (max_entropy - entropy).clamp_min(0.0)
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


def _bad_path_quality_rank_loss(
    bad_path_logit: torch.Tensor | None,
    path_quality_bps: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    zero = torch.zeros((), device=device)
    if float(ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT) <= 0.0:
        return zero
    if bad_path_logit is None or not isinstance(bad_path_logit, torch.Tensor) or bad_path_logit.numel() == 0:
        return zero
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
    path_quality_pred: torch.Tensor | None,
    path_quality_bps: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    zero = torch.zeros((), device=device)
    if float(ENTRY_PATH_QUALITY_RANK_WEIGHT) <= 0.0:
        return zero
    if path_quality_pred is None or not isinstance(path_quality_pred, torch.Tensor) or path_quality_pred.numel() == 0:
        return zero
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
    """Opt-in XAU direction repair objective.

    The legacy LONG/SHORT/FLAT CE is kept for compatibility. These heads teach
    the missing decomposition explicitly: trade/no-trade first, then side given
    trade, then per-side path utility / bad-path / MAE.
    """
    total = torch.tensor(0.0, device=device)
    stats: Dict[str, float] = {
        "hier_trade_loss": 0.0,
        "hier_trade_global_prior_loss": 0.0,
        "hier_slice_trade_prior_loss": 0.0,
        "hier_slice_trade_accuracy_edge_loss": 0.0,
        "hier_flat_logit_margin_loss": 0.0,
        "hier_slice_flat_logit_margin_loss": 0.0,
        "hier_public_trade_loss": 0.0,
        "hier_public_trade_global_prior_loss": 0.0,
        "hier_public_slice_trade_prior_loss": 0.0,
        "hier_public_slice_trade_accuracy_edge_loss": 0.0,
        "hier_public_flat_logit_margin_loss": 0.0,
        "hier_public_slice_flat_logit_margin_loss": 0.0,
        "hier_public_flat_consistency_loss": 0.0,
        "hier_slice_public_flat_consistency_loss": 0.0,
        "hier_side_loss": 0.0,
        "hier_slice_side_ce_loss": 0.0,
        "hier_slice_side_margin_loss": 0.0,
        "hier_slice_side_accuracy_edge_loss": 0.0,
        "hier_side_global_prior_loss": 0.0,
        "hier_slice_side_prior_loss": 0.0,
        "hier_public_side_loss": 0.0,
        "hier_public_slice_side_ce_loss": 0.0,
        "hier_public_slice_side_margin_loss": 0.0,
        "hier_public_slice_side_accuracy_edge_loss": 0.0,
        "hier_public_side_global_prior_loss": 0.0,
        "hier_public_slice_side_prior_loss": 0.0,
        "hier_utility_loss": 0.0,
        "hier_bad_path_loss": 0.0,
        "hier_mae_loss": 0.0,
        "hier_side_validity_loss": 0.0,
        "hier_long_valid_target_rate": 0.0,
        "hier_short_valid_target_rate": 0.0,
        "hier_long_valid_prob_mean": 0.0,
        "hier_short_valid_prob_mean": 0.0,
        "hier_pocket_abstain_loss": 0.0,
        "hier_pocket_abstain_rows": 0.0,
        "hier_pocket_side_margin_loss": 0.0,
        "hier_pocket_anti_short_rows": 0.0,
        "hier_pocket_anti_long_rows": 0.0,
        "hier_side_rows": 0.0,
        "hier_side_acc": 0.0,
        "hier_public_side_rows": 0.0,
        "hier_public_side_acc": 0.0,
        "hier_long_bad_target_rate": 0.0,
        "hier_short_bad_target_rate": 0.0,
        "hier_countertrend_long_trap_rate": 0.0,
        "hier_countertrend_short_trap_rate": 0.0,
    }
    trade_logit = out.get("trade_logit")
    public_trade_logit = out.get("public_trade_logit")
    direction_logits = out.get("direction_logits")
    side_logits = out.get("side_logits")
    public_side_logits = out.get("public_side_logits")
    side_utility = out.get("side_utility")
    side_bad_path_logit = out.get("side_bad_path_logit")
    side_mae = out.get("side_mae")
    side_validity_logit = out.get("side_validity_logit")
    if (
        trade_logit is None
        and public_trade_logit is None
        and side_logits is None
        and public_side_logits is None
        and side_utility is None
        and side_bad_path_logit is None
        and side_mae is None
        and side_validity_logit is None
    ):
        return total, stats

    non_blocking = device.type == "cuda"
    y_trade = batch["y_trade"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    y_side = batch["y_side"].to(device, non_blocking=non_blocking).long().clamp(0, 1)
    y_side_mask = batch["y_side_mask"].to(device, non_blocking=non_blocking).float() > 0.5
    ctx_cat_value = batch.get("ctx_cat")
    ctx_cat = (
        ctx_cat_value.to(device, non_blocking=non_blocking)
        if isinstance(ctx_cat_value, torch.Tensor)
        else None
    )
    y_long_util = batch["y_long_path_utility_bps"].to(device, non_blocking=non_blocking).float()
    y_short_util = batch["y_short_path_utility_bps"].to(device, non_blocking=non_blocking).float()
    y_long_bad = batch["y_long_bad_path"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    y_short_bad = batch["y_short_bad_path"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    y_long_mae = batch["y_long_expected_mae_bps"].to(device, non_blocking=non_blocking).float().clamp_min(0.0)
    y_short_mae = batch["y_short_expected_mae_bps"].to(device, non_blocking=non_blocking).float().clamp_min(0.0)

    def optional_label(name: str, like: torch.Tensor) -> torch.Tensor:
        value = batch.get(name)
        if value is None:
            return torch.zeros_like(like)
        return value.to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)

    # Direction repair: turn explicit continuation/trap labels into side-specific
    # bad-path supervision. This is a learned target contract, not a live rule:
    # the model still decides, but it is punished for assigning low bad-path
    # probability to the wrong side in known support/resistance continuation traps.
    y_support_retest_continue = optional_label("y_support_retest_continuation", y_short_bad)
    y_resistance_retest_continue = optional_label("y_resistance_retest_continuation", y_long_bad)
    y_countertrend_short_trap = optional_label("y_countertrend_short_trap", y_short_bad)
    y_countertrend_long_trap = optional_label("y_countertrend_long_trap", y_long_bad)
    y_short_early_fail = optional_label("y_short_high_mae_low_mfe_early_failure", y_short_bad)
    y_long_early_fail = optional_label("y_long_high_mae_low_mfe_early_failure", y_long_bad)
    y_rising_support_touch = optional_label("y_rising_channel_support_touch", y_trade)
    y_falling_resistance_touch = optional_label("y_falling_channel_resistance_touch", y_trade)
    y_short_bad = torch.maximum(
        y_short_bad,
        torch.maximum(
            torch.maximum(y_support_retest_continue, y_countertrend_short_trap),
            y_short_early_fail,
        ),
    )
    y_long_bad = torch.maximum(
        y_long_bad,
        torch.maximum(
            torch.maximum(y_resistance_retest_continue, y_countertrend_long_trap),
            y_long_early_fail,
        ),
    )
    stats["hier_long_bad_target_rate"] = float(y_long_bad.detach().mean().cpu().item())
    stats["hier_short_bad_target_rate"] = float(y_short_bad.detach().mean().cpu().item())
    stats["hier_countertrend_long_trap_rate"] = float(y_countertrend_long_trap.detach().mean().cpu().item())
    stats["hier_countertrend_short_trap_rate"] = float(y_countertrend_short_trap.detach().mean().cpu().item())
    util_scale = max(1.0, float(ENTRY_AUX_PATH_SCALE_BPS))
    mae_scale = max(1.0, float(ENTRY_AUX_MFE_SCALE_BPS))
    valid_long_trade_target = (
        (y_long_util >= float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS))
        & (y_long_bad < 0.5)
        & (y_long_early_fail <= 0.5)
    )
    valid_short_trade_target = (
        (y_short_util >= float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS))
        & (y_short_bad < 0.5)
        & (y_short_early_fail <= 0.5)
    )
    support_touch_abstain = (
        (y_rising_support_touch > 0.5)
        & (y_support_retest_continue <= 0.5)
        & (~valid_long_trade_target)
    )
    resistance_touch_abstain = (
        (y_falling_resistance_touch > 0.5)
        & (y_resistance_retest_continue <= 0.5)
        & (~valid_short_trade_target)
    )
    long_failure_abstain = (y_long_early_fail > 0.5) & (~valid_short_trade_target)
    short_failure_abstain = (y_short_early_fail > 0.5) & (~valid_long_trade_target)
    pocket_abstain_mask = (
        support_touch_abstain
        | resistance_touch_abstain
        | long_failure_abstain
        | short_failure_abstain
    )
    anti_short_mask = (
        (y_rising_support_touch > 0.5)
        | (y_support_retest_continue > 0.5)
        | (y_countertrend_short_trap > 0.5)
        | (y_short_early_fail > 0.5)
    )
    anti_long_mask = (
        (y_falling_resistance_touch > 0.5)
        | (y_resistance_retest_continue > 0.5)
        | (y_countertrend_long_trap > 0.5)
        | (y_long_early_fail > 0.5)
    )
    stats["hier_pocket_abstain_rows"] = float(int(pocket_abstain_mask.sum().detach().cpu().item()))
    stats["hier_pocket_anti_short_rows"] = float(int(anti_short_mask.sum().detach().cpu().item()))
    stats["hier_pocket_anti_long_rows"] = float(int(anti_long_mask.sum().detach().cpu().item()))

    if trade_logit is not None and float(ENTRY_HIER_TRADE_WEIGHT) > 0.0:
        raw = nn.functional.binary_cross_entropy_with_logits(
            trade_logit.squeeze(1),
            y_trade,
            pos_weight=torch.tensor(float(trade_pos_weight), device=device, dtype=trade_logit.dtype),
        )
        weighted = float(ENTRY_HIER_TRADE_WEIGHT) * raw
        total = total + weighted
        stats["hier_trade_loss"] = float(weighted.detach().cpu().item())

    if trade_logit is not None:
        hier_trade_global_prior = _hier_trade_global_prior_match_term(trade_logit, y_trade)
        if hier_trade_global_prior.numel() == 1:
            total = total + hier_trade_global_prior
            stats["hier_trade_global_prior_loss"] = float(hier_trade_global_prior.detach().cpu().item())
        hier_slice_trade_prior = _hier_slice_trade_prior_match_term(trade_logit, y_trade, ctx_cat)
        if hier_slice_trade_prior.numel() == 1:
            total = total + hier_slice_trade_prior
            stats["hier_slice_trade_prior_loss"] = float(hier_slice_trade_prior.detach().cpu().item())
        hier_slice_trade_accuracy_edge = _hier_slice_trade_accuracy_edge_term(
            trade_logit,
            y_trade,
            ctx_cat,
        )
        if hier_slice_trade_accuracy_edge.numel() == 1:
            total = total + hier_slice_trade_accuracy_edge
            stats["hier_slice_trade_accuracy_edge_loss"] = float(
                hier_slice_trade_accuracy_edge.detach().cpu().item()
            )
        hier_flat_logit_margin = _hier_flat_logit_margin_term(trade_logit, y_trade)
        if hier_flat_logit_margin.numel() == 1:
            total = total + hier_flat_logit_margin
            stats["hier_flat_logit_margin_loss"] = float(hier_flat_logit_margin.detach().cpu().item())
        hier_slice_flat_logit_margin = _hier_slice_flat_logit_margin_term(trade_logit, y_trade, ctx_cat)
        if hier_slice_flat_logit_margin.numel() == 1:
            total = total + hier_slice_flat_logit_margin
            stats["hier_slice_flat_logit_margin_loss"] = float(
                hier_slice_flat_logit_margin.detach().cpu().item()
            )
        public_flat_source_logit = (
            public_trade_logit
            if isinstance(public_trade_logit, torch.Tensor)
            else trade_logit
        )
        hier_public_flat_consistency = _hier_public_flat_consistency_term(
            direction_logits if isinstance(direction_logits, torch.Tensor) else None,
            public_flat_source_logit,
            y_trade,
        )
        if hier_public_flat_consistency.numel() == 1:
            total = total + hier_public_flat_consistency
            stats["hier_public_flat_consistency_loss"] = float(
                hier_public_flat_consistency.detach().cpu().item()
            )
        hier_slice_public_flat_consistency = _hier_slice_public_flat_consistency_term(
            direction_logits if isinstance(direction_logits, torch.Tensor) else None,
            public_flat_source_logit,
            y_trade,
            ctx_cat,
        )
        if hier_slice_public_flat_consistency.numel() == 1:
            total = total + hier_slice_public_flat_consistency
            stats["hier_slice_public_flat_consistency_loss"] = float(
                hier_slice_public_flat_consistency.detach().cpu().item()
            )

    if public_trade_logit is not None and float(ENTRY_HIER_TRADE_WEIGHT) > 0.0:
        raw = nn.functional.binary_cross_entropy_with_logits(
            public_trade_logit.reshape(-1),
            y_trade,
            pos_weight=torch.tensor(
                float(trade_pos_weight),
                device=device,
                dtype=public_trade_logit.dtype,
            ),
        )
        weighted = float(ENTRY_HIER_TRADE_WEIGHT) * raw
        total = total + weighted
        stats["hier_public_trade_loss"] = float(weighted.detach().cpu().item())

    if public_trade_logit is not None:
        hier_public_trade_global_prior = _hier_trade_global_prior_match_term(
            public_trade_logit,
            y_trade,
        )
        if hier_public_trade_global_prior.numel() == 1:
            total = total + hier_public_trade_global_prior
            stats["hier_public_trade_global_prior_loss"] = float(
                hier_public_trade_global_prior.detach().cpu().item()
            )
        hier_public_slice_trade_prior = _hier_slice_trade_prior_match_term(
            public_trade_logit,
            y_trade,
            ctx_cat,
        )
        if hier_public_slice_trade_prior.numel() == 1:
            total = total + hier_public_slice_trade_prior
            stats["hier_public_slice_trade_prior_loss"] = float(
                hier_public_slice_trade_prior.detach().cpu().item()
            )
        hier_public_slice_trade_accuracy_edge = _hier_slice_trade_accuracy_edge_term(
            public_trade_logit,
            y_trade,
            ctx_cat,
        )
        if hier_public_slice_trade_accuracy_edge.numel() == 1:
            total = total + hier_public_slice_trade_accuracy_edge
            stats["hier_public_slice_trade_accuracy_edge_loss"] = float(
                hier_public_slice_trade_accuracy_edge.detach().cpu().item()
            )
        hier_public_flat_logit_margin = _hier_flat_logit_margin_term(public_trade_logit, y_trade)
        if hier_public_flat_logit_margin.numel() == 1:
            total = total + hier_public_flat_logit_margin
            stats["hier_public_flat_logit_margin_loss"] = float(
                hier_public_flat_logit_margin.detach().cpu().item()
            )
        hier_public_slice_flat_logit_margin = _hier_slice_flat_logit_margin_term(
            public_trade_logit,
            y_trade,
            ctx_cat,
        )
        if hier_public_slice_flat_logit_margin.numel() == 1:
            total = total + hier_public_slice_flat_logit_margin
            stats["hier_public_slice_flat_logit_margin_loss"] = float(
                hier_public_slice_flat_logit_margin.detach().cpu().item()
            )

    if (
        trade_logit is not None
        and float(ENTRY_HIER_POCKET_ABSTAIN_WEIGHT) > 0.0
        and pocket_abstain_mask.any()
    ):
        raw = nn.functional.binary_cross_entropy_with_logits(
            trade_logit.reshape(-1)[pocket_abstain_mask],
            torch.zeros_like(y_trade[pocket_abstain_mask], dtype=trade_logit.dtype),
        )
        weighted = float(ENTRY_HIER_POCKET_ABSTAIN_WEIGHT) * raw
        total = total + weighted
        stats["hier_pocket_abstain_loss"] = float(weighted.detach().cpu().item())

    if side_logits is not None:
        if float(ENTRY_HIER_SIDE_WEIGHT) > 0.0 and y_side_mask.any():
            raw = nn.functional.cross_entropy(side_logits[y_side_mask], y_side[y_side_mask])
            weighted = float(ENTRY_HIER_SIDE_WEIGHT) * raw
            total = total + weighted
            pred_side = torch.argmax(side_logits[y_side_mask], dim=1)
            stats["hier_side_rows"] = float(int(y_side_mask.sum().detach().cpu().item()))
            stats["hier_side_acc"] = float((pred_side == y_side[y_side_mask]).float().mean().detach().cpu().item())
            stats["hier_side_loss"] = float(weighted.detach().cpu().item())
        hier_slice_side_ce = _hier_slice_side_balanced_ce_term(side_logits, y_side, y_side_mask, ctx_cat)
        if hier_slice_side_ce.numel() == 1:
            total = total + hier_slice_side_ce
            stats["hier_slice_side_ce_loss"] = float(hier_slice_side_ce.detach().cpu().item())
        hier_slice_side_margin = _hier_slice_side_true_margin_term(side_logits, y_side, y_side_mask, ctx_cat)
        if hier_slice_side_margin.numel() == 1:
            total = total + hier_slice_side_margin
            stats["hier_slice_side_margin_loss"] = float(hier_slice_side_margin.detach().cpu().item())
        hier_slice_side_accuracy_edge = _hier_slice_side_accuracy_edge_term(
            side_logits,
            y_side,
            y_side_mask,
            ctx_cat,
        )
        if hier_slice_side_accuracy_edge.numel() == 1:
            total = total + hier_slice_side_accuracy_edge
            stats["hier_slice_side_accuracy_edge_loss"] = float(
                hier_slice_side_accuracy_edge.detach().cpu().item()
            )
        hier_side_global_prior = _hier_side_global_prior_match_term(side_logits, y_side, y_side_mask)
        if hier_side_global_prior.numel() == 1:
            total = total + hier_side_global_prior
            stats["hier_side_global_prior_loss"] = float(hier_side_global_prior.detach().cpu().item())
        hier_slice_side_prior = _hier_slice_side_prior_match_term(side_logits, y_side, y_side_mask, ctx_cat)
        if hier_slice_side_prior.numel() == 1:
            total = total + hier_slice_side_prior
            stats["hier_slice_side_prior_loss"] = float(hier_slice_side_prior.detach().cpu().item())

    if public_side_logits is not None and isinstance(public_side_logits, torch.Tensor):
        if float(ENTRY_HIER_SIDE_WEIGHT) > 0.0 and y_side_mask.any():
            raw = nn.functional.cross_entropy(public_side_logits[y_side_mask], y_side[y_side_mask])
            weighted = float(ENTRY_HIER_SIDE_WEIGHT) * raw
            total = total + weighted
            pred_side = torch.argmax(public_side_logits[y_side_mask], dim=1)
            stats["hier_public_side_rows"] = float(int(y_side_mask.sum().detach().cpu().item()))
            stats["hier_public_side_acc"] = float(
                (pred_side == y_side[y_side_mask]).float().mean().detach().cpu().item()
            )
            stats["hier_public_side_loss"] = float(weighted.detach().cpu().item())
        hier_public_slice_side_ce = _hier_slice_side_balanced_ce_term(
            public_side_logits,
            y_side,
            y_side_mask,
            ctx_cat,
        )
        if hier_public_slice_side_ce.numel() == 1:
            total = total + hier_public_slice_side_ce
            stats["hier_public_slice_side_ce_loss"] = float(
                hier_public_slice_side_ce.detach().cpu().item()
            )
        hier_public_slice_side_margin = _hier_slice_side_true_margin_term(
            public_side_logits,
            y_side,
            y_side_mask,
            ctx_cat,
        )
        if hier_public_slice_side_margin.numel() == 1:
            total = total + hier_public_slice_side_margin
            stats["hier_public_slice_side_margin_loss"] = float(
                hier_public_slice_side_margin.detach().cpu().item()
            )
        hier_public_slice_side_accuracy_edge = _hier_slice_side_accuracy_edge_term(
            public_side_logits,
            y_side,
            y_side_mask,
            ctx_cat,
        )
        if hier_public_slice_side_accuracy_edge.numel() == 1:
            total = total + hier_public_slice_side_accuracy_edge
            stats["hier_public_slice_side_accuracy_edge_loss"] = float(
                hier_public_slice_side_accuracy_edge.detach().cpu().item()
            )
        hier_public_side_global_prior = _hier_side_global_prior_match_term(
            public_side_logits,
            y_side,
            y_side_mask,
        )
        if hier_public_side_global_prior.numel() == 1:
            total = total + hier_public_side_global_prior
            stats["hier_public_side_global_prior_loss"] = float(
                hier_public_side_global_prior.detach().cpu().item()
            )
        hier_public_slice_side_prior = _hier_slice_side_prior_match_term(
            public_side_logits,
            y_side,
            y_side_mask,
            ctx_cat,
        )
        if hier_public_slice_side_prior.numel() == 1:
            total = total + hier_public_slice_side_prior
            stats["hier_public_slice_side_prior_loss"] = float(
                hier_public_slice_side_prior.detach().cpu().item()
            )

    if side_utility is not None and float(ENTRY_HIER_UTILITY_WEIGHT) > 0.0:
        util_target = (torch.stack([y_long_util, y_short_util], dim=1) / util_scale).to(dtype=side_utility.dtype)
        raw = nn.functional.smooth_l1_loss(side_utility, util_target)
        weighted = float(ENTRY_HIER_UTILITY_WEIGHT) * raw
        total = total + weighted
        stats["hier_utility_loss"] = float(weighted.detach().cpu().item())

    if side_bad_path_logit is not None and float(ENTRY_HIER_BAD_PATH_WEIGHT) > 0.0:
        bad_target = torch.stack([y_long_bad, y_short_bad], dim=1).to(dtype=side_bad_path_logit.dtype)
        if isinstance(side_bad_path_pos_weight, (list, tuple, np.ndarray)):
            weights = [float(x) for x in list(side_bad_path_pos_weight)[:2]]
            if len(weights) != 2:
                weights = [1.0, 1.0]
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

    if side_mae is not None and float(ENTRY_HIER_MAE_WEIGHT) > 0.0:
        mae_target = (torch.stack([y_long_mae, y_short_mae], dim=1) / mae_scale).to(dtype=side_mae.dtype)
        raw = nn.functional.smooth_l1_loss(side_mae, mae_target)
        weighted = float(ENTRY_HIER_MAE_WEIGHT) * raw
        total = total + weighted
        stats["hier_mae_loss"] = float(weighted.detach().cpu().item())

    if side_validity_logit is not None and float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT) > 0.0:
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

    pocket_margin_terms = []
    if float(ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT) > 0.0:
        margin = float(ENTRY_TRENDLINE_RAIL_MARGIN)
        if side_logits is not None and isinstance(side_logits, torch.Tensor) and side_logits.ndim == 2 and side_logits.shape[1] >= 2:
            side_logits_f = side_logits.float()
            if anti_short_mask.any():
                pocket_margin_terms.append(
                    nn.functional.softplus(
                        side_logits_f[anti_short_mask, 1] - side_logits_f[anti_short_mask, 0] + margin
                    ).mean()
                )
            if anti_long_mask.any():
                pocket_margin_terms.append(
                    nn.functional.softplus(
                        side_logits_f[anti_long_mask, 0] - side_logits_f[anti_long_mask, 1] + margin
                    ).mean()
                )
        if side_bad_path_logit is not None and isinstance(side_bad_path_logit, torch.Tensor) and side_bad_path_logit.ndim == 2 and side_bad_path_logit.shape[1] >= 2:
            side_bad_f = side_bad_path_logit.float()
            if anti_short_mask.any():
                pocket_margin_terms.append(
                    nn.functional.softplus(
                        side_bad_f[anti_short_mask, 0] - side_bad_f[anti_short_mask, 1] + margin
                    ).mean()
                )
            if anti_long_mask.any():
                pocket_margin_terms.append(
                    nn.functional.softplus(
                        side_bad_f[anti_long_mask, 1] - side_bad_f[anti_long_mask, 0] + margin
                    ).mean()
                )
        if side_utility is not None and isinstance(side_utility, torch.Tensor) and side_utility.ndim == 2 and side_utility.shape[1] >= 2:
            side_utility_f = side_utility.float()
            utility_margin = float(ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS) / util_scale
            if anti_short_mask.any():
                pocket_margin_terms.append(
                    nn.functional.softplus(
                        side_utility_f[anti_short_mask, 1] - side_utility_f[anti_short_mask, 0] + utility_margin
                    ).mean()
                )
            if anti_long_mask.any():
                pocket_margin_terms.append(
                    nn.functional.softplus(
                        side_utility_f[anti_long_mask, 0] - side_utility_f[anti_long_mask, 1] + utility_margin
                    ).mean()
                )
    if pocket_margin_terms:
        raw = torch.stack(pocket_margin_terms).mean()
        weighted = float(ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT) * raw
        total = total + weighted
        stats["hier_pocket_side_margin_loss"] = float(weighted.detach().cpu().item())

    return total, stats


def _trendline_rail_aux_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    direction_probs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    logits = out.get("trendline_rail_logits")
    stats = {
        "trendline_rail_loss": 0.0,
        "trendline_rail_rows": 0.0,
        "trendline_rising_rows": 0.0,
        "trendline_falling_rows": 0.0,
        "trendline_wrong_side_prob": 0.0,
        "trendline_rising_wrong_short_prob": 0.0,
        "trendline_falling_wrong_long_prob": 0.0,
        "trendline_final_margin_loss": 0.0,
        "trendline_hier_margin_loss": 0.0,
        "trendline_flat_trade_loss": 0.0,
        "trendline_utility_margin_loss": 0.0,
    }
    if logits is None or not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        return torch.tensor(0.0, device=device), stats
    if float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT) <= 0.0:
        return torch.tensor(0.0, device=device), stats

    non_blocking = device.type == "cuda"
    rising = batch["y_rising_channel_support_touch"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    falling = batch["y_falling_channel_resistance_touch"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    short_trap = batch["y_countertrend_short_trap"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    long_trap = batch["y_countertrend_long_trap"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
    like = rising

    def optional_label(name: str) -> torch.Tensor:
        value = batch.get(name)
        if value is None:
            return torch.zeros_like(like)
        return value.to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)

    support_continue = optional_label("y_support_retest_continuation")
    resistance_continue = optional_label("y_resistance_retest_continuation")
    short_early_fail = optional_label("y_short_high_mae_low_mfe_early_failure")
    long_early_fail = optional_label("y_long_high_mae_low_mfe_early_failure")
    target_parts = [rising, falling, short_trap, long_trap]
    if logits.ndim == 2 and logits.shape[1] >= 6:
        target_parts.extend([short_early_fail, long_early_fail])
    targets = torch.stack(target_parts, dim=1).to(dtype=logits.dtype)
    if logits.ndim != 2 or logits.shape[1] != targets.shape[1]:
        raise RuntimeError(
            "[ENTRY_TRENDLINE_RAIL_OUTPUT_DIM_MISMATCH] "
            f"logits_shape={tuple(logits.shape)} targets_shape={tuple(targets.shape)}"
        )
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets)

    anti_short_score = torch.maximum(
        torch.maximum(torch.maximum(rising, support_continue), short_trap),
        short_early_fail,
    )
    anti_long_score = torch.maximum(
        torch.maximum(torch.maximum(falling, resistance_continue), long_trap),
        long_early_fail,
    )
    rising_wrong_side = anti_short_score > 0.5
    falling_wrong_side = anti_long_score > 0.5
    wrong_side_terms = []
    weighted_wrong_side_terms = []
    if rising_wrong_side.any():
        rising_wrong_short = direction_probs[rising_wrong_side, 1].mean()
        wrong_side_terms.append(rising_wrong_short)
        weighted_wrong_side_terms.append(
            float(ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT) * rising_wrong_short
        )
        stats["trendline_rising_wrong_short_prob"] = float(rising_wrong_short.detach().cpu().item())
    if falling_wrong_side.any():
        falling_wrong_long = direction_probs[falling_wrong_side, 0].mean()
        wrong_side_terms.append(falling_wrong_long)
        weighted_wrong_side_terms.append(
            float(ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT) * falling_wrong_long
        )
        stats["trendline_falling_wrong_long_prob"] = float(falling_wrong_long.detach().cpu().item())
    wrong_side = (
        torch.stack(wrong_side_terms).mean()
        if wrong_side_terms
        else torch.tensor(0.0, device=device, dtype=direction_probs.dtype)
    )
    weighted_wrong_side = (
        torch.stack(weighted_wrong_side_terms).mean()
        if weighted_wrong_side_terms
        else torch.tensor(0.0, device=device, dtype=direction_probs.dtype)
    )
    loss = float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT) * raw
    if weighted_wrong_side_terms:
        loss = loss + weighted_wrong_side
    margin = float(ENTRY_TRENDLINE_RAIL_MARGIN)
    direction_logits = out.get("direction_logits")
    if (
        direction_logits is not None
        and isinstance(direction_logits, torch.Tensor)
        and direction_logits.ndim == 2
        and direction_logits.shape[1] >= 3
        and float(ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT) > 0.0
    ):
        dir_logits = direction_logits.float()
        final_margin_terms = []
        if rising_wrong_side.any():
            allowed = torch.maximum(dir_logits[:, 0], dir_logits[:, 2])
            short_margin = nn.functional.softplus(dir_logits[rising_wrong_side, 1] - allowed[rising_wrong_side] + margin).mean()
            final_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT) * short_margin)
        if falling_wrong_side.any():
            allowed = torch.maximum(dir_logits[:, 1], dir_logits[:, 2])
            long_margin = nn.functional.softplus(dir_logits[falling_wrong_side, 0] - allowed[falling_wrong_side] + margin).mean()
            final_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT) * long_margin)
        if final_margin_terms:
            final_margin_loss = torch.stack(final_margin_terms).mean()
            weighted_final_margin = float(ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT) * final_margin_loss
            loss = loss + weighted_final_margin
            stats["trendline_final_margin_loss"] = float(weighted_final_margin.detach().cpu().item())

    hier_margin_terms = []
    side_logits = out.get("side_logits")
    if (
        side_logits is not None
        and isinstance(side_logits, torch.Tensor)
        and side_logits.ndim == 2
        and side_logits.shape[1] >= 2
        and float(ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT) > 0.0
    ):
        side_logits_f = side_logits.float()
        if rising_wrong_side.any():
            side_short_margin = nn.functional.softplus(
                side_logits_f[rising_wrong_side, 1] - side_logits_f[rising_wrong_side, 0] + margin
            ).mean()
            hier_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT) * side_short_margin)
        if falling_wrong_side.any():
            side_long_margin = nn.functional.softplus(
                side_logits_f[falling_wrong_side, 0] - side_logits_f[falling_wrong_side, 1] + margin
            ).mean()
            hier_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT) * side_long_margin)

    utility_margin_terms = []
    side_utility = out.get("side_utility")
    if (
        side_utility is not None
        and isinstance(side_utility, torch.Tensor)
        and side_utility.ndim == 2
        and side_utility.shape[1] >= 2
        and float(ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT) > 0.0
    ):
        side_utility_f = side_utility.float()
        utility_margin = float(ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS) / max(1.0, float(ENTRY_AUX_PATH_SCALE_BPS))
        if rising_wrong_side.any():
            short_utility_margin = nn.functional.softplus(
                side_utility_f[rising_wrong_side, 1] - side_utility_f[rising_wrong_side, 0] + utility_margin
            ).mean()
            utility_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT) * short_utility_margin)
        if falling_wrong_side.any():
            long_utility_margin = nn.functional.softplus(
                side_utility_f[falling_wrong_side, 0] - side_utility_f[falling_wrong_side, 1] + utility_margin
            ).mean()
            utility_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT) * long_utility_margin)
    if utility_margin_terms:
        utility_margin_loss = torch.stack(utility_margin_terms).mean()
        weighted_utility_margin = float(ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT) * utility_margin_loss
        loss = loss + weighted_utility_margin
        stats["trendline_utility_margin_loss"] = float(weighted_utility_margin.detach().cpu().item())

    side_bad_path_logit = out.get("side_bad_path_logit")
    if (
        side_bad_path_logit is not None
        and isinstance(side_bad_path_logit, torch.Tensor)
        and side_bad_path_logit.ndim == 2
        and side_bad_path_logit.shape[1] >= 2
        and float(ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT) > 0.0
    ):
        side_bad_f = side_bad_path_logit.float()
        if rising_wrong_side.any():
            short_bad_margin = nn.functional.softplus(
                side_bad_f[rising_wrong_side, 0] - side_bad_f[rising_wrong_side, 1] + margin
            ).mean()
            hier_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT) * short_bad_margin)
        if falling_wrong_side.any():
            long_bad_margin = nn.functional.softplus(
                side_bad_f[falling_wrong_side, 1] - side_bad_f[falling_wrong_side, 0] + margin
            ).mean()
            hier_margin_terms.append(float(ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT) * long_bad_margin)
    if hier_margin_terms:
        hier_margin_loss = torch.stack(hier_margin_terms).mean()
        weighted_hier_margin = float(ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT) * hier_margin_loss
        loss = loss + weighted_hier_margin
        stats["trendline_hier_margin_loss"] = float(weighted_hier_margin.detach().cpu().item())

    trade_logit = out.get("trade_logit")
    if (
        trade_logit is not None
        and isinstance(trade_logit, torch.Tensor)
        and float(ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT) > 0.0
        and "y_trade" in batch
    ):
        y_trade = batch["y_trade"].to(device, non_blocking=non_blocking).float().clamp(0.0, 1.0)
        flat_pocket = (rising_wrong_side | falling_wrong_side) & (y_trade <= 0.5)
        if flat_pocket.any():
            flat_trade_loss = nn.functional.binary_cross_entropy_with_logits(
                trade_logit.reshape(-1)[flat_pocket],
                torch.zeros_like(y_trade[flat_pocket], dtype=trade_logit.dtype),
            )
            weighted_flat_trade = float(ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT) * flat_trade_loss
            loss = loss + weighted_flat_trade
            stats["trendline_flat_trade_loss"] = float(weighted_flat_trade.detach().cpu().item())
    stats["trendline_rail_loss"] = float(loss.detach().cpu().item())
    stats["trendline_rail_rows"] = float(int((targets.max(dim=1).values > 0.5).sum().detach().cpu().item()))
    stats["trendline_rising_rows"] = float(int(rising_wrong_side.sum().detach().cpu().item()))
    stats["trendline_falling_rows"] = float(int(falling_wrong_side.sum().detach().cpu().item()))
    stats["trendline_wrong_side_prob"] = float(wrong_side.detach().cpu().item())
    return loss, stats


def _aux_selector_mask(
    y_selector_long_mask: torch.Tensor,
    y_selector_short_mask: torch.Tensor,
) -> torch.Tensor:
    if ENTRY_SYMMETRIC_NEGATIVES:
        return (y_selector_long_mask.float() + y_selector_short_mask.float()) > 0.5
    return y_selector_long_mask.float() > 0.5


def _aux_clean_edge_target(
    y_clean_edge_long: torch.Tensor,
    y_clean_edge_bidir: torch.Tensor,
) -> torch.Tensor:
    return (y_clean_edge_bidir if ENTRY_SYMMETRIC_NEGATIVES else y_clean_edge_long).float()


def _aux_survival_target(
    y_survival_long: torch.Tensor,
    y_survival_bidir: torch.Tensor,
) -> torch.Tensor:
    return (y_survival_bidir if ENTRY_SYMMETRIC_NEGATIVES else y_survival_long).float()


def _clean_edge_rank_masks(
    y_teacher_winner_long: torch.Tensor,
    y_teacher_bad_long: torch.Tensor,
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
            (y_teacher_bad_long.float() > 0.5)
            | (y_dead_negative_long.float() > 0.5)
            | (y_teaser_negative_long.float() > 0.5)
            | (residual_hard_neg_long.float() > 0.5)
            | (y_dead_negative_short.float() > 0.5)
            | (y_teaser_negative_short.float() > 0.5)
            | (residual_hard_neg_short.float() > 0.5)
        )
        return clean_pos, ranked_neg

    clean_pos = y_teacher_winner_long.float() > 0.5
    ranked_neg = y_teacher_bad_long.float() > 0.5
    if not clean_pos.any() or not ranked_neg.any():
        clean_pos = y_clean_edge_long.float() > 0.5
        ranked_neg = (
            (y_teacher_bad_long.float() > 0.5)
            | (y_dead_negative_long.float() > 0.5)
            | (y_teaser_negative_long.float() > 0.5)
        )
    return clean_pos, ranked_neg


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    short_lead_margin: float,
    long_penalty_weight: float,
    residual_side_bias_alpha: float,
    timing_target_bps: float,
    timing_loss_scale: float,
    aux_early_weight: float,
    aux_quality_weight: float,
    aux_path_weight: float,
    aux_mfe_weight: float,
    aux_tradable_weight: float,
    aux_quality_scale_bps: float,
    aux_path_scale_bps: float,
    aux_mfe_scale_bps: float,
    tradable_pos_weight: float,
    clean_edge_pos_weight: float,
    survival_pos_weight: float,
    bad_path_pos_weight: float,
    hier_trade_pos_weight: float,
    hier_bad_path_pos_weight: Any,
    scheduler=None,  # GX1_FAST_TRAIN: cosine+warmup scheduler, stepped per opt.step()
):
    model.train()
    # V2 fast-train: gradient accumulation. Read from env so signature stays compatible.
    try:
        from gx1.utils.fast_train import grad_accum_steps_from_env
        _accum_steps = grad_accum_steps_from_env()
    except Exception:
        _accum_steps = 1
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
    total_timing = 0.0
    total_aux_early = 0.0
    total_aux_quality = 0.0
    total_aux_path = 0.0
    total_aux_mfe = 0.0
    total_aux_tradable = 0.0
    total_aux_clean_edge = 0.0
    total_aux_survival = 0.0
    total_clean_edge_rank = 0.0
    total_aux_bad_path = 0.0
    specialist_gate_loss_sum = 0.0
    specialist_gate_entropy_sum = 0.0
    specialist_gate_min_mean_sum = 0.0
    bad_path_quality_rank_loss_sum = 0.0
    path_quality_rank_loss_sum = 0.0
    n = 0
    short_total = 0
    short_pred_long = 0
    short_lead_count = 0
    short_lead_long_prob_sum = 0.0
    anchor_abs_sum = 0.0
    delta_abs_sum = 0.0
    scaled_delta_abs_sum = 0.0
    final_minus_anchor_abs_sum = 0.0
    timing_mae_sum = 0.0
    timing_penalty_sum = 0.0
    early_loss_sum = 0.0
    quality_loss_sum = 0.0
    path_loss_sum = 0.0
    mfe_loss_sum = 0.0
    tradable_loss_sum = 0.0
    clean_edge_loss_sum = 0.0
    survival_loss_sum = 0.0
    clean_edge_rank_loss_sum = 0.0
    bad_path_loss_sum = 0.0
    hard_neg_prob_loss_sum = 0.0
    tail_direction_rows = 0
    hier_trade_loss_sum = 0.0
    hier_trade_global_prior_loss_sum = 0.0
    hier_slice_trade_prior_loss_sum = 0.0
    hier_slice_trade_accuracy_edge_loss_sum = 0.0
    hier_flat_logit_margin_loss_sum = 0.0
    hier_slice_flat_logit_margin_loss_sum = 0.0
    hier_public_trade_loss_sum = 0.0
    hier_public_trade_global_prior_loss_sum = 0.0
    hier_public_slice_trade_prior_loss_sum = 0.0
    hier_public_slice_trade_accuracy_edge_loss_sum = 0.0
    hier_public_flat_logit_margin_loss_sum = 0.0
    hier_public_slice_flat_logit_margin_loss_sum = 0.0
    hier_public_flat_consistency_loss_sum = 0.0
    hier_slice_public_flat_consistency_loss_sum = 0.0
    hier_side_loss_sum = 0.0
    hier_slice_side_ce_loss_sum = 0.0
    hier_slice_side_margin_loss_sum = 0.0
    hier_slice_side_accuracy_edge_loss_sum = 0.0
    hier_side_global_prior_loss_sum = 0.0
    hier_slice_side_prior_loss_sum = 0.0
    hier_public_side_loss_sum = 0.0
    hier_public_slice_side_ce_loss_sum = 0.0
    hier_public_slice_side_margin_loss_sum = 0.0
    hier_public_slice_side_accuracy_edge_loss_sum = 0.0
    hier_public_side_global_prior_loss_sum = 0.0
    hier_public_slice_side_prior_loss_sum = 0.0
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
    hier_public_side_rows_sum = 0
    hier_public_side_correct_sum = 0.0
    trendline_rail_loss_sum = 0.0
    trendline_rail_rows_sum = 0
    trendline_rising_rows_sum = 0
    trendline_falling_rows_sum = 0
    trendline_wrong_side_prob_sum = 0.0

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
        y_teacher_bad_long = batch["y_teacher_bad_long"].to(device, non_blocking=non_blocking)
        y_teacher_winner_long = batch["y_teacher_winner_long"].to(device, non_blocking=non_blocking)
        y_selector_long_mask = batch["y_selector_long_mask"].to(device, non_blocking=non_blocking)
        # SYM (vedtak v10_symmetric_negatives_20260603): short-side selector + bidir quality
        # labels (already built in the dataset, never read by cement). Used only when symmetric.
        y_selector_short_mask = batch["y_selector_short_mask"].to(device, non_blocking=non_blocking)
        y_clean_edge_bidir = batch["y_clean_edge_bidir"].to(device, non_blocking=non_blocking)
        y_survival_bidir = batch["y_survival_bidir"].to(device, non_blocking=non_blocking)

        # Grad accum: zero_grad happens AFTER step (or at start of epoch).
        # See loss.backward() / optimizer.step() block below for the gated step.
        out = _autocast_forward(model, seq_x.device, seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **_multi_tf_kwargs_from_batch(batch, seq_x.device))
        logits = out["direction_logits"]
        path_pred = out.get("path_quality")
        mfe_pred = out.get("mfe_first_n")
        tradable_logit = out.get("tradable_logit")
        bad_path_logit = out.get("bad_path_logit")
        clean_edge_logit = out.get("clean_edge_logit")
        survival_logit = out.get("survival_logit")
        anchor_logits = out.get("anchor_logits")
        delta_logits = out.get("delta_logits")
        specialist_gate_loss, specialist_gate_stats = _specialist_gate_regularization(out, device)
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
        legacy_ce_mult = float(ENTRY_HIER_LEGACY_CE_MULT) if "trade_logit" in out else 1.0
        ce_loss = float(ENTRY_DIRECTION_CE_SCALE) * legacy_ce_mult * ce_loss_raw
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
        trendline_rail_loss, trendline_stats = _trendline_rail_aux_loss(out, batch, probs, device)
        if trendline_rail_loss.numel() == 1:
            loss = loss + trendline_rail_loss
        hard_neg_prob_loss = torch.tensor(0.0, device=device)
        dead_neg_prob_loss = torch.tensor(0.0, device=device)
        teaser_neg_prob_loss = torch.tensor(0.0, device=device)
        bad_path_prob_loss = torch.tensor(0.0, device=device)
        if residual_side_bias_alpha > 0.0 and delta_logits is not None:
            residual_gap = delta_logits[:, 0] - delta_logits[:, 1]
            residual_side_bias_loss = residual_gap.mean().pow(2)
            loss = loss + (float(residual_side_bias_alpha) * residual_side_bias_loss)
        dead_neg_mask = y_dead_negative_long.float() > 0.5
        teaser_neg_mask = y_teaser_negative_long.float() > 0.5
        bad_path_neg_mask = y_bad_path.float() > 0.5
        hard_neg_mask = residual_hard_neg_long > 0.5
        if float(ENTRY_DEAD_LONG_PROB_PENALTY) > 0.0 and dead_neg_mask.any():
            dead_neg_prob_loss = float(ENTRY_DEAD_LONG_PROB_PENALTY) * probs[dead_neg_mask, 0].mean()
            loss = loss + dead_neg_prob_loss
        if float(ENTRY_TEASER_LONG_PROB_PENALTY) > 0.0 and teaser_neg_mask.any():
            teaser_neg_prob_loss = float(ENTRY_TEASER_LONG_PROB_PENALTY) * probs[teaser_neg_mask, 0].mean()
            loss = loss + teaser_neg_prob_loss
        if float(ENTRY_BAD_PATH_PROB_PENALTY) > 0.0 and bad_path_neg_mask.any():
            bad_path_prob_loss = float(ENTRY_BAD_PATH_PROB_PENALTY) * probs[bad_path_neg_mask, 0].mean()
            loss = loss + bad_path_prob_loss
        if float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) > 0.0 and hard_neg_mask.any():
            hard_neg_prob_loss = float(ENTRY_HARD_NEG_LONG_PROB_PENALTY) * probs[hard_neg_mask, 0].mean()
            loss = loss + hard_neg_prob_loss
        # SYMMETRIC SHORT prob-penalties (vedtak v10_symmetric_negatives_20260603) — push down
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
        if aux_path_weight > 0.0 and path_pred is not None:
            if positive_mask.any():
                p_scale = max(1.0, float(aux_path_scale_bps))
                # Path-quality should be learned from premium tradable rows, not diluted by parked zeros.
                path_target = (y_path_quality[positive_mask] / p_scale).clamp(min=0.0)
                # V10 v3+ Target 2: if heteroscedastic head is active, use Gaussian NLL
                # so model learns uncertainty (high var on regime-conflict samples).
                path_log_var = out.get("path_quality_log_var")
                if path_log_var is not None:
                    # NLL = 0.5 * (log_var + (y - mu)^2 / exp(log_var))
                    mu = path_pred.squeeze(1)[positive_mask]
                    lv = path_log_var.squeeze(1)[positive_mask].clamp(min=-5.0, max=5.0)
                    sq_err = (path_target.float() - mu) ** 2
                    path_loss = 0.5 * (lv + sq_err / torch.exp(lv)).mean()
                else:
                    path_loss = nn.functional.smooth_l1_loss(
                        path_pred.squeeze(1)[positive_mask], path_target.float()
                    )
                path_loss = float(aux_path_weight) * path_loss
                loss = loss + path_loss
                path_loss_sum += float(path_loss.item()) * y.shape[0]
        if aux_mfe_weight > 0.0 and mfe_pred is not None:
            if positive_mask.any():
                m_scale = max(1.0, float(aux_mfe_scale_bps))
                mfe_target = (y_mfe_first[positive_mask] / m_scale).clamp(min=0.0)
                mfe_loss = nn.functional.smooth_l1_loss(
                    mfe_pred.squeeze(1)[positive_mask], mfe_target.float()
                )
                mfe_loss = float(aux_mfe_weight) * mfe_loss
                loss = loss + mfe_loss
                mfe_loss_sum += float(mfe_loss.item()) * y.shape[0]
        if aux_tradable_weight > 0.0 and tradable_logit is not None:
            if selector_mask.any():
                tradable_loss = nn.functional.binary_cross_entropy_with_logits(
                    tradable_logit.squeeze(1)[selector_mask],
                    y_tradable.float()[selector_mask],
                    pos_weight=torch.tensor(float(tradable_pos_weight), device=device, dtype=tradable_logit.dtype),
                )
                tradable_loss = float(aux_tradable_weight) * tradable_loss
                loss = loss + tradable_loss
                tradable_loss_sum += float(tradable_loss.item()) * y.shape[0]
        if float(ENTRY_AUX_BAD_PATH_WEIGHT) > 0.0 and bad_path_logit is not None:
            if selector_mask.any():
                bad_path_loss = nn.functional.binary_cross_entropy_with_logits(
                    bad_path_logit.squeeze(1)[selector_mask],
                    y_bad_path.float()[selector_mask],
                    pos_weight=torch.tensor(float(bad_path_pos_weight), device=device, dtype=bad_path_logit.dtype),
                )
                bad_path_loss = float(ENTRY_AUX_BAD_PATH_WEIGHT) * bad_path_loss
                loss = loss + bad_path_loss
                bad_path_loss_sum += float(bad_path_loss.item()) * y.shape[0]
        if float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) > 0.0 and clean_edge_logit is not None:
            if selector_mask.any():
                clean_edge_loss = nn.functional.binary_cross_entropy_with_logits(
                    clean_edge_logit.squeeze(1)[selector_mask],
                    clean_edge_target[selector_mask],
                    pos_weight=torch.tensor(float(clean_edge_pos_weight), device=device, dtype=clean_edge_logit.dtype),
                )
                clean_edge_loss = float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) * clean_edge_loss
                loss = loss + clean_edge_loss
                clean_edge_loss_sum += float(clean_edge_loss.item()) * y.shape[0]
        if float(ENTRY_AUX_SURVIVAL_WEIGHT) > 0.0 and survival_logit is not None:
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
            clean_edge_prob = (
                torch.sigmoid(clean_edge_logit.squeeze(1))
                if clean_edge_logit is not None
                else probs[:, 0]
            )
            clean_pos, ranked_neg = _clean_edge_rank_masks(
                y_teacher_winner_long,
                y_teacher_bad_long,
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
        if long_penalty_weight > 0.0:
            short_lead_mask = (snap_x[:, 1] - snap_x[:, 0]) >= float(short_lead_margin)
            if short_lead_mask.any():
                short_lead_count += int(short_lead_mask.sum().item())
                short_lead_long_prob = probs[short_lead_mask, 0].mean()
                short_lead_long_prob_sum += float(short_lead_long_prob.item()) * int(short_lead_mask.sum().item())
                loss = loss + float(long_penalty_weight) * short_lead_long_prob

        # V10 v3+ Target 1: tf_agreement MSE loss (only when aux head enabled in model)
        if "tf_agreement_logit" in out:
            y_tf_agreement = batch["y_tf_agreement_score"].to(device, non_blocking=non_blocking)
            tf_pred = torch.sigmoid(out["tf_agreement_logit"]).squeeze(-1)
            tf_agreement_loss = torch.nn.functional.mse_loss(tf_pred, y_tf_agreement)
            loss = loss + 0.3 * tf_agreement_loss  # weight conservative; primary task = direction. Reduced from 0.5 after first retrain to free more capacity for direction head.

        # V10 v3+ Target 3: position-size BCE-style loss (only when aux head enabled)
        if "position_size_logit" in out:
            y_pos_size = batch["y_position_size_target"].to(device, non_blocking=non_blocking)
            pos_pred = torch.sigmoid(out["position_size_logit"]).squeeze(-1)
            # MSE is fine here since target is continuous in [0,1] and well-defined.
            position_size_loss = torch.nn.functional.mse_loss(pos_pred, y_pos_size)
            loss = loss + 0.2 * position_size_loss  # secondary task; reduced from 0.3 after first retrain

        # V10 v3+ Target 4: hold-horizon MSE loss (only when aux head enabled)
        if "hold_horizon_logit" in out:
            y_hold = batch["y_hold_horizon_target"].to(device, non_blocking=non_blocking)
            hold_pred = torch.sigmoid(out["hold_horizon_logit"]).squeeze(-1)
            hold_horizon_loss = torch.nn.functional.mse_loss(hold_pred, y_hold)
            loss = loss + 0.2 * hold_horizon_loss  # reduced from 0.3 after first retrain

        # Forceful MTF→direction aux CE (2026-06-06): force the multi-TF repr to
        # predict direction (LONG/SHORT/FLAT). Mirrors the active direction
        # repair recipe: class weights, bad-path/side sample weights and
        # prediction-balance, NOT selector-masked.
        if "mtf_dir_logits" in out and float(ENTRY_MTF_DIR_AUX_WEIGHT) > 0.0:
            mtf_dir_loss = _direction_aux_ce_loss(out["mtf_dir_logits"], y, criterion, ce_sample_weight)
            loss = loss + float(ENTRY_MTF_DIR_AUX_WEIGHT) * mtf_dir_loss

        # Dip-head (18, pinball) + forecast-head (4, smooth_l1). Returns a 0-tensor
        # if heads/targets absent (gated) → harmless. Conservative weight (0.2):
        # primary task is direction; dip/forecast shape the representation.
        loss = loss + 0.2 * dip_forecast_loss(out, batch, device)

        preds = torch.argmax(probs, dim=1)
        short_mask = y == 1
        if short_mask.any():
            short_total += int(short_mask.sum().item())
            short_pred_long += int(((preds == 0) & short_mask).sum().item())
        if anchor_logits is not None and delta_logits is not None:
            residual_scale = float(getattr(model, "residual_scale", 1.0))
            scaled_delta = delta_logits * residual_scale
            anchor_abs_sum += float(anchor_logits.abs().mean().item()) * y.shape[0]
            delta_abs_sum += float(delta_logits.abs().mean().item()) * y.shape[0]
            scaled_delta_abs_sum += float(scaled_delta.abs().mean().item()) * y.shape[0]
            final_minus_anchor_abs_sum += float((logits - anchor_logits).abs().mean().item()) * y.shape[0]
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
        specialist_gate_entropy_sum += float(specialist_gate_stats.get("entropy", 0.0)) * bs
        specialist_gate_min_mean_sum += float(specialist_gate_stats.get("min_mean", 0.0)) * bs
        bad_path_quality_rank_loss_sum += float(bad_path_quality_rank_loss.detach().cpu().item()) * bs
        path_quality_rank_loss_sum += float(path_quality_rank_loss.detach().cpu().item()) * bs
        hard_neg_prob_loss_sum += float(hard_neg_prob_loss) * bs
        bad_path_loss_sum += float(bad_path_prob_loss) * bs
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
        hier_public_trade_loss_sum += float(hier_stats.get("hier_public_trade_loss", 0.0)) * bs
        hier_public_trade_global_prior_loss_sum += (
            float(hier_stats.get("hier_public_trade_global_prior_loss", 0.0)) * bs
        )
        hier_public_slice_trade_prior_loss_sum += (
            float(hier_stats.get("hier_public_slice_trade_prior_loss", 0.0)) * bs
        )
        hier_public_slice_trade_accuracy_edge_loss_sum += (
            float(hier_stats.get("hier_public_slice_trade_accuracy_edge_loss", 0.0)) * bs
        )
        hier_public_flat_logit_margin_loss_sum += (
            float(hier_stats.get("hier_public_flat_logit_margin_loss", 0.0)) * bs
        )
        hier_public_slice_flat_logit_margin_loss_sum += (
            float(hier_stats.get("hier_public_slice_flat_logit_margin_loss", 0.0)) * bs
        )
        hier_public_flat_consistency_loss_sum += (
            float(hier_stats.get("hier_public_flat_consistency_loss", 0.0)) * bs
        )
        hier_slice_public_flat_consistency_loss_sum += (
            float(hier_stats.get("hier_slice_public_flat_consistency_loss", 0.0)) * bs
        )
        hier_side_loss_sum += float(hier_stats.get("hier_side_loss", 0.0)) * bs
        hier_slice_side_ce_loss_sum += float(hier_stats.get("hier_slice_side_ce_loss", 0.0)) * bs
        hier_slice_side_margin_loss_sum += float(hier_stats.get("hier_slice_side_margin_loss", 0.0)) * bs
        hier_slice_side_accuracy_edge_loss_sum += float(
            hier_stats.get("hier_slice_side_accuracy_edge_loss", 0.0)
        ) * bs
        hier_side_global_prior_loss_sum += float(hier_stats.get("hier_side_global_prior_loss", 0.0)) * bs
        hier_slice_side_prior_loss_sum += float(hier_stats.get("hier_slice_side_prior_loss", 0.0)) * bs
        hier_public_side_loss_sum += float(hier_stats.get("hier_public_side_loss", 0.0)) * bs
        hier_public_slice_side_ce_loss_sum += float(
            hier_stats.get("hier_public_slice_side_ce_loss", 0.0)
        ) * bs
        hier_public_slice_side_margin_loss_sum += float(
            hier_stats.get("hier_public_slice_side_margin_loss", 0.0)
        ) * bs
        hier_public_slice_side_accuracy_edge_loss_sum += float(
            hier_stats.get("hier_public_slice_side_accuracy_edge_loss", 0.0)
        ) * bs
        hier_public_side_global_prior_loss_sum += float(
            hier_stats.get("hier_public_side_global_prior_loss", 0.0)
        ) * bs
        hier_public_slice_side_prior_loss_sum += float(
            hier_stats.get("hier_public_slice_side_prior_loss", 0.0)
        ) * bs
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
        _public_side_rows = int(hier_stats.get("hier_public_side_rows", 0.0))
        if _public_side_rows > 0:
            hier_public_side_rows_sum += _public_side_rows
            hier_public_side_correct_sum += (
                float(hier_stats.get("hier_public_side_acc", 0.0)) * _public_side_rows
            )
        trendline_rail_loss_sum += float(trendline_stats.get("trendline_rail_loss", 0.0)) * bs
        trendline_rail_rows_sum += int(trendline_stats.get("trendline_rail_rows", 0.0))
        trendline_rising_rows_sum += int(trendline_stats.get("trendline_rising_rows", 0.0))
        trendline_falling_rows_sum += int(trendline_stats.get("trendline_falling_rows", 0.0))
        trendline_wrong_side_prob_sum += float(trendline_stats.get("trendline_wrong_side_prob", 0.0)) * bs
        if aux_path_weight > 0.0:
            total_aux_path += float(path_loss) * bs
        if aux_mfe_weight > 0.0:
            total_aux_mfe += float(mfe_loss) * bs
        if aux_tradable_weight > 0.0:
            total_aux_tradable += float(tradable_loss) * bs
        if float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) > 0.0:
            total_aux_clean_edge += float(clean_edge_loss) * bs
        if float(ENTRY_AUX_SURVIVAL_WEIGHT) > 0.0:
            total_aux_survival += float(survival_loss) * bs
        if float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT) > 0.0:
            total_clean_edge_rank += float(clean_edge_rank_loss) * bs
        n += bs

    return total / max(1, n), {
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
        "specialist_gate_entropy_mean": (specialist_gate_entropy_sum / max(1, n)),
        "specialist_gate_min_mean": (specialist_gate_min_mean_sum / max(1, n)),
        "bad_path_quality_rank_loss_mean": (bad_path_quality_rank_loss_sum / max(1, n)),
        "path_quality_rank_loss_mean": (path_quality_rank_loss_sum / max(1, n)),
        "hard_neg_prob_loss_mean": (hard_neg_prob_loss_sum / max(1, n)),
        "bad_path_prob_loss_mean": (bad_path_loss_sum / max(1, n)),
        "aux_path_loss_mean": (total_aux_path / max(1, n)),
        "aux_mfe_loss_mean": (total_aux_mfe / max(1, n)),
        "aux_tradable_loss_mean": (total_aux_tradable / max(1, n)),
        "aux_clean_edge_loss_mean": (total_aux_clean_edge / max(1, n)),
        "aux_survival_loss_mean": (total_aux_survival / max(1, n)),
        "clean_edge_rank_loss_mean": (total_clean_edge_rank / max(1, n)),
        "short_pred_long_rate": (short_pred_long / short_total if short_total > 0 else 0.0),
        "short_lead_count": short_lead_count,
        "short_lead_long_prob_mean": (short_lead_long_prob_sum / short_lead_count if short_lead_count > 0 else 0.0),
        "anchor_abs_mean": (anchor_abs_sum / max(1, n)),
        "delta_abs_mean": (delta_abs_sum / max(1, n)),
        "scaled_delta_abs_mean": (scaled_delta_abs_sum / max(1, n)),
        "final_minus_anchor_abs_mean": (final_minus_anchor_abs_sum / max(1, n)),
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
        "hier_public_trade_loss_mean": (hier_public_trade_loss_sum / max(1, n)),
        "hier_public_trade_global_prior_loss_mean": (
            hier_public_trade_global_prior_loss_sum / max(1, n)
        ),
        "hier_public_slice_trade_prior_loss_mean": (
            hier_public_slice_trade_prior_loss_sum / max(1, n)
        ),
        "hier_public_slice_trade_accuracy_edge_loss_mean": (
            hier_public_slice_trade_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_public_flat_logit_margin_loss_mean": (
            hier_public_flat_logit_margin_loss_sum / max(1, n)
        ),
        "hier_public_slice_flat_logit_margin_loss_mean": (
            hier_public_slice_flat_logit_margin_loss_sum / max(1, n)
        ),
        "hier_public_flat_consistency_loss_mean": (
            hier_public_flat_consistency_loss_sum / max(1, n)
        ),
        "hier_slice_public_flat_consistency_loss_mean": (
            hier_slice_public_flat_consistency_loss_sum / max(1, n)
        ),
        "hier_side_loss_mean": (hier_side_loss_sum / max(1, n)),
        "hier_slice_side_ce_loss_mean": (hier_slice_side_ce_loss_sum / max(1, n)),
        "hier_slice_side_margin_loss_mean": (hier_slice_side_margin_loss_sum / max(1, n)),
        "hier_slice_side_accuracy_edge_loss_mean": (
            hier_slice_side_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_side_global_prior_loss_mean": (hier_side_global_prior_loss_sum / max(1, n)),
        "hier_slice_side_prior_loss_mean": (hier_slice_side_prior_loss_sum / max(1, n)),
        "hier_public_side_loss_mean": (hier_public_side_loss_sum / max(1, n)),
        "hier_public_slice_side_ce_loss_mean": (
            hier_public_slice_side_ce_loss_sum / max(1, n)
        ),
        "hier_public_slice_side_margin_loss_mean": (
            hier_public_slice_side_margin_loss_sum / max(1, n)
        ),
        "hier_public_slice_side_accuracy_edge_loss_mean": (
            hier_public_slice_side_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_public_side_global_prior_loss_mean": (
            hier_public_side_global_prior_loss_sum / max(1, n)
        ),
        "hier_public_slice_side_prior_loss_mean": (
            hier_public_slice_side_prior_loss_sum / max(1, n)
        ),
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
        "hier_public_side_rows": int(hier_public_side_rows_sum),
        "hier_public_side_acc": (
            hier_public_side_correct_sum / hier_public_side_rows_sum
            if hier_public_side_rows_sum > 0
            else 0.0
        ),
        "trendline_rail_loss_mean": (trendline_rail_loss_sum / max(1, n)),
        "trendline_rail_rows": int(trendline_rail_rows_sum),
        "trendline_rising_rows": int(trendline_rising_rows_sum),
        "trendline_falling_rows": int(trendline_falling_rows_sum),
        "trendline_wrong_side_prob_mean": (trendline_wrong_side_prob_sum / max(1, n)),
    }


def _aux_head_diagnostics(
    head_preds: "dict[str, list]",
    binary_labels: "dict[str, list]",
    realized: "dict[str, list]",
) -> "tuple[dict, list]":
    """V10-AUX-02 (2026-06-03 cross-model scan): read-only WARN-level build signal.

    Computes, on the accumulated validation predictions:
      (a) cross-head Spearman between aux-head predictions -> catches the rho~0.99
          head-collapse (clean_edge/survival/tradable becoming redundant);
      (b) per-head AUC vs each head's own binary label -> catches a head that stopped
          discriminating;
      (c) Spearman(head_pred, realized outcome) -> catches MIS-TARGETING such as the
          documented bad_path head predicting volatility instead of loss (bad_path prob
          should correlate NEGATIVELY with realized path_quality_bps).

    This NEVER touches loss/gradients/checkpoint-selection. It is observability only, so it
    fails SOFT: a missing head/label/dependency skips that metric with a WARN, it does not
    raise. Returns (metrics_dict, warn_messages).
    """
    metrics: "dict[str, float]" = {}
    warns: "list[str]" = []
    try:
        from scipy.stats import spearmanr  # local import: fail-soft if scipy absent
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # pragma: no cover - diagnostics only
        warns.append(f"[V10-AUX-02] diagnostics skipped (import failed: {exc})")
        return metrics, warns

    def _cat(d, k):
        vs = d.get(k) or []
        if not vs:
            return None
        try:
            arr = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in vs])
        except Exception:
            return None
        return arr if arr.size else None

    pred_arrays = {k: _cat(head_preds, k) for k in head_preds}
    pred_arrays = {k: v for k, v in pred_arrays.items() if v is not None}

    # (a) cross-head Spearman -> report the max |rho| off-diagonal + flag redundant pairs.
    names = sorted(pred_arrays.keys())
    max_abs_rho = 0.0
    max_pair = ""
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = pred_arrays[names[i]], pred_arrays[names[j]]
            if a.size != b.size or a.size < 16:
                continue
            try:
                rho = spearmanr(a, b).correlation
            except Exception:
                continue
            if rho is None or not np.isfinite(rho):
                continue
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

    # (b) per-head AUC vs its own binary label.
    for k in ("tradable", "bad_path", "clean_edge", "survival"):
        pred = pred_arrays.get(k)
        lbl = _cat(binary_labels, k)
        if pred is None or lbl is None or pred.size != lbl.size:
            continue
        ub = np.unique(lbl[np.isfinite(lbl)])
        if ub.size < 2:
            warns.append(f"[V10-AUX-02] {k} AUC skipped (label has one class)")
            continue
        m = np.isfinite(pred) & np.isfinite(lbl)
        if m.sum() < 16:
            continue
        try:
            auc = float(roc_auc_score((lbl[m] > 0.5).astype(int), pred[m]))
        except Exception as exc:
            warns.append(f"[V10-AUX-02] {k} AUC failed: {exc}")
            continue
        metrics[f"auc__{k}"] = auc
        if auc < 0.52:
            warns.append(f"[V10-AUX-02] {k} AUC={auc:.3f} (~chance => head not discriminating)")

    # (c) Spearman(head_pred, realized outcome) -> mis-targeting detector.
    for rk in realized:
        rv = _cat(realized, rk)
        if rv is None:
            continue
        for hk, pred in pred_arrays.items():
            if pred.size != rv.size or pred.size < 16:
                continue
            m = np.isfinite(pred) & np.isfinite(rv)
            if m.sum() < 16:
                continue
            try:
                rho = spearmanr(pred[m], rv[m]).correlation
            except Exception:
                continue
            if rho is None or not np.isfinite(rho):
                continue
            metrics[f"realized_rho__{hk}__{rk}"] = float(rho)
            # Documented failure: bad_path (prob of a BAD path) should be NEGATIVELY
            # correlated with realized path_quality_bps. A positive sign = anti-targeted.
            if hk == "bad_path" and rk == "path_quality_bps" and rho > -0.02:
                warns.append(
                    f"[V10-AUX-02] bad_path ANTI-TARGETED: Spearman(bad_path, "
                    f"path_quality_bps)={rho:+.3f} (expected strongly negative; "
                    f"head may be predicting volatility, not loss)"
                )
    return metrics, warns


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
    targets_i = np.asarray(targets_np, dtype=np.int64).reshape(-1)
    preds_i = np.asarray(preds_np, dtype=np.int64).reshape(-1)
    if ctx_cat_np is None or targets_i.size <= 0 or preds_i.size != targets_i.size:
        return {
            "direction_slice_audited_count": 0,
            "direction_slice_failure_count": 0,
            "direction_slice_accuracy_failure_count": 0,
            "direction_slice_pred_rate_failure_count": 0,
            "direction_slice_accuracy_deficit": 0.0,
            "direction_slice_pred_rate_shortfall": 0.0,
            "direction_slice_failure_details": [],
        }

    cat = np.asarray(ctx_cat_np)
    if cat.ndim == 1:
        cat = cat.reshape(-1, 1)
    if cat.ndim != 2 or cat.shape[0] != targets_i.size:
        return {
            "direction_slice_audited_count": 0,
            "direction_slice_failure_count": 0,
            "direction_slice_accuracy_failure_count": 0,
            "direction_slice_pred_rate_failure_count": 0,
            "direction_slice_accuracy_deficit": 0.0,
            "direction_slice_pred_rate_shortfall": 0.0,
            "direction_slice_failure_details": [],
        }

    min_rows = max(_DIRECTION_AUDIT_MIN_SLICE_ROWS, int(ENTRY_DIRECTION_SLICE_MIN_ROWS))
    min_label_rate = max(_DIRECTION_AUDIT_MIN_LABEL_RATE, float(ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE))
    min_pred_rate = max(_DIRECTION_AUDIT_MIN_PRED_RATE, float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE))
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
        for raw_value in sorted(set(int(v) for v in values[finite].ravel())):
            mask = finite & (cat[:, idx].astype(np.int64, copy=False) == int(raw_value))
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
            acc = float(np.mean(preds_s == labels_s)) if rows > 0 else 0.0
            majority = float(label_rates.max()) if label_rates.size else 0.0
            acc_deficit = max(0.0, majority - acc)
            accuracy_failed = bool(acc <= majority + 1e-12)
            if accuracy_failed:
                accuracy_failures += 1
                accuracy_deficit += acc_deficit
            required = np.maximum(label_rates * pred_to_label, min_pred_rate)
            shortfalls = np.maximum(required[active] - pred_rates[active], 0.0)
            pred_rate_failed = shortfalls > 1e-12
            pred_rate_failures += int(np.sum(pred_rate_failed))
            pred_rate_shortfall += float(shortfalls.sum())
            if accuracy_failed or bool(np.any(pred_rate_failed)):
                active_classes = np.flatnonzero(active)
                pred_failed_classes = [
                    int(cls)
                    for cls, failed in zip(active_classes.tolist(), pred_rate_failed.tolist())
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
                        "rows": int(rows),
                        "accuracy": float(acc),
                        "majority": float(majority),
                        "accuracy_failed": bool(accuracy_failed),
                        "accuracy_deficit": float(acc_deficit),
                        "label_rates": [float(v) for v in label_rates[:3].tolist()],
                        "pred_rates": [float(v) for v in pred_rates[:3].tolist()],
                        "required_pred_rates": [float(v) for v in required[:3].tolist()],
                        "pred_rate_failed_classes": pred_failed_classes,
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
        + _DIRECTION_SLICE_CKPT_DEFICIT_PENALTY * (rate_shortfall + acc_deficit)
    )


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
    residual_side_bias_alpha: float,
    aux_early_weight: float,
    aux_quality_weight: float,
    aux_path_weight: float,
    aux_mfe_weight: float,
    aux_tradable_weight: float,
    aux_quality_scale_bps: float,
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
    bad_path_quality_rank_loss_sum = 0.0
    path_quality_rank_loss_sum = 0.0
    n = 0
    preds, targets = [], []
    ctx_cats: List[np.ndarray] = []
    short_total = 0
    short_pred_long = 0
    anchor_abs_sum = 0.0
    delta_abs_sum = 0.0
    scaled_delta_abs_sum = 0.0
    final_minus_anchor_abs_sum = 0.0
    early_loss_sum = 0.0
    quality_loss_sum = 0.0
    path_loss_sum = 0.0
    mfe_loss_sum = 0.0
    tradable_loss_sum = 0.0
    clean_edge_loss_sum = 0.0
    survival_loss_sum = 0.0
    clean_edge_rank_loss_sum = 0.0
    bad_path_loss_sum = 0.0
    hard_neg_prob_loss_sum = 0.0
    tail_direction_rows = 0
    hier_trade_loss_sum = 0.0
    hier_trade_global_prior_loss_sum = 0.0
    hier_slice_trade_prior_loss_sum = 0.0
    hier_slice_trade_accuracy_edge_loss_sum = 0.0
    hier_flat_logit_margin_loss_sum = 0.0
    hier_slice_flat_logit_margin_loss_sum = 0.0
    hier_public_trade_loss_sum = 0.0
    hier_public_trade_global_prior_loss_sum = 0.0
    hier_public_slice_trade_prior_loss_sum = 0.0
    hier_public_slice_trade_accuracy_edge_loss_sum = 0.0
    hier_public_flat_logit_margin_loss_sum = 0.0
    hier_public_slice_flat_logit_margin_loss_sum = 0.0
    hier_public_flat_consistency_loss_sum = 0.0
    hier_slice_public_flat_consistency_loss_sum = 0.0
    hier_side_loss_sum = 0.0
    hier_slice_side_ce_loss_sum = 0.0
    hier_slice_side_margin_loss_sum = 0.0
    hier_slice_side_accuracy_edge_loss_sum = 0.0
    hier_side_global_prior_loss_sum = 0.0
    hier_slice_side_prior_loss_sum = 0.0
    hier_public_side_loss_sum = 0.0
    hier_public_slice_side_ce_loss_sum = 0.0
    hier_public_slice_side_margin_loss_sum = 0.0
    hier_public_slice_side_accuracy_edge_loss_sum = 0.0
    hier_public_side_global_prior_loss_sum = 0.0
    hier_public_slice_side_prior_loss_sum = 0.0
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
    hier_public_side_rows_sum = 0
    hier_public_side_correct_sum = 0.0
    trendline_rail_loss_sum = 0.0
    trendline_rail_rows_sum = 0
    trendline_rising_rows_sum = 0
    trendline_falling_rows_sum = 0
    trendline_wrong_side_prob_sum = 0.0
    # V10-AUX-02: read-only accumulators for the cross-head / AUC / realized-target panel.
    _diag_pred: "dict[str, list]" = {k: [] for k in (
        "tradable", "bad_path", "clean_edge", "survival", "path_quality", "mfe_first_n")}
    _diag_lbl: "dict[str, list]" = {k: [] for k in ("tradable", "bad_path", "clean_edge", "survival")}
    _diag_real: "dict[str, list]" = {k: [] for k in ("mfe_first_n_bps", "path_quality_bps")}
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
            y_teacher_bad_long = batch["y_teacher_bad_long"].to(device, non_blocking=non_blocking)
            y_teacher_winner_long = batch["y_teacher_winner_long"].to(device, non_blocking=non_blocking)
            y_selector_long_mask = batch["y_selector_long_mask"].to(device, non_blocking=non_blocking)
            y_selector_short_mask = batch["y_selector_short_mask"].to(device, non_blocking=non_blocking)
            y_clean_edge_bidir = batch["y_clean_edge_bidir"].to(device, non_blocking=non_blocking)
            y_survival_bidir = batch["y_survival_bidir"].to(device, non_blocking=non_blocking)

            out = _autocast_forward(model, seq_x.device, seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **_multi_tf_kwargs_from_batch(batch, seq_x.device))
            logits = out["direction_logits"]
            path_pred = out.get("path_quality")
            mfe_pred = out.get("mfe_first_n")
            tradable_logit = out.get("tradable_logit")
            bad_path_logit = out.get("bad_path_logit")
            clean_edge_logit = out.get("clean_edge_logit")
            survival_logit = out.get("survival_logit")
            trade_logit = out.get("trade_logit")
            public_trade_logit = out.get("public_trade_logit")
            side_logits = out.get("side_logits")
            public_side_logits = out.get("public_side_logits")
            anchor_logits = out.get("anchor_logits")
            delta_logits = out.get("delta_logits")
            specialist_gate_loss, _specialist_gate_stats = _specialist_gate_regularization(out, device)
            bad_path_quality_rank_loss = _bad_path_quality_rank_loss(bad_path_logit, y_path_quality, device)
            path_quality_rank_loss = _path_quality_rank_loss(path_pred, y_path_quality, device)

            # V10-AUX-02: accumulate per-head probs/preds + labels + realized targets for
            # the read-only diagnostic panel (computed after the loop). Detached, no grad.
            def _np1d(t):
                return t.detach().float().cpu().numpy().reshape(-1)
            if tradable_logit is not None:
                _diag_pred["tradable"].append(_np1d(torch.sigmoid(tradable_logit)))
            if bad_path_logit is not None:
                _diag_pred["bad_path"].append(_np1d(torch.sigmoid(bad_path_logit)))
            if clean_edge_logit is not None:
                _diag_pred["clean_edge"].append(_np1d(torch.sigmoid(clean_edge_logit)))
            if survival_logit is not None:
                _diag_pred["survival"].append(_np1d(torch.sigmoid(survival_logit)))
            if path_pred is not None:
                _diag_pred["path_quality"].append(_np1d(path_pred))
            if mfe_pred is not None:
                _diag_pred["mfe_first_n"].append(_np1d(mfe_pred))
            trade_metric_logit = (
                public_trade_logit
                if isinstance(public_trade_logit, torch.Tensor)
                else trade_logit
            )
            if trade_metric_logit is not None:
                hierarchy_trade_prob_chunks.append(_np1d(torch.sigmoid(trade_metric_logit)))
            side_metric_logits = (
                public_side_logits
                if (
                    public_side_logits is not None
                    and isinstance(public_side_logits, torch.Tensor)
                    and public_side_logits.ndim == 2
                )
                else side_logits
            )
            if (
                side_metric_logits is not None
                and isinstance(side_metric_logits, torch.Tensor)
                and side_metric_logits.ndim == 2
            ):
                side_probs = torch.softmax(side_metric_logits.float(), dim=1)
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
            legacy_ce_mult = float(ENTRY_HIER_LEGACY_CE_MULT) if "trade_logit" in out else 1.0
            ce_loss = float(ENTRY_DIRECTION_CE_SCALE) * legacy_ce_mult * ce_loss_raw
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
            trendline_rail_loss, trendline_stats = _trendline_rail_aux_loss(out, batch, probs, device)
            if trendline_rail_loss.numel() == 1:
                loss = loss + trendline_rail_loss
            hard_neg_prob_loss = torch.tensor(0.0, device=device)
            dead_neg_prob_loss = torch.tensor(0.0, device=device)
            teaser_neg_prob_loss = torch.tensor(0.0, device=device)
            bad_path_prob_loss = torch.tensor(0.0, device=device)
            if residual_side_bias_alpha > 0.0 and delta_logits is not None:
                residual_gap = delta_logits[:, 0] - delta_logits[:, 1]
                residual_side_bias_loss = residual_gap.mean().pow(2)
                loss = loss + (float(residual_side_bias_alpha) * residual_side_bias_loss)
            dead_neg_mask = y_dead_negative_long.float() > 0.5
            teaser_neg_mask = y_teaser_negative_long.float() > 0.5
            bad_path_neg_mask = y_bad_path.float() > 0.5
            hard_neg_mask = residual_hard_neg_long > 0.5
            if float(ENTRY_DEAD_LONG_PROB_PENALTY) > 0.0 and dead_neg_mask.any():
                dead_neg_prob_loss = float(ENTRY_DEAD_LONG_PROB_PENALTY) * probs[dead_neg_mask, 0].mean()
                loss = loss + dead_neg_prob_loss
            if float(ENTRY_TEASER_LONG_PROB_PENALTY) > 0.0 and teaser_neg_mask.any():
                teaser_neg_prob_loss = float(ENTRY_TEASER_LONG_PROB_PENALTY) * probs[teaser_neg_mask, 0].mean()
                loss = loss + teaser_neg_prob_loss
            if float(ENTRY_BAD_PATH_PROB_PENALTY) > 0.0 and bad_path_neg_mask.any():
                bad_path_prob_loss = float(ENTRY_BAD_PATH_PROB_PENALTY) * probs[bad_path_neg_mask, 0].mean()
                loss = loss + bad_path_prob_loss
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
            bad_path_loss = torch.tensor(0.0, device=device)
            path_loss = torch.tensor(0.0, device=device)
            mfe_loss = torch.tensor(0.0, device=device)
            positive_mask = y_tradable.float() > 0.5
            selector_mask = _aux_selector_mask(y_selector_long_mask, y_selector_short_mask)
            clean_edge_target = _aux_clean_edge_target(y_clean_edge_long, y_clean_edge_bidir)
            survival_target = _aux_survival_target(y_survival_long, y_survival_bidir)
            if aux_path_weight > 0.0 and path_pred is not None:
                if positive_mask.any():
                    p_scale = max(1.0, float(aux_path_scale_bps))
                    path_target = (y_path_quality[positive_mask] / p_scale).clamp(min=0.0)
                    # V10 v3+ Target 2: heteroscedastic path_quality NLL (val)
                    path_log_var = out.get("path_quality_log_var")
                    if path_log_var is not None:
                        mu = path_pred.squeeze(1)[positive_mask]
                        lv = path_log_var.squeeze(1)[positive_mask].clamp(min=-5.0, max=5.0)
                        sq_err = (path_target.float() - mu) ** 2
                        path_loss = 0.5 * (lv + sq_err / torch.exp(lv)).mean()
                    else:
                        path_loss = nn.functional.smooth_l1_loss(
                            path_pred.squeeze(1)[positive_mask], path_target.float()
                        )
                    loss = loss + (float(aux_path_weight) * path_loss)
                    path_loss_sum += float(path_loss.item()) * y.shape[0]
            if aux_mfe_weight > 0.0 and mfe_pred is not None:
                if positive_mask.any():
                    m_scale = max(1.0, float(aux_mfe_scale_bps))
                    mfe_target = (y_mfe_first[positive_mask] / m_scale).clamp(min=0.0)
                    mfe_loss = nn.functional.smooth_l1_loss(
                        mfe_pred.squeeze(1)[positive_mask], mfe_target.float()
                    )
                    loss = loss + (float(aux_mfe_weight) * mfe_loss)
                    mfe_loss_sum += float(mfe_loss.item()) * y.shape[0]
            if aux_tradable_weight > 0.0 and tradable_logit is not None:
                if selector_mask.any():
                    tradable_loss = nn.functional.binary_cross_entropy_with_logits(
                        tradable_logit.squeeze(1)[selector_mask],
                        y_tradable.float()[selector_mask],
                        pos_weight=torch.tensor(float(tradable_pos_weight), device=device, dtype=tradable_logit.dtype),
                    )
                    loss = loss + (float(aux_tradable_weight) * tradable_loss)
                    tradable_loss_sum += float(tradable_loss.item()) * y.shape[0]
            if float(ENTRY_AUX_BAD_PATH_WEIGHT) > 0.0 and bad_path_logit is not None:
                if selector_mask.any():
                    bad_path_loss = nn.functional.binary_cross_entropy_with_logits(
                        bad_path_logit.squeeze(1)[selector_mask],
                        y_bad_path.float()[selector_mask],
                        pos_weight=torch.tensor(float(bad_path_pos_weight), device=device, dtype=bad_path_logit.dtype),
                    )
                    loss = loss + (float(ENTRY_AUX_BAD_PATH_WEIGHT) * bad_path_loss)
                    bad_path_loss_sum += float(bad_path_loss.item()) * y.shape[0]
            if float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) > 0.0 and clean_edge_logit is not None:
                if selector_mask.any():
                    clean_edge_loss = nn.functional.binary_cross_entropy_with_logits(
                        clean_edge_logit.squeeze(1)[selector_mask],
                        clean_edge_target[selector_mask],
                        pos_weight=torch.tensor(float(clean_edge_pos_weight), device=device, dtype=clean_edge_logit.dtype),
                    )
                    loss = loss + (float(ENTRY_AUX_CLEAN_EDGE_WEIGHT) * clean_edge_loss)
                    clean_edge_loss_sum += float(clean_edge_loss.item()) * y.shape[0]
            if float(ENTRY_AUX_SURVIVAL_WEIGHT) > 0.0 and survival_logit is not None:
                if selector_mask.any():
                    survival_loss = nn.functional.binary_cross_entropy_with_logits(
                        survival_logit.squeeze(1)[selector_mask],
                        survival_target[selector_mask],
                        pos_weight=torch.tensor(float(survival_pos_weight), device=device, dtype=survival_logit.dtype),
                    )
                    loss = loss + (float(ENTRY_AUX_SURVIVAL_WEIGHT) * survival_loss)
                    survival_loss_sum += float(survival_loss.item()) * y.shape[0]
            if float(ENTRY_CLEAN_EDGE_RANKING_WEIGHT) > 0.0:
                clean_edge_prob = (
                    torch.sigmoid(clean_edge_logit.squeeze(1))
                    if clean_edge_logit is not None
                    else probs[:, 0]
                )
                clean_pos, ranked_neg = _clean_edge_rank_masks(
                    y_teacher_winner_long,
                    y_teacher_bad_long,
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
            if float(XGB_SHORT_LONG_PENALTY) > 0.0:
                short_lead_mask = (snap_x[:, 1] - snap_x[:, 0]) >= float(XGB_SHORT_LEAD_MARGIN)
                if short_lead_mask.any():
                    loss = loss + float(XGB_SHORT_LONG_PENALTY) * probs[short_lead_mask, 0].mean()
            if "tf_agreement_logit" in out:
                y_tf_agreement = batch["y_tf_agreement_score"].to(device, non_blocking=non_blocking)
                tf_pred = torch.sigmoid(out["tf_agreement_logit"]).squeeze(-1)
                loss = loss + 0.3 * torch.nn.functional.mse_loss(tf_pred, y_tf_agreement)
            if "position_size_logit" in out:
                y_pos_size = batch["y_position_size_target"].to(device, non_blocking=non_blocking)
                pos_pred = torch.sigmoid(out["position_size_logit"]).squeeze(-1)
                loss = loss + 0.2 * torch.nn.functional.mse_loss(pos_pred, y_pos_size)
            if "hold_horizon_logit" in out:
                y_hold = batch["y_hold_horizon_target"].to(device, non_blocking=non_blocking)
                hold_pred = torch.sigmoid(out["hold_horizon_logit"]).squeeze(-1)
                loss = loss + 0.2 * torch.nn.functional.mse_loss(hold_pred, y_hold)
            if "mtf_dir_logits" in out and float(ENTRY_MTF_DIR_AUX_WEIGHT) > 0.0:
                mtf_dir_loss = _direction_aux_ce_loss(out["mtf_dir_logits"], y, criterion, ce_sample_weight)
                loss = loss + float(ENTRY_MTF_DIR_AUX_WEIGHT) * mtf_dir_loss
            loss = loss + 0.2 * dip_forecast_loss(out, batch, device)
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
            bad_path_quality_rank_loss_sum += float(bad_path_quality_rank_loss.detach().cpu().item()) * bs
            path_quality_rank_loss_sum += float(path_quality_rank_loss.detach().cpu().item()) * bs
            hard_neg_prob_loss_sum += float(hard_neg_prob_loss) * bs
            bad_path_loss_sum += float(bad_path_prob_loss) * bs
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
            hier_public_trade_loss_sum += float(hier_stats.get("hier_public_trade_loss", 0.0)) * bs
            hier_public_trade_global_prior_loss_sum += (
                float(hier_stats.get("hier_public_trade_global_prior_loss", 0.0)) * bs
            )
            hier_public_slice_trade_prior_loss_sum += (
                float(hier_stats.get("hier_public_slice_trade_prior_loss", 0.0)) * bs
            )
            hier_public_slice_trade_accuracy_edge_loss_sum += (
                float(hier_stats.get("hier_public_slice_trade_accuracy_edge_loss", 0.0)) * bs
            )
            hier_public_flat_logit_margin_loss_sum += (
                float(hier_stats.get("hier_public_flat_logit_margin_loss", 0.0)) * bs
            )
            hier_public_slice_flat_logit_margin_loss_sum += (
                float(hier_stats.get("hier_public_slice_flat_logit_margin_loss", 0.0)) * bs
            )
            hier_public_flat_consistency_loss_sum += (
                float(hier_stats.get("hier_public_flat_consistency_loss", 0.0)) * bs
            )
            hier_slice_public_flat_consistency_loss_sum += (
                float(hier_stats.get("hier_slice_public_flat_consistency_loss", 0.0)) * bs
            )
            hier_side_loss_sum += float(hier_stats.get("hier_side_loss", 0.0)) * bs
            hier_slice_side_ce_loss_sum += float(hier_stats.get("hier_slice_side_ce_loss", 0.0)) * bs
            hier_slice_side_margin_loss_sum += float(hier_stats.get("hier_slice_side_margin_loss", 0.0)) * bs
            hier_slice_side_accuracy_edge_loss_sum += float(
                hier_stats.get("hier_slice_side_accuracy_edge_loss", 0.0)
            ) * bs
            hier_side_global_prior_loss_sum += float(hier_stats.get("hier_side_global_prior_loss", 0.0)) * bs
            hier_slice_side_prior_loss_sum += float(hier_stats.get("hier_slice_side_prior_loss", 0.0)) * bs
            hier_public_side_loss_sum += float(hier_stats.get("hier_public_side_loss", 0.0)) * bs
            hier_public_slice_side_ce_loss_sum += float(
                hier_stats.get("hier_public_slice_side_ce_loss", 0.0)
            ) * bs
            hier_public_slice_side_margin_loss_sum += float(
                hier_stats.get("hier_public_slice_side_margin_loss", 0.0)
            ) * bs
            hier_public_slice_side_accuracy_edge_loss_sum += float(
                hier_stats.get("hier_public_slice_side_accuracy_edge_loss", 0.0)
            ) * bs
            hier_public_side_global_prior_loss_sum += float(
                hier_stats.get("hier_public_side_global_prior_loss", 0.0)
            ) * bs
            hier_public_slice_side_prior_loss_sum += float(
                hier_stats.get("hier_public_slice_side_prior_loss", 0.0)
            ) * bs
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
            _public_side_rows = int(hier_stats.get("hier_public_side_rows", 0.0))
            if _public_side_rows > 0:
                hier_public_side_rows_sum += _public_side_rows
                hier_public_side_correct_sum += (
                    float(hier_stats.get("hier_public_side_acc", 0.0)) * _public_side_rows
                )
            trendline_rail_loss_sum += float(trendline_stats.get("trendline_rail_loss", 0.0)) * bs
            trendline_rail_rows_sum += int(trendline_stats.get("trendline_rail_rows", 0.0))
            trendline_rising_rows_sum += int(trendline_stats.get("trendline_rising_rows", 0.0))
            trendline_falling_rows_sum += int(trendline_stats.get("trendline_falling_rows", 0.0))
            trendline_wrong_side_prob_sum += float(trendline_stats.get("trendline_wrong_side_prob", 0.0)) * bs
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
            if anchor_logits is not None and delta_logits is not None:
                residual_scale = float(getattr(model, "residual_scale", 1.0))
                scaled_delta = delta_logits * residual_scale
                anchor_abs_sum += float(anchor_logits.abs().mean().item()) * bs
                delta_abs_sum += float(delta_logits.abs().mean().item()) * bs
                scaled_delta_abs_sum += float(scaled_delta.abs().mean().item()) * bs
                final_minus_anchor_abs_sum += float((logits - anchor_logits).abs().mean().item()) * bs

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
    stats = {
        "anchor_abs_mean": (anchor_abs_sum / max(1, n)),
        "delta_abs_mean": (delta_abs_sum / max(1, n)),
        "scaled_delta_abs_mean": (scaled_delta_abs_sum / max(1, n)),
        "final_minus_anchor_abs_mean": (final_minus_anchor_abs_sum / max(1, n)),
        "aux_path_loss_mean": (path_loss_sum / max(1, n)),
        "aux_mfe_loss_mean": (mfe_loss_sum / max(1, n)),
        "aux_tradable_loss_mean": (tradable_loss_sum / max(1, n)),
        "bad_path_prob_loss_mean": (bad_path_loss_sum / max(1, n)),
        "aux_clean_edge_loss_mean": (clean_edge_loss_sum / max(1, n)),
        "aux_survival_loss_mean": (survival_loss_sum / max(1, n)),
        "clean_edge_rank_loss_mean": (clean_edge_rank_loss_sum / max(1, n)),
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
        "bad_path_quality_rank_loss_mean": (bad_path_quality_rank_loss_sum / max(1, n)),
        "path_quality_rank_loss_mean": (path_quality_rank_loss_sum / max(1, n)),
        "hard_neg_prob_loss_mean": (hard_neg_prob_loss_sum / max(1, n)),
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
        "hier_public_trade_loss_mean": (hier_public_trade_loss_sum / max(1, n)),
        "hier_public_trade_global_prior_loss_mean": (
            hier_public_trade_global_prior_loss_sum / max(1, n)
        ),
        "hier_public_slice_trade_prior_loss_mean": (
            hier_public_slice_trade_prior_loss_sum / max(1, n)
        ),
        "hier_public_slice_trade_accuracy_edge_loss_mean": (
            hier_public_slice_trade_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_public_flat_logit_margin_loss_mean": (
            hier_public_flat_logit_margin_loss_sum / max(1, n)
        ),
        "hier_public_slice_flat_logit_margin_loss_mean": (
            hier_public_slice_flat_logit_margin_loss_sum / max(1, n)
        ),
        "hier_public_flat_consistency_loss_mean": (
            hier_public_flat_consistency_loss_sum / max(1, n)
        ),
        "hier_slice_public_flat_consistency_loss_mean": (
            hier_slice_public_flat_consistency_loss_sum / max(1, n)
        ),
        "hier_side_loss_mean": (hier_side_loss_sum / max(1, n)),
        "hier_slice_side_ce_loss_mean": (hier_slice_side_ce_loss_sum / max(1, n)),
        "hier_slice_side_margin_loss_mean": (hier_slice_side_margin_loss_sum / max(1, n)),
        "hier_slice_side_accuracy_edge_loss_mean": (
            hier_slice_side_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_side_global_prior_loss_mean": (hier_side_global_prior_loss_sum / max(1, n)),
        "hier_slice_side_prior_loss_mean": (hier_slice_side_prior_loss_sum / max(1, n)),
        "hier_public_side_loss_mean": (hier_public_side_loss_sum / max(1, n)),
        "hier_public_slice_side_ce_loss_mean": (
            hier_public_slice_side_ce_loss_sum / max(1, n)
        ),
        "hier_public_slice_side_margin_loss_mean": (
            hier_public_slice_side_margin_loss_sum / max(1, n)
        ),
        "hier_public_slice_side_accuracy_edge_loss_mean": (
            hier_public_slice_side_accuracy_edge_loss_sum / max(1, n)
        ),
        "hier_public_side_global_prior_loss_mean": (
            hier_public_side_global_prior_loss_sum / max(1, n)
        ),
        "hier_public_slice_side_prior_loss_mean": (
            hier_public_slice_side_prior_loss_sum / max(1, n)
        ),
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
        "hier_public_side_rows": int(hier_public_side_rows_sum),
        "hier_public_side_acc": (
            hier_public_side_correct_sum / hier_public_side_rows_sum
            if hier_public_side_rows_sum > 0
            else 0.0
        ),
        "trendline_rail_loss_mean": (trendline_rail_loss_sum / max(1, n)),
        "trendline_rail_rows": int(trendline_rail_rows_sum),
        "trendline_rising_rows": int(trendline_rising_rows_sum),
        "trendline_falling_rows": int(trendline_falling_rows_sum),
        "trendline_wrong_side_prob_mean": (trendline_wrong_side_prob_sum / max(1, n)),
    }
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
    # V10-AUX-02: cross-head / AUC / realized-target diagnostics. Fail-soft, WARN-level,
    # does NOT affect the returned loss or any checkpoint-selection metric.
    try:
        _diag_metrics, _diag_warns = _aux_head_diagnostics(_diag_pred, _diag_lbl, _diag_real)
        stats.update(_diag_metrics)
        for _w in _diag_warns:
            log.warning(_w)
    except Exception as _exc:  # pragma: no cover - diagnostics must never break val
        log.warning(f"[V10-AUX-02] head diagnostics panel failed (ignored): {_exc}")
    # AUC is intentionally disabled for this 3-class path (previously hardcoded 0.0)
    return total / max(1, n), float("nan"), acc, short_pred_long_rate, stats


def _validate_eval(model, loader, criterion, device, residual_side_bias_alpha: float):
    """
    Eval with non-finite guard; returns loss/acc and raises on NaN/Inf.
    """
    model.eval()
    total = 0.0
    n = 0
    preds, targets = [], []
    session_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
    anchor_counts = {"long": 0, "short": 0, "flat": 0}
    final_counts = {"long": 0, "short": 0, "flat": 0}
    flip_counts = {
        "anchor_short_to_short": 0,
        "anchor_short_to_long": 0,
        "anchor_long_to_short": 0,
        "anchor_long_to_long": 0,
        "anchor_flat_to_long": 0,
        "anchor_flat_to_short": 0,
    }
    anchor_counts_by_session = {name: {"long": 0, "short": 0, "flat": 0} for name in session_map.values()}
    final_counts_by_session = {name: {"long": 0, "short": 0, "flat": 0} for name in session_map.values()}
    flip_counts_by_session = {
        name: {k: 0 for k in flip_counts.keys()} for name in session_map.values()
    }
    residual_gap_chunks: List[np.ndarray] = []
    anchor_gap_chunks: List[np.ndarray] = []
    ratio_chunks: List[np.ndarray] = []
    ratio_eps = 1e-8

    with torch.no_grad():
        for batch in loader:
            non_blocking = device.type == "cuda"
            seq_x = batch["seq_x"].to(device, non_blocking=non_blocking)
            snap_x = batch["snap_x"].to(device, non_blocking=non_blocking)
            ctx_cont = batch["ctx_cont"].to(device, non_blocking=non_blocking)
            ctx_cat = batch["ctx_cat"].to(device, non_blocking=non_blocking)
            y = batch["y"].to(device, non_blocking=non_blocking)

            out = _autocast_forward(model, seq_x.device, seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **_multi_tf_kwargs_from_batch(batch, seq_x.device))
            logits = out["direction_logits"]
            anchor_logits = out.get("anchor_logits")
            delta_logits = out.get("delta_logits")

            if not torch.isfinite(logits).all():
                raise RuntimeError("[EVAL_NON_FINITE] logits contain non-finite values")

            loss = float(ENTRY_DIRECTION_CE_SCALE) * criterion(logits, y)
            if residual_side_bias_alpha > 0.0 and delta_logits is not None:
                residual_gap = delta_logits[:, 0] - delta_logits[:, 1]
                residual_side_bias_loss = residual_gap.mean().pow(2)
                loss = loss + (float(residual_side_bias_alpha) * residual_side_bias_loss)
            bs = y.shape[0]
            total += float(loss) * bs
            n += bs

            prob = torch.softmax(logits, dim=1)
            if not torch.isfinite(prob).all():
                raise RuntimeError("[EVAL_NON_FINITE] probs contain non-finite values")

            preds.extend(np.argmax(prob.cpu().numpy(), axis=1).tolist())
            targets.extend(y.cpu().numpy().tolist())

            if anchor_logits is not None:
                anchor_side = torch.argmax(anchor_logits, dim=1)
                final_side = torch.argmax(logits, dim=1)

                anchor_counts["long"] += int((anchor_side == 0).sum().item())
                anchor_counts["short"] += int((anchor_side == 1).sum().item())
                anchor_counts["flat"] += int((anchor_side == 2).sum().item())
                final_counts["long"] += int((final_side == 0).sum().item())
                final_counts["short"] += int((final_side == 1).sum().item())
                final_counts["flat"] += int((final_side == 2).sum().item())

                flip_counts["anchor_short_to_short"] += int(((anchor_side == 1) & (final_side == 1)).sum().item())
                flip_counts["anchor_short_to_long"] += int(((anchor_side == 1) & (final_side == 0)).sum().item())
                flip_counts["anchor_long_to_short"] += int(((anchor_side == 0) & (final_side == 1)).sum().item())
                flip_counts["anchor_long_to_long"] += int(((anchor_side == 0) & (final_side == 0)).sum().item())
                flip_counts["anchor_flat_to_long"] += int(((anchor_side == 2) & (final_side == 0)).sum().item())
                flip_counts["anchor_flat_to_short"] += int(((anchor_side == 2) & (final_side == 1)).sum().item())

                sessions = ctx_cat[:, 0].cpu().numpy()
                for sess_id, sess_name in session_map.items():
                    sess_mask = sessions == sess_id
                    if not np.any(sess_mask):
                        continue
                    sess_anchor = anchor_side[sess_mask]
                    sess_final = final_side[sess_mask]
                    anchor_counts_by_session[sess_name]["long"] += int((sess_anchor == 0).sum().item())
                    anchor_counts_by_session[sess_name]["short"] += int((sess_anchor == 1).sum().item())
                    anchor_counts_by_session[sess_name]["flat"] += int((sess_anchor == 2).sum().item())
                    final_counts_by_session[sess_name]["long"] += int((sess_final == 0).sum().item())
                    final_counts_by_session[sess_name]["short"] += int((sess_final == 1).sum().item())
                    final_counts_by_session[sess_name]["flat"] += int((sess_final == 2).sum().item())
                    flip_counts_by_session[sess_name]["anchor_short_to_short"] += int(
                        ((sess_anchor == 1) & (sess_final == 1)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_short_to_long"] += int(
                        ((sess_anchor == 1) & (sess_final == 0)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_long_to_short"] += int(
                        ((sess_anchor == 0) & (sess_final == 1)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_long_to_long"] += int(
                        ((sess_anchor == 0) & (sess_final == 0)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_flat_to_long"] += int(
                        ((sess_anchor == 2) & (sess_final == 0)).sum().item()
                    )
                    flip_counts_by_session[sess_name]["anchor_flat_to_short"] += int(
                        ((sess_anchor == 2) & (sess_final == 1)).sum().item()
                    )

                if delta_logits is not None:
                    residual_scale = float(getattr(model, "residual_scale", 1.0))
                    residual_gap = residual_scale * (delta_logits[:, 0] - delta_logits[:, 1])
                    anchor_gap = anchor_logits[:, 1] - anchor_logits[:, 0]
                    ratio = torch.abs(residual_gap) / (torch.abs(anchor_gap) + ratio_eps)
                    residual_gap_chunks.append(residual_gap.detach().cpu().numpy())
                    anchor_gap_chunks.append(anchor_gap.detach().cpu().numpy())
                    ratio_chunks.append(ratio.detach().cpu().numpy())

    preds_np = np.asarray(preds)
    targets_np = np.asarray(targets)

    acc = float(accuracy_score(targets_np.astype(int), preds_np.astype(int)))

    def _stat_summary(values: np.ndarray) -> Dict[str, Optional[float]]:
        if values.size == 0:
            return {"mean": None, "median": None, "p90": None, "p95": None, "max": None}
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }

    total_anchor = sum(anchor_counts.values())
    total_final = sum(final_counts.values())
    if total_anchor > 0:
        if total_final > 0:
            log.info(
                "[ENTRY_PRED_DIST_PROOF] pred_long_pct=%.6f pred_short_pct=%.6f pred_flat_pct=%.6f",
                final_counts["long"] / total_final,
                final_counts["short"] / total_final,
                final_counts["flat"] / total_final,
            )
            for sess_name in session_map.values():
                sess_total = sum(final_counts_by_session[sess_name].values())
                if sess_total > 0:
                    log.info(
                        "[ENTRY_PRED_DIST_PROOF] session=%s pred_long_pct=%.6f pred_short_pct=%.6f pred_flat_pct=%.6f",
                        sess_name,
                        final_counts_by_session[sess_name]["long"] / sess_total,
                        final_counts_by_session[sess_name]["short"] / sess_total,
                        final_counts_by_session[sess_name]["flat"] / sess_total,
                    )
        log.info(
            "[ENTRY_ANCHOR_DIST_PROOF] anchor_long_pct=%.6f anchor_short_pct=%.6f anchor_flat_pct=%.6f",
            anchor_counts["long"] / total_anchor,
            anchor_counts["short"] / total_anchor,
            anchor_counts["flat"] / total_anchor,
        )
        for sess_name in session_map.values():
            sess_total = sum(anchor_counts_by_session[sess_name].values())
            if sess_total > 0:
                log.info(
                    "[ENTRY_ANCHOR_DIST_PROOF] session=%s anchor_long_pct=%.6f anchor_short_pct=%.6f anchor_flat_pct=%.6f",
                    sess_name,
                    anchor_counts_by_session[sess_name]["long"] / sess_total,
                    anchor_counts_by_session[sess_name]["short"] / sess_total,
                    anchor_counts_by_session[sess_name]["flat"] / sess_total,
                )

        log.info(
            "[ENTRY_ANCHOR_FLIP_PROOF] anchor_short_to_short=%d anchor_short_to_long=%d anchor_long_to_short=%d "
            "anchor_long_to_long=%d anchor_flat_to_long=%d anchor_flat_to_short=%d",
            flip_counts["anchor_short_to_short"],
            flip_counts["anchor_short_to_long"],
            flip_counts["anchor_long_to_short"],
            flip_counts["anchor_long_to_long"],
            flip_counts["anchor_flat_to_long"],
            flip_counts["anchor_flat_to_short"],
        )
        for sess_name in session_map.values():
            sess_counts = flip_counts_by_session[sess_name]
            if sum(sess_counts.values()) > 0:
                log.info(
                    "[ENTRY_ANCHOR_FLIP_PROOF] session=%s anchor_short_to_short=%d anchor_short_to_long=%d "
                    "anchor_long_to_short=%d anchor_long_to_long=%d anchor_flat_to_long=%d anchor_flat_to_short=%d",
                    sess_name,
                    sess_counts["anchor_short_to_short"],
                    sess_counts["anchor_short_to_long"],
                    sess_counts["anchor_long_to_short"],
                    sess_counts["anchor_long_to_long"],
                    sess_counts["anchor_flat_to_long"],
                    sess_counts["anchor_flat_to_short"],
                )

    if residual_gap_chunks and anchor_gap_chunks and ratio_chunks:
        residual_gap_all = np.concatenate(residual_gap_chunks, axis=0)
        anchor_gap_all = np.concatenate(anchor_gap_chunks, axis=0)
        ratio_all = np.concatenate(ratio_chunks, axis=0)

        res_stats = _stat_summary(residual_gap_all)
        anc_stats = _stat_summary(anchor_gap_all)
        ratio_stats = _stat_summary(ratio_all)

        log.info(
            "[ENTRY_RESIDUAL_GAP_PROOF] mean_gap=%.6f median_gap=%.6f p90_gap=%.6f p95_gap=%.6f max_gap=%.6f",
            res_stats["mean"] or 0.0,
            res_stats["median"] or 0.0,
            res_stats["p90"] or 0.0,
            res_stats["p95"] or 0.0,
            res_stats["max"] or 0.0,
        )
        log.info(
            "[ENTRY_ANCHOR_GAP_PROOF] mean_anchor_gap=%.6f median_anchor_gap=%.6f p90_anchor_gap=%.6f",
            anc_stats["mean"] or 0.0,
            anc_stats["median"] or 0.0,
            anc_stats["p90"] or 0.0,
        )
        log.info(
            "[ENTRY_RESIDUAL_VS_ANCHOR_PROOF] mean_ratio=%.6f p90_ratio=%.6f p95_ratio=%.6f",
            ratio_stats["mean"] or 0.0,
            ratio_stats["p90"] or 0.0,
            ratio_stats["p95"] or 0.0,
        )

    # AUC disabled for this 3-class path
    return total / max(1, n), float("nan"), acc

# -----------------------------------------------------------------------------
# Sanity check
# -----------------------------------------------------------------------------
def run_sanity_check(
    seq_len: int,
    seed: int,
    device: torch.device,
    out_bundle_dir: Path,
    dataset_manifest: Optional[Path] = None,
    deterministic: bool = True,
    enable_specialist_fusion: bool = False,
    specialist_audit_json: Optional[Path] = None,
    specialist_contract_mode: str = "foundation_seq146",
    specialist_num_layers: int = 1,
    specialist_fusion_scale: float = 0.25,
) -> None:
    """
    Contract + dummy forward + write minimal bundle + reload with runtime loader (strict).
    Fail-fast with clear error labels.
    """
    _guard_no_rl()

    signal_contract = _signal_contract_from_manifest_path(dataset_manifest)
    seq_input_dim = int(signal_contract["seq_input_dim"])
    snap_input_dim = int(signal_contract["snap_input_dim"])

    if dataset_manifest is not None:
        p = Path(dataset_manifest).expanduser().resolve()
        _require(p.exists(), f"[SANITY_MANIFEST_MISSING] {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        fc = data.get("feature_contract") or {}
        fc_ctx_cont_dim = int(fc.get("ctx_cont_dim") or -1)
        fc_ctx_cat_dim = int(fc.get("ctx_cat_dim") or -1)
        # R4: validate by DIMS (the real contract), not a brittle literal-tag string.
        # ctx_cat is contract-driven (5/6); the ctx_tag is informational/self-describing.
        _exp_cat_dim = _expected_ctx_cat_dim()
        _require(
            fc_ctx_cont_dim >= 6
            and fc_ctx_cat_dim == _exp_cat_dim,
            f"[SANITY_MANIFEST_CONTRACT] manifest feature_contract must be ctx_cont_dim>=6 ctx_cat_dim={_exp_cat_dim}, got {fc}",
        )
        if fc.get("ctx_cont_base_dim") is not None:
            _require(
                int(fc.get("ctx_cont_base_dim")) == 6,
                f"[SANITY_MANIFEST_CTX_BASE] expected ctx_cont_base_dim=6, got {fc.get('ctx_cont_base_dim')}",
            )
        bridge_id = str(fc.get("signal_bridge_id") or "")
        _require(
            bridge_id.startswith("XGB_SIGNAL_BRIDGE_"),
            f"[SANITY_MANIFEST_SIGNAL] expected XGB_SIGNAL_BRIDGE_*, got {bridge_id}",
        )
        log.info(f"[SANITY] manifest contract OK: {p}")

    ctx = get_canonical_ctx_contract()
    expected_ctx_cat_dim = _expected_ctx_cat_dim()
    if not _is_vnext():
        _require(
            int(ctx.get("ctx_cont_dim") or 0) >= 6 and expected_ctx_cat_dim > 0,
            f"[SANITY_CTX_DIM_MISMATCH] expected ctx_cont_base>=6 ctx_cat_dim={expected_ctx_cat_dim}",
        )
    _require(seq_input_dim == snap_input_dim and seq_input_dim > 0, f"[SANITY_SIGNAL_DIM] seq={seq_input_dim} snap={snap_input_dim}")

    if dataset_manifest is not None:
        ctx_cont_dim = int(fc_ctx_cont_dim)
        ctx_cat_dim = int(fc_ctx_cat_dim)
    else:
        ctx_cont_dim = int(ctx.get("ctx_cont_dim") or 6)
        ctx_cat_dim = expected_ctx_cat_dim
    if _is_vnext():
        ctx_cont_dim = max(ctx_cont_dim, 21)

    log.info(
        f"[SANITY] seed={seed} device={device} "
        f"signal_dim={seq_input_dim} ctx_cont={ctx_cont_dim} ctx_cat={ctx_cat_dim} seq_len={seq_len}"
    )
    specialist_indices: Dict[str, list[int]] | None = None
    specialist_meta: Dict[str, Any] | None = None
    if enable_specialist_fusion:
        specialist_indices, specialist_meta = _load_specialist_fusion_contract(
            specialist_audit_json,
            expected_signal_dim=seq_input_dim,
            contract_mode=specialist_contract_mode,
        )
        log.info("[SPECIALIST_FUSION] sanity enabled groups=%s", sorted(specialist_indices))

    _set_deterministic(seed, device, deterministic=deterministic)

    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        residual_scale=float(ENTRY_RESIDUAL_SCALE),
        anchor_eps=float(ENTRY_ANCHOR_EPS),
        hierarchical_composition_residual_logit_cap=float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
        hierarchical_composition_residual_side_neutral=bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
        enable_specialist_fusion=bool(enable_specialist_fusion),
        specialist_input_indices=specialist_indices,
        specialist_num_layers=int(specialist_num_layers),
        specialist_fusion_scale=float(specialist_fusion_scale),
    ).to(device)
    try:
        head_out = int(getattr(model.head_direction, "out_features", -1))
    except Exception:
        head_out = -1
    log.info("[ENTRY_3CLASS_PROOF] head_direction_out=%s", head_out)
    try:
        head_out = int(getattr(model.head_direction, "out_features", -1))
    except Exception:
        head_out = -1
    log.info("[ENTRY_3CLASS_PROOF] head_direction_out=%s", head_out)

    # Dummy batch: per-sample ctx (B, ctx_*) as in dataset/trening
    B, T = 4, seq_len
    dummy_seq = torch.randn(B, T, seq_input_dim, device=device, dtype=torch.float32)
    dummy_snap = torch.randn(B, snap_input_dim, device=device, dtype=torch.float32)
    dummy_ctx_cont = torch.randn(B, ctx_cont_dim, device=device, dtype=torch.float32)
    dummy_ctx_cat = torch.randint(0, 256, (B, ctx_cat_dim), device=device, dtype=torch.int64)

    with torch.no_grad():
        out = model(
            dummy_seq,
            dummy_snap,
            ctx_cat=dummy_ctx_cat,
            ctx_cont=dummy_ctx_cont,
        )

    direction_logits = out["direction_logits"]
    _require(
        direction_logits.dim() == 2 and direction_logits.shape[1] == 3,
        f"[SANITY_OUTPUT_SHAPE] expected (B,3) got {tuple(direction_logits.shape)}",
    )
    _require(
        direction_logits.dtype == torch.float32,
        f"[SANITY_OUTPUT_DTYPE] expected float32 got {direction_logits.dtype}",
    )
    if torch.isnan(direction_logits).any() or torch.isinf(direction_logits).any():
        raise RuntimeError("[SANITY_NAN_INF] direction_logits contains NaN/Inf")

    log.info(
        f"[SANITY] forward OK shapes seq={dummy_seq.shape} snap={dummy_snap.shape} "
        f"ctx_cont=(B,{ctx_cont_dim}) ctx_cat=(B,{ctx_cat_dim}) out={direction_logits.shape}"
    )

    # Write minimal sanity bundle
    out_bundle_dir = Path(out_bundle_dir).expanduser().resolve()
    out_bundle_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_bundle_dir / "model_state_dict.pt"
    # Unwrap torch.compile wrapper to avoid `_orig_mod.` prefix in saved keys.
    _save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dict = {k: v.cpu().clone() for k, v in _save_model.state_dict().items()}
    torch.save(state_dict, state_path)

    ordered_ctx_cont_names = _build_ordered_ctx_cont_names(ctx_cont_dim, list(ctx.get("ctx_cont_names") or []))

    lock = {
        "version": "entry_v10_ctx_lock_v1",
        "created_at_utc": _utc_now(),
        "signal_bridge_contract_sha256": SIGNAL_BRIDGE_CONTRACT_SHA256,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "seq_len": seq_len,
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": _sha256_file(state_path),
    }
    (out_bundle_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(
        json.dumps(lock, indent=2)
    )

    feature_meta_path = out_bundle_dir / "feature_meta.json"
    feature_meta_path.write_text(json.dumps({"sanity": True, "placeholder": True}))

    meta = {
        "created_at_utc": _utc_now(),
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "seq_len": seq_len,
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "supports_context_features": True,
        "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
        "ctx_tag": f"CTX6CAT{ctx_cat_dim}",
        "ordered_ctx_cont_names": ordered_ctx_cont_names,
        "ordered_ctx_cat_names": list(ORDERED_CTX_CAT_NAMES_V3),
        "feature_meta_path": str(feature_meta_path.name),
        "sanity_bundle": True,
    }
    if specialist_meta:
        meta["specialist_fusion"] = {
            **specialist_meta,
            "num_layers": int(specialist_num_layers),
            "fusion_scale": float(specialist_fusion_scale),
        }
    (out_bundle_dir / "bundle_metadata.json").write_text(json.dumps(meta, indent=2))

    # Reload with runtime loader (strict=True in loader)
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle

    bundle = load_entry_v10_ctx_bundle(
        bundle_dir=out_bundle_dir,
        feature_meta_path=feature_meta_path,
        device="cpu",
        xgb_models=None,
    )
    with torch.no_grad():
        _ = bundle.transformer_model(
            dummy_seq.cpu(),
            dummy_snap.cpu(),
            ctx_cat=dummy_ctx_cat.cpu(),
            ctx_cont=dummy_ctx_cont.cpu(),
        )
    log.info("[SANITY] strict load + forward OK")

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
def run_train(
    train_parquet: Path,
    val_parquet: Path,
    seq_len: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    out_bundle_dir: Path,
    gx1_data_override: str,
    allow_constant_labels: bool,
    num_workers: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    deterministic: bool = True,
    # V12.2 multi-TF
    enable_multi_tf: bool = False,
    m5_prebuilt_path: Optional[Path] = None,
    multi_tf_seq_len: int = 96,
    multi_tf_scale: float = 0.5,
    subsample_rows: int = 0,
    # V10 v3+ aux heads (Targets 1-4)
    enable_tf_agreement_head: bool = False,
    enable_path_quality_variance_head: bool = False,
    enable_position_size_head: bool = False,
    enable_hold_horizon_head: bool = False,
    # Positional encoding (temporal order of every sequence)
    enable_pos_enc: bool = False,
    enable_mtf_direction_head: bool = False,
    mtf_dir_scale_init: float = 0.2,
    enable_regime_film: bool = False,   # BIG-9: FiLM regime-conditioning (default OFF = bit-parity)
    enable_dip_head: bool = False,
    enable_forecast_head: bool = False,
    enable_cross_tf_attn: bool = False,
    enable_timing_head: bool = False,
    enable_tail_risk_head: bool = False,
    enable_vol_forecast_head: bool = False,
    enable_anchor_gate: bool = False,
    anchor_gate_init: float = 1.0,
    enable_hierarchical_entry_heads: bool = False,
    enable_hierarchical_direction_composition: bool = False,
    enable_side_validity_head: bool = False,
    enable_trendline_rail_head: bool = False,
    enable_specialist_fusion: bool = False,
    specialist_audit_json: Optional[Path] = None,
    specialist_contract_mode: str = "foundation_seq146",
    specialist_num_layers: int = 1,
    specialist_fusion_scale: float = 0.25,
    # V2 fast-train extras
    per_tf_seq_len_h4: int = 0,
    per_tf_seq_len_d1: int = 0,
    smoke_date_from: str = "",
    smoke_date_to: str = "",
    grad_accum_steps: int = 0,
    init_from_state_dict: Optional[Path] = None,
    # 2026-06-02: per-TF learnable input scaling (V10 v5+)
    enable_tf_input_scale: bool = False,
    tf_input_scale_init_m5: float = 1.0,
    tf_input_scale_init_m15: float = 1.0,
    tf_input_scale_init_h1: float = 0.7,
    tf_input_scale_init_h4: float = 0.5,
    tf_input_scale_init_d1: float = 0.3,
    vedtak_id: str = "",
) -> None:
    _guard_no_rl()

    # Multi-TF×5 is MANDATORY — fail closed if a caller ever passes it off or
    # forgets the M5 prebuilt. There is no single-TF V10 (toggle removed 2026-05-26).
    if not enable_multi_tf or m5_prebuilt_path is None:
        raise RuntimeError(
            "[MULTI_TF_MANDATORY] V10 must train multi-TF×5 (M5/M15/H1/H4/D1). "
            f"Got enable_multi_tf={enable_multi_tf} m5_prebuilt_path={m5_prebuilt_path}."
        )

    ctx = get_canonical_ctx_contract()
    expected_ctx_cat_dim = _expected_ctx_cat_dim()
    _require(expected_ctx_cat_dim > 0 and int(ctx.get("ctx_cont_dim") or 0) >= 6, "[CTX_CONTRACT_DIM]")

    log.info(
        f"[TRAIN] seed={seed} device={device} batch_size={batch_size} epochs={epochs} lr={lr} "
        f"signal_dim=dynamic ctx_cont=dynamic ctx_cat={expected_ctx_cat_dim} early_stop_patience={early_stopping_patience} "
        f"early_stop_min_delta={early_stopping_min_delta}"
    )

    _set_deterministic(seed, device, deterministic=deterministic)

    _log_label_distribution(train_parquet, split="train")
    _log_label_distribution(val_parquet, split="val")

    # V12.2: pre-build multi-TF features BEFORE loading train_parquet so peak
    # memory = max(train_parquet, M5_prebuilt) instead of their sum. Without
    # this, OOM on 15GB hosts during Dataset construction (1.5GB parquet ×
    # pandas overhead + 1.5GB M5 prebuilt × pandas overhead > 15GB).
    if enable_multi_tf and m5_prebuilt_path is not None:
        v2_mode = True  # MANDATORY V2 5×25 (env-gate removed 2026-05-26)
        cache_key = f"{Path(m5_prebuilt_path).resolve()}|v2={v2_mode}"
        if cache_key not in _MULTI_TF_CACHE:
            from gx1.features.htf_features import (
                build_multi_tf_per_bar_features,
                build_multi_tf_per_bar_features_v2,
            )
            log.info(f"[MULTI_TF] pre-building features (peak-mem-fix): {Path(m5_prebuilt_path).name} v2={v2_mode}")
            load_cols = ["time", "open", "high", "low", "close"]
            if v2_mode:
                import pyarrow.parquet as pq
                if "volume" in pq.ParquetFile(m5_prebuilt_path).schema_arrow.names:
                    load_cols.append("volume")
            m5 = pd.read_parquet(m5_prebuilt_path, columns=load_cols)
            m5["time"] = pd.to_datetime(m5["time"], utc=True)
            m5 = m5.set_index("time").sort_index()
            for c in ("open", "high", "low", "close"):
                m5[c] = m5[c].astype(np.float32)
            if "volume" in m5.columns:
                m5["volume"] = m5["volume"].astype(np.float32)
            if v2_mode:
                feats = build_multi_tf_per_bar_features_v2(m5)
            else:
                feats = build_multi_tf_per_bar_features(m5)
                feats = {k: v for k, v in feats.items() if k != "M5"}
            del m5
            import gc; gc.collect()
            _MULTI_TF_CACHE[cache_key] = feats
            for tf_name, df in feats.items():
                log.info(f"[MULTI_TF] {tf_name}: {len(df):,} bars × {df.shape[1]} feats")

    # V2 fast-train: build per-TF seq_lens dict and forward smoke-date.
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
    xau_direction_repair_mode = bool(
        enable_anchor_gate
        or enable_hierarchical_entry_heads
        or enable_side_validity_head
        or enable_trendline_rail_head
    )
    train_ds = EntryV10CtxDataset(
        train_parquet,
        seq_len=seq_len,
        allow_constant_labels=allow_constant_labels,
        enable_multi_tf=enable_multi_tf,
        m5_prebuilt_path=m5_prebuilt_path,
        multi_tf_seq_len=multi_tf_seq_len,
        per_tf_seq_lens=_per_tf_lens,
        multi_tf_closed_bar=True if xau_direction_repair_mode else None,
        smoke_date_from=smoke_date_from,
        smoke_date_to=smoke_date_to,
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
        allow_constant_labels=True,
        enable_multi_tf=enable_multi_tf,
        m5_prebuilt_path=m5_prebuilt_path,
        multi_tf_seq_len=multi_tf_seq_len,
        per_tf_seq_lens=_per_tf_lens,
        multi_tf_closed_bar=True if xau_direction_repair_mode else None,
        smoke_date_from=smoke_date_from,
        smoke_date_to=smoke_date_to,
    )
    if xau_direction_repair_mode:
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
            if not bool(getattr(ds_obj, "neutral_xgb_bridge", False)):
                contract_failures.append(
                    f"{split_name} dataset manifest must declare neutral_xgb_bridge=true "
                    "for XAU direction repair heads"
                )
            if str(getattr(ds_obj, "xgb_bridge_source", "") or "") != "neutral_uniform_proba":
                contract_failures.append(
                    f"{split_name} dataset manifest must declare xgb_bridge_source=neutral_uniform_proba "
                    f"(got {getattr(ds_obj, 'xgb_bridge_source', '')!r})"
                )
            contract_failures.extend(_xau_direction_repair_target_failures(split_name, ds_obj.df))
        if contract_failures:
            raise RuntimeError(
                "[ENTRY_XAU_DIRECTION_REPAIR_CONTRACT_INVALID] "
                + "; ".join(contract_failures)
            )
    if enable_hierarchical_entry_heads:
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
    train_bad_path_rate = float(train_ds.df["y_bad_path"].astype(float).mean()) if "y_bad_path" in train_ds.df.columns else 0.0
    val_bad_path_rate = float(val_ds.df["y_bad_path"].astype(float).mean()) if "y_bad_path" in val_ds.df.columns else 0.0
    train_tradable_rate = float(train_ds.df["y_tradable"].astype(float).mean()) if "y_tradable" in train_ds.df.columns else 0.0
    val_tradable_rate = float(val_ds.df["y_tradable"].astype(float).mean()) if "y_tradable" in val_ds.df.columns else 0.0
    train_trade_rate = float(train_ds.df["y_trade"].astype(float).mean()) if "y_trade" in train_ds.df.columns else train_tradable_rate
    val_trade_rate = float(val_ds.df["y_trade"].astype(float).mean()) if "y_trade" in val_ds.df.columns else val_tradable_rate
    if {"y_long_bad_path", "y_short_bad_path"}.issubset(train_ds.df.columns):
        train_long_bad_path_rate = float(train_ds.df["y_long_bad_path"].astype(float).mean())
        train_short_bad_path_rate = float(train_ds.df["y_short_bad_path"].astype(float).mean())
        _train_side_bad_arr = pd.concat(
            [
                train_ds.df["y_long_bad_path"].astype(float),
                train_ds.df["y_short_bad_path"].astype(float),
            ],
            ignore_index=True,
        )
        train_side_bad_path_rate = float(_train_side_bad_arr.mean())
    else:
        train_long_bad_path_rate = train_bad_path_rate
        train_short_bad_path_rate = train_bad_path_rate
        train_side_bad_path_rate = train_bad_path_rate
    if {"y_long_bad_path", "y_short_bad_path"}.issubset(val_ds.df.columns):
        val_long_bad_path_rate = float(val_ds.df["y_long_bad_path"].astype(float).mean())
        val_short_bad_path_rate = float(val_ds.df["y_short_bad_path"].astype(float).mean())
        _val_side_bad_arr = pd.concat(
            [
                val_ds.df["y_long_bad_path"].astype(float),
                val_ds.df["y_short_bad_path"].astype(float),
            ],
            ignore_index=True,
        )
        val_side_bad_path_rate = float(_val_side_bad_arr.mean())
    else:
        val_long_bad_path_rate = val_bad_path_rate
        val_short_bad_path_rate = val_bad_path_rate
        val_side_bad_path_rate = val_bad_path_rate
    train_hard_neg_long_rate = float(train_ds.df["y_hard_negative_long"].astype(float).mean()) if "y_hard_negative_long" in train_ds.df.columns else 0.0
    val_hard_neg_long_rate = float(val_ds.df["y_hard_negative_long"].astype(float).mean()) if "y_hard_negative_long" in val_ds.df.columns else 0.0
    train_dead_neg_long_rate = float(train_ds.df["y_dead_negative_long"].astype(float).mean()) if "y_dead_negative_long" in train_ds.df.columns else 0.0
    val_dead_neg_long_rate = float(val_ds.df["y_dead_negative_long"].astype(float).mean()) if "y_dead_negative_long" in val_ds.df.columns else 0.0
    train_teaser_neg_long_rate = float(train_ds.df["y_teaser_negative_long"].astype(float).mean()) if "y_teaser_negative_long" in train_ds.df.columns else 0.0
    val_teaser_neg_long_rate = float(val_ds.df["y_teaser_negative_long"].astype(float).mean()) if "y_teaser_negative_long" in val_ds.df.columns else 0.0
    train_clean_edge_rate = float(train_ds.df["y_clean_edge_long"].astype(float).mean()) if "y_clean_edge_long" in train_ds.df.columns else 0.0
    val_clean_edge_rate = float(val_ds.df["y_clean_edge_long"].astype(float).mean()) if "y_clean_edge_long" in val_ds.df.columns else 0.0
    train_survival_rate = float(train_ds.df["y_survival_long"].astype(float).mean()) if "y_survival_long" in train_ds.df.columns else 0.0
    val_survival_rate = float(val_ds.df["y_survival_long"].astype(float).mean()) if "y_survival_long" in val_ds.df.columns else 0.0
    train_teacher_bad_rate = float(train_ds.df["y_teacher_bad_long"].astype(float).mean()) if "y_teacher_bad_long" in train_ds.df.columns else 0.0
    val_teacher_bad_rate = float(val_ds.df["y_teacher_bad_long"].astype(float).mean()) if "y_teacher_bad_long" in val_ds.df.columns else 0.0
    train_teacher_winner_rate = float(train_ds.df["y_teacher_winner_long"].astype(float).mean()) if "y_teacher_winner_long" in train_ds.df.columns else 0.0
    val_teacher_winner_rate = float(val_ds.df["y_teacher_winner_long"].astype(float).mean()) if "y_teacher_winner_long" in val_ds.df.columns else 0.0
    train_selector_long_mask_rate = float(train_ds.df["y_selector_long_mask"].astype(float).mean()) if "y_selector_long_mask" in train_ds.df.columns else 0.0
    val_selector_long_mask_rate = float(val_ds.df["y_selector_long_mask"].astype(float).mean()) if "y_selector_long_mask" in val_ds.df.columns else 0.0
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
    raw_clean_edge_pos_weight = ((1.0 - train_clean_edge_rate) / max(train_clean_edge_rate, 1e-9)) if train_clean_edge_rate > 0.0 else 1.0
    clean_edge_pos_weight = float(
        min(float(ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP), max(1.0, raw_clean_edge_pos_weight))
    )
    raw_survival_pos_weight = ((1.0 - train_survival_rate) / max(train_survival_rate, 1e-9)) if train_survival_rate > 0.0 else 1.0
    survival_pos_weight = float(
        min(float(ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP), max(1.0, raw_survival_pos_weight))
    )
    # 2026-05-26: SYMMETRIC + sqrt-softened directional class weights. The old
    # per-side inverse-frequency weights made the model over-predict SHORT (short
    # rarer → higher weight: 4.67 vs long 4.22) AND over-predict direction vs flat
    # (4.5 vs 1.0). With the re-balanced labels (~18/18/63) that aggressive scheme
    # over-corrects. Fix: ONE shared directional weight from the COMBINED directional
    # rate (removes long/short asymmetry), sqrt-softened (shrinks the directional-vs-
    # flat gap). Env caps still clamp. flat stays 1.0.
    _dir_rate = 0.5 * (float(train_long_rate) + float(train_short_rate))
    _raw_dir = ((1.0 - _dir_rate) / max(_dir_rate, 1e-9)) if _dir_rate > 0.0 else 1.0
    _dir_w = float(np.sqrt(max(_raw_dir, 1.0)))  # sqrt-soften; >=1
    raw_long_class_weight = _raw_dir   # kept for the proof log
    raw_short_class_weight = _raw_dir
    long_class_weight = float(min(float(ENTRY_LONG_CLASS_WEIGHT_CAP), max(1.0, _dir_w)))
    if train_short_rate > 0.0:
        short_class_weight = float(min(float(ENTRY_SHORT_CLASS_WEIGHT_CAP), max(1.0, _dir_w)))
    else:
        short_class_weight = 0.0
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
        "[ENTRY_CLEAN_EDGE_RATE_PROOF] train_rate=%.6f val_rate=%.6f raw_pos_weight=%.6f capped_pos_weight=%.6f cap=%.3f",
        train_clean_edge_rate,
        val_clean_edge_rate,
        raw_clean_edge_pos_weight,
        clean_edge_pos_weight,
        float(ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP),
    )
    log.info(
        "[ENTRY_SURVIVAL_RATE_PROOF] train_rate=%.6f val_rate=%.6f raw_pos_weight=%.6f capped_pos_weight=%.6f cap=%.3f",
        train_survival_rate,
        val_survival_rate,
        raw_survival_pos_weight,
        survival_pos_weight,
        float(ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP),
    )
    log.info(
        "[ENTRY_TEACHER_RATE_PROOF] train_bad=%.6f val_bad=%.6f train_winner=%.6f val_winner=%.6f selector_mask_train=%.6f selector_mask_val=%.6f",
        train_teacher_bad_rate,
        val_teacher_bad_rate,
        train_teacher_winner_rate,
        val_teacher_winner_rate,
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
            "batch_size=%d min_rows=%d min_label_rate=%.3f num_samples=%d",
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

    # Before first epoch: log sample shapes and confirm contract signal=7, ctx_cat=6, ctx_cont>=6
    sample = next(iter(train_loader))
    seq_input_dim = int(sample["seq_x"].shape[2])
    snap_input_dim = int(sample["snap_x"].shape[1])
    _require(seq_input_dim == snap_input_dim and seq_input_dim > 0, f"[SIGNAL_DIM_INVALID] seq={seq_input_dim} snap={snap_input_dim}")
    ctx_cont_dim = int(sample["ctx_cont"].shape[1])
    ctx_cat_dim = int(sample["ctx_cat"].shape[1])
    base_ctx_cont_names = list(ctx.get("ctx_cont_names") or [])
    ordered_ctx_cont_names = _build_ordered_ctx_cont_names(ctx_cont_dim, base_ctx_cont_names)
    # M3 fix (2026-06-06): use the REGIME-AWARE 5-name ctx_cat (signal_bridge_v3), NOT the stale
    # signal_bridge_v1 6-name list (which keeps the dropped trend_regime_id) — so the bundle metadata
    # matches the trained ctx_cat_dim (5 under GX1_REGIME_V4) + the strict v3 contract validator.
    ordered_ctx_cat_names = list(ORDERED_CTX_CAT_NAMES_V3)
    if len(ordered_ctx_cat_names) != ctx_cat_dim:
        raise RuntimeError(
            f"[CTX_CAT_NAME_DIM_MISMATCH] ordered_ctx_cat_names={len(ordered_ctx_cat_names)} "
            f"!= trained ctx_cat_dim={ctx_cat_dim} (REGIME_V4={os.environ.get('GX1_REGIME_V4','1')})"
        )
    log.info(
        f"[TRAIN_CONTRACT] seq_x={sample['seq_x'].shape} snap_x={sample['snap_x'].shape} "
        f"ctx_cont={sample['ctx_cont'].shape} ctx_cat={sample['ctx_cat'].shape}"
    )
    log.info(
        "[ENTRY_INPUT_SCHEMA_PROOF] signal_dim=%d ctx_cont_dim=%d ctx_cat_dim=%d neutral_xgb_bridge=%s",
        seq_input_dim,
        ctx_cont_dim,
        ctx_cat_dim,
        bool(getattr(train_ds, "neutral_xgb_bridge", False)),
    )
    expected_ctx_cont_dim = _expected_ctx_cont_dim()
    _require(
        ctx_cont_dim == expected_ctx_cont_dim,
        f"[ENTRY_CTX_CONT_DIM_MISMATCH] expected ctx_cont_dim={expected_ctx_cont_dim} got={ctx_cont_dim}",
    )
    _exp_ctx_cat_dim = _expected_ctx_cat_dim()
    _require(ctx_cat_dim == _exp_ctx_cat_dim, f"[ENTRY_CTX_CAT_DIM_MISMATCH] expected ctx_cat_dim={_exp_ctx_cat_dim} got={ctx_cat_dim}")
    if ctx_cont_dim > 6:
        log.info(
            "[ENTRY_MICRO_FEATURES_PROOF] names=%s count=%d",
            list(MICRO_FEATURE_NAMES),
            len(MICRO_FEATURE_NAMES),
        )
        log.info(
            "[ENTRY_SWING_FEATURES_PROOF] names=%s count=%d",
            list(SWING_FEATURE_NAMES),
            len(SWING_FEATURE_NAMES),
        )
    _require(
        sample["seq_x"].shape[2] == seq_input_dim
        and sample["snap_x"].shape[1] == snap_input_dim
        and sample["ctx_cont"].shape[1] == ctx_cont_dim
        and sample["ctx_cat"].shape[1] == ctx_cat_dim,
        f"[TRAIN_CONTRACT_MISMATCH] expected signal={seq_input_dim} ctx_cont={ctx_cont_dim} ctx_cat={ctx_cat_dim}",
    )

    # V12.2: detect multi-TF feature count from dataset (avoid hardcoding 19)
    _mtf_feat_count = train_ds._multi_tf_feature_count if enable_multi_tf else 0
    # V2 mode: dataset adds M5 branch (5 TFs total) + 25-feat per TF
    _mtf_v2 = bool(getattr(train_ds, "_multi_tf_v2", False)) if enable_multi_tf else False
    # Per-TF seq_len overrides (default 0 → fall back to global multi_tf_seq_len).
    _h4_len = int(per_tf_seq_len_h4) if int(per_tf_seq_len_h4) > 0 else int(multi_tf_seq_len)
    _d1_len = int(per_tf_seq_len_d1) if int(per_tf_seq_len_d1) > 0 else (252 if _tapered else int(multi_tf_seq_len))
    _m15_len = 64 if _tapered else int(multi_tf_seq_len)  # B10 tapered-MTF — mirrors _per_tf_lens above (train==serve via bundle meta)
    if _h4_len != multi_tf_seq_len or _d1_len != multi_tf_seq_len or _m15_len != multi_tf_seq_len:
        log.info("[PER_TF_SEQ_LEN] M15=%d H4=%d D1=%d (global=%d)", _m15_len, _h4_len, _d1_len, int(multi_tf_seq_len))
    specialist_indices: Dict[str, list[int]] | None = None
    specialist_meta: Dict[str, Any] | None = None
    if enable_specialist_fusion:
        specialist_indices, specialist_meta = _load_specialist_fusion_contract(
            specialist_audit_json,
            expected_signal_dim=seq_input_dim,
            contract_mode=specialist_contract_mode,
        )
        log.info("[SPECIALIST_FUSION] train enabled groups=%s", sorted(specialist_indices))
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        residual_scale=float(ENTRY_RESIDUAL_SCALE),
        anchor_eps=float(ENTRY_ANCHOR_EPS),
        enable_multi_tf=enable_multi_tf,
        m15_seq_dim=_mtf_feat_count,
        h1_seq_dim=_mtf_feat_count,
        h4_seq_dim=_mtf_feat_count,
        d1_seq_dim=_mtf_feat_count,
        m15_seq_len=_m15_len,
        h1_seq_len=multi_tf_seq_len,
        h4_seq_len=_h4_len,
        d1_seq_len=_d1_len,
        # V2 only: enable M5 branch (V1 leaves enable_multi_tf_m5=False)
        enable_multi_tf_m5=_mtf_v2,
        m5_seq_dim=_mtf_feat_count if _mtf_v2 else 0,
        m5_seq_len=multi_tf_seq_len,
        multi_tf_scale=multi_tf_scale,
        # V10 v3+ aux heads (Targets 1-4)
        enable_tf_agreement_head=enable_tf_agreement_head,
        enable_path_quality_variance_head=enable_path_quality_variance_head,
        enable_position_size_head=enable_position_size_head,
        enable_hold_horizon_head=enable_hold_horizon_head,
        enable_pos_enc=enable_pos_enc,
        enable_regime_film=enable_regime_film,
        enable_dip_head=enable_dip_head,
        enable_forecast_head=enable_forecast_head,
        enable_cross_tf_attn=enable_cross_tf_attn,
        enable_timing_head=enable_timing_head,
        enable_tail_risk_head=enable_tail_risk_head,
        enable_vol_forecast_head=enable_vol_forecast_head,
        enable_anchor_gate=bool(enable_anchor_gate),
        anchor_gate_init=float(anchor_gate_init),
        enable_hierarchical_entry_heads=bool(enable_hierarchical_entry_heads),
        enable_hierarchical_direction_composition=bool(enable_hierarchical_direction_composition),
        hierarchical_composition_residual_logit_cap=float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
        hierarchical_composition_residual_side_neutral=bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
        hierarchical_composition_public_flat_from_trade=bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE),
        hierarchical_public_direction_composition=str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
        enable_hierarchical_public_trade_head=bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
        enable_hierarchical_public_side_head=bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
        enable_hierarchical_ctx_prior_adapter=bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
        hierarchical_ctx_prior_adapter_scale=float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
        enable_hierarchical_ctx_direction_calibration=bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
        hierarchical_ctx_direction_calibration_scale=float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE),
        hierarchical_ctx_direction_calibration_cap=float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
        enable_side_validity_head=bool(enable_side_validity_head),
        enable_trendline_rail_head=bool(enable_trendline_rail_head),
        trendline_rail_output_dim=6 if bool(enable_trendline_rail_head) else 4,
        enable_mtf_direction_head=enable_mtf_direction_head,
        mtf_dir_scale_init=mtf_dir_scale_init,
        enable_specialist_fusion=bool(enable_specialist_fusion),
        specialist_input_indices=specialist_indices,
        specialist_num_layers=int(specialist_num_layers),
        specialist_fusion_scale=float(specialist_fusion_scale),
        # 2026-06-02: per-TF learnable input scale (passes through to model arch)
        enable_tf_input_scale=enable_tf_input_scale,
        tf_input_scale_init_m5=tf_input_scale_init_m5,
        tf_input_scale_init_m15=tf_input_scale_init_m15,
        tf_input_scale_init_h1=tf_input_scale_init_h1,
        tf_input_scale_init_h4=tf_input_scale_init_h4,
        tf_input_scale_init_d1=tf_input_scale_init_d1,
    ).to(device)
    if enable_tf_input_scale:
        log.info(
            "[TF_INPUT_SCALE] learnable per-TF inits: M5=%.2f M15=%.2f H1=%.2f H4=%.2f D1=%.2f",
            tf_input_scale_init_m5, tf_input_scale_init_m15,
            tf_input_scale_init_h1, tf_input_scale_init_h4, tf_input_scale_init_d1,
        )
    if enable_tf_agreement_head or enable_path_quality_variance_head or enable_position_size_head or enable_hold_horizon_head:
        log.info(
            "[V10_V3PLUS_HEADS] tf_agreement=%s path_var=%s position_size=%s hold_horizon=%s",
            enable_tf_agreement_head, enable_path_quality_variance_head,
            enable_position_size_head, enable_hold_horizon_head,
        )
    log.info(
        "[ENTRY_DIRECTION_REPAIR_HEADS] anchor_gate=%s anchor_gate_init=%.4f hierarchy=%s side_validity=%s trendline_rail=%s",
        bool(enable_anchor_gate),
        float(anchor_gate_init),
        bool(enable_hierarchical_entry_heads),
        bool(enable_side_validity_head),
        bool(enable_trendline_rail_head),
    )
    if enable_multi_tf:
        _tfs = "M5+M15+H1+H4+D1 (V2)" if _mtf_v2 else "M15+H1+H4+D1 (V1)"
        log.info(
            "[MULTI_TF_PROOF] enabled=True  TFs=%s  per_tf_dim=%d  per_tf_len=%d  total_extra_params≈%dK",
            _tfs, _mtf_feat_count, multi_tf_seq_len,
            (sum(p.numel() for p in model.parameters()) - 691977) // 1000,
        )
    # Warm-start: load a state_dict (likely from warm_start_v10_v2_from_v1.py)
    # BEFORE torch.compile wrap, since compile prefixes keys with _orig_mod.
    if init_from_state_dict is not None:
        _isd_path = Path(init_from_state_dict)
        if not _isd_path.is_file():
            raise FileNotFoundError(f"[INIT_STATE_DICT_MISSING] {_isd_path}")
        log.info(f"[WARM_START] loading state_dict: {_isd_path}")
        _isd = torch.load(_isd_path, map_location="cpu", weights_only=True)
        _isd = {k.removeprefix("_orig_mod."): v for k, v in _isd.items()}
        _model_state = model.state_dict()
        _shape_dropped = [
            k for k, v in _isd.items()
            if k in _model_state and tuple(getattr(v, "shape", ())) != tuple(_model_state[k].shape)
        ]
        if _shape_dropped:
            _isd = {k: v for k, v in _isd.items() if k not in set(_shape_dropped)}
            log.info(
                "[WARM_START] dropped %d shape-mismatched keys for dynamic input dims: %s",
                len(_shape_dropped),
                sorted(_shape_dropped)[:8],
            )
        _ld = model.load_state_dict(_isd, strict=False)
        log.info(
            f"[WARM_START] loaded {len(_isd)} keys. missing={len(_ld.missing_keys)} unexpected={len(_ld.unexpected_keys)}"
        )
        if _ld.missing_keys:
            log.info(f"[WARM_START] missing (model has, state lacks): {sorted(_ld.missing_keys)[:5]}...")
        if _ld.unexpected_keys:
            log.info(f"[WARM_START] unexpected (state has, model lacks): {sorted(_ld.unexpected_keys)[:5]}...")
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
        "[ENTRY_ANCHORED_PROOF] enabled=1 residual_scale=%.3f anchor_source=signal7_p_long_short_flat anchor_eps=%.8f",
        float(ENTRY_RESIDUAL_SCALE),
        float(ENTRY_ANCHOR_EPS),
    )

    _require_nonneg("ENTRY_COST_SENSITIVE_SCALE", ENTRY_COST_SENSITIVE_SCALE)
    _require_nonneg("ENTRY_COST_LONG_TO_SHORT", ENTRY_COST_LONG_TO_SHORT)
    _require_nonneg("ENTRY_COST_LONG_TO_FLAT", ENTRY_COST_LONG_TO_FLAT)
    _require_nonneg("ENTRY_COST_SHORT_TO_LONG", ENTRY_COST_SHORT_TO_LONG)
    _require_nonneg("ENTRY_COST_SHORT_TO_FLAT", ENTRY_COST_SHORT_TO_FLAT)
    _require_nonneg("ENTRY_COST_FLAT_TO_LONG", ENTRY_COST_FLAT_TO_LONG)
    _require_nonneg("ENTRY_COST_FLAT_TO_SHORT", ENTRY_COST_FLAT_TO_SHORT)
    _require_nonneg("ENTRY_PRED_BALANCE_ALPHA", ENTRY_PRED_BALANCE_ALPHA)
    _require_nonneg("ENTRY_RESIDUAL_SCALE", ENTRY_RESIDUAL_SCALE)
    _require_nonneg("ENTRY_ANCHOR_EPS", ENTRY_ANCHOR_EPS)
    _require_nonneg("ENTRY_RESIDUAL_SIDE_BIAS_ALPHA", ENTRY_RESIDUAL_SIDE_BIAS_ALPHA)
    _require_nonneg("ENTRY_TAIL_DIRECTION_CE_WEIGHT", ENTRY_TAIL_DIRECTION_CE_WEIGHT)
    _require_nonneg("ENTRY_HIER_TRADE_WEIGHT", ENTRY_HIER_TRADE_WEIGHT)
    _require_nonneg("ENTRY_HIER_SIDE_WEIGHT", ENTRY_HIER_SIDE_WEIGHT)
    _require_nonneg("ENTRY_HIER_UTILITY_WEIGHT", ENTRY_HIER_UTILITY_WEIGHT)
    _require_nonneg("ENTRY_HIER_BAD_PATH_WEIGHT", ENTRY_HIER_BAD_PATH_WEIGHT)
    _require_nonneg("ENTRY_HIER_MAE_WEIGHT", ENTRY_HIER_MAE_WEIGHT)
    _require_nonneg("ENTRY_HIER_LEGACY_CE_MULT", ENTRY_HIER_LEGACY_CE_MULT)
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
    _require_nonneg("ENTRY_HIER_POCKET_ABSTAIN_WEIGHT", ENTRY_HIER_POCKET_ABSTAIN_WEIGHT)
    _require_nonneg("ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT", ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS", ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_AUX_WEIGHT", ENTRY_TRENDLINE_RAIL_AUX_WEIGHT)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT", ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT)
    _require_nonneg(
        "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT",
        ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT,
    )
    _require_nonneg(
        "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT",
        ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT,
    )
    _require_nonneg("ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT", ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT", ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT", ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT", ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_MARGIN", ENTRY_TRENDLINE_RAIL_MARGIN)
    _require_nonneg("ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS", ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS)
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
    _require_nonneg("ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION", ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION)
    _require_nonneg("ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP", ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP)
    _require_nonneg("ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL", ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL)
    _require_nonneg("ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE", ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE)
    _require_nonneg("ENTRY_HIER_PUBLIC_TRADE_HEAD", ENTRY_HIER_PUBLIC_TRADE_HEAD)
    _require_nonneg("ENTRY_HIER_PUBLIC_SIDE_HEAD", ENTRY_HIER_PUBLIC_SIDE_HEAD)
    _require_nonneg("ENTRY_HIER_CTX_PRIOR_ADAPTER", ENTRY_HIER_CTX_PRIOR_ADAPTER)
    _require_nonneg("ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE", ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE)
    _require_nonneg("ENTRY_HIER_CTX_DIRECTION_CALIBRATION", ENTRY_HIER_CTX_DIRECTION_CALIBRATION)
    _require_nonneg(
        "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE",
        ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE,
    )
    _require_nonneg(
        "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP",
        ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP,
    )
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
    _require_nonneg("ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT", ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT)
    _require_nonneg(
        "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE",
        ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE,
    )
    _require_nonneg(
        "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT",
        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT,
    )
    _require_nonneg(
        "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE",
        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE,
    )
    _require_nonneg(
        "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS",
        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS,
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
    if int(ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION) not in (0, 1):
        raise RuntimeError(
            "[ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION_INVALID] "
            f"ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION={ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION} expected 0 or 1"
        )
    if int(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL) not in (0, 1):
        raise RuntimeError(
            "[ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL_INVALID] "
            "ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL="
            f"{ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL} expected 0 or 1"
        )
    if int(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE) not in (0, 1):
        raise RuntimeError(
            "[ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE_INVALID] "
            "ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE="
            f"{ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE} expected 0 or 1"
        )
    if ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION not in {"logprob", "margin"}:
        raise RuntimeError(
            "[ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION_INVALID] "
            f"ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION={ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION!r} "
            "expected 'logprob' or 'margin'"
        )
    if int(ENTRY_HIER_PUBLIC_TRADE_HEAD) not in (0, 1):
        raise RuntimeError(
            "[ENTRY_HIER_PUBLIC_TRADE_HEAD_INVALID] "
            f"ENTRY_HIER_PUBLIC_TRADE_HEAD={ENTRY_HIER_PUBLIC_TRADE_HEAD} expected 0 or 1"
        )
    if int(ENTRY_HIER_PUBLIC_SIDE_HEAD) not in (0, 1):
        raise RuntimeError(
            "[ENTRY_HIER_PUBLIC_SIDE_HEAD_INVALID] "
            f"ENTRY_HIER_PUBLIC_SIDE_HEAD={ENTRY_HIER_PUBLIC_SIDE_HEAD} expected 0 or 1"
        )
    if int(ENTRY_HIER_CTX_PRIOR_ADAPTER) not in (0, 1):
        raise RuntimeError(
            "[ENTRY_HIER_CTX_PRIOR_ADAPTER_INVALID] "
            f"ENTRY_HIER_CTX_PRIOR_ADAPTER={ENTRY_HIER_CTX_PRIOR_ADAPTER} expected 0 or 1"
        )
    if int(ENTRY_HIER_CTX_DIRECTION_CALIBRATION) not in (0, 1):
        raise RuntimeError(
            "[ENTRY_HIER_CTX_DIRECTION_CALIBRATION_INVALID] "
            f"ENTRY_HIER_CTX_DIRECTION_CALIBRATION={ENTRY_HIER_CTX_DIRECTION_CALIBRATION} expected 0 or 1"
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
    if ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE="
            f"{ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE > 1.0:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE_INVALID] "
            "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE="
            f"{ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE:.6f} expected <=1.0"
        )
    if ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS < 2:
        raise RuntimeError(
            "[ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS_INVALID] "
            "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS="
            f"{ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS} expected >=2"
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
    if bool(enable_side_validity_head) and ENTRY_HIER_SIDE_VALIDITY_WEIGHT <= 0.0:
        raise RuntimeError(
            "[ENTRY_SIDE_VALIDITY_HEAD_UNTRAINED] enable_side_validity_head=true requires "
            f"ENTRY_HIER_SIDE_VALIDITY_WEIGHT>0, got {ENTRY_HIER_SIDE_VALIDITY_WEIGHT:.6f}"
        )
    if bool(enable_anchor_gate and enable_hierarchical_entry_heads):
        repair_failures: list[str] = []
        repair_failures.extend(
            _xau_direction_repair_source_failures(
                {
                    "train_parquet": train_parquet,
                    "val_parquet": val_parquet,
                    "m5_prebuilt_path": m5_prebuilt_path,
                }
            )
        )
        for split_name, ds_obj in (("train", train_ds), ("val", val_ds)):
            if not bool(getattr(ds_obj, "neutral_xgb_bridge", False)):
                repair_failures.append(
                    f"{split_name} dataset manifest must declare neutral_xgb_bridge=true "
                    "for XAU direction repair heads"
                )
            if str(getattr(ds_obj, "xgb_bridge_source", "") or "") != "neutral_uniform_proba":
                repair_failures.append(
                    f"{split_name} dataset manifest must declare xgb_bridge_source=neutral_uniform_proba "
                    f"(got {getattr(ds_obj, 'xgb_bridge_source', '')!r})"
                )
        if list(getattr(train_ds, "signal_names", [])) != list(getattr(val_ds, "signal_names", [])):
            repair_failures.append("train/val signal_names differ for XAU direction repair heads")
        if int(getattr(train_ds, "seq_input_dim", -1)) != int(getattr(val_ds, "seq_input_dim", -2)):
            repair_failures.append("train/val seq_input_dim differ for XAU direction repair heads")
        if float(ENTRY_BAD_PATH_PROB_PENALTY) > 0.0:
            repair_failures.append(
                "ENTRY_BAD_PATH_PROB_PENALTY="
                f"{ENTRY_BAD_PATH_PROB_PENALTY:.3f} expected 0.0 for side-specific XAU repair"
            )
        if float(anchor_gate_init) > 0.05:
            repair_failures.append(f"anchor_gate_init={float(anchor_gate_init):.3f} expected <=0.05")
        if ENTRY_PRED_BALANCE_ALPHA < 0.45:
            repair_failures.append(f"ENTRY_PRED_BALANCE_ALPHA={ENTRY_PRED_BALANCE_ALPHA:.3f} expected >=0.45")
        if [float(value) for value in ENTRY_PRED_BALANCE_CLASS_WEIGHTS] != [1.0, 1.0, 4.0]:
            repair_failures.append(
                "ENTRY_PRED_BALANCE_CLASS_WEIGHTS="
                + ",".join(str(float(value)) for value in ENTRY_PRED_BALANCE_CLASS_WEIGHTS)
                + " expected 1.0,1.0,4.0"
            )
        if ENTRY_DIRECTION_CE_SCALE < 2.0:
            repair_failures.append(f"ENTRY_DIRECTION_CE_SCALE={ENTRY_DIRECTION_CE_SCALE:.3f} expected >=2.0")
        if ENTRY_CKPT_MONITOR != "dir_acc":
            repair_failures.append(f"GX1_V10_CKPT_MONITOR={ENTRY_CKPT_MONITOR!r} expected 'dir_acc'")
        if ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT < 0.50:
            repair_failures.append(
                "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT="
                f"{ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT:.3f} expected >=0.50"
            )
        if ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL < 0.35:
            repair_failures.append(
                "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL="
                f"{ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL:.3f} expected >=0.35"
            )
        if ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE < 0.05:
            repair_failures.append(
                "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE="
                f"{ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE:.3f} expected >=0.05"
            )
        if not bool(ENTRY_CKPT_DIRECTION_SLICE_GUARD):
            repair_failures.append("ENTRY_CKPT_DIRECTION_SLICE_GUARD=0 expected 1")
        if not bool(ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER):
            repair_failures.append("ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER=0 expected 1")
        if ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS="
                f"{ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS} expected >=8"
            )
        if ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT < 2.50:
            repair_failures.append(
                "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT="
                f"{ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT:.3f} expected >=2.50"
            )
        if ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION < 0.50:
            repair_failures.append(
                "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION="
                f"{ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION:.3f} expected >=0.50"
            )
        if ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR < 0.05:
            repair_failures.append(
                "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR="
                f"{ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR:.3f} expected >=0.05"
            )
        if ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE > 0.05:
            repair_failures.append(
                "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE="
                f"{ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE:.3f} expected <=0.05"
            )
        if ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT < 8.0:
            repair_failures.append(
                "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT="
                f"{ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT:.3f} expected >=8.0"
            )
        if ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE > 0.02:
            repair_failures.append(
                "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE="
                f"{ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE:.3f} expected <=0.02"
            )
        if ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE="
                f"{ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT < 2.0:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT="
                f"{ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT:.3f} expected >=2.0"
            )
        if ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE="
                f"{ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS="
                f"{ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS} expected >=8"
            )
        if ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT < 2.0:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT="
                f"{ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT:.3f} expected >=2.0"
            )
        if ENTRY_DIRECTION_SLICE_TRUE_MARGIN < 0.05:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_TRUE_MARGIN="
                f"{ENTRY_DIRECTION_SLICE_TRUE_MARGIN:.3f} expected >=0.05"
            )
        if ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE="
                f"{ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS="
                f"{ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS} expected >=8"
            )
        if ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION != "mean_max":
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION="
                f"{ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION!r} expected 'mean_max'"
            )
        if ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE < 1:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE="
                f"{ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE} expected >=1"
            )
        if ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS < 1:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS="
                f"{ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS} expected >=1"
            )
        if ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT="
                f"{ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN < 0.02:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN="
                f"{ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN:.3f} expected >=0.02"
            )
        if ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT="
                f"{ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN < 0.02:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN="
                f"{ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN:.3f} expected >=0.02"
            )
        if ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE="
                f"{ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS="
                f"{ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS} expected >=8"
            )
        if ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT < 3.0:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT="
                f"{ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT:.3f} expected >=3.0"
            )
        if ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE > 0.02:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE="
                f"{ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE:.3f} expected <=0.02"
            )
        if ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE="
                f"{ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS="
                f"{ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS} expected >=8"
            )
        if ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT="
                f"{ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE > 0.02:
            repair_failures.append(
                "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE="
                f"{ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE:.3f} expected <=0.02"
            )
        if ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE="
                f"{ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT="
                f"{ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE > 0.02:
            repair_failures.append(
                "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE="
                f"{ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE:.3f} expected <=0.02"
            )
        if ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE="
                f"{ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS="
                f"{ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS} expected >=8"
            )
        if ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT="
                f"{ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN < 0.02:
            repair_failures.append(
                "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN="
                f"{ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN:.3f} expected >=0.02"
            )
        if ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT < 8.0:
            repair_failures.append(
                "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT="
                f"{ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT:.3f} expected >=8.0"
            )
        if ENTRY_HIER_FLAT_LOGIT_MARGIN < 0.10:
            repair_failures.append(
                "ENTRY_HIER_FLAT_LOGIT_MARGIN="
                f"{ENTRY_HIER_FLAT_LOGIT_MARGIN:.3f} expected >=0.10"
            )
        if ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE="
                f"{ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT < 8.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT="
                f"{ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT:.3f} expected >=8.0"
            )
        if ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN="
                f"{ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE="
                f"{ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS="
                f"{ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS} expected >=8"
            )
        if ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT="
                f"{ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE="
                f"{ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT="
                f"{ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE="
                f"{ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS="
                f"{ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS} expected >=8"
            )
        if ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT < 3.0:
            repair_failures.append(
                "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT="
                f"{ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT:.3f} expected >=3.0"
            )
        if ENTRY_DIRECTION_VS_FLAT_MARGIN < 0.05:
            repair_failures.append(
                "ENTRY_DIRECTION_VS_FLAT_MARGIN="
                f"{ENTRY_DIRECTION_VS_FLAT_MARGIN:.3f} expected >=0.05"
            )
        if ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT="
                f"{ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS > 15.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS="
                f"{ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS:.3f} expected <=15.0"
            )
        if ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN="
                f"{ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT < 6.0:
            repair_failures.append(
                "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT="
                f"{ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT:.3f} expected >=6.0"
            )
        if ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS > 15.0:
            repair_failures.append(
                "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS="
                f"{ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS:.3f} expected <=15.0"
            )
        if ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN="
                f"{ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT < 8.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT="
                f"{ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT:.3f} expected >=8.0"
            )
        if ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS > 15.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS="
                f"{ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS:.3f} expected <=15.0"
            )
        if ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS > 0.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS="
                f"{ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS:.3f} expected <=0.0"
            )
        if ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH > 0.50:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH="
                f"{ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH:.3f} expected <=0.50"
            )
        if ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN="
                f"{ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT < 8.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT="
                f"{ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT:.3f} expected >=8.0"
            )
        if ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS > 15.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS="
                f"{ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS:.3f} expected <=15.0"
            )
        if ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS > 0.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS="
                f"{ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS:.3f} expected <=0.0"
            )
        if ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH > 0.50:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH="
                f"{ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH:.3f} expected <=0.50"
            )
        if ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP < 2.0:
            repair_failures.append(
                "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP="
                f"{ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP:.3f} expected >=2.0"
            )
        if not bool(enable_hierarchical_direction_composition):
            repair_failures.append("ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION=0 expected 1")
        if ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP < 0.10:
            repair_failures.append(
                "ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP="
                f"{ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP:.3f} expected >=0.10"
            )
        if ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP > 0.20:
            repair_failures.append(
                "ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP="
                f"{ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP:.3f} expected <=0.20"
            )
        if not bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL):
            repair_failures.append("ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL=0 expected 1")
        if not bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE):
            repair_failures.append("ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE=0 expected 1")
        if ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION != "margin":
            repair_failures.append(
                "ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION="
                f"{ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION!r} expected 'margin'"
            )
        if not bool(ENTRY_HIER_PUBLIC_TRADE_HEAD):
            repair_failures.append("ENTRY_HIER_PUBLIC_TRADE_HEAD=0 expected 1")
        if not bool(ENTRY_HIER_PUBLIC_SIDE_HEAD):
            repair_failures.append("ENTRY_HIER_PUBLIC_SIDE_HEAD=0 expected 1")
        if not bool(ENTRY_HIER_CTX_PRIOR_ADAPTER):
            repair_failures.append("ENTRY_HIER_CTX_PRIOR_ADAPTER=0 expected 1")
        if ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE < 0.25:
            repair_failures.append(
                "ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE="
                f"{ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE:.3f} expected >=0.25"
            )
        if ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE > 1.00:
            repair_failures.append(
                "ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE="
                f"{ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE:.3f} expected <=1.00"
            )
        if not bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION):
            repair_failures.append("ENTRY_HIER_CTX_DIRECTION_CALIBRATION=0 expected 1")
        if ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE < 0.25:
            repair_failures.append(
                "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE="
                f"{ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE:.3f} expected >=0.25"
            )
        if ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE > 1.00:
            repair_failures.append(
                "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE="
                f"{ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE:.3f} expected <=1.00"
            )
        if ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP < 0.10:
            repair_failures.append(
                "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP="
                f"{ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP:.3f} expected >=0.10"
            )
        if ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP > 0.50:
            repair_failures.append(
                "ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP="
                f"{ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP:.3f} expected <=0.50"
            )
        if ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT < 8.0:
            repair_failures.append(
                "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT="
                f"{ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT:.3f} expected >=8.0"
            )
        if ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE="
                f"{ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS="
                f"{ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS} expected >=8"
            )
        if ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION < 0.50:
            repair_failures.append(
                "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION="
                f"{ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION:.3f} expected >=0.50"
            )
        if ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR="
                f"{ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR:.3f} expected >=0.10"
            )
        if ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN < 0.10:
            repair_failures.append(
                "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN="
                f"{ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN:.3f} expected >=0.10"
            )
        if ENTRY_HIER_LEGACY_CE_MULT < 1.0:
            repair_failures.append(f"ENTRY_HIER_LEGACY_CE_MULT={ENTRY_HIER_LEGACY_CE_MULT:.3f} expected >=1.0")
        if not bool(enable_side_validity_head):
            repair_failures.append("enable_side_validity_head=false expected true for XAU direction repair")
        if ENTRY_HIER_SIDE_VALIDITY_WEIGHT < 1.50:
            repair_failures.append(
                "ENTRY_HIER_SIDE_VALIDITY_WEIGHT="
                f"{ENTRY_HIER_SIDE_VALIDITY_WEIGHT:.3f} expected >=1.50"
            )
        if ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS < 15.0:
            repair_failures.append(
                "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS="
                f"{ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS:.3f} expected >=15.0"
            )
        if ENTRY_HIER_SLICE_SIDE_CE_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT="
                f"{ENTRY_HIER_SLICE_SIDE_CE_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT < 3.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT="
                f"{ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT:.3f} expected >=3.0"
            )
        if ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN="
                f"{ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT="
                f"{ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN < 0.02:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN="
                f"{ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN:.3f} expected >=0.02"
            )
        if ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE="
                f"{ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_SIDE_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_MIN_ROWS="
                f"{ENTRY_HIER_SLICE_SIDE_MIN_ROWS} expected >=8"
            )
        if ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT="
                f"{ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE > 0.02:
            repair_failures.append(
                "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE="
                f"{ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE:.3f} expected <=0.02"
            )
        if ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE="
                f"{ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT="
                f"{ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE > 0.02:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE="
                f"{ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE:.3f} expected <=0.02"
            )
        if ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE < 0.10:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE="
                f"{ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE:.3f} expected >=0.10"
            )
        if ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS < 8:
            repair_failures.append(
                "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS="
                f"{ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS} expected >=8"
            )
        if ENTRY_HIER_POCKET_ABSTAIN_WEIGHT < 5.0:
            repair_failures.append(
                "ENTRY_HIER_POCKET_ABSTAIN_WEIGHT="
                f"{ENTRY_HIER_POCKET_ABSTAIN_WEIGHT:.3f} expected >=5.0"
            )
        if ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT < 3.0:
            repair_failures.append(
                "ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT="
                f"{ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT:.3f} expected >=3.0"
            )
        if not bool(enable_trendline_rail_head):
            repair_failures.append("enable_trendline_rail_head=false expected true for XAU direction repair")
        if ENTRY_TRENDLINE_RAIL_AUX_WEIGHT < 1.0:
            repair_failures.append(
                f"ENTRY_TRENDLINE_RAIL_AUX_WEIGHT={ENTRY_TRENDLINE_RAIL_AUX_WEIGHT:.3f} expected >=1.0"
            )
        if ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT < 1.50:
            repair_failures.append(
                "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT="
                f"{ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT:.3f} expected >=1.50"
            )
        if ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT < 5.0:
            repair_failures.append(
                "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT="
                f"{ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT:.3f} expected >=5.0"
            )
        if ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT < 4.0:
            repair_failures.append(
                "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT="
                f"{ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT:.3f} expected >=4.0"
            )
        if ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT < 3.0:
            repair_failures.append(
                "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT="
                f"{ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT:.3f} expected >=3.0"
            )
        if ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT < 5.0:
            repair_failures.append(
                "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT="
                f"{ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT:.3f} expected >=5.0"
            )
        if repair_failures:
            raise RuntimeError("[ENTRY_XAU_DIRECTION_REPAIR_RECIPE_INVALID] " + "; ".join(repair_failures))
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
        "[ENTRY_TRAIN_RECIPE] direction_ce_scale=%.3f tail_direction_w=%.3f tail_direction_q=%.3f residual_scale=%.3f tradable_w=%.3f path_w=%.3f mfe_w=%.3f tradable_pos_weight=%.3f bad_path_w=%.3f bad_path_pos_weight=%.3f clean_edge_w=%.3f clean_edge_pos_weight=%.3f survival_w=%.3f survival_pos_weight=%.3f rank_w=%.3f rank_margin=%.3f",
        float(ENTRY_DIRECTION_CE_SCALE),
        float(ENTRY_TAIL_DIRECTION_CE_WEIGHT),
        float(ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE),
        float(ENTRY_RESIDUAL_SCALE),
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
        "[ENTRY_TRAIN_PARKED] cost_sensitive=%d cost_scale=%.3f pred_balance_alpha=%.3f pred_balance_class_weights=%s residual_side_bias_alpha=%.3f "
        "timing_scale=%.3f early_w=%.3f quality_w=%.3f bad_path_w=%.3f xgb_short_penalty=%.3f short_class_weight=%.3f",
        int(bool(ENTRY_COST_SENSITIVE_ENABLED)),
        float(ENTRY_COST_SENSITIVE_SCALE),
        float(ENTRY_PRED_BALANCE_ALPHA),
        ",".join(f"{float(value):.3f}" for value in ENTRY_PRED_BALANCE_CLASS_WEIGHTS),
        float(ENTRY_RESIDUAL_SIDE_BIAS_ALPHA),
        float(ENTRY_TIMING_LOSS_SCALE),
        float(ENTRY_AUX_EARLY_WEIGHT),
        float(ENTRY_AUX_QUALITY_WEIGHT),
        float(ENTRY_AUX_BAD_PATH_WEIGHT),
        float(XGB_SHORT_LONG_PENALTY),
        float(SHORT_CLASS_WEIGHT),
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
        "utility_triad_ce_class_weight_cap=%.3f hierarchical_composition=%d "
        "hier_compose_residual_cap=%.3f hier_compose_residual_side_neutral=%d "
        "hier_compose_public_flat_from_trade=%d hier_public_trade_head=%d "
        "hier_public_side_head=%d hier_public_direction_composition=%s "
        "hier_ctx_prior_adapter=%d "
        "hier_ctx_prior_adapter_scale=%.3f hier_ctx_direction_calibration=%d "
        "hier_ctx_direction_calibration_scale=%.3f hier_ctx_direction_calibration_cap=%.3f "
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
        int(bool(enable_hierarchical_direction_composition)),
        float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
        int(bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL)),
        int(bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE)),
        int(bool(ENTRY_HIER_PUBLIC_TRADE_HEAD)),
        int(bool(ENTRY_HIER_PUBLIC_SIDE_HEAD)),
        str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
        int(bool(ENTRY_HIER_CTX_PRIOR_ADAPTER)),
        float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
        int(bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION)),
        float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE),
        float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
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
        "[ENTRY_HIER_RECIPE] enabled=%d legacy_ce_mult=%.3f trade_w=%.3f side_w=%.3f utility_w=%.3f bad_path_w=%.3f mae_w=%.3f "
        "trade_global_prior_w=%.3f trade_global_prior_tol=%.3f trade_global_prior_min_label_rate=%.3f "
        "slice_trade_prior_w=%.3f slice_trade_prior_tol=%.3f slice_trade_prior_min_rows=%d slice_trade_prior_min_label_rate=%.3f "
        "slice_trade_acc_edge_w=%.3f slice_trade_acc_edge_margin=%.3f "
        "flat_logit_margin_w=%.3f flat_logit_margin=%.3f flat_logit_margin_min_label_rate=%.3f "
        "slice_flat_logit_margin_w=%.3f slice_flat_logit_margin=%.3f slice_flat_logit_margin_min_rows=%d slice_flat_logit_margin_min_label_rate=%.3f "
        "public_flat_consistency_w=%.3f public_flat_consistency_min_label_rate=%.3f "
        "slice_public_flat_consistency_w=%.3f slice_public_flat_consistency_min_rows=%d slice_public_flat_consistency_min_label_rate=%.3f "
        "slice_side_ce_w=%.3f slice_side_margin_w=%.3f slice_side_margin=%.3f "
        "slice_side_acc_edge_w=%.3f slice_side_acc_edge_margin=%.3f "
        "slice_side_min_rows=%d slice_side_min_label_rate=%.3f "
        "side_global_prior_w=%.3f side_global_prior_tol=%.3f side_global_prior_min_label_rate=%.3f "
        "slice_side_prior_w=%.3f slice_side_prior_tol=%.3f slice_side_prior_min_rows=%d slice_side_prior_min_label_rate=%.3f "
        "trade_pos_weight=%.3f bad_path_pos_weight_long=%.3f bad_path_pos_weight_short=%.3f "
        "utility_scale_bps=%.3f mae_scale_bps=%.3f pocket_abstain_w=%.3f pocket_side_margin_w=%.3f pocket_utility_margin_bps=%.3f",
        int(bool(enable_hierarchical_entry_heads)),
        float(ENTRY_HIER_LEGACY_CE_MULT),
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
        float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT),
        float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE),
        float(ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT),
        int(ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS),
        float(ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE),
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
        float(ENTRY_HIER_POCKET_ABSTAIN_WEIGHT),
        float(ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT),
        float(ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS),
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
            _steps_per_epoch = max(1, len(train_loader))
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
    _ckpt_monitor = ENTRY_CKPT_MONITOR if ENTRY_CKPT_MONITOR in {"val_loss", "dir_acc"} else "val_loss"
    log.info(
        "[CKPT_MONITOR] selecting best checkpoint on %s class_balance_guard_weight=%.3f min_pred_to_label=%.3f min_pred_rate=%.3f",
        _ckpt_monitor,
        float(ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT),
        float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL),
        float(ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE),
    )

    for epoch in range(epochs):
        last_epoch = epoch + 1
        tr_loss, tr_stats = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            short_lead_margin=XGB_SHORT_LEAD_MARGIN,
            long_penalty_weight=XGB_SHORT_LONG_PENALTY,
            residual_side_bias_alpha=ENTRY_RESIDUAL_SIDE_BIAS_ALPHA,
            timing_target_bps=ENTRY_TIMING_TARGET_BPS,
            timing_loss_scale=ENTRY_TIMING_LOSS_SCALE,
            aux_early_weight=ENTRY_AUX_EARLY_WEIGHT,
            aux_quality_weight=ENTRY_AUX_QUALITY_WEIGHT,
            aux_path_weight=ENTRY_AUX_PATH_WEIGHT,
            aux_mfe_weight=ENTRY_AUX_MFE_WEIGHT,
            aux_tradable_weight=ENTRY_AUX_TRADABLE_WEIGHT,
            aux_quality_scale_bps=ENTRY_AUX_QUALITY_SCALE_BPS,
            aux_path_scale_bps=ENTRY_AUX_PATH_SCALE_BPS,
            aux_mfe_scale_bps=ENTRY_AUX_MFE_SCALE_BPS,
            tradable_pos_weight=tradable_pos_weight,
            clean_edge_pos_weight=clean_edge_pos_weight,
            survival_pos_weight=survival_pos_weight,
            bad_path_pos_weight=bad_path_pos_weight,
            hier_trade_pos_weight=hier_trade_pos_weight,
            hier_bad_path_pos_weight=hier_bad_path_pos_weight,
            scheduler=_scheduler,
        )
        va_loss, auc, acc, val_short_to_long, val_stats = validate(
            model,
            val_loader,
            criterion,
            device,
            residual_side_bias_alpha=ENTRY_RESIDUAL_SIDE_BIAS_ALPHA,
            aux_early_weight=ENTRY_AUX_EARLY_WEIGHT,
            aux_quality_weight=ENTRY_AUX_QUALITY_WEIGHT,
            aux_path_weight=ENTRY_AUX_PATH_WEIGHT,
            aux_mfe_weight=ENTRY_AUX_MFE_WEIGHT,
            aux_tradable_weight=ENTRY_AUX_TRADABLE_WEIGHT,
            aux_quality_scale_bps=ENTRY_AUX_QUALITY_SCALE_BPS,
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
        if tr_stats:
            anchor_abs_mean = float(tr_stats.get("anchor_abs_mean") or 0.0)
            delta_abs_mean = float(tr_stats.get("delta_abs_mean") or 0.0)
            scaled_delta_abs_mean = float(tr_stats.get("scaled_delta_abs_mean") or 0.0)
            final_minus_anchor_abs_mean = float(tr_stats.get("final_minus_anchor_abs_mean") or 0.0)
            ratio = (scaled_delta_abs_mean / max(anchor_abs_mean, 1e-12))
            log.info(
                "[ENTRY_RESIDUAL_MAG_PROOF] split=train epoch=%d "
                "anchor_abs_mean=%.6f delta_abs_mean=%.6f scaled_delta_abs_mean=%.6f "
                "final_minus_anchor_abs_mean=%.6f scaled_delta_to_anchor_ratio=%.6f",
                epoch + 1,
                anchor_abs_mean,
                delta_abs_mean,
                scaled_delta_abs_mean,
                final_minus_anchor_abs_mean,
                ratio,
            )
        if val_stats:
            anchor_abs_mean = float(val_stats.get("anchor_abs_mean") or 0.0)
            delta_abs_mean = float(val_stats.get("delta_abs_mean") or 0.0)
            scaled_delta_abs_mean = float(val_stats.get("scaled_delta_abs_mean") or 0.0)
            final_minus_anchor_abs_mean = float(val_stats.get("final_minus_anchor_abs_mean") or 0.0)
            ratio = (scaled_delta_abs_mean / max(anchor_abs_mean, 1e-12))
            log.info(
                "[ENTRY_RESIDUAL_MAG_PROOF] split=val epoch=%d "
                "anchor_abs_mean=%.6f delta_abs_mean=%.6f scaled_delta_abs_mean=%.6f "
                "final_minus_anchor_abs_mean=%.6f scaled_delta_to_anchor_ratio=%.6f",
                epoch + 1,
                anchor_abs_mean,
                delta_abs_mean,
                scaled_delta_abs_mean,
                final_minus_anchor_abs_mean,
                ratio,
            )
            log.info(
                "[ENTRY_LOSS_SUMMARY] split=val epoch=%d ce=%.6f min_pred=%.6f global_prior=%.6f slice_min_pred=%.6f flat_margin=%.6f utility_margin=%.6f side_utility_conviction=%.6f utility_trade_conviction=%.6f utility_triad_ce=%.6f flat_starvation=%.6f slice_recall=%.6f slice_bal_ce=%.6f slice_true_margin=%.6f slice_acc_edge=%.6f slice_confusion_pair=%.6f slice_prior=%.6f tail_direction=%.6f tail_rows=%d path=%.6f mfe=%.6f tradable=%.6f hier_trade=%.6f hier_trade_global_prior=%.6f hier_slice_trade_prior=%.6f hier_flat_logit_margin=%.6f hier_slice_flat_logit_margin=%.6f hier_public_flat_consistency=%.6f hier_slice_public_flat_consistency=%.6f hier_side=%.6f hier_slice_side_ce=%.6f hier_slice_side_margin=%.6f hier_slice_side_acc_edge=%.6f hier_side_global_prior=%.6f hier_slice_side_prior=%.6f hier_side_acc=%.4f total=%.6f",
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
                float(val_stats.get("hier_public_flat_consistency_loss_mean", 0.0)),
                float(val_stats.get("hier_slice_public_flat_consistency_loss_mean", 0.0)),
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
            "[SHORT_TO_LONG_TRAIN] rate=%.6f short_lead_count=%d short_lead_long_prob_mean=%.6f",
            float(tr_stats.get("short_pred_long_rate", 0.0)),
            int(tr_stats.get("short_lead_count", 0)),
            float(tr_stats.get("short_lead_long_prob_mean", 0.0)),
        )
        if tr_stats:
            log.info(
                "[ENTRY_LOSS_SUMMARY] split=train epoch=%d ce=%.6f min_pred=%.6f global_prior=%.6f slice_min_pred=%.6f flat_margin=%.6f utility_margin=%.6f side_utility_conviction=%.6f utility_trade_conviction=%.6f utility_triad_ce=%.6f flat_starvation=%.6f slice_recall=%.6f slice_bal_ce=%.6f slice_true_margin=%.6f slice_acc_edge=%.6f slice_confusion_pair=%.6f slice_prior=%.6f tail_direction=%.6f tail_rows=%d path=%.6f mfe=%.6f tradable=%.6f hier_trade=%.6f hier_trade_global_prior=%.6f hier_slice_trade_prior=%.6f hier_flat_logit_margin=%.6f hier_slice_flat_logit_margin=%.6f hier_public_flat_consistency=%.6f hier_slice_public_flat_consistency=%.6f hier_side=%.6f hier_slice_side_ce=%.6f hier_slice_side_margin=%.6f hier_slice_side_acc_edge=%.6f hier_side_global_prior=%.6f hier_slice_side_prior=%.6f hier_side_acc=%.4f total=%.6f",
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
                float(tr_stats.get("hier_public_flat_consistency_loss_mean", 0.0)),
                float(tr_stats.get("hier_slice_public_flat_consistency_loss_mean", 0.0)),
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
                "[ENTRY_SPECIALIST_GATE_LOSS] split=train epoch=%d loss=%.6f entropy=%.6f min_mean=%.6f",
                epoch + 1,
                float(tr_stats.get("specialist_gate_loss_mean", 0.0)),
                float(tr_stats.get("specialist_gate_entropy_mean", 0.0)),
                float(tr_stats.get("specialist_gate_min_mean", 0.0)),
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
                        "direction_slice_ckpt_score" if bool(xau_direction_repair_mode) else "direction_ckpt_score",
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
                "balance_guard_ok=%d slice_contract_ok=%d monitor=%s",
                best_epoch,
                best_val,
                acc,
                best_dir_ckpt_score,
                int(bool(best_direction_balance_guard_ok)),
                int(bool(best_direction_slice_contract_ok)),
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
                "vedtak_id": str(vedtak_id or ""),
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
                    "direction_hierarchical_composition": bool(enable_hierarchical_direction_composition),
                    "hier_compose_residual_logit_cap": float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
                    "hier_compose_residual_side_neutral": bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
                    "hier_compose_public_flat_from_trade": bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE),
                    "hier_public_direction_composition": str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
                    "hier_public_trade_head": bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
                    "hier_public_side_head": bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
                    "hier_ctx_prior_adapter": bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
                    "hier_ctx_prior_adapter_scale": float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
                    "hier_ctx_direction_calibration": bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
                    "hier_ctx_direction_calibration_scale": float(
                        ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE
                    ),
                    "hier_ctx_direction_calibration_cap": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
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
                    "hier_public_flat_consistency_weight": float(
                        ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT
                    ),
                    "hier_public_flat_consistency_min_label_rate": float(
                        ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
                    ),
                    "hier_slice_public_flat_consistency_weight": float(
                        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT
                    ),
                    "hier_slice_public_flat_consistency_min_label_rate": float(
                        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
                    ),
                    "hier_slice_public_flat_consistency_min_rows": int(
                        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS
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
                "vedtak_id": str(vedtak_id or ""),
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
                    "direction_hierarchical_composition": bool(enable_hierarchical_direction_composition),
                    "hier_compose_residual_logit_cap": float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
                    "hier_compose_residual_side_neutral": bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
                    "hier_compose_public_flat_from_trade": bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE),
                    "hier_public_direction_composition": str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
                    "hier_public_trade_head": bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
                    "hier_public_side_head": bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
                    "hier_ctx_prior_adapter": bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
                    "hier_ctx_prior_adapter_scale": float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
                    "hier_ctx_direction_calibration": bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
                    "hier_ctx_direction_calibration_scale": float(
                        ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE
                    ),
                    "hier_ctx_direction_calibration_cap": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
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
                    "hier_public_flat_consistency_weight": float(
                        ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT
                    ),
                    "hier_public_flat_consistency_min_label_rate": float(
                        ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
                    ),
                    "hier_slice_public_flat_consistency_weight": float(
                        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT
                    ),
                    "hier_slice_public_flat_consistency_min_label_rate": float(
                        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
                    ),
                    "hier_slice_public_flat_consistency_min_rows": int(
                        ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS
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

    # Resolve output bundle dir (under GX1_DATA if relative)
    out_bundle_dir = _resolve_train_out_bundle_dir(out_bundle_dir, gx1_data_override)
    out_bundle_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_bundle_dir / "model_state_dict.pt"
    torch.save(best_state, model_path)
    state_dict_sha256 = _sha256_file(model_path)
    trained_signal_names = list(getattr(train_ds, "signal_names", _default_signal_names(seq_input_dim)))
    trained_neutral_xgb_bridge = bool(getattr(train_ds, "neutral_xgb_bridge", False))
    trained_xgb_bridge_source = str(getattr(train_ds, "xgb_bridge_source", "") or "")
    trained_smart520_state_contract = _smart520_state_contract_for_parquet(Path(train_parquet))
    if xau_direction_repair_mode:
        state_contract_failures = _smart520_state_contract_failures(
            trained_smart520_state_contract,
            split="train",
        )
        if state_contract_failures:
            raise RuntimeError(
                "[XAU_DIRECTION_REPAIR_STATE_CONTRACT_FAIL] "
                + " | ".join(state_contract_failures)
            )

    lock = {
        "version": "entry_v10_ctx_lock_v1",
        "created_at_utc": _utc_now(),
        "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
        "signal_bridge_contract_sha256": SIGNAL_BRIDGE_CONTRACT_SHA256,
        "ctx_tag": f"CTX6CAT{ctx_cat_dim}",
        "ctx_cont_dim": ctx_cont_dim,
        "ctx_cat_dim": ctx_cat_dim,
        "ordered_ctx_cont_names": list(ordered_ctx_cont_names),
        "ordered_ctx_cat_names": list(ordered_ctx_cat_names),
        "ordered_signal_names": trained_signal_names,
        "neutral_xgb_bridge": trained_neutral_xgb_bridge,
        "xgb_bridge_source": trained_xgb_bridge_source,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "seq_len": seq_len,
        "num_classes": 3,
        "class_order": [0, 1, 2],
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": state_dict_sha256,
    }
    (out_bundle_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(
        json.dumps(lock, indent=2)
    )

    # 2026-06-02: extract learned per-TF input scales from best_state.
    # These are saved next to the initial priors so inference can rebuild the
    # model identically (init values define the Parameter, then state_dict
    # overwrites with the learned values).
    learned_tf_input_scales: Dict[str, float] = {}
    if enable_tf_input_scale:
        for _tf in ("m5", "m15", "h1", "h4", "d1"):
            _key = f"tf_input_scale_{_tf}"
            if _key in best_state:
                learned_tf_input_scales[_tf] = float(best_state[_key].item())
            elif f"_orig_mod.{_key}" in best_state:  # torch.compile prefix
                learned_tf_input_scales[_tf] = float(best_state[f"_orig_mod.{_key}"].item())
        if learned_tf_input_scales:
            log.info(
                "[TF_INPUT_SCALE_LEARNED] %s",
                {k: round(v, 4) for k, v in learned_tf_input_scales.items()},
            )

    active_heads = _build_active_head_names(
        enable_tf_agreement_head=enable_tf_agreement_head,
        enable_path_quality_variance_head=enable_path_quality_variance_head,
        enable_position_size_head=enable_position_size_head,
        enable_hold_horizon_head=enable_hold_horizon_head,
        enable_mtf_direction_head=enable_mtf_direction_head,
        enable_dip_head=enable_dip_head,
        enable_forecast_head=enable_forecast_head,
        enable_timing_head=enable_timing_head,
        enable_tail_risk_head=enable_tail_risk_head,
        enable_vol_forecast_head=enable_vol_forecast_head,
        enable_anchor_gate=bool(enable_anchor_gate),
        enable_hierarchical_entry_heads=bool(enable_hierarchical_entry_heads),
        enable_hierarchical_public_trade_head=bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
        enable_hierarchical_public_side_head=bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
        hierarchical_public_direction_composition=str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
        enable_hierarchical_ctx_prior_adapter=bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
        enable_hierarchical_ctx_direction_calibration=bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
        enable_side_validity_head=bool(enable_side_validity_head),
        enable_trendline_rail_head=bool(enable_trendline_rail_head),
    )

    meta = {
        "created_at_utc": _utc_now(),
        "git_commit": _git_commit(),
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
            "enabled": bool(enable_multi_tf),
            "v2_mode": bool(_mtf_v2),
            "m5_seq_dim": int(_mtf_feat_count) if (enable_multi_tf and _mtf_v2) else 0,
            "m5_seq_len": int(multi_tf_seq_len) if (enable_multi_tf and _mtf_v2) else 0,
            "m15_seq_dim": int(_mtf_feat_count) if enable_multi_tf else 0,
            "h1_seq_dim": int(_mtf_feat_count) if enable_multi_tf else 0,
            "h4_seq_dim": int(_mtf_feat_count) if enable_multi_tf else 0,
            "d1_seq_dim": int(_mtf_feat_count) if enable_multi_tf else 0,
            "m15_seq_len": int(_m15_len),
            "h1_seq_len": int(multi_tf_seq_len),
            "h4_seq_len": int(_h4_len),
            "d1_seq_len": int(_d1_len),
            "multi_tf_scale": float(multi_tf_scale),
            "feature_contract": "MULTI_TF_PER_BAR_V2" if _mtf_v2 else "MULTI_TF_PER_BAR_V1",
            "closed_bar_target_availability": bool(
                getattr(train_ds, "_multi_tf_target_availability_shift", pd.Timedelta(0)) > pd.Timedelta(0)
            ),
            "target_availability_shift_minutes": float(
                getattr(train_ds, "_multi_tf_target_availability_shift", pd.Timedelta(0)).total_seconds()
                / 60.0
            ),
        },
        # 2026-06-02: per-TF learnable input scaling marker. Inference must
        # init the model with `enable_tf_input_scale=True` and the same init
        # values used at train time so state_dict load is shape-compatible.
        # Learned values overwrite the inits via state_dict; we surface them
        # here for inspection/debugging.
        "tf_input_scale": {
            "enabled": bool(enable_tf_input_scale),
            "init": {
                "m5": float(tf_input_scale_init_m5),
                "m15": float(tf_input_scale_init_m15),
                "h1": float(tf_input_scale_init_h1),
                "h4": float(tf_input_scale_init_h4),
                "d1": float(tf_input_scale_init_d1),
            },
            "learned": learned_tf_input_scales,
        },
        # Positional encoding marker — buffer is persistent=False (not in
        # state_dict), so the live bundle loader MUST read this to rebuild the
        # model with matching forward behaviour.
        "enable_pos_enc": bool(enable_pos_enc),
        "enable_regime_film": bool(enable_regime_film),
        "enable_mtf_direction_head": bool(enable_mtf_direction_head),  # forceful MTF→dir (2026-06-06)
        "mtf_dir_aux_weight": float(ENTRY_MTF_DIR_AUX_WEIGHT),
        "mtf_dir_aux_uses_direction_balance_repair": bool(
            enable_mtf_direction_head and float(ENTRY_MTF_DIR_AUX_WEIGHT) > 0.0
        ),
        "batch_size": batch_size,
        "seed": seed,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "ordered_signal_names": trained_signal_names,
        "neutral_xgb_bridge": trained_neutral_xgb_bridge,
        "xgb_bridge_source": trained_xgb_bridge_source,
        "smart520_state_contract": trained_smart520_state_contract,
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
        "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
        "ctx_tag": f"CTX6CAT{ctx_cat_dim}",
        "model_class": "EntryV10CtxHybridTransformer",
        "arch_id": "entry_v10_ctx_hybrid_transformer",
        "specialist_fusion": (
            {
                **specialist_meta,
                "num_layers": int(specialist_num_layers),
                "fusion_scale": float(specialist_fusion_scale),
            }
            if specialist_meta
            else {"enabled": False}
        ),
        "state_dict_sha256": state_dict_sha256,
        "anchored_entry_enabled": True,
        "anchor_source": "signal7_p_long_short_flat",
        "residual_scale": float(ENTRY_RESIDUAL_SCALE),
        "anchor_eps": float(ENTRY_ANCHOR_EPS),
        "anchor_gate": {
            "enabled": bool(enable_anchor_gate),
            "init": float(anchor_gate_init),
            "purpose": "learned per-regime suppression of signal-bridge anchor logits",
        },
        "hierarchical_direction_composition": {
            "enabled": bool(enable_hierarchical_direction_composition),
            "residual_logit_cap": float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
            "residual_side_neutral": bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
            "public_flat_from_trade": bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE),
            "public_direction_composition": str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
            "public_trade_head": {
                "enabled": bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
                "input": "shared_entry_representation",
                "applies_to": [
                    "public_long_logit",
                    "public_short_logit",
                    "public_flat_logit",
                ],
                "side_source": (
                    "public_side_logits"
                    if bool(ENTRY_HIER_PUBLIC_SIDE_HEAD)
                    else "side_logits"
                ),
                "direct_trade_supervision": bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
                "supervision_losses": [
                    "binary_cross_entropy_on_y_trade",
                    "global_prior_match",
                    "slice_prior_match",
                    "slice_accuracy_edge",
                    "flat_logit_margin",
                    "slice_flat_logit_margin",
                ],
                "runtime_rule_free": True,
            },
            "public_side_head": {
                "enabled": bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
                "input": "shared_entry_representation",
                "applies_to": ["public_long_logit", "public_short_logit"],
                "flat_source": (
                    "public_trade_logit"
                    if bool(ENTRY_HIER_PUBLIC_TRADE_HEAD)
                    else "trade_logit"
                ),
                "direct_side_supervision": bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
                "supervision_losses": [
                    "cross_entropy_on_y_side_mask",
                    "slice_balanced_ce",
                    "slice_true_margin",
                    "slice_accuracy_edge",
                    "global_prior_match",
                    "slice_prior_match",
                ],
                "runtime_rule_free": True,
            },
            "ctx_prior_adapter": {
                "enabled": bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
                "scale": float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
                "input": "ctx_cat_embeddings",
                "applies_to": (
                    ["trade_logit", "public_trade_logit", "side_logits"]
                    if bool(ENTRY_HIER_PUBLIC_TRADE_HEAD)
                    else ["trade_logit", "side_logits"]
                ),
                "runtime_rule_free": True,
            },
            "ctx_direction_calibration": {
                "enabled": bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
                "scale": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE),
                "cap": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
                "input": "ctx_cat_embeddings",
                "applies_to": ["public_direction_logits"],
                "runtime_rule_free": True,
            },
            "formula": (
                (
                    "logits=[public_trade_logit+public_side_long_logit, "
                    "public_trade_logit+public_side_short_logit, -public_trade_logit] "
                    "+ common(capped(residual_scale*delta_logits)) + capped(ctx direction calibration); "
                    "margin composition avoids joint-probability argmax starvation, "
                    "common residual is softmax-invariant, public trade/flat and public side use separate learned heads, "
                    "and public FLAT comes from hierarchy no-trade"
                )
                if bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE)
                and ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION == "margin"
                else (
                    "logits=[log P(trade)+log P(public long|trade), log P(trade)+log P(public short|trade), "
                    "log P(flat)] + common(capped(residual_scale*delta_logits)) + capped(ctx direction calibration); "
                    "common residual is softmax-invariant, public trade/flat and public side use separate learned heads, "
                    "and public FLAT comes from hierarchy no-trade"
                )
                if bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE)
                else
                (
                    "logits=[log P(trade)+log P(long|trade), log P(trade)+log P(short|trade), "
                    "log P(flat)] + capped(side_neutral(residual_scale*delta_logits))"
                )
                if bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL)
                else (
                    "logits=[log P(trade)+log P(long|trade), log P(trade)+log P(short|trade), "
                    "log P(flat)] + capped(residual_scale*delta_logits)"
                )
            ),
            "public_output": "direction_logits",
            "residual_delta_logits": (
                "head_direction residual is retained for diagnostics but is public-softmax invariant"
                if bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE)
                else "head_direction remains trainable through public direction_logits"
            ),
            "runtime_rule_free": True,
        },
        "hierarchical_entry_heads": {
            "enabled": bool(enable_hierarchical_entry_heads),
            "selection_score": "expected_utility_side",
            "side_utility_scale_bps": max(1.0, float(ENTRY_AUX_PATH_SCALE_BPS)),
            "side_mae_scale_bps": max(1.0, float(ENTRY_AUX_MFE_SCALE_BPS)),
            "ctx_prior_adapter": {
                "enabled": bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
                "scale": float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
                "input": "ctx_cat_embeddings",
                "runtime_rule_free": True,
            },
            "heads": [
                "trade_vs_flat",
                "long_vs_short_given_trade",
                "side_path_utility_bps",
                "side_bad_path_probability",
                "side_expected_mae_bps",
                "side_valid_trade_probability",
            ],
            "side_validity": {
                "enabled": bool(enable_side_validity_head),
                "loss_weight": float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT),
                "min_utility_bps": float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS),
                "pos_weight_cap": float(ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP),
                "targets": {
                    "long_valid_trade": [
                        "y_long_path_utility_bps >= min_utility_bps",
                        "y_long_bad_path == 0",
                        "y_long_high_mae_low_mfe_early_failure == 0",
                    ],
                    "short_valid_trade": [
                        "y_short_path_utility_bps >= min_utility_bps",
                        "y_short_bad_path == 0",
                        "y_short_high_mae_low_mfe_early_failure == 0",
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
                    or float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT) > 0.0
                    or float(ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT) > 0.0
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
                "public_flat_consistency_weight": float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT),
                "public_flat_consistency_min_label_rate": float(
                    ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
                ),
                "slice_public_flat_consistency_weight": float(
                    ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT
                ),
                "slice_public_flat_consistency_min_label_rate": float(
                    ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
                ),
                "slice_public_flat_consistency_min_rows": int(
                    ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS
                ),
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
                "applies_to": (
                    ["side_logits", "public_side_logits"]
                    if bool(ENTRY_HIER_PUBLIC_SIDE_HEAD)
                    else ["side_logits"]
                ),
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
            "side_bad_path_target_augmentation": {
                "enabled": True,
                "short_bad_path_or_labels": [
                    "y_short_bad_path",
                    "y_support_retest_continuation",
                    "y_countertrend_short_trap",
                    "y_short_high_mae_low_mfe_early_failure",
                ],
                "long_bad_path_or_labels": [
                    "y_long_bad_path",
                    "y_resistance_retest_continuation",
                    "y_countertrend_long_trap",
                    "y_long_high_mae_low_mfe_early_failure",
                ],
                "runtime_rule_free": True,
            },
            "pocket_abstention": {
                "enabled": float(ENTRY_HIER_POCKET_ABSTAIN_WEIGHT) > 0.0,
                "abstain_weight": float(ENTRY_HIER_POCKET_ABSTAIN_WEIGHT),
                "side_margin_weight": float(ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT),
                "utility_margin_bps": float(ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS),
                "no_trade_labels": [
                    "rising_channel_support_touch_without_valid_long_continuation",
                    "falling_channel_resistance_touch_without_valid_short_continuation",
                    "long_high_mae_low_mfe_early_failure_without_valid_short_target",
                    "short_high_mae_low_mfe_early_failure_without_valid_long_target",
                ],
                "runtime_rule_free": True,
            },
        },
        "trendline_rail_head": {
            "enabled": bool(enable_trendline_rail_head),
            "output_dim": 6 if bool(enable_trendline_rail_head) else 4,
            "labels": [
                "y_rising_channel_support_touch",
                "y_falling_channel_resistance_touch",
                "y_countertrend_short_trap",
                "y_countertrend_long_trap",
                "y_short_high_mae_low_mfe_early_failure",
                "y_long_high_mae_low_mfe_early_failure",
            ],
            "aux_weight": float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT),
            "wrong_side_weight": float(ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT),
            "rising_wrong_short_weight": float(ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT),
            "falling_wrong_long_weight": float(ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT),
            "final_margin_weight": float(ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT),
            "hier_margin_weight": float(ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT),
            "flat_trade_weight": float(ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT),
            "utility_margin_weight": float(ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT),
            "margin": float(ENTRY_TRENDLINE_RAIL_MARGIN),
            "utility_margin_bps": float(ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS),
            "runtime_rule_free": True,
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
        "residual_side_bias_alpha": float(ENTRY_RESIDUAL_SIDE_BIAS_ALPHA),
        "direction_ce_scale": float(ENTRY_DIRECTION_CE_SCALE),
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
        "direction_hierarchical_composition": bool(enable_hierarchical_direction_composition),
        "hier_compose_residual_logit_cap": float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
        "hier_compose_residual_side_neutral": bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
        "hier_compose_public_flat_from_trade": bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE),
        "hier_public_direction_composition": str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
        "hier_public_trade_head": bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
        "hier_public_side_head": bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
        "hier_ctx_prior_adapter": bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
        "hier_ctx_prior_adapter_scale": float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
        "hier_ctx_direction_calibration": bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
        "hier_ctx_direction_calibration_scale": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE),
        "hier_ctx_direction_calibration_cap": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
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
        "hier_public_flat_consistency_weight": float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT),
        "hier_public_flat_consistency_min_label_rate": float(
            ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
        ),
        "hier_slice_public_flat_consistency_weight": float(
            ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT
        ),
        "hier_slice_public_flat_consistency_min_label_rate": float(
            ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
        ),
        "hier_slice_public_flat_consistency_min_rows": int(
            ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS
        ),
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
                enable_mtf_direction_head and float(ENTRY_MTF_DIR_AUX_WEIGHT) > 0.0
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
            "direction_hierarchical_composition": bool(enable_hierarchical_direction_composition),
            "hier_compose_residual_logit_cap": float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
            "hier_compose_residual_side_neutral": bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
            "hier_compose_public_flat_from_trade": bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE),
            "hier_public_direction_composition": str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
            "hier_public_trade_head": bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
            "hier_public_side_head": bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
            "hier_ctx_prior_adapter": bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
            "hier_ctx_prior_adapter_scale": float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
            "hier_ctx_direction_calibration": bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
            "hier_ctx_direction_calibration_scale": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE),
            "hier_ctx_direction_calibration_cap": float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
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
            "hier_public_flat_consistency_weight": float(ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_WEIGHT),
            "hier_public_flat_consistency_min_label_rate": float(
                ENTRY_HIER_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
            ),
            "hier_slice_public_flat_consistency_weight": float(
                ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_WEIGHT
            ),
            "hier_slice_public_flat_consistency_min_label_rate": float(
                ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_LABEL_RATE
            ),
            "hier_slice_public_flat_consistency_min_rows": int(
                ENTRY_HIER_SLICE_PUBLIC_FLAT_CONSISTENCY_MIN_ROWS
            ),
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
            "residual_scale": float(ENTRY_RESIDUAL_SCALE),
            "anchor_gate_enabled": bool(enable_anchor_gate),
            "anchor_gate_init": float(anchor_gate_init),
            "hierarchical_entry_heads_enabled": bool(enable_hierarchical_entry_heads),
            "side_validity_head_enabled": bool(enable_side_validity_head),
            "trendline_rail_head_enabled": bool(enable_trendline_rail_head),
            "trendline_rail_output_dim": 6 if bool(enable_trendline_rail_head) else 4,
            "trendline_rail_aux_weight": float(ENTRY_TRENDLINE_RAIL_AUX_WEIGHT),
            "trendline_rail_wrong_side_weight": float(ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT),
            "trendline_rail_rising_wrong_short_weight": float(ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT),
            "trendline_rail_falling_wrong_long_weight": float(ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT),
            "trendline_rail_final_margin_weight": float(ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT),
            "trendline_rail_hier_margin_weight": float(ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT),
            "trendline_rail_flat_trade_weight": float(ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT),
            "trendline_rail_utility_margin_weight": float(ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT),
            "trendline_rail_margin": float(ENTRY_TRENDLINE_RAIL_MARGIN),
            "trendline_rail_utility_margin_bps": float(ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS),
            "hier_legacy_ce_mult": float(ENTRY_HIER_LEGACY_CE_MULT),
            "hier_trade_weight": float(ENTRY_HIER_TRADE_WEIGHT),
            "hier_side_weight": float(ENTRY_HIER_SIDE_WEIGHT),
            "hier_utility_weight": float(ENTRY_HIER_UTILITY_WEIGHT),
            "hier_bad_path_weight": float(ENTRY_HIER_BAD_PATH_WEIGHT),
            "hier_mae_weight": float(ENTRY_HIER_MAE_WEIGHT),
            "hier_side_validity_weight": float(ENTRY_HIER_SIDE_VALIDITY_WEIGHT),
            "hier_side_validity_min_utility_bps": float(ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS),
            "hier_side_validity_pos_weight_cap": float(ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP),
            "hier_pocket_abstain_weight": float(ENTRY_HIER_POCKET_ABSTAIN_WEIGHT),
            "hier_pocket_side_margin_weight": float(ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT),
            "hier_pocket_utility_margin_bps": float(ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS),
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
                "path/rank, specialist gate, V3+ aux heads, MTF-direction, and dip/forecast terms. "
                "Standalone --eval remains direction-only and declares that scope separately."
            ),
            "aux_selector_mode": "long_short_union" if ENTRY_SYMMETRIC_NEGATIVES else "long_only",
            "clean_edge_target_mode": "bidir" if ENTRY_SYMMETRIC_NEGATIVES else "long",
            "survival_target_mode": "bidir" if ENTRY_SYMMETRIC_NEGATIVES else "long",
            "bad_path_ce_in_direction_loss": bool(ENTRY_BAD_PATH_CE_MULTIPLIER > 1.0),
            "bad_path_prob_penalty_in_validation": bool(ENTRY_BAD_PATH_PROB_PENALTY > 0.0),
            "symmetric_short_prob_penalties": bool(ENTRY_SYMMETRIC_NEGATIVES),
            "symmetric_clean_edge_rank": bool(ENTRY_SYMMETRIC_NEGATIVES),
            "teacher_v6_mined": bool(ENTRY_CLEAN_EDGE_RANKING_WEIGHT > 0.0),  # A5: honest (dead targets ⇒ off)
            "aux_regression_positive_only": True,
            "path_quality_rank_full_batch": bool(ENTRY_PATH_QUALITY_RANK_WEIGHT > 0.0),
            "active_heads": active_heads,
        },
        "lane_contract": {
            # A7 2026-06-06: model trained BIDIRECTIONAL (symmetric long/short); the stale
            # LONG_ONLY_PREMIUM label drove the live lane to suppress shorts. entry_admission_policy
            # left as-is this wave (governs admission ORDER only; consumers not yet re-verified).
            "direction_policy": "BIDIRECTIONAL_PREMIUM",
            "entry_admission_policy": "OVERLAP_LONG_REPLACES_OLDEST_OVERLAP_SHORT_WHEN_FULL",
            "max_open_trades": 10,
        },
        "parked_features": {
            "cost_sensitive_loss_enabled": bool(ENTRY_COST_SENSITIVE_ENABLED),
            "pred_balance_alpha": float(ENTRY_PRED_BALANCE_ALPHA),
            "residual_side_bias_alpha": float(ENTRY_RESIDUAL_SIDE_BIAS_ALPHA),
            "timing_loss_scale": float(ENTRY_TIMING_LOSS_SCALE),
            "aux_early_weight": float(ENTRY_AUX_EARLY_WEIGHT),
            "aux_quality_weight": float(ENTRY_AUX_QUALITY_WEIGHT),
            "aux_bad_path_weight": float(ENTRY_AUX_BAD_PATH_WEIGHT),
            "xgb_short_penalty_weight": float(XGB_SHORT_LONG_PENALTY),
            "tradable_pos_weight_cap": float(ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP),
        },
    }
    (out_bundle_dir / "bundle_metadata.json").write_text(json.dumps(meta, indent=2))

    # Post-export verify: strict load. Match aux-head flags so model2 has
    # the same parameters as the trained model (otherwise the v3+ head
    # weights would be flagged as unexpected_keys).
    model2 = EntryV10CtxHybridTransformer(
        seq_input_dim=seq_input_dim,
        snap_input_dim=snap_input_dim,
        seq_len=seq_len,
        ctx_cont_dim=ctx_cont_dim,
        ctx_cat_dim=ctx_cat_dim,
        enable_multi_tf=enable_multi_tf,
        m15_seq_dim=_mtf_feat_count,
        h1_seq_dim=_mtf_feat_count,
        h4_seq_dim=_mtf_feat_count,
        d1_seq_dim=_mtf_feat_count,
        m15_seq_len=_m15_len,
        h1_seq_len=multi_tf_seq_len,
        h4_seq_len=_h4_len,
        d1_seq_len=_d1_len,
        # V2: mirror M5 branch + dim from training-time model.
        enable_multi_tf_m5=_mtf_v2,
        m5_seq_dim=_mtf_feat_count if _mtf_v2 else 0,
        m5_seq_len=multi_tf_seq_len,
        enable_tf_agreement_head=enable_tf_agreement_head,
        enable_path_quality_variance_head=enable_path_quality_variance_head,
        enable_position_size_head=enable_position_size_head,
        enable_hold_horizon_head=enable_hold_horizon_head,
        # 2026-05-26: the post-export verify model MUST mirror the trained model's
        # new heads, else their state_dict keys read as "unexpected" and the verify
        # raises (caught in pre-train smoke). Keep in lockstep with run_train args.
        enable_dip_head=enable_dip_head,
        enable_forecast_head=enable_forecast_head,
        enable_cross_tf_attn=enable_cross_tf_attn,
        enable_timing_head=enable_timing_head,
        enable_tail_risk_head=enable_tail_risk_head,
        enable_vol_forecast_head=enable_vol_forecast_head,
        enable_anchor_gate=bool(enable_anchor_gate),
        anchor_gate_init=float(anchor_gate_init),
        enable_hierarchical_entry_heads=bool(enable_hierarchical_entry_heads),
        enable_hierarchical_direction_composition=bool(enable_hierarchical_direction_composition),
        hierarchical_composition_residual_logit_cap=float(ENTRY_HIER_COMPOSE_RESIDUAL_LOGIT_CAP),
        hierarchical_composition_residual_side_neutral=bool(ENTRY_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL),
        hierarchical_composition_public_flat_from_trade=bool(ENTRY_HIER_COMPOSE_PUBLIC_FLAT_FROM_TRADE),
        hierarchical_public_direction_composition=str(ENTRY_HIER_PUBLIC_DIRECTION_COMPOSITION),
        enable_hierarchical_public_trade_head=bool(ENTRY_HIER_PUBLIC_TRADE_HEAD),
        enable_hierarchical_public_side_head=bool(ENTRY_HIER_PUBLIC_SIDE_HEAD),
        enable_hierarchical_ctx_prior_adapter=bool(ENTRY_HIER_CTX_PRIOR_ADAPTER),
        hierarchical_ctx_prior_adapter_scale=float(ENTRY_HIER_CTX_PRIOR_ADAPTER_SCALE),
        enable_hierarchical_ctx_direction_calibration=bool(ENTRY_HIER_CTX_DIRECTION_CALIBRATION),
        hierarchical_ctx_direction_calibration_scale=float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_SCALE),
        hierarchical_ctx_direction_calibration_cap=float(ENTRY_HIER_CTX_DIRECTION_CALIBRATION_CAP),
        enable_side_validity_head=bool(enable_side_validity_head),
        enable_trendline_rail_head=bool(enable_trendline_rail_head),
        trendline_rail_output_dim=6 if bool(enable_trendline_rail_head) else 4,
        enable_mtf_direction_head=enable_mtf_direction_head,
        mtf_dir_scale_init=mtf_dir_scale_init,
        enable_specialist_fusion=bool(enable_specialist_fusion),
        specialist_input_indices=specialist_indices,
        specialist_num_layers=int(specialist_num_layers),
        specialist_fusion_scale=float(specialist_fusion_scale),
        # 2026-06-03 BIG-9: regime_film.* are real params in state_dict -> verify model MUST
        # mirror the flag or strict-load reads them as unexpected_keys and raises.
        enable_regime_film=enable_regime_film,
        # 2026-06-02: mirror training-time tf_input_scale config — state_dict
        # contains tf_input_scale_* Parameters only when enabled, so the verify
        # model must match the flag exactly.
        enable_tf_input_scale=enable_tf_input_scale,
        tf_input_scale_init_m5=tf_input_scale_init_m5,
        tf_input_scale_init_m15=tf_input_scale_init_m15,
        tf_input_scale_init_h1=tf_input_scale_init_h1,
        tf_input_scale_init_h4=tf_input_scale_init_h4,
        tf_input_scale_init_d1=tf_input_scale_init_d1,
    )
    _load_entry_model_state_compat(model2, torch.load(model_path, map_location="cpu"), label="post_export_verify")
    model2.eval()
    with torch.no_grad():
        B = 2
        dummy_seq = torch.zeros(B, seq_len, seq_input_dim)
        dummy_snap = torch.zeros(B, snap_input_dim)
        dummy_cat = torch.zeros(B, ctx_cat_dim, dtype=torch.long)
        dummy_cont = torch.zeros(B, ctx_cont_dim)
        mtf_kwargs = {}
        if enable_multi_tf:
            mtf_kwargs = {
                "seq_m15": torch.zeros(B, multi_tf_seq_len, _mtf_feat_count),
                "seq_h1": torch.zeros(B, multi_tf_seq_len, _mtf_feat_count),
                "seq_h4": torch.zeros(B, _h4_len, _mtf_feat_count),
                "seq_d1": torch.zeros(B, _d1_len, _mtf_feat_count),
            }
            if _mtf_v2:
                mtf_kwargs["seq_m5"] = torch.zeros(B, multi_tf_seq_len, _mtf_feat_count)
        _ = model2(dummy_seq, dummy_snap, ctx_cat=dummy_cat, ctx_cont=dummy_cont, **mtf_kwargs)
    log.info(f"[DONE] Bundle OK strict load verified: {out_bundle_dir}")

    # Bundle load proof via runtime loader (strict)
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
    _ = load_entry_v10_ctx_bundle(
        bundle_dir=out_bundle_dir,
        device="cpu",
        xgb_models=None,
    )

    # ── ALWAYS-RUN feature-liveness audit (user vedtak 2026-06-06: NOTHING ignored, rule 9) ──────
    # Fail LOUD if an input feature went silently dead (off the allowlist) or the multi-TF block broke.
    # Two correctness requirements (learned 2026-06-06 from a false-fail):
    #   (1) use the FULL 123 ctx_cont names (ordered_ctx_cont_names may be the truncated base list →
    #       unnamed indices can't be allowlist-matched → false-flag allowlisted feats like vol_pct_*1yr);
    #   (2) sample from TRAIN (2020-2025 = broad period). The val window alone (6 months) is too narrow:
    #       slow-varying D1 regime feats (trend-age/regime-change) look const there → false-fail.
    # FeatureLivenessError FAILS the retrain (a silently-ignored input is a regression); a transient
    # load issue only warns.
    try:
        from gx1.audit.feature_liveness import assert_v10_batch_liveness, FeatureLivenessError
        try:
            from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import ORDERED_CTX_CONT_NAMES_V3 as _FULL_CC
            _live_cc = list(_FULL_CC) if len(_FULL_CC) == int(ctx_cont_dim) else list(ordered_ctx_cont_names)
        except Exception:
            _live_cc = list(ordered_ctx_cont_names)
        _live_ds = train_ds if len(train_ds) > 0 else val_ds  # broad period so slow-varying feats vary
        _snap_names = list(getattr(_live_ds, "signal_names", _default_signal_names(seq_input_dim)))
        if bool(getattr(_live_ds, "_advanced", False)) and hasattr(_live_ds, "_np_ctx_cont") and hasattr(_live_ds, "_np_snap"):
            _ab = {
                "ctx_cont": np.asarray(_live_ds._np_ctx_cont, dtype=np.float32),
                "snap_x": np.asarray(_live_ds._np_snap, dtype=np.float32),
            }
            if getattr(_live_ds, "_multi_tf_feats", None):
                for _tf, _feats in _live_ds._multi_tf_feats.items():
                    _arr = np.asarray(_feats.attrs.get("feats_np"), dtype=np.float32)
                    if _arr.size:
                        _ab[f"seq_{str(_tf).lower()}"] = _arr.reshape(1, _arr.shape[0], _arr.shape[1])
            log.info(
                "[FEATURE_LIVENESS] using full advanced arrays rows=%d snap_dim=%d ctx_cont_dim=%d",
                int(_ab["ctx_cont"].shape[0]),
                int(_ab["snap_x"].shape[1]),
                int(_ab["ctx_cont"].shape[1]),
            )
        else:
            _ab = next(iter(DataLoader(_live_ds, batch_size=min(8192, len(_live_ds)),
                                       shuffle=True, num_workers=2)))
        if bool(getattr(_live_ds, "neutral_xgb_bridge", False)):
            _ab = dict(_ab)
            _ab["snap_x"] = _ab["snap_x"][:, 7:]
            _snap_names = _snap_names[7:]
            log.info("[FEATURE_LIVENESS] neutral XGB bridge detected; skipping first 7 compat bridge slots")
        assert_v10_batch_liveness(_ab, ctx_cont_names=_live_cc,
                                  snap_names=_snap_names, raise_on_fail=True)
        log.info("[FEATURE_LIVENESS] post-export audit OK — nothing ignored (all inputs alive/allowlisted)")
    except FeatureLivenessError:
        raise
    except Exception as _e:  # transient dataset/load issue must not fail a good retrain
        log.warning("[FEATURE_LIVENESS] audit skipped (non-fatal): %r", _e)


def run_eval(
    bundle_dir: Path,
    train_parquet: Optional[Path],
    val_parquet: Optional[Path],
    test_parquet: Path,
    m5_prebuilt_path: Path,
    seq_len: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    gx1_data_override: str,
) -> None:
    """
    Deterministic eval of an existing bundle on a test parquet.
    No bundle mutation; writes EVAL_TEST.json alongside the bundle.
    """
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_seed(seed)
    set_thread_limits(1)

    bd = Path(bundle_dir).expanduser()
    if not bd.is_absolute():
        gx1_data = _resolve_gx1_data(gx1_data_override)
        bd = (gx1_data / bd).resolve()
    _require(bd.is_dir(), f"[ENTRY_V10_CTX_BUNDLE_DIR_MISSING] {bd}")

    model_path = bd / "model_state_dict.pt"
    meta_path = bd / "bundle_metadata.json"
    _require(model_path.exists(), f"[ENTRY_V10_CTX_MODEL_MISSING] {model_path}")
    _require(meta_path.exists(), f"[ENTRY_V10_CTX_META_MISSING] {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _require(meta.get("signal_bridge_id") == "XGB_SIGNAL_BRIDGE_V1", "[EVAL_CONTRACT_BRIDGE]")
    # R4: the ctx_tag self-describes the bundle's own ctx_cat_dim. Validate
    # self-consistency, not a frozen literal. ctx_cat_dim is read below.
    _meta_cat_dim = int(meta.get("ctx_cat_dim") or -1)
    _require(meta.get("ctx_tag") == f"CTX6CAT{_meta_cat_dim}", "[EVAL_CONTRACT_CTX_TAG]")
    ctx_cont_dim = int(meta.get("ctx_cont_dim") or -1)
    ctx_cat_dim = int(meta.get("ctx_cat_dim") or -1)
    seq_input_dim = int(meta.get("seq_input_dim") or SEQ_SIGNAL_DIM)
    snap_input_dim = int(meta.get("snap_input_dim") or seq_input_dim)
    _require(ctx_cont_dim >= 6, "[EVAL_CONTRACT_CTX_CONT]")
    _require(ctx_cat_dim == _expected_ctx_cat_dim(), "[EVAL_CONTRACT_CTX_CAT]")
    _require(seq_input_dim == snap_input_dim and seq_input_dim > 0, f"[EVAL_CONTRACT_SIGNAL_DIM] seq={seq_input_dim} snap={snap_input_dim}")

    state_dict_sha = _sha256_file(model_path)

    # Eval must construct the exact bundle architecture (multi-TF V2, cross-TF
    # attention, new aux heads, optional scale params). Reusing the runtime bundle
    # loader prevents --eval from drifting behind training/export.
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
    bundle = load_entry_v10_ctx_bundle(bundle_dir=bd, device=str(device), xgb_models=None)
    model = bundle.transformer_model

    mtf_meta = (meta.get("multi_tf") or {}) if isinstance(meta, dict) else {}
    eval_dataset_kwargs: Dict[str, Any] = {}
    if bool(mtf_meta.get("enabled", False)):
        m5_path = Path(m5_prebuilt_path).expanduser()
        _require(m5_path.is_file(), f"[EVAL_M5_PREBUILT_MISSING] {m5_path}")
        mtf_seq_len = int(mtf_meta.get("m15_seq_len", 96))
        eval_dataset_kwargs = {
            "enable_multi_tf": True,
            "m5_prebuilt_path": m5_path,
            "multi_tf_seq_len": mtf_seq_len,
            "multi_tf_closed_bar": bool(float(mtf_meta.get("target_availability_shift_minutes", 0.0) or 0.0) > 0.0),
            "per_tf_seq_lens": {
                "M5": int(mtf_meta.get("m5_seq_len", mtf_seq_len)),
                "M15": int(mtf_meta.get("m15_seq_len", mtf_seq_len)),
                "H1": int(mtf_meta.get("h1_seq_len", mtf_seq_len)),
                "H4": int(mtf_meta.get("h4_seq_len", mtf_seq_len)),
                "D1": int(mtf_meta.get("d1_seq_len", mtf_seq_len)),
            },
        }

    dataset = EntryV10CtxDataset(
        parquet_path=test_parquet,
        seq_len=seq_len,
        allow_constant_labels=True,
        **eval_dataset_kwargs,
    )
    _require(len(dataset) > 0, "[EVAL_NO_SAMPLES]")
    sample = dataset[0]
    _require(
        sample["seq_x"].shape[0] == seq_len,
        f"[EVAL_SEQ_LEN_MISMATCH] dataset seq_len {sample['seq_x'].shape[0]} != {seq_len}",
    )
    _require(
        sample["seq_x"].shape[1] == seq_input_dim
        and sample["snap_x"].shape[0] == snap_input_dim
        and sample["ctx_cont"].shape[0] == ctx_cont_dim
        and sample["ctx_cat"].shape[0] == ctx_cat_dim,
        f"[EVAL_CONTRACT_MISMATCH] expected signal={seq_input_dim} ctx_cont={ctx_cont_dim} ctx_cat={ctx_cat_dim}",
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    meta_cost = meta.get("cost_matrix") or {}
    meta_cw = meta.get("class_weights") or {}
    cost_enabled = bool(meta.get("cost_sensitive_loss_enabled", ENTRY_COST_SENSITIVE_ENABLED))
    cost_scale = float(meta.get("cost_sensitive_loss_scale", ENTRY_COST_SENSITIVE_SCALE))
    _require_nonneg("EVAL_COST_SENSITIVE_SCALE", cost_scale)
    balance_alpha = float(meta.get("pred_balance_alpha", ENTRY_PRED_BALANCE_ALPHA))
    balance_target = str(meta.get("pred_balance_target", ENTRY_PRED_BALANCE_TARGET)).strip().lower()
    raw_balance_weights = meta.get("pred_balance_class_weights", ENTRY_PRED_BALANCE_CLASS_WEIGHTS)
    if isinstance(raw_balance_weights, str):
        balance_class_weights_values = _parse_three_float_value(
            "EVAL_PRED_BALANCE_CLASS_WEIGHTS",
            raw_balance_weights,
        )
    else:
        balance_class_weights_values = tuple(float(value) for value in raw_balance_weights)
        if len(balance_class_weights_values) != 3 or any(
            (not np.isfinite(value)) or value <= 0.0 for value in balance_class_weights_values
        ):
            raise RuntimeError(
                "[EVAL_PRED_BALANCE_CLASS_WEIGHTS_INVALID] expected three finite positive weights"
            )
    _require_nonneg("EVAL_PRED_BALANCE_ALPHA", balance_alpha)
    residual_side_bias_alpha = float(
        meta.get("residual_side_bias_alpha", ENTRY_RESIDUAL_SIDE_BIAS_ALPHA)
    )
    _require_nonneg("EVAL_RESIDUAL_SIDE_BIAS_ALPHA", residual_side_bias_alpha)
    if balance_target not in ("label", "uniform"):
        raise RuntimeError(
            f"[EVAL_BALANCE_TARGET_INVALID] pred_balance_target={balance_target!r} expected 'label' or 'uniform'"
        )

    cw_long = float(meta_cw.get("long", 1.0))
    cw_short = float(meta_cw.get("short", SHORT_CLASS_WEIGHT))
    cw_flat = float(meta_cw.get("flat", 1.0))
    class_weights = torch.tensor([cw_long, cw_short, cw_flat], device=device)
    pred_balance_class_weights = torch.tensor(
        [float(value) for value in balance_class_weights_values],
        device=device,
    )

    cost_long_to_short = float(meta_cost.get("long_to_short", ENTRY_COST_LONG_TO_SHORT))
    cost_long_to_flat = float(meta_cost.get("long_to_flat", ENTRY_COST_LONG_TO_FLAT))
    cost_short_to_long = float(meta_cost.get("short_to_long", ENTRY_COST_SHORT_TO_LONG))
    cost_short_to_flat = float(meta_cost.get("short_to_flat", ENTRY_COST_SHORT_TO_FLAT))
    cost_flat_to_long = float(meta_cost.get("flat_to_long", ENTRY_COST_FLAT_TO_LONG))
    cost_flat_to_short = float(meta_cost.get("flat_to_short", ENTRY_COST_FLAT_TO_SHORT))
    _require_nonneg("EVAL_COST_LONG_TO_SHORT", cost_long_to_short)
    _require_nonneg("EVAL_COST_LONG_TO_FLAT", cost_long_to_flat)
    _require_nonneg("EVAL_COST_SHORT_TO_LONG", cost_short_to_long)
    _require_nonneg("EVAL_COST_SHORT_TO_FLAT", cost_short_to_flat)
    _require_nonneg("EVAL_COST_FLAT_TO_LONG", cost_flat_to_long)
    _require_nonneg("EVAL_COST_FLAT_TO_SHORT", cost_flat_to_short)

    criterion, _ = _build_cost_sensitive_criterion(
        device=device,
        class_weights=class_weights,
        cost_long_to_short=cost_long_to_short,
        cost_long_to_flat=cost_long_to_flat,
        cost_short_to_long=cost_short_to_long,
        cost_short_to_flat=cost_short_to_flat,
        cost_flat_to_long=cost_flat_to_long,
        cost_flat_to_short=cost_flat_to_short,
        cost_scale=cost_scale,
        enabled=cost_enabled,
        balance_alpha=balance_alpha,
        balance_target=balance_target,
        balance_class_weights=pred_balance_class_weights,
    )
    test_loss, test_auc, test_acc = _validate_eval(
        model,
        loader,
        criterion,
        device,
        residual_side_bias_alpha=residual_side_bias_alpha,
    )

    eval_artifact = {
        "created_at_utc": _utc_now(),
        "bundle_dir": str(bd),
        "bundle_state_dict_sha256": state_dict_sha,
        "test_parquet": str(test_parquet),
        "test_parquet_sha256": _sha256_file(Path(test_parquet)),
        "seq_len": seq_len,
        "seq_input_dim": seq_input_dim,
        "snap_input_dim": snap_input_dim,
        "batch_size": batch_size,
        "device": str(device),
        "seed": seed,
        "test_loss": test_loss,
        "test_loss_scope": "direction_only_ce_plus_residual_side_bias",
        "validation_objective_matches_train": False,
        "train_objective_metrics_included": False,
        "hierarchical_loss_metrics_included": False,
        "eval_scope_note": (
            "Standalone --eval uses the frozen direction-only evaluator for deterministic "
            "bundle checks. Use train/val or selective-edge/replay artifacts for the full "
            "hierarchical trade/side/utility objective."
        ),
        "test_auc": test_auc,
        "test_auc_status": "DISABLED",
        "test_acc": test_acc,
        "n_test_samples": len(dataset),
    }
    eval_path = bd / "EVAL_TEST.json"
    eval_path.write_text(json.dumps(eval_artifact, indent=2), encoding="utf-8")
    auc_display = "DISABLED" if not np.isfinite(test_auc) else f"{test_auc:.4f}"
    log.info(
        f"[EVAL_DONE] {eval_path} loss={test_loss:.6f} auc={auc_display} acc={test_acc:.4f}"
    )

    _run_entry_training_bias_audit(
        bundle_dir=bd,
        model=model,
        device=device,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        test_parquet=test_parquet,
        dataset_kwargs=eval_dataset_kwargs,
    )


def _mean_median(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def _init_confusion_bucket() -> Dict[str, Dict[str, List[float]]]:
    return {
        "label_SHORT__pred_LONG": {"p_long": [], "p_short": [], "p_flat": []},
        "label_SHORT__pred_SHORT": {"p_long": [], "p_short": [], "p_flat": []},
        "label_LONG__pred_LONG": {"p_long": [], "p_short": [], "p_flat": []},
        "label_LONG__pred_SHORT": {"p_long": [], "p_short": [], "p_flat": []},
    }


def _finalize_confusion(conf: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    out = {}
    for key, vals in conf.items():
        out[key] = {
            "p_long": _mean_median(vals["p_long"]),
            "p_short": _mean_median(vals["p_short"]),
            "p_flat": _mean_median(vals["p_flat"]),
            "count": len(vals["p_long"]),
        }
    return out


def _compute_bias_stats(
    model: nn.Module,
    dataset: EntryV10CtxDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    session_map = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
    conf_global = _init_confusion_bucket()
    conf_by_session: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    totals_by_session: Dict[str, int] = {}
    total_samples = 0
    total_label_long_short = 0
    label_long_total = 0
    label_long_pred_long = 0
    label_long_pred_short = 0
    label_long_pred_flat = 0
    label_long_margin_ge_000 = 0
    label_long_margin_ge_002 = 0
    label_long_margin_ge_005 = 0

    label_long_by_session: Dict[str, Dict[str, int]] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            ctx_cont = batch["ctx_cont"].to(device)
            ctx_cat = batch["ctx_cat"].to(device)
            y = batch["y"].to(device)

            out = _autocast_forward(model, seq_x.device, seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **_multi_tf_kwargs_from_batch(batch, seq_x.device))
            logits = out["direction_logits"]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            p_long = probs[:, 0].cpu().numpy()
            p_short = probs[:, 1].cpu().numpy()
            p_flat = probs[:, 2].cpu().numpy()
            labels = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            sessions = ctx_cat[:, 0].cpu().numpy()

            total_samples += len(labels)

            for i in range(len(labels)):
                label = int(labels[i])
                pred = int(preds_np[i])
                sess_id = int(sessions[i]) if sessions is not None else -1
                sess_name = session_map.get(sess_id, f"UNKNOWN_{sess_id}")

                totals_by_session[sess_name] = totals_by_session.get(sess_name, 0) + 1
                if sess_name not in conf_by_session:
                    conf_by_session[sess_name] = _init_confusion_bucket()
                    label_long_by_session[sess_name] = {
                        "total": 0,
                        "pred_long": 0,
                        "pred_short": 0,
                        "pred_flat": 0,
                        "margin_ge_000": 0,
                        "margin_ge_002": 0,
                        "margin_ge_005": 0,
                    }

                if label in (0, 1):
                    total_label_long_short += 1
                    if label == 0:
                        label_long_total += 1
                        label_long_by_session[sess_name]["total"] += 1
                        if pred == 0:
                            label_long_pred_long += 1
                            label_long_by_session[sess_name]["pred_long"] += 1
                        if pred == 1:
                            label_long_pred_short += 1
                            label_long_by_session[sess_name]["pred_short"] += 1
                        if pred == 2:
                            label_long_pred_flat += 1
                            label_long_by_session[sess_name]["pred_flat"] += 1
                        margin = float(p_long[i] - p_flat[i])
                        if margin >= 0.00:
                            label_long_margin_ge_000 += 1
                            label_long_by_session[sess_name]["margin_ge_000"] += 1
                        if margin >= 0.02:
                            label_long_margin_ge_002 += 1
                            label_long_by_session[sess_name]["margin_ge_002"] += 1
                        if margin >= 0.05:
                            label_long_margin_ge_005 += 1
                            label_long_by_session[sess_name]["margin_ge_005"] += 1

                    if label == 1 and pred == 0:
                        key = "label_SHORT__pred_LONG"
                    elif label == 1 and pred == 1:
                        key = "label_SHORT__pred_SHORT"
                    elif label == 0 and pred == 0:
                        key = "label_LONG__pred_LONG"
                    elif label == 0 and pred == 1:
                        key = "label_LONG__pred_SHORT"
                    else:
                        key = None

                    if key:
                        conf_global[key]["p_long"].append(float(p_long[i]))
                        conf_global[key]["p_short"].append(float(p_short[i]))
                        conf_global[key]["p_flat"].append(float(p_flat[i]))
                        conf_by_session[sess_name][key]["p_long"].append(float(p_long[i]))
                        conf_by_session[sess_name][key]["p_short"].append(float(p_short[i]))
                        conf_by_session[sess_name][key]["p_flat"].append(float(p_flat[i]))

    confusion_counts = {k: len(v["p_long"]) for k, v in conf_global.items()}
    confusion_rates = {
        k: (count / total_label_long_short if total_label_long_short > 0 else 0.0)
        for k, count in confusion_counts.items()
    }

    session_stats = {}
    for sess_name, conf in conf_by_session.items():
        session_label_long = label_long_by_session.get(sess_name, {})
        session_total = totals_by_session.get(sess_name, 0)
        session_conf_counts = {k: len(v["p_long"]) for k, v in conf.items()}
        session_conf_rates = {
            k: (count / sum(session_conf_counts.values()) if sum(session_conf_counts.values()) > 0 else 0.0)
            for k, count in session_conf_counts.items()
        }
        long_total = session_label_long.get("total", 0)
        pred_long = session_label_long.get("pred_long", 0)
        pred_short = session_label_long.get("pred_short", 0)
        pred_flat = session_label_long.get("pred_flat", 0)
        session_stats[sess_name] = {
            "total_samples": session_total,
            "confusion_counts": session_conf_counts,
            "confusion_rates": session_conf_rates,
            "prob_stats": _finalize_confusion(conf),
            "label_long": {
                "total": long_total,
                "pred_long_count": pred_long,
                "pred_short_count": pred_short,
                "pred_flat_count": pred_flat,
                "pred_long_rate": (pred_long / long_total if long_total > 0 else 0.0),
                "pred_short_rate": (pred_short / long_total if long_total > 0 else 0.0),
                "pred_flat_rate": (pred_flat / long_total if long_total > 0 else 0.0),
                "p_long_minus_p_flat_ge_0.00_count": session_label_long.get("margin_ge_000", 0),
                "p_long_minus_p_flat_ge_0.00_rate": (
                    session_label_long.get("margin_ge_000", 0) / long_total if long_total > 0 else 0.0
                ),
                "p_long_minus_p_flat_ge_0.02_count": session_label_long.get("margin_ge_002", 0),
                "p_long_minus_p_flat_ge_0.02_rate": (
                    session_label_long.get("margin_ge_002", 0) / long_total if long_total > 0 else 0.0
                ),
                "p_long_minus_p_flat_ge_0.05_count": session_label_long.get("margin_ge_005", 0),
                "p_long_minus_p_flat_ge_0.05_rate": (
                    session_label_long.get("margin_ge_005", 0) / long_total if long_total > 0 else 0.0
                ),
            },
        }

    return {
        "total_samples": total_samples,
        "label_long_short_total": total_label_long_short,
        "confusion_counts": confusion_counts,
        "confusion_rates": confusion_rates,
        "prob_stats": _finalize_confusion(conf_global),
        "label_long": {
            "total": label_long_total,
            "pred_long_count": label_long_pred_long,
            "pred_short_count": label_long_pred_short,
            "pred_flat_count": label_long_pred_flat,
            "pred_long_rate": (label_long_pred_long / label_long_total if label_long_total > 0 else 0.0),
            "pred_short_rate": (label_long_pred_short / label_long_total if label_long_total > 0 else 0.0),
            "pred_flat_rate": (label_long_pred_flat / label_long_total if label_long_total > 0 else 0.0),
            "p_long_minus_p_flat_ge_0.00_count": label_long_margin_ge_000,
            "p_long_minus_p_flat_ge_0.00_rate": (
                label_long_margin_ge_000 / label_long_total if label_long_total > 0 else 0.0
            ),
            "p_long_minus_p_flat_ge_0.02_count": label_long_margin_ge_002,
            "p_long_minus_p_flat_ge_0.02_rate": (
                label_long_margin_ge_002 / label_long_total if label_long_total > 0 else 0.0
            ),
            "p_long_minus_p_flat_ge_0.05_count": label_long_margin_ge_005,
            "p_long_minus_p_flat_ge_0.05_rate": (
                label_long_margin_ge_005 / label_long_total if label_long_total > 0 else 0.0
            ),
        },
        "sessions": session_stats,
    }


def _run_entry_training_bias_audit(
    bundle_dir: Path,
    model: nn.Module,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    train_parquet: Optional[Path],
    val_parquet: Optional[Path],
    test_parquet: Optional[Path],
    dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    splits = {
        "train": train_parquet,
        "val": val_parquet,
        "test": test_parquet,
    }

    results = {
        "created_at_utc": _utc_now(),
        "bundle_dir": str(bundle_dir),
        "splits": {},
    }

    for split_name, parquet_path in splits.items():
        if parquet_path is None:
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] split=%s status=skip reason=missing_parquet", split_name)
            continue
        if not Path(parquet_path).exists():
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] split=%s status=skip reason=missing_file path=%s", split_name, parquet_path)
            continue

        dataset = EntryV10CtxDataset(
            parquet_path=Path(parquet_path),
            seq_len=seq_len,
            allow_constant_labels=True,
            **(dataset_kwargs or {}),
        )
        if len(dataset) == 0:
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] split=%s status=skip reason=empty_dataset", split_name)
            continue

        stats = _compute_bias_stats(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        results["splits"][split_name] = stats

    audit_path = Path(bundle_dir) / "ENTRY_TRAINING_BIAS_AUDIT.json"
    audit_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    session_bias = {"created_at_utc": _utc_now(), "bundle_dir": str(bundle_dir), "splits": {}}
    for split_name, split_stats in results.get("splits", {}).items():
        split_sessions = split_stats.get("sessions", {})
        session_bias["splits"][split_name] = {}
        for session_name in ["ASIA", "EU", "OVERLAP", "US"]:
            sess_stats = split_sessions.get(session_name, {})
            label_long = sess_stats.get("label_long", {})
            session_bias["splits"][split_name][session_name] = {
                "label_long_total": label_long.get("total", 0),
                "pred_long_rate": label_long.get("pred_long_rate", 0.0),
                "pred_short_rate": label_long.get("pred_short_rate", 0.0),
                "pred_flat_rate": label_long.get("pred_flat_rate", 0.0),
                "margin_ge_0.00_rate": label_long.get("p_long_minus_p_flat_ge_0.00_rate", 0.0),
                "margin_ge_0.02_rate": label_long.get("p_long_minus_p_flat_ge_0.02_rate", 0.0),
                "margin_ge_0.05_rate": label_long.get("p_long_minus_p_flat_ge_0.05_rate", 0.0),
            }

    session_bias_path = Path(bundle_dir) / "ENTRY_SESSION_BIAS_AUDIT.json"
    session_bias_path.write_text(json.dumps(session_bias, indent=2), encoding="utf-8")

    log.info("[ENTRY_TRAINING_BIAS_AUDIT]")
    log.info("bundle_dir=%s", bundle_dir)
    log.info("splits=%s", json.dumps(list(results.get("splits", {}).keys())))
    for split_name, split_stats in results.get("splits", {}).items():
        long_stats = split_stats.get("label_long", {})
        log.info(
            "[ENTRY_TRAINING_BIAS_AUDIT] split=%s label_long_total=%s pred_long_rate=%.6f pred_flat_rate=%.6f margin_ge_0.00_rate=%.6f margin_ge_0.02_rate=%.6f margin_ge_0.05_rate=%.6f",
            split_name,
            long_stats.get("total"),
            float(long_stats.get("pred_long_rate") or 0.0),
            float(long_stats.get("pred_flat_rate") or 0.0),
            float(long_stats.get("p_long_minus_p_flat_ge_0.00_rate") or 0.0),
            float(long_stats.get("p_long_minus_p_flat_ge_0.02_rate") or 0.0),
            float(long_stats.get("p_long_minus_p_flat_ge_0.05_rate") or 0.0),
        )

    log.info("[ENTRY_SESSION_BIAS_AUDIT]")
    log.info("bundle_dir=%s", bundle_dir)
    for split_name, split_sessions in session_bias.get("splits", {}).items():
        for session_name, metrics in split_sessions.items():
            log.info(
                "[ENTRY_SESSION_BIAS_AUDIT] split=%s session=%s label_long_total=%s pred_long_rate=%.6f pred_short_rate=%.6f pred_flat_rate=%.6f margin_ge_0.00_rate=%.6f margin_ge_0.02_rate=%.6f margin_ge_0.05_rate=%.6f",
                split_name,
                session_name,
                metrics.get("label_long_total"),
                float(metrics.get("pred_long_rate") or 0.0),
                float(metrics.get("pred_short_rate") or 0.0),
                float(metrics.get("pred_flat_rate") or 0.0),
                float(metrics.get("margin_ge_0.00_rate") or 0.0),
                float(metrics.get("margin_ge_0.02_rate") or 0.0),
                float(metrics.get("margin_ge_0.05_rate") or 0.0),
            )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    _enforce_canonical_train_env_contract()

    parser = argparse.ArgumentParser("ENTRY_V10_CTX canonical trainer")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sanity", action="store_true", help="Run sanity check only and exit 0/1")
    mode.add_argument("--train", action="store_true", help="Run training")
    mode.add_argument("--eval", action="store_true", help="Run eval on test split")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10, help="Max epochs (used with early stopping)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN_V3)  # V2: 96 (was 30)
    parser.add_argument("--dataset_manifest", type=Path, default=None, help="Path to dataset .manifest.json (train parquet from output_data_path; val = same dir stem_val.parquet)")
    parser.add_argument("--dataset_dir", type=Path, default=None, help="Directory with *_train.parquet and *_val.parquet")
    parser.add_argument("--dataset_train_parquet", type=Path, default=None, help="Optional: explicit train parquet path when dataset_dir has multiple pairs")
    parser.add_argument("--out_bundle_dir", type=Path, required=False, help="Output bundle directory (under GX1_DATA if relative for train/sanity)")
    parser.add_argument("--bundle_dir", type=Path, default=None, help="Existing bundle directory for eval mode")
    parser.add_argument("--test_parquet", type=Path, default=None, help="Explicit test parquet path (optional)")
    parser.add_argument("--gx1-data", type=str, default="")
    parser.add_argument("--allow-constant-labels", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable faster (non-deterministic) training: cudnn benchmark on, deterministic off",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    # ── Multi-TF (V12.2) — MANDATORY, no toggle ──────────────────────
    # 2026-05-26: the --enable-multi-tf on/off flag was REMOVED. Multi-TF×5
    # (M5/M15/H1/H4/D1, V2 25-feat) is ALWAYS on — it must never be possible to
    # accidentally train a single-TF V10. --m5-prebuilt-path is therefore
    # required. (rule: multi_tf_always_mandatory)
    # ── V10 v3+ aux heads (Targets 1-4) ──────────────────────────────
    parser.add_argument(
        "--enable-tf-agreement-head", action="store_true",
        help="V10 v3+ Target 1: add multi-TF agreement aux head + MSE loss "
             "× 0.5 weight on y_tf_agreement_score label.",
    )
    parser.add_argument(
        "--enable-path-quality-variance-head", action="store_true",
        help="V10 v3+ Target 2: add log-variance head for path_quality; "
             "swap smooth_l1 loss for Gaussian NLL so model learns uncertainty.",
    )
    parser.add_argument(
        "--enable-position-size-head", action="store_true",
        help="V10 v3+ Target 3: add position-size aux head + MSE loss × 0.3 "
             "weight on y_position_size_target label.",
    )
    parser.add_argument(
        "--enable-hold-horizon-head", action="store_true",
        help="V10 v3+ Target 4: add hold-horizon aux head + MSE loss × 0.3 "
             "weight on y_hold_horizon_target label.",
    )
    parser.add_argument(
        "--enable-v10-v3plus-all-heads", action="store_true",
        help="Convenience flag: enable all 4 V10 v3+ aux heads (T1+T2+T3+T4).",
    )
    parser.add_argument(
        "--enable-pos-enc", action=argparse.BooleanOptionalAction, default=True,
        help="Sinusoidal positional encoding on the base seq + every per-TF "
             "sequence so the transformer uses temporal ORDER (default ON for "
             "new runs). Without it the encoder+mean-pool are permutation-"
             "invariant — the model is blind to bar order. --no-enable-pos-enc "
             "restores the old order-blind behaviour.",
    )
    parser.add_argument(
        "--enable-regime-film", action="store_true", default=False,
        help="BIG-9 (2026-06-03): FiLM regime-conditioning of a separate z_dir for the "
             "direction head (zero-init -> bit-parity at start). Default OFF; enable for the "
             "regime-robust retrain (pair with GX1_TREND_REGIME_FROM_D1=1 so the regime slot varies).",
    )
    parser.add_argument(
        "--enable-mtf-direction-head", action="store_true", default=False,
        help="Forceful MTF→direction (2026-06-06): a dedicated head reads the GATED cross-TF "
             "repr and adds a non-zeroable term to direction logits, + an aux CE forcing the 5 "
             "multi-TF streams to predict direction. Requires --enable-cross-tf-attn (default ON).",
    )
    parser.add_argument(
        "--mtf-dir-scale-init", type=float, default=0.2,
        help="Initial value of the learnable mtf_dir_scale (default 0.2, NON-zero so the multi-TF "
             "direction term contributes from epoch 0).",
    )
    parser.add_argument(
        "--enable-dip-head", action=argparse.BooleanOptionalAction, default=True,
        help="Distributional dip-analysis head (18: dir×K×{dip_p50,dip_p90,recovery_p50}, "
             "pinball loss vs mae_before_mfe/mfe). Default ON for the dip-aware rebuild.",
    )
    parser.add_argument(
        "--enable-forecast-head", action=argparse.BooleanOptionalAction, default=True,
        help="Self-supervised forecast head (4: cum future return @ K{1,5,12,24}). Default ON.",
    )
    parser.add_argument(
        "--enable-cross-tf-attn", action=argparse.BooleanOptionalAction, default=True,
        help="Cross-TF attention + learnable per-TF gates fusion (regime-dependent). Default ON.",
    )
    parser.add_argument(
        "--enable-timing-head", action=argparse.BooleanOptionalAction, default=True,
        help="Dip-timing head (12: dir×K×{dip_bottom_frac,time_to_mfe_frac}). WHEN the dip "
             "bottoms / favorable peak hits. Default ON for the dip-aware rebuild.",
    )
    parser.add_argument(
        "--enable-tail-risk-head", action=argparse.BooleanOptionalAction, default=True,
        help="Tail-risk head (6: dir×K, pinball q=0.9 of worst adverse over full horizon). Default ON.",
    )
    parser.add_argument(
        "--enable-vol-forecast-head", action=argparse.BooleanOptionalAction, default=True,
        help="Volatility-forecast head (3: forward realized vol bps @ K{12,48,96}). Default ON.",
    )
    parser.add_argument(
        "--enable-anchor-gate", action="store_true", default=False,
        help="XAU direction repair: learn a per-regime gate on signal-bridge anchor logits "
             "so geometry/MTF can suppress a wrong anchor instead of only adding residuals.",
    )
    parser.add_argument(
        "--anchor-gate-init", type=float, default=1.0,
        help="Initial anchor-gate value when --enable-anchor-gate is active. 1.0 starts "
             "near legacy anchored behavior; lower values train a more anchor-light challenger.",
    )
    parser.add_argument(
        "--enable-hierarchical-entry-heads", action="store_true", default=False,
        help="XAU direction repair: add trade-vs-flat, conditional side, side utility, "
             "side bad-path and side MAE heads.",
    )
    parser.add_argument(
        "--enable-side-validity-head", action="store_true", default=False,
        help="XAU direction repair: add learned long/short valid-trade logits so broad "
             "touch/failure setups can be learned as no-trade instead of forced side flips.",
    )
    parser.add_argument(
        "--enable-trendline-rail-head", action="store_true", default=False,
        help="XAU direction repair: add supervised trendline/rail pocket head for rising support, "
             "falling resistance and countertrend trap labels.",
    )
    parser.add_argument(
        "--enable-xau-direction-repair-heads", action="store_true", default=False,
        help="Convenience flag: enable anchor-gate, hierarchical entry heads and trendline rail head.",
    )
    parser.add_argument(
        "--enable-specialist-fusion", action=argparse.BooleanOptionalAction, default=False,
        help="Enable audited seq146 specialist feature-family encoders and gated fusion. "
             "Default OFF for legacy bundle compatibility.",
    )
    parser.add_argument(
        "--specialist-audit-json", type=Path, default=DEFAULT_SPECIALIST_AUDIT_JSON,
        help="PASS specialist feature-group audit JSON that supplies specialist input indices.",
    )
    parser.add_argument(
        "--specialist-contract-mode",
        choices=SPECIALIST_CONTRACT_MODES,
        default="foundation_seq146",
        help="Specialist contract mode expected by the trainer loader.",
    )
    parser.add_argument(
        "--specialist-num-layers", type=int, default=1,
        help="TransformerEncoder layers per specialist branch when --enable-specialist-fusion is on.",
    )
    parser.add_argument(
        "--specialist-fusion-scale", type=float, default=0.25,
        help="Residual correction scale for specialist-fusion output.",
    )
    parser.add_argument(
        "--m5-prebuilt-path", type=Path, required=True,
        help="REQUIRED: path to canonical_v3 M5 OHLC parquet (used by dataset to "
             "resample the M5/M15/H1/H4/D1 multi-TF features). Multi-TF is mandatory.",
    )
    parser.add_argument(
        "--multi-tf-seq-len", type=int, default=96,
        help="V12.2: number of bars per TF (default: 96 → ~4d H1, ~16d H4, ~3mo D1). "
             "Per-TF overrides via --per-tf-seq-len-{h4,d1}.",
    )
    parser.add_argument(
        "--per-tf-seq-len-h4", type=int, default=0,
        help="V2: override H4 multi-TF seq_len (default 0 = use --multi-tf-seq-len). "
             "Recommended 48 — H4@96 = 16d which is overkill, 48 = 8d.",
    )
    parser.add_argument(
        "--per-tf-seq-len-d1", type=int, default=0,
        help="V2: override D1 multi-TF seq_len (default 0 = use --multi-tf-seq-len). "
             "Recommended 30 — D1@96 = 3mo which is overkill, 30 = 1mo.",
    )
    parser.add_argument(
        "--prelim-no-aux-heads", action="store_true",
        help="V2 prelim mode: disable V10 v3+ aux heads (tradable/clean_edge/survival/etc) "
             "during prelim training. Speeds up ~10%% per step. Final retrain re-enables them.",
    )
    parser.add_argument(
        "--smoke-date-from", type=str, default="",
        help="V2 smoke mode: subset train/val to samples >= this UTC date (YYYY-MM-DD). "
             "Use with --smoke-date-to for short-fold smoke (e.g. 6mo)."
    )
    parser.add_argument(
        "--smoke-date-to", type=str, default="",
        help="V2 smoke mode: subset train/val to samples <= this UTC date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=0,
        help="Gradient accumulation steps (effective_batch = batch_size × this). "
             "Default 0 = use env GX1_FAST_TRAIN_GRAD_ACCUM (which defaults to 1).",
    )
    parser.add_argument(
        "--init-from-state-dict", type=Path, default=None,
        help="V2 warm-start: load a state_dict (.pt) into the model right after construction. "
             "Use scripts/warm_start_v10_v2_from_v1.py to build a V2 warm-start from a V1 bundle. "
             "Strict=False — missing/unexpected keys are logged but allowed.",
    )
    parser.add_argument(
        "--multi-tf-scale", type=float, default=0.5,
        help="V12.2: scale applied to multi-TF pool before fusion (default 0.5). "
             "Lower (e.g. 0.25) dampens multi-TF contribution and helps if "
             "gradient explosion is observed in delta_abs_mean during early epochs.",
    )
    # 2026-06-02: per-TF learnable input scaling. Initialized with priors and
    # adjusted via backprop during training. Saves cement model from uniform
    # per-TF gates that fail to differentiate macro vs micro signals.
    parser.add_argument("--enable-tf-input-scale", action="store_true",
        help="Enable learnable per-TF input scaling (V10 v5+). Each TF input "
             "is multiplied by a learnable scalar before encoding. Defaults to "
             "user-specified priors that down-weight macro TFs (D1, H4, H1).")
    parser.add_argument("--tf-input-scale-init-m5",  type=float, default=1.0,
        help="Initial value of learnable M5 scale (default 1.0)")
    parser.add_argument("--tf-input-scale-init-m15", type=float, default=1.0,
        help="Initial value of learnable M15 scale (default 1.0)")
    parser.add_argument("--tf-input-scale-init-h1",  type=float, default=0.7,
        help="Initial value of learnable H1 scale (default 0.7 — down-weight macro)")
    parser.add_argument("--tf-input-scale-init-h4",  type=float, default=0.5,
        help="Initial value of learnable H4 scale (default 0.5 — down-weight macro)")
    parser.add_argument("--tf-input-scale-init-d1",  type=float, default=0.3,
        help="Initial value of learnable D1 scale (default 0.3 — down-weight macro)")
    parser.add_argument(
        "--subsample-rows", type=int, default=0,
        help="V12.2 sweep: stratified-subsample train set to at most N rows "
             "(0 = use all). Stratification preserves long/short/flat ratios. "
             "Use for hyperparameter sweeps where each trial needs fast epochs.",
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=1.0,
        help="V12.2: max grad norm for clipping (default 1.0). Use 0.5 for more "
             "aggressive clipping when multi-TF causes train delta to explode.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5,
        help="V12.2: AdamW weight decay (default 1e-5 — V10 v3 baseline). "
             "Increase (e.g. 1e-3) to fight 'weight blow-up' overfit pattern "
             "where train delta grows unboundedly while val is stable.",
    )

    parser.add_argument(
        "--vedtak", type=str, default=None,
        help="Explicit user decision id authorizing this retrain. REQUIRED for --train "
             "(never auto-retrain). Any non-empty string from a deliberate human go.",
    )
    args = parser.parse_args()
    from gx1.runtime.entry_next_edge_legacy_guard import enforce_legacy_entry_research_ack
    enforce_legacy_entry_research_ack("legacy Entry V10_CTX trainer/evaluator")
    # Multi-TF is mandatory — m5-prebuilt-path is required by argparse; nothing to gate.
    # V12.2: apply grad-clip-norm + weight-decay to module-level variables
    global _GRAD_CLIP_NORM, _WEIGHT_DECAY
    _GRAD_CLIP_NORM = float(args.grad_clip_norm)
    _WEIGHT_DECAY = float(args.weight_decay)
    log.info(f"[CONFIG] grad_clip_norm={_GRAD_CLIP_NORM} weight_decay={_WEIGHT_DECAY}")

    _guard_no_rl()

    device = _resolve_device(args.device)
    log.info(f"[CONFIG] seed={args.seed} device={device} deterministic={not args.fast}")
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        log.info(
            "[CUDA_PROOF] cuda_available=%s device=%s device_name=%s",
            True,
            device,
            name,
        )
    else:
        log.info("[CUDA_PROOF] cuda_available=%s device=%s", False, device)

    if args.sanity:
        if args.out_bundle_dir is None:
            parser.error("--out_bundle_dir is required for --sanity")
        _log_manifest_proof(args.dataset_manifest)
        run_sanity_check(
            seq_len=args.seq_len,
            seed=args.seed,
            device=device,
            out_bundle_dir=args.out_bundle_dir,
            dataset_manifest=args.dataset_manifest,
            deterministic=not args.fast,
            enable_specialist_fusion=bool(args.enable_specialist_fusion),
            specialist_audit_json=args.specialist_audit_json,
            specialist_contract_mode=str(args.specialist_contract_mode),
            specialist_num_layers=int(args.specialist_num_layers),
            specialist_fusion_scale=float(args.specialist_fusion_scale),
        )
        return

    if args.train:
        # NEVER auto-retrain — fail-closed unless an explicit --vedtak is passed.
        from gx1_guards.gates import require_retrain_vedtak, GateError
        try:
            require_retrain_vedtak(args.vedtak)
        except GateError as e:
            parser.error(str(e))
        if args.out_bundle_dir is None:
            parser.error("--out_bundle_dir is required for --train")
        _log_manifest_proof(args.dataset_manifest)
        gx1_data = _resolve_gx1_data(args.gx1_data)
        train_parquet, val_parquet = _resolve_train_val_parquets(
            args.dataset_manifest,
            args.dataset_dir,
            gx1_data,
            train_parquet_hint=args.dataset_train_parquet,
        )
        try:
            test_parquet = _resolve_test_parquet(
                args.dataset_manifest,
                args.dataset_dir,
                args.test_parquet,
                gx1_data,
            )
            _log_label_distribution(test_parquet, split="test")
        except Exception as e:
            log.warning("[ENTRY_LABEL_DISTRIBUTION] split=test status=skip reason=%s", e)
        # V2 prelim-mode: force-disable all V10 v3+ aux heads for prelim training (item 16).
        # Final retrain (without --prelim-no-aux-heads) re-enables them.
        _aux_tf = args.enable_tf_agreement_head or args.enable_v10_v3plus_all_heads
        _aux_pqv = args.enable_path_quality_variance_head or args.enable_v10_v3plus_all_heads
        _aux_ps = args.enable_position_size_head or args.enable_v10_v3plus_all_heads
        _aux_hh = args.enable_hold_horizon_head or args.enable_v10_v3plus_all_heads
        _anchor_gate = bool(args.enable_anchor_gate or args.enable_xau_direction_repair_heads)
        _hier_heads = bool(args.enable_hierarchical_entry_heads or args.enable_xau_direction_repair_heads)
        _side_validity_head = bool(args.enable_side_validity_head or args.enable_xau_direction_repair_heads)
        _trendline_head = bool(args.enable_trendline_rail_head or args.enable_xau_direction_repair_heads)
        if args.prelim_no_aux_heads:
            log.info("[PRELIM_NO_AUX_HEADS] disabling tf_agreement/path_var/pos_size/hold_horizon heads for prelim")
            _aux_tf = _aux_pqv = _aux_ps = _aux_hh = False
        run_train(
            train_parquet=train_parquet,
            val_parquet=val_parquet,
            seq_len=args.seq_len,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            out_bundle_dir=args.out_bundle_dir,
            gx1_data_override=args.gx1_data,
            allow_constant_labels=args.allow_constant_labels,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            deterministic=not args.fast,
            enable_multi_tf=True,  # MANDATORY — never single-TF (toggle removed 2026-05-26)
            m5_prebuilt_path=args.m5_prebuilt_path,
            multi_tf_seq_len=args.multi_tf_seq_len,
            multi_tf_scale=args.multi_tf_scale,
            subsample_rows=args.subsample_rows,
            enable_tf_agreement_head=_aux_tf,
            enable_path_quality_variance_head=_aux_pqv,
            enable_position_size_head=_aux_ps,
            enable_hold_horizon_head=_aux_hh,
            enable_pos_enc=bool(args.enable_pos_enc),
            enable_regime_film=bool(args.enable_regime_film),
            enable_dip_head=bool(args.enable_dip_head),
            enable_forecast_head=bool(args.enable_forecast_head),
            enable_cross_tf_attn=bool(args.enable_cross_tf_attn),
            enable_timing_head=bool(args.enable_timing_head),
            enable_tail_risk_head=bool(args.enable_tail_risk_head),
            enable_vol_forecast_head=bool(args.enable_vol_forecast_head),
        enable_anchor_gate=_anchor_gate,
        anchor_gate_init=float(args.anchor_gate_init),
        enable_hierarchical_entry_heads=_hier_heads,
        enable_hierarchical_direction_composition=bool(ENTRY_DIRECTION_HIERARCHICAL_COMPOSITION),
        enable_side_validity_head=_side_validity_head,
            enable_trendline_rail_head=_trendline_head,
            enable_specialist_fusion=bool(args.enable_specialist_fusion),
            specialist_audit_json=args.specialist_audit_json,
            specialist_contract_mode=str(args.specialist_contract_mode),
            specialist_num_layers=int(args.specialist_num_layers),
            specialist_fusion_scale=float(args.specialist_fusion_scale),
            enable_mtf_direction_head=bool(args.enable_mtf_direction_head),
            mtf_dir_scale_init=float(args.mtf_dir_scale_init),
            # V2 fast-train extras
            per_tf_seq_len_h4=int(args.per_tf_seq_len_h4),
            per_tf_seq_len_d1=int(args.per_tf_seq_len_d1),
            smoke_date_from=str(args.smoke_date_from or ""),
            smoke_date_to=str(args.smoke_date_to or ""),
            grad_accum_steps=int(args.grad_accum_steps),
            init_from_state_dict=args.init_from_state_dict,
            # 2026-06-02: per-TF learnable input scaling
            enable_tf_input_scale=bool(args.enable_tf_input_scale),
            tf_input_scale_init_m5=float(args.tf_input_scale_init_m5),
            tf_input_scale_init_m15=float(args.tf_input_scale_init_m15),
            tf_input_scale_init_h1=float(args.tf_input_scale_init_h1),
            tf_input_scale_init_h4=float(args.tf_input_scale_init_h4),
            tf_input_scale_init_d1=float(args.tf_input_scale_init_d1),
            vedtak_id=str(args.vedtak or ""),
        )
        return

    if args.eval:
        _require(args.bundle_dir is not None, "[ENTRY_V10_CTX_EVAL_BUNDLE_REQUIRED]")
        _log_manifest_proof(args.dataset_manifest)
        gx1_data = _resolve_gx1_data(args.gx1_data)
        test_parquet = _resolve_test_parquet(
            args.dataset_manifest,
            args.dataset_dir,
            args.test_parquet,
            gx1_data,
            bundle_dir=args.bundle_dir,
        )
        # Resolve train/val from bundle metadata if available
        train_parquet = None
        val_parquet = None
        try:
            meta_path = Path(args.bundle_dir).expanduser() / "bundle_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("train_data"):
                    train_parquet = Path(meta["train_data"])
                if meta.get("val_data"):
                    val_parquet = Path(meta["val_data"])
        except Exception as e:
            log.warning("[ENTRY_TRAINING_BIAS_AUDIT] failed to resolve train/val from bundle metadata: %s", e)

        run_eval(
            bundle_dir=args.bundle_dir,
            train_parquet=train_parquet,
            val_parquet=val_parquet,
            test_parquet=test_parquet,
            m5_prebuilt_path=args.m5_prebuilt_path,
            seq_len=args.seq_len,
            seed=args.seed,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gx1_data_override=args.gx1_data,
        )
        return

if __name__ == "__main__":
    main()
