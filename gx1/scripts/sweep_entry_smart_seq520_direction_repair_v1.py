#!/usr/bin/env python3
"""Bounded XAU smart-seq520 direction-repair sweep orchestrator.

This runner is deliberately XAU-only and defaults to planning/dry-run. It does
not introduce runtime trading rules; it varies model-learning weights and train
hyperparameters around the audited smart-seq520 recipe.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.scripts.audit_xau_direction_repair_pretrain_v1 import (
    DEFAULT_DATASET_DIR,
    DEFAULT_STEM,
    run as run_pretrain_audit,
)


REPO = Path("/home/andre2/src/GX1_ENGINE")
DATA = Path("/home/andre2/GX1_DATA")
WRAPPER = REPO / "scripts/run_entry_foundation_seq146_candidate_train.sh"
DEFAULT_SWEEP_ROOT = DATA / (
    "runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/"
    "v10_entry_smart_seq520_xau_direction_repair_sweep_v1"
)
DEFAULT_PLAN_OUT = DATA / "reports/xau_direction_repair_sweep_20260713_v1"


@dataclass(frozen=True)
class Space:
    name: str
    kind: str
    lo: float | None = None
    hi: float | None = None
    choices: tuple[float | str, ...] = ()


SWEEP_SPACES: tuple[Space, ...] = (
    Space("ENTRY_FOUNDATION_CANDIDATE_LR", "log", 1.0e-4, 6.0e-4),
    Space("ENTRY_FOUNDATION_CANDIDATE_WEIGHT_DECAY", "log", 1.0e-6, 3.0e-4),
    Space("ENTRY_FOUNDATION_CANDIDATE_GRAD_CLIP_NORM", "choice", choices=(0.5, 0.75, 1.0, 1.25)),
    Space("ENTRY_FOUNDATION_CANDIDATE_MULTI_TF_SCALE", "choice", choices=(0.25, 0.35, 0.50, 0.65)),
    Space("ENTRY_FOUNDATION_CANDIDATE_MTF_DIR_SCALE_INIT", "choice", choices=(0.15, 0.25, 0.35, 0.50)),
    Space("ENTRY_FOUNDATION_CANDIDATE_SPECIALIST_FUSION_SCALE", "choice", choices=(0.15, 0.25, 0.35)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE", "choice", choices=(3.0, 4.0, 5.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA", "choice", choices=(0.45, 0.50)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT", "choice", choices=(8.0, 12.0, 16.0)),
    Space(
        "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE",
        "choice",
        choices=(0.03, 0.05),
    ),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT", "choice", choices=(8.0, 12.0, 16.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT", "choice", choices=(4.0, 8.0, 12.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT", "choice", choices=(0.0, 4.0, 8.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_WEIGHT", "choice", choices=(2.0, 3.0, 4.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT", "choice", choices=(2.0, 3.0, 4.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT", "choice", choices=(4.0, 6.0, 8.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT", "choice", choices=(3.0, 4.0, 5.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_WEIGHT", "choice", choices=(2.00, 2.50)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT", "choice", choices=(4.0, 6.0, 8.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT", "choice", choices=(4.0, 6.0, 8.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_FLAT_LOGIT_MARGIN_WEIGHT", "choice", choices=(8.0, 10.0, 12.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT", "choice", choices=(8.0, 10.0, 12.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_WEIGHT", "choice", choices=(1.75, 2.00, 2.25)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_CE_WEIGHT", "choice", choices=(4.0, 5.0, 6.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT", "choice", choices=(3.0, 4.0, 5.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT", "choice", choices=(4.0, 6.0, 8.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT", "choice", choices=(4.0, 6.0, 8.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_WEIGHT", "choice", choices=(1.50, 2.00)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_ABSTAIN_WEIGHT", "choice", choices=(5.0, 7.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_SIDE_MARGIN_WEIGHT", "choice", choices=(3.0, 4.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT", "choice", choices=(1.5, 2.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT", "choice", choices=(5.0, 7.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT", "choice", choices=(4.0, 5.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT", "choice", choices=(5.0, 7.0)),
    Space("ENTRY_FOUNDATION_CANDIDATE_FLAT_CLASS_WEIGHT_FLOOR", "choice", choices=(1.00, 1.50, 2.00)),
)

FIXED_ENV: dict[str, str] = {
    "ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS": "1.0,1.0,4.0",
    "ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_TARGET": "label",
    "ENTRY_FOUNDATION_CANDIDATE_ANCHOR_GATE_INIT": "0.0",
    "ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_PROB_PENALTY": "0.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FRACTION": "0.50",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_FLOOR": "0.05",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION": "0.50",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR": "0.05",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_CTX_CAT_INDICES": "0,1,2,3,4",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_PROB_FLOOR": "0.30",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN": "0.02",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_LOSS_AGGREGATION": "mean_max",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_SAMPLER": "1",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE": "3",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS": "6",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT": "4.00",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_MARGIN_WEIGHT": "4.00",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_MIN_GAP_BPS": "15.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_LOGIT_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT": "6.00",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS": "15.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT": "8.00",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS": "15.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS": "0.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH": "0.50",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_WEIGHT": "8.00",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS": "15.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS": "0.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH": "0.50",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP": "4.0",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_HIERARCHICAL_COMPOSITION": "1",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_COMPOSE_RESIDUAL_LOGIT_CAP": "0.18",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL": "1",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_WEIGHT": "8.00",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_PRED_FRACTION": "0.50",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_PRED_FLOOR": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_LEGACY_CE_MULT": "1.00",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_UTILITY_WEIGHT": "1.00",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_BAD_PATH_WEIGHT": "1.25",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_MAE_WEIGHT": "0.35",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS": "15.0",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP": "8.0",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_FLAT_LOGIT_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_TRUE_MARGIN": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_FOUNDATION_CANDIDATE_HIER_POCKET_UTILITY_MARGIN_BPS": "30.0",
    "ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_AUX_WEIGHT": "1.00",
    "ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT": "1.50",
    "ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT": "1.75",
    "ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT": "3.00",
    "ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_MARGIN": "1.00",
    "ENTRY_FOUNDATION_CANDIDATE_TRENDLINE_RAIL_UTILITY_MARGIN_BPS": "30.0",
}


def lint_trial_env(env: dict[str, str]) -> list[str]:
    failures: list[str] = []
    direction_ce = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_CE_SCALE", "0"))
    pred_balance_alpha = float(env.get("ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_ALPHA", "0"))
    bad_path_penalty = float(env.get("ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_PROB_PENALTY", "nan"))
    anchor_gate = float(env.get("ENTRY_FOUNDATION_CANDIDATE_ANCHOR_GATE_INIT", "nan"))
    if direction_ce < 2.0:
        failures.append(f"DIRECTION_CE_SCALE must be >= 2.0 for XAU repair, got {direction_ce}")
    if pred_balance_alpha < 0.45 or pred_balance_alpha > 0.50:
        failures.append(
            "PRED_BALANCE_ALPHA must stay within [0.45, 0.50] for strict smart XAU repair, "
            f"got {pred_balance_alpha}"
        )
    min_pred_rate_weight = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT", "0"))
    if min_pred_rate_weight < 2.5:
        failures.append(
            "DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT must be >= 2.5 for strict XAU repair, "
            f"got {min_pred_rate_weight}"
        )
    min_pred_rate_temp = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", "1.0")
    )
    if min_pred_rate_temp <= 0.0 or min_pred_rate_temp > 0.05:
        failures.append(
            "DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE must be in (0.0, 0.05] for strict XAU repair, "
            f"got {min_pred_rate_temp}"
        )
    global_prior_match_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT", "0")
    )
    global_prior_match_tolerance = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE", "1")
    )
    global_prior_match_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", "0")
    )
    if global_prior_match_weight < 8.0:
        failures.append(
            "DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT must be >= 8.0 for strict XAU repair, "
            f"got {global_prior_match_weight}"
        )
    if global_prior_match_tolerance > 0.02:
        failures.append(
            "DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE must be <= 0.02 for strict XAU repair, "
            f"got {global_prior_match_tolerance}"
        )
    if global_prior_match_min_label_rate != 0.10:
        failures.append(
            "DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE must stay 0.10 for strict XAU repair, "
            f"got {global_prior_match_min_label_rate}"
        )
    flat_margin_weight = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN_WEIGHT", "0"))
    flat_margin = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_VS_FLAT_MARGIN", "0"))
    if flat_margin_weight < 3.0:
        failures.append(
            "DIRECTION_VS_FLAT_MARGIN_WEIGHT must be >= 3.0 for strict XAU repair, "
            f"got {flat_margin_weight}"
        )
    if flat_margin < 0.05:
        failures.append(f"DIRECTION_VS_FLAT_MARGIN must be >= 0.05 for strict XAU repair, got {flat_margin}")
    utility_margin_weight = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_MARGIN_WEIGHT", "0"))
    utility_min_gap = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_MIN_GAP_BPS", "999"))
    utility_logit_margin = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_LOGIT_MARGIN", "0"))
    if utility_margin_weight < 4.0:
        failures.append(
            "DIRECTION_UTILITY_MARGIN_WEIGHT must be >= 4.0 for strict XAU repair, "
            f"got {utility_margin_weight}"
        )
    if utility_min_gap > 15.0:
        failures.append(
            "DIRECTION_UTILITY_MIN_GAP_BPS must be <= 15.0 for strict XAU repair, "
            f"got {utility_min_gap}"
        )
    if utility_logit_margin < 0.10:
        failures.append(
            "DIRECTION_UTILITY_LOGIT_MARGIN must be >= 0.10 for strict XAU repair, "
            f"got {utility_logit_margin}"
        )
    side_utility_conviction_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT", "0")
    )
    side_utility_conviction_min_gap = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS", "999")
    )
    side_utility_conviction_logit_margin = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN", "0")
    )
    if side_utility_conviction_weight < 6.0:
        failures.append(
            "DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT must be >= 6.0 for strict XAU repair, "
            f"got {side_utility_conviction_weight}"
        )
    if side_utility_conviction_min_gap > 15.0:
        failures.append(
            "DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS must be <= 15.0 for strict XAU repair, "
            f"got {side_utility_conviction_min_gap}"
        )
    if side_utility_conviction_logit_margin < 0.10:
        failures.append(
            "DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN must be >= 0.10 for strict XAU repair, "
            f"got {side_utility_conviction_logit_margin}"
        )
    utility_trade_conviction_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT", "0")
    )
    utility_trade_conviction_min_gap = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS", "999")
    )
    utility_trade_conviction_min_utility = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS", "999")
    )
    utility_trade_conviction_max_bad_path = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH", "999")
    )
    utility_trade_conviction_logit_margin = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN", "0")
    )
    if utility_trade_conviction_weight < 8.0:
        failures.append(
            "DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT must be >= 8.0 for strict XAU repair, "
            f"got {utility_trade_conviction_weight}"
        )
    if utility_trade_conviction_min_gap > 15.0:
        failures.append(
            "DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS must be <= 15.0 for strict XAU repair, "
            f"got {utility_trade_conviction_min_gap}"
        )
    if utility_trade_conviction_min_utility > 0.0:
        failures.append(
            "DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS must be <= 0.0 for strict XAU repair, "
            f"got {utility_trade_conviction_min_utility}"
        )
    if utility_trade_conviction_max_bad_path > 0.50:
        failures.append(
            "DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH must be <= 0.50 for strict XAU repair, "
            f"got {utility_trade_conviction_max_bad_path}"
        )
    if utility_trade_conviction_logit_margin < 0.10:
        failures.append(
            "DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN must be >= 0.10 for strict XAU repair, "
            f"got {utility_trade_conviction_logit_margin}"
        )
    utility_triad_ce_weight = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_WEIGHT", "0"))
    utility_triad_ce_min_gap = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS", "999")
    )
    utility_triad_ce_min_utility = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS", "999")
    )
    utility_triad_ce_max_bad_path = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH", "999")
    )
    utility_triad_ce_class_weight_cap = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP", "0")
    )
    if utility_triad_ce_weight < 8.0:
        failures.append(
            "DIRECTION_UTILITY_TRIAD_CE_WEIGHT must be >= 8.0 for strict XAU repair, "
            f"got {utility_triad_ce_weight}"
        )
    if utility_triad_ce_min_gap > 15.0:
        failures.append(
            "DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS must be <= 15.0 for strict XAU repair, "
            f"got {utility_triad_ce_min_gap}"
        )
    if utility_triad_ce_min_utility > 0.0:
        failures.append(
            "DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS must be <= 0.0 for strict XAU repair, "
            f"got {utility_triad_ce_min_utility}"
        )
    if utility_triad_ce_max_bad_path > 0.50:
        failures.append(
            "DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH must be <= 0.50 for strict XAU repair, "
            f"got {utility_triad_ce_max_bad_path}"
        )
    if utility_triad_ce_class_weight_cap < 2.0:
        failures.append(
            "DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP must be >= 2.0 for strict XAU repair, "
            f"got {utility_triad_ce_class_weight_cap}"
        )
    hierarchical_composition = int(float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_HIERARCHICAL_COMPOSITION", "0")))
    if hierarchical_composition != 1:
        failures.append(
            "DIRECTION_HIERARCHICAL_COMPOSITION must be 1 for strict XAU repair, "
            f"got {hierarchical_composition}"
        )
    hier_compose_residual_logit_cap = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_COMPOSE_RESIDUAL_LOGIT_CAP", "0")
    )
    if hier_compose_residual_logit_cap < 0.10 or hier_compose_residual_logit_cap > 0.20:
        failures.append(
            "HIER_COMPOSE_RESIDUAL_LOGIT_CAP must stay within [0.10, 0.20] for strict XAU repair, "
            f"got {hier_compose_residual_logit_cap}"
        )
    hier_compose_residual_side_neutral = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL", "0"))
    )
    if hier_compose_residual_side_neutral != 1:
        failures.append(
            "HIER_COMPOSE_RESIDUAL_SIDE_NEUTRAL must be 1 for strict XAU repair, "
            f"got {hier_compose_residual_side_neutral}"
        )
    hier_trade_global_prior_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT", "0")
    )
    hier_trade_global_prior_tolerance = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE", "1")
    )
    hier_trade_global_prior_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", "0")
    )
    hier_slice_trade_prior_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT", "0")
    )
    hier_slice_trade_prior_tolerance = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE", "1")
    )
    hier_slice_trade_prior_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE", "0")
    )
    hier_slice_trade_prior_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS", "0"))
    )
    hier_flat_logit_margin_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_FLAT_LOGIT_MARGIN_WEIGHT", "0")
    )
    hier_flat_logit_margin = float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_FLAT_LOGIT_MARGIN", "0"))
    hier_flat_logit_margin_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE", "0")
    )
    hier_slice_flat_logit_margin_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT", "0")
    )
    hier_slice_flat_logit_margin = float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN", "0"))
    hier_slice_flat_logit_margin_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE", "0")
    )
    hier_slice_flat_logit_margin_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS", "0"))
    )
    if hier_trade_global_prior_weight < 4.0:
        failures.append(
            "HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT must be >= 4.0 for strict XAU repair, "
            f"got {hier_trade_global_prior_weight}"
        )
    if hier_trade_global_prior_tolerance > 0.02:
        failures.append(
            "HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE must be <= 0.02 for strict XAU repair, "
            f"got {hier_trade_global_prior_tolerance}"
        )
    if hier_trade_global_prior_min_label_rate < 0.10:
        failures.append(
            "HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE must be >= 0.10 for strict XAU repair, "
            f"got {hier_trade_global_prior_min_label_rate}"
        )
    if hier_slice_trade_prior_weight < 4.0:
        failures.append(
            "HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT must be >= 4.0 for strict XAU repair, "
            f"got {hier_slice_trade_prior_weight}"
        )
    if hier_slice_trade_prior_tolerance > 0.02:
        failures.append(
            "HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE must be <= 0.02 for strict XAU repair, "
            f"got {hier_slice_trade_prior_tolerance}"
        )
    if hier_slice_trade_prior_min_label_rate < 0.10:
        failures.append(
            "HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE must be >= 0.10 for strict XAU repair, "
            f"got {hier_slice_trade_prior_min_label_rate}"
        )
    if hier_slice_trade_prior_min_rows < 8:
        failures.append(
            "HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS must be >= 8 for strict XAU repair, "
            f"got {hier_slice_trade_prior_min_rows}"
        )
    if hier_flat_logit_margin_weight < 8.0:
        failures.append(
            "HIER_FLAT_LOGIT_MARGIN_WEIGHT must be >= 8.0 for strict XAU repair, "
            f"got {hier_flat_logit_margin_weight}"
        )
    if hier_flat_logit_margin < 0.10:
        failures.append(
            "HIER_FLAT_LOGIT_MARGIN must be >= 0.10 for strict XAU repair, "
            f"got {hier_flat_logit_margin}"
        )
    if hier_flat_logit_margin_min_label_rate < 0.10:
        failures.append(
            "HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE must be >= 0.10 for strict XAU repair, "
            f"got {hier_flat_logit_margin_min_label_rate}"
        )
    if hier_slice_flat_logit_margin_weight < 8.0:
        failures.append(
            "HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT must be >= 8.0 for strict XAU repair, "
            f"got {hier_slice_flat_logit_margin_weight}"
        )
    if hier_slice_flat_logit_margin < 0.10:
        failures.append(
            "HIER_SLICE_FLAT_LOGIT_MARGIN must be >= 0.10 for strict XAU repair, "
            f"got {hier_slice_flat_logit_margin}"
        )
    if hier_slice_flat_logit_margin_min_label_rate < 0.10:
        failures.append(
            "HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE must be >= 0.10 for strict XAU repair, "
            f"got {hier_slice_flat_logit_margin_min_label_rate}"
        )
    if hier_slice_flat_logit_margin_min_rows < 8:
        failures.append(
            "HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS must be >= 8 for strict XAU repair, "
            f"got {hier_slice_flat_logit_margin_min_rows}"
        )
    hier_slice_side_ce_weight = float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_CE_WEIGHT", "0"))
    hier_slice_side_true_margin_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT", "0")
    )
    hier_slice_side_true_margin = float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_TRUE_MARGIN", "0"))
    hier_slice_side_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_MIN_LABEL_RATE", "0")
    )
    hier_slice_side_min_rows = int(float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_MIN_ROWS", "0")))
    if hier_slice_side_ce_weight < 4.0:
        failures.append(
            "HIER_SLICE_SIDE_CE_WEIGHT must be >= 4.0 for strict XAU repair, "
            f"got {hier_slice_side_ce_weight}"
        )
    if hier_slice_side_true_margin_weight < 3.0:
        failures.append(
            "HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT must be >= 3.0 for strict XAU repair, "
            f"got {hier_slice_side_true_margin_weight}"
        )
    if hier_slice_side_true_margin != 0.10:
        failures.append(
            "HIER_SLICE_SIDE_TRUE_MARGIN must stay 0.10 for strict XAU repair, "
            f"got {hier_slice_side_true_margin}"
        )
    if hier_slice_side_min_label_rate != 0.10:
        failures.append(
            "HIER_SLICE_SIDE_MIN_LABEL_RATE must stay 0.10 for strict XAU repair, "
            f"got {hier_slice_side_min_label_rate}"
        )
    if hier_slice_side_min_rows != 8:
        failures.append(
            "HIER_SLICE_SIDE_MIN_ROWS must stay 8 for strict XAU repair, "
            f"got {hier_slice_side_min_rows}"
        )
    hier_side_global_prior_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT", "0")
    )
    hier_side_global_prior_tolerance = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE", "1")
    )
    hier_side_global_prior_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", "0")
    )
    hier_slice_side_prior_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT", "0")
    )
    hier_slice_side_prior_tolerance = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE", "1")
    )
    hier_slice_side_prior_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE", "0")
    )
    hier_slice_side_prior_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS", "0"))
    )
    if hier_side_global_prior_weight < 4.0:
        failures.append(
            "HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT must be >= 4.0 for strict XAU repair, "
            f"got {hier_side_global_prior_weight}"
        )
    if hier_side_global_prior_tolerance > 0.02:
        failures.append(
            "HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE must be <= 0.02 for strict XAU repair, "
            f"got {hier_side_global_prior_tolerance}"
        )
    if hier_side_global_prior_min_label_rate < 0.10:
        failures.append(
            "HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE must be >= 0.10 for strict XAU repair, "
            f"got {hier_side_global_prior_min_label_rate}"
        )
    if hier_slice_side_prior_weight < 4.0:
        failures.append(
            "HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT must be >= 4.0 for strict XAU repair, "
            f"got {hier_slice_side_prior_weight}"
        )
    if hier_slice_side_prior_tolerance > 0.02:
        failures.append(
            "HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE must be <= 0.02 for strict XAU repair, "
            f"got {hier_slice_side_prior_tolerance}"
        )
    if hier_slice_side_prior_min_label_rate < 0.10:
        failures.append(
            "HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE must be >= 0.10 for strict XAU repair, "
            f"got {hier_slice_side_prior_min_label_rate}"
        )
    if hier_slice_side_prior_min_rows < 8:
        failures.append(
            "HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS must be >= 8 for strict XAU repair, "
            f"got {hier_slice_side_prior_min_rows}"
        )
    flat_starvation_weight = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_WEIGHT", "0"))
    flat_starvation_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE", "0")
    )
    flat_starvation_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_MIN_ROWS", "0"))
    )
    flat_starvation_pred_fraction = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_PRED_FRACTION", "0")
    )
    flat_starvation_pred_floor = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_PRED_FLOOR", "0")
    )
    flat_starvation_logit_margin = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN", "0")
    )
    if flat_starvation_weight < 8.0:
        failures.append(
            "DIRECTION_FLAT_STARVATION_WEIGHT must be >= 8.0 for strict XAU repair, "
            f"got {flat_starvation_weight}"
        )
    if flat_starvation_min_label_rate < 0.10:
        failures.append(
            "DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE must be >= 0.10 for strict XAU repair, "
            f"got {flat_starvation_min_label_rate}"
        )
    if flat_starvation_min_rows < 8:
        failures.append(
            "DIRECTION_FLAT_STARVATION_MIN_ROWS must be >= 8 for strict XAU repair, "
            f"got {flat_starvation_min_rows}"
        )
    if flat_starvation_pred_fraction < 0.50:
        failures.append(
            "DIRECTION_FLAT_STARVATION_PRED_FRACTION must be >= 0.50 for strict XAU repair, "
            f"got {flat_starvation_pred_fraction}"
        )
    if flat_starvation_pred_floor < 0.10:
        failures.append(
            "DIRECTION_FLAT_STARVATION_PRED_FLOOR must be >= 0.10 for strict XAU repair, "
            f"got {flat_starvation_pred_floor}"
        )
    if flat_starvation_logit_margin < 0.10:
        failures.append(
            "DIRECTION_FLAT_STARVATION_LOGIT_MARGIN must be >= 0.10 for strict XAU repair, "
            f"got {flat_starvation_logit_margin}"
        )
    slice_min_pred_rate_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT", "0")
    )
    if slice_min_pred_rate_weight < 0.0:
        failures.append(
            "DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT must be non-negative, "
            f"got {slice_min_pred_rate_weight}"
        )
    slice_recall_weight = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_RECALL_LOSS_WEIGHT", "0"))
    if slice_recall_weight < 0.0:
        failures.append(f"DIRECTION_SLICE_RECALL_LOSS_WEIGHT must be non-negative, got {slice_recall_weight}")
    slice_balanced_ce_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_WEIGHT", "0")
    )
    slice_balanced_ce_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE", "0")
    )
    slice_balanced_ce_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS", "0"))
    )
    if slice_balanced_ce_weight < 2.0:
        failures.append(
            "DIRECTION_SLICE_BALANCED_CE_WEIGHT must be >= 2.0 for strict XAU repair, "
            f"got {slice_balanced_ce_weight}"
        )
    if slice_balanced_ce_min_label_rate != 0.10:
        failures.append(
            "DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE must stay 0.10 for strict XAU repair, "
            f"got {slice_balanced_ce_min_label_rate}"
        )
    if slice_balanced_ce_min_rows != 8:
        failures.append(
            "DIRECTION_SLICE_BALANCED_CE_MIN_ROWS must stay 8 for strict XAU repair, "
            f"got {slice_balanced_ce_min_rows}"
        )
    slice_true_margin_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT", "0")
    )
    slice_true_margin = float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN", "0"))
    slice_true_margin_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE", "0")
    )
    slice_true_margin_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS", "0"))
    )
    if slice_true_margin_weight < 2.0:
        failures.append(
            "DIRECTION_SLICE_TRUE_MARGIN_WEIGHT must be >= 2.0 for strict XAU repair, "
            f"got {slice_true_margin_weight}"
        )
    if slice_true_margin != 0.10:
        failures.append(
            "DIRECTION_SLICE_TRUE_MARGIN must stay 0.10 for strict XAU repair, "
            f"got {slice_true_margin}"
        )
    if slice_true_margin_min_label_rate != 0.10:
        failures.append(
            "DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE must stay 0.10 for strict XAU repair, "
            f"got {slice_true_margin_min_label_rate}"
        )
    if slice_true_margin_min_rows != 8:
        failures.append(
            "DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS must stay 8 for strict XAU repair, "
            f"got {slice_true_margin_min_rows}"
        )
    slice_accuracy_edge_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT", "0")
    )
    slice_accuracy_edge_margin = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN", "0")
    )
    slice_accuracy_edge_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE", "0")
    )
    slice_accuracy_edge_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS", "0"))
    )
    if slice_accuracy_edge_weight < 4.0:
        failures.append(
            "DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT must be >= 4.0 for strict XAU repair, "
            f"got {slice_accuracy_edge_weight}"
        )
    if slice_accuracy_edge_margin < 0.02:
        failures.append(
            "DIRECTION_SLICE_ACCURACY_EDGE_MARGIN must be >= 0.02 for strict XAU repair, "
            f"got {slice_accuracy_edge_margin}"
        )
    if slice_accuracy_edge_min_label_rate != 0.10:
        failures.append(
            "DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE must stay 0.10 for strict XAU repair, "
            f"got {slice_accuracy_edge_min_label_rate}"
        )
    if slice_accuracy_edge_min_rows != 8:
        failures.append(
            "DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS must stay 8 for strict XAU repair, "
            f"got {slice_accuracy_edge_min_rows}"
        )
    slice_prior_match_weight = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT", "0")
    )
    slice_prior_match_tolerance = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE", "1")
    )
    slice_prior_match_min_label_rate = float(
        env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE", "0")
    )
    slice_prior_match_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS", "0"))
    )
    if slice_prior_match_weight < 3.0:
        failures.append(
            "DIRECTION_SLICE_PRIOR_MATCH_WEIGHT must be >= 3.0 for strict XAU repair, "
            f"got {slice_prior_match_weight}"
        )
    if slice_prior_match_tolerance > 0.02:
        failures.append(
            "DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE must be <= 0.02 for strict XAU repair, "
            f"got {slice_prior_match_tolerance}"
        )
    if slice_prior_match_min_label_rate != 0.10:
        failures.append(
            "DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE must stay 0.10 for strict XAU repair, "
            f"got {slice_prior_match_min_label_rate}"
        )
    if slice_prior_match_min_rows != 8:
        failures.append(
            "DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS must stay 8 for strict XAU repair, "
            f"got {slice_prior_match_min_rows}"
        )
    if env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_LOSS_AGGREGATION") != "mean_max":
        failures.append("DIRECTION_SLICE_LOSS_AGGREGATION must stay mean_max for strict XAU repair")
    if env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_SAMPLER") != "1":
        failures.append("DIRECTION_SLICE_BALANCED_SAMPLER must stay 1 for strict XAU repair")
    slice_sampler_min_rows = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS", "0"))
    )
    if slice_sampler_min_rows != 8:
        failures.append(
            "DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS must stay 8 for strict XAU repair, "
            f"got {slice_sampler_min_rows}"
        )
    hard_red_stop_patience = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE", "0"))
    )
    hard_red_stop_min_epochs = int(
        float(env.get("ENTRY_FOUNDATION_CANDIDATE_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS", "0"))
    )
    if hard_red_stop_patience != 3:
        failures.append(
            "DIRECTION_SLICE_HARD_RED_STOP_PATIENCE must stay 3 for strict XAU repair, "
            f"got {hard_red_stop_patience}"
        )
    if hard_red_stop_min_epochs != 6:
        failures.append(
            "DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS must stay 6 for strict XAU repair, "
            f"got {hard_red_stop_min_epochs}"
        )
    if bad_path_penalty != 0.0:
        failures.append(f"BAD_PATH_PROB_PENALTY must stay 0.0, got {bad_path_penalty}")
    if anchor_gate != 0.0:
        failures.append(f"ANCHOR_GATE_INIT must stay 0.0, got {anchor_gate}")
    if env.get("ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_TARGET") != "label":
        failures.append("PRED_BALANCE_TARGET must stay label")
    if env.get("ENTRY_FOUNDATION_CANDIDATE_PRED_BALANCE_CLASS_WEIGHTS") != "1.0,1.0,4.0":
        failures.append("PRED_BALANCE_CLASS_WEIGHTS must stay 1.0,1.0,4.0")
    return failures


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _fmt_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if abs(float(value)) < 0.001:
        return f"{float(value):.6g}"
    return f"{float(value):.4g}"


def _latin_values(space: Space, trials: int, rng: np.random.Generator) -> list[float | str]:
    if space.kind == "choice":
        choices = list(space.choices)
        return [choices[int(rng.integers(0, len(choices)))] for _ in range(trials)]
    if space.lo is None or space.hi is None:
        raise RuntimeError(f"continuous space missing bounds: {space.name}")
    edges = np.linspace(0.0, 1.0, int(trials) + 1)
    points = [float(rng.uniform(edges[i], edges[i + 1])) for i in range(int(trials))]
    rng.shuffle(points)
    if space.kind == "log":
        lo = math.log(float(space.lo))
        hi = math.log(float(space.hi))
        return [float(math.exp(lo + point * (hi - lo))) for point in points]
    if space.kind == "linear":
        return [float(space.lo + point * (float(space.hi) - float(space.lo))) for point in points]
    raise RuntimeError(f"unknown sweep space kind: {space.kind}")


def sample_trials(*, trials: int, seed: int) -> list[dict[str, str]]:
    rng = np.random.default_rng(int(seed))
    columns = {space.name: _latin_values(space, int(trials), rng) for space in SWEEP_SPACES}
    out: list[dict[str, str]] = []
    for idx in range(int(trials)):
        env = dict(FIXED_ENV)
        for name, values in columns.items():
            env[name] = _fmt_value(values[idx])
        out.append(env)
    lint_failures = [
        {"trial_index": idx + 1, "failures": failures}
        for idx, env in enumerate(out)
        if (failures := lint_trial_env(env))
    ]
    if lint_failures:
        raise RuntimeError(f"XAU_SWEEP_SPACE_CONTRACT_FAIL: {lint_failures[:5]}")
    return out


def trial_command(
    *,
    trial_idx: int,
    trial_env: dict[str, str],
    dataset_dir: Path,
    out_bundle_dir: Path,
    vedtak: str,
    epochs: int,
    batch_size: int,
    subsample_rows: int,
    seed: int,
    dry_run: bool,
) -> list[str]:
    env_parts = ["env"] + [f"{key}={value}" for key, value in sorted(trial_env.items())]
    cmd = [
        str(WRAPPER),
        "--smart-seq520",
        "--vedtak",
        str(vedtak),
        "--dataset-dir",
        str(dataset_dir),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--seed",
        str(int(seed) + int(trial_idx)),
        "--subsample-rows",
        str(int(subsample_rows)),
        "--out-bundle-dir",
        str(out_bundle_dir),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return env_parts + cmd


def _audit_dataset(args: argparse.Namespace) -> dict[str, Any]:
    audit_args = argparse.Namespace(
        dataset_dir=str(args.dataset_dir),
        stem=str(args.stem),
        out_dir=str(args.audit_out_dir),
        data_splits="train,val,test",
        max_rows_per_split=int(args.audit_max_rows_per_split),
        max_row_groups_per_split=int(args.audit_max_row_groups_per_split),
        support_dominance_min=0.25,
        min_pocket_rows=30,
        min_channel_position_delta=0.05,
        max_channel_position_support_corr=-0.05,
        require_rail_features=True,
        require_xau_provenance=True,
        fail_on_audit_fail=False,
        quiet=True,
    )
    return run_pretrain_audit(audit_args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.execute and not str(args.vedtak or "").strip():
        raise SystemExit("--execute requires --vedtak with an explicit SMART/SEQ520 decision id")
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    plan_out_dir = Path(args.plan_out_dir).expanduser().resolve()
    plan_out_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    dry_run = not bool(args.execute)
    vedtak = str(args.vedtak or "SMART_SEQ520_XAU_DIRECTION_REPAIR_SWEEP_DRY_RUN")

    audit_report = _audit_dataset(args)
    training_allowed = audit_report.get("decision") == "PASS"
    if args.execute and not training_allowed:
        print(
            json.dumps(
                {
                    "decision": "BLOCKED",
                    "reason": "pretrain audit failed; rebuild XAU dataset before sweep training",
                    "audit_failures": audit_report.get("failures") or [],
                    "audit_json_path": audit_report.get("json_path"),
                },
                indent=2,
                default=_json_default,
            )
        )
        raise SystemExit(2)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trials = sample_trials(trials=int(args.trials), seed=int(args.seed))
    rows: list[dict[str, Any]] = []
    for idx, env in enumerate(trials, start=1):
        trial_name = f"trial_{idx:03d}"
        bundle_dir = out_root / f"{timestamp}_{trial_name}"
        cmd = trial_command(
            trial_idx=idx,
            trial_env=env,
            dataset_dir=dataset_dir,
            out_bundle_dir=bundle_dir,
            vedtak=vedtak,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            subsample_rows=int(args.subsample_rows),
            seed=int(args.seed),
            dry_run=dry_run,
        )
        rows.append(
            {
                "trial": trial_name,
                "bundle_dir": str(bundle_dir),
                "env": env,
                "command": cmd,
                "command_string": shlex.join(cmd),
            }
        )

    plan = {
        "schema_version": "xau_smart_seq520_direction_repair_sweep_plan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry_run",
        "training_allowed": bool(training_allowed),
        "blocked_reason": None if training_allowed else "pretrain audit failed; rebuild XAU dataset before sweep training",
        "dataset_dir": str(dataset_dir),
        "stem": str(args.stem),
        "out_root": str(out_root),
        "audit": {
            "decision": audit_report.get("decision"),
            "json_path": audit_report.get("json_path"),
            "failures": audit_report.get("failures") or [],
        },
        "fixed_policy": {
            "xau_only": True,
            "runtime_direction_rules_added": False,
            "strict_core_repair_no_weaker_than_baseline": True,
            "anchor_gate_init_fixed": FIXED_ENV["ENTRY_FOUNDATION_CANDIDATE_ANCHOR_GATE_INIT"],
            "bad_path_prob_penalty_fixed": FIXED_ENV["ENTRY_FOUNDATION_CANDIDATE_BAD_PATH_PROB_PENALTY"],
        },
        "trials": rows,
    }
    plan_path = plan_out_dir / f"XAU_DIRECTION_REPAIR_SWEEP_PLAN_{timestamp}.json"
    latest_path = plan_out_dir / "XAU_DIRECTION_REPAIR_SWEEP_PLAN_latest.json"
    plan["plan_path"] = str(plan_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    latest_path.write_text(plan_path.read_text(encoding="utf-8"), encoding="utf-8")
    plan["latest_plan_path"] = str(latest_path)

    if dry_run:
        print(
            json.dumps(
                {
                    "decision": "PLAN_ONLY" if training_allowed else "BLOCKED_PLAN_ONLY",
                    "training_allowed": training_allowed,
                    "audit_failures": audit_report.get("failures") or [],
                    "plan_path": str(plan_path),
                    "first_command": rows[0]["command_string"] if rows else "",
                },
                indent=2,
                default=_json_default,
            )
        )
        return plan

    for row in rows:
        print(f"[XAU_SWEEP] starting {row['trial']} -> {row['bundle_dir']}", flush=True)
        env = os.environ.copy()
        env.update(row["env"])
        cmd = list(row["command"][1 + len(row["env"]) :])
        subprocess.run(cmd, cwd=REPO, env=env, check=True)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--plan-out-dir", type=Path, default=DEFAULT_PLAN_OUT)
    parser.add_argument("--audit-out-dir", type=Path, default=DEFAULT_PLAN_OUT / "pretrain_audit")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--subsample-rows", type=int, default=90000)
    parser.add_argument("--audit-max-rows-per-split", type=int, default=25000)
    parser.add_argument("--audit-max-row-groups-per-split", type=int, default=5)
    parser.add_argument("--vedtak", default="")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
