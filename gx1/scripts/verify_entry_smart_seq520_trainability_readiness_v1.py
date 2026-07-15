#!/usr/bin/env python3
"""Report-only smart_seq520 trainability readiness gate.

This gate is deliberately stricter than smoke-readiness. It does not train,
replay, distill IQL, rebuild data, or touch shadow/live paths. It only proves
whether the smart_seq520 candidate has a fully wired train/proof lane before a
future trainer can be reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_CONTRACT_MODES,
    required_training_specialists_for_mode,
    specialist_contract_training_allowed_for_mode,
)
from gx1.scripts.verify_entry_foundation_state_v1 import REPO, REPORTS_ROOT


CONTRACT_MODE = "smart_seq520_candidate"
EXPECTED_SIGNAL_DIM = 520
EXPECTED_SPECIALIST_COUNT = 8
EXPECTED_CTX_TAG = "CTX6CAT5"
EXPECTED_CTX_CONT_DIM = 142
EXPECTED_CTX_CAT_DIM = 5
READY_DECISION = "READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW"
BLOCKED_DECISION = "BLOCKED_SMART_SEQ520_TRAINABILITY_READINESS"
PATH_CALIBRATION_RECIPE_CONTRACT = {
    "path_quality_rank_full_batch": True,
    "path_quality_rank_weight": 2.0,
    "path_quality_rank_margin": 0.25,
    "path_quality_rank_quantile": 0.25,
    "bad_path_quality_rank_weight": 2.0,
    "bad_path_quality_rank_margin": 0.25,
    "bad_path_quality_rank_quantile": 0.25,
}
PATH_CALIBRATION_ENV_TEMPLATE = {
    "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT": "2.00",
    "ENTRY_BAD_PATH_QUALITY_RANK_MARGIN": "0.25",
    "ENTRY_BAD_PATH_QUALITY_RANK_QUANTILE": "0.25",
    "ENTRY_PATH_QUALITY_RANK_WEIGHT": "2.00",
    "ENTRY_PATH_QUALITY_RANK_MARGIN": "0.25",
    "ENTRY_PATH_QUALITY_RANK_QUANTILE": "0.25",
}
PATH_CALIBRATION_ENV_KEYS = tuple(PATH_CALIBRATION_ENV_TEMPLATE)
DIRECTION_BALANCE_RECIPE_CONTRACT = {
    "pred_balance_alpha": 0.50,
    "pred_balance_target": "label",
    "pred_balance_class_weights": [1.0, 1.0, 4.0],
    "direction_ce_scale": 4.00,
    "ckpt_monitor": "dir_acc",
    "ckpt_class_balance_guard_weight": 0.50,
    "ckpt_class_balance_min_pred_to_label": 0.35,
    "ckpt_class_balance_min_pred_rate": 0.05,
    "ckpt_direction_slice_guard": True,
    "direction_min_pred_rate_loss_weight": 12.00,
    "direction_min_pred_rate_fraction": 0.50,
    "direction_min_pred_rate_floor": 0.05,
    "direction_min_pred_rate_softmax_temperature": 0.05,
    "direction_global_prior_match_weight": 8.00,
    "direction_global_prior_match_tolerance": 0.02,
    "direction_global_prior_match_min_label_rate": 0.10,
    "direction_slice_min_pred_rate_loss_weight": 8.00,
    "direction_slice_min_pred_rate_fraction": 0.50,
    "direction_slice_min_pred_rate_floor": 0.05,
    "direction_slice_min_label_rate": 0.10,
    "direction_slice_min_rows": 8,
    "direction_slice_ctx_cat_indices": "0,1,2,3,4",
    "direction_slice_recall_loss_weight": 4.00,
    "direction_slice_recall_prob_floor": 0.30,
    "direction_slice_recall_min_label_rate": 0.10,
    "direction_slice_recall_min_rows": 8,
    "direction_slice_balanced_ce_weight": 2.00,
    "direction_slice_balanced_ce_min_label_rate": 0.10,
    "direction_slice_balanced_ce_min_rows": 8,
    "direction_slice_true_margin_weight": 2.00,
    "direction_slice_true_margin": 0.10,
    "direction_slice_true_margin_min_label_rate": 0.10,
    "direction_slice_true_margin_min_rows": 8,
    "direction_slice_accuracy_edge_weight": 4.00,
    "direction_slice_accuracy_edge_margin": 0.02,
    "direction_slice_accuracy_edge_min_label_rate": 0.10,
    "direction_slice_accuracy_edge_min_rows": 8,
    "direction_slice_prior_match_weight": 3.00,
    "direction_slice_prior_match_tolerance": 0.02,
    "direction_slice_prior_match_min_label_rate": 0.10,
    "direction_slice_prior_match_min_rows": 8,
    "direction_slice_loss_aggregation": "mean_max",
    "direction_slice_balanced_sampler": True,
    "direction_slice_balanced_sampler_min_rows": 8,
    "direction_slice_hard_red_stop_patience": 3,
    "direction_slice_hard_red_stop_min_epochs": 6,
    "direction_vs_flat_margin_weight": 4.00,
    "direction_vs_flat_margin": 0.10,
    "hier_legacy_ce_mult": 1.00,
    "hier_trade_weight": 2.00,
    "hier_side_weight": 1.75,
    "hier_utility_weight": 1.00,
    "hier_bad_path_weight": 1.25,
    "hier_mae_weight": 0.35,
    "residual_scale": 0.35,
    "anchor_eps": 1e-6,
    "hierarchical_entry_heads_enabled": True,
    "side_validity_head_enabled": True,
    "hier_side_validity_weight": 1.50,
    "hier_side_validity_min_utility_bps": 15.0,
    "hier_side_validity_pos_weight_cap": 8.0,
    "hier_pocket_abstain_weight": 5.00,
    "hier_pocket_side_margin_weight": 3.00,
    "hier_pocket_utility_margin_bps": 30.0,
    "trendline_rail_head_enabled": True,
    "trendline_rail_aux_weight": 1.00,
    "trendline_rail_wrong_side_weight": 1.50,
    "trendline_rail_rising_wrong_short_weight": 1.50,
    "trendline_rail_falling_wrong_long_weight": 1.75,
    "trendline_rail_final_margin_weight": 5.00,
    "trendline_rail_hier_margin_weight": 4.00,
    "trendline_rail_flat_trade_weight": 3.00,
    "trendline_rail_utility_margin_weight": 5.00,
    "trendline_rail_margin": 1.00,
    "trendline_rail_utility_margin_bps": 30.0,
    "anchor_gate_enabled": True,
    "anchor_gate_init": 0.0,
}
DIRECTION_BALANCE_ENV_TEMPLATE = {
    "ENTRY_PRED_BALANCE_ALPHA": "0.50",
    "ENTRY_PRED_BALANCE_TARGET": "label",
    "ENTRY_PRED_BALANCE_CLASS_WEIGHTS": "1.0,1.0,4.0",
    "ENTRY_DIRECTION_CE_SCALE": "4.00",
    "GX1_V10_CKPT_MONITOR": "dir_acc",
    "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT": "0.50",
    "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL": "0.35",
    "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE": "0.05",
    "ENTRY_CKPT_DIRECTION_SLICE_GUARD": "1",
    "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT": "12.00",
    "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION": "0.50",
    "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR": "0.05",
    "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE": "0.05",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT": "8.00",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT": "8.00",
    "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION": "0.50",
    "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR": "0.05",
    "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES": "0,1,2,3,4",
    "ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT": "4.00",
    "ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR": "0.30",
    "ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT": "2.00",
    "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT": "2.00",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN": "0.10",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT": "4.00",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN": "0.02",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT": "3.00",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION": "mean_max",
    "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER": "1",
    "ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_MIN_ROWS": "8",
    "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE": "3",
    "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS": "6",
    "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT": "4.00",
    "ENTRY_DIRECTION_VS_FLAT_MARGIN": "0.10",
    "ENTRY_HIER_LEGACY_CE_MULT": "1.00",
    "ENTRY_HIER_TRADE_WEIGHT": "2.00",
    "ENTRY_HIER_SIDE_WEIGHT": "1.75",
    "ENTRY_HIER_UTILITY_WEIGHT": "1.00",
    "ENTRY_HIER_BAD_PATH_WEIGHT": "1.25",
    "ENTRY_HIER_MAE_WEIGHT": "0.35",
    "ENTRY_RESIDUAL_SCALE": "0.35",
    "ENTRY_ANCHOR_EPS": "1e-6",
    "ENTRY_HIER_SIDE_VALIDITY_WEIGHT": "1.50",
    "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS": "15.0",
    "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP": "8.0",
    "ENTRY_HIER_POCKET_ABSTAIN_WEIGHT": "5.00",
    "ENTRY_HIER_POCKET_SIDE_MARGIN_WEIGHT": "3.00",
    "ENTRY_HIER_POCKET_UTILITY_MARGIN_BPS": "30.0",
    "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT": "1.00",
    "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT": "1.50",
    "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT": "1.50",
    "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT": "1.75",
    "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT": "5.00",
    "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT": "4.00",
    "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT": "3.00",
    "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT": "5.00",
    "ENTRY_TRENDLINE_RAIL_MARGIN": "1.00",
    "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS": "30.0",
}
DIRECTION_BALANCE_ENV_KEYS = tuple(DIRECTION_BALANCE_ENV_TEMPLATE)
TAIL_DIRECTION_RECIPE_CONTRACT = {
    "tail_direction_ce_weight": 0.35,
    "tail_direction_quality_quantile": 0.70,
    "tail_direction_min_batch": 8,
    "tail_direction_mask": "directional_tradable_clean_path_top_quality",
}
TAIL_DIRECTION_ENV_TEMPLATE = {
    "ENTRY_TAIL_DIRECTION_CE_WEIGHT": "0.35",
    "ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE": "0.70",
    "ENTRY_TAIL_DIRECTION_MIN_BATCH": "8",
}
TAIL_DIRECTION_ENV_KEYS = tuple(TAIL_DIRECTION_ENV_TEMPLATE)
DIRECTION_CONTEXT_SLICE_CONTRACT = {
    "source": "post_smoke_audit.direction_slice_contract",
    "ctx_cat_names": ["session_id", "vol_regime_id", "atr_bucket", "spread_bucket", "H4_trend_sign_cat"],
    "min_rows": 64,
    "requires_majority_baseline": True,
    "requires_class_distribution_coverage": True,
    "skips_low_label_diversity": True,
}

DEFAULT_POST_REBUILD_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_smart_dataset_post_rebuild_readiness_20260630_v1"
    / "ENTRY_SMART_DATASET_POST_REBUILD_READINESS_latest.json"
)
DEFAULT_SMOKE_READINESS_JSON = (
    REPORTS_ROOT
    / "entry_smart_seq520_smoke_readiness_20260630_v1"
    / "ENTRY_SMART_SEQ520_SMOKE_READINESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_seq520_trainability_readiness_20260630_v1"
DEFAULT_CONTROL_SCRIPT = REPO / "scripts/entry_next_edge_control.sh"
DEFAULT_TRAINER_SOURCE = REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
DEFAULT_SMOKE_WRAPPER = REPO / "scripts/run_entry_foundation_seq146_smoke_train.sh"
DEFAULT_CANDIDATE_WRAPPER = REPO / "scripts/run_entry_foundation_seq146_candidate_train.sh"
DEFAULT_CANDIDATE_READINESS = REPO / "gx1/scripts/verify_entry_candidate_readiness_v1.py"
DEFAULT_SELECTIVE_EDGE = REPO / "gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py"
DEFAULT_REPLAY_EVIDENCE = REPO / "gx1/scripts/materialize_entry_candidate_replay_evidence_v1.py"
DEFAULT_REPLAY_READINESS = REPO / "gx1/scripts/verify_entry_replay_readiness_v1.py"

SIDE_EFFECTS_CLOSED = {
    "dataset_rebuild": False,
    "training": False,
    "replay": False,
    "iql_distillation": False,
    "shadow": False,
    "live": False,
    "promotion": False,
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "sha256": _sha256_file(path),
    }


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details if details is not None else {}}


def _future_train_contract(smoke_readiness: dict[str, Any]) -> dict[str, Any]:
    contracts = smoke_readiness.get("future_command_contracts")
    if not isinstance(contracts, dict):
        return {}
    contract = contracts.get("smart_smoke_train")
    return contract if isinstance(contract, dict) else {}


def _path_calibration_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("path_calibration_recipe_contract")
    env_template = contract.get("path_calibration_env_template")
    recipe_exact = isinstance(recipe, dict) and recipe == PATH_CALIBRATION_RECIPE_CONTRACT
    env_exact = isinstance(env_template, dict) and all(
        env_template.get(key) == value for key, value in PATH_CALIBRATION_ENV_TEMPLATE.items()
    )
    argv = contract.get("inner_train_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_env = all(f"{key}=" in argv_text for key in PATH_CALIBRATION_ENV_KEYS)
    return {
        "ok": bool(
            contract.get("requires_path_calibration_recipe_contract") is True
            and recipe_exact
            and env_exact
            and argv_declares_env
        ),
        "requires_path_calibration_recipe_contract": contract.get("requires_path_calibration_recipe_contract"),
        "recipe_exact": recipe_exact,
        "env_template_exact": env_exact,
        "inner_train_argv_declares_env": argv_declares_env,
        "expected_recipe": PATH_CALIBRATION_RECIPE_CONTRACT,
        "observed_recipe": recipe,
        "expected_env_template": PATH_CALIBRATION_ENV_TEMPLATE,
        "observed_env_template": env_template,
    }


def _direction_balance_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("direction_balance_recipe_contract")
    env_template = contract.get("direction_balance_env_template")
    recipe_exact = isinstance(recipe, dict) and recipe == DIRECTION_BALANCE_RECIPE_CONTRACT
    env_exact = isinstance(env_template, dict) and all(
        env_template.get(key) == value for key, value in DIRECTION_BALANCE_ENV_TEMPLATE.items()
    )
    argv = contract.get("inner_train_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_env = all(f"{key}=" in argv_text for key in DIRECTION_BALANCE_ENV_KEYS)
    argv_declares_xau_repair_heads = (
        "--enable-xau-direction-repair-heads" in argv_text
        and "--anchor-gate-init 0.0" in argv_text
    )
    return {
        "ok": bool(
            contract.get("requires_direction_balance_recipe_contract") is True
            and recipe_exact
            and env_exact
            and argv_declares_env
            and argv_declares_xau_repair_heads
        ),
        "requires_direction_balance_recipe_contract": contract.get("requires_direction_balance_recipe_contract"),
        "recipe_exact": recipe_exact,
        "env_template_exact": env_exact,
        "inner_train_argv_declares_env": argv_declares_env,
        "inner_train_argv_declares_xau_repair_heads": argv_declares_xau_repair_heads,
        "expected_recipe": DIRECTION_BALANCE_RECIPE_CONTRACT,
        "observed_recipe": recipe,
        "expected_env_template": DIRECTION_BALANCE_ENV_TEMPLATE,
        "observed_env_template": env_template,
    }


def _tail_direction_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("tail_direction_recipe_contract")
    env_template = contract.get("tail_direction_env_template")
    recipe_exact = isinstance(recipe, dict) and recipe == TAIL_DIRECTION_RECIPE_CONTRACT
    env_exact = isinstance(env_template, dict) and all(
        env_template.get(key) == value for key, value in TAIL_DIRECTION_ENV_TEMPLATE.items()
    )
    argv = contract.get("inner_train_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_env = all(f"{key}=" in argv_text for key in TAIL_DIRECTION_ENV_KEYS)
    return {
        "ok": bool(
            contract.get("requires_tail_direction_recipe_contract") is True
            and recipe_exact
            and env_exact
            and argv_declares_env
        ),
        "requires_tail_direction_recipe_contract": contract.get("requires_tail_direction_recipe_contract"),
        "recipe_exact": recipe_exact,
        "env_template_exact": env_exact,
        "inner_train_argv_declares_env": argv_declares_env,
        "expected_recipe": TAIL_DIRECTION_RECIPE_CONTRACT,
        "observed_recipe": recipe,
        "expected_env_template": TAIL_DIRECTION_ENV_TEMPLATE,
        "observed_env_template": env_template,
    }


def _direction_context_slice_review(contract: dict[str, Any]) -> dict[str, Any]:
    observed = contract.get("direction_context_slice_contract")
    exact = isinstance(observed, dict) and observed == DIRECTION_CONTEXT_SLICE_CONTRACT
    return {
        "ok": bool(contract.get("requires_direction_context_slice_contract") is True and exact),
        "requires_direction_context_slice_contract": contract.get("requires_direction_context_slice_contract"),
        "contract_exact": exact,
        "expected_contract": DIRECTION_CONTEXT_SLICE_CONTRACT,
        "observed_contract": observed,
    }


def _wrapper_path_calibration_env_review(text: str) -> dict[str, Any]:
    missing_entry_env = [key for key in PATH_CALIBRATION_ENV_KEYS if key not in text]
    missing_foundation_env = [
        key.replace("ENTRY_", "ENTRY_FOUNDATION_SMOKE_")
        for key in PATH_CALIBRATION_ENV_KEYS
        if key.replace("ENTRY_", "ENTRY_FOUNDATION_SMOKE_") not in text
    ]
    missing_candidate_env = [
        key.replace("ENTRY_", "ENTRY_FOUNDATION_CANDIDATE_")
        for key in PATH_CALIBRATION_ENV_KEYS
        if key.replace("ENTRY_", "ENTRY_FOUNDATION_CANDIDATE_") not in text
    ]
    return {
        "entry_env_keys": list(PATH_CALIBRATION_ENV_KEYS),
        "missing_entry_env_keys": missing_entry_env,
        "missing_foundation_smoke_env_keys": missing_foundation_env,
        "missing_foundation_candidate_env_keys": missing_candidate_env,
        "has_any_foundation_defaults": not missing_foundation_env or not missing_candidate_env,
        "ok": not missing_entry_env and (not missing_foundation_env or not missing_candidate_env),
    }


def _direction_balance_foundation_env_name(key: str, prefix: str) -> str:
    if key.startswith("ENTRY_"):
        return key.replace("ENTRY_", prefix, 1)
    if key.startswith("GX1_V10_"):
        return key.replace("GX1_V10_", prefix, 1)
    return key


def _wrapper_direction_balance_env_review(text: str) -> dict[str, Any]:
    missing_entry_env = [key for key in DIRECTION_BALANCE_ENV_KEYS if key not in text]
    missing_foundation_env = [
        _direction_balance_foundation_env_name(key, "ENTRY_FOUNDATION_SMOKE_")
        for key in DIRECTION_BALANCE_ENV_KEYS
        if _direction_balance_foundation_env_name(key, "ENTRY_FOUNDATION_SMOKE_") not in text
    ]
    missing_candidate_env = [
        _direction_balance_foundation_env_name(key, "ENTRY_FOUNDATION_CANDIDATE_")
        for key in DIRECTION_BALANCE_ENV_KEYS
        if _direction_balance_foundation_env_name(key, "ENTRY_FOUNDATION_CANDIDATE_") not in text
    ]
    expected_smart_fragments = [
        "PRED_BALANCE_ALPHA=0.50",
        "PRED_BALANCE_CLASS_WEIGHTS=1.0,1.0,4.0",
        "DIRECTION_CE_SCALE=4.00",
        "CKPT_CLASS_BALANCE_GUARD_WEIGHT=0.50",
        "CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL=0.35",
        "CKPT_CLASS_BALANCE_MIN_PRED_RATE=0.05",
        "DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT=12.00",
        "DIRECTION_MIN_PRED_RATE_FRACTION=0.50",
        "DIRECTION_MIN_PRED_RATE_FLOOR=0.05",
        "DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE=0.05",
        "DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT=8.00",
        "DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE=0.02",
        "DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE=0.10",
        "DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT=8.00",
        "DIRECTION_SLICE_MIN_PRED_RATE_FRACTION=0.50",
        "DIRECTION_SLICE_MIN_PRED_RATE_FLOOR=0.05",
        "DIRECTION_SLICE_RECALL_LOSS_WEIGHT=4.00",
        "DIRECTION_SLICE_BALANCED_CE_WEIGHT=2.00",
        "DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE=0.10",
        "DIRECTION_SLICE_BALANCED_CE_MIN_ROWS=8",
        "DIRECTION_SLICE_TRUE_MARGIN_WEIGHT=2.00",
        "DIRECTION_SLICE_TRUE_MARGIN=0.10",
        "DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE=0.10",
        "DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS=8",
        "DIRECTION_SLICE_LOSS_AGGREGATION=mean_max",
        "DIRECTION_VS_FLAT_MARGIN_WEIGHT=4.00",
        "DIRECTION_VS_FLAT_MARGIN=0.10",
        "HIER_LEGACY_CE_MULT=1.00",
        "HIER_SIDE_VALIDITY_WEIGHT=1.50",
        "HIER_POCKET_ABSTAIN_WEIGHT=5.00",
        "HIER_POCKET_SIDE_MARGIN_WEIGHT=3.00",
        "TRENDLINE_RAIL_AUX_WEIGHT=1.00",
        "TRENDLINE_RAIL_WRONG_SIDE_WEIGHT=1.50",
        "TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT=5.00",
        "ANCHOR_GATE_INIT=0.0",
        "--enable-xau-direction-repair-heads",
        "anchor-gate-init",
    ]
    missing_expected_smart_fragments = [
        fragment for fragment in expected_smart_fragments if fragment not in text
    ]
    return {
        "entry_env_keys": list(DIRECTION_BALANCE_ENV_KEYS),
        "missing_entry_env_keys": missing_entry_env,
        "missing_foundation_smoke_env_keys": missing_foundation_env,
        "missing_foundation_candidate_env_keys": missing_candidate_env,
        "expected_smart_fragments": expected_smart_fragments,
        "missing_expected_smart_fragments": missing_expected_smart_fragments,
        "has_any_foundation_defaults": not missing_foundation_env or not missing_candidate_env,
        "ok": (
            not missing_entry_env
            and (not missing_foundation_env or not missing_candidate_env)
            and not missing_expected_smart_fragments
        ),
    }


def _wrapper_tail_direction_env_review(text: str) -> dict[str, Any]:
    missing_entry_env = [key for key in TAIL_DIRECTION_ENV_KEYS if key not in text]
    missing_foundation_env = [
        key.replace("ENTRY_", "ENTRY_FOUNDATION_SMOKE_", 1)
        for key in TAIL_DIRECTION_ENV_KEYS
        if key.replace("ENTRY_", "ENTRY_FOUNDATION_SMOKE_", 1) not in text
    ]
    missing_candidate_env = [
        key.replace("ENTRY_", "ENTRY_FOUNDATION_CANDIDATE_", 1)
        for key in TAIL_DIRECTION_ENV_KEYS
        if key.replace("ENTRY_", "ENTRY_FOUNDATION_CANDIDATE_", 1) not in text
    ]
    return {
        "entry_env_keys": list(TAIL_DIRECTION_ENV_KEYS),
        "missing_entry_env_keys": missing_entry_env,
        "missing_foundation_smoke_env_keys": missing_foundation_env,
        "missing_foundation_candidate_env_keys": missing_candidate_env,
        "has_any_foundation_defaults": not missing_foundation_env or not missing_candidate_env,
        "ok": not missing_entry_env and (not missing_foundation_env or not missing_candidate_env),
    }


def _walk_json(value: Any, *, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_json(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            yield from _walk_json(item, path=f"{path}[{idx}]")


def _contains_exact_string(payloads: dict[str, Any], needle: str) -> list[str]:
    matches: list[str] = []
    for root, payload in payloads.items():
        for path, value in _walk_json(payload, path=root):
            if value == needle:
                matches.append(path)
    return matches


def _ctx_contract_rows(payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root, payload in payloads.items():
        for path, value in _walk_json(payload, path=root):
            if not isinstance(value, dict):
                continue
            if "ctx_contract" in value and isinstance(value.get("ctx_contract"), dict):
                ctx = value["ctx_contract"]
                rows.append(
                    {
                        "path": f"{path}.ctx_contract",
                        "tag": ctx.get("tag") or ctx.get("ctx_tag"),
                        "ctx_cont_dim": ctx.get("ctx_cont_dim"),
                        "ctx_cat_dim": ctx.get("ctx_cat_dim"),
                    }
                )
            elif ("ctx_tag" in value or "tag" in value) and (
                "ctx_cont_dim" in value or "ctx_cat_dim" in value
            ):
                rows.append(
                    {
                        "path": path,
                        "tag": value.get("ctx_tag") or value.get("tag"),
                        "ctx_cont_dim": value.get("ctx_cont_dim"),
                        "ctx_cat_dim": value.get("ctx_cat_dim"),
                    }
                )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row.get("path"), row.get("tag"), row.get("ctx_cont_dim"), row.get("ctx_cat_dim"))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _ctx_metadata_contract(payloads: dict[str, Any]) -> dict[str, Any]:
    rows = _ctx_contract_rows(payloads)
    stale_ctx6cat6_paths = _contains_exact_string(payloads, "CTX6CAT6")
    declared_rows = [row for row in rows if row.get("tag") or row.get("ctx_cont_dim") or row.get("ctx_cat_dim")]
    mismatched_rows = [
        row
        for row in declared_rows
        if not (
            row.get("tag") == EXPECTED_CTX_TAG
            and int(row.get("ctx_cont_dim") or 0) == EXPECTED_CTX_CONT_DIM
            and int(row.get("ctx_cat_dim") or 0) == EXPECTED_CTX_CAT_DIM
        )
    ]
    return {
        "expected": {
            "ctx_tag": EXPECTED_CTX_TAG,
            "ctx_cont_dim": EXPECTED_CTX_CONT_DIM,
            "ctx_cat_dim": EXPECTED_CTX_CAT_DIM,
        },
        "declared_ctx_contract_count": int(len(declared_rows)),
        "declared_ctx_contracts": declared_rows,
        "mismatched_ctx_contracts": mismatched_rows,
        "stale_ctx6cat6_paths": stale_ctx6cat6_paths,
        "no_stale_ctx6cat6": not stale_ctx6cat6_paths,
        "declared_ctx_contracts_match_expected": not mismatched_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    post_rebuild_json = Path(args.smart_post_rebuild_readiness_json).expanduser().resolve()
    smoke_readiness_json = Path(args.smart_smoke_readiness_json).expanduser().resolve()
    control_script = Path(args.control_script).expanduser().resolve()
    trainer_source = Path(args.trainer_source).expanduser().resolve()
    smoke_wrapper = Path(args.smoke_wrapper).expanduser().resolve()
    candidate_wrapper = Path(args.candidate_wrapper).expanduser().resolve()
    candidate_readiness_script = Path(args.candidate_readiness_script).expanduser().resolve()
    selective_edge_script = Path(args.selective_edge_script).expanduser().resolve()
    replay_evidence_script = Path(args.replay_evidence_script).expanduser().resolve()
    replay_readiness_script = Path(args.replay_readiness_script).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    post_rebuild = _read_json_or_empty(post_rebuild_json)
    smoke_readiness = _read_json_or_empty(smoke_readiness_json)
    future_train = _future_train_contract(smoke_readiness)
    source_metadata_contract = _ctx_metadata_contract(
        {
            "smart_post_rebuild_readiness": post_rebuild,
            "smart_smoke_readiness": smoke_readiness,
        }
    )

    control_text = _read_text(control_script)
    trainer_text = _read_text(trainer_source)
    smoke_wrapper_text = _read_text(smoke_wrapper)
    candidate_wrapper_text = _read_text(candidate_wrapper)
    candidate_readiness_text = _read_text(candidate_readiness_script)
    selective_edge_text = _read_text(selective_edge_script)
    replay_evidence_text = _read_text(replay_evidence_script)
    replay_readiness_text = _read_text(replay_readiness_script)

    try:
        required_specialists = tuple(required_training_specialists_for_mode(CONTRACT_MODE))
    except Exception:
        required_specialists = ()
    try:
        registry_training_allowed = specialist_contract_training_allowed_for_mode(CONTRACT_MODE)
    except Exception:
        registry_training_allowed = False
    path_calibration_review = _path_calibration_recipe_review(future_train)
    direction_balance_review = _direction_balance_recipe_review(future_train)
    tail_direction_review = _tail_direction_recipe_review(future_train)
    direction_context_slice_review = _direction_context_slice_review(future_train)
    smoke_wrapper_path_calibration_review = _wrapper_path_calibration_env_review(smoke_wrapper_text)
    candidate_wrapper_path_calibration_review = _wrapper_path_calibration_env_review(candidate_wrapper_text)
    smoke_wrapper_direction_balance_review = _wrapper_direction_balance_env_review(smoke_wrapper_text)
    candidate_wrapper_direction_balance_review = _wrapper_direction_balance_env_review(candidate_wrapper_text)
    smoke_wrapper_tail_direction_review = _wrapper_tail_direction_env_review(smoke_wrapper_text)
    candidate_wrapper_tail_direction_review = _wrapper_tail_direction_env_review(candidate_wrapper_text)

    checks = [
        _check("smart post-rebuild dataset audit is ready", post_rebuild.get("decision") == "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW", post_rebuild.get("decision")),
        _check("smart smoke readiness is ready", smoke_readiness.get("decision") == "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW", smoke_readiness.get("decision")),
        _check("smart specialist mode is accepted by trainer contract modes", CONTRACT_MODE in SPECIALIST_CONTRACT_MODES, list(SPECIALIST_CONTRACT_MODES)),
        _check(
            "smart specialist registry is trainable only through explicit candidate gate",
            registry_training_allowed is True,
            {
                "registry_training_allowed": registry_training_allowed,
                "candidate_training_allowed_by_this_report": False,
                "requires_candidate_readiness_and_explicit_vedtak": True,
            },
        ),
        _check("smart required specialist count is eight", len(required_specialists) == EXPECTED_SPECIALIST_COUNT, list(required_specialists)),
        _check("trainer CLI can accept smart specialist contract mode", CONTRACT_MODE in SPECIALIST_CONTRACT_MODES and "specialist-contract-mode" in trainer_text, _artifact_meta(trainer_source)),
        _check(
            "trainer ctx contract is not hard-coded to stale CTX6CAT6 for smart",
            "CTX6CAT6" not in trainer_text,
            _artifact_meta(trainer_source),
        ),
        _check(
            "smart source metadata has no stale CTX6CAT6 ctx contract",
            source_metadata_contract["no_stale_ctx6cat6"],
            source_metadata_contract,
        ),
        _check(
            "declared smart source ctx metadata matches CTX6CAT5",
            source_metadata_contract["declared_ctx_contracts_match_expected"],
            source_metadata_contract,
        ),
        _check("smart smoke wrapper exposes --smart-seq520 lane", "--smart-seq520" in smoke_wrapper_text and CONTRACT_MODE in smoke_wrapper_text, _artifact_meta(smoke_wrapper)),
        _check(
            "smart smoke wrapper exposes path calibration rank env",
            bool(smoke_wrapper_path_calibration_review["ok"]),
            smoke_wrapper_path_calibration_review,
        ),
        _check(
            "smart smoke wrapper exposes direction balance env",
            bool(smoke_wrapper_direction_balance_review["ok"]),
            smoke_wrapper_direction_balance_review,
        ),
        _check(
            "smart smoke wrapper exposes tail direction env",
            bool(smoke_wrapper_tail_direction_review["ok"]),
            smoke_wrapper_tail_direction_review,
        ),
        _check("smart smoke train is wired in control surface", "smart-smoke-train)" in control_text and "smart-smoke-train --vedtak <id>" in control_text, _artifact_meta(control_script)),
        _check("smart smoke future contract is implemented in control surface", future_train.get("implemented_in_control_surface") is True, future_train),
        _check(
            "smart smoke future contract declares path calibration rank recipe",
            bool(path_calibration_review["ok"]),
            path_calibration_review,
        ),
        _check(
            "smart smoke future contract declares direction balance recipe",
            bool(direction_balance_review["ok"]),
            direction_balance_review,
        ),
        _check(
            "smart smoke future contract declares tail direction recipe",
            bool(tail_direction_review["ok"]),
            tail_direction_review,
        ),
        _check(
            "smart smoke future contract declares direction context slice audit",
            bool(direction_context_slice_review["ok"]),
            direction_context_slice_review,
        ),
        _check(
            "trainer supports path calibration rank env",
            all(key in trainer_text for key in PATH_CALIBRATION_ENV_KEYS),
            {"required_env_keys": list(PATH_CALIBRATION_ENV_KEYS), "trainer_source": _artifact_meta(trainer_source)},
        ),
        _check(
            "trainer supports direction balance env",
            all(key in trainer_text for key in DIRECTION_BALANCE_ENV_KEYS),
            {"required_env_keys": list(DIRECTION_BALANCE_ENV_KEYS), "trainer_source": _artifact_meta(trainer_source)},
        ),
        _check(
            "trainer supports tail direction env",
            all(key in trainer_text for key in TAIL_DIRECTION_ENV_KEYS),
            {"required_env_keys": list(TAIL_DIRECTION_ENV_KEYS), "trainer_source": _artifact_meta(trainer_source)},
        ),
        _check("smart candidate-readiness supports smart_seq520", CONTRACT_MODE in candidate_readiness_text and str(EXPECTED_SIGNAL_DIM) in candidate_readiness_text, _artifact_meta(candidate_readiness_script)),
        _check("smart candidate train wrapper exposes --smart-seq520 lane", "--smart-seq520" in candidate_wrapper_text and CONTRACT_MODE in candidate_wrapper_text, _artifact_meta(candidate_wrapper)),
        _check(
            "smart candidate train wrapper exposes path calibration rank env",
            bool(candidate_wrapper_path_calibration_review["ok"]),
            candidate_wrapper_path_calibration_review,
        ),
        _check(
            "smart candidate train wrapper exposes direction balance env",
            bool(candidate_wrapper_direction_balance_review["ok"]),
            candidate_wrapper_direction_balance_review,
        ),
        _check(
            "smart candidate train wrapper exposes tail direction env",
            bool(candidate_wrapper_tail_direction_review["ok"]),
            candidate_wrapper_tail_direction_review,
        ),
        _check("smart selective-edge supports smart_seq520", CONTRACT_MODE in selective_edge_text and str(EXPECTED_SIGNAL_DIM) in selective_edge_text, _artifact_meta(selective_edge_script)),
        _check("smart replay evidence supports smart_seq520", CONTRACT_MODE in replay_evidence_text and str(EXPECTED_SIGNAL_DIM) in replay_evidence_text, _artifact_meta(replay_evidence_script)),
        _check("smart replay-readiness supports smart_seq520", CONTRACT_MODE in replay_readiness_text and str(EXPECTED_SIGNAL_DIM) in replay_readiness_text, _artifact_meta(replay_readiness_script)),
        _check("side effects remain closed", all(value is False for value in SIDE_EFFECTS_CLOSED.values()), SIDE_EFFECTS_CLOSED),
    ]
    failures = [row for row in checks if not row["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_{stamp}.json"
    md_path = out_dir / f"ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_{stamp}.md"
    report = {
        "schema_version": "entry_smart_seq520_trainability_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "manifest_variant": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
        "required_training_specialists": list(required_specialists),
        "inputs": {
            "smart_post_rebuild_readiness": _artifact_meta(post_rebuild_json),
            "smart_smoke_readiness": _artifact_meta(smoke_readiness_json),
            "control_script": _artifact_meta(control_script),
            "trainer_source": _artifact_meta(trainer_source),
            "smoke_wrapper": _artifact_meta(smoke_wrapper),
            "candidate_wrapper": _artifact_meta(candidate_wrapper),
            "candidate_readiness_script": _artifact_meta(candidate_readiness_script),
            "selective_edge_script": _artifact_meta(selective_edge_script),
            "replay_evidence_script": _artifact_meta(replay_evidence_script),
            "replay_readiness_script": _artifact_meta(replay_readiness_script),
        },
        "future_train_contract": future_train,
        "source_metadata_contract": source_metadata_contract,
        "checks": checks,
        "failures": failures,
        "blockers": [row["name"] for row in failures],
        "training_allowed": False,
        "candidate_training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_promotion_allowed": False,
        "execution_allowed_now": False,
        "side_effects_started": dict(SIDE_EFFECTS_CLOSED),
        "next_required_gate": (
            "review explicit smart smoke-train implementation package"
            if ready
            else "wire smart_seq520 trainer/wrapper/control/candidate/replay surfaces before any smart training vedtak"
        ),
        "json_path": str(json_path),
        "md_path": str(md_path),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    latest_json = out_dir / "ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.json"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    md_lines = [
        "# Entry Smart Seq520 Trainability Readiness",
        "",
        f"- decision: `{decision}`",
        "- report_only: `true`",
        "- training_allowed: `false`",
        f"- failures: `{len(failures)}`",
        "",
        "## Failures",
        "",
    ]
    md_lines.extend([f"- `{row['name']}`" for row in failures] or ["- None"])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "ENTRY_SMART_SEQ520_TRAINABILITY_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures and bool(args.fail_on_not_ready):
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-post-rebuild-readiness-json", default=str(DEFAULT_POST_REBUILD_READINESS_JSON))
    ap.add_argument("--smart-smoke-readiness-json", default=str(DEFAULT_SMOKE_READINESS_JSON))
    ap.add_argument("--control-script", default=str(DEFAULT_CONTROL_SCRIPT))
    ap.add_argument("--trainer-source", default=str(DEFAULT_TRAINER_SOURCE))
    ap.add_argument("--smoke-wrapper", default=str(DEFAULT_SMOKE_WRAPPER))
    ap.add_argument("--candidate-wrapper", default=str(DEFAULT_CANDIDATE_WRAPPER))
    ap.add_argument("--candidate-readiness-script", default=str(DEFAULT_CANDIDATE_READINESS))
    ap.add_argument("--selective-edge-script", default=str(DEFAULT_SELECTIVE_EDGE))
    ap.add_argument("--replay-evidence-script", default=str(DEFAULT_REPLAY_EVIDENCE))
    ap.add_argument("--replay-readiness-script", default=str(DEFAULT_REPLAY_READINESS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--fail-on-not-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
