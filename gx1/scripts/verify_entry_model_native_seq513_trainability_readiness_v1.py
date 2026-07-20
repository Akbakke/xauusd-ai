#!/usr/bin/env python3
"""Report-only model-native seq513 trainability readiness evidence gate.

This gate is deliberately stricter than smoke-readiness. It does not train,
replay, distill IQL, rebuild data, or touch shadow/live paths. It only proves
whether the model-native seq513 candidate has a fully wired train/proof lane before a
future trainer can be reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_full_input_liveness_v1 import (
    SCHEMA_VERSION as FULL_INPUT_LIVENESS_SCHEMA,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    RECIPE_AUDIT_SCHEMA,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    SCHEMA_VERSION as TRAINING_OBJECTIVE_SCHEMA,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_CONTRACT_MODES,
    required_training_specialists_for_mode,
    specialist_contract_training_allowed_for_mode,
)
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
CONTRACT_MODE = MODEL_NATIVE_CONTRACT_MODE
EXPECTED_SIGNAL_DIM = MODEL_NATIVE_SIGNAL_DIM
EXPECTED_SPECIALIST_COUNT = 8
EXPECTED_CTX_TAG = "CTX6CAT5"
EXPECTED_CTX_CONT_DIM = 142
EXPECTED_CTX_CAT_DIM = 5
READY_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW"
BLOCKED_DECISION = "BLOCKED_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_TRAINABILITY_READINESS"
_TIMESTAMPED_JSON_RE = re.compile(
    r"^.+_\d{8}T\d{6}(?:\d{6})?Z\.json$"
)
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
    "direction_slice_confusion_pair_weight": 4.00,
    "direction_slice_confusion_pair_margin": 0.02,
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
    "direction_utility_margin_weight": 4.00,
    "direction_utility_min_gap_bps": 15.0,
    "direction_utility_logit_margin": 0.10,
    "direction_side_utility_conviction_weight": 6.00,
    "direction_side_utility_conviction_min_gap_bps": 15.0,
    "direction_side_utility_conviction_logit_margin": 0.10,
    "direction_utility_trade_conviction_weight": 8.00,
    "direction_utility_trade_conviction_min_gap_bps": 15.0,
    "direction_utility_trade_conviction_min_utility_bps": 0.0,
    "direction_utility_trade_conviction_max_bad_path": 0.50,
    "direction_utility_trade_conviction_logit_margin": 0.10,
    "direction_utility_triad_ce_weight": 8.00,
    "direction_utility_triad_ce_min_gap_bps": 15.0,
    "direction_utility_triad_ce_min_utility_bps": 0.0,
    "direction_utility_triad_ce_max_bad_path": 0.50,
    "direction_utility_triad_ce_class_weight_cap": 4.0,
    "hier_trade_global_prior_match_weight": 4.00,
    "hier_trade_global_prior_match_tolerance": 0.02,
    "hier_trade_global_prior_match_min_label_rate": 0.10,
    "hier_slice_trade_prior_match_weight": 4.00,
    "hier_slice_trade_prior_match_tolerance": 0.02,
    "hier_slice_trade_prior_match_min_label_rate": 0.10,
    "hier_slice_trade_prior_match_min_rows": 8,
    "hier_flat_logit_margin_weight": 8.00,
    "hier_flat_logit_margin": 0.10,
    "hier_flat_logit_margin_min_label_rate": 0.10,
    "hier_slice_flat_logit_margin_weight": 8.00,
    "hier_slice_flat_logit_margin": 0.10,
    "hier_slice_flat_logit_margin_min_label_rate": 0.10,
    "hier_slice_flat_logit_margin_min_rows": 8,
    "hier_slice_side_ce_weight": 4.00,
    "hier_slice_side_true_margin_weight": 3.00,
    "hier_slice_side_true_margin": 0.10,
    "hier_slice_side_min_label_rate": 0.10,
    "hier_slice_side_min_rows": 8,
    "hier_side_global_prior_match_weight": 4.00,
    "hier_side_global_prior_match_tolerance": 0.02,
    "hier_side_global_prior_match_min_label_rate": 0.10,
    "hier_slice_side_prior_match_weight": 4.00,
    "hier_slice_side_prior_match_tolerance": 0.02,
    "hier_slice_side_prior_match_min_label_rate": 0.10,
    "hier_slice_side_prior_match_min_rows": 8,
    "direction_flat_starvation_weight": 8.00,
    "direction_flat_starvation_min_label_rate": 0.10,
    "direction_flat_starvation_min_rows": 8,
    "direction_flat_starvation_pred_fraction": 0.50,
    "direction_flat_starvation_pred_floor": 0.10,
    "direction_flat_starvation_logit_margin": 0.10,
    "hier_trade_weight": 2.00,
    "hier_side_weight": 1.75,
    "hier_utility_weight": 1.00,
    "hier_bad_path_weight": 1.25,
    "hier_mae_weight": 0.35,
    "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    "hierarchical_entry_heads_enabled": True,
    "side_validity_head_enabled": True,
    "hier_side_validity_weight": 1.50,
    "hier_side_validity_min_utility_bps": 15.0,
    "hier_side_validity_pos_weight_cap": 8.0,
    "trendline_rail_head_enabled": True,
    "trendline_rail_aux_weight": 1.00,
    "trendline_rail_utility_margin_weight": 5.00,
    "trendline_rail_margin": 1.00,
    "trendline_rail_utility_margin_bps": 30.0,
    "anchor_gate_enabled": False,
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
    "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT": "4.00",
    "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN": "0.02",
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
    "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT": "4.00",
    "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN": "0.10",
    "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT": "6.00",
    "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN": "0.10",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT": "8.00",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS": "0.0",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH": "0.50",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN": "0.10",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT": "8.00",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS": "15.0",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS": "0.0",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH": "0.50",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP": "4.0",
    "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT": "8.00",
    "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE": "0.10",
    "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS": "8",
    "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION": "0.50",
    "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR": "0.10",
    "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN": "0.10",
    "ENTRY_HIER_TRADE_WEIGHT": "2.00",
    "ENTRY_HIER_SIDE_WEIGHT": "1.75",
    "ENTRY_HIER_UTILITY_WEIGHT": "1.00",
    "ENTRY_HIER_BAD_PATH_WEIGHT": "1.25",
    "ENTRY_HIER_MAE_WEIGHT": "0.35",
    "ENTRY_HIER_SIDE_VALIDITY_WEIGHT": "1.50",
    "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS": "15.0",
    "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP": "8.0",
    "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT": "4.00",
    "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT": "4.00",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT": "4.00",
    "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN": "0.02",
    "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT": "8.00",
    "ENTRY_HIER_FLAT_LOGIT_MARGIN": "0.10",
    "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT": "8.00",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN": "0.10",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS": "8",
    "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT": "4.00",
    "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT": "3.00",
    "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN": "0.10",
    "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT": "4.00",
    "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN": "0.02",
    "ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_SIDE_MIN_ROWS": "8",
    "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT": "4.00",
    "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT": "4.00",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE": "0.02",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE": "0.10",
    "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS": "8",
    "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT": "1.00",
    "ENTRY_OFFLINE_RL_Q_WEIGHT": "0.50",
    "ENTRY_OFFLINE_RL_V_WEIGHT": "0.20",
    "ENTRY_OFFLINE_RL_RANK_WEIGHT": "0.05",
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
CANONICAL_DIRECTION_DECISION_CONTRACT = model_direction_decision_contract_metadata()

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


def _require_timestamped_evidence_path(path: Path, *, label: str) -> None:
    if path.name.endswith("_latest.json") or not _TIMESTAMPED_JSON_RE.fullmatch(
        path.name
    ):
        raise RuntimeError(
            f"{label} must be an explicit timestamped JSON evidence event, got {path}"
        )


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    recipe_exact = isinstance(recipe, dict) and recipe == PATH_CALIBRATION_RECIPE_CONTRACT
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    recipe_keys_exact = recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    required_rank_keys_present = {
        "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT",
        "ENTRY_PATH_QUALITY_RANK_WEIGHT",
    }.issubset(recipe_keys)
    argv = contract.get("wrapper_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_recipe_audit = "--recipe-audit-json" in argv_text
    return {
        "ok": bool(
            contract.get("requires_path_calibration_recipe_contract") is True
            and recipe_exact
            and recipe_keys_exact
            and required_rank_keys_present
            and argv_declares_recipe_audit
        ),
        "requires_path_calibration_recipe_contract": contract.get("requires_path_calibration_recipe_contract"),
        "recipe_exact": recipe_exact,
        "recipe_keys_exact": recipe_keys_exact,
        "required_rank_keys_present": required_rank_keys_present,
        "wrapper_argv_declares_recipe_audit": argv_declares_recipe_audit,
        "expected_recipe": PATH_CALIBRATION_RECIPE_CONTRACT,
        "observed_recipe": recipe,
    }


def _direction_balance_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("direction_balance_recipe_contract")
    recipe_exact = isinstance(recipe, dict) and recipe == DIRECTION_BALANCE_RECIPE_CONTRACT
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    recipe_keys_exact = recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    required_objective_keys_present = set(REQUIRED_POSITIVE_LOSS_WEIGHTS).issubset(
        recipe_keys
    )
    argv = contract.get("wrapper_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_recipe_audit = "--recipe-audit-json" in argv_text
    argv_uses_exact_wrapper = (
        contract.get("control_route") == "model-native-smoke-train"
        and "gx1.models.entry_v10.entry_v10_ctx_train_v3" not in argv_text
        and "--anchor-gate-init" not in argv_text
    )
    return {
        "ok": bool(
            contract.get("requires_direction_balance_recipe_contract") is True
            and recipe_exact
            and recipe_keys_exact
            and required_objective_keys_present
            and argv_declares_recipe_audit
            and argv_uses_exact_wrapper
        ),
        "requires_direction_balance_recipe_contract": contract.get("requires_direction_balance_recipe_contract"),
        "recipe_exact": recipe_exact,
        "recipe_keys_exact": recipe_keys_exact,
        "required_objective_keys_present": required_objective_keys_present,
        "wrapper_argv_declares_recipe_audit": argv_declares_recipe_audit,
        "wrapper_argv_uses_exact_route": argv_uses_exact_wrapper,
        "expected_recipe": DIRECTION_BALANCE_RECIPE_CONTRACT,
        "observed_recipe": recipe,
    }


def _tail_direction_recipe_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe = contract.get("tail_direction_recipe_contract")
    recipe_exact = isinstance(recipe, dict) and recipe == TAIL_DIRECTION_RECIPE_CONTRACT
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    recipe_keys_exact = recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    argv = contract.get("wrapper_argv_template")
    argv_text = " ".join(str(part) for part in argv) if isinstance(argv, list) else ""
    argv_declares_recipe_audit = "--recipe-audit-json" in argv_text
    return {
        "ok": bool(
            contract.get("requires_tail_direction_recipe_contract") is True
            and recipe_exact
            and recipe_keys_exact
            and "ENTRY_TAIL_DIRECTION_CE_WEIGHT" in recipe_keys
            and argv_declares_recipe_audit
        ),
        "requires_tail_direction_recipe_contract": contract.get("requires_tail_direction_recipe_contract"),
        "recipe_exact": recipe_exact,
        "recipe_keys_exact": recipe_keys_exact,
        "wrapper_argv_declares_recipe_audit": argv_declares_recipe_audit,
        "expected_recipe": TAIL_DIRECTION_RECIPE_CONTRACT,
        "observed_recipe": recipe,
    }


def _training_objective_future_review(contract: dict[str, Any]) -> dict[str, Any]:
    recipe_keys = set(contract.get("recipe_env_keys") or ())
    required = set(contract.get("required_positive_loss_weights") or ())
    ok = bool(
        contract.get("recipe_audit_schema") == RECIPE_AUDIT_SCHEMA
        and contract.get("training_objective_schema") == TRAINING_OBJECTIVE_SCHEMA
        and contract.get("requires_exact_model_native_training_objective") is True
        and recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
        and required == set(REQUIRED_POSITIVE_LOSS_WEIGHTS)
        and required.issubset(recipe_keys)
    )
    return {
        "ok": ok,
        "recipe_audit_schema": contract.get("recipe_audit_schema"),
        "training_objective_schema": contract.get("training_objective_schema"),
        "recipe_env_keys_exact": recipe_keys == set(MODEL_NATIVE_RECIPE_ENV_KEYS),
        "required_positive_loss_weights_exact": required
        == set(REQUIRED_POSITIVE_LOSS_WEIGHTS),
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


def _canonical_direction_decision_review(contract: dict[str, Any]) -> dict[str, Any]:
    observed = contract.get("canonical_direction_decision_contract")
    exact = (
        isinstance(observed, dict)
        and observed == CANONICAL_DIRECTION_DECISION_CONTRACT
    )
    return {
        "ok": bool(
            contract.get("requires_canonical_direction_decision_contract")
            is True
            and exact
        ),
        "requires_canonical_direction_decision_contract": contract.get(
            "requires_canonical_direction_decision_contract"
        ),
        "contract_exact": exact,
        "expected_contract": CANONICAL_DIRECTION_DECISION_CONTRACT,
        "observed_contract": observed,
    }


def _wrapper_recipe_audit_review(text: str, required_env_keys: tuple[str, ...]) -> dict[str, Any]:
    recipe_keys = set(MODEL_NATIVE_RECIPE_ENV_KEYS)
    missing_recipe_keys = [key for key in required_env_keys if key not in recipe_keys]
    required_wiring = (
        "--recipe-audit-json",
        "--pretrain-audit-json",
        "--full-input-liveness-audit-json",
        "--trainability-readiness-json",
        "gx1.contracts.entry_model_native_train_launch_v1",
        "--vedtak",
        "--execute",
    )
    missing_wiring = [fragment for fragment in required_wiring if fragment not in text]
    forbidden_inline_contracts = [
        fragment
        for fragment in (
            "ENTRY_FOUNDATION_",
            "--anchor-gate-init",
            "GX1_ALLOW_LEGACY_ENTRY_V10_RESEARCH",
        )
        if fragment in text
    ]
    return {
        "recipe_audit_schema": RECIPE_AUDIT_SCHEMA,
        "required_recipe_env_keys": list(required_env_keys),
        "missing_recipe_env_keys": missing_recipe_keys,
        "required_wrapper_wiring": list(required_wiring),
        "missing_wrapper_wiring": missing_wiring,
        "forbidden_inline_contracts": forbidden_inline_contracts,
        "ok": not missing_recipe_keys and not missing_wiring and not forbidden_inline_contracts,
    }


def _wrapper_path_calibration_env_review(text: str) -> dict[str, Any]:
    return _wrapper_recipe_audit_review(
        text,
        (
            "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT",
            "ENTRY_PATH_QUALITY_RANK_WEIGHT",
        ),
    )


def _wrapper_direction_balance_env_review(text: str) -> dict[str, Any]:
    return _wrapper_recipe_audit_review(
        text,
        (
            "ENTRY_DIRECTION_CE_SCALE",
            "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT",
            "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT",
            "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT",
            "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT",
            "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT",
            "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT",
            "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT",
            "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT",
            "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT",
            "ENTRY_HIER_SIDE_VALIDITY_WEIGHT",
            "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT",
            "ENTRY_OFFLINE_RL_Q_WEIGHT",
            "ENTRY_OFFLINE_RL_V_WEIGHT",
            "ENTRY_OFFLINE_RL_RANK_WEIGHT",
        ),
    )


def _wrapper_tail_direction_env_review(text: str) -> dict[str, Any]:
    return _wrapper_recipe_audit_review(
        text,
        ("ENTRY_TAIL_DIRECTION_CE_WEIGHT",),
    )


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


def _path_str(raw: object) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return str(Path(raw).expanduser().resolve())
    return None


def _argv_value(argv: object, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    for idx, value in enumerate(argv[:-1]):
        if value == flag:
            return _path_str(argv[idx + 1])
    return None


def _fresh_source_identity_contract(post_rebuild: dict[str, Any], smoke_readiness: dict[str, Any], future_train: dict[str, Any]) -> dict[str, Any]:
    post_contract = (
        post_rebuild.get("post_rebuild_refresh_command_contract")
        if isinstance(post_rebuild.get("post_rebuild_refresh_command_contract"), dict)
        else {}
    )
    smoke_inputs = smoke_readiness.get("inputs") if isinstance(smoke_readiness.get("inputs"), dict) else {}
    source_dataset = _path_str(post_rebuild.get("dataset_dir"))
    post_smoke_dataset = _path_str(post_contract.get("smart_smoke_dataset_dir"))
    readiness_source_dataset = _path_str(smoke_inputs.get("smart_dataset_dir"))
    readiness_smoke_dataset = _path_str(smoke_inputs.get("smart_smoke_dataset_dir"))
    train_argv = future_train.get("wrapper_argv_template")
    train_dataset = _argv_value(train_argv, "--dataset-dir")
    train_out_bundle = _argv_value(train_argv, "--out-bundle-dir")
    source_root = _path_str(str(Path(source_dataset).parent)) if source_dataset else None
    smoke_root = _path_str(str(Path(post_smoke_dataset).parent)) if post_smoke_dataset else None
    out_root = _path_str(str(Path(train_out_bundle).parent)) if train_out_bundle else None
    return {
        "source_dataset": source_dataset,
        "post_rebuild_smoke_dataset": post_smoke_dataset,
        "smoke_readiness_source_dataset": readiness_source_dataset,
        "smoke_readiness_smoke_dataset": readiness_smoke_dataset,
        "future_train_dataset": train_dataset,
        "future_train_out_bundle": train_out_bundle,
        "source_rebuild_root": source_root,
        "smoke_rebuild_root": smoke_root,
        "future_train_out_root": out_root,
        "source_matches_smoke_readiness": bool(source_dataset) and source_dataset == readiness_source_dataset,
        "smoke_matches_smoke_readiness": bool(post_smoke_dataset) and post_smoke_dataset == readiness_smoke_dataset,
        "future_train_dataset_matches_smoke": bool(post_smoke_dataset) and post_smoke_dataset == train_dataset,
        "future_train_out_under_source_root": bool(source_root) and source_root == out_root,
        "source_and_smoke_share_rebuild_root": bool(source_root) and source_root == smoke_root,
    }


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

    full_input_liveness_json = Path(args.full_input_liveness_json).expanduser().resolve()
    for label, path in (
        ("post-rebuild readiness", post_rebuild_json),
        ("smoke readiness", smoke_readiness_json),
        ("full-input liveness", full_input_liveness_json),
    ):
        _require_timestamped_evidence_path(path, label=label)

    post_rebuild = _read_json_or_empty(post_rebuild_json)
    smoke_readiness = _read_json_or_empty(smoke_readiness_json)
    post_liveness_meta = (
        post_rebuild.get("full_input_liveness_contract")
        if isinstance(post_rebuild.get("full_input_liveness_contract"), dict)
        else {}
    )
    smoke_inputs = (
        smoke_readiness.get("inputs")
        if isinstance(smoke_readiness.get("inputs"), dict)
        else {}
    )
    smoke_liveness_meta = (
        smoke_inputs.get("full_input_liveness_contract")
        if isinstance(smoke_inputs.get("full_input_liveness_contract"), dict)
        else {}
    )
    smoke_liveness_validation = (
        smoke_readiness.get("full_input_liveness_validation")
        if isinstance(smoke_readiness.get("full_input_liveness_validation"), dict)
        else {}
    )
    full_input_liveness_validation = validate_full_input_liveness_artifact(
        full_input_liveness_json,
        expected_sha256=str(post_liveness_meta.get("sha256") or ""),
        expected_dataset_dir=post_rebuild.get("dataset_dir") or "",
        expected_contract_mode=CONTRACT_MODE,
        expected_field_order_sha256=(
            post_liveness_meta.get("field_order_sha256")
            if isinstance(post_liveness_meta.get("field_order_sha256"), dict)
            else {}
        ),
    )
    future_train = _future_train_contract(smoke_readiness)
    fresh_source_identity_contract = _fresh_source_identity_contract(post_rebuild, smoke_readiness, future_train)
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
    training_objective_review = _training_objective_future_review(future_train)
    direction_context_slice_review = _direction_context_slice_review(future_train)
    canonical_direction_decision_review = _canonical_direction_decision_review(
        future_train
    )
    smoke_wrapper_path_calibration_review = _wrapper_path_calibration_env_review(smoke_wrapper_text)
    candidate_wrapper_path_calibration_review = _wrapper_path_calibration_env_review(candidate_wrapper_text)
    smoke_wrapper_direction_balance_review = _wrapper_direction_balance_env_review(smoke_wrapper_text)
    candidate_wrapper_direction_balance_review = _wrapper_direction_balance_env_review(candidate_wrapper_text)
    smoke_wrapper_tail_direction_review = _wrapper_tail_direction_env_review(smoke_wrapper_text)
    candidate_wrapper_tail_direction_review = _wrapper_tail_direction_env_review(candidate_wrapper_text)

    checks = [
        _check("smart post-rebuild dataset audit is ready", post_rebuild.get("decision") == "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW", post_rebuild.get("decision")),
        _check(
            "model-native seq513 smoke readiness is ready",
            smoke_readiness.get("decision")
            == "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW",
            smoke_readiness.get("decision"),
        ),
        _check(
            "smart post-rebuild binds canonical full-input liveness artifact",
            post_liveness_meta.get("schema_version") == FULL_INPUT_LIVENESS_SCHEMA
            and _path_str(post_liveness_meta.get("path")) == str(full_input_liveness_json)
            and post_liveness_meta.get("decision") == "PASS"
            and post_liveness_meta.get("atr_ood_status") == "GREEN",
            post_liveness_meta,
        ),
        _check(
            "smart smoke readiness revalidates the same full-input liveness bytes",
            smoke_liveness_meta.get("path") == str(full_input_liveness_json)
            and smoke_liveness_meta.get("sha256") == post_liveness_meta.get("sha256")
            and smoke_liveness_validation.get("ok") is True
            and smoke_liveness_validation.get("sha256") == post_liveness_meta.get("sha256")
            and smoke_liveness_validation.get("schema_version") == FULL_INPUT_LIVENESS_SCHEMA
            and smoke_liveness_validation.get("field_order_sha256")
            == post_liveness_meta.get("field_order_sha256"),
            {
                "post_rebuild_binding": post_liveness_meta,
                "smoke_input_binding": smoke_liveness_meta,
                "smoke_validation": smoke_liveness_validation,
            },
        ),
        _check(
            "full-input liveness artifact hash schema fields and ATR OOD validate for trainability",
            bool(full_input_liveness_validation["ok"]),
            full_input_liveness_validation,
        ),
        _check(
            "smart smoke readiness uses same source dataset as post-rebuild readiness",
            fresh_source_identity_contract["source_matches_smoke_readiness"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart smoke readiness uses same smoke dataset as post-rebuild contract",
            fresh_source_identity_contract["smoke_matches_smoke_readiness"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart future train dataset matches fresh smoke dataset",
            fresh_source_identity_contract["future_train_dataset_matches_smoke"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart future train output stays under fresh source rebuild root",
            fresh_source_identity_contract["future_train_out_under_source_root"],
            fresh_source_identity_contract,
        ),
        _check(
            "smart source and smoke datasets share rebuild root",
            fresh_source_identity_contract["source_and_smoke_share_rebuild_root"],
            fresh_source_identity_contract,
        ),
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
        _check(
            "smoke wrapper exposes the model-native seq513 lane",
            CONTRACT_MODE in smoke_wrapper_text
            and "--anchor-gate-init" not in smoke_wrapper_text,
            _artifact_meta(smoke_wrapper),
        ),
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
        _check(
            "model-native smoke train is wired in control surface",
            "model-native-smoke-train)" in control_text
            and "model-native-smoke-train --vedtak <id>" in control_text,
            _artifact_meta(control_script),
        ),
        _check("smart smoke future contract is implemented in control surface", future_train.get("implemented_in_control_surface") is True, future_train),
        _check(
            "smart smoke future contract uses only the compact wrapper route",
            future_train.get("control_route") == "model-native-smoke-train"
            and future_train.get("wrapper_path")
            == "scripts/run_entry_model_native_seq513_smoke_train.sh"
            and future_train.get("wrapper_argv_template")
            == future_train.get("argv_template")
            and "gx1.models.entry_v10.entry_v10_ctx_train_v3"
            not in " ".join(future_train.get("argv_template") or ())
            and "audit-smoke-bundle"
            not in (future_train.get("argv_template") or ()),
            future_train,
        ),
        _check(
            "smart smoke future contract records unavailable immutable smoke audit route",
            future_train.get("requires_edge_audit") is True
            and future_train.get("post_smoke_audit_control_route_exposed") is False
            and bool(future_train.get("post_smoke_audit_blocker")),
            future_train,
        ),
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
            "smart smoke future contract declares exact positive training objective",
            bool(training_objective_review["ok"]),
            training_objective_review,
        ),
        _check(
            "smart smoke future contract declares direction context slice audit",
            bool(direction_context_slice_review["ok"]),
            direction_context_slice_review,
        ),
        _check(
            "smart smoke future contract declares canonical derived direction pair",
            bool(canonical_direction_decision_review["ok"]),
            canonical_direction_decision_review,
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
        _check("candidate-readiness supports model-native seq513", CONTRACT_MODE in candidate_readiness_text and str(EXPECTED_SIGNAL_DIM) in candidate_readiness_text, _artifact_meta(candidate_readiness_script)),
        _check(
            "candidate train wrapper exposes the model-native seq513 lane",
            CONTRACT_MODE in candidate_wrapper_text
            and "--anchor-gate-init" not in candidate_wrapper_text,
            _artifact_meta(candidate_wrapper),
        ),
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
        _check("selective-edge supports model-native seq513", CONTRACT_MODE in selective_edge_text and str(EXPECTED_SIGNAL_DIM) in selective_edge_text, _artifact_meta(selective_edge_script)),
        _check("replay evidence supports model-native seq513", CONTRACT_MODE in replay_evidence_text and str(EXPECTED_SIGNAL_DIM) in replay_evidence_text, _artifact_meta(replay_evidence_script)),
        _check("replay-readiness supports model-native seq513", CONTRACT_MODE in replay_readiness_text and str(EXPECTED_SIGNAL_DIM) in replay_readiness_text, _artifact_meta(replay_readiness_script)),
        _check("side effects remain closed", all(value is False for value in SIDE_EFFECTS_CLOSED.values()), SIDE_EFFECTS_CLOSED),
    ]
    failures = [row for row in checks if not row["ok"]]
    ready = not failures
    decision = READY_DECISION if ready else BLOCKED_DECISION
    report = {
        "schema_version": "entry_model_native_seq513_trainability_readiness_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "manifest_variant": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
        "required_training_specialists": list(required_specialists),
        "inputs": {
            "smart_post_rebuild_readiness": _artifact_meta(post_rebuild_json),
            "smart_smoke_readiness": _artifact_meta(smoke_readiness_json),
            "full_input_liveness_contract": _artifact_meta(full_input_liveness_json),
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
        "fresh_source_identity_contract": fresh_source_identity_contract,
        "source_metadata_contract": source_metadata_contract,
        "full_input_liveness_validation": full_input_liveness_validation,
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
            else "wire model-native seq513 trainer/wrapper/control/candidate/replay surfaces before any training vedtak"
        ),
    }
    report["evidence_binding_sha256"] = _sha256_json(report["inputs"])
    _, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-post-rebuild-readiness-json", required=True)
    ap.add_argument("--smart-smoke-readiness-json", required=True)
    ap.add_argument("--full-input-liveness-json", required=True)
    ap.add_argument("--control-script", required=True)
    ap.add_argument("--trainer-source", required=True)
    ap.add_argument("--smoke-wrapper", required=True)
    ap.add_argument("--candidate-wrapper", required=True)
    ap.add_argument("--candidate-readiness-script", required=True)
    ap.add_argument("--selective-edge-script", required=True)
    ap.add_argument("--replay-evidence-script", required=True)
    ap.add_argument("--replay-readiness-script", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
