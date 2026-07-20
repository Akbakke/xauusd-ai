#!/usr/bin/env python3
"""Report-only model-native seq513 smoke-readiness evidence gate.

This gate does not rebuild data, write a smoke train manifest, start training,
run replay, distill IQL, touch shadow, or touch live paths. It records the
fail-closed requirements that must be true before a future model-native seq513 smoke
manifest or smoke train wrapper can be reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.contracts.entry_full_input_liveness_v1 import (
    SCHEMA_VERSION as FULL_INPUT_LIVENESS_SCHEMA,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
    require_foundation_audit_report_policy,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS,
    require_model_native_aux_target_contract,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    require_offline_rl_contract_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
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
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
CONTRACT_MODE = MODEL_NATIVE_CONTRACT_MODE
MANIFEST_VARIANT = MODEL_NATIVE_CONTRACT_MODE
EXPECTED_SIGNAL_DIM = MODEL_NATIVE_SIGNAL_DIM
EXPECTED_BASE_SIGNAL_FEATURES = MODEL_NATIVE_BASE_SIGNAL_DIM
EXPECTED_SELECTED_FEATURES = EXPECTED_SIGNAL_DIM - EXPECTED_BASE_SIGNAL_FEATURES
REQUIRED_SPECIALISTS = tuple(required_training_specialists_for_mode(CONTRACT_MODE))
EXPECTED_MODEL_CONTRACT = specialist_model_contract_for_mode(CONTRACT_MODE)

EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
SMOKE_MANIFEST_READY_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW"
SMOKE_MANIFEST_SCHEMA = "entry_model_native_seq513_smoke_manifest_v2"
SMOKE_DATASET_MANIFEST_SCHEMA = "entry_model_native_seq513_smoke_dataset_v2"
SMOKE_SPLIT_MANIFEST_SCHEMA = "entry_model_native_seq513_smoke_split_manifest_v2"
_TIMESTAMPED_JSON_RE = re.compile(
    r"^.+_\d{8}T\d{6}(?:\d{6})?Z\.json$"
)
REQUIRED_SMOKE_MANIFEST_PROVENANCE_CHECKS = (
    "smart post-rebuild readiness proves orchestration provenance",
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
DIRECTION_CONTEXT_SLICE_CONTRACT = {
    "source": "post_smoke_audit.direction_slice_contract",
    "ctx_cat_names": ["session_id", "vol_regime_id", "atr_bucket", "spread_bucket", "H4_trend_sign_cat"],
    "min_rows": 64,
    "requires_majority_baseline": True,
    "requires_class_distribution_coverage": True,
    "skips_low_label_diversity": True,
}
CANONICAL_DIRECTION_DECISION_CONTRACT = model_direction_decision_contract_metadata()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
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


def _contract_path(raw: object, *, label: str) -> Path:
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    return (Path("/") / f"GX1_MISSING_{label}").resolve()


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details if details is not None else {}}


def _gate(name: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "decision": "PASS" if all(bool(row.get("ok")) for row in checks) else "FAIL",
        "checks": checks,
    }


def _path_calibration_recipe_ok(contract: dict[str, Any]) -> bool:
    recipe = contract.get("path_calibration_recipe_contract")
    recipe_keys = contract.get("recipe_env_keys")
    if not isinstance(recipe, dict) or not isinstance(recipe_keys, list):
        return False
    if recipe != PATH_CALIBRATION_RECIPE_CONTRACT:
        return False
    return (
        set(recipe_keys) == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
        and {
            "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT",
            "ENTRY_PATH_QUALITY_RANK_WEIGHT",
        }.issubset(recipe_keys)
    )


def _direction_balance_recipe_ok(contract: dict[str, Any]) -> bool:
    recipe = contract.get("direction_balance_recipe_contract")
    recipe_keys = contract.get("recipe_env_keys")
    if not isinstance(recipe, dict) or not isinstance(recipe_keys, list):
        return False
    if recipe != DIRECTION_BALANCE_RECIPE_CONTRACT:
        return False
    return (
        set(recipe_keys) == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
        and set(REQUIRED_POSITIVE_LOSS_WEIGHTS).issubset(recipe_keys)
    )


def _tail_direction_recipe_ok(contract: dict[str, Any]) -> bool:
    recipe = contract.get("tail_direction_recipe_contract")
    recipe_keys = contract.get("recipe_env_keys")
    if not isinstance(recipe, dict) or not isinstance(recipe_keys, list):
        return False
    if recipe != TAIL_DIRECTION_RECIPE_CONTRACT:
        return False
    return (
        set(recipe_keys) == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
        and "ENTRY_TAIL_DIRECTION_CE_WEIGHT" in recipe_keys
    )


def _training_objective_recipe_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("recipe_audit_schema") == RECIPE_AUDIT_SCHEMA
        and contract.get("training_objective_schema") == TRAINING_OBJECTIVE_SCHEMA
        and contract.get("requires_exact_model_native_training_objective") is True
        and set(contract.get("recipe_env_keys") or ())
        == set(MODEL_NATIVE_RECIPE_ENV_KEYS)
        and set(contract.get("required_positive_loss_weights") or ())
        == set(REQUIRED_POSITIVE_LOSS_WEIGHTS)
    )


def _direction_context_slice_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("requires_direction_context_slice_contract") is True
        and contract.get("direction_context_slice_contract") == DIRECTION_CONTEXT_SLICE_CONTRACT
    )


def _canonical_direction_decision_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("requires_canonical_direction_decision_contract") is True
        and contract.get("canonical_direction_decision_contract")
        == CANONICAL_DIRECTION_DECISION_CONTRACT
    )


def _git_status_short(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "status", "--short"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return [f"git status failed: {proc.stderr.strip()}"]
    return proc.stdout.splitlines()


def _dataset_path_matches(report: dict[str, Any], expected: Path) -> bool:
    for key in ("dataset_dir", "source_dir", "out_dir"):
        raw = str(report.get(key) or "")
        if not raw:
            continue
        try:
            if Path(raw).expanduser().resolve() == expected.expanduser().resolve():
                return True
        except OSError:
            continue
    return False


def _path_equals(raw: Any, expected: Path) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    try:
        return Path(text).expanduser().resolve() == expected.expanduser().resolve()
    except OSError:
        return False


def _split_artifact_hash_review(splits: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        row = splits.get(split) if isinstance(splits.get(split), dict) else {}
        parquet_path = Path(str(row.get("out_parquet") or "")).expanduser()
        manifest_path = Path(str(row.get("out_manifest") or "")).expanduser()
        parquet_exists = bool(str(row.get("out_parquet") or "").strip()) and parquet_path.is_file()
        manifest_exists = bool(str(row.get("out_manifest") or "").strip()) and manifest_path.is_file()
        parquet_sha = _sha256_file(parquet_path) if parquet_exists else None
        manifest_sha = _sha256_file(manifest_path) if manifest_exists else None
        details[split] = {
            "out_parquet": str(row.get("out_parquet") or ""),
            "out_manifest": str(row.get("out_manifest") or ""),
            "out_parquet_exists": parquet_exists,
            "out_manifest_exists": manifest_exists,
            "expected_out_parquet_sha256": row.get("out_parquet_sha256"),
            "expected_out_manifest_sha256": row.get("out_manifest_sha256"),
            "observed_out_parquet_sha256": parquet_sha,
            "observed_out_manifest_sha256": manifest_sha,
            "parquet_hash_matches": parquet_sha == row.get("out_parquet_sha256"),
            "manifest_hash_matches": manifest_sha == row.get("out_manifest_sha256"),
        }
    return {
        "ok": set(splits) == {"train", "val", "test"}
        and all(
            details[split]["out_parquet_exists"]
            and details[split]["out_manifest_exists"]
            and details[split]["parquet_hash_matches"]
            and details[split]["manifest_hash_matches"]
            for split in ("train", "val", "test")
        ),
        "details": details,
    }


def _all_side_effects_false(payload: dict[str, Any], *, extra: tuple[str, ...] = ()) -> bool:
    side_effects = payload.get("side_effects_started")
    if not isinstance(side_effects, dict):
        return False
    required = ("training", "replay", "iql_distillation", "shadow", "live", *extra)
    return all(side_effects.get(key) is False for key in required)


def _liveness_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "foundation_objective_liveness",
        "foundation_source_field_liveness",
        "specialist_input_liveness",
        "feature_liveness",
        "numeric_feature_liveness",
    ):
        value = report.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _rows_have_no_nonfinite_or_collapse(
    rows: list[dict[str, Any]],
    *,
    allow_near_constant_count: bool = False,
) -> bool:
    if not rows:
        return False
    for row in rows:
        if int(row.get("nonfinite_count") or 0) != 0:
            return False
        if int(row.get("nan_count") or 0) != 0:
            return False
        if int(row.get("inf_count") or 0) != 0:
            return False
        if int(row.get("missing_count") or 0) != 0:
            return False
        if not allow_near_constant_count and int(row.get("near_constant_count") or 0) != 0:
            return False
        if bool(row.get("near_constant")):
            return False
        if "live_feature_count" in row and int(row.get("live_feature_count") or 0) < int(
            row.get("min_required_live_feature_count") or 1
        ):
            return False
    return True


def _target_head_contract(report: dict[str, Any]) -> dict[str, Any]:
    contract = report.get("target_head_contract")
    return contract if isinstance(contract, dict) else {}


def _target_aux_contract_report(report: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if report.get("schema_version") != FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION:
        failures.append("target audit schema is stale")
    try:
        require_foundation_audit_report_policy(
            report,
            audit_kind="target",
            context="SMOKE_READINESS_TARGET_AUDIT",
        )
    except RuntimeError as exc:
        failures.append(str(exc))
    try:
        require_model_native_aux_target_contract(
            report.get("model_native_aux_target_contract"),
            context="SMOKE_READINESS_TARGET_AUDIT",
        )
    except RuntimeError as exc:
        failures.append(str(exc))
    offline_rl = report.get("offline_rl_target_contract")
    if (
        not isinstance(offline_rl, dict)
        or offline_rl.get("decision") != "PASS"
        or offline_rl.get("failures") != []
    ):
        failures.append("offline-RL target proof is not a zero-failure PASS")
    else:
        try:
            require_offline_rl_contract_metadata(
                offline_rl.get("offline_rl_contract"),
                context="SMOKE_READINESS_TARGET_AUDIT",
            )
        except RuntimeError as exc:
            failures.append(str(exc))
    heads = _target_head_contract(report)
    if tuple(heads.get("extra_active_target_heads") or ()) != tuple(
        MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
    ) or not all(
        (heads.get("extra_active_target_head_liveness") or {}).get(head) is True
        for head in MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
    ):
        failures.append("extra active target-head liveness is unproven")
    return {
        "ok": not failures,
        "failures": failures,
        "required_schema_version": FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
        "required_extra_active_target_heads": list(
            MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
        ),
    }


def _specialist_model_contract_exact(report: dict[str, Any]) -> bool:
    observed = report.get("specialist_model_contract")
    return isinstance(observed, dict) and json.loads(json.dumps(observed)) == json.loads(
        json.dumps(EXPECTED_MODEL_CONTRACT)
    )


def _recommended_heads(report: dict[str, Any]) -> tuple[set[str], set[str]]:
    arch = report.get("architecture_contract") if isinstance(report.get("architecture_contract"), dict) else {}
    recommended = arch.get("recommended_fusion") if isinstance(arch.get("recommended_fusion"), dict) else {}
    active = {str(x) for x in recommended.get("active_heads") or recommended.get("heads") or [] if str(x)}
    blocked = {str(x) for x in recommended.get("blocked_heads") or [] if str(x)}
    return active, blocked


def _trainer_loader_probe(specialist_audit_json: Path) -> dict[str, Any]:
    details: dict[str, Any] = {
        "audit_json": str(specialist_audit_json),
        "contract_mode": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
    }
    try:
        from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _load_specialist_fusion_contract

        indices, meta = _load_specialist_fusion_contract(
            specialist_audit_json,
            expected_signal_dim=EXPECTED_SIGNAL_DIM,
            contract_mode=CONTRACT_MODE,
        )
        details.update(
            {
                "ok": True,
                "loaded_specialists": sorted(str(name) for name in indices),
                "group_feature_counts": {str(name): int(len(vals)) for name, vals in indices.items()},
                "meta_contract_mode": meta.get("contract_mode"),
                "meta_signal_field_count": int(meta.get("signal_field_count") or 0),
            }
        )
    except Exception as exc:
        details.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "loaded_specialists": [],
            }
        )
    return details


def _future_contracts(
    *,
    smart_dataset_dir: Path,
    smart_smoke_dataset_dir: Path,
    smoke_splits: dict[str, Any],
    full_input_liveness_json: Path,
    feature_audit_json: Path,
    target_audit_json: Path,
    specialist_audit_json: Path,
    smoke_manifest_event_json: Path,
    memory_cap: str,
    swap_cap: str,
) -> dict[str, Any]:
    def _split_artifact(split: str, key: str) -> str:
        row = smoke_splits.get(split)
        if isinstance(row, dict) and str(row.get(key) or "").strip():
            return str(Path(str(row[key])).expanduser().resolve())
        return f"<MISSING_IMMUTABLE_{split.upper()}_{key.upper()}>"

    out_bundle = str(
        smart_dataset_dir.parent / "v10_entry_model_native_seq513_smoke_<STAMP>"
    )
    wrapper_argv = [
        "scripts/entry_next_edge_control.sh",
        "model-native-smoke-train",
        "--run-id",
        "<MODEL_NATIVE_SEQ513_SMOKE_RUN_ID_ID>",
        "--dataset-dir",
        str(smart_smoke_dataset_dir),
        "--train-manifest-json",
        _split_artifact("train", "out_manifest"),
        "--val-manifest-json",
        _split_artifact("val", "out_manifest"),
        "--test-manifest-json",
        _split_artifact("test", "out_manifest"),
        "--train-parquet",
        _split_artifact("train", "out_parquet"),
        "--val-parquet",
        _split_artifact("val", "out_parquet"),
        "--test-parquet",
        _split_artifact("test", "out_parquet"),
        "--m5-prebuilt-path",
        "<IMMUTABLE_TIMESTAMPED_M5_PREBUILT_PATH>",
        "--full-input-liveness-audit-json",
        str(full_input_liveness_json),
        "--feature-audit-json",
        str(feature_audit_json),
        "--target-audit-json",
        str(target_audit_json),
        "--specialist-audit-json",
        str(specialist_audit_json),
        "--pretrain-audit-json",
        "<IMMUTABLE_TIMESTAMPED_PRETRAIN_AUDIT_JSON>",
        "--recipe-audit-json",
        "<IMMUTABLE_TIMESTAMPED_RECIPE_AUDIT_JSON>",
        "--smoke-manifest-json",
        str(smoke_manifest_event_json),
        "--smoke-readiness-json",
        "<THIS_IMMUTABLE_TIMESTAMPED_SMOKE_READINESS_JSON>",
        "--trainability-readiness-json",
        "<IMMUTABLE_TIMESTAMPED_TRAINABILITY_READINESS_JSON>",
        "--out-bundle-dir",
        out_bundle,
        "--gx1-data-root",
        "<ABSOLUTE_CANONICAL_GX1_DATA_ROOT>",
        "--device",
        "cuda",
        "--seed",
        "1337",
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--learning-rate",
        "0.0003",
        "--early-stop-patience",
        "1",
        "--early-stop-min-delta",
        "0.0",
        "--grad-clip-norm",
        "1.0",
        "--weight-decay",
        "0.00001",
        "--multi-tf-scale",
        "0.5",
        "--specialist-fusion-scale",
        "0.25",
        "--subsample-rows",
        "10000",
        "--memory-cap",
        memory_cap,
        "--swap-cap",
        swap_cap,
        "<EXACTLY_ONE_OF_DRY_RUN_OR_EXECUTE>",
    ]
    train = {
        "argv_template": wrapper_argv,
        "wrapper_argv_template": wrapper_argv,
        "mode": "future_exact_wrapper_contract",
        "implemented_in_control_surface": True,
        "control_route": "model-native-smoke-train",
        "wrapper_path": "scripts/run_entry_model_native_seq513_smoke_train.sh",
        "execution_allowed_now": False,
        "run_lineage_required": True,
        "requires_clean_git": True,
        "requires_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "memory_cap": memory_cap,
        "swap_cap": swap_cap,
        "num_workers": 0,
        "starts_trainer": True,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "requires_edge_audit": True,
        "post_smoke_audit_control_route_exposed": False,
        "post_smoke_audit_blocker": (
            "no immutable fail-closed smoke-bundle audit route is exposed in the compact control surface"
        ),
        "recipe_audit_schema": RECIPE_AUDIT_SCHEMA,
        "recipe_env_keys": list(MODEL_NATIVE_RECIPE_ENV_KEYS),
        "required_positive_loss_weights": list(REQUIRED_POSITIVE_LOSS_WEIGHTS),
        "training_objective_schema": TRAINING_OBJECTIVE_SCHEMA,
        "requires_exact_model_native_training_objective": True,
        "requires_path_calibration_recipe_contract": True,
        "path_calibration_recipe_contract": dict(PATH_CALIBRATION_RECIPE_CONTRACT),
        "requires_direction_balance_recipe_contract": True,
        "direction_balance_recipe_contract": dict(DIRECTION_BALANCE_RECIPE_CONTRACT),
        "requires_tail_direction_recipe_contract": True,
        "tail_direction_recipe_contract": dict(TAIL_DIRECTION_RECIPE_CONTRACT),
        "requires_direction_context_slice_contract": True,
        "direction_context_slice_contract": dict(DIRECTION_CONTEXT_SLICE_CONTRACT),
        "requires_canonical_direction_decision_contract": True,
        "canonical_direction_decision_contract": dict(
            CANONICAL_DIRECTION_DECISION_CONTRACT
        ),
        "specialist_contract_mode": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
        "required_training_specialists": list(REQUIRED_SPECIALISTS),
        "requires_exact_specialist_contract_proof": True,
    }
    manifest = {
        "mode": "future_manifest_only_contract",
        "implemented_in_control_surface": False,
        "execution_allowed_now": False,
        "run_lineage_required": True,
        "requires_clean_git": True,
        "starts_trainer": False,
        "starts_replay": False,
        "starts_iql_distillation": False,
        "touches_shadow_or_live": False,
        "declares_ram_cap": True,
        "ram_cap_runner": "scripts/gx1_capped_run.sh",
        "memory_cap": memory_cap,
        "swap_cap": swap_cap,
        "num_workers": 0,
        "requires_edge_audit_in_followup_train": True,
        "specialist_contract_mode": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
        "required_training_specialists": list(REQUIRED_SPECIALISTS),
        "requires_exact_specialist_contract_proof": True,
    }
    return {"smart_smoke_manifest": manifest, "smart_smoke_train": train}


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    post_rebuild_readiness_json = Path(args.smart_post_rebuild_readiness_json).expanduser().resolve()
    post_rebuild = _read_json_or_empty(post_rebuild_readiness_json)
    post_rebuild_contract = (
        post_rebuild.get("post_rebuild_refresh_command_contract")
        if isinstance(post_rebuild.get("post_rebuild_refresh_command_contract"), dict)
        else {}
    )
    post_liveness_meta = (
        post_rebuild.get("full_input_liveness_contract")
        if isinstance(post_rebuild.get("full_input_liveness_contract"), dict)
        else {}
    )
    smart_dataset_dir = _contract_path(
        args.smart_dataset_dir,
        label="SMART_SEQ513_MODEL_NATIVE_SOURCE_DATASET_DIR",
    )
    smart_smoke_dataset_dir = _contract_path(
        args.smart_smoke_dataset_dir,
        label="MODEL_NATIVE_SEQ513_SMOKE_DATASET_DIR",
    )
    rebuild_preflight_json = Path(
        args.model_native_rebuild_preflight_json
    ).expanduser().resolve()
    feature_audit_json = Path(args.smart_feature_audit_json).expanduser().resolve()
    target_audit_json = Path(args.smart_target_audit_json).expanduser().resolve()
    specialist_audit_json = Path(args.smart_specialist_audit_json).expanduser().resolve()
    smoke_manifest_event_json = Path(args.smoke_manifest_event_json).expanduser().resolve()
    full_input_liveness_json = Path(args.full_input_liveness_json).expanduser().resolve()

    for label, path in (
        ("post-rebuild readiness", post_rebuild_readiness_json),
        ("full-input liveness", full_input_liveness_json),
        ("rebuild preflight", rebuild_preflight_json),
        ("feature audit", feature_audit_json),
        ("target audit", target_audit_json),
        ("specialist audit", specialist_audit_json),
        ("smoke manifest", smoke_manifest_event_json),
    ):
        _require_timestamped_evidence_path(path, label=label)

    rebuild = _read_json_or_empty(rebuild_preflight_json)
    feature = _read_json_or_empty(feature_audit_json)
    target = _read_json_or_empty(target_audit_json)
    specialist = _read_json_or_empty(specialist_audit_json)
    smoke_manifest_event = _read_json_or_empty(smoke_manifest_event_json)
    smoke_manifest = (
        smoke_manifest_event.get("smoke_manifest")
        if isinstance(smoke_manifest_event.get("smoke_manifest"), dict)
        else {}
    )
    smoke_manifest_readiness = smoke_manifest_event
    smoke_manifest_readiness_checks = {
        str(row.get("name") or ""): row
        for row in smoke_manifest_readiness.get("checks", [])
        if isinstance(row, dict)
    }
    missing_smoke_manifest_provenance_checks = [
        name for name in REQUIRED_SMOKE_MANIFEST_PROVENANCE_CHECKS if name not in smoke_manifest_readiness_checks
    ]
    failed_smoke_manifest_provenance_checks = [
        name for name in REQUIRED_SMOKE_MANIFEST_PROVENANCE_CHECKS
        if name in smoke_manifest_readiness_checks and not bool(smoke_manifest_readiness_checks[name].get("ok"))
    ]
    git_status = _git_status_short(Path(args.repo_dir).expanduser().resolve())
    trainer_probe = _trainer_loader_probe(specialist_audit_json) if specialist_audit_json.exists() else {"ok": False}
    future_contracts = _future_contracts(
        smart_dataset_dir=smart_dataset_dir,
        smart_smoke_dataset_dir=smart_smoke_dataset_dir,
        smoke_splits=(
            smoke_manifest.get("splits")
            if isinstance(smoke_manifest.get("splits"), dict)
            else {}
        ),
        full_input_liveness_json=full_input_liveness_json,
        feature_audit_json=feature_audit_json,
        target_audit_json=target_audit_json,
        specialist_audit_json=specialist_audit_json,
        smoke_manifest_event_json=smoke_manifest_event_json,
        memory_cap=str(args.memory_cap),
        swap_cap=str(args.swap_cap),
    )

    rebuild_counts = rebuild.get("counts") if isinstance(rebuild.get("counts"), dict) else {}
    target_head_contract = _target_head_contract(target)
    target_aux_contract = _target_aux_contract_report(target)
    target_active_heads = {str(x) for x in target_head_contract.get("active_training_heads") or []}
    target_blocked_heads = {str(x) for x in target_head_contract.get("blocked_heads") or []}
    specialist_active_heads, specialist_blocked_heads = _recommended_heads(specialist)
    specialist_rows = _liveness_rows(specialist)
    feature_rows = _liveness_rows(feature)
    smoke_splits = smoke_manifest.get("splits") if isinstance(smoke_manifest.get("splits"), dict) else {}
    split_artifact_hash_review = _split_artifact_hash_review(smoke_splits)
    full_input_liveness_validation = validate_full_input_liveness_artifact(
        full_input_liveness_json,
        expected_sha256=str(post_liveness_meta.get("sha256") or ""),
        expected_dataset_dir=smart_dataset_dir,
        expected_contract_mode=CONTRACT_MODE,
        expected_field_order_sha256=(
            post_liveness_meta.get("field_order_sha256")
            if isinstance(post_liveness_meta.get("field_order_sha256"), dict)
            else {}
        ),
    )

    gates = [
        _gate(
            "model_native_rebuild_preflight",
            [
                _check(
                    "smart post-rebuild readiness exists",
                    post_rebuild_readiness_json.exists(),
                    _artifact_meta(post_rebuild_readiness_json),
                ),
                _check(
                    "smart post-rebuild readiness is ready",
                    post_rebuild.get("decision") == "ENTRY_SMART_DATASET_READY_FOR_TRAIN_READINESS_REVIEW",
                    {"decision": post_rebuild.get("decision")},
                ),
                _check(
                    "smart post-rebuild binds exact full-input liveness artifact",
                    post_liveness_meta.get("schema_version") == FULL_INPUT_LIVENESS_SCHEMA
                    and _path_equals(post_liveness_meta.get("path"), full_input_liveness_json)
                    and post_liveness_meta.get("decision") == "PASS"
                    and post_liveness_meta.get("atr_ood_status") == "GREEN",
                    {
                        "post_rebuild_binding": post_liveness_meta,
                        "selected_path": str(full_input_liveness_json),
                    },
                ),
                _check(
                    "full-input liveness artifact hash schema fields and ATR OOD validate",
                    bool(full_input_liveness_validation["ok"]),
                    full_input_liveness_validation,
                ),
                _check(
                    "smart post-rebuild readiness points at selected source dataset",
                    _dataset_path_matches(post_rebuild, smart_dataset_dir),
                    {"dataset_dir": post_rebuild.get("dataset_dir"), "expected": str(smart_dataset_dir)},
                ),
                _check(
                    "smart post-rebuild contract points at selected smoke dataset",
                    _path_equals(post_rebuild_contract.get("smart_smoke_dataset_dir"), smart_smoke_dataset_dir),
                    {
                        "contract_smart_smoke_dataset_dir": post_rebuild_contract.get("smart_smoke_dataset_dir"),
                        "expected": str(smart_smoke_dataset_dir),
                    },
                ),
                _check(
                    "smart source and smoke datasets share the same rebuild root",
                    smart_dataset_dir.parent == smart_smoke_dataset_dir.parent,
                    {
                        "source_rebuild_root": str(smart_dataset_dir.parent),
                        "smoke_rebuild_root": str(smart_smoke_dataset_dir.parent),
                    },
                ),
                _check("model-native rebuild preflight exists", rebuild_preflight_json.exists(), _artifact_meta(rebuild_preflight_json)),
                _check(
                    "model-native rebuild preflight is evidence-ready",
                    rebuild.get("decision")
                    == "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD",
                    {"decision": rebuild.get("decision")},
                ),
                _check("model-native rebuild preflight is report-only", rebuild.get("report_only") is True, rebuild.get("report_only")),
                _check(
                    "model-native rebuild decision is an exact boolean",
                    isinstance(rebuild.get("dataset_rebuild_allowed"), bool),
                    rebuild.get("dataset_rebuild_allowed"),
                ),
                _check("model-native rebuild preflight keeps training closed", rebuild.get("training_allowed") is False, rebuild.get("training_allowed")),
                _check(
                    "model-native rebuild preflight side effects are closed",
                    _all_side_effects_false(rebuild, extra=("dataset_rebuild",)),
                    rebuild.get("side_effects_started"),
                ),
                _check(
                    "model-native rebuild preflight pins exact seq513 contract",
                    rebuild_counts.get("manifest_variant") == MANIFEST_VARIANT
                    and int(rebuild_counts.get("expected_seq_snap_width") or 0) == EXPECTED_SIGNAL_DIM,
                    rebuild_counts,
                ),
            ],
        ),
        _gate(
            "smart_dataset_audit",
            [
                _check("smart feature audit exists", feature_audit_json.exists(), _artifact_meta(feature_audit_json)),
                _check("smart feature audit PASS", feature.get("decision") == "PASS", {"decision": feature.get("decision")}),
                _check("smart feature audit has zero failures", not feature.get("failures"), feature.get("failures")),
                _check(
                    "smart feature audit points at smart dataset",
                    _dataset_path_matches(feature, smart_dataset_dir),
                    {"dataset_dir": feature.get("dataset_dir"), "expected": str(smart_dataset_dir)},
                ),
                _check(
                    "smart feature audit proves finite live features",
                    bool(feature.get("foundation_objective_liveness_all_live"))
                    and bool(feature.get("foundation_source_field_liveness_all_live"))
                    and _rows_have_no_nonfinite_or_collapse(feature_rows),
                    {
                        "foundation_objective_liveness_all_live": feature.get("foundation_objective_liveness_all_live"),
                        "foundation_source_field_liveness_all_live": feature.get("foundation_source_field_liveness_all_live"),
                        "liveness_row_count": len(feature_rows),
                    },
                ),
                _check("smart target audit exists", target_audit_json.exists(), _artifact_meta(target_audit_json)),
                _check("smart target audit PASS", target.get("decision") == "PASS", {"decision": target.get("decision")}),
                _check("smart target audit has zero failures", not target.get("failures"), target.get("failures")),
                _check(
                    "smart target audit proves exact aux-v3 and offline-RL targets",
                    bool(target_aux_contract["ok"]),
                    target_aux_contract,
                ),
                _check(
                    "smart target audit points at smart dataset",
                    _dataset_path_matches(target, smart_dataset_dir),
                    {"dataset_dir": target.get("dataset_dir"), "expected": str(smart_dataset_dir)},
                ),
                _check(
                    "smart target head contract is exact active/blocked set",
                    target_active_heads == set(SPECIALIST_FUSION_ACTIVE_HEADS)
                    and target_blocked_heads == set(SPECIALIST_FUSION_BLOCKED_HEADS)
                    and not (target_active_heads & target_blocked_heads),
                    {
                        "active": sorted(target_active_heads),
                        "blocked": sorted(target_blocked_heads),
                    },
                ),
            ],
        ),
        _gate(
            "smart_specialist_contract",
            [
                _check("smart specialist audit exists", specialist_audit_json.exists(), _artifact_meta(specialist_audit_json)),
                _check("smart specialist audit PASS", specialist.get("decision") == "PASS", {"decision": specialist.get("decision")}),
                _check("smart specialist audit has zero failures", not specialist.get("failures"), specialist.get("failures")),
                _check(
                    "smart specialist audit uses model-native seq513 contract mode",
                    specialist.get("contract_mode") == CONTRACT_MODE,
                    {"contract_mode": specialist.get("contract_mode"), "expected": CONTRACT_MODE},
                ),
                _check(
                    "smart specialist audit has exact smart signal width",
                    int(specialist.get("signal_field_count") or 0) == EXPECTED_SIGNAL_DIM
                    and int(specialist.get("selected_feature_count") or 0) == EXPECTED_SELECTED_FEATURES,
                    {
                        "signal_field_count": specialist.get("signal_field_count"),
                        "selected_feature_count": specialist.get("selected_feature_count"),
                    },
                ),
                _check(
                    "smart specialist required training set is exact",
                    set(str(x) for x in specialist.get("required_training_specialists") or []) == set(REQUIRED_SPECIALISTS),
                    {"observed": specialist.get("required_training_specialists"), "expected": list(REQUIRED_SPECIALISTS)},
                ),
                _check(
                    "smart specialist model contract is exact",
                    bool(specialist.get("specialist_model_contract_valid"))
                    and not specialist.get("specialist_model_contract_failures")
                    and _specialist_model_contract_exact(specialist),
                    {
                        "specialist_model_contract_valid": specialist.get("specialist_model_contract_valid"),
                        "specialist_model_contract_failures": specialist.get("specialist_model_contract_failures"),
                    },
                ),
                _check(
                    "smart specialist active and blocked heads are exact",
                    specialist_active_heads == set(SPECIALIST_FUSION_ACTIVE_HEADS)
                    and specialist_blocked_heads == set(SPECIALIST_FUSION_BLOCKED_HEADS)
                    and not (specialist_active_heads & specialist_blocked_heads),
                    {
                        "active": sorted(specialist_active_heads),
                        "blocked": sorted(specialist_blocked_heads),
                    },
                ),
                _check(
                    "smart specialist routing has no unmapped signal or context fields",
                    bool(specialist.get("signal_routing_all_mapped"))
                    and int(specialist.get("signal_unmapped_count") or 0) == 0
                    and bool(specialist.get("context_routing_all_mapped"))
                    and int(specialist.get("context_routing_unmapped_count") or 0) == 0,
                    {
                        "signal_unmapped_count": specialist.get("signal_unmapped_count"),
                        "context_routing_unmapped_count": specialist.get("context_routing_unmapped_count"),
                    },
                ),
                _check(
                    "smart specialist input has no NaN inf or liveness collapse",
                    bool(specialist.get("specialist_input_liveness_all_live"))
                    and _rows_have_no_nonfinite_or_collapse(
                        specialist_rows,
                        allow_near_constant_count=True,
                    ),
                    {
                        "specialist_input_liveness_all_live": specialist.get("specialist_input_liveness_all_live"),
                        "liveness_row_count": len(specialist_rows),
                    },
                ),
                _check(
                    "trainer loader accepts exact smart specialist contract",
                    bool(trainer_probe.get("ok"))
                    and set(trainer_probe.get("loaded_specialists") or []) == set(REQUIRED_SPECIALISTS),
                    trainer_probe,
                ),
            ],
        ),
        _gate(
            "smart_smoke_dataset_manifest",
            [
                _check(
                    "explicit model-native smoke manifest event exists",
                    smoke_manifest_event_json.exists(),
                    _artifact_meta(smoke_manifest_event_json),
                ),
                _check(
                    "model-native smoke manifest event is ready",
                    smoke_manifest_readiness.get("schema_version") == SMOKE_MANIFEST_SCHEMA
                    and smoke_manifest_readiness.get("decision") == SMOKE_MANIFEST_READY_DECISION
                    and smoke_manifest_readiness.get("report_only") is True
                    and smoke_manifest_readiness.get("manifest_embedded") is True,
                    {
                        "schema_version": smoke_manifest_readiness.get("schema_version"),
                        "decision": smoke_manifest_readiness.get("decision"),
                        "report_only": smoke_manifest_readiness.get("report_only"),
                        "manifest_embedded": smoke_manifest_readiness.get("manifest_embedded"),
                    },
                ),
                _check(
                    "model-native smoke manifest proves post-rebuild orchestration provenance",
                    not missing_smoke_manifest_provenance_checks
                    and not failed_smoke_manifest_provenance_checks,
                    {
                        "required_checks": list(REQUIRED_SMOKE_MANIFEST_PROVENANCE_CHECKS),
                        "missing_checks": missing_smoke_manifest_provenance_checks,
                        "failed_checks": failed_smoke_manifest_provenance_checks,
                    },
                ),
                _check(
                    "model-native smoke manifest event hash-binds its embedded manifest",
                    smoke_manifest_readiness.get("manifest_sha256")
                    == _sha256_json(smoke_manifest),
                    {
                        "reported_manifest_sha256": smoke_manifest_readiness.get("manifest_sha256"),
                        "actual_manifest_sha256": _sha256_json(smoke_manifest),
                    },
                ),
                _check(
                    "model-native smoke manifest keeps side effects closed",
                    _all_side_effects_false(smoke_manifest_readiness, extra=("dataset_rebuild",)),
                    smoke_manifest_readiness.get("side_effects_started"),
                ),
                _check(
                    "smart smoke dataset manifest exists",
                    bool(smoke_manifest),
                    {"embedded": bool(smoke_manifest)},
                ),
                _check(
                    "smart smoke dataset manifest schema is model-native seq513",
                    smoke_manifest.get("schema_version") == SMOKE_DATASET_MANIFEST_SCHEMA,
                    {"schema_version": smoke_manifest.get("schema_version")},
                ),
                _check(
                    "smart smoke dataset manifest pins model-native seq513 candidate",
                    smoke_manifest.get("manifest_variant") == MANIFEST_VARIANT
                    and int(smoke_manifest.get("expected_seq_snap_width") or 0) == EXPECTED_SIGNAL_DIM,
                    {
                        "manifest_variant": smoke_manifest.get("manifest_variant"),
                        "expected_seq_snap_width": smoke_manifest.get("expected_seq_snap_width"),
                    },
                ),
                _check(
                    "smart smoke dataset manifest points at smart smoke dataset",
                    _dataset_path_matches(smoke_manifest, smart_smoke_dataset_dir),
                    {"out_dir": smoke_manifest.get("out_dir"), "expected": str(smart_smoke_dataset_dir)},
                ),
                _check(
                    "smart smoke dataset has train val test split hashes",
                    set(smoke_splits) == {"train", "val", "test"}
                    and all(
                        int((smoke_splits.get(split) or {}).get("rows") or 0) > 0
                        and len(str((smoke_splits.get(split) or {}).get("out_parquet_sha256") or "")) == 64
                        and len(str((smoke_splits.get(split) or {}).get("out_manifest_sha256") or "")) == 64
                        for split in ("train", "val", "test")
                    ),
                    {"splits": smoke_splits},
                ),
                _check(
                    "smart smoke split artifact files exist and hashes match manifest",
                    bool(split_artifact_hash_review["ok"]),
                    split_artifact_hash_review["details"],
                ),
                _check(
                    "smart smoke split manifests pin model-native seq513 split schema",
                    set(smoke_splits) == {"train", "val", "test"}
                    and all(
                        (smoke_splits.get(split) or {}).get("split_manifest_schema_version")
                        == SMOKE_SPLIT_MANIFEST_SCHEMA
                        for split in ("train", "val", "test")
                    ),
                    {
                        split: (smoke_splits.get(split) or {}).get("split_manifest_schema_version")
                        for split in ("train", "val", "test")
                    },
                ),
            ],
        ),
        _gate(
            "execution_hygiene",
            [
                _check(
                    "clean git required before smart smoke train",
                    not git_status,
                    {"dirty_count": len(git_status), "status_short_first_80": git_status[:80]},
                ),
            ],
        ),
        _gate(
            "future_command_contract",
            [
                _check(
                    "smart smoke train contract requires RAM cap",
                    future_contracts["smart_smoke_train"]["requires_ram_cap"] is True
                    and future_contracts["smart_smoke_train"]["ram_cap_runner"] == "scripts/gx1_capped_run.sh"
                    and bool(future_contracts["smart_smoke_train"]["memory_cap"]),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract uses only the compact exact wrapper route",
                    future_contracts["smart_smoke_train"].get("control_route")
                    == "model-native-smoke-train"
                    and future_contracts["smart_smoke_train"].get("wrapper_path")
                    == "scripts/run_entry_model_native_seq513_smoke_train.sh"
                    and future_contracts["smart_smoke_train"].get(
                        "wrapper_argv_template"
                    )
                    == future_contracts["smart_smoke_train"].get("argv_template")
                    and "gx1.models.entry_v10.entry_v10_ctx_train_v3"
                    not in " ".join(
                        future_contracts["smart_smoke_train"].get(
                            "argv_template", []
                        )
                    )
                    and "audit-smoke-bundle"
                    not in future_contracts["smart_smoke_train"].get(
                        "argv_template", []
                    ),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract pins num_workers zero",
                    future_contracts["smart_smoke_train"]["num_workers"] == 0,
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract records the unavailable immutable edge-audit route",
                    future_contracts["smart_smoke_train"]["requires_edge_audit"] is True
                    and future_contracts["smart_smoke_train"][
                        "post_smoke_audit_control_route_exposed"
                    ]
                    is False
                    and bool(
                        future_contracts["smart_smoke_train"][
                            "post_smoke_audit_blocker"
                        ]
                    ),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract declares path-quality and bad-path rank recipe",
                    future_contracts["smart_smoke_train"]["requires_path_calibration_recipe_contract"] is True
                    and _path_calibration_recipe_ok(future_contracts["smart_smoke_train"]),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract declares direction balance recipe",
                    future_contracts["smart_smoke_train"]["requires_direction_balance_recipe_contract"] is True
                    and _direction_balance_recipe_ok(future_contracts["smart_smoke_train"]),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract declares tail direction recipe",
                    future_contracts["smart_smoke_train"]["requires_tail_direction_recipe_contract"] is True
                    and _tail_direction_recipe_ok(future_contracts["smart_smoke_train"]),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract requires the exact positive training objective",
                    _training_objective_recipe_ok(
                        future_contracts["smart_smoke_train"]
                    ),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract declares direction context slice audit",
                    future_contracts["smart_smoke_train"]["requires_direction_context_slice_contract"] is True
                    and _direction_context_slice_ok(future_contracts["smart_smoke_train"]),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract declares canonical derived direction pair",
                    future_contracts["smart_smoke_train"]["requires_canonical_direction_decision_contract"]
                    is True
                    and _canonical_direction_decision_ok(future_contracts["smart_smoke_train"]),
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart smoke train contract has no replay IQL shadow or live side effects",
                    future_contracts["smart_smoke_train"]["starts_replay"] is False
                    and future_contracts["smart_smoke_train"]["starts_iql_distillation"] is False
                    and future_contracts["smart_smoke_train"]["touches_shadow_or_live"] is False,
                    future_contracts["smart_smoke_train"],
                ),
                _check(
                    "smart command contract is wired but not executed by this gate",
                    future_contracts["smart_smoke_train"]["implemented_in_control_surface"] is True
                    and future_contracts["smart_smoke_train"]["execution_allowed_now"] is False,
                    future_contracts["smart_smoke_train"],
                ),
            ],
        ),
    ]

    failures = [
        {"gate": gate["name"], **check}
        for gate in gates
        for check in gate["checks"]
        if not bool(check.get("ok"))
    ]
    ready = not failures
    decision = (
        "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW"
        if ready
        else "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_READINESS"
    )
    report = {
        "schema_version": "entry_model_native_seq513_smoke_readiness_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "report_only": True,
        "smart_candidate": {
            "manifest_variant": MANIFEST_VARIANT,
            "specialist_contract_mode": CONTRACT_MODE,
            "expected_signal_dim": EXPECTED_SIGNAL_DIM,
            "expected_selected_feature_count": EXPECTED_SELECTED_FEATURES,
            "required_training_specialists": list(REQUIRED_SPECIALISTS),
        },
        "training_allowed": False,
        "training_allowed_reason": (
            "report-only readiness design; a future smart smoke wrapper requires one immutable run "
            "lineage, clean git and all evidence gates, and this gate never starts training"
        ),
        "smart_smoke_manifest_allowed": bool(ready),
        "smart_smoke_training_allowed": False,
        "smart_trainability_readiness_required_before_training": True,
        "execution_allowed_now": False,
        "control_surface_mutated": False,
        "side_effects_started": {
            "dataset_rebuild": False,
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
        },
        "inputs": {
            "smart_post_rebuild_readiness": _artifact_meta(post_rebuild_readiness_json),
            "full_input_liveness_contract": _artifact_meta(full_input_liveness_json),
            "model_native_rebuild_preflight": _artifact_meta(rebuild_preflight_json),
            "smart_feature_audit": _artifact_meta(feature_audit_json),
            "smart_target_audit": _artifact_meta(target_audit_json),
            "smart_specialist_audit": _artifact_meta(specialist_audit_json),
            "model_native_smoke_manifest_event": _artifact_meta(smoke_manifest_event_json),
            "smart_dataset_dir": str(smart_dataset_dir),
            "smart_smoke_dataset_dir": str(smart_smoke_dataset_dir),
        },
        "future_command_contracts": future_contracts,
        "full_input_liveness_validation": full_input_liveness_validation,
        "gates": gates,
        "failures": failures,
        "blockers": [f"{row['gate']}: {row['name']}" for row in failures],
        "next_required_gate": (
            "run the capped smart dataset rebuild under one immutable run lineage, then smart feature/target/"
            "specialist audits, smart smoke dataset manifest, clean git and a separate smart smoke wrapper review; "
            "do not start candidate training, replay, IQL, shadow or live"
        ),
    }
    report["evidence_binding_sha256"] = _sha256_json(report["inputs"])
    _, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-post-rebuild-readiness-json", required=True)
    ap.add_argument("--full-input-liveness-json", required=True)
    ap.add_argument("--model-native-rebuild-preflight-json", required=True)
    ap.add_argument("--smart-dataset-dir", required=True)
    ap.add_argument("--smart-smoke-dataset-dir", required=True)
    ap.add_argument("--smart-feature-audit-json", required=True)
    ap.add_argument("--smart-target-audit-json", required=True)
    ap.add_argument("--smart-specialist-audit-json", required=True)
    ap.add_argument("--smoke-manifest-event-json", required=True)
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--memory-cap", default="22G")
    ap.add_argument("--swap-cap", default="2G")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
