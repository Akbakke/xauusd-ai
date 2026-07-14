#!/usr/bin/env python3
"""Report-only smart_seq520 smoke-readiness gate.

This gate does not rebuild data, write a smoke train manifest, start training,
run replay, distill IQL, touch shadow, or touch live paths. It records the
fail-closed requirements that must be true before a future smart_seq520 smoke
manifest or smoke train wrapper can be reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gx1.features.entry_specialist_feature_groups_v1 import (
    SPECIALIST_FUSION_ACTIVE_HEADS,
    SPECIALIST_FUSION_BLOCKED_HEADS,
    required_training_specialists_for_mode,
    specialist_model_contract_for_mode,
)
from gx1.scripts.verify_entry_foundation_state_v1 import (
    FOUNDATION_DATASET_DIR,
    REPO,
    REPORTS_ROOT,
)


CONTRACT_MODE = "smart_seq520_candidate"
MANIFEST_VARIANT = "smart_seq520_candidate"
EXPECTED_SIGNAL_DIM = 520
EXPECTED_BASE_SIGNAL_FEATURES = 41
EXPECTED_SELECTED_FEATURES = EXPECTED_SIGNAL_DIM - EXPECTED_BASE_SIGNAL_FEATURES
REQUIRED_SPECIALISTS = tuple(required_training_specialists_for_mode(CONTRACT_MODE))
EXPECTED_MODEL_CONTRACT = specialist_model_contract_for_mode(CONTRACT_MODE)

DEFAULT_SMART_REBUILD_PREFLIGHT = (
    REPORTS_ROOT / "entry_smart_seq_rebuild_preflight_20260630_v1/ENTRY_SMART_REBUILD_PREFLIGHT_latest.json"
)
DEFAULT_SMART_DATASET_DIR = (
    FOUNDATION_DATASET_DIR.parent.parent
    / "v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair"
)
DEFAULT_SMART_SMOKE_DATASET_DIR = (
    FOUNDATION_DATASET_DIR.parent.parent
    / "v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair_smoke"
)
DEFAULT_FEATURE_AUDIT = (
    REPORTS_ROOT
    / "entry_feature_foundation_audit_20260628_v1/smart_seq520_candidate_20260630/ENTRY_FEATURE_FOUNDATION_AUDIT_latest.json"
)
DEFAULT_TARGET_AUDIT = (
    REPORTS_ROOT
    / "entry_target_foundation_audit_20260628_v1/smart_seq520_candidate_20260630/ENTRY_TARGET_FOUNDATION_AUDIT_latest.json"
)
DEFAULT_SPECIALIST_AUDIT = (
    REPORTS_ROOT
    / "entry_specialist_feature_group_audit_20260628_v1/smart_seq520_candidate_20260630/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
)
DEFAULT_SMOKE_DATASET_MANIFEST = (
    REPORTS_ROOT / "entry_smart_seq520_smoke_manifest_20260630_v1/SMOKE_DATASET_MANIFEST.json"
)
DEFAULT_SMOKE_MANIFEST_READINESS = (
    REPORTS_ROOT
    / "entry_smart_seq520_smoke_manifest_20260630_v1/ENTRY_SMART_SEQ520_SMOKE_MANIFEST_READINESS_latest.json"
)
DEFAULT_OUT_DIR = REPORTS_ROOT / "entry_smart_seq520_smoke_readiness_20260630_v1"
SMOKE_MANIFEST_READINESS_READY_DECISION = "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW"
SMOKE_MANIFEST_READINESS_SCHEMA = "entry_smart_seq520_smoke_manifest_readiness_v1"
SMOKE_DATASET_MANIFEST_SCHEMA = "entry_smart_seq520_smoke_dataset_v1"
SMOKE_SPLIT_MANIFEST_SCHEMA = "entry_smart_seq520_smoke_split_manifest_v1"
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
    "direction_min_pred_rate_softmax_temperature": 0.15,
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
    "direction_slice_loss_aggregation": "mean",
    "direction_vs_flat_margin_weight": 4.00,
    "direction_vs_flat_margin": 0.10,
    "hier_legacy_ce_mult": 1.00,
    "hier_trade_weight": 2.00,
    "hier_side_weight": 1.75,
    "hier_utility_weight": 1.00,
    "hier_bad_path_weight": 1.25,
    "hier_mae_weight": 0.35,
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
    "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE": "0.15",
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
    "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION": "mean",
    "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT": "4.00",
    "ENTRY_DIRECTION_VS_FLAT_MARGIN": "0.10",
    "ENTRY_HIER_LEGACY_CE_MULT": "1.00",
    "ENTRY_HIER_TRADE_WEIGHT": "2.00",
    "ENTRY_HIER_SIDE_WEIGHT": "1.75",
    "ENTRY_HIER_UTILITY_WEIGHT": "1.00",
    "ENTRY_HIER_BAD_PATH_WEIGHT": "1.25",
    "ENTRY_HIER_MAE_WEIGHT": "0.35",
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
    env_template = contract.get("path_calibration_env_template")
    if not isinstance(recipe, dict) or not isinstance(env_template, dict):
        return False
    if recipe != PATH_CALIBRATION_RECIPE_CONTRACT:
        return False
    return all(env_template.get(key) == value for key, value in PATH_CALIBRATION_ENV_TEMPLATE.items())


def _direction_balance_recipe_ok(contract: dict[str, Any]) -> bool:
    recipe = contract.get("direction_balance_recipe_contract")
    env_template = contract.get("direction_balance_env_template")
    if not isinstance(recipe, dict) or not isinstance(env_template, dict):
        return False
    if recipe != DIRECTION_BALANCE_RECIPE_CONTRACT:
        return False
    return all(env_template.get(key) == value for key, value in DIRECTION_BALANCE_ENV_TEMPLATE.items())


def _tail_direction_recipe_ok(contract: dict[str, Any]) -> bool:
    recipe = contract.get("tail_direction_recipe_contract")
    env_template = contract.get("tail_direction_env_template")
    if not isinstance(recipe, dict) or not isinstance(env_template, dict):
        return False
    if recipe != TAIL_DIRECTION_RECIPE_CONTRACT:
        return False
    return all(env_template.get(key) == value for key, value in TAIL_DIRECTION_ENV_TEMPLATE.items())


def _direction_context_slice_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("requires_direction_context_slice_contract") is True
        and contract.get("direction_context_slice_contract") == DIRECTION_CONTEXT_SLICE_CONTRACT
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
    smart_smoke_dataset_dir: Path,
    specialist_audit_json: Path,
    memory_cap: str,
    swap_cap: str,
) -> dict[str, Any]:
    train_parquet = smart_smoke_dataset_dir / "v10_smart_seq520_smoke__HOLD_03B_train.parquet"
    out_bundle = (
        "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
        "v10_6yr_rebuild_20260628_foundation_seq146/v10_entry_smart_seq520_smoke_<STAMP>"
    )
    env_prefix = [
        "env",
        *[f"{key}={value}" for key, value in PATH_CALIBRATION_ENV_TEMPLATE.items()],
        *[f"{key}={value}" for key, value in DIRECTION_BALANCE_ENV_TEMPLATE.items()],
        *[f"{key}={value}" for key, value in TAIL_DIRECTION_ENV_TEMPLATE.items()],
    ]
    train_inner = [
        *env_prefix,
        ".venv/bin/python",
        "-m",
        "gx1.models.entry_v10.entry_v10_ctx_train_v3",
        "--train",
        "--vedtak",
        "<SMART_SEQ520_SMOKE_VEDTAK_ID>",
        "--dataset_dir",
        str(smart_smoke_dataset_dir),
        "--dataset_train_parquet",
        str(train_parquet),
        "--out_bundle_dir",
        out_bundle,
        "--epochs",
        "1",
        "--batch_size",
        "64",
        "--num-workers",
        "0",
        "--multi-tf-seq-len",
        "16",
        "--per-tf-seq-len-h4",
        "8",
        "--per-tf-seq-len-d1",
        "8",
        "--enable-specialist-fusion",
        "--specialist-audit-json",
        str(specialist_audit_json),
        "--specialist-contract-mode",
        CONTRACT_MODE,
        "--enable-xau-direction-repair-heads",
        "--anchor-gate-init",
        "0.0",
    ]
    audit = [
        "scripts/entry_next_edge_control.sh",
        "audit-smoke-bundle",
        "--bundle-dir",
        out_bundle,
        "--dataset-dir",
        str(smart_smoke_dataset_dir),
        "--require-head-contract",
        "--require-edge",
        "--pretrain-manifest-json",
        "<SMART_SEQ520_PRETRAIN_MANIFEST_JSON>",
    ]
    train = {
        "argv_template": [
            "scripts/gx1_capped_run.sh",
            "--mem",
            memory_cap,
            "--swap",
            swap_cap,
            "--",
            *train_inner,
        ],
        "inner_train_argv_template": train_inner,
        "mode": "future_train_contract",
        "implemented_in_control_surface": True,
        "execution_allowed_now": False,
        "requires_explicit_vedtak": True,
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
        "requires_path_calibration_recipe_contract": True,
        "path_calibration_recipe_contract": dict(PATH_CALIBRATION_RECIPE_CONTRACT),
        "path_calibration_env_template": dict(PATH_CALIBRATION_ENV_TEMPLATE),
        "requires_direction_balance_recipe_contract": True,
        "direction_balance_recipe_contract": dict(DIRECTION_BALANCE_RECIPE_CONTRACT),
        "direction_balance_env_template": dict(DIRECTION_BALANCE_ENV_TEMPLATE),
        "requires_tail_direction_recipe_contract": True,
        "tail_direction_recipe_contract": dict(TAIL_DIRECTION_RECIPE_CONTRACT),
        "tail_direction_env_template": dict(TAIL_DIRECTION_ENV_TEMPLATE),
        "requires_direction_context_slice_contract": True,
        "direction_context_slice_contract": dict(DIRECTION_CONTEXT_SLICE_CONTRACT),
        "post_smoke_audit_argv_template": audit,
        "specialist_contract_mode": CONTRACT_MODE,
        "expected_signal_dim": EXPECTED_SIGNAL_DIM,
        "required_training_specialists": list(REQUIRED_SPECIALISTS),
        "requires_exact_specialist_contract_proof": True,
    }
    manifest = {
        "mode": "future_manifest_only_contract",
        "implemented_in_control_surface": False,
        "execution_allowed_now": False,
        "requires_explicit_vedtak": True,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    smart_dataset_dir = Path(args.smart_dataset_dir).expanduser().resolve()
    smart_smoke_dataset_dir = Path(args.smart_smoke_dataset_dir).expanduser().resolve()
    rebuild_preflight_json = Path(args.smart_rebuild_preflight_json).expanduser().resolve()
    feature_audit_json = Path(args.smart_feature_audit_json).expanduser().resolve()
    target_audit_json = Path(args.smart_target_audit_json).expanduser().resolve()
    specialist_audit_json = Path(args.smart_specialist_audit_json).expanduser().resolve()
    smoke_dataset_manifest_json = Path(args.smart_smoke_dataset_manifest_json).expanduser().resolve()
    smoke_manifest_readiness_json = Path(
        getattr(args, "smart_smoke_dataset_manifest_readiness_json", DEFAULT_SMOKE_MANIFEST_READINESS)
    ).expanduser().resolve()

    rebuild = _read_json_or_empty(rebuild_preflight_json)
    feature = _read_json_or_empty(feature_audit_json)
    target = _read_json_or_empty(target_audit_json)
    specialist = _read_json_or_empty(specialist_audit_json)
    smoke_manifest = _read_json_or_empty(smoke_dataset_manifest_json)
    smoke_manifest_readiness = _read_json_or_empty(smoke_manifest_readiness_json)
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
        smart_smoke_dataset_dir=smart_smoke_dataset_dir,
        specialist_audit_json=specialist_audit_json,
        memory_cap=str(args.memory_cap),
        swap_cap=str(args.swap_cap),
    )

    rebuild_counts = rebuild.get("counts") if isinstance(rebuild.get("counts"), dict) else {}
    target_head_contract = _target_head_contract(target)
    target_active_heads = {str(x) for x in target_head_contract.get("active_training_heads") or []}
    target_blocked_heads = {str(x) for x in target_head_contract.get("blocked_heads") or []}
    specialist_active_heads, specialist_blocked_heads = _recommended_heads(specialist)
    specialist_rows = _liveness_rows(specialist)
    feature_rows = _liveness_rows(feature)
    smoke_splits = smoke_manifest.get("splits") if isinstance(smoke_manifest.get("splits"), dict) else {}
    split_artifact_hash_review = _split_artifact_hash_review(smoke_splits)

    gates = [
        _gate(
            "smart_rebuild_preflight",
            [
                _check("smart rebuild preflight exists", rebuild_preflight_json.exists(), _artifact_meta(rebuild_preflight_json)),
                _check(
                    "smart rebuild preflight is ready for vedtak review",
                    rebuild.get("decision") == "READY_FOR_SMART_REBUILD_VEDTAK_REVIEW",
                    {"decision": rebuild.get("decision")},
                ),
                _check("smart rebuild preflight is report-only", rebuild.get("report_only") is True, rebuild.get("report_only")),
                _check(
                    "smart rebuild preflight keeps dataset rebuild closed without vedtak",
                    rebuild.get("dataset_rebuild_allowed_without_vedtak") is False,
                    rebuild.get("dataset_rebuild_allowed_without_vedtak"),
                ),
                _check("smart rebuild preflight keeps training closed", rebuild.get("training_allowed") is False, rebuild.get("training_allowed")),
                _check(
                    "smart rebuild preflight side effects are closed",
                    _all_side_effects_false(rebuild, extra=("dataset_rebuild",)),
                    rebuild.get("side_effects_started"),
                ),
                _check(
                    "smart rebuild preflight pins smart_seq520_candidate",
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
                    "smart specialist audit uses smart_seq520_candidate contract mode",
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
                    "latest smart smoke manifest readiness report exists",
                    smoke_manifest_readiness_json.exists(),
                    _artifact_meta(smoke_manifest_readiness_json),
                ),
                _check(
                    "latest smart smoke manifest readiness report is ready",
                    smoke_manifest_readiness.get("schema_version") == SMOKE_MANIFEST_READINESS_SCHEMA
                    and smoke_manifest_readiness.get("decision") == SMOKE_MANIFEST_READINESS_READY_DECISION
                    and smoke_manifest_readiness.get("report_only") is True
                    and smoke_manifest_readiness.get("manifest_written") is True,
                    {
                        "schema_version": smoke_manifest_readiness.get("schema_version"),
                        "decision": smoke_manifest_readiness.get("decision"),
                        "report_only": smoke_manifest_readiness.get("report_only"),
                        "manifest_written": smoke_manifest_readiness.get("manifest_written"),
                    },
                ),
                _check(
                    "latest smart smoke manifest readiness proves post-rebuild orchestration provenance",
                    not missing_smoke_manifest_provenance_checks
                    and not failed_smoke_manifest_provenance_checks,
                    {
                        "required_checks": list(REQUIRED_SMOKE_MANIFEST_PROVENANCE_CHECKS),
                        "missing_checks": missing_smoke_manifest_provenance_checks,
                        "failed_checks": failed_smoke_manifest_provenance_checks,
                    },
                ),
                _check(
                    "latest smart smoke manifest readiness points at this manifest",
                    _path_equals(smoke_manifest_readiness.get("manifest_path"), smoke_dataset_manifest_json)
                    and smoke_manifest_readiness.get("manifest_sha256") == _sha256_file(smoke_dataset_manifest_json),
                    {
                        "manifest_path": smoke_manifest_readiness.get("manifest_path"),
                        "expected_manifest_path": str(smoke_dataset_manifest_json),
                        "reported_manifest_sha256": smoke_manifest_readiness.get("manifest_sha256"),
                        "actual_manifest_sha256": _sha256_file(smoke_dataset_manifest_json),
                    },
                ),
                _check(
                    "latest smart smoke manifest readiness keeps side effects closed",
                    _all_side_effects_false(smoke_manifest_readiness, extra=("dataset_rebuild",)),
                    smoke_manifest_readiness.get("side_effects_started"),
                ),
                _check(
                    "smart smoke dataset manifest exists",
                    smoke_dataset_manifest_json.exists(),
                    _artifact_meta(smoke_dataset_manifest_json),
                ),
                _check(
                    "smart smoke dataset manifest schema is smart seq520",
                    smoke_manifest.get("schema_version") == SMOKE_DATASET_MANIFEST_SCHEMA,
                    {"schema_version": smoke_manifest.get("schema_version")},
                ),
                _check(
                    "smart smoke dataset manifest pins smart_seq520_candidate",
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
                    "smart smoke split manifests pin smart seq520 split schema",
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
                    "smart smoke train contract pins num_workers zero",
                    future_contracts["smart_smoke_train"]["num_workers"] == 0
                    and "--num-workers" in future_contracts["smart_smoke_train"]["inner_train_argv_template"]
                    and "0" in future_contracts["smart_smoke_train"]["inner_train_argv_template"],
                    future_contracts["smart_smoke_train"]["inner_train_argv_template"],
                ),
                _check(
                    "smart smoke train contract requires edge audit",
                    future_contracts["smart_smoke_train"]["requires_edge_audit"] is True
                    and "--require-edge" in future_contracts["smart_smoke_train"]["post_smoke_audit_argv_template"],
                    future_contracts["smart_smoke_train"]["post_smoke_audit_argv_template"],
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
                    "smart smoke train contract declares direction context slice audit",
                    future_contracts["smart_smoke_train"]["requires_direction_context_slice_contract"] is True
                    and _direction_context_slice_ok(future_contracts["smart_smoke_train"]),
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
    decision = "READY_FOR_SMART_SEQ520_SMOKE_MANIFEST_REVIEW" if ready else "BLOCKED_SMART_SEQ520_SMOKE_READINESS"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "entry_smart_seq520_smoke_readiness_v1",
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
            "report-only readiness design; a future smart smoke wrapper still requires explicit vedtak, "
            "clean git and all gates, and this gate never starts training"
        ),
        "smart_smoke_manifest_allowed_without_vedtak": False,
        "smart_smoke_manifest_allowed_after_explicit_vedtak_and_gates": bool(ready),
        "smart_smoke_training_allowed_without_vedtak": False,
        "smart_smoke_training_allowed_after_explicit_vedtak_and_gates": False,
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
            "smart_rebuild_preflight": _artifact_meta(rebuild_preflight_json),
            "smart_feature_audit": _artifact_meta(feature_audit_json),
            "smart_target_audit": _artifact_meta(target_audit_json),
            "smart_specialist_audit": _artifact_meta(specialist_audit_json),
            "smart_smoke_dataset_manifest": _artifact_meta(smoke_dataset_manifest_json),
            "smart_smoke_dataset_manifest_readiness": _artifact_meta(smoke_manifest_readiness_json),
            "smart_dataset_dir": str(smart_dataset_dir),
            "smart_smoke_dataset_dir": str(smart_smoke_dataset_dir),
        },
        "future_command_contracts": future_contracts,
        "gates": gates,
        "failures": failures,
        "blockers": [f"{row['gate']}: {row['name']}" for row in failures],
        "next_required_gate": (
            "run the capped smart dataset rebuild only after explicit rebuild vedtak, then smart feature/target/"
            "specialist audits, smart smoke dataset manifest, clean git and a separate smart smoke wrapper review; "
            "do not start candidate training, replay, IQL, shadow or live"
        ),
    }
    json_path = out_dir / f"ENTRY_SMART_SEQ520_SMOKE_READINESS_{stamp}.json"
    md_path = out_dir / f"ENTRY_SMART_SEQ520_SMOKE_READINESS_{stamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (out_dir / "ENTRY_SMART_SEQ520_SMOKE_READINESS_latest.json").write_text(
        json_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    md_lines = [
        "# Entry Smart Seq520 Smoke Readiness",
        "",
        f"- decision: `{decision}`",
        "- report_only: `true`",
        "- training_allowed: `false`",
        f"- manifest_variant: `{MANIFEST_VARIANT}`",
        f"- expected_signal_dim: `{EXPECTED_SIGNAL_DIM}`",
        f"- failures: `{len(failures)}`",
        "",
        "Next: smart dataset rebuild/audits first, then separate smart smoke wrapper review.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "ENTRY_SMART_SEQ520_SMOKE_READINESS_latest.md").write_text(
        md_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures and not args.no_fail_on_not_ready:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-rebuild-preflight-json", default=str(DEFAULT_SMART_REBUILD_PREFLIGHT))
    ap.add_argument("--smart-dataset-dir", default=str(DEFAULT_SMART_DATASET_DIR))
    ap.add_argument("--smart-smoke-dataset-dir", default=str(DEFAULT_SMART_SMOKE_DATASET_DIR))
    ap.add_argument("--smart-feature-audit-json", default=str(DEFAULT_FEATURE_AUDIT))
    ap.add_argument("--smart-target-audit-json", default=str(DEFAULT_TARGET_AUDIT))
    ap.add_argument("--smart-specialist-audit-json", default=str(DEFAULT_SPECIALIST_AUDIT))
    ap.add_argument("--smart-smoke-dataset-manifest-json", default=str(DEFAULT_SMOKE_DATASET_MANIFEST))
    ap.add_argument(
        "--smart-smoke-dataset-manifest-readiness-json",
        default=str(DEFAULT_SMOKE_MANIFEST_READINESS),
    )
    ap.add_argument("--repo-dir", default=str(REPO))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--memory-cap", default="22G")
    ap.add_argument("--swap-cap", default="2G")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-fail-on-not-ready", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
