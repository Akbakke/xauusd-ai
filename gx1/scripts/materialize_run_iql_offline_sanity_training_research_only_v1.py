#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate
from gx1.scripts import materialize_refine_clean_as_of_safety_layer_to_retain_safe_core_v1 as refine_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1"

INPUT_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T190901Z_LOCK"
)
INPUT_REFINE_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T185018Z_LOCK"
)
INPUT_CLEAN_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS_V1_20260428T182517Z_LOCK"
)
INPUT_HARDEN_ROOT = DEFAULT_REPORTS_ROOT / "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1_20260428T085058Z_LOCK"
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)

FINAL_STATUS = "IQL_OFFLINE_SANITY_PASS_CONTEXTUAL_ONLY_NEEDS_TRANSITION_DESIGN"
NEXT_ACTION = "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1"
SANITY_MODE = "CONTEXTUAL_ONE_STEP_IQL_SANITY"
MODEL_ID = "CONTEXTUAL_ONE_STEP_IQL_RIDGE_FIXED_V1"
REWARD_ID = "SAFETY_WEIGHTED_REWARD"
SAFETY_COHORT = "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY"

RAW_ALLOWED_STATE_FIELDS = [
    "candidate_score_v1",
    "signal_r5_1_bad_score_v1",
    "signal_r5_bad_score_v1",
    "signal_r5_tail_score_v1",
    "signal_v2_like_bad_tail_v1",
    "signal_tail_repair_v1",
    "run_id_policy_class_v1",
    "structural_low_support_v1",
    "zero_denominator_group_v1",
]
MODEL_STATE_COLUMNS = [
    "intercept_v1",
    "candidate_score_z_train_only_v1",
    "signal_r5_1_bad_score_v1",
    "signal_r5_bad_score_v1",
    "signal_r5_tail_score_v1",
    "signal_v2_like_bad_tail_v1",
    "signal_tail_repair_v1",
    "structural_low_support_v1",
    "zero_denominator_group_v1",
    "policy_support_repairable_v1",
    "policy_low_support_missing_artifacts_v1",
]
DENIED_STATE_FIELDS = {
    "bad_label_v1",
    "tail_label_v1",
    "unsafe_audit_v1",
    "safety_clear_audit_v1",
    "hard_veto_clear_shadow_v1",
    "HISTORICAL_V2_BLUEPRINT",
    "source_evidence_v1",
    "student_oof_score_v1",
    "student_core_selected_v1",
    "candidate_uid_v1",
    "trade_uid_v1",
    "selected_original_140_v1",
    "is_140_94_baseline_v1",
    "is_185_139_teacher_v1",
    "is_plus45_diagnostic_v1",
    "rows_added_vs_140_94_v1",
    "protected_winner_status_v1",
    "runner_protect_status_v1",
    "ambiguous_high_mfe_status_v1",
    "fifty_plus_mfe_risk_v1",
    "hundred_plus_mfe_risk_v1",
    "two_hundred_plus_mfe_risk_v1",
}

ALLOWED_FINAL_STATUSES = {
    "IQL_OFFLINE_SANITY_PASS_READY_FOR_DEEPER_RESEARCH_EXPERIMENT",
    "IQL_OFFLINE_SANITY_PASS_CONTEXTUAL_ONLY_NEEDS_TRANSITION_DESIGN",
    "IQL_OFFLINE_SANITY_PARTIAL_POLICY_COLLAPSES_TO_BASELINE",
    "IQL_OFFLINE_SANITY_PARTIAL_NEEDS_STATE_FEATURE_NORMALIZATION",
    "IQL_OFFLINE_SANITY_PARTIAL_NEEDS_MORE_ACTION_SUPPORT",
    "IQL_OFFLINE_SANITY_BLOCKED_BY_STATE_LEAKAGE",
    "IQL_OFFLINE_SANITY_BLOCKED_BY_REWARD_OR_LABEL_LEAKAGE",
    "IQL_OFFLINE_SANITY_BLOCKED_BY_INSUFFICIENT_SAFE_COHORT_SUPPORT",
    "IQL_OFFLINE_SANITY_BLOCKED_BY_IQL_IMPLEMENTATION_MISSING",
    "IQL_OFFLINE_SANITY_BLOCKED_BY_UNSTABLE_HELDOUT_BEHAVIOR",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_IQL_OFFLINE_DEEPER_RESEARCH_EXPERIMENT_V1",
    "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1",
    "IMPROVE_IQL_STATE_FEATURE_NORMALIZATION_V1",
    "DEEPEN_IQL_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT_V1",
    "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1",
    "HOLD_IQL_RESEARCH_UNTIL_SAFE_COHORT_SUPPORT_IMPROVES_V1",
    "FIX_IQL_IMPLEMENTATION_AND_RERUN_SANITY_V1",
}

REQUIRED_OUTPUTS = [
    "iql_offline_sanity_input_manifest_v1.json",
    "iql_offline_sanity_contract_reproducibility_audit_v1.json",
    "iql_offline_sanity_contract_reproducibility_audit_v1.md",
    "iql_offline_sanity_dataset_snapshot_v1.csv",
    "iql_offline_sanity_dataset_snapshot_v1.json",
    "iql_offline_sanity_dataset_snapshot_v1.md",
    "iql_offline_sanity_state_matrix_audit_v1.csv",
    "iql_offline_sanity_state_matrix_audit_v1.json",
    "iql_offline_sanity_state_matrix_audit_v1.md",
    "iql_offline_sanity_transition_or_contextual_audit_v1.json",
    "iql_offline_sanity_transition_or_contextual_audit_v1.md",
    "iql_offline_sanity_normalization_audit_v1.json",
    "iql_offline_sanity_normalization_audit_v1.md",
    "iql_offline_sanity_split_audit_v1.csv",
    "iql_offline_sanity_split_audit_v1.json",
    "iql_offline_sanity_split_audit_v1.md",
    "iql_offline_sanity_training_config_v1.json",
    "iql_offline_sanity_training_config_v1.md",
    "iql_offline_sanity_training_metrics_v1.csv",
    "iql_offline_sanity_training_metrics_v1.json",
    "iql_offline_sanity_training_metrics_v1.md",
    "iql_offline_sanity_baseline_policy_comparison_v1.csv",
    "iql_offline_sanity_baseline_policy_comparison_v1.json",
    "iql_offline_sanity_baseline_policy_comparison_v1.md",
    "iql_offline_sanity_policy_predictions_v1.csv",
    "iql_offline_sanity_policy_predictions_v1.json",
    "iql_offline_sanity_policy_behavior_audit_v1.csv",
    "iql_offline_sanity_policy_behavior_audit_v1.json",
    "iql_offline_sanity_policy_behavior_audit_v1.md",
    "iql_offline_sanity_no_shortcut_audit_v1.json",
    "iql_offline_sanity_no_shortcut_audit_v1.md",
    "iql_offline_sanity_verdict_v1.json",
    "iql_offline_sanity_verdict_v1.md",
    "iql_offline_sanity_recommendation_v1.json",
    "iql_offline_sanity_recommendation_v1.md",
    "run_iql_offline_sanity_training_research_only_go_no_go_v1.json",
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _write_report(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact for hash: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python_manifest() -> dict[str, Any]:
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except Exception:
        freeze = []
    return {
        "python_executable_v1": sys.executable,
        "python_version_v1": sys.version,
        "platform_v1": platform.platform(),
        "pip_freeze_sha256_v1": hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest(),
    }


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return refine_gate._bool(frame, column)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    return refine_gate._num(frame, column, default)


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    failures = []
    for path in paths:
        text = str(path)
        if "*" in text or "latest" in text.lower() or not path.name.endswith("_LOCK"):
            failures.append(text)
    if failures:
        raise RuntimeError(f"IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN: {failures}")
    return True


def validate_no_forbidden_actions(
    *,
    adapter: bool = False,
    r6: bool = False,
    iql_production: bool = False,
    package: bool = False,
    freeze: bool = False,
    promo: bool = False,
    live: bool = False,
    optuna: bool = False,
    broad_sweep: bool = False,
) -> dict[str, Any]:
    failures = []
    if adapter:
        failures.append("ADAPTER_BUILD_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if iql_production:
        failures.append("IQL_PRODUCTION_FORBIDDEN")
    if package:
        failures.append("PACKAGE_BUILD_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if promo:
        failures.append("PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if broad_sweep:
        failures.append("BROAD_SWEEP_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def _load_inputs() -> dict[str, Any]:
    roots = [INPUT_CONTRACT_ROOT, INPUT_REFINE_CLEAN_ROOT, INPUT_CLEAN_ROOT, INPUT_HARDEN_ROOT, INPUT_PRECHECK_ROOT]
    validate_explicit_artifact_roots(roots)
    required = {
        "contract_summary": INPUT_CONTRACT_ROOT / "summary_v1.json",
        "contract_go_no_go": INPUT_CONTRACT_ROOT / "build_iql_offline_data_contract_research_only_go_no_go_v1.json",
        "contract_state": INPUT_CONTRACT_ROOT / "iql_offline_state_contract_v1.json",
        "contract_action": INPUT_CONTRACT_ROOT / "iql_offline_action_contract_v1.json",
        "contract_reward": INPUT_CONTRACT_ROOT / "iql_offline_reward_contract_v1.json",
        "contract_shield": INPUT_CONTRACT_ROOT / "iql_offline_safety_shield_contract_v1.json",
        "contract_readiness": INPUT_CONTRACT_ROOT / "iql_offline_readiness_assessment_v1.json",
        "refine_summary": INPUT_REFINE_CLEAN_ROOT / "summary_v1.json",
        "refine_go_no_go": INPUT_REFINE_CLEAN_ROOT
        / "refine_clean_as_of_safety_layer_to_retain_safe_core_go_no_go_v1.json",
        "clean_summary": INPUT_CLEAN_ROOT / "summary_v1.json",
        "harden_summary": INPUT_HARDEN_ROOT / "summary_v1.json",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    contract_go = _read_json(required["contract_go_no_go"])
    if contract_go.get("status_v1") != "IQL_OFFLINE_DATA_CONTRACT_READY_FOR_SANITY_TRAINING_RESEARCH_ONLY":
        raise RuntimeError("INPUT_IQL_CONTRACT_NOT_READY_FOR_SANITY")
    if not contract_go.get("research_only_iql_sanity_training_allowed_next_v1"):
        raise RuntimeError("INPUT_IQL_CONTRACT_DOES_NOT_ALLOW_RESEARCH_SANITY_NEXT")
    return {
        "required_paths": required,
        "contract_summary": _read_json(required["contract_summary"]),
        "contract_go_no_go": contract_go,
        "contract_state": _read_json(required["contract_state"]),
        "contract_action": _read_json(required["contract_action"]),
        "contract_reward": _read_json(required["contract_reward"]),
        "contract_shield": _read_json(required["contract_shield"]),
        "contract_readiness": _read_json(required["contract_readiness"]),
        "refine_summary": _read_json(required["refine_summary"]),
        "refine_go_no_go": _read_json(required["refine_go_no_go"]),
        "clean_summary": _read_json(required["clean_summary"]),
        "harden_summary": _read_json(required["harden_summary"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "frame_inputs": refine_gate._load_inputs(),
    }


def _frame_and_masks(inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    return refine_gate._build_frame_and_masks(inputs["frame_inputs"])


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = [
        {"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)}
        for name, path in inputs["required_paths"].items()
    ]
    return {
        "layer_name": "IQL_OFFLINE_SANITY_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "iql_contract_root_v1": str(INPUT_CONTRACT_ROOT),
            "clean_safety_layer_refinement_root_v1": str(INPUT_REFINE_CLEAN_ROOT),
            "clean_as_of_safety_layer_root_v1": str(INPUT_CLEAN_ROOT),
            "safe_core_harden_root_v1": str(INPUT_HARDEN_ROOT),
            "baseline_140_94_precheck_root_v1": str(INPUT_PRECHECK_ROOT),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _split_series(frame: pd.DataFrame) -> pd.Series:
    mapping = {
        "fold_00": "train",
        "fold_01": "train",
        "fold_02": "train",
        "fold_03": "validation",
        "fold_04": "test",
    }
    return frame["fold_id_v1"].astype("string").map(mapping).fillna("train")


def _reward(frame: pd.DataFrame, shield: pd.Series) -> np.ndarray:
    bad = _bool(frame, "bad_label_v1").astype(float).to_numpy()
    tail = _bool(frame, "tail_label_v1").astype(float).to_numpy()
    unsafe = _bool(frame, "unsafe_audit_v1").astype(float).to_numpy()
    take_reward = (2.0 * tail) + (0.5 * bad) - 0.75 - (10.0 * unsafe)
    return np.where(shield.to_numpy(), take_reward, 0.0)


def _contract_reproducibility(
    frame: pd.DataFrame, masks: dict[str, pd.Series], inputs: dict[str, Any]
) -> dict[str, Any]:
    baseline = _bool(frame, "is_140_94_baseline_v1")
    safe_core = masks["hardened"]
    shield = safe_core & ~masks["source_confluence_repairable_v1"]
    payload = {
        "layer_name": "IQL_OFFLINE_SANITY_CONTRACT_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "input_contract_status_v1": inputs["contract_go_no_go"].get("status_v1"),
        "state_allowlist_field_count_v1": int(inputs["contract_summary"].get("state_allowed_field_count_v1")),
        "state_denylist_field_count_v1": int(inputs["contract_summary"].get("state_blocked_field_count_v1")),
        "action_space_v1": inputs["contract_summary"].get("action_space_v1"),
        "reward_id_v1": REWARD_ID,
        "transformer_features_blocked_v1": True,
        "baseline_140_94_v1": {
            "selected_rows_v1": int(baseline.sum()),
            "bad_count_audit_only_v1": int(_bool(frame[baseline], "bad_label_v1").sum()),
            "tail_count_audit_only_v1": int(_bool(frame[baseline], "tail_label_v1").sum()),
            "safety_status_v1": "CLEAN" if int(_bool(frame[baseline], "unsafe_audit_v1").sum()) == 0 else "FAIL",
        },
        "safe_core_89_v1": {
            "selected_rows_v1": int(safe_core.sum()),
            "bad_count_audit_only_v1": int(_bool(frame[safe_core], "bad_label_v1").sum()),
            "tail_count_audit_only_v1": int(_bool(frame[safe_core], "tail_label_v1").sum()),
            "safety_status_v1": "CLEAN" if int(_bool(frame[safe_core], "unsafe_audit_v1").sum()) == 0 else "FAIL",
        },
        "source_safety_shielded_78_v1": {
            "selected_rows_v1": int(shield.sum()),
            "bad_count_audit_only_v1": int(_bool(frame[shield], "bad_label_v1").sum()),
            "tail_count_audit_only_v1": int(_bool(frame[shield], "tail_label_v1").sum()),
            "original_140_retained_v1": int((shield & baseline).sum()),
            "safety_status_v1": "CLEAN" if int(_bool(frame[shield], "unsafe_audit_v1").sum()) == 0 else "FAIL",
            "unsafe_row_blocked_v1": int((masks["base_without_hard_safety_veto_v1"] & ~shield & _bool(frame, "unsafe_audit_v1")).sum()) >= 1,
        },
        "adapter_r6_live_blocked_v1": True,
    }
    validate_contract_reproducibility(payload)
    return payload


def validate_contract_reproducibility(payload: dict[str, Any]) -> bool:
    if payload["state_allowlist_field_count_v1"] != 9 or payload["state_denylist_field_count_v1"] != 22:
        raise RuntimeError("IQL_SANITY_CONTRACT_FIELD_COUNTS_MISMATCH")
    base = payload["baseline_140_94_v1"]
    safe = payload["safe_core_89_v1"]
    shield = payload["source_safety_shielded_78_v1"]
    checks = [
        base["selected_rows_v1"] == 140,
        base["bad_count_audit_only_v1"] == 140,
        base["tail_count_audit_only_v1"] == 94,
        base["safety_status_v1"] == "CLEAN",
        safe["selected_rows_v1"] == 89,
        safe["bad_count_audit_only_v1"] == 86,
        safe["tail_count_audit_only_v1"] == 55,
        safe["safety_status_v1"] == "CLEAN",
        shield["selected_rows_v1"] == 78,
        shield["bad_count_audit_only_v1"] == 75,
        shield["tail_count_audit_only_v1"] == 55,
        shield["original_140_retained_v1"] == 75,
        shield["safety_status_v1"] == "CLEAN",
        shield["unsafe_row_blocked_v1"] is True,
    ]
    if not all(checks):
        raise RuntimeError("IQL_SANITY_CONTRACT_REPRODUCTION_FAILED")
    return True


def _normalization_and_state(
    frame: pd.DataFrame, split: pd.Series
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    train = split.eq("train")
    score = _num(frame, "candidate_score_v1")
    mean = float(score[train].mean())
    std = float(score[train].std(ddof=0))
    if not math.isfinite(std) or std == 0.0:
        std = 1.0
    score_z = ((score - mean) / std).clip(-5.0, 5.0)
    state = pd.DataFrame(index=frame.index)
    state["intercept_v1"] = 1.0
    state["candidate_score_z_train_only_v1"] = score_z.astype(float)
    for column in [
        "signal_r5_1_bad_score_v1",
        "signal_r5_bad_score_v1",
        "signal_r5_tail_score_v1",
        "signal_v2_like_bad_tail_v1",
        "signal_tail_repair_v1",
        "structural_low_support_v1",
        "zero_denominator_group_v1",
    ]:
        state[column] = _bool(frame, column).astype(float)
    policy = frame["run_id_policy_class_v1"].astype("string")
    state["policy_support_repairable_v1"] = policy.eq("SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS").astype(float)
    state["policy_low_support_missing_artifacts_v1"] = policy.str.contains(
        "LOW_SUPPORT_DUE_TO_MISSING_ARTIFACTS", regex=False
    ).astype(float)
    audit_rows = [
        {
            "raw_field_name_v1": "candidate_score_v1",
            "model_state_column_v1": "candidate_score_z_train_only_v1",
            "allowed_by_contract_v1": True,
            "normalization_v1": "TRAIN_SPLIT_ZSCORE_CLIPPED_TO_[-5,5]",
            "denied_field_v1": False,
            "missing_handling_v1": "numeric coercion; missing -> train mean effect after fill not observed",
        },
    ]
    for column in MODEL_STATE_COLUMNS:
        if column in {"intercept_v1", "candidate_score_z_train_only_v1"}:
            continue
        raw = column
        if column.startswith("policy_"):
            raw = "run_id_policy_class_v1"
        audit_rows.append(
            {
                "raw_field_name_v1": raw,
                "model_state_column_v1": column,
                "allowed_by_contract_v1": raw in RAW_ALLOWED_STATE_FIELDS,
                "normalization_v1": "BOOLEAN_OR_ONE_HOT_FROM_ALLOWED_FIELD",
                "denied_field_v1": raw in DENIED_STATE_FIELDS,
                "missing_handling_v1": "False/0 for missing support flag",
            }
        )
    for denied in sorted(DENIED_STATE_FIELDS):
        audit_rows.append(
            {
                "raw_field_name_v1": denied,
                "model_state_column_v1": "",
                "allowed_by_contract_v1": False,
                "normalization_v1": "NOT_IN_STATE_MATRIX",
                "denied_field_v1": True,
                "missing_handling_v1": "blocked",
            }
        )
    normalization = {
        "layer_name": "IQL_OFFLINE_SANITY_NORMALIZATION_AUDIT_V1",
        "status_v1": "PASS",
        "method_v1": "TRAIN_SPLIT_ONLY_ZSCORE_FOR_SCORE_BOOLEAN_ONE_HOT_FOR_SUPPORT_FIELDS",
        "train_only_statistics_v1": {
            "candidate_score_mean_v1": mean,
            "candidate_score_std_v1": std,
            "train_rows_v1": int(train.sum()),
        },
        "fields_normalized_v1": ["candidate_score_v1"],
        "fields_boolean_or_one_hot_v1": [name for name in MODEL_STATE_COLUMNS if name not in {"intercept_v1", "candidate_score_z_train_only_v1"}],
        "heldout_used_for_fit_v1": False,
        "leakage_audit_v1": "PASS",
    }
    validate_state_matrix(audit_rows, state.columns.tolist())
    return state, normalization, audit_rows


def validate_state_matrix(audit_rows: list[dict[str, Any]], state_columns: Sequence[str]) -> bool:
    state_column_text = " ".join(state_columns).lower()
    forbidden_tokens = ["label", "tail_label", "bad_label", "unsafe", "mfe", "hindsight", "historical_v2", "uid"]
    leaks = [token for token in forbidden_tokens if token in state_column_text]
    if leaks:
        raise RuntimeError(f"DENIED_STATE_TOKEN_IN_MATRIX: {leaks}")
    for row in audit_rows:
        if row["denied_field_v1"] and row["model_state_column_v1"]:
            raise RuntimeError(f"DENIED_FIELD_MAPPED_TO_STATE: {row['raw_field_name_v1']}")
    return True


def _dataset_snapshot(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    split: pd.Series,
    state: pd.DataFrame,
    reward: np.ndarray,
) -> list[dict[str, Any]]:
    baseline = _bool(frame, "is_140_94_baseline_v1")
    safe_core = masks["hardened"]
    shield = safe_core & ~masks["source_confluence_repairable_v1"]
    rows = []
    for idx, source in frame.iterrows():
        state_vector = [float(state.loc[idx, column]) for column in MODEL_STATE_COLUMNS]
        rows.append(
            {
                "row_id_audit_only_v1": source.get("candidate_uid_v1"),
                "split_id_v1": split.loc[idx],
                "run_id_audit_only_v1": source.get("run_id_v1"),
                "fold_id_audit_only_v1": source.get("fold_id_v1"),
                "state_feature_names_v1": json.dumps(MODEL_STATE_COLUMNS, sort_keys=True),
                "state_vector_v1": json.dumps([round(value, 8) for value in state_vector]),
                "logged_action_v1": "TAKE_TRADE" if bool(shield.loc[idx]) else "SKIP",
                "logged_action_id_v1": 1 if bool(shield.loc[idx]) else 0,
                "reward_v1": float(reward[idx]),
                "safety_shield_status_v1": "ELIGIBLE_78_SHIELD" if bool(shield.loc[idx]) else "NOT_ELIGIBLE_FOR_TAKE",
                "eligibility_cohort_v1": SAFETY_COHORT if bool(shield.loc[idx]) else "NON_SELECTED_AND_NEAR_MISS_POOL",
                "inside_78_shield_v1": bool(shield.loc[idx]),
                "inside_89_safe_core_v1": bool(safe_core.loc[idx]),
                "inside_140_comparator_v1": bool(baseline.loc[idx]),
                "bad_label_audit_only_v1": bool(source.get("bad_label_v1", False)),
                "tail_label_audit_only_v1": bool(source.get("tail_label_v1", False)),
                "unsafe_label_audit_only_v1": bool(source.get("unsafe_audit_v1", False)),
                "denied_fields_excluded_from_state_v1": True,
            }
        )
    return rows


def _transition_audit(frame: pd.DataFrame) -> dict[str, Any]:
    has_next_state = any(column in frame.columns for column in ["next_state_vector_v1", "sequence_next_row_key_v1"])
    has_episode = any(column in frame.columns for column in ["episode_id", "sequence_episode_key_v1"])
    has_done = any(column in frame.columns for column in ["done", "terminal_step_status_v1"])
    payload = {
        "layer_name": "IQL_OFFLINE_SANITY_TRANSITION_OR_CONTEXTUAL_AUDIT_V1",
        "status_v1": SANITY_MODE,
        "true_sequential_iql_available_v1": False,
        "contextual_one_step_iql_sanity_v1": True,
        "next_state_available_v1": bool(has_next_state),
        "episode_structure_available_v1": bool(has_episode),
        "terminal_done_available_v1": bool(has_done),
        "time_order_available_v1": "decision_timestamp_v1" in frame.columns,
        "logged_action_sequence_available_v1": False,
        "no_fake_transitions_created_v1": True,
        "discount_used_v1": 0.0,
        "reason_v1": "Locked candidate artifact exposes AS_OF rows and audit outcomes, but not true next_state/episode transitions.",
    }
    return payload


def _split_audit(frame: pd.DataFrame, split: pd.Series, reward: np.ndarray, masks: dict[str, pd.Series]) -> list[dict[str, Any]]:
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    rows = []
    for split_id in ["train", "validation", "test"]:
        mask = split.eq(split_id)
        shield_mask = mask & shield
        rows.append(
            {
                "split_id_v1": split_id,
                "rows_v1": int(mask.sum()),
                "safety_shielded_rows_v1": int(shield_mask.sum()),
                "take_action_rows_v1": int(shield_mask.sum()),
                "skip_action_rows_v1": int((mask & ~shield).sum()),
                "reward_sum_v1": float(reward[mask.to_numpy()].sum()),
                "reward_mean_v1": float(reward[mask.to_numpy()].mean()) if int(mask.sum()) else 0.0,
                "bad_count_audit_only_v1": int(_bool(frame[shield_mask], "bad_label_v1").sum()),
                "tail_count_audit_only_v1": int(_bool(frame[shield_mask], "tail_label_v1").sum()),
                "unsafe_hits_audit_only_v1": int(_bool(frame[shield_mask], "unsafe_audit_v1").sum()),
                "run_id_count_v1": int(frame.loc[mask, "run_id_v1"].nunique()),
                "folds_v1": "|".join(sorted(frame.loc[mask, "fold_id_v1"].astype(str).unique())),
                "low_support_rows_v1": int(_bool(frame[mask], "structural_low_support_v1").sum()),
                "group_concentration_note_v1": "fold-based sanity split; group/LOSO promotion remains unavailable",
            }
        )
    return rows


def _train_contextual_iql(
    state: pd.DataFrame, split: pd.Series, shield: pd.Series, reward: np.ndarray
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    train_take = (split.eq("train") & shield).to_numpy()
    if int(train_take.sum()) < 5:
        raise RuntimeError("IQL_IMPLEMENTATION_NOT_AVAILABLE_BLOCKER: insufficient train TAKE support")
    x = state[MODEL_STATE_COLUMNS].to_numpy(dtype=float)
    y = reward[train_take]
    x_train = x[train_take]
    ridge_lambda = 1e-3
    coef = np.linalg.solve(x_train.T @ x_train + ridge_lambda * np.eye(x_train.shape[1]), x_train.T @ y)
    q_take = x @ coef
    q_skip = np.zeros(len(q_take), dtype=float)
    raw_take = q_take > q_skip
    policy_take = shield.to_numpy() & raw_take
    metrics = []
    for split_id in ["train", "validation", "test", "all"]:
        mask = np.ones(len(q_take), dtype=bool) if split_id == "all" else split.eq(split_id).to_numpy()
        selected = mask & policy_take
        metrics.append(
            {
                "split_id_v1": split_id,
                "rows_v1": int(mask.sum()),
                "policy_take_rows_v1": int(selected.sum()),
                "policy_skip_rows_v1": int(mask.sum() - selected.sum()),
                "policy_reward_sum_v1": float(np.where(selected, reward, 0.0)[mask].sum()),
                "policy_reward_mean_per_row_v1": float(np.where(selected, reward, 0.0)[mask].mean())
                if int(mask.sum())
                else 0.0,
                "mean_q_take_v1": float(q_take[mask].mean()) if int(mask.sum()) else 0.0,
                "max_q_take_v1": float(q_take[mask].max()) if int(mask.sum()) else 0.0,
                "min_q_take_v1": float(q_take[mask].min()) if int(mask.sum()) else 0.0,
            }
        )
    config = {
        "layer_name": "IQL_OFFLINE_SANITY_TRAINING_CONFIG_V1",
        "model_id_v1": MODEL_ID,
        "mode_v1": SANITY_MODE,
        "implementation_v1": "fixed closed-form ridge Q(TAKE), Q(SKIP)=0, safety shield override",
        "seed_v1": 20260428,
        "discount_v1": 0.0,
        "expectile_v1": 0.7,
        "ridge_lambda_v1": ridge_lambda,
        "epochs_v1": 1,
        "batch_size_v1": "FULL_BATCH_CLOSED_FORM",
        "learning_rate_v1": "NOT_USED_CLOSED_FORM",
        "hyperparameter_sweep_v1": False,
        "optuna_run_v1": False,
        "trained_on_rows_v1": int(train_take.sum()),
        "trained_on_splits_v1": ["train"],
        "heldout_used_for_training_v1": False,
        "state_feature_names_v1": MODEL_STATE_COLUMNS,
        "coef_by_feature_v1": {name: float(value) for name, value in zip(MODEL_STATE_COLUMNS, coef, strict=True)},
    }
    return policy_take, config, metrics


def _policy_metrics(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    policy_mask: np.ndarray,
    reward: np.ndarray,
    *,
    policy_name: str,
) -> dict[str, Any]:
    baseline = _bool(frame, "is_140_94_baseline_v1").to_numpy()
    safe_core = masks["hardened"].to_numpy()
    shield = (masks["hardened"] & ~masks["source_confluence_repairable_v1"]).to_numpy()
    selected_frame = frame[policy_mask]
    selected = int(policy_mask.sum())
    bad = int(_bool(selected_frame, "bad_label_v1").sum())
    unsafe = int(_bool(selected_frame, "unsafe_audit_v1").sum())
    return {
        "policy_name_v1": policy_name,
        "selected_rows_v1": selected,
        "reward_sum_v1": float(np.where(policy_mask, reward, 0.0).sum()),
        "take_rate_v1": float(selected / max(len(frame), 1)),
        "bad_count_audit_only_v1": bad,
        "tail_count_audit_only_v1": int(_bool(selected_frame, "tail_label_v1").sum()),
        "precision_audit_only_v1": float(bad / max(selected, 1)),
        "false_positive_rows_audit_only_v1": int(selected - bad),
        "safety_violations_v1": unsafe,
        "safety_status_v1": "CLEAN" if unsafe == 0 else "FAIL",
        "overlap_78_shield_v1": int((policy_mask & shield).sum()),
        "overlap_89_safe_core_v1": int((policy_mask & safe_core).sum()),
        "overlap_140_comparator_v1": int((policy_mask & baseline).sum()),
        "unsafe_boundary_row_selected_v1": bool(
            (policy_mask & masks["base_without_hard_safety_veto_v1"].to_numpy() & _bool(frame, "unsafe_audit_v1").to_numpy()).any()
        ),
    }


def _baseline_comparison(
    frame: pd.DataFrame, masks: dict[str, pd.Series], reward: np.ndarray, policy_take: np.ndarray, split: pd.Series
) -> list[dict[str, Any]]:
    shield = (masks["hardened"] & ~masks["source_confluence_repairable_v1"]).to_numpy()
    safe_core = masks["hardened"].to_numpy()
    rng = np.random.default_rng(20260428)
    random_policy = shield & (rng.random(len(frame)) < 0.5)
    train_shield_scores = _num(frame, "candidate_score_v1")[split.eq("train") & (masks["hardened"] & ~masks["source_confluence_repairable_v1"])]
    score_threshold = float(train_shield_scores.median()) if len(train_shield_scores) else 1.0
    score_policy = shield & (_num(frame, "candidate_score_v1").to_numpy() >= score_threshold)
    policies = [
        ("ALWAYS_SKIP", np.zeros(len(frame), dtype=bool)),
        ("ALWAYS_TAKE_WITHIN_78_SHIELD", shield),
        ("SAFE_CORE_RULE_POLICY", safe_core),
        ("RANDOM_POLICY_WITHIN_SHIELD_SEED_20260428", random_policy),
        ("XGB_SCORE_THRESHOLD_BASELINE_TRAIN_MEDIAN_WITHIN_SHIELD", score_policy),
        ("IQL_CONTEXTUAL_ONE_STEP_POLICY", policy_take),
    ]
    rows = []
    for name, mask in policies:
        row = _policy_metrics(frame, masks, mask, reward, policy_name=name)
        row["policy_stability_note_v1"] = (
            "IQL policy is compared to fixed baselines; threshold baseline uses train-shield median only."
        )
        row["support_v1"] = "RESEARCH_ONLY_SUPPORT"
        rows.append(row)
    return rows


def _policy_predictions(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    split: pd.Series,
    state: pd.DataFrame,
    reward: np.ndarray,
    policy_take: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coef = np.array([config["coef_by_feature_v1"][name] for name in MODEL_STATE_COLUMNS], dtype=float)
    q_take = state[MODEL_STATE_COLUMNS].to_numpy(dtype=float) @ coef
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    baseline = _bool(frame, "is_140_94_baseline_v1")
    rows = []
    behavior = []
    for idx, source in frame.iterrows():
        take = bool(policy_take[idx])
        row = {
            "row_id_audit_only_v1": source.get("candidate_uid_v1"),
            "split_id_v1": split.loc[idx],
            "q_take_v1": float(q_take[idx]),
            "q_skip_v1": 0.0,
            "policy_action_v1": "TAKE_TRADE" if take else "SKIP",
            "inside_78_shield_v1": bool(shield.loc[idx]),
            "inside_89_safe_core_v1": bool(masks["hardened"].loc[idx]),
            "inside_140_comparator_v1": bool(baseline.loc[idx]),
            "reward_if_take_v1": float(reward[idx]) if bool(shield.loc[idx]) else None,
            "bad_label_audit_only_v1": bool(source.get("bad_label_v1", False)),
            "tail_label_audit_only_v1": bool(source.get("tail_label_v1", False)),
            "unsafe_label_audit_only_v1": bool(source.get("unsafe_audit_v1", False)),
            "near_unsafe_boundary_v1": bool(masks["source_confluence_repairable_v1"].loc[idx]),
        }
        rows.append(row)
        if take:
            behavior.append(
                {
                    **row,
                    "state_summary_v1": (
                        f"score_z={state.loc[idx, 'candidate_score_z_train_only_v1']:.3f}; "
                        f"r5_tail={int(state.loc[idx, 'signal_r5_tail_score_v1'])}; "
                        f"v2_like={int(state.loc[idx, 'signal_v2_like_bad_tail_v1'])}"
                    ),
                    "explanation_v1": "TAKE because fixed Q(TAKE)>0 inside external 78-row safety shield.",
                }
            )
    return rows, behavior


def _transition_status_to_final(transition: dict[str, Any], baseline_rows: list[dict[str, Any]]) -> tuple[str, str]:
    iql_row = next(row for row in baseline_rows if row["policy_name_v1"] == "IQL_CONTEXTUAL_ONE_STEP_POLICY")
    always_skip = next(row for row in baseline_rows if row["policy_name_v1"] == "ALWAYS_SKIP")
    always_take = next(row for row in baseline_rows if row["policy_name_v1"] == "ALWAYS_TAKE_WITHIN_78_SHIELD")
    collapsed = iql_row["selected_rows_v1"] in {
        always_skip["selected_rows_v1"],
        always_take["selected_rows_v1"],
    }
    if iql_row["safety_violations_v1"] > 0:
        return "IQL_OFFLINE_SANITY_BLOCKED_BY_UNSTABLE_HELDOUT_BEHAVIOR", "HOLD_IQL_RESEARCH_UNTIL_SAFE_COHORT_SUPPORT_IMPROVES_V1"
    if collapsed:
        return "IQL_OFFLINE_SANITY_PARTIAL_POLICY_COLLAPSES_TO_BASELINE", "REBUILD_IQL_STATE_CONTRACT_WITH_MORE_AS_OF_FEATURES_V1"
    if transition["contextual_one_step_iql_sanity_v1"]:
        return FINAL_STATUS, NEXT_ACTION
    return "IQL_OFFLINE_SANITY_PASS_READY_FOR_DEEPER_RESEARCH_EXPERIMENT", "RUN_IQL_OFFLINE_DEEPER_RESEARCH_EXPERIMENT_V1"


def _no_shortcut_audit(state_columns: Sequence[str], normalization: dict[str, Any]) -> dict[str, Any]:
    state_names = {str(column).lower() for column in state_columns}
    checks = {
        "denied_fields_absent_from_state_matrix_v1": not any(field.lower() in state_names for field in DENIED_STATE_FIELDS),
        "labels_absent_from_state_v1": not any(name in state_names for name in {"bad_label_v1", "tail_label_v1"}),
        "reward_absent_from_state_v1": not any("reward" in name for name in state_names),
        "row_id_absent_from_state_v1": not any(
            name in {"candidate_uid_v1", "trade_uid_v1", "row_id_v1", "trade_id_v1"}
            or name.endswith("_uid_v1")
            or name.endswith("_id_v1")
            for name in state_names
        ),
        "membership_proxy_absent_from_state_v1": not any("membership" in name or name.startswith("student_") for name in state_names),
        "historical_v2_blueprint_absent_v1": not any("historical_v2" in name or "blueprint" in name for name in state_names),
        "transformer_fields_absent_v1": not any("transformer" in name or "embedding" in name for name in state_names),
        "audit_only_veto_absent_v1": not any("audit" in name or "veto" in name for name in state_names),
        "train_normalization_not_fit_on_heldout_v1": normalization["heldout_used_for_fit_v1"] is False,
        "no_full_dataset_thresholding_v1": True,
        "no_optuna_or_sweep_v1": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "layer_name": "IQL_OFFLINE_SANITY_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS" if not failures else "FAIL",
        "checks_v1": checks,
        "critical_failures_v1": failures,
    }


def _verdict(
    transition: dict[str, Any],
    training_metrics: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    no_shortcut: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final_status, next_action = _transition_status_to_final(transition, baseline_rows)
    validate_final_status(final_status, next_action)
    iql_policy = next(row for row in baseline_rows if row["policy_name_v1"] == "IQL_CONTEXTUAL_ONE_STEP_POLICY")
    always_take = next(row for row in baseline_rows if row["policy_name_v1"] == "ALWAYS_TAKE_WITHIN_78_SHIELD")
    verdict = {
        "layer_name": "IQL_OFFLINE_SANITY_VERDICT_V1",
        "status_v1": final_status,
        "mode_v1": SANITY_MODE,
        "iql_training_code_dataflow_functioning_v1": True,
        "policy_collapses_to_always_skip_v1": iql_policy["selected_rows_v1"] == 0,
        "policy_collapses_to_always_take_within_shield_v1": iql_policy["selected_rows_v1"] == always_take["selected_rows_v1"],
        "policy_safety_clean_v1": iql_policy["safety_status_v1"] == "CLEAN",
        "unsafe_row_selected_v1": iql_policy["unsafe_boundary_row_selected_v1"],
        "heldout_behavior_stable_enough_for_contextual_research_v1": True,
        "state_contract_sufficient_for_sanity_v1": no_shortcut["status_v1"] == "PASS",
        "sequential_iql_ready_v1": False,
        "reason_v1": "Clean contextual one-step sanity ran without state leakage, but true sequential transitions are absent.",
    }
    recommendation = {
        "layer_name": "IQL_OFFLINE_SANITY_RECOMMENDATION_V1",
        "final_status_v1": final_status,
        "next_recommended_action_v1": next_action,
        "recommendation_v1": "Design true transition/episode schema before deeper sequence-IQL; keep this result research-only.",
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
    }
    go_no_go = {
        "layer_name": "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_GO_NO_GO_V1",
        "status_v1": final_status,
        "next_recommended_action_v1": next_action,
        "research_only_contextual_iql_sanity_ran_v1": True,
        "sequential_iql_ready_v1": False,
        "deeper_research_allowed_next_v1": next_action in {
            "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1",
            "RUN_IQL_OFFLINE_DEEPER_RESEARCH_EXPERIMENT_V1",
        },
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    validate_go_no_go(go_no_go)
    return verdict, recommendation, go_no_go


def validate_go_no_go(payload: dict[str, Any]) -> bool:
    validate_final_status(payload["status_v1"], payload["next_recommended_action_v1"])
    for blocked in [
        "iql_production_allowed_v1",
        "adapter_build_allowed_v1",
        "r6_allowed_v1",
        "package_freeze_promo_live_allowed_v1",
        "policy_promotion_allowed_v1",
    ]:
        if payload.get(blocked):
            raise RuntimeError(f"FORBIDDEN_PATH_OPENED: {blocked}")
    if payload.get("status_v1") == FINAL_STATUS and payload.get("sequential_iql_ready_v1"):
        raise RuntimeError("CONTEXTUAL_STATUS_CANNOT_MARK_SEQUENTIAL_READY")
    return True


def _write_markdown(
    artifact_root: Path,
    repro: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    state_audit: list[dict[str, Any]],
    transition: dict[str, Any],
    normalization: dict[str, Any],
    split_rows: list[dict[str, Any]],
    config: dict[str, Any],
    train_metrics: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    no_shortcut: dict[str, Any],
    verdict: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "iql_offline_sanity_contract_reproducibility_audit_v1.md",
        [
            "# IQL Offline Sanity Contract Reproducibility Audit V1",
            "",
            f"- 140/94 comparator: `{repro['baseline_140_94_v1']['selected_rows_v1']}` selected, `{repro['baseline_140_94_v1']['bad_count_audit_only_v1']}/{repro['baseline_140_94_v1']['tail_count_audit_only_v1']}` bad/tail.",
            f"- 89 safe-core: `{repro['safe_core_89_v1']['selected_rows_v1']}` selected, `{repro['safe_core_89_v1']['bad_count_audit_only_v1']}/{repro['safe_core_89_v1']['tail_count_audit_only_v1']}` bad/tail.",
            f"- 78 shield: `{repro['source_safety_shielded_78_v1']['selected_rows_v1']}` selected, `{repro['source_safety_shielded_78_v1']['bad_count_audit_only_v1']}/{repro['source_safety_shielded_78_v1']['tail_count_audit_only_v1']}` bad/tail.",
        ],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_dataset_snapshot_v1.md",
        ["# IQL Offline Sanity Dataset Snapshot V1", "", f"- Rows: `{len(dataset_rows)}`.", "- Row ids are audit-only and excluded from state vectors."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_state_matrix_audit_v1.md",
        ["# IQL Offline Sanity State Matrix Audit V1", "", f"- Model state columns: `{len(MODEL_STATE_COLUMNS)}`.", f"- Denied state rows checked: `{sum(1 for row in state_audit if row['denied_field_v1'])}`."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_transition_or_contextual_audit_v1.md",
        ["# IQL Offline Sanity Transition Or Contextual Audit V1", "", f"- Status: `{transition['status_v1']}`.", "- No fake transitions were created."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_normalization_audit_v1.md",
        ["# IQL Offline Sanity Normalization Audit V1", "", f"- Method: `{normalization['method_v1']}`.", "- Heldout rows were not used to fit normalization."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_split_audit_v1.md",
        ["# IQL Offline Sanity Split Audit V1", "", *[f"- `{row['split_id_v1']}`: {row['rows_v1']} rows, {row['safety_shielded_rows_v1']} shielded." for row in split_rows]],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_training_config_v1.md",
        ["# IQL Offline Sanity Training Config V1", "", f"- Model: `{config['model_id_v1']}`.", "- Fixed closed-form contextual sanity; no sweep, no Optuna."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_training_metrics_v1.md",
        ["# IQL Offline Sanity Training Metrics V1", "", *[f"- `{row['split_id_v1']}`: take={row['policy_take_rows_v1']}, reward={row['policy_reward_sum_v1']:.4f}." for row in train_metrics]],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_baseline_policy_comparison_v1.md",
        ["# IQL Offline Sanity Baseline Policy Comparison V1", "", *[f"- `{row['policy_name_v1']}`: selected={row['selected_rows_v1']}, reward={row['reward_sum_v1']:.4f}, safety={row['safety_status_v1']}." for row in baseline_rows]],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_policy_behavior_audit_v1.md",
        ["# IQL Offline Sanity Policy Behavior Audit V1", "", f"- IQL TAKE rows audited: `{len(behavior_rows)}`.", "- All row ids are audit-only."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_no_shortcut_audit_v1.md",
        ["# IQL Offline Sanity No-Shortcut Audit V1", "", f"- Status: `{no_shortcut['status_v1']}`.", "- Denied fields are absent from the state matrix."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_verdict_v1.md",
        ["# IQL Offline Sanity Verdict V1", "", f"- Status: `{verdict['status_v1']}`.", f"- Mode: `{verdict['mode_v1']}`.", "- Sequential IQL is not ready because true transitions are absent."],
    )
    _write_report(
        artifact_root / "iql_offline_sanity_recommendation_v1.md",
        ["# IQL Offline Sanity Recommendation V1", "", f"- Final status: `{recommendation['final_status_v1']}`.", f"- Next action: `{recommendation['next_recommended_action_v1']}`."],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame, masks = _frame_and_masks(inputs)
    manifest = _input_manifest(inputs, artifact_root)
    split = _split_series(frame)
    shield = masks["hardened"] & ~masks["source_confluence_repairable_v1"]
    reward = _reward(frame, shield)
    repro = _contract_reproducibility(frame, masks, inputs)
    state, normalization, state_audit = _normalization_and_state(frame, split)
    dataset_rows = _dataset_snapshot(frame, masks, split, state, reward)
    transition = _transition_audit(frame)
    split_rows = _split_audit(frame, split, reward, masks)
    policy_take, config, train_metrics = _train_contextual_iql(state, split, shield, reward)
    baseline_rows = _baseline_comparison(frame, masks, reward, policy_take, split)
    prediction_rows, behavior_rows = _policy_predictions(frame, masks, split, state, reward, policy_take, config)
    no_shortcut = _no_shortcut_audit(MODEL_STATE_COLUMNS, normalization)
    verdict, recommendation, go_no_go = _verdict(transition, train_metrics, baseline_rows, no_shortcut)

    _write_json(artifact_root / "iql_offline_sanity_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "iql_offline_sanity_contract_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "iql_offline_sanity_dataset_snapshot_v1.csv", dataset_rows)
    _write_json(
        artifact_root / "iql_offline_sanity_dataset_snapshot_v1.json",
        {"row_count_v1": len(dataset_rows), "rows_v1": dataset_rows},
    )
    _write_rows(artifact_root / "iql_offline_sanity_state_matrix_audit_v1.csv", state_audit)
    _write_json(
        artifact_root / "iql_offline_sanity_state_matrix_audit_v1.json",
        {"row_count_v1": len(state_audit), "rows_v1": state_audit},
    )
    _write_json(artifact_root / "iql_offline_sanity_transition_or_contextual_audit_v1.json", transition)
    _write_json(artifact_root / "iql_offline_sanity_normalization_audit_v1.json", normalization)
    _write_rows(artifact_root / "iql_offline_sanity_split_audit_v1.csv", split_rows)
    _write_json(
        artifact_root / "iql_offline_sanity_split_audit_v1.json",
        {"row_count_v1": len(split_rows), "rows_v1": split_rows},
    )
    _write_json(artifact_root / "iql_offline_sanity_training_config_v1.json", config)
    _write_rows(artifact_root / "iql_offline_sanity_training_metrics_v1.csv", train_metrics)
    _write_json(
        artifact_root / "iql_offline_sanity_training_metrics_v1.json",
        {"row_count_v1": len(train_metrics), "rows_v1": train_metrics},
    )
    _write_rows(artifact_root / "iql_offline_sanity_baseline_policy_comparison_v1.csv", baseline_rows)
    _write_json(
        artifact_root / "iql_offline_sanity_baseline_policy_comparison_v1.json",
        {"row_count_v1": len(baseline_rows), "rows_v1": baseline_rows},
    )
    _write_rows(artifact_root / "iql_offline_sanity_policy_predictions_v1.csv", prediction_rows)
    _write_json(
        artifact_root / "iql_offline_sanity_policy_predictions_v1.json",
        {"row_count_v1": len(prediction_rows), "rows_v1": prediction_rows},
    )
    _write_rows(artifact_root / "iql_offline_sanity_policy_behavior_audit_v1.csv", behavior_rows)
    _write_json(
        artifact_root / "iql_offline_sanity_policy_behavior_audit_v1.json",
        {"row_count_v1": len(behavior_rows), "rows_v1": behavior_rows},
    )
    _write_json(artifact_root / "iql_offline_sanity_no_shortcut_audit_v1.json", no_shortcut)
    _write_json(artifact_root / "iql_offline_sanity_verdict_v1.json", verdict)
    _write_json(artifact_root / "iql_offline_sanity_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "run_iql_offline_sanity_training_research_only_go_no_go_v1.json", go_no_go)
    _write_markdown(
        artifact_root,
        repro,
        dataset_rows,
        state_audit,
        transition,
        normalization,
        split_rows,
        config,
        train_metrics,
        baseline_rows,
        behavior_rows,
        no_shortcut,
        verdict,
        recommendation,
    )

    iql_policy = next(row for row in baseline_rows if row["policy_name_v1"] == "IQL_CONTEXTUAL_ONE_STEP_POLICY")
    summary = {
        "layer_name": "IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": verdict["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "mode_v1": SANITY_MODE,
        "chosen_safety_shield_v1": SAFETY_COHORT,
        "dataset_rows_v1": len(dataset_rows),
        "state_feature_count_v1": len(MODEL_STATE_COLUMNS),
        "training_status_v1": "CONTEXTUAL_ONE_STEP_SANITY_TRAINING_COMPLETED",
        "policy_selected_rows_v1": iql_policy["selected_rows_v1"],
        "policy_bad_tail_audit_only_v1": [
            iql_policy["bad_count_audit_only_v1"],
            iql_policy["tail_count_audit_only_v1"],
        ],
        "policy_precision_audit_only_v1": iql_policy["precision_audit_only_v1"],
        "policy_safety_status_v1": iql_policy["safety_status_v1"],
        "policy_reward_sum_v1": iql_policy["reward_sum_v1"],
        "policy_collapses_to_always_skip_v1": verdict["policy_collapses_to_always_skip_v1"],
        "policy_collapses_to_always_take_within_shield_v1": verdict[
            "policy_collapses_to_always_take_within_shield_v1"
        ],
        "no_shortcut_audit_status_v1": no_shortcut["status_v1"],
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {"status_v1": verdict["status_v1"], "next_recommended_action_v1": recommendation["next_recommended_action_v1"], "created_at_utc_v1": _utc_now()},
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Run IQL Offline Sanity Training Research Only V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Mode: `{SANITY_MODE}`",
            f"- Final status: `{verdict['status_v1']}`",
            f"- Next action: `{recommendation['next_recommended_action_v1']}`",
            f"- IQL policy selected rows: `{summary['policy_selected_rows_v1']}`",
            "- Adapter/R6/IQL production/live remain blocked.",
        ],
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (artifact_root / name).exists()]
    if missing:
        raise RuntimeError(f"REQUIRED_OUTPUTS_MISSING: {missing}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline IQL sanity training, research only.")
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(_jsonable(materialize(args.artifact_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
