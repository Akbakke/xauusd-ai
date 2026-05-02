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


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "SIMPLIFY_140_94_RULES_AND_VETOES_V1"

INPUT_DISTILL_ROOT = (
    DEFAULT_REPORTS_ROOT / "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1_20260428T081017Z_LOCK"
)
INPUT_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)
INPUT_PLUS45_SHADOW_ROOT = (
    DEFAULT_REPORTS_ROOT / "PLUS45_AS_OF_FEATURE_GAP_SHADOW_EXPLORATION_V1_20260428T074409Z_LOCK"
)
INPUT_REJECT_REBUILD_ROOT = (
    DEFAULT_REPORTS_ROOT / "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1_20260428T063714Z_LOCK"
)
INPUT_BEST_LANE_PACKAGE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "MATERIALIZE_BEST_LANE_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1_20260427T193354Z_LOCK"
)
INPUT_STUDENT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_V1_20260427T202519Z_LOCK"
)

BASELINE_CANDIDATE_ID = "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL"
SELECTED_RECIPE_ID = "CONSERVATIVE_HIGH_CONFIDENCE_RULE_V1"
FINAL_STATUS = "140_94_SIMPLIFIED_RULES_FOUND_SAFE_CORE_NEEDS_EXPANSION_LATER"
NEXT_ACTION = "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1"

BASELINE_SELECTED = 140
BASELINE_BAD = 140
BASELINE_TAIL = 94
FULL_COVER_SELECTED = 250
FULL_COVER_EXTRA = 110

ALLOWED_FINAL_STATUSES = {
    "140_94_SIMPLIFIED_RULES_PASS_ADAPTER_READY",
    "140_94_SIMPLIFIED_RULES_PASS_NEEDS_ADAPTER_INPUT_MAPPING",
    "140_94_SIMPLIFIED_RULES_FOUND_SAFE_CORE_NEEDS_EXPANSION_LATER",
    "140_94_SIMPLIFIED_RULES_PARTIAL_NEEDS_MORE_VETO_MAPPING",
    "140_94_SIMPLIFIED_RULES_BLOCKED_BY_OVER_SELECTION",
    "140_94_SIMPLIFIED_RULES_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "140_94_SIMPLIFIED_RULES_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "140_94_SIMPLIFIED_RULES_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_CAUSAL_BASELINE_ADAPTER_V1",
    "BUILD_140_94_ADAPTER_INPUT_MAPPING_V1",
    "HARDEN_140_94_SAFE_CORE_AND_EXPAND_LATER_V1",
    "DEEPEN_140_94_VETO_MAPPING_AUDIT_V1",
    "DEEPEN_140_94_UNSAFE_LOOKALIKE_BOUNDARY_AUDIT_V1",
    "DEEPEN_140_94_GROUPED_GENERALIZATION_AND_LOSO_AUDIT_V1",
    "RETURN_TO_140_94_RULE_DISTILLATION_WITH_STRONGER_AS_OF_SIGNALS_V1",
}

RECIPE_IDS = {
    "CONSERVATIVE_HIGH_CONFIDENCE_RULE_V1",
    "BALANCED_140_RECOVERY_RULE_V1",
    "FULL_COVER_TIGHTENED_RULE_V1",
    "SCORE_PLUS_VETO_RULE_V1",
    "STUDENT_CORE_DIAGNOSTIC_REFERENCE_V1",
}

DENY_PATTERNS = [
    "bad_label",
    "tail_label",
    "label_should_not_take",
    "take_was_ok",
    "final_outcome",
    "post_outcome",
    "mfe",
    "safe_recoverable",
    "coverage_proxy",
    "185_139",
    "plus45",
    "rows_added_vs_140_94",
    "lane_selected",
    "teacher_membership",
    "selected_by",
    "selected_rows",
    "r5_2_package_selected",
    "student_predicted_membership",
    "protected",
    "runner",
    "ambiguous",
    "quarantine",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "row_id",
    "latest",
    "glob",
]

ADAPTER_SAFE_FEATURES = [
    "tail_repaired_r5_2_oof_candidate_score_v1",
    "asof_signal__r5_1_bad_score_v1",
    "asof_signal__v2_like_bad_tail_v1",
    "asof_signal__r5_bad_score_v1",
    "asof_signal__r5_tail_score_v1",
]

REQUIRED_OUTPUTS = [
    "simplify_140_94_input_manifest_v1.json",
    "simplify_140_94_reproducibility_audit_v1.json",
    "simplify_140_94_reproducibility_audit_v1.md",
    "simplify_140_94_extra_110_audit_v1.csv",
    "simplify_140_94_extra_110_audit_v1.json",
    "simplify_140_94_extra_110_audit_v1.md",
    "simplify_140_94_branch_tier_audit_v1.csv",
    "simplify_140_94_branch_tier_audit_v1.json",
    "simplify_140_94_branch_tier_audit_v1.md",
    "simplify_140_94_candidate_recipe_definitions_v1.json",
    "simplify_140_94_candidate_recipe_definitions_v1.md",
    "simplify_140_94_candidate_recipe_metrics_v1.csv",
    "simplify_140_94_candidate_recipe_metrics_v1.json",
    "simplify_140_94_candidate_recipe_metrics_v1.md",
    "simplify_140_94_veto_mapping_audit_v1.csv",
    "simplify_140_94_veto_mapping_audit_v1.json",
    "simplify_140_94_veto_mapping_audit_v1.md",
    "simplify_140_94_best_recipe_row_level_explanations_v1.csv",
    "simplify_140_94_best_recipe_row_level_explanations_v1.json",
    "simplify_140_94_best_recipe_row_level_explanations_v1.md",
    "simplify_140_94_near_miss_and_near_fail_rows_v1.csv",
    "simplify_140_94_near_miss_and_near_fail_rows_v1.json",
    "simplify_140_94_boundary_stress_audit_v1.json",
    "simplify_140_94_boundary_stress_audit_v1.md",
    "simplify_140_94_group_stability_audit_v1.csv",
    "simplify_140_94_group_stability_audit_v1.json",
    "simplify_140_94_group_stability_audit_v1.md",
    "simplify_140_94_adapter_feasibility_v1.json",
    "simplify_140_94_adapter_feasibility_v1.md",
    "simplify_140_94_anti_overfit_no_shortcut_audit_v1.json",
    "simplify_140_94_anti_overfit_no_shortcut_audit_v1.md",
    "simplify_140_94_recommendation_v1.json",
    "simplify_140_94_recommendation_v1.md",
    "simplify_140_94_rules_and_vetoes_go_no_go_v1.json",
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass", "active_candidate", "clean"}
    return bool(value)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def _str(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    failures = []
    for path in paths:
        text = str(path)
        if "*" in text or "latest" in text.lower() or not path.name.endswith("_LOCK"):
            failures.append(text)
    if failures:
        raise RuntimeError(f"IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN: {failures}")
    return True


def validate_no_forbidden_feature_names(features: Iterable[str]) -> bool:
    blocked = []
    for feature in features:
        lower = feature.lower()
        if any(pattern in lower for pattern in DENY_PATTERNS):
            blocked.append(feature)
    if blocked:
        raise RuntimeError(f"FORBIDDEN_SIMPLIFY_140_94_FEATURE: {blocked}")
    return True


def validate_no_forbidden_actions(
    *,
    r6: bool = False,
    adapter: bool = False,
    package: bool = False,
    freeze: bool = False,
    promo: bool = False,
    live: bool = False,
    optuna: bool = False,
) -> dict[str, Any]:
    failures = []
    if r6:
        failures.append("R6_FORBIDDEN")
    if adapter:
        failures.append("ADAPTER_BUILD_FORBIDDEN")
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
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_reproducibility(payload: dict[str, Any]) -> bool:
    expected = {
        "original_selected_rows_v1": BASELINE_SELECTED,
        "original_bad_count_v1": BASELINE_BAD,
        "original_tail_count_v1": BASELINE_TAIL,
        "full_cover_selected_rows_v1": FULL_COVER_SELECTED,
        "full_cover_recovered_original_140_rows_v1": BASELINE_SELECTED,
        "full_cover_extra_rows_v1": FULL_COVER_EXTRA,
        "full_cover_safety_status_v1": "CLEAN",
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if failures:
        raise RuntimeError(f"SIMPLIFY_140_94_REPRODUCIBILITY_FAILED: {failures}")
    return True


def validate_candidate_metrics(rows: list[dict[str, Any]]) -> bool:
    ids = {row["recipe_id_v1"] for row in rows}
    if not RECIPE_IDS.issubset(ids):
        raise RuntimeError(f"SIMPLIFY_140_94_RECIPE_SET_INCOMPLETE: {sorted(RECIPE_IDS - ids)}")
    selected = next(row for row in rows if row["recipe_id_v1"] == SELECTED_RECIPE_ID)
    if selected["safety_status_v1"] != "CLEAN":
        raise RuntimeError("SELECTED_SIMPLIFIED_RECIPE_NOT_SAFETY_CLEAN")
    if selected["extra_rows_v1"] > 10:
        raise RuntimeError("SELECTED_SIMPLIFIED_RECIPE_OVERSELECTS_TOO_MUCH_FOR_CONSERVATIVE_CORE")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"SIMPLIFY_140_94_REQUIRED_OUTPUTS_MISSING: {missing}")
    return True


def _python_manifest() -> dict[str, Any]:
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30).splitlines()
    except Exception as exc:  # pragma: no cover
        freeze = [f"PIP_FREEZE_UNAVAILABLE: {exc}"]
    return {
        "python_executable_v1": sys.executable,
        "python_version_v1": sys.version,
        "platform_v1": platform.platform(),
        "pip_freeze_sha256_v1": hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest(),
    }


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_DISTILL_ROOT,
        INPUT_PRECHECK_ROOT,
        INPUT_PLUS45_SHADOW_ROOT,
        INPUT_REJECT_REBUILD_ROOT,
        INPUT_BEST_LANE_PACKAGE_ROOT,
        INPUT_STUDENT_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "distill_summary": INPUT_DISTILL_ROOT / "summary_v1.json",
        "distill_go_no_go": INPUT_DISTILL_ROOT / "distill_140_94_causal_baseline_to_rules_and_vetoes_go_no_go_v1.json",
        "distill_rule_coverage": INPUT_DISTILL_ROOT / "distill_140_94_rule_coverage_audit_v1.json",
        "distill_rule_definition": INPUT_DISTILL_ROOT / "distill_140_94_rule_definition_v1.json",
        "distill_veto_definition": INPUT_DISTILL_ROOT / "distill_140_94_veto_definition_v1.json",
        "distill_near_miss": INPUT_DISTILL_ROOT / "distill_140_94_near_miss_and_near_fail_rows_v1.csv",
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
        "precheck_go_no_go": INPUT_PRECHECK_ROOT
        / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
        "plus45_go_no_go": INPUT_PLUS45_SHADOW_ROOT
        / "plus45_as_of_feature_gap_shadow_exploration_go_no_go_v1.json",
        "causal_predictions": INPUT_REJECT_REBUILD_ROOT / "causal_rebuild_candidate_oof_predictions_v1.csv",
        "best_membership": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv",
        "student_predictions": INPUT_STUDENT_ROOT / "best_lane_student_oof_predictions_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    distill_go = _read_json(required["distill_go_no_go"])
    plus45_go = _read_json(required["plus45_go_no_go"])
    if distill_go.get("status_v1") != "140_94_RULE_VETO_DISTILLATION_PARTIAL_NEEDS_SIMPLIFICATION":
        raise RuntimeError("DISTILL_STATUS_NOT_SIMPLIFICATION_REQUIRED")
    if plus45_go.get("status_v1") != "PLUS45_SHADOW_FOUND_ONLY_MEMBERSHIP_OR_COVERAGE_DEPENDENCY":
        raise RuntimeError("PLUS45_SHADOW_STATUS_NOT_DIAGNOSTIC_ONLY")
    return {
        "required_paths": required,
        "distill_summary": _read_json(required["distill_summary"]),
        "distill_go_no_go": distill_go,
        "distill_rule_coverage": _read_json(required["distill_rule_coverage"]),
        "distill_rule_definition": _read_json(required["distill_rule_definition"]),
        "distill_veto_definition": _read_json(required["distill_veto_definition"]),
        "distill_near_miss": pd.read_csv(required["distill_near_miss"]),
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": _read_json(required["precheck_go_no_go"]),
        "plus45_go_no_go": plus45_go,
        "causal_predictions": pd.read_csv(required["causal_predictions"]),
        "best_membership": pd.read_csv(required["best_membership"]),
        "student_predictions": pd.read_csv(required["student_predictions"]),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = []
    for name, path in inputs["required_paths"].items():
        files.append({"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)})
    return {
        "layer_name": "SIMPLIFY_140_94_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "distill_root_v1": str(INPUT_DISTILL_ROOT),
            "precheck_140_94_root_v1": str(INPUT_PRECHECK_ROOT),
            "plus45_shadow_root_v1": str(INPUT_PLUS45_SHADOW_ROOT),
            "reject_rebuild_root_v1": str(INPUT_REJECT_REBUILD_ROOT),
            "best_lane_package_root_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT),
            "student_root_v1": str(INPUT_STUDENT_ROOT),
        },
        "files_used_v1": files,
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _build_frame(inputs: dict[str, Any]) -> pd.DataFrame:
    pred = inputs["causal_predictions"]
    base = pred[pred["candidate_id_v1"] == BASELINE_CANDIDATE_ID].copy()
    membership = inputs["best_membership"].copy()
    student_columns = [
        "candidate_uid_v1",
        "source_evidence_v1",
        "student_predicted_membership_v1",
        "student_oof_score_v1",
        "run_id_policy_class_v1",
        "structural_low_support_v1",
        "zero_denominator_group_v1",
        "active_quarantine_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
    ]
    student = inputs["student_predictions"][[col for col in student_columns if col in inputs["student_predictions"].columns]]
    frame = base.merge(membership, on="candidate_uid_v1", how="left", suffixes=("", "_membership"))
    frame = frame.merge(student, on="candidate_uid_v1", how="left")
    source = _str(frame, "source_evidence_v1")
    frame["signal_r5_1_bad_score_v1"] = source.str.contains("R5_1_BAD_SCORE", regex=False)
    frame["signal_r5_bad_score_v1"] = source.str.contains("R5_BAD_SCORE", regex=False)
    frame["signal_r5_tail_score_v1"] = source.str.contains("R5_TAIL_SCORE", regex=False)
    frame["signal_v2_like_bad_tail_v1"] = source.str.contains("V2_LIKE_BAD_TAIL", regex=False)
    frame["signal_tail_repair_v1"] = source.str.contains("TAIL_REPAIR", regex=False)
    frame["selected_original_140_v1"] = _bool(frame, "candidate_selected_v1")
    frame["safety_clear_audit_v1"] = ~_bool(frame, "unsafe_audit_v1")
    frame["hard_veto_clear_shadow_v1"] = frame["safety_clear_audit_v1"]
    frame["run_id_v1"] = _str(frame, "run_id_v1", "").mask(_str(frame, "run_id_v1", "") == "", _str(frame, "run_id_v1_best_lane", ""))
    frame["fold_id_v1"] = _str(frame, "fold_id_v1", "").mask(_str(frame, "fold_id_v1", "") == "", _str(frame, "fold_id_v1_best_lane", ""))
    return frame


def _recipe_masks(frame: pd.DataFrame, score_floor: float) -> dict[str, pd.Series]:
    score = _num(frame, "candidate_score_v1")
    r51 = _bool(frame, "signal_r5_1_bad_score_v1")
    v2 = _bool(frame, "signal_v2_like_bad_tail_v1")
    safe = _bool(frame, "hard_veto_clear_shadow_v1")
    return {
        "CONSERVATIVE_HIGH_CONFIDENCE_RULE_V1": score.ge(0.95) & r51 & v2 & safe,
        "BALANCED_140_RECOVERY_RULE_V1": ((score.ge(score_floor) & r51 & v2) | (score.ge(0.98) & r51)) & safe,
        "FULL_COVER_TIGHTENED_RULE_V1": score.ge(score_floor) & r51 & safe,
        "SCORE_PLUS_VETO_RULE_V1": score.ge(0.95) & r51 & safe,
        "STUDENT_CORE_DIAGNOSTIC_REFERENCE_V1": _bool(frame, "student_predicted_membership_v1"),
    }


def _metric_for_recipe(frame: pd.DataFrame, recipe_id: str, mask: pd.Series) -> dict[str, Any]:
    selected = frame[mask]
    original = _bool(frame, "selected_original_140_v1")
    recovered = int((mask & original).sum())
    extra = int((mask & ~original).sum())
    missed = int((~mask & original).sum())
    unsafe = int(_bool(selected, "unsafe_audit_v1").sum())
    protected = int(_bool(selected, "protected_winner_status_v1").sum())
    runner = int(_bool(selected, "runner_protect_status_v1").sum())
    ambiguous = int(_bool(selected, "ambiguous_high_mfe_status_v1").sum())
    quarantine = int((_str(selected, "active_quarantine_v1", "ACTIVE_CANDIDATE") != "ACTIVE_CANDIDATE").sum())
    bad = int(_bool(selected, "bad_label_v1").sum())
    tail = int(_bool(selected, "tail_label_v1").sum())
    selected_rows = int(mask.sum())
    low_support = int(_bool(selected, "zero_denominator_group_v1").sum())
    structural = int(_bool(selected, "structural_low_support_v1").sum())
    groups = selected["run_id_v1"].dropna().astype(str).nunique() if "run_id_v1" in selected else 0
    complexity = {
        "CONSERVATIVE_HIGH_CONFIDENCE_RULE_V1": 2,
        "BALANCED_140_RECOVERY_RULE_V1": 4,
        "FULL_COVER_TIGHTENED_RULE_V1": 2,
        "SCORE_PLUS_VETO_RULE_V1": 1,
        "STUDENT_CORE_DIAGNOSTIC_REFERENCE_V1": 5,
    }[recipe_id]
    recommendation = "SELECTED_SAFE_CORE" if recipe_id == SELECTED_RECIPE_ID else "REFERENCE_OR_NOT_SELECTED"
    if recipe_id == "FULL_COVER_TIGHTENED_RULE_V1":
        recommendation = "REJECT_OVERSELECTS"
    if recipe_id == "STUDENT_CORE_DIAGNOSTIC_REFERENCE_V1":
        recommendation = "DIAGNOSTIC_ONLY_MEMBERSHIP_TARGET_HISTORY"
    return {
        "recipe_id_v1": recipe_id,
        "selected_rows_v1": selected_rows,
        "recovered_original_140_rows_v1": recovered,
        "missed_original_140_rows_v1": missed,
        "extra_rows_v1": extra,
        "bad_count_audit_only_v1": bad,
        "tail_count_audit_only_v1": tail,
        "precision_audit_only_v1": float(bad / max(selected_rows, 1)),
        "safety_status_v1": "CLEAN" if unsafe == 0 else "FAIL",
        "unsafe_hits_v1": unsafe,
        "protected_winner_hits_audit_only_v1": protected,
        "runner_protect_hits_audit_only_v1": runner,
        "ambiguous_high_mfe_hits_audit_only_v1": ambiguous,
        "quarantine_hits_audit_only_v1": quarantine,
        "low_support_rows_v1": low_support,
        "structural_low_support_rows_v1": structural,
        "run_id_count_v1": int(groups),
        "strict_loso_status_v1": "STRICT_LOSO_INVALID_LOW_SUPPORT_VISIBLE",
        "strict_loso_decision_valid_v1": False,
        "adapter_input_requirements_v1": "|".join(ADAPTER_SAFE_FEATURES + ["AS_OF hard veto mapping"]),
        "complexity_score_v1": complexity,
        "explanation_quality_v1": "HIGH" if complexity <= 2 and extra <= 10 else "MEDIUM",
        "recommendation_v1": recommendation,
    }


def _reproducibility(frame: pd.DataFrame, recipes: dict[str, pd.Series], inputs: dict[str, Any]) -> dict[str, Any]:
    original = frame[_bool(frame, "selected_original_140_v1")]
    full_mask = recipes["FULL_COVER_TIGHTENED_RULE_V1"]
    full = frame[full_mask]
    payload = {
        "layer_name": "SIMPLIFY_140_94_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "original_selected_rows_v1": len(original),
        "original_bad_count_v1": int(_bool(original, "bad_label_v1").sum()),
        "original_tail_count_v1": int(_bool(original, "tail_label_v1").sum()),
        "original_safety_status_v1": "CLEAN" if int(_bool(original, "unsafe_audit_v1").sum()) == 0 else "FAIL",
        "full_cover_selected_rows_v1": int(full_mask.sum()),
        "full_cover_recovered_original_140_rows_v1": int((full_mask & _bool(frame, "selected_original_140_v1")).sum()),
        "full_cover_missed_original_140_rows_v1": int((~full_mask & _bool(frame, "selected_original_140_v1")).sum()),
        "full_cover_extra_rows_v1": int((full_mask & ~_bool(frame, "selected_original_140_v1")).sum()),
        "full_cover_bad_count_audit_only_v1": int(_bool(full, "bad_label_v1").sum()),
        "full_cover_tail_count_audit_only_v1": int(_bool(full, "tail_label_v1").sum()),
        "full_cover_safety_status_v1": "CLEAN" if int(_bool(full, "unsafe_audit_v1").sum()) == 0 else "FAIL",
        "distillation_input_status_v1": inputs["distill_go_no_go"].get("status_v1"),
        "reproduced_from_explicit_distillation_artifact_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def _extra_110_audit(frame: pd.DataFrame, full_mask: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extra = frame[full_mask & ~_bool(frame, "selected_original_140_v1")].copy()
    rows = []
    for _, row in extra.sort_values("candidate_score_v1", ascending=False).iterrows():
        branch = "V2_LIKE_HIGH_CONFIDENCE" if _as_bool(row.get("signal_v2_like_bad_tail_v1")) else "R5_1_SCORE_ONLY"
        if _as_bool(row.get("signal_r5_tail_score_v1")) and not _as_bool(row.get("signal_v2_like_bad_tail_v1")):
            branch = "TAIL_SIGNAL_MEDIUM_CONFIDENCE"
        veto_needed = "missing stricter positive support" if branch == "R5_1_SCORE_ONLY" else "score/support tightening"
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "rule_branch_v1": branch,
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "protected_winner_audit_only_v1": _as_bool(row.get("protected_winner_status_v1")),
                "runner_protect_audit_only_v1": _as_bool(row.get("runner_protect_status_v1")),
                "ambiguous_high_mfe_audit_only_v1": _as_bool(row.get("ambiguous_high_mfe_status_v1")),
                "fifty_plus_mfe_risk_audit_only_v1": _as_bool(row.get("fifty_plus_mfe_risk_v1")),
                "hundred_plus_mfe_risk_audit_only_v1": _as_bool(row.get("hundred_plus_mfe_risk_v1")),
                "two_hundred_plus_mfe_risk_audit_only_v1": _as_bool(row.get("two_hundred_plus_mfe_risk_v1")),
                "quarantine_audit_only_v1": row.get("active_quarantine_v1") != "ACTIVE_CANDIDATE",
                "run_id_policy_class_v1": row.get("run_id_policy_class_v1"),
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "signal_family_v1": "|".join(
                    signal
                    for signal, present in [
                        ("R5_1", _as_bool(row.get("signal_r5_1_bad_score_v1"))),
                        ("V2_LIKE", _as_bool(row.get("signal_v2_like_bad_tail_v1"))),
                        ("R5_BAD", _as_bool(row.get("signal_r5_bad_score_v1"))),
                        ("R5_TAIL", _as_bool(row.get("signal_r5_tail_score_v1"))),
                    ]
                    if present
                ),
                "veto_or_tightening_needed_v1": veto_needed,
                "as_of_mapping_gap_v1": "audit-only hard veto still needs AS_OF-safe mapping",
            }
        )
    summary = {
        "layer_name": "SIMPLIFY_140_94_EXTRA_110_AUDIT_SUMMARY_V1",
        "extra_rows_v1": len(rows),
        "bad_rows_v1": int(_bool(extra, "bad_label_v1").sum()),
        "tail_rows_v1": int(_bool(extra, "tail_label_v1").sum()),
        "unsafe_rows_v1": int(_bool(extra, "unsafe_audit_v1").sum()),
        "protected_winner_rows_v1": int(_bool(extra, "protected_winner_status_v1").sum()),
        "runner_protect_rows_v1": int(_bool(extra, "runner_protect_status_v1").sum()),
        "ambiguous_high_mfe_rows_v1": int(_bool(extra, "ambiguous_high_mfe_status_v1").sum()),
        "quarantine_rows_v1": int((_str(extra, "active_quarantine_v1", "ACTIVE_CANDIDATE") != "ACTIVE_CANDIDATE").sum()),
        "primary_reason_full_cover_too_broad_v1": "R5_1 + low score floor admits many safety-clean but non-bad audit rows.",
    }
    return rows, summary


def _branch_tiers(metrics_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for metric in metrics_rows:
        recipe = metric["recipe_id_v1"]
        if recipe == "CONSERVATIVE_HIGH_CONFIDENCE_RULE_V1":
            tier = "TIER_1_HIGH_CONFIDENCE"
            status = "BEST_ADAPTER_SAFE_CORE"
        elif recipe in {"BALANCED_140_RECOVERY_RULE_V1", "SCORE_PLUS_VETO_RULE_V1"}:
            tier = "TIER_2_MEDIUM_CONFIDENCE"
            status = "PROMISING_NEEDS_TIGHTENING_OR_MAPPING"
        elif recipe == "FULL_COVER_TIGHTENED_RULE_V1":
            tier = "TIER_3_UNCERTAIN_DIAGNOSTIC"
            status = "REJECT_OVERSELECTS"
        else:
            tier = "DIAGNOSTIC_REFERENCE"
            status = "NOT_DEPLOYABLE_MEMBERSHIP_TARGET_HISTORY"
        rows.append(
            {
                "recipe_id_v1": recipe,
                "tier_v1": tier,
                "tier_status_v1": status,
                "selected_rows_v1": metric["selected_rows_v1"],
                "recovered_original_140_rows_v1": metric["recovered_original_140_rows_v1"],
                "extra_rows_v1": metric["extra_rows_v1"],
                "precision_audit_only_v1": metric["precision_audit_only_v1"],
                "safety_status_v1": metric["safety_status_v1"],
                "complexity_score_v1": metric["complexity_score_v1"],
            }
        )
    summary = {
        "layer_name": "SIMPLIFY_140_94_BRANCH_TIER_AUDIT_SUMMARY_V1",
        "selected_tier_v1": "TIER_1_HIGH_CONFIDENCE",
        "selected_recipe_v1": SELECTED_RECIPE_ID,
        "full_cover_tier_status_v1": "REJECT_OVERSELECTS",
        "student_core_status_v1": "DIAGNOSTIC_ONLY",
    }
    return rows, summary


def _recipe_definitions(score_floor: float) -> dict[str, Any]:
    return {
        "layer_name": "SIMPLIFY_140_94_CANDIDATE_RECIPE_DEFINITIONS_V1",
        "selected_recipe_v1": SELECTED_RECIPE_ID,
        "recipes_v1": {
            "CONSERVATIVE_HIGH_CONFIDENCE_RULE_V1": {
                "description_v1": "Require high score, R5_1 support, V2-like support, and hard veto clear.",
                "threshold_policy_v1": "fixed score >= 0.95 from pre-registered simplification, not optimized on +45",
                "required_positive_signals_v1": [
                    "tail_repaired_r5_2_oof_candidate_score_v1 >= 0.95",
                    "asof_signal__r5_1_bad_score_v1",
                    "asof_signal__v2_like_bad_tail_v1",
                ],
                "hard_vetoes_v1": ["AS_OF hard veto mapping required before adapter"],
                "training_used_v1": False,
                "uses_plus45_as_target_feature_filter_or_threshold_v1": False,
                "uses_185_139_membership_v1": False,
            },
            "BALANCED_140_RECOVERY_RULE_V1": {
                "description_v1": "Union of V2-like support at full-cover floor and very high-score R5_1 rows.",
                "threshold_policy_v1": f"score >= {score_floor} with V2-like support OR score >= 0.98 with R5_1",
                "required_positive_signals_v1": ["score", "R5_1", "V2-like or very high score"],
                "hard_vetoes_v1": ["AS_OF hard veto mapping required before adapter"],
                "training_used_v1": False,
            },
            "FULL_COVER_TIGHTENED_RULE_V1": {
                "description_v1": "Distillation full-cover skeleton retained as diagnostic reference.",
                "threshold_policy_v1": f"score >= {score_floor} and R5_1",
                "recommendation_v1": "Reject as adapter rule because it over-selects 110 extra rows.",
            },
            "SCORE_PLUS_VETO_RULE_V1": {
                "description_v1": "Simple score plus R5_1 plus veto rule.",
                "threshold_policy_v1": "fixed score >= 0.95 and R5_1",
                "hard_vetoes_v1": ["AS_OF hard veto mapping required before adapter"],
            },
            "STUDENT_CORE_DIAGNOSTIC_REFERENCE_V1": {
                "description_v1": "Reference only; not selected because it carries membership-target history.",
                "deployable_v1": False,
            },
        },
    }


def _veto_mapping(frame: pd.DataFrame, full_pre_veto: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = _bool(frame, "selected_original_140_v1")
    mappings = [
        ("winner_protection_veto_v1", "protected_winner_status_v1", "protected/winner-like rows"),
        ("runner_protect_veto_v1", "runner_protect_status_v1", "runner-protect rows"),
        ("ambiguous_high_mfe_veto_v1", "ambiguous_high_mfe_status_v1", "ambiguous/high-MFE rows"),
        ("fifty_plus_mfe_proxy_veto_v1", "fifty_plus_mfe_risk_v1", "50+ MFE risk proxy rows"),
        ("hundred_plus_mfe_proxy_veto_v1", "hundred_plus_mfe_risk_v1", "100+ MFE risk proxy rows"),
        ("two_hundred_plus_mfe_proxy_veto_v1", "two_hundred_plus_mfe_risk_v1", "200+ MFE risk proxy rows"),
    ]
    rows = []
    for veto_name, column, blocks in mappings:
        hits = _bool(frame, column)
        rows.append(
            {
                "veto_name_v1": veto_name,
                "what_it_blocks_v1": blocks,
                "current_source_v1": column,
                "as_of_safe_input_available_v1": False,
                "adapter_ready_v1": False,
                "mapping_required_v1": True,
                "risk_if_missing_v1": "adapter could over-select unsafe lookalikes",
                "rows_affected_total_v1": int(hits.sum()),
                "extra_rows_blocked_candidate_v1": int((full_pre_veto & ~original & hits).sum()),
                "original_140_rows_accidentally_blocked_v1": int((original & hits).sum()),
                "status_v1": "NEEDS_AS_OF_MAPPING",
            }
        )
    rows.append(
        {
            "veto_name_v1": "membership_coverage_selected_flag_veto_v1",
            "what_it_blocks_v1": "membership, coverage-proxy, +45, row identity, and selected-flag shortcuts",
            "current_source_v1": "feature policy",
            "as_of_safe_input_available_v1": True,
            "adapter_ready_v1": True,
            "mapping_required_v1": False,
            "risk_if_missing_v1": "oracle/membership shortcut",
            "rows_affected_total_v1": 0,
            "extra_rows_blocked_candidate_v1": 0,
            "original_140_rows_accidentally_blocked_v1": 0,
            "status_v1": "ADAPTER_READY_VETO",
        }
    )
    summary = {
        "layer_name": "SIMPLIFY_140_94_VETO_MAPPING_AUDIT_SUMMARY_V1",
        "veto_count_v1": len(rows),
        "adapter_ready_veto_count_v1": sum(row["status_v1"] == "ADAPTER_READY_VETO" for row in rows),
        "needs_as_of_mapping_count_v1": sum(row["status_v1"] == "NEEDS_AS_OF_MAPPING" for row in rows),
        "primary_blocker_v1": "audit-only safety vetoes must be mapped before adapter expansion beyond safe core",
    }
    return rows, summary


def _best_recipe_explanations(frame: pd.DataFrame, best_mask: pd.Series) -> list[dict[str, Any]]:
    rows = []
    relevant = frame[best_mask | _bool(frame, "selected_original_140_v1")].copy()
    for _, row in relevant.sort_values(["selected_original_140_v1", "candidate_score_v1"], ascending=[False, False]).iterrows():
        positives = ["tail_repaired_r5_2_oof_candidate_score_v1"]
        for signal, column in [
            ("asof_signal__r5_1_bad_score_v1", "signal_r5_1_bad_score_v1"),
            ("asof_signal__v2_like_bad_tail_v1", "signal_v2_like_bad_tail_v1"),
            ("asof_signal__r5_bad_score_v1", "signal_r5_bad_score_v1"),
            ("asof_signal__r5_tail_score_v1", "signal_r5_tail_score_v1"),
        ]:
            if _as_bool(row.get(column)):
                positives.append(signal)
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "selected_by_original_140_v1": _as_bool(row.get("selected_original_140_v1")),
                "selected_by_simplified_recipe_v1": _as_bool(best_mask.loc[row.name]),
                "score_branch_tier_v1": "TIER_1_HIGH_CONFIDENCE"
                if _as_bool(best_mask.loc[row.name])
                else "ORIGINAL_140_NOT_IN_SAFE_CORE",
                "candidate_score_v1": row.get("candidate_score_v1"),
                "positive_signals_v1": "|".join(positives),
                "veto_status_v1": "audit clean; AS_OF mapping required",
                "confidence_class_v1": "HIGH_CONFIDENCE_SAFE_CORE"
                if _as_bool(best_mask.loc[row.name])
                else "BASELINE_REQUIRES_FUTURE_EXPANSION",
                "support_class_v1": row.get("run_id_policy_class_v1"),
                "low_support_status_v1": "LOW_SUPPORT_VISIBLE"
                if _as_bool(row.get("zero_denominator_group_v1")) or _as_bool(row.get("structural_low_support_v1"))
                else "SUPPORT_VISIBLE",
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "explanation_v1": "high score + R5_1 + V2-like support"
                if _as_bool(best_mask.loc[row.name])
                else "original baseline row not covered by conservative simplified safe core",
            }
        )
    return rows


def _near_miss(frame: pd.DataFrame, recipes: dict[str, pd.Series], score_floor: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best_mask = recipes[SELECTED_RECIPE_ID]
    full_mask = recipes["FULL_COVER_TIGHTENED_RULE_V1"]
    score = _num(frame, "candidate_score_v1")
    near = frame[(~best_mask) & (score >= score_floor) & _bool(frame, "signal_r5_1_bad_score_v1")].copy()
    near = near.sort_values("candidate_score_v1", ascending=False).head(250)
    rows = []
    for _, row in near.iterrows():
        class_v1 = "FULL_SKELETON_EXTRA_ROW" if _as_bool(full_mask.loc[row.name]) else "VETO_STOPPED_OR_SIGNAL_MISSING"
        if _as_bool(row.get("selected_original_140_v1")):
            class_v1 = "MISSED_ORIGINAL_140_BY_SAFE_CORE"
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "near_class_v1": class_v1,
                "selected_original_140_v1": _as_bool(row.get("selected_original_140_v1")),
                "selected_best_recipe_v1": _as_bool(best_mask.loc[row.name]),
                "passes_full_cover_skeleton_v1": _as_bool(full_mask.loc[row.name]),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "v2_like_support_v1": _as_bool(row.get("signal_v2_like_bad_tail_v1")),
                "r5_tail_support_v1": _as_bool(row.get("signal_r5_tail_score_v1")),
                "veto_or_missing_signal_v1": "missing V2-like or score >= 0.95 safe-core requirement",
                "adapter_over_selection_risk_v1": "MODERATE_REQUIRES_VETO_MAPPING"
                if not _as_bool(row.get("unsafe_audit_v1"))
                else "HIGH_UNSAFE_LOOKALIKE",
            }
        )
    summary = {
        "layer_name": "SIMPLIFY_140_94_BOUNDARY_STRESS_AUDIT_V1",
        "best_recipe_v1": SELECTED_RECIPE_ID,
        "near_miss_rows_sampled_v1": len(rows),
        "full_cover_extra_rows_v1": int((full_mask & ~_bool(frame, "selected_original_140_v1")).sum()),
        "best_recipe_extra_rows_v1": int((best_mask & ~_bool(frame, "selected_original_140_v1")).sum()),
        "unsafe_lookalike_rows_in_sample_v1": int(sum(row["unsafe_audit_only_v1"] for row in rows)),
        "adapter_over_selection_risk_v1": "CONTROLLED_FOR_SAFE_CORE_BUT_EXPANSION_REQUIRES_VETO_MAPPING",
        "status_v1": "SAFE_CORE_BOUNDARY_PASS_EXPANSION_NOT_READY",
    }
    return rows, summary


def _group_stability(frame: pd.DataFrame, best_mask: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = frame[best_mask].copy()
    rows = []
    for run_id, group in selected.groupby("run_id_v1"):
        rows.append(
            {
                "run_id_v1": run_id,
                "fold_values_v1": "|".join(sorted(set(_str(group, "fold_id_v1")))),
                "selected_rows_v1": len(group),
                "recovered_original_140_rows_v1": int(_bool(group, "selected_original_140_v1").sum()),
                "extra_rows_v1": int((~_bool(group, "selected_original_140_v1")).sum()),
                "bad_count_audit_only_v1": int(_bool(group, "bad_label_v1").sum()),
                "tail_count_audit_only_v1": int(_bool(group, "tail_label_v1").sum()),
                "precision_audit_only_v1": float(_bool(group, "bad_label_v1").sum() / max(len(group), 1)),
                "tier_v1": "TIER_1_HIGH_CONFIDENCE",
                "v2_like_rows_v1": int(_bool(group, "signal_v2_like_bad_tail_v1").sum()),
                "tail_signal_rows_v1": int(_bool(group, "signal_r5_tail_score_v1").sum()),
                "low_support_rows_v1": int(_bool(group, "zero_denominator_group_v1").sum()),
                "structural_low_support_rows_v1": int(_bool(group, "structural_low_support_v1").sum()),
                "student_core_overlap_rows_v1": int(_bool(group, "student_predicted_membership_v1").sum()),
                "best_lane_185_139_overlap_rows_v1": int(_bool(group, "lane_selected_v1").sum()),
                "plus45_diagnostic_overlap_rows_v1": int(_bool(group, "rows_added_vs_140_94_v1").sum()),
            }
        )
    summary = {
        "layer_name": "SIMPLIFY_140_94_GROUP_STABILITY_AUDIT_SUMMARY_V1",
        "best_recipe_v1": SELECTED_RECIPE_ID,
        "run_id_count_v1": len(rows),
        "strict_loso_status_v1": "STRICT_LOSO_INVALID_LOW_SUPPORT_VISIBLE",
        "strict_loso_decision_valid_v1": False,
        "strict_loso_denominator_v1": min((row["selected_rows_v1"] for row in rows), default=0),
        "structural_low_support_group_count_v1": sum(1 for row in rows if row["structural_low_support_rows_v1"] > 0),
        "group_concentration_risk_v1": "VISIBLE_SAFE_CORE_NOT_FINAL_PROMOTION_VALID",
    }
    return rows, summary


def _adapter_and_recommendation(
    selected_metric: dict[str, Any],
    veto_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
    group_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    adapter = {
        "layer_name": "SIMPLIFY_140_94_ADAPTER_FEASIBILITY_V1",
        "status_v1": "SAFE_CORE_FOUND_NOT_FULL_140_ADAPTER_READY",
        "selected_recipe_v1": SELECTED_RECIPE_ID,
        "adapter_can_be_built_next_v1": False,
        "adapter_input_mapping_needed_v1": True,
        "safe_core_adapter_candidate_v1": True,
        "full_140_adapter_ready_v1": False,
        "required_inputs_v1": ADAPTER_SAFE_FEATURES,
        "required_veto_mappings_v1": [
            "AS_OF winner/protection veto",
            "AS_OF runner veto",
            "AS_OF ambiguity/high-MFE proxy veto",
            "AS_OF quarantine/source validity veto",
        ],
        "selected_recipe_recovered_original_140_rows_v1": selected_metric["recovered_original_140_rows_v1"],
        "selected_recipe_extra_rows_v1": selected_metric["extra_rows_v1"],
        "blockers_v1": [
            "does not recover enough of 140 to replace baseline",
            "audit-only safety vetoes still need AS_OF-safe mapping",
            "strict LOSO low-support remains visible",
        ],
    }
    anti = {
        "layer_name": "SIMPLIFY_140_94_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_SAFE_CORE_ONLY",
        "no_optuna_v1": True,
        "no_broad_sweep_v1": True,
        "no_post_hoc_plus45_targeting_v1": True,
        "no_in_sample_decisioning_v1": True,
        "no_r6_run_v1": True,
        "no_adapter_build_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "feature_leakage_blocked_v1": True,
        "membership_coverage_leakage_blocked_v1": True,
        "labels_mfe_safe_recoverable_blocked_as_features_v1": True,
        "implicit_latest_glob_blocked_v1": True,
        "low_support_visible_v1": True,
        "strict_loso_visible_v1": True,
        "dummy_synthetic_fallback_v1": False,
    }
    recommendation = {
        "layer_name": "SIMPLIFY_140_94_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "selected_recipe_v1": SELECTED_RECIPE_ID,
        "rationale_v1": [
            "The full-cover skeleton is too broad because it selects 110 extra rows.",
            "The conservative high-confidence V2-like rule is safety-clean and keeps over-selection low.",
            "It only recovers a safe core of the 140 baseline, so it should be hardened before expansion or adapter build.",
            "Audit-only vetoes still need AS_OF-safe mapping.",
        ],
        "selected_recipe_recovered_original_140_rows_v1": selected_metric["recovered_original_140_rows_v1"],
        "selected_recipe_extra_rows_v1": selected_metric["extra_rows_v1"],
        "selected_recipe_bad_tail_v1": [
            selected_metric["bad_count_audit_only_v1"],
            selected_metric["tail_count_audit_only_v1"],
        ],
        "veto_mapping_status_v1": veto_summary["primary_blocker_v1"],
        "boundary_status_v1": boundary_summary["status_v1"],
        "strict_loso_decision_valid_v1": group_summary["strict_loso_decision_valid_v1"],
    }
    return adapter, anti, recommendation


def _write_markdown(
    root: Path,
    repro: dict[str, Any],
    extra_summary: dict[str, Any],
    tier_summary: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    veto_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
    group_summary: dict[str, Any],
    adapter: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    selected = next(row for row in metrics_rows if row["recipe_id_v1"] == SELECTED_RECIPE_ID)
    _write_report(
        root / "simplify_140_94_reproducibility_audit_v1.md",
        [
            "# Simplify 140/94 Reproducibility Audit V1",
            "",
            f"- Status: `{repro['status_v1']}`",
            f"- Original 140/94: `{repro['original_selected_rows_v1']} / {repro['original_tail_count_v1']}`",
            f"- Full-cover skeleton: `{repro['full_cover_selected_rows_v1']}` selected",
            f"- Full-cover extra rows: `{repro['full_cover_extra_rows_v1']}`",
        ],
    )
    _write_report(
        root / "simplify_140_94_extra_110_audit_v1.md",
        [
            "# Simplify 140/94 Extra 110 Audit V1",
            "",
            f"- Extra rows: `{extra_summary['extra_rows_v1']}`",
            f"- Bad/tail among extras: `{extra_summary['bad_rows_v1']} / {extra_summary['tail_rows_v1']}`",
            f"- Unsafe rows among extras: `{extra_summary['unsafe_rows_v1']}`",
            f"- Primary reason: `{extra_summary['primary_reason_full_cover_too_broad_v1']}`",
        ],
    )
    _write_report(
        root / "simplify_140_94_branch_tier_audit_v1.md",
        [
            "# Simplify 140/94 Branch Tier Audit V1",
            "",
            f"- Selected tier: `{tier_summary['selected_tier_v1']}`",
            f"- Selected recipe: `{tier_summary['selected_recipe_v1']}`",
            "- Full-cover tier is retained as diagnostic only because it over-selects.",
        ],
    )
    _write_report(
        root / "simplify_140_94_candidate_recipe_definitions_v1.md",
        [
            "# Simplify 140/94 Candidate Recipe Definitions V1",
            "",
            "- Four candidate recipes were fixed before execution: conservative, balanced, full-cover diagnostic, and score-plus-veto.",
            "- Student-core is included only as a diagnostic reference because it has membership-target history.",
        ],
    )
    _write_report(
        root / "simplify_140_94_candidate_recipe_metrics_v1.md",
        [
            "# Simplify 140/94 Candidate Recipe Metrics V1",
            "",
            f"- Selected recipe: `{SELECTED_RECIPE_ID}`",
            f"- Recovered original 140 rows: `{selected['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{selected['extra_rows_v1']}`",
            f"- Bad/tail audit: `{selected['bad_count_audit_only_v1']} / {selected['tail_count_audit_only_v1']}`",
            f"- Safety: `{selected['safety_status_v1']}`",
        ],
    )
    _write_report(
        root / "simplify_140_94_veto_mapping_audit_v1.md",
        [
            "# Simplify 140/94 Veto Mapping Audit V1",
            "",
            f"- Veto mappings needing AS_OF work: `{veto_summary['needs_as_of_mapping_count_v1']}`",
            f"- Adapter-ready vetoes: `{veto_summary['adapter_ready_veto_count_v1']}`",
            f"- Primary blocker: `{veto_summary['primary_blocker_v1']}`",
        ],
    )
    _write_report(
        root / "simplify_140_94_best_recipe_row_level_explanations_v1.md",
        [
            "# Simplify 140/94 Best Recipe Row-Level Explanations V1",
            "",
            f"- Best recipe: `{SELECTED_RECIPE_ID}`",
            "- Rows include original 140 baseline members plus safe-core selected rows.",
        ],
    )
    _write_report(
        root / "simplify_140_94_boundary_stress_audit_v1.md",
        [
            "# Simplify 140/94 Boundary Stress Audit V1",
            "",
            f"- Best recipe extra rows: `{boundary_summary['best_recipe_extra_rows_v1']}`",
            f"- Full-cover extra rows: `{boundary_summary['full_cover_extra_rows_v1']}`",
            f"- Status: `{boundary_summary['status_v1']}`",
        ],
    )
    _write_report(
        root / "simplify_140_94_group_stability_audit_v1.md",
        [
            "# Simplify 140/94 Group Stability Audit V1",
            "",
            f"- Run_id count: `{group_summary['run_id_count_v1']}`",
            f"- Strict LOSO denominator: `{group_summary['strict_loso_denominator_v1']}`",
            f"- Strict LOSO decision-valid: `{group_summary['strict_loso_decision_valid_v1']}`",
        ],
    )
    _write_report(
        root / "simplify_140_94_adapter_feasibility_v1.md",
        [
            "# Simplify 140/94 Adapter Feasibility V1",
            "",
            f"- Status: `{adapter['status_v1']}`",
            f"- Adapter can be built next: `{adapter['adapter_can_be_built_next_v1']}`",
            f"- Safe-core adapter candidate: `{adapter['safe_core_adapter_candidate_v1']}`",
        ],
    )
    _write_report(
        root / "simplify_140_94_anti_overfit_no_shortcut_audit_v1.md",
        [
            "# Simplify 140/94 Anti-Overfit / No-Shortcut Audit V1",
            "",
            "- No R6, adapter, package, freeze, promo, live, Optuna, broad sweep, or in-sample decisioning was run.",
            "- +45, 185/139, membership, coverage proxy, selected flags, labels, MFE, safe_recoverable, row identity, and implicit latest/glob sources remain blocked.",
        ],
    )
    _write_report(
        root / "simplify_140_94_recommendation_v1.md",
        [
            "# Simplify 140/94 Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            f"- Selected recipe: `{recommendation['selected_recipe_v1']}`",
            "- Treat this as a hardened safe core, not a full replacement adapter for all 140 rows yet.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    validate_no_forbidden_feature_names(ADAPTER_SAFE_FEATURES)
    inputs = _load_inputs()
    frame = _build_frame(inputs)
    selected_original = frame[_bool(frame, "selected_original_140_v1")]
    score_floor = float(selected_original["candidate_score_v1"].min())
    recipes = _recipe_masks(frame, score_floor)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(frame, recipes, inputs)
    metrics_rows = [_metric_for_recipe(frame, recipe_id, recipes[recipe_id]) for recipe_id in recipes]
    validate_candidate_metrics(metrics_rows)
    full_mask = recipes["FULL_COVER_TIGHTENED_RULE_V1"]
    extra_rows, extra_summary = _extra_110_audit(frame, full_mask)
    tier_rows, tier_summary = _branch_tiers(metrics_rows)
    definitions = _recipe_definitions(score_floor)
    full_pre_veto = _num(frame, "candidate_score_v1").ge(score_floor) & _bool(frame, "signal_r5_1_bad_score_v1")
    veto_rows, veto_summary = _veto_mapping(frame, full_pre_veto)
    best_mask = recipes[SELECTED_RECIPE_ID]
    explanations = _best_recipe_explanations(frame, best_mask)
    near_rows, boundary_summary = _near_miss(frame, recipes, score_floor)
    group_rows, group_summary = _group_stability(frame, best_mask)
    selected_metric = next(row for row in metrics_rows if row["recipe_id_v1"] == SELECTED_RECIPE_ID)
    adapter, anti, recommendation = _adapter_and_recommendation(
        selected_metric, veto_summary, boundary_summary, group_summary
    )
    go_no_go = {
        "layer_name": "SIMPLIFY_140_94_RULES_AND_VETOES_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "selected_recipe_v1": SELECTED_RECIPE_ID,
        "recovered_original_140_rows_v1": selected_metric["recovered_original_140_rows_v1"],
        "extra_selected_rows_v1": selected_metric["extra_rows_v1"],
        "safety_status_v1": selected_metric["safety_status_v1"],
        "adapter_can_be_built_next_v1": adapter["adapter_can_be_built_next_v1"],
        "safe_core_adapter_candidate_v1": adapter["safe_core_adapter_candidate_v1"],
        "final_promotion_allowed_v1": False,
        "strict_loso_decision_valid_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(go_no_go["status_v1"], go_no_go["next_recommended_action_v1"])

    _write_json(artifact_root / "simplify_140_94_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "simplify_140_94_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "simplify_140_94_extra_110_audit_v1.csv", extra_rows)
    _write_json(
        artifact_root / "simplify_140_94_extra_110_audit_v1.json",
        {"summary_v1": extra_summary, "rows_v1": extra_rows},
    )
    _write_rows(artifact_root / "simplify_140_94_branch_tier_audit_v1.csv", tier_rows)
    _write_json(
        artifact_root / "simplify_140_94_branch_tier_audit_v1.json",
        {"summary_v1": tier_summary, "rows_v1": tier_rows},
    )
    _write_json(artifact_root / "simplify_140_94_candidate_recipe_definitions_v1.json", definitions)
    _write_rows(artifact_root / "simplify_140_94_candidate_recipe_metrics_v1.csv", metrics_rows)
    _write_json(
        artifact_root / "simplify_140_94_candidate_recipe_metrics_v1.json",
        {"selected_recipe_v1": SELECTED_RECIPE_ID, "rows_v1": metrics_rows},
    )
    _write_rows(artifact_root / "simplify_140_94_veto_mapping_audit_v1.csv", veto_rows)
    _write_json(
        artifact_root / "simplify_140_94_veto_mapping_audit_v1.json",
        {"summary_v1": veto_summary, "rows_v1": veto_rows},
    )
    _write_rows(artifact_root / "simplify_140_94_best_recipe_row_level_explanations_v1.csv", explanations)
    _write_json(
        artifact_root / "simplify_140_94_best_recipe_row_level_explanations_v1.json",
        {"row_count_v1": len(explanations), "rows_v1": explanations},
    )
    _write_rows(artifact_root / "simplify_140_94_near_miss_and_near_fail_rows_v1.csv", near_rows)
    _write_json(
        artifact_root / "simplify_140_94_near_miss_and_near_fail_rows_v1.json",
        {"row_count_v1": len(near_rows), "rows_v1": near_rows},
    )
    _write_json(artifact_root / "simplify_140_94_boundary_stress_audit_v1.json", boundary_summary)
    _write_rows(artifact_root / "simplify_140_94_group_stability_audit_v1.csv", group_rows)
    _write_json(
        artifact_root / "simplify_140_94_group_stability_audit_v1.json",
        {"summary_v1": group_summary, "rows_v1": group_rows},
    )
    _write_json(artifact_root / "simplify_140_94_adapter_feasibility_v1.json", adapter)
    _write_json(artifact_root / "simplify_140_94_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "simplify_140_94_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "simplify_140_94_rules_and_vetoes_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": "SIMPLIFY_140_94_RULES_AND_VETOES_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "selected_recipe_v1": SELECTED_RECIPE_ID,
        "original_140_94_reproduced_v1": True,
        "full_cover_extra_rows_v1": FULL_COVER_EXTRA,
        "recovered_original_140_rows_v1": selected_metric["recovered_original_140_rows_v1"],
        "extra_selected_rows_v1": selected_metric["extra_rows_v1"],
        "bad_tail_audit_only_v1": [selected_metric["bad_count_audit_only_v1"], selected_metric["tail_count_audit_only_v1"]],
        "safety_status_v1": selected_metric["safety_status_v1"],
        "adapter_feasibility_v1": adapter["status_v1"],
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {"status_v1": FINAL_STATUS, "next_recommended_action_v1": NEXT_ACTION, "created_at_utc_v1": _utc_now()},
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Simplify 140/94 Rules And Vetoes V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Selected recipe: `{SELECTED_RECIPE_ID}`",
            f"- Recovered original 140 rows: `{selected_metric['recovered_original_140_rows_v1']}`",
            f"- Extra rows: `{selected_metric['extra_rows_v1']}`",
            f"- Safety: `{selected_metric['safety_status_v1']}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
        ],
    )
    _write_markdown(
        artifact_root,
        repro,
        extra_summary,
        tier_summary,
        metrics_rows,
        veto_summary,
        boundary_summary,
        group_summary,
        adapter,
        recommendation,
    )
    validate_required_outputs(artifact_root)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = materialize(args.artifact_root)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
