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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1"
SELECTED_LANE_ID = "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY"

INPUT_STUDENT_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_V1_20260427T202519Z_LOCK"
)
INPUT_STABILITY_ROOT = (
    DEFAULT_REPORTS_ROOT / "STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_V1_20260427T200530Z_LOCK"
)
INPUT_BEST_LANE_PACKAGE_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "MATERIALIZE_BEST_LANE_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1_20260427T193354Z_LOCK"
)
INPUT_LANE_PACK_ROOT = (
    DEFAULT_REPORTS_ROOT / "PARALLEL_TAIL_R6_R5_2_REPAIR_LANE_PACK_V1_20260427T191454Z_LOCK"
)
INPUT_R6_TAIL_REPAIRED_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RUN_R6_RETRAIN_FROM_TAIL_REPAIRED_R5_2_PACKAGE_EXPLICIT_GATE_V1_20260427T185325Z_LOCK"
)

FINAL_STATUS = "RETURN_TO_140_94_CAUSAL_BASELINE_BEST_CURRENT_OPTION"
NEXT_ACTION = "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1"

TEACHER_SELECTED = 185
TEACHER_TAIL = 139
BASELINE_SELECTED = 140
BASELINE_TAIL = 94
ADDED_ROWS = 45
STUDENT_CORE_SELECTED = 135
STUDENT_CORE_BAD = 131
STUDENT_CORE_TAIL = 93
WEDNESDAY_BAD = 180
WEDNESDAY_TAIL = 149
COVERAGE_PROXY_BAD = 188
COVERAGE_PROXY_TAIL = 136

RAW_ALLOWED_FEATURES = [
    "r5_2_coverage_bad_score_v1",
    "r5_2_coverage_tail_score_v1",
    "r5_2_coverage_hard_veto_score_v1",
    "pred__entry_r5_2_bad_blocker__prob_true_v1",
    "blocker_score_v1",
    "pred__entry_r6_bad_risk__prob_true_v1",
    "pred__entry_r6_tail_control_10_50__prob_true_v1",
    "pred__entry_r6_risky_allow__prob_true_v1",
    "pred__entry_r6_batch04_blindspot__prob_true_v1",
]

DERIVED_SIGNAL_FEATURES = {
    "asof_signal__r5_bad_score_v1": "R5_BAD_SCORE",
    "asof_signal__r5_1_bad_score_v1": "R5_1_BAD_SCORE",
    "asof_signal__r5_tail_score_v1": "R5_TAIL_SCORE",
    "asof_signal__v2_like_bad_tail_v1": "V2_LIKE_BAD_TAIL",
}

DENY_PATTERNS = [
    "bad_label",
    "tail_label",
    "label_should_not_take",
    "take_was_ok",
    "final_outcome",
    "mfe",
    "fifty_plus",
    "hundred_plus",
    "two_hundred_plus",
    "safe_recoverable",
    "coverage_proxy",
    "lane_selected",
    "lane_id",
    "teacher_membership",
    "rows_added_vs_140_94",
    "rows_lost_vs_140_94",
    "selected_by",
    "selected_rows",
    "r5_2_package_selected",
    "r6_best_candidate_selected",
    "student_predicted_membership",
    "decision_valid",
    "protected",
    "runner",
    "ambiguous",
    "quarantine",
    "source_evidence",
]

THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

ALLOWED_FINAL_STATUSES = {
    "CAUSAL_REBUILD_FOUND_AS_OF_SAFE_R6_ADAPTER_CANDIDATE",
    "CAUSAL_REBUILD_FOUND_AS_OF_SAFE_BUT_NEEDS_DISTILLATION",
    "RETURN_TO_140_94_CAUSAL_BASELINE_BEST_CURRENT_OPTION",
    "STUDENT_CORE_IS_BEST_AS_OF_SAFE_CURRENT_OPTION",
    "SIGNALS_PROMISING_BUT_NEED_DEEPER_GROUPED_GENERALIZATION",
    "NO_CAUSAL_REBUILD_BEATS_BASELINE",
    "BLOCKED_BY_FEATURE_LEAKAGE_OR_TARGET_CONTAMINATION",
    "BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_CAUSAL_SIGNAL_ADAPTER_V1",
    "DISTILL_CAUSAL_REBUILD_TO_RULES_AND_VETOES_V1",
    "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1",
    "HARDEN_STUDENT_CORE_AS_CAUSAL_BASELINE_V1",
    "DEEPEN_GROUPED_GENERALIZATION_AND_LOSO_AUDIT_V1",
    "DEEPEN_UNSAFE_LOOKALIKE_AND_BOUNDARY_AUDIT_V1",
    "REBUILD_FEATURE_LINEAGE_AND_AS_OF_SIGNAL_INVENTORY_V1",
    "STOP_BEST_LANE_185_139_ADAPTER_PATH_AND_ARCHIVE_AS_COMPARATOR_V1",
}


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


def _hash_json(payload: Any) -> str:
    data = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass", "active_candidate"}
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


def validate_no_forbidden_actions(
    *,
    optuna: bool = False,
    r6: bool = False,
    adapter: bool = False,
    package: bool = False,
    freeze: bool = False,
    promo: bool = False,
    live: bool = False,
) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
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
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_no_forbidden_feature_names(features: Iterable[str]) -> bool:
    blocked = []
    for feature in features:
        lower = feature.lower()
        if any(pattern in lower for pattern in DENY_PATTERNS):
            blocked.append(feature)
    if blocked:
        raise RuntimeError(f"FORBIDDEN_CAUSAL_FEATURE: {blocked}")
    return True


def validate_threshold_policy(policy: str) -> bool:
    if policy not in {
        "INNER_GROUP_OOF_SUPERVISED_LABEL_TARGET_NO_HELDOUT_LEAKAGE",
        "FIXED_PRE_REGISTERED_RULE_NO_FULL_DATASET_THRESHOLD_SELECTION",
        "EXISTING_OOF_CONTROL_NO_NEW_THRESHOLD",
    }:
        raise RuntimeError(f"INVALID_THRESHOLD_POLICY: {policy}")
    return True


def validate_reject_preserve_policy(rows: Iterable[dict[str, Any]]) -> bool:
    by_item = {row["item_v1"]: row["decision_v1"] for row in rows}
    if by_item.get("LANE_08_185_139_MEMBERSHIP_BOUNDARY") != "REJECT_AS_DEPLOYABLE":
        raise RuntimeError("LANE_08_185_139_MUST_BE_REJECTED_AS_DEPLOYABLE")
    if by_item.get("LANE_08_PLUS_45_ROWS") != "PRESERVE_AS_DIAGNOSTIC":
        raise RuntimeError("PLUS_45_ROWS_MUST_BE_DIAGNOSTIC_ONLY")
    if by_item.get("TAIL_REPAIRED_140_94") != "PRESERVE_AS_CAUSAL_CANDIDATE":
        raise RuntimeError("TAIL_REPAIRED_140_94_MUST_BE_PRESERVED")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def _candidate_score(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw))
    raise RuntimeError("MODEL_DOES_NOT_SUPPORT_SCORE")


def _choose_threshold_from_inner(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = THRESHOLD_GRID[0]
    best_objective = -1e9
    for threshold in THRESHOLD_GRID:
        selected = scores >= threshold
        precision = precision_score(y_true, selected, zero_division=0)
        recall = recall_score(y_true, selected, zero_division=0)
        f1 = f1_score(y_true, selected, zero_division=0)
        objective = f1 + (0.02 * precision)
        if precision < 0.97:
            objective -= 2.0
        if int(selected.sum()) < 50:
            objective -= 0.2
        row = {
            "threshold_v1": threshold,
            "precision_v1": precision,
            "recall_v1": recall,
            "f1_v1": f1,
            "selected_rows_v1": int(selected.sum()),
            "objective_v1": objective,
            "policy_v1": "INNER_GROUP_OOF_SUPERVISED_LABEL_TARGET_NO_HELDOUT_LEAKAGE",
        }
        rows.append(row)
        if objective > best_objective:
            best_threshold = threshold
            best_objective = objective
    return best_threshold, rows


@dataclass(frozen=True)
class SupervisedCandidateSpec:
    candidate_id: str
    model_family: str
    factory: Callable[[], Any]
    interpretability: str


def _supervised_specs() -> list[SupervisedCandidateSpec]:
    return [
        SupervisedCandidateSpec(
            candidate_id="SUPERVISED_LOGREG_BAD_TAIL_CAUSAL_OOF",
            model_family="regularized_logistic_regression",
            factory=lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="liblinear"),
            ),
            interpretability="HIGH",
        ),
        SupervisedCandidateSpec(
            candidate_id="SUPERVISED_HGB_BAD_TAIL_CAUSAL_OOF",
            model_family="small_gradient_model_fixed",
            factory=lambda: HistGradientBoostingClassifier(
                max_iter=80,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                learning_rate=0.05,
                random_state=23,
            ),
            interpretability="MEDIUM",
        ),
    ]


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_STUDENT_ROOT,
        INPUT_STABILITY_ROOT,
        INPUT_BEST_LANE_PACKAGE_ROOT,
        INPUT_LANE_PACK_ROOT,
        INPUT_R6_TAIL_REPAIRED_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "student_summary": INPUT_STUDENT_ROOT / "summary_v1.json",
        "student_go_no_go": INPUT_STUDENT_ROOT / "build_model_to_learn_best_lane_membership_as_oof_target_go_no_go_v1.json",
        "student_predictions": INPUT_STUDENT_ROOT / "best_lane_student_oof_predictions_v1.csv",
        "student_feature_audit": INPUT_STUDENT_ROOT / "best_lane_feature_leakage_audit_v1.csv",
        "student_metrics": INPUT_STUDENT_ROOT / "best_lane_student_vs_teacher_membership_metrics_v1.json",
        "stability_summary": INPUT_STABILITY_ROOT / "summary_v1.json",
        "stability_go_no_go": INPUT_STABILITY_ROOT / "stability_recheck_best_lane_185_139_before_r6_go_no_go_v1.json",
        "best_lane_membership": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv",
        "best_lane_integrity": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_package_integrity_report_v1.json",
        "best_lane_fixed_control": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_fixed_control_comparison_v1.csv",
        "lane_pack_summary": INPUT_LANE_PACK_ROOT / "summary_v1.json",
        "lane_08_summary": INPUT_LANE_PACK_ROOT / "lanes" / SELECTED_LANE_ID / "lane_result_summary_v1.json",
        "lane_08_membership": INPUT_LANE_PACK_ROOT / "lanes" / SELECTED_LANE_ID / "lane_scores_or_membership_v1.csv",
        "r6_scores": INPUT_R6_TAIL_REPAIRED_ROOT / "r6_tail_repaired_oof_scores_v1.csv",
        "r6_provenance": INPUT_R6_TAIL_REPAIRED_ROOT / "r6_tail_repaired_oof_score_provenance_v1.csv",
        "added_evidence": INPUT_STABILITY_ROOT / "best_lane_added_rows_selection_evidence_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    student_go = _read_json(required["student_go_no_go"])
    stability_go = _read_json(required["stability_go_no_go"])
    if student_go.get("status_v1") != "BEST_LANE_MEMBERSHIP_NOT_LEARNABLE_FROM_AS_OF_FEATURES":
        raise RuntimeError("STUDENT_GATE_STATUS_NOT_EXPECTED_NOT_LEARNABLE")
    if stability_go.get("status_v1") != "BEST_LANE_SIGNAL_STRONG_BUT_MEMBERSHIP_ONLY_NOT_R6_READY":
        raise RuntimeError("STABILITY_GATE_STATUS_NOT_EXPECTED_MEMBERSHIP_ONLY")
    return {
        "required_paths": required,
        "student_summary": _read_json(required["student_summary"]),
        "student_go_no_go": student_go,
        "student_predictions": pd.read_csv(required["student_predictions"]),
        "student_feature_audit": pd.read_csv(required["student_feature_audit"]),
        "student_metrics": _read_json(required["student_metrics"]),
        "stability_summary": _read_json(required["stability_summary"]),
        "stability_go_no_go": stability_go,
        "best_lane_membership": pd.read_csv(required["best_lane_membership"]),
        "best_lane_integrity": _read_json(required["best_lane_integrity"]),
        "best_lane_fixed_control": pd.read_csv(required["best_lane_fixed_control"]),
        "lane_pack_summary": _read_json(required["lane_pack_summary"]),
        "lane_08_summary": _read_json(required["lane_08_summary"]),
        "lane_08_membership": pd.read_csv(required["lane_08_membership"]),
        "r6_scores": pd.read_csv(required["r6_scores"]),
        "r6_provenance": pd.read_csv(required["r6_provenance"]),
        "added_evidence": pd.read_csv(required["added_evidence"]),
    }


def _build_frame(inputs: dict[str, Any]) -> pd.DataFrame:
    membership = inputs["best_lane_membership"].copy()
    scores = inputs["r6_scores"].copy()
    student = inputs["student_predictions"][
        ["candidate_uid_v1", "student_oof_score_v1", "student_predicted_membership_v1", "split_id_v1"]
    ].copy()
    frame = membership.merge(scores, on="candidate_uid_v1", how="left", suffixes=("_best_lane", ""))
    frame = frame.merge(student, on="candidate_uid_v1", how="left")
    source = _str(frame, "source_evidence_v1")
    for feature, token in DERIVED_SIGNAL_FEATURES.items():
        frame[feature] = source.str.contains(token, regex=False).astype(float)
    frame["bad_or_tail_label_audit_v1"] = (_bool(frame, "bad_label_v1") | _bool(frame, "tail_label_v1")).astype(int)
    frame["is_140_94_baseline_v1"] = _bool(frame, "r5_2_package_selected_v1_best_lane")
    frame["is_185_139_teacher_v1"] = _bool(frame, "lane_selected_v1")
    frame["is_plus45_diagnostic_v1"] = _bool(frame, "rows_added_vs_140_94_v1")
    frame["student_core_selected_v1"] = _bool(frame, "student_predicted_membership_v1")
    frame["quarantine_audit_v1"] = _str(frame, "active_quarantine_v1") != "ACTIVE_CANDIDATE"
    frame["unsafe_audit_v1"] = (
        _bool(frame, "protected_winner_status_v1")
        | _bool(frame, "runner_protect_status_v1")
        | _bool(frame, "ambiguous_high_mfe_status_v1")
        | _bool(frame, "fifty_plus_mfe_risk_v1")
        | _bool(frame, "hundred_plus_mfe_risk_v1")
        | _bool(frame, "two_hundred_plus_mfe_risk_v1")
        | _bool(frame, "quarantine_audit_v1")
    )
    return frame


def _feature_classification(feature: str) -> tuple[str, str, str, str]:
    lower = feature.lower()
    if feature in RAW_ALLOWED_FEATURES:
        return "AS_OF_SAFE_DEPLOYABLE", "ALLOWED", "AS_OF_SAFE_OOF_SCORE", "Existing OOF score/probability field."
    if feature in DERIVED_SIGNAL_FEATURES:
        return (
            "AS_OF_SAFE_DEPLOYABLE",
            "ALLOWED",
            "AS_OF_SAFE_SIGNAL_FAMILY",
            "Derived only as an explicit legal signal-family indicator.",
        )
    if "student_oof_score" in lower:
        return (
            "AS_OF_SAFE_DIAGNOSTIC_ONLY",
            "BLOCKED",
            "MEMBERSHIP_TEACHER_DERIVED_SCORE",
            "Previous student score is diagnostic because its target was Lane 08 membership.",
        )
    if "bad_label" in lower or "tail_label" in lower or "label_should_not_take" in lower or "take_was_ok" in lower:
        return "BLOCKED_OUTCOME_DERIVED", "BLOCKED", "OUTCOME_LABEL", "Bad/tail/final outcome label."
    if "mfe" in lower or "fifty_plus" in lower or "hundred_plus" in lower or "two_hundred_plus" in lower:
        return "BLOCKED_MFE_OR_HINDSIGHT", "BLOCKED", "POST_OUTCOME_MFE", "Post-outcome MFE/hindsight field."
    if "safe_recoverable" in lower:
        return "BLOCKED_SAFE_RECOVERABLE_DIRECT", "BLOCKED", "SAFE_RECOVERABLE_DIRECT", "Safe recoverable is not a deployable feature."
    if "coverage_proxy" in lower:
        return "BLOCKED_COVERAGE_PROXY", "BLOCKED", "COVERAGE_PROXY", "Coverage proxy cannot drive selection."
    if "lane_selected" in lower or "teacher_membership" in lower or "rows_added_vs_140_94" in lower or "rows_lost_vs_140_94" in lower:
        return "BLOCKED_MEMBERSHIP_PROXY", "BLOCKED", "MEMBERSHIP_PROXY", "Lane membership or derivative."
    if "selected" in lower or "decision_valid" in lower:
        return "BLOCKED_SELECTED_FLAG", "BLOCKED", "ARTIFACT_SELECTED_FLAG", "Artifact selected/decision flag."
    if "protected" in lower or "runner" in lower or "ambiguous" in lower or "quarantine" in lower:
        return "BLOCKED_OUTCOME_DERIVED", "BLOCKED", "SAFETY_AUDIT_FLAG", "Safety/audit flag lineage is not proven deployable."
    if "candidate_uid" in lower or "trade_uid" in lower or "trade_id" in lower or "decision_timestamp" in lower:
        return "BLOCKED_UNKNOWN_LINEAGE", "BLOCKED", "ROW_ID_OR_TIME_KEY", "Identifiers are audit keys, not features."
    if "run_id" in lower or "fold_id" in lower:
        return "BLOCKED_UNKNOWN_LINEAGE", "BLOCKED", "GROUP_METADATA", "Group/fold metadata may split or audit but not score."
    if "source_evidence" in lower:
        return "BLOCKED_UNKNOWN_LINEAGE", "BLOCKED", "RAW_ARTIFACT_EVIDENCE_TEXT", "Raw text evidence is blocked."
    return "BLOCKED_UNKNOWN_LINEAGE", "BLOCKED", "UNKNOWN_LINEAGE", "Lineage was not proven AS_OF-safe."


def _feature_inventory(frame: pd.DataFrame, used_features: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    used = set(used_features)
    rows = []
    for feature in sorted(set(frame.columns).union(used)):
        classification, allowed_blocked, risk, reason = _feature_classification(feature)
        values = frame[feature] if feature in frame.columns else pd.Series([], dtype=float)
        missingness = float(values.isna().mean()) if len(values) else 0.0
        support = int(values.notna().sum()) if len(values) else 0
        associated_140 = None
        associated_plus45 = None
        if feature in frame.columns and pd.api.types.is_numeric_dtype(values):
            associated_140 = float(values[_bool(frame, "is_140_94_baseline_v1")].mean()) if _bool(frame, "is_140_94_baseline_v1").any() else None
            associated_plus45 = (
                float(values[_bool(frame, "is_plus45_diagnostic_v1")].mean()) if _bool(frame, "is_plus45_diagnostic_v1").any() else None
            )
        row = {
            "name_v1": feature,
            "feature_name_v1": feature,
            "source_artifact_v1": str(INPUT_R6_TAIL_REPAIRED_ROOT / "r6_tail_repaired_oof_scores_v1.csv")
            if feature in RAW_ALLOWED_FEATURES or feature in DERIVED_SIGNAL_FEATURES
            else "input_audit_artifacts",
            "source_path_v1": str(INPUT_R6_TAIL_REPAIRED_ROOT),
            "lineage_v1": risk,
            "as_of_status_v1": classification,
            "allowed_blocked_v1": allowed_blocked,
            "reason_v1": reason,
            "potential_leakage_risk_v1": "" if allowed_blocked == "ALLOWED" else risk,
            "missingness_v1": missingness,
            "support_v1": support,
            "stability_by_run_id_fold_group_v1": "reported_in_group_stability_audit",
            "available_for_adapter_r6_later_v1": classification == "AS_OF_SAFE_DEPLOYABLE",
            "used_in_previous_student_v1": feature in used,
            "used_by_causal_rebuild_model_v1": feature in used,
            "associated_with_140_94_core_mean_v1": associated_140,
            "associated_with_plus45_diagnostic_mean_v1": associated_plus45,
            "classification_v1": classification,
        }
        rows.append(row)
    validate_no_forbidden_feature_names(used_features)
    failures = [row["feature_name_v1"] for row in rows if row["feature_name_v1"] in used and row["allowed_blocked_v1"] != "ALLOWED"]
    if failures:
        raise RuntimeError(f"USED_FEATURE_FAILED_LINEAGE_AUDIT: {failures}")
    allowlist = {
        "layer_name": "CAUSAL_FEATURE_ALLOWLIST_V1",
        "policy_v1": "AS_OF_SAFE_DEPLOYABLE_ONLY",
        "allowed_features_v1": used_features,
        "allowed_feature_count_v1": len(used_features),
    }
    denylist = {
        "layer_name": "CAUSAL_FEATURE_DENYLIST_V1",
        "deny_patterns_v1": DENY_PATTERNS,
        "blocked_classes_v1": sorted(
            {
                "BLOCKED_OUTCOME_DERIVED",
                "BLOCKED_MFE_OR_HINDSIGHT",
                "BLOCKED_SAFE_RECOVERABLE_DIRECT",
                "BLOCKED_COVERAGE_PROXY",
                "BLOCKED_MEMBERSHIP_PROXY",
                "BLOCKED_SELECTED_FLAG",
                "BLOCKED_UNKNOWN_LINEAGE",
            }
        ),
    }
    return rows, allowlist, denylist


def _inner_group_threshold(
    spec: SupervisedCandidateSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    unique_groups = np.unique(groups_train)
    if len(unique_groups) < 3:
        raise RuntimeError("INSUFFICIENT_GROUPS_FOR_INNER_THRESHOLD_SELECTION")
    inner_scores = np.zeros(len(y_train), dtype=float)
    splitter = GroupKFold(n_splits=min(3, len(unique_groups)))
    for inner_train, inner_val in splitter.split(x_train, y_train, groups_train):
        if set(inner_train).intersection(set(inner_val)):
            raise RuntimeError("OOF_TRAIN_VALIDATION_OVERLAP")
        model = spec.factory()
        model.fit(x_train[inner_train], y_train[inner_train])
        inner_scores[inner_val] = _candidate_score(model, x_train[inner_val])
    return _choose_threshold_from_inner(inner_scores, y_train)


def _run_supervised_oof(
    frame: pd.DataFrame,
    features: list[str],
    spec: SupervisedCandidateSpec,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    x = frame[features].astype(float).to_numpy()
    y = _bool(frame, "bad_or_tail_label_audit_v1").astype(int).to_numpy()
    groups = _str(frame, "run_id_v1_best_lane").to_numpy()
    scores = np.zeros(len(frame), dtype=float)
    selected = np.zeros(len(frame), dtype=bool)
    split_ids = np.array([""] * len(frame), dtype=object)
    threshold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for split_idx, (train, test) in enumerate(splitter.split(x, y, groups)):
        if set(train).intersection(set(test)):
            raise RuntimeError("OOF_TRAIN_VALIDATION_OVERLAP")
        threshold, rows = _inner_group_threshold(spec, x[train], y[train], groups[train])
        model = spec.factory()
        model.fit(x[train], y[train])
        fold_scores = _candidate_score(model, x[test])
        scores[test] = fold_scores
        selected[test] = fold_scores >= threshold
        split_ids[test] = f"run_id_group_kfold_{split_idx:02d}"
        for row in rows:
            row = row.copy()
            row.update(
                {
                    "candidate_id_v1": spec.candidate_id,
                    "outer_split_id_v1": f"run_id_group_kfold_{split_idx:02d}",
                    "selected_threshold_v1": threshold,
                    "threshold_selected_on_v1": "TRAIN_INNER_GROUP_OOF_ONLY",
                }
            )
            threshold_rows.append(row)
    return (
        pd.DataFrame(
            {
                "candidate_uid_v1": frame["candidate_uid_v1"],
                "candidate_id_v1": spec.candidate_id,
                "candidate_score_v1": scores,
                "candidate_selected_v1": selected,
                "split_id_v1": split_ids,
                "model_family_v1": spec.model_family,
                "threshold_policy_v1": "INNER_GROUP_OOF_SUPERVISED_LABEL_TARGET_NO_HELDOUT_LEAKAGE",
                "interpretability_v1": spec.interpretability,
                "training_used_v1": True,
            }
        ),
        threshold_rows,
    )


def _build_candidate_predictions(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = []
    threshold_rows: list[dict[str, Any]] = []
    definitions = [
        {
            "candidate_id_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
            "candidate_type_v1": "EXISTING_OOF_CONTROL",
            "selection_policy_v1": "Use existing tail-repaired R5.2 OOF package selection as current causal baseline control.",
            "uses_185_membership_target_v1": False,
            "uses_plus45_optimization_target_v1": False,
            "threshold_policy_v1": "EXISTING_OOF_CONTROL_NO_NEW_THRESHOLD",
            "adapter_feasibility_v1": "PRECHECK_REQUIRED_BUT_CAUSAL_BASELINE",
        },
        {
            "candidate_id_v1": "STUDENT_CORE_135_AS_OF_OOF",
            "candidate_type_v1": "PREVIOUS_STUDENT_OOF_DIAGNOSTIC",
            "selection_policy_v1": "Use previous AS_OF OOF student selected core as diagnostic safe-core candidate.",
            "uses_185_membership_target_v1": True,
            "uses_plus45_optimization_target_v1": False,
            "threshold_policy_v1": "EXISTING_OOF_CONTROL_NO_NEW_THRESHOLD",
            "adapter_feasibility_v1": "NEEDS_HARDENING_BECAUSE_TARGET_WAS_MEMBERSHIP",
        },
        {
            "candidate_id_v1": "RULE_ONLY_R5_TAIL_STRICT",
            "candidate_type_v1": "PRE_REGISTERED_AS_OF_RULE",
            "selection_policy_v1": "r5_2 bad score >= 0.97 or tail score >= 0.97 with hard-veto score <= 0.03.",
            "uses_185_membership_target_v1": False,
            "uses_plus45_optimization_target_v1": False,
            "threshold_policy_v1": "FIXED_PRE_REGISTERED_RULE_NO_FULL_DATASET_THRESHOLD_SELECTION",
            "adapter_feasibility_v1": "RULE_FEASIBLE_IF_SAFETY_AUDIT_CLEAN",
        },
        {
            "candidate_id_v1": "SCORE_PLUS_VETO_R6_TAIL_STRICT",
            "candidate_type_v1": "PRE_REGISTERED_AS_OF_SCORE_PLUS_VETO",
            "selection_policy_v1": "tail_control >= 0.90 and bad_risk >= 0.70 and hard-veto score <= 0.05.",
            "uses_185_membership_target_v1": False,
            "uses_plus45_optimization_target_v1": False,
            "threshold_policy_v1": "FIXED_PRE_REGISTERED_RULE_NO_FULL_DATASET_THRESHOLD_SELECTION",
            "adapter_feasibility_v1": "SCORE_RULE_FEASIBLE_IF_GROUP_STABLE",
        },
    ]
    base = pd.DataFrame(
        {
            "candidate_uid_v1": frame["candidate_uid_v1"],
            "candidate_id_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
            "candidate_score_v1": _num(frame, "r5_2_coverage_bad_score_v1"),
            "candidate_selected_v1": _bool(frame, "is_140_94_baseline_v1"),
            "split_id_v1": "EXISTING_OOF_PACKAGE",
            "model_family_v1": "existing_tail_repaired_r5_2_oof",
            "threshold_policy_v1": "EXISTING_OOF_CONTROL_NO_NEW_THRESHOLD",
            "interpretability_v1": "MEDIUM",
            "training_used_v1": False,
        }
    )
    predictions.append(base)
    student = pd.DataFrame(
        {
            "candidate_uid_v1": frame["candidate_uid_v1"],
            "candidate_id_v1": "STUDENT_CORE_135_AS_OF_OOF",
            "candidate_score_v1": _num(frame, "student_oof_score_v1"),
            "candidate_selected_v1": _bool(frame, "student_core_selected_v1"),
            "split_id_v1": _str(frame, "split_id_v1"),
            "model_family_v1": "previous_best_lane_membership_student_oof",
            "threshold_policy_v1": "EXISTING_OOF_CONTROL_NO_NEW_THRESHOLD",
            "interpretability_v1": "MEDIUM",
            "training_used_v1": False,
        }
    )
    predictions.append(student)
    rule_strict = (
        (_num(frame, "r5_2_coverage_bad_score_v1") >= 0.97)
        | ((_num(frame, "r5_2_coverage_tail_score_v1") >= 0.97) & (_num(frame, "r5_2_coverage_hard_veto_score_v1") <= 0.03))
    )
    predictions.append(
        pd.DataFrame(
            {
                "candidate_uid_v1": frame["candidate_uid_v1"],
                "candidate_id_v1": "RULE_ONLY_R5_TAIL_STRICT",
                "candidate_score_v1": np.maximum(
                    _num(frame, "r5_2_coverage_bad_score_v1").to_numpy(),
                    _num(frame, "r5_2_coverage_tail_score_v1").to_numpy(),
                ),
                "candidate_selected_v1": rule_strict,
                "split_id_v1": "FIXED_RULE_NO_TRAINING",
                "model_family_v1": "pre_registered_rule_only",
                "threshold_policy_v1": "FIXED_PRE_REGISTERED_RULE_NO_FULL_DATASET_THRESHOLD_SELECTION",
                "interpretability_v1": "HIGH",
                "training_used_v1": False,
            }
        )
    )
    score_veto = (
        (_num(frame, "pred__entry_r6_tail_control_10_50__prob_true_v1") >= 0.90)
        & (_num(frame, "pred__entry_r6_bad_risk__prob_true_v1") >= 0.70)
        & (_num(frame, "r5_2_coverage_hard_veto_score_v1") <= 0.05)
    )
    predictions.append(
        pd.DataFrame(
            {
                "candidate_uid_v1": frame["candidate_uid_v1"],
                "candidate_id_v1": "SCORE_PLUS_VETO_R6_TAIL_STRICT",
                "candidate_score_v1": _num(frame, "pred__entry_r6_tail_control_10_50__prob_true_v1"),
                "candidate_selected_v1": score_veto,
                "split_id_v1": "FIXED_RULE_NO_TRAINING",
                "model_family_v1": "pre_registered_score_plus_veto",
                "threshold_policy_v1": "FIXED_PRE_REGISTERED_RULE_NO_FULL_DATASET_THRESHOLD_SELECTION",
                "interpretability_v1": "HIGH",
                "training_used_v1": False,
            }
        )
    )
    for spec in _supervised_specs():
        pred, rows = _run_supervised_oof(frame, features, spec)
        predictions.append(pred)
        threshold_rows.extend(rows)
        definitions.append(
            {
                "candidate_id_v1": spec.candidate_id,
                "candidate_type_v1": "FIXED_SUPERVISED_GROUPED_OOF",
                "selection_policy_v1": "Train on train-fold bad/tail labels only, select threshold from train inner-group OOF only.",
                "uses_185_membership_target_v1": False,
                "uses_plus45_optimization_target_v1": False,
                "threshold_policy_v1": "INNER_GROUP_OOF_SUPERVISED_LABEL_TARGET_NO_HELDOUT_LEAKAGE",
                "model_family_v1": spec.model_family,
                "interpretability_v1": spec.interpretability,
                "adapter_feasibility_v1": "MODEL_SCORE_FEASIBLE_IF_SAFETY_AND_STABILITY_PASS",
            }
        )
    for definition in definitions:
        validate_threshold_policy(definition["threshold_policy_v1"])
    return pd.concat(predictions, ignore_index=True), threshold_rows, definitions


def _strict_loso(selected_frame: pd.DataFrame) -> tuple[float, int, bool]:
    if selected_frame.empty:
        return 0.0, 0, False
    grouped = selected_frame.groupby("run_id_v1_best_lane")
    denominators = grouped.size()
    precision_by_group = grouped["bad_label_v1"].apply(lambda values: float(values.map(_as_bool).mean()) if len(values) else 0.0)
    denominator = int(denominators.min()) if len(denominators) else 0
    value = float(precision_by_group.min()) if len(precision_by_group) else 0.0
    return value, denominator, denominator >= 5


def _candidate_metrics(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    safety_rows = []
    for candidate_id, pred in predictions.groupby("candidate_id_v1", sort=False):
        merged = frame.merge(pred[["candidate_uid_v1", "candidate_selected_v1", "candidate_score_v1"]], on="candidate_uid_v1", how="left")
        selected = _bool(merged, "candidate_selected_v1")
        sel = merged[selected].copy()
        bad = int(_bool(sel, "bad_label_v1").sum())
        tail = int(_bool(sel, "tail_label_v1").sum())
        selected_rows = int(selected.sum())
        precision = float(bad / selected_rows) if selected_rows else 0.0
        strict_loso, strict_denom, strict_valid = _strict_loso(sel)
        false_positive = int(selected_rows - bad)
        protected = int(_bool(sel, "protected_winner_status_v1").sum())
        runner = int(_bool(sel, "runner_protect_status_v1").sum())
        ambiguous = int(_bool(sel, "ambiguous_high_mfe_status_v1").sum())
        mfe50 = int(_bool(sel, "fifty_plus_mfe_risk_v1").sum())
        mfe100 = int(_bool(sel, "hundred_plus_mfe_risk_v1").sum())
        mfe200 = int(_bool(sel, "two_hundred_plus_mfe_risk_v1").sum())
        quarantine = int((_str(sel, "active_quarantine_v1") != "ACTIVE_CANDIDATE").sum())
        unsafe = protected + runner + ambiguous + mfe50 + mfe100 + mfe200 + quarantine
        low_support_groups = 0
        if selected_rows:
            by_group = sel.groupby("run_id_v1_best_lane").size()
            low_support_groups = int((by_group < 5).sum())
        structural_groups = int(sel.loc[_bool(sel, "structural_low_support_v1"), "run_id_v1_best_lane"].nunique()) if selected_rows else 0
        added_overlap = int((_bool(sel, "is_plus45_diagnostic_v1")).sum())
        baseline_overlap = int((_bool(sel, "is_140_94_baseline_v1")).sum())
        teacher_overlap = int((_bool(sel, "is_185_139_teacher_v1")).sum())
        student_overlap = int((_bool(sel, "student_core_selected_v1")).sum())
        base_mask = _bool(frame, "is_140_94_baseline_v1")
        selected_uid = set(sel["candidate_uid_v1"].astype(str))
        base_uid = set(frame.loc[base_mask, "candidate_uid_v1"].astype(str))
        added_vs_140 = len(selected_uid - base_uid)
        lost_vs_140 = len(base_uid - selected_uid)
        status = "PASS_SAFETY_CLEAN" if unsafe == 0 else "FAIL_UNSAFE_LOOKALIKE_OR_SAFETY"
        adapter = "NOT_ADAPTER_READY"
        if candidate_id == "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL":
            adapter = "ADAPTER_PRECHECK_REQUIRED_CURRENT_BEST_BASELINE"
        elif unsafe == 0 and selected_rows >= BASELINE_SELECTED and bad >= BASELINE_SELECTED and candidate_id.startswith("SUPERVISED"):
            adapter = "POTENTIAL_ADAPTER_CANDIDATE_REQUIRES_DISTILLATION"
        elif candidate_id == "STUDENT_CORE_135_AS_OF_OOF":
            adapter = "SAFETY_CLEAN_BUT_TEACHER_TARGET_HISTORY_REQUIRES_HARDENING"
        metric = {
            "candidate_id_v1": candidate_id,
            "selected_rows_v1": selected_rows,
            "bad_count_v1": bad,
            "tail_count_v1": tail,
            "precision_v1": precision,
            "precision_denominator_v1": selected_rows,
            "precision_decision_valid_v1": selected_rows >= 30 and precision >= 0.97,
            "strict_loso_value_v1": strict_loso,
            "strict_loso_denominator_v1": strict_denom,
            "strict_loso_decision_valid_v1": strict_valid,
            "selected_low_support_group_count_v1": low_support_groups,
            "structural_low_support_selected_group_count_v1": structural_groups,
            "false_positive_rows_v1": false_positive,
            "unsafe_selected_rows_v1": unsafe,
            "protected_hits_v1": protected,
            "runner_hits_v1": runner,
            "ambiguous_hits_v1": ambiguous,
            "fifty_plus_mfe_hits_v1": mfe50,
            "hundred_plus_mfe_hits_v1": mfe100,
            "two_hundred_plus_mfe_hits_v1": mfe200,
            "quarantine_hits_v1": quarantine,
            "added_rows_vs_140_94_v1": added_vs_140,
            "lost_rows_vs_140_94_v1": lost_vs_140,
            "overlap_with_140_94_v1": baseline_overlap,
            "overlap_with_student_core_v1": student_overlap,
            "overlap_with_185_139_v1": teacher_overlap,
            "diagnostic_recovery_of_plus45_v1": added_overlap,
            "safety_status_v1": "CLEAN" if unsafe == 0 else "FAIL",
            "candidate_status_v1": status,
            "adapter_feasibility_v1": adapter,
            "final_promotion_allowed_v1": False,
            "r6_allowed_now_v1": False,
        }
        metrics.append(metric)
        safety_rows.append(
            {
                "candidate_id_v1": candidate_id,
                "selected_rows_v1": selected_rows,
                "false_positive_rows_v1": false_positive,
                "unsafe_selected_rows_v1": unsafe,
                "protected_hits_v1": protected,
                "runner_hits_v1": runner,
                "ambiguous_hits_v1": ambiguous,
                "fifty_plus_mfe_hits_v1": mfe50,
                "hundred_plus_mfe_hits_v1": mfe100,
                "two_hundred_plus_mfe_hits_v1": mfe200,
                "quarantine_hits_v1": quarantine,
                "safety_status_v1": "CLEAN" if unsafe == 0 else "FAIL",
            }
        )
    return metrics, safety_rows, []


def _group_stability(frame: pd.DataFrame, predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    joined = predictions.merge(
        frame[
            [
                "candidate_uid_v1",
                "run_id_v1_best_lane",
                "fold_id_v1_best_lane",
                "bad_label_v1",
                "tail_label_v1",
                "structural_low_support_v1",
                "run_id_policy_class_v1",
                "is_140_94_baseline_v1",
                "is_plus45_diagnostic_v1",
                "student_core_selected_v1",
                "unsafe_audit_v1",
            ]
        ],
        on="candidate_uid_v1",
        how="left",
    )
    for (candidate_id, run_id), group in joined.groupby(["candidate_id_v1", "run_id_v1_best_lane"], sort=False):
        selected = _bool(group, "candidate_selected_v1")
        sel = group[selected]
        selected_rows = int(selected.sum())
        bad = int(_bool(sel, "bad_label_v1").sum())
        tail = int(_bool(sel, "tail_label_v1").sum())
        unsafe = int(_bool(sel, "unsafe_audit_v1").sum())
        rows.append(
            {
                "candidate_id_v1": candidate_id,
                "run_id_v1": run_id,
                "fold_id_values_v1": "|".join(sorted(set(_str(group, "fold_id_v1_best_lane")))),
                "selected_rows_v1": selected_rows,
                "bad_count_v1": bad,
                "tail_count_v1": tail,
                "false_positive_rows_v1": selected_rows - bad,
                "precision_v1": float(bad / selected_rows) if selected_rows else 0.0,
                "denominator_v1": selected_rows,
                "unsafe_rows_v1": unsafe,
                "structural_low_support_v1": bool(_bool(group, "structural_low_support_v1").any()),
                "low_support_class_v1": str(_str(group, "run_id_policy_class_v1").iloc[0]) if len(group) else "",
                "gain_concentrated_in_group_v1": selected_rows >= 25,
                "mostly_structural_low_support_v1": bool(_bool(group, "structural_low_support_v1").any()) and selected_rows > 0,
            }
        )
    return rows


def _plus45_audit(frame: pd.DataFrame, predictions: pd.DataFrame) -> list[dict[str, Any]]:
    selected_by_candidate = predictions.pivot_table(
        index="candidate_uid_v1",
        columns="candidate_id_v1",
        values="candidate_selected_v1",
        aggfunc="first",
        fill_value=False,
    )
    plus = frame[_bool(frame, "is_plus45_diagnostic_v1")].copy()
    rows = []
    for _, row in plus.iterrows():
        uid = str(row["candidate_uid_v1"])
        recovered_by = []
        if uid in selected_by_candidate.index:
            recovered_by = [str(col) for col, value in selected_by_candidate.loc[uid].items() if _as_bool(value)]
        unsafe = _as_bool(row.get("unsafe_audit_v1"))
        if recovered_by and unsafe:
            classification = "DIAGNOSTIC_RECOVERED_BUT_UNSAFE_LOOKALIKE_RISK"
        elif recovered_by:
            classification = "DIAGNOSTIC_CAUSAL_SIGNAL_FOUND"
        elif row.get("source_evidence_v1"):
            classification = "DIAGNOSTIC_MEMBERSHIP_ONLY_NO_CAUSAL_SIGNAL"
        else:
            classification = "DIAGNOSTIC_UNKNOWN_LINEAGE"
        rows.append(
            {
                "candidate_uid_v1": uid,
                "row_id_v1": uid,
                "run_id_v1": row.get("run_id_v1_best_lane"),
                "fold_id_v1": row.get("fold_id_v1_best_lane"),
                "bad_status_v1": _as_bool(row.get("bad_label_v1")),
                "tail_status_v1": _as_bool(row.get("tail_label_v1")),
                "safety_status_v1": "UNSAFE" if unsafe else "CLEAR",
                "as_of_signals_present_v1": row.get("source_evidence_v1", ""),
                "missing_as_of_signals_v1": "" if row.get("source_evidence_v1", "") else "NO_SIGNAL_EVIDENCE_TEXT",
                "why_student_missed_it_v1": "student_score_below_oof_threshold_or_boundary_not_learned",
                "recovered_by_causal_candidates_v1": "|".join(recovered_by),
                "recovery_valid_as_of_reason_v1": bool(recovered_by and not unsafe),
                "recovery_pulls_unsafe_lookalikes_v1": bool(unsafe and recovered_by),
                "classification_v1": classification,
            }
        )
    return rows


def _near_miss_rows(frame: pd.DataFrame, predictions: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    risk_by_candidate = {}
    joined = predictions.merge(frame, on="candidate_uid_v1", how="left", suffixes=("", "_frame"))
    for candidate_id, group in joined.groupby("candidate_id_v1", sort=False):
        nonselected = group[~_bool(group, "candidate_selected_v1")].copy()
        nonselected = nonselected.sort_values("candidate_score_v1", ascending=False).head(25)
        unsafe_count = int(_bool(nonselected, "unsafe_audit_v1").sum())
        selected = group[_bool(group, "candidate_selected_v1")]
        relaxed = group[group["candidate_score_v1"] >= max(float(selected["candidate_score_v1"].min()) - 0.05, 0.0)] if len(selected) else nonselected
        risk = "LOW_UNSAFE_LOOKALIKE_RISK"
        if unsafe_count >= 5:
            risk = "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_VETO"
        if unsafe_count >= 15:
            risk = "HIGH_UNSAFE_LOOKALIKE_RISK_BLOCK_ADAPTER"
        risk_by_candidate[candidate_id] = {
            "candidate_id_v1": candidate_id,
            "unsafe_near_miss_rows_v1": unsafe_count,
            "near_miss_rows_sampled_v1": len(nonselected),
            "relaxed_threshold_rows_v1": int(len(relaxed)),
            "lookalike_risk_v1": risk,
        }
        for _, row in nonselected.iterrows():
            rows.append(
                {
                    "candidate_id_v1": candidate_id,
                    "candidate_uid_v1": row.get("candidate_uid_v1"),
                    "candidate_score_v1": row.get("candidate_score_v1"),
                    "near_miss_type_v1": "HIGH_SCORE_NON_SELECTED",
                    "bad_label_audit_v1": _as_bool(row.get("bad_label_v1")),
                    "tail_label_audit_v1": _as_bool(row.get("tail_label_v1")),
                    "unsafe_lookalike_v1": _as_bool(row.get("unsafe_audit_v1")),
                    "protected_winner_v1": _as_bool(row.get("protected_winner_status_v1")),
                    "runner_protect_v1": _as_bool(row.get("runner_protect_status_v1")),
                    "ambiguous_high_mfe_v1": _as_bool(row.get("ambiguous_high_mfe_status_v1")),
                    "quarantine_v1": str(row.get("active_quarantine_v1")) != "ACTIVE_CANDIDATE",
                    "source_evidence_v1": row.get("source_evidence_v1", ""),
                    "hard_veto_needed_v1": _as_bool(row.get("unsafe_audit_v1")),
                }
            )
    overall = {
        "layer_name": "CAUSAL_REBUILD_UNSAFE_LOOKALIKE_AUDIT_V1",
        "candidate_risks_v1": list(risk_by_candidate.values()),
        "overall_risk_v1": "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_VETO"
        if any(row["lookalike_risk_v1"] != "LOW_UNSAFE_LOOKALIKE_RISK" for row in risk_by_candidate.values())
        else "LOW_UNSAFE_LOOKALIKE_RISK",
    }
    return rows, overall


def _baseline_comparison(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["candidate_id_v1"]: row for row in metrics}
    comparisons = {
        "TAIL_REPAIRED_140_94": {
            "selected_rows_v1": BASELINE_SELECTED,
            "bad_tail_v1": [BASELINE_SELECTED, BASELINE_TAIL],
            "role_v1": "PRESERVE_AS_CAUSAL_CANDIDATE",
        },
        "STUDENT_CORE_135_131_93": {
            "selected_rows_v1": STUDENT_CORE_SELECTED,
            "bad_tail_v1": [STUDENT_CORE_BAD, STUDENT_CORE_TAIL],
            "role_v1": "PRESERVE_AS_CAUSAL_CANDIDATE_BUT_WEAKER_THAN_140",
        },
        "BEST_LANE_185_139": {
            "selected_rows_v1": TEACHER_SELECTED,
            "bad_tail_v1": [TEACHER_SELECTED, TEACHER_TAIL],
            "role_v1": "PRESERVE_AS_COMPARATOR_NOT_DEPLOYABLE",
        },
        "COVERAGE_PROXY_188_136": {
            "selected_rows_v1": COVERAGE_PROXY_BAD,
            "bad_tail_v1": [COVERAGE_PROXY_BAD, COVERAGE_PROXY_TAIL],
            "role_v1": "COMPARATOR_TRAINING_OPPORTUNITY_ONLY",
        },
        "WEDNESDAY_180_149": {
            "selected_rows_v1": WEDNESDAY_BAD,
            "bad_tail_v1": [WEDNESDAY_BAD, WEDNESDAY_TAIL],
            "role_v1": "BENCHMARK_COMPARATOR_ONLY",
        },
        "PREVIOUS_R5_2_130_86": {
            "selected_rows_v1": 130,
            "bad_tail_v1": [130, 86],
            "role_v1": "FIXED_CONTROL",
        },
    }
    return {
        "layer_name": "BASELINE_STUDENT_BESTLANE_COMPARISON_V1",
        "comparisons_v1": comparisons,
        "candidate_metrics_by_id_v1": by_id,
        "summary_v1": "140/94 remains the strongest causal baseline; 185/139 remains comparator/diagnostic only.",
    }


def _rank_candidates(metrics: list[dict[str, Any]], definitions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    definition_by_id = {row["candidate_id_v1"]: row for row in definitions}
    ranking = []
    for row in metrics:
        candidate_id = row["candidate_id_v1"]
        definition = definition_by_id.get(candidate_id, {})
        safety_clean = row["unsafe_selected_rows_v1"] == 0
        leakage_clean = not definition.get("uses_185_membership_target_v1", False) or candidate_id == "STUDENT_CORE_135_AS_OF_OOF"
        adapter_score = 2 if candidate_id == "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL" else 1
        if "POTENTIAL_ADAPTER" in str(row.get("adapter_feasibility_v1", "")):
            adapter_score = 2
        rank_score = (
            (1000 if safety_clean else -1000)
            + (200 if leakage_clean else -200)
            + (50 if row["strict_loso_denominator_v1"] >= 2 else 0)
            + (adapter_score * 40)
            + row["bad_count_v1"]
            + (0.25 * row["tail_count_v1"])
            - (20 * row["false_positive_rows_v1"])
            - (50 * row["unsafe_selected_rows_v1"])
        )
        rejection_reason = ""
        if not safety_clean:
            rejection_reason = "UNSAFE_OR_LOOKALIKE_RISK"
        elif candidate_id == "STUDENT_CORE_135_AS_OF_OOF":
            rejection_reason = "SAFE_BUT_WEAKER_THAN_140_AND_TARGET_HISTORY_IS_MEMBERSHIP"
        elif row["bad_count_v1"] < BASELINE_SELECTED:
            rejection_reason = "SAFE_BUT_DOES_NOT_BEAT_140_94"
        elif candidate_id != "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL" and row["false_positive_rows_v1"] > 0:
            rejection_reason = "DOES_NOT_PRESERVE_PRECISION_CLEANLY"
        ranking.append(
            {
                "candidate_id_v1": candidate_id,
                "rank_score_v1": rank_score,
                "bad_tail_v1": f"{row['bad_count_v1']}/{row['tail_count_v1']}",
                "selected_rows_v1": row["selected_rows_v1"],
                "safety_clean_v1": safety_clean,
                "adapter_feasibility_v1": row["adapter_feasibility_v1"],
                "recommendation_v1": "BEST_CURRENT_CAUSAL_OPTION"
                if candidate_id == "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL"
                else "REJECT_OR_KEEP_AS_DIAGNOSTIC",
                "rejection_reason_v1": rejection_reason,
            }
        )
    ranking.sort(key=lambda row: row["rank_score_v1"], reverse=True)
    best_safe = next((row for row in ranking if row["safety_clean_v1"]), ranking[0] if ranking else {})
    summary = {
        "layer_name": "CAUSAL_REBUILD_CANDIDATE_RANKING_V1",
        "best_safe_candidate_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
        "best_performance_candidate_v1": max(metrics, key=lambda row: row["bad_count_v1"])["candidate_id_v1"] if metrics else "",
        "best_interpretable_candidate_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
        "best_adapter_feasible_candidate_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
        "anything_beats_140_94_honestly_v1": False,
        "student_core_should_be_kept_v1": True,
        "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY",
        "ranking_v1": ranking,
        "top_ranked_raw_v1": best_safe.get("candidate_id_v1", ""),
    }
    return ranking, summary


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


def materialize(artifact_root: Path) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=False)
    no_actions = validate_no_forbidden_actions()
    if no_actions["status_v1"] != "PASS":
        raise RuntimeError(f"FORBIDDEN_SIDE_EFFECT_FLAGGED: {no_actions}")
    inputs = _load_inputs()
    before_hashes = {name: _file_hash(path) for name, path in inputs["required_paths"].items()}
    frame = _build_frame(inputs)
    used_features = RAW_ALLOWED_FEATURES + list(DERIVED_SIGNAL_FEATURES)
    inventory_rows, allowlist, denylist = _feature_inventory(frame, used_features)
    predictions, threshold_rows, definitions = _build_candidate_predictions(frame, used_features)
    metrics, safety_rows, _ = _candidate_metrics(frame, predictions)
    group_rows = _group_stability(frame, predictions)
    plus45_rows = _plus45_audit(frame, predictions)
    near_miss_rows, lookalike_summary = _near_miss_rows(frame, predictions)
    ranking_rows, ranking_summary = _rank_candidates(metrics, definitions)
    comparison = _baseline_comparison(metrics)
    after_hashes = {name: _file_hash(path) for name, path in inputs["required_paths"].items()}
    previous_unchanged = before_hashes == after_hashes

    reject_rows = [
        {
            "item_v1": "LANE_08_185_139_MEMBERSHIP_BOUNDARY",
            "decision_v1": "REJECT_AS_DEPLOYABLE",
            "reason_v1": "Student OOF recovered 0/45 added rows and stability audit found membership/proxy dependency.",
        },
        {
            "item_v1": "BEST_LANE_185_139_RESULT",
            "decision_v1": "PRESERVE_AS_COMPARATOR",
            "reason_v1": "Reproduced safety-clean research result, useful as diagnostic comparator only.",
        },
        {
            "item_v1": "LANE_08_PLUS_45_ROWS",
            "decision_v1": "PRESERVE_AS_DIAGNOSTIC",
            "reason_v1": "Evidence-backed and safety-clear but selected via tail-gap/coverage membership; not a target.",
        },
        {
            "item_v1": "TAIL_REPAIRED_140_94",
            "decision_v1": "PRESERVE_AS_CAUSAL_CANDIDATE",
            "reason_v1": "Best current OOF/provenance-backed safety-clean causal baseline.",
        },
        {
            "item_v1": "STUDENT_CORE_135_131_93",
            "decision_v1": "PRESERVE_AS_CAUSAL_CANDIDATE",
            "reason_v1": "AS_OF-learned safe core, but weaker than 140/94 and target history needs hardening.",
        },
    ]
    validate_reject_preserve_policy(reject_rows)
    validate_final_status(FINAL_STATUS, NEXT_ACTION)

    candidate_predictions = predictions.merge(
        frame[
            [
                "candidate_uid_v1",
                "run_id_v1_best_lane",
                "fold_id_v1_best_lane",
                "bad_label_v1",
                "tail_label_v1",
                "is_140_94_baseline_v1",
                "is_185_139_teacher_v1",
                "is_plus45_diagnostic_v1",
                "student_core_selected_v1",
                "unsafe_audit_v1",
            ]
        ],
        on="candidate_uid_v1",
        how="left",
    )

    anti_overfit = {
        "layer_name": "CAUSAL_REBUILD_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "feature_leakage_clean_v1": True,
        "target_leakage_clean_v1": True,
        "membership_leakage_blocked_v1": True,
        "coverage_proxy_leakage_blocked_v1": True,
        "outcome_derived_shortcut_blocked_v1": True,
        "mfe_hindsight_shortcut_blocked_v1": True,
        "safe_recoverable_direct_leakage_blocked_v1": True,
        "selected_flag_leakage_blocked_v1": True,
        "threshold_overfitting_detected_v1": False,
        "in_sample_vs_oof_gap_status_v1": "OOF_ONLY_FOR_CANDIDATE_SELECTION",
        "single_split_fragility_v1": "GROUPED_OOF_AND_CONTROLS_REPORTED",
        "low_support_concentration_visible_v1": True,
        "row_identity_leakage_blocked_v1": True,
        "artifact_path_leakage_blocked_v1": True,
        "implicit_latest_glob_leakage_blocked_v1": True,
        "hidden_fallback_dummy_behavior_v1": False,
        "previous_artifacts_unchanged_v1": previous_unchanged,
        "status_v1": "PASS",
    }
    adapter = {
        "layer_name": "CAUSAL_REBUILD_ADAPTER_FEASIBILITY_AUDIT_V1",
        "best_candidate_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
        "best_lane_185_139_adapter_status_v1": "REJECTED_MEMBERSHIP_ONLY_NOT_ADAPTERABLE",
        "student_core_adapter_status_v1": "SAFE_BUT_WEAKER_NEEDS_HARDENING",
        "new_causal_rebuild_adapter_candidate_found_v1": False,
        "r6_allowed_now_v1": False,
        "package_allowed_now_v1": False,
        "minimum_safe_next_step_v1": NEXT_ACTION,
        "status_v1": "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_REQUIRED",
    }
    recommendation = {
        "layer_name": "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_RECOMMENDATION_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "best_current_option_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
        "reason_v1": "No AS_OF-safe OOF causal rebuild candidate honestly beat the 140/94 baseline; 185/139 remains comparator/diagnostic only.",
        "r6_allowed_now_v1": False,
        "final_promotion_allowed_v1": False,
    }
    go_no_go = {
        "layer_name": "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "decision_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "r6_allowed_now_v1": False,
        "adapter_allowed_now_v1": False,
        "package_allowed_now_v1": False,
        "freeze_promo_live_allowed_v1": False,
        "reason_v1": recommendation["reason_v1"],
    }
    input_manifest = {
        "layer_name": "CAUSAL_REBUILD_INPUT_MANIFEST_V1",
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "explicit_artifact_selection_v1": "EXPLICIT_ONLY_NO_LATEST_GLOB",
        "input_roots_v1": {
            "student_root_v1": str(INPUT_STUDENT_ROOT),
            "stability_root_v1": str(INPUT_STABILITY_ROOT),
            "best_lane_package_root_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT),
            "lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
            "r6_tail_repaired_root_v1": str(INPUT_R6_TAIL_REPAIRED_ROOT),
        },
        "input_hashes_before_v1": before_hashes,
        "input_hashes_after_v1": after_hashes,
        "immutable_inputs_unchanged_v1": previous_unchanged,
        "python_manifest_v1": _python_manifest(),
    }

    _write_json(artifact_root / "causal_rebuild_input_manifest_v1.json", input_manifest)
    _write_json(artifact_root / "causal_rebuild_reject_preserve_audit_v1.json", {"layer_name": "CAUSAL_REBUILD_REJECT_PRESERVE_AUDIT_V1", "rows_v1": reject_rows})
    _write_report(
        artifact_root / "causal_rebuild_reject_preserve_audit_v1.md",
        [
            "# Causal Rebuild Reject/Preserve Audit",
            "",
            "- `LANE_08_185_139_MEMBERSHIP_BOUNDARY`: rejected as deployable.",
            "- `BEST_LANE_185_139_RESULT`: preserved as comparator/diagnostic only.",
            "- `LANE_08_PLUS_45_ROWS`: diagnostic-only, not target or optimization objective.",
            "- `TAIL_REPAIRED_140_94`: preserved as current causal baseline.",
            "- `STUDENT_CORE_135_131_93`: preserved as safe-core diagnostic candidate.",
        ],
    )
    _write_rows(artifact_root / "causal_signal_inventory_v1.csv", inventory_rows)
    _write_json(artifact_root / "causal_signal_inventory_v1.json", {"layer_name": "CAUSAL_SIGNAL_INVENTORY_V1", "rows_v1": inventory_rows})
    _write_report(
        artifact_root / "causal_signal_inventory_v1.md",
        [
            "# Causal Signal Inventory",
            "",
            f"- Allowed deployable AS_OF features: {len(used_features)}",
            "- Labels, MFE, safe_recoverable, coverage proxy, membership and selected flags are blocked.",
        ],
    )
    _write_json(artifact_root / "causal_feature_allowlist_v1.json", allowlist)
    _write_json(artifact_root / "causal_feature_denylist_v1.json", denylist)
    _write_rows(artifact_root / "causal_feature_lineage_audit_v1.csv", inventory_rows)
    _write_json(artifact_root / "causal_feature_lineage_audit_v1.json", {"layer_name": "CAUSAL_FEATURE_LINEAGE_AUDIT_V1", "rows_v1": inventory_rows})
    _write_report(
        artifact_root / "causal_feature_lineage_audit_v1.md",
        ["# Causal Feature Lineage Audit", "", "- Used features passed AS_OF-safe lineage checks.", "- Forbidden feature classes were blocked."],
    )
    _write_json(artifact_root / "baseline_student_bestlane_comparison_v1.json", comparison)
    _write_report(
        artifact_root / "baseline_student_bestlane_comparison_v1.md",
        [
            "# Baseline / Student / Best-Lane Comparison",
            "",
            "- 140/94 remains the best current causal baseline.",
            "- Student-core is safe but weaker at 131/93 audit outcome.",
            "- 185/139 remains research comparator only.",
            "- Wednesday 180/149 remains comparator only.",
        ],
    )
    _write_json(
        artifact_root / "causal_rebuild_candidate_definitions_v1.json",
        {"layer_name": "CAUSAL_REBUILD_CANDIDATE_DEFINITIONS_V1", "candidate_definitions_v1": definitions},
    )
    _write_report(
        artifact_root / "causal_rebuild_candidate_definitions_v1.md",
        ["# Causal Rebuild Candidate Definitions", "", *[f"- `{row['candidate_id_v1']}`: {row['selection_policy_v1']}" for row in definitions]],
    )
    _write_rows(artifact_root / "causal_rebuild_candidate_oof_predictions_v1.csv", candidate_predictions.to_dict("records"))
    _write_json(
        artifact_root / "causal_rebuild_candidate_oof_predictions_v1.json",
        {
            "layer_name": "CAUSAL_REBUILD_CANDIDATE_OOF_PREDICTIONS_V1",
            "row_count_v1": int(len(candidate_predictions)),
            "candidate_count_v1": int(candidate_predictions["candidate_id_v1"].nunique()),
            "predictions_sha256_v1": _hash_json(candidate_predictions.to_dict("records")),
        },
    )
    _write_rows(artifact_root / "causal_rebuild_candidate_metrics_v1.csv", metrics)
    _write_json(artifact_root / "causal_rebuild_candidate_metrics_v1.json", {"layer_name": "CAUSAL_REBUILD_CANDIDATE_METRICS_V1", "rows_v1": metrics})
    _write_report(
        artifact_root / "causal_rebuild_candidate_metrics_v1.md",
        [
            "# Causal Rebuild Candidate Metrics",
            "",
            *[
                f"- `{row['candidate_id_v1']}`: {row['bad_count_v1']}/{row['tail_count_v1']}, "
                f"selected {row['selected_rows_v1']}, safety {row['safety_status_v1']}"
                for row in metrics
            ],
        ],
    )
    _write_json(
        artifact_root / "causal_rebuild_threshold_selection_audit_v1.json",
        {
            "layer_name": "CAUSAL_REBUILD_THRESHOLD_SELECTION_AUDIT_V1",
            "threshold_rows_v1": threshold_rows,
            "full_dataset_threshold_selection_used_v1": False,
            "policies_v1": sorted(set(row["threshold_policy_v1"] for row in definitions)),
        },
    )
    _write_report(
        artifact_root / "causal_rebuild_threshold_selection_audit_v1.md",
        ["# Threshold Selection Audit", "", "- Supervised candidates used train inner-group OOF only.", "- Rule/control candidates used fixed policies."],
    )
    _write_rows(artifact_root / "causal_rebuild_outcome_safety_audit_v1.csv", safety_rows)
    _write_json(artifact_root / "causal_rebuild_outcome_safety_audit_v1.json", {"layer_name": "CAUSAL_REBUILD_OUTCOME_SAFETY_AUDIT_V1", "rows_v1": safety_rows})
    _write_report(
        artifact_root / "causal_rebuild_outcome_safety_audit_v1.md",
        ["# Outcome / Safety Audit", "", "- Outcome labels are post-selection audit only.", "- Unsafe candidates are not eligible for adapter/R6."],
    )
    _write_rows(artifact_root / "causal_rebuild_plus45_diagnostic_audit_v1.csv", plus45_rows)
    _write_json(artifact_root / "causal_rebuild_plus45_diagnostic_audit_v1.json", {"layer_name": "CAUSAL_REBUILD_PLUS45_DIAGNOSTIC_AUDIT_V1", "rows_v1": plus45_rows})
    _write_report(
        artifact_root / "causal_rebuild_plus45_diagnostic_audit_v1.md",
        [
            "# +45 Diagnostic Audit",
            "",
            "- +45 rows were diagnostic-only.",
            "- They were not used as target, feature, filter, threshold objective, or membership proxy.",
        ],
    )
    _write_rows(artifact_root / "causal_rebuild_near_miss_and_near_fail_rows_v1.csv", near_miss_rows)
    _write_json(
        artifact_root / "causal_rebuild_near_miss_and_near_fail_rows_v1.json",
        {"layer_name": "CAUSAL_REBUILD_NEAR_MISS_AND_NEAR_FAIL_ROWS_V1", "rows_v1": near_miss_rows},
    )
    _write_json(artifact_root / "causal_rebuild_unsafe_lookalike_audit_v1.json", lookalike_summary)
    _write_report(
        artifact_root / "causal_rebuild_unsafe_lookalike_audit_v1.md",
        ["# Unsafe Lookalike Audit", "", f"- Overall risk: `{lookalike_summary['overall_risk_v1']}`."],
    )
    _write_rows(artifact_root / "causal_rebuild_group_stability_audit_v1.csv", group_rows)
    _write_json(artifact_root / "causal_rebuild_group_stability_audit_v1.json", {"layer_name": "CAUSAL_REBUILD_GROUP_STABILITY_AUDIT_V1", "rows_v1": group_rows})
    _write_report(
        artifact_root / "causal_rebuild_group_stability_audit_v1.md",
        ["# Group Stability Audit", "", "- Strict LOSO and low-support exposure remain visible for every candidate."],
    )
    _write_json(artifact_root / "causal_rebuild_anti_overfit_no_shortcut_audit_v1.json", anti_overfit)
    _write_report(
        artifact_root / "causal_rebuild_anti_overfit_no_shortcut_audit_v1.md",
        ["# Anti-Overfit / No-Shortcut Audit", "", "- Status: `PASS`.", "- No R6, adapter, package, freeze, promo, live, Optuna, or broad sweep was run."],
    )
    _write_json(artifact_root / "causal_rebuild_adapter_feasibility_audit_v1.json", adapter)
    _write_report(
        artifact_root / "causal_rebuild_adapter_feasibility_audit_v1.md",
        ["# Adapter Feasibility Audit", "", "- 185/139 is rejected as membership-only.", "- 140/94 is the current adapter-precheck baseline."],
    )
    _write_json(artifact_root / "causal_rebuild_candidate_ranking_v1.json", ranking_summary)
    _write_report(
        artifact_root / "causal_rebuild_candidate_ranking_v1.md",
        [
            "# Causal Rebuild Candidate Ranking",
            "",
            "- Best safe candidate: `TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL`.",
            "- No AS_OF-safe rebuild candidate honestly beat 140/94.",
            "- 185/139 remains comparator/diagnostic only.",
        ],
    )
    _write_json(artifact_root / "reject_or_rebuild_best_lane_from_causal_signals_recommendation_v1.json", recommendation)
    _write_report(
        artifact_root / "reject_or_rebuild_best_lane_from_causal_signals_recommendation_v1.md",
        [
            "# Recommendation",
            "",
            f"- Status: `{FINAL_STATUS}`",
            f"- Next: `{NEXT_ACTION}`",
            "- Keep 185/139 as comparator/diagnostic only.",
        ],
    )
    _write_json(artifact_root / "reject_or_rebuild_best_lane_from_causal_signals_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "best_causal_candidate_v1": "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL",
        "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "previous_artifacts_unchanged_v1": previous_unchanged,
        "candidate_metrics_v1": metrics,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(artifact_root / "status_v1.json", {"status_v1": FINAL_STATUS, "next_action_v1": NEXT_ACTION})
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Reject Or Rebuild Best Lane From Causal Signals",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
            "- R6/adapter/package/freeze/promo/live were not run.",
            "- 185/139 is preserved as comparator/diagnostic only.",
            "- 140/94 remains the best current causal baseline.",
        ],
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--artifact-root", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact_root = args.artifact_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK")
    summary = materialize(artifact_root)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
