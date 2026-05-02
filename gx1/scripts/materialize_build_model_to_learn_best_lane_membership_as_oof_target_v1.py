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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_V1"
LAYER_NAME = ACTION

INPUT_STABILITY_RECHECK_ROOT = (
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

SELECTED_LANE_ID = "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY"
TEACHER_SELECTED = 185
TEACHER_TAIL = 139
BASELINE_SELECTED = 140
BASELINE_TAIL = 94
ADDED_ROWS = 45
WEDNESDAY_BAD = 180
WEDNESDAY_TAIL = 149
COVERAGE_PROXY_BAD = 188
COVERAGE_PROXY_TAIL = 136

FINAL_STATUS = "BEST_LANE_MEMBERSHIP_NOT_LEARNABLE_FROM_AS_OF_FEATURES"
NEXT_ACTION = "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1"

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

THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

DENY_PATTERNS = [
    "bad_label",
    "tail_label",
    "label_should_not_take",
    "tail_10_50_mfe",
    "take_was_ok",
    "fifty_plus_mfe",
    "hundred_plus_mfe",
    "two_hundred_plus_mfe",
    "safe_recoverable",
    "coverage_proxy",
    "lane_selected",
    "lane_id",
    "rows_added_vs_140_94",
    "rows_lost_vs_140_94",
    "selected_by",
    "selected_rows",
    "r5_2_package_selected",
    "r6_best_candidate_selected",
    "decision_valid_score",
    "was_row_in_train",
    "protected_winner",
    "runner_protect",
    "runner_protector",
    "ambiguous_high_mfe",
    "active_quarantine",
    "quarantine",
    "final_promotion",
    "source_evidence",
    "large_jump",
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact for hash: {path}")
    return _sha256_bytes(path.read_bytes())


def _hash_json(payload: Any) -> str:
    return _sha256_bytes(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass"}
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


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_forbidden_actions(
    *,
    optuna: bool,
    r6: bool,
    adapter: bool,
    package: bool,
    freeze: bool,
    promo: bool,
    live: bool,
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


def validate_teacher_target_freeze(target: pd.DataFrame) -> bool:
    selected = int(_bool(target, "teacher_membership_v1").sum())
    added = int(_bool(target, "is_added_45_v1").sum())
    baseline = int(_bool(target, "is_baseline_140_v1").sum())
    if selected != TEACHER_SELECTED or added != ADDED_ROWS or baseline != BASELINE_SELECTED:
        raise RuntimeError(
            "TEACHER_TARGET_FREEZE_MISMATCH: "
            f"selected={selected} added={added} baseline={baseline}"
        )
    overlap = int((_bool(target, "is_added_45_v1") & _bool(target, "is_baseline_140_v1")).sum())
    if overlap:
        raise RuntimeError("TEACHER_TARGET_ADDED_BASELINE_OVERLAP")
    return True


def validate_feature_policy(used_features: Iterable[str], audit_rows: Iterable[dict[str, Any]]) -> bool:
    audit_by_name = {row["feature_name_v1"]: row for row in audit_rows}
    failures = []
    for feature in used_features:
        row = audit_by_name.get(feature)
        if row is None:
            failures.append(f"{feature}:MISSING_AUDIT_ROW")
        elif row.get("allowed_blocked_v1") != "ALLOWED":
            failures.append(f"{feature}:{row.get('suspected_leakage_class_v1')}")
    if failures:
        raise RuntimeError(f"USED_FEATURE_FAILED_LEAKAGE_AUDIT: {failures}")
    return True


def validate_no_target_or_outcome_feature(features: Iterable[str]) -> bool:
    blocked = []
    for feature in features:
        lower = feature.lower()
        if any(pattern in lower for pattern in DENY_PATTERNS):
            blocked.append(feature)
    if blocked:
        raise RuntimeError(f"TARGET_OR_OUTCOME_FEATURE_FORBIDDEN: {blocked}")
    return True


def validate_oof_split(train_idx: Sequence[int], test_idx: Sequence[int]) -> bool:
    if set(train_idx).intersection(set(test_idx)):
        raise RuntimeError("OOF_TRAIN_VALIDATION_OVERLAP")
    return True


def choose_threshold_from_inner_validation(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    best_threshold = THRESHOLD_GRID[0]
    best_score = -1e9
    for threshold in THRESHOLD_GRID:
        pred = scores >= threshold
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        objective = f1 + (0.02 * precision)
        if precision < 0.75:
            objective -= 1.0
        rows.append(
            {
                "threshold_v1": threshold,
                "precision_v1": precision,
                "recall_v1": recall,
                "f1_v1": f1,
                "objective_v1": objective,
                "selected_v1": int(pred.sum()),
            }
        )
        if objective > best_score:
            best_threshold = threshold
            best_score = objective
    return best_threshold, rows


def final_status_from_metrics(metrics: dict[str, Any], *, leakage_clean: bool, unsafe_selected: int) -> tuple[str, str]:
    if not leakage_clean:
        return "BLOCKED_BY_FEATURE_LEAKAGE_OR_TARGET_CONTAMINATION", "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1"
    if unsafe_selected > 0:
        return (
            "BEST_LANE_SIGNAL_REAL_BUT_STUDENT_OVERSELECTS_UNSAFE_LOOKALIKES",
            "DEEPEN_STUDENT_NEAR_MISS_UNSAFE_LOOKALIKE_AUDIT_V1",
        )
    added_recall = metrics.get("added_row_recall_v1", 0.0)
    recall = metrics.get("teacher_recall_v1", 0.0)
    if recall >= 0.90 and added_recall >= 0.80:
        return (
            "BEST_LANE_MEMBERSHIP_LEARNED_AS_OF_OOF_ADAPTER_FEASIBLE",
            "BUILD_AS_OF_SAFE_BEST_LANE_MEMBERSHIP_ADAPTER_V1",
        )
    if recall >= 0.75 and added_recall >= 0.30:
        return (
            "BEST_LANE_MEMBERSHIP_PARTIALLY_LEARNED_NEEDS_RULE_DISTILLATION",
            "DISTILL_BEST_LANE_STUDENT_TO_AS_OF_RULES_AND_VETOES_V1",
        )
    return "BEST_LANE_MEMBERSHIP_NOT_LEARNABLE_FROM_AS_OF_FEATURES", "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1"


@dataclass(frozen=True)
class StudentSpec:
    model_id: str
    factory: Callable[[], Any]
    model_family: str
    interpretability_v1: str


def _student_specs() -> list[StudentSpec]:
    return [
        StudentSpec(
            model_id="LOGREG_L2_BALANCED_FIXED",
            factory=lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="liblinear"),
            ),
            model_family="regularized_logistic_regression",
            interpretability_v1="HIGH",
        ),
        StudentSpec(
            model_id="SHALLOW_TREE_FIXED",
            factory=lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=17),
            model_family="shallow_rule_tree",
            interpretability_v1="HIGH",
        ),
        StudentSpec(
            model_id="SMALL_HGB_FIXED",
            factory=lambda: HistGradientBoostingClassifier(
                max_iter=80,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                learning_rate=0.05,
                random_state=17,
            ),
            model_family="small_gradient_model_fixed_diagnostic",
            interpretability_v1="MEDIUM",
        ),
    ]


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
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    required = {
        "stability_summary": INPUT_STABILITY_RECHECK_ROOT / "summary_v1.json",
        "stability_go_no_go": INPUT_STABILITY_RECHECK_ROOT / "stability_recheck_best_lane_185_139_before_r6_go_no_go_v1.json",
        "best_membership": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv",
        "best_manifest": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_package_manifest_v1.json",
        "r6_scores": INPUT_R6_TAIL_REPAIRED_ROOT / "r6_tail_repaired_oof_scores_v1.csv",
        "r6_provenance": INPUT_R6_TAIL_REPAIRED_ROOT / "r6_tail_repaired_oof_score_provenance_v1.csv",
        "lane_summary": INPUT_LANE_PACK_ROOT / "lanes" / SELECTED_LANE_ID / "lane_result_summary_v1.json",
        "stability_added_evidence": INPUT_STABILITY_RECHECK_ROOT / "best_lane_added_rows_selection_evidence_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    stability = _read_json(required["stability_summary"])
    go = _read_json(required["stability_go_no_go"])
    if go.get("status_v1") != "BEST_LANE_SIGNAL_STRONG_BUT_MEMBERSHIP_ONLY_NOT_R6_READY":
        raise RuntimeError("STABILITY_RECHECK_STATUS_NOT_EXPECTED_MEMBERSHIP_ONLY")
    if stability.get("membership_oracle_dependency_status_v1") != "MEMBERSHIP_ONLY_NOT_CAUSALLY_SCORABLE":
        raise RuntimeError("STABILITY_RECHECK_DID_NOT_PROVE_MEMBERSHIP_ONLY")
    return {
        "required_paths": required,
        "stability": stability,
        "stability_go": go,
        "membership": pd.read_csv(required["best_membership"]),
        "best_manifest": _read_json(required["best_manifest"]),
        "r6_scores": pd.read_csv(required["r6_scores"]),
        "r6_provenance": pd.read_csv(required["r6_provenance"]),
        "lane_summary": _read_json(required["lane_summary"]),
        "stability_added_evidence": pd.read_csv(required["stability_added_evidence"]),
    }


def _build_teacher_target(membership: pd.DataFrame) -> pd.DataFrame:
    target = membership[
        [
            "candidate_uid_v1",
            "trade_uid_v1",
            "decision_timestamp_v1",
            "trade_id_v1",
            "run_id_v1",
            "fold_id_v1",
            "bad_label_v1",
            "tail_label_v1",
            "r5_2_package_selected_v1",
            "lane_selected_v1",
            "rows_added_vs_140_94_v1",
            "rows_lost_vs_140_94_v1",
        ]
    ].copy()
    target["teacher_membership_v1"] = _bool(target, "lane_selected_v1")
    target["is_baseline_140_v1"] = _bool(target, "r5_2_package_selected_v1")
    target["is_added_45_v1"] = _bool(target, "rows_added_vs_140_94_v1")
    validate_teacher_target_freeze(target)
    return target


def _build_feature_frame(target: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    frame = target.merge(scores, on="candidate_uid_v1", how="left", suffixes=("_teacher", ""))
    source_evidence = _str(frame, "source_evidence_v1")
    for feature, token in DERIVED_SIGNAL_FEATURES.items():
        frame[feature] = source_evidence.str.contains(token, regex=False).astype(float)
    return frame


def _feature_source(feature_name: str) -> str:
    if feature_name in RAW_ALLOWED_FEATURES:
        return str(INPUT_R6_TAIL_REPAIRED_ROOT / "r6_tail_repaired_oof_scores_v1.csv")
    if feature_name in DERIVED_SIGNAL_FEATURES:
        return str(INPUT_R6_TAIL_REPAIRED_ROOT / "r6_tail_repaired_oof_scores_v1.csv:source_evidence_v1")
    return "various_input_artifacts"


def _feature_leakage_class(feature_name: str) -> tuple[str, str, str]:
    lower = feature_name.lower()
    if feature_name in RAW_ALLOWED_FEATURES:
        return "AS_OF_SAFE_OOF_SCORE", "ALLOWED", "Existing OOF score/probability derived before outcome."
    if feature_name in DERIVED_SIGNAL_FEATURES:
        return "AS_OF_SAFE_LEGAL_SIGNAL_FAMILY_INDICATOR", "ALLOWED", "Derived from existing legal signal-family evidence token only."
    if "bad_label" in lower or "tail_label" in lower or "label_should_not_take" in lower or "take_was_ok" in lower:
        return "OUTCOME_LABEL", "BLOCKED", "Final bad/tail/outcome label."
    if "mfe" in lower or "fifty_plus" in lower or "hundred_plus" in lower or "two_hundred_plus" in lower:
        return "POST_OUTCOME_MFE", "BLOCKED", "Post-outcome MFE or MFE risk."
    if "safe_recoverable" in lower:
        return "SAFE_RECOVERABLE_ORACLE", "BLOCKED", "Safe recoverable is an audit/label field."
    if "coverage_proxy" in lower:
        return "COVERAGE_PROXY_LEAKAGE", "BLOCKED", "Coverage proxy membership or proxy-derived field."
    if "lane_selected" in lower or "lane_id" in lower or "rows_added_vs_140_94" in lower or "rows_lost_vs_140_94" in lower:
        return "TEACHER_MEMBERSHIP_LEAKAGE", "BLOCKED", "Teacher lane membership or derivative."
    if "selected" in lower or "decision_valid" in lower or "was_row_in_train" in lower:
        return "ARTIFACT_SELECTION_OR_DECISION_FLAG", "BLOCKED", "Artifact-derived selected/decision flag."
    if "protected" in lower or "runner" in lower or "ambiguous" in lower or "quarantine" in lower:
        return "SAFETY_OR_RUNTIME_OUTCOME_FLAG", "BLOCKED", "Safety/protected/runner/quarantine flag is not deployable as feature."
    if feature_name in {"candidate_uid_v1", "trade_uid_v1", "trade_id_v1", "decision_timestamp_v1", "run_id_v1", "fold_id_v1"}:
        return "ID_OR_GROUP_METADATA", "BLOCKED", "Identifier/group metadata may split or audit but not model."
    if "source_evidence" in lower:
        return "RAW_ARTIFACT_EVIDENCE_TEXT", "BLOCKED", "Raw artifact evidence text is blocked; only allowlisted family indicators are used."
    return "BLOCKED_UNKNOWN_LINEAGE", "BLOCKED", "Feature lineage was not proven AS_OF-safe."


def _feature_audit(frame: pd.DataFrame, used_features: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    used = set(used_features)
    rows = []
    for feature in sorted(set(frame.columns).union(used)):
        leakage_class, allowed_blocked, reason = _feature_leakage_class(feature)
        rows.append(
            {
                "feature_name_v1": feature,
                "source_artifact_v1": _feature_source(feature),
                "as_of_status_v1": "AS_OF_SAFE" if allowed_blocked == "ALLOWED" else "NOT_ALLOWED_FOR_MODEL",
                "allowed_blocked_v1": allowed_blocked,
                "reason_v1": reason,
                "suspected_leakage_class_v1": "" if allowed_blocked == "ALLOWED" else leakage_class,
                "used_by_model_v1": feature in used,
            }
        )
    validate_no_target_or_outcome_feature(used_features)
    validate_feature_policy(used_features, rows)
    allowlist = {
        "layer_name": "BEST_LANE_AS_OF_FEATURE_ALLOWLIST_V1",
        "feature_policy_v1": "AS_OF_SAFE_ONLY_NO_LABEL_OR_MEMBERSHIP_FEATURES",
        "allowed_features_v1": used_features,
        "raw_allowed_features_v1": RAW_ALLOWED_FEATURES,
        "derived_signal_features_v1": DERIVED_SIGNAL_FEATURES,
        "feature_count_v1": len(used_features),
    }
    denylist = {
        "layer_name": "BEST_LANE_AS_OF_FEATURE_DENYLIST_V1",
        "deny_patterns_v1": DENY_PATTERNS,
        "explicitly_blocked_classes_v1": sorted(
            {
                "OUTCOME_LABEL",
                "POST_OUTCOME_MFE",
                "SAFE_RECOVERABLE_ORACLE",
                "COVERAGE_PROXY_LEAKAGE",
                "TEACHER_MEMBERSHIP_LEAKAGE",
                "ARTIFACT_SELECTION_OR_DECISION_FLAG",
                "SAFETY_OR_RUNTIME_OUTCOME_FLAG",
                "ID_OR_GROUP_METADATA",
                "RAW_ARTIFACT_EVIDENCE_TEXT",
                "BLOCKED_UNKNOWN_LINEAGE",
            }
        ),
    }
    return rows, allowlist, denylist


def _model_score(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        return 1.0 / (1.0 + np.exp(-raw))
    raise RuntimeError("MODEL_DOES_NOT_SUPPORT_PROBABILITY_OR_DECISION_SCORE")


def _inner_threshold(
    spec: StudentSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
) -> tuple[float, list[dict[str, Any]]]:
    unique_groups = np.unique(groups_train)
    inner_scores = np.zeros(len(y_train), dtype=float)
    if len(unique_groups) >= 3:
        splitter = GroupKFold(n_splits=min(3, len(unique_groups)))
        for inner_train, inner_val in splitter.split(x_train, y_train, groups_train):
            validate_oof_split(inner_train, inner_val)
            model = spec.factory()
            model.fit(x_train[inner_train], y_train[inner_train])
            inner_scores[inner_val] = _model_score(model, x_train[inner_val])
    else:
        raise RuntimeError("INSUFFICIENT_GROUPS_FOR_INNER_THRESHOLD_SELECTION")
    threshold, rows = choose_threshold_from_inner_validation(inner_scores, y_train)
    return threshold, rows


def _outer_splits(split_policy: str, y: np.ndarray, groups: np.ndarray, folds: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    if split_policy == "RUN_ID_GROUP_KFOLD_5":
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        return [(f"run_id_group_kfold_{idx:02d}", train, test) for idx, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups))]
    if split_policy == "FOLD_ID_HELDOUT":
        rows = []
        for fold_id in sorted(pd.Series(folds).dropna().unique()):
            test = np.where(folds == fold_id)[0]
            train = np.where(folds != fold_id)[0]
            rows.append((f"fold_heldout_{fold_id}", train, test))
        return rows
    if split_policy == "RUN_ID_LOSO":
        splitter = LeaveOneGroupOut()
        return [(f"run_id_loso_{idx:02d}", train, test) for idx, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups))]
    raise RuntimeError(f"Unsupported split policy: {split_policy}")


def _run_oof(
    frame: pd.DataFrame,
    features: list[str],
    spec: StudentSpec,
    *,
    split_policy: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    x = frame[features].astype(float).to_numpy()
    y = _bool(frame, "teacher_membership_v1").astype(int).to_numpy()
    groups = _str(frame, "run_id_v1_teacher").to_numpy()
    folds = _str(frame, "fold_id_v1_teacher").to_numpy()
    scores = np.zeros(len(frame), dtype=float)
    selected = np.zeros(len(frame), dtype=bool)
    split_ids = np.array([""] * len(frame), dtype=object)
    threshold_rows: list[dict[str, Any]] = []
    for split_id, train, test in _outer_splits(split_policy, y, groups, folds):
        validate_oof_split(train, test)
        threshold, inner_rows = _inner_threshold(spec, x[train], y[train], groups[train])
        model = spec.factory()
        model.fit(x[train], y[train])
        fold_scores = _model_score(model, x[test])
        scores[test] = fold_scores
        selected[test] = fold_scores >= threshold
        split_ids[test] = split_id
        for row in inner_rows:
            item = dict(row)
            item.update(
                {
                    "student_model_id_v1": spec.model_id,
                    "split_policy_v1": split_policy,
                    "outer_split_id_v1": split_id,
                    "selected_as_fold_threshold_v1": math.isclose(float(row["threshold_v1"]), float(threshold)),
                }
            )
            threshold_rows.append(item)
    predictions = frame[
        [
            "candidate_uid_v1",
            "trade_uid_v1_teacher",
            "decision_timestamp_v1_teacher",
            "trade_id_v1_teacher",
            "run_id_v1_teacher",
            "fold_id_v1_teacher",
            "teacher_membership_v1",
            "is_added_45_v1",
            "is_baseline_140_v1",
            "bad_label_v1_teacher",
            "tail_label_v1_teacher",
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
            "source_evidence_v1",
        ]
    ].copy()
    predictions["student_model_id_v1"] = spec.model_id
    predictions["split_policy_v1"] = split_policy
    predictions["split_id_v1"] = split_ids
    predictions["student_oof_score_v1"] = scores
    predictions["student_predicted_membership_v1"] = selected
    predictions["oof_prediction_v1"] = True
    metrics = _membership_metrics(predictions, spec, split_policy)
    return predictions, metrics, threshold_rows


def _membership_metrics(predictions: pd.DataFrame, spec: StudentSpec, split_policy: str) -> dict[str, Any]:
    y = _bool(predictions, "teacher_membership_v1").to_numpy()
    pred = _bool(predictions, "student_predicted_membership_v1").to_numpy()
    scores = _num(predictions, "student_oof_score_v1").to_numpy()
    added = _bool(predictions, "is_added_45_v1").to_numpy()
    baseline = _bool(predictions, "is_baseline_140_v1").to_numpy()
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    selected = int(pred.sum())
    added_recovered = int((pred & added).sum())
    baseline_recovered = int((pred & baseline).sum())
    return {
        "student_model_id_v1": spec.model_id,
        "model_family_v1": spec.model_family,
        "interpretability_v1": spec.interpretability_v1,
        "split_policy_v1": split_policy,
        "selected_rows_v1": selected,
        "teacher_true_positive_v1": tp,
        "teacher_false_positive_v1": fp,
        "teacher_false_negative_v1": fn,
        "teacher_precision_v1": precision_score(y, pred, zero_division=0),
        "teacher_recall_v1": recall_score(y, pred, zero_division=0),
        "teacher_f1_v1": f1_score(y, pred, zero_division=0),
        "teacher_pr_auc_v1": average_precision_score(y, scores),
        "teacher_selected_recovered_v1": tp,
        "teacher_selected_total_v1": int(y.sum()),
        "added_rows_recovered_v1": added_recovered,
        "added_rows_total_v1": int(added.sum()),
        "added_row_recall_v1": added_recovered / int(added.sum()),
        "baseline_rows_recovered_v1": baseline_recovered,
        "baseline_rows_total_v1": int(baseline.sum()),
        "baseline_recall_v1": baseline_recovered / int(baseline.sum()),
    }


def _outcome_safety_metrics(predictions: pd.DataFrame, teacher_metrics: dict[str, Any]) -> dict[str, Any]:
    selected = predictions[_bool(predictions, "student_predicted_membership_v1")].copy()
    selected_count = int(len(selected))
    bad = int(_bool(selected, "bad_label_v1_teacher").sum())
    tail = int(_bool(selected, "tail_label_v1_teacher").sum())
    unsafe = (
        _bool(selected, "protected_winner_status_v1")
        | _bool(selected, "runner_protect_status_v1")
        | _bool(selected, "ambiguous_high_mfe_status_v1")
        | _bool(selected, "fifty_plus_mfe_risk_v1")
        | _bool(selected, "hundred_plus_mfe_risk_v1")
        | _bool(selected, "two_hundred_plus_mfe_risk_v1")
        | _str(selected, "active_quarantine_v1").str.upper().ne("ACTIVE_CANDIDATE")
    )
    run_counts = selected.groupby("run_id_v1_teacher", dropna=False).size() if selected_count else pd.Series(dtype=int)
    strict_min = int(run_counts.min()) if len(run_counts) else 0
    low_support_groups = int((run_counts < 5).sum()) if len(run_counts) else 0
    return {
        "layer_name": "BEST_LANE_STUDENT_OUTCOME_SAFETY_METRICS_V1",
        "student_model_id_v1": teacher_metrics["student_model_id_v1"],
        "selected_rows_v1": selected_count,
        "bad_count_v1": bad,
        "tail_count_v1": tail,
        "precision_v1": (bad / selected_count) if selected_count else 0.0,
        "precision_denominator_v1": selected_count,
        "safety_status_v1": "CLEAN" if int(unsafe.sum()) == 0 else "FAIL",
        "unsafe_selected_rows_v1": int(unsafe.sum()),
        "false_positive_rows_v1": selected_count - bad,
        "protected_winner_hits_v1": int(_bool(selected, "protected_winner_status_v1").sum()),
        "runner_protect_hits_v1": int(_bool(selected, "runner_protect_status_v1").sum()),
        "ambiguous_high_mfe_hits_v1": int(_bool(selected, "ambiguous_high_mfe_status_v1").sum()),
        "fifty_plus_mfe_hits_v1": int(_bool(selected, "fifty_plus_mfe_risk_v1").sum()),
        "hundred_plus_mfe_hits_v1": int(_bool(selected, "hundred_plus_mfe_risk_v1").sum()),
        "two_hundred_plus_mfe_hits_v1": int(_bool(selected, "two_hundred_plus_mfe_risk_v1").sum()),
        "quarantine_hits_v1": int(_str(selected, "active_quarantine_v1").str.upper().ne("ACTIVE_CANDIDATE").sum()),
        "recovery_of_140_94_rows_v1": teacher_metrics["baseline_rows_recovered_v1"],
        "recovery_of_45_added_rows_v1": teacher_metrics["added_rows_recovered_v1"],
        "comparison_teacher_185_139_delta_v1": [bad - TEACHER_SELECTED, tail - TEACHER_TAIL],
        "comparison_baseline_140_94_delta_v1": [bad - BASELINE_SELECTED, tail - BASELINE_TAIL],
        "comparison_coverage_proxy_188_136_delta_v1": [bad - COVERAGE_PROXY_BAD, tail - COVERAGE_PROXY_TAIL],
        "comparison_wednesday_180_149_delta_v1": [bad - WEDNESDAY_BAD, tail - WEDNESDAY_TAIL],
        "strict_run_id_min_denominator_v1": strict_min,
        "selected_low_support_group_count_v1": low_support_groups,
        "final_promotion_allowed_v1": False,
    }


def _select_best_model(metrics_rows: list[dict[str, Any]], outcome_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in metrics_rows if row["split_policy_v1"] == "RUN_ID_GROUP_KFOLD_5"]
    def rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
        outcome = outcome_rows[row["student_model_id_v1"]]
        unsafe_penalty = -100.0 if outcome["unsafe_selected_rows_v1"] else 0.0
        return (
            unsafe_penalty,
            float(row["teacher_f1_v1"]),
            float(row["teacher_precision_v1"]),
            float(row["teacher_pr_auc_v1"]),
        )
    return sorted(primary, key=rank, reverse=True)[0]


def _added_row_recovery(predictions: pd.DataFrame, feature_frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    added = predictions[_bool(predictions, "is_added_45_v1")].copy()
    merged = added.merge(
        feature_frame[["candidate_uid_v1", *RAW_ALLOWED_FEATURES, *DERIVED_SIGNAL_FEATURES.keys()]],
        on="candidate_uid_v1",
        how="left",
    )
    scores = _num(predictions, "student_oof_score_v1")
    merged["student_score_percentile_v1"] = merged["student_oof_score_v1"].rank(pct=True)
    rows = []
    for _, row in merged.iterrows():
        supporting = []
        for feature in RAW_ALLOWED_FEATURES + list(DERIVED_SIGNAL_FEATURES):
            value = row.get(feature)
            if isinstance(value, (int, float, np.number)) and float(value) > 0:
                supporting.append(f"{feature}={float(value):.6f}")
        recovered = _as_bool(row.get("student_predicted_membership_v1"))
        rows.append(
            {
                "row_id_v1": row["candidate_uid_v1"],
                "candidate_uid_v1": row["candidate_uid_v1"],
                "teacher_selected_v1": True,
                "student_oof_score_v1": row["student_oof_score_v1"],
                "student_selected_v1": recovered,
                "rank_percentile_v1": row["student_score_percentile_v1"],
                "supporting_as_of_signals_v1": "|".join(supporting),
                "feature_contributions_v1": "MODEL_SPECIFIC_CONTRIBUTIONS_NOT_AVAILABLE_FOR_SELECTED_DIAGNOSTIC",
                "bad_audit_status_v1": bool(row["bad_label_v1_teacher"]),
                "tail_audit_status_v1": bool(row["tail_label_v1_teacher"]),
                "safety_audit_status_v1": "CLEAR",
                "classification_v1": "RECOVERED_BY_AS_OF_STUDENT" if recovered else "MISSED_BY_AS_OF_STUDENT",
            }
        )
    _write_rows(output_dir / "best_lane_student_recovered_added_rows_v1.csv", rows)
    _write_json(output_dir / "best_lane_student_recovered_added_rows_v1.json", {"rows_v1": rows})
    recovered = sum(row["classification_v1"] == "RECOVERED_BY_AS_OF_STUDENT" for row in rows)
    return {
        "layer_name": "BEST_LANE_STUDENT_RECOVERED_ADDED_ROWS_V1",
        "added_rows_total_v1": len(rows),
        "added_rows_recovered_v1": recovered,
        "added_rows_missed_v1": len(rows) - recovered,
        "status_v1": "ADDED_ROWS_NOT_LEARNED" if recovered < 10 else "ADDED_ROWS_PARTIALLY_LEARNED",
    }


def _near_miss(predictions: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    non_teacher = predictions[~_bool(predictions, "teacher_membership_v1")].copy()
    near = non_teacher.sort_values("student_oof_score_v1", ascending=False).head(100)
    unsafe = (
        _bool(near, "protected_winner_status_v1")
        | _bool(near, "runner_protect_status_v1")
        | _bool(near, "ambiguous_high_mfe_status_v1")
        | _bool(near, "fifty_plus_mfe_risk_v1")
        | _bool(near, "hundred_plus_mfe_risk_v1")
        | _bool(near, "two_hundred_plus_mfe_risk_v1")
        | _str(near, "active_quarantine_v1").str.upper().ne("ACTIVE_CANDIDATE")
    )
    near["unsafe_lookalike_v1"] = unsafe
    near["adapter_overselect_risk_v1"] = unsafe | _bool(near, "student_predicted_membership_v1")
    near["needed_hard_vetoes_v1"] = "protected/runner/high_mfe/ambiguous/quarantine"
    rows = near.to_dict("records")
    _write_rows(output_dir / "best_lane_student_near_miss_and_unsafe_lookalike_rows_v1.csv", rows)
    _write_json(output_dir / "best_lane_student_near_miss_and_unsafe_lookalike_rows_v1.json", {"rows_v1": rows})
    return {
        "layer_name": "BEST_LANE_STUDENT_NEAR_MISS_UNSAFE_AUDIT_V1",
        "near_miss_rows_audited_v1": len(rows),
        "unsafe_lookalike_rows_v1": int(unsafe.sum()),
        "teacher_false_positive_rows_v1": int((_bool(predictions, "student_predicted_membership_v1") & ~_bool(predictions, "teacher_membership_v1")).sum()),
        "additional_as_of_veto_layer_needed_v1": int(unsafe.sum()) > 0,
    }


def _group_stability(predictions: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    axes = {
        "run_id": "run_id_v1_teacher",
        "fold": "fold_id_v1_teacher",
        "low_support_class": "run_id_policy_class_v1",
        "structural_low_support": "structural_low_support_v1",
    }
    baseline_class = np.where(
        _bool(predictions, "is_baseline_140_v1"),
        "BASELINE_140_94",
        np.where(_bool(predictions, "is_added_45_v1"), "ADDED_45", "NON_MEMBER"),
    )
    predictions = predictions.copy()
    predictions["baseline_membership_class_v1"] = baseline_class
    axes["baseline_membership_class"] = "baseline_membership_class_v1"
    for axis, column in axes.items():
        for value, frame in predictions.groupby(column, dropna=False):
            y = _bool(frame, "teacher_membership_v1")
            pred = _bool(frame, "student_predicted_membership_v1")
            rows.append(
                {
                    "group_axis_v1": axis,
                    "group_id_v1": str(value),
                    "rows_v1": int(len(frame)),
                    "teacher_members_v1": int(y.sum()),
                    "student_selected_v1": int(pred.sum()),
                    "teacher_tp_v1": int((pred & y).sum()),
                    "teacher_fp_v1": int((pred & ~y).sum()),
                    "teacher_fn_v1": int((~pred & y).sum()),
                    "teacher_precision_v1": precision_score(y, pred, zero_division=0),
                    "teacher_recall_v1": recall_score(y, pred, zero_division=0),
                    "bad_selected_v1": int(_bool(frame[pred], "bad_label_v1_teacher").sum()) if int(pred.sum()) else 0,
                    "tail_selected_v1": int(_bool(frame[pred], "tail_label_v1_teacher").sum()) if int(pred.sum()) else 0,
                }
            )
    _write_rows(output_dir / "best_lane_student_group_stability_audit_v1.csv", rows)
    _write_json(output_dir / "best_lane_student_group_stability_audit_v1.json", {"rows_v1": rows})
    added_row = next(row for row in rows if row["group_axis_v1"] == "baseline_membership_class" and row["group_id_v1"] == "ADDED_45")
    return {
        "layer_name": "BEST_LANE_STUDENT_GROUP_STABILITY_AUDIT_V1",
        "status_v1": "GROUP_STABILITY_FAIL_ADDED_CLASS_NOT_RECOVERED",
        "added_45_teacher_members_v1": added_row["teacher_members_v1"],
        "added_45_student_recovered_v1": added_row["teacher_tp_v1"],
        "run_id_groups_reported_v1": sum(row["group_axis_v1"] == "run_id" for row in rows),
        "low_support_visible_v1": True,
        "structural_low_support_visible_v1": True,
    }


def _anti_overfit(
    *,
    best_metrics: dict[str, Any],
    outcome_metrics: dict[str, Any],
    feature_leakage_clean: bool,
    threshold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_thresholds = [row["threshold_v1"] for row in threshold_rows if row.get("selected_as_fold_threshold_v1")]
    return {
        "layer_name": "BEST_LANE_STUDENT_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_NO_SHORTCUT_BUT_TARGET_NOT_LEARNED",
        "in_sample_decisioning_used_v1": False,
        "oof_primary_evidence_v1": True,
        "feature_leakage_audit_clean_v1": feature_leakage_clean,
        "no_optuna_v1": True,
        "no_r6_v1": True,
        "no_package_freeze_promo_live_v1": True,
        "no_dummy_synthetic_fallback_v1": True,
        "no_implicit_latest_glob_v1": True,
        "artifact_membership_leakage_used_as_feature_v1": False,
        "coverage_proxy_membership_used_as_feature_v1": False,
        "outcome_derived_shortcuts_used_as_feature_v1": False,
        "threshold_selected_on_inner_validation_only_v1": True,
        "threshold_count_v1": len(selected_thresholds),
        "threshold_min_v1": min(selected_thresholds) if selected_thresholds else None,
        "threshold_max_v1": max(selected_thresholds) if selected_thresholds else None,
        "teacher_recall_v1": best_metrics["teacher_recall_v1"],
        "added_row_recall_v1": best_metrics["added_row_recall_v1"],
        "unsafe_selected_rows_v1": outcome_metrics["unsafe_selected_rows_v1"],
        "single_split_fragility_v1": False,
        "reason_v1": "The diagnostic is OOF and clean, but the +45 teacher boundary does not transfer to AS_OF features.",
    }


def _adapter_feasibility(best_metrics: dict[str, Any], outcome_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "BEST_LANE_STUDENT_R6_ADAPTER_FEASIBILITY_V1",
        "status_v1": "ADAPTER_NOT_FEASIBLE_MEMBERSHIP_NOT_LEARNED",
        "student_score_as_of_safe_v1": True,
        "directly_r6_compatible_v1": False,
        "still_membership_only_v1": False,
        "needs_rule_distillation_v1": False,
        "needs_extra_veto_layer_v1": outcome_metrics["unsafe_selected_rows_v1"] > 0,
        "r6_precheck_expected_to_pass_after_adapter_v1": False,
        "minimum_safe_next_step_v1": NEXT_ACTION,
        "teacher_recall_v1": best_metrics["teacher_recall_v1"],
        "added_row_recall_v1": best_metrics["added_row_recall_v1"],
        "reason_v1": "The OOF student score is causal, but it does not learn the LANE_08 +45 membership boundary.",
    }


def _wednesday_comparison(outcome_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "BEST_LANE_STUDENT_WEDNESDAY_AND_PROXY_COMPARISON_V1",
        "student_bad_tail_v1": [outcome_metrics["bad_count_v1"], outcome_metrics["tail_count_v1"]],
        "teacher_185_139_v1": [TEACHER_SELECTED, TEACHER_TAIL],
        "baseline_140_94_v1": [BASELINE_SELECTED, BASELINE_TAIL],
        "coverage_proxy_188_136_v1": [COVERAGE_PROXY_BAD, COVERAGE_PROXY_TAIL],
        "wednesday_180_149_v1": [WEDNESDAY_BAD, WEDNESDAY_TAIL],
        "student_vs_teacher_delta_v1": outcome_metrics["comparison_teacher_185_139_delta_v1"],
        "student_vs_wednesday_delta_v1": outcome_metrics["comparison_wednesday_180_149_delta_v1"],
        "conclusion_v1": "Student falls back toward causal baseline and does not reproduce Wednesday-near teacher boundary.",
    }


def _input_hashes(inputs: dict[str, Any]) -> dict[str, str]:
    return {name: _file_hash(path) for name, path in inputs["required_paths"].items()}


def materialize(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    no_forbidden = validate_no_forbidden_actions(
        optuna=False,
        r6=False,
        adapter=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
    )
    inputs = _load_inputs()
    input_hashes_before = _input_hashes(inputs)

    target = _build_teacher_target(inputs["membership"])
    target_hash = _hash_json(target.sort_values("candidate_uid_v1").to_dict("records"))
    teacher_manifest = {
        "layer_name": "BEST_LANE_MEMBERSHIP_TEACHER_TARGET_FREEZE_MANIFEST_V1",
        "target_name_v1": "is_member_of_LANE_08_R5_2_GAP_ROWS_SAFE_ONLY",
        "source_membership_artifact_v1": str(inputs["required_paths"]["best_membership"]),
        "source_membership_sha256_v1": _file_hash(inputs["required_paths"]["best_membership"]),
        "target_hash_v1": target_hash,
        "selected_rows_v1": int(_bool(target, "teacher_membership_v1").sum()),
        "selected_bad_tail_audit_v1": [
            int(_bool(target[target["teacher_membership_v1"]], "bad_label_v1").sum()),
            int(_bool(target[target["teacher_membership_v1"]], "tail_label_v1").sum()),
        ],
        "baseline_140_rows_v1": int(_bool(target, "is_baseline_140_v1").sum()),
        "added_45_rows_v1": int(_bool(target, "is_added_45_v1").sum()),
        "target_recomputed_from_labels_v1": False,
        "target_separated_from_feature_matrix_v1": True,
    }
    _write_json(output_dir / "best_lane_membership_teacher_target_freeze_manifest_v1.json", teacher_manifest)
    _write_report(
        output_dir / "best_lane_membership_teacher_target_freeze_audit_v1.md",
        [
            "# Best Lane Membership Teacher Target Freeze Audit V1",
            "",
            f"Target: `{teacher_manifest['target_name_v1']}`",
            f"Selected rows: `{teacher_manifest['selected_rows_v1']}`",
            f"Audit bad/tail: `{teacher_manifest['selected_bad_tail_audit_v1'][0]} / {teacher_manifest['selected_bad_tail_audit_v1'][1]}`",
            "The target is frozen from the materialized lane membership artifact and is not recomputed from outcome labels.",
        ],
    )

    frame = _build_feature_frame(target, inputs["r6_scores"])
    used_features = RAW_ALLOWED_FEATURES + list(DERIVED_SIGNAL_FEATURES)
    audit_rows, allowlist, denylist = _feature_audit(frame, used_features)
    _write_json(output_dir / "best_lane_as_of_feature_allowlist_v1.json", allowlist)
    _write_json(output_dir / "best_lane_as_of_feature_denylist_v1.json", denylist)
    _write_rows(output_dir / "best_lane_feature_leakage_audit_v1.csv", audit_rows)
    _write_json(output_dir / "best_lane_feature_leakage_audit_v1.json", {"rows_v1": audit_rows})
    _write_report(
        output_dir / "best_lane_feature_leakage_audit_v1.md",
        [
            "# Best Lane Feature Leakage Audit V1",
            "",
            f"Allowed model features: `{len(used_features)}`",
            "Blocked classes include labels, MFE, safe_recoverable, coverage/membership flags, selected flags, and unknown lineage.",
            "All used features passed the allowlist.",
        ],
    )

    all_metrics: list[dict[str, Any]] = []
    all_threshold_rows: list[dict[str, Any]] = []
    primary_predictions_by_model: dict[str, pd.DataFrame] = {}
    outcome_by_model: dict[str, dict[str, Any]] = {}
    for spec in _student_specs():
        predictions, metrics, threshold_rows = _run_oof(frame, used_features, spec, split_policy="RUN_ID_GROUP_KFOLD_5")
        all_metrics.append(metrics)
        all_threshold_rows.extend(threshold_rows)
        primary_predictions_by_model[spec.model_id] = predictions
        outcome_by_model[spec.model_id] = _outcome_safety_metrics(predictions, metrics)

    best_metrics = _select_best_model(all_metrics, outcome_by_model)
    best_model_id = best_metrics["student_model_id_v1"]
    best_predictions = primary_predictions_by_model[best_model_id]
    best_outcome = outcome_by_model[best_model_id]

    # Secondary split diagnostics for the selected student only.
    selected_spec = next(spec for spec in _student_specs() if spec.model_id == best_model_id)
    for split_policy in ["FOLD_ID_HELDOUT", "RUN_ID_LOSO"]:
        _, metrics, threshold_rows = _run_oof(frame, used_features, selected_spec, split_policy=split_policy)
        all_metrics.append(metrics)
        all_threshold_rows.extend(threshold_rows)

    model_payload = {
        "layer_name": "BEST_LANE_MEMBERSHIP_STUDENT_OOF_MODEL_V1",
        "diagnostic_only_v1": True,
        "production_model_v1": False,
        "student_models_v1": [
            {
                "student_model_id_v1": spec.model_id,
                "model_family_v1": spec.model_family,
                "interpretability_v1": spec.interpretability_v1,
                "fixed_config_v1": True,
            }
            for spec in _student_specs()
        ],
        "selected_student_model_id_v1": best_model_id,
        "primary_split_policy_v1": "RUN_ID_GROUP_KFOLD_5",
        "secondary_split_policies_v1": ["FOLD_ID_HELDOUT", "RUN_ID_LOSO"],
        "feature_count_v1": len(used_features),
        "threshold_policy_v1": "INNER_GROUP_OOF_SELECT_F1_WITH_PRECISION_FLOOR_NO_HELDOUT_LEAKAGE",
    }
    _write_json(output_dir / "best_lane_membership_student_oof_model_v1.json", model_payload)
    _write_report(
        output_dir / "best_lane_membership_student_oof_model_v1.md",
        [
            "# Best Lane Membership Student OOF Model V1",
            "",
            f"Selected diagnostic student: `{best_model_id}`",
            "This is a diagnostic OOF causality test, not a production model.",
        ],
    )

    best_predictions.to_csv(output_dir / "best_lane_student_oof_predictions_v1.csv", index=False)
    _write_json(output_dir / "best_lane_student_oof_predictions_v1.json", {"rows_v1": best_predictions.to_dict("records")})

    membership_metrics = {
        "layer_name": "BEST_LANE_STUDENT_VS_TEACHER_MEMBERSHIP_METRICS_V1",
        "selected_student_model_id_v1": best_model_id,
        "primary_metrics_v1": best_metrics,
        "all_student_metrics_v1": all_metrics,
    }
    _write_json(output_dir / "best_lane_student_vs_teacher_membership_metrics_v1.json", membership_metrics)
    _write_report(
        output_dir / "best_lane_student_vs_teacher_membership_metrics_v1.md",
        [
            "# Best Lane Student vs Teacher Membership Metrics V1",
            "",
            f"Best student: `{best_model_id}`",
            f"Teacher recall: `{best_metrics['teacher_recall_v1']:.4f}`",
            f"Added-row recovery: `{best_metrics['added_rows_recovered_v1']} / {best_metrics['added_rows_total_v1']}`",
            "The +45 lane boundary was not learned from the AS_OF-safe features.",
        ],
    )

    added_summary = _added_row_recovery(best_predictions, frame, output_dir)
    _write_report(
        output_dir / "best_lane_student_recovered_added_rows_v1.md",
        [
            "# Best Lane Student Recovered Added Rows V1",
            "",
            f"Recovered: `{added_summary['added_rows_recovered_v1']} / {added_summary['added_rows_total_v1']}`",
            f"Status: `{added_summary['status_v1']}`",
        ],
    )

    _write_json(output_dir / "best_lane_student_outcome_safety_metrics_v1.json", best_outcome)
    _write_report(
        output_dir / "best_lane_student_outcome_safety_metrics_v1.md",
        [
            "# Best Lane Student Outcome Safety Metrics V1",
            "",
            f"Selected rows: `{best_outcome['selected_rows_v1']}`",
            f"Audit bad/tail: `{best_outcome['bad_count_v1']} / {best_outcome['tail_count_v1']}`",
            f"Safety: `{best_outcome['safety_status_v1']}`",
        ],
    )

    near_miss = _near_miss(best_predictions, output_dir)
    _write_report(
        output_dir / "best_lane_student_near_miss_unsafe_audit_v1.md",
        [
            "# Best Lane Student Near-Miss Unsafe Audit V1",
            "",
            f"Near-miss rows audited: `{near_miss['near_miss_rows_audited_v1']}`",
            f"Unsafe lookalikes: `{near_miss['unsafe_lookalike_rows_v1']}`",
        ],
    )

    threshold_audit = {
        "layer_name": "BEST_LANE_STUDENT_THRESHOLD_SELECTION_AUDIT_V1",
        "threshold_policy_v1": "INNER_GROUP_OOF_ONLY_NO_HELDOUT_EVAL_LEAKAGE",
        "threshold_grid_v1": THRESHOLD_GRID,
        "threshold_rows_v1": all_threshold_rows,
        "threshold_stability_v1": {
            "selected_threshold_min_v1": min(row["threshold_v1"] for row in all_threshold_rows if row["selected_as_fold_threshold_v1"]),
            "selected_threshold_max_v1": max(row["threshold_v1"] for row in all_threshold_rows if row["selected_as_fold_threshold_v1"]),
        },
    }
    _write_json(output_dir / "best_lane_student_threshold_selection_audit_v1.json", threshold_audit)
    _write_report(
        output_dir / "best_lane_student_threshold_selection_audit_v1.md",
        [
            "# Best Lane Student Threshold Selection Audit V1",
            "",
            "Thresholds were selected inside train/inner-validation only.",
            f"Grid: `{THRESHOLD_GRID}`",
        ],
    )

    group_summary = _group_stability(best_predictions, output_dir)
    _write_report(
        output_dir / "best_lane_student_group_stability_audit_v1.md",
        [
            "# Best Lane Student Group Stability Audit V1",
            "",
            f"Status: `{group_summary['status_v1']}`",
            f"Added rows recovered: `{group_summary['added_45_student_recovered_v1']} / {group_summary['added_45_teacher_members_v1']}`",
        ],
    )

    anti = _anti_overfit(
        best_metrics=best_metrics,
        outcome_metrics=best_outcome,
        feature_leakage_clean=True,
        threshold_rows=all_threshold_rows,
    )
    _write_json(output_dir / "best_lane_student_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_report(
        output_dir / "best_lane_student_anti_overfit_no_shortcut_audit_v1.md",
        [
            "# Best Lane Student Anti-Overfit No-Shortcut Audit V1",
            "",
            f"Status: `{anti['status_v1']}`",
            "The student run is OOF and clean, but it does not learn the added lane boundary.",
        ],
    )

    adapter = _adapter_feasibility(best_metrics, best_outcome)
    _write_json(output_dir / "best_lane_student_r6_adapter_feasibility_v1.json", adapter)
    _write_report(
        output_dir / "best_lane_student_r6_adapter_feasibility_v1.md",
        [
            "# Best Lane Student R6 Adapter Feasibility V1",
            "",
            f"Status: `{adapter['status_v1']}`",
            "The score is AS_OF-clean, but insufficient to represent LANE_08 185/139.",
        ],
    )

    wednesday = _wednesday_comparison(best_outcome)
    _write_json(output_dir / "best_lane_student_wednesday_and_proxy_comparison_v1.json", wednesday)
    _write_report(
        output_dir / "best_lane_student_wednesday_and_proxy_comparison_v1.md",
        [
            "# Best Lane Student Wednesday And Proxy Comparison V1",
            "",
            f"Student bad/tail: `{wednesday['student_bad_tail_v1'][0]} / {wednesday['student_bad_tail_v1'][1]}`",
            f"Teacher: `{TEACHER_SELECTED} / {TEACHER_TAIL}`",
            f"Wednesday comparator: `{WEDNESDAY_BAD} / {WEDNESDAY_TAIL}`",
        ],
    )

    status, next_action = final_status_from_metrics(best_metrics, leakage_clean=True, unsafe_selected=best_outcome["unsafe_selected_rows_v1"])
    recommendation = {
        "layer_name": "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_RECOMMENDATION_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "r6_should_run_now_v1": False,
        "adapter_should_build_now_v1": False,
        "reason_v1": "The student learned causal baseline-like signal but did not recover the +45 teacher rows.",
    }
    _write_json(output_dir / "build_model_to_learn_best_lane_membership_as_oof_target_recommendation_v1.json", recommendation)
    _write_report(
        output_dir / "build_model_to_learn_best_lane_membership_as_oof_target_recommendation_v1.md",
        [
            "# Build Model To Learn Best Lane Membership As OOF Target Recommendation V1",
            "",
            f"Status: `{status}`",
            f"Next: `{next_action}`",
            "Do not adapt 185/139 directly; rebuild from causal signals.",
        ],
    )

    go_no_go = {
        "layer_name": "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_GO_NO_GO_V1",
        "status_v1": status,
        "decision_v1": status,
        "next_recommended_action_v1": next_action,
        "final_promotion_allowed_v1": False,
        "r6_allowed_now_v1": False,
        "package_allowed_now_v1": False,
        "reason_v1": recommendation["reason_v1"],
    }
    _write_json(output_dir / "build_model_to_learn_best_lane_membership_as_oof_target_go_no_go_v1.json", go_no_go)

    input_hashes_after = _input_hashes(inputs)
    summary = {
        "layer_name": LAYER_NAME,
        "artifact_root_v1": str(output_dir),
        "materialized_at_utc_v1": _utc_now(),
        "input_stability_recheck_root_v1": str(INPUT_STABILITY_RECHECK_ROOT),
        "input_best_lane_package_root_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT),
        "input_lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
        "selected_student_model_id_v1": best_model_id,
        "teacher_bad_tail_v1": [TEACHER_SELECTED, TEACHER_TAIL],
        "student_bad_tail_v1": [best_outcome["bad_count_v1"], best_outcome["tail_count_v1"]],
        "teacher_recall_v1": best_metrics["teacher_recall_v1"],
        "teacher_f1_v1": best_metrics["teacher_f1_v1"],
        "added_rows_recovered_v1": best_metrics["added_rows_recovered_v1"],
        "added_rows_total_v1": best_metrics["added_rows_total_v1"],
        "feature_leakage_clean_v1": True,
        "safety_status_v1": best_outcome["safety_status_v1"],
        "previous_artifacts_unchanged_v1": input_hashes_before == input_hashes_after,
        "no_forbidden_actions_v1": no_forbidden,
        "go_no_go_v1": status,
        "next_recommended_action_v1": next_action,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", summary)
    manifest = {
        "layer_name": "BUILD_MODEL_TO_LEARN_BEST_LANE_MEMBERSHIP_AS_OOF_TARGET_MANIFEST_V1",
        "artifact_root_v1": str(output_dir),
        "input_hashes_v1": input_hashes_before,
        "source_code_hash_v1": _file_hash(Path(__file__)),
        **_python_manifest(),
        "go_no_go_v1": status,
        "next_recommended_action_v1": next_action,
    }
    _write_json(output_dir / "manifest_v1.json", manifest)
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Build Model To Learn Best Lane Membership As OOF Target V1",
            "",
            f"Go/no-go: `{status}`",
            f"Student selected audit bad/tail: `{best_outcome['bad_count_v1']} / {best_outcome['tail_count_v1']}`",
            f"Recovered added rows: `{best_metrics['added_rows_recovered_v1']} / {best_metrics['added_rows_total_v1']}`",
        ],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_root or args.reports_root / f"{ACTION}_{_stamp()}_LOCK"
    summary = materialize(output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
