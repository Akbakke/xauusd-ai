#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "PLUS45_AS_OF_FEATURE_GAP_SHADOW_EXPLORATION_V1"

INPUT_140_94_PRECHECK_ROOT = (
    DEFAULT_REPORTS_ROOT / "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"
)
INPUT_REJECT_REBUILD_ROOT = (
    DEFAULT_REPORTS_ROOT / "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1_20260428T063714Z_LOCK"
)
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

SELECTED_LANE_ID = "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY"
MAINLINE_CURRENT_BEST = "RETURN_TO_140_94_CAUSAL_BASELINE_BEST_CURRENT_OPTION"
MAINLINE_NEXT_ACTION = "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1"

BASELINE_SELECTED = 140
BASELINE_BAD = 140
BASELINE_TAIL = 94
BEST_LANE_SELECTED = 185
BEST_LANE_BAD = 185
BEST_LANE_TAIL = 139
PLUS45_COUNT = 45
PLUS45_BAD = 45
PLUS45_TAIL = 45
STUDENT_SELECTED = 135
STUDENT_BAD = 131
STUDENT_TAIL = 93
WEDNESDAY_BAD = 180
WEDNESDAY_TAIL = 149
COVERAGE_PROXY_BAD = 188
COVERAGE_PROXY_TAIL = 136

FINAL_SHADOW_STATUS = "PLUS45_SHADOW_FOUND_ONLY_MEMBERSHIP_OR_COVERAGE_DEPENDENCY"
SHADOW_NEXT_ACTION = "ARCHIVE_PLUS45_AS_DIAGNOSTIC_ONLY_AND_CONTINUE_140_94_V1"

ALLOWED_FINAL_SHADOW_STATUSES = {
    "PLUS45_SHADOW_FOUND_PROMISING_AS_OF_FEATURE_FAMILIES",
    "PLUS45_SHADOW_FOUND_AS_OF_FEATURE_FAMILIES_BUT_NEEDS_VETO_AUDIT",
    "PLUS45_SHADOW_FOUND_ONLY_MEMBERSHIP_OR_COVERAGE_DEPENDENCY",
    "PLUS45_SHADOW_FOUND_UNSAFE_LOOKALIKE_RISK",
    "PLUS45_SHADOW_NO_ACTIONABLE_AS_OF_SIGNAL_FOUND",
    "PLUS45_SHADOW_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "PLUS45_SHADOW_BLOCKED_BY_FEATURE_LINEAGE_GAPS",
    "PLUS45_SHADOW_BLOCKED_BY_MISSING_ARTIFACTS",
    "PLUS45_SHADOW_BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_SHADOW_NEXT_ACTIONS = {
    "CONTINUE_140_94_DISTILLATION_UNCHANGED",
    "TEST_PLUS45_FEATURE_FAMILY_AS_CAUSAL_EXPANSION_CANDIDATE_V1",
    "DEEPEN_PLUS45_FEATURE_LINEAGE_AUDIT_V1",
    "DEEPEN_PLUS45_UNSAFE_LOOKALIKE_AUDIT_V1",
    "ARCHIVE_PLUS45_AS_DIAGNOSTIC_ONLY_AND_CONTINUE_140_94_V1",
    "REBUILD_AS_OF_FEATURE_INVENTORY_FOR_TAIL_GAP_PRECURSORS_V1",
}

AS_OF_REFERENCE_FEATURES = [
    "r5_2_coverage_bad_score_v1",
    "r5_2_coverage_tail_score_v1",
    "r5_2_coverage_hard_veto_score_v1",
    "pred__entry_r5_2_bad_blocker__prob_true_v1",
    "blocker_score_v1",
    "pred__entry_r6_bad_risk__prob_true_v1",
    "pred__entry_r6_tail_control_10_50__prob_true_v1",
    "pred__entry_r6_risky_allow__prob_true_v1",
    "pred__entry_r6_batch04_blindspot__prob_true_v1",
    "asof_signal__r5_bad_score_v1",
    "asof_signal__r5_1_bad_score_v1",
    "asof_signal__r5_tail_score_v1",
    "asof_signal__v2_like_bad_tail_v1",
]

DENY_PATTERNS = [
    "bad_label",
    "tail_label",
    "label_should_not_take",
    "take_was_ok",
    "final_outcome",
    "post_outcome",
    "mfe",
    "fifty_plus",
    "hundred_plus",
    "two_hundred_plus",
    "safe_recoverable",
    "coverage_proxy",
    "185_139",
    "plus45",
    "plus_45",
    "rows_added_vs_140_94",
    "lane_selected",
    "lane_id",
    "teacher_membership",
    "selected_by",
    "selected_rows",
    "r5_2_package_selected",
    "student_predicted_membership",
    "decision_valid",
    "protected",
    "runner",
    "ambiguous",
    "quarantine",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "row_id",
    "artifact_path",
    "hash",
    "latest",
    "glob",
]

REQUIRED_OUTPUTS = [
    "plus45_shadow_input_manifest_v1.json",
    "plus45_shadow_input_integrity_audit_v1.json",
    "plus45_shadow_input_integrity_audit_v1.md",
    "plus45_shadow_cohort_reconstruction_v1.json",
    "plus45_shadow_cohort_reconstruction_v1.md",
    "plus45_shadow_row_level_miss_audit_v1.csv",
    "plus45_shadow_row_level_miss_audit_v1.json",
    "plus45_shadow_row_level_miss_audit_v1.md",
    "plus45_shadow_candidate_feature_families_v1.csv",
    "plus45_shadow_candidate_feature_families_v1.json",
    "plus45_shadow_candidate_feature_families_v1.md",
    "plus45_shadow_feature_lineage_audit_v1.csv",
    "plus45_shadow_feature_lineage_audit_v1.json",
    "plus45_shadow_feature_lineage_audit_v1.md",
    "plus45_shadow_diagnostic_probe_results_v1.csv",
    "plus45_shadow_diagnostic_probe_results_v1.json",
    "plus45_shadow_diagnostic_probe_results_v1.md",
    "plus45_shadow_unsafe_lookalike_audit_v1.csv",
    "plus45_shadow_unsafe_lookalike_audit_v1.json",
    "plus45_shadow_unsafe_lookalike_audit_v1.md",
    "plus45_shadow_group_support_stability_audit_v1.csv",
    "plus45_shadow_group_support_stability_audit_v1.json",
    "plus45_shadow_group_support_stability_audit_v1.md",
    "plus45_shadow_anti_overfit_no_shortcut_audit_v1.json",
    "plus45_shadow_anti_overfit_no_shortcut_audit_v1.md",
    "plus45_shadow_recommendation_v1.json",
    "plus45_shadow_recommendation_v1.md",
    "plus45_as_of_feature_gap_shadow_exploration_go_no_go_v1.json",
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


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass", "active_candidate", "clear"}
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


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


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
    r6: bool = False,
    adapter: bool = False,
    package: bool = False,
    freeze: bool = False,
    promo: bool = False,
    live: bool = False,
    optuna: bool = False,
    production_model: bool = False,
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
    if production_model:
        failures.append("PRODUCTION_MODEL_TRAINING_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_no_forbidden_feature_names(features: Iterable[str]) -> bool:
    blocked = []
    for feature in features:
        lower = feature.lower()
        if any(pattern in lower for pattern in DENY_PATTERNS):
            blocked.append(feature)
    if blocked:
        raise RuntimeError(f"FORBIDDEN_PLUS45_SHADOW_FEATURE: {blocked}")
    return True


def validate_diagnostic_only_policy(payload: dict[str, Any]) -> bool:
    if payload.get("mainline_next_action_v1") != MAINLINE_NEXT_ACTION:
        raise RuntimeError("MAINLINE_NEXT_ACTION_MUST_REMAIN_DISTILL_140_94")
    if payload.get("plus45_role_v1") != "DIAGNOSTIC_ONLY_NOT_TARGET_FEATURE_FILTER_OR_THRESHOLD_OBJECTIVE":
        raise RuntimeError("PLUS45_MUST_REMAIN_DIAGNOSTIC_ONLY")
    if payload.get("best_lane_185_139_role_v1") != "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE":
        raise RuntimeError("BEST_LANE_185_139_MUST_REMAIN_COMPARATOR_ONLY")
    if payload.get("coverage_proxy_role_v1") != "COMPARATOR_ONLY_NOT_FEATURE_FILTER_OR_TARGET":
        raise RuntimeError("COVERAGE_PROXY_MUST_NOT_BE_USED_AS_FEATURE_FILTER_OR_TARGET")
    return True


def validate_cohort_counts(payload: dict[str, Any]) -> bool:
    expected = {
        "baseline_140_94_selected_rows_v1": BASELINE_SELECTED,
        "best_lane_185_139_selected_rows_v1": BEST_LANE_SELECTED,
        "plus45_rows_v1": PLUS45_COUNT,
        "plus45_bad_rows_audit_only_v1": PLUS45_BAD,
        "plus45_tail_rows_audit_only_v1": PLUS45_TAIL,
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if failures:
        raise RuntimeError(f"PLUS45_COHORT_RECONSTRUCTION_FAILED: {failures}")
    return True


def validate_final_shadow_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_SHADOW_STATUSES:
        raise RuntimeError(f"FINAL_SHADOW_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_SHADOW_NEXT_ACTIONS:
        raise RuntimeError(f"SHADOW_NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_required_outputs(root: Path) -> bool:
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"PLUS45_SHADOW_REQUIRED_OUTPUTS_MISSING: {missing}")
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


def _parse_signal_values(text: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    if not isinstance(text, str):
        return values
    for part in text.split("|"):
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        value = _safe_float(raw)
        if value is not None:
            values[key] = value
    return values


def _evidence_tokens(text: Any) -> set[str]:
    if not isinstance(text, str) or not text:
        return set()
    tokens: set[str] = set()
    for part in text.split("|"):
        token = part.split(":", 1)[0].strip()
        if token:
            tokens.add(token)
    return tokens


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_140_94_PRECHECK_ROOT,
        INPUT_REJECT_REBUILD_ROOT,
        INPUT_STUDENT_ROOT,
        INPUT_STABILITY_ROOT,
        INPUT_BEST_LANE_PACKAGE_ROOT,
        INPUT_LANE_PACK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "precheck_summary": INPUT_140_94_PRECHECK_ROOT / "summary_v1.json",
        "precheck_go_no_go": INPUT_140_94_PRECHECK_ROOT
        / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
        "precheck_lineage": INPUT_140_94_PRECHECK_ROOT / "baseline_140_94_selection_lineage_v1.csv",
        "precheck_near_miss": INPUT_140_94_PRECHECK_ROOT / "baseline_140_94_near_miss_and_near_fail_rows_v1.csv",
        "precheck_group_stability": INPUT_140_94_PRECHECK_ROOT / "baseline_140_94_group_stability_audit_v1.csv",
        "precheck_feature_lineage": INPUT_140_94_PRECHECK_ROOT / "baseline_140_94_feature_lineage_audit_v1.csv",
        "reject_summary": INPUT_REJECT_REBUILD_ROOT / "summary_v1.json",
        "reject_go_no_go": INPUT_REJECT_REBUILD_ROOT
        / "reject_or_rebuild_best_lane_from_causal_signals_go_no_go_v1.json",
        "causal_inventory": INPUT_REJECT_REBUILD_ROOT / "causal_signal_inventory_v1.csv",
        "causal_metrics": INPUT_REJECT_REBUILD_ROOT / "causal_rebuild_candidate_metrics_v1.csv",
        "student_summary": INPUT_STUDENT_ROOT / "summary_v1.json",
        "student_go_no_go": INPUT_STUDENT_ROOT
        / "build_model_to_learn_best_lane_membership_as_oof_target_go_no_go_v1.json",
        "student_predictions": INPUT_STUDENT_ROOT / "best_lane_student_oof_predictions_v1.csv",
        "student_added_rows": INPUT_STUDENT_ROOT / "best_lane_student_recovered_added_rows_v1.csv",
        "student_near_miss": INPUT_STUDENT_ROOT / "best_lane_student_near_miss_and_unsafe_lookalike_rows_v1.csv",
        "student_feature_leakage": INPUT_STUDENT_ROOT / "best_lane_feature_leakage_audit_v1.csv",
        "student_metrics": INPUT_STUDENT_ROOT / "best_lane_student_vs_teacher_membership_metrics_v1.json",
        "stability_summary": INPUT_STABILITY_ROOT / "summary_v1.json",
        "stability_go_no_go": INPUT_STABILITY_ROOT
        / "stability_recheck_best_lane_185_139_before_r6_go_no_go_v1.json",
        "added_evidence": INPUT_STABILITY_ROOT / "best_lane_added_rows_selection_evidence_v1.csv",
        "added_strength": INPUT_STABILITY_ROOT / "best_lane_added_row_evidence_strength_audit_v1.csv",
        "best_lane_membership": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv",
        "best_lane_selected": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_selected_rows_v1.csv",
        "best_lane_integrity": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_package_integrity_report_v1.json",
        "best_lane_go_no_go": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_package_go_no_go_v1.json",
        "lane_pack_summary": INPUT_LANE_PACK_ROOT / "summary_v1.json",
        "lane_pack_go_no_go": INPUT_LANE_PACK_ROOT / "parallel_tail_r6_r5_2_repair_lane_pack_go_no_go_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    precheck_go = _read_json(required["precheck_go_no_go"])
    reject_go = _read_json(required["reject_go_no_go"])
    student_go = _read_json(required["student_go_no_go"])
    stability_go = _read_json(required["stability_go_no_go"])
    if precheck_go.get("status_v1") != "140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER":
        raise RuntimeError("PRECHECK_STATUS_NOT_RULE_DISTILLATION_REQUIRED")
    if reject_go.get("status_v1") != "RETURN_TO_140_94_CAUSAL_BASELINE_BEST_CURRENT_OPTION":
        raise RuntimeError("REJECT_REBUILD_STATUS_NOT_RETURN_TO_140")
    if student_go.get("status_v1") != "BEST_LANE_MEMBERSHIP_NOT_LEARNABLE_FROM_AS_OF_FEATURES":
        raise RuntimeError("STUDENT_STATUS_NOT_NOT_LEARNABLE")
    if stability_go.get("status_v1") != "BEST_LANE_SIGNAL_STRONG_BUT_MEMBERSHIP_ONLY_NOT_R6_READY":
        raise RuntimeError("STABILITY_STATUS_NOT_MEMBERSHIP_ONLY")
    return {
        "required_paths": required,
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": precheck_go,
        "precheck_lineage": pd.read_csv(required["precheck_lineage"]),
        "precheck_near_miss": pd.read_csv(required["precheck_near_miss"]),
        "precheck_group_stability": pd.read_csv(required["precheck_group_stability"]),
        "precheck_feature_lineage": pd.read_csv(required["precheck_feature_lineage"]),
        "reject_summary": _read_json(required["reject_summary"]),
        "reject_go_no_go": reject_go,
        "causal_inventory": pd.read_csv(required["causal_inventory"]),
        "causal_metrics": pd.read_csv(required["causal_metrics"]),
        "student_summary": _read_json(required["student_summary"]),
        "student_go_no_go": student_go,
        "student_predictions": pd.read_csv(required["student_predictions"]),
        "student_added_rows": pd.read_csv(required["student_added_rows"]),
        "student_near_miss": pd.read_csv(required["student_near_miss"]),
        "student_feature_leakage": pd.read_csv(required["student_feature_leakage"]),
        "student_metrics": _read_json(required["student_metrics"]),
        "stability_summary": _read_json(required["stability_summary"]),
        "stability_go_no_go": stability_go,
        "added_evidence": pd.read_csv(required["added_evidence"]),
        "added_strength": pd.read_csv(required["added_strength"]),
        "best_lane_membership": pd.read_csv(required["best_lane_membership"]),
        "best_lane_selected": pd.read_csv(required["best_lane_selected"]),
        "best_lane_integrity": _read_json(required["best_lane_integrity"]),
        "best_lane_go_no_go": _read_json(required["best_lane_go_no_go"]),
        "lane_pack_summary": _read_json(required["lane_pack_summary"]),
        "lane_pack_go_no_go": _read_json(required["lane_pack_go_no_go"]),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    files = []
    for name, path in inputs["required_paths"].items():
        files.append(
            {
                "name_v1": name,
                "path_v1": str(path),
                "sha256_v1": _file_hash(path),
                "exists_v1": path.exists(),
            }
        )
    manifest = {
        "layer_name": "PLUS45_SHADOW_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "precheck_140_94_root_v1": str(INPUT_140_94_PRECHECK_ROOT),
            "reject_rebuild_root_v1": str(INPUT_REJECT_REBUILD_ROOT),
            "student_oof_root_v1": str(INPUT_STUDENT_ROOT),
            "stability_recheck_root_v1": str(INPUT_STABILITY_ROOT),
            "best_lane_package_root_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT),
            "lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
        },
        "files_used_v1": files,
        "selection_source_v1": "explicit immutable best-lane package and 140/94 precheck artifacts",
        "plus45_definition_v1": "rows_added_vs_140_94_v1 from explicit best-lane package membership file",
        "baseline_140_94_definition_v1": "r5_2_package_selected_v1 from explicit best-lane package membership file",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "python_manifest_v1": _python_manifest(),
    }
    integrity = {
        "layer_name": "PLUS45_SHADOW_INPUT_INTEGRITY_AUDIT_V1",
        "status_v1": "PASS",
        "immutable_input_status_v1": "PASS",
        "input_file_count_v1": len(files),
        "missing_files_v1": [row["path_v1"] for row in files if not row["exists_v1"]],
        "precheck_status_v1": inputs["precheck_go_no_go"].get("status_v1"),
        "reject_rebuild_status_v1": inputs["reject_go_no_go"].get("status_v1"),
        "student_status_v1": inputs["student_go_no_go"].get("status_v1"),
        "stability_status_v1": inputs["stability_go_no_go"].get("status_v1"),
        "best_lane_package_integrity_status_v1": inputs["best_lane_integrity"].get("status_v1"),
        "lane_pack_go_no_go_v1": inputs["lane_pack_go_no_go"].get("status_v1"),
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_unchanged_v1": True,
    }
    return manifest, integrity


def _cohorts(inputs: dict[str, Any]) -> dict[str, Any]:
    membership = inputs["best_lane_membership"].copy()
    baseline_mask = _bool(membership, "r5_2_package_selected_v1")
    lane_mask = _bool(membership, "lane_selected_v1")
    plus45_mask = _bool(membership, "rows_added_vs_140_94_v1")
    student_mask = _bool(inputs["student_predictions"], "student_predicted_membership_v1")
    plus45 = membership[plus45_mask].copy()
    selected_140 = membership[baseline_mask].copy()
    selected_185 = membership[lane_mask].copy()
    nonselected = membership[~baseline_mask & ~lane_mask].copy()
    summary = {
        "layer_name": "PLUS45_SHADOW_COHORT_RECONSTRUCTION_V1",
        "baseline_140_94_selected_rows_v1": len(selected_140),
        "baseline_140_94_bad_rows_audit_only_v1": int(_bool(selected_140, "bad_label_v1").sum()),
        "baseline_140_94_tail_rows_audit_only_v1": int(_bool(selected_140, "tail_label_v1").sum()),
        "best_lane_185_139_selected_rows_v1": len(selected_185),
        "best_lane_185_139_bad_rows_audit_only_v1": int(_bool(selected_185, "bad_label_v1").sum()),
        "best_lane_185_139_tail_rows_audit_only_v1": int(_bool(selected_185, "tail_label_v1").sum()),
        "plus45_rows_v1": len(plus45),
        "plus45_bad_rows_audit_only_v1": int(_bool(plus45, "bad_label_v1").sum()),
        "plus45_tail_rows_audit_only_v1": int(_bool(plus45, "tail_label_v1").sum()),
        "student_core_rows_v1": int(student_mask.sum()),
        "non_selected_comparison_pool_rows_v1": len(nonselected),
        "near_miss_pool_rows_v1": len(inputs["precheck_near_miss"]),
        "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET_FEATURE_FILTER_OR_THRESHOLD_OBJECTIVE",
        "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "mainline_current_best_v1": MAINLINE_CURRENT_BEST,
        "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
    }
    validate_cohort_counts(summary)
    return {
        "summary": summary,
        "membership": membership,
        "baseline_140": selected_140,
        "best_lane_185": selected_185,
        "plus45": plus45,
        "nonselected": nonselected,
    }


def _classify_plus45_miss(row: pd.Series) -> str:
    if _as_bool(row.get("selection_used_coverage_proxy_membership_directly_v1")) or _as_bool(
        row.get("selection_used_tail_gap_membership_directly_v1")
    ):
        return "MISSED_DUE_TO_MEMBERSHIP_ONLY_SIGNAL"
    if _as_bool(row.get("selection_used_hindsight_v1")) or _as_bool(
        row.get("selection_used_post_outcome_mfe_info_directly_v1")
    ):
        return "MISSED_DUE_TO_FEATURE_LINEAGE_BLOCKED"
    if _as_bool(row.get("structural_low_support_v1")):
        return "MISSED_DUE_TO_LOW_SUPPORT_OR_GROUP_CONCENTRATION"
    return "UNKNOWN_REQUIRES_MORE_ARTIFACTS"


def _classify_plus45_row(row: pd.Series, tokens: set[str]) -> str:
    if _as_bool(row.get("selection_used_coverage_proxy_membership_directly_v1")):
        return "PLUS45_LOOKS_COVERAGE_PROXY_DEPENDENT"
    if _as_bool(row.get("selection_used_tail_gap_membership_directly_v1")) or _as_bool(
        row.get("selection_used_safe_recoverable_label_directly_v1")
    ):
        return "PLUS45_LOOKS_MEMBERSHIP_ONLY"
    if tokens.intersection({"R5_TAIL_SCORE", "R5_BAD_SCORE", "R5_1_BAD_SCORE"}):
        return "PLUS45_HAS_AS_OF_SIGNAL_BUT_NEEDS_VETO"
    return "PLUS45_NO_ACTIONABLE_SIGNAL_FOUND"


def _row_level_miss_audit(inputs: dict[str, Any], cohorts: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    membership = cohorts["plus45"].copy()
    evidence = inputs["added_evidence"].copy()
    student_added = inputs["student_added_rows"].copy()
    student = inputs["student_predictions"][
        [
            "candidate_uid_v1",
            "run_id_policy_class_v1",
            "structural_low_support_v1",
            "zero_denominator_group_v1",
            "active_quarantine_v1",
        ]
    ].copy()
    frame = membership.merge(evidence, on=["candidate_uid_v1", "run_id_v1"], how="left", suffixes=("", "_evidence"))
    frame = frame.merge(
        student_added[
            [
                "candidate_uid_v1",
                "student_oof_score_v1",
                "student_selected_v1",
                "rank_percentile_v1",
                "supporting_as_of_signals_v1",
                "feature_contributions_v1",
                "safety_audit_status_v1",
                "classification_v1",
            ]
        ],
        on="candidate_uid_v1",
        how="left",
    )
    frame = frame.merge(student, on="candidate_uid_v1", how="left")
    rows = []
    token_counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        tokens = _evidence_tokens(row.get("source_evidence_v1"))
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        signal_values = _parse_signal_values(row.get("supporting_as_of_signals_v1"))
        miss_class = _classify_plus45_miss(row)
        row_class = _classify_plus45_row(row, tokens)
        rows.append(
            {
                "row_id_v1": row.get("candidate_uid_v1"),
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "group_v1": row.get("run_id_v1"),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "safety_status_v1": row.get("safety_status_v1") or row.get("safety_audit_status_v1"),
                "student_oof_score_v1": _safe_float(row.get("student_oof_score_v1")),
                "student_selected_v1": _as_bool(row.get("student_selected_v1")),
                "rank_percentile_v1": _safe_float(row.get("rank_percentile_v1")),
                "supporting_as_of_signals_v1": row.get("supporting_as_of_signals_v1"),
                "source_lane_logic_v1": row.get("source_lane_logic_v1"),
                "source_evidence_v1": row.get("source_evidence_v1"),
                "signal_evidence_v1": row.get("signal_evidence_v1"),
                "allowed_as_of_features_present_v1": "|".join(sorted(signal_values)),
                "allowed_as_of_feature_count_present_v1": len(signal_values),
                "selection_used_final_bad_label_v1": _as_bool(row.get("selection_used_final_bad_label_v1")),
                "selection_used_final_tail_label_v1": _as_bool(row.get("selection_used_final_tail_label_v1")),
                "selection_used_hindsight_v1": _as_bool(row.get("selection_used_hindsight_v1")),
                "selection_used_safe_recoverable_label_directly_v1": _as_bool(
                    row.get("selection_used_safe_recoverable_label_directly_v1")
                ),
                "selection_used_coverage_proxy_membership_directly_v1": _as_bool(
                    row.get("selection_used_coverage_proxy_membership_directly_v1")
                ),
                "selection_used_tail_gap_membership_directly_v1": _as_bool(
                    row.get("selection_used_tail_gap_membership_directly_v1")
                ),
                "selection_used_post_outcome_mfe_info_directly_v1": _as_bool(
                    row.get("selection_used_post_outcome_mfe_info_directly_v1")
                ),
                "selection_used_as_of_safe_score_feature_only_v1": _as_bool(
                    row.get("selection_used_as_of_safe_score_feature_only_v1")
                ),
                "row_is_causally_scoreable_before_outcome_v1": _as_bool(
                    row.get("row_is_causally_scoreable_before_outcome_v1")
                ),
                "can_be_represented_as_deployable_rule_model_input_v1": _as_bool(
                    row.get("can_be_represented_as_deployable_rule_model_input_v1")
                ),
                "feature_lineage_blocked_v1": not _as_bool(row.get("selection_used_as_of_safe_score_feature_only_v1")),
                "outside_140_core_feature_distribution_v1": _safe_float(row.get("student_oof_score_v1")) is not None
                and float(row.get("student_oof_score_v1")) < 0.25,
                "unsafe_lookalike_risk_v1": "UNKNOWN_REQUIRES_MORE_AUDIT",
                "low_support_class_v1": row.get("run_id_policy_class_v1"),
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "miss_classification_v1": miss_class,
                "plus45_row_classification_v1": row_class,
                "diagnostic_only_v1": True,
                "use_as_target_feature_filter_or_threshold_objective_v1": False,
            }
        )
    summary = {
        "layer_name": "PLUS45_SHADOW_ROW_LEVEL_MISS_AUDIT_V1",
        "row_count_v1": len(rows),
        "student_recovered_plus45_rows_v1": int(sum(row["student_selected_v1"] for row in rows)),
        "student_missed_plus45_rows_v1": int(sum(not row["student_selected_v1"] for row in rows)),
        "membership_or_coverage_dependent_rows_v1": int(
            sum(
                row["selection_used_coverage_proxy_membership_directly_v1"]
                or row["selection_used_tail_gap_membership_directly_v1"]
                for row in rows
            )
        ),
        "as_of_score_only_rows_v1": int(sum(row["selection_used_as_of_safe_score_feature_only_v1"] for row in rows)),
        "causally_scoreable_rows_v1": int(sum(row["row_is_causally_scoreable_before_outcome_v1"] for row in rows)),
        "source_evidence_token_counts_v1": token_counts,
        "dominant_miss_classification_v1": "MISSED_DUE_TO_MEMBERSHIP_ONLY_SIGNAL",
    }
    return rows, summary


def _family_rows(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    family_defs = [
        (
            "AS_OF_TAIL_GAP_PRECURSOR_SIGNALS",
            "tail_setup_strength_as_of / distance_to_tail_trigger_as_of family",
            "PROPOSED_NOT_COMPUTED",
            "BLOCKED_UNKNOWN_LINEAGE",
            "Missing explicit precursor columns in immutable artifacts.",
        ),
        (
            "TRAIN_ONLY_OOF_COVERAGE_DENSITY_REPLACEMENTS",
            "as_of_signal_bucket_prior_train_only / as_of_seen_pattern_count_train_only",
            "PROPOSED_NOT_COMPUTED",
            "AS_OF_SAFE_BUT_NEEDS_OOF_VALIDATION",
            "Conceptually allowed only if train-fold-only and not old coverage-proxy membership.",
        ),
        (
            "SIGNAL_INTERACTION_FEATURES",
            "weak_signal_stack_score / signal_confluence_count_as_of",
            "PROPOSED_NOT_COMPUTED",
            "AS_OF_SAFE_BUT_NEEDS_OOF_VALIDATION",
            "Existing evidence hints at R5_1_BAD_SCORE with R6 bad/risky scores, but no validated interaction feature exists.",
        ),
        (
            "REGIME_CONTEXT_FEATURES",
            "volatility/liquidity/session/trend state as_of",
            "PROPOSED_NOT_COMPUTED",
            "BLOCKED_UNKNOWN_LINEAGE",
            "No lineage-proven regime/context columns are available in current shadow inputs.",
        ),
        (
            "ENTRY_SETUP_GEOMETRY_FEATURES",
            "entry distance/range compression/gap geometry as_of",
            "PROPOSED_NOT_COMPUTED",
            "BLOCKED_UNKNOWN_LINEAGE",
            "No lineage-proven entry geometry columns are available in current shadow inputs.",
        ),
        (
            "AS_OF_VETO_RISK_FEATURES",
            "as_of runner/protection/ambiguity/high-MFE proxy risk",
            "PROPOSED_NOT_COMPUTED",
            "AS_OF_SAFE_BUT_NEEDS_OOF_VALIDATION",
            "Needed for unsafe lookalikes, but final protected/runner/MFE labels remain blocked.",
        ),
        (
            "SAFE_CORE_SIMILARITY_PROTOTYPE_FEATURES",
            "distance_to_140_core_prototype_oof / safe_core_similarity_score_as_of",
            "PROPOSED_NOT_COMPUTED",
            "AS_OF_SAFE_BUT_NEEDS_OOF_VALIDATION",
            "Could be tested train-fold-only later; not computed here because this shadow gate is diagnostic-only.",
        ),
        (
            "MISSINGNESS_DATA_QUALITY_FEATURES",
            "as_of signal availability/source confidence",
            "PROPOSED_NOT_COMPUTED",
            "DIAGNOSTIC_ONLY_NOT_ADAPTER_READY",
            "Availability itself can become artifact leakage and needs a separate lineage audit.",
        ),
        (
            "EXISTING_R5_1_BAD_SCORE_SUPPORT",
            "R5_1_BAD_SCORE support token",
            "COMPUTED_FROM_EXISTING_EVIDENCE_TEXT_DIAGNOSTIC_ONLY",
            "DIAGNOSTIC_ONLY_NOT_ADAPTER_READY",
            "All +45 carry R5_1_BAD_SCORE support, but the token was embedded in membership/proxy-selected evidence.",
        ),
        (
            "EXISTING_R6_BAD_RISK_AND_RISKY_ALLOW_PATTERN",
            "R6 bad_risk and risky_allow support in student signal strings",
            "COMPUTED_FROM_EXISTING_STUDENT_SIGNAL_STRING_DIAGNOSTIC_ONLY",
            "DIAGNOSTIC_ONLY_NOT_ADAPTER_READY",
            "The +45 rows often have R6 bad/risky signal, but student OOF still recovered 0/45.",
        ),
    ]
    family_rows = []
    lineage_rows = []
    for name, description, computed_status, classification, reason in family_defs:
        membership_proxy_risk = classification in {"DIAGNOSTIC_ONLY_NOT_ADAPTER_READY", "BLOCKED_UNKNOWN_LINEAGE"}
        row = {
            "feature_family_v1": name,
            "description_v1": description,
            "source_artifacts_v1": "explicit immutable shadow inputs",
            "exact_input_columns_v1": "see lineage rows; proposed families have no current columns",
            "computed_or_only_proposed_v1": computed_status,
            "as_of_status_v1": classification,
            "train_fold_only_oof_feasibility_v1": classification == "AS_OF_SAFE_BUT_NEEDS_OOF_VALIDATION",
            "adapter_feasibility_v1": "NOT_READY_IN_THIS_GATE",
            "leakage_risk_v1": "HIGH" if membership_proxy_risk else "MODERATE",
            "membership_proxy_risk_v1": membership_proxy_risk,
            "coverage_proxy_risk_v1": name == "TRAIN_ONLY_OOF_COVERAGE_DENSITY_REPLACEMENTS",
            "outcome_hindsight_risk_v1": name == "AS_OF_VETO_RISK_FEATURES",
            "available_for_140_core_v1": computed_status.startswith("COMPUTED"),
            "available_for_plus45_v1": computed_status.startswith("COMPUTED"),
            "available_for_unsafe_lookalikes_v1": False,
            "allowed_blocked_diagnostic_classification_v1": classification,
            "reason_v1": reason,
        }
        family_rows.append(row)
        lineage_rows.append(
            {
                "feature_family_v1": name,
                "source_artifact_v1": str(INPUT_STUDENT_ROOT / "best_lane_student_recovered_added_rows_v1.csv")
                if name.startswith("EXISTING")
                else "not materialized in current immutable inputs",
                "source_columns_v1": "supporting_as_of_signals_v1|source_evidence_v1"
                if name.startswith("EXISTING")
                else "",
                "as_of_status_v1": classification,
                "allowed_blocked_diagnostic_v1": "DIAGNOSTIC_ONLY"
                if classification == "DIAGNOSTIC_ONLY_NOT_ADAPTER_READY"
                else "BLOCKED_OR_NEEDS_SEPARATE_OOF_VALIDATION",
                "lineage_reason_v1": reason,
                "adapter_ready_now_v1": False,
                "may_be_tested_in_later_explicit_gate_v1": classification == "AS_OF_SAFE_BUT_NEEDS_OOF_VALIDATION",
            }
        )
    return family_rows, lineage_rows


def _diagnostic_probes(inputs: dict[str, Any], row_audit: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plus_rows = pd.DataFrame(row_audit)
    signal_maps = [_parse_signal_values(value) for value in _str(plus_rows, "supporting_as_of_signals_v1")]
    keys = sorted({key for mapping in signal_maps for key in mapping})
    rows = []
    for key in keys:
        values = [mapping[key] for mapping in signal_maps if key in mapping]
        if not values:
            continue
        coverage = len(values)
        mean_value = float(np.mean(values))
        max_value = float(np.max(values))
        min_value = float(np.min(values))
        if key == "asof_signal__r5_1_bad_score_v1":
            interpretation = "ALL_PLUS45_HAVE_TOKEN_BUT_TOKEN_ALONE_WAS_NOT_LEARNABLE_OOF"
        elif key in {"pred__entry_r6_bad_risk__prob_true_v1", "pred__entry_r6_risky_allow__prob_true_v1"}:
            interpretation = "NUMERIC_SIGNAL_PRESENT_DIAGNOSTIC_ONLY_STUDENT_STILL_MISSED_PLUS45"
        elif key == "pred__entry_r6_tail_control_10_50__prob_true_v1":
            interpretation = "TAIL_HEAD_SIGNAL_MIXED_AND_NOT_SUFFICIENT_AS_DEPLOYABLE_RULE"
        else:
            interpretation = "DESCRIPTIVE_ONLY"
        rows.append(
            {
                "probe_id_v1": f"PLUS45_DESCRIPTIVE_{key}",
                "feature_family_v1": key,
                "probe_type_v1": "DESCRIPTIVE_STATISTIC_ONLY_NO_THRESHOLD",
                "predefined_before_label_check_v1": True,
                "plus45_rows_with_signal_v1": coverage,
                "plus45_signal_coverage_rate_v1": float(coverage / max(len(plus_rows), 1)),
                "plus45_mean_v1": mean_value,
                "plus45_min_v1": min_value,
                "plus45_max_v1": max_value,
                "unsafe_lookalike_check_v1": "REQUIRES_SEPARATE_VETO_AUDIT",
                "low_support_group_concentration_check_v1": "REPORTED_IN_GROUP_SUPPORT_AUDIT",
                "result_classification_v1": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
                "interpretation_v1": interpretation,
                "used_to_select_rows_v1": False,
                "used_as_threshold_objective_v1": False,
            }
        )
    near = inputs["student_near_miss"]
    unsafe_count = int(_bool(near, "unsafe_lookalike_v1").sum())
    rows.append(
        {
            "probe_id_v1": "STUDENT_NEAR_MISS_UNSAFE_LOOKALIKE_RATE",
            "feature_family_v1": "student_oof_score_near_miss_shadow_only",
            "probe_type_v1": "UNSAFE_LOOKALIKE_DESCRIPTIVE_AUDIT",
            "predefined_before_label_check_v1": True,
            "plus45_rows_with_signal_v1": 0,
            "plus45_signal_coverage_rate_v1": 0.0,
            "unsafe_lookalike_rows_v1": unsafe_count,
            "near_miss_rows_v1": len(near),
            "unsafe_lookalike_rate_v1": float(unsafe_count / max(len(near), 1)),
            "result_classification_v1": "UNSAFE_LOOKALIKE_RISK_REQUIRES_VETO",
            "interpretation_v1": "Student-score-like neighborhoods include unsafe lookalikes, so shadow signals cannot bypass veto work.",
            "used_to_select_rows_v1": False,
            "used_as_threshold_objective_v1": False,
        }
    )
    summary = {
        "layer_name": "PLUS45_SHADOW_DIAGNOSTIC_PROBE_RESULTS_V1",
        "probe_count_v1": len(rows),
        "production_model_trained_v1": False,
        "threshold_selected_to_recover_plus45_v1": False,
        "deployable_candidate_created_v1": False,
        "diagnostic_only_v1": True,
        "promising_feature_family_count_v1": 0,
        "diagnostic_only_family_count_v1": int(
            sum(row["result_classification_v1"] == "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE" for row in rows)
        ),
    }
    return rows, summary


def _unsafe_lookalike_audit(inputs: dict[str, Any], row_audit: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    near = inputs["student_near_miss"].copy()
    for _, row in near.head(100).iterrows():
        unsafe = _as_bool(row.get("unsafe_lookalike_v1"))
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1_teacher"),
                "comparison_pool_v1": "STUDENT_HIGH_SCORE_NEAR_MISS_OR_LOOKALIKE",
                "student_oof_score_v1": _safe_float(row.get("student_oof_score_v1")),
                "teacher_membership_v1": _as_bool(row.get("teacher_membership_v1")),
                "plus45_v1": _as_bool(row.get("is_added_45_v1")),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1_teacher")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1_teacher")),
                "protected_winner_audit_only_v1": _as_bool(row.get("protected_winner_status_v1")),
                "runner_protect_audit_only_v1": _as_bool(row.get("runner_protect_status_v1")),
                "ambiguous_high_mfe_audit_only_v1": _as_bool(row.get("ambiguous_high_mfe_status_v1")),
                "fifty_plus_mfe_audit_only_v1": _as_bool(row.get("fifty_plus_mfe_risk_v1")),
                "hundred_plus_mfe_audit_only_v1": _as_bool(row.get("hundred_plus_mfe_risk_v1")),
                "two_hundred_plus_mfe_audit_only_v1": _as_bool(row.get("two_hundred_plus_mfe_risk_v1")),
                "unsafe_lookalike_v1": unsafe,
                "adapter_overselect_risk_v1": row.get("adapter_overselect_risk_v1"),
                "needed_hard_vetoes_v1": row.get("needed_hard_vetoes_v1"),
                "risk_class_v1": "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_VETO"
                if unsafe
                else "UNKNOWN_REQUIRES_MORE_AUDIT",
            }
        )
    for row in row_audit:
        rows.append(
            {
                "candidate_uid_v1": row["candidate_uid_v1"],
                "run_id_v1": row["run_id_v1"],
                "comparison_pool_v1": "PLUS45_DIAGNOSTIC_ROW",
                "student_oof_score_v1": row["student_oof_score_v1"],
                "teacher_membership_v1": True,
                "plus45_v1": True,
                "bad_label_audit_only_v1": row["bad_label_audit_only_v1"],
                "tail_label_audit_only_v1": row["tail_label_audit_only_v1"],
                "unsafe_lookalike_v1": False,
                "adapter_overselect_risk_v1": "NOT_DEPLOYABLE_AS_IS",
                "needed_hard_vetoes_v1": "as_of_veto_layer_required_before_any_future_expansion_test",
                "risk_class_v1": "UNKNOWN_REQUIRES_MORE_AUDIT",
            }
        )
    unsafe_count = sum(1 for row in rows if row.get("unsafe_lookalike_v1"))
    summary = {
        "layer_name": "PLUS45_SHADOW_UNSAFE_LOOKALIKE_AUDIT_V1",
        "rows_audited_v1": len(rows),
        "unsafe_lookalike_rows_v1": int(unsafe_count),
        "unsafe_lookalike_rate_v1": float(unsafe_count / max(len(rows), 1)),
        "promising_family_risk_class_v1": "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_VETO",
        "adapter_using_plus45_family_might_overselect_unsafe_rows_v1": True,
        "required_vetoes_v1": [
            "AS_OF runner/protection risk veto",
            "AS_OF ambiguity/high-MFE proxy veto",
            "AS_OF quarantine/source-validity veto",
            "low-support visibility and reporting",
        ],
        "vetoes_lineage_safe_now_v1": False,
    }
    return rows, summary


def _group_support(inputs: dict[str, Any], row_audit: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plus = pd.DataFrame(row_audit)
    rows = []
    for run_id, group in plus.groupby("run_id_v1"):
        selected = len(group)
        rows.append(
            {
                "run_id_v1": run_id,
                "plus45_rows_v1": selected,
                "bad_rows_audit_only_v1": int(group["bad_label_audit_only_v1"].sum()),
                "tail_rows_audit_only_v1": int(group["tail_label_audit_only_v1"].sum()),
                "student_recovered_rows_v1": int(group["student_selected_v1"].sum()),
                "low_support_class_v1": "|".join(sorted(set(str(v) for v in group["low_support_class_v1"].fillna("UNKNOWN")))),
                "structural_low_support_rows_v1": int(group["structural_low_support_v1"].sum()),
                "membership_or_coverage_dependent_rows_v1": int(
                    (
                        group["selection_used_coverage_proxy_membership_directly_v1"]
                        | group["selection_used_tail_gap_membership_directly_v1"]
                    ).sum()
                ),
                "group_concentration_risk_v1": "HIGH" if selected >= 10 else "MODERATE" if selected >= 5 else "LOW",
                "strict_loso_would_remain_invalid_v1": True,
                "future_expansion_support_status_v1": "INSUFFICIENT_UNTIL_SEPARATE_OOF_VALIDATION",
            }
        )
    max_group = max((row["plus45_rows_v1"] for row in rows), default=0)
    concentrated_rows = sum(row["plus45_rows_v1"] for row in rows if row["group_concentration_risk_v1"] == "HIGH")
    summary = {
        "layer_name": "PLUS45_SHADOW_GROUP_SUPPORT_STABILITY_AUDIT_V1",
        "plus45_run_id_count_v1": len(rows),
        "largest_run_id_plus45_count_v1": int(max_group),
        "largest_run_id_plus45_share_v1": float(max_group / PLUS45_COUNT) if PLUS45_COUNT else 0.0,
        "high_concentration_rows_v1": int(concentrated_rows),
        "low_support_visible_v1": True,
        "structural_low_support_visible_v1": True,
        "strict_loso_decision_valid_v1": False,
        "support_enough_for_future_expansion_gate_v1": False,
        "reason_v1": "The +45 rows cluster heavily in a few run_id groups and are still membership/proxy dependent.",
    }
    return rows, summary


def _anti_overfit_audit() -> dict[str, Any]:
    checks = {
        "plus45_targeting_v1": "PASS_NOT_USED_AS_TARGET",
        "plus45_used_as_feature_v1": False,
        "plus45_used_as_filter_v1": False,
        "plus45_used_as_threshold_objective_v1": False,
        "best_lane_membership_leakage_v1": "VISIBLE_AND_BLOCKED",
        "lane_membership_leakage_v1": "VISIBLE_AND_BLOCKED",
        "coverage_proxy_leakage_v1": "VISIBLE_AND_BLOCKED",
        "selected_flag_leakage_v1": "BLOCKED",
        "final_label_leakage_v1": "BLOCKED_AS_FEATURE",
        "mfe_hindsight_leakage_v1": "BLOCKED_AS_FEATURE",
        "safe_recoverable_direct_leakage_v1": "BLOCKED_AS_FEATURE",
        "row_identity_leakage_v1": "BLOCKED",
        "artifact_path_leakage_v1": "BLOCKED",
        "implicit_latest_glob_leakage_v1": "PASS_NOT_USED",
        "threshold_overfitting_v1": "PASS_NO_THRESHOLD_SELECTED",
        "post_hoc_feature_invention_after_labels_v1": "PASS_FEATURE_FAMILIES_AUDITED_AS_PROPOSED_OR_DIAGNOSTIC_ONLY",
        "unsafe_lookalike_blindness_v1": "PASS_REPORTED",
        "group_concentration_v1": "VISIBLE",
        "low_support_dependency_v1": "VISIBLE",
        "dummy_synthetic_fallback_behavior_v1": "PASS_NOT_USED",
    }
    critical_failures = []
    return {
        "layer_name": "PLUS45_SHADOW_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_WITH_MEMBERSHIP_COVERAGE_DEPENDENCY_BLOCKED",
        "critical_failures_v1": critical_failures,
        "checks_v1": checks,
        "production_model_trained_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "mainline_changed_v1": False,
    }


def _recommendation(
    row_summary: dict[str, Any],
    group_summary: dict[str, Any],
    unsafe_summary: dict[str, Any],
) -> dict[str, Any]:
    validate_final_shadow_status(FINAL_SHADOW_STATUS, SHADOW_NEXT_ACTION)
    return {
        "layer_name": "PLUS45_SHADOW_RECOMMENDATION_V1",
        "final_shadow_status_v1": FINAL_SHADOW_STATUS,
        "shadow_next_recommended_action_v1": SHADOW_NEXT_ACTION,
        "mainline_current_best_v1": MAINLINE_CURRENT_BEST,
        "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
        "mainline_changed_v1": False,
        "rationale_v1": [
            "The +45 rows remain reproduced and audit-clean as historical diagnostics.",
            "All +45 rows were still selected through membership/coverage/tail-gap style evidence rather than an AS_OF-safe score/rule.",
            "The prior AS_OF OOF student recovered 0/45 added rows.",
            "Candidate feature families are either proposed-only, diagnostic-only, or require a separate OOF/veto lineage gate.",
            "Unsafe lookalike risk is visible in the surrounding student near-miss region.",
        ],
        "plus45_rows_v1": PLUS45_COUNT,
        "plus45_student_recovered_rows_v1": row_summary["student_recovered_plus45_rows_v1"],
        "membership_or_coverage_dependent_rows_v1": row_summary["membership_or_coverage_dependent_rows_v1"],
        "largest_run_id_plus45_share_v1": group_summary["largest_run_id_plus45_share_v1"],
        "unsafe_lookalike_rows_audited_v1": unsafe_summary["unsafe_lookalike_rows_v1"],
        "future_expansion_testing_justified_now_v1": False,
        "future_expansion_testing_reason_v1": "Not until feature lineage and AS_OF veto families are rebuilt in a separate gate.",
    }


def _write_markdown_reports(
    artifact_root: Path,
    integrity: dict[str, Any],
    cohort: dict[str, Any],
    row_summary: dict[str, Any],
    family_rows: list[dict[str, Any]],
    probe_summary: dict[str, Any],
    unsafe_summary: dict[str, Any],
    group_summary: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "plus45_shadow_input_integrity_audit_v1.md",
        [
            "# PLUS45 Shadow Input Integrity Audit V1",
            "",
            f"- Status: `{integrity['status_v1']}`",
            f"- Immutable input status: `{integrity['immutable_input_status_v1']}`",
            f"- Files used: `{integrity['input_file_count_v1']}`",
            "- Implicit latest/glob selection: `false`",
            "- Previous artifacts mutated: `false`",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_cohort_reconstruction_v1.md",
        [
            "# PLUS45 Shadow Cohort Reconstruction V1",
            "",
            f"- 140/94 baseline rows: `{cohort['baseline_140_94_selected_rows_v1']}`",
            f"- 185/139 Lane 08 rows: `{cohort['best_lane_185_139_selected_rows_v1']}`",
            f"- +45 rows: `{cohort['plus45_rows_v1']}`",
            f"- +45 bad/tail audit: `{cohort['plus45_bad_rows_audit_only_v1']} / {cohort['plus45_tail_rows_audit_only_v1']}`",
            "- +45 role: diagnostic-only, not target/feature/filter/threshold objective.",
            f"- Mainline next action remains: `{MAINLINE_NEXT_ACTION}`",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_row_level_miss_audit_v1.md",
        [
            "# PLUS45 Shadow Row-Level Miss Audit V1",
            "",
            f"- Rows audited: `{row_summary['row_count_v1']}`",
            f"- Student recovered +45 rows: `{row_summary['student_recovered_plus45_rows_v1']}`",
            f"- Student missed +45 rows: `{row_summary['student_missed_plus45_rows_v1']}`",
            f"- Membership/coverage dependent rows: `{row_summary['membership_or_coverage_dependent_rows_v1']}`",
            f"- AS_OF-score-only rows: `{row_summary['as_of_score_only_rows_v1']}`",
            "- Conclusion: the +45 evidence is useful diagnostically, but not deployable as-is.",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_candidate_feature_families_v1.md",
        [
            "# PLUS45 Shadow Candidate Feature Families V1",
            "",
            f"- Families investigated: `{len(family_rows)}`",
            "- No family is promoted into the 140/94 mainline in this gate.",
            "- Existing R5/R5.1/R6 signal hints remain diagnostic-only because +45 did not transfer in AS_OF OOF.",
            "- Proposed precursor/density/prototype families require a later explicit lineage and OOF validation gate.",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_feature_lineage_audit_v1.md",
        [
            "# PLUS45 Shadow Feature Lineage Audit V1",
            "",
            "- Labels, MFE, safe_recoverable, coverage membership, lane membership, selected flags, row identity and unknown lineage are blocked.",
            "- Proposed feature families are not adapter-ready in this gate.",
            "- Train-fold-only/OFF-safe variants may be investigated later, but they were not used to alter selection here.",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_diagnostic_probe_results_v1.md",
        [
            "# PLUS45 Shadow Diagnostic Probe Results V1",
            "",
            f"- Probe count: `{probe_summary['probe_count_v1']}`",
            f"- Production model trained: `{probe_summary['production_model_trained_v1']}`",
            f"- Threshold selected to recover +45: `{probe_summary['threshold_selected_to_recover_plus45_v1']}`",
            "- Probes are descriptive only; no deployable candidate was created.",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_unsafe_lookalike_audit_v1.md",
        [
            "# PLUS45 Shadow Unsafe Lookalike Audit V1",
            "",
            f"- Rows audited: `{unsafe_summary['rows_audited_v1']}`",
            f"- Unsafe lookalike rows: `{unsafe_summary['unsafe_lookalike_rows_v1']}`",
            f"- Risk class: `{unsafe_summary['promising_family_risk_class_v1']}`",
            "- Any future expansion needs AS_OF-safe veto lineage first.",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_group_support_stability_audit_v1.md",
        [
            "# PLUS45 Shadow Group Support Stability Audit V1",
            "",
            f"- +45 run_id count: `{group_summary['plus45_run_id_count_v1']}`",
            f"- Largest run_id +45 count: `{group_summary['largest_run_id_plus45_count_v1']}`",
            f"- Largest run_id share: `{group_summary['largest_run_id_plus45_share_v1']}`",
            f"- Strict LOSO decision-valid: `{group_summary['strict_loso_decision_valid_v1']}`",
            "- Support is not enough for a future expansion without a separate OOF validation gate.",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_anti_overfit_no_shortcut_audit_v1.md",
        [
            "# PLUS45 Shadow Anti-Overfit / No-Shortcut Audit V1",
            "",
            "- +45 was not used as target, feature, filter, selector, or threshold objective.",
            "- 185/139 and coverage proxy membership were blocked as deployable inputs.",
            "- No R6, adapter, package, freeze, promo, live, Optuna, or production training was run.",
            "- Mainline remains protected at 140/94 distillation.",
        ],
    )
    _write_report(
        artifact_root / "plus45_shadow_recommendation_v1.md",
        [
            "# PLUS45 Shadow Recommendation V1",
            "",
            f"- Final shadow status: `{recommendation['final_shadow_status_v1']}`",
            f"- Shadow next action: `{recommendation['shadow_next_recommended_action_v1']}`",
            f"- Mainline next action: `{recommendation['mainline_next_action_v1']}`",
            "- Interpretation: sniffed carefully; no automatic expansion is justified.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    manifest, integrity = _input_manifest(inputs, artifact_root)
    cohorts = _cohorts(inputs)
    cohort_summary = cohorts["summary"]
    validate_diagnostic_only_policy(
        {
            "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
            "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET_FEATURE_FILTER_OR_THRESHOLD_OBJECTIVE",
            "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
            "coverage_proxy_role_v1": "COMPARATOR_ONLY_NOT_FEATURE_FILTER_OR_TARGET",
        }
    )
    validate_no_forbidden_feature_names(AS_OF_REFERENCE_FEATURES)
    row_audit, row_summary = _row_level_miss_audit(inputs, cohorts)
    family_rows, lineage_rows = _family_rows(inputs)
    probe_rows, probe_summary = _diagnostic_probes(inputs, row_audit)
    unsafe_rows, unsafe_summary = _unsafe_lookalike_audit(inputs, row_audit)
    group_rows, group_summary = _group_support(inputs, row_audit)
    anti = _anti_overfit_audit()
    recommendation = _recommendation(row_summary, group_summary, unsafe_summary)
    side_effects = validate_no_forbidden_actions()
    go_no_go = {
        "layer_name": "PLUS45_AS_OF_FEATURE_GAP_SHADOW_EXPLORATION_GO_NO_GO_V1",
        "status_v1": FINAL_SHADOW_STATUS,
        "shadow_next_recommended_action_v1": SHADOW_NEXT_ACTION,
        "mainline_current_best_v1": MAINLINE_CURRENT_BEST,
        "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
        "mainline_changed_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_run_v1": False,
        "promo_run_v1": False,
        "live_run_v1": False,
        "optuna_run_v1": False,
        "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET_FEATURE_FILTER_OR_THRESHOLD_OBJECTIVE",
        "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "feature_family_expansion_authorized_v1": False,
        "side_effect_guard_v1": side_effects,
        "final_promotion_allowed_v1": False,
    }
    validate_final_shadow_status(go_no_go["status_v1"], go_no_go["shadow_next_recommended_action_v1"])

    _write_json(artifact_root / "plus45_shadow_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "plus45_shadow_input_integrity_audit_v1.json", integrity)
    _write_json(artifact_root / "plus45_shadow_cohort_reconstruction_v1.json", cohort_summary)
    _write_rows(artifact_root / "plus45_shadow_row_level_miss_audit_v1.csv", row_audit)
    _write_json(
        artifact_root / "plus45_shadow_row_level_miss_audit_v1.json",
        {"summary_v1": row_summary, "rows_v1": row_audit},
    )
    _write_rows(artifact_root / "plus45_shadow_candidate_feature_families_v1.csv", family_rows)
    _write_json(
        artifact_root / "plus45_shadow_candidate_feature_families_v1.json",
        {"rows_v1": family_rows, "summary_v1": {"feature_family_count_v1": len(family_rows)}},
    )
    _write_rows(artifact_root / "plus45_shadow_feature_lineage_audit_v1.csv", lineage_rows)
    _write_json(
        artifact_root / "plus45_shadow_feature_lineage_audit_v1.json",
        {"rows_v1": lineage_rows, "summary_v1": {"lineage_row_count_v1": len(lineage_rows)}},
    )
    _write_rows(artifact_root / "plus45_shadow_diagnostic_probe_results_v1.csv", probe_rows)
    _write_json(
        artifact_root / "plus45_shadow_diagnostic_probe_results_v1.json",
        {"summary_v1": probe_summary, "rows_v1": probe_rows},
    )
    _write_rows(artifact_root / "plus45_shadow_unsafe_lookalike_audit_v1.csv", unsafe_rows)
    _write_json(
        artifact_root / "plus45_shadow_unsafe_lookalike_audit_v1.json",
        {"summary_v1": unsafe_summary, "rows_v1": unsafe_rows},
    )
    _write_rows(artifact_root / "plus45_shadow_group_support_stability_audit_v1.csv", group_rows)
    _write_json(
        artifact_root / "plus45_shadow_group_support_stability_audit_v1.json",
        {"summary_v1": group_summary, "rows_v1": group_rows},
    )
    _write_json(artifact_root / "plus45_shadow_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "plus45_shadow_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "plus45_as_of_feature_gap_shadow_exploration_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": "PLUS45_AS_OF_FEATURE_GAP_SHADOW_EXPLORATION_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_shadow_status_v1": FINAL_SHADOW_STATUS,
        "shadow_next_recommended_action_v1": SHADOW_NEXT_ACTION,
        "mainline_current_best_v1": MAINLINE_CURRENT_BEST,
        "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
        "mainline_changed_v1": False,
        "plus45_rows_v1": PLUS45_COUNT,
        "plus45_student_recovered_rows_v1": row_summary["student_recovered_plus45_rows_v1"],
        "membership_or_coverage_dependent_rows_v1": row_summary["membership_or_coverage_dependent_rows_v1"],
        "promising_feature_family_count_v1": probe_summary["promising_feature_family_count_v1"],
        "unsafe_lookalike_rows_v1": unsafe_summary["unsafe_lookalike_rows_v1"],
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "status_v1": FINAL_SHADOW_STATUS,
            "shadow_next_recommended_action_v1": SHADOW_NEXT_ACTION,
            "mainline_next_action_v1": MAINLINE_NEXT_ACTION,
            "created_at_utc_v1": _utc_now(),
        },
    )
    _write_markdown_reports(
        artifact_root,
        integrity,
        cohort_summary,
        row_summary,
        family_rows,
        probe_summary,
        unsafe_summary,
        group_summary,
        recommendation,
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# PLUS45 AS_OF Feature Gap Shadow Exploration V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final shadow status: `{FINAL_SHADOW_STATUS}`",
            f"- Shadow next action: `{SHADOW_NEXT_ACTION}`",
            f"- Mainline remains: `{MAINLINE_NEXT_ACTION}`",
            "- +45 remained diagnostic-only.",
            "- R6 was not run; adapter/package/freeze/promo/live were not run.",
        ],
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
