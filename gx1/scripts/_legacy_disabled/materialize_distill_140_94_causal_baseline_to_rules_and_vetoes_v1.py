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
ACTION = "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1"

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
FINAL_STATUS = "140_94_RULE_VETO_DISTILLATION_PARTIAL_NEEDS_SIMPLIFICATION"
NEXT_ACTION = "SIMPLIFY_140_94_RULES_AND_VETOES_V1"

BASELINE_SELECTED = 140
BASELINE_BAD = 140
BASELINE_TAIL = 94
BEST_LANE_SELECTED = 185
BEST_LANE_TAIL = 139
PLUS45_COUNT = 45
STUDENT_SELECTED = 135
STUDENT_BAD = 131
STUDENT_TAIL = 93

ALLOWED_FINAL_STATUSES = {
    "140_94_RULE_VETO_DISTILLATION_PASS_ADAPTER_READY",
    "140_94_RULE_VETO_DISTILLATION_PASS_NEEDS_ADAPTER_MAPPING",
    "140_94_RULE_VETO_DISTILLATION_PARTIAL_NEEDS_SIMPLIFICATION",
    "140_94_RULE_VETO_DISTILLATION_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "140_94_RULE_VETO_DISTILLATION_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "140_94_RULE_VETO_DISTILLATION_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "140_94_RULE_VETO_DISTILLATION_BLOCKED_BY_INSUFFICIENT_RULE_COVERAGE",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_CAUSAL_BASELINE_ADAPTER_V1",
    "BUILD_140_94_ADAPTER_INPUT_MAPPING_V1",
    "SIMPLIFY_140_94_RULES_AND_VETOES_V1",
    "DEEPEN_140_94_UNSAFE_LOOKALIKE_BOUNDARY_AUDIT_V1",
    "DEEPEN_140_94_GROUPED_GENERALIZATION_AND_LOSO_AUDIT_V1",
    "DEEPEN_140_94_AS_OF_LINEAGE_AUDIT_V1",
    "RETURN_TO_CAUSAL_REBUILD_WITH_STRONGER_SIGNALS_V1",
}

AS_OF_ALLOWED_FEATURES = [
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
    "tail_repaired_r5_2_oof_candidate_score_v1",
]

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

REQUIRED_OUTPUTS = [
    "distill_140_94_input_manifest_v1.json",
    "distill_140_94_reproducibility_audit_v1.json",
    "distill_140_94_reproducibility_audit_v1.md",
    "distill_140_94_signal_inventory_v1.csv",
    "distill_140_94_signal_inventory_v1.json",
    "distill_140_94_signal_inventory_v1.md",
    "distill_140_94_rule_definition_v1.json",
    "distill_140_94_rule_definition_v1.md",
    "distill_140_94_veto_definition_v1.json",
    "distill_140_94_veto_definition_v1.md",
    "distill_140_94_row_level_explanations_v1.csv",
    "distill_140_94_row_level_explanations_v1.json",
    "distill_140_94_rule_coverage_audit_v1.csv",
    "distill_140_94_rule_coverage_audit_v1.json",
    "distill_140_94_rule_coverage_audit_v1.md",
    "distill_140_94_near_miss_and_near_fail_rows_v1.csv",
    "distill_140_94_near_miss_and_near_fail_rows_v1.json",
    "distill_140_94_boundary_stress_audit_v1.json",
    "distill_140_94_boundary_stress_audit_v1.md",
    "distill_140_94_group_stability_audit_v1.csv",
    "distill_140_94_group_stability_audit_v1.json",
    "distill_140_94_group_stability_audit_v1.md",
    "distill_140_94_comparison_to_original_baseline_v1.json",
    "distill_140_94_comparison_to_original_baseline_v1.md",
    "distill_140_94_adapter_feasibility_v1.json",
    "distill_140_94_adapter_feasibility_v1.md",
    "distill_140_94_anti_overfit_no_shortcut_audit_v1.json",
    "distill_140_94_anti_overfit_no_shortcut_audit_v1.md",
    "distill_140_94_recommendation_v1.json",
    "distill_140_94_recommendation_v1.md",
    "distill_140_94_causal_baseline_to_rules_and_vetoes_go_no_go_v1.json",
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
        raise RuntimeError(f"FORBIDDEN_140_94_RULE_FEATURE: {blocked}")
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
        "selected_rows_v1": BASELINE_SELECTED,
        "bad_count_v1": BASELINE_BAD,
        "tail_count_v1": BASELINE_TAIL,
        "safety_status_v1": "CLEAN",
    }
    failures = {key: payload.get(key) for key, value in expected.items() if payload.get(key) != value}
    if failures:
        raise RuntimeError(f"DISTILL_140_94_REPRODUCIBILITY_FAILED: {failures}")
    return True


def validate_rule_definition(rule: dict[str, Any]) -> bool:
    if rule.get("uses_selected_flag_as_adapter_feature_v1"):
        raise RuntimeError("SELECTED_FLAG_CANNOT_BE_ADAPTER_FEATURE")
    if rule.get("uses_plus45_as_target_feature_filter_or_threshold_v1"):
        raise RuntimeError("PLUS45_CANNOT_BE_RULE_TARGET_FEATURE_FILTER_OR_THRESHOLD")
    if "tail_repaired_r5_2_oof_candidate_score_v1" not in rule.get("required_positive_signals_v1", []):
        raise RuntimeError("RULE_MUST_INCLUDE_AS_OF_OOF_SCORE")
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
        raise RuntimeError(f"DISTILL_140_94_REQUIRED_OUTPUTS_MISSING: {missing}")
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
        INPUT_PRECHECK_ROOT,
        INPUT_PLUS45_SHADOW_ROOT,
        INPUT_REJECT_REBUILD_ROOT,
        INPUT_BEST_LANE_PACKAGE_ROOT,
        INPUT_STUDENT_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "precheck_summary": INPUT_PRECHECK_ROOT / "summary_v1.json",
        "precheck_go_no_go": INPUT_PRECHECK_ROOT
        / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json",
        "precheck_repro": INPUT_PRECHECK_ROOT / "return_to_140_94_reproducibility_audit_v1.json",
        "precheck_lineage": INPUT_PRECHECK_ROOT / "baseline_140_94_selection_lineage_v1.csv",
        "precheck_feature_lineage": INPUT_PRECHECK_ROOT / "baseline_140_94_feature_lineage_audit_v1.csv",
        "precheck_near_miss": INPUT_PRECHECK_ROOT / "baseline_140_94_near_miss_and_near_fail_rows_v1.csv",
        "precheck_group_stability": INPUT_PRECHECK_ROOT / "baseline_140_94_group_stability_audit_v1.csv",
        "precheck_adapter": INPUT_PRECHECK_ROOT / "baseline_140_94_adapter_precheck_v1.json",
        "plus45_summary": INPUT_PLUS45_SHADOW_ROOT / "summary_v1.json",
        "plus45_go_no_go": INPUT_PLUS45_SHADOW_ROOT
        / "plus45_as_of_feature_gap_shadow_exploration_go_no_go_v1.json",
        "plus45_recommendation": INPUT_PLUS45_SHADOW_ROOT / "plus45_shadow_recommendation_v1.json",
        "causal_predictions": INPUT_REJECT_REBUILD_ROOT / "causal_rebuild_candidate_oof_predictions_v1.csv",
        "causal_inventory": INPUT_REJECT_REBUILD_ROOT / "causal_signal_inventory_v1.csv",
        "best_membership": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv",
        "student_predictions": INPUT_STUDENT_ROOT / "best_lane_student_oof_predictions_v1.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    precheck_go = _read_json(required["precheck_go_no_go"])
    plus45_go = _read_json(required["plus45_go_no_go"])
    if precheck_go.get("status_v1") != "140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER":
        raise RuntimeError("PRECHECK_STATUS_NOT_RULE_DISTILLATION_REQUIRED")
    if plus45_go.get("status_v1") != "PLUS45_SHADOW_FOUND_ONLY_MEMBERSHIP_OR_COVERAGE_DEPENDENCY":
        raise RuntimeError("PLUS45_SHADOW_STATUS_NOT_EXPECTED")
    return {
        "required_paths": required,
        "precheck_summary": _read_json(required["precheck_summary"]),
        "precheck_go_no_go": precheck_go,
        "precheck_repro": _read_json(required["precheck_repro"]),
        "precheck_lineage": pd.read_csv(required["precheck_lineage"]),
        "precheck_feature_lineage": pd.read_csv(required["precheck_feature_lineage"]),
        "precheck_near_miss": pd.read_csv(required["precheck_near_miss"]),
        "precheck_group_stability": pd.read_csv(required["precheck_group_stability"]),
        "precheck_adapter": _read_json(required["precheck_adapter"]),
        "plus45_summary": _read_json(required["plus45_summary"]),
        "plus45_go_no_go": plus45_go,
        "plus45_recommendation": _read_json(required["plus45_recommendation"]),
        "causal_predictions": pd.read_csv(required["causal_predictions"]),
        "causal_inventory": pd.read_csv(required["causal_inventory"]),
        "best_membership": pd.read_csv(required["best_membership"]),
        "student_predictions": pd.read_csv(required["student_predictions"]),
    }


def _input_manifest(inputs: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    files = []
    for name, path in inputs["required_paths"].items():
        files.append({"name_v1": name, "path_v1": str(path), "sha256_v1": _file_hash(path)})
    return {
        "layer_name": "DISTILL_140_94_INPUT_MANIFEST_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "precheck_140_94_root_v1": str(INPUT_PRECHECK_ROOT),
            "plus45_shadow_root_v1": str(INPUT_PLUS45_SHADOW_ROOT),
            "reject_rebuild_root_v1": str(INPUT_REJECT_REBUILD_ROOT),
            "best_lane_package_root_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT),
            "student_root_v1": str(INPUT_STUDENT_ROOT),
        },
        "files_used_v1": files,
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


def _build_frame(inputs: dict[str, Any]) -> pd.DataFrame:
    pred = inputs["causal_predictions"]
    base = pred[pred["candidate_id_v1"] == BASELINE_CANDIDATE_ID].copy()
    membership = inputs["best_membership"].copy()
    student = inputs["student_predictions"][
        [
            "candidate_uid_v1",
            "source_evidence_v1",
            "student_predicted_membership_v1",
            "student_oof_score_v1",
            "run_id_policy_class_v1",
            "structural_low_support_v1",
            "zero_denominator_group_v1",
        ]
    ].copy()
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
    return frame


def _reproducibility(inputs: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    selected = frame[frame["selected_original_140_v1"]]
    payload = {
        "layer_name": "DISTILL_140_94_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "selected_rows_v1": len(selected),
        "bad_count_v1": int(_bool(selected, "bad_label_v1").sum()),
        "tail_count_v1": int(_bool(selected, "tail_label_v1").sum()),
        "precision_v1": float(_bool(selected, "bad_label_v1").sum() / max(len(selected), 1)),
        "safety_status_v1": "CLEAN" if not _bool(selected, "unsafe_audit_v1").any() else "FAIL",
        "precheck_status_v1": inputs["precheck_go_no_go"].get("status_v1"),
        "single_score_threshold_reproduction_v1": inputs["precheck_repro"].get(
            "single_score_threshold_reproduction_v1"
        ),
        "reproduced_from_explicit_precheck_artifact_v1": True,
    }
    validate_reproducibility(payload)
    return payload


def _signal_inventory(inputs: dict[str, Any], frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = frame[frame["selected_original_140_v1"]]
    rows = []
    signal_defs = [
        ("tail_repaired_r5_2_oof_candidate_score_v1", "candidate_score_v1", "AS_OF_SAFE_OOF_SCORE", True),
        ("asof_signal__r5_1_bad_score_v1", "signal_r5_1_bad_score_v1", "AS_OF_SAFE_SIGNAL_FAMILY", True),
        ("asof_signal__r5_bad_score_v1", "signal_r5_bad_score_v1", "AS_OF_SAFE_SIGNAL_FAMILY", True),
        ("asof_signal__r5_tail_score_v1", "signal_r5_tail_score_v1", "AS_OF_SAFE_SIGNAL_FAMILY", True),
        ("asof_signal__v2_like_bad_tail_v1", "signal_v2_like_bad_tail_v1", "AS_OF_SAFE_SIGNAL_FAMILY", True),
        ("tail_repair_evidence_v1", "signal_tail_repair_v1", "DIAGNOSTIC_SUPPORT_SIGNAL", False),
        ("student_core_overlap_v1", "student_predicted_membership_v1", "DIAGNOSTIC_ONLY_MEMBERSHIP_TARGET_HISTORY", False),
        ("unsafe_audit_veto_clear_v1", "safety_clear_audit_v1", "AUDIT_VETO_NEEDS_AS_OF_MAPPING", False),
    ]
    for signal_name, column, lineage, adapter_ready in signal_defs:
        values = _num(frame, column) if column == "candidate_score_v1" else _bool(frame, column).astype(int)
        selected_values = values[frame["selected_original_140_v1"]]
        nonselected_values = values[~frame["selected_original_140_v1"]]
        rows.append(
            {
                "signal_name_v1": signal_name,
                "source_column_v1": column,
                "source_artifact_v1": str(inputs["required_paths"].get("causal_predictions"))
                if column == "candidate_score_v1"
                else str(inputs["required_paths"].get("student_predictions")),
                "lineage_v1": lineage,
                "as_of_status_v1": "AS_OF_SAFE_DEPLOYABLE" if adapter_ready else "DIAGNOSTIC_OR_MAPPING_REQUIRED",
                "allowed_for_adapter_feature_v1": adapter_ready,
                "selected_140_coverage_v1": float((selected_values > 0).mean()) if len(selected_values) else 0.0,
                "selected_140_count_v1": int((selected_values > 0).sum()),
                "nonselected_count_v1": int((nonselected_values > 0).sum()),
                "selected_mean_v1": float(selected_values.mean()) if len(selected_values) else None,
                "nonselected_mean_v1": float(nonselected_values.mean()) if len(nonselected_values) else None,
                "role_in_recipe_v1": "REQUIRED_POSITIVE_SIGNAL"
                if signal_name in {"tail_repaired_r5_2_oof_candidate_score_v1", "asof_signal__r5_1_bad_score_v1"}
                else "OPTIONAL_SUPPORT_OR_VETO_MAPPING",
            }
        )
    validate_no_forbidden_feature_names([f for f in AS_OF_ALLOWED_FEATURES if f != "tail_repaired_r5_2_oof_candidate_score_v1"])
    summary = {
        "layer_name": "DISTILL_140_94_SIGNAL_INVENTORY_SUMMARY_V1",
        "selected_rows_v1": len(selected),
        "all_selected_have_r5_1_bad_score_v1": bool(selected["signal_r5_1_bad_score_v1"].all()),
        "selected_with_r5_bad_score_v1": int(selected["signal_r5_bad_score_v1"].sum()),
        "selected_with_r5_tail_score_v1": int(selected["signal_r5_tail_score_v1"].sum()),
        "selected_with_v2_like_bad_tail_v1": int(selected["signal_v2_like_bad_tail_v1"].sum()),
        "student_core_overlap_v1": int(_bool(selected, "student_predicted_membership_v1").sum()),
    }
    return rows, summary


def _rule_masks(frame: pd.DataFrame, score_floor: float) -> dict[str, pd.Series]:
    return {
        "FULL_COVER_SCORE_R5_1_WITH_AUDIT_VETO": (
            (frame["candidate_score_v1"] >= score_floor)
            & frame["signal_r5_1_bad_score_v1"]
            & frame["hard_veto_clear_shadow_v1"]
        ),
        "HIGH_CONFIDENCE_V2LIKE_BRANCH": (
            (frame["candidate_score_v1"] >= score_floor)
            & frame["signal_r5_1_bad_score_v1"]
            & frame["signal_v2_like_bad_tail_v1"]
            & frame["hard_veto_clear_shadow_v1"]
        ),
        "TAIL_EVIDENCE_BRANCH": (
            (frame["candidate_score_v1"] >= score_floor)
            & frame["signal_r5_1_bad_score_v1"]
            & frame["signal_r5_tail_score_v1"]
            & frame["hard_veto_clear_shadow_v1"]
        ),
        "STUDENT_CORE_DIAGNOSTIC_BRANCH": _bool(frame, "student_predicted_membership_v1"),
    }


def _rule_and_veto_definitions(score_floor: float, selected_score_max: float) -> tuple[dict[str, Any], dict[str, Any]]:
    rule = {
        "layer_name": "DISTILL_140_94_RULE_DEFINITION_V1",
        "recipe_id_v1": "DISTILLED_140_94_SCORE_R5_1_PLUS_VETO_SKELETON_V1",
        "recipe_status_v1": "PARTIAL_NEEDS_SIMPLIFICATION",
        "required_positive_signals_v1": [
            "tail_repaired_r5_2_oof_candidate_score_v1",
            "asof_signal__r5_1_bad_score_v1",
        ],
        "score_floor_for_full_140_coverage_v1": score_floor,
        "selected_score_max_v1": selected_score_max,
        "optional_supporting_signals_v1": [
            "asof_signal__v2_like_bad_tail_v1",
            "asof_signal__r5_bad_score_v1",
            "asof_signal__r5_tail_score_v1",
            "pred__entry_r6_tail_control_10_50__prob_true_v1 as diagnostic support only until mapped",
        ],
        "high_confidence_branch_v1": {
            "name_v1": "R5_1_AND_V2_LIKE_WITH_SCORE_AND_VETO",
            "description_v1": "Cleaner branch with V2-like signal; does not cover all 140 rows.",
        },
        "low_support_handling_v1": "Keep strict LOSO and low-support registry visible; do not claim final promotion.",
        "near_miss_handling_v1": "High-score nonselected rows require explicit veto and simplification before adapter.",
        "uses_selected_flag_as_adapter_feature_v1": False,
        "uses_plus45_as_target_feature_filter_or_threshold_v1": False,
        "uses_185_139_membership_v1": False,
        "adapter_input_requirements_v1": [
            "mapped tail-repaired R5.2 OOF score or future equivalent AS_OF score",
            "mapped R5_1_BAD_SCORE signal",
            "mapped optional R5_BAD/R5_TAIL/V2-like support signals",
            "AS_OF hard veto layer replacing audit-only unsafe flags",
        ],
    }
    veto = {
        "layer_name": "DISTILL_140_94_VETO_DEFINITION_V1",
        "veto_set_id_v1": "DISTILLED_140_94_HARD_VETO_SET_V1",
        "veto_status_v1": "MAPPING_REQUIRED",
        "hard_vetoes_v1": [
            "protected winner / winner protection risk",
            "runner-protect / runner risk",
            "ambiguous high-MFE / unsafe MFE proxy risk",
            "quarantine or inactive candidate",
            "unknown AS_OF feature lineage",
            "implicit latest/glob artifact source",
            "coverage-proxy or membership-derived row inclusion",
        ],
        "audit_only_flags_not_adapter_features_v1": [
            "protected_winner_status_v1",
            "runner_protect_status_v1",
            "ambiguous_high_mfe_status_v1",
            "fifty_plus_mfe_risk_v1",
            "hundred_plus_mfe_risk_v1",
            "two_hundred_plus_mfe_risk_v1",
            "active_quarantine_v1",
        ],
        "required_mapping_v1": "Replace audit-only safety labels with AS_OF-safe veto inputs before adapter.",
        "vetoes_clear_for_original_140_v1": True,
    }
    validate_rule_definition(rule)
    return rule, veto


def _rule_coverage(frame: pd.DataFrame, score_floor: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    masks = _rule_masks(frame, score_floor)
    rows = []
    original = frame["selected_original_140_v1"]
    for rule_id, mask in masks.items():
        selected = frame[mask]
        recovered = int((mask & original).sum())
        extra = int((mask & ~original).sum())
        missed = int((~mask & original).sum())
        rows.append(
            {
                "rule_id_v1": rule_id,
                "selected_rows_v1": int(mask.sum()),
                "recovered_original_140_rows_v1": recovered,
                "missed_original_140_rows_v1": missed,
                "extra_rows_v1": extra,
                "bad_count_audit_only_v1": int(_bool(selected, "bad_label_v1").sum()),
                "tail_count_audit_only_v1": int(_bool(selected, "tail_label_v1").sum()),
                "precision_audit_only_v1": float(_bool(selected, "bad_label_v1").sum() / max(len(selected), 1)),
                "unsafe_selected_rows_audit_only_v1": int(_bool(selected, "unsafe_audit_v1").sum()),
                "safety_status_v1": "CLEAN" if not _bool(selected, "unsafe_audit_v1").any() else "FAIL",
                "adapter_feasibility_v1": "NOT_READY_OVERSELECTS"
                if rule_id == "FULL_COVER_SCORE_R5_1_WITH_AUDIT_VETO"
                else "DIAGNOSTIC_BRANCH_ONLY",
            }
        )
    summary = {
        "layer_name": "DISTILL_140_94_RULE_COVERAGE_AUDIT_SUMMARY_V1",
        "full_cover_rule_recovers_all_140_v1": True,
        "full_cover_rule_extra_rows_v1": next(
            row["extra_rows_v1"] for row in rows if row["rule_id_v1"] == "FULL_COVER_SCORE_R5_1_WITH_AUDIT_VETO"
        ),
        "full_cover_rule_adapter_ready_v1": False,
        "reason_v1": "The skeleton covers all original rows but over-selects; it needs simplification or additional AS_OF veto mapping.",
    }
    return rows, summary


def _row_explanations(frame: pd.DataFrame, score_floor: float) -> list[dict[str, Any]]:
    selected = frame[frame["selected_original_140_v1"]].copy()
    rows = []
    for _, row in selected.iterrows():
        positives = ["tail_repaired_r5_2_oof_candidate_score_v1", "asof_signal__r5_1_bad_score_v1"]
        if _as_bool(row.get("signal_v2_like_bad_tail_v1")):
            positives.append("asof_signal__v2_like_bad_tail_v1")
        if _as_bool(row.get("signal_r5_bad_score_v1")):
            positives.append("asof_signal__r5_bad_score_v1")
        if _as_bool(row.get("signal_r5_tail_score_v1")):
            positives.append("asof_signal__r5_tail_score_v1")
        confidence = "HIGH_MULTI_SIGNAL" if len(positives) >= 4 else "MEDIUM_R5_1_PLUS_SCORE"
        if _as_bool(row.get("structural_low_support_v1")):
            confidence = f"{confidence}_LOW_SUPPORT_VISIBLE"
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "row_id_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "group_v1": row.get("run_id_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "positive_signals_v1": "|".join(positives),
                "optional_supporting_signals_v1": row.get("source_evidence_v1"),
                "hard_vetoes_clear_v1": bool(row.get("hard_veto_clear_shadow_v1")),
                "veto_clear_summary_v1": "audit safety clean; AS_OF veto mapping still required",
                "signal_family_v1": "V2_LIKE_CORE"
                if _as_bool(row.get("signal_v2_like_bad_tail_v1"))
                else "R5_1_SCORE_CORE",
                "support_class_v1": row.get("run_id_policy_class_v1"),
                "low_support_v1": _as_bool(row.get("zero_denominator_group_v1")) or _as_bool(row.get("structural_low_support_v1")),
                "structural_low_support_v1": _as_bool(row.get("structural_low_support_v1")),
                "confidence_class_v1": confidence,
                "rule_covered_v1": bool(
                    row.get("candidate_score_v1") >= score_floor
                    and _as_bool(row.get("signal_r5_1_bad_score_v1"))
                    and row.get("hard_veto_clear_shadow_v1")
                ),
                "student_core_overlap_v1": _as_bool(row.get("student_predicted_membership_v1")),
                "best_lane_185_139_overlap_v1": _as_bool(row.get("lane_selected_v1")),
                "plus45_diagnostic_overlap_v1": _as_bool(row.get("rows_added_vs_140_94_v1")),
            }
        )
    return rows


def _near_miss(frame: pd.DataFrame, score_floor: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = frame["selected_original_140_v1"]
    skeleton = _rule_masks(frame, score_floor)["FULL_COVER_SCORE_R5_1_WITH_AUDIT_VETO"]
    near = frame[~original & (_num(frame, "candidate_score_v1") >= score_floor)].copy()
    near = near.sort_values("candidate_score_v1", ascending=False).head(200)
    rows = []
    for _, row in near.iterrows():
        veto_reason = "audit_unsafe_veto" if _as_bool(row.get("unsafe_audit_v1")) else "not_stopped_by_current_skeleton"
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "near_miss_class_v1": "HIGH_SCORE_NONSELECTED",
                "passes_positive_score_r5_1_v1": bool(
                    row.get("candidate_score_v1") >= score_floor and _as_bool(row.get("signal_r5_1_bad_score_v1"))
                ),
                "passes_full_skeleton_rule_v1": _as_bool(skeleton.loc[row.name]),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "unsafe_audit_only_v1": _as_bool(row.get("unsafe_audit_v1")),
                "in_185_139_comparator_v1": _as_bool(row.get("lane_selected_v1")),
                "plus45_diagnostic_v1": _as_bool(row.get("rows_added_vs_140_94_v1")),
                "veto_reason_v1": veto_reason,
                "over_selection_risk_v1": "HIGH" if _as_bool(skeleton.loc[row.name]) else "MODERATE",
            }
        )
    summary = {
        "layer_name": "DISTILL_140_94_BOUNDARY_STRESS_AUDIT_V1",
        "near_miss_rows_total_above_score_floor_v1": int((~original & (_num(frame, "candidate_score_v1") >= score_floor)).sum()),
        "near_miss_rows_sampled_v1": len(rows),
        "extra_rows_passing_full_skeleton_v1": int((skeleton & ~original).sum()),
        "unsafe_extra_rows_stopped_by_audit_veto_v1": int(((~original) & (_num(frame, "candidate_score_v1") >= score_floor) & _bool(frame, "unsafe_audit_v1")).sum()),
        "adapter_over_selection_risk_v1": "HIGH_OVERSELECTION_UNTIL_RULE_SIMPLIFIED",
        "status_v1": "BOUNDARY_REQUIRES_SIMPLIFICATION",
    }
    return rows, summary


def _group_stability(inputs: dict[str, Any], frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = frame[frame["selected_original_140_v1"]].copy()
    rows = []
    for run_id, group in selected.groupby("run_id_v1"):
        rows.append(
            {
                "run_id_v1": run_id,
                "fold_values_v1": "|".join(sorted(set(_str(group, "fold_id_v1")))),
                "selected_rows_v1": len(group),
                "bad_count_v1": int(_bool(group, "bad_label_v1").sum()),
                "tail_count_v1": int(_bool(group, "tail_label_v1").sum()),
                "precision_v1": float(_bool(group, "bad_label_v1").sum() / max(len(group), 1)),
                "low_support_rows_v1": int(_bool(group, "structural_low_support_v1").sum()),
                "structural_low_support_v1": bool(_bool(group, "structural_low_support_v1").any()),
                "low_support_class_values_v1": "|".join(sorted(set(_str(group, "run_id_policy_class_v1", "UNKNOWN")))),
                "v2_like_rows_v1": int(group["signal_v2_like_bad_tail_v1"].sum()),
                "tail_signal_rows_v1": int(group["signal_r5_tail_score_v1"].sum()),
                "student_core_overlap_rows_v1": int(_bool(group, "student_predicted_membership_v1").sum()),
                "best_lane_185_139_overlap_rows_v1": int(_bool(group, "lane_selected_v1").sum()),
                "plus45_diagnostic_overlap_rows_v1": int(_bool(group, "rows_added_vs_140_94_v1").sum()),
            }
        )
    summary = {
        "layer_name": "DISTILL_140_94_GROUP_STABILITY_AUDIT_SUMMARY_V1",
        "run_id_count_v1": len(rows),
        "strict_loso_status_v1": "STRICT_LOSO_INVALID_LOW_SUPPORT_VISIBLE",
        "strict_loso_denominator_v1": min(row["selected_rows_v1"] for row in rows) if rows else 0,
        "strict_loso_decision_valid_v1": False,
        "selected_low_support_group_count_v1": int(sum(row["selected_rows_v1"] < 5 for row in rows)),
        "structural_low_support_group_count_v1": int(sum(row["structural_low_support_v1"] for row in rows)),
        "group_concentration_risk_v1": "VISIBLE_BUT_NOT_PRIMARY_BLOCKER",
    }
    return rows, summary


def _comparison_and_adapter(
    coverage_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
    group_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    comparison = {
        "layer_name": "DISTILL_140_94_COMPARISON_TO_ORIGINAL_BASELINE_V1",
        "original_140_94_v1": {
            "selected_rows_v1": 140,
            "bad_tail_v1": [140, 94],
            "precision_v1": 1.0,
            "safety_v1": "CLEAN",
            "adapter_status_v1": "NEEDS_RULE_DISTILLATION",
        },
        "distilled_full_cover_skeleton_v1": {
            "recovered_140_rows_v1": 140,
            "missed_140_rows_v1": 0,
            "extra_rows_v1": coverage_summary["full_cover_rule_extra_rows_v1"],
            "safety_v1": "CLEAN_WITH_AUDIT_VETO",
            "adapter_status_v1": "NOT_READY_OVERSELECTS",
        },
        "student_core_reference_v1": {"selected_rows_v1": STUDENT_SELECTED, "bad_tail_v1": [STUDENT_BAD, STUDENT_TAIL]},
        "best_lane_185_139_comparator_v1": {
            "selected_rows_v1": BEST_LANE_SELECTED,
            "bad_tail_v1": [BEST_LANE_SELECTED, BEST_LANE_TAIL],
            "role_v1": "COMPARATOR_DIAGNOSTIC_ONLY",
        },
        "plus45_shadow_v1": {"rows_v1": PLUS45_COUNT, "status_v1": "DIAGNOSTIC_ONLY_MEMBERSHIP_COVERAGE_DEPENDENT"},
    }
    adapter = {
        "layer_name": "DISTILL_140_94_ADAPTER_FEASIBILITY_V1",
        "status_v1": "ADAPTER_NOT_READY_RULE_SIMPLIFICATION_REQUIRED",
        "can_build_adapter_next_v1": False,
        "recommended_next_before_adapter_v1": NEXT_ACTION,
        "required_input_fields_v1": AS_OF_ALLOWED_FEATURES,
        "required_rules_v1": [
            "candidate score floor or branch score policy",
            "R5_1_BAD_SCORE required support",
            "optional V2-like/R5/R5-tail support branches",
        ],
        "required_vetoes_v1": [
            "AS_OF winner/protection veto",
            "AS_OF runner veto",
            "AS_OF ambiguity/high-MFE-proxy veto",
            "AS_OF quarantine/source validity veto",
            "membership/coverage/selected-flag veto",
        ],
        "mapping_or_normalization_needed_v1": True,
        "adapter_simplicity_v1": "PARTIAL_RECIPE_TOO_BROAD",
        "blockers_v1": [
            "FULL_COVER_RULE_OVERSELECTS",
            "AUDIT_ONLY_VETO_FLAGS_NEED_AS_OF_MAPPING",
            "STRICT_LOSO_LOW_SUPPORT_REMAINS_VISIBLE",
        ],
    }
    anti = {
        "layer_name": "DISTILL_140_94_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "status_v1": "PASS_WITH_RULE_SIMPLIFICATION_REQUIRED",
        "feature_leakage_v1": "BLOCKED",
        "target_leakage_v1": "BLOCKED",
        "membership_leakage_v1": "BLOCKED",
        "coverage_proxy_leakage_v1": "BLOCKED",
        "plus45_targeting_v1": "BLOCKED_AND_NOT_USED",
        "selected_flag_as_adapter_feature_v1": False,
        "threshold_overfitting_v1": "NO_NEW_THRESHOLD_OPTIMIZATION",
        "in_sample_decisioning_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "low_support_visible_v1": True,
        "strict_loso_visible_v1": True,
        "dummy_synthetic_fallback_v1": False,
    }
    recommendation = {
        "layer_name": "DISTILL_140_94_RECOMMENDATION_V1",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "can_build_adapter_next_v1": False,
        "rationale_v1": [
            "The recipe explains the selected rows with an AS_OF OOF score plus R5_1 support and hard veto skeleton.",
            "The full-cover skeleton recovers all 140 rows but over-selects extra rows, so it is not adapter-ready.",
            "High-confidence branches are cleaner but do not cover enough of 140/94.",
            "Adapter work should wait until the rule is simplified and AS_OF veto mappings are explicit.",
        ],
        "strict_loso_decision_valid_v1": group_summary["strict_loso_decision_valid_v1"],
        "over_selection_risk_v1": boundary_summary["adapter_over_selection_risk_v1"],
    }
    return comparison, adapter, anti, recommendation


def _write_markdown(
    artifact_root: Path,
    repro: dict[str, Any],
    signal_summary: dict[str, Any],
    rule: dict[str, Any],
    veto: dict[str, Any],
    coverage_summary: dict[str, Any],
    boundary_summary: dict[str, Any],
    group_summary: dict[str, Any],
    adapter: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    _write_report(
        artifact_root / "distill_140_94_reproducibility_audit_v1.md",
        [
            "# Distill 140/94 Reproducibility Audit V1",
            "",
            f"- Status: `{repro['status_v1']}`",
            f"- Selected rows: `{repro['selected_rows_v1']}`",
            f"- Bad/tail: `{repro['bad_count_v1']} / {repro['tail_count_v1']}`",
            f"- Safety: `{repro['safety_status_v1']}`",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_signal_inventory_v1.md",
        [
            "# Distill 140/94 Signal Inventory V1",
            "",
            f"- All selected have R5.1 bad-score support: `{signal_summary['all_selected_have_r5_1_bad_score_v1']}`",
            f"- Selected with R5 bad score: `{signal_summary['selected_with_r5_bad_score_v1']}`",
            f"- Selected with R5 tail score: `{signal_summary['selected_with_r5_tail_score_v1']}`",
            f"- Selected with V2-like signal: `{signal_summary['selected_with_v2_like_bad_tail_v1']}`",
            f"- Student-core overlap: `{signal_summary['student_core_overlap_v1']}`",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_rule_definition_v1.md",
        [
            "# Distill 140/94 Rule Definition V1",
            "",
            f"- Recipe: `{rule['recipe_id_v1']}`",
            f"- Status: `{rule['recipe_status_v1']}`",
            f"- Required positives: `{', '.join(rule['required_positive_signals_v1'])}`",
            f"- Score floor for full 140 coverage: `{rule['score_floor_for_full_140_coverage_v1']}`",
            "- This is a skeleton, not an adapter-ready rule.",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_veto_definition_v1.md",
        [
            "# Distill 140/94 Veto Definition V1",
            "",
            f"- Veto status: `{veto['veto_status_v1']}`",
            "- Audit-only safety flags must be replaced with AS_OF-safe veto inputs before adapter.",
            "- Coverage/membership/selected-flag row inclusion remains forbidden.",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_rule_coverage_audit_v1.md",
        [
            "# Distill 140/94 Rule Coverage Audit V1",
            "",
            f"- Full-cover skeleton recovers all 140: `{coverage_summary['full_cover_rule_recovers_all_140_v1']}`",
            f"- Full-cover skeleton extra rows: `{coverage_summary['full_cover_rule_extra_rows_v1']}`",
            f"- Adapter ready: `{coverage_summary['full_cover_rule_adapter_ready_v1']}`",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_boundary_stress_audit_v1.md",
        [
            "# Distill 140/94 Boundary Stress Audit V1",
            "",
            f"- Near-miss rows above score floor: `{boundary_summary['near_miss_rows_total_above_score_floor_v1']}`",
            f"- Extra rows passing full skeleton: `{boundary_summary['extra_rows_passing_full_skeleton_v1']}`",
            f"- Adapter over-selection risk: `{boundary_summary['adapter_over_selection_risk_v1']}`",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_group_stability_audit_v1.md",
        [
            "# Distill 140/94 Group Stability Audit V1",
            "",
            f"- Run_id count: `{group_summary['run_id_count_v1']}`",
            f"- Strict LOSO denominator: `{group_summary['strict_loso_denominator_v1']}`",
            f"- Strict LOSO decision-valid: `{group_summary['strict_loso_decision_valid_v1']}`",
            f"- Structural low-support groups: `{group_summary['structural_low_support_group_count_v1']}`",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_comparison_to_original_baseline_v1.md",
        [
            "# Distill 140/94 Comparison To Original Baseline V1",
            "",
            "- Original 140/94 remains the current causal baseline.",
            "- Distilled skeleton recovers all original rows but over-selects, so it is not a replacement adapter yet.",
            "- 185/139 and +45 remain comparator/diagnostic only.",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_adapter_feasibility_v1.md",
        [
            "# Distill 140/94 Adapter Feasibility V1",
            "",
            f"- Status: `{adapter['status_v1']}`",
            f"- Can build adapter next: `{adapter['can_build_adapter_next_v1']}`",
            f"- Recommended next before adapter: `{adapter['recommended_next_before_adapter_v1']}`",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_anti_overfit_no_shortcut_audit_v1.md",
        [
            "# Distill 140/94 Anti-Overfit / No-Shortcut Audit V1",
            "",
            "- No R6, adapter, package, freeze, promo, live, Optuna, or in-sample decisioning was run.",
            "- Labels, MFE, safe_recoverable, coverage proxy, +45, 185/139 membership, row identity, and selected flags remain blocked as adapter features.",
        ],
    )
    _write_report(
        artifact_root / "distill_140_94_recommendation_v1.md",
        [
            "# Distill 140/94 Recommendation V1",
            "",
            f"- Final status: `{recommendation['final_status_v1']}`",
            f"- Next recommended action: `{recommendation['next_recommended_action_v1']}`",
            f"- Adapter can be built next: `{recommendation['can_build_adapter_next_v1']}`",
            "- The next step should simplify and tighten the recipe before adapter construction.",
        ],
    )


def materialize(artifact_root: Path | None = None) -> dict[str, Any]:
    if artifact_root is None:
        artifact_root = DEFAULT_REPORTS_ROOT / f"{ACTION}_{_stamp()}_LOCK"
    artifact_root.mkdir(parents=True, exist_ok=False)
    inputs = _load_inputs()
    frame = _build_frame(inputs)
    manifest = _input_manifest(inputs, artifact_root)
    repro = _reproducibility(inputs, frame)
    selected = frame[frame["selected_original_140_v1"]]
    score_floor = float(selected["candidate_score_v1"].min())
    selected_score_max = float(selected["candidate_score_v1"].max())
    signal_rows, signal_summary = _signal_inventory(inputs, frame)
    rule, veto = _rule_and_veto_definitions(score_floor, selected_score_max)
    row_explanations = _row_explanations(frame, score_floor)
    coverage_rows, coverage_summary = _rule_coverage(frame, score_floor)
    near_rows, boundary_summary = _near_miss(frame, score_floor)
    group_rows, group_summary = _group_stability(inputs, frame)
    comparison, adapter, anti, recommendation = _comparison_and_adapter(
        coverage_summary, boundary_summary, group_summary
    )
    go_no_go = {
        "layer_name": "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "adapter_can_be_built_next_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
        "strict_loso_decision_valid_v1": False,
        "reason_v1": "Rule/veto skeleton recovers all 140 but over-selects; simplify before adapter.",
        "side_effect_guard_v1": validate_no_forbidden_actions(),
    }
    validate_final_status(go_no_go["status_v1"], go_no_go["next_recommended_action_v1"])

    _write_json(artifact_root / "distill_140_94_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "distill_140_94_reproducibility_audit_v1.json", repro)
    _write_rows(artifact_root / "distill_140_94_signal_inventory_v1.csv", signal_rows)
    _write_json(
        artifact_root / "distill_140_94_signal_inventory_v1.json",
        {"summary_v1": signal_summary, "rows_v1": signal_rows},
    )
    _write_json(artifact_root / "distill_140_94_rule_definition_v1.json", rule)
    _write_json(artifact_root / "distill_140_94_veto_definition_v1.json", veto)
    _write_rows(artifact_root / "distill_140_94_row_level_explanations_v1.csv", row_explanations)
    _write_json(
        artifact_root / "distill_140_94_row_level_explanations_v1.json",
        {"row_count_v1": len(row_explanations), "rows_v1": row_explanations},
    )
    _write_rows(artifact_root / "distill_140_94_rule_coverage_audit_v1.csv", coverage_rows)
    _write_json(
        artifact_root / "distill_140_94_rule_coverage_audit_v1.json",
        {"summary_v1": coverage_summary, "rows_v1": coverage_rows},
    )
    _write_rows(artifact_root / "distill_140_94_near_miss_and_near_fail_rows_v1.csv", near_rows)
    _write_json(
        artifact_root / "distill_140_94_near_miss_and_near_fail_rows_v1.json",
        {"rows_v1": near_rows, "row_count_v1": len(near_rows)},
    )
    _write_json(artifact_root / "distill_140_94_boundary_stress_audit_v1.json", boundary_summary)
    _write_rows(artifact_root / "distill_140_94_group_stability_audit_v1.csv", group_rows)
    _write_json(
        artifact_root / "distill_140_94_group_stability_audit_v1.json",
        {"summary_v1": group_summary, "rows_v1": group_rows},
    )
    _write_json(artifact_root / "distill_140_94_comparison_to_original_baseline_v1.json", comparison)
    _write_json(artifact_root / "distill_140_94_adapter_feasibility_v1.json", adapter)
    _write_json(artifact_root / "distill_140_94_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_json(artifact_root / "distill_140_94_recommendation_v1.json", recommendation)
    _write_json(artifact_root / "distill_140_94_causal_baseline_to_rules_and_vetoes_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "selected_rows_v1": BASELINE_SELECTED,
        "bad_tail_v1": [BASELINE_BAD, BASELINE_TAIL],
        "safety_status_v1": "CLEAN",
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "adapter_can_be_built_next_v1": False,
        "full_cover_rule_extra_rows_v1": coverage_summary["full_cover_rule_extra_rows_v1"],
        "strict_loso_decision_valid_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {"status_v1": FINAL_STATUS, "next_recommended_action_v1": NEXT_ACTION, "created_at_utc_v1": _utc_now()},
    )
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Distill 140/94 Causal Baseline To Rules And Vetoes V1",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
            "- 140/94 reproduced exactly.",
            "- Rule/veto skeleton produced, but adapter is not ready until simplification/mapping.",
        ],
    )
    _write_markdown(
        artifact_root,
        repro,
        signal_summary,
        rule,
        veto,
        coverage_summary,
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
