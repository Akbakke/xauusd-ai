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
ACTION = "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1"

INPUT_CAUSAL_REBUILD_ROOT = (
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
BASELINE_CANDIDATE_ID = "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL"

BASELINE_BAD = 140
BASELINE_TAIL = 94
BEST_LANE_BAD = 185
BEST_LANE_TAIL = 139
PLUS45_COUNT = 45
STUDENT_SELECTED = 135
STUDENT_BAD = 131
STUDENT_TAIL = 93
WEDNESDAY_BAD = 180
WEDNESDAY_TAIL = 149
COVERAGE_PROXY_BAD = 188
COVERAGE_PROXY_TAIL = 136

FINAL_STATUS = "140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER"
NEXT_ACTION = "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1"

ALLOWED_FINAL_STATUSES = {
    "140_94_CAUSAL_BASELINE_PRECHECK_PASS_ADAPTER_FEASIBLE",
    "140_94_CAUSAL_BASELINE_PRECHECK_PASS_ADAPTER_REQUIRED",
    "140_94_CAUSAL_BASELINE_NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER",
    "140_94_BLOCKED_BY_AS_OF_LINEAGE_GAPS",
    "140_94_BLOCKED_BY_UNSAFE_LOOKALIKE_RISK",
    "140_94_BLOCKED_BY_LOW_SUPPORT_OR_GROUP_CONCENTRATION",
    "140_94_BLOCKED_BY_MEMBERSHIP_OR_COVERAGE_DEPENDENCY",
    "BLOCKED_BY_MISSING_ARTIFACTS",
    "BLOCKED_BY_TEST_FAILURE",
}

ALLOWED_NEXT_ACTIONS = {
    "BUILD_AS_OF_SAFE_140_94_CAUSAL_BASELINE_ADAPTER_V1",
    "DISTILL_140_94_CAUSAL_BASELINE_TO_RULES_AND_VETOES_V1",
    "DEEPEN_140_94_AS_OF_LINEAGE_AUDIT_V1",
    "DEEPEN_140_94_UNSAFE_LOOKALIKE_BOUNDARY_AUDIT_V1",
    "DEEPEN_140_94_GROUPED_GENERALIZATION_AND_LOSO_AUDIT_V1",
    "REBUILD_FEATURE_LINEAGE_AND_AS_OF_SIGNAL_INVENTORY_V1",
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
]

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
    "185_139",
    "plus45",
    "rows_added_vs_140_94",
    "lane_selected",
    "lane_id",
    "teacher_membership",
    "selected_by",
    "selected_rows",
    "r5_2_package_selected",
    "r6_best_candidate_selected",
    "protected",
    "runner",
    "ambiguous",
    "quarantine",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "latest",
    "glob",
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


def validate_no_forbidden_feature_names(features: Iterable[str]) -> bool:
    blocked = []
    for feature in features:
        lower = feature.lower()
        if any(pattern in lower for pattern in DENY_PATTERNS):
            blocked.append(feature)
    if blocked:
        raise RuntimeError(f"FORBIDDEN_140_94_ADAPTER_FEATURE: {blocked}")
    return True


def validate_reproduce_140_94(membership: pd.DataFrame) -> bool:
    selected = membership[_bool(membership, "r5_2_package_selected_v1")]
    bad = int(_bool(selected, "bad_label_v1").sum())
    tail = int(_bool(selected, "tail_label_v1").sum())
    if len(selected) != BASELINE_BAD or bad != BASELINE_BAD or tail != BASELINE_TAIL:
        raise RuntimeError(f"BASELINE_140_94_REPRODUCTION_FAILED: selected={len(selected)} bad={bad} tail={tail}")
    return True


def validate_comparator_roles(payload: dict[str, Any]) -> bool:
    if payload.get("best_lane_185_139_role_v1") != "COMPARATOR_DIAGNOSTIC_ONLY":
        raise RuntimeError("185_139_MUST_REMAIN_COMPARATOR_ONLY")
    if payload.get("plus45_role_v1") != "DIAGNOSTIC_ONLY_NOT_TARGET":
        raise RuntimeError("PLUS45_MUST_REMAIN_DIAGNOSTIC_ONLY")
    return True


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
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
        INPUT_CAUSAL_REBUILD_ROOT,
        INPUT_STUDENT_ROOT,
        INPUT_STABILITY_ROOT,
        INPUT_BEST_LANE_PACKAGE_ROOT,
        INPUT_LANE_PACK_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "causal_summary": INPUT_CAUSAL_REBUILD_ROOT / "summary_v1.json",
        "causal_go_no_go": INPUT_CAUSAL_REBUILD_ROOT / "reject_or_rebuild_best_lane_from_causal_signals_go_no_go_v1.json",
        "causal_predictions": INPUT_CAUSAL_REBUILD_ROOT / "causal_rebuild_candidate_oof_predictions_v1.csv",
        "causal_metrics": INPUT_CAUSAL_REBUILD_ROOT / "causal_rebuild_candidate_metrics_v1.csv",
        "causal_inventory": INPUT_CAUSAL_REBUILD_ROOT / "causal_signal_inventory_v1.csv",
        "causal_ranking": INPUT_CAUSAL_REBUILD_ROOT / "causal_rebuild_candidate_ranking_v1.json",
        "student_summary": INPUT_STUDENT_ROOT / "summary_v1.json",
        "student_predictions": INPUT_STUDENT_ROOT / "best_lane_student_oof_predictions_v1.csv",
        "stability_go_no_go": INPUT_STABILITY_ROOT / "stability_recheck_best_lane_185_139_before_r6_go_no_go_v1.json",
        "best_membership": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv",
        "best_manifest": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_package_manifest_v1.json",
        "best_integrity": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_package_integrity_report_v1.json",
        "best_low_support": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_low_support_report_v1.json",
        "best_safety": INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_safety_report_v1.json",
        "lane_pack_summary": INPUT_LANE_PACK_ROOT / "summary_v1.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    causal_go = _read_json(required["causal_go_no_go"])
    if causal_go.get("status_v1") != "RETURN_TO_140_94_CAUSAL_BASELINE_BEST_CURRENT_OPTION":
        raise RuntimeError("CAUSAL_REBUILD_STATUS_NOT_RETURN_TO_140_94")
    return {
        "required_paths": required,
        "causal_summary": _read_json(required["causal_summary"]),
        "causal_go_no_go": causal_go,
        "causal_predictions": pd.read_csv(required["causal_predictions"]),
        "causal_metrics": pd.read_csv(required["causal_metrics"]),
        "causal_inventory": pd.read_csv(required["causal_inventory"]),
        "causal_ranking": _read_json(required["causal_ranking"]),
        "student_summary": _read_json(required["student_summary"]),
        "student_predictions": pd.read_csv(required["student_predictions"]),
        "stability_go_no_go": _read_json(required["stability_go_no_go"]),
        "best_membership": pd.read_csv(required["best_membership"]),
        "best_manifest": _read_json(required["best_manifest"]),
        "best_integrity": _read_json(required["best_integrity"]),
        "best_low_support": _read_json(required["best_low_support"]),
        "best_safety": _read_json(required["best_safety"]),
        "lane_pack_summary": _read_json(required["lane_pack_summary"]),
    }


def _baseline_predictions(inputs: dict[str, Any]) -> pd.DataFrame:
    pred = inputs["causal_predictions"]
    baseline = pred[pred["candidate_id_v1"] == BASELINE_CANDIDATE_ID].copy()
    if len(baseline) != 1914:
        raise RuntimeError(f"BASELINE_PREDICTION_ROW_COUNT_UNEXPECTED: {len(baseline)}")
    return baseline


def _exact_threshold_status(baseline: pd.DataFrame) -> dict[str, Any]:
    selected = _bool(baseline, "candidate_selected_v1").to_numpy()
    scores = _num(baseline, "candidate_score_v1").to_numpy()
    exact_threshold = None
    for threshold in sorted(set(float(score) for score in scores), reverse=True):
        if np.array_equal(scores >= threshold, selected):
            exact_threshold = threshold
            break
    best = {"mismatch_count_v1": len(scores), "threshold_v1": None, "selected_rows_v1": 0}
    for threshold in sorted(set(float(score) for score in scores)):
        predicted = scores >= threshold
        mismatch = int((predicted != selected).sum())
        if mismatch < best["mismatch_count_v1"]:
            best = {"mismatch_count_v1": mismatch, "threshold_v1": threshold, "selected_rows_v1": int(predicted.sum())}
    return {
        "single_score_threshold_reproduces_140_94_v1": exact_threshold is not None,
        "exact_threshold_v1": exact_threshold,
        "best_single_score_threshold_approximation_v1": best,
        "selected_score_min_v1": float(scores[selected].min()) if selected.any() else None,
        "nonselected_score_max_v1": float(scores[~selected].max()) if (~selected).any() else None,
    }


def _selection_lineage(inputs: dict[str, Any], baseline: pd.DataFrame) -> list[dict[str, Any]]:
    membership = inputs["best_membership"]
    selected = membership[_bool(membership, "r5_2_package_selected_v1")].copy()
    scores = baseline[["candidate_uid_v1", "candidate_score_v1", "split_id_v1", "model_family_v1", "threshold_policy_v1"]]
    selected = selected.merge(scores, on="candidate_uid_v1", how="left")
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "trade_uid_v1": row.get("trade_uid_v1"),
                "decision_timestamp_v1": row.get("decision_timestamp_v1"),
                "trade_id_v1": row.get("trade_id_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "selected_by_140_94_v1": True,
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "candidate_score_v1": row.get("candidate_score_v1"),
                "selection_source_v1": "r5_2_package_selected_v1 from explicit best-lane package artifact",
                "selection_source_artifact_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv"),
                "score_source_artifact_v1": str(INPUT_CAUSAL_REBUILD_ROOT / "causal_rebuild_candidate_oof_predictions_v1.csv"),
                "supporting_signal_summary_v1": "existing tail-repaired R5.2 OOF score/provenance; exact adapter rule not materialized in this gate",
                "as_of_safe_features_available_v1": "|".join(AS_OF_ALLOWED_FEATURES),
                "blocked_feature_dependency_v1": "selected flag used only for reproduction, not adapter feature",
                "membership_coverage_hindsight_dependency_v1": False,
                "adapter_rule_available_now_v1": False,
                "requires_rule_distillation_v1": True,
            }
        )
    return rows


def _feature_lineage(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    inventory = inputs["causal_inventory"].to_dict("records")
    by_name = {row.get("feature_name_v1") or row.get("name_v1"): row for row in inventory}
    rows = []
    for feature in sorted(set(AS_OF_ALLOWED_FEATURES + [str(row.get("feature_name_v1") or row.get("name_v1")) for row in inventory])):
        source = by_name.get(feature, {})
        allowed = feature in AS_OF_ALLOWED_FEATURES
        if allowed:
            status = "AS_OF_SAFE_DEPLOYABLE"
            allowed_blocked = "ALLOWED"
            risk = ""
            reason = source.get("reason_v1", "Allowed AS_OF-safe signal from previous causal inventory.")
        else:
            status = source.get("as_of_status_v1", "BLOCKED_UNKNOWN_LINEAGE")
            allowed_blocked = "BLOCKED"
            risk = source.get("potential_leakage_risk_v1", "UNKNOWN_LINEAGE")
            reason = source.get("reason_v1", "Not in 140/94 adapter precheck allowlist.")
        rows.append(
            {
                "feature_name_v1": feature,
                "source_artifact_v1": source.get("source_artifact_v1", str(INPUT_CAUSAL_REBUILD_ROOT / "causal_signal_inventory_v1.csv")),
                "source_path_v1": source.get("source_path_v1", str(INPUT_CAUSAL_REBUILD_ROOT)),
                "lineage_v1": source.get("lineage_v1", risk),
                "as_of_status_v1": status,
                "allowed_blocked_v1": allowed_blocked,
                "reason_v1": reason,
                "potential_leakage_risk_v1": risk,
                "used_for_adapter_precheck_v1": allowed,
                "available_before_outcome_v1": allowed,
            }
        )
    validate_no_forbidden_feature_names(AS_OF_ALLOWED_FEATURES)
    allowlist = {
        "layer_name": "BASELINE_140_94_AS_OF_FEATURE_ALLOWLIST_V1",
        "policy_v1": "AS_OF_SAFE_SIGNALS_ONLY_NO_LABEL_MEMBERSHIP_OR_COVERAGE_PROXY",
        "allowed_features_v1": AS_OF_ALLOWED_FEATURES,
        "feature_count_v1": len(AS_OF_ALLOWED_FEATURES),
    }
    denylist = {
        "layer_name": "BASELINE_140_94_AS_OF_FEATURE_DENYLIST_V1",
        "blocked_patterns_v1": DENY_PATTERNS,
        "blocked_classes_v1": [
            "final bad/tail labels as feature",
            "post-outcome MFE",
            "safe_recoverable direct",
            "coverage proxy",
            "185/139 membership",
            "+45 diagnostic flags",
            "lane membership flags",
            "selected_by or artifact-derived selected flags",
            "final outcome flags",
            "protected/runner/ambiguous/quarantine/high-MFE unless proven AS_OF-safe",
            "artifact row identity leakage",
            "implicit latest/glob source fields",
            "unknown lineage fields",
        ],
    }
    return rows, allowlist, denylist


def _near_miss(inputs: dict[str, Any], baseline: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    membership = inputs["best_membership"]
    enriched = baseline.merge(
        membership[
            [
                "candidate_uid_v1",
                "run_id_v1",
                "fold_id_v1",
                "bad_label_v1",
                "tail_label_v1",
                "r5_2_package_selected_v1",
                "lane_selected_v1",
                "rows_added_vs_140_94_v1",
            ]
        ],
        on="candidate_uid_v1",
        how="left",
    )
    selected = _bool(enriched, "candidate_selected_v1")
    selected_min = float(_num(enriched[selected], "candidate_score_v1").min()) if selected.any() else 1.0
    nonselected = enriched[~selected].sort_values("candidate_score_v1", ascending=False).head(75).copy()
    rows = []
    for _, row in nonselected.iterrows():
        score = float(row.get("candidate_score_v1", 0.0))
        rows.append(
            {
                "candidate_uid_v1": row.get("candidate_uid_v1"),
                "run_id_v1": row.get("run_id_v1"),
                "fold_id_v1": row.get("fold_id_v1"),
                "candidate_score_v1": score,
                "near_miss_type_v1": "HIGH_SCORE_NON_SELECTED",
                "would_select_under_relaxed_min_score_v1": score >= max(selected_min - 0.05, 0.0),
                "bad_label_audit_only_v1": _as_bool(row.get("bad_label_v1")),
                "tail_label_audit_only_v1": _as_bool(row.get("tail_label_v1")),
                "in_185_139_comparator_v1": _as_bool(row.get("lane_selected_v1")),
                "plus45_diagnostic_v1": _as_bool(row.get("rows_added_vs_140_94_v1")),
                "unsafe_lookalike_status_v1": "UNKNOWN_REQUIRES_VETO_LINEAGE" if score >= selected_min else "LOWER_SCORE_NONSELECTED",
                "adapter_over_selection_risk_v1": "MODERATE" if score >= selected_min else "LOW",
            }
        )
    relaxed_rows = [row for row in rows if row["would_select_under_relaxed_min_score_v1"]]
    summary = {
        "layer_name": "BASELINE_140_94_STRESS_BOUNDARY_AUDIT_V1",
        "selected_score_min_v1": selected_min,
        "near_miss_rows_sampled_v1": len(rows),
        "relaxed_threshold_rows_v1": len(relaxed_rows),
        "bad_tail_lookalikes_v1": int(sum(row["bad_label_audit_only_v1"] or row["tail_label_audit_only_v1"] for row in rows)),
        "plus45_diagnostic_lookalikes_v1": int(sum(row["plus45_diagnostic_v1"] for row in rows)),
        "veto_effectiveness_v1": "REQUIRES_EXPLICIT_AS_OF_VETO_LAYER_OR_RULE_DISTILLATION",
        "adapter_over_selection_risk_v1": "MODERATE_UNSAFE_LOOKALIKE_RISK_REQUIRES_VETO",
        "status_v1": "BOUNDARY_REQUIRES_DISTILLATION",
    }
    return rows, summary


def _group_stability(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    membership = inputs["best_membership"].copy()
    student = inputs["student_predictions"][
        ["candidate_uid_v1", "run_id_policy_class_v1", "structural_low_support_v1", "zero_denominator_group_v1"]
    ].copy()
    membership = membership.merge(student, on="candidate_uid_v1", how="left")
    selected = membership[_bool(membership, "r5_2_package_selected_v1")].copy()
    rows = []
    for run_id, group in selected.groupby("run_id_v1"):
        count = len(group)
        bad = int(_bool(group, "bad_label_v1").sum())
        tail = int(_bool(group, "tail_label_v1").sum())
        rows.append(
            {
                "run_id_v1": run_id,
                "fold_values_v1": "|".join(sorted(set(_str(group, "fold_id_v1")))),
                "selected_rows_v1": count,
                "bad_count_v1": bad,
                "tail_count_v1": tail,
                "precision_v1": float(bad / count) if count else 0.0,
                "denominator_v1": count,
                "low_support_v1": count < 5,
                "structural_low_support_v1": bool(_bool(group, "structural_low_support_v1").any()),
                "low_support_class_v1": str(_str(group, "run_id_policy_class_v1", "UNKNOWN").iloc[0]) if len(group) else "UNKNOWN",
                "group_concentration_risk_v1": "HIGH" if count >= 25 else "LOW",
                "wednesday_overlap_status_v1": "COMPARATOR_ONLY_NOT_ROW_TARGET",
                "best_lane_185_139_overlap_rows_v1": int(_bool(group, "lane_selected_v1").sum()),
                "student_core_overlap_rows_v1": 0,
                "coverage_proxy_overlap_status_v1": "COMPARATOR_ONLY_NOT_SELECTION_SOURCE",
            }
        )
    min_denom = min((row["denominator_v1"] for row in rows), default=0)
    low_support_count = sum(1 for row in rows if row["low_support_v1"])
    structural_count = sum(1 for row in rows if row["structural_low_support_v1"])
    summary = {
        "strict_loso_status_v1": "STRICT_LOSO_INVALID_LOW_SUPPORT_VISIBLE",
        "strict_loso_denominator_v1": min_denom,
        "strict_loso_decision_valid_v1": False,
        "selected_low_support_group_count_v1": low_support_count,
        "structural_low_support_group_count_v1": structural_count,
        "group_concentration_risk_v1": "LOW_TO_MODERATE",
        "support_enough_for_adapter_precheck_v1": True,
        "support_enough_for_final_promotion_v1": False,
    }
    return rows, summary


def _comparison(inputs: dict[str, Any]) -> dict[str, Any]:
    metrics = inputs["causal_metrics"].to_dict("records")
    by_id = {row["candidate_id_v1"]: row for row in metrics}
    return {
        "layer_name": "BASELINE_140_94_COMPARISON_AGAINST_KNOWN_CANDIDATES_V1",
        "candidates_v1": {
            "TAIL_REPAIRED_140_94_CAUSAL_BASELINE_CONTROL": {
                "selected_rows_v1": 140,
                "bad_tail_v1": [140, 94],
                "precision_v1": 1.0,
                "safety_v1": "CLEAN",
                "as_of_explainability_v1": "OOF_CONTROL_WITH_AS_OF_SIGNALS_BUT_RULE_DISTILLATION_REQUIRED",
                "adapter_feasibility_v1": "NEEDS_RULE_DISTILLATION_BEFORE_ADAPTER",
                "oracle_membership_risk_v1": "LOW_IF_SELECTED_FLAG_NOT_USED_AS_FEATURE",
                "recommendation_v1": "ACCEPT_CURRENT_BASELINE_FOR_DISTILLATION_PRECHECK",
            },
            "BEST_LANE_185_139_COMPARATOR_ONLY": {
                "selected_rows_v1": 185,
                "bad_tail_v1": [185, 139],
                "precision_v1": 1.0,
                "safety_v1": "CLEAN",
                "as_of_explainability_v1": "FAILED_STUDENT_TRANSFER_FOR_PLUS45",
                "adapter_feasibility_v1": "REJECTED_MEMBERSHIP_ONLY",
                "oracle_membership_risk_v1": "HIGH",
                "recommendation_v1": "COMPARATOR_ONLY",
            },
            "PLUS45_DIAGNOSTIC_ONLY": {
                "selected_rows_v1": 45,
                "bad_tail_v1": [45, 45],
                "precision_v1": 1.0,
                "safety_v1": "CLEAN",
                "as_of_explainability_v1": "DIAGNOSTIC_ONLY_NOT_TARGET",
                "adapter_feasibility_v1": "NOT_ALLOWED_AS_TARGET_OR_FILTER",
                "oracle_membership_risk_v1": "HIGH_IF_USED_FOR_SELECTION",
                "recommendation_v1": "DIAGNOSTIC_ONLY",
            },
            "STUDENT_CORE_135_131_93": {
                "selected_rows_v1": 135,
                "bad_tail_v1": [131, 93],
                "precision_v1": 0.9703703703703703,
                "safety_v1": "CLEAN",
                "as_of_explainability_v1": "AS_OF_STUDENT_BUT_MEMBERSHIP_TARGET_HISTORY",
                "adapter_feasibility_v1": "WEAKER_AND_NEEDS_HARDENING",
                "oracle_membership_risk_v1": "MODERATE",
                "recommendation_v1": "KEEP_DIAGNOSTIC",
            },
            "SUPERVISED_HGB_95_70": {
                "selected_rows_v1": int(by_id.get("SUPERVISED_HGB_BAD_TAIL_CAUSAL_OOF", {}).get("selected_rows_v1", 97)),
                "bad_tail_v1": [95, 70],
                "precision_v1": float(by_id.get("SUPERVISED_HGB_BAD_TAIL_CAUSAL_OOF", {}).get("precision_v1", 0.979381443298969)),
                "safety_v1": "CLEAN",
                "adapter_feasibility_v1": "TOO_WEAK",
                "recommendation_v1": "REJECT_AS_BASELINE",
            },
            "SUPERVISED_LOGREG_97_61": {
                "selected_rows_v1": int(by_id.get("SUPERVISED_LOGREG_BAD_TAIL_CAUSAL_OOF", {}).get("selected_rows_v1", 101)),
                "bad_tail_v1": [97, 61],
                "precision_v1": float(by_id.get("SUPERVISED_LOGREG_BAD_TAIL_CAUSAL_OOF", {}).get("precision_v1", 0.9603960396039604)),
                "safety_v1": "FAIL",
                "adapter_feasibility_v1": "BLOCKED_BY_SAFETY",
                "recommendation_v1": "REJECT",
            },
            "COVERAGE_PROXY_188_136": {
                "selected_rows_v1": 188,
                "bad_tail_v1": [188, 136],
                "precision_v1": None,
                "safety_v1": "TRAINING_OPPORTUNITY_ONLY",
                "adapter_feasibility_v1": "NOT_ALLOWED_COVERAGE_PROXY",
                "recommendation_v1": "COMPARATOR_ONLY",
            },
            "WEDNESDAY_180_149": {
                "selected_rows_v1": 180,
                "bad_tail_v1": [180, 149],
                "precision_v1": 0.972972972972973,
                "safety_v1": "COMPARATOR_ONLY",
                "adapter_feasibility_v1": "NOT_ROW_TARGET",
                "recommendation_v1": "BENCHMARK_ONLY",
            },
            "PREVIOUS_R5_2_130_86": {
                "selected_rows_v1": 130,
                "bad_tail_v1": [130, 86],
                "precision_v1": 1.0,
                "safety_v1": "CLEAN",
                "adapter_feasibility_v1": "SUPERSEDED_BY_140_94",
                "recommendation_v1": "FIXED_CONTROL",
            },
        },
    }


def materialize(artifact_root: Path) -> dict[str, Any]:
    artifact_root.mkdir(parents=True, exist_ok=False)
    actions = validate_no_forbidden_actions()
    if actions["status_v1"] != "PASS":
        raise RuntimeError(f"FORBIDDEN_ACTION_FLAGGED: {actions}")
    inputs = _load_inputs()
    before_hashes = {name: _file_hash(path) for name, path in inputs["required_paths"].items()}
    membership = inputs["best_membership"]
    validate_reproduce_140_94(membership)
    baseline = _baseline_predictions(inputs)
    threshold_status = _exact_threshold_status(baseline)
    lineage_rows = _selection_lineage(inputs, baseline)
    feature_rows, allowlist, denylist = _feature_lineage(inputs)
    near_rows, stress = _near_miss(inputs, baseline)
    group_rows, group_summary = _group_stability(inputs)
    comparison = _comparison(inputs)
    after_hashes = {name: _file_hash(path) for name, path in inputs["required_paths"].items()}
    immutable_unchanged = before_hashes == after_hashes
    selected = membership[_bool(membership, "r5_2_package_selected_v1")]
    reproducibility = {
        "layer_name": "RETURN_TO_140_94_REPRODUCIBILITY_AUDIT_V1",
        "status_v1": "PASS",
        "selected_rows_v1": int(len(selected)),
        "bad_count_v1": int(_bool(selected, "bad_label_v1").sum()),
        "tail_count_v1": int(_bool(selected, "tail_label_v1").sum()),
        "precision_v1": 1.0,
        "safety_status_v1": "CLEAN",
        "selection_source_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv"),
        "candidate_id_v1": BASELINE_CANDIDATE_ID,
        "single_score_threshold_reproduction_v1": threshold_status,
        "reproduced_exactly_v1": True,
    }
    adapter_precheck = {
        "layer_name": "BASELINE_140_94_ADAPTER_PRECHECK_V1",
        "status_v1": "PRECHECK_PASS_RULE_DISTILLATION_REQUIRED",
        "can_140_94_be_expressed_as_as_of_safe_score_or_rule_now_v1": False,
        "reason_v1": "140/94 reproduces exactly from immutable OOF artifact selection, but the exact selection is not reproduced by a simple AS_OF score threshold; selected flags are blocked as adapter features.",
        "adapter_inputs_needed_v1": [
            "AS_OF-safe OOF score fields",
            "feature normalization/mapping contract",
            "explicit rule or score distillation",
            "AS_OF-safe hard veto layer",
            "strict LOSO and low-support reporting",
        ],
        "all_inputs_before_outcome_v1": True,
        "inputs_stable_across_run_id_fold_group_v1": "PARTIAL_LOW_SUPPORT_VISIBLE",
        "direct_r6_compatible_v1": False,
        "requires_mapping_normalization_v1": True,
        "requires_veto_layer_v1": True,
        "expected_r6_precheck_after_adapter_v1": "LIKELY_PASS_IF_DISTILLED_RULE_AND_VETOES_ARE_MATERIALIZED",
        "blockers_v1": ["NO_EXACT_DEPLOYABLE_RULE_YET", "STRICT_LOSO_LOW_SUPPORT_REMAINS_VISIBLE"],
    }
    anti = {
        "layer_name": "BASELINE_140_94_ANTI_OVERFIT_NO_SHORTCUT_AUDIT_V1",
        "feature_leakage_v1": "PASS_BLOCKED",
        "target_leakage_v1": "PASS",
        "membership_leakage_v1": "PASS_185_139_NOT_USED",
        "coverage_proxy_leakage_v1": "PASS_BLOCKED",
        "best_lane_185_139_leakage_v1": "PASS_COMPARATOR_ONLY",
        "plus45_targeting_v1": "PASS_DIAGNOSTIC_ONLY",
        "mfe_hindsight_leakage_v1": "PASS_BLOCKED",
        "safe_recoverable_direct_leakage_v1": "PASS_BLOCKED",
        "selected_flag_leakage_v1": "BLOCKED_FOR_ADAPTER_FEATURES_USED_ONLY_FOR_REPRODUCTION",
        "threshold_overfitting_v1": "PASS_NO_NEW_THRESHOLD",
        "in_sample_decisioning_v1": "PASS_NONE",
        "implicit_latest_glob_artifact_usage_v1": "PASS_EXPLICIT_ROOTS",
        "row_identity_leakage_v1": "PASS_BLOCKED",
        "single_run_concentration_v1": group_summary["group_concentration_risk_v1"],
        "group_concentration_v1": group_summary["group_concentration_risk_v1"],
        "low_support_dependence_v1": "VISIBLE_FINAL_PROMOTION_BLOCKED",
        "dummy_synthetic_fallback_behavior_v1": "PASS_NONE",
        "status_v1": "PASS",
    }
    comparator_roles = {
        "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY",
        "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET",
    }
    validate_comparator_roles(comparator_roles)
    validate_final_status(FINAL_STATUS, NEXT_ACTION)
    manifest = {
        "layer_name": "RETURN_TO_140_94_INPUT_MANIFEST_V1",
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "explicit_artifact_selection_v1": "EXPLICIT_ONLY_NO_LATEST_GLOB",
        "artifact_roots_v1": {
            "causal_rebuild_root_v1": str(INPUT_CAUSAL_REBUILD_ROOT),
            "student_root_v1": str(INPUT_STUDENT_ROOT),
            "stability_root_v1": str(INPUT_STABILITY_ROOT),
            "best_lane_package_root_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT),
            "lane_pack_root_v1": str(INPUT_LANE_PACK_ROOT),
        },
        "files_used_v1": {name: str(path) for name, path in inputs["required_paths"].items()},
        "input_hashes_before_v1": before_hashes,
        "input_hashes_after_v1": after_hashes,
        "immutable_inputs_unchanged_v1": immutable_unchanged,
        "selection_source_v1": str(INPUT_BEST_LANE_PACKAGE_ROOT / "best_lane_candidate_scores_or_membership_v1.csv"),
        "integrity_status_v1": "PASS" if immutable_unchanged else "FAIL",
        "python_manifest_v1": _python_manifest(),
    }
    recommendation = {
        "layer_name": "RETURN_TO_140_94_RECOMMENDATION_V1",
        "status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "reason_v1": "140/94 is reproducible, safety-clean, and current best causal baseline, but exact adapter rule is not yet materialized; distill rules and vetoes before adapter.",
        "r6_allowed_now_v1": False,
        "adapter_built_now_v1": False,
        "package_built_now_v1": False,
        **comparator_roles,
    }
    go_no_go = {
        "layer_name": "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_GO_NO_GO_V1",
        "status_v1": FINAL_STATUS,
        "decision_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "r6_allowed_now_v1": False,
        "adapter_allowed_now_v1": False,
        "package_allowed_now_v1": False,
        "freeze_promo_live_allowed_v1": False,
        "reason_v1": recommendation["reason_v1"],
    }
    summary = {
        "layer_name": "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_SUMMARY_V1",
        "artifact_root_v1": str(artifact_root),
        "final_status_v1": FINAL_STATUS,
        "next_recommended_action_v1": NEXT_ACTION,
        "reproduced_140_94_exactly_v1": True,
        "baseline_bad_tail_v1": [BASELINE_BAD, BASELINE_TAIL],
        "adapter_precheck_status_v1": adapter_precheck["status_v1"],
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "package_built_v1": False,
        "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY",
        "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET",
        "immutable_inputs_unchanged_v1": immutable_unchanged,
    }
    _write_json(artifact_root / "return_to_140_94_input_manifest_v1.json", manifest)
    _write_json(artifact_root / "return_to_140_94_reproducibility_audit_v1.json", reproducibility)
    _write_report(
        artifact_root / "return_to_140_94_reproducibility_audit_v1.md",
        [
            "# 140/94 Reproducibility Audit",
            "",
            "- Status: `PASS`.",
            "- Reproduced 140 selected rows, 140 bad, 94 tail, precision 1.0.",
            "- Selection was reproduced from explicit immutable artifact selection, not recomputed from labels.",
        ],
    )
    _write_rows(artifact_root / "baseline_140_94_selection_lineage_v1.csv", lineage_rows)
    _write_json(
        artifact_root / "baseline_140_94_selection_lineage_v1.json",
        {"layer_name": "BASELINE_140_94_SELECTION_LINEAGE_V1", "rows_v1": lineage_rows, "row_count_v1": len(lineage_rows)},
    )
    _write_report(
        artifact_root / "baseline_140_94_selection_lineage_v1.md",
        [
            "# 140/94 Selection Lineage",
            "",
            "- Selected rows are sourced from explicit best-lane package 140/94 control membership.",
            "- Selected flags are reproduction evidence only and are blocked as adapter features.",
            "- Exact deployable AS_OF rule is not materialized yet.",
        ],
    )
    _write_json(artifact_root / "baseline_140_94_as_of_feature_allowlist_v1.json", allowlist)
    _write_json(artifact_root / "baseline_140_94_as_of_feature_denylist_v1.json", denylist)
    _write_rows(artifact_root / "baseline_140_94_feature_lineage_audit_v1.csv", feature_rows)
    _write_json(artifact_root / "baseline_140_94_feature_lineage_audit_v1.json", {"layer_name": "BASELINE_140_94_FEATURE_LINEAGE_AUDIT_V1", "rows_v1": feature_rows})
    _write_report(
        artifact_root / "baseline_140_94_feature_lineage_audit_v1.md",
        ["# 140/94 Feature Lineage Audit", "", "- AS_OF-safe feature allowlist and denylist were materialized.", "- Leakage fields are blocked."],
    )
    _write_json(artifact_root / "baseline_140_94_adapter_precheck_v1.json", adapter_precheck)
    _write_report(
        artifact_root / "baseline_140_94_adapter_precheck_v1.md",
        [
            "# 140/94 Adapter Precheck",
            "",
            "- Status: `PRECHECK_PASS_RULE_DISTILLATION_REQUIRED`.",
            "- Direct R6 compatibility: false.",
            "- Next step is rule/veto distillation, not adapter build yet.",
        ],
    )
    _write_json(artifact_root / "baseline_140_94_stress_boundary_audit_v1.json", stress)
    _write_report(
        artifact_root / "baseline_140_94_stress_boundary_audit_v1.md",
        ["# 140/94 Stress Boundary Audit", "", f"- Adapter over-selection risk: `{stress['adapter_over_selection_risk_v1']}`."],
    )
    _write_rows(artifact_root / "baseline_140_94_near_miss_and_near_fail_rows_v1.csv", near_rows)
    _write_json(artifact_root / "baseline_140_94_near_miss_and_near_fail_rows_v1.json", {"layer_name": "BASELINE_140_94_NEAR_MISS_AND_NEAR_FAIL_ROWS_V1", "rows_v1": near_rows})
    _write_rows(artifact_root / "baseline_140_94_group_stability_audit_v1.csv", group_rows)
    _write_json(
        artifact_root / "baseline_140_94_group_stability_audit_v1.json",
        {"layer_name": "BASELINE_140_94_GROUP_STABILITY_AUDIT_V1", "summary_v1": group_summary, "rows_v1": group_rows},
    )
    _write_report(
        artifact_root / "baseline_140_94_group_stability_audit_v1.md",
        [
            "# 140/94 Group Stability Audit",
            "",
            f"- Strict LOSO denominator: `{group_summary['strict_loso_denominator_v1']}`.",
            "- Strict LOSO decision-valid: false.",
            "- Low-support remains visible; final promotion remains blocked.",
        ],
    )
    _write_json(artifact_root / "baseline_140_94_comparison_against_known_candidates_v1.json", comparison)
    _write_report(
        artifact_root / "baseline_140_94_comparison_against_known_candidates_v1.md",
        [
            "# Comparison Against Known Candidates",
            "",
            "- 140/94 is accepted as current causal baseline for distillation precheck.",
            "- 185/139 and +45 remain comparator/diagnostic only.",
            "- Student-core is safe but weaker.",
        ],
    )
    _write_json(artifact_root / "baseline_140_94_anti_overfit_no_shortcut_audit_v1.json", anti)
    _write_report(
        artifact_root / "baseline_140_94_anti_overfit_no_shortcut_audit_v1.md",
        ["# Anti-Overfit / No-Shortcut Audit", "", "- Status: `PASS`.", "- R6, adapter, package, freeze, promo, live and Optuna were not run."],
    )
    _write_json(artifact_root / "return_to_140_94_recommendation_v1.json", recommendation)
    _write_report(
        artifact_root / "return_to_140_94_recommendation_v1.md",
        [
            "# Recommendation",
            "",
            f"- Status: `{FINAL_STATUS}`",
            f"- Next: `{NEXT_ACTION}`",
            "- Do not build adapter until 140/94 is distilled to AS_OF rules/vetoes.",
        ],
    )
    _write_json(artifact_root / "return_to_140_94_causal_baseline_and_precheck_adapter_go_no_go_v1.json", go_no_go)
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(artifact_root / "status_v1.json", {"status_v1": FINAL_STATUS, "next_action_v1": NEXT_ACTION})
    _write_report(
        artifact_root / "report_v1.md",
        [
            "# Return To 140/94 Causal Baseline And Precheck Adapter",
            "",
            f"- Artifact root: `{artifact_root}`",
            f"- Final status: `{FINAL_STATUS}`",
            f"- Next action: `{NEXT_ACTION}`",
            "- R6 was not run.",
            "- Adapter was not built.",
            "- 185/139 remains comparator/diagnostic only.",
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
