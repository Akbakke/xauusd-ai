#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
ACTION = "BUILD_R5_2_PACKAGE_FROM_CANDIDATE_REQUIRES_EXPLICIT_GATE_V1"
LAYER_NAME = ACTION
SOURCE_CANDIDATE_ROOT = DEFAULT_REPORTS_ROOT / "BUILD_R5_2_FROM_COVERAGE_AWARE_OPPORTUNITY_BASE_WITH_FIXED_CONTROLS_V1_20260427T150214Z_LOCK"
SELECTED_THRESHOLD = "RECALL"
PACKAGE_TYPE = "R5_2_CANDIDATE_PACKAGE_NOT_PROMOTED"
NEXT_REQUIRED_GATE = "RUN_R6_RETRAIN_FROM_R5_2_CANDIDATE_PACKAGE_EXPLICIT_GATE_V1"

REQUIRED_FILE_MAP = {
    "r5_2_oof_scores_v1.csv": "r5_2_candidate_oof_scores_v1.csv",
    "r5_2_oof_score_provenance_v1.csv": "r5_2_candidate_oof_score_provenance_v1.csv",
    "r5_2_oof_fold_assignment_v1.csv": "r5_2_candidate_oof_fold_assignment_v1.csv",
    "r5_2_train_validation_membership_v1.csv": "r5_2_candidate_train_validation_membership_v1.csv",
    "r5_2_score_source_manifest_v1.json": "r5_2_candidate_score_source_manifest_v1.json",
    "r5_2_feature_label_hash_manifest_v1.json": "r5_2_candidate_feature_label_hash_manifest_v1.json",
    "r5_2_oof_metric_denominator_report_v1.csv": "r5_2_candidate_metric_denominator_report_v1.csv",
    "r5_2_oof_metric_denominator_report_v1.json": "r5_2_candidate_metric_denominator_report_v1.json",
    "r5_2_oof_safety_report_v1.csv": "r5_2_candidate_safety_report_v1.csv",
    "r5_2_oof_safety_report_v1.json": "r5_2_candidate_safety_report_v1.json",
    "r5_2_oof_low_support_report_v1.csv": "r5_2_candidate_low_support_report_v1.csv",
    "r5_2_oof_low_support_report_v1.json": "r5_2_candidate_low_support_report_v1.json",
    "r5_2_oof_fixed_control_comparison_v1.csv": "r5_2_candidate_fixed_control_comparison_v1.csv",
    "r5_2_oof_fixed_control_comparison_v1.json": "r5_2_candidate_fixed_control_comparison_v1.json",
    "r5_2_threshold_selection_report_v1.md": "r5_2_candidate_threshold_selection_report_v1.md",
    "r5_2_threshold_candidate_grid_v1.csv": "r5_2_candidate_threshold_selection_report_v1.csv",
    "r5_2_threshold_candidate_grid_v1.json": "r5_2_candidate_threshold_selection_report_v1.json",
    "r5_2_no_in_sample_decisioning_attestation_v1.json": "r5_2_candidate_no_in_sample_decisioning_attestation_v1.json",
    "no_fallback_no_dummy_no_synthetic_attestation_v1.json": "r5_2_candidate_no_fallback_no_dummy_no_synthetic_attestation_v1.json",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA:
        return None
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


def _write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing required json artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact for hash: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING_LOCAL_ARTIFACT"


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_forbidden_actions(*, optuna: bool, r6: bool, promoted: bool, freeze: bool, live: bool) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if promoted:
        failures.append("PROMOTION_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_not_promoted(payload: dict[str, Any]) -> bool:
    forbidden = [
        bool(payload.get("promoted_v1")),
        bool(payload.get("freeze_ready_v1")),
        bool(payload.get("live_ready_v1")),
        bool(payload.get("final_promotion_allowed_v1")),
    ]
    if any(forbidden):
        raise RuntimeError("CANDIDATE_PACKAGE_CANNOT_BE_PROMOTED_OR_LIVE_READY")
    return True


def validate_strict_loso_visible(metrics: dict[str, Any]) -> bool:
    if bool(metrics.get("strict_all_run_id_decision_valid_v1")):
        raise RuntimeError("STRICT_LOSO_INVALIDITY_MUST_NOT_BE_HIDDEN")
    if int(metrics.get("strict_all_run_id_worst_loso_denominator_v1", 0) or 0) != 2:
        raise RuntimeError("STRICT_LOSO_DENOMINATOR_MUST_REMAIN_2")
    return True


def validate_metric_preservation(metrics: dict[str, Any]) -> bool:
    expected = {
        "threshold_candidate_id_v1": SELECTED_THRESHOLD,
        "bad_count_v1": 130,
        "tail_count_v1": 86,
        "precision_v1": 1.0,
        "precision_denominator_v1": 130,
        "precision_decision_valid_v1": True,
        "strict_all_run_id_worst_loso_denominator_v1": 2,
        "strict_all_run_id_decision_valid_v1": False,
        "safety_clean_v1": True,
    }
    mismatches = {
        key: {"expected_v1": value, "observed_v1": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Candidate package metric preservation failure: {mismatches}")
    return True


def validate_final_fit_policy(policy: dict[str, Any]) -> bool:
    if policy.get("status_v1") == "FINAL_FIT_CREATED_NON_EVAL_FUTURE_SCORING_ONLY":
        if policy.get("metrics_used_for_decisioning_v1"):
            raise RuntimeError("FINAL_FIT_METRICS_CANNOT_BE_DECISIONING_EVIDENCE")
    return True


def r6_precheck_authorizes_r6(precheck: dict[str, Any]) -> bool:
    return bool(precheck.get("r6_run_authorized_v1"))


def _package_id(output_dir: Path) -> str:
    return output_dir.name


def _python_manifest() -> dict[str, Any]:
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, timeout=30).splitlines()
    except Exception as exc:  # pragma: no cover - defensive only
        freeze = [f"PIP_FREEZE_UNAVAILABLE: {exc}"]
    return {
        "python_executable_v1": sys.executable,
        "python_version_v1": sys.version,
        "platform_v1": platform.platform(),
        "pip_freeze_sha256_v1": hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest(),
        "pip_freeze_v1": freeze,
    }


def _load_source_metrics(source_root: Path) -> dict[str, Any]:
    best = _read_json(source_root / "r5_2_best_candidate_from_coverage_aware_base_v1.json")
    metrics = best.get("best_threshold_candidate_v1") or {}
    validate_metric_preservation(metrics)
    validate_strict_loso_visible(metrics)
    return metrics


def _source_required_files(source_root: Path) -> list[dict[str, Any]]:
    rows = []
    missing = []
    for source_name, package_name in REQUIRED_FILE_MAP.items():
        source_path = source_root / source_name
        if not source_path.exists():
            missing.append(str(source_path))
        rows.append(
            {
                "source_name_v1": source_name,
                "package_name_v1": package_name,
                "source_path_v1": str(source_path),
                "source_exists_v1": source_path.exists(),
            }
        )
    if missing:
        raise RuntimeError(f"Missing required source artifact(s): {missing}")
    return rows


def _copy_required_files(source_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for source_name, package_name in REQUIRED_FILE_MAP.items():
        source_path = source_root / source_name
        package_path = output_dir / package_name
        shutil.copy2(source_path, package_path)
        source_hash = _file_hash(source_path)
        package_hash = _file_hash(package_path)
        if source_hash != package_hash:
            raise RuntimeError(f"Package file hash mismatch for {source_name} -> {package_name}")
        rows.append(
            {
                "source_name_v1": source_name,
                "package_name_v1": package_name,
                "source_path_v1": str(source_path),
                "package_path_v1": str(package_path),
                "source_sha256_v1": source_hash,
                "package_sha256_v1": package_hash,
                "hash_match_v1": True,
            }
        )
    return rows


def _contract(source_root: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": "R5_2_CANDIDATE_PACKAGE_CONTRACT_V1",
        "package_type_v1": PACKAGE_TYPE,
        "source_candidate_root_v1": str(source_root),
        "selected_threshold_candidate_v1": SELECTED_THRESHOLD,
        "oof_result_bad_tail_v1": [metrics["bad_count_v1"], metrics["tail_count_v1"]],
        "safety_status_v1": "CLEAN",
        "strict_loso_status_v1": "INVALID_DUE_TO_STRUCTURAL_LOW_SUPPORT",
        "final_promotion_allowed_v1": False,
        "r6_ready_v1": False,
        "r6_ready_condition_v1": "SEPARATE_EXPLICIT_R6_GATE_REQUIRED",
        "freeze_live_allowed_v1": False,
        "may_be_used_as_v1": [
            "candidate package",
            "R6 retrain input after explicit gate",
            "audit/comparison artifact",
        ],
        "may_not_be_used_as_v1": [
            "final promoted R5.2",
            "live model",
            "freeze model",
            "proof that structural low-support is resolved",
        ],
    }


def _final_fit_policy() -> dict[str, Any]:
    policy = {
        "layer_name": "R5_2_CANDIDATE_FINAL_FIT_POLICY_V1",
        "status_v1": "FINAL_FIT_NOT_CREATED_NOT_REQUIRED",
        "final_fit_created_v1": False,
        "final_fit_role_v1": "NOT_APPLICABLE",
        "metrics_used_for_decisioning_v1": False,
        "oof_metrics_remain_only_evaluation_evidence_v1": True,
        "reason_v1": "Existing candidate package conventions for this gated artifact only require OOF candidate evidence and R6 input references; no final-fit scoring artifact is required before the separate R6 gate.",
    }
    validate_final_fit_policy(policy)
    return policy


def _manifest(
    *,
    output_dir: Path,
    source_root: Path,
    metrics: dict[str, Any],
    copied_files: list[dict[str, Any]],
    final_fit_policy: dict[str, Any],
) -> dict[str, Any]:
    source_manifest = _read_json(source_root / "r5_2_score_source_manifest_v1.json")
    hash_manifest = _read_json(source_root / "r5_2_feature_label_hash_manifest_v1.json")
    summary = _read_json(source_root / "summary_v1.json")
    hashes = hash_manifest.get("hashes_v1") or {}
    package_files = {row["package_name_v1"]: row["package_path_v1"] for row in copied_files}
    python = _python_manifest()
    return {
        "layer_name": "R5_2_CANDIDATE_PACKAGE_MANIFEST_V1",
        "package_id_v1": _package_id(output_dir),
        "package_root_v1": str(output_dir),
        "package_type_v1": PACKAGE_TYPE,
        "input_candidate_root_v1": str(source_root),
        "selected_threshold_candidate_v1": SELECTED_THRESHOLD,
        "selected_threshold_parameters_v1": {
            "bad_threshold_v1": metrics["bad_threshold_v1"],
            "tail_threshold_v1": metrics["tail_threshold_v1"],
            "hard_veto_max_v1": metrics["hard_veto_max_v1"],
            "policy_v1": metrics["policy_v1"],
        },
        "model_family_v1": source_manifest.get("model_family_v1"),
        "feature_count_v1": source_manifest.get("feature_count_v1"),
        "feature_families_v1": source_manifest.get("feature_families_v1"),
        "feature_matrix_hash_v1": hashes.get("feature_matrix_hash_v1"),
        "label_table_hash_v1": hashes.get("label_table_hash_v1"),
        "config_hash_v1": hashes.get("config_hash_v1"),
        "source_code_hash_v1": hashes.get("source_hash_v1"),
        "v2_training_utility_source_hash_v1": hashes.get("v2_training_utility_source_hash_v1"),
        "dependency_manifest_hash_v1": python["pip_freeze_sha256_v1"],
        "python_executable_v1": python["python_executable_v1"],
        "python_version_v1": python["python_version_v1"],
        "foundation_rows_v1": summary.get("foundation_rows_v1"),
        "active_rows_v1": summary.get("active_rows_v1"),
        "quarantine_rows_v1": summary.get("quarantine_rows_v1"),
        "as_of_columns_v1": summary.get("as_of_columns_v1"),
        "oof_score_file_path_v1": package_files["r5_2_candidate_oof_scores_v1.csv"],
        "oof_provenance_file_path_v1": package_files["r5_2_candidate_oof_score_provenance_v1.csv"],
        "fold_assignment_path_v1": package_files["r5_2_candidate_oof_fold_assignment_v1.csv"],
        "train_validation_membership_path_v1": package_files["r5_2_candidate_train_validation_membership_v1.csv"],
        "score_source_manifest_path_v1": package_files["r5_2_candidate_score_source_manifest_v1.json"],
        "safety_report_path_v1": package_files["r5_2_candidate_safety_report_v1.json"],
        "metric_denominator_report_path_v1": package_files["r5_2_candidate_metric_denominator_report_v1.json"],
        "low_support_report_path_v1": package_files["r5_2_candidate_low_support_report_v1.json"],
        "fixed_control_comparison_path_v1": package_files["r5_2_candidate_fixed_control_comparison_v1.json"],
        "no_dummy_synthetic_fallback_attestation_path_v1": package_files["r5_2_candidate_no_fallback_no_dummy_no_synthetic_attestation_v1.json"],
        "final_fit_policy_v1": final_fit_policy,
        "final_promotion_allowed_v1": False,
        "strict_loso_decision_valid_v1": False,
        "reason_v1": "STRUCTURAL_LOW_SUPPORT_REMAINS",
        "next_required_gate_v1": NEXT_REQUIRED_GATE,
    }


def _integrity_report(source_root: Path, output_dir: Path, metrics: dict[str, Any], copied_files: list[dict[str, Any]]) -> dict[str, Any]:
    source_summary = _read_json(source_root / "summary_v1.json")
    source_go = _read_json(source_root / "r5_2_from_coverage_aware_base_go_no_go_v1.json")
    source_no_sample = _read_json(source_root / "r5_2_no_in_sample_decisioning_attestation_v1.json")
    source_no_fallback = _read_json(source_root / "no_fallback_no_dummy_no_synthetic_attestation_v1.json")
    required_package_files = [output_dir / name for name in REQUIRED_FILE_MAP.values()]
    missing_package = [str(path) for path in required_package_files if not path.exists()]
    hash_mismatches = [row for row in copied_files if row["source_sha256_v1"] != row["package_sha256_v1"]]
    checks = {
        "source_candidate_root_exists_v1": source_root.exists(),
        "all_required_package_files_exist_v1": not missing_package,
        "copied_files_match_source_hashes_v1": not hash_mismatches,
        "selected_threshold_is_recall_v1": metrics.get("threshold_candidate_id_v1") == SELECTED_THRESHOLD,
        "oof_bad_tail_remains_130_86_v1": [metrics.get("bad_count_v1"), metrics.get("tail_count_v1")] == [130, 86],
        "precision_remains_1_denominator_130_v1": metrics.get("precision_v1") == 1.0 and metrics.get("precision_denominator_v1") == 130,
        "strict_loso_denominator_remains_2_v1": metrics.get("strict_all_run_id_worst_loso_denominator_v1") == 2,
        "strict_loso_decision_valid_remains_false_v1": metrics.get("strict_all_run_id_decision_valid_v1") is False,
        "safety_remains_clean_v1": metrics.get("safety_clean_v1") is True,
        "low_support_groups_remain_reported_v1": int(metrics.get("selected_low_support_group_count_v1") or 0) > 0,
        "final_promotion_remains_false_v1": source_summary.get("final_promotion_allowed_v1") is False,
        "r6_not_run_v1": source_go.get("r6_run_v1") is False,
        "package_not_promoted_v1": True,
        "no_dummy_synthetic_fallback_v1": source_no_fallback.get("status_v1") == "PASS",
        "no_in_sample_decisioning_v1": source_no_sample.get("status_v1") == "PASS",
        "no_train_validation_overlap_v1": (source_no_sample.get("train_validation_overlap_v1") or {}).get("overlap_count_v1") == 0,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "layer_name": "R5_2_CANDIDATE_PACKAGE_INTEGRITY_REPORT_V1",
        "status_v1": status,
        "checks_v1": checks,
        "missing_package_files_v1": missing_package,
        "hash_mismatches_v1": hash_mismatches,
        "copied_file_count_v1": len(copied_files),
    }


def _r6_precheck(output_dir: Path, integrity: dict[str, Any]) -> dict[str, Any]:
    required = [
        "r5_2_candidate_oof_scores_v1.csv",
        "r5_2_candidate_oof_score_provenance_v1.csv",
        "r5_2_candidate_score_source_manifest_v1.json",
        "r5_2_candidate_threshold_selection_report_v1.json",
        "r5_2_candidate_low_support_report_v1.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        status = "R6_INPUT_PACKAGE_INCOMPLETE"
    elif not integrity["checks_v1"].get("safety_remains_clean_v1"):
        status = "R6_INPUT_PACKAGE_BLOCKED_BY_SAFETY"
    elif not integrity["checks_v1"].get("low_support_groups_remain_reported_v1"):
        status = "R6_INPUT_PACKAGE_BLOCKED_BY_LOW_SUPPORT_FOR_FINAL_PROMOTION_ONLY"
    else:
        status = "R6_INPUT_PACKAGE_READY_BUT_R6_NOT_AUTHORIZED"
    return {
        "layer_name": "R5_2_CANDIDATE_PACKAGE_R6_INPUT_READINESS_PRECHECK_V1",
        "status_v1": status,
        "r6_input_files_present_v1": not missing,
        "missing_r6_input_files_v1": missing,
        "selected_scorefields_exposed_v1": True,
        "threshold_policy_config_exposed_v1": True,
        "provenance_and_low_support_status_exposed_v1": True,
        "final_promotion_blocked_status_exposed_v1": True,
        "r6_run_authorized_v1": False,
        "r6_was_run_v1": False,
        "required_next_action_v1": NEXT_REQUIRED_GATE,
    }


def _go_no_go(integrity: dict[str, Any], precheck: dict[str, Any]) -> dict[str, Any]:
    if integrity["status_v1"] != "PASS":
        decision = "R5_2_CANDIDATE_PACKAGE_BLOCKED_BY_INTEGRITY_FAILURE"
        next_action = "REPAIR_PACKAGE_INTEGRITY_V1"
    elif precheck["status_v1"] == "R6_INPUT_PACKAGE_INCOMPLETE":
        decision = "R5_2_CANDIDATE_PACKAGE_READY_BUT_R6_INPUT_INCOMPLETE"
        next_action = "REPAIR_R5_2_CANDIDATE_PACKAGE_FOR_R6_INPUT_V1"
    else:
        decision = "R5_2_CANDIDATE_PACKAGE_READY_FOR_R6_EXPLICIT_GATE"
        next_action = NEXT_REQUIRED_GATE
    return {
        "layer_name": "R5_2_CANDIDATE_PACKAGE_GO_NO_GO_V1",
        "decision_v1": decision,
        "next_recommended_action_v1": next_action,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
        "package_type_v1": PACKAGE_TYPE,
    }


def _reports(
    output_dir: Path,
    *,
    contract: dict[str, Any],
    integrity: dict[str, Any],
    precheck: dict[str, Any],
    go_no_go: dict[str, Any],
    final_fit_policy: dict[str, Any],
) -> None:
    _write_report(
        output_dir / "r5_2_candidate_package_contract_v1.md",
        [
            "# R5.2 Candidate Package Contract V1",
            "",
            f"Package type: `{contract['package_type_v1']}`",
            f"Selected threshold: `{contract['selected_threshold_candidate_v1']}`",
            f"OOF bad/tail: `{contract['oof_result_bad_tail_v1'][0]}` / `{contract['oof_result_bad_tail_v1'][1]}`",
            f"Strict LOSO status: `{contract['strict_loso_status_v1']}`",
            f"Final promotion allowed: `{contract['final_promotion_allowed_v1']}`",
            f"R6-ready: `{contract['r6_ready_v1']}` until separate explicit gate.",
        ],
    )
    _write_report(
        output_dir / "r5_2_candidate_package_integrity_report_v1.md",
        [
            "# R5.2 Candidate Package Integrity Report V1",
            "",
            f"Status: `{integrity['status_v1']}`",
            f"Copied file count: `{integrity['copied_file_count_v1']}`",
            f"Hash mismatches: `{len(integrity['hash_mismatches_v1'])}`",
            "Strict LOSO invalidity, low-support reporting, and final-promotion block were preserved.",
        ],
    )
    _write_report(
        output_dir / "r5_2_candidate_package_r6_input_readiness_precheck_v1.md",
        [
            "# R5.2 Candidate Package R6 Input Readiness Precheck V1",
            "",
            f"Status: `{precheck['status_v1']}`",
            f"R6 authorized now: `{precheck['r6_run_authorized_v1']}`",
            f"Required next action: `{precheck['required_next_action_v1']}`",
        ],
    )
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Build R5.2 Candidate Package V1",
            "",
            f"Go/no-go: `{go_no_go['decision_v1']}`",
            f"R6 precheck: `{precheck['status_v1']}`",
            f"Final fit policy: `{final_fit_policy['status_v1']}`",
            "R6/freeze/promo/live were not run.",
        ],
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    source_candidate_root: Path = SOURCE_CANDIDATE_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    no_forbidden = validate_no_forbidden_actions(optuna=False, r6=False, promoted=False, freeze=False, live=False)
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)
    if not source_candidate_root.exists():
        raise RuntimeError(f"Source candidate root missing: {source_candidate_root}")

    _source_required_files(source_candidate_root)
    metrics = _load_source_metrics(source_candidate_root)
    contract = _contract(source_candidate_root, metrics)
    validate_not_promoted(contract)
    final_fit_policy = _final_fit_policy()
    copied_files = _copy_required_files(source_candidate_root, output_dir)
    manifest = _manifest(
        output_dir=output_dir,
        source_root=source_candidate_root,
        metrics=metrics,
        copied_files=copied_files,
        final_fit_policy=final_fit_policy,
    )
    integrity = _integrity_report(source_candidate_root, output_dir, metrics, copied_files)
    precheck = _r6_precheck(output_dir, integrity)
    if r6_precheck_authorizes_r6(precheck):
        raise RuntimeError("R6_PRECHECK_MUST_NOT_AUTHORIZE_R6_WITHOUT_EXPLICIT_GATE")
    go_no_go = _go_no_go(integrity, precheck)

    _write_json(output_dir / "r5_2_candidate_package_contract_v1.json", contract)
    _write_json(output_dir / "r5_2_candidate_final_fit_policy_v1.json", final_fit_policy)
    _write_json(output_dir / "r5_2_candidate_package_manifest_v1.json", manifest)
    _write_rows(output_dir / "r5_2_candidate_package_file_hashes_v1.csv", copied_files)
    _write_json(output_dir / "r5_2_candidate_package_integrity_report_v1.json", integrity)
    _write_json(output_dir / "r5_2_candidate_package_r6_input_readiness_precheck_v1.json", precheck)
    _write_json(output_dir / "r5_2_candidate_go_no_go_v1.json", go_no_go)
    _write_json(output_dir / "r5_2_candidate_package_go_no_go_v1.json", go_no_go)
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "package_root_v1": str(output_dir),
        "input_candidate_root_v1": str(source_candidate_root),
        "selected_threshold_candidate_v1": SELECTED_THRESHOLD,
        "bad_count_v1": metrics["bad_count_v1"],
        "tail_count_v1": metrics["tail_count_v1"],
        "precision_v1": metrics["precision_v1"],
        "precision_denominator_v1": metrics["precision_denominator_v1"],
        "precision_decision_valid_v1": metrics["precision_decision_valid_v1"],
        "strict_all_run_id_worst_loso_v1": metrics["strict_all_run_id_worst_loso_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": metrics["strict_all_run_id_worst_loso_denominator_v1"],
        "strict_all_run_id_decision_valid_v1": metrics["strict_all_run_id_decision_valid_v1"],
        "selected_low_support_group_count_v1": metrics["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": metrics["structural_low_support_selected_group_count_v1"],
        "safety_clean_v1": metrics["safety_clean_v1"],
        "final_fit_policy_status_v1": final_fit_policy["status_v1"],
        "package_integrity_status_v1": integrity["status_v1"],
        "r6_input_readiness_precheck_status_v1": precheck["status_v1"],
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
        "go_no_go_v1": go_no_go["decision_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
        "no_forbidden_actions_v1": no_forbidden,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": go_no_go["decision_v1"]})
    _reports(output_dir, contract=contract, integrity=integrity, precheck=precheck, go_no_go=go_no_go, final_fit_policy=final_fit_policy)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--explicit-action", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-candidate-root", type=Path, default=SOURCE_CANDIDATE_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        source_candidate_root=args.source_candidate_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
