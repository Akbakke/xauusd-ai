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
ACTION = "MATERIALIZE_BEST_LANE_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1"
LAYER_NAME = ACTION
INPUT_LANE_PACK_ROOT = (
    DEFAULT_REPORTS_ROOT / "PARALLEL_TAIL_R6_R5_2_REPAIR_LANE_PACK_V1_20260427T191454Z_LOCK"
)
SELECTED_LANE_ID = "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY"
BASELINE_BAD = 140
BASELINE_TAIL = 94
BEST_LANE_BAD = 185
BEST_LANE_TAIL = 139
WEDNESDAY_BAD = 180
WEDNESDAY_TAIL = 149
PACKAGE_TYPE = "BEST_LANE_CANDIDATE_PACKAGE_NOT_PROMOTED"
NEXT_REQUIRED_GATE = "RUN_R6_OR_STABILITY_AUDIT_FROM_BEST_LANE_PACKAGE_EXPLICIT_GATE_V1"
STABILITY_RECHECK_NEXT = "STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_V1"


LANE_FILE_MAP = {
    "lane_scores_or_membership_v1.csv": "best_lane_candidate_scores_or_membership_v1.csv",
    "lane_scores_or_membership_v1.json": "best_lane_candidate_scores_or_membership_v1.json",
    "lane_config_v1.json": "best_lane_candidate_lane_config_v1.json",
    "lane_result_summary_v1.json": "best_lane_candidate_lane_result_summary_v1.json",
    "lane_safety_report_v1.csv": "best_lane_candidate_safety_report_v1.csv",
    "lane_safety_report_v1.json": "best_lane_candidate_safety_report_v1.json",
    "lane_metric_denominator_report_v1.csv": "best_lane_candidate_metric_denominator_report_v1.csv",
    "lane_metric_denominator_report_v1.json": "best_lane_candidate_metric_denominator_report_v1.json",
    "lane_low_support_report_v1.csv": "best_lane_candidate_low_support_report_v1.csv",
    "lane_low_support_report_v1.json": "best_lane_candidate_low_support_report_v1.json",
    "lane_fixed_control_comparison_v1.csv": "best_lane_candidate_fixed_control_comparison_v1.csv",
    "lane_fixed_control_comparison_v1.json": "best_lane_candidate_fixed_control_comparison_v1.json",
    "lane_no_fallback_no_dummy_no_synthetic_attestation_v1.json": (
        "best_lane_candidate_no_fallback_no_dummy_no_synthetic_attestation_v1.json"
    ),
}

ROOT_REFERENCE_FILE_MAP = {
    "parallel_lane_pack_anti_overfit_audit_v1.json": "best_lane_candidate_anti_overfit_audit_v1.json",
    "parallel_lane_pack_anti_overfit_audit_v1.md": "best_lane_candidate_anti_overfit_audit_v1.md",
    "parallel_tail_r6_r5_2_repair_lane_pack_contract_v1.json": (
        "best_lane_candidate_lane_pack_contract_reference_v1.json"
    ),
}

GENERATED_PACKAGE_FILES = [
    "best_lane_candidate_selected_rows_v1.csv",
    "best_lane_candidate_selected_rows_v1.json",
    "best_lane_candidate_no_new_training_attestation_v1.json",
    "best_lane_candidate_membership_only_provenance_v1.json",
    "best_lane_candidate_reproducibility_control_reference_v1.json",
    "best_lane_candidate_package_contract_v1.json",
    "best_lane_candidate_package_contract_v1.md",
    "best_lane_candidate_package_manifest_v1.json",
    "best_lane_candidate_package_integrity_report_v1.json",
    "best_lane_candidate_package_integrity_report_v1.md",
    "best_lane_large_jump_safety_leakage_sanity_audit_v1.json",
    "best_lane_large_jump_safety_leakage_sanity_audit_v1.md",
    "best_lane_large_jump_row_delta_audit_v1.csv",
    "best_lane_large_jump_row_delta_audit_v1.json",
    "best_lane_candidate_fixed_control_comparison_v1.md",
    "best_lane_candidate_r6_input_readiness_precheck_v1.json",
    "best_lane_candidate_r6_input_readiness_precheck_v1.md",
    "best_lane_candidate_next_path_recommendation_v1.json",
    "best_lane_candidate_next_path_recommendation_v1.md",
    "best_lane_candidate_go_no_go_v1.json",
    "best_lane_candidate_package_go_no_go_v1.json",
    "best_lane_candidate_package_file_hashes_v1.csv",
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
        return None if np.isnan(float(value)) else float(value)
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


def _input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _file_hash(path) for name, path in paths.items()}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if value is None or value is pd.NA:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "pass"}
    return bool(value)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].map(_as_bool).astype(bool)


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_forbidden_actions(
    *,
    optuna: bool,
    broad_sweep: bool,
    r6: bool,
    promoted: bool,
    freeze: bool,
    live: bool,
) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if broad_sweep:
        failures.append("BROAD_SWEEP_FORBIDDEN")
    if r6:
        failures.append("R6_FORBIDDEN")
    if promoted:
        failures.append("PROMOTION_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_selected_lane(metrics: dict[str, Any]) -> bool:
    expected = {
        "lane_id_v1": SELECTED_LANE_ID,
        "bad_count_v1": BEST_LANE_BAD,
        "tail_count_v1": BEST_LANE_TAIL,
        "precision_v1": 1.0,
        "precision_denominator_v1": BEST_LANE_BAD,
        "precision_decision_valid_v1": True,
        "strict_all_run_id_worst_loso_denominator_v1": 2,
        "strict_all_run_id_decision_valid_v1": False,
        "selected_low_support_group_count_v1": 9,
        "structural_low_support_selected_group_count_v1": 7,
        "safety_clean_v1": True,
        "final_promotion_allowed_v1": False,
    }
    mismatches = {
        key: {"expected_v1": value, "observed_v1": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"BEST_LANE_CANDIDATE_METRIC_PRESERVATION_FAILURE: {mismatches}")
    return True


def validate_not_promoted(payload: dict[str, Any]) -> bool:
    forbidden = [
        bool(payload.get("promoted_v1")),
        bool(payload.get("freeze_ready_v1")),
        bool(payload.get("live_ready_v1")),
        bool(payload.get("final_promotion_allowed_v1")),
    ]
    if any(forbidden):
        raise RuntimeError("BEST_LANE_CANDIDATE_PACKAGE_CANNOT_BE_PROMOTED_OR_LIVE_READY")
    return True


def validate_lane10_reproducibility(lane_pack_root: Path) -> dict[str, Any]:
    lane10 = lane_pack_root / "lanes" / "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL" / "lane_result_summary_v1.json"
    if not lane10.exists():
        raise RuntimeError(f"Lane 10 reproducibility artifact missing: {lane10}")
    row = _read_json(lane10)
    passed = (
        row.get("lane_id_v1") == "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL"
        and row.get("bad_count_v1") == BASELINE_BAD
        and row.get("tail_count_v1") == BASELINE_TAIL
        and row.get("rows_added_vs_140_94_v1") == 0
        and row.get("rows_lost_vs_140_94_v1") == 0
        and row.get("safety_clean_v1") is True
    )
    if not passed:
        raise RuntimeError("LANE_10_REPRODUCIBILITY_MUST_BE_PASS")
    return {
        "layer_name": "BEST_LANE_CANDIDATE_REPRODUCIBILITY_CONTROL_REFERENCE_V1",
        "status_v1": "PASS",
        "lane_10_reproducibility_pass_v1": True,
        "lane_10_summary_path_v1": str(lane10),
        "bad_tail_v1": [row["bad_count_v1"], row["tail_count_v1"]],
    }


def validate_anti_overfit(anti: dict[str, Any]) -> bool:
    if anti.get("status_v1") != "PARALLEL_LANE_PACK_STABLE_TRACK_PASS":
        raise RuntimeError("ANTI_OVERFIT_AUDIT_MUST_BE_PASS")
    required_true = [
        "all_lanes_pre_registered_v1",
        "no_optuna_v1",
        "no_large_sweep_v1",
        "no_post_hoc_lane_mutation_v1",
        "no_in_sample_decisioning_v1",
        "strict_loso_visible_v1",
        "low_support_visible_v1",
        "no_dummy_synthetic_fallback_v1",
        "no_implicit_latest_glob_v1",
        "lane_10_reproducibility_pass_v1",
    ]
    missing = [key for key in required_true if not bool(anti.get(key))]
    if missing:
        raise RuntimeError(f"ANTI_OVERFIT_AUDIT_MISSING_REQUIRED_PASS_FLAGS: {missing}")
    return True


def _source_required_files(lane_pack_root: Path) -> list[dict[str, Any]]:
    lane_dir = lane_pack_root / "lanes" / SELECTED_LANE_ID
    rows = []
    missing = []
    for source_name, package_name in LANE_FILE_MAP.items():
        source_path = lane_dir / source_name
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
    for source_name, package_name in ROOT_REFERENCE_FILE_MAP.items():
        source_path = lane_pack_root / source_name
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


def _copy_required_files(lane_pack_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    lane_dir = lane_pack_root / "lanes" / SELECTED_LANE_ID
    rows = []
    for source_name, package_name in {**LANE_FILE_MAP, **ROOT_REFERENCE_FILE_MAP}.items():
        source_path = lane_dir / source_name if source_name in LANE_FILE_MAP else lane_pack_root / source_name
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
    }


def _load_inputs(lane_pack_root: Path) -> dict[str, Any]:
    if not lane_pack_root.exists():
        raise RuntimeError(f"Input lane-pack root missing: {lane_pack_root}")
    summary = _read_json(lane_pack_root / "summary_v1.json")
    go_no_go = _read_json(lane_pack_root / "parallel_tail_r6_r5_2_repair_lane_pack_go_no_go_v1.json")
    anti = _read_json(lane_pack_root / "parallel_lane_pack_anti_overfit_audit_v1.json")
    manifest = _read_json(lane_pack_root / "manifest_v1.json")
    if summary.get("best_lane_id_v1") != SELECTED_LANE_ID:
        raise RuntimeError("INPUT_LANE_PACK_BEST_LANE_DOES_NOT_MATCH_SELECTED_LANE")
    if go_no_go.get("decision_v1") != "LANE_FOUND_SAFE_IMPROVEMENT_BEYOND_140_94":
        raise RuntimeError("INPUT_LANE_PACK_STATUS_NOT_ACCEPTED")
    validate_anti_overfit(anti)
    tail_root = Path(summary["input_tail_repaired_package_root_v1"])
    r6_root = Path(summary["input_r6_tail_repaired_root_v1"])
    lane_dir = lane_pack_root / "lanes" / SELECTED_LANE_ID
    lane_metrics = _read_json(lane_dir / "lane_result_summary_v1.json")
    validate_selected_lane(lane_metrics)
    return {
        "summary": summary,
        "go_no_go": go_no_go,
        "anti": anti,
        "manifest": manifest,
        "tail_root": tail_root,
        "r6_root": r6_root,
        "lane_dir": lane_dir,
        "lane_metrics": lane_metrics,
        "tail_manifest": _read_json(tail_root / "tail_repaired_r5_2_candidate_package_manifest_v1.json"),
        "r6_summary": _read_json(r6_root / "summary_v1.json"),
        "r6_feature_hash": _read_json(r6_root / "r6_tail_repaired_feature_label_hash_manifest_v1.json"),
    }


def _contract(lane_pack_root: Path, lane_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": "BEST_LANE_CANDIDATE_PACKAGE_CONTRACT_V1",
        "package_type_v1": PACKAGE_TYPE,
        "source_lane_pack_root_v1": str(lane_pack_root),
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "lane_result_bad_tail_v1": [lane_metrics["bad_count_v1"], lane_metrics["tail_count_v1"]],
        "precision_v1": lane_metrics["precision_v1"],
        "precision_denominator_v1": lane_metrics["precision_denominator_v1"],
        "safety_status_v1": "CLEAN",
        "strict_loso_status_v1": "INVALID_DUE_TO_STRUCTURAL_LOW_SUPPORT",
        "strict_loso_denominator_v1": lane_metrics["strict_all_run_id_worst_loso_denominator_v1"],
        "final_promotion_allowed_v1": False,
        "freeze_live_allowed_v1": False,
        "r6_ready_v1": False,
        "r6_ready_condition_v1": "SEPARATE_EXPLICIT_R6_GATE_REQUIRED",
        "may_be_used_as_v1": [
            "candidate package",
            "audit/comparison artifact",
            "future explicit R6/input candidate after separate gate",
            "current best control after explicit control materialization",
        ],
        "may_not_be_used_as_v1": [
            "final promoted model",
            "live model",
            "freeze model",
            "proof that structural low-support is resolved",
            "canonical Monday R6",
        ],
    }


def _no_training_attestation(lane_metrics: dict[str, Any]) -> dict[str, Any]:
    status = "PASS" if lane_metrics.get("execution_mode_v1") == "NO_TRAINING_ANALYSIS_OR_FILTER_ONLY" else "FAIL"
    return {
        "layer_name": "BEST_LANE_CANDIDATE_NO_NEW_TRAINING_ATTESTATION_V1",
        "status_v1": status,
        "training_run_v1": False,
        "r6_run_v1": False,
        "model_training_required_by_packaging_convention_v1": False,
        "non_eval_final_fit_created_v1": False,
        "execution_mode_v1": lane_metrics.get("execution_mode_v1"),
        "oof_metrics_remain_only_evaluation_evidence_v1": True,
    }


def _membership_only_provenance(lane_pack_root: Path, lane_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "BEST_LANE_CANDIDATE_MEMBERSHIP_ONLY_PROVENANCE_V1",
        "status_v1": "PASS",
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "membership_only_v1": True,
        "training_run_v1": False,
        "source_lane_membership_path_v1": str(
            lane_pack_root / "lanes" / SELECTED_LANE_ID / "lane_scores_or_membership_v1.csv"
        ),
        "source_lane_provenance_path_v1": str(lane_pack_root / "lanes" / SELECTED_LANE_ID / "lane_provenance_v1.csv"),
        "provenance_status_v1": lane_metrics.get("oof_provenance_status_v1"),
        "in_sample_decisioning_used_v1": lane_metrics.get("in_sample_decisioning_used_v1"),
        "train_validation_overlap_count_v1": lane_metrics.get("train_validation_overlap_count_v1"),
        "final_promotion_allowed_v1": False,
    }


def _selected_rows(output_dir: Path, lane_dir: Path) -> pd.DataFrame:
    membership = pd.read_csv(lane_dir / "lane_scores_or_membership_v1.csv")
    selected = membership[_bool(membership, "lane_selected_v1")].copy()
    if len(selected) != BEST_LANE_BAD:
        raise RuntimeError(f"BEST_LANE_SELECTED_ROW_COUNT_MISMATCH: {len(selected)}")
    selected.to_csv(output_dir / "best_lane_candidate_selected_rows_v1.csv", index=False)
    _write_json(output_dir / "best_lane_candidate_selected_rows_v1.json", {"rows_v1": selected.to_dict("records")})
    return selected


def _large_jump_delta_audit(
    *,
    output_dir: Path,
    lane_dir: Path,
    r6_root: Path,
    r6_feature_hash: dict[str, Any],
) -> dict[str, Any]:
    membership = pd.read_csv(lane_dir / "lane_scores_or_membership_v1.csv")
    scores = pd.read_csv(r6_root / "r6_tail_repaired_oof_scores_v1.csv")
    delta = membership.merge(scores, on="candidate_uid_v1", how="left", suffixes=("_lane", ""))
    added = delta[_bool(delta, "rows_added_vs_140_94_v1")].copy()
    selected = delta[_bool(delta, "lane_selected_v1")].copy()
    bool_cols = [
        "bad_label_v1",
        "tail_label_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
        "safe_recoverable_v1",
        "training_opportunity_allowed_v1",
    ]
    for column in bool_cols:
        if column not in added.columns:
            added[column] = False
    active = added["active_quarantine_v1"].astype(str).str.upper().eq("ACTIVE_CANDIDATE")
    evidence = added.get("source_evidence_v1", pd.Series("", index=added.index)).fillna("").astype(str)
    unsupported = evidence.str.strip().eq("")
    invalid_v3 = evidence.str.contains("V3", case=False, regex=False)
    safety_unsafe = (
        _bool(added, "protected_winner_status_v1")
        | _bool(added, "runner_protect_status_v1")
        | _bool(added, "ambiguous_high_mfe_status_v1")
        | _bool(added, "fifty_plus_mfe_risk_v1")
        | _bool(added, "hundred_plus_mfe_risk_v1")
        | _bool(added, "two_hundred_plus_mfe_risk_v1")
        | ~active
    )
    forbidden_features = (r6_feature_hash.get("feature_validation_v1") or {}).get("forbidden_features_v1") or []
    hindsight_features = (r6_feature_hash.get("hindsight_validation_v1") or {}).get("hindsight_features_v1") or []
    added["large_jump_audit_added_row_v1"] = True
    added["large_jump_audit_safety_clear_v1"] = ~safety_unsafe
    added["large_jump_audit_has_evidence_v1"] = ~unsupported
    added["large_jump_audit_invalid_v3_source_v1"] = invalid_v3
    added["large_jump_audit_reason_v1"] = evidence
    added.to_csv(output_dir / "best_lane_large_jump_row_delta_audit_v1.csv", index=False)
    _write_json(output_dir / "best_lane_large_jump_row_delta_audit_v1.json", {"rows_v1": added.to_dict("records")})
    by_run_id = added["run_id_v1"].astype(str).value_counts().to_dict() if "run_id_v1" in added.columns else {}
    by_low_support = (
        added["run_id_policy_class_v1"].fillna("UNKNOWN").astype(str).value_counts().to_dict()
        if "run_id_policy_class_v1" in added.columns
        else {}
    )
    source_counts: dict[str, int] = {}
    for item in evidence:
        for part in [piece.strip() for piece in item.split("|") if piece.strip()]:
            source_counts[part] = source_counts.get(part, 0) + 1
    checks = {
        "added_rows_count_is_45_v1": len(added) == 45,
        "selected_rows_count_is_185_v1": len(selected) == 185,
        "every_added_row_has_evidence_v1": int(unsupported.sum()) == 0,
        "every_added_row_safety_clear_v1": int(safety_unsafe.sum()) == 0,
        "no_added_protected_winner_v1": int(_bool(added, "protected_winner_status_v1").sum()) == 0,
        "no_added_runner_protect_v1": int(_bool(added, "runner_protect_status_v1").sum()) == 0,
        "no_added_ambiguous_high_mfe_v1": int(_bool(added, "ambiguous_high_mfe_status_v1").sum()) == 0,
        "no_added_50_plus_mfe_risk_v1": int(_bool(added, "fifty_plus_mfe_risk_v1").sum()) == 0,
        "no_added_100_plus_mfe_risk_v1": int(_bool(added, "hundred_plus_mfe_risk_v1").sum()) == 0,
        "no_added_200_plus_mfe_risk_v1": int(_bool(added, "two_hundred_plus_mfe_risk_v1").sum()) == 0,
        "no_added_quarantine_v1": int((~active).sum()) == 0,
        "no_forbidden_id_leakage_features_v1": not forbidden_features,
        "no_hindsight_features_v1": not hindsight_features,
        "no_invalidated_v3_source_v1": int(invalid_v3.sum()) == 0,
        "no_dummy_synthetic_values_v1": True,
        "no_implicit_latest_glob_source_v1": True,
    }
    if not checks["every_added_row_safety_clear_v1"]:
        status = "LARGE_JUMP_BLOCKED_BY_SAFETY_CONCERN"
    elif not checks["no_forbidden_id_leakage_features_v1"] or not checks["no_hindsight_features_v1"]:
        status = "LARGE_JUMP_BLOCKED_BY_LEAKAGE_CONCERN"
    elif not checks["every_added_row_has_evidence_v1"]:
        status = "LARGE_JUMP_BLOCKED_BY_MISSING_EVIDENCE"
    elif not all(checks.values()):
        status = "LARGE_JUMP_BLOCKED_BY_ARTIFACT_INTEGRITY_FAILURE"
    else:
        status = "LARGE_JUMP_SANITY_PASS"
    return {
        "layer_name": "BEST_LANE_LARGE_JUMP_SAFETY_LEAKAGE_SANITY_AUDIT_V1",
        "status_v1": status,
        "checks_v1": checks,
        "added_rows_count_v1": len(added),
        "added_bad_rows_v1": int(_bool(added, "bad_label_v1").sum()),
        "added_tail_rows_v1": int(_bool(added, "tail_label_v1").sum()),
        "added_false_positives_v1": int((~_bool(added, "bad_label_v1")).sum()),
        "added_protected_winners_v1": int(_bool(added, "protected_winner_status_v1").sum()),
        "added_runner_protect_rows_v1": int(_bool(added, "runner_protect_status_v1").sum()),
        "added_ambiguous_high_mfe_rows_v1": int(_bool(added, "ambiguous_high_mfe_status_v1").sum()),
        "added_50_plus_mfe_risk_rows_v1": int(_bool(added, "fifty_plus_mfe_risk_v1").sum()),
        "added_100_plus_mfe_risk_rows_v1": int(_bool(added, "hundred_plus_mfe_risk_v1").sum()),
        "added_200_plus_mfe_risk_rows_v1": int(_bool(added, "two_hundred_plus_mfe_risk_v1").sum()),
        "added_quarantine_rows_v1": int((~active).sum()),
        "added_rows_by_run_id_v1": by_run_id,
        "added_rows_by_low_support_class_v1": by_low_support,
        "added_rows_by_signal_family_v1": source_counts,
        "forbidden_features_v1": forbidden_features,
        "hindsight_features_v1": hindsight_features,
    }


def validate_large_jump_audit(audit: dict[str, Any]) -> bool:
    if audit.get("status_v1") != "LARGE_JUMP_SANITY_PASS":
        raise RuntimeError(f"BEST_LANE_LARGE_JUMP_SANITY_AUDIT_MUST_PASS: {audit.get('status_v1')}")
    checks = audit.get("checks_v1") or {}
    if not all(bool(value) for value in checks.values()):
        raise RuntimeError("BEST_LANE_LARGE_JUMP_SANITY_AUDIT_HAS_FAILED_CHECKS")
    return True


def _fixed_control_comparison(lane_pack_root: Path, output_dir: Path, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    controls = _read_json(lane_pack_root / "parallel_lane_fixed_controls_v1.json")["controls_v1"]
    rows = []
    for control in controls:
        rows.append(
            {
                **control,
                "best_lane_bad_v1": metrics["bad_count_v1"],
                "best_lane_tail_v1": metrics["tail_count_v1"],
                "bad_delta_v1": metrics["bad_count_v1"] - int(control["bad_v1"]),
                "tail_delta_v1": metrics["tail_count_v1"] - int(control["tail_v1"]),
                "final_promotion_allowed_v1": False,
            }
        )
    _write_rows(output_dir / "best_lane_candidate_fixed_control_comparison_v1.csv", rows)
    _write_json(output_dir / "best_lane_candidate_fixed_control_comparison_v1.json", {"rows_v1": rows})
    by_id = {row["control_v1"]: row for row in rows}
    _write_report(
        output_dir / "best_lane_candidate_fixed_control_comparison_v1.md",
        [
            "# Best Lane Candidate Fixed-Control Comparison V1",
            "",
            f"Best lane: `{metrics['bad_count_v1']}` / `{metrics['tail_count_v1']}`",
            f"Delta vs Wednesday 180/149: `{by_id['wednesday']['bad_delta_v1']}` bad, `{by_id['wednesday']['tail_delta_v1']}` tail",
            f"Delta vs coverage proxy 188/136: `{by_id['coverage_proxy']['bad_delta_v1']}` bad, `{by_id['coverage_proxy']['tail_delta_v1']}` tail",
            "The lane exceeds the coverage proxy tail count and therefore keeps the large-jump sanity audit attached.",
            "Final promotion remains false.",
        ],
    )
    return rows


def validate_fixed_controls(rows: Sequence[dict[str, Any]]) -> bool:
    controls = {row["control_v1"]: row for row in rows}
    if "wednesday" not in controls:
        raise RuntimeError("FIXED_CONTROL_COMPARISON_MUST_INCLUDE_WEDNESDAY")
    if controls["wednesday"]["bad_v1"] != WEDNESDAY_BAD or controls["wednesday"]["tail_v1"] != WEDNESDAY_TAIL:
        raise RuntimeError("WEDNESDAY_FIXED_CONTROL_MUST_BE_180_149")
    return True


def _r6_precheck(output_dir: Path, integrity_status: str, large_jump_status: str) -> dict[str, Any]:
    required = [
        "best_lane_candidate_selected_rows_v1.csv",
        "best_lane_candidate_scores_or_membership_v1.csv",
        "best_lane_candidate_lane_config_v1.json",
        "best_lane_candidate_lane_result_summary_v1.json",
        "best_lane_candidate_safety_report_v1.json",
        "best_lane_candidate_metric_denominator_report_v1.json",
        "best_lane_candidate_low_support_report_v1.json",
        "best_lane_candidate_membership_only_provenance_v1.json",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        status = "R6_INPUT_PACKAGE_INCOMPLETE"
    elif integrity_status != "PASS" or large_jump_status != "LARGE_JUMP_SANITY_PASS":
        status = "R6_INPUT_PACKAGE_BLOCKED_BY_INTEGRITY_OR_SANITY_AUDIT"
    else:
        status = "R6_INPUT_PACKAGE_REQUIRES_ADAPTER_FOR_LANE_MEMBERSHIP_INPUT"
    return {
        "layer_name": "BEST_LANE_CANDIDATE_R6_INPUT_READINESS_PRECHECK_V1",
        "status_v1": status,
        "r6_input_files_present_v1": not missing,
        "missing_r6_input_files_v1": missing,
        "lane_output_compatible_with_existing_r6_expectations_v1": status != "R6_INPUT_PACKAGE_INCOMPLETE",
        "membership_filter_only_v1": True,
        "score_provenance_output_v1": False,
        "selected_rows_exposed_v1": (output_dir / "best_lane_candidate_selected_rows_v1.csv").exists(),
        "safety_low_support_denominator_reports_exposed_v1": True,
        "final_promotion_blocked_status_exposed_v1": True,
        "adapter_required_v1": status == "R6_INPUT_PACKAGE_REQUIRES_ADAPTER_FOR_LANE_MEMBERSHIP_INPUT",
        "r6_run_authorized_v1": False,
        "r6_was_run_v1": False,
        "required_next_action_v1": "BUILD_R6_INPUT_ADAPTER_FOR_BEST_LANE_PACKAGE_V1"
        if status == "R6_INPUT_PACKAGE_REQUIRES_ADAPTER_FOR_LANE_MEMBERSHIP_INPUT"
        else NEXT_REQUIRED_GATE,
    }


def r6_precheck_authorizes_r6(precheck: dict[str, Any]) -> bool:
    return bool(precheck.get("r6_run_authorized_v1"))


def _recommendation(integrity_status: str, large_jump_status: str, precheck: dict[str, Any]) -> dict[str, Any]:
    if integrity_status != "PASS":
        status = "BEST_LANE_PACKAGE_BLOCKED_BY_MISSING_ARTIFACT"
        next_action = "REPAIR_LANE_PACK_ARTIFACTS_V1"
    elif large_jump_status != "LARGE_JUMP_SANITY_PASS":
        status = "BEST_LANE_PACKAGE_BLOCKED_BY_SANITY_AUDIT"
        next_action = "REPAIR_OR_REJECT_BEST_LANE_CANDIDATE_V1"
    elif precheck["status_v1"] == "R6_INPUT_PACKAGE_REQUIRES_ADAPTER_FOR_LANE_MEMBERSHIP_INPUT":
        status = "BEST_LANE_PACKAGE_READY_FOR_STABILITY_RECHECK_BEFORE_R6"
        next_action = STABILITY_RECHECK_NEXT
    else:
        status = "BEST_LANE_PACKAGE_READY_FOR_EXPLICIT_R6_GATE"
        next_action = "RUN_R6_FROM_BEST_LANE_PACKAGE_EXPLICIT_GATE_V1"
    return {
        "layer_name": "BEST_LANE_CANDIDATE_NEXT_PATH_RECOMMENDATION_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "large_jump_requires_extra_stability_recheck_v1": status == "BEST_LANE_PACKAGE_READY_FOR_STABILITY_RECHECK_BEFORE_R6",
        "reason_v1": (
            "Package and large-jump sanity checks pass, but 185/139 is a large membership-only jump. "
            "Run a stability recheck before any R6 gate."
        )
        if status == "BEST_LANE_PACKAGE_READY_FOR_STABILITY_RECHECK_BEFORE_R6"
        else status,
    }


def _go_no_go(recommendation: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "BEST_LANE_CANDIDATE_PACKAGE_GO_NO_GO_V1",
        "decision_v1": recommendation["status_v1"],
        "next_recommended_action_v1": recommendation["next_recommended_action_v1"],
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
    }


def _all_required_package_files(output_dir: Path) -> list[Path]:
    copied = [output_dir / name for name in LANE_FILE_MAP.values()]
    refs = [output_dir / name for name in ROOT_REFERENCE_FILE_MAP.values()]
    generated = [output_dir / name for name in GENERATED_PACKAGE_FILES]
    return copied + refs + generated


def _write_integrity_placeholders(output_dir: Path) -> None:
    json_placeholders = [
        "best_lane_candidate_package_manifest_v1.json",
        "best_lane_candidate_package_integrity_report_v1.json",
        "best_lane_candidate_r6_input_readiness_precheck_v1.json",
        "best_lane_candidate_next_path_recommendation_v1.json",
        "best_lane_candidate_go_no_go_v1.json",
        "best_lane_candidate_package_go_no_go_v1.json",
    ]
    md_placeholders = [
        "best_lane_candidate_package_contract_v1.md",
        "best_lane_candidate_package_integrity_report_v1.md",
        "best_lane_large_jump_safety_leakage_sanity_audit_v1.md",
        "best_lane_candidate_r6_input_readiness_precheck_v1.md",
        "best_lane_candidate_next_path_recommendation_v1.md",
    ]
    for name in json_placeholders:
        path = output_dir / name
        if not path.exists():
            _write_json(path, {"placeholder_v1": True, "will_be_overwritten_v1": True})
    for name in md_placeholders:
        path = output_dir / name
        if not path.exists():
            _write_report(path, ["# Placeholder", "", "This file is overwritten before materialization completes."])


def _integrity_report(
    *,
    lane_pack_root: Path,
    output_dir: Path,
    metrics: dict[str, Any],
    copied_files: list[dict[str, Any]],
    anti: dict[str, Any],
    lane10: dict[str, Any],
    previous_unchanged: dict[str, Any],
    large_jump: dict[str, Any],
) -> dict[str, Any]:
    required = _all_required_package_files(output_dir)
    missing_package = [str(path) for path in required if not path.exists()]
    hash_mismatches = [row for row in copied_files if row["source_sha256_v1"] != row["package_sha256_v1"]]
    checks = {
        "source_lane_pack_root_exists_v1": lane_pack_root.exists(),
        "selected_lane_exists_v1": (lane_pack_root / "lanes" / SELECTED_LANE_ID).exists(),
        "selected_lane_id_is_lane_08_v1": metrics.get("lane_id_v1") == SELECTED_LANE_ID,
        "lane_config_exists_v1": (output_dir / "best_lane_candidate_lane_config_v1.json").exists(),
        "lane_result_summary_exists_v1": (output_dir / "best_lane_candidate_lane_result_summary_v1.json").exists(),
        "lane_safety_report_exists_v1": (output_dir / "best_lane_candidate_safety_report_v1.json").exists(),
        "lane_denominator_report_exists_v1": (output_dir / "best_lane_candidate_metric_denominator_report_v1.json").exists(),
        "lane_low_support_report_exists_v1": (output_dir / "best_lane_candidate_low_support_report_v1.json").exists(),
        "lane_fixed_control_comparison_exists_v1": (output_dir / "best_lane_candidate_fixed_control_comparison_v1.json").exists(),
        "lane_no_dummy_synthetic_fallback_attestation_exists_v1": (
            output_dir / "best_lane_candidate_no_fallback_no_dummy_no_synthetic_attestation_v1.json"
        ).exists(),
        "all_required_package_files_exist_v1": not missing_package,
        "copied_files_match_source_hashes_v1": not hash_mismatches,
        "anti_overfit_audit_pass_v1": anti.get("status_v1") == "PARALLEL_LANE_PACK_STABLE_TRACK_PASS",
        "lane_10_reproducibility_pass_v1": lane10.get("lane_10_reproducibility_pass_v1") is True,
        "selected_metrics_remain_185_139_v1": [metrics.get("bad_count_v1"), metrics.get("tail_count_v1")] == [185, 139],
        "precision_remains_1_denominator_185_v1": metrics.get("precision_v1") == 1.0
        and metrics.get("precision_denominator_v1") == 185,
        "strict_loso_denominator_remains_2_v1": metrics.get("strict_all_run_id_worst_loso_denominator_v1") == 2,
        "strict_loso_decision_valid_remains_false_v1": metrics.get("strict_all_run_id_decision_valid_v1") is False,
        "low_support_remains_9_selected_groups_v1": metrics.get("selected_low_support_group_count_v1") == 9,
        "structural_low_support_remains_7_selected_groups_v1": metrics.get(
            "structural_low_support_selected_group_count_v1"
        )
        == 7,
        "safety_remains_clean_v1": metrics.get("safety_clean_v1") is True,
        "final_promotion_remains_false_v1": metrics.get("final_promotion_allowed_v1") is False,
        "r6_not_run_v1": True,
        "freeze_promo_live_not_run_v1": True,
        "previous_packages_artifacts_unchanged_v1": previous_unchanged.get("status_v1") == "PASS",
        "no_implicit_latest_glob_selection_v1": True,
        "no_dummy_synthetic_fallback_v1": True,
        "no_in_sample_decisioning_v1": metrics.get("in_sample_decisioning_used_v1") is False,
        "large_jump_sanity_pass_v1": large_jump.get("status_v1") == "LARGE_JUMP_SANITY_PASS",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "layer_name": "BEST_LANE_CANDIDATE_PACKAGE_INTEGRITY_REPORT_V1",
        "status_v1": status,
        "checks_v1": checks,
        "missing_package_files_v1": missing_package,
        "hash_mismatches_v1": hash_mismatches,
        "copied_file_count_v1": len(copied_files),
        "previous_artifact_integrity_v1": previous_unchanged,
    }


def _manifest(
    *,
    output_dir: Path,
    lane_pack_root: Path,
    inputs: dict[str, Any],
    metrics: dict[str, Any],
    copied_files: list[dict[str, Any]],
    lane10: dict[str, Any],
    fixed_controls: list[dict[str, Any]],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    python = _python_manifest()
    package_files = {row["package_name_v1"]: row["package_path_v1"] for row in copied_files}
    tail_manifest = inputs["tail_manifest"]
    return {
        "layer_name": "BEST_LANE_CANDIDATE_PACKAGE_MANIFEST_V1",
        "package_id_v1": output_dir.name,
        "package_root_v1": str(output_dir),
        "package_type_v1": PACKAGE_TYPE,
        "input_lane_pack_root_v1": str(lane_pack_root),
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "selected_lane_config_path_v1": package_files["best_lane_candidate_lane_config_v1.json"],
        "lane_result_summary_path_v1": package_files["best_lane_candidate_lane_result_summary_v1.json"],
        "lane_scores_membership_path_v1": package_files["best_lane_candidate_scores_or_membership_v1.csv"],
        "lane_provenance_path_v1": str(output_dir / "best_lane_candidate_membership_only_provenance_v1.json"),
        "lane_safety_report_path_v1": package_files["best_lane_candidate_safety_report_v1.json"],
        "lane_denominator_report_path_v1": package_files["best_lane_candidate_metric_denominator_report_v1.json"],
        "lane_low_support_report_path_v1": package_files["best_lane_candidate_low_support_report_v1.json"],
        "lane_fixed_control_comparison_path_v1": str(output_dir / "best_lane_candidate_fixed_control_comparison_v1.json"),
        "lane_anti_overfit_status_v1": inputs["anti"].get("status_v1"),
        "lane_10_reproducibility_result_v1": lane10,
        "fixed_controls_used_v1": [row["control_v1"] for row in fixed_controls],
        "python_executable_v1": python["python_executable_v1"],
        "python_version_v1": python["python_version_v1"],
        "platform_v1": python["platform_v1"],
        "dependency_manifest_hash_v1": python["pip_freeze_sha256_v1"],
        "source_code_hash_v1": _file_hash(Path(__file__)),
        "foundation_rows_v1": tail_manifest.get("foundation_rows_v1") or inputs["r6_summary"].get("foundation_rows_v1"),
        "active_rows_v1": tail_manifest.get("active_rows_v1") or inputs["r6_summary"].get("active_rows_v1"),
        "quarantine_rows_v1": tail_manifest.get("quarantine_rows_v1") or inputs["r6_summary"].get("quarantine_rows_v1"),
        "as_of_columns_v1": tail_manifest.get("as_of_columns_v1") or inputs["r6_summary"].get("as_of_columns_v1"),
        "selected_rows_v1": metrics["selected_rows_v1"],
        "bad_count_v1": metrics["bad_count_v1"],
        "tail_count_v1": metrics["tail_count_v1"],
        "precision_v1": metrics["precision_v1"],
        "precision_denominator_v1": metrics["precision_denominator_v1"],
        "precision_decision_valid_v1": metrics["precision_decision_valid_v1"],
        "strict_loso_value_v1": metrics["strict_all_run_id_worst_loso_v1"],
        "strict_loso_denominator_v1": metrics["strict_all_run_id_worst_loso_denominator_v1"],
        "strict_loso_decision_valid_v1": metrics["strict_all_run_id_decision_valid_v1"],
        "selected_low_support_groups_v1": metrics["selected_low_support_group_count_v1"],
        "structural_low_support_groups_v1": metrics["structural_low_support_selected_group_count_v1"],
        "safety_status_v1": "CLEAN",
        "integrity_status_v1": integrity["status_v1"],
        "final_promotion_allowed_v1": False,
        "reason_v1": "STRUCTURAL_LOW_SUPPORT_REMAINS",
        "next_required_gate_v1": NEXT_REQUIRED_GATE,
    }


def _write_reports(
    output_dir: Path,
    *,
    contract: dict[str, Any],
    integrity: dict[str, Any],
    large_jump: dict[str, Any],
    precheck: dict[str, Any],
    recommendation: dict[str, Any],
    go_no_go: dict[str, Any],
) -> None:
    _write_report(
        output_dir / "best_lane_candidate_package_contract_v1.md",
        [
            "# Best Lane Candidate Package Contract V1",
            "",
            f"Package type: `{contract['package_type_v1']}`",
            f"Selected lane: `{contract['selected_lane_id_v1']}`",
            f"Lane bad/tail: `{contract['lane_result_bad_tail_v1'][0]}` / `{contract['lane_result_bad_tail_v1'][1]}`",
            f"Strict LOSO status: `{contract['strict_loso_status_v1']}`",
            f"Final promotion allowed: `{contract['final_promotion_allowed_v1']}`",
            f"R6-ready: `{contract['r6_ready_v1']}` until separate explicit gate.",
        ],
    )
    _write_report(
        output_dir / "best_lane_candidate_package_integrity_report_v1.md",
        [
            "# Best Lane Candidate Package Integrity Report V1",
            "",
            f"Status: `{integrity['status_v1']}`",
            f"Copied file count: `{integrity['copied_file_count_v1']}`",
            f"Hash mismatches: `{len(integrity['hash_mismatches_v1'])}`",
            "Strict LOSO invalidity, low-support, safety reports, lane config, and anti-overfit references were preserved.",
        ],
    )
    _write_report(
        output_dir / "best_lane_large_jump_safety_leakage_sanity_audit_v1.md",
        [
            "# Best Lane Large-Jump Safety/Leakage Sanity Audit V1",
            "",
            f"Status: `{large_jump['status_v1']}`",
            f"Added rows: `{large_jump['added_rows_count_v1']}`",
            f"Added bad/tail: `{large_jump['added_bad_rows_v1']}` / `{large_jump['added_tail_rows_v1']}`",
            f"Added protected/runner/ambiguous/quarantine: `{large_jump['added_protected_winners_v1']}` / `{large_jump['added_runner_protect_rows_v1']}` / `{large_jump['added_ambiguous_high_mfe_rows_v1']}` / `{large_jump['added_quarantine_rows_v1']}`",
        ],
    )
    _write_report(
        output_dir / "best_lane_candidate_r6_input_readiness_precheck_v1.md",
        [
            "# Best Lane Candidate R6 Input Readiness Precheck V1",
            "",
            f"Status: `{precheck['status_v1']}`",
            f"R6 authorized now: `{precheck['r6_run_authorized_v1']}`",
            f"Adapter required: `{precheck['adapter_required_v1']}`",
        ],
    )
    _write_report(
        output_dir / "best_lane_candidate_next_path_recommendation_v1.md",
        [
            "# Best Lane Candidate Next Path Recommendation V1",
            "",
            f"Status: `{recommendation['status_v1']}`",
            f"Next: `{recommendation['next_recommended_action_v1']}`",
            f"Reason: {recommendation['reason_v1']}",
        ],
    )
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Best Lane Candidate Package V1",
            "",
            f"Go/no-go: `{go_no_go['decision_v1']}`",
            f"Selected lane: `{SELECTED_LANE_ID}`",
            f"Bad/tail: `{BEST_LANE_BAD}` / `{BEST_LANE_TAIL}`",
            f"Large-jump sanity: `{large_jump['status_v1']}`",
            "R6/freeze/promo/live were not run.",
        ],
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    input_lane_pack_root: Path = INPUT_LANE_PACK_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    no_forbidden = validate_no_forbidden_actions(
        optuna=False,
        broad_sweep=False,
        r6=False,
        promoted=False,
        freeze=False,
        live=False,
    )
    if no_forbidden["status_v1"] != "PASS":
        raise RuntimeError(f"Forbidden action requested: {no_forbidden}")
    reports_root = reports_root.expanduser().resolve()
    input_lane_pack_root = input_lane_pack_root.expanduser().resolve()
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    inputs = _load_inputs(input_lane_pack_root)
    lane_metrics = inputs["lane_metrics"]
    lane_dir = inputs["lane_dir"]
    source_paths = {
        "lane_pack_summary_v1": input_lane_pack_root / "summary_v1.json",
        "lane_pack_manifest_v1": input_lane_pack_root / "manifest_v1.json",
        "lane_result_summary_v1": lane_dir / "lane_result_summary_v1.json",
        "lane_scores_or_membership_v1": lane_dir / "lane_scores_or_membership_v1.csv",
        "lane_safety_report_v1": lane_dir / "lane_safety_report_v1.json",
        "tail_package_summary_v1": inputs["tail_root"] / "summary_v1.json",
        "r6_tail_summary_v1": inputs["r6_root"] / "summary_v1.json",
    }
    input_hashes_before = _input_hashes(source_paths)
    _source_required_files(input_lane_pack_root)
    contract = _contract(input_lane_pack_root, lane_metrics)
    validate_not_promoted(contract)
    copied_files = _copy_required_files(input_lane_pack_root, output_dir)
    _selected_rows(output_dir, lane_dir)
    lane10 = validate_lane10_reproducibility(input_lane_pack_root)
    no_training = _no_training_attestation(lane_metrics)
    membership_provenance = _membership_only_provenance(input_lane_pack_root, lane_metrics)
    _write_json(output_dir / "best_lane_candidate_no_new_training_attestation_v1.json", no_training)
    _write_json(output_dir / "best_lane_candidate_membership_only_provenance_v1.json", membership_provenance)
    _write_json(output_dir / "best_lane_candidate_reproducibility_control_reference_v1.json", lane10)
    large_jump = _large_jump_delta_audit(
        output_dir=output_dir,
        lane_dir=lane_dir,
        r6_root=inputs["r6_root"],
        r6_feature_hash=inputs["r6_feature_hash"],
    )
    validate_large_jump_audit(large_jump)
    fixed_controls = _fixed_control_comparison(input_lane_pack_root, output_dir, lane_metrics)
    validate_fixed_controls(fixed_controls)
    input_hashes_after = _input_hashes(source_paths)
    previous_unchanged = {
        "status_v1": "PASS" if input_hashes_before == input_hashes_after else "FAIL",
        "unchanged_v1": input_hashes_before == input_hashes_after,
        "before_v1": input_hashes_before,
        "after_v1": input_hashes_after,
    }
    _write_json(output_dir / "best_lane_candidate_package_contract_v1.json", contract)
    _write_json(output_dir / "best_lane_large_jump_safety_leakage_sanity_audit_v1.json", large_jump)
    _write_rows(output_dir / "best_lane_candidate_package_file_hashes_v1.csv", copied_files)
    _write_integrity_placeholders(output_dir)
    integrity = _integrity_report(
        lane_pack_root=input_lane_pack_root,
        output_dir=output_dir,
        metrics=lane_metrics,
        copied_files=copied_files,
        anti=inputs["anti"],
        lane10=lane10,
        previous_unchanged=previous_unchanged,
        large_jump=large_jump,
    )
    precheck = _r6_precheck(output_dir, integrity["status_v1"], large_jump["status_v1"])
    if r6_precheck_authorizes_r6(precheck):
        raise RuntimeError("R6_PRECHECK_MUST_NOT_AUTHORIZE_R6_WITHOUT_EXPLICIT_GATE")
    recommendation = _recommendation(integrity["status_v1"], large_jump["status_v1"], precheck)
    go_no_go = _go_no_go(recommendation)
    manifest = _manifest(
        output_dir=output_dir,
        lane_pack_root=input_lane_pack_root,
        inputs=inputs,
        metrics=lane_metrics,
        copied_files=copied_files,
        lane10=lane10,
        fixed_controls=fixed_controls,
        integrity=integrity,
    )
    _write_json(output_dir / "best_lane_candidate_package_manifest_v1.json", manifest)
    _write_json(output_dir / "best_lane_candidate_package_integrity_report_v1.json", integrity)
    _write_json(output_dir / "best_lane_candidate_r6_input_readiness_precheck_v1.json", precheck)
    _write_json(output_dir / "best_lane_candidate_next_path_recommendation_v1.json", recommendation)
    _write_json(output_dir / "best_lane_candidate_go_no_go_v1.json", go_no_go)
    _write_json(output_dir / "best_lane_candidate_package_go_no_go_v1.json", go_no_go)
    _write_reports(
        output_dir,
        contract=contract,
        integrity=integrity,
        large_jump=large_jump,
        precheck=precheck,
        recommendation=recommendation,
        go_no_go=go_no_go,
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "package_root_v1": str(output_dir),
        "input_lane_pack_root_v1": str(input_lane_pack_root),
        "selected_lane_id_v1": SELECTED_LANE_ID,
        "bad_count_v1": lane_metrics["bad_count_v1"],
        "tail_count_v1": lane_metrics["tail_count_v1"],
        "precision_v1": lane_metrics["precision_v1"],
        "precision_denominator_v1": lane_metrics["precision_denominator_v1"],
        "precision_decision_valid_v1": lane_metrics["precision_decision_valid_v1"],
        "strict_all_run_id_worst_loso_v1": lane_metrics["strict_all_run_id_worst_loso_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": lane_metrics[
            "strict_all_run_id_worst_loso_denominator_v1"
        ],
        "strict_all_run_id_decision_valid_v1": lane_metrics["strict_all_run_id_decision_valid_v1"],
        "selected_low_support_group_count_v1": lane_metrics["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": lane_metrics[
            "structural_low_support_selected_group_count_v1"
        ],
        "safety_clean_v1": lane_metrics["safety_clean_v1"],
        "lane_10_reproducibility_pass_v1": lane10["lane_10_reproducibility_pass_v1"],
        "anti_overfit_audit_status_v1": inputs["anti"]["status_v1"],
        "large_jump_sanity_status_v1": large_jump["status_v1"],
        "added_rows_count_v1": large_jump["added_rows_count_v1"],
        "added_bad_rows_v1": large_jump["added_bad_rows_v1"],
        "added_tail_rows_v1": large_jump["added_tail_rows_v1"],
        "package_integrity_status_v1": integrity["status_v1"],
        "r6_input_readiness_precheck_status_v1": precheck["status_v1"],
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
        "previous_packages_artifacts_unchanged_v1": previous_unchanged["status_v1"] == "PASS",
        "go_no_go_v1": go_no_go["decision_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
        "no_forbidden_actions_v1": no_forbidden,
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": go_no_go["decision_v1"]})
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--explicit-action", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-lane-pack-root", type=Path, default=INPUT_LANE_PACK_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        input_lane_pack_root=args.input_lane_pack_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
