#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import run_r5_2_objective_v2_parallel_rebuild_runner_v1 as historical_v2
from gx1.scripts import run_r5_2_objective_v2_replay_with_oof_provenance_v1 as v2_replay
from gx1.scripts import train_monday_r6_on_foundation_scores_v1 as existing_r6
from gx1.scripts.materialize_r6_retrain_from_true_r5_2_rescue_package_v1 import _refresh_r6_labels
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_BLINDSPOT_PROB,
    R6_HEAD_SPECS,
    R6_RISKY_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
    WEDNESDAY_LOCKED_THRESHOLDS,
    WEDNESDAY_R6_BENCHMARK,
    _jsonable as _r6_jsonable,
)
from gx1.scripts.train_r6_entry_runner_first_retrain_v1 import R6Candidate, _policy_mask as _existing_r6_policy_mask


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")
ACTION = "RUN_R6_RETRAIN_FROM_R5_2_CANDIDATE_PACKAGE_EXPLICIT_GATE_V1"
LAYER_NAME = ACTION
INPUT_PACKAGE_ROOT = (
    DEFAULT_REPORTS_ROOT / "BUILD_R5_2_PACKAGE_FROM_CANDIDATE_REQUIRES_EXPLICIT_GATE_V1_20260427T152500Z_LOCK"
)
DENOMINATOR_TARGET = 5
DEFAULT_FOLD_COUNT = 5

R6_SCOREFIELDS = [spec.output_col for spec in R6_HEAD_SPECS]
R6_HEAD_NAMES = [spec.head_id for spec in R6_HEAD_SPECS]

REQUIRED_PACKAGE_FILES = [
    "r5_2_candidate_package_manifest_v1.json",
    "r5_2_candidate_package_integrity_report_v1.json",
    "r5_2_candidate_package_r6_input_readiness_precheck_v1.json",
    "r5_2_candidate_oof_scores_v1.csv",
    "r5_2_candidate_oof_score_provenance_v1.csv",
    "r5_2_candidate_oof_fold_assignment_v1.csv",
    "r5_2_candidate_train_validation_membership_v1.csv",
    "r5_2_candidate_score_source_manifest_v1.json",
    "r5_2_candidate_no_fallback_no_dummy_no_synthetic_attestation_v1.json",
    "r5_2_candidate_low_support_report_v1.json",
    "r5_2_candidate_safety_report_v1.json",
    "r5_2_candidate_metric_denominator_report_v1.json",
    "r5_2_candidate_fixed_control_comparison_v1.json",
    "r5_2_candidate_threshold_selection_report_v1.json",
    "summary_v1.json",
]

FIXED_CONTROLS = [
    {
        "control_v1": "r5_2_package_pass_through",
        "bad_v1": 130,
        "tail_v1": 86,
        "role_v1": "R5_2_PACKAGE_PASS_THROUGH_CONTROL_STRICT_LOSO_INVALID_LOW_SUPPORT_VISIBLE",
    },
    {"control_v1": "historical_v2", "bad_v1": 95, "tail_v1": 61, "role_v1": "BLUEPRINT_COMPARATOR_ONLY"},
    {"control_v1": "v2_oof", "bad_v1": 69, "tail_v1": 53, "role_v1": "PROVENANCE_VALID_SIGNAL_CONTROL"},
    {"control_v1": "optuna", "bad_v1": 56, "tail_v1": 55, "role_v1": "WEAK_SEARCH_SPACE_CONTROL"},
    {"control_v1": "v3", "bad_v1": 17, "tail_v1": 13, "role_v1": "WEAK_OOF_CONTROL"},
    {"control_v1": "wednesday_benchmark", "bad_v1": 180, "tail_v1": 149, "role_v1": "COMPARATOR_ONLY_NOT_ROW_TARGET"},
]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value: Any) -> Any:
    return _r6_jsonable(value)


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
        raise RuntimeError(f"Missing required json artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing required artifact for hash: {path}")
    return _sha256_bytes(path.read_bytes())


def _hash_json(payload: Any) -> str:
    return _sha256_bytes(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _hash_list(values: Sequence[Any]) -> str:
    return _hash_json([str(value) for value in values])


def _hash_frame(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    work = frame[list(columns)].copy() if columns is not None else frame.copy()
    work = work.sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(work, index=False).to_numpy(dtype="uint64")
    return _sha256_bytes(hashed.tobytes())


def _row_hashes(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    work = frame[list(columns)].copy().sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(work, index=False).astype("uint64")
    return hashed.map(lambda value: hashlib.sha256(str(int(value)).encode("utf-8")).hexdigest())


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


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def validate_explicit_artifact_selection(selection_policy: str) -> bool:
    if selection_policy != "EXPLICIT_ONLY_NO_LATEST_GLOB":
        raise RuntimeError("IMPLICIT_LATEST_GLOB_ARTIFACT_SELECTION_FORBIDDEN")
    return True


def validate_no_forbidden_actions(*, optuna: bool, freeze: bool, promo: bool, live: bool) -> dict[str, Any]:
    failures = []
    if optuna:
        failures.append("OPTUNA_FORBIDDEN")
    if freeze:
        failures.append("FREEZE_FORBIDDEN")
    if promo:
        failures.append("PROMO_FORBIDDEN")
    if live:
        failures.append("LIVE_FORBIDDEN")
    return {"status_v1": "PASS" if not failures else "FAIL", "failures_v1": failures}


def validate_candidate_grid(grid: Sequence[dict[str, Any]]) -> bool:
    ids = {str(row.get("candidate_id_v1")) for row in grid}
    if "R5_2_PASS_THROUGH_CONTROL" not in ids:
        raise RuntimeError("R6_CANDIDATE_GRID_MUST_INCLUDE_R5_2_PASS_THROUGH_CONTROL")
    if len(grid) > 8:
        raise RuntimeError("R6_CANDIDATE_GRID_MUST_NOT_BE_LARGE_SWEEP_OR_OPTUNA")
    if any("OPTUNA" in str(row).upper() for row in grid):
        raise RuntimeError("R6_CANDIDATE_GRID_MUST_NOT_BE_OPTUNA")
    return True


def validate_final_promotion_blocked(payload: dict[str, Any]) -> bool:
    if bool(payload.get("final_promotion_allowed_v1")):
        raise RuntimeError("R6_CANDIDATE_CANNOT_AUTHORIZE_FINAL_PROMOTION")
    return True


def validate_no_train_validation_overlap(membership: pd.DataFrame) -> dict[str, Any]:
    overlap = membership[membership["is_train_v1"].astype(bool) & membership["is_validation_v1"].astype(bool)]
    return {
        "status_v1": "PASS" if overlap.empty else "FAIL",
        "overlap_count_v1": int(len(overlap)),
        "decision_valid_v1": bool(overlap.empty),
    }


def validate_no_in_sample_scoring(scores: pd.DataFrame) -> dict[str, Any]:
    if "was_row_in_train_for_scoring_model_v1" not in scores.columns:
        return {"status_v1": "FAIL", "in_sample_scored_count_v1": -1, "decision_valid_v1": False}
    count = int(scores["was_row_in_train_for_scoring_model_v1"].fillna(True).astype(bool).sum())
    return {"status_v1": "PASS" if count == 0 else "FAIL", "in_sample_scored_count_v1": count, "decision_valid_v1": count == 0}


def validate_r6_provenance_complete(scores: pd.DataFrame, provenance: pd.DataFrame) -> dict[str, Any]:
    expected = {
        (str(row["candidate_uid_v1"]), scorefield)
        for _, row in scores.iterrows()
        for scorefield in R6_SCOREFIELDS
    }
    observed = {
        (str(row["candidate_uid_v1"]), str(row["scorefield_v1"]))
        for _, row in provenance.iterrows()
    }
    missing = expected - observed
    invalid = int(provenance["provenance_valid_v1"].fillna(False).astype(bool).eq(False).sum()) if "provenance_valid_v1" in provenance.columns else len(provenance)
    return {
        "status_v1": "PASS" if not missing and invalid == 0 else "FAIL",
        "missing_provenance_rows_v1": int(len(missing)),
        "invalid_provenance_rows_v1": invalid,
        "decision_valid_v1": not missing and invalid == 0,
    }


def validate_input_package(package_root: Path) -> dict[str, Any]:
    package_root = package_root.expanduser().resolve()
    if not package_root.exists():
        raise RuntimeError("R6_INPUT_PACKAGE_VALIDATION_FAILED: package root missing")
    missing = [name for name in REQUIRED_PACKAGE_FILES if not (package_root / name).exists()]
    if missing:
        raise RuntimeError(f"R6_INPUT_PACKAGE_VALIDATION_FAILED: missing package files {missing}")
    manifest = _read_json(package_root / "r5_2_candidate_package_manifest_v1.json")
    integrity = _read_json(package_root / "r5_2_candidate_package_integrity_report_v1.json")
    precheck = _read_json(package_root / "r5_2_candidate_package_r6_input_readiness_precheck_v1.json")
    summary = _read_json(package_root / "summary_v1.json")
    checks = {
        "package_root_exists_v1": True,
        "package_manifest_exists_v1": True,
        "package_integrity_pass_v1": integrity.get("status_v1") == "PASS",
        "selected_threshold_recall_v1": summary.get("selected_threshold_candidate_v1") == "RECALL",
        "oof_bad_tail_130_86_v1": summary.get("bad_count_v1") == 130 and summary.get("tail_count_v1") == 86,
        "precision_denominator_130_v1": summary.get("precision_denominator_v1") == 130,
        "strict_loso_denominator_2_v1": summary.get("strict_all_run_id_worst_loso_denominator_v1") == 2,
        "strict_loso_decision_valid_false_v1": summary.get("strict_all_run_id_decision_valid_v1") is False,
        "safety_clean_v1": summary.get("safety_clean_v1") is True,
        "low_support_report_present_v1": (package_root / "r5_2_candidate_low_support_report_v1.json").exists(),
        "oof_scores_present_v1": (package_root / "r5_2_candidate_oof_scores_v1.csv").exists(),
        "oof_provenance_present_v1": (package_root / "r5_2_candidate_oof_score_provenance_v1.csv").exists(),
        "fold_assignment_present_v1": (package_root / "r5_2_candidate_oof_fold_assignment_v1.csv").exists(),
        "train_validation_membership_present_v1": (package_root / "r5_2_candidate_train_validation_membership_v1.csv").exists(),
        "score_source_manifest_present_v1": (package_root / "r5_2_candidate_score_source_manifest_v1.json").exists(),
        "no_dummy_synthetic_fallback_attestation_present_v1": (package_root / "r5_2_candidate_no_fallback_no_dummy_no_synthetic_attestation_v1.json").exists(),
        "final_promotion_false_v1": manifest.get("final_promotion_allowed_v1") is False,
        "r6_readiness_precheck_not_authorized_status_v1": precheck.get("status_v1") == "R6_INPUT_PACKAGE_READY_BUT_R6_NOT_AUTHORIZED",
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"R6_INPUT_PACKAGE_VALIDATION_FAILED: {failed}")
    return {
        "layer_name": "R6_INPUT_PACKAGE_VALIDATION_V1",
        "package_root_v1": str(package_root),
        "status_v1": "PASS",
        "checks_v1": checks,
        "manifest_v1": manifest,
        "summary_v1": summary,
        "integrity_status_v1": integrity.get("status_v1"),
        "r6_readiness_precheck_status_v1": precheck.get("status_v1"),
    }


def _package_hashes(package_root: Path) -> dict[str, str]:
    return {name: _file_hash(package_root / name) for name in REQUIRED_PACKAGE_FILES}


def _source_mapping(output_dir: Path) -> dict[str, Any]:
    return {
        "layer_name": "R6_EXISTING_SOURCE_MAPPING_V1",
        "r6_existing_path_found_v1": True,
        "existing_r6_path_reused_v1": True,
        "thin_wrapper_needed_v1": True,
        "existing_implementation_can_consume_package_after_staging_v1": True,
        "r6_source_files_v1": {
            "five_head_training_utility_v1": str(Path(existing_r6._train_r6_heads.__code__.co_filename).resolve()),
            "r6_on_foundation_scores_orchestrator_v1": str(Path(existing_r6.__file__).resolve()),
            "r6_entry_runner_first_retrain_v1": str(Path(sys.modules[R6Candidate.__module__].__file__).resolve()),
            "current_thin_wrapper_v1": str(Path(__file__).resolve()),
        },
        "r6_five_head_implementation_source_v1": {
            "heads_v1": R6_HEAD_NAMES,
            "scorefields_v1": R6_SCOREFIELDS,
        },
        "r6_candidate_eval_utilities_v1": [
            "_train_r6_heads",
            "_foundation_r6_feature_names",
            "_policy_mask",
            "R6Candidate",
        ],
        "r6_policy_eval_surface_source_v1": "train_r6_entry_runner_first_retrain_v1._policy_mask",
        "r6_feature_input_source_v1": "v2_replay._prepare_inputs plus explicit R5.2 candidate package scores staged into existing R6 score-frame schema",
        "r6_label_truth_source_v1": "v2_replay._prepare_inputs legal label table plus existing _refresh_r6_labels",
        "r6_safety_eval_source_v1": "package safety columns and explicit candidate metric checks",
        "r6_threshold_candidate_grid_source_v1": "small deterministic wrapper grid using existing R6Candidate policy mask",
        "output_root_v1": str(output_dir),
    }


def _no_reimplementation_attestation(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "R6_NO_UNNECESSARY_REIMPLEMENTATION_ATTESTATION_V1",
        "existing_r6_training_utilities_used_v1": True,
        "existing_r6_five_head_logic_reused_v1": True,
        "existing_r6_policy_mask_used_v1": True,
        "existing_feature_builder_used_v1": True,
        "existing_label_source_tables_used_v1": True,
        "new_r6_head_logic_introduced_v1": False,
        "new_feature_surface_introduced_v1": False,
        "disconnected_r6_clone_created_v1": False,
        "wrapper_only_v1": True,
        "reason_v1": "The new code validates and stages the explicit R5.2 package, then calls existing R6 five-head training and policy utilities in grouped OOF mode.",
        "source_mapping_v1": mapping,
    }


def _candidate_grid() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id_v1": "R5_2_PASS_THROUGH_CONTROL",
            "candidate_type_v1": "PASS_THROUGH",
            "description_v1": "No R6 filtering beyond the packaged R5.2 selected rows.",
            "use_existing_r6_policy_mask_v1": False,
        },
        {
            "candidate_id_v1": "R6_WEDNESDAY_THRESHOLD_DIAGNOSTIC",
            "candidate_type_v1": "R6_POLICY_MASK",
            "family_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "bad_threshold_v1": 0.95,
            "runner_threshold_v1": 0.60,
            "tail_threshold_v1": 0.90,
            "risky_threshold_v1": 0.85,
            "blindspot_threshold_v1": 0.70,
            "r5_2_runner_threshold_v1": 0.74,
            "use_r5_2_base_v1": True,
            "hard_asof_runner_guard_v1": True,
            "representability_status_v1": "REPRESENTABLE_WITH_EXISTING_R6_POLICY_MASK",
        },
        {
            "candidate_id_v1": "R6_SAFETY_FIRST",
            "candidate_type_v1": "R6_POLICY_MASK",
            "family_v1": "R6_CONSERVATIVE_HIGH_PRECISION",
            "bad_threshold_v1": 0.85,
            "runner_threshold_v1": 0.82,
            "tail_threshold_v1": 0.85,
            "risky_threshold_v1": 0.80,
            "blindspot_threshold_v1": 0.85,
            "r5_2_runner_threshold_v1": 0.74,
            "use_r5_2_base_v1": True,
            "hard_asof_runner_guard_v1": True,
        },
        {
            "candidate_id_v1": "R6_BALANCED",
            "candidate_type_v1": "R6_POLICY_MASK",
            "family_v1": "R6_BATCH04_AWARE_ROBUST",
            "bad_threshold_v1": 0.65,
            "runner_threshold_v1": 0.70,
            "tail_threshold_v1": 0.75,
            "risky_threshold_v1": 0.65,
            "blindspot_threshold_v1": 0.70,
            "r5_2_runner_threshold_v1": 0.74,
            "use_r5_2_base_v1": True,
            "hard_asof_runner_guard_v1": True,
        },
        {
            "candidate_id_v1": "R6_TAIL_FOCUSED",
            "candidate_type_v1": "R6_POLICY_MASK",
            "family_v1": "R6_THREE_HEAD_BLOCK_RUNNER_TAIL",
            "bad_threshold_v1": 0.75,
            "runner_threshold_v1": 0.82,
            "tail_threshold_v1": 0.50,
            "risky_threshold_v1": 0.80,
            "blindspot_threshold_v1": 0.85,
            "r5_2_runner_threshold_v1": 0.74,
            "use_r5_2_base_v1": False,
            "hard_asof_runner_guard_v1": False,
        },
        {
            "candidate_id_v1": "R6_RECALL_FOCUSED_WITH_HARD_VETO",
            "candidate_type_v1": "R6_POLICY_MASK",
            "family_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "bad_threshold_v1": 0.55,
            "runner_threshold_v1": 0.60,
            "tail_threshold_v1": 0.55,
            "risky_threshold_v1": 0.55,
            "blindspot_threshold_v1": 0.70,
            "r5_2_runner_threshold_v1": 0.74,
            "use_r5_2_base_v1": True,
            "hard_asof_runner_guard_v1": True,
        },
    ]


def _to_r6_candidate(row: dict[str, Any]) -> R6Candidate:
    return R6Candidate(
        policy_name=str(row["candidate_id_v1"]),
        family=str(row["family_v1"]),
        bad_threshold=float(row["bad_threshold_v1"]),
        runner_threshold=float(row["runner_threshold_v1"]),
        tail_threshold=float(row["tail_threshold_v1"]),
        risky_threshold=float(row["risky_threshold_v1"]),
        blindspot_threshold=float(row["blindspot_threshold_v1"]),
        r5_2_runner_threshold=float(row["r5_2_runner_threshold_v1"]),
        use_r5_2_base=bool(row["use_r5_2_base_v1"]),
        hard_asof_runner_guard=bool(row["hard_asof_runner_guard_v1"]),
    )


def _active_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.get("calendar_quarantine_status_v1", pd.Series("ACTIVE_CANDIDATE", index=frame.index)).astype("string").eq("ACTIVE_CANDIDATE")


def _load_and_stage_package_frame(
    package_root: Path,
    *,
    spec_dir: Path,
    foundation_score_dir: Path | None,
    label_table: Path | None,
) -> dict[str, Any]:
    inputs = v2_replay._prepare_inputs(spec_dir, foundation_score_dir, label_table)
    frame = inputs["training_frame"].copy()
    frame["candidate_uid"] = frame["candidate_uid"].astype(str)
    package_scores = pd.read_csv(package_root / "r5_2_candidate_oof_scores_v1.csv")
    package_scores["candidate_uid"] = package_scores["candidate_uid_v1"].astype(str)
    package_cols = [
        "candidate_uid",
        "r5_2_coverage_bad_score_v1",
        "r5_2_coverage_tail_score_v1",
        "r5_2_coverage_hard_veto_score_v1",
        "r5_2_best_candidate_selected_v1",
        "fold_id_v1",
        "run_id_policy_class_v1",
        "structural_low_support_v1",
        "zero_denominator_group_v1",
        "training_opportunity_allowed_v1",
        "final_promotion_evidence_allowed_v1",
        "active_quarantine_v1",
        "bad_label_v1",
        "tail_label_v1",
        "safe_recoverable_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
        "source_evidence_v1",
    ]
    missing = [column for column in package_cols if column not in package_scores.columns]
    if missing:
        raise RuntimeError(f"R5.2 package scores missing required columns for R6 staging: {missing}")
    staged = frame.merge(package_scores[package_cols], on="candidate_uid", how="left", validate="one_to_one")
    if staged["r5_2_coverage_bad_score_v1"].isna().any():
        raise RuntimeError("R5.2 package scores do not align one-to-one with foundation frame")
    staged[R5_2_BAD_PROB] = _num(staged, "r5_2_coverage_bad_score_v1")
    staged[R5_2_RUNNER_PROB] = _num(staged, "r5_2_coverage_hard_veto_score_v1")
    staged["blocker_score_v1"] = _num(staged, "r5_2_coverage_bad_score_v1")
    staged["runner_protector_score_v1"] = _num(staged, "r5_2_coverage_hard_veto_score_v1")
    staged["r5_2_selected_candidate__block_v1"] = _bool(staged, "r5_2_best_candidate_selected_v1")
    staged["r5_2_base_membership_contract_id_v1"] = "R5_2_CANDIDATE_PACKAGE_RECALL_130_86_EXPLICIT_R6_INPUT"
    staged = _refresh_r6_labels(staged)
    feature_names = existing_r6._foundation_r6_feature_names(staged)
    forbidden = validate_no_forbidden_features(feature_names)
    hindsight = validate_no_hindsight_features(feature_names)
    if forbidden["status_v1"] != "PASS":
        raise RuntimeError(f"R6 feature surface contains forbidden/id leakage features: {forbidden}")
    if hindsight["status_v1"] != "PASS":
        raise RuntimeError(f"R6 feature surface contains hindsight features: {hindsight}")
    return {
        "inputs": inputs,
        "frame": staged,
        "package_scores": package_scores,
        "feature_names": feature_names,
        "feature_validation": forbidden,
        "hindsight_validation": hindsight,
    }


def validate_no_forbidden_features(feature_names: Sequence[str]) -> dict[str, Any]:
    id_leakage_names = {"candidate_uid", "trade_uid", "trade_id", "run_id", "decision_timestamp"}
    existing_r6_presence_allowlist = {
        "entry_observation_present_v1",
        "entry_raw_state_present_v1",
        "management_observation_present_v1",
        "entry_coverage_original_entry_observation_present_v1",
        "entry_coverage_original_entry_raw_state_present_v1",
        "entry_coverage_repair_applied_v1",
        "entry_coverage_repair_source_v1",
    }
    forbidden = []
    for feature in feature_names:
        lower = str(feature).lower()
        if str(feature) in existing_r6_presence_allowlist:
            continue
        matches = []
        if lower in id_leakage_names or lower.endswith("_uid") or lower.endswith("_id"):
            matches.append("id_leakage_key")
        for pattern in historical_v2.FORBIDDEN_FEATURE_PATTERNS:
            if pattern in lower:
                matches.append(pattern)
        if matches:
            forbidden.append({"feature_v1": str(feature), "matches_v1": sorted(set(matches))})
    return {"status_v1": "PASS" if not forbidden else "FAIL", "forbidden_features_v1": forbidden}


def validate_no_hindsight_features(feature_names: Sequence[str]) -> dict[str, Any]:
    patterns = ["hindsight", "future_", "post_decision"]
    forbidden = [str(feature) for feature in feature_names if any(pattern in str(feature).lower() for pattern in patterns)]
    return {"status_v1": "PASS" if not forbidden else "FAIL", "hindsight_features_v1": forbidden}


def _fold_assignment(package_root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(package_root / "r5_2_candidate_oof_fold_assignment_v1.csv")
    raw["candidate_uid"] = raw["candidate_uid_v1"].astype(str)
    lookup = raw[["candidate_uid", "fold_id_v1", "group_key_v1", "split_policy_v1"]]
    out = frame[["candidate_uid", "trade_uid", "decision_timestamp", "trade_id", "run_id"]].copy()
    out = out.merge(lookup, on="candidate_uid", how="left", validate="one_to_one")
    if out["fold_id_v1"].isna().any():
        raise RuntimeError("R5.2 package fold assignment does not align one-to-one with R6 staging frame")
    out = out.rename(
        columns={
            "candidate_uid": "candidate_uid_v1",
            "trade_uid": "trade_uid_v1",
            "decision_timestamp": "decision_timestamp_v1",
            "trade_id": "trade_id_v1",
            "run_id": "run_id_v1",
        }
    )
    out["split_policy_v1"] = "REUSED_R5_2_PACKAGE_GROUPED_OOF_BY_RUN_ID_FOR_R6"
    return out


def _run_r6_grouped_oof(
    output_dir: Path,
    *,
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    fold_assignment: pd.DataFrame,
    config_payload: dict[str, Any],
    n_estimators: int,
    early_stopping_rounds: int,
    learning_rate: float,
    max_depth: int,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    source_hash = _file_hash(Path(__file__).resolve())
    r6_training_source_hash = _file_hash(Path(existing_r6._train_r6_heads.__code__.co_filename).resolve())
    policy_source_hash = _file_hash(Path(sys.modules[R6Candidate.__module__].__file__).resolve())
    config_hash = _hash_json(config_payload)
    feature_matrix_hash = _hash_frame(frame, feature_names)
    label_columns = ["candidate_uid", *[spec.label_col for spec in R6_HEAD_SPECS]]
    label_table_hash = _hash_frame(frame, label_columns)
    feature_row_hashes = _row_hashes(frame, feature_names)
    label_row_hashes = _row_hashes(frame, label_columns)
    fold_lookup = fold_assignment.set_index("candidate_uid_v1")["fold_id_v1"].astype(str)
    fold_ids = frame["candidate_uid"].astype(str).map(fold_lookup)
    if fold_ids.isna().any():
        raise RuntimeError("R6 fold assignment missing for staged rows")

    score_cols = [
        "candidate_uid",
        "trade_uid",
        "decision_timestamp",
        "trade_id",
        "run_id",
        "label_should_not_take_v1",
        "tail_10_50_mfe_v1",
        "take_was_ok_v1",
        "fifty_plus_mfe_v1",
        "hundred_plus_mfe_v1",
        "two_hundred_plus_mfe_v1",
        "r5_2_selected_candidate__block_v1",
        "r5_2_coverage_bad_score_v1",
        "r5_2_coverage_tail_score_v1",
        "r5_2_coverage_hard_veto_score_v1",
        R5_2_BAD_PROB,
        R5_2_RUNNER_PROB,
        "blocker_score_v1",
        "runner_protector_score_v1",
        "run_id_policy_class_v1",
        "structural_low_support_v1",
        "zero_denominator_group_v1",
        "training_opportunity_allowed_v1",
        "final_promotion_evidence_allowed_v1",
        "active_quarantine_v1",
        "bad_label_v1",
        "tail_label_v1",
        "safe_recoverable_v1",
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
        "hundred_plus_mfe_risk_v1",
        "two_hundred_plus_mfe_risk_v1",
        "source_evidence_v1",
    ]
    scores = frame[[column for column in score_cols if column in frame.columns]].copy()
    scores = scores.rename(
        columns={
            "candidate_uid": "candidate_uid_v1",
            "trade_uid": "trade_uid_v1",
            "decision_timestamp": "decision_timestamp_v1",
            "trade_id": "trade_id_v1",
            "run_id": "run_id_v1",
            "r5_2_selected_candidate__block_v1": "r5_2_package_selected_v1",
        }
    )
    scores["fold_id_v1"] = fold_ids.values
    for scorefield in R6_SCOREFIELDS:
        scores[scorefield] = np.nan

    membership_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    head_metric_frames: list[pd.DataFrame] = []
    fold_model_rows: list[dict[str, Any]] = []

    for fold_idx, fold_id in enumerate(sorted(fold_ids.unique())):
        validation_mask = fold_ids.eq(str(fold_id))
        train_mask = ~validation_mask
        train_uids = frame.loc[train_mask, "candidate_uid"].astype(str).tolist()
        validation_uids = frame.loc[validation_mask, "candidate_uid"].astype(str).tolist()
        train_hash = _hash_list(train_uids)
        validation_hash = _hash_list(validation_uids)
        for idx, row in frame.iterrows():
            is_train = bool(train_mask.loc[idx])
            is_validation = bool(validation_mask.loc[idx])
            membership_rows.append(
                {
                    "candidate_uid_v1": row["candidate_uid"],
                    "trade_uid_v1": row["trade_uid"],
                    "decision_timestamp_v1": row["decision_timestamp"],
                    "trade_id_v1": row["trade_id"],
                    "run_id_v1": row["run_id"],
                    "fold_id_v1": str(fold_id),
                    "is_train_v1": is_train,
                    "is_validation_v1": is_validation,
                    "train_membership_hash_v1": train_hash,
                    "validation_membership_hash_v1": validation_hash,
                    "train_validation_overlap_v1": bool(is_train and is_validation),
                }
            )
        pred, metrics = existing_r6._train_r6_heads(
            frame=frame.drop(columns=R6_SCOREFIELDS, errors="ignore"),
            feature_names=feature_names,
            train_mask=train_mask,
            validation_mask=validation_mask,
            output_dir=output_dir,
            model_tag=f"r6_from_r5_2_candidate_package_oof_{fold_id}",
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds,
            learning_rate=learning_rate,
            max_depth=max_depth,
            seed=seed + fold_idx * 101,
            n_jobs=n_jobs,
        )
        metrics = metrics.copy()
        metrics["fold_id_v1"] = str(fold_id)
        metrics["train_membership_hash_v1"] = train_hash
        metrics["validation_membership_hash_v1"] = validation_hash
        head_metric_frames.append(metrics)
        pred = pred.set_index("candidate_uid")
        for spec in R6_HEAD_SPECS:
            output_col = spec.output_col
            model_source_id = f"{ACTION}:{fold_id}:{spec.head_id}:seed={seed + fold_idx * 101}"
            validation_index = frame.index[validation_mask]
            scores.loc[validation_index, output_col] = frame.loc[validation_index, "candidate_uid"].map(pred[output_col]).to_numpy()
            fold_model_rows.append(
                {
                    "fold_id_v1": str(fold_id),
                    "head_name_v1": spec.head_id,
                    "label_col_v1": spec.label_col,
                    "scorefield_v1": output_col,
                    "model_source_identifier_v1": model_source_id,
                    "train_rows_v1": int(train_mask.sum()),
                    "validation_rows_v1": int(validation_mask.sum()),
                    "source_hash_v1": source_hash,
                    "r6_training_source_hash_v1": r6_training_source_hash,
                    "policy_source_hash_v1": policy_source_hash,
                    "config_hash_v1": config_hash,
                    "decisioning_scope_v1": "GROUPED_OOF_VALIDATION_ONLY",
                }
            )
            for idx in validation_index:
                row = frame.loc[idx]
                provenance_rows.append(
                    {
                        "candidate_uid_v1": row["candidate_uid"],
                        "trade_uid_v1": row["trade_uid"],
                        "decision_timestamp_v1": row["decision_timestamp"],
                        "trade_id_v1": row["trade_id"],
                        "run_id_v1": row["run_id"],
                        "fold_id_v1": str(fold_id),
                        "scorefield_v1": output_col,
                        "head_v1": spec.head_id,
                        "variant_v1": "R6_FROM_R5_2_CANDIDATE_PACKAGE_EXPLICIT_GATE",
                        "model_source_identifier_v1": model_source_id,
                        "train_membership_hash_v1": train_hash,
                        "validation_membership_hash_v1": validation_hash,
                        "was_row_in_train_for_scoring_model_v1": False,
                        "feature_matrix_hash_v1": feature_matrix_hash,
                        "feature_row_hash_v1": feature_row_hashes.loc[idx],
                        "label_table_hash_v1": label_table_hash,
                        "label_row_hash_v1": label_row_hashes.loc[idx],
                        "config_hash_v1": config_hash,
                        "source_hash_v1": source_hash,
                        "r6_training_source_hash_v1": r6_training_source_hash,
                        "policy_source_hash_v1": policy_source_hash,
                        "seed_v1": seed + fold_idx * 101,
                        "score_value_v1": scores.loc[idx, output_col],
                        "decision_valid_v1": True,
                        "provenance_valid_v1": True,
                        "oof_status_v1": "OOF_VALIDATION_SCORE",
                    }
                )
    if scores[R6_SCOREFIELDS].isna().any().any():
        missing = scores[R6_SCOREFIELDS].isna().sum().to_dict()
        raise RuntimeError(f"R6 OOF prediction matrix has missing scores: {missing}")
    scores["was_row_in_train_for_scoring_model_v1"] = False
    scores["decision_valid_score_v1"] = True
    return {
        "scores": scores,
        "membership": pd.DataFrame(membership_rows),
        "provenance": pd.DataFrame(provenance_rows),
        "head_metrics": pd.concat(head_metric_frames, ignore_index=True),
        "fold_models": fold_model_rows,
        "hashes": {
            "feature_matrix_hash_v1": feature_matrix_hash,
            "label_table_hash_v1": label_table_hash,
            "config_hash_v1": config_hash,
            "source_hash_v1": source_hash,
            "r6_training_source_hash_v1": r6_training_source_hash,
            "policy_source_hash_v1": policy_source_hash,
        },
    }


def _metric_ratio(name: str, numerator: int, denominator: int, min_denominator: int = DENOMINATOR_TARGET) -> dict[str, Any]:
    if denominator <= 0:
        return {
            f"{name}_v1": np.nan,
            f"{name}_numerator_v1": numerator,
            f"{name}_denominator_v1": denominator,
            f"{name}_denominator_status_v1": "EMPTY_DENOMINATOR",
            f"{name}_decision_valid_v1": False,
        }
    status = "OK" if denominator >= min_denominator else "TOO_SMALL_DENOMINATOR"
    return {
        f"{name}_v1": numerator / denominator,
        f"{name}_numerator_v1": numerator,
        f"{name}_denominator_v1": denominator,
        f"{name}_denominator_status_v1": status,
        f"{name}_decision_valid_v1": status == "OK",
    }


def _safety_counts(scores: pd.DataFrame, selected: pd.Series) -> dict[str, int]:
    return {
        "fifty_plus_mfe_overlap_v1": int((selected & _bool(scores, "fifty_plus_mfe_risk_v1")).sum()),
        "hundred_plus_mfe_overlap_v1": int((selected & _bool(scores, "hundred_plus_mfe_risk_v1")).sum()),
        "two_hundred_plus_mfe_overlap_v1": int((selected & _bool(scores, "two_hundred_plus_mfe_risk_v1")).sum()),
        "strongest_winner_overlap_v1": int((selected & _bool(scores, "protected_winner_status_v1")).sum()),
        "protected_winner_selected_v1": int((selected & _bool(scores, "protected_winner_status_v1")).sum()),
        "runner_protect_leakage_v1": int((selected & _bool(scores, "runner_protect_status_v1")).sum()),
        "ambiguous_high_mfe_leakage_v1": int((selected & _bool(scores, "ambiguous_high_mfe_status_v1")).sum()),
        "quarantine_selected_v1": int((selected & scores["active_quarantine_v1"].astype(str).str.upper().ne("ACTIVE_CANDIDATE")).sum()),
    }


def candidate_passes_hard_safety(row: dict[str, Any]) -> bool:
    return all(
        int(row.get(key, 0) or 0) == 0
        for key in [
            "fifty_plus_mfe_overlap_v1",
            "hundred_plus_mfe_overlap_v1",
            "two_hundred_plus_mfe_overlap_v1",
            "strongest_winner_overlap_v1",
            "protected_winner_selected_v1",
            "runner_protect_leakage_v1",
            "ambiguous_high_mfe_leakage_v1",
            "quarantine_selected_v1",
        ]
    )


def _loso_rows(scores: pd.DataFrame, selected: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bad = _bool(scores, "bad_label_v1")
    rows: list[dict[str, Any]] = []
    for run_id, part in pd.DataFrame({"selected": selected.astype(bool), "bad": bad, "run_id": scores["run_id_v1"].astype(str)}).groupby("run_id"):
        denominator = int(part["selected"].sum())
        numerator = int((part["selected"] & part["bad"]).sum())
        value = numerator / denominator if denominator else np.nan
        rows.append(
            {
                "run_id_v1": str(run_id),
                "selected_denominator_v1": denominator,
                "selected_bad_numerator_v1": numerator,
                "group_precision_v1": value,
                "denominator_status_v1": "OK"
                if denominator >= DENOMINATOR_TARGET
                else ("EMPTY_SELECTED_GROUP" if denominator == 0 else "TOO_SMALL_DENOMINATOR"),
            }
        )
    non_empty = [row for row in rows if int(row["selected_denominator_v1"]) > 0]
    worst = min(non_empty, key=lambda row: float(row["group_precision_v1"])) if non_empty else {
        "run_id_v1": "EMPTY_SELECTED_GROUP_SET",
        "selected_denominator_v1": 0,
        "selected_bad_numerator_v1": 0,
        "group_precision_v1": np.nan,
    }
    evaluable = [row for row in rows if int(row["selected_denominator_v1"]) >= DENOMINATOR_TARGET]
    evaluable_worst = min(evaluable, key=lambda row: float(row["group_precision_v1"])) if evaluable else None
    for row in rows:
        row["is_worst_loso_group_v1"] = row["run_id_v1"] == worst["run_id_v1"]
    return rows, {
        "strict_all_run_id_worst_loso_v1": worst["group_precision_v1"],
        "strict_all_run_id_worst_loso_group_v1": worst["run_id_v1"],
        "strict_all_run_id_worst_loso_numerator_v1": int(worst["selected_bad_numerator_v1"]),
        "strict_all_run_id_worst_loso_denominator_v1": int(worst["selected_denominator_v1"]),
        "strict_all_run_id_worst_loso_denominator_status_v1": "OK"
        if int(worst["selected_denominator_v1"]) >= DENOMINATOR_TARGET
        else "TOO_SMALL_DENOMINATOR",
        "strict_all_run_id_decision_valid_v1": int(worst["selected_denominator_v1"]) >= DENOMINATOR_TARGET,
        "selected_low_support_group_count_v1": int(sum(0 < int(row["selected_denominator_v1"]) < DENOMINATOR_TARGET for row in rows)),
        "zero_selected_group_count_v1": int(sum(int(row["selected_denominator_v1"]) == 0 for row in rows)),
        "evaluable_group_count_v1": int(len(evaluable)),
        "evaluable_groups_loso_v1": None if evaluable_worst is None else evaluable_worst["group_precision_v1"],
        "evaluable_groups_denominator_min_v1": 0 if not evaluable else min(int(row["selected_denominator_v1"]) for row in evaluable),
        "evaluable_groups_decision_valid_v1": bool(evaluable),
    }


def _evaluate_candidate(scores: pd.DataFrame, selected: pd.Series, candidate_id: str, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = selected.reindex(scores.index).fillna(False).astype(bool)
    bad = _bool(scores, "bad_label_v1")
    tail = _bool(scores, "tail_label_v1")
    r5_2_base = _bool(scores, "r5_2_package_selected_v1")
    precision = _metric_ratio("precision", int((selected & bad).sum()), int(selected.sum()))
    loso_detail, loso = _loso_rows(scores, selected)
    safety = _safety_counts(scores, selected)
    structural_groups = int((selected & _bool(scores, "structural_low_support_v1")).groupby(scores["run_id_v1"].astype(str)).any().sum())
    payload = {
        "candidate_id_v1": candidate_id,
        "threshold_config_v1": json.dumps(_jsonable(config), sort_keys=True),
        "selected_rows_v1": int(selected.sum()),
        "bad_count_v1": int((selected & bad).sum()),
        "tail_count_v1": int((selected & tail).sum()),
        **precision,
        **loso,
        "selected_low_support_group_count_v1": loso["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": structural_groups,
        "final_promotion_allowed_v1": False,
        "r5_2_package_rows_retained_v1": int((selected & r5_2_base).sum()),
        "r5_2_package_rows_blocked_v1": int((~selected & r5_2_base).sum()),
        "r5_2_package_rows_added_v1": int((selected & ~r5_2_base).sum()),
        "outside_base_rows_used_v1": int((selected & ~r5_2_base).sum()) > 0,
        "outside_base_rows_safety_proven_v1": False,
        "no_in_sample_scored_rows_v1": True,
        "train_validation_overlap_count_v1": 0,
        **safety,
    }
    payload["safety_clean_v1"] = candidate_passes_hard_safety(payload)
    payload["outside_base_rows_safety_proven_v1"] = bool(payload["safety_clean_v1"])
    fail_reasons = []
    if not payload["safety_clean_v1"]:
        fail_reasons.append("TRUE_SAFETY_VIOLATION")
    if not payload["precision_decision_valid_v1"]:
        fail_reasons.append("PRECISION_DENOMINATOR_INVALID")
    payload["candidate_constraint_pass_v1"] = not fail_reasons
    payload["fail_reason_v1"] = "|".join(fail_reasons)
    return payload, [{"candidate_id_v1": candidate_id, **row} for row in loso_detail]


def _candidate_masks(scores: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, pd.Series]]:
    grid = _candidate_grid()
    validate_candidate_grid(grid)
    frame = scores.rename(
        columns={
            "candidate_uid_v1": "candidate_uid",
            "run_id_v1": "run_id",
            "r5_2_package_selected_v1": "r5_2_selected_candidate__block_v1",
        }
    ).copy()
    masks: dict[str, pd.Series] = {}
    active = scores["active_quarantine_v1"].astype(str).str.upper().eq("ACTIVE_CANDIDATE")
    for row in grid:
        candidate_id = str(row["candidate_id_v1"])
        if row["candidate_type_v1"] == "PASS_THROUGH":
            masks[candidate_id] = _bool(scores, "r5_2_package_selected_v1") & active
            continue
        candidate = _to_r6_candidate(row)
        masks[candidate_id] = _existing_r6_policy_mask(frame, candidate).reindex(scores.index).fillna(False).astype(bool) & active
    return grid, masks


def _select_best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe = [row for row in rows if bool(row.get("candidate_constraint_pass_v1")) and bool(row.get("precision_decision_valid_v1"))]
    if not safe:
        return rows[0]
    package_bad = 130
    package_tail = 86

    def key(row: dict[str, Any]) -> tuple[int, int, int, float, int]:
        bad = int(row.get("bad_count_v1") or 0)
        tail = int(row.get("tail_count_v1") or 0)
        dominates_package = int(bad >= package_bad and tail >= package_tail and (bad > package_bad or tail > package_tail))
        preserves_package = int(bad >= package_bad and tail >= package_tail)
        return (
            dominates_package,
            preserves_package,
            tail,
            float(row.get("precision_v1") or 0.0),
            bad,
        )

    return sorted(safe, key=key, reverse=True)[0]


def _best_status(best: dict[str, Any], *, provenance_ok: bool, no_in_sample_ok: bool, no_overlap_ok: bool) -> tuple[str, str]:
    if not provenance_ok or not no_in_sample_ok or not no_overlap_ok:
        return "R6_CANDIDATE_FAILS_PROVENANCE_OR_IN_SAMPLE_GUARD", "REPAIR_R6_OOF_PROVENANCE_OR_SPLIT_V1"
    if not bool(best.get("safety_clean_v1")):
        return "R6_CANDIDATE_FAILS_TRUE_SAFETY", "REPAIR_R6_SAFETY_HEADS_OR_ADD_HARD_VETO_LAYER_V1"
    bad = int(best.get("bad_count_v1") or 0)
    tail = int(best.get("tail_count_v1") or 0)
    if bad >= 130 and tail >= 86 and (bad > 130 or tail > 86):
        return "R6_CANDIDATE_STRONGER_THAN_R5_2_PACKAGE_BUT_FINAL_PROMOTION_BLOCKED", "MATERIALIZE_R6_CANDIDATE_PACKAGE_REQUIRES_EXPLICIT_GATE_V1"
    if bad >= 130 and tail >= 86:
        return "R6_CANDIDATE_RETURNS_R5_2_LEVEL_WITH_STRONGER_HEAD_DIAGNOSTICS", "R6_HEAD_SIGNAL_AUDIT_OR_R5_2_BASE_EXPANSION_V1"
    return "R6_CANDIDATE_SAFE_BUT_WEAKER_THAN_R5_2_PACKAGE", "R6_HEAD_SIGNAL_AUDIT_OR_KEEP_R5_2_PACKAGE_AS_CONTROL_V1"


def _fixed_control_comparison(best: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for control in FIXED_CONTROLS:
        rows.append(
            {
                **control,
                "candidate_bad_v1": best["bad_count_v1"],
                "candidate_tail_v1": best["tail_count_v1"],
                "bad_delta_v1": int(best["bad_count_v1"]) - int(control["bad_v1"]),
                "tail_delta_v1": int(best["tail_count_v1"]) - int(control["tail_v1"]),
            }
        )
    return rows


def _contract(package_root: Path, package_validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "contract": "R6_FROM_R5_2_CANDIDATE_PACKAGE_CONTRACT_V1",
        "input_r5_2_package_root_v1": str(package_root),
        "input_package_validation_status_v1": package_validation["status_v1"],
        "selection_policy_v1": "EXPLICIT_ONLY_NO_LATEST_GLOB",
        "foundation_rows_required_v1": 1914,
        "active_rows_required_v1": 1852,
        "quarantine_rows_required_v1": 62,
        "as_of_count_required_v1": 109,
        "r5_2_package_score_provenance_preserved_v1": True,
        "r6_grouped_oof_execution_required_v1": True,
        "validation_only_scoring_required_v1": True,
        "train_validation_membership_required_v1": True,
        "r6_score_provenance_required_v1": True,
        "r6_score_source_manifest_required_v1": True,
        "no_in_sample_decisioning_required_v1": True,
        "no_dummy_synthetic_fallback_required_v1": True,
        "five_head_r6_required_v1": R6_HEAD_NAMES,
        "hard_runner_guard_if_available_v1": "hard_asof_runner_guard",
        "strict_all_run_id_loso_reporting_required_v1": True,
        "low_support_registry_reporting_required_v1": True,
        "safety_reporting_required_v1": True,
        "fixed_controls_required_v1": [row["control_v1"] for row in FIXED_CONTROLS],
        "freeze_promo_live_forbidden_v1": True,
    }


def _write_markdown_artifacts(output_dir: Path, payload: dict[str, Any]) -> None:
    _write_report(
        output_dir / "r6_input_package_validation_v1.md",
        [
            "# R6 Input Package Validation V1",
            "",
            f"Status: `{payload['package_validation']['status_v1']}`",
            f"Package root: `{payload['package_root']}`",
            "The package is accepted only because this materializer is the separate explicit R6 gate.",
        ],
    )
    _write_report(
        output_dir / "r6_existing_source_mapping_v1.md",
        [
            "# R6 Existing Source Mapping V1",
            "",
            f"Existing R6 path reused: `{payload['source_mapping']['existing_r6_path_reused_v1']}`",
            f"Thin wrapper needed: `{payload['source_mapping']['thin_wrapper_needed_v1']}`",
            f"Heads: `{', '.join(R6_HEAD_NAMES)}`",
        ],
    )
    _write_report(
        output_dir / "r6_from_r5_2_candidate_package_contract_v1.md",
        [
            "# R6 From R5.2 Candidate Package Contract V1",
            "",
            "This is candidate R6 eval only. It is not freeze, promotion, live, or canonical Monday R6.",
            "Strict LOSO and structural low-support reporting remain mandatory.",
        ],
    )
    _write_report(
        output_dir / "r6_fixed_controls_v1.md",
        [
            "# R6 Fixed Controls V1",
            "",
            *[f"- `{row['control_v1']}`: `{row['bad_v1']}` / `{row['tail_v1']}` ({row['role_v1']})" for row in FIXED_CONTROLS],
            "",
            "Wednesday is comparator only, not a row-for-row target.",
        ],
    )
    _write_report(
        output_dir / "r6_candidate_grid_contract_v1.md",
        [
            "# R6 Candidate Grid Contract V1",
            "",
            "Small deterministic candidate grid only; no Optuna or large sweep.",
            "The pass-through control is mandatory.",
        ],
    )
    _write_report(
        output_dir / "r6_best_candidate_v1.md",
        [
            "# R6 Best Candidate V1",
            "",
            f"Status: `{payload['status']}`",
            f"Best candidate: `{payload['best']['candidate_id_v1']}`",
            f"Bad/tail: `{payload['best']['bad_count_v1']}` / `{payload['best']['tail_count_v1']}`",
            f"Final promotion allowed: `{payload['best']['final_promotion_allowed_v1']}`",
            f"Next: `{payload['next_action']}`",
        ],
    )


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    package_root: Path = INPUT_PACKAGE_ROOT,
    spec_dir: Path = historical_v2.DEFAULT_SPEC_DIR,
    foundation_score_dir: Path | None = None,
    label_table: Path | None = None,
    explicit_action: str = ACTION,
    n_estimators: int = 160,
    early_stopping_rounds: int = 30,
    learning_rate: float = 0.04,
    max_depth: int = 3,
    seed: int = 20260427,
    n_jobs: int = 2,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB")
    no_forbidden = validate_no_forbidden_actions(optuna=False, freeze=False, promo=False, live=False)
    if no_forbidden["status_v1"] != "PASS":
        raise RuntimeError(f"Forbidden action requested: {no_forbidden}")
    reports_root = reports_root.expanduser().resolve()
    package_root = package_root.expanduser().resolve()
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    input_hashes_before = _package_hashes(package_root)
    package_validation = validate_input_package(package_root)
    source_mapping = _source_mapping(output_dir)
    attestation = _no_reimplementation_attestation(source_mapping)
    contract = _contract(package_root, package_validation)
    staged = _load_and_stage_package_frame(
        package_root,
        spec_dir=spec_dir,
        foundation_score_dir=foundation_score_dir,
        label_table=label_table,
    )
    frame = staged["frame"]
    package_scores = staged["package_scores"]
    feature_names = staged["feature_names"]
    fold_assignment = _fold_assignment(package_root, frame)
    foundation = staged["inputs"]["foundation"]
    config_payload = {
        "action_v1": ACTION,
        "input_package_root_v1": str(package_root),
        "candidate_grid_v1": _candidate_grid(),
        "n_estimators_v1": n_estimators,
        "early_stopping_rounds_v1": early_stopping_rounds,
        "learning_rate_v1": learning_rate,
        "max_depth_v1": max_depth,
        "seed_v1": seed,
        "denominator_target_v1": DENOMINATOR_TARGET,
    }
    oof = _run_r6_grouped_oof(
        output_dir,
        frame=frame,
        feature_names=feature_names,
        fold_assignment=fold_assignment,
        config_payload=config_payload,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        learning_rate=learning_rate,
        max_depth=max_depth,
        seed=seed,
        n_jobs=n_jobs,
    )
    scores = oof["scores"]
    membership = oof["membership"]
    provenance = oof["provenance"]
    grid, masks = _candidate_masks(scores)
    candidate_rows: list[dict[str, Any]] = []
    loso_detail: list[dict[str, Any]] = []
    for row in grid:
        candidate_id = str(row["candidate_id_v1"])
        metrics, group_rows = _evaluate_candidate(scores, masks[candidate_id], candidate_id, row)
        candidate_rows.append({**row, **metrics})
        loso_detail.extend(group_rows)
    best = _select_best_candidate(candidate_rows)
    scores["r6_best_candidate_selected_v1"] = masks[str(best["candidate_id_v1"])].values
    no_in_sample = validate_no_in_sample_scoring(scores)
    no_overlap = validate_no_train_validation_overlap(membership)
    provenance_check = validate_r6_provenance_complete(scores, provenance)
    status, next_action = _best_status(
        best,
        provenance_ok=provenance_check["decision_valid_v1"],
        no_in_sample_ok=no_in_sample["decision_valid_v1"],
        no_overlap_ok=no_overlap["decision_valid_v1"],
    )
    fixed_comparison = _fixed_control_comparison(best)
    input_hashes_after = _package_hashes(package_root)
    package_unchanged = input_hashes_before == input_hashes_after

    denominator_rows = [
        {
            "candidate_id_v1": best["candidate_id_v1"],
            "metric_v1": "precision",
            "value_v1": best["precision_v1"],
            "numerator_v1": best["precision_numerator_v1"],
            "denominator_v1": best["precision_denominator_v1"],
            "denominator_status_v1": best["precision_denominator_status_v1"],
            "decision_valid_v1": best["precision_decision_valid_v1"],
        },
        {
            "candidate_id_v1": best["candidate_id_v1"],
            "metric_v1": "strict_all_run_id_worst_loso",
            "value_v1": best["strict_all_run_id_worst_loso_v1"],
            "numerator_v1": best["strict_all_run_id_worst_loso_numerator_v1"],
            "denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
            "denominator_status_v1": best["strict_all_run_id_worst_loso_denominator_status_v1"],
            "decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        },
    ]
    safety_rows = [
        {"candidate_id_v1": best["candidate_id_v1"], "safety_metric_v1": key, "value_v1": best[key], "pass_v1": int(best[key]) == 0}
        for key in [
            "fifty_plus_mfe_overlap_v1",
            "hundred_plus_mfe_overlap_v1",
            "two_hundred_plus_mfe_overlap_v1",
            "strongest_winner_overlap_v1",
            "protected_winner_selected_v1",
            "runner_protect_leakage_v1",
            "ambiguous_high_mfe_leakage_v1",
            "quarantine_selected_v1",
        ]
    ]
    low_support_report = {
        "candidate_id_v1": best["candidate_id_v1"],
        "strict_all_run_id_decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
        "selected_low_support_group_count_v1": best["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": best["structural_low_support_selected_group_count_v1"],
        "zero_selected_group_count_v1": best["zero_selected_group_count_v1"],
        "evaluable_group_count_v1": best["evaluable_group_count_v1"],
        "evaluable_groups_loso_v1": best["evaluable_groups_loso_v1"],
        "evaluable_groups_denominator_min_v1": best["evaluable_groups_denominator_min_v1"],
        "final_promotion_allowed_v1": False,
    }
    head_contribution_rows = [
        {
            "head_name_v1": spec.head_id,
            "scorefield_v1": spec.output_col,
            "feature_inputs_used_v1": len(feature_names),
            "provenance_status_v1": "PASS" if provenance_check["status_v1"] == "PASS" else "FAIL",
            "no_train_validation_overlap_proof_v1": no_overlap["status_v1"],
        }
        for spec in R6_HEAD_SPECS
    ]
    best_candidate = {
        "layer_name": "R6_BEST_CANDIDATE_V1",
        "status_v1": status,
        "next_recommended_action_v1": next_action,
        "best_candidate_v1": best,
        "selection_priority_v1": [
            "safety clean",
            "OOF provenance PASS",
            "no in-sample decisioning",
            "no train/validation overlap",
            "precision denominator valid",
            "strict LOSO reported",
            "low-support reported",
            "preserve or beat R5.2 package",
        ],
        "fixed_control_comparison_v1": fixed_comparison,
        "final_promotion_allowed_v1": False,
    }
    go_no_go = {
        "layer_name": "R6_FROM_R5_2_CANDIDATE_PACKAGE_GO_NO_GO_V1",
        "decision_v1": status,
        "next_recommended_action_v1": next_action,
        "r6_candidate_eval_only_v1": True,
        "freeze_promo_live_run_v1": False,
        "final_promotion_allowed_v1": False,
        "strict_loso_decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        "structural_low_support_visible_v1": best["structural_low_support_selected_group_count_v1"] > 0,
    }
    validate_final_promotion_blocked(go_no_go)

    score_source_manifest = {
        "layer_name": "R6_SCORE_SOURCE_MANIFEST_V1",
        "input_r5_2_package_root_v1": str(package_root),
        "r6_heads_v1": R6_HEAD_NAMES,
        "scorefields_v1": R6_SCOREFIELDS,
        "fold_models_v1": oof["fold_models"],
        "model_family_v1": "XGBClassifier via existing R6 utility",
        "feature_count_v1": len(feature_names),
        "no_new_feature_surface_v1": True,
        "existing_r6_path_reused_v1": True,
    }
    feature_label_hash_manifest = {
        "layer_name": "R6_FEATURE_LABEL_HASH_MANIFEST_V1",
        "hashes_v1": oof["hashes"],
        "feature_count_v1": len(feature_names),
        "feature_validation_v1": staged["feature_validation"],
        "hindsight_validation_v1": staged["hindsight_validation"],
    }
    no_in_sample_attestation = {
        "layer_name": "R6_NO_IN_SAMPLE_DECISIONING_ATTESTATION_V1",
        **no_in_sample,
        "train_validation_overlap_v1": no_overlap,
    }
    no_fallback_attestation = {
        "layer_name": "R6_NO_FALLBACK_NO_DUMMY_NO_SYNTHETIC_ATTESTATION_V1",
        "status_v1": "PASS",
        "dummy_input_used_v1": False,
        "synthetic_input_used_v1": False,
        "degraded_fallback_used_v1": False,
        "decision_valid_v1": True,
        "no_forbidden_actions_v1": no_forbidden,
    }

    package_validation_path = output_dir / "r6_input_package_validation_v1.json"
    _write_json(package_validation_path, package_validation)
    _write_json(output_dir / "r6_existing_source_mapping_v1.json", source_mapping)
    _write_json(output_dir / "r6_no_unnecessary_reimplementation_attestation_v1.json", attestation)
    _write_json(output_dir / "r6_from_r5_2_candidate_package_contract_v1.json", contract)
    _write_json(output_dir / "r6_fixed_controls_v1.json", {"controls_v1": FIXED_CONTROLS})
    _write_json(output_dir / "r6_candidate_grid_contract_v1.json", {"contract_v1": "SMALL_DETERMINISTIC_GRID_NO_OPTUNA", "candidate_count_v1": len(grid), "pass_through_required_v1": True})
    _write_rows(output_dir / "r6_candidate_grid_v1.csv", grid)
    _write_json(output_dir / "r6_candidate_grid_v1.json", {"candidates_v1": grid})
    scores.to_csv(output_dir / "r6_oof_scores_v1.csv", index=False)
    provenance.to_csv(output_dir / "r6_oof_score_provenance_v1.csv", index=False)
    fold_assignment.to_csv(output_dir / "r6_oof_fold_assignment_v1.csv", index=False)
    membership.to_csv(output_dir / "r6_train_validation_membership_v1.csv", index=False)
    oof["head_metrics"].to_csv(output_dir / "r6_oof_head_training_metrics_v1.csv", index=False)
    _write_json(output_dir / "r6_score_source_manifest_v1.json", score_source_manifest)
    _write_json(output_dir / "r6_feature_label_hash_manifest_v1.json", feature_label_hash_manifest)
    _write_json(output_dir / "r6_no_in_sample_decisioning_attestation_v1.json", no_in_sample_attestation)
    _write_json(output_dir / "r6_no_fallback_no_dummy_no_synthetic_attestation_v1.json", no_fallback_attestation)
    _write_json(
        output_dir / "r6_candidate_eval_summary_v1.json",
        {
            "layer_name": "R6_CANDIDATE_EVAL_SUMMARY_V1",
            "best_candidate_id_v1": best["candidate_id_v1"],
            "oof_provenance_status_v1": provenance_check["status_v1"],
            "train_validation_overlap_status_v1": no_overlap["status_v1"],
            "in_sample_scored_status_v1": no_in_sample["status_v1"],
            **best,
        },
    )
    _write_rows(output_dir / "r6_candidate_eval_metrics_v1.csv", candidate_rows)
    _write_rows(output_dir / "r6_candidate_metric_denominator_report_v1.csv", denominator_rows)
    _write_json(output_dir / "r6_candidate_metric_denominator_report_v1.json", {"rows_v1": denominator_rows})
    _write_rows(output_dir / "r6_candidate_safety_report_v1.csv", safety_rows)
    _write_json(output_dir / "r6_candidate_safety_report_v1.json", {"rows_v1": safety_rows, "safety_clean_v1": best["safety_clean_v1"]})
    _write_rows(output_dir / "r6_candidate_low_support_report_v1.csv", [low_support_report])
    _write_json(output_dir / "r6_candidate_low_support_report_v1.json", low_support_report)
    _write_rows(output_dir / "r6_candidate_fixed_control_comparison_v1.csv", fixed_comparison)
    _write_json(output_dir / "r6_candidate_fixed_control_comparison_v1.json", {"controls_v1": fixed_comparison})
    _write_rows(output_dir / "r6_candidate_head_contribution_report_v1.csv", head_contribution_rows)
    _write_json(output_dir / "r6_candidate_head_contribution_report_v1.json", {"heads_v1": head_contribution_rows})
    _write_rows(output_dir / "r6_candidate_loso_group_detail_v1.csv", loso_detail)
    _write_json(output_dir / "r6_best_candidate_v1.json", best_candidate)
    _write_json(output_dir / "r6_from_r5_2_candidate_package_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "input_r5_2_package_root_v1": str(package_root),
            "input_hashes_before_v1": input_hashes_before,
            "input_hashes_after_v1": input_hashes_after,
            "input_r5_2_package_unchanged_v1": package_unchanged,
            "feature_source_v1": str(staged["inputs"]["score_dir"]),
            "label_source_v1": str(staged["inputs"]["label_path"]),
        },
    )
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "input_r5_2_package_root_v1": str(package_root),
        "input_r5_2_package_unchanged_v1": package_unchanged,
        "r6_source_mapping_status_v1": "EXISTING_R6_PATH_REUSED_WITH_THIN_WRAPPER",
        "wrapper_only_v1": True,
        "foundation_rows_v1": int(foundation["foundation_rows_v1"]),
        "active_rows_v1": int(foundation["active_rows_v1"]),
        "quarantine_rows_v1": int(foundation["quarantine_rows_v1"]),
        "as_of_columns_v1": int(foundation["asof_columns_v1"]),
        "r6_five_head_status_v1": "PASS",
        "r6_head_count_v1": len(R6_HEAD_SPECS),
        "candidate_grid_count_v1": len(grid),
        "best_candidate_v1": best["candidate_id_v1"],
        "bad_count_v1": best["bad_count_v1"],
        "tail_count_v1": best["tail_count_v1"],
        "precision_v1": best["precision_v1"],
        "precision_denominator_v1": best["precision_denominator_v1"],
        "precision_decision_valid_v1": best["precision_decision_valid_v1"],
        "strict_all_run_id_worst_loso_v1": best["strict_all_run_id_worst_loso_v1"],
        "strict_all_run_id_worst_loso_denominator_v1": best["strict_all_run_id_worst_loso_denominator_v1"],
        "strict_all_run_id_decision_valid_v1": best["strict_all_run_id_decision_valid_v1"],
        "selected_low_support_group_count_v1": best["selected_low_support_group_count_v1"],
        "structural_low_support_selected_group_count_v1": best["structural_low_support_selected_group_count_v1"],
        "safety_clean_v1": best["safety_clean_v1"],
        "r5_2_package_rows_retained_v1": best["r5_2_package_rows_retained_v1"],
        "r5_2_package_rows_blocked_v1": best["r5_2_package_rows_blocked_v1"],
        "r5_2_package_rows_added_v1": best["r5_2_package_rows_added_v1"],
        "oof_provenance_status_v1": provenance_check["status_v1"],
        "r6_provenance_rows_v1": int(len(provenance)),
        "train_validation_membership_rows_v1": int(len(membership)),
        "train_validation_overlap_count_v1": no_overlap["overlap_count_v1"],
        "in_sample_scored_count_v1": no_in_sample["in_sample_scored_count_v1"],
        "final_promotion_allowed_v1": False,
        "r6_candidate_eval_only_v1": True,
        "freeze_promo_live_run_v1": False,
        "go_no_go_v1": status,
        "next_recommended_action_v1": next_action,
        "first_full_pytest_note_required_v1": "FIRST_FULL_PYTEST_INTERRUPTED_RERUN_PASS",
    }
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "decision_v1": status})
    _write_report(
        output_dir / "report_v1.md",
        [
            "# Run R6 Retrain From R5.2 Candidate Package Explicit Gate V1",
            "",
            f"Go/no-go: `{status}`",
            f"Best candidate: `{best['candidate_id_v1']}`",
            f"Bad/tail: `{best['bad_count_v1']}` / `{best['tail_count_v1']}`",
            f"Precision: `{best['precision_v1']}` denominator `{best['precision_denominator_v1']}`",
            f"Strict LOSO: `{best['strict_all_run_id_worst_loso_v1']}` denominator `{best['strict_all_run_id_worst_loso_denominator_v1']}`",
            f"Safety clean: `{best['safety_clean_v1']}`",
            f"Final promotion allowed: `{best['final_promotion_allowed_v1']}`",
            "This is R6 candidate eval only; no freeze, promotion, live, or canonical Monday R6 action was run.",
        ],
    )
    _write_markdown_artifacts(
        output_dir,
        {
            "package_validation": package_validation,
            "package_root": str(package_root),
            "source_mapping": source_mapping,
            "status": status,
            "best": best,
            "next_action": next_action,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--explicit-action", required=True)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--r5-2-package-root", type=Path, default=INPUT_PACKAGE_ROOT)
    parser.add_argument("--spec-dir", type=Path, default=historical_v2.DEFAULT_SPEC_DIR)
    parser.add_argument("--foundation-score-dir", type=Path, default=None)
    parser.add_argument("--label-table", type=Path, default=None)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        package_root=args.r5_2_package_root,
        spec_dir=args.spec_dir,
        foundation_score_dir=args.foundation_score_dir,
        label_table=args.label_table,
        explicit_action=args.explicit_action,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
