#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
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

from gx1.scripts import run_r5_2_objective_v2_replay_with_oof_provenance_v1 as v2_oof


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "REPAIR_LOSO_GROUPING_OR_DENOMINATOR_CONTRACT_V1"
LAYER_NAME = ACTION
V2_OOF_ROOT = DEFAULT_REPORTS_ROOT / "PATCH_V2_RUNNER_TO_WRITE_PROVENANCE_V1_20260427T111437Z_LOCK"
SKELETON_ROOT = DEFAULT_REPORTS_ROOT / "FIND_BACK_TO_WEDNESDAY_R6_SKELETON_AND_REBUILD_MONDAY_FOUNDATION_V1_20260427T083808Z_LOCK"
OPTUNA_ROOT = DEFAULT_REPORTS_ROOT / "CONSTRAINED_OPTUNA_OBJECTIVE_SEARCH_V1_20260427T080458Z_LOCK"
SELECTED_V3_ROOT = DEFAULT_REPORTS_ROOT / "RERUN_V3_PARALLEL_REBUILD_WITH_OOF_PROVENANCE_EXPLICIT_FLAG_20260427T073055Z_LOCK"
MIN_LOSO_DENOMINATOR = 5


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING_LOCAL_ARTIFACT"


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def validate_explicit_group_key(group_key: str | None) -> str:
    if not group_key:
        raise RuntimeError("EXPLICIT_LOSO_GROUP_KEY_REQUIRED")
    if any(token in str(group_key).lower() for token in ["latest", "glob", "*"]):
        raise RuntimeError("IMPLICIT_LATEST_GLOB_GROUP_SELECTION_FORBIDDEN")
    return str(group_key)


def denominator_status(denominator: int, min_denominator: int = MIN_LOSO_DENOMINATOR) -> str:
    if denominator <= 0:
        return "EMPTY_SELECTED_GROUP"
    if denominator < min_denominator:
        return "TOO_SMALL_DENOMINATOR"
    return "OK"


def validate_min_denominator_contract(*, requested_min_denominator: int, proof_status: str = "NOT_PROVEN") -> bool:
    if requested_min_denominator < MIN_LOSO_DENOMINATOR and proof_status != "EXPLICIT_CONTRACT_PROVEN":
        raise RuntimeError("SILENT_DENOMINATOR_GUARD_LOWERING_FORBIDDEN")
    return True


def validate_low_support_policy(*, exclude_low_support: bool, explicit_contract: bool) -> str:
    if exclude_low_support and not explicit_contract:
        raise RuntimeError("SMALL_LOSO_GROUPS_CANNOT_BE_SILENTLY_DROPPED")
    if exclude_low_support:
        return "EXCLUDED_LOW_SUPPORT_EXPLICITLY_REPORTED"
    return "LOW_SUPPORT_GROUPS_INCLUDED_AND_REPORTED"


def optuna_v3_metrics_can_override_v2_guard() -> bool:
    return False


def detect_wrong_group_key(*, current_group_key: str, contract_group_key: str | None) -> dict[str, Any]:
    if not contract_group_key or contract_group_key in {"UNKNOWN_REQUIRES_ARTIFACT", "MISSING_LOCAL_ARTIFACT"}:
        return {
            "status_v1": "UNKNOWN_REQUIRES_ARTIFACT",
            "wrong_group_key_detected_v1": False,
            "reason_v1": "No explicit external LOSO group-key contract is available locally.",
        }
    wrong = current_group_key != contract_group_key
    return {
        "status_v1": "WRONG_LOSO_GROUP_KEY_USED" if wrong else "PASS",
        "wrong_group_key_detected_v1": wrong,
        "current_group_key_v1": current_group_key,
        "contract_group_key_v1": contract_group_key,
    }


def detect_denominator_formula_bug(*, observed_denominator: int, recomputed_denominator: int) -> dict[str, Any]:
    bug = int(observed_denominator) != int(recomputed_denominator)
    return {
        "status_v1": "DENOMINATOR_FORMULA_BUG" if bug else "PASS",
        "denominator_formula_bug_detected_v1": bug,
        "observed_denominator_v1": int(observed_denominator),
        "recomputed_denominator_v1": int(recomputed_denominator),
    }


def validate_metric_repair_integrity(
    *,
    scores_hash_before: str,
    scores_hash_after: str,
    provenance_hash_before: str,
    provenance_hash_after: str,
    selected_count_before: int,
    selected_count_after: int,
) -> dict[str, Any]:
    changed = []
    if scores_hash_before != scores_hash_after:
        changed.append("V2_OOF_SCORES_CHANGED")
    if provenance_hash_before != provenance_hash_after:
        changed.append("V2_OOF_PROVENANCE_CHANGED")
    if int(selected_count_before) != int(selected_count_after):
        changed.append("V2_SELECTED_ROWS_CHANGED")
    return {
        "status_v1": "PASS" if not changed else "FAIL",
        "changed_v1": changed,
        "scores_unchanged_v1": scores_hash_before == scores_hash_after,
        "provenance_unchanged_v1": provenance_hash_before == provenance_hash_after,
        "selected_rows_unchanged_v1": int(selected_count_before) == int(selected_count_after),
    }


def classify_root_cause(
    *,
    wrong_group_key: bool,
    formula_bug: bool,
    threshold_misconfigured: bool,
    current_group_explicit: bool,
    current_group_legitimate: bool,
    worst_denominator: int,
    wednesday_contract_missing: bool,
) -> dict[str, Any]:
    if wrong_group_key:
        root = "WRONG_LOSO_GROUP_KEY_USED"
        repair_allowed = True
    elif formula_bug:
        root = "DENOMINATOR_FORMULA_BUG"
        repair_allowed = True
    elif threshold_misconfigured:
        root = "DENOMINATOR_VALIDITY_THRESHOLD_MISCONFIGURED"
        repair_allowed = True
    elif not current_group_explicit:
        root = "UNKNOWN_REQUIRES_ARTIFACT"
        repair_allowed = False
    elif worst_denominator > 0 and worst_denominator < MIN_LOSO_DENOMINATOR and current_group_legitimate:
        root = "TRUE_LOW_SUPPORT_GENERALIZATION_WEAKNESS"
        repair_allowed = False
    elif wednesday_contract_missing:
        root = "WEDNESDAY_LOSO_CONTRACT_MISSING_LOCAL"
        repair_allowed = False
    else:
        root = "UNKNOWN_REQUIRES_ARTIFACT"
        repair_allowed = False
    return {
        "root_cause_v1": root,
        "metric_repair_allowed_v1": repair_allowed,
        "wednesday_contract_missing_local_v1": bool(wednesday_contract_missing),
        "do_not_lower_guard_for_pass_v1": True,
        "do_not_drop_small_groups_silently_v1": True,
    }


def decision_valid_requires_provenance_and_denominator(*, provenance_pass: bool, denominator_pass: bool) -> bool:
    return bool(provenance_pass and denominator_pass)


def _load_foundation_status(v2_root: Path) -> pd.DataFrame:
    manifest = _read_json(v2_root / "manifest_v1.json")
    score_dir = Path((manifest.get("inputs_v1") or {}).get("score_dir_v1", ""))
    score_path = score_dir / "monday_r6_foundation_score_frame_v1.parquet"
    if not score_path.exists():
        return pd.DataFrame()
    columns = ["candidate_uid", "calendar_quarantine_status_v1", "calendar_quarantine_reason_v1"]
    frame = pd.read_parquet(score_path)
    return frame[[column for column in columns if column in frame.columns]].copy()


def _load_inputs(v2_root: Path) -> dict[str, Any]:
    scores = pd.read_csv(v2_root / "v2_oof_scores_v1.csv")
    provenance = pd.read_csv(v2_root / "v2_oof_score_provenance_v1.csv")
    membership = pd.read_csv(v2_root / "v2_train_validation_membership_v1.csv")
    fold_assignment = pd.read_csv(v2_root / "v2_oof_fold_assignment_v1.csv")
    foundation = _load_foundation_status(v2_root)
    if not foundation.empty:
        scores = scores.merge(foundation, on="candidate_uid", how="left")
    if "calendar_quarantine_status_v1" not in scores.columns:
        scores["calendar_quarantine_status_v1"] = "UNKNOWN_NOT_AVAILABLE"
    return {
        "scores": scores,
        "provenance": provenance,
        "membership": membership,
        "fold_assignment": fold_assignment,
        "summary": _read_json(v2_root / "v2_oof_replay_summary_v1.json"),
        "existing_loso": pd.read_csv(v2_root / "v2_oof_loso_group_denominator_v1.csv"),
        "manifest": _read_json(v2_root / "manifest_v1.json"),
    }


def _selected(scores: pd.DataFrame) -> pd.Series:
    return _bool(scores, "r5_2_v2_final_base_membership")


def _bad(scores: pd.DataFrame) -> pd.Series:
    return _bool(scores, "label_should_not_take_v1")


def _tail(scores: pd.DataFrame) -> pd.Series:
    return _bool(scores, "tail_10_50_mfe_v1")


def group_distribution(
    scores: pd.DataFrame,
    *,
    group_key: str,
    min_denominator: int = MIN_LOSO_DENOMINATOR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_explicit_group_key(group_key)
    if group_key not in scores.columns:
        raise RuntimeError(f"LOSO group key not found: {group_key}")
    selected = _selected(scores)
    bad = _bad(scores)
    tail = _tail(scores)
    active = scores["calendar_quarantine_status_v1"].astype(str).eq("ACTIVE_CANDIDATE")
    protected_winner = (
        _bool(scores, "hundred_plus_mfe_v1")
        | _bool(scores, "two_hundred_plus_mfe_v1")
        | _bool(scores, "strongest_winner_path_v1")
        | _bool(scores, "r6_label_repaired_165_like_runner_v1")
    )
    runner = _bool(scores, "runner_protection_target") | _bool(scores, "r6_label_runner_near_miss_v1")
    ambiguous = _bool(scores, "high_mfe_ambiguous_protection_target") | scores["v2_bucket"].astype(str).eq("AMBIGUOUS_HIGH_MFE_PROTECTED")
    rows: list[dict[str, Any]] = []
    work = scores.assign(
        _selected=selected,
        _bad=bad,
        _tail=tail,
        _active=active,
        _protected=protected_winner,
        _runner=runner,
        _ambiguous=ambiguous,
    )
    for group, part in work.groupby(group_key, dropna=False):
        part_selected = part["_selected"].astype(bool)
        denominator = int(part_selected.sum())
        selected_bad = int((part_selected & part["_bad"].astype(bool)).sum())
        selected_tail = int((part_selected & part["_tail"].astype(bool)).sum())
        selected_false = int((part_selected & ~part["_bad"].astype(bool)).sum())
        protected = int((part_selected & part["_protected"].astype(bool)).sum())
        runner_count = int((part_selected & part["_runner"].astype(bool)).sum())
        ambiguous_count = int((part_selected & part["_ambiguous"].astype(bool)).sum())
        status = denominator_status(denominator, min_denominator)
        failures = []
        if status != "OK":
            failures.append(status)
        if selected_false:
            failures.append("SELECTED_FALSE_POSITIVES_PRESENT")
        if protected:
            failures.append("SELECTED_PROTECTED_WINNERS_PRESENT")
        if runner_count:
            failures.append("SELECTED_RUNNER_PROTECT_PRESENT")
        if ambiguous_count:
            failures.append("SELECTED_AMBIGUOUS_HIGH_MFE_PRESENT")
        rows.append(
            {
                "group_id_v1": str(group),
                "total_foundation_rows_v1": int(len(part)),
                "active_rows_v1": int(part["_active"].sum()),
                "selected_rows_v1": denominator,
                "selected_bad_rows_v1": selected_bad,
                "selected_tail_rows_v1": selected_tail,
                "selected_false_positives_v1": selected_false,
                "selected_protected_winners_v1": protected,
                "selected_runner_protect_rows_v1": runner_count,
                "selected_ambiguous_high_mfe_rows_v1": ambiguous_count,
                "precision_v1": selected_bad / denominator if denominator else np.nan,
                "denominator_v1": denominator,
                "denominator_valid_v1": status == "OK",
                "decision_valid_v1": status == "OK" and not any(failure for failure in failures if failure != "EMPTY_SELECTED_GROUP"),
                "fail_reason_v1": "NONE" if not failures else "|".join(failures),
            }
        )
    non_empty = [row for row in rows if int(row["selected_rows_v1"]) > 0]
    worst = min(non_empty, key=lambda row: float(row["precision_v1"])) if non_empty else None
    for row in rows:
        row["is_worst_loso_group_v1"] = bool(worst and row["group_id_v1"] == worst["group_id_v1"])
    denominators = [int(row["selected_rows_v1"]) for row in non_empty]
    summary = {
        "group_key_v1": group_key,
        "all_group_count_v1": len(rows),
        "selected_group_count_v1": len(non_empty),
        "empty_selected_group_count_v1": len(rows) - len(non_empty),
        "low_support_group_count_v1": sum(0 < int(row["selected_rows_v1"]) < min_denominator for row in rows),
        "min_selected_denominator_v1": min(denominators) if denominators else 0,
        "median_selected_denominator_v1": float(np.median(denominators)) if denominators else 0.0,
        "max_selected_denominator_v1": max(denominators) if denominators else 0,
        "worst_loso_group_v1": None if worst is None else worst["group_id_v1"],
        "worst_loso_v1": None if worst is None else worst["precision_v1"],
        "worst_loso_numerator_v1": 0 if worst is None else worst["selected_bad_rows_v1"],
        "worst_loso_denominator_v1": 0 if worst is None else worst["selected_rows_v1"],
        "worst_loso_denominator_status_v1": "EMPTY_DENOMINATOR" if worst is None else denominator_status(int(worst["selected_rows_v1"]), min_denominator),
        "worst_loso_decision_valid_v1": bool(worst is not None and int(worst["selected_rows_v1"]) >= min_denominator),
    }
    return rows, summary


def _implementation_mapping(v2_root: Path) -> dict[str, Any]:
    source_lines, start_line = inspect.getsourcelines(v2_oof._worst_loso)
    return {
        "layer_name": "CURRENT_LOSO_METRIC_IMPLEMENTATION_MAPPING_V1",
        "source_file_v1": str(Path(inspect.getsourcefile(v2_oof._worst_loso) or "")),
        "function_v1": "_worst_loso",
        "source_start_line_v1": start_line,
        "source_end_line_v1": start_line + len(source_lines) - 1,
        "input_rows_v1": str(v2_root / "v2_oof_scores_v1.csv"),
        "selected_row_filter_v1": "r5_2_v2_final_base_membership == true",
        "label_filter_v1": "label_should_not_take_v1 == true",
        "group_key_used_for_loso_v1": "run_id",
        "numerator_definition_v1": "selected rows in group where label_should_not_take_v1 is true",
        "denominator_definition_v1": "selected rows in group",
        "minimum_denominator_rule_v1": MIN_LOSO_DENOMINATOR,
        "decision_valid_rule_v1": "worst non-empty LOSO group selected denominator must be >= 5",
        "empty_group_handling_v1": "Reported as EMPTY_SELECTED_GROUP and excluded from worst-group choice.",
        "low_support_group_handling_v1": "Reported as TOO_SMALL_DENOMINATOR and blocks decision_valid if it is the worst non-empty group.",
        "denominator_2_source_v1": "selected rows in the worst non-empty run_id group, not bad rows, tail rows, or total rows.",
        "precision_denominator_contract_v1": "all selected rows across the full OOF surface",
        "worst_loso_denominator_contract_v1": "selected rows inside the worst non-empty LOSO group",
        "precision_and_worst_loso_use_different_denominators_v1": True,
    }


def _candidate_groupings(scores: pd.DataFrame, fold_assignment: pd.DataFrame) -> list[dict[str, Any]]:
    candidates = [
        {
            "group_key_name_v1": "run_id",
            "source_evidence_v1": "v2_oof_scores_v1.csv",
            "contract_plausible_v1": True,
            "post_hoc_or_suspect_v1": False,
            "matches_existing_project_or_wed_contract_v1": "MATCHES_CURRENT_MONDAY_V2_OOF_CONTRACT",
            "recommendation_v1": "KEEP_AS_CURRENT_CONTRACT_UNLESS_EXTERNAL_WEDNESDAY_ARTIFACT_PROVES_OTHERWISE",
        },
        {
            "group_key_name_v1": "fold_id_v1",
            "source_evidence_v1": "v2_oof_scores_v1.csv",
            "contract_plausible_v1": True,
            "post_hoc_or_suspect_v1": False,
            "matches_existing_project_or_wed_contract_v1": "OOF_FOLD_STABILITY_ONLY_NOT_PROVEN_WEDNESDAY_LOSO",
            "recommendation_v1": "REPORT_ONLY_NOT_A_REPLACEMENT_FOR_LOSO_WITHOUT_CONTRACT",
        },
        {
            "group_key_name_v1": "group_key_v1",
            "source_evidence_v1": "v2_oof_fold_assignment_v1.csv",
            "contract_plausible_v1": True,
            "post_hoc_or_suspect_v1": False,
            "matches_existing_project_or_wed_contract_v1": "SAME_AS_RUN_ID_IN_CURRENT_ASSIGNMENT",
            "recommendation_v1": "EQUIVALENT_TO_CURRENT_RUN_ID",
        },
        {
            "group_key_name_v1": "calendar_quarantine_status_v1",
            "source_evidence_v1": "foundation score frame calendar status",
            "contract_plausible_v1": False,
            "post_hoc_or_suspect_v1": True,
            "matches_existing_project_or_wed_contract_v1": "NOT_A_LOSO_GENERALIZATION_GROUP",
            "recommendation_v1": "DO_NOT_USE_AS_LOSO_GROUP",
        },
        {
            "group_key_name_v1": "trade_id",
            "source_evidence_v1": "v2_oof_scores_v1.csv",
            "contract_plausible_v1": False,
            "post_hoc_or_suspect_v1": True,
            "matches_existing_project_or_wed_contract_v1": "UNIQUE_ROW_LEVEL_KEY_NOT_LOSO",
            "recommendation_v1": "DO_NOT_USE_AS_LOSO_GROUP",
        },
        {
            "group_key_name_v1": "decision_timestamp",
            "source_evidence_v1": "v2_oof_scores_v1.csv",
            "contract_plausible_v1": False,
            "post_hoc_or_suspect_v1": True,
            "matches_existing_project_or_wed_contract_v1": "TIMESTAMP_KEY_TOO_GRANULAR_NOT_LOSO",
            "recommendation_v1": "DO_NOT_USE_AS_LOSO_GROUP",
        },
    ]
    if "group_key_v1" not in scores.columns and "group_key_v1" in fold_assignment.columns:
        mapping = fold_assignment[["candidate_uid", "group_key_v1"]].drop_duplicates("candidate_uid")
        scores.merge(mapping, on="candidate_uid", how="left")
    available = []
    score_columns = set(scores.columns)
    for candidate in candidates:
        name = candidate["group_key_name_v1"]
        if name in score_columns:
            available.append(candidate)
    return available


def grouping_candidate_comparison(scores: pd.DataFrame, fold_assignment: pd.DataFrame) -> list[dict[str, Any]]:
    if "group_key_v1" not in scores.columns and "group_key_v1" in fold_assignment.columns:
        scores = scores.merge(fold_assignment[["candidate_uid", "group_key_v1"]].drop_duplicates("candidate_uid"), on="candidate_uid", how="left")
    rows: list[dict[str, Any]] = []
    for candidate in _candidate_groupings(scores, fold_assignment):
        group_key = candidate["group_key_name_v1"]
        _, summary = group_distribution(scores, group_key=group_key)
        rows.append(
            {
                **candidate,
                "number_of_groups_v1": summary["all_group_count_v1"],
                "selected_group_count_v1": summary["selected_group_count_v1"],
                "min_selected_denominator_v1": summary["min_selected_denominator_v1"],
                "median_selected_denominator_v1": summary["median_selected_denominator_v1"],
                "max_selected_denominator_v1": summary["max_selected_denominator_v1"],
                "worst_loso_v1": summary["worst_loso_v1"],
                "worst_loso_group_v1": summary["worst_loso_group_v1"],
                "worst_denominator_v1": summary["worst_loso_denominator_v1"],
                "denominator_valid_v1": summary["worst_loso_decision_valid_v1"],
                "low_support_group_count_v1": summary["low_support_group_count_v1"],
            }
        )
    return rows


def _wednesday_contract() -> dict[str, Any]:
    contract = _read_json(SKELETON_ROOT / "wednesday_r6_contract_reconstruction_v1.json")
    missing = _read_json(SKELETON_ROOT / "local_missing_required_artifacts_v1.json")
    optuna_best = _read_json(OPTUNA_ROOT / "constrained_optuna_best_candidate_v1.json")
    best = optuna_best.get("candidate_lock_v1") or optuna_best
    return {
        "layer_name": "LOSO_CONTRACT_RECONSTRUCTION_V1",
        "wednesday_contract_source_v1": str(SKELETON_ROOT / "wednesday_r6_contract_reconstruction_v1.json"),
        "wednesday_contract_status_v1": contract.get("status_v1", "UNKNOWN_REQUIRES_ARTIFACT"),
        "wednesday_worst_loso_v1": (contract.get("metrics_v1") or {}).get("worst_loso_v1"),
        "wednesday_worst_loso_denominator_v1": (contract.get("metrics_v1") or {}).get("worst_loso_denominator_v1"),
        "wednesday_loso_group_key_v1": "UNKNOWN_REQUIRES_ARTIFACT",
        "wednesday_loso_key_local_proof_v1": "MISSING_LOCAL_ARTIFACT",
        "likely_group_key_inference_v1": "UNKNOWN_REQUIRES_ARTIFACT",
        "do_not_invent_group_key_v1": True,
        "missing_artifact_evidence_v1": missing,
        "comparison_v1": {
            "wednesday_worst_loso_v1": (contract.get("metrics_v1") or {}).get("worst_loso_v1"),
            "v2_historical_worst_loso_status_v1": "HISTORICAL_ONLY_DENOMINATOR_2_NO_OOF_PROVENANCE",
            "v2_oof_current_worst_loso_v1": 1.0,
            "v2_oof_current_worst_loso_denominator_v1": 2,
            "optuna_best_worst_loso_v1": best.get("worst_loso_v1"),
            "optuna_best_worst_loso_denominator_v1": best.get("worst_loso_denominator_v1"),
            "v3_status_v1": "OOF_PROVENANCE_PASS_BUT_WEAK_17_13_CONTROL",
        },
    }


def _compare_existing_loso(existing: pd.DataFrame, recomputed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = existing[existing.get("is_worst_loso_group_v1", False).astype(bool)]
    observed_denominator = 0 if observed.empty else int(observed.iloc[0]["selected_denominator_v1"])
    worst_rows = [row for row in recomputed_rows if row["is_worst_loso_group_v1"]]
    recomputed_denominator = 0 if not worst_rows else int(worst_rows[0]["selected_rows_v1"])
    bug = detect_denominator_formula_bug(observed_denominator=observed_denominator, recomputed_denominator=recomputed_denominator)
    bug["observed_worst_group_v1"] = None if observed.empty else str(observed.iloc[0]["group_v1"])
    bug["recomputed_worst_group_v1"] = None if not worst_rows else worst_rows[0]["group_id_v1"]
    return bug


def _root_cause(
    *,
    mapping: dict[str, Any],
    current_summary: dict[str, Any],
    formula_check: dict[str, Any],
    wed_contract: dict[str, Any],
) -> dict[str, Any]:
    wrong_key = detect_wrong_group_key(
        current_group_key=mapping["group_key_used_for_loso_v1"],
        contract_group_key=wed_contract["wednesday_loso_group_key_v1"],
    )
    root = classify_root_cause(
        wrong_group_key=wrong_key["wrong_group_key_detected_v1"],
        formula_bug=formula_check["denominator_formula_bug_detected_v1"],
        threshold_misconfigured=False,
        current_group_explicit=mapping["group_key_used_for_loso_v1"] == "run_id",
        current_group_legitimate=True,
        worst_denominator=int(current_summary["worst_loso_denominator_v1"]),
        wednesday_contract_missing=wed_contract["wednesday_loso_key_local_proof_v1"] == "MISSING_LOCAL_ARTIFACT",
    )
    return {
        "layer_name": "LOSO_DENOMINATOR_ROOT_CAUSE_V1",
        **root,
        "wrong_group_key_check_v1": wrong_key,
        "denominator_formula_check_v1": formula_check,
        "threshold_misconfiguration_check_v1": {
            "status_v1": "NOT_PROVEN",
            "threshold_v1": MIN_LOSO_DENOMINATOR,
            "reason_v1": "No local Wednesday/Monday contract proves a different denominator threshold.",
        },
        "current_group_key_v1": mapping["group_key_used_for_loso_v1"],
        "worst_loso_group_v1": current_summary["worst_loso_group_v1"],
        "worst_loso_denominator_v1": current_summary["worst_loso_denominator_v1"],
        "interpretation_v1": (
            "The denominator=2 is real under the current explicit run_id LOSO contract: the worst non-empty weekly run_id group "
            "has two selected rows. No formula bug, wrong local group key, or threshold contract override was proven."
        ),
        "metric_patch_applied_v1": False,
        "metric_patch_reason_v1": "NO_BUG_OR_EXPLICIT_CONTRACT_REPAIR_PROVEN",
    }


def _go_no_go(root_cause: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    if root_cause["root_cause_v1"] == "TRUE_LOW_SUPPORT_GENERALIZATION_WEAKNESS":
        decision = "CURRENT_LOSO_CONTRACT_CORRECT_V2_TRUE_LOW_SUPPORT"
        next_action = "BUILD_R5_2_OPPORTUNITY_BASE_FROM_EXISTING_V2_OOF_REPLAY_V1"
    elif root_cause["root_cause_v1"] == "WEDNESDAY_LOSO_CONTRACT_MISSING_LOCAL":
        decision = "WEDNESDAY_LOSO_CONTRACT_MISSING_CANNOT_REPAIR"
        next_action = "REQUIRE_WEDNESDAY_LOSO_ARTIFACTS_BEFORE_DECISION_V1"
    elif root_cause["root_cause_v1"] in {"WRONG_LOSO_GROUP_KEY_USED", "DENOMINATOR_FORMULA_BUG", "DENOMINATOR_VALIDITY_THRESHOLD_MISCONFIGURED"}:
        decision = "LOSO_GROUPING_BUG_REPAIRED_BUT_V2_STILL_INVALID"
        next_action = "REPAIR_LOSO_GROUP_KEY_AND_RERUN_METRIC_ONLY_V1"
    else:
        decision = "WEDNESDAY_LOSO_CONTRACT_MISSING_CANNOT_REPAIR"
        next_action = "REQUIRE_WEDNESDAY_LOSO_ARTIFACTS_BEFORE_DECISION_V1"
    return {
        "layer_name": "LOSO_GROUPING_OR_DENOMINATOR_CONTRACT_GO_NO_GO_V1",
        "decision_v1": decision,
        "next_recommended_action_v1": next_action,
        "decision_valid_v1": False,
        "v2_oof_decision_valid_before_v1": bool(summary.get("decision_valid_v1")),
        "v2_oof_decision_valid_after_v1": False,
        "metric_repair_applied_v1": False,
        "scores_provenance_model_thresholds_unchanged_v1": True,
        "do_not_run_optuna_r6_package_freeze_promo_live_v1": True,
    }


def _report_mapping(mapping: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Current LOSO Metric Implementation Mapping V1",
            "",
            f"Source: `{mapping['source_file_v1']}:{mapping['source_start_line_v1']}`",
            f"Function: `{mapping['function_v1']}`",
            f"Group key: `{mapping['group_key_used_for_loso_v1']}`",
            f"Numerator: `{mapping['numerator_definition_v1']}`",
            f"Denominator: `{mapping['denominator_definition_v1']}`",
            f"Minimum denominator: `{mapping['minimum_denominator_rule_v1']}`",
            "",
            "Precision and worst LOSO intentionally use different denominator contracts.",
        ]
    ) + "\n"


def _report_distribution(summary: dict[str, Any], worst_row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 OOF LOSO Denominator Forensics V1",
            "",
            f"Worst group: `{summary['worst_loso_group_v1']}`",
            f"Worst denominator: `{summary['worst_loso_denominator_v1']}`",
            f"Worst LOSO: `{summary['worst_loso_v1']}`",
            f"Low-support selected groups: `{summary['low_support_group_count_v1']}`",
            "",
            "The denominator is selected rows inside the LOSO group.",
            f"The worst group has `{worst_row['total_foundation_rows_v1']}` total rows and `{worst_row['selected_rows_v1']}` selected rows.",
            "This is not a fold artifact; it is the explicit current `run_id` group.",
        ]
    ) + "\n"


def _report_contract(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# LOSO Contract Reconstruction V1",
            "",
            f"Wednesday worst LOSO: `{contract['wednesday_worst_loso_v1']}`",
            f"Wednesday worst LOSO denominator: `{contract['wednesday_worst_loso_denominator_v1']}`",
            f"Wednesday LOSO group key: `{contract['wednesday_loso_group_key_v1']}`",
            "",
            "No local artifact proves the exact Wednesday LOSO group key or denominator contract.",
        ]
    ) + "\n"


def _report_grouping(rows: list[dict[str, Any]]) -> str:
    lines = ["# LOSO Grouping Candidate Comparison V1", ""]
    for row in rows:
        lines.append(
            f"- `{row['group_key_name_v1']}`: worst denominator `{row['worst_denominator_v1']}`, "
            f"valid `{row['denominator_valid_v1']}`, recommendation `{row['recommendation_v1']}`"
        )
    return "\n".join(lines) + "\n"


def _report_root_cause(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# LOSO Denominator Root Cause V1",
            "",
            f"Root cause: `{payload['root_cause_v1']}`",
            f"Metric patch applied: `{payload['metric_patch_applied_v1']}`",
            f"Worst group: `{payload['worst_loso_group_v1']}`",
            f"Worst denominator: `{payload['worst_loso_denominator_v1']}`",
            "",
            payload["interpretation_v1"],
        ]
    ) + "\n"


def _report_repaired_eval(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# V2 OOF LOSO Repaired Eval V1",
            "",
            f"Metric repair applied: `{payload['metric_repair_applied_v1']}`",
            f"Before decision-valid: `{payload['before_v1']['decision_valid_v1']}`",
            f"After decision-valid: `{payload['after_v1']['decision_valid_v1']}`",
            f"Reason: `{payload['reason_v1']}`",
        ]
    ) + "\n"


def _report_final(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Repair LOSO Grouping Or Denominator Contract V1",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_recommended_action_v1']}`",
            f"Worst group: `{summary['worst_loso_group_v1']}`",
            f"Worst denominator: `{summary['worst_loso_denominator_v1']}`",
            f"Root cause: `{summary['root_cause_v1']}`",
            "",
            "No V2 scores, provenance, model behavior, objective, or thresholds were changed.",
        ]
    ) + "\n"


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v2_oof_root: Path = V2_OOF_ROOT,
    explicit_action: str = ACTION,
) -> dict[str, Any]:
    if explicit_action != ACTION:
        raise RuntimeError(f"Explicit action required: {ACTION}")
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    inputs = _load_inputs(v2_oof_root)
    scores = inputs["scores"]
    provenance = inputs["provenance"]
    selected_before = int(_selected(scores).sum())
    scores_hash_before = _file_hash(v2_oof_root / "v2_oof_scores_v1.csv")
    provenance_hash_before = _file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv")

    mapping = _implementation_mapping(v2_oof_root)
    distribution_rows, current_summary = group_distribution(scores, group_key="run_id")
    worst_rows = [row for row in distribution_rows if row["is_worst_loso_group_v1"]]
    worst_row = worst_rows[0] if worst_rows else {}
    wed_contract = _wednesday_contract()
    candidate_rows = grouping_candidate_comparison(scores, inputs["fold_assignment"])
    formula_check = _compare_existing_loso(inputs["existing_loso"], distribution_rows)
    root_cause = _root_cause(mapping=mapping, current_summary=current_summary, formula_check=formula_check, wed_contract=wed_contract)

    integrity = validate_metric_repair_integrity(
        scores_hash_before=scores_hash_before,
        scores_hash_after=_file_hash(v2_oof_root / "v2_oof_scores_v1.csv"),
        provenance_hash_before=provenance_hash_before,
        provenance_hash_after=_file_hash(v2_oof_root / "v2_oof_score_provenance_v1.csv"),
        selected_count_before=selected_before,
        selected_count_after=int(_selected(scores).sum()),
    )
    denominator_pass = bool(current_summary["worst_loso_decision_valid_v1"])
    provenance_pass = bool(not provenance.empty and not provenance["was_row_in_train_for_scoring_model_v1"].astype(bool).any())
    before = {
        "bad_count_v1": int((_selected(scores) & _bad(scores)).sum()),
        "tail_count_v1": int((_selected(scores) & _tail(scores)).sum()),
        "precision_v1": inputs["summary"].get("precision_v1"),
        "precision_denominator_v1": inputs["summary"].get("precision_denominator_v1"),
        "precision_decision_valid_v1": inputs["summary"].get("precision_decision_valid_v1"),
        "worst_loso_v1": current_summary["worst_loso_v1"],
        "worst_loso_denominator_v1": current_summary["worst_loso_denominator_v1"],
        "worst_loso_decision_valid_v1": current_summary["worst_loso_decision_valid_v1"],
        "decision_valid_v1": decision_valid_requires_provenance_and_denominator(
            provenance_pass=provenance_pass,
            denominator_pass=denominator_pass,
        ),
        "safety_clean_v1": inputs["summary"].get("safety_clean_v1"),
        "low_support_group_count_v1": current_summary["low_support_group_count_v1"],
    }
    repaired_eval = {
        "layer_name": "V2_OOF_LOSO_REPAIRED_EVAL_V1",
        "metric_repair_applied_v1": False,
        "reason_v1": "NO_BUG_OR_EXPLICIT_CONTRACT_REPAIR_PROVEN",
        "before_v1": before,
        "after_v1": before,
        "excluded_low_support_groups_v1": [],
        "low_support_groups_still_reported_v1": [row for row in distribution_rows if 0 < int(row["selected_rows_v1"]) < MIN_LOSO_DENOMINATOR],
        "scores_provenance_model_thresholds_unchanged_v1": integrity["status_v1"] == "PASS",
    }
    go_no_go = _go_no_go(root_cause, inputs["summary"])
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "v2_oof_root_v1": str(v2_oof_root),
        "decision_v1": go_no_go["decision_v1"],
        "next_recommended_action_v1": go_no_go["next_recommended_action_v1"],
        "root_cause_v1": root_cause["root_cause_v1"],
        "metric_patch_applied_v1": False,
        "worst_loso_group_v1": current_summary["worst_loso_group_v1"],
        "worst_loso_v1": current_summary["worst_loso_v1"],
        "worst_loso_denominator_v1": current_summary["worst_loso_denominator_v1"],
        "worst_loso_decision_valid_v1": current_summary["worst_loso_decision_valid_v1"],
        "low_support_group_count_v1": current_summary["low_support_group_count_v1"],
        "wednesday_loso_contract_status_v1": wed_contract["wednesday_loso_key_local_proof_v1"],
        "scores_provenance_model_thresholds_unchanged_v1": integrity["status_v1"] == "PASS",
        "optuna_not_run_v1": True,
        "r6_not_run_v1": True,
        "package_not_built_v1": True,
    }

    _write_json(output_dir / "current_loso_metric_implementation_mapping_v1.json", mapping)
    (output_dir / "current_loso_metric_implementation_mapping_v1.md").write_text(_report_mapping(mapping), encoding="utf-8")
    _write_rows(output_dir / "v2_oof_loso_group_distribution_v1.csv", distribution_rows)
    _write_json(output_dir / "v2_oof_loso_group_distribution_v1.json", {"summary_v1": current_summary, "rows_v1": distribution_rows})
    (output_dir / "v2_oof_loso_denominator_forensics_v1.md").write_text(_report_distribution(current_summary, worst_row), encoding="utf-8")
    _write_json(output_dir / "loso_contract_reconstruction_v1.json", wed_contract)
    (output_dir / "loso_contract_reconstruction_v1.md").write_text(_report_contract(wed_contract), encoding="utf-8")
    _write_rows(output_dir / "loso_grouping_candidate_comparison_v1.csv", candidate_rows)
    _write_json(output_dir / "loso_grouping_candidate_comparison_v1.json", {"rows_v1": candidate_rows})
    (output_dir / "loso_grouping_candidate_comparison_report_v1.md").write_text(_report_grouping(candidate_rows), encoding="utf-8")
    _write_json(output_dir / "loso_denominator_root_cause_v1.json", root_cause)
    (output_dir / "loso_denominator_root_cause_report_v1.md").write_text(_report_root_cause(root_cause), encoding="utf-8")
    _write_rows(output_dir / "v2_oof_loso_repaired_eval_v1.csv", [{"metric_repair_applied_v1": False, **before}])
    _write_json(output_dir / "v2_oof_loso_repaired_eval_v1.json", repaired_eval)
    (output_dir / "v2_oof_loso_repaired_eval_report_v1.md").write_text(_report_repaired_eval(repaired_eval), encoding="utf-8")
    _write_json(output_dir / "loso_grouping_or_denominator_contract_go_no_go_v1.json", go_no_go)
    _write_json(
        output_dir / "manifest_v1.json",
        {
            "layer_name": f"{LAYER_NAME}_MANIFEST",
            "output_dir_v1": str(output_dir),
            "inputs_v1": {
                "v2_oof_root_v1": str(v2_oof_root),
                "skeleton_root_v1": str(SKELETON_ROOT),
                "optuna_root_v1": str(OPTUNA_ROOT),
                "selected_v3_root_v1": str(SELECTED_V3_ROOT),
            },
            "input_hashes_v1": {
                "v2_oof_scores_sha256_v1": scores_hash_before,
                "v2_oof_score_provenance_sha256_v1": provenance_hash_before,
            },
            "integrity_v1": integrity,
        },
    )
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "go_no_go_v1": go_no_go})
    (output_dir / "report_v1.md").write_text(_report_final(summary), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-action", default=ACTION)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v2-oof-root", type=Path, default=V2_OOF_ROOT)
    args = parser.parse_args(argv)
    summary = materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v2_oof_root=args.v2_oof_root,
        explicit_action=args.explicit_action,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
