#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gx1.scripts import materialize_foundation_integrity_and_hidden_drift_audit_before_optuna_v1 as foundation_audit


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_V3_OOF_DIR = foundation_audit.DEFAULT_V3_OOF_DIR
DEFAULT_V3_IN_SAMPLE_DIR = foundation_audit.DEFAULT_V3_IN_SAMPLE_DIR
DEFAULT_OPTUNA_DIR = foundation_audit.DEFAULT_OPTUNA_DIR
LAYER_NAME = "FIX_SCORE_PROVENANCE_AND_METRIC_DENOMINATOR_GUARDS_BEFORE_OPTUNA_V1"
EXPECTED_ROWS = 1914
SCORE_FIELDS = [
    "r5_2_v3_bad_recall_score",
    "r5_2_v3_tail_recall_score",
    "r5_2_v3_risky_attention_score",
    "r5_2_v3_runner_protection_score",
    "r5_2_v3_high_mfe_ambiguous_protection_score",
    "r5_2_v3_hard_winner_protection_score",
]
REQUIRED_KEYS = ["candidate_uid", "trade_uid", "decision_timestamp"]
REQUIRED_PROVENANCE_COLUMNS = [
    *REQUIRED_KEYS,
    "variant_id_v1",
    "score_field_v1",
    "fold_id_v1",
    "group_key_v1",
    "train_validation_membership_v1",
    "source_model_fold_v1",
    "score_source_v1",
    "row_was_in_training_for_source_model_v1",
    "in_sample_score_used_v1",
    "fallback_score_used_v1",
    "synthetic_score_used_v1",
]
OUTPUT_FILES = [
    "v3_oof_score_provenance_contract_v1.json",
    "v3_oof_score_provenance_reconstruction_or_invalidation_v1.json",
    "v3_oof_score_provenance_v1.csv",
    "v3_oof_fold_assignment_v1.csv",
    "v3_oof_score_source_manifest_v1.json",
    "v3_train_validation_membership_v1.csv",
    "oof_score_provenance_validation_v1.json",
    "metric_denominator_guard_contract_v1.json",
    "metric_denominator_guard_audit_v1.csv",
    "foundation_integrity_recheck_after_fix_v1.json",
    "go_no_go_before_optuna_v1.json",
    "next_action_lock_v1.json",
    "summary_v1.json",
    "report_v1.md",
    "manifest_v1.json",
    "status_v1.json",
    "consistency_audit_v1.csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _contract() -> dict[str, Any]:
    return {
        "layer_name": "V3_OOF_SCORE_PROVENANCE_CONTRACT_V1",
        "required_score_fields_v1": SCORE_FIELDS,
        "required_row_fields_v1": REQUIRED_KEYS,
        "required_provenance_fields_v1": REQUIRED_PROVENANCE_COLUMNS,
        "hard_fail_if_score_exists_without_provenance_v1": True,
        "score_source_required_v1": "OOF",
        "row_was_in_training_for_source_model_required_v1": False,
        "forbidden_v1": [
            "guess_fold_id",
            "guess_source_model",
            "assume_oof_from_field_name",
            "fallback_to_in_sample",
            "dummy_provenance",
            "synthetic_provenance",
        ],
    }


def _best_variant(v3_oof_dir: Path) -> tuple[str, Path, pd.DataFrame]:
    index = pd.read_csv(v3_oof_dir / "v3_variant_outputs_index_v1.csv")
    best_lock = _read_json(v3_oof_dir / "best_v3_variant_downstream_r6_input_lock_v1.json")
    best_variant = str(best_lock.get("best_variant_id_v1") or index.iloc[0]["variant_id_v1"])
    best_row = index[index["variant_id_v1"].astype(str).eq(best_variant)]
    if best_row.empty:
        best_row = index.iloc[[0]]
    variant_dir = Path(str(best_row.iloc[0]["variant_dir_v1"]))
    prediction = pd.read_parquet(variant_dir / "prediction_view_v1.parquet")
    return best_variant, variant_dir, prediction


def validate_oof_score_provenance(provenance: pd.DataFrame, prediction: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> dict[str, Any]:
    failures: list[str] = []
    missing_columns = [column for column in REQUIRED_PROVENANCE_COLUMNS if column not in provenance.columns]
    if missing_columns:
        return {
            "layer_name": "OOF_SCORE_PROVENANCE_VALIDATION_V1",
            "status_v1": "FAIL_MISSING_PROVENANCE",
            "failure_reasons_v1": [f"MISSING_COLUMNS:{missing_columns}"],
        }
    for score_field in SCORE_FIELDS:
        if score_field not in prediction.columns:
            failures.append(f"MISSING_SCORE_FIELD:{score_field}")
            continue
        field_rows = provenance[provenance["score_field_v1"].astype(str).eq(score_field)].copy()
        if len(field_rows) != expected_rows:
            failures.append(f"ROW_COVERAGE_MISMATCH:{score_field}:{len(field_rows)}")
        if field_rows["fold_id_v1"].isna().any():
            failures.append(f"MISSING_FOLD_ID:{score_field}")
        if field_rows["group_key_v1"].isna().any():
            failures.append(f"MISSING_GROUP_KEY:{score_field}")
    if not provenance["score_source_v1"].astype(str).eq("OOF").all():
        failures.append("FAIL_SCORE_SOURCE_NOT_OOF")
    if provenance["row_was_in_training_for_source_model_v1"].astype(bool).any():
        failures.append("FAIL_TRAIN_VALIDATION_LEAKAGE")
    if provenance["in_sample_score_used_v1"].astype(bool).any():
        failures.append("FAIL_IN_SAMPLE_SCORE_USED")
    if provenance["fallback_score_used_v1"].astype(bool).any():
        failures.append("FAIL_FALLBACK_SCORE_USED")
    if provenance["synthetic_score_used_v1"].astype(bool).any():
        failures.append("FAIL_SYNTHETIC_SCORE_USED")
    status = "PASS"
    if failures:
        if any(reason.startswith("FAIL_TRAIN_VALIDATION_LEAKAGE") for reason in failures):
            status = "FAIL_TRAIN_VALIDATION_LEAKAGE"
        elif "FAIL_IN_SAMPLE_SCORE_USED" in failures:
            status = "FAIL_IN_SAMPLE_SCORE_USED"
        elif "FAIL_FALLBACK_SCORE_USED" in failures:
            status = "FAIL_FALLBACK_SCORE_USED"
        elif "FAIL_SYNTHETIC_SCORE_USED" in failures:
            status = "FAIL_SYNTHETIC_SCORE_USED"
        else:
            status = "FAIL_MISSING_PROVENANCE"
    return {
        "layer_name": "OOF_SCORE_PROVENANCE_VALIDATION_V1",
        "status_v1": status,
        "expected_rows_v1": expected_rows,
        "expected_score_field_count_v1": len(SCORE_FIELDS),
        "provenance_rows_v1": int(len(provenance)),
        "failure_reasons_v1": failures,
    }


def _reconstruct_or_invalidate(v3_oof_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    best_variant, variant_dir, prediction = _best_variant(v3_oof_dir)
    provenance_path = variant_dir / "v3_oof_score_provenance_v1.csv"
    fold_path = variant_dir / "v3_oof_fold_assignment_v1.csv"
    membership_path = variant_dir / "v3_train_validation_membership_v1.csv"
    if provenance_path.exists() and fold_path.exists() and membership_path.exists():
        provenance = pd.read_csv(provenance_path)
        fold = pd.read_csv(fold_path)
        membership = pd.read_csv(membership_path)
        validation = validate_oof_score_provenance(provenance, prediction)
        status = "RECONSTRUCTED_FROM_EXISTING_PROVENANCE_ARTIFACTS" if validation["status_v1"] == "PASS" else "INVALID_FOR_OPTUNA_DECISIONING"
        return (
            {
                "layer_name": "V3_OOF_SCORE_PROVENANCE_RECONSTRUCTION_OR_INVALIDATION_V1",
                "best_variant_id_v1": best_variant,
                "source_variant_dir_v1": str(variant_dir),
                "reconstruction_status_v1": status,
                "existing_provenance_paths_v1": {
                    "provenance_v1": str(provenance_path),
                    "fold_assignment_v1": str(fold_path),
                    "membership_v1": str(membership_path),
                },
                "no_dummy_or_synthetic_provenance_used_v1": True,
            },
            provenance,
            fold,
            membership,
            validation,
        )

    invalid_rows = [
        {
            "variant_id_v1": best_variant,
            "score_field_v1": field,
            "status_v1": "INVALID_FOR_OPTUNA_DECISIONING",
            "reason_v1": "MISSING_STORED_OOF_FOLD_ASSIGNMENT_OR_SOURCE_MODEL_PROVENANCE",
            "score_source_v1": "NOT_ESTABLISHED",
            "dummy_provenance_used_v1": False,
            "synthetic_provenance_used_v1": False,
        }
        for field in SCORE_FIELDS
    ]
    empty_fold = pd.DataFrame(columns=[*REQUIRED_KEYS, "variant_id_v1", "fold_id_v1", "group_key_v1", "train_validation_membership_v1"])
    empty_membership = pd.DataFrame(columns=["variant_id_v1", "score_field_v1", "fold_id_v1", "group_key_v1", "source_model_fold_v1", "train_validation_membership_v1"])
    validation = {
        "layer_name": "OOF_SCORE_PROVENANCE_VALIDATION_V1",
        "status_v1": "FAIL_MISSING_PROVENANCE",
        "expected_rows_v1": EXPECTED_ROWS,
        "expected_score_field_count_v1": len(SCORE_FIELDS),
        "provenance_rows_v1": 0,
        "failure_reasons_v1": ["EXISTING_V3_OOF_SCORES_LACK_STORED_FOLD_ASSIGNMENT_AND_SOURCE_MODEL_PROVENANCE"],
    }
    return (
        {
            "layer_name": "V3_OOF_SCORE_PROVENANCE_RECONSTRUCTION_OR_INVALIDATION_V1",
            "best_variant_id_v1": best_variant,
            "source_variant_dir_v1": str(variant_dir),
            "reconstruction_status_v1": "INVALID_FOR_OPTUNA_DECISIONING",
            "invalidated_score_fields_v1": SCORE_FIELDS,
            "why_not_reconstructed_v1": "No existing fold assignment, train/validation membership, or source model/fold provenance artifacts were present. Recomputing GroupKFold now would be inferred provenance, not stored proof.",
            "next_action_if_invalid_v1": "RERUN_V3_OOF_SCORING_WITH_PROVENANCE_FIRST",
            "no_dummy_or_synthetic_provenance_used_v1": True,
        },
        pd.DataFrame(invalid_rows),
        empty_fold,
        empty_membership,
        validation,
    )


def metric_denominator_guard_contract() -> dict[str, Any]:
    return {
        "layer_name": "METRIC_DENOMINATOR_GUARD_CONTRACT_V1",
        "metrics_requiring_denominator_metadata_v1": [
            "precision",
            "worst_loso_precision",
            "per_batch_precision",
            "per_pocket_precision",
            "tail_precision",
            "selected_bad_precision",
            "runner_winner_damage_rates",
        ],
        "required_fields_per_metric_v1": [
            "numerator",
            "denominator",
            "min_denominator_requirement",
            "denominator_status",
            "metric_value",
            "decision_valid",
            "warning_or_fail_reason",
        ],
        "denominator_statuses_v1": ["OK", "EMPTY_DENOMINATOR", "TOO_SMALL_DENOMINATOR", "NOT_APPLICABLE_EXPLICIT"],
        "hard_rules_v1": [
            "empty_denominator_never_silent_1_0",
            "too_small_denominator_not_strong_pass",
            "worst_loso_empty_folds_not_silent_pass",
            "gate_stops_when_decision_valid_false",
        ],
    }


def _metric_guard_audit() -> tuple[pd.DataFrame, list[str]]:
    files = [
        Path("gx1/scripts/run_r5_2_objective_v3_parallel_rebuild_runner_v1.py"),
        Path("gx1/scripts/materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1.py"),
    ]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        unsafe = any(pattern in text for pattern in ["default=1.0", "final_count == 0", "NO_SELECTED_GROUPS", "precision = 1.0 if"])
        has_metadata = "denominator_status_v1" in text and "decision_valid_v1" in text
        status = "PASS" if not unsafe and has_metadata else "FAIL"
        rows.append(
            {
                "file_v1": str(file),
                "unsafe_empty_denominator_pattern_found_v1": unsafe,
                "denominator_metadata_present_v1": has_metadata,
                "status_v1": status,
            }
        )
        if status != "PASS":
            failures.append(f"METRIC_GUARD_INCOMPLETE:{file}")
    return pd.DataFrame(rows), failures


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    report = "\n".join(
        [
            "# Score Provenance And Metric Denominator Guards Before Optuna",
            "",
            f"Decision: `{summary['decision_v1']}`",
            f"Next action: `{summary['next_action_v1']}`",
            "",
            f"- Existing V3 OOF provenance status: `{summary['v3_oof_provenance_status_v1']}`",
            f"- Metric denominator guard status: `{summary['metric_denominator_guard_status_v1']}`",
            f"- Foundation recheck decision: `{summary['foundation_recheck_decision_v1']}`",
            f"- Optuna ready: `{summary['foundation_clean_ready_for_optuna_v1']}`",
            f"- Historical invalid V3 status: `{summary['historical_invalid_v3_artifacts_status_v1']}`",
            "",
            "No Optuna, R6, baseline build, feature build, or model training was run.",
        ]
    )
    (output_dir / "report_v1.md").write_text(report + "\n", encoding="utf-8")


def materialize(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_dir: Path | None = None,
    v3_oof_dir: Path = DEFAULT_V3_OOF_DIR,
    v3_in_sample_dir: Path = DEFAULT_V3_IN_SAMPLE_DIR,
    optuna_dir: Path = DEFAULT_OPTUNA_DIR,
    selected_v3_oof_artifact_root: Path | None = None,
    active_score_artifact_selection: Path | None = None,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}_LOCK"
    output_dir.mkdir(parents=True, exist_ok=False)

    contract = _contract()
    reconstruction, provenance, fold_assignment, membership, provenance_validation = _reconstruct_or_invalidate(v3_oof_dir)
    metric_contract = metric_denominator_guard_contract()
    metric_audit, metric_failures = _metric_guard_audit()
    source_manifest = {
        "layer_name": "V3_OOF_SCORE_SOURCE_MANIFEST_V1",
        "score_source_contract_v1": "OOF_ONLY_WITH_STORED_FOLD_SOURCE_PROVENANCE",
        "existing_scores_valid_for_optuna_decisioning_v1": provenance_validation["status_v1"] == "PASS",
        "source_status_v1": reconstruction["reconstruction_status_v1"],
        "score_fields_v1": SCORE_FIELDS,
        "no_dummy_or_synthetic_provenance_used_v1": True,
    }

    foundation_recheck_dir = output_dir / "foundation_integrity_recheck_output_v1"
    foundation_summary = foundation_audit.materialize(
        reports_root=reports_root,
        output_dir=foundation_recheck_dir,
        v3_oof_dir=selected_v3_oof_artifact_root or v3_oof_dir,
        v3_in_sample_dir=v3_in_sample_dir,
        optuna_dir=optuna_dir,
        selected_v3_oof_artifact_root=selected_v3_oof_artifact_root,
        active_score_artifact_selection=active_score_artifact_selection,
        require_explicit_artifact_selection=selected_v3_oof_artifact_root is not None or active_score_artifact_selection is not None,
        reject_invalidated_decision_scorefields=True,
        fail_on_missing_oof_provenance=True,
        fail_on_invalid_metric_denominator=True,
    )
    foundation_recheck = {
        "layer_name": "FOUNDATION_INTEGRITY_RECHECK_AFTER_FIX_V1",
        "foundation_integrity_recheck_output_dir_v1": str(foundation_recheck_dir),
        "foundation_integrity_recheck_summary_v1": foundation_summary,
        "historical_invalid_v3_artifact_root_v1": str(v3_oof_dir),
        "selected_decision_artifact_root_v1": str(selected_v3_oof_artifact_root) if selected_v3_oof_artifact_root is not None else None,
        "oof_scoreprovenance_blocker_removed_v1": provenance_validation["status_v1"] == "PASS",
        "metric_denominator_blocker_removed_v1": not metric_failures and int(foundation_summary.get("metric_contract_failure_count_v1", 0)) == 0,
        "feature_matrix_still_clean_v1": foundation_summary.get("feature_count_v1") == 97,
        "no_fallback_dummy_synthetic_v1": True,
    }

    if provenance_validation["status_v1"] != "PASS":
        decision = "RERUN_V3_OOF_SCORING_WITH_PROVENANCE_FIRST"
        next_action = "RERUN_V3_PARALLEL_REBUILD_WITH_OOF_PROVENANCE_EXPLICIT_FLAG"
    elif metric_failures or int(foundation_summary.get("metric_contract_failure_count_v1", 0)) > 0:
        decision = "FIX_METRIC_DENOMINATOR_GUARDS_FIRST"
        next_action = "FIX_METRIC_DENOMINATOR_GUARDS_FIRST"
    elif foundation_summary.get("foundation_clean_for_constrained_optuna_v1") is True or foundation_summary.get("foundation_clean_ready_for_optuna_v1") is True:
        decision = "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA"
        next_action = "INSTALL_OPTUNA_AND_RUN_CONSTRAINED_OBJECTIVE_SEARCH"
    else:
        decision = str(foundation_summary.get("decision_v1", "NOT_ESTABLISHED"))
        next_action = decision

    go_no_go = {
        "layer_name": "GO_NO_GO_BEFORE_OPTUNA_V1",
        "decision_v1": decision,
        "foundation_clean_ready_for_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
        "foundation_clean_for_constrained_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
        "provenance_validation_status_v1": provenance_validation["status_v1"],
        "metric_guard_pass_v1": not metric_failures,
    }
    next_lock = {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "next_action_v1": next_action,
        "blocked_actions_v1": ["RUN_OPTUNA_NOW", "RUN_R6_NOW", "USE_EXISTING_V3_OOF_WITHOUT_PROVENANCE"],
    }
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "decision_v1": decision,
        "next_action_v1": next_action,
        "v3_oof_provenance_status_v1": provenance_validation["status_v1"],
        "existing_v3_oof_scores_valid_for_optuna_v1": provenance_validation["status_v1"] == "PASS",
        "historical_invalid_v3_artifacts_status_v1": (
            "QUARANTINED_NOT_SELECTED_HISTORY_ONLY" if selected_v3_oof_artifact_root is not None else "ACTIVE_BLOCKER_UNTIL_NEW_SELECTED_ROOT_EXISTS"
        ),
        "selected_decision_artifact_root_v1": str(selected_v3_oof_artifact_root) if selected_v3_oof_artifact_root is not None else None,
        "metric_denominator_guard_status_v1": "PASS" if not metric_failures else "FAIL",
        "foundation_recheck_decision_v1": foundation_summary.get("decision_v1"),
        "foundation_clean_ready_for_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
        "foundation_clean_for_constrained_optuna_v1": decision == "FOUNDATION_CLEAN_FOR_CONSTRAINED_OPTUNA",
        "training_started_v1": False,
        "optuna_started_v1": False,
        "r6_started_v1": False,
        "hard_status_v1": "BEVIST",
    }
    manifest = {
        "layer_name": f"{LAYER_NAME}_MANIFEST",
        "output_files_v1": {name: str(output_dir / name) for name in OUTPUT_FILES},
        "input_dirs_v1": {"v3_oof_dir_v1": str(v3_oof_dir), "v3_in_sample_dir_v1": str(v3_in_sample_dir)},
    }
    audit_rows = [
        {"check_v1": "no_optuna", "status_v1": "PASS", "evidence_v1": False},
        {"check_v1": "no_training", "status_v1": "PASS", "evidence_v1": False},
        {"check_v1": "metric_guard", "status_v1": "PASS" if not metric_failures else "FAIL", "evidence_v1": metric_failures},
        {"check_v1": "provenance_validation", "status_v1": "PASS" if provenance_validation["status_v1"] == "PASS" else "FAIL", "evidence_v1": provenance_validation["status_v1"]},
        {"check_v1": "foundation_recheck", "status_v1": "PASS" if decision == "FOUNDATION_CLEAN_READY_FOR_OPTUNA" else "BLOCKED", "evidence_v1": foundation_summary.get("decision_v1")},
    ]

    _write_json(output_dir / "v3_oof_score_provenance_contract_v1.json", contract)
    _write_json(output_dir / "v3_oof_score_provenance_reconstruction_or_invalidation_v1.json", reconstruction)
    provenance.to_csv(output_dir / "v3_oof_score_provenance_v1.csv", index=False)
    fold_assignment.to_csv(output_dir / "v3_oof_fold_assignment_v1.csv", index=False)
    membership.to_csv(output_dir / "v3_train_validation_membership_v1.csv", index=False)
    _write_json(output_dir / "v3_oof_score_source_manifest_v1.json", source_manifest)
    _write_json(output_dir / "oof_score_provenance_validation_v1.json", provenance_validation)
    _write_json(output_dir / "metric_denominator_guard_contract_v1.json", metric_contract)
    metric_audit.to_csv(output_dir / "metric_denominator_guard_audit_v1.csv", index=False)
    _write_json(output_dir / "foundation_integrity_recheck_after_fix_v1.json", foundation_recheck)
    _write_json(output_dir / "go_no_go_before_optuna_v1.json", go_no_go)
    _write_json(output_dir / "next_action_lock_v1.json", next_lock)
    _write_json(output_dir / "summary_v1.json", summary)
    _write_json(output_dir / "status_v1.json", {**summary, "status_v1": decision})
    _write_json(output_dir / "manifest_v1.json", manifest)
    _write_csv(output_dir / "consistency_audit_v1.csv", audit_rows)
    _write_report(output_dir, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--v3-oof-dir", type=Path, default=DEFAULT_V3_OOF_DIR)
    parser.add_argument("--v3-in-sample-dir", type=Path, default=DEFAULT_V3_IN_SAMPLE_DIR)
    parser.add_argument("--optuna-dir", type=Path, default=DEFAULT_OPTUNA_DIR)
    parser.add_argument("--selected-v3-oof-artifact-root", type=Path, default=None)
    parser.add_argument("--active-score-artifact-selection", type=Path, default=None)
    args = parser.parse_args(argv)
    materialize(
        reports_root=args.reports_root,
        output_dir=args.output_dir,
        v3_oof_dir=args.v3_oof_dir,
        v3_in_sample_dir=args.v3_in_sample_dir,
        optuna_dir=args.optuna_dir,
        selected_v3_oof_artifact_root=args.selected_v3_oof_artifact_root,
        active_score_artifact_selection=args.active_score_artifact_selection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
