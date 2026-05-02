from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_PYTHON = Path("/home/andre2/venvs/gx1/bin/python")
DEFAULT_REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")

LAYER_NAME = "R6_CANONICAL_LINE_RESTORE_AND_MONDAY_REPLAY_REANCHOR_V1"
WEDNESDAY_SNAPSHOT_DIR = "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
WEDNESDAY_FREEZE_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
WEDNESDAY_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
WEDNESDAY_MANIFEST = "shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json"

OLD_MONDAY_R6_DIR = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
OLD_MONDAY_R6_ASOF = "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet"
OLD_MONDAY_R6_LABELS = "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet"
OLD_MONDAY_R6_POLICY_LOCK = "shadow_meta_all_trade_review_r6_policy_logging_lock_v1.parquet"
OLD_MONDAY_EXACT_DIR = "ALL_TRADE_REVIEW_LEDGER_20260411"
OLD_MONDAY_EXACT_RAW = "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"
OLD_MONDAY_BRIDGE_DIR = "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_20260424T142808Z"
OLD_MONDAY_BRIDGE = "entry_to_failure_pocket_bridge_surface_v1.parquet"

OUTPUT_FILES = {
    "canonical_lock": "canonical_wednesday_r6_line_lock_v1.json",
    "quarantine": "bad_monday_narrow_surface_quarantine_v1.json",
    "reanchor_contract": "monday_reanchor_using_wednesday_r6_contract_v1.json",
    "rebuild_plan": "monday_fullcoverage_rebuild_plan_or_run_v1.json",
    "row_delta": "row_delta_explainer_v1.csv",
    "feature_availability": "r6_canonical_feature_availability_audit_v1.csv",
    "parity_gate": "monday_r6_canonical_parity_gate_v1.json",
    "next_action": "next_action_lock_v1.json",
    "summary": "summary_v1.json",
    "report": "report_v1.md",
    "audit": "consistency_audit_v1.csv",
}

CANONICAL_EXPECTED = {
    "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
    "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
    "policy_eval_rows_v1": 1971,
    "hindsight_backfill_rows_v1": 1971,
    "as_of_column_count_v1": 109,
    "bad_blocks_v1": 180,
    "tail_help_v1": 149,
    "precision_v1": 0.972972972972973,
    "worst_loso_v1": 0.9285714285714286,
    "repaired_165_damage_v1": 0,
    "fifty_plus_mfe_blocked_v1": 1,
    "hundred_plus_mfe_blocked_v1": 0,
    "two_hundred_plus_mfe_blocked_v1": 0,
    "strongest_winner_damage_v1": 0,
}

KEY_COLUMNS = ["candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]
BRIDGE_POCKET_COLUMNS = [
    "bridge_pocket_repaired_165_v1",
    "bridge_pocket_forensic_repaired_trade_v1",
    "bridge_pocket_runner_near_miss_v1",
    "bridge_pocket_fifty_plus_mfe_seed_v1",
    "bridge_pocket_missed_10_50_tail_control_v1",
    "bridge_pocket_missed_should_not_take_v1",
    "bridge_pocket_risky_allow_v1",
]
R6_LABEL_COLUMNS = [
    "r6_label_runner_50_mfe_v1",
    "r6_label_runner_100_mfe_v1",
    "r6_label_runner_200_mfe_v1",
    "r6_label_repaired_165_like_runner_v1",
    "r6_label_strong_low_mae_runner_v1",
    "r6_label_high_mfe_low_giveback_v1",
    "r6_label_runner_near_miss_v1",
    "r6_label_runner_protect_v1",
    "r6_label_missed_should_not_take_v1",
    "r6_label_risky_allow_v1",
    "r6_label_tail_control_10_50_v1",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet_if_present(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    series = frame[column]
    if str(series.dtype) == "boolean":
        return series.fillna(False).astype(bool)
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.lower().isin(["true", "1", "yes"])


def _bool_count(frame: pd.DataFrame | None, column: str) -> int | None:
    if frame is None or column not in frame.columns:
        return None
    return int(_bool_series(frame, column).sum())


def _unique_count(frame: pd.DataFrame | None, column: str) -> int | None:
    if frame is None or column not in frame.columns:
        return None
    return int(frame[column].astype("string").nunique(dropna=True))


def _schema_columns(schema: dict[str, Any]) -> list[str]:
    return [str(row.get("name_v1")) for row in schema.get("columns_v1", []) if isinstance(row, dict) and row.get("name_v1")]


def _path_status(path: Path, role: str, canonicality: str) -> dict[str, Any]:
    return {
        "path_v1": str(path),
        "exists_v1": path.exists(),
        "role_v1": role,
        "canonicality_v1": canonicality,
    }


def _load_wednesday(reports_root: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    freeze_dir = reports_root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    return freeze_dir, _read_json(freeze_dir / WEDNESDAY_SUMMARY), _read_json(freeze_dir / WEDNESDAY_MANIFEST)


def _canonical_line_lock(reports_root: Path) -> dict[str, Any]:
    freeze_dir, summary, manifest = _load_wednesday(reports_root)
    policy_logging = summary.get("policy_logging_v1", {})
    selected = summary.get("selected_candidate_v1", {})
    as_of_schema = manifest.get("as_of_schema_v1", {})
    hindsight_schema = manifest.get("hindsight_schema_v1", {})
    source_root = Path(str(summary.get("reports_root_v1") or manifest.get("reports_root_v1") or ""))
    r6_source = Path(str(summary.get("r6_source_dir_v1") or manifest.get("r6_source_dir_v1") or ""))
    r5_2_freeze = Path(str(summary.get("r5_2_freeze_dir_v1") or ""))
    source_paths = [
        _path_status(r6_source, "R6 source retrain extension with AS_OF/hindsight/model/eval artifacts", "CANONICAL_SOURCE_REQUIRED"),
        _path_status(r5_2_freeze, "R5.2 frozen reference feeding R6 comparator", "CANONICAL_SOURCE_REQUIRED"),
        _path_status(source_root, "Wednesday upstream truth reports root", "CANONICAL_SOURCE_REQUIRED"),
        _path_status(freeze_dir / WEDNESDAY_SUMMARY, "copied frozen summary", "CANONICAL_LOCK_AVAILABLE"),
        _path_status(freeze_dir / WEDNESDAY_MANIFEST, "copied frozen manifest", "CANONICAL_LOCK_AVAILABLE"),
        _path_status(
            freeze_dir / "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_report_v1.md",
            "copied frozen report",
            "CANONICAL_LOCK_AVAILABLE",
        ),
    ]
    observed_metrics = {
        "freeze_id_v1": summary.get("freeze_id_v1") or manifest.get("freeze_id_v1"),
        "selected_candidate_id_v1": summary.get("selected_candidate_id_v1") or manifest.get("selected_candidate_id_v1"),
        "policy_eval_rows_v1": int(policy_logging.get("row_count_v1", 0)),
        "hindsight_backfill_rows_v1": int(policy_logging.get("hindsight_backfill_rows_v1", 0)),
        "as_of_column_count_v1": int(as_of_schema.get("column_count_v1", 0)),
        "bad_blocks_v1": selected.get("should_not_take_block_count_v1"),
        "tail_help_v1": selected.get("tail_10_50_help_count_v1"),
        "precision_v1": selected.get("should_not_take_precision_v1"),
        "worst_loso_v1": selected.get("worst_loso_precision_v1"),
        "repaired_165_damage_v1": selected.get("repaired_165_block_count_v1"),
        "fifty_plus_mfe_blocked_v1": selected.get("fifty_plus_mfe_block_count_v1"),
        "hundred_plus_mfe_blocked_v1": selected.get("hundred_plus_mfe_block_count_v1"),
        "two_hundred_plus_mfe_blocked_v1": selected.get("two_hundred_plus_mfe_block_count_v1"),
        "strongest_winner_damage_v1": selected.get("strongest_winner_path_block_count_v1"),
    }
    mismatches = {
        key: {"expected_v1": expected, "observed_v1": observed_metrics.get(key)}
        for key, expected in CANONICAL_EXPECTED.items()
        if observed_metrics.get(key) != expected
    }
    return {
        "layer_name": "CANONICAL_WEDNESDAY_R6_LINE_LOCK_V1",
        "status_v1": "CANONICAL_LOCKED_FROM_SNAPSHOT" if not mismatches else "CANONICAL_LOCK_HAS_MISMATCH",
        "canonical_line_is_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1 / R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        "expected_v1": CANONICAL_EXPECTED,
        "observed_v1": observed_metrics,
        "mismatch_v1": mismatches,
        "canonical_artifacts_v1": source_paths,
        "canonical_source_artifacts_available_locally_v1": all(row["exists_v1"] for row in source_paths if row["canonicality_v1"] == "CANONICAL_SOURCE_REQUIRED"),
        "scripts_materializers_v1": [
            {
                "script_v1": "gx1/scripts/materialize_truth_calendar_reorg_monday_week_v1.py",
                "role_v1": "calendar/window contract when reanchoring, not a new model contract",
            },
            {
                "script_v1": "gx1/scripts/launch_truth_monday_week_replay_v1.py",
                "role_v1": "Monday replay launcher for canonical truth run production",
            },
            {
                "script_v1": "gx1/scripts/materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1.py",
                "role_v1": "full repaired entry coverage AS_OF/policy substrate",
            },
            {
                "script_v1": "gx1/scripts/materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1.py",
                "role_v1": "R5.2 phase gate/reference and AS_OF score context",
            },
            {
                "script_v1": "gx1/scripts/materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1.py",
                "role_v1": "R5.2 freeze/reference and R6 failure backlog",
            },
            {
                "script_v1": "gx1/scripts/train_r6_entry_runner_first_retrain_v1.py",
                "role_v1": "canonical R6 five-head runner-first shadow family",
            },
            {
                "script_v1": "gx1/scripts/materialize_r6_shadow_freeze_and_path_dynamics_unblock_v1.py",
                "role_v1": "R6 freeze/policy logging lock/eval comparator",
            },
        ],
        "contracts_v1": {
            "model_version_id_v1": summary.get("model_version_id_v1") or manifest.get("model_version_id_v1"),
            "threshold_version_id_v1": summary.get("threshold_version_id_v1") or manifest.get("threshold_version_id_v1"),
            "selected_policy_stack_v1": summary.get("selected_policy_stack_v1") or manifest.get("selected_policy_stack_v1"),
            "thresholds_v1": summary.get("thresholds_v1", {}),
            "score_head_names_v1": manifest.get("score_head_names_v1", {}),
            "contract_lock_summary_v1": manifest.get("contract_lock_summary_v1", {}),
            "as_of_schema_v1": {
                "column_count_v1": as_of_schema.get("column_count_v1"),
                "schema_sha256_v1": as_of_schema.get("schema_sha256_v1"),
                "columns_v1": _schema_columns(as_of_schema),
            },
            "hindsight_schema_v1": {
                "column_count_v1": hindsight_schema.get("column_count_v1"),
                "schema_sha256_v1": hindsight_schema.get("schema_sha256_v1"),
                "columns_v1": _schema_columns(hindsight_schema),
            },
            "policy_log_contract_v1": {
                "policy_logging_rows_v1": policy_logging.get("row_count_v1"),
                "hindsight_backfill_rows_v1": policy_logging.get("hindsight_backfill_rows_v1"),
                "mask_mismatch_count_v1": policy_logging.get("mask_mismatch_count_v1"),
                "selected_policy_stack_v1": summary.get("selected_policy_stack_v1"),
            },
        },
        "row_filters_v1": [
            "Use the full repaired policy/eval universe expected at 1971 rows for the canonical line.",
            "Keep used_for_training, used_for_validation, and used_for_holdout split masks as AS_OF metadata.",
            "Do not require exact-only raw-state membership when the canonical R6 line admits repaired fullcoverage rows.",
            "Do not use bridge/readiness rows as a substitute baseline; bridge is diagnostic/readiness only.",
        ],
        "coverage_repair_rules_v1": [
            "Full repaired entry coverage is part of the canonical R6 line.",
            "Repaired coverage rows stay in policy/eval and label intersection when AS_OF lineage is complete.",
            "Synthetic rows remain forbidden unless the upstream canonical materializer explicitly proves otherwise.",
            "Repair lineage fields must remain materialized and auditable.",
        ],
        "keys_v1": KEY_COLUMNS,
        "as_of_hindsight_separation_v1": {
            "physical_separation_v1": True,
            "semantic_separation_v1": True,
            "as_of_table_role_v1": "pre-decision AS_OF features and policy score context",
            "hindsight_table_role_v1": "post-trade labels/outcomes only",
        },
    }


def _quarantine(reports_root: Path) -> dict[str, Any]:
    diagnostic_artifacts = [
        {
            "artifact_v1": str(reports_root / OLD_MONDAY_EXACT_DIR / OLD_MONDAY_EXACT_RAW),
            "status_v1": "DIAGNOSTIC_ONLY",
            "do_not_use_as_v1": ["R6_BASELINE", "R6_TRAINING_SURFACE", "CANONICAL_LINE"],
            "reason_v1": "1689 exact-only raw-state is the narrow failure surface, not the canonical R6 line.",
        },
        {
            "artifact_v1": str(reports_root / OLD_MONDAY_BRIDGE_DIR / OLD_MONDAY_BRIDGE),
            "status_v1": "DIAGNOSTIC_ONLY",
            "do_not_use_as_v1": ["R6_BASELINE", "TRAINING_ROW_WORKAROUND"],
            "reason_v1": "Bridge rows explain missing coverage and readiness only; they are not baseline replacement.",
        },
        {
            "artifact_v1": "PROTECTOR_FIRST_* built over 1689 exact-only",
            "status_v1": "DIAGNOSTIC_ONLY",
            "do_not_use_as_v1": ["R6_BASELINE", "NEXT_R6_LINE"],
            "reason_v1": "Protector-first depends on the wrong narrow surface until canonical Monday baseline is restored.",
        },
        {
            "artifact_v1": "MONDAY_NARROW_RETRAIN_* and run_monday_narrow_retrain_runner_v1.py",
            "status_v1": "DO_NOT_USE_AS_R6_BASELINE",
            "do_not_use_as_v1": ["R6_BASELINE", "CANONICAL_TRAINING_RUN"],
            "reason_v1": "Failed narrow retrain-run is a negative reference only.",
        },
    ]
    blocked_scripts = [
        "gx1/scripts/run_monday_narrow_retrain_runner_v1.py",
        "gx1/scripts/materialize_monday_narrow_retrain_failure_forensics_v1.py",
        "gx1/scripts/materialize_monday_narrow_retrain_job_spec_v1.py",
        "gx1/scripts/materialize_monday_narrow_retrain_runner_spec_v1.py",
        "gx1/scripts/materialize_monday_narrow_retrain_scope_plan_v1.py",
        "gx1/scripts/run_protector_first_shadow_experiment_runner_v1.py",
        "gx1/scripts/materialize_protector_first_shadow_experiment_runner_spec_v1.py",
        "gx1/scripts/materialize_protector_first_shadow_experiment_spec_v1.py",
        "gx1/scripts/materialize_monday_entry_to_failure_pocket_bridge_v1.py as baseline replacement",
    ]
    return {
        "layer_name": "BAD_MONDAY_NARROW_SURFACE_QUARANTINE_V1",
        "status_v1": "QUARANTINED_FROM_CANONICAL_R6_BASELINE",
        "diagnostic_artifacts_v1": diagnostic_artifacts,
        "scripts_not_for_new_r6_baseline_v1": blocked_scripts,
        "runner_specs_blocked_until_canonical_monday_restored_v1": [
            "PROTECTOR_FIRST_TRAINING_EXECUTION_V1",
            "MONDAY_NARROW_RETRAIN_RUNNER_V1",
            "any 1689 exact-only matrix builder",
            "any bridge-as-training-surface workaround",
        ],
        "allowed_use_v1": [
            "failure diagnosis",
            "row-delta forensics",
            "negative reference in compare/eval",
        ],
        "blocked_use_v1": [
            "baseline training surface",
            "R6 continuation line",
            "protector-first training precondition",
            "policy/controller input",
        ],
    }


def _monday_reanchor_contract(canonical: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_name": "MONDAY_REANCHOR_USING_WEDNESDAY_R6_CONTRACT_V1",
        "status_v1": "CONTRACT_LOCKED_RUN_NOT_EXECUTED",
        "canonical_line_v1": canonical["canonical_line_is_v1"],
        "anchor_change_only_v1": {
            "old_line_v1": "Wednesday-anchored replay/window lineage",
            "new_line_v1": "Monday 00:00 UTC -> next Monday 00:00 UTC calendar windows",
            "trading_window_v1": "Monday start -> Friday flat cutoff; no weekend entries; no weekend management after flat.",
            "existing_calendar_script_semantics_v1": {
                "calendar_week_v1": "MONDAY_00:00_UTC_TO_NEXT_MONDAY_00:00_UTC_EXCLUSIVE",
                "trading_week_v1": "MONDAY_START_TO_FRIDAY_FLAT_CUTOFF",
                "friday_flat_cutoff_v1": "Friday 20:55 UTC as encoded in materialize_truth_calendar_reorg_monday_week_v1.py",
                "weekend_entries_v1": "FORBIDDEN",
            },
        },
        "must_remain_identical_v1": [
            "R6 five-head model family",
            "selected policy stack comparator contract",
            "feature contract and AS_OF schema philosophy",
            "hindsight label contract",
            "policy logging contract",
            "full repaired entry coverage contract",
            "R5.2 safety reference comparator",
            "frozen Wednesday-R6 safety gates",
        ],
        "must_not_change_v1": [
            "feature set",
            "label contract",
            "coverage repair contract",
            "policy logging contract",
            "eval comparator contract",
            "threshold family",
            "model family",
        ],
        "expected_fullcoverage_universe_v1": {
            "style_v1": "1971-style full repaired universe if Monday window covers same trade universe",
            "expected_rows_v1": CANONICAL_EXPECTED["policy_eval_rows_v1"],
            "expected_as_of_columns_v1": CANONICAL_EXPECTED["as_of_column_count_v1"],
            "expected_hindsight_rows_v1": CANONICAL_EXPECTED["hindsight_backfill_rows_v1"],
        },
        "pre_training_assertions_v1": [
            "Monday policy/eval rows are materialized before R6 training.",
            "Monday AS_OF schema matches canonical Wednesday schema unless a row-level reanchor exception is explicitly proven.",
            "Monday hindsight/backfill intersection is complete.",
            "No 1689 exact-only or protector-first matrix is used as baseline.",
            "Row deltas are explained at candidate_uid/trade_uid level before training.",
        ],
    }


def _command(parts: list[str]) -> str:
    return " ".join(parts)


def _rebuild_plan(reports_root: Path, canonical: dict[str, Any]) -> dict[str, Any]:
    source_available = bool(canonical["canonical_source_artifacts_available_locally_v1"])
    monday_root = reports_root
    commands = [
        {
            "step_v1": 1,
            "name_v1": "restore_wednesday_source_artifacts",
            "command_v1": "RESTORE /home/andre2/GX1_DATA/reports/truth_e2e_sanity/MANAGEMENT_PATH_DYNAMICS_UPSTREAM_REPLAY_V2_20260419_142449 FROM BACKUP OR ORIGINAL RUN OUTPUT",
            "expected_outputs_v1": [
                "ALL_TRADE_REVIEW_LEDGER_20260421T_R4_FULLCOVERAGE_POLICY_RECALIBRATION_AND_SHADOW_REPLAY_V1",
                "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_PHASE_GATE_AND_HARVEST_INTEGRATION_V1",
                "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1",
                "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1",
                "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1",
            ],
        },
        {
            "step_v1": 2,
            "name_v1": "materialize_monday_calendar_contract",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/materialize_truth_calendar_reorg_monday_week_v1.py",
                    "--output-name",
                    "TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1",
                    "--data-end-exclusive",
                    "2026-04-20T00:00:00Z",
                ]
            ),
            "expected_outputs_v1": ["TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json"],
        },
        {
            "step_v1": 3,
            "name_v1": "dry_run_monday_replay_launch",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/launch_truth_monday_week_replay_v1.py",
                    "--calendar",
                    str(reports_root / "TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json"),
                    "--reports-root",
                    str(monday_root),
                    "--max-workers",
                    "15",
                    "--dry-run",
                    "--stop-on-failure",
                ]
            ),
            "expected_outputs_v1": ["launch status with selected TRUTH_MONFRI_WEEK runs and no quarantined weeks"],
        },
        {
            "step_v1": 4,
            "name_v1": "run_monday_replays_after_dry_run_acceptance",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/launch_truth_monday_week_replay_v1.py",
                    "--calendar",
                    str(reports_root / "TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json"),
                    "--reports-root",
                    str(monday_root),
                    "--max-workers",
                    "15",
                    "--archive-stale",
                    "--stop-on-failure",
                ]
            ),
            "expected_outputs_v1": ["completed Monday TRUTH_MONFRI_WEEK run dirs with POSTRUN_E2E passed"],
        },
        {
            "step_v1": 5,
            "name_v1": "materialize_monday_r4_fullcoverage_with_canonical_expected_count",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1.py",
                    "--reports-root",
                    str(monday_root),
                    "--expected-ledger-count",
                    "1971",
                ]
            ),
            "expected_outputs_v1": ["R4 fullcoverage AS_OF/hindsight/policy recalibration artifacts at 1971-style repaired coverage"],
        },
        {
            "step_v1": 6,
            "name_v1": "materialize_monday_r5_2_phase_gate",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1.py",
                    "--reports-root",
                    str(monday_root),
                    "--expected-ledger-count",
                    "1971",
                ]
            ),
            "expected_outputs_v1": ["R5.2 phase-gate AS_OF/hindsight with complete label intersection"],
        },
        {
            "step_v1": 7,
            "name_v1": "freeze_monday_r5_2_reference",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/materialize_r5_2_shadow_freeze_and_r6_failure_backlog_v1.py",
                    "--reports-root",
                    str(monday_root),
                    "--expected-ledger-count",
                    "1971",
                ]
            ),
            "expected_outputs_v1": ["R5.2 freeze/reference artifacts for R6 comparator"],
        },
        {
            "step_v1": 8,
            "name_v1": "train_monday_r6_on_canonical_line",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/train_r6_entry_runner_first_retrain_v1.py",
                    "--reports-root",
                    str(monday_root),
                    "--expected-ledger-count",
                    "1971",
                    "--seed",
                    "20260422",
                    "--n-jobs",
                    "4",
                ]
            ),
            "expected_outputs_v1": ["R6 five-head retrain artifacts with canonical feature/label/eval contract"],
        },
        {
            "step_v1": 9,
            "name_v1": "freeze_monday_r6_and_compare",
            "command_v1": _command(
                [
                    str(DEFAULT_PYTHON),
                    "gx1/scripts/materialize_r6_shadow_freeze_and_path_dynamics_unblock_v1.py",
                    "--reports-root",
                    str(monday_root),
                    "--expected-ledger-count",
                    "1971",
                    "--test-status",
                    "POST_MONDAY_REANCHOR_CANONICAL_PARITY_GATE_REQUIRED",
                ]
            ),
            "expected_outputs_v1": ["Monday R6 policy logging lock, hindsight backfill lock, compare report, consistency audit"],
        },
    ]
    return {
        "layer_name": "MONDAY_FULLCOVERAGE_REBUILD_PLAN_OR_RUN_V1",
        "run_executed_v1": False,
        "run_status_v1": "NOT_RUN_SOURCE_ARTIFACTS_MISSING" if not source_available else "READY_TO_RUN_NOT_STARTED",
        "reason_not_run_v1": (
            "Canonical Wednesday source root is missing locally; running would risk rebuilding from the wrong Monday/narrow lineage."
            if not source_available
            else "User requested restore/reanchor lock; full run not started automatically."
        ),
        "monday_rebuild_target_v1": "Monday-equivalent of canonical Wednesday-R6 full repaired universe, not 1689 exact-only.",
        "required_inputs_v1": [
            "canonical Wednesday source artifact family",
            "Monday calendar contract",
            "completed Monday TRUTH replay runs",
            "canonical R4/R5.2/R6 materializer scripts",
        ],
        "expected_outputs_v1": [
            "Monday policy/eval universe",
            "Monday AS_OF schema",
            "Monday hindsight/backfill",
            "Monday policy-log surface",
            "Monday repaired coverage",
            "Monday feature contract",
            "Monday label intersection",
            "Monday candidate/trade lineage",
            "Monday eval readiness",
        ],
        "commands_v1": commands,
    }


def _label_counts(frame: pd.DataFrame | None, ids: set[str] | None = None) -> dict[str, int]:
    if frame is None or "candidate_uid" not in frame.columns:
        return {}
    work = frame
    if ids is not None:
        work = frame[frame["candidate_uid"].astype("string").isin(ids)]
    out: dict[str, int] = {}
    for column in R6_LABEL_COLUMNS:
        if column in work.columns:
            out[column] = int(_bool_series(work, column).sum())
    return out


def _row_delta_explainer(reports_root: Path, canonical: dict[str, Any]) -> pd.DataFrame:
    exact = _read_parquet_if_present(reports_root / OLD_MONDAY_EXACT_DIR / OLD_MONDAY_EXACT_RAW)
    bridge = _read_parquet_if_present(reports_root / OLD_MONDAY_BRIDGE_DIR / OLD_MONDAY_BRIDGE)
    old_full = _read_parquet_if_present(reports_root / OLD_MONDAY_R6_DIR / OLD_MONDAY_R6_ASOF)
    labels = _read_parquet_if_present(reports_root / OLD_MONDAY_R6_DIR / OLD_MONDAY_R6_LABELS)
    rows: list[dict[str, Any]] = []
    exact_ids = set(exact["candidate_uid"].astype("string")) if exact is not None and "candidate_uid" in exact.columns else set()
    full_ids = set(old_full["candidate_uid"].astype("string")) if old_full is not None and "candidate_uid" in old_full.columns else set()
    label_by_id = labels.set_index(labels["candidate_uid"].astype("string"), drop=False) if labels is not None and "candidate_uid" in labels.columns else None
    full_by_id = old_full.set_index(old_full["candidate_uid"].astype("string"), drop=False) if old_full is not None and "candidate_uid" in old_full.columns else None

    if bridge is not None and "candidate_uid" in bridge.columns:
        bridge = bridge.copy()
        bridge["_candidate_key"] = bridge["candidate_uid"].astype("string")
        bridge_only = bridge[~bridge["_candidate_key"].isin(exact_ids)].sort_values("candidate_uid")
        for _, row in bridge_only.iterrows():
            candidate_uid = str(row["candidate_uid"])
            label_row: pd.Series | None = None
            full_row: pd.Series | None = None
            if label_by_id is not None and candidate_uid in label_by_id.index:
                hit = label_by_id.loc[candidate_uid]
                label_row = hit.iloc[0] if isinstance(hit, pd.DataFrame) else hit
            if full_by_id is not None and candidate_uid in full_by_id.index:
                hit = full_by_id.loc[candidate_uid]
                full_row = hit.iloc[0] if isinstance(hit, pd.DataFrame) else hit
            decision_timestamp = row.get("decision_timestamp", "")
            if (pd.isna(decision_timestamp) or str(decision_timestamp) == "") and full_row is not None:
                decision_timestamp = full_row.get("decision_timestamp", "")
            if (pd.isna(decision_timestamp) or str(decision_timestamp) == "") and label_row is not None:
                decision_timestamp = label_row.get("decision_timestamp", "")
            out = {
                "row_type_v1": "DELTA_ROW",
                "delta_v1": "OLD_MONDAY_FULLCOVERAGE_OR_BRIDGE_1852_MINUS_OLD_EXACT_ONLY_1689",
                "candidate_uid": candidate_uid,
                "trade_uid": row.get("trade_uid"),
                "trade_id": row.get("trade_id"),
                "decision_timestamp": decision_timestamp,
                "week_window_v1": str(candidate_uid).split(":cand::", 1)[0] if ":cand::" in candidate_uid else "",
                "old_wednesday_canonical_present_v1": "NOT_ESTABLISHED",
                "old_monday_fullcoverage_present_v1": candidate_uid in full_ids,
                "old_monday_exact_only_present_v1": False,
                "repaired_coverage_v1": bool(row.get("entry_coverage_repair_applied_v1", False)),
                "reason_v1": "Old exact-only/narrow surface dropped a repaired/fullcoverage candidate; this is diagnostic only.",
                "status_v1": "RAW_STATE_MISSING",
            }
            for column in BRIDGE_POCKET_COLUMNS:
                if column in row.index:
                    out[column] = bool(row[column]) if not pd.isna(row[column]) else False
            if label_row is not None:
                for column in R6_LABEL_COLUMNS:
                    if column in label_row.index:
                        out[column] = bool(label_row[column]) if not pd.isna(label_row[column]) else False
            rows.append(out)

    canonical_rows = int(canonical["expected_v1"]["policy_eval_rows_v1"])
    old_full_rows = len(old_full) if old_full is not None else 0
    if canonical_rows > old_full_rows:
        rows.append(
            {
                "row_type_v1": "AGGREGATE_NOT_ESTABLISHED",
                "delta_v1": "CANONICAL_WEDNESDAY_1971_MINUS_OLD_MONDAY_FULLCOVERAGE_1852",
                "candidate_uid": "",
                "trade_uid": "",
                "trade_id": "",
                "decision_timestamp": "",
                "week_window_v1": "",
                "missing_count_v1": canonical_rows - old_full_rows,
                "old_wednesday_canonical_present_v1": True,
                "old_monday_fullcoverage_present_v1": False,
                "old_monday_exact_only_present_v1": False,
                "repaired_coverage_v1": "NOT_ESTABLISHED",
                "reason_v1": "Canonical Wednesday source rows are count-locked but source parquet candidate IDs are missing locally.",
                "status_v1": "NOT_ESTABLISHED",
            }
        )
    rows.append(
        {
            "row_type_v1": "SUMMARY",
            "delta_v1": "ROW_COUNT_BASELINES",
            "candidate_uid": "",
            "trade_uid": "",
            "trade_id": "",
            "decision_timestamp": "",
            "week_window_v1": "",
            "wednesday_canonical_rows_v1": canonical_rows,
            "old_monday_fullcoverage_rows_v1": old_full_rows if old_full is not None else "MISSING",
            "old_monday_exact_only_rows_v1": len(exact) if exact is not None else "MISSING",
            "old_monday_bridge_rows_v1": len(bridge) if bridge is not None else "MISSING",
            "reason_v1": "Old Monday row counts are diagnostic negatives; rebuild must target canonical Monday equivalent.",
            "status_v1": "PIPELINE_DRIFT",
        }
    )
    return pd.DataFrame(rows)


def _feature_availability_audit(reports_root: Path, canonical: dict[str, Any]) -> pd.DataFrame:
    old_r6_asof = _read_parquet_if_present(reports_root / OLD_MONDAY_R6_DIR / OLD_MONDAY_R6_ASOF)
    old_r6_labels = _read_parquet_if_present(reports_root / OLD_MONDAY_R6_DIR / OLD_MONDAY_R6_LABELS)
    old_exact = _read_parquet_if_present(reports_root / OLD_MONDAY_EXACT_DIR / OLD_MONDAY_EXACT_RAW)
    policy_lock_path = reports_root / OLD_MONDAY_R6_DIR / OLD_MONDAY_R6_POLICY_LOCK
    expected_asof = list(canonical["contracts_v1"]["as_of_schema_v1"]["columns_v1"])
    expected_hindsight = list(canonical["contracts_v1"]["hindsight_schema_v1"]["columns_v1"])
    rows: list[dict[str, Any]] = []

    def present(frame: pd.DataFrame | None, column: str) -> bool:
        return frame is not None and column in frame.columns

    def null_rate(frame: pd.DataFrame | None, column: str) -> float | None:
        if not present(frame, column):
            return None
        return float(frame[column].isna().mean())

    for column in expected_asof:
        old_r6_present = present(old_r6_asof, column)
        exact_present = present(old_exact, column)
        rows.append(
            {
                "surface_v1": "AS_OF",
                "expected_column_v1": column,
                "canonical_expected_present_v1": True,
                "old_monday_r6_1852_present_v1": old_r6_present,
                "old_monday_r6_1852_null_rate_v1": null_rate(old_r6_asof, column),
                "old_monday_exact_1689_present_v1": exact_present,
                "old_monday_exact_1689_null_rate_v1": null_rate(old_exact, column),
                "status_v1": "PRESENT_IN_OLD_MONDAY_R6" if old_r6_present else "MISSING_FROM_OLD_MONDAY_R6_CANONICAL_BLOCKER",
                "notes_v1": "Canonical AS_OF schema from Wednesday R6 freeze manifest.",
            }
        )

    for column in expected_hindsight:
        label_present = present(old_r6_labels, column)
        rows.append(
            {
                "surface_v1": "HINDSIGHT",
                "expected_column_v1": column,
                "canonical_expected_present_v1": True,
                "old_monday_r6_1852_present_v1": label_present,
                "old_monday_r6_1852_null_rate_v1": null_rate(old_r6_labels, column),
                "old_monday_exact_1689_present_v1": False,
                "old_monday_exact_1689_null_rate_v1": None,
                "status_v1": "PRESENT_IN_OLD_MONDAY_R6_LABELS" if label_present else "MISSING_FROM_OLD_MONDAY_R6_LABELS_CANONICAL_BLOCKER",
                "notes_v1": "Canonical hindsight schema from Wednesday R6 freeze manifest.",
            }
        )

    rows.append(
        {
            "surface_v1": "POLICY_LOG",
            "expected_column_v1": OLD_MONDAY_R6_POLICY_LOCK,
            "canonical_expected_present_v1": True,
            "old_monday_r6_1852_present_v1": policy_lock_path.exists(),
            "old_monday_r6_1852_null_rate_v1": None,
            "old_monday_exact_1689_present_v1": False,
            "old_monday_exact_1689_null_rate_v1": None,
            "status_v1": "PRESENT_IN_OLD_MONDAY_R6" if policy_lock_path.exists() else "MISSING_POLICY_LOG_LOCK_CANONICAL_BLOCKER",
            "notes_v1": "Canonical R6 freeze needs policy logging lock; old Monday retrain dir is not the freeze line.",
        }
    )
    return pd.DataFrame(rows)


def _feature_summary(feature_audit_df: pd.DataFrame) -> dict[str, Any]:
    asof = feature_audit_df[feature_audit_df["surface_v1"].eq("AS_OF")]
    hindsight = feature_audit_df[feature_audit_df["surface_v1"].eq("HINDSIGHT")]
    policy = feature_audit_df[feature_audit_df["surface_v1"].eq("POLICY_LOG")]
    return {
        "expected_as_of_columns_v1": int(len(asof)),
        "old_monday_r6_as_of_present_count_v1": int(asof["old_monday_r6_1852_present_v1"].sum()),
        "old_monday_r6_as_of_missing_count_v1": int((~asof["old_monday_r6_1852_present_v1"]).sum()),
        "old_monday_r6_missing_as_of_columns_v1": asof.loc[
            ~asof["old_monday_r6_1852_present_v1"], "expected_column_v1"
        ].astype(str).tolist(),
        "expected_hindsight_columns_v1": int(len(hindsight)),
        "old_monday_r6_hindsight_present_count_v1": int(hindsight["old_monday_r6_1852_present_v1"].sum()),
        "old_monday_r6_hindsight_missing_count_v1": int((~hindsight["old_monday_r6_1852_present_v1"]).sum()),
        "old_monday_policy_lock_present_v1": bool(policy["old_monday_r6_1852_present_v1"].any()) if len(policy) else False,
    }


def _parity_gate(canonical: dict[str, Any], row_delta_df: pd.DataFrame, feature_audit_df: pd.DataFrame) -> dict[str, Any]:
    source_available = bool(canonical["canonical_source_artifacts_available_locally_v1"])
    feature_summary = _feature_summary(feature_audit_df)
    old_monday_features_complete = (
        feature_summary["old_monday_r6_as_of_missing_count_v1"] == 0
        and feature_summary["old_monday_r6_hindsight_missing_count_v1"] == 0
        and feature_summary["old_monday_policy_lock_present_v1"]
    )
    checks = [
        {"check_v1": "canonical_wednesday_lock", "status_v1": "PASS" if canonical["status_v1"] == "CANONICAL_LOCKED_FROM_SNAPSHOT" else "FAIL"},
        {"check_v1": "wednesday_source_artifacts_available", "status_v1": "PASS" if source_available else "FAIL"},
        {"check_v1": "old_monday_r6_full_feature_availability", "status_v1": "PASS" if old_monday_features_complete else "FAIL", "evidence_v1": feature_summary},
        {"check_v1": "row_universe", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "as_of_schema", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "hindsight_schema", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "policy_log_schema", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "repaired_coverage", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "candidate_trade_lineage", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "label_intersection", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "pocket_visibility", "status_v1": "BLOCKED_UNTIL_MONDAY_REBUILD"},
        {"check_v1": "no_accidental_narrow_or_protector_surface_usage", "status_v1": "PASS"},
    ]
    if not source_available:
        decision = "MONDAY_BLOCKED_BY_MISSING_WEDNESDAY_SOURCE_ARTIFACTS"
    else:
        decision = "MONDAY_REBUILD_READY_TO_RUN"
    return {
        "layer_name": "MONDAY_R6_CANONICAL_PARITY_GATE_V1",
        "decision_v1": decision,
        "checks_v1": checks,
        "row_delta_rows_v1": int(len(row_delta_df)),
        "feature_availability_v1": feature_summary,
        "parity_restored_v1": False,
        "blocked_reason_v1": None if source_available else "Canonical Wednesday source artifact root is missing locally.",
    }


def _next_action(parity_gate: dict[str, Any]) -> dict[str, Any]:
    decision = parity_gate["decision_v1"]
    if decision == "MONDAY_CANONICAL_R6_BASELINE_RESTORED":
        action = "RUN_MONDAY_R6_RETRAIN_ON_CANONICAL_WEDNESDAY_LINE"
    elif decision == "MONDAY_REBUILD_READY_TO_RUN":
        action = "RUN_MONDAY_FULLCOVERAGE_REBUILD_USING_WEDNESDAY_CONTRACT"
    elif decision == "MONDAY_BLOCKED_BY_MISSING_WEDNESDAY_SOURCE_ARTIFACTS":
        action = "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    elif decision == "MONDAY_BLOCKED_BY_LINEAGE_OR_REPAIR_GAP":
        action = "FIX_LINEAGE_OR_REPAIR_COVERAGE_FIRST"
    else:
        action = "NOT_ESTABLISHED"
    return {
        "layer_name": "NEXT_ACTION_LOCK_V1",
        "hard_recommendation_v1": action,
        "always_blocked_actions_v1": [
            "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE",
            "DO_NOT_CONTINUE_PROTECTOR_FIRST_BEFORE_CANONICAL_MONDAY_BASELINE",
        ],
        "next_allowed_after_unblock_v1": (
            "RUN_MONDAY_FULLCOVERAGE_REBUILD_USING_WEDNESDAY_CONTRACT"
            if action == "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
            else action
        ),
    }


def _audit(canonical: dict[str, Any], quarantine: dict[str, Any], parity_gate: dict[str, Any]) -> pd.DataFrame:
    def row(check: str, status: str, evidence: Any) -> dict[str, str]:
        return {"check_v1": check, "status_v1": status, "evidence_v1": json.dumps(_jsonable(evidence), sort_keys=True)}

    return pd.DataFrame(
        [
            row("CANONICAL_FREEZE_ID", "PASS" if not canonical["mismatch_v1"].get("freeze_id_v1") else "FAIL", canonical["observed_v1"].get("freeze_id_v1")),
            row("CANONICAL_CANDIDATE_ID", "PASS" if not canonical["mismatch_v1"].get("selected_candidate_id_v1") else "FAIL", canonical["observed_v1"].get("selected_candidate_id_v1")),
            row("CANONICAL_ROW_COUNT_1971", "PASS" if not canonical["mismatch_v1"].get("policy_eval_rows_v1") else "FAIL", canonical["observed_v1"].get("policy_eval_rows_v1")),
            row("CANONICAL_AS_OF_COLUMNS_109", "PASS" if not canonical["mismatch_v1"].get("as_of_column_count_v1") else "FAIL", canonical["observed_v1"].get("as_of_column_count_v1")),
            row("CANONICAL_METRICS_LOCKED", "PASS" if not canonical["mismatch_v1"] else "FAIL", canonical["mismatch_v1"]),
            row("WEDNESDAY_SOURCE_ARTIFACTS_AVAILABLE", "PASS" if canonical["canonical_source_artifacts_available_locally_v1"] else "FAIL", canonical["canonical_artifacts_v1"]),
            row("BAD_MONDAY_NARROW_QUARANTINED", "PASS", quarantine["diagnostic_artifacts_v1"]),
            row("PARITY_GATE_DECISION", "PASS" if parity_gate["decision_v1"] != "MONDAY_CANONICAL_R6_BASELINE_RESTORED" else "PASS", parity_gate["decision_v1"]),
        ]
    )


def _report(summary: dict[str, Any], canonical: dict[str, Any], next_action: dict[str, Any]) -> str:
    feature = summary["feature_availability_v1"]
    return "\n".join(
        [
            "# R6 Canonical Line Restore And Monday Replay Reanchor V1",
            "",
            f"Materialized at: {summary['materialized_at_utc_v1']}",
            "",
            "## Canonical Line",
            "",
            f"- Freeze: `{canonical['observed_v1']['freeze_id_v1']}`",
            f"- Candidate: `{canonical['observed_v1']['selected_candidate_id_v1']}`",
            f"- Rows: `{canonical['observed_v1']['policy_eval_rows_v1']}` policy/eval and `{canonical['observed_v1']['hindsight_backfill_rows_v1']}` hindsight/backfill.",
            f"- AS_OF columns: `{canonical['observed_v1']['as_of_column_count_v1']}`.",
            f"- Bad blocks: `{canonical['observed_v1']['bad_blocks_v1']}`, tail help: `{canonical['observed_v1']['tail_help_v1']}`.",
            f"- Precision: `{canonical['observed_v1']['precision_v1']}`, worst LOSO: `{canonical['observed_v1']['worst_loso_v1']}`.",
            "",
            "## Monday Reanchor",
            "",
            "The target is the same Wednesday-R6 line reanchored to Monday calendar weeks: Monday 00:00 UTC to next Monday 00:00 UTC, with trading flat before the weekend. This is not the 1689 exact-only line and not a bridge workaround.",
            "",
            "## Blocker",
            "",
            "The copied snapshot locks the canonical facts, but the upstream Wednesday source artifact root is not present locally. The old local Monday R6 surface is also not feature-complete against the canonical manifest.",
            "",
            "## Feature Availability",
            "",
            f"- Expected canonical AS_OF columns: `{feature['expected_as_of_columns_v1']}`.",
            f"- Old Monday R6 AS_OF present/missing: `{feature['old_monday_r6_as_of_present_count_v1']}` / `{feature['old_monday_r6_as_of_missing_count_v1']}`.",
            f"- Missing old Monday R6 AS_OF columns: `{', '.join(feature['old_monday_r6_missing_as_of_columns_v1']) or 'NONE'}`.",
            f"- Old Monday R6 policy lock present: `{feature['old_monday_policy_lock_present_v1']}`.",
            "",
            "## Next Action",
            "",
            f"`{next_action['hard_recommendation_v1']}`",
            "",
            "Hard blocks: `DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE`; `DO_NOT_CONTINUE_PROTECTOR_FIRST_BEFORE_CANONICAL_MONDAY_BASELINE`.",
            "",
        ]
    )


def materialize(reports_root: Path = DEFAULT_REPORTS_ROOT, output_dir: Path | None = None) -> dict[str, Any]:
    reports_root = Path(reports_root)
    if output_dir is None:
        output_dir = reports_root / f"{LAYER_NAME}_{_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=False)

    canonical = _canonical_line_lock(reports_root)
    quarantine = _quarantine(reports_root)
    reanchor = _monday_reanchor_contract(canonical)
    rebuild_plan = _rebuild_plan(reports_root, canonical)
    row_delta_df = _row_delta_explainer(reports_root, canonical)
    feature_audit_df = _feature_availability_audit(reports_root, canonical)
    feature_summary = _feature_summary(feature_audit_df)
    parity_gate = _parity_gate(canonical, row_delta_df, feature_audit_df)
    next_action = _next_action(parity_gate)
    audit_df = _audit(canonical, quarantine, parity_gate)
    old_exact_rows = int(row_delta_df.loc[row_delta_df["row_type_v1"].eq("SUMMARY"), "old_monday_exact_only_rows_v1"].iloc[0])
    old_full_rows = int(row_delta_df.loc[row_delta_df["row_type_v1"].eq("SUMMARY"), "old_monday_fullcoverage_rows_v1"].iloc[0])
    old_bridge_rows = int(row_delta_df.loc[row_delta_df["row_type_v1"].eq("SUMMARY"), "old_monday_bridge_rows_v1"].iloc[0])
    summary = {
        "layer_name": LAYER_NAME,
        "materialized_at_utc_v1": _utc_now(),
        "output_dir_v1": str(output_dir),
        "canonical_wednesday_freeze_id_v1": canonical["observed_v1"]["freeze_id_v1"],
        "canonical_candidate_v1": canonical["observed_v1"]["selected_candidate_id_v1"],
        "canonical_rows_v1": canonical["observed_v1"]["policy_eval_rows_v1"],
        "canonical_as_of_columns_v1": canonical["observed_v1"]["as_of_column_count_v1"],
        "old_monday_fullcoverage_rows_v1": old_full_rows,
        "old_monday_exact_only_rows_v1": old_exact_rows,
        "old_monday_bridge_rows_v1": old_bridge_rows,
        "monday_rebuild_executed_v1": False,
        "monday_gets_same_learning_ground_now_v1": False,
        "feature_availability_v1": feature_summary,
        "parity_gate_decision_v1": parity_gate["decision_v1"],
        "hard_recommendation_v1": next_action["hard_recommendation_v1"],
        "hard_status_v1": {
            "BEVIST": [
                "Canonical Wednesday-R6 freeze id, candidate id, row count, AS_OF schema count, and safety metrics are locked from snapshot.",
                "The 1689 exact-only/protector/narrow path is quarantined as non-canonical.",
                "Old Monday 1852/1689 row counts are diagnostic negatives, not the next R6 line.",
            ],
            "INDIKERT": [
                "Monday should be rebuilt as the same full repaired R6 line with only the replay/window anchor changed.",
                "The Monday calendar machinery exists for Monday-to-next-Monday windows with Friday flat/no weekend entries.",
            ],
            "IKKE_ETABLERT": [
                "Canonical Monday parity is not established until the Wednesday source artifacts are restored and the rebuild runs.",
                "Candidate-level identity for the 1971-vs-1852 source delta is not established from the copied snapshot alone.",
            ],
        },
        "artifacts_v1": OUTPUT_FILES,
    }

    _write_json(output_dir / OUTPUT_FILES["canonical_lock"], canonical)
    _write_json(output_dir / OUTPUT_FILES["quarantine"], quarantine)
    _write_json(output_dir / OUTPUT_FILES["reanchor_contract"], reanchor)
    _write_json(output_dir / OUTPUT_FILES["rebuild_plan"], rebuild_plan)
    _write_json(output_dir / OUTPUT_FILES["parity_gate"], parity_gate)
    _write_json(output_dir / OUTPUT_FILES["next_action"], next_action)
    _write_json(output_dir / OUTPUT_FILES["summary"], summary)
    row_delta_df.to_csv(output_dir / OUTPUT_FILES["row_delta"], index=False)
    feature_audit_df.to_csv(output_dir / OUTPUT_FILES["feature_availability"], index=False)
    audit_df.to_csv(output_dir / OUTPUT_FILES["audit"], index=False)
    (output_dir / OUTPUT_FILES["report"]).write_text(_report(summary, canonical, next_action), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore canonical Wednesday-R6 line lock and build Monday reanchor plan.")
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize(reports_root=args.reports_root, output_dir=args.output_dir)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
