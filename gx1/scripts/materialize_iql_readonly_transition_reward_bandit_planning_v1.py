#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")

LAYER_ID = "IQL_READONLY_TRANSITION_REWARD_BANDIT_PLANNING_V1"
FOUNDATION_LAYER_ID = "IQL_FOUNDATION_MDP_CONTRACT_AND_DATASET_SCAFFOLD_V1"
FOUNDATION_DIRNAME = "ALL_TRADE_REVIEW_LEDGER_20260422T_IQL_FOUNDATION_MDP_CONTRACT_AND_DATASET_SCAFFOLD_V1"
R5_2_FREEZE_ID = "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"
R6_FREEZE_ID = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"

FOUNDATION_TOP_SUMMARY = "truth_iql_foundation_mdp_contract_and_dataset_scaffold_v1.json"
LEDGER_FILE = "shadow_meta_all_trade_review_ledger_closed_trades.parquet"

OUTPUTS = {
    "contract": "iql_readonly_transition_reward_bandit_planning_contract_v1.json",
    "blocker_recheck": "iql_readonly_blocker_recheck_v1.json",
    "transition_gap_diagnosis": "iql_readonly_transition_gap_diagnosis_v1.json",
    "reward_contract_draft": "iql_readonly_reward_contract_draft_v1.csv",
    "reward_contract_draft_json": "iql_readonly_reward_contract_draft_v1.json",
    "bandit_planning": "iql_readonly_bandit_rl_planning_v1.csv",
    "boundary_lock": "iql_readonly_r5_2_r6_r7_boundary_lock_v1.json",
    "non_interference_audit": "iql_readonly_replay_non_interference_audit_v1.csv",
    "non_interference_audit_json": "iql_readonly_replay_non_interference_audit_v1.json",
    "next_action_matrix": "iql_readonly_next_action_matrix_v1.csv",
    "summary": "iql_readonly_summary_v1.json",
    "report": "iql_readonly_report_v1.md",
    "manifest": "iql_readonly_manifest_v1.json",
    "status": "iql_readonly_status_v1.json",
    "consistency_audit": "iql_readonly_consistency_audit_v1.csv",
}

PATH_DYNAMICS_V2_FIELDS = [
    "last_peak_ts",
    "last_mfe_ts",
    "last_peak_mfe",
    "max_mfe_without_mae",
    "mfe_mae_sequence_order",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty active truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return reports_root / "IQL_READINESS" / f"{LAYER_ID}_{stamp}"


def _json_ready(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_parquet_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_foundation_dir(reports_root: Path, foundation_dir_arg: str | None) -> Path:
    if foundation_dir_arg:
        path = Path(foundation_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Foundation dir does not exist: {path}")
        return path
    top = _read_json_optional(reports_root / FOUNDATION_TOP_SUMMARY)
    artifact_dir = top.get("artifact_dir_v1")
    if artifact_dir and Path(str(artifact_dir)).exists():
        return Path(str(artifact_dir)).expanduser().resolve()
    fallback = reports_root / FOUNDATION_DIRNAME
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"No IQL foundation output found under {reports_root}")


def _source_paths(
    reports_root: Path,
    foundation_dir: Path,
    foundation_contract: dict[str, Any],
) -> dict[str, str | None]:
    source_truth = foundation_contract.get("source_truth_v1", {}) if isinstance(foundation_contract.get("source_truth_v1"), dict) else {}
    return {
        "reports_root_v1": str(reports_root),
        "foundation_dir_v1": str(foundation_dir),
        "locked_ledger_source_v1": source_truth.get("locked_ledger_source_file_v1"),
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
        "entry_observability_summary_v1": str(reports_root / "truth_entry_rl_observability_v1.json"),
        "harvest_observability_summary_v1": str(reports_root / "truth_harvest_retrain_candidate_v1.json"),
        "rl_unified_observability_summary_v1": str(reports_root / "truth_rl_unified_observability_v1.json"),
    }


def _build_blocker_recheck(
    foundation_summary: dict[str, Any],
    transition: dict[str, Any],
    support: dict[str, Any],
    harness: dict[str, Any],
    mdp_feasibility_df: pd.DataFrame,
) -> dict[str, Any]:
    entry_status = foundation_summary.get("entry_iql_suitability_v1", "NOT_ESTABLISHED")
    if not mdp_feasibility_df.empty and "domain_v1" in mdp_feasibility_df.columns:
        entry_rows = mdp_feasibility_df.loc[mdp_feasibility_df["domain_v1"].astype("string").eq("ENTRY_IQL_FOUNDATION")]
        if not entry_rows.empty:
            entry_status = str(entry_rows.iloc[0].get("verdict_v1", entry_status))
    return {
        "audit_id_v1": "READONLY_IQL_BLOCKER_RECHECK_V1",
        "mode_v1": "READ_ONLY_RECHECK_NO_REPAIR",
        "management_status_v1": foundation_summary.get("management_mdp_verdict_v1", "NOT_ESTABLISHED"),
        "entry_status_v1": entry_status,
        "strict_transition_count_v1": int(transition.get("full_sequence_ready_transition_count_v1", foundation_summary.get("full_sequence_ready_transition_count_v1", 0)) or 0),
        "bandit_ready_row_count_v1": int(transition.get("bandit_only_row_count_v1", foundation_summary.get("bandit_only_row_count_v1", 0)) or 0),
        "hold_to_next_state_transition_count_v1": int(transition.get("hold_to_next_state_transition_count_v1", foundation_summary.get("hold_to_next_state_transition_count_v1", 0)) or 0),
        "hold_next_state_status_v1": "MISSING_BLOCKS_SEQUENCE_IQL" if int(transition.get("hold_to_next_state_transition_count_v1", 0) or 0) == 0 else "PARTIAL_OR_READY",
        "support_ood_verdict_v1": support.get("overall_support_verdict_v1", foundation_summary.get("support_ood_verdict_v1", "NOT_ESTABLISHED")),
        "bandit_support_verdict_v1": support.get("bandit_support_verdict_v1", foundation_summary.get("bandit_support_verdict_v1", "NOT_ESTABLISHED")),
        "training_harness_status_v1": harness.get("status_v1", foundation_summary.get("training_harness_status_v1", "NOT_ESTABLISHED")),
        "no_repair_attempted_v1": True,
        "no_training_attempted_v1": True,
    }


def _build_transition_gap_diagnosis(transition: dict[str, Any], dataset_schema: dict[str, Any]) -> dict[str, Any]:
    hold_next = int(transition.get("hold_to_next_state_transition_count_v1", 0) or 0)
    exact_next = int(transition.get("exact_next_management_state_count_v1", 0) or 0)
    bandit_only = int(transition.get("bandit_only_row_count_v1", 0) or 0)
    strict = int(transition.get("full_sequence_ready_transition_count_v1", 0) or 0)
    primary_gap = str(transition.get("primary_transition_gap_v1", "NOT_ESTABLISHED"))
    schema_fields = dataset_schema.get("fields_v1", []) if isinstance(dataset_schema.get("fields_v1"), list) else []
    next_state_declared = any(isinstance(row, dict) and row.get("field_name_v1") == "next_state_vector" for row in schema_fields)
    diagnosis = "NOT_ESTABLISHED"
    if hold_next == 0 and bandit_only > 0 and primary_gap == "HOLD_NEXT_STATE_LINKS_NOT_LOGGED":
        diagnosis = "LOGGING_GAP_AND_SINGLE_SNAPSHOT_PROBLEM_INDICATED"
    elif hold_next == 0 and next_state_declared:
        diagnosis = "SCHEMA_DECLARED_BUT_HOLD_NEXT_COVERAGE_MISSING"
    return {
        "audit_id_v1": "READONLY_TRANSITION_GAP_DIAGNOSIS_V1",
        "mode_v1": "READ_ONLY_FINISHED_ARTIFACTS_ONLY",
        "diagnosis_v1": diagnosis,
        "primary_gap_from_foundation_v1": primary_gap,
        "strict_transition_count_v1": strict,
        "exact_next_management_state_count_v1": exact_next,
        "hold_to_next_state_transition_count_v1": hold_next,
        "bandit_only_row_count_v1": bandit_only,
        "schema_gap_status_v1": "SCHEMA_FIELD_DECLARED" if next_state_declared else "SCHEMA_FIELD_NOT_ESTABLISHED",
        "logging_gap_status_v1": "INDIKERT" if diagnosis.startswith("LOGGING_GAP") else "IKKE_ETABLERT",
        "join_gap_status_v1": "NOT_ESTABLISHED_READONLY_NO_JOIN_ATTEMPT",
        "single_snapshot_problem_status_v1": "INDIKERT" if hold_next == 0 and bandit_only > 0 else "IKKE_ETABLERT",
        "canonical_linker_built_v1": False,
        "dataset_changed_v1": False,
        "in_progress_replay_used_v1": False,
    }


def _build_reward_contract_draft(reward_audit_df: pd.DataFrame) -> pd.DataFrame:
    if reward_audit_df.empty:
        return pd.DataFrame(
            columns=[
                "reward_candidate_v1",
                "draft_status_v1",
                "formula_v1",
                "coverage_rate_v1",
                "distribution_count_v1",
                "hindsight_only_v1",
                "leakage_risk_v1",
                "source_verdict_v1",
                "lock_decision_v1",
            ]
        )
    lockable = {
        "REALIZED_PNL_REWARD",
        "MFE_CAPTURE_REWARD",
        "MAE_PENALTY_REWARD",
        "GIVEBACK_PENALTY_REWARD",
        "TAIL_CONTROL_REWARD",
    }
    audit_only = {"RUNNER_DAMAGE_PENALTY", "TRANSPARENT_COMBINED_REWARD"}
    rows: list[dict[str, Any]] = []
    for row in reward_audit_df.to_dict(orient="records"):
        name = str(row.get("reward_candidate_v1", ""))
        source_verdict = str(row.get("verdict_v1", "NOT_READY"))
        coverage = float(row.get("coverage_rate_v1", 0.0) or 0.0)
        if name in lockable and source_verdict == "USABLE_FOR_OFFLINE_RESEARCH" and coverage > 0:
            draft_status = "LOCKABLE_AFTER_REVIEW"
        elif name in audit_only or source_verdict == "AUDIT_ONLY":
            draft_status = "AUDIT_ONLY"
        else:
            draft_status = "NOT_READY"
        rows.append(
            {
                "reward_candidate_v1": name,
                "draft_status_v1": draft_status,
                "formula_v1": row.get("formula_v1"),
                "coverage_rate_v1": coverage,
                "distribution_count_v1": int(row.get("distribution_count_v1", 0) or 0),
                "hindsight_only_v1": bool(row.get("hindsight_only_v1", True)),
                "leakage_risk_v1": row.get("leakage_risk_v1"),
                "source_verdict_v1": source_verdict,
                "lock_decision_v1": "DRAFT_ONLY_NO_SCALAR_REWARD_LOCKED",
                "training_use_v1": "DO_NOT_USE_FOR_TRAINING_UNTIL_REVIEW_LOCK",
                "trading_performance_interpretation_v1": "NOT_PERFORMED_FOUNDATION_ONLY",
            }
        )
    return pd.DataFrame.from_records(rows)


def _build_bandit_planning(reward_draft_df: pd.DataFrame, blocker: dict[str, Any], baseline_spec: dict[str, Any]) -> pd.DataFrame:
    lockable = sorted(
        reward_draft_df.loc[reward_draft_df["draft_status_v1"].eq("LOCKABLE_AFTER_REVIEW"), "reward_candidate_v1"].astype(str).tolist()
    ) if not reward_draft_df.empty else []
    baseline_slots = sorted((baseline_spec.get("baseline_comparator_presence_v1") or {}).keys())
    rows = [
        {
            "planning_area_v1": "dataset_schema",
            "planned_contract_v1": "contextual_bandit_rows_from_management_dm_candidate_view_after_reward_lock",
            "status_v1": "DRAFT_READY_NO_DATASET_BUILT",
            "blocker_v1": "reward_version_not_locked",
        },
        {
            "planning_area_v1": "action_space",
            "planned_contract_v1": "HOLD=0, EXIT_NOW=1",
            "status_v1": "READY_FROM_FOUNDATION",
            "blocker_v1": "none_for_planning",
        },
        {
            "planning_area_v1": "reward_version",
            "planned_contract_v1": ",".join(lockable) if lockable else "no_lockable_reward_candidate",
            "status_v1": "PENDING_REVIEW_LOCK",
            "blocker_v1": "final_scalar_reward_not_locked",
        },
        {
            "planning_area_v1": "baseline_comparator_registry",
            "planned_contract_v1": ",".join(baseline_slots),
            "status_v1": "REFERENCE_REGISTERED_CALIBRATION_PENDING",
            "blocker_v1": "baseline_calibration_outside_this_readonly_job",
        },
        {
            "planning_area_v1": "safety_metrics",
            "planned_contract_v1": "action_distribution,support_status,ood_action_rate,worst_slice_stability,no_hindsight_state_leakage",
            "status_v1": "DRAFT_READY",
            "blocker_v1": "requires_locked_dataset_for_execution",
        },
        {
            "planning_area_v1": "support_checks",
            "planned_contract_v1": f"sequence_support={blocker.get('support_ood_verdict_v1')}; bandit_support={blocker.get('bandit_support_verdict_v1')}",
            "status_v1": "READONLY_RECHECK_COMPLETE",
            "blocker_v1": "sequence_iql_support_too_thin",
        },
        {
            "planning_area_v1": "expected_blockers",
            "planned_contract_v1": "HOLD_next_state_missing,reward_version_not_locked,path_dynamics_not_canonical,R7_not_started",
            "status_v1": "BEVIST_FOR_SEQUENCE_IQL",
            "blocker_v1": "do_not_train",
        },
    ]
    return pd.DataFrame.from_records(rows)


def _build_boundary_lock(reports_root: Path) -> dict[str, Any]:
    r5_summary = _read_json_optional(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json")
    r6_summary = _read_json_optional(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json")
    return {
        "lock_id_v1": "R5_2_R6_R7_BOUNDARY_LOCK_V1",
        "r5_2_v1": {
            "role_v1": "FROZEN_HISTORICAL_REFERENCE",
            "expected_freeze_id_v1": R5_2_FREEZE_ID,
            "observed_freeze_id_v1": r5_summary.get("freeze_id_v1"),
            "status_v1": "REFERENCE_REGISTERED" if r5_summary.get("freeze_id_v1") == R5_2_FREEZE_ID else "MISSING_OR_MISMATCH",
            "not_rl_agent_v1": True,
        },
        "r6_v1": {
            "role_v1": "CURRENT_FROZEN_SHADOW_CANDIDATE",
            "expected_freeze_id_v1": R6_FREEZE_ID,
            "observed_freeze_id_v1": r6_summary.get("freeze_id_v1"),
            "status_v1": "REFERENCE_REGISTERED" if r6_summary.get("freeze_id_v1") == R6_FREEZE_ID else "MISSING_OR_MISMATCH",
            "not_rl_agent_v1": True,
        },
        "r7_v1": {
            "role_v1": "NOT_STARTED",
            "status_v1": "NOT_STARTED",
            "requires_completed_path_dynamics_replay_v1": True,
            "requires_post_replay_audit_v1": True,
            "training_started_by_this_job_v1": False,
            "future_evaluation_boundary_v1": "IF_TRAINED_LATER_R7_MUST_BE_EVALUATED_AGAINST_R6_CONTRACT_AFTER_REPLAY",
        },
        "no_policy_promotion_v1": True,
        "no_live_gate_v1": True,
    }


def _path_dynamics_planning_statuses() -> list[dict[str, Any]]:
    return [
        {
            "field_id_v1": field,
            "replay_status_v1": "PENDING_REPLAY",
            "canonical_status_v1": "NOT_CANONICAL_YET",
            "training_status_v1": "DO_NOT_USE_FOR_TRAINING",
        }
        for field in PATH_DYNAMICS_V2_FIELDS
    ]


def _build_non_interference_audit(
    *,
    output_dir: Path,
    source_paths: dict[str, str | None],
    boundary_lock: dict[str, Any],
    exit_manager_sha_before: str | None,
    exit_manager_sha_after: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    input_paths = [str(value) for value in source_paths.values() if value]
    checks = [
        {
            "check_name_v1": "OUTPUT_DIR_IS_IQL_READINESS_NAMESPACE",
            "status_v1": "PASS" if "IQL_READINESS" in output_dir.parts else "FAIL",
            "observed_value_v1": str(output_dir),
            "expected_value_v1": "path contains IQL_READINESS",
            "note_v1": "Append-only read-only planning artifacts must land in their own namespace.",
        },
        {
            "check_name_v1": "OUTPUT_DIR_NOT_REPLAY_DIRECTORY",
            "status_v1": "PASS" if "PATH_DYNAMICS_LOGGING_V2_REPLAY" not in str(output_dir) else "FAIL",
            "observed_value_v1": str(output_dir),
            "expected_value_v1": "no PATH_DYNAMICS_LOGGING_V2_REPLAY path segment",
            "note_v1": "This job must not write into replay directories.",
        },
        {
            "check_name_v1": "NO_IN_PROGRESS_REPLAY_OUTPUT_USED_AS_CANONICAL",
            "status_v1": "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in input_paths) else "FAIL",
            "observed_value_v1": json.dumps(input_paths, ensure_ascii=True, sort_keys=True),
            "expected_value_v1": "no replay output path in canonical inputs",
            "note_v1": "Path-dynamics v2 fields are only marked pending/non-canonical.",
        },
        {
            "check_name_v1": "RAW_STATE_REBUILD_NOT_REQUESTED",
            "status_v1": "PASS",
            "observed_value_v1": "not_invoked",
            "expected_value_v1": "not_invoked",
            "note_v1": "The materializer reads finished artifacts only.",
        },
        {
            "check_name_v1": "POLICY_LOG_REBUILD_NOT_REQUESTED",
            "status_v1": "PASS",
            "observed_value_v1": "not_invoked",
            "expected_value_v1": "not_invoked",
            "note_v1": "Policy logs are not rebuilt or overwritten.",
        },
        {
            "check_name_v1": "EXIT_MANAGER_UNCHANGED",
            "status_v1": "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL",
            "observed_value_v1": exit_manager_sha_after,
            "expected_value_v1": exit_manager_sha_before,
            "note_v1": "exit_manager.py must not be edited by this planning job.",
        },
        {
            "check_name_v1": "R7_NOT_STARTED",
            "status_v1": "PASS" if boundary_lock.get("r7_v1", {}).get("status_v1") == "NOT_STARTED" else "FAIL",
            "observed_value_v1": boundary_lock.get("r7_v1", {}).get("status_v1"),
            "expected_value_v1": "NOT_STARTED",
            "note_v1": "R7 training is outside this read-only planning task.",
        },
        {
            "check_name_v1": "PATH_DYNAMICS_FIELDS_NON_CANONICAL_FOR_TRAINING",
            "status_v1": "PASS",
            "observed_value_v1": "PENDING_REPLAY|NOT_CANONICAL_YET|DO_NOT_USE_FOR_TRAINING",
            "expected_value_v1": "PENDING_REPLAY|NOT_CANONICAL_YET|DO_NOT_USE_FOR_TRAINING",
            "note_v1": "In-progress path-dynamics replay is not admitted as canonical source.",
        },
    ]
    df = pd.DataFrame.from_records(checks)
    summary = {
        "audit_id_v1": "REPLAY_NON_INTERFERENCE_AUDIT_V1",
        "output_dir_v1": str(output_dir),
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": checks,
    }
    return summary, df


def _build_next_action_matrix(blocker: dict[str, Any], reward_draft_df: pd.DataFrame) -> pd.DataFrame:
    lockable_count = int(reward_draft_df["draft_status_v1"].eq("LOCKABLE_AFTER_REVIEW").sum()) if not reward_draft_df.empty else 0
    rows = [
        {
            "action_v1": "LOCK_REWARD_CONTRACT_NEXT",
            "recommendation_v1": "SAFE_TO_WORK_NOW_READONLY_REVIEW",
            "hard_status_v1": "INDIKERT" if lockable_count else "IKKE_ETABLERT",
            "reason_v1": f"{lockable_count} reward candidates are draft-lockable after review; no scalar reward is locked here.",
        },
        {
            "action_v1": "BUILD_BANDIT_DATASET_AFTER_REWARD",
            "recommendation_v1": "WAIT_FOR_REWARD_LOCK",
            "hard_status_v1": "INDIKERT",
            "reason_v1": "Bandit rows exist, but reward_version must be locked first.",
        },
        {
            "action_v1": "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
            "recommendation_v1": "WAIT_FOR_REPLAY_FOR_SEQUENCE_CHAIN",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Path-dynamics fields are non-canonical and HOLD next_state remains missing.",
        },
        {
            "action_v1": "DO_NOT_START_R7_YET",
            "recommendation_v1": "DO_NOT_START",
            "hard_status_v1": "BEVIST",
            "reason_v1": "R7 requires completed path-dynamics replay and post-replay audit.",
        },
        {
            "action_v1": "DO_NOT_START_IQL_YET",
            "recommendation_v1": "DO_NOT_START",
            "hard_status_v1": "BEVIST",
            "reason_v1": f"Management status is {blocker.get('management_status_v1')}; HOLD next_state count is {blocker.get('hold_to_next_state_transition_count_v1')}.",
        },
    ]
    return pd.DataFrame.from_records(rows)


def _build_consistency_audit(
    *,
    ledger_count: int | None,
    blocker: dict[str, Any],
    reward_draft_df: pd.DataFrame,
    boundary_lock: dict[str, Any],
    non_interference: dict[str, Any],
) -> pd.DataFrame:
    checks = [
        {
            "check_name_v1": "LOCKED_LEDGER_REFERENCE_AVAILABLE",
            "status_v1": "PASS" if ledger_count == 1971 else "FAIL",
            "observed_value_v1": ledger_count,
            "expected_value_v1": 1971,
            "note_v1": "Read-only reference to locked canonical ledger.",
        },
        {
            "check_name_v1": "FOUNDATION_RECHECK_STILL_NOT_TRAINING_READY",
            "status_v1": "PASS" if blocker.get("training_harness_status_v1") == "NOT_READY_FOR_IQL_TRAINING" else "FAIL",
            "observed_value_v1": blocker.get("training_harness_status_v1"),
            "expected_value_v1": "NOT_READY_FOR_IQL_TRAINING",
            "note_v1": "This layer plans next steps; it does not unlock training.",
        },
        {
            "check_name_v1": "NO_SCALAR_REWARD_LOCKED",
            "status_v1": "PASS" if "DRAFT_ONLY_NO_SCALAR_REWARD_LOCKED" in set(reward_draft_df.get("lock_decision_v1", pd.Series(dtype="string")).astype("string")) else "FAIL",
            "observed_value_v1": sorted(set(reward_draft_df.get("lock_decision_v1", pd.Series(dtype="string")).astype("string"))),
            "expected_value_v1": "DRAFT_ONLY_NO_SCALAR_REWARD_LOCKED",
            "note_v1": "Reward contract remains draft-only.",
        },
        {
            "check_name_v1": "R7_REMAINS_NOT_STARTED",
            "status_v1": "PASS" if boundary_lock.get("r7_v1", {}).get("status_v1") == "NOT_STARTED" else "FAIL",
            "observed_value_v1": boundary_lock.get("r7_v1", {}).get("status_v1"),
            "expected_value_v1": "NOT_STARTED",
            "note_v1": "No R7 training is started.",
        },
        {
            "check_name_v1": "REPLAY_NON_INTERFERENCE_PASSED",
            "status_v1": "PASS" if int(non_interference.get("failed_check_count_v1", 1) or 0) == 0 else "FAIL",
            "observed_value_v1": non_interference.get("failed_check_count_v1"),
            "expected_value_v1": 0,
            "note_v1": "The job must not write to replay or use replay outputs canonically.",
        },
    ]
    return pd.DataFrame.from_records(checks)


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# IQL Readonly Transition Reward Bandit Planning V1",
        "",
        "## Scope",
        "",
        "- Read-only planning over finished artifacts.",
        "- No replay mutation, no raw-state rebuild, no policy-log rebuild, no R7 training, no IQL training.",
        "",
        "## Recheck",
        "",
        f"- Management status: `{summary['management_status_v1']}`",
        f"- Entry status: `{summary['entry_status_v1']}`",
        f"- Strict transitions: `{summary['strict_transition_count_v1']}`",
        f"- Bandit-ready rows: `{summary['bandit_ready_row_count_v1']}`",
        f"- HOLD -> next_state: `{summary['hold_to_next_state_transition_count_v1']}`",
        f"- Support/OOD: `{summary['support_ood_verdict_v1']}`",
        f"- Harness: `{summary['training_harness_status_v1']}`",
        "",
        "## Planning",
        "",
        f"- Transition gap diagnosis: `{summary['transition_gap_diagnosis_v1']}`",
        f"- Reward draft lockable count: `{summary['reward_lockable_after_review_count_v1']}`",
        f"- Path-dynamics training status: `{summary['path_dynamics_training_status_v1']}`",
        f"- Primary safe-now action: `{summary['primary_safe_now_action_v1']}`",
        "",
        "## Non-Interference",
        "",
        f"- Replay touched: `{summary['replay_touched_v1']}`",
        f"- Raw-state rebuilt: `{summary['raw_state_rebuilt_v1']}`",
        f"- Policy-log rebuilt: `{summary['policy_log_rebuilt_v1']}`",
        f"- R7 started: `{summary['r7_started_v1']}`",
        "",
        "## Next",
        "",
        "- `LOCK_REWARD_CONTRACT_NEXT` can be prepared now as review work.",
        "- `BUILD_BANDIT_DATASET_AFTER_REWARD` must wait for a locked reward version.",
        "- `WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN`, `DO_NOT_START_R7_YET`, and `DO_NOT_START_IQL_YET` remain active constraints.",
    ]
    return "\n".join(lines) + "\n"


def build_iql_readonly_planning(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    foundation_dir = foundation_dir or _resolve_foundation_dir(reports_root, None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    foundation_summary = _read_json_optional(foundation_dir / "iql_foundation_summary_v1.json")
    foundation_status = _read_json_optional(foundation_dir / "iql_foundation_status_v1.json")
    foundation_contract = _read_json_optional(foundation_dir / "iql_foundation_mdp_contract_v1.json")
    transition = _read_json_optional(foundation_dir / "iql_foundation_transition_linkage_audit_v1.json")
    support = _read_json_optional(foundation_dir / "iql_foundation_support_ood_audit_v1.json")
    dataset_schema = _read_json_optional(foundation_dir / "iql_foundation_dataset_schema_v1.json")
    baseline_spec = _read_json_optional(foundation_dir / "iql_foundation_baseline_comparator_spec_v1.json")
    harness = _read_json_optional(foundation_dir / "iql_foundation_training_harness_stub_v1.json")
    reward_audit_df = _read_csv_optional(foundation_dir / "iql_foundation_reward_audit_v1.csv")
    mdp_feasibility_df = _read_csv_optional(foundation_dir / "iql_foundation_mdp_domain_feasibility_audit_v1.csv")

    source_paths = _source_paths(reports_root, foundation_dir, foundation_contract)
    ledger_source = source_paths.get("locked_ledger_source_v1")
    ledger_count = _read_parquet_row_count(Path(str(ledger_source))) if ledger_source else None

    blocker = _build_blocker_recheck(foundation_summary, transition, support, harness, mdp_feasibility_df)
    transition_gap = _build_transition_gap_diagnosis(transition, dataset_schema)
    reward_draft_df = _build_reward_contract_draft(reward_audit_df)
    boundary_lock = _build_boundary_lock(reports_root)
    bandit_planning_df = _build_bandit_planning(reward_draft_df, blocker, baseline_spec)
    non_interference, non_interference_df = _build_non_interference_audit(
        output_dir=output_dir,
        source_paths=source_paths,
        boundary_lock=boundary_lock,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_after,
    )
    next_action_df = _build_next_action_matrix(blocker, reward_draft_df)
    consistency_df = _build_consistency_audit(
        ledger_count=ledger_count,
        blocker=blocker,
        reward_draft_df=reward_draft_df,
        boundary_lock=boundary_lock,
        non_interference=non_interference,
    )

    path_statuses = _path_dynamics_planning_statuses()
    lockable_rewards = reward_draft_df.loc[
        reward_draft_df["draft_status_v1"].eq("LOCKABLE_AFTER_REVIEW"), "reward_candidate_v1"
    ].astype(str).tolist() if not reward_draft_df.empty else []
    audit_only_rewards = reward_draft_df.loc[
        reward_draft_df["draft_status_v1"].eq("AUDIT_ONLY"), "reward_candidate_v1"
    ].astype(str).tolist() if not reward_draft_df.empty else []

    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_PLANNING_ONLY",
        "foundation_layer_v1": FOUNDATION_LAYER_ID,
        "source_paths_v1": source_paths,
        "hard_boundaries_v1": {
            "do_not_touch_replay_v1": True,
            "do_not_start_replay_v1": True,
            "do_not_rebuild_raw_state_v1": True,
            "do_not_rebuild_policy_log_v1": True,
            "do_not_modify_exit_manager_v1": True,
            "do_not_train_r7_v1": True,
            "do_not_train_iql_v1": True,
            "do_not_use_in_progress_path_dynamics_as_canonical_v1": True,
        },
        "path_dynamics_v2_fields_v1": path_statuses,
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "reports_root_v1": str(reports_root),
        "foundation_dir_v1": str(foundation_dir),
        "management_status_v1": blocker["management_status_v1"],
        "entry_status_v1": blocker["entry_status_v1"],
        "strict_transition_count_v1": blocker["strict_transition_count_v1"],
        "bandit_ready_row_count_v1": blocker["bandit_ready_row_count_v1"],
        "hold_to_next_state_transition_count_v1": blocker["hold_to_next_state_transition_count_v1"],
        "support_ood_verdict_v1": blocker["support_ood_verdict_v1"],
        "training_harness_status_v1": blocker["training_harness_status_v1"],
        "transition_gap_diagnosis_v1": transition_gap["diagnosis_v1"],
        "reward_lockable_after_review_count_v1": int(len(lockable_rewards)),
        "reward_lockable_after_review_candidates_v1": lockable_rewards,
        "reward_audit_only_candidates_v1": audit_only_rewards,
        "path_dynamics_training_status_v1": "DO_NOT_USE_FOR_TRAINING",
        "baseline_calibration_status_v1": baseline_spec.get("baseline_calibration_status_v1", "PENDING_EXTERNAL_CALIBRATION"),
        "bandit_dataset_waits_for_reward_lock_v1": True,
        "iql_still_waits_v1": True,
        "r7_started_v1": False,
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "primary_safe_now_action_v1": "LOCK_REWARD_CONTRACT_NEXT",
        "must_wait_actions_v1": [
            "BUILD_BANDIT_DATASET_AFTER_REWARD",
            "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
            "DO_NOT_START_R7_YET",
            "DO_NOT_START_IQL_YET",
        ],
        "hard_status_partition_v1": {
            "BEVIST": [
                "foundation_harness_not_ready_for_iql_training",
                "hold_next_state_count_is_zero",
                "r7_not_started",
                "no_replay_directory_written_by_this_job",
                "path_dynamics_fields_marked_do_not_use_for_training",
            ],
            "INDIKERT": [
                "reward_contract_review_can_continue_now",
                "bandit_planning_can_be_prepared_without_training",
                "transition_gap_looks_like_logging_or_single_snapshot_gap_from_finished_artifacts",
            ],
            "IKKE_ETABLERT": [
                "locked_scalar_reward_version",
                "canonical_hold_next_state_transitions",
                "bandit_dataset_build",
                "r7_training_readiness",
                "iql_training_readiness",
            ],
        },
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_READONLY_PLANNING_ONLY",
        "training_executed_v1": False,
        "replay_touched_v1": False,
        "r7_started_v1": False,
        "failed_consistency_check_count_v1": int((consistency_df["status_v1"] != "PASS").sum()),
        "failed_non_interference_check_count_v1": int(non_interference["failed_check_count_v1"]),
    }
    return {
        "contract": contract,
        "blocker": blocker,
        "transition_gap": transition_gap,
        "reward_draft_df": reward_draft_df,
        "bandit_planning_df": bandit_planning_df,
        "boundary_lock": boundary_lock,
        "non_interference": non_interference,
        "non_interference_df": non_interference_df,
        "next_action_df": next_action_df,
        "summary": summary,
        "status": status,
        "consistency_df": consistency_df,
        "foundation_status": foundation_status,
        "source_paths": source_paths,
        "ledger_count": ledger_count,
    }


def write_iql_readonly_planning_artifacts(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_output_dir(reports_root, built_at).resolve()
    exit_manager_path = Path("/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py")
    exit_manager_sha_before = _sha256(exit_manager_path)
    payload = build_iql_readonly_planning(
        reports_root,
        foundation_dir=foundation_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)

    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    _write_json(target_dir / OUTPUTS["blocker_recheck"], payload["blocker"])
    _write_json(target_dir / OUTPUTS["transition_gap_diagnosis"], payload["transition_gap"])
    payload["reward_draft_df"].to_csv(target_dir / OUTPUTS["reward_contract_draft"], index=False)
    _write_json(
        target_dir / OUTPUTS["reward_contract_draft_json"],
        {
            "draft_id_v1": "REWARD_CONTRACT_DRAFT_V1",
            "lock_status_v1": "DRAFT_ONLY_NO_SCALAR_REWARD_LOCKED",
            "rows_v1": payload["reward_draft_df"].to_dict(orient="records"),
        },
    )
    payload["bandit_planning_df"].to_csv(target_dir / OUTPUTS["bandit_planning"], index=False)
    _write_json(target_dir / OUTPUTS["boundary_lock"], payload["boundary_lock"])

    exit_manager_sha_after = _sha256(exit_manager_path)
    non_interference, non_interference_df = _build_non_interference_audit(
        output_dir=target_dir,
        source_paths=payload["source_paths"],
        boundary_lock=payload["boundary_lock"],
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_after,
    )
    payload["non_interference"] = non_interference
    payload["non_interference_df"] = non_interference_df
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["consistency_df"] = _build_consistency_audit(
        ledger_count=payload["ledger_count"],
        blocker=payload["blocker"],
        reward_draft_df=payload["reward_draft_df"],
        boundary_lock=payload["boundary_lock"],
        non_interference=non_interference,
    )
    payload["status"]["failed_consistency_check_count_v1"] = int((payload["consistency_df"]["status_v1"] != "PASS").sum())

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["next_action_df"].to_csv(target_dir / OUTPUTS["next_action_matrix"], index=False)
    _write_json(target_dir / OUTPUTS["summary"], payload["summary"])
    (target_dir / OUTPUTS["report"]).write_text(_markdown_report(payload), encoding="utf-8")
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)

    artifact_paths = {key: str(target_dir / filename) for key, filename in OUTPUTS.items()}
    manifest = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": payload["summary"]["built_at_utc_v1"],
        "output_dir_v1": str(target_dir),
        "artifact_paths_v1": artifact_paths,
        "append_only_namespace_v1": "IQL_READINESS",
        "source_paths_v1": payload["source_paths"],
        "not_trainer_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
        "read_only_references_v1": True,
    }
    _write_json(target_dir / OUTPUTS["manifest"], manifest)
    _write_json(target_dir / OUTPUTS["status"], payload["status"])

    return {
        "output_dir": str(target_dir),
        "artifact_paths": artifact_paths,
        "summary": payload["summary"],
        "status": payload["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize read-only IQL transition/reward/bandit planning artifacts.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--foundation-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    foundation_dir = Path(args.foundation_dir).expanduser().resolve() if args.foundation_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_iql_readonly_planning_artifacts(
        reports_root,
        foundation_dir=foundation_dir,
        output_dir=output_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
