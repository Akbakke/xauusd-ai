#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    _json_ready,
    _read_csv_optional,
    _read_json_optional,
    _resolve_foundation_dir,
    _resolve_reports_root,
    _sha256,
    _utc_now,
    _write_json,
)


LAYER_ID = "LOCK_FIRST_BANDIT_REWARD_VERSION_V1"
CONTRACT_LOCK_LAYER_ID = "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1"
FIRST_LOCKED_REWARD_VERSION_ID = "MGMT_BANDIT_REALIZED_PNL_BPS_V1"
SELECTED_REWARD_NAME = "REALIZED_PNL_REWARD"
R6_FREEZE_ID = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"

LOCKABLE_REWARDS = [
    "REALIZED_PNL_REWARD",
    "MFE_CAPTURE_REWARD",
    "MAE_PENALTY_REWARD",
    "GIVEBACK_PENALTY_REWARD",
    "TAIL_CONTROL_REWARD",
]
NON_LOCK_REWARDS = ["RUNNER_DAMAGE_PENALTY", "TRANSPARENT_COMBINED_REWARD"]

OUTPUTS = {
    "contract": "lock_first_bandit_reward_version_contract_v1.json",
    "selection_lock_csv": "first_bandit_reward_selection_lock_v1.csv",
    "selection_lock_json": "first_bandit_reward_selection_lock_v1.json",
    "reward_contract": "first_bandit_reward_contract_v1.json",
    "exclusions_csv": "reward_exclusions_and_non_locks_v1.csv",
    "exclusions_json": "reward_exclusions_and_non_locks_v1.json",
    "readiness_update": "bandit_reward_readiness_status_update_v1.json",
    "next_step_matrix": "post_lock_next_step_matrix_v1.csv",
    "summary": "lock_first_bandit_reward_version_summary_v1.json",
    "report": "lock_first_bandit_reward_version_report_v1.md",
    "manifest": "lock_first_bandit_reward_version_manifest_v1.json",
    "status": "lock_first_bandit_reward_version_status_v1.json",
    "consistency_audit": "lock_first_bandit_reward_version_consistency_audit_v1.csv",
    "non_interference_audit": "lock_first_bandit_reward_version_non_interference_audit_v1.csv",
    "non_interference_audit_json": "lock_first_bandit_reward_version_non_interference_audit_v1.json",
}


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    return reports_root / "IQL_INTEGRATION" / f"{LAYER_ID}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def _latest_contract_lock_dir(reports_root: Path, contract_lock_dir_arg: str | None) -> Path:
    if contract_lock_dir_arg:
        path = Path(contract_lock_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Contract lock dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(
        base.glob(f"{CONTRACT_LOCK_LAYER_ID}_*/iql_reward_comparator_bandit_contract_lock_summary_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {CONTRACT_LOCK_LAYER_ID} output found under {base}")
    return candidates[0].parent.resolve()


def _source_paths(reports_root: Path, foundation_dir: Path, contract_lock_dir: Path, foundation_contract: dict[str, Any]) -> dict[str, str | None]:
    source_truth = foundation_contract.get("source_truth_v1", {}) if isinstance(foundation_contract.get("source_truth_v1"), dict) else {}
    return {
        "reports_root_v1": str(reports_root),
        "foundation_dir_v1": str(foundation_dir),
        "contract_lock_dir_v1": str(contract_lock_dir),
        "locked_ledger_source_v1": source_truth.get("locked_ledger_source_file_v1"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
    }


def _reward_review_by_name(reward_review_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if reward_review_df.empty:
        return {}
    return {str(row.get("reward_candidate_v1")): row for row in reward_review_df.to_dict(orient="records")}


def _foundation_stats_by_name(reward_audit_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if reward_audit_df.empty:
        return {}
    return {str(row.get("reward_candidate_v1")): row for row in reward_audit_df.to_dict(orient="records")}


def _build_selection_lock(reward_review_df: pd.DataFrame, reward_audit_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    review = _reward_review_by_name(reward_review_df)
    stats = _foundation_stats_by_name(reward_audit_df)
    rows: list[dict[str, Any]] = []
    for name in LOCKABLE_REWARDS:
        row = review.get(name, {})
        stat = stats.get(name, {})
        if name == SELECTED_REWARD_NAME:
            verdict = "PRIMARY_LOCK_CANDIDATE"
            reason = "Direct terminal realized PnL is the simplest fully covered scalar outcome, with low leakage risk when used only as HINDSIGHT reward."
            robustness = "HIGH_FOR_FIRST_BANDIT_BASELINE_RESEARCH"
        elif name == "TAIL_CONTROL_REWARD":
            verdict = "KEEP_FOR_REVIEW_ONLY"
            reason = "Useful tail-aware candidate, but fixed penalty weights should not be first scalar lock before calibration review."
            robustness = "MEDIUM_REQUIRES_WEIGHT_REVIEW"
        else:
            verdict = "SECONDARY_CANDIDATE"
            reason = "Useful management side objective, but less direct than realized PnL for the first scalar lock."
            robustness = "MEDIUM_AS_SIDE_METRIC_OR_FUTURE_REWARD"
        rows.append(
            {
                "reward_name_v1": name,
                "formula_draft_v1": row.get("formula_draft_v1") or stat.get("formula_v1"),
                "sign_direction_v1": row.get("sign_direction_v1"),
                "required_inputs_v1": row.get("required_inputs_v1"),
                "coverage_rate_v1": float(row.get("coverage_rate_v1", stat.get("coverage_rate_v1", 0.0)) or 0.0),
                "coverage_count_v1": int(row.get("coverage_count_v1", stat.get("distribution_count_v1", 0)) or 0),
                "hindsight_only_v1": bool(row.get("hindsight_only_v1", True)),
                "leakage_risk_v1": row.get("leakage_risk_v1"),
                "interpretability_v1": "HIGH" if name == SELECTED_REWARD_NAME else "MEDIUM",
                "first_bandit_reward_robustness_v1": robustness,
                "management_alignment_v1": "ALIGNED_WITH_HOLD_EXIT_NOW_TERMINAL_OUTCOME_RESEARCH",
                "strengths_v1": reason,
                "weaknesses_v1": "Does not by itself solve support, transition, tail, or slice safety checks.",
                "selection_reason_v1": reason,
                "hard_verdict_v1": verdict,
                "selected_as_first_locked_reward_v1": name == SELECTED_REWARD_NAME,
            }
        )
    df = pd.DataFrame.from_records(rows)
    selected = df.loc[df["selected_as_first_locked_reward_v1"]]
    can_lock = int(len(selected)) == 1 and float(selected.iloc[0]["coverage_rate_v1"]) >= 1.0
    lock = {
        "lock_id_v1": "FIRST_BANDIT_REWARD_SELECTION_LOCK_V1",
        "selection_status_v1": "FIRST_LOCKED_BANDIT_REWARD_VERSION" if can_lock else "REWARD_VERSION_NOT_LOCKED",
        "selected_reward_name_v1": SELECTED_REWARD_NAME if can_lock else None,
        "selected_reward_version_id_v1": FIRST_LOCKED_REWARD_VERSION_ID if can_lock else None,
        "scalar_reward_locked_v1": bool(can_lock),
        "scope_v1": "MANAGEMENT_CONTEXTUAL_BANDIT_RESEARCH_ONLY_NOT_IQL_REWARD",
        "not_universal_reward_v1": True,
        "not_sequence_iql_reward_lock_v1": True,
        "selection_rows_v1": df.to_dict(orient="records"),
    }
    return df, lock


def _build_reward_contract(
    selection_lock: dict[str, Any],
    reward_audit_df: pd.DataFrame,
    source_paths: dict[str, str | None],
) -> dict[str, Any]:
    if not selection_lock.get("scalar_reward_locked_v1"):
        return {
            "contract_id_v1": "FIRST_BANDIT_REWARD_CONTRACT_V1",
            "verdict_v1": "REWARD_LOCK_FAILED",
            "reason_v1": "No single eligible reward candidate passed the fail-closed lock conditions.",
        }
    stats = _foundation_stats_by_name(reward_audit_df).get(SELECTED_REWARD_NAME, {})
    return {
        "contract_id_v1": "FIRST_BANDIT_REWARD_CONTRACT_V1",
        "verdict_v1": "LOCKED_FOR_MANAGEMENT_BANDIT_RESEARCH_ONLY",
        "reward_version_id_v1": FIRST_LOCKED_REWARD_VERSION_ID,
        "reward_name_v1": SELECTED_REWARD_NAME,
        "exact_scalar_formula_v1": "reward_bps = terminal_realized_pnl_bps",
        "formula_components_v1": ["terminal_realized_pnl_bps"],
        "sign_convention_v1": "MAXIMIZE_HIGHER_REALIZED_PNL_BPS",
        "clipping_normalization_rules_v1": "NONE_IN_V1_KEEP_CANONICAL_BPS_SCALE",
        "allowed_value_range_v1": {
            "type_v1": "FINITE_FLOAT_BPS",
            "observed_min_v1": stats.get("distribution_min_v1"),
            "observed_max_v1": stats.get("distribution_max_v1"),
            "range_note_v1": "Observed audit range is descriptive; validator requires finite numeric reward.",
        },
        "missing_value_policy_v1": "FAIL_CLOSED_IF_NULL_NAN_INF_OR_UNJOINABLE",
        "canonical_source_artifacts_v1": source_paths,
        "hindsight_backfill_requirements_v1": [
            "terminal realized PnL must come from locked/frozen HINDSIGHT outcome truth",
            "row must remain traceable to candidate_uid_exact and locked ledger source",
            "no synthetic or fallback reward rows",
        ],
        "prohibited_inputs_v1": [
            "AS_OF state features",
            "path-dynamics v2 fields",
            "HOLD next_state transition fields",
            "counterfactual runner-damage fields",
            "policy score fields",
        ],
        "explicit_leakage_boundary_v1": "HINDSIGHT terminal outcome may be used as reward only; never as state_t or action provenance.",
        "allowed_use_v1": [
            "management contextual bandit research dataset after dataset build",
            "offline comparator/evaluation with locked fail-check policy",
        ],
        "forbidden_use_v1": [
            "IQL training",
            "sequence-IQL dataset",
            "R7 training",
            "live gate/controller/policy promotion",
            "entry controller reward",
            "path-dynamics canonical training feature",
        ],
        "bandit_suitability_v1": "SUITABLE_FOR_FIRST_MANAGEMENT_CONTEXTUAL_BANDIT_RESEARCH",
        "sequence_suitability_v1": "TERMINAL_REWARD_ONLY_DOES_NOT_UNBLOCK_SEQUENCE_IQL",
        "audit_suitability_v1": "BASELINE_SCALAR_PLUS_SIDE_METRIC_COMPARATOR_REQUIRED",
        "comparator_notes_v1": "Must be paired with locked comparator/fail-check contract; headline PnL alone is not enough.",
        "caveats_v1": [
            "First simple scalar reward, not universal RL objective.",
            "Does not solve HOLD transition truth.",
            "Does not incorporate tail/runner/giveback safety by itself.",
        ],
    }


def _build_exclusions(selection_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selection_df.to_dict(orient="records"):
        if row["selected_as_first_locked_reward_v1"]:
            continue
        name = str(row["reward_name_v1"])
        rows.append(
            {
                "reward_name_v1": name,
                "why_not_selected_now_v1": row["selection_reason_v1"],
                "still_useful_for_audit_v1": True,
                "still_useful_as_side_metric_v1": True,
                "can_enter_later_combined_reward_review_v1": True,
                "status_v1": "KEEP_FOR_FUTURE_REVIEW" if row["hard_verdict_v1"] == "SECONDARY_CANDIDATE" else "NOT_FOR_FIRST_LOCK",
            }
        )
    for name in NON_LOCK_REWARDS:
        reason = (
            "Counterfactual runner damage locality is not locked."
            if name == "RUNNER_DAMAGE_PENALTY"
            else "Composite weights and counterfactual component are not locked."
        )
        rows.append(
            {
                "reward_name_v1": name,
                "why_not_selected_now_v1": reason,
                "still_useful_for_audit_v1": True,
                "still_useful_as_side_metric_v1": True,
                "can_enter_later_combined_reward_review_v1": True,
                "status_v1": "AUDIT_ONLY",
            }
        )
    df = pd.DataFrame.from_records(rows)
    return df, {
        "lock_id_v1": "REWARD_EXCLUSIONS_AND_NON_LOCKS_V1",
        "rows_v1": df.to_dict(orient="records"),
    }


def _build_readiness_update(selection_lock: dict[str, Any], contract_lock_summary: dict[str, Any]) -> dict[str, Any]:
    locked = bool(selection_lock.get("scalar_reward_locked_v1"))
    return {
        "update_id_v1": "BANDIT_REWARD_READINESS_STATUS_UPDATE_V1",
        "first_scalar_bandit_reward_version_locked_v1": locked,
        "reward_version_id_v1": selection_lock.get("selected_reward_version_id_v1"),
        "management_bandit_dataset_previous_status_v1": contract_lock_summary.get("bandit_dataset_contract_verdict_v1"),
        "management_bandit_dataset_new_status_v1": "READY_TO_BUILD_WITH_LOCKED_REWARD" if locked else "STILL_BLOCKED_NO_REWARD_LOCK",
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "r7_status_changed_v1": False,
        "replay_status_changed_v1": False,
        "hard_status_v1": {
            "BEVIST": [
                "reward_version_locked_for_management_bandit_research_only" if locked else "reward_version_not_locked",
                "sequence_iql_still_blocked",
                "r7_status_unchanged",
                "replay_status_unchanged",
            ],
            "INDIKERT": [
                "bandit_dataset_can_be_built_next" if locked else "reward_lock_review_should_continue",
            ],
            "IKKE_ETABLERT": [
                "canonical_hold_next_state_transitions",
                "sequence_iql_readiness",
                "r7_training_readiness",
            ],
        },
    }


def _build_next_step_matrix(readiness_update: dict[str, Any]) -> pd.DataFrame:
    dataset_ready = readiness_update["management_bandit_dataset_new_status_v1"] == "READY_TO_BUILD_WITH_LOCKED_REWARD"
    rows = [
        {
            "decision_v1": "BUILD_BANDIT_DATASET_NEXT",
            "recommendation_v1": "DO_NEXT" if dataset_ready else "WAIT",
            "hard_status_v1": "BEVIST" if dataset_ready else "IKKE_ETABLERT",
            "reason_v1": "Reward_version is locked and field contract already exists." if dataset_ready else "Reward_version not locked.",
        },
        {
            "decision_v1": "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
            "recommendation_v1": "WAIT",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Path-dynamics remains non-canonical for training.",
        },
        {
            "decision_v1": "DO_NOT_START_R7_YET",
            "recommendation_v1": "DO_NOT_START",
            "hard_status_v1": "BEVIST",
            "reason_v1": "R7 still requires completed replay and post-replay audit.",
        },
        {
            "decision_v1": "DO_NOT_START_IQL_YET",
            "recommendation_v1": "DO_NOT_START",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Harness remains NOT_READY_FOR_IQL_TRAINING and sequence transitions are incomplete.",
        },
        {
            "decision_v1": "SEQUENCE_IQL_BLOCKED",
            "recommendation_v1": "BLOCKED",
            "hard_status_v1": "BEVIST",
            "reason_v1": "Reward lock does not create HOLD next_state truth.",
        },
        {
            "decision_v1": "BANDIT_FIRST_IF_TRANSITIONS_STAY_MISSING",
            "recommendation_v1": "BANDIT_FIRST_PATH",
            "hard_status_v1": "INDIKERT",
            "reason_v1": "Bandit remains the first honest RL-adjacent path if transitions stay missing.",
        },
    ]
    return pd.DataFrame.from_records(rows)


def _build_non_interference(
    output_dir: Path,
    source_paths: dict[str, str | None],
    exit_manager_sha_before: str | None,
    exit_manager_sha_after: str | None,
    r6_sha_before: str | None,
    r6_sha_after: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_values = [str(value) for value in source_paths.values() if value]
    checks = [
        ("OUTPUT_DIR_IS_IQL_INTEGRATION_NAMESPACE", "PASS" if "IQL_INTEGRATION" in output_dir.parts else "FAIL", str(output_dir), "path contains IQL_INTEGRATION"),
        ("OUTPUT_DIR_NOT_REPLAY_DIRECTORY", "PASS" if "PATH_DYNAMICS_LOGGING_V2_REPLAY" not in str(output_dir) else "FAIL", str(output_dir), "no replay path"),
        ("NO_IN_PROGRESS_REPLAY_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no replay source path"),
        ("RAW_STATE_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("POLICY_LOG_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("EXIT_MANAGER_UNTOUCHED", "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL", exit_manager_sha_after, exit_manager_sha_before),
        ("R6_FREEZE_UNTOUCHED", "PASS" if r6_sha_before == r6_sha_after else "FAIL", r6_sha_after, r6_sha_before),
        ("R7_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("IQL_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("SEQUENCE_IQL_DATASET_NOT_BUILT", "PASS", "not_built", "not_built"),
    ]
    df = pd.DataFrame.from_records(
        [
            {
                "check_name_v1": name,
                "status_v1": status,
                "observed_value_v1": observed,
                "expected_value_v1": expected,
            }
            for name, status, observed, expected in checks
        ]
    )
    return df, {
        "audit_id_v1": "NON_INTERFERENCE_RECHECK_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _build_consistency(
    selection_lock: dict[str, Any],
    reward_contract: dict[str, Any],
    readiness_update: dict[str, Any],
    non_interference: dict[str, Any],
) -> pd.DataFrame:
    checks = [
        (
            "EXACTLY_ONE_REWARD_SELECTED",
            selection_lock.get("selection_status_v1") == "FIRST_LOCKED_BANDIT_REWARD_VERSION"
            and selection_lock.get("selected_reward_name_v1") == SELECTED_REWARD_NAME,
            selection_lock.get("selected_reward_name_v1"),
            SELECTED_REWARD_NAME,
        ),
        (
            "REWARD_VERSION_ID_LOCKED",
            selection_lock.get("selected_reward_version_id_v1") == FIRST_LOCKED_REWARD_VERSION_ID,
            selection_lock.get("selected_reward_version_id_v1"),
            FIRST_LOCKED_REWARD_VERSION_ID,
        ),
        (
            "REWARD_CONTRACT_SCOPE_IS_BANDIT_ONLY",
            reward_contract.get("verdict_v1") == "LOCKED_FOR_MANAGEMENT_BANDIT_RESEARCH_ONLY",
            reward_contract.get("verdict_v1"),
            "LOCKED_FOR_MANAGEMENT_BANDIT_RESEARCH_ONLY",
        ),
        (
            "BANDIT_DATASET_READY_AFTER_REWARD_LOCK",
            readiness_update.get("management_bandit_dataset_new_status_v1") == "READY_TO_BUILD_WITH_LOCKED_REWARD",
            readiness_update.get("management_bandit_dataset_new_status_v1"),
            "READY_TO_BUILD_WITH_LOCKED_REWARD",
        ),
        (
            "SEQUENCE_IQL_STILL_BLOCKED",
            readiness_update.get("sequence_iql_still_blocked_v1") is True,
            readiness_update.get("sequence_iql_still_blocked_v1"),
            True,
        ),
        (
            "NON_INTERFERENCE_PASSED",
            int(non_interference.get("failed_check_count_v1", 1) or 0) == 0,
            non_interference.get("failed_check_count_v1"),
            0,
        ),
    ]
    return pd.DataFrame.from_records(
        [
            {
                "check_name_v1": name,
                "status_v1": "PASS" if passed else "FAIL",
                "observed_value_v1": observed,
                "expected_value_v1": expected,
            }
            for name, passed, observed, expected in checks
        ]
    )


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# Lock First Bandit Reward Version V1",
            "",
            "## Reward Lock",
            "",
            f"- Selected reward: `{summary['selected_reward_name_v1']}`",
            f"- Reward version: `{summary['reward_version_id_v1']}`",
            f"- Lock verdict: `{summary['reward_lock_verdict_v1']}`",
            f"- Bandit dataset status: `{summary['bandit_dataset_new_status_v1']}`",
            "",
            "## Boundaries",
            "",
            "- This is a management contextual bandit research reward, not an IQL reward.",
            "- It does not unblock sequence-IQL, R7, replay, or HOLD transition truth.",
            "",
            "## Next",
            "",
            "- `BUILD_BANDIT_DATASET_NEXT` is now the next safe build step.",
            "- `WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN`, `DO_NOT_START_R7_YET`, `DO_NOT_START_IQL_YET`, and `SEQUENCE_IQL_BLOCKED` remain active.",
        ]
    ) + "\n"


def build_reward_lock(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    contract_lock_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
    r6_sha_before: str | None = None,
    r6_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    foundation_dir = foundation_dir or _resolve_foundation_dir(reports_root, None)
    contract_lock_dir = contract_lock_dir or _latest_contract_lock_dir(reports_root, None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    foundation_contract = _read_json_optional(foundation_dir / "iql_foundation_mdp_contract_v1.json")
    reward_audit_df = _read_csv_optional(foundation_dir / "iql_foundation_reward_audit_v1.csv")
    contract_summary = _read_json_optional(contract_lock_dir / "iql_reward_comparator_bandit_contract_lock_summary_v1.json")
    reward_review_df = _read_csv_optional(contract_lock_dir / "iql_reward_contract_lock_review_v1.csv")
    comparator_lock = _read_json_optional(contract_lock_dir / "iql_baseline_comparator_and_failcheck_lock_v1.json")

    source_paths = _source_paths(reports_root, foundation_dir, contract_lock_dir, foundation_contract)
    selection_df, selection_lock = _build_selection_lock(reward_review_df, reward_audit_df)
    reward_contract = _build_reward_contract(selection_lock, reward_audit_df, source_paths)
    exclusions_df, exclusions = _build_exclusions(selection_df)
    readiness_update = _build_readiness_update(selection_lock, contract_summary)
    next_step_df = _build_next_step_matrix(readiness_update)
    non_interference_df, non_interference = _build_non_interference(
        output_dir,
        source_paths,
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    consistency_df = _build_consistency(selection_lock, reward_contract, readiness_update, non_interference)

    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_APPEND_ONLY_REWARD_VERSION_LOCK",
        "source_paths_v1": source_paths,
        "hard_boundaries_v1": {
            "do_not_touch_replay_v1": True,
            "do_not_start_replay_v1": True,
            "do_not_rebuild_raw_state_v1": True,
            "do_not_rebuild_policy_log_v1": True,
            "do_not_modify_exit_manager_v1": True,
            "do_not_train_r7_v1": True,
            "do_not_train_iql_v1": True,
            "do_not_build_sequence_iql_dataset_v1": True,
            "do_not_use_in_progress_replay_as_canonical_v1": True,
            "do_not_modify_r6_freeze_v1": True,
            "do_not_modify_locked_ledger_v1": True,
            "not_iql_reward_v1": True,
        },
        "input_comparator_failcheck_contract_v1": comparator_lock.get("lock_id_v1"),
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "selected_reward_name_v1": selection_lock.get("selected_reward_name_v1"),
        "reward_version_id_v1": selection_lock.get("selected_reward_version_id_v1"),
        "reward_lock_succeeded_v1": bool(selection_lock.get("scalar_reward_locked_v1")),
        "reward_lock_verdict_v1": reward_contract.get("verdict_v1"),
        "why_selected_v1": "Direct terminal realized PnL is fully covered, simple, low leakage when reward-only, independent of path-dynamics and HOLD transitions.",
        "explicitly_not_selected_v1": exclusions_df["reward_name_v1"].astype(str).tolist(),
        "bandit_dataset_previous_status_v1": readiness_update.get("management_bandit_dataset_previous_status_v1"),
        "bandit_dataset_new_status_v1": readiness_update.get("management_bandit_dataset_new_status_v1"),
        "bandit_dataset_ready_to_build_v1": readiness_update.get("management_bandit_dataset_new_status_v1") == "READY_TO_BUILD_WITH_LOCKED_REWARD",
        "sequence_iql_still_blocked_v1": True,
        "hold_transition_truth_status_v1": readiness_update.get("hold_transition_truth_status_v1"),
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "exit_manager_modified_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "recommended_next_step_v1": "BUILD_BANDIT_DATASET_NEXT",
        "hard_status_partition_v1": {
            "BEVIST": [
                "first_management_bandit_reward_version_locked",
                "bandit_dataset_ready_to_build_with_locked_reward",
                "sequence_iql_still_blocked",
                "r7_not_started",
                "iql_training_not_started",
            ],
            "INDIKERT": [
                "bandit_first_if_transitions_stay_missing",
            ],
            "IKKE_ETABLERT": [
                "canonical_hold_next_state_transitions",
                "sequence_iql_readiness",
                "r7_training_readiness",
            ],
        },
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_FIRST_BANDIT_REWARD_VERSION_LOCK",
        "reward_lock_succeeded_v1": summary["reward_lock_succeeded_v1"],
        "training_executed_v1": False,
        "r7_started_v1": False,
        "replay_touched_v1": False,
        "failed_consistency_check_count_v1": int((consistency_df["status_v1"] != "PASS").sum()),
        "failed_non_interference_check_count_v1": int(non_interference["failed_check_count_v1"]),
    }
    return {
        "contract": contract,
        "selection_df": selection_df,
        "selection_lock": selection_lock,
        "reward_contract": reward_contract,
        "exclusions_df": exclusions_df,
        "exclusions": exclusions,
        "readiness_update": readiness_update,
        "next_step_df": next_step_df,
        "non_interference_df": non_interference_df,
        "non_interference": non_interference,
        "consistency_df": consistency_df,
        "summary": summary,
        "status": status,
        "source_paths": source_paths,
    }


def write_reward_lock_artifacts(
    reports_root: Path,
    *,
    foundation_dir: Path | None = None,
    contract_lock_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    target_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_output_dir(reports_root, built_at).resolve()
    exit_manager_path = Path("/home/andre2/src/GX1_ENGINE/gx1/execution/exit_manager.py")
    r6_path = reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"
    exit_manager_sha_before = _sha256(exit_manager_path)
    r6_sha_before = _sha256(r6_path)

    payload = build_reward_lock(
        reports_root,
        foundation_dir=foundation_dir,
        contract_lock_dir=contract_lock_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)

    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    payload["selection_df"].to_csv(target_dir / OUTPUTS["selection_lock_csv"], index=False)
    _write_json(target_dir / OUTPUTS["selection_lock_json"], payload["selection_lock"])
    _write_json(target_dir / OUTPUTS["reward_contract"], payload["reward_contract"])
    payload["exclusions_df"].to_csv(target_dir / OUTPUTS["exclusions_csv"], index=False)
    _write_json(target_dir / OUTPUTS["exclusions_json"], payload["exclusions"])
    _write_json(target_dir / OUTPUTS["readiness_update"], payload["readiness_update"])
    payload["next_step_df"].to_csv(target_dir / OUTPUTS["next_step_matrix"], index=False)

    exit_manager_sha_after = _sha256(exit_manager_path)
    r6_sha_after = _sha256(r6_path)
    non_interference_df, non_interference = _build_non_interference(
        target_dir,
        payload["source_paths"],
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    payload["non_interference_df"] = non_interference_df
    payload["non_interference"] = non_interference
    payload["consistency_df"] = _build_consistency(
        payload["selection_lock"],
        payload["reward_contract"],
        payload["readiness_update"],
        non_interference,
    )
    payload["summary"]["exit_manager_modified_v1"] = exit_manager_sha_before != exit_manager_sha_after
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["status"]["failed_consistency_check_count_v1"] = int((payload["consistency_df"]["status_v1"] != "PASS").sum())

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)
    _write_json(target_dir / OUTPUTS["summary"], payload["summary"])
    (target_dir / OUTPUTS["report"]).write_text(_markdown_report(payload), encoding="utf-8")

    artifact_paths = {key: str(target_dir / filename) for key, filename in OUTPUTS.items()}
    manifest = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": payload["summary"]["built_at_utc_v1"],
        "output_dir_v1": str(target_dir),
        "append_only_namespace_v1": "IQL_INTEGRATION",
        "artifact_paths_v1": artifact_paths,
        "source_paths_v1": payload["source_paths"],
        "read_only_references_v1": True,
        "not_trainer_v1": True,
        "not_iql_reward_v1": True,
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
    parser = argparse.ArgumentParser(description="Lock the first management contextual bandit reward version.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--foundation-dir", type=str, default=None)
    parser.add_argument("--contract-lock-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    reports_root = _resolve_reports_root(args.reports_root)
    foundation_dir = Path(args.foundation_dir).expanduser().resolve() if args.foundation_dir else None
    contract_lock_dir = Path(args.contract_lock_dir).expanduser().resolve() if args.contract_lock_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_reward_lock_artifacts(
        reports_root,
        foundation_dir=foundation_dir,
        contract_lock_dir=contract_lock_dir,
        output_dir=output_dir,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
