#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    PATH_DYNAMICS_V2_FIELDS,
    _json_ready,
    _read_json_optional,
    _resolve_reports_root,
    _sha256,
    _utc_now,
    _write_json,
)


LAYER_ID = "WAIT_STATE_AND_POST_REPLAY_READY_LOCK_V1"
EVAL_LAYER_ID = "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1"

OUTPUTS = {
    "contract": "wait_state_post_replay_ready_lock_contract_v1.json",
    "wait_state_lock": "post_bandit_wait_state_lock_v1.json",
    "bandit_limitation_lock": "bandit_track_limitation_lock_v1.json",
    "allowed_continuation": "bandit_allowed_continuation_v1.csv",
    "forbidden_continuation": "bandit_forbidden_continuation_v1.csv",
    "allowed_claims": "bandit_allowed_claims_after_eval_v1.csv",
    "forbidden_claims": "bandit_forbidden_claims_after_eval_v1.csv",
    "replay_priority_lock": "replay_main_path_priority_lock_v1.json",
    "post_replay_gate": "post_replay_execution_gate_lock_v1.json",
    "post_replay_gate_steps": "post_replay_execution_gate_steps_v1.csv",
    "post_replay_gap_definitions": "post_replay_gap_definitions_v1.csv",
    "r7_sequence_block": "r7_and_sequence_block_lock_refresh_v1.json",
    "optional_tasks": "optional_next_small_research_tasks_v1.json",
    "optional_allowed_tasks": "optional_allowed_small_research_tasks_v1.csv",
    "optional_forbidden_tasks": "optional_forbidden_jobs_while_waiting_v1.csv",
    "next_step_lock": "summary_and_next_step_lock_v1.json",
    "summary": "wait_state_post_replay_ready_lock_summary_v1.json",
    "report": "wait_state_post_replay_ready_lock_report_v1.md",
    "manifest": "wait_state_post_replay_ready_lock_manifest_v1.json",
    "status": "wait_state_post_replay_ready_lock_status_v1.json",
    "consistency_audit": "wait_state_post_replay_ready_lock_consistency_audit_v1.csv",
    "consistency_audit_json": "wait_state_post_replay_ready_lock_consistency_audit_v1.json",
    "non_interference_audit": "wait_state_post_replay_ready_lock_non_interference_audit_v1.csv",
    "non_interference_audit_json": "wait_state_post_replay_ready_lock_non_interference_audit_v1.json",
}


def _default_output_dir(reports_root: Path, now: datetime) -> Path:
    return reports_root / "IQL_INTEGRATION" / f"{LAYER_ID}_{now.strftime('%Y%m%dT%H%M%SZ')}"


def _latest_eval_dir(reports_root: Path, eval_dir_arg: str | None) -> Path:
    if eval_dir_arg:
        path = Path(eval_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Eval dir does not exist: {path}")
        return path
    base = reports_root / "IQL_INTEGRATION"
    candidates = sorted(
        base.glob(f"{EVAL_LAYER_ID}_*/run_first_bandit_research_eval_summary_v1.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {EVAL_LAYER_ID} output found under {base}")
    return candidates[0].parent.resolve()


def _source_paths(reports_root: Path, eval_dir: Path, eval_manifest: dict[str, Any]) -> dict[str, str | None]:
    manifest_sources = eval_manifest.get("source_paths_v1", {}) if isinstance(eval_manifest.get("source_paths_v1"), dict) else {}
    return {
        "reports_root_v1": str(reports_root),
        "first_bandit_eval_dir_v1": str(eval_dir),
        "first_bandit_eval_summary_v1": str(eval_dir / "run_first_bandit_research_eval_summary_v1.json"),
        "first_bandit_eval_final_verdict_v1": str(eval_dir / "first_bandit_eval_final_verdict_v1.json"),
        "first_bandit_failcheck_review_v1": str(eval_dir / "first_bandit_failcheck_and_safety_review_v1.json"),
        "first_bandit_post_status_v1": str(eval_dir / "post_first_bandit_eval_status_update_v1.json"),
        "dataset_dir_v1": manifest_sources.get("dataset_dir_v1"),
        "eval_prep_dir_v1": manifest_sources.get("eval_prep_dir_v1"),
        "locked_ledger_source_v1": manifest_sources.get("locked_ledger_source_v1"),
        "r5_2_freeze_summary_v1": str(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json"),
        "r6_freeze_summary_v1": str(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json"),
    }


def _rows(items: list[tuple[str, str, str]], key_name: str) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [{"item_id_v1": item_id, key_name: text, "status_v1": status} for item_id, text, status in items]
    )


def _build_wait_state(eval_summary: dict[str, Any], final_verdict: dict[str, Any], failcheck: dict[str, Any]) -> dict[str, Any]:
    return {
        "lock_id_v1": "POST_BANDIT_WAIT_STATE_LOCK_V1",
        "wait_state_verdicts_v1": [
            "RESEARCH_ONLY_WAIT_STATE",
            "NO_POSITIVE_POLICY_CLAIM",
            "REPLAY_DEPENDENT_MAIN_PATH",
        ],
        "proved_after_first_eval_v1": [
            "bandit_pipeline_exists",
            "first_bandit_eval_completed",
            "dataset_reward_eval_contract_still_stands",
            "comparator_failcheck_contract_still_governing",
            "safety_verdict_no_positive_claim_allowed",
            "sequence_iql_still_blocked",
            "r7_still_blocked",
            "replay_status_unchanged",
        ],
        "indicated_only_v1": [
            "further_bandit_research_possible_only_under_limitations",
            f"eval_signal_{str(eval_summary.get('signal_polarity_v1', 'INCONCLUSIVE')).lower()}",
        ],
        "not_established_v1": [
            "positive_policy_claim",
            "sequence_iql_readiness",
            "r7_readiness",
            "canonical_hold_next_state_transitions",
            "path_dynamics_training_canonical_status",
            "policy_promotion_readiness",
        ],
        "why_bandit_research_only_v1": "First eval ended weak/inconclusive with safety verdict NO_POSITIVE_CLAIM_ALLOWED, OOD support failure, hard-gate blocks, severe action imbalance, and no transition truth.",
        "why_no_positive_policy_claim_v1": {
            "final_verdict_v1": final_verdict.get("final_verdict_v1"),
            "safety_verdict_v1": failcheck.get("safety_verdict_v1"),
            "hard_gate_block_count_v1": failcheck.get("hard_gate_block_count_v1"),
            "fail_count_v1": failcheck.get("fail_count_v1"),
            "indeterminate_count_v1": failcheck.get("indeterminate_count_v1"),
        },
        "why_main_path_is_replay_rebuild_hold_diagnosis_v1": "Sequence-IQL requires true transitions; HOLD -> next_state remains zero and path-dynamics is not canonical for training.",
        "hard_status_v1": {
            "BEVIST": [
                "weak_or_inconclusive_bandit_eval",
                "no_positive_policy_claim_allowed",
                "sequence_iql_still_blocked",
                "r7_still_blocked",
            ],
            "INDIKERT": [
                "limited_bandit_side_research_may_continue",
            ],
            "IKKE_ETABLERT": [
                "positive_policy_claim",
                "sequence_iql_readiness",
                "r7_readiness",
                "canonical_hold_transition_truth",
            ],
        },
    }


def _build_bandit_limitation_lock(eval_summary: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    allowed_continuation = _rows(
        [
            ("support_ood_audit", "support and OOD audit without policy claims", "ALLOWED_RESEARCH_ONLY"),
            ("comparator_calibration_audit", "comparator calibration audit without promotion language", "ALLOWED_RESEARCH_ONLY"),
            ("status_consolidation", "reporting cleanup and status consolidation", "ALLOWED_RESEARCH_ONLY"),
            ("no_op_monitoring_lock", "no-op monitoring/status lock while replay completes", "ALLOWED_RESEARCH_ONLY"),
        ],
        "allowed_continuation_v1",
    )
    forbidden_continuation = _rows(
        [
            ("r7_training", "R7 training", "FORBIDDEN"),
            ("iql_training", "IQL training", "FORBIDDEN"),
            ("sequence_dataset_build", "sequence-IQL dataset build", "FORBIDDEN"),
            ("transition_repair_pre_replay", "transition repair before replay truth", "FORBIDDEN"),
            ("policy_promotion_claim", "policy-promotion claims", "FORBIDDEN"),
            ("live_controller_claim", "live/controller claims", "FORBIDDEN"),
        ],
        "forbidden_continuation_v1",
    )
    allowed_claims = _rows(
        [
            ("bandit_pipeline_exists", "bandit pipeline exists as research-only infrastructure", "BEVIST"),
            ("first_eval_inconclusive", "first eval was weak/inconclusive and did not earn positive claim", "BEVIST"),
            ("limited_side_research", "small side research may continue under strict boundaries", "INDIKERT"),
        ],
        "allowed_claim_v1",
    )
    forbidden_claims = _rows(
        [
            ("positive_bandit_policy_claim", "positive bandit policy claim", "FORBIDDEN"),
            ("iql_ready", "IQL-ready", "FORBIDDEN"),
            ("sequence_ready", "sequence-ready", "FORBIDDEN"),
            ("r7_ready", "R7-ready", "FORBIDDEN"),
            ("live_ready", "live-ready", "FORBIDDEN"),
            ("hold_truth_established", "HOLD transition truth established", "FORBIDDEN"),
            ("path_dynamics_canonical", "path-dynamics canonical for training", "FORBIDDEN"),
        ],
        "forbidden_claim_v1",
    )
    lock = {
        "lock_id_v1": "BANDIT_TRACK_LIMITATION_LOCK_V1",
        "verdicts_v1": [
            "LIMITED_BANDIT_RESEARCH_ONLY",
            "NOT_SEQUENCE_RELEVANT_YET",
            "NOT_R7_RELEVANT_YET",
        ],
        "limitations_v1": {
            "signal_v1": eval_summary.get("final_verdict_v1"),
            "positive_claim_allowed_v1": False,
            "action_imbalance_v1": {
                "hold_rows_v1": eval_summary.get("hold_rows_v1"),
                "exit_now_rows_v1": eval_summary.get("exit_now_rows_v1"),
            },
            "support_ood_v1": eval_summary.get("support_ood_verdict_v1"),
            "ood_action_rate_failed_v1": True,
            "indeterminate_safety_check_count_v1": eval_summary.get("failcheck_indeterminate_count_v1"),
            "hard_gate_block_count_v1": eval_summary.get("hard_gate_block_count_v1"),
            "not_sequence_ready_v1": True,
            "not_r7_relevant_v1": True,
            "not_live_or_promotion_relevant_v1": True,
        },
        "allowed_continuation_rows_v1": allowed_continuation.to_dict(orient="records"),
        "forbidden_continuation_rows_v1": forbidden_continuation.to_dict(orient="records"),
        "allowed_claim_rows_v1": allowed_claims.to_dict(orient="records"),
        "forbidden_claim_rows_v1": forbidden_claims.to_dict(orient="records"),
    }
    return lock, allowed_continuation, forbidden_continuation, allowed_claims, forbidden_claims


def _build_replay_priority_lock() -> dict[str, Any]:
    return {
        "lock_id_v1": "REPLAY_MAIN_PATH_PRIORITY_LOCK_V1",
        "verdicts_v1": [
            "REPLAY_FIRST",
            "HOLD_TRUTH_STILL_CRITICAL",
            "SEQUENCE_IQL_STILL_BLOCKED",
        ],
        "why_replay_completion_is_most_important_v1": "Current sequence block is transition truth, not scalar reward or bandit eval. Replay completion is needed before canonical chain/HOLD diagnosis can be trusted.",
        "why_canonical_rebuild_needed_v1": "Post-replay artifacts must be frozen, coverage-checked, leakage-checked, and rebuilt into a canonical chain before they can influence MDP readiness.",
        "why_hold_diagnosis_critical_v1": "HOLD -> next_state remains zero; without true HOLD transitions sequence-IQL cannot be honestly evaluated.",
        "path_dynamics_training_status_v1": "PENDING_REPLAY_NOT_CANONICAL_YET_DO_NOT_USE_FOR_TRAINING",
        "why_sequence_iql_still_blocked_v1": "Bandit reward/dataset/eval do not create state_t_plus_1 or done truth for HOLD transitions.",
        "priority_order_v1": [
            "1_replay_completion",
            "2_post_replay_canonical_rebuild",
            "3_hold_transition_diagnosis",
            "4_mdp_readiness_classification",
            "5_only_then_reassess_sequence_track",
        ],
    }


def _build_post_replay_gate() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    steps = [
        ("verify_replay_completion", "verify completed replay marker and immutable output namespace", "BLOCK_IF_NOT_COMPLETE"),
        ("verify_finished_frozen_artifacts_only", "read only finished/frozen artifacts, never in-progress outputs", "BLOCK_IF_IN_PROGRESS"),
        ("verify_coverage_null_rate", "measure row coverage and null-rate for required AS_OF/path fields", "BLOCK_IF_COVERAGE_FAILS"),
        ("verify_leakage_status", "prove no HINDSIGHT outcome leaks into AS_OF state", "BLOCK_IF_LEAKAGE_RISK"),
        ("rebuild_canonical_chain", "rebuild/read canonical transition chain from frozen replay truth", "BLOCK_IF_CHAIN_MISSING"),
        ("rerun_hold_transition_diagnosis", "rerun HOLD -> next_state, ordering, terminal, and same-episode checks", "BLOCK_IF_HOLD_TRUTH_MISSING"),
        ("classify_readiness", "classify MDP_READY, BANDIT_ONLY, TRANSITION_LOGGING_REQUIRED, or NOT_ESTABLISHED", "REQUIRED_OUTPUT"),
    ]
    step_df = pd.DataFrame.from_records(
        [{"step_id_v1": step_id, "execution_rule_v1": rule, "fail_closed_condition_v1": fail} for step_id, rule, fail in steps]
    )
    gaps = [
        ("logging_gap", "required decision/transition event was never logged in finished artifacts"),
        ("join_gap", "candidate/trade/decision keys cannot join exactly across frozen sources"),
        ("ordering_gap", "timestamps cannot prove same-episode next-state order"),
        ("single_snapshot_gap", "only one management snapshot exists where a sequence requires multiple decision states"),
        ("ambiguity_gap", "multiple possible next states or duplicate keys prevent exact transition truth"),
        ("leakage_gap", "state candidate includes terminal/outcome/hindsight information"),
    ]
    gap_df = pd.DataFrame.from_records([{"gap_type_v1": gap, "definition_v1": definition} for gap, definition in gaps])
    gate = {
        "gate_id_v1": "POST_REPLAY_EXECUTION_GATE_LOCK_V1",
        "mode_v1": "PLAN_ONLY_NOT_EXECUTED_NOW",
        "classification_outputs_v1": [
            "MDP_READY",
            "BANDIT_ONLY",
            "TRANSITION_LOGGING_REQUIRED",
            "NOT_ESTABLISHED",
        ],
        "artifacts_to_read_first_v1": [
            "completed replay manifest/status",
            "frozen post-replay management observations",
            "locked canonical ledger",
            "AS_OF decision/policy logs",
            "HINDSIGHT outcome truth",
            "existing IQL foundation/bandit/eval contracts",
        ],
        "truth_source_priority_v1": [
            "locked_1971_ledger",
            "finished_frozen_post_replay_artifacts",
            "AS_OF_policy_decision_logs",
            "HINDSIGHT_outcome_truth_reward_only",
            "research_contracts_and_manifests",
        ],
        "join_keys_timestamps_to_check_v1": [
            "candidate_uid_exact",
            "trade_uid_exact / trade_uid",
            "management_row_key_v1",
            "decision_ts",
            "entry_timestamp",
            "exit_timestamp",
            "same_episode_ordering",
        ],
        "ambiguity_checks_v1": [
            "unique row_id",
            "unique candidate_uid per decision",
            "single exact next state per HOLD row",
            "monotonic decision_ts within episode",
            "terminal EXIT_NOW done semantics",
        ],
        "leakage_checks_v1": [
            "no hindsight/terminal/reward/outcome tokens in state_feature_names",
            "path-dynamics fields AS_OF provenance only",
            "HINDSIGHT columns reward/outcome only",
            "no post-exit information in state_t",
        ],
        "step_rows_v1": step_df.to_dict(orient="records"),
        "gap_definitions_v1": gap_df.to_dict(orient="records"),
    }
    return gate, step_df, gap_df


def _build_r7_sequence_block(eval_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "lock_id_v1": "R7_AND_SEQUENCE_BLOCK_LOCK_REFRESH_V1",
        "verdicts_v1": [
            "R7_STILL_BLOCKED",
            "SEQUENCE_IQL_STILL_BLOCKED",
            "BANDIT_EVAL_DID_NOT_UNLOCK_NEXT_PHASE",
        ],
        "r7_not_started_v1": True,
        "r7_still_blocked_v1": True,
        "sequence_iql_still_blocked_v1": True,
        "first_bandit_eval_unlocked_anything_v1": False,
        "positive_bandit_claim_earned_v1": False,
        "hold_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "path_dynamics_status_v1": "PENDING_REPLAY_NOT_CANONICAL_YET_DO_NOT_USE_FOR_TRAINING",
        "future_r7_requires_v1": [
            "completed path-dynamics replay",
            "post-replay audit",
            "comparator/fail-check calibration",
            "transition truth known",
            "R6 contract to beat if later trained",
        ],
        "future_sequence_iql_requires_v1": [
            "true HOLD transitions",
            "state_t/action_t/reward_t/state_t_plus_1/done_t contract",
            "reward/support/MDP readiness",
            "training harness readiness",
        ],
        "bandit_eval_reference_v1": {
            "final_verdict_v1": eval_summary.get("final_verdict_v1"),
            "safety_verdict_v1": eval_summary.get("safety_verdict_v1"),
        },
    }


def _build_optional_tasks() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    allowed = pd.DataFrame.from_records(
        [
            {
                "task_v1": "support_and_ood_audit",
                "allowed_reason_v1": "does not touch replay or train policy",
                "can_answer_v1": "where support is thin and which pockets are risky",
                "cannot_answer_v1": "sequence readiness or R7 readiness",
                "priority_v1": "OPTIONAL_USEFUL",
            },
            {
                "task_v1": "comparator_calibration_audit",
                "allowed_reason_v1": "clarifies future eval interpretation without promotion",
                "can_answer_v1": "which comparator thresholds are still uncalibrated",
                "cannot_answer_v1": "policy lift or live readiness",
                "priority_v1": "OPTIONAL_USEFUL",
            },
            {
                "task_v1": "reporting_cleanup_status_consolidation",
                "allowed_reason_v1": "append-only status/reporting work only",
                "can_answer_v1": "what is proved/indicated/not established",
                "cannot_answer_v1": "new trading performance truth",
                "priority_v1": "OPTIONAL",
            },
            {
                "task_v1": "no_op_monitoring_lock",
                "allowed_reason_v1": "records wait-state while replay runs",
                "can_answer_v1": "whether boundaries are still respected",
                "cannot_answer_v1": "readiness advancement",
                "priority_v1": "OPTIONAL",
            },
        ]
    )
    forbidden = pd.DataFrame.from_records(
        [
            ("R7 training", "FORBIDDEN"),
            ("IQL training", "FORBIDDEN"),
            ("sequence dataset build", "FORBIDDEN"),
            ("transition repair before replay truth", "FORBIDDEN"),
            ("policy-promotion claims", "FORBIDDEN"),
            ("live/controller claims", "FORBIDDEN"),
        ],
        columns=["forbidden_job_v1", "status_v1"],
    )
    lock = {
        "lock_id_v1": "OPTIONAL_NEXT_SMALL_RESEARCH_TASKS_V1",
        "verdict_v1": "ONLY_SMALL_RESEARCH_ALLOWED_WHILE_WAITING",
        "allowed_tasks_v1": allowed.to_dict(orient="records"),
        "forbidden_jobs_v1": forbidden.to_dict(orient="records"),
    }
    return lock, allowed, forbidden


def _build_next_step_lock() -> dict[str, Any]:
    return {
        "lock_id_v1": "SUMMARY_AND_NEXT_STEP_LOCK_V1",
        "decisions_v1": [
            "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
            "OPTIONAL_SUPPORT_AND_OOD_AUDIT_ONLY",
            "DO_NOT_START_R7_YET",
            "DO_NOT_START_IQL_YET",
            "DO_NOT_MAKE_POSITIVE_BANDIT_CLAIMS",
            "SEQUENCE_IQL_BLOCKED",
        ],
        "main_path_now_v1": "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
        "small_side_jobs_allowed_v1": [
            "support/OOD audit",
            "comparator calibration audit",
            "reporting/status consolidation",
            "no-op monitoring lock",
        ],
        "explicitly_do_not_do_v1": [
            "R7 training",
            "IQL training",
            "sequence dataset build",
            "transition repair before replay truth",
            "policy/live/controller claims",
            "positive bandit claim",
        ],
        "why_v1": "First bandit eval was weak/inconclusive and did not unlock transition truth; replay completion remains prerequisite for sequence-sporet.",
    }


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
        ("NO_IN_PROGRESS_REPLAY_USED_AS_CANONICAL", "PASS" if all("PATH_DYNAMICS_LOGGING_V2_REPLAY" not in path for path in source_values) else "FAIL", json.dumps(source_values, ensure_ascii=True), "no replay canonical source"),
        ("RAW_STATE_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("POLICY_LOG_UNTOUCHED", "PASS", "not_rebuilt", "not_rebuilt"),
        ("EXIT_MANAGER_UNTOUCHED", "PASS" if exit_manager_sha_before == exit_manager_sha_after else "FAIL", exit_manager_sha_after, exit_manager_sha_before),
        ("R6_FREEZE_UNTOUCHED", "PASS" if r6_sha_before == r6_sha_after else "FAIL", r6_sha_after, r6_sha_before),
        ("R7_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("IQL_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
        ("SEQUENCE_IQL_DATASET_NOT_BUILT", "PASS", "not_built", "not_built"),
        ("POLICY_PROMOTION_NOT_ATTEMPTED", "PASS", "not_attempted", "not_attempted"),
        ("BANDIT_TRAINING_NOT_STARTED", "PASS", "not_started", "not_started"),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": status, "observed_value_v1": observed, "expected_value_v1": expected} for name, status, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "NON_INTERFERENCE_RECHECK_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _build_consistency(
    eval_summary: dict[str, Any],
    wait_state: dict[str, Any],
    bandit_limit: dict[str, Any],
    replay_priority: dict[str, Any],
    post_replay_gate: dict[str, Any],
    r7_sequence: dict[str, Any],
    optional_tasks: dict[str, Any],
    next_step: dict[str, Any],
    non_interference: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks = [
        ("BANDIT_EVAL_WEAK_OR_INCONCLUSIVE", eval_summary.get("final_verdict_v1") == "WEAK_OR_INCONCLUSIVE_SIGNAL", eval_summary.get("final_verdict_v1"), "WEAK_OR_INCONCLUSIVE_SIGNAL"),
        ("NO_POSITIVE_CLAIM_LOCKED", "NO_POSITIVE_POLICY_CLAIM" in wait_state.get("wait_state_verdicts_v1", []), wait_state.get("wait_state_verdicts_v1"), "NO_POSITIVE_POLICY_CLAIM"),
        ("BANDIT_LIMITED_RESEARCH_ONLY", "LIMITED_BANDIT_RESEARCH_ONLY" in bandit_limit.get("verdicts_v1", []), bandit_limit.get("verdicts_v1"), "LIMITED_BANDIT_RESEARCH_ONLY"),
        ("REPLAY_MAIN_PATH_PRIORITY", replay_priority.get("priority_order_v1", [None])[0] == "1_replay_completion", replay_priority.get("priority_order_v1"), "1_replay_completion first"),
        ("POST_REPLAY_GATE_PLAN_ONLY", post_replay_gate.get("mode_v1") == "PLAN_ONLY_NOT_EXECUTED_NOW", post_replay_gate.get("mode_v1"), "PLAN_ONLY_NOT_EXECUTED_NOW"),
        ("R7_SEQUENCE_STILL_BLOCKED", "R7_STILL_BLOCKED" in r7_sequence.get("verdicts_v1", []) and "SEQUENCE_IQL_STILL_BLOCKED" in r7_sequence.get("verdicts_v1", []), r7_sequence.get("verdicts_v1"), "R7_STILL_BLOCKED|SEQUENCE_IQL_STILL_BLOCKED"),
        ("ONLY_SMALL_RESEARCH_ALLOWED", optional_tasks.get("verdict_v1") == "ONLY_SMALL_RESEARCH_ALLOWED_WHILE_WAITING", optional_tasks.get("verdict_v1"), "ONLY_SMALL_RESEARCH_ALLOWED_WHILE_WAITING"),
        ("NEXT_STEP_WAIT_FOR_REPLAY", next_step.get("main_path_now_v1") == "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN", next_step.get("main_path_now_v1"), "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN"),
        ("NON_INTERFERENCE_PASSED", int(non_interference.get("failed_check_count_v1", 1) or 0) == 0, non_interference.get("failed_check_count_v1"), 0),
    ]
    df = pd.DataFrame.from_records(
        [{"check_name_v1": name, "status_v1": "PASS" if passed else "FAIL", "observed_value_v1": observed, "expected_value_v1": expected} for name, passed, observed, expected in checks]
    )
    return df, {
        "audit_id_v1": "WAIT_STATE_AND_POST_REPLAY_READY_LOCK_CONSISTENCY_AUDIT_V1",
        "failed_check_count_v1": int((df["status_v1"] != "PASS").sum()),
        "passed_check_count_v1": int((df["status_v1"] == "PASS").sum()),
        "checks_v1": df.to_dict(orient="records"),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    return "\n".join(
        [
            "# Wait State And Post Replay Ready Lock V1",
            "",
            "## Verdict",
            "",
            f"- Wait-state: `{summary['wait_state_verdict_v1']}`",
            f"- Main path: `{summary['main_priority_v1']}`",
            f"- Bandit status: `{summary['bandit_track_status_v1']}`",
            f"- Sequence-IQL: `{summary['sequence_iql_status_v1']}`",
            f"- R7: `{summary['r7_status_v1']}`",
            "",
            "## Meaning",
            "",
            "- First bandit eval was useful because it prevented a false positive.",
            "- No positive bandit/policy claim is allowed.",
            "- Replay -> canonical rebuild -> HOLD diagnosis remains the main path.",
            "- Only small research/status side jobs are allowed while waiting.",
        ]
    ) + "\n"


def build_wait_state_lock(
    reports_root: Path,
    *,
    eval_dir: Path | None = None,
    output_dir: Path | None = None,
    built_at: datetime | None = None,
    exit_manager_sha_before: str | None = None,
    exit_manager_sha_after: str | None = None,
    r6_sha_before: str | None = None,
    r6_sha_after: str | None = None,
) -> dict[str, Any]:
    reports_root = Path(reports_root).expanduser().resolve()
    built_at = built_at or _utc_now()
    eval_dir = eval_dir or _latest_eval_dir(reports_root, None)
    output_dir = output_dir or _default_output_dir(reports_root, built_at)

    eval_summary = _read_json_optional(eval_dir / "run_first_bandit_research_eval_summary_v1.json")
    final_verdict = _read_json_optional(eval_dir / "first_bandit_eval_final_verdict_v1.json")
    failcheck = _read_json_optional(eval_dir / "first_bandit_failcheck_and_safety_review_v1.json")
    post_eval_status = _read_json_optional(eval_dir / "post_first_bandit_eval_status_update_v1.json")
    eval_manifest = _read_json_optional(eval_dir / "run_first_bandit_research_eval_manifest_v1.json")
    source_paths = _source_paths(reports_root, eval_dir, eval_manifest)

    wait_state = _build_wait_state(eval_summary, final_verdict, failcheck)
    bandit_limit, allowed_continuation, forbidden_continuation, allowed_claims, forbidden_claims = _build_bandit_limitation_lock(eval_summary)
    replay_priority = _build_replay_priority_lock()
    post_replay_gate, post_replay_steps, post_replay_gaps = _build_post_replay_gate()
    r7_sequence = _build_r7_sequence_block(eval_summary)
    optional_tasks, optional_allowed, optional_forbidden = _build_optional_tasks()
    next_step = _build_next_step_lock()
    non_interference_df, non_interference = _build_non_interference(
        output_dir,
        source_paths,
        exit_manager_sha_before,
        exit_manager_sha_after,
        r6_sha_before,
        r6_sha_after,
    )
    consistency_df, consistency = _build_consistency(
        eval_summary,
        wait_state,
        bandit_limit,
        replay_priority,
        post_replay_gate,
        r7_sequence,
        optional_tasks,
        next_step,
        non_interference,
    )

    contract = {
        "contract_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "mode_v1": "READONLY_APPEND_ONLY_WAIT_STATE_AND_POST_REPLAY_READY_LOCK",
        "source_paths_v1": source_paths,
        "not_replay_job_v1": True,
        "not_raw_state_rebuild_v1": True,
        "not_policy_log_rebuild_v1": True,
        "not_sequence_iql_dataset_build_v1": True,
        "not_iql_training_v1": True,
        "not_r7_training_v1": True,
        "not_policy_promotion_v1": True,
        "not_new_bandit_training_v1": True,
        "path_dynamics_v2_status_v1": {
            "status_v1": "PENDING_REPLAY_NOT_CANONICAL_YET_DO_NOT_USE_FOR_TRAINING",
            "fields_v1": PATH_DYNAMICS_V2_FIELDS,
        },
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
            "do_not_make_positive_bandit_claim_v1": True,
        },
    }
    summary = {
        "layer_id_v1": LAYER_ID,
        "built_at_utc_v1": built_at.isoformat(),
        "wait_state_verdict_v1": "RESEARCH_ONLY_WAIT_STATE",
        "wait_state_verdicts_v1": wait_state["wait_state_verdicts_v1"],
        "main_priority_v1": "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN",
        "bandit_track_status_v1": "LIMITED_BANDIT_RESEARCH_ONLY_NO_POSITIVE_CLAIM",
        "bandit_eval_final_verdict_v1": eval_summary.get("final_verdict_v1"),
        "bandit_eval_safety_verdict_v1": eval_summary.get("safety_verdict_v1"),
        "bandit_signal_polarity_v1": eval_summary.get("signal_polarity_v1"),
        "sequence_iql_status_v1": "SEQUENCE_IQL_STILL_BLOCKED",
        "r7_status_v1": "R7_STILL_BLOCKED",
        "hold_transition_truth_status_v1": "MISSING_HOLD_NEXT_STATE_COUNT_ZERO",
        "post_replay_gate_status_v1": "LOCKED_PLAN_ONLY_NOT_EXECUTED",
        "optional_side_jobs_v1": next_step["small_side_jobs_allowed_v1"],
        "forbidden_jobs_v1": next_step["explicitly_do_not_do_v1"],
        "recommended_next_steps_v1": next_step["decisions_v1"],
        "first_bandit_eval_reference_v1": str(eval_dir),
        "replay_touched_v1": False,
        "raw_state_rebuilt_v1": False,
        "policy_log_rebuilt_v1": False,
        "exit_manager_modified_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "hard_status_partition_v1": {
            "BEVIST": [
                "research_only_wait_state_locked",
                "no_positive_policy_claim_allowed",
                "replay_dependent_main_path",
                "sequence_iql_still_blocked",
                "r7_still_blocked",
            ],
            "INDIKERT": [
                "optional_small_research_side_jobs_may_continue",
            ],
            "IKKE_ETABLERT": [
                "positive_bandit_policy_claim",
                "sequence_iql_readiness",
                "r7_readiness",
                "canonical_hold_transition_truth",
                "path_dynamics_training_canonical_status",
            ],
        },
    }
    status = {
        "layer_id_v1": LAYER_ID,
        "status_v1": "MATERIALIZED_WAIT_STATE_AND_POST_REPLAY_READY_LOCK",
        "wait_state_locked_v1": True,
        "training_executed_v1": False,
        "r7_started_v1": False,
        "iql_training_started_v1": False,
        "sequence_iql_dataset_built_v1": False,
        "replay_touched_v1": False,
        "failed_consistency_check_count_v1": int(consistency["failed_check_count_v1"]),
        "failed_non_interference_check_count_v1": int(non_interference["failed_check_count_v1"]),
    }
    return {
        "contract": contract,
        "wait_state": wait_state,
        "bandit_limit": bandit_limit,
        "allowed_continuation": allowed_continuation,
        "forbidden_continuation": forbidden_continuation,
        "allowed_claims": allowed_claims,
        "forbidden_claims": forbidden_claims,
        "replay_priority": replay_priority,
        "post_replay_gate": post_replay_gate,
        "post_replay_steps": post_replay_steps,
        "post_replay_gaps": post_replay_gaps,
        "r7_sequence": r7_sequence,
        "optional_tasks": optional_tasks,
        "optional_allowed": optional_allowed,
        "optional_forbidden": optional_forbidden,
        "next_step": next_step,
        "non_interference_df": non_interference_df,
        "non_interference": non_interference,
        "consistency_df": consistency_df,
        "consistency": consistency,
        "summary": summary,
        "status": status,
        "source_paths": source_paths,
        "post_eval_status": post_eval_status,
    }


def write_wait_state_lock_artifacts(
    reports_root: Path,
    *,
    eval_dir: Path | None = None,
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
    payload = build_wait_state_lock(
        reports_root,
        eval_dir=eval_dir,
        output_dir=target_dir,
        built_at=built_at,
        exit_manager_sha_before=exit_manager_sha_before,
        exit_manager_sha_after=exit_manager_sha_before,
        r6_sha_before=r6_sha_before,
        r6_sha_after=r6_sha_before,
    )
    target_dir.mkdir(parents=True, exist_ok=False)

    _write_json(target_dir / OUTPUTS["contract"], payload["contract"])
    _write_json(target_dir / OUTPUTS["wait_state_lock"], payload["wait_state"])
    _write_json(target_dir / OUTPUTS["bandit_limitation_lock"], payload["bandit_limit"])
    payload["allowed_continuation"].to_csv(target_dir / OUTPUTS["allowed_continuation"], index=False)
    payload["forbidden_continuation"].to_csv(target_dir / OUTPUTS["forbidden_continuation"], index=False)
    payload["allowed_claims"].to_csv(target_dir / OUTPUTS["allowed_claims"], index=False)
    payload["forbidden_claims"].to_csv(target_dir / OUTPUTS["forbidden_claims"], index=False)
    _write_json(target_dir / OUTPUTS["replay_priority_lock"], payload["replay_priority"])
    _write_json(target_dir / OUTPUTS["post_replay_gate"], payload["post_replay_gate"])
    payload["post_replay_steps"].to_csv(target_dir / OUTPUTS["post_replay_gate_steps"], index=False)
    payload["post_replay_gaps"].to_csv(target_dir / OUTPUTS["post_replay_gap_definitions"], index=False)
    _write_json(target_dir / OUTPUTS["r7_sequence_block"], payload["r7_sequence"])
    _write_json(target_dir / OUTPUTS["optional_tasks"], payload["optional_tasks"])
    payload["optional_allowed"].to_csv(target_dir / OUTPUTS["optional_allowed_tasks"], index=False)
    payload["optional_forbidden"].to_csv(target_dir / OUTPUTS["optional_forbidden_tasks"], index=False)
    _write_json(target_dir / OUTPUTS["next_step_lock"], payload["next_step"])

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
    eval_summary = _read_json_optional(Path(payload["source_paths"]["first_bandit_eval_summary_v1"]))
    payload["consistency_df"], payload["consistency"] = _build_consistency(
        eval_summary,
        payload["wait_state"],
        payload["bandit_limit"],
        payload["replay_priority"],
        payload["post_replay_gate"],
        payload["r7_sequence"],
        payload["optional_tasks"],
        payload["next_step"],
        non_interference,
    )
    payload["summary"]["exit_manager_modified_v1"] = exit_manager_sha_before != exit_manager_sha_after
    payload["status"]["failed_non_interference_check_count_v1"] = int(non_interference["failed_check_count_v1"])
    payload["status"]["failed_consistency_check_count_v1"] = int(payload["consistency"]["failed_check_count_v1"])
    payload["non_interference_df"] = non_interference_df
    payload["non_interference"] = non_interference

    non_interference_df.to_csv(target_dir / OUTPUTS["non_interference_audit"], index=False)
    _write_json(target_dir / OUTPUTS["non_interference_audit_json"], non_interference)
    payload["consistency_df"].to_csv(target_dir / OUTPUTS["consistency_audit"], index=False)
    _write_json(target_dir / OUTPUTS["consistency_audit_json"], payload["consistency"])
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
        "not_training_v1": True,
        "not_replay_job_v1": True,
        "not_iql_v1": True,
        "not_r7_v1": True,
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
    parser = argparse.ArgumentParser(description="Materialize wait-state and post-replay readiness gate lock.")
    parser.add_argument("--reports-root", type=str, default=None)
    parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    reports_root = _resolve_reports_root(args.reports_root)
    eval_dir = Path(args.eval_dir).expanduser().resolve() if args.eval_dir else None
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    result = write_wait_state_lock_artifacts(reports_root, eval_dir=eval_dir, output_dir=output_dir)
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
