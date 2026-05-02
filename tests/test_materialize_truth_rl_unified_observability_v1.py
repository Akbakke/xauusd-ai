from __future__ import annotations

from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_truth_rl_unified_observability_v1 import (
    build_unified_rl_observability_payload,
)


def test_build_unified_rl_observability_keeps_entry_propensity_honest_and_links_management() -> None:
    entry_df = pd.DataFrame(
        [
            {
                "entry_row_key_v1": "e1",
                "run_id": "RUN_A",
                "candidate_uid": "c1",
                "trade_uid": "t1",
                "trade_id": "1",
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "split_bucket_v1": "train",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "logged_entry_action_v1": "TAKE_NOW",
                "entry_action_truth_status_v1": "COMPOSITIONAL_DIRECT_ENTRY_READ_MODEL",
                "policy_version_v1": "policy-a",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_id_status_v1": "POLICY_SNAPSHOT_ONLY_NOT_LOGGED_DIRECT_POLICY",
                "behavior_policy_kind_v1": "VERSIONED_ENTRY_MODEL_SNAPSHOT_ONLY",
                "entry_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
                "entry_reward_semantics_status_v1": "DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE",
                "terminal_outcome_available_v1": True,
                "entry_review_focus_v1": "ENTRY_OK_OR_MIXED",
                "actualized_take_v1": True,
                "as_of_hour_utc_v1": 0,
                "as_of_candidate_tradable_prob_v1": 0.9,
            },
            {
                "entry_row_key_v1": "e2",
                "run_id": "RUN_A",
                "candidate_uid": "c2",
                "trade_uid": "t2",
                "trade_id": "2",
                "decision_timestamp": "2026-01-01T00:05:00Z",
                "split_bucket_v1": "train",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "logged_entry_action_v1": "TAKE_NOW",
                "entry_action_truth_status_v1": "COMPOSITIONAL_DIRECT_ENTRY_READ_MODEL",
                "policy_version_v1": "policy-a",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_id_status_v1": "POLICY_SNAPSHOT_ONLY_NOT_LOGGED_DIRECT_POLICY",
                "behavior_policy_kind_v1": "VERSIONED_ENTRY_MODEL_SNAPSHOT_ONLY",
                "entry_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
                "entry_reward_semantics_status_v1": "DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE",
                "terminal_outcome_available_v1": True,
                "entry_review_focus_v1": "ENTRY_OK_OR_MIXED",
                "actualized_take_v1": True,
                "as_of_hour_utc_v1": 0,
                "as_of_candidate_tradable_prob_v1": 0.7,
            },
            {
                "entry_row_key_v1": "e3",
                "run_id": "RUN_A",
                "candidate_uid": "c3",
                "trade_uid": "t3",
                "trade_id": "3",
                "decision_timestamp": "2026-01-01T00:10:00Z",
                "split_bucket_v1": "train",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "logged_entry_action_v1": "SKIP",
                "entry_action_truth_status_v1": "COMPOSITIONAL_DIRECT_ENTRY_READ_MODEL",
                "policy_version_v1": "policy-a",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_id_status_v1": "POLICY_SNAPSHOT_ONLY_NOT_LOGGED_DIRECT_POLICY",
                "behavior_policy_kind_v1": "VERSIONED_ENTRY_MODEL_SNAPSHOT_ONLY",
                "entry_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
                "entry_reward_semantics_status_v1": "SKIP_AVOIDED_LOSS_HINDSIGHT_ONLY",
                "terminal_outcome_available_v1": False,
                "entry_review_focus_v1": "SHOULD_HAVE_SKIPPED",
                "actualized_take_v1": False,
                "as_of_hour_utc_v1": 0,
                "as_of_candidate_tradable_prob_v1": 0.1,
            },
        ]
    )
    management_row_df = pd.DataFrame(
        [
            {
                "management_row_key_v1": "m1",
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "1",
                "run_id": "RUN_A",
                "decision_anchor_timestamp_utc_v1": "2026-01-01T00:15:00Z",
                "action_label_v1": "HOLD",
                "split_bucket_v1": "train",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "entry_actualization_presence_status_v1": "ACTUALIZED_TAKE_EXACT",
                "management_path_relation_v1": "REALIZED_PATH_COMPATIBLE",
                "terminal_outcome_availability_status_v1": "EXACT_TERMINAL_OUTCOME_AVAILABLE",
                "management_decision_index_v1": 0,
                "as_of_atr_bps_v1": 12.0,
                "as_of_management_candidate_p_hat_v1": 0.8,
            },
            {
                "management_row_key_v1": "m2",
                "candidate_uid_exact_v1": "c3",
                "trade_uid_exact_v1": "t3",
                "trade_id_exact_v1": "3",
                "run_id": "RUN_A",
                "decision_anchor_timestamp_utc_v1": "2026-01-01T00:20:00Z",
                "action_label_v1": "EXIT_NOW",
                "split_bucket_v1": "train",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "entry_actualization_presence_status_v1": "DIRECT_ENTRY_ROUTE_EXACT_NON_ACTUALIZED",
                "management_path_relation_v1": "REALIZED_PATH_COMPATIBLE",
                "terminal_outcome_availability_status_v1": "EXACT_TERMINAL_OUTCOME_AVAILABLE",
                "management_decision_index_v1": 0,
                "as_of_atr_bps_v1": 11.0,
                "as_of_management_candidate_p_hat_v1": 0.2,
            },
        ]
    )
    management_policy_log_df = pd.DataFrame(
        [
            {
                "management_row_key_v1": "m1",
                "observed_action_v1": "HOLD",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_id_status_v1": "READY",
                "behavior_policy_kind_v1": "DETERMINISTIC",
                "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "policy_logging_propensity_status_v1": "DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT",
                "observed_action_propensity_v1": 1.0,
                "per_action_propensity_vector_v1": '{"HOLD": 1.0, "EXIT_NOW": 0.0}',
            },
            {
                "management_row_key_v1": "m2",
                "observed_action_v1": "EXIT_NOW",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_id_status_v1": "READY",
                "behavior_policy_kind_v1": "DETERMINISTIC",
                "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "policy_logging_propensity_status_v1": "DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT",
                "observed_action_propensity_v1": 1.0,
                "per_action_propensity_vector_v1": '{"HOLD": 0.0, "EXIT_NOW": 1.0}',
            },
        ]
    )
    scored_df = pd.DataFrame(
        [
            {"management_row_key_v1": "m1", "primary_model_score_v1": 0.7},
            {"management_row_key_v1": "m2", "primary_model_score_v1": 0.2},
        ]
    )

    payload = build_unified_rl_observability_payload(
        entry_df=entry_df,
        management_row_df=management_row_df,
        management_transition_df=management_row_df[["management_row_key_v1"]].copy(),
        management_bandit_df=management_row_df[["management_row_key_v1"]].copy(),
        management_exit_local_scored_df=scored_df,
        management_policy_log_df=management_policy_log_df,
        entry_contract={
            "observation_feature_names_v1": ["as_of_hour_utc_v1", "as_of_candidate_tradable_prob_v1"]
        },
        management_observation_contract={
            "observation_vector_feature_names_v1": ["as_of_atr_bps_v1", "as_of_management_candidate_p_hat_v1"]
        },
        entry_status={
            "ENTRY_POLICY_SNAPSHOT_STATUS": "READY_EXACT_CANDIDATE_POLICY_HASH_AND_MODEL_SNAPSHOT",
            "ENTRY_DIRECT_ACTION_STATUS": "HIERARCHICAL_COMPOSITION_READY_NOT_POLICY_TRUTH",
            "ENTRY_PROPENSITY_STATUS": "NOT_ESTABLISHED",
        },
        management_readiness_status={"MANAGEMENT_RL_READINESS_STATUS": "OFFLINE_RL_READINESS_SUBSTRATE_ONLY"},
        management_bandit_status={"MANAGEMENT_BANDIT_STATUS": "BANDIT_ACTION_REWARD_SUBSTRATE_ONLY"},
        management_sequence_status={"MANAGEMENT_RL_SEQUENCE_STATUS": "OFFLINE_RL_SEQUENCE_SUBSTRATE_ONLY"},
        management_policy_summary={
            "propensity_status_counts_v1": {"DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT": 2}
        },
        entry_handoff_summary={
            "management_handoff_status_counts_v1": {
                "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD": 1,
                "ACTUAL_TAKE_WITH_MANAGEMENT_DIAGNOSTIC_ONLY_REVIEW": 1,
            }
        },
        review_dir=Path("/tmp/review"),
        management_policy_dir=Path("/tmp/policy"),
    )

    assert payload["summary_v1"]["failed_check_count_v1"] == 0
    assert payload["summary_v1"]["entry_episode_rows_v1"] == 3
    assert payload["summary_v1"]["decision_event_rows_v1"] == 5
    assert payload["summary_v1"]["actualized_entry_without_management_rows_v1"] == 1
    assert payload["status_v1"]["ENTRY_PROPENSITY_STATUS"] == "NOT_ESTABLISHED"
    assert payload["status_v1"]["MANAGEMENT_PROPENSITY_STATUS"] == "READY_DETERMINISTIC_LOGGED_ACTION_PROPENSITY"
    assert set(payload["decision_event_view_v1_df"]["rl_domain_v1"]) == {"ENTRY", "MANAGEMENT"}


def test_build_unified_rl_observability_adds_management_only_and_closed_ledger_only_episodes() -> None:
    entry_df = pd.DataFrame(
        [
            {
                "entry_row_key_v1": "e1",
                "run_id": "RUN_A",
                "candidate_uid": "c-entry",
                "trade_uid": "t-entry",
                "trade_id": "1",
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "split_bucket_v1": "TRAIN",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "logged_entry_action_v1": "TAKE_NOW",
                "entry_action_truth_status_v1": "COMPOSITIONAL_DIRECT_ENTRY_READ_MODEL",
                "policy_version_v1": "policy-a",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_id_status_v1": "POLICY_SNAPSHOT_ONLY_NOT_LOGGED_DIRECT_POLICY",
                "behavior_policy_kind_v1": "VERSIONED_ENTRY_MODEL_SNAPSHOT_ONLY",
                "entry_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
                "entry_reward_semantics_status_v1": "DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE",
                "terminal_outcome_available_v1": True,
                "entry_review_focus_v1": "ENTRY_OK_OR_MIXED",
                "actualized_take_v1": True,
                "as_of_hour_utc_v1": 0,
            }
        ]
    )
    management_row_df = pd.DataFrame(
        [
            {
                "management_row_key_v1": "m-entry",
                "candidate_uid_exact_v1": "c-entry",
                "trade_uid_exact_v1": "t-entry",
                "trade_id_exact_v1": "1",
                "run_id": "RUN_A",
                "decision_anchor_timestamp_utc_v1": "2026-01-01T00:15:00Z",
                "action_label_v1": "HOLD",
                "split_bucket_v1": "TRAIN",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "entry_actualization_presence_status_v1": "ACTUALIZED_TAKE_EXACT",
                "management_path_relation_v1": "REALIZED_PATH_COMPATIBLE",
                "terminal_outcome_availability_status_v1": "EXACT_TERMINAL_OUTCOME_AVAILABLE",
                "management_decision_index_v1": 0,
                "as_of_atr_bps_v1": 12.0,
            },
            {
                "management_row_key_v1": "m-only",
                "candidate_uid_exact_v1": "c-management",
                "trade_uid_exact_v1": "t-management",
                "trade_id_exact_v1": "2",
                "run_id": "RUN_A",
                "decision_anchor_timestamp_utc_v1": "2026-01-01T00:20:00Z",
                "action_label_v1": "HOLD",
                "split_bucket_v1": "TRAIN",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "entry_actualization_presence_status_v1": "NO_FROZEN_ENTRY_ROUTE_EXACT_LINK",
                "management_path_relation_v1": "REALIZED_PATH_COMPATIBLE",
                "terminal_outcome_availability_status_v1": "EXACT_TERMINAL_OUTCOME_AVAILABLE",
                "management_decision_index_v1": 0,
                "as_of_atr_bps_v1": 13.0,
            },
        ]
    )
    management_policy_log_df = pd.DataFrame(
        [
            {
                "management_row_key_v1": key,
                "observed_action_v1": "HOLD",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_id_status_v1": "READY",
                "behavior_policy_kind_v1": "DETERMINISTIC",
                "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "policy_logging_propensity_status_v1": "DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT",
                "observed_action_propensity_v1": 1.0,
                "per_action_propensity_vector_v1": '{"HOLD": 1.0, "EXIT_NOW": 0.0}',
            }
            for key in ["m-entry", "m-only"]
        ]
    )
    closed_trade_ledger_df = pd.DataFrame(
        [
            {
                "run_id": "RUN_A",
                "candidate_uid": "c-entry",
                "trade_uid": "t-entry",
                "trade_id": "1",
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
            },
            {
                "run_id": "RUN_A",
                "candidate_uid": "c-management",
                "trade_uid": "t-management",
                "trade_id": "2",
                "decision_timestamp": "2026-01-01T00:20:00Z",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
            },
            {
                "run_id": "RUN_A",
                "candidate_uid": "c-ledger",
                "trade_uid": "t-ledger",
                "trade_id": "3",
                "decision_timestamp": "2026-01-01T00:30:00Z",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "policy_hash": "policy-a",
                "hindsight_entry_decision_review_v1": "TAKE_WAS_OK",
            },
        ]
    )
    candidate_snapshot_df = pd.DataFrame(
        [
            {
                "candidate_uid": "c-ledger",
                "hour_utc": 0,
                "p_hat": 0.8,
                "p_long": 0.8,
                "p_short": 0.1,
                "p_flat": 0.1,
            }
        ]
    )

    payload = build_unified_rl_observability_payload(
        entry_df=entry_df,
        management_row_df=management_row_df,
        management_transition_df=management_row_df[["management_row_key_v1"]].copy(),
        management_bandit_df=management_row_df[["management_row_key_v1"]].copy(),
        management_exit_local_scored_df=pd.DataFrame({"management_row_key_v1": ["m-entry", "m-only"]}),
        management_policy_log_df=management_policy_log_df,
        closed_trade_ledger_df=closed_trade_ledger_df,
        candidate_snapshot_df=candidate_snapshot_df,
        entry_contract={"observation_feature_names_v1": ["as_of_hour_utc_v1", "as_of_skip_xgb_p_hat_v1"]},
        management_observation_contract={"observation_vector_feature_names_v1": ["as_of_atr_bps_v1"]},
        entry_status={
            "ENTRY_POLICY_SNAPSHOT_STATUS": "READY_EXACT_CANDIDATE_POLICY_HASH_AND_MODEL_SNAPSHOT",
            "ENTRY_DIRECT_ACTION_STATUS": "HIERARCHICAL_COMPOSITION_READY_NOT_POLICY_TRUTH",
            "ENTRY_PROPENSITY_STATUS": "NOT_ESTABLISHED",
        },
        management_readiness_status={"MANAGEMENT_RL_READINESS_STATUS": "OFFLINE_RL_READINESS_SUBSTRATE_ONLY"},
        management_bandit_status={"MANAGEMENT_BANDIT_STATUS": "BANDIT_ACTION_REWARD_SUBSTRATE_ONLY"},
        management_sequence_status={"MANAGEMENT_RL_SEQUENCE_STATUS": "OFFLINE_RL_SEQUENCE_SUBSTRATE_ONLY"},
        management_policy_summary={
            "propensity_status_counts_v1": {"DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT": 2}
        },
        entry_handoff_summary={
            "management_handoff_status_counts_v1": {
                "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD": 1,
                "ACTUAL_TAKE_WITH_MANAGEMENT_DIAGNOSTIC_ONLY_REVIEW": 0,
            }
        },
        review_dir=Path("/tmp/review"),
        management_policy_dir=Path("/tmp/policy"),
    )

    assert payload["summary_v1"]["failed_check_count_v1"] == 0
    assert payload["summary_v1"]["entry_episode_rows_v1"] == 3
    assert payload["summary_v1"]["closed_trade_ledger_episode_covered_count_v1"] == 3
    assert payload["summary_v1"]["episode_source_counts_v1"] == {
        "ENTRY_DIRECT_WITH_MANAGEMENT_ROWS": 1,
        "MANAGEMENT_ONLY_NO_FROZEN_ENTRY_ROUTE_EXACT_LINK": 1,
        "CLOSED_TRADE_LEDGER_ONLY_NO_ENTRY_OR_MANAGEMENT_ROWS": 1,
    }
    assert payload["summary_v1"]["closed_trade_ledger_event_rows_v1"] == 1
    assert set(payload["decision_event_view_v1_df"]["rl_domain_v1"]) == {
        "ENTRY",
        "MANAGEMENT",
        "CLOSED_TRADE_LEDGER",
    }
