from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_build_management_bandit_dataset_v1 import (
    REWARD_VERSION_ID,
    build_management_bandit_dataset,
    write_management_bandit_dataset_artifacts,
)
from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import FOUNDATION_DIRNAME


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    foundation = root / FOUNDATION_DIRNAME
    management_dir = root / "MANAGEMENT_CANONICAL_FIXTURE"
    policy_dir = root / "POLICY_LOG_FIXTURE"
    reward_lock = root / "IQL_INTEGRATION" / "LOCK_FIRST_BANDIT_REWARD_VERSION_V1_FIXTURE"
    contract_lock = root / "IQL_INTEGRATION" / "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1_FIXTURE"
    for path in (foundation, management_dir, policy_dir, reward_lock, contract_lock):
        path.mkdir(parents=True, exist_ok=True)

    state_features = [
        "as_of_atr_bps_v1",
        "as_of_session_v1",
        "as_of_management_exit_prob_v1",
        "as_of_management_exit_prob_available_v1",
    ]
    _write_json(
        foundation / "iql_foundation_mdp_contract_v1.json",
        {
            "source_truth_v1": {
                "locked_ledger_source_file_v1": str(root / "ledger.parquet"),
                "management_substrate_dir_v1": str(management_dir),
                "policy_log_dir_v1": str(policy_dir),
                "reports_root_v1": str(root),
            }
        },
    )
    _write_json(
        foundation / "iql_foundation_dataset_schema_v1.json",
        {
            "schema_id_v1": "IQL_DATASET_SCHEMA_V1",
            "state_feature_names_v1": state_features,
            "state_feature_count_v1": len(state_features),
        },
    )
    _write_json(
        foundation / "iql_foundation_management_mdp_contract_v1.json",
        {"state_contract_v1": {"state_feature_names_v1": state_features, "state_feature_count_v1": len(state_features)}},
    )
    _write_json(foundation / "iql_foundation_support_ood_audit_v1.json", {"overall_support_verdict_v1": "SUPPORT_TOO_THIN"})

    dm = pd.DataFrame(
        [
            {
                "management_row_key_v1": "row-hold",
                "trade_uid_exact_v1": "trade-1",
                "candidate_uid_exact_v1": "cand-1",
                "decision_timestamp": "2026-04-22T10:00:00Z",
                "action_label_v1": "HOLD",
                "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "bandit_action_reward_eligibility_status_v1": "BANDIT_DM_ELIGIBLE",
                "sequence_dataset_membership_v1": "BANDIT_SAFE_ONLY",
                "observation_contract_v1": "MANAGEMENT_RL_OBSERVATION_CONTRACT_V1",
                "terminal_outcome_availability_status_v1": "EXACT_TERMINAL_OUTCOME_AVAILABLE",
                "hindsight_reward_realized_pnl_bps_v1": 12.5,
                "as_of_atr_bps_v1": 50.0,
                "as_of_session_v1": "NY",
                "as_of_management_exit_prob_v1": None,
                "as_of_management_exit_prob_available_v1": 0,
            },
            {
                "management_row_key_v1": "row-exit",
                "trade_uid_exact_v1": "trade-2",
                "candidate_uid_exact_v1": "cand-2",
                "decision_timestamp": "2026-04-22T11:00:00Z",
                "action_label_v1": "EXIT_NOW",
                "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "bandit_action_reward_eligibility_status_v1": "BANDIT_DM_ELIGIBLE",
                "sequence_dataset_membership_v1": "STRICT_SEQUENCE_SUBSTRATE",
                "observation_contract_v1": "MANAGEMENT_RL_OBSERVATION_CONTRACT_V1",
                "terminal_outcome_availability_status_v1": "EXACT_TERMINAL_OUTCOME_AVAILABLE",
                "hindsight_reward_realized_pnl_bps_v1": -3.0,
                "as_of_atr_bps_v1": 60.0,
                "as_of_session_v1": "LONDON",
                "as_of_management_exit_prob_v1": 0.4,
                "as_of_management_exit_prob_available_v1": 1,
            },
        ]
    )
    dm.to_parquet(management_dir / "shadow_meta_all_trade_review_management_bandit_direct_method_candidate_view_v1.parquet", index=False)

    policy = pd.DataFrame(
        [
            {
                "management_row_key_v1": "row-hold",
                "candidate_uid_exact_v1": "cand-1",
                "policy_version_v1": "policy-a",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_kind_v1": "DETERMINISTIC_VERSIONED_LOGGED_ACTION_POLICY",
                "behavior_policy_id_status_v1": "EXACT_POLICY_HASH_BEHAVIOR_POLICY_ID",
                "policy_logging_propensity_status_v1": "DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT",
                "observed_action_v1": "HOLD",
                "support_tier_v1": "MIXED_FEATURE_SUPPORT",
            },
            {
                "management_row_key_v1": "row-exit",
                "candidate_uid_exact_v1": "cand-2",
                "policy_version_v1": "policy-a",
                "behavior_policy_id_v1": "policy-a",
                "behavior_policy_kind_v1": "DETERMINISTIC_VERSIONED_LOGGED_ACTION_POLICY",
                "behavior_policy_id_status_v1": "EXACT_POLICY_HASH_BEHAVIOR_POLICY_ID",
                "policy_logging_propensity_status_v1": "DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT",
                "observed_action_v1": "EXIT_NOW",
                "support_tier_v1": "STRONG_FEATURE_SUPPORT",
            },
        ]
    )
    policy.to_parquet(policy_dir / "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet", index=False)

    _write_json(
        reward_lock / "lock_first_bandit_reward_version_summary_v1.json",
        {
            "reward_lock_succeeded_v1": True,
            "reward_version_id_v1": REWARD_VERSION_ID,
            "reward_lock_verdict_v1": "LOCKED_FOR_MANAGEMENT_BANDIT_RESEARCH_ONLY",
        },
    )
    _write_json(
        reward_lock / "first_bandit_reward_contract_v1.json",
        {
            "contract_id_v1": "FIRST_BANDIT_REWARD_CONTRACT_V1",
            "reward_version_id_v1": REWARD_VERSION_ID,
            "verdict_v1": "LOCKED_FOR_MANAGEMENT_BANDIT_RESEARCH_ONLY",
        },
    )
    _write_json(contract_lock / "iql_management_bandit_dataset_contract_lock_v1.json", {"contract_id_v1": "MANAGEMENT_BANDIT_DATASET_CONTRACT_LOCK_V1"})
    _write_json(contract_lock / "iql_baseline_comparator_and_failcheck_lock_v1.json", {"lock_id_v1": "BASELINE_COMPARATOR_AND_FAILCHECK_LOCK_V1"})
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return foundation, reward_lock, contract_lock


def test_build_management_bandit_dataset_is_bandit_only_and_readonly(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    foundation, reward_lock, contract_lock = _write_fixture(reports_root)
    output_dir = reports_root / "IQL_INTEGRATION" / "BUILD_MANAGEMENT_BANDIT_DATASET_V1_FIXTURE"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_management_bandit_dataset(
        reports_root,
        foundation_dir=foundation,
        reward_lock_dir=reward_lock,
        contract_lock_dir=contract_lock,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
        r6_sha_before="same",
        r6_sha_after="same",
    )

    assert len(payload["dataset_df"]) == 2
    assert payload["summary"]["excluded_rows_v1"] == 0
    assert set(payload["dataset_df"]["action"]) == {"HOLD", "EXIT_NOW"}
    assert payload["dataset_df"]["reward_version"].eq(REWARD_VERSION_ID).all()
    assert "next_state" not in payload["dataset_df"].columns
    assert "next_state_vector" not in payload["dataset_df"].columns
    assert payload["profile"]["verdict_v1"] == "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS"
    assert payload["non_interference"]["failed_check_count_v1"] == 0
    assert payload["consistency"]["failed_check_count_v1"] == 0

    result = write_management_bandit_dataset_artifacts(
        reports_root,
        foundation_dir=foundation,
        reward_lock_dir=reward_lock,
        contract_lock_dir=contract_lock,
        output_dir=output_dir,
        built_at=built_at,
    )

    dataset_path = Path(result["artifact_paths"]["dataset_parquet"])
    assert dataset_path.exists()
    written = pd.read_parquet(dataset_path)
    assert len(written) == 2
    assert result["summary"]["included_rows_v1"] == 2
    assert result["summary"]["sequence_iql_still_blocked_v1"] is True
    assert result["status"]["training_executed_v1"] is False
