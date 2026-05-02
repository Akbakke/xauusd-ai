from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_iql_reward_comparator_and_bandit_contract_lock_v1 import (
    LAYER_ID,
    build_contract_lock,
    write_contract_lock_artifacts,
)
from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import FOUNDATION_DIRNAME


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path]:
    foundation = root / FOUNDATION_DIRNAME
    readonly = root / "IQL_READINESS" / "IQL_READONLY_TRANSITION_REWARD_BANDIT_PLANNING_V1_FIXTURE"
    foundation.mkdir(parents=True)
    readonly.mkdir(parents=True)

    _write_json(root / "truth_iql_foundation_mdp_contract_and_dataset_scaffold_v1.json", {"artifact_dir_v1": str(foundation)})
    _write_json(
        foundation / "iql_foundation_summary_v1.json",
        {
            "management_mdp_verdict_v1": "PARTIAL_TRANSITIONS",
            "full_sequence_ready_transition_count_v1": 45,
            "bandit_only_row_count_v1": 1751,
            "hold_to_next_state_transition_count_v1": 0,
            "support_ood_verdict_v1": "SUPPORT_TOO_THIN",
            "training_harness_status_v1": "NOT_READY_FOR_IQL_TRAINING",
        },
    )
    _write_json(
        foundation / "iql_foundation_mdp_contract_v1.json",
        {"source_truth_v1": {"locked_ledger_source_file_v1": str(root / "ledger.parquet")}},
    )
    _write_json(
        foundation / "iql_foundation_dataset_schema_v1.json",
        {
            "state_feature_names_v1": [
                "as_of_atr_bps_v1",
                "as_of_hour_utc_v1",
                "as_of_session_v1",
                "as_of_side_v1",
                "as_of_trend_regime_v1",
                "as_of_vol_regime_v1",
                "as_of_weekday_utc_v1",
                "as_of_management_core_minutes_held_at_anchor_v1",
                "as_of_management_core_giveback_ratio_from_peak_v1",
            ],
            "canonical_management_core_9_inputs_v1": {"status_v1": "READY"},
            "fields_v1": [
                {"field_name_v1": "episode_id", "status_v1": "READY"},
                {"field_name_v1": "state_vector", "status_v1": "PARTIAL"},
                {"field_name_v1": "state_feature_names", "status_v1": "READY"},
                {"field_name_v1": "action", "status_v1": "READY"},
                {"field_name_v1": "action_id", "status_v1": "READY"},
                {"field_name_v1": "reward", "status_v1": "PARTIAL"},
                {"field_name_v1": "decision_ts", "status_v1": "READY"},
                {"field_name_v1": "candidate_uid_exact", "status_v1": "READY"},
                {"field_name_v1": "source_policy_version", "status_v1": "READY"},
                {"field_name_v1": "behavior_policy_status", "status_v1": "PARTIAL"},
                {"field_name_v1": "support_status", "status_v1": "PARTIAL"},
                {"field_name_v1": "as_of_schema_version", "status_v1": "READY"},
                {"field_name_v1": "reward_version", "status_v1": "NOT_ESTABLISHED"},
                {"field_name_v1": "outcome_backfill_version", "status_v1": "READY"},
            ],
        },
    )
    _write_json(
        foundation / "iql_foundation_management_mdp_contract_v1.json",
        {
            "state_contract_v1": {
                "policy_log_fields_v1": ["policy_version_v1", "observed_action_v1"],
                "path_dynamics_optional_fields_v1": [],
            }
        },
    )
    _write_json(
        foundation / "iql_foundation_baseline_comparator_spec_v1.json",
        {
            "baseline_calibration_status_v1": "PENDING_EXTERNAL_CALIBRATION",
            "baseline_comparator_presence_v1": {},
        },
    )
    pd.DataFrame(
        [
            {
                "reward_candidate_v1": "REALIZED_PNL_REWARD",
                "draft_status_v1": "LOCKABLE_AFTER_REVIEW",
                "formula_v1": "terminal_realized_pnl_bps",
                "coverage_rate_v1": 1.0,
                "distribution_count_v1": 1796,
                "leakage_risk_v1": "LOW_IF_USED_ONLY_AS_REWARD",
            },
            {
                "reward_candidate_v1": "RUNNER_DAMAGE_PENALTY",
                "draft_status_v1": "AUDIT_ONLY",
                "formula_v1": "-max(hindsight_hold_longer_extra_value_bps, 0)",
                "coverage_rate_v1": 1.0,
                "distribution_count_v1": 1796,
                "leakage_risk_v1": "HIGH_COUNTERFACTUAL_HINDSIGHT_LOCALITY_NOT_LOCKED",
            },
        ]
    ).to_csv(readonly / "iql_readonly_reward_contract_draft_v1.csv", index=False)
    _write_json(
        readonly / "iql_readonly_summary_v1.json",
        {
            "management_status_v1": "PARTIAL_TRANSITIONS",
            "training_harness_status_v1": "NOT_READY_FOR_IQL_TRAINING",
        },
    )
    _write_json(
        readonly / "iql_readonly_r5_2_r6_r7_boundary_lock_v1.json",
        {"r7_v1": {"status_v1": "NOT_STARTED"}},
    )
    _write_json(root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json", {"freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"})
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return foundation, readonly


def test_contract_lock_is_readonly_and_fail_closed(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    foundation, readonly = _write_fixture(reports_root)
    output_dir = reports_root / "IQL_INTEGRATION" / "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1_FIXTURE"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_contract_lock(
        reports_root,
        foundation_dir=foundation,
        readonly_dir=readonly,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
    )

    assert payload["contract"]["contract_id_v1"] == LAYER_ID
    assert payload["reward_aggregate"]["scalar_reward_version_locked_now_v1"] is False
    assert payload["reward_aggregate"]["first_bandit_reward_version_lock_next_v1"] is True
    assert payload["bandit_contract"]["verdict_v1"] == "READY_TO_BUILD_AFTER_REWARD_LOCK"
    assert payload["bandit_contract"]["dataset_build_executed_v1"] is False
    assert payload["r7_lock"]["short_verdict_v1"] == "R7_STILL_BLOCKED_DO_NOT_START"
    assert payload["summary"]["iql_training_started_v1"] is False
    assert payload["non_interference"]["failed_check_count_v1"] == 0
    assert payload["consistency_df"]["status_v1"].eq("PASS").all()

    result = write_contract_lock_artifacts(
        reports_root,
        foundation_dir=foundation,
        readonly_dir=readonly,
        output_dir=output_dir,
        built_at=built_at,
    )

    assert Path(result["artifact_paths"]["summary"]).exists()
    assert Path(result["artifact_paths"]["bandit_contract"]).exists()
    assert result["status"]["training_executed_v1"] is False
    assert result["status"]["r7_started_v1"] is False
    assert result["status"]["replay_touched_v1"] is False
