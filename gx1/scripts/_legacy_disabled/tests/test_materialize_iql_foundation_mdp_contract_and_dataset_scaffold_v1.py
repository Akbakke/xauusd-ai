from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_iql_foundation_mdp_contract_and_dataset_scaffold_v1 import (
    BANDIT_DM_VIEW_FILE,
    BANDIT_EXIT_LOCAL_VIEW_FILE,
    BANDIT_HOLD_RETURN_VIEW_FILE,
    BANDIT_OBSERVED_VIEW_FILE,
    BANDIT_STATUS_FILE,
    LEDGER_FILE,
    OBSERVATION_CONTRACT_FILE,
    POLICY_LOG_FILE,
    SEQUENCE_ROW_VIEW_FILE,
    SEQUENCE_STATUS_FILE,
    STRICT_TRANSITION_VIEW_FILE,
    build_iql_foundation,
    write_iql_foundation_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _base_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.DataFrame(
        [
            {
                "trade_uid": "t1",
                "candidate_uid": "c1",
                "realized_pnl_bps": 10.0,
                "mfe_bps": 20.0,
                "mae_bps": 4.0,
                "hindsight_peak_mfe_bps_v1": 20.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 10.0,
                "hindsight_hold_longer_extra_value_bps_v1": 0.0,
                "bad_trade": False,
                "cata_loser": False,
            },
            {
                "trade_uid": "t2",
                "candidate_uid": "c2",
                "realized_pnl_bps": -5.0,
                "mfe_bps": 8.0,
                "mae_bps": 12.0,
                "hindsight_peak_mfe_bps_v1": 8.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 13.0,
                "hindsight_hold_longer_extra_value_bps_v1": 7.0,
                "bad_trade": True,
                "cata_loser": False,
            },
            {
                "trade_uid": "t3",
                "candidate_uid": "c3",
                "realized_pnl_bps": 2.0,
                "mfe_bps": 12.0,
                "mae_bps": 3.0,
                "hindsight_peak_mfe_bps_v1": 12.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 10.0,
                "hindsight_hold_longer_extra_value_bps_v1": 1.0,
                "bad_trade": False,
                "cata_loser": False,
            },
        ]
    )
    seq = pd.DataFrame(
        [
            {
                "management_row_key_v1": "r1",
                "sequence_episode_key_v1": "e1",
                "sequence_step_index_v1": 0,
                "sequence_step_action_v1": "EXIT_NOW",
                "sequence_next_row_key_v1": pd.NA,
                "sequence_next_link_status_v1": "TERMINAL_EXIT_STEP",
                "sequence_terminal_step_status_v1": "TERMINAL_REALIZED_EXIT",
                "sequence_dataset_membership_v1": "STRICT_SEQUENCE_SUBSTRATE",
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "id1",
                "run_id": "run1",
                "as_of_row_uid_v1": "r1",
                "decision_timestamp": "2026-01-01T00:00:00+00:00",
                "action_label_v1": "EXIT_NOW",
                "as_of_session_v1": "US",
                "as_of_vol_regime_v1": "HIGH",
                "as_of_trend_regime_v1": "TREND",
                "as_of_side_v1": "long",
                "as_of_atr_bps_v1": 1.0,
                "as_of_hour_utc_v1": 0,
                "as_of_weekday_utc_v1": 1,
                "as_of_management_core_minutes_held_at_anchor_v1": 10.0,
                "terminal_realized_pnl_bps_v1": 10.0,
            },
            {
                "management_row_key_v1": "r2",
                "sequence_episode_key_v1": "e2",
                "sequence_step_index_v1": 0,
                "sequence_step_action_v1": "HOLD",
                "sequence_next_row_key_v1": pd.NA,
                "sequence_next_link_status_v1": "NO_EXACT_NEXT_ELIGIBLE_STEP",
                "sequence_terminal_step_status_v1": "NON_TERMINAL_HOLD",
                "sequence_dataset_membership_v1": "BANDIT_SAFE_ONLY",
                "candidate_uid_exact_v1": "c2",
                "trade_uid_exact_v1": "t2",
                "trade_id_exact_v1": "id2",
                "run_id": "run1",
                "as_of_row_uid_v1": "r2",
                "decision_timestamp": "2026-01-01T00:01:00+00:00",
                "action_label_v1": "HOLD",
                "as_of_session_v1": "US",
                "as_of_vol_regime_v1": "HIGH",
                "as_of_trend_regime_v1": "TREND",
                "as_of_side_v1": "long",
                "as_of_atr_bps_v1": 1.0,
                "as_of_hour_utc_v1": 0,
                "as_of_weekday_utc_v1": 1,
                "as_of_management_core_minutes_held_at_anchor_v1": 20.0,
                "terminal_realized_pnl_bps_v1": -5.0,
            },
            {
                "management_row_key_v1": "r3",
                "sequence_episode_key_v1": "e3",
                "sequence_step_index_v1": 0,
                "sequence_step_action_v1": "EXIT_NOW",
                "sequence_next_row_key_v1": pd.NA,
                "sequence_next_link_status_v1": "EPISODE_LINK_UNRESOLVED",
                "sequence_terminal_step_status_v1": "TERMINAL_STATUS_UNRESOLVED",
                "sequence_dataset_membership_v1": "SEQUENCE_INELIGIBLE",
                "candidate_uid_exact_v1": "c3",
                "trade_uid_exact_v1": "t3",
                "trade_id_exact_v1": "id3",
                "run_id": "run1",
                "as_of_row_uid_v1": "r3",
                "decision_timestamp": "2026-01-01T00:02:00+00:00",
                "action_label_v1": "EXIT_NOW",
                "as_of_session_v1": "US",
                "as_of_vol_regime_v1": "LOW",
                "as_of_trend_regime_v1": "TREND",
                "as_of_side_v1": "long",
                "as_of_atr_bps_v1": 1.0,
                "as_of_hour_utc_v1": 0,
                "as_of_weekday_utc_v1": 1,
                "as_of_management_core_minutes_held_at_anchor_v1": 40.0,
                "terminal_realized_pnl_bps_v1": 2.0,
            },
        ]
    )
    return ledger, seq


def test_iql_foundation_scaffold_stops_before_training_and_reports_hold_gap(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    management_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_REWARD_CHANNEL_FIX_R1_CANONICAL"
    policy_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260420T195402Z_MANAGEMENT_AUDIT_EXTENSION_V1"
    management_dir.mkdir(parents=True)
    policy_dir.mkdir(parents=True)

    ledger, seq = _base_rows()
    dm = seq.loc[seq["sequence_dataset_membership_v1"].isin(["STRICT_SEQUENCE_SUBSTRATE", "BANDIT_SAFE_ONLY"])].copy()
    dm["hindsight_reward_realized_pnl_bps_v1"] = dm["terminal_realized_pnl_bps_v1"]
    dm["observed_action_status_v1"] = "OBSERVED_REALIZED_PATH_ACTION_EXACT"
    dm["observed_action_propensity_status_v1"] = "PROPENSITY_NOT_ESTABLISHED"
    dm["bandit_action_reward_eligibility_status_v1"] = "BANDIT_DM_ELIGIBLE"
    dm["bandit_reward_locality_status_v1"] = [
        "LOCAL_EXIT_ACTION_WITH_EXACT_TERMINAL_OUTCOME",
        "HOLD_WITH_EPISODE_TERMINAL_RETURN_ONLY",
    ]

    ledger.to_parquet(management_dir / LEDGER_FILE, index=False)
    seq.to_parquet(management_dir / SEQUENCE_ROW_VIEW_FILE, index=False)
    seq.loc[seq["sequence_dataset_membership_v1"].eq("STRICT_SEQUENCE_SUBSTRATE")].to_parquet(
        management_dir / STRICT_TRANSITION_VIEW_FILE,
        index=False,
    )
    seq.to_parquet(management_dir / BANDIT_OBSERVED_VIEW_FILE, index=False)
    dm.to_parquet(management_dir / BANDIT_DM_VIEW_FILE, index=False)
    dm.loc[dm["action_label_v1"].eq("EXIT_NOW")].to_parquet(management_dir / BANDIT_EXIT_LOCAL_VIEW_FILE, index=False)
    dm.loc[dm["action_label_v1"].eq("HOLD")].to_parquet(management_dir / BANDIT_HOLD_RETURN_VIEW_FILE, index=False)

    _write_json(
        management_dir / OBSERVATION_CONTRACT_FILE,
        {
            "observation_vector_feature_names_v1": [
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
        },
    )
    _write_json(
        management_dir / SEQUENCE_STATUS_FILE,
        {"MANAGEMENT_RL_SEQUENCE_BLOCKER_STATUS": "NEXT_STEP_LINKS_MISSING_FOR_SOME_REALIZED_HOLD_ROWS"},
    )
    _write_json(
        management_dir / BANDIT_STATUS_FILE,
        {
            "MANAGEMENT_BANDIT_STATUS": "BANDIT_ACTION_REWARD_SUBSTRATE_ONLY",
            "MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1": 2,
        },
    )
    dm.assign(policy_version_v1="hash1", observed_action_v1=dm["action_label_v1"]).to_parquet(
        policy_dir / POLICY_LOG_FILE,
        index=False,
    )
    _write_json(reports_root / "truth_entry_rl_observability_v1.json", {"observed_direct_entry_rows_v1": 2})
    _write_json(reports_root / "truth_harvest_retrain_candidate_v1.json", {"candidate_count_v1": 3})
    _write_json(reports_root / "truth_rl_unified_observability_v1.json", {"entry_episode_rows_v1": 3})
    _write_json(reports_root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    _write_json(reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json", {"freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"})

    payload = build_iql_foundation(reports_root, management_dir=management_dir, policy_log_dir=policy_dir)

    assert payload["summary"]["management_mdp_verdict_v1"] == "PARTIAL_TRANSITIONS"
    assert payload["summary"]["full_sequence_ready_transition_count_v1"] == 1
    assert payload["summary"]["bandit_only_row_count_v1"] == 1
    assert payload["summary"]["hold_to_next_state_transition_count_v1"] == 0
    assert payload["training_harness"]["status_v1"] == "NOT_READY_FOR_IQL_TRAINING"
    assert payload["training_harness"]["training_executed_v1"] is False
    assert payload["consistency_audit_df"].set_index("check_name_v1").loc[
        "AS_OF_STATE_EXCLUDES_HINDSIGHT_AND_REWARD_FIELDS", "status_v1"
    ] == "PASS"
    assert payload["consistency_audit_df"].set_index("check_name_v1").loc[
        "CANONICAL_MANAGEMENT_CORE_9_INPUTS_INCLUDED", "status_v1"
    ] == "PASS"
    assert payload["dataset_schema"]["canonical_management_core_9_inputs_v1"]["status_v1"] == "READY"
    usable = set(
        payload["reward_audit_df"].loc[
            payload["reward_audit_df"]["verdict_v1"].eq("USABLE_FOR_OFFLINE_RESEARCH"),
            "reward_candidate_v1",
        ]
    )
    assert "REALIZED_PNL_REWARD" in usable

    result = write_iql_foundation_artifacts(
        reports_root,
        output_dir=tmp_path / "out",
        management_dir=management_dir,
        policy_log_dir=policy_dir,
    )
    assert Path(result["artifact_paths"]["summary"]).exists()
    assert Path(result["artifact_paths"]["training_harness"]).exists()
