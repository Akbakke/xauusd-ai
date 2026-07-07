from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts import materialize_monday_management_policy_logging_runtime_v1 as script


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_build_runtime_only_policy_logging_payload_smoke() -> None:
    observed_sample_df = pd.DataFrame(
        [
            {
                "management_row_key_v1": "mrk-1",
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "tid1",
                "run_id": "TRUTH_MONFRI_WEEK_20260105_20260112",
                "as_of_row_uid_v1": "asof-1",
                "decision_timestamp": "2026-01-06T10:00:00Z",
                "action_label_v1": "HOLD",
                "decision_anchor_type_v1": "MANAGEMENT_EXIT_ANCHOR",
                "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "observed_action_source_v1": "REALIZED_PATH",
                "route_status_v1": "ROUTE_OK",
                "entry_actualization_presence_status_v1": "ENTRY_PRESENT",
                "rl_transition_eligibility_status_v1": "ELIGIBLE",
                "management_path_relation_v1": "DIRECT",
                "sequence_dataset_membership_v1": "STRICT_SEQUENCE",
                "sequence_next_link_status_v1": "HAS_NEXT",
                "sequence_terminal_step_status_v1": "NON_TERMINAL",
                "split_bucket_v1": "TRAIN",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "activation_origin_v1": "DIRECT",
            }
        ]
    )
    direct_method_df = observed_sample_df.copy()
    direct_method_df["as_of_session_v1"] = "US"
    direct_method_df["as_of_weekday_utc_v1"] = 2
    direct_method_df["as_of_atr_bps_v1"] = 12.0
    direct_method_df["as_of_hour_utc_v1"] = 10
    direct_method_df["as_of_side_v1"] = "LONG"
    direct_method_df["as_of_trend_regime_v1"] = "TREND"
    direct_method_df["as_of_vol_regime_v1"] = "EXTREME"
    direct_method_df["as_of_management_core_entry_spread_bps_v1"] = 3.0
    direct_method_df["as_of_management_core_minutes_held_at_anchor_v1"] = 12.0
    direct_method_df["as_of_management_core_giveback_ratio_from_peak_v1"] = 0.4
    direct_method_df["as_of_management_core_minutes_since_last_peak_v1"] = 4.0
    direct_method_df["as_of_management_core_minutes_since_last_mfe_v1"] = 2.0
    direct_method_df["as_of_management_core_peak_to_anchor_bps_v1"] = 10.0
    direct_method_df["as_of_management_core_mfe_to_anchor_ratio_v1"] = 1.5
    direct_method_df["as_of_management_core_mfe_to_anchor_ratio_available_v1"] = True
    direct_method_df["as_of_management_core_mfe_bps_so_far_v1"] = 15.0
    direct_method_df["as_of_management_core_mae_bps_so_far_v1"] = -2.0
    direct_method_df["as_of_management_core_distance_from_peak_mfe_bps_v1"] = 5.0
    direct_method_df["as_of_management_core_bars_since_peak_mfe_v1"] = 2.0
    direct_method_df["as_of_management_replay_micro_momentum_3_v1"] = 0.1
    direct_method_df["as_of_management_replay_micro_momentum_5_v1"] = 0.2
    direct_method_df["as_of_management_replay_micro_acceleration_v1"] = 0.05
    direct_method_df["as_of_management_replay_wick_ratio_v1"] = 0.3
    direct_method_df["as_of_management_replay_retracement_from_last_impulse_v1"] = 0.4
    direct_method_df["as_of_management_replay_minutes_since_session_open_v1"] = 60.0
    direct_method_df["as_of_management_replay_minutes_to_next_session_boundary_v1"] = 120.0
    direct_method_df["as_of_management_replay_session_change_flag_v1"] = False
    direct_method_df["as_of_management_replay_session_tradable_v1"] = True
    direct_method_df["as_of_management_exit_model_evaluated_v1"] = True
    direct_method_df["as_of_management_exit_prob_v1"] = 0.22
    direct_method_df["as_of_management_exit_prob_available_v1"] = True
    direct_method_df["as_of_management_exit_threshold_v1"] = 0.5
    direct_method_df["as_of_management_candidate_p_long_v1"] = 0.7
    direct_method_df["as_of_management_candidate_p_short_v1"] = 0.1
    direct_method_df["as_of_management_candidate_p_flat_v1"] = 0.2
    direct_method_df["as_of_management_candidate_p_hat_v1"] = 0.7
    direct_method_df["as_of_management_candidate_margin_v1"] = 0.5
    direct_method_df["as_of_management_candidate_uncertainty_score_v1"] = 0.1
    direct_method_df["as_of_management_candidate_tradable_prob_v1"] = 0.9
    direct_method_df["as_of_management_candidate_mfe_first_n_pred_v1"] = 12.0
    direct_method_df["as_of_management_candidate_path_quality_pred_v1"] = 0.8
    direct_method_df["as_of_management_xgb_p_long_v1"] = 0.6
    direct_method_df["as_of_management_xgb_p_short_v1"] = 0.2
    direct_method_df["as_of_management_xgb_p_flat_v1"] = 0.2
    direct_method_df["as_of_management_xgb_p_hat_v1"] = 0.6
    direct_method_df["as_of_management_xgb_pred_side_v1"] = "LONG"
    direct_method_df["as_of_management_xgb_has_ctx_v1"] = True
    direct_method_df["hindsight_reward_realized_pnl_bps_v1"] = 11.0
    direct_method_df["hindsight_reward_trade_outcome_class_v1"] = "WIN"
    direct_method_df["hindsight_reward_good_trade_v1"] = True
    direct_method_df["hindsight_reward_good_trade_mfe20_mae5_v1"] = True
    direct_method_df["hindsight_reward_bad_trade_v1"] = False
    direct_method_df["hindsight_reward_good_exit_v1"] = True
    direct_method_df["hindsight_reward_premature_exit_v1"] = False
    direct_method_df["hindsight_reward_late_exit_v1"] = False
    direct_method_df["hindsight_reward_exit_reason_v1"] = "THRESHOLD"
    direct_method_df["bandit_candidate_mode_v1"] = "DIRECT_METHOD"
    direct_method_df["observation_contract_v1"] = "READY"
    direct_method_df["propensity_contract_v1"] = "PARTIAL"

    eligible_df = direct_method_df[
        [
            "management_row_key_v1",
            "candidate_uid_exact_v1",
            "trade_uid_exact_v1",
            "trade_id_exact_v1",
            "decision_anchor_type_v1",
            "as_of_session_v1",
            "as_of_vol_regime_v1",
            "as_of_management_core_minutes_held_at_anchor_v1",
            "as_of_management_core_giveback_ratio_from_peak_v1",
        ]
    ].copy()
    eligible_df["primary_model_name_v1"] = "TREE_REGRESSION_BASELINE"
    eligible_df["primary_model_score_v1"] = 0.91
    eligible_df["primary_model_score_rank_within_split_v1"] = 1

    raw_state_df = pd.DataFrame(
        [
            {
                "as_of_row_uid_v1": "asof-1",
                "run_id": "TRUTH_MONFRI_WEEK_20260105_20260112",
                "decision_timestamp": "2026-01-06T10:00:00Z",
                "anchor_timestamp_utc": "2026-01-06T10:00:00Z",
                "as_of_mgmt_trace_last_peak_ts_utc_v1": "2026-01-06T09:56:00Z",
                "as_of_mgmt_trace_last_mfe_ts_utc_v1": "2026-01-06T09:58:00Z",
                "as_of_mgmt_trace_peak_price_v1": 2750.0,
                "as_of_mgmt_trace_anchor_price_v1": 2749.0,
                "as_of_mgmt_trace_mfe_bps_at_anchor_v1": 15.0,
                "as_of_mgmt_trace_last_peak_mfe_bps_v1": 20.0,
                "as_of_mgmt_trace_max_mfe_without_mae_bps_v1": 18.0,
                "as_of_mgmt_trace_mfe_mae_sequence_order_v1": "MFE_FIRST",
                "as_of_mgmt_trace_last_peak_ts_utc_null_reason_v1": "NONE",
                "as_of_mgmt_trace_last_mfe_ts_utc_null_reason_v1": "NONE",
                "as_of_mgmt_trace_last_peak_mfe_bps_null_reason_v1": "NONE",
                "as_of_mgmt_trace_max_mfe_without_mae_bps_null_reason_v1": "NONE",
                "as_of_mgmt_trace_mfe_mae_sequence_order_null_reason_v1": "NONE",
            }
        ]
    )

    as_of_df = pd.DataFrame(
        [
            {
                "candidate_uid": "c1",
                "trade_uid": "t1",
                "trade_id": "tid1",
                "decision_anchor_type_v1": "MANAGEMENT_EXIT_ANCHOR",
                "as_of_trade_pocket_v1": "MID",
                "run_id": "TRUTH_MONFRI_WEEK_20260105_20260112",
                "as_of_row_uid_v1": "asof-1",
                "as_of_candidate_policy_hash_v1": "policy-123",
                "as_of_candidate_entry_bundle_sha256_v1": "entry-sha",
                "as_of_candidate_exit_bundle_sha256_v1": "exit-sha",
            }
        ]
    )

    closed_trades_df = pd.DataFrame(
        [
            {
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "tid1",
                "entry_timestamp": "2026-01-06T09:00:00Z",
                "exit_timestamp": "2026-01-06T11:00:00Z",
                "realized_pnl_bps": 11.0,
                "mfe_bps": 25.0,
                "mae_bps": -3.0,
                "holding_time_bars": 12,
                "trade_outcome_class": "WIN",
                "exit_reason": "THRESHOLD",
                "good_exit": True,
                "premature_exit": False,
                "late_exit": False,
                "hindsight_management_review_v1": "GOOD",
                "hindsight_peak_mfe_bps_v1": 25.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 4.0,
                "policy_hash": "policy-123",
            }
        ]
    )

    hindsight_review_export_df = pd.DataFrame(
        [
            {
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "tid1",
                "review_exit_bucket_v1": "GOOD_EXIT",
                "review_entry_bucket_v1": "GOOD_ENTRY",
                "hindsight_rl_review_reason_v1": "OK",
                "hindsight_rl_review_domain_support_v1": "STRONG",
                "hindsight_peak_to_worst_after_peak_bps_v1": 2.0,
            }
        ]
    )

    payload = script._build_runtime_only_policy_logging_payload(
        observed_sample_df=observed_sample_df,
        direct_method_df=direct_method_df,
        eligible_df=eligible_df,
        raw_state_df=raw_state_df,
        as_of_df=script.shadow_meta._rename_exact_join_ids_v1(as_of_df),
        closed_trades_df=closed_trades_df,
        hindsight_review_export_df=hindsight_review_export_df,
        management_bandit_action_reward_contract_v1={"action_space_v1": ["HOLD", "EXIT_NOW"], "layer_name": "TEST"},
        management_bandit_observed_action_contract_v1={"layer_name": "OBSERVED_ACTION_CONTRACT"},
        management_bandit_status_v1={"MANAGEMENT_BANDIT_PROPENSITY_STATUS": "PROPENSITY_NOT_ESTABLISHED"},
        as_of_supervision_join_coverage_summary={"exact_rows": 1, "fallback_rows": 0, "unjoinable_rows": 0},
        leakage_guard_summary={"status": "PASS"},
        build_id_v1="BUILD_TEST",
        build_timestamp_utc_v1="2026-04-24T10:00:00+00:00",
        source_control_date_v1="2026-04-11",
    )

    decision_df = payload["management_policy_logging_decision_log_harness_v1_df"]
    outcome_df = payload["management_policy_logging_outcome_backfill_harness_v1_df"]
    consistency_df = payload["management_policy_logging_consistency_audit_v1_df"]

    assert len(decision_df) == 1
    assert len(outcome_df) == 1
    assert float(decision_df.loc[0, "shadow_score_v1"]) == 0.91
    assert decision_df.loc[0, "overlay_composite_v1"] == "US|EXTREME|EARLY_0_30M|LOW_LT_0P50"
    assert decision_df.loc[0, "path_dynamics_raw_state_join_mode_v1"] == "AS_OF_ROW_UID_EXACT"
    assert decision_df.loc[0, "as_of_management_core_last_peak_mfe_bps_v1"] == 20.0
    assert decision_df.loc[0, "manual_review_provenance_v1"] == "RETIRED_MANUAL_REVIEW_CONTEXT_NOT_REBUILT"
    assert decision_df.loc[0, "behavior_policy_id_v1"] == "policy-123"
    assert outcome_df.loc[0, "outcome_backfill_status_v1"] == "EXACT_TERMINAL_OUTCOME_BACKFILL"
    assert (consistency_df["status_v1"] == "PASS").all()


def test_materialize_writes_policy_logging_and_skips_follow_on(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    ledger_dir.mkdir()

    observed = pd.DataFrame(
        [
            {
                "management_row_key_v1": "mrk-1",
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "tid1",
                "run_id": "run-1",
                "as_of_row_uid_v1": "asof-1",
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "action_label_v1": "HOLD",
                "decision_anchor_type_v1": "MANAGEMENT_EXIT_ANCHOR",
                "split_bucket_v1": "TRAIN",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "as_of_session_v1": "US",
                "as_of_weekday_utc_v1": 3,
                "activation_origin_v1": "DIRECT",
                "route_status_v1": "ROUTE_OK",
                "entry_actualization_presence_status_v1": "ENTRY_PRESENT",
                "rl_transition_eligibility_status_v1": "ELIGIBLE",
                "management_path_relation_v1": "DIRECT",
                "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "observed_action_source_v1": "REALIZED_PATH",
                "observed_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
                "bandit_action_reward_eligibility_status_v1": "ELIGIBLE",
                "bandit_reward_locality_status_v1": "LOCAL",
                "terminal_outcome_availability_status_v1": "AVAILABLE",
                "sequence_dataset_membership_v1": "STRICT_SEQUENCE",
                "sequence_next_link_status_v1": "HAS_NEXT",
                "sequence_terminal_step_status_v1": "NON_TERMINAL",
                "as_of_atr_bps_v1": 10.0,
                "as_of_hour_utc_v1": 12,
                "as_of_side_v1": "LONG",
                "as_of_trend_regime_v1": "TREND",
                "as_of_vol_regime_v1": "EXTREME",
                "as_of_management_core_entry_spread_bps_v1": 2.0,
                "as_of_management_core_minutes_held_at_anchor_v1": 10.0,
                "as_of_management_core_giveback_ratio_from_peak_v1": 0.2,
                "as_of_management_core_minutes_since_last_peak_v1": 1.0,
                "as_of_management_core_minutes_since_last_mfe_v1": 1.0,
                "as_of_management_core_peak_to_anchor_bps_v1": 4.0,
                "as_of_management_core_mfe_to_anchor_ratio_v1": 1.2,
                "as_of_management_core_mfe_to_anchor_ratio_available_v1": True,
                "as_of_management_core_mfe_bps_so_far_v1": 8.0,
                "as_of_management_core_mae_bps_so_far_v1": -1.0,
                "as_of_management_core_distance_from_peak_mfe_bps_v1": 2.0,
                "as_of_management_core_bars_since_peak_mfe_v1": 1.0,
                "as_of_management_replay_micro_momentum_3_v1": 0.1,
                "as_of_management_replay_micro_momentum_5_v1": 0.1,
                "as_of_management_replay_micro_acceleration_v1": 0.1,
                "as_of_management_replay_wick_ratio_v1": 0.1,
                "as_of_management_replay_retracement_from_last_impulse_v1": 0.1,
                "as_of_management_replay_minutes_since_session_open_v1": 20.0,
                "as_of_management_replay_minutes_to_next_session_boundary_v1": 100.0,
                "as_of_management_replay_session_change_flag_v1": False,
                "as_of_management_replay_session_tradable_v1": True,
                "as_of_management_exit_model_evaluated_v1": True,
                "as_of_management_exit_prob_v1": 0.2,
                "as_of_management_exit_prob_available_v1": True,
                "as_of_management_exit_threshold_v1": 0.5,
                "as_of_management_candidate_p_long_v1": 0.7,
                "as_of_management_candidate_p_short_v1": 0.1,
                "as_of_management_candidate_p_flat_v1": 0.2,
                "as_of_management_candidate_p_hat_v1": 0.7,
                "as_of_management_candidate_margin_v1": 0.5,
                "as_of_management_candidate_uncertainty_score_v1": 0.1,
                "as_of_management_candidate_tradable_prob_v1": 0.9,
                "as_of_management_candidate_mfe_first_n_pred_v1": 8.0,
                "as_of_management_candidate_path_quality_pred_v1": 0.8,
                "as_of_management_xgb_p_long_v1": 0.6,
                "as_of_management_xgb_p_short_v1": 0.2,
                "as_of_management_xgb_p_flat_v1": 0.2,
                "as_of_management_xgb_p_hat_v1": 0.6,
                "as_of_management_xgb_pred_side_v1": "LONG",
                "as_of_management_xgb_has_ctx_v1": True,
                "hindsight_reward_realized_pnl_bps_v1": 5.0,
                "hindsight_reward_trade_outcome_class_v1": "WIN",
                "hindsight_reward_good_trade_v1": True,
                "hindsight_reward_good_trade_mfe20_mae5_v1": True,
                "hindsight_reward_bad_trade_v1": False,
                "hindsight_reward_good_exit_v1": True,
                "hindsight_reward_premature_exit_v1": False,
                "hindsight_reward_late_exit_v1": False,
                "hindsight_reward_exit_reason_v1": "THRESHOLD",
                "bandit_candidate_mode_v1": "DIRECT_METHOD",
                "observation_contract_v1": "READY",
                "propensity_contract_v1": "PARTIAL",
            }
        ]
    )
    observed.to_parquet(ledger_dir / "shadow_meta_all_trade_review_management_bandit_observed_sample_view_v1.parquet", index=False)
    observed.to_parquet(ledger_dir / "shadow_meta_all_trade_review_management_bandit_direct_method_candidate_view_v1.parquet", index=False)
    eligible = observed[
        [
            "management_row_key_v1",
            "candidate_uid_exact_v1",
            "trade_uid_exact_v1",
            "trade_id_exact_v1",
            "decision_anchor_type_v1",
            "as_of_session_v1",
            "as_of_vol_regime_v1",
            "as_of_management_core_minutes_held_at_anchor_v1",
            "as_of_management_core_giveback_ratio_from_peak_v1",
        ]
    ].copy()
    eligible["primary_model_name_v1"] = "TREE_REGRESSION_BASELINE"
    eligible["primary_model_score_v1"] = 0.55
    eligible["primary_model_score_rank_within_split_v1"] = 7
    eligible.to_parquet(ledger_dir / "shadow_meta_all_trade_review_management_exit_local_all_eligible_scored_view_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "as_of_row_uid_v1": "asof-1",
                "run_id": "run-1",
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "anchor_timestamp_utc": "2026-01-01T00:00:00Z",
                "as_of_mgmt_trace_last_peak_ts_utc_v1": "2025-12-31T23:55:00Z",
                "as_of_mgmt_trace_last_mfe_ts_utc_v1": "2025-12-31T23:58:00Z",
                "as_of_mgmt_trace_peak_price_v1": 100.0,
                "as_of_mgmt_trace_anchor_price_v1": 99.8,
                "as_of_mgmt_trace_mfe_bps_at_anchor_v1": 8.0,
                "as_of_mgmt_trace_last_peak_mfe_bps_v1": 10.0,
                "as_of_mgmt_trace_max_mfe_without_mae_bps_v1": 9.0,
                "as_of_mgmt_trace_mfe_mae_sequence_order_v1": "MFE_FIRST",
                "as_of_mgmt_trace_last_peak_ts_utc_null_reason_v1": "NONE",
                "as_of_mgmt_trace_last_mfe_ts_utc_null_reason_v1": "NONE",
                "as_of_mgmt_trace_last_peak_mfe_bps_null_reason_v1": "NONE",
                "as_of_mgmt_trace_max_mfe_without_mae_bps_null_reason_v1": "NONE",
                "as_of_mgmt_trace_mfe_mae_sequence_order_null_reason_v1": "NONE",
            }
        ]
    ).to_parquet(ledger_dir / "shadow_meta_all_trade_review_management_anchor_raw_state_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "c1",
                "trade_uid": "t1",
                "trade_id": "tid1",
                "decision_anchor_type_v1": "MANAGEMENT_EXIT_ANCHOR",
                "as_of_trade_pocket_v1": "MID",
                "run_id": "run-1",
                "as_of_row_uid_v1": "asof-1",
                "as_of_candidate_policy_hash_v1": "policy-1",
                "as_of_candidate_entry_bundle_sha256_v1": "entry-1",
                "as_of_candidate_exit_bundle_sha256_v1": "exit-1",
            }
        ]
    ).to_parquet(ledger_dir / "shadow_meta_all_trade_review_as_of_decision_moment_ledger_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "tid1",
                "entry_timestamp": "2025-12-31T23:00:00Z",
                "exit_timestamp": "2026-01-01T01:00:00Z",
                "realized_pnl_bps": 5.0,
                "mfe_bps": 10.0,
                "mae_bps": -1.0,
                "holding_time_bars": 4,
                "trade_outcome_class": "WIN",
                "exit_reason": "THRESHOLD",
                "good_exit": True,
                "premature_exit": False,
                "late_exit": False,
                "hindsight_management_review_v1": "GOOD",
                "hindsight_peak_mfe_bps_v1": 10.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 2.0,
                "policy_hash": "policy-1",
            }
        ]
    ).to_parquet(ledger_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid_exact_v1": "c1",
                "trade_uid_exact_v1": "t1",
                "trade_id_exact_v1": "tid1",
                "review_exit_bucket_v1": "GOOD_EXIT",
                "review_entry_bucket_v1": "GOOD_ENTRY",
                "hindsight_rl_review_reason_v1": "OK",
                "hindsight_rl_review_domain_support_v1": "OK",
                "hindsight_peak_to_worst_after_peak_bps_v1": 1.0,
            }
        ]
    ).to_parquet(ledger_dir / "shadow_meta_all_trade_review_hindsight_trade_export_closed_trades.parquet", index=False)
    _write_json(ledger_dir / "shadow_meta_all_trade_review_management_bandit_action_reward_contract_v1.json", {"action_space_v1": ["HOLD", "EXIT_NOW"], "layer_name": "TEST"})
    _write_json(ledger_dir / "shadow_meta_all_trade_review_management_bandit_observed_action_contract_v1.json", {"layer_name": "OBS"})
    _write_json(ledger_dir / "shadow_meta_all_trade_review_management_bandit_status_v1.json", {"MANAGEMENT_BANDIT_PROPENSITY_STATUS": "PROPENSITY_NOT_ESTABLISHED"})
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_ledger_build_summary.json",
        {
            "control_date": "2026-04-11",
            "as_of_supervision_join_coverage": {"exact_rows": 1, "fallback_rows": 0, "unjoinable_rows": 0},
            "leakage_guard": {"status": "PASS"},
            "artifact_paths": {},
        },
    )
    _write_json(
        reports_root / "truth_downstream_canonical_rebuild_v1.json",
        {
            "ledger_dir": str(ledger_dir),
            "steps": [{"step": "all_trade_review_ledger", "status": "ok"}],
        },
    )

    result = script.materialize(reports_root=reports_root, rerun_follow_on=False)

    assert (ledger_dir / "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet").exists()
    assert (ledger_dir / "shadow_meta_all_trade_review_management_policy_logging_outcome_backfill_harness_v1.parquet").exists()
    summary = result["summary"]
    assert summary["decision_v1"] == "MONDAY_RUNTIME_POLICY_LOGGING_BUILT"
    assert summary["failed_consistency_count_v1"] == 0
