from __future__ import annotations

from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_truth_entry_rl_observability_v1 import (
    build_entry_rl_observability_payload,
)


def test_build_entry_rl_observability_payload_marks_entry_as_compositional_not_policy_truth() -> None:
    direct_df = pd.DataFrame(
        [
            {
                "run_id": "RUN_A",
                "candidate_uid": "cand-1",
                "trade_uid": "trade-1",
                "trade_id": "SIM-1",
                "decision_timestamp": "2025-01-01T10:00:00+00:00",
                "entry_timestamp": "2025-01-01T10:00:00+00:00",
                "exit_timestamp": "2025-01-01T10:30:00+00:00",
                "direct_entry_as_of_row_uid_v1": "asof-1",
                "domain_v1": "ENTRY",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "split_bucket_v1": "TRAIN",
                "decision_anchor_type_v1": "ENTRY_DECISION_ANCHOR",
                "decision_anchor_domain_v1": "ENTRY",
                "direct_entry_timestamp_utc_v1": "2025-01-01T10:00:00+00:00",
                "as_of_timestamp_utc_v1": "2025-01-01T10:00:00+00:00",
                "as_of_session_v1": "US",
                "as_of_weekday_utc_v1": 2,
                "skipability_head_target_v1": "NON_SKIP",
                "original_entry_action_label_v1": "TAKE_NOW",
                "policy_stack_stage_v1": "ENTRY_DIRECT_POLICY_COMPOSITE_V1",
                "skipability_branch_row_uid_v1": "asof-1",
                "timing_branch_row_uid_v1": "asof-1",
                "timing_head_target_v1": "TAKE_NOW",
                "timing_branch_split_bucket_v1": "TRAIN",
                "timing_direct_timestamp_utc_v1": "2025-01-01T10:00:00+00:00",
                "skipability_head_available_v1": True,
                "timing_head_required_v1": True,
                "timing_head_available_v1": True,
                "timing_branch_source_artifact_v1": "timing.parquet",
                "final_direct_entry_action_v1": "TAKE_NOW",
                "direct_composite_status_v1": "DIRECT_TAKE_NOW_FINAL",
                "wait_followthrough_status_v1": "NOT_APPLICABLE",
                "confirmation_branch_row_uid_v1": None,
                "confirmation_timestamp_utc_v1": None,
            },
            {
                "run_id": "RUN_A",
                "candidate_uid": "cand-2",
                "trade_uid": "trade-2",
                "trade_id": "SIM-2",
                "decision_timestamp": "2025-01-01T10:05:00+00:00",
                "entry_timestamp": "2025-01-01T10:05:00+00:00",
                "exit_timestamp": None,
                "direct_entry_as_of_row_uid_v1": "asof-2",
                "domain_v1": "ENTRY",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "split_bucket_v1": "TRAIN",
                "decision_anchor_type_v1": "ENTRY_DECISION_ANCHOR",
                "decision_anchor_domain_v1": "ENTRY",
                "direct_entry_timestamp_utc_v1": "2025-01-01T10:05:00+00:00",
                "as_of_timestamp_utc_v1": "2025-01-01T10:05:00+00:00",
                "as_of_session_v1": "US",
                "as_of_weekday_utc_v1": 2,
                "skipability_head_target_v1": "NON_SKIP",
                "original_entry_action_label_v1": "WAIT",
                "policy_stack_stage_v1": "ENTRY_DIRECT_POLICY_COMPOSITE_V1",
                "skipability_branch_row_uid_v1": "asof-2",
                "timing_branch_row_uid_v1": "asof-2",
                "timing_head_target_v1": "WAIT",
                "timing_branch_split_bucket_v1": "TRAIN",
                "timing_direct_timestamp_utc_v1": "2025-01-01T10:05:00+00:00",
                "skipability_head_available_v1": True,
                "timing_head_required_v1": True,
                "timing_head_available_v1": True,
                "timing_branch_source_artifact_v1": "timing.parquet",
                "final_direct_entry_action_v1": "WAIT",
                "direct_composite_status_v1": "DIRECT_WAIT_FINAL",
                "wait_followthrough_status_v1": "WAIT_WITH_PROVABLE_CONFIRMATION",
                "confirmation_branch_row_uid_v1": "confirm-2",
                "confirmation_timestamp_utc_v1": "2025-01-01T10:07:00+00:00",
            },
            {
                "run_id": "RUN_A",
                "candidate_uid": "cand-3",
                "trade_uid": "trade-3",
                "trade_id": "SIM-3",
                "decision_timestamp": "2025-01-01T10:10:00+00:00",
                "entry_timestamp": None,
                "exit_timestamp": None,
                "direct_entry_as_of_row_uid_v1": "asof-3",
                "domain_v1": "ENTRY",
                "used_for_training": False,
                "used_for_validation": True,
                "used_for_holdout": False,
                "split_bucket_v1": "VALIDATION",
                "decision_anchor_type_v1": "ENTRY_DECISION_ANCHOR",
                "decision_anchor_domain_v1": "ENTRY",
                "direct_entry_timestamp_utc_v1": "2025-01-01T10:10:00+00:00",
                "as_of_timestamp_utc_v1": "2025-01-01T10:10:00+00:00",
                "as_of_session_v1": "US",
                "as_of_weekday_utc_v1": 2,
                "skipability_head_target_v1": "SKIP",
                "original_entry_action_label_v1": "SKIP",
                "policy_stack_stage_v1": "ENTRY_DIRECT_POLICY_COMPOSITE_V1",
                "skipability_branch_row_uid_v1": "asof-3",
                "timing_branch_row_uid_v1": None,
                "timing_head_target_v1": None,
                "timing_branch_split_bucket_v1": None,
                "timing_direct_timestamp_utc_v1": None,
                "skipability_head_available_v1": True,
                "timing_head_required_v1": False,
                "timing_head_available_v1": False,
                "timing_branch_source_artifact_v1": None,
                "final_direct_entry_action_v1": "SKIP",
                "direct_composite_status_v1": "DIRECT_SKIP_FINAL",
                "wait_followthrough_status_v1": "NOT_APPLICABLE",
                "confirmation_branch_row_uid_v1": None,
                "confirmation_timestamp_utc_v1": None,
            },
        ]
    )
    asof_df = pd.DataFrame(
        [
            {
                "as_of_row_uid_v1": "asof-1",
                "candidate_uid": "cand-1",
                "as_of_candidate_decision_v1": "LONG",
                "as_of_candidate_decision_reason_v1": "pre_quality",
                "as_of_candidate_side_v1": "long",
                "as_of_candidate_session_v1": "US",
                "as_of_candidate_atr_bps_v1": 12.0,
                "as_of_candidate_entry_spread_bps_v1": 1.1,
                "as_of_candidate_uncertainty_score_v1": 0.2,
                "as_of_candidate_tradable_prob_v1": 0.8,
                "as_of_candidate_mfe_first_n_pred_v1": 3.0,
                "as_of_candidate_vol_regime_v1": "HIGH",
                "as_of_candidate_trend_regime_v1": "TREND_UP",
                "as_of_candidate_policy_hash_v1": "hash-a",
                "as_of_candidate_entry_bundle_sha256_v1": "entry-a",
                "as_of_candidate_exit_bundle_sha256_v1": "exit-a",
            },
            {
                "as_of_row_uid_v1": "asof-2",
                "candidate_uid": "cand-2",
                "as_of_candidate_decision_v1": "LONG",
                "as_of_candidate_decision_reason_v1": "pre_quality",
                "as_of_candidate_side_v1": "long",
                "as_of_candidate_session_v1": "US",
                "as_of_candidate_atr_bps_v1": 13.0,
                "as_of_candidate_entry_spread_bps_v1": 1.2,
                "as_of_candidate_uncertainty_score_v1": 0.3,
                "as_of_candidate_tradable_prob_v1": 0.7,
                "as_of_candidate_mfe_first_n_pred_v1": 2.0,
                "as_of_candidate_vol_regime_v1": "HIGH",
                "as_of_candidate_trend_regime_v1": "TREND_UP",
                "as_of_candidate_policy_hash_v1": "hash-a",
                "as_of_candidate_entry_bundle_sha256_v1": "entry-a",
                "as_of_candidate_exit_bundle_sha256_v1": "exit-a",
            },
            {
                "as_of_row_uid_v1": "asof-3",
                "candidate_uid": "cand-3",
                "as_of_candidate_decision_v1": "LONG",
                "as_of_candidate_decision_reason_v1": "pre_quality",
                "as_of_candidate_side_v1": "long",
                "as_of_candidate_session_v1": "US",
                "as_of_candidate_atr_bps_v1": 10.0,
                "as_of_candidate_entry_spread_bps_v1": 1.0,
                "as_of_candidate_uncertainty_score_v1": 0.4,
                "as_of_candidate_tradable_prob_v1": 0.4,
                "as_of_candidate_mfe_first_n_pred_v1": 1.0,
                "as_of_candidate_vol_regime_v1": "LOW",
                "as_of_candidate_trend_regime_v1": "RANGE",
                "as_of_candidate_policy_hash_v1": "hash-b",
                "as_of_candidate_entry_bundle_sha256_v1": "entry-b",
                "as_of_candidate_exit_bundle_sha256_v1": "exit-b",
            },
        ]
    )
    skip_raw_df = pd.DataFrame(
        [
            {
                "as_of_row_uid_v1": "asof-1",
                "skip_raw_xgb_exact_available_v1": True,
                "as_of_skip_candidate_margin_v1": 0.11,
                "as_of_skip_candidate_path_quality_pred_v1": 0.81,
                "as_of_skip_xgb_has_ctx_v1": 1,
                "as_of_skip_xgb_p_flat_v1": 0.4,
                "as_of_skip_xgb_p_hat_v1": 0.55,
                "as_of_skip_xgb_p_long_v1": 0.55,
                "as_of_skip_xgb_p_short_v1": 0.05,
                "as_of_skip_xgb_pred_side_v1": "LONG",
            },
            {
                "as_of_row_uid_v1": "asof-2",
                "skip_raw_xgb_exact_available_v1": True,
                "as_of_skip_candidate_margin_v1": 0.12,
                "as_of_skip_candidate_path_quality_pred_v1": 0.72,
                "as_of_skip_xgb_has_ctx_v1": 1,
                "as_of_skip_xgb_p_flat_v1": 0.3,
                "as_of_skip_xgb_p_hat_v1": 0.62,
                "as_of_skip_xgb_p_long_v1": 0.62,
                "as_of_skip_xgb_p_short_v1": 0.08,
                "as_of_skip_xgb_pred_side_v1": "LONG",
            },
            {
                "as_of_row_uid_v1": "asof-3",
                "skip_raw_xgb_exact_available_v1": True,
                "as_of_skip_candidate_margin_v1": 0.03,
                "as_of_skip_candidate_path_quality_pred_v1": 0.2,
                "as_of_skip_xgb_has_ctx_v1": 1,
                "as_of_skip_xgb_p_flat_v1": 0.7,
                "as_of_skip_xgb_p_hat_v1": 0.2,
                "as_of_skip_xgb_p_long_v1": 0.2,
                "as_of_skip_xgb_p_short_v1": 0.1,
                "as_of_skip_xgb_pred_side_v1": "LONG",
            },
        ]
    )
    entry_anchor_raw_df = pd.DataFrame(
        [
            {
                "as_of_row_uid_v1": "asof-1",
                "entry_raw_xgb_multi_horizon_exact_available_v1": True,
                "as_of_entry_candidate_margin_v1": 0.11,
                "as_of_entry_candidate_path_quality_pred_v1": 0.81,
            },
            {
                "as_of_row_uid_v1": "asof-2",
                "entry_raw_xgb_multi_horizon_exact_available_v1": True,
                "as_of_entry_candidate_margin_v1": 0.12,
                "as_of_entry_candidate_path_quality_pred_v1": 0.72,
            },
            {
                "as_of_row_uid_v1": "asof-3",
                "entry_raw_xgb_multi_horizon_exact_available_v1": True,
                "as_of_entry_candidate_margin_v1": 0.03,
                "as_of_entry_candidate_path_quality_pred_v1": 0.20,
            },
        ]
    )
    supervision_df = pd.DataFrame(
        [
            {
                "candidate_uid": "cand-1",
                "as_of_row_uid_v1": "asof-1",
                "hindsight_policy_action_domain_v1": "ENTRY",
                "hindsight_policy_action_projection_kind_v1": "DIRECT_ENTRY_DECISION",
                "hindsight_policy_action_v1": "TAKE_NOW",
                "hindsight_policy_action_reason_path_v1": "ENTRY / TAKE_NOW / ENTRY_TIMING / ENTER_NOW_OK",
                "hindsight_policy_counterfactual_value_bps_v1": None,
                "hindsight_policy_counterfactual_value_source_v1": "SUPPORT_ONLY",
                "hindsight_policy_priority_abs_bps_v1": None,
                "hindsight_policy_action_support_v1": "entry_reason_family=ENTRY_TIMING;entry_reason_code=ENTER_NOW_OK;entry_bucket=entry_good;trade_bucket=good_trade;trade_outcome_class=positive_exit;adverse_first=FALSE;peak_mfe_bps=25.0;mae_bps=8.0",
                "hindsight_policy_action_semantic_contract_v1": "HINDSIGHT_ONLY",
                "hindsight_supervision_join_contract_v1": "JOIN_EXACT",
            },
            {
                "candidate_uid": "cand-2",
                "as_of_row_uid_v1": "asof-2",
                "hindsight_policy_action_domain_v1": "ENTRY",
                "hindsight_policy_action_projection_kind_v1": "DIRECT_ENTRY_DECISION",
                "hindsight_policy_action_v1": "WAIT",
                "hindsight_policy_action_reason_path_v1": "ENTRY / WAIT / ENTRY_TIMING / WAIT_TO_REDUCE_MAE",
                "hindsight_policy_counterfactual_value_bps_v1": None,
                "hindsight_policy_counterfactual_value_source_v1": "SUPPORT_ONLY",
                "hindsight_policy_priority_abs_bps_v1": None,
                "hindsight_policy_action_support_v1": "entry_reason_family=ENTRY_TIMING;entry_reason_code=WAIT_TO_REDUCE_MAE;entry_bucket=entry_good_but_fragile;trade_bucket=underheld_trade;trade_outcome_class=positive_exit;adverse_first=TRUE;confirmation_entry_localizable=TRUE;confirmation_entry_reason=FIRST_MEANINGFUL_MFE_BAR_INDEX_MINUTE_CADENCE_LOCALIZATION;confirmation_entry_ts=2025-01-01T10:07:00+00:00",
                "hindsight_policy_action_semantic_contract_v1": "HINDSIGHT_ONLY",
                "hindsight_supervision_join_contract_v1": "JOIN_EXACT",
            },
            {
                "candidate_uid": "cand-3",
                "as_of_row_uid_v1": "asof-3",
                "hindsight_policy_action_domain_v1": "ENTRY",
                "hindsight_policy_action_projection_kind_v1": "DIRECT_ENTRY_DECISION",
                "hindsight_policy_action_v1": "SKIP",
                "hindsight_policy_action_reason_path_v1": "ENTRY / SKIP / BAD_TRADE_QUALITY / LARGE_MAE_BEFORE_MEANINGFUL_MFE",
                "hindsight_policy_counterfactual_value_bps_v1": 18.0,
                "hindsight_policy_counterfactual_value_source_v1": "SKIP_TRADE_AVOIDED_LOSS_BPS",
                "hindsight_policy_priority_abs_bps_v1": 18.0,
                "hindsight_policy_action_support_v1": "entry_reason_family=BAD_TRADE_QUALITY;entry_reason_code=LARGE_MAE_BEFORE_MEANINGFUL_MFE;entry_bucket=entry_bad;trade_bucket=bad_trade;trade_outcome_class=negative_exit;adverse_first=TRUE;skip_trade_avoided_loss_bps=18.0;mae_bps=24.0;peak_mfe_bps=2.0",
                "hindsight_policy_action_semantic_contract_v1": "HINDSIGHT_ONLY",
                "hindsight_supervision_join_contract_v1": "JOIN_EXACT",
            },
        ]
    )
    wait_lifecycle_df = pd.DataFrame(
        [
            {
                "candidate_uid": "cand-2",
                "wait_followthrough_status_v1": "WAIT_WITH_PROVABLE_CONFIRMATION",
                "wait_lifecycle_rollup_status_v1": "WAIT_WITH_PROVABLE_CONFIRMATION",
                "wait_lifecycle_terminal_status_v1": "PROVABLE_CONFIRMATION_TAKE_NOW_EXACT",
                "wait_lifecycle_terminal_reason_v1": "PROVABLE_CONFIRMATION_TAKE_NOW_EXACT",
                "confirmation_delay_minutes_v1": 2.0,
                "has_provable_confirmation_v1": True,
                "confirmation_transition_allowed_v1": True,
                "management_transition_allowed_v1": True,
                "coverage_status_v1": "CONFIRMATION_FLOW_ESTABLISHED",
            }
        ]
    )
    actual_take_terminal_df = pd.DataFrame(
        [
            {
                "candidate_uid": "cand-1",
                "route_status_v1": "DIRECT_TAKE_NOW_ACTUALIZED",
                "activation_origin_v1": "DIRECT_TAKE_NOW",
                "activation_timestamp_utc_v1": "2025-01-01T10:00:00+00:00",
                "management_handoff_status_v1": "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD",
                "management_anchor_type_v1": "ACTUAL_EXIT_DECISION_ANCHOR",
                "management_action_label_v1": "EXIT_NOW",
                "management_projection_kind_v1": "DIRECT_ACTUAL_EXIT_DECISION",
                "closed_exit_reason_v1": "TP_HIT",
                "realized_pnl_bps": 22.0,
                "trade_outcome_class": "positive_exit",
                "terminal_outcome_available_v1": True,
            },
            {
                "candidate_uid": "cand-2",
                "route_status_v1": "DIRECT_WAIT_THEN_CONFIRMATION_TAKE_NOW_ACTUALIZED",
                "activation_origin_v1": "WAIT_TO_CONFIRMATION_TAKE_NOW",
                "activation_timestamp_utc_v1": "2025-01-01T10:07:00+00:00",
                "management_handoff_status_v1": "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD",
                "management_anchor_type_v1": "ACTUAL_EXIT_DECISION_ANCHOR",
                "management_action_label_v1": "EXIT_NOW",
                "management_projection_kind_v1": "DIRECT_ACTUAL_EXIT_DECISION",
                "closed_exit_reason_v1": "BE_PLUS_FLOOR",
                "realized_pnl_bps": 6.0,
                "trade_outcome_class": "positive_exit",
                "terminal_outcome_available_v1": True,
            },
        ]
    )
    hindsight_export_df = pd.DataFrame(
        [
            {
                "candidate_uid": "cand-1",
                "post_trade_quality_bucket": "good_trade",
                "post_trade_good_trade_flag_v1": True,
                "post_trade_good_trade_mfe20_mae5_v1": True,
                "post_trade_bad_trade_flag_v1": False,
                "review_entry_bucket_v1": "entry_good",
                "review_exit_bucket_v1": "exit_ok",
                "review_good_exit_v1": True,
                "review_premature_exit_v1": False,
                "review_late_exit_v1": False,
                "review_entry_good_but_fragile_v1": False,
                "review_entry_looked_good_but_failed_v1": False,
                "hindsight_entry_decision_review_v1": "TAKE_WAS_OK",
                "hindsight_should_skip_trade_v1": False,
                "hindsight_take_was_ok_v1": True,
                "hindsight_entry_review_unresolved_v1": False,
                "hindsight_management_review_v1": "MANAGED_OK",
                "hindsight_should_hold_longer_v1": False,
                "hindsight_should_exit_earlier_v1": False,
                "hindsight_managed_ok_v1": True,
                "hindsight_hold_longer_extra_value_bps_v1": None,
                "hindsight_exit_earlier_saved_bps_v1": None,
                "hindsight_skip_trade_avoided_loss_bps_v1": None,
                "hindsight_peak_mfe_bps_v1": 25.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 3.0,
            },
            {
                "candidate_uid": "cand-2",
                "post_trade_quality_bucket": "underheld_trade",
                "post_trade_good_trade_flag_v1": False,
                "post_trade_good_trade_mfe20_mae5_v1": False,
                "post_trade_bad_trade_flag_v1": False,
                "review_entry_bucket_v1": "entry_good_but_fragile",
                "review_exit_bucket_v1": "late_exit",
                "review_good_exit_v1": False,
                "review_premature_exit_v1": False,
                "review_late_exit_v1": True,
                "review_entry_good_but_fragile_v1": True,
                "review_entry_looked_good_but_failed_v1": False,
                "hindsight_entry_decision_review_v1": "TAKE_WAS_OK_BUT_WAIT_BETTER",
                "hindsight_should_skip_trade_v1": False,
                "hindsight_take_was_ok_v1": True,
                "hindsight_entry_review_unresolved_v1": False,
                "hindsight_management_review_v1": "SHOULD_EXIT_EARLIER",
                "hindsight_should_hold_longer_v1": True,
                "hindsight_should_exit_earlier_v1": False,
                "hindsight_managed_ok_v1": False,
                "hindsight_hold_longer_extra_value_bps_v1": 9.0,
                "hindsight_exit_earlier_saved_bps_v1": None,
                "hindsight_skip_trade_avoided_loss_bps_v1": None,
                "hindsight_peak_mfe_bps_v1": 16.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 10.0,
            },
        ]
    )

    payload = build_entry_rl_observability_payload(
        direct_df=direct_df,
        asof_df=asof_df,
        skip_raw_df=skip_raw_df,
        entry_anchor_raw_df=entry_anchor_raw_df,
        supervision_df=supervision_df,
        wait_lifecycle_df=wait_lifecycle_df,
        actual_take_terminal_df=actual_take_terminal_df,
        hindsight_export_df=hindsight_export_df,
        skipability_pressure_summary={
            "completed_zero_trade_runs": 10,
            "candidate_rich_zero_trade_runs": 10,
        },
        market_opportunity_summary={
            "opportunity_rich_zero_trade_runs_anchor": [{"run_id": "RUN_Z1"}, {"run_id": "RUN_Z2"}],
        },
        review_dir=Path("/tmp/fake_review"),
    )

    summary = payload["entry_rl_observability_summary_v1"]
    status = payload["entry_rl_observability_status_v1"]
    view = payload["entry_rl_observability_view_v1_df"]

    assert summary["observed_direct_entry_rows_v1"] == 3
    assert summary["logged_action_counts_v1"] == {"TAKE_NOW": 1, "WAIT": 1, "SKIP": 1}
    assert summary["policy_hash_available_rows_v1"] == 3
    assert summary["actualized_take_count_v1"] == 2
    assert summary["wait_followthrough_counts_v1"] == {"WAIT_WITH_PROVABLE_CONFIRMATION": 1}
    assert summary["reward_semantics_counts_v1"]["DIRECT_TAKE_REALIZED_OUTCOME_AVAILABLE"] == 1
    assert summary["reward_semantics_counts_v1"]["WAIT_THEN_CONFIRMATION_REALIZED_OUTCOME_AVAILABLE"] == 1
    assert summary["reward_semantics_counts_v1"]["SKIP_AVOIDED_LOSS_HINDSIGHT_ONLY"] == 1
    assert status["ENTRY_POLICY_SNAPSHOT_STATUS"] == "READY_EXACT_CANDIDATE_POLICY_HASH_AND_MODEL_SNAPSHOT"
    assert status["ENTRY_PROPENSITY_STATUS"] == "NOT_ESTABLISHED"
    assert status["ENTRY_DIRECT_ACTION_STATUS"] == "HIERARCHICAL_COMPOSITION_READY_NOT_POLICY_TRUTH"
    assert view.loc[view["candidate_uid"] == "cand-3", "entry_reason_code_v1"].iloc[0] == "LARGE_MAE_BEFORE_MEANINGFUL_MFE"
    assert view.loc[view["candidate_uid"] == "cand-2", "entry_review_focus_v1"].iloc[0] == "SHOULD_HAVE_WAITED"
