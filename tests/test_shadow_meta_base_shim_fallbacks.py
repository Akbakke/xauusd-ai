import pandas as pd
import inspect

from gx1.analysis import shadow_meta_v1 as shadow_meta


def _sample_shadow_meta_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": ["LONG", "SHORT", "LONG", "SHORT", "LONG", "SHORT"],
            "session": ["US", "EU", "OVERLAP", "US", "EU", "OVERLAP"],
            "weekday_utc": [1, 2, 3, 4, 1, 2],
            "hour_utc": [12, 13, 14, 15, 16, 17],
            "atr_bps": [10, 11, 12, 13, 14, 15],
            "entry_spread_bps": [1, 1, 1, 1, 1, 1],
            "p_long": [0.80, 0.15, 0.75, 0.20, 0.85, 0.10],
            "p_short": [0.10, 0.75, 0.15, 0.70, 0.05, 0.80],
            "p_flat": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            "p_hat": [0.80, 0.75, 0.75, 0.70, 0.85, 0.80],
            "margin": [0.50, 0.45, 0.55, 0.40, 0.60, 0.35],
            "uncertainty_score": [0.10, 0.25, 0.10, 0.20, 0.08, 0.22],
            "tradable_prob": [0.85, 0.65, 0.80, 0.60, 0.90, 0.55],
            "mfe_first_n_pred": [18, 4, 22, 5, 26, 6],
            "path_quality_pred": [0.85, 0.20, 0.90, 0.25, 0.92, 0.18],
            "vol_regime": ["HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW"],
            "trend_regime": ["TREND_UP", "TREND_DOWN", "TREND_UP", "TREND_DOWN", "TREND_UP", "TREND_DOWN"],
            "meta_allow_label_v1": [True, False, True, False, True, False],
            "positive_exit": [True, False, True, False, True, False],
            "cata": [False, True, False, False, False, True],
            "never_mfe": [False, False, False, True, False, False],
            "good_mfe_then_rot": [False, False, False, False, False, False],
            "pnl_bps": [12, -18, 22, -7, 28, -16],
            "mfe_bps": [30, 4, 40, 6, 45, 3],
            "mae_bps": [-3, -25, -2, -10, -1, -22],
            "accepted": [True, True, True, True, True, True],
            "trainable_mask_v1": [True, True, True, True, True, True],
            "post_exit_mfe_bps": [14, 0, 8, 0, 3, 0],
            "post_trade_quality_bucket": [
                "good_trade",
                "cata_trade",
                "good_trade",
                "bad_trade",
                "good_trade",
                "cata_trade",
            ],
        }
    )


def test_shadow_meta_shim_fallback_model_and_threshold_surface():
    frame = _sample_shadow_meta_frame()

    feature_spec = shadow_meta.derive_feature_spec(frame)
    model = shadow_meta.fit_shadow_meta_model(frame.iloc[:4], frame.iloc[2:], feature_spec)
    scores = shadow_meta.predict_allow_score(model, frame, feature_spec)
    eval_df = shadow_meta.add_offline_business_labels(frame.assign(meta_allow_score_v1=scores))
    threshold_df = shadow_meta.build_threshold_sweep(eval_df)

    assert len(feature_spec.feature_cols) >= 8
    assert len(scores) == len(frame)
    assert threshold_df["threshold"].eq(0.5).any()
    assert {"false_veto_total", "cata_capture", "business_score_v2_sum_kept"}.issubset(threshold_df.columns)
    assert shadow_meta.choose_threshold_from_validation_business(threshold_df) == 0.5
    assert shadow_meta.choose_threshold_from_validation(threshold_df) == 0.5


def test_shadow_meta_shim_fallback_offline_review_surface_populates_teacher_fields():
    frame = _sample_shadow_meta_frame()
    frame = frame.assign(
        meta_allow_score_v1=[0.92, 0.18, 0.88, 0.35, 0.79, 0.12],
        open_ts_utc=pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
        close_ts_utc=pd.date_range("2026-01-01 00:30", periods=6, freq="h", tz="UTC"),
        decision_ts_utc=pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
    )

    review_df, summary = shadow_meta.build_offline_review_surface(frame, selected_threshold=0.5)

    assert len(review_df) == len(frame)
    assert set(review_df["hindsight_entry_decision_review_v1"].astype(str)) == {"TAKE_WAS_OK", "SHOULD_SKIP_TRADE"}
    assert set(review_df["hindsight_management_review_v1"].astype(str)) >= {"MANAGED_OK", "SHOULD_HOLD_LONGER"}
    assert {"review_entry_bucket_v1", "review_exit_bucket_v1", "hindsight_rl_review_reason_v1"}.issubset(review_df.columns)
    assert summary["rows"] == len(frame)
    assert "entry_decision_review_counts" in summary["rl_teacher_summary"]


def test_entry_skipability_no_change_still_materializes_baseline_branch(tmp_path):
    rows = []
    replay_rows = []
    labels = ["SKIP", "TAKE_NOW", "WAIT", "SKIP"] * 3
    splits = ["TRAIN"] * 4 + ["VALIDATION"] * 4 + ["HOLDOUT"] * 4
    for idx, (label, split) in enumerate(zip(labels, splits), start=1):
        ts = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(hours=idx)
        rows.append(
            {
                "run_id": "RUN_A",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_anchor_type_v1": "DIRECT",
                "decision_anchor_domain_v1": "ENTRY",
                "as_of_timestamp_utc_v1": ts.isoformat().replace("+00:00", "Z"),
                "decision_anchor_timestamp_utc_v1": ts.isoformat().replace("+00:00", "Z"),
                "as_of_row_uid_v1": f"row{idx}",
                "hindsight_policy_action_projection_kind_v1": "DIRECT_ENTRY_DECISION",
                "action_label_v1": label,
                "used_for_training": split == "TRAIN",
                "used_for_validation": split == "VALIDATION",
                "used_for_holdout": split == "HOLDOUT",
                "split_bucket_v1": split,
                "as_of_session_v1": "US",
                "as_of_weekday_utc_v1": 3,
            }
        )
        replay_rows.append(
            {
                "projection_kind_v1": "DIRECT_ENTRY_DECISION",
                "as_of_row_uid_v1": f"row{idx}",
                "entry_raw_replay_bar_exact_available_v1": True,
                "entry_raw_candidate_snapshot_exact_available_v1": False,
                "entry_raw_xgb_multi_horizon_exact_available_v1": False,
                "as_of_entry_replay_pressure_v1": 1.0,
            }
        )

    entry_policy_training_df = pd.DataFrame(rows)
    entry_anchor_raw_state_df = pd.DataFrame(replay_rows)
    entry_anchor_raw_state_contract_df = pd.DataFrame(
        {
            "feature_name": ["as_of_entry_replay_pressure_v1"],
            "source_family": ["replay_chunk_bar_exact"],
            "semantic_group": ["PRESSURE"],
            "source_specific_v1": [False],
        }
    )
    as_of_feature_contract_v6_df = pd.DataFrame(
        {
            "feature_name": ["as_of_session_v1", "as_of_weekday_utc_v1"],
            "feature_role_v1": ["INPUT_ALLOWED_ENTRY_CORE", "INPUT_ALLOWED_ENTRY_CORE"],
            "canonical_input_allowed_v1": [True, True],
        }
    )
    as_of_decision_moment_ledger_v6_df = pd.DataFrame({"as_of_row_uid_v1": [f"row{i}" for i in range(1, 13)]})

    payload = shadow_meta._build_entry_skipability_direct_state_expansion_and_branch_review_v1(
        reports_root=tmp_path,
        as_of_decision_moment_ledger_v6_df=as_of_decision_moment_ledger_v6_df,
        as_of_feature_contract_v6_df=as_of_feature_contract_v6_df,
        entry_policy_training_df=entry_policy_training_df,
        entry_anchor_raw_state_df=entry_anchor_raw_state_df,
        entry_anchor_raw_state_contract_df=entry_anchor_raw_state_contract_df,
        entry_skipability_freeze_note_v2={"skipability_holdout_macro_f1_enriched": 0.50},
        entry_timing_context_branch_review_v1={},
    )

    branch_df = payload["entry_skipability_branch_v1_df"]
    assert not branch_df.empty
    assert set(branch_df["action_label_v1"].astype(str)) == {"SKIP", "NON_SKIP"}
    assert set(branch_df["original_entry_action_label_v1"].astype(str)) == {"SKIP", "TAKE_NOW", "WAIT"}
    assert payload["entry_skipability_branch_summary"]["baseline_branch_materialized_v1"] is True
    assert payload["entry_skipability_branch_summary"]["decision"].endswith("_BASELINE_BRANCH_MATERIALIZED")
    assert not payload["as_of_feature_contract_v7_df"].empty
    assert payload["as_of_feature_contract_v7_summary"]["baseline_contract_materialized_v1"] is True


def test_resolve_truth_run_dir_uses_runs_namespace_when_direct_root_missing(tmp_path):
    reports_root = tmp_path / "truth_root"
    run_dir = reports_root / "runs" / "RUN_123"
    run_dir.mkdir(parents=True)

    resolved = shadow_meta._resolve_truth_run_dir(reports_root, "RUN_123")

    assert resolved == run_dir


def test_raw_state_builders_use_truth_run_dir_resolver():
    management_source = inspect.getsource(shadow_meta._build_management_raw_state_expansion_and_reprobe)
    entry_replay_source = inspect.getsource(shadow_meta._build_entry_replay_bar_state_expansion_and_reprobe)
    skipability_source = inspect.getsource(shadow_meta._build_entry_skipability_direct_state_expansion_and_branch_review_v1)

    assert "_resolve_truth_run_dir(reports_root, run_id)" in management_source
    assert "_resolve_truth_run_dir(reports_root, run_id)" in entry_replay_source
    assert "_resolve_truth_run_dir(reports_root, run_id)" in skipability_source


def test_management_rl_readiness_uses_exact_raw_state_aliases_and_dynamic_counts():
    timestamps = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    management_v4 = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "candidate_uid": ["c1", "c2"],
            "trade_uid": ["t1", "t2"],
            "trade_id": ["1", "2"],
            "decision_timestamp": timestamps.astype(str),
            "entry_timestamp": timestamps.astype(str),
            "exit_timestamp": (timestamps + pd.Timedelta(minutes=30)).astype(str),
            "as_of_row_uid_v1": ["row1", "row2"],
            "domain_v1": ["MANAGEMENT", "MANAGEMENT"],
            "action_label_v1": ["EXIT_NOW", "HOLD"],
            "used_for_training": [True, True],
            "used_for_validation": [False, False],
            "used_for_holdout": [False, False],
            "split_bucket_v1": ["TRAIN", "TRAIN"],
            "decision_anchor_type_v1": ["ACTUAL_EXIT_DECISION_ANCHOR", "ACTUAL_EXIT_DECISION_ANCHOR"],
            "decision_anchor_domain_v1": ["MANAGEMENT", "MANAGEMENT"],
            "decision_anchor_timestamp_utc_v1": timestamps.astype(str),
            "as_of_timestamp_utc_v1": timestamps.astype(str),
            "as_of_join_mode_v1": ["EXACT", "EXACT"],
            "as_of_join_source_v1": ["trade_journal_exact_close_ts_utc", "trade_journal_exact_close_ts_utc"],
            "hindsight_policy_action_projection_kind_v1": [
                "DIRECT_ACTUAL_EXIT_DECISION",
                "DIRECT_ACTUAL_EXIT_DECISION",
            ],
            "as_of_atr_bps_v1": [10.0, 12.0],
            "as_of_hour_utc_v1": [12, 13],
            "as_of_session_v1": ["US", "OVERLAP"],
            "as_of_side_v1": ["LONG", "SHORT"],
            "as_of_trend_regime_v1": ["TREND_UP", "TREND_DOWN"],
            "as_of_vol_regime_v1": ["HIGH", "LOW"],
            "as_of_weekday_utc_v1": [4, 4],
            "as_of_management_core_entry_spread_bps_v1": [pd.NA, pd.NA],
            "as_of_management_core_minutes_held_at_anchor_v1": [pd.NA, pd.NA],
            "as_of_management_core_giveback_ratio_from_peak_v1": [pd.NA, pd.NA],
        }
    )
    supervision_join = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "decision_timestamp": timestamps.astype(str),
            "candidate_uid": ["c1", "c2"],
            "as_of_row_uid_v1": ["row1", "row2"],
            "hindsight_decision_anchor_timestamp_utc_v1": timestamps.astype(str),
            "hindsight_policy_action_v1": ["EXIT_NOW", "HOLD"],
            "hindsight_decision_anchor_type_v1": [
                "ACTUAL_EXIT_DECISION_ANCHOR",
                "ACTUAL_EXIT_DECISION_ANCHOR",
            ],
            "hindsight_policy_action_projection_kind_v1": [
                "DIRECT_ACTUAL_EXIT_DECISION",
                "DIRECT_ACTUAL_EXIT_DECISION",
            ],
            "hindsight_policy_action_domain_v1": ["MANAGEMENT", "MANAGEMENT"],
        }
    )
    route_audit = pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2"],
            "final_direct_entry_action_v1": ["TAKE_NOW", "TAKE_NOW"],
            "route_status_v1": ["DIRECT_TAKE_NOW", "DIRECT_TAKE_NOW"],
            "activation_origin_v1": ["DIRECT_TAKE_NOW", "DIRECT_TAKE_NOW"],
            "wait_lifecycle_rollup_status_v1": ["NOT_APPLICABLE", "NOT_APPLICABLE"],
            "wait_lifecycle_terminal_status_v1": ["NOT_APPLICABLE", "NOT_APPLICABLE"],
            "wait_lifecycle_terminal_reason_v1": [pd.NA, pd.NA],
        }
    )
    actual_take_terminal = pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2"],
            "activation_origin_v1": ["DIRECT_TAKE_NOW", "DIRECT_TAKE_NOW"],
            "route_status_v1": ["DIRECT_TAKE_NOW", "DIRECT_TAKE_NOW"],
            "management_handoff_status_v1": [
                "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD",
                "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD",
            ],
        }
    )
    closed_trades = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "decision_timestamp": timestamps.astype(str),
            "candidate_uid": ["c1", "c2"],
            "realized_pnl_bps": [15.0, 9.0],
            "trade_outcome_class": ["positive_exit", "positive_exit"],
            "good_trade": [True, True],
            "good_trade_mfe20_mae5": [True, True],
            "bad_trade": [False, False],
            "good_exit": [True, True],
            "premature_exit": [False, False],
            "late_exit": [False, False],
            "exit_reason": ["POSITIVE_EXIT", "POSITIVE_EXIT"],
        }
    )
    management_raw_state = pd.DataFrame(
        {
            "as_of_row_uid_v1": ["row1", "row2"],
            "as_of_mgmt_trace_minutes_held_at_anchor_v1": [5.0, 9.0],
            "as_of_mgmt_trace_giveback_ratio_from_peak_v1": [0.12, 0.08],
            "as_of_mgmt_trace_last_peak_ts_utc_v1": [
                (timestamps[0] - pd.Timedelta(minutes=30)).isoformat(),
                (timestamps[1] - pd.Timedelta(minutes=20)).isoformat(),
            ],
            "as_of_mgmt_trace_last_mfe_ts_utc_v1": [
                (timestamps[0] - pd.Timedelta(minutes=10)).isoformat(),
                (timestamps[1] - pd.Timedelta(minutes=5)).isoformat(),
            ],
            "as_of_mgmt_trace_peak_price_v1": [102.0, 200.0],
            "as_of_mgmt_trace_anchor_price_v1": [100.0, 200.0],
            "as_of_mgmt_trace_mfe_bps_at_anchor_v1": [100.0, 75.0],
            "as_of_mgmt_trace_mfe_bps_so_far_v1": [120.0, 85.0],
            "as_of_mgmt_trace_mae_bps_so_far_v1": [-8.0, -6.0],
            "as_of_mgmt_trace_exit_model_evaluated_v1": [1.0, 1.0],
            "as_of_mgmt_trace_exit_prob_v1": [0.24, pd.NA],
            "as_of_mgmt_trace_exit_threshold_v1": [0.80, 0.82],
            "as_of_mgmt_trace_distance_from_peak_mfe_bps_v1": [20.0, 12.0],
            "as_of_mgmt_trace_bars_since_peak_mfe_v1": [2.0, 1.0],
            "as_of_mgmt_replay_micro_momentum_3_v1": [0.30, -0.10],
            "as_of_mgmt_replay_micro_momentum_5_v1": [0.42, -0.05],
            "as_of_mgmt_replay_micro_acceleration_v1": [0.08, -0.02],
            "as_of_mgmt_replay_wick_ratio_v1": [0.15, 0.25],
            "as_of_mgmt_replay_retracement_from_last_impulse_v1": [0.20, 0.35],
            "as_of_mgmt_replay_minutes_since_session_open_v1": [60.0, 90.0],
            "as_of_mgmt_replay_minutes_to_next_session_boundary_v1": [180.0, 120.0],
            "as_of_mgmt_replay_session_change_flag_v1": [False, True],
            "as_of_mgmt_replay_session_tradable_v1": [True, True],
            "as_of_mgmt_candidate_entry_spread_bps_v1": [1.2, 1.5],
            "as_of_mgmt_candidate_p_long_v1": [0.82, 0.12],
            "as_of_mgmt_candidate_p_short_v1": [0.08, 0.78],
            "as_of_mgmt_candidate_p_flat_v1": [0.10, 0.10],
            "as_of_mgmt_candidate_p_hat_v1": [0.82, 0.78],
            "as_of_mgmt_candidate_margin_v1": [0.48, 0.43],
            "as_of_mgmt_candidate_uncertainty_score_v1": [0.09, 0.21],
            "as_of_mgmt_candidate_tradable_prob_v1": [0.88, 0.59],
            "as_of_mgmt_candidate_mfe_first_n_pred_v1": [18.0, 6.0],
            "as_of_mgmt_candidate_path_quality_pred_v1": [0.84, 0.29],
            "as_of_mgmt_xgb_p_long_v1": [0.79, 0.18],
            "as_of_mgmt_xgb_p_short_v1": [0.11, 0.72],
            "as_of_mgmt_xgb_p_flat_v1": [0.10, 0.10],
            "as_of_mgmt_xgb_p_hat_v1": [0.79, 0.72],
            "as_of_mgmt_xgb_pred_side_v1": ["LONG", "SHORT"],
            "as_of_mgmt_xgb_has_ctx_v1": [1.0, 1.0],
        }
    )

    payload = shadow_meta._build_management_rl_readiness_substrate_v1(
        management_policy_training_v4_df=management_v4,
        policy_action_supervision_join_df=supervision_join,
        entry_actualization_route_audit_v1_df=route_audit,
        entry_actual_take_terminal_outcome_view_v1_df=actual_take_terminal,
        closed_trades_df=closed_trades,
        entry_actualization_status_v1={"ENTRY_ACTUALIZATION_STATUS": "COMPOSITIONAL_READ_MODEL_READY"},
        entry_terminal_outcome_status_v1={"ENTRY_TERMINAL_OUTCOME_STATUS": "END_TO_END_READ_MODEL_READY"},
        management_anchor_raw_state_df=management_raw_state,
    )

    row_semantics = payload["management_rl_row_semantics_view_v1_df"]
    transition_view = payload["management_rl_transition_eligible_view_v1_df"]
    summary = payload["management_rl_readiness_summary_v1"]
    status = payload["management_rl_readiness_status_v1"]
    consistency = payload["management_rl_readiness_consistency_audit_v1_summary"]

    assert row_semantics["rl_observation_status_v1"].astype(str).eq("RL_OBSERVATION_AS_OF_CANONICAL_V1").all()
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_core_minutes_held_at_anchor_v1"] == 2
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_core_giveback_ratio_from_peak_v1"] == 2
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_core_entry_spread_bps_v1"] == 2
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_core_mfe_bps_so_far_v1"] == 2
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_replay_micro_momentum_3_v1"] == 2
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_candidate_p_long_v1"] == 2
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_xgb_p_hat_v1"] == 2
    assert summary["raw_exact_observation_alias_fill_counts_v1"]["as_of_management_exit_prob_v1"] == 1
    assert summary["derived_observation_fill_counts_v1"]["as_of_management_core_minutes_since_last_peak_v1"] == 2
    assert summary["derived_observation_fill_counts_v1"]["as_of_management_core_peak_to_anchor_bps_v1"] == 2
    assert summary["optional_sparse_signal_availability_counts_v1"]["as_of_management_exit_prob_available_v1"] == 1
    assert summary["optional_sparse_signal_availability_counts_v1"]["as_of_management_core_mfe_to_anchor_ratio_available_v1"] == 1
    assert summary["observation_feature_coverage_v1"]["as_of_management_core_minutes_since_last_peak_v1"]["available_count_v1"] == 2
    assert summary["observation_feature_coverage_v1"]["as_of_management_candidate_tradable_prob_v1"]["available_count_v1"] == 2
    assert summary["optional_sparse_observation_feature_coverage_v1"]["as_of_management_exit_prob_v1"]["available_count_v1"] == 1
    assert summary["optional_sparse_observation_feature_coverage_v1"]["as_of_management_core_mfe_to_anchor_ratio_v1"]["available_count_v1"] == 1
    assert summary["optional_sparse_availability_coverage_v1"]["as_of_management_exit_prob_available_v1"]["available_count_v1"] == 2
    assert summary["eligible_observation_feature_coverage_v1"]["as_of_management_replay_micro_momentum_5_v1"]["available_count_v1"] == 2
    assert summary["rl_transition_eligibility_counts_v1"]["RL_TRANSITION_ELIGIBLE"] == 2
    assert summary["terminal_bool_reward_channel_missing_eligible_count_v1"] == 0
    assert summary["terminal_bool_reward_channel_coverage_v1"]["terminal_good_trade_mfe20_mae5_v1"][
        "eligible_rows_bool_text_count_v1"
    ] == 2
    assert status["MANAGEMENT_RL_TRANSITION_STATUS"] == "REALIZED_PATH_ELIGIBLE_SUBSTRATE_PRESENT"
    assert status["MANAGEMENT_RL_OBSERVATION_STATUS"] == "AS_OF_CANONICAL_OBSERVATION_WITH_OPTIONAL_SIGNAL_MASKS_LOCKED"
    assert consistency["failed_check_count_v1"] == 0
    assert {
        "as_of_management_core_entry_spread_bps_v1",
        "as_of_management_core_minutes_since_last_peak_v1",
        "as_of_management_core_minutes_since_last_mfe_v1",
        "as_of_management_core_peak_to_anchor_bps_v1",
        "as_of_management_core_mfe_to_anchor_ratio_v1",
        "as_of_management_core_mfe_bps_so_far_v1",
        "as_of_management_core_mae_bps_so_far_v1",
        "as_of_management_exit_prob_v1",
        "as_of_management_candidate_p_long_v1",
        "as_of_management_candidate_tradable_prob_v1",
        "as_of_management_xgb_p_hat_v1",
        "as_of_management_replay_micro_momentum_3_v1",
        "as_of_management_replay_session_tradable_v1",
    }.issubset(set(transition_view.columns))
    assert transition_view.loc[0, "as_of_management_core_entry_spread_bps_v1"] == 1.2
    assert transition_view.loc[0, "as_of_management_core_minutes_since_last_peak_v1"] == 30.0
    assert transition_view.loc[0, "as_of_management_core_minutes_since_last_mfe_v1"] == 10.0
    assert abs(float(transition_view.loc[0, "as_of_management_core_peak_to_anchor_bps_v1"]) - 200.0) < 1e-9
    assert abs(float(transition_view.loc[0, "as_of_management_core_mfe_to_anchor_ratio_v1"]) - 0.5) < 1e-9
    assert abs(float(transition_view.loc[0, "as_of_management_exit_prob_v1"]) - 0.24) < 1e-9
    assert bool(transition_view.loc[0, "as_of_management_exit_prob_available_v1"]) is True
    assert transition_view.loc[0, "as_of_management_core_mfe_bps_so_far_v1"] == 120.0
    assert transition_view.loc[0, "as_of_management_core_mae_bps_so_far_v1"] == -8.0
    assert abs(float(transition_view.loc[0, "as_of_management_candidate_p_long_v1"]) - 0.82) < 1e-9
    assert abs(float(transition_view.loc[1, "as_of_management_candidate_tradable_prob_v1"]) - 0.59) < 1e-9
    assert abs(float(transition_view.loc[1, "as_of_management_xgb_p_hat_v1"]) - 0.72) < 1e-9
    assert transition_view.loc[1, "as_of_management_replay_micro_momentum_3_v1"] == -0.10
    assert bool(transition_view.loc[1, "as_of_management_replay_session_change_flag_v1"]) is True
    assert pd.isna(transition_view.loc[1, "as_of_management_exit_prob_v1"])
    assert bool(transition_view.loc[1, "as_of_management_exit_prob_available_v1"]) is False
    assert pd.isna(transition_view.loc[1, "as_of_management_core_mfe_to_anchor_ratio_v1"])
    assert bool(transition_view.loc[1, "as_of_management_core_mfe_to_anchor_ratio_available_v1"]) is False
    assert transition_view["hindsight_terminal_good_trade_v1"].astype(str).tolist() == ["TRUE", "TRUE"]
    assert transition_view["hindsight_terminal_good_trade_mfe20_mae5_v1"].astype(str).tolist() == ["TRUE", "TRUE"]
    assert transition_view["hindsight_terminal_bad_trade_v1"].astype(str).tolist() == ["FALSE", "FALSE"]


def test_hindsight_review_labels_are_attached_before_rl_read_models():
    ledger = pd.DataFrame(
        {
            "run_id": ["RUN_A"],
            "trade_id": ["1"],
            "trade_uid": ["t1"],
            "candidate_uid": ["c1"],
            "decision_timestamp": ["2026-01-01T00:00:00+00:00"],
            "trade_outcome_class": ["never_mfe"],
            "hindsight_score": [float("nan")],
            "hindsight_verdict_class": [shadow_meta._LEDGER_NOT_AVAILABLE],
            "hindsight_exit_quality_class": [shadow_meta._LEDGER_NOT_AVAILABLE],
            "hindsight_trade_quality_class": [shadow_meta._LEDGER_NOT_AVAILABLE],
            "fragile_winner": [shadow_meta._LEDGER_NOT_AVAILABLE],
            "label_source": ["NO_TRADE_LEVEL_REVIEW_EXPORT_IN_TRUTH_E2E"],
            "label_available": [False],
        }
    )
    hindsight_export = pd.DataFrame(
        {
            "run_id": ["RUN_A"],
            "trade_id": ["1"],
            "trade_uid": ["t1"],
            "candidate_uid": ["c1"],
            "decision_timestamp": ["2026-01-01T00:00:00+00:00"],
            "meta_allow_score_v1": [0.42],
            "post_trade_quality_bucket": ["bad_trade"],
            "post_trade_good_trade_flag_v1": [False],
            "post_trade_good_trade_mfe20_mae5_v1": [False],
            "post_trade_bad_trade_flag_v1": [False],
            "review_exit_bucket_v1": ["late_exit"],
            "review_entry_bucket_v1": ["entry_bad"],
            "review_good_exit_v1": [False],
            "review_premature_exit_v1": [False],
            "review_late_exit_v1": [True],
            "review_entry_good_but_fragile_v1": [False],
            "review_entry_looked_good_but_failed_v1": [False],
            "selected_threshold_business_v2": [0.5],
            "hindsight_entry_decision_review_v1": ["SHOULD_SKIP_TRADE"],
            "hindsight_should_skip_trade_v1": [True],
            "hindsight_take_was_ok_v1": [False],
            "hindsight_entry_review_unresolved_v1": [False],
            "hindsight_management_review_v1": ["SHOULD_EXIT_EARLIER"],
            "hindsight_should_hold_longer_v1": [False],
            "hindsight_should_exit_earlier_v1": [True],
            "hindsight_managed_ok_v1": [False],
            "hindsight_management_review_unresolved_v1": [False],
            "hindsight_rl_review_reason_v1": ["entry_bad|late_exit"],
            "hindsight_rl_review_domain_support_v1": ["review_surface"],
            "hindsight_rl_review_semantic_contract_v1": ["HINDSIGHT_ONLY_RL_TEACHER_V1"],
            "hindsight_hold_longer_extra_value_bps_v1": [0.0],
            "hindsight_exit_earlier_saved_bps_v1": [12.0],
            "hindsight_skip_trade_avoided_loss_bps_v1": [4.0],
            "hindsight_peak_mfe_bps_v1": [2.0],
            "hindsight_peak_to_exit_giveback_bps_v1": [12.0],
            "hindsight_peak_to_worst_after_peak_bps_v1": [pd.NA],
        }
    )

    enriched = shadow_meta._attach_hindsight_review_labels_to_closed_trade_ledger_v1(ledger, hindsight_export)

    assert enriched.loc[0, "good_trade"] == "FALSE"
    assert enriched.loc[0, "good_trade_mfe20_mae5"] == "FALSE"
    assert enriched.loc[0, "bad_trade"] == "TRUE"
    assert enriched.loc[0, "good_exit"] == "FALSE"
    assert enriched.loc[0, "late_exit"] == "TRUE"
    assert bool(enriched.loc[0, "label_available"]) is True


def test_entry_actualization_handoff_classifies_diagnostic_only_management_bridge_without_fabricating_head():
    timestamps = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    entry_direct = pd.DataFrame(
        {
            "run_id": ["RUN_A"],
            "candidate_uid": ["c1"],
            "trade_uid": ["t1"],
            "trade_id": ["1"],
            "decision_timestamp": [timestamps[0].isoformat()],
            "entry_timestamp": [timestamps[0].isoformat()],
            "exit_timestamp": [(timestamps[0] + pd.Timedelta(minutes=30)).isoformat()],
            "direct_entry_as_of_row_uid_v1": ["row1"],
            "split_bucket_v1": ["TRAIN"],
            "used_for_training": [True],
            "used_for_validation": [False],
            "used_for_holdout": [False],
            "as_of_session_v1": ["US"],
            "as_of_weekday_utc_v1": [4],
            "skipability_head_target_v1": ["NON_SKIP"],
            "timing_head_target_v1": ["TAKE_NOW"],
            "final_direct_entry_action_v1": ["TAKE_NOW"],
            "direct_composite_status_v1": ["CANONICAL_BRANCH_FROZEN"],
            "wait_followthrough_status_v1": ["NOT_APPLICABLE"],
            "direct_entry_timestamp_utc_v1": [timestamps[0].isoformat()],
            "confirmation_branch_row_uid_v1": [pd.NA],
            "confirmation_timestamp_utc_v1": [pd.NA],
            "skipability_branch_row_uid_v1": ["skip-row-1"],
            "timing_branch_row_uid_v1": ["timing-row-1"],
        }
    )
    wait_lifecycle = pd.DataFrame(
        {
            "direct_entry_as_of_row_uid_v1": ["row1"],
            "wait_lifecycle_rollup_status_v1": ["NOT_APPLICABLE"],
            "wait_lifecycle_terminal_status_v1": ["NOT_APPLICABLE"],
            "wait_lifecycle_terminal_reason_v1": [pd.NA],
            "confirmation_action_label_v1": [pd.NA],
            "confirmation_delay_minutes_v1": [pd.NA],
            "wait_lifecycle_source_artifact_v1": ["WAIT_LIFECYCLE_V1"],
            "confirmation_followthrough_source_artifact_v1": ["CONFIRMATION_FOLLOWTHROUGH_V1"],
            "has_provable_confirmation_v1": [False],
            "management_transition_allowed_v1": [True],
            "coverage_status_v1": ["NOT_APPLICABLE"],
        }
    )
    confirmation_followthrough = pd.DataFrame({"direct_entry_as_of_row_uid_v1": ["row1"]})
    management_v4 = pd.DataFrame(
        {
            "candidate_uid": ["other-candidate"],
            "trade_uid": ["other-trade-uid"],
            "trade_id": ["other-trade-id"],
            "as_of_row_uid_v1": ["mgmt-row-other"],
            "decision_anchor_type_v1": ["ACTUAL_EXIT_DECISION_ANCHOR"],
            "decision_anchor_domain_v1": ["TRADE_MANAGEMENT_REVIEW_V1"],
            "decision_anchor_timestamp_utc_v1": [timestamps[1].isoformat()],
            "action_label_v1": ["HOLD"],
            "split_bucket_v1": ["TRAIN"],
            "hindsight_policy_action_projection_kind_v1": ["DIRECT_ACTUAL_EXIT_DECISION"],
            "as_of_join_mode_v1": ["EXACT"],
            "as_of_join_source_v1": ["trade_journal_exact_close_ts_utc"],
        }
    )
    decision_bridge = pd.DataFrame(
        {
            "candidate_uid": ["c1"],
            "trade_uid": ["t1"],
            "trade_id": ["1"],
            "hindsight_decision_anchor_domain_v1": ["TRADE_MANAGEMENT_REVIEW_V1"],
            "hindsight_decision_anchor_type_v1": ["ACTUAL_EXIT_DECISION_ANCHOR"],
            "hindsight_decision_anchor_timestamp_utc_v1": [timestamps[1].isoformat()],
            "hindsight_policy_action_v1": ["DIAGNOSTIC_ONLY"],
            "hindsight_policy_action_trainable_v1": [False],
            "hindsight_policy_action_projection_kind_v1": ["DIAGNOSTIC_ONLY"],
        }
    )
    supervision_join = pd.DataFrame(
        {
            "candidate_uid": ["other-candidate"],
            "trade_uid": ["other-trade-uid"],
            "trade_id": ["other-trade-id"],
            "as_of_row_uid_v1": ["mgmt-row-other"],
            "hindsight_decision_anchor_type_v1": ["ACTUAL_EXIT_DECISION_ANCHOR"],
            "hindsight_decision_anchor_timestamp_utc_v1": [timestamps[1].isoformat()],
            "hindsight_policy_action_v1": ["HOLD"],
            "hindsight_policy_action_projection_kind_v1": ["DIRECT_ACTUAL_EXIT_DECISION"],
            "hindsight_policy_action_domain_v1": ["MANAGEMENT"],
            "as_of_split_bucket_v1": ["TRAIN"],
        }
    )

    payload = shadow_meta._build_entry_actualization_handoff_v1(
        entry_direct_policy_composite_v1_df=entry_direct,
        entry_wait_lifecycle_view_v1_df=wait_lifecycle,
        entry_confirmation_followthrough_v1_df=confirmation_followthrough,
        entry_policy_training_df=entry_direct.assign(action_label_v1=["TAKE_NOW"]),
        management_policy_training_v4_df=management_v4,
        decision_moment_teacher_bridge_df=decision_bridge,
        policy_action_supervision_join_df=supervision_join,
        entry_wait_lifecycle_status_v1={"WAIT_CONFIRMATION_COVERAGE_STATUS": "NOT_APPLICABLE"},
        entry_policy_stack_status_v1={
            "ENTRY_TIMING_STATUS": "CANONICAL_BRANCH_FROZEN",
            "ENTRY_SKIPABILITY_STATUS": "CANONICAL_BRANCH_FROZEN",
        },
    )

    audit_df = payload["entry_actual_take_to_management_handoff_audit_v1_df"]
    summary = payload["entry_actual_take_to_management_handoff_summary_v1"]
    status = payload["entry_actualization_status_v1"]

    assert len(audit_df) == 1
    assert audit_df.loc[0, "management_handoff_status_v1"] == "ACTUAL_TAKE_WITH_MANAGEMENT_DIAGNOSTIC_ONLY_REVIEW"
    assert bool(audit_df.loc[0, "management_bridge_review_present_v1"]) is True
    assert bool(audit_df.loc[0, "management_bridge_diagnostic_only_v1"]) is True
    assert pd.isna(audit_df.loc[0, "management_as_of_row_uid_v1"])
    assert pd.isna(audit_df.loc[0, "management_supervision_as_of_row_uid_v1"])
    assert audit_df.loc[0, "management_handoff_join_key_v1"] == "candidate_uid_exact_bridge_diagnostic_only"
    assert summary["management_handoff_status_counts_v1"] == {
        "ACTUAL_TAKE_WITH_MANAGEMENT_DIAGNOSTIC_ONLY_REVIEW": 1
    }
    assert summary["management_bridge_review_present_count_v1"] == 1
    assert summary["management_bridge_diagnostic_only_count_v1"] == 1
    assert status["ENTRY_TO_MANAGEMENT_HANDOFF_STATUS"] == "HANDOFF_COVERAGE_NOT_FULLY_ESTABLISHED"


def test_entry_wait_lifecycle_audit_uses_current_wait_universe_counts():
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    entry_direct = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A", "RUN_A"],
            "candidate_uid": ["c1", "c2", "c3"],
            "trade_uid": ["t1", "t2", "t3"],
            "trade_id": ["1", "2", "3"],
            "decision_timestamp": [ts.isoformat() for ts in timestamps],
            "entry_timestamp": [ts.isoformat() for ts in timestamps],
            "exit_timestamp": [(ts + pd.Timedelta(minutes=30)).isoformat() for ts in timestamps],
            "direct_entry_as_of_row_uid_v1": ["row1", "row2", "row3"],
            "split_bucket_v1": ["TRAIN", "VALIDATION", "HOLDOUT"],
            "used_for_training": [True, False, False],
            "used_for_validation": [False, True, False],
            "used_for_holdout": [False, False, True],
            "as_of_session_v1": ["US", "EU", "US"],
            "as_of_weekday_utc_v1": [3, 3, 3],
            "direct_projection_kind_v1": ["DIRECT_ENTRY_DECISION"] * 3,
            "direct_composite_status_v1": ["CANONICAL_BRANCH_FROZEN"] * 3,
            "final_direct_entry_action_v1": ["WAIT", "WAIT", "TAKE_NOW"],
            "wait_followthrough_status_v1": [
                "WAIT_WITH_PROVABLE_CONFIRMATION",
                "WAIT_CONFIRMATION_UNJOINABLE",
                "NOT_APPLICABLE",
            ],
        }
    )
    wait_followthrough = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "candidate_uid": ["c1", "c2"],
            "trade_uid": ["t1", "t2"],
            "trade_id": ["1", "2"],
            "decision_timestamp": [timestamps[0].isoformat(), timestamps[1].isoformat()],
            "entry_timestamp": [timestamps[0].isoformat(), timestamps[1].isoformat()],
            "exit_timestamp": [
                (timestamps[0] + pd.Timedelta(minutes=30)).isoformat(),
                (timestamps[1] + pd.Timedelta(minutes=30)).isoformat(),
            ],
            "direct_entry_as_of_row_uid_v1": ["row1", "row2"],
            "split_bucket_v1": ["TRAIN", "VALIDATION"],
            "used_for_training": [True, False],
            "used_for_validation": [False, True],
            "used_for_holdout": [False, False],
            "as_of_session_v1": ["US", "EU"],
            "as_of_weekday_utc_v1": [3, 3],
            "direct_projection_kind_v1": ["DIRECT_ENTRY_DECISION", "DIRECT_ENTRY_DECISION"],
            "direct_composite_status_v1": ["CANONICAL_BRANCH_FROZEN", "CANONICAL_BRANCH_FROZEN"],
            "wait_followthrough_status_v1": [
                "WAIT_WITH_PROVABLE_CONFIRMATION",
                "WAIT_CONFIRMATION_UNJOINABLE",
            ],
            "confirmation_branch_row_uid_v1": ["confirm-row-1", pd.NA],
            "confirmation_timestamp_utc_v1": [timestamps[0].isoformat(), pd.NA],
            "confirmation_action_label_v1": ["TAKE_NOW", pd.NA],
            "confirmation_delay_minutes_v1": [5.0, pd.NA],
            "reason_excluded_v1": [pd.NA, pd.NA],
            "joinable_flag": [True, False],
            "source_missing_flag": [False, False],
        }
    )
    confirmation_followthrough = pd.DataFrame({"direct_entry_as_of_row_uid_v1": ["row1"]})

    payload = shadow_meta._build_entry_wait_lifecycle_audit_v1(
        entry_hierarchical_policy_contract_v1={"layer_name": "ENTRY_HIERARCHICAL_POLICY_CONTRACT_V1"},
        entry_direct_policy_composite_v1_df=entry_direct,
        entry_wait_followthrough_audit_v1_df=wait_followthrough,
        entry_confirmation_followthrough_v1_df=confirmation_followthrough,
        entry_policy_stack_status_v1={
            "ENTRY_TIMING_STATUS": "CANONICAL_BRANCH_FROZEN",
            "MANAGEMENT_STATUS": "MANAGEMENT_GOOD_ENOUGH_FREEZE_FOR_NOW",
        },
    )

    summary = payload["entry_wait_lifecycle_audit_v1_summary"]
    contract = payload["entry_wait_lifecycle_contract_v1"]
    consistency = payload["entry_wait_lifecycle_consistency_audit_v1_summary"]
    status = payload["entry_wait_lifecycle_status_v1"]

    assert summary["wait_lifecycle_rollup_counts_v1"] == {
        "WAIT_CONFIRMATION_UNJOINABLE": 1,
        "WAIT_WITH_PROVABLE_CONFIRMATION": 1,
    }
    assert contract["confirmation_lock_v1"] == {
        "wait_with_provable_confirmation_count_v1": 1,
        "wait_without_localizable_confirmation_count_v1": 0,
        "wait_confirmation_unjoinable_count_v1": 1,
        "missing_confirmation_is_not_wait_error_v1": True,
        "no_fabricated_confirmation_take_now_v1": True,
        "no_management_without_actual_take_now_v1": True,
    }
    assert contract["wait_universe_v1"] == "The 2 direct WAIT rows from ENTRY_DIRECT_POLICY_COMPOSITE_V1"
    assert consistency["failed_check_count_v1"] == 0
    assert status["WAIT_LIFECYCLE_STATUS"] == "HINDSIGHT_AUDIT_LAYER_READY"


def test_entry_terminal_outcome_audit_uses_dynamic_component_row_counts():
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    payload = shadow_meta._build_entry_terminal_outcome_audit_v1(
        entry_actualization_contract_v1={"layer_name": "ENTRY_ACTUALIZATION_CONTRACT_V1"},
        entry_skipability_branch_v1_df=pd.DataFrame({"x": [1, 2, 3, 4]}),
        entry_timing_context_branch_v1_df=pd.DataFrame({"x": [1, 2, 3, 4, 5]}),
        entry_direct_policy_composite_v1_df=pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]}),
        entry_wait_lifecycle_view_v1_df=pd.DataFrame({"x": [1, 2]}),
        entry_actualization_route_audit_v1_df=pd.DataFrame(
            {
                "candidate_uid": ["c1", "c2"],
                "route_status_v1": ["DIRECT_TAKE_NOW_ACTUALIZED", "DIRECT_SKIP"],
            }
        ),
        entry_actual_take_view_v1_df=pd.DataFrame(
            {
                "run_id": ["RUN_A"],
                "candidate_uid": ["c1"],
                "trade_uid": ["t1"],
                "trade_id": ["1"],
                "decision_timestamp": [ts.isoformat()],
                "entry_timestamp": [ts.isoformat()],
                "exit_timestamp": [(ts + pd.Timedelta(minutes=30)).isoformat()],
                "direct_entry_as_of_row_uid_v1": ["row1"],
                "split_bucket_v1": ["TRAIN"],
                "used_for_training": [True],
                "used_for_validation": [False],
                "used_for_holdout": [False],
                "as_of_session_v1": ["US"],
                "as_of_weekday_utc_v1": [3],
                "skipability_branch_row_uid_v1": ["skip-row-1"],
                "timing_branch_row_uid_v1": ["timing-row-1"],
                "confirmation_branch_row_uid_v1": ["confirm-row-1"],
                "route_status_v1": ["DIRECT_TAKE_NOW_ACTUALIZED"],
                "activation_origin_v1": ["DIRECT_TAKE_NOW"],
                "activation_timestamp_utc_v1": [ts.isoformat()],
            }
        ),
        entry_actual_take_to_management_handoff_audit_v1_df=pd.DataFrame(
            {
                "run_id": ["RUN_A"],
                "candidate_uid": ["c1"],
                "trade_uid": ["t1"],
                "trade_id": ["1"],
                "management_handoff_status_v1": ["ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD"],
                "management_handoff_join_key_v1": ["candidate_uid_exact"],
                "management_core_v4_present_v1": [True],
                "management_supervision_join_present_v1": [True],
                "management_as_of_row_uid_v1": ["mgmt-row-1"],
                "management_anchor_type_v1": ["ACTUAL_EXIT_DECISION_ANCHOR"],
                "management_anchor_domain_v1": ["TRADE_MANAGEMENT_REVIEW_V1"],
                "management_anchor_timestamp_utc_v1": [(ts + pd.Timedelta(minutes=15)).isoformat()],
                "management_action_label_v1": ["HOLD"],
                "management_projection_kind_v1": ["DIRECT_ACTUAL_EXIT_DECISION"],
                "management_split_bucket_v1": ["TRAIN"],
            }
        ),
        management_policy_training_v4_df=pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7]}),
        closed_trades_df=pd.DataFrame(
            {
                "candidate_uid": ["c1"],
                "trade_uid": ["t1"],
                "trade_id": ["1"],
                "decision_timestamp": [ts.isoformat()],
                "entry_timestamp": [ts.isoformat()],
                "exit_timestamp": [(ts + pd.Timedelta(minutes=30)).isoformat()],
                "exit_reason": ["POSITIVE_EXIT"],
                "realized_pnl_bps": [12.0],
                "trade_outcome_class": ["positive_exit"],
                "good_trade": [True],
                "bad_trade": [False],
                "good_exit": [True],
                "premature_exit": [False],
                "late_exit": [False],
                "used_for_training": [True],
                "used_for_validation": [False],
                "used_for_holdout": [False],
                "session": ["US"],
                "weekday_utc": [3],
                "hour_utc": [12],
            }
        ),
        entry_actualization_status_v1={"ENTRY_ACTUALIZATION_STATUS": "COMPOSITIONAL_READ_MODEL_READY"},
        entry_policy_stack_status_v1={"ENTRY_COMPOSITE_STATUS": "HIERARCHICAL_COMPOSITION_READY"},
    )

    components = payload["decomposed_lifecycle_pack_manifest_v1"]["components_v1"]

    assert components["ENTRY_SKIPABILITY_HEAD_V1"]["row_count_v1"] == 4
    assert components["ENTRY_TIMING_HEAD_V1"]["row_count_v1"] == 5
    assert components["ENTRY_DIRECT_COMPOSITE_V1"]["row_count_v1"] == 6
    assert components["ENTRY_WAIT_LIFECYCLE_VIEW_V1"]["row_count_v1"] == 2
    assert components["ENTRY_ACTUAL_TAKE_VIEW_V1"]["row_count_v1"] == 1
    assert components["ENTRY_ACTUAL_TAKE_TO_MANAGEMENT_HANDOFF_AUDIT_V1"]["row_count_v1"] == 1
    assert components["ENTRY_ACTUAL_TAKE_TERMINAL_OUTCOME_VIEW_V1"]["row_count_v1"] == 1
    assert components["MANAGEMENT_HEAD_V4"]["row_count_v1"] == 7


def test_attach_management_behavior_policy_identity_uses_asof_policy_hash_when_available():
    decision_df = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "as_of_row_uid_v1": ["row-1", "row-2"],
            "candidate_uid_exact_v1": ["cand-1", "cand-2"],
            "action_label_v1": ["HOLD", "EXIT_NOW"],
        }
    )
    asof_df = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A", "RUN_A"],
            "as_of_row_uid_v1": ["row-1", "row-2", "row-3"],
            "candidate_uid": ["cand-1", "cand-2", "cand-2"],
            "as_of_candidate_policy_hash_v1": ["hash-123", "NOT_AVAILABLE", "hash-456"],
            "as_of_candidate_entry_bundle_sha256_v1": ["entry-abc", "NOT_AVAILABLE", "entry-def"],
            "as_of_candidate_exit_bundle_sha256_v1": ["exit-def", "NOT_AVAILABLE", "exit-ghi"],
        }
    )

    enriched_df, summary = shadow_meta._attach_management_behavior_policy_identity_v1(decision_df, asof_df)

    assert enriched_df.loc[0, "policy_version_v1"] == "hash-123"
    assert enriched_df.loc[0, "policy_version_status_v1"] == "EXACT_CANDIDATE_POLICY_HASH_ATTACHED"
    assert enriched_df.loc[0, "behavior_policy_identity_source_v1"] == "AS_OF_CANDIDATE_POLICY_HASH_EXACT"
    assert enriched_df.loc[1, "policy_version_v1"] == "hash-456"
    assert enriched_df.loc[1, "policy_version_status_v1"] == "EXACT_CANDIDATE_POLICY_HASH_RECOVERED_BY_CANDIDATE_UID"
    assert enriched_df.loc[1, "behavior_policy_identity_source_v1"] == "CANDIDATE_UID_POLICY_HASH_EXACT_RECOVERY"
    assert summary["policy_hash_available_rows_v1"] == 2
    assert summary["policy_hash_not_available_rows_v1"] == 0
    assert summary["behavior_policy_readiness_v1"] == "READY_EXACT_CANDIDATE_POLICY_HASH"


def test_attach_management_behavior_policy_identity_recovers_when_primary_join_is_nan():
    decision_df = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "as_of_row_uid_v1": ["row-1", "row-missing"],
            "candidate_uid_exact_v1": ["cand-1", "cand-2"],
            "action_label_v1": ["HOLD", "EXIT_NOW"],
        }
    )
    asof_df = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "as_of_row_uid_v1": ["row-1", "row-3"],
            "candidate_uid": ["cand-1", "cand-2"],
            "as_of_candidate_policy_hash_v1": ["hash-123", "hash-456"],
            "as_of_candidate_entry_bundle_sha256_v1": ["entry-abc", "entry-def"],
            "as_of_candidate_exit_bundle_sha256_v1": ["exit-abc", "exit-def"],
        }
    )

    enriched_df, summary = shadow_meta._attach_management_behavior_policy_identity_v1(decision_df, asof_df)

    assert enriched_df.loc[1, "policy_version_v1"] == "hash-456"
    assert enriched_df.loc[1, "policy_version_status_v1"] == "EXACT_CANDIDATE_POLICY_HASH_RECOVERED_BY_CANDIDATE_UID"
    assert enriched_df.loc[1, "behavior_policy_identity_source_v1"] == "CANDIDATE_UID_POLICY_HASH_EXACT_RECOVERY"
    assert summary["policy_hash_available_rows_v1"] == 2
    assert summary["policy_hash_not_available_rows_v1"] == 0


def test_attach_management_behavior_policy_identity_recovers_from_candidate_uid_exact_asof_column():
    decision_df = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "as_of_row_uid_v1": ["row-1", "row-missing"],
            "candidate_uid_exact_v1": ["cand-1", "cand-2"],
            "action_label_v1": ["HOLD", "EXIT_NOW"],
        }
    )
    asof_df = pd.DataFrame(
        {
            "run_id": ["RUN_A", "RUN_A"],
            "as_of_row_uid_v1": ["row-1", "row-3"],
            "candidate_uid_exact_v1": ["cand-1", "cand-2"],
            "as_of_candidate_policy_hash_v1": ["hash-123", "hash-456"],
            "as_of_candidate_entry_bundle_sha256_v1": ["entry-abc", "entry-def"],
            "as_of_candidate_exit_bundle_sha256_v1": ["exit-abc", "exit-def"],
        }
    )

    enriched_df, summary = shadow_meta._attach_management_behavior_policy_identity_v1(decision_df, asof_df)

    assert enriched_df.loc[1, "policy_version_v1"] == "hash-456"
    assert enriched_df.loc[1, "policy_version_status_v1"] == "EXACT_CANDIDATE_POLICY_HASH_RECOVERED_BY_CANDIDATE_UID"
    assert enriched_df.loc[1, "behavior_policy_identity_source_v1"] == "CANDIDATE_UID_POLICY_HASH_EXACT_RECOVERY"
    assert summary["policy_hash_available_rows_v1"] == 2
    assert summary["policy_hash_not_available_rows_v1"] == 0


def test_attach_management_behavior_policy_identity_recovers_from_closed_trade_lineage():
    decision_df = pd.DataFrame(
        {
            "run_id": ["RUN_A"],
            "as_of_row_uid_v1": ["row-missing"],
            "candidate_uid_exact_v1": ["cand-1"],
            "trade_uid_exact_v1": ["trade-1"],
            "trade_id_exact_v1": ["tid-1"],
            "action_label_v1": ["HOLD"],
        }
    )
    asof_df = pd.DataFrame(
        {
            "run_id": ["RUN_A"],
            "as_of_row_uid_v1": ["row-missing"],
            "candidate_uid_exact_v1": ["cand-1"],
            "as_of_candidate_policy_hash_v1": ["NOT_AVAILABLE"],
            "as_of_candidate_entry_bundle_sha256_v1": ["NOT_AVAILABLE"],
            "as_of_candidate_exit_bundle_sha256_v1": ["NOT_AVAILABLE"],
        }
    )
    closed_trades_df = pd.DataFrame(
        {
            "run_id": ["RUN_A"],
            "candidate_uid_exact_v1": ["cand-1"],
            "trade_uid_exact_v1": ["trade-1"],
            "trade_id_exact_v1": ["tid-1"],
            "policy_hash": ["hash-closed"],
        }
    )

    enriched_df, summary = shadow_meta._attach_management_behavior_policy_identity_v1(
        decision_df,
        asof_df,
        closed_trades_df,
    )

    assert enriched_df.loc[0, "policy_version_v1"] == "hash-closed"
    assert enriched_df.loc[0, "policy_version_status_v1"] == "EXACT_CLOSED_TRADE_POLICY_HASH_RECOVERED_BY_LINEAGE"
    assert enriched_df.loc[0, "behavior_policy_identity_source_v1"] == "CLOSED_TRADE_POLICY_HASH_EXACT_LINEAGE_RECOVERY"
    assert summary["policy_hash_available_rows_v1"] == 1
    assert summary["policy_hash_not_available_rows_v1"] == 0
    assert summary["closed_trade_lineage_recovery_rows_v1"] == 1


def test_attach_management_deterministic_propensity_contract_uses_policy_hash_and_observed_action():
    decision_df = pd.DataFrame(
        {
            "observed_action_v1": ["HOLD", "EXIT_NOW", "HOLD"],
            "observed_action_status_v1": [
                "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "OBSERVED_REALIZED_PATH_ACTION_EXACT",
                "OBSERVED_REALIZED_PATH_ACTION_EXACT",
            ],
            "policy_version_v1": ["hash-1", "hash-2", "NOT_AVAILABLE"],
        }
    )

    enriched_df, summary = shadow_meta._attach_management_deterministic_propensity_contract_v1(
        decision_df,
        action_space_v1=["HOLD", "EXIT_NOW"],
    )

    assert enriched_df.loc[0, "behavior_policy_id_v1"] == "hash-1"
    assert enriched_df.loc[0, "policy_logging_propensity_status_v1"] == "DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT"
    assert enriched_df.loc[0, "observed_action_propensity_v1"] == 1.0
    assert enriched_df.loc[0, "propensity_hold_v1"] == 1.0
    assert enriched_df.loc[0, "propensity_exit_now_v1"] == 0.0
    assert enriched_df.loc[1, "behavior_policy_id_v1"] == "hash-2"
    assert enriched_df.loc[1, "propensity_hold_v1"] == 0.0
    assert enriched_df.loc[1, "propensity_exit_now_v1"] == 1.0
    assert enriched_df.loc[2, "behavior_policy_id_v1"] == "NOT_AVAILABLE"
    assert (
        enriched_df.loc[2, "policy_logging_propensity_status_v1"]
        == "POLICY_HASH_NOT_AVAILABLE_PROPENSITY_NOT_ESTABLISHED"
    )
    assert summary["deterministic_propensity_rows_v1"] == 2
    assert summary["propensity_not_established_rows_v1"] == 1
    assert summary["propensity_readiness_v1"] == "PARTIAL_READY_DETERMINISTIC_LOGGED_ACTION_PROPENSITY"


def test_attach_management_deterministic_propensity_contract_falls_back_to_action_label():
    decision_df = pd.DataFrame(
        {
            "action_label_v1": ["EXIT_NOW"],
            "observed_action_status_v1": ["OBSERVED_REALIZED_PATH_ACTION_EXACT"],
            "policy_version_v1": ["hash-1"],
        }
    )

    enriched_df, summary = shadow_meta._attach_management_deterministic_propensity_contract_v1(
        decision_df,
        action_space_v1=["HOLD", "EXIT_NOW"],
    )

    assert enriched_df.loc[0, "policy_logging_propensity_status_v1"] == "DETERMINISTIC_LOGGED_POLICY_PROPENSITY_EXACT"
    assert enriched_df.loc[0, "propensity_hold_v1"] == 0.0
    assert enriched_df.loc[0, "propensity_exit_now_v1"] == 1.0
    assert summary["deterministic_propensity_rows_v1"] == 1


def test_management_exit_local_reward_baseline_sanitizes_sparse_pandas_na_in_dm_scoring():
    timestamps = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    observation_fields = list(shadow_meta._MANAGEMENT_RL_OBSERVATION_FIELDS_V1)
    exit_rows = []
    dm_rows = []

    def _base_feature_row(index: int) -> dict:
        row = {}
        for field_name in observation_fields:
            if field_name.endswith("_available_v1"):
                row[field_name] = True
            elif any(token in field_name for token in ["session", "side", "regime", "pred_side"]):
                row[field_name] = "US" if "session" in field_name else "LONG"
            else:
                row[field_name] = float(index + 1)
        row["as_of_management_xgb_pred_side_v1"] = "LONG"
        row["as_of_management_replay_session_change_flag_v1"] = False
        row["as_of_management_replay_session_tradable_v1"] = True
        row["as_of_management_exit_model_evaluated_v1"] = 1.0
        return row

    split_rows = [
        ("TRAIN", True, False, False, 12.0),
        ("TRAIN", True, False, False, -6.0),
        ("VALIDATION", False, True, False, 9.0),
        ("VALIDATION", False, True, False, -4.0),
        ("HOLDOUT", False, False, True, 15.0),
        ("HOLDOUT", False, False, True, -3.0),
    ]
    for idx, (split, is_train, is_val, is_holdout, pnl) in enumerate(split_rows, start=1):
        row = {
            "management_row_key_v1": f"exit-row-{idx}",
            "candidate_uid_exact_v1": f"c{idx}",
            "trade_uid_exact_v1": f"t{idx}",
            "trade_id_exact_v1": str(idx),
            "run_id": "RUN_A",
            "as_of_row_uid_v1": f"row-{idx}",
            "decision_timestamp": timestamps[idx - 1].isoformat(),
            "action_label_v1": "EXIT_NOW",
            "decision_anchor_type_v1": "ACTUAL_EXIT_DECISION_ANCHOR",
            "split_bucket_v1": split,
            "used_for_training": is_train,
            "used_for_validation": is_val,
            "used_for_holdout": is_holdout,
            "activation_origin_v1": "DIRECT_TAKE_NOW",
            "route_status_v1": "DIRECT_TAKE_NOW",
            "entry_actualization_presence_status_v1": "ACTUALIZED_TAKE_EXACT",
            "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
            "observed_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
            "bandit_action_reward_eligibility_status_v1": "BANDIT_DM_ELIGIBLE",
            "bandit_reward_locality_status_v1": "LOCAL_EXIT_ACTION_WITH_EXACT_TERMINAL_OUTCOME",
            "sequence_dataset_membership_v1": "STRICT_SEQUENCE_SUBSTRATE",
            "sequence_next_link_status_v1": "TERMINAL_EXIT_STEP",
            "sequence_terminal_step_status_v1": "TERMINAL_REALIZED_EXIT",
            "hindsight_reward_realized_pnl_bps_v1": pnl,
            "hindsight_reward_trade_outcome_class_v1": "positive_exit" if pnl > 0 else "negative_exit",
            "hindsight_reward_exit_reason_v1": "POSITIVE_EXIT" if pnl > 0 else "STOP_LOSS",
        }
        row.update(_base_feature_row(idx))
        exit_rows.append(row)

    management_bandit_exit_local_reward_view_v1_df = pd.DataFrame(exit_rows)
    management_bandit_direct_method_candidate_view_v1_df = management_bandit_exit_local_reward_view_v1_df.copy()
    for hold_idx in range(2):
        idx = hold_idx + 7
        split = "HOLDOUT" if hold_idx else "VALIDATION"
        row = {
            "management_row_key_v1": f"hold-row-{idx}",
            "candidate_uid_exact_v1": f"c{idx}",
            "trade_uid_exact_v1": f"t{idx}",
            "trade_id_exact_v1": str(idx),
            "run_id": "RUN_A",
            "as_of_row_uid_v1": f"row-{idx}",
            "decision_timestamp": timestamps[idx - 1].isoformat(),
            "action_label_v1": "HOLD",
            "decision_anchor_type_v1": "ACTUAL_EXIT_DECISION_ANCHOR",
            "split_bucket_v1": split,
            "used_for_training": False,
            "used_for_validation": split == "VALIDATION",
            "used_for_holdout": split == "HOLDOUT",
            "activation_origin_v1": "DIRECT_TAKE_NOW",
            "route_status_v1": "DIRECT_TAKE_NOW",
            "entry_actualization_presence_status_v1": "ACTUALIZED_TAKE_EXACT",
            "observed_action_status_v1": "OBSERVED_REALIZED_PATH_ACTION_EXACT",
            "observed_action_propensity_status_v1": "PROPENSITY_NOT_ESTABLISHED",
            "bandit_action_reward_eligibility_status_v1": "BANDIT_DM_ELIGIBLE",
            "bandit_reward_locality_status_v1": "HOLD_WITH_EPISODE_TERMINAL_RETURN_ONLY",
            "sequence_dataset_membership_v1": "BANDIT_SAFE_ONLY",
            "sequence_next_link_status_v1": "NO_EXACT_NEXT_ELIGIBLE_STEP",
            "sequence_terminal_step_status_v1": "NON_TERMINAL_HOLD",
            "hindsight_reward_realized_pnl_bps_v1": 4.0,
            "hindsight_reward_trade_outcome_class_v1": "positive_exit",
            "hindsight_reward_exit_reason_v1": "POSITIVE_EXIT",
        }
        row.update(_base_feature_row(idx))
        row["as_of_management_exit_prob_v1"] = pd.NA
        row["as_of_management_exit_prob_available_v1"] = False
        row["as_of_management_core_mfe_to_anchor_ratio_v1"] = pd.NA
        row["as_of_management_core_mfe_to_anchor_ratio_available_v1"] = False
        row["as_of_management_xgb_pred_side_v1"] = pd.NA
        dm_rows.append(row)

    management_bandit_direct_method_candidate_view_v1_df = pd.concat(
        [management_bandit_direct_method_candidate_view_v1_df, pd.DataFrame(dm_rows)],
        ignore_index=True,
    )

    payload = shadow_meta._build_management_exit_local_reward_baseline_v1(
        management_bandit_exit_local_reward_view_v1_df=management_bandit_exit_local_reward_view_v1_df,
        management_bandit_direct_method_candidate_view_v1_df=management_bandit_direct_method_candidate_view_v1_df,
        management_bandit_status_v1={
            "MANAGEMENT_BANDIT_DM_CANDIDATE_STATUS": "DIRECT_METHOD_CANDIDATE",
            "MANAGEMENT_BANDIT_OBSERVED_SAMPLE_ROW_COUNT_V1": len(
                management_bandit_direct_method_candidate_view_v1_df
            ),
            "MANAGEMENT_BANDIT_STATUS": "BANDIT_ACTION_REWARD_SUBSTRATE_ONLY",
            "MANAGEMENT_BANDIT_TRAINER_RECOMMENDATION": "EXIT_LOCAL_REWARD_BASELINE_FIRST",
        },
        as_of_supervision_join_coverage_summary={
            "exact_join_count": len(management_bandit_direct_method_candidate_view_v1_df),
            "total_rows": len(management_bandit_direct_method_candidate_view_v1_df),
        },
        leakage_guard_summary={"status": "PASS", "failed_rows": 0},
    )

    scored_df = payload["management_exit_local_all_eligible_scored_view_v1_df"]
    assert len(scored_df) == len(management_bandit_direct_method_candidate_view_v1_df)
    assert scored_df["primary_model_score_v1"].notna().all()
    assert (
        scored_df["train_action_domain_status_v1"].astype(str).eq("OUT_OF_TRAIN_ACTION_DOMAIN_RESEARCH_ONLY").sum()
        == 2
    )


def test_management_candidate_entry_spread_series_prefers_management_raw_alias():
    run_work = pd.DataFrame(
        {
            "as_of_mgmt_candidate_entry_spread_bps_v1": [1.25, 1.50],
            "as_of_candidate_entry_spread_bps_v1": [9.0, 9.0],
        }
    )

    spread = shadow_meta._management_candidate_entry_spread_series_v1(run_work)

    assert spread.tolist() == [1.25, 1.50]


def test_management_candidate_entry_spread_series_falls_back_to_legacy_alias():
    run_work = pd.DataFrame(
        {
            "as_of_candidate_entry_spread_bps_v1": [0.95, 1.05],
        }
    )

    spread = shadow_meta._management_candidate_entry_spread_series_v1(run_work)

    assert spread.tolist() == [0.95, 1.05]


def test_join_coverage_contract_allows_dynamic_exact_zero_fallback_lines():
    ok, observed, expected = shadow_meta._evaluate_join_coverage_contract_v1(
        {
            "trainable_policy_rows": 4030,
            "exact_rows": 4029,
            "fallback_rows": 0,
            "unjoinable_rows": 1,
        }
    )

    assert ok is True
    assert observed == {
        "trainable_policy_rows": 4030,
        "exact_rows": 4029,
        "fallback_rows": 0,
        "unjoinable_rows": 1,
    }
    assert expected["fallback_rows"] == 0
    assert expected["exact_rows_positive_v1"] is True
    assert expected["exact_rows_plus_unjoinable_rows_equals_trainable_policy_rows_v1"] is True
