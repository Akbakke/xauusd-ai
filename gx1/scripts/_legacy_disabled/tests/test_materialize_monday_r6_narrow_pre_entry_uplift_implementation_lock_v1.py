from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_narrow_pre_entry_uplift_implementation_lock_v1 import (
    CONSISTENCY_AUDIT,
    LEGALITY_TEST_PLAN,
    NARROW_PLAN,
    NEXT_ACTION,
    POST_IMPLEMENTATION_PLAN,
    PROXY_CONTRACTS,
    READINESS_GATE,
    RUNNER_GUARD_SPEC,
    SUMMARY,
    WIRING_PLAN,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path) -> Path:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    prior_spec_dir = reports_root / "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_AND_RETRAIN_PREREQS_LOCK_V1_20260424T122237Z"
    prior_spec_dir.mkdir()
    pd.DataFrame(
        [
            {"feature_or_family_v1": "last_peak_ts", "classification_v1": "NOT_LEGAL_FOR_ENTRY"},
            {"feature_or_family_v1": "last_mfe_ts", "classification_v1": "NOT_LEGAL_FOR_ENTRY"},
            {"feature_or_family_v1": "last_peak_mfe", "classification_v1": "NOT_LEGAL_FOR_ENTRY"},
            {"feature_or_family_v1": "max_mfe_without_mae", "classification_v1": "NOT_LEGAL_FOR_ENTRY"},
            {"feature_or_family_v1": "mfe_mae_sequence_order", "classification_v1": "NOT_LEGAL_FOR_ENTRY"},
            {"feature_or_family_v1": "management_policy_scores_or_decision_log_fields", "classification_v1": "NOT_LEGAL_FOR_ENTRY"},
        ]
    ).to_csv(prior_spec_dir / "entry_feature_legality_boundary_lock_v1.csv", index=False)
    pd.DataFrame(
        [
            {"candidate_name_v1": "pre_entry_volatility_expansion_compression_stack_v1", "legality_v1": "PRE_ENTRY_LEGAL", "expected_value_v1": "HIGH", "priority_v1": "HIGH"},
            {"candidate_name_v1": "pre_entry_directional_asymmetry_proxy_v1", "legality_v1": "PRE_ENTRY_LEGAL", "expected_value_v1": "HIGH", "priority_v1": "HIGH"},
            {"candidate_name_v1": "pre_entry_swing_retracement_alignment_v1", "legality_v1": "PRE_ENTRY_LEGAL", "expected_value_v1": "HIGH", "priority_v1": "HIGH"},
            {"candidate_name_v1": "pre_entry_tail_leakage_pocket_proxy_v1", "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED", "expected_value_v1": "MEDIUM_HIGH", "priority_v1": "HIGH"},
            {"candidate_name_v1": "runner_protection_guard_score_v1", "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED", "expected_value_v1": "HIGH", "priority_v1": "HIGH"},
            {"candidate_name_v1": "pre_entry_session_pocket_runner_expectancy_v1", "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED", "expected_value_v1": "MEDIUM_HIGH", "priority_v1": "HIGH"},
            {"candidate_name_v1": "pre_entry_adverse_first_risk_proxy_v1", "legality_v1": "PRE_ENTRY_LEGAL_IF_DERIVED", "expected_value_v1": "HIGH", "priority_v1": "HIGH"},
            {"candidate_name_v1": "spread_cost_pressure_hardening_v1", "legality_v1": "PRE_ENTRY_LEGAL", "expected_value_v1": "MEDIUM", "priority_v1": "MEDIUM"},
        ]
    ).to_csv(prior_spec_dir / "legal_pre_entry_path_context_candidates_v1.csv", index=False)
    _write_json(
        prior_spec_dir / "repaired_165_and_runner_pocket_protection_lock_v1.json",
        {
            "repaired_165_zero_tolerance_v1": True,
            "explicit_next_eval_gates_v1": [
                "repaired_165_damage == 0",
                "50+ MFE blocked <= 1",
                "100+ MFE blocked == 0",
                "200+ MFE blocked == 0",
                "strongest_winner_path_damage == 0",
            ],
        },
    )
    _write_json(
        prior_spec_dir / "retrain_prerequisites_lock_v1.json",
        {"decision_v1": "READY_FOR_NARROW_IMPLEMENTATION_PHASE", "retrain_now_v1": False},
    )
    _write_json(prior_spec_dir / "summary_v1.json", {"next_action_v1": "TRANSLATE_PATH_DYNAMICS_TO_LEGAL_PRE_ENTRY_PROXIES"})

    prior_diag_dir = reports_root / "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_20260424T120208Z"
    prior_diag_dir.mkdir()
    _write_json(
        prior_diag_dir / "repaired_165_damage_forensic_v1.json",
        {
            "deterministic_trade_key_v1": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
            "take_was_ok_v1": True,
            "label_should_not_take_v1": False,
        },
    )
    pd.DataFrame(
        [
            {"bucket_id_v1": "MISSED_SHOULD_NOT_TAKE", "count_v1": 462},
            {"bucket_id_v1": "MISSED_10_50_TAIL_CONTROL", "count_v1": 198},
            {"bucket_id_v1": "RISKY_ALLOW", "count_v1": 347},
            {"bucket_id_v1": "RUNNER_NEAR_MISS", "count_v1": 83},
        ]
    ).to_csv(prior_diag_dir / "failure_backlog_gap_map_v1.csv", index=False)

    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    ledger_dir.mkdir()
    raw_df = pd.DataFrame(
        [
            {
                "candidate_uid": "A",
                "as_of_skip_replay_h1_range_compression_ratio_v1": 0.9,
                "as_of_skip_replay_m15_range_compression_ratio_v1": 0.8,
                "as_of_skip_replay_bb_squeeze_20_2_v1": 0.2,
                "as_of_skip_replay_bb_bandwidth_delta_10_v1": 0.1,
                "as_of_skip_replay_window_range_ratio_mean_5_v1": 1.1,
                "as_of_skip_replay_window_realized_vol_3_bps_v1": 12.0,
                "as_of_skip_replay_window_realized_vol_5_bps_v1": 15.0,
                "as_of_skip_replay_d1_atr_percentile_252_v1": 0.7,
                "as_of_skip_replay_window_up_move_15_bps_v1": 10.0,
                "as_of_skip_replay_window_up_move_60_bps_v1": 25.0,
                "as_of_skip_replay_window_up_move_240_bps_v1": 45.0,
                "as_of_skip_replay_window_down_move_15_bps_v1": 5.0,
                "as_of_skip_replay_window_down_move_60_bps_v1": 9.0,
                "as_of_skip_replay_window_down_move_240_bps_v1": 12.0,
                "as_of_skip_replay_window_directional_imbalance_15_bps_v1": 5.0,
                "as_of_skip_replay_window_directional_imbalance_60_bps_v1": 16.0,
                "as_of_skip_replay_window_directional_imbalance_240_bps_v1": 33.0,
                "as_of_skip_replay_window_close_in_range_15_v1": 0.8,
                "as_of_skip_replay_window_close_in_range_60_v1": 0.7,
                "as_of_skip_replay_window_close_in_range_240_v1": 0.6,
                "as_of_skip_replay_micro_momentum_3_v1": 2.0,
                "as_of_skip_replay_micro_momentum_5_v1": 3.0,
                "as_of_skip_replay_micro_acceleration_v1": 1.0,
                "as_of_skip_replay_dist_last_swing_high_atr_v1": 0.5,
                "as_of_skip_replay_dist_last_swing_low_atr_v1": 1.5,
                "as_of_skip_replay_bars_since_swing_high_v1": 6,
                "as_of_skip_replay_bars_since_swing_low_v1": 18,
                "as_of_skip_replay_retracement_from_last_impulse_v1": 0.35,
                "as_of_skip_replay_distance_ema_fast_v1": 0.2,
                "as_of_skip_replay_d1_dist_from_ema200_atr_v1": 0.4,
                "as_of_skip_replay_close_in_bar_v1": 0.75,
                "as_of_skip_replay_body_share_v1": 0.55,
                "as_of_skip_replay_upper_wick_share_v1": 0.15,
                "as_of_skip_replay_lower_wick_share_v1": 0.10,
                "as_of_skip_replay_minutes_to_next_session_boundary_v1": 45.0,
                "as_of_skip_replay_session_change_flag_v1": 0,
                "as_of_skip_replay_spread_bps_v1": 6.0,
                "as_of_skip_candidate_p_hat_v1": 0.82,
                "as_of_skip_candidate_margin_v1": 0.15,
                "as_of_skip_candidate_path_quality_pred_v1": 0.66,
            }
        ]
    )
    raw_df.to_parquet(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)
    pd.DataFrame([{"candidate_uid": "A", "as_of_hour_utc_v1": 13}]).to_parquet(
        ledger_dir / "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet", index=False
    )
    return reports_root


def test_materialize_monday_r6_narrow_pre_entry_uplift_lock_materializes(tmp_path: Path) -> None:
    reports_root = _build_fixture(tmp_path)
    extension_dir = reports_root / "spec"
    result = materialize(reports_root, extension_dir=extension_dir)
    assert result["status"]["SPEC_STATUS"] == "MATERIALIZED_READ_ONLY"
    for artifact in [
        NARROW_PLAN,
        PROXY_CONTRACTS,
        RUNNER_GUARD_SPEC,
        WIRING_PLAN,
        LEGALITY_TEST_PLAN,
        READINESS_GATE,
        POST_IMPLEMENTATION_PLAN,
        NEXT_ACTION,
        SUMMARY,
        CONSISTENCY_AUDIT,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["implementation_readiness_v1"] == "READY_TO_IMPLEMENT_NARROW_FEATURES"
    plan = pd.read_csv(extension_dir / NARROW_PLAN)
    assert plan["implementation_decision_v1"].astype("string").eq("SELECT_NOW").sum() == 5
    proxy = pd.read_csv(extension_dir / PROXY_CONTRACTS)
    assert proxy["feature_name_v1"].astype("string").eq("as_of_pre_entry_runner_protection_guard_score_v1").any()
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
