from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.analysis.shadow_meta_v1 import (
    _build_entry_skipability_pre_entry_proxy_fields_v1,
    _validate_entry_pre_entry_proxy_input_fields_v1,
)
from gx1.scripts.materialize_monday_selected_pre_entry_proxies_and_readiness_pack_v1 import (
    CONSISTENCY_AUDIT,
    FEATURE_COVERAGE_REPORT,
    IMPLEMENTATION_SUMMARY,
    LEGALITY_TEST_REPORT,
    NEXT_ACTION,
    RAW_STATE_CONTRACT_EXTENSION,
    READINESS_RECHECK,
    RUNNER_GUARD_LOCK,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_raw_state() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_uid": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
                "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406",
                "trade_uid": "T1",
                "trade_id": "1",
                "anchor_type": "ENTRY",
                "anchor_domain": "ENTRY",
                "anchor_timestamp_utc": "2026-03-31T10:00:00Z",
                "as_of_row_uid_v1": "ROW1",
                "projection_kind_v1": "DIRECT_ENTRY_DECISION",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "split_bucket_v1": "TRAIN",
                "skip_raw_replay_exact_available_v1": True,
                "skip_raw_candidate_snapshot_exact_available_v1": True,
                "skip_raw_xgb_exact_available_v1": True,
                "skipability_raw_semantic_contract_v1": "DIRECT_ONLY",
                "as_of_skip_replay_h1_range_compression_ratio_v1": 0.75,
                "as_of_skip_replay_m15_range_compression_ratio_v1": 0.65,
                "as_of_skip_replay_bb_squeeze_20_2_v1": 0.20,
                "as_of_skip_replay_bb_bandwidth_delta_10_v1": 0.15,
                "as_of_skip_replay_window_range_ratio_mean_5_v1": 1.25,
                "as_of_skip_replay_window_realized_vol_3_bps_v1": 12.0,
                "as_of_skip_replay_window_realized_vol_5_bps_v1": 14.0,
                "as_of_skip_replay_d1_atr_percentile_252_v1": 0.72,
                "as_of_skip_replay_window_up_move_15_bps_v1": 15.0,
                "as_of_skip_replay_window_up_move_60_bps_v1": 35.0,
                "as_of_skip_replay_window_up_move_240_bps_v1": 55.0,
                "as_of_skip_replay_window_down_move_15_bps_v1": 5.0,
                "as_of_skip_replay_window_down_move_60_bps_v1": 8.0,
                "as_of_skip_replay_window_down_move_240_bps_v1": 12.0,
                "as_of_skip_replay_window_directional_imbalance_15_bps_v1": 10.0,
                "as_of_skip_replay_window_directional_imbalance_60_bps_v1": 27.0,
                "as_of_skip_replay_window_directional_imbalance_240_bps_v1": 43.0,
                "as_of_skip_replay_window_close_in_range_15_v1": 0.85,
                "as_of_skip_replay_window_close_in_range_60_v1": 0.78,
                "as_of_skip_replay_window_close_in_range_240_v1": 0.71,
                "as_of_skip_replay_micro_momentum_3_v1": 2.2,
                "as_of_skip_replay_micro_momentum_5_v1": 3.4,
                "as_of_skip_replay_micro_acceleration_v1": 1.1,
                "as_of_skip_replay_dist_last_swing_high_atr_v1": 0.8,
                "as_of_skip_replay_dist_last_swing_low_atr_v1": 1.7,
                "as_of_skip_replay_bars_since_swing_high_v1": 7,
                "as_of_skip_replay_bars_since_swing_low_v1": 21,
                "as_of_skip_replay_retracement_from_last_impulse_v1": 0.41,
                "as_of_skip_replay_distance_ema_fast_v1": 0.18,
                "as_of_skip_replay_d1_dist_from_ema200_atr_v1": 0.45,
                "as_of_skip_replay_close_in_bar_v1": 0.78,
                "as_of_skip_replay_body_share_v1": 0.58,
                "as_of_skip_replay_upper_wick_share_v1": 0.12,
                "as_of_skip_replay_lower_wick_share_v1": 0.11,
                "as_of_skip_replay_minutes_to_next_session_boundary_v1": 42.0,
                "as_of_skip_replay_session_change_flag_v1": 0.0,
                "as_of_skip_replay_spread_bps_v1": 6.0,
                "as_of_skip_candidate_p_hat_v1": 0.84,
                "as_of_skip_candidate_margin_v1": 0.18,
                "as_of_skip_candidate_path_quality_pred_v1": 0.70,
            },
            {
                "candidate_uid": "B",
                "run_id": "TRUTH_MONFRI_WEEK_20260316_20260323",
                "trade_uid": "T2",
                "trade_id": "2",
                "anchor_type": "ENTRY",
                "anchor_domain": "ENTRY",
                "anchor_timestamp_utc": "2026-03-17T10:00:00Z",
                "as_of_row_uid_v1": "ROW2",
                "projection_kind_v1": "DIRECT_ENTRY_DECISION",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "split_bucket_v1": "TRAIN",
                "skip_raw_replay_exact_available_v1": True,
                "skip_raw_candidate_snapshot_exact_available_v1": True,
                "skip_raw_xgb_exact_available_v1": True,
                "skipability_raw_semantic_contract_v1": "DIRECT_ONLY",
                "as_of_skip_replay_h1_range_compression_ratio_v1": np.nan,
                "as_of_skip_replay_m15_range_compression_ratio_v1": np.nan,
                "as_of_skip_replay_bb_squeeze_20_2_v1": np.nan,
                "as_of_skip_replay_bb_bandwidth_delta_10_v1": np.nan,
                "as_of_skip_replay_window_range_ratio_mean_5_v1": np.nan,
                "as_of_skip_replay_window_realized_vol_3_bps_v1": np.nan,
                "as_of_skip_replay_window_realized_vol_5_bps_v1": np.nan,
                "as_of_skip_replay_d1_atr_percentile_252_v1": np.nan,
                "as_of_skip_replay_window_up_move_15_bps_v1": 8.0,
                "as_of_skip_replay_window_up_move_60_bps_v1": 10.0,
                "as_of_skip_replay_window_up_move_240_bps_v1": 12.0,
                "as_of_skip_replay_window_down_move_15_bps_v1": 7.0,
                "as_of_skip_replay_window_down_move_60_bps_v1": 8.0,
                "as_of_skip_replay_window_down_move_240_bps_v1": 12.0,
                "as_of_skip_replay_window_directional_imbalance_15_bps_v1": 1.0,
                "as_of_skip_replay_window_directional_imbalance_60_bps_v1": 2.0,
                "as_of_skip_replay_window_directional_imbalance_240_bps_v1": 0.0,
                "as_of_skip_replay_window_close_in_range_15_v1": 0.52,
                "as_of_skip_replay_window_close_in_range_60_v1": 0.49,
                "as_of_skip_replay_window_close_in_range_240_v1": 0.50,
                "as_of_skip_replay_micro_momentum_3_v1": 0.1,
                "as_of_skip_replay_micro_momentum_5_v1": 0.2,
                "as_of_skip_replay_micro_acceleration_v1": 0.0,
                "as_of_skip_replay_dist_last_swing_high_atr_v1": 0.4,
                "as_of_skip_replay_dist_last_swing_low_atr_v1": 0.5,
                "as_of_skip_replay_bars_since_swing_high_v1": 1,
                "as_of_skip_replay_bars_since_swing_low_v1": 2,
                "as_of_skip_replay_retracement_from_last_impulse_v1": 0.9,
                "as_of_skip_replay_distance_ema_fast_v1": 1.8,
                "as_of_skip_replay_d1_dist_from_ema200_atr_v1": 2.5,
                "as_of_skip_replay_close_in_bar_v1": 0.49,
                "as_of_skip_replay_body_share_v1": 0.25,
                "as_of_skip_replay_upper_wick_share_v1": 0.35,
                "as_of_skip_replay_lower_wick_share_v1": 0.30,
                "as_of_skip_replay_minutes_to_next_session_boundary_v1": 5.0,
                "as_of_skip_replay_session_change_flag_v1": 1.0,
                "as_of_skip_replay_spread_bps_v1": 12.0,
                "as_of_skip_candidate_p_hat_v1": 0.40,
                "as_of_skip_candidate_margin_v1": 0.04,
                "as_of_skip_candidate_path_quality_pred_v1": 0.20,
            },
        ]
    )


def _build_fixture(tmp_path: Path) -> Path:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    lock_dir = reports_root / "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_V1_20260424T130635Z"
    lock_dir.mkdir()
    _write_json(lock_dir / "summary_v1.json", {"implementation_readiness_v1": "READY_TO_IMPLEMENT_NARROW_FEATURES"})

    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    ledger_dir.mkdir()
    raw_df = _sample_raw_state()
    raw_df.to_parquet(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "feature_name": "as_of_skip_replay_spread_bps_v1",
                "source_family": "replay_chunk_bar_exact",
                "direct_entry_coverage": 2,
                "non_null_count": 2,
                "non_null_rate": 1.0,
                "semantic_group": "CURRENT_MICROSTRUCTURE_COST",
                "dtype": "float64",
                "as_of_safe_v1": True,
                "leakage_risk_v1": "LOW",
                "source_specific_v1": False,
                "direct_only_allowed_v1": True,
                "potential_canonical_alias_group_v1": "ENTRY_SKIPABILITY::CURRENT_MICROSTRUCTURE_COST",
                "research_input_allowed_v1": True,
                "canonical_promotion_candidate_v1": True,
                "raw_state_role_v1": "DIRECT_ONLY_CANONICAL_CANDIDATE",
                "contract_note_v1": "base",
            }
        ]
    ).to_csv(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv", index=False)
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json",
        {
            "layer_name": "ENTRY_SKIPABILITY_DIRECT_STATE_EXPANSION_V1",
            "row_count": 2,
            "source_analysis_v1": {"source_family_rows": []},
            "role_counts": {"DIRECT_ONLY_CANONICAL_CANDIDATE": 1},
            "direct_only_canonical_candidate_count": 1,
        },
    )

    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
    r6_dir.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_uid": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
                "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406",
                "is_repaired_165_v1": True,
                "fifty_plus_mfe_v1": True,
                "r6_selected_candidate__block_v1": False,
            },
            {"candidate_uid": "B", "run_id": "TRUTH_MONFRI_WEEK_20260316_20260323", "is_repaired_165_v1": False, "fifty_plus_mfe_v1": False, "r6_selected_candidate__block_v1": False},
        ]
    ).to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
                "r6_label_runner_near_miss_v1": True,
                "r6_label_tail_control_10_50_v1": False,
                "r6_label_missed_should_not_take_v1": False,
                "r6_label_risky_allow_v1": False,
                "r6_label_repaired_165_like_runner_v1": True,
            },
            {
                "candidate_uid": "B",
                "r6_label_runner_near_miss_v1": False,
                "r6_label_tail_control_10_50_v1": True,
                "r6_label_missed_should_not_take_v1": True,
                "r6_label_risky_allow_v1": True,
                "r6_label_repaired_165_like_runner_v1": False,
            },
        ]
    ).to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet", index=False)
    return reports_root


def test_pre_entry_proxy_legality_guards_and_null_policy() -> None:
    _validate_entry_pre_entry_proxy_input_fields_v1(
        "positive",
        ["as_of_skip_replay_spread_bps_v1", "as_of_skip_candidate_p_hat_v1"],
    )
    try:
        _validate_entry_pre_entry_proxy_input_fields_v1("negative", ["last_peak_ts"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected illegal management/exit field to be rejected")

    raw_df = _sample_raw_state()
    base = _build_entry_skipability_pre_entry_proxy_fields_v1(raw_df)
    future_variant = raw_df.copy()
    future_variant["hindsight_peak_mfe_bps_v1"] = [100.0, 200.0]
    future_variant["policy_log_runner_protector_score_v1"] = [0.9, 0.1]
    future_scores = _build_entry_skipability_pre_entry_proxy_fields_v1(future_variant)
    assert base.fillna(-9999.0).equals(future_scores.fillna(-9999.0))

    null_variant = raw_df.copy()
    for field in [
        "as_of_skip_replay_h1_range_compression_ratio_v1",
        "as_of_skip_replay_m15_range_compression_ratio_v1",
        "as_of_skip_replay_bb_squeeze_20_2_v1",
        "as_of_skip_replay_bb_bandwidth_delta_10_v1",
        "as_of_skip_replay_window_range_ratio_mean_5_v1",
        "as_of_skip_replay_window_realized_vol_3_bps_v1",
        "as_of_skip_replay_window_realized_vol_5_bps_v1",
        "as_of_skip_replay_d1_atr_percentile_252_v1",
    ]:
        null_variant.loc[:, field] = np.nan
    null_scores = _build_entry_skipability_pre_entry_proxy_fields_v1(null_variant)
    assert bool(null_scores["as_of_pre_entry_vol_exp_comp_score_v1"].isna().all())


def test_materialize_selected_pre_entry_proxies_and_readiness_pack(tmp_path: Path) -> None:
    reports_root = _build_fixture(tmp_path)
    extension_dir = reports_root / "pack"
    result = materialize(reports_root, extension_dir=extension_dir)
    assert result["status"]["SPEC_STATUS"] == "IMPLEMENTED_AND_AUDITED"
    for artifact in [
        IMPLEMENTATION_SUMMARY,
        RAW_STATE_CONTRACT_EXTENSION,
        RUNNER_GUARD_LOCK,
        LEGALITY_TEST_REPORT,
        FEATURE_COVERAGE_REPORT,
        READINESS_RECHECK,
        NEXT_ACTION,
        SUMMARY,
        CONSISTENCY_AUDIT,
    ]:
        assert (extension_dir / artifact).exists()

    updated_raw = pd.read_parquet(reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411" / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet")
    for field_name in [
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    ]:
        assert field_name in updated_raw.columns

    legality = pd.read_csv(extension_dir / LEGALITY_TEST_REPORT)
    assert not legality["status_v1"].astype("string").eq("FAIL").any()
    coverage = pd.read_csv(extension_dir / FEATURE_COVERAGE_REPORT)
    assert set(coverage["feature_name_v1"].astype("string")) == {
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    }
    readiness = json.loads((extension_dir / READINESS_RECHECK).read_text(encoding="utf-8"))
    assert readiness["decision_v1"] in {
        "READY_FOR_RETRAIN_READINESS_RECHECK",
        "READY_FOR_MORE_NARROW_FEATURE_HARDENING",
        "WAIT_FOR_COVERAGE_FIXES",
    }
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].astype("string").eq("FAIL").any()
