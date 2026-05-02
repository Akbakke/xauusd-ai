from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.analysis.shadow_meta_v1 import _build_entry_skipability_pre_entry_proxy_fields_v1
from gx1.scripts.materialize_monday_entry_to_failure_pocket_bridge_v1 import (
    BRIDGE_IMPLEMENTATION_SUMMARY,
    BRIDGE_SURFACE,
    BRIDGE_SURFACE_CONTRACT,
    CONSISTENCY_AUDIT,
    FAILURE_POCKET_TAGGING_REPORT,
    FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF,
    LEGALITY_NO_POLLUTION_GUARD_REPORT,
    NEXT_ACTION,
    POST_BRIDGE_READINESS_RECHECK,
    RUNNER_NEAR_MISS_BRIDGE_READINESS,
    SUMMARY,
    materialize,
)


FORENSIC_TRADE = "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_asof_row(candidate_uid: str, run_id: str, trade_uid: str, trade_id: str, shift: float = 0.0) -> dict[str, object]:
    return {
        "candidate_uid": candidate_uid,
        "run_id": run_id,
        "trade_uid": trade_uid,
        "trade_id": trade_id,
        "entry_coverage_original_entry_observation_present_v1": True,
        "entry_coverage_original_entry_raw_state_present_v1": True,
        "entry_coverage_repair_applied_v1": False,
        "entry_coverage_repair_source_v1": "ORIGINAL_R2_ENTRY_OBSERVABILITY",
        "as_of_skip_replay_h1_range_compression_ratio_v1": 0.75 + shift,
        "as_of_skip_replay_m15_range_compression_ratio_v1": 0.65 + shift,
        "as_of_skip_replay_bb_squeeze_20_2_v1": 0.20 + shift * 0.01,
        "as_of_skip_replay_bb_bandwidth_delta_10_v1": 0.15 + shift * 0.01,
        "as_of_skip_replay_window_range_ratio_mean_5_v1": 1.25 + shift * 0.05,
        "as_of_skip_replay_window_realized_vol_3_bps_v1": 12.0 + shift,
        "as_of_skip_replay_window_realized_vol_5_bps_v1": 14.0 + shift,
        "as_of_skip_replay_d1_atr_percentile_252_v1": 0.72,
        "as_of_skip_replay_window_up_move_15_bps_v1": 15.0 + shift,
        "as_of_skip_replay_window_up_move_60_bps_v1": 35.0 + shift,
        "as_of_skip_replay_window_up_move_240_bps_v1": 55.0 + shift,
        "as_of_skip_replay_window_down_move_15_bps_v1": 5.0 + shift * 0.1,
        "as_of_skip_replay_window_down_move_60_bps_v1": 8.0 + shift * 0.1,
        "as_of_skip_replay_window_down_move_240_bps_v1": 12.0 + shift * 0.1,
        "as_of_skip_replay_window_directional_imbalance_15_bps_v1": 10.0 + shift,
        "as_of_skip_replay_window_directional_imbalance_60_bps_v1": 27.0 + shift,
        "as_of_skip_replay_window_directional_imbalance_240_bps_v1": 43.0 + shift,
        "as_of_skip_replay_window_close_in_range_15_v1": 0.85,
        "as_of_skip_replay_window_close_in_range_60_v1": 0.78,
        "as_of_skip_replay_window_close_in_range_240_v1": 0.71,
        "as_of_skip_replay_micro_momentum_3_v1": 2.2 + shift * 0.1,
        "as_of_skip_replay_micro_momentum_5_v1": 3.4 + shift * 0.1,
        "as_of_skip_replay_micro_acceleration_v1": 1.1 + shift * 0.05,
        "as_of_skip_replay_dist_last_swing_high_atr_v1": 0.8 + shift * 0.05,
        "as_of_skip_replay_dist_last_swing_low_atr_v1": 1.7 + shift * 0.05,
        "as_of_skip_replay_bars_since_swing_high_v1": 7,
        "as_of_skip_replay_bars_since_swing_low_v1": 21,
        "as_of_skip_replay_retracement_from_last_impulse_v1": 0.41,
        "as_of_skip_replay_distance_ema_fast_v1": 0.18 + shift * 0.01,
        "as_of_skip_replay_d1_dist_from_ema200_atr_v1": 0.45 + shift * 0.02,
        "as_of_skip_replay_close_in_bar_v1": 0.78,
        "as_of_skip_replay_body_share_v1": 0.58,
        "as_of_skip_replay_upper_wick_share_v1": 0.12,
        "as_of_skip_replay_lower_wick_share_v1": 0.11,
        "as_of_skip_replay_minutes_to_next_session_boundary_v1": 42.0,
        "as_of_skip_replay_session_change_flag_v1": 0.0,
        "as_of_skip_replay_spread_bps_v1": 6.0 + shift * 0.2,
        "as_of_skip_candidate_p_hat_v1": 0.84 - shift * 0.01,
        "as_of_skip_candidate_margin_v1": 0.18 - shift * 0.005,
        "as_of_skip_candidate_path_quality_pred_v1": 0.70 - shift * 0.01,
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, int, set[str]]:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    selected_pack = reports_root / "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_V1_20260424T135846Z"
    selected_pack.mkdir()
    _write_json(selected_pack / "summary_v1.json", {"readiness_decision_v1": "READY_FOR_MORE_NARROW_FEATURE_HARDENING"})

    exact_rows = pd.DataFrame(
        [
            _base_asof_row("A", "RUN_A", "TRADE_A", "TID_A", 0.0),
            _base_asof_row("B", "RUN_B", "TRADE_B", "TID_B", 1.0),
        ]
    )
    exact_proxy_df = _build_entry_skipability_pre_entry_proxy_fields_v1(exact_rows)
    raw_state_df = pd.concat(
        [exact_rows[["candidate_uid", "run_id", "trade_uid", "trade_id"]].reset_index(drop=True), exact_proxy_df.reset_index(drop=True)],
        axis=1,
    )

    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    ledger_dir.mkdir()
    raw_state_df.to_parquet(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)

    asof_rows = pd.DataFrame(
        [
            _base_asof_row("A", "RUN_A", "TRADE_A", "TID_A", 0.0),
            _base_asof_row("B", "RUN_B", "TRADE_B", "TID_B", 1.0),
            {
                **_base_asof_row(FORENSIC_TRADE, "TRUTH_MONFRI_WEEK_20260330_20260406", "TRADE_C", "TID_C", 2.0),
                "entry_coverage_original_entry_observation_present_v1": False,
                "entry_coverage_original_entry_raw_state_present_v1": False,
                "entry_coverage_repair_applied_v1": True,
                "entry_coverage_repair_source_v1": "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT",
            },
            {
                **_base_asof_row("D", "RUN_D", "TRADE_D", "TID_D", 3.0),
                "entry_coverage_original_entry_observation_present_v1": False,
                "entry_coverage_original_entry_raw_state_present_v1": False,
                "entry_coverage_repair_applied_v1": True,
                "entry_coverage_repair_source_v1": "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT",
            },
        ]
    )

    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
    r6_dir.mkdir()
    asof_rows.to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet", index=False)

    policy_df = pd.DataFrame(
        [
            {"candidate_uid": "A", "run_id": "RUN_A", "trade_uid": "TRADE_A", "trade_id": "TID_A", "is_repaired_165_v1": False, "fifty_plus_mfe_v1": True},
            {"candidate_uid": FORENSIC_TRADE, "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406", "trade_uid": "TRADE_C", "trade_id": "TID_C", "is_repaired_165_v1": True, "fifty_plus_mfe_v1": True},
        ]
    )
    policy_df.to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet", index=False)

    hindsight_df = pd.DataFrame(
        [
            {"candidate_uid": "B", "run_id": "RUN_B", "trade_uid": "TRADE_B", "trade_id": "TID_B", "r6_label_runner_near_miss_v1": True, "r6_label_tail_control_10_50_v1": False, "r6_label_missed_should_not_take_v1": False, "r6_label_risky_allow_v1": False, "r6_label_repaired_165_like_runner_v1": False},
            {"candidate_uid": FORENSIC_TRADE, "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406", "trade_uid": "TRADE_C", "trade_id": "TID_C", "r6_label_runner_near_miss_v1": True, "r6_label_tail_control_10_50_v1": False, "r6_label_missed_should_not_take_v1": False, "r6_label_risky_allow_v1": False, "r6_label_repaired_165_like_runner_v1": True},
            {"candidate_uid": "D", "run_id": "RUN_D", "trade_uid": "TRADE_D", "trade_id": "TID_D", "r6_label_runner_near_miss_v1": True, "r6_label_tail_control_10_50_v1": True, "r6_label_missed_should_not_take_v1": True, "r6_label_risky_allow_v1": True, "r6_label_repaired_165_like_runner_v1": False},
        ]
    )
    hindsight_df.to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet", index=False)

    return reports_root, len(raw_state_df), set(raw_state_df["candidate_uid"].astype("string"))


def test_materialize_monday_entry_to_failure_pocket_bridge(tmp_path: Path) -> None:
    reports_root, raw_before_count, raw_before_keys = _build_fixture(tmp_path)
    extension_dir = reports_root / "pack"

    result = materialize(reports_root, extension_dir=extension_dir)
    assert result["status"]["SPEC_STATUS"] == "IMPLEMENTED_AND_AUDITED"

    for artifact in [
        BRIDGE_SURFACE,
        BRIDGE_IMPLEMENTATION_SUMMARY,
        BRIDGE_SURFACE_CONTRACT,
        FAILURE_POCKET_TAGGING_REPORT,
        LEGALITY_NO_POLLUTION_GUARD_REPORT,
        FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF,
        RUNNER_NEAR_MISS_BRIDGE_READINESS,
        POST_BRIDGE_READINESS_RECHECK,
        NEXT_ACTION,
        SUMMARY,
        CONSISTENCY_AUDIT,
    ]:
        assert (extension_dir / artifact).exists()

    bridge_df = pd.read_parquet(extension_dir / BRIDGE_SURFACE)
    assert len(bridge_df) == 4
    assert int(bridge_df["bridge_surface_origin_v1"].astype("string").eq("EXACT_CANONICAL_RAW_STATE").sum()) == 2
    assert int(bridge_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY").sum()) == 2
    for field_name in [
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    ]:
        assert field_name in bridge_df.columns
        assert bridge_df.loc[
            bridge_df["bridge_surface_origin_v1"].astype("string").eq("FULLCOVERAGE_R6_ASOF_BRIDGE_ONLY"),
            field_name,
        ].notna().all()

    raw_after_df = pd.read_parquet(reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411" / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet")
    assert len(raw_after_df) == raw_before_count
    assert set(raw_after_df["candidate_uid"].astype("string")) == raw_before_keys

    legality = pd.read_csv(extension_dir / LEGALITY_NO_POLLUTION_GUARD_REPORT)
    assert not legality["status_v1"].astype("string").eq("FAIL").any()

    pockets = pd.read_csv(extension_dir / FAILURE_POCKET_TAGGING_REPORT).set_index("pocket_id_v1")
    assert int(pockets.loc["repaired_165", "readiness_trackable_count_v1"]) == 1
    assert int(pockets.loc["repaired_165", "bridge_only_visible_count_v1"]) == 1
    assert int(pockets.loc["runner_near_miss", "readiness_trackable_count_v1"]) == 3
    assert int(pockets.loc["runner_near_miss", "exact_only_visible_count_v1"]) == 1
    assert int(pockets.loc["runner_near_miss", "bridge_only_visible_count_v1"]) == 2
    assert int(pockets.loc["fifty_plus_mfe_seed", "readiness_trackable_count_v1"]) == 2

    forensic = json.loads((extension_dir / FORENSIC_REPAIRED_TRADE_BRIDGE_PROOF).read_text(encoding="utf-8"))
    assert forensic["exists_on_bridge_surface_v1"] is True
    assert forensic["readiness_trackable_v1"] is True

    runner = json.loads((extension_dir / RUNNER_NEAR_MISS_BRIDGE_READINESS).read_text(encoding="utf-8"))
    assert runner["fully_accounted_for_v1"] is True
    assert runner["readiness_trackable_count_v1"] == 3

    readiness = json.loads((extension_dir / POST_BRIDGE_READINESS_RECHECK).read_text(encoding="utf-8"))
    assert readiness["decision_v1"] == "READY_FOR_RETRAIN_READINESS_RECHECK"
    assert readiness["retrain_now_v1"] is False

    next_action = json.loads((extension_dir / NEXT_ACTION).read_text(encoding="utf-8"))
    assert next_action["primary_action_v1"] == "RUN_RETRAIN_READINESS_RECHECK_NEXT"
    assert "DO_NOT_RETRAIN_YET" in next_action["supporting_actions_v1"]

    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].astype("string").eq("FAIL").any()
