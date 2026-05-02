from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r5_2_shadow_phase_gate_and_harvest_integration_v1 import (
    AS_OF_TABLE,
    CALIBRATION_AUDIT,
    CONSISTENCY_AUDIT,
    DECISION_MATRIX,
    FAILURE_MODE_TABLE,
    HARVEST_IMPACT,
    HINDSIGHT_TABLE,
    POLICY_LOGGING_EXPLAINABILITY,
    ROBUSTNESS_STRESS_MATRIX,
    SHADOW_REPLAY_BAKEOFF,
    SUMMARY,
    materialize,
)
from gx1.scripts.materialize_truth_rl_recommendation_candidate_v1 import (
    RECOMMENDATION_SUMMARY,
    RECOMMENDATION_TRADE_VIEW,
)
from gx1.scripts.materialize_truth_rl_unified_observability_v1 import (
    UNIFIED_RL_EPISODE_VIEW,
    UNIFIED_RL_SUMMARY,
)
from gx1.scripts.train_r5_2_entry_runner_aware_retrain_and_loso_selection_v1 import (
    BAD_PROB,
    RUNNER_PROB,
)
from gx1.scripts.train_truth_harvest_retrain_candidate_v1 import (
    RETRAIN_AUDIT,
    RETRAIN_PREDICTION_VIEW,
    RETRAIN_SUMMARY,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_ids(count: int) -> list[str]:
    starts = pd.date_range("2025-01-01", periods=count, freq="7D")
    ends = starts + pd.Timedelta(days=7)
    return [f"E2E_SANITY_ORDERFIX_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}" for start, end in zip(starts, ends)]


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    reports_root = tmp_path / "reports"
    r5_2_dir = reports_root / "r5_2"
    harvest_dir = reports_root / "harvest"
    rl_dir = reports_root / "rl_recommendation"
    unified_dir = reports_root / "rl_unified"
    for directory in [r5_2_dir, harvest_dir, rl_dir, unified_dir, reports_root / "runs"]:
        directory.mkdir(parents=True)
    run_ids = _run_ids(5)
    for run_id in run_ids:
        (reports_root / "runs" / run_id).mkdir()

    asof_rows: list[dict[str, object]] = []
    hindsight_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    harvest_rows: list[dict[str, object]] = []
    rl_rows: list[dict[str, object]] = []
    for idx in range(50):
        uid = f"cand-{idx:03d}"
        run_id = run_ids[idx % len(run_ids)]
        should = idx % 5 == 0
        strong_runner = idx in {1, 6, 11, 16, 21, 26}
        repaired = idx in {40, 41, 42}
        r52_block = should and idx not in {25}
        peak = 220.0 if strong_runner and idx == 1 else (80.0 if strong_runner else (30.0 if should else 22.0))
        pnl = -50.0 if should else 25.0
        asof_rows.append(
            {
                "candidate_uid": uid,
                "run_id": run_id,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "used_for_training": idx < 30,
                "used_for_validation": 30 <= idx < 40,
                "used_for_holdout": idx >= 40,
                "as_of_session_v1": "US" if idx % 2 else "EU",
                "as_of_side_v1": "LONG" if idx % 2 else "SHORT",
                "as_of_candidate_tradable_prob_v1": 0.95 if strong_runner else 0.65,
                "as_of_entry_candidate_path_quality_pred_v1": 0.85 if strong_runner else 0.45,
                "as_of_candidate_mfe_first_n_pred_v1": 2.1 if strong_runner else 1.0,
                "as_of_skip_candidate_p_flat_v1": 0.25 if strong_runner else 0.55,
                "as_of_skip_replay_window_range_15_bps_v1": 45.0 if strong_runner else 95.0,
                "as_of_skip_replay_window_realized_vol_5_bps_v1": 5.0 if strong_runner else 14.0,
                "as_of_skip_replay_retracement_from_last_impulse_v1": 0.72 if strong_runner else 0.24,
                "as_of_skip_replay_clv_v1": 0.63 if strong_runner else 0.30,
                "as_of_atr_bps_v1": 35.0,
                "entry_observation_present_v1": True,
                "entry_raw_state_present_v1": True,
                "entry_coverage_repair_applied_v1": repaired,
                "entry_coverage_repair_source_v1": "fixture_repair" if repaired else "direct",
            }
        )
        hindsight_rows.append(
            {
                "candidate_uid": uid,
                "run_id": run_id,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "baseline_realized_pnl_bps_v1": pnl,
                "peak_mfe_bps_v1": peak,
                "mae_abs_bps_v1": 60.0 if should else 8.0,
                "giveback_bps_v1": 18.0,
                "hindsight_entry_decision_review_v1": "SHOULD_SKIP_TRADE" if should else "TAKE_WAS_OK",
                "hindsight_management_review_v1": "MANAGED_OK",
                "r5_2_label_bad_blocker_v1": should,
                "r5_2_label_runner_protect_v1": strong_runner or repaired,
                "r5_2_label_runner_50_mfe_v1": strong_runner,
                "r5_2_label_runner_100_mfe_v1": idx == 1,
                "r5_2_label_runner_200_mfe_v1": idx == 1,
                "r5_2_label_repaired_165_like_runner_v1": repaired,
                "r5_2_label_strong_low_mae_runner_v1": strong_runner,
                "r5_2_label_high_mfe_tail_risk_ambiguous_v1": False,
            }
        )
        pred_rows.append(
            {
                "candidate_uid": uid,
                "run_id": run_id,
                "label_should_not_take_v1": should,
                "take_was_ok_v1": not should,
                "label_strong_trade_candidate_v1": strong_runner,
                "fifty_plus_mfe_v1": peak >= 50.0,
                "hundred_plus_mfe_v1": peak >= 100.0,
                "two_hundred_plus_mfe_v1": peak >= 200.0,
                "tail_10_50_mfe_v1": should and peak < 50.0,
                "strongest_winner_path_v1": peak >= 200.0,
                "is_repaired_165_v1": repaired,
                "r5_2_batch04_hard_negative_runner_v1": idx in {1, 6, 11, 16, 21, 26},
                "r5_2_hard_negative_like_asof_v1": strong_runner,
                "r5_2_hard_negative_similarity_distance_v1": 0.1 if strong_runner else 1.0,
                "peak_mfe_bps_v1": peak,
                "mae_abs_bps_v1": 60.0 if should else 8.0,
                "baseline_realized_pnl_bps_v1": pnl,
                BAD_PROB: 0.82 if should else 0.22,
                RUNNER_PROB: 0.92 if strong_runner or repaired else 0.20,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.85 if should else 0.20,
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.80 if should else 0.20,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.90 if strong_runner or repaired else 0.20,
                "pred__entry_r5_strong_trade_candidate__prob_true_v1": 0.90 if strong_runner else 0.15,
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.85 if should and peak < 50.0 else 0.15,
                "pred__entry_r5_take_was_ok__prob_true_v1": 0.90 if not should else 0.10,
                "no_entry_fallback_baseline__block_v1": False,
                "r2_fallback_reference__block_v1": should and idx % 2 == 0,
                "r4_current_reference__block_v1": should,
                "r5_current_reference__block_v1": should or (strong_runner and idx == 1),
                "r5_1_selected_reference__block_v1": should and idx % 2 == 0,
                "r5_2_selected_candidate__block_v1": r52_block,
            }
        )
        harvest_rows.append(
            {
                "candidate_uid": uid,
                "harvest_quality_bucket_v1": "ENTRY_OR_RISK_FILTER_FAILURE" if should else "EXIT_TOO_EARLY_UNDERHARVEST",
                "exit_harvest_policy_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if should else "HOLD_LONGER_RUNNER_TRAIL",
                "rl_priority_entry_skip_delta_bps_v1": 50.0 if should else 0.0,
                "rl_priority_exit_earlier_delta_bps_v1": 0.0,
                "rl_priority_hold_longer_delta_bps_v1": 30.0 if not should else 0.0,
                "management_rl_harvest_reward_bps_raw_v1": 50.0 if should else 30.0,
                "entry_xgb_harvest_label_v1": "REJECT_OR_LOW_SIZE" if should else "PRIORITIZE_CLEAN_RUNNER",
                "entry_xgb_binary_take_target_v1": not should,
                "exit_transformer_supervision_label_v1": "NO_EXIT_TRAINING_ENTRY_FILTER" if should else "HOLD_LONGER_OR_RUNNER_TRAIL",
                "management_rl_harvest_action_label_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if should else "HOLD_LONGER_RUNNER_TRAIL",
                "candidate_shadow_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if should else "HOLD_LONGER_RUNNER_TRAIL",
                "candidate_shadow_action_source_v1": "ENTRY_MODEL_SUPPRESS_FALLBACK" if should else "MANAGEMENT_RL_ACTION_MODEL",
                "candidate_shadow_action_matches_harvest_target_v1": True,
                "candidate_shadow_delta_bps_v1": 50.0 if should else 30.0,
                "candidate_shadow_delta_clipped_200_bps_v1": 50.0 if should else 30.0,
            }
        )
        rl_rows.append(
            {
                "candidate_uid": uid,
                "rl_entry_recommendation_v1": "SKIP_TRADE" if should else "KEEP_ENTRY_BASELINE",
                "rl_management_recommendation_v1": "KEEP_MANAGEMENT_BASELINE" if should else "HOLD_LONGER",
                "rl_priority_recommendation_v1": "SKIP_TRADE" if should else "HOLD_LONGER",
                "unified_episode_coverage_status_v1": "COVERED_BY_UNIFIED_ENTRY_EPISODE",
            }
        )

    pd.DataFrame(asof_rows).to_parquet(r5_2_dir / "shadow_meta_all_trade_review_r5_2_as_of_feature_table_v1.parquet", index=False)
    pd.DataFrame(hindsight_rows).to_parquet(r5_2_dir / "shadow_meta_all_trade_review_r5_2_hindsight_label_outcome_table_v1.parquet", index=False)
    pd.DataFrame(pred_rows).to_parquet(r5_2_dir / "shadow_meta_all_trade_review_r5_2_policy_prediction_view_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "policy_name_v1": "R5_2_SELECTED_CANDIDATE",
                "scope_v1": f"BATCH_{idx:02d}",
                "slice_safety_pass_v1": True,
                "slice_safety_failure_reasons_v1": "",
                "should_not_take_precision_v1": 1.0,
                "block_count_v1": 2,
                "should_not_take_block_count_v1": 2,
                "fifty_plus_mfe_block_count_v1": 0,
                "repaired_165_block_count_v1": 0,
                "two_hundred_plus_mfe_block_count_v1": 0,
            }
            for idx in range(1, 6)
        ]
    ).to_csv(r5_2_dir / "shadow_meta_all_trade_review_r5_2_loso_metrics_v1.csv", index=False)
    pd.DataFrame({"policy_name_v1": ["R5_2_SELECTED_CANDIDATE"], "scope_v1": ["ALL_1971"], "should_not_take_block_count_v1": [9]}).to_csv(
        r5_2_dir / "shadow_meta_all_trade_review_r5_2_head_to_head_vs_r2_r4_r5_r5_1_v1.csv", index=False
    )
    pd.DataFrame({"check_name_v1": ["fixture"], "status_v1": ["PASS"], "details_json_v1": ["{}"]}).to_csv(
        r5_2_dir / "shadow_meta_all_trade_review_r5_2_consistency_audit_v1.csv", index=False
    )
    _write_json(r5_2_dir / "shadow_meta_all_trade_review_r5_2_entry_runner_aware_contract_v1.json", {"layer_name": "fixture_contract"})
    _write_json(
        r5_2_dir / "shadow_meta_all_trade_review_r5_2_summary_v1.json",
        {
            "coverage_v1": {
                "ledger_trade_count_v1": 50,
                "entry_coverage_v1": 50,
                "entry_raw_coverage_v1": 50,
                "missing_count_v1": 0,
                "synthetic_count_v1": 0,
                "repaired_rows_v1": 3,
            },
            "selected_candidate_v1": {
                "policy_name_v1": "R5_2_CANDIDATE_FIXTURE_SELECTED",
                "stack_family_v1": "TWO_HEAD_DIRECT",
                "guard_mode_v1": "none",
                "thresholds_json_v1": json.dumps(
                    {
                        "bad_threshold_v1": 0.42,
                        "runner_threshold_v1": 0.5,
                        "tail_threshold_v1": 0.75,
                        "r5_bad_threshold_v1": 0.5,
                        "runner_margin_v1": 0.0,
                        "stack_family_v1": "TWO_HEAD_DIRECT",
                        "guard_mode_v1": "none",
                    },
                    sort_keys=True,
                ),
            },
            "decision_v1": {"recommended_next_step_v1": "fixture"},
        },
    )

    pd.DataFrame(harvest_rows).to_parquet(harvest_dir / RETRAIN_PREDICTION_VIEW, index=False)
    pd.DataFrame({"check_name_v1": ["fixture"], "status_v1": ["PASS"], "details_json_v1": ["{}"]}).to_csv(harvest_dir / RETRAIN_AUDIT, index=False)
    _write_json(harvest_dir / RETRAIN_SUMMARY, {"status_v1": {"HARVEST_RETRAIN_CANDIDATE_STATUS": "fixture"}})

    pd.DataFrame(rl_rows).to_parquet(rl_dir / RECOMMENDATION_TRADE_VIEW, index=False)
    _write_json(rl_dir / RECOMMENDATION_SUMMARY, {"status_v1": {"RL_RECOMMENDATION_CANDIDATE_STATUS": "fixture"}})
    _write_json(reports_root / "truth_rl_recommendation_candidate_v1.json", {"extension_dir_v1": str(rl_dir)})

    pd.DataFrame({"candidate_uid": [row["candidate_uid"] for row in asof_rows]}).to_parquet(unified_dir / UNIFIED_RL_EPISODE_VIEW, index=False)
    _write_json(unified_dir / UNIFIED_RL_SUMMARY, {"status_v1": {"UNIFIED_RL_OBSERVABILITY_STATUS": "fixture"}})
    _write_json(reports_root / "truth_rl_unified_observability_v1.json", {"extension_dir_v1": str(unified_dir)})
    return reports_root, r5_2_dir, harvest_dir, rl_dir, unified_dir


def test_r5_2_shadow_phase_gate_materializes(tmp_path: Path) -> None:
    reports_root, r5_2_dir, harvest_dir, rl_dir, unified_dir = _build_fixture(tmp_path)
    extension_dir = reports_root / "phase_gate"
    result = materialize(
        reports_root,
        r5_2_dir=r5_2_dir,
        harvest_dir=harvest_dir,
        rl_recommendation_dir=rl_dir,
        rl_unified_dir=unified_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        expected_ledger_count=50,
    )
    assert result["status"]["not_live_gate"] is True
    for artifact in [
        AS_OF_TABLE,
        HINDSIGHT_TABLE,
        SHADOW_REPLAY_BAKEOFF,
        ROBUSTNESS_STRESS_MATRIX,
        CALIBRATION_AUDIT,
        FAILURE_MODE_TABLE,
        HARVEST_IMPACT,
        POLICY_LOGGING_EXPLAINABILITY,
        DECISION_MATRIX,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["coverage_v1"]["entry_coverage_v1"] == 50
    assert summary["coverage_v1"]["synthetic_count_v1"] == 0
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
    harvest = pd.read_csv(extension_dir / HARVEST_IMPACT)
    assert int(harvest.loc[harvest["scope_v1"].eq("ALL_1971"), "unified_episode_covered_count_v1"].iloc[0]) == 50


def test_r5_2_phase_gate_batch05_absent_is_not_reported_as_fail(tmp_path: Path) -> None:
    reports_root, r5_2_dir, harvest_dir, rl_dir, unified_dir = _build_fixture(tmp_path)
    loso_path = r5_2_dir / "shadow_meta_all_trade_review_r5_2_loso_metrics_v1.csv"
    loso = pd.read_csv(loso_path)
    loso = loso[loso["scope_v1"].astype("string") != "BATCH_05"].copy()
    loso.to_csv(loso_path, index=False)
    extension_dir = reports_root / "phase_gate_compact"
    materialize(
        reports_root,
        r5_2_dir=r5_2_dir,
        harvest_dir=harvest_dir,
        rl_recommendation_dir=rl_dir,
        rl_unified_dir=unified_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        expected_ledger_count=50,
    )
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["decision_v1"]["batch05_loso_pass_v1"] is None
