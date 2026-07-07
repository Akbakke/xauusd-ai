from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r4_five_microtest_shadow_bakeoff_v1 import (
    AS_OF_TABLE,
    COMPONENT_ABLATION,
    CONSISTENCY_AUDIT,
    HINDSIGHT_TABLE,
    MICROTEST_RESULTS,
    SUMMARY,
    TAIL_CONTROL,
    WALKFORWARD_ROBUSTNESS,
    WINNER_GUARD_STRESS,
    materialize,
)
from gx1.scripts.materialize_r4_fullcoverage_policy_recalibration_and_shadow_replay_v1 import (
    materialize as materialize_fullcoverage,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _run_ids(count: int) -> list[str]:
    starts = pd.date_range("2025-01-01", periods=count, freq="7D")
    ends = starts + pd.Timedelta(days=7)
    return [f"E2E_SANITY_ORDERFIX_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}" for s, e in zip(starts, ends)]


def _build_fullcoverage_fixture(tmp_path: Path) -> tuple[Path, Path]:
    reports_root = tmp_path / "reports"
    readiness_dir = reports_root / "readiness"
    r3_dir = reports_root / "r3"
    r4_dir = reports_root / "r4"
    fullcoverage_dir = reports_root / "fullcoverage"
    for directory in [readiness_dir, r3_dir, r4_dir, reports_root / "runs"]:
        directory.mkdir(parents=True)
    run_ids = _run_ids(5)
    for run_id in run_ids:
        (reports_root / "runs" / run_id).mkdir()

    asof_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    r3_rows: list[dict[str, object]] = []
    r4_rows: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []
    for idx in range(20):
        uid = f"cand-{idx:03d}"
        run_id = run_ids[idx % len(run_ids)]
        repaired = idx in {17, 18, 19}
        should = idx in {0, 1, 2, 3, 4, 5, 6}
        strong = idx in {14, 15, 17, 18, 19}
        two_hundred = idx in {17, 19}
        peak_mfe = 240.0 if two_hundred else (120.0 if strong else (25.0 if should else 42.0))
        mae = 180.0 if should else (12.0 if strong else 28.0)
        pnl = -95.0 if should else 50.0
        p_should = 0.74 if idx in {0, 1, 2, 3} else (0.52 if idx in {4, 5, 6} else 0.12)
        p_mae = 0.86 if idx in {0, 1, 2, 3} else (0.78 if idx in {4, 5, 6} else 0.20)
        p_wait = 0.82 if idx in {4, 5, 6} else 0.10
        p_strong = 0.90 if strong else 0.20
        p_direct = 0.30 if should else 0.82
        r2_block = idx in {0, 1, 2, 14}
        r3_block = idx in {0, 1, 4}
        r4_block = idx in {0, 1, 2}
        asof_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{idx + 1:02d}T12:00:00+00:00",
                "used_for_training": idx < 12,
                "used_for_validation": 12 <= idx < 16,
                "used_for_holdout": idx >= 16,
                "entry_observation_present_v1": True,
                "entry_raw_state_present_v1": True,
                "entry_coverage_original_entry_observation_present_v1": not repaired,
                "entry_coverage_original_entry_raw_state_present_v1": not repaired,
                "entry_coverage_repair_applied_v1": repaired,
                "entry_coverage_repair_source_v1": "RUN_SHADOW_META_CANDIDATES_PLUS_REPLAY_CHUNK_PLUS_XGB_EXACT" if repaired else "ORIGINAL_R2_ENTRY_OBSERVABILITY",
                "as_of_signal_v1": float(idx),
            }
        )
        label_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{idx + 1:02d}T12:00:00+00:00",
                "hindsight_entry_decision_review_v1": "SHOULD_SKIP_TRADE" if should else "TAKE_WAS_OK",
                "hindsight_management_review_v1": "MANAGED_OK",
                "trade_outcome_class": "never_mfe" if should else "positive_exit",
                "exit_reason": "THRESHOLD",
                "session": "US",
                "vol_regime": "HIGH",
                "trend_regime": "TREND_NEUTRAL",
                "baseline_realized_pnl_bps_v1": pnl,
                "peak_mfe_bps_v1": peak_mfe,
                "mae_abs_bps_v1": mae,
                "giveback_bps_v1": 20.0,
                "harvest_capture_ratio_v1": 0.7 if strong else 0.2,
                "exit_harvest_policy_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if should else "KEEP_BASELINE",
                "home_run_200bps_opportunity_v1": peak_mfe >= 200,
                "runner_100bps_opportunity_v1": peak_mfe >= 100,
                "runner_50bps_opportunity_v1": peak_mfe >= 50,
                "label_should_not_take_v1": should,
                "label_immediate_mae_risk_v1": should,
                "label_wait_would_have_helped_v1": idx in {4, 5, 6},
                "label_good_mfe_bad_capture_v1": False,
                "label_low_mfe_low_value_v1": should,
                "label_strong_trade_candidate_v1": strong,
                "label_direct_take_ok_v1": not should,
            }
        )
        r3_rows.append(
            {
                "candidate_uid": uid,
                "entry_r3_feature_available_v1": True,
                "entry_r3_shadow_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW" if r3_block else "ENTRY_ALLOW_BASELINE_SHADOW",
                "entry_r3_shadow_action_source_v1": "fixture",
                "pred__entry_r3_should_not_take__prob_true_v1": p_should,
                "pred__entry_r3_immediate_mae_risk__prob_true_v1": p_mae,
                "pred__entry_r3_wait_would_have_helped__prob_true_v1": p_wait,
                "pred__entry_r3_strong_trade_candidate__prob_true_v1": p_strong,
                "pred__entry_r3_direct_take_ok__prob_true_v1": p_direct,
                "pred__entry_r3_good_mfe_bad_capture__prob_true_v1": 0.1,
            }
        )
        r4_rows.append(
            {
                "candidate_uid": uid,
                "r2_entry_fallback_row_v1": r2_block,
                "r2_entry_fallback_correct_v1": bool(r2_block and should),
                "r3_conservative_blocks_v1": r3_block,
                "r4_entry_fallback_block_v1": r4_block,
                "r4_entry_fallback_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW" if r4_block else "ENTRY_ALLOW_BASELINE_SHADOW",
                "r4_entry_fallback_source_v1": "fixture",
            }
        )
        if repaired:
            repair_rows.append(
                {
                    "candidate_uid": uid,
                    "repair_timestamp_utc_v1": "2025-01-01T00:00:00+00:00",
                    "replay_timestamp_utc_v1": "2025-01-01T00:00:00+00:00",
                    "xgb_timestamp_utc_v1": "2025-01-01T00:00:00+00:00",
                    "entry_coverage_repair_status_v1": "RECOVERED_EXACT_RUN_SOURCE",
                    "synthetic_value_used_v1": False,
                    "hindsight_label_used_for_as_of_repair_v1": False,
                    "recovery_source_v1": "shadow_meta_candidates + replay/chunk_0/chunk_0_data + xgb_multi_horizon_predictions",
                    "entry_gap_reason_code_v1": "missing entry observation",
                    "coverage_gap_scope_v1": "MISSING_ENTRY_ONLY",
                }
            )

    pd.DataFrame(asof_rows).to_parquet(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet", index=False)
    pd.DataFrame(label_rows).to_parquet(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet", index=False)
    pd.DataFrame(repair_rows).to_csv(readiness_dir / "shadow_meta_all_trade_review_entry_coverage_repair_audit_v1.csv", index=False)
    pd.DataFrame(r3_rows).to_parquet(r3_dir / "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet", index=False)
    pd.DataFrame(r4_rows).to_parquet(r4_dir / "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_policy_prediction_view_v1.parquet", index=False)

    _write_json(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json", {"as_of_feature_names_v1": ["as_of_signal_v1"]})
    _write_json(
        readiness_dir / "shadow_meta_all_trade_review_entry_coverage_repair_summary_v1.json",
        {
            "recovered_from_as_of_decision_moment_ledger_rows_v1": 3,
            "recovered_from_run_shadow_meta_candidates_rows_v1": 3,
            "recovered_replay_chunk_exact_rows_v1": 3,
            "recovered_xgb_exact_rows_v1": 3,
        },
    )
    _write_json(r3_dir / "shadow_meta_all_trade_review_r3_entry_label_feature_summary_v1.json", {"r3_holdout_min_balanced_accuracy_v1": 0.6, "r3_policy_safety_v1": {}})
    _write_json(r4_dir / "shadow_meta_all_trade_review_r4_entry_calibrated_fallback_summary_v1.json", {"selected_policy_name_v1": "fixture", "selected_policy_metrics_v1": {}})

    materialize_fullcoverage(
        reports_root,
        readiness_dir=readiness_dir,
        r3_dir=r3_dir,
        r4_dir=r4_dir,
        extension_dir=fullcoverage_dir,
        batch_weeks=1,
        expected_ledger_count=20,
    )
    return reports_root, fullcoverage_dir


def test_materialize_r4_five_microtest_shadow_bakeoff(tmp_path: Path) -> None:
    reports_root, fullcoverage_dir = _build_fullcoverage_fixture(tmp_path)
    extension_dir = reports_root / "microtest"

    result = materialize(
        reports_root,
        fullcoverage_dir=fullcoverage_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        expected_ledger_count=20,
    )

    assert result["status"]["R4_FIVE_MICROTEST_SHADOW_BAKEOFF_STATUS"] == "MICROTEST_BAKEOFF_READY_NOT_LIVE_GATE"
    for artifact in [
        AS_OF_TABLE,
        HINDSIGHT_TABLE,
        MICROTEST_RESULTS,
        COMPONENT_ABLATION,
        WINNER_GUARD_STRESS,
        TAIL_CONTROL,
        WALKFORWARD_ROBUSTNESS,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()

    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["coverage_v1"]["entry_coverage_v1"] == 20
    assert summary["coverage_v1"]["missing_count_v1"] == 0
    assert summary["coverage_v1"]["synthetic_count_v1"] == 0
    assert summary["status_v1"]["not_live_gate"] is True
    assert summary["direction_leaderboard_v1"]

    microtests = pd.read_csv(extension_dir / MICROTEST_RESULTS)
    assert set(microtests["microtest_name_v1"]) == {
        "R4_THRESHOLD_NUDGE_TEST_V1",
        "R4_STACK_ABLATION_TEST_V1",
        "R4_WINNER_GUARD_STRESS_TEST_V1",
        "R4_TAIL_CONTROL_TARGET_TEST_V1",
        "R4_WALKFORWARD_ROBUSTNESS_TEST_V1",
    }
    assert microtests["safety_constraint_pass_v1"].notna().all()

    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
