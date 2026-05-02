from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.train_r5_2_entry_runner_aware_retrain_and_loso_selection_v1 import (
    CONSISTENCY_AUDIT,
    HARD_NEGATIVE_AUDIT,
    HEAD_TO_HEAD,
    LOSO_METRICS,
    PARETO_FRONTIER,
    SUMMARY,
    TWO_HEAD_STACK_BAKEOFF,
    materialize,
)
from gx1.scripts.train_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1 import R5_PROB


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_ids(count: int) -> list[str]:
    starts = pd.date_range("2025-01-01", periods=count, freq="7D")
    ends = starts + pd.Timedelta(days=7)
    return [f"E2E_SANITY_ORDERFIX_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}" for s, e in zip(starts, ends)]


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    reports_root = tmp_path / "reports"
    r5_dir = reports_root / "r5"
    r5_1_dir = reports_root / "r5_1"
    r4_dir = reports_root / "r4"
    repair_dir = reports_root / "repair"
    for directory in [r5_dir, r5_1_dir, r4_dir, repair_dir, reports_root / "runs"]:
        directory.mkdir(parents=True)
    run_ids = _run_ids(5)
    for run_id in run_ids:
        (reports_root / "runs" / run_id).mkdir()

    feature_names = [
        "as_of_candidate_tradable_prob_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_skip_candidate_p_flat_v1",
        "as_of_entry_candidate_margin_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        "as_of_skip_replay_clv_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        "as_of_skip_replay_window_realized_vol_5_bps_v1",
        "as_of_skip_replay_body_bps_v1",
        "as_of_skip_replay_wick_ratio_v1",
        "as_of_session_v1",
    ]
    asof_rows: list[dict[str, object]] = []
    hindsight_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    r4_rows: list[dict[str, object]] = []
    for idx in range(90):
        run_id = run_ids[idx % len(run_ids)]
        should = idx % 4 == 0
        hard_negative = idx in {3, 8, 13, 18, 23, 28}
        strong = hard_negative or idx % 7 == 0
        repaired = idx in {80, 81, 82}
        uid = f"cand-{idx:03d}"
        peak = 498.0 if idx == 3 else (120.0 if strong else (30.0 if should else 24.0))
        pnl = -70.0 if should else 28.0
        asof_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "used_for_training": idx < 55,
                "used_for_validation": 55 <= idx < 72,
                "used_for_holdout": idx >= 72,
                "as_of_candidate_tradable_prob_v1": 0.97 if strong else 0.72,
                "as_of_entry_candidate_path_quality_pred_v1": 0.88 if strong else 0.52,
                "as_of_candidate_mfe_first_n_pred_v1": 2.4 if strong else 1.25,
                "as_of_skip_candidate_p_flat_v1": 0.32 if strong else 0.56,
                "as_of_entry_candidate_margin_v1": 0.16 if strong else 0.05,
                "as_of_skip_replay_retracement_from_last_impulse_v1": 0.78 if strong else 0.25,
                "as_of_skip_replay_clv_v1": 0.68 if strong else 0.33,
                "as_of_skip_replay_window_range_15_bps_v1": 55.0 if strong else 105.0,
                "as_of_skip_replay_window_realized_vol_5_bps_v1": 7.0 if strong else 13.0,
                "as_of_skip_replay_body_bps_v1": 20.0 if strong else 60.0,
                "as_of_skip_replay_wick_ratio_v1": 0.4 if strong else 1.2,
                "as_of_session_v1": "US" if idx % 2 else "EU",
                "entry_observation_present_v1": True,
                "entry_raw_state_present_v1": True,
                "entry_coverage_repair_applied_v1": repaired,
                "entry_coverage_repair_source_v1": "fixture_repair" if repaired else "original",
            }
        )
        hindsight_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "baseline_realized_pnl_bps_v1": pnl,
                "peak_mfe_bps_v1": peak,
                "mae_abs_bps_v1": 75.0 if should else 12.0,
                "giveback_bps_v1": 20.0,
                "hindsight_entry_decision_review_v1": "SHOULD_SKIP_TRADE" if should else "TAKE_WAS_OK",
                "hindsight_management_review_v1": "MANAGED_OK",
                "r5_label_should_not_take_v1": should,
                "r5_label_immediate_mae_risk_v1": should,
                "r5_label_runner_protect_v1": (not should) and (strong or repaired),
                "r5_label_strong_trade_candidate_v1": strong,
                "r5_label_tail_control_10_50_risk_v1": should and peak < 50.0,
                "r5_label_take_was_ok_v1": not should,
                "r5_label_bad_trade_but_high_runner_risk_v1": should and peak >= 50.0,
                "r5_label_wait_or_delay_advisory_v1": idx % 6 == 0,
                "r5_hindsight_label_contract_v1": "fixture",
            }
        )
        pred = {
            "candidate_uid": uid,
            "no_entry_fallback_baseline__block_v1": False,
            "r2_fallback_reference__block_v1": should and idx % 3 == 0,
            "r3_fullcoverage_conservative__block_v1": should and idx % 2 == 0,
            "r4_current_reference__block_v1": should or hard_negative,
            "r5_selected_candidate__block_v1": should or hard_negative,
        }
        probs = {
            "should_not_take": 0.82 if should else (0.75 if hard_negative else 0.25),
            "immediate_MAE_risk": 0.84 if should else 0.30,
            "runner_protect": 0.88 if strong and not should else 0.25,
            "strong_trade_candidate": 0.90 if strong else 0.20,
            "tail_control_10_50_risk": 0.82 if should and peak < 50.0 else 0.20,
            "take_was_ok": 0.90 if not should else 0.20,
            "bad_trade_but_high_runner_risk": 0.15,
            "wait_or_delay_advisory": 0.25,
        }
        for label_id, column in R5_PROB.items():
            pred[column] = probs[label_id]
        pred_rows.append(pred)
        r4_rows.append(
            {
                "candidate_uid": uid,
                "best_constrained_recalibrated_r4__block_v1": should or hard_negative,
            }
        )

    pd.DataFrame(asof_rows).to_parquet(r5_dir / "shadow_meta_all_trade_review_r5_entry_as_of_feature_table_v1.parquet", index=False)
    pd.DataFrame(hindsight_rows).to_parquet(r5_dir / "shadow_meta_all_trade_review_r5_entry_hindsight_label_outcome_table_v1.parquet", index=False)
    pd.DataFrame(pred_rows).to_parquet(r5_dir / "shadow_meta_all_trade_review_r5_entry_policy_prediction_view_v1.parquet", index=False)
    _write_json(r5_dir / "shadow_meta_all_trade_review_r5_entry_retrain_contract_v1.json", {"as_of_feature_names_v1": feature_names})
    _write_json(
        r5_dir / "shadow_meta_all_trade_review_r5_entry_summary_v1.json",
        {"coverage_v1": {"ledger_trade_count_v1": 90, "entry_coverage_v1": 90, "entry_raw_coverage_v1": 90, "missing_count_v1": 0, "synthetic_count_v1": 0, "repaired_rows_v1": 3}},
    )

    failure_rows = []
    for idx in {3, 8, 13, 18, 23, 28}:
        failure_rows.append(
            {
                "candidate_uid": f"cand-{idx:03d}",
                "batch04_loso_failure_role_v1": "FALSE_BLOCK",
                "two_hundred_plus_runner_false_block_v1": idx == 3,
                "fifty_plus_runner_false_block_v1": True,
            }
        )
    pd.DataFrame(failure_rows).to_csv(r5_1_dir / "shadow_meta_all_trade_review_r5_1_batch04_failure_attribution_v1.csv", index=False)
    r5_1_pred = pd.DataFrame({"candidate_uid": [row["candidate_uid"] for row in asof_rows], "r5_1_selected_candidate__block_v1": [False] * 90})
    r5_1_pred.to_parquet(r5_1_dir / "shadow_meta_all_trade_review_r5_1_policy_prediction_view_v1.parquet", index=False)
    _write_json(r5_1_dir / "shadow_meta_all_trade_review_r5_1_summary_v1.json", {"decision_v1": {"recommended_next_step_v1": "fixture"}})
    pd.DataFrame({"guard_mode_v1": ["fixture"]}).to_csv(r5_1_dir / "shadow_meta_all_trade_review_r5_1_batch04_like_as_of_guard_audit_v1.csv", index=False)
    pd.DataFrame({"policy_name_v1": ["fixture"]}).to_csv(r5_1_dir / "shadow_meta_all_trade_review_r5_1_head_to_head_vs_r2_r4_r5_v1.csv", index=False)

    pd.DataFrame(r4_rows).to_parquet(r4_dir / "shadow_meta_all_trade_review_r4_fullcoverage_policy_prediction_view_v1.parquet", index=False)
    _write_json(r4_dir / "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_summary_v1.json", {"decision_v1": {"recommended_next_step_v1": "fixture"}})
    _write_json(repair_dir / "shadow_meta_all_trade_review_entry_coverage_repair_summary_v1.json", {"repaired_entry_coverage_v1": 90})
    pd.DataFrame({"candidate_uid": ["cand-080", "cand-081", "cand-082"]}).to_csv(repair_dir / "shadow_meta_all_trade_review_entry_coverage_repair_audit_v1.csv", index=False)
    return reports_root, r5_dir, r5_1_dir, r4_dir, repair_dir


def test_r5_2_entry_runner_aware_materializes(tmp_path: Path) -> None:
    reports_root, r5_dir, r5_1_dir, r4_dir, repair_dir = _build_fixture(tmp_path)
    extension_dir = reports_root / "r5_2"
    result = materialize(
        reports_root,
        r5_dir=r5_dir,
        r5_1_dir=r5_1_dir,
        r4_dir=r4_dir,
        repair_dir=repair_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        expected_ledger_count=90,
    )
    assert result["status"]["not_live_gate"] is True
    for artifact in [
        HARD_NEGATIVE_AUDIT,
        TWO_HEAD_STACK_BAKEOFF,
        PARETO_FRONTIER,
        LOSO_METRICS,
        HEAD_TO_HEAD,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["coverage_v1"]["entry_coverage_v1"] == 90
    assert summary["hard_negative_v1"]["batch04_false_block_hard_negative_count_v1"] == 6
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()


def test_r5_2_batch05_absent_is_not_reported_as_fail(tmp_path: Path) -> None:
    reports_root, r5_dir, r5_1_dir, r4_dir, repair_dir = _build_fixture(tmp_path)
    extension_dir = reports_root / "r5_2_compact"
    materialize(
        reports_root,
        r5_dir=r5_dir,
        r5_1_dir=r5_1_dir,
        r4_dir=r4_dir,
        repair_dir=repair_dir,
        extension_dir=extension_dir,
        batch_weeks=2,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        expected_ledger_count=90,
    )
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["decision_v1"]["batch05_loso_pass_v1"] is None
