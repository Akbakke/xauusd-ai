from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.train_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1 import (
    CONSISTENCY_AUDIT,
    FEATURE_AUDIT,
    HEAD_TO_HEAD,
    LABEL_AUDIT,
    LOSO,
    MODEL_BAKEOFF,
    MODEL_METRICS,
    POLICY_PREDICTION_VIEW,
    SUMMARY,
    THRESHOLD_CALIBRATION,
    WALKFORWARD,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _run_ids(count: int) -> list[str]:
    starts = pd.date_range("2025-01-01", periods=count, freq="7D")
    ends = starts + pd.Timedelta(days=7)
    return [f"E2E_SANITY_ORDERFIX_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}" for s, e in zip(starts, ends)]


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    reports_root = tmp_path / "reports"
    readiness_dir = reports_root / "readiness"
    r3_dir = reports_root / "r3"
    r4_dir = reports_root / "r4_fullcoverage"
    micro_dir = reports_root / "micro"
    for directory in [readiness_dir, r3_dir, r4_dir, micro_dir, reports_root / "runs"]:
        directory.mkdir(parents=True)
    run_ids = _run_ids(5)
    for run_id in run_ids:
        (reports_root / "runs" / run_id).mkdir()

    asof_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    r3_rows: list[dict[str, object]] = []
    r4_rows: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []
    for idx in range(60):
        should = idx % 4 == 0
        repaired = idx in {51, 52, 53, 54, 55}
        strong = idx % 5 == 0 or repaired
        peak = 220.0 if idx % 17 == 0 else (90.0 if strong or repaired else (25.0 if should else 38.0))
        pnl = -80.0 if should else 35.0
        run_id = run_ids[idx % len(run_ids)]
        uid = f"cand-{idx:03d}"
        asof_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "used_for_training": idx < 40,
                "used_for_validation": 40 <= idx < 50,
                "used_for_holdout": idx >= 50,
                "as_of_signal_a_v1": float(idx % 11),
                "as_of_signal_b_v1": float((idx * 3) % 13),
                "as_of_session_v1": "US" if idx % 2 else "EU",
                "entry_observation_present_v1": True,
                "entry_raw_state_present_v1": True,
                "entry_coverage_repair_applied_v1": repaired,
                "entry_coverage_repair_source_v1": "EXACT_REPAIR" if repaired else "ORIGINAL",
            }
        )
        label_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "hindsight_entry_decision_review_v1": "SHOULD_SKIP_TRADE" if should else "TAKE_WAS_OK",
                "hindsight_management_review_v1": "MANAGED_OK",
                "baseline_realized_pnl_bps_v1": pnl,
                "peak_mfe_bps_v1": peak,
                "mae_abs_bps_v1": 90.0 if should else 12.0,
                "giveback_bps_v1": 30.0,
                "harvest_capture_ratio_v1": 0.7 if strong else 0.2,
                "exit_harvest_policy_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if should else "KEEP_BASELINE",
                "trade_outcome_class": "bad" if should else "ok",
                "exit_reason": "TEST",
                "session": "US",
                "vol_regime": "HIGH",
                "trend_regime": "TREND_NEUTRAL",
                "support_adverse_first_v1": idx % 6 == 0,
                "confirmation_delay_minutes_v1": float((idx % 10) + 1),
                "has_provable_confirmation_v1": idx % 3 == 0,
                "teacher_should_wait_entry_v1": idx % 6 == 0,
                "label_should_not_take_v1": should,
                "label_immediate_mae_risk_v1": idx % 3 == 0,
                "label_wait_would_have_helped_v1": idx % 6 == 0,
                "label_good_mfe_bad_capture_v1": idx % 8 == 0,
                "label_low_mfe_low_value_v1": should,
                "label_strong_trade_candidate_v1": strong,
                "label_direct_take_ok_v1": not should,
            }
        )
        r3_rows.append(
            {
                "candidate_uid": uid,
                "entry_r3_feature_available_v1": True,
                "entry_r3_shadow_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW" if should and idx % 2 == 0 else "ENTRY_ALLOW_BASELINE_SHADOW",
                "entry_r3_shadow_action_source_v1": "fixture",
            }
        )
        r4_block = should and idx % 3 != 0
        r4_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "is_repaired_165_v1": repaired,
                "label_should_not_take_v1": should,
                "label_strong_trade_candidate_v1": strong,
                "take_was_ok_v1": not should,
                "peak_mfe_bps_v1": peak,
                "mae_abs_bps_v1": 90.0 if should else 12.0,
                "giveback_bps_v1": 30.0,
                "baseline_realized_pnl_bps_v1": pnl,
                "no_entry_fallback_baseline__block_v1": False,
                "r2_fallback_reference__block_v1": should and idx % 5 == 0,
                "r3_fullcoverage_conservative__block_v1": should and idx % 4 == 0,
                "r4_repaired_selected_reference__block_v1": r4_block,
                "best_constrained_recalibrated_r4__block_v1": r4_block,
            }
        )
        if repaired:
            repair_rows.append({"candidate_uid": uid, "synthetic_value_used_v1": False})

    pd.DataFrame(asof_rows).to_parquet(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet", index=False)
    pd.DataFrame(label_rows).to_parquet(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet", index=False)
    pd.DataFrame(repair_rows).to_csv(readiness_dir / "shadow_meta_all_trade_review_entry_coverage_repair_audit_v1.csv", index=False)
    pd.DataFrame(r3_rows).to_parquet(r3_dir / "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet", index=False)
    pd.DataFrame(r4_rows).to_parquet(r4_dir / "shadow_meta_all_trade_review_r4_fullcoverage_policy_prediction_view_v1.parquet", index=False)
    _write_json(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json", {"as_of_feature_names_v1": ["as_of_signal_a_v1", "as_of_signal_b_v1", "as_of_session_v1"]})
    _write_json(r4_dir / "shadow_meta_all_trade_review_r4_fullcoverage_policy_recalibration_summary_v1.json", {"decision_v1": {"recommended_next_step_v1": "fixture"}})
    _write_json(micro_dir / "shadow_meta_all_trade_review_r4_five_microtest_summary_v1.json", {"decision_v1": {"recommended_next_step_v1": "fixture"}})
    return reports_root, readiness_dir, r3_dir, r4_dir, micro_dir


def test_train_r5_entry_retrain_with_repaired_coverage(tmp_path: Path) -> None:
    reports_root, readiness_dir, r3_dir, r4_dir, micro_dir = _build_fixture(tmp_path)
    extension_dir = reports_root / "r5"
    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        r3_dir=r3_dir,
        r4_fullcoverage_dir=r4_dir,
        r4_microtest_dir=micro_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        expected_ledger_count=60,
    )
    assert result["status"]["R5_ENTRY_RETRAIN_STATUS"] == "TRAINED_SHADOW_RESEARCH_READY_NOT_LIVE_GATE"
    for artifact in [
        LABEL_AUDIT,
        FEATURE_AUDIT,
        MODEL_METRICS,
        MODEL_BAKEOFF,
        THRESHOLD_CALIBRATION,
        WALKFORWARD,
        LOSO,
        HEAD_TO_HEAD,
        POLICY_PREDICTION_VIEW,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["coverage_v1"]["entry_coverage_v1"] == 60
    assert summary["coverage_v1"]["synthetic_count_v1"] == 0
    assert summary["status_v1"]["not_live_gate"] is True
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
