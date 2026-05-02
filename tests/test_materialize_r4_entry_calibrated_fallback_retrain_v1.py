from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r4_entry_calibrated_fallback_retrain_v1 import (
    R4_CONSISTENCY_AUDIT,
    R4_POLICY_PREDICTION_VIEW,
    R4_POLICY_STACK_CANDIDATES,
    R4_R2_FALLBACK_PRESERVATION_AUDIT,
    R4_READINESS_MATRIX,
    R4_SUMMARY,
    R4_WALKFORWARD_SAFETY_REPLAY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    reports_root = tmp_path / "reports"
    readiness_dir = reports_root / "readiness"
    r2_dir = reports_root / "r2"
    r3_dir = reports_root / "r3"
    extension_dir = reports_root / "r4"
    runs_root = reports_root / "runs"
    for directory in [readiness_dir, r2_dir, r3_dir, runs_root]:
        directory.mkdir(parents=True)
    run_ids = [f"E2E_SANITY_ORDERFIX_202501{idx + 1:02d}_202501{idx + 8:02d}" for idx in range(4)]
    for run_id in run_ids:
        (runs_root / run_id).mkdir()

    feature_names = ["as_of_signal_bad_v1", "as_of_signal_strong_v1", "as_of_session_v1"]
    asof_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    r2_rows: list[dict[str, object]] = []
    r3_rows: list[dict[str, object]] = []
    for idx in range(24):
        uid = f"cand-{idx:03d}"
        run_id = run_ids[idx % len(run_ids)]
        should = idx % 4 == 0
        strong = idx % 7 == 0
        r2_fb = idx in {0, 4, 8, 12, 14, 16}
        available = idx not in {21, 22}
        peak = 120.0 if strong else (25.0 if should else 40.0)
        baseline = -12.0 if should else 18.0
        p_should = 0.72 if should else 0.35
        p_strong = 0.85 if strong else 0.20
        p_direct = 0.25 if should else 0.75
        asof_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 24) + 1:02d}T12:00:00+00:00",
                "used_for_training": idx < 14,
                "used_for_validation": 14 <= idx < 19,
                "used_for_holdout": idx >= 19,
                "entry_observation_present_v1": available,
                "entry_raw_state_present_v1": available,
                "management_observation_present_v1": True,
                "as_of_signal_bad_v1": p_should,
                "as_of_signal_strong_v1": p_strong,
                "as_of_session_v1": "NY" if idx % 2 else "LONDON",
            }
        )
        label_rows.append(
            {
                "candidate_uid": uid,
                "label_low_mfe_low_value_v1": should,
                "hindsight_entry_decision_review_v1": "SHOULD_SKIP_TRADE" if should else "TAKE_WAS_OK",
                "hindsight_management_review_v1": "SHOULD_EXIT_EARLIER",
                "session": "US",
                "vol_regime": "HIGH",
                "trend_regime": "RANGE",
            }
        )
        coverage_rows.append(
            {
                "candidate_uid": uid,
                "entry_observation_present_v1": available,
                "entry_gap_reason_code_v1": "COVERED" if available else "missing entry observation",
                "entry_gap_reason_detail_v1": "covered" if available else "test missing",
                "management_gap_reason_code_v1": "COVERED",
                "management_gap_reason_detail_v1": "covered",
                "coverage_gap_scope_v1": "FULLY_COVERED" if available else "MISSING_ENTRY_ONLY",
            }
        )
        r2_rows.append(
            {
                "candidate_uid": uid,
                "candidate_shadow_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if r2_fb else "KEEP_BASELINE",
                "candidate_shadow_action_source_v1": "ENTRY_MODEL_SUPPRESS_FALLBACK" if r2_fb else "MANAGEMENT_RL_ACTION_MODEL",
                "candidate_shadow_action_matches_harvest_target_v1": bool(r2_fb and should),
                "candidate_shadow_delta_bps_v1": 10.0 if r2_fb else 0.0,
                "candidate_shadow_pnl_bps_v1": baseline + (10.0 if r2_fb else 0.0),
                "pred__entry_xgb_binary_take__prob_false_v1": p_should,
                "pred__entry_xgb_binary_take__prob_true_v1": 1.0 - p_should,
                "pred__entry_xgb_harvest_label__prob_reject_or_low_size_v1": p_should,
            }
        )
        r3_block = bool(p_should >= 0.6 and p_strong < 0.75)
        r3_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 24) + 1:02d}T12:00:00+00:00",
                "used_for_training": idx < 14,
                "used_for_validation": 14 <= idx < 19,
                "used_for_holdout": idx >= 19,
                "entry_r3_feature_available_v1": available,
                "baseline_realized_pnl_bps_v1": baseline,
                "peak_mfe_bps_v1": peak,
                "mae_abs_bps_v1": 8.0 if strong else 35.0,
                "giveback_bps_v1": 5.0 if strong else 20.0,
                "harvest_capture_ratio_v1": 0.8 if strong else 0.2,
                "exit_harvest_policy_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if should else "KEEP_BASELINE",
                "trade_outcome_class": "never_mfe" if should else "positive_exit",
                "exit_reason": "THRESHOLD",
                "label_should_not_take_v1": should,
                "label_strong_trade_candidate_v1": strong,
                "label_immediate_mae_risk_v1": should,
                "label_good_mfe_bad_capture_v1": not should and not strong,
                "label_direct_take_ok_v1": not should,
                "label_wait_would_have_helped_v1": idx % 5 == 0,
                "pred__entry_r3_should_not_take__prob_true_v1": p_should,
                "pred__entry_r3_immediate_mae_risk__prob_true_v1": p_should,
                "pred__entry_r3_wait_would_have_helped__prob_true_v1": 0.9 if idx % 5 == 0 else 0.2,
                "pred__entry_r3_strong_trade_candidate__prob_true_v1": p_strong,
                "pred__entry_r3_direct_take_ok__prob_true_v1": p_direct,
                "pred__entry_r3_good_mfe_bad_capture__prob_true_v1": 0.7 if not should and not strong else 0.2,
                "entry_r3_shadow_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW" if r3_block else "ENTRY_ALLOW_BASELINE_SHADOW",
                "entry_r3_shadow_action_source_v1": "TEST",
            }
        )

    pd.DataFrame(asof_rows).to_parquet(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet", index=False)
    pd.DataFrame(label_rows).to_parquet(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet", index=False)
    pd.DataFrame(coverage_rows).to_csv(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_coverage_gap_audit_v1.csv", index=False)
    pd.DataFrame(r2_rows).to_parquet(r2_dir / "shadow_meta_all_trade_review_harvest_retrain_candidate_prediction_view_v1.parquet", index=False)
    pd.DataFrame(r3_rows).to_parquet(r3_dir / "shadow_meta_all_trade_review_r3_entry_label_feature_prediction_view_v1.parquet", index=False)
    _write_json(readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json", {"as_of_feature_names_v1": feature_names})
    _write_json(r3_dir / "shadow_meta_all_trade_review_r3_entry_label_feature_summary_v1.json", {"r3_holdout_min_balanced_accuracy_v1": 0.6})
    return reports_root, readiness_dir, r2_dir, r3_dir, extension_dir


def test_materialize_r4_entry_calibrated_fallback(tmp_path: Path) -> None:
    reports_root, readiness_dir, r2_dir, r3_dir, extension_dir = _build_fixture(tmp_path)

    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        r2_dir=r2_dir,
        r3_dir=r3_dir,
        extension_dir=extension_dir,
        batch_weeks=2,
        expected_ledger_count=24,
    )

    assert result["status"]["R4_ENTRY_CALIBRATED_FALLBACK_STATUS"] == "R4_SHADOW_REPLAY_CANDIDATE_NOT_LIVE_GATE"
    for artifact in [
        R4_POLICY_PREDICTION_VIEW,
        R4_R2_FALLBACK_PRESERVATION_AUDIT,
        R4_POLICY_STACK_CANDIDATES,
        R4_WALKFORWARD_SAFETY_REPLAY,
        R4_READINESS_MATRIX,
        R4_CONSISTENCY_AUDIT,
        R4_SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()

    prediction = pd.read_parquet(extension_dir / R4_POLICY_PREDICTION_VIEW)
    policy = pd.read_csv(extension_dir / R4_POLICY_STACK_CANDIDATES)
    selected = policy[
        policy["policy_name_v1"].eq("R4_R2_PRESERVED_PLUS_SHOULD_DIRECT_STRONG_PROTECTED")
        & policy["scope_v1"].eq("ALL")
    ].iloc[0]

    assert len(prediction) == 24
    assert selected["r2_fallback_should_not_take_preserved_v1"] >= 1
    assert selected["strong_trade_false_block_count_v1"] == 0

