from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r4_seed_policy_from_r3_v1 import (
    R2_AS_OF_TABLE,
    R2_LABEL_TABLE,
    R3_PREDICTION_VIEW,
    R4_POLICY_PREDICTION_VIEW,
    R4_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_materialize_monday_r4_seed_policy_from_r3(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    readiness_dir = reports_root / "readiness"
    r3_dir = reports_root / "r3"
    readiness_dir.mkdir(parents=True)
    r3_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-a",
                "trade_uid": "trade-a",
                "trade_id": "1",
                "decision_timestamp": "2025-01-07T13:22:00+00:00",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "entry_observation_present_v1": True,
                "entry_raw_state_present_v1": True,
            },
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-b",
                "trade_uid": "trade-b",
                "trade_id": "2",
                "decision_timestamp": "2025-01-07T13:23:00+00:00",
                "used_for_training": False,
                "used_for_validation": True,
                "used_for_holdout": False,
                "entry_observation_present_v1": True,
                "entry_raw_state_present_v1": True,
            },
        ]
    ).to_parquet(readiness_dir / R2_AS_OF_TABLE, index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "cand-a",
                "hindsight_entry_decision_review_v1": "SHOULD_NOT_TAKE",
                "baseline_realized_pnl_bps_v1": -10.0,
                "peak_mfe_bps_v1": 8.0,
                "mae_abs_bps_v1": 35.0,
                "giveback_bps_v1": 2.0,
                "harvest_capture_ratio_v1": -1.25,
                "label_should_not_take_v1": True,
                "label_immediate_mae_risk_v1": True,
                "label_wait_would_have_helped_v1": False,
                "label_strong_trade_candidate_v1": False,
                "label_direct_take_ok_v1": False,
            },
            {
                "candidate_uid": "cand-b",
                "hindsight_entry_decision_review_v1": "TAKE_WAS_OK",
                "baseline_realized_pnl_bps_v1": 20.0,
                "peak_mfe_bps_v1": 70.0,
                "mae_abs_bps_v1": 10.0,
                "giveback_bps_v1": 10.0,
                "harvest_capture_ratio_v1": 0.3,
                "label_should_not_take_v1": False,
                "label_immediate_mae_risk_v1": False,
                "label_wait_would_have_helped_v1": False,
                "label_strong_trade_candidate_v1": True,
                "label_direct_take_ok_v1": True,
            },
        ]
    ).to_parquet(readiness_dir / R2_LABEL_TABLE, index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "cand-a",
                "entry_r3_feature_available_v1": True,
                "entry_r3_shadow_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE_SHADOW",
                "entry_r3_shadow_action_source_v1": "R3_POLICY",
                "pred__entry_r3_should_not_take__prob_true_v1": 0.9,
                "pred__entry_r3_immediate_mae_risk__prob_true_v1": 0.85,
                "pred__entry_r3_wait_would_have_helped__prob_true_v1": 0.1,
                "pred__entry_r3_strong_trade_candidate__prob_true_v1": 0.1,
                "pred__entry_r3_direct_take_ok__prob_true_v1": 0.2,
                "pred__entry_r3_good_mfe_bad_capture__prob_true_v1": 0.05,
            },
            {
                "candidate_uid": "cand-b",
                "entry_r3_feature_available_v1": True,
                "entry_r3_shadow_action_v1": "KEEP_BASELINE_SHADOW",
                "entry_r3_shadow_action_source_v1": "R3_POLICY",
                "pred__entry_r3_should_not_take__prob_true_v1": 0.2,
                "pred__entry_r3_immediate_mae_risk__prob_true_v1": 0.1,
                "pred__entry_r3_wait_would_have_helped__prob_true_v1": 0.2,
                "pred__entry_r3_strong_trade_candidate__prob_true_v1": 0.8,
                "pred__entry_r3_direct_take_ok__prob_true_v1": 0.9,
                "pred__entry_r3_good_mfe_bad_capture__prob_true_v1": 0.2,
            },
        ]
    ).to_parquet(r3_dir / R3_PREDICTION_VIEW, index=False)
    _write_json(r3_dir / "shadow_meta_all_trade_review_r3_entry_label_feature_summary_v1.json", {"status_v1": {"ok": True}})

    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        r3_dir=r3_dir,
        extension_dir=reports_root / "r4seed",
        expected_ledger_count=2,
    )
    extension_dir = result["extension_dir"]
    assert (extension_dir / R4_POLICY_PREDICTION_VIEW).exists()
    assert (extension_dir / R4_SUMMARY).exists()
    pred = pd.read_parquet(extension_dir / R4_POLICY_PREDICTION_VIEW).set_index("candidate_uid")
    assert bool(pred.loc["cand-a", "r2_entry_fallback_row_v1"])
    assert bool(pred.loc["cand-a", "r4_entry_fallback_block_v1"])
    assert not bool(pred.loc["cand-b", "r4_entry_fallback_block_v1"])

