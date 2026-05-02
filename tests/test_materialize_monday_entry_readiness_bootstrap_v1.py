from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_entry_readiness_bootstrap_v1 import (
    AS_OF_TABLE,
    CONSISTENCY_AUDIT,
    COVERAGE_AUDIT,
    HINDSIGHT_TABLE,
    LEDGER_CLOSED_TRADES,
    HINDSIGHT_EXPORT,
    ENTRY_OBSERVABILITY,
    ENTRY_RAW_STATE,
    MANAGEMENT_RAW_STATE,
    AS_OF_LEDGER,
    RUN_ROLLUP,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_materialize_monday_entry_readiness_bootstrap(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    review_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_TEST"
    review_dir.mkdir(parents=True)

    ledger = pd.DataFrame(
        [
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-covered",
                "trade_uid": "trade-covered",
                "trade_id": "1",
                "decision_timestamp": "2025-01-07T13:22:00+00:00",
                "realized_pnl_bps": 12.5,
                "mfe_bps": 45.0,
                "mae_bps": -8.0,
                "trade_outcome_class": "positive_exit",
                "exit_reason": "THRESHOLD",
                "session": "OVERLAP",
                "vol_regime": "MEDIUM",
                "trend_regime": "TREND_UP",
                "good_trade": True,
                "good_trade_mfe20_mae5": False,
            },
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-missing",
                "trade_uid": "trade-missing",
                "trade_id": "2",
                "decision_timestamp": "2025-01-07T13:23:00+00:00",
                "realized_pnl_bps": -15.0,
                "mfe_bps": 6.0,
                "mae_bps": -42.0,
                "trade_outcome_class": "cata",
                "exit_reason": "CATASTROPHIC_GUARD",
                "session": "OVERLAP",
                "vol_regime": "HIGH",
                "trend_regime": "TREND_NEUTRAL",
                "good_trade": False,
                "good_trade_mfe20_mae5": False,
            },
        ]
    )
    ledger.to_parquet(review_dir / LEDGER_CLOSED_TRADES, index=False)
    pd.DataFrame(
        [
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-covered",
                "trade_uid": "trade-covered",
                "trade_id": "1",
                "decision_timestamp": "2025-01-07T13:22:00+00:00",
                "post_trade_good_trade_flag_v1": True,
                "post_trade_good_trade_mfe20_mae5_v1": False,
                "hindsight_entry_decision_review_v1": "TAKE_WAS_OK",
                "hindsight_management_review_v1": "MANAGED_OK",
                "hindsight_should_skip_trade_v1": False,
                "hindsight_take_was_ok_v1": True,
                "hindsight_should_hold_longer_v1": False,
                "hindsight_should_exit_earlier_v1": False,
                "hindsight_peak_mfe_bps_v1": 45.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 8.0,
                "hindsight_peak_to_worst_after_peak_bps_v1": 4.0,
            },
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-missing",
                "trade_uid": "trade-missing",
                "trade_id": "2",
                "decision_timestamp": "2025-01-07T13:23:00+00:00",
                "post_trade_good_trade_flag_v1": False,
                "post_trade_good_trade_mfe20_mae5_v1": False,
                "hindsight_entry_decision_review_v1": "SHOULD_NOT_TAKE",
                "hindsight_management_review_v1": "N/A",
                "hindsight_should_skip_trade_v1": True,
                "hindsight_take_was_ok_v1": False,
                "hindsight_should_hold_longer_v1": False,
                "hindsight_should_exit_earlier_v1": False,
                "hindsight_peak_mfe_bps_v1": 6.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 3.0,
                "hindsight_peak_to_worst_after_peak_bps_v1": 10.0,
            },
        ]
    ).to_parquet(review_dir / HINDSIGHT_EXPORT, index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "cand-covered",
                "as_of_hour_utc_v1": 13,
                "as_of_weekday_utc_v1": 1,
                "as_of_session_v1": "OVERLAP",
                "as_of_side_v1": "LONG",
                "as_of_atr_bps_v1": 20.0,
                "as_of_candidate_entry_spread_bps_v1": 1.2,
                "as_of_candidate_uncertainty_score_v1": 0.2,
                "as_of_candidate_tradable_prob_v1": 0.9,
                "as_of_candidate_mfe_first_n_pred_v1": 2.1,
                "as_of_candidate_trend_regime_v1": "TREND_UP",
                "as_of_candidate_vol_regime_v1": "MEDIUM",
                "as_of_entry_candidate_margin_v1": 0.3,
                "as_of_entry_candidate_path_quality_pred_v1": 0.7,
                "as_of_skip_xgb_p_flat_v1": 0.2,
                "as_of_skip_xgb_p_hat_v1": 0.7,
                "as_of_skip_xgb_p_long_v1": 0.7,
                "as_of_skip_xgb_p_short_v1": 0.1,
                "as_of_skip_xgb_pred_side_v1": "LONG",
                "as_of_skip_xgb_has_ctx_v1": 1,
                "teacher_should_wait_entry_v1": False,
                "support_adverse_first_v1": False,
                "confirmation_delay_minutes_v1": 0,
                "has_provable_confirmation_v1": True,
            }
        ]
    ).to_parquet(review_dir / ENTRY_OBSERVABILITY, index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "cand-covered",
                "entry_raw_replay_bar_exact_available_v1": True,
                "entry_raw_candidate_snapshot_exact_available_v1": True,
                "entry_raw_xgb_multi_horizon_exact_available_v1": True,
                "as_of_entry_candidate_p_flat_v1": 0.2,
                "as_of_entry_candidate_p_hat_v1": 0.7,
                "as_of_entry_candidate_p_long_v1": 0.7,
                "as_of_entry_candidate_p_short_v1": 0.1,
                "as_of_entry_candidate_entry_spread_bps_v1": 1.2,
                "as_of_entry_candidate_margin_v1": 0.3,
                "as_of_entry_candidate_path_quality_pred_v1": 0.7,
                "as_of_entry_replay_range_bps_v1": 14.0,
                "as_of_entry_replay_window_ret_1_bps_v1": 2.0,
                "as_of_entry_replay_window_realized_vol_3_bps_v1": 5.0,
            }
        ]
    ).to_parquet(review_dir / ENTRY_RAW_STATE, index=False)
    pd.DataFrame([{"candidate_uid": "cand-covered"}]).to_parquet(review_dir / MANAGEMENT_RAW_STATE, index=False)
    pd.DataFrame(
        [
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-covered",
                "used_for_training": True,
                "used_for_validation": False,
                "used_for_holdout": False,
                "as_of_split_bucket_v1": "TRAIN",
            },
            {
                "run_id": "TRUTH_MONFRI_WEEK_20250106_20250113",
                "candidate_uid": "cand-missing",
                "used_for_training": False,
                "used_for_validation": True,
                "used_for_holdout": False,
                "as_of_split_bucket_v1": "VALIDATION",
            },
        ]
    ).to_parquet(review_dir / AS_OF_LEDGER, index=False)

    result = materialize(reports_root, review_dir=review_dir, extension_dir=reports_root / "bootstrap")
    extension_dir = result["extension_dir"]

    for artifact in [AS_OF_TABLE, HINDSIGHT_TABLE, COVERAGE_AUDIT, RUN_ROLLUP, SUMMARY, CONSISTENCY_AUDIT]:
        assert (extension_dir / artifact).exists()

    asof = pd.read_parquet(extension_dir / AS_OF_TABLE).set_index("candidate_uid")
    labels = pd.read_parquet(extension_dir / HINDSIGHT_TABLE).set_index("candidate_uid")
    coverage = pd.read_csv(extension_dir / COVERAGE_AUDIT).set_index("candidate_uid")

    assert bool(asof.loc["cand-covered", "entry_observation_present_v1"])
    assert not bool(asof.loc["cand-missing", "entry_observation_present_v1"])
    assert float(asof.loc["cand-covered", "as_of_skip_candidate_p_long_v1"]) == 0.7
    assert float(asof.loc["cand-covered", "as_of_skip_replay_range_bps_v1"]) == 14.0
    assert bool(labels.loc["cand-missing", "label_should_not_take_v1"])
    assert coverage.loc["cand-missing", "entry_gap_reason_code_v1"] == "missing entry observation and raw-state"

