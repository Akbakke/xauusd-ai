from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_truth_management_coarse_teacher_v1 import (
    build_management_coarse_teacher_payload,
    write_management_coarse_teacher_artifacts,
)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_build_management_coarse_teacher_payload_materializes_feedback_and_grid(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir()
    extension_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260420T133345Z_MANAGEMENT_AUDIT_EXTENSION_V1"
    extension_dir.mkdir()
    (extension_dir / "shadow_meta_all_trade_review_management_audit_extension_build_summary_v1.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    rows = []
    outcome_rows = []
    quality_as_of_rows = []
    quality_hindsight_rows = []

    for idx in range(40):
        is_hold = idx < 20
        action = "HOLD" if is_hold else "EXIT_NOW"
        cand = f"cand_{idx:03d}"
        trade_uid = f"trade_uid_{idx:03d}"
        trade_id = f"trade_id_{idx:03d}"
        strong = is_hold
        weak = not is_hold
        good_exit = False if is_hold else idx < 30
        premature_exit = False if is_hold else idx >= 30
        rows.append(
            {
                "management_row_key_v1": f"row_{idx:03d}",
                "run_id": "run_a",
                "candidate_uid_exact_v1": cand,
                "trade_uid_exact_v1": trade_uid,
                "trade_id_exact_v1": trade_id,
                "as_of_row_uid_v1": f"asof_{idx:03d}",
                "decision_ts_utc_v1": f"2026-01-01T00:{idx:02d}:00+00:00",
                "decision_anchor_type_v1": "ACTUAL_EXIT_DECISION_ANCHOR",
                "split_bucket_v1": "TRAIN" if idx < 24 else ("VALIDATION" if idx < 32 else "HOLDOUT"),
                "observed_action_v1": action,
                "shadow_score_v1": float(idx),
                "shadow_bucket_status_v1": "TRAIN_SCORE_Q5_TOP_20_PCT_APPROX" if idx >= 32 else "TRAIN_SCORE_Q1_LOWEST_20_PCT_APPROX",
                "shadow_bucket_rank_v1": 5 if idx >= 32 else 1,
                "overlay_session_axis_v1": "US",
                "overlay_vol_axis_v1": "EXTREME",
                "overlay_hold_age_axis_v1": "LATE_120M_PLUS",
                "overlay_giveback_axis_v1": "LOW_LT_0P50",
                "as_of_management_core_minutes_held_at_anchor_v1": 140.0,
                "as_of_management_core_giveback_ratio_from_peak_v1": 0.25,
                "as_of_atr_bps_v1": 22.0,
                "as_of_hour_utc_v1": 14.0,
                "as_of_session_v1": "US",
                "as_of_side_v1": "LONG",
                "as_of_trend_regime_v1": "TREND",
                "as_of_vol_regime_v1": "EXTREME",
            }
        )
        outcome_rows.append(
            {
                "management_row_key_v1": f"row_{idx:03d}",
                "realized_pnl_bps": 40.0 if strong else -20.0,
                "mfe_bps": 60.0 if strong else 15.0,
                "mae_bps": -5.0 if strong else -35.0,
                "trade_outcome_class": "positive_exit" if strong else "cata",
                "exit_reason": "tp" if strong else "catastrophic",
                "good_exit": good_exit,
                "premature_exit": premature_exit,
                "late_exit": False,
                "hindsight_peak_mfe_bps_v1": 75.0 if is_hold else 35.0,
                "hindsight_peak_to_exit_giveback_bps_v1": 8.0 if is_hold else 28.0,
            }
        )
        quality_as_of_rows.append(
            {
                "candidate_uid_exact_v1": cand,
                "trade_uid_exact_v1": trade_uid,
                "trade_id_exact_v1": trade_id,
                "walkforward_slice_v1": "slice_a" if idx < 20 else "slice_b",
                "walkforward_slice_start_utc_v1": "2026-01-01T00:00:00+00:00",
                "walkforward_slice_end_utc_v1": "2026-02-01T00:00:00+00:00",
            }
        )
        quality_hindsight_rows.append(
            {
                "candidate_uid_exact_v1": cand,
                "trade_uid_exact_v1": trade_uid,
                "trade_id_exact_v1": trade_id,
                "realized_pnl_bps_v1": 40.0 if strong else -20.0,
                "mfe_bps_v1": 60.0 if strong else 15.0,
                "mae_bps_v1": -5.0 if strong else -35.0,
                "giveback_ratio_v1": 0.25 if strong else 0.92,
                "holding_time_bars_v1": 120 if strong else 40,
                "quality_band_strong_capture_v1": strong,
                "quality_band_weak_capture_v1": weak,
                "quality_band_high_giveback_v1": weak,
                "quality_band_low_mae_v1": strong,
                "quality_band_tail_risk_v1": weak,
            }
        )

    _write_parquet(
        extension_dir / "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet",
        pd.DataFrame.from_records(rows),
    )
    _write_parquet(
        extension_dir / "shadow_meta_all_trade_review_management_policy_logging_outcome_backfill_harness_v1.parquet",
        pd.DataFrame.from_records(outcome_rows),
    )
    _write_parquet(
        extension_dir / "shadow_meta_all_trade_review_management_outcome_quality_regime_audit_as_of_v1.parquet",
        pd.DataFrame.from_records(quality_as_of_rows),
    )
    _write_parquet(
        extension_dir / "shadow_meta_all_trade_review_management_outcome_quality_regime_audit_hindsight_v1.parquet",
        pd.DataFrame.from_records(quality_hindsight_rows),
    )

    payload = build_management_coarse_teacher_payload(
        reports_root=reports_root,
        extension_dir=extension_dir,
        min_cell_rows=10,
        min_rows_per_action=5,
    )

    summary = payload["summary"]
    assert summary["row_count_v1"] == 40
    assert summary["binary_teacher_target_summary_v1"]["eligible_rows_v1"] == 40
    assert summary["recommended_grid_v1"]["grid_name_v1"] == "SESSION_HOLD_GIVEBACK"
    assert summary["recommended_next_step_v1"] == "KEEP_AS_RESEARCH_SURFACE_UNTIL_MORE_SIGNAL"

    view_df = payload["view_df"]
    assert "recommended_coarse_grid_value_v1" in view_df.columns
    assert view_df["coarse_teacher_feedback_label_v1"].astype("string").isin(
        [
            "OBSERVED_HOLD_DEFENSIBLE",
            "OBSERVED_EXIT_DEFENSIBLE",
            "OBSERVED_EXIT_TOO_EARLY",
        ]
    ).all()

    written = write_management_coarse_teacher_artifacts(
        reports_root=reports_root,
        extension_dir=extension_dir,
        min_cell_rows=10,
        min_rows_per_action=5,
    )
    assert Path(written["view_path"]).exists()
    assert Path(written["cell_rollup_path"]).exists()
    assert Path(written["summary_path"]).exists()

    summary_on_disk = json.loads(Path(written["summary_path"]).read_text(encoding="utf-8"))
    assert summary_on_disk["row_count_v1"] == 40
