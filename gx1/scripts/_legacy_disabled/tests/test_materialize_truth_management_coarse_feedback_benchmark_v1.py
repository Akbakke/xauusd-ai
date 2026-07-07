from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_truth_management_coarse_feedback_benchmark_v1 import (
    build_management_coarse_feedback_benchmark_payload,
    write_management_coarse_feedback_benchmark_artifacts,
)


def test_build_management_coarse_feedback_benchmark_payload_trains_hold_only_surface(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir()
    teacher_view_path = reports_root / "truth_management_coarse_teacher_v1.parquet"

    rows = []
    split_map = {**{idx: "TRAIN" for idx in range(72)}, **{idx: "VALIDATION" for idx in range(72, 96)}, **{idx: "HOLDOUT" for idx in range(96, 120)}}
    for idx in range(120):
        target = 1 if idx % 4 in {0, 1} else 0
        score = 2.5 if target == 1 else -2.0
        session = "US" if target == 1 else "OVERLAP"
        hold_age = "LATE_120M_PLUS" if target == 1 else "EARLY_0_30M"
        giveback = "LOW_LT_0P50" if target == 1 else "HIGH_GE_0P85"
        rows.append(
            {
                "management_row_key_v1": f"hold_{idx:03d}",
                "split_bucket_v1": split_map[idx],
                "observed_action_v1": "HOLD",
                "coarse_teacher_binary_target_v1": target,
                "coarse_teacher_binary_target_eligible_v1": True,
                "coarse_teacher_feedback_label_v1": "OBSERVED_HOLD_DEFENSIBLE" if target == 1 else "OBSERVED_HOLD_TOO_WEAK",
                "realized_pnl_bps": 45.0 if target == 1 else -18.0,
                "hold_longer_extra_value_bps_v1": 8.0 if target == 1 else 32.0,
                "recommended_coarse_grid_name_v1": "SESSION_HOLD_GIVEBACK",
                "recommended_coarse_grid_value_v1": f"{session}|{hold_age}|{giveback}",
                "recommended_coarse_grid_viable_cell_v1": True,
                "shadow_score_v1": score,
                "shadow_bucket_status_v1": "TRAIN_SCORE_Q3_MID_20_PCT_APPROX",
                "shadow_bucket_rank_v1": 3,
                "shadow_score_coarse_band_v1": "HIGH" if target == 1 else "LOW",
                "overlay_session_axis_v1": session,
                "overlay_hold_age_axis_v1": hold_age,
                "overlay_giveback_axis_v1": giveback,
                "as_of_management_core_minutes_held_at_anchor_v1": 140.0 if target == 1 else 18.0,
                "as_of_management_core_giveback_ratio_from_peak_v1": 0.20 if target == 1 else 0.92,
                "as_of_atr_bps_v1": 28.0 if target == 1 else 11.0,
            }
        )

    for idx in range(12):
        rows.append(
            {
                "management_row_key_v1": f"exit_{idx:03d}",
                "split_bucket_v1": "TRAIN" if idx < 8 else "HOLDOUT",
                "observed_action_v1": "EXIT_NOW",
                "coarse_teacher_binary_target_v1": 1,
                "coarse_teacher_binary_target_eligible_v1": True,
                "coarse_teacher_feedback_label_v1": "OBSERVED_EXIT_DEFENSIBLE",
                "realized_pnl_bps": 12.0,
                "hold_longer_extra_value_bps_v1": 4.0,
                "recommended_coarse_grid_name_v1": "SESSION_HOLD_GIVEBACK",
                "recommended_coarse_grid_value_v1": "US|MID_31_120M|LOW_LT_0P50",
                "recommended_coarse_grid_viable_cell_v1": False,
                "shadow_score_v1": 0.5,
                "shadow_bucket_status_v1": "TRAIN_SCORE_Q3_MID_20_PCT_APPROX",
                "shadow_bucket_rank_v1": 3,
                "shadow_score_coarse_band_v1": "MID",
                "overlay_session_axis_v1": "US",
                "overlay_hold_age_axis_v1": "MID_31_120M",
                "overlay_giveback_axis_v1": "LOW_LT_0P50",
                "as_of_management_core_minutes_held_at_anchor_v1": 60.0,
                "as_of_management_core_giveback_ratio_from_peak_v1": 0.35,
                "as_of_atr_bps_v1": 16.0,
            }
        )

    pd.DataFrame.from_records(rows).to_parquet(teacher_view_path, index=False)

    payload = build_management_coarse_feedback_benchmark_payload(
        reports_root=reports_root,
        teacher_view_path=teacher_view_path,
        min_train_rows=20,
        min_validation_rows=10,
        min_holdout_rows=10,
    )

    summary = payload["summary"]
    assert summary["training_action_v1"] == "HOLD"
    assert summary["universe_counts_v1"]["eligible_action_rows_v1"] == 120
    assert summary["exit_feedback_status_v1"] == "POSITIVE_ONLY_NOT_RUN"
    assert summary["beats_current_bucket_baseline_v1"] is True
    assert summary["recommended_next_step_v1"] == "PROMOTE_TO_SHADOW_MANAGEMENT_RETRAIN_CANDIDATE"

    prediction_df = payload["prediction_df"]
    assert set(prediction_df["split_bucket_v1"].astype("string")) == {"VALIDATION", "HOLDOUT"}
    assert prediction_df["predicted_positive_prob_v1"].between(0.0, 1.0).all()

    threshold_sweep_df = payload["threshold_sweep_df"]
    assert not threshold_sweep_df.empty
    assert threshold_sweep_df["coverage_count_v1"].max() > 0

    written = write_management_coarse_feedback_benchmark_artifacts(
        reports_root=reports_root,
        teacher_view_path=teacher_view_path,
        min_train_rows=20,
        min_validation_rows=10,
        min_holdout_rows=10,
    )
    assert Path(written["predictions_path"]).exists()
    assert Path(written["threshold_sweep_path"]).exists()
    assert Path(written["summary_path"]).exists()

    on_disk = json.loads(Path(written["summary_path"]).read_text(encoding="utf-8"))
    assert on_disk["training_action_v1"] == "HOLD"
