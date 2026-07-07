import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_rehydrated_wednesday_contract_v1 import (
    ID_COLS,
    SCORE_OR_POLICY_COLS,
    WEDNESDAY_FREEZE_DIR,
    WEDNESDAY_MANIFEST,
    WEDNESDAY_SNAPSHOT_DIR,
    WEDNESDAY_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _schema(names: list[str]) -> dict:
    return {
        "column_count_v1": len(names),
        "columns_v1": [{"name_v1": name, "dtype_v1": "object"} for name in names],
    }


def test_rehydrator_restores_wednesday_schema_shape_and_blocks_missing_scores(tmp_path: Path) -> None:
    asof_cols = [
        *ID_COLS,
        "used_for_training",
        "used_for_validation",
        "used_for_holdout",
        "as_of_hour_utc_v1",
        "as_of_weekday_utc_v1",
        "as_of_session_v1",
        "as_of_side_v1",
        "as_of_candidate_entry_spread_bps_v1",
        "as_of_skip_xgb_p_hat_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        *sorted(SCORE_OR_POLICY_COLS),
        "entry_observation_present_v1",
        "entry_raw_state_present_v1",
        "management_observation_present_v1",
        "entry_coverage_original_entry_observation_present_v1",
        "entry_coverage_original_entry_raw_state_present_v1",
        "entry_coverage_repair_applied_v1",
        "entry_coverage_repair_source_v1",
        "r6_as_of_feature_contract_v1",
    ]
    while len(asof_cols) < 109:
        asof_cols.append(f"as_of_fixture_padding_{len(asof_cols)}_v1")

    hindsight_cols = [
        *ID_COLS,
        "baseline_realized_pnl_bps_v1",
        "peak_mfe_bps_v1",
        "mae_abs_bps_v1",
        "giveback_bps_v1",
        "hindsight_entry_decision_review_v1",
        "hindsight_management_review_v1",
        "r6_label_runner_50_mfe_v1",
        "r6_label_runner_near_miss_v1",
        "r6_label_bad_risk_v1",
        "r6_hindsight_contract_v1",
    ]
    while len(hindsight_cols) < 30:
        hindsight_cols.append(f"r6_label_fixture_padding_{len(hindsight_cols)}_v1")

    freeze_dir = tmp_path / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    _write_json(
        freeze_dir / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        },
    )
    _write_json(
        freeze_dir / WEDNESDAY_MANIFEST,
        {
            "as_of_schema_v1": _schema(asof_cols),
            "hindsight_schema_v1": _schema(hindsight_cols),
            "score_head_names_v1": {"blocker_score_v1": "pred__entry_r6_bad_risk__prob_true_v1"},
        },
    )

    monday_dir = tmp_path / "MONDAY_R6_CANONICAL_TRUTH_V1_fixture"
    monday_dir.mkdir()
    pd.DataFrame(
        {
            "run_id": ["TRUTH_MONFRI_WEEK_20260105_20260112"],
            "candidate_uid": ["cand-1"],
            "trade_uid": ["trade-1"],
            "trade_id": ["SIM-1"],
            "decision_timestamp_v1": ["2026-01-06T10:00:00Z"],
            "canonical_entry_ts_utc_v1": ["2026-01-06T10:00:00Z"],
            "canonical_pnl_bps_v1": [12.0],
            "canonical_mfe_bps_v1": [75.0],
            "canonical_mae_bps_v1": [-3.0],
            "entry_candidate_hour_utc_v1": [10],
            "entry_candidate_weekday_utc_v1": [1],
            "entry_candidate_session_v1": ["OVERLAP"],
            "entry_candidate_side_v1": ["long"],
            "entry_candidate_entry_spread_bps_v1": [1.0],
            "entry_xgb_p_hat_v1": [0.8],
            "journal_exit_reason_v1": ["THRESHOLD"],
            "truth_cata_or_friday_flat_damage_v1": [False],
            "truth_exit_too_early_regret_replay_end_v1": [False],
        }
    ).to_parquet(monday_dir / "monday_r6_trade_truth_v1.parquet", index=False)
    pd.DataFrame(
        {
            "run_id": ["TRUTH_MONFRI_WEEK_20260105_20260112"] * 15,
            "time": pd.date_range("2026-01-06T09:46:00Z", periods=15, freq="min"),
            "open": [100.0 + i for i in range(15)],
            "high": [101.0 + i for i in range(15)],
            "low": [99.0 + i for i in range(15)],
            "close": [100.5 + i for i in range(15)],
            "spread_bps": [1.0] * 15,
        }
    ).to_parquet(monday_dir / "monday_r6_bar_feature_surface_v1.parquet", index=False)

    summary = materialize(reports_root=tmp_path, monday_truth_dir=monday_dir, output_dir=tmp_path / "out")

    assert summary["as_of_column_count_v1"] == 109
    assert summary["hindsight_column_count_v1"] == 30
    assert summary["blocked_score_column_count_v1"] == len(SCORE_OR_POLICY_COLS)
    assert summary["training_started_v1"] is False
    assert summary["decision_v1"] == "MONDAY_R6_WEDNESDAY_CONTRACT_SHAPE_REHYDRATED_BUT_NOT_TRAINING_READY"

    asof = pd.read_parquet(tmp_path / "out" / "monday_r6_entry_runner_first_as_of_feature_table_v1.parquet")
    hindsight = pd.read_parquet(tmp_path / "out" / "monday_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet")
    assert list(asof.columns) == asof_cols
    assert list(hindsight.columns) == hindsight_cols
    assert asof["used_for_training"].tolist() == [False]
    assert asof["pred__entry_r5_2_bad_blocker__prob_true_v1"].isna().all()
