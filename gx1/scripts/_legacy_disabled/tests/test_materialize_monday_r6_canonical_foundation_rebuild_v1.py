import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_canonical_foundation_rebuild_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import AS_OF_TABLE, HINDSIGHT_TABLE, TRUTH_TABLE


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_foundation_rebuild_materializes_monday_actual_fullcoverage_not_1689_or_1852(tmp_path: Path) -> None:
    reports_root = tmp_path
    monday_dir = reports_root / "MONDAY_R6_CANONICAL_TRUTH_V1_fixture"
    rehydrated_dir = reports_root / "MONDAY_R6_REHYDRATED_WEDNESDAY_CONTRACT_V1_fixture"
    monday_dir.mkdir()
    rehydrated_dir.mkdir()

    rows = [
        {
            "run_id": "TRUTH_MONFRI_WEEK_20251201_20251208",
            "candidate_uid": "c1",
            "trade_uid": "t1",
            "trade_id": "trade1",
            "decision_timestamp": "2025-12-02T12:00:00Z",
            "used_for_training": False,
            "used_for_validation": False,
            "used_for_holdout": False,
            "as_of_candidate_tradable_prob_v1": 0.96,
            "as_of_entry_candidate_path_quality_pred_v1": 0.75,
            "as_of_candidate_mfe_first_n_pred_v1": 2.0,
            "as_of_skip_candidate_p_flat_v1": 0.2,
        },
        {
            "run_id": "TRUTH_MONFRI_WEEK_20260105_20260112",
            "candidate_uid": "c2",
            "trade_uid": "t2",
            "trade_id": "trade2",
            "decision_timestamp": "2026-01-06T12:00:00Z",
            "used_for_training": False,
            "used_for_validation": False,
            "used_for_holdout": False,
            "as_of_candidate_tradable_prob_v1": 0.70,
            "as_of_entry_candidate_path_quality_pred_v1": 0.40,
            "as_of_candidate_mfe_first_n_pred_v1": 0.5,
            "as_of_skip_candidate_p_flat_v1": 0.7,
        },
    ]
    asof = pd.DataFrame(rows)
    for idx in range(109 - len(asof.columns)):
        asof[f"as_of_dummy_{idx:03d}_v1"] = float(idx)
    asof.to_parquet(rehydrated_dir / AS_OF_TABLE, index=False)

    hindsight = pd.DataFrame(
        [
            {
                "run_id": "TRUTH_MONFRI_WEEK_20251201_20251208",
                "candidate_uid": "c1",
                "trade_uid": "t1",
                "trade_id": "trade1",
                "decision_timestamp": "2025-12-02T12:00:00Z",
                "baseline_realized_pnl_bps_v1": 15.0,
                "peak_mfe_bps_v1": 80.0,
                "mae_abs_bps_v1": 10.0,
                "giveback_bps_v1": 5.0,
                "hindsight_entry_decision_review_v1": "TAKE_WAS_OK",
                "hindsight_management_review_v1": "OK",
            },
            {
                "run_id": "TRUTH_MONFRI_WEEK_20260105_20260112",
                "candidate_uid": "c2",
                "trade_uid": "t2",
                "trade_id": "trade2",
                "decision_timestamp": "2026-01-06T12:00:00Z",
                "baseline_realized_pnl_bps_v1": -10.0,
                "peak_mfe_bps_v1": 5.0,
                "mae_abs_bps_v1": 55.0,
                "giveback_bps_v1": 0.0,
                "hindsight_entry_decision_review_v1": "SHOULD_NOT_TAKE",
                "hindsight_management_review_v1": "BAD",
            },
        ]
    )
    hindsight.to_parquet(rehydrated_dir / HINDSIGHT_TABLE, index=False)

    truth = pd.DataFrame(
        [
            {
                "candidate_uid": "c1",
                "calendar_quarantine_status_v1": "QUARANTINED",
                "calendar_quarantine_reason_v1": "fixture",
                "truth_cata_or_friday_flat_damage_v1": False,
                "truth_exit_too_early_regret_replay_end_v1": False,
                "canonical_entry_ts_utc_v1": "2025-12-02T12:00:00Z",
            },
            {
                "candidate_uid": "c2",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "calendar_quarantine_reason_v1": None,
                "truth_cata_or_friday_flat_damage_v1": True,
                "truth_exit_too_early_regret_replay_end_v1": False,
                "canonical_entry_ts_utc_v1": "2026-01-06T12:00:00Z",
            },
        ]
    )
    truth.to_parquet(monday_dir / TRUTH_TABLE, index=False)
    pd.DataFrame(
        [
            {"run_id": "TRUTH_MONFRI_WEEK_20251201_20251208", "calendar_quarantine_status_v1": "QUARANTINED", "outcome_rows_v1": 1},
            {"run_id": "TRUTH_MONFRI_WEEK_20260105_20260112", "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE", "outcome_rows_v1": 1},
        ]
    ).to_csv(monday_dir / "monday_r6_truth_run_inventory_v1.csv", index=False)

    output_dir = tmp_path / "out"
    summary = materialize(reports_root=reports_root, monday_truth_dir=monday_dir, rehydrated_dir=rehydrated_dir, output_dir=output_dir)

    assert summary["decision_v1"] == "MONDAY_R6_ACTUAL_FULLCOVERAGE_FOUNDATION_BUILT"
    assert summary["training_started_v1"] is False
    assert summary["as_of_column_count_v1"] == 109
    assert summary["row_count_v1"] == 2
    assert summary["quarantine_rows_v1"] == 1
    assert "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE" in summary["blocked_action_v1"]
    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    audit = pd.read_csv(output_dir / OUTPUT_FILES["audit"])
    assert audit.set_index("check_v1").loc["AS_OF_109_PRESENT", "status_v1"] == "PASS"
