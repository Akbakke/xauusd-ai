import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_wednesday_contract_completeness_check_v1 import (
    LOCAL_R6_DIR_NAME,
    OUTPUT_FILES,
    WEDNESDAY_FREEZE_DIR,
    WEDNESDAY_MANIFEST,
    WEDNESDAY_SNAPSHOT_DIR,
    WEDNESDAY_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_completeness_check_blocks_local_r6_that_is_not_canonical_wednesday(tmp_path: Path) -> None:
    freeze_dir = tmp_path / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    _write_json(
        freeze_dir / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "policy_logging_v1": {"row_count_v1": 1971, "hindsight_backfill_rows_v1": 1971},
        },
    )
    _write_json(
        freeze_dir / WEDNESDAY_MANIFEST,
        {
            "model_version_id_v1": "R6_ENTRY_RUNNER_FIRST_GLOBAL_FIVE_HEAD_20260422_V1",
            "threshold_version_id_v1": "R6_THRESHOLDS_20260422T_SELECTED_CANDIDATE_04761_V1",
            "r6_source_dir_v1": str(tmp_path / "missing_canonical_source"),
            "score_head_names_v1": {"blocker_score_v1": "pred__entry_r6_bad_risk__prob_true_v1"},
            "as_of_schema_v1": {
                "column_count_v1": 3,
                "columns_v1": [
                    {"name_v1": "run_id", "dtype_v1": "string"},
                    {"name_v1": "candidate_uid", "dtype_v1": "object"},
                    {"name_v1": "pred__entry_r6_bad_risk__prob_true_v1", "dtype_v1": "float64"},
                ],
            },
            "hindsight_schema_v1": {
                "column_count_v1": 2,
                "columns_v1": [
                    {"name_v1": "candidate_uid", "dtype_v1": "object"},
                    {"name_v1": "r6_label_tail_control_10_50_v1", "dtype_v1": "bool"},
                ],
            },
        },
    )

    monday_dir = tmp_path / "MONDAY_R6_CANONICAL_TRUTH_V1_fixture"
    monday_dir.mkdir()
    _write_json(
        monday_dir / "monday_r6_truth_coverage_summary_v1.json",
        {
            "trade_truth_rows_v1": 1914,
            "candidate_surface_rows_v1": 10,
            "bar_feature_rows_v1": 10,
            "exit_eval_trace_rows_v1": 10,
            "xgb_signal_rows_v1": 10,
            "feature_manifest_rows_v1": 2,
        },
    )
    pd.DataFrame(
        [
            {"surface_v1": "trade_truth", "feature_name_v1": "candidate_uid", "role_v1": "id"},
            {"surface_v1": "trade_truth", "feature_name_v1": "run_id", "role_v1": "id"},
        ]
    ).to_csv(monday_dir / "monday_r6_truth_feature_manifest_v1.csv", index=False)

    local_r6 = tmp_path / LOCAL_R6_DIR_NAME
    local_r6.mkdir()
    _write_json(
        local_r6 / "shadow_meta_all_trade_review_r6_summary_v1.json",
        {"selected_candidate_v1": {"policy_name_v1": "R6_CANDIDATE_04789_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON"}},
    )
    pd.DataFrame({"run_id": ["r"], "candidate_uid": ["c"]}).to_parquet(
        local_r6 / "shadow_meta_all_trade_review_r6_entry_runner_first_as_of_feature_table_v1.parquet",
        index=False,
    )
    pd.DataFrame({"candidate_uid": ["c"]}).to_parquet(
        local_r6 / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet",
        index=False,
    )
    pd.DataFrame({"candidate_uid": ["c"]}).to_parquet(
        local_r6 / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet",
        index=False,
    )

    output_dir = tmp_path / "out"
    summary = materialize(reports_root=tmp_path, monday_truth_dir=monday_dir, output_dir=output_dir)

    assert summary["decision_v1"] == "MONDAY_R6_TRUTH_BUILT_BUT_WEDNESDAY_R6_CONTRACT_NOT_FULLY_RESTORED"
    assert summary["local_r6_alternate_assessment_v1"] == "PRESENT_BUT_NOT_CANONICAL_WEDNESDAY_R6"
    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    audit = pd.read_csv(output_dir / "consistency_audit_v1.csv")
    assert audit.set_index("check_v1").loc["LOCAL_R6_ALTERNATE_IS_CANONICAL_WEDNESDAY_R6", "status_v1"] == "FAIL"
