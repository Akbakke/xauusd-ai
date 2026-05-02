from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_readonly_diagnosis_and_next_step_lock_v1 import (
    COMPARATOR_HIERARCHY,
    CONSISTENCY_AUDIT,
    FAILURE_GAP_MAP,
    NEXT_STEP_LOCK,
    PATH_DYNAMICS_LOCK,
    REPAIRED_165_FORENSIC,
    RESULT_RECHECK,
    RETRAIN_DECISION,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path) -> Path:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
    freeze_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R5_2_SHADOW_FREEZE_AND_R6_FAILURE_BACKLOG_V1"
    r6_dir.mkdir()
    freeze_dir.mkdir()

    _write_json(
        reports_root / "truth_r6_entry_runner_first_retrain_v1.json",
        {
            "extension_dir_v1": str(r6_dir),
            "decision_v1": {
                "recommended_next_step_v1": "R6_FEATURES_INSUFFICIENT",
                "r5_2_bad_blocks_v1": 106,
                "r5_2_tail_help_v1": 82,
            },
            "selected_candidate_v1": {
                "family_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                "policy_name_v1": "R6_CANDIDATE_04789_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                "should_not_take_block_count_v1": 84,
                "tail_10_50_help_count_v1": 84,
                "should_not_take_precision_v1": 0.9545454545,
                "worst_loso_precision_v1": 0.8888888888,
                "batch04_loso_pass_v1": True,
                "batch05_loso_pass_v1": None,
                "fifty_plus_mfe_block_count_v1": 1,
                "hundred_plus_mfe_block_count_v1": 0,
                "two_hundred_plus_mfe_block_count_v1": 0,
                "strong_trade_false_block_count_v1": 0,
                "strongest_winner_path_block_count_v1": 0,
                "repaired_165_block_count_v1": 1,
                "r6_contract_failure_reasons_v1": "repaired_165_block_count_v1!=0,precision<R5.2,bad_blocks<=R5.2,worst_loso_precision<R5.2",
            },
            "status_v1": {"R6_STATUS": "TRAINED_SHADOW_RESEARCH_NOT_PROMOTED"},
        },
    )
    _write_json(
        reports_root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json",
        {
            "extension_dir_v1": str(freeze_dir),
            "failure_counts_v1": {
                "missed_should_not_take_v1": 462,
                "missed_10_50_tail_control_v1": 198,
                "risky_allows_v1": 347,
                "runner_near_misses_v1": 83,
            },
            "selected_policy_stack_v1": "R5_2_CANDIDATE_00001_TWO_HEAD_DIRECT_none",
        },
    )
    _write_json(
        reports_root / "truth_r5_loso_batch04_robustness_retrain_v1.json",
        {
            "selected_candidate_v1": {
                "policy_name_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like",
                "should_not_take_block_count_v1": 66,
                "tail_10_50_help_count_v1": 66,
                "should_not_take_precision_v1": 0.9295774647,
            }
        },
    )

    pd.DataFrame(
        [
            {"policy_name_v1": "R6_SELECTED_CANDIDATE", "scope_v1": "ALL_1971", "should_not_take_block_count_v1": 84},
            {"policy_name_v1": "R5_1_SAFETY_REFERENCE", "scope_v1": "ALL_1971", "should_not_take_block_count_v1": 66},
        ]
    ).to_csv(r6_dir / "shadow_meta_all_trade_review_r6_head_to_head_vs_r2_r4_r5_r5_1_r5_2_v1.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy_name_v1": "R6_CANDIDATE_04789_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                "scope_v1": "BATCH_04",
                "slice_safety_pass_v1": True,
            }
        ]
    ).to_csv(r6_dir / "shadow_meta_all_trade_review_r6_loso_metrics_v1.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "cand-bad",
                "run_id": "TRUTH_MONFRI_WEEK_20260330_20260406",
                "trade_uid": "trade-bad",
                "is_repaired_165_v1": True,
                "r6_selected_candidate__block_v1": True,
                "peak_mfe_bps_v1": 96.010851,
                "baseline_realized_pnl_bps_v1": 62.95,
                "mae_abs_bps_v1": 39.446657,
                "label_should_not_take_v1": False,
                "take_was_ok_v1": True,
                "pred__entry_r6_bad_risk__prob_true_v1": 0.996817,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.102381,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.948108,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.997687,
            },
            {
                "candidate_uid": "cand-good",
                "run_id": "TRUTH_MONFRI_WEEK_20260119_20260126",
                "trade_uid": "trade-good",
                "is_repaired_165_v1": True,
                "r6_selected_candidate__block_v1": False,
                "peak_mfe_bps_v1": 221.1,
                "baseline_realized_pnl_bps_v1": 198.16,
                "mae_abs_bps_v1": 4.16,
                "label_should_not_take_v1": False,
                "take_was_ok_v1": True,
                "pred__entry_r6_bad_risk__prob_true_v1": 0.1,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.9,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.1,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.1,
            },
        ]
    ).to_parquet(r6_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "contrast_name_v1": "PATH_DYNAMICS_LOGGING_COVERAGE",
                "feature_family_v1": "new_path_dynamics_logging",
                "feature_count_v1": 1,
                "positive_count_v1": 0,
                "negative_count_v1": 1,
                "mean_top5_effect_score_v1": None,
                "max_effect_score_v1": None,
                "top_features_json_v1": '[{"feature_v1":"as_of_last_peak_ts_utc_v1","present_v1":false}]',
                "path_dynamics_status_v1": "LOGGING_BLOCKED",
            }
        ]
    ).to_csv(r6_dir / "shadow_meta_all_trade_review_r6_feature_path_dynamics_audit_v1.csv", index=False)

    pd.DataFrame(
        [
            {"failure_type_v1": "MISSED_SHOULD_NOT_TAKE", "count_v1": 462, "failure_driver_assessment_v1": "FEATURE_OR_LABEL_BLIND_SPOT_DRIVEN"},
            {"failure_type_v1": "MISSED_10_50_TAIL_CONTROL", "count_v1": 198, "failure_driver_assessment_v1": "FEATURE_OR_LABEL_BLIND_SPOT_DRIVEN"},
            {"failure_type_v1": "RISKY_ALLOW", "count_v1": 347, "failure_driver_assessment_v1": "CALIBRATION_OR_PROTECTOR_SUPPRESSION_DRIVEN"},
            {"failure_type_v1": "RUNNER_NEAR_MISS", "count_v1": 83, "failure_driver_assessment_v1": "PROTECTION_DRIVEN_R6_MUST_STRENGTHEN_RUNNER_GUARD"},
        ]
    ).to_csv(freeze_dir / "shadow_meta_all_trade_review_r6_failure_cluster_table_v1.csv", index=False)
    pd.DataFrame(
        [
            {
                "r6_direction_v1": "BETTER_SHOULD_NOT_TAKE_AND_RISKY_ALLOW_LABELS",
                "addressed_failure_types_v1": "MISSED_SHOULD_NOT_TAKE,RISKY_ALLOW",
                "evidence_v1": "462 missed should-not-take; blocker under-recall remains.",
            },
            {
                "r6_direction_v1": "BETTER_10_50_TAIL_CONTROL_LABELS_AND_FEATURES",
                "addressed_failure_types_v1": "MISSED_10_50_TAIL_CONTROL",
                "evidence_v1": "198 tail misses remain.",
            },
            {
                "r6_direction_v1": "RUNNER_NEAR_MISS_PROTECTION_BEFORE_MORE_RECALL",
                "addressed_failure_types_v1": "RUNNER_NEAR_MISS",
                "evidence_v1": "83 runner near-misses remain.",
            },
        ]
    ).to_csv(freeze_dir / "shadow_meta_all_trade_review_r6_label_feature_opportunity_audit_v1.csv", index=False)
    pd.DataFrame(
        [
            {"requirement_v1": "repaired_165_damage", "r5_2_baseline_value_v1": 0, "r6_required_value_v1": 0},
            {"requirement_v1": "bad_blocks", "r5_2_baseline_value_v1": 106, "r6_required_value_v1": 107},
        ]
    ).to_csv(freeze_dir / "shadow_meta_all_trade_review_r5_2_vs_r6_go_no_go_matrix_v1.csv", index=False)

    comparator_dir = reports_root / "MONDAY_TOP_PRE_RL_BASELINE_COMPARATOR_V1_20260424T063650Z"
    comparator_dir.mkdir()
    _write_json(
        comparator_dir / "summary_v1.json",
        {
            "benchmark_r6_v1": {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"},
        },
    )

    snapshot_dir = reports_root / "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
    snapshot_dir.mkdir()
    _write_json(snapshot_dir / "summary_v1.json", {"artifact_name_v1": "snapshot"})
    _write_json(
        snapshot_dir
        / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json",
        {
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "selected_candidate_v1": {
                "should_not_take_block_count_v1": 180,
                "tail_10_50_help_count_v1": 149,
                "should_not_take_precision_v1": 0.9729729729,
                "worst_loso_precision_v1": 0.9285714285,
                "repaired_165_block_count_v1": 0,
            },
        },
    )
    _write_json(
        snapshot_dir
        / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/shadow_meta_all_trade_review_r6_shadow_freeze_manifest_v1.json",
        {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"},
    )
    _write_json(
        snapshot_dir
        / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1/shadow_meta_path_dynamics_instrumentation_spec_v2.json",
        {
            "fields_v1": [
                {
                    "field_name_v1": "as_of_last_peak_ts_utc_v1",
                    "as_of_semantics_v1": "LEGAL_FOR_MANAGEMENT_EXIT_ANCHOR; NOT_LEGAL_FOR_PRE_ENTRY_IF_COMPUTED_FROM_THIS_TRADE_FUTURE_PATH",
                },
                {
                    "field_name_v1": "as_of_last_mfe_ts_utc_v1",
                    "as_of_semantics_v1": "LEGAL_FOR_MANAGEMENT_EXIT_ANCHOR; NOT_LEGAL_FOR_PRE_ENTRY_IF_COMPUTED_FROM_THIS_TRADE_FUTURE_PATH",
                },
                {
                    "field_name_v1": "as_of_last_peak_mfe_bps_v1",
                    "as_of_semantics_v1": "LEGAL_FOR_MANAGEMENT_EXIT_ANCHOR; NOT_LEGAL_FOR_PRE_ENTRY_IF_COMPUTED_FROM_THIS_TRADE_FUTURE_PATH",
                },
                {
                    "field_name_v1": "as_of_max_mfe_without_mae_bps_v1",
                    "as_of_semantics_v1": "LEGAL_FOR_MANAGEMENT_EXIT_ANCHOR; NOT_LEGAL_FOR_PRE_ENTRY_IF_COMPUTED_FROM_THIS_TRADE_FUTURE_PATH",
                },
                {
                    "field_name_v1": "as_of_mfe_mae_sequence_order_v1",
                    "as_of_semantics_v1": "LEGAL_FOR_MANAGEMENT_EXIT_ANCHOR; NOT_LEGAL_FOR_PRE_ENTRY_IF_COMPUTED_FROM_THIS_TRADE_FUTURE_PATH",
                },
            ]
        },
    )

    path_dir = reports_root / "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1_20260424T075335Z"
    path_dir.mkdir()
    coverage_rows = []
    for field_id, trace_name, raw_name, policy_name in [
        ("last_peak_ts", "last_peak_ts_utc", "as_of_mgmt_trace_last_peak_ts_utc_v1", "as_of_management_core_last_peak_ts_utc_v1"),
        ("last_mfe_ts", "last_mfe_ts_utc", "as_of_mgmt_trace_last_mfe_ts_utc_v1", "as_of_management_core_last_mfe_ts_utc_v1"),
        ("last_peak_mfe", "last_peak_mfe_bps", "as_of_mgmt_trace_last_peak_mfe_bps_v1", "as_of_management_core_last_peak_mfe_bps_v1"),
        ("max_mfe_without_mae", "max_mfe_without_mae_bps", "as_of_mgmt_trace_max_mfe_without_mae_bps_v1", "as_of_management_core_max_mfe_without_mae_bps_v1"),
        ("mfe_mae_sequence_order", "mfe_mae_sequence_order", "as_of_mgmt_trace_mfe_mae_sequence_order_v1", "as_of_management_core_mfe_mae_sequence_order_v1"),
    ]:
        coverage_rows.extend(
            [
                {"field_id": field_id, "field_name": trace_name, "layer_name": "EXIT_EVAL_TRACE", "null_count": 0},
                {"field_id": field_id, "field_name": raw_name, "layer_name": "RAW_STATE", "null_count": 0},
                {"field_id": field_id, "field_name": policy_name, "layer_name": "POLICY_LOG", "null_count": 0},
            ]
        )
    _write_json(path_dir / "shadow_meta_path_dynamics_logging_v2_summary_v1.json", {"coverage_summary_v1": coverage_rows})
    return reports_root


def test_monday_r6_readonly_diagnosis_materializes(tmp_path: Path) -> None:
    reports_root = _build_fixture(tmp_path)
    extension_dir = reports_root / "diagnosis"
    result = materialize(reports_root, extension_dir=extension_dir)
    assert result["status"]["READONLY_DIAGNOSIS_STATUS"] == "MATERIALIZED"
    for artifact in [
        RESULT_RECHECK,
        COMPARATOR_HIERARCHY,
        REPAIRED_165_FORENSIC,
        FAILURE_GAP_MAP,
        PATH_DYNAMICS_LOCK,
        RETRAIN_DECISION,
        NEXT_STEP_LOCK,
        SUMMARY,
        CONSISTENCY_AUDIT,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["monday_r6_rechecked_v1"] is True
    assert summary["correct_benchmark_v1"] == "FROZEN_R6_BENCHMARK"
    retrain = json.loads((extension_dir / RETRAIN_DECISION).read_text(encoding="utf-8"))
    assert retrain["decision_v1"] == "DO_NOT_RETRAIN_YET"
    forensic = json.loads((extension_dir / REPAIRED_165_FORENSIC).read_text(encoding="utf-8"))
    assert forensic["blocked_repaired_row_count_v1"] == 1
    path_df = pd.read_csv(extension_dir / PATH_DYNAMICS_LOCK)
    assert path_df["future_use_status_v1"].astype("string").eq("NOT_CANONICAL_YET").all()
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
