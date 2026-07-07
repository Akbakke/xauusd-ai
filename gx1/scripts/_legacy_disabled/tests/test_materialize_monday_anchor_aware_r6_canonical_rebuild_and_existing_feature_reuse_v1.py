import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_anchor_aware_r6_canonical_rebuild_and_existing_feature_reuse_v1 import (
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


def _seed_wednesday(root: Path) -> None:
    freeze = root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    _write_json(
        freeze / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "policy_logging_v1": {"row_count_v1": 1971, "hindsight_backfill_rows_v1": 1971},
            "selected_candidate_v1": {
                "true_block_should_not_take_count_v1": 180,
                "true_block_tail_10_50_count_v1": 149,
                "precision_v1": 0.972972972972973,
                "worst_loso_precision_v1": 0.9285714285714286,
            },
        },
    )
    _write_json(
        freeze / WEDNESDAY_MANIFEST,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "as_of_schema_v1": {"column_count_v1": 109},
            "hindsight_schema_v1": {"column_count_v1": 30},
            "thresholds_v1": {
                "bad_threshold_v1": 0.95,
                "risky_threshold_v1": 0.85,
                "tail_threshold_v1": 0.9,
                "runner_threshold_v1": 0.6,
                "r5_2_runner_threshold_v1": 0.74,
                "blindspot_threshold_v1": 0.7,
                "use_r5_2_base_v1": True,
                "guard_v1": "hard_asof_runner_guard",
            },
        },
    )


def _foundation(root: Path) -> Path:
    out = root / "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_fixture"
    _write_json(
        out / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R6_ACTUAL_FULLCOVERAGE_FOUNDATION_BUILT",
            "row_count_v1": 3,
            "active_rows_v1": 2,
            "quarantine_rows_v1": 1,
            "as_of_column_count_v1": 109,
            "hindsight_output_column_count_v1": 58,
        },
    )
    _write_json(out / "foundation_contract_v1.json", {"foundation_universe_v1": "MONDAY_ACTUAL_FULLCOVERAGE_68_WEEK_REANCHOR"})
    _write_json(out / "foundation_label_summary_v1.json", {"row_count_v1": 3})
    frame = pd.DataFrame(
        {
            "run_id": ["W1", "W1", "W2"],
            "candidate_uid": ["c1", "c2", "c3"],
            "trade_uid": ["t1", "t2", "t3"],
            "trade_id": ["1", "2", "3"],
            "decision_timestamp": pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-08"], utc=True),
            "as_of_hour_utc_v1": [1, 2, 3],
            "as_of_skip_replay_window_range_15_bps_v1": [1.0, 2.0, 3.0],
            "calendar_quarantine_status_v1": ["ACTIVE_CANDIDATE", "ACTIVE_CANDIDATE", "QUARANTINED"],
            "calendar_quarantine_reason_v1": ["", "", "DECEMBER_BOUNDARY"],
            "label_should_not_take_v1": [True, False, True],
            "tail_10_50_mfe_v1": [True, False, False],
        }
    )
    frame.to_parquet(out / "monday_r6_foundation_training_frame_pre_score_v1.parquet", index=False)
    frame[["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "as_of_hour_utc_v1"]].to_parquet(
        out / "monday_r6_foundation_as_of_109_v1.parquet", index=False
    )
    frame[["candidate_uid", "run_id", "trade_uid", "trade_id", "decision_timestamp", "label_should_not_take_v1", "tail_10_50_mfe_v1"]].to_parquet(
        out / "monday_r6_foundation_hindsight_with_labels_v1.parquet", index=False
    )
    pd.DataFrame(
        [
            {"universe_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK", "row_count_v1": 1971, "delta_vs_monday_foundation_v1": 1968, "status_v1": "BENCHMARK_NOT_ROW_IDENTITY_TARGET_AFTER_MONDAY_REANCHOR"},
            {"universe_v1": "MONDAY_ACTUAL_FULLCOVERAGE_FOUNDATION", "row_count_v1": 3, "delta_vs_monday_foundation_v1": 0, "status_v1": "FOUNDATION_ROW_UNIVERSE"},
            {"universe_v1": "RUN::WEEK_ZERO", "row_count_v1": 0, "delta_vs_monday_foundation_v1": 0, "status_v1": "ACTIVE_CANDIDATE"},
        ]
    ).to_csv(out / "row_universe_delta_v1.csv", index=False)
    pd.DataFrame(
        [{"feature_v1": "as_of_hour_utc_v1", "role_v1": "AS_OF"}, {"feature_v1": "label_should_not_take_v1", "role_v1": "HINDSIGHT"}]
    ).to_csv(out / "feature_contract_audit_v1.csv", index=False)
    return out


def _score(root: Path, foundation_dir: Path) -> Path:
    out = root / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_fixture"
    _write_json(
        out / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
            "foundation_dir_v1": str(foundation_dir),
            "row_count_v1": 3,
        },
    )
    _write_json(
        out / "score_rebuild_summary_v1.json",
        {
            "r5_2_selected_policy_v1": {
                "metrics_v1": {"bad_blocks_v1": 1, "tail_help_v1": 1, "precision_v1": 1.0},
                "wednesday_safety_pass_v1": True,
            }
        },
    )
    pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2", "c3"],
            "as_of_hour_utc_v1": [1, 2, 3],
            "pred__entry_r5_2_bad_blocker__prob_true_v1": [0.9, 0.1, 0.2],
        }
    ).to_parquet(out / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    pd.DataFrame(
        [{"feature_v1": "as_of_hour_utc_v1"}, {"feature_v1": "pred__entry_r5_2_bad_blocker__prob_true_v1"}]
    ).to_csv(out / "feature_manifest_v1.csv", index=False)
    return out


def _r6(root: Path, score_dir: Path) -> Path:
    out = root / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_fixture"
    _write_json(
        out / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER",
            "score_dir_v1": str(score_dir),
            "compare_verdict_v1": "MONDAY_R6_EXPLICIT_REBUILD_SAFE_BUT_NOT_BETTER",
            "selected_policy_source_v1": "R6_FAMILY_GRID_SAFE_CANDIDATE",
        },
    )
    pd.DataFrame([{"feature_v1": "as_of_hour_utc_v1"}, {"feature_v1": "pred__entry_r5_2_bad_blocker__prob_true_v1"}]).to_csv(
        out / "feature_manifest_v1.csv", index=False
    )
    return out


def _recall(root: Path) -> Path:
    out = root / "MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK_V1_fixture"
    _write_json(out / "recall_gap_summary_v1.json", {"source_decision_v1": "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER"})
    pd.DataFrame(
        {
            "candidate_uid": ["c4", "c5"],
            "split_scope_v1": ["HOLDOUT", "VALIDATION"],
            "calendar_quarantine_status_v1": ["ACTIVE_CANDIDATE", "ACTIVE_CANDIDATE"],
            "miss_reason_not_r5_2_base_v1": [True, True],
            "miss_reason_r6_risky_score_below_099_v1": [True, True],
        }
    ).to_csv(out / "missed_bad_rows_v1.csv", index=False)
    pd.DataFrame({"candidate_uid": ["c4"], "split_scope_v1": ["HOLDOUT"]}).to_csv(out / "missed_tail_rows_v1.csv", index=False)
    pd.DataFrame(
        {"split_scope_v1": ["HOLDOUT"], "row_count_v1": [1], "bad_population_v1": [1], "selected_bad_blocks_v1": [0]}
    ).to_csv(out / "split_recall_gap_v1.csv", index=False)
    return out


def _path_dynamics(root: Path) -> Path:
    out = root / "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1_fixture"
    _write_json(out / "shadow_meta_path_dynamics_logging_v2_summary_v1.json", {"decision_v1": "PATH_DYNAMICS_V2_READY_FOR_R7_RETRAIN"})
    pd.DataFrame({"candidate_uid": ["c1"], "as_of_mgmt_trace_last_peak_mfe_bps_v1": [1.0]}).to_parquet(
        out / "shadow_meta_path_dynamics_logging_v2_as_of_raw_state_table_v1.parquet", index=False
    )
    pd.DataFrame({"candidate_uid": ["c1"], "as_of_management_core_mfe_bps_at_anchor_v1": [1.0]}).to_parquet(
        out / "shadow_meta_path_dynamics_logging_v2_policy_log_table_v1.parquet", index=False
    )
    return out


def test_anchor_aware_rebuild_materializes_reuse_and_r5_2_gate(tmp_path: Path) -> None:
    _seed_wednesday(tmp_path)
    foundation_dir = _foundation(tmp_path)
    score_dir = _score(tmp_path, foundation_dir)
    r6_dir = _r6(tmp_path, score_dir)
    recall_dir = _recall(tmp_path)
    path_dir = _path_dynamics(tmp_path)
    output_dir = tmp_path / "out"

    summary = materialize(
        reports_root=tmp_path,
        output_dir=output_dir,
        foundation_dir=foundation_dir,
        score_dir=score_dir,
        r6_dir=r6_dir,
        recall_gap_dir=recall_dir,
        path_dynamics_dir=path_dir,
    )

    assert summary["wednesday_is_comparator_not_monday_row_identity_target_v1"] is True
    assert summary["monday_expected_replay_rows_v1"] == 3
    assert summary["decision_v1"] == "FIX_R5_2_RECALL_BASE_FIRST"
    assert summary["next_action_v1"] == "FIX_R5_2_RECALL_BASE_FIRST"
    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    row_contract = json.loads((output_dir / "anchor_aware_row_universe_contract_v1.json").read_text())
    assert row_contract["wednesday_benchmark_universe_v1"]["role_v1"] == "BENCHMARK_AND_COMPARATOR_NOT_AUTOMATIC_MONDAY_TARGET"
    delta = pd.read_csv(output_dir / "anchor_aware_row_delta_explainer_v1.csv")
    assert "EXPECTED_DUE_TO_MONDAY_ANCHOR" in set(delta["status_v1"])
    assert "EXPECTED_QUARANTINE" in set(delta["status_v1"])
    inventory = pd.read_csv(output_dir / "existing_feature_asset_inventory_v1.csv")
    assert "REUSE_NOW" in set(inventory["status_v1"])
    assert "REUSE_FOR_TRANSFORMER_OR_RL_ONLY" in set(inventory["status_v1"])
    reuse = pd.read_csv(output_dir / "entry_exit_transformer_and_pre_rl_reuse_map_v1.csv")
    assert "exit-transformer candidate features" in set(reuse["surface_v1"])
