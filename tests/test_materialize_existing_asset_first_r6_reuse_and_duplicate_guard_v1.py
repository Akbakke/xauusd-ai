import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_existing_asset_first_r6_reuse_and_duplicate_guard_v1 import (
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


def _seed_snapshot(root: Path) -> None:
    freeze = root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    _write_json(
        freeze / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "policy_logging_v1": {"row_count_v1": 1971},
        },
    )
    _write_json(
        freeze / WEDNESDAY_MANIFEST,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "as_of_schema_v1": {"column_count_v1": 109},
        },
    )


def _seed_foundation(root: Path) -> Path:
    out = root / "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_fixture"
    _write_json(out / "summary_v1.json", {"row_count_v1": 3, "active_rows_v1": 2, "quarantine_rows_v1": 1, "as_of_column_count_v1": 109})
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
            "label_should_not_take_v1": [True, False, True],
            "calendar_quarantine_status_v1": ["ACTIVE_CANDIDATE", "ACTIVE_CANDIDATE", "QUARANTINED"],
        }
    )
    frame.to_parquet(out / "monday_r6_foundation_training_frame_pre_score_v1.parquet", index=False)
    frame[["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp", "as_of_hour_utc_v1"]].to_parquet(
        out / "monday_r6_foundation_as_of_109_v1.parquet", index=False
    )
    frame[["candidate_uid", "run_id", "trade_uid", "trade_id", "decision_timestamp", "label_should_not_take_v1"]].to_parquet(
        out / "monday_r6_foundation_hindsight_with_labels_v1.parquet", index=False
    )
    pd.DataFrame(
        [
            {"universe_v1": "MONDAY_ACTUAL_FULLCOVERAGE_FOUNDATION", "row_count_v1": 3, "status_v1": "FOUNDATION_ROW_UNIVERSE"},
            {"universe_v1": "RUN::WEEK_ZERO", "row_count_v1": 0, "status_v1": "ACTIVE_CANDIDATE"},
        ]
    ).to_csv(out / "row_universe_delta_v1.csv", index=False)
    pd.DataFrame([{"feature_v1": "as_of_hour_utc_v1"}, {"feature_v1": "label_should_not_take_v1"}]).to_csv(
        out / "feature_contract_audit_v1.csv", index=False
    )
    return out


def _seed_score(root: Path, foundation: Path) -> Path:
    out = root / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_fixture"
    _write_json(out / "summary_v1.json", {"decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED", "foundation_dir_v1": str(foundation), "row_count_v1": 3})
    _write_json(
        out / "score_rebuild_summary_v1.json",
        {"r5_2_selected_policy_v1": {"metrics_v1": {"bad_blocks_v1": 1, "tail_help_v1": 1}}},
    )
    score = pd.DataFrame({"candidate_uid": ["c1", "c2", "c3"], "pred__entry_r5_2_bad_blocker__prob_true_v1": [0.9, 0.1, 0.2]})
    for filename in [
        "monday_r6_foundation_score_frame_v1.parquet",
        "monday_r5_score_prediction_view_v1.parquet",
        "monday_r5_1_score_prediction_view_v1.parquet",
        "monday_r5_2_score_prediction_view_v1.parquet",
    ]:
        score.to_parquet(out / filename, index=False)
    pd.DataFrame([{"feature_v1": "pred__entry_r5_2_bad_blocker__prob_true_v1"}]).to_csv(out / "feature_manifest_v1.csv", index=False)
    pd.DataFrame([{"head_v1": "r5_2"}]).to_csv(out / "model_metrics_v1.csv", index=False)
    return out


def _seed_r6(root: Path, score: Path) -> Path:
    out = root / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_fixture"
    _write_json(
        out / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER",
            "score_dir_v1": str(score),
            "foundation_score_context_column_count_v1": 17,
        },
    )
    frame = pd.DataFrame({"candidate_uid": ["c1", "c2"], "pred__entry_r6_bad_risk__prob_true_v1": [0.8, 0.2]})
    frame.to_parquet(out / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    frame.to_parquet(out / "monday_r6_on_foundation_scores_prediction_view_v1.parquet", index=False)
    _write_json(out / "eval_summary_v1.json", {"bad_blocks_v1": 1})
    _write_json(out / "compare_against_wednesday_r6_v1.json", {"verdict_v1": "SAFE_BUT_NOT_BETTER"})
    pd.DataFrame([{"policy_name_v1": "R6_CANDIDATE"}]).to_csv(out / "r6_family_grid_replay_v1.csv", index=False)
    pd.DataFrame([{"feature_v1": "pred__entry_r6_bad_risk__prob_true_v1"}]).to_csv(out / "feature_manifest_v1.csv", index=False)
    pd.DataFrame([{"head_v1": "r6"}]).to_csv(out / "model_metrics_v1.csv", index=False)
    return out


def _seed_recall(root: Path) -> Path:
    out = root / "MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK_V1_fixture"
    _write_json(out / "recall_gap_summary_v1.json", {"source_decision_v1": "SAFE_BUT_NOT_BETTER"})
    pd.DataFrame({"candidate_uid": ["c4"], "miss_reason_not_r5_2_base_v1": [True]}).to_csv(out / "missed_bad_rows_v1.csv", index=False)
    pd.DataFrame({"candidate_uid": ["c5"]}).to_csv(out / "missed_tail_rows_v1.csv", index=False)
    pd.DataFrame({"split_scope_v1": ["HOLDOUT"], "selected_bad_blocks_v1": [0]}).to_csv(out / "split_recall_gap_v1.csv", index=False)
    return out


def _seed_path_and_anchor(root: Path) -> tuple[Path, Path]:
    path_dir = root / "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1_fixture"
    _write_json(path_dir / "shadow_meta_path_dynamics_logging_v2_summary_v1.json", {"decision_v1": "PATH_DYNAMICS_V2_READY_FOR_R7_RETRAIN"})
    _write_json(path_dir / "shadow_meta_path_dynamics_logging_v2_contract_v1.json", {"not_live_gate_v1": True})
    pd.DataFrame({"candidate_uid": ["c1"], "as_of_mgmt_trace_last_peak_mfe_bps_v1": [1.0]}).to_parquet(
        path_dir / "shadow_meta_path_dynamics_logging_v2_as_of_raw_state_table_v1.parquet", index=False
    )
    pd.DataFrame({"candidate_uid": ["c1"], "as_of_management_core_mfe_bps_at_anchor_v1": [1.0]}).to_parquet(
        path_dir / "shadow_meta_path_dynamics_logging_v2_policy_log_table_v1.parquet", index=False
    )
    anchor = root / "MONDAY_ANCHOR_AWARE_R6_CANONICAL_REBUILD_AND_EXISTING_FEATURE_REUSE_V1_fixture"
    _write_json(anchor / "summary_v1.json", {"decision_v1": "FIX_R5_2_RECALL_BASE_FIRST"})
    pd.DataFrame(
        [{"week_window_v1": "MONDAY_EXPECTED_REPLAY_UNIVERSE", "status_v1": "EXPECTED_DUE_TO_MONDAY_ANCHOR", "row_count_v1": 3}]
    ).to_csv(anchor / "anchor_aware_row_delta_explainer_v1.csv", index=False)
    pd.DataFrame([{"surface_v1": "entry-XGB R6 features", "exists_now_v1": True, "what_exists_v1": "AS_OF", "can_use_now_v1": True, "must_wait_v1": False, "illegal_direct_entry_v1": False, "status_v1": "REUSE_NOW"}]).to_csv(
        anchor / "entry_exit_transformer_and_pre_rl_reuse_map_v1.csv", index=False
    )
    _write_json(anchor / "r5_2_base_reconstruction_using_existing_assets_v1.json", {"recall_base_ready_v1": False})
    pd.DataFrame([{"field_name_v1": "as_of_hour_utc_v1"}]).to_csv(anchor / "existing_feature_asset_inventory_v1.csv", index=False)
    _write_json(anchor / "anchor_aware_row_universe_contract_v1.json", {"monday_expected_replay_universe_v1": {"row_count_v1": 3}})
    return path_dir, anchor


def test_existing_asset_first_guard_materializes_inventory_and_blocks_duplicates(tmp_path: Path) -> None:
    _seed_snapshot(tmp_path)
    foundation = _seed_foundation(tmp_path)
    score = _seed_score(tmp_path, foundation)
    r6 = _seed_r6(tmp_path, score)
    recall = _seed_recall(tmp_path)
    path_dir, anchor = _seed_path_and_anchor(tmp_path)
    narrow = tmp_path / "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_V1_fixture"
    _write_json(narrow / "summary_v1.json", {"raw_rows_v1": 1689})
    protector = tmp_path / "PROTECTOR_FIRST_SHADOW_EXPERIMENT_DRY_PRELAUNCH_V1_fixture"
    _write_json(protector / "summary_v1.json", {"raw_rows_v1": 1689})

    out = tmp_path / "out"
    summary = materialize(
        reports_root=tmp_path,
        output_dir=out,
        foundation_dir=foundation,
        score_dir=score,
        r6_dir=r6,
        recall_gap_dir=recall,
        path_dynamics_dir=path_dir,
        anchor_dir=anchor,
    )

    assert summary["decision_v1"] == "EXISTING_ASSET_FIRST_DUPLICATE_GUARD_ACTIVE"
    assert summary["next_action_v1"] == "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST"
    assert summary["new_surface_created_v1"] is False
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()

    inventory = pd.read_csv(out / "existing_asset_inventory_v1.csv")
    assert {"CANONICAL_REUSE", "REUSE_AS_INPUT", "DIAGNOSTIC_ONLY"}.issubset(set(inventory["status_v1"]))
    guard = json.loads((out / "no_duplicate_analysis_guard_v1.json").read_text())
    assert "1689 exact-only or other narrow surface used as R6 baseline" in guard["hard_fail_if_v1"]
    graph = json.loads((out / "canonical_r6_source_graph_v1.json").read_text())
    assert "R5.2 recall/base selection" in graph["score_layers_that_must_be_repaired_or_rebuilt_v1"]
