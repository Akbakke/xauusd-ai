import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_wire_existing_r5_2_and_r6_assets_first_v1 import OUTPUT_FILES, materialize


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_frame() -> pd.DataFrame:
    rows = [
        {
            "run_id": "W1",
            "candidate_uid": "c1",
            "trade_uid": "t1",
            "trade_id": "1",
            "decision_timestamp": "2025-01-06T13:00:00Z",
            "split_scope_v1": "TRAIN",
            "label_should_not_take_v1": True,
            "take_was_ok_v1": False,
            "tail_10_50_mfe_v1": True,
            "fifty_plus_mfe_v1": False,
            "hundred_plus_mfe_v1": False,
            "two_hundred_plus_mfe_v1": False,
            "strongest_winner_path_v1": False,
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": False,
            "pred__entry_r5_should_not_take__prob_true_v1": 0.90,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.20,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.10,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.90,
            "r5_1_bad_blocker_score_v1": 0.90,
            "r5_1_runner_guard_score_v1": 0.10,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.50,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.10,
            "blocker_score_v1": 0.50,
            "runner_protector_score_v1": 0.10,
            "r5_selected_candidate__block_v1": True,
            "r5_1_selected_candidate__block_v1": True,
            "r5_2_selected_candidate__block_v1": True,
            "pred__entry_r6_bad_risk__prob_true_v1": 0.96,
            "pred__entry_r6_runner_protector__prob_true_v1": 0.10,
            "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.96,
            "pred__entry_r6_risky_allow__prob_true_v1": 0.97,
            "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.01,
            "selected_candidate_block_v1": True,
            "asof_runner_guard_v1": False,
        },
        {
            "run_id": "W1",
            "candidate_uid": "c2",
            "trade_uid": "t2",
            "trade_id": "2",
            "decision_timestamp": "2025-01-06T13:01:00Z",
            "split_scope_v1": "TRAIN",
            "label_should_not_take_v1": True,
            "take_was_ok_v1": False,
            "tail_10_50_mfe_v1": True,
            "fifty_plus_mfe_v1": False,
            "hundred_plus_mfe_v1": False,
            "two_hundred_plus_mfe_v1": False,
            "strongest_winner_path_v1": False,
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": False,
            "pred__entry_r5_should_not_take__prob_true_v1": 0.80,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.20,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.20,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.80,
            "r5_1_bad_blocker_score_v1": 0.80,
            "r5_1_runner_guard_score_v1": 0.20,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.30,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.30,
            "blocker_score_v1": 0.30,
            "runner_protector_score_v1": 0.30,
            "r5_selected_candidate__block_v1": False,
            "r5_1_selected_candidate__block_v1": False,
            "r5_2_selected_candidate__block_v1": False,
            "pred__entry_r6_bad_risk__prob_true_v1": 0.90,
            "pred__entry_r6_runner_protector__prob_true_v1": 0.20,
            "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.90,
            "pred__entry_r6_risky_allow__prob_true_v1": 0.20,
            "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.01,
            "selected_candidate_block_v1": False,
            "asof_runner_guard_v1": False,
        },
        {
            "run_id": "W1",
            "candidate_uid": "c3",
            "trade_uid": "t3",
            "trade_id": "3",
            "decision_timestamp": "2025-01-06T13:02:00Z",
            "split_scope_v1": "TRAIN",
            "label_should_not_take_v1": True,
            "take_was_ok_v1": False,
            "tail_10_50_mfe_v1": False,
            "fifty_plus_mfe_v1": False,
            "hundred_plus_mfe_v1": False,
            "two_hundred_plus_mfe_v1": False,
            "strongest_winner_path_v1": False,
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": False,
            "pred__entry_r5_should_not_take__prob_true_v1": 0.70,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.20,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.30,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.10,
            "r5_1_bad_blocker_score_v1": 0.70,
            "r5_1_runner_guard_score_v1": 0.30,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.30,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.30,
            "blocker_score_v1": 0.30,
            "runner_protector_score_v1": 0.30,
            "r5_selected_candidate__block_v1": False,
            "r5_1_selected_candidate__block_v1": False,
            "r5_2_selected_candidate__block_v1": False,
            "pred__entry_r6_bad_risk__prob_true_v1": 0.90,
            "pred__entry_r6_runner_protector__prob_true_v1": 0.20,
            "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.10,
            "pred__entry_r6_risky_allow__prob_true_v1": 0.20,
            "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.01,
            "selected_candidate_block_v1": False,
            "asof_runner_guard_v1": False,
        },
        {
            "run_id": "W1",
            "candidate_uid": "c4",
            "trade_uid": "t4",
            "trade_id": "4",
            "decision_timestamp": "2025-01-06T13:03:00Z",
            "split_scope_v1": "TRAIN",
            "label_should_not_take_v1": False,
            "take_was_ok_v1": True,
            "tail_10_50_mfe_v1": False,
            "fifty_plus_mfe_v1": True,
            "hundred_plus_mfe_v1": False,
            "two_hundred_plus_mfe_v1": False,
            "strongest_winner_path_v1": False,
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": True,
            "pred__entry_r5_should_not_take__prob_true_v1": 0.95,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.20,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.10,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.10,
            "r5_1_bad_blocker_score_v1": 0.95,
            "r5_1_runner_guard_score_v1": 0.10,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.30,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.30,
            "blocker_score_v1": 0.30,
            "runner_protector_score_v1": 0.30,
            "r5_selected_candidate__block_v1": True,
            "r5_1_selected_candidate__block_v1": False,
            "r5_2_selected_candidate__block_v1": False,
            "pred__entry_r6_bad_risk__prob_true_v1": 0.90,
            "pred__entry_r6_runner_protector__prob_true_v1": 0.20,
            "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.10,
            "pred__entry_r6_risky_allow__prob_true_v1": 0.20,
            "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.01,
            "selected_candidate_block_v1": False,
            "asof_runner_guard_v1": False,
        },
    ]
    return pd.DataFrame(rows)


def _seed_score_dir(root: Path, frame: pd.DataFrame) -> Path:
    out = root / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_fixture"
    _write_json(out / "summary_v1.json", {"decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED", "row_count_v1": len(frame)})
    _write_json(
        out / "score_rebuild_summary_v1.json",
        {
            "r5_2_selected_policy_v1": {
                "params_v1": {"bad_threshold_v1": 0.40563851594924927, "runner_max_v1": 0.2},
                "metrics_v1": {"bad_blocks_v1": 1, "tail_help_v1": 1},
            }
        },
    )
    frame.to_parquet(out / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "split_scope_v1",
            "pred__entry_r5_should_not_take__prob_true_v1",
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
            "pred__entry_r5_runner_protect__prob_true_v1",
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
            "r5_selected_candidate__block_v1",
        ]
    ].to_parquet(out / "monday_r5_score_prediction_view_v1.parquet", index=False)
    frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "split_scope_v1",
            "r5_1_bad_blocker_score_v1",
            "r5_1_runner_guard_score_v1",
            "r5_1_selected_candidate__block_v1",
        ]
    ].to_parquet(out / "monday_r5_1_score_prediction_view_v1.parquet", index=False)
    frame[
        [
            "run_id",
            "candidate_uid",
            "trade_uid",
            "trade_id",
            "decision_timestamp",
            "split_scope_v1",
            "pred__entry_r5_2_bad_blocker__prob_true_v1",
            "pred__entry_r5_2_runner_protector__prob_true_v1",
            "r5_2_selected_candidate__block_v1",
        ]
    ].to_parquet(out / "monday_r5_2_score_prediction_view_v1.parquet", index=False)
    return out


def _seed_r6_dir(root: Path, score_dir: Path, frame: pd.DataFrame) -> Path:
    out = root / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_fixture"
    _write_json(
        out / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER",
            "score_dir_v1": str(score_dir),
            "row_count_v1": len(frame),
            "family_grid_selected_policy_v1": {
                "policy_name_v1": "R6_CANDIDATE_fixture_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                "params_v1": {
                    "bad_threshold_v1": 0.85,
                    "runner_threshold_v1": 0.3,
                    "tail_threshold_v1": 0.85,
                    "risky_threshold_v1": 0.99,
                    "blindspot_threshold_v1": 0.7,
                    "r5_2_runner_threshold_v1": 0.74,
                    "use_r5_2_base_v1": True,
                    "hard_asof_runner_guard_v1": True,
                },
            },
        },
    )
    frame.to_parquet(out / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    frame.to_parquet(out / "monday_r6_on_foundation_scores_prediction_view_v1.parquet", index=False)
    pd.DataFrame([{"policy_name_v1": "fixture"}]).to_csv(out / "r6_family_grid_replay_v1.csv", index=False)
    return out


def _seed_recall_gap(root: Path) -> Path:
    out = root / "MONDAY_R6_RECALL_GAP_BEFORE_CANONICAL_LOCK_V1_fixture"
    _write_json(out / "recall_gap_summary_v1.json", {"decision_v1": "MONDAY_R6_RECALL_GAP_CONFIRMED_BEFORE_CANONICAL_LOCK"})
    pd.DataFrame({"candidate_uid": ["c2", "c3"]}).to_csv(out / "missed_bad_rows_v1.csv", index=False)
    pd.DataFrame({"candidate_uid": ["c2"]}).to_csv(out / "missed_tail_rows_v1.csv", index=False)
    return out


def test_wire_existing_assets_materializes_forensics_without_training_or_new_surface(tmp_path: Path) -> None:
    frame = _fixture_frame()
    score_dir = _seed_score_dir(tmp_path, frame)
    r6_dir = _seed_r6_dir(tmp_path, score_dir, frame)
    recall_dir = _seed_recall_gap(tmp_path)
    out = tmp_path / "out"

    summary = materialize(
        reports_root=tmp_path,
        output_dir=out,
        score_dir=score_dir,
        r6_dir=r6_dir,
        recall_gap_dir=recall_dir,
    )

    assert summary["decision_v1"] == "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST_COMPLETED"
    assert summary["gate_decision_v1"] == "R5_2_BASE_FLAG_TOO_RESTRICTIVE_NEEDS_CONTRACT_FIX"
    assert summary["next_action_v1"] == "FIX_R5_2_BASE_MEMBERSHIP_CONTRACT_NEXT"
    assert summary["training_started_v1"] is False
    assert summary["new_baseline_built_v1"] is False
    assert summary["new_feature_surface_built_v1"] is False
    assert summary["r6_uses_requested_score_dir_v1"] is True
    assert summary["safe_wiring_fix_implemented_v1"] is False

    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()

    audit = pd.read_csv(out / "r5_2_r6_existing_score_wiring_audit_v1.csv")
    assert not audit[(audit["audit_section_v1"] == "KEY_ALIGNMENT") & (audit["status_v1"] == "FAIL")].shape[0]
    assert set(audit[audit["audit_section_v1"] == "SCORE_COLUMN_ALIAS_CHECK"]["status_v1"]) == {"PASS"}

    forensics = pd.read_csv(out / "r5_2_base_membership_forensics_v1.csv")
    assert len(forensics) == 3
    assert forensics["score_missing_v1"].sum() == 0
    assert forensics["r5_2_base_flag_v1"].sum() == 0
    assert set(forensics["exclusion_class_v1"]) == {"SCORE_PRESENT_BUT_NOT_BASE"}

    simulations = pd.read_csv(out / "existing_asset_recovery_simulation_v1.csv")
    current = simulations[simulations["simulation_v1"] == "current_selected_policy"].iloc[0]
    recomputed = simulations[simulations["simulation_v1"] == "recompute_current_r6_ultra_policy_from_existing_columns"].iloc[0]
    assert int(current["bad_blocks_v1"]) == int(recomputed["bad_blocks_v1"])
    assert int(current["tail_help_v1"]) == int(recomputed["tail_help_v1"])
    assert simulations[simulations["simulation_v1"] == "base_membership_union_existing_r5_r5_1_r5_2_selected_flags"]["wednesday_safety_pass_v1"].iloc[0] in [False, "False"]

    fix_spec = json.loads((out / "wire_fix_candidate_spec_v1.json").read_text())
    assert fix_spec["safe_wiring_fix_proven_v1"] is False
    implementation = json.loads((out / "safe_wiring_fix_implementation_report_v1.json").read_text())
    assert implementation["implemented_code_fix_v1"] is False
