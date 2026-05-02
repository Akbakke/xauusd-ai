import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r5_r5_1_r5_2_score_rebuild_with_contract_fix_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import R5_2_BASE_MEMBERSHIP_CONTRACT


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frame(new: bool) -> pd.DataFrame:
    rows = [
        {
            "run_id": "W1",
            "candidate_uid": "c1",
            "trade_uid": "t1",
            "trade_id": "1",
            "decision_timestamp": "2025-01-06T13:00:00Z",
            "split_scope_v1": "TRAIN",
            "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
            "label_should_not_take_v1": True,
            "tail_10_50_mfe_v1": True,
            "take_was_ok_v1": False,
            "fifty_plus_mfe_v1": False,
            "hundred_plus_mfe_v1": False,
            "two_hundred_plus_mfe_v1": False,
            "strongest_winner_path_v1": False,
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": False,
            "pred__entry_r5_should_not_take__prob_true_v1": 0.90,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.20,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.10,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.80,
            "r5_1_bad_blocker_score_v1": 0.90,
            "r5_1_runner_guard_score_v1": 0.10,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.50,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.10,
            "r5_selected_candidate__block_v1": True,
            "r5_1_selected_candidate__block_v1": True,
            "r5_2_selected_candidate__block_v1": True,
            "blocker_score_v1": 0.50,
            "runner_protector_score_v1": 0.10,
        },
        {
            "run_id": "W1",
            "candidate_uid": "c2",
            "trade_uid": "t2",
            "trade_id": "2",
            "decision_timestamp": "2025-01-06T13:01:00Z",
            "split_scope_v1": "TRAIN",
            "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
            "label_should_not_take_v1": True,
            "tail_10_50_mfe_v1": False,
            "take_was_ok_v1": False,
            "fifty_plus_mfe_v1": False,
            "hundred_plus_mfe_v1": False,
            "two_hundred_plus_mfe_v1": False,
            "strongest_winner_path_v1": False,
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": False,
            "pred__entry_r5_should_not_take__prob_true_v1": 0.80,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.80,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.20,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.20,
            "r5_1_bad_blocker_score_v1": 0.80,
            "r5_1_runner_guard_score_v1": 0.20,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.50,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.20,
            "r5_selected_candidate__block_v1": False,
            "r5_1_selected_candidate__block_v1": False,
            "r5_2_selected_candidate__block_v1": bool(new),
            "blocker_score_v1": 0.50,
            "runner_protector_score_v1": 0.20,
        },
        {
            "run_id": "W1",
            "candidate_uid": "c3",
            "trade_uid": "t3",
            "trade_id": "3",
            "decision_timestamp": "2025-01-06T13:02:00Z",
            "split_scope_v1": "TRAIN",
            "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
            "label_should_not_take_v1": False,
            "tail_10_50_mfe_v1": False,
            "take_was_ok_v1": True,
            "fifty_plus_mfe_v1": True,
            "hundred_plus_mfe_v1": False,
            "two_hundred_plus_mfe_v1": False,
            "strongest_winner_path_v1": False,
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": True,
            "pred__entry_r5_should_not_take__prob_true_v1": 0.90,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.20,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.20,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.20,
            "r5_1_bad_blocker_score_v1": 0.90,
            "r5_1_runner_guard_score_v1": 0.20,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.50,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.20,
            "r5_selected_candidate__block_v1": False,
            "r5_1_selected_candidate__block_v1": False,
            "r5_2_selected_candidate__block_v1": False,
            "blocker_score_v1": 0.50,
            "runner_protector_score_v1": 0.20,
        },
    ]
    return pd.DataFrame(rows)


def _seed_score_dir(path: Path, new: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _frame(new).to_parquet(path / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    r5_2 = {
        "params_v1": {"bad_threshold_v1": 0.40563851594924927, "runner_max_v1": 0.2},
        "metrics_v1": {"bad_blocks_v1": 2 if new else 1, "tail_help_v1": 1},
    }
    if new:
        r5_2.update(
            {
                "base_membership_contract_applied_v1": True,
                "base_membership_contract_v1": R5_2_BASE_MEMBERSHIP_CONTRACT,
                "base_membership_contract_added_rows_v1": 1,
                "base_membership_contract_added_bad_blocks_v1": 1,
                "base_membership_contract_added_tail_help_v1": 0,
                "base_metrics_before_contract_v1": {"block_count_v1": 1},
            }
        )
    _write_json(path / "score_rebuild_summary_v1.json", {"r5_2_selected_policy_v1": r5_2})
    _write_json(
        path / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
            "explicit_score_rebuild_flag_v1": bool(new),
            "r6_heads_trained_v1": False,
            "active_rows_v1": 3,
            "quarantine_rows_v1": 0,
            "as_of_column_count_v1": 109,
        },
    )


def test_rebuild_with_contract_fix_audit_passes_and_locks_r6_next_action(tmp_path: Path) -> None:
    old_dir = tmp_path / "old_score"
    new_dir = tmp_path / "new_score"
    foundation = tmp_path / "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_20260425T_FOUNDATION_LOCK_V4"
    simulation = tmp_path / "simulation"
    _seed_score_dir(old_dir, new=False)
    _seed_score_dir(new_dir, new=True)
    simulation.mkdir()
    pd.DataFrame({"candidate_uid": ["c2"]}).to_csv(simulation / "r5_2_base_membership_contract_added_rows_v1.csv", index=False)
    out = tmp_path / "out"

    summary = materialize(
        reports_root=tmp_path,
        output_dir=out,
        old_score_dir=old_dir,
        new_score_dir=new_dir,
        foundation_dir=foundation,
        simulation_dir=simulation,
    )

    assert summary["decision_v1"] == "R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_PASS"
    assert summary["next_action_v1"] == "RUN_R6_RETRAIN_FROM_FIXED_R5_2_SCORE_PACKAGE_EXPLICIT_FLAG"
    assert summary["new_bad_blocks_v1"] == 2
    assert summary["new_score_package_ready_for_r6_retrain_v1"] is True
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()

    audit = pd.read_csv(out / "r5_2_contract_application_audit_v1.csv")
    assert audit.set_index("check_v1").loc["CONTRACT_PRESENT_IN_SCORE_SUMMARY", "status_v1"] == "PASS"
    added = pd.read_csv(out / "added_base_rows_forensics_v1.csv")
    assert added["candidate_uid"].tolist() == ["c2"]
    assert added["recoverability_status_v1"].tolist() == ["SAFE_RECOVERABLE"]
    gate = json.loads((out / "r5_2_score_rebuild_gate_v1.json").read_text())
    assert gate["checks_v1"]["surface_guard_ok_v1"] is True
