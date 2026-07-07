import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_fix_r5_2_base_membership_contract_next_v1 import OUTPUT_FILES, materialize


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_score_dir(root: Path) -> Path:
    out = root / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_fixture"
    out.mkdir(parents=True, exist_ok=True)
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
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.42,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.10,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.30,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.10,
            "r5_1_runner_guard_score_v1": 0.10,
            "r5_2_selected_candidate__block_v1": True,
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
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.42,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.30,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.80,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.20,
            "r5_1_runner_guard_score_v1": 0.20,
            "r5_2_selected_candidate__block_v1": False,
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
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.42,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.30,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.30,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.20,
            "r5_1_runner_guard_score_v1": 0.20,
            "r5_2_selected_candidate__block_v1": False,
        },
    ]
    pd.DataFrame(rows).to_parquet(out / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    _write_json(
        out / "score_rebuild_summary_v1.json",
        {
            "r5_2_selected_policy_v1": {
                "params_v1": {"bad_threshold_v1": 0.40563851594924927, "runner_max_v1": 0.2}
            }
        },
    )
    return out


def test_base_membership_contract_fix_materializes_safe_existing_score_extension(tmp_path: Path) -> None:
    score_dir = _seed_score_dir(tmp_path)
    wire_dir = tmp_path / "WIRE_EXISTING_R5_2_AND_R6_ASSETS_FIRST_V1_fixture"
    _write_json(wire_dir / "summary_v1.json", {"gate_decision_v1": "R5_2_BASE_FLAG_TOO_RESTRICTIVE_NEEDS_CONTRACT_FIX"})
    out = tmp_path / "out"

    summary = materialize(reports_root=tmp_path, output_dir=out, score_dir=score_dir, wire_dir=wire_dir)

    assert summary["decision_v1"] == "R5_2_BASE_MEMBERSHIP_CONTRACT_FIXED_IN_CODE_READY_FOR_SCORE_REBUILD"
    assert summary["next_action_v1"] == "RUN_R5_R5_1_R5_2_SCORE_REBUILD_WITH_CONTRACT_FIX_EXPLICIT_FLAG"
    assert summary["training_started_v1"] is False
    assert summary["new_baseline_built_v1"] is False
    assert summary["new_feature_surface_built_v1"] is False
    assert summary["contract_added_rows_v1"] == 1
    assert summary["contract_added_bad_blocks_v1"] == 1
    assert summary["contract_wednesday_safety_pass_v1"] is True

    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()

    added = pd.read_csv(out / "r5_2_base_membership_contract_added_rows_v1.csv")
    assert added["candidate_uid"].tolist() == ["c2"]
    audit = pd.read_csv(out / "consistency_audit_v1.csv")
    assert audit.set_index("check_v1").loc["CONTRACT_SIMULATION_SAFETY", "status_v1"] == "PASS"
