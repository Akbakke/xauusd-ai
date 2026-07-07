import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_investigate_r5_2_objective_v2_or_r6_head_recall_next_v1 as investigate
from gx1.scripts.materialize_safe_true_r5_2_rescue_base_rule_v1 import RESCUE_BASE_FLAG_COL
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
    R6_BAD_PROB,
    R6_BLINDSPOT_PROB,
    R6_RISKY_PROB,
    R6_RUNNER_PROB,
    R6_TAIL_PROB,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_r6_rescue_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(5):
        selected = idx in {0, 1}
        dangerous = idx == 3
        rows.append(
            {
                "run_id": "W0",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "split_scope_v1": "HOLDOUT",
                "batch_scope_v1": "BATCH_01",
                "label_should_not_take_v1": idx < 4,
                "take_was_ok_v1": idx == 4,
                "tail_10_50_mfe_v1": idx in {0, 1, 2},
                "fifty_plus_mfe_v1": dangerous,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "r5_2_label_high_mfe_tail_risk_ambiguous_v1": dangerous,
                "r5_2_label_runner_protect_v1": False,
                RESCUE_BASE_FLAG_COL: selected,
                "in_v3_base_v1": selected,
                "raw_true_base_membership_v1": selected or dangerous,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.8,
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.2,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.1,
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.8,
                "r5_1_bad_blocker_score_v1": 0.8,
                "r5_1_runner_guard_score_v1": 0.1,
                R5_2_BAD_PROB: 0.8,
                R5_2_RUNNER_PROB: 0.1,
                R6_BAD_PROB: 0.2 if idx == 2 else 0.8,
                R6_RISKY_PROB: 0.2,
                R6_TAIL_PROB: 0.1,
                R6_RUNNER_PROB: 0.1,
                R6_BLINDSPOT_PROB: 0.1,
                "peak_mfe_bps_v1": 120.0 if dangerous else 20.0,
                "mae_abs_bps_v1": 40.0,
                "asof_runner_guard_v1": False,
            }
        )
    pd.DataFrame(rows).to_parquet(path / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "policy_name_v1": "R6_CANDIDATE_TEST_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                "family_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                "bad_threshold_v1": 0.85,
                "runner_threshold_v1": 0.3,
                "tail_threshold_v1": 0.85,
                "risky_threshold_v1": 0.85,
                "blindspot_threshold_v1": 0.7,
                "r5_2_runner_threshold_v1": 0.74,
                "use_r5_2_base_v1": True,
                "hard_asof_runner_guard_v1": True,
                "row_count_v1": 5,
                "block_count_v1": 2,
                "bad_blocks_v1": 2,
                "tail_help_v1": 2,
                "precision_v1": 1.0,
                "worst_loso_v1": 1.0,
                "false_take_ok_blocks_v1": 0,
                "fifty_plus_mfe_blocked_v1": 0,
                "hundred_plus_mfe_blocked_v1": 0,
                "two_hundred_plus_mfe_blocked_v1": 0,
                "strongest_winner_damage_v1": 0,
                "repaired_165_damage_v1": 0,
                "quarantine_blocks_v1": 0,
                "runner_near_miss_blocked_v1": 0,
                "wednesday_safety_pass_v1": True,
            }
        ]
    ).to_csv(path / "r6_family_grid_replay_v1.csv", index=False)


def test_investigation_materializes_gap_and_locks_r5_2_v2_next(tmp_path: Path) -> None:
    r6_dir = tmp_path / "r6"
    rescue_dir = tmp_path / "rescue"
    out = tmp_path / "out"
    _seed_r6_rescue_dir(r6_dir)
    _write_json(
        rescue_dir / "rescue_rule_application_audit_v1.json",
        {
            "raw_true_v1": {"bad_v1": 4, "tail_v1": 3, "fifty_plus_overlap_v1": 1},
            "rescued_v1": {"bad_v1": 2, "tail_v1": 2},
        },
    )

    summary = investigate.materialize(reports_root=tmp_path, r6_rescue_dir=r6_dir, rescue_dir=rescue_dir, output_dir=out)

    assert summary["training_started_v1"] is False
    assert summary["r6_rerun_started_v1"] is False
    assert summary["decision_v1"] == "R5_2_OBJECTIVE_V2_REBUILD"
    assert summary["next_action_v1"] == "DESIGN_R5_2_OBJECTIVE_V2_REBUILD_NEXT"
    for filename in investigate.OUTPUT_FILES.values():
        assert (out / filename).exists()
    gap = pd.read_csv(out / investigate.OUTPUT_FILES["gap_map"])
    assert not gap.empty
    audit = pd.read_csv(out / investigate.OUTPUT_FILES["audit"])
    assert set(audit["status_v1"]) == {"PASS"}
