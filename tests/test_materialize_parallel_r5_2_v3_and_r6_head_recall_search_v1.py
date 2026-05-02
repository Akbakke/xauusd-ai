import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_parallel_r5_2_v3_and_r6_head_recall_search_v1 as scan
from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
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


def _seed_r6_v2_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(5):
        selected = idx == 0
        safe_gap = idx in {1, 2}
        vetoed = idx == 3
        dangerous = idx == 4
        rows.append(
            {
                "run_id": "W0",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "split_scope_v1": "TRAIN",
                "batch_scope_v1": "BATCH_01",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": True,
                "tail_10_50_mfe_v1": idx in {0, 1},
                "peak_mfe_bps_v1": 30 if not dangerous else 120,
                "mae_abs_bps_v1": 40,
                "fifty_plus_mfe_v1": dangerous,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "r5_2_label_high_mfe_tail_risk_ambiguous_v1": False,
                "r5_2_label_runner_protect_v1": False,
                scan.V2_BAD_SCORE: 0.3 if safe_gap else 0.9,
                scan.V2_TAIL_SCORE: 0.1,
                scan.V2_RISKY_SCORE: 0.2,
                scan.V2_RUNNER_PROTECT_SCORE: 0.1,
                scan.V2_AMBIGUOUS_PROTECT_SCORE: 0.1,
                scan.V2_HARD_WINNER_PROTECT_SCORE: 0.1,
                scan.V2_PRE_VETO_BASE_FLAG: selected or vetoed,
                scan.V2_HARD_VETO_FLAG: vetoed,
                scan.V2_FINAL_BASE_FLAG: selected,
                "v2_base_reason_v1": "BASE" if selected else "NOT_BASE",
                R6_BAD_PROB: 0.1,
                R6_RISKY_PROB: 0.1,
                R6_TAIL_PROB: 0.1,
                R6_RUNNER_PROB: 0.1,
                R6_BLINDSPOT_PROB: 0.1,
                R5_2_RUNNER_PROB: 0.1,
                "selected_candidate_block_v1": False,
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
                "block_count_v1": 1,
                "bad_blocks_v1": 1,
                "tail_help_v1": 1,
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
    _write_json(path / "summary_v1.json", {"bad_blocks_v1": 1, "tail_help_v1": 1})


def test_parallel_v3_recall_search_materializes_read_only_outputs(tmp_path: Path) -> None:
    r6_dir = tmp_path / "r6"
    v2_dir = tmp_path / "v2"
    out = tmp_path / "out"
    _seed_r6_v2_dir(r6_dir)
    _write_json(
        v2_dir / "best_v2_variant_downstream_r6_input_lock_v1.json",
        {
            "best_variant_id_v1": scan.BEST_VARIANT_ID,
            "base_flag_for_r6_v1": scan.V2_FINAL_BASE_FLAG,
        },
    )

    summary = scan.materialize(reports_root=tmp_path, r6_v2_dir=r6_dir, v2_execution_dir=v2_dir, output_dir=out)

    assert summary["training_started_v1"] is False
    assert summary["r6_started_v1"] is False
    assert summary["lane_count_v1"] == 10
    assert summary["next_action_v1"] in {
        "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER",
        "IMPLEMENT_HYBRID_V3_PLUS_R6_MICRO_RECOVERY_SPEC",
        "DESIGN_R6_SAFE_OUTSIDE_BASE_RECOVERY_RUNNER",
        "STOP_RETRAIN_LOOP_AND_REVIEW_FEATURE_SIGNAL",
    }
    for filename in scan.OUTPUT_FILES.values():
        assert (out / filename).exists()
    lane01 = pd.read_csv(out / scan.OUTPUT_FILES["lane_01"])
    assert "gap_bucket_v1" in lane01.columns
