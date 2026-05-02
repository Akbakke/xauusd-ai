import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_r6_retrain_from_true_r5_2_rescue_package_v1 as r6_rescue
from gx1.scripts.materialize_safe_true_r5_2_rescue_base_rule_v1 import CONTRACT_ID, RESCUE_BASE_FLAG_COL
from gx1.scripts.run_true_r5_2_rebuild_runner_v1 import BASE_FLAG_COL
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


def _seed_rescue_dir(path: Path) -> None:
    _write_json(path / "manifest_v1.json", {"input_paths_v1": {"score_path_v1": str(path / "unused.parquet")}})
    _write_json(
        path / "true_r5_2_rescue_downstream_r6_input_manifest_v1.json",
        {
            "contract_id_v1": CONTRACT_ID,
            "base_flag_for_r6_v1": RESCUE_BASE_FLAG_COL,
            "score_package_path_v1": str(path / "unused_score.parquet"),
        },
    )
    _write_json(
        path / "rescue_rule_application_audit_v1.json",
        {
            "safety_pass_v1": True,
            "raw_true_v1": {"bad_v1": 4, "tail_v1": 3, "precision_v1": 0.75, "worst_loso_v1": 0.0},
        },
    )


def _seed_existing_r6_output(path: Path) -> None:
    staged = path / "staged_true_r5_2_rescue_score_package_for_r6_v1"
    _write_json(staged / "summary_v1.json", {"contract_id_v1": CONTRACT_ID})
    rows = []
    for idx in range(4):
        in_v3 = idx in {0, 1}
        added = idx == 2
        raw_only_unsafe = idx == 3
        selected = in_v3 or added
        rows.append(
            {
                "run_id": "W0",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "label_should_not_take_v1": True,
                "take_was_ok_v1": False,
                "tail_10_50_mfe_v1": idx in {0, 1, 2},
                "fifty_plus_mfe_v1": raw_only_unsafe,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "batch_scope_v1": "BATCH_04" if idx == 2 else "BATCH_01",
                "r5_2_selected_candidate__block_v1": selected,
                RESCUE_BASE_FLAG_COL: selected,
                "in_v3_base_v1": in_v3,
                "raw_true_base_membership_v1": selected or raw_only_unsafe,
                "added_by_true_rescue_rule_v1": added,
                R5_2_BAD_PROB: 0.9,
                R5_2_RUNNER_PROB: 0.1,
                R6_BAD_PROB: 0.1,
                R6_RISKY_PROB: 0.1,
                R6_TAIL_PROB: 0.1,
                R6_RUNNER_PROB: 0.1,
                R6_BLINDSPOT_PROB: 0.1,
                "asof_runner_guard_v1": False,
                "selected_candidate_block_v1": False,
            }
        )
    pd.DataFrame(rows).to_parquet(path / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    _write_json(path / "summary_v1.json", {"layer_name": "WRAPPER_SUMMARY_ALREADY_OVERWROTE_STANDARD"})
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
                "row_count_v1": 4,
                "block_count_v1": 3,
                "bad_blocks_v1": 3,
                "tail_help_v1": 3,
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


def test_r6_rescue_wrapper_reuses_completed_run_and_blocks_raw_true(tmp_path: Path) -> None:
    rescue_dir = tmp_path / "rescue"
    output_dir = tmp_path / "out"
    v3_dir = tmp_path / "v3"
    _seed_rescue_dir(rescue_dir)
    _seed_existing_r6_output(output_dir)
    _write_json(
        v3_dir / "summary_v1.json",
        {
            "family_grid_selected_policy_v1": {
                "metrics_v1": {"bad_blocks_v1": 2, "tail_help_v1": 2, "precision_v1": 1.0},
                "candidate_worst_loso_v1": 1.0,
            }
        },
    )

    summary = r6_rescue.materialize(
        reports_root=tmp_path,
        rescue_dir=rescue_dir,
        v3_r6_dir=v3_dir,
        output_dir=output_dir,
        run_r6_rebuild=True,
    )

    assert summary["r6_training_started_v1"] is True
    assert summary["bad_blocks_v1"] == 3
    assert summary["tail_help_v1"] == 3
    assert summary["rescued_rows_retained_by_r6_v1"] == 1
    assert summary["raw_true_base_blocked_v1"] is True
    guard = json.loads((output_dir / r6_rescue.OUTPUT_FILES["runtime_guard"]).read_text())
    assert guard["guard_pass_v1"] is True
    manifest = json.loads((output_dir / r6_rescue.OUTPUT_FILES["manifest"]).read_text())
    assert manifest["input_rescue_manifest_v1"]["base_flag_for_r6_v1"] == RESCUE_BASE_FLAG_COL
    assert manifest["input_rescue_manifest_v1"]["base_flag_for_r6_v1"] != BASE_FLAG_COL
