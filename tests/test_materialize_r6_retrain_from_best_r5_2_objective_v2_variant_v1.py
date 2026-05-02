import json
from pathlib import Path

import pandas as pd
import pytest

import gx1.scripts.materialize_r6_retrain_from_best_r5_2_objective_v2_variant_v1 as r6_v2
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


def _seed_v2_execution(path: Path) -> None:
    variant_dir = path / "variants" / r6_v2.BEST_VARIANT_ID
    _write_json(
        path / "best_v2_variant_downstream_r6_input_lock_v1.json",
        {
            "ready_for_downstream_r6_v1": True,
            "best_variant_id_v1": r6_v2.BEST_VARIANT_ID,
            "best_profile_id_v1": r6_v2.BEST_PROFILE_ID,
            "base_flag_for_r6_v1": r6_v2.V2_FINAL_BASE_FLAG,
            "raw_pre_veto_base_not_allowed_v1": r6_v2.V2_PRE_VETO_BASE_FLAG,
            "score_package_path_v1": str(variant_dir / "score_package_v1.parquet"),
            "prediction_view_path_v1": str(variant_dir / "prediction_view_v1.parquet"),
            "base_membership_path_v1": str(variant_dir / "base_membership_package_v1.parquet"),
            "downstream_r6_input_manifest_path_v1": str(variant_dir / "downstream_r6_input_manifest_v1.json"),
        },
    )
    _write_json(path / "summary_v1.json", {"best_variant_id_v1": r6_v2.BEST_VARIANT_ID})
    _write_json(
        variant_dir / "downstream_r6_input_manifest_v1.json",
        {
            "ready_for_downstream_r6_v1": True,
            "variant_id_v1": r6_v2.BEST_VARIANT_ID,
            "base_flag_for_r6_v1": r6_v2.V2_FINAL_BASE_FLAG,
            "raw_pre_veto_base_not_allowed_v1": r6_v2.V2_PRE_VETO_BASE_FLAG,
            "score_package_path_v1": str(variant_dir / "score_package_v1.parquet"),
        },
    )
    _write_json(variant_dir / "safety_guard_report_v1.json", {"safety_pass_v1": True})


def _seed_completed_r6_output(path: Path, *, bad_pre_veto_guard: bool = False) -> None:
    staged = path / "staged_best_r5_2_objective_v2_score_package_for_r6_v1"
    _write_json(staged / "summary_v1.json", {"contract_id_v1": r6_v2.V2_CONTRACT_ID})
    rows = []
    for idx in range(5):
        rescue = idx in {0, 1}
        v2_added = idx in {2, 3}
        raw_only_unsafe = idx == 4
        final = rescue or v2_added
        pre_veto = final or raw_only_unsafe
        if bad_pre_veto_guard:
            final = pre_veto
        rows.append(
            {
                "run_id": "W0",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "label_should_not_take_v1": not raw_only_unsafe,
                "take_was_ok_v1": raw_only_unsafe,
                "tail_10_50_mfe_v1": idx in {0, 1, 2},
                "fifty_plus_mfe_v1": raw_only_unsafe,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "batch_scope_v1": "BATCH_04" if idx == 3 else "BATCH_01",
                "r5_2_selected_candidate__block_v1": final,
                r6_v2.V2_FINAL_BASE_FLAG: final,
                r6_v2.V2_PRE_VETO_BASE_FLAG: pre_veto,
                r6_v2.V2_HARD_VETO_FLAG: raw_only_unsafe,
                "r5_2_true_rescue_base_membership_v1": rescue,
                "raw_true_base_membership_v1": pre_veto,
                "in_v3_base_v1": idx == 0,
                "v2_base_reason_v1": "ADDED_BY_BAD_RECALL" if v2_added else ("VETOED_AFTER_PRE_VETO_RECALL" if raw_only_unsafe else "RESCUE"),
                r6_v2.V2_BAD_SCORE: 0.91,
                r6_v2.V2_TAIL_SCORE: 0.72,
                r6_v2.V2_RISKY_SCORE: 0.66,
                r6_v2.V2_RUNNER_PROTECT_SCORE: 0.1,
                r6_v2.V2_AMBIGUOUS_PROTECT_SCORE: 0.1,
                r6_v2.V2_HARD_WINNER_PROTECT_SCORE: 0.1,
                R5_2_BAD_PROB: 0.91,
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
                "row_count_v1": 5,
                "block_count_v1": 4,
                "bad_blocks_v1": 4,
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


def test_r6_v2_objective_wrapper_reuses_completed_run_and_blocks_pre_veto(tmp_path: Path) -> None:
    v2_dir = tmp_path / "v2"
    output_dir = tmp_path / "out"
    _seed_v2_execution(v2_dir)
    _seed_completed_r6_output(output_dir)

    summary = r6_v2.materialize(
        reports_root=tmp_path,
        v2_execution_dir=v2_dir,
        output_dir=output_dir,
        run_r6_rebuild=True,
    )

    assert summary["r6_training_started_v1"] is True
    assert summary["best_v2_package_used_v1"] is True
    assert summary["pre_veto_or_unsafe_base_blocked_v1"] is True
    assert summary["bad_blocks_v1"] == 4
    assert summary["tail_help_v1"] == 3
    assert summary["v2_final_rows_selected_by_r6_v1"] == 2
    guard = json.loads((output_dir / r6_v2.OUTPUT_FILES["runtime_guard"]).read_text())
    assert guard["guard_pass_v1"] is True
    assert guard["required_base_flag_v1"] == r6_v2.V2_FINAL_BASE_FLAG
    pass_through = pd.read_csv(output_dir / r6_v2.OUTPUT_FILES["pass_through"])
    assert len(pass_through) == 2


def test_r6_v2_objective_wrapper_hard_fails_when_pre_veto_is_final(tmp_path: Path) -> None:
    v2_dir = tmp_path / "v2"
    output_dir = tmp_path / "out"
    _seed_v2_execution(v2_dir)
    _seed_completed_r6_output(output_dir, bad_pre_veto_guard=True)

    with pytest.raises(RuntimeError, match="runtime guard failed"):
        r6_v2.materialize(
            reports_root=tmp_path,
            v2_execution_dir=v2_dir,
            output_dir=output_dir,
            run_r6_rebuild=True,
        )
