import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_rebuild_canonical_r5_2_base_and_r6_from_wednesday_contract_v1 import (
    REQUIRED_OUTPUTS,
    WEDNESDAY_FREEZE_DIR,
    WEDNESDAY_MANIFEST,
    WEDNESDAY_SNAPSHOT_DIR,
    WEDNESDAY_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_wednesday_snapshot(root: Path) -> Path:
    freeze_dir = root / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    _write_json(
        freeze_dir / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "policy_logging_v1": {"row_count_v1": 1971, "hindsight_backfill_rows_v1": 1971},
            "selected_candidate_v1": {
                "true_block_should_not_take_count_v1": 180,
                "true_block_tail_10_50_count_v1": 149,
                "precision_v1": 0.972972972972973,
                "worst_loso_precision_v1": 0.9285714285714286,
                "repaired_165_damage_count_v1": 0,
                "fifty_plus_mfe_block_count_v1": 1,
                "hundred_plus_mfe_block_count_v1": 0,
                "two_hundred_plus_mfe_block_count_v1": 0,
                "strongest_winner_damage_count_v1": 0,
            },
        },
    )
    _write_json(
        freeze_dir / WEDNESDAY_MANIFEST,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "r6_source_dir_v1": "/missing/r6",
            "r5_2_benchmark_freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1",
            "as_of_schema_v1": {"column_count_v1": 109, "schema_sha256_v1": "asof"},
            "hindsight_schema_v1": {"column_count_v1": 58, "schema_sha256_v1": "hindsight"},
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
    return freeze_dir


def _seed_restore(root: Path) -> Path:
    restore_dir = root / "MONDAY_R6_WEDNESDAY_SOURCE_RESTORE_ATTEMPT_V1_fixture"
    _write_json(
        restore_dir / "summary_v1.json",
        {
            "decision_v1": "WEDNESDAY_SOURCE_ARTIFACTS_NOT_FOUND_LOCALLY",
            "missing_hash_count_v1": 15,
            "expected_hash_rows_v1": 15,
            "required_source_artifact_missing_count_v1": 8,
            "archive_restorable_candidate_count_v1": 0,
        },
    )
    return restore_dir


def _seed_foundation(root: Path, rows: int) -> Path:
    foundation_dir = root / "MONDAY_R6_CANONICAL_FOUNDATION_REBUILD_V1_fixture"
    _write_json(
        foundation_dir / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R6_ACTUAL_FULLCOVERAGE_FOUNDATION_BUILT",
            "row_count_v1": rows,
            "active_rows_v1": rows - 62,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
            "foundation_as_of_output_column_count_v1": 109,
            "hindsight_output_column_count_v1": 58,
            "base_feature_count_v1": 88,
        },
    )
    return foundation_dir


def _seed_score(root: Path, foundation_dir: Path, rows: int) -> Path:
    score_dir = root / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_fixture"
    _write_json(
        score_dir / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
            "foundation_dir_v1": str(foundation_dir),
            "row_count_v1": rows,
            "active_rows_v1": rows - 62,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
            "base_feature_count_v1": 88,
            "r5_head_count_v1": 8,
            "r5_1_policy_materialized_v1": True,
            "r5_2_head_count_v1": 2,
            "r5_2_feature_count_v1": 99,
            "score_column_count_v1": 17,
            "train_rows_v1": 846,
            "validation_rows_v1": 388,
            "holdout_rows_v1": 680,
            "not_freeze_or_promo_v1": True,
        },
    )
    _write_json(score_dir / "score_rebuild_summary_v1.json", {"status_v1": "OK"})
    return score_dir


def _seed_r6(root: Path, score_dir: Path, rows: int, bad_blocks: int, tail_help: int) -> Path:
    r6_dir = root / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_fixture"
    _write_json(
        r6_dir / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER",
            "score_dir_v1": str(score_dir),
            "training_started_v1": True,
            "r6_training_started_v1": True,
            "row_count_v1": rows,
            "active_rows_v1": rows - 62,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
            "r6_head_count_v1": 5,
            "r6_feature_count_v1": 105,
            "not_freeze_or_promo_v1": True,
            "family_grid_selected_policy_v1": {
                "policy_name_v1": "R6_CANDIDATE_04504_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
                "candidate_worst_loso_v1": 1.0,
                "metrics_v1": {
                    "bad_blocks_v1": bad_blocks,
                    "tail_help_v1": tail_help,
                    "precision_v1": 1.0,
                    "repaired_165_damage_count_v1": 0,
                    "fifty_plus_mfe_block_count_v1": 1,
                    "hundred_plus_mfe_block_count_v1": 0,
                    "two_hundred_plus_mfe_block_count_v1": 0,
                    "strongest_winner_damage_count_v1": 0,
                },
            },
        },
    )
    _write_json(r6_dir / "eval_summary_v1.json", {})
    _write_json(r6_dir / "compare_against_wednesday_r6_v1.json", {})
    return r6_dir


def test_materializes_contract_driven_rebuild_lock_and_blocks_noncanonical_surfaces(tmp_path: Path) -> None:
    _seed_wednesday_snapshot(tmp_path)
    restore_dir = _seed_restore(tmp_path)
    foundation_dir = _seed_foundation(tmp_path, rows=1914)
    score_dir = _seed_score(tmp_path, foundation_dir, rows=1914)
    r6_dir = _seed_r6(tmp_path, score_dir, rows=1914, bad_blocks=64, tail_help=42)
    output_dir = tmp_path / "out"

    summary = materialize(
        reports_root=tmp_path,
        output_dir=output_dir,
        foundation_dir=foundation_dir,
        score_dir=score_dir,
        r6_dir=r6_dir,
        source_restore_dir=restore_dir,
    )

    assert summary["restore_or_rebuild_v1"] == "CONTRACT_DRIVEN_REBUILD"
    assert summary["frozen_wednesday_exact_restore_possible_v1"] is False
    assert summary["monday_foundation_rows_v1"] == 1914
    assert summary["r5_2_status_v1"] == "R5_2_REBUILT_FROM_CONTRACT"
    assert summary["r6_eval_verdict_v1"] == "MONDAY_R6_REBUILD_SAFE_BUT_BELOW_WEDNESDAY"
    assert summary["next_action_v1"] == "RESTORE_OR_RECONSTRUCT_REQUIRED_R5_2_INPUTS_FIRST"
    for filename in REQUIRED_OUTPUTS.values():
        assert (output_dir / filename).exists()

    truth = json.loads((output_dir / "rebuild_truth_and_scope_lock_v1.json").read_text())
    assert "MONDAY_EXACT_ONLY_1689_TRAINING_SURFACE" in truth["do_not_use_for_canonical_r6_v1"]
    contract = json.loads((output_dir / "wednesday_r6_contract_extraction_v1.json").read_text())
    assert contract["expected_rows_v1"] == 1971
    assert contract["thresholds_v1"]["guard_v1"] == "hard_asof_runner_guard"
    delta = pd.read_csv(output_dir / "row_and_schema_delta_explainer_v1.csv")
    assert set(delta["reason_v1"]).issuperset({"REFERENCE", "EXPECTED_MONDAY_ANCHOR_DELTA"})


def test_gate_can_mark_canonical_ready_only_when_contract_targets_pass(tmp_path: Path) -> None:
    _seed_wednesday_snapshot(tmp_path)
    restore_dir = _seed_restore(tmp_path)
    foundation_dir = _seed_foundation(tmp_path, rows=1971)
    score_dir = _seed_score(tmp_path, foundation_dir, rows=1971)
    r6_dir = _seed_r6(tmp_path, score_dir, rows=1971, bad_blocks=180, tail_help=149)
    output_dir = tmp_path / "out"

    summary = materialize(
        reports_root=tmp_path,
        output_dir=output_dir,
        foundation_dir=foundation_dir,
        score_dir=score_dir,
        r6_dir=r6_dir,
        source_restore_dir=restore_dir,
    )

    assert summary["decision_v1"] == "CANONICAL_MONDAY_R6_READY"
    gate = json.loads((output_dir / "canonical_monday_r6_gate_v1.json").read_text())
    assert gate["checks_v1"]["foundation_1971_rows_v1"] is True
    assert gate["checks_v1"]["r6_contract_eval_pass_v1"] is True
