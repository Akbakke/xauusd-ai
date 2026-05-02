import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_root_cause_and_next_action_lock_v1 import (
    ROOT_CAUSE_MATRIX,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_root_cause_lock_flags_missing_r5_2_base_and_canonical_sources(tmp_path: Path) -> None:
    rebuild_dir = tmp_path / "MONDAY_R6_EXPLICIT_REBUILD_FROM_REHYDRATED_CONTRACT_V1_fixture"
    restore_dir = tmp_path / "MONDAY_R6_CANONICAL_SCORE_AND_LABEL_RESTORE_OR_REBUILD_V1_fixture"
    output_dir = tmp_path / "out"
    rebuild_dir.mkdir()
    restore_dir.mkdir()
    _write_json(
        rebuild_dir / SUMMARY,
        {
            "training_started_v1": True,
            "r6_family_grid_replay_v1": {
                "wednesday_safety_candidate_count_v1": 0,
                "zero_hard_damage_candidate_count_v1": 0,
                "max_observed_precision_v1": 0.92,
            },
        },
    )
    _write_json(
        rebuild_dir / "wednesday_locked_policy_replay_v1.json",
        {
            "r5_2_base_block_count_v1": 0,
            "wednesday_safety_pass_v1": False,
            "safety_failures_v1": ["precision_below_wednesday_r6"],
        },
    )
    pd.DataFrame(
        [
            {
                "failure_tags_v1": "FALSE_TAKE_OK_BLOCK",
                "take_was_ok_v1": True,
                "fifty_plus_mfe_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "candidate_uid": "c1",
                "run_id": "r1",
            }
        ]
    ).to_csv(rebuild_dir / "safety_failure_rows_v1.csv", index=False)
    _write_json(
        restore_dir / SUMMARY,
        {
            "decision_v1": "CANONICAL_SCORE_AND_EXACT_LABEL_SOURCES_NOT_RESTORED",
            "canonical_source_tree_present_v1": False,
            "canonical_hash_rows_v1": 15,
            "canonical_hash_scan_match_count_v1": 0,
            "canonical_hash_scan_missing_count_v1": 15,
            "expected_r5_2_freeze_found_v1": False,
            "expected_r5_2_freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1",
        },
    )

    summary = materialize(
        reports_root=tmp_path,
        rebuild_dir=rebuild_dir,
        restore_dir=restore_dir,
        output_dir=output_dir,
    )
    matrix = pd.read_csv(output_dir / ROOT_CAUSE_MATRIX)

    assert summary["decision_v1"] == "MONDAY_R6_BLOCKED_BY_MISSING_CANONICAL_R5_2_BASE_AND_SCORE_SOURCE"
    assert summary["next_action_v1"] == "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    assert summary["canonical_monday_r6_green_v1"] is False
    assert "DO_NOT_FREEZE_OR_PROMOTE_MONDAY_R6" in summary["blocked_action_v1"]
    assert matrix.set_index("check_v1").loc["WEDNESDAY_LOCKED_POLICY_R5_2_BASE_NONZERO", "status_v1"] == "FAIL"
    assert matrix.set_index("check_v1").loc["CANONICAL_WEDNESDAY_HASH_SCAN_FOUND_ALL", "status_v1"] == "FAIL"
    assert (output_dir / "next_action_lock_v1.json").exists()
