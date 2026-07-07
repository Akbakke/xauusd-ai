import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_canonical_r5_2_base_rebuild_plan_v1 import (
    OUTPUT_FILES,
    R5_1_DIR_NAME,
    R5_1_SUMMARY,
    R5_2_FREEZE_DIR_NAME,
    R5_2_FREEZE_MANIFEST,
    R5_2_DIR_NAME,
    R5_2_SUMMARY,
    R5_DIR_NAME,
    R5_SUMMARY,
    R6_DIR_NAME,
    R6_SUMMARY,
    REHYDRATED_GLOB,
    RESTORE_GLOB,
    WEDNESDAY_FREEZE_DIR,
    WEDNESDAY_MANIFEST,
    WEDNESDAY_SNAPSHOT_DIR,
    WEDNESDAY_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_r5_2_base_plan_blocks_when_foundation_and_10176_freeze_are_missing(tmp_path: Path) -> None:
    freeze_dir = tmp_path / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    _write_json(
        freeze_dir / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "policy_logging_v1": {"row_count_v1": 1971},
        },
    )
    _write_json(
        freeze_dir / WEDNESDAY_MANIFEST,
        {
            "r5_2_benchmark_freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1",
            "as_of_schema_v1": {"column_count_v1": 109},
        },
    )

    restore_dir = tmp_path / f"{RESTORE_GLOB.rstrip('*')}fixture"
    _write_json(
        restore_dir / "summary_v1.json",
        {
            "canonical_hash_rows_v1": 15,
            "canonical_hash_scan_match_count_v1": 0,
            "canonical_hash_scan_missing_count_v1": 15,
        },
    )
    rehydrated_dir = tmp_path / f"{REHYDRATED_GLOB.rstrip('*')}fixture"
    _write_json(rehydrated_dir / "summary_v1.json", {"as_of_column_count_v1": 109})
    pd.DataFrame(
        [
            {"field_v1": "pred__entry_r6_bad_risk__prob_true_v1", "surface_v1": "POLICY", "status_v1": "MISSING"},
            {"field_v1": "r6_label_runner_near_miss_v1", "surface_v1": "HINDSIGHT", "status_v1": "PROXY"},
        ]
    ).to_csv(rehydrated_dir / "monday_r6_rehydration_blocked_fields_v1.csv", index=False)

    _write_json(tmp_path / R5_DIR_NAME / R5_SUMMARY, {"coverage_v1": {"entry_coverage_v1": 1852}})
    _write_json(tmp_path / R5_1_DIR_NAME / R5_1_SUMMARY, {"coverage_v1": {"entry_coverage_v1": 1852}})
    _write_json(tmp_path / R5_2_DIR_NAME / R5_2_SUMMARY, {"coverage_v1": {"entry_coverage_v1": 1852}})
    _write_json(tmp_path / R5_2_FREEZE_DIR_NAME / R5_2_FREEZE_MANIFEST, {"freeze_id_v1": "R5_2_SHADOW_FREEZE_ADBB99533B5FC91B_V1"})
    _write_json(
        tmp_path / R6_DIR_NAME / R6_SUMMARY,
        {
            "coverage_v1": {"entry_coverage_v1": 1852},
            "selected_candidate_v1": {"policy_name_v1": "R6_CANDIDATE_04789_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON"},
        },
    )

    output_dir = tmp_path / "out"
    summary = materialize(reports_root=tmp_path, restore_dir=restore_dir, rehydrated_dir=rehydrated_dir, output_dir=output_dir)

    assert summary["training_started_v1"] is False
    assert summary["decision_v1"] == "MONDAY_R5_2_BASE_REBUILD_BLOCKED_BY_MISSING_WEDNESDAY_SOURCE_HASHES"
    assert summary["next_action_v1"] == "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    assert "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE" in summary["blocked_action_v1"]
    assert "R5_FOUNDATION_FULL_1971" in summary["failed_prerequisites_v1"]
    assert "R5_2_EXPECTED_FREEZE_10176_PRESENT" in summary["failed_prerequisites_v1"]
    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    commands = pd.read_csv(output_dir / OUTPUT_FILES["command_plan"])
    assert "RUN_R5_2_CANONICAL_BASE_REBUILD" in set(commands["action_v1"])
    blocked = pd.read_csv(output_dir / OUTPUT_FILES["blocked_fields_to_resolve"])
    assert set(blocked["required_action_v1"]) == {"RESTORE_CANONICAL_SCORE_SOURCE", "RESTORE_CANONICAL_EXACT_LABEL_SOURCE"}
