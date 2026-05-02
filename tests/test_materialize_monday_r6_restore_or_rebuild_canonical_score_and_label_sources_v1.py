import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_restore_or_rebuild_canonical_score_and_label_sources_v1 import (
    LOCAL_R5_2_FREEZE_DIR_NAME,
    LOCAL_R6_DIR_NAME,
    OUTPUT_FILES,
    R5_2_FREEZE_MANIFEST,
    R5_2_FREEZE_SUMMARY,
    R6_AS_OF_TABLE,
    R6_CONTRACT,
    R6_HINDSIGHT_TABLE,
    R6_POLICY_VIEW,
    R6_SUMMARY,
    REHYDRATED_BLOCKED_FIELDS,
    REHYDRATED_SUMMARY,
    WEDNESDAY_FREEZE_DIR,
    WEDNESDAY_MANIFEST,
    WEDNESDAY_SNAPSHOT_DIR,
    WEDNESDAY_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_restore_or_rebuild_blocks_missing_canonical_sources_and_rejects_local_1852_line(tmp_path: Path) -> None:
    freeze_dir = tmp_path / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    missing_source = tmp_path / "missing_canonical_source" / LOCAL_R6_DIR_NAME
    _write_json(
        freeze_dir / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "policy_logging_v1": {"row_count_v1": 1971, "hindsight_backfill_rows_v1": 1971},
        },
    )
    _write_json(
        freeze_dir / WEDNESDAY_MANIFEST,
        {
            "r6_source_dir_v1": str(missing_source),
            "r5_2_benchmark_freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1",
            "model_version_id_v1": "R6_ENTRY_RUNNER_FIRST_GLOBAL_FIVE_HEAD_20260422_V1",
            "as_of_schema_v1": {"column_count_v1": 109, "columns_v1": []},
            "hindsight_schema_v1": {"column_count_v1": 30, "columns_v1": []},
            "hashes_v1": {
                "model_hashes_v1": [
                    {
                        "relative_path_v1": f"{LOCAL_R6_DIR_NAME}/models/global_r6_runner_first/bad_risk/model.joblib",
                        "absolute_path_v1": str(missing_source / "models/global_r6_runner_first/bad_risk/model.joblib"),
                        "sha256_v1": "deadbeef",
                        "hash_kind_v1": "model_hash",
                    }
                ]
            },
        },
    )

    monday_dir = tmp_path / "MONDAY_R6_CANONICAL_TRUTH_V1_fixture"
    monday_dir.mkdir()
    rehydrated_dir = tmp_path / "MONDAY_R6_REHYDRATED_WEDNESDAY_CONTRACT_V1_fixture"
    _write_json(
        rehydrated_dir / REHYDRATED_SUMMARY,
        {"blocked_score_column_count_v1": 12, "hindsight_proxy_column_count_v1": 3},
    )
    pd.DataFrame(
        [
            {"field_v1": "pred__entry_r6_bad_risk__prob_true_v1", "surface_v1": "AS_OF", "status_v1": "MISSING_FILLED_NULL"},
            {"field_v1": "r6_label_runner_near_miss_v1", "surface_v1": "HINDSIGHT", "status_v1": "PROXY_NOT_EXACT"},
        ]
    ).to_csv(rehydrated_dir / REHYDRATED_BLOCKED_FIELDS, index=False)

    local_r6 = tmp_path / LOCAL_R6_DIR_NAME
    local_r6.mkdir()
    _write_json(
        local_r6 / R6_SUMMARY,
        {"selected_candidate_v1": {"policy_name_v1": "R6_CANDIDATE_04789_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON"}},
    )
    _write_json(local_r6 / R6_CONTRACT, {"freeze_benchmark_v1": {"freeze_id_v1": "R5_2_SHADOW_FREEZE_ADBB99533B5FC91B_V1"}})
    pd.DataFrame({"candidate_uid": ["c1", "c2"], "run_id": ["r", "r"]}).to_parquet(local_r6 / R6_AS_OF_TABLE, index=False)
    pd.DataFrame({"candidate_uid": ["c1", "c2"]}).to_parquet(local_r6 / R6_HINDSIGHT_TABLE, index=False)
    pd.DataFrame({"candidate_uid": ["c1", "c2"]}).to_parquet(local_r6 / R6_POLICY_VIEW, index=False)

    local_r5 = tmp_path / LOCAL_R5_2_FREEZE_DIR_NAME
    local_r5.mkdir()
    _write_json(local_r5 / R5_2_FREEZE_SUMMARY, {"freeze_id_v1": "R5_2_SHADOW_FREEZE_ADBB99533B5FC91B_V1"})
    _write_json(local_r5 / R5_2_FREEZE_MANIFEST, {"freeze_id_v1": "R5_2_SHADOW_FREEZE_ADBB99533B5FC91B_V1"})

    output_dir = tmp_path / "out"
    summary = materialize(
        reports_root=tmp_path,
        monday_truth_dir=monday_dir,
        rehydrated_dir=rehydrated_dir,
        output_dir=output_dir,
    )

    assert summary["decision_v1"] == "CANONICAL_SCORE_AND_EXACT_LABEL_SOURCES_NOT_RESTORED"
    assert summary["next_action_v1"] == "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    assert summary["training_started_v1"] is False
    assert summary["noncanonical_scores_used_v1"] is False
    assert summary["canonical_source_tree_present_v1"] is False
    assert summary["local_noncanonical_rejection_count_v1"] == 2
    assert "DO_NOT_USE_LOCAL_1852_04789_R6_AS_CANONICAL_SCORE_SOURCE" in summary["blocked_action_v1"]
    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    rejection = pd.read_csv(output_dir / OUTPUT_FILES["local_noncanonical_source_rejection"])
    assert set(rejection["assessment_v1"]) == {"REJECTED_NONCANONICAL"}
    audit = pd.read_csv(output_dir / OUTPUT_FILES["audit"])
    assert audit.set_index("check_v1").loc["NO_NONCANONICAL_SCORE_FILL", "status_v1"] == "PASS"
