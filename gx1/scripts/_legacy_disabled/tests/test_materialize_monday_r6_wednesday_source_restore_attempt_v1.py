import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_r6_wednesday_source_restore_attempt_v1 import (
    OUTPUT_FILES,
    WEDNESDAY_FREEZE_DIR,
    WEDNESDAY_MANIFEST,
    WEDNESDAY_SNAPSHOT_DIR,
    WEDNESDAY_SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_restore_attempt_materializes_missing_source_lock(tmp_path: Path) -> None:
    freeze_dir = tmp_path / WEDNESDAY_SNAPSHOT_DIR / WEDNESDAY_FREEZE_DIR
    source_dir = tmp_path / "missing_source" / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"
    expected_payload = b"canonical-model"
    expected_sha = hashlib.sha256(expected_payload).hexdigest()
    _write_json(
        freeze_dir / WEDNESDAY_SUMMARY,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        },
    )
    _write_json(
        freeze_dir / WEDNESDAY_MANIFEST,
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_id_v1": "R6_CANDIDATE_04761_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
            "r6_source_dir_v1": str(source_dir),
            "hashes_v1": {
                "model_hashes_v1": [
                    {
                        "relative_path_v1": "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1/models/global_r6_runner_first/bad_risk/model.joblib",
                        "absolute_path_v1": str(source_dir / "models/global_r6_runner_first/bad_risk/model.joblib"),
                        "sha256_v1": expected_sha,
                        "byte_size_v1": len(expected_payload),
                        "hash_kind_v1": "model_hash",
                    }
                ]
            },
        },
    )
    scan_root = tmp_path / "scan"
    scan_root.mkdir()
    (scan_root / "model.joblib").write_bytes(b"wrong-model")
    archive_path = tmp_path / "backup.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(scan_root / "model.joblib", arcname="unrelated/model.joblib")

    output_dir = tmp_path / "out"
    summary = materialize(
        reports_root=tmp_path,
        output_dir=output_dir,
        scan_roots=[scan_root],
        archive_paths=[archive_path],
    )

    assert summary["decision_v1"] == "WEDNESDAY_SOURCE_ARTIFACTS_NOT_FOUND_LOCALLY"
    assert summary["next_action_v1"] == "RESTORE_WEDNESDAY_SOURCE_ARTIFACTS_FIRST"
    assert summary["training_started_v1"] is False
    assert summary["missing_hash_count_v1"] == 1
    assert summary["archive_scan_count_v1"] == 1
    for filename in OUTPUT_FILES.values():
        assert (output_dir / filename).exists()
    fs_scan = pd.read_csv(output_dir / OUTPUT_FILES["filesystem_hash_scan"])
    assert fs_scan.loc[0, "status_v1"] == "NOT_FOUND_BY_HASH"
    archive_scan = pd.read_csv(output_dir / OUTPUT_FILES["archive_member_scan"])
    assert archive_scan.loc[0, "restorable_from_archive_v1"] is False or str(archive_scan.loc[0, "restorable_from_archive_v1"]) == "False"
