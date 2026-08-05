from __future__ import annotations

from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PrefreezeTestSealLineageError,
    require_prefreeze_test_seal_lineage,
)
from gx1.models.entry_v10.entry_v10_bundle import (
    _require_current_prefreeze_test_seal_lineage,
)
from tests.model_native_test_seal_support import (
    write_prefreeze_test_seal_fixture,
)


DATASET_RUN_ID = "MODEL_NATIVE_TEST_SEAL_DATASET_PYTEST_V1"


def _seal(tmp_path: Path) -> tuple[Path, str, dict, set[Path]]:
    dataset_dir = (tmp_path / "dataset").resolve()
    dataset_dir.mkdir()
    manifest = dataset_dir / "xau_seq513_test.manifest.json"
    parquet = dataset_dir / "xau_seq513_test.parquet"
    seal_path, seal_sha256, lineage = write_prefreeze_test_seal_fixture(
        authority_dir=tmp_path / "rebuild_authority",
        dataset_dir=dataset_dir,
        dataset_run_id=DATASET_RUN_ID,
        manifest_path=manifest,
        manifest_sha256="a" * 64,
        parquet_path=parquet,
        parquet_sha256="b" * 64,
        rows=17,
    )
    return seal_path, seal_sha256, lineage, {manifest, parquet}


def test_seal_lineage_hashes_only_event_and_keeps_test_artifacts_opaque(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal_path, seal_sha256, expected, forbidden = _seal(tmp_path)
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(self: Path, *args, **kwargs):
        assert self not in forbidden, f"sealed TEST bytes opened: {self}"
        return original_open(self, *args, **kwargs)

    def guarded_stat(self: Path, *args, **kwargs):
        assert self not in forbidden, f"sealed TEST path statted: {self}"
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)

    observed = require_prefreeze_test_seal_lineage(
        seal_path,
        seal_sha256,
        expected_dataset_run_id=DATASET_RUN_ID,
        expected_dataset_dir=expected["dataset_dir"],
    )

    assert observed == expected
    assert observed["access_proof"] == {
        "seal_event_bytes_hash_validated": True,
        "test_dataset_bytes_read": False,
        "test_manifest_bytes_read": False,
        "test_metrics_read": False,
        "test_paths_resolved_or_statted": False,
    }


def test_seal_lineage_rejects_changed_event_bytes(tmp_path: Path) -> None:
    seal_path, seal_sha256, lineage, _ = _seal(tmp_path)
    seal_path.write_bytes(seal_path.read_bytes() + b" ")

    with pytest.raises(PrefreezeTestSealLineageError, match="sha256 mismatch"):
        require_prefreeze_test_seal_lineage(
            seal_path,
            seal_sha256,
            expected_dataset_run_id=DATASET_RUN_ID,
            expected_dataset_dir=lineage["dataset_dir"],
        )


def test_bundle_revalidates_event_and_rejects_declared_identity_split_brain(
    tmp_path: Path,
) -> None:
    _, _, lineage, _ = _seal(tmp_path)
    metadata = {
        "run_lineage": {"dataset_run_id": DATASET_RUN_ID},
        "prefreeze_test_seal_lineage": lineage,
    }
    assert _require_current_prefreeze_test_seal_lineage(metadata) == lineage

    split_brain = {**lineage, "test_manifest": dict(lineage["test_manifest"])}
    split_brain["test_manifest"]["sha256"] = "c" * 64
    metadata["prefreeze_test_seal_lineage"] = split_brain
    with pytest.raises(RuntimeError, match="EVENT_LINEAGE_MISMATCH"):
        _require_current_prefreeze_test_seal_lineage(metadata)
