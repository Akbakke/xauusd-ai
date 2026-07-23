from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS,
    MANIFEST_NAME,
    require_bundle_commit_manifest,
    write_bundle_commit_manifest,
)


def _stage(tmp_path: Path) -> Path:
    stage = (tmp_path / "bundle").resolve()
    stage.mkdir()
    for index, name in enumerate(CORE_ARTIFACTS):
        (stage / name).write_bytes(f"artifact-{index}".encode("utf-8"))
    return stage


def test_bundle_commit_binds_exact_inventory_bytes_and_sizes(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    written = write_bundle_commit_manifest(
        bundle_dir=stage,
        artifact_names=CORE_ARTIFACTS,
        bundle_kind="trained",
        created_at_utc="2026-07-23T12:00:00+00:00",
    )

    assert require_bundle_commit_manifest(stage) == written
    assert sorted(path.name for path in stage.iterdir()) == sorted(
        [*CORE_ARTIFACTS, MANIFEST_NAME]
    )


def test_bundle_commit_rejects_payload_tamper(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    write_bundle_commit_manifest(
        bundle_dir=stage,
        artifact_names=CORE_ARTIFACTS,
        bundle_kind="trained",
        created_at_utc="2026-07-23T12:00:00+00:00",
    )
    (stage / "model_state_dict.pt").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="ARTIFACT_MISMATCH"):
        require_bundle_commit_manifest(stage)


def test_bundle_commit_rejects_uncommitted_extra_artifact(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    write_bundle_commit_manifest(
        bundle_dir=stage,
        artifact_names=CORE_ARTIFACTS,
        bundle_kind="trained",
        created_at_utc="2026-07-23T12:00:00+00:00",
    )
    (stage / "stale_checkpoint.pt").write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="DIRECTORY_INVENTORY_MISMATCH"):
        require_bundle_commit_manifest(stage)


def test_bundle_commit_rejects_manifest_hash_tamper(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    write_bundle_commit_manifest(
        bundle_dir=stage,
        artifact_names=CORE_ARTIFACTS,
        bundle_kind="trained",
        created_at_utc="2026-07-23T12:00:00+00:00",
    )
    path = stage / MANIFEST_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bundle_kind"] = "calibrated"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="COMMIT_HASH_INVALID"):
        require_bundle_commit_manifest(stage)
