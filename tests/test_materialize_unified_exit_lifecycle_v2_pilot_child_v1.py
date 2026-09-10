from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.materialize_unified_exit_lifecycle_v2_pilot_child_v1 import (
    materialize_pilot_child_lifecycle_v2,
)


def _child_witness(tmp_path: Path) -> dict[str, object]:
    return {
        "witness_sha256": "1" * 64,
        "child_dataset_run_id": "PILOT_CHILD",
        "m1_source_binding": {
            "parquet_path": str(tmp_path / "m1.parquet"),
            "parquet_sha256": "2" * 64,
            "manifest_path": str(tmp_path / "m1.manifest.json"),
            "manifest_sha256": "3" * 64,
        },
        "splits": {},
    }


def _patch_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    m1 = tmp_path / "m1.parquet"
    manifest = tmp_path / "m1.manifest.json"
    m1.write_bytes(b"m1")
    manifest.write_bytes(b"manifest")
    monkeypatch.setattr(
        "gx1.scripts.materialize_unified_exit_lifecycle_v2_pilot_child_v1."
        "validate_pilot_child_view",
        lambda **_kwargs: _child_witness(tmp_path),
    )
    monkeypatch.setattr(
        "gx1.scripts.materialize_unified_exit_lifecycle_v2_pilot_child_v1."
        "_validate_m1_source",
        lambda *_args: (
            pd.DatetimeIndex(pd.to_datetime(["2025-06-01T00:00:00Z"])),
            {},
            "3" * 64,
            "2" * 64,
        ),
    )
    return m1, manifest


def test_missing_economics_returns_blocked_after_child_and_m1_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m1, manifest = _patch_admission(monkeypatch, tmp_path)
    pilot = tmp_path / "pilot"
    report = materialize_pilot_child_lifecycle_v2(
        source_recipe_path=tmp_path / "source.json",
        source_recipe_sha256="0" * 64,
        pilot_root=pilot,
        child_root_path=tmp_path / "child.json",
        output_dir=pilot / "LIFECYCLE_V2",
        m1_source_path=m1,
        m1_source_manifest_path=manifest,
        train_economic_authority_path=None,
        val_economic_authority_path=None,
        publish=False,
    )
    assert report["decision"] == "BLOCKED"
    assert report["blockers"] == [
        "missing_train_economic_authority",
        "missing_val_economic_authority",
    ]
    assert report["test_accessed"] is False
    assert not pilot.exists()


def test_missing_economics_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m1, manifest = _patch_admission(monkeypatch, tmp_path)
    pilot = tmp_path / "pilot"
    with pytest.raises(RuntimeError, match="PILOT_COMPACT_PUBLISH_BLOCKED"):
        materialize_pilot_child_lifecycle_v2(
            source_recipe_path=tmp_path / "source.json",
            source_recipe_sha256="0" * 64,
            pilot_root=pilot,
            child_root_path=tmp_path / "child.json",
            output_dir=pilot / "LIFECYCLE_V2",
            m1_source_path=m1,
            m1_source_manifest_path=manifest,
            train_economic_authority_path=None,
            val_economic_authority_path=None,
            publish=True,
        )
    assert not pilot.exists()


def test_pilot_epochs_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m1, manifest = _patch_admission(monkeypatch, tmp_path)
    pilot = tmp_path / "pilot"
    with pytest.raises(RuntimeError, match="PILOT_COMPACT_INVOCATION_INVALID"):
        materialize_pilot_child_lifecycle_v2(
            source_recipe_path=tmp_path / "source.json",
            source_recipe_sha256="0" * 64,
            pilot_root=pilot,
            child_root_path=tmp_path / "child.json",
            output_dir=pilot / "LIFECYCLE_V2",
            m1_source_path=m1,
            m1_source_manifest_path=manifest,
            train_economic_authority_path=None,
            val_economic_authority_path=None,
            planned_epochs=30,
            publish=False,
        )
