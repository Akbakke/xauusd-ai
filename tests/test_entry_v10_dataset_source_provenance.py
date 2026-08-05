from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gx1.contracts.xau_tape_provenance_v1 import XAU_INSTRUMENT
from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _bound_tape(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    root = (tmp_path / "canonical-m5").resolve()
    year_dir = root / "year=2026"
    year_dir.mkdir(parents=True)
    part = year_dir / "part-000.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-02T10:00:00Z", "2026-01-02T10:05:00Z"],
                utc=True,
            ),
            "open": [2000.0, 2001.0],
            "high": [2002.0, 2003.0],
            "low": [1999.0, 2000.0],
            "close": [2001.0, 2002.0],
        }
    ).to_parquet(part, index=False)
    manifest = root / "MANIFEST.json"
    manifest.write_text('{"fixture":"exact-source-binding"}\n', encoding="utf-8")
    provenance: dict[str, object] = {
        "schema_version": "xau_canonical_native_source_v4",
        "instrument": XAU_INSTRUMENT,
        "tape_root": str(root),
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "year_sha256": {"year=2026": _sha256(part)},
    }
    return root, provenance, part


def _load(root: Path, provenance: dict[str, object]) -> pd.DataFrame:
    return builder._load_canonical_tape(
        tape_root=root,
        tape_provenance=provenance,
        t_min=pd.Timestamp("2026-01-02T10:00:00Z"),
        t_max=pd.Timestamp("2026-01-02T10:05:00Z"),
        required_cols=["open", "high", "low", "close"],
    )


def test_exact_manifest_bound_partition_loads(tmp_path: Path) -> None:
    root, provenance, _part = _bound_tape(tmp_path)

    loaded = _load(root, provenance)

    assert loaded["time"].tolist() == list(
        pd.to_datetime(
            ["2026-01-02T10:00:00Z", "2026-01-02T10:05:00Z"],
            utc=True,
        )
    )
    assert loaded["close"].tolist() == [2001.0, 2002.0]


def test_missing_manifest_bound_partition_fails_closed(tmp_path: Path) -> None:
    root, provenance, part = _bound_tape(tmp_path)
    part.unlink()

    with pytest.raises(RuntimeError, match="TAPE_EXPECTED_PARTITION_MISSING"):
        _load(root, provenance)


def test_unrelated_parquet_cannot_replace_declared_partition(
    tmp_path: Path,
) -> None:
    root, provenance, part = _bound_tape(tmp_path)
    unrelated = root / "archive" / "arbitrary.parquet"
    unrelated.parent.mkdir()
    unrelated.write_bytes(part.read_bytes())
    part.unlink()

    with pytest.raises(RuntimeError, match="TAPE_EXPECTED_PARTITION_MISSING"):
        _load(root, provenance)


def test_extra_parquet_inside_declared_year_is_rejected(tmp_path: Path) -> None:
    root, provenance, part = _bound_tape(tmp_path)
    (part.parent / "stale.parquet").write_bytes(part.read_bytes())

    with pytest.raises(RuntimeError, match="TAPE_PARTITION_LAYOUT_INVALID"):
        _load(root, provenance)


def test_missing_git_identity_fails_without_unknown_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="DATASET_PRODUCER_GIT_IDENTITY_UNAVAILABLE",
    ):
        builder._require_producer_source_identity()

