"""Exact immutable dataset split identities for model-native Entry audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION = (
    "entry_dataset_split_artifacts_v1"
)
_BINDING_KEYS = frozenset(
    {"manifest_path", "manifest_sha256", "parquet_sha256"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, context: str) -> str:
    observed = str(value or "").strip().lower()
    if len(observed) != 64 or any(
        character not in "0123456789abcdef" for character in observed
    ):
        raise RuntimeError(f"{context}: expected exact SHA-256")
    return observed


def _immutable_file(
    raw_path: Any,
    *,
    dataset_dir: Path,
    suffix: str,
    context: str,
) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or path.parent != dataset_dir
        or not path.name.endswith(suffix)
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise RuntimeError(f"{context}: invalid immutable file identity: {path}")
    return path


def require_dataset_split_artifacts(
    dataset_dir: Path,
    bindings: Mapping[str, Mapping[str, Any]] | Any,
    *,
    expected_splits: Sequence[str],
    context: str,
) -> dict[str, dict[str, str]]:
    """Resolve parquets only through exact, hash-bound split manifests."""

    dataset_dir = Path(dataset_dir).expanduser()
    splits = tuple(str(split) for split in expected_splits)
    if (
        not dataset_dir.is_absolute()
        or dataset_dir.resolve() != dataset_dir
        or dataset_dir.is_symlink()
        or not dataset_dir.is_dir()
        or any("latest" in part.lower() for part in dataset_dir.parts)
    ):
        raise RuntimeError(f"{context}: dataset_dir is not immutable: {dataset_dir}")
    if not isinstance(bindings, Mapping) or tuple(bindings) != splits:
        raise RuntimeError(
            f"{context}: split binding set/order must be exactly {splits}"
        )

    resolved: dict[str, dict[str, str]] = {}
    for split in splits:
        row = bindings[split]
        if not isinstance(row, Mapping) or set(row) != _BINDING_KEYS:
            raise RuntimeError(f"{context}.{split}: binding keys are not exact")
        manifest_sha = _sha256(
            row["manifest_sha256"],
            context=f"{context}.{split}.manifest_sha256",
        )
        parquet_sha = _sha256(
            row["parquet_sha256"],
            context=f"{context}.{split}.parquet_sha256",
        )
        manifest_path = _immutable_file(
            row["manifest_path"],
            dataset_dir=dataset_dir,
            suffix=f"_{split}.manifest.json",
            context=f"{context}.{split}.manifest",
        )
        if _sha256_file(manifest_path) != manifest_sha:
            raise RuntimeError(f"{context}.{split}: manifest hash mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"{context}.{split}: manifest JSON is invalid: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError(f"{context}.{split}: manifest root must be an object")
        parquet_path = _immutable_file(
            manifest.get("output_data_path"),
            dataset_dir=dataset_dir,
            suffix=f"_{split}.parquet",
            context=f"{context}.{split}.parquet",
        )
        if _sha256_file(parquet_path) != parquet_sha:
            raise RuntimeError(f"{context}.{split}: parquet hash mismatch")
        if _sha256_file(manifest_path) != manifest_sha:
            raise RuntimeError(f"{context}.{split}: manifest changed during validation")
        resolved[split] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
        }
    return resolved
