"""Atomic commit manifest for one immutable model-native Entry bundle."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "entry_model_native_bundle_commit_v3"
MANIFEST_NAME = "ENTRY_MODEL_NATIVE_BUNDLE_COMMIT.json"
CORE_ARTIFACTS = (
    "MASTER_TRANSFORMER_LOCK.json",
    "bundle_metadata.json",
    "model_state_dict.pt",
)
ALLOWED_BUNDLE_KINDS = (
    "trained",
    "calibrated",
    "sizing_finalized",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_bundle_directory_noreplace(
    source: Path,
    destination: Path,
) -> None:
    """Atomically publish one verified sibling directory without replacement."""

    source_path = Path(source)
    destination_path = Path(destination)
    if (
        not source_path.is_absolute()
        or source_path.is_symlink()
        or not source_path.is_dir()
        or not destination_path.is_absolute()
        or source_path.parent != destination_path.parent
    ):
        raise RuntimeError("[ENTRY_BUNDLE_ATOMIC_PUBLISH_PATH_INVALID]")
    if destination_path.exists() or destination_path.is_symlink():
        raise RuntimeError(
            "[ENTRY_BUNDLE_IMMUTABLE_DESTINATION_EXISTS] "
            f"immutable output bundle already exists: {destination_path}"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("[ENTRY_BUNDLE_ATOMIC_PUBLISH_UNAVAILABLE]")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source_path),
        -100,
        os.fsencode(destination_path),
        1,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise RuntimeError(
                "[ENTRY_BUNDLE_IMMUTABLE_DESTINATION_EXISTS] "
                f"{destination_path}"
            )
        raise RuntimeError(
            "[ENTRY_BUNDLE_ATOMIC_PUBLISH_FAILED] "
            f"{os.strerror(code)}"
        )
    parent_fd = os.open(
        destination_path.parent,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _require_artifact_name(value: Any) -> str:
    name = str(value or "")
    if (
        not name
        or name in {".", "..", MANIFEST_NAME}
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise RuntimeError(
            f"[ENTRY_BUNDLE_COMMIT_ARTIFACT_NAME_INVALID] {name!r}"
        )
    return name


def build_bundle_commit_manifest(
    *,
    bundle_dir: Path,
    artifact_names: Sequence[str],
    bundle_kind: str,
    created_at_utc: str,
) -> dict[str, Any]:
    directory = Path(bundle_dir)
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_DIRECTORY_INVALID]")
    kind = str(bundle_kind)
    if kind not in ALLOWED_BUNDLE_KINDS:
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_KIND_INVALID]")
    try:
        created = datetime.fromisoformat(
            str(created_at_utc).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_TIMESTAMP_INVALID]") from exc
    if created.utcoffset() is None or created.utcoffset().total_seconds() != 0:
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_TIMESTAMP_INVALID]")
    names = sorted(_require_artifact_name(name) for name in artifact_names)
    if len(names) != len(set(names)) or not set(CORE_ARTIFACTS).issubset(names):
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_ARTIFACT_SET_INVALID]")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in names:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"[ENTRY_BUNDLE_COMMIT_ARTIFACT_INVALID] name={name}"
            )
        artifacts[name] = {
            "sha256": _sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "bundle_kind": kind,
        "created_at_utc": str(created_at_utc),
        "artifact_names": names,
        "artifact_names_sha256": _canonical_sha256(names),
        "artifacts": artifacts,
        "artifact_set_sha256": _canonical_sha256(artifacts),
    }
    payload["commit_sha256"] = _canonical_sha256(payload)
    return payload


def write_bundle_commit_manifest(
    *,
    bundle_dir: Path,
    artifact_names: Sequence[str],
    bundle_kind: str,
    created_at_utc: str,
) -> dict[str, Any]:
    directory = Path(bundle_dir)
    manifest = build_bundle_commit_manifest(
        bundle_dir=directory,
        artifact_names=artifact_names,
        bundle_kind=bundle_kind,
        created_at_utc=created_at_utc,
    )
    path = directory / MANIFEST_NAME
    encoded = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest


def require_bundle_commit_manifest(bundle_dir: Path) -> dict[str, Any]:
    directory = Path(bundle_dir)
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_DIRECTORY_INVALID]")
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_MANIFEST_MISSING]")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_MANIFEST_INVALID_JSON]") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_MANIFEST_INVALID]")
    data = dict(value)
    expected_keys = {
        "schema_version",
        "bundle_kind",
        "created_at_utc",
        "artifact_names",
        "artifact_names_sha256",
        "artifacts",
        "artifact_set_sha256",
        "commit_sha256",
    }
    if (
        set(data) != expected_keys
        or data["schema_version"] != SCHEMA_VERSION
        or data["bundle_kind"] not in ALLOWED_BUNDLE_KINDS
    ):
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_MANIFEST_SCHEMA_INVALID]")
    without_commit = dict(data)
    observed_commit = str(without_commit.pop("commit_sha256", ""))
    if (
        _SHA256_RE.fullmatch(observed_commit) is None
        or observed_commit != _canonical_sha256(without_commit)
    ):
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_HASH_INVALID]")
    names_raw = data["artifact_names"]
    if not isinstance(names_raw, list):
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_ARTIFACT_SET_INVALID]")
    names = [_require_artifact_name(name) for name in names_raw]
    if (
        names != sorted(set(names))
        or not set(CORE_ARTIFACTS).issubset(names)
        or data["artifact_names_sha256"] != _canonical_sha256(names)
    ):
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_ARTIFACT_SET_INVALID]")
    artifacts = data["artifacts"]
    if (
        not isinstance(artifacts, Mapping)
        or list(artifacts) != names
        or data["artifact_set_sha256"] != _canonical_sha256(artifacts)
    ):
        raise RuntimeError("[ENTRY_BUNDLE_COMMIT_ARTIFACT_BINDING_INVALID]")
    actual_names = sorted(path.name for path in directory.iterdir())
    if actual_names != sorted([*names, MANIFEST_NAME]):
        raise RuntimeError(
            "[ENTRY_BUNDLE_COMMIT_DIRECTORY_INVENTORY_MISMATCH] "
            f"declared={sorted([*names, MANIFEST_NAME])} actual={actual_names}"
        )
    for name in names:
        binding = artifacts[name]
        path = directory / name
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"sha256", "size_bytes"}
            or path.is_symlink()
            or not path.is_file()
            or isinstance(binding["size_bytes"], bool)
            or int(binding["size_bytes"]) != int(path.stat().st_size)
            or _SHA256_RE.fullmatch(str(binding["sha256"])) is None
            or str(binding["sha256"]) != _sha256_file(path)
        ):
            raise RuntimeError(
                f"[ENTRY_BUNDLE_COMMIT_ARTIFACT_MISMATCH] name={name}"
            )
    return data
