"""Fail-closed ownership and byte-identity contract for destructive cleanup.

The July 2026 cleanup incident proved that exclusion-based parent deletion is
not safe: abbreviated exclusions were accepted and protected artifacts were
destroyed.  This contract therefore permits only explicitly enumerated leaf
targets.  Exclusions are forbidden.  Every target is inventoried byte-for-byte
and every authoritative registry/launch path is protected from overlap.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import gx1.contracts.immutable_event_authority_v1 as immutable_events
from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
)


SCHEMA_VERSION = "gx1_evidence_retention_cleanup_plan_v1"
PLAN_EVENT_PREFIX = "GX1_EVIDENCE_RETENTION_CLEANUP_PLAN"
PLAN_MODE = "EXACT_TARGETS_NO_EXCLUSIONS"
REPO_ROOT = Path(__file__).resolve().parents[2]
GX1_DATA_ROOT = Path("/home/andre2/GX1_DATA")
DEFAULT_ALLOWED_ROOTS = (GX1_DATA_ROOT,)
CANONICAL_ARTIFACT_REGISTRY = REPO_ROOT / "PROJECT_STATE_artifacts.json"
CANONICAL_LAUNCH_CONTRACT = REPO_ROOT / "PROJECT_STATE_xau_direction_launch.json"
CANONICAL_DELETE_INCIDENT = REPO_ROOT / "PROJECT_STATE_entry_iql_delete_incident.json"
_VEDTAK_RE = re.compile(r"[A-Z0-9][A-Z0-9_.:-]{7,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
MAX_AUTHORITY_JSON_BYTES = 128 * 1024 * 1024
MAX_AUTHORITY_TOTAL_JSON_BYTES = 1024 * 1024 * 1024
MAX_AUTHORITY_JSON_FILES = 4096
MAX_AUTHORITY_VALUES = MAX_AUTHORITY_TOTAL_JSON_BYTES // 2
MAX_AUTHORITY_DEPTH = 64
_REFERENCE_SUFFIXES = (
    "_path", "_json", "_manifest", "_artifact", "_file", "_filename",
    "_parquet", "_npy", "_npz", "_dir", "_root",
)
_REFERENCE_KEYS = frozenset({
    "path", "json_path", "file", "filename", "manifest", "artifact",
    "parquet", "npy", "npz", "directory", "root", "dir",
})
_DIRECTORY_MANIFEST_NAMES = (
    "manifest.json", "MANIFEST.json", "PAIR_MANIFEST.json",
    "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json", "REPAIR_MANIFEST.json",
    "DATASET_BUILD_PROOF.json",
)
_PLAN_KEYS = {
    "schema_version",
    "created_utc",
    "json_path",
    "vedtak",
    "mode",
    "authority",
    "targets",
    "exclusions",
}
_AUTHORITY_KEYS = {
    "artifact_registry_json",
    "artifact_registry_sha256",
    "launch_contract_json",
    "launch_contract_sha256",
    "delete_incident_json",
    "delete_incident_sha256",
}
_TARGET_KEYS = {
    "path",
    "kind",
    "reason",
    "file_count",
    "directory_count",
    "total_bytes",
    "inventory_sha256",
    "inventory_jsonl",
    "inventory_jsonl_sha256",
}


class EvidenceRetentionError(RuntimeError):
    """Raised before destructive cleanup can touch a target."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_sha256(value: object, *, context: str) -> str:
    observed = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(observed) is None:
        raise EvidenceRetentionError(f"{context}: expected exact SHA-256")
    return observed


def _canonical_path(
    value: object,
    *,
    context: str,
    must_exist: bool,
) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise EvidenceRetentionError(f"{context}: path is empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise EvidenceRetentionError(f"{context}: path must be absolute: {raw}")
    if "..." in path.parts or any(part == ".." for part in path.parts):
        raise EvidenceRetentionError(
            f"{context}: abbreviated or parent-relative path is forbidden: {raw}"
        )
    try:
        resolved = path.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise EvidenceRetentionError(
            f"{context}: path cannot be resolved exactly: {raw}: {exc}"
        ) from exc
    if path != resolved:
        raise EvidenceRetentionError(
            f"{context}: path must already be canonical and symlink-free: {raw}"
        )
    if must_exist and not path.exists():
        raise EvidenceRetentionError(f"{context}: path does not exist: {raw}")
    if path.is_symlink():
        raise EvidenceRetentionError(f"{context}: symlink path is forbidden: {raw}")
    return path


def _strict_json(path: Path, expected_sha256: str, *, context: str) -> dict[str, Any]:
    canonical = _canonical_path(path, context=context, must_exist=True)
    if not canonical.is_file():
        raise EvidenceRetentionError(f"{context}: expected a regular JSON file")
    expected = _exact_sha256(expected_sha256, context=f"{context} hash")
    payload, observed, _ = _authority_json(canonical, byte_limit=MAX_AUTHORITY_JSON_BYTES)
    if observed != expected:
        raise EvidenceRetentionError(
            f"{context}: SHA-256 mismatch expected={expected} observed={observed}"
        )
    return payload


def _file_inventory_row(path: Path, *, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise EvidenceRetentionError(f"cleanup inventory contains symlink: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise EvidenceRetentionError(
            f"cleanup inventory contains non-regular filesystem entry: {path}"
        )
    digest = sha256_file(path)
    after = path.stat(follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise EvidenceRetentionError(f"cleanup inventory file changed while hashing: {path}")
    relative = "." if path == root else path.relative_to(root).as_posix()
    return {
        "relative_path": relative,
        "kind": "file",
        "device": int(after.st_dev),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _mount_points() -> frozenset[Path]:
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceRetentionError(
            "cannot prove mount boundaries from /proc/self/mountinfo"
        ) from exc
    points: set[Path] = set()
    for line in lines:
        fields = line.split(" - ", 1)[0].split()
        if len(fields) < 5:
            raise EvidenceRetentionError("malformed /proc/self/mountinfo row")
        raw = fields[4]
        for encoded, decoded in (
            (r"\040", " "),
            (r"\011", "\t"),
            (r"\012", "\n"),
            (r"\134", "\\"),
        ):
            raw = raw.replace(encoded, decoded)
        points.add(Path(raw))
    return frozenset(points)


def _inventory_with_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a deterministic byte/topology inventory and its canonical rows."""

    root = _canonical_path(path, context="cleanup target", must_exist=True)
    root_stat = root.stat(follow_symlinks=False)
    root_device = int(root_stat.st_dev)
    mount_points = _mount_points()
    if root in mount_points:
        raise EvidenceRetentionError(
            f"cleanup target cannot be a filesystem mount point: {root}"
        )
    if root.is_file():
        kind = "file"
        rows = [_file_inventory_row(root, root=root)]
    elif root.is_dir():
        kind = "directory"
        rows = [
            {
                "relative_path": ".",
                "kind": "directory",
                "device": root_device,
            }
        ]
        def inventory_directory(directory: Path) -> None:
            try:
                with os.scandir(directory) as scan:
                    entries = sorted(scan, key=lambda entry: entry.name)
            except OSError as exc:
                raise EvidenceRetentionError(
                    f"cannot enumerate cleanup target directory: {directory}: {exc}"
                ) from exc
            for entry in entries:
                child = directory / entry.name
                if entry.is_symlink():
                    raise EvidenceRetentionError(
                        f"cleanup target contains symlink and cannot be deleted: {child}"
                    )
                child_stat = entry.stat(follow_symlinks=False)
                if int(child_stat.st_dev) != root_device:
                    raise EvidenceRetentionError(
                        f"cleanup target crosses a filesystem boundary: {child}"
                    )
                if stat.S_ISDIR(child_stat.st_mode):
                    if child in mount_points:
                        raise EvidenceRetentionError(
                            f"cleanup target contains a mount point: {child}"
                        )
                    rows.append(
                        {
                            "relative_path": child.relative_to(root).as_posix(),
                            "kind": "directory",
                            "device": root_device,
                        }
                    )
                    inventory_directory(child)
                    continue
                rows.append(_file_inventory_row(child, root=root))

        inventory_directory(root)
    else:
        raise EvidenceRetentionError(f"cleanup target type is unsupported: {root}")
    aggregate_digest = hashlib.sha256()
    manifest_digest = hashlib.sha256()
    aggregate_digest.update(b"[")
    for index, row in enumerate(rows):
        encoded_row = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if index:
            aggregate_digest.update(b",")
        aggregate_digest.update(encoded_row)
        manifest_digest.update(encoded_row)
        manifest_digest.update(b"\n")
    aggregate_digest.update(b"]")
    summary = {
        "path": str(root),
        "kind": kind,
        "file_count": sum(row["kind"] == "file" for row in rows),
        "directory_count": sum(row["kind"] == "directory" for row in rows),
        "total_bytes": sum(
            int(row.get("size_bytes", 0)) for row in rows if row["kind"] == "file"
        ),
        "inventory_sha256": aggregate_digest.hexdigest(),
        "manifest_sha256": manifest_digest.hexdigest(),
    }
    return summary, rows


def inventory_path(path: Path) -> dict[str, Any]:
    """Return a deterministic byte/topology inventory for one exact path."""

    summary, _ = _inventory_with_rows(path)
    return summary


def write_inventory_manifest(target: Path, manifest_jsonl: Path) -> dict[str, Any]:
    """Publish a no-replace, fsynced per-entry inventory for one target."""

    summary, rows = _inventory_with_rows(target)
    manifest = _canonical_path(
        manifest_jsonl,
        context="cleanup inventory manifest",
        must_exist=False,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _canonical_path(
        manifest.parent,
        context="cleanup inventory manifest directory",
        must_exist=True,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(manifest, flags, 0o644)
    except FileExistsError as exc:
        raise EvidenceRetentionError(
            f"cleanup inventory manifest already exists: {manifest}"
        ) from exc
    digest = hashlib.sha256()
    try:
        for row in rows:
            encoded = json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8") + b"\n"
            digest.update(encoded)
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(f"short inventory write: {manifest}")
                view = view[written:]
        os.fsync(fd)
    except Exception:
        try:
            manifest.unlink(missing_ok=True)
        finally:
            raise
    finally:
        os.close(fd)
    directory_fd = os.open(
        manifest.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    observed = digest.hexdigest()
    if observed != summary["manifest_sha256"]:
        raise EvidenceRetentionError("cleanup inventory manifest encoding mismatch")
    return {
        **summary,
        "inventory_jsonl": str(manifest),
        "inventory_jsonl_sha256": observed,
    }


def _authority_json(path: Path, *, byte_limit: int) -> tuple[dict[str, Any], str, int]:
    """Read bounded metadata through symlink-free directory descriptors only."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number {token}")
        return value

    descriptor = None
    try:
        descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        for part in path.parts[1:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        child = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=descriptor,
        )
        os.close(descriptor)
        descriptor = child
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceRetentionError(f"authority JSON is not a regular file: {path}")
        if before.st_size > byte_limit:
            raise EvidenceRetentionError(f"authority JSON byte limit exceeded: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            encoded = handle.read(byte_limit + 1)
            after = os.fstat(handle.fileno())
        if len(encoded) > byte_limit:
            raise EvidenceRetentionError(f"authority JSON byte limit exceeded: {path}")
        def identity(entry: os.stat_result) -> tuple[int, ...]:
            return (
                entry.st_dev, entry.st_ino, entry.st_size,
                entry.st_mtime_ns, entry.st_ctime_ns,
            )
        if identity(before) != identity(after) or identity(after) != identity(path.lstat()):
            raise EvidenceRetentionError(f"authority JSON changed during read: {path}")
        _canonical_path(path, context="authority JSON", must_exist=True)
        payload = json.loads(
            encoded.decode("utf-8"), object_pairs_hook=unique_object,
            parse_float=finite_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise EvidenceRetentionError(f"invalid authority JSON: {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise EvidenceRetentionError(f"authority JSON root must be an object: {path}")
    return payload, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _reference_hash(payload: Mapping[str, Any], key: str) -> str | None:
    candidates = [key + "_sha256"]
    for suffix in ("_path", "_json", "_artifact", "_file", "_filename"):
        if key.endswith(suffix):
            candidates.append(key[:-len(suffix)] + "_sha256")
    if key in _REFERENCE_KEYS:
        candidates.append("sha256")
    for name in candidates:
        if name in payload and not isinstance(payload[name], str):
            if payload[name] is not None or payload.get(key) is not None:
                raise EvidenceRetentionError(f"authority {key} hash: expected exact SHA-256")
    hashes = {
        _exact_sha256(payload[name], context=f"authority {key} hash")
        for name in candidates if name in payload and payload[name] is not None
    }
    if len(hashes) > 1:
        raise EvidenceRetentionError(f"conflicting authority hashes for {key}")
    return next(iter(hashes), None)


def _authority_reference_path(raw: str, *, manifest: Path) -> Path:
    if not raw or raw != raw.strip() or "\x00" in raw or "://" in raw or raw.startswith("~"):
        raise EvidenceRetentionError(f"invalid authority reference: {raw!r}")
    reference = Path(raw)
    if not reference.is_absolute():
        base = manifest.parent
        parts = list(reference.parts)
        while parts and parts[0] == "..":
            base = base.parent
            parts.pop(0)
        reference = base.joinpath(*parts)
    return _canonical_path(reference, context="authority-protected path", must_exist=False)


def _semantic_authority_mapping(value: Mapping[str, Any], *, key: str) -> bool:
    if key == "target_contract":
        from gx1.contracts.entry_causal_m1_outcomes_v1 import causal_m1_target_contract
        from gx1.contracts.entry_causal_m1_target_policy_v1 import (
            train_feature_ranking_target_contract,
        )

        expected = causal_m1_target_contract()
        if value != expected:
            expected = train_feature_ranking_target_contract()
    elif key == "blocked_head_reasons":
        from gx1.contracts.entry_model_native_readiness_v1 import model_native_blocked_head_reasons

        expected = model_native_blocked_head_reasons()
    elif key == "recommended_fusion":
        from gx1.features.entry_specialist_feature_groups_v1 import model_native_recommended_fusion_metadata

        expected = model_native_recommended_fusion_metadata()
    else:
        return False
    return value == expected and json.dumps(
        value, sort_keys=True, allow_nan=False,
    ) == json.dumps(expected, sort_keys=True, allow_nan=False)


def _native_authority_reference_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "producer_source_files" not in payload:
        return payload
    from gx1.contracts.xau_tape_provenance_v1 import (
        CANONICAL_NATIVE_SOURCE_SCHEMA,
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
        require_native_producer_source_inventory_metadata,
    )

    if payload.get("schema_version") not in (
        CANONICAL_NATIVE_SOURCE_SCHEMA, CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    ):
        return payload
    try:
        inventory = require_native_producer_source_inventory_metadata(payload)
    except (RuntimeError, ValueError) as exc:
        raise EvidenceRetentionError(f"native producer source metadata invalid: {exc}") from exc
    observed = dict(payload)
    del observed["source_endpoint"]
    observed["producer_source_files"] = [
        {
            "path": record["snapshot_relative_path"],
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for record in inventory
    ]
    return observed


def _authority_event_prefix(path: Path) -> str | None:
    """Recognize events with the owner's grammar without inferring scope roots."""

    if path.suffix.lower() != ".json":
        return None
    try:
        prefix, _ = path.stem.rsplit("_", 1)
        immutable_events._require_event_prefix(prefix)
        immutable_events._event_time_from_name(path, event_prefix=prefix)
    except (ValueError, ImmutableEventAuthorityError) as exc:
        witness = immutable_events._order_path(path)
        if witness.exists() or witness.is_symlink():
            raise EvidenceRetentionError(
                f"publication witness has an invalid event filename: {path}"
            ) from exc
        return None
    return prefix


def _authority_event_inventory(
    path: Path,
    prefix: str,
    *,
    authority_root: Path | None = None,
    scope_dir_glob: str | None = None,
    max_total_bytes: int | None = None,
    max_events: int | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    byte_budget = MAX_AUTHORITY_TOTAL_JSON_BYTES if max_total_bytes is None else max_total_bytes
    event_budget = MAX_AUTHORITY_JSON_FILES if max_events is None else max_events
    current = _canonical_path(path, context="publication authority", must_exist=True)
    if current.stat(follow_symlinks=False).st_size > MAX_AUTHORITY_JSON_BYTES:
        raise EvidenceRetentionError("publication authority JSON byte limit exceeded")
    try:
        inventory = immutable_events.validated_immutable_event_authority_inventory(
            path, prefix, authority_root=authority_root, scope_dir_glob=scope_dir_glob,
            max_document_bytes=MAX_AUTHORITY_JSON_BYTES,
            max_total_bytes=byte_budget, max_events=event_budget,
        )
    except (ImmutableEventAuthorityError, OSError, ValueError, RecursionError) as exc:
        raise EvidenceRetentionError(f"invalid publication authority: {path}: {exc}") from exc
    if inventory.get("schema_version") != immutable_events.INVENTORY_SCHEMA_VERSION:
        raise EvidenceRetentionError("publication authority inventory schema is invalid")
    event_paths = inventory["event_paths"]
    witness_paths = inventory["witness_paths"]
    if (
        len(event_paths) > event_budget
        or len(witness_paths) > event_budget
    ):
        raise EvidenceRetentionError("publication authority file limit exceeded")
    total_bytes = 0
    for raw_path in (inventory["authority_root"], *event_paths, *witness_paths):
        observed = _canonical_path(raw_path, context="publication authority", must_exist=True)
        if raw_path == inventory["authority_root"]:
            continue
        size = observed.stat(follow_symlinks=False).st_size
        if size > MAX_AUTHORITY_JSON_BYTES:
            raise EvidenceRetentionError("publication authority JSON byte limit exceeded")
        total_bytes += size
        if total_bytes > byte_budget:
            raise EvidenceRetentionError("publication authority total byte limit exceeded")
    if usage is not None:
        usage.update(total_bytes=total_bytes, json_files=len(event_paths))
    return inventory


def authority_protected_paths(
    artifact_registry: Mapping[str, Any],
    launch_contract: Mapping[str, Any],
    delete_incident: Mapping[str, Any],
    *,
    artifact_registry_json: Path = CANONICAL_ARTIFACT_REGISTRY,
    launch_contract_json: Path = CANONICAL_LAUNCH_CONTRACT,
    delete_incident_json: Path | None = None,
    _reserved_bytes: int = 0,
    _reserved_json_files: int = 0,
) -> tuple[Path, ...]:
    """Follow declared JSON references, never recursively enumerate data trees.

    Relative artifact filenames use their declaring manifest's directory.
    Directories protect their entire subtree and must expose a known direct
    manifest; opaque directories, sealed TEST references and missing metadata
    block cleanup rather than pretend to prove their outbound dependencies.
    Binary payloads are protected, not read or hashed. One metadata document
    is decoded at a time; the per-file bound accommodates the operator-reported
    91,059,197-byte selective-edge JSON and 34,085,492-byte MTF manifest.
    The publication owner validates event scopes; their roots and witnesses
    are protected and all history events enter the metadata queue. Witnesses
    are never parsed as ordinary metadata. The scope snapshot is revalidated
    after traversal. Remaining byte/event budgets enter the publication owner
    before historical decoding, including witness bootstrap and growth checks.
    Limits are resource policy, not
    model/data-quality thresholds or proof that a real closure fits them.
    """

    if not all(isinstance(root, Mapping) for root in (
        artifact_registry, launch_contract, delete_incident,
    )):
        raise EvidenceRetentionError("authority roots must be objects")
    if artifact_registry.get("schema_version") != "gx1_artifact_selection_v2":
        raise EvidenceRetentionError("artifact registry schema_version is invalid")
    if artifact_registry.get("project") != "XAUUSD":
        raise EvidenceRetentionError("artifact registry project is not XAUUSD")
    for section in ("active", "retired", "history"):
        if not isinstance(artifact_registry.get(section), (dict, list)):
            raise EvidenceRetentionError(
                f"artifact registry {section} inventory is missing or invalid"
            )
    if launch_contract.get("project") != "XAUUSD":
        raise EvidenceRetentionError("launch contract project is not XAUUSD")
    if launch_contract.get("schema_version") != "gx1_xau_direction_launch_state_v1":
        raise EvidenceRetentionError("launch contract schema_version is invalid")
    if delete_incident.get("schema_version") != "gx1_entry_iql_delete_incident_v1":
        raise EvidenceRetentionError("delete incident schema_version is invalid")
    if delete_incident.get("project") != "XAUUSD":
        raise EvidenceRetentionError("delete incident project is not XAUUSD")
    roots = (
        (artifact_registry, artifact_registry_json),
        (
            {key: value for key, value in launch_contract.items()
             if key != "reviewed_local_runtime_exclusions"},
            launch_contract_json,
        ),
        (delete_incident, delete_incident_json or CANONICAL_DELETE_INCIDENT),
    )
    pending = deque((payload, Path(path), 0) for payload, path in roots)
    protected: set[Path] = set()
    visited: dict[Path, str] = {}
    bindings: dict[Path, str | None] = {}
    directories: set[Path] = set()
    event_scopes: dict[Path, dict[str, Any]] = {}
    scoped_events: set[Path] = set()
    scoped_roots: set[Path] = set()
    scoped_witnesses: set[Path] = set()
    document_sizes: dict[Path, int] = {}
    byte_budget = MAX_AUTHORITY_TOTAL_JSON_BYTES - _reserved_bytes
    file_budget = MAX_AUTHORITY_JSON_FILES - _reserved_json_files
    if byte_budget < 0 or file_budget < 0:
        raise EvidenceRetentionError("authority JSON reserved resource limit exceeded")
    total_bytes = 0
    values = 0
    while pending:
        payload, manifest, graph_depth = pending.popleft()
        if graph_depth > MAX_AUTHORITY_DEPTH:
            raise EvidenceRetentionError("authority graph depth limit exceeded")
        if payload is None:
            payload, observed_hash, size = _authority_json(
                manifest,
                byte_limit=min(
                    MAX_AUTHORITY_JSON_BYTES,
                    byte_budget - total_bytes,
                ),
            )
            expected_hash = bindings[manifest]
            if expected_hash is not None and observed_hash != expected_hash:
                raise EvidenceRetentionError(
                    f"authority JSON SHA-256 mismatch: {manifest}"
                )
            total_bytes += size
            document_sizes[manifest] = size
            visited[manifest] = observed_hash
            event_prefix = _authority_event_prefix(manifest)
            if event_prefix is not None and manifest not in scoped_events:
                inventory = _authority_event_inventory(
                    manifest, event_prefix,
                    max_total_bytes=byte_budget - total_bytes + size,
                    max_events=file_budget - len(visited) + 1,
                )
                event_scopes[manifest] = inventory
                scope_events = {Path(path) for path in inventory["event_paths"]}
                scoped_events.update(scope_events)
                protected.update(scope_events)
                scope_root = Path(inventory["authority_root"])
                scoped_roots.add(scope_root)
                protected.add(scope_root)
                for raw_witness in inventory["witness_paths"]:
                    witness = Path(raw_witness)
                    protected.add(witness)
                    if witness not in scoped_witnesses:
                        document_sizes[witness] = witness.stat(follow_symlinks=False).st_size
                        total_bytes += document_sizes[witness]
                        scoped_witnesses.add(witness)
                if total_bytes > byte_budget:
                    raise EvidenceRetentionError("authority JSON total byte limit exceeded")
                if len(scoped_events) > file_budget:
                    raise EvidenceRetentionError("authority JSON file limit exceeded")
                pending.append((
                    {"paths": inventory["event_paths"]}, manifest, graph_depth + 1,
                ))
        traversal_payload = _native_authority_reference_payload(payload)
        stack = [(iter((("", traversal_payload),)), {}, None, 0, False)]
        while stack:
            iterator, siblings, fixed_key, depth, inherited_seal = stack[-1]
            try:
                raw_key, value = next(iterator)
            except StopIteration:
                stack.pop()
                continue
            if fixed_key is None and not isinstance(raw_key, str):
                raise EvidenceRetentionError("authority JSON keys must be strings")
            key = raw_key if fixed_key is None else fixed_key
            sealed = inherited_seal or str(raw_key).lower() == "test"
            values += 1
            if values > MAX_AUTHORITY_VALUES or depth > MAX_AUTHORITY_DEPTH:
                raise EvidenceRetentionError("authority JSON value/depth limit exceeded")
            declared = key in _REFERENCE_KEYS or key.endswith(_REFERENCE_SUFFIXES)
            multiple = key == "paths" or key.endswith("_paths")
            if (
                declared and key not in {"manifest", "artifact"}
                and isinstance(value, (Mapping, list))
            ):
                raise EvidenceRetentionError(
                    f"authority reference {key} must be a path string"
                )
            if multiple and not isinstance(value, (Mapping, list)):
                raise EvidenceRetentionError(f"authority reference {key} must be a path inventory")
            if isinstance(value, Mapping):
                if fixed_key is None and not sealed and _semantic_authority_mapping(value, key=key):
                    continue
                stack.append((
                    iter(value.items()), value, "path" if multiple else None,
                    depth + 1, sealed,
                ))
                continue
            if isinstance(value, list):
                stack.append((
                    enumerate(value), {}, "path" if multiple else key,
                    depth + 1, sealed,
                ))
                continue
            absolute = isinstance(value, str) and value.startswith("/")
            if not declared and not absolute:
                continue
            expected_hash = (
                None if key == "json_path" and value == str(manifest)
                else _reference_hash(siblings, key)
            )
            if value is None and expected_hash is None:
                continue
            if not isinstance(value, str):
                raise EvidenceRetentionError(f"authority reference {key} must be a path string")
            lexical = Path(value)
            if (
                sealed or key.lower().startswith("test_")
                or any(part.lower() == "test" for part in lexical.parts)
                or re.match(r"test[_.-]", lexical.name, re.IGNORECASE)
            ):
                raise EvidenceRetentionError(
                    "sealed TEST reachability requires separate authority; "
                    "no TEST path was opened"
                )
            reference = _authority_reference_path(value, manifest=manifest)
            protected.add(reference)
            if reference.suffix.lower() == ".order":
                raise EvidenceRetentionError(
                    "publication-order witnesses require explicit authority-scope "
                    f"retention and bounded history dependency proof: {reference}"
                )
            metadata = (
                reference.suffix.lower() == ".json"
                or "manifest" in key.split("_") or "json" in key.split("_")
            )
            if reference.suffix.lower() == ".jsonl":
                raise EvidenceRetentionError(
                    f"unresolved authority JSONL dependencies: {reference}"
                )
            if not metadata:
                if reference in scoped_roots:
                    continue
                if key.startswith(("out_", "deleted_")) and not reference.exists():
                    continue
                if not reference.exists():
                    raise EvidenceRetentionError(f"unresolved authority reference: {reference}")
                if not reference.is_file() and not reference.is_dir():
                    raise EvidenceRetentionError(f"unsupported authority reference: {reference}")
                if reference.is_dir() and reference not in directories:
                    directories.add(reference)
                    found = []
                    for name in _DIRECTORY_MANIFEST_NAMES:
                        candidate = reference / name
                        if candidate.exists() or candidate.is_symlink():
                            found.append(str(candidate))
                    if not found:
                        raise EvidenceRetentionError(
                            f"unresolved authority directory manifest: {reference}"
                        )
                    pending.append(({"paths": found}, manifest, graph_depth + 1))
                continue
            if reference in visited:
                if expected_hash is not None and visited[reference] != expected_hash:
                    raise EvidenceRetentionError(
                        f"authority JSON SHA-256 mismatch: {reference}"
                    )
                continue
            if reference in bindings:
                if expected_hash is not None:
                    if bindings[reference] not in (None, expected_hash):
                        raise EvidenceRetentionError(
                            f"authority JSON SHA-256 mismatch: {reference}"
                        )
                    bindings[reference] = expected_hash
                continue
            if len(bindings) >= file_budget:
                raise EvidenceRetentionError("authority JSON file limit exceeded")
            bindings[reference] = expected_hash
            pending.append((None, reference, graph_depth + 1))
        payload = None
        traversal_payload = None
        value = None
        siblings = None
    for event, expected in event_scopes.items():
        scope_bytes = sum(
            document_sizes[Path(path)]
            for path in (*expected["event_paths"], *expected["witness_paths"])
        )
        observed = _authority_event_inventory(
            event, expected["event_prefix"],
            authority_root=Path(expected["authority_root"]),
            scope_dir_glob=expected["scope_dir_glob"],
            max_total_bytes=byte_budget - total_bytes + scope_bytes,
            max_events=file_budget - len(visited) + len(expected["event_paths"]),
        )
        if observed != expected:
            raise EvidenceRetentionError("publication authority changed during traversal")
    return tuple(sorted(protected, key=lambda item: item.as_posix()))


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _allowed_target(path: Path, allowed_roots: Sequence[Path]) -> None:
    canonical_roots = tuple(
        _canonical_path(root, context="cleanup allowed root", must_exist=True)
        for root in allowed_roots
    )
    if path in canonical_roots:
        raise EvidenceRetentionError(f"cleanup target cannot equal an allowed root: {path}")
    if not any(path.is_relative_to(root) for root in canonical_roots):
        raise EvidenceRetentionError(
            f"cleanup target is outside the allowed roots: {path}"
        )


def _validate_vedtak(value: object) -> str:
    vedtak = str(value or "").strip()
    if _VEDTAK_RE.fullmatch(vedtak) is None:
        raise EvidenceRetentionError(
            "cleanup vedtak must be an explicit 8-128 character uppercase identifier"
        )
    return vedtak


def build_cleanup_plan_payload(
    *,
    targets: Sequence[Path],
    reason: str,
    vedtak: str,
    artifact_registry_json: Path,
    launch_contract_json: Path,
    delete_incident_json: Path | None = None,
    inventory_dir: Path,
    created_utc: str,
    allowed_roots: Sequence[Path] = DEFAULT_ALLOWED_ROOTS,
) -> dict[str, Any]:
    """Build one exact-target plan and its immutable per-entry inventories."""

    _validate_vedtak(vedtak)
    if not isinstance(reason, str) or len(reason.strip()) < 12:
        raise EvidenceRetentionError("cleanup reason must be explicit and non-trivial")
    try:
        created = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise EvidenceRetentionError("cleanup created_utc is invalid") from exc
    if created.tzinfo is None:
        raise EvidenceRetentionError("cleanup created_utc must be timezone-aware")
    registry = _canonical_path(
        artifact_registry_json,
        context="artifact registry",
        must_exist=True,
    )
    launch = _canonical_path(
        launch_contract_json,
        context="launch contract",
        must_exist=True,
    )
    incident = _canonical_path(
        delete_incident_json or CANONICAL_DELETE_INCIDENT,
        context="delete incident",
        must_exist=True,
    )
    if not targets:
        raise EvidenceRetentionError("cleanup plan has no targets")
    registry_payload, registry_sha, _ = _authority_json(
        registry, byte_limit=MAX_AUTHORITY_JSON_BYTES,
    )
    launch_payload, launch_sha, _ = _authority_json(
        launch, byte_limit=MAX_AUTHORITY_JSON_BYTES,
    )
    incident_payload, incident_sha, _ = _authority_json(
        incident, byte_limit=MAX_AUTHORITY_JSON_BYTES,
    )
    protected = set(
        authority_protected_paths(
            registry_payload,
            launch_payload,
            incident_payload,
            artifact_registry_json=registry,
            launch_contract_json=launch,
            delete_incident_json=incident,
        )
    )
    protected.update((registry, launch, incident))
    manifest_root = _canonical_path(
        inventory_dir,
        context="cleanup inventory directory",
        must_exist=False,
    )
    seen: list[Path] = []
    for raw_target in targets:
        target = _canonical_path(raw_target, context="cleanup target", must_exist=True)
        _allowed_target(target, allowed_roots)
        if any(_paths_overlap(target, prior) for prior in seen):
            raise EvidenceRetentionError(
                f"cleanup targets overlap and must be exact disjoint leaves: {target}"
            )
        seen.append(target)
        overlapping = [path for path in protected if _paths_overlap(target, path)]
        if overlapping:
            raise EvidenceRetentionError(
                "cleanup target overlaps authority-protected path(s): "
                + ", ".join(str(path) for path in sorted(overlapping))
            )
        if _paths_overlap(target, manifest_root):
            raise EvidenceRetentionError(
                f"cleanup inventory directory overlaps target: {manifest_root} vs {target}"
            )

    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    target_rows: list[dict[str, Any]] = []
    written_manifests: list[Path] = []
    try:
        for index, target in enumerate(seen):
            manifest = (
                manifest_root
                / f"GX1_EVIDENCE_CLEANUP_INVENTORY_{stamp}_{index:04d}.jsonl"
            )
            inventory = write_inventory_manifest(target, manifest)
            written_manifests.append(manifest)
            target_rows.append(
                {
                    "path": str(target),
                    "kind": inventory["kind"],
                    "reason": reason.strip(),
                    "file_count": inventory["file_count"],
                    "directory_count": inventory["directory_count"],
                    "total_bytes": inventory["total_bytes"],
                    "inventory_sha256": inventory["inventory_sha256"],
                    "inventory_jsonl": inventory["inventory_jsonl"],
                    "inventory_jsonl_sha256": inventory[
                        "inventory_jsonl_sha256"
                    ],
                }
            )
    except Exception:
        for manifest in written_manifests:
            manifest.unlink(missing_ok=True)
        raise
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "vedtak": vedtak,
        "mode": PLAN_MODE,
        "authority": {
            "artifact_registry_json": str(registry),
            "artifact_registry_sha256": registry_sha,
            "launch_contract_json": str(launch),
            "launch_contract_sha256": launch_sha,
            "delete_incident_json": str(incident),
            "delete_incident_sha256": incident_sha,
        },
        "targets": target_rows,
        "exclusions": [],
    }


def validate_cleanup_plan(
    plan_json: Path,
    plan_sha256: str,
    *,
    vedtak: str,
    allowed_roots: Sequence[Path] = DEFAULT_ALLOWED_ROOTS,
    required_artifact_registry_json: Path = CANONICAL_ARTIFACT_REGISTRY,
    required_launch_contract_json: Path = CANONICAL_LAUNCH_CONTRACT,
    required_delete_incident_json: Path | None = None,
    verify_target_bytes: bool = True,
    require_targets_exist: bool = True,
) -> dict[str, Any]:
    """Validate exact authority and target bytes immediately before deletion."""

    plan_path = _canonical_path(plan_json, context="cleanup plan", must_exist=True)
    plan_hash = _exact_sha256(plan_sha256, context="cleanup plan hash")
    plan = _strict_json(plan_path, plan_hash, context="cleanup plan")
    plan_usage: dict[str, int] = {}
    try:
        plan_scope = _authority_event_inventory(
            plan_path, PLAN_EVENT_PREFIX, authority_root=plan_path.parent,
            usage=plan_usage,
        )
        if plan_scope["selected_event_path"] != str(plan_path):
            raise EvidenceRetentionError("requested cleanup plan is not the selected publication")
    except EvidenceRetentionError as exc:
        raise EvidenceRetentionError(
            f"cleanup plan is not newest immutable authority: {exc}"
        ) from exc
    if set(plan) != _PLAN_KEYS:
        raise EvidenceRetentionError(
            f"cleanup plan keys must be exact: observed={sorted(plan)}"
        )
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("mode") != PLAN_MODE:
        raise EvidenceRetentionError("cleanup plan schema or exact-target mode is invalid")
    expected_vedtak = _validate_vedtak(vedtak)
    if plan.get("vedtak") != expected_vedtak:
        raise EvidenceRetentionError("cleanup plan vedtak does not match invocation")
    exclusions = plan.get("exclusions")
    if exclusions != []:
        raise EvidenceRetentionError(
            "cleanup exclusions are forbidden; enumerate exact disjoint leaf targets"
        )
    authority = plan.get("authority")
    if not isinstance(authority, dict) or set(authority) != _AUTHORITY_KEYS:
        raise EvidenceRetentionError("cleanup plan authority binding is invalid")
    registry_path = _canonical_path(
        authority["artifact_registry_json"],
        context="artifact registry",
        must_exist=True,
    )
    launch_path = _canonical_path(
        authority["launch_contract_json"],
        context="launch contract",
        must_exist=True,
    )
    incident_path = _canonical_path(
        authority["delete_incident_json"],
        context="delete incident",
        must_exist=True,
    )
    required_registry = _canonical_path(
        required_artifact_registry_json,
        context="required artifact registry",
        must_exist=True,
    )
    required_launch = _canonical_path(
        required_launch_contract_json,
        context="required launch contract",
        must_exist=True,
    )
    required_incident = _canonical_path(
        required_delete_incident_json or CANONICAL_DELETE_INCIDENT,
        context="required delete incident",
        must_exist=True,
    )
    if (
        registry_path != required_registry
        or launch_path != required_launch
        or incident_path != required_incident
    ):
        raise EvidenceRetentionError(
            "cleanup plan authority paths do not match the pinned canonical files"
        )
    registry_sha = _exact_sha256(
        authority["artifact_registry_sha256"],
        context="artifact registry hash",
    )
    launch_sha = _exact_sha256(
        authority["launch_contract_sha256"],
        context="launch contract hash",
    )
    incident_sha = _exact_sha256(
        authority["delete_incident_sha256"],
        context="delete incident hash",
    )
    registry = _strict_json(registry_path, registry_sha, context="artifact registry")
    launch = _strict_json(launch_path, launch_sha, context="launch contract")
    incident = _strict_json(incident_path, incident_sha, context="delete incident")
    protected = set(authority_protected_paths(
        registry, launch, incident,
        artifact_registry_json=registry_path,
        launch_contract_json=launch_path,
        delete_incident_json=incident_path,
        _reserved_bytes=plan_usage["total_bytes"],
        _reserved_json_files=plan_usage["json_files"],
    ))
    revalidated_plan_scope = _authority_event_inventory(
        plan_path, PLAN_EVENT_PREFIX, authority_root=plan_path.parent,
        max_total_bytes=plan_usage["total_bytes"], max_events=plan_usage["json_files"],
    )
    if revalidated_plan_scope != plan_scope:
        raise EvidenceRetentionError("cleanup plan publication authority changed during traversal")
    protected.add(Path(plan_scope["authority_root"]))
    protected.update(Path(path) for path in plan_scope["event_paths"])
    protected.update(Path(path) for path in plan_scope["witness_paths"])
    protected.update((plan_path, registry_path, launch_path, incident_path))

    target_values = plan.get("targets")
    if not isinstance(target_values, list) or not target_values:
        raise EvidenceRetentionError("cleanup plan targets must be a non-empty list")
    validated_targets: list[dict[str, Any]] = []
    target_paths: list[Path] = []
    manifest_bindings: list[tuple[Path, str]] = []
    for index, declared in enumerate(target_values):
        context = f"cleanup target[{index}]"
        if not isinstance(declared, dict) or set(declared) != _TARGET_KEYS:
            raise EvidenceRetentionError(f"{context}: keys must be exact")
        target = _canonical_path(
            declared["path"],
            context=context,
            must_exist=require_targets_exist,
        )
        _allowed_target(target, allowed_roots)
        if any(_paths_overlap(target, prior) for prior in target_paths):
            raise EvidenceRetentionError(
                f"{context}: targets overlap and are not disjoint leaves"
            )
        target_paths.append(target)
        overlapping = [
            path for path in protected if _paths_overlap(target, path)
        ]
        if overlapping:
            raise EvidenceRetentionError(
                f"{context}: target overlaps authority-protected path(s): "
                + ", ".join(str(path) for path in sorted(overlapping))
            )
        if not isinstance(declared["reason"], str) or len(declared["reason"].strip()) < 12:
            raise EvidenceRetentionError(f"{context}: reason is missing")
        manifest_path = _canonical_path(
            declared["inventory_jsonl"],
            context=f"{context} inventory manifest",
            must_exist=True,
        )
        if not manifest_path.is_file():
            raise EvidenceRetentionError(
                f"{context}: inventory manifest is not a regular file"
            )
        if _paths_overlap(target, manifest_path):
            raise EvidenceRetentionError(
                f"{context}: inventory manifest overlaps cleanup target"
            )
        manifest_sha = _exact_sha256(
            declared["inventory_jsonl_sha256"],
            context=f"{context} inventory manifest hash",
        )
        if sha256_file(manifest_path) != manifest_sha:
            raise EvidenceRetentionError(
                f"{context}: inventory manifest SHA-256 mismatch"
            )
        manifest_bindings.append((manifest_path, manifest_sha))
        summary_fields = (
            "kind",
            "file_count",
            "directory_count",
            "total_bytes",
            "inventory_sha256",
        )
        if declared["kind"] not in {"file", "directory"}:
            raise EvidenceRetentionError(f"{context}: target kind is invalid")
        for field in ("file_count", "directory_count", "total_bytes"):
            if not isinstance(declared[field], int) or declared[field] < 0:
                raise EvidenceRetentionError(f"{context}: {field} is invalid")
        _exact_sha256(
            declared["inventory_sha256"],
            context=f"{context} inventory hash",
        )
        if verify_target_bytes:
            if not require_targets_exist:
                raise EvidenceRetentionError(
                    "target-byte verification requires existing cleanup targets"
                )
            observed = inventory_path(target)
            for field in summary_fields:
                if declared[field] != observed[field]:
                    raise EvidenceRetentionError(
                        f"{context}: {field} changed; declared={declared[field]!r} "
                        f"observed={observed[field]!r}"
                    )
            if observed["manifest_sha256"] != manifest_sha:
                raise EvidenceRetentionError(
                    f"{context}: per-entry inventory manifest differs from target"
                )
        else:
            observed = {
                "path": str(target),
                **{field: declared[field] for field in summary_fields},
                "manifest_sha256": manifest_sha,
            }
        observed["inventory_jsonl"] = str(manifest_path)
        observed["inventory_jsonl_sha256"] = manifest_sha
        validated_targets.append(observed)

    if sha256_file(plan_path) != plan_hash:
        raise EvidenceRetentionError("cleanup plan changed during validation")
    if sha256_file(registry_path) != registry_sha:
        raise EvidenceRetentionError("artifact registry changed during validation")
    if sha256_file(launch_path) != launch_sha:
        raise EvidenceRetentionError("launch contract changed during validation")
    if sha256_file(incident_path) != incident_sha:
        raise EvidenceRetentionError("delete incident changed during validation")
    for manifest_path, manifest_sha in manifest_bindings:
        if sha256_file(manifest_path) != manifest_sha:
            raise EvidenceRetentionError(
                f"cleanup inventory manifest changed during validation: {manifest_path}"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_json": str(plan_path),
        "plan_sha256": plan_hash,
        "vedtak": expected_vedtak,
        "mode": PLAN_MODE,
        "authority": dict(authority),
        "protected_path_count": len(protected),
        "targets": validated_targets,
        "target_bytes_verified": verify_target_bytes,
        "validated": True,
    }
