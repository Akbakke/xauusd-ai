#!/usr/bin/env python3
"""The sole supported destructive-cleanup route for GX1 evidence paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from gx1.contracts.evidence_retention_v1 import (
    GX1_DATA_ROOT,
    PLAN_EVENT_PREFIX,
    REPO_ROOT,
    build_cleanup_plan_payload,
    inventory_path,
    sha256_file,
    validate_cleanup_plan,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)


CLEARANCE_PREFIX = "GX1_EVIDENCE_RETENTION_CLEARANCE"
APPROVAL_PREFIX = "GX1_EVIDENCE_RETENTION_APPROVAL"
STARTED_PREFIX = "GX1_EVIDENCE_CLEANUP_STARTED"
STAGED_PREFIX = "GX1_EVIDENCE_CLEANUP_STAGED"
EXECUTION_PREFIX = "GX1_EVIDENCE_CLEANUP_EXECUTION"
RECOVERY_PREFIX = "GX1_EVIDENCE_CLEANUP_RECOVERY"
DEFAULT_REGISTRY = REPO_ROOT / "PROJECT_STATE_artifacts.json"
DEFAULT_LAUNCH = REPO_ROOT / "PROJECT_STATE_xau_direction_launch.json"
DEFAULT_PLAN_DIR = GX1_DATA_ROOT / "reports/gx1_evidence_retention_cleanup_plans"
DEFAULT_APPROVAL_DIR = GX1_DATA_ROOT / "reports/gx1_evidence_retention_cleanup_approvals"
DEFAULT_REPORT_DIR = GX1_DATA_ROOT / "reports/gx1_evidence_retention_cleanup_reports"
_APPROVAL_KEYS = {
    "schema_version",
    "created_utc",
    "json_path",
    "decision",
    "plan_json",
    "plan_sha256",
    "vedtak",
    "approved_by",
    "targets",
    "direction_authority",
    "launch_authority",
}
_STARTED_KEYS = {
    "schema_version",
    "created_utc",
    "json_path",
    "decision",
    "plan_json",
    "plan_sha256",
    "approval_json",
    "approval_sha256",
    "vedtak",
    "stage_plan",
    "direction_authority",
    "launch_authority",
}
_STAGED_KEYS = _STARTED_KEYS | {"staged"}
_INVENTORY_FIELDS = (
    "kind",
    "file_count",
    "directory_count",
    "total_bytes",
    "inventory_sha256",
    "manifest_sha256",
)


def _json(value: dict[str, Any], *, quiet: bool) -> None:
    if not quiet:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _plan(args: argparse.Namespace) -> int:
    created_utc = datetime.now(timezone.utc).isoformat()
    out_dir = Path(args.out_dir).expanduser().resolve()
    payload = build_cleanup_plan_payload(
        targets=[Path(value) for value in args.target],
        reason=args.reason,
        vedtak=args.vedtak,
        artifact_registry_json=DEFAULT_REGISTRY,
        launch_contract_json=DEFAULT_LAUNCH,
        inventory_dir=out_dir,
        created_utc=created_utc,
        allowed_roots=(GX1_DATA_ROOT,),
    )
    out_dir = _validated_output_dir(
        out_dir,
        payload["targets"],
        context="cleanup plan",
    )
    path, event = write_immutable_json_event(
        out_dir,
        PLAN_EVENT_PREFIX,
        payload,
    )
    _json(
        {"plan_json": str(path), "plan_sha256": sha256_file(path), "plan": event},
        quiet=args.quiet,
    )
    return 0


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validated_output_dir(
    out_dir: Path,
    targets: Sequence[dict[str, Any]],
    *,
    context: str,
) -> Path:
    resolved = Path(out_dir).expanduser().resolve()
    for target in targets:
        target_path = Path(target["path"])
        if _paths_overlap(target_path, resolved):
            raise RuntimeError(
                f"{context} directory overlaps target: {resolved} vs {target_path}"
            )
    return resolved


def publish_cleanup_approval(
    *,
    plan_json: Path,
    plan_sha256: str,
    vedtak: str,
    approved_by: str,
    out_dir: Path,
    allowed_roots: Sequence[Path] = (GX1_DATA_ROOT,),
    required_artifact_registry_json: Path = DEFAULT_REGISTRY,
    required_launch_contract_json: Path = DEFAULT_LAUNCH,
) -> tuple[Path, str]:
    """Publish a separate immutable approval bound to one validated plan."""

    approver = approved_by.strip()
    if len(approver) < 3 or len(approver) > 128:
        raise RuntimeError("cleanup approved_by must identify the approving operator")
    validated = validate_cleanup_plan(
        plan_json,
        plan_sha256,
        vedtak=vedtak,
        allowed_roots=allowed_roots,
        required_artifact_registry_json=required_artifact_registry_json,
        required_launch_contract_json=required_launch_contract_json,
        verify_target_bytes=False,
    )
    approval_dir = _validated_output_dir(
        out_dir,
        validated["targets"],
        context="cleanup approval",
    )
    approval_path, _ = write_immutable_json_event(
        approval_dir,
        APPROVAL_PREFIX,
        {
            "schema_version": "gx1_evidence_retention_approval_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "APPROVED_EXACT_TARGET_DELETE",
            "plan_json": validated["plan_json"],
            "plan_sha256": validated["plan_sha256"],
            "vedtak": validated["vedtak"],
            "approved_by": approver,
            "targets": validated["targets"],
            "direction_authority": False,
            "launch_authority": False,
        },
    )
    return approval_path, sha256_file(approval_path)


def _validate_cleanup_approval(
    approval_json: Path,
    approval_sha256: str,
    *,
    validated_plan: dict[str, Any],
) -> dict[str, Any]:
    raw_path = Path(approval_json).expanduser()
    if not raw_path.is_absolute():
        raise RuntimeError("cleanup approval path must be absolute")
    if raw_path.is_symlink():
        raise RuntimeError(f"cleanup approval cannot be a symlink: {raw_path}")
    path = raw_path.resolve(strict=True)
    if raw_path != path or not path.is_file():
        raise RuntimeError(f"cleanup approval is not a regular file: {path}")
    expected_sha = approval_sha256.strip().lower()
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise RuntimeError("cleanup approval SHA-256 is invalid")
    if sha256_file(path) != expected_sha:
        raise RuntimeError("cleanup approval SHA-256 mismatch")
    try:
        approval = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"cleanup approval is not strict JSON: {exc}") from exc
    if not isinstance(approval, dict) or set(approval) != _APPROVAL_KEYS:
        raise RuntimeError("cleanup approval keys are invalid")
    require_newest_immutable_event(path, APPROVAL_PREFIX)
    required = {
        "schema_version": "gx1_evidence_retention_approval_v1",
        "decision": "APPROVED_EXACT_TARGET_DELETE",
        "plan_json": validated_plan["plan_json"],
        "plan_sha256": validated_plan["plan_sha256"],
        "vedtak": validated_plan["vedtak"],
        "targets": validated_plan["targets"],
        "direction_authority": False,
        "launch_authority": False,
    }
    for field, expected in required.items():
        if approval.get(field) != expected:
            raise RuntimeError(f"cleanup approval {field} does not match plan")
    if not isinstance(approval.get("approved_by"), str) or len(approval["approved_by"].strip()) < 3:
        raise RuntimeError("cleanup approval does not identify the approving operator")
    if sha256_file(path) != expected_sha:
        raise RuntimeError("cleanup approval changed during validation")
    return approval


def _validate_cleanup_started_event(
    started_json: Path,
    started_sha256: str,
    *,
    validated_plan: dict[str, Any],
    approval_json: Path,
    approval_sha256: str,
    expected_stage_plan: Sequence[dict[str, str]],
) -> tuple[Path, dict[str, Any]]:
    raw_path = Path(started_json).expanduser()
    if not raw_path.is_absolute() or raw_path.is_symlink():
        raise RuntimeError("cleanup started event path must be absolute and not a symlink")
    path = raw_path.resolve(strict=True)
    if raw_path != path or not path.is_file():
        raise RuntimeError("cleanup started event is not an exact regular file")
    expected_sha = started_sha256.strip().lower()
    if len(expected_sha) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha
    ):
        raise RuntimeError("cleanup started event SHA-256 is invalid")
    if sha256_file(path) != expected_sha:
        raise RuntimeError("cleanup started event SHA-256 mismatch")
    try:
        event = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"cleanup started event is not strict JSON: {exc}") from exc
    if not isinstance(event, dict) or set(event) != _STARTED_KEYS:
        raise RuntimeError("cleanup started event keys are invalid")
    require_newest_immutable_event(path, STARTED_PREFIX)
    required = {
        "schema_version": "gx1_evidence_cleanup_started_v1",
        "decision": "ATOMIC_STAGING_STARTED",
        "json_path": str(path),
        "plan_json": validated_plan["plan_json"],
        "plan_sha256": validated_plan["plan_sha256"],
        "approval_json": str(Path(approval_json).expanduser().resolve(strict=True)),
        "approval_sha256": approval_sha256.strip().lower(),
        "vedtak": validated_plan["vedtak"],
        "stage_plan": list(expected_stage_plan),
        "direction_authority": False,
        "launch_authority": False,
    }
    for field, expected in required.items():
        if event.get(field) != expected:
            raise RuntimeError(f"cleanup started event {field} does not match plan")
    if sha256_file(path) != expected_sha:
        raise RuntimeError("cleanup started event changed during validation")
    return path, event


def _validate_cleanup_staged_event(
    staged_json: Path,
    staged_sha256: str,
    *,
    validated_plan: dict[str, Any],
    approval_json: Path,
    approval_sha256: str,
    expected_stage_plan: Sequence[dict[str, str]],
) -> tuple[Path, dict[str, Any]]:
    """Bind the immutable STAGED event of an interrupted execution.

    The event records every target's post-staging inventory, re-verified by the
    execution that wrote it. It is the only admissible authority for resuming a
    delete loop that was interrupted after staging.
    """

    raw_path = Path(staged_json).expanduser()
    if not raw_path.is_absolute() or raw_path.is_symlink():
        raise RuntimeError("cleanup staged event path must be absolute and not a symlink")
    path = raw_path.resolve(strict=True)
    if raw_path != path or not path.is_file():
        raise RuntimeError("cleanup staged event is not an exact regular file")
    expected_sha = staged_sha256.strip().lower()
    if len(expected_sha) != 64 or any(
        char not in "0123456789abcdef" for char in expected_sha
    ):
        raise RuntimeError("cleanup staged event SHA-256 is invalid")
    if sha256_file(path) != expected_sha:
        raise RuntimeError("cleanup staged event SHA-256 mismatch")
    try:
        event = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"cleanup staged event is not strict JSON: {exc}") from exc
    if not isinstance(event, dict) or set(event) != _STAGED_KEYS:
        raise RuntimeError("cleanup staged event keys are invalid")
    require_newest_immutable_event(path, STAGED_PREFIX)
    required = {
        "schema_version": "gx1_evidence_cleanup_staged_v1",
        "decision": "EXACT_TARGETS_STAGED_AND_REVALIDATED",
        "json_path": str(path),
        "plan_json": validated_plan["plan_json"],
        "plan_sha256": validated_plan["plan_sha256"],
        "approval_json": str(Path(approval_json).expanduser().resolve(strict=True)),
        "approval_sha256": approval_sha256.strip().lower(),
        "vedtak": validated_plan["vedtak"],
        "stage_plan": list(expected_stage_plan),
        "direction_authority": False,
        "launch_authority": False,
    }
    for field, expected in required.items():
        if event.get(field) != expected:
            raise RuntimeError(f"cleanup staged event {field} does not match plan")
    staged = event.get("staged")
    targets = validated_plan["targets"]
    if not isinstance(staged, list) or len(staged) != len(targets):
        raise RuntimeError("cleanup staged event does not cover every plan target")
    for entry, mapping, target in zip(staged, expected_stage_plan, targets, strict=True):
        if not isinstance(entry, dict):
            raise RuntimeError("cleanup staged event entry is invalid")
        for field, expected in mapping.items():
            if entry.get(field) != expected:
                raise RuntimeError(
                    f"cleanup staged event entry {field} does not match the stage plan"
                )
        if not _same_inventory(target, entry):
            raise RuntimeError(
                "cleanup staged event inventory differs from the plan target: "
                f"{mapping['source_path']}"
            )
    if sha256_file(path) != expected_sha:
        raise RuntimeError("cleanup staged event changed during validation")
    return path, event


def _require_staged_delete_authority_unchanged(
    *,
    validated_plan: dict[str, Any],
    approval_json: Path,
    approval_sha256: str,
) -> None:
    """Recheck immutable authority without rescanning every staged target."""

    plan_path = Path(validated_plan["plan_json"])
    if sha256_file(plan_path) != validated_plan["plan_sha256"]:
        raise RuntimeError("cleanup plan changed after staging")
    try:
        require_newest_immutable_event(plan_path, PLAN_EVENT_PREFIX)
    except Exception as exc:
        raise RuntimeError(
            f"cleanup plan lost newest immutable authority after staging: {exc}"
        ) from exc

    authority = validated_plan["authority"]
    for label, path_field, sha_field in (
        (
            "artifact registry",
            "artifact_registry_json",
            "artifact_registry_sha256",
        ),
        ("launch contract", "launch_contract_json", "launch_contract_sha256"),
        ("delete incident", "delete_incident_json", "delete_incident_sha256"),
    ):
        path = Path(authority[path_field])
        if sha256_file(path) != authority[sha_field]:
            raise RuntimeError(f"{label} changed after staging")

    approval_path = Path(approval_json).expanduser()
    expected_approval_sha = approval_sha256.strip().lower()
    if (
        not approval_path.is_absolute()
        or approval_path.is_symlink()
        or approval_path.resolve(strict=True) != approval_path
        or sha256_file(approval_path) != expected_approval_sha
    ):
        raise RuntimeError("cleanup approval changed after staging")


def _same_inventory(declared: dict[str, Any], observed: dict[str, Any]) -> bool:
    return all(declared[field] == observed[field] for field in _INVENTORY_FIELDS)


def _stage_plan(
    targets: Sequence[dict[str, Any]],
    *,
    plan_sha256: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, target in enumerate(targets):
        source = Path(target["path"])
        wrapper = source.parent / f".gx1_delete_{plan_sha256[:16]}_{index:04d}"
        rows.append(
            {
                "source_path": str(source),
                "quarantine_wrapper": str(wrapper),
                "quarantine_path": str(wrapper / "payload"),
            }
        )
    return rows


def _open_absolute_directory_nofollow(path: Path) -> int:
    if not path.is_absolute():
        raise RuntimeError(f"cleanup directory is not absolute: {path}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except Exception:
        os.close(directory_fd)
        raise
    return directory_fd


def _sha256_fd(file_fd: int) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _inventory_manifest_rows(target: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = Path(target["inventory_jsonl"])
    if sha256_file(manifest_path) != target["inventory_jsonl_sha256"]:
        raise RuntimeError("cleanup inventory manifest changed before delete")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"cleanup inventory manifest line {line_number} is invalid"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError("cleanup inventory manifest row is not an object")
        rows.append(row)
    if not rows:
        raise RuntimeError("cleanup inventory manifest is empty")
    relative_paths = [str(row.get("relative_path", "")) for row in rows]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("cleanup inventory manifest has duplicate paths")
    for relative in relative_paths:
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or not relative:
            raise RuntimeError("cleanup inventory manifest path is unsafe")
    return rows


def _reject_open_writer_fds(target: Path) -> None:
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("cannot prove open-writer state without /proc")
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                linked = Path(os.readlink(descriptor))
                flags_line = next(
                    line
                    for line in (process / "fdinfo" / descriptor.name)
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.startswith("flags:")
                )
                flags = int(flags_line.split()[1], 8)
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                StopIteration,
                ValueError,
            ):
                continue
            if linked == target or target in linked.parents:
                if (flags & os.O_ACCMODE) in {os.O_WRONLY, os.O_RDWR}:
                    raise RuntimeError(
                        f"cleanup target has an open writer fd: pid={process.name} "
                        f"fd={descriptor.name} path={linked}"
                    )


def _delete_manifest_file(root: Path, row: dict[str, Any]) -> None:
    relative = str(row["relative_path"])
    if relative == ".":
        parent = root.parent
        name = root.name
    else:
        parsed = PurePosixPath(relative)
        parent = root.joinpath(*parsed.parent.parts)
        name = parsed.name
    parent_fd = _open_absolute_directory_nofollow(parent)
    file_fd: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"cleanup manifest file type changed: {relative}")
        file_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        opened = os.fstat(file_fd)
        observed_sha = _sha256_fd(file_fd)
        after = os.fstat(file_fd)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity != after_identity:
            raise RuntimeError(f"cleanup manifest file changed while hashing: {relative}")
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(f"cleanup manifest file identity changed: {relative}")
        if opened.st_size != row.get("size_bytes") or observed_sha != row.get("sha256"):
            raise RuntimeError(f"cleanup manifest file bytes changed: {relative}")
        os.unlink(name, dir_fd=parent_fd)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _prove_interrupted_payload_subset(
    quarantine: Path,
    plan_target: dict[str, Any],
) -> set[str]:
    """Prove a mid-deletion quarantine holds only plan-manifest paths.

    A delete loop killed part-way through leaves a payload that can never match
    its plan inventory again: some entries are already unlinked. The surviving
    bytes are still admissible for deletion — they were hash-verified into the
    plan and approved — but only once it is proven that nothing foreign appeared
    in their place. Returns the manifest-relative paths proven absent.
    """

    rows = _inventory_manifest_rows(plan_target)
    declared_files = {
        str(row["relative_path"]) for row in rows if row.get("kind") == "file"
    }
    declared_dirs = {
        str(row["relative_path"]) for row in rows if row.get("kind") == "directory"
    }
    present_files: set[str] = set()
    present_dirs: set[str] = {"."}
    for path in quarantine.rglob("*"):
        relative = str(PurePosixPath(path.relative_to(quarantine)))
        if path.is_symlink():
            raise RuntimeError(
                f"cleanup interrupted payload holds a symlink: {relative}"
            )
        if path.is_file():
            if relative not in declared_files:
                raise RuntimeError(
                    f"cleanup interrupted payload holds a foreign file: {relative}"
                )
            present_files.add(relative)
        elif path.is_dir():
            if relative not in declared_dirs:
                raise RuntimeError(
                    f"cleanup interrupted payload holds a foreign directory: {relative}"
                )
            present_dirs.add(relative)
        else:
            raise RuntimeError(
                f"cleanup interrupted payload holds a non-regular entry: {relative}"
            )
    if not (declared_files - present_files) and not (declared_dirs - present_dirs):
        raise RuntimeError(
            "cleanup interrupted payload is complete; it is not an interrupted delete"
        )
    return (declared_files - present_files) | (declared_dirs - present_dirs)


def _delete_staged_manifest_exact(
    staged_target: dict[str, Any],
    plan_target: dict[str, Any],
    *,
    absent_relative_paths: frozenset[str] = frozenset(),
) -> None:
    root = Path(staged_target["quarantine_path"])
    rows = _inventory_manifest_rows(plan_target)
    _reject_open_writer_fds(root)
    if staged_target["kind"] == "file":
        if len(rows) != 1 or rows[0].get("kind") != "file":
            raise RuntimeError("cleanup file manifest shape is invalid")
        _delete_manifest_file(root, rows[0])
        return
    if rows[0].get("relative_path") != "." or rows[0].get("kind") != "directory":
        raise RuntimeError("cleanup directory manifest root is invalid")
    for row in rows:
        if row.get("kind") == "file":
            if str(row["relative_path"]) in absent_relative_paths:
                continue
            _delete_manifest_file(root, row)
    directories = [
        PurePosixPath(str(row["relative_path"]))
        for row in rows[1:]
        if row.get("kind") == "directory"
        and str(row["relative_path"]) not in absent_relative_paths
    ]
    for relative in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory = root.joinpath(*relative.parts)
        parent_fd = _open_absolute_directory_nofollow(directory.parent)
        try:
            os.rmdir(directory.name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)
    wrapper_fd = _open_absolute_directory_nofollow(root.parent)
    try:
        os.rmdir(root.name, dir_fd=wrapper_fd)
    finally:
        os.close(wrapper_fd)


def _stage_exact_target(
    target: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    source = Path(target["path"])
    if mapping["source_path"] != str(source):
        raise RuntimeError("cleanup stage mapping does not match target")
    wrapper = Path(mapping["quarantine_wrapper"])
    quarantine = Path(mapping["quarantine_path"])
    # Do not make an active artifact disappear from its canonical path before
    # checking whether any process still has it open for writing.
    _reject_open_writer_fds(source)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = _open_absolute_directory_nofollow(source.parent)
    try:
        os.mkdir(wrapper.name, mode=0o700, dir_fd=parent_fd)
        wrapper_fd = os.open(wrapper.name, flags, dir_fd=parent_fd)
        try:
            os.rename(
                source.name,
                "payload",
                src_dir_fd=parent_fd,
                dst_dir_fd=wrapper_fd,
            )
        except Exception:
            os.close(wrapper_fd)
            os.rmdir(wrapper.name, dir_fd=parent_fd)
            raise
        else:
            os.close(wrapper_fd)
    finally:
        os.close(parent_fd)
    observed = inventory_path(quarantine)
    if not _same_inventory(target, observed):
        raise RuntimeError(
            f"cleanup staged inventory differs; quarantine retained: {quarantine}"
        )
    return {
        **mapping,
        **{field: observed[field] for field in _INVENTORY_FIELDS},
    }


def _restore_staged_target(
    target: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    source = Path(mapping["source_path"])
    wrapper = Path(mapping["quarantine_wrapper"])
    quarantine = Path(mapping["quarantine_path"])
    if source.exists() or source.is_symlink():
        raise RuntimeError(f"cleanup recovery source already exists: {source}")
    if (
        wrapper.is_symlink()
        or not wrapper.is_dir()
        or quarantine.is_symlink()
        or not quarantine.exists()
    ):
        raise RuntimeError(f"cleanup recovery quarantine is invalid: {quarantine}")
    observed = inventory_path(quarantine)
    if not _same_inventory(target, observed):
        raise RuntimeError(
            f"cleanup recovery quarantine inventory differs: {quarantine}"
        )
    _reject_open_writer_fds(quarantine)

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = _open_absolute_directory_nofollow(source.parent)
    wrapper_fd = os.open(wrapper.name, flags, dir_fd=parent_fd)
    try:
        os.rename(
            "payload",
            source.name,
            src_dir_fd=wrapper_fd,
            dst_dir_fd=parent_fd,
        )
        os.rmdir(wrapper.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(wrapper_fd)
        os.close(parent_fd)

    restored = inventory_path(source)
    if not _same_inventory(target, restored):
        raise RuntimeError(f"cleanup recovered source inventory differs: {source}")
    return {
        **mapping,
        **{field: restored[field] for field in _INVENTORY_FIELDS},
    }


def recover_interrupted_cleanup(
    *,
    plan_json: Path,
    plan_sha256: str,
    vedtak: str,
    approval_json: Path,
    approval_sha256: str,
    started_json: Path,
    started_sha256: str,
    out_dir: Path,
    recover: bool,
    quiet: bool,
    allowed_roots: Sequence[Path] = (GX1_DATA_ROOT,),
    required_artifact_registry_json: Path = DEFAULT_REGISTRY,
    required_launch_contract_json: Path = DEFAULT_LAUNCH,
) -> int:
    """Restore an interrupted pre-STAGED transaction to its exact source paths."""

    if not recover:
        raise RuntimeError("cleanup recovery requires explicit --recover")
    validated = validate_cleanup_plan(
        plan_json,
        plan_sha256,
        vedtak=vedtak,
        allowed_roots=allowed_roots,
        required_artifact_registry_json=required_artifact_registry_json,
        required_launch_contract_json=required_launch_contract_json,
        verify_target_bytes=False,
        require_targets_exist=False,
    )
    _validate_cleanup_approval(
        approval_json,
        approval_sha256,
        validated_plan=validated,
    )
    stage_plan = _stage_plan(
        validated["targets"],
        plan_sha256=validated["plan_sha256"],
    )
    started_path, _ = _validate_cleanup_started_event(
        started_json,
        started_sha256,
        validated_plan=validated,
        approval_json=approval_json,
        approval_sha256=approval_sha256,
        expected_stage_plan=stage_plan,
    )
    report_dir = _validated_output_dir(
        out_dir,
        validated["targets"],
        context="cleanup recovery report",
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    initial_state: list[dict[str, str]] = []
    for target, mapping in zip(validated["targets"], stage_plan, strict=True):
        source = Path(mapping["source_path"])
        wrapper = Path(mapping["quarantine_wrapper"])
        quarantine = Path(mapping["quarantine_path"])
        source_present = source.exists() and not source.is_symlink()
        staged_present = (
            wrapper.is_dir()
            and not wrapper.is_symlink()
            and quarantine.exists()
            and not quarantine.is_symlink()
        )
        if source_present == staged_present:
            raise RuntimeError(
                "cleanup recovery requires exactly one source/quarantine copy: "
                f"{source}"
            )
        observed = inventory_path(source if source_present else quarantine)
        if not _same_inventory(target, observed):
            raise RuntimeError(f"cleanup recovery inventory differs: {source}")
        initial_state.append(
            {
                **mapping,
                "state": "SOURCE_PRESENT" if source_present else "STAGED_PRESENT",
            }
        )

    restored: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        for target, mapping, state in zip(
            validated["targets"],
            stage_plan,
            initial_state,
            strict=True,
        ):
            if state["state"] == "SOURCE_PRESENT":
                continue
            _require_staged_delete_authority_unchanged(
                validated_plan=validated,
                approval_json=approval_json,
                approval_sha256=approval_sha256,
            )
            restored.append(_restore_staged_target(target, mapping))
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    _, event = write_immutable_json_event(
        report_dir,
        RECOVERY_PREFIX,
        {
            "schema_version": "gx1_evidence_cleanup_recovery_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "RESTORE_COMPLETE" if failure is None else "RESTORE_PARTIAL_FAILURE",
            "plan_json": validated["plan_json"],
            "plan_sha256": validated["plan_sha256"],
            "approval_json": str(Path(approval_json).expanduser().resolve()),
            "approval_sha256": approval_sha256.strip().lower(),
            "started_json": str(started_path),
            "started_sha256": started_sha256.strip().lower(),
            "vedtak": validated["vedtak"],
            "initial_state": initial_state,
            "restored": restored,
            "failure": failure,
            "direction_authority": False,
            "launch_authority": False,
        },
    )
    _json(event, quiet=quiet)
    if failure is not None:
        raise RuntimeError(f"cleanup recovery stopped after partial failure: {failure}")
    return 0


def execute_cleanup(
    *,
    plan_json: Path,
    plan_sha256: str,
    vedtak: str,
    out_dir: Path,
    execute: bool,
    quiet: bool,
    approval_json: Path | None = None,
    approval_sha256: str | None = None,
    allowed_roots: Sequence[Path] = (GX1_DATA_ROOT,),
    required_artifact_registry_json: Path = DEFAULT_REGISTRY,
    required_launch_contract_json: Path = DEFAULT_LAUNCH,
) -> int:
    """Validate a cleanup plan and delete only after explicit execution consent."""

    validated = validate_cleanup_plan(
        plan_json,
        plan_sha256,
        vedtak=vedtak,
        allowed_roots=allowed_roots,
        required_artifact_registry_json=required_artifact_registry_json,
        required_launch_contract_json=required_launch_contract_json,
        verify_target_bytes=not execute,
    )
    plan_path = Path(validated["plan_json"])
    if not execute:
        _json(
            {
                "decision": "DRY_RUN_VALIDATED_NO_DELETE",
                "validated_plan": validated,
                "side_effects": {"clearance_event": False, "delete": False},
            },
            quiet=quiet,
        )
        return 0
    if approval_json is None or approval_sha256 is None:
        raise RuntimeError(
            "cleanup execution requires exact --approval-json and --approval-sha256"
        )
    _validate_cleanup_approval(
        approval_json,
        approval_sha256,
        validated_plan=validated,
    )

    report_dir = _validated_output_dir(
        out_dir,
        validated["targets"],
        context="cleanup report",
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    clearance_payload = {
        "schema_version": "gx1_evidence_retention_clearance_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "CLEAR_TO_DELETE_EXACT_TARGETS",
        "plan_json": validated["plan_json"],
        "plan_sha256": validated["plan_sha256"],
        "approval_json": str(Path(approval_json).expanduser().resolve()),
        "approval_sha256": approval_sha256.lower(),
        "vedtak": validated["vedtak"],
        "targets": validated["targets"],
        "direction_authority": False,
        "launch_authority": False,
        "side_effects": {"delete_authorized": True, "stage_started": False},
    }
    clearance_path, _ = write_immutable_json_event(
        report_dir,
        CLEARANCE_PREFIX,
        clearance_payload,
    )
    require_newest_immutable_event(clearance_path, CLEARANCE_PREFIX)
    clearance_sha = sha256_file(clearance_path)

    # Recompute every byte identity and all authority bindings immediately
    # before the first destructive operation.
    validated = validate_cleanup_plan(
        plan_path,
        plan_sha256,
        vedtak=vedtak,
        allowed_roots=allowed_roots,
        required_artifact_registry_json=required_artifact_registry_json,
        required_launch_contract_json=required_launch_contract_json,
        verify_target_bytes=True,
    )
    _validate_cleanup_approval(
        approval_json,
        approval_sha256,
        validated_plan=validated,
    )
    stage_plan = _stage_plan(
        validated["targets"],
        plan_sha256=plan_sha256.lower(),
    )
    write_immutable_json_event(
        report_dir,
        STARTED_PREFIX,
        {
            "schema_version": "gx1_evidence_cleanup_started_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "ATOMIC_STAGING_STARTED",
            "plan_json": str(plan_path),
            "plan_sha256": plan_sha256.lower(),
            "approval_json": str(Path(approval_json).expanduser().resolve()),
            "approval_sha256": approval_sha256.lower(),
            "vedtak": vedtak,
            "stage_plan": stage_plan,
            "direction_authority": False,
            "launch_authority": False,
        },
    )
    staged: list[dict[str, Any]] = []
    deleted: list[str] = []
    failure: str | None = None
    staged_path: Path | None = None
    staged_sha: str | None = None
    try:
        for target, mapping in zip(
            validated["targets"],
            stage_plan,
            strict=True,
        ):
            staged.append(
                _stage_exact_target(
                    target,
                    mapping,
                )
            )
        staged_path, _ = write_immutable_json_event(
            report_dir,
            STAGED_PREFIX,
            {
                "schema_version": "gx1_evidence_cleanup_staged_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "decision": "EXACT_TARGETS_STAGED_AND_REVALIDATED",
                "plan_json": str(plan_path),
                "plan_sha256": plan_sha256.lower(),
                "approval_json": str(Path(approval_json).expanduser().resolve()),
                "approval_sha256": approval_sha256.lower(),
                "vedtak": vedtak,
                "stage_plan": stage_plan,
                "staged": staged,
                "direction_authority": False,
                "launch_authority": False,
            },
        )
        staged_sha = sha256_file(staged_path)
        for staged_target, plan_target in zip(
            staged,
            validated["targets"],
            strict=True,
        ):
            _require_staged_delete_authority_unchanged(
                validated_plan=validated,
                approval_json=approval_json,
                approval_sha256=approval_sha256,
            )
            _delete_staged_manifest_exact(staged_target, plan_target)
            deleted.append(staged_target["source_path"])
            Path(staged_target["quarantine_wrapper"]).rmdir()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    execution_payload = {
        "schema_version": "gx1_evidence_cleanup_execution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "DELETE_COMPLETE" if failure is None else "CLEANUP_PARTIAL_FAILURE",
        "plan_json": str(plan_path),
        "plan_sha256": plan_sha256.lower(),
        "vedtak": vedtak,
        "clearance_json": str(clearance_path),
        "clearance_sha256": clearance_sha,
        "approval_json": str(Path(approval_json).expanduser().resolve()),
        "approval_sha256": approval_sha256.lower(),
        "staged_json": None if staged_path is None else str(staged_path),
        "staged_sha256": staged_sha,
        "stage_plan": stage_plan,
        "staged": staged,
        "deleted": deleted,
        "failure": failure,
        "direction_authority": False,
        "launch_authority": False,
    }
    _, execution = write_immutable_json_event(
        report_dir,
        EXECUTION_PREFIX,
        execution_payload,
    )
    _json(execution, quiet=quiet)
    if failure is not None:
        raise RuntimeError(f"cleanup stopped after partial failure: {failure}")
    return 0


def resume_interrupted_cleanup(
    *,
    plan_json: Path,
    plan_sha256: str,
    vedtak: str,
    approval_json: Path,
    approval_sha256: str,
    staged_json: Path,
    staged_sha256: str,
    out_dir: Path,
    resume: bool,
    quiet: bool,
    allow_interrupted_payload: bool = False,
    allowed_roots: Sequence[Path] = (GX1_DATA_ROOT,),
    required_artifact_registry_json: Path = DEFAULT_REGISTRY,
    required_launch_contract_json: Path = DEFAULT_LAUNCH,
) -> int:
    """Finish the delete loop of an execution interrupted after STAGED.

    `execute` stages every target, revalidates it by content hash, writes the
    immutable STAGED event and only then removes bytes. An execution killed
    inside that final loop leaves a transaction that `execute` cannot repeat
    (its sources are gone) and that `recover` cannot restore (some targets are
    already deleted, so its exactly-one-copy invariant fails). This completes
    it from the STAGED event, which carries the already-revalidated inventory
    of every target.

    Fail-closed states: a present source (nothing to resume — that target was
    never staged or was recovered), a source and a quarantine copy at once, a
    quarantine whose inventory no longer matches the plan, or any authority
    byte changed since the plan was approved.

    `allow_interrupted_payload` admits the one remaining state: a target the
    interrupted run died *inside*, whose payload is a strict subset of its plan
    manifest. It can never match its inventory again, so it is admitted only
    after proving every surviving entry is a declared manifest path and that no
    foreign path appeared; every surviving file is still hash-checked against
    the plan immediately before it is unlinked. It never widens what may be
    deleted — only which already-approved bytes may still be reached.
    """

    if not resume:
        raise RuntimeError("cleanup resume requires explicit --resume")
    validated = validate_cleanup_plan(
        plan_json,
        plan_sha256,
        vedtak=vedtak,
        allowed_roots=allowed_roots,
        required_artifact_registry_json=required_artifact_registry_json,
        required_launch_contract_json=required_launch_contract_json,
        verify_target_bytes=False,
        require_targets_exist=False,
    )
    _validate_cleanup_approval(
        approval_json,
        approval_sha256,
        validated_plan=validated,
    )
    stage_plan = _stage_plan(
        validated["targets"],
        plan_sha256=validated["plan_sha256"],
    )
    staged_path, staged_event = _validate_cleanup_staged_event(
        staged_json,
        staged_sha256,
        validated_plan=validated,
        approval_json=approval_json,
        approval_sha256=approval_sha256,
        expected_stage_plan=stage_plan,
    )
    report_dir = _validated_output_dir(
        out_dir,
        validated["targets"],
        context="cleanup resume report",
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    # Classify and revalidate every target before removing a single byte.
    pending: list[tuple[dict[str, Any], dict[str, Any], frozenset[str]]] = []
    already_deleted: list[str] = []
    interrupted: list[dict[str, Any]] = []
    for target, mapping, entry in zip(
        validated["targets"],
        stage_plan,
        staged_event["staged"],
        strict=True,
    ):
        source = Path(mapping["source_path"])
        wrapper = Path(mapping["quarantine_wrapper"])
        quarantine = Path(mapping["quarantine_path"])
        source_present = source.exists() or source.is_symlink()
        wrapper_present = wrapper.exists() or wrapper.is_symlink()
        if source_present and wrapper_present:
            raise RuntimeError(
                f"cleanup resume found a source and a quarantine copy: {source}"
            )
        if source_present:
            raise RuntimeError(
                "cleanup resume requires every target staged or already deleted; "
                f"source is present: {source}"
            )
        if not wrapper_present:
            already_deleted.append(str(source))
            continue
        if (
            wrapper.is_symlink()
            or not wrapper.is_dir()
            or quarantine.is_symlink()
            or not quarantine.exists()
        ):
            raise RuntimeError(f"cleanup resume quarantine is invalid: {quarantine}")
        observed = inventory_path(quarantine)
        if _same_inventory(target, observed):
            pending.append(
                (
                    target,
                    {**entry, **{f: observed[f] for f in _INVENTORY_FIELDS}},
                    frozenset(),
                )
            )
            continue
        if not allow_interrupted_payload:
            raise RuntimeError(
                f"cleanup resume quarantine inventory differs: {quarantine}"
            )
        absent = _prove_interrupted_payload_subset(quarantine, target)
        interrupted.append(
            {
                **entry,
                **{f: observed[f] for f in _INVENTORY_FIELDS},
                "absent_relative_paths": sorted(absent),
            }
        )
        pending.append(
            (
                target,
                {**entry, **{f: observed[f] for f in _INVENTORY_FIELDS}},
                frozenset(absent),
            )
        )

    deleted: list[str] = []
    failure: str | None = None
    try:
        for plan_target, staged_target, absent in pending:
            _require_staged_delete_authority_unchanged(
                validated_plan=validated,
                approval_json=approval_json,
                approval_sha256=approval_sha256,
            )
            _delete_staged_manifest_exact(
                staged_target,
                plan_target,
                absent_relative_paths=absent,
            )
            deleted.append(staged_target["source_path"])
            Path(staged_target["quarantine_wrapper"]).rmdir()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    _, execution = write_immutable_json_event(
        report_dir,
        EXECUTION_PREFIX,
        {
            "schema_version": "gx1_evidence_cleanup_execution_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": (
                "DELETE_COMPLETE" if failure is None else "CLEANUP_PARTIAL_FAILURE"
            ),
            "plan_json": validated["plan_json"],
            "plan_sha256": validated["plan_sha256"],
            "vedtak": validated["vedtak"],
            "resumed_from_staged_json": str(staged_path),
            "resumed_from_staged_sha256": staged_sha256.strip().lower(),
            "approval_json": str(Path(approval_json).expanduser().resolve()),
            "approval_sha256": approval_sha256.strip().lower(),
            "stage_plan": stage_plan,
            "staged": [staged_target for _, staged_target, _absent in pending],
            "deleted": deleted,
            "already_deleted_before_resume": already_deleted,
            "interrupted_payloads": interrupted,
            "failure": failure,
            "direction_authority": False,
            "launch_authority": False,
        },
    )
    _json(execution, quiet=quiet)
    if failure is not None:
        raise RuntimeError(f"cleanup resume stopped after partial failure: {failure}")
    return 0


def _resume(args: argparse.Namespace) -> int:
    return resume_interrupted_cleanup(
        plan_json=Path(args.plan_json),
        plan_sha256=args.plan_sha256,
        vedtak=args.vedtak,
        approval_json=Path(args.approval_json),
        approval_sha256=args.approval_sha256,
        staged_json=Path(args.staged_json),
        staged_sha256=args.staged_sha256,
        out_dir=Path(args.out_dir),
        resume=args.resume,
        quiet=args.quiet,
        allow_interrupted_payload=args.allow_interrupted_payload,
    )


def _execute(args: argparse.Namespace) -> int:
    return execute_cleanup(
        plan_json=Path(args.plan_json),
        plan_sha256=args.plan_sha256,
        vedtak=args.vedtak,
        out_dir=Path(args.out_dir),
        execute=args.execute,
        quiet=args.quiet,
        approval_json=None if args.approval_json is None else Path(args.approval_json),
        approval_sha256=args.approval_sha256,
    )


def _approve(args: argparse.Namespace) -> int:
    if not args.approve:
        raise RuntimeError("approval publication requires explicit --approve")
    path, digest = publish_cleanup_approval(
        plan_json=Path(args.plan_json),
        plan_sha256=args.plan_sha256,
        vedtak=args.vedtak,
        approved_by=args.approved_by,
        out_dir=Path(args.out_dir),
    )
    _json(
        {"approval_json": str(path), "approval_sha256": digest},
        quiet=args.quiet,
    )
    return 0


def _recover(args: argparse.Namespace) -> int:
    return recover_interrupted_cleanup(
        plan_json=Path(args.plan_json),
        plan_sha256=args.plan_sha256,
        vedtak=args.vedtak,
        approval_json=Path(args.approval_json),
        approval_sha256=args.approval_sha256,
        started_json=Path(args.started_json),
        started_sha256=args.started_sha256,
        out_dir=Path(args.out_dir),
        recover=args.recover,
        quiet=args.quiet,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--target", action="append", required=True)
    plan.add_argument("--reason", required=True)
    plan.add_argument("--vedtak", required=True)
    plan.add_argument("--out-dir", default=str(DEFAULT_PLAN_DIR))
    plan.add_argument("--quiet", action="store_true")
    plan.set_defaults(handler=_plan)

    approve = subparsers.add_parser("approve")
    approve.add_argument("--plan-json", required=True)
    approve.add_argument("--plan-sha256", required=True)
    approve.add_argument("--vedtak", required=True)
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--out-dir", default=str(DEFAULT_APPROVAL_DIR))
    approve.add_argument("--approve", action="store_true")
    approve.add_argument("--quiet", action="store_true")
    approve.set_defaults(handler=_approve)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--plan-json", required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--vedtak", required=True)
    execute.add_argument("--approval-json")
    execute.add_argument("--approval-sha256")
    execute.add_argument("--out-dir", default=str(DEFAULT_REPORT_DIR))
    execute.add_argument("--execute", action="store_true")
    execute.add_argument("--quiet", action="store_true")
    execute.set_defaults(handler=_execute)

    resume = subparsers.add_parser("resume")
    resume.add_argument("--plan-json", required=True)
    resume.add_argument("--plan-sha256", required=True)
    resume.add_argument("--vedtak", required=True)
    resume.add_argument("--approval-json", required=True)
    resume.add_argument("--approval-sha256", required=True)
    resume.add_argument("--staged-json", required=True)
    resume.add_argument("--staged-sha256", required=True)
    resume.add_argument("--out-dir", default=str(DEFAULT_REPORT_DIR))
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--allow-interrupted-payload", action="store_true")
    resume.add_argument("--quiet", action="store_true")
    resume.set_defaults(handler=_resume)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--plan-json", required=True)
    recover.add_argument("--plan-sha256", required=True)
    recover.add_argument("--vedtak", required=True)
    recover.add_argument("--approval-json", required=True)
    recover.add_argument("--approval-sha256", required=True)
    recover.add_argument("--started-json", required=True)
    recover.add_argument("--started-sha256", required=True)
    recover.add_argument("--out-dir", default=str(DEFAULT_REPORT_DIR))
    recover.add_argument("--recover", action="store_true")
    recover.add_argument("--quiet", action="store_true")
    recover.set_defaults(handler=_recover)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
