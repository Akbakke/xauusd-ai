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


def _delete_staged_manifest_exact(
    staged_target: dict[str, Any],
    plan_target: dict[str, Any],
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
            _delete_manifest_file(root, row)
    directories = [
        PurePosixPath(str(row["relative_path"]))
        for row in rows[1:]
        if row.get("kind") == "directory"
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
            authority_now = validate_cleanup_plan(
                plan_path,
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
                validated_plan=authority_now,
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
