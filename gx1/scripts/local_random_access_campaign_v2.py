"""Transactional CLI for the random-access Windows reboot campaign v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.local_random_access_campaign_v2 import (
    ACTIVE_SCHEMA,
    REBOOT_INTENT_SCHEMA,
    REBOOT_RECEIPT_SCHEMA,
    RECEIPT_SCHEMA,
    RandomAccessCampaignError,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    next_action,
    require_boot_identity,
    require_clean_source,
    require_plan,
    require_progress,
    require_receipt,
    require_receipt_chain,
)


def _absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        raise argparse.ArgumentTypeError("absolute normalized path required")
    return path


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RandomAccessCampaignError("JSON object required")
    return value


def _atomic_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RandomAccessCampaignError(f"refusing to replace existing file: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_private_directory(path: Path, *, label: str) -> None:
    """Create a mutable-output parent without following directory symlinks."""
    missing: list[Path] = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    for existing in (cursor, *cursor.parents):
        info = existing.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RandomAccessCampaignError(f"{label} parent chain invalid")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RandomAccessCampaignError(f"{label} parent creation invalid")
        os.chmod(directory, 0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RandomAccessCampaignError(f"{label} parent invalid")
    os.chmod(path, 0o700)
    if stat.S_IMODE(path.lstat().st_mode) != 0o700:
        raise RandomAccessCampaignError(f"{label} parent is not private")


def _snapshot_invocation_evidence(
    *, runtime: Path, invocation_number: int, sources: Mapping[str, Path]
) -> dict[str, dict[str, str]]:
    """Atomically publish exact, immutable evidence bytes for one invocation."""
    evidence_root = runtime / "invocation-evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise RandomAccessCampaignError("invocation evidence root invalid")
    destination = evidence_root / f"invocation-{invocation_number:04d}"
    if destination.exists() or destination.is_symlink():
        raise RandomAccessCampaignError("invocation evidence collision")
    staging = evidence_root / (
        f".invocation-{invocation_number:04d}.{secrets.token_hex(8)}.tmp"
    )
    staging.mkdir(mode=0o700)
    digests: dict[str, str] = {}
    try:
        for name, source in sources.items():
            if not source.is_file() or source.is_symlink():
                raise RandomAccessCampaignError(f"{name} evidence unavailable")
            before = file_sha256(source)
            target = staging / name
            copied = hashlib.sha256()
            with source.open("rb") as reader, target.open("xb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    copied.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            if copied.hexdigest() != before or file_sha256(source) != before:
                raise RandomAccessCampaignError(
                    f"{name} evidence changed while being snapshotted"
                )
            os.chmod(target, 0o400)
            digests[name] = before
        directory_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(staging, destination)
        root_fd = os.open(evidence_root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        name: {"path": str(destination / name), "sha256": digest}
        for name, digest in digests.items()
    }


def _receipts(runtime: Path) -> list[dict[str, Any]]:
    root = runtime / "receipts"
    if not root.exists():
        return []
    if root.is_symlink() or not root.is_dir():
        raise RandomAccessCampaignError("receipt directory invalid")
    paths = sorted(root.glob("invocation-*.json"))
    expected = [
        root / f"invocation-{index:04d}.json" for index in range(1, len(paths) + 1)
    ]
    if paths != expected:
        raise RandomAccessCampaignError("receipt filenames are not contiguous")
    return [_read(path) for path in paths]


def _load_plan(plan_path: Path, plan_file_sha256: str) -> dict[str, Any]:
    if file_sha256(plan_path) != plan_file_sha256:
        raise RandomAccessCampaignError("plan file SHA-256 mismatch")
    return require_plan(_read(plan_path), verify_files=True)


def _active_path(runtime: Path) -> Path:
    return runtime / "ACTIVE_INVOCATION.json"


def _reconcile_active(
    *, plan: Mapping[str, Any], receipts: list[dict[str, Any]]
) -> None:
    runtime = Path(plan["runtime_root"])
    active_path = _active_path(runtime)
    if not active_path.exists():
        return
    active = _read(active_path)
    number = active.get("invocation_number")
    if type(number) is not int or number < 1:
        raise RandomAccessCampaignError("active invocation marker invalid")
    receipt_path = runtime / "receipts" / f"invocation-{number:04d}.json"
    if not receipt_path.is_file():
        raise RandomAccessCampaignError("BLOCKED_RECOVERY_RECEIPT_REQUIRED")
    if number > len(receipts):
        raise RandomAccessCampaignError("active invocation receipt sequence invalid")
    receipt = receipts[number - 1]
    if receipt.get("active_marker_sha256") != file_sha256(active_path):
        raise RandomAccessCampaignError("active invocation/receipt mismatch")
    archive = runtime / "active-archive" / f"invocation-{number:04d}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() or archive.is_symlink():
        raise RandomAccessCampaignError("active archive collision")
    os.replace(active_path, archive)


def _required_reboot_number(receipt_count: int) -> int:
    return receipt_count


def _require_reboot_receipt(
    *,
    plan: Mapping[str, Any],
    receipts: list[dict[str, Any]],
    current_boot: Mapping[str, Any],
) -> None:
    number = _required_reboot_number(len(receipts))
    path = Path(plan["runtime_root"]) / "reboot-receipts" / f"after-{number:04d}.json"
    if not path.is_file() or path.is_symlink():
        raise RandomAccessCampaignError("confirmed reboot request receipt required")
    value = _read(path)
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        set(value)
        != {
            "schema_version",
            "plan_sha256",
            "after_invocation_number",
            "next_invocation_number",
            "requested_from_boot",
            "request_nonce",
            "shutdown_exit_code",
            "requested_utc",
            "receipt_sha256",
        }
        or value.get("schema_version") != REBOOT_RECEIPT_SCHEMA
        or value.get("plan_sha256") != plan["plan_sha256"]
        or value.get("after_invocation_number") != number
        or value.get("next_invocation_number") != number + 1
        or value.get("shutdown_exit_code") != 0
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise RandomAccessCampaignError("reboot request receipt invalid")
    requested_boot = require_boot_identity(value["requested_from_boot"])
    predecessor_boot = (
        plan["prepared_windows_boot"] if number == 0 else receipts[number - 1]["boot"]
    )
    if requested_boot["identity_sha256"] != predecessor_boot["identity_sha256"]:
        raise RandomAccessCampaignError("reboot request predecessor boot differs")
    checked_current = require_boot_identity(current_boot)
    if (
        checked_current["boot_id"] <= requested_boot["boot_id"]
        or checked_current["identity_sha256"] == requested_boot["identity_sha256"]
    ):
        raise RandomAccessCampaignError("confirmed physical reboot not observed")


def inspect_campaign(
    *, plan_path: Path, plan_file_sha256: str, current_boot: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _load_plan(plan_path, plan_file_sha256)
    require_clean_source(plan)
    runtime = Path(plan["runtime_root"])
    receipts = _receipts(runtime)
    require_receipt_chain(plan, receipts, verify_files=True)
    _reconcile_active(plan=plan, receipts=receipts)
    action = next_action(plan, receipts, current_boot=current_boot)
    if action["decision"] == "LAUNCH" and receipts:
        _require_reboot_receipt(plan=plan, receipts=receipts, current_boot=current_boot)
    return {
        "ok": True,
        "campaign_id": plan["campaign_id"],
        "plan_sha256": plan["plan_sha256"],
        "action": action,
        "controller_sources": plan["controller_sources"],
        "policy": plan["policy"],
        "cuda_started": False,
        "test_authorized": False,
    }


def begin_invocation(
    *,
    plan_path: Path,
    plan_file_sha256: str,
    current_boot: Mapping[str, Any],
) -> dict[str, Any]:
    inspected = inspect_campaign(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha256,
        current_boot=current_boot,
    )
    action = inspected["action"]
    if action["decision"] != "LAUNCH":
        raise RandomAccessCampaignError("campaign has no admissible launch")
    plan = _load_plan(plan_path, plan_file_sha256)
    runtime = Path(plan["runtime_root"])
    invocation = action["invocation"]
    pointer = Path(invocation["checkpoint"]["pointer_path"])
    progress_path = Path(invocation["progress_path"])
    guard_path = Path(invocation["guard_log_path"])
    for parent, label in (
        (pointer.parent, "checkpoint"),
        (progress_path.parent, "progress"),
        (guard_path.parent, "guard"),
    ):
        _prepare_private_directory(parent, label=label)
    if guard_path.exists() or guard_path.is_symlink():
        raise RandomAccessCampaignError("fresh guard output already exists")
    before_mode = invocation["checkpoint"]["before_mode"]
    rollout_cursor_before: str | None = None
    if before_mode == "GENESIS":
        if pointer.exists() or pointer.is_symlink():
            raise RandomAccessCampaignError("GENESIS pointer already exists")
        pointer_before = "GENESIS"
    elif before_mode == "FINAL_AUTHORITY":
        authority = plan.get("checked_final_train_checkpoint_authority")
        if (
            not pointer.is_file()
            or pointer.is_symlink()
            or not isinstance(authority, Mapping)
        ):
            raise RandomAccessCampaignError("final authority pointer unavailable")
        pointer_before = file_sha256(pointer)
        if pointer_before != authority["final_checkpoint_pointer"]["sha256"]:
            raise RandomAccessCampaignError("final authority pointer differs")
        cursor = Path(invocation["rollout_cursor_path"])
        if cursor.exists() or cursor.is_symlink():
            raise RandomAccessCampaignError("initial rollout cursor must be absent")
        rollout_cursor_before = "GENESIS"
    else:
        if not pointer.is_file() or pointer.is_symlink():
            raise RandomAccessCampaignError("resume pointer unavailable")
        predecessor = invocation["checkpoint"]["predecessor_invocation_number"]
        prior = _receipts(runtime)[predecessor - 1]
        pointer_before = file_sha256(pointer)
        if pointer_before != prior["checkpoint_pointer_after"]["sha256"]:
            raise RandomAccessCampaignError(
                "resume pointer differs from predecessor receipt"
            )
        if invocation["kind"] == "full_val_window":
            cursor = Path(invocation["rollout_cursor_path"])
            if not cursor.is_file() or cursor.is_symlink():
                raise RandomAccessCampaignError("rollout cursor unavailable")
            rollout_cursor_before = file_sha256(cursor)
            if rollout_cursor_before != prior["rollout_cursor_snapshot"]["sha256"]:
                raise RandomAccessCampaignError(
                    "rollout cursor differs from predecessor receipt"
                )
    marker = {
        "schema_version": ACTIVE_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "invocation_sha256": invocation["invocation_sha256"],
        "invocation_number": invocation["invocation_number"],
        "invocation_id": invocation["invocation_id"],
        "kind": invocation["kind"],
        "boot": require_boot_identity(current_boot),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pointer_before_sha256": pointer_before,
        "launcher_argv_sha256": invocation["launcher_argv_sha256"],
    }
    if invocation["kind"] == "full_val_window":
        marker["rollout_cursor_before_sha256"] = rollout_cursor_before
    marker["marker_sha256"] = canonical_sha256(marker)
    path = _active_path(runtime)
    _atomic_new(path, marker)
    return {
        "ok": True,
        "active_marker": {"path": str(path), "sha256": file_sha256(path)},
        "invocation": invocation,
    }


def _require_guard_log(path: Path, *, success: bool) -> None:
    if not path.is_file() or path.is_symlink():
        raise RandomAccessCampaignError("signed guard log unavailable")
    text = path.read_text(encoding="utf-8", errors="strict")
    if (
        "telemetry_owner=signed_windows_bridge" not in text
        or "event=telemetry" not in text
    ):
        raise RandomAccessCampaignError("signed guard telemetry evidence incomplete")
    if success and "event=exit child_status=0" not in text:
        raise RandomAccessCampaignError(
            "signed guard successful terminal evidence missing"
        )


def record_invocation(
    *,
    plan_path: Path,
    plan_file_sha256: str,
    trainer_guard_exit_code: int,
    progress_observer_exit_code: int,
    outcome: str,
) -> dict[str, Any]:
    plan = _load_plan(plan_path, plan_file_sha256)
    runtime = Path(plan["runtime_root"])
    active_path = _active_path(runtime)
    if not active_path.is_file() or active_path.is_symlink():
        raise RandomAccessCampaignError("active invocation marker unavailable")
    active = _read(active_path)
    marker_unsigned = {
        key: item for key, item in active.items() if key != "marker_sha256"
    }
    if (
        active.get("schema_version") != ACTIVE_SCHEMA
        or active.get("plan_sha256") != plan["plan_sha256"]
        or active.get("marker_sha256") != canonical_sha256(marker_unsigned)
    ):
        raise RandomAccessCampaignError("active invocation marker invalid")
    invocations = plan["checked_invocations"]
    number = active["invocation_number"]
    invocation = invocations[number - 1]
    if active.get("invocation_sha256") != invocation["invocation_sha256"]:
        raise RandomAccessCampaignError("active invocation binding differs")
    pointer = Path(invocation["checkpoint"]["pointer_path"])
    if not pointer.is_file() or pointer.is_symlink():
        raise RandomAccessCampaignError(
            "checkpoint pointer after invocation unavailable"
        )
    pointer_after = file_sha256(pointer)
    progress_path = Path(invocation["progress_path"])
    if not progress_path.is_file() or progress_path.is_symlink():
        raise RandomAccessCampaignError("progress receipt unavailable")
    progress = require_progress(
        _read(progress_path),
        plan_sha256=plan["plan_sha256"],
        invocation=invocation,
        expected_selection_receipt_sha256=plan["selection_artifact_sha256"],
        verify_file=True,
    )
    if progress["outcome"] != outcome:
        raise RandomAccessCampaignError("progress outcome differs")
    success = trainer_guard_exit_code == 0 and progress_observer_exit_code == 0
    guard_path = Path(invocation["guard_log_path"])
    _require_guard_log(guard_path, success=success)
    evidence_sources = {
        "CHECKPOINT_POINTER.json": pointer,
        "PROGRESS.json": progress_path,
        "SIGNED_GUARD.log": guard_path,
    }
    rollout_cursor: Path | None = None
    if invocation["kind"] == "full_val_window":
        rollout_cursor = Path(invocation["rollout_cursor_path"])
        evidence_sources["ROLLOUT_CURSOR.json"] = rollout_cursor
    snapshots = _snapshot_invocation_evidence(
        runtime=runtime,
        invocation_number=number,
        sources=evidence_sources,
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "invocation_sha256": invocation["invocation_sha256"],
        "invocation_number": number,
        "invocation_id": invocation["invocation_id"],
        "kind": invocation["kind"],
        "selection_receipt_sha256": plan["selection_artifact_sha256"],
        "boot": active["boot"],
        "started_utc": active["started_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
        "trainer_guard_exit_code": trainer_guard_exit_code,
        "progress_observer_exit_code": progress_observer_exit_code,
        "pointer_before_sha256": active["pointer_before_sha256"],
        "checkpoint_pointer_after": {
            "path": str(pointer),
            "sha256": pointer_after,
        },
        "checkpoint_pointer_snapshot": snapshots["CHECKPOINT_POINTER.json"],
        "progress": snapshots["PROGRESS.json"],
        "guard_log": snapshots["SIGNED_GUARD.log"],
        "guard_decision": "PASS" if success and outcome != "FAILED" else "FAILED",
        "signed_guard_telemetry_owner": "gx1_guarded_trainer_exec.sh",
        "active_marker_sha256": file_sha256(active_path),
        "test_data_used": False,
    }
    if invocation["kind"] == "full_val_window":
        if rollout_cursor is None:
            raise RandomAccessCampaignError("rollout cursor unavailable")
        receipt["rollout_cursor_before_sha256"] = active.get(
            "rollout_cursor_before_sha256"
        )
        receipt["rollout_cursor_after"] = {
            "path": str(rollout_cursor),
            "sha256": file_sha256(rollout_cursor),
        }
        receipt["rollout_cursor_snapshot"] = snapshots["ROLLOUT_CURSOR.json"]
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    require_receipt(
        receipt,
        plan_sha256=plan["plan_sha256"],
        invocation=invocation,
        verify_files=True,
    )
    receipt_path = runtime / "receipts" / f"invocation-{number:04d}.json"
    _atomic_new(receipt_path, receipt)
    archive = runtime / "active-archive" / f"invocation-{number:04d}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() or archive.is_symlink():
        raise RandomAccessCampaignError("active archive collision")
    os.replace(active_path, archive)
    return {
        "ok": True,
        "receipt": receipt,
        "path": str(receipt_path),
        "sha256": file_sha256(receipt_path),
    }


def prepare_reboot(
    *, plan_path: Path, plan_file_sha256: str, current_boot: Mapping[str, Any]
) -> dict[str, Any]:
    plan = _load_plan(plan_path, plan_file_sha256)
    receipts = _receipts(Path(plan["runtime_root"]))
    require_receipt_chain(plan, receipts, verify_files=True)
    after = len(receipts)
    if after >= len(plan["checked_invocations"]):
        raise RandomAccessCampaignError("campaign complete; reboot not required")
    intent = {
        "schema_version": REBOOT_INTENT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "after_invocation_number": after,
        "next_invocation_number": after + 1,
        "requested_from_boot": require_boot_identity(current_boot),
        "request_nonce": secrets.token_hex(32),
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
    }
    intent["intent_sha256"] = canonical_sha256(intent)
    path = Path(plan["runtime_root"]) / "reboot-pending" / f"after-{after:04d}.json"
    _atomic_new(path, intent)
    return {
        "ok": True,
        "intent": intent,
        "path": str(path),
        "sha256": file_sha256(path),
    }


def confirm_reboot(
    *,
    plan_path: Path,
    plan_file_sha256: str,
    request_nonce: str,
    shutdown_exit_code: int,
) -> dict[str, Any]:
    if shutdown_exit_code != 0:
        raise RandomAccessCampaignError("shutdown.exe did not accept reboot request")
    plan = _load_plan(plan_path, plan_file_sha256)
    receipts = _receipts(Path(plan["runtime_root"]))
    after = len(receipts)
    pending = Path(plan["runtime_root"]) / "reboot-pending" / f"after-{after:04d}.json"
    if not pending.is_file() or pending.is_symlink():
        raise RandomAccessCampaignError("pending reboot intent unavailable")
    intent = _read(pending)
    unsigned_intent = {
        key: item for key, item in intent.items() if key != "intent_sha256"
    }
    if (
        intent.get("schema_version") != REBOOT_INTENT_SCHEMA
        or intent.get("plan_sha256") != plan["plan_sha256"]
        or intent.get("after_invocation_number") != after
        or intent.get("request_nonce") != request_nonce
        or intent.get("intent_sha256") != canonical_sha256(unsigned_intent)
    ):
        raise RandomAccessCampaignError("pending reboot intent invalid")
    receipt = {
        "schema_version": REBOOT_RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "after_invocation_number": after,
        "next_invocation_number": after + 1,
        "requested_from_boot": intent["requested_from_boot"],
        "request_nonce": request_nonce,
        "shutdown_exit_code": shutdown_exit_code,
        "requested_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    final = Path(plan["runtime_root"]) / "reboot-receipts" / f"after-{after:04d}.json"
    _atomic_new(final, receipt)
    archive = Path(plan["runtime_root"]) / "reboot-intent-archive" / pending.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() or archive.is_symlink():
        raise RandomAccessCampaignError("reboot intent archive collision")
    os.replace(pending, archive)
    return {
        "ok": True,
        "receipt": receipt,
        "path": str(final),
        "sha256": file_sha256(final),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "begin", "prepare-reboot"):
        command = sub.add_parser(name)
        command.add_argument("--plan-json", type=_absolute, required=True)
        command.add_argument("--plan-file-sha256", required=True)
        command.add_argument("--boot-json", type=_absolute, required=True)
    record = sub.add_parser("record")
    record.add_argument("--plan-json", type=_absolute, required=True)
    record.add_argument("--plan-file-sha256", required=True)
    record.add_argument("--trainer-guard-exit-code", type=int, required=True)
    record.add_argument("--progress-observer-exit-code", type=int, required=True)
    record.add_argument(
        "--outcome", choices=("COMPLETE", "RESUMABLE", "FAILED"), required=True
    )
    confirm = sub.add_parser("confirm-reboot")
    confirm.add_argument("--plan-json", type=_absolute, required=True)
    confirm.add_argument("--plan-file-sha256", required=True)
    confirm.add_argument("--request-nonce", required=True)
    confirm.add_argument("--shutdown-exit-code", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in {"inspect", "begin", "prepare-reboot"}:
            boot = require_boot_identity(_read(args.boot_json))
            common = {
                "plan_path": args.plan_json,
                "plan_file_sha256": args.plan_file_sha256,
                "current_boot": boot,
            }
            if args.command == "inspect":
                result = inspect_campaign(**common)
            elif args.command == "begin":
                result = begin_invocation(**common)
            else:
                result = prepare_reboot(**common)
        elif args.command == "record":
            result = record_invocation(
                plan_path=args.plan_json,
                plan_file_sha256=args.plan_file_sha256,
                trainer_guard_exit_code=args.trainer_guard_exit_code,
                progress_observer_exit_code=args.progress_observer_exit_code,
                outcome=args.outcome,
            )
        else:
            result = confirm_reboot(
                plan_path=args.plan_json,
                plan_file_sha256=args.plan_file_sha256,
                request_nonce=args.request_nonce,
                shutdown_exit_code=args.shutdown_exit_code,
            )
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        json.JSONDecodeError,
        RandomAccessCampaignError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
