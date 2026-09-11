"""Fail-closed random-access campaign and physical-reboot state machine v2."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

PLAN_SCHEMA = "gx1_local_random_access_campaign_v2"
INVOCATION_SCHEMA = "gx1_local_random_access_invocation_v2"
BOOT_SCHEMA = "gx1_windows_boot_identity_v1"
ACTIVE_SCHEMA = "gx1_local_random_access_active_invocation_v2"
RECEIPT_SCHEMA = "gx1_local_random_access_invocation_receipt_v2"
REBOOT_INTENT_SCHEMA = "gx1_local_random_access_reboot_intent_v1"
REBOOT_RECEIPT_SCHEMA = "gx1_local_random_access_reboot_request_receipt_v1"
PROGRESS_SCHEMA = "gx1_local_random_access_progress_v2"
STATUS_SCHEMA = "gx1_local_random_access_status_v2"

_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_KINDS = {"smoke_arm", "resume_proof", "epoch1_window", "full_val"}
_SUCCESS_OUTCOMES = {"COMPLETE", "RESUMABLE"}


class RandomAccessCampaignError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RandomAccessCampaignError(f"{label} SHA-256 invalid")
    return value


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise RandomAccessCampaignError(f"{label} UTC timestamp invalid")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RandomAccessCampaignError(f"{label} UTC timestamp invalid") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise RandomAccessCampaignError(f"{label} UTC timestamp invalid")
    return result


def _absolute(value: Any, label: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute() or path.resolve() != path:
        raise RandomAccessCampaignError(f"{label} absolute normalized path required")
    return path


def _regular(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    try:
        info = path.lstat()
    except OSError as exc:
        raise RandomAccessCampaignError(f"{label} file unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RandomAccessCampaignError(f"{label} regular file required")
    return path


def read_bound_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = _regular(path, "bound JSON")
    if file_sha256(path) != _sha(expected_sha256, "bound JSON"):
        raise RandomAccessCampaignError("bound JSON file SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RandomAccessCampaignError("bound JSON invalid") from exc
    if not isinstance(value, dict):
        raise RandomAccessCampaignError("bound JSON object required")
    return value


def require_binding(
    value: Any, *, label: str, verify_file: bool = True
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RandomAccessCampaignError(f"{label} binding invalid")
    path = _absolute(value.get("path"), label)
    digest = _sha(value.get("sha256"), label)
    if verify_file and file_sha256(_regular(path, label)) != digest:
        raise RandomAccessCampaignError(f"{label} file SHA-256 mismatch")
    return {"path": str(path), "sha256": digest}


def require_boot_identity(value: Any) -> dict[str, Any]:
    keys = {
        "schema_version",
        "computer_name",
        "last_boot_utc",
        "boot_id",
        "identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RandomAccessCampaignError("Windows boot identity fields differ")
    result = dict(value)
    claimed = _sha(result.pop("identity_sha256"), "Windows boot identity")
    if (
        result.get("schema_version") != BOOT_SCHEMA
        or not isinstance(result.get("computer_name"), str)
        or not result["computer_name"]
        or isinstance(result.get("boot_id"), bool)
        or not isinstance(result.get("boot_id"), int)
        or result["boot_id"] < 0
        or claimed != canonical_sha256(result)
    ):
        raise RandomAccessCampaignError("Windows boot identity invalid")
    _utc(result["last_boot_utc"], "Windows boot")
    result["identity_sha256"] = claimed
    return result


def fresh_boot(current: Mapping[str, Any], previous: Mapping[str, Any]) -> bool:
    now = require_boot_identity(current)
    prior = require_boot_identity(previous)
    return (
        now["computer_name"] == prior["computer_name"]
        and now["boot_id"] > prior["boot_id"]
        and _utc(now["last_boot_utc"], "current boot")
        > _utc(prior["last_boot_utc"], "previous boot")
        and now["identity_sha256"] != prior["identity_sha256"]
    )


def _require_launcher(
    argv: Any, *, source_repo: Path, kind: str, claimed_sha256: Any
) -> list[str]:
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
        or canonical_sha256(argv) != _sha(claimed_sha256, "launcher argv")
    ):
        raise RandomAccessCampaignError("launcher argv invalid")
    runner = str(source_repo / "scripts/gx1_capped_run.sh")
    python = str(source_repo / ".venv/bin/python")
    if argv[0] != runner or argv.count("--") != 1:
        raise RandomAccessCampaignError("canonical capped runner required")
    separator = argv.index("--")
    outer, target = argv[1:separator], argv[separator + 1 :]
    if len(target) < 3 or target[:2] != [python, "-m"]:
        raise RandomAccessCampaignError("direct source-bound Python module required")
    if outer[:2] != ["--class", "trainer"]:
        raise RandomAccessCampaignError("trainer class required")
    if "--mem" not in outer or "20G" not in outer or "--swap" not in outer or "512M" not in outer:
        raise RandomAccessCampaignError("canonical memory/swap envelope required")
    attended = "--attended-smoke" in outer
    if attended != (kind in {"smoke_arm", "resume_proof"}):
        raise RandomAccessCampaignError("attended-smoke scope differs from invocation kind")
    lowered = [token.lower() for token in argv]
    if any(token == "--test" or token == "test" for token in lowered):
        raise RandomAccessCampaignError("TEST token forbidden")
    return list(argv)


def require_invocation(
    value: Any,
    *,
    source_repo: Path,
    source_commit: str,
    verify_files: bool = True,
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "decision",
        "invocation_number",
        "invocation_id",
        "kind",
        "batch_size",
        "epoch_index",
        "window_index",
        "window_count",
        "optimizer_step_budget",
        "expected_success_outcome",
        "source_commit",
        "execution_manifest",
        "launcher_argv",
        "launcher_argv_sha256",
        "progress_path",
        "guard_log_path",
        "checkpoint",
        "maximum_wall_seconds",
        "requires_fresh_windows_boot",
        "signed_guard_only",
        "test_data_used",
        "invocation_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RandomAccessCampaignError("invocation fields differ")
    result = dict(value)
    claimed = _sha(result.pop("invocation_sha256"), "invocation")
    kind = result.get("kind")
    if (
        result.get("schema_version") != INVOCATION_SCHEMA
        or result.get("decision") != "PASS"
        or kind not in _KINDS
        or result.get("source_commit") != source_commit
        or result.get("requires_fresh_windows_boot") is not True
        or result.get("signed_guard_only") is not True
        or result.get("test_data_used") is not False
        or claimed != canonical_sha256(result)
    ):
        raise RandomAccessCampaignError("invocation identity invalid")
    number = result.get("invocation_number")
    if type(number) is not int or number < 1 or result.get("invocation_id") != f"invocation-{number:04d}":
        raise RandomAccessCampaignError("invocation number invalid")
    batch_size = result.get("batch_size")
    if batch_size not in (4, 8, 16):
        raise RandomAccessCampaignError("invocation batch size invalid")
    for name in ("epoch_index", "window_index", "window_count"):
        if type(result.get(name)) is not int or result[name] < 0:
            raise RandomAccessCampaignError(f"invocation {name} invalid")
    budget = result.get("optimizer_step_budget")
    if kind == "full_val":
        if budget is not None or result["expected_success_outcome"] != "COMPLETE":
            raise RandomAccessCampaignError("full VAL budget/outcome invalid")
    elif type(budget) is not int or budget < 1:
        raise RandomAccessCampaignError("optimizer-step budget invalid")
    if result["expected_success_outcome"] not in _SUCCESS_OUTCOMES:
        raise RandomAccessCampaignError("expected outcome invalid")
    maximum_wall = result.get("maximum_wall_seconds")
    if type(maximum_wall) is not int or not 1 <= maximum_wall <= 7200:
        raise RandomAccessCampaignError("invocation wall bound invalid")
    result["execution_manifest"] = require_binding(
        result["execution_manifest"], label="execution manifest", verify_file=verify_files
    )
    result["progress_path"] = str(_absolute(result["progress_path"], "progress"))
    result["guard_log_path"] = str(_absolute(result["guard_log_path"], "guard log"))
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "pointer_path",
        "before_mode",
        "predecessor_invocation_number",
        "write_mode",
    }:
        raise RandomAccessCampaignError("checkpoint policy invalid")
    pointer_path = str(_absolute(checkpoint.get("pointer_path"), "checkpoint pointer"))
    before_mode = checkpoint.get("before_mode")
    predecessor = checkpoint.get("predecessor_invocation_number")
    if before_mode == "GENESIS":
        if predecessor is not None:
            raise RandomAccessCampaignError("GENESIS checkpoint predecessor forbidden")
    elif before_mode == "PREVIOUS_RECEIPT_AFTER":
        if type(predecessor) is not int or not 1 <= predecessor < number:
            raise RandomAccessCampaignError("checkpoint predecessor invalid")
    else:
        raise RandomAccessCampaignError("checkpoint before mode invalid")
    if checkpoint.get("write_mode") not in {"ATOMIC_UPDATE", "READ_ONLY"}:
        raise RandomAccessCampaignError("checkpoint write mode invalid")
    result["checkpoint"] = {
        "pointer_path": pointer_path,
        "before_mode": before_mode,
        "predecessor_invocation_number": predecessor,
        "write_mode": checkpoint["write_mode"],
    }
    result["launcher_argv"] = _require_launcher(
        result["launcher_argv"],
        source_repo=source_repo,
        kind=str(kind),
        claimed_sha256=result["launcher_argv_sha256"],
    )
    result["invocation_sha256"] = claimed
    return result


def _require_sequence(invocations: Sequence[Mapping[str, Any]]) -> None:
    if len(invocations) < 6:
        raise RandomAccessCampaignError("campaign sequence is incomplete")
    expected_smoke = [("smoke_arm", 4), ("smoke_arm", 8), ("smoke_arm", 16)]
    for index, (kind, batch) in enumerate(expected_smoke):
        item = invocations[index]
        if (
            item["kind"] != kind
            or item["batch_size"] != batch
            or item["optimizer_step_budget"] != 3
            or item["checkpoint"]["before_mode"] != "GENESIS"
            or item["checkpoint"]["write_mode"] != "ATOMIC_UPDATE"
            or item["expected_success_outcome"] != "COMPLETE"
        ):
            raise RandomAccessCampaignError("smoke-arm sequence invalid")
    resume = invocations[3]
    if (
        resume["kind"] != "resume_proof"
        or resume["batch_size"] != 16
        or resume["optimizer_step_budget"] != 1
        or resume["checkpoint"]["before_mode"] != "PREVIOUS_RECEIPT_AFTER"
        or resume["checkpoint"]["predecessor_invocation_number"] != 3
        or resume["checkpoint"]["pointer_path"]
        != invocations[2]["checkpoint"]["pointer_path"]
        or resume["expected_success_outcome"] != "COMPLETE"
    ):
        raise RandomAccessCampaignError("selected resume-proof sequence invalid")
    epoch = [item for item in invocations[4:-1] if item["kind"] == "epoch1_window"]
    if len(epoch) != len(invocations[4:-1]) or not epoch:
        raise RandomAccessCampaignError("epoch1 window sequence invalid")
    count = len(epoch)
    for offset, item in enumerate(epoch):
        if (
            item["batch_size"] != 16
            or item["epoch_index"] != 0
            or item["window_index"] != offset
            or item["window_count"] != count
            or item["checkpoint"]["write_mode"] != "ATOMIC_UPDATE"
            or item["expected_success_outcome"]
            != ("COMPLETE" if offset == count - 1 else "RESUMABLE")
        ):
            raise RandomAccessCampaignError("epoch1 window identity invalid")
        expected_mode = "GENESIS" if offset == 0 else "PREVIOUS_RECEIPT_AFTER"
        if item["checkpoint"]["before_mode"] != expected_mode:
            raise RandomAccessCampaignError("epoch1 checkpoint chain invalid")
        if offset > 0 and (
            item["checkpoint"]["predecessor_invocation_number"]
            != epoch[offset - 1]["invocation_number"]
            or item["checkpoint"]["pointer_path"]
            != epoch[offset - 1]["checkpoint"]["pointer_path"]
        ):
            raise RandomAccessCampaignError("epoch1 checkpoint predecessor invalid")
    val = invocations[-1]
    if (
        val["kind"] != "full_val"
        or val["batch_size"] != 16
        or val["checkpoint"]["before_mode"] != "PREVIOUS_RECEIPT_AFTER"
        or val["checkpoint"]["predecessor_invocation_number"]
        != epoch[-1]["invocation_number"]
        or val["checkpoint"]["pointer_path"]
        != epoch[-1]["checkpoint"]["pointer_path"]
        or val["checkpoint"]["write_mode"] != "READ_ONLY"
    ):
        raise RandomAccessCampaignError("full VAL sequence invalid")
    for number, item in enumerate(invocations, 1):
        if item["invocation_number"] != number:
            raise RandomAccessCampaignError("invocation sequence is not contiguous")


def require_plan(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    keys = {
        "schema_version",
        "decision",
        "campaign_id",
        "created_utc",
        "source_repo",
        "source_commit",
        "runtime_root",
        "gpu_uuid",
        "prepared_windows_boot",
        "invocations",
        "signed_guard_sources",
        "controller_sources",
        "policy",
        "authority",
        "test_data_used",
        "plan_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RandomAccessCampaignError("campaign plan fields differ")
    result = dict(value)
    claimed = _sha(result.pop("plan_sha256"), "campaign plan")
    repo = _absolute(result.get("source_repo"), "source repo")
    runtime = _absolute(result.get("runtime_root"), "runtime root")
    if (
        result.get("schema_version") != PLAN_SCHEMA
        or result.get("decision") != "PASS_PREPARED"
        or not isinstance(result.get("campaign_id"), str)
        or not result["campaign_id"]
        or not isinstance(result.get("source_commit"), str)
        or _COMMIT.fullmatch(result["source_commit"]) is None
        or not isinstance(result.get("gpu_uuid"), str)
        or not result["gpu_uuid"].startswith("GPU-")
        or result.get("test_data_used") is not False
        or claimed != canonical_sha256(result)
    ):
        raise RandomAccessCampaignError("campaign plan identity invalid")
    _utc(result["created_utc"], "campaign creation")
    result["prepared_windows_boot"] = require_boot_identity(result["prepared_windows_boot"])
    expected_policy = {
        "physical_power_limit_w": 160,
        "maximum_actual_power_draw_w": 170,
        "maximum_core_temperature_c": 65,
        "maximum_memory_junction_temperature_c": 80,
        "maximum_vram_mib": 12288,
        "signed_local_telemetry_seconds": 1,
        "human_status_seconds": 900,
        "fresh_physical_windows_boot_before_every_invocation": True,
        "automatic_power_limit_change": False,
    }
    expected_authority = {
        "test": False,
        "promotion": False,
        "paper": False,
        "live": False,
        "cloud_spend": False,
    }
    if result.get("policy") != expected_policy or result.get("authority") != expected_authority:
        raise RandomAccessCampaignError("campaign policy differs")
    guard = result.get("signed_guard_sources")
    controllers = result.get("controller_sources")
    if not isinstance(guard, Mapping) or set(guard) != {"runner", "guard", "query", "certificate"}:
        raise RandomAccessCampaignError("signed guard source set invalid")
    if not isinstance(controllers, Mapping) or set(controllers) != {"controller", "observer", "campaign_cli"}:
        raise RandomAccessCampaignError("controller source set invalid")
    result["signed_guard_sources"] = {
        name: require_binding(binding, label=f"guard {name}", verify_file=verify_files)
        for name, binding in guard.items()
    }
    result["controller_sources"] = {
        name: require_binding(binding, label=f"controller {name}", verify_file=verify_files)
        for name, binding in controllers.items()
    }
    raw = result.get("invocations")
    if not isinstance(raw, list):
        raise RandomAccessCampaignError("invocation binding list invalid")
    bindings = [
        require_binding(binding, label=f"invocation {index}", verify_file=verify_files)
        for index, binding in enumerate(raw, 1)
    ]
    invocations = []
    if verify_files:
        for binding in bindings:
            invocations.append(
                require_invocation(
                    read_bound_json(Path(binding["path"]), binding["sha256"]),
                    source_repo=repo,
                    source_commit=result["source_commit"],
                    verify_files=True,
                )
            )
        _require_sequence(invocations)
    result["source_repo"] = str(repo)
    result["runtime_root"] = str(runtime)
    result["invocations"] = bindings
    result["plan_sha256"] = claimed
    if verify_files:
        result["checked_invocations"] = invocations
    return result


def require_clean_source(plan: Mapping[str, Any]) -> None:
    repo = str(plan["source_repo"])
    try:
        head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RandomAccessCampaignError("source state unavailable") from exc
    if head != plan["source_commit"] or dirty:
        raise RandomAccessCampaignError("source must be exact clean committed state")


def require_receipt(
    value: Any,
    *,
    plan_sha256: str,
    invocation: Mapping[str, Any],
    verify_files: bool = True,
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "plan_sha256",
        "invocation_sha256",
        "invocation_number",
        "invocation_id",
        "kind",
        "boot",
        "started_utc",
        "finished_utc",
        "outcome",
        "trainer_guard_exit_code",
        "progress_observer_exit_code",
        "pointer_before_sha256",
        "pointer_after_sha256",
        "progress",
        "guard_log",
        "active_marker_sha256",
        "test_data_used",
        "receipt_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RandomAccessCampaignError("receipt fields differ")
    result = dict(value)
    claimed = _sha(result.pop("receipt_sha256"), "receipt")
    if (
        result.get("schema_version") != RECEIPT_SCHEMA
        or result.get("plan_sha256") != plan_sha256
        or result.get("invocation_sha256") != invocation["invocation_sha256"]
        or result.get("invocation_number") != invocation["invocation_number"]
        or result.get("invocation_id") != invocation["invocation_id"]
        or result.get("kind") != invocation["kind"]
        or result.get("test_data_used") is not False
        or claimed != canonical_sha256(result)
    ):
        raise RandomAccessCampaignError("receipt identity invalid")
    result["boot"] = require_boot_identity(result["boot"])
    started = _utc(result["started_utc"], "receipt start")
    finished = _utc(result["finished_utc"], "receipt finish")
    if finished < started:
        raise RandomAccessCampaignError("receipt time interval invalid")
    for name in ("trainer_guard_exit_code", "progress_observer_exit_code"):
        if type(result.get(name)) is not int:
            raise RandomAccessCampaignError("receipt exit code invalid")
    outcome = result.get("outcome")
    success = (
        result["trainer_guard_exit_code"] == 0
        and result["progress_observer_exit_code"] == 0
    )
    if outcome == "FAILED":
        if success:
            raise RandomAccessCampaignError("failed receipt has successful processes")
    elif outcome != invocation["expected_success_outcome"] or not success:
        raise RandomAccessCampaignError("receipt outcome/exit codes invalid")
    before = result.get("pointer_before_sha256")
    if invocation["checkpoint"]["before_mode"] == "GENESIS":
        if before != "GENESIS":
            raise RandomAccessCampaignError("GENESIS pointer-before invalid")
    else:
        _sha(before, "pointer before")
    _sha(result.get("pointer_after_sha256"), "pointer after")
    _sha(result.get("active_marker_sha256"), "active marker")
    result["progress"] = require_binding(
        result["progress"], label="progress receipt", verify_file=verify_files
    )
    result["guard_log"] = require_binding(
        result["guard_log"], label="signed guard log", verify_file=verify_files
    )
    result["receipt_sha256"] = claimed
    return result


def require_receipt_chain(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], *, verify_files: bool = True
) -> list[dict[str, Any]]:
    checked_plan = (
        dict(plan)
        if isinstance(plan, Mapping) and "checked_invocations" in plan
        else require_plan(plan, verify_files=verify_files)
    )
    invocations = checked_plan.get("checked_invocations")
    if invocations is None:
        raise RandomAccessCampaignError("receipt chain requires verified invocation files")
    if len(receipts) > len(invocations):
        raise RandomAccessCampaignError("too many receipts")
    checked: list[dict[str, Any]] = []
    for index, raw in enumerate(receipts):
        invocation = invocations[index]
        item = require_receipt(
            raw,
            plan_sha256=checked_plan["plan_sha256"],
            invocation=invocation,
            verify_files=verify_files,
        )
        if index == 0:
            previous_boot = checked_plan["prepared_windows_boot"]
        else:
            previous_boot = checked[-1]["boot"]
        if not fresh_boot(item["boot"], previous_boot):
            raise RandomAccessCampaignError("each heavy invocation requires a fresh Windows boot")
        checkpoint = invocation["checkpoint"]
        if checkpoint["before_mode"] == "PREVIOUS_RECEIPT_AFTER":
            predecessor = checkpoint["predecessor_invocation_number"]
            prior = checked[predecessor - 1]
            if item["pointer_before_sha256"] != prior["pointer_after_sha256"]:
                raise RandomAccessCampaignError("checkpoint receipt chain mismatch")
        if invocation["checkpoint"]["write_mode"] == "READ_ONLY" and (
            item["pointer_after_sha256"] != item["pointer_before_sha256"]
        ):
            raise RandomAccessCampaignError("read-only checkpoint changed during VAL")
        checked.append(item)
    return checked


def next_action(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    current_boot: Mapping[str, Any],
) -> dict[str, Any]:
    checked_plan = (
        dict(plan)
        if isinstance(plan, Mapping) and "checked_invocations" in plan
        else require_plan(plan, verify_files=True)
    )
    checked_receipts = require_receipt_chain(checked_plan, receipts, verify_files=True)
    invocations = checked_plan["checked_invocations"]
    if checked_receipts and checked_receipts[-1]["outcome"] == "FAILED":
        return {"decision": "BLOCKED_FAILED_INVOCATION"}
    if len(checked_receipts) == len(invocations):
        return {"decision": "COMPLETE"}
    prior_boot = (
        checked_plan["prepared_windows_boot"]
        if not checked_receipts
        else checked_receipts[-1]["boot"]
    )
    if not fresh_boot(current_boot, prior_boot):
        return {
            "decision": "REBOOT_REQUIRED",
            "next_invocation_number": len(checked_receipts) + 1,
        }
    invocation = invocations[len(checked_receipts)]
    return {
        "decision": "LAUNCH",
        "invocation_number": invocation["invocation_number"],
        "invocation_id": invocation["invocation_id"],
        "kind": invocation["kind"],
        "invocation": invocation,
        "invocation_binding": checked_plan["invocations"][len(checked_receipts)],
    }


__all__ = [
    "ACTIVE_SCHEMA",
    "BOOT_SCHEMA",
    "INVOCATION_SCHEMA",
    "PLAN_SCHEMA",
    "PROGRESS_SCHEMA",
    "REBOOT_INTENT_SCHEMA",
    "REBOOT_RECEIPT_SCHEMA",
    "RECEIPT_SCHEMA",
    "STATUS_SCHEMA",
    "RandomAccessCampaignError",
    "canonical_bytes",
    "canonical_sha256",
    "file_sha256",
    "fresh_boot",
    "next_action",
    "read_bound_json",
    "require_binding",
    "require_boot_identity",
    "require_invocation",
    "require_plan",
    "require_clean_source",
    "require_receipt",
    "require_receipt_chain",
]
