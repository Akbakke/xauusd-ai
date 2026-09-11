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
EXECUTION_MANIFEST_SCHEMA = "gx1_local_random_access_execution_manifest_v2"
BOOT_SCHEMA = "gx1_windows_boot_identity_v1"
ACTIVE_SCHEMA = "gx1_local_random_access_active_invocation_v2"
RECEIPT_SCHEMA = "gx1_local_random_access_invocation_receipt_v2"
REBOOT_INTENT_SCHEMA = "gx1_local_random_access_reboot_intent_v1"
REBOOT_RECEIPT_SCHEMA = "gx1_local_random_access_reboot_request_receipt_v1"
PROGRESS_SCHEMA = "gx1_local_random_access_progress_v2"
STATUS_SCHEMA = "gx1_local_random_access_status_v2"

_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_KINDS = {
    "smoke_arm",
    "reference_run",
    "resume_proof_first",
    "resume_proof_second",
    "epoch1_window",
    "full_val_window",
}
_SUCCESS_OUTCOMES = {"COMPLETE", "RESUMABLE"}
_EXPECTED_SUCCESS_OUTCOMES = _SUCCESS_OUTCOMES | {"RESUMABLE_OR_COMPLETE"}


class RandomAccessCampaignError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
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
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{8,}(?:Z|\+00:00)",
        value,
    ):
        raise RandomAccessCampaignError(f"{label} UTC timestamp invalid")
    # PowerShell/.NET round-trip format emits seven fractional digits (100 ns),
    # while Python 3.10 accepts at most six. Preserve the original string for
    # identity hashing and truncate only the value used for time comparison.
    match = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{7})(Z|\+00:00)",
        value,
    )
    parse_value = (
        f"{match.group(1)}.{match.group(2)[:6]}{match.group(3)}"
        if match is not None
        else value
    )
    try:
        result = datetime.fromisoformat(parse_value.replace("Z", "+00:00"))
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
    if (
        "--mem" not in outer
        or "20G" not in outer
        or "--swap" not in outer
        or "512M" not in outer
    ):
        raise RandomAccessCampaignError("canonical memory/swap envelope required")
    attended = "--attended-smoke" in outer
    if attended != (
        kind
        in {"smoke_arm", "reference_run", "resume_proof_first", "resume_proof_second"}
    ):
        raise RandomAccessCampaignError(
            "attended-smoke scope differs from invocation kind"
        )
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
    if (
        type(number) is not int
        or number < 1
        or result.get("invocation_id") != f"invocation-{number:04d}"
    ):
        raise RandomAccessCampaignError("invocation number invalid")
    batch_size = result.get("batch_size")
    if batch_size not in (4, 8, 16):
        raise RandomAccessCampaignError("invocation batch size invalid")
    for name in ("epoch_index", "window_index", "window_count"):
        if type(result.get(name)) is not int or result[name] < 0:
            raise RandomAccessCampaignError(f"invocation {name} invalid")
    budget = result.get("optimizer_step_budget")
    if kind == "full_val_window":
        if (
            budget is not None
            or result["expected_success_outcome"] != "RESUMABLE_OR_COMPLETE"
            or result["window_count"] < 1
        ):
            raise RandomAccessCampaignError("full VAL window budget/outcome invalid")
    elif type(budget) is not int or budget < 1:
        raise RandomAccessCampaignError("optimizer-step budget invalid")
    if result["expected_success_outcome"] not in _EXPECTED_SUCCESS_OUTCOMES:
        raise RandomAccessCampaignError("expected outcome invalid")
    maximum_wall = result.get("maximum_wall_seconds")
    if type(maximum_wall) is not int or not 1 <= maximum_wall <= 7200:
        raise RandomAccessCampaignError("invocation wall bound invalid")
    result["execution_manifest"] = require_binding(
        result["execution_manifest"],
        label="execution manifest",
        verify_file=verify_files,
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
    if before_mode in {"GENESIS", "FINAL_AUTHORITY"}:
        if predecessor is not None:
            raise RandomAccessCampaignError(
                f"{before_mode} checkpoint predecessor forbidden"
            )
        if before_mode == "FINAL_AUTHORITY" and kind != "full_val_window":
            raise RandomAccessCampaignError("FINAL_AUTHORITY limited to full VAL")
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
    if verify_files:
        execution = read_bound_json(
            Path(result["execution_manifest"]["path"]),
            result["execution_manifest"]["sha256"],
        )
        unsigned = {
            key: item for key, item in execution.items() if key != "artifact_sha256"
        }
        separator = result["launcher_argv"].index("--")
        target_module = result["launcher_argv"][separator + 3]
        if (
            set(execution)
            != {
                "schema_version",
                "decision",
                "invocation_kind",
                "python_module",
                "source_commit",
                "launcher_argv_sha256",
                "prelaunch_manifest",
                "prelaunch_manifest_sha256",
                "train_session_manifest",
                "train_session_manifest_sha256",
                "test_data_used",
                "artifact_sha256",
            }
            or execution.get("schema_version") != EXECUTION_MANIFEST_SCHEMA
            or execution.get("decision") != "PASS_EXACT_INVOCATION"
            or execution.get("invocation_kind") != kind
            or execution.get("python_module") != target_module
            or execution.get("source_commit") != source_commit
            or execution.get("launcher_argv_sha256") != result["launcher_argv_sha256"]
            or execution.get("test_data_used") is not False
            or execution.get("artifact_sha256") != canonical_sha256(unsigned)
        ):
            raise RandomAccessCampaignError("execution manifest binding invalid")
        if kind == "full_val_window":
            prelaunch_binding = require_binding(
                execution["prelaunch_manifest"],
                label="full VAL launch manifest",
                verify_file=True,
            )
            prelaunch = read_bound_json(
                Path(prelaunch_binding["path"]), prelaunch_binding["sha256"]
            )
            if (
                prelaunch.get("manifest_sha256")
                != _sha(
                    execution["prelaunch_manifest_sha256"],
                    "full VAL launch manifest artifact",
                )
                or execution["train_session_manifest"] is not None
                or execution["train_session_manifest_sha256"] is not None
                or result["launcher_argv"].count("--launch-manifest") != 1
                or result["launcher_argv"][
                    result["launcher_argv"].index("--launch-manifest") + 1
                ]
                != prelaunch_binding["path"]
            ):
                raise RandomAccessCampaignError("full VAL launch binding invalid")
            if result["launcher_argv"].count("--rollout-progress-path") != 1:
                raise RandomAccessCampaignError("full VAL rollout cursor missing")
            result["rollout_cursor_path"] = str(
                _absolute(
                    result["launcher_argv"][
                        result["launcher_argv"].index("--rollout-progress-path") + 1
                    ],
                    "full VAL rollout cursor",
                )
            )
        else:
            prelaunch_binding = require_binding(
                execution["prelaunch_manifest"],
                label="prelaunch manifest",
                verify_file=True,
            )
            prelaunch = read_bound_json(
                Path(prelaunch_binding["path"]), prelaunch_binding["sha256"]
            )
            prelaunch_sha = _sha(
                execution["prelaunch_manifest_sha256"],
                "prelaunch manifest artifact",
            )
            if (
                prelaunch.get("manifest_sha256") != prelaunch_sha
                or "--launch-manifest" not in result["launcher_argv"]
                or result["launcher_argv"][
                    result["launcher_argv"].index("--launch-manifest") + 1
                ]
                != prelaunch_binding["path"]
            ):
                raise RandomAccessCampaignError("prelaunch manifest provenance invalid")
            if kind == "smoke_arm":
                if (
                    execution["train_session_manifest"] is not None
                    or execution["train_session_manifest_sha256"] is not None
                    or "--train-session-manifest" in result["launcher_argv"]
                ):
                    raise RandomAccessCampaignError(
                        "smoke arm train-session binding forbidden"
                    )
            else:
                session_binding = require_binding(
                    execution["train_session_manifest"],
                    label="train session manifest",
                    verify_file=True,
                )
                session = read_bound_json(
                    Path(session_binding["path"]), session_binding["sha256"]
                )
                session_sha = _sha(
                    execution["train_session_manifest_sha256"],
                    "train session manifest artifact",
                )
                if (
                    session.get("manifest_sha256") != session_sha
                    or "--train-session-manifest" not in result["launcher_argv"]
                    or result["launcher_argv"][
                        result["launcher_argv"].index("--train-session-manifest") + 1
                    ]
                    != session_binding["path"]
                ):
                    raise RandomAccessCampaignError(
                        "train session manifest provenance invalid"
                    )
    result["invocation_sha256"] = claimed
    return result


def _require_sequence(
    invocations: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    selected_batch_size: int | None,
    expected_epoch_optimizer_steps: int | None = None,
) -> None:
    if phase == "gpu_selection":
        if len(invocations) != 3:
            raise RandomAccessCampaignError("GPU-selection invocation sequence invalid")
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
    elif phase in {"resume_proof", "selected_training"}:
        if selected_batch_size not in (4, 8, 16) or not invocations:
            raise RandomAccessCampaignError("selected training sequence invalid")
        has_proof = invocations[0]["kind"] == "reference_run"
        if phase == "resume_proof" and (not has_proof or len(invocations) != 3):
            raise RandomAccessCampaignError("resume-proof sequence invalid")
        if has_proof:
            if len(invocations) < 3:
                raise RandomAccessCampaignError("resume-proof sequence incomplete")
            reference, first, second = invocations[:3]
            if (
                reference["kind"] != "reference_run"
                or reference["batch_size"] != selected_batch_size
                or reference["optimizer_step_budget"] != 4
                or reference["checkpoint"]["before_mode"] != "GENESIS"
                or reference["expected_success_outcome"] != "COMPLETE"
            ):
                raise RandomAccessCampaignError("selected reference-run sequence invalid")
            if (
                first["kind"] != "resume_proof_first"
                or first["batch_size"] != selected_batch_size
                or first["optimizer_step_budget"] != 3
                or first["checkpoint"]["before_mode"] != "GENESIS"
                or first["expected_success_outcome"] != "COMPLETE"
            ):
                raise RandomAccessCampaignError("resume-proof first sequence invalid")
            if (
                second["kind"] != "resume_proof_second"
                or second["batch_size"] != selected_batch_size
                or second["optimizer_step_budget"] != 1
                or second["checkpoint"]["before_mode"] != "PREVIOUS_RECEIPT_AFTER"
                or second["checkpoint"]["predecessor_invocation_number"]
                != first["invocation_number"]
                or second["checkpoint"]["pointer_path"]
                != first["checkpoint"]["pointer_path"]
                or second["expected_success_outcome"] != "COMPLETE"
            ):
                raise RandomAccessCampaignError("resume-proof second sequence invalid")
        if phase == "resume_proof":
            return
        epoch = list(invocations[3:] if has_proof else invocations)
        if not epoch or any(item["kind"] != "epoch1_window" for item in epoch):
            raise RandomAccessCampaignError("epoch1 window sequence invalid")
        count = len(epoch)
        expected_total_batches = (
            -(-16384 // selected_batch_size)
            if expected_epoch_optimizer_steps is None else expected_epoch_optimizer_steps
        )
        if (
            sum(int(item["optimizer_step_budget"]) for item in epoch)
            != expected_total_batches
        ):
            raise RandomAccessCampaignError("epoch1 optimizer-step coverage invalid")
        for offset, item in enumerate(epoch):
            if (
                item["batch_size"] != selected_batch_size
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
    elif phase == "full_val":
        if selected_batch_size not in (4, 8, 16) or not invocations:
            raise RandomAccessCampaignError("full VAL sequence invalid")
        count = len(invocations)
        pointer_path = invocations[0]["checkpoint"]["pointer_path"]
        rollout_cursor_path = invocations[0].get("rollout_cursor_path")
        for offset, item in enumerate(invocations):
            expected_mode = (
                "FINAL_AUTHORITY" if offset == 0 else "PREVIOUS_RECEIPT_AFTER"
            )
            if (
                item["kind"] != "full_val_window"
                or item["batch_size"] != selected_batch_size
                or item["optimizer_step_budget"] is not None
                or item["epoch_index"] != 0
                or item["window_index"] != offset
                or item["window_count"] != count
                or item["expected_success_outcome"] != "RESUMABLE_OR_COMPLETE"
                or item["checkpoint"]["before_mode"] != expected_mode
                or item["checkpoint"]["write_mode"] != "READ_ONLY"
                or item["checkpoint"]["pointer_path"] != pointer_path
                or item.get("rollout_cursor_path") != rollout_cursor_path
            ):
                raise RandomAccessCampaignError("full VAL window identity invalid")
            if offset > 0 and (
                item["checkpoint"]["predecessor_invocation_number"]
                != invocations[offset - 1]["invocation_number"]
            ):
                raise RandomAccessCampaignError("full VAL window predecessor invalid")
    else:
        raise RandomAccessCampaignError("campaign phase invalid")
    for number, item in enumerate(invocations, 1):
        if item["invocation_number"] != number:
            raise RandomAccessCampaignError("invocation sequence is not contiguous")



def _full_population_session_for_plan(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    from gx1.contracts.unified_exit_full_population_train_session_v1 import (
        require_full_population_train_session,
    )
    def bound_json(value: Any, label: str) -> tuple[dict[str, Any], dict[str, str]]:
        binding = require_binding(value, label=label, verify_file=True)
        return read_bound_json(Path(binding["path"]), binding["sha256"]), binding
    if not isinstance(plan["invocations"], list) or not plan["invocations"]:
        raise RandomAccessCampaignError("full-year invocation missing")
    invocation, _ = bound_json(plan["invocations"][0], "full-year invocation")
    execution, _ = bound_json(invocation["execution_manifest"], "full-year execution")
    raw_session, binding = bound_json(execution["train_session_manifest"], "full-year session")
    try:
        session = require_full_population_train_session(raw_session, verify_files=True)
    except (RuntimeError, KeyError, TypeError, ValueError, OSError) as exc:
        raise RandomAccessCampaignError("full-year session unavailable or invalid") from exc
    authority, _ = bound_json(session["prefix_checkpoint_authority"], "prefix authority")
    if (
        session["source_commit"] != plan["source_commit"]
        or session["source_repo"] != plan["source_repo"]
        or session["entry_pairs_per_epoch"] != plan["entry_pairs_per_epoch"]
        or session["transition_budget_per_epoch"] != plan["transitions_per_epoch"]
        or session["selected_batch_size"] != plan["selected_batch_size"]
        or session["gpu_batch_selection"] != plan["selection_receipt"]
        or execution["train_session_manifest_sha256"] != session["manifest_sha256"]
        or invocation["kind"] != "epoch1_window"
    ):
        raise RandomAccessCampaignError("full-year session provenance invalid")
    return session, binding, authority["campaign_plan"]

def require_plan(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    keys = {
        "schema_version",
        "decision",
        "campaign_id",
        "phase",
        "created_utc",
        "source_repo",
        "source_commit",
        "runtime_root",
        "gpu_uuid",
        "prepared_windows_boot",
        "prior_campaign",
        "selection_receipt",
        "final_train_checkpoint_authority",
        "selected_batch_size",
        "entry_pairs_per_epoch",
        "transitions_per_epoch",
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
        or result.get("phase") not in {"gpu_selection", "resume_proof", "selected_training", "full_val"}
        or type(result.get("entry_pairs_per_epoch")) is not int
        or result["entry_pairs_per_epoch"] < 1
        or type(result.get("transitions_per_epoch")) is not int
        or result["transitions_per_epoch"] != 4 * result["entry_pairs_per_epoch"]
        or (result.get("phase") not in {"selected_training", "full_val"} and result["entry_pairs_per_epoch"] != 16384)
        or not isinstance(result.get("source_commit"), str)
        or _COMMIT.fullmatch(result["source_commit"]) is None
        or not isinstance(result.get("gpu_uuid"), str)
        or not result["gpu_uuid"].startswith("GPU-")
        or result.get("test_data_used") is not False
        or claimed != canonical_sha256(result)
    ):
        raise RandomAccessCampaignError("campaign plan identity invalid")
    _utc(result["created_utc"], "campaign creation")
    result["prepared_windows_boot"] = require_boot_identity(
        result["prepared_windows_boot"]
    )
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
    if (
        result.get("policy") != expected_policy
        or result.get("authority") != expected_authority
    ):
        raise RandomAccessCampaignError("campaign policy differs")
    guard = result.get("signed_guard_sources")
    controllers = result.get("controller_sources")
    if not isinstance(guard, Mapping) or set(guard) != {
        "runner",
        "guard",
        "query",
        "certificate",
    }:
        raise RandomAccessCampaignError("signed guard source set invalid")
    if not isinstance(controllers, Mapping) or set(controllers) != {
        "controller",
        "observer",
        "campaign_cli",
    }:
        raise RandomAccessCampaignError("controller source set invalid")
    result["signed_guard_sources"] = {
        name: require_binding(binding, label=f"guard {name}", verify_file=verify_files)
        for name, binding in guard.items()
    }
    result["controller_sources"] = {
        name: require_binding(
            binding, label=f"controller {name}", verify_file=verify_files
        )
        for name, binding in controllers.items()
    }
    phase = result["phase"]
    full_session = None
    full_session_binding = None
    prefix_campaign = None
    if verify_files and phase == "selected_training" and result["entry_pairs_per_epoch"] != 16384:
        full_session, full_session_binding, prefix_campaign = _full_population_session_for_plan(result)
    if phase == "gpu_selection":
        if (
            result.get("prior_campaign") is not None
            or result.get("selection_receipt") is not None
            or result.get("selected_batch_size") is not None
            or result.get("final_train_checkpoint_authority") is not None
        ):
            raise RandomAccessCampaignError(
                "GPU-selection plan cannot preselect a winner"
            )
    else:
        selected = result.get("selected_batch_size")
        if selected not in (4, 8, 16):
            raise RandomAccessCampaignError("selected batch size invalid")
        prior_label = (
            "prior GPU-selection campaign"
            if phase in {"resume_proof", "selected_training"}
            else "prior selected-training campaign"
        )
        prior_binding = require_binding(
            result.get("prior_campaign"),
            label=prior_label,
            verify_file=verify_files,
        )
        selection_binding = require_binding(
            result.get("selection_receipt"),
            label="GPU batch selection receipt",
            verify_file=verify_files,
        )
        if verify_files:
            prior = require_plan(
                read_bound_json(Path(prior_binding["path"]), prior_binding["sha256"]),
                verify_files=True,
            )
            selection = read_bound_json(
                Path(selection_binding["path"]), selection_binding["sha256"]
            )
            try:
                from gx1.contracts.unified_exit_gpu_batch_selection_v1 import (
                    require_selection,
                )

                selection = require_selection(selection, verify_files=True)
            except (ImportError, RuntimeError) as exc:
                raise RandomAccessCampaignError(
                    "canonical GPU batch selection unavailable or invalid"
                ) from exc
            expected_prior_phases = (
                {"selected_training"} if full_session is not None
                else {"gpu_selection"} if phase == "resume_proof"
                else {"gpu_selection", "resume_proof"} if phase == "selected_training"
                else {"selected_training"}
            )
            if (
                prior["phase"] not in expected_prior_phases
                or (phase != "full_val" and prior["source_commit"] != (
                    result["source_commit"] if full_session is None
                    else full_session["predecessor_source_commit"]
                ))
                or (full_session is not None and prior_binding != prefix_campaign)
                or selection.get("selected_batch_size") != selected
                or selection.get("entry_pairs_per_epoch") != 16384
                or selection.get("transition_budget_per_epoch") != 65536
                or selection.get("total_batches_per_epoch") != -(-16384 // selected)
                or selection.get("test_data_used") is not False
            ):
                raise RandomAccessCampaignError(f"{phase} campaign provenance invalid")
        result["prior_campaign"] = prior_binding
        result["selection_receipt"] = selection_binding
        result["selection_artifact_sha256"] = selection["artifact_sha256"]
        if phase in {"resume_proof", "selected_training"}:
            if result.get("final_train_checkpoint_authority") is not None:
                raise RandomAccessCampaignError(
                    "selected training final authority forbidden"
                )
        else:
            authority_binding = require_binding(
                result.get("final_train_checkpoint_authority"),
                label="final TRAIN checkpoint authority",
                verify_file=verify_files,
            )
            if verify_files:
                try:
                    from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
                        require_final_train_checkpoint_authority,
                        FULL_POPULATION_SCHEMA_VERSION,
                    )

                    authority = require_final_train_checkpoint_authority(
                        read_bound_json(
                            Path(authority_binding["path"]),
                            authority_binding["sha256"],
                        ),
                        verify_files=True,
                    )
                except (ImportError, RuntimeError) as exc:
                    raise RandomAccessCampaignError(
                        "final TRAIN checkpoint authority unavailable or invalid"
                    ) from exc
                if (
                    authority["source_commit"] != prior["source_commit"]
                    or (
                        authority["schema_version"] != FULL_POPULATION_SCHEMA_VERSION
                        and authority["source_commit"] != result["source_commit"]
                    )
                    or authority["entry_pair_count"] != result["entry_pairs_per_epoch"]
                    or authority["transition_count"] != result["transitions_per_epoch"]
                    or authority["campaign_plan"] != prior_binding
                    or authority["gpu_batch_selection"] != selection_binding
                    or authority["selected_batch_size"] != selected
                    or authority["gpu_batch_selection_artifact_sha256"]
                    != selection["artifact_sha256"]
                ):
                    raise RandomAccessCampaignError(
                        "full VAL final authority provenance invalid"
                    )
            result["final_train_checkpoint_authority"] = authority_binding
            if verify_files:
                result["checked_final_train_checkpoint_authority"] = authority
    if phase == "gpu_selection":
        result["selection_artifact_sha256"] = None
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
        _require_sequence(
            invocations,
            phase=result["phase"],
            selected_batch_size=result["selected_batch_size"],
            expected_epoch_optimizer_steps=(
                None if full_session is None else full_session["remaining_optimizer_steps"]
            ),
        )
        if full_session is not None:
            for item in invocations:
                execution = read_bound_json(
                    Path(item["execution_manifest"]["path"]), item["execution_manifest"]["sha256"]
                )
                if (
                    execution["train_session_manifest"] != full_session_binding
                    or item["checkpoint"]["pointer_path"] != str(Path(full_session["checkpoint_dir"]) / "RESUME_POINTER.json")
                    or item["progress_path"] != str(Path(full_session["checkpoint_dir"]) / "PROGRESS.json")
                ):
                    raise RandomAccessCampaignError("full-year window session differs")
            result["checked_full_population_session"] = full_session
        if phase == "selected_training":
            epoch_only = invocations[0]["kind"] == "epoch1_window"
            if epoch_only != (prior["phase"] == "resume_proof" or full_session is not None):
                raise RandomAccessCampaignError("training/proof campaign order invalid")
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


def require_progress(
    value: Any,
    *,
    plan_sha256: str,
    invocation: Mapping[str, Any],
    expected_selection_receipt_sha256: str | None,
    verify_file: bool = True,
) -> dict[str, Any]:
    keys = {
        "schema_version",
        "plan_sha256",
        "invocation_sha256",
        "phase",
        "epoch_index",
        "global_optimizer_steps",
        "next_batch_offset",
        "total_batches",
        "completed_units",
        "total_units",
        "epoch_schedule_sha256",
        "selection_receipt_sha256",
        "checkpoint_pointer",
        "terminal",
        "outcome",
        "observed_utc",
        "progress_sha256",
    }
    is_full_val = invocation.get("kind") == "full_val_window"
    if is_full_val:
        keys.add("rollout_cursor")
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RandomAccessCampaignError("progress fields differ")
    result = dict(value)
    claimed = _sha(result.pop("progress_sha256"), "progress")
    selection = result.get("selection_receipt_sha256")
    if selection is not None:
        _sha(selection, "selection receipt")
    if selection != expected_selection_receipt_sha256:
        raise RandomAccessCampaignError("progress selection receipt differs")
    if (
        result.get("schema_version") != PROGRESS_SCHEMA
        or result.get("plan_sha256") != plan_sha256
        or result.get("invocation_sha256") != invocation["invocation_sha256"]
        or result.get("phase") != invocation["kind"]
        or result.get("epoch_index") != invocation["epoch_index"]
        or result.get("terminal") is not True
        or result.get("outcome") not in {"COMPLETE", "RESUMABLE", "FAILED"}
        or claimed != canonical_sha256(result)
    ):
        raise RandomAccessCampaignError("progress identity invalid")
    _utc(result.get("observed_utc"), "progress")
    for name in (
        "global_optimizer_steps",
        "next_batch_offset",
        "total_batches",
        "completed_units",
        "total_units",
    ):
        if type(result.get(name)) is not int or result[name] < 0:
            raise RandomAccessCampaignError("progress counter invalid")
    if (
        result["total_batches"] < 1
        or result["total_units"] < 1
        or result["next_batch_offset"] > result["total_batches"]
        or result["completed_units"] > result["total_units"]
    ):
        raise RandomAccessCampaignError("progress range invalid")
    _sha(result.get("epoch_schedule_sha256"), "epoch schedule")
    result["checkpoint_pointer"] = require_binding(
        result["checkpoint_pointer"],
        label="progress checkpoint pointer",
        verify_file=verify_file,
    )
    if result["checkpoint_pointer"]["path"] != invocation["checkpoint"]["pointer_path"]:
        raise RandomAccessCampaignError("progress checkpoint pointer path differs")
    if is_full_val:
        result["rollout_cursor"] = require_binding(
            result["rollout_cursor"],
            label="full VAL rollout cursor",
            verify_file=verify_file,
        )
        if result["rollout_cursor"]["path"] != invocation.get("rollout_cursor_path"):
            raise RandomAccessCampaignError("progress rollout cursor path differs")
    result["progress_sha256"] = claimed
    return result


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
        "selection_receipt_sha256",
        "boot",
        "started_utc",
        "finished_utc",
        "outcome",
        "trainer_guard_exit_code",
        "progress_observer_exit_code",
        "pointer_before_sha256",
        "checkpoint_pointer_after",
        "checkpoint_pointer_snapshot",
        "progress",
        "guard_log",
        "guard_decision",
        "signed_guard_telemetry_owner",
        "active_marker_sha256",
        "test_data_used",
        "receipt_sha256",
    }
    is_full_val = invocation.get("kind") == "full_val_window"
    if is_full_val:
        keys.update(
            {
                "rollout_cursor_before_sha256",
                "rollout_cursor_after",
                "rollout_cursor_snapshot",
            }
        )
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
    selection = result.get("selection_receipt_sha256")
    if selection is not None:
        _sha(selection, "receipt selection")
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
    else:
        expected = invocation["expected_success_outcome"]
        expected_matches = (
            outcome in _SUCCESS_OUTCOMES
            if expected == "RESUMABLE_OR_COMPLETE"
            else outcome == expected
        )
        if not expected_matches or not success:
            raise RandomAccessCampaignError("receipt outcome/exit codes invalid")
    before = result.get("pointer_before_sha256")
    if invocation["checkpoint"]["before_mode"] == "GENESIS":
        if before != "GENESIS":
            raise RandomAccessCampaignError("GENESIS pointer-before invalid")
    else:
        _sha(before, "pointer before")
    # The logical after-binding keeps the canonical live pointer path for
    # resume/final-authority consumers.  It is intentionally not rehashed:
    # later invocations atomically replace that file.  The immutable snapshot
    # below is the byte evidence for this receipt.
    result["checkpoint_pointer_after"] = require_binding(
        result["checkpoint_pointer_after"],
        label="checkpoint pointer after",
        verify_file=False,
    )
    if (
        result["checkpoint_pointer_after"]["path"]
        != invocation["checkpoint"]["pointer_path"]
    ):
        raise RandomAccessCampaignError(
            "checkpoint pointer path differs from invocation"
        )
    result["checkpoint_pointer_snapshot"] = require_binding(
        result["checkpoint_pointer_snapshot"],
        label="checkpoint pointer snapshot",
        verify_file=verify_files,
    )
    if (
        result["checkpoint_pointer_snapshot"]["sha256"]
        != result["checkpoint_pointer_after"]["sha256"]
    ):
        raise RandomAccessCampaignError("checkpoint pointer snapshot differs")
    _sha(result.get("active_marker_sha256"), "active marker")
    expected_guard_decision = "PASS" if success and outcome != "FAILED" else "FAILED"
    if (
        result.get("guard_decision") != expected_guard_decision
        or result.get("signed_guard_telemetry_owner") != "gx1_guarded_trainer_exec.sh"
    ):
        raise RandomAccessCampaignError("signed guard decision invalid")
    result["progress"] = require_binding(
        result["progress"], label="progress receipt", verify_file=verify_files
    )
    if verify_files:
        # The immutable progress bytes bind the logical live pointer and its
        # digest. Rehashing that mutable path would invalidate older receipts
        # after a legitimate continuation.
        progress = require_progress(
            read_bound_json(
                Path(result["progress"]["path"]), result["progress"]["sha256"]
            ),
            plan_sha256=plan_sha256,
            invocation=invocation,
            expected_selection_receipt_sha256=selection,
            verify_file=False,
        )
        if progress["checkpoint_pointer"] != result["checkpoint_pointer_after"]:
            raise RandomAccessCampaignError("progress checkpoint evidence differs")
    if is_full_val:
        before_cursor = result.get("rollout_cursor_before_sha256")
        if before_cursor != "GENESIS":
            _sha(before_cursor, "rollout cursor before")
        result["rollout_cursor_after"] = require_binding(
            result["rollout_cursor_after"],
            label="rollout cursor after",
            verify_file=False,
        )
        result["rollout_cursor_snapshot"] = require_binding(
            result["rollout_cursor_snapshot"],
            label="rollout cursor snapshot",
            verify_file=verify_files,
        )
        if (
            result["rollout_cursor_after"]["path"]
            != invocation.get("rollout_cursor_path")
            or result["rollout_cursor_after"]["sha256"]
            != result["rollout_cursor_snapshot"]["sha256"]
        ):
            raise RandomAccessCampaignError("rollout cursor evidence differs")
        if (
            verify_files
            and progress["rollout_cursor"] != result["rollout_cursor_after"]
        ):
            raise RandomAccessCampaignError("progress rollout cursor differs")
    result["guard_log"] = require_binding(
        result["guard_log"], label="signed guard log", verify_file=verify_files
    )
    snapshot_root = Path(result["checkpoint_pointer_snapshot"]["path"]).parent
    expected_snapshot_root_name = f"invocation-{invocation['invocation_number']:04d}"
    if (
        snapshot_root.name != expected_snapshot_root_name
        or Path(result["checkpoint_pointer_snapshot"]["path"]).name
        != "CHECKPOINT_POINTER.json"
        or Path(result["progress"]["path"]) != snapshot_root / "PROGRESS.json"
        or Path(result["guard_log"]["path"]) != snapshot_root / "SIGNED_GUARD.log"
        or result["checkpoint_pointer_snapshot"]["path"]
        == result["checkpoint_pointer_after"]["path"]
        or (
            is_full_val
            and (
                Path(result["rollout_cursor_snapshot"]["path"])
                != snapshot_root / "ROLLOUT_CURSOR.json"
                or result["rollout_cursor_snapshot"]["path"]
                == result["rollout_cursor_after"]["path"]
            )
        )
    ):
        raise RandomAccessCampaignError("immutable invocation evidence layout invalid")
    result["receipt_sha256"] = claimed
    return result


def require_receipt_chain(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    verify_files: bool = True,
) -> list[dict[str, Any]]:
    checked_plan = (
        dict(plan)
        if isinstance(plan, Mapping) and "checked_invocations" in plan
        else require_plan(plan, verify_files=verify_files)
    )
    invocations = checked_plan.get("checked_invocations")
    if invocations is None:
        raise RandomAccessCampaignError(
            "receipt chain requires verified invocation files"
        )
    if len(receipts) > len(invocations):
        raise RandomAccessCampaignError("too many receipts")
    checked: list[dict[str, Any]] = []
    for index, raw in enumerate(receipts):
        if (
            checked_plan["phase"] == "full_val"
            and checked
            and checked[-1]["outcome"] == "COMPLETE"
        ):
            raise RandomAccessCampaignError(
                "receipt after terminal full VAL completion forbidden"
            )
        invocation = invocations[index]
        item = require_receipt(
            raw,
            plan_sha256=checked_plan["plan_sha256"],
            invocation=invocation,
            verify_files=verify_files,
        )
        if (
            item["selection_receipt_sha256"]
            != checked_plan["selection_artifact_sha256"]
        ):
            raise RandomAccessCampaignError("receipt selection differs from campaign")
        if index == 0:
            previous_boot = checked_plan["prepared_windows_boot"]
        else:
            previous_boot = checked[-1]["boot"]
        if not fresh_boot(item["boot"], previous_boot):
            raise RandomAccessCampaignError(
                "each heavy invocation requires a fresh Windows boot"
            )
        checkpoint = invocation["checkpoint"]
        if checkpoint["before_mode"] == "FINAL_AUTHORITY":
            authority = checked_plan.get("checked_final_train_checkpoint_authority")
            if (
                not isinstance(authority, Mapping)
                or item["pointer_before_sha256"]
                != authority["final_checkpoint_pointer"]["sha256"]
                or item["rollout_cursor_before_sha256"] != "GENESIS"
            ):
                raise RandomAccessCampaignError(
                    "initial full VAL authority/cursor binding differs"
                )
        elif checkpoint["before_mode"] == "PREVIOUS_RECEIPT_AFTER":
            predecessor = checkpoint["predecessor_invocation_number"]
            prior = checked[predecessor - 1]
            if (
                item["pointer_before_sha256"]
                != prior["checkpoint_pointer_after"]["sha256"]
            ):
                raise RandomAccessCampaignError("checkpoint receipt chain mismatch")
            if invocation["kind"] == "full_val_window" and (
                item["rollout_cursor_before_sha256"]
                != prior["rollout_cursor_snapshot"]["sha256"]
            ):
                raise RandomAccessCampaignError("rollout cursor chain mismatch")
        if invocation["checkpoint"]["write_mode"] == "READ_ONLY" and (
            item["checkpoint_pointer_after"]["sha256"] != item["pointer_before_sha256"]
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
    if (
        checked_plan["phase"] == "full_val"
        and checked_receipts
        and checked_receipts[-1]["outcome"] == "COMPLETE"
    ):
        return {"decision": "COMPLETE"}
    if len(checked_receipts) == len(invocations):
        if checked_plan["phase"] == "full_val":
            return {"decision": "BLOCKED_VAL_WINDOWS_EXHAUSTED"}
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
    "EXECUTION_MANIFEST_SCHEMA",
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
    "require_progress",
    "require_clean_source",
    "require_receipt",
    "require_receipt_chain",
]
