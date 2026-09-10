"""Fail-closed lifecycle-v2 pilot campaign and reboot-resume decisions."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "gx1_local_lifecycle_v2_pilot_campaign_v1"
GATE_SCHEMA = "gx1_local_lifecycle_v2_full_launch_gate_v1"
INVOCATION_SCHEMA = "gx1_local_lifecycle_v2_cuda_invocation_v1"
RECEIPT_SCHEMA = "gx1_local_lifecycle_v2_invocation_receipt_v1"
_REQUIRED_GATE_CHECKS = frozenset({
    "clean_source_commit", "train_val_only", "test_sealed",
    "lifecycle_v2_admission", "economics_provider_chain",
    "checkpoint_resume", "canonical_capped_entrypoint",
    "signed_telemetry", "physical_160w_observed",
})
_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_UUID = re.compile(r"GPU-[0-9a-fA-F-]{36}")
_STAGE_ORDER = {"smoke": 0, "epoch1": 1}


class PilotCampaignError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise PilotCampaignError(f"{label} UTC timestamp required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PilotCampaignError(f"{label} UTC timestamp required") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise PilotCampaignError(f"{label} UTC timestamp required")
    return result


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise PilotCampaignError(f"{label} SHA-256 invalid")
    return value


def _read(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    if not path.is_absolute() or path.resolve() != path:
        raise PilotCampaignError("absolute normalized file required")
    try:
        info = path.lstat()
    except OSError as exc:
        raise PilotCampaignError(f"file unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
        raise PilotCampaignError(f"regular bounded file required: {path}")
    if expected_sha is not None and file_sha256(path) != expected_sha:
        raise PilotCampaignError(f"file SHA-256 mismatch: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotCampaignError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PilotCampaignError("JSON object required")
    return value


def _binding(value: Any, label: str, verify_files: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PilotCampaignError(f"{label} binding invalid")
    path = Path(str(value["path"] or ""))
    digest = _sha(value["sha256"], label)
    if not path.is_absolute() or path.resolve() != path:
        raise PilotCampaignError(f"{label} path invalid")
    if verify_files:
        _read(path, digest)
    return {"path": str(path), "sha256": digest}


def _command(invocation: Mapping[str, Any], repo: Path) -> None:
    command = invocation["launcher_command"]
    if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
        raise PilotCampaignError("launcher command invalid")
    runner = str(repo / "scripts/gx1_capped_run.sh")
    prefix = [runner, "--class", "trainer", "--mem", "20G", "--swap", "512M"]
    if command[:7] != prefix or command.count("--") != 1:
        raise PilotCampaignError("canonical capped runner required")
    if any(token == "--test" or token.lower() == "test" for token in command):
        raise PilotCampaignError("TEST token forbidden")
    separator = command.index("--")
    target = command[separator + 1 :]
    python = str(repo / ".venv/bin/python")
    if invocation["stage"] == "smoke":
        if "--attended-smoke" not in command[:separator] or target[:3] != [
            python, "-m", "gx1.scripts.attended_model_native_hardware_smoke_v1"
        ]:
            raise PilotCampaignError("short canonical smoke command required")
    else:
        if "--attended-smoke" in command[:separator] or target[:3] != [
            python, "-m", "gx1.models.entry_v10.entry_v10_ctx_train_v3"
        ] or target.count("--train") != 1:
            raise PilotCampaignError("canonical epoch1 trainer command required")


def require_invocation(
    value: Mapping[str, Any], *, stage: str, campaign_id: str,
    gate_sha256: str, source_repo: Path, source_commit: str,
) -> dict[str, Any]:
    keys = {"schema_version", "decision", "stage", "campaign_id", "full_launch_gate_sha256",
            "source_repo", "source_commit", "launcher_command", "progress_json",
            "pointer_path", "maximum_wall_seconds", "resume_allowed",
            "test_data_used", "invocation_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PilotCampaignError("invocation fields differ")
    result = dict(value)
    digest = _sha(result.pop("invocation_sha256"), "invocation")
    if digest != canonical_sha256(result):
        raise PilotCampaignError("invocation content digest mismatch")
    if (result["schema_version"] != INVOCATION_SCHEMA or result["decision"] != "PASS"
            or result["stage"] != stage or result["campaign_id"] != campaign_id
            or result["full_launch_gate_sha256"] != gate_sha256
            or result["source_repo"] != str(source_repo)
            or result["source_commit"] != source_commit
            or result["test_data_used"] is not False):
        raise PilotCampaignError("invocation identity differs")
    for key in ("progress_json", "pointer_path"):
        path = Path(str(result[key] or ""))
        if not path.is_absolute() or path.resolve() != path:
            raise PilotCampaignError(f"invocation {key} invalid")
    expected_resume = stage == "epoch1"
    limit = result["maximum_wall_seconds"]
    if type(limit) is not int or not 1 <= limit <= (600 if stage == "smoke" else 7200):
        raise PilotCampaignError("invocation wall bound invalid")
    if result["resume_allowed"] is not expected_resume:
        raise PilotCampaignError("invocation resume policy invalid")
    _command(result, source_repo)
    result["invocation_sha256"] = digest
    return result


def require_gate(
    value: Mapping[str, Any], *, campaign_id: str, source_commit: str,
) -> dict[str, Any]:
    keys = {"schema_version", "decision", "campaign_id", "source_commit", "checks",
            "cuda_execution_authorized", "test_data_used", "gate_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PilotCampaignError("launch gate fields differ")
    result = dict(value)
    digest = _sha(result.pop("gate_sha256"), "gate")
    checks = result.get("checks")
    if (digest != canonical_sha256(result) or result["schema_version"] != GATE_SCHEMA
            or result["decision"] != "PASS" or result["campaign_id"] != campaign_id
            or result["source_commit"] != source_commit
            or not isinstance(checks, Mapping) or set(checks) != _REQUIRED_GATE_CHECKS
            or any(value is not True for value in checks.values())
            or result["cuda_execution_authorized"] is not True
            or result["test_data_used"] is not False):
        raise PilotCampaignError("full launch gate is not PASS")
    result["gate_sha256"] = digest
    return result


def require_plan(value: Mapping[str, Any], verify_files: bool = True) -> dict[str, Any]:
    keys = {"schema_version", "decision", "campaign_id", "created_utc", "source_repo",
            "source_commit", "gpu_uuid", "runtime_root", "launch_gate",
            "invocations", "policy", "authority", "plan_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PilotCampaignError("campaign plan fields differ")
    result = dict(value)
    digest = _sha(result.pop("plan_sha256"), "plan")
    if digest != canonical_sha256(result):
        raise PilotCampaignError("campaign plan digest mismatch")
    repo = Path(str(result["source_repo"] or ""))
    runtime = Path(str(result["runtime_root"] or ""))
    if (result["schema_version"] != SCHEMA or result["decision"] != "PREPARED_GATE_REQUIRED"
            or not isinstance(result["campaign_id"], str) or not result["campaign_id"]
            or not repo.is_absolute() or repo.resolve() != repo or not repo.is_dir()
            or not _COMMIT.fullmatch(str(result["source_commit"]))
            or not _UUID.fullmatch(str(result["gpu_uuid"]))
            or not runtime.is_absolute() or runtime.resolve() != runtime
            or runtime.is_symlink() or not runtime.is_dir()):
        raise PilotCampaignError("campaign plan identity invalid")
    _utc(result["created_utc"], "created")
    policy = result["policy"]
    expected_policy = {
        "physical_power_limit_w": 160,
        "maximum_power_draw_w": 160,
        "automatic_power_limit_change": False,
        "local_telemetry_sample_seconds": 1,
        "human_status_cadence_seconds": 900,
        "meaningful_event_immediate": True,
        "smoke_must_complete_first": True,
        "reboot_between_smoke_and_epoch1": True,
        "reboot_before_each_additional_heavy_invocation": True,
    }
    authority = {
        "cuda_only_after_full_gate_pass": True, "test": False,
        "promotion": False, "paper": False, "live": False, "cloud_spend": False,
    }
    if policy != expected_policy or result["authority"] != authority:
        raise PilotCampaignError("campaign safety policy differs")
    gate_binding = _binding(result["launch_gate"], "launch gate", verify_files)
    invocations = result["invocations"]
    if not isinstance(invocations, Mapping) or set(invocations) != {"smoke", "epoch1"}:
        raise PilotCampaignError("campaign invocation set differs")
    invocation_bindings = {
        stage: _binding(invocations[stage], f"{stage} invocation", verify_files)
        for stage in ("smoke", "epoch1")
    }
    if verify_files:
        gate = require_gate(
            _read(Path(gate_binding["path"]), gate_binding["sha256"]),
            campaign_id=result["campaign_id"], source_commit=result["source_commit"],
        )
        for stage, binding in invocation_bindings.items():
            require_invocation(
                _read(Path(binding["path"]), binding["sha256"]), stage=stage,
                campaign_id=result["campaign_id"], gate_sha256=gate["gate_sha256"],
                source_repo=repo, source_commit=result["source_commit"],
            )
    result["launch_gate"] = gate_binding
    result["invocations"] = invocation_bindings
    result["plan_sha256"] = digest
    return result


def require_clean_source(plan: Mapping[str, Any]) -> None:
    repo = str(plan["source_repo"])
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain=v1", "--untracked-files=all"],
                               check=True, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise PilotCampaignError("source state unavailable") from exc
    if head != plan["source_commit"] or dirty:
        raise PilotCampaignError("source must be exact clean committed state")


def require_receipt(value: Mapping[str, Any], *, plan_sha256: str) -> dict[str, Any]:
    keys = {"schema_version", "plan_sha256", "invocation_number", "stage", "boot_utc",
            "outcome", "pointer_before_sha256", "pointer_after_sha256",
            "telemetry_jsonl", "telemetry_sha256", "telemetry_sample_count",
            "maximum_observed_power_limit_w", "maximum_observed_power_draw_w",
            "test_data_used", "receipt_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PilotCampaignError("receipt fields differ")
    result = dict(value)
    digest = _sha(result.pop("receipt_sha256"), "receipt")
    if digest != canonical_sha256(result) or result["schema_version"] != RECEIPT_SCHEMA:
        raise PilotCampaignError("receipt digest/schema invalid")
    if result["plan_sha256"] != plan_sha256 or result["stage"] not in _STAGE_ORDER:
        raise PilotCampaignError("receipt campaign/stage invalid")
    if type(result["invocation_number"]) is not int or result["invocation_number"] < 1:
        raise PilotCampaignError("receipt invocation invalid")
    _utc(result["boot_utc"], "receipt boot")
    if result["outcome"] not in {"COMPLETE", "RESUMABLE", "FAILED"}:
        raise PilotCampaignError("receipt outcome invalid")
    for key in ("pointer_before_sha256", "pointer_after_sha256", "telemetry_sha256"):
        _sha(result[key], key)
    telemetry_path = Path(str(result["telemetry_jsonl"] or ""))
    if not telemetry_path.is_absolute() or file_sha256(telemetry_path) != result["telemetry_sha256"]:
        raise PilotCampaignError("receipt telemetry binding invalid")
    if type(result["telemetry_sample_count"]) is not int or result["telemetry_sample_count"] < 1:
        raise PilotCampaignError("receipt telemetry count invalid")
    power_breach = False
    for key in ("maximum_observed_power_limit_w", "maximum_observed_power_draw_w"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or result[key] < 0:
            raise PilotCampaignError("receipt power telemetry invalid")
        power_breach = power_breach or result[key] > 160
    if power_breach and result["outcome"] != "FAILED":
        raise PilotCampaignError("receipt exceeds 160 W without failed outcome")
    if result["test_data_used"] is not False:
        raise PilotCampaignError("receipt accessed TEST")
    result["receipt_sha256"] = digest
    return result


def next_action(plan: Mapping[str, Any], receipts: list[Mapping[str, Any]], *, current_boot_utc: str) -> dict[str, Any]:
    checked = require_plan(plan)
    boot = _utc(current_boot_utc, "current boot")
    verified: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts, 1):
        item = require_receipt(receipt, plan_sha256=checked["plan_sha256"])
        if item["invocation_number"] != index:
            raise PilotCampaignError("receipt sequence is not contiguous")
        expected_stage = "smoke" if index == 1 else "epoch1"
        if item["stage"] != expected_stage:
            raise PilotCampaignError("smoke must be the first and only pre-epoch invocation")
        if verified and _utc(item["boot_utc"], "receipt boot") <= _utc(verified[-1]["boot_utc"], "prior boot"):
            raise PilotCampaignError("each heavy invocation requires a new Windows boot")
        verified.append(item)
    if not verified:
        return {"decision": "LAUNCH_SMOKE", "stage": "smoke", "invocation_number": 1}
    last = verified[-1]
    if last["outcome"] == "FAILED":
        return {"decision": "BLOCKED", "stage": last["stage"], "invocation_number": len(verified)}
    if boot <= _utc(last["boot_utc"], "last heavy boot"):
        return {"decision": "REBOOT_REQUIRED", "stage": "epoch1", "invocation_number": len(verified) + 1}
    if last["stage"] == "smoke":
        return {"decision": "LAUNCH_EPOCH1", "stage": "epoch1", "invocation_number": len(verified) + 1}
    if last["outcome"] == "RESUMABLE":
        return {"decision": "RESUME_EPOCH1", "stage": "epoch1", "invocation_number": len(verified) + 1}
    return {"decision": "COMPLETE", "stage": "epoch1", "invocation_number": len(verified)}


__all__ = [
    "GATE_SCHEMA", "INVOCATION_SCHEMA", "PilotCampaignError", "RECEIPT_SCHEMA",
    "SCHEMA", "canonical_bytes", "canonical_sha256", "file_sha256", "next_action",
    "require_clean_source", "require_gate", "require_invocation", "require_plan",
    "require_receipt",
]
