"""Exact host attestation for one guarded GX1 Hopper training VM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "gx1_cloud_training_host_profile_v1"
DECISION = "PASS_HOST_QUALIFIED_NOT_TRAINING_AUTHORITY"
TECHNICAL_SCOPE = "offline_train_val_only_no_test_no_broker_v1"
TELEMETRY_OWNER = "root_owned_linux_loopback_v1"
HARD_DEADLINE_SECONDS = 48 * 60 * 60
ADMISSION_MAX_PROJECTED_SECONDS = 43 * 60 * 60 + 12 * 60
HARD_COST_CAP_NOK = 2500.0
GIB = 1024**3

PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "created_utc",
        "technical_scope",
        "source",
        "provider",
        "host",
        "gpu",
        "telemetry",
        "limits",
        "budget",
        "termination",
        "activation_authority",
        "report_only",
        "side_effects_started",
    }
)
SOURCE_KEYS = frozenset({"repo_root", "commit", "clean"})
PROVIDER_KEYS = frozenset(
    {"name", "region", "instance_type", "instance_id", "hourly_price_usd"}
)
HOST_KEYS = frozenset(
    {
        "machine_id_sha256",
        "kernel_release",
        "os_id",
        "os_version_id",
        "logical_cpu_count",
        "memory_total_bytes",
        "systemd_available",
        "cgroup_v2",
        "scratch_path",
        "scratch_free_bytes",
    }
)
GPU_KEYS = frozenset(
    {
        "uuid",
        "name",
        "driver_version",
        "compute_capability",
        "memory_total_mib",
        "bf16_supported",
        "torch_version",
        "torch_cuda_version",
    }
)
TELEMETRY_KEYS = frozenset(
    {
        "owner",
        "url",
        "certificate_path",
        "certificate_sha256",
        "service_unit",
        "service_executable_path",
        "service_executable_sha256",
        "timeout_seconds",
    }
)
LIMIT_KEYS = frozenset(
    {
        "memory_max_bytes",
        "memory_swap_max_bytes",
        "tasks_max",
        "cpu_affinity",
        "numerical_threads",
        "max_wall_seconds",
        "max_core_temp_c",
        "max_memory_temp_c",
        "max_power_limit_w",
        "max_power_draw_w",
        "max_memory_used_mib",
        "monitor_interval_seconds",
        "minimum_available_memory_bytes",
        "minimum_scratch_free_bytes",
    }
)
BUDGET_KEYS = frozenset(
    {
        "hard_cost_cap_nok",
        "hard_deadline_seconds",
        "admission_max_projected_seconds",
        "nok_per_usd",
        "fx_observed_utc",
        "cost_buffer_fraction",
    }
)
TERMINATION_KEYS = frozenset(
    {
        "provider_managed_delete",
        "provider_deadline_utc",
        "provider_proof_path",
        "provider_proof_sha256",
        "local_timer_unit",
        "local_timer_active",
    }
)
PROVIDER_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "provider",
        "instance_id",
        "deadline_utc",
        "deadline_epoch",
        "delete_command_path",
        "delete_command_sha256",
        "provider_managed_delete",
        "hard_cost_cap_nok",
        "local_timer_unit",
    }
)
SIDE_EFFECTS_ZERO = {
    "training": False,
    "test_read": False,
    "replay": False,
    "paper": False,
    "live": False,
}
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_LOOPBACK_URL_RE = re.compile(
    r"^http://127\.0\.0\.1:[1-9][0-9]{0,4}/gx1/v1/telemetry/$"
)
_AFFINITY_RE = re.compile(r"^[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*$")


class CloudTrainingHostProfileError(RuntimeError):
    """The cloud host profile or its live attestation is invalid."""


def _mapping(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != keys:
        raise CloudTrainingHostProfileError(f"{label}: keys are not exact")
    return value


def _text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        raise CloudTrainingHostProfileError(f"{label}: expected nonempty text")
    return value


def _absolute(value: Any, label: str) -> Path:
    raw = str(value) if isinstance(value, (str, Path)) else ""
    _text(raw, label)
    path = Path(raw)
    if (
        not path.is_absolute()
        or str(path) != raw
        or any(part in {".", ".."} for part in path.parts)
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise CloudTrainingHostProfileError(f"{label}: expected canonical absolute path")
    return path


def _sha(value: Any, label: str) -> str:
    text = str(value or "")
    if _SHA_RE.fullmatch(text) is None:
        raise CloudTrainingHostProfileError(f"{label}: expected lowercase SHA-256")
    return text


def _integer(value: Any, label: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CloudTrainingHostProfileError(f"{label}: invalid integer")
    if maximum is not None and value > maximum:
        raise CloudTrainingHostProfileError(f"{label}: integer exceeds ceiling")
    return value


def _number(
    value: Any,
    label: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CloudTrainingHostProfileError(f"{label}: invalid number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise CloudTrainingHostProfileError(f"{label}: invalid number")
    if maximum is not None and number > maximum:
        raise CloudTrainingHostProfileError(f"{label}: number exceeds ceiling")
    return number


def _affinity_cpu_count(value: str) -> int:
    if _AFFINITY_RE.fullmatch(value) is None:
        raise CloudTrainingHostProfileError("limits.cpu_affinity: invalid syntax")
    selected: set[int] = set()
    for part in value.split(","):
        start_text, separator, end_text = part.partition("-")
        start = int(start_text)
        end = int(end_text) if separator else start
        if end < start or end > 4095:
            raise CloudTrainingHostProfileError("limits.cpu_affinity: invalid range")
        current = set(range(start, end + 1))
        if selected.intersection(current):
            raise CloudTrainingHostProfileError("limits.cpu_affinity: overlapping range")
        selected.update(current)
    return len(selected)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_cloud_training_host_profile(value: Any) -> dict[str, Any]:
    """Validate one exact, report-only Hopper host profile without I/O."""

    profile = _mapping(value, PROFILE_KEYS, "cloud host profile")
    created_utc = str(profile.get("created_utc") or "")
    if (
        profile.get("schema_version") != SCHEMA_VERSION
        or profile.get("decision") != DECISION
        or profile.get("technical_scope") != TECHNICAL_SCOPE
        or profile.get("activation_authority") is not False
        or profile.get("report_only") is not True
        or profile.get("side_effects_started") != SIDE_EFFECTS_ZERO
        or _UTC_RE.fullmatch(created_utc) is None
    ):
        raise CloudTrainingHostProfileError("cloud host safety boundary invalid")

    source = _mapping(profile.get("source"), SOURCE_KEYS, "source")
    repo_root = _absolute(source.get("repo_root"), "source.repo_root")
    if _GIT_SHA_RE.fullmatch(str(source.get("commit") or "")) is None:
        raise CloudTrainingHostProfileError("source.commit: invalid Git commit")
    if source.get("clean") is not True:
        raise CloudTrainingHostProfileError("source.clean: dirty source forbidden")

    provider = _mapping(profile.get("provider"), PROVIDER_KEYS, "provider")
    for key in ("name", "region", "instance_type", "instance_id"):
        _text(provider.get(key), f"provider.{key}")
    hourly_price_usd = _number(
        provider.get("hourly_price_usd"), "provider.hourly_price_usd", 0.01, 100.0
    )

    host = _mapping(profile.get("host"), HOST_KEYS, "host")
    _sha(host.get("machine_id_sha256"), "host.machine_id_sha256")
    _text(host.get("kernel_release"), "host.kernel_release")
    _text(host.get("os_id"), "host.os_id")
    _text(host.get("os_version_id"), "host.os_version_id")
    logical_cpu_count = _integer(
        host.get("logical_cpu_count"), "host.logical_cpu_count", 8, 4096
    )
    memory_total_bytes = _integer(
        host.get("memory_total_bytes"), "host.memory_total_bytes", 64 * GIB
    )
    if host.get("systemd_available") is not True or host.get("cgroup_v2") is not True:
        raise CloudTrainingHostProfileError("host requires systemd and cgroup v2")
    scratch_path = _absolute(host.get("scratch_path"), "host.scratch_path")
    scratch_free_bytes = _integer(
        host.get("scratch_free_bytes"), "host.scratch_free_bytes", 100 * GIB
    )

    gpu = _mapping(profile.get("gpu"), GPU_KEYS, "gpu")
    if _GPU_UUID_RE.fullmatch(str(gpu.get("uuid") or "")) is None:
        raise CloudTrainingHostProfileError("gpu.uuid: malformed")
    gpu_name = _text(gpu.get("name"), "gpu.name")
    if "H100" not in gpu_name and "H200" not in gpu_name:
        raise CloudTrainingHostProfileError("gpu.name: only NVIDIA H100/H200 admitted")
    _text(gpu.get("driver_version"), "gpu.driver_version")
    if gpu.get("compute_capability") != [9, 0] or gpu.get("bf16_supported") is not True:
        raise CloudTrainingHostProfileError("gpu: Hopper BF16 capability missing")
    gpu_memory_total_mib = _integer(
        gpu.get("memory_total_mib"), "gpu.memory_total_mib", 78 * 1024, 256 * 1024
    )
    _text(gpu.get("torch_version"), "gpu.torch_version")
    _text(gpu.get("torch_cuda_version"), "gpu.torch_cuda_version")

    telemetry = _mapping(profile.get("telemetry"), TELEMETRY_KEYS, "telemetry")
    if telemetry.get("owner") != TELEMETRY_OWNER:
        raise CloudTrainingHostProfileError("telemetry.owner: invalid")
    if _LOOPBACK_URL_RE.fullmatch(str(telemetry.get("url") or "")) is None:
        raise CloudTrainingHostProfileError("telemetry.url: loopback endpoint required")
    certificate_path = _absolute(
        telemetry.get("certificate_path"), "telemetry.certificate_path"
    )
    certificate_sha256 = _sha(
        telemetry.get("certificate_sha256"), "telemetry.certificate_sha256"
    )
    if telemetry.get("service_unit") != "gx1-host-telemetry.service":
        raise CloudTrainingHostProfileError("telemetry.service_unit: invalid")
    service_path = _absolute(
        telemetry.get("service_executable_path"),
        "telemetry.service_executable_path",
    )
    service_sha256 = _sha(
        telemetry.get("service_executable_sha256"),
        "telemetry.service_executable_sha256",
    )
    timeout_seconds = _integer(
        telemetry.get("timeout_seconds"), "telemetry.timeout_seconds", 1, 5
    )

    limits = _mapping(profile.get("limits"), LIMIT_KEYS, "limits")
    memory_max_bytes = _integer(
        limits.get("memory_max_bytes"), "limits.memory_max_bytes", 32 * GIB
    )
    if memory_max_bytes > memory_total_bytes - 16 * GIB:
        raise CloudTrainingHostProfileError("limits.memory_max_bytes: host reserve too small")
    memory_swap_max_bytes = _integer(
        limits.get("memory_swap_max_bytes"),
        "limits.memory_swap_max_bytes",
        1024,
        8 * GIB,
    )
    if memory_max_bytes % 1024 != 0 or memory_swap_max_bytes % 1024 != 0:
        raise CloudTrainingHostProfileError(
            "limits memory and swap ceilings must be exact KiB multiples"
        )
    tasks_max = _integer(limits.get("tasks_max"), "limits.tasks_max", 128, 4096)
    affinity = _text(limits.get("cpu_affinity"), "limits.cpu_affinity")
    affinity_count = _affinity_cpu_count(affinity)
    if affinity_count > logical_cpu_count:
        raise CloudTrainingHostProfileError("limits.cpu_affinity: exceeds host CPUs")
    numerical_threads = _integer(
        limits.get("numerical_threads"), "limits.numerical_threads", 1, affinity_count
    )
    if limits.get("max_wall_seconds") != HARD_DEADLINE_SECONDS:
        raise CloudTrainingHostProfileError("limits.max_wall_seconds: must be hard 48h")
    max_core_temp = _number(
        limits.get("max_core_temp_c"), "limits.max_core_temp_c", 40.0, 85.0
    )
    max_memory_temp = _number(
        limits.get("max_memory_temp_c"), "limits.max_memory_temp_c", 40.0, 100.0
    )
    max_power_limit = _number(
        limits.get("max_power_limit_w"), "limits.max_power_limit_w", 100.0, 1000.0
    )
    max_power_draw = _number(
        limits.get("max_power_draw_w"), "limits.max_power_draw_w", max_power_limit, 1100.0
    )
    max_memory_used_mib = _integer(
        limits.get("max_memory_used_mib"),
        "limits.max_memory_used_mib",
        32 * 1024,
        gpu_memory_total_mib - 4096,
    )
    if limits.get("monitor_interval_seconds") != 1:
        raise CloudTrainingHostProfileError("limits.monitor_interval_seconds: must be one")
    minimum_available_memory_bytes = _integer(
        limits.get("minimum_available_memory_bytes"),
        "limits.minimum_available_memory_bytes",
        16 * GIB,
        memory_total_bytes,
    )
    minimum_scratch_free_bytes = _integer(
        limits.get("minimum_scratch_free_bytes"),
        "limits.minimum_scratch_free_bytes",
        100 * GIB,
        scratch_free_bytes,
    )

    budget = _mapping(profile.get("budget"), BUDGET_KEYS, "budget")
    if float(budget.get("hard_cost_cap_nok", -1)) != HARD_COST_CAP_NOK:
        raise CloudTrainingHostProfileError("budget.hard_cost_cap_nok: must be 2500")
    if budget.get("hard_deadline_seconds") != HARD_DEADLINE_SECONDS:
        raise CloudTrainingHostProfileError("budget.hard_deadline_seconds: must be 48h")
    if budget.get("admission_max_projected_seconds") != ADMISSION_MAX_PROJECTED_SECONDS:
        raise CloudTrainingHostProfileError(
            "budget.admission_max_projected_seconds: must retain 10% time reserve"
        )
    nok_per_usd = _number(budget.get("nok_per_usd"), "budget.nok_per_usd", 5.0, 25.0)
    if _UTC_RE.fullmatch(str(budget.get("fx_observed_utc") or "")) is None:
        raise CloudTrainingHostProfileError("budget.fx_observed_utc: invalid")
    if float(budget.get("cost_buffer_fraction", -1)) != 0.10:
        raise CloudTrainingHostProfileError("budget.cost_buffer_fraction: must be 0.10")
    worst_case_cost_nok = (
        hourly_price_usd
        * (HARD_DEADLINE_SECONDS / 3600.0)
        * nok_per_usd
        * 1.10
    )
    if worst_case_cost_nok > HARD_COST_CAP_NOK:
        raise CloudTrainingHostProfileError(
            "provider price can exceed the hard NOK cap before the 48h deadline"
        )

    termination = _mapping(
        profile.get("termination"), TERMINATION_KEYS, "termination"
    )
    if termination.get("provider_managed_delete") is not True:
        raise CloudTrainingHostProfileError("termination: provider delete is mandatory")
    provider_deadline_utc = str(termination.get("provider_deadline_utc") or "")
    if _UTC_RE.fullmatch(provider_deadline_utc) is None:
        raise CloudTrainingHostProfileError("termination.provider_deadline_utc: invalid")
    created_at = datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    deadline_at = datetime.strptime(
        provider_deadline_utc, "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    deadline_seconds = int((deadline_at - created_at).total_seconds())
    if deadline_seconds <= 0 or deadline_seconds > HARD_DEADLINE_SECONDS:
        raise CloudTrainingHostProfileError(
            "termination.provider_deadline_utc: must be within 48h"
        )
    provider_proof_path = _absolute(
        termination.get("provider_proof_path"), "termination.provider_proof_path"
    )
    provider_proof_sha256 = _sha(
        termination.get("provider_proof_sha256"),
        "termination.provider_proof_sha256",
    )
    if termination.get("local_timer_unit") != "gx1-cloud-deadline.timer":
        raise CloudTrainingHostProfileError("termination.local_timer_unit: invalid")
    if termination.get("local_timer_active") is not True:
        raise CloudTrainingHostProfileError("termination: local deadline timer inactive")

    normalized = json.loads(json.dumps(profile, sort_keys=True, allow_nan=False))
    normalized["source"]["repo_root"] = str(repo_root)
    normalized["host"]["scratch_path"] = str(scratch_path)
    normalized["telemetry"]["certificate_path"] = str(certificate_path)
    normalized["telemetry"]["certificate_sha256"] = certificate_sha256
    normalized["telemetry"]["service_executable_path"] = str(service_path)
    normalized["telemetry"]["service_executable_sha256"] = service_sha256
    normalized["telemetry"]["timeout_seconds"] = timeout_seconds
    normalized["limits"]["memory_max_bytes"] = memory_max_bytes
    normalized["limits"]["memory_swap_max_bytes"] = memory_swap_max_bytes
    normalized["limits"]["tasks_max"] = tasks_max
    normalized["limits"]["numerical_threads"] = numerical_threads
    normalized["limits"]["max_core_temp_c"] = max_core_temp
    normalized["limits"]["max_memory_temp_c"] = max_memory_temp
    normalized["limits"]["max_power_limit_w"] = max_power_limit
    normalized["limits"]["max_power_draw_w"] = max_power_draw
    normalized["limits"]["max_memory_used_mib"] = max_memory_used_mib
    normalized["limits"]["minimum_available_memory_bytes"] = minimum_available_memory_bytes
    normalized["limits"]["minimum_scratch_free_bytes"] = minimum_scratch_free_bytes
    normalized["termination"]["provider_proof_path"] = str(provider_proof_path)
    normalized["termination"]["provider_proof_sha256"] = provider_proof_sha256
    return normalized


def _regular_root_owned(path: Path, expected_sha256: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise CloudTrainingHostProfileError(f"{label}: unavailable regular file")
    mode = path.stat()
    if mode.st_uid != 0 or mode.st_gid != 0 or mode.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise CloudTrainingHostProfileError(f"{label}: must be root-owned and immutable to users")
    if sha256_file(path) != expected_sha256:
        raise CloudTrainingHostProfileError(f"{label}: SHA-256 mismatch")


def _run(command: list[str], label: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CloudTrainingHostProfileError(f"{label}: command failed") from exc
    return result.stdout.strip()


def require_cloud_training_host_profile_file(
    path: Path,
    expected_sha256: str,
    *,
    repo: Path,
    verify_runtime: bool = True,
) -> dict[str, Any]:
    """Load a hash-bound profile and optionally re-attest the live host."""

    profile_path = _absolute(path, "cloud host profile path")
    expected = _sha(expected_sha256, "cloud host profile SHA-256")
    if profile_path.is_symlink() or not profile_path.is_file() or profile_path.resolve() != profile_path:
        raise CloudTrainingHostProfileError("cloud host profile must be a regular file")
    if sha256_file(profile_path) != expected:
        raise CloudTrainingHostProfileError("cloud host profile SHA-256 mismatch")
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise CloudTrainingHostProfileError("cloud host profile JSON invalid") from exc
    profile = require_cloud_training_host_profile(payload)
    if not verify_runtime:
        return profile

    repo_root = repo.resolve(strict=True)
    if str(repo_root) != profile["source"]["repo_root"]:
        raise CloudTrainingHostProfileError("live repository root differs from host profile")
    if _run(["/usr/bin/git", "-C", str(repo_root), "rev-parse", "HEAD"], "Git HEAD") != profile["source"]["commit"]:
        raise CloudTrainingHostProfileError("live source commit differs from host profile")
    if _run(
        ["/usr/bin/git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        "Git status",
    ):
        raise CloudTrainingHostProfileError("live source tree is dirty")
    machine_id = Path("/etc/machine-id").read_bytes().strip()
    if hashlib.sha256(machine_id).hexdigest() != profile["host"]["machine_id_sha256"]:
        raise CloudTrainingHostProfileError("live machine identity differs from host profile")
    if os.uname().release != profile["host"]["kernel_release"]:
        raise CloudTrainingHostProfileError("live kernel differs from host profile")
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        raise CloudTrainingHostProfileError("live cgroup v2 unavailable")
    os_release: dict[str, str] = {}
    for row in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        key, separator, raw = row.partition("=")
        if separator:
            os_release[key] = raw.strip().strip('"')
    if (
        os_release.get("ID") != profile["host"]["os_id"]
        or os_release.get("VERSION_ID") != profile["host"]["os_version_id"]
    ):
        raise CloudTrainingHostProfileError("live OS identity differs from host profile")
    if os.cpu_count() != profile["host"]["logical_cpu_count"]:
        raise CloudTrainingHostProfileError("live CPU count differs from host profile")
    memory_rows = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    memory = {
        row.split(":", 1)[0]: int(row.split()[1]) * 1024
        for row in memory_rows
        if row.startswith(("MemTotal:", "MemAvailable:"))
    }
    if memory.get("MemTotal") != profile["host"]["memory_total_bytes"]:
        raise CloudTrainingHostProfileError("live memory total differs from host profile")
    if memory.get("MemAvailable", 0) < profile["limits"]["minimum_available_memory_bytes"]:
        raise CloudTrainingHostProfileError("live available memory is below host profile")
    scratch_path = Path(profile["host"]["scratch_path"])
    if not scratch_path.is_dir() or scratch_path.is_symlink() or scratch_path.resolve() != scratch_path:
        raise CloudTrainingHostProfileError("live scratch path is unavailable")
    if os.statvfs(scratch_path).f_bavail * os.statvfs(scratch_path).f_frsize < profile["limits"]["minimum_scratch_free_bytes"]:
        raise CloudTrainingHostProfileError("live scratch capacity is below host profile")

    telemetry = profile["telemetry"]
    _regular_root_owned(
        Path(telemetry["certificate_path"]),
        telemetry["certificate_sha256"],
        "telemetry certificate",
    )
    _regular_root_owned(
        Path(telemetry["service_executable_path"]),
        telemetry["service_executable_sha256"],
        "telemetry service executable",
    )
    proof = profile["termination"]
    _regular_root_owned(
        Path(proof["provider_proof_path"]),
        proof["provider_proof_sha256"],
        "provider termination proof",
    )
    try:
        proof_payload = json.loads(
            Path(proof["provider_proof_path"]).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise CloudTrainingHostProfileError(
            "provider termination proof is invalid JSON"
        ) from exc
    proof_payload = _mapping(
        proof_payload, PROVIDER_PROOF_KEYS, "provider termination proof"
    )
    expected_deadline_epoch = int(
        datetime.strptime(
            profile["termination"]["provider_deadline_utc"],
            "%Y-%m-%dT%H:%M:%SZ",
        )
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    if (
        proof_payload.get("schema_version")
        != "gx1_cloud_provider_termination_proof_v1"
        or proof_payload.get("provider") != profile["provider"]["name"]
        or proof_payload.get("instance_id") != profile["provider"]["instance_id"]
        or proof_payload.get("deadline_utc")
        != profile["termination"]["provider_deadline_utc"]
        or proof_payload.get("deadline_epoch") != expected_deadline_epoch
        or proof_payload.get("provider_managed_delete") is not True
        or float(proof_payload.get("hard_cost_cap_nok", -1)) != HARD_COST_CAP_NOK
        or proof_payload.get("local_timer_unit")
        != profile["termination"]["local_timer_unit"]
    ):
        raise CloudTrainingHostProfileError(
            "provider termination proof differs from host profile"
        )
    if datetime.now(timezone.utc).timestamp() >= expected_deadline_epoch:
        raise CloudTrainingHostProfileError("provider termination deadline has expired")
    delete_command_path = _absolute(
        proof_payload.get("delete_command_path"),
        "provider termination delete command",
    )
    delete_command_sha256 = _sha(
        proof_payload.get("delete_command_sha256"),
        "provider termination delete command sha256",
    )
    _regular_root_owned(
        delete_command_path,
        delete_command_sha256,
        "provider termination delete command",
    )
    if _run(
        ["/usr/bin/systemctl", "is-active", telemetry["service_unit"]],
        "telemetry systemd service",
    ) != "active":
        raise CloudTrainingHostProfileError("telemetry systemd service is not active")
    service_user = _run(
        [
            "/usr/bin/systemctl",
            "show",
            telemetry["service_unit"],
            "--property=User",
            "--value",
        ],
        "telemetry systemd user",
    )
    if service_user != "root":
        raise CloudTrainingHostProfileError("telemetry systemd service is not root-owned")
    fragment_value = _run(
        [
            "/usr/bin/systemctl",
            "show",
            telemetry["service_unit"],
            "--property=FragmentPath",
            "--value",
        ],
        "telemetry systemd unit path",
    )
    if not fragment_value.startswith("/"):
        raise CloudTrainingHostProfileError("telemetry systemd unit path is unavailable")
    fragment_path = Path(fragment_value)
    if fragment_path.is_symlink() or not fragment_path.is_file() or fragment_path.stat().st_uid != 0:
        raise CloudTrainingHostProfileError("telemetry systemd unit is not root-owned")
    if _run(
        ["/usr/bin/systemctl", "is-active", proof["local_timer_unit"]],
        "deadline systemd timer",
    ) != "active":
        raise CloudTrainingHostProfileError("deadline systemd timer is not active")
    timer_fragment_value = _run(
        [
            "/usr/bin/systemctl",
            "show",
            proof["local_timer_unit"],
            "--property=FragmentPath",
            "--value",
        ],
        "deadline systemd timer path",
    )
    timer_fragment_path = Path(timer_fragment_value)
    if (
        not timer_fragment_value.startswith("/")
        or timer_fragment_path.is_symlink()
        or not timer_fragment_path.is_file()
        or timer_fragment_path.stat().st_uid != 0
    ):
        raise CloudTrainingHostProfileError("deadline systemd timer is not root-owned")

    gpu = profile["gpu"]
    gpu_row = _run(
        [
            "/usr/bin/nvidia-smi",
            f"--id={gpu['uuid']}",
            "--query-gpu=uuid,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        "NVIDIA identity",
    )
    rows = [row.strip() for row in gpu_row.splitlines() if row.strip()]
    if len(rows) != 1:
        raise CloudTrainingHostProfileError("live NVIDIA identity is not singular")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 5:
        raise CloudTrainingHostProfileError("live NVIDIA identity row malformed")
    live_uuid, live_name, live_driver, live_memory, live_capability = fields
    if (
        live_uuid != gpu["uuid"]
        or live_name != gpu["name"]
        or live_driver != gpu["driver_version"]
        or int(float(live_memory)) != gpu["memory_total_mib"]
        or live_capability != "9.0"
    ):
        raise CloudTrainingHostProfileError("live NVIDIA identity differs from host profile")
    query_path = repo_root / "scripts" / "gx1_host_telemetry_bridge_query.sh"
    _run(
        [
            str(query_path),
            telemetry["url"],
            telemetry["certificate_path"],
            telemetry["certificate_sha256"],
            gpu["uuid"],
            str(telemetry["timeout_seconds"]),
        ],
        "signed host telemetry",
        timeout=10,
    )
    return profile


def runner_fields(profile: Mapping[str, Any]) -> dict[str, str]:
    """Return the only profile values the capped runner may consume."""

    validated = require_cloud_training_host_profile(profile)
    limits = validated["limits"]
    telemetry = validated["telemetry"]
    gpu = validated["gpu"]
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu["uuid"]),
        "CPU_AFFINITY": str(limits["cpu_affinity"]),
        "NUMERICAL_THREAD_COUNT": str(limits["numerical_threads"]),
        "SAFE_JOB_MEMORY_KIB": str(int(limits["memory_max_bytes"]) // 1024),
        "SAFE_JOB_SWAP_KIB": str(int(limits["memory_swap_max_bytes"]) // 1024),
        "MIN_HOST_MEMORY_KIB": str(int(validated["host"]["memory_total_bytes"]) // 1024),
        "MIN_AVAILABLE_MEMORY_KIB": str(int(limits["minimum_available_memory_bytes"]) // 1024),
        "TASKS_MAX": str(limits["tasks_max"]),
        "TRAINER_TASKS_MAX": str(limits["tasks_max"]),
        "TRAINER_MAX_WALL_SECONDS": str(limits["max_wall_seconds"]),
        "TRAINER_MODEL_MAX_WALL_SECONDS": str(limits["max_wall_seconds"]),
        "TRAINER_GPU_MAX_CORE_TEMP_C": str(limits["max_core_temp_c"]),
        "TRAINER_GPU_MAX_MEMORY_TEMP_C": str(limits["max_memory_temp_c"]),
        "TRAINER_GPU_MAX_POWER_LIMIT_W": str(limits["max_power_limit_w"]),
        "TRAINER_GPU_MAX_POWER_DRAW_W": str(limits["max_power_draw_w"]),
        "TRAINER_GPU_MAX_MEMORY_USED_MIB": str(limits["max_memory_used_mib"]),
        "TRAINER_GPU_MONITOR_INTERVAL_SECONDS": str(limits["monitor_interval_seconds"]),
        "TRAINER_HOST_TELEMETRY_URL": str(telemetry["url"]),
        "TRAINER_HOST_TELEMETRY_CERT_PATH": str(telemetry["certificate_path"]),
        "TRAINER_HOST_TELEMETRY_CERT_SHA256": str(telemetry["certificate_sha256"]),
        "TRAINER_HOST_TELEMETRY_GPU_UUID": str(gpu["uuid"]),
        "TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS": str(telemetry["timeout_seconds"]),
        "TRAINER_TELEMETRY_OWNER": TELEMETRY_OWNER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--emit-runner-fields", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.emit_runner_fields == args.metadata_only:
        parser.error("choose exactly one output mode")
    try:
        profile = require_cloud_training_host_profile_file(
            args.profile_json,
            args.profile_sha256,
            repo=args.repo,
            verify_runtime=not args.metadata_only,
        )
    except (CloudTrainingHostProfileError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.emit_runner_fields:
        for key, value in sorted(runner_fields(profile).items()):
            print(f"{key}\t{value}")
    else:
        print(json.dumps(profile, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
