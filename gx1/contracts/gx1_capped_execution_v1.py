"""Fail-closed execution proofs for capped GX1 entry points.

The shell runner owns process containment and the telemetry guard. A Python
module that can allocate CUDA must nevertheless prove that it is already
inside that protected cgroup before it asks PyTorch about a device.
CPU audits additionally verify kernel-reported affinity and runner thread limits.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


_MAX_MEMORY_BYTES = 20 * 1024**3
_MAX_AUDIT_MEMORY_BYTES = 4 * 1024**3
_MAX_SWAP_BYTES = 512 * 1024**2
_MAX_PIDS = 64
_LIMIT_ENV = {
    "memory": "GX1_CAPPED_MEMORY_BYTES",
    "swap": "GX1_CAPPED_SWAP_BYTES",
    "pids": "GX1_CAPPED_TASKS_MAX",
}
_LOCK_PROC_MAX_BYTES = 65536
_LOCK_MAX_ANCESTORS = 128
# Operator decision 2026-09-28: independent projects may run concurrently.
# Keep one CURRENT job and the same kernel/cgroup proof; never unlink a live lock.
_PROJECT_LOCK_FILENAME = "gx1-current-heavy-job.lock"


def _canonical_lock_state(*, required: bool) -> tuple[Path, os.stat_result | None]:
    user_id = os.getuid()
    if os.geteuid() != user_id:
        raise ValueError("real and effective UID differ")
    runtime = Path(f"/run/user/{user_id}")
    for directory in (Path("/"), Path("/run"), Path("/run/user"), runtime):
        metadata = directory.lstat()
        expected_uid = user_id if directory == runtime else 0
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != expected_uid
            or metadata.st_mode & 0o7022
            or (directory == runtime and stat.S_IMODE(metadata.st_mode) != 0o700)
        ):
            raise ValueError(f"unsafe canonical lock directory: {directory}")
    lock_path = runtime / _PROJECT_LOCK_FILENAME
    try:
        metadata = lock_path.lstat()
    except FileNotFoundError:
        if required:
            raise
        return lock_path, None
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != user_id
        or metadata.st_mode & 0o7022
        or metadata.st_nlink != 1
    ):
        raise ValueError("unsafe canonical lock file")
    return lock_path, metadata


def canonical_heavy_job_lock_path() -> Path:
    """Use CURRENT's protected per-UID lock, never ambient XDG or /tmp."""

    try:
        return _canonical_lock_state(required=False)[0]
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"[GX1_CAPPED_LOCK_PATH_INVALID] {exc}") from exc


def _read_lock_proc(path: Path) -> bytes:
    with path.open("rb") as stream:
        contents = stream.read(_LOCK_PROC_MAX_BYTES + 1)
    if len(contents) > _LOCK_PROC_MAX_BYTES:
        raise ValueError(f"oversized kernel lock metadata: {path}")
    return contents


def _canonical_runner_process(process_dir: Path, runner_path: Path) -> bool:
    executable = Path(os.readlink(process_dir / "exe"))
    if executable != Path("/bin/bash").resolve(strict=True):
        return False
    command = _read_lock_proc(process_dir / "cmdline")
    if not command.endswith(b"\0"):
        raise ValueError("unterminated runner command line")
    arguments = [os.fsdecode(value) for value in command[:-1].split(b"\0")]
    argument_index = 1
    while argument_index < len(arguments):
        argument = arguments[argument_index]
        if argument == "--":
            argument_index += 1
            break
        if argument in {"--noprofile", "--norc"} or re.fullmatch(r"-[euxv]+", argument):
            argument_index += 1
            continue
        if argument.startswith(("-", "+")):
            return False
        break
    if argument_index >= len(arguments) or not arguments[argument_index]:
        return False
    script = Path(arguments[argument_index])
    relative_operand = not script.is_absolute()
    if relative_operand:
        script = Path(os.readlink(process_dir / "cwd")) / script
    try:
        return script.resolve(strict=True) == runner_path
    except FileNotFoundError:
        if not relative_operand or script.name == runner_path.name:
            raise
        descriptor = process_dir / "fd/255"
        opened = descriptor.stat()
        target = Path(os.readlink(descriptor))
        if not stat.S_ISREG(opened.st_mode) or not target.is_absolute():
            raise ValueError("missing wrapper operand has no regular Bash script witness")
        resolved_target = target.resolve(strict=True)
        target_metadata = resolved_target.stat()
        runner_metadata = runner_path.stat()
        opened_identity = (opened.st_dev, opened.st_ino)
        if (
            resolved_target.name != script.name
            or resolved_target == runner_path
            or (target_metadata.st_dev, target_metadata.st_ino) != opened_identity
            or (runner_metadata.st_dev, runner_metadata.st_ino) == opened_identity
        ):
            raise ValueError("missing wrapper operand has ambiguous Bash script identity")
        return False


@dataclass(frozen=True)
class _LockProcess:
    pid: int
    parent_pid: int
    start_ticks: int
    user_ids: tuple[int, ...]
    cgroup: str
    canonical_runner: bool


def _lock_process(process_pid: int, runner_path: Path) -> _LockProcess:
    process_dir = Path(f"/proc/{process_pid}")
    process_stat = _read_lock_proc(process_dir / "stat")
    stat_head, stat_tail = process_stat.rsplit(b")", 1)
    fields = stat_tail.split()
    if (
        stat_head.split(b"(", 1)[0].strip() != str(process_pid).encode("ascii")
        or len(fields) < 20
        or fields[0] in {b"Z", b"X", b"x"}
        or not fields[1].isdigit()
        or not fields[19].isdigit()
    ):
        raise ValueError("invalid ancestor process identity")
    user_lines = [
        line.split()[1:]
        for line in _read_lock_proc(process_dir / "status").splitlines()
        if line.startswith(b"Uid:")
    ]
    if len(user_lines) != 1 or len(user_lines[0]) != 4 or any(
        not value.isdigit() for value in user_lines[0]
    ):
        raise ValueError("invalid ancestor UID metadata")
    user_ids = tuple(int(value) for value in user_lines[0])
    cgroup_lines = _read_lock_proc(process_dir / "cgroup").decode("utf-8").splitlines()
    unified = [line[3:] for line in cgroup_lines if line.startswith("0::")]
    if len(unified) != 1 or not unified[0].startswith("/") or (
        unified[0] != "/"
        and any(part in {"", ".", ".."} for part in unified[0].split("/")[1:])
    ):
        raise ValueError("invalid ancestor cgroup metadata")
    return _LockProcess(
        pid=process_pid,
        parent_pid=int(fields[1]),
        start_ticks=int(fields[19]),
        user_ids=user_ids,
        cgroup=unified[0],
        canonical_runner=(
            user_ids == (os.getuid(),) * 4
            and _canonical_runner_process(process_dir, runner_path)
        ),
    )


def _lock_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_mode,
        metadata.st_uid, metadata.st_gid, metadata.st_nlink,
    )


def _require_owner_lock_descriptor(owner_pid: int, metadata: os.stat_result) -> None:
    descriptor = Path(f"/proc/{owner_pid}/fd/9")
    if _lock_file_identity(descriptor.stat()) != _lock_file_identity(metadata):
        raise ValueError("ancestor FD9 does not identify the canonical lock")
    lines = _read_lock_proc(Path(f"/proc/{owner_pid}/fdinfo/9")).decode("ascii").splitlines()
    inode_fields = [line.split()[1:] for line in lines if line.startswith("ino:")]
    flag_fields = [line.split()[1:] for line in lines if line.startswith("flags:")]
    lock_fields = [line.split() for line in lines if line.startswith("lock:")]
    if inode_fields != [[str(metadata.st_ino)]] or len(flag_fields) != 1 or (
        len(flag_fields[0]) != 1
        or not re.fullmatch(r"[0-7]+", flag_fields[0][0])
        or int(flag_fields[0][0], 8) & os.O_ACCMODE not in {os.O_WRONLY, os.O_RDWR}
    ):
        raise ValueError("invalid ancestor FD9 metadata")
    if len(lock_fields) != 1:
        raise ValueError("ancestor FD9 has no unique kernel lock")
    fields = lock_fields[0]
    if (
        len(fields) != 9
        or not re.fullmatch(r"[0-9]+:", fields[1])
        or fields[2:5] != ["FLOCK", "ADVISORY", "WRITE"]
        or not fields[5].isdigit()
        or fields[7:] != ["0", "EOF"]
        or not re.fullmatch(r"[0-9a-fA-F]+:[0-9a-fA-F]+:[0-9]+", fields[6])
    ):
        raise ValueError("ancestor FD9 lacks an exclusive whole-file FLOCK")
    device_major, device_minor, inode = fields[6].split(":")
    if (
        os.makedev(int(device_major, 16), int(device_minor, 16)) != metadata.st_dev
        or int(inode) != metadata.st_ino
        or _lock_file_identity(descriptor.stat()) != _lock_file_identity(metadata)
    ):
        raise ValueError("ancestor FLOCK identity changed or differs")


def require_capped_lock_ancestry() -> dict[str, Any]:
    """Prove a stable outer runner holds FD9 above one actual descendant scope.

    This kernel snapshot does not rely on inherited descriptors or environment
    markers. A FLOCK PID of zero is valid; the descriptor's process is the owner
    witness. Same-UID source and the host proc/mount namespace remain trusted.
    """

    try:
        lock_path, metadata = _canonical_lock_state(required=True)
        if metadata is None:
            raise ValueError("missing canonical lock")
        runner_path = Path(__file__).resolve().parents[2] / "scripts/gx1_capped_run.sh"
        runner_path = runner_path.resolve(strict=True)
        ancestors: list[_LockProcess] = []
        seen_pids: set[int] = set()
        process_pid = os.getpid()
        while process_pid:
            if process_pid in seen_pids or len(ancestors) >= _LOCK_MAX_ANCESTORS:
                raise ValueError("cyclic or excessive process ancestry")
            seen_pids.add(process_pid)
            ancestor = _lock_process(process_pid, runner_path)
            ancestors.append(ancestor)
            if process_pid == 1:
                if ancestor.parent_pid != 0:
                    raise ValueError("invalid init ancestry")
                break
            if ancestor.parent_pid <= 0 or ancestor.start_ticks <= 0:
                raise ValueError("incomplete process ancestry")
            process_pid = ancestor.parent_pid
        owners = [index for index, ancestor in enumerate(ancestors) if ancestor.canonical_runner]
        if not owners or owners[-1] == 0:
            raise ValueError("no outer canonical runner ancestor")
        owner_index = owners[-1]
        owner = ancestors[owner_index]
        scope = ancestors[0].cgroup
        if (
            not scope.endswith(".scope")
            or owner.cgroup == scope
            or any(
                ancestor.cgroup != scope or ancestor.user_ids != owner.user_ids
                for ancestor in ancestors[:owner_index]
            )
        ):
            raise ValueError("outer runner must precede exactly one descendant scope")
        _require_owner_lock_descriptor(owner.pid, metadata)
        for ancestor in reversed(ancestors):
            if _lock_process(ancestor.pid, runner_path) != ancestor:
                raise ValueError("process ancestry changed during lock proof")
        checked_path, checked_metadata = _canonical_lock_state(required=True)
        if (
            checked_path != lock_path
            or checked_metadata is None
            or _lock_file_identity(checked_metadata) != _lock_file_identity(metadata)
        ):
            raise ValueError("canonical lock changed during ancestry proof")
        _require_owner_lock_descriptor(owner.pid, metadata)
        if _lock_process(owner.pid, runner_path) != owner:
            raise ValueError("lock owner changed during descriptor recheck")
        return {
            "lock_path": str(lock_path),
            "lock_device": metadata.st_dev,
            "lock_inode": metadata.st_ino,
            "owner_pid": owner.pid,
            "owner_start_ticks": owner.start_ticks,
            "scope": scope,
        }
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError(f"[GX1_CAPPED_LOCK_PROOF_INVALID] {exc}") from exc


def _require_capped_cgroup_limits(
    *,
    environ: Mapping[str, str],
    read_text: Callable[[Path], str] | None,
    max_memory_bytes: int,
    error_prefix: str,
    max_pids: int = _MAX_PIDS,
) -> dict[str, Any]:
    """Match finite runner declarations to the current cgroup's hard limits."""

    expected: dict[str, int] = {}
    for label, name in _LIMIT_ENV.items():
        raw = str(environ.get(name) or "")
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(f"[{error_prefix}_ENV_PROOF_INVALID] field={name}")
        expected[label] = int(raw)
    if (
        expected["memory"] > max_memory_bytes
        or expected["swap"] > _MAX_SWAP_BYTES
        or expected["pids"] > max_pids
    ):
        raise RuntimeError(f"[{error_prefix}_ENV_LIMIT_EXCEEDED]")

    reader = read_text or (lambda path: path.read_text(encoding="utf-8"))
    try:
        cgroup_lines = str(reader(Path("/proc/self/cgroup"))).splitlines()
    except Exception as exc:
        raise RuntimeError(f"[{error_prefix}_CGROUP_PATH_UNAVAILABLE]") from exc
    unified = [
        line.split(":", 2)[2]
        for line in cgroup_lines
        if len(line.split(":", 2)) == 3
        and line.split(":", 2)[0] == "0"
        and line.split(":", 2)[1] == ""
    ]
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise RuntimeError(f"[{error_prefix}_CGROUP_PATH_INVALID]")
    relative_parts = Path(unified[0]).parts[1:]
    if (
        not relative_parts
        or not relative_parts[-1].endswith(".scope")
        or any(part in {"", ".", ".."} for part in relative_parts)
    ):
        raise RuntimeError(f"[{error_prefix}_CGROUP_PATH_INVALID]")
    cgroup_dir = Path("/sys/fs/cgroup").joinpath(*relative_parts)

    def _read_limit(name: str) -> int:
        try:
            raw = str(reader(cgroup_dir / name)).strip()
        except Exception as exc:
            raise RuntimeError(
                f"[{error_prefix}_CGROUP_LIMIT_UNAVAILABLE] field={name}"
            ) from exc
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(
                f"[{error_prefix}_CGROUP_LIMIT_INVALID] field={name}"
            )
        return int(raw)

    actual = {
        "memory_max": _read_limit("memory.max"),
        "memory_high": _read_limit("memory.high"),
        "swap": _read_limit("memory.swap.max"),
        "pids": _read_limit("pids.max"),
    }
    if (
        actual["memory_max"] > max_memory_bytes
        or actual["memory_high"] > max_memory_bytes
        or actual["swap"] > _MAX_SWAP_BYTES
        or actual["pids"] > max_pids
    ):
        raise RuntimeError(f"[{error_prefix}_CGROUP_ACTUAL_LIMIT_EXCEEDED]")
    if (
        actual["memory_max"] != expected["memory"]
        or actual["memory_high"] != expected["memory"]
        or actual["swap"] != expected["swap"]
        or actual["pids"] != expected["pids"]
    ):
        raise RuntimeError(f"[{error_prefix}_CGROUP_ENV_ACTUAL_MISMATCH]")
    return {"cgroup_path": str(cgroup_dir), **actual}


def require_guarded_cuda_trainer_execution() -> dict[str, Any]:
    """Verify the existing canonical trainer's 20G/512M/128-task scope."""
    required = {
        "GX1_CAPPED_CLASS": "trainer", "GX1_CUDA_PRODUCER_GUARD": "false",
        "GX1_TRAINER_DEVICE": "cuda", "GX1_TRAINER_EXECUTION_MODE": "canonical",
        "GX1_CAPPED_MEMORY_BYTES": str(20 * 1024**3),
        "GX1_CAPPED_SWAP_BYTES": str(512 * 1024**2),
        "GX1_CAPPED_TASKS_MAX": "128",
    }
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        required[name] = "8"
    for name in ("ARROW_NUM_THREADS", "POLARS_MAX_THREADS"):
        required[name] = "1"
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"[GX1_CUDA_TRAINER_ENV_INVALID] field={name}")
    if os.sched_getaffinity(0) != set(range(19)):
        raise RuntimeError("[GX1_CUDA_TRAINER_AFFINITY_INVALID]")
    proof = _require_capped_cgroup_limits(
        environ=os.environ, read_text=None, max_memory_bytes=_MAX_MEMORY_BYTES,
        max_pids=128, error_prefix="GX1_CUDA_TRAINER",
    )
    return {**proof, "lock": require_capped_lock_ancestry()}


def require_guarded_cuda_producer_execution(
    *,
    environ: Mapping[str, str] | None = None,
    read_text: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Prove a CUDA evaluator is in the runner's guarded producer scope."""

    env = os.environ if environ is None else environ
    required_exact = {
        "GX1_CAPPED_CLASS": "producer",
        "GX1_CUDA_PRODUCER_GUARD": "true",
        "GX1_TRAINER_DEVICE": "cuda",
        "GX1_TRAINER_EXECUTION_MODE": "cuda_producer",
    }
    for name, expected_value in required_exact.items():
        if str(env.get(name) or "") != expected_value:
            raise RuntimeError(f"[GX1_CUDA_PRODUCER_{name}_INVALID]")
    proof = _require_capped_cgroup_limits(
        environ=env,
        read_text=read_text,
        max_memory_bytes=_MAX_MEMORY_BYTES,
        error_prefix="GX1_CUDA_PRODUCER",
    )
    return {
        "class": "producer",
        "execution_mode": "cuda_producer",
        **proof,
    }


def require_capped_cpu_audit_execution(
    *,
    environ: Mapping[str, str] | None = None,
    read_text: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Prove the audit scope and process leader's kernel-reported CPU affinity.

    This is a boundary snapshot, not an ongoing watchdog or library-thread
    introspection. The caller still owns CPU-only model execution, deterministic
    FP32 and Torch thread configuration. No model or device API is imported.
    """

    env = os.environ if environ is None else environ
    required_exact = {
        "GX1_CAPPED_CLASS": "audit",
        "GX1_CUDA_PRODUCER_GUARD": "false",
        "GX1_TRAINER_DEVICE": "",
        "GX1_TRAINER_EXECUTION_MODE": "canonical",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "1",
    }
    for name, expected_value in required_exact.items():
        if env.get(name) != expected_value:
            raise RuntimeError(f"[GX1_CPU_AUDIT_{name}_INVALID]")
    reader = read_text or (lambda path: path.read_text(encoding="utf-8"))
    proof = _require_capped_cgroup_limits(
        environ=env,
        read_text=reader,
        max_memory_bytes=_MAX_AUDIT_MEMORY_BYTES,
        error_prefix="GX1_CPU_AUDIT",
    )
    try:
        status_lines = str(reader(Path("/proc/self/status"))).splitlines()
    except Exception as exc:
        raise RuntimeError("[GX1_CPU_AUDIT_AFFINITY_UNAVAILABLE]") from exc
    allowed_lists = [
        line.split(":", 1)[1].strip()
        for line in status_lines
        if line.startswith("Cpus_allowed_list:")
    ]
    if len(allowed_lists) != 1 or not allowed_lists[0]:
        raise RuntimeError("[GX1_CPU_AUDIT_AFFINITY_INVALID]")
    allowed_cpus: list[int] = []
    for interval in allowed_lists[0].split(","):
        bounds = interval.split("-")
        if len(bounds) not in (1, 2) or any(
            len(bound) != 1 or bound not in "01234567" for bound in bounds
        ):
            raise RuntimeError("[GX1_CPU_AUDIT_AFFINITY_INVALID]")
        first, last = int(bounds[0]), int(bounds[-1])
        if first > last or (allowed_cpus and first <= allowed_cpus[-1]):
            raise RuntimeError("[GX1_CPU_AUDIT_AFFINITY_INVALID]")
        allowed_cpus.extend(range(first, last + 1))
    return {
        "class": "audit",
        "execution_mode": "cpu_audit",
        **proof,
        "cpu_affinity": allowed_cpus,
        "numerical_threads": 1,
    }



def require_capped_cpu_research_producer_execution(
    *,
    environ: Mapping[str, str] | None = None,
    read_text: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Prove the one bounded CPU market-state research-fit scope.

    This grants no model or data authority. The caller must additionally bind
    its exact one-use plan and policy before reading payloads or fitting.
    """
    env = os.environ if environ is None else environ
    memory = 12 * 1024**3
    swap = 512 * 1024**2
    required = {
        "GX1_CAPPED_CLASS": "producer",
        "GX1_CUDA_PRODUCER_GUARD": "false",
        "GX1_TRAINER_DEVICE": "",
        "GX1_TRAINER_EXECUTION_MODE": "canonical",
        "GX1_CAPPED_MEMORY_BYTES": str(memory),
        "GX1_CAPPED_SWAP_BYTES": str(swap),
        "GX1_CAPPED_TASKS_MAX": "64",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "1",
    }
    for name, expected in required.items():
        if env.get(name) != expected:
            raise RuntimeError(f"[GX1_CPU_RESEARCH_{name}_INVALID]")
    reader = read_text or (lambda path: path.read_text(encoding="utf-8"))
    proof = _require_capped_cgroup_limits(
        environ=env, read_text=reader, max_memory_bytes=memory,
        max_pids=64, error_prefix="GX1_CPU_RESEARCH",
    )
    if (proof["memory_max"] != memory or proof["memory_high"] != memory
            or proof["swap"] != swap or proof["pids"] != 64):
        raise RuntimeError("[GX1_CPU_RESEARCH_EXACT_LIMITS]")
    try:
        status = str(reader(Path("/proc/self/status"))).splitlines()
    except Exception as exc:
        raise RuntimeError("[GX1_CPU_RESEARCH_AFFINITY_UNAVAILABLE]") from exc
    allowed = [line.split(":", 1)[1].strip() for line in status
               if line.startswith("Cpus_allowed_list:")]
    if allowed != ["0-7"]:
        raise RuntimeError("[GX1_CPU_RESEARCH_AFFINITY_INVALID]")
    return {
        "class": "producer", "execution_mode": "cpu_research",
        **proof, "cpu_affinity": list(range(8)), "numerical_threads": 1,
        "lock": require_capped_lock_ancestry(),
    }



def require_native_future_state_cpu_mechanics_execution(
    *,
    plan_sha256: str,
    lane: str,
    environ: Mapping[str, str] | None = None,
    read_text: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Exact private CPU technical mode, never general native training authority.

    The canonical qualifier additionally verifies its frozen recipe, source,
    target/head/prior identities and one-use policy before any optimizer work.
    The runner authenticates only the fixed plan path, SHA, CLI and lane, then
    enforces the existing CPU scope plus a 900-second hard process deadline.
    """
    env = os.environ if environ is None else environ
    if not isinstance(plan_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_CPU_MECHANICS_PLAN_SHA256_INVALID]")
    if lane not in ("continuous", "replay") or type(lane) is not str:
        raise RuntimeError("[GX1_NATIVE_CPU_MECHANICS_LANE_INVALID]")
    expected = {
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "true",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE": lane,
    }
    for name, value in expected.items():
        if env.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_CPU_MECHANICS_{name}_INVALID]")
    proof = require_capped_cpu_research_producer_execution(environ=env, read_text=read_text)
    plan_path = Path(__file__).resolve().parents[2] / (
        "configs/research/NATIVE_M5_FUTURE_STATE_CPU_MECHANICS_20261121_001.json"
    )
    return {**proof, "execution_mode": "native_future_state_cpu_mechanics",
            "technical_only": True, "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "lane": lane, "outer_seconds": 900, "kill_after_seconds": 10}



def require_native_future_state_cuda_qualification_execution(
    *, plan_sha256: str, lane: str,
) -> dict[str, Any]:
    """Exact one-use CUDA qualification inside the unchanged canonical trainer guard.

    The qualifier separately authenticates the recipe, runtime, input identities
    and closed one-use policy. This proof grants no main-fit or producer route.
    """
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_CUDA_QUALIFICATION_PLAN_SHA256_INVALID]")
    if type(lane) is not str or lane not in ("continuous", "replay", "initial"):
        raise RuntimeError("[GX1_NATIVE_CUDA_QUALIFICATION_LANE_INVALID]")
    seconds = 5400 if lane == "initial" else 1800
    expected = {
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "true",
        "GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_FUTURE_STATE_CUDA_LANE": lane,
        "GX1_TRAINER_MAX_WALL_SECONDS": str(seconds),
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": str(seconds),
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_CUDA_QUALIFICATION_{name}_INVALID]")
    proof = require_guarded_cuda_trainer_execution()
    root = Path("/home/andre2/GX1_RUNS")
    basename = ("NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001" if lane == "initial"
                else "NATIVE_M5_FUTURE_STATE_CUDA_MECHANICS_20261121_001")
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = root / basename if lane == "initial" else root / basename / lane
    return {**proof, "class": "trainer", "execution_mode": "canonical", "technical_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256}, "plan_path": str(plan_path),
            "output_root": str(output_root), "lane": lane, "outer_seconds": seconds,
            "internal_seconds": seconds - 30, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


_NATIVE_FUTURE_STATE_GUARD_PARENT_COMMIT = 'c1cd096cfde51719bb1ba5bbd9036864526081df'
_NATIVE_FUTURE_STATE_GUARD_PARENT_HASHES = {
    'gx1/contracts/gx1_capped_execution_v1.py': '31d3d805cd368b2f4e6ea379e266fc30f82a1bf3ede127e147f91a7de4c9172d',
    'scripts/gx1_capped_run.sh': 'fa8326946291ba324e30848cb169380411ddf1c1ba964afc4bf82fcf999c32fb',
}

_NATIVE_FUTURE_STATE_GUARD_SHELL_ADDITIONS = (('TRAINER_MAX_WALL_SECONDS=7200\n', 'before', '# BEGIN GX1 FUTURE LEAF ROUTES 20261121 STATE\nNATIVE_FUTURE_STATE_THROUGHPUT_MODULE=gx1.audit.measure_native_future_state_throughput_v1\nNATIVE_FUTURE_STATE_TRAINING_MODULE=gx1.audit.native_future_state_training_v1\nNATIVE_FUTURE_STATE_THROUGHPUT=false\nNATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256=\nNATIVE_FUTURE_STATE_TRAINING=false\nNATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256=\nNATIVE_FUTURE_STATE_TRAINING_LANE=\n# END GX1 FUTURE LEAF ROUTES 20261121 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 FUTURE LEAF ROUTES 20261121 MATCHERS\nmatches_bound_native_future_state_throughput() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_THROUGHPUT_20261121_003.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_FUTURE_STATE_THROUGHPUT_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nmatches_bound_native_future_state_training() {\n  local expected_plan\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_FUTURE_STATE_TRAINING_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ && "$6" == --lane ]] || return 1\n  case "$7" in\n    joint|joint_replay|m5_primary|m5_primary_replay)\n      expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001.json" ;;\n    main) expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001.json" ;;\n    recovery_probe) expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001.json" ;;\n    main_resume) expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001.json" ;;\n    *) return 1 ;;\n  esac\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_future_state_leaf() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: native future-state leaf requires exact canonical trainer scope" >&2\n    exit 75\n  }\n  local future_leaf_output future_leaf_seconds future_leaf_kind\n  if [[ "${3:-}" == "$NATIVE_FUTURE_STATE_THROUGHPUT_MODULE" ]] \\\n    && matches_bound_native_future_state_throughput "$@"; then\n    future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_THROUGHPUT_20261121_003\n    future_leaf_seconds=1800\n    future_leaf_kind=throughput\n  elif [[ "${3:-}" == "$NATIVE_FUTURE_STATE_TRAINING_MODULE" ]] \\\n    && matches_bound_native_future_state_training "$@"; then\n    future_leaf_kind=training\n    case "$7" in\n      joint|m5_primary) future_leaf_seconds=5400 ;;\n      joint_replay|m5_primary_replay|recovery_probe) future_leaf_seconds=1800 ;;\n      main|main_resume) future_leaf_seconds=12000 ;;\n    esac\n    case "$7" in\n      main) future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001 ;;\n      recovery_probe) future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001 ;;\n      main_resume) future_leaf_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001 ;;\n      *) future_leaf_output="/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001/$7" ;;\n    esac\n  else\n    echo "FATAL: native future-state leaf CLI/plan is not exactly bound" >&2\n    exit 75\n  fi\n  [[ ! -e "$future_leaf_output" && ! -L "$future_leaf_output" \\\n    && -d "${future_leaf_output%/*}" && ! -L "${future_leaf_output%/*}" \\\n    && "$(readlink -f "${future_leaf_output%/*}")" == "${future_leaf_output%/*}" ]] || {\n    echo "FATAL: native future-state leaf output/guard-log parent is not fresh canonical scope" >&2\n    exit 75\n  }\n  if [[ "$future_leaf_kind" == throughput ]]; then\n    NATIVE_FUTURE_STATE_THROUGHPUT=true\n    NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256="$5"\n  else\n    NATIVE_FUTURE_STATE_TRAINING=true\n    NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256="$5"\n    NATIVE_FUTURE_STATE_TRAINING_LANE="$7"\n  fi\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$future_leaf_output"\n  TRAINER_MAX_WALL_SECONDS="$future_leaf_seconds"\n  TRAINER_MODEL_MAX_WALL_SECONDS="$future_leaf_seconds"\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=300\n  TRAINER_GPU_MAX_POWER_DRAW_W=310\n  TRAINER_GPU_MAX_CORE_TEMP_C=85\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n}\n# END GX1 FUTURE LEAF ROUTES 20261121 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 FUTURE LEAF ROUTES 20261121 DISPATCH\n  local future_leaf_arg future_leaf_reference=false\n  for future_leaf_arg in "$@"; do\n    case "$future_leaf_arg" in\n      "$NATIVE_FUTURE_STATE_THROUGHPUT_MODULE"|"$NATIVE_FUTURE_STATE_TRAINING_MODULE"|*/gx1/audit/measure_native_future_state_throughput_v1.py|*/gx1/audit/native_future_state_training_v1.py|gx1/audit/measure_native_future_state_throughput_v1.py|gx1/audit/native_future_state_training_v1.py)\n        future_leaf_reference=true ;;\n    esac\n  done\n  if [[ "$future_leaf_reference" == true ]]; then\n    configure_native_future_state_leaf "$@"\n    return\n  fi\n# END GX1 FUTURE LEAF ROUTES 20261121 DISPATCH\n'), ('  if [[ "$NATIVE_FUTURE_STATE_CUDA_QUALIFICATION" == true ]]; then\n', 'before', '# BEGIN GX1 FUTURE LEAF ROUTES 20261121 NESTED\n  [[ "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == "$NATIVE_FUTURE_STATE_THROUGHPUT" \\\n    && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == "$NATIVE_FUTURE_STATE_TRAINING" ]] || {\n    echo "FATAL: nested native future-state leaf mode differs from parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_FUTURE_STATE_THROUGHPUT" == true || "$NATIVE_FUTURE_STATE_TRAINING" == true ]]; then\n    [[ "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "$NATIVE_FUTURE_STATE_TRAINING_LANE" \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == "$TRAINER_MAX_WALL_SECONDS" \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == "$TRAINER_MODEL_MAX_WALL_SECONDS" \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 300 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 310 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 85 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 ]] || {\n      echo "FATAL: nested native future-state leaf plan/deadline/hardware differs from parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 FUTURE LEAF ROUTES 20261121 NESTED\n'), ('  --setenv=GX1_POWER_BENCHMARK_SCOPE_JSON="$POWER_BENCHMARK_SCOPE_JSON" \\\n', 'before', '  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT="$NATIVE_FUTURE_STATE_THROUGHPUT" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256="$NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_TRAINING="$NATIVE_FUTURE_STATE_TRAINING" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256="$NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_TRAINING_LANE="$NATIVE_FUTURE_STATE_TRAINING_LANE" \\\n'))


_NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 STATE\n', 'before', '# BEGIN GX1 FROZEN BIAS ROUTE 20261122 STATE\nNATIVE_FUTURE_STATE_BIAS_FROZEN_MODULE=gx1.audit.diagnose_native_future_state_bias_v1\nNATIVE_FUTURE_STATE_BIAS_FROZEN=false\nNATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256=\n# END GX1 FROZEN BIAS ROUTE 20261122 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 FROZEN BIAS ROUTE 20261122 MATCHERS\nmatches_bound_native_future_state_bias_frozen() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_BIAS_FROZEN_20261122_001.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_FUTURE_STATE_BIAS_FROZEN_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_future_state_bias_frozen() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: frozen bias diagnosis requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_future_state_bias_frozen "$@" || {\n    echo "FATAL: frozen bias diagnosis CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local frozen_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_FROZEN_20261122_001\n  [[ ! -e "$frozen_output" && ! -L "$frozen_output" \\\n    && -d "${frozen_output%/*}" && ! -L "${frozen_output%/*}" \\\n    && "$(readlink -f "${frozen_output%/*}")" == "${frozen_output%/*}" ]] || {\n    echo "FATAL: frozen bias output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_FUTURE_STATE_BIAS_FROZEN=true\n  NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$frozen_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3600\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=300\n  TRAINER_GPU_MAX_POWER_DRAW_W=310\n  TRAINER_GPU_MAX_CORE_TEMP_C=85\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n}\n# END GX1 FROZEN BIAS ROUTE 20261122 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 FROZEN BIAS ROUTE 20261122 DISPATCH\n  local frozen_bias_arg frozen_bias_reference=false\n  for frozen_bias_arg in "$@"; do\n    case "$frozen_bias_arg" in\n      "$NATIVE_FUTURE_STATE_BIAS_FROZEN_MODULE"|*/gx1/audit/diagnose_native_future_state_bias_v1.py|gx1/audit/diagnose_native_future_state_bias_v1.py)\n        frozen_bias_reference=true ;;\n    esac\n  done\n  if [[ "$frozen_bias_reference" == true ]]; then\n    configure_native_future_state_bias_frozen "$@"\n    return\n  fi\n# END GX1 FROZEN BIAS ROUTE 20261122 DISPATCH\n'), ('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 NESTED\n', 'before', '# BEGIN GX1 FROZEN BIAS ROUTE 20261122 NESTED\n  [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == "$NATIVE_FUTURE_STATE_BIAS_FROZEN" ]] || {\n    echo "FATAL: nested frozen bias mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_FUTURE_STATE_BIAS_FROZEN" == true ]]; then\n    [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 300 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 310 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 85 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 ]] || {\n      echo "FATAL: nested frozen bias plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 FROZEN BIAS ROUTE 20261122 NESTED\n'), ('  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT="$NATIVE_FUTURE_STATE_THROUGHPUT" \\\n', 'before', '  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN="$NATIVE_FUTURE_STATE_BIAS_FROZEN" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256="$NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256" \\\n'))


_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 STATE\n', 'before', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 STATE\nNATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE=gx1.audit.diagnose_native_future_state_bias_suffix_v1\nNATIVE_FUTURE_STATE_BIAS_SUFFIX=false\nNATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256=\n# END GX1 BIAS SUFFIX ROUTE 20261122 003 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 MATCHERS\nmatches_bound_native_future_state_bias_suffix() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_003.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_future_state_bias_suffix() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: bias suffix order study requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_future_state_bias_suffix "$@" || {\n    echo "FATAL: bias suffix order study CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local suffix_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_003\n  [[ ! -e "$suffix_output" && ! -L "$suffix_output" \\\n    && -d "${suffix_output%/*}" && ! -L "${suffix_output%/*}" \\\n    && "$(readlink -f "${suffix_output%/*}")" == "${suffix_output%/*}" ]] || {\n    echo "FATAL: bias suffix output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_FUTURE_STATE_BIAS_SUFFIX=true\n  NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$suffix_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n}\n# END GX1 BIAS SUFFIX ROUTE 20261122 003 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 DISPATCH\n  local frozen_bias_arg frozen_bias_reference=false\n  for frozen_bias_arg in "$@"; do\n    case "$frozen_bias_arg" in\n      "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE"|*/gx1/audit/diagnose_native_future_state_bias_suffix_v1.py|gx1/audit/diagnose_native_future_state_bias_suffix_v1.py)\n        frozen_bias_reference=true ;;\n    esac\n  done\n  if [[ "$frozen_bias_reference" == true ]]; then\n    configure_native_future_state_bias_suffix "$@"\n    return\n  fi\n# END GX1 BIAS SUFFIX ROUTE 20261122 003 DISPATCH\n'), ('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 NESTED\n', 'before', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 003 NESTED\n  [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX" ]] || {\n    echo "FATAL: nested bias suffix mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_FUTURE_STATE_BIAS_SUFFIX" == true ]]; then\n    [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested bias suffix plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 BIAS SUFFIX ROUTE 20261122 003 NESTED\n'), ('  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT="$NATIVE_FUTURE_STATE_THROUGHPUT" \\\n', 'before', '  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX="$NATIVE_FUTURE_STATE_BIAS_SUFFIX" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256="$NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256" \\\n'), ('TRAINER_HOST_TELEMETRY_QUERY_PATH="$REPO_ROOT/scripts/gx1_host_telemetry_bridge_query.sh"\n', 'before', '# Conservative initial host ceilings, not hardware limits or an optimization claim.\nTRAINER_CPU_MAX_TEMP_C=80\nTRAINER_CPU_MAX_PACKAGE_POWER_W=125\nTRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=$((8 * 1024 * 1024 * 1024))\nTRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=$((8 * 1024 * 1024 * 1024))\n'), ('  [[ "$TRAINER_HOST_TELEMETRY_CERT_SHA256" =~ ^[0-9a-f]{64}$ \\\n', 'before', '  for host_limit in "$TRAINER_CPU_MAX_TEMP_C" "$TRAINER_CPU_MAX_PACKAGE_POWER_W" \\\n    "$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES" "$TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES"; do\n    [[ "$host_limit" =~ ^[1-9][0-9]*$ ]] || {\n      echo "FATAL: protected host resource limit is malformed" >&2\n      exit 75\n    }\n  done\n'), ('  --setenv=GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH="$TRAINER_HOST_TELEMETRY_QUERY_PATH" \\\n', 'before', '  --setenv=GX1_TRAINER_CPU_MAX_TEMP_C="$TRAINER_CPU_MAX_TEMP_C" \\\n  --setenv=GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W="$TRAINER_CPU_MAX_PACKAGE_POWER_W" \\\n  --setenv=GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES="$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES" \\\n  --setenv=GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES="$TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES" \\\n'))


def native_future_state_driver_requalification_metadata(original, current):
    """Declare one unproven driver transition; this grants no restore or update authority."""
    from gx1.audit.qualify_native_m5_full_gc_direction_v1 import future_cuda_require_runtime
    future_cuda_require_runtime(current)
    valid = (type(original) is dict and set(original) == {"library_versions", "gpu"}
             and type(original["gpu"]) is dict and set(original["gpu"]) == set(current["gpu"])
             and type(original["library_versions"]) is dict and set(original["library_versions"]) == set(current["library_versions"])
             and original["library_versions"] == current["library_versions"]
             and all(type(original[group][key]) is type(current[group][key]) for group in ("library_versions", "gpu") for key in current[group])
             and original["gpu"].get("driver_version") == "591.86"
             and current["gpu"]["driver_version"] == "610.88"
             and all(original["gpu"][k] == current["gpu"][k] for k in current["gpu"] if k != "driver_version"))
    if not valid:
        raise RuntimeError("[GX1_NATIVE_DRIVER_REQUALIFICATION_METADATA_INVALID]")
    return {"schema_version": "gx1_native_driver_requalification_transition_v1",
            "original_driver": "591.86", "current_driver": "610.88",
            "other_runtime_fields_identical": True, "equivalence_proven": False}


def require_native_future_state_bias_suffix_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Bounded suffix-order containment; technical update authority belongs only to the leaf."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_SUFFIX_BIAS_PLAN_SHA256_INVALID]")
    expected = {
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "true",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_TRAINER_MAX_WALL_SECONDS": "3600",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "3570",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_SUFFIX_BIAS_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_003"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "bias_suffix_driver_requalification_order",
            "training_authority_granted": False, "optimizer_authority_granted": False,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}



def native_future_state_bias_guard_source_bridge(previous_source_owners, *, producer_commit, driver_requalification=False):
    """Prove MAIN→001, the fixed qualified host-resource delta, and only the named002 route."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_NATIVE_FROZEN_BIAS_GUARD_BRIDGE_" + reason + "]")
    require(type(driver_requalification) is bool, "DRIVER_REQUALIFICATION_MODE")
    repo = Path(__file__).resolve().parents[2]
    prior_commit = "63a1ff0ac8889046691dafe1acd1a63b55974cc0"
    resource_commit = "6749062726c8562db35194857f56ee8aa4859d23"
    guards = ("gx1/contracts/gx1_capped_execution_v1.py", "scripts/gx1_capped_run.sh")
    hashes = ("d29344a8b591e58500e71b38dd41d4124a3f3f234d09751fffc31585a2366f3f",
              "d6883f0f642f5e5304ee7f615fd8f80d364c7472d9f53cfbd12592a2c031f17f")
    historical = {name: subprocess.check_output(["git", "show", prior_commit + ":" + name], cwd=repo) for name in guards}
    qualified = {name: subprocess.check_output(["git", "show", resource_commit + ":" + name], cwd=repo) for name in guards}
    require(all(hashlib.sha256(historical[name]).hexdigest() == sha for name, sha in zip(guards, hashes)), "AUTHENTIC_001_GUARDS")
    require(qualified[guards[0]] == historical[guards[0]], "RESOURCE_PYTHON_UNCHANGED")
    require(hashlib.sha256(qualified[guards[1]]).hexdigest() == "8a88a7da0daaff455e1bf25670c121163ccd123b6ac82c743e910d3fc864b3a1", "QUALIFIED_RESOURCE_SHELL")
    diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "--no-renames", "--no-color", "--unified=3", prior_commit, resource_commit, "--", guards[1]], cwd=repo)
    diff_sha = hashlib.sha256(diff).hexdigest()
    require(diff_sha == "45627527b5019aa87d92acb18c2fff1177fdd5ec457daf835b5715c30a879804", "EXACT_RESOURCE_DIFF")
    historical_tree = ast.parse(historical[guards[0]])
    historical_suffix = ast.literal_eval(next(n.value for n in historical_tree.body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS"))
    def verify_main_to_001():
        """Prove only named bias-diagnosis containment additions versus authentic MAIN guards."""
        import ast
        import hashlib
        import subprocess
        def require(ok, reason):
            if not ok:
                raise RuntimeError("[GX1_NATIVE_FROZEN_BIAS_GUARD_BRIDGE_" + reason + "]")
        require(producer_commit == "f96d89f88ea742a9a90460d81cc386f8abea6e2a", "PRODUCER_COMMIT")
        require(type(previous_source_owners) is dict and len(previous_source_owners) == 62, "OWNER_COUNT")
        repo = Path(__file__).resolve().parents[2]
        guards = ("gx1/contracts/gx1_capped_execution_v1.py", "scripts/gx1_capped_run.sh")
        added_functions = {"require_native_future_state_bias_frozen_execution", "require_native_future_state_bias_suffix_execution", "native_future_state_bias_guard_source_bridge"}
        added_constants = {"_NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS", "_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS"}
        _NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS = historical_suffix
        report = {}
        for name in guards:
            original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
            expected_sha = hashlib.sha256(original).hexdigest()
            require(previous_source_owners.get(name) == expected_sha, "AUTHENTIC_MAIN_GUARD_HASH")
            current = historical[name]
            if name.endswith(".py"):
                old_tree, new_tree = ast.parse(original), ast.parse(current)
                names = [n.name for n in new_tree.body if isinstance(n, ast.FunctionDef)]
                require(len(names) == len(set(names)), "DUPLICATE_FUNCTION")
                retained = []; removed_functions = set(); removed_constants = set()
                for node in new_tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name in added_functions:
                        removed_functions.add(node.name)
                    elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                          and isinstance(node.targets[0], ast.Name) and node.targets[0].id in added_constants):
                        require(node.targets[0].id not in removed_constants, "DUPLICATE_CONSTANT"); removed_constants.add(node.targets[0].id)
                    else:
                        retained.append(node)
                require(removed_functions == added_functions and removed_constants == added_constants, "EXACT_ADDITIONS")
                bridge = [n for n in retained if isinstance(n, ast.FunctionDef) and n.name == "native_future_state_guard_source_bridge"]
                require(len(bridge) == 1, "ORIGINAL_BRIDGE_OWNER")
                normalized = {"functions": 0, "constants": 0, "shell_loop": 0}
                for node in ast.walk(bridge[0]):
                    if (isinstance(node, ast.Assign) and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Set)):
                        name_ = node.targets[0].id
                        if name_ in ("added_functions", "added_constants"):
                            remove = added_functions if name_ == "added_functions" else added_constants
                            matched = {x.value for x in node.value.elts if isinstance(x, ast.Constant) and x.value in remove}
                            require(matched == remove, "EXACT_BRIDGE_WHITELIST")
                            node.value.elts = [x for x in node.value.elts if not (isinstance(x, ast.Constant) and x.value in remove)]
                            normalized["functions" if name_ == "added_functions" else "constants"] += 1
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "reversed"
                            and len(node.args) == 1 and ast.dump(node.args[0], include_attributes=False) == ast.dump(
                                ast.parse("_NATIVE_FUTURE_STATE_GUARD_SHELL_ADDITIONS + _NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS + _NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS", mode="eval").body,
                                include_attributes=False)):
                        node.args = [ast.Name(id="_NATIVE_FUTURE_STATE_GUARD_SHELL_ADDITIONS", ctx=ast.Load())]
                        normalized["shell_loop"] += 1
                require(normalized == {"functions": 1, "constants": 1, "shell_loop": 1}, "EXACT_BRIDGE_NORMALIZATION")
                new_tree.body = retained
                before, after = ast.dump(old_tree, include_attributes=False), ast.dump(new_tree, include_attributes=False)
                require(before == after, "OLD_PYTHON_MODULE_CHANGED")
                normalized_sha = hashlib.sha256(after.encode()).hexdigest()
                proof = "whole_MAIN_module_AST_equal_after_named_bias_additions_and_exact_bridge_whitelist_normalization"
            else:
                text = current.decode()
                for anchor, position, fragment in reversed(_NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS + _NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS):
                    composite = fragment + anchor if position == "before" else anchor + fragment
                    require(text.count(fragment) == text.count(composite) == 1, "EXACT_SHELL_INSERTION")
                    text = text.replace(composite, anchor, 1)
                require(text.encode() == original, "OLD_SHELL_BYTES_CHANGED")
                normalized_sha = hashlib.sha256(text.encode()).hexdigest()
                proof = "all_MAIN_shell_bytes_equal_after_only_named_bias_insertions"
            report[name] = {"original_sha256": expected_sha, "current_sha256": hashlib.sha256(current).hexdigest(),
                            "normalized_sha256": normalized_sha, "proof": proof}
        return {"producer_commit": producer_commit, "checked_guard_owners": 2,
                "other_main_owners_to_be_checked_by_caller": 60, "owners": report}

    result = verify_main_to_001()
    additions = _NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS
    require(len(historical_suffix) == 5 and len(additions) == 8, "EXACT_002_AND_RESOURCE_INSERTION_COUNTS")
    # These three removals must reverse exactly the reviewed16-line resource delta.
    stripped = qualified[guards[1]].decode()
    for anchor, position, fragment in reversed(additions[5:]):
        require(position == "before" and stripped.count(fragment + anchor) == 1, "EXACT_RESOURCE_INSERTION")
        stripped = stripped.replace(fragment + anchor, anchor, 1)
    require(stripped.encode() == historical[guards[1]], "RESOURCE_DELTA_BYTES")
    current_shell = (repo / guards[1]).read_text()
    for (anchor, position, fragment), (old_anchor, old_position, old_fragment) in zip(additions[:5], historical_suffix):
        require((anchor, position) == (old_anchor, old_position), "EXACT_002_ROUTE_ANCHOR")
        composite = fragment + anchor if position == "before" else anchor + fragment
        replacement = old_fragment + anchor if position == "before" else anchor + old_fragment
        require(current_shell.count(fragment) == current_shell.count(composite) == 1, "EXACT_002_ROUTE_INSERTION")
        current_shell = current_shell.replace(composite, replacement, 1)
    require(current_shell.encode() == qualified[guards[1]], "ONLY_002_ROUTE_CHANGED")
    current_tree = ast.parse((repo / guards[0]).read_bytes())
    names = [n.name for n in current_tree.body if isinstance(n, ast.FunctionDef)]
    require(len(names) == len(set(names)), "DUPLICATE_FUNCTION")
    changed_functions = {"native_future_state_bias_guard_source_bridge", "require_native_future_state_bias_suffix_execution", "native_future_state_driver_requalification_metadata"}
    require(changed_functions <= set(names), "EXACT_002_GUARD_FUNCTIONS")
    route_constants = [n for n in current_tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS"]
    require(len(route_constants) == 1, "EXACT_002_ROUTE_CONSTANT")
    require(ast.literal_eval(route_constants[0].value) == additions, "CURRENT_002_ROUTE_LITERAL")
    def retained(tree):
        if tree is current_tree:
            old_bridge = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_future_state_guard_source_bridge")
            assignments = [n for n in ast.walk(old_bridge) if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "added_functions" and isinstance(n.value, ast.Set)]
            require(len(assignments) == 1, "EXACT_DRIVER_METADATA_WHITELIST_OWNER")
            values = assignments[0].value.elts
            require(sum(isinstance(n, ast.Constant) and n.value == "native_future_state_driver_requalification_metadata" for n in values) == 1, "EXACT_DRIVER_METADATA_WHITELIST")
            assignments[0].value.elts = [n for n in values if not (isinstance(n, ast.Constant) and n.value == "native_future_state_driver_requalification_metadata")]
        return ast.Module(body=[n for n in tree.body if not (
            isinstance(n, ast.FunctionDef) and n.name in changed_functions or
            isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS")], type_ignores=[])
    require(ast.dump(retained(current_tree), include_attributes=False) == ast.dump(retained(historical_tree), include_attributes=False), "OLD_001_PYTHON_MODULE_CHANGED")
    for name in guards:
        result["owners"][name]["current_sha256"] = hashlib.sha256((repo / name).read_bytes()).hexdigest()
    result["qualified_resource_delta"] = {"from_commit": prior_commit, "to_commit": resource_commit,
        "diff_sha256": diff_sha, "qualified_shell_sha256": hashlib.sha256(qualified[guards[1]]).hexdigest(),
        "current_change": "only_named003_route_and_explicit_driver_metadata_guard_owners"}
    DRIVER_RESTORE_BLOCK = "runtime_binding=runtime_compatible(trajectory['runtime_compatibility'],current_runtime,driver_requalification=driver_requalification)\nif driver_requalification is not None:\n    need(original_plan_ref==runtime_binding['original_plan']and expected_cursor==runtime_binding['expected_cursor']\n        and expected_state_proof==runtime_binding['expected_state_proof'],'EXACT_DRIVER_REQUALIFICATION_RESTORE_SCOPE')\n"
    if driver_requalification:
        loader = "gx1/audit/native_future_state_training_v1.py"
        original = subprocess.check_output(["git", "show", producer_commit + ":" + loader], cwd=repo)
        require(previous_source_owners.get(loader) == hashlib.sha256(original).hexdigest(), "AUTHENTIC_MAIN_LOADER_HASH")
        original_tree = ast.parse(original); restored_tree = ast.parse((repo / loader).read_bytes())
        old_nodes = {n.name: n for n in original_tree.body if isinstance(n, ast.FunctionDef)}
        new_nodes = {n.name: n for n in restored_tree.body if isinstance(n, ast.FunctionDef)}
        require(len(new_nodes) == len([n for n in restored_tree.body if isinstance(n, ast.FunctionDef)]), "DUPLICATE_LOADER_FUNCTION")
        strict = new_nodes["strict_restore"]
        require(strict.args.kwonlyargs[-1].arg == "driver_requalification"
                and isinstance(strict.args.kw_defaults[-1], ast.Constant) and strict.args.kw_defaults[-1].value is None,
                "EXACT_DRIVER_LOADER_OPTION")
        strict.args.kwonlyargs.pop(); strict.args.kw_defaults.pop()
        expected_block = ast.parse(DRIVER_RESTORE_BLOCK).body
        matches = [i for i in range(len(strict.body) - 1) if ast.dump(ast.Module(body=strict.body[i:i+2], type_ignores=[]), include_attributes=False)
                   == ast.dump(ast.Module(body=expected_block, type_ignores=[]), include_attributes=False)]
        require(len(matches) == 1, "EXACT_DRIVER_RESTORE_METADATA_BLOCK")
        strict.body[matches[0]:matches[0]+2] = ast.parse("runtime_compatible(trajectory['runtime_compatibility'],current_runtime)").body
        restored_tree.body = [old_nodes["runtime_compatible"] if isinstance(n, ast.FunctionDef) and n.name == "runtime_compatible" else n for n in restored_tree.body]
        require(ast.dump(restored_tree, include_attributes=False) == ast.dump(original_tree, include_attributes=False), "ALL_LOADER_KERNEL_AST_UNCHANGED")
        result["owners"][loader] = {"original_sha256": hashlib.sha256(original).hexdigest(),
            "current_sha256": hashlib.sha256((repo / loader).read_bytes()).hexdigest(),
            "normalized_sha256": hashlib.sha256(ast.dump(restored_tree, include_attributes=False).encode()).hexdigest(),
            "proof": "only_optional_driver_metadata_gate_and_exact_pre_copy_binding;all_tensor_optimizer_rng_operations_AST_equal"}
        result["checked_guard_owners"] = 3
        result["other_main_owners_to_be_checked_by_caller"] = 59
    return result


def require_native_future_state_bias_frozen_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Read-only checkpoint diagnosis containment; grants no optimizer or training authority."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_FROZEN_BIAS_PLAN_SHA256_INVALID]")
    expected = {
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "true",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_TRAINER_MAX_WALL_SECONDS": "3600",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "3600",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "300",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "310",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "85",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_FROZEN_BIAS_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_FUTURE_STATE_BIAS_FROZEN_20261122_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "bias_frozen",
            "training_authority_granted": False, "optimizer_authority_granted": False,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def _native_future_state_leaf_execution(*, plan_sha256: str, kind: str, lane: str | None) -> dict[str, Any]:
    """Containment only; exact source, recipe and one-use training authority belong to the leaf."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_FUTURE_LEAF_PLAN_SHA256_INVALID]")
    if kind == "throughput" and lane is None:
        seconds, basename, output_lane = 1800, "NATIVE_FUTURE_STATE_THROUGHPUT_20261121_003", None
    elif kind == "training" and type(lane) is str and lane in (
            "joint", "joint_replay", "m5_primary", "m5_primary_replay", "main", "recovery_probe", "main_resume"):
        seconds = (12000 if lane in ("main", "main_resume") else
                   1800 if lane.endswith("_replay") or lane == "recovery_probe" else 5400)
        fixed = {"main": "NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001",
                 "recovery_probe": "NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001",
                 "main_resume": "NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001"}
        basename = fixed.get(lane, "NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001")
        output_lane = None if lane in fixed else lane
    else:
        raise RuntimeError("[GX1_NATIVE_FUTURE_LEAF_KIND_LANE_INVALID]")
    expected = {
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "true" if kind == "throughput" else "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": plan_sha256 if kind == "throughput" else "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "true" if kind == "training" else "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": plan_sha256 if kind == "training" else "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": lane if kind == "training" else "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_TRAINER_MAX_WALL_SECONDS": str(seconds),
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": str(seconds),
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0", "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "300",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "310", "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "85",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80", "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "", "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_FUTURE_LEAF_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    if output_lane is not None:
        output_root /= output_lane
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": kind,
            "training_authority_granted": False, "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root), "lane": lane,
            "outer_seconds": seconds, "internal_seconds": seconds - 30, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def require_native_future_state_throughput_execution(*, plan_sha256: str) -> dict[str, Any]:
    return _native_future_state_leaf_execution(plan_sha256=plan_sha256, kind="throughput", lane=None)


def require_native_future_state_training_execution(*, plan_sha256: str, lane: str) -> dict[str, Any]:
    return _native_future_state_leaf_execution(plan_sha256=plan_sha256, kind="training", lane=lane)


def native_future_state_guard_source_bridge(previous_source_owners, *, producer_commit):
    """Metadata-only two-owner bridge; callers authenticate the full INITIAL58 inventory and other56."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_NATIVE_FUTURE_GUARD_BRIDGE_" + reason + "]")
    require(type(producer_commit) is str and producer_commit == _NATIVE_FUTURE_STATE_GUARD_PARENT_COMMIT,
            "PRODUCER_COMMIT")
    require(type(previous_source_owners) is dict and len(previous_source_owners) == 58, "OWNER_COUNT")
    for name, digest in previous_source_owners.items():
        require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
                and str(Path(name)) == name and Path(name).suffix in (".py", ".sh"), "SOURCE_PATH")
        require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "SOURCE_SHA")
    require(all(previous_source_owners.get(name) == digest
                for name, digest in _NATIVE_FUTURE_STATE_GUARD_PARENT_HASHES.items()), "PARENT_GUARD_HASHES")
    repo = Path(__file__).resolve().parents[2]
    report = {}
    for name, expected_sha in _NATIVE_FUTURE_STATE_GUARD_PARENT_HASHES.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == expected_sha, "ORIGINAL_GIT_BYTES")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            old_tree, new_tree = ast.parse(original), ast.parse(current)
            names = [node.name for node in new_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            require(len(names) == len(set(names)), "DUPLICATE_FUNCTION")
            added_functions = {"_native_future_state_leaf_execution", "require_native_future_state_throughput_execution",
                "native_future_state_driver_requalification_metadata", "require_native_future_state_training_execution", "require_native_future_state_bias_frozen_execution", "require_native_future_state_bias_suffix_execution", "native_future_state_bias_guard_source_bridge", "native_future_state_guard_source_bridge"}
            added_constants = {"_NATIVE_FUTURE_STATE_GUARD_PARENT_COMMIT", "_NATIVE_FUTURE_STATE_GUARD_PARENT_HASHES",
                "_NATIVE_FUTURE_STATE_GUARD_SHELL_ADDITIONS", "_NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS", "_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS"}
            retained = []
            seen_functions, seen_constants = set(), set()
            for node in new_tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in added_functions:
                    seen_functions.add(node.name)
                elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in added_constants:
                    require(node.targets[0].id not in seen_constants, "DUPLICATE_CONSTANT")
                    seen_constants.add(node.targets[0].id)
                else:
                    retained.append(node)
            require(seen_functions == added_functions and seen_constants == added_constants, "EXACT_ADDITIONS")
            new_tree.body = retained
            before, after = ast.dump(old_tree, include_attributes=False), ast.dump(new_tree, include_attributes=False)
            require(before == after, "OLD_PYTHON_MODULE_CHANGED")
            normalized_sha = hashlib.sha256(after.encode()).hexdigest()
            proof = "whole_module_AST_equal_after_only_named_additions"
        else:
            text = current.decode()
            for anchor, position, fragment in reversed(_NATIVE_FUTURE_STATE_GUARD_SHELL_ADDITIONS + _NATIVE_FUTURE_STATE_BIAS_GUARD_SHELL_ADDITIONS + _NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            require(text.encode() == original, "OLD_SHELL_BYTES_CHANGED")
            normalized_sha = hashlib.sha256(text.encode()).hexdigest()
            proof = "all_old_shell_bytes_equal_after_only_exact_anchored_insertions"
        report[name] = {"original_sha256": expected_sha, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": normalized_sha, "proof": proof}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2,
            "other_parent_owners_to_be_checked_by_caller": 56, "owners": report}


_NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 STATE\n', 'before', '# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 STATE\nNATIVE_FUTURE_STATE_BIAS_CONTINUE_MODULE=gx1.audit.continue_native_future_state_bias_suffix_v1\nNATIVE_FUTURE_STATE_BIAS_CONTINUE=false\nNATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256=\n# END GX1 BIAS CONTINUE ROUTE 20261122 004 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 MATCHERS\nmatches_bound_native_future_state_bias_suffix_continuation() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_FUTURE_STATE_BIAS_CONTINUE_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_future_state_bias_suffix_continuation() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: bias suffix continuation requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_future_state_bias_suffix_continuation "$@" || {\n    echo "FATAL: bias suffix continuation CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local suffix_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004\n  [[ ! -e "$suffix_output" && ! -L "$suffix_output" \\\n    && -d "${suffix_output%/*}" && ! -L "${suffix_output%/*}" \\\n    && "$(readlink -f "${suffix_output%/*}")" == "${suffix_output%/*}" ]] || {\n    echo "FATAL: bias suffix continuation output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_FUTURE_STATE_BIAS_CONTINUE=true\n  NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$suffix_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n}\n# END GX1 BIAS CONTINUE ROUTE 20261122 004 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 DISPATCH\n  local frozen_bias_arg frozen_bias_reference=false\n  for frozen_bias_arg in "$@"; do\n    case "$frozen_bias_arg" in\n      "$NATIVE_FUTURE_STATE_BIAS_CONTINUE_MODULE"|*/gx1/audit/continue_native_future_state_bias_suffix_v1.py|gx1/audit/continue_native_future_state_bias_suffix_v1.py)\n        frozen_bias_reference=true ;;\n    esac\n  done\n  if [[ "$frozen_bias_reference" == true ]]; then\n    configure_native_future_state_bias_suffix_continuation "$@"\n    return\n  fi\n# END GX1 BIAS CONTINUE ROUTE 20261122 004 DISPATCH\n'), ('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 NESTED\n', 'before', '# BEGIN GX1 BIAS CONTINUE ROUTE 20261122 004 NESTED\n  [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == "$NATIVE_FUTURE_STATE_BIAS_CONTINUE" ]] || {\n    echo "FATAL: nested bias suffix continuation mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_FUTURE_STATE_BIAS_CONTINUE" == true ]]; then\n    [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested bias suffix continuation plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 BIAS CONTINUE ROUTE 20261122 004 NESTED\n'), ('  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT="$NATIVE_FUTURE_STATE_THROUGHPUT" \\\n', 'before', '  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE="$NATIVE_FUTURE_STATE_BIAS_CONTINUE" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256="$NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256" \\\n'))


def require_native_future_state_bias_suffix_continuation_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Cold-qualified checkpoint continuation containment; technical update authority belongs only to the leaf."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_SUFFIX_CONTINUE_PLAN_SHA256_INVALID]")
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "true",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_TRAINER_MAX_WALL_SECONDS": "3600",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "3570",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_SUFFIX_CONTINUE_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "bias_suffix_continuation",
            "training_authority_granted": False, "optimizer_authority_granted": False,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}



def native_future_state_bias_suffix_continuation_source_bridge(previous_source_owners, *, producer_commit="8ead4ba4e9c8fb4fcd35e6cd40818d6920fe16ed"):
    """Authenticate consumed003 sources, allowing only the named004 guard additions."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_NATIVE_SUFFIX_CONTINUE_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "8ead4ba4e9c8fb4fcd35e6cd40818d6920fe16ed", "PRODUCER_COMMIT")
    require(type(previous_source_owners) is dict and len(previous_source_owners) == 74, "OWNER_COUNT")
    for name, digest in previous_source_owners.items():
        require(type(name) is str, "OWNER_SCHEMA")
        path = Path(name)
        require(type(name) is str and not path.is_absolute() and ".." not in path.parts and str(path) == name
            and path.suffix in (".py", ".sh", ".ps1") and type(digest) is str
            and re.fullmatch("[0-9a-f]{64}", digest) is not None, "OWNER_SCHEMA")
    repo = Path(__file__).resolve().parents[2]
    guards = {"gx1/contracts/gx1_capped_execution_v1.py": "baae091dab72f0dc13f5be08e27cb8e0987b4de3da2c8bf8bf6a702334ebdaaa",
              "scripts/gx1_capped_run.sh": "beee246f2c4b6e716531bdea6fccf1d35c5c82bc30edb77e742706c0ad044495"}
    functions = {"require_native_future_state_bias_suffix_continuation_execution", "native_future_state_bias_suffix_continuation_source_bridge"}
    constant = "_NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS"
    report = {}
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(previous_source_owners.get(name) == hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_003_GUARD_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            old_tree = ast.parse(original); new_tree = ast.parse(current)
            names = [n.name for n in new_tree.body if isinstance(n, ast.FunctionDef)]
            require(len(names) == len(set(names)), "DUPLICATE_FUNCTION")
            require(functions <= set(names), "EXACT_004_FUNCTIONS")
            nodes = [n for n in new_tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == constant]
            require(len(nodes) == 1 and ast.literal_eval(nodes[0].value) == _NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS,
                    "EXACT_004_ROUTE_CONSTANT")
            new_tree.body = [n for n in new_tree.body if not (
                isinstance(n, ast.FunctionDef) and n.name in functions or n is nodes[0])]
            normalized = ast.dump(new_tree, include_attributes=False)
            require(normalized == ast.dump(old_tree, include_attributes=False), "ALL_003_PYTHON_AST_UNCHANGED")
            proof = "all003moduleAST_equal_after_only_two_named004_functions_and_one_literal_route_tuple"
        else:
            text = current.decode()
            require(len(_NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS) == 5, "EXACT_004_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_004_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            require(text.encode() == original, "ALL_003_SHELL_BYTES_UNCHANGED")
            normalized = text
            proof = "all003shellbytes_equal_after_only_five_exact_named004_insertions"
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(), "proof": proof}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2,
            "other_parent_owners_to_be_checked_by_caller": 72, "owners": report}



_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 STATE\n', 'before', '# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 STATE\nNATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_MODULE=gx1.audit.capture_native_future_state_full_train_v1\nNATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE=false\nNATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256=\n# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 MATCHERS\nmatches_bound_native_future_state_full_train_capture() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_future_state_full_train_capture() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: full TRAIN capture requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_future_state_full_train_capture "$@" || {\n    echo "FATAL: full TRAIN capture CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local suffix_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001\n  [[ ! -e "$suffix_output" && ! -L "$suffix_output" \\\n    && -d "${suffix_output%/*}" && ! -L "${suffix_output%/*}" \\\n    && "$(readlink -f "${suffix_output%/*}")" == "${suffix_output%/*}" ]] || {\n    echo "FATAL: full TRAIN capture output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE=true\n  NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$suffix_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n}\n# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 DISPATCH\n  local frozen_bias_arg frozen_bias_reference=false\n  for frozen_bias_arg in "$@"; do\n    case "$frozen_bias_arg" in\n      "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_MODULE"|*/gx1/audit/capture_native_future_state_full_train_v1.py|gx1/audit/capture_native_future_state_full_train_v1.py)\n        frozen_bias_reference=true ;;\n    esac\n  done\n  if [[ "$frozen_bias_reference" == true ]]; then\n    configure_native_future_state_full_train_capture "$@"\n    return\n  fi\n# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 DISPATCH\n'), ('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 NESTED\n', 'before', '# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 NESTED\n  [[ "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE" ]] || {\n    echo "FATAL: nested full TRAIN capture mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE" == true ]]; then\n    [[ "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested full TRAIN capture plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 NESTED\n'), ('  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT="$NATIVE_FUTURE_STATE_THROUGHPUT" \\\n', 'before', '  --setenv=GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE="$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256="$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256" \\\n'))


def require_native_future_state_full_train_capture_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Read-only full TRAIN containment; no optimizer or training authority."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_FULL_TRAIN_CAPTURE_PLAN_SHA256_INVALID]")
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE": "true",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_TRAINER_MAX_WALL_SECONDS": "3600",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "3570",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_FULL_TRAIN_CAPTURE_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_20261122_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "full_train_capture",
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}



def native_future_state_full_train_capture_source_bridge(previous_source_owners, *, producer_commit="0e8fe2ebe117a7eed5f774db4ec09ffd299c8df0"):
    """Authenticate qualified77-owner sources; allow only named read-only capture additions."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_NATIVE_FULL_TRAIN_CAPTURE_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "0e8fe2ebe117a7eed5f774db4ec09ffd299c8df0", "PRODUCER_COMMIT")
    require(type(previous_source_owners) is dict and len(previous_source_owners) == 77, "OWNER_COUNT")
    for name, digest in previous_source_owners.items():
        require(type(name) is str, "OWNER_SCHEMA")
        path = Path(name)
        require(type(name) is str and not path.is_absolute() and ".." not in path.parts and str(path) == name
            and path.suffix in (".py", ".sh", ".ps1") and type(digest) is str
            and re.fullmatch("[0-9a-f]{64}", digest) is not None, "OWNER_SCHEMA")
    repo = Path(__file__).resolve().parents[2]
    guards = {"gx1/contracts/gx1_capped_execution_v1.py": "e0ee841c0f666eb980afc83454bba583b2403cab46aa2d8a279a6d59a90d39fc",
              "scripts/gx1_capped_run.sh": "3d82060b161d1880624eaf7ecb889c91fe58d15452d32baab780f6b939129481"}
    functions = {"require_native_future_state_full_train_capture_execution", "native_future_state_full_train_capture_source_bridge"}
    constant = "_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_GUARD_SHELL_ADDITIONS"
    report = {}
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(previous_source_owners.get(name) == hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_PARENT_GUARD_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            old_tree = ast.parse(original); new_tree = ast.parse(current)
            names = [n.name for n in new_tree.body if isinstance(n, ast.FunctionDef)]
            require(len(names) == len(set(names)), "DUPLICATE_FUNCTION")
            require(functions <= set(names), "EXACT_CAPTURE_FUNCTIONS")
            nodes = [n for n in new_tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == constant]
            require(len(nodes) == 1 and ast.literal_eval(nodes[0].value) == _NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_GUARD_SHELL_ADDITIONS,
                    "EXACT_CAPTURE_ROUTE_CONSTANT")
            new_tree.body = [n for n in new_tree.body if not (
                isinstance(n, ast.FunctionDef) and n.name in functions or n is nodes[0])]
            normalized = ast.dump(new_tree, include_attributes=False)
            require(normalized == ast.dump(old_tree, include_attributes=False), "ALL_PARENT_PYTHON_AST_UNCHANGED")
            proof = "all_parentmoduleAST_equal_after_only_two_named_capture_functions_and_one_literal_route_tuple"
        else:
            text = current.decode()
            require(len(_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_GUARD_SHELL_ADDITIONS) == 5, "EXACT_CAPTURE_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_CAPTURE_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            require(text.encode() == original, "ALL_PARENT_SHELL_BYTES_UNCHANGED")
            normalized = text
            proof = "all_parentshellbytes_equal_after_only_five_exact_named_capture_insertions"
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(), "proof": proof}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2,
            "other_parent_owners_to_be_checked_by_caller": 75, "owners": report}



_NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 STATE\n', 'before', '# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 STATE\nNATIVE_PROVIDER_HLC_INITIAL_MODULE=gx1.audit.native_future_state_provider_hlc_initial_v1\nNATIVE_PROVIDER_HLC_INITIAL=false\nNATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256=\n# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 MATCHERS\nmatches_bound_native_provider_hlc_initial() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_initial() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_initial "$@" || {\n    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local initial_output=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002\n  [[ ! -e "$initial_output" && ! -L "$initial_output" \\\n    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \\\n    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {\n    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_INITIAL=true\n  NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$initial_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 DISPATCH\n  local provider_initial_arg provider_initial_reference=false\n  for provider_initial_arg in "$@"; do\n    case "$provider_initial_arg" in\n      "$NATIVE_PROVIDER_HLC_INITIAL_MODULE"|*/gx1/audit/native_future_state_provider_hlc_initial_v1.py|gx1/audit/native_future_state_provider_hlc_initial_v1.py)\n        provider_initial_reference=true ;;\n    esac\n  done\n  if [[ "$provider_initial_reference" == true ]]; then\n    configure_native_provider_hlc_initial "$@"\n    return\n  fi\n# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 DISPATCH\n'), ('# BEGIN GX1 FULL TRAIN CAPTURE ROUTE 20261122 001 NESTED\n', 'before', '# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == "$NATIVE_PROVIDER_HLC_INITIAL" ]] || {\n    echo "FATAL: nested provider HLC INITIAL mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_INITIAL" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC INITIAL plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 NESTED\n'), ('  --setenv=GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE="$NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_INITIAL="$NATIVE_PROVIDER_HLC_INITIAL" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256="$NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256" \\\n'))


def require_native_provider_hlc_initial_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Read-only full TRAIN containment; no optimizer or training authority."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256_INVALID]")
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL": "true",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE": "false",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE": "",
        "GX1_TRAINER_MAX_WALL_SECONDS": "3600",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "3570",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_PROVIDER_HLC_INITIAL_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_initial",
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}



def native_provider_hlc_initial_source_bridge(previous_guard_owners, *, producer_commit="914e29a516d6762b59f6c8c21801e40af751e238"):
    """Prove only this new baseline route differs from the authenticated generation."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok: raise RuntimeError("[GX1_PROVIDER_HLC_INITIAL_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "914e29a516d6762b59f6c8c21801e40af751e238", "PRODUCER_COMMIT")
    guards = {'gx1/contracts/gx1_capped_execution_v1.py': '3a785511650592f990e644759f13d54da615313272bc1b6a4d6be89434b35563', 'scripts/gx1_capped_run.sh': '0c97154564fdc6cbb9759b989e825ed0776033ba357de22b01ff13bfdf93d1e4'}
    require(type(previous_guard_owners) is dict and previous_guard_owners == guards, "EXACT_ANCHOR_GUARDS")
    repo = Path(__file__).resolve().parents[2];report = {}
    functions = {"require_native_provider_hlc_initial_execution", "native_provider_hlc_initial_source_bridge"}
    constant = "_NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS"
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_ANCHOR_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            old = ast.parse(original);tree = ast.parse(current)
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            require(len(names) == len(set(names)) and functions <= set(names), "EXACT_NEW_FUNCTIONS")
            nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == constant]
            require(len(nodes) == 1 and ast.literal_eval(nodes[0].value) == _NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS, "EXACT_ROUTE_CONSTANT")
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in functions or n is nodes[0])]
            normalized = ast.dump(tree, include_attributes=False)
            require(normalized == ast.dump(old, include_attributes=False), "ALL_OLD_PYTHON_AST_UNCHANGED")
        else:
            text = current.decode();require(len(_NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS) == 5, "EXACT_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_NEW_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            require(text.encode() == original, "ALL_OLD_SHELL_BYTES_UNCHANGED");normalized = text
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest()}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2, "old_guard_routes_preserved": True, "owners": report}


_NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 STATE\n', 'before', '# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 STATE\nNATIVE_PROVIDER_HLC_REPLAY_MODULE=gx1.audit.native_provider_hlc_replay_v1\nNATIVE_PROVIDER_HLC_REPLAY=false\nNATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_REPLAY_LANE=\n# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 MATCHERS\nmatches_bound_native_provider_hlc_replay() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001.json"\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_replay() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC replay requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_replay "$@" || {\n    echo "FATAL: provider HLC replay CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local replay_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001_$7"\n  [[ ! -e "$replay_output" && ! -L "$replay_output" \\\n    && -d "${replay_output%/*}" && ! -L "${replay_output%/*}" \\\n    && "$(readlink -f "${replay_output%/*}")" == "${replay_output%/*}" ]] || {\n    echo "FATAL: provider HLC replay output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_REPLAY=true\n  NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256="$5"\n  NATIVE_PROVIDER_HLC_REPLAY_LANE="$7"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$replay_output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 DISPATCH\n  local provider_replay_arg provider_replay_reference=false\n  for provider_replay_arg in "$@"; do\n    case "$provider_replay_arg" in\n      "$NATIVE_PROVIDER_HLC_REPLAY_MODULE"|*/gx1/audit/native_provider_hlc_replay_v1.py|gx1/audit/native_provider_hlc_replay_v1.py)\n        provider_replay_reference=true ;;\n    esac\n  done\n  if [[ "$provider_replay_reference" == true ]]; then\n    configure_native_provider_hlc_replay "$@"\n    return\n  fi\n# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 DISPATCH\n'), ('# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 NESTED\n', 'before', '# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == "$NATIVE_PROVIDER_HLC_REPLAY" ]] || {\n    echo "FATAL: nested provider HLC replay mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_REPLAY" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "$NATIVE_PROVIDER_HLC_REPLAY_LANE" \\\n      && ( "$NATIVE_PROVIDER_HLC_REPLAY_LANE" == warm || "$NATIVE_PROVIDER_HLC_REPLAY_LANE" == cold ) \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 1800 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 1770 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC replay plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 NESTED\n'), ('  --setenv=GX1_NATIVE_PROVIDER_HLC_INITIAL="$NATIVE_PROVIDER_HLC_INITIAL" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_REPLAY="$NATIVE_PROVIDER_HLC_REPLAY" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256="$NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE="$NATIVE_PROVIDER_HLC_REPLAY_LANE" \\\n'))


def require_native_provider_hlc_replay_execution(*, plan_sha256: str, lane: str) -> dict[str, Any]:
    """Contain only the exact technical replay; its leaf owns optimizer admission."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256_INVALID]")
    if type(lane) is not str or lane not in ("warm", "cold"):
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE_INVALID]")
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY": "true",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE": lane,
        "GX1_NATIVE_PROVIDER_HLC_INITIAL": "false",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE": "false",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE": "",
        "GX1_TRAINER_MAX_WALL_SECONDS": "1800",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "1770",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_PROVIDER_HLC_REPLAY_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_PROVIDER_HLC_COLD_REPLAY_20261123_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / (basename + "_" + lane)
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_replay", "lane": lane,
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": False, "resource_proof_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 1800, "internal_seconds": 1770, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def native_provider_hlc_replay_source_bridge(previous_guard_owners, *, producer_commit="27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49"):
    """Prove only this replay route differs from the qualified INITIAL002 generation."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok: raise RuntimeError("[GX1_PROVIDER_HLC_REPLAY_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49", "PRODUCER_COMMIT")
    guards = {'gx1/contracts/gx1_capped_execution_v1.py': 'dc34de2817016bbba33c27c4b8a5bcda8ea4f77fb978daa69a81aee3b4c5043b', 'scripts/gx1_capped_run.sh': '791527f0c502867d2b5a78cc2cd976de70e3f5d02eee67407aa4d2a74d062e05'}
    require(type(previous_guard_owners) is dict and previous_guard_owners == guards, "EXACT_ANCHOR_GUARDS")
    repo = Path(__file__).resolve().parents[2];report = {}
    functions = {"require_native_provider_hlc_replay_execution", "native_provider_hlc_replay_source_bridge"}
    constant = "_NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS"
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_ANCHOR_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            old = ast.parse(original);tree = ast.parse(current)
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            require(len(names) == len(set(names)) and functions <= set(names), "EXACT_NEW_FUNCTIONS")
            nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == constant]
            require(len(nodes) == 1 and ast.literal_eval(nodes[0].value) == _NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS, "EXACT_ROUTE_CONSTANT")
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in functions or n is nodes[0])]
            normalized = ast.dump(tree, include_attributes=False)
            require(normalized == ast.dump(old, include_attributes=False), "ALL_OLD_PYTHON_AST_UNCHANGED")
        else:
            text = current.decode();require(len(_NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS) == 5, "EXACT_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_NEW_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            require(text.encode() == original, "ALL_OLD_SHELL_BYTES_UNCHANGED");normalized = text
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest()}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2, "old_guard_routes_preserved": True, "owners": report}


_NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 STATE\n', 'before', '# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 STATE\nNATIVE_PROVIDER_HLC_LEARNING_MODULE=gx1.audit.native_provider_hlc_learning_v1\nNATIVE_PROVIDER_HLC_LEARNING=false\nNATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256=\n# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 MATCHERS\nmatches_bound_native_provider_hlc_learning() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_LEARNING_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_learning() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC learning requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_learning "$@" || {\n    echo "FATAL: provider HLC learning CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001"\n  [[ ! -e "$learning_output" && ! -L "$learning_output" \\\n    && -d "${learning_output%/*}" && ! -L "${learning_output%/*}" \\\n    && "$(readlink -f "${learning_output%/*}")" == "${learning_output%/*}" ]] || {\n    echo "FATAL: provider HLC learning output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_LEARNING=true\n  NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$learning_output"\n  TRAINER_MAX_WALL_SECONDS=12000\n  TRAINER_MODEL_MAX_WALL_SECONDS=11970\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 DISPATCH\n  local provider_learning_arg provider_learning_reference=false\n  for provider_learning_arg in "$@"; do\n    case "$provider_learning_arg" in\n      "$NATIVE_PROVIDER_HLC_LEARNING_MODULE"|*/gx1/audit/native_provider_hlc_learning_v1.py|gx1/audit/native_provider_hlc_learning_v1.py)\n        provider_learning_reference=true ;;\n    esac\n  done\n  if [[ "$provider_learning_reference" == true ]]; then\n    configure_native_provider_hlc_learning "$@"\n    return\n  fi\n# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 DISPATCH\n'), ('# BEGIN GX1 PROVIDER HLC REPLAY ROUTE 20261123 001 NESTED\n', 'before', '# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-false}" == "$NATIVE_PROVIDER_HLC_LEARNING" ]] || {\n    echo "FATAL: nested provider HLC learning mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_LEARNING" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_ATTENDED_STAGE_REQUIRED:-}" == false \\\n      && -z "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" \\\n      && -z "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 12000 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 11970 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC learning plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 NESTED\n'), ('  --setenv=GX1_NATIVE_PROVIDER_HLC_REPLAY="$NATIVE_PROVIDER_HLC_REPLAY" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING="$NATIVE_PROVIDER_HLC_LEARNING" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256="$NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256" \\\n'))


def require_native_provider_hlc_learning_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Contain only the exact full TRAIN learning plan; its leaf owns optimizer admission."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256_INVALID]")
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING": "true",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_PROVIDER_HLC_REPLAY": "false",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE": "",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL": "false",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE": "false",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE": "",
        "GX1_TRAINER_MAX_WALL_SECONDS": "12000",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "11970",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_PROVIDER_HLC_LEARNING_ENV_INVALID] field={name}")
    basename = "NATIVE_PROVIDER_HLC_ONE_EPOCH_20261123_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    repo = Path(__file__).resolve().parents[2]
    if str(repo) != "/home/andre2/src/GX1_CURRENT" or Path.cwd().resolve(strict=True) != repo:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_CWD_INVALID]")
    command = [str(repo / ".venv/bin/python"), "-m", "gx1.audit.native_provider_hlc_learning_v1",
               "--plan-sha256", plan_sha256]
    if sys.orig_argv not in (command, command + ["--preflight"]) or sys.executable != command[0]:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_CLI_INVALID]")
    import hashlib
    try:
        if (not stat.S_ISREG(plan_path.lstat().st_mode) or plan_path.resolve(strict=True) != plan_path
                or hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256):
            raise ValueError("plan path/hash")
        parent = output_root.parent
        if (output_root.exists() or output_root.is_symlink() or not stat.S_ISDIR(parent.lstat().st_mode)
                or parent.resolve(strict=True) != parent):
            raise ValueError("output path")
    except (OSError, ValueError) as exc:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_OUTPUT_INVALID]") from exc
    proof = require_guarded_cuda_trainer_execution()
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_learning",
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": False, "resource_proof_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 12000, "internal_seconds": 11970, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def native_provider_hlc_learning_source_bridge(previous_guard_owners, *, producer_commit="ab6454433d269fdcd77532a4c50c036a2f45bc3c"):
    """Prove only this learning route differs from the qualified cold replay generation."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok: raise RuntimeError("[GX1_PROVIDER_HLC_LEARNING_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "ab6454433d269fdcd77532a4c50c036a2f45bc3c", "PRODUCER_COMMIT")
    guards = {'gx1/contracts/gx1_capped_execution_v1.py': '845f46ac635714ec2d0bc9529e6d059573c8aa038a9869086232791fb380b4e5', 'scripts/gx1_capped_run.sh': 'bd19d5524e6043ab9dfbc6e261d20a078cb9c9d8b1d67175eb93576ed44d8641'}
    require(type(previous_guard_owners) is dict and previous_guard_owners == guards, "EXACT_ANCHOR_GUARDS")
    repo = Path(__file__).resolve().parents[2];report = {}
    functions = {"require_native_provider_hlc_learning_execution", "native_provider_hlc_learning_source_bridge"}
    constant = "_NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS"
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_ANCHOR_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            old = ast.parse(original);tree = ast.parse(current)
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            require(len(names) == len(set(names)) and functions <= set(names), "EXACT_NEW_FUNCTIONS")
            nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == constant]
            require(len(nodes) == 1 and ast.literal_eval(nodes[0].value) == _NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS, "EXACT_ROUTE_CONSTANT")
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in functions or n is nodes[0])]
            normalized = ast.dump(tree, include_attributes=False)
            require(normalized == ast.dump(old, include_attributes=False), "ALL_OLD_PYTHON_AST_UNCHANGED")
        else:
            text = current.decode();require(len(_NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS) == 5, "EXACT_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_LEARNING_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_NEW_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            require(text.encode() == original, "ALL_OLD_SHELL_BYTES_UNCHANGED");normalized = text
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest()}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2, "old_guard_routes_preserved": True, "owners": report}


_NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 STATE\n', 'before', '# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 STATE\nNATIVE_PROVIDER_HLC_LEARNING_RESUME_MODULE=gx1.audit.native_provider_hlc_learning_v1\nNATIVE_PROVIDER_HLC_LEARNING_RESUME=false\nNATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256=\n# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 MATCHERS\nmatches_bound_native_provider_hlc_learning_resume() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_004.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME_MODULE" \\\n    && "$4" == --resume-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_learning_resume() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC learning resume requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_learning_resume "$@" || {\n    echo "FATAL: provider HLC learning resume CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_004"\n  [[ ! -e "$learning_output" && ! -L "$learning_output" \\\n    && -d "${learning_output%/*}" && ! -L "${learning_output%/*}" \\\n    && "$(readlink -f "${learning_output%/*}")" == "${learning_output%/*}" ]] || {\n    echo "FATAL: provider HLC learning resume output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_LEARNING_RESUME=true\n  NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$learning_output"\n  TRAINER_MAX_WALL_SECONDS=12000\n  TRAINER_MODEL_MAX_WALL_SECONDS=11970\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 DISPATCH\n  local provider_learning_resume_arg provider_learning_resume_reference=false\n  for provider_learning_resume_arg in "$@"; do\n    case "$provider_learning_resume_arg" in\n      --resume-plan-sha256|--resume-plan-sha256=*)\n        provider_learning_resume_reference=true ;;\n    esac\n  done\n  if [[ "$provider_learning_resume_reference" == true ]]; then\n    configure_native_provider_hlc_learning_resume "$@"\n    return\n  fi\n# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 DISPATCH\n'), ('# BEGIN GX1 PROVIDER HLC LEARNING ROUTE 20261123 001 NESTED\n', 'before', '# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-false}" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256" ]] || {\n    echo "FATAL: nested provider HLC learning resume mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_LEARNING_RESUME" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_ATTENDED_STAGE_REQUIRED:-}" == false \\\n      && -z "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" \\\n      && -z "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 12000 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 11970 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC learning resume plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 NESTED\n'), ('  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING="$NATIVE_PROVIDER_HLC_LEARNING" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME="$NATIVE_PROVIDER_HLC_LEARNING_RESUME" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256="$NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256" \\\n'))


def require_native_provider_hlc_learning_resume_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Contain only the exact cursor512 resume; its leaf owns optimizer admission."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256_INVALID]")
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME": "true",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_PROVIDER_HLC_LEARNING": "false",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY": "false",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE": "",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL": "false",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE": "false",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE": "",
        "GX1_TRAINER_MAX_WALL_SECONDS": "12000",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "11970",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_ENV_INVALID] field={name}")
    basename = "NATIVE_PROVIDER_HLC_ONE_EPOCH_RESUME_20261123_004"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    repo = Path(__file__).resolve().parents[2]
    if str(repo) != "/home/andre2/src/GX1_CURRENT" or Path.cwd().resolve(strict=True) != repo:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_CWD_INVALID]")
    command = [str(repo / ".venv/bin/python"), "-m", "gx1.audit.native_provider_hlc_learning_v1",
               "--resume-plan-sha256", plan_sha256]
    if sys.orig_argv not in (command, command + ["--preflight"]) or sys.executable != command[0]:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_CLI_INVALID]")
    import hashlib
    try:
        if (not stat.S_ISREG(plan_path.lstat().st_mode) or plan_path.resolve(strict=True) != plan_path
                or hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256):
            raise ValueError("plan path/hash")
        parent = output_root.parent
        if (output_root.exists() or output_root.is_symlink() or not stat.S_ISDIR(parent.lstat().st_mode)
                or parent.resolve(strict=True) != parent):
            raise ValueError("output path")
    except (OSError, ValueError) as exc:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_OUTPUT_INVALID]") from exc
    proof = require_guarded_cuda_trainer_execution()
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_learning_resume",
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": False, "resource_proof_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 12000, "internal_seconds": 11970, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def native_provider_hlc_learning_resume_source_bridge(previous_guard_owners, *, producer_commit="aa6f23bb6ad0d867fc23f3604b8399e89b633ed6"):
    """Prove only this resume route differs from the stopped learning generation."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok: raise RuntimeError("[GX1_PROVIDER_HLC_LEARNING_RESUME_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "aa6f23bb6ad0d867fc23f3604b8399e89b633ed6", "PRODUCER_COMMIT")
    guards = {'gx1/contracts/gx1_capped_execution_v1.py': '2a26866baba68182d460a38f556004b043d8aef1aaaa8d21e4eaecc77c534f8c', 'scripts/gx1_capped_run.sh': '313d1848242840f99637b66df7aad46cb3847129cf8675009901ce79fc3a4b31'}
    require(type(previous_guard_owners) is dict and previous_guard_owners == guards, "EXACT_ANCHOR_GUARDS")
    repo = Path(__file__).resolve().parents[2];report = {}
    functions = {"require_native_provider_hlc_learning_resume_execution", "native_provider_hlc_learning_resume_source_bridge"}
    constant = "_NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS"
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_ANCHOR_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            old = ast.parse(original);tree = ast.parse(current)
            names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            require(len(names) == len(set(names)) and functions <= set(names), "EXACT_NEW_FUNCTIONS")
            nodes = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == constant]
            require(len(nodes) == 1 and ast.literal_eval(nodes[0].value) == _NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS, "EXACT_ROUTE_CONSTANT")
            tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in functions or n is nodes[0])]
            normalized = ast.dump(tree, include_attributes=False)
            require(normalized == ast.dump(old, include_attributes=False), "ALL_OLD_PYTHON_AST_UNCHANGED")
        else:
            text = current.decode();require(len(_NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS) == 5, "EXACT_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_LEARNING_RESUME_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_NEW_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            require(text.encode() == original, "ALL_OLD_SHELL_BYTES_UNCHANGED");normalized = text
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest()}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2, "old_guard_routes_preserved": True, "owners": report}


# BEGIN GX1 PROVIDER HLC ALL TF GUARD ROUTES 20261124 001
_NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 STATE\n', 'before', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 STATE\nNATIVE_PROVIDER_HLC_ALL_TF_INITIAL=false\nNATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_ALL_TF_REPLAY=false\nNATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE=\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 MATCHERS\nmatches_bound_native_provider_hlc_all_tf_initial() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_001.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \\\n    && "$4" == --all-tf-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_all_tf_initial() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_all_tf_initial "$@" || {\n    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local initial_output=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_001\n  [[ ! -e "$initial_output" && ! -L "$initial_output" \\\n    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \\\n    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {\n    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=true\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$initial_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n\nmatches_bound_native_provider_hlc_all_tf_replay() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001.json"\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" \\\n    && "$4" == --all-tf-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_all_tf_replay() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC replay requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_all_tf_replay "$@" || {\n    echo "FATAL: provider HLC replay CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local replay_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001_$7"\n  [[ ! -e "$replay_output" && ! -L "$replay_output" \\\n    && -d "${replay_output%/*}" && ! -L "${replay_output%/*}" \\\n    && "$(readlink -f "${replay_output%/*}")" == "${replay_output%/*}" ]] || {\n    echo "FATAL: provider HLC replay output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY=true\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$5"\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$7"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$replay_output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 DISPATCH\n  if [[ "${4:-}" == --all-tf-plan-sha256 ]]; then\n    case "${3:-}" in\n      "$NATIVE_PROVIDER_HLC_INITIAL_MODULE")\n        configure_native_provider_hlc_all_tf_initial "$@"; return ;;\n      "$NATIVE_PROVIDER_HLC_REPLAY_MODULE")\n        configure_native_provider_hlc_all_tf_replay "$@"; return ;;\n      *)\n        echo "FATAL: ALL-TF plan flag requires its exact native module" >&2\n        exit 75 ;;\n    esac\n  fi\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 DISPATCH\n'), ('# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 NESTED\n', 'before', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 NESTED\n# BEGIN GX1 PROVIDER HLC ALL TF INITIAL ROUTE 20261124 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" ]] || {\n    echo "FATAL: nested provider HLC INITIAL mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC INITIAL plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC ALL TF INITIAL ROUTE 20261124 001 NESTED\n# BEGIN GX1 PROVIDER HLC ALL TF REPLAY ROUTE 20261124 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" ]] || {\n    echo "FATAL: nested provider HLC replay mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" \\\n      && ( "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" == warm || "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" == cold ) \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 1800 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 1770 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC replay plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC ALL TF REPLAY ROUTE 20261124 001 NESTED\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 NESTED\n'), ('  --setenv=GX1_NATIVE_PROVIDER_HLC_INITIAL="$NATIVE_PROVIDER_HLC_INITIAL" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL="$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" \\\n'))


def require_native_provider_hlc_all_tf_initial_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Read-only full TRAIN containment; no optimizer or training authority."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256_INVALID]")
    _native_provider_hlc_all_tf_bound_plan(plan_sha256=plan_sha256, lane=None)
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL": "true",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_PROVIDER_HLC_INITIAL": "false",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY": "false",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE": "",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY": "false",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE": "",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING": "false",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME": "false",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE": "false",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE": "",
        "GX1_TRAINER_MAX_WALL_SECONDS": "3600",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "3570",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_all_tf_initial",
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def require_native_provider_hlc_all_tf_replay_execution(*, plan_sha256: str, lane: str) -> dict[str, Any]:
    """Contain only the exact technical replay; its leaf owns optimizer admission."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256_INVALID]")
    if type(lane) is not str or lane not in ("warm", "cold"):
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE_INVALID]")
    _native_provider_hlc_all_tf_bound_plan(plan_sha256=plan_sha256, lane=lane)
    expected = {
        "GX1_CAPPED_MEMORY_BYTES": "21474836480",
        "GX1_CAPPED_SWAP_BYTES": "536870912",
        "GX1_CAPPED_TASKS_MAX": "128",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY": "true",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256": plan_sha256,
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE": lane,
        "GX1_NATIVE_PROVIDER_HLC_INITIAL": "false",
        "GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY": "false",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE": "",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL": "false",
        "GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING": "false",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256": "",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME": "false",
        "GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE": "false",
        "GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN": "false",
        "GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT": "false",
        "GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING": "false",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_TRAINING_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION": "false",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS": "false",
        "GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CUDA_LANE": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256": "",
        "GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE": "",
        "GX1_TRAINER_MAX_WALL_SECONDS": "1800",
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": "1770",
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": "false",
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": "250",
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": "250",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "75",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "80",
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": "12288",
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        "GX1_TRAINER_CPU_MAX_TEMP_C": "80",
        "GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W": "125",
        "GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES": "8589934592",
        "GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES": "8589934592",
        "GX1_POWER_BENCHMARK_SCOPE_JSON": "",
        "GX1_POWER_BENCHMARK_SCOPE_SHA256": "",
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / (basename + "_" + lane)
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_all_tf_replay", "lane": lane,
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": False, "resource_proof_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 1800, "internal_seconds": 1770, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def _native_provider_hlc_all_tf_bound_plan(*, plan_sha256, lane):
    """Bind the new command and None-policy plan before any CUDA allocation."""
    import hashlib
    import json
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_PROVIDER_HLC_ALL_TF_BINDING_" + reason + "]")
    require(type(plan_sha256) is str and re.fullmatch("[0-9a-f]{64}", plan_sha256) is not None, "SHA")
    require(lane is None or type(lane) is str and lane in ("warm", "cold"), "LANE")
    repo = Path(__file__).resolve().parents[2]
    basename = ("NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_001" if lane is None
                else "NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001")
    owner = ("native_future_state_provider_hlc_initial_v1.py" if lane is None
             else "native_provider_hlc_replay_v1.py")
    arguments = ["--all-tf-plan-sha256", plan_sha256]
    if lane is not None:
        arguments += ["--lane", lane]
    if sys.argv[-1:] == ["--preflight"]:
        arguments += ["--preflight"]
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / "gx1/audit" / owner
            and Path(sys.executable).absolute() == repo / ".venv/bin/python"
            and sys.argv[1:] == arguments, "EXACT_ARGV")
    plan_path = repo / "configs/research" / (basename + ".json")
    root = Path("/home/andre2/GX1_RUNS") / (basename + ("_" + lane if lane is not None else ""))
    require(plan_path.is_file() and not plan_path.is_symlink() and plan_path.resolve() == plan_path, "CANONICAL_PLAN")
    require(not root.exists() and not root.is_symlink() and root.parent.is_dir()
            and not root.parent.is_symlink() and root.parent.resolve() == root.parent, "FRESH_ROOT")
    try:
        raw = plan_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == plan_sha256, "PLAN_SHA")
        plan = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as error:
        raise RuntimeError("[GX1_PROVIDER_HLC_ALL_TF_BINDING_PLAN_JSON]") from error
    require(type(plan) is dict and plan.get("stage") == basename.lower(), "PLAN_STAGE")
    recipe, contract = plan.get("recipe"), plan.get("model_contract")
    require(type(recipe) is dict and "backbone_gradient_policy" in recipe
            and recipe["backbone_gradient_policy"] is None and type(contract) is dict
            and "backbone_gradient_policy" not in contract
            and contract.get("contract_sha256") == "6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0",
            "EXPLICIT_NONE_CONTRACT")
    if lane is None:
        require(plan.get("output_root") == str(root), "PLAN_ROOT")
    else:
        require(plan.get("output_roots") == {name: str(root.parent / (basename + "_" + name))
                                           for name in ("warm", "cold")}, "PLAN_ROOTS")


def _native_provider_hlc_all_tf_source_bridge(previous_guard_owners, *, producer_commit):
    """Remove only exact ALL-TF additions and prove both f33 parent files."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_PROVIDER_HLC_ALL_TF_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8", "PRODUCER_COMMIT")
    guards = {
        "gx1/contracts/gx1_capped_execution_v1.py": "bbe6d6fe74ae289365759718bef1b045550c8250df08f937d87716b93a6d2c90",
        "scripts/gx1_capped_run.sh": "ca334f3e843bd0d227883b783421c6bbc475a808b3b9a3d8c4ef6a0b622c5cc1",
    }
    require(type(previous_guard_owners) is dict and previous_guard_owners == guards, "EXACT_ANCHOR_GUARDS")
    repo = Path(__file__).resolve().parents[2]
    functions = {
        "_native_provider_hlc_all_tf_bound_plan", "_native_provider_hlc_all_tf_source_bridge",
        "require_native_provider_hlc_all_tf_initial_execution", "require_native_provider_hlc_all_tf_replay_execution",
        "native_provider_hlc_all_tf_initial_source_bridge", "native_provider_hlc_all_tf_replay_source_bridge",
    }
    report = {}
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_ANCHOR_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            begin = b"# BEGIN GX1 PROVIDER HLC ALL TF GUARD ROUTES 20261124 001\n"
            end = b"# END GX1 PROVIDER HLC ALL TF GUARD ROUTES 20261124 001\n\n"
            require(current.count(begin) == current.count(end) == 1, "SINGLE_ADDITIVE_BLOCK")
            first, tail = current.split(begin)
            block, last = tail.split(end)
            nodes = ast.parse(block).body
            found = [node.name for node in nodes if isinstance(node, ast.FunctionDef)]
            require(set(found) == functions and len(found) == len(functions), "EXACT_ADDITIVE_FUNCTIONS")
            constants = [node for node in nodes if isinstance(node, ast.Assign)]
            require(len(nodes) == len(functions) + 1 and len(constants) == 1
                    and len(constants[0].targets) == 1 and isinstance(constants[0].targets[0], ast.Name)
                    and constants[0].targets[0].id == "_NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS"
                    and ast.literal_eval(constants[0].value) == _NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS,
                    "EXACT_ROUTE_CONSTANT")
            normalized = first + last
            require(normalized == original, "ALL_OLD_PYTHON_BYTES_UNCHANGED")
        else:
            normalized = current.decode()
            require(len(_NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS) == 5, "EXACT_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(normalized.count(fragment) == normalized.count(composite) == 1, "EXACT_SHELL_INSERTION")
                normalized = normalized.replace(composite, anchor, 1)
            normalized = normalized.encode()
            require(normalized == original, "ALL_OLD_SHELL_BYTES_UNCHANGED")
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized).hexdigest()}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2,
            "old_guard_routes_preserved": True, "owners": report}


def native_provider_hlc_all_tf_initial_source_bridge(previous_guard_owners, *,
        producer_commit="f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8"):
    return _native_provider_hlc_all_tf_source_bridge(previous_guard_owners, producer_commit=producer_commit)


def native_provider_hlc_all_tf_replay_source_bridge(previous_guard_owners, *,
        producer_commit="f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8"):
    return _native_provider_hlc_all_tf_source_bridge(previous_guard_owners, producer_commit=producer_commit)
# END GX1 PROVIDER HLC ALL TF GUARD ROUTES 20261124 001

def _main() -> int:
    parser = argparse.ArgumentParser(description="Canonical capped-runner lock proof")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--lock-path", action="store_true")
    operation.add_argument("--verify-lock-ancestry", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.lock_path:
            print(canonical_heavy_job_lock_path())
        else:
            require_capped_lock_ancestry()
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
