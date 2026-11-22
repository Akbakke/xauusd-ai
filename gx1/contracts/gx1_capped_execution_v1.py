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


_NATIVE_FUTURE_STATE_BIAS_SUFFIX_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 STATE\n', 'before', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 STATE\nNATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE=gx1.audit.diagnose_native_future_state_bias_suffix_v1\nNATIVE_FUTURE_STATE_BIAS_SUFFIX=false\nNATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256=\n# END GX1 BIAS SUFFIX ROUTE 20261122 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 MATCHERS\nmatches_bound_native_future_state_bias_suffix() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_001.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE" \\\n    && "$4" == --plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_future_state_bias_suffix() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: bias suffix order study requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_future_state_bias_suffix "$@" || {\n    echo "FATAL: bias suffix order study CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local suffix_output=/home/andre2/GX1_RUNS/NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_001\n  [[ ! -e "$suffix_output" && ! -L "$suffix_output" \\\n    && -d "${suffix_output%/*}" && ! -L "${suffix_output%/*}" \\\n    && "$(readlink -f "${suffix_output%/*}")" == "${suffix_output%/*}" ]] || {\n    echo "FATAL: bias suffix output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_FUTURE_STATE_BIAS_SUFFIX=true\n  NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$suffix_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3600\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=300\n  TRAINER_GPU_MAX_POWER_DRAW_W=310\n  TRAINER_GPU_MAX_CORE_TEMP_C=85\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n}\n# END GX1 BIAS SUFFIX ROUTE 20261122 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 DISPATCH\n  local frozen_bias_arg frozen_bias_reference=false\n  for frozen_bias_arg in "$@"; do\n    case "$frozen_bias_arg" in\n      "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_MODULE"|*/gx1/audit/diagnose_native_future_state_bias_suffix_v1.py|gx1/audit/diagnose_native_future_state_bias_suffix_v1.py)\n        frozen_bias_reference=true ;;\n    esac\n  done\n  if [[ "$frozen_bias_reference" == true ]]; then\n    configure_native_future_state_bias_suffix "$@"\n    return\n  fi\n# END GX1 BIAS SUFFIX ROUTE 20261122 DISPATCH\n'), ('# BEGIN GX1 FUTURE LEAF ROUTES 20261121 NESTED\n', 'before', '# BEGIN GX1 BIAS SUFFIX ROUTE 20261122 NESTED\n  [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX" ]] || {\n    echo "FATAL: nested bias suffix mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_FUTURE_STATE_BIAS_SUFFIX" == true ]]; then\n    [[ "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "$NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 300 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 310 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 85 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 ]] || {\n      echo "FATAL: nested bias suffix plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 BIAS SUFFIX ROUTE 20261122 NESTED\n'), ('  --setenv=GX1_NATIVE_FUTURE_STATE_THROUGHPUT="$NATIVE_FUTURE_STATE_THROUGHPUT" \\\n', 'before', '  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX="$NATIVE_FUTURE_STATE_BIAS_SUFFIX" \\\n  --setenv=GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256="$NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256" \\\n'))


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
            raise RuntimeError(f"[GX1_NATIVE_SUFFIX_BIAS_ENV_INVALID] field={name}")
    proof = require_guarded_cuda_trainer_execution()
    basename = "NATIVE_FUTURE_STATE_BIAS_SUFFIX_ORDER_20261122_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / basename
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "bias_suffix_order",
            "training_authority_granted": False, "optimizer_authority_granted": False,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}



def native_future_state_bias_guard_source_bridge(previous_source_owners, *, producer_commit):
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
    report = {}
    for name in guards:
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        expected_sha = hashlib.sha256(original).hexdigest()
        require(previous_source_owners.get(name) == expected_sha, "AUTHENTIC_MAIN_GUARD_HASH")
        current = (repo / name).read_bytes()
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
                "require_native_future_state_training_execution", "require_native_future_state_bias_frozen_execution", "require_native_future_state_bias_suffix_execution", "native_future_state_bias_guard_source_bridge", "native_future_state_guard_source_bridge"}
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
