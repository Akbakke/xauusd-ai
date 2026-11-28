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
_NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 STATE\n', 'before', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 STATE\nNATIVE_PROVIDER_HLC_ALL_TF_INITIAL=false\nNATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_ALL_TF_REPLAY=false\nNATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE=\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 STATE\n'), ('validate_target_command() {\n', 'before', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 MATCHERS\nmatches_bound_native_provider_hlc_all_tf_initial() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002.json"\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \\\n    && "$4" == --all-tf-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_all_tf_initial() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_all_tf_initial "$@" || {\n    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local initial_output=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002\n  [[ ! -e "$initial_output" && ! -L "$initial_output" \\\n    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \\\n    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {\n    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=true\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$initial_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n\nmatches_bound_native_provider_hlc_all_tf_replay() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001.json"\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" \\\n    && "$4" == --all-tf-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_all_tf_replay() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC replay requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_all_tf_replay "$@" || {\n    echo "FATAL: provider HLC replay CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local replay_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001_$7"\n  [[ ! -e "$replay_output" && ! -L "$replay_output" \\\n    && -d "${replay_output%/*}" && ! -L "${replay_output%/*}" \\\n    && "$(readlink -f "${replay_output%/*}")" == "${replay_output%/*}" ]] || {\n    echo "FATAL: provider HLC replay output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY=true\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$5"\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$7"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$replay_output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 DISPATCH\n  if [[ "${4:-}" == --all-tf-plan-sha256 ]]; then\n    case "${3:-}" in\n      "$NATIVE_PROVIDER_HLC_INITIAL_MODULE")\n        configure_native_provider_hlc_all_tf_initial "$@"; return ;;\n      "$NATIVE_PROVIDER_HLC_REPLAY_MODULE")\n        configure_native_provider_hlc_all_tf_replay "$@"; return ;;\n      *)\n        echo "FATAL: ALL-TF plan flag requires its exact native module" >&2\n        exit 75 ;;\n    esac\n  fi\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 DISPATCH\n'), ('# BEGIN GX1 PROVIDER HLC INITIAL ROUTE 20261123 001 NESTED\n', 'before', '# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 NESTED\n# BEGIN GX1 PROVIDER HLC ALL TF INITIAL ROUTE 20261124 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" ]] || {\n    echo "FATAL: nested provider HLC INITIAL mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 3600 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 3570 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC INITIAL plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC ALL TF INITIAL ROUTE 20261124 001 NESTED\n# BEGIN GX1 PROVIDER HLC ALL TF REPLAY ROUTE 20261124 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" ]] || {\n    echo "FATAL: nested provider HLC replay mode differs from protected parent" >&2\n    exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" == true ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" \\\n      && ( "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" == warm || "$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" == cold ) \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == canonical \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == 1800 \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == 1770 \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == 0 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == 1 \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == 250 \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == 75 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == 12288 \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == 80 \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == 125 \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == 8589934592 \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == 8589934592 ]] || {\n      echo "FATAL: nested provider HLC replay plan/deadline/hardware differs from protected parent" >&2\n      exit 75\n    }\n  fi\n# END GX1 PROVIDER HLC ALL TF REPLAY ROUTE 20261124 001 NESTED\n# END GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 NESTED\n'), ('  --setenv=GX1_NATIVE_PROVIDER_HLC_INITIAL="$NATIVE_PROVIDER_HLC_INITIAL" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL="$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE" \\\n'))


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
    basename = "NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002"
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
    proof = _require_provider_hlc_all_tf_replay_environment(plan_sha256, lane)
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
    basename = ("NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002" if lane is None
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

# BEGIN GX1 ALL TF INITIAL SEGMENT GUARD ROUTE 20261124 003
_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_SEGMENT_GUARD_SHELL_ADDITIONS = (('validate_target_command() {\n', 'before', '# BEGIN GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 MATCHERS\nmatches_bound_native_provider_hlc_all_tf_initial_segment() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \\\n    && "$4" == --all-tf-score-segment-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local base_root=/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003\n  local plan_dir="$base_root/PLANS"\n  local expected_plan="$plan_dir/$5.json"\n  [[ -d "$base_root" && ! -L "$base_root" && "$(readlink -f "$base_root")" == "$base_root" \\\n    && -d "$plan_dir" && ! -L "$plan_dir" && "$(readlink -f "$plan_dir")" == "$plan_dir" \\\n    && -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_all_tf_initial_segment() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_provider_hlc_all_tf_initial_segment "$@" || {\n    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local initial_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003/SEGMENT_$5"\n  [[ ! -e "$initial_output" && ! -L "$initial_output" \\\n    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \\\n    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {\n    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=true\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$initial_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n\n# END GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 MATCHERS\n\n'), ('validate_target_command() {\n', 'after', '# BEGIN GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 DISPATCH\n  if [[ "${4:-}" == --all-tf-score-segment-plan-sha256 ]]; then\n    configure_native_provider_hlc_all_tf_initial_segment "$@"\n    return\n  fi\n# END GX1 ALL TF INITIAL SEGMENT ROUTE 20261124 003 DISPATCH\n'))


def require_native_provider_hlc_all_tf_initial_segment_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Read-only full TRAIN containment; no optimizer or training authority."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256_INVALID]")
    bound = _native_provider_hlc_all_tf_initial_segment_bound_plan(plan_sha256=plan_sha256)
    proof = _require_provider_hlc_all_tf_initial_environment(plan_sha256)
    plan_path = Path(bound["plan"]["path"])
    output_root = Path(bound["output_root"])
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_all_tf_initial_segment",
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": True,
            "base_plan": bound["base_plan"], "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def _native_provider_hlc_all_tf_initial_segment_bound_plan(*, plan_sha256):
    """Allow only SHA-named scoring segments inside the fixed fresh003 lineage."""
    import hashlib
    import json
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_PROVIDER_HLC_INITIAL_SEGMENT_BINDING_" + reason + "]")
    require(type(plan_sha256) is str and re.fullmatch("[0-9a-f]{64}", plan_sha256) is not None, "SHA")
    repo = Path(__file__).resolve().parents[2]
    arguments = ["--all-tf-score-segment-plan-sha256", plan_sha256]
    if sys.argv[-1:] == ["--preflight"]:
        arguments += ["--preflight"]
    require(Path.cwd().resolve() == repo
            and Path(sys.argv[0]).resolve() == repo / "gx1/audit/native_future_state_provider_hlc_initial_v1.py"
            and Path(sys.executable).absolute() == repo / ".venv/bin/python"
            and sys.argv[1:] == arguments, "EXACT_ARGV")
    basename = "NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003"
    base_root = Path("/home/andre2/GX1_RUNS") / basename
    plans = base_root / "PLANS"
    require(base_root.is_dir() and not base_root.is_symlink() and base_root.resolve() == base_root
            and plans.is_dir() and not plans.is_symlink() and plans.resolve() == plans, "CANONICAL_BASE_ROOT")
    plan_path = plans / (plan_sha256 + ".json")
    output_root = base_root / ("SEGMENT_" + plan_sha256)
    require(not output_root.exists() and not output_root.is_symlink(), "FRESH_SEGMENT_ROOT")
    def read_bound(path, digest):
        require(type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest) is not None, "REFERENCE_SHA")
        require(path.is_file() and not path.is_symlink() and path.resolve() == path, "CANONICAL_PLAN")
        try:
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == digest, "PLAN_SHA")
            value = json.loads(raw)
        except (OSError, ValueError, UnicodeError) as error:
            raise RuntimeError("[GX1_PROVIDER_HLC_INITIAL_SEGMENT_BINDING_PLAN_JSON]") from error
        require(type(value) is dict, "PLAN_OBJECT")
        return value
    segment = read_bound(plan_path, plan_sha256)
    require(segment.get("schema_version") == "gx1_native_provider_hlc_all_tf_initial_segment_plan_v1"
            and segment.get("stage") == basename.lower() and segment.get("base_root") == str(base_root)
            and "plan_path" not in segment and "output_root" not in segment, "SEGMENT_SCHEMA")
    base_ref = segment.get("base_plan")
    base_path = repo / "configs/research" / (basename + ".json")
    require(type(base_ref) is dict and set(base_ref) == {"path", "sha256"}
            and base_ref["path"] == str(base_path), "FIXED_BASE_PLAN")
    base = read_bound(base_path, base_ref["sha256"])
    require(base.get("schema_version") == "gx1_native_provider_hlc_all_tf_resumable_initial_base_plan_v1"
            and base.get("stage") == basename.lower() and base.get("output_root") == str(base_root), "BASE_SCHEMA")
    recipe = {'backbone_gradient_policy': None, 'batch_rows': 256, 'batches': 1385, 'drop_last': False, 'last_rows': 246, 'no_optimization': True, 'rows': 354550, 'seed': 20260911, 'shuffle': False}
    execution = {'GPU_core_stop_C': 75, 'GPU_memory_stop_C': 80, 'GPU_power_draw_stop_W': 250, 'GPU_power_limit_W': 250, 'class': 'trainer', 'cpu_affinity': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'cuda_peak_reserved_bytes_max': 12884901888, 'device': 'cuda:0', 'host_CPU_package_stop_W': 125, 'host_CPU_stop_C': 80, 'host_min_available_bytes': 8589934592, 'host_min_commit_headroom_bytes': 8589934592, 'internal_seconds': 3570, 'kill_after_seconds': 10, 'memory': '20G', 'numerical_threads': 8, 'outer_seconds': 3600, 'swap': '512M', 'tasks_max': 128, 'watchdog_interval_seconds': 1}
    for value in (base, segment):
        contract = value.get("model_contract")
        require(value.get("recipe") == recipe and value.get("execution") == execution
                and type(contract) is dict and "backbone_gradient_policy" not in contract
                and contract.get("contract_sha256") == "6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0",
                "NONE_WHOLE_TRAIN_CONTRACT")
    require(segment["model_contract"] == base["model_contract"], "SAME_BASE_MODEL")
    return {"base_plan": base_ref, "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "output_root": str(output_root)}


def native_provider_hlc_all_tf_initial_segment_source_bridge(previous_guard_owners, *,
        producer_commit="9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f"):
    """Prove exact scoring-route additions, then retain the earlier f33 byte proof."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_PROVIDER_HLC_INITIAL_SEGMENT_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f", "PRODUCER_COMMIT")
    parents = {"gx1/contracts/gx1_capped_execution_v1.py": "926fbdf60e1a4204b3f4c38a39440d519a3a62b6c2cc76e4b5d8b5bd2db8ca3f",
               "scripts/gx1_capped_run.sh": "0450ec5f0393807c618e7112cb4a67b0ecceba480649b465282543f4d9ca3b66"}
    f33_hashes = {"gx1/contracts/gx1_capped_execution_v1.py": "bbe6d6fe74ae289365759718bef1b045550c8250df08f937d87716b93a6d2c90",
                  "scripts/gx1_capped_run.sh": "ca334f3e843bd0d227883b783421c6bbc475a808b3b9a3d8c4ef6a0b622c5cc1"}
    require(type(previous_guard_owners) is dict and previous_guard_owners == parents, "EXACT_PARENT_GUARDS")
    repo = Path(__file__).resolve().parents[2]
    report = {}
    for name, digest in parents.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_PARENT_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            begin = b"# BEGIN GX1 ALL TF INITIAL SEGMENT GUARD ROUTE 20261124 003\n"
            end = b"# END GX1 ALL TF INITIAL SEGMENT GUARD ROUTE 20261124 003\n\n"
            require(current.count(begin) == current.count(end) == 1, "SINGLE_ADDITIVE_BLOCK")
            first, tail = current.split(begin); block, last = tail.split(end)
            nodes = ast.parse(block).body
            functions = {"require_native_provider_hlc_all_tf_initial_segment_execution",
                         "_native_provider_hlc_all_tf_initial_segment_bound_plan",
                         "native_provider_hlc_all_tf_initial_segment_source_bridge"}
            names = [node.name for node in nodes if isinstance(node, ast.FunctionDef)]
            constants = [node for node in nodes if isinstance(node, ast.Assign)]
            require(set(names) == functions and len(names) == len(functions) and len(nodes) == 4
                    and len(constants) == 1 and len(constants[0].targets) == 1
                    and isinstance(constants[0].targets[0], ast.Name)
                    and constants[0].targets[0].id == "_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_SEGMENT_GUARD_SHELL_ADDITIONS"
                    and ast.literal_eval(constants[0].value) == _NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_SEGMENT_GUARD_SHELL_ADDITIONS,
                    "EXACT_ADDITIVE_SYMBOLS")
            normalized = first + last
            old_begin = b"# BEGIN GX1 PROVIDER HLC ALL TF GUARD ROUTES 20261124 001\n"
            old_end = b"# END GX1 PROVIDER HLC ALL TF GUARD ROUTES 20261124 001\n\n"
            require(original.count(old_begin) == original.count(old_end) == 1, "PARENT_ALL_TF_BLOCK")
            first, tail = original.split(old_begin); _, last = tail.split(old_end)
            f33 = first + last
        else:
            text = current.decode()
            require(len(_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_SEGMENT_GUARD_SHELL_ADDITIONS) == 2, "EXACT_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_SEGMENT_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            normalized = text.encode()
            text = original.decode()
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "PARENT_ALL_TF_INSERTION")
                text = text.replace(composite, anchor, 1)
            f33 = text.encode()
        require(normalized == original, "ALL_PARENT_BYTES_UNCHANGED")
        reference = subprocess.check_output(["git", "show", "f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8:" + name], cwd=repo)
        require(f33 == reference and hashlib.sha256(f33).hexdigest() == f33_hashes[name], "FULL_F33_BYTES_PRESERVED")
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized).hexdigest(), "f33_sha256": f33_hashes[name]}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2,
            "old_guard_routes_preserved": True, "f33_guard_routes_preserved": True, "owners": report}
# END GX1 ALL TF INITIAL SEGMENT GUARD ROUTE 20261124 003

# BEGIN GX1 ALL TF LEARNING GUARD ROUTE 20261124 001
_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS = (
    ('# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 STATE\n', 'before', '# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 STATE\nNATIVE_PROVIDER_HLC_ALL_TF_LEARNING=false\nNATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256=\nNATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE=\n# END GX1 ALL TF LEARNING ROUTE 20261124 001 STATE\n'),
    ('validate_target_command() {\n', 'before', '# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 MATCHERS\nmatches_bound_native_provider_hlc_all_tf_learning() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan\n  case "$3:$4" in\n    "$NATIVE_PROVIDER_HLC_LEARNING_MODULE:--all-tf-learning-plan-sha256")\n      expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002.json" ;;\n    "$NATIVE_PROVIDER_HLC_LEARNING_MODULE:--all-tf-learning-resume-plan-sha256")\n      expected_plan="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002/RECOVERY_PLANS/$5.json" ;;\n    gx1.audit.native_provider_hlc_block_order_v1:--block-order-plan-sha256)\n      expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001.json" ;;\n    gx1.audit.native_provider_hlc_block_order_v1:--block-order-resume-plan-sha256)\n      expected_plan="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001/RECOVERY_PLANS/$5.json" ;;\n    gx1.audit.native_provider_hlc_row_membership_fit_v1:--row-membership-plan-sha256)\n      expected_plan="$REPO_ROOT/configs/research/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001.json" ;;\n    gx1.audit.native_provider_hlc_row_membership_fit_v1:--row-membership-resume-plan-sha256)\n      expected_plan="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001/RECOVERY_PLANS/$5.json" ;;\n    *) return 1 ;;\n  esac\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_provider_hlc_all_tf_learning() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: ALL-TF learning requires exact canonical trainer containment" >&2; exit 75\n  }\n  matches_bound_native_provider_hlc_all_tf_learning "$@" || {\n    echo "FATAL: ALL-TF learning requires its exact canonical command/plan" >&2; exit 75\n  }\n  local learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002"\n  local learning_mode=main learning_env_name\n  if [[ "$3" == gx1.audit.native_provider_hlc_block_order_v1 ]]; then\n    learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001"\n  fi\n  if [[ "$3" == gx1.audit.native_provider_hlc_row_membership_fit_v1 ]]; then\n    learning_output="/home/andre2/GX1_RUNS/NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001"\n  fi\n  if [[ "$4" == --all-tf-learning-resume-plan-sha256 || "$4" == --block-order-resume-plan-sha256 || "$4" == --row-membership-resume-plan-sha256 ]]; then\n    learning_output="$learning_output/RECOVERY_$5"\n    learning_mode=resume\n  fi\n  [[ ! -e "$learning_output" && ! -L "$learning_output" \\\n    && -d "${learning_output%/*}" && ! -L "${learning_output%/*}" \\\n    && "$(readlink -f "${learning_output%/*}")" == "${learning_output%/*}" ]] || {\n    echo "FATAL: ALL-TF learning requires a fresh output in its canonical parent" >&2; exit 75\n  }\n  for learning_env_name in ${!GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING@}; do\n    case "$learning_env_name" in\n      GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING|GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256|GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE) ;;\n      *) echo "FATAL: unknown ALL-TF learning environment field" >&2; exit 75 ;;\n    esac\n  done\n  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-false}" == false \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-false}" == false \\\n    && "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-false}" == false \\\n    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-false}" == false \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-false}" == false \\\n    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-false}" == false \\\n    && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-false}" == false \\\n    && "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" == "" \\\n    && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" == "" ]] || {\n    echo "FATAL: ALL-TF learning forbids mixed legacy modes/plans/lanes" >&2; exit 75\n  }\n  if [[ -n "${GX1_CAPPED_CLASS:-}${GX1_CAPPED_MEMORY_BYTES:-}${GX1_CAPPED_SWAP_BYTES:-}${GX1_CAPPED_TASKS_MAX:-}" ]]; then\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-}" == true \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "$5" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "$learning_mode" ]] || {\n      echo "FATAL: ALL-TF learning mode/plan differs from protected parent" >&2; exit 75\n    }\n  else\n    [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-false}" == false \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" \\\n      && -z "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" ]] || {\n      echo "FATAL: unprotected ALL-TF learning environment pollution" >&2; exit 75\n    }\n  fi\n  NATIVE_PROVIDER_HLC_ALL_TF_LEARNING=true\n  NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256="$5"\n  NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE="$learning_mode"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$learning_output"\n  TRAINER_MAX_WALL_SECONDS=12000\n  TRAINER_MODEL_MAX_WALL_SECONDS=11970\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 ALL TF LEARNING ROUTE 20261124 001 MATCHERS\n\n'),
    ('validate_target_command() {\n', 'after', '# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 DISPATCH\n  local all_tf_learning_arg all_tf_learning_reference=false\n  for all_tf_learning_arg in "$@"; do\n    case "$all_tf_learning_arg" in\n      --all-tf-learning-plan-sha256|--all-tf-learning-plan-sha256=*|--all-tf-learning-resume-plan-sha256|--all-tf-learning-resume-plan-sha256=*|--block-order-plan-sha256|--block-order-plan-sha256=*|--block-order-resume-plan-sha256|--block-order-resume-plan-sha256=*|gx1.audit.native_provider_hlc_block_order_v1|*/gx1/audit/native_provider_hlc_block_order_v1.py|gx1/audit/native_provider_hlc_block_order_v1.py|--row-membership-plan-sha256|--row-membership-plan-sha256=*|--row-membership-resume-plan-sha256|--row-membership-resume-plan-sha256=*|gx1.audit.native_provider_hlc_row_membership_fit_v1|*/gx1/audit/native_provider_hlc_row_membership_fit_v1.py|gx1/audit/native_provider_hlc_row_membership_fit_v1.py)\n        all_tf_learning_reference=true ;;\n    esac\n  done\n  if [[ "$all_tf_learning_reference" == true ]]; then\n    configure_native_provider_hlc_all_tf_learning "$@"\n    return\n  fi\n# END GX1 ALL TF LEARNING ROUTE 20261124 001 DISPATCH\n'),
    ('# BEGIN GX1 PROVIDER HLC LEARNING RESUME ROUTE 20261123 001 NESTED\n', 'before', '# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 NESTED\n  [[ "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-false}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256" \\\n    && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" ]] || {\n    echo "FATAL: nested ALL-TF learning mode/plan differs from protected parent" >&2; exit 75\n  }\n  if [[ "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" == true ]]; then\n    [[ "${GX1_CAPPED_MEMORY_BYTES:-}" == "21474836480" \\\n      && "${GX1_CAPPED_SWAP_BYTES:-}" == "536870912" \\\n      && "${GX1_CAPPED_TASKS_MAX:-}" == "128" \\\n      && "${OMP_NUM_THREADS:-}" == "8" \\\n      && "${MKL_NUM_THREADS:-}" == "8" \\\n      && "${OPENBLAS_NUM_THREADS:-}" == "8" \\\n      && "${NUMEXPR_NUM_THREADS:-}" == "8" \\\n      && "${VECLIB_MAXIMUM_THREADS:-}" == "8" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL:-}" == "false" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL:-}" == "false" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY:-}" == "false" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY:-}" == "false" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING:-}" == "false" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME:-}" == "false" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_TRAINING_LANE:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS:-}" == "false" \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_CUDA_LANE:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE:-}" == "" \\\n      && "${GX1_TRAINER_MAX_WALL_SECONDS:-}" == "12000" \\\n      && "${GX1_TRAINER_MODEL_MAX_WALL_SECONDS:-}" == "11970" \\\n      && "${GX1_TRAINER_ATTENDED_STAGE_REQUIRED:-}" == "false" \\\n      && "${GX1_TRAINER_GPU_INDEX:-}" == "0" \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_LIMIT_W:-}" == "250" \\\n      && "${GX1_TRAINER_GPU_MAX_POWER_DRAW_W:-}" == "250" \\\n      && "${GX1_TRAINER_GPU_MAX_CORE_TEMP_C:-}" == "75" \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C:-}" == "80" \\\n      && "${GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB:-}" == "12288" \\\n      && "${GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS:-}" == "1" \\\n      && "${GX1_TRAINER_CPU_MAX_TEMP_C:-}" == "80" \\\n      && "${GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W:-}" == "125" \\\n      && "${GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES:-}" == "8589934592" \\\n      && "${GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES:-}" == "8589934592" \\\n      && "${GX1_POWER_BENCHMARK_SCOPE_JSON:-}" == "" \\\n      && "${GX1_POWER_BENCHMARK_SCOPE_SHA256:-}" == "" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING:-}" == "true" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256" \\\n      && "${GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE:-}" == "$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" \\\n      && "${GX1_CAPPED_CLASS:-}" == "trainer" \\\n      && "${GX1_CUDA_PRODUCER_GUARD:-}" == "false" \\\n      && "${GX1_TRAINER_DEVICE:-}" == "cuda" \\\n      && "${GX1_TRAINER_EXECUTION_MODE:-}" == "canonical" \\\n      && "${BLIS_NUM_THREADS:-}" == "8" \\\n      && "${ARROW_NUM_THREADS:-}" == "1" \\\n      && "${POLARS_MAX_THREADS:-}" == "1" ]] || {\n      echo "FATAL: nested ALL-TF learning resources or mixed modes differ from protected parent" >&2; exit 75\n    }\n  fi\n# END GX1 ALL TF LEARNING ROUTE 20261124 001 NESTED\n'),
    ('  --setenv=GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME="$NATIVE_PROVIDER_HLC_LEARNING_RESUME" \\\n', 'before', '  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE" \\\n'),
)


def _native_provider_hlc_all_tf_learning_bound_plan(*, plan_sha256, mode):
    """Bind a fresh None-policy full-TRAIN fit/recovery before the kernel proof."""
    import hashlib
    import json
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_PROVIDER_HLC_ALL_TF_LEARNING_BINDING_" + reason + "]")
    require(type(plan_sha256) is str and re.fullmatch("[0-9a-f]{64}", plan_sha256) is not None,
            "PLAN_SHA256_INVALID")
    require(type(mode) is str and mode in ("main", "resume"), "MODE_INVALID")
    repo = Path(__file__).resolve().parents[2]
    require(str(repo) == "/home/andre2/src/GX1_CURRENT" and Path.cwd().resolve(strict=True) == repo,
            "CWD_INVALID")
    row_flag = "--row-membership-plan-sha256" if mode == "main" else "--row-membership-resume-plan-sha256"
    row_membership = sys.orig_argv[1:4] == ["-m", "gx1.audit.native_provider_hlc_row_membership_fit_v1", row_flag]
    block_flag = "--block-order-plan-sha256" if mode == "main" else "--block-order-resume-plan-sha256"
    block_order = sys.orig_argv[1:4] == ["-m", "gx1.audit.native_provider_hlc_block_order_v1", block_flag]
    basename = ("NATIVE_PROVIDER_HLC_ALL_TF_BLOCK_ORDER_20261124_001" if block_order
                else "NATIVE_PROVIDER_HLC_ALL_TF_ONE_EPOCH_20261124_002")
    if row_membership:
        basename = "NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_ONE_EPOCH_20261125_001"
    base = Path("/home/andre2/GX1_RUNS") / basename
    plan_path = (repo / "configs/research" / (basename + ".json") if mode == "main"
                 else base / "RECOVERY_PLANS" / (plan_sha256 + ".json"))
    output = base if mode == "main" else base / ("RECOVERY_" + plan_sha256)
    flag = "--all-tf-learning-plan-sha256" if mode == "main" else "--all-tf-learning-resume-plan-sha256"
    module = "gx1.audit.native_provider_hlc_learning_v1"
    if block_order:
        module, flag = "gx1.audit.native_provider_hlc_block_order_v1", block_flag
    if row_membership:
        module, flag = "gx1.audit.native_provider_hlc_row_membership_fit_v1", row_flag
    command = [str(repo / ".venv/bin/python"), "-m", module, flag, plan_sha256]
    require(sys.orig_argv in (command, command + ["--preflight"]) and sys.executable == command[0],
            "CLI_INVALID")
    try:
        require(stat.S_ISREG(plan_path.lstat().st_mode) and plan_path.resolve(strict=True) == plan_path,
                "CANONICAL_PLAN")
        require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
                and not output.parent.is_symlink() and output.parent.resolve(strict=True) == output.parent,
                "FRESH_ROOT")
        raw = plan_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == plan_sha256, "PLAN_SHA")
        plan = json.loads(raw)
    except (OSError, ValueError, UnicodeError) as error:
        raise RuntimeError("[GX1_PROVIDER_HLC_ALL_TF_LEARNING_BINDING_PLAN_OUTPUT_INVALID]") from error
    schemas = (("gx1_native_provider_hlc_all_tf_one_epoch_plan_v1",) if mode == "main" else
               ("gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v1",
                "gx1_native_provider_hlc_all_tf_remaining_epoch_plan_v2"))
    if block_order:
        schemas = (("gx1_native_provider_hlc_block_order_one_epoch_plan_v1",) if mode == "main" else
                   ("gx1_native_provider_hlc_block_order_remaining_epoch_plan_v1",))
    if row_membership:
        schemas = (("gx1_native_provider_hlc_row_membership_one_epoch_plan_v1",) if mode == "main" else
                   ("gx1_native_provider_hlc_row_membership_remaining_epoch_plan_v1",))
    stage = basename.lower() + ("_recovery" if mode == "resume" else "")
    require(type(plan) is dict and plan.get("schema_version") in schemas and plan.get("stage") == stage,
            "PLAN_IDENTITY")
    require(plan.get("output_root") == str(base) and (mode == "main" or plan.get("output_directory_prefix") == "RECOVERY_"), "PLAN_ROOT")
    recipe, contract = plan.get("recipe"), plan.get("model_contract")
    require(type(recipe) is dict and type(recipe.get("passes")) is int and recipe["passes"] == 1
            and "backbone_gradient_policy" not in recipe
            and "backbone_gradient_policy" in plan and plan["backbone_gradient_policy"] is None
            and type(contract) is dict and "backbone_gradient_policy" not in contract
            and contract.get("contract_sha256") == "6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0",
            "EXPLICIT_NONE_ONE_EPOCH_CONTRACT")
    require(plan.get("execution") == {'class': 'trainer', 'device': 'cuda:0', 'memory': '20G', 'swap': '512M', 'numerical_threads': 8, 'cpu_affinity': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'tasks_max': 128, 'outer_seconds': 12000, 'internal_seconds': 11970, 'kill_after_seconds': 10, 'cuda_peak_reserved_bytes_max': 12884901888, 'GPU_core_stop_C': 75, 'GPU_memory_stop_C': 80, 'GPU_power_limit_W': 250, 'GPU_draw_stop_W': 250, 'CPU_stop_C': 80, 'CPU_package_stop_W': 125, 'host_min_available_bytes': 8589934592, 'host_min_commit_headroom_bytes': 8589934592, 'watchdog_interval_seconds': 1}, "EXACT_EXECUTION")
    return plan_path, output


def require_native_provider_hlc_all_tf_learning_execution(*, plan_sha256: str, mode: str = "main") -> dict[str, Any]:
    """Prove fixed resources and isolation; policy/optimizer authority stays with the leaf."""
    plan_path, output = _native_provider_hlc_all_tf_learning_bound_plan(plan_sha256=plan_sha256, mode=mode)
    expected = {'GX1_CAPPED_MEMORY_BYTES': '21474836480',
 'GX1_CAPPED_SWAP_BYTES': '536870912',
 'GX1_CAPPED_TASKS_MAX': '128',
 'OMP_NUM_THREADS': '8',
 'MKL_NUM_THREADS': '8',
 'OPENBLAS_NUM_THREADS': '8',
 'NUMEXPR_NUM_THREADS': '8',
 'VECLIB_MAXIMUM_THREADS': '8',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL': 'false',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_INITIAL': 'false',
 'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_REPLAY': 'false',
 'GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE': '',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY': 'false',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE': '',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING': 'false',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME': 'false',
 'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE': 'false',
 'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE': 'false',
 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX': 'false',
 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'false',
 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false',
 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false',
 'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '',
 'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false',
 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false',
 'GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_CUDA_LANE': '',
 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256': '',
 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE': '',
 'GX1_TRAINER_MAX_WALL_SECONDS': '12000',
 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '11970',
 'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false',
 'GX1_TRAINER_GPU_INDEX': '0',
 'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': '250',
 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': '250',
 'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '75',
 'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80',
 'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288',
 'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1',
 'GX1_TRAINER_CPU_MAX_TEMP_C': '80',
 'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W': '125',
 'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES': '8589934592',
 'GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES': '8589934592',
 'GX1_POWER_BENCHMARK_SCOPE_JSON': '',
 'GX1_POWER_BENCHMARK_SCOPE_SHA256': '',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING': 'true',
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_PLAN_SHA256': plan_sha256,
 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_MODE': mode,
 'GX1_CAPPED_CLASS': 'trainer',
 'GX1_CUDA_PRODUCER_GUARD': 'false',
 'GX1_TRAINER_DEVICE': 'cuda',
 'GX1_TRAINER_EXECUTION_MODE': 'canonical',
 'BLIS_NUM_THREADS': '8',
 'ARROW_NUM_THREADS': '1',
 'POLARS_MAX_THREADS': '1'}
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"[GX1_PROVIDER_HLC_ALL_TF_LEARNING_ENV_INVALID] field={name}")
    if any(name.startswith("GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING") and name not in expected
           for name in os.environ):
        raise RuntimeError("[GX1_PROVIDER_HLC_ALL_TF_LEARNING_ENV_POLLUTION]")
    proof = require_guarded_cuda_trainer_execution()
    return {**proof, "class": "trainer", "execution_mode": "canonical",
            "kind": "provider_hlc_all_tf_learning" + ("_resume" if mode == "resume" else ""),
            "mode": mode, "training_authority_granted": False, "optimizer_authority_granted": False,
            "read_only": False, "resource_proof_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output),
            "outer_seconds": 12000, "internal_seconds": 11970, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}


def native_provider_hlc_all_tf_learning_source_bridge(previous_guard_owners, *,
        producer_commit="1d83a588c21af611973afdfda6a7c960fd060436"):
    """Strip only declared fit/recovery additions and prove every anchored guard byte."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_PROVIDER_HLC_ALL_TF_LEARNING_SOURCE_BRIDGE_" + reason + "]")
    require(producer_commit == "1d83a588c21af611973afdfda6a7c960fd060436", "PRODUCER_COMMIT")
    guards = {'gx1/contracts/gx1_capped_execution_v1.py': 'e4fa7ef20eb8fe29ef84e93016e7823540c2c719556aa173e5fc9336b67ff963', 'scripts/gx1_capped_run.sh': 'b23f7471ab0c62b883324319739652982394e7572df554c8ea1fc0c20f59bb45'}
    require(type(previous_guard_owners) is dict and previous_guard_owners == guards, "EXACT_ANCHOR_GUARDS")
    repo = Path(__file__).resolve().parents[2]
    report = {}
    for name, digest in guards.items():
        original = subprocess.check_output(["git", "show", producer_commit + ":" + name], cwd=repo)
        require(hashlib.sha256(original).hexdigest() == digest, "AUTHENTIC_ANCHOR_HASH")
        current = (repo / name).read_bytes()
        if name.endswith(".py"):
            begin = b"# BEGIN GX1 ALL TF LEARNING GUARD ROUTE 20261124 001\n"
            end = b"# END GX1 ALL TF LEARNING GUARD ROUTE 20261124 001\n\n"
            require(current.count(begin) == current.count(end) == 1, "SINGLE_ADDITIVE_BLOCK")
            first, tail = current.split(begin); block, last = tail.split(end)
            nodes = ast.parse(block).body
            functions = {"_native_provider_hlc_all_tf_learning_bound_plan",
                         "require_native_provider_hlc_all_tf_learning_execution",
                         "native_provider_hlc_all_tf_learning_source_bridge"}
            names = [node.name for node in nodes if isinstance(node, ast.FunctionDef)]
            constants = [node for node in nodes if isinstance(node, ast.Assign)]
            require(set(names) == functions and len(names) == len(functions) and len(nodes) == 4
                    and len(constants) == 1 and len(constants[0].targets) == 1
                    and isinstance(constants[0].targets[0], ast.Name)
                    and constants[0].targets[0].id == "_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS"
                    and ast.literal_eval(constants[0].value) == _NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS,
                    "EXACT_ADDITIVE_SYMBOLS")
            normalized = first + last
            require(ast.dump(ast.parse(normalized), include_attributes=False)
                    == ast.dump(ast.parse(original), include_attributes=False), "ALL_OLD_PYTHON_AST_UNCHANGED")
        else:
            text = current.decode()
            require(len(_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS) == 5, "EXACT_FRAGMENT_COUNT")
            for anchor, position, fragment in reversed(_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS):
                composite = fragment + anchor if position == "before" else anchor + fragment
                require(text.count(fragment) == text.count(composite) == 1, "EXACT_SHELL_INSERTION")
                text = text.replace(composite, anchor, 1)
            normalized = text.encode()
        require(normalized == original, "ALL_OLD_GUARD_BYTES_UNCHANGED")
        report[name] = {"original_sha256": digest, "current_sha256": hashlib.sha256(current).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized).hexdigest()}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2,
            "old_guard_routes_preserved": True, "owners": report}
# END GX1 ALL TF LEARNING GUARD ROUTE 20261124 001

# BEGIN GX1 ROW MEMBERSHIP FIT GUARD BRIDGE 20261125 001
_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_FIT_GUARD_ANCHOR = "8fd74a76cc45ac7306da4b905c82e9cd9b12976d"


def native_provider_hlc_row_membership_fit_guard_source_bridge(*, producer_commit="8fd74a76cc45ac7306da4b905c82e9cd9b12976d"):
    """Normalize only this exact third namespace to the measured closed parent."""
    import ast
    import hashlib
    import subprocess
    def require(ok, reason):
        if not ok:
            raise RuntimeError("[GX1_ROW_MEMBERSHIP_FIT_GUARD_BRIDGE_" + reason + "]")
    require(producer_commit == _NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_FIT_GUARD_ANCHOR,
            "EXACT_PRODUCER")
    repo = Path(__file__).resolve().parents[2]
    python_name, shell_name = "gx1/contracts/gx1_capped_execution_v1.py", "scripts/gx1_capped_run.sh"
    before_python = subprocess.check_output(["git", "show", producer_commit + ":" + python_name], cwd=repo).decode()
    before_shell = subprocess.check_output(["git", "show", producer_commit + ":" + shell_name], cwd=repo).decode()
    current_python = (repo / python_name).read_text(); current_shell = (repo / shell_name).read_text()
    def named(text, name, assignment=False):
        nodes = [n for n in ast.parse(text).body if
                 (assignment and isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)
                 or (not assignment and isinstance(n, ast.FunctionDef) and n.name == name)]
        require(len(nodes) == 1, "EXACT_SOURCE_SYMBOL:" + name)
        return ast.get_source_segment(text, nodes[0]), nodes[0]
    begin = "# BEGIN GX1 ROW MEMBERSHIP FIT GUARD BRIDGE 20261125 001\n"
    end = "# END GX1 ROW MEMBERSHIP FIT GUARD BRIDGE 20261125 001\n\n"
    require(current_python.count(begin) == current_python.count(end) == 1, "ONE_BRIDGE_BLOCK")
    first, tail = current_python.split(begin); body, last = tail.split(end)
    nodes = ast.parse(body).body
    require(len(nodes) == 2 and isinstance(nodes[0], ast.Assign) and len(nodes[0].targets) == 1
            and isinstance(nodes[0].targets[0], ast.Name)
            and nodes[0].targets[0].id == "_NATIVE_PROVIDER_HLC_ROW_MEMBERSHIP_FIT_GUARD_ANCHOR"
            and ast.literal_eval(nodes[0].value) == producer_commit
            and isinstance(nodes[1], ast.FunctionDef)
            and nodes[1].name == "native_provider_hlc_row_membership_fit_guard_source_bridge", "EXACT_BRIDGE_SYMBOLS")
    normalized_python = first + last
    constant = "_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS"
    for name, assignment in (("_native_provider_hlc_all_tf_learning_bound_plan", False), (constant, True)):
        old, _ = named(before_python, name, assignment); now, _ = named(normalized_python, name, assignment)
        require(normalized_python.count(now) == 1, "EXACT_RESTORE_SYMBOL")
        normalized_python = normalized_python.replace(now, old, 1)
    require(normalized_python == before_python, "ALL_OTHER_PYTHON_BYTES")
    _, old_node = named(before_python, constant, True)
    old_parts = ast.literal_eval(old_node.value); new_parts = _NATIVE_PROVIDER_HLC_ALL_TF_LEARNING_GUARD_SHELL_ADDITIONS
    require(len(old_parts) == len(new_parts) == 5 and all(old_parts[i] == new_parts[i] for i in (0, 3, 4)),
            "EXACT_REUSED_STATE_NESTED_EXPORTS")
    normalized_shell = current_shell
    for index in (1, 2):
        old_anchor, old_position, old_fragment = old_parts[index]
        anchor, position, fragment = new_parts[index]
        require((anchor, position) == (old_anchor, old_position)
                and normalized_shell.count(fragment) == 1, "EXACT_SHELL_FRAGMENT")
        normalized_shell = normalized_shell.replace(fragment, old_fragment, 1)
    require(normalized_shell == before_shell, "ALL_OTHER_SHELL_BYTES")
    owners = {}
    for name, before, current, normalized in (
            (python_name, before_python, current_python, normalized_python),
            (shell_name, before_shell, current_shell, normalized_shell)):
        owners[name] = {"original_sha256": hashlib.sha256(before.encode()).hexdigest(),
                        "current_sha256": hashlib.sha256(current.encode()).hexdigest(),
                        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest()}
    return {"producer_commit": producer_commit, "checked_guard_owners": 2,
            "old_guard_routes_preserved": True, "old_numeric_resource_environment_function_unchanged": True,
            "owners": owners}
# END GX1 ROW MEMBERSHIP FIT GUARD BRIDGE 20261125 001

# BEGIN GX1 ALL TF V3 GUARDS 20261125 001
_ALL_TF_V3_GUARD_SHELL_ADDITIONS = (('# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 MATCHERS\n', 'before', '# BEGIN GX1 ALL TF V3 ROUTES 20261125 001 MATCHERS\nmatches_bound_native_all_tf_provider_hlc_v3_initial_segment() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" \\\n    && "$4" == --all-tf-v3-score-segment-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local base_root=/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_WHOLE_TRAIN_INITIAL_20261125_001\n  local plan_dir="$base_root/PLANS"\n  local expected_plan="$plan_dir/$5.json"\n  [[ -d "$base_root" && ! -L "$base_root" && "$(readlink -f "$base_root")" == "$base_root" \\\n    && -d "$plan_dir" && ! -L "$plan_dir" && "$(readlink -f "$plan_dir")" == "$plan_dir" \\\n    && -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_all_tf_provider_hlc_v3_initial_segment() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC INITIAL requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_all_tf_provider_hlc_v3_initial_segment "$@" || {\n    echo "FATAL: provider HLC INITIAL CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local initial_output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_WHOLE_TRAIN_INITIAL_20261125_001/SEGMENT_$5"\n  [[ ! -e "$initial_output" && ! -L "$initial_output" \\\n    && -d "${initial_output%/*}" && ! -L "${initial_output%/*}" \\\n    && "$(readlink -f "${initial_output%/*}")" == "${initial_output%/*}" ]] || {\n    echo "FATAL: provider HLC INITIAL output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL=true\n  NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$initial_output"\n  TRAINER_MAX_WALL_SECONDS=3600\n  TRAINER_MODEL_MAX_WALL_SECONDS=3570\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n\nmatches_bound_native_all_tf_provider_hlc_v3_replay() {\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001.json"\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" ]] || return 1\n  [[ "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" \\\n    && "$4" == --all-tf-v3-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane && ( "$7" == warm || "$7" == cold ) ]] || return 1\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" ]] || return 1\n  [[ "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\n\nconfigure_native_all_tf_provider_hlc_v3_replay() {\n  [[ "$JOB_CLASS" == trainer && "$ATTENDED_SMOKE" == false \\\n    && "$CUDA_PRODUCER_GUARD" == false && "$MEM" == 20G && "$SWAP" == 512M \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo "FATAL: provider HLC replay requires exact canonical trainer containment" >&2\n    exit 75\n  }\n  matches_bound_native_all_tf_provider_hlc_v3_replay "$@" || {\n    echo "FATAL: provider HLC replay CLI/plan is not exactly bound" >&2\n    exit 75\n  }\n  local replay_output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001_$7"\n  [[ ! -e "$replay_output" && ! -L "$replay_output" \\\n    && -d "${replay_output%/*}" && ! -L "${replay_output%/*}" \\\n    && "$(readlink -f "${replay_output%/*}")" == "${replay_output%/*}" ]] || {\n    echo "FATAL: provider HLC replay output is not fresh canonical scope" >&2\n    exit 75\n  }\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY=true\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256="$5"\n  NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE="$7"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$replay_output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 ALL TF V3 ROUTES 20261125 001 MATCHERS\n'), ('# BEGIN GX1 PROVIDER HLC ALL TF ROUTES 20261124 001 DISPATCH\n', 'before', '# BEGIN GX1 ALL TF V3 ROUTES 20261125 001 DISPATCH\n  if [[ "${4:-}" == --all-tf-v3-score-segment-plan-sha256 ]]; then\n    [[ "${3:-}" == "$NATIVE_PROVIDER_HLC_INITIAL_MODULE" ]] || { echo "FATAL: V3 INITIAL requires exact module" >&2; exit 75; }\n    configure_native_all_tf_provider_hlc_v3_initial_segment "$@"; return\n  fi\n  if [[ "${4:-}" == --all-tf-v3-plan-sha256 ]]; then\n    [[ "${3:-}" == "$NATIVE_PROVIDER_HLC_REPLAY_MODULE" ]] || { echo "FATAL: V3 replay requires exact module" >&2; exit 75; }\n    configure_native_all_tf_provider_hlc_v3_replay "$@"; return\n  fi\n# END GX1 ALL TF V3 ROUTES 20261125 001 DISPATCH\n'))

def _require_provider_hlc_all_tf_initial_environment(plan_sha256):
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
    return proof


def _require_provider_hlc_all_tf_replay_environment(plan_sha256, lane):
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
    return proof


def require_native_all_tf_provider_hlc_v3_initial_segment_execution(*, plan_sha256: str) -> dict[str, Any]:
    """Read-only full TRAIN containment; no optimizer or training authority."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256_INVALID]")
    bound = _native_all_tf_provider_hlc_v3_bound_plan(plan_sha256=plan_sha256)
    proof = _require_provider_hlc_all_tf_initial_environment(plan_sha256)
    plan_path = Path(bound["plan"]["path"])
    output_root = Path(bound["output_root"])
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_all_tf_v3_initial_segment",
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": True,
            "base_plan": bound["base_plan"], "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 3600, "internal_seconds": 3570, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}

def require_native_all_tf_provider_hlc_v3_replay_execution(*, plan_sha256: str, lane: str) -> dict[str, Any]:
    """Contain only the exact technical replay; its leaf owns optimizer admission."""
    if type(plan_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None:
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256_INVALID]")
    if type(lane) is not str or lane not in ("warm", "cold"):
        raise RuntimeError("[GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE_INVALID]")
    _native_all_tf_provider_hlc_v3_bound_plan(plan_sha256=plan_sha256, lane=lane)
    proof = _require_provider_hlc_all_tf_replay_environment(plan_sha256, lane)
    basename = "NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001"
    plan_path = Path(__file__).resolve().parents[2] / "configs/research" / (basename + ".json")
    output_root = Path("/home/andre2/GX1_RUNS") / (basename + "_" + lane)
    return {**proof, "class": "trainer", "execution_mode": "canonical", "kind": "provider_hlc_all_tf_v3_replay", "lane": lane,
            "training_authority_granted": False, "optimizer_authority_granted": False, "read_only": False, "resource_proof_only": True,
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "plan_path": str(plan_path), "output_root": str(output_root),
            "outer_seconds": 1800, "internal_seconds": 1770, "kill_after_seconds": 10,
            "cpu_affinity": list(range(19)), "numerical_threads": 8}

_ALL_TF_V3_GUARD_PARENT = {"path": "/home/andre2/GX1_RUNS/GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001/ALL_TF_V3_GUARD_PREDECESSOR_20261125_001.json", "sha256": "b13c73b5e2f467f4a219ae8c1137394c9c69959ce4c1c0122b5cd0fbe18dfa21"}

def _native_all_tf_provider_hlc_v3_bound_plan(*, plan_sha256, lane=None):
    import hashlib,json
    from gx1.contracts.native_future_state_provider_hlc_v1 import all_tf_provider_hlc_state_contract
    def require(ok,why):
        if not ok:raise RuntimeError("[GX1_ALL_TF_V3_BINDING_"+why+"]")
    require(type(plan_sha256)is str and re.fullmatch('[0-9a-f]{64}',plan_sha256)is not None,'SHA')
    require(lane is None or type(lane)is str and lane in('warm','cold'),'LANE')
    repo=Path(__file__).resolve().parents[2];initial=lane is None
    flag='--all-tf-v3-score-segment-plan-sha256'if initial else'--all-tf-v3-plan-sha256'
    args=[flag,plan_sha256]+([]if initial else['--lane',lane])
    if sys.argv[-1:]==['--preflight']:args+=['--preflight']
    module='native_future_state_provider_hlc_initial_v1.py'if initial else'native_provider_hlc_replay_v1.py'
    require(Path.cwd().resolve()==repo and Path(sys.argv[0]).resolve()==repo/'gx1/audit'/module and Path(sys.executable).absolute()==repo/'.venv/bin/python'and sys.argv[1:]==args,'EXACT_ARGV')
    def read(path,digest):
        require(path.is_file()and not path.is_symlink()and path.resolve()==path,'CANONICAL_PLAN')
        raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,'PLAN_SHA');value=json.loads(raw);require(type(value)is dict,'PLAN_OBJECT');return value
    name='NATIVE_ALL_TF_PROVIDER_HLC_V3_WHOLE_TRAIN_INITIAL_20261125_001'if initial else'NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001'
    root=Path('/home/andre2/GX1_RUNS')/name
    if initial:
        require(root.is_dir()and not root.is_symlink()and root.resolve()==root and(root/'PLANS').is_dir()and not(root/'PLANS').is_symlink(),'CANONICAL_BASE_ROOT')
        path=root/'PLANS'/(plan_sha256+'.json');plan=read(path,plan_sha256)
        require(plan.get('schema_version')=='gx1_native_all_tf_provider_hlc_v3_initial_segment_plan_v1'and plan.get('stage')==name.lower()and plan.get('base_root')==str(root)and'plan_path'not in plan and'output_root'not in plan,'SEGMENT_SCHEMA')
        base_ref=plan.get('base_plan');base_path=repo/'configs/research'/(name+'.json')
        require(type(base_ref)is dict and set(base_ref)=={'path','sha256'}and base_ref['path']==str(base_path),'BASE_REF')
        base=read(base_path,base_ref['sha256']);require(base.get('schema_version')=='gx1_native_all_tf_provider_hlc_v3_resumable_initial_base_plan_v1'and base.get('stage')==name.lower()and base.get('output_root')==str(root),'BASE_SCHEMA')
        recipe={'backbone_gradient_policy':None,'batch_rows':256,'batches':1385,'drop_last':False,'last_rows':246,'no_optimization':True,'rows':354550,'seed':20260911,'shuffle':False}
        require(plan.get('recipe')==base.get('recipe')==recipe and plan.get('execution')==base.get('execution')and plan.get('model_contract')==base.get('model_contract'),'SAME_BASE_RECIPE_EXECUTION_MODEL')
        output=root/('SEGMENT_'+plan_sha256)
    else:
        path=repo/'configs/research'/(name+'.json');plan=read(path,plan_sha256)
        require(plan.get('schema_version')=='gx1_native_all_tf_provider_hlc_v3_cold_replay_plan_v1'and plan.get('stage')==name.lower(),'REPLAY_SCHEMA')
        output=Path('/home/andre2/GX1_RUNS')/(name+'_'+lane);require(plan.get('output_roots',{}).get(lane)==str(output),'REPLAY_OUTPUT')
        base_ref=None
    contract=plan.get('model_contract');target=all_tf_provider_hlc_state_contract()
    require(type(contract)is dict and'backbone_gradient_policy'not in contract and contract.get('target_contract_sha256')==target['contract_sha256']and contract.get('target_metadata_as_model_input')is False,'EXACT_V3_MODEL_TARGET')
    require(not output.exists()and not output.is_symlink(),'FRESH_OUTPUT')
    return {'plan':{'path':str(path),'sha256':plan_sha256},'output_root':str(output),'base_plan':base_ref}

def _native_all_tf_provider_hlc_v3_source_bridge(previous_guard_owners,*,producer_commit):
    import ast,hashlib,json,subprocess
    def require(ok,why):
        if not ok:raise RuntimeError('[GX1_ALL_TF_V3_SOURCE_BRIDGE_'+why+']')
    require(producer_commit=='8fd74a76cc45ac7306da4b905c82e9cd9b12976d','PRODUCER')
    repo=Path(__file__).resolve().parents[2]
    def read(ref):
        path=Path(ref['path']);require(path.is_file()and not path.is_symlink()and path.resolve()==path,'CANONICAL_EVIDENCE');raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'EVIDENCE_SHA');return raw
    parent=json.loads(read(_ALL_TF_V3_GUARD_PARENT));proof=parent['existing_qualified_bridge'];names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    original={n:hashlib.sha256(subprocess.check_output(['git','show',producer_commit+':'+n],cwd=repo)).hexdigest()for n in names}
    require(previous_guard_owners==original and parent['status']=='PASS_AUTHENTICATED_8FD_ROWMEMBERSHIP_GUARD_PARENT'and proof['producer_commit']==producer_commit and proof['old_guard_routes_preserved']is True,'PARENT_BRIDGE')
    before={n:read(parent['parent_guard_snapshots'][n]).decode()for n in names};current={n:(repo/n).read_text()for n in names}
    for n in names:require(proof['owners'][n]['original_sha256']==original[n]and proof['owners'][n]['current_sha256']==hashlib.sha256(before[n].encode()).hexdigest(),'PARENT_OWNER_SHA')
    begin='# BEGIN GX1 ALL TF V3 GUARDS 20261125 001\n';end='# END GX1 ALL TF V3 GUARDS 20261125 001\n\n'
    text=current[names[0]];require(text.count(begin)==text.count(end)==1,'ONE_EXTENSION_BLOCK');first,tail=text.split(begin);body,last=tail.split(end);normalized=first+last
    try:nodes=ast.parse(body).body
    except SyntaxError as error:raise RuntimeError('[GX1_ALL_TF_V3_SOURCE_BRIDGE_EXTENSION_AST_INVALID]')from error
    functions=[n for n in nodes if isinstance(n,ast.FunctionDef)];assignments=[n for n in nodes if isinstance(n,ast.Assign)]
    require(len(nodes)==10 and len(functions)==8 and len(assignments)==2 and len({n.name for n in functions})==8 and all(len(n.targets)==1 and isinstance(n.targets[0],ast.Name)for n in assignments)and {n.targets[0].id for n in assignments}=={'_ALL_TF_V3_GUARD_PARENT','_ALL_TF_V3_GUARD_SHELL_ADDITIONS'},'EXACT_EXTENSION_AST')
    try:literals={n.targets[0].id:ast.literal_eval(n.value)for n in assignments}
    except (ValueError,TypeError,SyntaxError)as error:raise RuntimeError('[GX1_ALL_TF_V3_SOURCE_BRIDGE_EXTENSION_LITERALS_INVALID]')from error
    require(literals=={'_ALL_TF_V3_GUARD_PARENT':_ALL_TF_V3_GUARD_PARENT,'_ALL_TF_V3_GUARD_SHELL_ADDITIONS':_ALL_TF_V3_GUARD_SHELL_ADDITIONS},'EXACT_EXTENSION_LITERALS')
    symbols=[n.name for n in functions]
    require(set(symbols)=={'_require_provider_hlc_all_tf_initial_environment','_require_provider_hlc_all_tf_replay_environment','require_native_all_tf_provider_hlc_v3_initial_segment_execution','require_native_all_tf_provider_hlc_v3_replay_execution','_native_all_tf_provider_hlc_v3_bound_plan','_native_all_tf_provider_hlc_v3_source_bridge','native_all_tf_provider_hlc_v3_initial_source_bridge','native_all_tf_provider_hlc_v3_replay_source_bridge'},'EXACT_EXTENSION_SYMBOLS')
    def source(text,name):
        nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name];require(len(nodes)==1,'ONE_FUNCTION');return ast.get_source_segment(text,nodes[0])
    for name,helper,args,newname in(
        ('require_native_provider_hlc_all_tf_initial_segment_execution','_require_provider_hlc_all_tf_initial_environment','plan_sha256','require_native_all_tf_provider_hlc_v3_initial_segment_execution'),
        ('require_native_provider_hlc_all_tf_replay_execution','_require_provider_hlc_all_tf_replay_environment','plan_sha256, lane','require_native_all_tf_provider_hlc_v3_replay_execution')):
        now=source(normalized,name);old=source(before[names[0]],name);start=old.index('    expected = {');stop=old.index('    proof = require_guarded_cuda_trainer_execution()\n',start)+len('    proof = require_guarded_cuda_trainer_execution()\n')
        expected_helper='def '+helper+'('+args+'):\n'+old[start:stop]+'    return proof'
        require(source(body,helper)==expected_helper,'EXACT_SHARED_ENV_HELPER')
        expected_old=old[:start]+'    proof = '+helper+'('+args+')\n'+old[stop:]
        require(now==expected_old and normalized.count(now)==1,'EXACT_OLD_REQUIRE_EXTRACTION')
        expected_new=expected_old.replace('def '+name+'(','def '+newname+'(',1).replace('_native_provider_hlc_all_tf_initial_segment_bound_plan(', '_native_all_tf_provider_hlc_v3_bound_plan(').replace('_native_provider_hlc_all_tf_bound_plan(', '_native_all_tf_provider_hlc_v3_bound_plan(')
        expected_new=expected_new.replace('"NATIVE_PROVIDER_HLC_ALL_TF_COLD_REPLAY_20261124_001"','"NATIVE_ALL_TF_PROVIDER_HLC_V3_COLD_REPLAY_20261125_001"').replace('"kind": "provider_hlc_all_tf_initial_segment"','"kind": "provider_hlc_all_tf_v3_initial_segment"').replace('"kind": "provider_hlc_all_tf_replay"','"kind": "provider_hlc_all_tf_v3_replay"')
        require(source(body,newname)==expected_new,'EXACT_NEW_REQUIRE_RESOURCE_REUSE');normalized=normalized.replace(now,old,1)
    require(normalized==before[names[0]],'ALL_OTHER_PYTHON_BYTES')
    shell=current[names[1]]
    for anchor,position,fragment in _ALL_TF_V3_GUARD_SHELL_ADDITIONS:
        require(position=='before'and shell.count(anchor)==1 and shell.count(fragment)==1 and shell.count(fragment+anchor)==1,'ONE_SHELL_FRAGMENT');shell=shell.replace(fragment+anchor,anchor,1)
    require(shell==before[names[1]],'ALL_OTHER_SHELL_BYTES')
    return {'producer_commit':producer_commit,'predecessor':dict(_ALL_TF_V3_GUARD_PARENT),'checked_guard_owners':2,'old_guard_routes_preserved':True,'old_numeric_resource_environment_function_unchanged':True,'owners':{n:{'original_sha256':original[n],'pre_V3_sha256':hashlib.sha256(before[n].encode()).hexdigest(),'current_sha256':hashlib.sha256(current[n].encode()).hexdigest()}for n in names}}

def native_all_tf_provider_hlc_v3_initial_source_bridge(previous_guard_owners,*,producer_commit='8fd74a76cc45ac7306da4b905c82e9cd9b12976d'):
    return _native_all_tf_provider_hlc_v3_source_bridge(previous_guard_owners,producer_commit=producer_commit)

def native_all_tf_provider_hlc_v3_replay_source_bridge(previous_guard_owners,*,producer_commit='8fd74a76cc45ac7306da4b905c82e9cd9b12976d'):
    return _native_all_tf_provider_hlc_v3_source_bridge(previous_guard_owners,producer_commit=producer_commit)

# END GX1 ALL TF V3 GUARDS 20261125 001

# BEGIN GX1 V3 DEVELOPMENT GUARDS 20261127 001
_V3_DEVELOPMENT_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001'
_V3_DEVELOPMENT_PARENT = '3ef40c291594a5bcc6d923ec5dcc4fd4e18ab746'
_V3_DEVELOPMENT_SHELL_ADDITIONS = (('NATIVE_PROVIDER_HLC_INITIAL=false\n', 'before', 'NATIVE_V3_DEVELOPMENT=false\nNATIVE_V3_DEVELOPMENT_PLAN_SHA256=\nNATIVE_V3_DEVELOPMENT_PHASE=\n'), ('# BEGIN GX1 ALL TF V3 ROUTES 20261125 001 MATCHERS\n', 'before', '# BEGIN GX1 V3 DEVELOPMENT ROUTE 20261127 001\nmatches_bound_native_v3_development() {\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \\\n    && "$4" == --v3-development-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --phase ]] || return 1\n  case "$7" in cohort|warm|cold|fit) ;; *) return 1 ;; esac\n  local plan_suffix="${7^^}"\n  if [[ "$7" == warm ]]; then plan_suffix=WARM_002; fi\n  if [[ "$7" == fit ]]; then plan_suffix=FIT_004; fi\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001_${plan_suffix}.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_development() {\n  matches_bound_native_v3_development "$@" || { echo \'FATAL: exact V3 development CLI/plan required\' >&2; exit 75; }\n  [[ "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && "$SWAP" == 512M && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 development forbids alternate resource modes\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001_$7"\n  if [[ "$7" == fit ]]; then output="${output}_004"; fi\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 development output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_DEVELOPMENT=true\n  NATIVE_V3_DEVELOPMENT_PLAN_SHA256="$5"\n  NATIVE_V3_DEVELOPMENT_PHASE="$7"\n  if [[ "$7" == cohort ]]; then\n    [[ "$JOB_CLASS" == producer && "$MEM" == 12G ]] || { echo \'FATAL: V3 cohort requires CPU producer12G\' >&2; exit 75; }\n    return\n  fi\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G ]] || { echo \'FATAL: V3 NN requires canonical trainer20G\' >&2; exit 75; }\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  if [[ "$7" == fit ]]; then TRAINER_MAX_WALL_SECONDS=5400; TRAINER_MODEL_MAX_WALL_SECONDS=5370; fi\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 DEVELOPMENT ROUTE 20261127 001\n\n'), ('# BEGIN GX1 ALL TF LEARNING ROUTE 20261124 001 DISPATCH\n', 'before', '# BEGIN GX1 V3 DEVELOPMENT DISPATCH 20261127 001\n  local development_arg\n  for development_arg in "$@"; do\n    case "$development_arg" in\n      --v3-development-plan-sha256|--v3-development-plan-sha256=*) configure_native_v3_development "$@"; return ;;\n    esac\n  done\n# END GX1 V3 DEVELOPMENT DISPATCH 20261127 001\n'), ('  --setenv=GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING="$NATIVE_PROVIDER_HLC_ALL_TF_LEARNING" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_DEVELOPMENT="$NATIVE_V3_DEVELOPMENT" \\\n  --setenv=GX1_NATIVE_V3_DEVELOPMENT_PLAN_SHA256="$NATIVE_V3_DEVELOPMENT_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_V3_DEVELOPMENT_PHASE="$NATIVE_V3_DEVELOPMENT_PHASE" \\\n'))

def _native_v3_development_execution(phase):
    _native_v3_development_require(phase in ('cohort','warm','cold','fit'),'PHASE')
    if phase=='cohort':
        return {'class':'producer','device':'cpu','memory':'12G','swap':'512M','numerical_threads':1,
            'cpu_affinity':list(range(8)),'tasks_max':64,'outer_seconds':600,'internal_seconds':570}
    outer,inner=(5400,5370)if phase=='fit'else(1800,1770)
    return {'class':'trainer','device':'cuda:0','memory':'20G','swap':'512M','numerical_threads':8,
        'cpu_affinity':list(range(19)),'tasks_max':128,'outer_seconds':outer,'internal_seconds':inner,
        'kill_after_seconds':10,'cuda_peak_reserved_bytes_max':12884901888,'GPU_core_stop_C':75,
        'GPU_memory_stop_C':80,'GPU_power_limit_W':250,'GPU_draw_stop_W':250,'CPU_stop_C':80,
        'CPU_package_stop_W':125,'host_min_available_bytes':8589934592,
        'host_min_commit_headroom_bytes':8589934592,'watchdog_interval_seconds':1}

def _native_v3_development_require(ok,why):
    if not bool(ok):raise RuntimeError('[GX1_V3_DEVELOPMENT_'+why+']')

def _native_v3_development_read(path,digest=None):
    import hashlib,json
    require=_native_v3_development_require;path=Path(path)
    require(path.is_file()and not path.is_symlink()and path.resolve()==path,'CANONICAL_JSON')
    raw=path.read_bytes();require(digest is None or hashlib.sha256(raw).hexdigest()==digest,'JSON_SHA')
    value=json.loads(raw);require(type(value)is dict,'JSON_OBJECT');return value

def _native_v3_development_bound_plan(plan_sha256,phase):
    import hashlib,subprocess
    require=_native_v3_development_require
    require(type(plan_sha256)is str and re.fullmatch('[0-9a-f]{64}',plan_sha256)is not None,'PLAN_SHA')
    require(type(phase)is str and phase in ('cohort','warm','cold','fit'),'PHASE')
    repo=Path(__file__).resolve().parents[2];name=_V3_DEVELOPMENT_NAME
    args=['--v3-development-plan-sha256',plan_sha256,'--phase',phase]
    preflight=sys.argv[-1:]==['--preflight']
    if preflight:args+=['--preflight']
    require(Path.cwd().resolve()==repo and Path(sys.argv[0]).resolve()==repo/'gx1/audit/native_provider_hlc_learning_v1.py'
        and Path(sys.executable).absolute()==repo/'.venv/bin/python'and sys.argv[1:]==args,'EXACT_ARGV')
    path=repo/'configs/research'/(name+'_'+phase.upper()+('_002'if phase=='warm'else'_004'if phase=='fit'else'')+'.json');plan=_native_v3_development_read(path,plan_sha256)
    output=Path('/home/andre2/GX1_RUNS')/(name+'_'+phase+('_004'if phase=='fit'else''))
    require(plan.get('schema_version')=='gx1_native_all_tf_provider_hlc_v3_development_plan_v1'
        and plan.get('phase')==phase and plan.get('stage')==(name+'_'+phase).lower()
        and plan.get('output_root')==str(output),'PLAN_IDENTITY')
    require(not output.exists()and not output.is_symlink()and output.parent.is_dir()
        and not output.parent.is_symlink()and output.parent.resolve()==output.parent,'FRESH_OUTPUT')
    require(plan.get('source_commit')==subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'SOURCE_COMMIT')
    require(plan.get('execution')==_native_v3_development_execution(phase),'EXACT_EXECUTION')
    protocol_ref=plan.get('protocol');protocol_path=repo/'configs/research'/(name.replace('_20261127_001','_PROTOCOL_20261127_001')+'.json')
    require(type(protocol_ref)is dict and set(protocol_ref)=={'path','sha256'}and protocol_ref['path']==str(protocol_path)
        and type(protocol_ref['sha256'])is str and re.fullmatch('[0-9a-f]{64}',protocol_ref['sha256'])is not None,'PROTOCOL_REF')
    protocol=_native_v3_development_read(protocol_path,protocol_ref['sha256']);recipe=plan.get('recipe')
    require(protocol.get('schema_version')=='gx1_native_all_tf_provider_hlc_v3_development_protocol_v1'
        and protocol.get('experiment_id')==name and recipe==protocol.get('recipe')and type(recipe)is dict
        and recipe.get('passes')==2 and recipe.get('seed')==20260911 and recipe.get('train_batch_rows')==96
        and recipe.get('eval_batch_rows')==256 and recipe.get('gradient_clip_norm')==1.,'BOUNDED_RECIPE')
    selection=protocol.get('selection',{})
    require(selection.get('rows')==20727 and selection.get('whole_TRAIN_rows')==354550
        and selection.get('selected_issue_ns_sha256')=='ca9868f542d290045c2640c320d1e3517329c8a4223c8ec3a58a153d86a283f1'
        and selection.get('selected_source_rows_sha256')=='84fca333ffd882d679e6339ea9663811b9d16564cb8f8617072753f4f6db5025'
        and selection.get('target_filter')is False and selection.get('top_up')is False and selection.get('quotas')is False,'CLOCK_ONLY_COHORT')
    limits=plan.get('limits');require(type(limits)is dict and type(limits.get('optimizer_steps'))is int
        and limits['optimizer_steps']=={'cohort':0,'warm':3,'cold':1,'fit':432}[phase],'BOUNDED_UPDATES')
    contract=plan.get('model_contract',{});target=plan.get('metadata',{}).get('target_contract',{})
    require(contract.get('contract_sha256')=='ded7f2a05417d108ef8bed4d88f31d40ea1de21292e964586598ddaa818010c3'
        and 'backbone_gradient_policy'not in contract and contract.get('target_metadata_as_model_input')is False
        and contract.get('target_contract_sha256')==target.get('contract_sha256')=='58fd7f15bdf35646680829a718caa2991f2029ed99f48030e3a85c86aae380c5'
        and target.get('schema_version')=='gx1_native_m5_fully_future_state_dataset_v3_all_tf_provider_hlc'
        and plan.get('backbone_gradient_policy')is None,'EXACT_V3_FUNCTION')
    owners=plan.get('source_owners');require(type(owners)is dict and {'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh','gx1/audit/native_provider_hlc_learning_v1.py'}<=set(owners),'SOURCE_INVENTORY')
    for name,digest in owners.items():
        p=repo/name
        require(type(name)is str and not Path(name).is_absolute()and '..'not in Path(name).parts
            and p.is_file()and not p.is_symlink()and p.resolve()==p and hashlib.sha256(p.read_bytes()).hexdigest()==digest,'SOURCE_SHA')
    bridge=plan.get('guard_source_bridge',{});old={n:v['original_sha256']for n,v in bridge.get('owners',{}).items()}
    require(bridge==native_provider_hlc_v3_development_source_bridge(old,producer_commit=bridge.get('producer_commit')),'SOURCE_BRIDGE')
    policy=_native_v3_development_read(repo/'NEXT_RUN_POLICY.json');parent=policy.get('native_v3_bounded_development_test_20261127_001',{})
    require(parent.get('protocol')==protocol_ref and parent.get('main_fit_authorized')is False
        and parent.get('test_authorized')is False and parent.get('spending_authorized')is False,'PARENT_SCOPE')
    flags=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')
    require(all(policy.get(k)is (not preflight and phase!='cohort'and k in flags[:2])for k in flags)
        and policy.get('exit_training_and_evaluation_this_wave')is False,'EXACT_FLAGS')
    phases=parent.get('phase_gates',{});require(type(phases)is dict and set(phases)<=set(('cohort','warm','cold','fit')),'PHASE_GATES')
    if preflight:
        require(parent.get('execution_authorized')is False and parent.get('active_phase')in(None,'')
            and all(v.get('execution_authorized')is False and v.get('remaining_invocations')==0 for v in phases.values()),'PREFLIGHT_CLOSED')
    else:
        status='ADMITTED_EXACT_V3_DEVELOPMENT_'+phase.upper()
        gate={'status':status,'plan':{'path':str(path),'sha256':plan_sha256},'execution_authorized':True,
            'remaining_invocations':1,'relaunch_allowed':False,'training_authorized':phase!='cohort','test_authorized':False}
        require(parent.get('status')==status and parent.get('execution_authorized')is True
            and parent.get('active_phase')==phase and phases.get(phase)==gate
            and all(v.get('execution_authorized')is False and v.get('remaining_invocations')==0 for k,v in phases.items()if k!=phase),'ONE_USE_GATE')
    return {'plan':{'path':str(path),'sha256':plan_sha256},'output_root':str(output),'preflight':preflight,'execution':plan['execution']}

def _native_v3_development_environment(plan_sha256,phase,output):
    require=_native_v3_development_require;prefix='GX1_NATIVE_V3_DEVELOPMENT'
    expected={prefix:'true',prefix+'_PLAN_SHA256':plan_sha256,prefix+'_PHASE':phase,
        'GX1_POWER_BENCHMARK_SCOPE_JSON':'','GX1_POWER_BENCHMARK_SCOPE_SHA256':''}
    require(all(not k.startswith(prefix)or k in expected for k in os.environ),'UNKNOWN_DEVELOPMENT_ENV')
    require(all(v in('false','')for k,v in os.environ.items()if k.startswith('GX1_NATIVE_')and not k.startswith(prefix)),'MIXED_LEGACY_ENV')
    if phase!='cohort':
        execution=_native_v3_development_execution(phase)
        fields={'MAX_WALL_SECONDS':execution['outer_seconds'],'MODEL_MAX_WALL_SECONDS':execution['internal_seconds'],
            'ATTENDED_STAGE_REQUIRED':'false','GPU_INDEX':0,'GPU_MAX_POWER_LIMIT_W':250,'GPU_MAX_POWER_DRAW_W':250,
            'GPU_MAX_CORE_TEMP_C':75,'GPU_MAX_MEMORY_TEMP_C':80,'GPU_MAX_MEMORY_USED_MIB':12288,
            'GPU_MONITOR_INTERVAL_SECONDS':1,'CPU_MAX_TEMP_C':80,'CPU_MAX_PACKAGE_POWER_W':125,
            'HOST_MIN_AVAILABLE_MEMORY_BYTES':8589934592,'HOST_MIN_COMMIT_HEADROOM_BYTES':8589934592}
        expected.update({'GX1_TRAINER_'+k:str(v)for k,v in fields.items()})
    require(all(os.environ.get(k)==v for k,v in expected.items()),'EXACT_RESOURCE_ENV')
    return require_capped_cpu_research_producer_execution()if phase=='cohort'else require_guarded_cuda_trainer_execution()

def require_native_all_tf_provider_hlc_v3_development_execution(*,plan_sha256,phase):
    """Exact one-use development containment; the leaf owns atomic CLAIM publication."""
    bound=_native_v3_development_bound_plan(plan_sha256,phase)
    proof=_native_v3_development_environment(plan_sha256,phase,bound['output_root'])
    return {**proof,**bound,'kind':'provider_hlc_all_tf_v3_development','phase':phase,
        'resource_proof_only':True,'training_authority_granted':False,'optimizer_authority_granted':False}

def native_provider_hlc_v3_development_source_bridge(producer_hashes,*,producer_commit=_V3_DEVELOPMENT_PARENT):
    import ast,hashlib,subprocess
    require=_native_v3_development_require;require(producer_commit==_V3_DEVELOPMENT_PARENT,'BRIDGE_PARENT')
    repo=Path(__file__).resolve().parents[2];names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    before={n:subprocess.check_output(['git','show',producer_commit+':'+n],cwd=repo).decode()for n in names}
    original={n:hashlib.sha256(v.encode()).hexdigest()for n,v in before.items()};require(producer_hashes==original,'BRIDGE_PRODUCER_HASHES')
    current={n:(repo/n).read_text()for n in names}
    current=_native_v3_recovery_without_additions(current)
    begin='# BEGIN GX1 V3 DEVELOPMENT GUARDS 20261127 001\n';end='# END GX1 V3 DEVELOPMENT GUARDS 20261127 001\n\n'
    text=current[names[0]];require(text.count(begin)==text.count(end)==1,'ONE_DEVELOPMENT_BLOCK')
    first,tail=text.split(begin);body,last=tail.split(end);nodes=ast.parse(body).body
    functions=[n for n in nodes if isinstance(n,ast.FunctionDef)];assignments=[n for n in nodes if isinstance(n,ast.Assign)]
    require(len(nodes)==10 and len(functions)==7 and len(assignments)==3
        and {n.name for n in functions}=={'_native_v3_development_execution','_native_v3_development_require','_native_v3_development_read','_native_v3_development_bound_plan','_native_v3_development_environment','require_native_all_tf_provider_hlc_v3_development_execution','native_provider_hlc_v3_development_source_bridge'}
        and all(len(n.targets)==1 and isinstance(n.targets[0],ast.Name)for n in assignments),'EXACT_DEVELOPMENT_AST')
    literals={n.targets[0].id:ast.literal_eval(n.value)for n in assignments}
    require(literals=={'_V3_DEVELOPMENT_NAME':_V3_DEVELOPMENT_NAME,'_V3_DEVELOPMENT_PARENT':_V3_DEVELOPMENT_PARENT,'_V3_DEVELOPMENT_SHELL_ADDITIONS':_V3_DEVELOPMENT_SHELL_ADDITIONS},'EXACT_DEVELOPMENT_LITERALS')
    require(first+last==before[names[0]],'UNCHANGED_PREDECESSOR_PYTHON')
    shell=current[names[1]]
    for anchor,position,fragment in _V3_DEVELOPMENT_SHELL_ADDITIONS:
        require(position=='before'and shell.count(anchor)==shell.count(fragment)==shell.count(fragment+anchor)==1,'EXACT_SHELL_ADDITION')
        shell=shell.replace(fragment+anchor,anchor,1)
    require(shell==before[names[1]],'UNCHANGED_PREDECESSOR_SHELL')
    return {'producer_commit':producer_commit,'checked_guard_owners':2,'old_guard_routes_preserved':True,
        'old_numeric_resource_environment_function_unchanged':True,'owners':{n:{'original_sha256':original[n],
        'current_sha256':hashlib.sha256((repo/n).read_text().encode()).hexdigest()}for n in names}}

# END GX1 V3 DEVELOPMENT GUARDS 20261127 001

# BEGIN GX1 V3 RECOVERY GUARDS 20261128 001
_V3_RECOVERY_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002'
_V3_RECOVERY_PARENT = '0330fe6c8bb0af4d1f107a3f86ed72ed98692518'
_V3_RECOVERY_SHELL_ADDITIONS = (('NATIVE_V3_DEVELOPMENT=false\n', 'before', 'NATIVE_V3_RECOVERY=false\nNATIVE_V3_RECOVERY_PLAN_SHA256=\nNATIVE_V3_RECOVERY_LANE=\n'), ('# BEGIN GX1 V3 DEVELOPMENT ROUTE 20261127 001\n', 'before', '# BEGIN GX1 V3 RECOVERY ROUTE 20261128 001\nmatches_bound_native_v3_recovery() {\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \\\n    && "$4" == --v3-development-recovery-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane ]] || return 1\n  case "$7" in reference|cold|recovery) ;; *) return 1 ;; esac\n  local plan_suffix="${7^^}"\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002_${plan_suffix}.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_recovery() {\n  matches_bound_native_v3_recovery "$@" || { echo \'FATAL: exact V3 development CLI/plan required\' >&2; exit 75; }\n  [[ "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && "$SWAP" == 512M && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 development forbids alternate resource modes\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002_$7"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 development output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_RECOVERY=true\n  NATIVE_V3_RECOVERY_PLAN_SHA256="$5"\n  NATIVE_V3_RECOVERY_LANE="$7"\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G ]] || { echo \'FATAL: V3 NN requires canonical trainer20G\' >&2; exit 75; }\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  if [[ "$7" == recovery ]]; then TRAINER_MAX_WALL_SECONDS=5400; TRAINER_MODEL_MAX_WALL_SECONDS=5370; fi\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 RECOVERY ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 DEVELOPMENT DISPATCH 20261127 001\n', 'before', '# BEGIN GX1 V3 RECOVERY DISPATCH 20261128 001\n  local recovery_arg\n  for recovery_arg in "$@"; do\n    case "$recovery_arg" in\n      --v3-development-recovery-plan-sha256|--v3-development-recovery-plan-sha256=*) configure_native_v3_recovery "$@"; return ;;\n    esac\n  done\n# END GX1 V3 RECOVERY DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_DEVELOPMENT="$NATIVE_V3_DEVELOPMENT" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_RECOVERY="$NATIVE_V3_RECOVERY" \\\n  --setenv=GX1_NATIVE_V3_RECOVERY_PLAN_SHA256="$NATIVE_V3_RECOVERY_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_V3_RECOVERY_LANE="$NATIVE_V3_RECOVERY_LANE" \\\n'))
_V3_RECOVERY_CHECKPOINT_RECEIPT_SHA256 = 'adcc8717242aa5bac0505332fbe360dc3578315bb49f5e6230ebefe15e871529'


def _native_v3_recovery_execution(lane):
    _native_v3_development_require(lane in ('reference','cold','recovery'),'RECOVERY_LANE')
    return _native_v3_development_execution('fit' if lane=='recovery' else 'cold' if lane=='cold' else 'warm')

def _native_v3_recovery_bound_plan(plan_sha256,lane):
    import hashlib,subprocess
    require=_native_v3_development_require
    require(type(plan_sha256)is str and re.fullmatch('[0-9a-f]{64}',plan_sha256)is not None,'RECOVERY_PLAN_SHA')
    require(type(lane)is str and lane in ('reference','cold','recovery'),'RECOVERY_LANE')
    repo=Path(__file__).resolve().parents[2];name=_V3_RECOVERY_NAME
    args=['--v3-development-recovery-plan-sha256',plan_sha256,'--lane',lane]
    preflight=sys.argv[-1:]==['--preflight']
    if preflight:args+=['--preflight']
    require(Path.cwd().resolve()==repo and Path(sys.argv[0]).resolve()==repo/'gx1/audit/native_provider_hlc_learning_v1.py'
        and Path(sys.executable).absolute()==repo/'.venv/bin/python'and sys.argv[1:]==args,'RECOVERY_EXACT_ARGV')
    path=repo/'configs/research'/(name+'_'+lane.upper()+'.json');plan=_native_v3_development_read(path,plan_sha256)
    output=Path('/home/andre2/GX1_RUNS')/(name+'_'+lane)
    require(plan.get('schema_version')=='gx1_native_all_tf_provider_hlc_v3_development_recovery_plan_v1'
        and plan.get('lane')==lane and plan.get('stage')==(name+'_'+lane).lower()
        and plan.get('output_root')==str(output),'RECOVERY_PLAN_IDENTITY')
    require(not output.exists()and not output.is_symlink()and output.parent.is_dir()
        and not output.parent.is_symlink()and output.parent.resolve()==output.parent,'RECOVERY_FRESH_OUTPUT')
    require(plan.get('source_commit')==subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'RECOVERY_SOURCE_COMMIT')
    require(plan.get('execution')==_native_v3_recovery_execution(lane),'RECOVERY_EXACT_EXECUTION')
    original_ref={'path':str(repo/'configs/research'/(_V3_DEVELOPMENT_NAME+'_FIT_004.json')),
        'sha256':'ebf7222c9fb403cf0adc26738cc682c088035dc2b2dc2fc5aff603a741660437'}
    receipt_ref={'path':str(Path('/home/andre2/GX1_RUNS')/(_V3_DEVELOPMENT_NAME+'_fit_004')/'CURSOR001.json'),
        'sha256':_V3_RECOVERY_CHECKPOINT_RECEIPT_SHA256}
    require(plan.get('original_plan')==original_ref and plan.get('checkpoint_receipt')==receipt_ref,'RECOVERY_ORIGINAL_BINDINGS')
    original=_native_v3_development_read(original_ref['path'],original_ref['sha256'])
    receipt=_native_v3_development_read(receipt_ref['path'],receipt_ref['sha256'])
    require(original.get('schema_version')=='gx1_native_all_tf_provider_hlc_v3_development_plan_v1'
        and original.get('phase')=='fit' and original.get('output_root')==str(Path(receipt_ref['path']).parent)
        and receipt.get('original_plan')==receipt.get('execution_plan')==original_ref
        and receipt.get('logical_cursor')==receipt.get('optimizer_updates')==1,'RECOVERY_ORIGINAL_CURSOR1')
    require(plan.get('metadata',{}).get('original')==original and plan.get('metadata',{}).get('receipt')==receipt
        and plan.get('recipe')==original.get('recipe') and plan.get('model_contract')==original.get('model_contract')
        and plan.get('backbone_gradient_policy')is original.get('backbone_gradient_policy')is None,'RECOVERY_ORIGINAL_FUNCTION')
    proto_ref=plan.get('protocol');proto_path=repo/'configs/research'/(_V3_RECOVERY_NAME.replace('_20261128_002','_PROTOCOL_20261128_002')+'.json')
    require(type(proto_ref)is dict and set(proto_ref)=={'path','sha256'}and proto_ref['path']==str(proto_path)
        and type(proto_ref['sha256'])is str and re.fullmatch('[0-9a-f]{64}',proto_ref['sha256'])is not None,'RECOVERY_PROTOCOL_REF')
    protocol=_native_v3_development_read(proto_path,proto_ref['sha256'])
    require(protocol.get('schema_version')=='gx1_native_v3_exact_recovery_protocol_v1'
        and protocol.get('experiment_id')==name and protocol.get('original_plan')==original_ref
        and protocol.get('checkpoint_receipt')==receipt_ref and protocol.get('original_protocol')==original.get('protocol')
        and protocol.get('original_trajectory')==plan.get('metadata',{}).get('trajectory'),'RECOVERY_COMMON_PROTOCOL')
    require(original.get('model_contract',{}).get('contract_sha256')=='ded7f2a05417d108ef8bed4d88f31d40ea1de21292e964586598ddaa818010c3'
        and original.get('metadata',{}).get('target_contract',{}).get('contract_sha256')=='58fd7f15bdf35646680829a718caa2991f2029ed99f48030e3a85c86aae380c5'
        and type(plan.get('limits',{}).get('optimizer_steps'))is int
        and plan['limits']['optimizer_steps']=={'reference':2,'cold':1,'recovery':431}[lane],'RECOVERY_FUNCTION_BUDGET')
    owners=plan.get('source_owners');require(type(owners)is dict and {'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh','gx1/audit/native_provider_hlc_learning_v1.py'}<=set(owners),'RECOVERY_SOURCE_INVENTORY')
    for owner,digest in owners.items():
        p=repo/owner
        require(type(owner)is str and not Path(owner).is_absolute()and '..'not in Path(owner).parts
            and p.is_file()and not p.is_symlink()and p.resolve()==p and hashlib.sha256(p.read_bytes()).hexdigest()==digest,'RECOVERY_SOURCE_SHA')
    bridge=plan.get('guard_source_bridge',{});old={n:v['original_sha256']for n,v in bridge.get('owners',{}).items()}
    require(bridge==native_provider_hlc_v3_development_recovery_source_bridge(old,producer_commit=bridge.get('producer_commit')),'RECOVERY_SOURCE_BRIDGE')
    policy=_native_v3_development_read(repo/'NEXT_RUN_POLICY.json');parent=policy.get('native_v3_fit004_exact_recovery_20261128_002',{})
    require(parent.get('original_plan')==original_ref and parent.get('checkpoint_receipt')==receipt_ref
        and parent.get('protocol')==proto_ref and parent.get('main_fit_authorized')is False
        and parent.get('test_authorized')is False and parent.get('spending_authorized')is False,'RECOVERY_PARENT_SCOPE')
    old_parent=policy.get('native_v3_bounded_development_test_20261127_001',{})
    require(old_parent.get('execution_authorized')is False and old_parent.get('active_phase')in(None,'')
        and all(type(v)is dict and v.get('execution_authorized')is False and type(v.get('remaining_invocations'))is int and v['remaining_invocations']==0 for v in old_parent.get('phase_gates',{}).values()),'RECOVERY_CONSUMED_DEVELOPMENT_CLOSED')
    flags=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed','exit_training_and_evaluation_this_wave')
    require(all(policy.get(k)is (not preflight and k in flags[:2])for k in flags),'RECOVERY_EXACT_FLAGS')
    lanes=parent.get('lane_gates',{});require(type(lanes)is dict and set(lanes)<=set(('reference','cold','recovery'))
        and all(type(v)is dict and type(v.get('remaining_invocations'))is int for v in lanes.values()),'RECOVERY_LANE_GATES')
    if preflight:
        require(parent.get('execution_authorized')is False and parent.get('active_lane')in(None,'')
            and all(v.get('execution_authorized')is False and v.get('remaining_invocations')==0 for v in lanes.values()),'RECOVERY_PREFLIGHT_CLOSED')
    else:
        status='ADMITTED_EXACT_V3_RECOVERY_'+lane.upper()
        gate={'status':status,'plan':{'path':str(path),'sha256':plan_sha256},'execution_authorized':True,
            'remaining_invocations':1,'relaunch_allowed':False,'training_authorized':True,'test_authorized':False}
        require(parent.get('status')==status and parent.get('execution_authorized')is True
            and parent.get('active_lane')==lane and lanes.get(lane)==gate
            and all(v.get('execution_authorized')is False and v.get('remaining_invocations')==0 for k,v in lanes.items()if k!=lane),'RECOVERY_ONE_USE_GATE')
    return {'plan':{'path':str(path),'sha256':plan_sha256},'output_root':str(output),'preflight':preflight,'execution':plan['execution']}

def _native_v3_recovery_environment(plan_sha256,lane):
    require=_native_v3_development_require;prefix='GX1_NATIVE_V3_RECOVERY';execution=_native_v3_recovery_execution(lane)
    expected={prefix:'true',prefix+'_PLAN_SHA256':plan_sha256,prefix+'_LANE':lane,
        'GX1_POWER_BENCHMARK_SCOPE_JSON':'','GX1_POWER_BENCHMARK_SCOPE_SHA256':''}
    require(all(not k.startswith(prefix)or k in expected for k in os.environ),'UNKNOWN_RECOVERY_ENV')
    require(all(v in('false','')for k,v in os.environ.items()if k.startswith('GX1_NATIVE_')and not k.startswith(prefix)),'RECOVERY_MIXED_LEGACY_ENV')
    fields={'MAX_WALL_SECONDS':execution['outer_seconds'],'MODEL_MAX_WALL_SECONDS':execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED':'false','GPU_INDEX':0,'GPU_MAX_POWER_LIMIT_W':250,'GPU_MAX_POWER_DRAW_W':250,
        'GPU_MAX_CORE_TEMP_C':75,'GPU_MAX_MEMORY_TEMP_C':80,'GPU_MAX_MEMORY_USED_MIB':12288,
        'GPU_MONITOR_INTERVAL_SECONDS':1,'CPU_MAX_TEMP_C':80,'CPU_MAX_PACKAGE_POWER_W':125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES':8589934592,'HOST_MIN_COMMIT_HEADROOM_BYTES':8589934592}
    expected.update({'GX1_TRAINER_'+k:str(v)for k,v in fields.items()})
    require(all(os.environ.get(k)==v for k,v in expected.items()),'RECOVERY_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()

def require_native_all_tf_provider_hlc_v3_development_recovery_execution(*,plan_sha256,lane):
    # Exact original-trajectory recovery containment; the leaf owns atomic CLAIM.
    bound=_native_v3_recovery_bound_plan(plan_sha256,lane)
    proof=_native_v3_recovery_environment(plan_sha256,lane)
    return {**proof,**bound,'kind':'provider_hlc_all_tf_v3_development_recovery','lane':lane,
        'resource_proof_only':True,'training_authority_granted':False,'optimizer_authority_granted':False}


def _native_v3_recovery_without_additions(current):
    current=_native_v3_direction_gradient_without_additions(current)
    require=_native_v3_development_require;names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    current=dict(current);begin='# BEGIN GX1 V3 RECOVERY GUARDS 20261128 001\n';end='# END GX1 V3 RECOVERY GUARDS 20261128 001\n\n'
    text=current[names[0]];require(text.count(begin)==text.count(end)==1,'ONE_RECOVERY_BLOCK')
    first,tail=text.split(begin);_,last=tail.split(end);text=first+last
    hook='    current=_native_v3_recovery_without_additions(current)\n'
    require(text.count(hook)==1,'ONE_RECOVERY_BRIDGE_HOOK');text=text.replace(hook,'',1)
    actual="hashlib.sha256((repo/n).read_text().encode()).hexdigest()"
    require(text.count(actual)==1,'ONE_RECOVERY_ACTUAL_HASH');text=text.replace(actual,"hashlib.sha256(current[n].encode()).hexdigest()",1)
    current[names[0]]=text
    shell=current[names[1]]
    for anchor,position,fragment in _V3_RECOVERY_SHELL_ADDITIONS:
        require(position=='before'and shell.count(anchor)==shell.count(fragment)==shell.count(fragment+anchor)==1,'EXACT_RECOVERY_SHELL_ADDITION')
        shell=shell.replace(fragment+anchor,anchor,1)
    current[names[1]]=shell
    return current

def native_provider_hlc_v3_development_recovery_source_bridge(producer_hashes,*,producer_commit=_V3_RECOVERY_PARENT):
    import hashlib,subprocess
    require=_native_v3_development_require;require(producer_commit==_V3_RECOVERY_PARENT,'RECOVERY_BRIDGE_PARENT')
    repo=Path(__file__).resolve().parents[2];names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    before={n:subprocess.check_output(['git','show',producer_commit+':'+n],cwd=repo).decode()for n in names}
    original={n:hashlib.sha256(v.encode()).hexdigest()for n,v in before.items()};require(producer_hashes==original,'RECOVERY_BRIDGE_PRODUCER_HASHES')
    current={n:(repo/n).read_text()for n in names}
    require(_native_v3_recovery_without_additions(current)==before,'RECOVERY_UNCHANGED_PREDECESSOR')
    return {'producer_commit':producer_commit,'checked_guard_owners':2,'old_guard_routes_preserved':True,
        'old_numeric_resource_environment_function_unchanged':True,'new_recovery_controlflow_requires_scoped_review':True,
        'owners':{n:{'original_sha256':original[n],'current_sha256':hashlib.sha256(current[n].encode()).hexdigest()}for n in names}}

# END GX1 V3 RECOVERY GUARDS 20261128 001

# BEGIN GX1 V3 DIRECTION GRADIENT GUARDS 20261128 001
_V3_DIRECTION_GRADIENT_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_GRADIENT_20261128_001'
_V3_DIRECTION_GRADIENT_PARENT = '70b975652f96ccb47792a35235827aca7e1b849e'
_V3_DIRECTION_GRADIENT_SHELL_ADDITIONS = (('NATIVE_V3_RECOVERY=false\n', 'before', 'NATIVE_V3_DIRECTION_GRADIENT=false\nNATIVE_V3_DIRECTION_GRADIENT_PLAN_SHA256=\n'), ('# BEGIN GX1 V3 RECOVERY ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 DIRECTION GRADIENT ROUTE 20261128 001\nmatches_bound_native_v3_direction_gradient() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \\\n    && "$4" == --v3-direction-gradient-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_GRADIENT_20261128_001.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_direction_gradient() {\n  matches_bound_native_v3_direction_gradient "$@" || { echo \'FATAL: exact V3 direction diagnostic CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 direction diagnostic requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_GRADIENT_20261128_001"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 direction diagnostic output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_DIRECTION_GRADIENT=true\n  NATIVE_V3_DIRECTION_GRADIENT_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 DIRECTION GRADIENT ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 RECOVERY DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 DIRECTION GRADIENT DISPATCH 20261128 001\n  local direction_gradient_arg\n  for direction_gradient_arg in "$@"; do\n    case "$direction_gradient_arg" in\n      --v3-direction-gradient-plan-sha256|--v3-direction-gradient-plan-sha256=*) configure_native_v3_direction_gradient "$@"; return ;;\n    esac\n  done\n# END GX1 V3 DIRECTION GRADIENT DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_RECOVERY="$NATIVE_V3_RECOVERY" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_DIRECTION_GRADIENT="$NATIVE_V3_DIRECTION_GRADIENT" \\\n  --setenv=GX1_NATIVE_V3_DIRECTION_GRADIENT_PLAN_SHA256="$NATIVE_V3_DIRECTION_GRADIENT_PLAN_SHA256" \\\n'))


def _native_v3_direction_gradient_without_additions(current):
    current=_native_v3_direction_block_order_without_additions(current)
    require=_native_v3_development_require;names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    current=dict(current);begin='# BEGIN GX1 V3 DIRECTION GRADIENT GUARDS 20261128 001\n';end='# END GX1 V3 DIRECTION GRADIENT GUARDS 20261128 001\n\n'
    text=current[names[0]];require(text.count(begin)==text.count(end)==1,'ONE_DIRECTION_GRADIENT_BLOCK')
    first,tail=text.split(begin);_,last=tail.split(end);text=first+last
    hook='    current=_native_v3_direction_gradient_without_additions(current)\n'
    require(text.count(hook)==1,'ONE_DIRECTION_GRADIENT_BRIDGE_HOOK')
    current[names[0]]=text.replace(hook,'',1)
    shell=current[names[1]]
    for anchor,position,fragment in _V3_DIRECTION_GRADIENT_SHELL_ADDITIONS:
        require(position=='before'and shell.count(anchor)==shell.count(fragment)==shell.count(fragment+anchor)==1,'EXACT_DIRECTION_GRADIENT_SHELL_ADDITION')
        shell=shell.replace(fragment+anchor,anchor,1)
    current[names[1]]=shell
    return current

def native_v3_direction_gradient_source_bridge(producer_hashes,*,producer_commit=_V3_DIRECTION_GRADIENT_PARENT):
    import hashlib,subprocess
    require=_native_v3_development_require;require(producer_commit==_V3_DIRECTION_GRADIENT_PARENT,'DIRECTION_GRADIENT_BRIDGE_PARENT')
    repo=Path(__file__).resolve().parents[2];names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    before={n:subprocess.check_output(['git','show',producer_commit+':'+n],cwd=repo).decode()for n in names}
    original={n:hashlib.sha256(v.encode()).hexdigest()for n,v in before.items()};require(producer_hashes==original,'DIRECTION_GRADIENT_BRIDGE_PRODUCER_HASHES')
    current={n:(repo/n).read_text()for n in names}
    require(_native_v3_direction_gradient_without_additions(current)==before,'DIRECTION_GRADIENT_UNCHANGED_PREDECESSOR')
    return {'producer_commit':producer_commit,'checked_guard_owners':2,'old_guard_routes_preserved':True,
        'old_numeric_resource_environment_function_unchanged':True,'new_diagnostic_controlflow_requires_scoped_review':True,
        'owners':{n:{'original_sha256':original[n],'current_sha256':hashlib.sha256(current[n].encode()).hexdigest()}for n in names}}


def _native_v3_direction_gradient_bound_plan(plan_sha256):
    import hashlib,subprocess
    require=_native_v3_development_require
    require(type(plan_sha256)is str and re.fullmatch('[0-9a-f]{64}',plan_sha256)is not None,'DIRECTION_GRADIENT_PLAN_SHA')
    repo=Path(__file__).resolve().parents[2];name=_V3_DIRECTION_GRADIENT_NAME
    args=['--v3-direction-gradient-plan-sha256',plan_sha256];preflight=sys.argv[-1:]==['--preflight']
    if preflight:args+=['--preflight']
    require(Path.cwd().resolve()==repo and Path(sys.argv[0]).resolve()==repo/'gx1/audit/native_provider_hlc_learning_v1.py'
        and Path(sys.executable).absolute()==repo/'.venv/bin/python'and sys.argv[1:]==args,'DIRECTION_GRADIENT_EXACT_ARGV')
    path=repo/'configs/research'/(name+'.json');plan=_native_v3_development_read(path,plan_sha256)
    output=Path('/home/andre2/GX1_RUNS')/name
    require(plan.get('schema_version')=='gx1_native_v3_direction_gradient_plan_v1'
        and plan.get('stage')==name.lower()and plan.get('output_root')==str(output),'DIRECTION_GRADIENT_PLAN_IDENTITY')
    require(not output.exists()and not output.is_symlink()and output.parent.is_dir()
        and not output.parent.is_symlink()and output.parent.resolve()==output.parent,'DIRECTION_GRADIENT_FRESH_OUTPUT')
    require(plan.get('source_commit')==subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'DIRECTION_GRADIENT_SOURCE_COMMIT')
    require(plan.get('execution')==_native_v3_development_execution('warm'),'DIRECTION_GRADIENT_EXACT_EXECUTION')
    original_ref={'path':str(repo/'configs/research'/(_V3_DEVELOPMENT_NAME+'_FIT_004.json')),
        'sha256':'ebf7222c9fb403cf0adc26738cc682c088035dc2b2dc2fc5aff603a741660437'}
    final_root=Path('/home/andre2/GX1_RUNS')/'NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002_recovery'
    receipt_ref={'path':str(final_root/'FINAL.json'),'sha256':'a5ac56d7297a75a8157daf507717c51aeffa6e4c61999f9d26711933811a364f'}
    design_ref={'path':str(Path('/home/andre2/GX1_RUNS')/'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'/'V3_DIRECTION_GRADIENT_DESIGN_20261128_001.json'),
        'sha256':'86c1b9675b29ad2f058fb9424afbd07f79ed796ab45a78fb360b9eb670f56e75'}
    require(plan.get('original_plan')==original_ref and plan.get('checkpoint_receipt')==receipt_ref
        and plan.get('design')==design_ref,'DIRECTION_GRADIENT_BOUND_REFS')
    original=_native_v3_development_read(original_ref['path'],original_ref['sha256'])
    receipt=_native_v3_development_read(receipt_ref['path'],receipt_ref['sha256'])
    _native_v3_development_read(design_ref['path'],design_ref['sha256'])
    require(receipt.get('logical_cursor')==receipt.get('optimizer_updates')==432
        and receipt.get('producer_lane')=='recovery'and receipt.get('original_plan')==original_ref
        and plan.get('metadata',{}).get('original')==original and plan.get('metadata',{}).get('receipt')==receipt
        and plan.get('model_contract')==original.get('model_contract')
        and plan.get('inputs',{}).get('final_checkpoint_receipt')==receipt_ref
        and plan.get('inputs',{}).get('final_checkpoint')==receipt.get('checkpoint'),'DIRECTION_GRADIENT_FINAL_ONLY')
    limits=plan.get('limits');expected={k:0 for k in original['limits']}
    expected.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,
        target_archive_loads=1,checkpoint_loads=1,model_restores=1,model_forwards=3,
        backbone_forwards=3,readout_forwards=3,autograd_calls=30)
    require(type(limits)is dict and all(type(v)is int for v in limits.values())and limits==expected
        and plan.get('panel_indices')==[0,107,214]and all(type(v)is int for v in plan['panel_indices']),'DIRECTION_GRADIENT_NO_UPDATE_BUDGET')
    owners=plan.get('source_owners');require(type(owners)is dict and {'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh','gx1/audit/native_provider_hlc_learning_v1.py'}<=set(owners),'DIRECTION_GRADIENT_SOURCE_INVENTORY')
    for owner,digest in owners.items():
        p=repo/owner
        require(type(owner)is str and not Path(owner).is_absolute()and '..'not in Path(owner).parts
            and p.is_file()and not p.is_symlink()and p.resolve()==p and hashlib.sha256(p.read_bytes()).hexdigest()==digest,'DIRECTION_GRADIENT_SOURCE_SHA')
    bridge=plan.get('guard_source_bridge',{});old={n:v['original_sha256']for n,v in bridge.get('owners',{}).items()}
    require(bridge==native_v3_direction_gradient_source_bridge(old,producer_commit=bridge.get('producer_commit')),'DIRECTION_GRADIENT_SOURCE_BRIDGE')
    policy=_native_v3_development_read(repo/'NEXT_RUN_POLICY.json');parent=policy.get(name.lower(),{})
    flags=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed','exit_training_and_evaluation_this_wave')
    require(all(policy.get(k)is False for k in flags),'DIRECTION_GRADIENT_ALL_TRAINING_CLOSED')
    require(parent.get('design')==design_ref and type(parent.get('remaining_invocations'))is int
        and type(parent.get('execution_authorized'))is bool and parent.get('relaunch_allowed')is False
        and parent.get('training_authorized')is False and parent.get('test_authorized')is False
        and parent.get('spending_authorized')is False,'DIRECTION_GRADIENT_DIAGNOSTIC_SCOPE')
    for scope in ('native_v3_bounded_development_test_20261127_001','native_v3_fit004_exact_recovery_20261128_001','native_v3_fit004_exact_recovery_20261128_002'):
        prior=policy.get(scope,{})
        require(prior.get('execution_authorized')is False and prior.get('active_phase')in(None,'')and prior.get('active_lane')in(None,'')
            and all(type(v)is dict and v.get('execution_authorized')is False and type(v.get('remaining_invocations'))is int
                and v['remaining_invocations']==0 for v in prior.get('lane_gates',prior.get('phase_gates',{})).values()),'DIRECTION_GRADIENT_CONSUMED_ROUTES_CLOSED')
    if preflight:
        require(parent['execution_authorized']is False and parent['remaining_invocations']==0,'DIRECTION_GRADIENT_PREFLIGHT_CLOSED')
    else:
        require(parent.get('status')=='ADMITTED_EXACT_V3_DIRECTION_GRADIENT'
            and parent.get('plan')=={'path':str(path),'sha256':plan_sha256}
            and parent['execution_authorized']is True and parent['remaining_invocations']==1,'DIRECTION_GRADIENT_ONE_USE_GATE')
    return {'plan':{'path':str(path),'sha256':plan_sha256},'output_root':str(output),'preflight':preflight,'execution':plan['execution']}

def _native_v3_direction_gradient_environment(plan_sha256):
    require=_native_v3_development_require;prefix='GX1_NATIVE_V3_DIRECTION_GRADIENT';execution=_native_v3_development_execution('warm')
    expected={prefix:'true',prefix+'_PLAN_SHA256':plan_sha256,
        'GX1_POWER_BENCHMARK_SCOPE_JSON':'','GX1_POWER_BENCHMARK_SCOPE_SHA256':''}
    require(all(not k.startswith(prefix)or k in expected for k in os.environ),'UNKNOWN_DIRECTION_GRADIENT_ENV')
    require(all(v in('false','')for k,v in os.environ.items()if k.startswith('GX1_NATIVE_')and not k.startswith(prefix)),'DIRECTION_GRADIENT_MIXED_LEGACY_ENV')
    fields={'MAX_WALL_SECONDS':execution['outer_seconds'],'MODEL_MAX_WALL_SECONDS':execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED':'false','GPU_INDEX':0,'GPU_MAX_POWER_LIMIT_W':250,'GPU_MAX_POWER_DRAW_W':250,
        'GPU_MAX_CORE_TEMP_C':75,'GPU_MAX_MEMORY_TEMP_C':80,'GPU_MAX_MEMORY_USED_MIB':12288,
        'GPU_MONITOR_INTERVAL_SECONDS':1,'CPU_MAX_TEMP_C':80,'CPU_MAX_PACKAGE_POWER_W':125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES':8589934592,'HOST_MIN_COMMIT_HEADROOM_BYTES':8589934592}
    expected.update({'GX1_TRAINER_'+k:str(v)for k,v in fields.items()})
    require(all(os.environ.get(k)==v for k,v in expected.items()),'DIRECTION_GRADIENT_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()

def require_native_v3_direction_gradient_execution(*,plan_sha256):
    # Resource containment only; no optimizer or training authority is created.
    bound=_native_v3_direction_gradient_bound_plan(plan_sha256)
    proof=_native_v3_direction_gradient_environment(plan_sha256)
    return {**proof,**bound,'kind':'provider_hlc_v3_direction_gradient','resource_proof_only':True,
        'training_authority_granted':False,'optimizer_authority_granted':False}

# END GX1 V3 DIRECTION GRADIENT GUARDS 20261128 001

# BEGIN GX1 V3 DIRECTION BLOCK ORDER GUARDS 20261128 001
_V3_DIRECTION_BLOCK_ORDER_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_BLOCK_ORDER_20261128_001'
_V3_DIRECTION_BLOCK_ORDER_PARENT = '6297d53e28e7362a9561c8cc6ef26c9122f1822c'
_V3_DIRECTION_BLOCK_ORDER_SHELL_ADDITIONS = (('NATIVE_V3_DIRECTION_GRADIENT=false\n', 'before', 'NATIVE_V3_DIRECTION_BLOCK_ORDER=false\nNATIVE_V3_DIRECTION_BLOCK_ORDER_PLAN_SHA256=\n'), ('# BEGIN GX1 V3 DIRECTION GRADIENT ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 DIRECTION BLOCK ORDER ROUTE 20261128 001\nmatches_bound_native_v3_direction_block_order() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \\\n    && "$4" == --v3-direction-block-order-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_BLOCK_ORDER_20261128_001.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_direction_block_order() {\n  matches_bound_native_v3_direction_block_order "$@" || { echo \'FATAL: exact V3 block order CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 block order requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_BLOCK_ORDER_20261128_001"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 block order output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_DIRECTION_BLOCK_ORDER=true\n  NATIVE_V3_DIRECTION_BLOCK_ORDER_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=5400\n  TRAINER_MODEL_MAX_WALL_SECONDS=5370\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 DIRECTION BLOCK ORDER ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 DIRECTION GRADIENT DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 DIRECTION BLOCK ORDER DISPATCH 20261128 001\n  local direction_block_order_arg\n  for direction_block_order_arg in "$@"; do\n    case "$direction_block_order_arg" in\n      --v3-direction-block-order-plan-sha256|--v3-direction-block-order-plan-sha256=*) configure_native_v3_direction_block_order "$@"; return ;;\n    esac\n  done\n# END GX1 V3 DIRECTION BLOCK ORDER DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_DIRECTION_GRADIENT="$NATIVE_V3_DIRECTION_GRADIENT" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER="$NATIVE_V3_DIRECTION_BLOCK_ORDER" \\\n  --setenv=GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER_PLAN_SHA256="$NATIVE_V3_DIRECTION_BLOCK_ORDER_PLAN_SHA256" \\\n'))

def _native_v3_direction_block_order_without_additions(current):
    current=_native_v3_block_order_recovery_without_additions(current)
    require=_native_v3_development_require;names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    current=dict(current);begin='# BEGIN GX1 V3 DIRECTION BLOCK ORDER GUARDS 20261128 001\n';end='# END GX1 V3 DIRECTION BLOCK ORDER GUARDS 20261128 001\n\n'
    text=current[names[0]];require(text.count(begin)==text.count(end)==1,'ONE_DIRECTION_BLOCK_ORDER_BLOCK')
    first,tail=text.split(begin);_,last=tail.split(end);text=first+last
    hook='    current=_native_v3_direction_block_order_without_additions(current)\n'
    require(text.count(hook)==1,'ONE_DIRECTION_BLOCK_ORDER_BRIDGE_HOOK')
    current[names[0]]=text.replace(hook,'',1)
    shell=current[names[1]]
    for anchor,position,fragment in _V3_DIRECTION_BLOCK_ORDER_SHELL_ADDITIONS:
        require(position=='before'and shell.count(anchor)==shell.count(fragment)==shell.count(fragment+anchor)==1,'EXACT_DIRECTION_BLOCK_ORDER_SHELL_ADDITION')
        shell=shell.replace(fragment+anchor,anchor,1)
    current[names[1]]=shell
    return current

def native_v3_direction_block_order_source_bridge(producer_hashes,*,producer_commit=_V3_DIRECTION_BLOCK_ORDER_PARENT):
    import hashlib,subprocess
    require=_native_v3_development_require;require(producer_commit==_V3_DIRECTION_BLOCK_ORDER_PARENT,'DIRECTION_BLOCK_ORDER_BRIDGE_PARENT')
    repo=Path(__file__).resolve().parents[2];names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    before={n:subprocess.check_output(['git','show',producer_commit+':'+n],cwd=repo).decode()for n in names}
    original={n:hashlib.sha256(v.encode()).hexdigest()for n,v in before.items()};require(producer_hashes==original,'DIRECTION_BLOCK_ORDER_BRIDGE_PRODUCER_HASHES')
    current={n:(repo/n).read_text()for n in names}
    require(_native_v3_direction_block_order_without_additions(current)==before,'DIRECTION_BLOCK_ORDER_UNCHANGED_PREDECESSOR')
    return {'producer_commit':producer_commit,'checked_guard_owners':2,'old_guard_routes_preserved':True,
        'old_numeric_resource_environment_function_unchanged':True,'new_order_controlflow_requires_scoped_review':True,
        'owners':{n:{'original_sha256':original[n],'current_sha256':hashlib.sha256(current[n].encode()).hexdigest()}for n in names}}


def _native_v3_direction_block_order_bound_plan(plan_sha256):
    import hashlib,subprocess
    require=_native_v3_development_require
    require(type(plan_sha256)is str and re.fullmatch('[0-9a-f]{64}',plan_sha256)is not None,'DIRECTION_BLOCK_ORDER_PLAN_SHA')
    repo=Path(__file__).resolve().parents[2];name=_V3_DIRECTION_BLOCK_ORDER_NAME
    args=['--v3-direction-block-order-plan-sha256',plan_sha256];preflight=sys.argv[-1:]==['--preflight']
    if preflight:args+=['--preflight']
    require(Path.cwd().resolve()==repo and Path(sys.argv[0]).resolve()==repo/'gx1/audit/native_provider_hlc_learning_v1.py'
        and Path(sys.executable).absolute()==repo/'.venv/bin/python'and sys.argv[1:]==args,'DIRECTION_BLOCK_ORDER_EXACT_ARGV')
    path=repo/'configs/research'/(name+'.json');plan=_native_v3_development_read(path,plan_sha256)
    output=Path('/home/andre2/GX1_RUNS')/name
    require(plan.get('schema_version')=='gx1_native_v3_direction_block_order_plan_v1'
        and plan.get('stage')==name.lower()and plan.get('output_root')==str(output),'DIRECTION_BLOCK_ORDER_PLAN_IDENTITY')
    require(not output.exists()and not output.is_symlink()and output.parent.is_dir()
        and not output.parent.is_symlink()and output.parent.resolve()==output.parent,'DIRECTION_BLOCK_ORDER_FRESH_OUTPUT')
    require(plan.get('source_commit')==subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'DIRECTION_BLOCK_ORDER_SOURCE_COMMIT')
    require(plan.get('execution')==_native_v3_development_execution('fit'),'DIRECTION_BLOCK_ORDER_EXACT_EXECUTION')
    original_ref={'path':str(repo/'configs/research'/(_V3_DEVELOPMENT_NAME+'_FIT_004.json')),
        'sha256':'ebf7222c9fb403cf0adc26738cc682c088035dc2b2dc2fc5aff603a741660437'}
    design_ref={'path':str(Path('/home/andre2/GX1_RUNS')/'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'/'V3_DIRECTION_BLOCK_ORDER_DESIGN_20261128_001.json'),
        'sha256':'573e747a4ebea2f91a228a5853f4f16eb5a7411280dc62832d46a3e43f250942'}
    require(plan.get('original_plan')==original_ref and plan.get('design')==design_ref,'DIRECTION_BLOCK_ORDER_BOUND_REFS')
    original=_native_v3_development_read(original_ref['path'],original_ref['sha256'])
    _native_v3_development_read(design_ref['path'],design_ref['sha256'])
    require(plan.get('metadata',{}).get('original')==original
        and plan.get('model_contract')==original.get('model_contract')
        and plan.get('initial_reference')==original.get('initial_reference')
        and plan.get('recipe')==original.get('recipe'),'DIRECTION_BLOCK_ORDER_UNCHANGED_FUNCTION_FACTORY_RECIPE')
    schedule=plan.get('schedule',{});canonical=original['schedule'];permutations=schedule.get('permutations')
    require(schedule.get('schema_version')=='gx1_native_v3_whole_batch_order_schedule_v1'
        and schedule.get('canonical_schedule')==canonical and type(permutations)is list and len(permutations)==2
        and all(type(v)is list and len(v)==216 and all(type(i)is int for i in v)
            and sorted(v)==list(range(216))for v in permutations),'DIRECTION_BLOCK_ORDER_EXACT_PERMUTATIONS')
    ids=[i for permutation in permutations for i in permutation]
    require(schedule.get('logical_original_batch_ids')==ids
        and all(type(i)is int for i in schedule['logical_original_batch_ids'])
        and schedule.get('ordered_batch_source_rows_sha256')==[canonical['batch_source_rows_sha256'][i]for i in ids]
        and schedule.get('ordered_batch_issue_ns_sha256')==[canonical['batch_issue_ns_sha256'][i]for i in ids]
        and all(type(schedule.get(k))is int and schedule[k]==v for k,v in
            {'rows':20727,'batch_rows':96,'passes':2,'batches_per_pass':216,'logical_batches':432,'optimizer_updates':432,'skipped_batches':0}.items())
        and schedule.get('update_flags')==[True]*432 and all(type(v)is bool for v in schedule['update_flags'])
        and plan.get('metadata',{}).get('trajectory',{}).get('schedule')==schedule,'DIRECTION_BLOCK_ORDER_SOURCE_BOUND_CURSOR')
    expected=dict(original['limits'])
    expected.update(model_forwards=597,backbone_forwards=597,readout_forwards=597,
        logit_archive_writes=2,logit_archive_readbacks=2)
    limits=plan.get('limits')
    require(type(limits)is dict and all(type(v)is int for v in limits.values())and limits==expected
        and plan.get('checkpoint_cursors')==[1,432]and all(type(v)is int for v in plan['checkpoint_cursors'])
        and plan.get('panel_indices')==[0,107,214]and all(type(v)is int for v in plan['panel_indices']),'DIRECTION_BLOCK_ORDER_EXACT_BUDGET')
    owners=plan.get('source_owners');require(type(owners)is dict and {'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh','gx1/audit/native_provider_hlc_learning_v1.py'}<=set(owners),'DIRECTION_BLOCK_ORDER_SOURCE_INVENTORY')
    for owner,digest in owners.items():
        p=repo/owner
        require(type(owner)is str and not Path(owner).is_absolute()and '..'not in Path(owner).parts
            and p.is_file()and not p.is_symlink()and p.resolve()==p and hashlib.sha256(p.read_bytes()).hexdigest()==digest,'DIRECTION_BLOCK_ORDER_SOURCE_SHA')
    bridge=plan.get('guard_source_bridge',{});old={n:v['original_sha256']for n,v in bridge.get('owners',{}).items()}
    require(bridge==native_v3_direction_block_order_source_bridge(old,producer_commit=bridge.get('producer_commit')),'DIRECTION_BLOCK_ORDER_SOURCE_BRIDGE')
    policy=_native_v3_development_read(repo/'NEXT_RUN_POLICY.json');parent=policy.get(name.lower(),{})
    flags=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed','exit_training_and_evaluation_this_wave')
    require(all(policy.get(k)is (not preflight and k in flags[:2])for k in flags),'DIRECTION_BLOCK_ORDER_EXACT_FLAGS')
    require(parent.get('design')==design_ref and type(parent.get('remaining_invocations'))is int
        and type(parent.get('execution_authorized'))is bool and parent.get('relaunch_allowed')is False
        and parent.get('training_authorized')is (not preflight)and parent.get('test_authorized')is False
        and parent.get('spending_authorized')is False,'DIRECTION_BLOCK_ORDER_FIT_SCOPE')
    for scope in ('native_v3_bounded_development_test_20261127_001','native_v3_fit004_exact_recovery_20261128_001','native_v3_fit004_exact_recovery_20261128_002'):
        prior=policy.get(scope,{})
        require(prior.get('execution_authorized')is False and prior.get('active_phase')in(None,'')and prior.get('active_lane')in(None,'')
            and all(type(v)is dict and v.get('execution_authorized')is False and type(v.get('remaining_invocations'))is int
                and v['remaining_invocations']==0 for v in prior.get('lane_gates',prior.get('phase_gates',{})).values()),'DIRECTION_BLOCK_ORDER_CONSUMED_ROUTES_CLOSED')
    diagnostic=policy.get(_V3_DIRECTION_GRADIENT_NAME.lower(),{})
    require(diagnostic.get('execution_authorized')is False and diagnostic.get('remaining_invocations')==0
        and type(diagnostic.get('remaining_invocations'))is int and diagnostic.get('training_authorized')is False,'DIRECTION_BLOCK_ORDER_DIAGNOSTIC_CLOSED')
    if preflight:
        require(parent['execution_authorized']is False and parent['remaining_invocations']==0,'DIRECTION_BLOCK_ORDER_PREFLIGHT_CLOSED')
    else:
        require(parent.get('status')=='ADMITTED_EXACT_V3_DIRECTION_BLOCK_ORDER'
            and parent.get('plan')=={'path':str(path),'sha256':plan_sha256}
            and parent['execution_authorized']is True and parent['remaining_invocations']==1,'DIRECTION_BLOCK_ORDER_ONE_USE_GATE')
    return {'plan':{'path':str(path),'sha256':plan_sha256},'output_root':str(output),'preflight':preflight,'execution':plan['execution']}

def _native_v3_direction_block_order_environment(plan_sha256):
    require=_native_v3_development_require;prefix='GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER';execution=_native_v3_development_execution('fit')
    expected={prefix:'true',prefix+'_PLAN_SHA256':plan_sha256,
        'GX1_POWER_BENCHMARK_SCOPE_JSON':'','GX1_POWER_BENCHMARK_SCOPE_SHA256':''}
    require(all(not k.startswith(prefix)or k in expected for k in os.environ),'UNKNOWN_DIRECTION_BLOCK_ORDER_ENV')
    require(all(v in('false','')for k,v in os.environ.items()if k.startswith('GX1_NATIVE_')and not k.startswith(prefix)),'DIRECTION_BLOCK_ORDER_MIXED_LEGACY_ENV')
    fields={'MAX_WALL_SECONDS':execution['outer_seconds'],'MODEL_MAX_WALL_SECONDS':execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED':'false','GPU_INDEX':0,'GPU_MAX_POWER_LIMIT_W':250,'GPU_MAX_POWER_DRAW_W':250,
        'GPU_MAX_CORE_TEMP_C':75,'GPU_MAX_MEMORY_TEMP_C':80,'GPU_MAX_MEMORY_USED_MIB':12288,
        'GPU_MONITOR_INTERVAL_SECONDS':1,'CPU_MAX_TEMP_C':80,'CPU_MAX_PACKAGE_POWER_W':125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES':8589934592,'HOST_MIN_COMMIT_HEADROOM_BYTES':8589934592}
    expected.update({'GX1_TRAINER_'+k:str(v)for k,v in fields.items()})
    require(all(os.environ.get(k)==v for k,v in expected.items()),'DIRECTION_BLOCK_ORDER_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()

def require_native_v3_direction_block_order_execution(*,plan_sha256):
    # Resource containment only; the leaf owns exact claim and scoped fit admission.
    bound=_native_v3_direction_block_order_bound_plan(plan_sha256)
    proof=_native_v3_direction_block_order_environment(plan_sha256)
    return {**proof,**bound,'kind':'provider_hlc_v3_direction_block_order','resource_proof_only':True,
        'training_authority_granted':False,'optimizer_authority_granted':False}

# END GX1 V3 DIRECTION BLOCK ORDER GUARDS 20261128 001

# BEGIN GX1 V3 BLOCK ORDER RECOVERY GUARDS 20261128 001
_V3_BLOCK_ORDER_RECOVERY_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_BLOCK_ORDER_RECOVERY_20261128_001'
_V3_BLOCK_ORDER_RECOVERY_PARENT = 'd405fbebdcd0621501def940bd90855bd1c5709f'
_V3_BLOCK_ORDER_RECOVERY_SHELL_ADDITIONS = (('NATIVE_V3_DIRECTION_BLOCK_ORDER=false\n', 'before', 'NATIVE_V3_BLOCK_ORDER_RECOVERY=false\nNATIVE_V3_BLOCK_ORDER_RECOVERY_PLAN_SHA256=\nNATIVE_V3_BLOCK_ORDER_RECOVERY_LANE=\n'), ('# BEGIN GX1 V3 DIRECTION BLOCK ORDER ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 BLOCK ORDER RECOVERY ROUTE 20261128 001\nmatches_bound_native_v3_block_order_recovery() {\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \\\n    && "$4" == --v3-block-order-recovery-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane ]] || return 1\n  case "$7" in reference|cold|recovery) ;; *) return 1 ;; esac\n  local plan_suffix="${7^^}"\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_BLOCK_ORDER_RECOVERY_20261128_001_${plan_suffix}.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_block_order_recovery() {\n  matches_bound_native_v3_block_order_recovery "$@" || { echo \'FATAL: exact V3 development CLI/plan required\' >&2; exit 75; }\n  [[ "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && "$SWAP" == 512M && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 development forbids alternate resource modes\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_BLOCK_ORDER_RECOVERY_20261128_001_$7"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 development output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_BLOCK_ORDER_RECOVERY=true\n  NATIVE_V3_BLOCK_ORDER_RECOVERY_PLAN_SHA256="$5"\n  NATIVE_V3_BLOCK_ORDER_RECOVERY_LANE="$7"\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G ]] || { echo \'FATAL: V3 NN requires canonical trainer20G\' >&2; exit 75; }\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  if [[ "$7" == recovery ]]; then TRAINER_MAX_WALL_SECONDS=5400; TRAINER_MODEL_MAX_WALL_SECONDS=5370; fi\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 BLOCK ORDER RECOVERY ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 DIRECTION BLOCK ORDER DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 BLOCK ORDER RECOVERY DISPATCH 20261128 001\n  local block_order_recovery_arg\n  for block_order_recovery_arg in "$@"; do\n    case "$block_order_recovery_arg" in\n      --v3-block-order-recovery-plan-sha256|--v3-block-order-recovery-plan-sha256=*) configure_native_v3_block_order_recovery "$@"; return ;;\n    esac\n  done\n# END GX1 V3 BLOCK ORDER RECOVERY DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER="$NATIVE_V3_DIRECTION_BLOCK_ORDER" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY="$NATIVE_V3_BLOCK_ORDER_RECOVERY" \\\n  --setenv=GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY_PLAN_SHA256="$NATIVE_V3_BLOCK_ORDER_RECOVERY_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY_LANE="$NATIVE_V3_BLOCK_ORDER_RECOVERY_LANE" \\\n'))

def _native_v3_block_order_recovery_without_additions(current):
    current=_native_v3_issue_mix_without_additions(current)
    require=_native_v3_development_require;names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    current=dict(current);begin='# BEGIN GX1 V3 BLOCK ORDER RECOVERY GUARDS 20261128 001\n';end='# END GX1 V3 BLOCK ORDER RECOVERY GUARDS 20261128 001\n\n'
    text=current[names[0]];require(text.count(begin)==text.count(end)==1,'ONE_BLOCK_ORDER_RECOVERY_BLOCK')
    first,tail=text.split(begin);_,last=tail.split(end);text=first+last
    hook='    current=_native_v3_block_order_recovery_without_additions(current)\n'
    require(text.count(hook)==1,'ONE_BLOCK_ORDER_RECOVERY_BRIDGE_HOOK')
    current[names[0]]=text.replace(hook,'',1)
    shell=current[names[1]]
    for anchor,position,fragment in _V3_BLOCK_ORDER_RECOVERY_SHELL_ADDITIONS:
        require(position=='before'and shell.count(anchor)==shell.count(fragment)==shell.count(fragment+anchor)==1,'EXACT_BLOCK_ORDER_RECOVERY_SHELL_ADDITION')
        shell=shell.replace(fragment+anchor,anchor,1)
    current[names[1]]=shell
    return current

def native_v3_block_order_recovery_source_bridge(producer_hashes,*,producer_commit=_V3_BLOCK_ORDER_RECOVERY_PARENT):
    import hashlib,subprocess
    require=_native_v3_development_require;require(producer_commit==_V3_BLOCK_ORDER_RECOVERY_PARENT,'BLOCK_ORDER_RECOVERY_BRIDGE_PARENT')
    repo=Path(__file__).resolve().parents[2];names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    before={n:subprocess.check_output(['git','show',producer_commit+':'+n],cwd=repo).decode()for n in names}
    original={n:hashlib.sha256(v.encode()).hexdigest()for n,v in before.items()};require(producer_hashes==original,'BLOCK_ORDER_RECOVERY_BRIDGE_PRODUCER_HASHES')
    current={n:(repo/n).read_text()for n in names}
    require(_native_v3_block_order_recovery_without_additions(current)==before,'BLOCK_ORDER_RECOVERY_UNCHANGED_PREDECESSOR')
    return {'producer_commit':producer_commit,'checked_guard_owners':2,'old_guard_routes_preserved':True,
        'old_numeric_resource_environment_function_unchanged':True,'new_order_recovery_controlflow_requires_scoped_review':True,
        'owners':{n:{'original_sha256':original[n],'current_sha256':hashlib.sha256(current[n].encode()).hexdigest()}for n in names}}


def _native_v3_block_order_recovery_bound_plan(plan_sha256,lane):
    import hashlib,subprocess
    require=_native_v3_development_require
    require(type(plan_sha256)is str and re.fullmatch('[0-9a-f]{64}',plan_sha256)is not None,'BLOCK_ORDER_RECOVERY_PLAN_SHA')
    require(type(lane)is str and lane in ('reference','cold','recovery'),'BLOCK_ORDER_RECOVERY_LANE')
    repo=Path(__file__).resolve().parents[2];name=_V3_BLOCK_ORDER_RECOVERY_NAME
    args=['--v3-block-order-recovery-plan-sha256',plan_sha256,'--lane',lane];preflight=sys.argv[-1:]==['--preflight']
    if preflight:args+=['--preflight']
    require(Path.cwd().resolve()==repo and Path(sys.argv[0]).resolve()==repo/'gx1/audit/native_provider_hlc_learning_v1.py'
        and Path(sys.executable).absolute()==repo/'.venv/bin/python'and sys.argv[1:]==args,'BLOCK_ORDER_RECOVERY_EXACT_ARGV')
    path=repo/'configs/research'/(name+'_'+lane.upper()+'.json');plan=_native_v3_development_read(path,plan_sha256)
    output=Path('/home/andre2/GX1_RUNS')/(name+'_'+lane)
    require(plan.get('schema_version')=='gx1_native_v3_block_order_recovery_plan_v1'
        and plan.get('stage')==name.lower()and plan.get('lane')==lane
        and plan.get('output_root')==str(output),'BLOCK_ORDER_RECOVERY_PLAN_IDENTITY')
    require(not output.exists()and not output.is_symlink()and output.parent.is_dir()
        and not output.parent.is_symlink()and output.parent.resolve()==output.parent,'BLOCK_ORDER_RECOVERY_FRESH_OUTPUT')
    require(plan.get('source_commit')==subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'BLOCK_ORDER_RECOVERY_SOURCE_COMMIT')
    require(plan.get('execution')==_native_v3_recovery_execution(lane),'BLOCK_ORDER_RECOVERY_EXACT_EXECUTION')
    original_ref={'path':str(repo/'configs/research'/(_V3_DIRECTION_BLOCK_ORDER_NAME+'.json')),
        'sha256':'15a59b3840d61d57d6e217967312a6b76e4e89d256ea902b619ec3fa8099a008'}
    receipt_ref={'path':str(Path('/home/andre2/GX1_RUNS')/_V3_DIRECTION_BLOCK_ORDER_NAME/'CURSOR001.json'),
        'sha256':'9d6f3ec4908f2cd2c60f1a6c5e8c9980f59bd8b2933dd0500e50bf8b3548a7f7'}
    design_ref={'path':str(Path('/home/andre2/GX1_RUNS')/'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'/'V3_BLOCK_ORDER_EXACT_RECOVERY_DESIGN_20261128_001.json'),
        'sha256':'737dd510ca83958d1f1bb192ef22ca40e072c0aa9b5c757cc3787bb464ebb830'}
    require(plan.get('original_plan')==original_ref and plan.get('checkpoint_receipt')==receipt_ref
        and plan.get('design')==design_ref,'BLOCK_ORDER_RECOVERY_BOUND_REFS')
    original=_native_v3_development_read(original_ref['path'],original_ref['sha256'])
    receipt=_native_v3_development_read(receipt_ref['path'],receipt_ref['sha256'])
    _native_v3_development_read(design_ref['path'],design_ref['sha256'])
    require(receipt.get('logical_cursor')==receipt.get('optimizer_updates')==1
        and receipt.get('producer_lane')=='block_order'and receipt.get('original_plan')==receipt.get('execution_plan')==original_ref
        and receipt.get('trajectory_sha256')==original['metadata']['trajectory']['contract_sha256']=='214fceb7d316c33edd928936612fa0ad3fc2fd3f9843396cea35e41fcbd3a0b6'
        and original['schedule']['contract_sha256']=='dba16db2b9c04c1fd988a79d3fd713b9719bbfc3514b76b38174b92ccfef19de'
        and plan.get('metadata',{}).get('original')==original and plan.get('metadata',{}).get('receipt')==receipt
        and plan.get('metadata',{}).get('trajectory')==original['metadata']['trajectory']
        and all(plan.get(k)==original.get(k)for k in ('schedule','recipe','model_contract','initial_reference')),'BLOCK_ORDER_RECOVERY_ORIGINAL_SHUFFLED_C1')
    protocol_ref=plan.get('protocol');protocol_path=repo/'configs/research'/(name.replace('_20261128_001','_PROTOCOL_20261128_001')+'.json')
    require(type(protocol_ref)is dict and set(protocol_ref)=={'path','sha256'}and protocol_ref['path']==str(protocol_path)
        and type(protocol_ref['sha256'])is str and re.fullmatch('[0-9a-f]{64}',protocol_ref['sha256'])is not None,'BLOCK_ORDER_RECOVERY_PROTOCOL_REF')
    protocol=_native_v3_development_read(protocol_path,protocol_ref['sha256'])
    require(protocol.get('schema_version')=='gx1_v3_block_order_recovery_protocol_v1'
        and protocol.get('experiment_id')==name and protocol.get('original_plan')==original_ref
        and protocol.get('checkpoint_receipt')==receipt_ref and protocol.get('original_trajectory')==original['metadata']['trajectory']
        and protocol.get('original_schedule')==original['schedule'],'BLOCK_ORDER_RECOVERY_COMMON_PROTOCOL')
    expected={k:0 for k in original['limits']};updates={'reference':2,'cold':1,'recovery':431}[lane];forwards={'reference':2,'cold':1,'recovery':596}[lane]
    expected.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,
        target_archive_loads=1,model_forwards=forwards,backbone_forwards=forwards,readout_forwards=forwards,
        optimizer_steps=updates,backward_calls=updates,checkpoint_writes=1,checkpoint_readbacks=1,
        checkpoint_receipt_writes=1,checkpoint_prepublication_loads=1)
    if lane!='reference':expected.update(checkpoint_loads=1,model_restores=1)
    if lane=='recovery':expected.update(autograd_calls=9,logit_archive_writes=2,logit_archive_readbacks=2)
    limits=plan.get('limits');require(type(limits)is dict and all(type(v)is int for v in limits.values())
        and limits==expected,'BLOCK_ORDER_RECOVERY_EXACT_BUDGET')
    owners=plan.get('source_owners');require(type(owners)is dict and {'gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh','gx1/audit/native_provider_hlc_learning_v1.py'}<=set(owners),'BLOCK_ORDER_RECOVERY_SOURCE_INVENTORY')
    for owner,digest in owners.items():
        p=repo/owner
        require(type(owner)is str and not Path(owner).is_absolute()and '..'not in Path(owner).parts
            and p.is_file()and not p.is_symlink()and p.resolve()==p and hashlib.sha256(p.read_bytes()).hexdigest()==digest,'BLOCK_ORDER_RECOVERY_SOURCE_SHA')
    bridge=plan.get('guard_source_bridge',{});old={n:v['original_sha256']for n,v in bridge.get('owners',{}).items()}
    require(bridge==native_v3_block_order_recovery_source_bridge(old,producer_commit=bridge.get('producer_commit')),'BLOCK_ORDER_RECOVERY_SOURCE_BRIDGE')
    policy=_native_v3_development_read(repo/'NEXT_RUN_POLICY.json');parent=policy.get(name.lower(),{})
    flags=('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed','exit_training_and_evaluation_this_wave')
    require(all(policy.get(k)is (not preflight and k in flags[:2])for k in flags),'BLOCK_ORDER_RECOVERY_EXACT_FLAGS')
    require(parent.get('original_plan')==original_ref and parent.get('checkpoint_receipt')==receipt_ref
        and parent.get('design')==design_ref and parent.get('protocol')==protocol_ref
        and parent.get('main_fit_authorized')is False and parent.get('test_authorized')is False
        and parent.get('spending_authorized')is False,'BLOCK_ORDER_RECOVERY_SCOPE')
    for scope in (_V3_DIRECTION_BLOCK_ORDER_NAME.lower(),_V3_DIRECTION_GRADIENT_NAME.lower()):
        prior=policy.get(scope,{})
        require(prior.get('execution_authorized')is False and prior.get('remaining_invocations')==0
            and type(prior.get('remaining_invocations'))is int and prior.get('training_authorized')is False
            and prior.get('relaunch_allowed')is False,'BLOCK_ORDER_RECOVERY_CONSUMED_ORIGINAL_CLOSED')
    lanes=parent.get('lane_gates',{})
    require(type(lanes)is dict and set(lanes)<=set(('reference','cold','recovery'))
        and all(type(v)is dict and type(v.get('remaining_invocations'))is int
            and all(type(v.get(k))is bool for k in ('execution_authorized','relaunch_allowed','training_authorized','test_authorized'))for v in lanes.values()),'BLOCK_ORDER_RECOVERY_TYPED_LANES')
    closed=lambda v:v['execution_authorized']is False and v['remaining_invocations']==0 and v['training_authorized']is False and v['test_authorized']is False and v['relaunch_allowed']is False
    if preflight:
        require(parent.get('execution_authorized')is False and parent.get('active_lane')in(None,'')
            and all(closed(v)for v in lanes.values()),'BLOCK_ORDER_RECOVERY_PREFLIGHT_CLOSED')
    else:
        status='ADMITTED_EXACT_V3_BLOCK_ORDER_RECOVERY_'+lane.upper()
        gate={'status':status,'plan':{'path':str(path),'sha256':plan_sha256},'execution_authorized':True,
            'remaining_invocations':1,'relaunch_allowed':False,'training_authorized':True,'test_authorized':False}
        require(parent.get('status')==status and parent.get('execution_authorized')is True and parent.get('active_lane')==lane
            and lanes.get(lane)==gate and all(closed(v)for k,v in lanes.items()if k!=lane),'BLOCK_ORDER_RECOVERY_ONE_USE_GATE')
    return {'plan':{'path':str(path),'sha256':plan_sha256},'output_root':str(output),'preflight':preflight,'execution':plan['execution']}

def _native_v3_block_order_recovery_environment(plan_sha256,lane):
    require=_native_v3_development_require;prefix='GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY';execution=_native_v3_recovery_execution(lane)
    expected={prefix:'true',prefix+'_PLAN_SHA256':plan_sha256,prefix+'_LANE':lane,
        'GX1_POWER_BENCHMARK_SCOPE_JSON':'','GX1_POWER_BENCHMARK_SCOPE_SHA256':''}
    require(all(not k.startswith(prefix)or k in expected for k in os.environ),'UNKNOWN_BLOCK_ORDER_RECOVERY_ENV')
    require(all(v in('false','')for k,v in os.environ.items()if k.startswith('GX1_NATIVE_')and not k.startswith(prefix)),'BLOCK_ORDER_RECOVERY_MIXED_LEGACY_ENV')
    fields={'MAX_WALL_SECONDS':execution['outer_seconds'],'MODEL_MAX_WALL_SECONDS':execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED':'false','GPU_INDEX':0,'GPU_MAX_POWER_LIMIT_W':250,'GPU_MAX_POWER_DRAW_W':250,
        'GPU_MAX_CORE_TEMP_C':75,'GPU_MAX_MEMORY_TEMP_C':80,'GPU_MAX_MEMORY_USED_MIB':12288,
        'GPU_MONITOR_INTERVAL_SECONDS':1,'CPU_MAX_TEMP_C':80,'CPU_MAX_PACKAGE_POWER_W':125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES':8589934592,'HOST_MIN_COMMIT_HEADROOM_BYTES':8589934592}
    expected.update({'GX1_TRAINER_'+k:str(v)for k,v in fields.items()})
    require(all(os.environ.get(k)==v for k,v in expected.items()),'BLOCK_ORDER_RECOVERY_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()

def require_native_v3_block_order_recovery_execution(*,plan_sha256,lane):
    # Resource containment only; semantic replay and the exact claim remain leaf-owned.
    bound=_native_v3_block_order_recovery_bound_plan(plan_sha256,lane)
    proof=_native_v3_block_order_recovery_environment(plan_sha256,lane)
    return {**proof,**bound,'kind':'provider_hlc_v3_block_order_recovery','lane':lane,'resource_proof_only':True,
        'training_authority_granted':False,'optimizer_authority_granted':False}

# END GX1 V3 BLOCK ORDER RECOVERY GUARDS 20261128 001

# BEGIN GX1 V3 ISSUE MIX GUARDS 20261128 001
_V3_ISSUE_MIX_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_20261128_001'
_V3_ISSUE_MIX_PARENT = '4cb788b15fb47f2b6b2273d34485a34753649f72'
_V3_ISSUE_MIX_PROFILE = 'gx1_native_provider_hlc_v3_issue_mix_trainer20g_v1'
_V3_ISSUE_MIX_SHELL_ADDITIONS = (('NATIVE_V3_BLOCK_ORDER_RECOVERY=false\n', 'before', 'NATIVE_V3_ISSUE_MIX=false\nNATIVE_V3_ISSUE_MIX_PLAN_SHA256=\nNATIVE_V3_ISSUE_MIX_LANE=\n'), ('# BEGIN GX1 V3 BLOCK ORDER RECOVERY ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 ISSUE MIX ROUTE 20261128 001\nmatches_bound_native_v3_issue_mix() {\n  if [[ $# -eq 8 ]]; then\n    [[ "$8" == --preflight ]] || return 1\n    set -- "${@:1:7}"\n  fi\n  [[ $# -eq 7 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \\\n    && "$4" == --v3-issue-mix-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ \\\n    && "$6" == --lane ]] || return 1\n  case "$7" in reference|cold|fit) ;; *) return 1 ;; esac\n  local plan_suffix="${7^^}"\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_20261128_001_${plan_suffix}.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_issue_mix() {\n  matches_bound_native_v3_issue_mix "$@" || { echo \'FATAL: exact V3 issue mix CLI/plan required\' >&2; exit 75; }\n  [[ "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && "$SWAP" == 512M && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 issue mix forbids alternate resource modes\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_20261128_001_$7"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 issue mix output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_ISSUE_MIX=true\n  NATIVE_V3_ISSUE_MIX_PLAN_SHA256="$5"\n  NATIVE_V3_ISSUE_MIX_LANE="$7"\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G ]] || { echo \'FATAL: V3 NN requires canonical trainer20G\' >&2; exit 75; }\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  if [[ "$7" == fit ]]; then TRAINER_MAX_WALL_SECONDS=5400; TRAINER_MODEL_MAX_WALL_SECONDS=5370; fi\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 ISSUE MIX ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 BLOCK ORDER RECOVERY DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 ISSUE MIX DISPATCH 20261128 001\n  local issue_mix_arg\n  for issue_mix_arg in "$@"; do\n    case "$issue_mix_arg" in\n      --v3-issue-mix-plan-sha256|--v3-issue-mix-plan-sha256=*) configure_native_v3_issue_mix "$@"; return ;;\n    esac\n  done\n# END GX1 V3 ISSUE MIX DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY="$NATIVE_V3_BLOCK_ORDER_RECOVERY" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_ISSUE_MIX="$NATIVE_V3_ISSUE_MIX" \\\n  --setenv=GX1_NATIVE_V3_ISSUE_MIX_PLAN_SHA256="$NATIVE_V3_ISSUE_MIX_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_V3_ISSUE_MIX_LANE="$NATIVE_V3_ISSUE_MIX_LANE" \\\n'))


def _native_v3_issue_mix_execution(lane):
    _native_v3_development_require(lane in ('reference', 'cold', 'fit'), 'ISSUE_MIX_LANE')
    return _native_v3_development_execution('fit' if lane == 'fit' else 'cold' if lane == 'cold' else 'warm')


def _native_v3_issue_mix_without_additions(current):
    current=_native_v3_train_eval_gap_without_additions(current)
    require = _native_v3_development_require
    names = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')
    current = dict(current)
    begin = '# BEGIN GX1 V3 ISSUE MIX GUARDS 20261128 001\n'
    end = '# END GX1 V3 ISSUE MIX GUARDS 20261128 001\n\n'
    text = current[names[0]]
    require(text.count(begin) == text.count(end) == 1, 'ONE_ISSUE_MIX_BLOCK')
    first, tail = text.split(begin); _, last = tail.split(end); text = first + last
    hook = '    current=_native_v3_issue_mix_without_additions(current)\n'
    require(text.count(hook) == 1, 'ONE_ISSUE_MIX_BRIDGE_HOOK')
    current[names[0]] = text.replace(hook, '', 1)
    shell = current[names[1]]
    for anchor, position, fragment in _V3_ISSUE_MIX_SHELL_ADDITIONS:
        require(position == 'before' and shell.count(anchor) == shell.count(fragment)
                == shell.count(fragment + anchor) == 1, 'EXACT_ISSUE_MIX_SHELL_ADDITION')
        shell = shell.replace(fragment + anchor, anchor, 1)
    current[names[1]] = shell
    return current


def native_v3_issue_mix_source_bridge(producer_hashes, *, producer_commit=_V3_ISSUE_MIX_PARENT):
    import hashlib, subprocess
    require = _native_v3_development_require
    require(producer_commit == _V3_ISSUE_MIX_PARENT, 'ISSUE_MIX_BRIDGE_PARENT')
    repo = Path(__file__).resolve().parents[2]
    names = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')
    before = {n: subprocess.check_output(['git', 'show', producer_commit + ':' + n], cwd=repo).decode() for n in names}
    original = {n: hashlib.sha256(v.encode()).hexdigest() for n, v in before.items()}
    require(producer_hashes == original, 'ISSUE_MIX_BRIDGE_PRODUCER_HASHES')
    current = {n: (repo / n).read_text() for n in names}
    require(_native_v3_issue_mix_without_additions(current) == before, 'ISSUE_MIX_UNCHANGED_PREDECESSOR')
    return {'producer_commit': producer_commit, 'checked_guard_owners': 2, 'old_guard_routes_preserved': True,
            'old_numeric_resource_environment_function_unchanged': True,
            'new_issue_mix_controlflow_requires_scoped_review': True,
            'owners': {n: {'original_sha256': original[n],
                          'current_sha256': hashlib.sha256(current[n].encode()).hexdigest()} for n in names}}


def _native_v3_issue_mix_bound_plan(plan_sha256, lane):
    import hashlib, subprocess
    require = _native_v3_development_require
    require(type(plan_sha256) is str and re.fullmatch('[0-9a-f]{64}', plan_sha256) is not None, 'ISSUE_MIX_PLAN_SHA')
    require(type(lane) is str and lane in ('reference', 'cold', 'fit'), 'ISSUE_MIX_LANE')
    repo = Path(__file__).resolve().parents[2]; name = _V3_ISSUE_MIX_NAME
    args = ['--v3-issue-mix-plan-sha256', plan_sha256, '--lane', lane]
    preflight = sys.argv[-1:] == ['--preflight']
    if preflight:
        args += ['--preflight']
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / 'gx1/audit/native_provider_hlc_learning_v1.py'
            and Path(sys.executable).absolute() == repo / '.venv/bin/python' and sys.argv[1:] == args, 'ISSUE_MIX_EXACT_ARGV')
    path = repo / 'configs/research' / (name + '_' + lane.upper() + '.json')
    plan = _native_v3_development_read(path, plan_sha256)
    output = Path('/home/andre2/GX1_RUNS') / (name + '_' + lane)
    require(plan.get('schema_version') == 'gx1_native_v3_direction_issue_mix_plan_v1'
            and plan.get('stage') == name.lower() and plan.get('lane') == lane
            and plan.get('output_root') == str(output), 'ISSUE_MIX_PLAN_IDENTITY')
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent.resolve() == output.parent, 'ISSUE_MIX_FRESH_OUTPUT')
    require(plan.get('source_commit') == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'ISSUE_MIX_SOURCE_COMMIT')
    require(plan.get('execution_profile') == _V3_ISSUE_MIX_PROFILE
            and plan.get('execution') == _native_v3_issue_mix_execution(lane), 'ISSUE_MIX_EXACT_EXECUTION')
    control_ref = {'path': str(repo / 'configs/research' / (_V3_DIRECTION_BLOCK_ORDER_NAME + '.json')),
                   'sha256': '15a59b3840d61d57d6e217967312a6b76e4e89d256ea902b619ec3fa8099a008'}
    evidence = Path('/home/andre2/GX1_RUNS') / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = {'path': str(evidence / 'V3_DIRECTION_ISSUE_MIX_DESIGN_20261128_001.json'),
                  'sha256': '93a59021614b48f67a0b71e66c29926e3fc302d2f4bcf6187e9f8bdf7c0d75d2'}
    require(plan.get('original_plan') == control_ref and plan.get('design') == design_ref, 'ISSUE_MIX_BOUND_REFS')
    control = _native_v3_development_read(control_ref['path'], control_ref['sha256'])
    design = _native_v3_development_read(design_ref['path'], design_ref['sha256'])
    original = control['metadata']['original']; metadata = plan.get('metadata', {})
    require(metadata.get('control') == control and metadata.get('original') == original
            and all(plan.get(k) == original.get(k) for k in ('recipe', 'model_contract', 'initial_reference'))
            and plan.get('backbone_gradient_policy') is None, 'ISSUE_MIX_UNCHANGED_FUNCTION_FACTORY')
    protocol_ref = plan.get('protocol')
    protocol_path = repo / 'configs/research' / 'NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_PROTOCOL_20261128_001.json'
    require(type(protocol_ref) is dict and set(protocol_ref) == {'path', 'sha256'}
            and protocol_ref['path'] == str(protocol_path) and type(protocol_ref['sha256']) is str
            and re.fullmatch('[0-9a-f]{64}', protocol_ref['sha256']) is not None, 'ISSUE_MIX_PROTOCOL_REF')
    protocol = _native_v3_development_read(protocol_path, protocol_ref['sha256'])
    schedule = plan.get('schedule'); trajectory = metadata.get('trajectory')
    require(protocol.get('schema_version') == 'gx1_v3_direction_issue_mix_protocol_v1'
            and protocol.get('experiment_id') == name and protocol.get('original_plan') == control_ref
            and protocol.get('design') == design_ref and protocol.get('original_schedule') == control['schedule']
            and protocol.get('recipe') == original['recipe'] and protocol.get('schedule') == schedule
            and protocol.get('original_trajectory') == trajectory and type(trajectory) is dict
            and trajectory.get('schedule') == schedule and trajectory.get('control_plan') == control_ref
            and trajectory.get('issue_mix_design') == design_ref
            and plan.get('initial_reuse') == protocol.get('initial_reuse') == control['initial_reuse']
            and plan.get('physical_input_identity') == protocol.get('physical_input_identity'), 'ISSUE_MIX_COMMON_PROTOCOL')
    scope = protocol.get('scope', {})
    require(plan.get('scope') == scope and scope.get('candidate_phase') == 'PASS2'
            and scope.get('single_changed_variable') == design['single_changed_variable']
            and scope.get('only_all5_and_original_stability_criteria') is True
            and scope.get('technical_weights_never_initializer') is True
            and scope.get('TEST_VAL_M1_GC_trade_spending') is False, 'ISSUE_MIX_ABSOLUTE_EVALUATION_SCOPE')
    ordering = design['ordering']
    geometry = {'rows': 20727, 'passes': 2, 'batch_rows': 96, 'batches_per_pass': 216,
                'logical_batches': 432, 'optimizer_updates': 432, 'skipped_batches': 0, 'seed': 20260911}
    require(type(schedule) is dict and schedule.get('schema_version') == 'gx1_native_v3_issue_mix_schedule_v1'
            and all(type(schedule.get(k)) is int and schedule[k] == v for k, v in geometry.items())
            and schedule.get('drop_last') is False and schedule.get('shuffle') is True
            and schedule.get('generator') == ordering['generator'] == 'private_numpy_Generator_PCG64'
            and type(schedule.get('update_flags')) is list and len(schedule['update_flags']) == 432
            and all(v is True for v in schedule['update_flags']), 'ISSUE_MIX_FIXED_GEOMETRY')
    sizes = schedule.get('logical_step_rows'); takes = schedule.get('cohort_takes')
    expected_sizes = [control['schedule']['original_batch_rows'][i] for i in control['schedule']['logical_original_batch_ids']]
    require(type(sizes) is list and all(type(n) is int for n in sizes)
            and sizes == expected_sizes == ordering['logical_step_rows']
            and type(takes) is list and len(takes) == 432
            and all(type(t) is list and len(t) == sizes[j] and all(type(i) is int and 0 <= i < 20727 for i in t)
                    and t == sorted(set(t)) for j, t in enumerate(takes)), 'ISSUE_MIX_TYPED_SORTED_MEMBERSHIP')
    require(all(sorted(i for t in takes[p * 216:(p + 1) * 216] for i in t) == list(range(20727)) for p in range(2)), 'ISSUE_MIX_COMPLETE_PASS_BIJECTIONS')
    hash_fields = {'raw_permutation_state_sha256_by_pass': 'raw_permutation_state_sha256_by_pass',
                   'actual_sorted_partition_stream_state_sha256_by_pass': 'actual_sorted_partition_stream_state_sha256_by_pass',
                   'cohort_take_state_sha256_by_logical_batch': 'actual_cohort_take_state_sha256_by_logical_batch'}
    require(all(schedule.get(k) == ordering[v] for k, v in hash_fields.items())
            and schedule.get('control_schedule_sha256') == ordering['control_schedule_sha256'] == control['schedule']['contract_sha256']
            and all(schedule.get(k) == original['schedule'][k] for k in ('source_rows_sha256', 'issue_ns_sha256')),
            'ISSUE_MIX_DESIGN_STREAM_BINDINGS')
    checkpoints = {'reference': [1, 2], 'cold': [2], 'fit': [1, 432]}[lane]
    require(plan.get('checkpoint_cursors') == checkpoints and all(type(v) is int for v in plan['checkpoint_cursors'])
            and plan.get('panel_indices') == [0, 107, 214] and all(type(v) is int for v in plan['panel_indices'])
            and type(plan.get('starting_cursor')) is int and plan['starting_cursor'] == (1 if lane == 'cold' else 0)
            and type(plan.get('ending_cursor')) is int and plan['ending_cursor'] == (432 if lane == 'fit' else 2), 'ISSUE_MIX_EXACT_CURSORS')
    expected_limits = {}
    for role in ('reference', 'cold', 'fit'):
        limits = dict.fromkeys(original['limits'], 0)
        updates, forwards, writes = {'reference': (2, 2, 2), 'cold': (1, 1, 1), 'fit': (432, 597, 2)}[role]
        limits.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1, target_archive_loads=1,
            model_forwards=forwards, backbone_forwards=forwards, readout_forwards=forwards, optimizer_steps=updates,
            backward_calls=updates, checkpoint_writes=writes, checkpoint_readbacks=writes,
            checkpoint_receipt_writes=writes, checkpoint_prepublication_loads=writes)
        if role == 'cold':
            limits.update(checkpoint_loads=1, model_restores=1)
        if role == 'fit':
            limits.update(autograd_calls=9, logit_archive_writes=2, logit_archive_readbacks=2)
        expected_limits[role] = limits
    limits = plan.get('limits')
    require(type(limits) is dict and all(type(v) is int for v in limits.values()) and limits == expected_limits[lane]
            and protocol.get('limits') == expected_limits
            and all(type(v) is int for values in protocol['limits'].values() for v in values.values()), 'ISSUE_MIX_EXACT_BUDGET')
    owners = plan.get('source_owners')
    require(type(owners) is dict and {'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh',
            'gx1/audit/native_provider_hlc_learning_v1.py'} <= set(owners), 'ISSUE_MIX_SOURCE_INVENTORY')
    for owner, digest in owners.items():
        p = repo / owner
        require(type(owner) is str and not Path(owner).is_absolute() and '..' not in Path(owner).parts
                and p.is_file() and not p.is_symlink() and p.resolve() == p
                and hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'ISSUE_MIX_SOURCE_SHA')
    bridge = plan.get('guard_source_bridge', {})
    old = {n: v['original_sha256'] for n, v in bridge.get('owners', {}).items()}
    require(bridge == native_v3_issue_mix_source_bridge(old, producer_commit=bridge.get('producer_commit')), 'ISSUE_MIX_SOURCE_BRIDGE')
    parents = plan.get('parents', {})
    required_parents = () if lane == 'reference' else ('reference',) if lane == 'cold' else ('reference', 'cold')
    require(type(parents) is dict and set(parents) == set(required_parents), 'ISSUE_MIX_QUALIFIED_ROLE_LINEAGE')
    for role, parent in parents.items():
        binding = parent.get('plan', {})
        require(type(binding) is dict and set(binding) == {'path', 'sha256'}
                and binding['path'] == str(repo / 'configs/research' / (name + '_' + role.upper() + '.json')), 'ISSUE_MIX_PARENT_PLAN_REF')
        previous = _native_v3_development_read(binding['path'], binding['sha256'])
        require(previous.get('lane') == role and previous.get('protocol') == protocol_ref
                and previous.get('source_commit') == plan['source_commit'] and previous.get('source_owners') == owners
                and previous.get('schedule') == schedule and previous.get('metadata', {}).get('trajectory') == trajectory,
                'ISSUE_MIX_FROZEN_ROLE_GENERATION')
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = _native_v3_development_read(policy_path)
    parent = policy.get(name.lower(), {})
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')
    require(all(policy.get(k) is (not preflight and k in flags[:2]) for k in flags), 'ISSUE_MIX_EXACT_FLAGS')
    require(parent.get('original_plan') == control_ref and parent.get('design') == design_ref
            and parent.get('protocol') == protocol_ref and all(parent.get(k) is False
                for k in ('main_fit_authorized', 'test_authorized', 'spending_authorized')), 'ISSUE_MIX_SCOPE')
    for scope in ('native_v3_bounded_development_test_20261127_001', 'native_v3_fit004_exact_recovery_20261128_001',
                  'native_v3_fit004_exact_recovery_20261128_002', _V3_BLOCK_ORDER_RECOVERY_NAME.lower()):
        prior = policy.get(scope, {})
        require(prior.get('execution_authorized') is False and prior.get('active_phase') in (None, '')
                and prior.get('active_lane') in (None, '')
                and all(type(v) is dict and v.get('execution_authorized') is False
                    and type(v.get('remaining_invocations')) is int and v['remaining_invocations'] == 0
                    for v in prior.get('lane_gates', prior.get('phase_gates', {})).values()), 'ISSUE_MIX_CONSUMED_ROUTES_CLOSED')
    for scope in (_V3_DIRECTION_GRADIENT_NAME.lower(), _V3_DIRECTION_BLOCK_ORDER_NAME.lower()):
        prior = policy.get(scope, {})
        require(prior.get('execution_authorized') is False and type(prior.get('remaining_invocations')) is int
                and prior['remaining_invocations'] == 0 and prior.get('training_authorized') is False
                and prior.get('relaunch_allowed') is False, 'ISSUE_MIX_CONSUMED_FLAT_ROUTES_CLOSED')
    lanes = parent.get('lane_gates', {})
    require(type(lanes) is dict and set(lanes) <= set(('reference', 'cold', 'fit'))
            and all(type(v) is dict and type(v.get('remaining_invocations')) is int
                and all(type(v.get(k)) is bool for k in ('execution_authorized', 'relaunch_allowed', 'training_authorized', 'test_authorized'))
                for v in lanes.values()), 'ISSUE_MIX_TYPED_LANES')
    closed = lambda v: v['execution_authorized'] is False and v['remaining_invocations'] == 0 and all(
        v[k] is False for k in ('training_authorized', 'test_authorized', 'relaunch_allowed'))
    status = 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_ISSUE_MIX_' + lane.upper()
    gate = {'status': status, 'plan': {'path': str(path), 'sha256': plan_sha256}, 'execution_authorized': not preflight,
            'remaining_invocations': int(not preflight), 'relaunch_allowed': False,
            'training_authorized': not preflight, 'test_authorized': False}
    require(lanes.get(lane) == gate and all(closed(v) for k, v in lanes.items() if k != lane)
            and all(lanes.get(role, {}).get('plan') == parents[role]['plan'] for role in required_parents), 'ISSUE_MIX_ONE_USE_ROLE_GATE')
    if preflight:
        require(parent.get('execution_authorized') is False and parent.get('active_lane') in (None, '')
                and all(closed(v) for v in lanes.values()), 'ISSUE_MIX_PREFLIGHT_CLOSED')
    else:
        require(parent.get('status') == status and parent.get('execution_authorized') is True
                and parent.get('active_lane') == lane, 'ISSUE_MIX_ONLY_ACTIVE_ROLE')
        snapshot = evidence / ('V3_DIRECTION_ISSUE_MIX_' + lane.upper() + '_ADMITTED_POLICY_20261128_001.json')
        require(_native_v3_development_read(snapshot, hashlib.sha256(policy_path.read_bytes()).hexdigest()) == policy,
                'ISSUE_MIX_EXACT_ROLE_POLICY_SNAPSHOT')
    return {'plan': {'path': str(path), 'sha256': plan_sha256}, 'output_root': str(output),
            'preflight': preflight, 'execution': plan['execution']}



def _native_v3_issue_mix_environment(plan_sha256, lane):
    require = _native_v3_development_require
    prefix = 'GX1_NATIVE_V3_ISSUE_MIX'; execution = _native_v3_issue_mix_execution(lane)
    expected = {prefix: 'true', prefix + '_PLAN_SHA256': plan_sha256, prefix + '_LANE': lane,
                'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': ''}
    require(all(not k.startswith(prefix) or k in expected for k in os.environ), 'UNKNOWN_ISSUE_MIX_ENV')
    require(all(v in ('false', '') for k, v in os.environ.items()
                if k.startswith('GX1_NATIVE_') and not k.startswith(prefix)), 'ISSUE_MIX_MIXED_LEGACY_ENV')
    fields = {'MAX_WALL_SECONDS': execution['outer_seconds'], 'MODEL_MAX_WALL_SECONDS': execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED': 'false', 'GPU_INDEX': 0, 'GPU_MAX_POWER_LIMIT_W': 250, 'GPU_MAX_POWER_DRAW_W': 250,
        'GPU_MAX_CORE_TEMP_C': 75, 'GPU_MAX_MEMORY_TEMP_C': 80, 'GPU_MAX_MEMORY_USED_MIB': 12288,
        'GPU_MONITOR_INTERVAL_SECONDS': 1, 'CPU_MAX_TEMP_C': 80, 'CPU_MAX_PACKAGE_POWER_W': 125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES': 8589934592, 'HOST_MIN_COMMIT_HEADROOM_BYTES': 8589934592}
    expected.update({'GX1_TRAINER_' + k: str(v) for k, v in fields.items()})
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'ISSUE_MIX_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()


def require_native_v3_issue_mix_execution(*, plan_sha256, lane):
    # Resource containment only; exact source-issue membership and claim remain leaf-owned.
    bound = _native_v3_issue_mix_bound_plan(plan_sha256, lane)
    proof = _native_v3_issue_mix_environment(plan_sha256, lane)
    return {**proof, **bound, 'kind': 'provider_hlc_v3_issue_mix', 'lane': lane, 'resource_proof_only': True,
            'training_authority_granted': False, 'optimizer_authority_granted': False}

# END GX1 V3 ISSUE MIX GUARDS 20261128 001

# BEGIN GX1 V3 TRAIN EVAL GAP GUARDS 20261128 001
_V3_TRAIN_EVAL_GAP_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_TRAIN_EVAL_GAP_20261128_001'
_V3_TRAIN_EVAL_GAP_PARENT = '60b81b2535aea22dc7cd5ba3a08e3a9f2258a6d3'
_V3_TRAIN_EVAL_GAP_PROFILE = 'gx1_native_provider_hlc_v3_train_eval_gap_trainer20g_v1'
_V3_TRAIN_EVAL_GAP_SHELL_ADDITIONS = (('NATIVE_V3_ISSUE_MIX=false\n', 'before', 'NATIVE_V3_TRAIN_EVAL_GAP=false\nNATIVE_V3_TRAIN_EVAL_GAP_PLAN_SHA256=\n'), ('# BEGIN GX1 V3 ISSUE MIX ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 TRAIN EVAL GAP ROUTE 20261128 001\nmatches_bound_native_v3_train_eval_gap() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_provider_hlc_learning_v1 \\\n    && "$4" == --v3-train-eval-gap-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_TRAIN_EVAL_GAP_20261128_001.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_train_eval_gap() {\n  matches_bound_native_v3_train_eval_gap "$@" || { echo \'FATAL: exact V3 train/eval gap diagnostic CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 train/eval gap diagnostic requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_ALL_TF_PROVIDER_HLC_V3_TRAIN_EVAL_GAP_20261128_001"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 train/eval gap diagnostic output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_TRAIN_EVAL_GAP=true\n  NATIVE_V3_TRAIN_EVAL_GAP_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 TRAIN EVAL GAP ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 ISSUE MIX DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 TRAIN EVAL GAP DISPATCH 20261128 001\n  local train_eval_gap_arg\n  for train_eval_gap_arg in "$@"; do\n    case "$train_eval_gap_arg" in\n      --v3-train-eval-gap-plan-sha256|--v3-train-eval-gap-plan-sha256=*) configure_native_v3_train_eval_gap "$@"; return ;;\n    esac\n  done\n# END GX1 V3 TRAIN EVAL GAP DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_ISSUE_MIX="$NATIVE_V3_ISSUE_MIX" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_TRAIN_EVAL_GAP="$NATIVE_V3_TRAIN_EVAL_GAP" \\\n  --setenv=GX1_NATIVE_V3_TRAIN_EVAL_GAP_PLAN_SHA256="$NATIVE_V3_TRAIN_EVAL_GAP_PLAN_SHA256" \\\n'))


def _native_v3_train_eval_gap_without_additions(current):
    current=_native_v3_actual_step_observer_without_additions(current)
    require=_native_v3_development_require;names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    current=dict(current);begin='# BEGIN GX1 V3 TRAIN EVAL GAP GUARDS 20261128 001\n';end='# END GX1 V3 TRAIN EVAL GAP GUARDS 20261128 001\n\n'
    text=current[names[0]];require(text.count(begin)==text.count(end)==1,'ONE_TRAIN_EVAL_GAP_BLOCK')
    first,tail=text.split(begin);_,last=tail.split(end);text=first+last
    hook='    current=_native_v3_train_eval_gap_without_additions(current)\n'
    require(text.count(hook)==1,'ONE_TRAIN_EVAL_GAP_BRIDGE_HOOK')
    current[names[0]]=text.replace(hook,'',1)
    shell=current[names[1]]
    for anchor,position,fragment in _V3_TRAIN_EVAL_GAP_SHELL_ADDITIONS:
        require(position=='before'and shell.count(anchor)==shell.count(fragment)==shell.count(fragment+anchor)==1,'EXACT_TRAIN_EVAL_GAP_SHELL_ADDITION')
        shell=shell.replace(fragment+anchor,anchor,1)
    current[names[1]]=shell
    return current

def native_v3_train_eval_gap_source_bridge(producer_hashes,*,producer_commit=_V3_TRAIN_EVAL_GAP_PARENT):
    import hashlib,subprocess
    require=_native_v3_development_require;require(producer_commit==_V3_TRAIN_EVAL_GAP_PARENT,'TRAIN_EVAL_GAP_BRIDGE_PARENT')
    repo=Path(__file__).resolve().parents[2];names=('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh')
    before={n:subprocess.check_output(['git','show',producer_commit+':'+n],cwd=repo).decode()for n in names}
    original={n:hashlib.sha256(v.encode()).hexdigest()for n,v in before.items()};require(producer_hashes==original,'TRAIN_EVAL_GAP_BRIDGE_PRODUCER_HASHES')
    current={n:(repo/n).read_text()for n in names}
    require(_native_v3_train_eval_gap_without_additions(current)==before,'TRAIN_EVAL_GAP_UNCHANGED_PREDECESSOR')
    return {'producer_commit':producer_commit,'checked_guard_owners':2,'old_guard_routes_preserved':True,
        'old_numeric_resource_environment_function_unchanged':True,'new_diagnostic_controlflow_requires_scoped_review':True,
        'owners':{n:{'original_sha256':original[n],'current_sha256':hashlib.sha256(current[n].encode()).hexdigest()}for n in names}}

def _native_v3_train_eval_gap_bound_plan(plan_sha256):
    import hashlib, subprocess
    require = _native_v3_development_require
    require(type(plan_sha256) is str and re.fullmatch('[0-9a-f]{64}', plan_sha256) is not None, 'TRAIN_EVAL_GAP_PLAN_SHA')
    repo = Path(__file__).resolve().parents[2]; name = _V3_TRAIN_EVAL_GAP_NAME
    args = ['--v3-train-eval-gap-plan-sha256', plan_sha256]; preflight = sys.argv[-1:] == ['--preflight']
    if preflight: args += ['--preflight']
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / 'gx1/audit/native_provider_hlc_learning_v1.py'
            and Path(sys.executable).absolute() == repo / '.venv/bin/python' and sys.argv[1:] == args, 'TRAIN_EVAL_GAP_EXACT_ARGV')
    path = repo / 'configs/research' / (name + '.json'); plan = _native_v3_development_read(path, plan_sha256)
    output = Path('/home/andre2/GX1_RUNS') / name
    require(plan.get('schema_version') == 'gx1_native_v3_train_eval_gap_plan_v1'
            and plan.get('stage') == name.lower() and plan.get('output_root') == str(output), 'TRAIN_EVAL_GAP_PLAN_IDENTITY')
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent.resolve() == output.parent, 'TRAIN_EVAL_GAP_FRESH_OUTPUT')
    require(plan.get('source_commit') == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'TRAIN_EVAL_GAP_SOURCE_COMMIT')
    require(plan.get('execution') == _native_v3_development_execution('warm')
            and plan.get('execution_profile') == _V3_TRAIN_EVAL_GAP_PROFILE, 'TRAIN_EVAL_GAP_EXACT_EXECUTION')
    evidence = Path('/home/andre2/GX1_RUNS') / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = {'path': str(evidence / 'V3_TRAIN_EVAL_GAP_DESIGN_20261128_003.json'),
                  'sha256': 'eb0f66e2e1454f3f0f56e10016be039d2cef56f421556a2363f5db461f34a000'}
    original_ref = {'path': str(repo / 'configs/research' / (_V3_ISSUE_MIX_NAME + '_FIT.json')),
                    'sha256': '25ca4cc9d780f5d16376a765ec58f6d84c58a1f16f55d48033d11168fda2d4a8'}
    final_root = Path('/home/andre2/GX1_RUNS') / (_V3_ISSUE_MIX_NAME + '_fit')
    receipt_ref = {'path': str(final_root / 'FINAL.json'), 'sha256': '1b67d9e51e211480c168b0eae05e5c1677e804e4e4d58e7b9bb1fb619271884d'}
    require(plan.get('original_plan') == original_ref and plan.get('checkpoint_receipt') == receipt_ref
            and plan.get('design') == design_ref, 'TRAIN_EVAL_GAP_BOUND_REFS')
    design = _native_v3_development_read(design_ref['path'], design_ref['sha256'])
    producer = _native_v3_development_read(original_ref['path'], original_ref['sha256'])
    receipt = _native_v3_development_read(receipt_ref['path'], receipt_ref['sha256'])
    original = producer.get('metadata', {}).get('original')
    require(type(original) is dict and type(receipt.get('logical_cursor')) is int
            and type(receipt.get('optimizer_updates')) is int and receipt['logical_cursor'] == receipt['optimizer_updates'] == 432
            and receipt.get('producer_lane') == 'fit' and receipt.get('original_plan') == receipt.get('execution_plan') == original_ref
            and receipt.get('checkpoint') == design.get('frozen_FINAL')
            and design.get('producer_fit_plan') == original_ref and design.get('frozen_FINAL_receipt') == receipt_ref
            and plan.get('metadata', {}).get('original') == original and plan.get('metadata', {}).get('producer') == producer
            and plan.get('metadata', {}).get('receipt') == receipt
            and plan.get('metadata', {}).get('trajectory') == producer.get('metadata', {}).get('trajectory')
            and plan.get('model_contract') == producer.get('model_contract') and plan.get('recipe') == producer.get('recipe')
            and plan.get('physical_input_identity') == producer.get('physical_input_identity'), 'TRAIN_EVAL_GAP_FINAL_ONLY')
    inputs = plan.get('inputs', {})
    require(all(inputs.get(k) == value for k, value in {
        'design': design_ref, 'final_checkpoint_receipt': receipt_ref, 'final_checkpoint': receipt['checkpoint'],
        'saved_eval_logits': design.get('saved_eval_logits'), 'saved_eval_score': design.get('saved_eval_score')}.items()), 'TRAIN_EVAL_GAP_INPUT_REFS')
    expected = dict.fromkeys(producer['limits'], 0)
    expected.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
        target_archive_loads=1, checkpoint_loads=1, model_restores=1, model_forwards=405,
        backbone_forwards=405, readout_forwards=405, logit_archive_writes=5, logit_archive_readbacks=5,
        saved_eval_logit_loads=1)
    limits = plan.get('limits')
    require(type(limits) is dict and all(type(v) is int for v in limits.values()) and limits == expected, 'TRAIN_EVAL_GAP_NO_UPDATE_BUDGET')
    passes = [{'label': 'EVAL_PARITY', 'training': False, 'equal_tf': True}] + [
        {'label': 'TRAIN_DRAW_' + str(i).zfill(2), 'training': True, 'equal_tf': False} for i in range(1, 5)]
    require(plan.get('passes') == passes and all(type(v.get(k)) is bool for v in plan['passes']
            for k in ('training', 'equal_tf')), 'TRAIN_EVAL_GAP_FIXED_PASSES')
    owners = plan.get('source_owners')
    require(type(owners) is dict and {'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh',
            'gx1/audit/native_provider_hlc_learning_v1.py'} <= set(owners), 'TRAIN_EVAL_GAP_SOURCE_INVENTORY')
    for owner, digest in owners.items():
        p = repo / owner
        require(type(owner) is str and not Path(owner).is_absolute() and '..' not in Path(owner).parts
                and p.is_file() and not p.is_symlink() and p.resolve() == p
                and hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'TRAIN_EVAL_GAP_SOURCE_SHA')
    bridge = plan.get('guard_source_bridge', {})
    old = {n: v['original_sha256'] for n, v in bridge.get('owners', {}).items()}
    require(bridge == native_v3_train_eval_gap_source_bridge(old, producer_commit=bridge.get('producer_commit')), 'TRAIN_EVAL_GAP_SOURCE_BRIDGE')
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = _native_v3_development_read(policy_path)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')
    require(all(policy.get(k) is False for k in flags), 'TRAIN_EVAL_GAP_ALL_TRAINING_CLOSED')
    for scope in ('native_v3_bounded_development_test_20261127_001', 'native_v3_fit004_exact_recovery_20261128_001',
                  'native_v3_fit004_exact_recovery_20261128_002', _V3_BLOCK_ORDER_RECOVERY_NAME.lower(), _V3_ISSUE_MIX_NAME.lower()):
        prior = policy.get(scope, {})
        gates = prior.get('lane_gates', prior.get('phase_gates', {}))
        require(prior.get('execution_authorized') is False and prior.get('active_phase') in (None, '')
                and prior.get('active_lane') in (None, '') and type(gates) is dict
                and all(type(v) is dict and v.get('execution_authorized') is False
                    and type(v.get('remaining_invocations')) is int and v['remaining_invocations'] == 0
                    and all(v.get(k) is False for k in ('training_authorized', 'test_authorized', 'relaunch_allowed'))
                    for v in gates.values()), 'TRAIN_EVAL_GAP_CONSUMED_ROUTES_CLOSED')
    for scope in (_V3_DIRECTION_GRADIENT_NAME.lower(), _V3_DIRECTION_BLOCK_ORDER_NAME.lower()):
        prior = policy.get(scope, {})
        require(prior.get('execution_authorized') is False and type(prior.get('remaining_invocations')) is int
                and prior['remaining_invocations'] == 0 and prior.get('training_authorized') is False
                and prior.get('relaunch_allowed') is False, 'TRAIN_EVAL_GAP_CONSUMED_FLAT_ROUTES_CLOSED')
    parent = policy.get(name.lower(), {})
    gate = {'status': 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_TRAIN_EVAL_GAP',
            'plan': {'path': str(path), 'sha256': plan_sha256}, 'execution_authorized': not preflight,
            'remaining_invocations': int(not preflight), 'relaunch_allowed': False,
            'training_authorized': False, 'test_authorized': False, 'spending_authorized': False}
    require(type(parent) is dict and type(parent.get('remaining_invocations')) is int
            and all(type(parent.get(k)) is bool for k in ('execution_authorized', 'relaunch_allowed',
                'training_authorized', 'test_authorized', 'spending_authorized'))
            and all(parent.get(k) == v for k, v in gate.items()) and parent.get('design') == design_ref,
            'TRAIN_EVAL_GAP_ONE_USE_GATE')
    if not preflight:
        snapshot = evidence / 'V3_TRAIN_EVAL_GAP_ADMITTED_POLICY_20261128_001.json'
        require(_native_v3_development_read(snapshot, hashlib.sha256(policy_path.read_bytes()).hexdigest()) == policy,
                'TRAIN_EVAL_GAP_EXACT_POLICY_SNAPSHOT')
    return {'plan': {'path': str(path), 'sha256': plan_sha256}, 'output_root': str(output),
            'preflight': preflight, 'execution': plan['execution']}


def _native_v3_train_eval_gap_environment(plan_sha256):
    require=_native_v3_development_require;prefix='GX1_NATIVE_V3_TRAIN_EVAL_GAP';execution=_native_v3_development_execution('warm')
    expected={prefix:'true',prefix+'_PLAN_SHA256':plan_sha256,
        'GX1_POWER_BENCHMARK_SCOPE_JSON':'','GX1_POWER_BENCHMARK_SCOPE_SHA256':''}
    require(all(not k.startswith(prefix)or k in expected for k in os.environ),'UNKNOWN_TRAIN_EVAL_GAP_ENV')
    require(all(v in('false','')for k,v in os.environ.items()if k.startswith('GX1_NATIVE_')and not k.startswith(prefix)),'TRAIN_EVAL_GAP_MIXED_LEGACY_ENV')
    fields={'MAX_WALL_SECONDS':execution['outer_seconds'],'MODEL_MAX_WALL_SECONDS':execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED':'false','GPU_INDEX':0,'GPU_MAX_POWER_LIMIT_W':250,'GPU_MAX_POWER_DRAW_W':250,
        'GPU_MAX_CORE_TEMP_C':75,'GPU_MAX_MEMORY_TEMP_C':80,'GPU_MAX_MEMORY_USED_MIB':12288,
        'GPU_MONITOR_INTERVAL_SECONDS':1,'CPU_MAX_TEMP_C':80,'CPU_MAX_PACKAGE_POWER_W':125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES':8589934592,'HOST_MIN_COMMIT_HEADROOM_BYTES':8589934592}
    expected.update({'GX1_TRAINER_'+k:str(v)for k,v in fields.items()})
    require(all(os.environ.get(k)==v for k,v in expected.items()),'TRAIN_EVAL_GAP_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()

def require_native_v3_train_eval_gap_execution(*,plan_sha256):
    # Resource containment only; no optimizer or training authority is created.
    bound=_native_v3_train_eval_gap_bound_plan(plan_sha256)
    proof=_native_v3_train_eval_gap_environment(plan_sha256)
    return {**proof,**bound,'kind':'provider_hlc_v3_train_eval_gap','resource_proof_only':True,
        'training_authority_granted':False,'optimizer_authority_granted':False}

# END GX1 V3 TRAIN EVAL GAP GUARDS 20261128 001

# BEGIN GX1 V3 ACTUAL STEP OBSERVER GUARDS 20261128 001
_V3_ACTUAL_STEP_OBSERVER_PARENT = '25cab1bf29f241275b2624110d736e85fcd3bc0e'
_V3_ACTUAL_STEP_OBSERVER_NAME = 'NATIVE_V3_ACTUAL_STEP_OBSERVER_QUALIFICATION_20261128_001'
_V3_ACTUAL_STEP_OBSERVER_STAGE = 'native_v3_actual_step_observer_20261128_001'
_V3_ACTUAL_STEP_OBSERVER_PROFILE = 'gx1_native_actual_step_observer_qualification_trainer20g_v1'
_V3_ACTUAL_STEP_OBSERVER_SHELL_ADDITIONS = (('NATIVE_V3_TRAIN_EVAL_GAP=false\n', 'before', 'NATIVE_V3_ACTUAL_STEP_OBSERVER=false\nNATIVE_V3_ACTUAL_STEP_OBSERVER_PLAN_SHA256=\nNATIVE_V3_ACTUAL_STEP_OBSERVER_LANE=\n'), ('# BEGIN GX1 V3 TRAIN EVAL GAP ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 ACTUAL STEP OBSERVER ROUTE 20261128 001\nmatches_bound_native_v3_actual_step_observer() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_future_state_step_observer_qualify_v1 \\\n    && "$4" == --v3-actual-step-observer-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local lane expected_plan matches=0 matched_lane=\n  for lane in reference cold; do\n    expected_plan="$REPO_ROOT/configs/research/NATIVE_V3_ACTUAL_STEP_OBSERVER_QUALIFICATION_20261128_001_${lane^^}.json"\n    if [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n      && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n      && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]; then\n      matches=$((matches+1)); matched_lane="$lane"\n    fi\n  done\n  [[ "$matches" == 1 ]] || return 1\n  NATIVE_V3_ACTUAL_STEP_OBSERVER_LANE="$matched_lane"\n}\nconfigure_native_v3_actual_step_observer() {\n  matches_bound_native_v3_actual_step_observer "$@" || { echo \'FATAL: exact V3 actual-step observer qualification CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 actual-step observer qualification requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_V3_ACTUAL_STEP_OBSERVER_QUALIFICATION_20261128_001_$NATIVE_V3_ACTUAL_STEP_OBSERVER_LANE"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 actual-step observer qualification output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_ACTUAL_STEP_OBSERVER=true\n  NATIVE_V3_ACTUAL_STEP_OBSERVER_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 ACTUAL STEP OBSERVER ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 TRAIN EVAL GAP DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 ACTUAL STEP OBSERVER DISPATCH 20261128 001\n  local actual_step_observer_arg\n  for actual_step_observer_arg in "$@"; do\n    case "$actual_step_observer_arg" in\n      --v3-actual-step-observer-plan-sha256|--v3-actual-step-observer-plan-sha256=*) configure_native_v3_actual_step_observer "$@"; return ;;\n    esac\n  done\n# END GX1 V3 ACTUAL STEP OBSERVER DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_TRAIN_EVAL_GAP="$NATIVE_V3_TRAIN_EVAL_GAP" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER="$NATIVE_V3_ACTUAL_STEP_OBSERVER" \\\n  --setenv=GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_PLAN_SHA256="$NATIVE_V3_ACTUAL_STEP_OBSERVER_PLAN_SHA256" \\\n  --setenv=GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_LANE="$NATIVE_V3_ACTUAL_STEP_OBSERVER_LANE" \\\n'))
_V3_ACTUAL_STEP_OBSERVER_CORE = 'gx1/audit/qualify_native_m5_full_gc_direction_v1.py'
_V3_ACTUAL_STEP_OBSERVER_CORE_SHA256 = 'c80fe2bec2febfa8e990adfc5f7f57213a42b1f58fb3a6de4c7fda5a7ada713d'
_V3_ACTUAL_STEP_OBSERVER_CORE_SIGNATURE = ('def future_cuda_step(model,optimizer,packet,plan,counts,index):\n',
    'def future_cuda_step(model,optimizer,packet,plan,counts,index,*,observer=None):\n')
_V3_ACTUAL_STEP_OBSERVER_CORE_ADDITIONS = (
    ("    need(not loss['skip_update']and bool(torch.isfinite(loss['loss'])),'FUTURE_CUDA_TRAIN_SUPPORT')\n",
     'FORWARD_LOSS', "    if observer is not None:observer('forward_loss',model=model,optimizer=optimizer,output=output,truth=packet['truth'],loss=loss,plan=plan,index=index)\n"),
    ("    loss['loss'].backward();counts['backward_calls']+=1\n",
     'BACKWARD', "    if observer is not None:observer('backward',model=model,optimizer=optimizer,index=index)\n"),
    ("    clipped=torch.nn.utils.clip_grad_norm_(gradients,plan['recipe']['gradient_clip_norm'],error_if_nonfinite=True)\n",
     'CLIP', "    if observer is not None:observer('clip',model=model,optimizer=optimizer,global_norm=clipped,index=index)\n"),
    ("    optimizer.step();counts['optimizer_steps']+=1\n",
     'STEP', "    if observer is not None:observer('step',model=model,optimizer=optimizer,index=index)\n"))


def _native_v3_actual_step_observer_core_projection(current):
    require = _native_v3_development_require
    old_signature, new_signature = _V3_ACTUAL_STEP_OBSERVER_CORE_SIGNATURE
    require(type(current) is str and current.count(new_signature) == 1
            and current.count(old_signature) == 0, 'ACTUAL_STEP_OBSERVER_EXACT_OPTIONAL_SIGNATURE')
    projected = current.replace(new_signature, old_signature, 1)
    for anchor, label, call in _V3_ACTUAL_STEP_OBSERVER_CORE_ADDITIONS:
        fragment = '    # BEGIN GX1 V3 ACTUAL STEP OBSERVER ' + label + ' 20261128\n' + call + \
            '    # END GX1 V3 ACTUAL STEP OBSERVER ' + label + ' 20261128\n'
        require(projected.count(fragment) == 1 and projected.count(anchor + fragment) == 1,
                'ACTUAL_STEP_OBSERVER_EXACT_ADDITIVE_PHASE_' + label)
        projected = projected.replace(anchor + fragment, anchor, 1)
    return projected


def _native_v3_actual_step_observer_without_additions(current):
    current=_native_v3_actual_step_observer_cold_recovery_without_additions(current)
    require = _native_v3_development_require; current = dict(current)
    name = 'gx1/contracts/gx1_capped_execution_v1.py'
    begin = '# BEGIN GX1 V3 ACTUAL STEP OBSERVER GUARDS 20261128 001\n'
    end = '# END GX1 V3 ACTUAL STEP OBSERVER GUARDS 20261128 001\n\n'
    text = current[name]
    require(text.count(begin) == text.count(end) == 1, 'ONE_ACTUAL_STEP_OBSERVER_GUARD_BLOCK')
    first, tail = text.split(begin); _, last = tail.split(end); text = first + last
    hook = '    current=_native_v3_actual_step_observer_without_additions(current)\n'
    require(text.count(hook) == 1, 'ONE_ACTUAL_STEP_OBSERVER_GUARD_BRIDGE_HOOK')
    current[name] = text.replace(hook, '', 1)
    shell = current['scripts/gx1_capped_run.sh']
    for anchor, position, fragment in _V3_ACTUAL_STEP_OBSERVER_SHELL_ADDITIONS:
        require(position == 'before' and shell.count(anchor) == shell.count(fragment)
                == shell.count(fragment + anchor) == 1, 'EXACT_ACTUAL_STEP_OBSERVER_SHELL_ADDITION')
        shell = shell.replace(fragment + anchor, anchor, 1)
    current['scripts/gx1_capped_run.sh'] = shell
    return current


def native_v3_actual_step_observer_source_bridge(producer_hashes, *, producer_commit=_V3_ACTUAL_STEP_OBSERVER_PARENT):
    """Prove exactly four additive observations and the observer-off core projection."""
    import ast, hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(producer_commit == _V3_ACTUAL_STEP_OBSERVER_PARENT, 'ACTUAL_STEP_OBSERVER_BRIDGE_PARENT')
    name = _V3_ACTUAL_STEP_OBSERVER_CORE
    original = subprocess.check_output(['git', 'show', producer_commit + ':' + name], cwd=repo)
    digest = hashlib.sha256(original).hexdigest()
    require(digest == _V3_ACTUAL_STEP_OBSERVER_CORE_SHA256 and producer_hashes == {name: digest},
            'ACTUAL_STEP_OBSERVER_AUTHENTIC_CORE_HASH')
    path = repo / name
    require(path.is_file() and not path.is_symlink() and path.resolve() == path, 'ACTUAL_STEP_OBSERVER_CANONICAL_CORE')
    current = path.read_bytes(); projected = _native_v3_actual_step_observer_core_projection(current.decode())
    require(projected.encode() == original, 'ACTUAL_STEP_OBSERVER_UNCHANGED_CORE_BYTES')
    require(ast.dump(ast.parse(projected), include_attributes=False) == ast.dump(ast.parse(original), include_attributes=False),
            'ACTUAL_STEP_OBSERVER_UNCHANGED_CORE_AST')
    return {'producer_commit': producer_commit, 'owner': name, 'original_sha256': digest,
            'current_sha256': hashlib.sha256(current).hexdigest(), 'additive_observer_phases': 4,
            'observer_off_projection_byte_exact': True, 'observer_off_projection_AST_exact': True,
            'old_numeric_step_unchanged': True, 'new_observer_requires_separate_qualification': True,
            'training_authority_granted': False, 'execution_authority_granted': False}


def native_v3_actual_step_observer_guard_source_bridge(producer_hashes, *, producer_commit=_V3_ACTUAL_STEP_OBSERVER_PARENT):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(producer_commit == _V3_ACTUAL_STEP_OBSERVER_PARENT, 'ACTUAL_STEP_OBSERVER_GUARD_PARENT')
    names = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')
    before = {n: subprocess.check_output(['git', 'show', producer_commit + ':' + n], cwd=repo).decode() for n in names}
    original = {n: hashlib.sha256(v.encode()).hexdigest() for n, v in before.items()}
    require(producer_hashes == original, 'ACTUAL_STEP_OBSERVER_GUARD_PRODUCER_HASHES')
    current = {n: (repo / n).read_text() for n in names}
    require(_native_v3_actual_step_observer_without_additions(current) == before, 'ACTUAL_STEP_OBSERVER_UNCHANGED_PREDECESSOR_GUARDS')
    return {'producer_commit': producer_commit, 'checked_guard_owners': 2, 'old_guard_routes_preserved': True,
            'old_numeric_resource_environment_function_unchanged': True, 'new_observer_controlflow_requires_scoped_review': True,
            'owners': {n: {'original_sha256': original[n], 'current_sha256': hashlib.sha256(current[n].encode()).hexdigest()} for n in names}}


def _native_v3_actual_step_observer_bound_plan(plan_sha256):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(type(plan_sha256) is str and re.fullmatch('[0-9a-f]{64}', plan_sha256) is not None, 'ACTUAL_STEP_OBSERVER_PLAN_SHA')
    args = ['--v3-actual-step-observer-plan-sha256', plan_sha256]; preflight = sys.argv[-1:] == ['--preflight']
    if preflight: args += ['--preflight']
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / 'gx1/audit/native_future_state_step_observer_qualify_v1.py'
            and Path(sys.executable).absolute() == repo / '.venv/bin/python' and sys.argv[1:] == args, 'ACTUAL_STEP_OBSERVER_EXACT_ARGV')
    candidates = [(lane, repo / 'configs/research' / (_V3_ACTUAL_STEP_OBSERVER_NAME + '_' + lane.upper() + '.json'))
                  for lane in ('reference', 'cold')]
    matches = [(lane, path) for lane, path in candidates if path.is_file() and not path.is_symlink()
               and path.resolve() == path and hashlib.sha256(path.read_bytes()).hexdigest() == plan_sha256]
    require(len(matches) == 1, 'ACTUAL_STEP_OBSERVER_ONE_CANONICAL_ROLE_PLAN')
    lane, path = matches[0]; plan = _native_v3_development_read(path, plan_sha256)
    output = Path('/home/andre2/GX1_RUNS') / (_V3_ACTUAL_STEP_OBSERVER_NAME + '_' + lane)
    require(plan.get('schema_version') == 'gx1_native_v3_actual_step_observer_qualification_plan_v1'
            and plan.get('stage') == _V3_ACTUAL_STEP_OBSERVER_STAGE and plan.get('lane') == lane
            and plan.get('output_root') == str(output), 'ACTUAL_STEP_OBSERVER_PLAN_IDENTITY')
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent.resolve() == output.parent, 'ACTUAL_STEP_OBSERVER_FRESH_OUTPUT')
    require(plan.get('source_commit') == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'ACTUAL_STEP_OBSERVER_SOURCE_COMMIT')
    require(plan.get('execution') == _native_v3_development_execution('warm')
            and plan.get('execution_profile') == _V3_ACTUAL_STEP_OBSERVER_PROFILE, 'ACTUAL_STEP_OBSERVER_EXACT_EXECUTION')
    evidence = Path('/home/andre2/GX1_RUNS') / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = {'path': str(evidence / 'V3_ACTUAL_STEP_OBSERVER_DESIGN_20261128_001.json'),
                  'sha256': 'faf07246d9fd7fdf92811a10668dc0431772efe90f086b2f967ecd2db23a3297'}
    producer_ref = {'path': str(repo / 'configs/research' / (_V3_ISSUE_MIX_NAME + '_FIT.json')),
                    'sha256': '25ca4cc9d780f5d16376a765ec58f6d84c58a1f16f55d48033d11168fda2d4a8'}
    final_root = Path('/home/andre2/GX1_RUNS') / (_V3_ISSUE_MIX_NAME + '_fit')
    receipt_ref = {'path': str(final_root / 'FINAL.json'), 'sha256': '1b67d9e51e211480c168b0eae05e5c1677e804e4e4d58e7b9bb1fb619271884d'}
    checkpoint_ref = {'path': str(final_root / 'FINAL.pt'), 'sha256': '577b5b37f5c4cfa98744cfeefb06e8156231488a7adc038f33cdfd685df86632'}
    require(plan.get('design') == design_ref and plan.get('producer_plan') == producer_ref
            and plan.get('authentic_final_receipt') == receipt_ref
            and plan.get('authentic_final_checkpoint') == checkpoint_ref, 'ACTUAL_STEP_OBSERVER_AUTHENTIC_FINAL_REFS')
    _native_v3_development_read(design_ref['path'], design_ref['sha256'])
    producer = _native_v3_development_read(producer_ref['path'], producer_ref['sha256'])
    receipt = _native_v3_development_read(receipt_ref['path'], receipt_ref['sha256'])
    require(type(receipt.get('logical_cursor')) is int and type(receipt.get('optimizer_updates')) is int
            and receipt['logical_cursor'] == receipt['optimizer_updates'] == 432 and receipt.get('producer_lane') == 'fit'
            and receipt.get('original_plan') == receipt.get('execution_plan') == producer_ref and receipt.get('checkpoint') == checkpoint_ref
            and plan.get('metadata', {}).get('producer') == producer
            and plan.get('metadata', {}).get('original') == producer.get('metadata', {}).get('original')
            and plan.get('model_contract') == producer.get('model_contract') and plan.get('recipe') == producer.get('recipe'),
            'ACTUAL_STEP_OBSERVER_FINAL432_IDENTITY')
    require(type(plan.get('starting_cursor')) is int and type(plan.get('ending_cursor')) is int
            and plan['starting_cursor'] == (432 if lane == 'reference' else 433) and plan['ending_cursor'] == 434
            and plan.get('checkpoint_cursors') == ([433, 434] if lane == 'reference' else [434])
            and all(type(v) is int for v in plan['checkpoint_cursors'])
            and plan.get('observer_enabled') is (lane == 'cold')
            and plan.get('technical_weights_never_initializer') is True, 'ACTUAL_STEP_OBSERVER_BOUNDED_TECHNICAL_ROLE')
    count = 2 if lane == 'reference' else 1
    limits = dict.fromkeys(producer['limits'], 0)
    limits.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
        target_archive_loads=1, checkpoint_loads=1, model_restores=1, model_forwards=count,
        backbone_forwards=count, readout_forwards=count, backward_calls=count, optimizer_steps=count,
        checkpoint_writes=count, checkpoint_readbacks=count, checkpoint_prepublication_loads=count,
        checkpoint_receipt_writes=count, observer_records=0 if lane == 'reference' else 1)
    require(type(plan.get('limits')) is dict and all(type(v) is int for v in plan['limits'].values())
            and plan['limits'] == limits, 'ACTUAL_STEP_OBSERVER_EXACT_TECHNICAL_BUDGET')
    owners = plan.get('source_owners')
    require(type(owners) is dict and set(producer['source_owners']) <= set(owners)
            and {'gx1/audit/native_future_state_step_observer_qualify_v1.py', 'gx1/audit/native_future_state_step_observer_v1.py'} <= set(owners),
            'ACTUAL_STEP_OBSERVER_SOURCE_INVENTORY')
    for owner, digest in owners.items():
        p = repo / owner
        require(type(owner) is str and not Path(owner).is_absolute() and '..' not in Path(owner).parts
                and p.is_file() and not p.is_symlink() and p.resolve() == p
                and hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'ACTUAL_STEP_OBSERVER_SOURCE_SHA')
    core_bridge = plan.get('core_source_bridge')
    require(core_bridge == native_v3_actual_step_observer_source_bridge(
        {_V3_ACTUAL_STEP_OBSERVER_CORE: producer['source_owners'][_V3_ACTUAL_STEP_OBSERVER_CORE]}), 'ACTUAL_STEP_OBSERVER_CORE_SOURCE_BRIDGE')
    bridge = plan.get('guard_source_bridge', {}); old = {n: v['original_sha256'] for n, v in bridge.get('owners', {}).items()}
    require(bridge == native_v3_actual_step_observer_guard_source_bridge(old, producer_commit=bridge.get('producer_commit')),
            'ACTUAL_STEP_OBSERVER_GUARD_SOURCE_BRIDGE')
    parents = plan.get('parents', {}); expected_roles = () if lane == 'reference' else ('reference',)
    require(type(parents) is dict and set(parents) == set(expected_roles), 'ACTUAL_STEP_OBSERVER_EXACT_PARENT_ROLES')
    for role, parent in parents.items():
        previous_ref = parent.get('plan', {})
        require(type(previous_ref) is dict and set(previous_ref) == {'path', 'sha256'}
                and previous_ref['path'] == str(repo / 'configs/research' / (_V3_ACTUAL_STEP_OBSERVER_NAME + '_REFERENCE.json')),
                'ACTUAL_STEP_OBSERVER_REFERENCE_PARENT_PLAN')
        previous = _native_v3_development_read(previous_ref['path'], previous_ref['sha256'])
        require(previous.get('lane') == 'reference' and previous.get('source_commit') == plan['source_commit']
                and previous.get('source_owners') == owners and previous.get('protocol') == plan.get('protocol')
                and previous.get('metadata', {}).get('trajectory') == plan.get('metadata', {}).get('trajectory'),
                'ACTUAL_STEP_OBSERVER_SAME_FROZEN_QUALIFICATION_GENERATION')
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = _native_v3_development_read(policy_path)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')
    require(all(policy.get(k) is (not preflight and k in flags[:2]) for k in flags), 'ACTUAL_STEP_OBSERVER_EXACT_FLAGS')
    for scope in ('native_v3_bounded_development_test_20261127_001', 'native_v3_fit004_exact_recovery_20261128_001',
                  'native_v3_fit004_exact_recovery_20261128_002', _V3_BLOCK_ORDER_RECOVERY_NAME.lower(), _V3_ISSUE_MIX_NAME.lower()):
        prior = policy.get(scope, {}); gates = prior.get('lane_gates', prior.get('phase_gates', {}))
        require(prior.get('execution_authorized') is False and prior.get('active_phase') in (None, '')
                and prior.get('active_lane') in (None, '') and type(gates) is dict
                and all(type(v) is dict and v.get('execution_authorized') is False
                    and type(v.get('remaining_invocations')) is int and v['remaining_invocations'] == 0
                    and all(v.get(k) is False for k in ('training_authorized', 'test_authorized', 'relaunch_allowed'))
                    for v in gates.values()), 'ACTUAL_STEP_OBSERVER_CONSUMED_ROUTES_CLOSED')
    old_scopes = [policy.get(k, {}) for k in (_V3_DIRECTION_GRADIENT_NAME.lower(), _V3_DIRECTION_BLOCK_ORDER_NAME.lower(),
        _V3_TRAIN_EVAL_GAP_NAME.lower(), 'native_v3_frozen_batch_residual_audit_20261128_001', 'native_v3_frozen_batch_residual_audit_20261128_002')]
    old_scopes.append(policy.get(_V3_TRAIN_EVAL_GAP_NAME.lower(), {}).get('aggregation_completion', {}))
    require(all(v.get('execution_authorized') is False and type(v.get('remaining_invocations')) is int
                and v['remaining_invocations'] == 0 and v.get('training_authorized') is False
                and v.get('relaunch_allowed') is False for v in old_scopes), 'ACTUAL_STEP_OBSERVER_CONSUMED_FLAT_ROUTES_CLOSED')
    parent = policy.get(_V3_ACTUAL_STEP_OBSERVER_STAGE, {})
    protocol_ref = {'path': str(evidence / 'V3_ACTUAL_STEP_OBSERVER_QUALIFICATION_PROTOCOL_20261128_001.json'),
                    'sha256': 'c225d94342af1bf4b787373f1f3cdc5a5a4a08a09464e5c53ed06571492aae40'}
    require(parent.get('source_preparation_authorized') is True and parent.get('design') == design_ref
            and plan.get('protocol') == protocol_ref and parent.get('protocol') == protocol_ref
            and parent.get('producer_plan') == producer_ref and parent.get('authentic_final_receipt') == receipt_ref
            and parent.get('authentic_final_checkpoint') == checkpoint_ref, 'ACTUAL_STEP_OBSERVER_POLICY_SCOPE_PROTOCOL')
    protocol = _native_v3_development_read(protocol_ref['path'], protocol_ref['sha256'])
    require(protocol.get('schema_version') == 'gx1_v3_actual_step_observer_qualification_protocol_v1',
            'ACTUAL_STEP_OBSERVER_PROTOCOL_SCHEMA')
    lanes = parent.get('lane_gates', {})
    require(type(lanes) is dict and set(lanes) <= {'reference', 'cold'}
            and all(type(v) is dict and type(v.get('remaining_invocations')) is int
                and all(type(v.get(k)) is bool for k in ('execution_authorized', 'relaunch_allowed', 'training_authorized', 'test_authorized'))
                for v in lanes.values()), 'ACTUAL_STEP_OBSERVER_TYPED_LANES')
    closed = lambda v: v['execution_authorized'] is False and v['remaining_invocations'] == 0 and all(
        v[k] is False for k in ('training_authorized', 'test_authorized', 'relaunch_allowed'))
    status = 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_ACTUAL_STEP_OBSERVER_' + lane.upper()
    gate = {'status': status, 'plan': {'path': str(path), 'sha256': plan_sha256}, 'execution_authorized': not preflight,
            'remaining_invocations': int(not preflight), 'relaunch_allowed': False,
            'training_authorized': not preflight, 'test_authorized': False}
    require(lanes.get(lane) == gate and all(closed(v) for k, v in lanes.items() if k != lane)
            and all(lanes.get(role, {}).get('plan') == parents[role]['plan'] for role in expected_roles),
            'ACTUAL_STEP_OBSERVER_ONE_USE_ROLE_GATE')
    if preflight:
        require(parent.get('execution_authorized') is False and parent.get('active_lane') in (None, '')
                and all(closed(v) for v in lanes.values()), 'ACTUAL_STEP_OBSERVER_PREFLIGHT_CLOSED')
    else:
        require(parent.get('status') == status and parent.get('execution_authorized') is True
                and parent.get('active_lane') == lane, 'ACTUAL_STEP_OBSERVER_ONLY_ACTIVE_ROLE')
        snapshot = evidence / ('V3_ACTUAL_STEP_OBSERVER_' + lane.upper() + '_ADMITTED_POLICY_20261128_001.json')
        require(_native_v3_development_read(snapshot, hashlib.sha256(policy_path.read_bytes()).hexdigest()) == policy,
                'ACTUAL_STEP_OBSERVER_EXACT_ROLE_POLICY_SNAPSHOT')
    return {'plan': {'path': str(path), 'sha256': plan_sha256}, 'output_root': str(output),
            'preflight': preflight, 'execution': plan['execution'], 'lane': lane}


def _native_v3_actual_step_observer_environment(plan_sha256, lane):
    require = _native_v3_development_require; prefix = 'GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER'
    execution = _native_v3_development_execution('warm')
    expected = {prefix: 'true', prefix + '_PLAN_SHA256': plan_sha256, prefix + '_LANE': lane,
                'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': ''}
    require(all(not k.startswith(prefix) or k in expected for k in os.environ), 'UNKNOWN_ACTUAL_STEP_OBSERVER_ENV')
    require(all(v in ('false', '') for k, v in os.environ.items() if k.startswith('GX1_NATIVE_') and not k.startswith(prefix)),
            'ACTUAL_STEP_OBSERVER_MIXED_LEGACY_ENV')
    fields = {'MAX_WALL_SECONDS': execution['outer_seconds'], 'MODEL_MAX_WALL_SECONDS': execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED': 'false', 'GPU_INDEX': 0, 'GPU_MAX_POWER_LIMIT_W': 250, 'GPU_MAX_POWER_DRAW_W': 250,
        'GPU_MAX_CORE_TEMP_C': 75, 'GPU_MAX_MEMORY_TEMP_C': 80, 'GPU_MAX_MEMORY_USED_MIB': 12288,
        'GPU_MONITOR_INTERVAL_SECONDS': 1, 'CPU_MAX_TEMP_C': 80, 'CPU_MAX_PACKAGE_POWER_W': 125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES': 8589934592, 'HOST_MIN_COMMIT_HEADROOM_BYTES': 8589934592}
    expected.update({'GX1_TRAINER_' + k: str(v) for k, v in fields.items()})
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'ACTUAL_STEP_OBSERVER_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()


def require_native_v3_actual_step_observer_execution(*, plan_sha256):
    bound = _native_v3_actual_step_observer_bound_plan(plan_sha256)
    proof = _native_v3_actual_step_observer_environment(plan_sha256, bound['lane'])
    return {**proof, **bound, 'kind': 'provider_hlc_v3_actual_step_observer_qualification', 'resource_proof_only': True,
            'training_authority_granted': False, 'optimizer_authority_granted': False}

# END GX1 V3 ACTUAL STEP OBSERVER GUARDS 20261128 001

# BEGIN GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY GUARDS 20261128 001
_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PARENT = '23b166541d86a02b91205c8ce3de7b63392c6be6'
_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_NAME = 'NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_20261128_001'
_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE = 'native_v3_actual_step_observer_cold_recovery_20261128_001'
_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PROFILE = 'gx1_native_actual_step_observer_cold_recovery_trainer20g_v1'
_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_SHELL_ADDITIONS = (('NATIVE_V3_ACTUAL_STEP_OBSERVER=false\n', 'before', 'NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY=false\nNATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PLAN_SHA256=\n'), ('# BEGIN GX1 V3 ACTUAL STEP OBSERVER ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY ROUTE 20261128 001\nmatches_bound_native_v3_actual_step_observer_cold_recovery() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_future_state_step_observer_cold_recovery_v1 \\\n    && "$4" == --v3-actual-step-observer-cold-recovery-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_20261128_001.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_actual_step_observer_cold_recovery() {\n  matches_bound_native_v3_actual_step_observer_cold_recovery "$@" || { echo \'FATAL: exact V3 actual-step observer cold recovery CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 actual-step observer cold recovery requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_20261128_001"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 actual-step observer cold recovery output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY=true\n  NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 ACTUAL STEP OBSERVER DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY DISPATCH 20261128 001\n  local actual_step_observer_cold_recovery_arg\n  for actual_step_observer_cold_recovery_arg in "$@"; do\n    case "$actual_step_observer_cold_recovery_arg" in\n      --v3-actual-step-observer-cold-recovery-plan-sha256|--v3-actual-step-observer-cold-recovery-plan-sha256=*) configure_native_v3_actual_step_observer_cold_recovery "$@"; return ;;\n    esac\n  done\n# END GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER="$NATIVE_V3_ACTUAL_STEP_OBSERVER" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY="$NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY" \\\n  --setenv=GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PLAN_SHA256="$NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PLAN_SHA256" \\\n'))


def _native_v3_actual_step_observer_cold_recovery_without_additions(current):
    current=_native_v3_observed_train_continuation_without_additions(current)
    require = _native_v3_development_require; current = dict(current)
    name = 'gx1/contracts/gx1_capped_execution_v1.py'
    begin = '# BEGIN GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY GUARDS 20261128 001\n'
    end = '# END GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY GUARDS 20261128 001\n\n'
    text = current[name]; require(text.count(begin) == text.count(end) == 1, 'ONE_OBSERVER_COLD_RECOVERY_BLOCK')
    first, tail = text.split(begin); _, last = tail.split(end); text = first + last
    hook = '    current=_native_v3_actual_step_observer_cold_recovery_without_additions(current)\n'
    require(text.count(hook) == 1, 'ONE_OBSERVER_COLD_RECOVERY_BRIDGE_HOOK')
    current[name] = text.replace(hook, '', 1)
    shell = current['scripts/gx1_capped_run.sh']
    for anchor, position, fragment in _V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_SHELL_ADDITIONS:
        require(position == 'before' and shell.count(anchor) == shell.count(fragment)
                == shell.count(fragment + anchor) == 1, 'EXACT_OBSERVER_COLD_RECOVERY_SHELL_ADDITION')
        shell = shell.replace(fragment + anchor, anchor, 1)
    current['scripts/gx1_capped_run.sh'] = shell
    return current


def native_v3_actual_step_observer_cold_recovery_guard_source_bridge(producer_hashes, *, producer_commit=_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PARENT):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(producer_commit == _V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PARENT, 'OBSERVER_COLD_RECOVERY_BRIDGE_PARENT')
    names = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')
    before = {n: subprocess.check_output(['git', 'show', producer_commit + ':' + n], cwd=repo).decode() for n in names}
    original = {n: hashlib.sha256(v.encode()).hexdigest() for n, v in before.items()}
    require(producer_hashes == original, 'OBSERVER_COLD_RECOVERY_BRIDGE_PRODUCER_HASHES')
    current = {n: (repo / n).read_text() for n in names}
    require(_native_v3_actual_step_observer_cold_recovery_without_additions(current) == before, 'OBSERVER_COLD_RECOVERY_OLD_GUARDS_BYTE_EXACT')
    immutable = ('gx1/audit/native_future_state_step_observer_qualify_v1.py',
        'gx1/audit/native_future_state_step_observer_v1.py', 'gx1/audit/native_future_state_training_v1.py',
        'gx1/audit/qualify_native_m5_full_gc_direction_v1.py', 'gx1/contracts/market_state_ta_integrated_model_v1.py',
        'gx1/contracts/market_state_ta_integrated_objective_v1.py')
    proof = {}
    for name in immutable:
        old = subprocess.check_output(['git', 'show', producer_commit + ':' + name], cwd=repo); path = repo / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path and path.read_bytes() == old,
                'OBSERVER_COLD_RECOVERY_IMMUTABLE_MATH_OR_EXECUTION_OWNER')
        proof[name] = hashlib.sha256(old).hexdigest()
    return {'producer_commit': producer_commit, 'checked_guard_owners': 2, 'old_guard_routes_preserved': True,
            'old_numeric_resource_environment_function_unchanged': True, 'old_qualifier_and_math_byte_unchanged': proof,
            'new_cold_recovery_controlflow_requires_scoped_review': True,
            'owners': {n: {'original_sha256': original[n], 'current_sha256': hashlib.sha256(current[n].encode()).hexdigest()} for n in names}}


def _native_v3_actual_step_observer_cold_recovery_bound_plan(plan_sha256):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(type(plan_sha256) is str and re.fullmatch('[0-9a-f]{64}', plan_sha256) is not None, 'OBSERVER_COLD_RECOVERY_PLAN_SHA')
    args = ['--v3-actual-step-observer-cold-recovery-plan-sha256', plan_sha256]; preflight = sys.argv[-1:] == ['--preflight']
    if preflight: args += ['--preflight']
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / 'gx1/audit/native_future_state_step_observer_cold_recovery_v1.py'
            and Path(sys.executable).absolute() == repo / '.venv/bin/python' and sys.argv[1:] == args, 'OBSERVER_COLD_RECOVERY_EXACT_ARGV')
    path = repo / 'configs/research' / (_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_NAME + '.json')
    plan = _native_v3_development_read(path, plan_sha256)
    output = Path('/home/andre2/GX1_RUNS') / _V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_NAME
    require(plan.get('schema_version') == 'gx1_native_v3_actual_step_observer_cold_recovery_plan_v1'
            and plan.get('stage') == _V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE and plan.get('output_root') == str(output),
            'OBSERVER_COLD_RECOVERY_PLAN_IDENTITY')
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent.resolve() == output.parent, 'OBSERVER_COLD_RECOVERY_FRESH_OUTPUT')
    require(plan.get('source_commit') == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'OBSERVER_COLD_RECOVERY_SOURCE_COMMIT')
    require(plan.get('execution') == _native_v3_development_execution('warm')
            and plan.get('execution_profile') == _V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PROFILE, 'OBSERVER_COLD_RECOVERY_EXACT_EXECUTION')
    evidence = Path('/home/andre2/GX1_RUNS') / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = {'path': str(evidence / 'V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_DESIGN_20261128_001.json'),
                  'sha256': 'a4e157293c9339b2afbc4ef2bbf34718c832d932bead198398fc0a636b32b173'}
    cold_ref = {'path': str(repo / 'configs/research' / (_V3_ACTUAL_STEP_OBSERVER_NAME + '_COLD.json')),
                'sha256': 'cb14e5d4cdc9d65b76a689bf9994cf5528388571771318b496362922d45d23ac'}
    reference_ref = {'path': str(repo / 'configs/research' / (_V3_ACTUAL_STEP_OBSERVER_NAME + '_REFERENCE.json')),
                     'sha256': 'c578814baea374e9ab11e7b9ea6dbd6ce0f76f1f550e4321c249f5ee117aa78f'}
    require(plan.get('recovery_design') == design_ref and plan.get('original_cold_plan') == cold_ref
            and plan.get('reference_plan') == reference_ref, 'OBSERVER_COLD_RECOVERY_BOUND_ORIGINAL_REFS')
    design = _native_v3_development_read(design_ref['path'], design_ref['sha256'])
    cold = _native_v3_development_read(cold_ref['path'], cold_ref['sha256'])
    require(design.get('original_failed_COLD', {}).get('plan') == cold_ref
            and design.get('reference', {}).get('plan') == reference_ref
            and plan.get('metadata', {}).get('original_cold') == cold, 'OBSERVER_COLD_RECOVERY_AUTHENTIC_OLD_PLAN')
    unchanged = ('design', 'protocol', 'producer_plan', 'authentic_final_receipt', 'authentic_final_checkpoint',
        'recipe', 'model_contract', 'physical_input_identity', 'backbone_gradient_policy', 'parents')
    require(all(plan.get(k) == cold.get(k) for k in unchanged)
            and all(plan.get('metadata', {}).get(k) == cold.get('metadata', {}).get(k)
                for k in ('original', 'producer', 'receipt', 'trajectory')),
            'OBSERVER_COLD_RECOVERY_UNCHANGED_FUNCTION_AND_TRAJECTORY')
    require(plan.get('lane') == 'cold' and type(plan.get('starting_cursor')) is int and plan['starting_cursor'] == 433
            and type(plan.get('ending_cursor')) is int and plan['ending_cursor'] == 434
            and plan.get('checkpoint_cursors') == [434] and type(plan['checkpoint_cursors'][0]) is int
            and plan.get('observer_enabled') is True and plan.get('technical_weights_never_initializer') is True,
            'OBSERVER_COLD_RECOVERY_ONE_TECHNICAL_STEP')
    require(type(plan.get('limits')) is dict and all(type(v) is int for v in plan['limits'].values())
            and plan['limits'] == cold.get('limits'), 'OBSERVER_COLD_RECOVERY_EXACT_OLD_COLD_BUDGET')
    inputs = plan.get('inputs', {})
    require(type(inputs) is dict and all(inputs.get(k) == v for k, v in cold['inputs'].items())
            and inputs.get('reference_cursor1_receipt') == design.get('reference_C433_receipt')
            and inputs.get('reference_cursor2_receipt') == design.get('reference_C434_receipt'),
            'OBSERVER_COLD_RECOVERY_AUTHENTIC_REFERENCE_C433_C434')
    owners = plan.get('source_owners')
    require(type(owners) is dict and set(cold['source_owners']) <= set(owners)
            and 'gx1/audit/native_future_state_step_observer_cold_recovery_v1.py' in owners, 'OBSERVER_COLD_RECOVERY_SOURCE_INVENTORY')
    mutable = {'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh',
               'tests/test_native_all_tf_provider_hlc_v3_development_guard_v1.py'}
    require(all(owners.get(k) == v for k, v in cold['source_owners'].items() if k not in mutable),
            'OBSERVER_COLD_RECOVERY_ALL_NON_GUARD_OWNERS_UNCHANGED')
    for owner, digest in owners.items():
        p = repo / owner
        require(type(owner) is str and not Path(owner).is_absolute() and '..' not in Path(owner).parts
                and p.is_file() and not p.is_symlink() and p.resolve() == p
                and hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'OBSERVER_COLD_RECOVERY_SOURCE_SHA')
    bridge = plan.get('guard_source_bridge', {}); old = {n: v['original_sha256'] for n, v in bridge.get('owners', {}).items()}
    require(bridge == native_v3_actual_step_observer_cold_recovery_guard_source_bridge(old, producer_commit=bridge.get('producer_commit')),
            'OBSERVER_COLD_RECOVERY_EXACT_GUARD_ONLY_SOURCE_BRIDGE')
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = _native_v3_development_read(policy_path)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')
    require(all(policy.get(k) is (not preflight and k in flags[:2]) for k in flags), 'OBSERVER_COLD_RECOVERY_EXACT_FLAGS')
    original = policy.get(_V3_ACTUAL_STEP_OBSERVER_STAGE, {}); lanes = original.get('lane_gates', {})
    require(original.get('execution_authorized') is False and original.get('training_authorized') is False
            and type(original.get('remaining_invocations')) is int and original['remaining_invocations'] == 0
            and original.get('active_lane') in (None, '') and original.get('relaunch_allowed') is False
            and type(lanes) is dict and set(lanes) == {'reference', 'cold'}, 'OBSERVER_COLD_RECOVERY_OLD_QUALIFICATION_CLOSED')
    for lane, status in (('reference', 'CLOSED_CONSUMED_COMPLETE'), ('cold', 'CLOSED_CONSUMED_FAILED_CONSUMED')):
        v = lanes[lane]
        require(v.get('status') == status and type(v.get('remaining_invocations')) is int and v['remaining_invocations'] == 0
                and all(v.get(k) is False for k in ('execution_authorized', 'training_authorized', 'test_authorized', 'relaunch_allowed'))
                and v.get('plan') == (reference_ref if lane == 'reference' else cold_ref), 'OBSERVER_COLD_RECOVERY_OLD_ROLE_TERMINALS_CLOSED')
    for name, prior in policy.items():
        if name.startswith('native_') and name not in (_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE, _V3_ACTUAL_STEP_OBSERVER_STAGE) and type(prior) is dict:
            require(all(k not in prior or prior[k] is False for k in ('execution_authorized', 'training_authorized'))
                    and ('remaining_invocations' not in prior or type(prior['remaining_invocations']) is int and prior['remaining_invocations'] == 0),
                    'OBSERVER_COLD_RECOVERY_OTHER_NATIVE_ROUTES_CLOSED')
            grouped = prior.get('lane_gates', prior.get('phase_gates', {}))
            require(type(grouped) is dict, 'OBSERVER_COLD_RECOVERY_TYPED_OLD_ROLE_GATES')
            aggregation = prior.get('aggregation_completion')
            if aggregation is not None:
                require(type(aggregation) is dict and type(aggregation.get('remaining_invocations')) is int
                        and aggregation['remaining_invocations'] == 0
                        and all(aggregation.get(k) is False for k in ('execution_authorized', 'training_authorized', 'relaunch_allowed')),
                        'OBSERVER_COLD_RECOVERY_OLD_AGGREGATION_CLOSED')
            for gate in grouped.values():
                require(type(gate) is dict and type(gate.get('remaining_invocations')) is int and gate['remaining_invocations'] == 0
                        and all(gate.get(k) is False for k in ('execution_authorized', 'training_authorized', 'test_authorized', 'relaunch_allowed')),
                        'OBSERVER_COLD_RECOVERY_OTHER_NATIVE_ROLE_GATES_CLOSED')
    parent = policy.get(_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE, {})
    status = 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY'
    gate = {'status': status, 'plan': {'path': str(path), 'sha256': plan_sha256}, 'execution_authorized': not preflight,
            'remaining_invocations': int(not preflight), 'relaunch_allowed': False, 'training_authorized': not preflight,
            'test_authorized': False, 'spending_authorized': False}
    require(parent.get('design') == design_ref and parent.get('source_preparation_authorized') is True
            and type(parent.get('remaining_invocations')) is int
            and all(type(parent.get(k)) is bool for k in ('execution_authorized', 'training_authorized', 'test_authorized',
                'spending_authorized', 'relaunch_allowed')) and all(parent.get(k) == v for k, v in gate.items()),
            'OBSERVER_COLD_RECOVERY_EXACT_ONE_USE_GATE')
    if not preflight:
        snapshot = evidence / 'V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_ADMITTED_POLICY_20261128_001.json'
        require(_native_v3_development_read(snapshot, hashlib.sha256(policy_path.read_bytes()).hexdigest()) == policy,
                'OBSERVER_COLD_RECOVERY_EXACT_ADMITTED_POLICY_SNAPSHOT')
    return {'plan': {'path': str(path), 'sha256': plan_sha256}, 'output_root': str(output), 'preflight': preflight,
            'execution': plan['execution'], 'lane': 'cold'}


def _native_v3_actual_step_observer_cold_recovery_environment(plan_sha256):
    require = _native_v3_development_require; prefix = 'GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY'
    execution = _native_v3_development_execution('warm')
    expected = {prefix: 'true', prefix + '_PLAN_SHA256': plan_sha256,
                'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': ''}
    require(all(not k.startswith(prefix) or k in expected for k in os.environ), 'UNKNOWN_OBSERVER_COLD_RECOVERY_ENV')
    require(all(v in ('false', '') for k, v in os.environ.items() if k.startswith('GX1_NATIVE_') and not k.startswith(prefix)),
            'OBSERVER_COLD_RECOVERY_MIXED_LEGACY_ENV')
    fields = {'MAX_WALL_SECONDS': execution['outer_seconds'], 'MODEL_MAX_WALL_SECONDS': execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED': 'false', 'GPU_INDEX': 0, 'GPU_MAX_POWER_LIMIT_W': 250, 'GPU_MAX_POWER_DRAW_W': 250,
        'GPU_MAX_CORE_TEMP_C': 75, 'GPU_MAX_MEMORY_TEMP_C': 80, 'GPU_MAX_MEMORY_USED_MIB': 12288,
        'GPU_MONITOR_INTERVAL_SECONDS': 1, 'CPU_MAX_TEMP_C': 80, 'CPU_MAX_PACKAGE_POWER_W': 125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES': 8589934592, 'HOST_MIN_COMMIT_HEADROOM_BYTES': 8589934592}
    expected.update({'GX1_TRAINER_' + k: str(v) for k, v in fields.items()})
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'OBSERVER_COLD_RECOVERY_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()


def require_native_v3_actual_step_observer_cold_recovery_execution(*, plan_sha256):
    bound = _native_v3_actual_step_observer_cold_recovery_bound_plan(plan_sha256)
    proof = _native_v3_actual_step_observer_cold_recovery_environment(plan_sha256)
    return {**proof, **bound, 'kind': 'provider_hlc_v3_actual_step_observer_cold_recovery', 'resource_proof_only': True,
            'training_authority_granted': False, 'optimizer_authority_granted': False}

# END GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY GUARDS 20261128 001

# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION GUARDS 20261128 001
_V3_OBSERVED_TRAIN_CONTINUATION_PARENT = '4c6124d988e977143006559840adb5ddffca9955'
_V3_OBSERVED_TRAIN_CONTINUATION_NAME = 'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_001'
_V3_OBSERVED_TRAIN_CONTINUATION_STAGE = 'native_v3_observed_train_continuation_20261128_001'
_V3_OBSERVED_TRAIN_CONTINUATION_PROFILE = 'gx1_native_observed_train_continuation_trainer20g_v1'
_V3_OBSERVED_TRAIN_CONTINUATION_SHELL_ADDITIONS = (('NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY=false\n', 'before', 'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION=false\nNATIVE_V3_OBSERVED_TRAIN_CONTINUATION_PLAN_SHA256=\n'), ('# BEGIN GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION ROUTE 20261128 001\nmatches_bound_native_v3_observed_train_continuation() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_future_state_observed_train_continuation_v1 \\\n    && "$4" == --v3-observed-train-continuation-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_001.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_observed_train_continuation() {\n  matches_bound_native_v3_observed_train_continuation "$@" || { echo \'FATAL: exact V3 observed TRAIN continuation CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 observed TRAIN continuation requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_001"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 observed TRAIN continuation output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_OBSERVED_TRAIN_CONTINUATION=true\n  NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 OBSERVED TRAIN CONTINUATION ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 ACTUAL STEP OBSERVER COLD RECOVERY DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION DISPATCH 20261128 001\n  local observed_train_continuation_arg\n  for observed_train_continuation_arg in "$@"; do\n    case "$observed_train_continuation_arg" in\n      --v3-observed-train-continuation-plan-sha256|--v3-observed-train-continuation-plan-sha256=*) configure_native_v3_observed_train_continuation "$@"; return ;;\n    esac\n  done\n# END GX1 V3 OBSERVED TRAIN CONTINUATION DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY="$NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION="$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION" \\\n  --setenv=GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_PLAN_SHA256="$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_PLAN_SHA256" \\\n'))


def _native_v3_observed_train_continuation_without_additions(current):
    current=_native_v3_observed_train_continuation_002_without_additions(current)
    require = _native_v3_development_require; current = dict(current)
    name = 'gx1/contracts/gx1_capped_execution_v1.py'
    begin = '# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION GUARDS 20261128 001\n'
    end = '# END GX1 V3 OBSERVED TRAIN CONTINUATION GUARDS 20261128 001\n\n'
    text = current[name]; require(text.count(begin) == text.count(end) == 1, 'ONE_OBSERVED_TRAIN_CONTINUATION_BLOCK')
    first, tail = text.split(begin); _, last = tail.split(end); text = first + last
    hook = '    current=_native_v3_observed_train_continuation_without_additions(current)\n'
    require(text.count(hook) == 1, 'ONE_OBSERVED_TRAIN_CONTINUATION_BRIDGE_HOOK')
    current[name] = text.replace(hook, '', 1)
    shell = current['scripts/gx1_capped_run.sh']
    for anchor, position, fragment in _V3_OBSERVED_TRAIN_CONTINUATION_SHELL_ADDITIONS:
        require(position == 'before' and shell.count(anchor) == shell.count(fragment)
                == shell.count(fragment + anchor) == 1, 'EXACT_OBSERVED_TRAIN_CONTINUATION_SHELL_ADDITION')
        shell = shell.replace(fragment + anchor, anchor, 1)
    current['scripts/gx1_capped_run.sh'] = shell
    return current


def native_v3_observed_train_continuation_guard_source_bridge(producer_hashes, *, producer_commit=_V3_OBSERVED_TRAIN_CONTINUATION_PARENT):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(producer_commit == _V3_OBSERVED_TRAIN_CONTINUATION_PARENT, 'OBSERVED_TRAIN_CONTINUATION_BRIDGE_PARENT')
    names = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')
    before = {n: subprocess.check_output(['git', 'show', producer_commit + ':' + n], cwd=repo).decode() for n in names}
    original = {n: hashlib.sha256(v.encode()).hexdigest() for n, v in before.items()}
    require(producer_hashes == original, 'OBSERVED_TRAIN_CONTINUATION_BRIDGE_PRODUCER_HASHES')
    current = {n: (repo / n).read_text() for n in names}
    require(_native_v3_observed_train_continuation_without_additions(current) == before, 'OBSERVED_TRAIN_CONTINUATION_OLD_GUARDS_BYTE_EXACT')
    immutable = ('gx1/audit/native_future_state_step_observer_qualify_v1.py',
        'gx1/audit/native_future_state_step_observer_v1.py', 'gx1/audit/native_future_state_training_v1.py',
        'gx1/audit/qualify_native_m5_full_gc_direction_v1.py', 'gx1/contracts/market_state_ta_integrated_model_v1.py',
        'gx1/contracts/market_state_ta_integrated_objective_v1.py')
    proof = {}
    for name in immutable:
        old = subprocess.check_output(['git', 'show', producer_commit + ':' + name], cwd=repo); path = repo / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path and path.read_bytes() == old,
                'OBSERVED_TRAIN_CONTINUATION_IMMUTABLE_NN_OWNER')
        proof[name] = hashlib.sha256(old).hexdigest()
    return {'producer_commit': producer_commit, 'checked_guard_owners': 2, 'old_guard_routes_preserved': True,
            'old_numeric_resource_environment_function_unchanged': True, 'old_NN_source_byte_unchanged': proof,
            'new_continuation_controlflow_requires_scoped_review': True,
            'owners': {n: {'original_sha256': original[n], 'current_sha256': hashlib.sha256(current[n].encode()).hexdigest()} for n in names}}


def _native_v3_observed_train_continuation_bound_plan(plan_sha256):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(type(plan_sha256) is str and re.fullmatch('[0-9a-f]{64}', plan_sha256) is not None, 'OBSERVED_CONTINUATION_PLAN_SHA')
    args = ['--v3-observed-train-continuation-plan-sha256', plan_sha256]; preflight = sys.argv[-1:] == ['--preflight']
    if preflight: args += ['--preflight']
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / 'gx1/audit/native_future_state_observed_train_continuation_v1.py'
            and Path(sys.executable).absolute() == repo / '.venv/bin/python' and sys.argv[1:] == args, 'OBSERVED_CONTINUATION_EXACT_ARGV')
    path = repo / 'configs/research' / (_V3_OBSERVED_TRAIN_CONTINUATION_NAME + '.json')
    plan = _native_v3_development_read(path, plan_sha256)
    output = Path('/home/andre2/GX1_RUNS') / _V3_OBSERVED_TRAIN_CONTINUATION_NAME
    require(plan.get('schema_version') == 'gx1_native_v3_observed_train_continuation_plan_v1'
            and plan.get('stage') == _V3_OBSERVED_TRAIN_CONTINUATION_STAGE and plan.get('output_root') == str(output),
            'OBSERVED_CONTINUATION_PLAN_IDENTITY')
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent.resolve() == output.parent, 'OBSERVED_CONTINUATION_FRESH_OUTPUT')
    require(plan.get('source_commit') == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'OBSERVED_CONTINUATION_SOURCE_COMMIT')
    require(plan.get('execution') == _native_v3_development_execution('warm')
            and plan.get('execution_profile') == _V3_OBSERVED_TRAIN_CONTINUATION_PROFILE, 'OBSERVED_CONTINUATION_EXACT_EXECUTION')
    evidence = Path('/home/andre2/GX1_RUNS') / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = {'path': str(evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_001.json'),
                  'sha256': '5b04f35e2765c45c2cf331cada40f0d96497b1561e69183add7a38be6a5eb269'}
    producer_ref = {'path': str(repo / 'configs/research' / (_V3_ISSUE_MIX_NAME + '_FIT.json')),
                    'sha256': '25ca4cc9d780f5d16376a765ec58f6d84c58a1f16f55d48033d11168fda2d4a8'}
    final_root = Path('/home/andre2/GX1_RUNS') / (_V3_ISSUE_MIX_NAME + '_fit')
    receipt_ref = {'path': str(final_root / 'FINAL.json'), 'sha256': '1b67d9e51e211480c168b0eae05e5c1677e804e4e4d58e7b9bb1fb619271884d'}
    checkpoint_ref = {'path': str(final_root / 'FINAL.pt'), 'sha256': '577b5b37f5c4cfa98744cfeefb06e8156231488a7adc038f33cdfd685df86632'}
    require(plan.get('design') == design_ref and plan.get('producer_plan') == producer_ref
            and plan.get('authentic_final_receipt') == receipt_ref and plan.get('authentic_final_checkpoint') == checkpoint_ref,
            'OBSERVED_CONTINUATION_GENUINE_FINAL432_REFS')
    design = _native_v3_development_read(design_ref['path'], design_ref['sha256'])
    producer = _native_v3_development_read(producer_ref['path'], producer_ref['sha256'])
    receipt = _native_v3_development_read(receipt_ref['path'], receipt_ref['sha256'])
    require(type(receipt.get('logical_cursor')) is int and type(receipt.get('optimizer_updates')) is int
            and receipt['logical_cursor'] == receipt['optimizer_updates'] == 432 and receipt.get('producer_lane') == 'fit'
            and receipt.get('original_plan') == receipt.get('execution_plan') == producer_ref and receipt.get('checkpoint') == checkpoint_ref
            and plan.get('metadata', {}).get('producer') == producer and plan.get('metadata', {}).get('receipt') == receipt
            and plan.get('metadata', {}).get('original') == producer.get('metadata', {}).get('original')
            and plan.get('recipe') == producer.get('recipe') and plan.get('model_contract') == producer.get('model_contract')
            and plan.get('physical_input_identity') == producer.get('physical_input_identity')
            and 'backbone_gradient_policy' in plan and plan['backbone_gradient_policy'] is None, 'OBSERVED_CONTINUATION_UNCHANGED_FUNCTION')
    require(type(plan.get('starting_cursor')) is int and plan['starting_cursor'] == 432
            and type(plan.get('ending_cursor')) is int and plan['ending_cursor'] == 648
            and plan.get('checkpoint_cursors') == [433, 648] and all(type(v) is int for v in plan['checkpoint_cursors'])
            and plan.get('observer_enabled') is True and plan.get('phase_alias') == design.get('phase_alias'),
            'OBSERVED_CONTINUATION_FIXED_ENDPOINT_AND_ALIAS')
    old = producer['schedule']; schedule = plan.get('schedule'); expected = dict(old); expected.pop('contract_sha256')
    expected.update(schema_version='gx1_native_v3_observed_train_continuation_schedule_v1', logical_batches=648,
        optimizer_updates=648, passes=3, skipped_batches=0, original_prefix_schedule_sha256=old['contract_sha256'],
        original_take_indices_appended=list(range(216, 432)), order_mode='repeat_locked_original_second_pass_once')
    for key in ('cohort_takes', 'logical_step_rows', 'cohort_take_state_sha256_by_logical_batch', 'update_flags'):
        expected[key] = old[key] + old[key][216:432]
    for key in ('raw_permutation_state_sha256_by_pass', 'actual_sorted_partition_stream_state_sha256_by_pass'):
        expected[key] = old[key] + [old[key][1]]
    require(type(schedule) is dict and set(schedule) == set(expected) | {'contract_sha256'}
            and all(schedule.get(k) == v for k, v in expected.items())
            and type(schedule.get('contract_sha256')) is str and re.fullmatch('[0-9a-f]{64}', schedule['contract_sha256']) is not None
            and all(type(schedule.get(k)) is int for k in ('logical_batches', 'optimizer_updates', 'passes', 'skipped_batches'))
            and all(type(v) is int for v in schedule['logical_step_rows'] + schedule['original_take_indices_appended'])
            and all(type(v) is bool for v in schedule['update_flags'])
            and all(type(take) is list and all(type(v) is int for v in take) for take in schedule['cohort_takes'])
            and plan.get('metadata', {}).get('trajectory', {}).get('schedule') == schedule, 'OBSERVED_CONTINUATION_EXACT_648_SCHEDULE')
    limits = dict.fromkeys(producer['limits'], 0)
    limits.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
        target_archive_loads=1, checkpoint_loads=1, model_restores=1, model_forwards=297, backbone_forwards=297,
        readout_forwards=297, backward_calls=216, optimizer_steps=216, observer_records=216,
        checkpoint_writes=2, checkpoint_readbacks=2, checkpoint_prepublication_loads=2,
        checkpoint_receipt_writes=2, logit_archive_writes=1, logit_archive_readbacks=1)
    require(type(plan.get('limits')) is dict and all(type(v) is int for v in plan['limits'].values())
            and plan['limits'] == limits, 'OBSERVED_CONTINUATION_EXACT_BUDGET')
    require(type(plan.get('inputs')) is dict and all(plan['inputs'].get(k) == v for k, v in producer['inputs'].items()),
            'OBSERVED_CONTINUATION_PRESERVED_INPUT_ROLES')
    owners = plan.get('source_owners'); new = {'gx1/audit/native_future_state_observed_train_continuation_v1.py',
        'tests/test_native_future_state_observed_train_continuation_v1.py', 'tests/test_native_observed_train_continuation_guard_v1.py'}
    require(type(owners) is dict and set(producer['source_owners']) | new <= set(owners), 'OBSERVED_CONTINUATION_SOURCE_INVENTORY')
    guards = {'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'}
    for owner, digest in owners.items():
        p = repo / owner
        require(type(owner) is str and not Path(owner).is_absolute() and '..' not in Path(owner).parts
                and p.is_file() and not p.is_symlink() and p.resolve() == p
                and hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'OBSERVED_CONTINUATION_SOURCE_SHA')
        if owner not in new | guards:
            previous = subprocess.check_output(['git', 'show', _V3_OBSERVED_TRAIN_CONTINUATION_PARENT + ':' + owner], cwd=repo)
            require(hashlib.sha256(previous).hexdigest() == digest, 'OBSERVED_CONTINUATION_ALL_OLD_NON_GUARD_SOURCE_UNCHANGED')
    bridge = plan.get('guard_source_bridge', {}); hashes = {n: v['original_sha256'] for n, v in bridge.get('owners', {}).items()}
    require(bridge == native_v3_observed_train_continuation_guard_source_bridge(hashes, producer_commit=bridge.get('producer_commit')),
            'OBSERVED_CONTINUATION_STRICT_GUARD_SOURCE_BRIDGE')
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = _native_v3_development_read(policy_path)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')
    require(all(policy.get(k) is (not preflight and k in flags[:2]) for k in flags), 'OBSERVED_CONTINUATION_EXACT_FLAGS')
    for name, prior in policy.items():
        if name.startswith('native_') and name != _V3_OBSERVED_TRAIN_CONTINUATION_STAGE and type(prior) is dict:
            require(all(k not in prior or prior[k] is False for k in ('execution_authorized', 'training_authorized'))
                    and ('remaining_invocations' not in prior or type(prior['remaining_invocations']) is int and prior['remaining_invocations'] == 0),
                    'OBSERVED_CONTINUATION_OLD_NATIVE_ROUTES_CLOSED')
            grouped = prior.get('lane_gates', prior.get('phase_gates', {})); require(type(grouped) is dict, 'OBSERVED_CONTINUATION_TYPED_OLD_LANES')
            for value in list(grouped.values()) + ([prior['aggregation_completion']] if 'aggregation_completion' in prior else []):
                require(type(value) is dict and type(value.get('remaining_invocations')) is int and value['remaining_invocations'] == 0
                        and all(value.get(k) is False for k in ('execution_authorized', 'training_authorized', 'relaunch_allowed')),
                        'OBSERVED_CONTINUATION_OLD_NATIVE_ROLE_GATES_CLOSED')
    recovery = policy.get(_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE, {})
    require(recovery.get('status') == 'CLOSED_CONSUMED_COMPLETE' and recovery.get('actual_review') == design.get('basis'),
            'OBSERVED_CONTINUATION_OPERATIONAL_OBSERVER_QUALIFIED')
    parent = policy.get(_V3_OBSERVED_TRAIN_CONTINUATION_STAGE, {})
    gate = {'status': 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_OBSERVED_TRAIN_CONTINUATION',
            'plan': {'path': str(path), 'sha256': plan_sha256}, 'execution_authorized': not preflight,
            'remaining_invocations': int(not preflight), 'relaunch_allowed': False, 'training_authorized': not preflight,
            'test_authorized': False, 'spending_authorized': False}
    require(parent.get('design') == design_ref and parent.get('source_preparation_authorized') is True
            and type(parent.get('remaining_invocations')) is int
            and all(type(parent.get(k)) is bool for k in ('execution_authorized', 'training_authorized', 'test_authorized',
                'spending_authorized', 'relaunch_allowed')) and all(parent.get(k) == v for k, v in gate.items()),
            'OBSERVED_CONTINUATION_EXACT_ONE_USE_GATE')
    if not preflight:
        snapshot = evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_20261128_001.json'
        require(_native_v3_development_read(snapshot, hashlib.sha256(policy_path.read_bytes()).hexdigest()) == policy,
                'OBSERVED_CONTINUATION_EXACT_ADMITTED_POLICY_SNAPSHOT')
    return {'plan': {'path': str(path), 'sha256': plan_sha256}, 'output_root': str(output), 'preflight': preflight, 'execution': plan['execution']}


def _native_v3_observed_train_continuation_environment(plan_sha256):
    require = _native_v3_development_require; prefix = 'GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION'
    execution = _native_v3_development_execution('warm')
    expected = {prefix: 'true', prefix + '_PLAN_SHA256': plan_sha256,
                'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': ''}
    require(all(not k.startswith(prefix) or k in expected for k in os.environ), 'UNKNOWN_OBSERVED_CONTINUATION_ENV')
    require(all(v in ('false', '') for k, v in os.environ.items() if k.startswith('GX1_NATIVE_') and not k.startswith(prefix)),
            'OBSERVED_CONTINUATION_MIXED_LEGACY_ENV')
    fields = {'MAX_WALL_SECONDS': execution['outer_seconds'], 'MODEL_MAX_WALL_SECONDS': execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED': 'false', 'GPU_INDEX': 0, 'GPU_MAX_POWER_LIMIT_W': 250, 'GPU_MAX_POWER_DRAW_W': 250,
        'GPU_MAX_CORE_TEMP_C': 75, 'GPU_MAX_MEMORY_TEMP_C': 80, 'GPU_MAX_MEMORY_USED_MIB': 12288,
        'GPU_MONITOR_INTERVAL_SECONDS': 1, 'CPU_MAX_TEMP_C': 80, 'CPU_MAX_PACKAGE_POWER_W': 125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES': 8589934592, 'HOST_MIN_COMMIT_HEADROOM_BYTES': 8589934592}
    expected.update({'GX1_TRAINER_' + k: str(v) for k, v in fields.items()})
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'OBSERVED_CONTINUATION_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()


def require_native_v3_observed_train_continuation_execution(*, plan_sha256):
    bound = _native_v3_observed_train_continuation_bound_plan(plan_sha256)
    proof = _native_v3_observed_train_continuation_environment(plan_sha256)
    return {**proof, **bound, 'kind': 'provider_hlc_v3_observed_train_continuation', 'resource_proof_only': True,
            'training_authority_granted': False, 'optimizer_authority_granted': False}

# END GX1 V3 OBSERVED TRAIN CONTINUATION GUARDS 20261128 001

# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION 002 GUARDS 20261128 001
_V3_OBSERVED_TRAIN_CONTINUATION_002_PARENT = 'ba00b1e121c3601494092c2f866bd6b912410e6c'
_V3_OBSERVED_TRAIN_CONTINUATION_002_NAME = 'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_002'
_V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE = 'native_v3_observed_train_continuation_20261128_002'
_V3_OBSERVED_TRAIN_CONTINUATION_002_PROFILE = 'gx1_native_observed_train_continuation_002_trainer20g_v1'
_V3_OBSERVED_TRAIN_CONTINUATION_002_SHELL_ADDITIONS = (('NATIVE_V3_OBSERVED_TRAIN_CONTINUATION=false\n', 'before', 'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002=false\nNATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002_PLAN_SHA256=\n'), ('# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION 002 ROUTE 20261128 001\nmatches_bound_native_v3_observed_train_continuation_002() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_future_state_observed_train_continuation_002_v1 \\\n    && "$4" == --v3-observed-train-continuation-002-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_002.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_observed_train_continuation_002() {\n  matches_bound_native_v3_observed_train_continuation_002 "$@" || { echo \'FATAL: exact V3 observed TRAIN continuation CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 observed TRAIN continuation requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_002"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 observed TRAIN continuation output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002=true\n  NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 OBSERVED TRAIN CONTINUATION 002 ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION 002 DISPATCH 20261128 001\n  local observed_train_continuation_002_arg\n  for observed_train_continuation_002_arg in "$@"; do\n    case "$observed_train_continuation_002_arg" in\n      --v3-observed-train-continuation-002-plan-sha256|--v3-observed-train-continuation-002-plan-sha256=*) configure_native_v3_observed_train_continuation_002 "$@"; return ;;\n    esac\n  done\n# END GX1 V3 OBSERVED TRAIN CONTINUATION 002 DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION="$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002="$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002" \\\n  --setenv=GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002_PLAN_SHA256="$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002_PLAN_SHA256" \\\n'))


def _native_v3_observed_train_continuation_002_without_additions(current):
    current=_native_v3_frozen_final864_mode_gap_without_additions(current)
    require = _native_v3_development_require; current = dict(current)
    name = 'gx1/contracts/gx1_capped_execution_v1.py'
    begin = '# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION 002 GUARDS 20261128 001\n'
    end = '# END GX1 V3 OBSERVED TRAIN CONTINUATION 002 GUARDS 20261128 001\n\n'
    text = current[name]; require(text.count(begin) == text.count(end) == 1, 'ONE_OBSERVED_CONTINUATION_002_BLOCK')
    first, tail = text.split(begin); _, last = tail.split(end); text = first + last
    hook = '    current=_native_v3_observed_train_continuation_002_without_additions(current)\n'
    require(text.count(hook) == 1, 'ONE_OBSERVED_CONTINUATION_002_BRIDGE_HOOK')
    current[name] = text.replace(hook, '', 1)
    shell = current['scripts/gx1_capped_run.sh']
    for anchor, position, fragment in _V3_OBSERVED_TRAIN_CONTINUATION_002_SHELL_ADDITIONS:
        require(position == 'before' and shell.count(anchor) == shell.count(fragment)
                == shell.count(fragment + anchor) == 1, 'EXACT_OBSERVED_CONTINUATION_002_SHELL_ADDITION')
        shell = shell.replace(fragment + anchor, anchor, 1)
    current['scripts/gx1_capped_run.sh'] = shell
    return current


def native_v3_observed_train_continuation_002_guard_source_bridge(producer_hashes, *, producer_commit=_V3_OBSERVED_TRAIN_CONTINUATION_002_PARENT):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(producer_commit == _V3_OBSERVED_TRAIN_CONTINUATION_002_PARENT, 'OBSERVED_CONTINUATION_002_BRIDGE_PARENT')
    names = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')
    before = {n: subprocess.check_output(['git', 'show', producer_commit + ':' + n], cwd=repo).decode() for n in names}
    original = {n: hashlib.sha256(v.encode()).hexdigest() for n, v in before.items()}
    require(producer_hashes == original, 'OBSERVED_CONTINUATION_002_BRIDGE_PRODUCER_HASHES')
    current = {n: (repo / n).read_text() for n in names}
    require(_native_v3_observed_train_continuation_002_without_additions(current) == before, 'OBSERVED_CONTINUATION_002_OLD_GUARDS_BYTE_EXACT')
    immutable = ('gx1/audit/native_future_state_observed_train_continuation_v1.py',
        'gx1/audit/native_future_state_step_observer_qualify_v1.py', 'gx1/audit/native_future_state_step_observer_v1.py',
        'gx1/audit/native_future_state_training_v1.py', 'gx1/audit/qualify_native_m5_full_gc_direction_v1.py',
        'gx1/contracts/market_state_ta_integrated_model_v1.py', 'gx1/contracts/market_state_ta_integrated_objective_v1.py')
    proof = {}
    for name in immutable:
        old = subprocess.check_output(['git', 'show', producer_commit + ':' + name], cwd=repo); path = repo / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path and path.read_bytes() == old,
                'OBSERVED_CONTINUATION_002_IMMUTABLE_NN_OR_001_OWNER')
        proof[name] = hashlib.sha256(old).hexdigest()
    return {'producer_commit': producer_commit, 'checked_guard_owners': 2, 'old_guard_routes_preserved': True,
            'old_numeric_resource_environment_function_unchanged': True, 'old_NN_and_001_source_byte_unchanged': proof,
            'new_002_controlflow_requires_scoped_review': True,
            'owners': {n: {'original_sha256': original[n], 'current_sha256': hashlib.sha256(current[n].encode()).hexdigest()} for n in names}}


def _native_v3_observed_train_continuation_002_bound_plan(plan_sha256):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(type(plan_sha256) is str and re.fullmatch('[0-9a-f]{64}', plan_sha256) is not None, 'OBSERVED_CONTINUATION_002_PLAN_SHA')
    args = ['--v3-observed-train-continuation-002-plan-sha256', plan_sha256]; preflight = sys.argv[-1:] == ['--preflight']
    if preflight: args += ['--preflight']
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / 'gx1/audit/native_future_state_observed_train_continuation_002_v1.py'
            and Path(sys.executable).absolute() == repo / '.venv/bin/python' and sys.argv[1:] == args, 'OBSERVED_CONTINUATION_002_EXACT_ARGV')
    path = repo / 'configs/research' / (_V3_OBSERVED_TRAIN_CONTINUATION_002_NAME + '.json')
    plan = _native_v3_development_read(path, plan_sha256)
    output = Path('/home/andre2/GX1_RUNS') / _V3_OBSERVED_TRAIN_CONTINUATION_002_NAME
    require(plan.get('schema_version') == 'gx1_native_v3_observed_train_continuation_plan_v2'
            and plan.get('stage') == _V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE and plan.get('output_root') == str(output),
            'OBSERVED_CONTINUATION_002_PLAN_IDENTITY')
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent.resolve() == output.parent, 'OBSERVED_CONTINUATION_002_FRESH_OUTPUT')
    require(plan.get('source_commit') == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'OBSERVED_CONTINUATION_002_SOURCE_COMMIT')
    require(plan.get('execution') == _native_v3_development_execution('warm')
            and plan.get('execution_profile') == _V3_OBSERVED_TRAIN_CONTINUATION_002_PROFILE, 'OBSERVED_CONTINUATION_002_EXACT_EXECUTION')
    evidence = Path('/home/andre2/GX1_RUNS') / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = {'path': str(evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_20261128_002.json'),
                  'sha256': 'f52d056a1715f13180b6aee375ae17fd45eb992644bf447f38a0239ebf34bf30'}
    parent_ref = {'path': str(repo / 'configs/research' / (_V3_OBSERVED_TRAIN_CONTINUATION_NAME + '.json')),
                  'sha256': 'c4e9b8ff8f760b062d773b08a585caeff88cec551ced3f6e57704da61c50f41f'}
    original_ref = {'path': str(repo / 'configs/research' / (_V3_ISSUE_MIX_NAME + '_FIT.json')),
                    'sha256': '25ca4cc9d780f5d16376a765ec58f6d84c58a1f16f55d48033d11168fda2d4a8'}
    final_root = Path('/home/andre2/GX1_RUNS') / _V3_OBSERVED_TRAIN_CONTINUATION_NAME
    receipt_ref = {'path': str(final_root / 'FINAL648.json'), 'sha256': 'f3534bdeeb460a56f04b5c8fa7b7142a56e43341ef6075b653fd30558fc8d0b3'}
    checkpoint_ref = {'path': str(final_root / 'FINAL648.pt'), 'sha256': 'c21159dcade8f0fe36eddd4d0d53c62d1fa07be2985a19b18b7b0d4232b8d150'}
    require(plan.get('design') == design_ref and plan.get('producer_plan') == parent_ref
            and plan.get('original_producer_plan') == original_ref and plan.get('authentic_final_receipt') == receipt_ref
            and plan.get('authentic_final_checkpoint') == checkpoint_ref, 'OBSERVED_CONTINUATION_002_OWN_FINAL648_REFS')
    design = _native_v3_development_read(design_ref['path'], design_ref['sha256'])
    parent_plan = _native_v3_development_read(parent_ref['path'], parent_ref['sha256'])
    original = _native_v3_development_read(original_ref['path'], original_ref['sha256'])
    receipt = _native_v3_development_read(receipt_ref['path'], receipt_ref['sha256'])
    require(type(receipt.get('logical_cursor')) is int and type(receipt.get('optimizer_updates')) is int
            and receipt['logical_cursor'] == receipt['optimizer_updates'] == 648 and receipt.get('producer_lane') == 'continuation'
            and receipt.get('original_plan') == receipt.get('execution_plan') == parent_ref and receipt.get('checkpoint') == checkpoint_ref
            and receipt.get('trajectory_sha256') == parent_plan.get('metadata', {}).get('trajectory', {}).get('contract_sha256')
            and plan.get('metadata', {}).get('parent_plan') == parent_ref and plan.get('metadata', {}).get('original_plan') == original_ref
            and plan.get('metadata', {}).get('receipt') == receipt and parent_plan.get('producer_plan') == original_ref
            and plan.get('recipe') == parent_plan.get('recipe') == original.get('recipe')
            and plan.get('model_contract') == parent_plan.get('model_contract') == original.get('model_contract')
            and plan.get('physical_input_identity') == parent_plan.get('physical_input_identity')
            and 'backbone_gradient_policy' in plan and plan['backbone_gradient_policy'] is None,
            'OBSERVED_CONTINUATION_002_AUTHENTIC_648_TRAJECTORY_FUNCTION')
    require(type(plan.get('starting_cursor')) is int and plan['starting_cursor'] == 648
            and type(plan.get('ending_cursor')) is int and plan['ending_cursor'] == 864
            and plan.get('checkpoint_cursors') == [649, 864] and all(type(v) is int for v in plan['checkpoint_cursors'])
            and plan.get('observer_enabled') is True and plan.get('phase_alias') == design.get('phase_alias'),
            'OBSERVED_CONTINUATION_002_LOCKED_ENDPOINT_ALIAS')
    source = parent_plan['schedule']; old = original['schedule']; schedule = plan.get('schedule')
    expected = dict(source); expected.pop('contract_sha256')
    expected.update(schema_version='gx1_native_v3_observed_train_continuation_002_schedule_v1', logical_batches=864,
        optimizer_updates=864, passes=4, skipped_batches=0, original_prefix_schedule_sha256=source['contract_sha256'],
        original_take_indices_appended=list(range(216, 432)), order_mode='repeat_locked_original_second_pass_once')
    for key in ('cohort_takes', 'logical_step_rows', 'cohort_take_state_sha256_by_logical_batch', 'update_flags'):
        expected[key] = source[key] + old[key][216:432]
    for key in ('raw_permutation_state_sha256_by_pass', 'actual_sorted_partition_stream_state_sha256_by_pass'):
        expected[key] = source[key] + [old[key][1]]
    require(type(schedule) is dict and set(schedule) == set(expected) | {'contract_sha256'}
            and all(schedule.get(k) == v for k, v in expected.items())
            and type(schedule.get('contract_sha256')) is str and re.fullmatch('[0-9a-f]{64}', schedule['contract_sha256']) is not None
            and all(type(schedule.get(k)) is int for k in ('logical_batches', 'optimizer_updates', 'passes', 'skipped_batches'))
            and all(type(v) is int for v in schedule['logical_step_rows'] + schedule['original_take_indices_appended'])
            and all(type(v) is bool for v in schedule['update_flags'])
            and all(type(take) is list and all(type(v) is int for v in take) for take in schedule['cohort_takes'])
            and plan.get('metadata', {}).get('trajectory', {}).get('schedule') == schedule
            and plan['metadata']['trajectory'].get('original_prefix_trajectory_sha256') == parent_plan['metadata']['trajectory']['contract_sha256'],
            'OBSERVED_CONTINUATION_002_EXACT_864_SCHEDULE')
    require(type(plan.get('limits')) is dict and all(type(v) is int for v in plan['limits'].values())
            and plan['limits'] == parent_plan['limits'] == design.get('limits'), 'OBSERVED_CONTINUATION_002_EXACT_297_216_BUDGET')
    require(type(plan.get('inputs')) is dict and all(plan['inputs'].get(k) == v for k, v in parent_plan['inputs'].items()),
            'OBSERVED_CONTINUATION_002_ALL_OLD_INPUT_ROLES_PRESERVED')
    owners = plan.get('source_owners'); new = {'gx1/audit/native_future_state_observed_train_continuation_002_v1.py',
        'tests/test_native_future_state_observed_train_continuation_002_v1.py', 'tests/test_native_observed_train_continuation_002_guard_v1.py'}
    require(type(owners) is dict and set(parent_plan['source_owners']) | new <= set(owners), 'OBSERVED_CONTINUATION_002_SOURCE_INVENTORY')
    guards = {'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'}
    for owner, digest in owners.items():
        p = repo / owner
        require(type(owner) is str and not Path(owner).is_absolute() and '..' not in Path(owner).parts
                and p.is_file() and not p.is_symlink() and p.resolve() == p
                and hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'OBSERVED_CONTINUATION_002_SOURCE_SHA')
        if owner not in new | guards:
            previous = subprocess.check_output(['git', 'show', _V3_OBSERVED_TRAIN_CONTINUATION_002_PARENT + ':' + owner], cwd=repo)
            require(hashlib.sha256(previous).hexdigest() == digest, 'OBSERVED_CONTINUATION_002_OLD_NON_GUARD_SOURCE_UNCHANGED')
    bridge = plan.get('guard_source_bridge', {}); hashes = {n: v['original_sha256'] for n, v in bridge.get('owners', {}).items()}
    require(bridge == native_v3_observed_train_continuation_002_guard_source_bridge(hashes, producer_commit=bridge.get('producer_commit')),
            'OBSERVED_CONTINUATION_002_STRICT_ADDITIVE_GUARD_BRIDGE')
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = _native_v3_development_read(policy_path)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')
    require(all(policy.get(k) is (not preflight and k in flags[:2]) for k in flags), 'OBSERVED_CONTINUATION_002_EXACT_FLAGS')
    for name, prior in policy.items():
        if name.startswith('native_') and name != _V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE and type(prior) is dict:
            require(all(k not in prior or prior[k] is False for k in ('execution_authorized', 'training_authorized'))
                    and ('remaining_invocations' not in prior or type(prior['remaining_invocations']) is int and prior['remaining_invocations'] == 0),
                    'OBSERVED_CONTINUATION_002_OLD_NATIVE_ROUTES_CLOSED')
            grouped = prior.get('lane_gates', prior.get('phase_gates', {})); require(type(grouped) is dict, 'OBSERVED_CONTINUATION_002_TYPED_OLD_LANES')
            for value in list(grouped.values()) + ([prior['aggregation_completion']] if 'aggregation_completion' in prior else []):
                require(type(value) is dict and type(value.get('remaining_invocations')) is int and value['remaining_invocations'] == 0
                        and all(value.get(k) is False for k in ('execution_authorized', 'training_authorized', 'relaunch_allowed')),
                        'OBSERVED_CONTINUATION_002_OLD_NATIVE_ROLE_GATES_CLOSED')
    previous = policy.get(_V3_OBSERVED_TRAIN_CONTINUATION_STAGE, {})
    require(previous.get('status') == 'CLOSED_CONSUMED_COMPLETE' and previous.get('plan') == parent_ref,
            'OBSERVED_CONTINUATION_002_001_CONSUMED_COMPLETE')
    backup_ref = previous.get('final_backup')
    require(type(backup_ref) is dict and set(backup_ref) == {'path', 'sha256'}, 'OBSERVED_CONTINUATION_002_BACKUP_BINDING')
    backup = _native_v3_development_read(backup_ref['path'], backup_ref['sha256'])
    require(all(backup.get(k) is True for k in ('F_FULLFSYNC', 'separate_mac_disk', 'independent_member_readback'))
            and backup.get('manifest', {}).get('sha256') == backup.get('manifest_sha256'), 'OBSERVED_CONTINUATION_002_VERIFIED_FINAL648_BACKUP')
    manifest_ref = backup['manifest']; manifest = _native_v3_development_read(manifest_ref['path'], manifest_ref['sha256'])
    members = {(v['path'], v['sha256']) for v in manifest['files']}
    require(all((v['path'], v['sha256']) in members for v in (receipt_ref, checkpoint_ref)),
            'OBSERVED_CONTINUATION_002_FINAL648_BACKUP_COVERAGE')
    parent = policy.get(_V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE, {})
    gate = {'status': 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_OBSERVED_TRAIN_CONTINUATION_002',
            'plan': {'path': str(path), 'sha256': plan_sha256}, 'execution_authorized': not preflight,
            'remaining_invocations': int(not preflight), 'relaunch_allowed': False, 'training_authorized': not preflight,
            'test_authorized': False, 'spending_authorized': False}
    require(parent.get('design') == design_ref and parent.get('source_preparation_authorized') is True
            and type(parent.get('remaining_invocations')) is int
            and all(type(parent.get(k)) is bool for k in ('execution_authorized', 'training_authorized', 'test_authorized',
                'spending_authorized', 'relaunch_allowed')) and all(parent.get(k) == v for k, v in gate.items()),
            'OBSERVED_CONTINUATION_002_EXACT_ONE_USE_GATE')
    if not preflight:
        snapshot = evidence / 'V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_20261128_002.json'
        require(_native_v3_development_read(snapshot, hashlib.sha256(policy_path.read_bytes()).hexdigest()) == policy,
                'OBSERVED_CONTINUATION_002_EXACT_ADMITTED_POLICY_SNAPSHOT')
    return {'plan': {'path': str(path), 'sha256': plan_sha256}, 'output_root': str(output), 'preflight': preflight, 'execution': plan['execution']}

def _native_v3_observed_train_continuation_002_environment(plan_sha256):
    require = _native_v3_development_require; prefix = 'GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002'
    execution = _native_v3_development_execution('warm')
    expected = {prefix: 'true', prefix + '_PLAN_SHA256': plan_sha256,
                'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': ''}
    require(all(not k.startswith(prefix) or k in expected for k in os.environ), 'UNKNOWN_OBSERVED_CONTINUATION_002_ENV')
    require(all(v in ('false', '') for k, v in os.environ.items() if k.startswith('GX1_NATIVE_') and not k.startswith(prefix)),
            'OBSERVED_CONTINUATION_002_MIXED_LEGACY_ENV')
    fields = {'MAX_WALL_SECONDS': execution['outer_seconds'], 'MODEL_MAX_WALL_SECONDS': execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED': 'false', 'GPU_INDEX': 0, 'GPU_MAX_POWER_LIMIT_W': 250, 'GPU_MAX_POWER_DRAW_W': 250,
        'GPU_MAX_CORE_TEMP_C': 75, 'GPU_MAX_MEMORY_TEMP_C': 80, 'GPU_MAX_MEMORY_USED_MIB': 12288,
        'GPU_MONITOR_INTERVAL_SECONDS': 1, 'CPU_MAX_TEMP_C': 80, 'CPU_MAX_PACKAGE_POWER_W': 125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES': 8589934592, 'HOST_MIN_COMMIT_HEADROOM_BYTES': 8589934592}
    expected.update({'GX1_TRAINER_' + k: str(v) for k, v in fields.items()})
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'OBSERVED_CONTINUATION_002_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()


def require_native_v3_observed_train_continuation_002_execution(*, plan_sha256):
    bound = _native_v3_observed_train_continuation_002_bound_plan(plan_sha256)
    proof = _native_v3_observed_train_continuation_002_environment(plan_sha256)
    return {**proof, **bound, 'kind': 'provider_hlc_v3_observed_train_continuation_002', 'resource_proof_only': True,
            'training_authority_granted': False, 'optimizer_authority_granted': False}

# END GX1 V3 OBSERVED TRAIN CONTINUATION 002 GUARDS 20261128 001

# BEGIN GX1 V3 FROZEN FINAL864 MODE GAP GUARDS 20261128 001
_V3_FROZEN_FINAL864_MODE_GAP_PARENT = '6ab45cfd46e0391b4f48d746214f0ab4518e7f9b'
_V3_FROZEN_FINAL864_MODE_GAP_NAME = 'NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001'
_V3_FROZEN_FINAL864_MODE_GAP_STAGE = 'native_v3_frozen_final864_mode_gap_20261128_001'
_V3_FROZEN_FINAL864_MODE_GAP_PROFILE = 'gx1_native_frozen_final864_mode_gap_trainer20g_v1'
_V3_FROZEN_FINAL864_MODE_GAP_SHELL_ADDITIONS = (('NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002=false\n', 'before', 'NATIVE_V3_FROZEN_FINAL864_MODE_GAP=false\nNATIVE_V3_FROZEN_FINAL864_MODE_GAP_PLAN_SHA256=\n'), ('# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION 002 ROUTE 20261128 001\n', 'before', '# BEGIN GX1 V3 FROZEN FINAL864 MODE GAP ROUTE 20261128 001\nmatches_bound_native_v3_frozen_final864_mode_gap() {\n  if [[ $# -eq 6 ]]; then\n    [[ "$6" == --preflight ]] || return 1\n    set -- "${@:1:5}"\n  fi\n  [[ $# -eq 5 && "$(pwd -P)" == "$REPO_ROOT" \\\n    && "$1" == "$CANONICAL_TRAINER_PYTHON" && "$2" == -m \\\n    && "$3" == gx1.audit.native_future_state_frozen_final864_mode_gap_v1 \\\n    && "$4" == --v3-frozen-final864-mode-gap-plan-sha256 && "$5" =~ ^[0-9a-f]{64}$ ]] || return 1\n  local expected_plan="$REPO_ROOT/configs/research/NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001.json"\n  [[ -f "$expected_plan" && ! -L "$expected_plan" \\\n    && "$(readlink -f "$expected_plan")" == "$expected_plan" \\\n    && "$(/usr/bin/sha256sum "$expected_plan" | /usr/bin/awk \'{print $1}\')" == "$5" ]]\n}\nconfigure_native_v3_frozen_final864_mode_gap() {\n  matches_bound_native_v3_frozen_final864_mode_gap "$@" || { echo \'FATAL: exact V3 observed TRAIN continuation CLI/plan required\' >&2; exit 75; }\n  [[ "$JOB_CLASS" == trainer && "$MEM" == 20G && "$SWAP" == 512M \\\n    && "$ATTENDED_SMOKE" == false && "$CUDA_PRODUCER_GUARD" == false \\\n    && -z "$POWER_BENCHMARK_SCOPE_JSON" && -z "$POWER_BENCHMARK_SCOPE_SHA256" ]] || {\n    echo \'FATAL: V3 observed TRAIN continuation requires canonical trainer containment\' >&2; exit 75;\n  }\n  local output="/home/andre2/GX1_RUNS/NATIVE_V3_FROZEN_FINAL864_MODE_GAP_20261128_001"\n  [[ ! -e "$output" && ! -L "$output" && -d "${output%/*}" \\\n    && ! -L "${output%/*}" && "$(readlink -f "${output%/*}")" == "${output%/*}" ]] || {\n    echo \'FATAL: V3 observed TRAIN continuation output must be fresh and canonical\' >&2; exit 75;\n  }\n  NATIVE_V3_FROZEN_FINAL864_MODE_GAP=true\n  NATIVE_V3_FROZEN_FINAL864_MODE_GAP_PLAN_SHA256="$5"\n  TRAINER_DEVICE=cuda\n  TRAINER_EXECUTION_MODE=canonical\n  TRAINER_OUT_BUNDLE_DIR="$output"\n  TRAINER_MAX_WALL_SECONDS=1800\n  TRAINER_MODEL_MAX_WALL_SECONDS=1770\n  TRAINER_GPU_INDEX=0\n  TRAINER_GPU_MAX_POWER_LIMIT_W=250\n  TRAINER_GPU_MAX_POWER_DRAW_W=250\n  TRAINER_GPU_MAX_CORE_TEMP_C=75\n  TRAINER_GPU_MAX_MEMORY_TEMP_C=80\n  TRAINER_GPU_MAX_MEMORY_USED_MIB=12288\n  TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1\n  TRAINER_CPU_MAX_TEMP_C=80\n  TRAINER_CPU_MAX_PACKAGE_POWER_W=125\n  TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592\n  TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592\n}\n# END GX1 V3 FROZEN FINAL864 MODE GAP ROUTE 20261128 001\n\n'), ('# BEGIN GX1 V3 OBSERVED TRAIN CONTINUATION 002 DISPATCH 20261128 001\n', 'before', '# BEGIN GX1 V3 FROZEN FINAL864 MODE GAP DISPATCH 20261128 001\n  local frozen_final864_mode_gap_arg\n  for frozen_final864_mode_gap_arg in "$@"; do\n    case "$frozen_final864_mode_gap_arg" in\n      --v3-frozen-final864-mode-gap-plan-sha256|--v3-frozen-final864-mode-gap-plan-sha256=*) configure_native_v3_frozen_final864_mode_gap "$@"; return ;;\n    esac\n  done\n# END GX1 V3 FROZEN FINAL864 MODE GAP DISPATCH 20261128 001\n'), ('  --setenv=GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002="$NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002" \\\n', 'before', '  --setenv=GX1_NATIVE_V3_FROZEN_FINAL864_MODE_GAP="$NATIVE_V3_FROZEN_FINAL864_MODE_GAP" \\\n  --setenv=GX1_NATIVE_V3_FROZEN_FINAL864_MODE_GAP_PLAN_SHA256="$NATIVE_V3_FROZEN_FINAL864_MODE_GAP_PLAN_SHA256" \\\n'))


def _native_v3_frozen_final864_mode_gap_without_additions(current):
    require = _native_v3_development_require; current = dict(current)
    name = 'gx1/contracts/gx1_capped_execution_v1.py'
    begin = '# BEGIN GX1 V3 FROZEN FINAL864 MODE GAP GUARDS 20261128 001\n'
    end = '# END GX1 V3 FROZEN FINAL864 MODE GAP GUARDS 20261128 001\n\n'
    text = current[name]; require(text.count(begin) == text.count(end) == 1, 'ONE_FROZEN_FINAL864_MODE_GAP_BLOCK')
    first, tail = text.split(begin); _, last = tail.split(end); text = first + last
    hook = '    current=_native_v3_frozen_final864_mode_gap_without_additions(current)\n'
    require(text.count(hook) == 1, 'ONE_FROZEN_FINAL864_MODE_GAP_BRIDGE_HOOK')
    current[name] = text.replace(hook, '', 1)
    shell = current['scripts/gx1_capped_run.sh']
    for anchor, position, fragment in _V3_FROZEN_FINAL864_MODE_GAP_SHELL_ADDITIONS:
        require(position == 'before' and shell.count(anchor) == shell.count(fragment)
                == shell.count(fragment + anchor) == 1, 'EXACT_FROZEN_FINAL864_MODE_GAP_SHELL_ADDITION')
        shell = shell.replace(fragment + anchor, anchor, 1)
    current['scripts/gx1_capped_run.sh'] = shell
    return current


def native_v3_frozen_final864_mode_gap_guard_source_bridge(producer_hashes, *, producer_commit=_V3_FROZEN_FINAL864_MODE_GAP_PARENT):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(producer_commit == _V3_FROZEN_FINAL864_MODE_GAP_PARENT, 'FROZEN_FINAL864_MODE_GAP_BRIDGE_PARENT')
    names = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')
    before = {n: subprocess.check_output(['git', 'show', producer_commit + ':' + n], cwd=repo).decode() for n in names}
    original = {n: hashlib.sha256(v.encode()).hexdigest() for n, v in before.items()}
    require(producer_hashes == original, 'FROZEN_FINAL864_MODE_GAP_BRIDGE_PRODUCER_HASHES')
    current = {n: (repo / n).read_text() for n in names}
    require(_native_v3_frozen_final864_mode_gap_without_additions(current) == before, 'FROZEN_FINAL864_MODE_GAP_OLD_GUARDS_BYTE_EXACT')
    immutable = ('gx1/audit/native_provider_hlc_learning_v1.py',
        'gx1/audit/native_future_state_observed_train_continuation_v1.py',
        'gx1/audit/native_future_state_observed_train_continuation_002_v1.py',
        'gx1/audit/native_future_state_step_observer_qualify_v1.py', 'gx1/audit/native_future_state_step_observer_v1.py',
        'gx1/audit/native_future_state_training_v1.py', 'gx1/audit/qualify_native_m5_full_gc_direction_v1.py',
        'gx1/contracts/market_state_ta_integrated_model_v1.py', 'gx1/contracts/market_state_ta_integrated_objective_v1.py')
    proof = {}
    for name in immutable:
        old = subprocess.check_output(['git', 'show', producer_commit + ':' + name], cwd=repo); path = repo / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path and path.read_bytes() == old,
                'FROZEN_FINAL864_MODE_GAP_IMMUTABLE_NN_OR_LOOP_OWNER')
        proof[name] = hashlib.sha256(old).hexdigest()
    return {'producer_commit': producer_commit, 'checked_guard_owners': 2, 'old_guard_routes_preserved': True,
            'old_numeric_resource_environment_function_unchanged': True, 'old_NN_and_mode_gap_loop_byte_unchanged': proof,
            'new_864_diagnostic_controlflow_requires_scoped_review': True,
            'owners': {n: {'original_sha256': original[n], 'current_sha256': hashlib.sha256(current[n].encode()).hexdigest()} for n in names}}


def _native_v3_frozen_final864_mode_gap_bound_plan(plan_sha256):
    import hashlib, subprocess
    require = _native_v3_development_require; repo = Path(__file__).resolve().parents[2]
    require(type(plan_sha256) is str and re.fullmatch('[0-9a-f]{64}', plan_sha256) is not None, 'FROZEN_FINAL864_PLAN_SHA')
    args = ['--v3-frozen-final864-mode-gap-plan-sha256', plan_sha256]; preflight = sys.argv[-1:] == ['--preflight']
    if preflight: args += ['--preflight']
    require(Path.cwd().resolve() == repo and Path(sys.argv[0]).resolve() == repo / 'gx1/audit/native_future_state_frozen_final864_mode_gap_v1.py'
            and Path(sys.executable).absolute() == repo / '.venv/bin/python' and sys.argv[1:] == args, 'FROZEN_FINAL864_EXACT_ARGV')
    path = repo / 'configs/research' / (_V3_FROZEN_FINAL864_MODE_GAP_NAME + '.json')
    plan = _native_v3_development_read(path, plan_sha256)
    output = Path('/home/andre2/GX1_RUNS') / _V3_FROZEN_FINAL864_MODE_GAP_NAME
    require(plan.get('schema_version') == 'gx1_native_v3_frozen_final864_mode_gap_plan_v1'
            and plan.get('stage') == _V3_FROZEN_FINAL864_MODE_GAP_STAGE and plan.get('output_root') == str(output), 'FROZEN_FINAL864_PLAN_IDENTITY')
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
            and not output.parent.is_symlink() and output.parent.resolve() == output.parent, 'FROZEN_FINAL864_FRESH_OUTPUT')
    require(plan.get('source_commit') == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'FROZEN_FINAL864_SOURCE_COMMIT')
    require(plan.get('execution') == _native_v3_development_execution('warm')
            and plan.get('execution_profile') == _V3_FROZEN_FINAL864_MODE_GAP_PROFILE, 'FROZEN_FINAL864_EXACT_EXECUTION')
    evidence = Path('/home/andre2/GX1_RUNS') / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = {'path': str(evidence / 'V3_FROZEN_FINAL864_MODE_GAP_DESIGN_20261128_001.json'),
                  'sha256': '3b49a1de3bb2bc257e157d9fd8de137b57121d94fe30b76c75edac0b7519c097'}
    parent_ref = {'path': str(repo / 'configs/research' / (_V3_OBSERVED_TRAIN_CONTINUATION_002_NAME + '.json')),
                  'sha256': 'be53cb01fce7c80f8ba120c88de2f36d299112f20a573e85b85cda82d8db6875'}
    original_ref = {'path': str(repo / 'configs/research' / (_V3_ISSUE_MIX_NAME + '_FIT.json')),
                    'sha256': '25ca4cc9d780f5d16376a765ec58f6d84c58a1f16f55d48033d11168fda2d4a8'}
    final_root = Path('/home/andre2/GX1_RUNS') / _V3_OBSERVED_TRAIN_CONTINUATION_002_NAME
    receipt_ref = {'path': str(final_root / 'FINAL864.json'), 'sha256': '06b48e4fa08306f1979d201c334573adaf80f9f6d9f4b05dfd220c15f69a605d'}
    checkpoint_ref = {'path': str(final_root / 'FINAL864.pt'), 'sha256': '26bbcc969348592b5cdea4ee85fe5629d9ed6250bd1aa9415e570fdc46250368'}
    require(plan.get('design') == design_ref and plan.get('producer_plan') == parent_ref
            and plan.get('original_producer_plan') == original_ref and plan.get('authentic_final_receipt') == receipt_ref
            and plan.get('authentic_final_checkpoint') == checkpoint_ref, 'FROZEN_FINAL864_OWN864_BOUND_REFS')
    design = _native_v3_development_read(design_ref['path'], design_ref['sha256'])
    parent_plan = _native_v3_development_read(parent_ref['path'], parent_ref['sha256'])
    original = _native_v3_development_read(original_ref['path'], original_ref['sha256'])
    receipt = _native_v3_development_read(receipt_ref['path'], receipt_ref['sha256'])
    require(type(receipt.get('logical_cursor')) is int and type(receipt.get('optimizer_updates')) is int
            and receipt['logical_cursor'] == receipt['optimizer_updates'] == 864 and receipt.get('producer_lane') == 'continuation'
            and receipt.get('original_plan') == receipt.get('execution_plan') == parent_ref and receipt.get('checkpoint') == checkpoint_ref
            and receipt.get('trajectory_sha256') == parent_plan.get('metadata', {}).get('trajectory', {}).get('contract_sha256')
            and plan.get('metadata', {}).get('parent_plan') == parent_ref and plan.get('metadata', {}).get('original_plan') == original_ref
            and plan.get('metadata', {}).get('receipt') == receipt
            and plan.get('metadata', {}).get('trajectory') == parent_plan.get('metadata', {}).get('trajectory')
            and parent_plan.get('original_producer_plan') == original_ref
            and plan.get('recipe') == parent_plan.get('recipe') == original.get('recipe')
            and plan.get('model_contract') == parent_plan.get('model_contract') == original.get('model_contract')
            and plan.get('physical_input_identity') == parent_plan.get('physical_input_identity')
            and 'backbone_gradient_policy' in plan and plan['backbone_gradient_policy'] is None, 'FROZEN_FINAL864_IMMUTABLE864_FUNCTION')
    require(type(plan.get('actual_global_cursor')) is int and plan['actual_global_cursor'] == 864
            and plan.get('numeric_owner_phase') == 'PASS2' and plan.get('passes') == design.get('passes')
            and all(type(v.get(k)) is bool for v in plan['passes'] for k in ('training', 'equal_tf')), 'FROZEN_FINAL864_EXACT_FIVE_PASSES')
    require(type(plan.get('limits')) is dict and all(type(v) is int for v in plan['limits'].values())
            and plan['limits'] == design.get('limits'), 'FROZEN_FINAL864_405_FORWARD_ZERO_UPDATE_BUDGET')
    inputs = plan.get('inputs'); require(type(inputs) is dict and all(inputs.get(k) == v for k, v in parent_plan['inputs'].items()),
            'FROZEN_FINAL864_ALL_OLD_INPUT_ROLES_PRESERVED')
    for key in ('saved_eval_logits', 'saved_eval_numeric_owner_score', 'saved_eval_score_wrapper'):
        require(inputs.get('frozen_final864_mode_gap_' + key) == design.get(key), 'FROZEN_FINAL864_SAVED_EVAL_ROLE_BINDINGS')
    owners = plan.get('source_owners'); new = {'gx1/audit/native_future_state_frozen_final864_mode_gap_v1.py',
        'tests/test_native_future_state_frozen_final864_mode_gap_v1.py', 'tests/test_native_frozen_final864_mode_gap_guard_v1.py'}
    require(type(owners) is dict and set(parent_plan['source_owners']) | new <= set(owners), 'FROZEN_FINAL864_SOURCE_INVENTORY')
    guards = {'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'}
    for owner, digest in owners.items():
        p = repo / owner
        require(type(owner) is str and not Path(owner).is_absolute() and '..' not in Path(owner).parts
                and p.is_file() and not p.is_symlink() and p.resolve() == p
                and hashlib.sha256(p.read_bytes()).hexdigest() == digest, 'FROZEN_FINAL864_SOURCE_SHA')
        if owner not in new | guards:
            previous = subprocess.check_output(['git', 'show', _V3_FROZEN_FINAL864_MODE_GAP_PARENT + ':' + owner], cwd=repo)
            require(hashlib.sha256(previous).hexdigest() == digest, 'FROZEN_FINAL864_OLD_NON_GUARD_SOURCE_UNCHANGED')
    bridge = plan.get('guard_source_bridge', {}); hashes = {n: v['original_sha256'] for n, v in bridge.get('owners', {}).items()}
    require(bridge == native_v3_frozen_final864_mode_gap_guard_source_bridge(hashes, producer_commit=bridge.get('producer_commit')),
            'FROZEN_FINAL864_STRICT_ADDITIVE_GUARD_BRIDGE')
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = _native_v3_development_read(policy_path)
    flags = ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
             'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave')
    require(all(policy.get(k) is False for k in flags), 'FROZEN_FINAL864_ALL_SIX_GLOBAL_FLAGS_FALSE')
    for name, prior in policy.items():
        if name.startswith('native_') and name != _V3_FROZEN_FINAL864_MODE_GAP_STAGE and type(prior) is dict:
            require(all(k not in prior or prior[k] is False for k in ('execution_authorized', 'training_authorized'))
                    and ('remaining_invocations' not in prior or type(prior['remaining_invocations']) is int and prior['remaining_invocations'] == 0),
                    'FROZEN_FINAL864_OLD_NATIVE_ROUTES_CLOSED')
            grouped = prior.get('lane_gates', prior.get('phase_gates', {})); require(type(grouped) is dict, 'FROZEN_FINAL864_TYPED_OLD_LANES')
            for value in list(grouped.values()) + ([prior['aggregation_completion']] if 'aggregation_completion' in prior else []):
                require(type(value) is dict and type(value.get('remaining_invocations')) is int and value['remaining_invocations'] == 0
                        and all(value.get(k) is False for k in ('execution_authorized', 'training_authorized', 'relaunch_allowed')),
                        'FROZEN_FINAL864_OLD_NATIVE_ROLE_GATES_CLOSED')
    previous = policy.get(_V3_OBSERVED_TRAIN_CONTINUATION_002_STAGE, {})
    require(previous.get('status') == 'CLOSED_CONSUMED_COMPLETE' and previous.get('plan') == parent_ref, 'FROZEN_FINAL864_002_CONSUMED_COMPLETE')
    backup_ref = previous.get('final_backup')
    require(type(backup_ref) is dict and set(backup_ref) == {'path', 'sha256'}, 'FROZEN_FINAL864_BACKUP_BINDING')
    backup = _native_v3_development_read(backup_ref['path'], backup_ref['sha256'])
    require(all(backup.get(k) is True for k in ('F_FULLFSYNC', 'separate_mac_disk', 'independent_member_readback'))
            and backup.get('manifest', {}).get('sha256') == backup.get('manifest_sha256'), 'FROZEN_FINAL864_VERIFIED_FINAL864_BACKUP')
    manifest_ref = backup['manifest']; manifest = _native_v3_development_read(manifest_ref['path'], manifest_ref['sha256'])
    members = {(v['path'], v['sha256']) for v in manifest['files']}
    require(all((v['path'], v['sha256']) in members for v in (receipt_ref, checkpoint_ref)), 'FROZEN_FINAL864_BACKUP_COVERAGE')
    parent = policy.get(_V3_FROZEN_FINAL864_MODE_GAP_STAGE, {})
    gate = {'status': 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_FROZEN_FINAL864_MODE_GAP',
            'plan': {'path': str(path), 'sha256': plan_sha256}, 'execution_authorized': not preflight,
            'remaining_invocations': int(not preflight), 'relaunch_allowed': False, 'training_authorized': False,
            'test_authorized': False, 'spending_authorized': False}
    require(parent.get('design') == design_ref and parent.get('source_preparation_authorized') is True
            and type(parent.get('remaining_invocations')) is int
            and all(type(parent.get(k)) is bool for k in ('execution_authorized', 'training_authorized', 'test_authorized',
                'spending_authorized', 'relaunch_allowed')) and all(parent.get(k) == v for k, v in gate.items()), 'FROZEN_FINAL864_EXACT_EXECUTION_ONLY_GATE')
    if not preflight:
        snapshot = evidence / 'V3_FROZEN_FINAL864_MODE_GAP_ADMITTED_POLICY_20261128_001.json'
        require(_native_v3_development_read(snapshot, hashlib.sha256(policy_path.read_bytes()).hexdigest()) == policy,
                'FROZEN_FINAL864_EXACT_ADMITTED_POLICY_SNAPSHOT')
    return {'plan': {'path': str(path), 'sha256': plan_sha256}, 'output_root': str(output), 'preflight': preflight, 'execution': plan['execution']}

def _native_v3_frozen_final864_mode_gap_environment(plan_sha256):
    require = _native_v3_development_require; prefix = 'GX1_NATIVE_V3_FROZEN_FINAL864_MODE_GAP'
    execution = _native_v3_development_execution('warm')
    expected = {prefix: 'true', prefix + '_PLAN_SHA256': plan_sha256,
                'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': ''}
    require(all(not k.startswith(prefix) or k in expected for k in os.environ), 'UNKNOWN_FROZEN_FINAL864_MODE_GAP_ENV')
    require(all(v in ('false', '') for k, v in os.environ.items() if k.startswith('GX1_NATIVE_') and not k.startswith(prefix)),
            'FROZEN_FINAL864_MODE_GAP_MIXED_LEGACY_ENV')
    fields = {'MAX_WALL_SECONDS': execution['outer_seconds'], 'MODEL_MAX_WALL_SECONDS': execution['internal_seconds'],
        'ATTENDED_STAGE_REQUIRED': 'false', 'GPU_INDEX': 0, 'GPU_MAX_POWER_LIMIT_W': 250, 'GPU_MAX_POWER_DRAW_W': 250,
        'GPU_MAX_CORE_TEMP_C': 75, 'GPU_MAX_MEMORY_TEMP_C': 80, 'GPU_MAX_MEMORY_USED_MIB': 12288,
        'GPU_MONITOR_INTERVAL_SECONDS': 1, 'CPU_MAX_TEMP_C': 80, 'CPU_MAX_PACKAGE_POWER_W': 125,
        'HOST_MIN_AVAILABLE_MEMORY_BYTES': 8589934592, 'HOST_MIN_COMMIT_HEADROOM_BYTES': 8589934592}
    expected.update({'GX1_TRAINER_' + k: str(v) for k, v in fields.items()})
    require(all(os.environ.get(k) == v for k, v in expected.items()), 'FROZEN_FINAL864_MODE_GAP_EXACT_RESOURCE_ENV')
    return require_guarded_cuda_trainer_execution()


def require_native_v3_frozen_final864_mode_gap_execution(*, plan_sha256):
    bound = _native_v3_frozen_final864_mode_gap_bound_plan(plan_sha256)
    proof = _native_v3_frozen_final864_mode_gap_environment(plan_sha256)
    return {**proof, **bound, 'kind': 'provider_hlc_v3_frozen_final864_mode_gap', 'resource_proof_only': True,
            'training_authority_granted': False, 'optimizer_authority_granted': False}

# END GX1 V3 FROZEN FINAL864 MODE GAP GUARDS 20261128 001

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
