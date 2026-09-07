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
    lock_path = runtime / "gx1-heavy-job.lock"
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
    """Use the protected per-UID runtime directory, never ambient XDG or /tmp."""

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
        or expected["pids"] > _MAX_PIDS
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
        or actual["pids"] > _MAX_PIDS
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
