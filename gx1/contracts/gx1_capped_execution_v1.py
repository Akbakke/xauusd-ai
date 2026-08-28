"""Fail-closed proof for CUDA-capable non-trainer GX1 entry points.

The shell runner owns process containment and the telemetry guard. A Python
module that can allocate CUDA must nevertheless prove that it is already
inside that protected cgroup before it asks PyTorch about a device.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping


_MAX_MEMORY_BYTES = 20 * 1024**3
_MAX_SWAP_BYTES = 512 * 1024**2
_MAX_PIDS = 64
_LIMIT_ENV = {
    "memory": "GX1_CAPPED_MEMORY_BYTES",
    "swap": "GX1_CAPPED_SWAP_BYTES",
    "pids": "GX1_CAPPED_TASKS_MAX",
}


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

    expected: dict[str, int] = {}
    for label, name in _LIMIT_ENV.items():
        raw = str(env.get(name) or "")
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(f"[GX1_CUDA_PRODUCER_ENV_PROOF_INVALID] field={name}")
        expected[label] = int(raw)
    if (
        expected["memory"] > _MAX_MEMORY_BYTES
        or expected["swap"] > _MAX_SWAP_BYTES
        or expected["pids"] > _MAX_PIDS
    ):
        raise RuntimeError("[GX1_CUDA_PRODUCER_ENV_LIMIT_EXCEEDED]")

    reader = read_text or (lambda path: path.read_text(encoding="utf-8"))
    try:
        cgroup_lines = str(reader(Path("/proc/self/cgroup"))).splitlines()
    except Exception as exc:
        raise RuntimeError("[GX1_CUDA_PRODUCER_CGROUP_PATH_UNAVAILABLE]") from exc
    unified = [
        line.split(":", 2)[2]
        for line in cgroup_lines
        if len(line.split(":", 2)) == 3
        and line.split(":", 2)[0] == "0"
        and line.split(":", 2)[1] == ""
    ]
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise RuntimeError("[GX1_CUDA_PRODUCER_CGROUP_PATH_INVALID]")
    relative_parts = Path(unified[0]).parts[1:]
    if (
        not relative_parts
        or not relative_parts[-1].endswith(".scope")
        or any(part in {"", ".", ".."} for part in relative_parts)
    ):
        raise RuntimeError("[GX1_CUDA_PRODUCER_CGROUP_PATH_INVALID]")
    cgroup_dir = Path("/sys/fs/cgroup").joinpath(*relative_parts)

    def _read_limit(name: str) -> int:
        try:
            raw = str(reader(cgroup_dir / name)).strip()
        except Exception as exc:
            raise RuntimeError(
                f"[GX1_CUDA_PRODUCER_CGROUP_LIMIT_UNAVAILABLE] field={name}"
            ) from exc
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            raise RuntimeError(
                f"[GX1_CUDA_PRODUCER_CGROUP_LIMIT_INVALID] field={name}"
            )
        return int(raw)

    actual = {
        "memory_max": _read_limit("memory.max"),
        "memory_high": _read_limit("memory.high"),
        "swap": _read_limit("memory.swap.max"),
        "pids": _read_limit("pids.max"),
    }
    if (
        actual["memory_max"] > _MAX_MEMORY_BYTES
        or actual["memory_high"] > _MAX_MEMORY_BYTES
        or actual["swap"] > _MAX_SWAP_BYTES
        or actual["pids"] > _MAX_PIDS
    ):
        raise RuntimeError("[GX1_CUDA_PRODUCER_CGROUP_ACTUAL_LIMIT_EXCEEDED]")
    if (
        actual["memory_max"] != expected["memory"]
        or actual["memory_high"] != expected["memory"]
        or actual["swap"] != expected["swap"]
        or actual["pids"] != expected["pids"]
    ):
        raise RuntimeError("[GX1_CUDA_PRODUCER_CGROUP_ENV_ACTUAL_MISMATCH]")
    return {
        "class": "producer",
        "execution_mode": "cuda_producer",
        "cgroup_path": str(cgroup_dir),
        **actual,
    }
