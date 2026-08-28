from __future__ import annotations

import pytest

from gx1.contracts.gx1_capped_execution_v1 import (
    require_guarded_cuda_producer_execution,
)


def _guarded_fixture() -> tuple[dict[str, str], dict[str, str]]:
    memory = 10 * 1024**3
    swap = 512 * 1024**2
    env = {
        "GX1_CAPPED_CLASS": "producer",
        "GX1_CUDA_PRODUCER_GUARD": "true",
        "GX1_TRAINER_DEVICE": "cuda",
        "GX1_TRAINER_EXECUTION_MODE": "cuda_producer",
        "GX1_CAPPED_MEMORY_BYTES": str(memory),
        "GX1_CAPPED_SWAP_BYTES": str(swap),
        "GX1_CAPPED_TASKS_MAX": "64",
    }
    files = {
        "/proc/self/cgroup": "0::/gx1-cuda-test.scope\n",
        "/sys/fs/cgroup/gx1-cuda-test.scope/memory.max": str(memory),
        "/sys/fs/cgroup/gx1-cuda-test.scope/memory.high": str(memory),
        "/sys/fs/cgroup/gx1-cuda-test.scope/memory.swap.max": str(swap),
        "/sys/fs/cgroup/gx1-cuda-test.scope/pids.max": "64",
    }
    return env, files


def test_cuda_producer_requires_guarded_cgroup_not_environment_alone() -> None:
    env, files = _guarded_fixture()

    def read_text(path) -> str:
        return files[str(path)]

    proof = require_guarded_cuda_producer_execution(
        environ=env,
        read_text=read_text,
    )
    assert proof["class"] == "producer"
    assert proof["memory_max"] == proof["memory_high"] == 10 * 1024**3

    with pytest.raises(RuntimeError, match="GX1_CUDA_PRODUCER_CGROUP_ENV_ACTUAL_MISMATCH"):
        require_guarded_cuda_producer_execution(
            environ={**env, "GX1_CAPPED_MEMORY_BYTES": str(9 * 1024**3)},
            read_text=read_text,
        )


def test_cuda_producer_rejects_direct_or_wrong_class_before_cgroup_read() -> None:
    env, _files = _guarded_fixture()

    with pytest.raises(RuntimeError, match="GX1_CUDA_PRODUCER_GX1_CAPPED_CLASS_INVALID"):
        require_guarded_cuda_producer_execution(
            environ={**env, "GX1_CAPPED_CLASS": "audit"},
            read_text=lambda _path: pytest.fail("must not read cgroup"),
        )
