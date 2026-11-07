"""Focused synthetic proof of the exact CPU market-state research cap."""
import pytest

import gx1.contracts.gx1_capped_execution_v1 as capped

MEMORY = 12 * 1024**3
SWAP = 512 * 1024**2
THREAD_NAMES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                "ARROW_NUM_THREADS", "POLARS_MAX_THREADS")


def _fixture():
    env = {
        "GX1_CAPPED_CLASS": "producer", "GX1_CUDA_PRODUCER_GUARD": "false",
        "GX1_TRAINER_DEVICE": "", "GX1_TRAINER_EXECUTION_MODE": "canonical",
        "GX1_CAPPED_MEMORY_BYTES": str(MEMORY),
        "GX1_CAPPED_SWAP_BYTES": str(SWAP), "GX1_CAPPED_TASKS_MAX": "64",
        **{name: "1" for name in THREAD_NAMES},
    }
    root = "/sys/fs/cgroup/user.slice/gx1-cpu-research.scope"
    files = {
        "/proc/self/cgroup": "0::/user.slice/gx1-cpu-research.scope\n",
        f"{root}/memory.max": str(MEMORY),
        f"{root}/memory.high": str(MEMORY),
        f"{root}/memory.swap.max": str(SWAP),
        f"{root}/pids.max": "64",
        "/proc/self/status": "Name:\tpython\nCpus_allowed_list:\t0-7\n",
    }
    return env, files


def test_cpu_research_producer_requires_exact_cgroup_affinity_and_lock(monkeypatch):
    env, files = _fixture()
    calls = []
    monkeypatch.setattr(capped, "require_capped_lock_ancestry",
                        lambda: calls.append("lock") or {"owner_pid": 123})
    proof = capped.require_capped_cpu_research_producer_execution(
        environ=env, read_text=lambda path: files[str(path)])
    assert proof == {
        "class": "producer", "execution_mode": "cpu_research",
        "cgroup_path": "/sys/fs/cgroup/user.slice/gx1-cpu-research.scope",
        "memory_max": MEMORY, "memory_high": MEMORY, "swap": SWAP, "pids": 64,
        "cpu_affinity": list(range(8)), "numerical_threads": 1,
        "lock": {"owner_pid": 123},
    }
    assert calls == ["lock"]


@pytest.mark.parametrize("name,value", [
    ("GX1_CAPPED_CLASS", "audit"),
    ("GX1_CUDA_PRODUCER_GUARD", "true"),
    ("GX1_TRAINER_DEVICE", "cpu"),
    ("GX1_CAPPED_MEMORY_BYTES", str(20 * 1024**3)),
    ("GX1_CAPPED_SWAP_BYTES", str(1024 * 1024**2)),
    ("GX1_CAPPED_TASKS_MAX", "128"),
    ("OMP_NUM_THREADS", "8"),
    ("ARROW_NUM_THREADS", "2"),
])
def test_cpu_research_producer_rejects_wrong_env_before_cgroup(name, value):
    env, _ = _fixture()
    env[name] = value
    with pytest.raises(RuntimeError, match=f"GX1_CPU_RESEARCH_{name}_INVALID"):
        capped.require_capped_cpu_research_producer_execution(
            environ=env, read_text=lambda _path: pytest.fail("cgroup read must not occur"))


@pytest.mark.parametrize("key,value,error", [
    ("memory.max", str(8 * 1024**3), "CGROUP_ENV_ACTUAL_MISMATCH"),
    ("memory.high", str(8 * 1024**3), "CGROUP_ENV_ACTUAL_MISMATCH"),
    ("memory.swap.max", str(256 * 1024**2), "CGROUP_ENV_ACTUAL_MISMATCH"),
    ("pids.max", "32", "CGROUP_ENV_ACTUAL_MISMATCH"),
])
def test_cpu_research_producer_rejects_actual_limit_mismatch(monkeypatch, key, value, error):
    env, files = _fixture()
    path = next(path for path in files if path.endswith("/" + key))
    files[path] = value
    monkeypatch.setattr(capped, "require_capped_lock_ancestry", lambda: pytest.fail("lock must not be consulted"))
    with pytest.raises(RuntimeError, match="GX1_CPU_RESEARCH_" + error):
        capped.require_capped_cpu_research_producer_execution(
            environ=env, read_text=lambda path: files[str(path)])


@pytest.mark.parametrize("affinity", ["0-6", "1-7", "0,2-7", "0-18", "", "8"])
def test_cpu_research_producer_rejects_other_affinity(monkeypatch, affinity):
    env, files = _fixture()
    files["/proc/self/status"] = f"Cpus_allowed_list:\t{affinity}\n"
    monkeypatch.setattr(capped, "require_capped_lock_ancestry", lambda: pytest.fail("lock must not be consulted"))
    with pytest.raises(RuntimeError, match="GX1_CPU_RESEARCH_AFFINITY_INVALID"):
        capped.require_capped_cpu_research_producer_execution(
            environ=env, read_text=lambda path: files[str(path)])


def test_cpu_research_producer_fails_if_lock_proof_fails(monkeypatch):
    env, files = _fixture()
    def bad_lock():
        raise RuntimeError("GX1_CAPPED_LOCK_PROOF_INVALID")
    monkeypatch.setattr(capped, "require_capped_lock_ancestry", bad_lock)
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        capped.require_capped_cpu_research_producer_execution(
            environ=env, read_text=lambda path: files[str(path)])
