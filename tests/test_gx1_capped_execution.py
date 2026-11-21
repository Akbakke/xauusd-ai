from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import gx1.contracts.gx1_capped_execution_v1 as capped_execution
from gx1.contracts.gx1_capped_execution_v1 import (
    canonical_heavy_job_lock_path,
    require_capped_cpu_audit_execution,
    require_capped_lock_ancestry,
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


def _cpu_audit_fixture() -> tuple[dict[str, str], dict[str, str]]:
    memory = 4 * 1024**3
    swap = 512 * 1024**2
    env = {
        "GX1_CAPPED_CLASS": "audit",
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
    files = {
        "/proc/self/cgroup": "0::/user.slice/gx1-audit-test.scope\n",
        "/sys/fs/cgroup/user.slice/gx1-audit-test.scope/memory.max": str(memory),
        "/sys/fs/cgroup/user.slice/gx1-audit-test.scope/memory.high": str(memory),
        "/sys/fs/cgroup/user.slice/gx1-audit-test.scope/memory.swap.max": str(swap),
        "/sys/fs/cgroup/user.slice/gx1-audit-test.scope/pids.max": "64",
        "/proc/self/status": "Name:\tpython\nCpus_allowed_list:\t0-7\n",
    }
    return env, files


def test_cpu_audit_requires_actual_cgroup_and_affinity_not_environment_alone() -> None:
    env, files = _cpu_audit_fixture()
    reads = []

    def read_text(path) -> str:
        reads.append(str(path))
        return files[str(path)]

    assert "GX1_CPU_AFFINITY" not in env
    proof = require_capped_cpu_audit_execution(environ=env, read_text=read_text)
    assert proof == {
        "class": "audit",
        "execution_mode": "cpu_audit",
        "cgroup_path": "/sys/fs/cgroup/user.slice/gx1-audit-test.scope",
        "memory_max": 4 * 1024**3,
        "memory_high": 4 * 1024**3,
        "swap": 512 * 1024**2,
        "pids": 64,
        "cpu_affinity": list(range(8)),
        "numerical_threads": 1,
    }
    assert reads == list(files)


@pytest.mark.parametrize("affinity, expected", [("0,2-3,7", [0, 2, 3, 7]), ("7", [7])])
def test_cpu_audit_accepts_stricter_bound_limits_and_sparse_affinity(affinity, expected) -> None:
    env, files = _cpu_audit_fixture()
    env.update({
        "GX1_CAPPED_MEMORY_BYTES": str(2 * 1024**3),
        "GX1_CAPPED_SWAP_BYTES": str(256 * 1024**2),
        "GX1_CAPPED_TASKS_MAX": "32",
    })
    for filename, value in (
        ("memory.max", env["GX1_CAPPED_MEMORY_BYTES"]),
        ("memory.high", env["GX1_CAPPED_MEMORY_BYTES"]),
        ("memory.swap.max", env["GX1_CAPPED_SWAP_BYTES"]),
        ("pids.max", env["GX1_CAPPED_TASKS_MAX"]),
    ):
        files[f"/sys/fs/cgroup/user.slice/gx1-audit-test.scope/{filename}"] = value
    files["/proc/self/status"] = f"Cpus_allowed_list:\t{affinity}\n"
    proof = require_capped_cpu_audit_execution(
        environ=env, read_text=lambda path: files[str(path)],
    )
    assert proof["memory_max"] == proof["memory_high"] == 2 * 1024**3
    assert proof["swap"] == 256 * 1024**2
    assert proof["pids"] == 32
    assert proof["cpu_affinity"] == expected


@pytest.mark.parametrize("name, value", [
    ("GX1_CAPPED_CLASS", None),
    ("GX1_CAPPED_CLASS", "producer"),
    ("GX1_CAPPED_CLASS", "trainer"),
    ("GX1_CUDA_PRODUCER_GUARD", None),
    ("GX1_CUDA_PRODUCER_GUARD", "true"),
    ("GX1_TRAINER_DEVICE", None),
    ("GX1_TRAINER_DEVICE", "cpu"),
    ("GX1_TRAINER_DEVICE", "cuda"),
    ("GX1_TRAINER_EXECUTION_MODE", None),
    ("GX1_TRAINER_EXECUTION_MODE", "cuda_producer"),
])
def test_cpu_audit_rejects_missing_or_wrong_runner_mode_before_read(name, value) -> None:
    env, _files = _cpu_audit_fixture()
    if value is None:
        env.pop(name)
    else:
        env[name] = value
    with pytest.raises(RuntimeError, match=rf"\[GX1_CPU_AUDIT_{name}_INVALID\]"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda _path: pytest.fail("must not read cgroup"),
        )


@pytest.mark.parametrize("name", [
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "ARROW_NUM_THREADS", "POLARS_MAX_THREADS",
])
@pytest.mark.parametrize("value", [None, "", "0", "2", "01", " 1"])
def test_cpu_audit_requires_every_exact_single_thread_marker(name, value) -> None:
    env, _files = _cpu_audit_fixture()
    if value is None:
        env.pop(name)
    else:
        env[name] = value
    with pytest.raises(RuntimeError, match=rf"\[GX1_CPU_AUDIT_{name}_INVALID\]"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda _path: pytest.fail("must not read cgroup"),
        )


@pytest.mark.parametrize("name", [
    "GX1_CAPPED_MEMORY_BYTES", "GX1_CAPPED_SWAP_BYTES", "GX1_CAPPED_TASKS_MAX",
])
@pytest.mark.parametrize("value", [None, "", "max", "0", "-1", "1.0", "١"])
def test_cpu_audit_rejects_invalid_declared_limits_before_read(name, value) -> None:
    env, _files = _cpu_audit_fixture()
    if value is None:
        env.pop(name)
    else:
        env[name] = value
    with pytest.raises(RuntimeError, match=rf"\[GX1_CPU_AUDIT_ENV_PROOF_INVALID\] field={name}"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda _path: pytest.fail("must not read cgroup"),
        )


@pytest.mark.parametrize("name, value", [
    ("GX1_CAPPED_MEMORY_BYTES", 4 * 1024**3 + 1),
    ("GX1_CAPPED_SWAP_BYTES", 512 * 1024**2 + 1),
    ("GX1_CAPPED_TASKS_MAX", 65),
])
def test_cpu_audit_rejects_declared_limits_above_audit_ceiling(name, value) -> None:
    env, _files = _cpu_audit_fixture()
    env[name] = str(value)
    with pytest.raises(RuntimeError, match="GX1_CPU_AUDIT_ENV_LIMIT_EXCEEDED"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda _path: pytest.fail("must not read cgroup"),
        )


@pytest.mark.parametrize("path, error", [
    ("/proc/self/cgroup", "CGROUP_PATH_UNAVAILABLE"),
    ("/proc/self/status", "AFFINITY_UNAVAILABLE"),
    *[
        (f"/sys/fs/cgroup/user.slice/gx1-audit-test.scope/{filename}", "CGROUP_LIMIT_UNAVAILABLE")
        for filename in ("memory.max", "memory.high", "memory.swap.max", "pids.max")
    ],
])
def test_cpu_audit_missing_actual_proof_fails_closed(path, error) -> None:
    env, files = _cpu_audit_fixture()
    files.pop(path)
    with pytest.raises(RuntimeError, match=f"GX1_CPU_AUDIT_{error}"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda requested: files[str(requested)],
        )


@pytest.mark.parametrize("scope", [
    "", "0::/\n", "0::relative.scope\n", "0::/gx1.service\n",
    "1:memory:/gx1.scope\n", "0::/one.scope\n0::/two.scope\n",
    "0::/user.slice/../gx1.scope\n", "0:memory:/gx1.scope\n",
])
def test_cpu_audit_rejects_malformed_or_unscoped_cgroup(scope) -> None:
    env, files = _cpu_audit_fixture()
    files["/proc/self/cgroup"] = scope
    with pytest.raises(RuntimeError, match="GX1_CPU_AUDIT_CGROUP_PATH_INVALID"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda path: files[str(path)],
        )


@pytest.mark.parametrize("filename", ["memory.max", "memory.high", "memory.swap.max", "pids.max"])
@pytest.mark.parametrize("value", ["max", "", "0", "-1", "1.0", "١"])
def test_cpu_audit_rejects_unbounded_or_invalid_actual_limits(filename, value) -> None:
    env, files = _cpu_audit_fixture()
    files[f"/sys/fs/cgroup/user.slice/gx1-audit-test.scope/{filename}"] = value
    with pytest.raises(RuntimeError, match="GX1_CPU_AUDIT_CGROUP_LIMIT_INVALID"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda path: files[str(path)],
        )


@pytest.mark.parametrize("filename, value", [
    ("memory.max", 4 * 1024**3 + 1),
    ("memory.high", 4 * 1024**3 + 1),
    ("memory.swap.max", 512 * 1024**2 + 1),
    ("pids.max", 65),
])
def test_cpu_audit_rejects_actual_limits_above_audit_ceiling(filename, value) -> None:
    env, files = _cpu_audit_fixture()
    files[f"/sys/fs/cgroup/user.slice/gx1-audit-test.scope/{filename}"] = str(value)
    with pytest.raises(RuntimeError, match="GX1_CPU_AUDIT_CGROUP_ACTUAL_LIMIT_EXCEEDED"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda path: files[str(path)],
        )


@pytest.mark.parametrize("filename, value", [
    ("memory.max", 3 * 1024**3),
    ("memory.high", 3 * 1024**3),
    ("memory.swap.max", 256 * 1024**2),
    ("pids.max", 32),
])
def test_cpu_audit_requires_every_actual_limit_to_match_environment(filename, value) -> None:
    env, files = _cpu_audit_fixture()
    files[f"/sys/fs/cgroup/user.slice/gx1-audit-test.scope/{filename}"] = str(value)
    with pytest.raises(RuntimeError, match="GX1_CPU_AUDIT_CGROUP_ENV_ACTUAL_MISMATCH"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda path: files[str(path)],
        )


@pytest.mark.parametrize("status", [
    "Name:\tpython\n", "Cpus_allowed_list:\t\n",
    "Cpus_allowed_list:\t0-7\nCpus_allowed_list:\t0-7\n",
    *[
        f"Cpus_allowed_list:\t{affinity}\n"
        for affinity in (
            "0-8", "8", "0,8", "0-9999999999999999999999", "3-1", "0,0",
            "0-4,4-7", "4,0", "-1", "0-", "0--7", "0-1-2", "0,", "01",
            "٠", "0, 1", "all",
        )
    ],
])
def test_cpu_audit_rejects_invalid_or_unconfined_affinity_despite_env_marker(status) -> None:
    env, files = _cpu_audit_fixture()
    env["GX1_CPU_AFFINITY"] = "0-7"
    files["/proc/self/status"] = status
    with pytest.raises(RuntimeError, match="GX1_CPU_AUDIT_AFFINITY_INVALID"):
        require_capped_cpu_audit_execution(
            environ=env, read_text=lambda path: files[str(path)],
        )


def _lock_metadata(
    mode=stat.S_IFREG | 0o644, user_id=1000, inode=35, links=1, device=78,
) -> os.stat_result:
    return os.stat_result((mode, inode, device, links, user_id, user_id, 0, 0, 0, 0))


def _lock_process_stat(process_pid, parent_pid, start_ticks=None) -> bytes:
    fields = ["S", str(parent_pid), *(["0"] * 17), str(start_ticks or process_pid * 10)]
    return f"{process_pid} (name with ) nested (parens)) {' '.join(fields)}\n".encode()


@pytest.fixture
def lock_kernel(monkeypatch):
    runner = Path(capped_execution.__file__).resolve().parents[2] / "scripts/gx1_capped_run.sh"
    bash = str(Path("/bin/bash").resolve())
    files = {}
    links = {}
    metadata = {
        "/": _lock_metadata(stat.S_IFDIR | 0o755, user_id=0),
        "/run": _lock_metadata(stat.S_IFDIR | 0o755, user_id=0),
        "/run/user": _lock_metadata(stat.S_IFDIR | 0o755, user_id=0),
        "/run/user/1000": _lock_metadata(stat.S_IFDIR | 0o700),
        "/run/user/1000/gx1-current-heavy-job.lock": _lock_metadata(),
        "/proc/100/fd/9": _lock_metadata(),
    }
    for process_pid, parent_pid in ((400, 300), (300, 200), (200, 100), (100, 1), (1, 0)):
        prefix = f"/proc/{process_pid}"
        files[f"{prefix}/stat"] = _lock_process_stat(process_pid, parent_pid)
        user_id = 0 if process_pid == 1 else 1000
        files[f"{prefix}/status"] = f"Uid:\t{user_id}\t{user_id}\t{user_id}\t{user_id}\n".encode()
        files[f"{prefix}/cgroup"] = (
            b"0::/user.slice/run-proof.scope\n" if process_pid > 100 else b"0::/\n"
        )
        links[f"{prefix}/exe"] = bash if process_pid in {100, 300} else "/usr/bin/python3"
        links[f"{prefix}/cwd"] = str(runner.parents[1])
        files[f"{prefix}/cmdline"] = b"bash\0" + os.fsencode(runner) + b"\0--class\0audit\0"
    files["/proc/100/fdinfo/9"] = (
        b"pos:\t0\nflags:\t0102001\nmnt_id:\t404\nino:\t35\n"
        b"lock:\t1: FLOCK ADVISORY WRITE 0 00:4e:35 0 EOF\n"
    )
    original_stat = Path.stat
    original_lstat = Path.lstat
    original_readlink = os.readlink

    def fixture_value(mapping, path):
        value = mapping[str(path)]
        if callable(value):
            value = value()
        if isinstance(value, Exception):
            raise value
        return value

    def read_proc(path):
        if str(path) not in files:
            raise FileNotFoundError(str(path))
        return fixture_value(files, path)

    def fixture_stat(path, *arguments, **keywords):
        if str(path) in metadata:
            return fixture_value(metadata, path)
        if str(path).startswith(("/proc/", "/run/")):
            raise FileNotFoundError(str(path))
        return original_stat(path, *arguments, **keywords)

    def fixture_lstat(path, *arguments, **keywords):
        if str(path) in metadata:
            return fixture_value(metadata, path)
        if str(path).startswith(("/proc/", "/run/")):
            raise FileNotFoundError(str(path))
        return original_lstat(path, *arguments, **keywords)

    def fixture_readlink(path, *arguments, **keywords):
        if str(path) in links:
            return fixture_value(links, path)
        if str(path).startswith("/proc/"):
            raise FileNotFoundError(str(path))
        return original_readlink(path, *arguments, **keywords)

    monkeypatch.setattr(capped_execution, "_read_lock_proc", read_proc)
    monkeypatch.setattr(Path, "stat", fixture_stat)
    monkeypatch.setattr(Path, "lstat", fixture_lstat)
    monkeypatch.setattr(os, "readlink", fixture_readlink)
    monkeypatch.setattr(os, "getuid", lambda: 1000)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(os, "getpid", lambda: 400)
    return files, metadata, links


@pytest.mark.parametrize("runtime", [None, "/tmp/alternate-runtime", "/run/user/999"])
def test_canonical_lock_ignores_xdg_and_preserves_existing_0644(lock_kernel, monkeypatch, runtime):
    if runtime is None:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    else:
        monkeypatch.setenv("XDG_RUNTIME_DIR", runtime)
    assert canonical_heavy_job_lock_path() == Path("/run/user/1000/gx1-current-heavy-job.lock")
    assert require_capped_lock_ancestry() == {
        "lock_path": "/run/user/1000/gx1-current-heavy-job.lock",
        "lock_device": 78,
        "lock_inode": 35,
        "owner_pid": 100,
        "owner_start_ticks": 1000,
        "scope": "/user.slice/run-proof.scope",
    }


def test_lock_path_allows_creation_but_ancestry_requires_existing_lock(lock_kernel):
    _, metadata, _ = lock_kernel
    del metadata["/run/user/1000/gx1-current-heavy-job.lock"]
    assert canonical_heavy_job_lock_path() == Path("/run/user/1000/gx1-current-heavy-job.lock")
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("path, replacement", [
    ("/run/user/1000", FileNotFoundError("runtime missing")),
    ("/run/user/1000", _lock_metadata(stat.S_IFDIR | 0o755)),
    ("/run/user/1000", _lock_metadata(stat.S_IFDIR | 0o700, user_id=999)),
    ("/run/user/1000", _lock_metadata(stat.S_IFLNK | 0o700)),
    ("/run/user", _lock_metadata(stat.S_IFDIR | 0o777, user_id=0)),
    ("/run/user", _lock_metadata(stat.S_IFDIR | 0o755, user_id=1000)),
    ("/run", _lock_metadata(stat.S_IFLNK | 0o755, user_id=0)),
    ("/run/user/1000/gx1-current-heavy-job.lock", _lock_metadata(stat.S_IFLNK | 0o644)),
    ("/run/user/1000/gx1-current-heavy-job.lock", _lock_metadata(stat.S_IFIFO | 0o600)),
    ("/run/user/1000/gx1-current-heavy-job.lock", _lock_metadata(stat.S_IFDIR | 0o700)),
    ("/run/user/1000/gx1-current-heavy-job.lock", _lock_metadata(stat.S_IFREG | 0o666)),
    ("/run/user/1000/gx1-current-heavy-job.lock", _lock_metadata(stat.S_IFREG | 0o2644)),
    ("/run/user/1000/gx1-current-heavy-job.lock", _lock_metadata(user_id=999)),
    ("/run/user/1000/gx1-current-heavy-job.lock", _lock_metadata(links=2)),
])
def test_canonical_lock_rejects_unsafe_paths_without_fallback(lock_kernel, monkeypatch, path, replacement):
    _, metadata, _ = lock_kernel
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp/otherwise-valid-runtime")
    metadata[path] = replacement
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PATH_INVALID"):
        canonical_heavy_job_lock_path()
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


def test_canonical_lock_rejects_effective_uid_change(lock_kernel, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(RuntimeError, match="real and effective UID differ"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("reported_pid", [0, 100, 999])
def test_lock_owner_is_descriptor_ancestor_not_flock_pid(lock_kernel, reported_pid):
    files, metadata, _ = lock_kernel
    files["/proc/100/fdinfo/9"] = files["/proc/100/fdinfo/9"].replace(
        b"WRITE 0 ", f"WRITE {reported_pid} ".encode(),
    )
    assert "/proc/400/fd/9" not in metadata
    assert "/proc/300/fd/9" not in metadata
    assert "/proc/200/fd/9" not in metadata
    assert require_capped_lock_ancestry()["owner_pid"] == 100


def test_lock_runner_resolves_relative_executed_script_against_actual_cwd(lock_kernel):
    files, _, _ = lock_kernel
    files["/proc/100/cmdline"] = b"bash\0--noprofile\0-eu\0--\0scripts/gx1_capped_run.sh\0--class\0audit\0"
    assert require_capped_lock_ancestry()["owner_pid"] == 100


def _changed_cwd_wrapper_ancestor(lock_kernel, tmp_path):
    files, metadata, links = lock_kernel
    original = tmp_path / "original"
    original.mkdir()
    wrapper = original / "outer-control.sh"
    wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
    files["/proc/100/stat"] = _lock_process_stat(100, 50)
    files["/proc/50/stat"] = _lock_process_stat(50, 1)
    files["/proc/50/status"] = b"Uid:\t1000\t1000\t1000\t1000\n"
    files["/proc/50/cgroup"] = b"0::/\n"
    files["/proc/50/cmdline"] = b"bash\0outer-control.sh\0"
    links["/proc/50/exe"] = str(Path("/bin/bash").resolve())
    links["/proc/50/cwd"] = str(tmp_path)
    links["/proc/50/fd/255"] = str(wrapper)
    metadata["/proc/50/fd/255"] = wrapper.stat()
    assert not (tmp_path / "outer-control.sh").exists()
    return wrapper


def test_unrelated_script_ancestor_may_have_changed_its_working_directory(lock_kernel, tmp_path):
    _changed_cwd_wrapper_ancestor(lock_kernel, tmp_path)
    assert require_capped_lock_ancestry()["owner_pid"] == 100


@pytest.mark.parametrize("failure", ["missing", "nonregular", "canonical", "deleted", "inode", "basename"])
def test_missing_wrapper_operand_requires_unambiguous_actual_script_identity(lock_kernel, tmp_path, failure):
    wrapper = _changed_cwd_wrapper_ancestor(lock_kernel, tmp_path)
    _, metadata, links = lock_kernel
    if failure == "missing":
        del metadata["/proc/50/fd/255"]
    elif failure == "nonregular":
        metadata["/proc/50/fd/255"] = _lock_metadata(stat.S_IFIFO | 0o600)
    elif failure == "canonical":
        runner = Path(capped_execution.__file__).resolve().parents[2] / "scripts/gx1_capped_run.sh"
        metadata["/proc/50/fd/255"] = runner.stat()
        links["/proc/50/fd/255"] = str(runner)
    elif failure == "deleted":
        links["/proc/50/fd/255"] = str(wrapper) + " (deleted)"
    elif failure == "inode":
        metadata["/proc/50/fd/255"] = _lock_metadata(inode=wrapper.stat().st_ino + 1, device=wrapper.stat().st_dev)
    elif failure == "basename":
        other = wrapper.with_name("different-wrapper.sh")
        other.write_text("#!/bin/bash\n", encoding="utf-8")
        metadata["/proc/50/fd/255"] = other.stat()
        links["/proc/50/fd/255"] = str(other)
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("replacement", [
    b"bash\0-c\0scripts/gx1_capped_run.sh\0",
    b"scripts/gx1_capped_run.sh\0-c\0true\0",
    b"bash\0-s\0scripts/gx1_capped_run.sh\0",
    b"bash\0CLAUDE.md\0scripts/gx1_capped_run.sh\0",
    b"bash\0scripts/gx1_capped_run.sh",
])
def test_lock_rejects_runner_name_in_nonexecuted_command_arguments(lock_kernel, replacement):
    files, _, _ = lock_kernel
    files["/proc/100/cmdline"] = replacement
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("field, replacement", [
    ("exe", "/usr/bin/python3"),
    ("cwd", "/"),
])
def test_lock_runner_requires_real_executable_and_cwd(lock_kernel, field, replacement):
    files, _, links = lock_kernel
    files["/proc/100/cmdline"] = b"bash\0scripts/gx1_capped_run.sh\0"
    links[f"/proc/100/{field}"] = replacement
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("replacement", [
    b"", b"lock: 1: FLOCK ADVISORY READ 0 00:4e:35 0 EOF\n",
    b"lock: 1: -> FLOCK ADVISORY WRITE 0 00:4e:35 0 EOF\n",
    b"lock: 1: POSIX ADVISORY WRITE 100 00:4e:35 0 EOF\n",
    b"lock: 1: FLOCK ADVISORY WRITE 0 00:4f:35 0 EOF\n",
    b"lock: 1: FLOCK ADVISORY WRITE 0 00:4e:36 0 EOF\n",
    b"lock: 1: FLOCK ADVISORY WRITE 0 00:4e:35 1 EOF\n",
    b"lock: 1: FLOCK ADVISORY WRITE 0 00:4e:35 0 10\n",
    b"lock: 1: FLOCK ADVISORY WRITE 0 00:4e:35 0 EOF\n" * 2,
])
def test_lock_requires_unique_exclusive_whole_file_kernel_lock(lock_kernel, replacement):
    files, _, _ = lock_kernel
    files["/proc/100/fdinfo/9"] = b"flags:\t0102001\nino:\t35\n" + replacement
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("before, after", [
    (b"ino:\t35", b"ino:\t36"),
    (b"flags:\t0102001", b"flags:\t0102000"),
    (b"flags:\t0102001", b"flags:\tinvalid"),
    (b"ino:\t35", b"ino:\t35\nino:\t35"),
])
def test_lock_rejects_invalid_descriptor_metadata(lock_kernel, before, after):
    files, _, _ = lock_kernel
    files["/proc/100/fdinfo/9"] = files["/proc/100/fdinfo/9"].replace(before, after)
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("replacement", [
    _lock_metadata(inode=36), _lock_metadata(device=79),
    _lock_metadata(stat.S_IFIFO | 0o644), FileNotFoundError("closed FD9"),
])
def test_lock_rejects_noncanonical_or_closed_owner_descriptor(lock_kernel, replacement):
    _, metadata, _ = lock_kernel
    metadata["/proc/100/fd/9"] = replacement
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


def test_matching_scope_and_forged_environment_cannot_replace_outer_lock(lock_kernel, monkeypatch):
    files, metadata, _ = lock_kernel
    env, _ = _cpu_audit_fixture()
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("GX1_CAPPED_LOCK_OWNER_PID", "100")
    metadata["/proc/300/fd/9"] = metadata.pop("/proc/100/fd/9")
    files["/proc/300/fdinfo/9"] = files["/proc/100/fdinfo/9"]
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("replacement", [
    b"0::/\n", b"0::/not-a-scope\n", b"0::/nested/../run-proof.scope\n",
    b"0::/first.scope\n0::/second.scope\n",
])
def test_lock_rejects_invalid_actual_scope(lock_kernel, replacement):
    files, _, _ = lock_kernel
    files["/proc/400/cgroup"] = replacement
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


def test_lock_rejects_manual_intervening_scope_despite_nearer_inherited_lock(lock_kernel):
    files, metadata, _ = lock_kernel
    files["/proc/400/cgroup"] = files["/proc/300/cgroup"] = b"0::/manual.scope\n"
    metadata["/proc/300/fd/9"] = metadata["/proc/100/fd/9"]
    files["/proc/300/fdinfo/9"] = files["/proc/100/fdinfo/9"]
    with pytest.raises(RuntimeError, match="exactly one descendant scope"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("path, replacement", [
    ("/proc/100/cgroup", b"0::/user.slice/run-proof.scope\n"),
    ("/proc/200/stat", _lock_process_stat(200, 300)),
    ("/proc/200/stat", _lock_process_stat(200, 0)),
    ("/proc/200/stat", b"unparseable\n"),
    ("/proc/200/status", b"Uid:\t1000\t0\t1000\t1000\n"),
    ("/proc/200/status", b"Uid:\t1000\t1000\n"),
    ("/proc/200/stat", FileNotFoundError("ancestor disappeared")),
    ("/proc/100/fdinfo/9", PermissionError("inaccessible lock metadata")),
])
def test_lock_rejects_missing_malformed_or_unstable_ancestry(lock_kernel, path, replacement):
    files, _, _ = lock_kernel
    files[path] = replacement
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("process_pid, parent_pid", [(100, 1), (200, 100), (400, 300)])
def test_lock_rechecks_kernel_start_ticks_against_pid_reuse(lock_kernel, process_pid, parent_pid):
    files, _, _ = lock_kernel
    reads = 0

    def changing_stat():
        nonlocal reads
        reads += 1
        return _lock_process_stat(process_pid, parent_pid, process_pid * 10 + (reads > 1))

    files[f"/proc/{process_pid}/stat"] = changing_stat
    with pytest.raises(RuntimeError, match="process ancestry changed"):
        require_capped_lock_ancestry()


@pytest.mark.parametrize("change", ["lock_path", "descriptor", "lock_release", "parent", "scope"])
def test_lock_rechecks_file_descriptor_and_ancestry_identity(lock_kernel, change):
    files, metadata, _ = lock_kernel
    original_info = files["/proc/100/fdinfo/9"]
    reads = 0

    def changing_info():
        nonlocal reads
        reads += 1
        if reads == 1:
            if change == "lock_path":
                metadata["/run/user/1000/gx1-current-heavy-job.lock"] = _lock_metadata(inode=36)
            elif change == "descriptor":
                metadata["/proc/100/fd/9"] = _lock_metadata(inode=36)
            elif change == "parent":
                files["/proc/200/stat"] = _lock_process_stat(200, 1)
            elif change == "scope":
                files["/proc/200/cgroup"] = b"0::/manual.scope\n"
        if reads > 1 and change == "lock_release":
            return b"flags:\t0102001\nino:\t35\n"
        return original_info

    files["/proc/100/fdinfo/9"] = changing_info
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


def test_lock_proof_has_no_public_proc_or_environment_override():
    with pytest.raises(TypeError):
        require_capped_lock_ancestry(proc_root="/tmp/fake-proc")
    with pytest.raises(TypeError):
        require_capped_lock_ancestry(environ={"GX1_CAPPED_LOCK_OWNER_PID": "100"})


def test_trainer_proof_uses_actual_cgroup_and_lock_with_existing_128_task_limit(monkeypatch):
    env, files = _guarded_fixture()
    env.update({'GX1_CAPPED_CLASS':'trainer', 'GX1_CUDA_PRODUCER_GUARD':'false',
                'GX1_TRAINER_EXECUTION_MODE':'canonical',
                'GX1_CAPPED_MEMORY_BYTES':str(20 * 1024**3), 'GX1_CAPPED_TASKS_MAX':'128'})
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        env[key] = '8'
    env.update({'ARROW_NUM_THREADS':'1', 'POLARS_MAX_THREADS':'1'})
    files['/sys/fs/cgroup/gx1-cuda-test.scope/memory.max'] = str(20 * 1024**3)
    files['/sys/fs/cgroup/gx1-cuda-test.scope/memory.high'] = str(20 * 1024**3)
    files['/sys/fs/cgroup/gx1-cuda-test.scope/pids.max'] = '128'
    monkeypatch.setattr(capped_execution.os, 'environ', env)
    monkeypatch.setattr(capped_execution.os, 'sched_getaffinity', lambda _pid: set(range(19)))
    monkeypatch.setattr(capped_execution.Path, 'read_text', lambda path, **kw: files[str(path)])
    calls = []
    def lock():
        calls.append(True)
        return {'verified_fixture': True}
    monkeypatch.setattr(capped_execution, 'require_capped_lock_ancestry', lock)
    proof = capped_execution.require_guarded_cuda_trainer_execution()
    assert proof['pids'] == 128 and proof['memory_max'] == 20 * 1024**3
    assert calls == [True]
    files['/sys/fs/cgroup/gx1-cuda-test.scope/memory.max'] = str(21 * 1024**3)
    with pytest.raises(RuntimeError, match='CGROUP_ACTUAL_LIMIT_EXCEEDED'):
        capped_execution.require_guarded_cuda_trainer_execution()
    assert calls == [True]


def test_producer_cannot_inherit_trainer_128_task_exception():
    env, files = _guarded_fixture()
    env['GX1_CAPPED_TASKS_MAX'] = '128'
    files['/sys/fs/cgroup/gx1-cuda-test.scope/pids.max'] = '128'
    with pytest.raises(RuntimeError, match='ENV_LIMIT_EXCEEDED'):
        require_guarded_cuda_producer_execution(environ=env, read_text=lambda path: files[str(path)])


def test_project_lock_is_independent_of_legacy_machine_lock(lock_kernel):
    # An unrelated project may hold the legacy inode. CURRENT proves its own
    # exact lock, not whatever path an environment variable or caller supplies.
    _, metadata, _ = lock_kernel
    metadata["/run/user/1000/gx1-heavy-job.lock"] = _lock_metadata(inode=999)
    proof = require_capped_lock_ancestry()
    assert proof["lock_path"] == "/run/user/1000/gx1-current-heavy-job.lock"
    assert proof["lock_inode"] == 35
    del metadata["/run/user/1000/gx1-current-heavy-job.lock"]
    with pytest.raises(RuntimeError, match="GX1_CAPPED_LOCK_PROOF_INVALID"):
        require_capped_lock_ancestry()


def _native_future_cpu_mechanics_fixture():
    env,files=_cpu_audit_fixture()
    memory=12*1024**3
    env.update(GX1_CAPPED_CLASS='producer',GX1_CAPPED_MEMORY_BYTES=str(memory),
        GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS='true',
        GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256='a'*64,
        GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE='continuous')
    files['/sys/fs/cgroup/user.slice/gx1-audit-test.scope/memory.max']=str(memory)
    files['/sys/fs/cgroup/user.slice/gx1-audit-test.scope/memory.high']=str(memory)
    return env,files


@pytest.mark.parametrize('lane',['continuous','replay'])
def test_native_future_state_cpu_mechanics_requires_explicit_mode_and_exact_cpu_scope(monkeypatch,lane):
    env,files=_native_future_cpu_mechanics_fixture()
    env['GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE']=lane
    lock={'scope':'/user.slice/gx1-audit-test.scope','exact_lock':'synthetic'}
    monkeypatch.setattr(capped_execution,'require_capped_lock_ancestry',lambda:lock)
    proof=capped_execution.require_native_future_state_cpu_mechanics_execution(
        plan_sha256='a'*64,lane=lane,environ=env,read_text=lambda path:files[str(path)])
    assert proof['execution_mode']=='native_future_state_cpu_mechanics'
    assert proof['technical_only'] is True and proof['lane']==lane
    assert proof['memory_max']==proof['memory_high']==12*1024**3
    assert proof['swap']==512*1024**2 and proof['pids']==64
    assert proof['cpu_affinity']==list(range(8)) and proof['numerical_threads']==1
    assert proof['lock']==lock and proof['outer_seconds']==900 and proof['kill_after_seconds']==10
    assert proof['plan']['path'].endswith('/configs/research/NATIVE_M5_FUTURE_STATE_CPU_MECHANICS_20261121_001.json')
    assert proof['plan']['sha256']=='a'*64


@pytest.mark.parametrize('field,value',[
    ('GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS','false'),
    ('GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256','b'*64),
    ('GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE','replay'),
    ('GX1_CAPPED_CLASS','trainer'),('GX1_CAPPED_MEMORY_BYTES',str(20*1024**3)),
    ('GX1_CAPPED_SWAP_BYTES','0'),('GX1_CAPPED_TASKS_MAX','128'),
    ('GX1_TRAINER_DEVICE','cuda'),('GX1_CUDA_PRODUCER_GUARD','true'),
    ('OMP_NUM_THREADS','8'),('GX1_TRAINER_EXECUTION_MODE','cuda_producer')])
def test_native_future_state_cpu_mechanics_rejects_broader_or_unproven_mode_before_cgroup(field,value):
    env,_=_native_future_cpu_mechanics_fixture();env[field]=value
    with pytest.raises(RuntimeError):
        capped_execution.require_native_future_state_cpu_mechanics_execution(
            plan_sha256='a'*64,lane='continuous',environ=env,
            read_text=lambda path:pytest.fail('reject before reading cgroup'))


@pytest.mark.parametrize('plan_sha,lane',[('bad','continuous'),('A'*64,'continuous'),
    ('a'*64,'main'),('a'*64,True),(None,'continuous')])
def test_native_future_state_cpu_mechanics_rejects_invalid_request_before_cgroup(plan_sha,lane):
    env,_=_native_future_cpu_mechanics_fixture()
    with pytest.raises(RuntimeError):
        capped_execution.require_native_future_state_cpu_mechanics_execution(
            plan_sha256=plan_sha,lane=lane,environ=env,
            read_text=lambda path:pytest.fail('reject before reading cgroup'))


@pytest.mark.parametrize('changed',('memory.high','memory.max','pids.max','memory.swap.max','affinity'))
def test_native_future_state_cpu_mechanics_rejects_actual_scope_drift(monkeypatch,changed):
    env,files=_native_future_cpu_mechanics_fixture()
    monkeypatch.setattr(capped_execution,'require_capped_lock_ancestry',lambda:{'synthetic':'lock'})
    path='/proc/self/status' if changed=='affinity' else '/sys/fs/cgroup/user.slice/gx1-audit-test.scope/'+changed
    files[path]='Cpus_allowed_list:\t0-18\n' if changed=='affinity' else '1'
    with pytest.raises(RuntimeError):
        capped_execution.require_native_future_state_cpu_mechanics_execution(
            plan_sha256='a'*64,lane='continuous',environ=env,read_text=lambda path:files[str(path)])


def _native_future_cuda_fixture(lane='continuous'):
    env, files = _guarded_fixture()
    seconds = 5400 if lane == 'initial' else 1800
    env.update(GX1_CAPPED_CLASS='trainer', GX1_CUDA_PRODUCER_GUARD='false',
        GX1_TRAINER_EXECUTION_MODE='canonical', GX1_CAPPED_MEMORY_BYTES=str(20*1024**3),
        GX1_CAPPED_TASKS_MAX='128', GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION='true',
        GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256='a'*64, GX1_NATIVE_FUTURE_STATE_CUDA_LANE=lane,
        GX1_TRAINER_MAX_WALL_SECONDS=str(seconds), GX1_TRAINER_MODEL_MAX_WALL_SECONDS=str(seconds))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        env[key] = '8'
    env.update(ARROW_NUM_THREADS='1', POLARS_MAX_THREADS='1')
    for key in ('memory.max','memory.high'):
        files['/sys/fs/cgroup/gx1-cuda-test.scope/'+key] = str(20*1024**3)
    files['/sys/fs/cgroup/gx1-cuda-test.scope/pids.max'] = '128'
    return env, files


@pytest.mark.parametrize('lane', ['continuous','replay','initial'])
def test_native_future_state_cuda_qualification_requires_existing_canonical_trainer_proof(monkeypatch,lane):
    env, files = _native_future_cuda_fixture(lane)
    monkeypatch.setattr(capped_execution.os, 'environ', env)
    monkeypatch.setattr(capped_execution.os, 'sched_getaffinity', lambda pid:set(range(19)))
    monkeypatch.setattr(capped_execution.Path, 'read_text', lambda path,**kw:files[str(path)])
    lock = {'canonical_lock':'synthetic'}
    monkeypatch.setattr(capped_execution, 'require_capped_lock_ancestry', lambda:lock)
    proof = capped_execution.require_native_future_state_cuda_qualification_execution(plan_sha256='a'*64,lane=lane)
    basename = 'NATIVE_M5_FUTURE_STATE_CUDA_'+('INITIAL' if lane=='initial' else 'MECHANICS')+'_20261121_001'
    assert proof['plan_path'].endswith('/configs/research/'+basename+'.json')
    assert proof['plan']=={'path':proof['plan_path'],'sha256':'a'*64}
    assert proof['output_root']=='/home/andre2/GX1_RUNS/'+basename+('' if lane=='initial' else '/'+lane)
    assert proof['class']=='trainer' and proof['execution_mode']=='canonical' and proof['technical_only'] is True
    assert proof['outer_seconds']==(5400 if lane=='initial' else 1800)
    assert proof['internal_seconds']==proof['outer_seconds']-30 and proof['kill_after_seconds']==10
    assert proof['memory_max']==proof['memory_high']==20*1024**3
    assert proof['swap']==512*1024**2 and proof['pids']==128
    assert proof['cpu_affinity']==list(range(19)) and proof['numerical_threads']==8 and proof['lock']==lock


@pytest.mark.parametrize('field,value', [
    ('GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION',None),
    ('GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION','false'),
    ('GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256','b'*64),
    ('GX1_NATIVE_FUTURE_STATE_CUDA_LANE','replay'),
    ('GX1_TRAINER_MAX_WALL_SECONDS','5400'),
    ('GX1_TRAINER_MODEL_MAX_WALL_SECONDS','5400')])
def test_native_future_state_cuda_qualification_rejects_private_mode_drift_before_trainer(monkeypatch,field,value):
    env, _ = _native_future_cuda_fixture()
    if value is None:env.pop(field)
    else:env[field]=value
    monkeypatch.setattr(capped_execution.os,'environ',env)
    monkeypatch.setattr(capped_execution,'require_guarded_cuda_trainer_execution',
        lambda:pytest.fail('private mismatch reached canonical trainer proof'))
    with pytest.raises(RuntimeError,match='GX1_NATIVE_CUDA_QUALIFICATION_'):
        capped_execution.require_native_future_state_cuda_qualification_execution(plan_sha256='a'*64,lane='continuous')


@pytest.mark.parametrize('plan_sha,lane', [('bad','continuous'),('A'*64,'continuous'),
    ('a'*64,'fit'),('a'*64,True),(None,'initial')])
def test_native_future_state_cuda_qualification_rejects_invalid_request_before_trainer(monkeypatch,plan_sha,lane):
    monkeypatch.setattr(capped_execution,'require_guarded_cuda_trainer_execution',
        lambda:pytest.fail('invalid request reached canonical trainer proof'))
    with pytest.raises(RuntimeError,match='GX1_NATIVE_CUDA_QUALIFICATION_'):
        capped_execution.require_native_future_state_cuda_qualification_execution(plan_sha256=plan_sha,lane=lane)


@pytest.mark.parametrize('field,value', [('GX1_CAPPED_CLASS','producer'),('GX1_TRAINER_DEVICE','cpu'),
    ('GX1_TRAINER_EXECUTION_MODE','attended_smoke'),('GX1_CUDA_PRODUCER_GUARD','true'),
    ('GX1_CAPPED_MEMORY_BYTES',str(12*1024**3)),('OMP_NUM_THREADS','1')])
def test_native_future_state_cuda_qualification_cannot_bypass_canonical_scope(monkeypatch,field,value):
    env, _ = _native_future_cuda_fixture();env[field]=value
    monkeypatch.setattr(capped_execution.os,'environ',env)
    monkeypatch.setattr(capped_execution.Path,'read_text',lambda *a,**kw:pytest.fail('invalid scope reached cgroup'))
    with pytest.raises(RuntimeError,match='GX1_CUDA_TRAINER_ENV_INVALID'):
        capped_execution.require_native_future_state_cuda_qualification_execution(plan_sha256='a'*64,lane='continuous')



def _native_future_leaf_fixture(lane=None):
    env, files = _native_future_cuda_fixture()
    seconds = 1800 if lane is None or lane == 'recovery_probe' or str(lane).endswith('_replay') else (12000 if lane in ('main','main_resume') else 5400)
    env.update(GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION='false', GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS='false',
        GX1_NATIVE_FUTURE_STATE_THROUGHPUT='true' if lane is None else 'false',
        GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256='a'*64 if lane is None else '',
        GX1_NATIVE_FUTURE_STATE_TRAINING='false' if lane is None else 'true',
        GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256='' if lane is None else 'a'*64,
        GX1_NATIVE_FUTURE_STATE_TRAINING_LANE='' if lane is None else lane,
        GX1_TRAINER_MAX_WALL_SECONDS=str(seconds), GX1_TRAINER_MODEL_MAX_WALL_SECONDS=str(seconds),
        GX1_TRAINER_ATTENDED_STAGE_REQUIRED='false', GX1_TRAINER_GPU_INDEX='0',
        GX1_TRAINER_GPU_MAX_POWER_LIMIT_W='300', GX1_TRAINER_GPU_MAX_POWER_DRAW_W='310',
        GX1_TRAINER_GPU_MAX_CORE_TEMP_C='85', GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C='80',
        GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB='12288', GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS='1',
        GX1_POWER_BENCHMARK_SCOPE_JSON='', GX1_POWER_BENCHMARK_SCOPE_SHA256='')
    return env, files, seconds


def _native_future_leaf_call(lane=None,sha='a'*64):
    if lane is None:return capped_execution.require_native_future_state_throughput_execution(plan_sha256=sha)
    return capped_execution.require_native_future_state_training_execution(plan_sha256=sha,lane=lane)


@pytest.mark.parametrize('lane',[None,'joint','joint_replay','m5_primary','m5_primary_replay','main','recovery_probe','main_resume'])
def test_native_future_state_leaf_reuses_exact_canonical_trainer_proof(monkeypatch,lane):
    env, files, seconds = _native_future_leaf_fixture(lane)
    monkeypatch.setattr(capped_execution.os,'environ',env)
    monkeypatch.setattr(capped_execution.os,'sched_getaffinity',lambda pid:set(range(19)))
    monkeypatch.setattr(capped_execution.Path,'read_text',lambda path,**kw:files[str(path)])
    monkeypatch.setattr(capped_execution,'require_capped_lock_ancestry',lambda:{'canonical':'lock'})
    proof=_native_future_leaf_call(lane)
    fixed={None:'NATIVE_FUTURE_STATE_THROUGHPUT_20261121_002','main':'NATIVE_FUTURE_STATE_ONE_EPOCH_20261121_001',
        'recovery_probe':'NATIVE_FUTURE_STATE_RECOVERY_PROBE_20261121_001','main_resume':'NATIVE_FUTURE_STATE_ONE_EPOCH_RESUME_20261121_001'}
    basename=fixed.get(lane,'NATIVE_FUTURE_STATE_PAIRED_LEARNING_20261121_001')
    assert proof['plan_path'].endswith('/configs/research/'+basename+'.json')
    assert proof['plan']=={'path':proof['plan_path'],'sha256':'a'*64}
    assert proof['output_root']=='/home/andre2/GX1_RUNS/'+basename+(''if lane in fixed else'/'+lane)
    assert proof['outer_seconds']==seconds and proof['internal_seconds']==seconds-30
    assert proof['memory_max']==proof['memory_high']==20*1024**3 and proof['pids']==128
    assert proof['swap']==512*1024**2 and proof['cpu_affinity']==list(range(19)) and proof['numerical_threads']==8
    assert proof['training_authority_granted']is False and proof['execution_mode']=='canonical'


@pytest.mark.parametrize('lane',[None,'joint'])
@pytest.mark.parametrize('field,value',[
    ('GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION','true'),('GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS','true'),
    ('GX1_TRAINER_MAX_WALL_SECONDS','12001'),('GX1_TRAINER_MODEL_MAX_WALL_SECONDS','12001'),
    ('GX1_TRAINER_GPU_MAX_CORE_TEMP_C','86'),('GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C','81'),
    ('GX1_TRAINER_GPU_MAX_POWER_LIMIT_W','301'),('GX1_TRAINER_GPU_MAX_POWER_DRAW_W','311'),
    ('GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB','12289'),('GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS','2'),
    ('GX1_TRAINER_GPU_INDEX','1'),('GX1_TRAINER_ATTENDED_STAGE_REQUIRED','true'),
    ('GX1_POWER_BENCHMARK_SCOPE_JSON','/unit/other.json'),('GX1_POWER_BENCHMARK_SCOPE_SHA256','b'*64)])
def test_native_future_state_leaf_rejects_private_or_hardware_drift_before_canonical_proof(monkeypatch,lane,field,value):
    env,_,_=_native_future_leaf_fixture(lane);env[field]=value
    monkeypatch.setattr(capped_execution.os,'environ',env)
    monkeypatch.setattr(capped_execution,'require_guarded_cuda_trainer_execution',lambda:pytest.fail('invalid private scope reached trainer'))
    with pytest.raises(RuntimeError,match='GX1_NATIVE_FUTURE_LEAF_ENV_INVALID'):_native_future_leaf_call(lane)


@pytest.mark.parametrize('lane',[None,'joint'])
@pytest.mark.parametrize('fault',['missing_mode','wrong_sha','foreign_mode','wrong_lane'])
def test_native_future_state_leaf_binds_its_private_route(monkeypatch,lane,fault):
    env,_,_=_native_future_leaf_fixture(lane)
    mode='THROUGHPUT'if lane is None else'TRAINING'
    if fault=='missing_mode':env.pop('GX1_NATIVE_FUTURE_STATE_'+mode)
    elif fault=='wrong_sha':env['GX1_NATIVE_FUTURE_STATE_'+mode+'_PLAN_SHA256']='b'*64
    elif fault=='foreign_mode':env['GX1_NATIVE_FUTURE_STATE_'+('TRAINING'if lane is None else'THROUGHPUT')]='true'
    elif fault=='wrong_lane':env['GX1_NATIVE_FUTURE_STATE_TRAINING_LANE']='main'
    monkeypatch.setattr(capped_execution.os,'environ',env)
    monkeypatch.setattr(capped_execution,'require_guarded_cuda_trainer_execution',lambda:pytest.fail('bad route reached trainer'))
    with pytest.raises(RuntimeError,match='GX1_NATIVE_FUTURE_LEAF_ENV_INVALID'):_native_future_leaf_call(lane)


@pytest.mark.parametrize('sha,lane',[('bad','joint'),('A'*64,'joint'),(None,'joint'),('a'*64,'fit'),('a'*64,True)])
def test_native_future_state_leaf_invalid_request_fails_before_any_proof(monkeypatch,sha,lane):
    monkeypatch.setattr(capped_execution,'require_guarded_cuda_trainer_execution',lambda:pytest.fail('bad request reached trainer'))
    with pytest.raises(RuntimeError,match='GX1_NATIVE_FUTURE_LEAF_'):_native_future_leaf_call(lane,sha)


def test_native_future_state_leaf_cannot_bypass_existing_trainer_environment(monkeypatch):
    env,_,_=_native_future_leaf_fixture('joint');env['GX1_CAPPED_CLASS']='producer'
    monkeypatch.setattr(capped_execution.os,'environ',env)
    monkeypatch.setattr(capped_execution.Path,'read_text',lambda *a,**kw:pytest.fail('invalid class reached cgroup'))
    with pytest.raises(RuntimeError,match='GX1_CUDA_TRAINER_ENV_INVALID'):_native_future_leaf_call('joint')


def _native_future_guard_parent():
    import json
    path=Path(__file__).resolve().parents[1]/'configs/research/NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001.json'
    value=json.loads(path.read_text())
    return value['source_owners'],value['metadata']['runtime']['source_commit']


def test_native_future_state_guard_bridge_preserves_whole_previous_modules():
    owners,commit=_native_future_guard_parent()
    proof=capped_execution.native_future_state_guard_source_bridge(owners,producer_commit=commit)
    assert proof['checked_guard_owners']==2 and proof['other_parent_owners_to_be_checked_by_caller']==56
    assert set(proof['owners'])=={'scripts/gx1_capped_run.sh','gx1/contracts/gx1_capped_execution_v1.py'}
    assert proof['owners']['scripts/gx1_capped_run.sh']['normalized_sha256']==owners['scripts/gx1_capped_run.sh']


@pytest.mark.parametrize('fault',['commit','absolute','payload_suffix','parent_traversal','bad_sha','wrong_guard_sha','missing_guard'])
def test_native_future_state_guard_bridge_rejects_invalid_inventory_before_source_IO(monkeypatch,fault):
    import subprocess
    owners,commit=_native_future_guard_parent();owners=dict(owners)
    if fault=='commit':commit='0'*40
    elif fault=='bad_sha':owners['scripts/gx1_capped_run.sh']='not-a-sha'
    elif fault=='wrong_guard_sha':owners['scripts/gx1_capped_run.sh']='0'*64
    elif fault=='missing_guard':owners['renamed.py']=owners.pop('scripts/gx1_capped_run.sh')
    else:
        name=next(k for k in owners if k not in capped_execution._NATIVE_FUTURE_STATE_GUARD_PARENT_HASHES)
        replacement={'absolute':'/unit/model.py','payload_suffix':'unit/payload.pt','parent_traversal':'unit/../payload.py'}[fault]
        owners[replacement]=owners.pop(name)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:pytest.fail('invalid bridge reached Git source read'))
    monkeypatch.setattr(capped_execution.Path,'read_bytes',lambda *a,**kw:pytest.fail('invalid bridge reached current source read'))
    with pytest.raises(RuntimeError,match='GX1_NATIVE_FUTURE_GUARD_BRIDGE_'):
        capped_execution.native_future_state_guard_source_bridge(owners,producer_commit=commit)


@pytest.mark.parametrize('fault',['duplicate_function','rename_function','old_constant','old_body','shell_old','shell_new','shell_duplicate'])
def test_native_future_state_guard_bridge_rejects_renames_redefinitions_and_guard_changes(monkeypatch,fault):
    owners,commit=_native_future_guard_parent();real=capped_execution.Path.read_bytes
    python_name='gx1/contracts/gx1_capped_execution_v1.py';shell_name='scripts/gx1_capped_run.sh'
    repo=Path(__file__).resolve().parents[1]
    target=repo/(shell_name if fault.startswith('shell_')else python_name)
    raw=real(target)
    if fault=='duplicate_function':changed=raw+b'\ndef require_guarded_cuda_trainer_execution(): return {}\n'
    elif fault=='rename_function':changed=raw.replace(b'def require_guarded_cuda_trainer_execution(',b'def renamed_trainer(',1)
    elif fault=='old_constant':changed=raw.replace(b'_MAX_MEMORY_BYTES = 20 * 1024**3',b'_MAX_MEMORY_BYTES = 21 * 1024**3',1)
    elif fault=='old_body':changed=raw.replace(b'max_pids=128, error_prefix="GX1_CUDA_TRAINER"',b'max_pids=129, error_prefix="GX1_CUDA_TRAINER"',1)
    elif fault=='shell_old':changed=raw.replace(b'TRAINER_GPU_MAX_MEMORY_TEMP_C=80',b'TRAINER_GPU_MAX_MEMORY_TEMP_C=81',1)
    elif fault=='shell_new':changed=raw.replace(b'NATIVE_FUTURE_STATE_THROUGHPUT=false',b'NATIVE_FUTURE_STATE_THROUGHPUT=true',1)
    else:changed=raw+capped_execution._NATIVE_FUTURE_STATE_GUARD_SHELL_ADDITIONS[0][2].encode()
    assert changed!=raw
    monkeypatch.setattr(capped_execution.Path,'read_bytes',lambda path:changed if path==target else real(path))
    with pytest.raises(RuntimeError,match='GX1_NATIVE_FUTURE_GUARD_BRIDGE_'):
        capped_execution.native_future_state_guard_source_bridge(owners,producer_commit=commit)
