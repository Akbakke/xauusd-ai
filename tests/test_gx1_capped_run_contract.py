from __future__ import annotations

import hashlib
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/gx1_capped_run.sh"
TRAINER_GUARD = REPO / "scripts/gx1_guarded_trainer_exec.sh"
TRAINER_MODULE = "gx1.models.entry_v10.entry_v10_ctx_train_v3"
CUDA_PRODUCER_MODULE = "gx1.scripts.evaluate_entry_candidate_selective_edge_v1"
TECHNICAL_VALIDATION_PRODUCER_MODULE = (
    "gx1.scripts.validate_entry_model_native_technical_checkpoint_v1"
)


def _hostile_nested_env(job_class: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GX1_CAPPED_CLASS": job_class,
            "GX1_CAPPED_MEMORY_BYTES": "1",
            "GX1_CAPPED_SWAP_BYTES": "1",
            "GX1_CAPPED_TASKS_MAX": "1",
        }
    )
    return env


def _run(
    job_class: str,
    memory: str,
    swap: str,
    *target: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--class",
            job_class,
            "--mem",
            memory,
            "--swap",
            swap,
            "--",
            *target,
        ],
        cwd=REPO,
        env=_hostile_nested_env(job_class),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _guard_env(
    *,
    device: str,
    nvidia_smi_path: Path,
    max_wall_seconds: int = 5,
    model_max_wall_seconds: int | None = None,
    execution_mode: str = "canonical",
    attended_stage_required: bool = False,
    max_power_limit_w: int = 250,
    max_power_draw_w: int = 250,
    max_memory_used_mib: int = 12288,
) -> dict[str, str]:
    cgroup_relative = next(
        row.split(":", 2)[2]
        for row in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        if row.split(":", 2)[0] == "0"
    )
    cgroup = Path("/sys/fs/cgroup") / cgroup_relative.lstrip("/")
    control_files = {
        "memory": cgroup / "memory.max",
        "swap": cgroup / "memory.swap.max",
        "tasks": cgroup / "pids.max",
    }
    if any(not path.is_file() for path in control_files.values()):
        pytest.skip("requires a delegated cgroup-v2 scope")
    protected = {
        "GX1_CAPPED_CLASS": "trainer",
        "GX1_CAPPED_MEMORY_BYTES": control_files["memory"].read_text().strip(),
        "GX1_CAPPED_SWAP_BYTES": control_files["swap"].read_text().strip(),
        "GX1_CAPPED_TASKS_MAX": control_files["tasks"].read_text().strip(),
        "GX1_TRAINER_DEVICE": device,
        "GX1_TRAINER_EXECUTION_MODE": execution_mode,
        "GX1_TRAINER_MAX_WALL_SECONDS": str(max_wall_seconds),
        "GX1_TRAINER_MODEL_MAX_WALL_SECONDS": str(
            model_max_wall_seconds
            if model_max_wall_seconds is not None
            else max_wall_seconds
        ),
        "GX1_TRAINER_ATTENDED_STAGE_REQUIRED": str(
            attended_stage_required
        ).lower(),
        "GX1_TRAINER_GPU_INDEX": "0",
        "GX1_TRAINER_GPU_MAX_CORE_TEMP_C": "78",
        "GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C": "90",
        "GX1_TRAINER_GPU_MAX_POWER_LIMIT_W": str(max_power_limit_w),
        "GX1_TRAINER_GPU_MAX_POWER_DRAW_W": str(max_power_draw_w),
        "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB": str(max_memory_used_mib),
        "GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS": "1",
        # The fake script accepts arbitrary arguments and returns the exact
        # five signed-bridge telemetry fields the guard consumes.
        "GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH": str(nvidia_smi_path),
        "GX1_TRAINER_HOST_TELEMETRY_URL": "http://172.30.224.1:38128/gx1/v1/telemetry/",
        "GX1_TRAINER_HOST_TELEMETRY_CERT_PATH": str(Path(__file__).resolve()),
        "GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "GX1_TRAINER_HOST_TELEMETRY_GPU_UUID": "GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29",
        "GX1_TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS": "2",
    }
    numeric_values = (
        value
        for key, value in protected.items()
        if key
        not in {
            "GX1_CAPPED_CLASS",
            "GX1_TRAINER_DEVICE",
            "GX1_TRAINER_EXECUTION_MODE",
            "GX1_TRAINER_ATTENDED_STAGE_REQUIRED",
            "GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH",
            "GX1_TRAINER_HOST_TELEMETRY_URL",
            "GX1_TRAINER_HOST_TELEMETRY_CERT_PATH",
            "GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256",
            "GX1_TRAINER_HOST_TELEMETRY_GPU_UUID",
            "GX1_TRAINER_HOST_TELEMETRY_TIMEOUT_SECONDS",
        }
    )
    if any(not value.isdigit() for value in numeric_values):
        pytest.skip("requires finite numeric cgroup controls")
    env = os.environ.copy()
    env.update(protected)
    return env


def test_capped_runner_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_trainer_guard_is_executable_and_has_valid_shell_syntax() -> None:
    assert TRAINER_GUARD.is_file()
    assert os.access(TRAINER_GUARD, os.X_OK)
    result = subprocess.run(
        ["bash", "-n", str(TRAINER_GUARD)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_trainer_guard_rejects_direct_unprotected_execution() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GX1_CAPPED_")
        and not key.startswith("GX1_TRAINER_")
    }
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 75
    assert "missing protected environment" in result.stderr


def test_trainer_guard_wall_clock_kills_cpu_process_group() -> None:
    env = _guard_env(
        device="cpu",
        nvidia_smi_path=Path("/bin/false"),
        max_wall_seconds=1,
    )
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/sleep", "30"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "wall-clock limit reached" in result.stderr
    assert "reason=wall_clock_limit_1s" in result.stderr


def _assert_guard_owned_process_stopped(pid: int) -> None:
    # A killed orphan can briefly await init's reap, but must not execute.
    stat_path = Path(f"/proc/{pid}/stat")
    process_state = None
    for _ in range(100):
        try:
            process_state = stat_path.read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            process_state = None
        if process_state in (None, "Z", "X"):
            break
        time.sleep(0.01)
    assert process_state in (None, "Z", "X"), (
        f"owned process {pid} survived guard exit: {process_state}"
    )


@pytest.mark.parametrize("ignore_term", [False, True], ids=["term", "kill-fallback"])
def test_trainer_guard_closed_stderr_still_stops_owned_group(
    tmp_path: Path, ignore_term: bool,
) -> None:
    """Losing the observer pipe must not strand an unguarded child."""
    identity = tmp_path / "owned-child.txt"
    child_source = """
import os
import signal
import sys
import time
from pathlib import Path
if sys.argv[2] == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(f'{os.getpid()} {os.getpgrp()}')
time.sleep(60)
"""
    env = _guard_env(
        device="cpu", nvidia_smi_path=Path("/bin/false"), max_wall_seconds=2,
    )
    guard = subprocess.Popen(
        ["bash", str(TRAINER_GUARD), sys.executable, "-c", child_source,
         str(identity), "ignore" if ignore_term else "term"],
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 5
        while not identity.exists() and guard.poll() is None:
            assert time.monotonic() < deadline, "guard child did not become ready"
            time.sleep(0.01)
        child_pid, group_id = map(int, identity.read_text().split())
        assert child_pid == group_id and group_id > 1 and group_id != os.getpgrp()
        assert guard.stderr is not None
        guard.stderr.close()
        guard.wait(timeout=20)
        assert guard.returncode != 0
        _assert_guard_owned_process_stopped(child_pid)
    finally:
        if identity.exists():
            child_pid, group_id = map(int, identity.read_text().split())
            assert child_pid == group_id and group_id > 1 and group_id != os.getpgrp()
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if guard.poll() is None:
            guard.kill()
        guard.wait(timeout=5)
        if guard.stderr is not None:
            guard.stderr.close()


@pytest.mark.parametrize("leader_exit", [False, True], ids=["term", "normal-exit"])
def test_trainer_guard_kills_term_ignoring_descendant_after_leader_exits(
    tmp_path: Path, leader_exit: bool,
) -> None:
    """Stopping/reaping the leader must never leave its child running."""
    descendant_file = tmp_path / "descendant.txt"
    child_source = """
import os
import signal
import sys
import time
from pathlib import Path

ready_read, ready_write = os.pipe()
descendant = os.fork()
if descendant == 0:
    os.close(ready_read)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    Path(sys.argv[1]).write_text(f'{os.getpid()} {os.getpgrp()}')
    os.write(ready_write, b'1')
    os.close(ready_write)
    while True:
        time.sleep(60)
os.close(ready_write)
assert os.read(ready_read, 1) == b'1'
os.close(ready_read)
if sys.argv[2] == 'exit':
    sys.exit(0)
while True:
    time.sleep(60)
"""
    env = _guard_env(
        device="cpu",
        nvidia_smi_path=Path("/bin/false"),
        max_wall_seconds=30 if leader_exit else 2,
    )
    guard = subprocess.Popen(
        [
            "bash", str(TRAINER_GUARD), sys.executable, "-c", child_source,
            str(descendant_file), "exit" if leader_exit else "wait",
        ],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _stdout, stderr = guard.communicate(timeout=30)
        assert guard.returncode == 75, stderr
        reason = (
            "orphaned_descendants_after_leader_exit"
            if leader_exit else "wall_clock_limit_2s"
        )
        assert f"reason={reason}" in stderr
        descendant_pid, group_id = map(int, descendant_file.read_text().split())
        assert group_id > 1 and group_id != os.getpgrp()
        _assert_guard_owned_process_stopped(descendant_pid)
    finally:
        if descendant_file.exists():
            _descendant_pid, group_id = map(int, descendant_file.read_text().split())
            assert group_id > 1 and group_id != os.getpgrp()
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if guard.poll() is None:
            guard.kill()
        guard.communicate(timeout=5)


@pytest.mark.parametrize("hazard", ["delayed-process-group", "wall-clock-rollback", "job-control", "graceful-cleanup", "external-kill-unavailable"])
def test_trainer_guard_deadline_survives_startup_and_clock_hazards(
    tmp_path: Path, hazard: str,
) -> None:
    """Adversarial helpers exist only in a private test copy of the guard."""
    source = TRAINER_GUARD.read_text(encoding="utf-8")
    owned_pid = tmp_path / "owned-child.pid"
    clock_calls = tmp_path / "clock.calls"
    term_count = tmp_path / "term.count"
    guard_log = tmp_path / "guard.log"
    guard_log.write_text("", encoding="utf-8")
    if hazard == "delayed-process-group":
        helper = tmp_path / "slow-setsid"
        helper.write_text(
            f"#!{sys.executable}\n"
            "import os, signal, sys, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"Path({str(owned_pid)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n"
            "os.execv('/usr/bin/setsid', ['/usr/bin/setsid', *sys.argv[1:]])\n",
            encoding="utf-8",
        )
        helper.chmod(0o755)
        source = source.replace("/usr/bin/setsid", str(helper))
    elif hazard == "wall-clock-rollback":
        helper = tmp_path / "rollback-date"
        helper.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "from pathlib import Path\n"
            f"counter = Path({str(clock_calls)!r})\n"
            "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
            "counter.write_text(str(count))\n"
            "if sys.argv[1:] == ['+%s']:\n"
            "    print(200 - count * 100)\n"
            "else:\n"
            "    print('2026-09-05T12:00:00Z' if count == 1 else '2020-01-01T00:00:00Z')\n",
            encoding="utf-8",
        )
        helper.chmod(0o755)
        source = source.replace("/bin/date", str(helper))
    elif hazard == "external-kill-unavailable":
        helper = tmp_path / "unavailable-external-kill"
        helper.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        helper.chmod(0o755)
        source = source.replace("/bin/kill", str(helper))
    fixture_guard = tmp_path / "guard-fixture.sh"
    fixture_guard.write_text(source, encoding="utf-8")
    child = (
        "import os, time; from pathlib import Path; "
        f"Path({str(owned_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    if hazard == "graceful-cleanup":
        child = (
            "import os, signal, sys, time\n"
            "from pathlib import Path\n"
            f"counter = Path({str(term_count)!r})\n"
            "def graceful_stop(_signum, _frame):\n"
            "    count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
            "    counter.write_text(str(count))\n"
            "    time.sleep(3)\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, graceful_stop)\n"
            f"Path({str(owned_pid)!r}).write_text(str(os.getpid()))\n"
            "while True: time.sleep(60)\n"
        )
    env = _guard_env(device="cpu", nvidia_smi_path=Path("/bin/false"), max_wall_seconds=2)
    env["GX1_TRAINER_GUARD_LOG_PATH"] = str(guard_log)
    command = ["bash"]
    if hazard == "job-control":
        command.append("-m")
    guard = subprocess.Popen(
        [*command, str(fixture_guard), sys.executable, "-c", child],
        cwd=REPO, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _stdout, stderr = guard.communicate(timeout=20)
        assert guard.returncode == 75, stderr
        assert "reason=wall_clock_limit_2s" in stderr
        assert "wall-clock limit reached" in stderr
        # Assert before cleanup: the finally safety net must not hide a leak.
        _assert_guard_owned_process_stopped(int(owned_pid.read_text()))
        if hazard == "delayed-process-group":
            assert "event=kill" in guard_log.read_text(encoding="utf-8")
        if hazard == "wall-clock-rollback":
            log = guard_log.read_text(encoding="utf-8")
            assert log.startswith("2026-09-05T12:00:00Z")
            assert "2020-01-01T00:00:00Z event=stop" in log
            assert int(clock_calls.read_text()) >= 2
        if hazard == "graceful-cleanup":
            assert term_count.read_text() == "1"
    finally:
        if owned_pid.exists():
            pid = int(owned_pid.read_text())
            assert pid > 1 and pid not in (os.getpid(), os.getpgrp())
            for target in (pid, -pid):
                try:
                    os.kill(target, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        # The private guard session also contains a not-yet-setsid child.
        try:
            os.killpg(guard.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        guard.communicate(timeout=5)


def test_attended_smoke_sigterm_unwinds_python_for_temp_scratch_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    registered: dict[int, object] = {}
    monkeypatch.setattr(
        trainer.signal,
        "signal",
        lambda signum, handler: registered.__setitem__(int(signum), handler),
    )

    trainer._install_attended_smoke_termination_handler()

    handler = registered[signal.SIGTERM]
    assert callable(handler)
    with pytest.raises(KeyboardInterrupt, match="attended smoke stopped"):
        handler(signal.SIGTERM, None)


def test_attended_preflight_marker_requires_an_attended_tier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    fifo = tmp_path / "preflight-ready"
    os.mkfifo(fifo, 0o600)
    reader = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    token = "a" * 64
    monkeypatch.setenv("GX1_TRAINER_ATTENDED_STAGE_FIFO", str(fifo))
    monkeypatch.setenv("GX1_TRAINER_ATTENDED_STAGE_TOKEN", token)

    try:
        trainer._announce_attended_preflight_ready(execution_tier="attended_only")
        expected = (
            f"gx1_attended_preflight_ready_v1:{token}\n".encode("ascii")
        )
        assert os.read(reader, len(expected)) == expected
        trainer._announce_attended_preflight_ready(
            execution_tier="attended_cpu_only"
        )
        assert os.read(reader, len(expected)) == expected
        with pytest.raises(RuntimeError, match="TIER_INVALID"):
            trainer._announce_attended_preflight_ready(execution_tier="canonical")
    finally:
        os.close(reader)


def _fake_nvidia_smi(tmp_path: Path, output: str, *, exit_code: int = 0) -> Path:
    path = tmp_path / "nvidia-smi"
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{output}'\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _stage_child(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "stage-child.sh"
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _run_staged_guard(
    tmp_path: Path,
    child: Path,
    *,
    data_preflight_seconds: int,
    model_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(TRAINER_GUARD), str(child)],
        cwd=REPO,
        env=_guard_env(
            device="cpu",
            nvidia_smi_path=Path("/bin/false"),
            execution_mode="attended_smoke",
            attended_stage_required=True,
            max_wall_seconds=data_preflight_seconds,
            model_max_wall_seconds=model_seconds,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # The guard deliberately gives a TERM-unwind window before KILLing a
        # process group. Under the one-core audit cap this test must not race
        # that safety window and kill the guard from the outside first.
        timeout=30,
        check=False,
    )


def test_staged_guard_rejects_missing_preflight_marker(tmp_path: Path) -> None:
    child = _stage_child(tmp_path, "sleep 30\n")

    result = _run_staged_guard(
        tmp_path, child, data_preflight_seconds=1, model_seconds=1
    )

    assert result.returncode == 75
    assert "reason=stage_data_preflight_wall_clock_limit_1s" in result.stderr
    assert "stage=data_preflight" in result.stderr


def test_staged_guard_accepts_one_valid_preflight_transition(tmp_path: Path) -> None:
    child = _stage_child(
        tmp_path,
        'printf "gx1_attended_preflight_ready_v1:%s\\n" "$GX1_TRAINER_ATTENDED_STAGE_TOKEN" > "$GX1_TRAINER_ATTENDED_STAGE_FIFO"\nsleep 2\n',
    )

    result = _run_staged_guard(
        tmp_path, child, data_preflight_seconds=3, model_seconds=3
    )

    assert result.returncode == 0, result.stderr
    assert "from=data_preflight to=model_smoke" in result.stderr


def test_staged_guard_rejects_invalid_preflight_marker(tmp_path: Path) -> None:
    child = _stage_child(
        tmp_path,
        'printf "%s\\n" "not-a-valid-marker" > "$GX1_TRAINER_ATTENDED_STAGE_FIFO"\nsleep 30\n',
    )

    result = _run_staged_guard(
        tmp_path, child, data_preflight_seconds=3, model_seconds=3
    )

    assert result.returncode == 75
    assert "reason=invalid_attended_stage_notification" in result.stderr


def test_staged_guard_rejects_duplicate_preflight_marker(tmp_path: Path) -> None:
    child = _stage_child(
        tmp_path,
        'printf "gx1_attended_preflight_ready_v1:%s\\n" "$GX1_TRAINER_ATTENDED_STAGE_TOKEN" > "$GX1_TRAINER_ATTENDED_STAGE_FIFO"\nprintf "gx1_attended_preflight_ready_v1:%s\\n" "$GX1_TRAINER_ATTENDED_STAGE_TOKEN" > "$GX1_TRAINER_ATTENDED_STAGE_FIFO"\nsleep 30\n',
    )

    result = _run_staged_guard(
        tmp_path, child, data_preflight_seconds=3, model_seconds=3
    )

    assert result.returncode == 75
    assert "from=data_preflight to=model_smoke" in result.stderr
    assert "reason=invalid_attended_stage_notification" in result.stderr


def test_staged_guard_enforces_separate_model_phase_timeout(tmp_path: Path) -> None:
    child = _stage_child(
        tmp_path,
        'printf "gx1_attended_preflight_ready_v1:%s\\n" "$GX1_TRAINER_ATTENDED_STAGE_TOKEN" > "$GX1_TRAINER_ATTENDED_STAGE_FIFO"\nsleep 30\n',
    )

    result = _run_staged_guard(
        tmp_path, child, data_preflight_seconds=3, model_seconds=1
    )

    assert result.returncode == 75
    assert "from=data_preflight to=model_smoke" in result.stderr
    assert "reason=stage_model_smoke_wall_clock_limit_1s" in result.stderr


def test_trainer_guard_accepts_complete_safe_cuda_telemetry(tmp_path: Path) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, 70, 100, 250, 1000")
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(device="cuda", nvidia_smi_path=nvidia_smi),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "[trainer_safety_guard] execution_mode=canonical device=cuda"
        in result.stderr
    )


def test_canonical_cuda_guard_requires_signed_host_query(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, 70, 100, 250, 1000")
    env = _guard_env(device="cuda", nvidia_smi_path=nvidia_smi)
    env["GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH"] = "/bin/false"
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "CUDA telemetry unavailable during preflight" in result.stderr


@pytest.mark.parametrize(
    ("telemetry", "expected"),
    [
        ("79, 70, 100, 250, 1000", "core temperature"),
        ("50, 91, 100, 250, 1000", "memory temperature"),
        ("50, 70, 251, 250, 1000", "GPU draw"),
        ("50, 70, 100, 251, 1000", "configured GPU power limit"),
    ],
)
def test_trainer_guard_rejects_unsafe_cuda_preflight(
    tmp_path: Path,
    telemetry: str,
    expected: str,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, telemetry)
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(device="cuda", nvidia_smi_path=nvidia_smi),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert expected in result.stderr


def test_trainer_guard_rejects_gpu_memory_residency_above_bound(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, 70, 100, 250, 12289")
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(device="cuda", nvidia_smi_path=nvidia_smi),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "GPU memory used 12289MiB exceeds 12288MiB" in result.stderr


def test_trainer_guard_rejects_unavailable_cuda_telemetry(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(
        tmp_path,
        "telemetry unavailable",
        exit_code=1,
    )
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(device="cuda", nvidia_smi_path=nvidia_smi),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "CUDA telemetry unavailable during preflight" in result.stderr


def test_trainer_guard_rejects_missing_signed_memory_junction_for_attended_smoke(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, N/A, 100, 390, 1000")
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(
            device="cuda",
            nvidia_smi_path=nvidia_smi,
            execution_mode="attended_smoke",
            max_power_limit_w=390,
            max_power_draw_w=250,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "CUDA telemetry unavailable during preflight" in result.stderr


def test_trainer_guard_allows_cpu_attended_recovery_without_cuda_telemetry(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(
            device="cpu",
            nvidia_smi_path=Path("/bin/false"),
            execution_mode="attended_cpu_smoke",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "execution_mode=attended_cpu_smoke" in result.stderr
    assert "attended_cpu_only" in result.stderr


def test_trainer_guard_rejects_retired_research_smoke_execution_mode(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, N/A, 100, 390, 1000")
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(
            device="cuda",
            nvidia_smi_path=nvidia_smi,
            execution_mode="research_smoke",
            max_power_limit_w=390,
            max_power_draw_w=250,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert (
        "must be canonical, attended_smoke, attended_cpu_smoke or cuda_producer"
        in result.stderr
    )


def test_trainer_guard_rejects_missing_signed_memory_junction_for_canonical_cuda(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, N/A, 100, 250, 1000")
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(device="cuda", nvidia_smi_path=nvidia_smi),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "CUDA telemetry unavailable during preflight" in result.stderr


def test_attended_smoke_rejects_draw_above_operator_authorized_ceiling(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, 70, 391, 390, 1000")
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=_guard_env(
            device="cuda",
            nvidia_smi_path=nvidia_smi,
            execution_mode="attended_smoke",
            max_power_limit_w=390,
            max_power_draw_w=390,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "GPU draw 391W exceeds 390W" in result.stderr


def test_trainer_guard_persists_terminal_telemetry_when_given_sidecar(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, 70, 100, 390, 1000")
    guard_log = tmp_path / "guard.log"
    guard_log.write_text("", encoding="utf-8")
    env = _guard_env(
        device="cuda",
        nvidia_smi_path=nvidia_smi,
        execution_mode="attended_smoke",
        max_power_limit_w=390,
        max_power_draw_w=390,
    )
    env["GX1_TRAINER_GUARD_LOG_PATH"] = str(guard_log)
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/true"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_text = guard_log.read_text(encoding="utf-8")
    assert "event=start execution_mode=attended_smoke" in log_text
    assert "event=telemetry phase=preflight" in log_text
    assert "event=exit child_status=0" in log_text
    assert "telemetry_samples=1" in log_text
    assert "peak_core_temp_c=50" in log_text
    assert "peak_memory_temp_c=70" in log_text
    assert "peak_power_draw_w=100" in log_text
    assert "peak_memory_used_mib=1000" in log_text


def _sequenced_nvidia_smi(
    tmp_path: Path,
    *,
    later_output: str,
    later_exit_code: int = 0,
) -> Path:
    counter = tmp_path / "telemetry-call-count"
    path = tmp_path / "nvidia-smi-sequenced"
    path.write_text(
        "#!/bin/sh\n"
        f"counter='{counter}'\n"
        "count=0\n"
        "[ ! -f \"$counter\" ] || count=$(cat \"$counter\")\n"
        "count=$((count + 1))\n"
        "printf '%s' \"$count\" >\"$counter\"\n"
        "if [ \"$count\" -eq 1 ]; then\n"
        # The production guard queries all five values, including residency.
        # Keep the first sequenced sample a fully valid telemetry record so
        # the second sample actually exercises the running-child stop path.
        "  printf '%s\\n' '50, 70, 100, 250, 1000'\n"
        "  exit 0\n"
        "fi\n"
        f"printf '%s\\n' '{later_output}'\n"
        f"exit {later_exit_code}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_trainer_guard_exit_records_peaks_across_short_cuda_session(
    tmp_path: Path,
) -> None:
    nvidia_smi = _sequenced_nvidia_smi(
        tmp_path,
        later_output="60, 80, 200, 250, 2000",
    )
    guard_log = tmp_path / "guard.log"
    guard_log.write_text("", encoding="utf-8")
    env = _guard_env(device="cuda", nvidia_smi_path=nvidia_smi)
    env["GX1_TRAINER_GUARD_LOG_PATH"] = str(guard_log)
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/sleep", "2"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_text = guard_log.read_text(encoding="utf-8")
    assert "event=exit child_status=0" in log_text
    assert "telemetry_samples=" in log_text
    assert "peak_core_temp_c=60" in log_text
    assert "peak_memory_temp_c=80" in log_text
    assert "peak_power_draw_w=200" in log_text
    assert "peak_memory_used_mib=2000" in log_text


def test_trainer_guard_kills_running_group_when_telemetry_disappears(
    tmp_path: Path,
) -> None:
    nvidia_smi = _sequenced_nvidia_smi(
        tmp_path,
        later_output="telemetry unavailable",
        later_exit_code=1,
    )
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/sleep", "60"],
        cwd=REPO,
        # This exercises loss of the second mock sample, not the wall-clock
        # stop. The five-second fixture default can win that race under the
        # audit CPU cap before the running-child sample gets scheduled.
        env=_guard_env(
            device="cuda", nvidia_smi_path=nvidia_smi, max_wall_seconds=30,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )

    assert result.returncode == 75
    assert "reason=telemetry_unavailable" in result.stderr
    assert "CUDA telemetry became unavailable" in result.stderr
    assert (tmp_path / "telemetry-call-count").read_text() == "2"


def test_trainer_guard_kills_running_group_on_thermal_breach(
    tmp_path: Path,
) -> None:
    nvidia_smi = _sequenced_nvidia_smi(
        tmp_path,
        later_output="50, 91, 100, 250, 1000",
    )
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/sleep", "60"],
        cwd=REPO,
        # Leave the thermal mock time to supply its second sample, with no
        # competing short fixture wall-clock limit.
        env=_guard_env(
            device="cuda", nvidia_smi_path=nvidia_smi, max_wall_seconds=30,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )

    assert result.returncode == 75
    assert "reason=memory_temperature" in result.stderr
    assert "GPU safety threshold breached" in result.stderr
    assert (tmp_path / "telemetry-call-count").read_text() == "2"


def test_trainer_guard_persists_child_stdio_when_path_is_precreated(
    tmp_path: Path,
) -> None:
    stdio_log = tmp_path / "trainer.log"
    stdio_log.write_text("", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/bash", "-c", "echo durable-child-output"],
        cwd=REPO,
        env={
            **_guard_env(device="cpu", nvidia_smi_path=Path("/bin/true")),
            "GX1_TRAINER_STDIO_LOG_PATH": str(stdio_log),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert stdio_log.read_text(encoding="utf-8") == "durable-child-output\n"


@pytest.mark.parametrize("wrapper", ["/usr/bin/env", "/bin/bash", "/bin/sh"])
def test_capped_runner_rejects_env_and_shell_targets_before_nested_fast_path(
    wrapper: str,
) -> None:
    result = _run("audit", "4G", "512M", wrapper, "/bin/true")

    assert result.returncode == 75
    assert "env and shell wrappers are forbidden as capped targets" in result.stderr
    assert "nested capped job" not in result.stderr


@pytest.mark.parametrize(
    "target",
    [
        ("/bin/echo", TRAINER_MODULE),
        (sys.executable, "-m", TRAINER_MODULE, "--train"),
        (sys.executable, "-c", f"import {TRAINER_MODULE}"),
    ],
)
def test_audit_class_rejects_direct_or_disguised_trainer_before_nested_fast_path(
    target: tuple[str, ...],
) -> None:
    result = _run("audit", "4G", "512M", *target)

    assert result.returncode == 75
    assert "canonical trainer requires --class trainer" in result.stderr
    assert "nested capped job" not in result.stderr


@pytest.mark.parametrize("fake_python", ["/tmp/notpython", "/usr/bin/python3"])
def test_trainer_class_rejects_noncanonical_python_before_nested_fast_path(
    fake_python: str,
) -> None:
    result = _run(
        "trainer",
        "10G",
        "512M",
        fake_python,
        "-m",
        TRAINER_MODULE,
        "--train",
    )

    assert result.returncode == 75
    assert "canonical trainer module as a direct target" in result.stderr
    assert "nested capped job" not in result.stderr


@pytest.mark.parametrize("train_flags", [(), ("--train", "--train")])
def test_trainer_class_requires_exactly_one_train_flag(
    train_flags: tuple[str, ...],
) -> None:
    result = _run(
        "trainer",
        "10G",
        "512M",
        sys.executable,
        "-m",
        TRAINER_MODULE,
        *train_flags,
    )

    assert result.returncode == 75
    assert "canonical --train mode exactly once" in result.stderr
    assert "nested capped job" not in result.stderr


@pytest.mark.parametrize(
    "device_args",
    [(), ("--device", "other"), ("--device", "cpu", "--device", "cuda")],
)
def test_trainer_class_requires_one_canonical_device(
    device_args: tuple[str, ...],
) -> None:
    result = _run(
        "trainer",
        "10G",
        "512M",
        sys.executable,
        "-m",
        TRAINER_MODULE,
        "--train",
        *device_args,
    )

    assert result.returncode == 75
    assert "requires exactly one canonical --device cpu|cuda" in result.stderr
    assert "nested capped job" not in result.stderr


@pytest.mark.parametrize(
    ("job_class", "memory", "swap", "expected"),
    [
        ("audit", "5G", "512M", "audit jobs may request at most 4G"),
        ("trainer", "21G", "512M", "safety ceiling (20G)"),
        ("audit", "4G", "1G", "safety ceiling (512M)"),
    ],
)
def test_capacity_ceilings_are_enforced_before_nested_fast_path(
    job_class: str,
    memory: str,
    swap: str,
    expected: str,
) -> None:
    target = (
        (sys.executable, "-m", TRAINER_MODULE, "--train")
        if job_class == "trainer"
        else ("/bin/true",)
    )
    result = _run(job_class, memory, swap, *target)

    assert result.returncode == 75
    assert expected in result.stderr
    assert "nested capped job" not in result.stderr


def test_capped_runner_preserves_hard_limits_global_lock_and_validation_order() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    guard_source = TRAINER_GUARD.read_text(encoding="utf-8")

    assert "SAFE_AUDIT_MEMORY_KIB=$((4 * 1024 * 1024))" in source
    assert "SAFE_JOB_MEMORY_KIB=$((20 * 1024 * 1024))" in source
    assert "SAFE_JOB_SWAP_KIB=$((512 * 1024))" in source
    assert "exec 9>>\"$LOCK_PATH\"" in source
    assert "flock -n 9" in source
    assert 'LOCK_PATH="$("$CANONICAL_TRAINER_PYTHON" -I -B "$CAPPED_EXECUTION_OWNER" --lock-path)"' in source
    assert "XDG_RUNTIME_DIR" not in source
    assert "/tmp/gx1-heavy-job" not in source
    assert "umask 077" in source
    assert 'umask "$LOCK_UMASK"' in source
    assert 'CAPPED_EXECUTION_OWNER="$REPO_ROOT/gx1/contracts/gx1_capped_execution_v1.py"' in source
    assert source.count("--verify-lock-ancestry") == 2
    assert '-p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax="$SWAP"' in source
    assert '--setenv=GX1_CAPPED_SWAP_BYTES="$((requested_swap_kib * 1024))"' in source
    assert '--setenv=GX1_CAPPED_TASKS_MAX="$TASKS_MAX"' in source
    assert "TRAINER_MAX_WALL_SECONDS=7200" in source
    assert "TRAINER_MODEL_MAX_WALL_SECONDS=7200" in source
    assert "TRAINER_MAX_WALL_SECONDS=600" in source
    assert "TRAINER_MODEL_MAX_WALL_SECONDS=300" in source
    assert 'if [[ "$TRAINER_ATTENDED_STAGE_REQUIRED" == true ]]; then' in source
    assert "hardware diagnostic remains a single five-minute run" in source
    assert "TRAINER_GPU_MAX_CORE_TEMP_C=65" in source
    assert "TRAINER_GPU_MAX_MEMORY_TEMP_C=80" in source
    assert "TRAINER_GPU_MAX_POWER_LIMIT_W=160" in source
    assert "TRAINER_GPU_MAX_POWER_DRAW_W=170" in source
    assert "TRAINER_GPU_MAX_MEMORY_USED_MIB=12288" in source
    assert "TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1" in source
    assert "TRAINER_EXECUTION_MODE=canonical" in source
    assert "CUDA_PRODUCER_MODULE=gx1.scripts.evaluate_entry_candidate_selective_edge_v1" in source
    assert "TECHNICAL_VALIDATION_PRODUCER_MODULE=gx1.scripts.validate_entry_model_native_technical_checkpoint_v1" in source
    assert "--cuda-producer" in source
    assert "--cuda-producer requires exactly one --device cuda" in source
    assert "CUDA_PRODUCER_GUARD=true" in source
    assert "signed host telemetry query is unavailable" in source
    assert "--attended-smoke" in source
    assert "disabled after the WSL/GPU reset" in source
    assert "TRAINER_MAX_WALL_SECONDS=86400" not in source
    assert "TRAINER_MODEL_MAX_WALL_SECONDS=86400" not in source
    assert "CPU_AFFINITY=0-7" in source
    assert "NUMERICAL_THREAD_COUNT=8" in source
    assert "TRAINER_HOST_TELEMETRY_URL='http://172.30.224.1:38128/gx1/v1/telemetry/'" in source
    assert "TRAINER_HOST_TELEMETRY_CERT_SHA256='25c9260c2168db53cf58c5f963f2008d5163d80aa69699c5726e0680ed74eb6e'" in source
    attended_block = source.split('if [[ "$ATTENDED_SMOKE" == true ]]; then', 1)[1]
    assert "TRAINER_GPU_MAX_POWER_DRAW_W=390" not in attended_block
    assert "TRAINER_GPU_MAX_POWER_DRAW_W=170" not in attended_block
    assert (
        '--setenv=GX1_TRAINER_MAX_WALL_SECONDS="$TRAINER_MAX_WALL_SECONDS"'
        in source
    )
    assert (
        '--setenv=GX1_TRAINER_MODEL_MAX_WALL_SECONDS="$TRAINER_MODEL_MAX_WALL_SECONDS"'
        in source
    )
    assert '"$GX1_GPU_GUARD_PATH" "$@"' in source
    assert "GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH" in guard_source
    assert "GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256" in guard_source
    assert "CUDA telemetry unavailable" in guard_source
    assert "GX1_TRAINER_EXECUTION_MODE" in guard_source
    assert "GX1_TRAINER_GPU_MAX_POWER_DRAW_W" in guard_source
    assert "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB" in guard_source
    assert "GX1_TRAINER_NVIDIA_SMI_PATH" not in guard_source
    assert "cuda_producer" in guard_source
    assert "producer cgroup is reserved here for CUDA inference only" in guard_source
    assert "GX1_TRAINER_GUARD_LOG_PATH" in source
    assert "event=heartbeat" in guard_source
    assert "event=exit child_status=$child_status" in guard_source
    assert '"$memory_temp" =~ ^[0-9]+' in guard_source
    assert 'builtin kill -TERM -- "-$child_pid"' in guard_source
    assert 'builtin kill -KILL -- "-$child_pid"' in guard_source
    assert 'builtin kill -0 "$child_pid"' in guard_source
    assert "/bin/kill" not in guard_source
    assert "stage_elapsed >= stage_limit" in guard_source
    assert "gx1_attended_preflight_ready_v1" in guard_source
    validation_call = source.index('\nvalidate_target_command "$@"\n')
    nested_fast_path = source.index('\nif [[ -n "${GX1_CAPPED_CLASS:-}"')
    assert validation_call < nested_fast_path
    host_preflight = source.index("\nhost_total_kib=")
    lock_acquisition = source.index('\nexec 9>>"$LOCK_PATH"')
    nested_proof = source.index("--verify-lock-ancestry")
    assert nested_fast_path < nested_proof < host_preflight < lock_acquisition
    assert source.index("nested capped job parent scope proof failed") < nested_proof
    assert source.index("nested guarded CUDA device differs") < nested_proof
    scope_guard = source.split("SCOPE_GUARD='", 1)[1]
    assert scope_guard.index("pids.max scope proof failed") < scope_guard.index("--verify-lock-ancestry")
    assert scope_guard.index("--verify-lock-ancestry") < scope_guard.index("exec /usr/bin/taskset")
    assert 'gx1-capped-scope "$CANONICAL_TRAINER_PYTHON" "$CAPPED_EXECUTION_OWNER" "$@"' in source


def test_capped_runner_source_binds_the_signed_windows_bridge() -> None:
    """Canonical CUDA never falls back to an ambient WSL GPU sensor."""
    source = RUNNER.read_text(encoding="utf-8")

    assert 'TRAINER_HOST_TELEMETRY_QUERY_PATH="$REPO_ROOT/scripts/gx1_host_telemetry_bridge_query.sh"' in source
    assert "signed_windows_bridge" in source
    assert "resolve_nvidia_smi_path" not in source
    assert "command -v nvidia-smi" not in source


def _require_canonical_audit_integration() -> None:
    required = {
        "GX1_CAPPED_CLASS": "audit",
        "GX1_CAPPED_MEMORY_BYTES": str(4 * 1024**3),
        "GX1_CAPPED_SWAP_BYTES": str(512 * 1024**2),
        "GX1_CAPPED_TASKS_MAX": "64",
    }
    if any(os.environ.get(key) != value for key, value in required.items()):
        pytest.skip("requires the canonical 4G/512M capped audit scope")


@pytest.mark.parametrize("runtime", ["inherited", "unset", "alternate"])
def test_matching_nested_audit_scope_can_execute_a_nontrainer_target(tmp_path: Path, runtime: str) -> None:
    _require_canonical_audit_integration()
    env = os.environ.copy()
    alternate = tmp_path / "alternate-runtime"
    alternate.mkdir(mode=0o700)
    if runtime == "unset":
        env.pop("XDG_RUNTIME_DIR", None)
    elif runtime == "alternate":
        env["XDG_RUNTIME_DIR"] = str(alternate)
    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "--class",
            "audit",
            "--mem",
            "4G",
            "--swap",
            "512M",
            "--",
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import os; assert not os.path.exists('/proc/self/fd/9'); print('closed-fd-target')",
        ],
        cwd=REPO,
        env=env,
        close_fds=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "closed-fd-target\n"
    assert not (alternate / "gx1-heavy-job.lock").exists()


def test_cwd_changed_wrapper_preserves_actual_outer_capped_lock_owner(tmp_path: Path) -> None:
    _require_canonical_audit_integration()
    wrapper = tmp_path / "gx1-cwd-lock-fixture-wrapper.sh"
    assert not (REPO / wrapper.name).exists()
    wrapper.write_text(
        "#!/bin/bash\nset -eu\n"
        f"cd {shlex.quote(str(REPO))}\n"
        f"bash {shlex.quote(str(RUNNER))} --class audit --mem 4G --swap 512M -- /bin/true\n"
        "printf 'wrapper-done\\n'\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", wrapper.name], cwd=tmp_path, close_fds=True,
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "wrapper-done\n"


def test_alternate_xdg_cannot_start_a_second_top_level_job(tmp_path: Path) -> None:
    _require_canonical_audit_integration()
    alternate = tmp_path / "alternate-runtime"
    alternate.mkdir(mode=0o700)
    forbidden_dispatch = tmp_path / "systemd-run"
    forbidden_dispatch.write_text(
        "#!/bin/sh\nprintf 'UNEXPECTED_SCOPE_DISPATCH\\n' >&2\nexit 99\n",
        encoding="utf-8",
    )
    forbidden_dispatch.chmod(0o700)
    env = os.environ.copy()
    for name in (
        "GX1_CAPPED_CLASS", "GX1_CAPPED_MEMORY_BYTES",
        "GX1_CAPPED_SWAP_BYTES", "GX1_CAPPED_TASKS_MAX",
    ):
        env.pop(name, None)
    env["XDG_RUNTIME_DIR"] = str(alternate)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        [
            "bash", str(RUNNER), "--class", "audit", "--mem", "4G",
            "--swap", "512M", "--", "/bin/true",
        ],
        cwd=REPO,
        env=env,
        close_fds=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 75, result.stderr
    assert (
        f"another GX1 heavy job owns the exclusive lock: /run/user/{os.getuid()}/gx1-heavy-job.lock"
        in result.stderr
    )
    assert "UNEXPECTED_SCOPE_DISPATCH" not in result.stderr
    assert not (alternate / "gx1-heavy-job.lock").exists()
