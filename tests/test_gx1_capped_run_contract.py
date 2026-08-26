from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/gx1_capped_run.sh"
TRAINER_GUARD = REPO / "scripts/gx1_guarded_trainer_exec.sh"
TRAINER_MODULE = "gx1.models.entry_v10.entry_v10_ctx_train_v3"


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
    if device == "cuda":
        bridge_query, bridge_certificate, bridge_certificate_sha256 = (
            _fake_host_telemetry_query(nvidia_smi_path)
        )
    else:
        bridge_query = Path("/bin/false")
        bridge_certificate = Path("/etc/hosts")
        bridge_certificate_sha256 = hashlib.sha256(
            bridge_certificate.read_bytes()
        ).hexdigest()
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
        "GX1_TRAINER_NVIDIA_SMI_PATH": str(nvidia_smi_path),
        "GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH": str(bridge_query),
        "GX1_TRAINER_HOST_TELEMETRY_URL": (
            "http://127.0.0.1:38127/gx1/v1/telemetry/"
        ),
        "GX1_TRAINER_HOST_TELEMETRY_CERT_PATH": str(bridge_certificate),
        "GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256": bridge_certificate_sha256,
        "GX1_TRAINER_HOST_TELEMETRY_GPU_UUID": (
            "GPU-8c6ac5f1-4254-6cec-9780-44b019cafd29"
        ),
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
            "GX1_TRAINER_NVIDIA_SMI_PATH",
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


def _fake_host_telemetry_query(nvidia_smi_path: Path) -> tuple[Path, Path, str]:
    """Expose a protected canonical bridge fixture with nvidia-like rows.

    The bridge query itself has dedicated RSA/schema tests.  These guard tests
    deliberately supply a source-owned executable which emits the same five
    telemetry columns as their former native-driver fixture, so the existing
    guard thresholds are exercised without weakening canonical's bridge-only
    routing.
    """

    directory = nvidia_smi_path.parent
    query = directory / "host-telemetry-query"
    certificate = directory / "host-telemetry-public.pem"
    query.write_text(
        "#!/bin/sh\n"
        f'exec "{nvidia_smi_path}" "$@"\n',
        encoding="utf-8",
    )
    query.chmod(0o755)
    certificate.write_text("test public certificate\n", encoding="utf-8")
    return query, certificate, hashlib.sha256(certificate.read_bytes()).hexdigest()


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


def test_canonical_cuda_guard_uses_host_bridge_not_wsl_nvidia_smi(
    tmp_path: Path,
) -> None:
    bridge_backed_nvidia = _fake_nvidia_smi(tmp_path, "50, 70, 100, 250, 1000")
    env = _guard_env(device="cuda", nvidia_smi_path=bridge_backed_nvidia)
    # The source-owned fake bridge continues to use bridge_backed_nvidia, but
    # canonical's ordinary WSL driver path is unavailable.  Success therefore
    # proves the guard routes canonical telemetry only through the bridge.
    env["GX1_TRAINER_NVIDIA_SMI_PATH"] = "/bin/false"
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


def test_canonical_cuda_guard_rejects_unbound_host_certificate(
    tmp_path: Path,
) -> None:
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, 70, 100, 250, 1000")
    env = _guard_env(device="cuda", nvidia_smi_path=nvidia_smi)
    env["GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256"] = "unbound"
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
    assert "certificate is not source-bound" in result.stderr


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


def test_trainer_guard_allows_only_literal_wsl_memory_na_for_attended_smoke(
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

    assert result.returncode == 0, result.stderr
    assert "execution_mode=attended_smoke" in result.stderr
    assert "attended_only" in result.stderr


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
    assert "must be canonical, attended_smoke or attended_cpu_smoke" in result.stderr


def test_trainer_guard_rejects_memory_na_for_canonical_cuda(
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
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, N/A, 391, 390, 1000")
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
    nvidia_smi = _fake_nvidia_smi(tmp_path, "50, N/A, 100, 390, 1000")
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
    assert "peak_memory_temp_c=N/A" in log_text
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
        ["bash", str(TRAINER_GUARD), "/bin/sleep", "30"],
        cwd=REPO,
        env=_guard_env(device="cuda", nvidia_smi_path=nvidia_smi),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "reason=telemetry_unavailable" in result.stderr
    assert "CUDA telemetry became unavailable" in result.stderr


def test_trainer_guard_kills_running_group_on_thermal_breach(
    tmp_path: Path,
) -> None:
    nvidia_smi = _sequenced_nvidia_smi(
        tmp_path,
        later_output="50, 91, 100, 250, 1000",
    )
    result = subprocess.run(
        ["bash", str(TRAINER_GUARD), "/bin/sleep", "30"],
        cwd=REPO,
        env=_guard_env(device="cuda", nvidia_smi_path=nvidia_smi),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        check=False,
    )

    assert result.returncode == 75
    assert "reason=memory_temperature" in result.stderr
    assert "GPU safety threshold breached" in result.stderr


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
    assert '-p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax="$SWAP"' in source
    assert '--setenv=GX1_CAPPED_SWAP_BYTES="$((requested_swap_kib * 1024))"' in source
    assert '--setenv=GX1_CAPPED_TASKS_MAX="$TASKS_MAX"' in source
    assert "TRAINER_MAX_WALL_SECONDS=1200" in source
    assert "TRAINER_MODEL_MAX_WALL_SECONDS=1200" in source
    assert "TRAINER_MAX_WALL_SECONDS=600" in source
    assert "TRAINER_MODEL_MAX_WALL_SECONDS=300" in source
    assert 'if [[ "$TRAINER_ATTENDED_STAGE_REQUIRED" == true ]]; then' in source
    assert "hardware diagnostic remains a single five-minute run" in source
    assert "TRAINER_GPU_MAX_CORE_TEMP_C=78" in source
    assert "TRAINER_GPU_MAX_MEMORY_TEMP_C=90" in source
    assert "TRAINER_GPU_MAX_POWER_LIMIT_W=250" in source
    assert "TRAINER_GPU_MAX_POWER_DRAW_W=250" in source
    assert "TRAINER_GPU_MAX_MEMORY_USED_MIB=24576" in source
    assert "TRAINER_GPU_MONITOR_INTERVAL_SECONDS=2" in source
    assert "TRAINER_EXECUTION_MODE=canonical" in source
    assert 'CANONICAL_HOST_TELEMETRY_CERT_SHA256=\'unbound\'' in source
    assert "canonical host telemetry certificate is not source-bound" in source
    assert "GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH" in source
    assert "GX1_TRAINER_HOST_TELEMETRY_CERT_SHA256" in source
    assert "--attended-smoke" in source
    assert "disabled after the WSL/GPU reset" in source
    assert "TRAINER_MAX_WALL_SECONDS=86400" not in source
    assert "TRAINER_MODEL_MAX_WALL_SECONDS=86400" not in source
    assert "TRAINER_GPU_MAX_CORE_TEMP_C=70" in source
    assert "TRAINER_GPU_MAX_POWER_DRAW_W=390" in source
    assert (
        '--setenv=GX1_TRAINER_MAX_WALL_SECONDS="$TRAINER_MAX_WALL_SECONDS"'
        in source
    )
    assert (
        '--setenv=GX1_TRAINER_MODEL_MAX_WALL_SECONDS="$TRAINER_MODEL_MAX_WALL_SECONDS"'
        in source
    )
    assert '"$GX1_GPU_GUARD_PATH" "$@"' in source
    assert (
        "--query-gpu=temperature.gpu,temperature.memory,power.draw,power.limit,memory.used"
        in guard_source
    )
    assert "CUDA telemetry unavailable" in guard_source
    assert "GX1_TRAINER_EXECUTION_MODE" in guard_source
    assert "GX1_TRAINER_GPU_MAX_POWER_DRAW_W" in guard_source
    assert "GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB" in guard_source
    assert "GX1_TRAINER_HOST_TELEMETRY_QUERY_PATH" in guard_source
    assert "canonical host telemetry certificate is not source-bound" in guard_source
    assert "GX1_TRAINER_GUARD_LOG_PATH" in source
    assert "event=heartbeat" in guard_source
    assert "event=exit child_status=$child_status" in guard_source
    assert '"$memory_temp" == N/A' in guard_source
    assert '/bin/kill -TERM -- "-$child_pid"' in guard_source
    assert '/bin/kill -KILL -- "-$child_pid"' in guard_source
    assert "stage_elapsed >= stage_limit" in guard_source
    assert "gx1_attended_preflight_ready_v1" in guard_source
    validation_call = source.index('\nvalidate_target_command "$@"\n')
    nested_fast_path = source.index('\nif [[ -n "${GX1_CAPPED_CLASS:-}"')
    assert validation_call < nested_fast_path


def test_capped_runner_resolves_only_explicit_system_nvidia_smi_paths() -> None:
    """WSL driver placement may differ, but caller-controlled PATH is unsafe."""
    source = RUNNER.read_text(encoding="utf-8")

    assert "resolve_nvidia_smi_path()" in source
    assert "for candidate in /usr/bin/nvidia-smi /usr/lib/wsl/lib/nvidia-smi; do" in source
    assert "command -v nvidia-smi" not in source
    assert 'TRAINER_NVIDIA_SMI_PATH="$(resolve_nvidia_smi_path || true)"' in source


def test_matching_nested_audit_scope_can_execute_a_nontrainer_target() -> None:
    required = {
        "GX1_CAPPED_CLASS": "audit",
        "GX1_CAPPED_MEMORY_BYTES": str(4 * 1024**3),
        "GX1_CAPPED_SWAP_BYTES": str(512 * 1024**2),
        "GX1_CAPPED_TASKS_MAX": "64",
    }
    if any(os.environ.get(key) != value for key, value in required.items()):
        pytest.skip("requires the canonical 4G/512M capped audit scope")

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
            "/bin/true",
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
