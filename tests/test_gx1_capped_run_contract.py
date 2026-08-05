from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/gx1_capped_run.sh"
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


def test_capped_runner_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
    ("job_class", "memory", "swap", "expected"),
    [
        ("audit", "5G", "512M", "audit jobs may request at most 4G"),
        ("trainer", "11G", "512M", "safety ceiling (10G)"),
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

    assert "SAFE_AUDIT_MEMORY_KIB=$((4 * 1024 * 1024))" in source
    assert "SAFE_JOB_MEMORY_KIB=$((10 * 1024 * 1024))" in source
    assert "SAFE_JOB_SWAP_KIB=$((512 * 1024))" in source
    assert "exec 9>>\"$LOCK_PATH\"" in source
    assert "flock -n 9" in source
    assert '-p MemoryMax="$MEM" -p MemoryHigh="$MEM" -p MemorySwapMax="$SWAP"' in source
    assert '--setenv=GX1_CAPPED_SWAP_BYTES="$((requested_swap_kib * 1024))"' in source
    assert '--setenv=GX1_CAPPED_TASKS_MAX="$TASKS_MAX"' in source
    validation_call = source.index('\nvalidate_target_command "$@"\n')
    nested_fast_path = source.index('\nif [[ -n "${GX1_CAPPED_CLASS:-}"')
    assert validation_call < nested_fast_path


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
