from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from gx1.contracts.local_random_access_campaign_v2 import (
    BOOT_SCHEMA,
    INVOCATION_SCHEMA,
    PLAN_SCHEMA,
    RandomAccessCampaignError,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    next_action,
    require_plan,
)
from gx1.scripts.local_random_access_campaign_v2 import (
    begin_invocation,
    confirm_reboot,
    inspect_campaign,
    prepare_reboot,
    record_invocation,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_bytes(canonical_bytes(value))
    else:
        path.write_text(str(value), encoding="utf-8")


def _seal(value: dict, key: str) -> dict:
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result


def _boot(number: int, minute: int) -> dict:
    return _seal(
        {
            "schema_version": BOOT_SCHEMA,
            "computer_name": "GX1-3090",
            "last_boot_utc": f"2026-09-11T10:{minute:02d}:00+00:00",
            "boot_id": number,
        },
        "identity_sha256",
    )


def _binding(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def _fixture(tmp_path: Path) -> tuple[dict, Path, list[dict]]:
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    runner = repo / "scripts/gx1_capped_run.sh"
    python = repo / ".venv/bin/python"
    _write(runner, "#!/bin/sh\n")
    _write(python, "#!/bin/sh\n")
    guard_sources = {}
    for name in ("runner", "guard", "query", "certificate"):
        path = runner if name == "runner" else repo / f"evidence/{name}.bin"
        if name != "runner":
            _write(path, name)
        guard_sources[name] = _binding(path)
    controller_sources = {}
    for name in ("controller", "observer", "campaign_cli"):
        path = repo / f"controller/{name}.txt"
        _write(path, name)
        controller_sources[name] = _binding(path)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "GX1 Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    source_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    definitions = [
        ("smoke_arm", 4, 3, "COMPLETE", "GENESIS", None, "ATOMIC_UPDATE", 0, 0),
        ("smoke_arm", 8, 3, "COMPLETE", "GENESIS", None, "ATOMIC_UPDATE", 0, 0),
        ("smoke_arm", 16, 3, "COMPLETE", "GENESIS", None, "ATOMIC_UPDATE", 0, 0),
        ("resume_proof", 16, 1, "COMPLETE", "PREVIOUS_RECEIPT_AFTER", 3, "ATOMIC_UPDATE", 0, 0),
        ("epoch1_window", 16, 5, "RESUMABLE", "GENESIS", None, "ATOMIC_UPDATE", 0, 2),
        ("epoch1_window", 16, 5, "COMPLETE", "PREVIOUS_RECEIPT_AFTER", 5, "ATOMIC_UPDATE", 1, 2),
        ("full_val", 16, None, "COMPLETE", "PREVIOUS_RECEIPT_AFTER", 6, "READ_ONLY", 0, 0),
    ]
    invocations: list[dict] = []
    invocation_bindings = []
    smoke16_pointer = runtime / "smoke-b16/RESUME_POINTER.json"
    epoch_pointer = runtime / "epoch1/RESUME_POINTER.json"
    for number, definition in enumerate(definitions, 1):
        kind, batch, budget, expected, before, predecessor, write_mode, window, windows = definition
        if kind == "smoke_arm":
            pointer = runtime / f"smoke-b{batch}/RESUME_POINTER.json"
        elif kind == "resume_proof":
            pointer = smoke16_pointer
        else:
            pointer = epoch_pointer
        manifest_path = tmp_path / "manifests" / f"invocation-{number:04d}.json"
        _write(manifest_path, {"source_commit": source_commit, "number": number})
        outer = [
            str(runner),
            "--class",
            "trainer",
            "--mem",
            "20G",
            "--swap",
            "512M",
        ]
        if kind in {"smoke_arm", "resume_proof"}:
            outer.append("--attended-smoke")
        argv = [
            *outer,
            "--",
            str(python),
            "-m",
            "gx1.scripts.fixture_executor",
            "--device",
            "cuda",
            "--invocation",
            str(number),
        ]
        invocation = _seal(
            {
                "schema_version": INVOCATION_SCHEMA,
                "decision": "PASS",
                "invocation_number": number,
                "invocation_id": f"invocation-{number:04d}",
                "kind": kind,
                "batch_size": batch,
                "epoch_index": 0,
                "window_index": window,
                "window_count": windows,
                "optimizer_step_budget": budget,
                "expected_success_outcome": expected,
                "source_commit": source_commit,
                "execution_manifest": _binding(manifest_path),
                "launcher_argv": argv,
                "launcher_argv_sha256": canonical_sha256(argv),
                "progress_path": str(runtime / f"progress-{number:04d}.json"),
                "guard_log_path": str(runtime / f"guard-{number:04d}.log"),
                "checkpoint": {
                    "pointer_path": str(pointer),
                    "before_mode": before,
                    "predecessor_invocation_number": predecessor,
                    "write_mode": write_mode,
                },
                "maximum_wall_seconds": 300 if number <= 4 else 7200,
                "requires_fresh_windows_boot": True,
                "signed_guard_only": True,
                "test_data_used": False,
            },
            "invocation_sha256",
        )
        path = tmp_path / "invocations" / f"invocation-{number:04d}.json"
        _write(path, invocation)
        invocations.append(invocation)
        invocation_bindings.append(_binding(path))

    plan = _seal(
        {
            "schema_version": PLAN_SCHEMA,
            "decision": "PASS_PREPARED",
            "campaign_id": "GX1_RANDOM_ACCESS_V2_FIXTURE",
            "created_utc": "2026-09-11T10:00:00+00:00",
            "source_repo": str(repo),
            "source_commit": source_commit,
            "runtime_root": str(runtime),
            "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "prepared_windows_boot": _boot(100, 0),
            "invocations": invocation_bindings,
            "signed_guard_sources": guard_sources,
            "controller_sources": controller_sources,
            "policy": {
                "physical_power_limit_w": 160,
                "maximum_actual_power_draw_w": 170,
                "maximum_core_temperature_c": 65,
                "maximum_memory_junction_temperature_c": 80,
                "maximum_vram_mib": 12288,
                "signed_local_telemetry_seconds": 1,
                "human_status_seconds": 900,
                "fresh_physical_windows_boot_before_every_invocation": True,
                "automatic_power_limit_change": False,
            },
            "authority": {
                "test": False,
                "promotion": False,
                "paper": False,
                "live": False,
                "cloud_spend": False,
            },
            "test_data_used": False,
        },
        "plan_sha256",
    )
    plan_path = tmp_path / "PLAN.json"
    _write(plan_path, plan)
    return plan, plan_path, invocations


def test_explicit_sequence_and_fresh_boot_gate(tmp_path: Path) -> None:
    plan, _, checked = _fixture(tmp_path)
    result = require_plan(plan)
    assert [item["kind"] for item in result["checked_invocations"]] == [
        "smoke_arm",
        "smoke_arm",
        "smoke_arm",
        "resume_proof",
        "epoch1_window",
        "epoch1_window",
        "full_val",
    ]
    assert [item["batch_size"] for item in result["checked_invocations"][:4]] == [4, 8, 16, 16]
    assert next_action(plan, [], current_boot=_boot(100, 0))["decision"] == "REBOOT_REQUIRED"
    assert next_action(plan, [], current_boot=_boot(101, 1))["kind"] == "smoke_arm"
    bad = dict(checked[2])
    bad["batch_size"] = 8
    bad.pop("invocation_sha256")
    bad["invocation_sha256"] = canonical_sha256(bad)
    path = Path(plan["invocations"][2]["path"])
    _write(path, bad)
    broken = dict(plan)
    broken["invocations"] = list(plan["invocations"])
    broken["invocations"][2] = _binding(path)
    broken.pop("plan_sha256")
    broken["plan_sha256"] = canonical_sha256(broken)
    with pytest.raises(RandomAccessCampaignError, match="smoke-arm"):
        require_plan(broken)


def test_bootstrap_active_receipt_archive_and_reboot_receipt(tmp_path: Path) -> None:
    plan, plan_path, invocations = _fixture(tmp_path)
    plan_file_sha = file_sha256(plan_path)
    prepared = prepare_reboot(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        current_boot=_boot(100, 0),
    )
    confirm_reboot(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        request_nonce=prepared["intent"]["request_nonce"],
        shutdown_exit_code=0,
    )
    boot = _boot(101, 1)
    assert inspect_campaign(
        plan_path=plan_path, plan_file_sha256=plan_file_sha, current_boot=boot
    )["action"]["invocation_number"] == 1
    active = begin_invocation(
        plan_path=plan_path, plan_file_sha256=plan_file_sha, current_boot=boot
    )
    assert Path(active["active_marker"]["path"]).is_file()

    invocation = invocations[0]
    pointer = Path(invocation["checkpoint"]["pointer_path"])
    _write(pointer, {"schema_version": "gx1_unified_exit_random_access_fixed_step_pointer_v1"})
    progress = {
        "schema_version": "gx1_local_random_access_progress_v2",
        "plan_sha256": plan["plan_sha256"],
        "invocation_sha256": invocation["invocation_sha256"],
        "terminal": True,
        "outcome": "COMPLETE",
    }
    _write(Path(invocation["progress_path"]), progress)
    _write(
        Path(invocation["guard_log_path"]),
        "2026-09-11T10:01:01Z event=start telemetry_owner=signed_windows_bridge\n"
        "2026-09-11T10:01:02Z event=telemetry power_draw_w=150\n"
        "2026-09-11T10:01:03Z event=exit child_status=0\n",
    )
    receipt = record_invocation(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        trainer_guard_exit_code=0,
        progress_observer_exit_code=0,
        outcome="COMPLETE",
    )
    assert receipt["receipt"]["pointer_before_sha256"] == "GENESIS"
    assert not (Path(plan["runtime_root"]) / "ACTIVE_INVOCATION.json").exists()
    assert (Path(plan["runtime_root"]) / "active-archive/invocation-0001.json").is_file()

    reboot = prepare_reboot(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        current_boot=boot,
    )
    with pytest.raises(RandomAccessCampaignError, match="did not accept"):
        confirm_reboot(
            plan_path=plan_path,
            plan_file_sha256=plan_file_sha,
            request_nonce=reboot["intent"]["request_nonce"],
            shutdown_exit_code=1,
        )


def test_active_without_receipt_blocks_recovery(tmp_path: Path) -> None:
    plan, plan_path, _ = _fixture(tmp_path)
    plan_file_sha = file_sha256(plan_path)
    prepared = prepare_reboot(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        current_boot=_boot(100, 0),
    )
    confirm_reboot(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        request_nonce=prepared["intent"]["request_nonce"],
        shutdown_exit_code=0,
    )
    boot = _boot(101, 1)
    begin_invocation(
        plan_path=plan_path, plan_file_sha256=plan_file_sha, current_boot=boot
    )
    with pytest.raises(RandomAccessCampaignError, match="RECOVERY_RECEIPT_REQUIRED"):
        inspect_campaign(
            plan_path=plan_path,
            plan_file_sha256=plan_file_sha,
            current_boot=boot,
        )
