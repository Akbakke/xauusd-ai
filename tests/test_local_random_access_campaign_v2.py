from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from gx1.contracts.local_random_access_campaign_v2 import (
    BOOT_SCHEMA,
    INVOCATION_SCHEMA,
    PLAN_SCHEMA,
    PROGRESS_SCHEMA,
    RECEIPT_SCHEMA,
    RandomAccessCampaignError,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    fresh_boot,
    next_action,
    require_boot_identity,
    require_plan,
    require_receipt,
)
from gx1.contracts.unified_exit_gpu_batch_selection_v1 import (
    build_arm_receipt,
    build_selection,
    canonical_sha256 as selection_sha256,
)
from gx1.scripts.local_random_access_campaign_v2 import (
    _prepare_private_directory,
    begin_invocation,
    confirm_reboot,
    inspect_campaign,
    prepare_reboot,
    record_invocation,
)
from gx1.scripts.materialize_local_random_access_campaign_v2 import (
    materialize_full_val_campaign,
    materialize_gpu_selection_campaign,
    materialize_selected_training_campaign,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        canonical_bytes(value)
        if isinstance(value, (dict, list))
        else str(value).encode()
    )


def _seal(value: dict, key: str) -> dict:
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result


def _seal_manifest(value: dict) -> dict:
    result = dict(value)
    result["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
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


def test_windows_round_trip_seven_digit_boot_timestamp_is_strict_utc() -> None:
    actual = _seal(
        {
            "schema_version": BOOT_SCHEMA,
            "computer_name": "GX1-3090",
            "last_boot_utc": "2026-09-10T15:58:05.5000000+00:00",
            "boot_id": 731,
        },
        "identity_sha256",
    )
    checked = require_boot_identity(actual)
    assert checked["last_boot_utc"] == "2026-09-10T15:58:05.5000000+00:00"
    assert checked["identity_sha256"] == actual["identity_sha256"]
    prior = _seal(
        {
            **{key: value for key, value in actual.items() if key != "identity_sha256"},
            "last_boot_utc": "2026-09-09T15:58:05.5000000+00:00",
            "boot_id": 730,
        },
        "identity_sha256",
    )
    assert fresh_boot(actual, prior)
    for malformed in (
        "2026-09-10T15:58:05.50000000+00:00",
        "2026-09-10T15:58:05.5000000+01:00",
    ):
        broken = dict(actual)
        broken["last_boot_utc"] = malformed
        broken.pop("identity_sha256")
        broken["identity_sha256"] = canonical_sha256(broken)
        with pytest.raises(RandomAccessCampaignError, match="UTC timestamp invalid"):
            require_boot_identity(broken)


def _binding(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def _source(tmp_path: Path) -> tuple[Path, str, dict, dict]:
    repo = tmp_path / "repo"
    runner = repo / "scripts/gx1_capped_run.sh"
    python = repo / ".venv/bin/python"
    _write(runner, "#!/bin/sh\n")
    _write(python, "#!/bin/sh\n")
    for relative in (
        "scripts/gx1_guarded_trainer_exec.sh",
        "scripts/gx1_host_telemetry_bridge_query.sh",
        "scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1",
        "scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1",
        "gx1/scripts/local_random_access_campaign_v2.py",
    ):
        _write(repo / relative, relative)
    guards = {}
    for name in ("runner", "guard", "query", "certificate"):
        path = runner if name == "runner" else repo / f"evidence/{name}.bin"
        if name != "runner":
            _write(path, name)
        guards[name] = _binding(path)
    controllers = {}
    for name in ("controller", "observer", "campaign_cli"):
        path = repo / f"controller/{name}.txt"
        _write(path, name)
        controllers[name] = _binding(path)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "GX1 Test"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    return repo, commit, guards, controllers


def _invocation(
    tmp_path: Path,
    repo: Path,
    commit: str,
    runtime: Path,
    number: int,
    kind: str,
    batch: int,
    budget: int | None,
    outcome: str,
    before: str,
    predecessor: int | None,
    write_mode: str,
    pointer: Path,
    window: int = 0,
    windows: int = 0,
) -> tuple[dict, dict]:
    manifest = tmp_path / runtime.name / "manifests" / f"{number:04d}.json"
    prelaunch = tmp_path / "PRELAUNCH.json"
    if not prelaunch.exists():
        _write(prelaunch, {"manifest_sha256": "a" * 64})
    session = tmp_path / "TRAIN_SESSION.json"
    if not session.exists():
        _write(session, {"manifest_sha256": "d" * 64})
    outer = [
        str(repo / "scripts/gx1_capped_run.sh"),
        "--class",
        "trainer",
        "--mem",
        "20G",
        "--swap",
        "512M",
    ]
    if kind in {
        "smoke_arm",
        "reference_run",
        "resume_proof_first",
        "resume_proof_second",
    }:
        outer.append("--attended-smoke")
    argv = [
        *outer,
        "--",
        str(repo / ".venv/bin/python"),
        "-m",
        "gx1.scripts.fixture_executor",
        *(["--launch-manifest", str(prelaunch)] if kind != "full_val" else []),
        *(
            ["--train-session-manifest", str(session)]
            if kind not in {"smoke_arm", "full_val"}
            else []
        ),
        "--device",
        "cuda",
        "--invocation",
        str(number),
    ]
    execution = _seal(
        {
            "schema_version": "gx1_local_random_access_execution_manifest_v2",
            "decision": "PASS_EXACT_INVOCATION",
            "invocation_kind": kind,
            "python_module": "gx1.scripts.fixture_executor",
            "source_commit": commit,
            "launcher_argv_sha256": canonical_sha256(argv),
            "prelaunch_manifest": _binding(prelaunch) if kind != "full_val" else None,
            "prelaunch_manifest_sha256": "a" * 64 if kind != "full_val" else None,
            "train_session_manifest": _binding(session)
            if kind not in {"smoke_arm", "full_val"}
            else None,
            "train_session_manifest_sha256": "d" * 64
            if kind not in {"smoke_arm", "full_val"}
            else None,
            "test_data_used": False,
        },
        "artifact_sha256",
    )
    _write(manifest, execution)
    value = _seal(
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
            "expected_success_outcome": outcome,
            "source_commit": commit,
            "execution_manifest": _binding(manifest),
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
            "maximum_wall_seconds": 300
            if kind not in {"epoch1_window", "full_val"}
            else 7200,
            "requires_fresh_windows_boot": True,
            "signed_guard_only": True,
            "test_data_used": False,
        },
        "invocation_sha256",
    )
    path = tmp_path / runtime.name / "invocations" / f"{number:04d}.json"
    _write(path, value)
    return value, _binding(path)


def _base(
    repo: Path, commit: str, runtime: Path, guards: dict, controllers: dict
) -> dict:
    return {
        "schema_version": PLAN_SCHEMA,
        "decision": "PASS_PREPARED",
        "campaign_id": f"GX1_RANDOM_ACCESS_V2_{runtime.name.upper()}",
        "created_utc": "2026-09-11T10:00:00+00:00",
        "source_repo": str(repo),
        "source_commit": commit,
        "runtime_root": str(runtime),
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "prepared_windows_boot": _boot(100, 0),
        "final_train_checkpoint_authority": None,
        "entry_pairs_per_epoch": 16384,
        "transitions_per_epoch": 65536,
        "signed_guard_sources": guards,
        "controller_sources": controllers,
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
    }


def _gpu_plan(tmp_path: Path) -> tuple:
    repo, commit, guards, controllers = _source(tmp_path)
    runtime = tmp_path / "runtime-gpu"
    runtime.mkdir()
    values, bindings = [], []
    for number, batch in enumerate((4, 8, 16), 1):
        value, binding = _invocation(
            tmp_path,
            repo,
            commit,
            runtime,
            number,
            "smoke_arm",
            batch,
            3,
            "COMPLETE",
            "GENESIS",
            None,
            "ATOMIC_UPDATE",
            runtime / f"smoke-b{batch}/RESUME_POINTER.json",
        )
        values.append(value)
        bindings.append(binding)
    plan = _base(repo, commit, runtime, guards, controllers)
    plan.update(
        phase="gpu_selection",
        prior_campaign=None,
        selection_receipt=None,
        selected_batch_size=None,
        invocations=bindings,
    )
    plan = _seal(plan, "plan_sha256")
    path = tmp_path / "GPU_PLAN.json"
    _write(path, plan)
    return plan, path, values, repo, commit, guards, controllers


def _selection(
    tmp_path: Path,
    gpu: dict,
    gpu_path: Path,
    gpu_invocations: list[dict],
) -> tuple[dict, Path]:
    bindings = []
    for number, (invocation, batch, seconds) in enumerate(
        zip(gpu_invocations, (4, 8, 16), (4.0, 3.0, 2.0), strict=True), 1
    ):
        root = tmp_path / "arms" / str(batch)
        pointer = Path(invocation["checkpoint"]["pointer_path"])
        guard = Path(invocation["guard_log_path"])
        progress = Path(invocation["progress_path"])
        _write(pointer, {"batch_size": batch})
        _write(guard, "signed")
        progress_value = _seal(
            {
                "schema_version": PROGRESS_SCHEMA,
                "plan_sha256": gpu["plan_sha256"],
                "invocation_sha256": invocation["invocation_sha256"],
                "phase": "smoke_arm",
                "epoch_index": 0,
                "global_optimizer_steps": 3,
                "next_batch_offset": 3,
                "total_batches": 3,
                "completed_units": 3,
                "total_units": 3,
                "epoch_schedule_sha256": "b" * 64,
                "selection_receipt_sha256": None,
                "checkpoint_pointer": _binding(pointer),
                "terminal": True,
                "outcome": "COMPLETE",
                "observed_utc": f"2026-09-11T10:{number:02d}:03+00:00",
            },
            "progress_sha256",
        )
        _write(progress, progress_value)
        snapshot_root = (
            Path(gpu["runtime_root"])
            / "invocation-evidence"
            / f"invocation-{number:04d}"
        )
        snapshot_root.mkdir(parents=True)
        pointer_snapshot = snapshot_root / "CHECKPOINT_POINTER.json"
        progress_snapshot = snapshot_root / "PROGRESS.json"
        guard_snapshot = snapshot_root / "SIGNED_GUARD.log"
        pointer_snapshot.write_bytes(pointer.read_bytes())
        progress_snapshot.write_bytes(progress.read_bytes())
        guard_snapshot.write_bytes(guard.read_bytes())
        campaign_receipt = _seal(
            {
                "schema_version": RECEIPT_SCHEMA,
                "plan_sha256": gpu["plan_sha256"],
                "invocation_sha256": invocation["invocation_sha256"],
                "invocation_number": number,
                "invocation_id": f"invocation-{number:04d}",
                "kind": "smoke_arm",
                "selection_receipt_sha256": None,
                "boot": _boot(100 + number, number),
                "started_utc": f"2026-09-11T10:{number:02d}:00+00:00",
                "finished_utc": f"2026-09-11T10:{number:02d}:04+00:00",
                "outcome": "COMPLETE",
                "trainer_guard_exit_code": 0,
                "progress_observer_exit_code": 0,
                "pointer_before_sha256": "GENESIS",
                "checkpoint_pointer_after": _binding(pointer),
                "checkpoint_pointer_snapshot": _binding(pointer_snapshot),
                "progress": _binding(progress_snapshot),
                "guard_log": _binding(guard_snapshot),
                "guard_decision": "PASS",
                "signed_guard_telemetry_owner": "gx1_guarded_trainer_exec.sh",
                "active_marker_sha256": "c" * 64,
                "test_data_used": False,
            },
            "receipt_sha256",
        )
        campaign_path = root / "campaign-receipt.json"
        _write(campaign_path, campaign_receipt)
        measurement = {
            "schema_version": "gx1_unified_exit_cuda_smoke_measurement_v1",
            "launch_manifest_sha256": "a" * 64,
            "batch_size": batch,
            "warmup_optimizer_steps": 1,
            "measured_optimizer_steps": 2,
            "measured_entry_rows": 2 * batch,
            "transitions_per_entry": 4,
            "measured_transition_count": 8 * batch,
            "measured_train_seconds": seconds,
            "test_data_used": False,
        }
        measurement["measurement_sha256"] = selection_sha256(measurement)
        measurement_path = root / "measurement.json"
        _write(measurement_path, measurement)
        invocation_path = Path(gpu["invocations"][number - 1]["path"])
        arm = build_arm_receipt(
            campaign_plan_binding=_binding(gpu_path),
            campaign_invocation_binding=_binding(invocation_path),
            campaign_receipt_binding=_binding(campaign_path),
            measurement_binding=_binding(measurement_path),
        )
        arm_path = root / "receipt.json"
        _write(arm_path, arm)
        bindings.append(_binding(arm_path))
    value = build_selection(bindings)
    path = tmp_path / "GPU_SELECTION.json"
    _write(path, value)
    return value, path


def _selected_plan(
    tmp_path: Path,
    gpu: dict,
    gpu_path: Path,
    gpu_invocations: list[dict],
    repo: Path,
    commit: str,
    guards: dict,
    controllers: dict,
) -> tuple:
    selection, selection_path = _selection(tmp_path, gpu, gpu_path, gpu_invocations)
    batch = selection["selected_batch_size"]
    assert batch == 16
    runtime = tmp_path / "runtime-selected"
    runtime.mkdir()
    total, half = -(-16384 // batch), 512
    ref = runtime / "reference/RESUME_POINTER.json"
    split = runtime / "resume-proof/RESUME_POINTER.json"
    epoch = runtime / "epoch1/RESUME_POINTER.json"
    definitions = [
        ("reference_run", 4, "COMPLETE", "GENESIS", None, "ATOMIC_UPDATE", ref, 0, 0),
        (
            "resume_proof_first",
            3,
            "COMPLETE",
            "GENESIS",
            None,
            "ATOMIC_UPDATE",
            split,
            0,
            0,
        ),
        (
            "resume_proof_second",
            1,
            "COMPLETE",
            "PREVIOUS_RECEIPT_AFTER",
            2,
            "ATOMIC_UPDATE",
            split,
            0,
            0,
        ),
        (
            "epoch1_window",
            half,
            "RESUMABLE",
            "GENESIS",
            None,
            "ATOMIC_UPDATE",
            epoch,
            0,
            2,
        ),
        (
            "epoch1_window",
            total - half,
            "COMPLETE",
            "PREVIOUS_RECEIPT_AFTER",
            4,
            "ATOMIC_UPDATE",
            epoch,
            1,
            2,
        ),
    ]
    values, bindings = [], []
    for number, args in enumerate(definitions, 1):
        kind, budget, outcome, before, predecessor, write, pointer, window, windows = (
            args
        )
        value, binding = _invocation(
            tmp_path,
            repo,
            commit,
            runtime,
            number,
            kind,
            batch,
            budget,
            outcome,
            before,
            predecessor,
            write,
            pointer,
            window,
            windows,
        )
        values.append(value)
        bindings.append(binding)
    plan = _base(repo, commit, runtime, guards, controllers)
    plan.update(
        phase="selected_training",
        prior_campaign=_binding(gpu_path),
        selection_receipt=_binding(selection_path),
        selected_batch_size=batch,
        invocations=bindings,
    )
    plan = _seal(plan, "plan_sha256")
    path = tmp_path / "SELECTED_PLAN.json"
    _write(path, plan)
    return plan, path, values


def test_two_phase_sequence_and_selected_epoch_budget(tmp_path: Path) -> None:
    gpu, gpu_path, gpu_invocations, repo, commit, guards, controllers = _gpu_plan(
        tmp_path
    )
    checked_gpu = require_plan(gpu)
    assert [x["batch_size"] for x in checked_gpu["checked_invocations"]] == [4, 8, 16]
    assert (
        next_action(gpu, [], current_boot=_boot(100, 0))["decision"]
        == "REBOOT_REQUIRED"
    )
    assert next_action(gpu, [], current_boot=_boot(101, 1))["kind"] == "smoke_arm"
    inspected = inspect_campaign(
        plan_path=gpu_path,
        plan_file_sha256=file_sha256(gpu_path),
        current_boot=_boot(101, 1),
    )
    assert inspected["action"]["decision"] == "LAUNCH"
    assert inspected["signed_guard_sources"] == checked_gpu["signed_guard_sources"]
    assert inspected["gpu_uuid"] == checked_gpu["gpu_uuid"]
    selected, _, invocations = _selected_plan(
        tmp_path, gpu, gpu_path, gpu_invocations, repo, commit, guards, controllers
    )
    checked = require_plan(selected)
    assert [x["kind"] for x in checked["checked_invocations"]] == [
        "reference_run",
        "resume_proof_first",
        "resume_proof_second",
        "epoch1_window",
        "epoch1_window",
    ]
    assert sum(x["optimizer_step_budget"] for x in invocations[3:5]) == 1024
    bad = dict(gpu_invocations[2])
    bad["batch_size"] = 8
    bad.pop("invocation_sha256")
    bad["invocation_sha256"] = canonical_sha256(bad)
    path = Path(gpu["invocations"][2]["path"])
    _write(path, bad)
    broken = dict(gpu)
    broken["invocations"] = list(gpu["invocations"])
    broken["invocations"][2] = _binding(path)
    broken.pop("plan_sha256")
    broken["plan_sha256"] = canonical_sha256(broken)
    with pytest.raises(RandomAccessCampaignError, match="smoke-arm"):
        require_plan(broken)


def test_atomic_receipt_archive_and_reboot_receipt(tmp_path: Path) -> None:
    plan, plan_path, invocations, *_ = _gpu_plan(tmp_path)
    plan_file_sha = file_sha256(plan_path)
    prepared = prepare_reboot(
        plan_path=plan_path, plan_file_sha256=plan_file_sha, current_boot=_boot(100, 0)
    )
    confirm_reboot(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        request_nonce=prepared["intent"]["request_nonce"],
        shutdown_exit_code=0,
    )
    boot = _boot(101, 1)
    assert (
        inspect_campaign(
            plan_path=plan_path, plan_file_sha256=plan_file_sha, current_boot=boot
        )["action"]["invocation_number"]
        == 1
    )
    active = begin_invocation(
        plan_path=plan_path, plan_file_sha256=plan_file_sha, current_boot=boot
    )
    assert Path(active["active_marker"]["path"]).is_file()
    invocation = invocations[0]
    assert not Path(invocation["checkpoint"]["pointer_path"]).parent.exists()
    for output_parent in (
        Path(invocation["checkpoint"]["pointer_path"]).parent.parent,
        Path(invocation["guard_log_path"]).parent,
    ):
        assert output_parent.stat().st_mode & 0o777 == 0o700
    assert not Path(invocation["guard_log_path"]).exists()
    pointer = Path(invocation["checkpoint"]["pointer_path"])
    _write(
        pointer,
        {"schema_version": "gx1_unified_exit_random_access_fixed_step_pointer_v1"},
    )
    from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import _write_campaign_progress

    progress = _write_campaign_progress(
        Path(invocation["progress_path"]),
        {
            "schema_version": PROGRESS_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "invocation_sha256": invocation["invocation_sha256"],
            "phase": invocation["kind"],
            "epoch_index": 0,
            "global_optimizer_steps": 3,
            "next_batch_offset": 3,
            "total_batches": 4096,
            "completed_units": 3,
            "total_units": 4096,
            "epoch_schedule_sha256": "b" * 64,
            "selection_receipt_sha256": None,
            "checkpoint_pointer": _binding(pointer),
            "terminal": True,
            "outcome": "COMPLETE",
            "observed_utc": "2026-09-11T10:01:03+00:00",
        },
    )
    _write(
        Path(invocation["guard_log_path"]),
        "event=start telemetry_owner=signed_windows_bridge\n"
        "event=telemetry power_draw_w=150\n"
        "event=exit child_status=0\n",
    )
    recorded = record_invocation(
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        trainer_guard_exit_code=0,
        progress_observer_exit_code=0,
        outcome="COMPLETE",
    )
    assert recorded["receipt"]["pointer_before_sha256"] == "GENESIS"
    assert recorded["receipt"]["checkpoint_pointer_after"]["path"] == str(pointer)
    assert (
        Path(recorded["receipt"]["checkpoint_pointer_snapshot"]["path"]).read_bytes()
        == pointer.read_bytes()
    )
    assert Path(recorded["receipt"]["progress"]["path"]).name == "PROGRESS.json"
    assert recorded["receipt"]["guard_decision"] == "PASS"
    original_snapshot = Path(
        recorded["receipt"]["checkpoint_pointer_snapshot"]["path"]
    ).read_bytes()
    _write(pointer, {"schema_version": "later-continuation"})
    _write(Path(invocation["progress_path"]), {"later": True})
    checked_invocation = require_plan(plan)["checked_invocations"][0]
    require_receipt(
        recorded["receipt"],
        plan_sha256=plan["plan_sha256"],
        invocation=checked_invocation,
        verify_files=True,
    )
    assert (
        Path(recorded["receipt"]["checkpoint_pointer_snapshot"]["path"]).read_bytes()
        == original_snapshot
    )
    assert not (Path(plan["runtime_root"]) / "ACTIVE_INVOCATION.json").exists()
    assert (
        Path(plan["runtime_root"]) / "active-archive/invocation-0001.json"
    ).is_file()
    reboot = prepare_reboot(
        plan_path=plan_path, plan_file_sha256=plan_file_sha, current_boot=boot
    )
    with pytest.raises(RandomAccessCampaignError, match="did not accept"):
        confirm_reboot(
            plan_path=plan_path,
            plan_file_sha256=plan_file_sha,
            request_nonce=reboot["intent"]["request_nonce"],
            shutdown_exit_code=1,
        )


def test_active_without_receipt_blocks_recovery(tmp_path: Path) -> None:
    plan, plan_path, *_ = _gpu_plan(tmp_path)
    sha = file_sha256(plan_path)
    prepared = prepare_reboot(
        plan_path=plan_path, plan_file_sha256=sha, current_boot=_boot(100, 0)
    )
    confirm_reboot(
        plan_path=plan_path,
        plan_file_sha256=sha,
        request_nonce=prepared["intent"]["request_nonce"],
        shutdown_exit_code=0,
    )
    boot = _boot(101, 1)
    begin_invocation(plan_path=plan_path, plan_file_sha256=sha, current_boot=boot)
    with pytest.raises(RandomAccessCampaignError, match="RECOVERY_RECEIPT_REQUIRED"):
        inspect_campaign(plan_path=plan_path, plan_file_sha256=sha, current_boot=boot)


def test_production_materializer_builds_acyclic_phase_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gx1.contracts.unified_exit_train_session_manifest_v1 import build_train_session_manifest
    from tests.test_unified_exit_fixed_step_resume_equivalence_v1 import _build as build_real_equivalence
    repo, commit, guards, _ = _source(tmp_path)
    boot_path = tmp_path / "BOOT.json"
    _write(boot_path, _boot(100, 0))
    launch = _seal_manifest(
        {
            "schema_version": "gx1_unified_exit_random_access_cuda_smoke_launch_v2",
            "decision": "PASS_GPU_SMOKE_MATRIX_ELIGIBLE",
            "source_repo": str(repo),
            "source_commit": commit,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "test_data_used": False,
        }
    )
    launch_path = tmp_path / "PRELAUNCH.json"
    _write(launch_path, launch)
    phase1 = materialize_gpu_selection_campaign(
        repo=repo,
        output=tmp_path / "phase1",
        runtime=tmp_path / "runtime-phase1",
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        prepared_boot_path=boot_path,
        prepared_boot_file_sha256=file_sha256(boot_path),
        prelaunch_path=launch_path,
        prelaunch_file_sha256=file_sha256(launch_path),
        certificate_path=Path(guards["certificate"]["path"]),
    )
    checked_phase1 = phase1["plan"]
    assert [item["batch_size"] for item in checked_phase1["checked_invocations"]] == [
        4,
        8,
        16,
    ]
    assert all(
        item["launcher_argv"][:7]
        == [
            str(repo / "scripts/gx1_capped_run.sh"),
            "--class",
            "trainer",
            "--mem",
            "20G",
            "--swap",
            "512M",
        ]
        for item in checked_phase1["checked_invocations"]
    )

    begun = begin_invocation(
        plan_path=Path(phase1["path"]), plan_file_sha256=phase1["sha256"],
        current_boot=_boot(101, 1),
    )
    checkpoint_dir = Path(begun["invocation"]["checkpoint"]["pointer_path"]).parent
    assert not checkpoint_dir.exists()
    assert checkpoint_dir.parent.stat().st_mode & 0o777 == 0o700
    assert not Path(begun["invocation"]["progress_path"]).parent.exists()

    selection = {
        "artifact_sha256": "6" * 64,
        "source_commit": commit,
        "launch_manifest_sha256": launch["manifest_sha256"],
        "selected_batch_size": 16,
        "entry_pairs_per_epoch": 16384,
        "transition_budget_per_epoch": 65536,
        "total_batches_per_epoch": 1024,
        "checkpoint_interval_optimizer_steps": 64,
        "test_data_used": False,
    }
    selection_path = tmp_path / "SELECTION.json"
    _write(selection_path, selection)
    monkeypatch.setattr(
        "gx1.scripts.materialize_local_random_access_campaign_v2.require_selection",
        lambda value, verify_files=True: dict(value),
    )
    monkeypatch.setattr(
        "gx1.contracts.unified_exit_gpu_batch_selection_v1.require_selection",
        lambda value, verify_files=True: dict(value),
    )
    monkeypatch.setattr(
        "gx1.contracts.unified_exit_train_session_manifest_v1.require_selection",
        lambda value, verify_files=True: dict(value),
    )
    resume_path = tmp_path / "RESUME_SESSION.json"
    epoch_path = tmp_path / "EPOCH_SESSION.json"
    session_args = dict(source_commit=commit, prelaunch_binding=_binding(launch_path),
                        prelaunch_manifest_sha256=launch["manifest_sha256"],
                        gpu_selection_binding=_binding(selection_path), gpu_selection=selection)
    _write(resume_path, build_train_session_manifest(phase="resume_proof", **session_args))
    proof = materialize_selected_training_campaign(
        repo=repo, output=tmp_path / "proof", runtime=tmp_path / "runtime-proof",
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        prepared_boot_path=boot_path, prepared_boot_file_sha256=file_sha256(boot_path),
        prelaunch_path=launch_path, prelaunch_file_sha256=file_sha256(launch_path),
        certificate_path=Path(guards["certificate"]["path"]),
        prior_campaign_path=Path(phase1["path"]), prior_campaign_file_sha256=phase1["sha256"],
        selection_path=selection_path, selection_file_sha256=file_sha256(selection_path),
        resume_session_path=resume_path, resume_session_file_sha256=file_sha256(resume_path),
        epoch_session_path=None, epoch_session_file_sha256=None, epoch_window_steps=64,
        resume_proof_only=True,
    )
    assert proof["plan"]["phase"] == "resume_proof"
    assert [i["optimizer_step_budget"] for i in proof["plan"]["checked_invocations"]] == [4, 3, 1]
    assert not epoch_path.exists()
    with pytest.raises(RuntimeError, match="EQ_REQUIRED"):
        build_train_session_manifest(phase="epoch1", **session_args)
    equivalence = build_real_equivalence(tmp_path)
    eq_path = tmp_path / "EQUIVALENCE.json"
    _write(eq_path, equivalence)
    _write(epoch_path, build_train_session_manifest(
        phase="epoch1", resume_equivalence_binding=_binding(eq_path),
        resume_equivalence=equivalence, **session_args))
    phase2 = materialize_selected_training_campaign(
        repo=repo,
        output=tmp_path / "phase2",
        runtime=tmp_path / "runtime-phase2",
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        prepared_boot_path=boot_path,
        prepared_boot_file_sha256=file_sha256(boot_path),
        prelaunch_path=launch_path,
        prelaunch_file_sha256=file_sha256(launch_path),
        certificate_path=Path(guards["certificate"]["path"]),
        prior_campaign_path=Path(proof["path"]),
        prior_campaign_file_sha256=proof["sha256"],
        selection_path=selection_path,
        selection_file_sha256=file_sha256(selection_path),
        resume_session_path=resume_path,
        resume_session_file_sha256=file_sha256(resume_path),
        epoch_session_path=epoch_path,
        epoch_session_file_sha256=file_sha256(epoch_path),
        epoch_window_steps=64,
    )
    invocations = phase2["plan"]["checked_invocations"]
    assert len(invocations) == 16
    assert all(item["kind"] == "epoch1_window" for item in invocations)
    assert invocations[0]["checkpoint"]["before_mode"] == "GENESIS"
    assert invocations[1]["checkpoint"]["predecessor_invocation_number"] == 1
    assert invocations[-1]["expected_success_outcome"] == "COMPLETE"
    assert sum(item["optimizer_step_budget"] for item in invocations) == 1024

    pointer_path = tmp_path / "epoch1" / "RESUME_POINTER.json"
    _write(pointer_path, {"pointer_sha256": "f" * 64})
    authority = {
        "schema_version": "gx1_unified_exit_final_train_checkpoint_authority_v1",
        "entry_pair_count": 16384,
        "transition_count": 65536,
        "source_commit": commit,
        "campaign_plan": {"path": phase2["path"], "sha256": phase2["sha256"]},
        "gpu_batch_selection": _binding(selection_path),
        "gpu_batch_selection_artifact_sha256": selection["artifact_sha256"],
        "launch_manifest": _binding(launch_path),
        "final_checkpoint_pointer": _binding(pointer_path),
        "selected_batch_size": 16,
    }
    authority_path = tmp_path / "FINAL_AUTHORITY.json"
    _write(authority_path, authority)
    monkeypatch.setattr(
        "gx1.contracts.unified_exit_final_train_checkpoint_authority_v1.require_final_train_checkpoint_authority",
        lambda value, verify_files=True: dict(value),
    )
    control_repo, _, _, _ = _source(tmp_path / "control")
    phase3 = materialize_full_val_campaign(
        repo=repo,
        controller_repo=control_repo,
        output=tmp_path / "phase3",
        runtime=tmp_path / "runtime-phase3",
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
        prepared_boot_path=boot_path,
        prepared_boot_file_sha256=file_sha256(boot_path),
        certificate_path=Path(guards["certificate"]["path"]),
        prior_campaign_path=Path(phase2["path"]),
        prior_campaign_file_sha256=phase2["sha256"],
        selection_path=selection_path,
        selection_file_sha256=file_sha256(selection_path),
        final_authority_path=authority_path,
        final_authority_file_sha256=file_sha256(authority_path),
        window_count=3,
        max_forwards_per_window=128,
        progress_interval_forwards=16,
        max_model_forwards=4096,
        max_materialized_state_views=65536,
        max_wall_seconds=3600,
    )
    val_invocations = phase3["plan"]["checked_invocations"]
    result_parent = tmp_path / "runtime-phase3" / "rollout"
    assert result_parent.is_dir() and not result_parent.is_symlink()
    assert result_parent.stat().st_mode & 0o777 == 0o700
    assert list(result_parent.iterdir()) == []
    assert phase3["plan"]["phase"] == "full_val"
    assert phase3["plan"]["source_repo"] == str(repo)
    assert phase3["plan"]["source_commit"] == commit
    assert phase3["plan"]["controller_sources"]["campaign_cli"]["path"] == str(
        control_repo / "gx1/scripts/local_random_access_campaign_v2.py"
    )
    assert all(
        item["launcher_argv"][0] == str(repo / "scripts/gx1_capped_run.sh")
        for item in val_invocations
    )
    assert [item["kind"] for item in val_invocations] == [
        "full_val_window",
        "full_val_window",
        "full_val_window",
    ]
    assert all(
        item["expected_success_outcome"] == "RESUMABLE_OR_COMPLETE"
        for item in val_invocations
    )
    assert val_invocations[0]["checkpoint"]["before_mode"] == "FINAL_AUTHORITY"
    assert val_invocations[1]["checkpoint"]["before_mode"] == "PREVIOUS_RECEIPT_AFTER"
    assert len({item["progress_path"] for item in val_invocations}) == 3
    assert len({item["rollout_cursor_path"] for item in val_invocations}) == 1

    phase3_path = Path(phase3["path"])
    boot1 = _boot(101, 1)
    import gx1.scripts.local_random_access_campaign_v2 as campaign_cli

    load_calls = []
    original_load = campaign_cli._load_plan

    def counted_load(*args):
        load_calls.append(args)
        return original_load(*args)

    with monkeypatch.context() as scoped:
        scoped.setattr(campaign_cli, "_load_plan", counted_load)
        begun = begin_invocation(
            plan_path=phase3_path,
            plan_file_sha256=phase3["sha256"],
            current_boot=boot1,
        )
    assert len(load_calls) == 1
    assert begun["invocation"]["window_index"] == 0
    cursor_path = Path(val_invocations[0]["rollout_cursor_path"])
    _write(cursor_path, {"next_state_index": 7})
    progress_path = Path(val_invocations[0]["progress_path"])
    progress = _seal(
        {
            "schema_version": PROGRESS_SCHEMA,
            "plan_sha256": phase3["plan"]["plan_sha256"],
            "invocation_sha256": val_invocations[0]["invocation_sha256"],
            "phase": "full_val_window",
            "epoch_index": 0,
            "global_optimizer_steps": 1024,
            "next_batch_offset": 1024,
            "total_batches": 1024,
            "completed_units": 100,
            "total_units": 5508,
            "epoch_schedule_sha256": "1" * 64,
            "selection_receipt_sha256": selection["artifact_sha256"],
            "checkpoint_pointer": _binding(pointer_path),
            "rollout_cursor": _binding(cursor_path),
            "terminal": True,
            "outcome": "RESUMABLE",
            "observed_utc": "2026-09-11T13:00:00+00:00",
        },
        "progress_sha256",
    )
    _write(progress_path, progress)
    _write(
        Path(val_invocations[0]["guard_log_path"]),
        "telemetry_owner=signed_windows_bridge event=telemetry\n"
        "event=exit child_status=0\n",
    )
    with pytest.raises(RandomAccessCampaignError, match="automatic success outcome"):
        record_invocation(
            plan_path=phase3_path,
            plan_file_sha256=phase3["sha256"],
            trainer_guard_exit_code=7,
            progress_observer_exit_code=0,
            outcome="AUTO",
        )
    assert not (Path(phase3["plan"]["runtime_root"]) / "receipts").exists()
    first_receipt = record_invocation(
        plan_path=phase3_path,
        plan_file_sha256=phase3["sha256"],
        trainer_guard_exit_code=0,
        progress_observer_exit_code=0,
        outcome="AUTO",
    )["receipt"]
    snapshot_path = Path(first_receipt["rollout_cursor_snapshot"]["path"])
    assert snapshot_path.read_bytes() == cursor_path.read_bytes()
    reboot = prepare_reboot(
        plan_path=phase3_path,
        plan_file_sha256=phase3["sha256"],
        current_boot=boot1,
    )
    confirm_reboot(
        plan_path=phase3_path,
        plan_file_sha256=phase3["sha256"],
        request_nonce=reboot["intent"]["request_nonce"],
        shutdown_exit_code=0,
    )
    boot2 = _boot(102, 2)
    original_cursor = snapshot_path.read_bytes()
    _write(cursor_path, {"next_state_index": 999})
    with pytest.raises(RandomAccessCampaignError, match="rollout cursor differs"):
        begin_invocation(
            plan_path=phase3_path,
            plan_file_sha256=phase3["sha256"],
            current_boot=boot2,
        )
    cursor_path.write_bytes(original_cursor)
    second = begin_invocation(
        plan_path=phase3_path,
        plan_file_sha256=phase3["sha256"],
        current_boot=boot2,
    )
    assert second["invocation"]["window_index"] == 1
    _write(cursor_path, {"next_state_index": 8, "complete": True})
    progress_path = Path(val_invocations[1]["progress_path"])
    progress["invocation_sha256"] = val_invocations[1]["invocation_sha256"]
    progress["completed_units"] = 5508
    progress["rollout_cursor"] = _binding(cursor_path)
    progress["outcome"] = "COMPLETE"
    progress.pop("progress_sha256")
    progress["progress_sha256"] = canonical_sha256(progress)
    _write(progress_path, progress)
    _write(
        Path(val_invocations[1]["guard_log_path"]),
        "telemetry_owner=signed_windows_bridge event=telemetry\n"
        "event=exit child_status=0\n",
    )
    record_invocation(
        plan_path=phase3_path,
        plan_file_sha256=phase3["sha256"],
        trainer_guard_exit_code=0,
        progress_observer_exit_code=0,
        outcome="AUTO",
    )
    final = inspect_campaign(
        plan_path=phase3_path,
        plan_file_sha256=phase3["sha256"],
        current_boot=boot2,
    )
    assert final["action"]["decision"] == "COMPLETE"


def test_private_invocation_parents_are_created_without_output_files(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "new" / "checkpoint"
    _prepare_private_directory(parent, label="checkpoint")
    assert parent.is_dir()
    assert parent.stat().st_mode & 0o777 == 0o700
    assert list(parent.iterdir()) == []

    linked = tmp_path / "linked"
    linked.symlink_to(parent, target_is_directory=True)
    with pytest.raises(RandomAccessCampaignError, match="parent chain invalid"):
        _prepare_private_directory(linked / "child", label="guard")


def test_running_control_cli_rejects_wrong_source_or_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gx1.scripts.local_random_access_campaign_v2 as cli

    source = tmp_path / "control.py"
    source.write_text("control")
    plan_path = tmp_path / "plan.json"
    _write(plan_path, {"controller_sources": {"campaign_cli": _binding(source)}})
    monkeypatch.setattr(cli, "__file__", str(source))
    cli._require_running_cli(plan_path, file_sha256(plan_path))
    source.write_text("different")
    with pytest.raises(RandomAccessCampaignError, match="running campaign CLI"):
        cli._require_running_cli(plan_path, file_sha256(plan_path))
    source.write_text("control")
    other = tmp_path / "other.py"
    other.write_text("control")
    monkeypatch.setattr(cli, "__file__", str(other))
    with pytest.raises(RandomAccessCampaignError, match="running campaign CLI"):
        cli._require_running_cli(plan_path, file_sha256(plan_path))
