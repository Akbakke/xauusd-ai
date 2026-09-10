from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from gx1.contracts.local_lifecycle_v2_pilot_campaign_v1 import (
    GATE_SCHEMA,
    INVOCATION_SCHEMA,
    RECEIPT_SCHEMA,
    SCHEMA,
    PilotCampaignError,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    next_action,
    require_clean_source,
    require_plan,
    require_receipt,
)
from gx1.contracts.local_lifecycle_v2_pilot_telemetry_v1 import (
    build_status,
    build_telemetry_sample,
)
from gx1.scripts.local_lifecycle_v2_pilot_campaign_v1 import inspect_campaign


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _seal(value: dict, key: str) -> dict:
    result = dict(value)
    result[key] = canonical_sha256(result)
    return result


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    repo, runtime, artifacts = tmp_path / "repo", tmp_path / "runtime", tmp_path / "artifacts"
    for path in (repo / "scripts", repo / ".venv/bin", runtime, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    (repo / "scripts/gx1_capped_run.sh").write_text("# canonical fixture\n")
    (repo / ".venv/bin/python").write_text("# python fixture\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "GX1 Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "source"], check=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    campaign_id = "GX1_LIFECYCLE_V2_LOCAL_PILOT_20260910"
    gate = _seal({
        "schema_version": GATE_SCHEMA, "decision": "PASS", "campaign_id": campaign_id,
        "source_commit": commit,
        "checks": {name: True for name in (
            "clean_source_commit", "train_val_only", "test_sealed",
            "lifecycle_v2_admission", "economics_provider_chain", "checkpoint_resume",
            "canonical_capped_entrypoint", "signed_telemetry", "physical_160w_observed",
        )},
        "cuda_execution_authorized": True, "test_data_used": False,
    }, "gate_sha256")
    gate_path = artifacts / "gate.json"
    _write(gate_path, gate)
    pointer = runtime / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
    pointer.write_text('{"fixture":"pointer"}\n')
    invocations = {}
    for stage in ("smoke", "epoch1"):
        before = [
            str(repo / "scripts/gx1_capped_run.sh"), "--class", "trainer",
            "--mem", "20G", "--swap", "512M",
        ]
        if stage == "smoke":
            before.append("--attended-smoke")
            target = [str(repo / ".venv/bin/python"), "-m", "gx1.scripts.attended_model_native_hardware_smoke_v1", "--attended-hardware-smoke", "--device", "cuda"]
        else:
            target = [str(repo / ".venv/bin/python"), "-m", "gx1.models.entry_v10.entry_v10_ctx_train_v3", "--train", "--device", "cuda"]
        invocation = _seal({
            "schema_version": INVOCATION_SCHEMA, "decision": "PASS", "stage": stage,
            "campaign_id": campaign_id, "full_launch_gate_sha256": gate["gate_sha256"],
            "source_repo": str(repo), "source_commit": commit,
            "launcher_command": [*before, "--", *target],
            "progress_json": str(runtime / "progress.json"), "pointer_path": str(pointer),
            "maximum_wall_seconds": 300 if stage == "smoke" else 7200,
            "resume_allowed": stage == "epoch1", "test_data_used": False,
        }, "invocation_sha256")
        path = artifacts / f"{stage}.json"
        _write(path, invocation)
        invocations[stage] = {"path": str(path), "sha256": file_sha256(path)}
    plan = _seal({
        "schema_version": SCHEMA, "decision": "PREPARED_GATE_REQUIRED",
        "campaign_id": campaign_id, "created_utc": "2026-09-10T20:00:00+00:00",
        "source_repo": str(repo), "source_commit": commit,
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "runtime_root": str(runtime),
        "launch_gate": {"path": str(gate_path), "sha256": file_sha256(gate_path)},
        "invocations": invocations,
        "policy": {
            "physical_power_limit_w": 160, "maximum_power_draw_w": 160,
            "automatic_power_limit_change": False, "local_telemetry_sample_seconds": 1,
            "human_status_cadence_seconds": 900, "meaningful_event_immediate": True,
            "smoke_must_complete_first": True, "reboot_between_smoke_and_epoch1": True,
            "reboot_before_each_additional_heavy_invocation": True,
        },
        "authority": {
            "cuda_only_after_full_gate_pass": True, "test": False, "promotion": False,
            "paper": False, "live": False, "cloud_spend": False,
        },
    }, "plan_sha256")
    plan_path = tmp_path / "plan.json"
    _write(plan_path, plan)
    return plan, plan_path


def _receipt(tmp_path: Path, plan: dict, number: int, stage: str, boot: str, outcome: str) -> dict:
    telemetry = tmp_path / f"telemetry-{number}.jsonl"
    telemetry.write_text('{"safe":true}\n')
    return _seal({
        "schema_version": RECEIPT_SCHEMA, "plan_sha256": plan["plan_sha256"],
        "invocation_number": number, "stage": stage, "boot_utc": boot,
        "outcome": outcome, "pointer_before_sha256": "1" * 64,
        "pointer_after_sha256": "2" * 64, "telemetry_jsonl": str(telemetry),
        "telemetry_sha256": file_sha256(telemetry), "telemetry_sample_count": 1,
        "maximum_observed_power_limit_w": 160.0,
        "maximum_observed_power_draw_w": 159.9, "test_data_used": False,
    }, "receipt_sha256")


def test_full_gate_and_exact_clean_source_are_required(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    assert require_plan(plan)["plan_sha256"] == plan["plan_sha256"]
    require_clean_source(plan)
    gate_path = Path(plan["launch_gate"]["path"])
    gate = json.loads(gate_path.read_text())
    gate["checks"]["economics_provider_chain"] = False
    gate.pop("gate_sha256")
    _write(gate_path, _seal(gate, "gate_sha256"))
    plan.pop("plan_sha256")
    plan["launch_gate"]["sha256"] = file_sha256(gate_path)
    plan = _seal(plan, "plan_sha256")
    with pytest.raises(PilotCampaignError, match="not PASS"):
        require_plan(plan)


def test_smoke_then_new_boot_then_epoch1_and_resume(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    assert next_action(plan, [], current_boot_utc="2026-09-10T20:05:00+00:00")["decision"] == "LAUNCH_SMOKE"
    smoke = _receipt(tmp_path, plan, 1, "smoke", "2026-09-10T20:00:00+00:00", "COMPLETE")
    assert next_action(plan, [smoke], current_boot_utc="2026-09-10T20:00:00+00:00")["decision"] == "REBOOT_REQUIRED"
    assert next_action(plan, [smoke], current_boot_utc="2026-09-10T20:10:00+00:00")["decision"] == "LAUNCH_EPOCH1"
    epoch = _receipt(tmp_path, plan, 2, "epoch1", "2026-09-10T20:10:00+00:00", "RESUMABLE")
    assert next_action(plan, [smoke, epoch], current_boot_utc="2026-09-10T20:10:00+00:00")["decision"] == "REBOOT_REQUIRED"
    assert next_action(plan, [smoke, epoch], current_boot_utc="2026-09-10T20:20:00+00:00")["decision"] == "RESUME_EPOCH1"



def test_crash_marker_blocks_automatic_duplicate_launch(tmp_path: Path) -> None:
    plan, plan_path = _fixture(tmp_path)
    first = inspect_campaign(
        plan_path=plan_path,
        plan_sha256=file_sha256(plan_path),
        current_windows_boot_utc="2026-09-10T20:05:00+00:00",
    )
    assert first["action"]["decision"] == "LAUNCH_SMOKE"
    marker = Path(plan["runtime_root"]) / "ACTIVE_INVOCATION.json"
    marker.write_text('{"stage":"smoke","invocation_number":1}\n')
    blocked = inspect_campaign(
        plan_path=plan_path,
        plan_sha256=file_sha256(plan_path),
        current_windows_boot_utc="2026-09-10T20:06:00+00:00",
    )
    assert blocked["action"]["decision"] == "BLOCKED_RECOVERY_RECEIPT_REQUIRED"

def test_receipt_fails_above_160w(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    receipt = _receipt(tmp_path, plan, 1, "smoke", "2026-09-10T20:00:00+00:00", "COMPLETE")
    receipt.pop("receipt_sha256")
    receipt["maximum_observed_power_draw_w"] = 160.01
    receipt = _seal(receipt, "receipt_sha256")
    with pytest.raises(PilotCampaignError, match="exceeds 160"):
        require_receipt(receipt, plan_sha256=plan["plan_sha256"])


def test_invocation_cannot_add_test_or_skip_capped_runner(tmp_path: Path) -> None:
    plan, _ = _fixture(tmp_path)
    invocation_path = Path(plan["invocations"]["epoch1"]["path"])
    value = json.loads(invocation_path.read_text())
    value.pop("invocation_sha256")
    value["launcher_command"].append("--test")
    _write(invocation_path, _seal(value, "invocation_sha256"))
    plan.pop("plan_sha256")
    plan["invocations"]["epoch1"]["sha256"] = file_sha256(invocation_path)
    plan = _seal(plan, "plan_sha256")
    with pytest.raises(PilotCampaignError, match="TEST"):
        require_plan(plan)


def test_machine_eta_and_human_cadence_are_separate() -> None:
    first = build_telemetry_sample(
        observed_utc="2026-09-10T20:00:00+00:00", stage="epoch1", invocation_number=2,
        gpu_uuid="GPU-x", power_limit_w=160, power_draw_w=150, core_temp_c=50,
        memory_temp_c=55, memory_used_mib=9000, utilization_percent=95,
        progress={"schema_version": "gx1_local_lifecycle_v2_progress_v1", "phase": "train", "completed_units": 10, "total_units": 100, "observed_utc": "2026-09-10T20:00:00+00:00"},
        previous_sample=None,
    )
    second = build_telemetry_sample(
        observed_utc="2026-09-10T20:00:01+00:00", stage="epoch1", invocation_number=2,
        gpu_uuid="GPU-x", power_limit_w=160, power_draw_w=151, core_temp_c=50,
        memory_temp_c=55, memory_used_mib=9000, utilization_percent=95,
        progress={"schema_version": "gx1_local_lifecycle_v2_progress_v1", "phase": "train", "completed_units": 11, "total_units": 100, "observed_utc": "2026-09-10T20:00:01+00:00"},
        previous_sample=first,
    )
    status1 = build_status(first, previous_status=None)
    status2 = build_status(second, previous_status=status1)
    assert second["eta_seconds"] == pytest.approx(89.0)
    assert status1["human_status_due"] is True
    assert status2["human_status_due"] is False
    assert status2["local_sample_seconds"] == 1
    assert status2["human_status_cadence_seconds"] == 900
    assert status2["codex_polling_required"] is False


def test_local_sample_kills_on_limit_or_draw_breach() -> None:
    sample = build_telemetry_sample(
        observed_utc="2026-09-10T20:00:00+00:00", stage="smoke", invocation_number=1,
        gpu_uuid="GPU-x", power_limit_w=160, power_draw_w=160.1, core_temp_c=40,
        memory_temp_c=45, memory_used_mib=1000, utilization_percent=80,
        progress=None, previous_sample=None,
    )
    assert sample["safety_decision"] == "KILL_AND_BLOCK"
