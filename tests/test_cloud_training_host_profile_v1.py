from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.cloud_training_host_profile_v1 import (
    ADMISSION_MAX_PROJECTED_SECONDS,
    HARD_DEADLINE_SECONDS,
    CloudTrainingHostProfileError,
    require_cloud_training_host_profile,
    require_cloud_training_host_profile_file,
    runner_fields,
)


def _profile(tmp_path: Path) -> dict[str, object]:
    proof = tmp_path / "provider-proof.json"
    proof.write_text("{}\n", encoding="utf-8")
    certificate = tmp_path / "public.pem"
    certificate.write_text("certificate\n", encoding="utf-8")
    service = tmp_path / "service.py"
    service.write_text("pass\n", encoding="utf-8")
    return {
        "schema_version": "gx1_cloud_training_host_profile_v1",
        "decision": "PASS_HOST_QUALIFIED_NOT_TRAINING_AUTHORITY",
        "created_utc": "2026-09-08T12:00:00Z",
        "technical_scope": "offline_train_val_only_no_test_no_broker_v1",
        "source": {
            "repo_root": "/srv/gx1/GX1_ENGINE",
            "commit": "a" * 40,
            "clean": True,
        },
        "provider": {
            "name": "provider",
            "region": "region-a",
            "instance_type": "h200-1x",
            "instance_id": "instance-1",
            "hourly_price_usd": 3.5,
        },
        "host": {
            "machine_id_sha256": "b" * 64,
            "kernel_release": "6.8.0",
            "os_id": "ubuntu",
            "os_version_id": "24.04",
            "logical_cpu_count": 16,
            "memory_total_bytes": 200 * 1024**3,
            "systemd_available": True,
            "cgroup_v2": True,
            "scratch_path": "/mnt/gx1",
            "scratch_free_bytes": 500 * 1024**3,
        },
        "gpu": {
            "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
            "name": "NVIDIA H200",
            "driver_version": "570.00",
            "compute_capability": [9, 0],
            "memory_total_mib": 143771,
            "bf16_supported": True,
            "torch_version": "2.6.0+cu124",
            "torch_cuda_version": "12.4",
        },
        "telemetry": {
            "owner": "root_owned_linux_loopback_v1",
            "url": "http://127.0.0.1:38128/gx1/v1/telemetry/",
            "certificate_path": str(certificate),
            "certificate_sha256": hashlib.sha256(certificate.read_bytes()).hexdigest(),
            "service_unit": "gx1-host-telemetry.service",
            "service_executable_path": str(service),
            "service_executable_sha256": hashlib.sha256(service.read_bytes()).hexdigest(),
            "timeout_seconds": 2,
        },
        "limits": {
            "memory_max_bytes": 160 * 1024**3,
            "memory_swap_max_bytes": 512 * 1024**2,
            "tasks_max": 512,
            "cpu_affinity": "0-15",
            "numerical_threads": 16,
            "max_wall_seconds": HARD_DEADLINE_SECONDS,
            "max_core_temp_c": 75,
            "max_memory_temp_c": 90,
            "max_power_limit_w": 700,
            "max_power_draw_w": 720,
            "max_memory_used_mib": 136000,
            "monitor_interval_seconds": 1,
            "minimum_available_memory_bytes": 180 * 1024**3,
            "minimum_scratch_free_bytes": 300 * 1024**3,
        },
        "budget": {
            "hard_cost_cap_nok": 2500.0,
            "hard_deadline_seconds": HARD_DEADLINE_SECONDS,
            "admission_max_projected_seconds": ADMISSION_MAX_PROJECTED_SECONDS,
            "nok_per_usd": 10.0,
            "fx_observed_utc": "2026-09-08T12:00:00Z",
            "cost_buffer_fraction": 0.10,
        },
        "termination": {
            "provider_managed_delete": True,
            "provider_deadline_utc": "2026-09-10T12:00:00Z",
            "provider_proof_path": str(proof),
            "provider_proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
            "local_timer_unit": "gx1-cloud-deadline.timer",
            "local_timer_active": True,
        },
        "activation_authority": False,
        "report_only": True,
        "side_effects_started": {
            "training": False,
            "test_read": False,
            "replay": False,
            "paper": False,
            "live": False,
        },
    }


def test_cloud_host_profile_accepts_exact_guarded_h200(tmp_path: Path) -> None:
    profile = require_cloud_training_host_profile(_profile(tmp_path))
    fields = runner_fields(profile)
    assert fields["CUDA_VISIBLE_DEVICES"] == profile["gpu"]["uuid"]
    assert fields["TRAINER_HOST_TELEMETRY_GPU_UUID"] == fields["CUDA_VISIBLE_DEVICES"]
    assert fields["TRAINER_TELEMETRY_OWNER"] == "root_owned_linux_loopback_v1"
    assert fields["SAFE_JOB_MEMORY_KIB"] == str(160 * 1024**2)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.update(report_only=False), "safety boundary"),
        (lambda value: value["gpu"].update(name="NVIDIA RTX 3090"), "H100/H200"),
        (lambda value: value["gpu"].update(compute_capability=[8, 0]), "Hopper"),
        (
            lambda value: value["telemetry"].update(url="http://0.0.0.0:38128/gx1/v1/telemetry/"),
            "loopback",
        ),
        (lambda value: value["limits"].update(cpu_affinity="0-15,15"), "overlapping"),
        (lambda value: value["limits"].update(max_wall_seconds=172799), "48h"),
        (lambda value: value["budget"].update(hard_cost_cap_nok=2600), "2500"),
        (lambda value: value["termination"].update(provider_managed_delete=False), "provider delete"),
    ],
)
def test_cloud_host_profile_fails_closed(
    tmp_path: Path, mutator: object, message: str
) -> None:
    profile = _profile(tmp_path)
    mutator(profile)
    with pytest.raises(CloudTrainingHostProfileError, match=message):
        require_cloud_training_host_profile(profile)


def test_cloud_host_profile_rejects_price_that_can_cross_cap(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile["provider"]["hourly_price_usd"] = 5.0
    with pytest.raises(CloudTrainingHostProfileError, match="hard NOK cap"):
        require_cloud_training_host_profile(profile)


def test_cloud_host_profile_file_is_hash_bound(tmp_path: Path) -> None:
    profile_path = tmp_path / "host-profile.json"
    profile_path.write_text(json.dumps(_profile(tmp_path)), encoding="utf-8")
    sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    loaded = require_cloud_training_host_profile_file(
        profile_path.resolve(),
        sha256,
        repo=tmp_path,
        verify_runtime=False,
    )
    assert loaded["decision"] == "PASS_HOST_QUALIFIED_NOT_TRAINING_AUTHORITY"
    profile_path.write_text("{}", encoding="utf-8")
    with pytest.raises(CloudTrainingHostProfileError, match="SHA-256 mismatch"):
        require_cloud_training_host_profile_file(
            profile_path.resolve(),
            sha256,
            repo=tmp_path,
            verify_runtime=False,
        )


def test_cloud_host_profile_keys_are_exact(tmp_path: Path) -> None:
    profile = copy.deepcopy(_profile(tmp_path))
    profile["gpu"]["extra"] = True
    with pytest.raises(CloudTrainingHostProfileError, match="keys are not exact"):
        require_cloud_training_host_profile(profile)
