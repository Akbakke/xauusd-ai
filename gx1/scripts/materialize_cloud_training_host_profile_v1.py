"""Materialize one immutable live H100/H200 host qualification profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gx1.contracts.cloud_training_host_profile_v1 import (
    ADMISSION_MAX_PROJECTED_SECONDS,
    HARD_COST_CAP_NOK,
    HARD_DEADLINE_SECONDS,
    SCHEMA_VERSION,
    SIDE_EFFECTS_ZERO,
    TECHNICAL_SCOPE,
    TELEMETRY_OWNER,
    require_cloud_training_host_profile_file,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[2]


class HostProfileMaterializationError(RuntimeError):
    pass


def _run(command: list[str], label: str) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostProfileMaterializationError(f"{label} failed") from exc
    return result.stdout.strip()


def _os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    for row in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        key, separator, value = row.partition("=")
        if separator:
            result[key] = value.strip().strip('"')
    return result


def _memory() -> dict[str, int]:
    rows = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    return {
        row.split(":", 1)[0]: int(row.split()[1]) * 1024
        for row in rows
        if row.startswith(("MemTotal:", "MemAvailable:"))
    }


def _write_exclusive(path: Path, payload: dict[str, object]) -> str:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise HostProfileMaterializationError("output must be a new absolute path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--hourly-price-usd", type=float, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--scratch-path", type=Path, required=True)
    parser.add_argument("--certificate-path", type=Path, required=True)
    parser.add_argument("--service-executable-path", type=Path, required=True)
    parser.add_argument("--provider-proof-path", type=Path, required=True)
    parser.add_argument("--provider-proof-sha256", required=True)
    parser.add_argument("--provider-deadline-utc", required=True)
    parser.add_argument("--nok-per-usd", type=float, required=True)
    parser.add_argument("--fx-observed-utc", required=True)
    parser.add_argument("--memory-max-gib", type=int, required=True)
    parser.add_argument("--memory-swap-max-mib", type=int, required=True)
    parser.add_argument("--tasks-max", type=int, required=True)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument("--numerical-threads", type=int, required=True)
    parser.add_argument("--max-core-temp-c", type=int, required=True)
    parser.add_argument("--max-memory-temp-c", type=int, required=True)
    parser.add_argument("--max-power-limit-w", type=int, required=True)
    parser.add_argument("--max-power-draw-w", type=int, required=True)
    parser.add_argument("--max-memory-used-mib", type=int, required=True)
    parser.add_argument("--minimum-available-memory-gib", type=int, required=True)
    parser.add_argument("--minimum-scratch-free-gib", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    head = _run(["/usr/bin/git", "-C", str(REPO), "rev-parse", "HEAD"], "Git HEAD")
    if _run(
        ["/usr/bin/git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
        "Git status",
    ):
        parser.error("source tree must be clean")
    all_gpu_rows = _run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        "NVIDIA inventory",
    ).splitlines()
    if len(all_gpu_rows) != 1:
        parser.error("cloud training host must expose exactly one physical GPU")
    gpu_fields = [field.strip() for field in all_gpu_rows[0].split(",")]
    if len(gpu_fields) != 5 or gpu_fields[0] != args.gpu_uuid:
        parser.error("NVIDIA identity does not match the requested GPU UUID")
    gpu_uuid, gpu_name, driver_version, memory_total_mib, compute_capability = gpu_fields
    if compute_capability != "9.0":
        parser.error("cloud training host must expose Hopper compute capability 9.0")
    try:
        import torch
    except ImportError as exc:
        parser.error(f"PyTorch unavailable: {exc}")
    if torch.version.cuda is None:
        parser.error("PyTorch CUDA build is unavailable")

    scratch = args.scratch_path.resolve(strict=True)
    if scratch.is_symlink() or not scratch.is_dir():
        parser.error("scratch path must be a canonical regular directory")
    scratch_free = os.statvfs(scratch).f_bavail * os.statvfs(scratch).f_frsize
    memory = _memory()
    os_release = _os_release()
    certificate = args.certificate_path.resolve(strict=True)
    service = args.service_executable_path.resolve(strict=True)
    proof = args.provider_proof_path.resolve(strict=True)
    if sha256_file(proof) != args.provider_proof_sha256:
        parser.error("provider termination proof SHA-256 mismatch")
    created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS_HOST_QUALIFIED_NOT_TRAINING_AUTHORITY",
        "created_utc": created_utc,
        "technical_scope": TECHNICAL_SCOPE,
        "source": {"repo_root": str(REPO), "commit": head, "clean": True},
        "provider": {
            "name": args.provider,
            "region": args.region,
            "instance_type": args.instance_type,
            "instance_id": args.instance_id,
            "hourly_price_usd": args.hourly_price_usd,
        },
        "host": {
            "machine_id_sha256": hashlib.sha256(Path("/etc/machine-id").read_bytes().strip()).hexdigest(),
            "kernel_release": os.uname().release,
            "os_id": os_release.get("ID", ""),
            "os_version_id": os_release.get("VERSION_ID", ""),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_bytes": memory.get("MemTotal", 0),
            "systemd_available": Path("/run/systemd/system").is_dir(),
            "cgroup_v2": Path("/sys/fs/cgroup/cgroup.controllers").is_file(),
            "scratch_path": str(scratch),
            "scratch_free_bytes": scratch_free,
        },
        "gpu": {
            "uuid": gpu_uuid,
            "name": gpu_name,
            "driver_version": driver_version,
            "compute_capability": [9, 0],
            "memory_total_mib": int(float(memory_total_mib)),
            "bf16_supported": True,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
        },
        "telemetry": {
            "owner": TELEMETRY_OWNER,
            "url": "http://127.0.0.1:38128/gx1/v1/telemetry/",
            "certificate_path": str(certificate),
            "certificate_sha256": sha256_file(certificate),
            "service_unit": "gx1-host-telemetry.service",
            "service_executable_path": str(service),
            "service_executable_sha256": sha256_file(service),
            "timeout_seconds": 2,
        },
        "limits": {
            "memory_max_bytes": args.memory_max_gib * 1024**3,
            "memory_swap_max_bytes": args.memory_swap_max_mib * 1024**2,
            "tasks_max": args.tasks_max,
            "cpu_affinity": args.cpu_affinity,
            "numerical_threads": args.numerical_threads,
            "max_wall_seconds": HARD_DEADLINE_SECONDS,
            "max_core_temp_c": args.max_core_temp_c,
            "max_memory_temp_c": args.max_memory_temp_c,
            "max_power_limit_w": args.max_power_limit_w,
            "max_power_draw_w": args.max_power_draw_w,
            "max_memory_used_mib": args.max_memory_used_mib,
            "monitor_interval_seconds": 1,
            "minimum_available_memory_bytes": args.minimum_available_memory_gib * 1024**3,
            "minimum_scratch_free_bytes": args.minimum_scratch_free_gib * 1024**3,
        },
        "budget": {
            "hard_cost_cap_nok": HARD_COST_CAP_NOK,
            "hard_deadline_seconds": HARD_DEADLINE_SECONDS,
            "admission_max_projected_seconds": ADMISSION_MAX_PROJECTED_SECONDS,
            "nok_per_usd": args.nok_per_usd,
            "fx_observed_utc": args.fx_observed_utc,
            "cost_buffer_fraction": 0.10,
        },
        "termination": {
            "provider_managed_delete": True,
            "provider_deadline_utc": args.provider_deadline_utc,
            "provider_proof_path": str(proof),
            "provider_proof_sha256": args.provider_proof_sha256,
            "local_timer_unit": "gx1-cloud-deadline.timer",
            "local_timer_active": True,
        },
        "activation_authority": False,
        "report_only": True,
        "side_effects_started": SIDE_EFFECTS_ZERO,
    }
    try:
        profile_sha256 = _write_exclusive(args.output_json, payload)
        require_cloud_training_host_profile_file(
            args.output_json,
            profile_sha256,
            repo=REPO,
            verify_runtime=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        args.output_json.unlink(missing_ok=True)
        parser.error(str(exc))
    print(json.dumps({"path": str(args.output_json), "sha256": profile_sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
