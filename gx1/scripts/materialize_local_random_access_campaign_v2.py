"""Materialize immutable phase-1/phase-2 random-access campaign plans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.local_random_access_campaign_v2 import (
    EXECUTION_MANIFEST_SCHEMA,
    INVOCATION_SCHEMA,
    PLAN_SCHEMA,
    RandomAccessCampaignError,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    require_boot_identity,
    require_plan,
)
from gx1.contracts.unified_exit_gpu_batch_selection_v1 import require_selection


def _read(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.resolve() != path or not path.is_file() or path.is_symlink():
        raise RandomAccessCampaignError("materializer input must be an absolute regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RandomAccessCampaignError("materializer JSON object required")
    return value


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _internal_manifest_sha(value: Mapping[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _require_manifest(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_file_sha256:
        raise RandomAccessCampaignError("manifest file SHA-256 mismatch")
    value = _read(path)
    claimed = value.get("manifest_sha256")
    if claimed != _internal_manifest_sha(value):
        raise RandomAccessCampaignError("manifest internal SHA-256 mismatch")
    if value.get("test_data_used") is not False:
        raise RandomAccessCampaignError("TEST-bound manifest forbidden")
    return value


def _source_commit(repo: Path) -> str:
    try:
        head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain=v1", "--untracked-files=all"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RandomAccessCampaignError("source repository state unavailable") from exc
    if dirty:
        raise RandomAccessCampaignError("source repository must be clean")
    return head


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RandomAccessCampaignError(f"refusing to replace output: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sources(repo: Path, certificate: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    guards = {
        "runner": _binding(repo / "scripts/gx1_capped_run.sh"),
        "guard": _binding(repo / "scripts/gx1_guarded_trainer_exec.sh"),
        "query": _binding(repo / "scripts/gx1_host_telemetry_bridge_query.sh"),
        "certificate": _binding(certificate),
    }
    controllers = {
        "controller": _binding(repo / "scripts/windows/GX1-RandomAccessCampaignV2Controller.ps1"),
        "observer": _binding(repo / "scripts/windows/GX1-RandomAccessCampaignV2Progress.ps1"),
        "campaign_cli": _binding(repo / "gx1/scripts/local_random_access_campaign_v2.py"),
    }
    return guards, controllers


def _base_plan(
    *, phase: str, campaign_id: str, repo: Path, commit: str, runtime: Path,
    gpu_uuid: str, boot: Mapping[str, Any], guards: Mapping[str, Any],
    controllers: Mapping[str, Any], prior: Mapping[str, Any] | None,
    selection: Mapping[str, Any] | None, selected_batch_size: int | None,
    invocations: list[Mapping[str, str]],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "decision": "PASS_PREPARED",
        "campaign_id": campaign_id,
        "phase": phase,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(repo),
        "source_commit": commit,
        "runtime_root": str(runtime),
        "gpu_uuid": gpu_uuid,
        "prepared_windows_boot": dict(boot),
        "prior_campaign": prior,
        "selection_receipt": selection,
        "selected_batch_size": selected_batch_size,
        "entry_pairs_per_epoch": 16384,
        "transitions_per_epoch": 65536,
        "invocations": invocations,
        "signed_guard_sources": dict(guards),
        "controller_sources": dict(controllers),
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
        "authority": {"test": False, "promotion": False, "paper": False, "live": False, "cloud_spend": False},
        "test_data_used": False,
    }
    value["plan_sha256"] = canonical_sha256(value)
    return value


def _launcher(
    *, repo: Path, stage: str, launch: Path, checkpoint_dir: Path, batch: int,
    progress_path: Path, session: Path | None = None, max_steps: int | None = None,
) -> list[str]:
    attended = stage in {"smoke-arm", "reference-4", "resume-proof-first", "resume-proof-second"}
    argv = [str(repo / "scripts/gx1_capped_run.sh"), "--class", "trainer", "--mem", "20G", "--swap", "512M"]
    if attended:
        argv.append("--attended-smoke")
    argv.extend([
        "--", str(repo / ".venv/bin/python"), "-m",
        "gx1.scripts.run_unified_exit_random_access_fixed_step_v1",
        "--launch-manifest", str(launch), "--stage", stage,
        "--checkpoint-dir", str(checkpoint_dir), "--arm-batch-size", str(batch),
        "--progress-path", str(progress_path),
    ])
    if session is not None:
        argv.extend(["--train-session-manifest", str(session)])
    if max_steps is not None:
        argv.extend(["--max-optimizer-steps", str(max_steps)])
    argv.extend(["--device", "cuda"])
    return argv


def _write_invocation(
    *, output: Path, repo: Path, commit: str, runtime: Path, number: int,
    kind: str, batch: int, budget: int, outcome: str, stage: str,
    launch_path: Path, launch: Mapping[str, Any], checkpoint_dir: Path,
    progress_path: Path, pointer_path: Path, before_mode: str,
    predecessor: int | None, session_path: Path | None,
    session: Mapping[str, Any] | None, window_index: int = 0,
    window_count: int = 0,
) -> tuple[dict[str, Any], dict[str, str]]:
    argv = _launcher(
        repo=repo, stage=stage, launch=launch_path, checkpoint_dir=checkpoint_dir,
        batch=batch, progress_path=progress_path, session=session_path,
        max_steps=budget if kind == "epoch1_window" else None,
    )
    execution: dict[str, Any] = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA,
        "decision": "PASS_EXACT_INVOCATION",
        "invocation_kind": kind,
        "python_module": "gx1.scripts.run_unified_exit_random_access_fixed_step_v1",
        "source_commit": commit,
        "launcher_argv_sha256": canonical_sha256(argv),
        "prelaunch_manifest": _binding(launch_path),
        "prelaunch_manifest_sha256": launch["manifest_sha256"],
        "train_session_manifest": _binding(session_path) if session_path else None,
        "train_session_manifest_sha256": session["manifest_sha256"] if session else None,
        "test_data_used": False,
    }
    execution["artifact_sha256"] = canonical_sha256(execution)
    execution_path = output / "execution-manifests" / f"invocation-{number:04d}.json"
    _atomic_json(execution_path, execution)
    invocation: dict[str, Any] = {
        "schema_version": INVOCATION_SCHEMA,
        "decision": "PASS",
        "invocation_number": number,
        "invocation_id": f"invocation-{number:04d}",
        "kind": kind,
        "batch_size": batch,
        "epoch_index": 0,
        "window_index": window_index,
        "window_count": window_count,
        "optimizer_step_budget": budget,
        "expected_success_outcome": outcome,
        "source_commit": commit,
        "execution_manifest": _binding(execution_path),
        "launcher_argv": argv,
        "launcher_argv_sha256": canonical_sha256(argv),
        "progress_path": str(progress_path),
        "guard_log_path": str(runtime / "guard" / f"invocation-{number:04d}.log"),
        "checkpoint": {
            "pointer_path": str(pointer_path), "before_mode": before_mode,
            "predecessor_invocation_number": predecessor, "write_mode": "ATOMIC_UPDATE",
        },
        "maximum_wall_seconds": 7200 if kind == "epoch1_window" else 1800,
        "requires_fresh_windows_boot": True,
        "signed_guard_only": True,
        "test_data_used": False,
    }
    invocation["invocation_sha256"] = canonical_sha256(invocation)
    invocation_path = output / "invocations" / f"invocation-{number:04d}.json"
    _atomic_json(invocation_path, invocation)
    return invocation, _binding(invocation_path)


def materialize_gpu_selection_campaign(
    *, repo: Path, output: Path, runtime: Path, gpu_uuid: str,
    prepared_boot_path: Path, prepared_boot_file_sha256: str,
    prelaunch_path: Path, prelaunch_file_sha256: str, certificate_path: Path,
) -> dict[str, Any]:
    commit = _source_commit(repo)
    if output.exists() or output.is_symlink():
        raise RandomAccessCampaignError("campaign output must not exist")
    output.mkdir(parents=True)
    boot = require_boot_identity(_read(prepared_boot_path))
    if file_sha256(prepared_boot_path) != prepared_boot_file_sha256:
        raise RandomAccessCampaignError("prepared boot file SHA-256 mismatch")
    launch = _require_manifest(prelaunch_path, prelaunch_file_sha256)
    if launch.get("schema_version") != "gx1_unified_exit_random_access_cuda_smoke_launch_v2":
        raise RandomAccessCampaignError("CUDA prelaunch schema invalid")
    checkpoint_root = Path(str(launch["checkpoint_dir"]))
    invocations: list[dict[str, str]] = []
    for number, batch in enumerate((4, 8, 16), 1):
        checkpoint_dir = checkpoint_root / f"batch_{batch}"
        _, binding = _write_invocation(
            output=output, repo=repo, commit=commit, runtime=runtime, number=number,
            kind="smoke_arm", batch=batch, budget=3, outcome="COMPLETE",
            stage="smoke-arm", launch_path=prelaunch_path, launch=launch,
            checkpoint_dir=checkpoint_dir, progress_path=checkpoint_dir / "PROGRESS.json",
            pointer_path=checkpoint_dir / "RESUME_POINTER.json", before_mode="GENESIS",
            predecessor=None, session_path=None, session=None,
        )
        invocations.append(binding)
    guards, controllers = _sources(repo, certificate_path)
    plan = _base_plan(
        phase="gpu_selection", campaign_id=f"GX1_RANDOM_ACCESS_GPU_SELECTION_{commit[:12]}",
        repo=repo, commit=commit, runtime=runtime, gpu_uuid=gpu_uuid, boot=boot,
        guards=guards, controllers=controllers, prior=None, selection=None,
        selected_batch_size=None, invocations=invocations,
    )
    plan_path = output / "CAMPAIGN_PLAN.json"
    _atomic_json(plan_path, plan)
    checked = require_plan(plan, verify_files=True)
    return {"plan": checked, "path": str(plan_path), "sha256": file_sha256(plan_path)}


def materialize_selected_training_campaign(
    *, repo: Path, output: Path, runtime: Path, gpu_uuid: str,
    prepared_boot_path: Path, prepared_boot_file_sha256: str,
    prelaunch_path: Path, prelaunch_file_sha256: str, certificate_path: Path,
    prior_campaign_path: Path, prior_campaign_file_sha256: str,
    selection_path: Path, selection_file_sha256: str,
    resume_session_path: Path, resume_session_file_sha256: str,
    epoch_session_path: Path, epoch_session_file_sha256: str,
    epoch_window_steps: int,
) -> dict[str, Any]:
    commit = _source_commit(repo)
    if output.exists() or output.is_symlink():
        raise RandomAccessCampaignError("campaign output must not exist")
    if epoch_window_steps < 1:
        raise RandomAccessCampaignError("epoch window steps must be positive")
    prior = require_plan(_read(prior_campaign_path), verify_files=True)
    if file_sha256(prior_campaign_path) != prior_campaign_file_sha256 or prior["source_commit"] != commit:
        raise RandomAccessCampaignError("prior campaign binding invalid")
    selection = require_selection(_read(selection_path), verify_files=True)
    if file_sha256(selection_path) != selection_file_sha256:
        raise RandomAccessCampaignError("selection file SHA-256 mismatch")
    launch = _require_manifest(prelaunch_path, prelaunch_file_sha256)
    resume_session = _require_manifest(resume_session_path, resume_session_file_sha256)
    epoch_session = _require_manifest(epoch_session_path, epoch_session_file_sha256)
    batch = int(selection["selected_batch_size"])
    if batch not in (4, 8, 16) or selection["source_commit"] != commit:
        raise RandomAccessCampaignError("selected batch provenance invalid")
    if selection["launch_manifest_sha256"] != launch["manifest_sha256"]:
        raise RandomAccessCampaignError("selection/prelaunch binding invalid")
    checkpoint_root = Path(str(launch["checkpoint_dir"]))
    definitions: list[dict[str, Any]] = [
        {"kind": "reference_run", "stage": "reference-4", "budget": 4,
         "dir": checkpoint_root / "comparison" / f"reference_4_batch_{batch}",
         "progress": checkpoint_root / "comparison" / f"reference_4_batch_{batch}" / "PROGRESS.json",
         "before": "GENESIS", "predecessor": None, "session_path": resume_session_path, "session": resume_session},
        {"kind": "resume_proof_first", "stage": "resume-proof-first", "budget": 3,
         "dir": checkpoint_root / "comparison" / f"split_batch_{batch}",
         "progress": checkpoint_root / "comparison" / f"split_batch_{batch}" / "PROGRESS_FIRST.json",
         "before": "GENESIS", "predecessor": None, "session_path": resume_session_path, "session": resume_session},
        {"kind": "resume_proof_second", "stage": "resume-proof-second", "budget": 1,
         "dir": checkpoint_root / "comparison" / f"split_batch_{batch}",
         "progress": checkpoint_root / "comparison" / f"split_batch_{batch}" / "PROGRESS_SECOND.json",
         "before": "PREVIOUS_RECEIPT_AFTER", "predecessor": 2, "session_path": resume_session_path, "session": resume_session},
    ]
    total_batches = math.ceil(16384 / batch)
    window_count = math.ceil(total_batches / epoch_window_steps)
    epoch_dir = checkpoint_root / f"epoch1_batch_{batch}"
    for index in range(window_count):
        budget = min(epoch_window_steps, total_batches - index * epoch_window_steps)
        definitions.append({
            "kind": "epoch1_window", "stage": "epoch1-window", "budget": budget,
            "dir": epoch_dir, "progress": epoch_dir / "PROGRESS.json",
            "before": "GENESIS" if index == 0 else "PREVIOUS_RECEIPT_AFTER",
            "predecessor": None if index == 0 else 3 + index,
            "session_path": epoch_session_path, "session": epoch_session,
            "window_index": index, "window_count": window_count,
        })
    invocations: list[dict[str, str]] = []
    for number, definition in enumerate(definitions, 1):
        _, binding = _write_invocation(
            output=output, repo=repo, commit=commit, runtime=runtime, number=number,
            kind=definition["kind"], batch=batch, budget=definition["budget"],
            outcome="COMPLETE" if definition["kind"] != "epoch1_window" or definition.get("window_index") == window_count - 1 else "RESUMABLE",
            stage=definition["stage"], launch_path=prelaunch_path, launch=launch,
            checkpoint_dir=definition["dir"], progress_path=definition["progress"],
            pointer_path=definition["dir"] / "RESUME_POINTER.json",
            before_mode=definition["before"], predecessor=definition["predecessor"],
            session_path=definition["session_path"], session=definition["session"],
            window_index=definition.get("window_index", 0), window_count=definition.get("window_count", 0),
        )
        invocations.append(binding)
    boot = require_boot_identity(_read(prepared_boot_path))
    if file_sha256(prepared_boot_path) != prepared_boot_file_sha256:
        raise RandomAccessCampaignError("prepared boot file SHA-256 mismatch")
    guards, controllers = _sources(repo, certificate_path)
    plan = _base_plan(
        phase="selected_training", campaign_id=f"GX1_RANDOM_ACCESS_SELECTED_TRAINING_{commit[:12]}",
        repo=repo, commit=commit, runtime=runtime, gpu_uuid=gpu_uuid, boot=boot,
        guards=guards, controllers=controllers, prior=_binding(prior_campaign_path),
        selection=_binding(selection_path), selected_batch_size=batch, invocations=invocations,
    )
    plan_path = output / "CAMPAIGN_PLAN.json"
    _atomic_json(plan_path, plan)
    checked = require_plan(plan, verify_files=True)
    return {"plan": checked, "path": str(plan_path), "sha256": file_sha256(plan_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gpu-selection", "selected-training"), required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--prepared-boot", type=Path, required=True)
    parser.add_argument("--prepared-boot-file-sha256", required=True)
    parser.add_argument("--prelaunch-manifest", type=Path, required=True)
    parser.add_argument("--prelaunch-file-sha256", required=True)
    parser.add_argument("--telemetry-certificate", type=Path, required=True)
    parser.add_argument("--prior-campaign", type=Path)
    parser.add_argument("--prior-campaign-file-sha256")
    parser.add_argument("--gpu-selection", type=Path)
    parser.add_argument("--gpu-selection-file-sha256")
    parser.add_argument("--resume-session-manifest", type=Path)
    parser.add_argument("--resume-session-file-sha256")
    parser.add_argument("--epoch-session-manifest", type=Path)
    parser.add_argument("--epoch-session-file-sha256")
    parser.add_argument("--epoch-window-steps", type=int, default=64)
    args = parser.parse_args(argv)
    common = dict(
        repo=args.source_repo.resolve(), output=args.output_root.resolve(),
        runtime=args.runtime_root.resolve(), gpu_uuid=args.gpu_uuid,
        prepared_boot_path=args.prepared_boot.resolve(),
        prepared_boot_file_sha256=args.prepared_boot_file_sha256,
        prelaunch_path=args.prelaunch_manifest.resolve(),
        prelaunch_file_sha256=args.prelaunch_file_sha256,
        certificate_path=args.telemetry_certificate.resolve(),
    )
    if args.phase == "gpu-selection":
        result = materialize_gpu_selection_campaign(**common)
    else:
        required = (
            args.prior_campaign, args.prior_campaign_file_sha256, args.gpu_selection,
            args.gpu_selection_file_sha256, args.resume_session_manifest,
            args.resume_session_file_sha256, args.epoch_session_manifest,
            args.epoch_session_file_sha256,
        )
        if any(value is None for value in required):
            raise RandomAccessCampaignError("selected-training inputs incomplete")
        result = materialize_selected_training_campaign(
            **common,
            prior_campaign_path=args.prior_campaign.resolve(),
            prior_campaign_file_sha256=args.prior_campaign_file_sha256,
            selection_path=args.gpu_selection.resolve(),
            selection_file_sha256=args.gpu_selection_file_sha256,
            resume_session_path=args.resume_session_manifest.resolve(),
            resume_session_file_sha256=args.resume_session_file_sha256,
            epoch_session_path=args.epoch_session_manifest.resolve(),
            epoch_session_file_sha256=args.epoch_session_file_sha256,
            epoch_window_steps=args.epoch_window_steps,
        )
    print(json.dumps({"ok": True, "path": result["path"], "sha256": result["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
