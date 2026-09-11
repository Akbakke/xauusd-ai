"""Materialize immutable random-access campaign plans at each evidence gate."""

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
    final_train_checkpoint_authority: Mapping[str, Any] | None = None,
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
        "final_train_checkpoint_authority": final_train_checkpoint_authority,
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
    epoch_session_path: Path | None, epoch_session_file_sha256: str | None,
    epoch_window_steps: int,
    resume_proof_only: bool = False,
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
    if resume_proof_only:
        if prior["phase"] != "gpu_selection" or epoch_session_path is not None or epoch_session_file_sha256 is not None:
            raise RandomAccessCampaignError("resume proof must precede epoch authorization")
        epoch_session = None
    else:
        if epoch_session_path is None or epoch_session_file_sha256 is None:
            raise RandomAccessCampaignError("epoch session required after resume proof")
        epoch_session = _require_manifest(epoch_session_path, epoch_session_file_sha256)
    if resume_proof_only or prior["phase"] == "resume_proof":
        from gx1.contracts.unified_exit_train_session_manifest_v1 import require_train_session_manifest
        for phase, session in [("resume_proof", resume_session)] + (
            [] if resume_proof_only else [("epoch1", epoch_session)]
        ):
            checked_session = require_train_session_manifest(session, expected_phase=phase, verify_files=True)
            if (checked_session["source_commit"] != commit
                or checked_session["prelaunch_manifest_sha256"] != launch["manifest_sha256"]
                or checked_session["gpu_batch_selection_artifact_sha256"] != selection["artifact_sha256"]):
                raise RandomAccessCampaignError("train session provenance invalid")
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
    if prior["phase"] == "resume_proof":
        definitions = []
    proof_invocation_count = len(definitions)
    for index in range(0 if resume_proof_only else window_count):
        budget = min(epoch_window_steps, total_batches - index * epoch_window_steps)
        definitions.append({
            "kind": "epoch1_window", "stage": "epoch1-window", "budget": budget,
            "dir": epoch_dir, "progress": epoch_dir / "PROGRESS.json",
            "before": "GENESIS" if index == 0 else "PREVIOUS_RECEIPT_AFTER",
            "predecessor": None if index == 0 else proof_invocation_count + index,
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
        phase="resume_proof" if resume_proof_only else "selected_training",
        campaign_id=f"GX1_RANDOM_ACCESS_{'RESUME_PROOF' if resume_proof_only else 'SELECTED_TRAINING'}_{commit[:12]}",
        repo=repo, commit=commit, runtime=runtime, gpu_uuid=gpu_uuid, boot=boot,
        guards=guards, controllers=controllers, prior=_binding(prior_campaign_path),
        selection=_binding(selection_path), selected_batch_size=batch, invocations=invocations,
    )
    plan_path = output / "CAMPAIGN_PLAN.json"
    _atomic_json(plan_path, plan)
    checked = require_plan(plan, verify_files=True)
    return {"plan": checked, "path": str(plan_path), "sha256": file_sha256(plan_path)}


def _full_val_launcher(
    *,
    repo: Path,
    launch_path: Path,
    authority_path: Path,
    authority_file_sha256: str,
    pointer_path: Path,
    campaign_progress_path: Path,
    rollout_progress_path: Path,
    result_path: Path,
    max_forwards: int,
    progress_interval_forwards: int,
    max_model_forwards: int,
    max_materialized_state_views: int,
    max_wall_seconds: int,
) -> list[str]:
    return [
        str(repo / "scripts/gx1_capped_run.sh"),
        "--class",
        "trainer",
        "--mem",
        "20G",
        "--swap",
        "512M",
        "--",
        str(repo / ".venv/bin/python"),
        "-m",
        "gx1.scripts.run_unified_exit_random_access_val_v1",
        "--launch-manifest",
        str(launch_path),
        "--final-train-checkpoint-authority",
        str(authority_path),
        "--final-train-checkpoint-authority-file-sha256",
        authority_file_sha256,
        "--checkpoint-pointer",
        str(pointer_path),
        "--progress-path",
        str(campaign_progress_path),
        "--rollout-progress-path",
        str(rollout_progress_path),
        "--result-path",
        str(result_path),
        "--device",
        "cuda",
        "--max-forwards-this-invocation",
        str(max_forwards),
        "--progress-interval-forwards",
        str(progress_interval_forwards),
        "--compute-guard-max-model-forwards",
        str(max_model_forwards),
        "--compute-guard-max-materialized-state-views",
        str(max_materialized_state_views),
        "--compute-guard-max-wall-seconds",
        str(max_wall_seconds),
    ]


def _write_full_val_invocation(
    *,
    output: Path,
    repo: Path,
    commit: str,
    runtime: Path,
    number: int,
    batch: int,
    window_count: int,
    launch_path: Path,
    launch: Mapping[str, Any],
    authority_path: Path,
    authority_file_sha256: str,
    pointer_path: Path,
    rollout_progress_path: Path,
    result_path: Path,
    max_forwards: int,
    progress_interval_forwards: int,
    max_model_forwards: int,
    max_materialized_state_views: int,
    max_wall_seconds: int,
) -> dict[str, str]:
    progress_path = runtime / "progress" / f"full-val-window-{number:04d}.json"
    argv = _full_val_launcher(
        repo=repo,
        launch_path=launch_path,
        authority_path=authority_path,
        authority_file_sha256=authority_file_sha256,
        pointer_path=pointer_path,
        campaign_progress_path=progress_path,
        rollout_progress_path=rollout_progress_path,
        result_path=result_path,
        max_forwards=max_forwards,
        progress_interval_forwards=progress_interval_forwards,
        max_model_forwards=max_model_forwards,
        max_materialized_state_views=max_materialized_state_views,
        max_wall_seconds=max_wall_seconds,
    )
    execution: dict[str, Any] = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA,
        "decision": "PASS_EXACT_INVOCATION",
        "invocation_kind": "full_val_window",
        "python_module": "gx1.scripts.run_unified_exit_random_access_val_v1",
        "source_commit": commit,
        "launcher_argv_sha256": canonical_sha256(argv),
        "prelaunch_manifest": _binding(launch_path),
        "prelaunch_manifest_sha256": launch["manifest_sha256"],
        "train_session_manifest": None,
        "train_session_manifest_sha256": None,
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
        "kind": "full_val_window",
        "batch_size": batch,
        "epoch_index": 0,
        "window_index": number - 1,
        "window_count": window_count,
        "optimizer_step_budget": None,
        "expected_success_outcome": "RESUMABLE_OR_COMPLETE",
        "source_commit": commit,
        "execution_manifest": _binding(execution_path),
        "launcher_argv": argv,
        "launcher_argv_sha256": canonical_sha256(argv),
        "progress_path": str(progress_path),
        "guard_log_path": str(runtime / "guard" / f"invocation-{number:04d}.log"),
        "checkpoint": {
            "pointer_path": str(pointer_path),
            "before_mode": "FINAL_AUTHORITY"
            if number == 1
            else "PREVIOUS_RECEIPT_AFTER",
            "predecessor_invocation_number": None if number == 1 else number - 1,
            "write_mode": "READ_ONLY",
        },
        "maximum_wall_seconds": max_wall_seconds,
        "requires_fresh_windows_boot": True,
        "signed_guard_only": True,
        "test_data_used": False,
    }
    invocation["invocation_sha256"] = canonical_sha256(invocation)
    invocation_path = output / "invocations" / f"invocation-{number:04d}.json"
    _atomic_json(invocation_path, invocation)
    return _binding(invocation_path)


def materialize_full_val_campaign(
    *,
    repo: Path,
    output: Path,
    runtime: Path,
    gpu_uuid: str,
    prepared_boot_path: Path,
    prepared_boot_file_sha256: str,
    certificate_path: Path,
    prior_campaign_path: Path,
    prior_campaign_file_sha256: str,
    selection_path: Path,
    selection_file_sha256: str,
    final_authority_path: Path,
    final_authority_file_sha256: str,
    window_count: int,
    max_forwards_per_window: int,
    progress_interval_forwards: int,
    max_model_forwards: int,
    max_materialized_state_views: int,
    max_wall_seconds: int,
) -> dict[str, Any]:
    from gx1.contracts.unified_exit_final_train_checkpoint_authority_v1 import (
        require_final_train_checkpoint_authority,
    )

    commit = _source_commit(repo)
    if output.exists() or output.is_symlink():
        raise RandomAccessCampaignError("campaign output must not exist")
    if (
        window_count < 1
        or max_forwards_per_window < 1
        or progress_interval_forwards < 1
        or max_model_forwards < max_forwards_per_window
        or max_materialized_state_views < max_model_forwards
        or not 1 <= max_wall_seconds <= 7200
    ):
        raise RandomAccessCampaignError("full VAL window bounds invalid")
    prior = require_plan(_read(prior_campaign_path), verify_files=True)
    selection = require_selection(_read(selection_path), verify_files=True)
    authority = require_final_train_checkpoint_authority(
        _read(final_authority_path), verify_files=True
    )
    if (
        file_sha256(prior_campaign_path) != prior_campaign_file_sha256
        or file_sha256(selection_path) != selection_file_sha256
        or file_sha256(final_authority_path) != final_authority_file_sha256
        or prior["phase"] != "selected_training"
        or prior["source_commit"] != commit
        or selection["artifact_sha256"]
        != authority["gpu_batch_selection_artifact_sha256"]
        or authority["campaign_plan"] != _binding(prior_campaign_path)
        or authority["gpu_batch_selection"] != _binding(selection_path)
        or authority["source_commit"] != commit
    ):
        raise RandomAccessCampaignError("full VAL authority provenance invalid")
    launch_path = Path(authority["launch_manifest"]["path"])
    launch = _require_manifest(launch_path, authority["launch_manifest"]["sha256"])
    pointer_path = Path(authority["final_checkpoint_pointer"]["path"])
    batch = int(authority["selected_batch_size"])
    rollout_path = runtime / "rollout" / "ROLLOUT_PROGRESS.json"
    result_path = runtime / "rollout" / "VAL_RESULT.json"
    output.mkdir(parents=True)
    invocations = [
        _write_full_val_invocation(
            output=output,
            repo=repo,
            commit=commit,
            runtime=runtime,
            number=number,
            batch=batch,
            window_count=window_count,
            launch_path=launch_path,
            launch=launch,
            authority_path=final_authority_path,
            authority_file_sha256=final_authority_file_sha256,
            pointer_path=pointer_path,
            rollout_progress_path=rollout_path,
            result_path=result_path,
            max_forwards=max_forwards_per_window,
            progress_interval_forwards=progress_interval_forwards,
            max_model_forwards=max_model_forwards,
            max_materialized_state_views=max_materialized_state_views,
            max_wall_seconds=max_wall_seconds,
        )
        for number in range(1, window_count + 1)
    ]
    boot = require_boot_identity(_read(prepared_boot_path))
    if file_sha256(prepared_boot_path) != prepared_boot_file_sha256:
        raise RandomAccessCampaignError("prepared boot file SHA-256 mismatch")
    guards, controllers = _sources(repo, certificate_path)
    plan = _base_plan(
        phase="full_val",
        campaign_id=f"GX1_RANDOM_ACCESS_FULL_VAL_{commit[:12]}",
        repo=repo,
        commit=commit,
        runtime=runtime,
        gpu_uuid=gpu_uuid,
        boot=boot,
        guards=guards,
        controllers=controllers,
        prior=_binding(prior_campaign_path),
        selection=_binding(selection_path),
        selected_batch_size=batch,
        invocations=invocations,
        final_train_checkpoint_authority=_binding(final_authority_path),
    )
    plan_path = output / "CAMPAIGN_PLAN.json"
    _atomic_json(plan_path, plan)
    checked = require_plan(plan, verify_files=True)
    return {"plan": checked, "path": str(plan_path), "sha256": file_sha256(plan_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("gpu-selection", "resume-proof", "selected-training", "full-val"),
        required=True,
    )
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--prepared-boot", type=Path, required=True)
    parser.add_argument("--prepared-boot-file-sha256", required=True)
    parser.add_argument("--prelaunch-manifest", type=Path)
    parser.add_argument("--prelaunch-file-sha256")
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
    parser.add_argument("--final-train-checkpoint-authority", type=Path)
    parser.add_argument("--final-train-checkpoint-authority-file-sha256")
    parser.add_argument("--full-val-window-count", type=int)
    parser.add_argument("--max-forwards-per-window", type=int)
    parser.add_argument("--progress-interval-forwards", type=int, default=64)
    parser.add_argument("--compute-guard-max-model-forwards", type=int)
    parser.add_argument("--compute-guard-max-materialized-state-views", type=int)
    parser.add_argument("--compute-guard-max-wall-seconds", type=int)
    args = parser.parse_args(argv)
    base = dict(
        repo=args.source_repo.resolve(),
        output=args.output_root.resolve(),
        runtime=args.runtime_root.resolve(),
        gpu_uuid=args.gpu_uuid,
        prepared_boot_path=args.prepared_boot.resolve(),
        prepared_boot_file_sha256=args.prepared_boot_file_sha256,
        certificate_path=args.telemetry_certificate.resolve(),
    )
    if args.phase == "gpu-selection":
        if args.prelaunch_manifest is None or args.prelaunch_file_sha256 is None:
            raise RandomAccessCampaignError("GPU-selection inputs incomplete")
        result = materialize_gpu_selection_campaign(
            **base,
            prelaunch_path=args.prelaunch_manifest.resolve(),
            prelaunch_file_sha256=args.prelaunch_file_sha256,
        )
    elif args.phase in {"resume-proof", "selected-training"}:
        required = (
            args.prelaunch_manifest,
            args.prelaunch_file_sha256,
            args.prior_campaign,
            args.prior_campaign_file_sha256,
            args.gpu_selection,
            args.gpu_selection_file_sha256,
            args.resume_session_manifest,
            args.resume_session_file_sha256,
        ) + (() if args.phase == "resume-proof" else (
            args.epoch_session_manifest, args.epoch_session_file_sha256,
        ))
        if any(value is None for value in required):
            raise RandomAccessCampaignError("selected-training inputs incomplete")
        result = materialize_selected_training_campaign(
            **base,
            prelaunch_path=args.prelaunch_manifest.resolve(),
            prelaunch_file_sha256=args.prelaunch_file_sha256,
            prior_campaign_path=args.prior_campaign.resolve(),
            prior_campaign_file_sha256=args.prior_campaign_file_sha256,
            selection_path=args.gpu_selection.resolve(),
            selection_file_sha256=args.gpu_selection_file_sha256,
            resume_session_path=args.resume_session_manifest.resolve(),
            resume_session_file_sha256=args.resume_session_file_sha256,
            epoch_session_path=args.epoch_session_manifest.resolve() if args.epoch_session_manifest else None,
            epoch_session_file_sha256=args.epoch_session_file_sha256,
            epoch_window_steps=args.epoch_window_steps,
            resume_proof_only=args.phase == "resume-proof",
        )
    else:
        required = (
            args.prior_campaign,
            args.prior_campaign_file_sha256,
            args.gpu_selection,
            args.gpu_selection_file_sha256,
            args.final_train_checkpoint_authority,
            args.final_train_checkpoint_authority_file_sha256,
            args.full_val_window_count,
            args.max_forwards_per_window,
            args.compute_guard_max_model_forwards,
            args.compute_guard_max_materialized_state_views,
            args.compute_guard_max_wall_seconds,
        )
        if any(value is None for value in required):
            raise RandomAccessCampaignError("full-VAL inputs incomplete")
        result = materialize_full_val_campaign(
            **base,
            prior_campaign_path=args.prior_campaign.resolve(),
            prior_campaign_file_sha256=args.prior_campaign_file_sha256,
            selection_path=args.gpu_selection.resolve(),
            selection_file_sha256=args.gpu_selection_file_sha256,
            final_authority_path=args.final_train_checkpoint_authority.resolve(),
            final_authority_file_sha256=(
                args.final_train_checkpoint_authority_file_sha256
            ),
            window_count=args.full_val_window_count,
            max_forwards_per_window=args.max_forwards_per_window,
            progress_interval_forwards=args.progress_interval_forwards,
            max_model_forwards=args.compute_guard_max_model_forwards,
            max_materialized_state_views=(
                args.compute_guard_max_materialized_state_views
            ),
            max_wall_seconds=args.compute_guard_max_wall_seconds,
        )
    print(
        json.dumps(
            {"ok": True, "path": result["path"], "sha256": result["sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
