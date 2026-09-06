#!/usr/bin/env python3
"""Deterministic, process-bound resume equivalence probe for candidate state.

The default mode is a tiny FP32 checkpoint integration probe. It exercises
the exact two-slot candidate session, optimizer, scheduler, RNG restoration and
the persisted global optimizer-step counter without training on market data.
It never reads a dataset, makes a prediction, or opens TEST.

The explicitly bound --prepare-guard-recovery mode instead transfers a real
first-epoch checkpoint into a new standard session after a guard-only repair.
It rehashes the declared TRAIN/VAL inputs and proves exact learning-state
preservation on CPU; it never starts CUDA, opens TEST or relaxes a launch gate.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import re
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


SCHEMA_VERSION = "gx1_candidate_checkpoint_resume_equivalence_v1"
_STEPS = 8
_HALF = _STEPS // 2


def _state_component_sha256(value: Any) -> str:
    """Exact typed CPU-state identity, including RNG and optimizer tensors."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.layout != torch.strided:
                raise RuntimeError("[GUARD_RECOVERY_STATE_TENSOR_INVALID]")
            digest.update(str(("tensor", str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        elif item is None or type(item) in (str, int, float, bool):
            digest.update(type(item).__name__.encode() + b":")
            raw = repr(item).encode("utf-8")
            digest.update(str(len(raw)).encode() + b":" + raw)
        else:
            raise RuntimeError(f"[GUARD_RECOVERY_STATE_TYPE_INVALID] {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def _require_guard_only_recipe_transition(
    original: Mapping[str, Any], successor: Mapping[str, Any]
) -> None:
    """No learning, data, geometry, run-ID or scope change is transferable."""
    changed_metadata = {
        "created_utc", "out_bundle_dir", "source_commit",
        "source_bindings", "source_bindings_sha256",
    }
    if (
        set(original) != set(successor)
        or {k: v for k, v in original.items() if k not in changed_metadata}
        != {k: v for k, v in successor.items() if k not in changed_metadata}
        or original.get("profile") != "candidate"
        or original.get("out_bundle_dir") == successor.get("out_bundle_dir")
    ):
        raise RuntimeError("[GUARD_RECOVERY_LEARNING_RECIPE_CHANGED]")
    old_bindings = original["source_bindings"]
    new_bindings = successor["source_bindings"]
    if set(old_bindings) != set(new_bindings):
        raise RuntimeError("[GUARD_RECOVERY_SOURCE_CLOSURE_CHANGED]")
    changed = set()
    for name, old in old_bindings.items():
        new = new_bindings[name]
        if old["path"] != new["path"]:
            raise RuntimeError("[GUARD_RECOVERY_SOURCE_PATH_CHANGED]")
        if (old["sha256"], old["size_bytes"]) != (new["sha256"], new["size_bytes"]):
            changed.add(name)
    if changed != {"trainer_safety_guard"}:
        raise RuntimeError(f"[GUARD_RECOVERY_NON_GUARD_SOURCE_CHANGED] {sorted(changed)}")


def _guard_recovery_timing(
    *, guard_log: Path, trainer_log: Path, pointer: Mapping[str, Any], session_dir: Path
) -> dict[str, Any]:
    """Qualify the saved update, never invent telemetry after guard exit.

    The unchanged trainer logs step_done AFTER optimizer + EMA update, then
    synchronously saves state before entering another batch. Serialization can
    finish after guard exit without introducing an unguarded optimizer update.
    This does not establish host telemetry in the unobserved interval.
    """
    guard_lines = guard_log.read_text(encoding="utf-8").splitlines()
    stop_lines = [line for line in guard_lines if " event=stop " in line]
    if len(stop_lines) != 1 or " reason=guard_exit " not in stop_lines[0]:
        raise RuntimeError("[GUARD_RECOVERY_INCIDENT_STOP_INVALID]")
    stop = datetime.fromisoformat(stop_lines[0].split()[0].replace("Z", "+00:00"))
    rows = trainer_log.read_text(encoding="utf-8").splitlines()
    step_rows = [line for line in rows if re.search(r"\[TRAIN_STEP\] batch=\d+ step_done$", line)]
    checkpoint_rows = [line for line in rows if "[CANDIDATE_TRAINING_CHECKPOINT]" in line]
    if not step_rows or not checkpoint_rows:
        raise RuntimeError("[GUARD_RECOVERY_INCIDENT_PROGRESS_MISSING]")
    expected_steps = int(pointer["global_optimizer_steps"])
    last_step = step_rows[-1]
    step_time = datetime.strptime(last_step[:23], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)
    fields = dict(re.findall(r"(\w+)=([^\s]+)", checkpoint_rows[-1]))
    expected = {
        "directory": str(session_dir), "checkpoint_index": str(pointer["checkpoint_index"]),
        "phase": "train", "epoch_index": "0", "next_batch_offset": str(expected_steps),
        "global_optimizer_steps": str(expected_steps), "complete": "0",
    }
    if (
        fields != expected
        or not last_step.endswith(f"batch={expected_steps} step_done")
        or not step_time < stop
        or rows.index(last_step) >= rows.index(checkpoint_rows[-1])
        or any("[TRAIN_STEP]" in row for row in rows[rows.index(checkpoint_rows[-1]) + 1:])
    ):
        raise RuntimeError("[GUARD_RECOVERY_UNGUARDED_OR_AMBIGUOUS_UPDATE]")
    return {
        "last_saved_optimizer_update_utc": step_time.isoformat(),
        "guard_exit_utc": stop.isoformat(),
        "checkpoint_log_utc": datetime.strptime(checkpoint_rows[-1][:23], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc).isoformat(),
        "saved_update_precedes_guard_exit": True,
        "telemetry_after_guard_exit_proven": False,
        "checkpoint_serialization_is_not_an_optimizer_update": True,
    }


def _guard_recovery_session_contract(
    recipe: Mapping[str, Any], provenance: Mapping[str, Any], normalization_sha256: str
) -> dict[str, Any]:
    """Use the trainer's real contract builder, not a second recipe mapping."""
    cli = recipe["trainer_cli"]
    same_names = (
        "seed", "batch_size", "epochs", "grad_accum_steps", "grad_clip_norm",
        "weight_decay", "dropout", "minimum_epochs_before_stop", "save_top_k",
        "seq_len", "multi_tf_num_layers", "specialist_num_layers",
        "multi_tf_scale", "specialist_fusion_scale", "cross_family_fusion_scale",
        "execution_tier",
    )
    artifacts = recipe["artifact_bindings"]
    return trainer._candidate_training_session_contract(
        **{name: cli[name] for name in same_names},
        out_bundle_dir=Path(recipe["out_bundle_dir"]), run_id=recipe["run_id"],
        dataset_run_id=recipe["dataset_run_id"],
        train_parquet=Path(artifacts["train_parquet"]["path"]),
        val_parquet=Path(artifacts["val_parquet"]["path"]),
        m5_prebuilt_path=Path(artifacts["m5_prebuilt"]["path"]),
        lifecycle_manifest_path=Path(artifacts["unified_exit_lifecycle_manifest"]["path"]),
        input_normalization={"contract_sha256": normalization_sha256},
        lr=cli["learning_rate"], early_stopping_patience=cli["early_stop_patience"],
        early_stopping_min_delta=cli["early_stop_min_delta"],
        per_tf_seq_lens={name: cli[f"per_tf_seq_len_{name.lower()}"] for name in trainer.MULTI_TF_TIMEFRAMES},
        device_type=cli["device"], recipe_source_provenance=provenance,
    )


def _require_finite_recovery_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise RuntimeError("[GUARD_RECOVERY_NONFINITE_LEARNING_STATE]")
    elif isinstance(value, Mapping):
        for child in value.values():
            _require_finite_recovery_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite_recovery_tensors(child)


def prepare_guard_recovery(args: argparse.Namespace) -> dict[str, Any]:
    """CPU-only, atomic state transfer into a new standard candidate session.

    Original session, recipes, checkpoints and logs are read-only. There is no
    runtime exception to source binding: the successor must pass the ordinary
    current-source recipe verifier and later obtain its own launch gate.
    """
    from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import require_pretest_technical_recipe_metadata
    from gx1.contracts.entry_model_native_train_launch_v1 import (
        require_training_recipe_source_provenance,
    )
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
    from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import artifact_binding

    repo = Path(__file__).resolve().parents[2]

    def bound(raw: Path, expected: str | None = None) -> dict[str, str]:
        result = artifact_binding(raw)
        if expected is not None and result["sha256"] != expected:
            raise RuntimeError("[GUARD_RECOVERY_INPUT_SHA_MISMATCH]")
        return result

    original_binding = bound(args.original_recipe_json, args.original_recipe_sha256)
    successor_binding = bound(args.successor_recipe_json, args.successor_recipe_sha256)
    original = require_pretest_technical_recipe_metadata(
        json.loads(args.original_recipe_json.read_text()), expected_profile="candidate"
    )
    successor = require_pretest_technical_recipe_metadata(
        json.loads(args.successor_recipe_json.read_text()), expected_profile="candidate"
    )
    _require_guard_only_recipe_transition(original, successor)
    provenance = require_training_recipe_source_provenance(
        recipe_audit_path=args.successor_recipe_json,
        recipe_audit_sha256=args.successor_recipe_sha256,
        repo=repo, profile="candidate", run_id=successor["run_id"],
        dataset_run_id=successor["dataset_run_id"],
        dataset_dir=Path(successor["dataset_dir"]),
        out_bundle_dir=Path(successor["out_bundle_dir"]),
    )
    if re.fullmatch(r"[0-9a-f]{40}", args.guard_repair_commit) is None:
        raise RuntimeError("[GUARD_RECOVERY_REPAIR_COMMIT_INVALID]")
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", args.guard_repair_commit, "HEAD"], check=True)
    for binding in original["source_bindings"].values():
        relative = Path(binding["path"]).relative_to(repo)
        frozen = subprocess.check_output(["git", "-C", str(repo), "show", f"{original['source_commit']}:{relative}"])
        if hashlib.sha256(frozen).hexdigest() != binding["sha256"]:
            raise RuntimeError("[GUARD_RECOVERY_ORIGINAL_SOURCE_UNPROVEN]")
    guard_binding = successor["source_bindings"]["trainer_safety_guard"]
    relative_guard = Path(guard_binding["path"]).relative_to(repo)
    repaired = subprocess.check_output(["git", "-C", str(repo), "show", f"{args.guard_repair_commit}:{relative_guard}"])
    if hashlib.sha256(repaired).hexdigest() != guard_binding["sha256"]:
        raise RuntimeError("[GUARD_RECOVERY_REPAIR_SOURCE_MISMATCH]")

    original_output = Path(original["out_bundle_dir"])
    original_dir = original_output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + original_output.name)
    contract_path = original_dir / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME
    pointer_path = original_dir / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    contract_binding = bound(contract_path)
    pointer_binding = bound(pointer_path, args.original_pointer_sha256)
    contract = trainer._candidate_training_session_read_json(contract_path, label="CONTRACT")
    pointer = trainer._candidate_training_session_read_json(pointer_path, label="ACTIVE")
    old_provenance = contract.get("recipe_source_provenance")
    if (
        not isinstance(old_provenance, Mapping)
        or old_provenance.get("recipe_audit_path") != original_binding["path"]
        or old_provenance.get("recipe_audit_sha256") != original_binding["sha256"]
        or old_provenance.get("source_bindings") != original["source_bindings"]
        or contract.get("run_id") != original["run_id"]
        or contract.get("out_bundle_dir") != str(original_output)
        or pointer.get("phase") != "train" or pointer.get("epoch_index") != 0
        or pointer.get("complete") is not False
        or pointer.get("global_optimizer_steps") != pointer.get("next_batch_offset")
        or int(pointer.get("global_optimizer_steps", 0)) <= 0
    ):
        raise RuntimeError("[GUARD_RECOVERY_ORIGINAL_SESSION_IDENTITY_INVALID]")
    old_session = trainer._CandidateTrainingSession(out_bundle_dir=original_output, contract=contract)
    state = old_session.load_checkpoint()
    if state is None or state["training_progress"]["checkpoint_selection"]["top_k_checkpoints"]:
        raise RuntimeError("[GUARD_RECOVERY_REQUIRES_FIRST_TRAIN_EPOCH]")
    expected_old_contract = _guard_recovery_session_contract(
        original, old_provenance, contract["input_normalization_sha256"]
    )
    if contract != expected_old_contract:
        raise RuntimeError("[GUARD_RECOVERY_ORIGINAL_CONTRACT_RECIPE_MISMATCH]")
    for key in ("model_state", "target_model_state", "optimizer_state", "weight_ema_state"):
        _require_finite_recovery_tensors(state[key])
    state_binding = bound(old_session._slot_path(int(pointer["slot"])), pointer["state_sha256"])
    guard_log_binding = bound(args.incident_guard_log)
    trainer_log_binding = bound(args.incident_trainer_log)
    timing = _guard_recovery_timing(
        guard_log=args.incident_guard_log, trainer_log=args.incident_trainer_log,
        pointer=pointer, session_dir=original_dir,
    )
    preserved = {key: _state_component_sha256(value) for key, value in state.items() if key != "session_contract_sha256"}
    successor_output = Path(successor["out_bundle_dir"])
    successor_dir = successor_output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + successor_output.name)
    if (
        not args.out_dir.is_absolute() or args.out_dir.resolve() != args.out_dir
        or args.out_dir.is_relative_to(original_dir)
        or args.out_dir.is_relative_to(successor_dir)
        or successor_output.is_relative_to(original_dir)
    ):
        raise RuntimeError("[GUARD_RECOVERY_OUTPUT_OVERLAPS_PRESERVED_SESSION]")
    if successor_dir.exists() or successor_dir.is_symlink() or successor_output.exists():
        raise RuntimeError("[GUARD_RECOVERY_SUCCESSOR_ALREADY_EXISTS]")
    successor_contract = _guard_recovery_session_contract(
        successor, provenance, contract["input_normalization_sha256"]
    )
    if successor_contract != {
        **contract, "out_bundle_dir": str(successor_output),
        "source_commit": provenance["source_commit"], "recipe_source_provenance": provenance,
    }:
        raise RuntimeError("[GUARD_RECOVERY_SUCCESSOR_CONTRACT_CHANGED]")
    # The canonical session owner writes all state; only immutable provenance
    # changes. Stage beside the final directory for no-replace publication.
    staging_name = f".guard-recovery-stage-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    staging_output = successor_output.with_name(staging_name)
    staging_dir = staging_output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + staging_output.name)
    if staging_dir.exists() or staging_dir.is_symlink():
        raise RuntimeError("[GUARD_RECOVERY_STAGING_ALREADY_EXISTS]")
    staged_session = trainer._CandidateTrainingSession(out_bundle_dir=staging_output, contract=successor_contract)
    staged_session.save_checkpoint({**state, "session_contract_sha256": staged_session.contract_sha256})
    restored = staged_session.load_checkpoint()
    if restored is None or {key: _state_component_sha256(value) for key, value in restored.items() if key != "session_contract_sha256"} != preserved:
        raise RuntimeError("[GUARD_RECOVERY_LEARNING_STATE_CHANGED]")
    staged_pointer = trainer._candidate_training_session_read_json(staged_session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME, label="ACTIVE")
    for binding in (original_binding, successor_binding, contract_binding, pointer_binding, state_binding, guard_log_binding, trainer_log_binding):
        bound(Path(binding["path"]), binding["sha256"])
    report = {
        "schema_version": "gx1_candidate_guard_recovery_v1",
        "decision": "PASS_EXACT_STATE_TRANSFER_NOT_CUDA_AUTHORITY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False, "activation_authority": False,
        "original_recipe": original_binding, "successor_recipe": successor_binding,
        "original_contract": contract_binding, "original_pointer": pointer_binding,
        "original_state": state_binding, "incident_guard_log": guard_log_binding,
        "incident_trainer_log": trainer_log_binding, "incident_timing": timing,
        "guard_repair_commit": args.guard_repair_commit,
        "source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "producer": bound(Path(__file__).resolve()),
        "changed_recipe_source_bindings": ["trainer_safety_guard"],
        "unchanged_state_component_sha256": preserved,
        "successor_session_dir": str(successor_dir),
        "successor_contract_sha256": staged_session.contract_sha256,
        "successor_initial_pointer": staged_pointer,
        "original_session_preserved": True,
        "cuda_started": False,
    }
    report_path, report = write_immutable_json_event(
        args.out_dir, "CANDIDATE_GUARD_RECOVERY", report
    )
    # Keep the immutable event outside the moving directory: its exact
    # self-reference remains valid after publication. The private session
    # retains an explicit hash-bound origin, never a mutable `latest` pointer.
    trainer._candidate_training_session_atomic_write_json(
        staged_session.directory / "CANDIDATE_GUARD_RECOVERY_ORIGIN.json",
        bound(report_path),
    )
    publish_bundle_directory_noreplace(staged_session.directory, successor_dir)
    reloaded = trainer._CandidateTrainingSession(
        out_bundle_dir=successor_output, contract=successor_contract
    ).load_checkpoint()
    if reloaded is None or {
        key: _state_component_sha256(value)
        for key, value in reloaded.items() if key != "session_contract_sha256"
    } != preserved:
        raise RuntimeError("[GUARD_RECOVERY_PUBLISHED_STATE_CHANGED]")
    return report


def _contract() -> dict[str, Any]:
    return {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "authority": {
            "candidate_training": True,
            "bundle": False,
            "validation": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "purpose": "deterministic_checkpoint_resume_probe_only",
    }


def _reset_rng() -> None:
    random.seed(9137)
    np.random.seed(9137)
    torch.manual_seed(9137)


def _new_components() -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    model = torch.nn.Linear(3, 2)
    target = copy.deepcopy(model)
    target.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_STEPS, eta_min=0.0
    )
    return model, target, optimizer, scheduler


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    # RNG-generated inputs make restoration of Python/NumPy/Torch state
    # meaningful.  CPU FP32 has an exact <=1e-6 acceptance threshold.
    x = torch.randn(7, 3)
    y = torch.from_numpy(np.random.standard_normal((7, 2)).astype(np.float32))
    random_scale = 0.75 + random.random() * 0.5
    optimizer.zero_grad(set_to_none=True)
    loss = ((model(x) - y).square().mean()) * random_scale
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach().cpu())


def _state(
    session: trainer._CandidateTrainingSession,
    model: torch.nn.Module,
    target: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    return {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "session_contract_sha256": session.contract_sha256,
        "checkpoint_index": 1,
        "phase": "train",
        "epoch_index": 0,
        "next_batch_offset": _HALF,
        "global_optimizer_steps": _HALF,
        "epoch_order": torch.arange(_STEPS, dtype=torch.int64),
        "model_state": model.state_dict(),
        "target_model_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "weight_ema_state": None,
        "lr_scheduler_state": scheduler.state_dict(),
        "rng_state": trainer._attended_session_rng_state(device=torch.device("cpu")),
        "training_progress": trainer._new_candidate_training_progress(),
        "complete": False,
    }


def _tensor_state_max_abs(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    if set(a) != set(b):
        raise RuntimeError("[RESUME_EQUIVALENCE_STATE_KEYS_MISMATCH]")
    return max(
        float((a[name].detach().cpu() - b[name].detach().cpu()).abs().max().item())
        for name in a
    )


def _optimizer_state_max_abs(a: Any, b: Any) -> float:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_SHAPE_MISMATCH]")
        return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_KEYS_MISMATCH]")
        return max((_optimizer_state_max_abs(a[key], b[key]) for key in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_LENGTH_MISMATCH]")
        return max((_optimizer_state_max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    if a != b:
        raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_VALUE_MISMATCH]")
    return 0.0


def _child_resume(out_bundle: Path) -> dict[str, Any]:
    session = trainer._CandidateTrainingSession(out_bundle_dir=out_bundle, contract=_contract())
    model, target, optimizer, scheduler = _new_components()
    state = session.load_checkpoint()
    if state is None:
        raise RuntimeError("[RESUME_EQUIVALENCE_CHECKPOINT_MISSING]")
    restored = trainer._restore_candidate_training_checkpoint(
        state,
        session=session,
        model=model,
        target_model=target,
        optimizer=optimizer,
        weight_ema=None,
        lr_scheduler=scheduler,
        device=torch.device("cpu"),
        dataset_rows=_STEPS,
    )
    if restored["global_optimizer_steps"] != _HALF:
        raise RuntimeError("[RESUME_EQUIVALENCE_GLOBAL_STEP_RESTORE_INVALID]")
    losses = [_step(model, optimizer, scheduler) for _ in range(_HALF, _STEPS)]
    reference = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(4, 3)
    return {
        "global_optimizer_steps": _STEPS,
        "losses": losses,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "prediction": model(reference).detach().cpu(),
    }


def run_equivalence() -> dict[str, Any]:
    _reset_rng()
    reference_model, _, reference_optimizer, reference_scheduler = _new_components()
    reference_losses = [
        _step(reference_model, reference_optimizer, reference_scheduler)
        for _ in range(_STEPS)
    ]
    reference_input = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(4, 3)
    reference_prediction = reference_model(reference_input).detach().cpu()

    with tempfile.TemporaryDirectory(prefix="gx1-candidate-resume-") as temporary:
        root = Path(temporary)
        out_bundle = root / "candidate_bundle_never_published"
        _reset_rng()
        model, target, optimizer, scheduler = _new_components()
        first_losses = [_step(model, optimizer, scheduler) for _ in range(_HALF)]
        session = trainer._CandidateTrainingSession(
            out_bundle_dir=out_bundle, contract=_contract()
        )
        session.save_checkpoint(_state(session, model, target, optimizer, scheduler))
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "gx1.scripts.verify_candidate_checkpoint_resume_v1",
                "--resume-child",
                "--out-bundle",
                str(out_bundle),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        if child.returncode != 0:
            raise RuntimeError(
                "[RESUME_EQUIVALENCE_CHILD_FAILED] " + child.stderr.strip()
            )
        resumed = torch.load(
            root / "child_result.pt", map_location="cpu", weights_only=True
        )
        # The child writes to the deterministic session parent so the parent
        # can compare exact state without entrusting hidden process memory.

    model_diff = _tensor_state_max_abs(
        reference_model.state_dict(), resumed["model_state"]
    )
    optimizer_diff = _optimizer_state_max_abs(
        reference_optimizer.state_dict(), resumed["optimizer_state"]
    )
    prediction_diff = float((reference_prediction - resumed["prediction"]).abs().max().item())
    loss_diff = max(
        abs(expected - observed)
        for expected, observed in zip(reference_losses[_HALF:], resumed["losses"], strict=True)
    )
    if (
        model_diff > 1e-6
        or optimizer_diff > 1e-6
        or prediction_diff > 1e-6
        or loss_diff > 1e-6
        or int(resumed["global_optimizer_steps"]) != _STEPS
        or int(resumed["scheduler_last_epoch"]) != int(reference_scheduler.last_epoch)
        or float(resumed["lr"]) != float(reference_optimizer.param_groups[0]["lr"])
    ):
        raise RuntimeError("[RESUME_EQUIVALENCE_FP32_TOLERANCE_EXCEEDED]")
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "test_accessed": False,
        "precision": "fp32",
        "steps_continuous": _STEPS,
        "steps_before_resume": _HALF,
        "steps_after_resume": _HALF,
        "global_optimizer_steps": _STEPS,
        "scheduler_last_epoch": int(reference_scheduler.last_epoch),
        "learning_rate": float(reference_optimizer.param_groups[0]["lr"]),
        "max_abs_model_weight_difference": model_diff,
        "max_abs_optimizer_state_difference": optimizer_diff,
        "max_abs_prediction_difference": prediction_diff,
        "max_abs_loss_difference": loss_diff,
        "tolerance": 1e-6,
        "amp_grad_scaler": "not_applicable_candidate_precision_is_deterministic_fp32",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-child", action="store_true")
    parser.add_argument("--prepare-guard-recovery", action="store_true")
    parser.add_argument("--out-bundle", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--original-recipe-json", type=Path)
    parser.add_argument("--original-recipe-sha256")
    parser.add_argument("--successor-recipe-json", type=Path)
    parser.add_argument("--successor-recipe-sha256")
    parser.add_argument("--original-pointer-sha256")
    parser.add_argument("--incident-guard-log", type=Path)
    parser.add_argument("--incident-trainer-log", type=Path)
    parser.add_argument("--guard-repair-commit")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    recovery_arguments = (
        args.original_recipe_json, args.original_recipe_sha256,
        args.successor_recipe_json, args.successor_recipe_sha256,
        args.original_pointer_sha256, args.incident_guard_log,
        args.incident_trainer_log, args.guard_repair_commit, args.out_dir,
    )
    if args.prepare_guard_recovery:
        if not all(value is not None for value in recovery_arguments) or (
            args.resume_child or args.out_bundle is not None or args.out_json is not None
        ):
            parser.error("guard recovery requires all explicit recovery bindings and no probe arguments")
        try:
            report = prepare_guard_recovery(args)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"FATAL: guard recovery failed; no CUDA authority: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, sort_keys=True))
        return 0
    if any(value is not None for value in recovery_arguments):
        parser.error("recovery arguments require --prepare-guard-recovery")
    if args.resume_child:
        if args.out_bundle is None:
            parser.error("--resume-child requires --out-bundle")
        result = _child_resume(args.out_bundle.resolve())
        target = args.out_bundle.resolve().parent / "child_result.pt"
        torch.save(result, target)
        return 0
    if args.out_json is None:
        parser.error("--out-json is required for the parent equivalence probe")
    if args.out_bundle is not None:
        parser.error("--out-bundle is child-only; the parent probe uses a temporary session")
    try:
        report = run_equivalence()
    except RuntimeError as exc:
        print(f"FATAL: candidate resume equivalence failed: {exc}", file=sys.stderr)
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
