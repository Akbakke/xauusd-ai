#!/usr/bin/env python3
"""Deterministic, process-bound resume equivalence probe for candidate state.

The default mode is a tiny FP32 checkpoint integration probe. It exercises
the exact two-slot candidate session, optimizer, scheduler, RNG restoration and
the persisted global optimizer-step counter without training on market data.
It never reads a dataset, makes a prediction, or opens TEST.

The explicitly bound --prepare-guard-recovery mode instead transfers a real
first-epoch checkpoint into a new standard session after a guard-only repair.
The separate --prepare-source-state-successor mode admits only a new source/run
identity and removes the exact 36 stateless, retired Exit parameters. Both
rehash declared TRAIN/VAL inputs and preserve learning state on CPU; neither
starts CUDA, opens TEST or relaxes a launch gate.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
import re
import random
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


SCHEMA_VERSION = "gx1_candidate_checkpoint_resume_equivalence_v1"
_STEPS = 8
_HALF = _STEPS // 2
_SOURCE_SUCCESSOR_OLD_GROUP_IDS = (
    tuple(range(748)),
    tuple(range(748, 758)),
)
_SOURCE_SUCCESSOR_RETIRED_PARAMETER_IDS = frozenset(
    (*range(579, 603), *range(604, 616))
)
_SOURCE_SUCCESSOR_NEW_GROUP_SIZES = (712, 10)
_SOURCE_SUCCESSOR_EXPECTED_CHANGED_ROLES = frozenset(
    {
        "capped_runner",
        "control_surface",
        "python:gx1/contracts/entry_causal_m1_target_policy_v1.py",
        "python:gx1/contracts/entry_model_native_readiness_v1.py",
        "python:gx1/contracts/entry_model_native_train_launch_v1.py",
        "python:gx1/contracts/evidence_retention_v1.py",
        "python:gx1/contracts/gx1_capped_execution_v1.py",
        "python:gx1/contracts/immutable_event_authority_v1.py",
        "python:gx1/contracts/unified_exit_lifecycle_v1.py",
        "python:gx1/contracts/xau_tape_provenance_v1.py",
        "python:gx1/features/basic_v1.py",
        "python:gx1/features/entry_specialist_feature_groups_v1.py",
        "python:gx1/features/model_native_market_context_v1.py",
        "python:gx1/models/entry_v10/direction_decision_contract.py",
        "python:gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
        "python:gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
        "python:gx1/scripts/augment_forward_outcome_v2.py",
    }
)
_SOURCE_SUCCESSOR_EXPECTED_ADDED_ROLES = frozenset(
    {
        "python:gx1/contracts/entry_exit_feature_usefulness_v1.py",
        "python:gx1/scripts/audit_entry_exit_feature_usefulness_v1.py",
        "python:gx1/scripts/entry_exit_feature_usefulness_native_v1.py",
        "python:gx1/scripts/evaluate_entry_candidate_selective_edge_v1.py",
    }
)


def _source_successor_retired_state_keys() -> frozenset[str]:
    encoder_suffixes = (
        "self_attn.in_proj_weight",
        "self_attn.in_proj_bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "linear1.weight",
        "linear1.bias",
        "linear2.weight",
        "linear2.bias",
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
    )
    return frozenset(
        [
            f"exit_path_encoder.layers.{layer}.{suffix}"
            for layer in range(2)
            for suffix in encoder_suffixes
        ]
        + [
            "exit_entry_query_norm.weight",
            "exit_entry_query_norm.bias",
            "exit_entry_path_attention.in_proj_weight",
            "exit_entry_path_attention.in_proj_bias",
            "exit_entry_path_attention.out_proj.weight",
            "exit_entry_path_attention.out_proj.bias",
            "exit_fuse.0.weight",
            "exit_fuse.0.bias",
            "exit_fuse.1.weight",
            "exit_fuse.1.bias",
            "exit_fuse.4.weight",
            "exit_fuse.4.bias",
        ]
    )


_SOURCE_SUCCESSOR_RETIRED_STATE_KEYS = _source_successor_retired_state_keys()


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


def _require_source_state_successor_recipe_transition(
    original: Mapping[str, Any], successor: Mapping[str, Any]
) -> None:
    """Admit only a new run/source identity over unchanged learning inputs."""

    changed_metadata = {
        "created_utc",
        "out_bundle_dir",
        "run_id",
        "source_commit",
        "source_bindings",
        "source_bindings_sha256",
    }
    if (
        set(original) != set(successor)
        or {key: value for key, value in original.items() if key not in changed_metadata}
        != {key: value for key, value in successor.items() if key not in changed_metadata}
        or original.get("profile") != "candidate"
        or original.get("run_id") == successor.get("run_id")
        or original.get("out_bundle_dir") == successor.get("out_bundle_dir")
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_LEARNING_RECIPE_CHANGED]")
    old_bindings = original.get("source_bindings")
    new_bindings = successor.get("source_bindings")
    if (
        not isinstance(old_bindings, Mapping)
        or not isinstance(new_bindings, Mapping)
        or not set(old_bindings) < set(new_bindings)
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_SOURCE_CLOSURE_INVALID]")
    for name, old in old_bindings.items():
        new = new_bindings[name]
        if not isinstance(old, Mapping) or not isinstance(new, Mapping):
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_SOURCE_BINDING_INVALID]")
        if old.get("path") != new.get("path"):
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_SOURCE_PATH_CHANGED]")
    changed_roles = {
        name
        for name, old in old_bindings.items()
        if (
            old.get("sha256"),
            old.get("size_bytes"),
        )
        != (
            new_bindings[name].get("sha256"),
            new_bindings[name].get("size_bytes"),
        )
    }
    added_roles = set(new_bindings) - set(old_bindings)
    if (
        changed_roles != _SOURCE_SUCCESSOR_EXPECTED_CHANGED_ROLES
        or added_roles != _SOURCE_SUCCESSOR_EXPECTED_ADDED_ROLES
    ):
        raise RuntimeError(
            "[SOURCE_STATE_SUCCESSOR_SOURCE_DELTA_INVALID] "
            f"changed={sorted(changed_roles)} added={sorted(added_roles)}"
        )


def _migrate_source_state_successor_checkpoint(
    state: Mapping[str, Any],
    *,
    successor_contract_sha256: str,
) -> dict[str, Any]:
    """Remove only the retired static Exit state and remap AdamW IDs."""

    if (
        not isinstance(successor_contract_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", successor_contract_sha256) is None
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_CONTRACT_SHA256_INVALID]")
    expected_state_keys = {
        "schema_version",
        "session_contract_sha256",
        "checkpoint_index",
        "phase",
        "epoch_index",
        "next_batch_offset",
        "global_optimizer_steps",
        "epoch_order",
        "model_state",
        "target_model_state",
        "optimizer_state",
        "weight_ema_state",
        "lr_scheduler_state",
        "rng_state",
        "training_progress",
        "complete",
    }
    if not isinstance(state, Mapping) or set(state) != expected_state_keys:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_CHECKPOINT_SCHEMA_INVALID]")
    original_sha256 = _state_component_sha256(state)

    def filtered_model_state(value: Any, *, label: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or any(
            not isinstance(name, str) for name in value
        ):
            raise RuntimeError(f"[SOURCE_STATE_SUCCESSOR_{label}_INVALID]")
        removed = set(value) & _SOURCE_SUCCESSOR_RETIRED_STATE_KEYS
        if removed != _SOURCE_SUCCESSOR_RETIRED_STATE_KEYS:
            raise RuntimeError(f"[SOURCE_STATE_SUCCESSOR_{label}_RETIREMENT_INVALID]")
        return {
            name: copy.deepcopy(tensor)
            for name, tensor in value.items()
            if name not in _SOURCE_SUCCESSOR_RETIRED_STATE_KEYS
        }

    model_state = filtered_model_state(state["model_state"], label="MODEL_STATE")
    target_state = filtered_model_state(
        state["target_model_state"],
        label="TARGET_MODEL_STATE",
    )
    if set(model_state) != set(target_state):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_MODEL_TARGET_KEYS_MISMATCH]")
    side_embedding = model_state.get("exit_side_embedding.weight")
    if (
        not isinstance(side_embedding, torch.Tensor)
        or side_embedding.shape != (2, 128)
        or side_embedding.dtype != torch.float32
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_SIDE_EMBEDDING_INVALID]")

    optimizer = state["optimizer_state"]
    if not isinstance(optimizer, Mapping) or set(optimizer) != {
        "state",
        "param_groups",
    }:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_OPTIMIZER_SCHEMA_INVALID]")
    optimizer_states = optimizer["state"]
    parameter_groups = optimizer["param_groups"]
    if (
        not isinstance(optimizer_states, Mapping)
        or not isinstance(parameter_groups, list)
        or len(parameter_groups) != len(_SOURCE_SUCCESSOR_OLD_GROUP_IDS)
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_OPTIMIZER_LAYOUT_INVALID]")
    for group, expected_ids in zip(
        parameter_groups,
        _SOURCE_SUCCESSOR_OLD_GROUP_IDS,
        strict=True,
    ):
        if not isinstance(group, Mapping) or tuple(group.get("params", ())) != expected_ids:
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_OPTIMIZER_GROUP_INVALID]")
    active_old_ids = tuple(
        parameter_id
        for group_ids in _SOURCE_SUCCESSOR_OLD_GROUP_IDS
        for parameter_id in group_ids
        if parameter_id not in _SOURCE_SUCCESSOR_RETIRED_PARAMETER_IDS
    )
    if set(optimizer_states) != set(active_old_ids):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_OPTIMIZER_STATE_COVERAGE_INVALID]")
    id_mapping = {
        parameter_id: new_id
        for new_id, parameter_id in enumerate(active_old_ids)
    }
    if id_mapping.get(603) != 579:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_SIDE_EMBEDDING_MAPPING_INVALID]")
    for parameter_id, optimizer_state in optimizer_states.items():
        if not isinstance(optimizer_state, Mapping) or set(optimizer_state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ADAMW_STATE_INVALID]")
        step = optimizer_state["step"]
        exp_avg = optimizer_state["exp_avg"]
        exp_avg_sq = optimizer_state["exp_avg_sq"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or exp_avg.shape != exp_avg_sq.shape
            or exp_avg.dtype != exp_avg_sq.dtype
        ):
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ADAMW_TENSOR_INVALID]")
        if parameter_id == 603 and (
            exp_avg.shape != (2, 128) or exp_avg.dtype != torch.float32
        ):
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_SIDE_EMBEDDING_MOMENT_INVALID]")
        _require_finite_recovery_tensors(optimizer_state)
    migrated_optimizer_states = {
        id_mapping[parameter_id]: copy.deepcopy(optimizer_state)
        for parameter_id, optimizer_state in optimizer_states.items()
    }
    migrated_parameter_groups = []
    for group_index, group in enumerate(parameter_groups):
        migrated_group = copy.deepcopy(dict(group))
        migrated_group["params"] = list(
            range(
                0 if group_index == 0 else _SOURCE_SUCCESSOR_NEW_GROUP_SIZES[0],
                sum(_SOURCE_SUCCESSOR_NEW_GROUP_SIZES[: group_index + 1]),
            )
        )
        migrated_parameter_groups.append(migrated_group)

    ema_state = state["weight_ema_state"]
    if ema_state is None:
        migrated_ema_state = None
    else:
        if not isinstance(ema_state, Mapping) or set(ema_state) != {
            "decay",
            "steps",
            "shadow",
        }:
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_EMA_SCHEMA_INVALID]")
        migrated_shadow = filtered_model_state(
            ema_state["shadow"],
            label="EMA_SHADOW",
        )
        if set(migrated_shadow) != set(model_state):
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_EMA_KEYS_MISMATCH]")
        migrated_ema_state = {
            "decay": copy.deepcopy(ema_state["decay"]),
            "steps": copy.deepcopy(ema_state["steps"]),
            "shadow": migrated_shadow,
        }

    migrated = copy.deepcopy(dict(state))
    migrated["session_contract_sha256"] = successor_contract_sha256
    migrated["model_state"] = model_state
    migrated["target_model_state"] = target_state
    migrated["optimizer_state"] = {
        "state": migrated_optimizer_states,
        "param_groups": migrated_parameter_groups,
    }
    migrated["weight_ema_state"] = migrated_ema_state
    _require_finite_recovery_tensors(
        {
            "model_state": migrated["model_state"],
            "target_model_state": migrated["target_model_state"],
            "optimizer_state": migrated["optimizer_state"],
            "weight_ema_state": migrated["weight_ema_state"],
        }
    )
    if _state_component_sha256(state) != original_sha256:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ORIGINAL_STATE_MUTATED]")
    return migrated


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


def prepare_source_state_successor(args: argparse.Namespace) -> dict[str, Any]:
    """CPU-only migration to the reviewed source/state successor contract."""

    from gx1.contracts.entry_model_native_bundle_commit_v1 import (
        publish_bundle_directory_noreplace,
    )
    from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
        require_pretest_technical_recipe_metadata,
    )
    from gx1.contracts.entry_model_native_train_launch_v1 import (
        require_training_recipe_source_provenance,
    )
    from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import artifact_binding
    from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event

    repo = Path(__file__).resolve().parents[2]

    def bound(raw: Path, expected: str | None = None) -> dict[str, str]:
        result = artifact_binding(raw)
        if expected is not None and result["sha256"] != expected:
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_INPUT_SHA_MISMATCH]")
        return result

    original_binding = bound(
        args.original_recipe_json,
        args.original_recipe_sha256,
    )
    successor_binding = bound(
        args.successor_recipe_json,
        args.successor_recipe_sha256,
    )
    original = require_pretest_technical_recipe_metadata(
        json.loads(args.original_recipe_json.read_text()),
        expected_profile="candidate",
    )
    successor = require_pretest_technical_recipe_metadata(
        json.loads(args.successor_recipe_json.read_text()),
        expected_profile="candidate",
    )
    _require_source_state_successor_recipe_transition(original, successor)
    provenance = require_training_recipe_source_provenance(
        recipe_audit_path=args.successor_recipe_json,
        recipe_audit_sha256=args.successor_recipe_sha256,
        repo=repo,
        profile="candidate",
        run_id=successor["run_id"],
        dataset_run_id=successor["dataset_run_id"],
        dataset_dir=Path(successor["dataset_dir"]),
        out_bundle_dir=Path(successor["out_bundle_dir"]),
    )
    for binding in original["source_bindings"].values():
        relative = Path(binding["path"]).relative_to(repo)
        frozen = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{original['source_commit']}:{relative}"]
        )
        if (
            hashlib.sha256(frozen).hexdigest() != binding["sha256"]
            or len(frozen) != binding["size_bytes"]
        ):
            raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ORIGINAL_SOURCE_UNPROVEN]")

    original_output = Path(original["out_bundle_dir"])
    original_dir = original_output.parent / (
        trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + original_output.name
    )
    contract_path = original_dir / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME
    pointer_path = original_dir / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    contract_binding = bound(contract_path)
    pointer_binding = bound(pointer_path, args.original_pointer_sha256)
    contract = trainer._candidate_training_session_read_json(
        contract_path,
        label="CONTRACT",
    )
    pointer = trainer._candidate_training_session_read_json(
        pointer_path,
        label="ACTIVE",
    )
    old_provenance = contract.get("recipe_source_provenance")
    if (
        not isinstance(old_provenance, Mapping)
        or old_provenance.get("recipe_audit_path") != original_binding["path"]
        or old_provenance.get("recipe_audit_sha256") != original_binding["sha256"]
        or old_provenance.get("source_bindings") != original["source_bindings"]
        or contract.get("run_id") != original["run_id"]
        or contract.get("out_bundle_dir") != str(original_output)
        or pointer.get("phase") != "train"
        or pointer.get("epoch_index") != 0
        or pointer.get("complete") is not False
        or pointer.get("global_optimizer_steps") != pointer.get("next_batch_offset")
        or int(pointer.get("global_optimizer_steps", 0)) <= 0
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ORIGINAL_SESSION_INVALID]")
    original_session = trainer._CandidateTrainingSession(
        out_bundle_dir=original_output,
        contract=contract,
    )
    state = original_session.load_checkpoint()
    if (
        state is None
        or state["training_progress"]["checkpoint_selection"]["top_k_checkpoints"]
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_REQUIRES_FIRST_TRAIN_EPOCH]")
    expected_old_contract = _guard_recovery_session_contract(
        original,
        old_provenance,
        contract["input_normalization_sha256"],
    )
    if contract != expected_old_contract:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ORIGINAL_CONTRACT_MISMATCH]")
    for key in (
        "model_state",
        "target_model_state",
        "optimizer_state",
        "weight_ema_state",
    ):
        _require_finite_recovery_tensors(state[key])
    state_binding = bound(
        original_session._slot_path(int(pointer["slot"])),
        pointer["state_sha256"],
    )
    original_directory_stat = {
        path.name: (
            path.stat(follow_symlinks=False).st_dev,
            path.stat(follow_symlinks=False).st_ino,
            path.stat(follow_symlinks=False).st_mode,
            path.stat(follow_symlinks=False).st_size,
            path.stat(follow_symlinks=False).st_mtime_ns,
            path.stat(follow_symlinks=False).st_ctime_ns,
        )
        for path in original_dir.iterdir()
    }
    original_state_sha256 = _state_component_sha256(state)

    successor_output = Path(successor["out_bundle_dir"])
    successor_dir = successor_output.parent / (
        trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + successor_output.name
    )
    if (
        not args.out_dir.is_absolute()
        or args.out_dir.resolve() != args.out_dir
        or args.out_dir.is_relative_to(original_dir)
        or args.out_dir.is_relative_to(successor_dir)
        or successor_output.is_relative_to(original_dir)
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_OUTPUT_OVERLAPS_SOURCE]")
    if (
        successor_dir.exists()
        or successor_dir.is_symlink()
        or successor_output.exists()
    ):
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ALREADY_EXISTS]")
    successor_contract = _guard_recovery_session_contract(
        successor,
        provenance,
        contract["input_normalization_sha256"],
    )
    expected_successor_contract = {
        **contract,
        "run_id": successor["run_id"],
        "out_bundle_dir": str(successor_output),
        "source_commit": provenance["source_commit"],
        "recipe_source_provenance": provenance,
    }
    if successor_contract != expected_successor_contract:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_CONTRACT_CHANGED]")

    staging_name = (
        ".source-state-successor-stage-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    staging_output = successor_output.with_name(staging_name)
    staging_dir = staging_output.parent / (
        trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + staging_output.name
    )
    if staging_dir.exists() or staging_dir.is_symlink():
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_STAGING_ALREADY_EXISTS]")
    staged_session = trainer._CandidateTrainingSession(
        out_bundle_dir=staging_output,
        contract=successor_contract,
    )
    migrated = _migrate_source_state_successor_checkpoint(
        state,
        successor_contract_sha256=staged_session.contract_sha256,
    )
    migrated_state_sha256 = _state_component_sha256(migrated)
    staged_session.save_checkpoint(migrated)
    restored = staged_session.load_checkpoint()
    if restored is None or _state_component_sha256(restored) != migrated_state_sha256:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_STAGED_STATE_CHANGED]")
    staged_pointer = trainer._candidate_training_session_read_json(
        staged_session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME,
        label="ACTIVE",
    )
    for binding in (
        original_binding,
        successor_binding,
        contract_binding,
        pointer_binding,
        state_binding,
    ):
        bound(Path(binding["path"]), binding["sha256"])
    if {
        path.name: (
            path.stat(follow_symlinks=False).st_dev,
            path.stat(follow_symlinks=False).st_ino,
            path.stat(follow_symlinks=False).st_mode,
            path.stat(follow_symlinks=False).st_size,
            path.stat(follow_symlinks=False).st_mtime_ns,
            path.stat(follow_symlinks=False).st_ctime_ns,
        )
        for path in original_dir.iterdir()
    } != original_directory_stat:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_ORIGINAL_SESSION_CHANGED]")

    changed_roles = sorted(
        role
        for role, binding in successor["source_bindings"].items()
        if role in original["source_bindings"]
        and binding != original["source_bindings"][role]
    )
    added_roles = sorted(
        set(successor["source_bindings"]) - set(original["source_bindings"])
    )
    report = {
        "schema_version": "gx1_candidate_source_state_successor_v1",
        "decision": "PASS_STRUCTURAL_STATE_SUCCESSOR_NOT_CUDA_AUTHORITY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "activation_authority": False,
        "actual_next_batch_equivalence_authority": False,
        "original_recipe": original_binding,
        "successor_recipe": successor_binding,
        "original_contract": contract_binding,
        "original_pointer": pointer_binding,
        "original_state": state_binding,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "producer": bound(Path(__file__).resolve()),
        "changed_source_binding_roles": changed_roles,
        "added_source_binding_roles": added_roles,
        "retired_model_state_keys": sorted(
            _SOURCE_SUCCESSOR_RETIRED_STATE_KEYS
        ),
        "retired_optimizer_parameter_ids": sorted(
            _SOURCE_SUCCESSOR_RETIRED_PARAMETER_IDS
        ),
        "optimizer_parameter_id_mapping": {
            str(parameter_id): new_id
            for new_id, parameter_id in enumerate(
                parameter_id
                for group in _SOURCE_SUCCESSOR_OLD_GROUP_IDS
                for parameter_id in group
                if parameter_id not in _SOURCE_SUCCESSOR_RETIRED_PARAMETER_IDS
            )
        },
        "exit_side_embedding_mapping": {"old_id": 603, "new_id": 579},
        "original_state_component_sha256": original_state_sha256,
        "migrated_state_component_sha256": migrated_state_sha256,
        "successor_session_dir": str(successor_dir),
        "successor_contract_sha256": staged_session.contract_sha256,
        "successor_initial_pointer": staged_pointer,
        "original_session_preserved": True,
        "cuda_started": False,
    }
    report_path, report = write_immutable_json_event(
        args.out_dir,
        "CANDIDATE_SOURCE_STATE_SUCCESSOR",
        report,
    )
    trainer._candidate_training_session_atomic_write_json(
        staged_session.directory / "CANDIDATE_SOURCE_STATE_SUCCESSOR_ORIGIN.json",
        bound(report_path),
    )
    publish_bundle_directory_noreplace(staged_session.directory, successor_dir)
    reloaded = trainer._CandidateTrainingSession(
        out_bundle_dir=successor_output,
        contract=successor_contract,
    ).load_checkpoint()
    if reloaded is None or _state_component_sha256(reloaded) != migrated_state_sha256:
        raise RuntimeError("[SOURCE_STATE_SUCCESSOR_PUBLISHED_STATE_CHANGED]")
    return report


_ACTUAL_NEXT_BATCH_CHILD_SCHEMA = (
    "gx1_candidate_source_state_next_batch_child_v1"
)
_ACTUAL_NEXT_BATCH_REPORT_SCHEMA = (
    "gx1_candidate_source_state_next_batch_equivalence_v1"
)
_TF_INPUT_SCALE_PREFIX = "tf_input_scale_"


class _ActualNextBatchCaptureComplete(RuntimeError):
    """Internal stop after one read-only production-equivalent CPU step."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_regular_file(
    path: Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = Path(path)
    if (
        not resolved.is_absolute()
        or resolved.is_symlink()
        or not resolved.is_file()
        or resolved.resolve() != resolved
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_FILE_INVALID]")
    observed = _file_sha256(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_FILE_SHA_MISMATCH]")
    return {
        "path": str(resolved),
        "sha256": observed,
        "size_bytes": resolved.stat().st_size,
    }


def _clone_cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [_clone_cpu_tree(child) for child in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise RuntimeError(
        "[SOURCE_STATE_NEXT_BATCH_CAPTURE_TYPE_INVALID] "
        f"{type(value).__name__}"
    )


def _value_manifest(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "kind": "torch.Tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "numpy.ndarray",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _value_manifest(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_value_manifest(child) for child in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_value_manifest(child) for child in value]}
    if value is None or type(value) in (str, int, float, bool):
        return {"kind": type(value).__name__, "value": value}
    raise RuntimeError(
        "[SOURCE_STATE_NEXT_BATCH_MANIFEST_TYPE_INVALID] "
        f"{type(value).__name__}"
    )


def _session_directory_snapshot(path: Path) -> dict[str, tuple[Any, ...]]:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_SESSION_DIRECTORY_INVALID]")
    result: dict[str, tuple[Any, ...]] = {}
    for child in path.iterdir():
        stat_result = child.stat(follow_symlinks=False)
        result[child.name] = (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_mode,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )
    return result


def _candidate_recipe_run_arguments(
    *,
    target_trainer: Any,
    recipe: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = recipe["artifact_bindings"]
    cli = recipe["trainer_cli"]

    def artifact_path(name: str) -> Path:
        return Path(artifacts[name]["path"])

    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CONTRACT_MODE,
    )

    target_trainer._GRAD_CLIP_NORM = float(cli["grad_clip_norm"])
    target_trainer._WEIGHT_DECAY = float(cli["weight_decay"])
    return {
        "train_parquet": artifact_path("train_parquet"),
        "train_manifest_path": artifact_path("train_manifest"),
        "val_parquet": artifact_path("val_parquet"),
        "unified_exit_lifecycle_manifest_path": artifact_path(
            "unified_exit_lifecycle_manifest"
        ),
        "seq_len": int(cli["seq_len"]),
        "seed": int(cli["seed"]),
        "device": torch.device("cpu"),
        "batch_size": int(cli["batch_size"]),
        "epochs": int(cli["epochs"]),
        "lr": float(cli["learning_rate"]),
        "out_bundle_dir": Path(recipe["out_bundle_dir"]),
        "gx1_data_override": str(cli["gx1_data_root"]),
        "num_workers": int(cli["num_workers"]),
        "early_stopping_patience": int(cli["early_stop_patience"]),
        "early_stopping_min_delta": float(cli["early_stop_min_delta"]),
        "minimum_epochs_before_stop": int(cli["minimum_epochs_before_stop"]),
        "save_top_k": int(cli["save_top_k"]),
        "m5_prebuilt_path": artifact_path("m5_prebuilt"),
        "specialist_audit_json": artifact_path("specialist_audit"),
        "specialist_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "dropout": float(cli["dropout"]),
        "multi_tf_num_layers": int(cli["multi_tf_num_layers"]),
        "per_tf_seq_len_m5": int(cli["per_tf_seq_len_m5"]),
        "per_tf_seq_len_m15": int(cli["per_tf_seq_len_m15"]),
        "per_tf_seq_len_h1": int(cli["per_tf_seq_len_h1"]),
        "per_tf_seq_len_h4": int(cli["per_tf_seq_len_h4"]),
        "per_tf_seq_len_d1": int(cli["per_tf_seq_len_d1"]),
        "multi_tf_scale": float(cli["multi_tf_scale"]),
        "subsample_rows": int(cli["subsample_rows"]),
        "train_time_window_start_utc": None,
        "train_time_window_end_utc": None,
        "specialist_num_layers": int(cli["specialist_num_layers"]),
        "specialist_fusion_scale": float(cli["specialist_fusion_scale"]),
        "cross_family_fusion_scale": float(cli["cross_family_fusion_scale"]),
        "grad_accum_steps": int(cli["grad_accum_steps"]),
        "prefreeze_test_seal_lineage": recipe["test_guard_lineage"],
        "recipe_source_provenance": contract["recipe_source_provenance"],
        "run_id": str(recipe["run_id"]),
        "dataset_run_id": str(recipe["dataset_run_id"]),
        "profile": "candidate",
        "execution_tier": "canonical",
        "train_sequence_source_audit_json": artifact_path(
            "train_sequence_source_reconstruction"
        ),
        "val_sequence_source_audit_json": artifact_path(
            "val_sequence_source_reconstruction"
        ),
    }


def _install_candidate_recipe_environment(recipe: Mapping[str, Any]) -> None:
    artifacts = recipe["artifact_bindings"]
    exact = {
        "ENTRY_TRAIN_LR_COSINE_DECAY": "1",
        "ENTRY_TRAIN_WEIGHT_EMA_DECAY": "epoch",
        "GX1_CTX_CONTRACT": "V_NEXT",
        "GX1_V10_CKPT_MONITOR": "entry_policy_pnl",
        "GX1_ENTRY_TRAIN_MANIFEST_SHA256": artifacts["train_manifest"]["sha256"],
        "GX1_ENTRY_VAL_MANIFEST_SHA256": artifacts["val_manifest"]["sha256"],
        "GX1_ENTRY_TRAIN_PARQUET_SHA256": artifacts["train_parquet"]["sha256"],
        "GX1_ENTRY_VAL_PARQUET_SHA256": artifacts["val_parquet"]["sha256"],
        "GX1_ENTRY_M5_PREBUILT_SHA256": artifacts["m5_prebuilt"]["sha256"],
        "GX1_ENTRY_UNIFIED_EXIT_LIFECYCLE_MANIFEST_SHA256": artifacts[
            "unified_exit_lifecycle_manifest"
        ]["sha256"],
        "GX1_ENTRY_TRAIN_SEQUENCE_SOURCE_AUDIT_SHA256": artifacts[
            "train_sequence_source_reconstruction"
        ]["sha256"],
        "GX1_ENTRY_VAL_SEQUENCE_SOURCE_AUDIT_SHA256": artifacts[
            "val_sequence_source_reconstruction"
        ]["sha256"],
        "GX1_ENTRY_DATASET_RUN_ID": str(recipe["dataset_run_id"]),
        "GX1_V10_MULTI_TF_V4_CACHE_DIR": str(
            Path(artifacts["multi_tf_cache_manifest"]["path"]).parent
        ),
    }
    for name in tuple(os.environ):
        if name.startswith(("ENTRY_", "GX1_")):
            os.environ.pop(name)
    os.environ.update(exact)


def _optimizer_state_by_name(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    state = {
        parameter_names[id(parameter)]: _clone_cpu_tree(values)
        for parameter, values in optimizer.state.items()
    }
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        groups.append(
            {
                **{
                    key: _clone_cpu_tree(value)
                    for key, value in group.items()
                    if key != "params"
                },
                "parameter_names": [
                    parameter_names[id(parameter)] for parameter in group["params"]
                ],
            }
        )
    return state, groups


def _run_actual_next_batch_child(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve()
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_SOURCE_ROOT_INVALID]")
    recipe_binding = _bound_regular_file(
        args.recipe_json.resolve(), args.recipe_sha256
    )
    recipe = json.loads(Path(recipe_binding["path"]).read_text(encoding="utf-8"))
    _install_candidate_recipe_environment(recipe)
    for name in tuple(sys.modules):
        if name == "gx1" or name.startswith("gx1."):
            del sys.modules[name]
    sys.path.insert(0, str(source_root))
    target_trainer = importlib.import_module(
        "gx1.models.entry_v10.entry_v10_ctx_train_v3"
    )
    imported_root = Path(target_trainer.__file__).resolve().parents[3]
    if imported_root != source_root:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_IMPORTED_ROOT_MISMATCH]")

    output = Path(recipe["out_bundle_dir"])
    session_dir = output.parent / (
        target_trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name
    )
    contract_path = session_dir / target_trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME
    pointer_path = session_dir / target_trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    pointer_binding = _bound_regular_file(pointer_path, args.pointer_sha256)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    contract_binding = _bound_regular_file(
        contract_path, pointer["session_contract_sha256"]
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    state_path = session_dir / target_trainer._CANDIDATE_TRAINING_STATE_FILENAMES[
        int(pointer["slot"])
    ]
    state_binding = _bound_regular_file(state_path, pointer["state_sha256"])
    if (
        contract.get("run_id") != recipe.get("run_id")
        or contract.get("out_bundle_dir") != str(output)
        or contract.get("source_commit") != args.source_commit
        or pointer.get("phase") != "train"
        or pointer.get("epoch_index") != 0
        or pointer.get("complete") is not False
        or int(pointer.get("next_batch_offset", 0)) <= 0
        or pointer.get("global_optimizer_steps") != pointer.get("next_batch_offset")
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_SESSION_BINDING_INVALID]")
    before_snapshot = _session_directory_snapshot(session_dir)
    constructor_arguments: dict[str, Any] = {
        "out_bundle_dir": output,
        "contract": contract,
    }
    if "read_only" in inspect.signature(
        target_trainer._CandidateTrainingSession
    ).parameters:
        constructor_arguments["read_only"] = True
    session = target_trainer._CandidateTrainingSession(**constructor_arguments)
    state = session.load_checkpoint()
    if state is None:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_STATE_MISSING]")

    capture: dict[str, Any] = {}

    def intercepted_candidate_training(**call: Any) -> dict[str, Any]:
        model = call["model"]
        optimizer = call["optimizer"]
        weight_ema = call["weight_ema"]
        lr_scheduler = call["lr_scheduler"]
        device = call["device"]
        train_dataset = call["train_ds"]
        target_model = copy.deepcopy(model).to(device)
        restore_state = dict(state)
        restore_rng_state = dict(state["rng_state"])
        checkpoint_cuda_rng = restore_rng_state.pop("torch_cuda", None)
        if not isinstance(checkpoint_cuda_rng, list) or not checkpoint_cuda_rng:
            raise RuntimeError(
                "[SOURCE_STATE_NEXT_BATCH_CHECKPOINT_CUDA_RNG_MISSING]"
            )
        restore_state["rng_state"] = restore_rng_state
        restored = target_trainer._restore_candidate_training_checkpoint(
            restore_state,
            session=session,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            weight_ema=weight_ema,
            lr_scheduler=lr_scheduler,
            device=device,
            dataset_rows=len(train_dataset),
        )
        if (
            restored["phase"] != "train"
            or restored["epoch_index"] != 0
            or restored["complete"] is not False
        ):
            raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_RESTORE_PHASE_INVALID]")
        batch_size = int(call["batch_size"])
        batch_offset = int(restored["next_batch_offset"])
        sampler = target_trainer._ExactIndexSampler(
            restored["epoch_order"],
            batch_offset=batch_offset,
            batch_size=batch_size,
        )
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
            generator=torch.Generator().manual_seed(int(call["seed"])),
        )
        batch = next(iter(loader))

        class _OneBatchLoader:
            dataset = train_dataset

            def __len__(self) -> int:
                return 2

            def __iter__(self):
                yield batch

        entry_forwards: list[dict[str, torch.Tensor]] = []
        exit_capture: dict[str, Any] = {}
        joint_capture: dict[str, Any] = {}
        raw_gradients: dict[str, torch.Tensor] = {}
        clipped_gradients: dict[str, torch.Tensor] = {}
        original_forward = target_trainer._model_forward_fp32
        original_exit = target_trainer._train_unified_exit_full_population
        original_joint = target_trainer._joint_task_loss
        original_optimizer_owner = target_trainer._optimizer_step_with_finite_gradients

        def captured_forward(*forward_args: Any, **forward_kwargs: Any):
            result = original_forward(*forward_args, **forward_kwargs)
            if len(entry_forwards) < 2:
                entry_forwards.append(
                    {
                        name: value.detach().cpu().contiguous().clone()
                        for name, value in result.items()
                        if isinstance(value, torch.Tensor)
                    }
                )
            return result

        def captured_exit(*exit_args: Any, **exit_kwargs: Any):
            result = original_exit(*exit_args, **exit_kwargs)
            exit_capture.update(
                {
                    "entry_representation_gradients": _clone_cpu_tree(result[0]),
                    "stats": _clone_cpu_tree(result[1]),
                    "entry_action_q_targets": _clone_cpu_tree(result[2]),
                    "entry_action_q_valid": _clone_cpu_tree(result[3]),
                }
            )
            return result

        def captured_joint(model_arg: Any, task_losses: Mapping[str, torch.Tensor]):
            result = original_joint(model_arg, task_losses)
            joint_capture.update(
                {
                    "task_losses": {
                        name: value.detach().cpu().clone()
                        for name, value in task_losses.items()
                    },
                    "joint_loss": result[0].detach().cpu().clone(),
                    "stats": _clone_cpu_tree(result[1]),
                }
            )
            return result

        def captured_optimizer_owner(*, model: Any, optimizer: Any, weight_ema=None):
            parameter_names = {
                id(parameter): name for name, parameter in model.named_parameters()
            }
            raw_gradients.update(
                {
                    parameter_names[id(parameter)]: parameter.grad.detach()
                    .cpu()
                    .contiguous()
                    .clone()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                }
            )
            original_step = optimizer.step

            def captured_step(*step_args: Any, **step_kwargs: Any):
                clipped_gradients.update(
                    {
                        parameter_names[id(parameter)]: parameter.grad.detach()
                        .cpu()
                        .contiguous()
                        .clone()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    }
                )
                return original_step(*step_args, **step_kwargs)

            optimizer.step = captured_step
            try:
                return original_optimizer_owner(
                    model=model,
                    optimizer=optimizer,
                    weight_ema=weight_ema,
                )
            finally:
                optimizer.step = original_step

        target_trainer._model_forward_fp32 = captured_forward
        target_trainer._train_unified_exit_full_population = captured_exit
        target_trainer._joint_task_loss = captured_joint
        target_trainer._optimizer_step_with_finite_gradients = (
            captured_optimizer_owner
        )
        supervision = dict(
            restored["training_progress"]["joint_task_supervision_observed"]
        )
        gradients_observed = dict(
            restored["training_progress"]["joint_task_gradient_observed"]
        )
        train_result = target_trainer.train_epoch(
            model,
            target_model,
            _OneBatchLoader(),
            optimizer,
            device,
            grad_accum_steps=1,
            task_supervision_observed=supervision,
            task_gradient_observed=gradients_observed,
            weight_ema=weight_ema,
            session_batch_offset=batch_offset,
            session_max_optimizer_steps=1,
            session_checkpoint_hook=lambda **_unused: None,
            session_exit_action_forward_chunk_rows=(
                target_trainer.UNIFIED_EXIT_ACTION_FORWARD_CHUNK_ROWS
            ),
            session_log_label="SOURCE_STATE_SUCCESSOR_PARITY",
        )
        if train_result[2] is not False or not train_result[1].get("partial"):
            raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_DID_NOT_PAUSE_AFTER_ONE_STEP]")
        optimizer_state, optimizer_groups = _optimizer_state_by_name(
            model, optimizer
        )
        active_indices = restored["epoch_order"][
            batch_offset * batch_size : (batch_offset + 1) * batch_size
        ]
        capture.update(
            {
                "schema_version": _ACTUAL_NEXT_BATCH_CHILD_SCHEMA,
                "source_commit": args.source_commit,
                "recipe": recipe_binding,
                "contract": contract_binding,
                "pointer": pointer_binding,
                "state": state_binding,
                "checkpoint_index": int(restored["checkpoint_index"]),
                "global_optimizer_steps_before": int(
                    restored["global_optimizer_steps"]
                ),
                "next_batch_offset_before": batch_offset,
                "next_batch_indices": active_indices.tolist(),
                "batch_manifest": _value_manifest(batch),
                "checkpoint_cuda_rng_manifest": _value_manifest(
                    checkpoint_cuda_rng
                ),
                "entry_forwards": entry_forwards,
                "exit": exit_capture,
                "joint": joint_capture,
                "raw_gradients": raw_gradients,
                "clipped_gradients": clipped_gradients,
                "model_state_after": _clone_cpu_tree(model.state_dict()),
                "target_model_state_after": _clone_cpu_tree(
                    target_model.state_dict()
                ),
                "optimizer_state_after": optimizer_state,
                "optimizer_groups_after": optimizer_groups,
                "weight_ema_state_after": (
                    _clone_cpu_tree(weight_ema.checkpoint_state())
                    if weight_ema is not None
                    else None
                ),
                "lr_scheduler_state_after": (
                    _clone_cpu_tree(lr_scheduler.state_dict())
                    if lr_scheduler is not None
                    else None
                ),
                "task_supervision_observed": supervision,
                "task_gradient_observed": gradients_observed,
                "rng_state_after_manifest": _value_manifest(
                    target_trainer._attended_session_rng_state(device=device)
                ),
                "cuda_started": False,
                "test_accessed": False,
            }
        )
        child_out = args.child_out.resolve()
        if child_out.exists() or child_out.is_symlink():
            raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_CHILD_OUTPUT_EXISTS]")
        torch.save(capture, child_out)
        raise _ActualNextBatchCaptureComplete

    target_trainer._run_resumable_candidate_training = intercepted_candidate_training
    try:
        target_trainer.run_train(
            **_candidate_recipe_run_arguments(
                target_trainer=target_trainer,
                recipe=recipe,
                contract=contract,
            )
        )
    except _ActualNextBatchCaptureComplete:
        pass
    else:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_CAPTURE_NOT_REACHED]")
    if len(capture.get("entry_forwards", ())) != 2:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_ENTRY_FORWARD_CAPTURE_INVALID]")
    if _session_directory_snapshot(session_dir) != before_snapshot:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_SESSION_MUTATED]")
    _bound_regular_file(contract_path, contract_binding["sha256"])
    _bound_regular_file(pointer_path, pointer_binding["sha256"])
    _bound_regular_file(state_path, state_binding["sha256"])
    return 0


def _is_retired_state_name(name: str) -> bool:
    return any(
        name == retired or name.startswith(retired + ".")
        for retired in (
            "exit_path_encoder",
            "exit_entry_query_norm",
            "exit_entry_path_attention",
            "exit_fuse",
        )
    )


def _compare_tensor_mapping(
    original: Mapping[str, torch.Tensor],
    successor: Mapping[str, torch.Tensor],
    *,
    component: str,
    allow_scale_tolerance: bool,
) -> dict[str, Any]:
    original_active = {
        name: value
        for name, value in original.items()
        if not _is_retired_state_name(name)
    }
    if set(original_active) != set(successor):
        raise RuntimeError(
            f"[SOURCE_STATE_NEXT_BATCH_{component.upper()}_KEY_MISMATCH]"
        )
    maximum = 0.0
    tolerant_names: list[str] = []
    for name in sorted(successor):
        before = original_active[name]
        after = successor[name]
        if (
            not isinstance(before, torch.Tensor)
            or not isinstance(after, torch.Tensor)
            or before.dtype != after.dtype
            or before.shape != after.shape
        ):
            raise RuntimeError(
                f"[SOURCE_STATE_NEXT_BATCH_{component.upper()}_TENSOR_INVALID:{name}]"
            )
        difference = (
            float((before - after).abs().max().item())
            if before.is_floating_point() and before.numel()
            else 0.0
        )
        maximum = max(maximum, difference)
        scale_value = name.startswith(_TF_INPUT_SCALE_PREFIX) or (
            "." + _TF_INPUT_SCALE_PREFIX
        ) in name
        if allow_scale_tolerance and scale_value and before.is_floating_point():
            try:
                torch.testing.assert_close(
                    before, after, atol=1e-6, rtol=1e-5
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"[SOURCE_STATE_NEXT_BATCH_{component.upper()}_"
                    f"SCALE_TOLERANCE_EXCEEDED:{name}]"
                ) from exc
            tolerant_names.append(name)
        elif not torch.equal(before, after):
            raise RuntimeError(
                f"[SOURCE_STATE_NEXT_BATCH_{component.upper()}_VALUE_MISMATCH:{name}]"
            )
    return {
        "tensor_count": len(successor),
        "tolerance_limited_to_tf_input_scales": sorted(tolerant_names),
        "max_abs_difference": maximum,
    }


def _flatten_optimizer_state(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    flattened: dict[str, torch.Tensor] = {}
    for parameter_name, state in value.items():
        if _is_retired_state_name(parameter_name):
            raise RuntimeError(
                "[SOURCE_STATE_NEXT_BATCH_RETIRED_OPTIMIZER_STATE_PRESENT]"
            )
        for state_name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    "[SOURCE_STATE_NEXT_BATCH_OPTIMIZER_STATE_INVALID]"
                )
            flattened[f"{parameter_name}.{state_name}"] = tensor
    return flattened


def _active_optimizer_groups(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **{key: value for key, value in group.items() if key != "parameter_names"},
            "parameter_names": [
                name
                for name in group["parameter_names"]
                if not _is_retired_state_name(name)
            ],
        }
        for group in groups
    ]


def _require_actual_next_batch_equivalence(
    original: Mapping[str, Any], successor: Mapping[str, Any]
) -> dict[str, Any]:
    exact_metadata = (
        "checkpoint_index",
        "global_optimizer_steps_before",
        "next_batch_offset_before",
        "next_batch_indices",
        "batch_manifest",
        "checkpoint_cuda_rng_manifest",
        "task_supervision_observed",
        "task_gradient_observed",
        "rng_state_after_manifest",
    )
    for name in exact_metadata:
        if original.get(name) != successor.get(name):
            raise RuntimeError(
                f"[SOURCE_STATE_NEXT_BATCH_METADATA_MISMATCH:{name}]"
            )
    if (
        original.get("schema_version") != _ACTUAL_NEXT_BATCH_CHILD_SCHEMA
        or successor.get("schema_version") != _ACTUAL_NEXT_BATCH_CHILD_SCHEMA
        or original.get("cuda_started") is not False
        or successor.get("cuda_started") is not False
        or original.get("test_accessed") is not False
        or successor.get("test_accessed") is not False
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_CHILD_REPORT_INVALID]")
    results: dict[str, Any] = {}
    for index, (before, after) in enumerate(
        zip(original["entry_forwards"], successor["entry_forwards"], strict=True)
    ):
        results[f"entry_forward_{index}"] = _compare_tensor_mapping(
            before,
            after,
            component=f"entry_forward_{index}",
            allow_scale_tolerance=False,
        )
    for name in (
        "entry_representation_gradients",
        "entry_action_q_targets",
        "entry_action_q_valid",
    ):
        results[f"exit_{name}"] = _compare_tensor_mapping(
            {name: original["exit"][name]},
            {name: successor["exit"][name]},
            component=f"exit_{name}",
            allow_scale_tolerance=False,
        )
    if original["exit"]["stats"] != successor["exit"]["stats"]:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_EXIT_STATS_MISMATCH]")
    if original["joint"]["stats"] != successor["joint"]["stats"]:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_JOINT_STATS_MISMATCH]")
    results["joint_task_losses"] = _compare_tensor_mapping(
        original["joint"]["task_losses"],
        successor["joint"]["task_losses"],
        component="joint_task_losses",
        allow_scale_tolerance=False,
    )
    results["joint_loss"] = _compare_tensor_mapping(
        {"joint_loss": original["joint"]["joint_loss"]},
        {"joint_loss": successor["joint"]["joint_loss"]},
        component="joint_loss",
        allow_scale_tolerance=False,
    )
    for name in ("raw_gradients", "clipped_gradients"):
        results[name] = _compare_tensor_mapping(
            original[name],
            successor[name],
            component=name,
            allow_scale_tolerance=True,
        )
        tolerated = results[name]["tolerance_limited_to_tf_input_scales"]
        if tolerated != [
            f"tf_input_scale_{timeframe}"
            for timeframe in ("d1", "h1", "h4", "m15", "m5")
        ]:
            raise RuntimeError(
                f"[SOURCE_STATE_NEXT_BATCH_{name.upper()}_TOLERANCE_SCOPE_INVALID]"
            )
    results["model_state_after"] = _compare_tensor_mapping(
        original["model_state_after"],
        successor["model_state_after"],
        component="model_state_after",
        allow_scale_tolerance=True,
    )
    results["target_model_state_after"] = _compare_tensor_mapping(
        original["target_model_state_after"],
        successor["target_model_state_after"],
        component="target_model_state_after",
        allow_scale_tolerance=False,
    )
    results["optimizer_state_after"] = _compare_tensor_mapping(
        _flatten_optimizer_state(original["optimizer_state_after"]),
        _flatten_optimizer_state(successor["optimizer_state_after"]),
        component="optimizer_state_after",
        allow_scale_tolerance=True,
    )
    original_ema = original["weight_ema_state_after"]
    successor_ema = successor["weight_ema_state_after"]
    if (
        not isinstance(original_ema, Mapping)
        or not isinstance(successor_ema, Mapping)
        or {key: value for key, value in original_ema.items() if key != "shadow"}
        != {key: value for key, value in successor_ema.items() if key != "shadow"}
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_EMA_METADATA_MISMATCH]")
    results["weight_ema_state_after"] = _compare_tensor_mapping(
        original_ema["shadow"],
        successor_ema["shadow"],
        component="weight_ema_state_after",
        allow_scale_tolerance=True,
    )
    if (
        _active_optimizer_groups(original["optimizer_groups_after"])
        != _active_optimizer_groups(successor["optimizer_groups_after"])
        or original["lr_scheduler_state_after"]
        != successor["lr_scheduler_state_after"]
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_OPTIMIZER_METADATA_MISMATCH]")
    return results


def _run_actual_next_batch_process(
    *,
    source_root: Path,
    source_commit: str,
    recipe_path: Path,
    recipe_sha256: str,
    pointer_sha256: str,
    child_out: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "gx1.scripts.verify_candidate_checkpoint_resume_v1",
        "--actual-next-batch-child",
        "--source-root",
        str(source_root),
        "--source-commit",
        source_commit,
        "--recipe-json",
        str(recipe_path),
        "--recipe-sha256",
        recipe_sha256,
        "--pointer-sha256",
        pointer_sha256,
        "--child-out",
        str(child_out),
    ]
    completed = subprocess.run(
        command,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    if completed.returncode != 0 or not child_out.is_file():
        raise RuntimeError(
            "[SOURCE_STATE_NEXT_BATCH_CHILD_FAILED] "
            f"source_commit={source_commit} exit={completed.returncode}"
        )


def verify_source_state_successor_next_batch(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
        require_pretest_technical_recipe_metadata,
    )
    from gx1.contracts.entry_model_native_train_launch_v1 import (
        require_training_recipe_source_provenance,
    )

    repo = Path(__file__).resolve().parents[2]
    original_binding = _bound_regular_file(
        args.original_recipe_json.resolve(), args.original_recipe_sha256
    )
    successor_binding = _bound_regular_file(
        args.successor_recipe_json.resolve(), args.successor_recipe_sha256
    )
    migration_binding = _bound_regular_file(
        args.source_state_successor_report.resolve(),
        args.source_state_successor_report_sha256,
    )
    original = require_pretest_technical_recipe_metadata(
        json.loads(args.original_recipe_json.read_text(encoding="utf-8")),
        expected_profile="candidate",
    )
    successor = require_pretest_technical_recipe_metadata(
        json.loads(args.successor_recipe_json.read_text(encoding="utf-8")),
        expected_profile="candidate",
    )
    _require_source_state_successor_recipe_transition(original, successor)
    require_training_recipe_source_provenance(
        recipe_audit_path=args.successor_recipe_json,
        recipe_audit_sha256=args.successor_recipe_sha256,
        repo=repo,
        profile="candidate",
        run_id=successor["run_id"],
        dataset_run_id=successor["dataset_run_id"],
        dataset_dir=Path(successor["dataset_dir"]),
        out_bundle_dir=Path(successor["out_bundle_dir"]),
    )
    migration = json.loads(
        args.source_state_successor_report.read_text(encoding="utf-8")
    )
    original_recipe_identity = {
        key: original_binding[key] for key in ("path", "sha256")
    }
    successor_recipe_identity = {
        key: successor_binding[key] for key in ("path", "sha256")
    }
    if (
        migration.get("schema_version")
        != "gx1_candidate_source_state_successor_v1"
        or migration.get("decision")
        != "PASS_STRUCTURAL_STATE_SUCCESSOR_NOT_CUDA_AUTHORITY"
        or migration.get("original_recipe") != original_recipe_identity
        or migration.get("successor_recipe") != successor_recipe_identity
        or migration.get("successor_session_dir")
        != str(
            Path(successor["out_bundle_dir"]).parent
            / (
                trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX
                + Path(successor["out_bundle_dir"]).name
            )
        )
        or migration.get("actual_next_batch_equivalence_authority") is not False
        or migration.get("cuda_started") is not False
        or migration.get("test_accessed") is not False
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_MIGRATION_REPORT_INVALID]")
    original_commit = str(original["source_commit"])
    current_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != successor["source_commit"]:
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_CURRENT_COMMIT_MISMATCH]")
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        text=True,
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_WORKTREE_NOT_CLEAN]")

    with tempfile.TemporaryDirectory(
        prefix="gx1-source-state-next-batch-"
    ) as temporary:
        temporary_root = Path(temporary)
        archive_path = temporary_root / "historical.tar"
        historical_root = temporary_root / "historical"
        historical_root.mkdir(mode=0o700)
        with archive_path.open("wb") as handle:
            subprocess.run(
                ["git", "-C", str(repo), "archive", original_commit],
                stdout=handle,
                check=True,
            )
        with tarfile.open(archive_path, mode="r:") as archive:
            archive.extractall(historical_root)
        for binding in original["source_bindings"].values():
            relative = Path(binding["path"]).relative_to(repo)
            frozen = historical_root / relative
            if (
                _file_sha256(frozen) != binding["sha256"]
                or frozen.stat().st_size != binding["size_bytes"]
            ):
                raise RuntimeError(
                    "[SOURCE_STATE_NEXT_BATCH_HISTORICAL_SOURCE_INVALID]"
                )
        original_child = temporary_root / "original.pt"
        successor_child = temporary_root / "successor.pt"
        _run_actual_next_batch_process(
            source_root=historical_root,
            source_commit=original_commit,
            recipe_path=args.original_recipe_json,
            recipe_sha256=args.original_recipe_sha256,
            pointer_sha256=args.original_pointer_sha256,
            child_out=original_child,
        )
        _run_actual_next_batch_process(
            source_root=repo,
            source_commit=current_commit,
            recipe_path=args.successor_recipe_json,
            recipe_sha256=args.successor_recipe_sha256,
            pointer_sha256=args.successor_pointer_sha256,
            child_out=successor_child,
        )
        original_result = torch.load(
            original_child, map_location="cpu", weights_only=True
        )
        successor_result = torch.load(
            successor_child, map_location="cpu", weights_only=True
        )
        comparisons = _require_actual_next_batch_equivalence(
            original_result, successor_result
        )

    report = {
        "schema_version": _ACTUAL_NEXT_BATCH_REPORT_SCHEMA,
        "decision": "PASS_ACTUAL_CPU_NEXT_BATCH_EQUIVALENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "activation_authority": False,
        "actual_next_batch_equivalence_authority": True,
        "cpu_only": True,
        "cuda_started": False,
        "test_accessed": False,
        "original_recipe": original_binding,
        "successor_recipe": successor_binding,
        "source_state_successor_report": migration_binding,
        "original_pointer": original_result["pointer"],
        "successor_pointer": successor_result["pointer"],
        "original_state": original_result["state"],
        "successor_state": successor_result["state"],
        "original_source_commit": original_commit,
        "successor_source_commit": current_commit,
        "checkpoint_index": original_result["checkpoint_index"],
        "global_optimizer_steps_before": original_result[
            "global_optimizer_steps_before"
        ],
        "next_batch_offset_before": original_result["next_batch_offset_before"],
        "next_batch_indices": original_result["next_batch_indices"],
        "batch_manifest": original_result["batch_manifest"],
        "comparisons": comparisons,
        "tolerance_policy": {
            "exact": "all batch bytes, outputs, losses, masks, targets, non-scale gradients and active non-scale state",
            "tf_input_scale_derived_atol": 1e-6,
            "tf_input_scale_derived_rtol": 1e-5,
            "tolerated_parameter_names": [
                f"tf_input_scale_{timeframe}"
                for timeframe in ("d1", "h1", "h4", "m15", "m5")
            ],
        },
        "retired_static_exit_state_absent_from_successor": True,
        "original_session_preserved": True,
        "successor_session_preserved": True,
        "producer": _bound_regular_file(Path(__file__).resolve()),
    }
    out_json = args.out_json.resolve()
    if (
        not out_json.is_absolute()
        or out_json.parent.is_symlink()
        or not out_json.parent.is_dir()
        or out_json.exists()
        or out_json.is_symlink()
    ):
        raise RuntimeError("[SOURCE_STATE_NEXT_BATCH_REPORT_OUTPUT_INVALID]")
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(out_json, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
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
        for _ in range(_HALF):
            _step(model, optimizer, scheduler)
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
    parser.add_argument("--prepare-source-state-successor", action="store_true")
    parser.add_argument(
        "--verify-source-state-successor-next-batch", action="store_true"
    )
    parser.add_argument("--actual-next-batch-child", action="store_true")
    parser.add_argument("--out-bundle", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--original-recipe-json", type=Path)
    parser.add_argument("--original-recipe-sha256")
    parser.add_argument("--successor-recipe-json", type=Path)
    parser.add_argument("--successor-recipe-sha256")
    parser.add_argument("--original-pointer-sha256")
    parser.add_argument("--successor-pointer-sha256")
    parser.add_argument("--source-state-successor-report", type=Path)
    parser.add_argument("--source-state-successor-report-sha256")
    parser.add_argument("--incident-guard-log", type=Path)
    parser.add_argument("--incident-trainer-log", type=Path)
    parser.add_argument("--guard-repair-commit")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--recipe-json", type=Path)
    parser.add_argument("--recipe-sha256")
    parser.add_argument("--pointer-sha256")
    parser.add_argument("--child-out", type=Path)
    args = parser.parse_args(argv)
    transition_arguments = (
        args.original_recipe_json, args.original_recipe_sha256,
        args.successor_recipe_json, args.successor_recipe_sha256,
        args.original_pointer_sha256,
    )
    guard_recovery_arguments = (
        args.incident_guard_log,
        args.incident_trainer_log,
        args.guard_repair_commit,
    )
    actual_child_arguments = (
        args.source_root,
        args.source_commit,
        args.recipe_json,
        args.recipe_sha256,
        args.pointer_sha256,
        args.child_out,
    )
    actual_parent_arguments = (
        *transition_arguments,
        args.successor_pointer_sha256,
        args.source_state_successor_report,
        args.source_state_successor_report_sha256,
        args.out_json,
    )
    if args.actual_next_batch_child:
        if (
            not all(value is not None for value in actual_child_arguments)
            or any(value is not None for value in actual_parent_arguments)
            or any(value is not None for value in guard_recovery_arguments)
            or args.resume_child
            or args.prepare_guard_recovery
            or args.prepare_source_state_successor
            or args.verify_source_state_successor_next_batch
            or args.out_bundle is not None
            or args.out_dir is not None
        ):
            parser.error(
                "actual next-batch child requires only its six explicit bindings"
            )
        try:
            return _run_actual_next_batch_child(args)
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                f"FATAL: actual next-batch child failed; no CUDA authority: {exc}",
                file=sys.stderr,
            )
            return 2
    if args.verify_source_state_successor_next_batch:
        if (
            not all(value is not None for value in actual_parent_arguments)
            or any(value is not None for value in guard_recovery_arguments)
            or any(value is not None for value in actual_child_arguments)
            or args.resume_child
            or args.prepare_guard_recovery
            or args.prepare_source_state_successor
            or args.out_bundle is not None
            or args.out_dir is not None
        ):
            parser.error(
                "actual source-state next-batch verification requires all "
                "recipe, pointer, migration-report and output bindings"
            )
        try:
            report = verify_source_state_successor_next_batch(args)
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.CalledProcessError,
        ) as exc:
            print(
                "FATAL: source-state next-batch verification failed; "
                f"no CUDA authority: {exc}",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.prepare_guard_recovery:
        if (
            args.verify_source_state_successor_next_batch
            or args.actual_next_batch_child
            or args.prepare_source_state_successor
            or not all(value is not None for value in transition_arguments)
            or args.out_dir is None
            or not all(value is not None for value in guard_recovery_arguments)
            or any(value is not None for value in actual_child_arguments)
            or args.successor_pointer_sha256 is not None
            or args.source_state_successor_report is not None
            or args.source_state_successor_report_sha256 is not None
            or args.resume_child
            or args.out_bundle is not None
            or args.out_json is not None
        ):
            parser.error("guard recovery requires all explicit recovery bindings and no probe arguments")
        try:
            report = prepare_guard_recovery(args)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"FATAL: guard recovery failed; no CUDA authority: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.prepare_source_state_successor:
        if (
            not all(value is not None for value in transition_arguments)
            or args.out_dir is None
            or any(value is not None for value in guard_recovery_arguments)
            or any(value is not None for value in actual_child_arguments)
            or args.successor_pointer_sha256 is not None
            or args.source_state_successor_report is not None
            or args.source_state_successor_report_sha256 is not None
            or args.resume_child
            or args.out_bundle is not None
            or args.out_json is not None
        ):
            parser.error(
                "source-state successor requires all explicit transition bindings "
                "and no guard/probe arguments"
            )
        try:
            report = prepare_source_state_successor(args)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            print(
                f"FATAL: source-state successor failed; no CUDA authority: {exc}",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(report, sort_keys=True))
        return 0
    if any(
        value is not None
        for value in (
            *transition_arguments,
            *guard_recovery_arguments,
            *actual_child_arguments,
            args.successor_pointer_sha256,
            args.source_state_successor_report,
            args.source_state_successor_report_sha256,
            args.out_dir,
        )
    ):
        parser.error(
            "transition arguments require --prepare-guard-recovery or "
            "--prepare-source-state-successor or actual next-batch verification"
        )
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
