"""Atomic checkpoint owner for fixed-step random-access Exit v2 training."""

from __future__ import annotations
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    strict_load_random_access_v2_state,
)

SCHEMA_VERSION = "gx1_unified_exit_random_access_fixed_step_checkpoint_v1"
POINTER_SCHEMA = "gx1_unified_exit_random_access_fixed_step_pointer_v1"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build_checkpoint(
    *,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    weight_ema_state: Mapping[str, Any] | None,
    global_step: int,
    next_batch_offset: int,
    epoch_index: int,
    batch_size: int,
    epoch_schedule_sha256: str,
    launch_manifest_sha256: str,
    selected_sampler_artifact_sha256: str,
    bootstrap_source_receipt_sha256: str,
    bootstrap_model_receipts: Mapping[str, Any],
    base_normalization_sha256: str,
    summary_normalization_sha256: str,
) -> dict[str, Any]:
    if (
        isinstance(global_step, bool)
        or global_step < 0
        or isinstance(next_batch_offset, bool)
        or next_batch_offset < 0
        or isinstance(epoch_index, bool)
        or epoch_index < 0
        or isinstance(batch_size, bool)
        or batch_size not in (4, 8, 16)
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_PROGRESS_INVALID")
    online = dict(model.state_dict())
    target = dict(target_model.state_dict())
    for state in (online, target):
        marker = state.get("unified_exit_random_access_architecture_sha256")
        if (
            not isinstance(marker, torch.Tensor)
            or bytes(marker.detach().cpu().tolist()).hex()
            != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
        ):
            raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_MARKER_INVALID")
    value = {
        "schema_version": SCHEMA_VERSION,
        "global_step": global_step,
        "next_batch_offset": next_batch_offset,
        "epoch_index": epoch_index,
        "batch_size": batch_size,
        "epoch_schedule_sha256": epoch_schedule_sha256,
        "launch_manifest_sha256": launch_manifest_sha256,
        "selected_sampler_artifact_sha256": selected_sampler_artifact_sha256,
        "bootstrap_source_receipt_sha256": bootstrap_source_receipt_sha256,
        "bootstrap_model_receipts": dict(bootstrap_model_receipts),
        "bootstrap_model_receipts_sha256": canonical_sha256(bootstrap_model_receipts),
        "base_normalization_sha256": base_normalization_sha256,
        "summary_normalization_sha256": summary_normalization_sha256,
        "online_model_state_sha256": canonical_model_state_sha256(online),
        "target_model_state_sha256": canonical_model_state_sha256(target),
        "online_model_state": online,
        "target_model_state": target,
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": (
            lr_scheduler.state_dict() if lr_scheduler is not None else None
        ),
        "weight_ema_state": (
            dict(weight_ema_state) if weight_ema_state is not None else None
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": {
            "bit_generator": np.random.get_state()[0],
            "keys": np.random.get_state()[1].astype(np.uint32).tolist(),
            "position": int(np.random.get_state()[2]),
            "has_gauss": int(np.random.get_state()[3]),
            "cached_gaussian": float(np.random.get_state()[4]),
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
        "rng_contract": "python_numpy_torch_cpu_and_all_cuda_v1",
        "old_v1_progress_reused": False,
        "test_data_used": False,
    }
    return value


def save_checkpoint_atomic(
    value: Mapping[str, Any], *, directory: Path
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    index = int(value["global_step"])
    state_path = directory / f"state_{index:08d}.pt"
    pointer_path = directory / "RESUME_POINTER.json"
    if state_path.exists() and pointer_path.exists():
        active = json.loads(pointer_path.read_text())
        if active.get("state_path") == str(state_path):
            raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_EXISTS")
    # A state at this never-published step is an orphan from a crash between
    # state and pointer publication. The atomic replace below safely supersedes it.
    fd, tmp = tempfile.mkstemp(prefix=f".{state_path.name}.", dir=directory)
    os.close(fd)
    try:
        torch.save(dict(value), tmp)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, state_path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    pointer = {
        "schema_version": POINTER_SCHEMA,
        "state_path": str(state_path),
        "state_file_sha256": file_sha256(state_path),
        "global_step": index,
        "next_batch_offset": int(value["next_batch_offset"]),
        "epoch_index": int(value["epoch_index"]),
        "batch_size": int(value["batch_size"]),
        "epoch_schedule_sha256": value["epoch_schedule_sha256"],
        "launch_manifest_sha256": value["launch_manifest_sha256"],
        "selected_sampler_artifact_sha256": value["selected_sampler_artifact_sha256"],
    }
    pointer["pointer_sha256"] = canonical_sha256(pointer)
    payload = (json.dumps(pointer, sort_keys=True, indent=2) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=".RESUME_POINTER.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pointer_path)
        _fsync_directory(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return pointer


def load_checkpoint_strict(
    *,
    pointer_path: Path,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    expected_launch_manifest_sha256: str,
    expected_selected_sampler_artifact_sha256: str,
    expected_bootstrap_source_receipt_sha256: str,
    expected_base_normalization_sha256: str,
    expected_summary_normalization_sha256: str,
    expected_batch_size: int,
    expected_epoch_schedule_sha256: str,
) -> dict[str, Any]:
    pointer = json.loads(pointer_path.read_text())
    data = dict(pointer)
    claimed = data.pop("pointer_sha256", None)
    state_path = Path(pointer.get("state_path", ""))
    if (
        pointer.get("schema_version") != POINTER_SCHEMA
        or claimed != canonical_sha256(data)
        or pointer.get("launch_manifest_sha256") != expected_launch_manifest_sha256
        or pointer.get("selected_sampler_artifact_sha256")
        != expected_selected_sampler_artifact_sha256
        or pointer.get("batch_size") != expected_batch_size
        or pointer.get("epoch_schedule_sha256") != expected_epoch_schedule_sha256
        or not state_path.is_file()
        or state_path.is_symlink()
        or file_sha256(state_path) != pointer.get("state_file_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_RESUME_POINTER_INVALID")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("launch_manifest_sha256") != expected_launch_manifest_sha256
        or state.get("selected_sampler_artifact_sha256")
        != expected_selected_sampler_artifact_sha256
        or state.get("old_v1_progress_reused") is not False
        or state.get("test_data_used") is not False
        or state.get("global_step") != pointer["global_step"]
        or state.get("next_batch_offset") != pointer["next_batch_offset"]
        or state.get("epoch_index") != pointer["epoch_index"]
        or state.get("batch_size") != pointer["batch_size"]
        or state.get("epoch_schedule_sha256") != pointer["epoch_schedule_sha256"]
        or state.get("rng_contract") != "python_numpy_torch_cpu_and_all_cuda_v1"
        or not isinstance(state.get("numpy_rng_state"), Mapping)
        or state.get("bootstrap_source_receipt_sha256")
        != expected_bootstrap_source_receipt_sha256
        or state.get("bootstrap_model_receipts_sha256")
        != canonical_sha256(state.get("bootstrap_model_receipts"))
        or state.get("base_normalization_sha256") != expected_base_normalization_sha256
        or state.get("summary_normalization_sha256")
        != expected_summary_normalization_sha256
        or not isinstance(state.get("weight_ema_state"), Mapping)
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_INVALID")
    online_digest = strict_load_random_access_v2_state(
        model, state["online_model_state"]
    )
    target_digest = strict_load_random_access_v2_state(
        target_model, state["target_model_state"]
    )
    if (
        online_digest != state["online_model_state_sha256"]
        or target_digest != state["target_model_state_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_STATE_INVALID")
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler_state = state["lr_scheduler_state"]
    if (lr_scheduler is None) != (scheduler_state is None):
        raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_SCHEDULER_INVALID")
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(scheduler_state)
    random.setstate(state["python_rng_state"])
    numpy_state = state["numpy_rng_state"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_rng_state"])
    if state["cuda_rng_states"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda_rng_states"]) != torch.cuda.device_count()
        ):
            raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_CUDA_RNG_INVALID")
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    return {
        "global_step": state["global_step"],
        "next_batch_offset": state["next_batch_offset"],
        "epoch_index": state["epoch_index"],
        "batch_size": state["batch_size"],
        "epoch_schedule_sha256": state["epoch_schedule_sha256"],
        "bootstrap_source_receipt_sha256": state["bootstrap_source_receipt_sha256"],
        "bootstrap_model_receipts": state["bootstrap_model_receipts"],
        "base_normalization_sha256": state["base_normalization_sha256"],
        "summary_normalization_sha256": state["summary_normalization_sha256"],
        "weight_ema_state": state["weight_ema_state"],
    }


__all__ = (
    "SCHEMA_VERSION",
    "POINTER_SCHEMA",
    "build_checkpoint",
    "save_checkpoint_atomic",
    "load_checkpoint_strict",
)
