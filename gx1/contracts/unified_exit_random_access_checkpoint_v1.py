"""Atomic checkpoint owner for fixed-step random-access Exit v2 training."""

from __future__ import annotations
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
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


def build_checkpoint(
    *,
    model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    next_batch_offset: int,
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
        "launch_manifest_sha256": launch_manifest_sha256,
        "selected_sampler_artifact_sha256": selected_sampler_artifact_sha256,
        "bootstrap_source_receipt_sha256": bootstrap_source_receipt_sha256,
        "bootstrap_model_receipts": dict(bootstrap_model_receipts),
        "base_normalization_sha256": base_normalization_sha256,
        "summary_normalization_sha256": summary_normalization_sha256,
        "online_model_state_sha256": canonical_model_state_sha256(online),
        "target_model_state_sha256": canonical_model_state_sha256(target),
        "online_model_state": online,
        "target_model_state": target,
        "optimizer_state": optimizer.state_dict(),
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
    if state_path.exists():
        raise RuntimeError("UNIFIED_EXIT_V2_CHECKPOINT_EXISTS")
    fd, tmp = tempfile.mkstemp(prefix=f".{state_path.name}.", dir=directory)
    os.close(fd)
    try:
        torch.save(dict(value), tmp)
        os.replace(tmp, state_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    pointer = {
        "schema_version": POINTER_SCHEMA,
        "state_path": str(state_path),
        "state_file_sha256": file_sha256(state_path),
        "global_step": index,
        "next_batch_offset": int(value["next_batch_offset"]),
        "launch_manifest_sha256": value["launch_manifest_sha256"],
        "selected_sampler_artifact_sha256": value["selected_sampler_artifact_sha256"],
    }
    pointer["pointer_sha256"] = canonical_sha256(pointer)
    pointer_path = directory / "RESUME_POINTER.json"
    payload = (json.dumps(pointer, sort_keys=True, indent=2) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=".RESUME_POINTER.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pointer_path)
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
    expected_launch_manifest_sha256: str,
    expected_selected_sampler_artifact_sha256: str,
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
        or not state_path.is_file()
        or state_path.is_symlink()
        or file_sha256(state_path) != pointer.get("state_file_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_V2_RESUME_POINTER_INVALID")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("old_v1_progress_reused") is not False
        or state.get("test_data_used") is not False
        or state.get("global_step") != pointer["global_step"]
        or state.get("next_batch_offset") != pointer["next_batch_offset"]
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
    return {
        "global_step": state["global_step"],
        "next_batch_offset": state["next_batch_offset"],
        "bootstrap_source_receipt_sha256": state["bootstrap_source_receipt_sha256"],
        "bootstrap_model_receipts": state["bootstrap_model_receipts"],
        "base_normalization_sha256": state["base_normalization_sha256"],
        "summary_normalization_sha256": state["summary_normalization_sha256"],
    }


__all__ = (
    "SCHEMA_VERSION",
    "POINTER_SCHEMA",
    "build_checkpoint",
    "save_checkpoint_atomic",
    "load_checkpoint_strict",
)
