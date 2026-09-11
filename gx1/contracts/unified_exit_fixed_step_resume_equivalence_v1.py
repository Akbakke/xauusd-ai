"""Bit-exact reference-4 versus split-3+fresh-process-resume-1 owner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "gx1_unified_exit_fixed_step_resume_equivalence_v1"
SCHEDULE_SCHEMA = "gx1_unified_exit_epoch_schedule_witness_v1"
_STATE_FAMILIES = (
    "online_model_state",
    "target_model_state",
    "optimizer_state",
    "lr_scheduler_state",
    "weight_ema_state",
    "python_rng_state",
    "numpy_rng_state",
    "torch_rng_state",
    "cuda_rng_states",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _update(h: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        h.update(b"tensor\0")
        h.update(str(tensor.dtype).encode())
        h.update(json.dumps(list(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        h.update(b"map\0")
        for key in sorted(value):
            h.update(str(key).encode() + b"\0")
            _update(h, value[key])
    elif isinstance(value, (list, tuple)):
        h.update(b"seq\0")
        for item in value:
            _update(h, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\0")
    else:
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_STATE_TYPE_INVALID")


def state_digest(state: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    _update(h, state)
    return h.hexdigest()


def _load_pointer(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_POINTER_INVALID")
    pointer = json.loads(path.read_text())
    data = dict(pointer)
    claimed = data.pop("pointer_sha256", None)
    state_path = Path(str(pointer.get("state_path", "")))
    if (
        claimed != canonical_sha256(data)
        or not state_path.is_absolute()
        or not state_path.is_file()
        or state_path.is_symlink()
        or file_sha256(state_path) != pointer.get("state_file_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_POINTER_INVALID")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(key not in state for key in _STATE_FAMILIES):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_STATE_INVALID")
    return pointer, state


def _load_schedule(path: Path, *, expected_batch_size: int) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_SCHEDULE_INVALID")
    value = json.loads(path.read_text())
    data = dict(value)
    claimed = data.pop("witness_sha256", None)
    next_batch = value.get("next_batch_after_optimizer_step_4")
    if not isinstance(next_batch, Mapping):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_SCHEDULE_INVALID")
    next_data = dict(next_batch)
    next_claimed = next_data.pop("identity_sha256", None)
    children = next_batch.get("child_entry_row_indices")
    parents = next_batch.get("parent_entry_row_indices")
    if (
        value.get("schema_version") != SCHEDULE_SCHEMA
        or value.get("epoch_index") != 0
        or value.get("batch_size") != expected_batch_size
        or value.get("entry_pair_count") != 16384
        or value.get("transition_count") != 65536
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
        or next_batch.get("batch_offset") != 4
        or next_batch.get("batch_size") != expected_batch_size
        or not isinstance(children, list)
        or not isinstance(parents, list)
        or len(children) != expected_batch_size
        or len(parents) != expected_batch_size
        or any(type(item) is not int or item < 0 for item in children + parents)
        or next_claimed != canonical_sha256(next_data)
    ):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_SCHEDULE_INVALID")
    return value


def build_equivalence(
    *,
    reference_pointer_path: Path,
    split_pointer_path: Path,
    reference_schedule_path: Path,
    split_schedule_path: Path,
    expected_launch_manifest_sha256: str,
    expected_gpu_batch_selection_artifact_sha256: str,
    expected_selected_sampler_artifact_sha256: str,
    expected_batch_size: int,
    expected_epoch_schedule_sha256: str,
) -> dict[str, Any]:
    ref_pointer, ref = _load_pointer(reference_pointer_path)
    split_pointer, split = _load_pointer(split_pointer_path)
    ref_schedule = _load_schedule(reference_schedule_path, expected_batch_size=expected_batch_size)
    split_schedule = _load_schedule(split_schedule_path, expected_batch_size=expected_batch_size)
    ref_digest = state_digest({key: ref[key] for key in _STATE_FAMILIES})
    split_digest = state_digest({key: split[key] for key in _STATE_FAMILIES})
    expected = {
        "launch_manifest_sha256": expected_launch_manifest_sha256,
        "selected_sampler_artifact_sha256": expected_selected_sampler_artifact_sha256,
        "batch_size": expected_batch_size,
        "epoch_schedule_sha256": expected_epoch_schedule_sha256,
        "global_step": 4,
        "next_batch_offset": 4,
        "epoch_index": 0,
    }
    if (
        any(ref.get(k) != v or split.get(k) != v for k, v in expected.items())
        or ref_digest != split_digest
        or ref_schedule != split_schedule
        or ref_schedule["epoch_schedule_sha256"] != expected_epoch_schedule_sha256
        or ref_schedule["selected_sampler_artifact_sha256"]
        != expected_selected_sampler_artifact_sha256
    ):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_MISMATCH")
    next_identity = dict(ref_schedule["next_batch_after_optimizer_step_4"])
    value = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "reference_pointer": {"path": str(reference_pointer_path), "sha256": file_sha256(reference_pointer_path)},
        "split_pointer": {"path": str(split_pointer_path), "sha256": file_sha256(split_pointer_path)},
        "reference_schedule": {"path": str(reference_schedule_path), "sha256": file_sha256(reference_schedule_path)},
        "split_schedule": {"path": str(split_schedule_path), "sha256": file_sha256(split_schedule_path)},
        "launch_manifest_sha256": expected_launch_manifest_sha256,
        "gpu_batch_selection_artifact_sha256": expected_gpu_batch_selection_artifact_sha256,
        "selected_sampler_artifact_sha256": expected_selected_sampler_artifact_sha256,
        "batch_size": expected_batch_size,
        "epoch_index": 0,
        "global_optimizer_steps": 4,
        "next_batch_offset": 4,
        "epoch_schedule_sha256": expected_epoch_schedule_sha256,
        "reference_state_digest": ref_digest,
        "split_state_digest": split_digest,
        "compared_state_families": list(_STATE_FAMILIES),
        "next_batch_identity": next_identity,
        "test_data_used": False,
    }
    value["artifact_sha256"] = canonical_sha256(value)
    return value


def require_equivalence(value: Mapping[str, Any], *, verify_files: bool = True) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("artifact_sha256", None)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("decision") != "PASS"
        or value.get("reference_state_digest") != value.get("split_state_digest")
        or value.get("global_optimizer_steps") != 4
        or value.get("next_batch_offset") != 4
        or value.get("test_data_used") is not False
        or claimed != canonical_sha256(data)
    ):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_INVALID")
    next_batch = value.get("next_batch_identity")
    if not isinstance(next_batch, Mapping):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_INVALID")
    next_data = dict(next_batch)
    next_claimed = next_data.pop("identity_sha256", None)
    if next_claimed != canonical_sha256(next_data):
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_INVALID")
    for key in ("reference_pointer", "split_pointer", "reference_schedule", "split_schedule"):
        binding = value.get(key)
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_BINDING_INVALID")
        path = Path(str(binding["path"]))
        if not path.is_absolute() or (verify_files and (not path.is_file() or path.is_symlink() or file_sha256(path) != binding["sha256"])):
            raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_BINDING_INVALID")
    if verify_files:
        rebuilt = build_equivalence(
            reference_pointer_path=Path(value["reference_pointer"]["path"]),
            split_pointer_path=Path(value["split_pointer"]["path"]),
            reference_schedule_path=Path(value["reference_schedule"]["path"]),
            split_schedule_path=Path(value["split_schedule"]["path"]),
            expected_launch_manifest_sha256=str(value["launch_manifest_sha256"]),
            expected_gpu_batch_selection_artifact_sha256=str(value["gpu_batch_selection_artifact_sha256"]),
            expected_selected_sampler_artifact_sha256=str(value["selected_sampler_artifact_sha256"]),
            expected_batch_size=int(value["batch_size"]),
            expected_epoch_schedule_sha256=str(value["epoch_schedule_sha256"]),
        )
        if rebuilt != dict(value):
            raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_REBUILD_MISMATCH")
    return dict(value)


def publish_equivalence(value: Mapping[str, Any], output: Path) -> None:
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_RESUME_EQ_OUTPUT_INVALID")
    checked = require_equivalence(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(checked, sort_keys=True, indent=2) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
