"""Read-only selection of the weight-EMA model from a strict Exit-v2 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    POINTER_SCHEMA,
    SCHEMA_VERSION,
    canonical_sha256,
    file_sha256,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
    strict_load_random_access_v2_state,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")

VAL_CHECKPOINT_BINDING_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_val_weight_ema_checkpoint_v1"
)


def _equal_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left.detach().cpu(), right.detach().cpu())
    )


def require_selected_weight_ema_checkpoint_binding_v1(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Validate the immutable read-only EMA selection receipt."""

    expected_keys = {
        "schema_version",
        "decision",
        "model_variant",
        "model_architecture_schema_version",
        "model_architecture_sha256",
        "model_state_sha256",
        "online_model_state_sha256",
        "target_model_state_sha256",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "checkpoint_pointer_path",
        "checkpoint_pointer_file_sha256",
        "checkpoint_pointer_sha256",
        "launch_manifest_sha256",
        "selected_sampler_artifact_sha256",
        "bootstrap_source_receipt_sha256",
        "base_normalization_sha256",
        "summary_normalization_sha256",
        "batch_size",
        "epoch_schedule_sha256",
        "epoch_index",
        "next_batch_offset",
        "global_step",
        "weight_ema_decay",
        "weight_ema_steps",
        "weight_ema_parameter_names_sha256",
        "online_buffers_preserved_exactly",
        "rng_mutated",
        "optimizer_or_scheduler_loaded",
        "test_data_used",
        "binding_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_BINDING_INVALID")
    result = dict(value)
    core = dict(result)
    claimed = core.pop("binding_sha256")
    sha_fields = (
        "model_architecture_sha256",
        "model_state_sha256",
        "online_model_state_sha256",
        "target_model_state_sha256",
        "checkpoint_file_sha256",
        "checkpoint_pointer_file_sha256",
        "checkpoint_pointer_sha256",
        "launch_manifest_sha256",
        "selected_sampler_artifact_sha256",
        "bootstrap_source_receipt_sha256",
        "base_normalization_sha256",
        "summary_normalization_sha256",
        "epoch_schedule_sha256",
        "weight_ema_parameter_names_sha256",
    )
    if (
        result["schema_version"] != VAL_CHECKPOINT_BINDING_SCHEMA_VERSION
        or result["decision"] != "PASS"
        or result["model_variant"] != "weight_ema"
        or result["model_architecture_schema_version"]
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or result["model_architecture_sha256"] != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
        or any(
            not isinstance(result[name], str) or _SHA256.fullmatch(result[name]) is None
            for name in sha_fields
        )
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or claimed != canonical_sha256(core)
        or result["batch_size"] not in (4, 8, 16)
        or any(
            type(result[name]) is not int or result[name] < minimum
            for name, minimum in (
                ("epoch_index", 0),
                ("next_batch_offset", 0),
                ("global_step", 1),
                ("weight_ema_steps", 1),
            )
        )
        or result["weight_ema_steps"] != result["global_step"]
        or isinstance(result["weight_ema_decay"], bool)
        or not isinstance(result["weight_ema_decay"], (int, float))
        or not math.isfinite(float(result["weight_ema_decay"]))
        or not 0.0 < float(result["weight_ema_decay"]) < 1.0
        or result["online_buffers_preserved_exactly"] is not True
        or result["rng_mutated"] is not False
        or result["optimizer_or_scheduler_loaded"] is not False
        or result["test_data_used"] is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_BINDING_INVALID")
    for path_name, sha_name in (
        ("checkpoint_path", "checkpoint_file_sha256"),
        ("checkpoint_pointer_path", "checkpoint_pointer_file_sha256"),
    ):
        path = Path(str(result[path_name]))
        if (
            not path.is_absolute()
            or path.resolve() != path
            or (
                verify_files
                and (
                    not path.is_file()
                    or path.is_symlink()
                    or file_sha256(path) != result[sha_name]
                )
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_BINDING_INVALID")
    return result


def load_selected_weight_ema_checkpoint_readonly_v1(
    *,
    pointer_path: Path,
    model: nn.Module,
    expected_checkpoint_pointer_file_sha256: str,
    expected_launch_manifest_sha256: str,
    expected_selected_sampler_artifact_sha256: str,
    expected_bootstrap_source_receipt_sha256: str,
    expected_base_normalization_sha256: str,
    expected_summary_normalization_sha256: str,
    expected_batch_size: int,
    expected_epoch_schedule_sha256: str,
    expected_weight_ema_decay: float,
) -> dict[str, Any]:
    """Strictly load EMA parameters while preserving online buffers and all RNG."""

    path = pointer_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_POINTER_INVALID")
    if file_sha256(path) != expected_checkpoint_pointer_file_sha256:
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_POINTER_INVALID")
    pointer = json.loads(path.read_text())
    pointer_core = dict(pointer)
    claimed_pointer = pointer_core.pop("pointer_sha256", None)
    state_path = Path(str(pointer.get("state_path", "")))
    if (
        pointer.get("schema_version") != POINTER_SCHEMA
        or claimed_pointer != canonical_sha256(pointer_core)
        or pointer.get("launch_manifest_sha256") != expected_launch_manifest_sha256
        or pointer.get("selected_sampler_artifact_sha256")
        != expected_selected_sampler_artifact_sha256
        or pointer.get("batch_size") != expected_batch_size
        or pointer.get("epoch_schedule_sha256") != expected_epoch_schedule_sha256
        or not state_path.is_absolute()
        or not state_path.is_file()
        or state_path.is_symlink()
        or file_sha256(state_path) != pointer.get("state_file_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_POINTER_INVALID")
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        [value.clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not torch.equal(torch.get_rng_state(), cpu_rng_before) or (
        torch.cuda.is_available()
        and any(
            not torch.equal(before, after)
            for before, after in zip(cuda_rng_before, torch.cuda.get_rng_state_all())
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_LOAD_MUTATED_RNG")
    if (
        not isinstance(state, Mapping)
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("launch_manifest_sha256") != expected_launch_manifest_sha256
        or state.get("selected_sampler_artifact_sha256")
        != expected_selected_sampler_artifact_sha256
        or state.get("bootstrap_source_receipt_sha256")
        != expected_bootstrap_source_receipt_sha256
        or state.get("base_normalization_sha256") != expected_base_normalization_sha256
        or state.get("summary_normalization_sha256")
        != expected_summary_normalization_sha256
        or state.get("batch_size") != expected_batch_size
        or state.get("epoch_schedule_sha256") != expected_epoch_schedule_sha256
        or state.get("global_step") != pointer.get("global_step")
        or state.get("next_batch_offset") != pointer.get("next_batch_offset")
        or state.get("epoch_index") != pointer.get("epoch_index")
        or state.get("old_v1_progress_reused") is not False
        or state.get("test_data_used") is not False
        or state.get("rng_contract") != "python_numpy_torch_cpu_and_all_cuda_v1"
        or state.get("bootstrap_model_receipts_sha256")
        != canonical_sha256(state.get("bootstrap_model_receipts"))
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_STATE_INVALID")
    online = state.get("online_model_state")
    target = state.get("target_model_state")
    expected = dict(model.state_dict())
    if (
        getattr(model, "unified_exit_random_access_architecture_version", None)
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or not isinstance(online, Mapping)
        or not isinstance(target, Mapping)
        or set(online) != set(expected)
        or set(target) != set(expected)
        or canonical_model_state_sha256(online)
        != state.get("online_model_state_sha256")
        or canonical_model_state_sha256(target)
        != state.get("target_model_state_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_MODEL_STATE_INVALID")
    for name, reference in expected.items():
        for observed in (online[name], target[name]):
            if (
                not isinstance(observed, torch.Tensor)
                or observed.shape != reference.shape
                or observed.dtype != reference.dtype
                or (
                    observed.is_floating_point()
                    and not bool(torch.isfinite(observed).all().item())
                )
            ):
                raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_MODEL_STATE_INVALID")
    marker = online.get("unified_exit_random_access_architecture_sha256")
    if (
        not isinstance(marker, torch.Tensor)
        or bytes(marker.detach().cpu().tolist()).hex()
        != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_MODEL_STATE_INVALID")
    ema = state.get("weight_ema_state")
    if not isinstance(ema, Mapping) or set(ema) != {
        "decay",
        "steps",
        "parameter_names",
        "shadow",
    }:
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_INVALID")
    expected_parameter_names = sorted(name for name, _ in model.named_parameters())
    expected_parameter_name_set = set(expected_parameter_names)
    parameter_names_raw = ema["parameter_names"]
    shadow = ema["shadow"]
    if (
        not isinstance(parameter_names_raw, list)
        or parameter_names_raw != expected_parameter_names
        or not all(isinstance(name, str) for name in parameter_names_raw)
        or not isinstance(shadow, Mapping)
        or set(shadow) != set(expected)
        or type(ema["steps"]) is not int
        or ema["steps"] < 1
        or type(state.get("global_step")) is not int
        or ema["steps"] != state["global_step"]
        or isinstance(ema["decay"], bool)
        or not isinstance(ema["decay"], (int, float))
        or not math.isfinite(float(ema["decay"]))
        or float(ema["decay"]) != float(expected_weight_ema_decay)
        or not 0.0 < float(ema["decay"]) < 1.0
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_INVALID")
    selected: dict[str, torch.Tensor] = {}
    for name, reference in expected.items():
        ema_tensor = shadow[name]
        if (
            not isinstance(ema_tensor, torch.Tensor)
            or ema_tensor.shape != reference.shape
            or ema_tensor.dtype != reference.dtype
            or (
                ema_tensor.is_floating_point()
                and not bool(torch.isfinite(ema_tensor).all().item())
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_INVALID")
        if name in expected_parameter_name_set:
            selected[name] = ema_tensor
        else:
            if not _equal_tensor(ema_tensor, online[name]):
                raise RuntimeError("UNIFIED_EXIT_VAL_CHECKPOINT_EMA_BUFFER_DRIFT")
            selected[name] = online[name]
    selected_digest = strict_load_random_access_v2_state(model, selected)
    model.requires_grad_(False)
    model.eval()
    binding = {
        "schema_version": VAL_CHECKPOINT_BINDING_SCHEMA_VERSION,
        "decision": "PASS",
        "model_variant": "weight_ema",
        "model_architecture_schema_version": RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
        "model_architecture_sha256": RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
        "model_state_sha256": selected_digest,
        "online_model_state_sha256": state["online_model_state_sha256"],
        "target_model_state_sha256": state["target_model_state_sha256"],
        "checkpoint_path": str(state_path),
        "checkpoint_file_sha256": pointer["state_file_sha256"],
        "checkpoint_pointer_path": str(path),
        "checkpoint_pointer_file_sha256": file_sha256(path),
        "checkpoint_pointer_sha256": claimed_pointer,
        "launch_manifest_sha256": expected_launch_manifest_sha256,
        "selected_sampler_artifact_sha256": (expected_selected_sampler_artifact_sha256),
        "bootstrap_source_receipt_sha256": (expected_bootstrap_source_receipt_sha256),
        "base_normalization_sha256": expected_base_normalization_sha256,
        "summary_normalization_sha256": expected_summary_normalization_sha256,
        "batch_size": int(state["batch_size"]),
        "epoch_schedule_sha256": state["epoch_schedule_sha256"],
        "epoch_index": int(state["epoch_index"]),
        "next_batch_offset": int(state["next_batch_offset"]),
        "global_step": int(state["global_step"]),
        "weight_ema_decay": float(ema["decay"]),
        "weight_ema_steps": int(ema["steps"]),
        "weight_ema_parameter_names_sha256": canonical_sha256(expected_parameter_names),
        "online_buffers_preserved_exactly": True,
        "rng_mutated": False,
        "optimizer_or_scheduler_loaded": False,
        "test_data_used": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return require_selected_weight_ema_checkpoint_binding_v1(binding)


__all__ = (
    "VAL_CHECKPOINT_BINDING_SCHEMA_VERSION",
    "load_selected_weight_ema_checkpoint_readonly_v1",
    "require_selected_weight_ema_checkpoint_binding_v1",
)
