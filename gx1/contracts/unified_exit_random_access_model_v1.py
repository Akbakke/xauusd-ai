"""Architecture and one-time bootstrap contract for random-access Exit v2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_legacy_state_v1 import RETIRED_STATIC_EXIT_STATE_KEYS

RANDOM_ACCESS_MODEL_SCHEMA_VERSION = "gx1_unified_exit_random_access_model_v1"
RANDOM_ACCESS_MODEL_SCHEMA_SHA256 = hashlib.sha256(
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION.encode("ascii")
).hexdigest()
_NEW_STATE_PREFIXES = (
    "unified_exit_random_access_architecture_sha256",
    "exit_random_access_summary_proj.",
    "exit_random_access_fuse.",
)


def liquidation_relative_action_values(raw_q_bps: torch.Tensor) -> torch.Tensor:
    """Reuse both head rows to learn HOLD advantage; EXIT_NOW is exactly zero.

    The common raw level cancels. No liquidation price, future price or new
    learned parameter enters this head. The economics owner reconstructs total
    trade value for Entry using the observed first-state liquidation value.
    """
    if (not isinstance(raw_q_bps, torch.Tensor) or raw_q_bps.ndim < 2
            or raw_q_bps.shape[-1] != 2 or raw_q_bps.dtype != torch.float32
            or not bool(torch.isfinite(raw_q_bps).all().item())):
        raise RuntimeError("UNIFIED_EXIT_LIQUIDATION_ADVANTAGE_HEAD_INVALID")
    hold = raw_q_bps[..., 0] - raw_q_bps[..., 1]
    if not bool(torch.isfinite(hold).all().item()):
        raise RuntimeError("UNIFIED_EXIT_LIQUIDATION_ADVANTAGE_HEAD_INVALID")
    return torch.stack((hold, torch.zeros_like(hold)), dim=-1)


def _new_state_keys(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name in model.state_dict()
        if any(
            name == prefix or name.startswith(prefix) for prefix in _NEW_STATE_PREFIXES
        )
    )


def _require_model(model: nn.Module) -> dict[str, torch.Tensor]:
    if (
        not isinstance(model, nn.Module)
        or getattr(model, "unified_exit_random_access_architecture_version", None)
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_MODEL_ARCHITECTURE_INVALID")
    state = dict(model.state_dict())
    new_keys = _new_state_keys(model)
    marker = state.get("unified_exit_random_access_architecture_sha256")
    if (
        not new_keys
        or not isinstance(marker, torch.Tensor)
        or marker.dtype != torch.uint8
        or marker.numel() != 32
        or bytes(marker.detach().cpu().tolist()).hex()
        != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_MODEL_ARCHITECTURE_INVALID")
    return state


def bootstrap_random_access_v2_from_pretrained(
    model: nn.Module, pretrained_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse every compatible v1 weight while retaining new v2 initialization."""

    current = _require_model(model)
    if not isinstance(pretrained_state, Mapping) or not pretrained_state:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BOOTSTRAP_STATE_INVALID")
    incoming = dict(pretrained_state)
    new_keys = set(_new_state_keys(model))
    if new_keys.intersection(incoming):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BOOTSTRAP_REQUIRES_V1_STATE")
    expected_old = set(current) - new_keys
    retired = set(incoming) - expected_old
    if (
        expected_old - set(incoming)
        or (retired and retired != RETIRED_STATIC_EXIT_STATE_KEYS)
        or retired.intersection(current)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BOOTSTRAP_KEYSET_INVALID")
    # Historical pre-successor checkpoints carry exactly the reviewed retired
    # static Exit modules. Reject partial retirement or any unknown extra key;
    # keep the full source digest while reusing every current backbone tensor.
    reusable = {name: incoming[name] for name in expected_old}
    for name in expected_old:
        value = incoming[name]
        expected = current[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_RANDOM_ACCESS_BOOTSTRAP_TENSOR_INVALID:{name}"
            )
    initialized_before = canonical_model_state_sha256(
        {name: current[name] for name in sorted(new_keys)}
    )
    merged = {**current, **reusable}
    model.load_state_dict(merged, strict=True)
    restored = _require_model(model)
    initialized_after = canonical_model_state_sha256(
        {name: restored[name] for name in sorted(new_keys)}
    )
    if initialized_after != initialized_before:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BOOTSTRAP_INIT_CHANGED")
    return {
        "schema_version": "gx1_unified_exit_random_access_bootstrap_receipt_v1",
        "decision": "PASS",
        "migration_kind": "one_time_v1_backbone_to_random_access_v2",
        "architecture_schema_version": RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
        "architecture_schema_sha256": RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
        "source_pretrained_state_sha256": canonical_model_state_sha256(incoming),
        "initialized_v2_state_sha256": initialized_after,
        "resulting_v2_state_sha256": canonical_model_state_sha256(restored),
        "reused_state_key_count": len(reusable),
        "retired_state_keys": sorted(retired),
        "retired_state_sha256": canonical_model_state_sha256(
            {name: incoming[name] for name in sorted(retired)}
        ) if retired else None,
        "initialized_state_keys": sorted(new_keys),
        "old_checkpoint_is_v2_resume": False,
        "strict_v2_restore_required_after_bootstrap": True,
    }


def bind_preserved_v7_input_normalization(
    model: nn.Module, legacy_contract: Mapping[str, Any]
) -> str:
    """Adopt the verified v7 contract after its buffers were bootstrapped.

    The v2 model is constructed with the schema-compatible child v8 contract.
    Bootstrap then restores every learned v1 tensor, including the historical
    normalization buffers. This explicit binding updates the Python-side
    contract only after proving field, alias and categorical identities match.
    """

    from gx1.contracts.unified_exit_random_access_cuda_smoke_v1 import (
        _require_legacy_v7_normalization,
    )

    _require_model(model)
    checked = _require_legacy_v7_normalization(legacy_contract)
    current = getattr(model, "_input_normalization_contract", None)
    if not isinstance(current, Mapping):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_NORMALIZATION_STATE_MISSING")
    old_surfaces = checked.get("surfaces", {})
    current_surfaces = current.get("surfaces", {})
    if (
        {
            name: list(surface.get("field_names") or [])
            for name, surface in old_surfaces.items()
        }
        != {
            name: list(surface.get("field_names") or [])
            for name, surface in current_surfaces.items()
        }
        or checked.get("temporal_aliases") != current.get("temporal_aliases")
        or checked.get("ctx_cat", {}).get("field_names")
        != current.get("ctx_cat", {}).get("field_names")
        or checked.get("ctx_cat", {}).get("domains")
        != current.get("ctx_cat", {}).get("domains")
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_NORMALIZATION_SCHEMA_MISMATCH")
    model._input_normalization_contract = checked
    model.require_input_normalization_state()
    return str(checked["contract_sha256"])


def strict_load_random_access_v2_state(
    model: nn.Module, state: Mapping[str, Any]
) -> str:
    """Strictly restore a checkpoint that already carries the v2 marker."""

    current = _require_model(model)
    if not isinstance(state, Mapping) or set(state) != set(current):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_V2_STATE_KEYSET_INVALID")
    marker = state.get("unified_exit_random_access_architecture_sha256")
    if (
        not isinstance(marker, torch.Tensor)
        or bytes(marker.detach().cpu().tolist()).hex()
        != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_V2_STATE_MARKER_INVALID")
    model.load_state_dict(state, strict=True)
    return canonical_model_state_sha256(model.state_dict())


__all__ = (
    "RANDOM_ACCESS_MODEL_SCHEMA_SHA256",
    "RANDOM_ACCESS_MODEL_SCHEMA_VERSION",
    "bootstrap_random_access_v2_from_pretrained",
    "bind_preserved_v7_input_normalization",
    "strict_load_random_access_v2_state",
)
