"""TRAIN-fitted Bellman targets for the sole Entry action authority.

Entry is the Bellman step immediately before unified Exit.  LONG and SHORT
receive zero transition reward and continue into the corresponding first
authoritative post-fill Exit state.  FLAT terminates immediately at zero bps.
Only an immutable TRAIN-fitted Exit target snapshot may provide the two
continuation values; future path outcomes are never read directly here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_fitted_q_iteration_state,
    unified_exit_fitted_q_contract,
)


ENTRY_FITTED_Q_SCHEMA_VERSION = "gx1_entry_fitted_q_v2"
ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION = (
    "gx1_entry_fitted_q_iteration_state_v2"
)
ENTRY_FITTED_Q_ACTION_ORDER = ("LONG", "SHORT", "FLAT")
ENTRY_FITTED_Q_TARGET_UNIT = "raw_bps"
ENTRY_FITTED_Q_GAMMA = 1.0
ENTRY_FITTED_Q_PRODUCTION_ECONOMICS_SCHEMA_VERSION = (
    "gx1_entry_fitted_q_production_economics_v2"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ITERATION_FIELDS = frozenset(
    {
        "schema_version",
        "iteration_index",
        "entry_target_model_state_sha256",
        "exit_target_model_state_sha256",
        "exit_fitted_q_iteration_state_sha256",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "normalization_sha256",
        "entry_fitted_q_contract",
        "exit_fitted_q_contract",
        "target_updated_from_val_or_test",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"ENTRY_FITTED_Q_{label}_SHA256_INVALID")
    return value


def entry_fitted_q_contract() -> dict[str, Any]:
    payload = {
        "schema_version": ENTRY_FITTED_Q_SCHEMA_VERSION,
        "action_order": list(ENTRY_FITTED_Q_ACTION_ORDER),
        "target_unit": ENTRY_FITTED_Q_TARGET_UNIT,
        "target_economics": "gross_spread_inclusive_research_only",
        "gamma": ENTRY_FITTED_Q_GAMMA,
        "long_target": (
            "stop_gradient(frozen_train_exit_target_model_"
            "V_at_first_authoritative_post_fill_long_state)"
        ),
        "short_target": (
            "stop_gradient(frozen_train_exit_target_model_"
            "V_at_first_authoritative_post_fill_short_state)"
        ),
        "flat_target_bps": 0.0,
        "long_short_transition_reward_bps": 0.0,
        "flat_terminal": True,
        "side_mask_source": "exact_hash_bound_unified_exit_episode_pack",
        "target_snapshot_fitted_splits": ["train"],
        "validation_or_test_updates_target_snapshot": False,
        "pathwise_hindsight_target": False,
        "decision": "unique_argmax(entry_action_q_bps_over_valid_actions)",
        "exact_q_ties": "fail_closed",
        "production_authority_ready": False,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def require_entry_fitted_q_contract(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected = entry_fitted_q_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_ENTRY_FITTED_Q_CONTRACT_INVALID]")
    return expected


def entry_fitted_q_production_economics_readiness() -> dict[str, Any]:
    """Declare why the current fitted-Q surface cannot authorize trading."""

    payload = {
        "schema_version": ENTRY_FITTED_Q_PRODUCTION_ECONOMICS_SCHEMA_VERSION,
        "production_authority_ready": False,
        "same_close_execution_is_causal": False,
        "causal_next_executable_bid_ask_bound": False,
        "immutable_net_cost_artifact_bound": False,
        "cost_components_required": [
            "spread_embedded_in_executable_bid_ask",
            "commission",
            "execution_slippage",
            "financing_or_swap",
        ],
        "financing_or_swap_broker_semantics_bound": False,
        "elapsed_wall_clock_state_bound": False,
        "source_gap_classification_bound": False,
        "variable_or_economic_terminal_bound": False,
        "fixed_512_capacity_forced_terminal_present": True,
        "portfolio_oos_replay_bound": False,
        "overlapping_trade_capital_constraints_bound": False,
        "immutable_gap_audit_artifact_bound": False,
        "gap_audit_source_manifest_sha256": None,
        "gap_audit_metrics": None,
        "gross_research_training_allowed": True,
        "gross_q_production_decision_authority_allowed": False,
        "candidate_ready_allowed": False,
        "bundle_serving_admission_allowed": False,
        "edge_claim_allowed": False,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def require_entry_fitted_q_production_economics_readiness(
    value: Mapping[str, Any], *, context: str, require_ready: bool
) -> dict[str, Any]:
    expected = entry_fitted_q_production_economics_readiness()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_ENTRY_FITTED_Q_ECONOMICS_INVALID]")
    if require_ready and expected["production_authority_ready"] is not True:
        raise RuntimeError(f"[{context}_ENTRY_FITTED_Q_ECONOMICS_NOT_READY]")
    return expected


def require_entry_fitted_q_iteration_state(
    value: Mapping[str, Any],
    *,
    exit_fitted_q_iteration_state: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ITERATION_FIELDS:
        raise RuntimeError(f"[{context}_ENTRY_FITTED_Q_ITERATION_SCHEMA_INVALID]")
    observed = dict(value)
    exit_state = require_unified_exit_fitted_q_iteration_state(
        exit_fitted_q_iteration_state,
        context=f"{context}_EXIT_TEACHER",
    )
    if (
        observed["schema_version"]
        != ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION
        or isinstance(observed["iteration_index"], bool)
        or not isinstance(observed["iteration_index"], int)
        or int(observed["iteration_index"]) < 0
        or int(observed["iteration_index"])
        != int(exit_state["iteration_index"])
        or observed["target_updated_from_val_or_test"] is not False
        or observed["exit_fitted_q_iteration_state_sha256"]
        != _canonical_sha256(exit_state)
        or observed["exit_target_model_state_sha256"]
        != exit_state["target_model_state_sha256"]
        or observed["train_split_sha256"] != exit_state["train_split_sha256"]
        or observed["train_fold_sha256"] != exit_state["train_fold_sha256"]
        or observed["source_lineage_sha256"]
        != exit_state["source_lineage_sha256"]
        or observed["normalization_sha256"]
        != exit_state["normalization_sha256"]
        or observed["exit_fitted_q_contract"] != unified_exit_fitted_q_contract()
    ):
        raise RuntimeError(f"[{context}_ENTRY_FITTED_Q_ITERATION_BINDING_INVALID]")
    for field in (
        "entry_target_model_state_sha256",
        "exit_target_model_state_sha256",
        "exit_fitted_q_iteration_state_sha256",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "normalization_sha256",
    ):
        _require_sha256(observed[field], label=field.upper())
    require_entry_fitted_q_contract(
        observed["entry_fitted_q_contract"], context=context
    )
    return observed


def entry_fill_binding_sha256(
    *,
    entry_row_index: int,
    episode_pack_sha256: str,
    first_exit_state_time_ns: int,
    exit_entry_bid_ask: Any,
) -> str:
    """Bind the exact counterfactual side fills used by the Exit episode."""

    if (
        isinstance(entry_row_index, bool)
        or not isinstance(entry_row_index, int)
        or entry_row_index < 0
        or isinstance(first_exit_state_time_ns, bool)
        or not isinstance(first_exit_state_time_ns, (int, np.integer))
        or int(first_exit_state_time_ns) <= 0
    ):
        raise RuntimeError("ENTRY_FITTED_Q_FILL_IDENTITY_INVALID")
    episode_sha = _require_sha256(
        episode_pack_sha256, label="EPISODE_PACK"
    )
    quotes = np.ascontiguousarray(exit_entry_bid_ask, dtype=np.dtype("<f4"))
    if (
        quotes.shape != (2, 2)
        or not np.isfinite(quotes).all()
        or np.any(quotes <= 0.0)
        or np.any(quotes[:, 1] <= quotes[:, 0])
    ):
        raise RuntimeError("ENTRY_FITTED_Q_FILL_QUOTES_INVALID")
    digest = hashlib.sha256()
    digest.update(b"gx1_entry_fitted_q_fill_binding_v1\0")
    digest.update(np.asarray([entry_row_index], dtype="<i8").tobytes())
    digest.update(np.asarray([first_exit_state_time_ns], dtype="<i8").tobytes())
    digest.update(episode_sha.encode("ascii"))
    digest.update(quotes.tobytes())
    return digest.hexdigest()


def build_entry_fitted_q_targets(
    *,
    frozen_exit_first_state_values_bps: torch.Tensor,
    exit_side_valid_mask: torch.Tensor,
    episode_pack_sha256: Sequence[str | None],
    fill_binding_sha256: Sequence[str | None],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build detached `[LONG, SHORT, FLAT]` raw-bps targets and masks."""

    values = frozen_exit_first_state_values_bps
    side_valid = exit_side_valid_mask
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 2
        or int(values.shape[1]) != 2
        or not isinstance(side_valid, torch.Tensor)
        or side_valid.dtype != torch.bool
        or tuple(side_valid.shape) != tuple(values.shape)
        or not bool(torch.isfinite(values).all().item())
    ):
        raise RuntimeError("ENTRY_FITTED_Q_TARGET_INPUT_INVALID")
    rows = int(values.shape[0])
    episodes = list(episode_pack_sha256)
    fills = list(fill_binding_sha256)
    if len(episodes) != rows or len(fills) != rows:
        raise RuntimeError("ENTRY_FITTED_Q_TARGET_BINDING_ROWS_INVALID")
    canonical_bindings: list[dict[str, Any]] = []
    for row, (episode_sha, fill_sha) in enumerate(zip(episodes, fills)):
        has_side = bool(side_valid[row].any().item())
        if has_side:
            canonical_bindings.append(
                {
                    "row": row,
                    "episode_pack_sha256": _require_sha256(
                        episode_sha, label="EPISODE_PACK"
                    ),
                    "fill_binding_sha256": _require_sha256(
                        fill_sha, label="FILL_BINDING"
                    ),
                    "side_valid": side_valid[row].tolist(),
                }
            )
        elif episode_sha is not None or fill_sha is not None:
            raise RuntimeError("ENTRY_FITTED_Q_INELIGIBLE_ROW_HAS_BINDING")
    targets = values.new_zeros((rows, 3))
    targets[:, :2] = values.detach()
    action_valid = torch.zeros(
        (rows, 3), dtype=torch.bool, device=values.device
    )
    action_valid[:, :2] = side_valid
    action_valid[:, 2] = True
    if targets.requires_grad or not bool(torch.isfinite(targets).all().item()):
        raise RuntimeError("ENTRY_FITTED_Q_TARGET_NOT_FROZEN")
    binding = {
        "schema_version": "gx1_entry_fitted_q_batch_binding_v1",
        "row_count": rows,
        "eligible_side_cells": int(side_valid.sum().item()),
        "flat_valid_rows": rows,
        "episode_fill_bindings": canonical_bindings,
        "action_equivalence_mask": entry_action_equivalence_mask(
            target_q_bps=targets,
            action_valid_mask=action_valid,
        ).cpu().tolist(),
    }
    binding["binding_sha256"] = _canonical_sha256(binding)
    return targets.detach(), action_valid, binding


def entry_action_equivalence_mask(
    *,
    target_q_bps: torch.Tensor,
    action_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return every exactly target-equivalent optimal action; never break ties."""

    if (
        not isinstance(target_q_bps, torch.Tensor)
        or target_q_bps.ndim != 2
        or int(target_q_bps.shape[1]) != 3
        or not isinstance(action_valid_mask, torch.Tensor)
        or action_valid_mask.dtype != torch.bool
        or tuple(action_valid_mask.shape) != tuple(target_q_bps.shape)
        or not bool(torch.isfinite(target_q_bps).all().item())
        or not bool(action_valid_mask[:, 2].all().item())
        or not bool(action_valid_mask.any(dim=1).all().item())
    ):
        raise RuntimeError("ENTRY_FITTED_Q_EQUIVALENCE_INPUT_INVALID")
    masked = target_q_bps.masked_fill(~action_valid_mask, float("-inf"))
    maxima = masked.max(dim=1, keepdim=True).values
    equivalent = action_valid_mask & target_q_bps.eq(maxima)
    if not bool(equivalent.any(dim=1).all().item()):
        raise RuntimeError("ENTRY_FITTED_Q_EQUIVALENCE_EMPTY")
    return equivalent.detach()


def replay_entry_fitted_q_policy(
    *,
    predicted_q_bps: Any,
    action_valid_mask: Any,
) -> np.ndarray:
    q = np.asarray(predicted_q_bps, dtype=np.float64)
    valid = np.asarray(action_valid_mask, dtype=np.bool_)
    if (
        q.ndim != 2
        or q.shape[1] != 3
        or valid.shape != q.shape
        or not np.isfinite(q).all()
        or not valid[:, 2].all()
        or np.any(~valid.any(axis=1))
    ):
        raise RuntimeError("ENTRY_FITTED_Q_POLICY_INPUT_INVALID")
    masked = np.where(valid, q, -np.inf)
    winners = masked == np.max(masked, axis=1, keepdims=True)
    if np.any(winners.sum(axis=1) != 1):
        raise RuntimeError("ENTRY_FITTED_Q_POLICY_TIED_ACTION")
    return np.argmax(masked, axis=1).astype(np.int64)


__all__ = (
    "ENTRY_FITTED_Q_ACTION_ORDER",
    "ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION",
    "ENTRY_FITTED_Q_SCHEMA_VERSION",
    "build_entry_fitted_q_targets",
    "entry_action_equivalence_mask",
    "entry_fill_binding_sha256",
    "entry_fitted_q_contract",
    "replay_entry_fitted_q_policy",
    "require_entry_fitted_q_contract",
    "require_entry_fitted_q_iteration_state",
)
