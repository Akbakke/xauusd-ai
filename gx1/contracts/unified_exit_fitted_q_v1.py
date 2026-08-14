"""Causal fitted-Q targets for the unified Exit policy.

Only the executable EXIT_NOW reward is observed directly.  HOLD supervision
is bootstrapped from one immutable, stop-gradient target-network snapshot at
the next causal state.  Future realized quotes are never maximized pathwise to
construct a training label; the finite-path hindsight optimum is a diagnostic
upper bound only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION = "gx1_unified_exit_fitted_q_v1"
UNIFIED_EXIT_FITTED_Q_GAMMA = 1.0
UNIFIED_EXIT_FITTED_Q_OPERATOR = "frozen_target_network_max"
UNIFIED_EXIT_FITTED_Q_TARGET_UNIT = "raw_bps"
UNIFIED_EXIT_FIRST_STATE_VALUE_SCHEMA_VERSION = (
    "gx1_unified_exit_first_state_target_value_v1"
)
UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION = (
    "gx1_unified_exit_fitted_q_iteration_state_v1"
)
_ITERATION_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "iteration_index",
        "target_model_state_sha256",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "normalization_sha256",
        "fitted_q_contract",
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


def unified_exit_fitted_q_contract() -> dict[str, Any]:
    payload = {
        "schema_version": UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
        "action_order": ["HOLD", "EXIT_NOW"],
        "target_unit": UNIFIED_EXIT_FITTED_Q_TARGET_UNIT,
        "gamma": UNIFIED_EXIT_FITTED_Q_GAMMA,
        "intermediate_hold_reward_bps": 0.0,
        "exit_target": "current_executable_trade_pnl_bps",
        "hold_target": "stop_gradient(max_valid_q_target_at_next_causal_state)",
        "entry_bridge": (
            "stop_gradient(max_valid_q_target_at_first_authoritative_"
            "post_fill_exit_state_per_side)"
        ),
        "operator": UNIFIED_EXIT_FITTED_Q_OPERATOR,
        "target_snapshot_update": "explicit_fitted_q_iteration_boundary_only",
        "target_snapshot_fitted_splits": ["train"],
        "validation_or_test_updates_target_snapshot": False,
        "terminal_valid_actions": ["EXIT_NOW"],
        "pathwise_hindsight_max_is_training_target": False,
        "pathwise_hindsight_role": "diagnostic_upper_bound_only",
        "double_q": {
            "active": False,
            "status": "separate_measured_variant_not_hidden_default",
        },
        "required_training_state_bindings": [
            "iteration_index",
            "target_model_state_sha256",
            "train_split_sha256",
            "train_fold_sha256",
            "source_lineage_sha256",
            "normalization_sha256",
        ],
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def require_unified_exit_fitted_q_contract(
    value: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    expected = unified_exit_fitted_q_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"{context}_UNIFIED_EXIT_FITTED_Q_CONTRACT_INVALID")
    return expected


def require_unified_exit_fitted_q_iteration_state(
    value: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ITERATION_STATE_FIELDS:
        raise RuntimeError(f"{context}_UNIFIED_EXIT_FITTED_Q_STATE_INVALID")
    observed = dict(value)
    if (
        observed["schema_version"]
        != UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION
        or isinstance(observed["iteration_index"], bool)
        or not isinstance(observed["iteration_index"], int)
        or observed["iteration_index"] < 0
        or observed["target_updated_from_val_or_test"] is not False
    ):
        raise RuntimeError(f"{context}_UNIFIED_EXIT_FITTED_Q_STATE_INVALID")
    for key in (
        "target_model_state_sha256",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "normalization_sha256",
    ):
        digest = observed[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(
                f"{context}_UNIFIED_EXIT_FITTED_Q_STATE_INVALID"
            )
    require_unified_exit_fitted_q_contract(
        observed["fitted_q_contract"], context=context
    )
    return observed


def build_unified_exit_fitted_q_targets(
    *,
    frozen_target_q_bps: torch.Tensor,
    exit_now_reward_bps: torch.Tensor,
    action_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
    terminal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one stop-gradient Bellman target tensor.

    Shapes are ``[B, side, state, action]`` for Q/mask and
    ``[B, side, state]`` for reward/state/terminal.  State capacity is not
    fixed here; explicit masks and terminal rows own episode length.
    """

    if frozen_target_q_bps.ndim != 4 or frozen_target_q_bps.shape[-1] != 2:
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_TARGET_Q_SHAPE_INVALID")
    expected_state_shape = frozen_target_q_bps.shape[:-1]
    if (
        tuple(exit_now_reward_bps.shape) != tuple(expected_state_shape)
        or tuple(state_valid_mask.shape) != tuple(expected_state_shape)
        or tuple(terminal_mask.shape) != tuple(expected_state_shape)
        or tuple(action_valid_mask.shape) != tuple(frozen_target_q_bps.shape)
        or action_valid_mask.dtype != torch.bool
        or state_valid_mask.dtype != torch.bool
        or terminal_mask.dtype != torch.bool
        or not bool(torch.isfinite(frozen_target_q_bps).all().item())
        or not bool(torch.isfinite(exit_now_reward_bps).all().item())
    ):
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_INPUT_INVALID")
    if frozen_target_q_bps.shape[-2] < 1:
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_EMPTY_EPISODE")
    if bool((terminal_mask & ~state_valid_mask).any().item()):
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_TERMINAL_MASK_INVALID")
    if bool((action_valid_mask[..., 1] != state_valid_mask).any().item()):
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_EXIT_ACTION_MASK_INVALID")
    if bool((action_valid_mask[..., 0] != (state_valid_mask & ~terminal_mask)).any().item()):
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_HOLD_ACTION_MASK_INVALID")

    target_q = exit_now_reward_bps.new_zeros(frozen_target_q_bps.shape)
    target_q[..., 1] = exit_now_reward_bps
    if frozen_target_q_bps.shape[-2] > 1:
        next_q = frozen_target_q_bps.detach()[..., 1:, :]
        next_valid = action_valid_mask[..., 1:, :]
        if bool((~next_valid.any(dim=-1) & state_valid_mask[..., 1:]).any().item()):
            raise RuntimeError("UNIFIED_EXIT_FITTED_Q_NEXT_ACTION_MASK_EMPTY")
        next_value = next_q.masked_fill(~next_valid, -torch.inf).amax(dim=-1)
        hold_rows = action_valid_mask[..., :-1, 0]
        if not bool(torch.isfinite(next_value[hold_rows]).all().item()):
            raise RuntimeError("UNIFIED_EXIT_FITTED_Q_NEXT_VALUE_NONFINITE")
        target_q[..., :-1, 0] = torch.where(
            hold_rows,
            next_value,
            torch.zeros_like(next_value),
        )
    if not bool(torch.isfinite(target_q[action_valid_mask]).all().item()):
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_TARGET_NONFINITE")
    return target_q.detach(), action_valid_mask


def unified_exit_first_state_side_values(
    *,
    frozen_target_q_bps: torch.Tensor,
    action_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen target-policy value at the first Exit state.

    This is the only supported bridge for a later Entry Bellman target:
    LONG/SHORT transition with zero initial reward to each side's first
    authoritative post-fill Exit state.  Snapshot/fold/source lineage remains
    owned by the fitted-Q training-state envelope that produced ``target_q``.
    """

    if (
        frozen_target_q_bps.ndim != 4
        or frozen_target_q_bps.shape[-1] != 2
        or tuple(action_valid_mask.shape) != tuple(frozen_target_q_bps.shape)
        or tuple(state_valid_mask.shape) != tuple(frozen_target_q_bps.shape[:-1])
        or action_valid_mask.dtype != torch.bool
        or state_valid_mask.dtype != torch.bool
        or frozen_target_q_bps.shape[-2] < 1
        or not bool(torch.isfinite(frozen_target_q_bps).all().item())
        or not bool(state_valid_mask[..., 0].all().item())
        or not bool(action_valid_mask[..., 0, :].any(dim=-1).all().item())
    ):
        raise RuntimeError("UNIFIED_EXIT_FIRST_STATE_VALUE_INPUT_INVALID")
    first_q = frozen_target_q_bps.detach()[..., 0, :]
    first_valid = action_valid_mask[..., 0, :]
    values = first_q.masked_fill(~first_valid, -torch.inf).amax(dim=-1)
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("UNIFIED_EXIT_FIRST_STATE_VALUE_NONFINITE")
    return values.detach()


def replay_unified_exit_fitted_q_policy(
    *,
    predicted_q_bps: Any,
    action_valid_mask: Any,
    exit_now_reward_bps: Any,
) -> dict[str, Any]:
    """Replay unique valid Q argmax until executable EXIT_NOW."""

    q = np.asarray(predicted_q_bps, dtype=np.float64)
    valid = np.asarray(action_valid_mask, dtype=np.bool_)
    rewards = np.asarray(exit_now_reward_bps, dtype=np.float64)
    if (
        q.ndim != 2
        or q.shape[1] != 2
        or valid.shape != q.shape
        or rewards.shape != (q.shape[0],)
        or q.shape[0] < 1
        or not np.isfinite(q).all()
        or not np.isfinite(rewards).all()
        or not valid[-1, 1]
        or valid[-1, 0]
    ):
        raise RuntimeError("UNIFIED_EXIT_FITTED_Q_POLICY_REPLAY_INPUT_INVALID")
    actions: list[int] = []
    for state in range(q.shape[0]):
        valid_indices = np.flatnonzero(valid[state])
        if valid_indices.size == 0:
            raise RuntimeError("UNIFIED_EXIT_FITTED_Q_POLICY_ACTION_MASK_EMPTY")
        values = q[state, valid_indices]
        if np.count_nonzero(values == np.max(values)) != 1:
            raise RuntimeError("UNIFIED_EXIT_FITTED_Q_POLICY_TIED_ACTION")
        action = int(valid_indices[int(np.argmax(values))])
        actions.append(action)
        if action == 1:
            return {
                "exit_state_index": state,
                "action_indices": actions,
                "realized_executable_pnl_bps": float(rewards[state]),
                "terminal_forced": state == q.shape[0] - 1,
            }
    raise RuntimeError("UNIFIED_EXIT_FITTED_Q_POLICY_DID_NOT_TERMINATE")


def build_unified_exit_first_state_value_envelope(
    *,
    entry_row_indices: list[int] | tuple[int, ...],
    frozen_target_q_bps: torch.Tensor,
    action_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
    fitted_q_iteration_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash-bind first-state LONG/SHORT values to their frozen TRAIN teacher."""

    values = unified_exit_first_state_side_values(
        frozen_target_q_bps=frozen_target_q_bps,
        action_valid_mask=action_valid_mask,
        state_valid_mask=state_valid_mask,
    )
    indices = list(entry_row_indices)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or len(indices) != int(values.shape[0])
        or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
    ):
        raise RuntimeError("UNIFIED_EXIT_FIRST_STATE_VALUE_ENVELOPE_INPUT_INVALID")
    iteration_state = require_unified_exit_fitted_q_iteration_state(
        fitted_q_iteration_state,
        context="UNIFIED_EXIT_FIRST_STATE_VALUE",
    )
    array = np.ascontiguousarray(
        values.detach().cpu().to(torch.float32).numpy(), dtype=np.dtype("<f4")
    )
    envelope = {
        "schema_version": UNIFIED_EXIT_FIRST_STATE_VALUE_SCHEMA_VERSION,
        "entry_row_indices": indices,
        "side_order": ["long", "short"],
        "value_unit": "raw_bps",
        "transition_reward_bps": 0.0,
        "values_bps": array.tolist(),
        "values_float32_le_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "iteration_index": iteration_state["iteration_index"],
        "fitted_q_iteration_state_sha256": _canonical_sha256(
            iteration_state
        ),
        "target_model_state_sha256": iteration_state[
            "target_model_state_sha256"
        ],
        "train_split_sha256": iteration_state["train_split_sha256"],
        "train_fold_sha256": iteration_state["train_fold_sha256"],
        "source_lineage_sha256": iteration_state[
            "source_lineage_sha256"
        ],
        "normalization_sha256": iteration_state[
            "normalization_sha256"
        ],
        "target_values_require_grad": False,
    }
    envelope["envelope_sha256"] = _canonical_sha256(envelope)
    return envelope


__all__ = (
    "UNIFIED_EXIT_FITTED_Q_GAMMA",
    "UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION",
    "UNIFIED_EXIT_FITTED_Q_OPERATOR",
    "UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION",
    "UNIFIED_EXIT_FIRST_STATE_VALUE_SCHEMA_VERSION",
    "build_unified_exit_first_state_value_envelope",
    "build_unified_exit_fitted_q_targets",
    "require_unified_exit_fitted_q_contract",
    "require_unified_exit_fitted_q_iteration_state",
    "replay_unified_exit_fitted_q_policy",
    "unified_exit_first_state_side_values",
    "unified_exit_fitted_q_contract",
)
