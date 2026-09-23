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


UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION = "gx1_unified_exit_fitted_q_v4"
UNIFIED_EXIT_FITTED_Q_GAMMA = 1.0
UNIFIED_EXIT_INTERMEDIATE_HOLD_REWARD_BPS = 0.0
UNIFIED_EXIT_FITTED_Q_OPERATOR = "frozen_target_network_max"
UNIFIED_EXIT_FITTED_Q_TARGET_UNIT = "raw_bps"
UNIFIED_EXIT_FIRST_STATE_VALUE_SCHEMA_VERSION = (
    "gx1_unified_exit_first_state_target_value_v1"
)
UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION = (
    "gx1_unified_exit_fitted_q_iteration_state_v2"
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
        "target_refresh_interval_optimizer_steps",
        "target_refreshes_completed",
    }
)


def unified_exit_target_refresh_interval_optimizer_steps(
    steps_per_epoch: int,
) -> int:
    """Derive the intra-epoch target-refresh cadence from the episode depth.

    ``Q_hold`` bootstraps exactly one causal state per target-snapshot
    refresh, so after k refreshes the bootstrapped value spans at most the
    first k of ``UNIFIED_EXIT_MAX_PATH_BARS`` states (a strictly downward
    truncation that penalizes only LONG/SHORT against the exact FLAT anchor).
    Refreshing every ``floor(steps_per_epoch / UNIFIED_EXIT_MAX_PATH_BARS)``
    optimizer steps guarantees at least ``UNIFIED_EXIT_MAX_PATH_BARS``
    refreshes within a single epoch, so the value function can span the whole
    episode horizon before the first checkpoint-selection judgment. Both
    inputs are named constants or declared recipe geometry; nothing here is a
    tuned magnitude. An epoch shorter than the path depth refreshes every
    step (interval 1), the fastest cadence the step clock admits.
    """

    if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int):
        raise RuntimeError(
            "[UNIFIED_EXIT_TARGET_REFRESH_STEPS_PER_EPOCH_INVALID]"
        )
    if steps_per_epoch < 1:
        raise RuntimeError(
            "[UNIFIED_EXIT_TARGET_REFRESH_STEPS_PER_EPOCH_INVALID]"
        )
    from gx1.models.entry_v10.direction_decision_contract import (
        UNIFIED_EXIT_MAX_PATH_BARS,
    )

    return max(1, int(steps_per_epoch) // int(UNIFIED_EXIT_MAX_PATH_BARS))


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
        "intermediate_hold_reward_bps": UNIFIED_EXIT_INTERMEDIATE_HOLD_REWARD_BPS,
        "exit_target": "current_executable_trade_pnl_bps",
        "hold_target": "stop_gradient(max_valid_q_target_at_next_causal_state)",
        "entry_bridge": (
            "stop_gradient(frozen_policy_n_step_exit_reward_or_"
            "observed_window_continuation_value_per_side)"
        ),
        "operator": UNIFIED_EXIT_FITTED_Q_OPERATOR,
        "target_snapshot_update": (
            "derived_intra_epoch_interval_steps_per_epoch_over_max_path_bars"
        ),
        "bounded_initialized_smoke_snapshot_exception": (
            "explicit_recipe_freeze_initial_teacher_sets_interval_to_total_steps_plus_one"
        ),
        "target_snapshot_fitted_splits": ["train"],
        "validation_or_test_updates_target_snapshot": False,
        "compute_window_end_valid_actions": ["HOLD", "EXIT_NOW"],
        "unobserved_next_state_hold_supervision": "masked_not_zero_or_forced_exit",
        "maximum_trade_duration": None,
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
        or isinstance(observed["target_refresh_interval_optimizer_steps"], bool)
        or not isinstance(
            observed["target_refresh_interval_optimizer_steps"], int
        )
        or observed["target_refresh_interval_optimizer_steps"] < 1
        or isinstance(observed["target_refreshes_completed"], bool)
        or not isinstance(observed["target_refreshes_completed"], int)
        or observed["target_refreshes_completed"] < 0
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
    # A HOLD at the last observed state has no next-state target in this
    # window. Exclude only that unknown label from the loss. Its action stays
    # valid in the policy and its frozen Q remains usable by the prior state.
    supervision_mask = action_valid_mask.clone()
    supervision_mask[..., -1, 0] = False
    return target_q.detach(), supervision_mask


def unified_exit_first_state_side_values(
    *,
    frozen_target_q_bps: torch.Tensor,
    action_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen target-policy value at the first Exit state.

    This is the diagnostic one-step view. Native Entry training uses the
    frozen-policy n-step bridge below. Snapshot/fold/source lineage remains
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



def unified_exit_frozen_policy_n_step_side_values(
    *,
    frozen_target_q_bps: torch.Tensor,
    exit_now_reward_bps: torch.Tensor,
    action_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the frozen policy through the available observation window.

    Use the observed reward at its first unique EXIT. At an exact action tie
    or an open window boundary, bootstrap the same frozen model without
    selecting an action or inventing a terminal. Intermediate HOLD reward is
    zero and gamma is one under this contract. Future price extrema are
    never used to choose the exit. Returned Entry labels are stop-gradient.
    """

    unified_exit_first_state_side_values(
        frozen_target_q_bps=frozen_target_q_bps,
        action_valid_mask=action_valid_mask,
        state_valid_mask=state_valid_mask,
    )
    if (
        exit_now_reward_bps.shape != state_valid_mask.shape
        or not bool(torch.isfinite(exit_now_reward_bps).all().item())
        or not torch.equal(action_valid_mask[..., 1], state_valid_mask)
        or bool((action_valid_mask[..., 0] & ~state_valid_mask).any().item())
        or UNIFIED_EXIT_FITTED_Q_GAMMA != 1.0
        or UNIFIED_EXIT_INTERMEDIATE_HOLD_REWARD_BPS != 0.0
    ):
        raise RuntimeError("UNIFIED_EXIT_POLICY_N_STEP_INPUT_INVALID")
    count = state_valid_mask.sum(dim=-1)
    clock = torch.arange(
        frozen_target_q_bps.shape[-2], device=frozen_target_q_bps.device
    )
    if not torch.equal(state_valid_mask, clock < count.unsqueeze(-1)):
        raise RuntimeError("UNIFIED_EXIT_POLICY_N_STEP_STATE_PREFIX_INVALID")

    q = frozen_target_q_bps.detach().masked_fill(~action_valid_mask, -torch.inf)
    exit_unique = state_valid_mask & (q[..., 1] > q[..., 0])
    tied = state_valid_mask & action_valid_mask.all(dim=-1) & (q[..., 0] == q[..., 1])
    stop = exit_unique | tied
    first_stop = torch.where(stop, clock, clock.numel()).amin(dim=-1)
    stop_index = torch.minimum(first_stop, count - 1).unsqueeze(-1)
    exits_here = exit_unique.gather(-1, stop_index).squeeze(-1)
    observed_exit = exit_now_reward_bps.detach().gather(-1, stop_index).squeeze(-1)
    continuation = q.amax(dim=-1).gather(-1, stop_index).squeeze(-1)
    values = torch.where(exits_here, observed_exit, continuation)
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("UNIFIED_EXIT_POLICY_N_STEP_VALUE_NONFINITE")
    return values.detach()


def replay_unified_exit_fitted_q_policy(
    *,
    predicted_q_bps: Any,
    action_valid_mask: Any,
    exit_now_reward_bps: Any,
) -> dict[str, Any]:
    """Replay learned actions; preserve a still-open position at a window end.

    marked_executable_pnl_bps includes both closed and right-censored paths.
    Realized PnL and exit_state_index are absent for an unclosed position.
    """

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
        or not valid.all()
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
                "marked_executable_pnl_bps": float(rewards[state]),
                "position_closed": True,
                "right_censored": False,
                "terminal_forced": False,
            }
    return {
        "exit_state_index": None,
        "last_observed_state_index": q.shape[0] - 1,
        "action_indices": actions,
        "realized_executable_pnl_bps": None,
        "marked_executable_pnl_bps": float(rewards[-1]),
        "position_closed": False,
        "right_censored": True,
        "terminal_forced": False,
    }


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
    "unified_exit_frozen_policy_n_step_side_values",
    "unified_exit_fitted_q_contract",
)
