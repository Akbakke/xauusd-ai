"""Opt-in CPU prototype for Q_mu; not wired to the native optimality target.

The stationary reference exits with probability 1/120 at each decision.
Its expectation is evaluated analytically. There is no random action draw,
maximum holding time, hindsight action choice, or new model head.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch


def reference_policy_contract() -> dict[str, Any]:
    """One fixed candidate, not a configurable policy/parameter search."""
    policy = {
        "schema_version": "gx1_exit_reference_policy_v1",
        "value_semantics": "Q_mu_stationary_reference_after_first_action",
        "value_coordinates": "advantage_over_executable_liquidation_bps",
        "hold_probability_numerator": 119,
        "hold_probability_denominator": 120,
        "maximum_observed_backup_steps": 120,
        "terminal_policy": "exit_if_economic_terminal",
        "observation_boundary_policy": "preserve_valid_teacher_bootstrap",
        "selection": "analytical_expectation_no_random_action_draw",
    }
    policy["policy_sha256"] = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return policy


def require_reference_policy_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = reference_policy_contract()
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_POLICY_CONTRACT_INVALID")
    observed = dict(value)
    claimed = observed.pop("policy_sha256", None)
    try:
        digest = hashlib.sha256(json.dumps(
            observed, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_POLICY_CONTRACT_INVALID") from exc
    if digest != claimed or digest != expected["policy_sha256"]:
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_POLICY_CONTRACT_INVALID")
    return expected


@torch.no_grad()
def build_reference_policy_hold_targets(
    *,
    policy: Mapping[str, Any],
    hold_reward_bps: torch.Tensor,
    elapsed_wall_clock_gamma: torch.Tensor,
    transition_available_mask: torch.Tensor,
    successor_terminal_mask: torch.Tensor,
    boundary_action_q_bps: torch.Tensor,
    boundary_action_valid_mask: torch.Tensor,
    boundary_right_censored_mask: torch.Tensor,
) -> dict[str, Any]:
    """Return Q_mu HOLD targets and their observable/bootstrap decomposition.

    Rows are independent LONG/SHORT pairs. Rewards have shape [N,H,2];
    availability is a nonempty prefix of [N,H], H<=120. Boundary Q is for
    the successor of each row's last available transition, NOT padded H.
    An economic terminal stops only that side. Censoring never zeros a valid
    boundary value. Invalid HOLD values may be NaN and are never consumed.

    r0 + p*g0*r1 + ... + p**n*prod(gamma)*Q_mu(boundary,HOLD).
    This evaluates the stated reference policy, not a Bellman optimality max.
    """
    contract = require_reference_policy_contract(policy)
    tensors = (
        hold_reward_bps, elapsed_wall_clock_gamma, transition_available_mask,
        successor_terminal_mask, boundary_action_q_bps,
        boundary_action_valid_mask, boundary_right_censored_mask,
    )
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_TRACE_INVALID")
    rewards = hold_reward_bps
    if rewards.ndim != 3 or rewards.shape[2] != 2:
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_TRACE_INVALID")
    rows, steps, _ = rewards.shape
    gamma = elapsed_wall_clock_gamma
    available = transition_available_mask
    terminal = successor_terminal_mask
    q, mask, censored = (boundary_action_q_bps, boundary_action_valid_mask,
                         boundary_right_censored_mask)
    if (
        rows < 1 or not 1 <= steps <= contract["maximum_observed_backup_steps"]
        or rewards.dtype != torch.float32 or gamma.dtype != torch.float32
        or q.dtype != torch.float32
        or gamma.shape != (rows, steps) or available.shape != (rows, steps)
        or terminal.shape != (rows, steps, 2)
        or q.shape != (rows, 2, 2) or mask.shape != q.shape
        or censored.shape != (rows, 2)
        or any(value.dtype != torch.bool for value in (available, terminal, mask, censored))
        or any(value.device != rewards.device for value in tensors)
        or not bool(available[:, 0].all())
        or bool((available[:, 1:] & ~available[:, :-1]).any())
        or not bool(torch.isfinite(rewards).all())
        or not bool(torch.isfinite(gamma).all())
        or bool(((gamma <= 0) | (gamma > 1)).any())
    ):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_TRACE_INVALID")
    # A side cannot resurrect after an economic terminal within observed data.
    if bool((terminal[:, :-1] & ~terminal[:, 1:] & available[:, 1:, None]).any()):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_TERMINAL_INVALID")
    lengths = available.sum(dim=1)
    last_terminal = terminal[torch.arange(rows, device=rewards.device), lengths - 1]
    if (
        not bool(mask[..., 1].all())
        or not torch.equal(mask[..., 0], ~last_terminal)
        or bool((censored & last_terminal).any())
        or not bool(torch.isfinite(q[mask]).all())
        or bool((q[..., 1] != 0).any())
    ):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_BOUNDARY_INVALID")
    p = contract["hold_probability_numerator"] / contract["hold_probability_denominator"]
    observed = torch.zeros((rows, 2), dtype=rewards.dtype, device=rewards.device)
    weight = torch.ones_like(observed)
    for offset in range(steps):
        present = available[:, offset, None]
        observed += torch.where(present, weight * rewards[:, offset], 0.0)
        next_weight = weight * gamma[:, offset, None] * p
        next_weight = next_weight.masked_fill(terminal[:, offset], 0.0)
        weight = torch.where(present, next_weight, weight)
    bootstrap = weight * q[..., 0].masked_fill(~mask[..., 0], 0.0)
    target = observed + bootstrap
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_TARGET_INVALID")
    return {
        "schema_version": "gx1_exit_reference_policy_hold_targets_v1",
        "value_semantics": contract["value_semantics"],
        "reference_policy_sha256": contract["policy_sha256"],
        "hold_target_bps": target,
        "exit_now_target_bps": torch.zeros_like(target),
        "observed_reward_component_bps": observed,
        "bootstrap_component_bps": bootstrap,
        "boundary_bootstrap_weight": weight,
        "observed_backup_step_count": lengths,
    }


@torch.no_grad()
def reference_policy_state_values(*, policy: Mapping[str, Any],
                                  action_q_bps: torch.Tensor,
                                  action_valid_mask: torch.Tensor) -> torch.Tensor:
    """V_mu at the current state; never choose an action using a future return.

    HOLD Q_mu may be an observed return sample. Average the declared causal
    action probabilities before regression. Terminal sides take EXIT=0.
    """
    contract = require_reference_policy_contract(policy)
    q, mask = action_q_bps, action_valid_mask
    if (not isinstance(q, torch.Tensor) or not isinstance(mask, torch.Tensor)
            or q.ndim != 3 or q.shape[0] < 1 or q.shape[1:] != (2, 2)
            or q.dtype != torch.float32 or mask.dtype != torch.bool
            or mask.shape != q.shape or mask.device != q.device
            or not bool(mask[..., 1].all()) or not bool(torch.isfinite(q[mask]).all())
            or not bool((q[..., 1] == 0).all())):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_STATE_VALUE_INVALID")
    p = contract["hold_probability_numerator"] / contract["hold_probability_denominator"]
    return p * q[..., 0].masked_fill(~mask[..., 0], 0.0)
