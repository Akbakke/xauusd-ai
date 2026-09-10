from __future__ import annotations

import pytest
import torch

from gx1.contracts.unified_exit_fitted_q_v1 import (
    build_unified_exit_fitted_q_targets,
    build_unified_exit_first_state_value_envelope,
    require_unified_exit_unbounded_training_readiness,
    unified_exit_first_state_side_values,
    unified_exit_fitted_q_contract,
)
from gx1.contracts import unified_exit_economics_objective_v2 as economics_owner


def _economics_readiness(rho: float = 0.08):
    train_split = "1" * 64
    train_fold = "2" * 64
    lineage = "3" * 64
    policy = "4" * 64
    hurdle = economics_owner.seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": economics_owner.CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": train_split,
            "train_fold_sha256": train_fold,
            "source_lineage_sha256": lineage,
            "annual_continuous_hurdle_rate": rho,
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": economics_owner.SECONDS_PER_YEAR,
            "fit_method": "unit_train_only",
            "fit_evidence_sha256": "5" * 64,
        }
    )
    objective = economics_owner.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256=train_split,
        expected_train_fold_sha256=train_fold,
        expected_source_lineage_sha256=lineage,
        policy_sha256=policy,
    )
    return {
        "schema_version": "gx1_unified_exit_training_economics_readiness_v3",
        "mode": "economics_objective_v2",
        "capital_hurdle_artifact": hurdle,
        "economics_objective_contract": objective,
        "expected_train_split_sha256": train_split,
        "expected_train_fold_sha256": train_fold,
        "expected_source_lineage_sha256": lineage,
        "policy_sha256": policy,
        "proper_policy_certificate_sha256": None,
        "test_data_used": False,
    }


def _two_episode_counterexample():
    # [episode, side, state, action].  At the shared causal state s0 both
    # episodes EXIT for -1 bps.  The next terminal realization is +8 / -12.
    target_q = torch.zeros((2, 1, 2, 2), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        target_q[0, 0, 1, 1] = 8.0
        target_q[1, 0, 1, 1] = -12.0
    rewards = torch.tensor(
        [[[ -1.0, 8.0]], [[-1.0, -12.0]]], dtype=torch.float32
    )
    state_valid = torch.ones((2, 1, 2), dtype=torch.bool)
    terminal = torch.zeros_like(state_valid)
    terminal[..., -1] = True
    action_valid = torch.ones((2, 1, 2, 2), dtype=torch.bool)
    action_valid[..., -1, 0] = False
    terminal_reason = torch.zeros_like(state_valid, dtype=torch.long)
    terminal_reason[..., -1] = 2
    return target_q, rewards, action_valid, state_valid, terminal, terminal_reason


def test_fitted_q_counterexample_does_not_learn_hindsight_expected_max():
    target_q, rewards, action_valid, state_valid, terminal, terminal_reason = (
        _two_episode_counterexample()
    )
    targets, valid = build_unified_exit_fitted_q_targets(
        frozen_target_q_bps=target_q,
        exit_now_reward_bps=rewards,
        action_valid_mask=action_valid,
        state_valid_mask=state_valid,
        terminal_mask=terminal,
        terminal_reason_index=terminal_reason,
    )
    assert torch.equal(valid, action_valid)
    # Fitted Bellman samples are the next-state target values themselves.
    assert targets[:, 0, 0, 0].tolist() == [8.0, -12.0]
    assert targets[:, 0, 0, 0].mean().item() == -2.0
    assert targets[:, 0, 0, 1].mean().item() == -1.0
    assert targets[:, 0, 0, 1].mean() > targets[:, 0, 0, 0].mean()
    # The forbidden pathwise oracle would first choose max(EXIT, future) per
    # realization: [8, -1], whose mean 3.5 falsely prefers HOLD.
    hindsight = torch.maximum(
        rewards[:, 0, 0], rewards[:, 0, 1]
    )
    assert hindsight.tolist() == [8.0, -1.0]
    assert hindsight.mean().item() == 3.5
    assert unified_exit_fitted_q_contract()[
        "pathwise_hindsight_max_is_training_target"
    ] is False


def test_fitted_q_targets_are_stop_gradient_and_capacity_agnostic():
    target_q, rewards, action_valid, state_valid, terminal, terminal_reason = (
        _two_episode_counterexample()
    )
    targets, _ = build_unified_exit_fitted_q_targets(
        frozen_target_q_bps=target_q,
        exit_now_reward_bps=rewards,
        action_valid_mask=action_valid,
        state_valid_mask=state_valid,
        terminal_mask=terminal,
        terminal_reason_index=terminal_reason,
    )
    assert targets.shape == (2, 1, 2, 2)
    assert not targets.requires_grad


def test_fitted_q_rejects_hidden_terminal_hold_action():
    target_q, rewards, action_valid, state_valid, terminal, terminal_reason = (
        _two_episode_counterexample()
    )
    action_valid[..., -1, 0] = True
    with pytest.raises(
        RuntimeError, match="UNIFIED_EXIT_FITTED_Q_HOLD_ACTION_MASK_INVALID"
    ):
        build_unified_exit_fitted_q_targets(
            frozen_target_q_bps=target_q,
            exit_now_reward_bps=rewards,
            action_valid_mask=action_valid,
            state_valid_mask=state_valid,
            terminal_mask=terminal,
            terminal_reason_index=terminal_reason,
        )


def test_nonterminal_chunk_boundary_requires_and_uses_explicit_successor():
    q = torch.zeros((1, 2, 3, 2), dtype=torch.float32)
    rewards = torch.zeros((1, 2, 3), dtype=torch.float32)
    state_valid = torch.ones_like(rewards, dtype=torch.bool)
    terminal = torch.zeros_like(state_valid)
    reason = torch.zeros_like(state_valid, dtype=torch.long)
    valid = torch.ones_like(q, dtype=torch.bool)
    with pytest.raises(
        RuntimeError, match="UNIFIED_EXIT_FITTED_Q_CHUNK_SUCCESSOR_REQUIRED"
    ):
        build_unified_exit_fitted_q_targets(
            frozen_target_q_bps=q,
            exit_now_reward_bps=rewards,
            action_valid_mask=valid,
            state_valid_mask=state_valid,
            terminal_mask=terminal,
            terminal_reason_index=reason,
        )
    successor_q = torch.tensor([[[4.0, 7.0], [9.0, 2.0]]])
    successor_valid = torch.ones_like(successor_q, dtype=torch.bool)
    targets, _ = build_unified_exit_fitted_q_targets(
        frozen_target_q_bps=q,
        exit_now_reward_bps=rewards,
        action_valid_mask=valid,
        state_valid_mask=state_valid,
        terminal_mask=terminal,
        terminal_reason_index=reason,
        chunk_successor_target_q_bps=successor_q,
        chunk_successor_action_valid_mask=successor_valid,
    )
    assert targets[0, :, -1, 0].tolist() == [7.0, 9.0]


def test_capacity_terminal_and_missing_unbounded_readiness_fail_closed():
    target_q, rewards, action_valid, state_valid, terminal, reason = (
        _two_episode_counterexample()
    )
    reason[..., -1] = 1
    with pytest.raises(
        RuntimeError, match="UNIFIED_EXIT_FITTED_Q_CAPACITY_TERMINAL_FORBIDDEN"
    ):
        build_unified_exit_fitted_q_targets(
            frozen_target_q_bps=target_q,
            exit_now_reward_bps=rewards,
            action_valid_mask=action_valid,
            state_valid_mask=state_valid,
            terminal_mask=terminal,
            terminal_reason_index=reason,
        )
    with pytest.raises(
        RuntimeError,
        match="UNBOUNDED_EXIT_ECONOMIC_TERMINAL_OR_PROPER_POLICY_REQUIRED",
    ):
        require_unified_exit_unbounded_training_readiness(
            None, context="UNIT"
        )


def test_right_censor_separates_policy_validity_from_bellman_mask():
    q = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    q[..., 1, 1] = 10.0
    rewards = torch.zeros((1, 1, 2), dtype=torch.float32)
    state_valid = torch.ones((1, 1, 2), dtype=torch.bool)
    terminal = torch.zeros_like(state_valid)
    reason = torch.zeros_like(state_valid, dtype=torch.long)
    policy_valid = torch.ones_like(q, dtype=torch.bool)
    bellman_valid = policy_valid.clone()
    bellman_valid[..., -1, 0] = False
    targets, target_mask = build_unified_exit_fitted_q_targets(
        frozen_target_q_bps=q,
        exit_now_reward_bps=rewards,
        action_valid_mask=policy_valid,
        state_valid_mask=state_valid,
        terminal_mask=terminal,
        terminal_reason_index=reason,
        bellman_target_valid_mask=bellman_valid,
        successor_observed_mask=torch.tensor([[[True, False]]]),
        right_censored_boundary_mask=torch.ones((1, 1), dtype=torch.bool),
        transition_discount=torch.tensor([[[0.5, 1.0]]]),
        hold_immediate_reward_bps=torch.tensor([[[1.25, 0.0]]]),
    )
    assert policy_valid[..., -1, 0].item() is True
    assert target_mask[..., -1, 0].item() is False
    assert targets[..., 0, 0].item() == 6.25


def test_economics_readiness_binds_verified_owner_contract():
    readiness = _economics_readiness()
    assert require_unified_exit_unbounded_training_readiness(
        readiness, context="UNIT"
    )["mode"] == "economics_objective_v2"
    readiness["mode"] = "undiscounted_proper_policy_v1"
    with pytest.raises(RuntimeError, match="PROPER_POLICY_REQUIRED"):
        require_unified_exit_unbounded_training_readiness(
            readiness, context="UNIT"
        )


def test_first_state_side_values_are_frozen_target_policy_values():
    q = torch.tensor(
        [[[[1.0, 2.0], [9.0, 8.0]], [[-3.0, -4.0], [7.0, 6.0]]]],
        requires_grad=True,
    )
    valid = torch.ones_like(q, dtype=torch.bool)
    state_valid = torch.ones(q.shape[:-1], dtype=torch.bool)
    values = unified_exit_first_state_side_values(
        frozen_target_q_bps=q,
        action_valid_mask=valid,
        state_valid_mask=state_valid,
    )
    assert torch.equal(values, torch.tensor([[2.0, -3.0]]))
    assert not values.requires_grad
    state = {
        "schema_version": "gx1_unified_exit_fitted_q_iteration_state_v1",
        "iteration_index": 4,
        "target_updated_from_val_or_test": False,
        "target_model_state_sha256": "1" * 64,
        "train_split_sha256": "2" * 64,
        "train_fold_sha256": "3" * 64,
        "source_lineage_sha256": "4" * 64,
        "normalization_sha256": "5" * 64,
        "fitted_q_contract": unified_exit_fitted_q_contract(),
    }
    envelope = build_unified_exit_first_state_value_envelope(
        entry_row_indices=[7],
        frozen_target_q_bps=q,
        action_valid_mask=valid,
        state_valid_mask=state_valid,
        fitted_q_iteration_state=state,
    )
    assert envelope["values_bps"] == [[2.0, -3.0]]
    assert envelope["target_model_state_sha256"] == "1" * 64
    assert envelope["iteration_index"] == 4
    assert envelope["train_split_sha256"] == "2" * 64
    assert len(envelope["envelope_sha256"]) == 64
