from __future__ import annotations

import pytest
import torch

from gx1.contracts.unified_exit_fitted_q_v1 import (
    build_unified_exit_fitted_q_targets,
    build_unified_exit_first_state_value_envelope,
    unified_exit_first_state_side_values,
    unified_exit_fitted_q_contract,
)


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
    return target_q, rewards, action_valid, state_valid, terminal


def test_fitted_q_counterexample_does_not_learn_hindsight_expected_max():
    target_q, rewards, action_valid, state_valid, terminal = (
        _two_episode_counterexample()
    )
    targets, valid = build_unified_exit_fitted_q_targets(
        frozen_target_q_bps=target_q,
        exit_now_reward_bps=rewards,
        action_valid_mask=action_valid,
        state_valid_mask=state_valid,
        terminal_mask=terminal,
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
    target_q, rewards, action_valid, state_valid, terminal = (
        _two_episode_counterexample()
    )
    targets, _ = build_unified_exit_fitted_q_targets(
        frozen_target_q_bps=target_q,
        exit_now_reward_bps=rewards,
        action_valid_mask=action_valid,
        state_valid_mask=state_valid,
        terminal_mask=terminal,
    )
    assert targets.shape == (2, 1, 2, 2)
    assert not targets.requires_grad


def test_fitted_q_rejects_hidden_terminal_hold_action():
    target_q, rewards, action_valid, state_valid, terminal = (
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
