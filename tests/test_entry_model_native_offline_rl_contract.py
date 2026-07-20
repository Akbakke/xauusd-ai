from __future__ import annotations

import pytest
import torch

from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_VALUE_DIM,
    ACTION_VALUE_TARGET_COLUMNS,
    ADVANTAGE_DIM,
    EXPECTILE_VALUE_DIM,
    HORIZON_COUNT,
    RANKING_MARGIN_SCALED,
    expectile_loss,
    offline_rl_contract_metadata,
    q_ranking_margin_loss,
    require_offline_rl_contract_metadata,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def test_offline_rl_metadata_is_exact_and_fail_closed() -> None:
    expected = offline_rl_contract_metadata()

    assert ACTION_VALUE_DIM == 9
    assert EXPECTILE_VALUE_DIM == 3
    assert ADVANTAGE_DIM == 9
    assert len(ACTION_VALUE_TARGET_COLUMNS) == 9
    assert expected["ambiguous_reward_ties_ranked"] is False
    assert expected["ranking_target"].startswith("unique_")
    assert ACTION_VALUE_TARGET_COLUMNS == (
        "y_action_value_long_K12",
        "y_action_value_long_K48",
        "y_action_value_long_K96",
        "y_action_value_short_K12",
        "y_action_value_short_K48",
        "y_action_value_short_K96",
        "y_action_value_flat_K12",
        "y_action_value_flat_K48",
        "y_action_value_flat_K96",
    )
    assert require_offline_rl_contract_metadata(expected, context="TEST") == expected

    broken = dict(expected)
    broken["reward_scale_bps"] = 1.0
    with pytest.raises(RuntimeError, match="TEST_OFFLINE_RL_CONTRACT_INVALID"):
        require_offline_rl_contract_metadata(broken, context="TEST")


def test_expectile_loss_weights_positive_and_negative_residuals() -> None:
    diff = torch.tensor([1.0, -1.0])

    actual = expectile_loss(diff, tau=0.8)

    assert actual.item() == pytest.approx(0.5)
    with pytest.raises(ValueError, match="expectile tau"):
        expectile_loss(diff, tau=0.5)


def test_q_ranking_margin_loss_is_zero_only_after_required_ordering_margin() -> None:
    rewards = torch.zeros(1, 3, HORIZON_COUNT)
    rewards[:, 0, :] = 1.0
    ordered_q = torch.zeros_like(rewards)
    ordered_q[:, 0, :] = RANKING_MARGIN_SCALED

    assert q_ranking_margin_loss(ordered_q, rewards).item() == pytest.approx(0.0)
    assert q_ranking_margin_loss(torch.zeros_like(rewards), rewards).item() == pytest.approx(
        RANKING_MARGIN_SCALED
    )

    with pytest.raises(ValueError, match="Q/reward shapes differ"):
        q_ranking_margin_loss(ordered_q, rewards[:, :, :1])


def test_q_ranking_margin_ignores_ambiguous_reward_ties() -> None:
    tied_rewards = torch.zeros(2, 3, HORIZON_COUNT)
    arbitrary_q = torch.randn_like(tied_rewards, requires_grad=True)

    loss = q_ranking_margin_loss(arbitrary_q, tied_rewards)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert arbitrary_q.grad is not None
    assert arbitrary_q.grad.abs().sum().item() == pytest.approx(0.0)


def test_trainer_offline_rl_loss_supervises_q_and_v_without_behavior_action() -> None:
    q_flat = torch.full((2, ACTION_VALUE_DIM), 0.2, requires_grad=True)
    value = torch.zeros(2, EXPECTILE_VALUE_DIM, requires_grad=True)
    advantage = q_flat.reshape(2, 3, 3) - value.unsqueeze(1)
    out = {
        "action_value": q_flat,
        "expectile_value": value,
        "action_advantage": advantage.reshape(2, ACTION_VALUE_DIM),
    }
    batch = {
        name: torch.full((2,), 50.0 if "_long_" in name else 0.0)
        for name in ACTION_VALUE_TARGET_COLUMNS
    }

    loss = trainer.offline_rl_aux_loss(out, batch, torch.device("cpu"))
    loss.backward()

    assert loss.item() > 0.0
    assert q_flat.grad is not None and q_flat.grad.abs().sum().item() > 0.0
    assert value.grad is not None and value.grad.abs().sum().item() > 0.0

    broken = dict(batch)
    del broken[ACTION_VALUE_TARGET_COLUMNS[0]]
    with pytest.raises(RuntimeError, match="ACTIVE_HEAD_TARGET_MISSING"):
        trainer.offline_rl_aux_loss(out, broken, torch.device("cpu"))
