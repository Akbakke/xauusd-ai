from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.unified_exit_random_access_model_v1 import liquidation_relative_action_values
from gx1.contracts.unified_exit_random_access_state_view_v1 import (
    LIQUIDATION_RELATIVE_STATE_VIEW_SCHEMA_VERSION,
    _structured_sha256,
    require_random_access_state_view,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    LIQUIDATION_RELATIVE_TRAIN_BATCH_SCHEMA_VERSION,
    RANDOM_ACCESS_TRAIN_BATCH_SCHEMA_VERSION,
    run_random_access_training_step,
)
from tests import test_unified_exit_random_access_state_view_v1 as state_fixture
from tests.test_unified_exit_random_access_training_v1 import _collate


@pytest.fixture
def relative_fixture(monkeypatch):
    objective = state_fixture._objective(economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING)
    monkeypatch.setattr(state_fixture, "_objective", lambda: objective)
    movement = [-5.0, 5.0]

    def projection(self, entry, side, start, stop, hold_stop):
        liquidations = -4.0 + movement[side] * np.arange(start, stop, dtype=np.float64)
        gamma = math.exp(-0.1 * 60 / economics.SECONDS_PER_YEAR)
        successor = -4.0 + movement[side] * np.arange(start + 1, hold_stop + 1, dtype=np.float64)
        return state_fixture.seal_economic_training_projection({
            "schema_version": state_fixture.ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
            "entry_row_index": entry, "side_index": side,
            "start_state_index": start, "stop_state_index": stop,
            "hold_stop_state_index": hold_stop,
            "exit_event_kind_index": np.zeros(stop - start, dtype="u1"),
            "exit_reward_bps": liquidations,
            "hold_event_kind_index": np.ones(hold_stop - start, dtype="u1"),
            "hold_reward_bps": -0.1 + (1 - gamma) * successor,
            "economic_step_model_sha256": "b" * 64,
            "economic_step_source_manifest_sha256": "c" * 64,
        })

    monkeypatch.setattr(state_fixture._EconomicProvider, "materialize_training_projection", projection)
    return movement


def _validate(view):
    return require_random_access_state_view(
        view, sampler_contract=state_fixture._contract(),
        sample=state_fixture._sample(state_fixture._contract(), 0),
        expected_m1_source_sha256=view["m1_source_sha256"],
        expected_market_closure_authority_sha256=view["market_closure_authority_sha256"],
        expected_economic_step_manifest_sha256=view["economic_step_manifest_sha256"],
        expected_economics_objective_contract_sha256=view["economics_objective_contract_sha256"],
    )


def test_current_inputs_do_not_see_future_liquidation(relative_fixture):
    view = state_fixture._materialize(0)
    assert view["schema_version"] == LIQUIDATION_RELATIVE_STATE_VIEW_SCHEMA_VERSION
    np.testing.assert_allclose(view["liquidation_relative_reward_bps"], [[-5.1, 0], [4.9, 0]], atol=1e-6)
    np.testing.assert_array_equal(view["current_liquidation_value_bps"], [-4, -4])
    relative_fixture[0] = -50
    changed = state_fixture._materialize(0)
    assert _structured_sha256(view["current"]) == _structured_sha256(changed["current"])
    assert changed["liquidation_relative_reward_bps"][0, 0] == pytest.approx(-50.1)
    assert "successor_liquidation_value_bps" not in changed["current"]


def test_resealed_inconsistent_relative_reward_is_rejected(relative_fixture):
    view = state_fixture._materialize(0)
    malformed = dict(view)
    rewards = view["liquidation_relative_reward_bps"].copy()
    rewards[0, 0] += 10
    rewards.setflags(write=False)
    malformed["liquidation_relative_reward_bps"] = rewards
    malformed.pop("state_view_sha256")
    malformed["state_view_sha256"] = _structured_sha256(malformed)
    with pytest.raises(RuntimeError, match="RELATIVE_REWARD_IDENTITY_INVALID"):
        _validate(malformed)


class _TwoRowHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 4)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[0.01], [-0.01], [0.01], [-0.01]]))
            self.head.bias.copy_(torch.tensor([0.25, 0.1, 0.25, 0.1]))
        self.task_log_variances = nn.ParameterDict({"unified_exit_action": nn.Parameter(torch.tensor(0.0))})
        self.calls = 0

    def forward_exit_random_access_batch(self, **inputs):
        self.calls += 1
        assert inputs["liquidation_relative_values"] is True
        assert "successor_liquidation_value_bps" not in inputs
        raw = self.head(inputs["entry_decision_representation"]).reshape(-1, 2, 2)
        return {"exit_action_q_bps": liquidation_relative_action_values(raw),
                "exit_action_valid_mask": inputs["action_valid_mask"]}


def test_training_uses_observed_changes_and_costs_once(relative_fixture):
    batch = _collate()
    assert batch["schema_version"] == LIQUIDATION_RELATIVE_TRAIN_BATCH_SCHEMA_VERSION
    model = _TwoRowHead()
    target = copy.deepcopy(model).requires_grad_(False).eval()
    entry = torch.tensor([[2.0], [3.0], [4.0]], requires_grad=True)
    result = run_random_access_training_step(
        model=model, target_model=target, entry_decision_representations=entry,
        target_entry_decision_representations=entry.detach().clone(), batch=batch, grad_accum_steps=1,
    )
    gamma = batch["elapsed_wall_clock_gamma"][0].item()
    expected = torch.tensor([[[-5.1 + gamma * 0.21, 0], [4.9 + gamma * 0.21, 0]]])
    torch.testing.assert_close(result["targets"], expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(result["entry_targets"][1], torch.tensor([-3.79, -3.79, 0]), rtol=0, atol=1e-6)
    assert result["entry_targets"][1].argmax().item() == 2
    assert model.calls == target.calls == 1
    assert result["entry_gradients"][1].abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())
    grad = model.head.bias.grad.reshape(2, 2)
    assert grad[0, 0] > 0 and grad[1, 0] < 0
    torch.testing.assert_close(grad[:, 0], -grad[:, 1], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["drop_flag", "false_flag", "legacy_schema", "drop_reward"])
def test_training_rejects_coordinate_mismatch(relative_fixture, mutation):
    batch = _collate()
    if mutation == "drop_flag":
        batch.pop("liquidation_relative_values")
    elif mutation == "false_flag":
        batch["liquidation_relative_values"] = False
    elif mutation == "legacy_schema":
        batch["schema_version"] = RANDOM_ACCESS_TRAIN_BATCH_SCHEMA_VERSION
    else:
        batch.pop("liquidation_relative_reward_bps")
    model = _TwoRowHead()
    with pytest.raises(RuntimeError, match="VALUE_COORDINATES_INVALID"):
        run_random_access_training_step(
            model=model, target_model=copy.deepcopy(model).requires_grad_(False).eval(),
            entry_decision_representations=torch.zeros(3, 1),
            target_entry_decision_representations=torch.zeros(3, 1), batch=batch, grad_accum_steps=1,
        )
    assert model.calls == 0


def test_relative_head_preserves_order_and_rejects_overflow():
    raw = torch.tensor([[[2.0, 1.0], [-1.0, 3.0]], [[-4.0, -4.0], [3.0, 0.0]]], requires_grad=True)
    q = liquidation_relative_action_values(raw)
    assert torch.equal(q.argmax(-1), raw.argmax(-1))
    assert torch.equal(q[..., 1], torch.zeros_like(q[..., 1]))
    q[..., 0].sum().backward()
    assert torch.equal(raw.grad[..., 0], torch.ones_like(q[..., 0]))
    assert torch.equal(raw.grad[..., 1], -torch.ones_like(q[..., 0]))
    with pytest.raises(RuntimeError, match="ADVANTAGE_HEAD_INVALID"):
        liquidation_relative_action_values(torch.tensor([[3e38, -3e38]]))


def test_actual_model_keeps_legacy_outputs_and_relative_order():
    from tests.test_unified_exit_random_access_model_v1 import _inputs, _make_model
    torch.manual_seed(20260914)
    model = _make_model(dropout=0.0).eval()
    inputs = _inputs()
    with torch.inference_mode():
        legacy = model.forward_exit_random_access_batch(**inputs)["exit_action_q_bps"]
        relative = model.forward_exit_random_access_batch(**inputs, liquidation_relative_values=True)["exit_action_q_bps"]
        explicit_legacy = model.forward_exit_random_access_batch(**inputs, liquidation_relative_values=False)["exit_action_q_bps"]
    torch.testing.assert_close(legacy, explicit_legacy, rtol=0, atol=0)
    torch.testing.assert_close(relative, liquidation_relative_action_values(legacy), rtol=0, atol=0)


def test_v4_provider_preserves_physical_projection_and_empty_hold_anchor():
    from tests.test_unified_exit_economic_step_provider_v1 import _provider
    marked, _ = _provider(reward_accounting=economics.MARK_TO_MARKET_REWARD_ACCOUNTING)
    relative, readiness = _provider(reward_accounting=economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING)
    assert readiness["mode"] == "economics_objective_v4"
    for side in (0, 1):
        old = marked.materialize_training_projection(0, side, 0, 3, 2)
        new = relative.materialize_training_projection(0, side, 0, 3, 2)
        for field in ("exit_reward_bps", "hold_reward_bps"):
            np.testing.assert_array_equal(old[field], new[field])
        anchor = relative.materialize_training_projection(0, side, 0, 1, 0)
        assert anchor["hold_reward_bps"].shape == (0,)
        assert anchor["exit_reward_bps"][0] == new["exit_reward_bps"][0]
