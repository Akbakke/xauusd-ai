"""The bounded calibration observer measures gradients without changing training."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _candidate_learning_diagnostics


class _RoutingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.family_tf_context_gate = nn.Linear(2, 1, bias=False)
        self.family_tf_token_gate = nn.Linear(2, 1, bias=False)
        self.log_variance = nn.Parameter(torch.tensor(7.0))


def _observe(model, weighted_losses):
    return _candidate_learning_diagnostics(
        model=model, weighted_losses=weighted_losses,
        entry_prediction=torch.tensor([[3.0, 1.0, 0.0], [-1.0, -2.0, 0.0], [-2.0, 4.0, 0.0]]),
        entry_target=torch.tensor([[2.0, -1.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 3.0, 0.0]]),
        entry_valid=torch.ones((3, 3), dtype=torch.bool),
        hold_target=torch.tensor([-2.0, 0.0, 3.0, 100.0]),
        hold_valid=torch.tensor([True, True, True, False]),
    )


def test_actual_weighted_route_gradients_retain_graph_grad_buffers_and_rng():
    model = _RoutingModel()
    x = torch.tensor([[1.0, 2.0]])
    representation = model.family_tf_context_gate(x) + model.family_tf_token_gate(x)
    weighted_losses = {
        "forecast_return_bps": 2.0 * representation.sum(),
        "entry_action_q": -3.0 * representation.sum(),
        "unified_exit_action": (representation * torch.full_like(representation, 4.0)).sum(),
    }
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.75)
    before_grad = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    before_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    before_rng = torch.get_rng_state().clone()
    evidence = _observe(model, weighted_losses)
    assert torch.equal(torch.get_rng_state(), before_rng)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.grad, before_grad[name])
        assert torch.equal(parameter, before_parameters[name])
    assert evidence["parameter_names"] == ["family_tf_context_gate.weight", "family_tf_token_gate.weight"]
    gradients = evidence["routing_task_gradients"]
    for name, scale in (("forecast_return_bps", 2.0), ("entry_action_q", 3.0), ("unified_exit_action", 4.0)):
        assert gradients[name]["l2_norm"] == pytest.approx(scale * math.sqrt(10.0))
        assert gradients[name]["status"] == "nonzero"
        assert gradients[name]["connected_parameter_names"] == evidence["parameter_names"]
        assert gradients[name]["unused_parameter_names"] == []
        assert gradients[name]["connected_zero_parameter_names"] == []
    assert evidence["routing_gradient_cosines"] == pytest.approx({
        "forecast_return_bps__entry_action_q": -1.0,
        "forecast_return_bps__unified_exit_action": 1.0,
        "entry_action_q__unified_exit_action": -1.0,
    })
    # The ordinary backward can still consume the retained original graph.
    sum(weighted_losses.values()).backward()
    for name in evidence["parameter_names"]:
        parameter = dict(model.named_parameters())[name]
        torch.testing.assert_close(parameter.grad, before_grad[name] + torch.tensor([[3.0, 6.0]]))
    assert torch.equal(model.log_variance.grad, before_grad["log_variance"])


def test_connected_zero_unused_and_absent_are_distinct_and_keep_grad_none():
    model = _RoutingModel()
    x = torch.tensor([[1.0, 2.0]])
    evidence = _observe(model, {
        "connected_zero": model.family_tf_context_gate(x).sum() * 0.0,
        "unrelated": model.log_variance.square(),
        "absent": None,
    })
    assert all(parameter.grad is None for parameter in model.parameters())
    rows = evidence["routing_task_gradients"]
    assert rows["connected_zero"]["status"] == "connected_zero"
    assert rows["connected_zero"]["connected_parameter_names"] == ["family_tf_context_gate.weight"]
    assert rows["connected_zero"]["connected_zero_parameter_names"] == ["family_tf_context_gate.weight"]
    assert rows["connected_zero"]["unused_parameter_names"] == ["family_tf_token_gate.weight"]
    assert rows["unrelated"]["status"] == rows["absent"]["status"] == "unused"
    assert rows["unrelated"]["loss_present"] is True
    assert rows["absent"]["loss_present"] is False
    assert all(row["l2_norm"] is None for row in rows.values())
    assert all(value is None for value in evidence["routing_gradient_cosines"].values())


def test_entry_choice_bias_and_relative_hold_summary_respect_masks_and_ties():
    evidence = _observe(_RoutingModel(), {"absent": None})
    assert evidence["entry"]["prediction"] == {
        "unique_greedy_count": {"LONG": 1, "SHORT": 1, "FLAT": 1}, "tied_row_count": 0,
    }
    assert evidence["entry"]["target"] == {
        "unique_greedy_count": {"LONG": 1, "SHORT": 1, "FLAT": 0}, "tied_row_count": 1,
    }
    actions = evidence["entry"]["by_action"]
    assert actions["LONG"]["prediction_minus_target_mean_bps"] == 0.0
    assert actions["SHORT"]["prediction_minus_target_mean_bps"] == pytest.approx(1.0 / 3.0)
    assert actions["FLAT"]["mean_absolute_error_bps"] == 0.0
    assert evidence["relative_hold_target_bps"] == pytest.approx({
        "valid_cell_count": 3, "positive_count": 1, "negative_count": 1, "zero_count": 1,
        "mean": 1.0 / 3.0, "mean_absolute": 5.0 / 3.0, "minimum": -2.0, "maximum": 3.0,
    })
    assert evidence["entry"]["target_semantics"] == "frozen_exit_value_estimates_not_realized_market_outcomes"
