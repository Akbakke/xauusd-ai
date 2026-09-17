"""The bounded calibration observer measures gradients without changing training."""

from __future__ import annotations

import ast
import inspect
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _candidate_learning_diagnostics, _entry_training_gradient_boundary_kwargs, train_epoch,
)
from tests.test_unified_exit_economics_objective_v2 import _contract


class _RoutingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.family_tf_context_gate = nn.Linear(2, 1, bias=False)
        self.family_tf_token_gate = nn.Linear(2, 1, bias=False)
        self.log_variance = nn.Parameter(torch.tensor(7.0))


def _observe(model, weighted_losses, **overrides):
    inputs = dict(
        model=model, weighted_losses=weighted_losses,
        entry_prediction=torch.tensor([[3.0, 1.0, 0.0], [-1.0, -2.0, 0.0], [-2.0, 4.0, 0.0]]),
        entry_target=torch.tensor([[2.0, -1.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 3.0, 0.0]]),
        entry_valid=torch.ones((3, 3), dtype=torch.bool),
        hold_target=torch.tensor([-2.0, 0.0, 3.0, 100.0]),
        hold_valid=torch.tensor([True, True, True, False]),
        hold_reward=torch.tensor([-3.0, -1.0, 0.0, float("nan")]),
        entry_first_liquidation=torch.tensor([[-3.0, -1.0], [-2.0, -2.0], [0.0, 0.0]]),
        entry_anchor_indices=torch.tensor([2, 0, 1]),
    )
    inputs.update(overrides)
    return _candidate_learning_diagnostics(**inputs)


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
    assert evidence["entry"]["target_semantics"] == "fitted_entry_targets_not_realized_trading_policy_profit"


def test_decomposition_uses_actual_masked_reward_and_selected_anchor_order():
    reward = torch.tensor([-3.0, -1.0, 0.0, float("nan")], requires_grad=True)
    liquidation = torch.tensor([[-3.0, -1.0], [-2.0, -2.0]], requires_grad=True)
    evidence = _observe(
        _RoutingModel(), {"absent": None}, hold_reward=reward,
        entry_first_liquidation=liquidation, entry_anchor_indices=torch.tensor([2, 0]),
        entry_valid=torch.tensor([[True, True, True], [False, False, True], [True, True, True]]),
    )
    hold = evidence["relative_hold_decomposition_bps"]
    assert hold["liquidation_relative_reward_bps"] == pytest.approx({
        "valid_cell_count": 3, "positive_count": 0, "negative_count": 2, "zero_count": 1,
        "mean": -4.0 / 3.0, "mean_absolute": 4.0 / 3.0, "minimum": -3.0, "maximum": 0.0,
    })
    assert hold["frozen_bootstrap_bps"] == pytest.approx({
        "valid_cell_count": 3, "positive_count": 3, "negative_count": 0, "zero_count": 0,
        "mean": 5.0 / 3.0, "mean_absolute": 5.0 / 3.0, "minimum": 1.0, "maximum": 3.0,
    })
    assert hold["nonpositive_reward_positive_target_count"] == 1
    entry = evidence["entry"]["target_decomposition_bps"]
    assert entry["first_executable_liquidation_bps"]["mean"] == -2.0
    assert entry["frozen_continuation_bps"] == pytest.approx({
        "valid_cell_count": 4, "positive_count": 4, "negative_count": 0, "zero_count": 0,
        "mean": 2.5, "mean_absolute": 2.5, "minimum": 1.0, "maximum": 4.0,
    })
    assert entry["source_counts"] == {
        "outer_entry_rows": 3, "selected_first_anchor_rows": 2,
        "rows_without_first_anchor": 1, "anchored_valid_side_cells": 4,
        "valid_side_cells_without_first_anchor": 0, "flat_zero_baseline_rows": 3,
        "nonpositive_liquidation_positive_target_cells": 2,
    }
    assert reward.grad is None and liquidation.grad is None
    torch.testing.assert_close(liquidation, torch.tensor([[-3.0, -1.0], [-2.0, -2.0]]))
    torch.testing.assert_close(reward, torch.tensor([-3.0, -1.0, 0.0, float("nan")]), equal_nan=True)


def test_decomposition_empty_sources_are_unavailable_not_zero_means():
    evidence = _observe(
        _RoutingModel(), {"absent": None},
        entry_first_liquidation=torch.empty((0, 2)), entry_anchor_indices=torch.empty(0, dtype=torch.int64),
        entry_valid=torch.tensor([[False, False, True]] * 3), hold_valid=torch.zeros(4, dtype=torch.bool),
    )
    entry = evidence["entry"]["target_decomposition_bps"]
    for summary in (entry["first_executable_liquidation_bps"], entry["frozen_continuation_bps"],
                    evidence["relative_hold_decomposition_bps"]["frozen_bootstrap_bps"]):
        assert summary["valid_cell_count"] == 0
        assert summary["mean"] is None and summary["minimum"] is None and summary["maximum"] is None
    assert entry["source_counts"]["rows_without_first_anchor"] == 3
    assert entry["source_counts"]["flat_zero_baseline_rows"] == 3


@pytest.mark.parametrize("indices", [torch.tensor([0, 0, 2]), torch.tensor([0, 1, 3])])
def test_decomposition_rejects_wrong_anchor_mapping(indices):
    with pytest.raises(RuntimeError, match="ENTRY_ANCHORS_INVALID"):
        _observe(_RoutingModel(), {"absent": None}, entry_anchor_indices=indices)


@pytest.mark.parametrize("accounting,expected", [
    ("terminal_cash_v2", {}), ("liquidation_value_increments_v1", {}),
    ("liquidation_advantage_v1", {"liquidation_relative_values": True}),
])
def test_online_gradient_boundary_is_derived_from_validated_economics(accounting, expected):
    objective = _contract(reward_accounting=accounting)
    adapter = SimpleNamespace(
        _readiness={"economics_objective_contract": objective},
        random_access_training_bindings_v1=lambda: {"economics_objective_contract_sha256": objective["contract_sha256"]},
    )
    assert _entry_training_gradient_boundary_kwargs(SimpleNamespace(_unified_exit_lifecycle_v2=adapter)) == expected
    assert _entry_training_gradient_boundary_kwargs(SimpleNamespace(_unified_exit_lifecycle_v2=None)) == {}


@pytest.mark.parametrize("tamper", ["objective_hash", "source_binding"])
def test_online_gradient_boundary_rejects_unbound_or_tampered_v4(tamper):
    objective = _contract(reward_accounting="liquidation_advantage_v1")
    bound_hash = objective["contract_sha256"]
    if tamper == "objective_hash":
        objective["contract_sha256"] = "0" * 64
    else:
        bound_hash = "0" * 64
    adapter = SimpleNamespace(
        _readiness={"economics_objective_contract": objective},
        random_access_training_bindings_v1=lambda: {"economics_objective_contract_sha256": bound_hash},
    )
    with pytest.raises(RuntimeError, match="OBJECTIVE_CONTRACT_INVALID|ECONOMICS_BINDING_MISMATCH"):
        _entry_training_gradient_boundary_kwargs(SimpleNamespace(_unified_exit_lifecycle_v2=adapter))


def test_train_epoch_wires_validated_boundary_only_to_online_forward():
    tree = ast.parse(inspect.getsource(train_epoch))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    binding = [node for node in calls if isinstance(node.func, ast.Name)
               and node.func.id == "_entry_training_gradient_boundary_kwargs"]
    assert len(binding) == 1 and ast.unparse(binding[0].args[0]) == "dataset"
    forwards = {node.args[0].id: node for node in calls
                if isinstance(node.func, ast.Name) and node.func.id == "_model_forward_fp32"}
    assert set(forwards) == {"model", "target_model"}
    def has_boundary(call):
        return any(kw.arg is None and isinstance(kw.value, ast.Name)
                   and kw.value.id == "entry_gradient_boundary_kwargs" for kw in call.keywords)
    assert has_boundary(forwards["model"])
    assert not has_boundary(forwards["target_model"])
