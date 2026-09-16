"""Native economic/state owners feed Q_mu without intermediate model inputs."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from gx1.contracts.unified_exit_reference_policy_v1 import (
    build_reference_policy_hold_targets, reference_policy_contract,
)
from gx1.contracts.unified_exit_random_access_state_view_v1 import (
    COMPACT_STATE_FIELDS, REFERENCE_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION,
    _structured_sha256,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    REFERENCE_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION,
    collate_random_access_training_items, run_random_access_training_step,
)
from tests import test_unified_exit_random_access_state_view_v1 as states
from tests.test_liquidation_relative_learning_v1 import relative_fixture, _validate, _TwoRowHead
from tests.test_unified_exit_random_access_training_v1 import _item, _normalization, _collate


def _batch(*, counts=(600, 600), terminals=(False, False), explicit_policy=True, reference_view=True):
    contract, item = _item(state_index=0)
    policy = reference_policy_contract()
    view = states._materialize(0, counts=counts, terminals=terminals,
                               reference_policy=policy if reference_view else None)
    item["transitions"][0]["state_view"] = view
    batch = collate_random_access_training_items(
        [item], outer_batch_size=3, sampler_contract=contract,
        normalization_artifact=_normalization(), expected_m1_source_sha256=view["m1_source_sha256"],
        expected_market_closure_authority_sha256=view["market_closure_authority_sha256"],
        expected_economic_step_manifest_sha256=view["economic_step_manifest_sha256"],
        expected_economics_objective_contract_sha256=view["economics_objective_contract_sha256"],
        device=torch.device("cpu"), reference_policy=policy if explicit_policy else None,
    )
    return batch, view


def test_120_rewards_materialize_only_current_successor_and_boundary(relative_fixture):
    calls = []
    view = states._materialize(0, reference_policy=reference_policy_contract(), model_state_times=calls)
    trace = view["reference_policy_trace"]
    assert view["schema_version"] == REFERENCE_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION
    assert len(trace["steps"]) == 120
    assert trace["boundary"]["state_index"] == 120
    assert calls == [view["current"]["bar_start_time_ns"], view["successor"]["bar_start_time_ns"], trace["boundary"]["bar_start_time_ns"]]
    assert all(set(step[name]) == set(COMPACT_STATE_FIELDS)
               for step in trace["steps"] for name in ("current", "successor"))
    legacy = states._materialize(0)
    explicit_default = states._materialize(0, reference_policy=None)
    assert legacy["state_view_sha256"] == explicit_default["state_view_sha256"]
    for key in ("current", "successor", "loss_weight", "sample_identity_sha256", "liquidation_relative_reward_bps"):
        assert _structured_sha256(legacy[key]) == _structured_sha256(view[key])
    # Compare the compact route with the existing native full-state reward owner,
    # including its actual wall-clock weekend transition and far boundary.
    for offset in (0, 1, 4, 5, 119):
        old = states._materialize(offset)
        step = trace["steps"][offset]
        for key in step:
            if key not in ("current", "successor"):
                assert _structured_sha256(step[key]) == _structured_sha256(old[key])
    assert trace["steps"][4]["transition_closure"]["wall_clock_delta_seconds"] > 60


@pytest.mark.parametrize("count", [2, 3, 121, 122])
@pytest.mark.parametrize("terminal", [False, True])
def test_boundary_inputs_and_bootstrap_use_actual_availability(relative_fixture, count, terminal):
    batch, view = _batch(counts=(count, count + 1), terminals=(terminal, False))
    trace = batch["reference_policy_trace"]
    n = min(120, count - 1)
    at_end = n == count - 1
    assert batch["schema_version"] == REFERENCE_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION
    assert batch["transition_count"] == 1
    assert batch["online_entry_batch_index"].tolist() == [1]
    assert batch["target_entry_batch_index"].tolist() == [1, 1]
    assert trace["state_view_sha256"] == [view["state_view_sha256"]]
    assert trace["transition_available_mask"].tolist() == [[True] * n]
    assert trace["boundary_right_censored_mask"].tolist() == [[at_end and not terminal, False]]
    assert trace["boundary_action_valid_mask"].tolist() == [[[not (at_end and terminal), True], [True, True]]]
    legacy = _collate()
    torch.testing.assert_close(batch["importance_weight"], legacy["importance_weight"], rtol=0, atol=0)
    torch.testing.assert_close(batch["entry_liquidation_value_bps"], legacy["entry_liquidation_value_bps"], rtol=0, atol=0)
    assert batch["anchor_state_view_sha256"] == legacy["anchor_state_view_sha256"]
    torch.testing.assert_close(trace["hold_reward_bps"][:, 0], batch["liquidation_relative_reward_bps"][..., 0], rtol=0, atol=0)
    q = torch.tensor([[[-17., 0.], [23., 0.]]])
    if terminal and at_end:
        q[0, 0, 0] = float("nan")  # Invalid HOLD is never consumed.
    result = build_reference_policy_hold_targets(
        policy=trace["policy"], boundary_action_q_bps=q,
        **{key: value for key, value in trace.items() if key not in ("policy", "state_view_sha256")},
    )
    expected = q[..., 0].masked_fill(~trace["boundary_action_valid_mask"][..., 0], 0)
    p = 119 / 120
    for offset in range(n - 1, -1, -1):
        expected = trace["hold_reward_bps"][:, offset] + p * trace["elapsed_wall_clock_gamma"][:, offset, None] * expected.masked_fill(trace["successor_terminal_mask"][:, offset], 0)
    torch.testing.assert_close(result["hold_target_bps"], expected, rtol=3e-6, atol=1e-5)
    assert result["observed_backup_step_count"].tolist() == [n]
    if not (terminal and at_end):
        assert result["bootstrap_component_bps"][0, 0] < 0  # No max with EXIT=0.


@pytest.mark.parametrize("mutation", ["reward", "link", "first", "terminal", "length", "policy", "boundary", "clock", "nonfinite_physical"])
def test_resealed_reference_evidence_corruption_is_rejected(relative_fixture, mutation):
    view = states._materialize(0, reference_policy=reference_policy_contract())
    trace = {**view["reference_policy_trace"], "steps": [dict(s) for s in view["reference_policy_trace"]["steps"]]}
    if mutation in ("reward", "first"):
        step = trace["steps"][0 if mutation == "first" else 50]
        rewards = step["liquidation_relative_reward_bps"].copy()
        rewards[0, 0] += 10
        rewards.setflags(write=False)
        step["liquidation_relative_reward_bps"] = rewards
    elif mutation == "nonfinite_physical":
        rewards = trace["steps"][50]["immediate_reward_bps"].copy()
        rewards[0, 0] = float("nan")
        rewards.setflags(write=False)
        trace["steps"][50]["immediate_reward_bps"] = rewards
    elif mutation == "link":
        trace["steps"][50]["current"] = {**trace["steps"][50]["current"], "state_index": 49}
    elif mutation == "terminal":
        mask = np.array([True, False]); mask.setflags(write=False)
        trace["steps"][50]["successor_terminal_mask"] = mask
    elif mutation == "length":
        trace["steps"].pop()
    elif mutation == "policy":
        trace["policy"] = {**trace["policy"], "maximum_observed_backup_steps": 5}
    elif mutation == "clock":
        trace["steps"][50]["successor"] = {**trace["steps"][50]["successor"], "decision_time_ns": 0}
    else:
        trace["boundary"] = view["successor"]
    malformed = {**view, "reference_policy_trace": trace}
    malformed.pop("state_view_sha256")
    malformed["state_view_sha256"] = _structured_sha256(malformed)
    with pytest.raises(RuntimeError):
        _validate(malformed)


@pytest.mark.parametrize("explicit_policy,reference_view", [(True, False), (False, True)])
def test_policy_cannot_be_inferred_or_silently_dropped(relative_fixture, explicit_policy, reference_view):
    with pytest.raises(RuntimeError, match="REFERENCE_POLICY_BINDING_INVALID"):
        _batch(explicit_policy=explicit_policy, reference_view=reference_view)


def test_unproven_market_gap_stays_closed(relative_fixture):
    with pytest.raises(RuntimeError, match="SUCCESSOR_GAP_CENSORED"):
        states._materialize(0, known_gap=False, reference_policy=reference_policy_contract())


@pytest.mark.parametrize("options", [{"backup_steps": 5}, {"anchor": True}])
def test_reference_is_separate_from_greedy_trace_and_anchor(relative_fixture, options):
    with pytest.raises(RuntimeError, match="REFERENCE_SCOPE_INVALID"):
        states._materialize(0, reference_policy=reference_policy_contract(), **options)


def test_reference_requires_relative_value_coordinates():
    with pytest.raises(RuntimeError, match="REFERENCE_SCOPE_INVALID"):
        states._materialize(0, reference_policy=reference_policy_contract())


@pytest.mark.parametrize("downgrade", [False, True])
def test_native_training_stays_closed_until_checkpoint_recipe_binding(relative_fixture, downgrade):
    batch, _ = _batch()
    if downgrade:
        batch["schema_version"] = "gx1_unified_exit_random_access_train_batch_v2"
        batch.pop("target_action_valid_mask")
    model = _TwoRowHead()
    target = copy.deepcopy(model).requires_grad_(False).eval()
    with pytest.raises(RuntimeError, match="REFERENCE_TRAINING_NOT_BOUND"):
        run_random_access_training_step(model=model, target_model=target,
            entry_decision_representations=torch.ones(3, 1), target_entry_decision_representations=torch.ones(3, 1),
            batch=batch, grad_accum_steps=1)
    assert model.calls == target.calls == 0
    assert all(parameter.grad is None for parameter in model.parameters())
