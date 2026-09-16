from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from gx1.contracts.unified_exit_random_access_state_view_v1 import (
    FROZEN_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION, _structured_sha256,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    FROZEN_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION,
    collate_random_access_training_items, frozen_policy_trace_hold_targets,
    run_random_access_training_step,
)
from tests import test_unified_exit_random_access_state_view_v1 as states
from tests.test_liquidation_relative_learning_v1 import relative_fixture, _validate, _TwoRowHead
from tests.test_unified_exit_random_access_training_v1 import _item, _normalization, _collate


def _batch(*, state_index=0, counts=(600, 600), terminals=(False, False)):
    contract, item = _item(state_index=state_index)
    view = states._materialize(state_index, counts=counts, terminals=terminals, backup_steps=5)
    item["transitions"][0]["state_view"] = view
    batch = collate_random_access_training_items(
        [item], outer_batch_size=3, sampler_contract=contract,
        normalization_artifact=_normalization(), expected_m1_source_sha256=view["m1_source_sha256"],
        expected_market_closure_authority_sha256=view["market_closure_authority_sha256"],
        expected_economic_step_manifest_sha256=view["economic_step_manifest_sha256"],
        expected_economics_objective_contract_sha256=view["economics_objective_contract_sha256"],
        device=torch.device("cpu"),
    )
    return batch, view


def _synthetic(*, available=5):
    q = torch.tensor([[[2., 0.], [3., 0.]]] * 5)
    mask = torch.ones_like(q, dtype=torch.bool)
    trace = {
        "successor_q_indices": torch.arange(5).reshape(1, 5),
        "hold_reward_bps": torch.tensor([[[-1., 1.], [-100., 2.], [3., 3.], [4., 4.], [5., 5.]]]),
        "elapsed_wall_clock_gamma": torch.tensor([[.9, .8, .7, .6, .5]]),
        "transition_available_mask": (torch.arange(5) < available).reshape(1, 5),
    }
    return q, mask, trace


def test_five_step_return_follows_teacher_even_when_realized_continuation_loses():
    q, mask, trace = _synthetic()
    actual = frozen_policy_trace_hold_targets(all_target_q=q, target_action_valid_mask=mask, trace=trace)
    expected = torch.tensor([2., 3.])
    for offset in range(4, -1, -1):
        expected = trace["hold_reward_bps"][0, offset] + trace["elapsed_wall_clock_gamma"][0, offset] * expected
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    assert actual[0, 0] < -80  # No hindsight max(EXIT=0, realized HOLD return).
    assert not actual.requires_grad


@pytest.mark.parametrize("mode", ["exit", "tie", "terminal"])
def test_intermediate_exit_tie_and_real_terminal_do_not_consume_later_rewards(mode):
    q, mask, trace = _synthetic()
    if mode == "exit":
        q[0, 0, 0] = -1
    elif mode == "tie":
        q[0, 0, 0] = 0
    else:
        mask[0, 0, 0] = False
    actual = frozen_policy_trace_hold_targets(all_target_q=q, target_action_valid_mask=mask, trace=trace)
    assert actual[0, 0].item() == -1
    assert actual[0, 1] > 1  # Other side still follows its own teacher.


@pytest.mark.parametrize("available", [1, 2, 4])
def test_missing_additional_successor_bootstraps_without_new_terminal(available):
    q, mask, trace = _synthetic(available=available)
    actual = frozen_policy_trace_hold_targets(all_target_q=q, target_action_valid_mask=mask, trace=trace)
    expected = q[available - 1, :, 0]
    for offset in range(available - 1, -1, -1):
        expected = trace["hold_reward_bps"][0, offset] + trace["elapsed_wall_clock_gamma"][0, offset] * expected
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)


def test_materialized_trace_keeps_sample_current_inputs_and_one_step_default(relative_fixture):
    default = states._materialize(0)
    explicit = states._materialize(0, backup_steps=1)
    trace = states._materialize(0, backup_steps=5)
    assert default["state_view_sha256"] == explicit["state_view_sha256"]
    assert trace["schema_version"] == FROZEN_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION
    for key in ("current", "successor", "loss_weight", "sample_identity_sha256", "liquidation_relative_reward_bps"):
        assert _structured_sha256(default[key]) == _structured_sha256(trace[key])
    steps = trace["frozen_policy_trace"]["steps"]
    assert [step["current"]["state_index"] for step in steps] == [1, 2, 3, 4]
    assert [step["successor"]["state_index"] for step in steps] == [2, 3, 4, 5]
    assert steps[-1]["elapsed_wall_clock_gamma"] < steps[0]["elapsed_wall_clock_gamma"]
    assert steps[-1]["transition_closure"]["wall_clock_delta_seconds"] > 60
    for step in steps:
        assert "sample_identity_sha256" not in step
        assert not step["terminal_mask"].any()
        assert not step["right_censored_mask"].any()


def test_trace_at_observation_boundary_preserves_policy_hold(relative_fixture):
    batch, view = _batch(counts=(3, 3))
    assert len(view["frozen_policy_trace"]["steps"]) == 1
    assert batch["frozen_policy_trace"]["transition_available_mask"].tolist() == [[True, True, False, False, False]]
    assert batch["target_action_valid_mask"].all()
    terminal_batch, terminal_view = _batch(counts=(3, 3), terminals=(True, False))
    assert terminal_view["frozen_policy_trace"]["steps"][-1]["successor_terminal_mask"].tolist() == [True, False]
    assert terminal_batch["target_action_valid_mask"][-1].tolist() == [[False, True], [True, True]]


@pytest.mark.parametrize("mutation", ["reward", "link", "mask", "length", "steps"])
def test_resealed_corrupt_trace_is_rejected(relative_fixture, mutation):
    view = states._materialize(0, backup_steps=5)
    trace = dict(view["frozen_policy_trace"])
    trace["steps"] = [dict(s) for s in trace["steps"]]
    if mutation == "reward":
        reward = trace["steps"][0]["liquidation_relative_reward_bps"].copy()
        reward[0, 0] += 10
        reward.setflags(write=False)
        trace["steps"][0]["liquidation_relative_reward_bps"] = reward
    elif mutation == "link":
        trace["steps"][0]["current"] = dict(trace["steps"][0]["current"], state_index=3)
    elif mutation == "mask":
        terminal = np.array([True, False]); terminal.setflags(write=False)
        trace["steps"][0]["successor_terminal_mask"] = terminal
    elif mutation == "length":
        trace["steps"].pop()
    else:
        trace["backup_steps"] = True
    malformed = {**view, "frozen_policy_trace": trace}
    malformed.pop("state_view_sha256")
    malformed["state_view_sha256"] = _structured_sha256(malformed)
    with pytest.raises(RuntimeError):
        _validate(malformed)


def test_unproven_extra_market_gap_is_rejected(relative_fixture):
    states._materialize(0, known_gap=False, backup_steps=1)
    with pytest.raises(RuntimeError, match="SUCCESSOR_GAP_CENSORED"):
        states._materialize(0, known_gap=False, backup_steps=5)


def test_one_forward_and_backward_same_anchor_and_weights_with_multistep(relative_fixture):
    batch, _ = _batch()
    legacy = _collate()
    assert batch["schema_version"] == FROZEN_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION
    torch.testing.assert_close(batch["importance_weight"], legacy["importance_weight"], rtol=0, atol=0)
    assert batch["target_entry_batch_index"].tolist() == [1] * 6
    assert batch["transition_count"] == 1
    assert batch["frozen_policy_trace"]["successor_q_indices"].tolist() == [[0, 2, 3, 4, 5]]
    outcomes = []
    for current_batch in (legacy, batch):
        model = _TwoRowHead()
        target = copy.deepcopy(model).requires_grad_(False).eval()
        entry = torch.tensor([[2.], [3.], [4.]], requires_grad=True)
        result = run_random_access_training_step(model=model, target_model=target,
            entry_decision_representations=entry, target_entry_decision_representations=entry.detach(),
            batch=current_batch, grad_accum_steps=1)
        assert model.calls == 1
        assert target.calls == (3 if current_batch is batch else 1)
        assert result["backward_calls"] == 1
        assert all(p.grad is None for p in target.parameters())
        outcomes.append(result)
    torch.testing.assert_close(outcomes[0]["entry_targets"], outcomes[1]["entry_targets"], rtol=0, atol=0)
    assert outcomes[0]["entry_bridge_binding"] == outcomes[1]["entry_bridge_binding"]
    assert outcomes[1]["targets"][0, 0, 0] < outcomes[0]["targets"][0, 0, 0]
    assert outcomes[1]["targets"][0, 1, 0] > outcomes[0]["targets"][0, 1, 0]


@pytest.mark.parametrize("steps", [0, 2, 6, True])
def test_only_explicit_supported_backup_scope_is_accepted(relative_fixture, steps):
    with pytest.raises(RuntimeError, match="BACKUP_STEPS_INVALID"):
        states._materialize(0, backup_steps=steps)


@pytest.mark.parametrize("mutation", ["reward", "gamma", "index", "schema"])
def test_trace_batch_cannot_replace_bound_first_transition(relative_fixture, mutation):
    batch, _ = _batch()
    if mutation == "reward":
        batch["frozen_policy_trace"]["hold_reward_bps"][0, 0, 0] += 1
    elif mutation == "gamma":
        batch["frozen_policy_trace"]["elapsed_wall_clock_gamma"][0, 0] = .5
    elif mutation == "index":
        batch["frozen_policy_trace"]["successor_q_indices"][0, 0] = 1
    else:
        batch["schema_version"] = "gx1_unified_exit_random_access_train_batch_v2"
    model = _TwoRowHead()
    target = copy.deepcopy(model).requires_grad_(False).eval()
    with pytest.raises(RuntimeError, match="TRACE_FIRST_STEP_MISMATCH|BACKUP_POLICY_INVALID"):
        run_random_access_training_step(model=model, target_model=target,
            entry_decision_representations=torch.ones(3, 1),
            target_entry_decision_representations=torch.ones(3, 1), batch=batch, grad_accum_steps=1)
    assert all(p.grad is None for p in model.parameters())
