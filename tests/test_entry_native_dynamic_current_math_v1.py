"""Current-state encoding and loss algebra only: no model, optimizer or backward."""
from copy import deepcopy
import math

import pytest
import torch

from gx1.contracts import entry_native_direction_research_v1 as owner


def test_current_encoding_preserves_unknown_and_independent_path_mask():
    state = torch.tensor([[0, 1, 2, -1, -1], [2, 1, 0, -1, 1]], dtype=torch.int8)
    observed = state >= 0
    path = torch.tensor([[True, False, True, True, False], [False, True, False, False, True]])
    encoded = owner.encode_historical_current_state(state, observed, path, batch=2, tfs=5, device=state.device)
    expected = torch.tensor([
        [[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]],
        [[0, 0, 1, 1, 0], [0, 1, 0, 1, 1], [1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 1]],
    ], dtype=torch.float32)
    assert torch.equal(encoded, expected)
    assert torch.equal(encoded[:, :, :3].sum(-1).bool(), observed)


@pytest.mark.parametrize("fault", ["unknown_as_observed", "known_as_unknown", "bad_state", "bad_mask_dtype", "bad_shape"])
def test_current_encoding_rejects_inconsistent_input(fault):
    state = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.int8)
    observed = state >= 0
    path = torch.zeros_like(observed)
    if fault == "unknown_as_observed": observed[0, 3] = True
    if fault == "known_as_unknown": observed[0, 1] = False
    if fault == "bad_state": state[0, 0] = 3
    if fault == "bad_mask_dtype": path = path.to(torch.int8)
    if fault == "bad_shape": path = path[:, :4]
    with pytest.raises(RuntimeError):
        owner.encode_historical_current_state(state, observed, path, batch=1, tfs=5, device=state.device)


def test_new_objective_is_explicit_and_prior_objective_is_unchanged():
    old = owner.dynamic_mtf_state_research_contract()
    assert old["contract_sha256"] == "fdfc47ed7b1d6ddbacdd2a8388a09049d0017555d8f79afaa88b5f200e8f532b"
    new = owner.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=1.0)
    assert new["contract_sha256"] != old["contract_sha256"]
    assert new["first_state_loss_weight"] == 1.0
    assert new["current_input_available_online"] is False
    assert new["target_contract"] == old["target_contract"]
    assert new["timeframes"] == old["timeframes"] and new["lead_minutes"] == old["lead_minutes"]


@pytest.mark.parametrize("weight", [None, 0.0, -1.0, float("inf"), float("nan"), 1, True])
def test_new_objective_rejects_unbound_or_invalid_first_state_weight(weight):
    with pytest.raises(RuntimeError, match="STATE_FIRST_LOSS_WEIGHT"):
        owner.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=weight)


def _small_loss(monkeypatch):
    contract = owner.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=2.0)
    contract["lead_minutes"] = [1, 2, 3]
    monkeypatch.setattr(owner, "dynamic_mtf_state_research_contract", lambda **kwargs: deepcopy(contract))
    minute = 60 * 10**9
    issue = torch.tensor([100 * minute], dtype=torch.int64)
    validity = issue[:, None] + torch.tensor([[1, 2, 3]], dtype=torch.int64) * minute
    initial = torch.tensor([[[0.0, 0.7, -0.3], [0.2, 0.0, 0.6], [0.8, -0.4, 0.1],
                             [-0.1, 0.2, 0.4], [0.1, 0.6, -0.2]]], dtype=torch.float32)
    states = torch.tensor([[[1, 1, 1], [-1, 2, 2], [0, 2, 2], [-1, 1, 1], [2, 1, 1]]], dtype=torch.int8)
    return dict(initial_state_logits=initial, state_transition_logits=torch.zeros(1, 5, 2, 3, 3),
                state=states, observed=states >= 0, issue_ns=issue, validity_ns=validity,
                maturity_ns=validity[:, None].expand(1, 5, 3).clone(), cutoff_ns=103 * minute)


def test_weighted_first_state_loss_is_exact_and_ignores_missing_and_immature(monkeypatch):
    args = _small_loss(monkeypatch)
    base = owner.native_dynamic_mtf_state_loss(**args)
    result = owner.native_dynamic_mtf_state_loss(**args, first_state_loss_weight=2.0)
    expected = -torch.log_softmax(args["initial_state_logits"], -1)[0, [0, 2, 4], [1, 0, 2]].mean()
    assert result["first_state_eligible_cells"] == 3
    torch.testing.assert_close(result["sequence_loss"], base["loss"])
    torch.testing.assert_close(result["first_state_loss"], expected)
    torch.testing.assert_close(result["loss"], base["loss"] + 2.0 * expected)
    assert result["counts"]["immature_observed_cells"] == 5


def test_missing_first_state_and_empty_labels_do_not_invent_supervision(monkeypatch):
    args = _small_loss(monkeypatch)
    args["state"][:, :, 0] = -1
    args["observed"][:, :, 0] = False
    base = owner.native_dynamic_mtf_state_loss(**args)
    result = owner.native_dynamic_mtf_state_loss(**args, first_state_loss_weight=2.0)
    assert result["first_state_eligible_cells"] == 0 and result["first_state_loss"] is None
    torch.testing.assert_close(result["loss"], base["loss"])
    args["state"].fill_(-1)
    args["observed"].fill_(False)
    empty = owner.native_dynamic_mtf_state_loss(**args, first_state_loss_weight=2.0)
    assert empty["loss"] is None and empty["update_allowed"] is False
