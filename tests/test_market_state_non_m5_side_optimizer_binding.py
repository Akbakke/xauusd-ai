"""Opt-in numerical/AdamW binding on tiny synthetic models; no forward, fit or data."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_time_query_native_F_model_v1 import contract as model_contract
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    COMMIT, DEVICE, ORDER, _capture, _seed,
)
from tests.test_market_state_time_query_non_m5_side_v1 import (
    CLONES, _state_pair, _snapshot, _same,
)

OLD_EQUAL_RECIPE_SHA = "5085579ff8ad90257412ad1d42406c6c481a1d272f0cf2849ae293da87974361"


def _recipe(*, enabled=True, equal=False):
    return fit.ta_sf_numerical_recipe(
        dedicated_m5_side=True, dedicated_non_m5_side=enabled,
        m5_equal_observation=equal)


@pytest.mark.parametrize("m5,equal", [(False, False), (True, False), (True, True)])
def test_false_option_preserves_existing_recipe_bytes(m5, equal):
    original = fit.ta_sf_numerical_recipe(
        dedicated_m5_side=m5, m5_equal_observation=equal)
    explicit = fit.ta_sf_numerical_recipe(
        dedicated_m5_side=m5, m5_equal_observation=equal, dedicated_non_m5_side=False)
    assert json.dumps(original, sort_keys=True, separators=(",", ":")) == json.dumps(
        explicit, sort_keys=True, separators=(",", ":"))
    assert "dedicated_non_m5_side" not in original
    if equal:
        assert original["contract_sha256"] == OLD_EQUAL_RECIPE_SHA


@pytest.mark.parametrize("equal", [False, True])
def test_trial_recipe_changes_only_bound_model_loss_and_recipe_identities(equal):
    original, trial = _recipe(enabled=False, equal=equal), _recipe(equal=equal)
    assert original.keys() == trial.keys()
    assert {name for name in trial if trial[name] != original[name]} == {
        "research_objective_sha256", "native_F_loss_contract_sha256", "contract_sha256"}
    assert trial["research_objective_sha256"] == model_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True)["contract_sha256"]
    assert trial["native_F_loss_contract_sha256"] == objective.contract(
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True,
        m5_equal_observation=equal)["contract_sha256"]
    assert trial["optimizer"] == original["optimizer"] and trial["future_weight"] == 1.0
    assert trial["contract_sha256"] == fit.campaign.canonical_sha256(
        {key: value for key, value in trial.items() if key != "contract_sha256"})


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_non_boolean_flag_is_rejected(value):
    with pytest.raises(ValueError, match="DEDICATED_NON_M5_SIDE_MODE"):
        fit.ta_sf_numerical_recipe(dedicated_m5_side=True, dedicated_non_m5_side=value)


def test_non_m5_recipe_requires_dedicated_m5():
    with pytest.raises(ValueError, match="DEDICATED_NON_M5_SIDE_MODE"):
        fit.ta_sf_numerical_recipe(dedicated_non_m5_side=True)


@pytest.mark.parametrize("equal", [False, True])
def test_fresh_optimizer_owns_exactly_old_active_plus_sixteen_new_tensors(monkeypatch, equal):
    source, model = _state_pair(monkeypatch)
    _, old_active, old_inactive = state._parameters(source)
    before_source = _snapshot(source)
    recipe = _recipe(equal=equal)
    rng = torch.get_rng_state().clone()
    optimizer, binding = state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    _same(_snapshot(source), before_source)
    assert torch.equal(torch.get_rng_state(), rng)
    assert type(optimizer) is torch.optim.AdamW and not optimizer.state
    assert optimizer.state_dict()["state"] == {}
    params = dict(model.named_parameters())
    expected_active = {name for name, _ in old_active} | set(CLONES)
    expected_inactive = {name for name, _ in old_inactive}
    assert len(CLONES) == 16
    assert set(binding["active_parameter_names"]) == expected_active
    assert set(binding["inactive_parameter_names"]) == expected_inactive
    assert expected_active.isdisjoint(expected_inactive)
    assert expected_active | expected_inactive == set(params)
    assert {name for name, parameter in params.items() if not parameter.requires_grad} == expected_inactive
    assert all(params[name].requires_grad for name in CLONES)
    assert sum(params[name].numel() for name in CLONES) == 8 * (model.state_model._hidden_width + 1)
    grouped = optimizer.param_groups[0]["params"]
    assert [id(parameter) for parameter in grouped] == [
        id(params[name]) for name in binding["active_parameter_names"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert len(grouped) == len({parameter.untyped_storage().data_ptr() for parameter in grouped})
    assert all(parameter.grad is None for parameter in model.parameters())
    assert binding["numerical_recipe"] == recipe
    assert binding["research_objective"] == model.research_objective
    assert not any(binding[name] for name in ("ema_created", "scheduler_created", "target_model_created"))


@pytest.mark.parametrize("equal", [False, True])
def test_zero_update_full_state_roundtrip_preserves_new_heads_empty_optimizer_modes_and_rng(monkeypatch, equal):
    _seed()
    _, model = _state_pair(monkeypatch)
    recipe = _recipe(equal=equal)
    optimizer, binding = state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    model.eval()
    model.state_model.non_m5_side_heads["H1"].train()
    model.future_non_m5_side_heads["D1"].train()
    _seed()
    saved = _capture(model, optimizer, recipe, 0)
    assert saved["optimizer_state"]["state"] == {}
    assert set(CLONES) <= set(saved["model_state"])
    _, restored = _state_pair(monkeypatch)
    restored_opt, restored_binding = state.build_fold1_fit_optimizer(restored, numerical_recipe=recipe)
    assert restored_binding == binding
    identities = {name: id(parameter) for name, parameter in restored.named_parameters()}
    progress = state.restore_fold1_fit_state(
        saved, restored, restored_opt, numerical_recipe=recipe,
        feature_view_identity=restored.state_model.feature_view_identity,
        parent_order=ORDER, source_commit=COMMIT, device=DEVICE)
    assert progress == {"cursor": 0, "optimizer_updates": 0, "skipped_batches": 0}
    actual = _capture(restored, restored_opt, recipe, 0)
    _same(actual, saved)
    assert not restored_opt.state
    assert identities == {name: id(parameter) for name, parameter in restored.named_parameters()}
    assert all(parameter.grad is None for parameter in restored.parameters())


@pytest.mark.parametrize("equal", [False, True])
@pytest.mark.parametrize("direction", ["old_recipe_new_model", "new_recipe_old_model"])
def test_wrong_variant_recipe_fails_before_optimizer_or_parameter_mutation(monkeypatch, equal, direction):
    source, trial = _state_pair(monkeypatch)
    model = trial if direction == "old_recipe_new_model" else source
    recipe = _recipe(enabled=direction != "old_recipe_new_model", equal=equal)
    before, rng = _snapshot(model), torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        pytest.fail("wrong variant reached AdamW construction")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(RuntimeError, match="TA_SF_NUMERICAL_RECIPE"):
        state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    _same(_snapshot(model), before)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("fault", ["model_contract", "loss_contract", "learning_rate", "extra_flag"])
def test_resealed_wrong_recipe_is_rejected_before_optimizer(monkeypatch, fault):
    _, model = _state_pair(monkeypatch)
    recipe, original = _recipe(equal=True), _recipe(enabled=False, equal=True)
    if fault == "model_contract":
        recipe["research_objective_sha256"] = original["research_objective_sha256"]
    elif fault == "loss_contract":
        recipe["native_F_loss_contract_sha256"] = original["native_F_loss_contract_sha256"]
    elif fault == "learning_rate":
        recipe["optimizer"] = copy.deepcopy(recipe["optimizer"])
        recipe["optimizer"]["kwargs"]["lr"] *= 2
    else:
        recipe["dedicated_non_m5_side"] = "true"
    recipe["contract_sha256"] = fit.campaign.canonical_sha256(
        {key: value for key, value in recipe.items() if key != "contract_sha256"})
    before = _snapshot(model)
    def forbidden(*args, **kwargs):
        pytest.fail("forged recipe reached AdamW construction")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(RuntimeError, match="TA_SF_NUMERICAL_RECIPE"):
        state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    _same(_snapshot(model), before)


def test_cross_task_parameter_alias_is_rejected_before_optimizer(monkeypatch):
    _, model = _state_pair(monkeypatch)
    model.future_non_m5_side_heads["D1"].weight = model.state_model.non_m5_side_heads["M15"].weight
    before = _snapshot(model)
    def forbidden(*args, **kwargs):
        pytest.fail("aliased parameters reached AdamW construction")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(RuntimeError, match="PARAMETER_ALIASES"):
        state.build_fold1_fit_optimizer(model, numerical_recipe=_recipe())
    _same(_snapshot(model), before)
