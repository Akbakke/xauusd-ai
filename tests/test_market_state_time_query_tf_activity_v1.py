"""Isolated TF activity readout mechanics; no native forward, market inputs or fit."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_time_query_model_contract,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import (
    NativeFutureDualModel, contract as dual_contract,
)
from tests.test_market_state_time_query_dedicated_m5_side_v1 import (
    WIDTH, LEGACY_TA_S_SHA, LEGACY_TA_DUAL_SHA, _backbone, _inputs,
)
from tests.test_market_state_time_query_non_m5_side_v1 import (
    M5_S_SHA, M5_DUAL_SHA, M5_EQUAL_LOSS_SHA, _same,
)

TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
# Existing native separate-side recipe identities, before this opt-in change.
# Pre-change 11aa70d6 with current anchor-span inputs: PARENT_CONTRACT_COMPARISON.json.
NON_M5_DUAL_SHA = "01e48b9078e1bccedd7bd1ee7a753faeda65a429488b58b572b60b025cd64709"
NON_M5_EQUAL_LOSS_SHA = "454539d254689205500c3118ba2dfc004c2c9cbd46ae401d2f9e536e66600f64"
IDENTITY_BUFFERS = {"state_model.query_objective_sha256", "native_F_objective_sha256"}
CLONES = {
    f"{destination}.{tf}.{kind}": f"{original}.{kind}"
    for destination, original in (
        ("state_model.tf_activity_heads", "state_model.query_head"),
        ("future_tf_activity_heads", "future_head"),
    )
    for tf in TIMEFRAMES for kind in ("weight", "bias")
}
OMITTED = object()


def _model(*, activity=True, non_m5=False, m5=True, ta_only=True):
    kwargs = {"dedicated_m5_side": m5, "dedicated_non_m5_side": non_m5}
    if activity is not OMITTED:
        kwargs["dedicated_tf_activity"] = activity
    return NativeFutureDualModel(FactorizedTimeQueryMarketStateModel(
        _backbone(ta_only=ta_only), **kwargs))


def _decode(model):
    hidden, query = _inputs()
    return (model.state_model.decode_raw_query_grid(hidden, query),
            model.decode_native_F_raw(hidden))


def _heads(model, task):
    if task == "state":
        return (model.state_model.query_head, model.state_model.m5_side_head,
                model.state_model.tf_activity_heads)
    return model.future_head, model.future_m5_side_head, model.future_tf_activity_heads


@pytest.mark.parametrize("ta_only,m5,non_m5", [
    (False, False, False), (True, False, False), (True, True, False), (True, True, True),
])
def test_false_option_preserves_existing_contracts_rng_state_keys_and_output(ta_only, m5, non_m5):
    flags = {"ta_only": ta_only, "dedicated_m5_side": m5, "dedicated_non_m5_side": non_m5}
    for contract in (factorized_time_query_model_contract, dual_contract, objective.contract):
        assert contract(**flags) == contract(**flags, dedicated_tf_activity=False)
        assert "dedicated_tf_activity" not in contract(**flags)
    if ta_only and not non_m5:
        assert factorized_time_query_model_contract(**flags)["contract_sha256"] == (
            M5_S_SHA if m5 else LEGACY_TA_S_SHA)
        assert dual_contract(**flags)["contract_sha256"] == (M5_DUAL_SHA if m5 else LEGACY_TA_DUAL_SHA)
    if m5:
        assert objective.contract(**flags, m5_equal_observation=True)["contract_sha256"] == (
            NON_M5_EQUAL_LOSS_SHA if non_m5 else M5_EQUAL_LOSS_SHA)
    if non_m5:
        assert dual_contract(**flags)["contract_sha256"] == NON_M5_DUAL_SHA
    torch.manual_seed(114)
    implicit = _model(activity=OMITTED, m5=m5, non_m5=non_m5, ta_only=ta_only)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(114)
    explicit = _model(activity=False, m5=m5, non_m5=non_m5, ta_only=ta_only)
    assert torch.equal(torch.get_rng_state(), rng)
    _same(explicit.state_dict(), implicit.state_dict())
    assert not any("tf_activity_heads" in name for name in explicit.state_dict())
    _same(_decode(explicit), _decode(implicit))


def test_activity_clone_adds_twenty_independent_tensors_preserving_side_and_rng():
    torch.manual_seed(94)
    original = _model(activity=False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(94)
    trial = _model()
    assert torch.equal(torch.get_rng_state(), rng)
    old_params, new_params = dict(original.named_parameters()), dict(trial.named_parameters())
    assert set(new_params) - set(old_params) == set(CLONES)
    assert len(CLONES) == 20 and sum(new_params[name].numel() for name in CLONES) == 1290
    for name, value in original.state_dict().items():
        if name in IDENTITY_BUFFERS:
            assert not torch.equal(trial.state_dict()[name], value)
        else:
            assert torch.equal(trial.state_dict()[name], value), name
    pointers = set()
    existing = {new_params[name].untyped_storage().data_ptr() for name in old_params}
    for name, source in CLONES.items():
        value = new_params[name]
        assert torch.equal(value, new_params[source][0:1]), name
        assert value.untyped_storage().data_ptr() not in pointers | existing
        pointers.add(value.untyped_storage().data_ptr())
    assert trial.state_model._dedicated_tf_activity is True
    assert trial.state_model._dedicated_non_m5_side is False
    for task in ("state", "future"):
        assert type(_heads(trial, task)[2]) is nn.ModuleDict
        assert tuple(_heads(trial, task)[2]) == TIMEFRAMES
    for old, new in zip(_decode(original), _decode(trial)):
        assert torch.equal(new[..., 1], old[..., 1])
        torch.testing.assert_close(new[..., 0], old[..., 0], rtol=1e-6, atol=2e-6)
    s = factorized_time_query_model_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True)
    f = dual_contract(ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True)
    assert trial.state_model.research_objective == s and trial.research_objective == f
    assert f["S_model_contract_sha256"] == s["contract_sha256"]
    assert s["contract_sha256"] != M5_S_SHA and f["contract_sha256"] != M5_DUAL_SHA
    trial._require_binding()


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("tf", TIMEFRAMES)
@pytest.mark.parametrize("non_m5", [False, True])
def test_activity_bias_perturbation_changes_only_its_task_timeframe_and_channel(task, tf, non_m5):
    model = _model(non_m5=non_m5)
    before = _decode(model)
    with torch.no_grad():
        _heads(model, task)[2][tf].bias.add_(0.75)
    after = _decode(model)
    selected = 0 if task == "state" else 1
    assert torch.equal(before[1-selected], after[1-selected])
    old, new = before[selected], after[selected]
    index = TIMEFRAMES.index(tf)
    keep = torch.ones_like(old, dtype=torch.bool)
    keep[..., index, 0] = False
    assert torch.equal(old[keep], new[keep])
    torch.testing.assert_close(new[..., index, 0] - old[..., index, 0],
                               torch.full_like(old[..., index, 0], 0.75), rtol=0, atol=1e-6)


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("tf", TIMEFRAMES)
@pytest.mark.parametrize("channel", [0, 1])
@pytest.mark.parametrize("non_m5", [False, True])
def test_gradient_routes_to_one_activity_head_or_unchanged_side_owner(task, tf, channel, non_m5):
    torch.manual_seed(53)
    model = _model(non_m5=non_m5)
    hidden, query = _inputs()
    hidden.requires_grad_(True)
    raw = (model.state_model.decode_raw_query_grid(hidden, query)[:, 0]
           if task == "state" else model.decode_native_F_raw(hidden))
    index = TIMEFRAMES.index(tf)
    raw[:, index, channel].sum().backward()
    shared, m5, activities = _heads(model, task)
    sides = (model.state_model.non_m5_side_heads if task == "state"
             else model.future_non_m5_side_heads) if non_m5 else {}
    selected = activities[tf] if channel == 0 else (m5 if tf == "M5" else sides[tf] if non_m5 else shared)
    def nonzero(value):
        return value is not None and bool(torch.count_nonzero(value))
    for head in (shared, m5, *activities.values(), *sides.values()):
        assert nonzero(head.weight.grad) == (head is selected)
        assert nonzero(head.bias.grad) == (head is selected)
    if shared.weight.grad is not None:
        assert not nonzero(shared.weight.grad[0]) and not nonzero(shared.bias.grad[0])
    other = _heads(model, "future" if task == "state" else "state")
    other_sides = (model.future_non_m5_side_heads if task == "state"
                   else model.state_model.non_m5_side_heads) if non_m5 else {}
    assert all(p.grad is None for head in (other[0], other[1], *other[2].values(), *other_sides.values())
               for p in head.parameters())
    assert nonzero(hidden.grad)
    assert nonzero(model.state_model.query_projection[0].weight.grad)


@pytest.mark.parametrize("value", [None, 0, 1, "true", np.bool_(True)])
def test_activity_option_requires_literal_boolean(value):
    for contract in (factorized_time_query_model_contract, dual_contract, objective.contract):
        with pytest.raises(ValueError, match="TF_ACTIVITY_MODE"):
            contract(ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=value)
    with pytest.raises(RuntimeError, match="TF_ACTIVITY_MODE"):
        _model(activity=value)


@pytest.mark.parametrize("ta_only,m5,non_m5", [
    (False, False, False), (False, True, False), (True, False, False),
])
def test_activity_variant_requires_ta_and_m5(ta_only, m5, non_m5):
    for contract in (factorized_time_query_model_contract, dual_contract, objective.contract):
        with pytest.raises(ValueError, match="MODE"):
            contract(ta_only=ta_only, dedicated_m5_side=m5,
                     dedicated_non_m5_side=non_m5, dedicated_tf_activity=True)
    with pytest.raises(RuntimeError, match="MODE"):
        _model(ta_only=ta_only, m5=m5, non_m5=non_m5)


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("fault", [
    "missing", "extra_key", "order", "shape", "type", "dtype", "nonfinite", "parameter_type",
    "aliased_module", "aliased_storage", "shared_alias", "contract", "unexpected", "container",
])
def test_malformed_activity_heads_and_identity_fail_closed(task, fault):
    model = _model(activity=fault != "unexpected")
    owner = model.state_model if task == "state" else model
    name = "tf_activity_heads" if task == "state" else "future_tf_activity_heads"
    if fault == "unexpected":
        setattr(owner, name, nn.ModuleDict({tf: nn.Linear(WIDTH, 1) for tf in TIMEFRAMES}))
    else:
        heads = getattr(owner, name)
        if fault == "missing":
            del heads["H1"]
        elif fault == "extra_key":
            heads["W1"] = nn.Linear(WIDTH, 1)
        elif fault == "order":
            setattr(owner, name, nn.ModuleDict(reversed(list(heads.items()))))
        elif fault == "shape":
            heads["D1"].weight = nn.Parameter(torch.zeros(2, WIDTH))
        elif fault == "type":
            heads["H1"] = nn.Identity()
        elif fault == "dtype":
            heads["H4"].double()
        elif fault == "nonfinite":
            with torch.no_grad():
                heads["M15"].bias.fill_(float("nan"))
        elif fault == "parameter_type":
            value = heads["M5"].weight.detach().clone()
            del heads["M5"].weight
            heads["M5"].register_buffer("weight", value)
        elif fault == "aliased_module":
            heads["H1"] = heads["M15"]
        elif fault == "aliased_storage":
            heads["H1"].weight = nn.Parameter(heads["M15"].weight.view(1, WIDTH))
        elif fault == "shared_alias":
            heads["D1"].weight = nn.Parameter(_heads(model, task)[0].weight[0:1])
        elif fault == "contract":
            getattr(owner, "query_objective_sha256" if task == "state"
                    else "native_F_objective_sha256")[0] ^= 1
        else:
            setattr(owner, name, nn.ModuleList(list(heads.values())))
    with pytest.raises(RuntimeError, match="BINDING|BUFFER|TF_ACTIVITY"):
        model._require_binding()


@pytest.mark.parametrize("source", ["activity", "shared", "backbone", "query_buffer"])
def test_activity_alias_to_other_task_or_backbone_is_rejected(source):
    model = _model()
    if source == "activity":
        model.future_tf_activity_heads["M5"].bias = model.state_model.tf_activity_heads["D1"].bias
    elif source == "shared":
        model.state_model.tf_activity_heads["M5"].bias = nn.Parameter(model.future_head.bias[0:1])
    elif source == "backbone":
        model.future_tf_activity_heads["M5"].bias = nn.Parameter(model.backbone.entry_q_joint_in.bias[0:1])
    else:
        model.future_tf_activity_heads["M5"].bias = nn.Parameter(model.state_model.query_tf_coordinates[0, 0:1])
    with pytest.raises(RuntimeError, match="TF_ACTIVITY_HEAD_ALIASES"):
        model._require_binding()


@pytest.mark.parametrize("value", [False, 1, None])
def test_changed_activity_flag_cannot_relabel_existing_parameters(value):
    model = _model()
    model.state_model._dedicated_tf_activity = value
    with pytest.raises(RuntimeError, match="TF_ACTIVITY"):
        model._require_binding()


def _loss_case(*, empty=False):
    allowed = np.array([[1, 2, 4, 5, 7], [4, 1, 2, 7, 5], [5, 4, 1, 2, 7]], np.uint8)
    eligible, mature = np.ones((3, 5), bool), np.ones((3, 5), bool)
    eligible[1, 1], mature[0, 2], eligible[:, 4] = False, False, False
    if empty:
        eligible[:] = False
    truth_s = {"allowed": allowed, "eligible": eligible, "mature": mature}
    truth_f = {"allowed": allowed[::-1].copy(), "eligible": eligible, "mature": mature}
    return dict(state_truth=truth_s, future_truth=truth_f,
                state_query_k=_inputs()[1][:, 1], future_weight=0.375,
                ta_only=True, dedicated_m5_side=True)


@pytest.mark.parametrize("equal_observation", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("non_m5", [False, True])
def test_activity_metadata_preserves_original_masked_loss_and_all_raw_gradients(equal_observation, empty, non_m5):
    kwargs = dict(_loss_case(empty=empty), dedicated_non_m5_side=non_m5)
    raws = (torch.linspace(-1, 2, 30).reshape(3, 5, 2),
            torch.linspace(0.6, -1.7, 30).reshape(3, 5, 2))
    reports, gradients = [], []
    for activity in (False, True):
        s, f = (value.clone().requires_grad_(True) for value in raws)
        report = objective.combined_loss(
            state_raw=s, future_raw=f, **kwargs, dedicated_tf_activity=activity,
            m5_equal_observation=equal_observation)
        reports.append(report)
        gradients.append(torch.autograd.grad(report["loss"], (s, f)))
        assert s.grad is None and f.grad is None
        assert report["skip_update"] is empty
    for key in reports[0].keys() - {"contract_sha256"}:
        _same(reports[1][key], reports[0][key])
    _same(gradients[1], gradients[0])
    for j, task in enumerate(("state", "future")):
        truth = kwargs[task+"_truth"]
        info = truth["eligible"] & truth["mature"] & (truth["allowed"] != 7)
        assert not torch.count_nonzero(gradients[1][j][~torch.from_numpy(info)])
    old, new = (objective.contract(
        ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=enabled,
        dedicated_non_m5_side=non_m5, m5_equal_observation=equal_observation)
        for enabled in (False, True))
    assert old["contract_sha256"] != new["contract_sha256"]
    assert reports[1]["contract_sha256"] == new["contract_sha256"]
    assert new["dual_model_contract_sha256"] == dual_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True,
        dedicated_non_m5_side=non_m5)["contract_sha256"]
    for key in old.keys() - {"contract_sha256", "dual_model_contract_sha256"}:
        _same(new[key], old[key])


def test_combined_loss_rejects_invalid_activity_flag():
    raw = torch.zeros(3, 5, 2)
    with pytest.raises(ValueError, match="TF_ACTIVITY_MODE"):
        objective.combined_loss(state_raw=raw, future_raw=raw, **_loss_case(), dedicated_tf_activity=1)


@pytest.mark.parametrize("non_m5,equal_observation", [(False, False), (False, True), (True, False), (True, True)])
def test_existing_recipes_reject_activity_model_before_optimizer_creation(monkeypatch, non_m5, equal_observation):
    model = _model()
    before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        pytest.fail("old recipe must fail before AdamW construction")
    recipe = fit.ta_sf_numerical_recipe(
        dedicated_m5_side=True, dedicated_non_m5_side=non_m5,
        m5_equal_observation=equal_observation)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(RuntimeError, match="TA_SF_HEAD_BINDING|TA_SF_NUMERICAL_RECIPE"):
        state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    assert before == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(parameter.grad is None for parameter in model.parameters())
