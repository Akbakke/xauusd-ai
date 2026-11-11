"""Synthetic opt-in readout mechanics; no native forward, market data or fit."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as digest
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_time_query_model_contract,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import (
    NativeFutureDualModel, contract as dual_contract,
)
from tests.test_market_state_time_query_dedicated_m5_side_v1 import (
    WIDTH, LEGACY_TA_S_SHA, LEGACY_TA_DUAL_SHA, _backbone, _inputs,
    _model as legacy_model,
)
from tests.test_market_state_ta_dedicated_m5_binding_v1 import _model as state_model

TIMEFRAMES = ("M15", "H1", "H4", "D1")
M5_S_SHA = "aa7e6919bc7326a766b10f7a35843a4a7e45ca601aad0c3a817c26f976b07dae"
M5_DUAL_SHA = "b7066eba10673122ec94765be414134da11946052c1bf74b7901b0001c8a0300"
M5_EQUAL_LOSS_SHA = "38cf7894de7cbe7de405ed7db9a073113ecfeb8888114dbe5548e4d13356adc0"
IDENTITY_BUFFERS = {"state_model.query_objective_sha256", "native_F_objective_sha256"}
CLONES = {
    f"{destination}.{tf}.{kind}": f"{original}.{kind}"
    for destination, original in (
        ("state_model.non_m5_side_heads", "state_model.query_head"),
        ("future_non_m5_side_heads", "future_head"),
    )
    for tf in TIMEFRAMES for kind in ("weight", "bias")
}


def _model(*, non_m5=True, m5=True, ta_only=True):
    return NativeFutureDualModel(FactorizedTimeQueryMarketStateModel(
        _backbone(ta_only=ta_only), dedicated_m5_side=m5,
        dedicated_non_m5_side=non_m5))


def _decode(model):
    hidden, query = _inputs()
    hidden = hidden[:, :model.state_model._hidden_width]
    return (model.state_model.decode_raw_query_grid(hidden, query),
            model.decode_native_F_raw(hidden))


def _heads(model, task):
    if task == "state":
        return (model.state_model.query_head, model.state_model.m5_side_head,
                model.state_model.non_m5_side_heads)
    return model.future_head, model.future_m5_side_head, model.future_non_m5_side_heads


def _same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _same(a, b)
    else:
        assert actual == expected


def _snapshot(model):
    return {
        "state": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        "modes": {name: module.training for name, module in model.named_modules()},
        "flags": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
        "parameter_ids": {name: id(parameter) for name, parameter in model.named_parameters()},
        "grads": {name: None if parameter.grad is None else parameter.grad.clone()
                  for name, parameter in model.named_parameters()},
        "view": copy.deepcopy(getattr(model.state_model, "feature_view_identity", None)),
    }


def _state_pair(monkeypatch):
    source = state_model(monkeypatch, True)
    target = NativeFutureDualModel(FactorizedTimeQueryMarketStateModel(
        copy.deepcopy(source.backbone), dedicated_m5_side=True, dedicated_non_m5_side=True))
    target.state_model.feature_view_identity = copy.deepcopy(source.state_model.feature_view_identity)
    target.state_model.feature_view_identity.update(
        contract_sha256="4" * 64,
        research_objective_sha256=target.state_model.research_objective["contract_sha256"])
    return source, target


@pytest.mark.parametrize("ta_only,m5", [(False, False), (True, False), (True, True)])
def test_false_option_preserves_contracts_initialization_rng_state_keys_and_outputs(ta_only, m5):
    flags = {"ta_only": ta_only, "dedicated_m5_side": m5}
    for contract in (factorized_time_query_model_contract, dual_contract, objective.contract):
        assert contract(**flags) == contract(**flags, dedicated_non_m5_side=False)
        assert "dedicated_non_m5_side" not in contract(**flags)
    if ta_only:
        assert factorized_time_query_model_contract(**flags)["contract_sha256"] == (
            M5_S_SHA if m5 else LEGACY_TA_S_SHA)
        assert dual_contract(**flags)["contract_sha256"] == (
            M5_DUAL_SHA if m5 else LEGACY_TA_DUAL_SHA)
    torch.manual_seed(174)
    old = legacy_model(ta_only=ta_only, dedicated_m5_side=m5)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(174)
    new = _model(ta_only=ta_only, m5=m5, non_m5=False)
    assert torch.equal(torch.get_rng_state(), old_rng)
    _same(new.state_dict(), old.state_dict())
    assert not any("non_m5_side_heads" in key for key in new.state_dict())
    _same(_decode(new), _decode(old))
    assert objective.contract(ta_only=True, dedicated_m5_side=True,
                              m5_equal_observation=True)["contract_sha256"] == M5_EQUAL_LOSS_SHA


def test_opt_in_clones_eight_independent_rows_preserving_existing_tensors_and_rng():
    torch.manual_seed(96)
    old = _model(non_m5=False)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(96)
    new = _model()
    assert torch.equal(torch.get_rng_state(), rng)
    old_params, new_params = dict(old.named_parameters()), dict(new.named_parameters())
    assert set(new_params) - set(old_params) == set(CLONES)
    assert sum(new_params[name].numel() for name in CLONES) == 1032
    for name, value in old.state_dict().items():
        if name not in IDENTITY_BUFFERS:
            assert torch.equal(new.state_dict()[name], value), name
    pointers = set()
    for name, original in CLONES.items():
        actual, shared = new_params[name], new_params[original]
        assert torch.equal(actual, shared[1:2]), name
        pointer = actual.untyped_storage().data_ptr()
        assert pointer != shared.untyped_storage().data_ptr() and pointer not in pointers
        pointers.add(pointer)
    for task in ("state", "future"):
        assert type(_heads(new, task)[2]) is nn.ModuleDict
        assert tuple(_heads(new, task)[2]) == TIMEFRAMES
    for a, b in zip(_decode(old), _decode(new)):
        assert torch.equal(a[..., 0], b[..., 0])
        assert torch.equal(a[..., 0, 1], b[..., 0, 1])
        torch.testing.assert_close(a[..., 1:, 1], b[..., 1:, 1], rtol=0, atol=1e-6)
    s = factorized_time_query_model_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True)
    f = dual_contract(ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True)
    assert new.state_model._dedicated_non_m5_side is True
    assert new.state_model.research_objective == s
    assert new.research_objective == f
    assert f["S_model_contract_sha256"] == s["contract_sha256"]
    assert s["contract_sha256"] != M5_S_SHA and f["contract_sha256"] != M5_DUAL_SHA
    new._require_binding()


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("tf", TIMEFRAMES)
def test_side_bias_perturbation_changes_only_its_task_and_timeframe(task, tf):
    model = _model()
    before = _decode(model)
    with torch.no_grad():
        _heads(model, task)[2][tf].bias.add_(0.75)
    after = _decode(model)
    changed = 0 if task == "state" else 1
    assert torch.equal(before[1 - changed], after[1 - changed])
    a, b = before[changed], after[changed]
    index = TIMEFRAMES.index(tf) + 1
    keep = torch.ones_like(a, dtype=torch.bool)
    keep[..., index, 1] = False
    assert torch.equal(a[keep], b[keep])
    torch.testing.assert_close(b[..., index, 1] - a[..., index, 1],
                               torch.full_like(a[..., index, 1], 0.75), rtol=0, atol=1e-6)


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("coordinate", ("M5",) + TIMEFRAMES + ("activity",))
def test_each_output_routes_gradient_only_to_its_head_and_shared_representation(task, coordinate):
    torch.manual_seed(13)
    model = _model()
    hidden, query = _inputs()
    hidden.requires_grad_(True)
    raw = (model.state_model.decode_raw_query_grid(hidden, query)[:, 0]
           if task == "state" else model.decode_native_F_raw(hidden))
    shared, m5, sides = _heads(model, task)
    if coordinate == "activity":
        selected, expected = raw[..., 0], shared
    else:
        index = (("M5",) + TIMEFRAMES).index(coordinate)
        selected = raw[:, index, 1]
        expected = m5 if coordinate == "M5" else sides[coordinate]
    selected.sum().backward()
    def nonzero(value):
        return value is not None and bool(torch.count_nonzero(value))
    for head in (shared, m5, *sides.values()):
        assert nonzero(head.weight.grad) == (head is expected)
        assert nonzero(head.bias.grad) == (head is expected)
    if coordinate == "activity":
        assert nonzero(shared.weight.grad[0]) and not nonzero(shared.weight.grad[1])
        assert nonzero(shared.bias.grad[0]) and not nonzero(shared.bias.grad[1])
    other = _heads(model, "future" if task == "state" else "state")
    assert all(p.grad is None for head in (other[0], other[1], *other[2].values())
               for p in head.parameters())
    assert nonzero(model.state_model.query_projection[0].weight.grad)
    assert nonzero(hidden.grad)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_non_boolean_option_is_rejected(value):
    for contract in (factorized_time_query_model_contract, dual_contract, objective.contract):
        with pytest.raises(ValueError, match="NON_M5"):
            contract(ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=value)
    with pytest.raises(RuntimeError, match="NON_M5"):
        _model(non_m5=value)


@pytest.mark.parametrize("ta_only,m5", [(False, False), (True, False)])
def test_option_requires_both_ta_and_dedicated_m5(ta_only, m5):
    for contract in (factorized_time_query_model_contract, dual_contract, objective.contract):
        with pytest.raises(ValueError, match="NON_M5"):
            contract(ta_only=ta_only, dedicated_m5_side=m5, dedicated_non_m5_side=True)
    with pytest.raises(RuntimeError, match="NON_M5"):
        _model(ta_only=ta_only, m5=m5)


@pytest.mark.parametrize("task", ["state", "future"])
@pytest.mark.parametrize("fault", ["missing", "extra_key", "shape", "dtype",
                                   "nonfinite", "contract", "unexpected", "container", "aliased"])
def test_damaged_head_or_contract_fails_closed(task, fault):
    model = _model(non_m5=fault != "unexpected")
    owner = model.state_model if task == "state" else model
    name = "non_m5_side_heads" if task == "state" else "future_non_m5_side_heads"
    if fault == "unexpected":
        setattr(owner, name, nn.ModuleDict({tf: nn.Linear(WIDTH, 1) for tf in TIMEFRAMES}))
    elif fault == "missing":
        del getattr(owner, name)["H1"]
    elif fault == "extra_key":
        getattr(owner, name)["M5"] = nn.Linear(WIDTH, 1)
    elif fault == "shape":
        getattr(owner, name)["D1"].weight = nn.Parameter(torch.zeros(2, WIDTH))
    elif fault == "dtype":
        getattr(owner, name)["H4"].double()
    elif fault == "nonfinite":
        with torch.no_grad():
            getattr(owner, name)["M15"].bias.fill_(float("nan"))
    elif fault == "aliased":
        getattr(owner, name)["H1"] = getattr(owner, name)["M15"]
    elif fault == "contract":
        getattr(owner, "query_objective_sha256" if task == "state"
                else "native_F_objective_sha256")[0] ^= 1
    else:
        setattr(owner, name, nn.ModuleList(list(getattr(owner, name).values())))
    with pytest.raises(RuntimeError, match="BINDING|BUFFER|NON_M5"):
        model._require_binding()


@pytest.mark.parametrize("equal_observation", [False, True])
def test_new_metadata_keeps_original_masked_weighted_loss_and_raw_gradients(equal_observation):
    allowed = np.array([[1, 2, 4, 5, 7], [4, 1, 2, 7, 5], [5, 4, 1, 2, 7]], np.uint8)
    eligible, mature = np.ones((3, 5), bool), np.ones((3, 5), bool)
    eligible[1, 1], mature[0, 2] = False, False
    truth_s = {"allowed": allowed, "eligible": eligible, "mature": mature}
    truth_f = {"allowed": allowed[::-1].copy(), "eligible": mature, "mature": eligible}
    raws = (torch.linspace(-1, 2, 30).reshape(3, 5, 2),
            torch.linspace(0.6, -1.7, 30).reshape(3, 5, 2))
    reports, gradients = [], []
    for enabled in (False, True):
        s, f = (value.clone().requires_grad_(True) for value in raws)
        report = objective.combined_loss(
            state_raw=s, future_raw=f, state_truth=truth_s, future_truth=truth_f,
            state_query_k=_inputs()[1][:, 1], future_weight=0.375,
            ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=enabled,
            m5_equal_observation=equal_observation)
        reports.append(report)
        gradients.append(torch.autograd.grad(report["loss"], (s, f)))
        assert s.grad is None and f.grad is None
    for key in reports[0].keys() - {"contract_sha256"}:
        _same(reports[1][key], reports[0][key])
    _same(gradients[1], gradients[0])
    old, new = (objective.contract(
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=enabled,
        m5_equal_observation=equal_observation) for enabled in (False, True))
    assert old["contract_sha256"] != new["contract_sha256"]
    assert reports[1]["contract_sha256"] == new["contract_sha256"]
    assert new["dual_model_contract_sha256"] == dual_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True)["contract_sha256"]
    for key in old.keys() - {"contract_sha256", "dual_model_contract_sha256"}:
        _same(new[key], old[key])


def test_weights_only_migration_preserves_source_modes_flags_rng_and_existing_predictor(monkeypatch):
    source, target = _state_pair(monkeypatch)
    source.train()
    source.state_model.query_head.eval()
    source.future_m5_side_head.eval()
    with torch.no_grad():
        source.state_model.query_head.weight.add_(0.5)
        source.future_head.bias.sub_(0.25)
        source.state_model.m5_side_head.bias.add_(1.25)
    for index, parameter in enumerate(source.parameters()):
        parameter.requires_grad_(index % 3 != 0)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    before, target_ids = _snapshot(source), {name: id(p) for name, p in target.named_parameters()}
    rng, source_sha = torch.get_rng_state().clone(), digest(source.state_dict())
    receipt = state.copy_dedicated_m5_weights_to_non_m5_sides(
        source, target, expected_legacy_model_sha256=source_sha)
    _same(_snapshot(source), before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert {name: id(p) for name, p in target.named_parameters()} == target_ids
    assert receipt["added_parameters"] == 8 * (source.state_model._hidden_width + 1)
    assert receipt["copied_side_rows"] == CLONES
    assert receipt["model_state_sha256"] == digest(target.state_dict())
    source_params, target_params = dict(source.named_parameters()), dict(target.named_parameters())
    for name, value in source.state_dict().items():
        if name in IDENTITY_BUFFERS:
            assert not torch.equal(value, target.state_dict()[name])
        else:
            assert torch.equal(value, target.state_dict()[name]), name
    for name, parameter in source_params.items():
        assert target_params[name].requires_grad == parameter.requires_grad, name
    for name, original in CLONES.items():
        assert torch.equal(target_params[name], source_params[original][1:2]), name
        assert target_params[name].requires_grad == source_params[original].requires_grad
        assert target_params[name].untyped_storage().data_ptr() != (
            source_params[original].untyped_storage().data_ptr())
    target_modes = dict(target.named_modules())
    for name, module in source.named_modules():
        assert target_modes[name].training == module.training, name
    for task in ("state", "future"):
        shared, _, sides = _heads(target, task)
        assert all(head.training == shared.training for head in sides.values())
    assert all(parameter.grad is None for parameter in target.parameters())
    _, active, _ = state._parameters(target)
    assert set(CLONES) <= {name for name, _ in active}
    for a, b in zip(_decode(source), _decode(target)):
        assert torch.equal(a[..., 0], b[..., 0])
        assert torch.equal(a[..., 0, 1], b[..., 0, 1])
        torch.testing.assert_close(a[..., 1:, 1], b[..., 1:, 1], rtol=0, atol=1e-6)


@pytest.mark.parametrize("fault", ["digest", "normalization", "buffer", "source_mode",
                                   "target_mode", "inventory", "source_grad", "target_grad"])
def test_bad_weights_only_migration_rejects_before_mutation(monkeypatch, fault):
    source, target = _state_pair(monkeypatch)
    if fault == "normalization":
        target.state_model.feature_view_identity["prefix_normalization"] = {"changed": True}
    elif fault == "buffer":
        target.backbone.input_norm_signal_center.add_(1)
    elif fault == "source_mode":
        source = state_model(monkeypatch, False)
    elif fault == "target_mode":
        target = copy.deepcopy(source)
    elif fault == "inventory":
        target.register_parameter("unexpected", nn.Parameter(torch.ones(1)))
    elif fault in ("source_grad", "target_grad"):
        parameter = next((source if fault == "source_grad" else target).parameters())
        parameter.grad = torch.zeros_like(parameter)
    expected = "0" * 64 if fault == "digest" else digest(source.state_dict())
    before_source, before_target = _snapshot(source), _snapshot(target)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="WARMSTART|PARTIAL|GRAD"):
        state.copy_dedicated_m5_weights_to_non_m5_sides(
            source, target, expected_legacy_model_sha256=expected)
    _same(_snapshot(source), before_source)
    _same(_snapshot(target), before_target)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("equal_observation", [False, True])
def test_variant_cannot_use_existing_numerical_or_optimizer_route(monkeypatch, equal_observation):
    _, model = _state_pair(monkeypatch)
    state._require_model_binding(model)
    recipe = fit.ta_sf_numerical_recipe(
        dedicated_m5_side=True, m5_equal_observation=equal_observation)
    before = _snapshot(model)
    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("Existing recipe must fail before constructing AdamW")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    for call in (lambda: state._numerical(recipe, model),
                 lambda: state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)):
        with pytest.raises(RuntimeError, match="NUMERICAL|NON_M5"):
            call()
    _same(_snapshot(model), before)
