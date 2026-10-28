"""Synthetic bps-supervision checks. No market payload or genuine native forward."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import entry_native_direction_bps_v1 as bps
from gx1.contracts import entry_native_direction_research_v1 as direction
from gx1.contracts import entry_native_direction_research_state_v1 as state
from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as runner
from tests import test_entry_native_direction_research_v1 as component
from tests import test_entry_native_direction_research_state_v1 as state_tests
from tests.test_entry_native_direction_feature_view_v1 import bound_case, _constructor_spy
from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner

M = bps.MINUTE_NS


def _arrays():
    D = np.asarray([100, 105, 110], np.int64) * M
    values = np.asarray([[.1, 2., 50.], [-.2, -3., -40.], [0., 4., 20.]], np.float64)
    return dict(parent_rows=np.arange(3, dtype=np.int64), decision_ns=D,
        terminal_time_ns=D[:, None] + np.asarray([1, 5, 15], np.int64)[None, :] * M,
        observed=np.ones((3, 3), np.bool_), sign=np.sign(values).astype(np.int8), mid_return_bps=values)


def _scale(arrays=None, cutoff=130*M):
    return bps.fit_direction_bps_scale(**(_arrays() if arrays is None else arrays),
        cutoff_ns=cutoff, target_source={"path": "/synthetic/targets.npz", "sha256": "b"*64}, fit_partition="train")


def _loss(values, cutoff=200*M):
    values = np.asarray(values, np.float64)
    n = len(values); D = torch.arange(100, 100+n, dtype=torch.int64)*M
    return dict(logit_up=torch.zeros(n, 3, requires_grad=True),
        mean_return_bps=torch.zeros(n, 3, requires_grad=True), rms_bps=torch.tensor([1., 2., 4.]),
        mid_return_bps=torch.from_numpy(values), sign=torch.from_numpy(np.sign(values).astype(np.int8)),
        observed=torch.ones(n, 3, dtype=torch.bool), decision_ns=D,
        terminal_time_ns=D[:, None]+torch.tensor([1, 5, 15])[None, :]*M, cutoff_ns=cutoff)


@pytest.mark.parametrize("polarity", [1, -1])
def test_equal_direction_gradient_but_distinct_magnitude_gradient(polarity):
    values = polarity*np.asarray([[.1, .1, .1], [50., 50., 50.]], np.float64)
    args = _loss(values)
    result = bps.native_direction_bps_loss(**args)
    result["loss"].backward()
    assert torch.equal(args["logit_up"].grad[0], args["logit_up"].grad[1])
    expected = -2*args["mid_return_bps"].float()/(6*args["rms_bps"]**2)
    torch.testing.assert_close(args["mean_return_bps"].grad, expected)
    torch.testing.assert_close(args["mean_return_bps"].grad[1]/args["mean_return_bps"].grad[0],
        torch.full((3,), 500.))


def test_observed_zero_learns_magnitude_without_a_flat_or_direction_label():
    args = _loss([[0., 0., 0.]])
    args["mean_return_bps"] = torch.ones(1, 3, requires_grad=True)
    result = bps.native_direction_bps_loss(**args)
    assert result["direction_loss"] is None and result["bps_eligible_cells"] == 3
    result["loss"].backward()
    assert args["logit_up"].grad is None
    assert torch.all(args["mean_return_bps"].grad > 0)


def test_missing_and_exact_cutoff_have_no_magnitude_gradient():
    args = _loss([[1., 0., 50.]], cutoff=115*M)
    args["observed"][0, 1] = False
    result = bps.native_direction_bps_loss(**args)
    assert result["bps_per_horizon_counts"] == [1, 0, 0]
    result["loss"].backward()
    assert args["mean_return_bps"].grad[0, 0] != 0
    assert torch.equal(args["mean_return_bps"].grad[0, 1:], torch.zeros(2))


def test_empty_supervision_remains_no_update():
    args = _loss([[0., 0., 0.]])
    args["observed"].zero_()
    result = bps.native_direction_bps_loss(**args)
    assert result["loss"] is None and not result["update_allowed"]
    assert result["bps_eligible_cells"] == 0


def test_scale_uses_mature_prefix_only_and_retains_zero_in_denominator():
    a = _arrays(); scale = _scale(a, cutoff=120*M)
    assert scale["counts"] == [3, 3, 1]
    expected = np.sqrt(np.sum(a["mid_return_bps"][:, 0].astype(np.float32).astype(np.float64)**2)/3)
    assert scale["rms_bps"][0] == expected
    changed = deepcopy(a); changed["mid_return_bps"][1:, 2] = [-9000., 8000.]
    assert _scale(changed, cutoff=120*M) == scale


@pytest.mark.parametrize("fault", ["zero", "empty", "later_row", "duplicate", "clock", "sign", "missing", "nan", "dtype", "control"])
def test_invalid_scale_population_target_or_scope_rejects(fault):
    a = _arrays()
    kwargs = dict(cutoff_ns=130*M, target_source={"path": "/synthetic/targets.npz", "sha256": "b"*64}, fit_partition="train")
    if fault == "zero": a["mid_return_bps"][:, 0] = 0; a["sign"][:, 0] = 0
    elif fault == "empty": a["observed"][:, 0] = False; a["mid_return_bps"][:, 0] = 0; a["sign"][:, 0] = 0
    elif fault == "later_row": kwargs["cutoff_ns"] = 110*M
    elif fault == "duplicate": a["parent_rows"][1] = 0
    elif fault == "clock": a["terminal_time_ns"][0, 0] += 1
    elif fault == "sign": a["sign"][0, 0] = -1
    elif fault == "missing": a["observed"][0, 0] = False
    elif fault == "nan": a["mid_return_bps"][0, 0] = np.nan
    elif fault == "dtype": a["mid_return_bps"] = a["mid_return_bps"].astype(np.float32)
    else: kwargs["fit_partition"] = "control"
    with pytest.raises(RuntimeError): bps.fit_direction_bps_scale(**a, **kwargs)


@pytest.mark.parametrize("fault", ["sign", "missing", "rms", "rms_grad", "target_grad", "target_dtype", "nonfinite"])
def test_loss_invalid_supervision_is_rejected(fault):
    args = _loss([[.1, 2., 50.]])
    if fault == "sign": args["sign"][0, 0] = -1
    elif fault == "missing": args["observed"][0, 0] = False
    elif fault == "rms": args["rms_bps"][0] = 0
    elif fault == "rms_grad": args["rms_bps"].requires_grad_(True)
    elif fault == "target_grad": args["mid_return_bps"].requires_grad_(True)
    elif fault == "target_dtype": args["mid_return_bps"] = args["mid_return_bps"].float()
    else: args["mean_return_bps"] = torch.full((1, 3), float("nan"))
    with pytest.raises(RuntimeError): bps.native_direction_bps_loss(**args)


def test_zero_bps_head_preserves_shared_initial_parameters_rng_and_one_forward(monkeypatch):
    monkeypatch.setattr(component.EntryV10CtxHybridTransformer, "forward", component._native_forward_spy)
    torch.manual_seed(420)
    baseline = direction.NativeEntryDirectionResearch(component._native_shell())
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(420)
    candidate = direction.NativeEntryDirectionResearch(component._native_shell(), bps_scale=_scale())
    assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
    out = candidate(**component._inputs())
    assert len(candidate.backbone._spy_calls) == 1
    assert torch.equal(out["logit_up"], baseline(**component._inputs())["logit_up"])
    assert torch.equal(out["mean_return_bps"], torch.zeros(2, 3))
    assert candidate.research_objective != baseline.research_objective


def test_bps_gradient_reaches_same_hidden_and_unused_legacy_heads_stay_unsupervised(monkeypatch):
    monkeypatch.setattr(component.EntryV10CtxHybridTransformer, "forward", component._native_forward_spy)
    model = direction.NativeEntryDirectionResearch(component._native_shell(), bps_scale=_scale())
    with torch.no_grad(): model.bps_mean_head.weight.fill_(.2)
    out = model(**component._inputs())
    args = _loss([[.1, 2., 50.], [-.2, -3., -40.]])
    args.update(logit_up=out["logit_up"], mean_return_bps=out["mean_return_bps"], rms_bps=model.bps_rms)
    bps.native_direction_bps_loss(**args)["bps_loss"].backward()
    assert torch.count_nonzero(model.backbone._synthetic_hidden.grad) > 0
    assert model.direction_head.weight.grad is None
    assert all(p.grad is None for p in model.backbone.head_dynamic_mfe.parameters())
    assert all(p.grad is None for p in model.backbone.head_forecast.parameters())


def _candidate(monkeypatch):
    base = state_tests._make(monkeypatch)
    model = direction.NativeEntryDirectionResearch(base.model.backbone, bps_scale=_scale())
    identity = state_tests._identity()
    identity["research_objective_sha256"] = model.research_objective["contract_sha256"]
    model.feature_view_identity = identity
    recipe = recipes.native_direction_research_recipe(batch_size=2, max_updates=4, signed_bps=True)
    optimizer, binding = state.build_native_direction_research_optimizer(model, recipe=recipe)
    model.train(); state_tests._seed(recipe["seed"])
    return SimpleNamespace(model=model, optimizer=optimizer, recipe=recipe, binding=binding, order=base.order)


def _step(case, step):
    a = _loss([[.1, 2., 50.], [-.2, -3., -40.]])
    target = {k: v for k, v in a.items() if k not in ("logit_up", "mean_return_bps", "rms_bps", "cutoff_ns")}
    counters = {k: 0 for k in ("forward_calls", "forward_returns", "empty_batches", "backward_calls",
        "backward_returns", "optimizer_step_calls", "optimizer_step_returns")}
    counters["forward_kind_calls"] = {}
    record, updated = runner._train_batch(case.model, case.optimizer, state_tests._features(step), target,
        cutoff=a["cutoff_ns"], counters=counters, ceiling=1, torch=torch, diagnostics=False)
    assert updated and counters["forward_calls"] == counters["forward_returns"] == 1
    assert record["bps_eligible_cells"] == 6 and record["bps_loss"] > 0
    return record


def _capture(case, cursor, updates):
    return state.capture_native_direction_research_state(case.model, case.optimizer, recipe=case.recipe,
        feature_view_identity=case.model.feature_view_identity, parent_order=case.order,
        cursor=cursor, optimizer_updates=updates, device=torch.device("cpu"))


def test_actual_train_adapter_updates_bps_head_and_exact_resume_matches(monkeypatch):
    first = _candidate(monkeypatch)
    before = first.model.bps_mean_head.weight.detach().clone()
    _step(first, 0)
    assert not torch.equal(first.model.bps_mean_head.weight, before)
    saved = _capture(first, 2, 1)
    _step(first, 1)
    expected = _capture(first, 4, 2)
    resumed = _candidate(monkeypatch)
    state.restore_native_direction_research_state(saved, resumed.model, resumed.optimizer, recipe=resumed.recipe,
        feature_view_identity=resumed.model.feature_view_identity, parent_order=resumed.order, device=torch.device("cpu"))
    _step(resumed, 1)
    assert _capture(resumed, 4, 2)["contract_sha256"] == expected["contract_sha256"]
    assert all(p.grad is None for p in resumed.model.parameters())
    assert all(name not in saved["binding"]["active_parameter_names"]
        for name in ["backbone.head_dynamic_mfe.weight", "backbone.head_forecast.weight"])


def test_bps_state_or_recipe_cannot_be_used_as_old_direction_state(monkeypatch):
    candidate = _candidate(monkeypatch)
    with pytest.raises(RuntimeError, match="RECIPE_OBJECTIVE"):
        state.build_native_direction_research_optimizer(candidate.model,
            recipe=recipes.native_direction_research_recipe(batch_size=2, max_updates=4))
    saved = _capture(candidate, 0, 0)
    old = state_tests._make(monkeypatch)
    with pytest.raises(RuntimeError, match="BINDING_IDENTITY"):
        state_tests._restore(old, saved)


def test_changed_scale_is_rejected_before_restore_mutation(monkeypatch):
    case = _candidate(monkeypatch); saved = _capture(case, 0, 0)
    saved["buffer_state"]["bps_rms"][0] += 1
    saved["model_state"]["bps_rms"][0] += 1
    state_tests._reseal(saved)
    before = state_tests._live_hash(case)
    with pytest.raises(RuntimeError, match="IMMUTABLE_BUFFER"):
        state.restore_native_direction_research_state(saved, case.model, case.optimizer, recipe=case.recipe,
            feature_view_identity=case.model.feature_view_identity, parent_order=case.order, device=torch.device("cpu"))
    assert state_tests._live_hash(case) == before


def test_target_selection_and_device_transfer_keep_exact_signed_bps():
    a = _arrays()
    class Archive(dict):
        files = list(runner.PRICE_MEMBERS)
    archive = Archive(a)
    old = runner._select_price_targets(archive)
    new = runner._select_price_targets(archive, signed_bps=True)
    assert set(old) == set(runner.TARGET_MEMBERS)
    assert new["mid_return_bps"] is a["mid_return_bps"]
    _, target = runner._device_batch({"features": {}}, new, torch, torch.device("cpu"))
    assert target["mid_return_bps"].dtype == torch.float64
    assert torch.equal(target["mid_return_bps"], torch.from_numpy(a["mid_return_bps"]))


def test_view_binds_same_inputs_but_new_objective_and_scale(bound_case, monkeypatch):
    base = bound_case["view"]; n = base["prefix"]["rows"]; cutoff = base["prefix"]["cutoff_ns"]
    D = cutoff - np.arange(n+20, 20, -1, dtype=np.int64)*5*M
    values = np.tile(np.asarray([1., -2., 3.], np.float64), (n, 1))
    scale = _scale(dict(parent_rows=np.arange(n, dtype=np.int64), decision_ns=D,
        terminal_time_ns=D[:, None]+np.asarray([1, 5, 15], np.int64)[None, :]*M,
        mid_return_bps=values, observed=np.ones((n, 3), np.bool_), sign=np.sign(values).astype(np.int8)), cutoff)
    view = view_owner.bind_native_direction_feature_view(**bound_case["arguments"], bps_scale=scale)
    assert view["canonical_routing_sha256"] == base["canonical_routing_sha256"]
    assert view["prefix_normalization"] == base["prefix_normalization"]
    assert view["research_objective"] == direction.native_direction_research_contract(signed_bps=True)
    calls, _ = _constructor_spy(monkeypatch)
    model = view_owner.build_native_direction_research_model(view,
        prefix_normalization=bound_case["arguments"]["prefix_normalization"],
        constructor_metadata=bound_case["arguments"]["constructor_metadata"], device=torch.device("cpu"))
    assert len(calls) == 1 and model.signed_bps
    assert model.feature_view_identity["research_objective_sha256"] == model.research_objective["contract_sha256"]


def test_bps_report_keeps_original_units_masks_and_frozen_reference():
    a = _arrays(); scale = _scale(a, 120*M)
    result = bps.score_direction_bps(np.zeros((3, 3), np.float32), mid_return_bps=a["mid_return_bps"],
        observed=a["observed"], terminal_time_ns=a["terminal_time_ns"], cutoff_ns=120*M, scale=scale)
    row = result["per_horizon"][2]
    assert row["count"] == 1 and row["mse_bps2"] == 2500
    assert row["sse_bps2"] == row["zero_reference_sse_bps2"]
    assert row["own_prefix_mean_reference_sse_bps2"] == 0


def _magnitude_weight_contract(arrays=None, cutoff=130*M):
    return bps.fit_direction_magnitude_weight_contract(
        **(_arrays() if arrays is None else arrays), cutoff_ns=cutoff,
        target_source={"path": "/synthetic/targets.npz", "sha256": "b"*64}, fit_partition="train")


def _magnitude_weight_loss(values, cutoff=200*M):
    args = _loss(values, cutoff=cutoff)
    del args["mean_return_bps"]
    args["raw_weight_mean"] = torch.tensor([.4, .5, .6])
    return args


def test_magnitude_weight_uses_only_own_mature_nonzero_prefix_and_normalizes():
    a = _arrays(); contract = _magnitude_weight_contract(a, cutoff=120*M)
    assert contract["eligible_counts"] == [2, 3, 1]
    scales = np.asarray(contract["base_scale"]["rms_bps"], np.float32)
    magnitude = np.abs(a["mid_return_bps"].astype(np.float32))
    raw = magnitude / (scales[None, :] + magnitude)
    eligible = a["observed"] & (a["sign"] != 0) & (a["terminal_time_ns"] < 120*M)
    for i in range(3):
        assert np.mean(raw[eligible[:, i], i] / contract["raw_weight_means"][i]) == pytest.approx(1.)
    changed = deepcopy(a)
    changed["mid_return_bps"][1:, 2] = [-9000., 8000.]
    assert _magnitude_weight_contract(changed, cutoff=120*M) == contract


@pytest.mark.parametrize("polarity", [1, -1])
def test_magnitude_weight_gives_larger_move_stronger_direction_gradient(polarity):
    args = _magnitude_weight_loss(polarity*np.asarray([[.1, .1, .1], [50., 50., 50.]]))
    result = bps.native_direction_magnitude_weighted_loss(**args)
    result["loss"].backward()
    assert torch.all(args["logit_up"].grad[1].abs() > args["logit_up"].grad[0].abs())
    assert torch.all(args["logit_up"].grad * polarity < 0)
    assert result["unweighted_direction_loss"] is not None


def test_magnitude_weight_same_size_matches_original_direction_loss_and_gradients():
    args = _magnitude_weight_loss([[2., -2., 2.], [-2., 2., -2.]])
    args["rms_bps"] = torch.full((3,), 2.)
    args["raw_weight_mean"] = torch.full((3,), .5)
    result = bps.native_direction_magnitude_weighted_loss(**args)
    torch.testing.assert_close(result["loss"], result["unweighted_direction_loss"])
    grad = torch.autograd.grad(result["loss"], args["logit_up"], retain_graph=True)[0]
    base = torch.autograd.grad(result["unweighted_direction_loss"], args["logit_up"])[0]
    torch.testing.assert_close(grad, base)


def test_magnitude_weight_zero_missing_and_cutoff_do_not_produce_direction_gradient():
    args = _magnitude_weight_loss([[1., 0., 50.]], cutoff=115*M)
    args["observed"][0, 1] = False
    result = bps.native_direction_magnitude_weighted_loss(**args)
    assert result["counts"]["eligible_cells"] == 1
    result["loss"].backward()
    assert args["logit_up"].grad[0, 0] != 0
    assert torch.equal(args["logit_up"].grad[0, 1:], torch.zeros(2))
    empty = _magnitude_weight_loss([[0., 0., 0.]])
    result = bps.native_direction_magnitude_weighted_loss(**empty)
    assert result["loss"] is None and not result["update_allowed"]


@pytest.mark.parametrize("fault", ["mean_zero", "mean_one", "mean_grad", "target_sign", "target_grad", "target_dtype"])
def test_magnitude_weight_invalid_inputs_fail_closed(fault):
    args = _magnitude_weight_loss([[.1, 2., 50.]])
    if fault == "mean_zero": args["raw_weight_mean"][0] = 0
    elif fault == "mean_one": args["raw_weight_mean"][0] = 1
    elif fault == "mean_grad": args["raw_weight_mean"].requires_grad_(True)
    elif fault == "target_sign": args["sign"][0, 0] = -1
    elif fault == "target_grad": args["mid_return_bps"].requires_grad_(True)
    else: args["mid_return_bps"] = args["mid_return_bps"].float()
    with pytest.raises(RuntimeError):
        bps.native_direction_magnitude_weighted_loss(**args)


def test_magnitude_weight_contract_rejects_tamper_and_nontrain_fit():
    contract = _magnitude_weight_contract()
    contract["raw_weight_means"][0] *= 2
    with pytest.raises(RuntimeError, match="MAG_WEIGHT_SEAL"):
        bps.require_direction_magnitude_weight_contract(contract)
    with pytest.raises(RuntimeError, match="TRAIN_ONLY"):
        bps.fit_direction_magnitude_weight_contract(**_arrays(), cutoff_ns=130*M,
            target_source={"path": "/synthetic/targets.npz", "sha256": "b"*64}, fit_partition="control")


def test_magnitude_weight_model_preserves_initial_logits_parameters_and_rng(monkeypatch):
    monkeypatch.setattr(component.EntryV10CtxHybridTransformer, "forward", component._native_forward_spy)
    torch.manual_seed(420)
    baseline = direction.NativeEntryDirectionResearch(component._native_shell())
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(420)
    weighted = direction.NativeEntryDirectionResearch(component._native_shell(),
        magnitude_weight_contract=_magnitude_weight_contract())
    assert torch.equal(torch.random.get_rng_state(), baseline_rng)
    for name, parameter in baseline.named_parameters():
        assert torch.equal(parameter, dict(weighted.named_parameters())[name])
    assert torch.equal(weighted(**component._inputs())["logit_up"], baseline(**component._inputs())["logit_up"])
    assert weighted.magnitude_weighted and not weighted.signed_bps
    assert weighted.research_objective != baseline.research_objective


def test_magnitude_weight_recipe_rejects_mixed_or_unweighted_identity():
    recipe = recipes.native_direction_research_recipe(batch_size=2, max_updates=4, magnitude_weighted=True)
    assert recipes.require_native_direction_research_recipe(recipe) == recipe
    assert recipe["research_objective_sha256"] == direction.native_direction_research_contract(magnitude_weighted=True)["contract_sha256"]
    with pytest.raises(RuntimeError, match="OBJECTIVE_MODE"):
        recipes.native_direction_research_recipe(batch_size=2, max_updates=4, signed_bps=True, magnitude_weighted=True)
    bad = deepcopy(recipe); bad["research_objective_sha256"] = direction.native_direction_research_contract()["contract_sha256"]
    with pytest.raises(RuntimeError, match="RECIPE_MISMATCH"):
        recipes.require_native_direction_research_recipe(bad)


def _magnitude_weight_candidate(monkeypatch):
    base = state_tests._make(monkeypatch)
    model = direction.NativeEntryDirectionResearch(base.model.backbone,
        magnitude_weight_contract=_magnitude_weight_contract())
    identity = state_tests._identity()
    identity["research_objective_sha256"] = model.research_objective["contract_sha256"]
    model.feature_view_identity = identity
    recipe = recipes.native_direction_research_recipe(batch_size=2, max_updates=4, magnitude_weighted=True)
    optimizer, binding = state.build_native_direction_research_optimizer(model, recipe=recipe)
    model.train(); state_tests._seed(recipe["seed"])
    return SimpleNamespace(model=model, optimizer=optimizer, recipe=recipe, binding=binding, order=base.order)


def _magnitude_weight_step(case, step):
    args = _magnitude_weight_loss([[.1, 2., 50.], [-.2, -3., -40.]])
    args["logit_up"] = case.model(**state_tests._features(step))["logit_up"]
    args["rms_bps"] = case.model.magnitude_rms_bps
    args["raw_weight_mean"] = case.model.magnitude_raw_weight_mean
    bps.native_direction_magnitude_weighted_loss(**args)["loss"].backward()
    case.optimizer.step()
    case.optimizer.zero_grad(set_to_none=True)


def test_magnitude_weight_model_synthetic_resume_matches(monkeypatch):
    first = _magnitude_weight_candidate(monkeypatch)
    _magnitude_weight_step(first, 0)
    saved = _capture(first, 2, 1)
    _magnitude_weight_step(first, 1)
    expected = _capture(first, 4, 2)
    resumed = _magnitude_weight_candidate(monkeypatch)
    state.restore_native_direction_research_state(saved, resumed.model, resumed.optimizer, recipe=resumed.recipe,
        feature_view_identity=resumed.model.feature_view_identity, parent_order=resumed.order, device=torch.device("cpu"))
    _magnitude_weight_step(resumed, 1)
    assert _capture(resumed, 4, 2)["contract_sha256"] == expected["contract_sha256"]


def test_magnitude_weight_changed_state_buffer_rejected_before_mutation(monkeypatch):
    case = _magnitude_weight_candidate(monkeypatch); saved = _capture(case, 0, 0)
    saved["buffer_state"]["magnitude_raw_weight_mean"][0] += .01
    saved["model_state"]["magnitude_raw_weight_mean"][0] += .01
    state_tests._reseal(saved)
    before = state_tests._live_hash(case)
    with pytest.raises(RuntimeError, match="IMMUTABLE_BUFFER"):
        state.restore_native_direction_research_state(saved, case.model, case.optimizer, recipe=case.recipe,
            feature_view_identity=case.model.feature_view_identity, parent_order=case.order, device=torch.device("cpu"))
    assert state_tests._live_hash(case) == before


def test_magnitude_weight_view_keeps_feature_routes_and_binds_scale(bound_case, monkeypatch):
    base = bound_case["view"]; n = base["prefix"]["rows"]; cutoff = base["prefix"]["cutoff_ns"]
    D = cutoff - np.arange(n+20, 20, -1, dtype=np.int64)*5*M
    values = np.tile(np.asarray([1., -2., 3.], np.float64), (n, 1))
    weight = _magnitude_weight_contract(dict(parent_rows=np.arange(n, dtype=np.int64), decision_ns=D,
        terminal_time_ns=D[:, None]+np.asarray([1, 5, 15], np.int64)[None, :]*M,
        mid_return_bps=values, observed=np.ones((n, 3), np.bool_), sign=np.sign(values).astype(np.int8)), cutoff)
    view = view_owner.bind_native_direction_feature_view(**bound_case["arguments"], magnitude_weight_contract=weight)
    assert view["canonical_routing_sha256"] == base["canonical_routing_sha256"]
    assert view["prefix_normalization"] == base["prefix_normalization"]
    assert view["research_objective"] == direction.native_direction_research_contract(magnitude_weighted=True)
    calls, _ = _constructor_spy(monkeypatch)
    model = view_owner.build_native_direction_research_model(view,
        prefix_normalization=bound_case["arguments"]["prefix_normalization"],
        constructor_metadata=bound_case["arguments"]["constructor_metadata"], device=torch.device("cpu"))
    assert len(calls) == 1 and model.magnitude_weighted
    assert model.feature_view_identity["research_objective_sha256"] == model.research_objective["contract_sha256"]
