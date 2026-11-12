"""Strict activity-only migration and fresh zero-state mechanics; no native fit."""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as digest
from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from tests.test_market_state_ta_dedicated_m5_binding_v1 import (
    COMMIT, DEVICE, ORDER, _capture, _model as source_model, _seed,
)
from tests.test_market_state_time_query_non_m5_side_v1 import _same, _snapshot, _decode
from tests.test_market_state_time_query_tf_activity_v1 import CLONES, IDENTITY_BUFFERS, TIMEFRAMES


def _pair(monkeypatch):
    source = source_model(monkeypatch, True)
    target = NativeFutureDualModel(FactorizedTimeQueryMarketStateModel(
        copy.deepcopy(source.backbone), dedicated_m5_side=True, dedicated_tf_activity=True))
    target.state_model.feature_view_identity = copy.deepcopy(source.state_model.feature_view_identity)
    target.state_model.feature_view_identity.update(
        contract_sha256="5" * 64,
        research_objective_sha256=target.state_model.research_objective["contract_sha256"])
    return source, target


def _rng():
    return random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone()


def _migrate(source, target, expected=None):
    return state.copy_dedicated_m5_weights_to_tf_activity(
        source, target, expected_legacy_model_sha256=digest(source.state_dict()) if expected is None else expected)


def _recipe(*, activity=True, equal=False):
    return fit.ta_sf_numerical_recipe(dedicated_m5_side=True,
        dedicated_tf_activity=activity, m5_equal_observation=equal)


def test_migration_copies_twenty_rows_preserving_source_rng_modes_flags_and_side(monkeypatch):
    source, target = _pair(monkeypatch)
    source.train()
    source.state_model.query_head.eval()
    source.future_head.train()
    source.future_m5_side_head.eval()
    source.backbone._drop.eval()
    with torch.no_grad():
        source.state_model.query_head.weight[0].add_(0.375)
        source.future_head.bias[0].sub_(0.25)
        source.state_model.m5_side_head.bias.add_(0.75)
    for i, parameter in enumerate(source.parameters()):
        parameter.requires_grad_(i % 3 != 0)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    before_source, before_rng = _snapshot(source), _rng()
    identities = {name: target.state_dict()[name].clone() for name in IDENTITY_BUFFERS}
    parameter_ids = {name:id(parameter) for name,parameter in target.named_parameters()}
    def forbidden(*args, **kwargs):
        pytest.fail("weights-only migration must not construct an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    receipt = _migrate(source, target)
    _same(_snapshot(source), before_source)
    _same(_rng(), before_rng)
    assert {name:id(parameter) for name,parameter in target.named_parameters()} == parameter_ids
    assert receipt["schema_version"] == "gx1_ta_tf_activity_weights_warmstart_v1"
    assert receipt["copied_activity_rows"] == CLONES and len(CLONES) == 20
    assert receipt["added_parameters"] == 10 * (source.state_model._hidden_width + 1)
    assert receipt["source_model_state_sha256"] == digest(source.state_dict())
    assert receipt["model_state_sha256"] == digest(target.state_dict())
    assert receipt["cursor"] == receipt["optimizer_updates"] == 0
    assert receipt["optimizer_policy"] == "fresh_AdamW_empty_history_no_inherited_updates"
    assert receipt["fresh_actual_initial_measurement_required"] is True
    assert receipt["prediction_parity_measured"] is False
    old, new = dict(source.named_parameters()), dict(target.named_parameters())
    for name,value in source.state_dict().items():
        if name in identities:
            assert torch.equal(target.state_dict()[name],identities[name])
            assert not torch.equal(target.state_dict()[name],value)
        else:
            assert torch.equal(target.state_dict()[name],value),name
    for name,parameter in new.items():
        origin=CLONES.get(name,name)
        assert parameter.requires_grad == old[origin].requires_grad,name
        if name in CLONES:
            assert torch.equal(parameter,old[origin][0:1]),name
    source_storage={value.untyped_storage().data_ptr() for value in source.state_dict().values()}
    target_storage={value.untyped_storage().data_ptr() for value in target.state_dict().values()}
    assert source_storage.isdisjoint(target_storage)
    new_modes=dict(target.named_modules())
    for name,module in source.named_modules():
        assert new_modes[name].training == module.training,name
    for prefix,shared in (("state_model.tf_activity_heads","state_model.query_head"),
                          ("future_tf_activity_heads","future_head")):
        for suffix in ("",*("."+tf for tf in TIMEFRAMES)):
            assert new_modes[prefix+suffix].training == new_modes[shared].training
    assert all(parameter.grad is None for parameter in target.parameters())
    for old_raw,new_raw in zip(_decode(source),_decode(target)):
        assert torch.equal(old_raw[...,1],new_raw[...,1])
        torch.testing.assert_close(old_raw[...,0],new_raw[...,0],rtol=1e-6,atol=2e-6)


@pytest.mark.parametrize("fault", [
    "digest", "normalization", "buffer", "source_profile", "target_profile", "source_activity",
    "source_non_m5", "target_non_m5", "inventory", "source_grad", "target_grad",
    "parameter_alias", "buffer_alias", "internal_source_alias", "internal_target_alias",
    "source_nonfinite", "target_nonfinite", "module_inventory", "module_mode", "identity",
])
def test_invalid_migration_fails_before_either_model_or_rng_changes(monkeypatch,fault):
    source,target=_pair(monkeypatch)
    if fault=="normalization":
        target.state_model.feature_view_identity["prefix_normalization"]={"changed":True}
    elif fault=="buffer":
        target.backbone.input_norm_signal_center.add_(1)
    elif fault=="source_profile":
        source=source_model(monkeypatch,False)
    elif fault=="target_profile":
        target=copy.deepcopy(source)
    elif fault=="source_activity":
        source=copy.deepcopy(target)
    elif fault in ("source_non_m5","target_non_m5"):
        variant=NativeFutureDualModel(FactorizedTimeQueryMarketStateModel(
            copy.deepcopy(source.backbone),dedicated_m5_side=True,dedicated_non_m5_side=True))
        if fault=="source_non_m5":source=variant
        else:target=variant
    elif fault=="inventory":
        target.register_parameter("unexpected",nn.Parameter(torch.ones(1)))
    elif fault in ("source_grad","target_grad"):
        parameter=next((source if fault=="source_grad" else target).parameters())
        parameter.grad=torch.zeros_like(parameter)
    elif fault=="parameter_alias":
        target.future_head.weight=nn.Parameter(source.future_head.weight.detach())
    elif fault=="buffer_alias":
        target.backbone.input_norm_signal_center=source.backbone.input_norm_signal_center
    elif fault in ("internal_source_alias","internal_target_alias"):
        backbone=(source if fault=="internal_source_alias" else target).backbone
        backbone.dynamic_m5_query_projection.bias=nn.Parameter(backbone.head_forecast.bias.detach())
    elif fault in ("source_nonfinite","target_nonfinite"):
        with torch.no_grad():
            (source if fault=="source_nonfinite" else target).backbone.entry_q_joint_in.weight[0,0]=float("inf")
    elif fault=="module_inventory":
        target.extra_module=nn.Identity()
    elif fault=="module_mode":
        target.backbone._drop.training="train"
    elif fault=="identity":
        target.native_F_objective_sha256[0]^=1
    expected="0"*64 if fault=="digest" else digest(source.state_dict())
    before_source,before_target,before_rng=_snapshot(source),_snapshot(target),_rng()
    expected_error = ("NATIVE_DIRECTION_RESEARCH_BACKBONE_FINITE"
                      if fault in ("source_nonfinite", "target_nonfinite")
                      else "WARMSTART|PARTIAL|GRAD|BINDING")
    with pytest.raises(RuntimeError,match=expected_error):
        _migrate(source,target,expected)
    _same(_snapshot(source),before_source)
    _same(_snapshot(target),before_target)
    _same(_rng(),before_rng)


@pytest.mark.parametrize("value",[None,1,"A"*64,"not-a-digest"])
def test_digest_type_and_format_are_checked_before_copy(monkeypatch,value):
    source,target=_pair(monkeypatch)
    before=_snapshot(target)
    with pytest.raises(RuntimeError,match="TF_ACTIVITY_WARMSTART_SOURCE_DIGEST"):
        state.copy_dedicated_m5_weights_to_tf_activity(source,target,expected_legacy_model_sha256=value)
    _same(_snapshot(target),before)


@pytest.mark.parametrize("equal",[False,True])
def test_matching_recipe_fresh_optimizer_and_zero_state_roundtrip(monkeypatch,equal):
    _seed()
    source,target=_pair(monkeypatch)
    _,old_active,old_inactive=state._parameters(source)
    _migrate(source,target)
    recipe=_recipe(equal=equal)
    old_recipe=_recipe(activity=False,equal=equal)
    assert old_recipe.keys()==recipe.keys()
    assert {key for key in recipe if recipe[key]!=old_recipe[key]}=={
        "research_objective_sha256","native_F_loss_contract_sha256","contract_sha256"}
    before_source=_snapshot(source)
    optimizer,binding=state.build_fold1_fit_optimizer(target,numerical_recipe=recipe)
    _same(_snapshot(source),before_source)
    assert type(optimizer) is torch.optim.AdamW and not optimizer.state
    assert set(binding["active_parameter_names"])=={name for name,_ in old_active}|set(CLONES)
    assert set(binding["inactive_parameter_names"])=={name for name,_ in old_inactive}
    params=dict(target.named_parameters())
    assert all(params[name].requires_grad for name in CLONES)
    assert [id(parameter) for parameter in optimizer.param_groups[0]["params"]]==[
        id(params[name]) for name in binding["active_parameter_names"]]
    assert binding["numerical_recipe"]==recipe and binding["research_objective"]==target.research_objective
    target.eval()
    target.state_model.tf_activity_heads["H1"].train()
    target.future_tf_activity_heads["D1"].train()
    _seed()
    saved=_capture(target,optimizer,recipe,0)
    assert saved["optimizer_state"]["state"]=={}
    assert saved["cursor"]==saved["optimizer_updates"]==saved["skipped_batches"]==0
    _,restored=_pair(monkeypatch)
    restored_optimizer,restored_binding=state.build_fold1_fit_optimizer(restored,numerical_recipe=recipe)
    assert restored_binding==binding
    ids={name:id(parameter) for name,parameter in restored.named_parameters()}
    progress=state.restore_fold1_fit_state(saved,restored,restored_optimizer,
        numerical_recipe=recipe,feature_view_identity=restored.state_model.feature_view_identity,
        parent_order=ORDER,source_commit=COMMIT,device=DEVICE)
    assert progress=={"cursor":0,"optimizer_updates":0,"skipped_batches":0}
    _same(_capture(restored,restored_optimizer,recipe,0),saved)
    assert ids=={name:id(parameter) for name,parameter in restored.named_parameters()}
    assert not restored_optimizer.state and all(p.grad is None for p in restored.parameters())


@pytest.mark.parametrize("equal",[False,True])
@pytest.mark.parametrize("direction",["old_recipe_new_model","new_recipe_old_model"])
def test_wrong_recipe_fails_before_optimizer_or_parameter_changes(monkeypatch,equal,direction):
    source,target=_pair(monkeypatch)
    _migrate(source,target)
    model=target if direction=="old_recipe_new_model" else source
    recipe=_recipe(activity=direction=="new_recipe_old_model",equal=equal)
    before,before_rng=_snapshot(model),_rng()
    def forbidden(*args,**kwargs):
        pytest.fail("wrong recipe reached AdamW")
    monkeypatch.setattr(torch.optim,"AdamW",forbidden)
    with pytest.raises(RuntimeError,match="TA_SF_NUMERICAL_RECIPE"):
        state.build_fold1_fit_optimizer(model,numerical_recipe=recipe)
    _same(_snapshot(model),before)
    _same(_rng(),before_rng)
