"""Activity-only view, factory, numerical and batch binding on synthetic inputs."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_time_query_binding_v1 import sampled_query_k
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_time_query_model_contract,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from tests.test_entry_native_direction_feature_view_v1 import (
    bound_case, _constructor_spy, _seal,
)
from tests.test_market_state_time_query_view_mode import time_query_case
from tests.test_market_state_time_query_v2_wiring import _v2_arguments
from tests.test_market_state_time_query_batch_v1 import _features
from tests.test_market_state_time_query_tf_activity_v1 import _model
from tests.test_market_state_non_m5_side_native_binding import _targets
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit


def _arguments(time_query_case):
    return dict(_v2_arguments(time_query_case),
                full_ta_feature_profile=True, dedicated_m5_side=True)


def test_false_flag_preserves_view_bytes_and_new_flag_changes_only_decoder_identity(time_query_case):
    args = _arguments(time_query_case)
    old = views.bind_native_direction_feature_view(**args)
    explicit = views.bind_native_direction_feature_view(**args, dedicated_tf_activity=False)
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert encode(old) == encode(explicit)
    assert "dedicated_tf_activity" not in old
    new = views.bind_native_direction_feature_view(**args, dedicated_tf_activity=True)
    assert new["dedicated_tf_activity"] is True
    assert new["research_objective"] == factorized_time_query_model_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True)
    assert new["contract_sha256"] != old["contract_sha256"]
    differences = {key for key in new if new[key] != old.get(key)}
    assert differences == {"dedicated_tf_activity", "research_objective", "contract_sha256"}
    assert views.bind_native_direction_feature_view(
        **args, dedicated_tf_activity=True, expected=new) == new
    assert views.bind_native_direction_feature_view(**args, expected=old) == old


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_binder_rejects_non_boolean_flag_before_any_constructor(time_query_case, monkeypatch, value):
    calls, _ = _constructor_spy(monkeypatch)
    with pytest.raises(RuntimeError, match="DEDICATED_TF_ACTIVITY_SCOPE"):
        views.bind_native_direction_feature_view(
            **_arguments(time_query_case), dedicated_tf_activity=value)
    assert not calls


@pytest.mark.parametrize("field", ["full_ta_feature_profile", "dedicated_m5_side",
                                  "market_state_time_query_v2"])
def test_binder_requires_explicit_ta_m5_and_v2(time_query_case, field):
    args = _arguments(time_query_case)
    args[field] = False
    with pytest.raises(RuntimeError, match="DEDICATED_.*_SCOPE"):
        views.bind_native_direction_feature_view(**args, dedicated_tf_activity=True)


@pytest.mark.parametrize("value", [False, None, 0, 1, "true"])
def test_present_view_key_requires_literal_true_even_after_resealing(time_query_case, value):
    value_view = views.bind_native_direction_feature_view(
        **_arguments(time_query_case), dedicated_tf_activity=True)
    value_view["dedicated_tf_activity"] = value
    with pytest.raises(RuntimeError, match="VIEW_KEYS"):
        views._require_view(_seal(value_view))


@pytest.mark.parametrize("removed", ["full_ta_feature_profile", "dedicated_m5_side"])
def test_resealed_trial_cannot_drop_required_mode_marker(time_query_case, removed):
    value = views.bind_native_direction_feature_view(
        **_arguments(time_query_case), dedicated_tf_activity=True)
    del value[removed]
    with pytest.raises(RuntimeError, match="VIEW_KEYS"):
        views._require_view(_seal(value))


def test_factory_passes_trial_flag_to_real_wrapper_and_preserves_feature_constructor(time_query_case, monkeypatch):
    args = _arguments(time_query_case)
    old = views.bind_native_direction_feature_view(**args)
    trial = views.bind_native_direction_feature_view(**args, dedicated_tf_activity=True)
    calls, _ = _constructor_spy(monkeypatch)
    models = [views.build_native_direction_research_model(
        view, prefix_normalization=args["prefix_normalization"],
        constructor_metadata=args["constructor_metadata"], device=torch.device("cpu"))
        for view in (old, trial)]
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert all(type(model) is FactorizedTimeQueryMarketStateModel for model in models)
    assert models[0]._dedicated_tf_activity is False
    assert not hasattr(models[0], "tf_activity_heads")
    assert models[1]._dedicated_tf_activity is True
    assert tuple(models[1].tf_activity_heads) == ("M5", "M15", "H1", "H4", "D1")
    assert models[1].feature_view_identity == {
        "contract_sha256": trial["contract_sha256"],
        "prefix_normalization": trial["prefix_normalization"],
        "research_objective_sha256": trial["research_objective"]["contract_sha256"],
    }
    dual = NativeFutureDualModel(models[1])
    assert tuple(dual.future_tf_activity_heads) == ("M5", "M15", "H1", "H4", "D1")
    dual._require_binding()


@pytest.mark.parametrize("fault", ["false_key", "old_objective"])
def test_factory_rejects_inconsistent_resealed_view_before_backbone(time_query_case, monkeypatch, fault):
    args = _arguments(time_query_case)
    value = views.bind_native_direction_feature_view(**args, dedicated_tf_activity=True)
    if fault == "false_key":
        value["dedicated_tf_activity"] = False
    else:
        value["research_objective"] = factorized_time_query_model_contract(
            ta_only=True, dedicated_m5_side=True)
    value = _seal(value)
    calls, _ = _constructor_spy(monkeypatch)
    with pytest.raises(RuntimeError):
        views.build_native_direction_research_model(
            value, prefix_normalization=args["prefix_normalization"],
            constructor_metadata=args["constructor_metadata"], device=torch.device("cpu"))
    assert not calls



@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("equal_observation", [False, True])
def test_fit_batch_uses_actual_model_flag_in_original_loss_metadata(monkeypatch, enabled, equal_observation):
    model = _model(activity=enabled)
    state, future, rows, issues, seed = _targets()
    features = {key: value for key, value in _features(16).items() if not key.startswith("gc_")}
    hidden = torch.linspace(-1, 1, 16 * 128).reshape(16, 128)
    calls = []
    def encode(**values):
        calls.append(set(values))
        assert all(values[key] is features[key] for key in values)
        return hidden
    monkeypatch.setattr(model, "encode_features", encode)
    with torch.no_grad():
        report = batch.evaluate_fit_batch(
            model, features, state, future, rows, issues, fold_id=1, pass_index=0,
            seed=seed, future_weight=1.0, m5_equal_observation=equal_observation,
            fit_batch_start=0 if equal_observation else None)
    assert calls == [set(features)]
    expected_query = sampled_query_k(issues, pass_index=0, seed=seed)
    assert np.array_equal(report["state_query_k"], expected_query)
    assert np.array_equal(report["future_native_k"], np.asarray(NATIVE_K))
    state_truth = state.gather_fit_sampled(rows, issues, fold_id=1, pass_index=0, seed=seed)
    future_truth = future.gather(rows, issues, fold_id=1, partition="fit")
    expected = objective.combined_loss(
        state_raw=report["state_activity_side_logits"], future_raw=report["future_activity_side_logits"],
        state_truth=state_truth, future_truth=future_truth,
        state_query_k=torch.from_numpy(expected_query), future_weight=1.0, ta_only=True,
        dedicated_m5_side=True, dedicated_tf_activity=enabled,
        m5_equal_observation=equal_observation)
    assert torch.equal(report["loss"], expected["loss"])
    assert report["loss_contract_sha256"] == expected["contract_sha256"]
    assert report["model_contract_sha256"] == model.research_objective["contract_sha256"]
    for task, truth in (("state", state_truth), ("future", future_truth)):
        for key in ("allowed", "mature"):
            assert np.array_equal(report[task + "_" + key], truth[key])
    if enabled:
        assert report["loss_contract_sha256"] != objective.contract(
            ta_only=True, dedicated_m5_side=True,
            m5_equal_observation=equal_observation)["contract_sha256"]


@pytest.mark.parametrize("where", ["binder", "view", "numerical"])
def test_activity_cannot_be_combined_with_non_m5_side_trial(time_query_case, where):
    if where == "binder":
        with pytest.raises(RuntimeError, match="DEDICATED_TF_ACTIVITY_SCOPE"):
            views.bind_native_direction_feature_view(
                **_arguments(time_query_case), dedicated_tf_activity=True,
                dedicated_non_m5_side=True)
    elif where == "view":
        view = views.bind_native_direction_feature_view(
            **_arguments(time_query_case), dedicated_tf_activity=True)
        view["dedicated_non_m5_side"] = True
        with pytest.raises(RuntimeError, match="VIEW_KEYS"):
            views._require_view(_seal(view))
    else:
        with pytest.raises(ValueError, match="TF_ACTIVITY_MODE"):
            fit.ta_sf_numerical_recipe(dedicated_m5_side=True,
                dedicated_tf_activity=True, dedicated_non_m5_side=True)


@pytest.mark.parametrize("m5,non_m5,equal", [
    (False, False, False), (True, False, False), (True, False, True),
    (True, True, False), (True, True, True)])
def test_default_recipe_preserves_all_existing_profiles(m5, non_m5, equal):
    kwargs = dict(dedicated_m5_side=m5, dedicated_non_m5_side=non_m5,
                  m5_equal_observation=equal)
    original = fit.ta_sf_numerical_recipe(**kwargs)
    explicit = fit.ta_sf_numerical_recipe(**kwargs, dedicated_tf_activity=False)
    assert original == explicit
    assert "dedicated_tf_activity" not in original
    if m5 and not non_m5 and equal:
        assert original["contract_sha256"] == "5085579ff8ad90257412ad1d42406c6c481a1d272f0cf2849ae293da87974361"


@pytest.mark.parametrize("equal", [False, True])
def test_activity_recipe_changes_only_objective_bindings(equal):
    old = fit.ta_sf_numerical_recipe(dedicated_m5_side=True,
                                    m5_equal_observation=equal)
    trial = fit.ta_sf_numerical_recipe(dedicated_m5_side=True,
        m5_equal_observation=equal, dedicated_tf_activity=True)
    assert old.keys() == trial.keys()
    assert {k for k in old if old[k] != trial[k]} == {
        "research_objective_sha256", "native_F_loss_contract_sha256", "contract_sha256"}
    assert trial["native_F_loss_contract_sha256"] == objective.contract(
        ta_only=True, dedicated_m5_side=True, dedicated_tf_activity=True,
        m5_equal_observation=equal)["contract_sha256"]
    assert trial["future_weight"] == 1.0
    assert trial["contract_sha256"] == fit.campaign.canonical_sha256(
        {k: v for k, v in trial.items() if k != "contract_sha256"})


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_activity_numerical_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="TF_ACTIVITY_MODE"):
        fit.ta_sf_numerical_recipe(dedicated_m5_side=True, dedicated_tf_activity=value)
