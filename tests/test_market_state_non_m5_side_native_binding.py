"""Synthetic view/factory and native-fit adapter binding; no market data or native forward."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_time_query_native_F_batch_v1 as batch
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_future_displacement_candidate_v1 import BoundFutureDisplacement
from gx1.contracts.market_state_time_query_binding_v1 import BoundTimeQueryTargets, sampled_query_k
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_time_query_model_contract,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TICK_NS, TimeQueryTargetTape
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import FOLD_BOUNDARIES_NS
from tests.test_entry_native_direction_feature_view_v1 import (
    bound_case, _constructor_spy, _seal,
)
from tests.test_market_state_time_query_view_mode import time_query_case
from tests.test_market_state_time_query_v2_wiring import _v2_arguments
from tests.test_market_state_time_query_batch_v1 import _features
from tests.test_market_state_time_query_non_m5_side_v1 import _model


def _arguments(time_query_case):
    return dict(_v2_arguments(time_query_case),
                full_ta_feature_profile=True, dedicated_m5_side=True)


def test_false_flag_preserves_view_bytes_and_new_flag_changes_only_decoder_identity(time_query_case):
    args = _arguments(time_query_case)
    old = views.bind_native_direction_feature_view(**args)
    explicit = views.bind_native_direction_feature_view(**args, dedicated_non_m5_side=False)
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert encode(old) == encode(explicit)
    assert "dedicated_non_m5_side" not in old
    new = views.bind_native_direction_feature_view(**args, dedicated_non_m5_side=True)
    assert new["dedicated_non_m5_side"] is True
    assert new["research_objective"] == factorized_time_query_model_contract(
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True)
    assert new["contract_sha256"] != old["contract_sha256"]
    differences = {key for key in new if new[key] != old.get(key)}
    assert differences == {"dedicated_non_m5_side", "research_objective", "contract_sha256"}
    assert views.bind_native_direction_feature_view(
        **args, dedicated_non_m5_side=True, expected=new) == new
    assert views.bind_native_direction_feature_view(**args, expected=old) == old


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_binder_rejects_non_boolean_flag_before_any_constructor(time_query_case, monkeypatch, value):
    calls, _ = _constructor_spy(monkeypatch)
    with pytest.raises(RuntimeError, match="DEDICATED_NON_M5_SIDE_SCOPE"):
        views.bind_native_direction_feature_view(
            **_arguments(time_query_case), dedicated_non_m5_side=value)
    assert not calls


@pytest.mark.parametrize("field", ["full_ta_feature_profile", "dedicated_m5_side",
                                  "market_state_time_query_v2"])
def test_binder_requires_explicit_ta_m5_and_v2(time_query_case, field):
    args = _arguments(time_query_case)
    args[field] = False
    with pytest.raises(RuntimeError, match="DEDICATED.*SIDE_SCOPE"):
        views.bind_native_direction_feature_view(**args, dedicated_non_m5_side=True)


@pytest.mark.parametrize("value", [False, None, 0, 1, "true"])
def test_present_view_key_requires_literal_true_even_after_resealing(time_query_case, value):
    value_view = views.bind_native_direction_feature_view(
        **_arguments(time_query_case), dedicated_non_m5_side=True)
    value_view["dedicated_non_m5_side"] = value
    with pytest.raises(RuntimeError, match="VIEW_KEYS"):
        views._require_view(_seal(value_view))


@pytest.mark.parametrize("removed", ["full_ta_feature_profile", "dedicated_m5_side"])
def test_resealed_trial_cannot_drop_required_mode_marker(time_query_case, removed):
    value = views.bind_native_direction_feature_view(
        **_arguments(time_query_case), dedicated_non_m5_side=True)
    del value[removed]
    with pytest.raises(RuntimeError, match="VIEW_KEYS"):
        views._require_view(_seal(value))


def test_factory_passes_trial_flag_to_real_wrapper_and_preserves_feature_constructor(time_query_case, monkeypatch):
    args = _arguments(time_query_case)
    old = views.bind_native_direction_feature_view(**args)
    trial = views.bind_native_direction_feature_view(**args, dedicated_non_m5_side=True)
    calls, _ = _constructor_spy(monkeypatch)
    models = [views.build_native_direction_research_model(
        view, prefix_normalization=args["prefix_normalization"],
        constructor_metadata=args["constructor_metadata"], device=torch.device("cpu"))
        for view in (old, trial)]
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert all(type(model) is FactorizedTimeQueryMarketStateModel for model in models)
    assert models[0]._dedicated_non_m5_side is False
    assert not hasattr(models[0], "non_m5_side_heads")
    assert models[1]._dedicated_non_m5_side is True
    assert tuple(models[1].non_m5_side_heads) == ("M15", "H1", "H4", "D1")
    assert models[1].feature_view_identity == {
        "contract_sha256": trial["contract_sha256"],
        "prefix_normalization": trial["prefix_normalization"],
        "research_objective_sha256": trial["research_objective"]["contract_sha256"],
    }
    dual = NativeFutureDualModel(models[1])
    assert tuple(dual.future_non_m5_side_heads) == ("M15", "H1", "H4", "D1")
    dual._require_binding()


@pytest.mark.parametrize("fault", ["false_key", "old_objective"])
def test_factory_rejects_inconsistent_resealed_view_before_backbone(time_query_case, monkeypatch, fault):
    args = _arguments(time_query_case)
    value = views.bind_native_direction_feature_view(**args, dedicated_non_m5_side=True)
    if fault == "false_key":
        value["dedicated_non_m5_side"] = False
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


def _targets():
    """One synthetic B16 group satisfying the real sampled/native target contracts."""
    rows = np.arange(100, 116, dtype=np.int64)
    issues = FOLD_BOUNDARIES_NS[0] - (600 - np.arange(16, dtype=np.int64)) * TICK_NS
    profile = objective.m5_equal_observation_profile()
    query = sampled_query_k(issues, pass_index=0, seed=profile["seed"])
    validity = np.array(sorted({int(issue + k * TICK_NS)
        for issue, sampled in zip(issues, query) for k in (*sampled, *NATIVE_K)}), np.int64)
    tape = TimeQueryTargetTape(validity, np.arange(100, 100 + len(validity), dtype=np.int64),
                              np.tile(np.array([1, 2, 4, 5, 7], np.uint8), (len(validity), 1)))
    native = np.broadcast_to(np.array(NATIVE_K, np.int64), (16, 5)).copy()
    original = tape.gather(issues, native, fold_upper_ns=FOLD_BOUNDARIES_NS[-1])
    state = BoundTimeQueryTargets(rows, issues, tape, issues,
        original["allowed"], original["m1_source_row"], train_end_ns=FOLD_BOUNDARIES_NS[-1])
    future = BoundFutureDisplacement(rows, issues, np.ones((16, 5), bool),
        np.tile(np.array([1, -1, 1, -1, 1], np.int8), (16, 1)),
        np.ones((16, 5), np.float64), np.zeros((16, 5), bool))
    return state, future, rows, issues, profile["seed"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("equal_observation", [False, True])
def test_fit_batch_uses_actual_model_flag_in_original_loss_metadata(monkeypatch, enabled, equal_observation):
    model = _model(non_m5=enabled)
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
        dedicated_m5_side=True, dedicated_non_m5_side=enabled,
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
