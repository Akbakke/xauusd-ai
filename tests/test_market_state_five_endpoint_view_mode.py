"""Synthetic five-endpoint feature-view mode; no native fit or market-data read."""
from copy import deepcopy

import pytest

from gx1.contracts import entry_native_direction_feature_view_v1 as view
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as targets
from gx1.contracts.market_state_five_endpoint_model_v1 import five_endpoint_model_contract
from tests.test_entry_native_direction_feature_view_v1 import _ref, bound_case as standard_case
from tests.test_entry_native_direction_prefix_units_routing import binding_case


def _new_arguments(binding_case):
    args = deepcopy(binding_case["arguments"])
    for key in ("dynamic_state_supervision", "current_state_first_loss_weight",
                "first_flat_activity_weight", "native_endpoint_loss"):
        del args[key]
    args["market_state_five_endpoint"] = True
    sources = args["source_bindings"]
    del sources["dynamic_state_target_source"]
    for key in view.FIVE_ENDPOINT_SOURCE_REF_KEYS - {"five_endpoint_corrected_mask"}:
        sources[key] = _ref(key + ".py")
    path, sha256 = targets.REFS["mask"]
    sources["five_endpoint_corrected_mask"] = {"path": path, "sha256": sha256}
    return args


def test_five_endpoint_view_is_distinct_from_old_state_and_exactly_rebinds(binding_case):
    args = _new_arguments(binding_case)
    bound = view.bind_native_direction_feature_view(**args)
    old = binding_case["view"]
    assert bound["research_objective"] == five_endpoint_model_contract()
    assert bound["research_objective"]["contract_sha256"] != old["research_objective"]["contract_sha256"]
    assert bound["contract_sha256"] != old["contract_sha256"]
    assert bound["market_state_five_endpoint"] is True
    assert "dynamic_state_supervision" not in bound
    assert "native_endpoint_loss" not in bound
    assert "dynamic_state_target_source" not in bound["source_bindings"]
    assert bound["source_bindings"]["five_endpoint_corrected_mask"] == {
        "path": targets.REFS["mask"][0], "sha256": targets.REFS["mask"][1]}
    assert view.bind_native_direction_feature_view(**args, expected=bound) == bound
    assert view._require_view(bound, unobserved_units_profile=binding_case["profile"]) == bound
    assert view.bind_native_direction_feature_view(
        **binding_case["arguments"], expected=old) == old


@pytest.mark.parametrize("change,reason", [
    ("missing_source", "SOURCE_REF_KEYS"),
    ("wrong_mask", "FIVE_ENDPOINT_CORRECTED_MASK"),
    ("old_state", "OBJECTIVE_MODE"),
    ("old_current", "OBJECTIVE_MODE"),
    ("old_endpoint", "OBJECTIVE_MODE"),
    ("not_full", "UNOBSERVED_UNITS_FIT_BINDING"),
])
def test_mode_fails_closed_on_source_or_old_objective_mix(binding_case, change, reason):
    args = _new_arguments(binding_case)
    if change == "missing_source":
        del args["source_bindings"]["five_endpoint_model_source"]
    elif change == "wrong_mask":
        args["source_bindings"]["five_endpoint_corrected_mask"] = _ref("wrong.npz")
    elif change == "old_state":
        args["dynamic_state_supervision"] = True
        args["source_bindings"]["dynamic_state_target_source"] = _ref("old.py")
    elif change == "old_current":
        args["current_state_first_loss_weight"] = .03
    elif change == "old_endpoint":
        args["native_endpoint_loss"] = True
    elif change == "not_full":
        args["full_feature_profile"] = False
    with pytest.raises(RuntimeError, match=reason):
        view.bind_native_direction_feature_view(**args)


def test_resealed_wrong_objective_is_rejected(binding_case):
    bound = view.bind_native_direction_feature_view(**_new_arguments(binding_case))
    forged = deepcopy(bound)
    forged["research_objective"] = binding_case["view"]["research_objective"]
    forged["contract_sha256"] = view._sha({k: v for k, v in forged.items() if k != "contract_sha256"})
    with pytest.raises(RuntimeError, match="VIEW_CANONICAL_IDENTITY"):
        view._require_view(forged, unobserved_units_profile=binding_case["profile"])


def test_builder_selects_only_new_wrapper_after_exact_rebind(binding_case, monkeypatch):
    bound = view.bind_native_direction_feature_view(**_new_arguments(binding_case))
    calls = []

    class BackboneShell:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.entry_execution_profile = kwargs["entry_execution_profile"]
            self.gc_mtf_history = kwargs["gc_mtf_history"]

        def to(self, device):
            return self

        def require_input_normalization_state(self):
            calls.append("runtime-normalization")

    class NewShell:
        def __init__(self, backbone):
            self.backbone = backbone
            calls.append("five-endpoint-wrapper")

    def old_wrapper(*args, **kwargs):
        raise AssertionError("old state wrapper must not be constructed")

    monkeypatch.setattr(view, "EntryV10CtxHybridTransformer", BackboneShell)
    monkeypatch.setattr(view, "FiveEndpointMarketStateModel", NewShell)
    monkeypatch.setattr(view, "NativeDynamicMTFStateResearch", old_wrapper)
    factory = {key: binding_case["arguments"][key]
               for key in ("prefix_normalization", "constructor_metadata")}
    with pytest.raises(RuntimeError, match="PREFIX_NORMALIZATION_SCOPE"):
        view.build_native_direction_research_model(bound, **factory, device="synthetic")
    assert calls == []
    result = view.build_native_direction_research_model(
        bound, **factory, device="synthetic",
        unobserved_units_profile=binding_case["profile"])
    kwargs = calls[0]
    assert kwargs["research_unobserved_units_profile"] == binding_case["profile"]
    assert kwargs["gc_mtf_history"] == factory["prefix_normalization"]["gc_mtf_history"]
    assert all(tf + "_seq_dim" in kwargs and tf + "_seq_len" in kwargs
               for tf in ("m5", "m15", "h1", "h4", "d1"))
    assert calls[1:] == ["runtime-normalization", "five-endpoint-wrapper"]
    assert result.feature_view_identity["contract_sha256"] == bound["contract_sha256"]
    assert result.feature_view_identity["research_objective_sha256"] == five_endpoint_model_contract()["contract_sha256"]



def test_standard_full_profile_rebinds_and_builds_without_october(standard_case, monkeypatch):
    args = deepcopy(standard_case["arguments"])
    args["full_feature_profile"] = True
    args["market_state_five_endpoint"] = True
    for key in view.FIVE_ENDPOINT_SOURCE_REF_KEYS - {"five_endpoint_corrected_mask"}:
        args["source_bindings"][key] = _ref(key + ".py")
    path, sha256 = targets.REFS["mask"]
    args["source_bindings"]["five_endpoint_corrected_mask"] = {"path": path, "sha256": sha256}
    bound = view.bind_native_direction_feature_view(**args)
    assert bound["research_objective"] == five_endpoint_model_contract()
    assert bound["contract_sha256"] != standard_case["view"]["contract_sha256"]
    assert "unobserved_units_profile" not in bound
    assert view.bind_native_direction_feature_view(**args, expected=bound) == bound
    assert view.bind_native_direction_feature_view(
        **standard_case["arguments"], expected=standard_case["view"]) == standard_case["view"]
    calls = []

    class BackboneShell:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.entry_execution_profile = kwargs["entry_execution_profile"]
            self.gc_mtf_history = kwargs["gc_mtf_history"]

        def to(self, device):
            return self

        def require_input_normalization_state(self):
            calls.append("runtime-normalization")

    class NewShell:
        def __init__(self, backbone):
            self.backbone = backbone
            calls.append("five-endpoint-wrapper")

    monkeypatch.setattr(view, "EntryV10CtxHybridTransformer", BackboneShell)
    monkeypatch.setattr(view, "FiveEndpointMarketStateModel", NewShell)
    factory = {key: args[key] for key in ("prefix_normalization", "constructor_metadata")}
    result = view.build_native_direction_research_model(bound, **factory, device="synthetic")
    assert "research_unobserved_units_profile" not in calls[0]
    assert calls[1:] == ["runtime-normalization", "five-endpoint-wrapper"]
    assert result.feature_view_identity["research_objective_sha256"] == five_endpoint_model_contract()["contract_sha256"]
