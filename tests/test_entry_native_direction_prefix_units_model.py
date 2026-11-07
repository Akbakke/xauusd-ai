"""Metadata delegation and tensor-only units checks; no model construction or fit."""
from copy import deepcopy
import inspect
from types import SimpleNamespace

import pytest
import torch

from gx1.contracts import entry_gc_mtf_history_v1 as history_owner
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as model_owner

Model = model_owner.EntryV10CtxHybridTransformer


class _BeforeModuleInitialization(Exception):
    pass


@pytest.mark.parametrize("profile_kind", [None, "october", "mature_fold1"])
def test_constructor_metadata_delegates_exact_profile_before_any_module_initialization(monkeypatch, profile_kind):
    profile = (None if profile_kind is None else units.october_prefix_units_profile() if profile_kind == "october"
               else units.mature_fold1_prefix_units_profile())
    history = {"synthetic_metadata": True}
    reference = {"name": model_owner.FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
                 "contract_sha256": "1" * 64}
    calls = []

    def require_history(value, *, unobserved_units_profile=None):
        assert value is history and unobserved_units_profile == profile
        calls.append("history")
        return value

    def require_reference(value, **kwargs):
        assert value == reference and kwargs["gc_mtf_history"] is history
        assert kwargs["unobserved_units_profile"] == profile
        calls.append("reference")
        return value

    def compile_profile(value, *, unobserved_units_profile=None):
        assert value is history and unobserved_units_profile == profile
        calls.append("compile")
        return {}

    def observe_architecture(**kwargs):
        assert kwargs["gc_mtf_history"] is history
        assert kwargs["unobserved_units_profile"] == profile
        calls.append("architecture_observation")
        return {"schemas": {}, "entry": {}, "exit": {}, "mtf": {}}

    def require_architecture(value, **kwargs):
        assert kwargs["gc_mtf_history"] is history
        assert kwargs["unobserved_units_profile"] == profile
        calls.append("architecture_require")
        raise _BeforeModuleInitialization

    monkeypatch.setattr(history_owner, "require_gc_mtf_history", require_history)
    monkeypatch.setattr(model_owner, "require_entry_execution_profile_reference", require_reference)
    monkeypatch.setattr(model_owner, "compile_full_gc_mtf_entry_execution_profile", compile_profile)
    monkeypatch.setattr(model_owner, "current_entry_exit_architecture_observation", observe_architecture)
    monkeypatch.setattr(model_owner, "require_entry_exit_production_architecture", require_architecture)
    kwargs = {name: None for name, parameter in inspect.signature(Model.__init__).parameters.items()
              if name != "self" and parameter.default is inspect.Parameter.empty}
    kwargs.update(seq_len=96, input_normalization={"schema_version": "synthetic"},
                  entry_training_phase=model_owner.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
                  entry_execution_profile=reference, gc_mtf_history=history,
                  research_unobserved_units_profile=profile)
    # This is an unbound metadata prefix on a plain namespace. The sentinel
    # stops before nn.Module.__init__, parameter creation, encoders or heads.
    with pytest.raises(_BeforeModuleInitialization):
        Model.__init__(SimpleNamespace(), **kwargs)
    assert calls == ["history", "reference", "compile", "architecture_observation", "architecture_require"]


def _runtime_shell(monkeypatch, profile=None):
    """Small mock contracts test delegation, not genuine normalization admission."""
    profile = units.october_prefix_units_profile() if profile is None else profile
    stats = {"field_names": ["synthetic_value", "synthetic_observed"],
             "center": [0., 0.], "scale": [1., 1.],
             "binary_mask": [0, 1], "categorical_mask": [0, 0],
             "categorical_domains": {}, "observed_fit_counts": [0, 8],
             "unobserved_units_profile_sha256": profile["contract_sha256"]}
    surfaces = {key: deepcopy(stats) for key in model_owner.INPUT_NORMALIZATION_SURFACES}
    contract = {"surfaces": surfaces, "contract_sha256": "1" * 64, "temporal_aliases": [],
                "ctx_cat": {"domains": {key: [0, 1] for key in model_owner.MODEL_NATIVE_CTX_CAT_FIELDS}}}
    history = {"layout": {"contract_sha256": "2" * 64, "timeframes": []},
               "normalization": {"contract_sha256": "3" * 64,
                                 "surfaces": {"gc_mtf_d1": deepcopy(stats)}}}
    shell = SimpleNamespace(
        _entry_execution_profile_identity=None,
        entry_execution_profile={"name": model_owner.FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME},
        entry_training_phase=model_owner.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        _research_unobserved_units_profile=deepcopy(profile), _research_zero_columns={},
        _input_normalization_contract=contract, gc_mtf_history=history,
        _gc_mtf_history_identity=("2" * 64, "3" * 64),
        cfg=SimpleNamespace(gc_mtf_history=history), _gc_mtf_specialist_index_tuples={},
        _input_normalization_field_names={key: list(stats["field_names"])
                                         for key in (*surfaces, "gc_mtf_d1")},
        _input_norm_categorical_domains={key: {} for key in (*surfaces, "gc_mtf_d1")},
        _input_norm_categorical_index_tuples={key: () for key in (*surfaces, "gc_mtf_d1")},
        input_norm_alias_signal_indices=torch.empty(0, dtype=torch.long),
        input_norm_alias_ctx_cont_indices=torch.empty(0, dtype=torch.long),
        input_norm_ctx_cat_min=torch.zeros(len(model_owner.MODEL_NATIVE_CTX_CAT_FIELDS), dtype=torch.long),
        input_norm_ctx_cat_max=torch.ones(len(model_owner.MODEL_NATIVE_CTX_CAT_FIELDS), dtype=torch.long),
    )
    for name, digest in (("research_unobserved_units_profile_sha256", profile["contract_sha256"]),
                         ("input_norm_contract_sha256", "1" * 64),
                         ("gc_mtf_layout_sha256", "2" * 64), ("gc_mtf_normalization_sha256", "3" * 64)):
        setattr(shell, name, torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8))
    for key in (*surfaces, "gc_mtf_d1"):
        for suffix, dtype in (("center", torch.float32), ("scale", torch.float32),
                              ("binary_mask", torch.bool), ("categorical_mask", torch.bool)):
            setattr(shell, f"input_norm_{key}_{suffix}", torch.tensor(stats[suffix], dtype=dtype))
    calls = []

    def check_profile(kwargs):
        assert kwargs["unobserved_units_profile"] == profile

    def require_base(value, **kwargs):
        check_profile(kwargs)
        calls.append("base")
        if value["surfaces"]["signal"]["observed_fit_counts"] != [0, 8]:
            raise RuntimeError("SYNTHETIC_BASE_COUNTS_REJECTED")
        return value

    def require_history(value, **kwargs):
        check_profile(kwargs)
        calls.append("history")
        if value["normalization"]["surfaces"]["gc_mtf_d1"]["observed_fit_counts"] != [0, 8]:
            raise RuntimeError("SYNTHETIC_GC_COUNTS_REJECTED")
        return value

    def normalization_surfaces(value, **kwargs):
        check_profile(kwargs)
        calls.append("surfaces")
        return value["normalization"]["surfaces"]

    def model_layout(value, **kwargs):
        check_profile(kwargs)
        calls.append("layout")
        return {"family_indices": {}}

    monkeypatch.setattr(model_owner, "require_input_normalization_contract", require_base)
    monkeypatch.setattr(history_owner, "require_gc_mtf_history", require_history)
    monkeypatch.setattr(history_owner, "gc_mtf_normalization_surfaces", normalization_surfaces)
    monkeypatch.setattr(history_owner, "gc_mtf_model_layout", model_layout)
    return shell, calls


@pytest.mark.parametrize("profile_kind", ["october", "mature_fold1"])
def test_runtime_revalidates_base_gc_layout_and_profile_with_one_explicit_identity(monkeypatch, profile_kind):
    profile = (units.october_prefix_units_profile() if profile_kind == "october"
               else units.mature_fold1_prefix_units_profile())
    shell, calls = _runtime_shell(monkeypatch, profile)
    original = deepcopy(shell._research_unobserved_units_profile)
    Model.require_input_normalization_state(shell)
    assert calls == ["base", "history", "surfaces", "layout"]
    assert shell._research_unobserved_units_profile == original
    assert shell._research_zero_columns == {}


@pytest.mark.parametrize("fault", ["missing_opt_in", "changed_profile", "changed_sha_buffer",
                                  "base_counts", "gc_counts", "december_marker", "december_projection"])
def test_runtime_rejects_changed_profile_counts_and_mixed_projection(monkeypatch, fault):
    shell, calls = _runtime_shell(monkeypatch)
    if fault == "missing_opt_in":
        shell._research_unobserved_units_profile = None
    elif fault == "changed_profile":
        shell._research_unobserved_units_profile["fit_prefix"]["rows"] += 1
    elif fault == "changed_sha_buffer":
        shell.research_unobserved_units_profile_sha256[0] ^= 1
    elif fault == "base_counts":
        shell._input_normalization_contract["surfaces"]["signal"]["observed_fit_counts"][0] = 1
    elif fault == "gc_counts":
        shell.gc_mtf_history["normalization"]["surfaces"]["gc_mtf_d1"]["observed_fit_counts"][0] = 1
    elif fault == "december_marker":
        shell.gc_mtf_history["normalization"]["surfaces"]["gc_mtf_d1"]["zero_support_profile_sha256"] = "0" * 64
    else:
        shell._research_zero_columns = {"gc_mtf_d1": (0, 1)}
    with pytest.raises(RuntimeError):
        Model.require_input_normalization_state(shell)
    if fault == "base_counts":
        assert calls == ["base"]
    elif fault == "gc_counts":
        assert calls == ["base", "history"]
    else:
        assert calls == []


def test_tensor_transform_preserves_newly_observed_units_without_any_model_or_fit():
    profile = units.october_prefix_units_profile()
    field = next(entry["field"] for entry in profile["fields"]
                 if entry["source_unit"] == "one_source_GC_price_tick")
    names = [field, field + "_observed"]
    shell = SimpleNamespace(
        _research_unobserved_units_profile=profile, _research_zero_columns={},
        _input_normalization_field_names={"gc_mtf_d1": names},
        _input_norm_categorical_domains={"gc_mtf_d1": {}},
        _input_norm_categorical_index_tuples={"gc_mtf_d1": ()},
        input_norm_gc_mtf_d1_center=torch.zeros(2),
        input_norm_gc_mtf_d1_scale=torch.ones(2),
        input_norm_gc_mtf_d1_binary_mask=torch.tensor([False, True]),
        input_norm_gc_mtf_d1_categorical_mask=torch.zeros(2, dtype=torch.bool),
    )
    raw = torch.tensor([[0., 0.], [0., 1.], [7., 1.], [-3., 1.]])
    original = raw.clone()
    encoded = Model._normalize_input_surface(shell, raw, surface="gc_mtf_d1")
    torch.testing.assert_close(encoded[:, 0], torch.asinh(raw[:, 0].double()).float(), rtol=0, atol=0)
    torch.testing.assert_close(encoded[:, 1], raw[:, 1], rtol=0, atol=0)
    assert encoded[0, 0] == encoded[1, 0] == 0
    assert encoded[0, 1] == 0 and encoded[1, 1] == 1
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
