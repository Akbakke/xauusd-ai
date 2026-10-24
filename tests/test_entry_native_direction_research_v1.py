"""Synthetic260 component checks; no genuine native encoder/feature evidence.

Native test shells below bypass the production constructor and use a patched
forward spy. They test nominal type/width/one-call wiring and an artificial
hidden leaf only. Fresh native initialization/gradient/resume/data qualification
remains separate. No optimizer, normalizer fit or market input is used.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gx1.contracts import entry_native_direction_research_v1 as owner
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)

MINUTE = 60 * 10**9


def _native_shell():
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, "d_model", 4)
    object.__setattr__(cfg, "entry_training_phase", owner.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(16, 4)
    native.head_dynamic_mfe = nn.Linear(4, 2)
    native.head_forecast = nn.Linear(4, 3)
    native.task_log_variances = nn.ParameterDict({"unused": nn.Parameter(torch.zeros(()))})
    native._synthetic_hidden = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]], requires_grad=True)
    native._spy_fault = None
    native._spy_calls = []
    return native


def _native_forward_spy(self, seq_x, snap_x, **features):
    self._spy_calls.append({"seq_x": seq_x, "snap_x": snap_x, **features})
    hidden = self._synthetic_hidden
    if self._spy_fault == "mapping":
        return None
    if self._spy_fault == "missing":
        return {}
    if self._spy_fault == "dtype":
        hidden = hidden.double()
    if self._spy_fault == "width":
        hidden = torch.ones(2, 5)
    if self._spy_fault == "batch":
        hidden = hidden[:1]
    if self._spy_fault == "nonfinite":
        hidden = torch.full((2, 4), float("nan"))
    return {"entry_q_joint_hidden": hidden,
        "dynamic_mfe_bps": self.head_dynamic_mfe(self._synthetic_hidden),
        "forecast_pred": self.head_forecast(self._synthetic_hidden)}


def _inputs():
    values = {"seq_x": torch.zeros(2, 1, 1), "snap_x": torch.zeros(2, 1),
        "ctx_cat": torch.zeros(2, 1, dtype=torch.int64), "ctx_cont": torch.zeros(2, 1)}
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        values["seq_" + tf] = torch.zeros(2, 1, 1)
        values["gc_seq_" + tf] = torch.ones(2, 1, 1)
    return values


def _component(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer, "forward", _native_forward_spy)
    native = _native_shell()
    return owner.NativeEntryDirectionResearch(native), native


def _loss_case():
    D = torch.tensor([100 * MINUTE, 101 * MINUTE], dtype=torch.int64)
    return {"logit_up": torch.zeros(2, 3, requires_grad=True),
        "sign": torch.tensor([[1, -1, 0], [-1, 0, 1]], dtype=torch.int8),
        "observed": torch.ones(2, 3, dtype=torch.bool), "decision_ns": D,
        "terminal_time_ns": D[:, None] + torch.tensor((1, 5, 15), dtype=torch.int64)[None, :] * MINUTE,
        "cutoff_ns": 200 * MINUTE}


def test_contract_and_returned_objective_are_independent_copies(monkeypatch):
    contract = owner.native_direction_research_contract()
    contract["horizons_minutes"][0] = 99
    contract["native_export_authorized"] = True
    component, _ = _component(monkeypatch)
    copied = component.research_objective
    copied["inactive_backbone_outputs"].clear()
    fresh = component.research_objective
    assert fresh == owner.native_direction_research_contract()
    assert fresh["horizons_minutes"] == [1, 5, 15]
    assert fresh["native_export_authorized"] is fresh["native_launch_authorized"] is False
    assert fresh["backbone_path_forecast_and_task_variance_supervised"] is False
    assert "bound_backbone" in fresh["numerical_policy"]
    assert not hasattr(component, "entry_training_phase")


@pytest.mark.parametrize("kind", ["none", "linear", "subclass"])
def test_constructor_requires_exact_native_type(kind):
    if kind == "subclass":
        class Derived(EntryV10CtxHybridTransformer):
            pass
        value = Derived.__new__(Derived)
        nn.Module.__init__(value)
    else:
        value = None if kind == "none" else nn.Linear(4, 3)
    with pytest.raises(RuntimeError, match="EXACT_NATIVE_TYPE"):
        owner.NativeEntryDirectionResearch(value)


@pytest.mark.parametrize("fault", ["phase", "cfg_phase", "cfg_type", "width_bool", "joint_width", "dtype"])
def test_constructor_rejects_invalid_native_binding(fault):
    native = _native_shell()
    if fault == "phase": native.entry_training_phase = "joint_fitted_q"
    if fault == "cfg_phase": object.__setattr__(native.cfg, "entry_training_phase", "joint_fitted_q")
    if fault == "cfg_type": native.cfg = SimpleNamespace(d_model=4, entry_training_phase=native.entry_training_phase)
    if fault == "width_bool": object.__setattr__(native.cfg, "d_model", True)
    if fault == "joint_width": native.entry_q_joint_in = nn.Linear(16, 5)
    if fault == "dtype": native.entry_q_joint_in.double()
    with pytest.raises(RuntimeError):
        owner.NativeEntryDirectionResearch(native)


def test_one_spy_forward_keeps_features_and_undetched_hidden_gradient(monkeypatch):
    component, native = _component(monkeypatch)
    with torch.no_grad():
        component.direction_head.weight.fill_(0.25)
        component.direction_head.bias.zero_()
    inputs = _inputs()
    output = component(**inputs)
    assert len(native._spy_calls) == 1
    assert all(native._spy_calls[0][key] is value for key, value in inputs.items())
    assert set(output) == {"logit_up", "horizons_minutes"}
    assert output["horizons_minutes"] == (1, 5, 15)
    assert output["logit_up"].shape == (2, 3) and output["logit_up"].dtype == torch.float32
    output["logit_up"].sum().backward()
    assert native._synthetic_hidden.grad is not None and torch.isfinite(native._synthetic_hidden.grad).all()
    assert torch.count_nonzero(native._synthetic_hidden.grad) > 0
    assert component.direction_head.weight.grad is not None and component.direction_head.bias.grad is not None
    assert all(parameter.grad is None for layer in (native.head_dynamic_mfe, native.head_forecast, native.task_log_variances)
               for parameter in layer.parameters())


@pytest.mark.parametrize("fault", ["mapping", "missing", "dtype", "width", "batch", "nonfinite"])
def test_spy_hidden_storage_is_strict(monkeypatch, fault):
    component, native = _component(monkeypatch)
    native._spy_fault = fault
    with pytest.raises(RuntimeError):
        component(**_inputs())
    assert len(native._spy_calls) == 1


@pytest.mark.parametrize("fault", ["metadata", "width", "head", "horizons"])
def test_runtime_binding_tamper_fails_before_spy(monkeypatch, fault):
    component, native = _component(monkeypatch)
    if fault == "metadata": component._research_contract_json = b"{}"
    if fault == "width": component._bound_hidden_width = 5
    if fault == "head": component.direction_head = nn.Linear(4, 2)
    if fault == "horizons": component.direction_horizon_minutes[1] = 6
    with pytest.raises(RuntimeError):
        component(**_inputs())
    assert native._spy_calls == []


def test_forward_has_only_native_feature_arguments(monkeypatch):
    component, native = _component(monkeypatch)
    names = set(inspect.signature(component.forward).parameters)
    assert not names.intersection({"sign", "observed", "decision_ns", "terminal_time_ns", "cutoff_ns", "target"})
    with pytest.raises(TypeError):
        component(**_inputs(), sign=torch.ones(2, 3, dtype=torch.int8))
    assert native._spy_calls == []


def test_loss_known_value_and_exact_masked_gradients():
    case = _loss_case()
    result = owner.native_direction_binary_loss(**case)
    assert result["update_allowed"] is True
    assert result["counts"]["eligible_cells"] == 4
    torch.testing.assert_close(result["loss"], torch.log(torch.tensor(2.0)))
    result["loss"].backward()
    expected = torch.tensor([[-0.125, 0.125, 0.], [0.125, 0., -0.125]])
    torch.testing.assert_close(case["logit_up"].grad, expected, rtol=0., atol=0.)


def test_loss_extreme_logits_are_finite_without_clipping():
    case = _loss_case()
    case["sign"] = torch.tensor([[1, -1, 1], [-1, 1, -1]], dtype=torch.int8)
    case["logit_up"] = torch.tensor([[-1000., 1000., 0.], [1000., -1000., 0.]], requires_grad=True)
    result = owner.native_direction_binary_loss(**case)
    expected = (torch.tensor(4000.) + 2 * torch.log(torch.tensor(2.))) / 6
    torch.testing.assert_close(result["loss"], expected)
    result["loss"].backward()
    assert torch.isfinite(case["logit_up"].grad).all()


def test_cutoff_equality_keeps_all_parent_tie_missing_and_immature_counts():
    case = _loss_case()
    case["cutoff_ns"] = 105 * MINUTE
    case["observed"][1, 1] = False
    result = owner.native_direction_binary_loss(**case)
    assert torch.equal(result["eligible_mask"], torch.tensor([[True, False, False], [True, False, False]]))
    counts = result["counts"]
    assert counts["parent_rows"] == 2 and counts["cells"] == 6
    assert counts["eligible_cells"] == 2 and counts["eligible_up_cells"] == counts["eligible_down_cells"] == 1
    assert counts["observed_tie_cells"] == 1 and counts["missing_cells"] == 1
    assert counts["immature_cells"] == 4 and counts["immature_observed_nonzero_cells"] == 2
    assert result["per_horizon_counts"][1]["eligible_cells"] == 0
    assert result["per_horizon_counts"][1]["immature_cells"] == 2


def test_empty_supervision_is_explicit_no_update_not_zero_loss():
    case = _loss_case()
    case["sign"].zero_()
    result = owner.native_direction_binary_loss(**case)
    assert result["loss"] is None and result["update_allowed"] is False
    assert result["counts"]["eligible_cells"] == 0 and result["counts"]["observed_tie_cells"] == 6
    assert case["logit_up"].grad is None


@pytest.mark.parametrize("fault", [
    "logit_dtype", "logit_nonfinite", "logit_width", "sign_dtype", "sign_domain", "sign_shape",
    "observed_dtype", "observed_shape", "missing_storage", "decision_dtype", "decision_shape",
    "decision_cutoff", "terminal_dtype", "terminal_shape", "terminal_clock", "overflow",
    "cutoff_bool", "cutoff_overflow",
])
def test_invalid_loss_storage_domain_clocks_and_cutoff_reject(fault):
    case = _loss_case()
    if fault == "logit_dtype": case["logit_up"] = case["logit_up"].double()
    if fault == "logit_nonfinite": case["logit_up"] = torch.full((2, 3), float("nan"))
    if fault == "logit_width": case["logit_up"] = torch.zeros(2, 2)
    if fault == "sign_dtype": case["sign"] = case["sign"].to(torch.int64)
    if fault == "sign_domain": case["sign"][0, 0] = 2
    if fault == "sign_shape": case["sign"] = case["sign"][:1]
    if fault == "observed_dtype": case["observed"] = case["observed"].to(torch.int8)
    if fault == "observed_shape": case["observed"] = case["observed"][:1]
    if fault == "missing_storage": case["observed"][0, 0] = False
    if fault == "decision_dtype": case["decision_ns"] = case["decision_ns"].to(torch.int32)
    if fault == "decision_shape": case["decision_ns"] = case["decision_ns"][:, None]
    if fault == "decision_cutoff": case["decision_ns"][0] = case["cutoff_ns"]
    if fault == "terminal_dtype": case["terminal_time_ns"] = case["terminal_time_ns"].double()
    if fault == "terminal_shape": case["terminal_time_ns"] = case["terminal_time_ns"][:, :2]
    if fault == "terminal_clock": case["terminal_time_ns"][0, 0] += 1
    if fault == "overflow":
        case["decision_ns"][0] = 2**63 - 2
        case["cutoff_ns"] = 2**63 - 1
    if fault == "cutoff_bool": case["cutoff_ns"] = True
    if fault == "cutoff_overflow": case["cutoff_ns"] = 2**63
    with pytest.raises(RuntimeError):
        owner.native_direction_binary_loss(**case)


def test_loss_device_mismatch_rejects_before_numerical_work():
    case = _loss_case()
    case["logit_up"] = torch.empty((2, 3), dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="DEVICE"):
        owner.native_direction_binary_loss(**case)
