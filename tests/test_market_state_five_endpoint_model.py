"""Synthetic output-path checks only; no native fit or market-data qualification."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.contracts.market_state_five_endpoint_contract_v1 import (
    CLASS_NAMES, LEAD_MINUTES, TIMEFRAMES, five_endpoint_contract,
)
from gx1.contracts.market_state_five_endpoint_model_v1 import (
    FiveEndpointMarketStateModel, five_endpoint_model_contract,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)


def _features(batch=2):
    x = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 1, 4) + 1
    values = {"seq_x": x, "snap_x": x[:, 0], "ctx_cat": torch.zeros(batch, 1, dtype=torch.int64),
              "ctx_cont": torch.zeros(batch, 1, dtype=torch.float32)}
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        values["seq_" + tf] = torch.ones(batch, 1, 1)
        values["gc_seq_" + tf] = torch.ones(batch, 1, 1)
    return values


def _model(monkeypatch):
    def synthetic_forward(self, seq_x, snap_x, **kwargs):
        assert kwargs.pop("research_representation_only") is True
        assert set(kwargs) == {"ctx_cat", "ctx_cont"} | {
            kind + tf for kind in ("seq_", "gc_seq_")
            for tf in ("m5", "m15", "h1", "h4", "d1")}
        return {"entry_q_joint_hidden": self.entry_q_joint_in(
            torch.cat((seq_x[:, 0, :],) * 4, dim=1))}

    monkeypatch.setattr(EntryV10CtxHybridTransformer, "forward", synthetic_forward)
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, "d_model", 4)
    object.__setattr__(cfg, "entry_training_phase", ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(16, 4)
    native.entry_execution_profile = {"name": FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
    native.gc_mtf_history = {"normalization": {}}
    return FiveEndpointMarketStateModel(native)


def test_distinct_objective_and_five_paired_query_coordinates(monkeypatch):
    model = _model(monkeypatch)
    objective = model.research_objective
    assert objective["target_contract_sha256"] == five_endpoint_contract()["contract_sha256"]
    assert objective["contract_sha256"] == five_endpoint_model_contract()["contract_sha256"]
    assert objective["timeframes"] == list(TIMEFRAMES)
    assert objective["lead_minutes"] == list(LEAD_MINUTES)
    assert objective["class_order"] == list(CLASS_NAMES)
    assert tuple(model.endpoint_query_coordinates.shape) == (5, 5)
    assert torch.equal(model.endpoint_query_coordinates, torch.eye(5))
    assert model.endpoint_lead_minutes.tolist() == list(LEAD_MINUTES)
    assert not hasattr(model, "state_transition_head")
    assert not hasattr(model, "state_current_projection")
    assert not hasattr(model, "dynamic_m5_query_projection")


def test_output_shape_gradient_and_batch_isolation(monkeypatch):
    torch.manual_seed(8)
    model = _model(monkeypatch)
    features = _features()
    logits = model(**features)["logits"]
    assert logits.shape == (2, 5, 3) and torch.isfinite(logits).all()
    assert torch.allclose(logits[:1], model(**_features(batch=1))["logits"])
    changed = copy.copy(features)
    changed["seq_x"] = features["seq_x"].clone()
    changed["seq_x"][1] *= 100
    assert torch.allclose(logits[:1], model(**changed)["logits"][:1])
    logits[..., 0].sum().backward()
    assert model.endpoint_head.weight.grad is not None
    assert model.endpoint_query_projection[0].weight.grad is not None
    assert model.backbone.entry_q_joint_in.weight.grad is not None
    assert torch.count_nonzero(model.backbone.entry_q_joint_in.weight.grad) > 0


def test_no_label_or_current_state_input_and_fresh_head_binding(monkeypatch):
    model = _model(monkeypatch)
    with pytest.raises(TypeError):
        model(**_features(), current_state=torch.zeros(2, 5))
    with pytest.raises(TypeError):
        model(**_features(), allowed_set=torch.ones(2, 5))
    old = model.state_dict()
    old["state_transition_head.weight"] = torch.ones(9, 4)
    with pytest.raises(RuntimeError):
        model.load_state_dict(old, strict=True)
    with torch.no_grad():
        model.endpoint_query_coordinates[0, 0] = 0
    with pytest.raises(RuntimeError, match="BUFFER:endpoint_query_coordinates"):
        model(**_features())


def test_backbone_phase_and_profile_fail_closed(monkeypatch):
    model = _model(monkeypatch)
    model.backbone.entry_training_phase = "wrong"
    with pytest.raises(RuntimeError, match="BACKBONE_PHASE"):
        model(**_features())
    model.backbone.entry_training_phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    model.backbone.entry_execution_profile = {"name": "wrong"}
    with pytest.raises(RuntimeError, match="FULL_GC_PROFILE"):
        model(**_features())
