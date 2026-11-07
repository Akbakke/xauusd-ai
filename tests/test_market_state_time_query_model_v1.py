"""Synthetic time-query output checks; no native fit or market-data qualification."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.contracts.market_state_five_endpoint_contract_v1 import CLASS_NAMES
from gx1.contracts.market_state_time_query_model_v1 import (
    TimeQueryMarketStateModel, time_query_model_contract,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)


def _features(batch=2):
    x = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 1, 4) + 1
    values = {"seq_x": x, "snap_x": x[:, 0],
              "ctx_cat": torch.zeros(batch, 1, dtype=torch.int64),
              "ctx_cont": torch.zeros(batch, 1, dtype=torch.float32)}
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        values["seq_" + tf] = torch.ones(batch, 1, 1)
        values["gc_seq_" + tf] = torch.ones(batch, 1, 1)
    return values


def _model(monkeypatch, calls=None):
    def synthetic_forward(self, seq_x, snap_x, **kwargs):
        if calls is not None:
            calls.append(1)
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
    return TimeQueryMarketStateModel(native)


def _query(batch=2):
    return torch.tensor(NATIVE_K, dtype=torch.int64).expand(batch, -1).clone()


def test_contract_and_query_coordinates(monkeypatch):
    model = _model(monkeypatch)
    objective = model.research_objective
    assert objective == time_query_model_contract()
    assert objective["timeframes"] == list(TIMEFRAMES)
    assert objective["native_k"] == list(NATIVE_K)
    assert objective["class_order"] == list(CLASS_NAMES)
    assert objective["old_checkpoint_compatible"] is False
    assert torch.equal(model.query_tf_coordinates, torch.eye(5))
    assert model.query_native_k.tolist() == list(NATIVE_K)
    assert bytes(model.query_objective_sha256.tolist()).hex() == objective["contract_sha256"]


def test_one_encoding_multiple_queries_and_time_dependence(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    with torch.no_grad():
        model.query_projection[0].weight.zero_()
        model.query_projection[0].bias.zero_()
        model.query_projection[0].weight[0, -1] = 1
        model.query_head.weight.zero_()
        model.query_head.bias.zero_()
        model.query_head.weight[0, 0] = 1
    hidden = model.encode_features(**_features())
    near = torch.ones(2, 5, dtype=torch.int64)
    native = _query()
    near_logits = model.decode_queries(hidden, near)
    native_logits = model.decode_queries(hidden, native)
    assert len(calls) == 1
    assert near_logits.shape == native_logits.shape == (2, 5, 3)
    assert torch.isfinite(near_logits).all() and torch.isfinite(native_logits).all()
    assert torch.allclose(near_logits[:, 0], native_logits[:, 0])
    assert torch.all(native_logits[:, 1:, 0] > near_logits[:, 1:, 0])
    composed = model(**_features(), query_k=native)["logits"]
    assert len(calls) == 2 and torch.allclose(composed, native_logits)
    assert torch.allclose(composed[:1], model(**_features(batch=1), query_k=_query(1))["logits"])


def test_zero_step_gradients_flow_through_backbone_and_shared_head(monkeypatch):
    torch.manual_seed(8)
    model = _model(monkeypatch)
    logits = model(**_features(), query_k=_query())["logits"]
    logits[..., 0].sum().backward()
    for weight in (model.backbone.entry_q_joint_in.weight,
                   model.query_projection[0].weight, model.query_head.weight):
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        assert torch.count_nonzero(weight.grad) > 0


@pytest.mark.parametrize("query", [
    lambda q: q.to(torch.float32),
    lambda q: q[:, :4],
    lambda q: q.index_fill(1, torch.tensor([0]), 0),
    lambda q: q.index_fill(1, torch.tensor([1]), NATIVE_K[1] + 1),
])
def test_invalid_time_query_rejected(monkeypatch, query):
    model = _model(monkeypatch)
    hidden = model.encode_features(**_features())
    with pytest.raises(RuntimeError, match="TIME_QUERY_MODEL_QUERY_K"):
        model.decode_queries(hidden, query(_query()))


def test_feature_and_hidden_tensor_guards(monkeypatch):
    model = _model(monkeypatch)
    features = _features()
    features["snap_x"] = features["snap_x"].to(torch.float64)
    with pytest.raises(RuntimeError, match="FEATURE:snap_x"):
        model(**features, query_k=_query())
    features = _features()
    features["seq_h1"][0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="FEATURE:seq_h1"):
        model(**features, query_k=_query())
    hidden = model.encode_features(**_features())
    with pytest.raises(RuntimeError, match="TIME_QUERY_MODEL_HIDDEN"):
        model.decode_queries(hidden.to(torch.float64), _query())
    with pytest.raises(RuntimeError, match="TIME_QUERY_MODEL_HIDDEN"):
        model.decode_queries(hidden[:, :3], _query())


def test_no_future_target_input_and_binding_guard(monkeypatch):
    model = _model(monkeypatch)
    with pytest.raises(TypeError):
        model(**_features(), query_k=_query(), allowed_set=torch.ones(2, 5))
    with pytest.raises(TypeError):
        model(**_features(), query_k=_query(), current_state=torch.ones(2, 5))
    with torch.no_grad():
        model.query_tf_coordinates[0, 0] = 0
    with pytest.raises(RuntimeError, match="BUFFER:query_tf_coordinates"):
        model(**_features(), query_k=_query())


def _query_grid(batch=2, steps=6):
    requested = torch.arange(1, steps + 1, dtype=torch.int64)[None, :, None]
    native = torch.tensor(NATIVE_K, dtype=torch.int64)[None, None, :]
    return torch.minimum(requested, native).expand(batch, -1, -1).clone()


def test_query_grid_matches_repeated_single_decodes_with_one_binding(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    hidden = model.encode_features(**_features())
    assert len(calls) == 1
    binding_checks = []
    original = model._require_binding

    def counted_binding():
        binding_checks.append(1)
        original()

    monkeypatch.setattr(model, "_require_binding", counted_binding)
    query = _query_grid()
    grid = model.decode_query_grid(hidden, query)
    assert len(binding_checks) == 1 and len(calls) == 1
    repeated = torch.stack([model.decode_queries(hidden, query[:, q, :])
                            for q in range(query.shape[1])], dim=1)
    assert grid.shape == (2, 6, 5, 3)
    assert torch.allclose(grid, repeated, atol=1e-6, rtol=1e-6)
    assert len(calls) == 1


def test_query_grid_zero_step_gradient_and_upper_block_bound(monkeypatch):
    torch.manual_seed(9)
    calls = []
    model = _model(monkeypatch, calls)
    hidden = model.encode_features(**_features())
    logits = model.decode_query_grid(hidden, _query_grid(steps=32))
    assert logits.shape == (2, 32, 5, 3) and len(calls) == 1
    logits[..., 0].sum().backward()
    for weight in (model.backbone.entry_q_joint_in.weight,
                   model.query_projection[0].weight, model.query_head.weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all()
        assert torch.count_nonzero(weight.grad) > 0


@pytest.mark.parametrize("query", [
    lambda q: q[:, :0],
    lambda q: _query_grid(steps=33),
    lambda q: q[:, :, :4],
    lambda q: q.to(torch.float32),
    lambda q: q.index_fill(2, torch.tensor([0]), 0),
    lambda q: q.index_fill(2, torch.tensor([1]), NATIVE_K[1] + 1),
])
def test_invalid_query_grid_rejected(monkeypatch, query):
    model = _model(monkeypatch)
    hidden = model.encode_features(**_features())
    with pytest.raises(RuntimeError, match="TIME_QUERY_MODEL_QUERY_K"):
        model.decode_query_grid(hidden, query(_query_grid()))
