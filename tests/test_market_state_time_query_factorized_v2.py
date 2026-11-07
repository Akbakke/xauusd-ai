"""Source and synthetic checks for the distinct factorized v2 state objective."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.contracts import market_state_time_query_loss_v1 as previous_loss
from gx1.contracts.market_state_time_query_loss_v2 import (
    allowed_set_loss, factorized_time_query_loss_contract, proper_score_sums,
)
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_log_probabilities,
    factorized_time_query_model_contract,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)


def _features(batch=1):
    x = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 1, 4) + 1
    values = {"seq_x": x, "snap_x": x[:, 0],
              "ctx_cat": torch.zeros(batch, 1, dtype=torch.int64),
              "ctx_cont": torch.zeros(batch, 1, dtype=torch.float32)}
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        values["seq_" + tf] = torch.ones(batch, 1, 1)
        values["gc_seq_" + tf] = torch.ones(batch, 1, 1)
    return values


def _model(monkeypatch):
    def synthetic_forward(self, seq_x, snap_x, **kwargs):
        assert kwargs.pop("research_representation_only") is True
        return {"entry_q_joint_hidden": self.entry_q_joint_in(
            torch.cat((seq_x[:, 0, :],) * 4, dim=1))}

    monkeypatch.setattr(EntryV10CtxHybridTransformer, "forward", synthetic_forward)
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, "d_model", 4)
    object.__setattr__(cfg, "entry_training_phase",
                       ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(16, 4)
    native.entry_execution_profile = {
        "name": FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
    native.gc_mtf_history = {"normalization": {}}
    return FactorizedTimeQueryMarketStateModel(native)


def _query(batch=1):
    return torch.tensor(NATIVE_K, dtype=torch.int64).expand(batch, -1).clone()


def test_actual_two_neuron_head_has_exact_masked_side_gradient(monkeypatch):
    """Check free trainable head rows, not a fixed activity-mass probability swap."""
    torch.manual_seed(51)
    model = _model(monkeypatch)
    ready = torch.ones((1, 5), dtype=torch.bool)
    for tf_index in range(len(TIMEFRAMES)):
        for bit in (5, 2, 1, 4, 7):
            model.zero_grad(set_to_none=True)
            output = model(**_features(), query_k=_query())
            assert set(output) == {"logits", "activity_side_logits"}
            assert output["activity_side_logits"].shape == (1, 5, 2)
            assert output["logits"].shape == (1, 5, 3)
            allowed = torch.full((1, 5), 7, dtype=torch.uint8)
            allowed[0, tf_index] = bit
            result = allowed_set_loss(
                output["activity_side_logits"], allowed, ready, ready, _query())
            result["loss"].backward()
            weight_grad = model.query_head.weight.grad
            bias_grad = model.query_head.bias.grad
            assert weight_grad is not None and bias_grad is not None
            if bit == 7:
                assert result["skip_update"] is True
                assert torch.count_nonzero(weight_grad) == 0
                assert torch.count_nonzero(bias_grad) == 0
                continue
            assert result["count"] == 1 and result["skip_update"] is False
            assert torch.count_nonzero(weight_grad[0]) > 0
            assert bias_grad[0] != 0
            if bit in (5, 2):
                assert torch.count_nonzero(weight_grad[1]) == 0
                assert bias_grad[1] == 0
                assert result["side_known_mask"].sum() == 0
            else:
                assert torch.count_nonzero(weight_grad[1]) > 0
                assert bias_grad[1] > 0 if bit == 1 else bias_grad[1] < 0
                assert result["side_known_mask"].sum() == 1


def test_immature_and_ineligible_side_have_zero_head_gradient(monkeypatch):
    model = _model(monkeypatch)
    allowed = torch.full((1, 5), 4, dtype=torch.uint8)
    eligible = torch.ones((1, 5), dtype=torch.bool)
    mature = eligible.clone()
    mature[0, :] = False
    for e, m in ((eligible, mature), (torch.zeros_like(eligible), eligible)):
        model.zero_grad(set_to_none=True)
        raw = model(**_features(), query_k=_query())["activity_side_logits"]
        result = allowed_set_loss(raw, allowed, e, m, _query())
        assert result["count"] == 0 and result["skip_update"] is True
        result["loss"].backward()
        assert torch.count_nonzero(model.query_head.weight.grad) == 0
        assert torch.count_nonzero(model.query_head.bias.grad) == 0


def test_factorized_probability_loss_and_proper_score_are_consistent():
    raw = torch.tensor([
        [[1.3, -2.2], [-1.0, 0.4], [0.5, 1.1], [2.0, -0.8], [0.0, 0.0]],
        [[-0.2, 1.5], [0.7, -0.6], [1.0, 0.3], [-1.2, 0.9], [0.8, -1.8]],
    ], dtype=torch.float32, requires_grad=True)
    logp = factorized_log_probabilities(raw)
    probabilities = logp.detach().exp().numpy()
    assert logp.shape == (2, 5, 3)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0,
                               rtol=0, atol=2e-7)
    allowed = torch.tensor([[1, 2, 4, 5, 7], [5, 4, 1, 2, 7]],
                           dtype=torch.uint8)
    ready = torch.ones((2, 5), dtype=torch.bool)
    query = _query(2)
    actual = allowed_set_loss(raw, allowed, ready, ready, query)
    old_algebra = previous_loss.allowed_set_loss(
        logp, allowed, ready, ready, query)
    assert float(actual["loss"]) == pytest.approx(
        float(old_algebra["loss"]), abs=2e-7)
    score = proper_score_sums(
        probabilities, allowed.numpy(), ready.numpy(), ready.numpy(),
        query.numpy())
    old_score = previous_loss.proper_score_sums(
        probabilities, allowed.numpy(), ready.numpy(), ready.numpy(),
        query.numpy())
    assert score["per_tf"] == old_score["per_tf"]
    assert score["contract_sha256"] == factorized_time_query_loss_contract()[
        "contract_sha256"]
    assert score["per_tf"]["H4"]["activity"]["count"] == 2
    assert score["per_tf"]["H4"]["conditional_side"]["count"] == 0


def test_model_contract_grid_and_buffer_binding(monkeypatch):
    model = _model(monkeypatch)
    contract = factorized_time_query_model_contract()
    assert model.research_objective == contract
    assert contract["loss_contract_sha256"] == factorized_time_query_loss_contract()[
        "contract_sha256"]
    assert bytes(model.query_objective_sha256.tolist()).hex() == contract[
        "contract_sha256"]
    assert model.query_head.out_features == 2
    hidden = model.encode_features(**_features())
    grid = _query()[:, None, :].expand(-1, 2, -1).clone()
    grid[:, 0, :] = 1
    raw_grid = model.decode_raw_query_grid(hidden, grid)
    logp_grid = model.decode_query_grid(hidden, grid)
    assert raw_grid.shape == (1, 2, 5, 2)
    assert logp_grid.shape == (1, 2, 5, 3)
    assert torch.allclose(logp_grid, factorized_log_probabilities(raw_grid))
    assert torch.allclose(logp_grid[:, 1],
                          model.decode_queries(hidden, _query()))
    with torch.no_grad():
        model.query_objective_sha256[0] ^= 1
    with pytest.raises(RuntimeError, match="BUFFER:query_objective_sha256"):
        model(**_features(), query_k=_query())


def test_reject_bad_primitive_and_query_domains():
    raw = torch.zeros((1, 5, 2), dtype=torch.float32)
    allowed = torch.full((1, 5), 5, dtype=torch.uint8)
    ready = torch.ones((1, 5), dtype=torch.bool)
    with pytest.raises(ValueError, match="TIME_QUERY_FACTORIZED_LOGITS"):
        factorized_log_probabilities(raw[..., :1])
    with pytest.raises(ValueError, match="TIME_QUERY_FACTORIZED_LOGITS"):
        allowed_set_loss(raw.double(), allowed, ready, ready, _query())
    query = _query()
    query[0, 0] = 0
    with pytest.raises(ValueError, match="TIME_QUERY_K"):
        allowed_set_loss(raw, allowed, ready, ready, query)
