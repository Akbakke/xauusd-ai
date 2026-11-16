"""One opt-in TA encode with TF-local M1 interaction and M5 structure output."""
import pytest
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import (
    IntegratedTAMarketStateModel, masked_structure_hit_sums,
)
from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from tests.test_market_state_time_query_dedicated_m5_side_v1 import WIDTH, _backbone
from tests.test_market_state_time_query_native_F_model import features


def _model(monkeypatch, calls):
    def synthetic_forward(self, seq_x, snap_x, **kwargs):
        assert kwargs.pop('research_representation_only') is True
        assert kwargs.pop('research_timeframe_representations') is True
        calls.append(1)
        rows = seq_x.shape[0]
        hidden = (seq_x[:, :1, :1] + torch.arange(5, dtype=torch.float32)[None, :, None])
        return {'entry_q_timeframe_hidden': hidden.expand(rows, 5, WIDTH).clone()}
    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', synthetic_forward)
    state = FactorizedTimeQueryMarketStateModel(
        _backbone(), dedicated_m5_side=True, dedicated_non_m5_side=True,
        dedicated_tf_activity=True, direct_tf_hidden=True)
    return IntegratedTAMarketStateModel(NativeFutureDualModel(state))


def _inputs(batch=2):
    path = torch.arange(batch * 20, dtype=torch.float32).reshape(batch, 5, 4) / 10
    path[:, -1, 3] = 0
    return dict(features=features(batch, ta_only=True),
                query_k=torch.tensor([NATIVE_K] * batch, dtype=torch.int64),
                m1_path_bps=path,
                m1_observed=torch.ones(batch, 5, dtype=torch.bool))


def test_zero_gate_preserves_dual_outputs_and_uses_one_backbone(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    args = _inputs()
    out = model(**args)
    assert calls == [1]
    calls.clear()
    old = model.dual(**args['features'], query_k=args['query_k'])
    assert calls == [1]
    for key in old:
        assert torch.equal(out[key], old[key])
    assert out['m5_structure_hit_logits'].shape == (2, 2)
    assert model.research_objective['M5_structure_truth_as_forward_input'] is False


def test_m1_gate_is_tf_local_and_reaches_side_and_structure_gradients(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    args = _inputs()
    baseline = model(**args)
    with torch.no_grad():
        model.m1_tf_gate.weight[2 * WIDTH, 0] = 0.1
    changed = dict(args)
    changed['m1_path_bps'] = args['m1_path_bps'].clone()
    changed['m1_path_bps'][:, 0, 0] += 1
    out = model(**changed)
    for task in ('state', 'future'):
        key = task + '_activity_side_logits'
        assert not torch.equal(out[key][:, 2], baseline[key][:, 2])
        assert torch.equal(out[key][:, [0, 1, 3, 4]], baseline[key][:, [0, 1, 3, 4]])
    assert torch.equal(out['m5_structure_hit_logits'], baseline['m5_structure_hit_logits'])
    model.zero_grad(set_to_none=True)
    fresh = model(**args)
    (fresh['state_activity_side_logits'][:, :, 1].sum()
     + fresh['future_activity_side_logits'][:, :, 1].sum()
     + fresh['m5_structure_hit_logits'].sum()).backward()
    assert torch.count_nonzero(model.m1_tf_gate.weight.grad) > 0
    assert torch.count_nonzero(model.m5_structure_head.weight.grad) > 0
    assert model.dual.state_model.m5_side_head.weight.grad is not None
    for head in model.dual.state_model.non_m5_side_heads.values():
        assert head.weight.grad is not None
    assert len(calls) == 3


def test_missing_m1_values_fail_and_structure_mask_has_zero_unknown_gradient(monkeypatch):
    model = _model(monkeypatch, [])
    args = _inputs()
    args['m1_observed'][0, 2] = False
    with pytest.raises(ValueError, match='M1_PATH'):
        model(**args)
    args = _inputs()
    args['m1_path_bps'][0, -1, 3] = 1
    with pytest.raises(ValueError, match='M1_PATH'):
        model(**args)
    logits = torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
    hits = torch.tensor([[True, False], [False, True]])
    observed = torch.tensor([[True, False], [False, True]])
    report = masked_structure_hit_sums(logits, hits, observed)
    assert report['observed_counts'].tolist() == [1, 1]
    report['loss_sums'].sum().backward()
    assert logits.grad[0, 1] == logits.grad[1, 0] == 0
    assert logits.grad[0, 0] != 0 and logits.grad[1, 1] != 0
