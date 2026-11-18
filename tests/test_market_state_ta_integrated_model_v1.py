"""M5/MTF direction uses one native encode and excludes M1 model input."""
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
    return dict(features=features(batch, ta_only=True),
                query_k=torch.tensor([NATIVE_K] * batch, dtype=torch.int64))


def test_direct_m5_mtf_heads_preserve_dual_outputs_and_use_one_backbone(monkeypatch):
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


def test_m5_input_reaches_side_and_structure_gradients(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    args = _inputs()
    args['features']['seq_x'].requires_grad_()
    out = model(**args)
    (out['state_activity_side_logits'][:, :, 1].sum()
     + out['future_activity_side_logits'][:, :, 1].sum()
     + out['m5_structure_hit_logits'].sum()).backward()
    assert torch.count_nonzero(args['features']['seq_x'].grad) > 0
    assert torch.count_nonzero(model.m5_structure_head.weight.grad) > 0
    assert model.dual.state_model.m5_side_head.weight.grad is not None
    for head in model.dual.state_model.non_m5_side_heads.values():
        assert head.weight.grad is not None
    assert calls == [1]


def test_m1_inputs_and_legacy_checkpoint_are_rejected(monkeypatch):
    model = _model(monkeypatch, [])
    args = _inputs()
    for name, value in (
        ('m1_path_bps', torch.zeros(2, 5, 4)),
        ('m1_observed', torch.ones(2, 5, dtype=torch.bool)),
    ):
        with pytest.raises(TypeError, match=name):
            model(**args, **{name: value})
    assert not any('m1' in name for name, _ in model.named_parameters())
    assert model.research_objective['forecast_input_timeframes'] == [
        'M5', 'M15', 'H1', 'H4', 'D1']
    legacy = dict(model.state_dict())
    legacy['m1_tf_gate.weight'] = torch.zeros(5 * WIDTH, 25)
    with pytest.raises(RuntimeError, match='Unexpected key'):
        model.load_state_dict(legacy)
    with torch.no_grad():
        model.integrated_objective_sha256.zero_()
    with pytest.raises(ValueError, match='MODEL_BINDING'):
        model(**args)


def test_structure_mask_has_zero_unknown_gradient():
    logits = torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
    hits = torch.tensor([[True, False], [False, True]])
    observed = torch.tensor([[True, False], [False, True]])
    report = masked_structure_hit_sums(logits, hits, observed)
    assert report['observed_counts'].tolist() == [1, 1]
    report['loss_sums'].sum().backward()
    assert logits.grad[0, 1] == logits.grad[1, 0] == 0
    assert logits.grad[0, 0] != 0 and logits.grad[1, 1] != 0


def test_paired_native_and_sampled_queries_share_one_encoder(monkeypatch):
    calls = []
    model = _model(monkeypatch, calls)
    args = _inputs()
    sampled = torch.tensor([[1, 1, 2, 4, 24]] * 2, dtype=torch.int64)
    native = torch.tensor([NATIVE_K] * 2, dtype=torch.int64)
    paired = model(**{**args, 'query_k': torch.stack((sampled, native), dim=1)})
    assert calls == [1]
    assert paired['state_native_activity_side_logits'].shape == (2, 5, 2)
    assert paired['state_native_log_probs'].shape == (2, 5, 3)
    calls.clear()
    old_sampled = model(**{**args, 'query_k': sampled})
    old_native = model(**{**args, 'query_k': native})
    assert calls == [1, 1]
    assert torch.equal(paired['state_activity_side_logits'], old_sampled['state_activity_side_logits'])
    torch.testing.assert_close(paired['state_native_activity_side_logits'],
                               old_native['state_activity_side_logits'], atol=1e-5, rtol=1e-6)
    assert torch.equal(paired['future_activity_side_logits'], old_native['future_activity_side_logits'])
    assert torch.equal(paired['m5_structure_hit_logits'], old_native['m5_structure_hit_logits'])
