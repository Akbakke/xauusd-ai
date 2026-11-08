"""Synthetic shared-encoder and separate-head evidence; no fit or raw source."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from gx1.audit import verify_market_state_time_query_v2_zero_step_v1 as zero
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as endpoint_binding
from gx1.contracts import market_state_future_displacement_candidate_v1 as future_binding
from gx1.contracts import market_state_time_query_binding_v1 as state_binding
from gx1.contracts import market_state_time_query_native_F_batch_v1 as dual_batch
from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.contracts import market_state_time_query_loss_v2 as loss
from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.features.entry_specialist_feature_groups_v1 import FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)


def features(batch=1, *, ta_only=False):
    width = 315 if ta_only else 4
    x = torch.arange(batch * width, dtype=torch.float32).reshape(batch, 1, width) + 1
    values = {'seq_x': x, 'snap_x': x[:, 0],
              'ctx_cat': torch.zeros(batch, 1, dtype=torch.int64),
              'ctx_cont': torch.zeros(batch, 132 if ta_only else 1, dtype=torch.float32)}
    for tf in ('m5', 'm15', 'h1', 'h4', 'd1'):
        values['seq_' + tf] = torch.ones(batch, 1, 253 if ta_only else 1)
        if not ta_only:
            values['gc_seq_' + tf] = torch.ones(batch, 1, 1)
    return values


def model(monkeypatch, calls, *, ta_only=False):
    def synthetic_forward(self, seq_x, snap_x, **kwargs):
        calls.append(1)
        assert kwargs.pop('research_representation_only') is True
        return {'entry_q_joint_hidden': self.entry_q_joint_in(
            torch.cat((seq_x[:, 0, :4],) * 4, dim=1))}
    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', synthetic_forward)
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, 'd_model', 4)
    object.__setattr__(cfg, 'entry_training_phase', ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(16, 4)
    native.entry_execution_profile = {'name': FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
    native.gc_mtf_history = {'normalization': {}}
    if ta_only:
        from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
        profile = compile_full_ta_entry_execution_profile()
        native.entry_execution_profile = {key: profile[key] for key in ('name', 'contract_sha256')}
        native.gc_mtf_history = None
    return FactorizedTimeQueryMarketStateModel(native)


def test_one_encoder_call_preserves_v2_S_output_and_adds_native_F(monkeypatch):
    torch.manual_seed(11)
    calls = []
    state = model(monkeypatch, calls)
    q = torch.ones((1, 5), dtype=torch.int64)
    original = state(**features(), query_k=q)
    calls.clear()
    dual = NativeFutureDualModel(state)
    out = dual(**features(), query_k=q)
    assert len(calls) == 1
    assert torch.equal(out['state_activity_side_logits'], original['activity_side_logits'])
    assert torch.equal(out['state_log_probs'], original['logits'])
    assert out['future_activity_side_logits'].shape == (1, 5, 2)
    assert out['future_log_probs'].shape == (1, 5, 3)
    hidden = dual.encode_features(**features())
    expected = dual.future_head(dual.state_model.query_projection(torch.cat((
        hidden[:, None, :].expand(-1, 5, -1),
        dual.state_model.query_tf_coordinates[None],
        torch.ones((1, 5, 1))), dim=-1)))
    assert torch.equal(out['future_activity_side_logits'], expected)


def test_F_loss_changes_shared_encoder_but_not_S_head_and_masks_F_side(monkeypatch):
    torch.manual_seed(12)
    dual = NativeFutureDualModel(model(monkeypatch, []))
    q = torch.tensor([NATIVE_K], dtype=torch.int64)
    out = dual(**features(), query_k=q)
    allowed = torch.tensor([[4, 2, 7, 1, 2]], dtype=torch.uint8)
    eligible = torch.tensor([[True, True, False, True, True]])
    mature = torch.ones((1, 5), dtype=torch.bool)
    report = loss.allowed_set_loss(out['future_activity_side_logits'],
                                   allowed, eligible, mature, q)
    out['future_activity_side_logits'].retain_grad()
    report['loss'].backward()
    grad = out['future_activity_side_logits'].grad
    assert grad is not None
    assert grad[0, 0, 1] != 0 and grad[0, 3, 1] != 0
    assert torch.equal(grad[0, [1, 2, 4], 1], torch.zeros(3))
    assert dual.future_head.weight.grad is not None
    assert dual.state_model.query_projection[0].weight.grad is not None
    assert dual.state_model.query_head.weight.grad is None


def test_real_target_rows_reach_dual_heads_after_one_synthetic_forward(monkeypatch):
    calls = []
    dual = NativeFutureDualModel(model(monkeypatch, calls))
    axes = endpoint_binding.load_bound_five_endpoint_targets().mature_oof_axes(fold_id=1)
    rows, issues = axes['fit_parent_rows'][:8], axes['fit_decision_ns'][:8]
    state_targets = state_binding.load_bound_time_query_targets()
    future_targets = future_binding.load_bound_future_displacement()
    out = dual_batch.evaluate_fit_batch(
        dual, features(batch=8), state_targets, future_targets, rows, issues,
        fold_id=1, pass_index=zero.PASS_INDEX, seed=zero.SEED,
        future_weight=0.25)
    assert len(calls) == 1
    assert out['state_count'] == 25 and out['future_count'] == 40
    assert out['state_query_k'][0].tolist() != list(NATIVE_K)
    assert out['future_native_k'].tolist() == list(NATIVE_K)
    out['loss'].backward()
    assert dual.state_model.query_head.weight.grad is not None
    assert dual.future_head.weight.grad is not None


def test_ta_dual_model_uses_exact_nine_inputs_once_and_rejects_gc(monkeypatch):
    from gx1.contracts.market_state_time_query_native_F_model_v1 import contract
    from gx1.features.entry_specialist_feature_groups_v1 import FULL_TA_ENTRY_FEATURE_KEYS
    calls = []
    state = model(monkeypatch, calls, ta_only=True)
    dual = NativeFutureDualModel(state)
    values = features(ta_only=True)
    assert set(values) == FULL_TA_ENTRY_FEATURE_KEYS
    assert dual.research_objective == contract(ta_only=True)
    assert dual.research_objective != contract()
    out = dual(**values, query_k=torch.ones((1, 5), dtype=torch.int64))
    assert calls == [1]
    assert out['state_activity_side_logits'].shape == out['future_activity_side_logits'].shape == (1, 5, 2)
    with pytest.raises(RuntimeError, match='TA_GC_FEATURES_FORBIDDEN'):
        dual(**values, gc_seq_m5=torch.ones(1, 1, 1), query_k=torch.ones((1, 5), dtype=torch.int64))
    assert calls == [1]
    state.backbone.gc_mtf_history = {'normalization': {}}
    with pytest.raises(RuntimeError, match='FULL_TA_PROFILE'):
        dual(**values, query_k=torch.ones((1, 5), dtype=torch.int64))
