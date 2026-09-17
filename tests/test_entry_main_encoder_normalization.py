"""Main-path scale control and exact preservation of the frozen teacher."""
import math
from contextlib import nullcontext

import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as module
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_entry_v10_ctx_model_shapes import _make_inputs, _make_model
from tests.test_unified_exit_random_access_model_v1 import _inputs as _exit_inputs
from tests.test_entry_residual_input_normalization import _assert_equal


def test_main_encoder_bounds_each_token_and_pool_without_mixing_rows_or_losing_gradients():
    torch.manual_seed(918)
    model = _make_model(dropout=0.0).train()
    encoder = model.encoder
    d = encoder.layers[0].self_attn.embed_dim
    assert encoder.norm.state_dict() == {} and not encoder.norm.elementwise_affine
    x = (torch.randn(2, 5, d) + 100 * torch.randn(1, 1, d)).requires_grad_()
    out = module._memory_bounded_transformer_encoder(encoder, x)
    assert torch.all(torch.linalg.vector_norm(out, dim=-1) <= math.sqrt(d) + 1e-5)
    assert torch.all(torch.linalg.vector_norm(out.mean(1), dim=-1) <= math.sqrt(d) + 1e-5)
    changed = x.detach().clone(); changed[1] *= 1000
    actual = module._memory_bounded_transformer_encoder(encoder, changed)
    torch.testing.assert_close(out[0], actual[0], atol=0, rtol=0)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in encoder.parameters())
    encoder.eval()
    with torch.inference_mode():
        inference = module._memory_bounded_transformer_encoder(encoder, x.detach())
    torch.testing.assert_close(out.detach(), inference, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('inference', [False, True])
def test_frozen_teacher_entry_token_exit_and_rng_match_original_constructor(monkeypatch, inference):
    torch.manual_seed(20260911)
    online = _make_model(dropout=0.0).eval()
    rng = torch.get_rng_state().clone()
    constructor = module.nn.TransformerEncoder
    def original_constructor(*args, **kwargs):
        kwargs.pop('norm', None)
        return constructor(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module.nn, 'TransformerEncoder', original_constructor)
        torch.manual_seed(20260911)
        original = _make_model(dropout=0.0).eval().requires_grad_(False)
    assert torch.equal(rng, torch.get_rng_state())
    _assert_equal(online.state_dict(), original.state_dict())
    teacher = trainer._copy_frozen_prefix_reference_model(online)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    assert online.encoder.norm is not None and teacher.encoder.norm is None
    _assert_equal(teacher.state_dict(), online.state_dict())
    seq, snap, cat, ctx, mtf = _make_inputs(2)
    exits = _exit_inputs(2)
    with torch.inference_mode() if inference else nullcontext():
        expected = original(seq, snap, ctx_cat=cat, ctx_cont=ctx, **mtf)
        actual = teacher(seq, snap, ctx_cat=cat, ctx_cont=ctx, **mtf)
        _assert_equal(expected, actual)
        expected_exit = original.forward_exit_random_access_batch(
            **(exits | {'entry_decision_representation': expected['entry_decision_representation']}))
        actual_exit = teacher.forward_exit_random_access_batch(
            **(exits | {'entry_decision_representation': actual['entry_decision_representation']}))
        _assert_equal(expected_exit, actual_exit)
        changed = online(seq, snap, ctx_cat=cat, ctx_cont=ctx, **mtf)
        assert not torch.equal(changed['entry_action_q_bps'], actual['entry_action_q_bps'])


def test_reference_copy_rejects_unbound_affine_norm_and_preserves_online():
    model = _make_model(dropout=0.0)
    model.encoder.norm = torch.nn.LayerNorm(model.encoder.layers[0].self_attn.embed_dim)
    before = model.encoder.norm
    with pytest.raises(RuntimeError, match='REFERENCE_ENCODER_FUNCTION_INVALID'):
        trainer._copy_frozen_prefix_reference_model(model)
    assert model.encoder.norm is before
