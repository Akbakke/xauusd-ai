"""Scale control, causal row isolation and fresh Entry/Exit equivalence."""
from contextlib import nullcontext

import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as module
from tests.test_entry_v10_ctx_model_shapes import _make_inputs, _make_model
from tests.test_unified_exit_random_access_model_v1 import _inputs as _exit_inputs


PORTS = ("specialist_out", "cross_tf_out", "family_tf_cooperation_out")


def _assert_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_equal(x, y)
    else:
        assert a == b


def test_projection_bounds_gain_without_batch_mixing_or_lost_gradients():
    torch.manual_seed(917)
    layer = module._InputNormalizedResidualLinear(128, 128)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(128))
        layer.bias.zero_()
    x = torch.randn(3, 128, requires_grad=True)
    output = layer(x)
    amplified = layer(x * 1000)
    assert torch.all(output.square().mean(-1) <= 1.000001)
    assert torch.allclose(output, amplified, atol=5e-5, rtol=5e-5)
    assert not torch.allclose(output[0], output[1])
    changed = x.detach().clone()
    changed[1:] *= 10000
    assert torch.equal(layer(changed)[0], output[0])
    (output * torch.randn_like(output)).sum().backward()
    for grad in (x.grad, layer.weight.grad, layer.bias.grad):
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    assert torch.equal(layer(torch.zeros_like(x)), torch.zeros_like(output))


@pytest.mark.parametrize("inference", [False, True])
def test_fresh_entry_token_and_random_access_exit_remain_exact(monkeypatch, inference):
    torch.manual_seed(20260911)
    candidate = _make_model(dropout=0.0).eval()
    candidate_rng = torch.get_rng_state().clone()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_InputNormalizedResidualLinear", torch.nn.Linear)
        torch.manual_seed(20260911)
        original = _make_model(dropout=0.0).eval()
    assert torch.equal(candidate_rng, torch.get_rng_state())
    _assert_equal(candidate.state_dict(), original.state_dict())
    for name in PORTS:
        layer = getattr(candidate, name)
        assert isinstance(layer, module._InputNormalizedResidualLinear)
        assert torch.count_nonzero(layer.weight) == torch.count_nonzero(layer.bias) == 0
    seq, snap, cat, ctx, mtf = _make_inputs(2)
    exits = _exit_inputs(2)
    with torch.inference_mode() if inference else nullcontext():
        reference = original(seq, snap, ctx_cat=cat, ctx_cont=ctx, **mtf)
        actual = candidate(seq, snap, ctx_cat=cat, ctx_cont=ctx, **mtf)
        _assert_equal(reference, actual)
        reference_exit = original.forward_exit_random_access_batch(
            **(exits | {"entry_decision_representation": reference["entry_decision_representation"]}))
        actual_exit = candidate.forward_exit_random_access_batch(
            **(exits | {"entry_decision_representation": actual["entry_decision_representation"]}))
        _assert_equal(reference_exit, actual_exit)
    if not inference:
        actual["entry_action_q_bps"].square().sum().backward()
        for name in PORTS:
            grad = getattr(candidate, name).weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0
