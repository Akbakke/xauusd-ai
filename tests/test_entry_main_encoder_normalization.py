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
        # The original six-layer fuse has no final normalization or extra RNG draw.
        original.fuse = original.fuse[:-1]
    assert torch.equal(rng, torch.get_rng_state())
    _assert_equal(online.state_dict(), original.state_dict())
    teacher = trainer._copy_frozen_prefix_reference_model(online)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    assert online.encoder.norm is not None and teacher.encoder.norm is None
    assert len(online.fuse) == 7 and len(teacher.fuse) == len(original.fuse) == 6
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


def test_entry_fuse_bounds_each_row_without_mixing_inputs_or_losing_gradients():
    torch.manual_seed(919)
    model = _make_model(dropout=0.0).train()
    fuse = model.fuse
    d = fuse[0].out_features
    assert fuse[-1].state_dict() == {} and not fuse[-1].elementwise_affine
    x = (torch.randn(2, 3*d) + 100*torch.randn(1, 3*d)).requires_grad_()
    out = fuse(x)
    assert torch.all(torch.linalg.vector_norm(out, dim=-1) <= math.sqrt(d) + 1e-5)
    changed = x.detach().clone(); changed[1] *= 1000
    torch.testing.assert_close(out[0], fuse(changed)[0], atol=0, rtol=0)
    (out*torch.randn_like(out)).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    assert all(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in fuse.parameters())
    model.fuse[-1] = torch.nn.LayerNorm(d)
    with pytest.raises(RuntimeError, match='REFERENCE_FUSE_FUNCTION_INVALID'):
        trainer._copy_frozen_prefix_reference_model(model)



def _v38_design():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json"
    return json.loads(path.read_text())


def test_frozen_v38_design_requires_current_function_and_legacy_stays_explicit():
    assert trainer._prefix_reference_model_functions(_v38_design()) == trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
    legacy = {"schema_version": "gx1_frozen_chronological_learning_design_v1",
              "calendar": {}, "initialization": {}}
    assert trainer._prefix_reference_model_functions(legacy) == trainer._PREFIX_MODEL_FUNCTIONS


@pytest.mark.parametrize("fault", ["same_source", "missing_namespace", "target", "old_init", "partial_current", "schema", "calendar_type"])
def test_current_teacher_design_rejects_ambiguous_or_mismatched_function(fault):
    design = _v38_design()
    if fault == "same_source":
        design["calendar"]["physical_source_splits"]["control"] = "train"
    elif fault == "missing_namespace":
        design["calendar"].pop("physical_coordinate_namespaces_are_separate")
    elif fault == "target":
        design["initialization"]["target"] = "legacy teacher"
    elif fault == "old_init":
        design["initialization"]["mode"] = "reuse_old_weights"
    elif fault == "partial_current":
        design["calendar"].pop("physical_source_splits")
        design["calendar"].pop("physical_coordinate_namespaces_are_separate")
    elif fault == "schema":
        design["schema_version"] = "unbound"
    else:
        design["calendar"] = None
    with pytest.raises(RuntimeError, match="REFERENCE_.*DESIGN"):
        trainer._prefix_reference_model_functions(design)


@pytest.mark.parametrize("inference", [False, True])
def test_current_teacher_preserves_entry_exit_function_weights_and_rng(inference):
    torch.manual_seed(20260911)
    online = _make_model(dropout=0.0).eval()
    before = trainer._model_state_sha256(online)
    rng = torch.get_rng_state().clone()
    functions = trainer._prefix_reference_model_functions(_v38_design())
    teacher = trainer._copy_frozen_prefix_reference_model(online, model_functions=functions)
    assert torch.equal(rng, torch.get_rng_state())
    assert trainer._model_state_sha256(teacher) == before == trainer._model_state_sha256(online)
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    assert teacher.encoder.norm is not None and teacher.encoder.norm is not online.encoder.norm
    assert len(teacher.fuse) == len(online.fuse) == 7
    assert all(p.data_ptr() != q.data_ptr() for p, q in zip(online.parameters(), teacher.parameters()))
    # Compare the same frozen execution mode. Merely changing requires_grad
    # can change CPU floating-point dispatch even inside no_grad/inference_mode.
    # The separate native train/serve gate must measure its own execution modes.
    online.requires_grad_(False)
    seq, snap, cat, ctx, mtf = _make_inputs(2)
    exits = _exit_inputs(2)
    with torch.inference_mode() if inference else torch.no_grad():
        expected = online(seq, snap, ctx_cat=cat, ctx_cont=ctx, **mtf)
        actual = teacher(seq, snap, ctx_cat=cat, ctx_cont=ctx, **mtf)
        _assert_equal(expected, actual)
        expected_exit = online.forward_exit_random_access_batch(
            **(exits | {"entry_decision_representation": expected["entry_decision_representation"]}))
        actual_exit = teacher.forward_exit_random_access_batch(
            **(exits | {"entry_decision_representation": actual["entry_decision_representation"]}))
        _assert_equal(expected_exit, actual_exit)


@pytest.mark.parametrize("functions", [{}, {"online": "current", "target": "current"},
                                        {**trainer._PREFIX_CURRENT_MODEL_FUNCTIONS, "extra": True}])
def test_teacher_copy_rejects_unknown_function_identity(functions):
    online = _make_model(dropout=0.0)
    before = trainer._model_state_sha256(online)
    with pytest.raises(RuntimeError, match="MODEL_FUNCTIONS_INVALID"):
        trainer._copy_frozen_prefix_reference_model(online, model_functions=functions)
    assert trainer._model_state_sha256(online) == before


def test_structure_only_validation_never_allocates_a_model_copy(monkeypatch):
    model = _make_model(dropout=0.0)
    before = trainer._model_state_sha256(model)
    def forbidden(*args, **kwargs):
        raise AssertionError("structure check must not copy model")
    monkeypatch.setattr(trainer.copy, "deepcopy", forbidden)
    trainer._require_prefix_online_function(model)
    assert trainer._model_state_sha256(model) == before
