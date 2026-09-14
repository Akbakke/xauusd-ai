from __future__ import annotations

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from tests.model_native_input_normalization_support import input_normalization_fixture
from tests.test_entry_v10_ctx_model_shapes import (
    SEQ_LEN, _make_inputs, _make_model, _specialist_indices,
)


def _model_and_inputs():
    torch.manual_seed(20260914)
    model = _make_model(dropout=0.0).eval()
    # Production checkpoints have trained corrections. Neutral initialization
    # zeros these two paths and would hide a connected router gradient.
    with torch.no_grad():
        model.cross_tf_out.weight.normal_(std=.01)
        model.family_tf_cooperation_out.weight.normal_(std=.01)
    return model, _make_inputs()


def _forward(model, inputs, **kwargs):
    seq, snap, categorical, continuous, mtf = inputs
    return model(seq, snap, ctx_cat=categorical, ctx_cont=continuous, **mtf, **kwargs)


def _nonzero(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(parameter.grad) > 0


def _entry_parameters(model):
    return (
        model.encoder.layers[0].self_attn.in_proj_weight,
        model.family_tf_context_gate.weight,
        model.family_tf_token_gate.weight,
        next(model.mtf_family_encoder.parameters()),
    )


def test_v4_forward_is_exact_with_all_configured_production_width_inputs():
    torch.manual_seed(20260914)
    signal_width = MODEL_NATIVE_SIGNAL_DIM
    tf_width = len(MULTI_TF_PER_BAR_FEATURES_V4)
    normalization = input_normalization_fixture(
        signal_names=[f"signal_{index}" for index in range(signal_width)],
        mtf_names=[f"mtf_{index}" for index in range(tf_width)],
    )
    model = _make_model(
        dropout=0.0, seq_input_dim=signal_width, snap_input_dim=signal_width,
        specialist_input_indices=_specialist_indices(signal_width),
        input_normalization=normalization,
        **{f"{name}_seq_dim": tf_width for name in ("m5", "m15", "h1", "h4", "d1")},
    ).eval()
    _, _, categorical, continuous, mtf_template = _make_inputs(batch_size=1)
    seq = torch.randn(1, SEQ_LEN, signal_width)
    inputs = (seq, seq[:, -1].clone(), categorical, continuous,
              {name: torch.randn(1, SEQ_LEN, tf_width) for name in mtf_template})
    before = canonical_model_state_sha256(model.state_dict())
    with torch.no_grad():
        legacy = _forward(model, inputs)
        explicit_legacy = _forward(model, inputs, liquidation_relative_values=False)
        relative = _forward(model, inputs, liquidation_relative_values=True)
    assert legacy.keys() == explicit_legacy.keys() == relative.keys()
    for name in legacy:
        assert torch.equal(legacy[name], explicit_legacy[name]), name
        assert torch.equal(legacy[name], relative[name]), name
    assert canonical_model_state_sha256(model.state_dict()) == before
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert inputs[0].shape[-1] == signal_width
    assert all(value.shape[-1] == tf_width for value in inputs[-1].values())


@pytest.mark.parametrize("relative", [False, True])
def test_q_teacher_trains_mixer_and_head_but_v4_blocks_entry_representation(relative):
    model, inputs = _model_and_inputs()
    output = _forward(model, inputs, liquidation_relative_values=relative)
    output["entry_action_q_bps"].square().sum().backward()
    for parameter in (model.head_entry_action_q.weight, model.entry_q_joint_in.weight,
                      model.entry_q_joint_norm.weight):
        _nonzero(parameter)
    for parameter in _entry_parameters(model):
        if relative:
            assert parameter.grad is None
        else:
            _nonzero(parameter)
    assert model.head_forecast.weight.grad is None


@pytest.mark.parametrize("relative", [False, True])
def test_exit_token_feedback_preserves_projection_learning_only_upstream_is_detached(relative):
    model, inputs = _model_and_inputs()
    output = _forward(model, inputs, liquidation_relative_values=relative)
    token = output["entry_decision_representation"]
    # Exactly the scalar VJP injection used by train_epoch after Exit's
    # separately streamed backward, not a detached-token surrogate.
    exit_entry_gradient = torch.randn_like(token)
    (token * exit_entry_gradient).sum().backward()
    _nonzero(model.entry_decision_token[1].weight)
    upstream = (*_entry_parameters(model), model.head_entry_action_q.weight,
                model.entry_q_joint_in.weight, model.entry_q_joint_norm.weight)
    for parameter in upstream:
        if relative:
            assert parameter.grad is None
        else:
            _nonzero(parameter)
    assert model.head_forecast.weight.grad is None


@pytest.mark.parametrize("output_name,head_name", [
    ("forecast_pred", "head_forecast"), ("side_mae_bps", "head_side_mae"),
])
def test_observed_market_head_still_trains_v4_entry_m5_and_mtf_routes(output_name, head_name):
    model, inputs = _model_and_inputs()
    output = _forward(model, inputs, liquidation_relative_values=True)
    observed_target = torch.randn_like(output[output_name])
    torch.nn.functional.l1_loss(output[output_name], observed_target).backward()
    _nonzero(getattr(model, head_name).weight)
    for parameter in _entry_parameters(model):
        _nonzero(parameter)
    assert model.head_entry_action_q.weight.grad is None
    assert model.entry_q_joint_in.weight.grad is None
    assert model.entry_decision_token[1].weight.grad is None


@pytest.mark.parametrize("invalid", [1, "v4", None])
def test_gradient_mode_must_be_explicit_boolean(invalid):
    model, inputs = _model_and_inputs()
    with pytest.raises(RuntimeError, match="ENTRY_LIQUIDATION_GRADIENT_MODE_INVALID"):
        _forward(model, inputs, liquidation_relative_values=invalid)
