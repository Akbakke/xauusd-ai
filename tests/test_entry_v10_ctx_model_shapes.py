#!/usr/bin/env python3
"""Shape and gradient proofs for the one exact ENTRY model-native architecture."""

from __future__ import annotations

import inspect

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import INPUTS
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EXACT_SPECIALIST_NAMES,
    EntryV10CtxHybridTransformer,
)


SEQ_DIM = 16
SEQ_LEN = 4
TF_DIM = 3


def _specialist_indices() -> dict[str, list[int]]:
    return {name: [index] for index, name in enumerate(EXACT_SPECIALIST_NAMES)}


def _make_model(**overrides) -> EntryV10CtxHybridTransformer:
    kwargs = {
        "seq_input_dim": SEQ_DIM,
        "snap_input_dim": SEQ_DIM,
        "seq_len": SEQ_LEN,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "m5_seq_dim": TF_DIM,
        "m15_seq_dim": TF_DIM,
        "h1_seq_dim": TF_DIM,
        "h4_seq_dim": TF_DIM,
        "d1_seq_dim": TF_DIM,
        "m5_seq_len": SEQ_LEN,
        "m15_seq_len": SEQ_LEN,
        "h1_seq_len": SEQ_LEN,
        "h4_seq_len": SEQ_LEN,
        "d1_seq_len": SEQ_LEN,
        "specialist_input_indices": _specialist_indices(),
    }
    kwargs.update(overrides)
    return EntryV10CtxHybridTransformer(**kwargs)


def _make_inputs(batch_size: int = 2) -> tuple:
    return (
        torch.randn(batch_size, SEQ_LEN, SEQ_DIM),
        torch.randn(batch_size, SEQ_DIM),
        torch.randint(0, 4, (batch_size, MODEL_NATIVE_CTX_CAT_DIM)),
        torch.randn(batch_size, MODEL_NATIVE_CTX_CONT_DIM),
        {
            f"seq_{tf}": torch.randn(batch_size, SEQ_LEN, TF_DIM)
            for tf in ("m5", "m15", "h1", "h4", "d1")
        },
    )


def _forward(model: EntryV10CtxHybridTransformer, batch_size: int = 2) -> dict:
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size)
    return model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


def test_exact_architecture_emits_every_mandatory_head_with_exact_width() -> None:
    model = _make_model().eval()
    out = _forward(model, batch_size=2)
    widths = {
        "direction_logits": 3,
        "raw_direction_logits": 3,
        "model_native_logits": 3,
        "mtf_dir_logits": 3,
        "path_quality_raw": 1,
        "path_quality": 1,
        "mfe_first_n": 1,
        "tradable_logit": 1,
        "bad_path_logit_raw": 1,
        "bad_path_logit": 1,
        "clean_edge_logit": 1,
        "survival_logit": 1,
        "specialist_gate": 8,
        "trade_logit": 1,
        "side_logits": 2,
        "side_utility": 2,
        "side_bad_path_logit": 2,
        "side_mae": 2,
        "side_validity_logit": 2,
        "trendline_rail_logits": 6,
        "tf_agreement_logit": 1,
        "path_quality_log_var": 1,
        "position_size_logit": 1,
        "dip_pred": 18,
        "forecast_pred": 4,
        "timing_pred": 12,
        "tail_risk_pred": 6,
        "vol_forecast_pred": 3,
        "public_trade_flat_decision_logits": 2,
    }
    for name, width in widths.items():
        assert out[name].shape == (2, width), name
        assert torch.isfinite(out[name]).all(), name


def test_public_trade_flat_decision_is_post_calibration_argmax_ssot() -> None:
    model = _make_model().eval()
    model.set_direction_calibration(
        2.0,
        torch.tensor([1.25, -0.75, 0.40], dtype=torch.float32),
    )
    out = _forward(model, batch_size=4)
    final_logits = out["direction_logits"]
    expected_final = out["raw_direction_logits"] / 2.0 + torch.tensor(
        [1.25, -0.75, 0.40], dtype=final_logits.dtype
    )
    expected_pair = torch.stack(
        (final_logits[:, :2].max(dim=1).values, final_logits[:, 2]), dim=1
    )
    assert torch.allclose(final_logits, expected_final, atol=1e-6)
    assert torch.allclose(out["public_trade_flat_decision_logits"], expected_pair, atol=1e-6)
    assert torch.equal(
        out["public_trade_flat_decision_logits"].argmax(dim=1) == 0,
        final_logits.argmax(dim=1) != 2,
    )


def test_public_direction_gradient_reaches_every_fused_evidence_head() -> None:
    model = _make_model().train()
    out = _forward(model, batch_size=4)
    loss = torch.nn.functional.cross_entropy(
        out["direction_logits"],
        torch.tensor([0, 1, 2, 0]),
    )
    loss.backward()
    for parameter in (
        model.head_direction.weight,
        model.head_mtf_direction.weight,
        model.head_path_quality.weight,
        model.head_path_quality_log_var.weight,
        model.head_mfe_first_n.weight,
        model.head_tradable.weight,
        model.head_bad_path.weight,
        model.head_clean_edge.weight,
        model.head_survival.weight,
        model.head_trade.weight,
        model.head_side.weight,
        model.head_side_utility.weight,
        model.head_side_bad_path.weight,
        model.head_side_mae.weight,
        model.head_side_validity.weight,
        model.head_trendline_rail.weight,
        model.head_tf_agreement.weight,
        model.head_position_size.weight,
        model.head_dip.weight,
        model.head_forecast.weight,
        model.head_timing.weight,
        model.head_tail_risk.weight,
        model.head_vol_forecast.weight,
        model.evidence_fusion_in.weight,
        model.evidence_fusion_out.weight,
        model.regime_film[-1].weight,
        model.cross_tf_out.weight,
        model.specialist_out.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_each_of_23_evidence_groups_changes_a_direction_class_margin() -> None:
    torch.manual_seed(90210)
    model = _make_model().eval()
    with torch.no_grad():
        out = _forward(model, batch_size=3)
        evidence = {name: out[name] for name, _ in INPUTS}
        baseline = model._fuse_direction_evidence(evidence)
        baseline_centered = baseline - baseline.mean(dim=1, keepdim=True)
        assert len(INPUTS) == 23
        for name, _ in INPUTS:
            ablated = dict(evidence)
            ablated[name] = torch.zeros_like(evidence[name])
            changed = model._fuse_direction_evidence(ablated)
            changed_centered = changed - changed.mean(dim=1, keepdim=True)
            assert not torch.allclose(
                baseline_centered,
                changed_centered,
                atol=1e-9,
                rtol=1e-7,
            ), name


def test_report_only_path_calibration_cannot_change_direction_fusion() -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size=3)
    with torch.no_grad():
        before = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
        model.set_path_calibration(1.7, 0.4, 2.3, -0.6)
        after = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
    assert torch.equal(before["path_quality_raw"], after["path_quality_raw"])
    assert torch.equal(before["bad_path_logit_raw"], after["bad_path_logit_raw"])
    assert torch.equal(before["raw_direction_logits"], after["raw_direction_logits"])
    assert torch.equal(before["direction_logits"], after["direction_logits"])
    assert not torch.equal(before["path_quality"], after["path_quality"])
    assert not torch.equal(before["bad_path_logit"], after["bad_path_logit"])


@pytest.mark.parametrize(
    "scale_name",
    (
        "multi_tf_scale",
        "specialist_fusion_scale",
        "tf_input_scale_init_m5",
        "tf_input_scale_init_m15",
        "tf_input_scale_init_h1",
        "tf_input_scale_init_h4",
        "tf_input_scale_init_d1",
    ),
)
def test_exact_architecture_rejects_zero_representation_scale(scale_name: str) -> None:
    with pytest.raises(RuntimeError, match="MANDATORY_REPRESENTATION_SCALE_INVALID"):
        _make_model(**{scale_name: 0.0})


def test_exact_architecture_requires_all_five_tf_inputs() -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs()
    del mtf["seq_h4"]
    with pytest.raises(TypeError, match="seq_h4"):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


@pytest.mark.parametrize(
    "bad_key",
    ("ctx_cat", "ctx_cont"),
)
def test_exact_architecture_rejects_wrong_context_width(bad_key: str) -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs()
    if bad_key == "ctx_cat":
        ctx_cat = torch.zeros(2, MODEL_NATIVE_CTX_CAT_DIM + 1, dtype=torch.long)
        pattern = "CTX_CAT_DIM_MISMATCH"
    else:
        ctx_cont = torch.zeros(2, MODEL_NATIVE_CTX_CONT_DIM + 1)
        pattern = "CTX_CONT_DIM_MISMATCH"
    with pytest.raises(RuntimeError, match=pattern):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)


def test_exact_architecture_eval_is_deterministic() -> None:
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont, mtf = _make_inputs(batch_size=1)
    first = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
    second = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf)
    for key in first:
        assert torch.allclose(first[key], second[key]), key


def test_model_and_config_expose_no_architecture_disable_switches() -> None:
    model_parameters = inspect.signature(EntryV10CtxHybridTransformer).parameters
    assert not [name for name in model_parameters if name.startswith("enable_")]
    config_fields = EntryV10CtxHybridTransformer.__init__.__annotations__
    assert not [name for name in config_fields if name.startswith("enable_")]
