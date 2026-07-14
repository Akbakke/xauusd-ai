#!/usr/bin/env python3
"""
Unit tests for the active ENTRY_V10_CTX shape contract.

This smoke reflects the active signal_bridge_v3 contract instead of a stale
lane-specific hardcode.

The test is intentionally simple: prove the model loads and the active
context shapes are wired correctly without depending on any stale helper class.
"""

from __future__ import annotations

import torch
import pytest

from gx1.contracts.signal_bridge_v3 import (
    DEFAULT_SEQ_LEN_V3,
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_SEQ_FIELDS_V3,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)

SEQ_DIM = len(ORDERED_SEQ_FIELDS_V3)
CTX_CONT_DIM = len(ORDERED_CTX_CONT_NAMES_V3)
CTX_CAT_DIM = len(ORDERED_CTX_CAT_NAMES_V3)
SEQ_LEN = DEFAULT_SEQ_LEN_V3


def _make_model() -> EntryV10CtxHybridTransformer:
    return EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_DIM,
        snap_input_dim=SEQ_DIM,
        seq_len=SEQ_LEN,
        ctx_cont_dim=CTX_CONT_DIM,
        ctx_cat_dim=CTX_CAT_DIM,
    )


def _make_inputs(batch_size: int = 4):
    seq_x = torch.randn(batch_size, SEQ_LEN, SEQ_DIM, dtype=torch.float32)
    snap_x = torch.randn(batch_size, SEQ_DIM, dtype=torch.float32)
    ctx_cat = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, CTX_CONT_DIM, dtype=torch.float32)
    return seq_x, snap_x, ctx_cat, ctx_cont


def test_entry_v10_ctx_forward_pass_v13_contract():
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont = _make_inputs(batch_size=2)

    out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)

    assert out["direction_logits"].shape == (2, 3)
    assert out["path_quality"].shape == (2, 1)
    assert out["mfe_first_n"].shape == (2, 1)
    assert out["tradable_logit"].shape == (2, 1)
    assert out["bad_path_logit"].shape == (2, 1)
    assert out["clean_edge_logit"].shape == (2, 1)
    assert out["survival_logit"].shape == (2, 1)

    for key, tensor in out.items():
        assert torch.isfinite(tensor).all(), f"{key} contains NaN/Inf"


def test_entry_v10_ctx_fail_fast_wrong_ctx_cat_dim():
    model = _make_model().eval()
    seq_x, snap_x, _, ctx_cont = _make_inputs(batch_size=2)
    bad_ctx_cat = torch.randint(0, 4, (2, CTX_CAT_DIM + 1), dtype=torch.int64)

    with pytest.raises(RuntimeError, match="CTX_CAT_DIM_MISMATCH"):
        model(seq_x, snap_x, ctx_cat=bad_ctx_cat, ctx_cont=ctx_cont)


def test_entry_v10_ctx_fail_fast_wrong_ctx_cont_dim():
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, _ = _make_inputs(batch_size=2)
    bad_ctx_cont = torch.randn(2, CTX_CONT_DIM + 1, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="CTX_CONT_DIM_MISMATCH"):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=bad_ctx_cont)


def test_entry_v10_ctx_multi_tf_requires_multi_tf_tensors():
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_DIM,
        snap_input_dim=SEQ_DIM,
        seq_len=SEQ_LEN,
        ctx_cont_dim=CTX_CONT_DIM,
        ctx_cat_dim=CTX_CAT_DIM,
        enable_multi_tf=True,
        m15_seq_dim=3,
        h1_seq_dim=3,
        h4_seq_dim=3,
        d1_seq_dim=3,
    ).eval()
    seq_x, snap_x, ctx_cat, ctx_cont = _make_inputs(batch_size=2)

    with pytest.raises(RuntimeError, match="MULTI_TF_INPUTS_MISSING"):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)


def test_entry_v10_ctx_eval_mode_is_deterministic():
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont = _make_inputs(batch_size=1)

    out1 = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
    out2 = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)

    for key in ("direction_logits", "path_quality", "mfe_first_n", "tradable_logit"):
        assert torch.allclose(out1[key], out2[key]), f"{key} should be deterministic in eval mode"


def test_entry_v10_ctx_direction_repair_heads_shape_contract():
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_DIM,
        snap_input_dim=SEQ_DIM,
        seq_len=SEQ_LEN,
        ctx_cont_dim=CTX_CONT_DIM,
        ctx_cat_dim=CTX_CAT_DIM,
        enable_anchor_gate=True,
        enable_hierarchical_entry_heads=True,
        enable_side_validity_head=True,
        enable_trendline_rail_head=True,
    ).eval()
    seq_x, snap_x, ctx_cat, ctx_cont = _make_inputs(batch_size=3)

    out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)

    assert out["anchor_gate"].shape == (3, 3)
    assert out["trade_logit"].shape == (3, 1)
    assert out["side_logits"].shape == (3, 2)
    assert out["side_utility"].shape == (3, 2)
    assert out["side_bad_path_logit"].shape == (3, 2)
    assert out["side_mae"].shape == (3, 2)
    assert out["side_validity_logit"].shape == (3, 2)
    assert out["trendline_rail_logits"].shape == (3, 4)
    assert torch.all((out["anchor_gate"] >= 0.0) & (out["anchor_gate"] <= 1.0))
    for key in (
        "trade_logit",
        "side_logits",
        "side_utility",
        "side_bad_path_logit",
        "side_mae",
        "side_validity_logit",
        "trendline_rail_logits",
    ):
        assert torch.isfinite(out[key]).all(), f"{key} contains NaN/Inf"


def test_entry_v10_ctx_side_validity_requires_hierarchy():
    with pytest.raises(RuntimeError, match="SIDE_VALIDITY_HEAD_REQUIRES_HIERARCHICAL_ENTRY_HEADS"):
        EntryV10CtxHybridTransformer(
            seq_input_dim=SEQ_DIM,
            snap_input_dim=SEQ_DIM,
            seq_len=SEQ_LEN,
            ctx_cont_dim=CTX_CONT_DIM,
            ctx_cat_dim=CTX_CAT_DIM,
            enable_side_validity_head=True,
        )


def test_entry_v10_ctx_trendline_rail_can_emit_early_failure_pockets():
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=SEQ_DIM,
        snap_input_dim=SEQ_DIM,
        seq_len=SEQ_LEN,
        ctx_cont_dim=CTX_CONT_DIM,
        ctx_cat_dim=CTX_CAT_DIM,
        enable_trendline_rail_head=True,
        trendline_rail_output_dim=6,
    ).eval()
    seq_x, snap_x, ctx_cat, ctx_cont = _make_inputs(batch_size=3)

    out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)

    assert out["trendline_rail_logits"].shape == (3, 6)
