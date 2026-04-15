#!/usr/bin/env python3
"""
Unit tests for the active ENTRY_V10_CTX shape contract.

This smoke reflects the current V13_BADPATH working lane:
- seq_input_dim=7
- snap_input_dim=7
- ctx_cont_dim=21
- ctx_cat_dim=6

The test is intentionally simple: prove the model loads and the active
context shapes are wired correctly without depending on any stale helper class.
"""

from __future__ import annotations

import torch
import pytest

from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)


def _make_model() -> EntryV10CtxHybridTransformer:
    return EntryV10CtxHybridTransformer(
        seq_input_dim=7,
        snap_input_dim=7,
        seq_len=30,
        ctx_cont_dim=21,
        ctx_cat_dim=6,
    )


def _make_inputs(batch_size: int = 4):
    seq_x = torch.randn(batch_size, 30, 7, dtype=torch.float32)
    snap_x = torch.randn(batch_size, 7, dtype=torch.float32)
    ctx_cat = torch.randint(0, 4, (batch_size, 6), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, 21, dtype=torch.float32)
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
    bad_ctx_cat = torch.randint(0, 4, (2, 5), dtype=torch.int64)

    with pytest.raises(RuntimeError, match="CTX_CAT_DIM_MISMATCH"):
        model(seq_x, snap_x, ctx_cat=bad_ctx_cat, ctx_cont=ctx_cont)


def test_entry_v10_ctx_fail_fast_wrong_ctx_cont_dim():
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, _ = _make_inputs(batch_size=2)
    bad_ctx_cont = torch.randn(2, 20, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="CTX_CONT_DIM_MISMATCH"):
        model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=bad_ctx_cont)


def test_entry_v10_ctx_eval_mode_is_deterministic():
    model = _make_model().eval()
    seq_x, snap_x, ctx_cat, ctx_cont = _make_inputs(batch_size=1)

    out1 = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
    out2 = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)

    for key in ("direction_logits", "path_quality", "mfe_first_n", "tradable_logit"):
        assert torch.allclose(out1[key], out2[key]), f"{key} should be deterministic in eval mode"

