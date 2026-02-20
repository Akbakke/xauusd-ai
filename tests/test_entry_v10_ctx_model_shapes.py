#!/usr/bin/env python3
"""
Unit tests for ENTRY_V10_CTX model shapes and contract validation.

ONE UNIVERSE: 6/6 ctx, 7/7 signal (signal_bridge_v1). No legacy 16/88 or 5/2.
"""

import pytest
import torch
import torch.nn as nn

from gx1.contracts.signal_bridge_v1 import SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    ContextEncoder,
    EntryV10CtxHybridTransformer,
)

CTX_CAT_DIM = 6
CTX_CONT_DIM = 6


def test_context_encoder_forward_pass():
    """Test ContextEncoder forward pass with 6/6 dimensions."""
    encoder = ContextEncoder()
    batch_size = 4
    ctx_cat = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, CTX_CONT_DIM, dtype=torch.float32)
    ctx_emb = encoder(ctx_cat, ctx_cont)
    assert ctx_emb.shape[0] == batch_size
    assert ctx_emb.dtype == torch.float32


def test_context_encoder_fail_fast_wrong_cat_dim():
    """Test ContextEncoder fails fast on wrong ctx_cat last dim."""
    encoder = ContextEncoder()
    batch_size = 2
    ctx_cat = torch.randint(0, 4, (batch_size, 4), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, CTX_CONT_DIM, dtype=torch.float32)
    with pytest.raises(ValueError, match="CTX_CAT_DIM_MISMATCH"):
        encoder(ctx_cat, ctx_cont)


def test_context_encoder_fail_fast_wrong_cont_dim():
    """Test ContextEncoder fails fast on wrong ctx_cont last dim."""
    encoder = ContextEncoder()
    batch_size = 2
    ctx_cat = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="CTX_CONT_DIM_MISMATCH"):
        encoder(ctx_cat, ctx_cont)


def test_entry_v10_ctx_forward_pass():
    """Test EntryV10CtxHybridTransformer forward pass: 7/7 signal, 6/6 ctx, ctx-only signature."""
    model = EntryV10CtxHybridTransformer(max_seq_len=30)
    batch_size = 2
    seq_len = 30
    seq_x = torch.randn(batch_size, seq_len, int(SEQ_SIGNAL_DIM), dtype=torch.float32)
    snap_x = torch.randn(batch_size, int(SNAP_SIGNAL_DIM), dtype=torch.float32)
    ctx_cat = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, CTX_CONT_DIM, dtype=torch.float32)
    outputs = model(
        seq_x=seq_x,
        snap_x=snap_x,
        ctx_cat=ctx_cat,
        ctx_cont=ctx_cont,
    )
    assert "direction_logit" in outputs
    assert outputs["direction_logit"].shape == (batch_size, 1)
    assert "early_move_logit" in outputs
    assert outputs["early_move_logit"].shape == (batch_size, 1)
    assert "quality_score" in outputs
    assert outputs["quality_score"].shape == (batch_size, 1)


def test_entry_v10_ctx_fail_fast_wrong_ctx_cat_dim():
    """Test EntryV10CtxHybridTransformer fails fast on wrong ctx_cat last dim."""
    model = EntryV10CtxHybridTransformer(max_seq_len=30)
    batch_size = 2
    seq_x = torch.randn(batch_size, 30, int(SEQ_SIGNAL_DIM), dtype=torch.float32)
    snap_x = torch.randn(batch_size, int(SNAP_SIGNAL_DIM), dtype=torch.float32)
    ctx_cat = torch.randint(0, 4, (batch_size, 4), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, CTX_CONT_DIM, dtype=torch.float32)
    with pytest.raises(ValueError, match="CTX_CAT_DIM_MISMATCH"):
        model(
            seq_x=seq_x,
            snap_x=snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )


def test_entry_v10_ctx_fail_fast_wrong_ctx_cont_dim():
    """Test EntryV10CtxHybridTransformer fails fast on wrong ctx_cont last dim."""
    model = EntryV10CtxHybridTransformer(max_seq_len=30)
    batch_size = 2
    seq_x = torch.randn(batch_size, 30, int(SEQ_SIGNAL_DIM), dtype=torch.float32)
    snap_x = torch.randn(batch_size, int(SNAP_SIGNAL_DIM), dtype=torch.float32)
    ctx_cat = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, 2, dtype=torch.float32)
    with pytest.raises(ValueError, match="CTX_CONT_DIM_MISMATCH"):
        model(
            seq_x=seq_x,
            snap_x=snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )


def test_entry_v10_ctx_fail_fast_wrong_seq_dim():
    """Test EntryV10CtxHybridTransformer fails fast on wrong seq_x last dim (must be SEQ_SIGNAL_DIM=7)."""
    model = EntryV10CtxHybridTransformer(max_seq_len=30)
    batch_size = 2
    seq_x = torch.randn(batch_size, 30, 5, dtype=torch.float32)  # wrong dim (not 7)
    snap_x = torch.randn(batch_size, int(SNAP_SIGNAL_DIM), dtype=torch.float32)
    ctx_cat = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, CTX_CONT_DIM, dtype=torch.float32)
    with pytest.raises(ValueError, match="SEQ_X"):
        model(
            seq_x=seq_x,
            snap_x=snap_x,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )


def test_context_encoder_determinism():
    """Test that ContextEncoder is deterministic (no dropout in context path)."""
    encoder = ContextEncoder()
    encoder.eval()
    batch_size = 2
    ctx_cat = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
    ctx_cont = torch.randn(batch_size, CTX_CONT_DIM, dtype=torch.float32)
    output1 = encoder(ctx_cat, ctx_cont)
    output2 = encoder(ctx_cat, ctx_cont)
    assert torch.allclose(output1, output2), "ContextEncoder should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
