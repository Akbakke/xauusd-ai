#!/usr/bin/env python3
"""
Unit tests for ENTRY_V10_CTX model shapes and contract validation.

STEP 3A: Tests for EntryV10CtxHybridTransformer and ContextEncoder.
"""

import pytest
import torch
import torch.nn as nn

from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    ContextEncoder,
    EntryV10CtxHybridTransformer,
)


def test_context_encoder_forward_pass():
    """Test ContextEncoder forward pass with valid dimensions."""
    encoder = ContextEncoder(
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
        embedding_dim=8,
    )
    
    batch_size = 4
    # Generate valid categorical values:
    # session_id: 0-3, trend_regime_id: 0-2, vol_regime_id: 0-3, atr_bucket: 0-3, spread_bucket: 0-2
    ctx_cat = torch.tensor([
        [0, 0, 0, 0, 0],  # Valid: ASIA, TREND_DOWN, LOW, LOW, LOW
        [1, 1, 1, 1, 1],  # Valid: EU, TREND_NEUTRAL, MEDIUM, MEDIUM, MEDIUM
        [2, 2, 2, 2, 2],  # Valid: US, TREND_UP, HIGH, HIGH, HIGH
        [3, 1, 3, 3, 2],  # Valid: OVERLAP, TREND_NEUTRAL, EXTREME, EXTREME, HIGH
    ], dtype=torch.int64)  # [B, 5]
    ctx_cont = torch.randn(batch_size, 2, dtype=torch.float32)  # [B, 2]
    
    # Forward pass
    ctx_emb = encoder(ctx_cat, ctx_cont)
    
    # Validate output shape
    assert ctx_emb.shape == (batch_size, 42), (
        f"Expected shape ({batch_size}, 42), got {ctx_emb.shape}"
    )
    assert ctx_emb.dtype == torch.float32, (
        f"Expected float32, got {ctx_emb.dtype}"
    )


def test_context_encoder_fail_fast_wrong_cat_dim():
    """Test ContextEncoder fails fast on wrong ctx_cat_dim."""
    # Should fail at init
    with pytest.raises(AssertionError, match="ctx_cat_dim must be 5"):
        ContextEncoder(
            ctx_cat_dim=4,  # Wrong dimension
            ctx_cont_dim=2,
            ctx_emb_dim=42,
            embedding_dim=8,
        )


def test_context_encoder_fail_fast_wrong_cont_dim():
    """Test ContextEncoder fails fast on wrong ctx_cont_dim."""
    # Should fail at init
    with pytest.raises(AssertionError, match="ctx_cont_dim must be 2"):
        ContextEncoder(
            ctx_cat_dim=5,
            ctx_cont_dim=3,  # Wrong dimension
            ctx_emb_dim=42,
            embedding_dim=8,
        )


def test_context_encoder_fail_fast_wrong_input_shape():
    """Test ContextEncoder fails fast on wrong input shape in forward."""
    encoder = ContextEncoder(
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
        embedding_dim=8,
    )
    
    batch_size = 4
    ctx_cat = torch.randint(0, 4, (batch_size, 4), dtype=torch.int64)  # Wrong: [B, 4] instead of [B, 5]
    ctx_cont = torch.randn(batch_size, 2, dtype=torch.float32)
    
    # Should fail at forward
    with pytest.raises(AssertionError, match="ctx_cat size\\(1\\) must be 5"):
        encoder(ctx_cat, ctx_cont)


def test_entry_v10_ctx_forward_pass():
    """Test EntryV10CtxHybridTransformer forward pass with valid dimensions."""
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=16,
        snap_input_dim=88,
        max_seq_len=30,
        variant="v10_ctx",
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
        ctx_embedding_dim=8,
    )
    
    batch_size = 2
    seq_len = 30
    
    seq_x = torch.randn(batch_size, seq_len, 16, dtype=torch.float32)  # [B, L, 16]
    snap_x = torch.randn(batch_size, 88, dtype=torch.float32)  # [B, 88]
    session_id = torch.randint(0, 3, (batch_size,), dtype=torch.int64)  # [B] (0-2: EU, OVERLAP, US)
    vol_regime_id = torch.randint(0, 4, (batch_size,), dtype=torch.int64)  # [B] (0-3: LOW, MEDIUM, HIGH, EXTREME)
    trend_regime_id = torch.randint(0, 3, (batch_size,), dtype=torch.int64)  # [B] (0-2: UP, DOWN, NEUTRAL)
    # Generate valid categorical values:
    # session_id: 0-3, trend_regime_id: 0-2, vol_regime_id: 0-3, atr_bucket: 0-3, spread_bucket: 0-2
    ctx_cat = torch.tensor([
        [0, 0, 0, 0, 0],  # Valid
        [1, 1, 1, 1, 1],  # Valid
    ], dtype=torch.int64)  # [B, 5]
    ctx_cont = torch.randn(batch_size, 2, dtype=torch.float32)  # [B, 2]
    
    # Forward pass
    outputs = model(
        seq_x=seq_x,
        snap_x=snap_x,
        session_id=session_id,
        vol_regime_id=vol_regime_id,
        trend_regime_id=trend_regime_id,
        ctx_cat=ctx_cat,
        ctx_cont=ctx_cont,
    )
    
    # Validate output shapes
    assert "direction_logit" in outputs, "Missing direction_logit in outputs"
    assert outputs["direction_logit"].shape == (batch_size, 1), (
        f"Expected direction_logit shape ({batch_size}, 1), got {outputs['direction_logit'].shape}"
    )
    
    assert "early_move_logit" in outputs, "Missing early_move_logit in outputs"
    assert outputs["early_move_logit"].shape == (batch_size, 1), (
        f"Expected early_move_logit shape ({batch_size}, 1), got {outputs['early_move_logit'].shape}"
    )
    
    assert "quality_score" in outputs, "Missing quality_score in outputs"
    assert outputs["quality_score"].shape == (batch_size, 1), (
        f"Expected quality_score shape ({batch_size}, 1), got {outputs['quality_score'].shape}"
    )


def test_entry_v10_ctx_fail_fast_wrong_ctx_cat_dim():
    """Test EntryV10CtxHybridTransformer fails fast on wrong ctx_cat_dim."""
    with pytest.raises(AssertionError, match="ctx_cat_dim must be 5"):
        EntryV10CtxHybridTransformer(
            ctx_cat_dim=4,  # Wrong dimension
            ctx_cont_dim=2,
            ctx_emb_dim=42,
        )


def test_entry_v10_ctx_fail_fast_wrong_ctx_cont_dim():
    """Test EntryV10CtxHybridTransformer fails fast on wrong ctx_cont_dim."""
    with pytest.raises(AssertionError, match="ctx_cont_dim must be 2"):
        EntryV10CtxHybridTransformer(
            ctx_cat_dim=5,
            ctx_cont_dim=3,  # Wrong dimension
            ctx_emb_dim=42,
        )


def test_entry_v10_ctx_fail_fast_wrong_input_shape():
    """Test EntryV10CtxHybridTransformer fails fast on wrong input shape in forward."""
    model = EntryV10CtxHybridTransformer(
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
    )
    
    batch_size = 2
    seq_len = 30
    
    seq_x = torch.randn(batch_size, seq_len, 16, dtype=torch.float32)
    snap_x = torch.randn(batch_size, 88, dtype=torch.float32)
    session_id = torch.randint(0, 3, (batch_size,), dtype=torch.int64)
    vol_regime_id = torch.randint(0, 4, (batch_size,), dtype=torch.int64)
    trend_regime_id = torch.randint(0, 3, (batch_size,), dtype=torch.int64)
    ctx_cat = torch.randint(0, 4, (batch_size, 4), dtype=torch.int64)  # Wrong: [B, 4] instead of [B, 5]
    ctx_cont = torch.randn(batch_size, 2, dtype=torch.float32)
    
    # Should fail at forward
    with pytest.raises(AssertionError, match="ctx_cat size\\(1\\) must be 5"):
        model(
            seq_x=seq_x,
            snap_x=snap_x,
            session_id=session_id,
            vol_regime_id=vol_regime_id,
            trend_regime_id=trend_regime_id,
            ctx_cat=ctx_cat,
            ctx_cont=ctx_cont,
        )


def test_backward_compatibility_old_model_unaffected():
    """Test that existing EntryV10HybridTransformer is unaffected."""
    # Import old model
    from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer
    
    # Old model should still work without ctx_cat/ctx_cont
    old_model = EntryV10HybridTransformer(
        seq_input_dim=16,
        snap_input_dim=88,
        max_seq_len=30,
        variant="v10",
    )
    
    batch_size = 2
    seq_len = 30
    
    seq_x = torch.randn(batch_size, seq_len, 16, dtype=torch.float32)
    snap_x = torch.randn(batch_size, 88, dtype=torch.float32)
    session_id = torch.randint(0, 3, (batch_size,), dtype=torch.int64)
    vol_regime_id = torch.randint(0, 4, (batch_size,), dtype=torch.int64)
    trend_regime_id = torch.randint(0, 3, (batch_size,), dtype=torch.int64)
    
    # Old model forward should work (no ctx_cat/ctx_cont)
    outputs = old_model(
        seq_x=seq_x,
        snap_x=snap_x,
        session_id=session_id,
        vol_regime_id=vol_regime_id,
        trend_regime_id=trend_regime_id,
    )
    
    # Validate output
    assert "direction_logit" in outputs
    assert outputs["direction_logit"].shape == (batch_size, 1)


def test_context_encoder_determinism():
    """Test that ContextEncoder is deterministic (no dropout in context path)."""
    encoder = ContextEncoder(
        ctx_cat_dim=5,
        ctx_cont_dim=2,
        ctx_emb_dim=42,
        embedding_dim=8,
    )
    
    encoder.eval()  # Ensure eval mode
    
    batch_size = 2
    # Generate valid categorical values:
    # session_id: 0-3, trend_regime_id: 0-2, vol_regime_id: 0-3, atr_bucket: 0-3, spread_bucket: 0-2
    ctx_cat = torch.tensor([
        [0, 0, 0, 0, 0],  # Valid
        [1, 1, 1, 1, 1],  # Valid
    ], dtype=torch.int64)  # [B, 5]
    ctx_cont = torch.randn(batch_size, 2, dtype=torch.float32)
    
    # Forward pass 1
    output1 = encoder(ctx_cat, ctx_cont)
    
    # Forward pass 2 (should be identical)
    output2 = encoder(ctx_cat, ctx_cont)
    
    # Should be identical (no dropout in context path)
    assert torch.allclose(output1, output2), "ContextEncoder should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

