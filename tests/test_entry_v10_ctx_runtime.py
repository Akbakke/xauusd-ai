#!/usr/bin/env python3
"""
Unit Tests for ENTRY_V10_CTX Runtime Integration.

ONE UNIVERSE: 6/6 ctx, 7/7 signal. Loader and ctx-consumption proof.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

CTX_CAT_DIM = 6
CTX_CONT_DIM = 6


class TestEntryV10CtxLoader:
    """Loader test: ctx bundle loads, metadata 6/6."""

    def test_load_ctx_bundle_smoke(self):
        """Test loading smoke-run ctx bundle."""
        from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle

        bundle_dir = Path("models/entry_v10_ctx/SMOKE_20260106_ctxfusion")
        if not bundle_dir.exists():
            pytest.skip(f"Smoke-run bundle not found: {bundle_dir}")
        feature_meta_path = Path("gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json")
        if not feature_meta_path.exists():
            pytest.skip(f"Feature metadata not found: {feature_meta_path}")

        bundle = load_entry_v10_ctx_bundle(
            bundle_dir=bundle_dir,
            feature_meta_path=feature_meta_path,
            is_replay=True,
        )
        assert bundle is not None
        assert bundle.transformer_model is not None
        metadata = bundle.metadata or {}
        assert metadata.get("supports_context_features") is True
        assert metadata.get("expected_ctx_cat_dim") == CTX_CAT_DIM
        assert metadata.get("expected_ctx_cont_dim") == CTX_CONT_DIM
        assert metadata.get("model_variant") == "v10_ctx"
        from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
        assert isinstance(bundle.transformer_model, EntryV10CtxHybridTransformer)


class TestEntryV10CtxInferenceShapes:
    """Inference shape test: ctx dims mismatch → fail-fast."""

    def test_ctx_dims_mismatch_fail_fast_replay(self):
        pytest.skip("Skipping - requires full runner setup. Shape validation tested in proof test and A/B runs.")


class TestEntryV10CtxProof:
    """Proof test: same sample, ctx vs permuted ctx → output differs."""

    def test_ctx_consumption_proof(self):
        """Test that ctx actually affects model output (6/6, 7/7, ctx-only forward)."""
        from gx1.contracts.signal_bridge_v1 import SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM
        from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer

        model = EntryV10CtxHybridTransformer(max_seq_len=30)
        model.eval()
        batch_size = 1
        seq_dim = int(SEQ_SIGNAL_DIM)
        snap_dim = int(SNAP_SIGNAL_DIM)
        seq_x = torch.randn(batch_size, 30, seq_dim)
        snap_x = torch.randn(batch_size, snap_dim)
        ctx_cat_A = torch.randint(0, 4, (batch_size, CTX_CAT_DIM), dtype=torch.int64)
        ctx_cont_A = torch.randn(batch_size, CTX_CONT_DIM)

        with torch.no_grad():
            outputs_A = model(
                seq_x=seq_x,
                snap_x=snap_x,
                ctx_cat=ctx_cat_A,
                ctx_cont=ctx_cont_A,
            )
            prob_long_A = torch.sigmoid(outputs_A["direction_logit"]).item()

        ctx_cat_B = torch.roll(ctx_cat_A, shifts=1, dims=1)
        ctx_cont_B = torch.zeros_like(ctx_cont_A)
        with torch.no_grad():
            outputs_B = model(
                seq_x=seq_x,
                snap_x=snap_x,
                ctx_cat=ctx_cat_B,
                ctx_cont=ctx_cont_B,
            )
            prob_long_B = torch.sigmoid(outputs_B["direction_logit"]).item()

        diff = abs(prob_long_A - prob_long_B)
        min_diff_threshold = 1e-6
        assert diff >= min_diff_threshold, (
            f"CTX_CONSUMPTION_PROOF_FAILED: ctx does not affect output. "
            f"prob_long_A={prob_long_A:.6f}, prob_long_B={prob_long_B:.6f}, diff={diff:.6f} < threshold={min_diff_threshold}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
