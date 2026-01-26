#!/usr/bin/env python3
"""
Unit tests for Entry Context Features (STEP 2).

Tests:
- Contract validation (missing fields)
- Tensor shape validation
- Mapping/bucketing determinism
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from gx1.execution.entry_context_features import (
    EntryContextFeatures,
    build_entry_context_features,
)


def test_entry_context_features_validation():
    """Test EntryContextFeatures validation."""
    # Valid context features
    ctx = EntryContextFeatures(
        session_id=1,
        trend_regime_id=0,
        vol_regime_id=1,
        atr_bucket=1,
        spread_bucket=0,
        atr_bps=50.0,
        spread_bps=10.0,
    )
    
    valid, error = ctx.validate(is_replay=True)
    assert valid, f"Valid context features failed validation: {error}"
    
    # Invalid: session_id out of range
    ctx_invalid = EntryContextFeatures(
        session_id=5,  # Out of range [0-3]
        trend_regime_id=0,
        vol_regime_id=1,
        atr_bucket=1,
        spread_bucket=0,
        atr_bps=50.0,
        spread_bps=10.0,
    )
    
    valid, error = ctx_invalid.validate(is_replay=True)
    assert not valid, "Invalid session_id should fail validation"
    assert "session_id out of range" in error
    
    # Invalid: atr_bps not finite
    ctx_nan = EntryContextFeatures(
        session_id=1,
        trend_regime_id=0,
        vol_regime_id=1,
        atr_bucket=1,
        spread_bucket=0,
        atr_bps=float('nan'),
        spread_bps=10.0,
    )
    
    valid, error = ctx_nan.validate(is_replay=True)
    assert not valid, "NaN atr_bps should fail validation"
    assert "atr_bps is not finite" in error


def test_entry_context_features_tensor_conversion():
    """Test tensor conversion methods."""
    ctx = EntryContextFeatures(
        session_id=1,
        trend_regime_id=0,
        vol_regime_id=1,
        atr_bucket=1,
        spread_bucket=0,
        atr_bps=50.0,
        spread_bps=10.0,
    )
    
    # Test categorical tensor
    ctx_cat = ctx.to_tensor_categorical()
    assert ctx_cat.dtype == np.int64, f"Expected int64, got {ctx_cat.dtype}"
    assert ctx_cat.shape == (5,), f"Expected shape (5,), got {ctx_cat.shape}"
    assert ctx_cat[0] == 1, "session_id should be 1"
    assert ctx_cat[1] == 0, "trend_regime_id should be 0"
    assert ctx_cat[2] == 1, "vol_regime_id should be 1"
    assert ctx_cat[3] == 1, "atr_bucket should be 1"
    assert ctx_cat[4] == 0, "spread_bucket should be 0"
    
    # Test continuous tensor
    ctx_cont = ctx.to_tensor_continuous()
    assert ctx_cont.dtype == np.float32, f"Expected float32, got {ctx_cont.dtype}"
    assert ctx_cont.shape == (2,), f"Expected shape (2,), got {ctx_cont.shape}"
    assert ctx_cont[0] == 50.0, "atr_bps should be 50.0"
    assert ctx_cont[1] == 10.0, "spread_bps should be 10.0"


def test_build_entry_context_features_determinism():
    """Test that build_entry_context_features is deterministic."""
    # Create test candles
    dates = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
    candles = pd.DataFrame({
        "open": 2650.0,
        "high": 2651.0,
        "low": 2649.0,
        "close": 2650.5,
        "bid_close": 2650.4,
        "ask_close": 2650.6,
    }, index=dates)
    
    policy_state = {"session": "EU"}
    
    # Build context features twice
    ctx1 = build_entry_context_features(
        candles=candles,
        policy_state=policy_state.copy(),
        atr_proxy=0.5,  # 50 bps at price 2650
        spread_bps=10.0,
        is_replay=True,
    )
    
    ctx2 = build_entry_context_features(
        candles=candles,
        policy_state=policy_state.copy(),
        atr_proxy=0.5,
        spread_bps=10.0,
        is_replay=True,
    )
    
    # Should be identical
    assert ctx1.session_id == ctx2.session_id, "session_id should be deterministic"
    assert ctx1.trend_regime_id == ctx2.trend_regime_id, "trend_regime_id should be deterministic"
    assert ctx1.vol_regime_id == ctx2.vol_regime_id, "vol_regime_id should be deterministic"
    assert ctx1.atr_bucket == ctx2.atr_bucket, "atr_bucket should be deterministic"
    assert ctx1.spread_bucket == ctx2.spread_bucket, "spread_bucket should be deterministic"
    assert abs(ctx1.atr_bps - ctx2.atr_bps) < 1e-6, "atr_bps should be deterministic"
    assert abs(ctx1.spread_bps - ctx2.spread_bps) < 1e-6, "spread_bps should be deterministic"


def test_build_entry_context_features_clipping():
    """Test that atr_bps and spread_bps are clipped to valid ranges."""
    dates = pd.date_range("2025-01-01", periods=100, freq="5min", tz="UTC")
    candles = pd.DataFrame({
        "open": 2650.0,
        "high": 2651.0,
        "low": 2649.0,
        "close": 2650.5,
        "bid_close": 2650.4,
        "ask_close": 2650.6,
    }, index=dates)
    
    policy_state = {"session": "EU"}
    
    # Test extreme ATR (should be clipped to 1000 bps)
    ctx_high_atr = build_entry_context_features(
        candles=candles,
        policy_state=policy_state.copy(),
        atr_proxy=100.0,  # Very high ATR (would be > 1000 bps)
        spread_bps=10.0,
        is_replay=True,
    )
    
    assert ctx_high_atr.atr_bps <= 1000.0, f"atr_bps should be clipped to 1000, got {ctx_high_atr.atr_bps}"
    
    # Test extreme spread (should be clipped to 500 bps)
    ctx_high_spread = build_entry_context_features(
        candles=candles,
        policy_state=policy_state.copy(),
        atr_proxy=0.5,
        spread_bps=1000.0,  # Very high spread
        is_replay=True,
    )
    
    assert ctx_high_spread.spread_bps <= 500.0, f"spread_bps should be clipped to 500, got {ctx_high_spread.spread_bps}"


def test_build_entry_context_features_missing_atr():
    """Test that missing ATR raises error in replay mode."""
    # Create empty candles (will cause ATR computation to fail)
    dates = pd.date_range("2025-01-01", periods=1, freq="5min", tz="UTC")
    candles = pd.DataFrame({
        "open": 2650.0,
        "high": 2651.0,
        "low": 2649.0,
        "close": 2650.5,
        "bid_close": 2650.4,
        "ask_close": 2650.6,
    }, index=dates)
    
    policy_state = {"session": "EU"}
    
    # In replay mode, should raise error (ATR proxy computation will fail with too few bars)
    # Note: With only 1 bar, _compute_cheap_atr_proxy will return None
    with pytest.raises(RuntimeError, match="CONTEXT_FEATURE_MISSING.*atr_bps"):
        build_entry_context_features(
            candles=candles,
            policy_state=policy_state.copy(),
            atr_proxy=None,  # Missing, will try to compute but fail (needs 14 bars)
            spread_bps=10.0,
            is_replay=True,
        )
    
    # In live mode, should use default (50.0 bps)
    ctx = build_entry_context_features(
        candles=candles,
        policy_state=policy_state.copy(),
        atr_proxy=None,  # Missing, will try to compute but fail, then use default
        spread_bps=10.0,
        is_replay=False,
    )
    
    # Should use default (50.0 bps) when ATR cannot be computed
    assert ctx.atr_bps == 50.0, f"Should use default atr_bps=50.0 in live mode when ATR unavailable, got {ctx.atr_bps}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

