#!/usr/bin/env python3
"""
Unit tests for prebuilt builder reserved columns filter.

Tests that sanitize_feature_columns() correctly:
- Drops CLOSE column silently
- Hard-fails on other reserved columns
- Passes through non-reserved columns
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.scripts.build_fullyear_features_parquet import sanitize_feature_columns


def test_drop_close_only():
    """Test 1: DF with CLOSE column only -> should be dropped silently."""
    df = pd.DataFrame({
        "CLOSE": [100.0, 101.0, 102.0],
        "_v1_atr14": [0.5, 0.6, 0.7],
        "other_feature": [1.0, 2.0, 3.0],
    })
    
    df_sanitized, metadata = sanitize_feature_columns(df)
    
    # CLOSE should be dropped
    assert "CLOSE" not in df_sanitized.columns
    assert "_v1_atr14" in df_sanitized.columns
    assert "other_feature" in df_sanitized.columns
    assert metadata["dropped_columns"] == ["CLOSE"]


def test_fail_on_close_and_close_collision():
    """Test 2: DF with both CLOSE and close -> should hard-fail (case collision)."""
    # Note: pandas automatically handles duplicate column names, so we need to create
    # the collision differently. We'll test that check_reserved_candle_columns detects both.
    # In practice, case collisions are caught by column_collision_guard before sanitize.
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0],
        "_v1_atr14": [0.5, 0.6, 0.7],
    })
    
    # Add CLOSE column manually (pandas would drop duplicate, but we simulate the check)
    df["CLOSE"] = [100.0, 101.0, 102.0]
    
    # sanitize_feature_columns should detect "close" (lowercase) as reserved and hard-fail
    with pytest.raises(RuntimeError, match="PREBUILT_SCHEMA_FAIL"):
        sanitize_feature_columns(df)


def test_fail_on_other_reserved_columns():
    """Test 3: DF with open, high, low etc -> should hard-fail."""
    df = pd.DataFrame({
        "open": [99.0, 100.0, 101.0],
        "_v1_atr14": [0.5, 0.6, 0.7],
    })
    
    with pytest.raises(RuntimeError, match="PREBUILT_SCHEMA_FAIL"):
        sanitize_feature_columns(df)
    
    # Test with high
    df = pd.DataFrame({
        "high": [101.0, 102.0, 103.0],
        "_v1_atr14": [0.5, 0.6, 0.7],
    })
    
    with pytest.raises(RuntimeError, match="PREBUILT_SCHEMA_FAIL"):
        sanitize_feature_columns(df)


def test_pass_without_reserved_columns():
    """Test 4: DF without reserved cols -> should pass through unchanged."""
    df = pd.DataFrame({
        "_v1_atr14": [0.5, 0.6, 0.7],
        "_v1_body_share_1": [0.3, 0.4, 0.5],
        "other_feature": [1.0, 2.0, 3.0],
    })
    
    df_sanitized, metadata = sanitize_feature_columns(df)
    
    # Should be unchanged
    assert len(df_sanitized.columns) == len(df.columns)
    assert set(df_sanitized.columns) == set(df.columns)
    assert metadata["dropped_columns"] == []
    assert metadata["reserved_found"] == []


def test_case_insensitive_reserved_detection():
    """Test that reserved column detection is case-insensitive."""
    # Test with lowercase 'close' (should be detected as reserved)
    df = pd.DataFrame({
        "close": [100.0, 101.0, 102.0],
        "_v1_atr14": [0.5, 0.6, 0.7],
    })
    
    with pytest.raises(RuntimeError, match="PREBUILT_SCHEMA_FAIL"):
        sanitize_feature_columns(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
