"""
Unit tests for column normalization in build_v9_live_base_features.
Tests collision detection for duplicate columns after case-insensitive normalization.
"""
import pytest
import pandas as pd
import numpy as np
from gx1.features.runtime_v9 import build_v9_live_base_features, V9RuntimeFeatureError


def test_column_normalization_unique_columns():
    """Test that unique columns (different cases) normalize correctly."""
    df = pd.DataFrame({
        'Close': [1.0, 2.0, 3.0],
        'Open': [0.9, 1.9, 2.9],
        'High': [1.1, 2.1, 3.1],
        'Low': [0.8, 1.8, 2.8],
        'Volume': [100, 200, 300],
    }, index=pd.date_range('2025-01-01', periods=3, freq='5min'))
    
    # Should not raise
    result = build_v9_live_base_features(df)
    
    # Verify columns are normalized to lowercase
    assert 'close' in result.columns
    assert 'open' in result.columns
    assert 'high' in result.columns
    assert 'low' in result.columns
    assert 'volume' in result.columns
    
    # Verify values are preserved
    assert np.allclose(result['close'].values, [1.0, 2.0, 3.0])


def test_column_normalization_collision_detection():
    """Test that collisions (multiple columns → same normalized name) raise ValueError."""
    df = pd.DataFrame({
        'Close': [1.0, 2.0, 3.0],
        'CLOSE': [1.1, 2.1, 3.1],  # Collision: maps to same 'close'
        'Open': [0.9, 1.9, 2.9],
        'High': [1.1, 2.1, 3.1],
        'Low': [0.8, 1.8, 2.8],
        'Volume': [100, 200, 300],
    }, index=pd.date_range('2025-01-01', periods=3, freq='5min'))
    
    with pytest.raises(V9RuntimeFeatureError) as exc_info:
        build_v9_live_base_features(df)
    
    error_msg = str(exc_info.value)
    assert 'Collisions' in error_msg or 'collision' in error_msg.lower()
    assert 'close' in error_msg.lower() or 'Close' in error_msg or 'CLOSE' in error_msg


def test_column_normalization_collision_multiple_required():
    """Test collision detection works for multiple required columns."""
    df = pd.DataFrame({
        'Close': [1.0, 2.0, 3.0],
        'CLOSE': [1.1, 2.1, 3.1],
        'Open': [0.9, 1.9, 2.9],
        'OPEN': [0.95, 1.95, 2.95],  # Another collision
        'High': [1.1, 2.1, 3.1],
        'Low': [0.8, 1.8, 2.8],
        'Volume': [100, 200, 300],
    }, index=pd.date_range('2025-01-01', periods=3, freq='5min'))
    
    with pytest.raises(V9RuntimeFeatureError) as exc_info:
        build_v9_live_base_features(df)
    
    error_msg = str(exc_info.value)
    assert 'Collisions' in error_msg or 'collision' in error_msg.lower()


def test_column_normalization_already_lowercase():
    """Test that already lowercase columns work correctly."""
    df = pd.DataFrame({
        'close': [1.0, 2.0, 3.0],
        'open': [0.9, 1.9, 2.9],
        'high': [1.1, 2.1, 3.1],
        'low': [0.8, 1.8, 2.8],
        'volume': [100, 200, 300],
    }, index=pd.date_range('2025-01-01', periods=3, freq='5min'))
    
    # Should not raise
    result = build_v9_live_base_features(df)
    
    # Verify columns remain lowercase
    assert 'close' in result.columns
    assert 'open' in result.columns
    assert np.allclose(result['close'].values, [1.0, 2.0, 3.0])


def test_column_normalization_mixed_case_unique():
    """Test that mixed case (but unique) columns normalize correctly."""
    df = pd.DataFrame({
        'Close': [1.0, 2.0, 3.0],
        'OPEN': [0.9, 1.9, 2.9],
        'high': [1.1, 2.1, 3.1],
        'LoW': [0.8, 1.8, 2.8],
        'VOLUME': [100, 200, 300],
    }, index=pd.date_range('2025-01-01', periods=3, freq='5min'))
    
    # Should not raise
    result = build_v9_live_base_features(df)
    
    # Verify all are normalized to lowercase
    assert 'close' in result.columns
    assert 'open' in result.columns
    assert 'high' in result.columns
    assert 'low' in result.columns
    assert 'volume' in result.columns

