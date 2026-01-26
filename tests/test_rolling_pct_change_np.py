"""
Unit tests for pct_change_np function.
"""
import numpy as np
import pytest
from gx1.features.rolling_np import pct_change_np


def test_pct_change_basic():
    """Test basic pct_change calculation."""
    x = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    result = pct_change_np(x, k=1)
    
    # Expected: [nan, (2-1)/1, (4-2)/2] = [nan, 1.0, 1.0]
    assert np.isnan(result[0])
    assert np.allclose(result[1], 1.0)
    assert np.allclose(result[2], 1.0)
    assert result.shape == (3,)
    assert result.dtype == np.float64


def test_pct_change_with_nan():
    """Test pct_change with NaN values."""
    x = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float64)
    result = pct_change_np(x, k=1)
    
    # Position 1 should be NaN (x[1] is NaN)
    assert np.isnan(result[1])
    # Position 2 should also be NaN (x[1] (prev) is NaN, so x[2]/x[1] is invalid)
    assert np.isnan(result[2])
    # Position 3: (4-3)/3 = 1/3 ≈ 0.333
    assert np.allclose(result[3], 1.0/3.0)


def test_pct_change_with_zero_denominator():
    """Test pct_change with zero denominator (should return NaN)."""
    x = np.array([1.0, 0.0, 3.0, 4.0], dtype=np.float64)
    result = pct_change_np(x, k=1)
    
    # Position 1: x[1]/x[0] = 0/1 = 0, but pct_change = (0-1)/1 = -1.0
    # Actually wait, pct_change = (x[i] - x[i-k]) / x[i-k]
    # So position 1: (x[1] - x[0]) / x[0] = (0 - 1) / 1 = -1.0
    assert np.allclose(result[1], -1.0)
    
    # Position 2: x[2]/x[1] but x[1]=0, so should be NaN
    assert np.isnan(result[2])


def test_pct_change_k_equals_n():
    """Test when k >= n (should return all NaN)."""
    x = np.array([1.0, 2.0], dtype=np.float64)
    result = pct_change_np(x, k=2)
    
    assert result.shape == (2,)
    assert np.all(np.isnan(result))


def test_pct_change_dtype():
    """Test dtype consistency (should return float64)."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = pct_change_np(x, k=1)
    
    assert result.dtype == np.float64


def test_pct_change_2d_input_rejected():
    """Test that 2D input is rejected."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    with pytest.raises(ValueError, match="requires 1D input"):
        pct_change_np(x, k=1)


def test_pct_change_empty_array():
    """Test empty array."""
    x = np.array([], dtype=np.float64)
    result = pct_change_np(x, k=1)
    
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_pct_change_k_validation():
    """Test k validation."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="k must be >= 1"):
        pct_change_np(x, k=0)
    
    with pytest.raises(ValueError, match="k must be >= 1"):
        pct_change_np(x, k=-1)


def test_pct_change_large_array():
    """Test with larger array (stress test)."""
    x = np.random.randn(1000).astype(np.float64)
    result = pct_change_np(x, k=1)
    
    assert result.shape == (1000,)
    assert result.dtype == np.float64
    assert np.isnan(result[0])  # First element should be NaN
    assert np.all(np.isfinite(result[1:]) | np.isnan(result[1:]))  # Rest should be finite or NaN


def test_pct_change_k_5():
    """Test with k=5."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0], dtype=np.float64)
    result = pct_change_np(x, k=5)
    
    # First 5 should be NaN
    assert np.all(np.isnan(result[:5]))
    # Position 5: (10-1)/1 = 9.0
    assert np.allclose(result[5], 9.0)
    # Position 6: (20-2)/2 = 9.0
    assert np.allclose(result[6], 9.0)

