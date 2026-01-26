"""
Unit tests for rolling_std_3 NumPy implementation.

Tests that rolling_std_3 matches pandas Series.rolling(3, min_periods=2).std(ddof=0)
"""

import numpy as np
import pandas as pd
import pytest
from gx1.features.rolling_np import rolling_std_3


def test_rolling_std_3_vs_pandas():
    """Test that rolling_std_3 matches pandas output on random data."""
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n).astype(np.float64)
    
    # Pandas reference
    s = pd.Series(x)
    expected = s.rolling(3, min_periods=2).std(ddof=0).to_numpy(dtype=np.float64)
    
    # NumPy implementation
    got = rolling_std_3(x, min_periods=2, ddof=0)
    
    # Compare: handle NaNs separately
    assert len(got) == len(expected), "Length mismatch"
    
    # Check NaN positions match
    expected_nan_mask = np.isnan(expected)
    got_nan_mask = np.isnan(got)
    np.testing.assert_array_equal(expected_nan_mask, got_nan_mask, 
                                   err_msg="NaN positions don't match")
    
    # Compare non-NaN values
    finite_mask = ~expected_nan_mask
    if np.any(finite_mask):
        np.testing.assert_allclose(
            got[finite_mask],
            expected[finite_mask],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Finite values don't match"
        )


def test_rolling_std_3_with_nan_inf():
    """Test that rolling_std_3 handles NaN/Inf correctly."""
    n = 100
    x = np.random.randn(n).astype(np.float64)
    
    # Insert some NaNs and Inf
    x[10] = np.nan
    x[20] = np.inf
    x[21] = -np.inf
    x[30] = np.nan
    x[31] = np.nan  # Consecutive NaNs
    
    # Pandas reference
    s = pd.Series(x)
    expected = s.rolling(3, min_periods=2).std(ddof=0).to_numpy(dtype=np.float64)
    
    # NumPy implementation
    got = rolling_std_3(x, min_periods=2, ddof=0)
    
    # Check NaN positions match
    expected_nan_mask = np.isnan(expected)
    got_nan_mask = np.isnan(got)
    np.testing.assert_array_equal(expected_nan_mask, got_nan_mask,
                                   err_msg="NaN positions don't match with NaN/Inf input")
    
    # Compare non-NaN values (should be very few or none due to NaN propagation)
    finite_mask = ~expected_nan_mask
    if np.any(finite_mask):
        np.testing.assert_allclose(
            got[finite_mask],
            expected[finite_mask],
            rtol=1e-10,
            atol=1e-12,
            err_msg="Finite values don't match with NaN/Inf input"
        )


def test_rolling_std_3_edge_cases():
    """Test edge cases: empty, short arrays."""
    # Empty array
    x_empty = np.array([], dtype=np.float64)
    result_empty = rolling_std_3(x_empty, min_periods=2, ddof=0)
    assert len(result_empty) == 0
    
    # Single element (should all be NaN)
    x1 = np.array([1.0], dtype=np.float64)
    result1 = rolling_std_3(x1, min_periods=2, ddof=0)
    assert len(result1) == 1
    assert np.isnan(result1[0])
    
    # Two elements (only index 1 should have value)
    x2 = np.array([1.0, 2.0], dtype=np.float64)
    result2 = rolling_std_3(x2, min_periods=2, ddof=0)
    assert len(result2) == 2
    assert np.isnan(result2[0])
    assert not np.isnan(result2[1])
    # Check value: std of [1.0, 2.0] with ddof=0
    expected_std = np.std([1.0, 2.0], ddof=0)
    np.testing.assert_allclose(result2[1], expected_std, rtol=1e-10)


def test_rolling_std_3_assertions():
    """Test that assertions are raised for unsupported parameters."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    # Test unsupported ddof
    with pytest.raises(AssertionError, match="ddof"):
        rolling_std_3(x, min_periods=2, ddof=1)
    
    # Test unsupported min_periods
    with pytest.raises(AssertionError, match="min_periods"):
        rolling_std_3(x, min_periods=1, ddof=0)

