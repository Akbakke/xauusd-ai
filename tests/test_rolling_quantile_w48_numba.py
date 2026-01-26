"""
Unit test for rolling_quantile_w48 Numba-accelerated implementation.
Tests that it matches pandas rolling quantile behavior for window=48.
Hard correctness test: if this fails, system is wrong.
"""
import numpy as np
import pandas as pd
import pytest

from gx1.features.rolling_np import rolling_quantile_w48


def test_rolling_quantile_w48_random_data():
    """Test rolling_quantile_w48 on random data without NaNs."""
    np.random.seed(42)
    data = np.random.randn(2000) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    q = 0.90
    
    # Pandas reference
    expected = s.rolling(window, min_periods=min_periods).quantile(q)
    
    # Numba implementation
    got = rolling_quantile_w48(data, q=q, min_periods=min_periods)
    
    # Compare non-NaN values
    mask = np.isfinite(expected.values) & np.isfinite(got)
    assert np.any(mask), "Expected should have some non-NaN values"
    
    assert np.allclose(
        got[mask],
        expected.values[mask],
        rtol=1e-6,
        atol=1e-6,
        equal_nan=False
    ), f"Max difference: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match
    assert np.all(np.isnan(got) == np.isnan(expected.values)), \
        f"NaN positions do not match. Expected NaN count: {np.sum(np.isnan(expected.values))}, Got NaN count: {np.sum(np.isnan(got))}"


def test_rolling_quantile_w48_multiple_q():
    """Test rolling_quantile_w48 with multiple quantile values."""
    np.random.seed(43)
    data = np.random.randn(500) * 5 + 20
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        # Pandas reference
        expected = s.rolling(window, min_periods=min_periods).quantile(q)
        
        # Numba implementation
        got = rolling_quantile_w48(data, q=q, min_periods=min_periods)
        
        # Compare finite values only
        mask = np.isfinite(expected.values) & np.isfinite(got)
        if np.any(mask):
            assert np.allclose(
                got[mask],
                expected.values[mask],
                rtol=1e-6,
                atol=1e-6,
                equal_nan=False
            ), f"Quantile {q} failed. Max diff: {np.max(np.abs(got[mask] - expected.values[mask]))}"


def test_rolling_quantile_w48_with_nans():
    """Test rolling_quantile_w48 with NaN injection."""
    np.random.seed(44)
    data = np.random.randn(500) * 5 + 20
    
    # Inject NaNs in a pattern
    nan_indices = np.arange(50, len(data), 100)
    data[nan_indices] = np.nan
    
    # Also inject some Inf values
    inf_indices = np.arange(75, len(data), 150)
    data[inf_indices] = np.inf
    
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    q = 0.90
    
    # Pandas reference
    expected = s.rolling(window, min_periods=min_periods).quantile(q)
    
    # Numba implementation
    got = rolling_quantile_w48(data, q=q, min_periods=min_periods)
    
    # Compare finite values only
    mask = np.isfinite(expected.values) & np.isfinite(got)
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"Max difference: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # NaN positions should match (where pandas outputs NaN, we should too)
    assert np.all(np.isnan(got) == np.isnan(expected.values)), \
        f"NaN positions do not match. Expected NaN count: {np.sum(np.isnan(expected.values))}, Got NaN count: {np.sum(np.isnan(got))}"


def test_rolling_quantile_w48_min_periods():
    """Test rolling_quantile_w48 with different min_periods values."""
    np.random.seed(45)
    data = np.random.randn(200) * 3 + 10
    s = pd.Series(data)
    
    window = 48
    q = 0.50
    
    for min_periods in [12, 24, 36]:
        # Pandas reference
        expected = s.rolling(window, min_periods=min_periods).quantile(q)
        
        # Numba implementation
        got = rolling_quantile_w48(data, q=q, min_periods=min_periods)
        
        # Compare finite values only
        mask = np.isfinite(expected.values) & np.isfinite(got)
        if np.any(mask):
            assert np.allclose(
                got[mask],
                expected.values[mask],
                rtol=1e-6,
                atol=1e-6,
                equal_nan=False
            ), f"min_periods={min_periods} failed. Max diff: {np.max(np.abs(got[mask] - expected.values[mask]))}"


def test_rolling_quantile_w48_edge_cases():
    """Test rolling_quantile_w48 edge cases."""
    np.random.seed(46)
    
    # Short array
    data_short = np.random.randn(30).astype(np.float64)
    result_short = rolling_quantile_w48(data_short, q=0.5, min_periods=24)
    assert len(result_short) == 30
    # Should have mostly NaNs (not enough data)
    assert np.sum(np.isfinite(result_short)) < 10
    
    # Constant array
    data_const = np.full(100, 5.0, dtype=np.float64)
    result_const = rolling_quantile_w48(data_const, q=0.5, min_periods=24)
    mask_const = np.isfinite(result_const)
    if np.any(mask_const):
        assert np.allclose(result_const[mask_const], 5.0, rtol=1e-6, atol=1e-6)
    
    # All NaN
    data_nan = np.full(100, np.nan, dtype=np.float64)
    result_nan = rolling_quantile_w48(data_nan, q=0.5, min_periods=24)
    assert np.all(np.isnan(result_nan))



