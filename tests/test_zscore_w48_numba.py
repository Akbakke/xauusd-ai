"""
Unit test for zscore_w48 Numba-accelerated implementation.
Tests that it matches pandas zscore behavior for window=48.
Hard correctness test: if this fails, system is wrong.
"""
import numpy as np
import pandas as pd
import pytest

from gx1.features.rolling_np import zscore_w48


def test_zscore_w48_random_data():
    """Test zscore_w48 on random data without NaNs."""
    np.random.seed(42)
    data = np.random.randn(2000) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 24  # win//2
    
    # Pandas reference
    r = s.rolling(window, min_periods=min_periods)
    mean_result = r.mean()
    std_result = r.std(ddof=0)
    expected = (s - mean_result) / (std_result + 1e-12)
    
    # Numba implementation
    got = zscore_w48(data, min_periods=min_periods)
    got_series = pd.Series(got, index=s.index)
    
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    assert np.any(mask), "Expected should have some non-NaN values"
    
    assert np.allclose(
        got[mask],
        expected.values[mask],
        rtol=1e-6,
        atol=1e-6,
        equal_nan=False
    ), f"Max difference: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match
    assert np.all(np.isnan(got[~mask]) == np.isnan(expected.values[~mask])), \
        f"NaN positions do not match. Expected NaN count: {np.sum(np.isnan(expected.values))}, Got NaN count: {np.sum(np.isnan(got))}"


def test_zscore_w48_with_nans():
    """Test zscore_w48 with NaN injection (simulating resample pattern)."""
    np.random.seed(43)
    data = np.random.randn(2000) * 10 + 50
    
    # Inject NaNs in a pattern (every 100th value, simulating resample artifacts)
    nan_indices = np.arange(100, len(data), 100)
    data[nan_indices] = np.nan
    
    # Also inject some Inf values
    inf_indices = np.arange(50, len(data), 200)
    data[inf_indices] = np.inf
    
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    # Pandas reference
    r = s.rolling(window, min_periods=min_periods)
    mean_result = r.mean()
    std_result = r.std(ddof=0)
    expected = (s - mean_result) / (std_result + 1e-12)
    
    # Numba implementation
    got = zscore_w48(data, min_periods=min_periods)
    
    # Compare finite values only (both got and expected must be finite)
    mask = np.isfinite(expected.values) & np.isfinite(got)
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"Max difference with NaNs: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match (at least: all expected NaNs should be NaNs in got)
    expected_nan_mask = np.isnan(expected.values)
    if np.any(expected_nan_mask):
        assert np.all(np.isnan(got[expected_nan_mask])), \
            "All positions where expected is NaN should also be NaN in got"


def test_zscore_w48_min_periods():
    """Test that min_periods is respected."""
    np.random.seed(44)
    data = np.random.randn(100) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    # Pandas reference
    r = s.rolling(window, min_periods=min_periods)
    mean_result = r.mean()
    std_result = r.std(ddof=0)
    expected = (s - mean_result) / (std_result + 1e-12)
    
    # Numba implementation
    got = zscore_w48(data, min_periods=min_periods)
    
    # First min_periods-1 values should be NaN
    assert np.all(np.isnan(got[:min_periods-1])), \
        f"First {min_periods-1} values should be NaN"
    
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        )


def test_zscore_w48_simple_case():
    """Test zscore_w48 on simple ascending sequence."""
    # Simple ascending sequence
    data = np.arange(1, 200, dtype=np.float64)
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    # Pandas reference
    r = s.rolling(window, min_periods=min_periods)
    mean_result = r.mean()
    std_result = r.std(ddof=0)
    expected = (s - mean_result) / (std_result + 1e-12)
    
    # Numba implementation
    got = zscore_w48(data, min_periods=min_periods)
    
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"Max difference in simple case: {np.max(np.abs(got[mask] - expected.values[mask]))}"


def test_zscore_w48_constant_series():
    """Test zscore_w48 on constant series (std=0)."""
    data = np.ones(200, dtype=np.float64) * 42.0
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    # Pandas reference
    r = s.rolling(window, min_periods=min_periods)
    mean_result = r.mean()
    std_result = r.std(ddof=0)
    expected = (s - mean_result) / (std_result + 1e-12)
    
    # Numba implementation
    got = zscore_w48(data, min_periods=min_periods)
    
    # For constant series, std=0, so z-score should be 0 (or NaN if division by zero)
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

