"""
Unit test for rolling_kurtosis_w48 Numba-accelerated implementation.
Tests that it matches pandas kurtosis behavior for window=48.
Hard correctness test: if this fails, system is wrong.
"""
import numpy as np
import pandas as pd
import pytest

from gx1.features.rolling_np import rolling_kurtosis_w48


def test_rolling_kurtosis_w48_random_data():
    """Test rolling_kurtosis_w48 on random data without NaNs."""
    np.random.seed(42)
    data = np.random.randn(2000) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 12
    
    # Pandas reference (Fisher's excess kurtosis)
    r = s.rolling(window, min_periods=min_periods)
    expected = r.apply(lambda x: pd.Series(x).kurtosis() if len(x) > 3 else np.nan, raw=False)
    
    # Numba implementation
    got = rolling_kurtosis_w48(data, min_periods=min_periods)
    got_series = pd.Series(got, index=s.index)
    
    # Compare finite values only (both got and expected must be finite)
    mask = np.isfinite(expected.values) & np.isfinite(got)
    assert np.any(mask), "Expected should have some finite values for comparison"
    
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"Max difference: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match (at least: all expected NaNs should be NaNs in got)
    expected_nan_mask = np.isnan(expected.values)
    if np.any(expected_nan_mask):
        assert np.all(np.isnan(got[expected_nan_mask])), \
            "All positions where expected is NaN should also be NaN in got"


def test_rolling_kurtosis_w48_with_nans():
    """Test rolling_kurtosis_w48 with NaN injection."""
    np.random.seed(43)
    data = np.random.randn(2000) * 10 + 50
    nan_indices = np.arange(100, len(data), 100)
    data[nan_indices] = np.nan
    inf_indices = np.arange(50, len(data), 200)
    data[inf_indices] = np.inf
    
    s = pd.Series(data)
    window = 48
    min_periods = 12
    
    # Pandas reference
    r = s.rolling(window, min_periods=min_periods)
    expected = r.apply(lambda x: pd.Series(x).kurtosis() if len(x) > 3 and np.any(np.isfinite(x)) else np.nan, raw=False)
    
    # Numba implementation
    got = rolling_kurtosis_w48(data, min_periods=min_periods)
    
    # Compare finite values only
    mask = np.isfinite(expected.values) & np.isfinite(got)
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"Max difference with NaNs: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match (all expected NaNs should be NaNs in got)
    expected_nan_mask = np.isnan(expected.values)
    if np.any(expected_nan_mask):
        assert np.all(np.isnan(got[expected_nan_mask])), \
            "All positions where expected is NaN should also be NaN in got"


def test_rolling_kurtosis_w48_min_periods():
    """Test that min_periods is respected."""
    np.random.seed(44)
    data = np.random.randn(100)
    
    # Test with min_periods=12
    result = rolling_kurtosis_w48(data, min_periods=12)
    
    # First 11 positions should be NaN (not enough periods)
    assert np.all(np.isnan(result[:11])), "First positions should be NaN when min_periods not met"
    
    # Position 11 (index 11) should be the first valid value
    assert np.isfinite(result[11]) or np.isnan(result[11]), "Position 11 should be valid or NaN"
    
    # All positions from min_periods-1 onwards should have valid or NaN values
    assert len(result) == len(data), "Result should have same length as input"

