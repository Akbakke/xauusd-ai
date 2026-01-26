"""
Unit test for zscore_w48_nanaware NumPy implementation.
Tests that it matches pandas _zscore behavior for window=48.
"""
import numpy as np
import pandas as pd
import pytest

from gx1.features.rolling_np import zscore_w48_nanaware
from gx1.features.basic_v1 import _zscore


def test_zscore_w48_nanaware_random_data():
    """Test zscore_w48_nanaware on random data without NaNs."""
    np.random.seed(42)
    data = np.random.randn(2000) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 24  # win//2
    
    # Pandas reference (using _zscore)
    expected = _zscore(s, window)
    
    # NumPy implementation
    got = zscore_w48_nanaware(data, min_periods=min_periods)
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


def test_zscore_w48_nanaware_with_nans():
    """Test zscore_w48_nanaware with NaN injection (simulating resample pattern)."""
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
    expected = _zscore(s, window)
    
    # NumPy implementation
    got = zscore_w48_nanaware(data, min_periods=min_periods)
    
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    if np.any(mask):
        assert np.allclose(
            got[mask],
            expected.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"Max difference with NaNs: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match (at least approximately - pandas may have more NaNs due to window effects)
    # For zscore, if mean or std is NaN, result should be NaN
    # We check that all positions where expected is NaN, got is also NaN
    expected_nan_mask = np.isnan(expected.values)
    if np.any(expected_nan_mask):
        assert np.all(np.isnan(got[expected_nan_mask])), \
            "All positions where expected is NaN should also be NaN in got"


def test_zscore_w48_nanaware_min_periods():
    """Test that min_periods is respected."""
    np.random.seed(44)
    data = np.random.randn(100) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    # Pandas reference
    expected = _zscore(s, window)
    
    # NumPy implementation
    got = zscore_w48_nanaware(data, min_periods=min_periods)
    
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


def test_zscore_w48_nanaware_simple_case():
    """Test zscore_w48_nanaware on simple ascending sequence."""
    # Simple ascending sequence
    data = np.arange(1, 200, dtype=np.float64)
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    # Pandas reference
    expected = _zscore(s, window)
    
    # NumPy implementation
    got = zscore_w48_nanaware(data, min_periods=min_periods)
    
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


def test_zscore_w48_nanaware_constant_series():
    """Test zscore_w48_nanaware on constant series (std=0)."""
    data = np.ones(200, dtype=np.float64) * 42.0
    s = pd.Series(data)
    
    window = 48
    min_periods = 24
    
    # Pandas reference
    expected = _zscore(s, window)
    
    # NumPy implementation
    got = zscore_w48_nanaware(data, min_periods=min_periods)
    
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

