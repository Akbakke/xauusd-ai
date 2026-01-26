"""
Unit test for rolling_kurtosis_w48 NumPy implementation.

Del 4: Tests that rolling_kurtosis_w48 matches pandas Series.kurtosis() behavior.
"""

import numpy as np
import pandas as pd
import pytest

from gx1.features.rolling_np import rolling_kurtosis_w48


def test_rolling_kurtosis_w48_random_data():
    """Test rolling_kurtosis_w48 against pandas on random data."""
    np.random.seed(42)
    n = 1000
    x = np.random.randn(n).astype(np.float64) * 10.0 + 100.0  # Avoid small numbers
    
    # NumPy implementation
    result_np = rolling_kurtosis_w48(x, min_periods=12, fisher=True, bias=True)
    
    # Pandas reference (same as fallback in basic_v1.py)
    s = pd.Series(x)
    result_pd = s.rolling(48, min_periods=12).apply(
        lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0.0,
        raw=False
    )
    
    # Compare (mask NaNs)
    mask = np.isfinite(result_np) & np.isfinite(result_pd.values)
    
    assert np.sum(mask) > 0, "Should have some valid comparisons"
    
    # Check differences
    diff = np.abs(result_np[mask] - result_pd.values[mask])
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Valid comparisons: {np.sum(mask)}")
    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    # Kurtosis can be numerically sensitive, so use reasonable tolerance
    # pandas uses scipy.stats internally which may have slightly different numerics
    np.testing.assert_allclose(
        result_np[mask],
        result_pd.values[mask],
        rtol=1e-5,
        atol=1e-4,
        equal_nan=True,
        err_msg="rolling_kurtosis_w48 should match pandas kurtosis closely"
    )
    
    # Also check NaN positions match
    np_nan_mask = np.isnan(result_np)
    pd_nan_mask = np.isnan(result_pd.values)
    assert np.all(np_nan_mask == pd_nan_mask), "NaN positions should match"


def test_rolling_kurtosis_w48_simple_case():
    """Test on simple known data."""
    # Create data where we can verify kurtosis manually
    # Use larger window for simpler case
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=np.float64)  # 100 points
    
    result_np = rolling_kurtosis_w48(x, min_periods=12, fisher=True, bias=True)
    s = pd.Series(x)
    result_pd = s.rolling(48, min_periods=12).apply(
        lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0.0,
        raw=False
    )
    
    # Compare non-NaN values
    mask = np.isfinite(result_np) & np.isfinite(result_pd.values)
    if np.any(mask):
        np.testing.assert_allclose(
            result_np[mask],
            result_pd.values[mask],
            rtol=1e-5,
            atol=1e-4,
            equal_nan=True
        )


def test_rolling_kurtosis_w48_with_nans_and_infs():
    """Test that NaN/Inf in input produces NaN in output."""
    np.random.seed(42)
    n = 200
    x = np.random.randn(n).astype(np.float64)
    
    # Inject NaN at position 50
    x[50] = np.nan
    
    # Inject Inf at position 100
    x[100] = np.inf
    
    result_np = rolling_kurtosis_w48(x, min_periods=12, fisher=True, bias=True)
    
    # Check that windows containing NaN/Inf produce NaN
    window = 48
    for i in range(len(x)):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_data = x[start_idx:end_idx]
        
        if not np.all(np.isfinite(window_data)):
            # If window contains NaN/Inf, result should be NaN
            assert np.isnan(result_np[i]), f"Expected NaN at index {i} (window contains NaN/Inf)"
    
    # Also compare with pandas (which should also produce NaN for these windows)
    s = pd.Series(x)
    result_pd = s.rolling(48, min_periods=12).apply(
        lambda x: pd.Series(x).kurtosis() if len(x) > 3 and np.all(np.isfinite(x)) else np.nan,
        raw=False
    )
    
    # NaN positions should match
    np_nan_mask = np.isnan(result_np)
    pd_nan_mask = np.isnan(result_pd.values)
    # Note: pandas may have slightly different NaN handling, so check overlap
    # If pandas has NaN, NumPy should also have NaN
    assert np.all(np_nan_mask[pd_nan_mask]), "NumPy should have NaN where pandas has NaN"


def test_rolling_kurtosis_w48_min_periods():
    """Test that min_periods is respected."""
    n = 100
    x = np.random.randn(n).astype(np.float64)
    
    result = rolling_kurtosis_w48(x, min_periods=12, fisher=True, bias=True)
    
    window = 48
    # First min_periods-1 values should be NaN
    for i in range(min(12, len(result))):
        assert np.isnan(result[i]), f"Expected NaN at index {i} (min_periods={12})"
    
    # Values after min_periods should be computed (if window is large enough)
    for i in range(min_periods, min(window, len(result))):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_size = end_idx - start_idx
        if window_size >= 12:
            # Should be finite if data is finite
            if np.all(np.isfinite(x[start_idx:end_idx])):
                assert np.isfinite(result[i]) or result[i] == 0.0, \
                    f"Expected finite value at index {i} (window_size={window_size})"


def test_rolling_kurtosis_w48_fisher_correction():
    """Test that Fisher correction (excess kurtosis) is applied correctly."""
    # Normal distribution should have Fisher kurtosis ≈ 0
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float64)
    
    result = rolling_kurtosis_w48(x, min_periods=12, fisher=True, bias=True)
    
    # For large samples from normal distribution, kurtosis should be close to 0
    mask = np.isfinite(result)
    if np.any(mask):
        mean_kurt = np.mean(result[mask])
        # Normal distribution: Fisher kurtosis ≈ 0
        assert abs(mean_kurt) < 1.0, f"Normal distribution should have mean kurtosis near 0, got {mean_kurt}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

