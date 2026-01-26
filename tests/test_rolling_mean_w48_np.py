import numpy as np
import pandas as pd
import pytest
from gx1.features.rolling_np import rolling_mean_w48


def test_rolling_mean_w48_random_data():
    """Test rolling_mean_w48 against pandas on random data."""
    np.random.seed(42)
    data = np.random.randn(2000) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 48
    
    expected = s.rolling(window=window, min_periods=min_periods).mean()
    got = rolling_mean_w48(data, min_periods=min_periods)
    
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    assert np.allclose(got[mask], expected.values[mask], rtol=1e-12, atol=1e-12, equal_nan=False), \
        f"Max difference: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match
    assert np.all(np.isnan(got[~mask]) == np.isnan(expected.values[~mask])), \
        "NaN positions do not match"


def test_rolling_mean_w48_min_periods_12():
    """Test rolling_mean_w48 with min_periods=12 (partial window)."""
    np.random.seed(42)
    data = np.random.randn(2000) * 10 + 50
    s = pd.Series(data)
    
    window = 48
    min_periods = 12
    
    expected = s.rolling(window=window, min_periods=min_periods).mean()
    got = rolling_mean_w48(data, min_periods=min_periods)
    
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    assert np.allclose(got[mask], expected.values[mask], rtol=1e-12, atol=1e-12, equal_nan=False), \
        f"Max difference: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match
    assert np.all(np.isnan(got[~mask]) == np.isnan(expected.values[~mask])), \
        "NaN positions do not match"


def test_rolling_mean_w48_simple_case():
    """Test rolling_mean_w48 on simple ascending sequence."""
    data = np.arange(100, dtype=np.float64)
    s = pd.Series(data)
    
    window = 48
    min_periods = 48
    
    expected = s.rolling(window=window, min_periods=min_periods).mean()
    got = rolling_mean_w48(data, min_periods=min_periods)
    
    assert np.allclose(got, expected.values, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_rolling_mean_w48_with_nans_and_infs():
    """Test rolling_mean_w48 with NaN and Inf in input."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf, 7.0, 8.0, 9.0, 10.0] * 50, dtype=np.float64)
    s = pd.Series(data)
    
    window = 48
    min_periods = 12
    
    expected = s.rolling(window=window, min_periods=min_periods).mean()
    got = rolling_mean_w48(data, min_periods=min_periods)
    
    # Compare non-NaN values
    mask = ~np.isnan(expected.values)
    if np.any(mask):
        assert np.allclose(got[mask], expected.values[mask], rtol=1e-12, atol=1e-12, equal_nan=False), \
            f"Max difference: {np.max(np.abs(got[mask] - expected.values[mask]))}"
    
    # Ensure NaN positions match (windows containing NaN/Inf should output NaN)
    assert np.all(np.isnan(got[~mask]) == np.isnan(expected.values[~mask])), \
        "NaN positions do not match"


def test_rolling_mean_w48_empty_input():
    """Test rolling_mean_w48 with empty input."""
    data = np.array([], dtype=np.float64)
    result = rolling_mean_w48(data, min_periods=48)
    
    assert len(result) == 0
    assert result.dtype == np.float64


def test_rolling_mean_w48_short_input():
    """Test rolling_mean_w48 with input shorter than min_periods."""
    data = np.arange(20, dtype=np.float64)
    result = rolling_mean_w48(data, min_periods=48)
    
    # All should be NaN since input is shorter than min_periods
    assert np.all(np.isnan(result))
    assert len(result) == 20
    assert result.dtype == np.float64


def test_rolling_mean_w48_exact_min_periods():
    """Test rolling_mean_w48 with input exactly min_periods long."""
    data = np.arange(48, dtype=np.float64)
    s = pd.Series(data)
    
    window = 48
    min_periods = 48
    
    expected = s.rolling(window=window, min_periods=min_periods).mean()
    got = rolling_mean_w48(data, min_periods=min_periods)
    
    # Last value should be non-NaN (mean of all 48 values)
    assert not np.isnan(got[-1])
    assert np.allclose(got, expected.values, rtol=1e-12, atol=1e-12, equal_nan=True)

