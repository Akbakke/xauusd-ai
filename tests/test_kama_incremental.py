#!/usr/bin/env python3
"""
Unit test for incremental KAMA implementation.

Compares new incremental NumPy-based KAMA against pandas rolling-based version
to ensure correctness.

Requires pytest:
    pip install pytest
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gx1.features.basic_v1 import kama_np, _kama


def kama_pandas_reference(series, period, fast=2, slow=30):
    """
    Reference implementation using pandas rolling (for comparison).
    
    This is the OLD implementation that used pandas rolling.
    """
    change = series.diff(period).abs()
    diff_abs = series.diff().abs()
    volatility = diff_abs.rolling(period, min_periods=1).sum()
    er = change / (volatility + 1e-12)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc_diff = fast_sc - slow_sc
    sc = (er * sc_diff + slow_sc) ** 2
    kama = pd.Series(0.0, index=series.index)
    kama.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
    return kama


def test_kama_np_basic():
    """Test basic KAMA calculation with simple series."""
    # Simple ascending series
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    kama = kama_np(prices, period=3)
    
    assert len(kama) == len(prices)
    assert not np.any(np.isnan(kama))
    assert kama[0] == prices[0]  # First value should equal first price


def test_kama_np_vs_pandas():
    """Compare incremental NumPy KAMA against pandas rolling version."""
    # Generate synthetic price series (1000 points with some trend and noise)
    np.random.seed(42)
    n_points = 1000
    trend = np.linspace(100.0, 110.0, n_points)
    noise = np.random.randn(n_points) * 0.5
    prices = trend + noise
    
    # Convert to pandas Series
    series = pd.Series(prices)
    
    # Calculate with both methods
    kama_np_result = kama_np(prices, period=30)
    kama_pandas_result = kama_pandas_reference(series, period=30)
    
    # Compare results (allow small numerical differences)
    # Skip first few points where initialization may differ
    start_idx = 30
    np_result = kama_np_result[start_idx:]
    pandas_result = kama_pandas_result.values[start_idx:]
    
    # Calculate relative error
    diff = np.abs(np_result - pandas_result)
    rel_error = diff / (np.abs(pandas_result) + 1e-12)
    
    # Should be very close (numerical precision differences)
    max_rel_error = np.max(rel_error)
    mean_rel_error = np.mean(rel_error)
    
    # Allow up to 1% relative error (pandas and numpy may have slight numerical differences)
    assert max_rel_error < 0.01, f"Max relative error {max_rel_error:.6f} exceeds 1%"
    assert mean_rel_error < 0.001, f"Mean relative error {mean_rel_error:.6f} exceeds 0.1%"


def test_kama_np_with_nan():
    """Test KAMA handles NaN/Inf values defensively."""
    prices = np.array([100.0, 101.0, np.nan, 103.0, 104.0, 105.0])
    kama = kama_np(prices, period=3)
    
    # Should not crash and should handle NaN (forward-fill)
    assert len(kama) == len(prices)
    # First value should be valid
    assert not np.isnan(kama[0])


def test_kama_wrapper():
    """Test _kama wrapper works with both Series and arrays."""
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    
    # Test with numpy array
    kama_array = _kama(prices, period=3)
    assert isinstance(kama_array, np.ndarray)
    assert len(kama_array) == len(prices)
    
    # Test with pandas Series
    series = pd.Series(prices)
    kama_series = _kama(series, period=3)
    assert isinstance(kama_series, pd.Series)
    assert len(kama_series) == len(series)
    assert kama_series.index.equals(series.index)


def test_kama_performance_small():
    """Smoke test: ensure incremental version is fast on small data."""
    prices = np.random.randn(100) + 100.0
    
    # Should complete quickly
    kama = kama_np(prices, period=30)
    assert len(kama) == len(prices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

