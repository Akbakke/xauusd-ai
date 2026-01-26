"""
Unit tests for pct_change_np function.
Tests equivalence with pandas Series.pct_change().
"""
import numpy as np
import pandas as pd
import pytest
from gx1.features.rolling_np import pct_change_np


def test_pct_change_np_random_data():
    """Test pct_change_np against pandas on random data."""
    np.random.seed(42)
    n = 1000
    data = np.random.randn(n) * 10 + 50  # Random values around 50
    
    for k in [1, 3, 5, 24]:
        # Pandas reference
        s = pd.Series(data)
        expected = s.pct_change(k).to_numpy()
        
        # NumPy implementation
        got = pct_change_np(data, k=k)
        
        # Compare NaN positions
        nan_mask_expected = np.isnan(expected)
        nan_mask_got = np.isnan(got)
        assert np.all(nan_mask_expected == nan_mask_got), \
            f"NaN positions do not match for k={k}. Expected NaN at: {np.where(nan_mask_expected)[0][:10]}, Got NaN at: {np.where(nan_mask_got)[0][:10]}"
        
        # Compare finite values
        finite_mask = ~nan_mask_expected
        if np.any(finite_mask):
            assert np.allclose(got[finite_mask], expected[finite_mask], rtol=1e-12, atol=1e-12, equal_nan=False), \
                f"Values do not match for k={k}. Max diff: {np.max(np.abs(got[finite_mask] - expected[finite_mask]))}"


def test_pct_change_np_simple_case():
    """Test pct_change_np on simple ascending sequence."""
    data = np.arange(1, 101, dtype=np.float64)  # [1, 2, 3, ..., 100]
    
    for k in [1, 3, 5]:
        # Pandas reference
        s = pd.Series(data)
        expected = s.pct_change(k).to_numpy()
        
        # NumPy implementation
        got = pct_change_np(data, k=k)
        
        # Compare (allowing for NaN in first k positions)
        assert np.allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True), \
            f"Simple case failed for k={k}"


def test_pct_change_np_with_nans_and_infs():
    """Test pct_change_np with NaN and Inf values."""
    data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf, 7.0, 8.0, 9.0, 10.0] * 100, dtype=np.float64)
    
    for k in [1, 3, 5]:
        # Pandas reference
        s = pd.Series(data)
        expected = s.pct_change(k).to_numpy()
        
        # NumPy implementation
        got = pct_change_np(data, k=k)
        
        # Compare NaN positions (must match exactly)
        nan_mask_expected = np.isnan(expected)
        nan_mask_got = np.isnan(got)
        assert np.all(nan_mask_expected == nan_mask_got), \
            f"NaN positions do not match for k={k} with NaN/Inf data"
        
        # Compare finite values
        finite_mask = ~nan_mask_expected & np.isfinite(expected) & np.isfinite(got)
        if np.any(finite_mask):
            assert np.allclose(got[finite_mask], expected[finite_mask], rtol=1e-12, atol=1e-12, equal_nan=False), \
                f"Values do not match for k={k} with NaN/Inf data"


def test_pct_change_np_division_by_zero():
    """Test pct_change_np behavior with division by zero (zero denominators)."""
    # Create data with zeros that will cause division by zero
    data = np.array([0.0, 1.0, 2.0, 0.0, 4.0, 5.0], dtype=np.float64)
    
    k = 3
    # Pandas reference
    s = pd.Series(data)
    expected = s.pct_change(k).to_numpy()
    
    # NumPy implementation
    got = pct_change_np(data, k=k)
    
    # Compare NaN positions
    nan_mask_expected = np.isnan(expected)
    nan_mask_got = np.isnan(got)
    assert np.all(nan_mask_expected == nan_mask_got), \
        "NaN positions do not match for division by zero case"
    
    # Compare finite and inf values
    finite_mask = ~nan_mask_expected & np.isfinite(expected) & np.isfinite(got)
    if np.any(finite_mask):
        assert np.allclose(got[finite_mask], expected[finite_mask], rtol=1e-12, atol=1e-12, equal_nan=False), \
            "Finite values do not match for division by zero case"
    
    # Check inf positions match (pandas returns inf for x/0 when x != 0)
    inf_mask_expected = np.isinf(expected)
    inf_mask_got = np.isinf(got)
    assert np.all(inf_mask_expected == inf_mask_got), \
        "Inf positions do not match for division by zero case"


def test_pct_change_np_empty_input():
    """Test pct_change_np with empty input."""
    data = np.array([], dtype=np.float64)
    expected = pd.Series(data).pct_change(1).to_numpy()
    got = pct_change_np(data, k=1)
    np.testing.assert_array_equal(got, expected)


def test_pct_change_np_short_input():
    """Test pct_change_np with input shorter than k."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    k = 5  # Longer than input
    
    expected = pd.Series(data).pct_change(k).to_numpy()
    got = pct_change_np(data, k=k)
    
    # All should be NaN
    assert np.all(np.isnan(got)) and np.all(np.isnan(expected)), \
        "Short input should result in all NaN"


def test_pct_change_np_invalid_k():
    """Test pct_change_np with invalid k values."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    # k < 1 should raise ValueError
    with pytest.raises(ValueError, match="k must be >= 1"):
        pct_change_np(data, k=0)
    
    with pytest.raises(ValueError, match="k must be >= 1"):
        pct_change_np(data, k=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

