"""
Unit test for incremental rolling quantile state (window=48).
Tests that incremental state matches batch reference (pandas).
Hard correctness test: if this fails, system is wrong.
"""
import numpy as np
import pandas as pd
import pytest

from gx1.features.rolling_state_numba import RollingR1Quantiles48State


def test_incremental_vs_batch_reference():
    """Test incremental state step-by-step vs batch reference."""
    np.random.seed(42)
    n = 500
    close_series = 100.0 + np.cumsum(np.random.randn(n) * 0.5)  # Random walk
    
    # Incremental path
    state = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    incremental_q10 = []
    incremental_q90 = []
    
    for close_now in close_series:
        q10, q90 = state.update(close_now)
        incremental_q10.append(q10)
        incremental_q90.append(q90)
    
    incremental_q10 = np.array(incremental_q10)
    incremental_q90 = np.array(incremental_q90)
    
    # Batch reference (pandas)
    s = pd.Series(close_series)
    r1 = s.pct_change().fillna(0.0)
    q10_batch = r1.rolling(48, min_periods=24).quantile(0.10).shift(1).fillna(0.0)
    q90_batch = r1.rolling(48, min_periods=24).quantile(0.90).shift(1).fillna(0.0)
    
    # Compare (skip first 24 values where batch has NaN)
    mask = np.isfinite(q10_batch.values) & np.isfinite(incremental_q10)
    if np.any(mask):
        assert np.allclose(
            incremental_q10[mask],
            q10_batch.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"q10 mismatch. Max diff: {np.max(np.abs(incremental_q10[mask] - q10_batch.values[mask]))}"
    
    mask = np.isfinite(q90_batch.values) & np.isfinite(incremental_q90)
    if np.any(mask):
        assert np.allclose(
            incremental_q90[mask],
            q90_batch.values[mask],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False
        ), f"q90 mismatch. Max diff: {np.max(np.abs(incremental_q90[mask] - q90_batch.values[mask]))}"


def test_incremental_nan_inf_edge_cases():
    """Test incremental state with NaN/Inf/0 close values."""
    state = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    
    # Test with NaN close
    q10, q90 = state.update(np.nan)
    assert np.isfinite(q10) and np.isfinite(q90), "NaN close should produce finite output (0.0 policy)"
    assert q10 == 0.0 and q90 == 0.0, "NaN close should produce 0.0 output"
    
    # Test with Inf close
    state2 = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    state2.update(100.0)  # Initialize with valid value
    q10, q90 = state2.update(np.inf)
    assert np.isfinite(q10) and np.isfinite(q90), "Inf close should produce finite output"
    
    # Test with 0 close (division by zero in pct_change)
    state3 = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    state3.update(100.0)  # Initialize
    q10, q90 = state3.update(0.0)
    assert np.isfinite(q10) and np.isfinite(q90), "0 close should produce finite output"
    
    # Test sequence with mixed valid/invalid
    state4 = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    closes = [100.0, 101.0, np.nan, 102.0, np.inf, 103.0, 0.0, 104.0]
    for close_val in closes:
        q10, q90 = state4.update(close_val)
        assert np.isfinite(q10) and np.isfinite(q90), f"All outputs should be finite, got q10={q10}, q90={q90} for close={close_val}"


def test_incremental_min_periods():
    """Test incremental state respects min_periods."""
    state = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    
    # First 23 updates should produce 0.0 (shift(1) from NaN)
    for i in range(23):
        close_val = 100.0 + i * 0.1
        q10, q90 = state.update(close_val)
        assert q10 == 0.0 and q90 == 0.0, f"First {i+1} updates should produce 0.0 (min_periods=24)"
    
    # After min_periods, should start producing non-zero values
    close_val = 100.0 + 23 * 0.1
    q10, q90 = state.update(close_val)
    # At this point, we have 24 values, so prev_q10/prev_q90 should be computed
    # But output is shift(1), so this output is from previous bar (which had < 24)
    # So it should still be 0.0
    assert q10 == 0.0 and q90 == 0.0, "24th update output should still be 0.0 (shift(1) from bar 23)"
    
    # Next update should have non-zero (shift(1) from bar 24 which had 24 values)
    close_val = 100.0 + 24 * 0.1
    q10, q90 = state.update(close_val)
    # This should be shift(1) from bar 24, which had 24 values, so should be computed
    # But might still be 0.0 if all r1 values are similar
    assert np.isfinite(q10) and np.isfinite(q90), "25th update should produce finite values"


def test_incremental_deterministic():
    """Test incremental state is deterministic (same input → same output)."""
    np.random.seed(43)
    closes = 100.0 + np.cumsum(np.random.randn(100) * 0.5)
    
    # Run 1
    state1 = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    results1_q10 = []
    results1_q90 = []
    for close_val in closes:
        q10, q90 = state1.update(close_val)
        results1_q10.append(q10)
        results1_q90.append(q90)
    
    # Run 2 (same input)
    state2 = RollingR1Quantiles48State(min_periods=24, q10=0.10, q90=0.90)
    results2_q10 = []
    results2_q90 = []
    for close_val in closes:
        q10, q90 = state2.update(close_val)
        results2_q10.append(q10)
        results2_q90.append(q90)
    
    # Results should be identical
    assert np.allclose(results1_q10, results2_q10, rtol=1e-10, atol=1e-10), "q10 should be deterministic"
    assert np.allclose(results1_q90, results2_q90, rtol=1e-10, atol=1e-10), "q90 should be deterministic"



