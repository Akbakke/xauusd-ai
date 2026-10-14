"""Numpy primitives used by the feature chain. Only `pct_change_np` has a caller
(gx1/features/basic_v1.py); the numba rolling variants were removed 14.10.2026 (git history).
"""
import numpy as np


def pct_change_np(x: np.ndarray, k: int) -> np.ndarray:
    """
    Equivalent to pandas Series.pct_change(k): x / x.shift(k) - 1
    
    Returns float64 array with NaN for first k positions.
    
    NaN/Inf policy: 
    - If x[i] or x[i-k] is not finite -> NaN at i
    - If denom==0 -> NaN (simplified, pandas returns inf/-inf but NaN is safer)
    - First k positions are always NaN (not enough history)
    
    Args:
        x: 1D numpy array (float)
        k: Periods to shift (must be >= 1)
    
    Returns:
        float64 array with percentage change values
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Explicit 1D input validation
    if x.ndim != 1:
        raise ValueError(f"pct_change_np requires 1D input, got ndim={x.ndim}, shape={x.shape}")
    
    n = len(x)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    
    out = np.full(n, np.nan, dtype=np.float64)
    
    if n <= k:
        # Not enough data for any valid calculation
        return out
    
    # Compute pct_change for positions k through n-1
    # Use direct indexing (no views) to avoid "too many indices" errors
    # idx = indices [k, k+1, ..., n-1] (length n-k)
    idx = np.arange(k, n, dtype=np.int64)
    
    # a = current values at idx, b = values at idx-k (shifted)
    a = x[idx]
    b = x[idx - k]
    
    # Create mask for valid (finite) values, excluding division by zero
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0.0)
    
    # Compute pct_change only for valid indices
    # Use safe division (b != 0 already in mask)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change_vals = (a[valid] - b[valid]) / b[valid]
    
    # Assign results using direct indexing (no views)
    out[idx[valid]] = pct_change_vals
    
    return out
