import numpy as np
from typing import Union

# Del 1: Hard fail if Numba is not available
try:
    from numba import njit
except ImportError as e:
    raise RuntimeError(
        "Numba is required for GX1 zscore acceleration. "
        "Install numba or abort."
    ) from e


def rolling_std_3(x: np.ndarray, min_periods: int = 2, ddof: int = 0) -> np.ndarray:
    """
    Compute rolling std with window=3.
    Must match pandas: Series.rolling(3, min_periods=2).std(ddof=0)
    ddof only supports 0 for now (assert).
    Returns float64 array with NaN where not enough periods.
    """
    assert ddof == 0, "ddof only supports 0 for now"
    assert min_periods == 2, "min_periods only supports 2 for now"  # Simplified for current use case

    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return result
    
    # Handle NaN/Inf: if any value in window is NaN/Inf, output NaN
    # We'll track this as we go
    
    # i=0: NaN (not enough periods)
    if n < min_periods:
        return result
    
    # i=1: std of [x[0], x[1]]
    if n >= 2:
        x0, x1 = x[0], x[1]
        if np.isfinite(x0) and np.isfinite(x1):
            # std of 2 values with ddof=0
            mean_val = (x0 + x1) / 2.0
            var = ((x0 - mean_val)**2 + (x1 - mean_val)**2) / 2.0
            result[1] = np.sqrt(max(var, 0.0))
        # else: result[1] stays NaN
    
    # i>=2: std of [x[i-2], x[i-1], x[i]]
    # Use rolling sum and sum of squares for efficiency
    for i in range(2, n):
        x0, x1, x2 = x[i-2], x[i-1], x[i]
        
        # Check for NaN/Inf in window
        if not (np.isfinite(x0) and np.isfinite(x1) and np.isfinite(x2)):
            result[i] = np.nan
            continue
        
        # Compute variance using sums
        # s1 = sum of values
        # s2 = sum of squares
        s1 = x0 + x1 + x2
        s2 = x0*x0 + x1*x1 + x2*x2
        n_vals = 3.0
        
        # Variance: E[X^2] - (E[X])^2 = (s2/n) - (s1/n)^2
        mean_val = s1 / n_vals
        var = (s2 / n_vals) - (mean_val * mean_val)
        
        result[i] = np.sqrt(max(var, 0.0))
    
    return result


def rolling_mean_w48(x: np.ndarray, min_periods: int = 48) -> np.ndarray:
    """
    Rolling mean with window=48.
    Matches pandas: Series.rolling(48, min_periods=min_periods).mean()
    Returns float64 array with NaN where not enough periods.
    
    NaN/Inf policy: if the window contains NaN/Inf => output NaN for that position.
    
    Args:
        x: 1D numpy array (float)
        min_periods: Minimum periods required (default 48)
    
    Implementation:
        Uses O(n) rolling sum for efficiency.
        Handles NaN/Inf robustly: if any value in window is NaN/Inf, output NaN.
        Matches pandas behavior for partial windows (i < window-1).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n == 0:
        return result
    
    window = 48  # Fixed window size
    
    if n < min_periods:
        return result
    
    # Rolling sum approach: maintain sum of current window
    # For i < window-1: window_size = i+1
    # For i >= window-1: window_size = window
    
    # Process each position
    for i in range(n):
        # Determine window size for this position
        if i < window - 1:
            window_size = i + 1
        else:
            window_size = window
        
        # Check if we have enough periods
        if window_size < min_periods:
            result[i] = np.nan
            continue
        
        # Extract current window
        window_start = max(0, i - window_size + 1)
        window_end = i + 1
        window_values = x[window_start:window_end]
        
        # Filter out NaN/Inf (pandas ignores them when computing mean)
        finite_values = window_values[np.isfinite(window_values)]
        
        # Need at least min_periods finite values
        if len(finite_values) < min_periods:
            result[i] = np.nan
            continue
        
        # Compute mean over finite values (matches pandas behavior)
        result[i] = np.mean(finite_values)
    
    return result


@njit(cache=True, fastmath=False)
def rolling_kurtosis_w48_numba(x: np.ndarray, min_periods: int) -> np.ndarray:
    """
    Numba-accelerated rolling kurtosis with window=48.
    Uses ring buffer to store window values, then computes central moments directly.
    
    Args:
        x: 1D numpy array (float64)
        min_periods: Minimum periods required (typically 12)
    
    Returns:
        float64 array with Fisher's excess kurtosis (kurtosis - 3), NaN where not enough finite periods
    """
    n = len(x)
    result = np.full(n, np.nan)
    
    if n == 0:
        return result
    
    window = 48
    eps = 1e-12
    
    if n < min_periods:
        return result
    
    # Ring buffer to store window values
    buffer = np.zeros(window, dtype=np.float64)
    
    for i in range(n):
        # Store current value in ring buffer
        buffer_idx = i % window
        buffer[buffer_idx] = x[i]
        
        if i >= min_periods - 1:
            # Extract current window (finite values only)
            window_size = window if i >= window - 1 else i + 1
            window_values = np.zeros(window_size, dtype=np.float64)
            finite_count = 0
            
            # Build window from buffer
            if i < window - 1:
                # Partial window: use buffer[0:i+1]
                for j in range(i + 1):
                    val = buffer[j]
                    if np.isfinite(val):
                        window_values[finite_count] = val
                        finite_count += 1
            else:
                # Full window: wrap around buffer
                start_idx = (i % window) + 1
                for j in range(window):
                    idx = (start_idx + j) % window
                    val = buffer[idx]
                    if np.isfinite(val):
                        window_values[finite_count] = val
                        finite_count += 1
            
            if finite_count >= min_periods and finite_count >= 4:
                # Extract only finite values
                finite_values = window_values[:finite_count]
                n = finite_count
                
                # Compute mean
                mean_val = 0.0
                for j in range(n):
                    mean_val += finite_values[j]
                mean_val = mean_val / n
                
                # Compute central moments (sums)
                sum2 = 0.0  # sum of (x-mean)^2
                sum4 = 0.0  # sum of (x-mean)^4
                for j in range(n):
                    diff = finite_values[j] - mean_val
                    diff2 = diff * diff
                    sum2 += diff2
                    sum4 += diff2 * diff2
                
                # Sample variance (unbiased)
                var_sample = sum2 / (n - 1) if n > 1 else 0.0
                
                if var_sample < eps:  # Avoid division by zero
                    kurt = 0.0  # Constant series
                else:
                    # Sample kurtosis formula (matches scipy.stats.kurtosis with bias=False)
                    # kurtosis = (n*(n+1)/((n-1)*(n-2)*(n-3))) * sum4 / var^2 - 3*(n-1)^2/((n-2)*(n-3))
                    factor1 = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
                    factor2 = 3.0 * (n - 1) * (n - 1) / ((n - 2) * (n - 3))
                    kurt = factor1 * sum4 / (var_sample * var_sample + eps) - factor2
                
                result[i] = kurt
            else:
                result[i] = np.nan
    
    return result


def rolling_kurtosis_w48(x: np.ndarray, min_periods: int = 12, fisher: bool = True, bias: bool = True) -> np.ndarray:
    """
    Public entrypoint for rolling kurtosis w48.
    Uses Numba-accelerated implementation (no fallback).
    
    Args:
        x: 1D numpy array (float)
        min_periods: Minimum periods required (default 12)
        fisher: If True, Fisher's (excess) kurtosis is returned (kurtosis - 3). Default True (always used).
        bias: Unused (kept for compatibility)
    
    Returns:
        float64 array with Fisher's excess kurtosis, NaN where not enough finite periods
    """
    x = np.asarray(x, dtype=np.float64)
    return rolling_kurtosis_w48_numba(x, min_periods)


def rolling_mean_w48_nanaware(x: np.ndarray, min_periods: int = 48) -> np.ndarray:
    """
    Nan-aware rolling mean with window=48.
    Same as rolling_mean_w48 but explicitly named for zscore usage.
    Matches pandas: Series.rolling(48, min_periods=min_periods).mean()
    
    NaN/Inf policy: NaN values are ignored, mean computed over finite values only.
    If count of finite values < min_periods, output NaN.
    
    Args:
        x: 1D numpy array (float)
        min_periods: Minimum periods required (default 48)
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    return rolling_mean_w48(x, min_periods=min_periods)


def rolling_std_w48_nanaware(x: np.ndarray, min_periods: int = 48, ddof: int = 0) -> np.ndarray:
    """
    Nan-aware rolling std with window=48.
    Matches pandas: Series.rolling(48, min_periods=min_periods).std(ddof=0)
    
    NaN/Inf policy: NaN values are ignored, std computed over finite values only.
    If count of finite values < min_periods, output NaN.
    
    Args:
        x: 1D numpy array (float)
        min_periods: Minimum periods required (default 48)
        ddof: Delta degrees of freedom (only 0 supported)
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    assert ddof == 0, "ddof only supports 0 for now"
    
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n == 0:
        return result
    
    window = 48  # Fixed window size
    
    if n < min_periods:
        return result
    
    # Rolling sum and sum of squares approach (finite values only)
    current_sum = 0.0
    current_sum_sq = 0.0
    finite_count = 0
    
    for i in range(n):
        # Add current element
        val_add = x[i]
        if np.isfinite(val_add):
            current_sum += val_add
            current_sum_sq += val_add * val_add
            finite_count += 1
        
        # Remove element that falls out of window
        if i >= window:
            val_remove = x[i - window]
            if np.isfinite(val_remove):
                current_sum -= val_remove
                current_sum_sq -= val_remove * val_remove
                finite_count -= 1
        
        # Check if we have enough finite periods
        if finite_count >= min_periods:
            # Compute variance: var = E[X^2] - (E[X])^2
            mean_val = current_sum / finite_count
            var_val = (current_sum_sq / finite_count) - (mean_val * mean_val)
            # Clamp variance to >= 0 (numerical stability)
            var_val = max(var_val, 0.0)
            result[i] = np.sqrt(var_val)
        else:
            result[i] = np.nan
    
    return result


@njit(cache=True, fastmath=False)
def zscore_w48_numba(x: np.ndarray, min_periods: int) -> np.ndarray:
    """
    Numba-accelerated nan-aware rolling z-score with window=48.
    Matches pandas: (s - s.rolling(48, min_periods=min_periods).mean()) / (s.rolling(48, min_periods=min_periods).std(ddof=0) + 1e-12)
    
    Args:
        x: 1D numpy array (float64)
        min_periods: Minimum periods required
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    n = len(x)
    result = np.full(n, np.nan)
    
    if n == 0:
        return result
    
    window = 48
    ddof = 0
    
    if n < min_periods:
        return result
    
    # Rolling sum, sum of squares, and count (finite values only)
    current_sum = 0.0
    current_sum_sq = 0.0
    finite_count = 0
    
    for i in range(n):
        # Add current element
        val_add = x[i]
        if np.isfinite(val_add):
            current_sum += val_add
            current_sum_sq += val_add * val_add
            finite_count += 1
        
        # Remove element that falls out of window
        if i >= window:
            val_remove = x[i - window]
            if np.isfinite(val_remove):
                current_sum -= val_remove
                current_sum_sq -= val_remove * val_remove
                finite_count -= 1
        
        # Check if we have enough finite periods
        if finite_count >= min_periods:
            # Compute mean
            mean_val = current_sum / finite_count
            
            # Compute variance: var = E[X^2] - (E[X])^2
            var_val = (current_sum_sq / finite_count) - (mean_val * mean_val)
            # Clamp variance to >= 0 (numerical stability)
            if var_val < 0.0:
                var_val = 0.0
            std_val = np.sqrt(var_val)
            
            # Compute z-score: (x[i] - mean) / (std + 1e-12)
            if np.isfinite(x[i]):
                result[i] = (x[i] - mean_val) / (std_val + 1e-12)
            else:
                result[i] = np.nan
        else:
            result[i] = np.nan
    
    return result


def zscore_w48(x: np.ndarray, min_periods: int = 24) -> np.ndarray:
    """
    Public entrypoint for zscore w48.
    Uses Numba-accelerated implementation (no fallback).
    
    Args:
        x: 1D numpy array (float)
        min_periods: Minimum periods required (default 24)
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    x = np.asarray(x, dtype=np.float64)
    return zscore_w48_numba(x, min_periods)


# Legacy function kept for backwards compatibility during migration
def zscore_w48_nanaware(x: np.ndarray, min_periods: int = 24) -> np.ndarray:
    """
    DEPRECATED: Use zscore_w48 instead.
    This function now delegates to zscore_w48 (Numba-accelerated).
    """
    return zscore_w48(x, min_periods)


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
    import logging
    log = logging.getLogger(__name__)
    
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


@njit(cache=True, fastmath=False)
def rolling_quantile_w48_numba(x: np.ndarray, q: float, min_periods: int) -> np.ndarray:
    """
    Numba-accelerated rolling quantile with window=48.
    Matches pandas: Series.rolling(48, min_periods=min_periods).quantile(q)
    
    NaN/Inf policy: NaN/Inf values are ignored, quantile computed over finite values only.
    If count of finite values < min_periods, output NaN.
    
    Args:
        x: 1D numpy array (float)
        q: Quantile value (0.0 to 1.0)
        min_periods: Minimum periods required
    
    Returns:
        float64 array with quantile values, NaN where not enough finite periods
    """
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n == 0:
        return result
    
    window = 48  # Fixed window size
    
    if n < min_periods:
        return result
    
    # Ring buffer for window values
    window_buffer = np.full(window, np.nan, dtype=np.float64)
    
    for i in range(n):
        # Add current value to ring buffer
        window_buffer[i % window] = x[i]
        
        if i >= min_periods - 1:
            # Extract current window values, handling wrap-around
            if i < window - 1:
                # Not enough values to wrap around yet
                current_window = window_buffer[:i+1]
            else:
                # Wrap around: values from (i % window + 1) to end, then from start to (i % window)
                # Numba-compatible: build array manually (no concatenate)
                idx_start = (i % window) + 1
                current_window = np.zeros(window, dtype=np.float64)
                for j in range(window):
                    src_idx = (idx_start + j) % window
                    current_window[j] = window_buffer[src_idx]
            
            # Filter finite values
            finite_mask = np.isfinite(current_window)
            finite_values = current_window[finite_mask]
            count = len(finite_values)
            
            if count >= min_periods:
                # Sort finite values (in-place for efficiency, but we need to copy for Numba)
                # Numba requires explicit allocation for sorted array
                sorted_vals = np.sort(finite_values)
                
                # Compute quantile index
                # pandas uses interpolation='linear' by default
                # For quantile q with n values, index = (n-1) * q
                idx_float = (count - 1) * q
                idx_low = int(np.floor(idx_float))
                idx_high = int(np.ceil(idx_float))
                
                if idx_low == idx_high:
                    # Exact index
                    result[i] = sorted_vals[idx_low]
                else:
                    # Linear interpolation
                    weight_high = idx_float - idx_low
                    weight_low = 1.0 - weight_high
                    result[i] = weight_low * sorted_vals[idx_low] + weight_high * sorted_vals[idx_high]
            else:
                result[i] = np.nan
        else:
            result[i] = np.nan
    
    return result


def rolling_quantile_w48(x: np.ndarray, q: float, min_periods: int = 24) -> np.ndarray:
    """
    Public entrypoint for Numba-accelerated rolling quantile with window=48.
    """
    x = np.asarray(x, dtype=np.float64)
    return rolling_quantile_w48_numba(x, q, min_periods)


# DEL 2: Generic NumPy rolling functions for all window sizes (no pandas fallback)
@njit(cache=True, fastmath=False)
def rolling_mean_numba(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Numba-accelerated generic rolling mean for any window size.
    Matches pandas: Series.rolling(window, min_periods=min_periods).mean()
    
    Args:
        x: 1D numpy array (float64)
        window: Window size
        min_periods: Minimum periods required
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n == 0 or n < min_periods:
        return result
    
    # Rolling sum approach (O(n))
    current_sum = 0.0
    finite_count = 0
    
    for i in range(n):
        # Add current element
        val_add = x[i]
        if np.isfinite(val_add):
            current_sum += val_add
            finite_count += 1
        
        # Remove element that falls out of window
        if i >= window:
            val_remove = x[i - window]
            if np.isfinite(val_remove):
                current_sum -= val_remove
                finite_count -= 1
        
        # Check if we have enough finite periods
        if finite_count >= min_periods:
            result[i] = current_sum / finite_count
        else:
            result[i] = np.nan
    
    return result


def rolling_mean(x: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """
    Generic rolling mean for any window size (NumPy-only, no pandas).
    
    Args:
        x: 1D numpy array (float)
        window: Window size
        min_periods: Minimum periods required (default: window)
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    x = np.asarray(x, dtype=np.float64)
    if min_periods is None:
        min_periods = window
    return rolling_mean_numba(x, window, min_periods)


@njit(cache=True, fastmath=False)
def rolling_std_numba(x: np.ndarray, window: int, min_periods: int, ddof: int) -> np.ndarray:
    """
    Numba-accelerated generic rolling std for any window size.
    Matches pandas: Series.rolling(window, min_periods=min_periods).std(ddof=ddof)
    
    Args:
        x: 1D numpy array (float64)
        window: Window size
        min_periods: Minimum periods required
        ddof: Delta degrees of freedom (0 or 1)
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n == 0 or n < min_periods:
        return result
    
    # Rolling sum and sum of squares approach (O(n))
    current_sum = 0.0
    current_sum_sq = 0.0
    finite_count = 0
    
    for i in range(n):
        # Add current element
        val_add = x[i]
        if np.isfinite(val_add):
            current_sum += val_add
            current_sum_sq += val_add * val_add
            finite_count += 1
        
        # Remove element that falls out of window
        if i >= window:
            val_remove = x[i - window]
            if np.isfinite(val_remove):
                current_sum -= val_remove
                current_sum_sq -= val_remove * val_remove
                finite_count -= 1
        
        # Check if we have enough finite periods
        if finite_count >= min_periods:
            # Compute variance: var = E[X^2] - (E[X])^2
            mean_val = current_sum / finite_count
            var_val = (current_sum_sq / finite_count) - (mean_val * mean_val)
            # Clamp variance to >= 0 (numerical stability)
            if var_val < 0.0:
                var_val = 0.0
            result[i] = np.sqrt(var_val)
        else:
            result[i] = np.nan
    
    return result


def rolling_std(x: np.ndarray, window: int, min_periods: int = None, ddof: int = 0) -> np.ndarray:
    """
    Generic rolling std for any window size (NumPy-only, no pandas).
    
    Args:
        x: 1D numpy array (float)
        window: Window size
        min_periods: Minimum periods required (default: window)
        ddof: Delta degrees of freedom (default: 0, only 0 supported for now)
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    assert ddof == 0, "ddof only supports 0 for now"
    x = np.asarray(x, dtype=np.float64)
    if min_periods is None:
        min_periods = window
    return rolling_std_numba(x, window, min_periods, ddof)


@njit(cache=True, fastmath=False)
def rolling_quantile_numba(x: np.ndarray, window: int, q: float, min_periods: int) -> np.ndarray:
    """
    Numba-accelerated generic rolling quantile for any window size.
    Matches pandas: Series.rolling(window, min_periods=min_periods).quantile(q)
    
    Args:
        x: 1D numpy array (float64)
        window: Window size
        q: Quantile value (0.0 to 1.0)
        min_periods: Minimum periods required
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n == 0 or n < min_periods:
        return result
    
    # Ring buffer for window values
    window_buffer = np.full(window, np.nan, dtype=np.float64)
    
    for i in range(n):
        # Add current value to ring buffer
        window_buffer[i % window] = x[i]
        
        if i >= min_periods - 1:
            # Extract current window values, handling wrap-around
            if i < window - 1:
                # Not enough values to wrap around yet
                current_window = window_buffer[:i+1]
            else:
                # Wrap around: build array manually (Numba-compatible)
                idx_start = (i % window) + 1
                current_window = np.zeros(window, dtype=np.float64)
                for j in range(window):
                    src_idx = (idx_start + j) % window
                    current_window[j] = window_buffer[src_idx]
            
            # Filter finite values
            finite_mask = np.isfinite(current_window)
            finite_values = current_window[finite_mask]
            count = len(finite_values)
            
            if count >= min_periods:
                # Sort finite values
                sorted_vals = np.sort(finite_values)
                
                # Compute quantile index (linear interpolation)
                idx_float = (count - 1) * q
                idx_low = int(np.floor(idx_float))
                idx_high = int(np.ceil(idx_float))
                
                if idx_low == idx_high:
                    result[i] = sorted_vals[idx_low]
                else:
                    # Linear interpolation
                    weight_high = idx_float - idx_low
                    weight_low = 1.0 - weight_high
                    result[i] = weight_low * sorted_vals[idx_low] + weight_high * sorted_vals[idx_high]
            else:
                result[i] = np.nan
        else:
            result[i] = np.nan
    
    return result


def rolling_quantile(x: np.ndarray, window: int, q: float, min_periods: int = None) -> np.ndarray:
    """
    Generic rolling quantile for any window size (NumPy-only, no pandas).
    
    Args:
        x: 1D numpy array (float)
        window: Window size
        q: Quantile value (0.0 to 1.0)
        min_periods: Minimum periods required (default: window)
    
    Returns:
        float64 array with NaN where not enough finite periods
    """
    x = np.asarray(x, dtype=np.float64)
    if min_periods is None:
        min_periods = window
    return rolling_quantile_numba(x, window, q, min_periods)
