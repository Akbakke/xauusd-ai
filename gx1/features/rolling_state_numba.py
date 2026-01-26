"""
Incremental rolling quantile state for window=48.
Hard requirement: Numba only, no fallback.
"""
import numpy as np

# Hard fail if Numba is not available
from numba import njit

@njit(cache=True, fastmath=False)
def rq48_step(
    buf: np.ndarray,
    tmp: np.ndarray,
    pos: int,
    filled: int,
    last_close: float,
    close_now: float,
    min_periods: int,
    q10: float,
    q90: float,
    prev_q10: float,
    prev_q90: float,
) -> tuple:
    """
    Update pct_change r1, push into ring buffer, compute current q10/q90 on finite values in window.
    
    Parameters
    ----------
    buf : np.ndarray[48]
        Ring buffer for r1 values (float64)
    tmp : np.ndarray[48]
        Workspace for sorting (float64)
    pos : int
        Current position in ring buffer (0..47)
    filled : int
        Number of values filled in buffer (0..48)
    last_close : float
        Previous close price (for pct_change)
    close_now : float
        Current close price
    min_periods : int
        Minimum periods required
    q10 : float
        Quantile value for q10 (0.10)
    q90 : float
        Quantile value for q90 (0.90)
    prev_q10 : float
        Previous q10 value (for shift(1) output)
    prev_q90 : float
        Previous q90 value (for shift(1) output)
    
    Returns
    -------
    tuple:
        (new_pos, new_filled, new_last_close,
         out_q10_shifted, out_q90_shifted,
         new_prev_q10, new_prev_q90)
    """
    # Compute r1_now (pct_change)
    if not np.isfinite(last_close) or last_close == 0.0:
        r1_now = 0.0
    else:
        r1_now = (close_now / last_close) - 1.0
        if not np.isfinite(r1_now):
            r1_now = 0.0
    
    # Push r1_now into ring buffer
    buf[pos] = r1_now
    new_pos = (pos + 1) % 48
    new_filled = min(filled + 1, 48)
    new_last_close = close_now
    
    # Extract current window values (handle wrap-around)
    k = 0
    if new_filled < 48:
        # Not enough values to wrap around yet
        for i in range(new_filled):
            val = buf[i]
            if np.isfinite(val):
                tmp[k] = val
                k += 1
    else:
        # Wrap around: values from new_pos to end, then from start to new_pos
        for i in range(new_pos, 48):
            val = buf[i]
            if np.isfinite(val):
                tmp[k] = val
                k += 1
        for i in range(new_pos):
            val = buf[i]
            if np.isfinite(val):
                tmp[k] = val
                k += 1
    
    # Compute current quantiles
    if k < min_periods:
        cur_q10 = np.nan
        cur_q90 = np.nan
    else:
        # Sort finite values (in-place on tmp[:k])
        # Simple bubble sort for small k (k <= 48)
        for i in range(k):
            for j in range(i + 1, k):
                if tmp[i] > tmp[j]:
                    # Swap
                    temp = tmp[i]
                    tmp[i] = tmp[j]
                    tmp[j] = temp
        
        # Compute q10 (linear interpolation, pandas-like)
        p10 = q10 * (k - 1)
        lo10 = int(np.floor(p10))
        hi10 = int(np.ceil(p10))
        if lo10 == hi10:
            cur_q10 = tmp[lo10]
        else:
            weight = p10 - lo10
            cur_q10 = tmp[lo10] + weight * (tmp[hi10] - tmp[lo10])
        
        # Compute q90 (linear interpolation, pandas-like)
        p90 = q90 * (k - 1)
        lo90 = int(np.floor(p90))
        hi90 = int(np.ceil(p90))
        if lo90 == hi90:
            cur_q90 = tmp[lo90]
        else:
            weight = p90 - lo90
            cur_q90 = tmp[lo90] + weight * (tmp[hi90] - tmp[lo90])
    
    # SHIFT(1) output: output for bar t is quantile from bar t-1
    out_q10_shifted = prev_q10 if np.isfinite(prev_q10) else 0.0
    out_q90_shifted = prev_q90 if np.isfinite(prev_q90) else 0.0
    
    # Update prev values for next iteration
    new_prev_q10 = cur_q10
    new_prev_q90 = cur_q90
    
    return (new_pos, new_filled, new_last_close,
            out_q10_shifted, out_q90_shifted,
            new_prev_q10, new_prev_q90)


class RollingR1Quantiles48State:
    """
    Stateful incremental rolling quantile calculator for window=48.
    Hard requirement: Numba only, no fallback.
    """
    
    def __init__(self, min_periods: int = 24, q10: float = 0.10, q90: float = 0.90):
        self.buf = np.full(48, np.nan, dtype=np.float64)
        self.tmp = np.empty(48, dtype=np.float64)
        self.pos = 0
        self.filled = 0
        self.last_close = np.nan
        self.prev_q10 = np.nan
        self.prev_q90 = np.nan
        self.min_periods = min_periods
        self.q10 = q10
        self.q90 = q90
    
    def update(self, close_now: float) -> tuple[float, float]:
        """
        Update state with new close price and return shifted quantiles.
        
        Parameters
        ----------
        close_now : float
            Current close price
            
        Returns
        -------
        tuple[float, float]
            (q10_shifted, q90_shifted) - quantiles from previous bar (shift(1) semantics)
        """
        (self.pos, self.filled, self.last_close,
         out_q10, out_q90,
         self.prev_q10, self.prev_q90) = rq48_step(
            self.buf, self.tmp, self.pos, self.filled,
            self.last_close, close_now,
            self.min_periods, self.q10, self.q90,
            self.prev_q10, self.prev_q90
        )
        return (out_q10, out_q90)



