"""
Generic NumPy rolling timer wrapper (NO PANDAS).

DEL 2: All rolling operations use NumPy directly - no pandas fallback.
Provides a wrapper function to automatically time NumPy rolling operations
and report to PerfCollector via perf_add/perf_inc.

Usage:
    from gx1.features.rolling_timer import timed_rolling
    
    # Instead of: series.rolling(20).mean()
    result = timed_rolling(series, 20, "mean")
"""

import os
import time
from typing import Union

import numpy as np
import pandas as pd

from gx1.utils.perf_timer import perf_add, perf_inc


def timed_rolling(
    series: Union[pd.Series, pd.DataFrame],
    window: int,
    operation: str,
    min_periods: int = None,
    **kwargs
) -> Union[pd.Series, pd.DataFrame]:
    """
    Time a NumPy rolling operation and report to PerfCollector (NO PANDAS).
    
    DEL 2: All operations use NumPy directly - no pandas fallback.
    
    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Input series/dataframe (will convert to NumPy array)
    window : int
        Rolling window size
    operation : str
        Operation name: "mean" or "std" (only these supported for now)
    min_periods : int, optional
        Minimum periods for rolling window (default: window)
    **kwargs
        Additional arguments (e.g., ddof for std)
    
    Returns
    -------
    Result of the rolling operation (pd.Series or pd.DataFrame)
    """
    # DEL 2: Always use NumPy - no pandas fallback
    from gx1.features.rolling_np import rolling_mean, rolling_std
    
    if min_periods is None:
        min_periods = window
    
    # Extract ddof from kwargs if present (for std)
    ddof = kwargs.get('ddof', 0)
    
    # Time NumPy rolling
    t_start = time.perf_counter()
    try:
        if isinstance(series, pd.Series):
            arr = series.to_numpy(dtype=np.float64)
            index = series.index
            
            if operation == "mean":
                result_arr = rolling_mean(arr, window, min_periods=min_periods)
            elif operation == "std":
                result_arr = rolling_std(arr, window, min_periods=min_periods, ddof=ddof)
            else:
                raise ValueError(f"Unsupported operation: {operation} (only 'mean' and 'std' supported)")
            
            result = pd.Series(result_arr, index=index, dtype=np.float64)
        else:
            # DataFrame case: apply to each column
            result_df = pd.DataFrame(index=series.index, columns=series.columns)
            for col in series.columns:
                arr = series[col].to_numpy(dtype=np.float64)
                if operation == "mean":
                    result_arr = rolling_mean(arr, window, min_periods=min_periods)
                elif operation == "std":
                    result_arr = rolling_std(arr, window, min_periods=min_periods, ddof=ddof)
                else:
                    raise ValueError(f"Unsupported operation: {operation} (only 'mean' and 'std' supported)")
                result_df[col] = pd.Series(result_arr, index=series.index, dtype=np.float64)
            result = result_df
    finally:
        t_end = time.perf_counter()
        # Report timing with window size
        perf_add(f"rolling.numpy.{operation}.w{window}", t_end - t_start)
        perf_inc(f"rolling.numpy.{operation}.w{window}")
    return result

