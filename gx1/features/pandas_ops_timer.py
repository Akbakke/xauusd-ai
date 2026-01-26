"""
Generic pandas rolling timer wrapper with callsite tagging for misc_roll.

Del 2: Provides wrapper functions to automatically time pandas rolling operations
and report to PerfCollector with callsite information.
"""

import inspect
import os
import time
from typing import Union, Optional

import numpy as np
import pandas as pd

from gx1.utils.perf_timer import perf_add, perf_inc


def timed_pandas_rolling(
    series: Union[pd.Series, pd.DataFrame],
    win: int,
    min_periods: int,
    op: str,
    **kwargs
) -> Union[pd.Series, pd.DataFrame]:
    """
    Time a pandas rolling operation and report to PerfCollector with callsite info.
    
    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        Input series/dataframe
    win : int
        Rolling window size
    min_periods : int
        Minimum periods for rolling window
    op : str
        Operation name: "mean", "std", "min", "max", "sum", "var", "quantile", "apply"
    **kwargs
        Additional arguments passed to rolling operation (e.g., ddof for std, q for quantile)
    
    Returns
    -------
    Result of the rolling operation
    """
    # Get callsite information
    frame = inspect.currentframe().f_back
    lineno = frame.f_lineno
    filename = frame.f_code.co_filename
    
    # Extract just the filename (not full path) for cleaner output
    import os.path
    filename_short = os.path.basename(filename)
    site = f"{filename_short}:{lineno}"
    
    # Build tag: for quantile, include q value if present
    if op == "quantile" and "q" in kwargs:
        q_val = kwargs["q"]
        tag = f"rolling.pandas.quantile.q{q_val}.w{win}"
    else:
        tag = f"rolling.pandas.{op}.w{win}"
    
    # Time the operation
    t_start = time.perf_counter()
    try:
        rolling_obj = series.rolling(win, min_periods=min_periods)
        result = getattr(rolling_obj, op)(**kwargs)
    finally:
        t_end = time.perf_counter()
        dt = t_end - t_start
        
        # Report timing with callsite info stored in tag (we'll extract sites separately if needed)
        # For now, just use the tag - sites can be tracked separately if needed
        perf_add(tag, dt)
        perf_inc(tag)
    
    return result

