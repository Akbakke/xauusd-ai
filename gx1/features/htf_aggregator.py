"""
Incremental HTF (Higher Timeframe) aggregator for H1/H4 bars from M5 input.
DEL 2: NumPy-only, no pandas. Deterministic, O(N) per call.

This replaces pandas resample() with an incremental aggregator that builds
HTF bars directly from M5 bars, only using completed HTF bars for features.
"""
import numpy as np
from typing import Dict, Optional, Tuple


class HTFAggregator:
    """
    Incremental aggregator for H1/H4 bars from M5 input.
    
    Semantics:
    - Only completed HTF bars are used (no partial bars leak into features)
    - H1: bucket = ts // 3600
    - H4: bucket = ts // (4 * 3600)
    - Aggregation: open=first, high=max, low=min, close=last in bucket
    """
    
    def __init__(self, interval_hours: int):
        """
        Initialize HTF aggregator.
        
        Args:
            interval_hours: 1 for H1, 4 for H4
        """
        self.interval_hours = interval_hours
        self.interval_seconds = interval_hours * 3600
        
        # State: current bucket being built
        self.current_bucket: Optional[int] = None
        self.current_open: Optional[float] = None
        self.current_high: Optional[float] = None
        self.current_low: Optional[float] = None
        self.current_close: Optional[float] = None
        
        # Completed HTF bars
        self.completed_bars: Dict[int, Tuple[float, float, float, float]] = {}
        # completed_bars[bucket] = (open, high, low, close)
        
        # HTF close times (for alignment to M5)
        self.htf_close_times: np.ndarray = np.array([], dtype=np.int64)
        # htf_close_times[i] = timestamp when HTF bar at index i closed
    
    def reset(self):
        """Reset aggregator state (for new dataset)."""
        self.current_bucket = None
        self.current_open = None
        self.current_high = None
        self.current_low = None
        self.current_close = None
        self.completed_bars = {}
        self.htf_close_times = np.array([], dtype=np.int64)
    
    def add_m5_bar(self, ts: int, open_val: float, high_val: float, low_val: float, close_val: float):
        """
        Add a single M5 bar to the aggregator.
        
        Args:
            ts: Timestamp (seconds since epoch, int64)
            open_val: Open price
            high_val: High price
            low_val: Low price
            close_val: Close price
        """
        # Calculate bucket for this M5 bar
        bucket = ts // self.interval_seconds
        
        if self.current_bucket is None:
            # First bar: start new bucket
            self.current_bucket = bucket
            self.current_open = open_val
            self.current_high = high_val
            self.current_low = low_val
            self.current_close = close_val
        elif bucket == self.current_bucket:
            # Same bucket: update aggregation
            # high = max(high, h)
            self.current_high = max(self.current_high, high_val)
            # low = min(low, l)
            self.current_low = min(self.current_low, low_val)
            # close = c (last close in bucket)
            self.current_close = close_val
        else:
            # Bucket changed: complete previous bucket and start new one
            # Save completed bar
            self.completed_bars[self.current_bucket] = (
                self.current_open,
                self.current_high,
                self.current_low,
                self.current_close
            )
            # HTF bar closes at start of next bucket
            htf_close_time = (self.current_bucket + 1) * self.interval_seconds
            self.htf_close_times = np.append(self.htf_close_times, htf_close_time)
            
            # Start new bucket
            self.current_bucket = bucket
            self.current_open = open_val
            self.current_high = high_val
            self.current_low = low_val
            self.current_close = close_val
    
    def finalize(self):
        """
        Finalize current bucket (if any) as completed bar.
        Call this after processing all M5 bars.
        """
        if self.current_bucket is not None:
            # Save current bucket as completed
            self.completed_bars[self.current_bucket] = (
                self.current_open,
                self.current_high,
                self.current_low,
                self.current_close
            )
            # HTF bar closes at start of next bucket
            htf_close_time = (self.current_bucket + 1) * self.interval_seconds
            self.htf_close_times = np.append(self.htf_close_times, htf_close_time)
    
    def get_htf_dataframe(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get completed HTF bars as NumPy arrays.
        
        Returns:
            tuple: (timestamps, open, high, low, close)
            All as NumPy arrays, sorted by timestamp
        """
        if len(self.completed_bars) == 0:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64)
            )
        
        # Sort buckets
        sorted_buckets = sorted(self.completed_bars.keys())
        n = len(sorted_buckets)
        
        timestamps = np.zeros(n, dtype=np.int64)
        open_arr = np.zeros(n, dtype=np.float64)
        high_arr = np.zeros(n, dtype=np.float64)
        low_arr = np.zeros(n, dtype=np.float64)
        close_arr = np.zeros(n, dtype=np.float64)
        
        for i, bucket in enumerate(sorted_buckets):
            timestamps[i] = bucket * self.interval_seconds
            open_arr[i], high_arr[i], low_arr[i], close_arr[i] = self.completed_bars[bucket]
        
        return (timestamps, open_arr, high_arr, low_arr, close_arr)
    
    def align_to_m5(self, m5_timestamps: np.ndarray) -> np.ndarray:
        """
        Align HTF values to M5 timestamps using searchsorted.
        
        For each M5 timestamp t:
        - Find last completed HTF bar where htf_close_time <= t
        - Use that HTF bar's value
        
        Args:
            m5_timestamps: M5 timestamps (seconds since epoch, int64 array)
        
        Returns:
            Indices into HTF bars array for each M5 timestamp
            -1 if no completed HTF bar available (warmup not satisfied)
        """
        if len(self.htf_close_times) == 0:
            # No completed HTF bars: return all -1
            return np.full(len(m5_timestamps), -1, dtype=np.int32)
        
        # Use searchsorted to find last completed HTF bar for each M5 timestamp
        # searchsorted(htf_close_times, t, side="right") - 1 gives last index where htf_close_times <= t
        indices = np.searchsorted(self.htf_close_times, m5_timestamps, side="right") - 1
        
        # Clamp to valid range (indices < 0 mean no completed HTF bar available)
        return indices


def build_htf_from_m5(
    m5_timestamps: np.ndarray,
    m5_open: np.ndarray,
    m5_high: np.ndarray,
    m5_low: np.ndarray,
    m5_close: np.ndarray,
    interval_hours: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build HTF bars from M5 data using incremental aggregator.
    
    Args:
        m5_timestamps: M5 timestamps (seconds since epoch, int64)
        m5_open: M5 open prices (float64)
        m5_high: M5 high prices (float64)
        m5_low: M5 low prices (float64)
        m5_close: M5 close prices (float64)
        interval_hours: 1 for H1, 4 for H4
    
    Returns:
        tuple: (htf_timestamps, htf_open, htf_high, htf_low, htf_close, htf_close_times)
        All as NumPy arrays
    """
    aggregator = HTFAggregator(interval_hours)
    
    # Process all M5 bars
    for i in range(len(m5_timestamps)):
        aggregator.add_m5_bar(
            int(m5_timestamps[i]),
            float(m5_open[i]),
            float(m5_high[i]),
            float(m5_low[i]),
            float(m5_close[i])
        )
    
    # Finalize last bucket
    aggregator.finalize()
    
    # Get HTF data
    htf_ts, htf_open, htf_high, htf_low, htf_close = aggregator.get_htf_dataframe()
    htf_close_times = aggregator.htf_close_times
    
    return (htf_ts, htf_open, htf_high, htf_low, htf_close, htf_close_times)
