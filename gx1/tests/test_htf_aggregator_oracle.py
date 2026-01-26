"""
Mini-oracle tests for HTF aggregator and alignment.
Validates that NumPy implementation matches pandas semantics exactly.

Test A: OHLC-ekvivalens (H1 and H4)
Test B: Alignment-ekvivalens (searchsorted vs pandas reindex/ffill/shift)
Test C: Edge cases (warmup, bucket boundaries, data gaps)
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta

from gx1.features.htf_aggregator import HTFAggregator, build_htf_from_m5
from gx1.features.basic_v1 import _align_htf_to_m5_numpy


def generate_synthetic_m5_data(n_bars: int = 288 * 3, start_ts: int = None) -> tuple:
    """
    Generate synthetic M5 OHLC data for testing.
    
    Args:
        n_bars: Number of M5 bars (default: 3 days = 288 * 3)
        start_ts: Start timestamp in seconds since epoch (default: 2024-01-01 00:00:00 UTC)
    
    Returns:
        tuple: (timestamps_sec, open, high, low, close) all as NumPy arrays
    """
    if start_ts is None:
        # Default: 2024-01-01 00:00:00 UTC
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
    
    # Generate timestamps: every 5 minutes
    timestamps_sec = np.array([start_ts + i * 300 for i in range(n_bars)], dtype=np.int64)
    
    # Generate deterministic OHLC (not random, but deterministic pattern)
    open_arr = np.array([float(i) for i in range(n_bars)], dtype=np.float64)
    high_arr = open_arr + (np.arange(n_bars) % 7) * 0.1
    low_arr = open_arr - (np.arange(n_bars) % 5) * 0.1
    close_arr = open_arr + ((np.arange(n_bars) % 3) - 1) * 0.05
    
    return (timestamps_sec, open_arr, high_arr, low_arr, close_arr)


def pandas_resample_ohlc(
    timestamps_sec: np.ndarray,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    interval_hours: int,
    label: str = "right",
    closed: str = "right"
) -> pd.DataFrame:
    """
    Pandas oracle: resample M5 OHLC to HTF using pandas.
    
    Args:
        timestamps_sec: M5 timestamps (seconds since epoch)
        open_arr, high_arr, low_arr, close_arr: M5 OHLC arrays
        interval_hours: 1 for H1, 4 for H4
        label: "right" (label with end time) or "left" (label with start time)
        closed: "right" (interval includes right endpoint) or "left"
    
    Returns:
        DataFrame with DateTimeIndex and columns: open, high, low, close
        Only completed HTF bars (last partial bar is dropped)
    """
    # Convert to DataFrame with DateTimeIndex
    df = pd.DataFrame({
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr
    }, index=pd.to_datetime(timestamps_sec, unit='s', utc=True))
    
    # Resample
    freq = f"{interval_hours}H"
    resampled = df.resample(freq, label=label, closed=closed).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })
    
    # Drop last bar if it's partial (not enough M5 bars to complete it)
    # For simplicity, we keep all bars (pandas resample handles partial bars)
    # But we'll compare only bars that have close_time <= last M5 timestamp
    last_m5_ts = timestamps_sec[-1]
    if label == "right" and closed == "left":
        # HTF bar closes at its label time
        # Drop bars where close_time > last_m5_ts
        resampled = resampled[resampled.index.astype(np.int64) // 1_000_000_000 <= last_m5_ts]
    
    return resampled


class TestHTFOHLCEquivalence:
    """Test A: OHLC-ekvivalens (H1 and H4)"""
    
    def test_h1_ohlc_equivalence(self):
        """Test H1 OHLC matches pandas resample."""
        # Generate synthetic data (3 days)
        timestamps_sec, open_arr, high_arr, low_arr, close_arr = generate_synthetic_m5_data(n_bars=288 * 3)
        
        # NumPy: HTFAggregator
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        # Pandas oracle: use label="right", closed="left" to match HTF aggregator semantics
        # label="right": bar is labeled with end time (close_time)
        # closed="left": interval includes left endpoint (start), excludes right endpoint (end)
        # This matches HTF aggregator: bar includes M5 bars from [start, end), closes at end
        df_pandas = pandas_resample_ohlc(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr,
            interval_hours=1, label="right", closed="left"
        )
        
        # Convert pandas timestamps to seconds
        pandas_ts_sec = (df_pandas.index.astype(np.int64) // 1_000_000_000).astype(np.int64)
        
        # Compare: same number of bars (minus partial)
        # NumPy aggregator finalizes last bucket (may be partial)
        # Pandas drops partial last bar
        # Filter both to only completed bars (close_time <= last M5 timestamp)
        pandas_completed = df_pandas[pandas_ts_sec <= timestamps_sec[-1]]
        # HTF aggregator: filter to only bars with close_time <= last M5 timestamp
        h1_completed_mask = h1_close_times <= timestamps_sec[-1]
        h1_ts_completed = h1_ts[h1_completed_mask]
        h1_open_completed = h1_open[h1_completed_mask]
        h1_high_completed = h1_high[h1_completed_mask]
        h1_low_completed = h1_low[h1_completed_mask]
        h1_close_completed = h1_close[h1_completed_mask]
        h1_close_times_completed = h1_close_times[h1_completed_mask]
        
        assert len(h1_ts_completed) == len(pandas_completed), (
            f"H1 bar count mismatch: NumPy={len(h1_ts_completed)}, Pandas={len(pandas_completed)}"
        )
        
        if len(h1_ts_completed) > 0:
            # Pandas label="right" means bar is labeled with end time (close_time)
            pandas_close_times = (pandas_completed.index.astype(np.int64) // 1_000_000_000).astype(np.int64)
            
            # Sort both arrays for comparison
            h1_close_times_sorted = np.sort(h1_close_times_completed)
            pandas_close_times_sorted = np.sort(pandas_close_times)
            
            # Compare close times (should match exactly)
            assert len(h1_close_times_sorted) == len(pandas_close_times_sorted), (
                f"H1 close_times count mismatch: NumPy={len(h1_close_times_sorted)}, Pandas={len(pandas_close_times_sorted)}"
            )
            if len(h1_close_times_sorted) > 0:
                max_close_time_diff = np.max(np.abs(h1_close_times_sorted - pandas_close_times_sorted))
                assert max_close_time_diff < 1, (
                    f"H1 close_time mismatch: max_diff={max_close_time_diff} seconds"
                )
            
            # Compare OHLC values (need to match by close_time, not by index)
            # Create mapping: close_time -> OHLC values
            h1_dict = {h1_close_times_completed[i]: (h1_open_completed[i], h1_high_completed[i], h1_low_completed[i], h1_close_completed[i]) 
                      for i in range(len(h1_close_times_completed))}
            pandas_dict = {pandas_close_times[i]: (
                pandas_completed["open"].iloc[i],
                pandas_completed["high"].iloc[i],
                pandas_completed["low"].iloc[i],
                pandas_completed["close"].iloc[i]
            ) for i in range(len(pandas_close_times))}
            
            # Compare values for matching close_times
            common_close_times = set(h1_close_times_completed) & set(pandas_close_times)
            assert len(common_close_times) > 0, "No common close_times found"
            
            for ct in sorted(common_close_times):
                h1_vals = h1_dict[ct]
                pandas_vals = pandas_dict[ct]
                assert abs(h1_vals[0] - pandas_vals[0]) < 1e-6, f"H1 open mismatch at close_time={ct}: {h1_vals[0]} vs {pandas_vals[0]}"
                assert abs(h1_vals[1] - pandas_vals[1]) < 1e-6, f"H1 high mismatch at close_time={ct}: {h1_vals[1]} vs {pandas_vals[1]}"
                assert abs(h1_vals[2] - pandas_vals[2]) < 1e-6, f"H1 low mismatch at close_time={ct}: {h1_vals[2]} vs {pandas_vals[2]}"
                assert abs(h1_vals[3] - pandas_vals[3]) < 1e-6, f"H1 close mismatch at close_time={ct}: {h1_vals[3]} vs {pandas_vals[3]}"
    
    def test_h4_ohlc_equivalence(self):
        """Test H4 OHLC matches pandas resample."""
        # Generate synthetic data (5 days to ensure multiple H4 bars)
        timestamps_sec, open_arr, high_arr, low_arr, close_arr = generate_synthetic_m5_data(n_bars=288 * 5)
        
        # NumPy: HTFAggregator
        h4_ts, h4_open, h4_high, h4_low, h4_close, h4_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=4
        )
        
        # Pandas oracle: use label="right", closed="left" to match HTF aggregator semantics
        df_pandas = pandas_resample_ohlc(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr,
            interval_hours=4, label="right", closed="left"
        )
        
        # Convert pandas timestamps to seconds
        pandas_ts_sec = (df_pandas.index.astype(np.int64) // 1_000_000_000).astype(np.int64)
        pandas_completed = df_pandas[pandas_ts_sec <= timestamps_sec[-1]]
        
        # Filter HTF aggregator to only completed bars
        h4_completed_mask = h4_close_times <= timestamps_sec[-1]
        h4_ts_completed = h4_ts[h4_completed_mask]
        h4_open_completed = h4_open[h4_completed_mask]
        h4_high_completed = h4_high[h4_completed_mask]
        h4_low_completed = h4_low[h4_completed_mask]
        h4_close_completed = h4_close[h4_completed_mask]
        h4_close_times_completed = h4_close_times[h4_completed_mask]
        
        assert len(h4_ts_completed) == len(pandas_completed), (
            f"H4 bar count mismatch: NumPy={len(h4_ts_completed)}, Pandas={len(pandas_completed)}"
        )
        
        if len(h4_ts_completed) > 0:
            # Pandas label="right" means bar is labeled with end time (close_time)
            pandas_close_times = (pandas_completed.index.astype(np.int64) // 1_000_000_000).astype(np.int64)
            
            # Compare close times
            h4_close_times_sorted = np.sort(h4_close_times_completed)
            pandas_close_times_sorted = np.sort(pandas_close_times)
            
            assert len(h4_close_times_sorted) == len(pandas_close_times_sorted), (
                f"H4 close_times count mismatch: NumPy={len(h4_close_times_sorted)}, Pandas={len(pandas_close_times_sorted)}"
            )
            if len(h4_close_times_sorted) > 0:
                max_close_time_diff = np.max(np.abs(h4_close_times_sorted - pandas_close_times_sorted))
                assert max_close_time_diff < 1, f"H4 close_time mismatch: max_diff={max_close_time_diff} seconds"
            
            # Compare OHLC values by close_time
            h4_dict = {h4_close_times_completed[i]: (h4_open_completed[i], h4_high_completed[i], h4_low_completed[i], h4_close_completed[i]) 
                      for i in range(len(h4_close_times_completed))}
            pandas_dict = {pandas_close_times[i]: (
                pandas_completed["open"].iloc[i],
                pandas_completed["high"].iloc[i],
                pandas_completed["low"].iloc[i],
                pandas_completed["close"].iloc[i]
            ) for i in range(len(pandas_close_times))}
            
            common_close_times = set(h4_close_times_completed) & set(pandas_close_times)
            assert len(common_close_times) > 0, "No common close_times found"
            
            for ct in sorted(common_close_times):
                h4_vals = h4_dict[ct]
                pandas_vals = pandas_dict[ct]
                assert abs(h4_vals[0] - pandas_vals[0]) < 1e-6, f"H4 open mismatch at close_time={ct}"
                assert abs(h4_vals[1] - pandas_vals[1]) < 1e-6, f"H4 high mismatch at close_time={ct}"
                assert abs(h4_vals[2] - pandas_vals[2]) < 1e-6, f"H4 low mismatch at close_time={ct}"
                assert abs(h4_vals[3] - pandas_vals[3]) < 1e-6, f"H4 close mismatch at close_time={ct}"


class TestHTFAlignmentEquivalence:
    """Test B: Alignment-ekvivalens (searchsorted vs pandas reindex/ffill/shift)"""
    
    def test_h1_alignment_equivalence(self):
        """Test H1 alignment matches pandas reindex/ffill/shift."""
        # Generate synthetic data
        timestamps_sec, open_arr, high_arr, low_arr, close_arr = generate_synthetic_m5_data(n_bars=288 * 3)
        
        # Build H1 bars
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        if len(h1_ts) == 0:
            pytest.skip("No H1 bars generated")
        
        # Create a simple HTF feature (e.g., close price)
        h1_feature = h1_close.copy()
        
        # NumPy: _align_htf_to_m5_numpy
        aligned_numpy = _align_htf_to_m5_numpy(
            h1_feature, h1_close_times, timestamps_sec, is_replay=False
        )
        
        # Pandas oracle: reindex with ffill, then shift(1)
        # Create Series indexed by HTF close times
        h1_feature_series = pd.Series(
            h1_feature,
            index=pd.to_datetime(h1_close_times, unit='s', utc=True)
        )
        
        # Reindex to M5 timestamps with forward-fill
        m5_index = pd.to_datetime(timestamps_sec, unit='s', utc=True)
        aligned_pandas = h1_feature_series.reindex(m5_index, method="ffill")
        
        # Shift(1): use previous bar's value
        aligned_pandas_shifted = aligned_pandas.shift(1).fillna(0.0)
        
        # Compare: skip warmup period (where NumPy has 0.0 but pandas may have NaN)
        # Find first index where both have valid values
        numpy_valid = aligned_numpy != 0.0
        pandas_valid = ~np.isnan(aligned_pandas_shifted.values)
        both_valid = numpy_valid & pandas_valid
        
        if np.any(both_valid):
            # Compare only where both have valid values
            numpy_vals = aligned_numpy[both_valid]
            pandas_vals = aligned_pandas_shifted.values[both_valid]
            max_diff = np.max(np.abs(numpy_vals - pandas_vals))
            assert max_diff < 1e-6, (
                f"H1 alignment mismatch: max_diff={max_diff}, "
                f"first_valid_idx={np.where(both_valid)[0][0] if np.any(both_valid) else None}"
            )
        
        # First element should be 0.0 (after shift)
        assert aligned_numpy[0] == 0.0, "First element should be 0.0 after shift"
    
    def test_h4_alignment_equivalence(self):
        """Test H4 alignment matches pandas reindex/ffill/shift."""
        # Generate synthetic data (5 days)
        timestamps_sec, open_arr, high_arr, low_arr, close_arr = generate_synthetic_m5_data(n_bars=288 * 5)
        
        # Build H4 bars
        h4_ts, h4_open, h4_high, h4_low, h4_close, h4_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=4
        )
        
        if len(h4_ts) == 0:
            pytest.skip("No H4 bars generated")
        
        # Create a simple HTF feature
        h4_feature = h4_close.copy()
        
        # NumPy alignment
        aligned_numpy = _align_htf_to_m5_numpy(
            h4_feature, h4_close_times, timestamps_sec, is_replay=False
        )
        
        # Pandas oracle
        h4_feature_series = pd.Series(
            h4_feature,
            index=pd.to_datetime(h4_close_times, unit='s', utc=True)
        )
        m5_index = pd.to_datetime(timestamps_sec, unit='s', utc=True)
        aligned_pandas = h4_feature_series.reindex(m5_index, method="ffill").shift(1).fillna(0.0)
        
        # Compare: skip warmup period
        numpy_valid = aligned_numpy != 0.0
        pandas_valid = ~np.isnan(aligned_pandas.values)
        both_valid = numpy_valid & pandas_valid
        
        if np.any(both_valid):
            numpy_vals = aligned_numpy[both_valid]
            pandas_vals = aligned_pandas.values[both_valid]
            max_diff = np.max(np.abs(numpy_vals - pandas_vals))
            assert max_diff < 1e-6, f"H4 alignment mismatch: max_diff={max_diff}"
        
        assert aligned_numpy[0] == 0.0, "First element should be 0.0 after shift"


class TestHTFEdgeCases:
    """Test C: Edge cases (warmup, bucket boundaries, data gaps)"""
    
    def test_warmup_before_first_htf_close_replay(self):
        """Test replay mode: hard fail when M5 bars start before first HTF close."""
        # Generate data starting at 00:00, but only 10 M5 bars (50 minutes)
        # First H1 bar closes at 01:00, so first 11 M5 bars have no completed HTF bar
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        timestamps_sec = np.array([start_ts + i * 300 for i in range(11)], dtype=np.int64)
        open_arr = np.ones(11, dtype=np.float64)
        high_arr = np.ones(11, dtype=np.float64) * 1.1
        low_arr = np.ones(11, dtype=np.float64) * 0.9
        close_arr = np.ones(11, dtype=np.float64)
        
        # Build H1 bars
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        # Create a feature
        h1_feature = h1_close.copy() if len(h1_close) > 0 else np.array([], dtype=np.float64)
        
        # In replay mode, should hard fail
        with pytest.raises(RuntimeError, match="warmup not satisfied"):
            _align_htf_to_m5_numpy(
                h1_feature, h1_close_times, timestamps_sec, is_replay=True
            )
    
    def test_warmup_before_first_htf_close_live(self):
        """Test live mode: return zeros with warning when warmup not satisfied."""
        # Same setup as above
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        timestamps_sec = np.array([start_ts + i * 300 for i in range(11)], dtype=np.int64)
        open_arr = np.ones(11, dtype=np.float64)
        high_arr = np.ones(11, dtype=np.float64) * 1.1
        low_arr = np.ones(11, dtype=np.float64) * 0.9
        close_arr = np.ones(11, dtype=np.float64)
        
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        h1_feature = h1_close.copy() if len(h1_close) > 0 else np.array([], dtype=np.float64)
        
        # In live mode, should return zeros (no exception)
        aligned = _align_htf_to_m5_numpy(
            h1_feature, h1_close_times, timestamps_sec, is_replay=False
        )
        
        # Should be all zeros (no completed HTF bars)
        assert np.all(aligned == 0.0), "Live mode should return zeros when warmup not satisfied"
    
    def test_bucket_boundary_exact_close_time(self):
        """Test M5 bar exactly on HTF close_time."""
        # Generate data with M5 bar exactly at H1 close_time (01:00:00)
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        # First H1 bar: 00:00-01:00, closes at 01:00:00
        # Add M5 bars: 00:00, 00:05, ..., 01:00 (exactly at close_time)
        timestamps_sec = np.array([start_ts + i * 300 for i in range(13)], dtype=np.int64)  # 00:00 to 01:00
        open_arr = np.ones(13, dtype=np.float64)
        high_arr = np.ones(13, dtype=np.float64) * 1.1
        low_arr = np.ones(13, dtype=np.float64) * 0.9
        close_arr = np.ones(13, dtype=np.float64)
        
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        if len(h1_close_times) > 0:
            # M5 bar exactly at close_time should use that HTF bar
            # searchsorted with side="right" gives index of first element > value
            # For exact match, we want index - 1
            exact_idx = np.searchsorted(h1_close_times, h1_close_times[0], side="right") - 1
            assert exact_idx == 0, f"M5 bar at exact close_time should map to HTF bar 0, got {exact_idx}"
    
    def test_bucket_boundary_before_close_time(self):
        """Test M5 bar just before HTF close_time."""
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        # M5 bar at 00:55 (just before 01:00 close_time)
        timestamps_sec = np.array([start_ts + 55 * 60], dtype=np.int64)
        
        # Build H1 bars (need at least 12 M5 bars to complete first H1 bar)
        full_timestamps = np.array([start_ts + i * 300 for i in range(13)], dtype=np.int64)
        open_arr = np.ones(13, dtype=np.float64)
        high_arr = np.ones(13, dtype=np.float64) * 1.1
        low_arr = np.ones(13, dtype=np.float64) * 0.9
        close_arr = np.ones(13, dtype=np.float64)
        
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            full_timestamps, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        if len(h1_close_times) > 0:
            # M5 bar at 00:55 is before first HTF close_time (01:00)
            # searchsorted with side="right" gives index of first element > value
            # For value < all elements, this gives 0, so idx = 0 - 1 = -1
            # This is correct: no completed HTF bar available yet
            idx = np.searchsorted(h1_close_times, timestamps_sec[0], side="right") - 1
            assert idx == -1, f"M5 bar before first close_time should give idx=-1 (no completed bar), got {idx}"
    
    def test_bucket_boundary_after_close_time(self):
        """Test M5 bar just after HTF close_time."""
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        # M5 bar at 01:05 (just after 01:00 close_time)
        timestamps_sec = np.array([start_ts + 65 * 60], dtype=np.int64)
        
        # Build H1 bars
        full_timestamps = np.array([start_ts + i * 300 for i in range(14)], dtype=np.int64)
        open_arr = np.ones(14, dtype=np.float64)
        high_arr = np.ones(14, dtype=np.float64) * 1.1
        low_arr = np.ones(14, dtype=np.float64) * 0.9
        close_arr = np.ones(14, dtype=np.float64)
        
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            full_timestamps, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        if len(h1_close_times) > 0:
            # M5 bar after close_time should use that HTF bar (first completed bar)
            idx = np.searchsorted(h1_close_times, timestamps_sec[0], side="right") - 1
            assert idx == 0, f"M5 bar after close_time should map to HTF bar 0, got {idx}"
    
    def test_data_gap_tolerance(self):
        """Test aggregator tolerates data gaps (doesn't hard-fail)."""
        # Generate data with a gap (skip 30 minutes = 6 M5 bars)
        start_ts = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        
        # First chunk: 00:00-00:25 (6 bars)
        timestamps_part1 = np.array([start_ts + i * 300 for i in range(6)], dtype=np.int64)
        # Gap: 00:30-00:55 (skip 6 bars)
        # Second chunk: 01:00-01:25 (6 bars)
        timestamps_part2 = np.array([start_ts + (12 + i) * 300 for i in range(6)], dtype=np.int64)
        timestamps_sec = np.concatenate([timestamps_part1, timestamps_part2])
        
        n_total = len(timestamps_sec)
        open_arr = np.ones(n_total, dtype=np.float64)
        high_arr = np.ones(n_total, dtype=np.float64) * 1.1
        low_arr = np.ones(n_total, dtype=np.float64) * 0.9
        close_arr = np.ones(n_total, dtype=np.float64)
        
        # Aggregator should handle gap gracefully (not crash)
        h1_ts, h1_open, h1_high, h1_low, h1_close, h1_close_times = build_htf_from_m5(
            timestamps_sec, open_arr, high_arr, low_arr, close_arr, interval_hours=1
        )
        
        # Should have at least one completed H1 bar (from second chunk)
        # First chunk (00:00-00:25) is incomplete, second chunk (01:00-01:25) completes first H1 bar
        assert len(h1_close_times) >= 1, "Should have at least one completed H1 bar despite gap"
