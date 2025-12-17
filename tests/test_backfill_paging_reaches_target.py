"""
Test robust backfill paging reaches target.

Tests that backfill_m5_candles_until_target pages backward correctly
and reaches target_bars when OANDA has sufficient historical data.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gx1.execution.oanda_backfill import backfill_m5_candles_until_target


class MockOandaClient:
    """Mock OANDA client that returns batches of candles."""
    
    def __init__(self, batches):
        """
        Initialize with list of DataFrames to return.
        
        batches: List[pd.DataFrame] - candles to return in order
        """
        self.batches = batches
        self.call_count = 0
    
    def get_candles(self, **kwargs):
        """Return next batch of candles."""
        if self.call_count >= len(self.batches):
            return pd.DataFrame()
        
        result = self.batches[self.call_count]
        self.call_count += 1
        return result


def test_backfill_paging_reaches_target():
    """Test that backfill pages backward and reaches target."""
    # Create mock batches
    # First batch: 283 candles (latest)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    batch1_times = pd.date_range(end=now, periods=283, freq="5min", tz="UTC")
    batch1 = pd.DataFrame(
        {
            "open": [2000.0] * 283,
            "high": [2001.0] * 283,
            "low": [1999.0] * 283,
            "close": [2000.5] * 283,
            "volume": [100] * 283,
        },
        index=batch1_times,
    )
    
    # Second batch: 200 older candles
    batch2_start = batch1_times.min() - pd.Timedelta(minutes=5)
    batch2_times = pd.date_range(end=batch2_start, periods=200, freq="5min", tz="UTC")
    batch2 = pd.DataFrame(
        {
            "open": [2000.0] * 200,
            "high": [2001.0] * 200,
            "low": [1999.0] * 200,
            "close": [2000.5] * 200,
            "volume": [100] * 200,
        },
        index=batch2_times,
    )
    
    # Create mock client
    mock_client = MockOandaClient([batch1, batch2])
    
    # Run backfill with target=388
    target_bars = 388
    result_df, meta = backfill_m5_candles_until_target(
        oanda_client=mock_client,
        instrument="XAU_USD",
        granularity="M5",
        target_bars=target_bars,
        price="M",
        max_batch=5000,
        max_iters=10,
        min_new_per_iter=5,
    )
    
    # Verify results
    assert len(result_df) >= target_bars, f"Expected >= {target_bars} bars, got {len(result_df)}"
    assert meta["total_bars"] >= target_bars
    assert meta["stop_reason"] == "target_reached"
    assert meta["iterations"] >= 2  # Should take at least 2 iterations
    
    # Verify no duplicate timestamps
    assert len(result_df.index) == len(result_df.index.unique()), "Duplicate timestamps found"
    
    # Verify sorted ascending
    assert result_df.index.is_monotonic_increasing, "DataFrame not sorted ascending"
    
    # Verify earliest < latest
    assert meta["earliest_time"] < meta["latest_time"]


def test_backfill_stops_when_no_more_data():
    """Test that backfill stops when no more data is available."""
    # Create single batch with 283 candles
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    batch1_times = pd.date_range(end=now, periods=283, freq="5min", tz="UTC")
    batch1 = pd.DataFrame(
        {
            "open": [2000.0] * 283,
            "high": [2001.0] * 283,
            "low": [1999.0] * 283,
            "close": [2000.5] * 283,
            "volume": [100] * 283,
        },
        index=batch1_times,
    )
    
    # Second batch: empty (no more data)
    batch2 = pd.DataFrame()
    
    mock_client = MockOandaClient([batch1, batch2])
    
    target_bars = 388
    result_df, meta = backfill_m5_candles_until_target(
        oanda_client=mock_client,
        instrument="XAU_USD",
        granularity="M5",
        target_bars=target_bars,
        price="M",
        max_batch=5000,
        max_iters=10,
        min_new_per_iter=5,
    )
    
    # Should stop with "no_more_data"
    assert meta["stop_reason"] == "no_more_data"
    assert len(result_df) == 283  # Only got first batch
    assert meta["total_bars"] == 283


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

