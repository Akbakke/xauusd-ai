#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test intratrade metrics calculation (MFE, MAE, intratrade drawdown).

Tests both long and short positions with synthetic price traces.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import Mock
import pandas as pd


class MockTrade:
    """Mock trade object for testing."""
    def __init__(self, trade_id: str, entry_price: float, side: str):
        self.trade_id = trade_id
        self.entry_price = entry_price
        self.side = side
        self.extra = {"_price_trace": []}


def test_intratrade_metrics_long_simple():
    """Test MFE/MAE/DD calculation for long position (simple case)."""
    from gx1.execution.exit_manager import ExitManager
    
    # Create mock runner
    runner = Mock()
    runner.trade_journal = None
    exit_mgr = ExitManager(runner)
    
    # Create trade: long at 100.0
    trade = MockTrade("TEST-001", entry_price=100.0, side="long")
    
    # Price trace: goes up to 101.0, down to 99.0, closes at 100.5
    trade.extra["_price_trace"] = [
        {"ts": "2025-01-01T10:00:00Z", "high": 100.5, "low": 100.0, "close": 100.2},
        {"ts": "2025-01-01T10:05:00Z", "high": 101.0, "low": 100.2, "close": 100.8},  # Peak
        {"ts": "2025-01-01T10:10:00Z", "high": 100.5, "low": 99.0, "close": 99.5},   # Trough
        {"ts": "2025-01-01T10:15:00Z", "high": 100.6, "low": 99.5, "close": 100.5},
    ]
    
    exit_price = 100.5
    
    metrics = exit_mgr._compute_intratrade_metrics(trade, exit_price)
    
    # Expected:
    # MFE: max favorable = (101.0 - 100.0) / 100.0 * 10000 = 100.0 bps
    # MAE: max adverse = (99.0 - 100.0) / 100.0 * 10000 = -100.0 bps
    # DD: peak at bar 2 (100.8 bps), trough at bar 3 (99.5 -> -50 bps), DD = 100.8 - (-50) = 150.8 bps
    # But wait, let's recalculate: unrealized curve = [20, 80, -50, 50]
    # Peak = 80, trough after peak = -50, DD = 80 - (-50) = 130 bps
    
    assert abs(metrics["max_mfe_bps"] - 100.0) < 0.1
    assert abs(metrics["max_mae_bps"] - (-100.0)) < 0.1
    assert abs(metrics["intratrade_drawdown_bps"] - 130.0) < 1.0


def test_intratrade_metrics_short_simple():
    """Test MFE/MAE/DD calculation for short position (simple case)."""
    from gx1.execution.exit_manager import ExitManager
    
    # Create mock runner
    runner = Mock()
    runner.trade_journal = None
    exit_mgr = ExitManager(runner)
    
    # Create trade: short at 100.0
    trade = MockTrade("TEST-002", entry_price=100.0, side="short")
    
    # Price trace: goes down to 99.0 (favorable), up to 101.0 (adverse), closes at 99.5
    trade.extra["_price_trace"] = [
        {"ts": "2025-01-01T10:00:00Z", "high": 100.5, "low": 99.5, "close": 100.0},
        {"ts": "2025-01-01T10:05:00Z", "high": 100.2, "low": 99.0, "close": 99.5},   # Best (lowest)
        {"ts": "2025-01-01T10:10:00Z", "high": 101.0, "low": 99.5, "close": 100.5},   # Worst (highest)
        {"ts": "2025-01-01T10:15:00Z", "high": 100.0, "low": 99.2, "close": 99.5},
    ]
    
    exit_price = 99.5
    
    metrics = exit_mgr._compute_intratrade_metrics(trade, exit_price)
    
    # Expected:
    # MFE: max favorable = (100.0 - 99.0) / 100.0 * 10000 = 100.0 bps
    # MAE: max adverse = (100.0 - 101.0) / 100.0 * 10000 = -100.0 bps
    # DD: unrealized curve = [0, 50, -50, 50]
    # Peak = 50, trough after peak = -50, DD = 50 - (-50) = 100 bps
    
    assert abs(metrics["max_mfe_bps"] - 100.0) < 0.1
    assert abs(metrics["max_mae_bps"] - (-100.0)) < 0.1
    assert abs(metrics["intratrade_drawdown_bps"] - 100.0) < 1.0


def test_intratrade_metrics_no_trace():
    """Test that missing price trace returns None metrics."""
    from gx1.execution.exit_manager import ExitManager
    
    runner = Mock()
    runner.trade_journal = None
    exit_mgr = ExitManager(runner)
    
    trade = MockTrade("TEST-003", entry_price=100.0, side="long")
    trade.extra["_price_trace"] = []
    
    metrics = exit_mgr._compute_intratrade_metrics(trade, exit_price=100.5)
    
    assert metrics["max_mfe_bps"] is None
    assert metrics["max_mae_bps"] is None
    assert metrics["intratrade_drawdown_bps"] is None


def test_intratrade_metrics_long_winning_trade():
    """Test metrics for a winning long trade (always positive)."""
    from gx1.execution.exit_manager import ExitManager
    
    runner = Mock()
    runner.trade_journal = None
    exit_mgr = ExitManager(runner)
    
    trade = MockTrade("TEST-004", entry_price=100.0, side="long")
    
    # Price trace: steady climb, no drawdown
    trade.extra["_price_trace"] = [
        {"ts": "2025-01-01T10:00:00Z", "high": 100.2, "low": 100.0, "close": 100.1},
        {"ts": "2025-01-01T10:05:00Z", "high": 100.4, "low": 100.1, "close": 100.3},
        {"ts": "2025-01-01T10:10:00Z", "high": 100.6, "low": 100.3, "close": 100.5},
    ]
    
    exit_price = 100.5
    
    metrics = exit_mgr._compute_intratrade_metrics(trade, exit_price)
    
    # MFE: max high = 100.6 -> (100.6 - 100.0) / 100.0 * 10000 = 60 bps
    # MAE: min low = 100.0 -> (100.0 - 100.0) / 100.0 * 10000 = 0 bps
    # DD: unrealized curve = [10, 30, 50], no drawdown -> 0 bps
    
    assert abs(metrics["max_mfe_bps"] - 60.0) < 0.1
    assert abs(metrics["max_mae_bps"] - 0.0) < 0.1
    assert abs(metrics["intratrade_drawdown_bps"] - 0.0) < 0.1


if __name__ == "__main__":
    print("Running intratrade metrics tests...")
    test_intratrade_metrics_long_simple()
    test_intratrade_metrics_short_simple()
    test_intratrade_metrics_no_trace()
    test_intratrade_metrics_long_winning_trade()
    print("✓ All tests passed!")

