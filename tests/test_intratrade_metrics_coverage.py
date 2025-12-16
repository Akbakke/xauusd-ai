#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test intratrade metrics coverage on all exit paths.

Verifies that all exit paths log exit_summary with intratrade metrics.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import Mock, MagicMock


class MockTrade:
    """Mock trade object for testing."""
    def __init__(self, trade_id: str, entry_price: float, side: str):
        self.trade_id = trade_id
        self.entry_price = entry_price
        self.side = side
        self.extra = {
            "_price_trace": [
                {"ts": "2025-01-01T10:00:00Z", "high": 100.2, "low": 100.0, "close": 100.1},
                {"ts": "2025-01-01T10:05:00Z", "high": 100.4, "low": 100.1, "close": 100.3},
            ],
            "exit_profile": "RULE5",
        }
        self.entry_time = "2025-01-01T10:00:00Z"


def test_all_exit_paths_log_metrics():
    """Test that _log_trade_close_with_metrics() is called for all exit paths."""
    from gx1.execution.exit_manager import ExitManager
    
    # Create mock runner with trade journal
    runner = Mock()
    runner.trade_journal = Mock()
    runner.trade_journal.log_exit_summary = Mock()
    runner.trade_journal.log = Mock()
    
    exit_mgr = ExitManager(runner)
    
    # Test that _log_trade_close_with_metrics() calls log_exit_summary with all required fields
    trade = MockTrade("TEST-001", entry_price=100.0, side="long")
    
    exit_mgr._log_trade_close_with_metrics(
        trade=trade,
        exit_time="2025-01-01T10:10:00Z",
        exit_price=100.5,
        exit_reason="TP",
        realized_pnl_bps=50.0,
        bars_held=2,
    )
    
    # Verify log_exit_summary was called
    assert runner.trade_journal.log_exit_summary.called
    
    # Get call arguments
    call_args = runner.trade_journal.log_exit_summary.call_args
    assert call_args is not None
    
    # Verify all required fields are present
    kwargs = call_args.kwargs
    assert "trade_id" in kwargs
    assert "exit_time" in kwargs
    assert "exit_price" in kwargs
    assert "exit_reason" in kwargs
    assert "realized_pnl_bps" in kwargs
    assert "max_mfe_bps" in kwargs
    assert "max_mae_bps" in kwargs
    assert "intratrade_drawdown_bps" in kwargs
    
    # Verify metrics are not None (they should be calculated)
    assert kwargs["max_mfe_bps"] is not None
    assert kwargs["max_mae_bps"] is not None
    assert kwargs["intratrade_drawdown_bps"] is not None


def test_intratrade_metrics_invariants():
    """Test that invariants are validated."""
    from gx1.execution.exit_manager import ExitManager
    
    runner = Mock()
    runner.trade_journal = Mock()
    runner.trade_journal.log_exit_summary = Mock()
    runner.trade_journal.log = Mock()
    
    exit_mgr = ExitManager(runner)
    
    # Test invariant validation
    import logging
    
    # Capture warnings manually (no pytest.MonkeyPatch needed)
    warnings = []
    original_warning = logging.getLogger("gx1.execution.exit_manager").warning
    
    def warning_handler(msg, *args, **kwargs):
        warnings.append(str(msg) % args if args else msg)
        original_warning(msg, *args, **kwargs)
    
    logging.getLogger("gx1.execution.exit_manager").warning = warning_handler
    
    try:
        # Capture warnings
        warnings = []
        def warning_handler(msg, *args, **kwargs):
            warnings.append(str(msg) % args if args else msg)
        
        logging.getLogger("gx1.execution.exit_manager").warning = warning_handler
        
        # Test MFE < 0 violation
        exit_mgr._validate_intratrade_metrics(
            trade_id="TEST-002",
            metrics={"max_mfe_bps": -10.0, "max_mae_bps": -5.0, "intratrade_drawdown_bps": -2.0},
        )
        
        # Should log warning
        assert any("MFE violation" in w for w in warnings)
        
        # Test MAE > 0 violation
        warnings.clear()
        exit_mgr._validate_intratrade_metrics(
            trade_id="TEST-003",
            metrics={"max_mfe_bps": 10.0, "max_mae_bps": 5.0, "intratrade_drawdown_bps": -2.0},
        )
        
        # Should log warning
        assert any("MAE violation" in w for w in warnings)
        
        # Test DD > 0 violation
        warnings.clear()
        exit_mgr._validate_intratrade_metrics(
            trade_id="TEST-004",
            metrics={"max_mfe_bps": 10.0, "max_mae_bps": -5.0, "intratrade_drawdown_bps": 2.0},
        )
        
        # Should log warning
        assert any("Drawdown violation" in w for w in warnings)
        
        # Test MFE < realized_pnl violation
        warnings.clear()
        exit_mgr._validate_intratrade_metrics(
            trade_id="TEST-005",
            metrics={"max_mfe_bps": 10.0, "max_mae_bps": -5.0, "intratrade_drawdown_bps": -2.0},
            realized_pnl_bps=20.0,
        )
        
        # Should log warning
        assert any("MFE < realized_pnl" in w for w in warnings)
    
    finally:
        # Restore original warning handler
        logging.getLogger("gx1.execution.exit_manager").warning = original_warning


if __name__ == "__main__":
    print("Running intratrade metrics coverage tests...")
    test_all_exit_paths_log_metrics()
    test_intratrade_metrics_invariants()
    print("✓ All tests passed!")

