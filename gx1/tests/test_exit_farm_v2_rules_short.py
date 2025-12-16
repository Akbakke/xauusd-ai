#!/usr/bin/env python3
"""
Tests for EXIT_FARM_V2_RULES short support.

Verifies:
- Short entry/exit price handling (bid/ask inversion)
- PnL calculation correctness for shorts
- Trailing stop logic for shorts
- MAE/MFE tracking for shorts
"""

import pytest
from gx1.policy.exit_farm_v2_rules import ExitFarmV2Rules
from gx1.utils.pnl import compute_pnl_bps


def test_short_entry_exit_prices():
    """Test that short trades use correct entry/exit prices."""
    exit_policy = ExitFarmV2Rules(enable_rule_a=True, rule_a_profit_min_bps=5.0, rule_a_profit_max_bps=10.0)
    
    # Short entry: sell at bid, buy back at ask
    entry_bid = 2000.0
    entry_ask = 2000.5
    exit_policy.reset_on_entry(entry_bid, entry_ask, entry_ts="2025-01-01T00:00:00Z", side="short", trade_id="TEST-001")
    
    # Entry price should be bid (we sell at bid for short)
    assert exit_policy.entry_price == entry_bid
    assert exit_policy.side == "short"


def test_short_pnl_calculation():
    """Test PnL calculation for short trades."""
    exit_policy = ExitFarmV2Rules(enable_rule_a=True, rule_a_profit_min_bps=5.0, rule_a_profit_max_bps=10.0)
    
    # Short entry: sell at bid 2000.0, buy back at ask 1995.0 (price went down = profit)
    entry_bid = 2000.0
    entry_ask = 2000.5
    exit_policy.reset_on_entry(entry_bid, entry_ask, entry_ts="2025-01-01T00:00:00Z", side="short", trade_id="TEST-001")
    
    # Price drops: bid=1995.0, ask=1995.5 (we profit on short)
    exit_bid = 1995.0
    exit_ask = 1995.5
    
    # Calculate PnL manually: (entry_bid - exit_ask) / entry_bid * 10000
    # = (2000.0 - 1995.5) / 2000.0 * 10000 = 22.5 bps
    expected_pnl = compute_pnl_bps(entry_bid, entry_ask, exit_bid, exit_ask, "short")
    
    # Simulate bar update
    decision = exit_policy.on_bar(exit_bid, exit_ask, "2025-01-01T00:05:00Z")
    
    # PnL should be positive (price went down, we profit on short)
    assert decision is not None
    assert decision.pnl_bps > 0
    assert abs(decision.pnl_bps - expected_pnl) < 0.01


def test_short_trailing_stop():
    """Test trailing stop logic for short trades."""
    exit_policy = ExitFarmV2Rules(
        enable_rule_a=True,
        rule_a_profit_min_bps=5.0,
        rule_a_profit_max_bps=10.0,
        rule_a_adaptive_threshold_bps=3.0,
        rule_a_trailing_stop_bps=2.0,
        rule_a_adaptive_bars=3,
    )
    
    entry_bid = 2000.0
    entry_ask = 2000.5
    exit_policy.reset_on_entry(entry_bid, entry_ask, entry_ts="2025-01-01T00:00:00Z", side="short", trade_id="TEST-001")
    
    # Bar 1: Price drops to 1997.0/1997.5 (profit ~12.5 bps) - should activate trailing
    decision = exit_policy.on_bar(1997.0, 1997.5, "2025-01-01T00:05:00Z")
    assert decision is None  # Should not exit yet (within profit range)
    assert exit_policy.rule_a_trailing_active  # Trailing should be active
    
    # Bar 2: Price recovers to 1998.0/1998.5 (profit ~7.5 bps) - trailing high should update
    decision = exit_policy.on_bar(1998.0, 1998.5, "2025-01-01T00:10:00Z")
    assert decision is None  # Still above trailing threshold
    
    # Bar 3: Price recovers more to 1999.5/2000.0 (profit ~0 bps) - should hit trailing stop
    decision = exit_policy.on_bar(1999.5, 2000.0, "2025-01-01T00:15:00Z")
    assert decision is not None
    assert decision.reason == "RULE_A_TRAILING"
    assert decision.exit_price == 2000.0  # Should use ask for short exit


def test_short_mae_mfe_tracking():
    """Test MAE/MFE tracking for short trades."""
    exit_policy = ExitFarmV2Rules(enable_rule_a=True)
    
    entry_bid = 2000.0
    entry_ask = 2000.5
    exit_policy.reset_on_entry(entry_bid, entry_ask, entry_ts="2025-01-01T00:00:00Z", side="short", trade_id="TEST-001")
    
    # Bar 1: Price drops (profit)
    exit_policy.on_bar(1995.0, 1995.5, "2025-01-01T00:05:00Z")
    assert exit_policy.mfe_bps > 0  # Best profit so far
    assert exit_policy.mae_bps == 0.0  # No adverse movement yet
    
    # Bar 2: Price recovers (adverse movement)
    exit_policy.on_bar(1998.0, 1998.5, "2025-01-01T00:10:00Z")
    assert exit_policy.mae_bps < exit_policy.mfe_bps  # MAE should be worse than MFE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

