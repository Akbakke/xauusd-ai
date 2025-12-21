"""
Regression test for Q4 A_TREND disable policy.

Ensures that Q4 × A_TREND trades are blocked (NO-TRADE) by default.
"""

import pytest
from gx1.sniper.policy.sniper_q4_atrend_size_overlay import apply_q4_atrend_overlay


def test_q4_atrend_disable_default():
    """Test that Q4 × A_TREND defaults to disable (NO-TRADE)."""
    # Q4 timestamp
    entry_time = "2025-10-15T12:00:00+00:00"
    
    # A_TREND regime inputs
    trend_regime = "TREND_UP"
    vol_regime = "LOW"
    atr_bps = 10.0
    spread_bps = 50.0
    
    # Default config (should use disable)
    cfg = {"enabled": True}
    
    units_out, meta = apply_q4_atrend_overlay(
        base_units=1,
        entry_time=entry_time,
        trend_regime=trend_regime,
        vol_regime=vol_regime,
        atr_bps=atr_bps,
        spread_bps=spread_bps,
        session="EU",
        cfg=cfg,
    )
    
    # Assert: units_out must be 0 (NO-TRADE)
    assert units_out == 0, f"Expected units_out=0 (NO-TRADE), got {units_out}"
    
    # Assert: overlay_applied must be True
    assert meta["overlay_applied"] is True, "Overlay should be applied"
    
    # Assert: action must be "disable"
    assert meta["action"] == "disable", f"Expected action='disable', got {meta['action']}"
    
    # Assert: reason must indicate high tail risk
    assert "Q4_A_TREND_high_tail_risk" in meta["reason"], f"Expected reason to contain 'Q4_A_TREND_high_tail_risk', got {meta['reason']}"
    
    # Assert: quarter must be Q4
    assert meta["quarter"] == "Q4", f"Expected quarter='Q4', got {meta['quarter']}"
    
    # Assert: regime_class must be A_TREND
    assert meta["regime_class"] == "A_TREND", f"Expected regime_class='A_TREND', got {meta['regime_class']}"


def test_q4_atrend_disable_explicit():
    """Test that explicit disable config works."""
    entry_time = "2025-10-15T12:00:00+00:00"
    
    cfg = {"enabled": True, "action": "disable"}
    
    units_out, meta = apply_q4_atrend_overlay(
        base_units=1,
        entry_time=entry_time,
        trend_regime="TREND_UP",
        vol_regime="LOW",
        atr_bps=10.0,
        spread_bps=50.0,
        session="EU",
        cfg=cfg,
    )
    
    assert units_out == 0, f"Expected units_out=0 (NO-TRADE), got {units_out}"
    assert meta["action"] == "disable"
    assert "Q4_A_TREND_high_tail_risk" in meta["reason"]


def test_q4_atrend_non_q4_not_blocked():
    """Test that non-Q4 A_TREND trades are not blocked."""
    # Q1 timestamp
    entry_time = "2025-01-15T12:00:00+00:00"
    
    cfg = {"enabled": True}
    
    units_out, meta = apply_q4_atrend_overlay(
        base_units=1,
        entry_time=entry_time,
        trend_regime="TREND_UP",
        vol_regime="LOW",
        atr_bps=10.0,
        spread_bps=50.0,
        session="EU",
        cfg=cfg,
    )
    
    # Non-Q4 should not trigger overlay
    assert meta["overlay_applied"] is False
    assert meta["reason"] == "not_q4"
    assert units_out == 1  # Unchanged


def test_q4_atrend_non_atrend_not_blocked():
    """Test that Q4 non-A_TREND trades are not blocked."""
    entry_time = "2025-10-15T12:00:00+00:00"
    
    cfg = {"enabled": True}
    
    units_out, meta = apply_q4_atrend_overlay(
        base_units=1,
        entry_time=entry_time,
        trend_regime="TREND_NEUTRAL",
        vol_regime="HIGH",
        atr_bps=20.0,
        spread_bps=100.0,
        session="EU",
        cfg=cfg,
    )
    
    # Non-A_TREND should not trigger overlay
    assert meta["overlay_applied"] is False
    assert "not_a_trend" in meta["reason"]
    assert units_out == 1  # Unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

