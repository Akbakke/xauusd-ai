#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNIPER Risk Guard V1

Purpose: Reduce tail risk (Q2 p90 loss / avg loss) with minimal EV loss by:
- Blocking entries during extreme spread/vol spikes
- Enforcing cooldown after entry
- Optional session-specific clamps (US / HIGH vol)

This guard is SNIPER-only and does not affect FARM or other policies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger(__name__)


# Reason codes
REASON_SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
REASON_ATR_TOO_HIGH = "ATR_TOO_HIGH"
REASON_VOL_EXTREME = "VOL_EXTREME"
REASON_SESSION_UNKNOWN = "SESSION_UNKNOWN"
REASON_COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
REASON_US_SPREAD_TOO_HIGH = "US_SPREAD_TOO_HIGH"
REASON_OVERLAP_SPREAD_TOO_HIGH = "OVERLAP_SPREAD_TOO_HIGH"
REASON_US_MIN_PROB_CLAMP = "US_MIN_PROB_CLAMP"
REASON_OVERLAP_MIN_PROB_CLAMP = "OVERLAP_MIN_PROB_CLAMP"


class SniperRiskGuardV1:
    """
    SNIPER-specific risk guard to reduce tail risk.
    
    Does not modify signals; only returns block/allow decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize guard with configuration.
        
        Args:
            config: Guard configuration dict (from SNIPER_RISK_GUARD_V1.yaml)
        """
        self.cfg = config.get("sniper_risk_guard_v1", {})
        self.enabled = self.cfg.get("enabled", False)
        
        # Global thresholds
        self.block_spread_bps = self.cfg.get("block_if_spread_bps_gte", 3500)
        self.block_atr_bps = self.cfg.get("block_if_atr_bps_gte", 25.0)
        self.block_vol_regimes = self.cfg.get("block_if_vol_regime_in", ["EXTREME"])
        
        # Fallbacks
        self.allow_if_missing_spread = self.cfg.get("allow_if_missing_spread", True)
        self.allow_if_missing_atr = self.cfg.get("allow_if_missing_atr", True)
        self.allow_if_missing_session = self.cfg.get("allow_if_missing_session", False)
        
        # Session configs
        self.us_cfg = self.cfg.get("us_session", {})
        self.overlap_cfg = self.cfg.get("overlap_session", {})
        
        # Cooldown tracking (per-run state)
        self._last_entry_bar_index: Optional[int] = None
    
    def should_block(
        self,
        entry_snapshot: Dict[str, Any],
        feature_context: Dict[str, Any],
        policy_state: Dict[str, Any],
        current_bar_index: int,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Determine if entry should be blocked.
        
        Args:
            entry_snapshot: Entry snapshot dict (from entry_manager)
            feature_context: Feature context dict (spread, ATR, etc.)
            policy_state: Policy state dict (session, vol_regime, etc.)
            current_bar_index: Current bar index (for cooldown)
        
        Returns:
            (should_block, reason_code, details_dict)
            - should_block: True if entry should be blocked
            - reason_code: Reason code string (or None if allowed)
            - details_dict: Additional details for logging
        """
        if not self.enabled:
            return (False, None, {})
        
        details = {}
        
        # Check cooldown
        cooldown_bars = self.cfg.get("cooldown_bars_after_entry", 1)
        if not self.cooldown_allows(current_bar_index, cooldown_bars):
            details["cooldown_bars"] = cooldown_bars
            details["last_entry_bar"] = self._last_entry_bar_index
            details["current_bar"] = current_bar_index
            return (True, REASON_COOLDOWN_ACTIVE, details)
        
        # Get session
        session = policy_state.get("session") or entry_snapshot.get("session")
        if not session or session == "UNKNOWN":
            if not self.allow_if_missing_session:
                details["session"] = session
                return (True, REASON_SESSION_UNKNOWN, details)
            # If allow_if_missing_session=True, continue with session=None
        
        # Get spread_bps
        spread_bps = feature_context.get("spread_bps") or entry_snapshot.get("spread_bps")
        if spread_bps is None:
            if not self.allow_if_missing_spread:
                details["spread_bps"] = None
                return (True, REASON_SPREAD_TOO_HIGH, details)
            # If allow_if_missing_spread=True, continue with spread_bps=None
        
        # Get atr_bps
        atr_bps = feature_context.get("atr_bps") or entry_snapshot.get("atr_bps")
        if atr_bps is None:
            if not self.allow_if_missing_atr:
                details["atr_bps"] = None
                return (True, REASON_ATR_TOO_HIGH, details)
            # If allow_if_missing_atr=True, continue with atr_bps=None
        
        # Get vol_regime
        vol_regime = policy_state.get("vol_regime") or entry_snapshot.get("vol_regime")
        
        # Check global spread threshold
        if spread_bps is not None and spread_bps >= self.block_spread_bps:
            details["spread_bps"] = spread_bps
            details["threshold"] = self.block_spread_bps
            return (True, REASON_SPREAD_TOO_HIGH, details)
        
        # Check global ATR threshold
        if atr_bps is not None and atr_bps >= self.block_atr_bps:
            details["atr_bps"] = atr_bps
            details["threshold"] = self.block_atr_bps
            return (True, REASON_ATR_TOO_HIGH, details)
        
        # Check vol regime
        if vol_regime and vol_regime in self.block_vol_regimes:
            details["vol_regime"] = vol_regime
            return (True, REASON_VOL_EXTREME, details)
        
        # Check session-specific thresholds
        if session == "US" and self.us_cfg.get("enabled", False):
            us_spread_threshold = self.us_cfg.get("block_if_spread_bps_gte", 3000)
            if spread_bps is not None and spread_bps >= us_spread_threshold:
                details["spread_bps"] = spread_bps
                details["threshold"] = us_spread_threshold
                details["session"] = session
                return (True, REASON_US_SPREAD_TOO_HIGH, details)
        
        if session == "OVERLAP" and self.overlap_cfg.get("enabled", False):
            overlap_spread_threshold = self.overlap_cfg.get("block_if_spread_bps_gte", 3200)
            if spread_bps is not None and spread_bps >= overlap_spread_threshold:
                details["spread_bps"] = spread_bps
                details["threshold"] = overlap_spread_threshold
                details["session"] = session
                return (True, REASON_OVERLAP_SPREAD_TOO_HIGH, details)
        
        # All checks passed - allow entry
        return (False, None, details)
    
    def cooldown_allows(self, now_bar_index: int, cooldown_bars: int) -> bool:
        """
        Check if cooldown allows entry at current bar index.
        
        Args:
            now_bar_index: Current bar index
            cooldown_bars: Number of bars to wait after last entry
        
        Returns:
            True if cooldown allows entry, False otherwise
        """
        if cooldown_bars <= 0:
            return True
        
        if self._last_entry_bar_index is None:
            return True
        
        bars_since_entry = now_bar_index - self._last_entry_bar_index
        return bars_since_entry > cooldown_bars
    
    def record_entry(self, bar_index: int) -> None:
        """
        Record that an entry occurred at this bar index (for cooldown tracking).
        
        Args:
            bar_index: Bar index where entry occurred
        """
        self._last_entry_bar_index = bar_index
    
    def get_session_clamp(self, session: Optional[str]) -> Optional[float]:
        """
        Get session-specific min_prob_long clamp adjustment.
        
        Args:
            session: Session name (US, OVERLAP, etc.)
        
        Returns:
            Extra min_prob_long adjustment (or None if no clamp)
        """
        if not self.enabled:
            return None
        
        if session == "US" and self.us_cfg.get("enabled", False):
            return self.us_cfg.get("extra_min_prob_long", 0.0)
        
        if session == "OVERLAP" and self.overlap_cfg.get("enabled", False):
            return self.overlap_cfg.get("extra_min_prob_long", 0.0)
        
        return None

