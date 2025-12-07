#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXIT_FARM_V2_RULES - Simple rule-based exit policy for FARM_V2B.

Three explicit rules:
A: Profit-capture (exit on +6 to +9 bps, adaptive trailing stop)
B: Fast loss-cut (exit if MAE < -4 bps before bar 6)
C: Time-based abandonment (exit within bar 8 if PnL < +2 bps)

Design Philosophy:
- Low complexity, explicit rules
- No meta-model or regime modeling
- No dynamic sizing
- Quick improvement over baseline timeout-heavy exit
- Create dataset for future AI-exit training
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

from gx1.utils.pnl import compute_pnl_bps

logger = logging.getLogger(__name__)


@dataclass
class ExitDecision:
    """Exit decision result."""
    exit_price: float
    reason: str
    bars_held: int
    pnl_bps: float
    mae_bps: float  # Maximum Adverse Excursion
    mfe_bps: float  # Maximum Favorable Excursion


class ExitFarmV2Rules:
    """
    Rule-based exit policy for FARM_V2B.
    
    Implements three explicit rules (A, B, C) that can be enabled/disabled independently.
    """
    
    def __init__(
        self,
        enable_rule_a: bool = False,  # Profit-capture
        enable_rule_b: bool = False,  # Fast loss-cut
        enable_rule_c: bool = False,  # Time-based abandonment
        # Rule A parameters
        rule_a_profit_min_bps: float = 6.0,  # Minimum profit to exit
        rule_a_profit_max_bps: float = 9.0,  # Maximum profit target
        rule_a_adaptive_threshold_bps: float = 4.0,  # If > this within 3 bars, use trailing stop
        rule_a_trailing_stop_bps: float = 2.0,  # Trailing stop distance
        rule_a_adaptive_bars: int = 3,  # Bars to check for adaptive behavior
        # Rule B parameters
        rule_b_mae_threshold_bps: float = -4.0,  # MAE threshold
        rule_b_max_bars: int = 6,  # Only apply before this bar
        # Rule C parameters
        rule_c_timeout_bars: int = 8,  # Exit within this many bars
        rule_c_min_profit_bps: float = 2.0,  # Minimum profit to avoid exit
        debug_trade_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            enable_rule_a: Enable profit-capture rule
            enable_rule_b: Enable fast loss-cut rule
            enable_rule_c: Enable time-based abandonment rule
            rule_a_profit_min_bps: Minimum profit to exit (Rule A)
            rule_a_profit_max_bps: Maximum profit target (Rule A)
            rule_a_adaptive_threshold_bps: Threshold for adaptive trailing stop (Rule A)
            rule_a_trailing_stop_bps: Trailing stop distance (Rule A)
            rule_a_adaptive_bars: Bars to check for adaptive behavior (Rule A)
            rule_b_mae_threshold_bps: MAE threshold for fast exit (Rule B)
            rule_b_max_bars: Only apply Rule B before this bar (Rule B)
            rule_c_timeout_bars: Exit within this many bars if profit too low (Rule C)
            rule_c_min_profit_bps: Minimum profit to avoid Rule C exit (Rule C)
        """
        self.enable_rule_a = enable_rule_a
        self.enable_rule_b = enable_rule_b
        self.enable_rule_c = enable_rule_c
        
        # Rule A parameters
        self.rule_a_profit_min_bps = rule_a_profit_min_bps
        self.rule_a_profit_max_bps = rule_a_profit_max_bps
        self.rule_a_adaptive_threshold_bps = rule_a_adaptive_threshold_bps
        self.rule_a_trailing_stop_bps = rule_a_trailing_stop_bps
        self.rule_a_adaptive_bars = rule_a_adaptive_bars
        
        # Rule B parameters
        self.rule_b_mae_threshold_bps = rule_b_mae_threshold_bps
        self.rule_b_max_bars = rule_b_max_bars
        
        # Rule C parameters
        self.rule_c_timeout_bars = rule_c_timeout_bars
        self.rule_c_min_profit_bps = rule_c_min_profit_bps
        self.debug_trade_ids = {tid.strip() for tid in (debug_trade_ids or []) if tid}
        
        # State
        self.entry_price: Optional[float] = None
        self.entry_bid: Optional[float] = None
        self.entry_ask: Optional[float] = None
        self.entry_ts = None
        self.side: str = "long"
        self.bars_held: int = 0
        self.mae_bps: float = 0.0  # Maximum Adverse Excursion (worst PnL so far)
        self.mfe_bps: float = 0.0  # Maximum Favorable Excursion (best PnL so far)
        self.rule_a_trailing_active: bool = False
        self.rule_a_trailing_high: float = 0.0  # Highest PnL when trailing is active
        self.trade_id: Optional[str] = None
        
        logger.info(
            f"[EXIT_FARM_V2_RULES] Initialized: "
            f"A={enable_rule_a}, B={enable_rule_b}, C={enable_rule_c}"
        )
    
    def reset_on_entry(self, entry_bid: float, entry_ask: float, entry_ts, side: str = "long", trade_id: Optional[str] = None) -> None:
        """
        Reset state for a new entry.
        
        Args:
            entry_bid: Entry bid price
            entry_ask: Entry ask price
            entry_ts: Entry timestamp
            side: "long" or "short" (only "long" supported for now)
        """
        if side != "long":
            raise ValueError(f"EXIT_FARM_V2_RULES only supports LONG positions, got: {side}")
        
        self.entry_bid = entry_bid
        self.entry_ask = entry_ask
        self.entry_price = entry_ask if side == "long" else entry_bid
        self.entry_ts = entry_ts
        self.side = side
        self.bars_held = 0
        self.mae_bps = 0.0
        self.mfe_bps = 0.0
        self.rule_a_trailing_active = False
        self.rule_a_trailing_high = 0.0
        self.trade_id = trade_id
        
        logger.debug(
            f"[EXIT_FARM_V2_RULES] Reset: entry_price={self.entry_price:.5f}, "
            f"rules A={self.enable_rule_a}, B={self.enable_rule_b}, C={self.enable_rule_c}, trade_id={self.trade_id}"
        )
    
    def on_bar(self, price_bid: float, price_ask: float, ts) -> Optional[ExitDecision]:
        """
        Process a new bar and check for exit conditions.
        
        Args:
            price_bid: Current bar bid close
            price_ask: Current bar ask close
            ts: Current bar timestamp
        
        Returns:
            ExitDecision if exit triggered, None otherwise
        """
        if self.entry_bid is None or self.entry_ask is None:
            raise RuntimeError("reset_on_entry() must be called before on_bar()")
        
        self.bars_held += 1
        
        # Calculate PnL in bps
        pnl_bps = compute_pnl_bps(self.entry_bid, self.entry_ask, price_bid, price_ask, self.side)
        
        # Update MAE and MFE
        if pnl_bps < self.mae_bps:
            self.mae_bps = pnl_bps
        if pnl_bps > self.mfe_bps:
            self.mfe_bps = pnl_bps
        
        # Update trailing stop high if active
        if self.rule_a_trailing_active:
            if pnl_bps > self.rule_a_trailing_high:
                self.rule_a_trailing_high = pnl_bps
        
        # ============================================================================
        # RULE A: Profit-capture (exit on +6 to +9 bps, adaptive trailing stop)
        # ============================================================================
        if self.enable_rule_a:
            # Check for adaptive trailing stop activation
            if (not self.rule_a_trailing_active and 
                self.bars_held <= self.rule_a_adaptive_bars and
                pnl_bps >= self.rule_a_adaptive_threshold_bps):
                # Activate trailing stop
                self.rule_a_trailing_active = True
                self.rule_a_trailing_high = pnl_bps
                logger.debug(
                    f"[EXIT_FARM_V2_RULES] Rule A: Trailing stop activated at bar {self.bars_held}, "
                    f"pnl_bps={pnl_bps:.2f}"
                )
            
            # Check trailing stop exit
            if self.rule_a_trailing_active:
                trailing_exit_price = self.rule_a_trailing_high - self.rule_a_trailing_stop_bps
                if pnl_bps <= trailing_exit_price:
                    logger.debug(
                        f"[EXIT_FARM_V2_RULES] Rule A: Trailing stop hit: "
                        f"bars_held={self.bars_held}, pnl_bps={pnl_bps:.2f}, "
                        f"trailing_high={self.rule_a_trailing_high:.2f}"
                    )
                    # Exit price: use bid for LONG (we sell at bid), ask for SHORT (we buy back at ask)
                    exit_price = float(price_bid if self.side == "long" else price_ask)
                    decision = ExitDecision(
                        exit_price=exit_price,
                        reason="RULE_A_TRAILING",
                        bars_held=self.bars_held,
                        pnl_bps=pnl_bps,
                        mae_bps=self.mae_bps,
                        mfe_bps=self.mfe_bps,
                    )
                    self._maybe_log_decision(decision)
                    return decision
            
            # Check profit target exit (if not using trailing stop)
            if not self.rule_a_trailing_active:
                if self.rule_a_profit_min_bps <= pnl_bps <= self.rule_a_profit_max_bps:
                    logger.debug(
                        f"[EXIT_FARM_V2_RULES] Rule A: Profit target hit: "
                        f"bars_held={self.bars_held}, pnl_bps={pnl_bps:.2f}"
                    )
                    # Exit price: use bid for LONG (we sell at bid), ask for SHORT (we buy back at ask)
                    exit_price = float(price_bid if self.side == "long" else price_ask)
                    decision = ExitDecision(
                        exit_price=exit_price,
                        reason="RULE_A_PROFIT",
                        bars_held=self.bars_held,
                        pnl_bps=pnl_bps,
                        mae_bps=self.mae_bps,
                        mfe_bps=self.mfe_bps,
                    )
                    self._maybe_log_decision(decision)
                    return decision
        
        # ============================================================================
        # RULE B: Fast loss-cut (exit if MAE < -4 bps before bar 6)
        # ============================================================================
        if self.enable_rule_b:
            if self.bars_held < self.rule_b_max_bars and self.mae_bps <= self.rule_b_mae_threshold_bps:
                logger.debug(
                    f"[EXIT_FARM_V2_RULES] Rule B: Fast loss-cut triggered: "
                    f"bars_held={self.bars_held}, mae_bps={self.mae_bps:.2f}, "
                    f"current_pnl_bps={pnl_bps:.2f}"
                )
                # Exit price: use bid for LONG (we sell at bid), ask for SHORT (we buy back at ask)
                exit_price = float(price_bid if self.side == "long" else price_ask)
                decision = ExitDecision(
                    exit_price=exit_price,
                    reason="RULE_B_FAST_LOSS",
                    bars_held=self.bars_held,
                    pnl_bps=pnl_bps,
                    mae_bps=self.mae_bps,
                    mfe_bps=self.mfe_bps,
                )
                self._maybe_log_decision(decision)
                return decision
        
        # ============================================================================
        # RULE C: Time-based abandonment (exit within bar 8 if PnL < +2 bps)
        # ============================================================================
        if self.enable_rule_c:
            if self.bars_held >= self.rule_c_timeout_bars and pnl_bps < self.rule_c_min_profit_bps:
                logger.debug(
                    f"[EXIT_FARM_V2_RULES] Rule C: Time-based abandonment: "
                    f"bars_held={self.bars_held}, pnl_bps={pnl_bps:.2f}"
                )
                # Exit price: use bid for LONG (we sell at bid), ask for SHORT (we buy back at ask)
                exit_price = float(price_bid if self.side == "long" else price_ask)
                decision = ExitDecision(
                    exit_price=exit_price,
                    reason="RULE_C_TIMEOUT",
                    bars_held=self.bars_held,
                    pnl_bps=pnl_bps,
                    mae_bps=self.mae_bps,
                    mfe_bps=self.mfe_bps,
                )
                self._maybe_log_decision(decision)
                return decision
        
        # No exit triggered
        return None
    
    def _maybe_log_decision(self, decision: ExitDecision) -> None:
        """Emit verbose logging for selected trades."""
        if not self.debug_trade_ids or not self.trade_id:
            return
        if self.trade_id in self.debug_trade_ids:
            logger.info(
                "[EXIT_FARM_V2_RULES][DEBUG] trade_id=%s entry_ts=%s exit_reason=%s "
                "entry_price=%.5f exit_price=%.5f pnl_bps=%.2f bars=%d mfe=%.2f mae=%.2f",
                self.trade_id,
                self.entry_ts,
                decision.reason,
                self.entry_price,
                decision.exit_price,
                decision.pnl_bps,
                decision.bars_held,
                decision.mfe_bps,
                decision.mae_bps,
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for debugging."""
        return {
            "bars_held": self.bars_held,
            "mae_bps": self.mae_bps,
            "mfe_bps": self.mfe_bps,
            "rule_a_trailing_active": self.rule_a_trailing_active,
            "rule_a_trailing_high": self.rule_a_trailing_high,
        }


def get_exit_policy_farm_v2_rules(
    enable_rule_a: bool = False,
    enable_rule_b: bool = False,
    enable_rule_c: bool = False,
    **kwargs
) -> ExitFarmV2Rules:
    """
    Factory function to create ExitFarmV2Rules instance.
    
    Args:
        enable_rule_a: Enable profit-capture rule
        enable_rule_b: Enable fast loss-cut rule
        enable_rule_c: Enable time-based abandonment rule
        **kwargs: Additional parameters for ExitFarmV2Rules
    
    Returns:
        ExitFarmV2Rules instance
    """
    return ExitFarmV2Rules(
        enable_rule_a=enable_rule_a,
        enable_rule_b=enable_rule_b,
        enable_rule_c=enable_rule_c,
        **kwargs
    )
