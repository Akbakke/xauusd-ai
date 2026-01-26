"""
Pre-Entry Wait Gate - DEL 1: Definer PRE-ENTRY "WAIT-FOR-DIP"

Arkitektur-prinsipp (DEL 0):
- Ingen exit skal "drepe gode MFE-trades"
- Hvis vi må "drepe" pga MFE → entry var for tidlig
- MFE-basert logikk hører hjemme før entry, på lik linje med regime/ATR/spread
- ExitV2 beholdes som sikkerhetsnett, ikke timing-motor

Gate-semantikk:
- PASS → entry evalueres normalt
- WAIT → ingen entry evalueres (ingen trade, ingen score, ingen exit senere)
- IKKE: generer entry og "avbryt" eller la exit ta dette

Dette er en før-entry beslutning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class WaitReason(Enum):
    """Reasons for waiting (not entering)."""
    PASS = "pass"  # Gate passed, entry can be evaluated
    PULLBACK_DEPTH = "pullback_depth"  # Price too far from recent high
    BARS_SINCE_LOW = "bars_since_low"  # Too soon after local low
    ADVERSE_MOVE = "adverse_move"  # Recent adverse move too large
    DISTANCE_TO_EMA = "distance_to_ema"  # Too far from fast EMA
    VOLATILITY_COOLING = "volatility_cooling"  # ATR still falling (volatility cooling)


@dataclass
class PreEntryWaitGateConfig:
    """
    Configuration for Pre-Entry Wait Gate.
    
    All thresholds are in ATR-normalized units (except where noted).
    """
    enabled: bool = False
    
    # Pullback depth: wait if price is too far below recent high
    pullback_depth_atr_max: float = 2.0  # Wait if pullback > 2.0 ATR
    
    # Bars since local low: wait if too soon after local low
    bars_since_low_min: int = 5  # Wait if < 5 bars since local low
    
    # Recent adverse move: wait if recent move against direction is too large
    adverse_move_bars: int = 10  # Look back N bars for adverse move
    adverse_move_atr_max: float = 1.5  # Wait if adverse move > 1.5 ATR
    
    # Distance to EMA: wait if too far from fast EMA
    ema_fast_period: int = 20  # Fast EMA period
    distance_to_ema_atr_max: float = 1.0  # Wait if distance > 1.0 ATR
    
    # Volatility cooling: wait if ATR is still falling (volatility cooling)
    atr_cooling_bars: int = 5  # Look back N bars for ATR trend
    atr_cooling_threshold: float = -0.1  # Wait if ATR change < -0.1 (falling)
    
    # Logging
    log_wait_reasons: bool = True


@dataclass
class PreEntryWaitGate:
    """
    Pre-Entry Wait Gate - evaluates if we should wait before entry evaluation.
    
    Features (DEL 1.1):
    - pullback_depth_atr: (recent_high - price) / ATR
    - bars_since_local_low: number of bars since local low
    - recent_adverse_move_bps: adverse move over N bars
    - distance_to_ema_fast / ATR: distance to fast EMA normalized by ATR
    - volatility_cooling: ATR falling vs peak
    
    All features are:
    - read-only (don't modify input data)
    - use existing features where possible
    - cheap to compute
    """
    
    config: PreEntryWaitGateConfig
    
    # Counters (for telemetry)
    counters: dict[str, int] = field(default_factory=lambda: {
        "pre_entry_wait_n_total": 0,
        "pre_entry_wait_n_wait": 0,
        "pre_entry_wait_n_pass": 0,
        "pre_entry_wait_n_pullback_depth": 0,
        "pre_entry_wait_n_bars_since_low": 0,
        "pre_entry_wait_n_adverse_move": 0,
        "pre_entry_wait_n_distance_to_ema": 0,
        "pre_entry_wait_n_volatility_cooling": 0,
    })
    
    def should_wait(
        self,
        candles: pd.DataFrame,
        atr_bps: float,
        current_price: float,
        entry_direction: Optional[str] = None,  # "long" or "short" (optional, for adverse move)
    ) -> tuple[bool, WaitReason]:
        """
        Evaluate if we should wait before entry evaluation.
        
        Args:
            candles: Candles DataFrame (must have at least N bars for lookback)
            atr_bps: ATR in basis points
            current_price: Current price (close or mid)
            entry_direction: Optional entry direction ("long" or "short") for adverse move check
        
        Returns:
            (should_wait: bool, reason: WaitReason)
            - If should_wait=True, reason indicates why (not PASS)
            - If should_wait=False, reason=PASS
        """
        if not self.config.enabled:
            return False, WaitReason.PASS
        
        self.counters["pre_entry_wait_n_total"] += 1
        
        # Convert ATR from bps to absolute
        atr_absolute = (atr_bps / 10000.0) * current_price if atr_bps > 0 and current_price > 0 else 0.0
        
        if atr_absolute <= 0:
            # ATR not available - pass (don't block)
            log.debug("[PRE_ENTRY_WAIT] ATR not available, passing")
            self.counters["pre_entry_wait_n_pass"] += 1
            return False, WaitReason.PASS
        
        # Need minimum bars for lookback
        min_bars_needed = max(
            self.config.bars_since_low_min,
            self.config.adverse_move_bars,
            self.config.ema_fast_period,
            self.config.atr_cooling_bars,
        )
        
        if len(candles) < min_bars_needed:
            # Not enough bars - pass (don't block)
            log.debug(f"[PRE_ENTRY_WAIT] Not enough bars ({len(candles)} < {min_bars_needed}), passing")
            self.counters["pre_entry_wait_n_pass"] += 1
            return False, WaitReason.PASS
        
        # Get price series (use mid if available, otherwise close)
        if "bid_close" in candles.columns and "ask_close" in candles.columns:
            price_series = (candles["bid_close"] + candles["ask_close"]) / 2.0
        else:
            price_series = candles.get("close", candles.index)
        
        # DEL 1.1: Compute features
        
        # 1. Pullback depth: (recent_high - price) / ATR
        lookback_for_high = min(50, len(price_series))  # Look back up to 50 bars
        recent_high = price_series.iloc[-lookback_for_high:].max()
        pullback_depth_atr = (recent_high - current_price) / atr_absolute if atr_absolute > 0 else 0.0
        
        if pullback_depth_atr > self.config.pullback_depth_atr_max:
            if self.config.log_wait_reasons:
                log.debug(
                    "[PRE_ENTRY_WAIT] WAIT: pullback_depth_atr=%.2f > %.2f",
                    pullback_depth_atr,
                    self.config.pullback_depth_atr_max,
                )
            self.counters["pre_entry_wait_n_wait"] += 1
            self.counters["pre_entry_wait_n_pullback_depth"] += 1
            return True, WaitReason.PULLBACK_DEPTH
        
        # 2. Bars since local low
        lookback_for_low = min(50, len(price_series))
        recent_low_idx = price_series.iloc[-lookback_for_low:].idxmin()
        recent_low_pos = price_series.index.get_loc(recent_low_idx)
        current_pos = len(price_series) - 1
        bars_since_low = current_pos - recent_low_pos
        
        if bars_since_low < self.config.bars_since_low_min:
            if self.config.log_wait_reasons:
                log.debug(
                    "[PRE_ENTRY_WAIT] WAIT: bars_since_low=%d < %d",
                    bars_since_low,
                    self.config.bars_since_low_min,
                )
            self.counters["pre_entry_wait_n_wait"] += 1
            self.counters["pre_entry_wait_n_bars_since_low"] += 1
            return True, WaitReason.BARS_SINCE_LOW
        
        # 3. Recent adverse move (if entry_direction provided)
        if entry_direction:
            adverse_move_bps = 0.0
            if entry_direction == "long":
                # For long: adverse move = price drop
                lookback = min(self.config.adverse_move_bars, len(price_series))
                recent_low_in_window = price_series.iloc[-lookback:].min()
                adverse_move_bps = ((current_price - recent_low_in_window) / current_price) * 10000.0
            elif entry_direction == "short":
                # For short: adverse move = price rise
                lookback = min(self.config.adverse_move_bars, len(price_series))
                recent_high_in_window = price_series.iloc[-lookback:].max()
                adverse_move_bps = ((recent_high_in_window - current_price) / current_price) * 10000.0
            
            adverse_move_atr = adverse_move_bps / atr_bps if atr_bps > 0 else 0.0
            
            if adverse_move_atr > self.config.adverse_move_atr_max:
                if self.config.log_wait_reasons:
                    log.debug(
                        "[PRE_ENTRY_WAIT] WAIT: adverse_move_atr=%.2f > %.2f (direction=%s)",
                        adverse_move_atr,
                        self.config.adverse_move_atr_max,
                        entry_direction,
                    )
                self.counters["pre_entry_wait_n_wait"] += 1
                self.counters["pre_entry_wait_n_adverse_move"] += 1
                return True, WaitReason.ADVERSE_MOVE
        
        # 4. Distance to EMA fast
        ema_fast = price_series.ewm(span=self.config.ema_fast_period, adjust=False).mean()
        current_ema = ema_fast.iloc[-1]
        distance_to_ema = abs(current_price - current_ema)
        distance_to_ema_atr = distance_to_ema / atr_absolute if atr_absolute > 0 else 0.0
        
        if distance_to_ema_atr > self.config.distance_to_ema_atr_max:
            if self.config.log_wait_reasons:
                log.debug(
                    "[PRE_ENTRY_WAIT] WAIT: distance_to_ema_atr=%.2f > %.2f",
                    distance_to_ema_atr,
                    self.config.distance_to_ema_atr_max,
                )
            self.counters["pre_entry_wait_n_wait"] += 1
            self.counters["pre_entry_wait_n_distance_to_ema"] += 1
            return True, WaitReason.DISTANCE_TO_EMA
        
        # 5. Volatility cooling (ATR falling)
        if len(candles) >= self.config.atr_cooling_bars + 1:
            # Compute ATR over lookback window
            if "bid_close" in candles.columns and "ask_close" in candles.columns:
                close = (candles["bid_close"] + candles["ask_close"]) / 2.0
                high = (candles.get("bid_high", candles.get("high", close)) + 
                       candles.get("ask_high", candles.get("high", close))) / 2.0
                low = (candles.get("bid_low", candles.get("low", close)) + 
                      candles.get("ask_low", candles.get("low", close))) / 2.0
            else:
                close = candles.get("close", candles.index)
                high = candles.get("high", close)
                low = candles.get("low", close)
            
            # True Range
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            
            # ATR14
            atr14 = tr.rolling(window=14, min_periods=1).mean()
            
            # Check if ATR is falling
            if len(atr14) >= self.config.atr_cooling_bars + 1:
                current_atr = atr14.iloc[-1]
                past_atr = atr14.iloc[-(self.config.atr_cooling_bars + 1)]
                atr_change = (current_atr - past_atr) / past_atr if past_atr > 0 else 0.0
                
                if atr_change < self.config.atr_cooling_threshold:
                    if self.config.log_wait_reasons:
                        log.debug(
                            "[PRE_ENTRY_WAIT] WAIT: atr_cooling=%.4f < %.4f (ATR falling)",
                            atr_change,
                            self.config.atr_cooling_threshold,
                        )
                    self.counters["pre_entry_wait_n_wait"] += 1
                    self.counters["pre_entry_wait_n_volatility_cooling"] += 1
                    return True, WaitReason.VOLATILITY_COOLING
        
        # All checks passed
        self.counters["pre_entry_wait_n_pass"] += 1
        return False, WaitReason.PASS
    
    def get_counters(self) -> dict[str, int]:
        """Get current counters for telemetry."""
        return self.counters.copy()
    
    def reset_counters(self) -> None:
        """Reset counters (for testing)."""
        self.counters = {
            "pre_entry_wait_n_total": 0,
            "pre_entry_wait_n_wait": 0,
            "pre_entry_wait_n_pass": 0,
            "pre_entry_wait_n_pullback_depth": 0,
            "pre_entry_wait_n_bars_since_low": 0,
            "pre_entry_wait_n_adverse_move": 0,
            "pre_entry_wait_n_distance_to_ema": 0,
            "pre_entry_wait_n_volatility_cooling": 0,
        }


def load_pre_entry_wait_gate_config(config_path: Optional[str] = None) -> PreEntryWaitGateConfig:
    """
    Load Pre-Entry Wait Gate configuration from YAML or environment.
    
    Args:
        config_path: Optional path to YAML config file
    
    Returns:
        PreEntryWaitGateConfig
    """
    import os
    import yaml
    from pathlib import Path
    
    # Check environment variable first
    enabled_env = os.getenv("GX1_PRE_ENTRY_WAIT_GATE_ENABLED", "0") == "1"
    
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
            return PreEntryWaitGateConfig(**config_dict)
    
    # Default config (can be overridden by env vars)
    return PreEntryWaitGateConfig(
        enabled=enabled_env,
        pullback_depth_atr_max=float(os.getenv("GX1_PRE_ENTRY_PULLBACK_ATR_MAX", "2.0")),
        bars_since_low_min=int(os.getenv("GX1_PRE_ENTRY_BARS_SINCE_LOW_MIN", "5")),
        adverse_move_bars=int(os.getenv("GX1_PRE_ENTRY_ADVERSE_MOVE_BARS", "10")),
        adverse_move_atr_max=float(os.getenv("GX1_PRE_ENTRY_ADVERSE_MOVE_ATR_MAX", "1.5")),
        ema_fast_period=int(os.getenv("GX1_PRE_ENTRY_EMA_FAST_PERIOD", "20")),
        distance_to_ema_atr_max=float(os.getenv("GX1_PRE_ENTRY_DISTANCE_EMA_ATR_MAX", "1.0")),
        atr_cooling_bars=int(os.getenv("GX1_PRE_ENTRY_ATR_COOLING_BARS", "5")),
        atr_cooling_threshold=float(os.getenv("GX1_PRE_ENTRY_ATR_COOLING_THRESHOLD", "-0.1")),
        log_wait_reasons=os.getenv("GX1_PRE_ENTRY_LOG_WAIT_REASONS", "1") == "1",
    )
