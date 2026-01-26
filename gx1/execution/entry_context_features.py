"""
Entry Context Features - Model Input (not gates).

STEP 2: Context features are INPUT to the model, allowing the model to learn
optimal entries per regime/session/spread context.

This is separate from V9 feature build - context features are cheaper to compute
and available earlier in the pipeline (after soft eligibility).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class EntryContextFeatures:
    """
    Context features for ENTRY_V10 model input.
    
    These are computed after soft eligibility (cheap ATR proxy available),
    but before full V9 feature build.
    
    All features are validated and normalized according to contract.
    """
    # Categorical features (integer IDs)
    session_id: int  # 0=ASIA, 1=EU, 2=US, 3=OVERLAP
    trend_regime_id: int  # 0=TREND_DOWN, 1=TREND_NEUTRAL, 2=TREND_UP
    vol_regime_id: int  # 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
    atr_bucket: int  # 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
    spread_bucket: int  # 0=LOW, 1=MEDIUM, 2=HIGH
    
    # Continuous features (normalized)
    atr_bps: float  # ATR in basis points, clipped [0, 1000]
    spread_bps: float  # Spread in basis points, clipped [0, 500]
    
    # Metadata (for validation/debugging)
    _atr_bps_raw: Optional[float] = None  # Raw ATR before clipping
    _spread_bps_raw: Optional[float] = None  # Raw spread before clipping
    _source: str = "computed"  # Source of features (for debugging)
    
    def to_tensor_categorical(self) -> np.ndarray:
        """
        Convert categorical features to int64 tensor.
        
        Returns:
            int64 array: [session_id, trend_regime_id, vol_regime_id, atr_bucket, spread_bucket]
        """
        return np.array([
            self.session_id,
            self.trend_regime_id,
            self.vol_regime_id,
            self.atr_bucket,
            self.spread_bucket,
        ], dtype=np.int64)
    
    def to_tensor_continuous(self) -> np.ndarray:
        """
        Convert continuous features to float32 tensor (normalized).
        
        Note: Normalization (Z-score) is applied by model/bundle, not here.
        This returns raw values (will be normalized by model).
        
        Returns:
            float32 array: [atr_bps, spread_bps]
        """
        return np.array([
            self.atr_bps,
            self.spread_bps,
        ], dtype=np.float32)
    
    def validate(self, is_replay: bool = True) -> tuple[bool, Optional[str]]:
        """
        Validate context features according to contract.
        
        Args:
            is_replay: If True, hard fail on invalid features
        
        Returns:
            (valid: bool, error_message: Optional[str])
        """
        # Validate categorical IDs
        if not (0 <= self.session_id <= 3):
            return False, f"session_id out of range: {self.session_id} (expected 0-3)"
        
        if not (0 <= self.trend_regime_id <= 2):
            return False, f"trend_regime_id out of range: {self.trend_regime_id} (expected 0-2)"
        
        if not (0 <= self.vol_regime_id <= 3):
            return False, f"vol_regime_id out of range: {self.vol_regime_id} (expected 0-3)"
        
        if not (0 <= self.atr_bucket <= 3):
            return False, f"atr_bucket out of range: {self.atr_bucket} (expected 0-3)"
        
        if not (0 <= self.spread_bucket <= 2):
            return False, f"spread_bucket out of range: {self.spread_bucket} (expected 0-2)"
        
        # Validate continuous features (must be finite)
        if not np.isfinite(self.atr_bps):
            return False, f"atr_bps is not finite: {self.atr_bps}"
        
        if not np.isfinite(self.spread_bps):
            return False, f"spread_bps is not finite: {self.spread_bps}"
        
        # Validate ranges (should be clipped, but double-check)
        if not (0.0 <= self.atr_bps <= 1000.0):
            return False, f"atr_bps out of range: {self.atr_bps} (expected [0, 1000])"
        
        if not (0.0 <= self.spread_bps <= 500.0):
            return False, f"spread_bps out of range: {self.spread_bps} (expected [0, 500])"
        
        return True, None


def build_entry_context_features(
    candles: pd.DataFrame,
    policy_state: dict,
    atr_proxy: Optional[float] = None,
    spread_bps: Optional[float] = None,
    is_replay: bool = True,
) -> EntryContextFeatures:
    """
    Build entry context features from minimal inputs.
    
    This is called after soft eligibility (cheap ATR proxy available),
    but before full V9 feature build.
    
    Args:
        candles: Candles DataFrame (for session inference)
        policy_state: Policy state dict (may contain regime info)
        atr_proxy: Cheap ATR proxy from soft eligibility (optional, will compute if None)
        spread_bps: Spread in bps from hard eligibility (optional, will compute if None)
        is_replay: If True, hard fail on missing features
    
    Returns:
        EntryContextFeatures object with all 7 features
    
    Raises:
        RuntimeError: If required features are missing and is_replay=True
    """
    from gx1.execution.live_features import infer_session_tag
    
    current_ts = candles.index[-1] if len(candles) > 0 else pd.Timestamp.now(tz="UTC")
    
    # 1. session_id (always available, time-based)
    current_session = policy_state.get("session")
    if not current_session:
        current_session = infer_session_tag(current_ts).upper()
        policy_state["session"] = current_session
    
    session_map = {"ASIA": 0, "EU": 1, "US": 2, "OVERLAP": 3}
    session_id = session_map.get(current_session, 0)  # Default to ASIA if unknown
    
    # 2. spread_bps (from hard eligibility or compute)
    if spread_bps is None:
        # Try to get from candles
        try:
            if "bid_close" in candles.columns and "ask_close" in candles.columns:
                bid = candles["bid_close"].iloc[-1]
                ask = candles["ask_close"].iloc[-1]
                if pd.notna(bid) and pd.notna(ask) and bid > 0:
                    spread_price = ask - bid
                    spread_bps_raw = (spread_price / bid) * 10000.0
                    spread_bps = float(spread_bps_raw)
                else:
                    spread_bps = 10.0  # Default
            else:
                spread_bps = 10.0  # Default
        except Exception:
            spread_bps = 10.0  # Default
    
    # Clip spread_bps to [0, 500]
    spread_bps_raw = spread_bps
    spread_bps = max(0.0, min(500.0, float(spread_bps)))
    
    # 3. spread_bucket (derived from spread_bps)
    # Simple percentile buckets (can be improved with historical distribution)
    if spread_bps < 10.0:
        spread_bucket = 0  # LOW
    elif spread_bps < 30.0:
        spread_bucket = 1  # MEDIUM
    else:
        spread_bucket = 2  # HIGH
    
    # 4. atr_bps (from soft eligibility ATR proxy or compute)
    if atr_proxy is None:
        # Compute cheap ATR proxy (same as soft eligibility)
        atr_proxy = _compute_cheap_atr_proxy(candles, window=14)
    
    if atr_proxy is None:
        if is_replay:
            raise RuntimeError(
                "CONTEXT_FEATURE_MISSING: atr_bps unavailable "
                f"(atr_proxy=None, candles_len={len(candles)})"
            )
        # Live mode: use default ATR in bps directly (skip conversion)
        atr_bps_raw = 50.0  # Default (median estimate in bps)
        log.warning("[CONTEXT_FEATURES] atr_bps unavailable, using default=50.0 bps")
    else:
        # Convert ATR proxy to bps
        close = candles.get("close", None)
        if close is not None and len(close) > 0:
            current_price = float(close.iloc[-1])
            if current_price > 0:
                atr_bps_raw = (atr_proxy / current_price) * 10000.0
            else:
                atr_bps_raw = 50.0  # Default
        else:
            atr_bps_raw = 50.0  # Default
    
    # Clip atr_bps to [0, 1000]
    atr_bps = max(0.0, min(1000.0, float(atr_bps_raw)))
    
    # 5. atr_bucket (derived from atr_bps)
    # Simple percentile buckets (can be improved with historical distribution)
    if atr_bps < 30.0:
        atr_bucket = 0  # LOW
    elif atr_bps < 100.0:
        atr_bucket = 1  # MEDIUM
    elif atr_bps < 200.0:
        atr_bucket = 2  # HIGH
    else:
        atr_bucket = 3  # EXTREME
    
    # 6. vol_regime_id (derived from atr_bps percentile, same as atr_bucket)
    vol_regime_id = atr_bucket  # Same mapping for now
    
    # 7. trend_regime_id (from policy_state or fallback)
    trend_regime = policy_state.get("brain_trend_regime", "UNKNOWN")
    trend_map = {
        "TREND_DOWN": 0,
        "TREND_NEUTRAL": 1,
        "TREND_UP": 2,
    }
    trend_regime_id = trend_map.get(trend_regime, 1)  # Default to NEUTRAL if UNKNOWN
    
    # Build context features object
    ctx = EntryContextFeatures(
        session_id=session_id,
        trend_regime_id=trend_regime_id,
        vol_regime_id=vol_regime_id,
        atr_bucket=atr_bucket,
        spread_bucket=spread_bucket,
        atr_bps=atr_bps,
        spread_bps=spread_bps,
        _atr_bps_raw=atr_bps_raw,
        _spread_bps_raw=spread_bps_raw,
        _source="computed",
    )
    
    # Validate
    valid, error_msg = ctx.validate(is_replay=is_replay)
    if not valid:
        if is_replay:
            raise RuntimeError(f"CONTEXT_FEATURE_INVALID: {error_msg}")
        else:
            log.warning(f"[CONTEXT_FEATURES] data_integrity_degraded: {error_msg}")
    
    return ctx


def _compute_cheap_atr_proxy(candles: pd.DataFrame, window: int = 14) -> Optional[float]:
    """
    Compute ultra-cheap ATR proxy from raw candles (no feature build required).
    
    Same implementation as in entry_manager.py soft eligibility.
    """
    if candles.empty or len(candles) < window:
        return None
    
    try:
        # Get OHLC
        high = candles.get("high", None)
        low = candles.get("low", None)
        close = candles.get("close", None)
        
        if high is None or low is None or close is None:
            return None
        
        # Convert to numpy arrays (last window bars)
        high_arr = high.iloc[-window:].values
        low_arr = low.iloc[-window:].values
        close_arr = close.iloc[-window:].values
        
        # True Range components
        tr1 = high_arr - low_arr  # High - Low
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))  # |High - Prev Close|
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))   # |Low - Prev Close|
        
        # True Range = max of three components
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # ATR = mean of True Range
        atr = np.mean(tr)
        
        return float(atr)
    except Exception:
        return None

