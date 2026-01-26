"""
Replay feature parity helpers.

Ensures that replay mode has the same tags (session, vol, trend) as live mode,
computed from candles/history when features are missing.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

# DEL 3: PREBUILT mode fix - move live_features imports to lazy imports
# live_features is forbidden in PREBUILT mode, so we only import it when needed (live mode only)
# infer_session_tag is used at runtime, so we import it locally where needed

log = logging.getLogger(__name__)

# Module-level flags for rate-limited logging
_vol_regime_logged = False
_trend_regime_logged = False


def ensure_replay_tags(
    row: Union[pd.Series, pd.DataFrame],
    candles: Optional[pd.DataFrame],
    policy_state: Dict,
    *,
    current_ts: Optional[pd.Timestamp] = None,
    min_bars_for_atr: int = 14,
    min_bars_for_trend: int = 20,
) -> Tuple[Union[pd.Series, pd.DataFrame], Dict]:
    """
    Ensure replay tags (session, vol_regime, trend_regime) are set from candles/history.
    
    This function ensures backward compatibility: only sets tags if they are missing or UNKNOWN.
    If tags already exist and are not UNKNOWN, they are preserved.
    
    Parameters
    ----------
    row : pd.Series or pd.DataFrame
        Current row(s) to update with tags
    candles : pd.DataFrame, optional
        Historical candles for computing ATR/trend (must have OHLC columns)
    policy_state : dict
        Policy state dict to update with tags
    current_ts : pd.Timestamp, optional
        Current timestamp (if None, inferred from row index or candles)
    min_bars_for_atr : int, default 14
        Minimum bars needed for ATR computation
    min_bars_for_trend : int, default 20
        Minimum bars needed for trend computation
    
    Returns
    -------
    Tuple[row, policy_state]
        Updated row and policy_state with tags set
    """
    # Extract current timestamp
    if current_ts is None:
        if isinstance(row.index, pd.DatetimeIndex) and len(row.index) > 0:
            current_ts = row.index[-1] if isinstance(row, pd.DataFrame) else row.index[0]
        elif candles is not None and len(candles) > 0:
            current_ts = candles.index[-1]
        else:
            log.warning("[REPLAY_TAGS] Cannot infer timestamp, skipping tag computation")
            return row, policy_state
    
    # Ensure timezone-aware
    if current_ts.tzinfo is None:
        current_ts = pd.Timestamp(current_ts, tz="UTC")
    
    # ============================================================================
    # 1. SESSION TAG
    # ============================================================================
    session_tag = None
    
    # Check if session already set in policy_state
    if "session" in policy_state and policy_state["session"] != "UNKNOWN":
        session_tag = policy_state["session"]
    elif isinstance(row, pd.Series) and "session" in row.index and row["session"] != "UNKNOWN":
        session_tag = row["session"]
    elif isinstance(row, pd.DataFrame) and "session" in row.columns:
        session_val = row["session"].iloc[-1] if len(row) > 0 else None
        if session_val not in (None, "UNKNOWN", np.nan):
            session_tag = session_val
    
    # Compute from timestamp if missing
    if session_tag is None or session_tag == "UNKNOWN":
        from gx1.execution.live_features import infer_session_tag
        session_tag = infer_session_tag(current_ts)
        policy_state["session"] = session_tag
        policy_state["session_id"] = {"EU": 0, "OVERLAP": 1, "US": 2, "ASIA": 3}.get(session_tag, 0)
        
        # Also update row
        if isinstance(row, pd.Series):
            row["session"] = session_tag
        elif isinstance(row, pd.DataFrame):
            # For DataFrame, set session column for all rows
            row["session"] = session_tag
            # Also ensure it's accessible via iloc[0]
            if len(row) > 0:
                row.iloc[0, row.columns.get_loc("session")] = session_tag
    
    # ============================================================================
    # 2. VOL REGIME (ATR-based)
    # ============================================================================
    vol_regime = None
    atr_regime_id = None
    
    # Check if vol_regime already set
    if "brain_vol_regime" in policy_state and policy_state["brain_vol_regime"] != "UNKNOWN":
        vol_regime = policy_state["brain_vol_regime"]
    elif "vol_regime" in policy_state and policy_state["vol_regime"] != "UNKNOWN":
        vol_regime = policy_state["vol_regime"]
    elif "atr_regime" in policy_state and policy_state["atr_regime"] != "UNKNOWN":
        vol_regime = policy_state["atr_regime"]
    elif isinstance(row, pd.Series):
        if "vol_regime" in row.index and row["vol_regime"] != "UNKNOWN":
            vol_regime = row["vol_regime"]
        elif "atr_regime" in row.index and row["atr_regime"] != "UNKNOWN":
            vol_regime = row["atr_regime"]
    elif isinstance(row, pd.DataFrame):
        if "vol_regime" in row.columns:
            vol_val = row["vol_regime"].iloc[-1] if len(row) > 0 else None
            if vol_val not in (None, "UNKNOWN", np.nan):
                vol_regime = vol_val
        elif "atr_regime" in row.columns:
            atr_val = row["atr_regime"].iloc[-1] if len(row) > 0 else None
            if atr_val not in (None, "UNKNOWN", np.nan):
                vol_regime = atr_val
    
    # Compute from candles if missing
    if (vol_regime is None or vol_regime == "UNKNOWN") and candles is not None and len(candles) >= min_bars_for_atr:
        try:
            # Compute ATR14 from candles
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
            
            # Calculate ATR14 (True Range then 14-period SMA)
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr14 = tr.rolling(window=14, min_periods=1).mean()
            
            # Get current ATR14 value
            current_atr14 = atr14.iloc[-1]
            
            # Compute percentile-based regime (use rolling percentile over last 100 bars)
            lookback = min(100, len(atr14))
            if lookback >= 14:
                atr14_window = atr14.iloc[-lookback:]
                atr14_pct = atr14_window.rank(pct=True).iloc[-1]
                
                # Map to regime: 0=LOW (0-33%), 1=MEDIUM (33-67%), 2=HIGH (67-100%)
                if atr14_pct < 0.33:
                    atr_regime_id = 0  # LOW
                elif atr14_pct < 0.67:
                    atr_regime_id = 1  # MEDIUM
                else:
                    atr_regime_id = 2  # HIGH
                
                mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
                vol_regime = mapping.get(atr_regime_id, "UNKNOWN")
                
                # Update policy_state
                policy_state["brain_vol_regime"] = vol_regime
                policy_state["vol_regime"] = vol_regime
                policy_state["atr_regime"] = vol_regime
                policy_state["_atr_regime_id"] = atr_regime_id
                
                # Also update row
                if isinstance(row, pd.Series):
                    row["vol_regime"] = vol_regime
                    row["atr_regime"] = vol_regime
                    row["_v1_atr_regime_id"] = atr_regime_id
                elif isinstance(row, pd.DataFrame):
                    row["vol_regime"] = vol_regime
                    row["atr_regime"] = vol_regime
                    row["_v1_atr_regime_id"] = atr_regime_id
                
                # Log once per run (module-level flag)
                global _vol_regime_logged
                if not _vol_regime_logged:
                    log.info(
                        "[REPLAY_TAGS] Computed vol_regime from candles: atr14_pct=%.3f -> regime=%s (id=%d)",
                        atr14_pct, vol_regime, atr_regime_id
                    )
                    _vol_regime_logged = True
        except Exception as e:
            log.warning("[REPLAY_TAGS] Failed to compute vol_regime from candles: %s", e)
    
    # ============================================================================
    # 3. TREND REGIME (EMA/ADX-based, simplified)
    # ============================================================================
    trend_regime = None
    
    # Check if trend_regime already set
    if "brain_trend_regime" in policy_state and policy_state["brain_trend_regime"] != "UNKNOWN":
        trend_regime = policy_state["brain_trend_regime"]
    elif "trend_regime" in policy_state and policy_state["trend_regime"] != "UNKNOWN":
        trend_regime = policy_state["trend_regime"]
    elif isinstance(row, pd.Series) and "trend_regime" in row.index and row["trend_regime"] != "UNKNOWN":
        trend_regime = row["trend_regime"]
    elif isinstance(row, pd.DataFrame) and "trend_regime" in row.columns:
        trend_val = row["trend_regime"].iloc[-1] if len(row) > 0 else None
        if trend_val not in (None, "UNKNOWN", np.nan):
            trend_regime = trend_val
    
    # Compute from candles if missing (simplified EMA-based trend)
    if (trend_regime is None or trend_regime == "UNKNOWN") and candles is not None and len(candles) >= min_bars_for_trend:
        try:
            # Use mid-price if bid/ask available
            if "bid_close" in candles.columns and "ask_close" in candles.columns:
                close = (candles["bid_close"] + candles["ask_close"]) / 2.0
            else:
                close = candles.get("close", candles.index)
            
            # Compute EMA12 and EMA26 (simplified trend detection)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            
            # Current EMA values
            current_ema12 = ema12.iloc[-1]
            current_ema26 = ema26.iloc[-1]
            
            # Simple trend: EMA12 > EMA26 = uptrend, else neutral/down
            if current_ema12 > current_ema26 * 1.001:  # 0.1% threshold
                trend_regime = "TREND_UP"
            elif current_ema12 < current_ema26 * 0.999:
                trend_regime = "TREND_DOWN"
            else:
                trend_regime = "TREND_NEUTRAL"
            
            # Update policy_state
            policy_state["brain_trend_regime"] = trend_regime
            policy_state["trend_regime"] = trend_regime
            
            # Also update row
            if isinstance(row, pd.Series):
                row["trend_regime"] = trend_regime
            elif isinstance(row, pd.DataFrame):
                row["trend_regime"] = trend_regime
            
            # Log once per run (module-level flag)
            global _trend_regime_logged
            if not _trend_regime_logged:
                log.info(
                    "[REPLAY_TAGS] Computed trend_regime from candles: ema12=%.2f ema26=%.2f -> regime=%s",
                    current_ema12, current_ema26, trend_regime
                )
                _trend_regime_logged = True
        except Exception as e:
            log.warning("[REPLAY_TAGS] Failed to compute trend_regime from candles: %s", e)
    
    return row, policy_state

