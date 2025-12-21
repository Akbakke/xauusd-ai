"""
Runtime regime inputs extractor for SNIPER overlays.

This module provides a single source of truth for extracting regime classification
inputs at runtime, ensuring overlays see the same data as offline analysis.

The extractor searches multiple sources in priority order to find regime inputs:
- prediction object (if available)
- feature_context dict (if available)
- entry_bundle/entry features (if available)
- policy_state (if available)

This ensures overlays can correctly classify regime even when some sources are missing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from gx1.execution.live_features import infer_session_tag

log = logging.getLogger(__name__)


def get_runtime_regime_inputs(
    *,
    prediction: Any = None,
    feature_context: Optional[Dict[str, Any]] = None,
    spread_pct: Optional[float] = None,
    current_atr_bps: Optional[float] = None,
    entry_bundle: Any = None,
    policy_state: Optional[Dict[str, Any]] = None,
    entry_time: Any = None,
) -> Dict[str, Any]:
    """
    Extract regime classification inputs from available runtime sources.
    
    Returns dict with keys:
    - trend_regime: str | None
    - vol_regime: str | None
    - atr_bps: float | None
    - spread_bps: float | None
    - session: str (never None, defaults to "UNKNOWN")
    
    Priority order (first non-None value wins):
    
    session:
    1. prediction.session (if prediction has .session)
    2. policy_state.get("session")
    3. infer_session_tag(entry_time) (if entry_time available)
    4. "UNKNOWN"
    
    trend_regime:
    1. prediction.trend_regime (if prediction has .trend_regime)
    2. policy_state.get("brain_trend_regime")
    3. policy_state.get("trend_regime")
    4. feature_context.get("trend_regime")
    5. None
    
    vol_regime:
    1. prediction.vol_regime (if prediction has .vol_regime)
    2. policy_state.get("brain_vol_regime")
    3. policy_state.get("vol_regime")
    4. feature_context.get("vol_regime")
    5. None
    
    atr_bps:
    1. current_atr_bps (already computed)
    2. feature_context.get("atr_bps")
    3. entry_bundle.atr_bps (if entry_bundle has .atr_bps)
    4. None
    
    spread_bps:
    1. Compute from spread_pct if present (spread_pct * 10000.0)
    2. feature_context.get("spread_bps")
    3. feature_context.get("spread_pct") * 10000.0 (if spread_pct exists)
    4. entry_bundle.spread_bps (if entry_bundle has .spread_bps)
    5. None
    """
    result: Dict[str, Any] = {
        "trend_regime": None,
        "vol_regime": None,
        "atr_bps": None,
        "spread_bps": None,
        "session": "UNKNOWN",
    }
    
    # Track sources for debug logging
    sources: Dict[str, str] = {}
    
    # Extract session
    if prediction is not None:
        try:
            if hasattr(prediction, "session") and prediction.session:
                result["session"] = str(prediction.session)
                sources["session"] = "prediction.session"
        except Exception:
            pass
    
    if result["session"] == "UNKNOWN" and policy_state:
        session_val = policy_state.get("session")
        if session_val:
            result["session"] = str(session_val)
            sources["session"] = "policy_state.session"
    
    if result["session"] == "UNKNOWN" and entry_time is not None:
        try:
            result["session"] = infer_session_tag(entry_time)
            sources["session"] = "infer_session_tag(entry_time)"
        except Exception:
            pass
    
    if result["session"] == "UNKNOWN":
        sources["session"] = "fallback:UNKNOWN"
    
    # Extract trend_regime
    if prediction is not None:
        try:
            if hasattr(prediction, "trend_regime") and prediction.trend_regime:
                result["trend_regime"] = str(prediction.trend_regime)
                sources["trend_regime"] = "prediction.trend_regime"
        except Exception:
            pass
    
    if result["trend_regime"] is None and policy_state:
        result["trend_regime"] = (
            policy_state.get("brain_trend_regime") or
            policy_state.get("trend_regime") or
            None
        )
        if result["trend_regime"]:
            result["trend_regime"] = str(result["trend_regime"])
            if policy_state.get("brain_trend_regime"):
                sources["trend_regime"] = "policy_state.brain_trend_regime"
            else:
                sources["trend_regime"] = "policy_state.trend_regime"
    
    if result["trend_regime"] is None and feature_context:
        trend_val = feature_context.get("trend_regime")
        if trend_val:
            result["trend_regime"] = str(trend_val)
            sources["trend_regime"] = "feature_context.trend_regime"
    
    if result["trend_regime"] is None:
        sources["trend_regime"] = "fallback:None"
    
    # Extract vol_regime
    if prediction is not None:
        try:
            if hasattr(prediction, "vol_regime") and prediction.vol_regime:
                result["vol_regime"] = str(prediction.vol_regime)
                sources["vol_regime"] = "prediction.vol_regime"
        except Exception:
            pass
    
    if result["vol_regime"] is None and policy_state:
        result["vol_regime"] = (
            policy_state.get("brain_vol_regime") or
            policy_state.get("vol_regime") or
            None
        )
        if result["vol_regime"]:
            result["vol_regime"] = str(result["vol_regime"])
            if policy_state.get("brain_vol_regime"):
                sources["vol_regime"] = "policy_state.brain_vol_regime"
            else:
                sources["vol_regime"] = "policy_state.vol_regime"
    
    if result["vol_regime"] is None and feature_context:
        vol_val = feature_context.get("vol_regime")
        if vol_val:
            result["vol_regime"] = str(vol_val)
            sources["vol_regime"] = "feature_context.vol_regime"
    
    if result["vol_regime"] is None:
        sources["vol_regime"] = "fallback:None"
    
    # Extract atr_bps
    if current_atr_bps is not None:
        try:
            result["atr_bps"] = float(current_atr_bps)
            sources["atr_bps"] = "current_atr_bps"
        except (ValueError, TypeError):
            pass
    
    if result["atr_bps"] is None and feature_context:
        atr_val = feature_context.get("atr_bps")
        if atr_val is not None:
            try:
                result["atr_bps"] = float(atr_val)
                sources["atr_bps"] = "feature_context.atr_bps"
            except (ValueError, TypeError):
                pass
    
    if result["atr_bps"] is None and entry_bundle is not None:
        try:
            if hasattr(entry_bundle, "atr_bps") and entry_bundle.atr_bps is not None:
                result["atr_bps"] = float(entry_bundle.atr_bps)
                sources["atr_bps"] = "entry_bundle.atr_bps"
        except (ValueError, TypeError, AttributeError):
            pass
    
    if result["atr_bps"] is None:
        sources["atr_bps"] = "fallback:None"
    
    # Extract spread_bps
    if spread_pct is not None:
        try:
            result["spread_bps"] = float(spread_pct) * 10000.0
            sources["spread_bps"] = f"spread_pct*10000 ({spread_pct})"
        except (ValueError, TypeError):
            pass
    
    if result["spread_bps"] is None and feature_context:
        spread_bps_val = feature_context.get("spread_bps")
        if spread_bps_val is not None:
            try:
                result["spread_bps"] = float(spread_bps_val)
                sources["spread_bps"] = "feature_context.spread_bps"
            except (ValueError, TypeError):
                pass
        
        # Also check for spread_pct in feature_context
        if result["spread_bps"] is None:
            spread_pct_val = feature_context.get("spread_pct")
            if spread_pct_val is not None:
                try:
                    result["spread_bps"] = float(spread_pct_val) * 10000.0
                    sources["spread_bps"] = f"feature_context.spread_pct*10000 ({spread_pct_val})"
                except (ValueError, TypeError):
                    pass
    
    if result["spread_bps"] is None and entry_bundle is not None:
        try:
            if hasattr(entry_bundle, "spread_bps") and entry_bundle.spread_bps is not None:
                result["spread_bps"] = float(entry_bundle.spread_bps)
                sources["spread_bps"] = "entry_bundle.spread_bps"
        except (ValueError, TypeError, AttributeError):
            pass
    
    if result["spread_bps"] is None:
        sources["spread_bps"] = "fallback:None"
    
    # Store sources in result for debug logging
    result["_sources"] = sources
    
    return result


__all__ = ["get_runtime_regime_inputs"]

