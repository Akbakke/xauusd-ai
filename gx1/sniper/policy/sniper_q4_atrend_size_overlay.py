"""
Q4 × A_TREND size overlay (runtime policy-only, no model/entry/exit changes).

This overlay reduces trade size for Q4 × A_TREND trades to mitigate negative EV.

Configuration:
- enabled: bool (default False)
- multiplier: float (default 0.30)
- action: str (default "disable") - "scale" or "disable"
  - "scale": Apply multiplier (min unit = 1 if base > 0) - DEPRECATED: ineffective with base_units=1
  - "disable": Set units = 0 (NO-TRADE) for Q4 × A_TREND - DEFAULT: blocks high tail-risk trades

Policy Decision (2025-12-21):
- Q4 × A_TREND trades have high tail risk (P90 loss: -96.45 bps vs Q4 total: -9.69 bps)
- Scale mode ineffective: base_units=1 prevents size reduction (min unit = 1)
- Default action: "disable" (NO-TRADE) for Q4 × A_TREND
- See docs/policies/Q4_A_TREND_POLICY.md for full rationale

Gating:
- quarter == "Q4"
- classify_regime(row) == "A_TREND"
- If action == "disable": units_out = 0 (NO-TRADE), no exceptions

Output:
- units = 0 (NO-TRADE) for "disable" mode (default)
- meta dict with overlay_name, quarter, regime_class, session, multiplier, action, reason, etc.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Tuple, Union

import logging
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime
from gx1.sniper.policy.sniper_regime_size_overlay import compute_quarter


logger = logging.getLogger(__name__)

# Implementation fingerprint
OVERLAY_IMPL_ID = "q4_atrend_overlay_v1_20251220_1640"


def apply_q4_atrend_overlay(
    base_units: int,
    entry_time: Union[str, float, int, pd.Timestamp, datetime],
    trend_regime: Any,
    vol_regime: Any,
    atr_bps: Any,
    spread_bps: Any,
    session: str,
    cfg: Mapping[str, Any] | None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Apply Q4 × A_TREND size overlay.

    Returns:
        units_out (int), overlay_meta (dict)
    """
    cfg = cfg or {}

    # Initialize all core variables defensively
    session_s = session or "UNKNOWN"
    try:
        quarter = compute_quarter(entry_time)
    except Exception:
        quarter = "UNKNOWN"

    enabled = bool(cfg.get("enabled", False))
    default_mult = float(cfg.get("multiplier", 0.30))
    mult = default_mult
    action = str(cfg.get("action", "disable")).lower()  # "scale" or "disable" (default: "disable")

    regime_class: Any = None
    regime_reason: str = "missing_fields"
    reason: str = "init"
    overlay_applied: bool = False
    units_out: int = int(base_units)
    effective_scale: bool = True  # True if size actually changed

    # Build row-like dict for reuse of classify_regime()
    row = {
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
        "session": session_s,
    }
    try:
        regime_class, regime_reason = classify_regime(row)
    except Exception as exc:
        regime_class = None
        regime_reason = f"classify_error:{type(exc).__name__}"

    # Gating without leaving variables undefined
    if not enabled:
        reason = "disabled"
    elif quarter != "Q4":
        reason = "not_q4"
    elif regime_class != "A_TREND":
        reason = f"not_a_trend:{regime_class}"
    elif abs(mult - 1.0) < 1e-9:
        reason = "multiplier_1.0"
    else:
        overlay_applied = True
        
        # Preserve sign for short trades
        sign = 1 if base_units >= 0 else -1
        try:
            units_abs = abs(int(base_units))
        except Exception:
            units_abs = abs(int(float(base_units)))
        
        if action == "disable":
            # NO-TRADE mode: Q4 × A_TREND → units = 0 (hard policy)
            # Policy decision: Q4 A_TREND trades have high tail risk (P90 loss: -96.45 bps)
            # Scale mode ineffective with base_units=1 (min unit = 1 prevents size reduction)
            units_out = 0
            effective_scale = False
            reason = "Q4_A_TREND_high_tail_risk"
            logger.info(
                "[SNIPER_Q4_ATREND] Disabled trade (Q4 × A_TREND policy): base_units=%s, mult=%.3f, "
                "trend_regime=%s, vol_regime=%s, atr_bps=%s, spread_bps=%s",
                base_units,
                mult,
                trend_regime,
                vol_regime,
                atr_bps,
                spread_bps,
            )
        else:
            # SCALE mode: apply multiplier with min unit = 1 (DEPRECATED: ineffective)
            reason = "Q4_A_TREND_gate"
            units_out_abs = int(round(units_abs * mult))
            if units_out_abs == 0 and units_abs > 0:
                logger.warning(
                    "[SNIPER_Q4_ATREND] Size overlay produced 0 units (base=%s, mult=%.3f); "
                    "keeping minimum of 1 unit in same direction.",
                    base_units,
                    mult,
                )
                units_out_abs = 1
                effective_scale = False  # Size didn't actually change
            elif units_out_abs == units_abs:
                effective_scale = False  # Size didn't change (rounding kept same)
            else:
                effective_scale = True  # Size actually changed
            units_out = sign * units_out_abs

    overlay_meta: Dict[str, Any] = {
        "overlay_name": "Q4_A_TREND_SIZE",
        "overlay_applied": overlay_applied,
        "quarter": quarter,
        "regime_class": regime_class,
        "regime_reason": regime_reason,
        "session": session_s,
        "multiplier": mult,
        "action": action,
        "size_before_units": base_units,
        "size_after_units": units_out,
        "effective_scale": effective_scale,
        "reason": reason,
        "trend_regime": str(trend_regime) if trend_regime else None,
        "vol_regime": str(vol_regime) if vol_regime else None,
        "atr_bps": float(atr_bps) if atr_bps is not None else None,
        "spread_bps": float(spread_bps) if spread_bps is not None else None,
        "impl_id": OVERLAY_IMPL_ID,
        "impl_file": __file__,
    }

    return units_out, overlay_meta


__all__ = ["apply_q4_atrend_overlay"]

